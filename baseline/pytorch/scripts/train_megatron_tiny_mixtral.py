#!/usr/bin/env python3
"""Run tiny Mixtral with Megatron-LM MoE for InfiniTrain loss validation.

Install Megatron Core, or pass --megatron_path to use a local Megatron-LM checkout:
    pip install megatron-core

Export an LLMC checkpoint aligned with InfiniTrain tiny_mixtral:
    python3 train_megatron_tiny_mixtral.py \
        --num_iterations 0 \
        --write_model_path /data/shared/InfiniTrain-dev/data/llmc/tiny_mixtral/tiny_mixtral_megatron_export.bin

Train from the exported LLMC checkpoint:
    python3 train_megatron_tiny_mixtral.py \
        --weights_path /data/shared/InfiniTrain-dev/data/llmc/tiny_mixtral/tiny_mixtral_megatron_export.bin \
        --input_bin /data/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin \
        --num_iterations 10 \
        --log_interval 1 \
        --print_timing
"""

import argparse
import math
import os
import pathlib
import struct
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MEGATRON_PATH = os.environ.get("MEGATRON_PATH", "")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(head_dim, end, theta, use_scaled_rope, device):
    if use_scaled_rope:
        raise RuntimeError("scaled RoPE is not supported by the tiny Mixtral runner")
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(end, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, freqs_cis):
    cos, sin = freqs_cis
    cos = cos[: x.size(1)].view(1, x.size(1), 1, x.size(-1) // 2)
    sin = sin[: x.size(1)].view(1, x.size(1), 1, x.size(-1) // 2)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    batch, seq_len, n_kv_head, head_dim = x.shape
    return x[:, :, :, None, :].expand(batch, seq_len, n_kv_head, n_rep, head_dim).reshape(
        batch, seq_len, n_kv_head * n_rep, head_dim
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_head = args.n_head
        self.n_kv_head = args.n_kv_head
        self.n_embd = args.n_embd
        self.head_dim = args.n_embd // args.n_head
        self.n_rep = args.n_head // args.n_kv_head
        self.c_attn = nn.Linear(args.n_embd, (args.n_head + 2 * args.n_kv_head) * self.head_dim, bias=False)
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=False)

    def forward(self, x, freqs_cis, mask):
        batch, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        q_rows = self.n_head * self.head_dim
        kv_rows = self.n_kv_head * self.head_dim
        q, k, v = qkv.split([q_rows, kv_rows, kv_rows], dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim)
        k = k.view(batch, seq_len, self.n_kv_head, self.head_dim)
        v = v.view(batch, seq_len, self.n_kv_head, self.head_dim)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        att = F.softmax(scores, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
        return self.c_proj(y)


def header_float(header, index):
    return struct.unpack("f", struct.pack("i", header[index]))[0]


def float_to_int32(value):
    return struct.unpack("i", struct.pack("f", float(value)))[0]


def write_fp32(tensor, file):
    file.write(tensor.detach().cpu().contiguous().float().numpy().tobytes())


def read_header(path):
    with open(path, "rb") as file:
        raw = file.read(256 * 4)
    if len(raw) != 256 * 4:
        raise RuntimeError(f"failed to read tiny Mixtral header from {path}")
    header = [struct.unpack_from("i", raw, i * 4)[0] for i in range(256)]
    if header[0] != 20260513 or header[1] != 2:
        raise RuntimeError(f"{path} is not a tiny Mixtral v2 checkpoint")
    return header


def apply_header(args):
    header = read_header(args.weights_path)
    args.max_position_embeddings = header[2]
    args.vocab_size = header[3]
    args.n_layer = header[4]
    args.n_head = header[5]
    args.n_kv_head = header[6]
    args.n_embd = header[7]
    args.num_experts = header[8]
    args.ffn_expansion_ratio = header_float(header, 9)
    args.ffn_dim_multiplier = header_float(header, 10)
    args.multiple_of = header[11]
    args.norm_eps = header_float(header, 12)
    args.rope_theta = header_float(header, 13)
    args.use_scaled_rope = bool(header[14])
    args.router_topk = header[15]
    args.moe_ffn_hidden_size = header[16]


def read_fp32(file, shape, device):
    numel = math.prod(shape)
    raw = file.read(numel * 4)
    if len(raw) != numel * 4:
        raise RuntimeError(f"unexpected EOF while reading tensor with shape {shape}")
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().view(*shape)
    return tensor.to(device)


def read_token_bin(path, sequence_length):
    with open(path, "rb") as file:
        header = file.read(1024)
        if len(header) != 1024:
            raise RuntimeError(f"failed to read dataset header from {path}")
        magic, _version, num_tokens = struct.unpack_from("iii", header, 0)
        if magic == 20240520:
            dtype = torch.uint16
            item_size = 2
        elif magic == 20240801:
            dtype = torch.int32
            item_size = 4
        else:
            raise RuntimeError(f"unsupported dataset magic {magic} in {path}")
        num_sequences = num_tokens // sequence_length
        raw = file.read(num_sequences * sequence_length * item_size)
        if len(raw) != num_sequences * sequence_length * item_size:
            raise RuntimeError(f"failed to read dataset tokens from {path}")
    return torch.frombuffer(bytearray(raw), dtype=dtype).to(torch.int64)


def get_batch(tokens, batch_idx, batch_size, sequence_length, device):
    num_samples = tokens.numel() // sequence_length - 1
    start = (batch_idx * batch_size) % num_samples
    xs = []
    ys = []
    for offset in range(batch_size):
        sample_idx = (start + offset) % num_samples
        token_offset = sample_idx * sequence_length
        xs.append(tokens[token_offset : token_offset + sequence_length])
        ys.append(tokens[token_offset + 1 : token_offset + sequence_length + 1])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def init_megatron(args):
    if args.megatron_path:
        if args.megatron_path not in sys.path:
            sys.path.insert(0, args.megatron_path)
    else:
        try:
            import megatron.core  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Megatron Core is not importable. Install `megatron-core` or pass "
                "--megatron_path /path/to/Megatron-LM."
            ) from exc

    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    torch.cuda.set_device(args.device_id)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)

    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(args.seed)


def cleanup_megatron():
    import torch.distributed as dist
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


class MegatronMoE(nn.Module):
    def __init__(self, args, layer_number):
        super().__init__()
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=args.n_layer,
            hidden_size=args.n_embd,
            num_attention_heads=args.n_head,
            ffn_hidden_size=args.moe_ffn_hidden_size,
            num_moe_experts=args.num_experts,
            moe_ffn_hidden_size=args.moe_ffn_hidden_size,
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            bias_activation_fusion=False,
            moe_router_topk=args.router_topk,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_z_loss_coeff=0.0,
            moe_input_jitter_eps=None,
            moe_router_score_function="softmax",
            moe_router_pre_softmax=False,
            moe_router_topk_scaling_factor=None,
            moe_token_dispatcher_type="allgather",
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        spec = get_gpt_layer_local_spec(num_experts=args.num_experts, moe_grouped_gemm=False)
        self.moe = MoELayer(config, spec.submodules.mlp.submodules, layer_number=layer_number)

    def forward(self, x):
        y, bias = self.moe(x.transpose(0, 1).contiguous())
        if bias is not None:
            y = y + bias
        return y.transpose(0, 1).contiguous()


class Block(nn.Module):
    def __init__(self, args, layer_number):
        super().__init__()
        self.ln_1 = RMSNorm(args.n_embd, args.norm_eps)
        self.attn = CausalSelfAttention(args)
        self.ln_2 = RMSNorm(args.n_embd, args.norm_eps)
        self.mlp = MegatronMoE(args, layer_number)

    def forward(self, x, freqs_cis, mask):
        x = x + self.attn(self.ln_1(x), freqs_cis, mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyMegatronMixtral(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = args
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(args.vocab_size, args.n_embd),
                h=nn.ModuleList([Block(args, layer + 1) for layer in range(args.n_layer)]),
                ln_f=RMSNorm(args.n_embd, args.norm_eps),
            )
        )
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.freqs_cis = None

    def forward(self, idx):
        _, seq_len = idx.shape
        if self.freqs_cis is None:
            self.freqs_cis = precompute_freqs_cis(
                self.config.n_embd // self.config.n_head,
                self.config.max_position_embeddings * 2,
                self.config.rope_theta,
                self.config.use_scaled_rope,
                idx.device,
            )
        freqs_cis = (self.freqs_cis[0][:seq_len], self.freqs_cis[1][:seq_len])
        mask = torch.triu(torch.ones((seq_len, seq_len), device=idx.device, dtype=torch.bool), diagonal=1).view(
            1, 1, seq_len, seq_len
        )
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, freqs_cis, mask)
        return self.lm_head(self.transformer.ln_f(x))


def load_weights(model, args, device):
    with open(args.weights_path, "rb") as file:
        header = file.read(256 * 4)
        if len(header) != 256 * 4:
            raise RuntimeError(f"failed to read header from {args.weights_path}")

        with torch.no_grad():
            model.transformer.wte.weight.copy_(read_fp32(file, (args.vocab_size, args.n_embd), device))
            head_dim = args.n_embd // args.n_head
            q_rows = args.n_head * head_dim
            kv_rows = args.n_kv_head * head_dim
            for block in model.transformer.h:
                block.ln_1.weight.copy_(read_fp32(file, (args.n_embd,), device))
                q = read_fp32(file, (q_rows, args.n_embd), device)
                k = read_fp32(file, (kv_rows, args.n_embd), device)
                v = read_fp32(file, (kv_rows, args.n_embd), device)
                block.attn.c_attn.weight.copy_(torch.cat([q, k, v], dim=0))
                block.attn.c_proj.weight.copy_(read_fp32(file, (args.n_embd, args.n_embd), device))
                block.ln_2.weight.copy_(read_fp32(file, (args.n_embd,), device))
                block.mlp.moe.router.weight.copy_(read_fp32(file, (args.num_experts, args.n_embd), device))
                for expert_idx in range(args.num_experts):
                    gate = read_fp32(file, (args.moe_ffn_hidden_size, args.n_embd), device)
                    up = read_fp32(file, (args.moe_ffn_hidden_size, args.n_embd), device)
                    down = read_fp32(file, (args.n_embd, args.moe_ffn_hidden_size), device)
                    expert = block.mlp.moe.experts.local_experts[expert_idx]
                    expert.linear_fc1.weight.copy_(torch.cat([gate, up], dim=0))
                    expert.linear_fc2.weight.copy_(down)
            model.transformer.ln_f.weight.copy_(read_fp32(file, (args.n_embd,), device))
            model.lm_head.weight.copy_(read_fp32(file, (args.vocab_size, args.n_embd), device))

        if file.read(1):
            raise RuntimeError(f"unexpected trailing bytes in {args.weights_path}")



def write_model(model, filename, args):
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20260513
    header[1] = 2
    header[2] = args.max_position_embeddings
    header[3] = args.vocab_size
    header[4] = args.n_layer
    header[5] = args.n_head
    header[6] = args.n_kv_head
    header[7] = args.n_embd
    header[8] = args.num_experts
    header[9] = float_to_int32(args.ffn_expansion_ratio)
    header[10] = float_to_int32(args.ffn_dim_multiplier)
    header[11] = args.multiple_of
    header[12] = float_to_int32(args.norm_eps)
    header[13] = float_to_int32(args.rope_theta)
    header[14] = int(args.use_scaled_rope)
    header[15] = args.router_topk
    header[16] = args.moe_ffn_hidden_size

    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        write_fp32(model.transformer.wte.weight, file)
        head_dim = args.n_embd // args.n_head
        q_rows = args.n_head * head_dim
        kv_rows = args.n_kv_head * head_dim
        for block in model.transformer.h:
            write_fp32(block.ln_1.weight, file)
            q, k, v = block.attn.c_attn.weight.split([q_rows, kv_rows, kv_rows], dim=0)
            write_fp32(q, file)
            write_fp32(k, file)
            write_fp32(v, file)
            write_fp32(block.attn.c_proj.weight, file)
            write_fp32(block.ln_2.weight, file)
            write_fp32(block.mlp.moe.router.weight, file)
            for expert_idx in range(args.num_experts):
                expert = block.mlp.moe.experts.local_experts[expert_idx]
                gate, up = expert.linear_fc1.weight.split([args.moe_ffn_hidden_size, args.moe_ffn_hidden_size], dim=0)
                write_fp32(gate, file)
                write_fp32(up, file)
                write_fp32(expert.linear_fc2.weight, file)
        write_fp32(model.transformer.ln_f.weight, file)
        write_fp32(model.lm_head.weight, file)
    print(f"wrote {filename}", flush=True)


@dataclass
class TinyMixtralConfig:
    max_position_embeddings: int = 32768
    vocab_size: int = 128256
    num_experts: int = 8
    router_topk: int = 2
    moe_ffn_hidden_size: int = 1792
    n_layer: int = 32
    n_head: int = 4
    n_kv_head: int = 1
    n_embd: int = 512
    ffn_expansion_ratio: float = 3.5
    ffn_dim_multiplier: float = 0.0
    multiple_of: int = 0
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    use_scaled_rope: bool = False

    def __post_init__(self):
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        assert self.n_embd % self.n_head == 0
        assert 0 < self.router_topk <= self.num_experts
        assert self.moe_ffn_hidden_size > 0


@dataclass
class TinyArgs(TinyMixtralConfig):
    weights_path: str = ""
    write_model_path: str = ""
    megatron_path: str = DEFAULT_MEGATRON_PATH
    micro_batch_size: int = 4
    global_batch_size: int = 4
    sequence_length: int = 64
    num_iterations: int = 10
    learning_rate: float = 1e-4
    dtype: str = "float32"
    input_bin: str = ""
    seed: int = 42
    device_id: int = 0
    master_port: int = 29571
    log_interval: int = 1
    print_timing: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Megatron MoE tiny Mixtral loss alignment runner")
    defaults = TinyArgs()

    # File system input / output.
    parser.add_argument("--weights_path", default=defaults.weights_path, help="LLMC model file to load before training")
    parser.add_argument("--write_model_path", default=defaults.write_model_path, help="path to write the initialized LLMC model")
    parser.add_argument("--input_bin", type=str, default=defaults.input_bin, help="token dataset .bin used for training")

    # Token layout for each optimization step.
    parser.add_argument("--micro_batch_size", type=int, default=defaults.micro_batch_size, help="micro batch size in samples")
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=defaults.global_batch_size,
        help="global batch size across gradient accumulation and data parallelism",
    )
    parser.add_argument("--sequence_length", type=int, default=defaults.sequence_length, help="tokens per sample")

    # Workload and optimization.
    parser.add_argument("--num_iterations", type=int, default=defaults.num_iterations, help="number of training steps")
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate, help="Adam learning rate")
    parser.add_argument("--dtype", type=str, default=defaults.dtype, choices=("float32", "bfloat16"), help="training precision")

    # Megatron runtime environment.
    parser.add_argument("--megatron_path", default=defaults.megatron_path, help="optional path to a Megatron-LM checkout; empty uses installed megatron-core")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="random seed")
    parser.add_argument("--device_id", type=int, default=defaults.device_id, help="CUDA device id")
    parser.add_argument("--master_port", type=int, default=defaults.master_port, help="single-rank NCCL master port")

    # Logging.
    parser.add_argument("--log_interval", type=int, default=defaults.log_interval, help="print train loss every N steps")
    parser.add_argument("--print_timing", action="store_true", default=defaults.print_timing, help="print elapsed time and token throughput")

    args = parser.parse_args(namespace=defaults)
    if args.weights_path:
        apply_header(args)
    return args


def configure_float32_matmul():
    # InfiniTrain CUDA GEMM uses CUBLAS_COMPUTE_32F for fp32.
    # Disable PyTorch/Megatron TF32 so fp32 loss validation compares the same math mode.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def validate_args(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Megatron MoE validation requires CUDA")
    assert args.n_embd % args.n_head == 0
    assert args.n_head % args.n_kv_head == 0
    assert 0 < args.router_topk <= args.num_experts
    assert args.moe_ffn_hidden_size > 0
    assert args.micro_batch_size > 0
    assert args.global_batch_size > 0
    if args.global_batch_size % args.micro_batch_size != 0:
        raise RuntimeError("global_batch_size must be divisible by micro_batch_size")
    if args.num_iterations > 0 and not args.input_bin:
        raise RuntimeError("Megatron tiny Mixtral training requires --input_bin")
    if args.sequence_length > args.max_position_embeddings:
        raise RuntimeError("sequence_length must not exceed TinyMixtralConfig.max_position_embeddings")
    assert not args.use_scaled_rope


def main():
    args = parse_args()
    validate_args(args)
    if args.weights_path:
        pathlib.Path(args.weights_path).resolve(strict=True)
    torch.manual_seed(args.seed)
    configure_float32_matmul()
    init_megatron(args)
    device = torch.device(f"cuda:{args.device_id}")
    try:
        model = TinyMegatronMixtral(args).to(device)
        if args.weights_path:
            load_weights(model, args, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        tokens = read_token_bin(args.input_bin, args.sequence_length) if args.input_bin else None

        step_durations_ms = []
        grad_accum_steps = args.global_batch_size // args.micro_batch_size
        tokens_per_step = args.global_batch_size * args.sequence_length
        autocast_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        use_autocast = args.dtype == "bfloat16"
        for step in range(args.num_iterations):
            torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for micro_step in range(grad_accum_steps):
                batch_idx = step * grad_accum_steps + micro_step
                x, y = get_batch(tokens, batch_idx, args.micro_batch_size, args.sequence_length, device)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss_value += loss.detach().float().item() / grad_accum_steps
                (loss / grad_accum_steps).backward()
            optimizer.step()
            torch.cuda.empty_cache()

            torch.cuda.synchronize()
            duration_ms = (time.perf_counter() - step_start) * 1e3
            step_durations_ms.append(duration_ms)
            if args.log_interval > 0 and ((step + 1) % args.log_interval == 0 or step + 1 == args.num_iterations):
                print(
                    f"step {step + 1:4d}/{args.num_iterations} | train loss {loss_value:.6f} "
                    f"| norm -1.0000 | lr {args.learning_rate:.2e} | "
                    f"({duration_ms:.2f} ms | {tokens_per_step / (duration_ms / 1e3):.0f} tok/s)",
                    flush=True,
                )
        if step_durations_ms:
            averaged = step_durations_ms[1:] if len(step_durations_ms) > 1 else step_durations_ms
            print(f"final {len(averaged)} iters avg: {sum(averaged) / len(averaged):.3f}ms", flush=True)
        if args.write_model_path:
            write_model(model, args.write_model_path, args)
    finally:
        cleanup_megatron()


if __name__ == "__main__":
    main()
