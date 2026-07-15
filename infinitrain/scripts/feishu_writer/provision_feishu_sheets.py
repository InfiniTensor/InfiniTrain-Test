#!/usr/bin/env python3
"""Provision Feishu spreadsheets for InfiniTrain benchmark tags and models.

This script creates/reuses:
  root machine folder -> tag folder -> model spreadsheet copies

It then writes the resulting spreadsheet tokens into token.json so the existing
write_to_feishu_sheet.py script can run unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_ROOT_FOLDER_TOKEN = "MW8Nfsd3ulIRmpdYSo1c7EBsn1G"
DEFAULT_ROOT_FOLDER_URL = (
    "https://gxtctab8no8.feishu.cn/drive/folder/"
    f"{DEFAULT_ROOT_FOLDER_TOKEN}"
)
DEFAULT_TEMPLATE_TOKEN = "X5mJskjzSh2mo3tzuERccAYxnib"
SCRIPT_DIR = Path(__file__).resolve().parent
WRITER_REPO_ROOT = SCRIPT_DIR.parents[2]


def resolve_path_from_cwd(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return path.resolve()


def resolve_path_from_writer_repo(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (WRITER_REPO_ROOT / path).resolve()

PROVISION_KEY = "FEISHU_PROVISION"
TAG_CONFIGS_KEY = "TAG_SPREADSHEET_CONFIGS"
MODEL_TOKENS_KEY = "MODEL_SPREADSHEET_TOKEN"


class CLIError(RuntimeError):
    def __init__(self, cmd: list[str], returncode: int, stdout: str, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(self._format())

    def _format(self) -> str:
        return (
            f"command failed with exit code {self.returncode}: "
            f"{' '.join(self.cmd)}\n{self.stderr or self.stdout}"
        )


class LarkCLI:
    def __init__(self, identity: str, dry_run: bool, verbose: bool):
        self.identity = identity
        self.dry_run = dry_run
        self.verbose = verbose
        self._yes_support_cache: dict[tuple[str, ...], bool] = {}

    def supports_yes(self, args: list[str]) -> bool:
        command_path = tuple(arg for arg in args if not arg.startswith("--"))
        for index, arg in enumerate(args):
            if arg.startswith("--"):
                command_path = tuple(args[:index])
                break
        if command_path in self._yes_support_cache:
            return self._yes_support_cache[command_path]

        proc = subprocess.run(
            ["lark-cli", *command_path, "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        help_text = f"{proc.stdout}\n{proc.stderr}"
        supported = "--yes" in help_text
        self._yes_support_cache[command_path] = supported
        return supported

    def run_json(
        self,
        args: list[str],
        *,
        write: bool = False,
        high_risk: bool = False,
        allow_existing_error: bool = False,
    ) -> Any | None:
        cmd = ["lark-cli", *args]
        if self.identity and "--as" not in cmd:
            cmd.extend(["--as", self.identity])
        if (
            high_risk
            and not self.dry_run
            and "--yes" not in cmd
            and self.supports_yes(args)
        ):
            cmd.append("--yes")

        if write and self.dry_run:
            print(f"[dry-run] {' '.join(cmd)}")
            return None

        if self.verbose:
            print(f"[cmd] {' '.join(cmd)}")

        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            if allow_existing_error and _looks_like_existing_error(proc.stderr):
                print("[ok] permission already exists")
                return {"already_exists": True}
            raise CLIError(cmd, proc.returncode, proc.stdout, proc.stderr)

        stdout = proc.stdout.strip()
        if not stdout:
            return {}

        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"lark-cli did not return JSON for {' '.join(cmd)}") from exc


def _looks_like_existing_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "already",
            "exist",
            "duplicate",
            "duplicated",
            "repeated",
            "重复",
            "已存在",
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=4)
        tmp.write("\n")
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def normalize_model_key(value: str) -> str:
    compact = value.strip().replace("-", "").replace("_", "")
    return compact.upper()


def extract_folder_token(value: str | None) -> str | None:
    if not value:
        return None
    marker = "/drive/folder/"
    if marker in value:
        return value.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]
    return value.strip()


def discover_tags(test_config: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for group in test_config.get("test_groups", []):
        tag = group.get("tag")
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def discover_models(test_config: dict[str, Any]) -> list[str]:
    variables = test_config.get("variables", {})
    if not isinstance(variables, dict):
        return []

    input_models = {
        key[: -len("_INPUT_BIN")]
        for key in variables
        if key.endswith("_INPUT_BIN")
    }
    filepath_models = {
        key[: -len("_LLMC_FILEPATH")]
        for key in variables
        if key.endswith("_LLMC_FILEPATH")
    }
    return sorted(normalize_model_key(model) for model in input_models & filepath_models)


def template_for_model(model: str) -> dict[str, str]:
    return {
        "template_token": DEFAULT_TEMPLATE_TOKEN,
        "doc_type": "sheet",
        "title": model,
    }


def collect_file_items(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("files", "items"):
                maybe_items = node.get(key)
                if isinstance(maybe_items, list):
                    for item in maybe_items:
                        if (
                            isinstance(item, dict)
                            and "name" in item
                            and "token" in item
                            and "type" in item
                        ):
                            found.append(item)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)

    deduped: dict[str, dict[str, Any]] = {}
    for item in found:
        deduped.setdefault(item["token"], item)
    return list(deduped.values())


def extract_created_file(value: Any) -> dict[str, Any]:
    candidates: list[Any] = []
    if isinstance(value, dict):
        candidates.extend([value.get("file"), value.get("folder")])
        data = value.get("data")
        if isinstance(data, dict):
            candidates.extend([data.get("file"), data.get("folder")])
    for item in candidates:
        if isinstance(item, dict) and item.get("token"):
            return item
    folder_token = find_key(value, "folder_token")
    if folder_token:
        return {"token": folder_token}
    token = find_key(value, "token")
    if token:
        return {"token": token}
    raise RuntimeError(f"Unable to find created file token in response: {value}")


def find_key(value: Any, key: str) -> Any | None:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            result = find_key(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find_key(child, key)
            if result is not None:
                return result
    return None


def find_title(value: Any, title: str) -> bool:
    if isinstance(value, dict):
        if value.get("title") == title:
            return True
        return any(find_title(child, title) for child in value.values())
    if isinstance(value, list):
        return any(find_title(child, title) for child in value)
    return False


class Provisioner:
    def __init__(self, cli: LarkCLI):
        self.cli = cli

    def list_folder(self, folder_token: str) -> list[dict[str, Any]]:
        response = self.cli.run_json(
            [
                "drive",
                "files",
                "list",
                "--params",
                json.dumps({"folder_token": folder_token, "page_size": 200}),
                "--page-all",
                "--page-limit",
                "0",
            ]
        )
        return collect_file_items(response)

    def find_child(
        self,
        folder_token: str,
        name: str,
        allowed_types: set[str],
    ) -> dict[str, Any] | None:
        matches = [
            item
            for item in self.list_folder(folder_token)
            if item.get("name") == name
            and str(item.get("type", "")).lower() in allowed_types
        ]
        if len(matches) > 1:
            print(
                f"[warn] found {len(matches)} existing entries named {name!r}; "
                f"using token={matches[0].get('token')}"
            )
        return matches[0] if matches else None

    def ensure_folder(self, parent_token: str, name: str) -> str:
        if parent_token.startswith("DRYRUN_"):
            existing = None
        else:
            existing = self.find_child(parent_token, name, {"folder"})
            if existing:
                print(f"[exists] folder {name}: {existing['token']}")
                return existing["token"]

        print(f"[create] folder {name}")
        response = self.cli.run_json(
            [
                "drive",
                "+create-folder",
                "--folder-token",
                parent_token,
                "--name",
                name,
            ],
            write=True,
        )
        if response is None:
            return f"DRYRUN_FOLDER_{safe_token_fragment(name)}"
        created = extract_created_file(response)
        return created["token"]

    def ensure_spreadsheet(
        self,
        tag_folder_token: str,
        model: str,
        template: dict[str, str],
        title: str,
    ) -> tuple[str, bool]:
        if tag_folder_token.startswith("DRYRUN_"):
            existing = None
        else:
            existing = self.find_child(tag_folder_token, title, {"sheet"})
            if existing:
                print(f"[exists] {model} spreadsheet {title}: {existing['token']}")
                return existing["token"], False

        print(f"[copy] {model} template -> {title}")
        response = self.cli.run_json(
            [
                "drive",
                "files",
                "copy",
                "--params",
                json.dumps({"file_token": template["template_token"]}),
                "--data",
                json.dumps(
                    {
                        "folder_token": tag_folder_token,
                        "name": title,
                        "type": template.get("doc_type", "sheet"),
                    }
                ),
            ],
            write=True,
            high_risk=True,
        )
        if response is None:
            return f"DRYRUN_SHEET_{safe_token_fragment(model + '_' + title)}", True
        created = extract_created_file(response)
        return created["token"], True

    def grant_permission(
        self,
        spreadsheet_token: str,
        member_type: str,
        member_id: str,
        perm: str,
        collaborator_type: str | None,
    ) -> None:
        data: dict[str, Any] = {
            "member_type": member_type,
            "member_id": member_id,
            "perm": perm,
        }
        if collaborator_type:
            data["type"] = collaborator_type

        print(f"[permission] grant {perm} to {member_type}:{member_id}")
        self.cli.run_json(
            [
                "drive",
                "permission.members",
                "create",
                "--params",
                json.dumps(
                    {
                        "token": spreadsheet_token,
                        "type": "sheet",
                        "need_notification": False,
                    }
                ),
                "--data",
                json.dumps(data),
            ],
            write=True,
            high_risk=True,
            allow_existing_error=True,
        )

    def check_template_sheet(self, spreadsheet_token: str) -> None:
        response = self.cli.run_json(
            [
                "sheets",
                "+info",
                "--spreadsheet-token",
                spreadsheet_token,
            ]
        )
        if not find_title(response, "模板"):
            raise RuntimeError(
                f"spreadsheet {spreadsheet_token} does not contain a sheet titled 模板"
            )
        print(f"[check] spreadsheet {spreadsheet_token} contains 模板")


def safe_token_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)[:48]


def tag_config_map(token_config: dict[str, Any]) -> dict[str, dict[str, str]]:
    configs = token_config.get(TAG_CONFIGS_KEY, [])
    result: dict[str, dict[str, str]] = {}
    for item in configs:
        tag = item.get("tag")
        model_tokens = item.get(MODEL_TOKENS_KEY, {})
        if tag and isinstance(model_tokens, dict):
            result[tag] = {
                normalize_model_key(model): token
                for model, token in model_tokens.items()
            }
    legacy = token_config.get(MODEL_TOKENS_KEY)
    if legacy and "basic" not in result:
        result["basic"] = {
            normalize_model_key(model): token
            for model, token in legacy.items()
        }
    return result


def write_tag_configs(
    token_config: dict[str, Any],
    mapping: dict[str, dict[str, str]],
    ordered_tags: list[str],
) -> None:
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag in ordered_tags:
        if tag in mapping:
            configs.append({"tag": tag, MODEL_TOKENS_KEY: mapping[tag]})
            seen.add(tag)
    for tag, model_tokens in mapping.items():
        if tag not in seen:
            configs.append({"tag": tag, MODEL_TOKENS_KEY: model_tokens})
    token_config[TAG_CONFIGS_KEY] = configs
    token_config.pop(MODEL_TOKENS_KEY, None)


def writer_command(token_file_path: Path) -> str:
    writer_path = SCRIPT_DIR / "write_to_feishu_sheet.py"
    cwd = Path.cwd()
    writer_arg = os.path.relpath(writer_path, cwd)
    token_arg = os.path.relpath(token_file_path, cwd)
    return (
        f"python {shlex.quote(writer_arg)} "
        f"{shlex.quote(token_arg)}"
    )


def main() -> int:
    writer_scripts_dir = SCRIPT_DIR
    default_token_file = writer_scripts_dir / "token.json"
    default_new_token_file = writer_scripts_dir / "new_token.json"
    parser = argparse.ArgumentParser(
        description="Create/reuse Feishu folders and spreadsheet copies for InfiniTrain tags."
    )
    parser.add_argument(
        "--test-config",
        required=True,
        help="Required: path to test_config.json. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--token-file",
        default=str(default_token_file),
        help="Path to seed token JSON in the writer repo. Relative paths are resolved from InfiniTrain-Test.",
    )
    parser.add_argument(
        "--output-token-file",
        help=(
            "Where to write the provisioned token JSON in the writer repo. "
            "Defaults to --token-file, or "
            "infinitrain/scripts/feishu_writer/new_token.json "
            "when --new-machine is set."
        ),
    )
    parser.add_argument(
        "--new-machine",
        default="",
        metavar="NAME",
        help=(
            "Create a fresh machine folder named NAME and provision every "
            "tag/model pair declared by test_config.json."
        ),
    )
    parser.add_argument(
        "--new-tags",
        default="",
        metavar="TAG1,TAG2",
        help=(
            "Provision the comma-separated test_config.json tags in the "
            "existing machine folder."
        ),
    )
    parser.add_argument(
        "--new-model",
        default="",
        metavar="MODEL",
        help=(
            "Provision MODEL in every tag folder recorded by the input token "
            "file. MODEL must be declared by test_config.json."
        ),
    )
    parser.add_argument("--root-folder-token")
    parser.add_argument("--root-folder-url", default=DEFAULT_ROOT_FOLDER_URL)
    parser.add_argument("--machine-folder-token")
    parser.add_argument("--as", dest="identity", default="user", choices=["user", "bot"])
    parser.add_argument("--permission-member-type")
    parser.add_argument("--permission-member-id")
    parser.add_argument("--permission-perm", choices=["view", "edit", "full_access"])
    parser.add_argument("--permission-collaborator-type")
    parser.add_argument("--skip-permission", action="store_true")
    parser.add_argument("--grant-existing", action="store_true")
    parser.add_argument("--skip-template-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    test_config_path = resolve_path_from_cwd(args.test_config)
    token_file_path = resolve_path_from_writer_repo(args.token_file)
    new_machine_name = args.new_machine.strip()
    requested_tags = list(dict.fromkeys(parse_csv(args.new_tags) or []))
    new_model = (
        normalize_model_key(args.new_model)
        if args.new_model.strip()
        else ""
    )
    mode_count = sum(
        bool(value)
        for value in (new_machine_name, requested_tags, new_model)
    )
    if mode_count != 1:
        parser.error(
            "exactly one mode is required: "
            "--new-machine NAME, --new-tags TAG1,TAG2, or --new-model MODEL"
        )

    output_token_file_path = (
        resolve_path_from_writer_repo(args.output_token_file)
        if args.output_token_file
        else (default_new_token_file if new_machine_name else token_file_path)
    )
    test_config = load_json(test_config_path)
    token_config = load_json(token_file_path)
    original_token_config = json.dumps(
        token_config,
        ensure_ascii=False,
        sort_keys=True,
    )

    provision = token_config.setdefault(PROVISION_KEY, {})

    if new_machine_name:
        print(
            "[new-machine] ignoring existing machine folder, tag folders, "
            "and TAG_SPREADSHEET_CONFIGS from the seed token config"
        )
        token_config[TAG_CONFIGS_KEY] = []
        token_config.pop(MODEL_TOKENS_KEY, None)
        provision.pop("machine_folder_token", None)
        provision.pop("machine_folder_name", None)
        provision["tag_folders"] = {}

    provision.pop("model_templates", None)
    token_config.pop("MODEL_TEMPLATES", None)

    configured_models = discover_models(test_config)
    if not configured_models:
        raise SystemExit(
            "No models found in test_config.json. Expected matching "
            "<MODEL>_INPUT_BIN and <MODEL>_LLMC_FILEPATH variables."
        )
    if new_model and new_model not in configured_models:
        raise SystemExit(
            f"Model not found in test_config.json: {new_model}"
        )
    selected_models = [new_model] if new_model else configured_models
    templates = {
        model: template_for_model(model)
        for model in selected_models
    }

    configured_tags = discover_tags(test_config)
    tag_folders = provision.setdefault("tag_folders", {})
    if new_machine_name:
        if not configured_tags:
            raise SystemExit("No test_group tags found in test_config.json.")
        selected_tags = configured_tags
    elif requested_tags:
        if not configured_tags:
            raise SystemExit("No test_group tags found in test_config.json.")
        unknown_tags = [tag for tag in requested_tags if tag not in configured_tags]
        if unknown_tags:
            raise SystemExit(
                "Tags not found in test_config.json: " + ", ".join(unknown_tags)
            )
        selected_tags = requested_tags
    else:
        if not isinstance(tag_folders, dict) or not tag_folders:
            raise SystemExit(
                "Cannot provision a new model without "
                "FEISHU_PROVISION.tag_folders in the input token file."
            )
        invalid_tag_folders = [
            tag
            for tag, folder_token in tag_folders.items()
            if not isinstance(folder_token, str) or not folder_token
        ]
        if invalid_tag_folders:
            raise SystemExit(
                "Missing folder token for tags: " + ", ".join(invalid_tag_folders)
            )
        selected_tags = list(tag_folders)

    print(
        f"[test-config] selected tags={selected_tags}, "
        f"models={selected_models}"
    )

    root_folder_token = (
        args.root_folder_token
        or provision.get("root_folder_token")
        or extract_folder_token(args.root_folder_url)
        or DEFAULT_ROOT_FOLDER_TOKEN
    )
    machine_folder_token = args.machine_folder_token or provision.get("machine_folder_token")

    permission_cfg = provision.get("permission", {})
    permission_member_type = (
        args.permission_member_type
        or permission_cfg.get("member_type")
        or "appid"
    )
    permission_member_id = args.permission_member_id
    if not permission_member_id:
        permission_member_id = permission_cfg.get("member_id")
    if not permission_member_id and permission_member_type == "appid":
        permission_member_id = token_config.get("APP_ID")
    if not args.skip_permission and not permission_member_id:
        raise SystemExit(
            "Cannot determine permission member id. Provide --permission-member-id "
            "or APP_ID in token.json."
        )

    cli = LarkCLI(args.identity, args.dry_run, args.verbose)
    provisioner = Provisioner(cli)

    if new_machine_name:
        machine_folder_token = provisioner.ensure_folder(
            root_folder_token,
            new_machine_name,
        )
        provision["root_folder_token"] = root_folder_token
        provision["machine_folder_token"] = machine_folder_token
        provision["machine_folder_name"] = new_machine_name
    elif requested_tags:
        if not machine_folder_token:
            raise SystemExit(
                "--new-tags requires --machine-folder-token or "
                "FEISHU_PROVISION.machine_folder_token in the input token file."
            )
        print(f"[exists] machine folder token: {machine_folder_token}")

    provision["permission"] = {
        "member_type": permission_member_type,
        "member_id": permission_member_id,
        "perm": args.permission_perm or permission_cfg.get("perm") or "edit",
    }
    if args.permission_collaborator_type:
        provision["permission"]["type"] = args.permission_collaborator_type
    model_titles = {model: model for model in selected_models}

    tokens_by_tag = tag_config_map(token_config)
    changed = False

    for tag in selected_tags:
        print(f"\n=== tag: {tag} ===")
        tag_folder_token = tag_folders.get(tag)
        if tag_folder_token:
            print(f"[exists] tag folder {tag}: {tag_folder_token}")
        elif new_model:
            raise SystemExit(f"Missing tag folder token for {tag}")
        else:
            tag_folder_token = provisioner.ensure_folder(machine_folder_token, tag)
            tag_folders[tag] = tag_folder_token
            changed = True

        model_tokens = tokens_by_tag.setdefault(tag, {})
        for model in selected_models:
            if new_model:
                previous_token = model_tokens.get(model)
                spreadsheet_token, copied = provisioner.ensure_spreadsheet(
                    tag_folder_token,
                    model,
                    templates[model],
                    model_titles[model],
                )
                model_tokens[model] = spreadsheet_token
                if previous_token != spreadsheet_token:
                    changed = True

                if copied and not args.skip_permission:
                    provisioner.grant_permission(
                        spreadsheet_token,
                        permission_member_type,
                        permission_member_id,
                        args.permission_perm or permission_cfg.get("perm") or "edit",
                        args.permission_collaborator_type,
                    )
                if (
                    copied
                    and not args.skip_template_check
                    and not spreadsheet_token.startswith("DRYRUN_")
                ):
                    provisioner.check_template_sheet(spreadsheet_token)
                continue

            existing_token = model_tokens.get(model)
            if existing_token:
                print(f"[exists] token.json {tag}/{model}: {existing_token}")
                if args.grant_existing and not args.skip_permission:
                    provisioner.grant_permission(
                        existing_token,
                        permission_member_type,
                        permission_member_id,
                        args.permission_perm or permission_cfg.get("perm") or "edit",
                        args.permission_collaborator_type,
                    )
                if not args.skip_template_check and not existing_token.startswith("DRYRUN_"):
                    provisioner.check_template_sheet(existing_token)
                continue

            spreadsheet_token, copied = provisioner.ensure_spreadsheet(
                tag_folder_token,
                model,
                templates[model],
                model_titles[model],
            )
            model_tokens[model] = spreadsheet_token
            changed = True

            if not args.skip_permission and (copied or args.grant_existing):
                provisioner.grant_permission(
                    spreadsheet_token,
                    permission_member_type,
                    permission_member_id,
                    args.permission_perm or permission_cfg.get("perm") or "edit",
                    args.permission_collaborator_type,
                )
            if not args.skip_template_check and not spreadsheet_token.startswith("DRYRUN_"):
                provisioner.check_template_sheet(spreadsheet_token)

    write_tag_configs(token_config, tokens_by_tag, selected_tags)

    if args.dry_run:
        print(f"\n[dry-run] {output_token_file_path} would be written")
    elif changed or original_token_config != json.dumps(
        token_config,
        ensure_ascii=False,
        sort_keys=True,
    ) or output_token_file_path != token_file_path:
        write_json_atomic(output_token_file_path, token_config)
        print(f"\n[write] wrote {output_token_file_path}")
    else:
        print("\n[ok] token.json already up to date")

    print("\nProvisioning complete.")
    if args.dry_run:
        print("After running provisioning without --dry-run, run:")
        print(f"  {writer_command(output_token_file_path)}")
    else:
        print("Next, run:")
        print(f"  {writer_command(output_token_file_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CLIError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(exc.returncode)
