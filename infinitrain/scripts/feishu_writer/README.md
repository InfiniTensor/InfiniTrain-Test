# 飞书 Writer 使用指南

本文说明 `infinitrain/scripts/` 下飞书写入相关脚本的配置格式、参数和典型运行方式。

## Quick Start

核心流程是两步，对应两个脚本：
1. 远端表格创建：在更换机器、新增测试组或新增模型时，先运行 `provision_feishu_sheets.py`，在远端查找或创建对应的表格云文档，然后在本地创建或更新 token JSON 文件。
2. 性能数据写入：运行 `write_to_feishu_sheet.py` 脚本，传入 provisioning 输出或更新后的 token JSON，实现数据写入。

下面介绍运行方式。在此之前，默认你已经拿到可用的 `APP_ID` 和 `APP_SECRET`，并已按本文末尾“附录：Lark-cli 配置”安装飞书 CLI 并完成本地授权。

### 路径约定

`test_config.json` 仍然在实际的 `InfiniTrain` 仓库里。因此运行第一个脚本时，需要事先配置：

```bash
export INFINITRAIN_ROOT=/path/to/your/InfiniTrain
export RUN_OUTPUT_DIR="$INFINITRAIN_ROOT/scripts/<run-output-dir>"
cd /path/to/your/InfiniTrain-Test
```

`RUN_OUTPUT_DIR` 指向一次测试的输出目录。该目录通常按
`scripts/<YYYYMMDD>/<branch>_<short-commit>` 组织，但这不是强制命名格式。
脚本只要求其中存在 `logs/` 目录；同级的 `profile_logs/` 目录可选。

未设置 `INFINITRAIN_ROOT` 时，`provision_feishu_sheets.py` 会直接停止，避免误把 `InfiniTrain-Test` 当作数据根目录。

### 1. 手动运行脚本

首先，最少保证 `infinitrain/scripts/feishu_writer/token.json` 中有：

```json
{
    "APP_ID": "...",
    "APP_SECRET": "..."
}
```

#### 1.1 **新增机器**

```bash
export INFINITRAIN_ROOT=/path/to/your/InfiniTrain
cd /path/to/your/InfiniTrain-Test

python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine "202605 node1"
```

`--new-machine` 的值就是新建的机器目录名。脚本会读取
`$INFINITRAIN_ROOT/scripts/test_config.json`，为其中全部 test group tag 和模型创建
`tag × model` 表格。运行输出默认写到 `infinitrain/scripts/feishu_writer/new_token.json`。

如果中途因为权限或网络失败，补完权限后用 `--grant-existing` 恢复：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine "202605 node1" \
  --grant-existing
```

#### 1.2 **新增 tag**

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-tags new_tag
```

如果 `token.json` 里没有机器目录 token，则需要显式传入：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-tags new_tag \
  --machine-folder-token <existing_machine_folder_token>
```

运行成功后，新创建的 spreadsheet token 会自动写入 `token.json` 的 `TAG_SPREADSHEET_CONFIGS` 字段，后续不需要再手工粘贴。这个模式将默认原地更新 `infinitrain/scripts/feishu_writer/token.json`（不会生成 `new_token.json`；如果想输出到其他文件，可以额外传 `--output-token-file <path>`）。

#### 1.3 **新增模型**

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-model NEW_MODEL
```

模型必须已经在 `test_config.json` 中声明。脚本会遍历输入 token 配置里的
`FEISHU_PROVISION.tag_folders`，在每个远端 tag 目录中查找名为 `NEW_MODEL`
的 spreadsheet：存在则复用并记录其 token，不存在则从统一模板复制并记录。

#### 1.4 **写入脚本**

`write_to_feishu_sheet.py` 不读取 `INFINITRAIN_ROOT`，需要通过 `--log-dir`
手动传入实际测试的输出目录。尽管参数名为 `--log-dir`，这里传入的是同时包含
`logs/` 和 `profile_logs/` 的父目录：

```bash
python3 infinitrain/scripts/feishu_writer/write_to_feishu_sheet.py infinitrain/scripts/feishu_writer/token.json --log-dir "$RUN_OUTPUT_DIR"
```

此外，也可以通过 `--skip-profile` 手动跳过 profile logs 的上传。

### 2. 利用 AI Agent

直接让 AI Agent 使用 `.agents/skills/infinitrain-feishu-writer/SKILL.md`。告诉它场景是“新增机器”“已有机器新增测试组”还是“已有机器新增模型”，以及对应的机器目录名、tag 或 model。`APP_SECRET` 通过本地 `token.json` 或隐藏输入提供，不要让 Agent 在回复中打印密钥。

## 文件说明

- `infinitrain/scripts/feishu_writer/provision_feishu_sheets.py`：创建或复用飞书云盘目录、复制模型模板表、授权文档应用，并生成可直接给 writer 使用的 token JSON。
- `infinitrain/scripts/feishu_writer/write_to_feishu_sheet.py`：读取本地 logs/profile reports，把 benchmark 结果写入已配置好的飞书表格。
- `infinitrain/scripts/feishu_writer/token.json`：测试仓库中的本机真实运行配置，包含密钥，已被 Git 忽略。
- `infinitrain/scripts/feishu_writer/new_token.json`：新增机器时默认生成到测试仓库的新配置文件，包含密钥，也已被 Git 忽略。
- `infinitrain/scripts/feishu_writer/token.example.json`：安全的配置示例，不包含真实密钥。

## token.json 格式

`write_to_feishu_sheet.py` 最少需要这些字段（与原先的使用方式一致）：

```json
{
    "APP_ID": "your_feishu_app_id",
    "APP_SECRET": "your_feishu_app_secret",
    "TAG_SPREADSHEET_CONFIGS": [
        {
            "tag": "basic",
            "MODEL_SPREADSHEET_TOKEN": {
                "GPT2": "spreadsheet_token_for_gpt2_basic",
                "LLAMA3": "spreadsheet_token_for_llama3_basic"
            }
        }
    ]
}
```

`provision_feishu_sheets.py` 会额外使用并维护 `FEISHU_PROVISION`，故完整的 token.json 可能包括：
```json
{
    "APP_ID": "cli_or_writer_app_id",
    "APP_SECRET": "do_not_commit_real_secret",
    "FEISHU_PROVISION": {
        "root_folder_token": "MW8Nfsd3ulIRmpdYSo1c7EBsn1G",
        "machine_folder_token": "",
        "machine_folder_name": "202605 machine-name",
        "permission": {
            "member_type": "appid",
            "member_id": "",
            "perm": "edit"
        },
        "tag_folders": {}
    },
    "TAG_SPREADSHEET_CONFIGS": []
}
```

注意：

- 不要提交或打印 `APP_SECRET`。
- `FEISHU_PROVISION` 是 provisioning 的状态记录，writer 不依赖它。
- `TAG_SPREADSHEET_CONFIGS` 是 writer 真正消费的映射。provisioning 成功后，不需要再人工复制 spreadsheet token。
- 当 `permission.member_type` 是 `appid` 且 `permission.member_id` 为空时，provisioning 会自动使用顶层 `APP_ID` 授权。
- 模型名会归一化，`GPT2`、`gpt2`、`gpt-2` 都会映射为 `GPT2`。
- 所有模型统一从模板 `X5mJskjzSh2mo3tzuERccAYxnib` 复制，模板 token 不再从 token JSON 或命令行配置。

## provision_feishu_sheets.py 参数

常用参数：

- `--test-config PATH`：测试配置 JSON，默认 `$INFINITRAIN_ROOT/scripts/test_config.json`。
- `--token-file PATH`：输入的 seed token 配置，默认 `InfiniTrain-Test/infinitrain/scripts/feishu_writer/token.json`。
- `--output-token-file PATH`：输出的 token 配置路径，相对路径从 `InfiniTrain-Test` 解析。
- `--new-machine "yyyymm name"`：新增机器模式，参数值同时作为新机器目录名。忽略输入 token 中已有的机器目录、tag 目录和 spreadsheet token，并为 `test_config.json` 中全部 `tag × model` 组合生成新配置。未指定 `--output-token-file` 时默认写入 `infinitrain/scripts/feishu_writer/new_token.json`。
- `--new-tags tag1,tag2`：已有机器新增 tag 模式。tag 必须存在于 `test_config.json`，并为每个 tag 创建该配置声明的全部模型表格。
- `--new-model MODEL`：已有机器新增模型模式。模型必须存在于 `test_config.json`；脚本根据输入 token 配置中的 `FEISHU_PROVISION.tag_folders` 对每个远端 tag 目录查找或创建同名 spreadsheet，并把远端 token 写回各 tag 的 `MODEL_SPREADSHEET_TOKEN`。
- `--new-machine`、`--new-tags` 和 `--new-model` 必须且只能选择一种；全部为空或同时传入多个都会报错。
- `--machine-folder-token TOKEN`：复用已有机器目录，不再新建机器目录。
- `--dry-run`：只打印计划，不创建飞书资源，也不写 token JSON。
- 不传 `--dry-run` 时默认执行远端写操作并更新 token JSON，不需要额外确认参数。
- Provisioning 完成后，脚本会根据实际输出 token 文件打印 `write_to_feishu_sheet.py` 命令；从 `InfiniTrain-Test` 运行时仍需补充 `--log-dir "$RUN_OUTPUT_DIR"`。

权限参数：

- `--skip-permission`：不授权文档应用。
- `--grant-existing`：对 token JSON 中已存在的 spreadsheet 也重新授权/检查。
- `--permission-member-type TYPE`：默认取配置里的值，否则为 `appid`。
- `--permission-member-id ID`：默认取配置里的值；当 member type 是 `appid` 时默认取 `APP_ID`。
- `--permission-perm view|edit|full_access`：默认取配置里的值，否则为 `edit`。

校验和调试参数：

- `--skip-template-check`：跳过“新表中是否存在 `模板` sheet”的检查。
- `--as user|bot`：飞书 CLI 身份，默认 `user`。
- `--verbose`：打印实际执行的 `lark-cli` 命令。

## 场景一：新增机器

目标：在固定根目录下创建新机器目录，读取
`$INFINITRAIN_ROOT/scripts/test_config.json` 中的 test group tag 和模型，为全部
`tag × model` 组合创建目录及模型表格副本，给文档应用授权，并输出可直接使用的
`infinitrain/scripts/feishu_writer/new_token.json`。

tag 来自 `test_groups[].tag`。模型来自 `variables` 中同时存在的
`<MODEL>_INPUT_BIN` 和 `<MODEL>_LLMC_FILEPATH` 变量；当前配置会得到
`GPT2` 和 `LLAMA3`。

预览：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine "202605 new-machine" \
  --dry-run
```

执行：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine "202605 new-machine"
```

输出结果：

- 飞书云盘目录：固定根目录 -> `202605 new-machine`
- 机器目录下：`test_config.json` 中每个 test group tag 一个子目录
- 每个 tag 子目录下：配置中每个模型一个从模板复制出来的 spreadsheet
- 本地配置：`infinitrain/scripts/feishu_writer/new_token.json`

后续直接写入，不需要人工粘贴 token：

```bash
python3 infinitrain/scripts/feishu_writer/write_to_feishu_sheet.py infinitrain/scripts/feishu_writer/new_token.json --log-dir "$RUN_OUTPUT_DIR"
```

如果执行过程中因为 scope 不足或网络中断停在半截，下一次重跑建议加 `--grant-existing`，这样已复制出来但尚未授权/检查的 spreadsheet 也会被补处理：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine "202605 new-machine" \
  --grant-existing
```

如果希望新机器直接使用 `infinitrain/scripts/feishu_writer/token.json` 作为正式配置，可以指定输出路径：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine "202605 new-machine" \
  --output-token-file infinitrain/scripts/feishu_writer/token.json
```

## 场景二：已有机器新增 tag

目标：在当前机器目录下新增一个 tag 子目录，复制每个模型的模板表，授权，并把新 token 信息追加到现有 `infinitrain/scripts/feishu_writer/token.json`。

先确保 `$INFINITRAIN_ROOT/scripts/test_config.json` 已经包含新的
`test_group.tag`，然后执行：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py --new-tags new_tag --dry-run
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py --new-tags new_tag
```

`new_tag` 必须已存在于 `test_config.json` 的 `test_groups[].tag` 中。脚本不会读取
`$RUN_OUTPUT_DIR/logs`；它会为该 tag 创建配置中声明的全部模型表格。

如果当前 `infinitrain/scripts/feishu_writer/token.json` 里还没有 `FEISHU_PROVISION.machine_folder_token`，第一次需要显式传入已有机器目录 token：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-tags new_tag \
  --machine-folder-token <existing_machine_folder_token>
```

这个流程会原地更新 `infinitrain/scripts/feishu_writer/token.json`。后续直接运行：

```bash
python3 infinitrain/scripts/feishu_writer/write_to_feishu_sheet.py infinitrain/scripts/feishu_writer/token.json --log-dir "$RUN_OUTPUT_DIR"
```

## 场景三：已有机器新增 model

如果未来新增模型类型，先在 `test_config.json:variables` 中加入成对的
`<MODEL>_INPUT_BIN` 和 `<MODEL>_LLMC_FILEPATH`。然后预览：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-model NEWMODEL \
  --dry-run
```

执行：

```bash
python3 infinitrain/scripts/feishu_writer/provision_feishu_sheets.py \
  --new-model NEWMODEL
```

脚本会：

1. 确认 `NEWMODEL` 已由 `test_config.json` 声明。
2. 遍历输入 token 配置的 `FEISHU_PROVISION.tag_folders`。
3. 在每个远端 tag 文件夹中查找标题为 `NEWMODEL` 的 spreadsheet。
4. 已存在时跳过复制并记录远端 token；不存在时从统一模板复制、授权并校验。
5. 将每个 tag 下 `NEWMODEL` 的 token 写回输入 token JSON。

## write_to_feishu_sheet.py 用法

```bash
python3 infinitrain/scripts/feishu_writer/write_to_feishu_sheet.py <token-config-json> --log-dir "$RUN_OUTPUT_DIR"
```

示例：

```bash
python3 infinitrain/scripts/feishu_writer/write_to_feishu_sheet.py infinitrain/scripts/feishu_writer/token.json --log-dir "$RUN_OUTPUT_DIR"
python3 infinitrain/scripts/feishu_writer/write_to_feishu_sheet.py infinitrain/scripts/feishu_writer/new_token.json --log-dir "$RUN_OUTPUT_DIR"
python3 infinitrain/scripts/feishu_writer/write_to_feishu_sheet.py infinitrain/scripts/feishu_writer/token.json --log-dir "$RUN_OUTPUT_DIR" --skip-profile
```

使用上述 `--log-dir` 时，它会从这些目录发现本地数据：

- `$RUN_OUTPUT_DIR/logs/<tag>/<model>_<testcase>.log`
- `$RUN_OUTPUT_DIR/profile_logs/<tag>/<model>_<testcase>_profile_<model>.report.rank0`

对每个已配置的 tag/model spreadsheet，它会：

1. 发现本地 testcase。
2. 查询远端 spreadsheet 里已有的 sheet。
3. 如果 testcase sheet 不存在，从远端 `模板` sheet 复制一个。
4. 解析 benchmark/profile 数据并 prepend 到对应 sheet。
5. 设置样式并合并元信息列。

如果 `$RUN_OUTPUT_DIR/profile_logs` 目录或单个 profile report 不存在，脚本仍会写入前 7 列 meta 信息，profile 对应列会保持为空字符串。也可以通过 `--skip-profile` 手动跳过 profile 解析，行为与 profile 数据不存在一致。

## 安全检查

- 可以先在对应模式命令中使用 `--dry-run` 预览；不带 `--dry-run` 时会直接执行并更新 token JSON。
- 不要提交 `infinitrain/scripts/feishu_writer/token.json` 或 `infinitrain/scripts/feishu_writer/new_token*.json`。
- 新增机器使用 `--new-machine "<name>"`。
- 已有机器新增 tag 使用 `--new-tags <tag>`。
- 已有机器新增 model 使用 `--new-model <model>`。
- 三种模式必须且只能选择一种。
- Provisioning 的 tag/model 完全来自 `test_config.json`，不根据本地日志增删。
- `write_to_feishu_sheet.py` 仍然依赖 `APP_ID` 和 `APP_SECRET` 直接鉴权，因此输出 token JSON 必须保留这两个字段。

## 附录：Lark-cli 配置

安装流程参考：https://www.feishu.cn/feishu-cli。

首次使用前，需要把 `lark-cli` 配置到对应飞书应用：

```bash
export INFINITRAIN_ROOT=/path/to/your/InfiniTrain
cd /path/to/your/InfiniTrain-Test

lark-cli config init \
  --app-id <APP_ID> \
  --app-secret-stdin \
  --brand feishu \
  --force-init
```

然后从 stdin 粘贴 `APP_SECRET`。不要把 `APP_SECRET` 写进命令行参数。

实测新增机器 provisioning 需要 user 身份至少授权这些 scope：

- `space:document:retrieve`：读取根目录和已有目录，避免重复创建。
- `space:folder:create`：创建机器目录和 tag 目录。
- `docs:document:copy`：从统一模板复制 spreadsheet。
- `docs:permission.member:create`：把 `APP_ID` 加为新 spreadsheet 的协作者。
- `sheets:spreadsheet.meta:read`：校验新 spreadsheet 中存在 `模板` sheet。

推荐一次性授权：

```bash
lark-cli auth login --scope "space:document:retrieve space:folder:create docs:document:copy docs:permission.member:create sheets:spreadsheet.meta:read"
```

如果一次性授权不稳定，就按报错提示缺哪个补哪个：

```bash
lark-cli auth login --scope "space:document:retrieve"
lark-cli auth login --scope "space:folder:create"
lark-cli auth login --scope "docs:document:copy"
lark-cli auth login --scope "docs:permission.member:create"
lark-cli auth login --scope "sheets:spreadsheet.meta:read"
```

所有模型都使用模型名作为复制出来的新表标题，并统一使用固定模板
`X5mJskjzSh2mo3tzuERccAYxnib`。
