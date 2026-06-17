---
name: infinitrain-feishu-writer
description: Use when provisioning InfiniTrain Feishu benchmark spreadsheets from the InfiniTrain-Test writer repo, creating machine/tag folders, copying model template sheets, granting document-app access, updating scripts/feishu_writer/token.json in InfiniTrain-Test, or preparing write_to_feishu_sheet.py to run with INFINITRAIN_ROOT pointing at the real InfiniTrain repo.
---

# InfiniTrain Feishu Writer

Use this skill for InfiniTrain benchmark result provisioning. The writer code and token JSON live in the `InfiniTrain-Test` repo. Benchmark logs, profile logs, test config, and git metadata live in the real `InfiniTrain` repo selected by `INFINITRAIN_ROOT`.

Prefer running or patching `scripts/feishu_writer/provision_feishu_sheets.py` and `scripts/feishu_writer/write_to_feishu_sheet.py` instead of reconstructing raw `lark-cli` calls by hand.

## Required Context

Before running either script, use the user's local checkout paths:

```bash
export INFINITRAIN_ROOT=/path/to/your/InfiniTrain
cd /path/to/your/InfiniTrain-Test
```

Read only what is needed:

- `scripts/feishu_writer/README.md` in `InfiniTrain-Test` for user-facing commands and token JSON schema.
- `scripts/feishu_writer/token.json` in `InfiniTrain-Test` for `APP_ID`, existing tag tokens, and optional `FEISHU_PROVISION`; never print `APP_SECRET`.
- `$INFINITRAIN_ROOT/scripts/logs/` to discover already-run benchmark tags for new-machine provisioning.
- `$INFINITRAIN_ROOT/scripts/profile_logs/` when preparing result writes.
- `$INFINITRAIN_ROOT/scripts/test_config.json` only as secondary context for benchmark definitions.

Also use the Lark skills:

- `lark-shared` for auth, identity, scope, and high-risk write handling.
- `lark-drive` for folders, file copy, and permission member creation.
- `lark-sheets` for spreadsheet info checks.

## Defaults

- Root folder token: `MW8Nfsd3ulIRmpdYSo1c7EBsn1G`
- GPT2 template token: `X5mJskjzSh2mo3tzuERccAYxnib`
- LLAMA3 template token: `NtT3syaRThyGXDtQyzdcpiyfnXd`
- New machine folder name default: `yyyymm <hostname>` unless the user provides `--machine-folder-name`.
- Permission default: grant `edit` to `APP_ID` as `member_type=appid`.
- Token input default: `InfiniTrain-Test/scripts/feishu_writer/token.json`.
- New-machine token output default: `InfiniTrain-Test/scripts/feishu_writer/new_token.json`.
- Data defaults: `$INFINITRAIN_ROOT/scripts/logs`, `$INFINITRAIN_ROOT/scripts/profile_logs`, and `$INFINITRAIN_ROOT/scripts/test_config.json`.

## Workflow

1. Confirm `INFINITRAIN_ROOT` is set to the real `InfiniTrain` repo, not `InfiniTrain-Test`. The scripts intentionally fail fast when it is missing.
2. Summarize `scripts/feishu_writer/token.json` structurally only: keys, tags, model names, machine folder presence. Do not display secrets.
3. Ensure the user identity has the required Feishu scopes before provisioning. The tested minimum scopes are `space:document:retrieve`, `space:folder:create`, `docs:document:copy`, `docs:permission.member:create`, and `sheets:spreadsheet.meta:read`.
4. For a new machine, dry-run a fresh output token file:
   ```bash
   python3 scripts/feishu_writer/provision_feishu_sheets.py --new-machine --machine-folder-name "202605 machine-name" --dry-run
   ```
   The tag list is discovered from `$INFINITRAIN_ROOT/scripts/logs/<tag>/<model>_*.log` unless `--tags` is passed. Execute with `--yes`; this writes `scripts/feishu_writer/new_token.json` in `InfiniTrain-Test` by default.
5. For an existing machine or a new tag, dry-run an in-place update:
   ```bash
   python3 scripts/feishu_writer/provision_feishu_sheets.py --tags new_tag --dry-run
   ```
   The script checks `$INFINITRAIN_ROOT/scripts/logs/new_tag` and warns if no model logs exist, but still provisions the tag. Narrow scope with `--tags tag1,tag2`, `--models GPT2,LLAMA3`, or `--machine-folder-name "202605 machine-name"` when needed.
6. If the plan is correct and the user asked to execute, run with `--yes`:
   ```bash
   python3 scripts/feishu_writer/provision_feishu_sheets.py --yes
   ```
7. If provisioning was interrupted after creating/copying some resources, rerun with `--grant-existing` so existing spreadsheets also receive app permission and template checks:
   ```bash
   python3 scripts/feishu_writer/provision_feishu_sheets.py --new-machine --yes --grant-existing
   ```
8. For an existing machine folder, pass either `--machine-folder-token <folder_token>` or rely on `FEISHU_PROVISION.machine_folder_token` in `scripts/feishu_writer/token.json`.
9. After provisioning, write results with the generated config:
   ```bash
   python3 scripts/feishu_writer/write_to_feishu_sheet.py scripts/feishu_writer/token.json
   python3 scripts/feishu_writer/write_to_feishu_sheet.py scripts/feishu_writer/new_token.json
   ```

## Safety

- `scripts/feishu_writer/token.json` is local secret-bearing state in `InfiniTrain-Test` and must stay ignored by Git.
- `scripts/feishu_writer/new_token*.json` is also secret-bearing output in `InfiniTrain-Test` and must stay ignored by Git.
- Use `--dry-run` before writes unless the user explicitly asks for immediate execution.
- The provisioning script requires `--yes` for mutations and passes `--yes` to high-risk Lark operations only then.
- If permission creation says the collaborator already exists, treat it as successful.
- If a model template is added later, use `--template-token MODEL=TOKEN` or add it under `FEISHU_PROVISION.model_templates` in `scripts/feishu_writer/token.json`.
