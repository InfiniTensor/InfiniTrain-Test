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
- `$INFINITRAIN_ROOT/scripts/test_config.json` for all provisioning tag/model pairs.
- `$INFINITRAIN_ROOT/scripts/logs/` when preparing result writes and discovering testcases.
- `$INFINITRAIN_ROOT/scripts/profile_logs/` when preparing result writes.

Also use the Lark skills:

- `lark-shared` for auth, identity, scope, and high-risk write handling.
- `lark-drive` for folders, file copy, and permission member creation.
- `lark-sheets` for spreadsheet info checks.

## Defaults

- Root folder token: `MW8Nfsd3ulIRmpdYSo1c7EBsn1G`
- Shared model template token: `X5mJskjzSh2mo3tzuERccAYxnib`
- Permission default: grant `edit` to `APP_ID` as `member_type=appid`.
- Token input default: `InfiniTrain-Test/scripts/feishu_writer/token.json`.
- New-machine token output default: `InfiniTrain-Test/scripts/feishu_writer/new_token.json`.
- Provision tags: `test_config.json` `test_groups[].tag`.
- Provision models: names having both `<MODEL>_INPUT_BIN` and `<MODEL>_LLMC_FILEPATH` under `test_config.json` `variables`.
- Provision mode: exactly one of `--new-machine NAME`, `--new-tags TAG1,TAG2`, or `--new-model MODEL`.

## Workflow

1. Confirm `INFINITRAIN_ROOT` is set to the real `InfiniTrain` repo, not `InfiniTrain-Test`. The scripts intentionally fail fast when it is missing.
2. Summarize `scripts/feishu_writer/token.json` structurally only: keys, tags, model names, machine folder presence. Do not display secrets.
3. Ensure the user identity has the required Feishu scopes before provisioning. The tested minimum scopes are `space:document:retrieve`, `space:folder:create`, `docs:document:copy`, `docs:permission.member:create`, and `sheets:spreadsheet.meta:read`.
4. For a new machine, dry-run a fresh output token file:
   ```bash
   python3 scripts/feishu_writer/provision_feishu_sheets.py --new-machine "202605 machine-name" --dry-run
   ```
   This provisions every `tag × model` pair declared by `test_config.json`. Run the same command without `--dry-run` to execute; it writes `scripts/feishu_writer/new_token.json` in `InfiniTrain-Test` by default.
5. For an existing machine or a new tag, dry-run an in-place update:
   ```bash
   python3 scripts/feishu_writer/provision_feishu_sheets.py --new-tags new_tag --dry-run
   ```
   Every requested tag must exist in `test_config.json`; each receives all models declared there. The provisioner never uses log contents to select tags or models.
6. To add a model to an existing machine, first confirm it is declared by `test_config.json`, then dry-run:
   ```bash
   python3 scripts/feishu_writer/provision_feishu_sheets.py --new-model NEWMODEL --dry-run
   ```
   This mode iterates `FEISHU_PROVISION.tag_folders`, searches each remote folder for a spreadsheet titled `NEWMODEL`, records existing remote tokens, and copies the shared template only where missing.
7. If the plan is correct and the user asked to execute, remove `--dry-run`; no confirmation flag is required.
8. If provisioning was interrupted after creating/copying some resources, rerun with `--grant-existing` so existing spreadsheets also receive app permission and template checks:
   ```bash
   python3 scripts/feishu_writer/provision_feishu_sheets.py --new-machine "202605 machine-name" --grant-existing
   ```
9. For an existing machine folder, pass either `--machine-folder-token <folder_token>` or rely on `FEISHU_PROVISION.machine_folder_token` in `scripts/feishu_writer/token.json`.
10. After provisioning, write results with the generated config:
   ```bash
   python3 scripts/feishu_writer/write_to_feishu_sheet.py scripts/feishu_writer/token.json
   python3 scripts/feishu_writer/write_to_feishu_sheet.py scripts/feishu_writer/new_token.json
   ```

## Safety

- `scripts/feishu_writer/token.json` is local secret-bearing state in `InfiniTrain-Test` and must stay ignored by Git.
- `scripts/feishu_writer/new_token*.json` is also secret-bearing output in `InfiniTrain-Test` and must stay ignored by Git.
- Use `--dry-run` before writes unless the user explicitly asks for immediate execution.
- Pass exactly one mode: `--new-machine NAME`, `--new-tags TAG1,TAG2`, or `--new-model MODEL`.
- Without `--dry-run`, provisioning writes immediately and updates the selected token JSON.
- If permission creation says the collaborator already exists, treat it as successful.
- If a model is added later, first declare it in `test_config.json` with matching `<MODEL>_INPUT_BIN` and `<MODEL>_LLMC_FILEPATH` variables, then use `--new-model MODEL`.
