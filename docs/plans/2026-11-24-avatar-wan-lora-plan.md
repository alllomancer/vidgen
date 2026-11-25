# Avatar Wan LoRA and one-person photo sets — Implementation Plan

Spec: docs/specs/2026-11-24-avatar-wan-lora.md
Workspace: current folder (~/Documents/private/myproject; not a git repository, so no commits or SHAs)
Jira: N/A

## Progress

- [x] Task 1: One-person check on photo sets and clips — reviewed; stuck-state blockers fixed
- [x] Task 2: Wan LoRA state on new avatars — reviewed; stuck-state blockers fixed
- [x] Task 3: Wan dataset and train job from the photo-set job — reviewed; stuck-state blockers fixed
- [x] Task 4: Trainer: avatar Wan builds — reviewed; version-collision blocker fixed (D6)
- [x] Task 5: Automatic avatar LoRA on video jobs — reviewed
- [x] Task 6: On-demand Wan LoRA builds, CLI and dashboard — reviewed, browser-checked
- [x] Task 7: Docs, deploy and live check — deployed; live: people check, on-demand build, auto-attach

## Tasks

### Task 1: One-person check on photo sets and clips

**What**: `models.count_people(path)` (Qwen2.5-VL via `_chat`, first integer of the answer, else -1). `_make_photo_set`
and `_dataset_photos` drop photos whose count isn't 1 (after the content filter); fewer than 16 left raises
`FilteredOut` with the reason ("only N of M photos show exactly one person; use reference images of just this
person"); results record `people_dropped`. `_avatar_media` drops clips whose middle still isn't exactly one person.
**Files**: worker/models.py, worker/worker.py, tests/test_worker.py
**Depends on**: none
**Verify**: `make test`: answer parsing ("1", "There are 2 people.", "none" -> -1); a set with 12 two-person photos
fails with the reason and writes no dataset/media; 6 dropped of 24 leaves 18 used everywhere; a reused dataset is
filtered; a two-person clip is dropped.

### Task 2: Wan LoRA state on new avatars

**What**: `_avatar_wan_lora_name`; `create_avatar` takes `wan_lora` (bool, default true, 400 otherwise); with it, the
photoset job's params carry `wan_lora: true`, `wan_lora_name`, `wan_version` and `avatar.json` gets
`wan_lora = {state: queued, lora, version, trigger, job_ids: [job]}` (else `{state: skipped}`); `_write_avatar` keeps
`wan_lora`; `list_avatars` returns it. Media-only jobs carry `wan_lora: false`.
**Files**: lambda/api/handler.py, tests/test_api.py
**Depends on**: none
**Verify**: `make test`: create with/without `wan_lora` (params and state), long names stay <= 32 chars and stable,
image edits keep `wan_lora`, invalid flag 400.

### Task 3: Wan dataset and train job from the photo-set job

**What**: After the media is stored, when `wan_lora` is true: write `datasets/<client>/<wan lora>/v<N>/` (raw photos
`p_NN.png` and clips `c_N.mp4`, manifest `kind: character`, `source: avatar`, captions keyed `p{i:04d}.png` /
`s{i:04d}_00.mp4`), conditionally put a `train` job (id = uuid5 of the photoset job id + ":wan") with
`{model: t2v, dataset, lora, version, trigger, avatar, avatar_created_at, steps: 1000, speed: standard,
kind: character, files, fingerprint, ...}`, link it in `wan_lora.job_ids` (state training), send it. Staleness via
`_current_wan`/`_update_wan`; a failure marks `wan_lora` failed without failing the job; redelivery re-links/re-sends.
**Files**: worker/worker.py, tests/test_worker.py
**Depends on**: Task 1, Task 2
**Verify**: `make test`: keys, captions and manifest; one train job even when redelivered; skipped when `wan_lora`
false or media-only; Wan failure leaves Krea and media unchanged.

### Task 4: Trainer: avatar Wan builds

**What**: In the Wan path, params with `lora`/`version` upload to `loras/<client>/<lora>/v<version>/` (dataset folder
unchanged), skip auto previews, and set the avatar's `wan_lora` ready (`ready = {lora, version}`) while the job is in
`wan_lora.job_ids`; failure/cancel set it failed (`_update_avatar_wan_lora`, mirroring the Krea helpers). IAM if the
trainer can't yet write `avatars/*/avatar.json`.
**Files**: trainer/trainer.py, tests/test_trainer_cli.py, terraform/training.tf
**Depends on**: Task 3
**Verify**: `make test`: upload keys and lora.json name/version for an avatar build; ready/failed/cancelled state with
staleness; non-avatar Wan jobs unchanged (name = dataset, previews queued).

### Task 5: Automatic avatar LoRA on video jobs

**What**: `create_job` for `video`: when the star avatar has `wan_lora.ready` and `avatar_lora` isn't false, put that
LoRA first in `params.loras` at weight 1.0 (resolved and pinned like user LoRAs), skipping it when the user already
named it; the 2-Wan-LoRA limit counts it (400 naming the limit). Summary shows it. CLI `--no-avatar-lora`; dashboard
sends `avatar_lora`.
**Files**: lambda/api/handler.py, cli/vidgen.py, cli/ui.html, tests/test_api.py, tests/test_trainer_cli.py
**Depends on**: Task 2
**Verify**: `make test`: attached first at 1.0; not with `avatar_lora: false`; not duplicated; 1 user LoRA ok, 2 -> 400;
stories unchanged; CLI flag sends `avatar_lora: false`.

### Task 6: On-demand Wan LoRA builds, CLI and dashboard

**What**: `POST /avatars/{name}/wan-lora` (404/409 busy/409 "make media first") queues a `wan_only` photoset job that
copies the avatar's current media photos and clips (people check applied) into a new Wan dataset version and queues
training. `vidgen avatar wan-train --name N`; `avatar list` and the dashboard show the Wan LoRA state with a "Train Wan
LoRA" button; Terraform route.
**Files**: lambda/api/handler.py, worker/worker.py, terraform/api.tf, cli/vidgen.py, cli/ui.py, cli/ui.html, tests
**Depends on**: Task 3
**Verify**: `make test`: route codes; the job copies media into the dataset and queues one train job without new photos
or clips; CLI/ui proxy calls; browser preview of the states and button.

### Task 7: Docs, deploy and live check

**What**: RUNBOOK, README, ADR, spec live-check section. `make plan`/`make apply` after the user's go-ahead (also ships
the before-start cancel fix), restart the render worker between jobs. Live: inbal's photo set with the people check;
`vidgen avatar wan-train --name mirabel`; the 50 s mirabel take with her automatic Wan LoRA vs the spike run.
**Files**: RUNBOOK.md, README.md, docs/specs/2026-11-24-avatar-wan-lora.md, docs/adr/avatar-media.md
**Depends on**: Task 1, Task 2, Task 3, Task 4, Task 5, Task 6
**Verify**: `make test`; inbal's multi-person photos dropped; mirabel's Wan LoRA ready and attached to the take
automatically; likeness at least as good as the spike.
