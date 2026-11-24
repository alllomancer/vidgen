# A Krea 2 LoRA for every avatar — Implementation Plan

Spec: docs/specs/2026-11-24-krea-avatar-lora.md
Workspace: current folder (~/Documents/private/myproject; not a git repository, so no commits or SHAs)
Jira: N/A

## Progress

- [x] Task 1: Download every LoRA locally — reviewed; review fixes applied
- [x] Task 2: Avatars record and queue a Krea LoRA build — reviewed; review fixes applied
- [x] Task 3: Filter-only scoring in the Krea venv — reviewed; review fixes applied
- [x] Task 4: avatar_photoset job builds the dataset and queues training — reviewed; review fixes applied
- [x] Task 5: Trainer trains Krea LoRAs and links them to the avatar — reviewed (no blocking); advisories applied
- [x] Task 6: Krea photos and first frames with an avatar's LoRA — reviewed; review fixes applied
- [x] Task 7: Dashboard support — reviewed; blocking fix verified in the browser preview
- [x] Task 8: Docs, deploy and live check — deployed; live avatar mirabel built and rendered

## Tasks

### Task 1: Download every LoRA locally

**What**: API route `GET /loras/{name}/versions/{N}/files` returning 1-hour presigned GET links for the
version's final files (`krea2.safetensors` or `t2v_high`/`t2v_low`) and `lora.json`, caller's own registered
versions only (404 otherwise); `--checkpoints` variant via `?checkpoints=1`. CLI: `vidgen lora pull [NAME]
[--checkpoints]` downloads missing versions to `<download dir>/loras/<name>/v<N>/` (stream to `.part`, rename;
skip when sizes match); the watcher's `check` runs the same sync at most every 10 min (timestamp in
`~/.vidgen/`); `lora list` marks local copies and shows the local total size.
**Files**: lambda/api/handler.py, cli/vidgen.py, tests/test_api.py, tests/test_trainer_cli.py
**Depends on**: none
**Verify**: `make test` passes with cases for own/other client and unregistered version (404), a new version
downloaded, an existing one skipped, an interrupted `.part` replaced, the 10-minute interval, `lora list` local
marks.

### Task 2: Avatars record and queue a Krea LoRA build

**What**: `create_avatar` and refresh/pick accept `krea_lora` (default true): true -> queue an `avatar_photoset`
job (`{avatar, version, trigger: <avatar name>, count: 24}`) and write `krea_lora: {state: "queued", trigger,
job_ids}` to `avatar.json`; false -> `state: "skipped"`. New `POST /avatars/{name}/krea-lora` queues a new build
on demand. `GET /avatars` returns the state. CLI: `--no-krea-lora` on avatar create/refresh/pick, `vidgen avatar
krea-train NAME`, `avatar list` shows the state.
**Files**: lambda/api/handler.py, cli/vidgen.py, tests/test_api.py, tests/test_trainer_cli.py
**Depends on**: Task 1
**Verify**: `make test`: create with/without the flag sets queued/skipped and queues one photoset job or none;
`krea-train` queues a new build; the trigger equals the avatar name; list shows the state.

### Task 3: Filter-only scoring in the Krea venv

**What**: `krea_gen.py` takes `{"mode": "score", "images": [paths]}` and returns each image's NSFW score with the
same Falconsai filter (no Krea pipeline load); `krea.score_images(paths) -> [score]` runs it as a subprocess like
`generate`, reusing `ensure_env`/`ensure_weights` (filter only) and the 0.5 threshold constant.
**Files**: worker/krea_gen.py, worker/krea.py, tests/test_worker.py
**Depends on**: none
**Verify**: `make test`: score mode loads only the filter (stub), returns scores in order, flags >= 0.5;
`score_images` passes paths and reads scores; the existing generate tests still pass.
**Parallelizable**: yes — does not share files with Task 1

### Task 4: avatar_photoset job builds the dataset and queues training

**What**: `KREA_DATASET_PROMPTS` (24 outfit/framing/light/place-varied prompts, from the spike set) in
`photos.py`. Worker runner `avatar_photoset`: `models.make_photo` per prompt with the avatar's first 3 images,
1024x1024, seeds base..base+23; `krea.score_images`; drop flagged; < 16 kept -> fail "the photo set was flagged by
the content filter" and set the avatar's `krea_lora.state = failed`; else write images + captions
(`<avatar name>, <prompt>`) + `manifest.json` to `datasets/<client>/avatar-<name>/v<N>/`, put a `train` job item
(`kind: krea2`, `dataset`, `lora: avatar-<name>`, `trigger`, `avatar`) in the jobs table, send it to the train
queue, and set `state: training`. IAM: worker may PutItem on the jobs table, SendMessage on the train queue, write
`datasets/*/avatar-*/*` and `avatars/*/avatar.json`.
**Files**: worker/worker.py, worker/photos.py, terraform/iam.tf, tests/test_worker.py
**Depends on**: Task 2, Task 3
**Verify**: `make test`: 24 photos at 1024x1024 with the avatar's images; captions start with the name; flagged
ones dropped; < 16 fails without queuing; the train job item and SQS message carry `kind: krea2`; avatar state
updates; `terraform validate` passes.

### Task 5: Trainer trains Krea LoRAs and links them to the avatar

**What**: `trainer.py` `Job` gains the Krea path for `kind: krea2`: `ensure_krea_models()` (RAW, Turbo,
`Comfy-Org/Qwen-Image` VAE, `Comfy-Org/Qwen3-VL` 4B; HF token from SSM; clear error for missing RAW access),
dataset TOML at 1024, `krea2_cache_latents` / `krea2_cache_text_encoder_outputs`, `krea2_train_network` with the
spike's arguments (960 steps, save every 240, Turbo samples), resume from the last saved state; upload
`loras/<client>/avatar-<name>/v<N>/krea2.safetensors` + `lora.json` (`models: ["krea2"]`, `source: "avatar"`,
trigger, steps) + samples; set the avatar's `krea_lora` to `ready` with the version (or `failed` with the
error). Trainer user data: a venv with torch >= 2.6 if the current one is older; IAM: trainer reads the HF token
parameter and writes `loras/*/krea2.safetensors` and `avatars/*/avatar.json`.
**Files**: trainer/trainer.py, terraform/templates/trainer_user_data.sh.tpl, terraform/iam.tf, tests/test_trainer_cli.py
**Depends on**: Task 4
**Verify**: `make test` with stubbed musubi/S3: the Krea command lines match the spike's; a resumed job skips
finished phases; upload writes the files and `lora.json`; the uploaded header passes `_check_krea2_lora`; avatar
state ready/failed; missing RAW access fails with the licence message; `terraform validate` passes.

### Task 6: Krea photos and first frames with an avatar's LoRA

**What**: `_validate_photos`: `model: krea2` with an avatar is allowed when its `krea_lora.state` is ready (else
400 naming the state); adds the avatar's LoRA (weight 1.0 unless given) to `krea_loras` with the name as trigger.
`create_job` video/story: `image_model: krea2` + one avatar with a ready LoRA -> Krea first frame with it (Qwen
keyframe for 2+ avatars). Worker: the name is put in front of prompts that don't contain it, also in raw mode,
in `run_photos_krea`, `krea_first_frames`, `story_first_frames`.
**Files**: lambda/api/handler.py, worker/worker.py, tests/test_api.py, tests/test_worker.py
**Depends on**: Task 2
**Verify**: `make test`: ready avatar -> 202 with `krea_loras` containing `avatar-<name>`; not ready -> 400 with
the state; prompts get the name prefixed only when missing; video with one ready avatar uses a Krea frame (no
`make_keyframe`), two avatars keep the keyframe.

### Task 7: Dashboard support

**What**: Avatar creation forms get "Also train a Krea 2 LoRA (~1.5 h GPU; only for people who agreed)"
(checked); the avatar list shows the Krea LoRA state with a "Train Krea LoRA" button (`POST
/avatars/{name}/krea-lora`); Photos "no person" stays as is, and Krea is offered with an avatar whose LoRA is
ready; video/story "First frame: Krea 2" is allowed with one ready avatar; uploaded/trained LoRA rows show "on this
Mac". `ui.py` routes for krea-train and local status.
**Files**: cli/ui.html, cli/ui.py, tests/test_trainer_cli.py
**Depends on**: Task 1, Task 2, Task 6
**Verify**: `make test` (ui.py routes); browser preview: the checkbox is sent, states render, Krea is offered only
for ready avatars, no console errors.

### Task 8: Docs, deploy and live check

**What**: RUNBOOK (avatar Krea LoRAs, triggers = names, local LoRA copies, costs, opt-out) and README; spec live
check section. `make plan` (Lambda, IAM, worker code, trainer user data) and `make apply` after the user's
go-ahead; restart the worker. Live: create a generated avatar from a description, follow it to `ready`, render
Krea photos (`--avatar NAME`) and a Krea-first-frame video, check `<download dir>/loras/avatar-NAME/v1/`.
**Files**: RUNBOOK.md, README.md, docs/specs/2026-11-24-krea-avatar-lora.md
**Depends on**: Task 1, Task 2, Task 3, Task 4, Task 5, Task 6, Task 7
**Verify**: `make test` passes; the live avatar reaches `ready`; Krea photos show the avatar's likeness; the
LoRA files are on the Mac; the content filter stats appear in the job results.
