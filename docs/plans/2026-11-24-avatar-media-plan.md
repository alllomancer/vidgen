# Avatar media (photos and 2 s clips) — Implementation Plan

Spec: docs/specs/2026-11-24-avatar-media.md
Workspace: current folder (~/Documents/private/myproject; not a git repository, so no commits or SHAs)
Jira: N/A

## Progress

- [x] Task 1: Every new avatar queues the photo-set job, with media state — reviewed with Task 2; fixes in D3
- [x] Task 2: Clips and stored media on the render box — reviewed; blockers fixed (D2, D3)
- [x] Task 3: Media API and media for existing avatars — reviewed; blocker (stuck 409) and advisories fixed
- [x] Task 4: CLI command and watcher download — reviewed with Task 5; sync advisories fixed
- [x] Task 5: Dashboard media state, preview and button — reviewed, browser-checked, accessible names added
- [x] Task 6: Docs, deploy and live check — deployed; live on mirabel and inbal; multi-person photo sets found (follow-up)

## Tasks

### Task 1: Every new avatar queues the photo-set job, with media state

**What**: `create_avatar` always queues `avatar_photoset`, with params `krea_lora: bool`; `avatar.json` gets
`media: {state: queued, job_id, created_at}`; `_write_avatar` preserves `media`. Worker `run_avatar_photoset`: with
`krea_lora` false it makes and filters the photos but writes no Krea dataset and queues no train job (the Krea state
stays `skipped`); media staleness helpers (`_current_media`, `_update_media`) mirror `_current_build`/`_update_build`.
**Files**: lambda/api/handler.py, worker/worker.py, tests/test_api.py, tests/test_worker.py
**Depends on**: none
**Verify**: `make test`: creating an avatar with and without `krea_lora` queues exactly one job with the flag and sets
`media.state` queued; image edits keep `media`; the worker with `krea_lora` false makes no dataset and no train job.

### Task 2: Clips and stored media on the render box

**What**: After the Krea training is queued (or skipped), the job renders four 2 s clips (`models.render_clip`: Wan
I2V, 33 frames, 16 fps, 832x832, no RIFE/upscale/sound/grain) from kept photos at positions 0, n/4, n/2, 3n/4 with
`photos.AVATAR_CLIP_MOTIONS`; filters first/middle/last frames with `krea.score_images` (flagged clip dropped); stores
`avatars/<client>/<name>/media/photo_NN.png`, `clip_N.mp4`, `media.json`; sets `media` ready (counts, preview) or
failed (no clip left / clip error) without touching the Krea build. Progress "clip N/4 of the avatar". IAM: the worker
may write `avatars/*/media/*`.
**Files**: worker/worker.py, worker/models.py, worker/photos.py, tests/test_worker.py, terraform/iam.tf
**Depends on**: Task 1
**Verify**: `make test`: the train job is queued before the first clip call; 4 clips with the right photos, motions,
seeds and frame count; a flagged clip is not stored; zero clips gives media failed and the Krea state unchanged; a
stale job doesn't overwrite media; S3 keys and `media.json` as specified.

### Task 3: Media API and media for existing avatars

**What**: `POST /avatars/{name}/media` queues a media-only `avatar_photoset` job (`media_only: true`, 404 unknown
avatar, 409 while media is queued/running); `GET /avatars/{name}/media` returns `{state, files: [{name, size, url}]}`
with 1 h presigned URLs (404 without media); `GET /avatars` includes each `media` summary. Worker: a media-only job on
an avatar with a ready Krea LoRA downloads that dataset's `raw/` photos instead of making new ones, never queues
training. Routes in Terraform.
**Files**: lambda/api/handler.py, worker/worker.py, terraform/api.tf, tests/test_api.py, tests/test_worker.py
**Depends on**: Task 2
**Verify**: `make test`: routes' 200/404/409 and presigned file list; the media-only job makes no `make_photo` calls
when a dataset exists, makes photos when none exists, and queues no train job.

### Task 4: CLI command and watcher download

**What**: `vidgen avatar media NAME`; `vidgen avatar list` shows the media state; watcher `maybe_sync_avatar_media`
(10-minute cadence and lock like the LoRA sync) downloads ready media into `<download dir>/avatars/<name>/` via
`.part` files with a `.files.json` record, refusing names other than `photo_NN.png`, `clip_N.mp4`, `media.json`.
**Files**: cli/vidgen.py, tests/test_trainer_cli.py
**Depends on**: Task 3
**Verify**: `make test`: the command posts to the route; sync downloads new/changed files only, skips avatars without
ready media, refuses `../x` and other names (logged), and failures are retried next time.

### Task 5: Dashboard media state, preview and button

**What**: The avatar list shows media state, plays the preview clip (muted, looped, from `/api/avatars/{name}/media`)
and offers "Make media" for avatars without media; `cli/ui.py` proxies the two media routes.
**Files**: cli/ui.py, cli/ui.html, tests/test_trainer_cli.py
**Depends on**: Task 4
**Verify**: `make test` for the proxy routes; browser preview: states and button render, preview `<video>` gets a URL,
no console errors (no media job is submitted from the preview).

### Task 6: Docs, deploy and live check

**What**: RUNBOOK (avatar media, command, download folder, time) and README; spec live-check section. `make plan` /
`make apply` after the user's go-ahead. Live: `vidgen avatar media mirabel` (reuses her photos, clips only), check
the files on the Mac and the clips' framing at 832x832; optionally one new avatar end to end.
**Files**: RUNBOOK.md, README.md, docs/specs/2026-11-24-avatar-media.md
**Depends on**: Task 1, Task 2, Task 3, Task 4, Task 5
**Verify**: `make test` passes; the live media job is ready with 1-4 clips; files appear in
`<download dir>/avatars/mirabel/`; clips look natural (else switch to 704x1280 per the spec's risk).
