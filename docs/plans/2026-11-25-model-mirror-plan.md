# Model mirror — Implementation Plan

Spec: docs/specs/2026-11-25-model-mirror.md
Workspace: current branch (avatar-media-and-wan-lora)
Jira: N/A

## Progress

- [ ] Task 1: Mirror module (pull, push, listing, CLI)
- [ ] Task 2: Ship the module to the trainer and grant both roles `models/` access
- [ ] Task 3: Render box Krea weights through the mirror, with timing
- [ ] Task 4: Render box backfill at worker start
- [ ] Task 5: Trainer Krea files: mirror, parallel pinned Hugging Face fallback, timing in the result
- [ ] Task 6: Trainer early background fetch and backfill at start
- [ ] Task 7: First-boot restore in user data, pins
- [ ] Task 8: Docs (RUNBOOK, README, decision log)

## Tasks

### Task 1: Mirror module (pull, push, listing, CLI)

**What**: New `worker/mirror.py`: `enabled()`, `local_files()` (skips `.cache/`, `.locks/`, `*.incomplete`,
`*.lock`, `*.part`, `.complete`; symlinks as links), `pull(unit, local_dir, progress=None)` (manifest, keep
files of the right size, 64 MB ranged GETs on 32 threads with `pwrite` into `.part`, size check, rename,
symlinks, 0-byte files without a GET; returns stats or `None`, never raises except the caller's cancel from
`progress`), `push(unit, local_dir, files=None)` (one prefix listing, upload missing/different sizes, manifest
last, no-op when the mirror's manifest ⊇ local, never raises), `push_async` (one background thread, a queue),
`pull_all(local_root, skip=())`, and a CLI (`pull UNIT DIR`, `pull-all ROOT [--skip P]`). `MODEL_MIRROR=0`
disables it. Logs `mirror_pull` / `mirror_push` JSON lines.
**Files**: `worker/mirror.py` (new), `tests/test_mirror.py` (new)
**Depends on**: none
**Verify**: `python3 -B tests/test_mirror.py` (spec criteria 1-10) and `make test`

### Task 2: Ship the module to the trainer and grant both roles `models/` access

**What**: One `aws_s3_object` uploading `worker/mirror.py` to `code/trainer/mirror.py` (when training is
enabled). Render box role: Get/Put on `models/hub/*`, `models/bf16/*`, `models/mmaudio/*`,
`models/krea-2-turbo/*`, `models/nsfw-filter/*`, and `models/*` added to its ListBucket prefixes. Trainer
role: Get/Put on `models/trainer/*` and `models/hub/*`, `models/*` added to its ListBucket prefixes.
**Files**: `terraform/training.tf`, `terraform/iam.tf`
**Depends on**: Task 1
**Verify**: `terraform -chdir=terraform validate` and `terraform fmt -check` on the two files (criterion 19)

### Task 3: Render box Krea weights through the mirror, with timing

**What**: `krea.ensure_weights` and `ensure_filter`: without the marker, `mirror.pull` first, else
`snapshot_download` as today; write the marker; `mirror.push_async`; print one `model_download` line (`unit`,
`source`, `bytes`, `seconds`, `mb_per_s`). `HF_XET_HIGH_PERFORMANCE=1` via `os.environ.setdefault` at the top
of `worker.py` (before other imports) and `krea.py`.
**Files**: `worker/krea.py`, `worker/worker.py`, `tests/test_worker.py`
**Depends on**: Task 1
**Verify**: `python3 -B tests/test_worker.py` (criteria 11, 12)

### Task 4: Render box backfill at worker start

**What**: `worker.mirror_backfill()` called from `main()` before the loop: queues `push_async` for bf16
folders with `model_index.json`, HF cache folders under `$HF_HOME/hub` (not `VIDEO_MODEL`/`I2V_MODEL`, none
containing `*.incomplete`), `mmaudio` when it has files, and the Krea / filter folders with their marker. Any
error is logged and never stops the worker.
**Files**: `worker/worker.py`, `tests/test_worker.py`
**Depends on**: Task 3
**Verify**: `python3 -B tests/test_worker.py` (criterion 13)

### Task 5: Trainer Krea files: mirror, parallel pinned Hugging Face fallback, timing in the result

**What**: `KREA_FILES` entries get a revision (spec "Pins"). New `krea_fetch()`: missing files ->
`mirror.pull("trainer/krea2")`; still-missing files downloaded concurrently (one thread each) with
`_hf_download(repo, filename, local_dir, token, revision)`, keeping the `krea/Krea-2-Raw` access message;
then `mirror.push_async` with the explicit file list; returns `{source, bytes, seconds}`. `krea_models` calls
it, logs `model_download`, stores it in the job state, and `krea_upload` adds it to the result as
`krea_download`. `HF_XET_HIGH_PERFORMANCE=1` at the top of `trainer.py`.
**Files**: `trainer/trainer.py`, `tests/test_trainer_cli.py`
**Depends on**: Task 1
**Verify**: `python3 -B tests/test_trainer_cli.py` (criteria 14, 15, 18)

### Task 6: Trainer early background fetch and backfill at start

**What**: A module-level single-flight fetch (`krea_fetch_start()` / `krea_fetch_wait()`): a Krea job starts
it right after the musubi-tuner support check (moved in front of `prepare`), `krea_models` waits for it and
re-raises its error; a second request while one runs joins it. `trainer.mirror_backfill()` at `main()` start:
`push_async` for `trainer/wan` and `trainer/krea2` when all their files exist, and the VLM's HF cache folder.
**Files**: `trainer/trainer.py`, `tests/test_trainer_cli.py`
**Depends on**: Task 5
**Verify**: `python3 -B tests/test_trainer_cli.py` (criteria 16, 17) and `make test`

### Task 7: First-boot restore in user data, pins

**What**: Render box template: `HF_XET_HIGH_PERFORMANCE=1` in the env file; after `pip install`,
`mirror.py pull-all /opt/models --skip trainer/` (failure tolerated); the two fp32 Wan downloads only when
that model's bf16 copy is missing. Trainer template: the same env line and an export for the shell;
`boto3==1.35.99`; `mirror.py pull trainer/wan /opt/trainer/models` else the `hf download` commands with
`--revision` (spec "Pins"); `mirror.py pull hub/models--… ` else the VLM download.
**Files**: `terraform/templates/user_data.sh.tpl`, `terraform/templates/trainer_user_data.sh.tpl`
**Depends on**: Task 1, Task 2
**Verify**: `terraform -chdir=terraform validate`; both templates rendered with dummy values pass `bash -n`

### Task 8: Docs (RUNBOOK, README, decision log)

**What**: RUNBOOK: a "Model mirror" section (what it holds, cost, how it seeds itself, how to check it, how
to turn it off or remove a unit, where the timing lines are) and updated first-boot / first-Krea-job wording;
README: one sentence; `docs/adr/model-mirror.md` with the decisions made (mirror in the project bucket,
manifest-last, size check not checksums, per-role prefixes, HF-cache models not pinned).
**Files**: `RUNBOOK.md`, `README.md`, `docs/adr/model-mirror.md` (new)
**Depends on**: Tasks 1-7
**Verify**: every command and path named in the docs exists in the code (`grep`)
