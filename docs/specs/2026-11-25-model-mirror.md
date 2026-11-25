# Model mirror: faster weight downloads on both machines

Jira: N/A
Date: 2026-11-25
Status: Approved
Last reviewed: 2026-11-25

## Problem

Model weights come from Hugging Face every time a machine gets a new disk:

- Trainer, first Krea job: ~66 GB, four files downloaded one after another (`trainer.py` `krea_models`).
- Render box, first Krea job: ~36 GB (`krea.py` `ensure_weights`).
- Render box, first boot: ~25 min of downloads plus the Wan fp32 -> bf16 conversion.
- Trainer, first boot: ~85 GB, ~35 min.

A disk is replaced whenever the instance is (instance type, AZ, spot <-> on-demand: the `on_demand` and
`train_availability_zone` variables make that more common). The GPU subnets reach the internet through a
transit gateway, while the VPC routes S3 through a gateway endpoint, and the disks allow 1000 MB/s. No
download speed was ever recorded, so nobody knows how slow the Hugging Face path is.

## Approach

Keep a copy of every weight file in the project's own bucket (`models/`) and restore from it first. Hugging
Face stays as the fallback and as the source for anything not mirrored yet; whatever a machine has on disk
and the mirror lacks is uploaded in the background. The Hugging Face path itself also gets faster (parallel
files, Xet high-performance mode, earlier start), and every download logs its size and time.

## Design

### `worker/mirror.py` (one module, shipped to both machines)

Terraform uploads it to `code/worker/mirror.py` (existing fileset) and also to `code/trainer/mirror.py` (one
extra `aws_s3_object`), so there is one source file. It only needs `boto3`.

A **unit** is one folder of weights, stored under `s3://<bucket>/models/<unit>/`:

- `manifest.json` — `{"files": {relative path: size}, "links": {relative path: symlink target}}`. Written
  **last**, so a unit without a manifest (an interrupted upload) is never used.
- the files, at their relative paths.

Functions:

- `pull(unit, local_dir, progress=None)` -> `{"source": "mirror", "bytes", "seconds"}`, or `None` when there
  is no usable mirror (no manifest, mirror off, access denied, any error: logged, never raised, so the caller
  falls back to Hugging Face). Files with the right size are kept; the rest are fetched with parallel ranged
  GETs (64 MB parts, 32 threads, `pwrite` into `<file>.part`, renamed when complete and the size matches).
  Symlinks are recreated. `progress` is called from the calling thread (so a cancel raises there).
- `push(unit, local_dir, files=None)` -> stats; never raises. Lists the unit's prefix once, uploads files that
  are missing or differ in size (multipart), then writes the manifest. Does nothing when the mirror's manifest
  already holds every local file with the same size (mirror ⊇ local), so two machines sharing a unit don't
  overwrite each other's manifest in turns.
- `push_async(unit, local_dir, files=None)` — queues a push on one background daemon thread (pushes run one
  after another). The file list is taken when it is queued.
- `pull_all(local_root, skip=())` — pulls every unit that has a manifest into `local_root/<unit>`.
- Local listing skips `.cache/`, `.locks/`, `*.incomplete`, `*.lock`, `*.part`, `.complete`.
- CLI for user data: `mirror.py pull UNIT DIR` (exit 0 restored, 1 not), `mirror.py pull-all ROOT [--skip P]`.
- `MODEL_MIRROR=0` in the environment turns all of it off (pull returns `None`, push does nothing).

### Units

| Unit | Local folder | Written by |
|------|--------------|-----------|
| `krea-2-turbo/<rev12>` | `/opt/models/krea-2-turbo/<rev12>` | render box |
| `nsfw-filter/<rev12>` | `/opt/models/nsfw-filter/<rev12>` | render box |
| `bf16/<name>` | `/opt/models/bf16/<name>` (the converted Wan pipelines) | render box |
| `mmaudio` | `/opt/models/mmaudio` | render box |
| `hub/models--<org>--<name>` | `/opt/models/hub/models--…` (HF cache: blobs, snapshots as symlinks, refs) | render box, trainer |
| `trainer/wan` | `/opt/trainer/models` (the 4 Wan training files, listed explicitly) | trainer |
| `trainer/krea2` | `/opt/trainer/models/krea2` (`KREA_FILES`, listed explicitly) | trainer |

Render-box units are named by their path under `/opt/models`, so first boot restores with one `pull-all`.
The fp32 Wan downloads (`VIDEO_MODEL`, `I2V_MODEL` in the HF cache) are never mirrored: the bf16 copy is.

### Render box

- `krea.ensure_weights`: no `.complete` marker -> `mirror.pull`; if `None`, `snapshot_download` as today;
  write the marker; `mirror.push_async`. Same for `ensure_filter`. Logs one `model_download` JSON line
  (`unit`, `source` = `mirror`/`huggingface`, `bytes`, `seconds`, `mb_per_s`).
- `worker.main` at start: `mirror_backfill()` queues a `push_async` for every unit present and complete on
  disk: bf16 folders with `model_index.json`; HF cache folders without `*.incomplete` files, except
  `VIDEO_MODEL`/`I2V_MODEL`; `mmaudio` when it has files; the Krea and filter folders with their marker.
  Already-mirrored units cost one small GET each. This is how the existing box seeds the mirror without
  downloading anything.
- User data (new boxes only; `user_data` changes are ignored on existing ones): after the code sync and
  `pip install`, `mirror.py pull-all /opt/models --skip trainer/`, then today's commands. The fp32 Wan
  downloads run only when that model's bf16 copy is missing; the other commands find their files and do
  nothing.

### Trainer

- `krea_models`: missing files -> `mirror.pull("trainer/krea2")`; files still missing are downloaded from
  Hugging Face **concurrently** (one thread per file); then `mirror.push_async`. Source, bytes and seconds go
  to the log and into the job result as `krea_download`.
- The fetch **starts in the background when a Krea job starts** (after the musubi-tuner support check), so it
  overlaps `prepare`, `caption` and the Krea venv build; the `krea_models` phase waits for it and raises its
  error. One fetch at a time per process (a second job joins the running one).
- `trainer.main` at start: `push_async` for `trainer/wan`, `trainer/krea2` (when all their files exist) and
  the VLM's HF cache folder.
- User data (new boxes only): `mirror.py pull trainer/wan …` else the `hf download` commands; the same for
  the VLM's cache folder. Krea stays on the first Krea job (ADR krea2-lora D13).

### Faster Hugging Face path

- `HF_XET_HIGH_PERFORMANCE=1`: set with `os.environ.setdefault` at the top of `worker.py`, `krea.py` and
  `trainer.py` (existing boxes), in both env files and the trainer's user-data shell (new boxes).

### IAM (least privilege)

- Render box role: `s3:GetObject` + `s3:PutObject` on `models/hub/*`, `models/bf16/*`, `models/mmaudio/*`,
  `models/krea-2-turbo/*`, `models/nsfw-filter/*`; `s3:ListBucket` for prefix `models/*`.
- Trainer role: the same two actions on `models/trainer/*` and `models/hub/*`; `s3:ListBucket` for `models/*`.
- The render box cannot write the trainer's units (they include a pickled `.pth`), and the reverse. The HF
  cache folder of the caption/planner model is the one unit both write.
- No access for the API Lambda or clients. No lifecycle rule on `models/` (kept until deleted).

### Pins

- Trainer `KREA_FILES` get a revision each: `krea/Krea-2-Raw` `6b0ece7f…`, `krea/Krea-2-Turbo` `98e0fe11…`
  (the render box's pin), `Comfy-Org/Qwen-Image` `1f12b17b…`, `Comfy-Org/Qwen3-VL` `02f0d3ee…`.
- Trainer user data: `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` `5df3a4a3…`, `Wan-AI/Wan2.1-T2V-14B` `a064a6c7…`;
  `boto3==1.35.99` (the worker's version).
- Not pinned: the models both machines load from the HF cache by repo id (`video_model`, `i2v_model`,
  `vlm_model`, `edit_model`). Downloading those at a revision does not write `refs/main`, which the offline
  lookup needs. The mirror holds their exact bytes instead.

## Acceptance Criteria

Mirror module:

1. When `pull` is called for a unit with a manifest, the system shall create every listed file with the listed
   size and content, recreate every listed symlink, and return `source: "mirror"` with the bytes fetched.
2. When `pull` is called and a local file already has the listed size, the system shall not fetch it.
3. If the unit has no `manifest.json`, or reading it fails (access denied), then `pull` shall return `None`
   and leave no `.part` files.
4. If a fetched file's size differs from the manifest, then `pull` shall return `None` and not leave that file
   under its final name.
5. When `push` runs, the system shall upload only files that are missing or differ in size, and write
   `manifest.json` after all of them.
6. When `push` runs and the mirror's manifest already contains every local file with the same size, the
   system shall upload nothing and not rewrite the manifest.
7. If an upload fails, then `push` shall not write the manifest and shall not raise.
8. The local listing shall exclude `.cache/`, `.locks/`, `*.incomplete`, `*.lock`, `*.part` and `.complete`,
   and record symlinks as links, not files.
9. While `MODEL_MIRROR=0`, `pull` shall return `None` and `push` shall make no S3 call.
10. When a file of 0 bytes is listed, `pull` shall create it without a ranged GET.

Render box:

11. When `ensure_weights` runs without the marker and the mirror has the unit, the system shall not call
    `snapshot_download`, shall write the marker, and shall log `model_download` with `source: "mirror"`.
12. When the mirror has no such unit, the system shall call `snapshot_download` as before, write the marker,
    queue a push of the unit, and log `source: "huggingface"`.
13. When the worker starts, `mirror_backfill` shall queue pushes for complete bf16 folders, HF cache folders
    (not `VIDEO_MODEL`/`I2V_MODEL`, none with `*.incomplete` files), `mmaudio`, and marked Krea/filter folders.

Trainer:

14. When Krea files are missing and the mirror has `trainer/krea2`, `krea_models` shall not call Hugging Face.
15. When files are still missing after the mirror, the system shall download them from Hugging Face
    concurrently, each at its pinned revision, and queue a push.
16. When a Krea job starts, the fetch shall start before `prepare`; `krea_models` shall wait for it and raise
    its error (the `krea/Krea-2-Raw` access message is kept).
17. If a second fetch is requested while one runs, the system shall not start another.
18. The Krea job result shall include `krea_download` (`source`, `bytes`, `seconds`) when a download happened.

Infrastructure:

19. `terraform validate` shall pass; the two roles get the object and list permissions above and nothing wider.
20. `make test` shall pass, including the new `tests/test_mirror.py`.

## Out of Scope

- Splitting the branch's uncommitted work into commits.
- Deleting stale units (an old Krea revision stays in the bucket until removed by hand).
- Checksums: a restore is verified by size, as the LoRA copy to the Mac is.
- Mirroring pip wheels or the venvs.
- Pinning the HF-cache models (above).
- Moving the trainer's Krea download into its user data.

## Risks & Open Questions

- **Speed is not measured yet.** The restore is unit-tested, not run against real S3 at this size. The first
  real numbers come from the `model_download` log lines. (Deferred: read them after the next fresh disk.)
- **Xet:** whether the Krea repos are Xet-backed, and how much `HF_XET_HIGH_PERFORMANCE` helps, is unverified.
  It is harmless when not used. (Deferred.)
- **Gateway endpoint:** an S3 route exists in the VPC's route tables; that the GPU subnets use that table was
  not confirmed. Without it S3 traffic also crosses the transit gateway (slower, per-GB cost). (Deferred:
  the `mb_per_s` in the log will show it.)
- **Backfill reads the disk while the box works:** the first start after deploy uploads ~210 GB (render box)
  in the background, sharing the 1000 MB/s volume with model loads. One time; an idle stop interrupts it and
  the next start continues (files already uploaded are skipped). (Accepted.)
- **A file truncated by an earlier killed download** is mirrored as it is (it is just as broken locally).
  Units are only pushed when their completeness check passes. (Accepted.)
- **Cost:** ~400 GB in S3 Standard, about $9 a month. (Resolved: approved, everything mirrored.)
- **License:** the Krea weights are gated; the copy stays in a private bucket readable only by the two machine
  roles, for the same use the license was accepted for. (Resolved.)
- **Pinned `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` revision** is today's head (changed 2026-11-18, after the
  trainer was built), so a rebuilt trainer could get slightly different files than the running one; the mirror
  (seeded from the running trainer) is used first. (Accepted.)
- Feature flag: `MODEL_MIRROR=0` on a machine disables it. No wider flag needed.

## Testing

- `tests/test_mirror.py`: an in-memory S3 with ranged GETs and a temp folder, covering criteria 1-10.
- `tests/test_worker.py`: `ensure_weights` with and without a mirror; `mirror_backfill` unit selection.
- `tests/test_trainer_cli.py`: mirror hit, Hugging Face fallback in parallel with revisions, early start,
  single flight, `krea_download` in the result.
- `terraform validate`.
- Live (after `terraform apply`, by the user): start each machine once and check the `mirror_push` lines and
  `aws s3 ls s3://<bucket>/models/`; on the next fresh disk, read the `model_download` lines.
