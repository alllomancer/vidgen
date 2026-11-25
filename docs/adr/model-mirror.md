# Model mirror — decision log

Spec: docs/specs/2026-11-25-model-mirror.md

## D1 — [2026-11-25] — Task 1: Mirror module (pull, push, listing, CLI)

**Status**: Accepted
**Decision**: Restores use our own parallel ranged GETs (64 MB parts, 32 threads, `pwrite` into a preallocated `.part`) instead of boto3's `download_file`; uploads use boto3's multipart `upload_file`.
**Reason**: The disk allows 1000 MB/s and s3transfer funnels every chunk through one writer thread; uploads happen once in the background, so their speed doesn't matter.
**Alternatives considered**: `aws s3 sync` with the CRT client (the CLI version on the boxes is unknown); `boto3[crt]` (a new dependency in three venvs).

## D2 — [2026-11-25] — Task 1: Mirror module (pull, push, listing, CLI)

**Status**: Accepted
**Decision**: A restore never trusts the manifest: paths must stay in the unit's folder, links are leaves whose targets never pass through or end at another link, every write checks the real location of its folder, the total is checked against free disk, and the manifest is named `.mirror-manifest.json`.
**Reason**: It runs as root on data another machine role can write; the security review showed a two-pull symlink-chain escape with a lexical-only check, and a weights folder may hold its own `manifest.json`.
**Alternatives considered**: Allowing only the Hugging Face cache layout for links (other folders may have links); checksums in the manifest (hashing ~400 GB on every start; sizes are what the LoRA copy uses too).

## D3 — [2026-11-25] — Task 1: Mirror module (pull, push, listing, CLI)

**Status**: Accepted
**Decision**: A push merges its files into the existing manifest and does nothing when the manifest already covers the local files; a manifest that can't be read (anything but "not there" or "not usable") stops the push.
**Reason**: Both machines write the caption model's cache folder with slightly different file sets; rewriting on a throttled read would drop the other machine's files (code review finding).
**Alternatives considered**: One unit per machine for that model (stores it twice).

## D4 — [2026-11-25] — Task 2: Ship the module to the trainer and grant both roles `models/` access

**Status**: Accepted
**Decision**: The trainer role may write `models/trainer/*` and only the caption model's folder under `models/hub/`, not all of `models/hub/*` as the spec said.
**Reason**: The render box restores every `hub/` folder at first boot; a narrower write scope limits what a compromised trainer could plant there.
**Alternatives considered**: Separate copies of the caption model per machine.

## D5 — [2026-11-25] — Task 3: Render box Krea weights through the mirror, with timing

**Status**: Accepted
**Decision**: The render box and trainer mirror tests live in new files (`tests/test_worker_mirror.py`, `tests/test_trainer_mirror.py`), not in `test_worker.py` / `test_trainer_cli.py` as the plan said.
**Reason**: `test_worker.py` replaces `krea.ensure_weights` early on, and both files carried unrelated uncommitted work.
**Alternatives considered**: none.

## D6 — [2026-11-25] — Task 6: Trainer early background fetch and backfill at start

**Status**: Accepted
**Decision**: The early fetch is not a recorded phase and is skipped on a trainer whose musubi-tuner lacks Krea support; `krea_env` keeps raising that error. A cancel while waiting stops the job but not the download.
**Reason**: The fetch belongs to the process, not the job (a second job joins it), and a finished download is useful to the next job.
**Alternatives considered**: Moving the support check in front of `prepare` (changes the phase order existing jobs resume by).

## D7 — [2026-11-25] — Task 7: First-boot restore in user data, pins

**Status**: Accepted
**Decision**: Models loaded from the Hugging Face cache by repo id (`video_model`, `i2v_model`, `vlm_model`, `edit_model`) stay unpinned; only the trainer's `--local-dir` downloads and `KREA_FILES` are pinned.
**Reason**: A download at a revision doesn't write `refs/main`, which the worker's offline lookup needs; the mirror holds their exact bytes instead.
**Alternatives considered**: Revision variables plus `revision=` in every loader (touches all model loading).
