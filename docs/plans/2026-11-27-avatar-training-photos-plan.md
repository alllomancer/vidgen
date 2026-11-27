# Avatar training photos — Implementation Plan

Spec: docs/specs/2026-11-27-avatar-training-photos.md
Workspace: current branch (main)
Jira: N/A

## Progress

- [ ] Task 1: Spike: face-similarity threshold and Krea non-square images
- [ ] Task 2: Training-photo pool in the API (add, remove, list, delete with avatar)
- [ ] Task 3: CLI `vidgen avatar train-photos` and the refresh tip
- [ ] Task 4: Waker queues the debounced automatic retrain
- [ ] Task 5: Worker face check (facenet-pytorch, mirrored weights)
- [ ] Task 6: Worker builds the mixed training set and writes photo statuses
- [ ] Task 7: Dashboard "Training photos" strip
- [ ] Task 8: Phone: add training photos
- [ ] Task 9: Docs and live check

## Tasks

### Task 1: Spike: face-similarity threshold and Krea non-square images

**What**: On the render box, embed mirabel's and inbal's reference images and their stored media photos with
facenet-pytorch VGGFace2; record same-person vs different-person similarities and pick `SAME_PERSON_MIN` (start 0.6).
Run a 50-step Krea LoRA training on 4 non-square images to see whether the trainer buckets them or needs squares.
Results go in the spec's Risks (both marked Resolved with the numbers).
**Files**: `trainer/spike/face_threshold.py` (new), `docs/specs/2026-11-27-avatar-training-photos.md`
**Depends on**: none
**Verify**: spec Risks updated with the measured similarities, the chosen threshold, and resize-vs-crop decided
**Parallelizable**: yes — does not share files with Task 2

### Task 2: Training-photo pool in the API (add, remove, list, delete with avatar)

**What**: `POST /avatars/{name}/training-photos` (1-50 `upload_keys`, image checks as reference images, moves to
`avatars/<client>/<name>/training/<id>.<ext>`, appends `{id, key, added_at, status: new}`, sets
`retrain_at = now + 600` and `avatars/<client>/retrain.json`), `DELETE .../training-photos/{id}` (same
`retrain_at`), `GET /avatars/{name}` returns `training_photos` with 1-hour preview links and `retrain_at`; avatar
delete removes `training/` and its `retrain.json` entry. Terraform adds the two routes.
**Files**: `lambda/api/handler.py`, `terraform/api.tf`, `tests/test_training_photos_api.py` (new)
**Depends on**: none
**Verify**: `python3 -B tests/test_training_photos_api.py` (spec criteria 1, 9, and the 12:00/12:04 → 12:14
`retrain_at` part of criterion 2)
**Parallelizable**: yes — does not share files with Task 1

### Task 3: CLI `vidgen avatar train-photos` and the refresh tip

**What**: `vidgen avatar train-photos add [--name N] FILE...` (uploads, then the new endpoint; prints "retrains at
HH:MM"), `list` (id, status, reason), `remove ID`. The `avatar refresh` tip points at `train-photos add`.
**Files**: `cli/vidgen.py`, `tests/test_training_photos_cli.py` (new)
**Depends on**: Task 2
**Verify**: `python3 -B tests/test_training_photos_cli.py` with the API stubbed; `make test`

### Task 4: Waker queues the debounced automatic retrain

**What**: `waker()` reads each client's `retrain.json`; for each avatar with `retrain_at <= now` and no live Krea or
Wan build, it clears `retrain_at` and calls `_queue_krea_lora(client, name, wan=True)` with
`training_photos: true` (Wan only for `krea_lora: false` avatars); with a live build it leaves `retrain_at`.
Clients are listed from the `avatars/` prefix (one `list_objects_v2` with `Delimiter`).
**Files**: `lambda/api/handler.py`, `tests/test_training_photos_waker.py` (new)
**Depends on**: Task 2
**Verify**: `python3 -B tests/test_training_photos_waker.py` (spec criteria 2, 3: exactly one build at/after
12:14, none while a build is live, `retrain_at` kept)

### Task 5: Worker face check (facenet-pytorch, mirrored weights)

**What**: `worker/faces.py`: MTCNN + `InceptionResnetV1(pretrained="vggface2")` on CPU, `reference_vector(paths)`
(normalised mean, skips faceless; none → `ValueError("no face found in the reference images")`) and
`check(path, ref) -> (ok, reason, similarity)` with `SAME_PERSON_MIN` from Task 1. Pinned in
`worker/requirements.txt`; weights added to the model mirror and fetched from it.
**Files**: `worker/faces.py` (new), `worker/requirements.txt`, `worker/mirror.py`, `tests/test_faces.py` (new, model
mocked)
**Depends on**: Task 1
**Verify**: `python3 -B tests/test_faces.py` (threshold edge 0.43 → `looks like someone else (0.43)`, no face, two
faces); `python3 -B tests/test_worker_mirror.py`

### Task 6: Worker builds the mixed training set and writes photo statuses

**What**: In `run_avatar_photoset` (when `training_photos` or the avatar has any): load training photos, check
people count + face, EXIF-rotate and resize/crop per Task 1, caption with Qwen2.5-VL `<trigger>, <scene>`; generate
`clamp(36 - kept_real, 12, 24)` photos and face-check them too; fail under `AVATAR_DATASET_MIN` naming both counts;
manifest gets `real_photos`/`generated_photos`; Krea and Wan datasets both use the set; write each photo's
`status`/`reason` to `avatar.json` while still the current build.
**Files**: `worker/worker.py`, `worker/models.py` (caption helper), `tests/test_training_photos_worker.py` (new)
**Depends on**: Tasks 2, 5
**Verify**: `python3 -B tests/test_training_photos_worker.py` (spec criteria 4-8, 10 with models mocked);
`python3 -B tests/test_worker.py` still passes

### Task 7: Dashboard "Training photos" strip

**What**: Avatars tab: per-avatar strip of training-photo thumbnails with kept/rejected/new badges (reason on
hover), Add (multi-select, via `ui.py` `POST /api/avatars/{name}/training-photos` using the same base64 path as
`create_avatar`), remove ×, "Retrains at HH:MM".
**Files**: `cli/ui.py`, `cli/ui.html`, `tests/test_ui_page.py`
**Depends on**: Task 2
**Verify**: `make test`; in `vidgen ui`, add 3 photos and see them listed as new with a retrain time

### Task 8: Phone: add training photos

**What**: Phone Avatars screen: "Add training photos" from the camera roll (JPEG conversion + presigned PUT from
the phone plan's Task 7), list with badges, remove.
**Files**: `lambda/api/phone/app.js`, `tests/test_phone_page.py`
**Depends on**: Task 2, and phone-ui plan Task 8
**Verify**: `python3 -B tests/test_phone_page.py`; from the phone, photos appear in the dashboard's strip

### Task 9: Docs and live check

**What**: README (avatars paragraph), RUNBOOK (training photos, automatic retrain, rejection reasons),
`CHANGELOG.md` dated entry. Live check from the spec: 15 real photos (2 of someone else, 1 group) on a test avatar;
results in the spec.
**Files**: `README.md`, `RUNBOOK.md`, `CHANGELOG.md`, `docs/specs/2026-11-27-avatar-training-photos.md`
**Depends on**: Tasks 3, 4, 6, 7
**Verify**: the 3 bad photos rejected with the right reasons, one rebuild ~10 min after the last add, likeness held
in a Krea photo and a 30 s video; results written in the spec
