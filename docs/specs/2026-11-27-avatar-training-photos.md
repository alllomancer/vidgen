# Avatar training photos

Jira: N/A
Date: 2026-11-27
Status: Approved
Last reviewed: 2026-11-27

## Problem

The user wants to give an avatar more real photos so its LoRAs learn the person better. Today extra photos don't
help training:

- An avatar holds at most 12 reference images (`MAX_AVATAR_IMAGES`).
- The Krea and Wan LoRAs train only on 24 photos Qwen generates from the **first 3** reference images
  (`_make_photo_set`, `worker/worker.py`); images 4-12 and the user's real photos never reach training. Training on
  uploaded photos was out of scope in docs/specs/2026-11-24-krea-avatar-lora.md.
- The only identity check is "exactly one person" (`_one_person`); nothing checks it is the *same* person, which is
  how 12 of inbal's 24 photos mixed faces (docs/specs/2026-11-24-avatar-wan-lora.md).
- `vidgen avatar refresh` prints "more photos of the same person improve likeness: vidgen.py avatar add", which isn't
  true for training.

## Approach

Each avatar gets a separate pool of up to 50 **training photos**, used only for its LoRAs; the 12 reference images
and how keyframes use them don't change. A LoRA build trains on the real photos that pass the checks plus generated
photos for variety. Every photo, real or generated, must show exactly one person and match the reference face
(face-embedding similarity). Adding photos schedules an automatic rebuild 10 minutes after the last addition.

## Design

### Storage and API

- Files: `avatars/<client>/<name>/training/<id>.<ext>` (png/jpg/webp, max 25 MB each, same checks as reference
  images). `avatar.json` gains `training_photos: [{id, key, added_at, status, reason?}]`, `status` one of `new`,
  `kept`, `rejected` (set by the last build), and `retrain_at` (unix time or absent).
- `POST /avatars/{name}/training-photos` `{upload_keys: [...]}` (1-50 per call): moves the uploads under
  `training/`, appends them with `status: new`, sets `retrain_at = now + 600`. Over 50 in total → 400
  `an avatar holds at most 50 training photos ('<name>' has N)`.
- `DELETE /avatars/{name}/training-photos/{id}`: removes one; sets `retrain_at = now + 600` too.
- `GET /avatars/{name}` returns `training_photos` with 1-hour preview links and `retrain_at`.
- Deleting an avatar deletes its training photos.

### Clients

- CLI: `vidgen avatar train-photos add [--name N] FILE...`, `vidgen avatar train-photos list [--name N]` (id,
  status, reason), `vidgen avatar train-photos remove [--name N] ID`. The `avatar refresh` tip points at
  `train-photos add`.
- Dashboard (Avatars tab): a "Training photos" strip per avatar with thumbnails, status badges (kept / rejected with
  reason on hover / new), an Add button (multi-select) and a remove ×, and "Retrains at HH:MM" while `retrain_at` is
  set.
- Phone (docs/specs/2026-11-27-phone-ui.md): Avatars → Add training photos from the camera roll.

### Automatic retrain

- The waker Lambda (EventBridge, every minute) lists avatars with `retrain_at <= now` (an index object
  `avatars/<client>/retrain.json` `{name: retrain_at}` written alongside `avatar.json`, so it doesn't scan every
  avatar). For each, when no Krea or Wan build of that avatar is live (`KREA_LORA_BUSY`/`WAN_LORA_BUSY` with a live
  job), it clears `retrain_at` and queues a build with the existing `_queue_krea_lora(client, name, wan=True)` plus
  `training_photos: true`; when a build is live, it leaves `retrain_at` and checks again next minute.
- So a batch added over a few minutes costs one rebuild; photos added during a build start one more after it.
- The rebuild reuses the avatar's stored media clips for the Wan dataset (as `_wan_only` does) and doesn't make new
  clips. The last ready LoRAs stay linked until the new versions are ready (existing behaviour).
- Avatars created with `krea_lora: false` get only the Wan rebuild.

### Building the training set (worker, `avatar_photoset`)

1. **Reference face:** a face embedding (facenet-pytorch `InceptionResnetV1(pretrained="vggface2")` with MTCNN for
   detection, CPU) of each of the first 3 reference images; the reference vector is their normalised mean. A
   reference image with no face is skipped; none with a face → the build fails with
   `no face found in the reference images`.
2. **Real photos:** each training photo must have exactly one person (`models.count_people`) and one detected face
   with cosine similarity ≥ `SAME_PERSON_MIN = 0.6` to the reference vector. Failures get `status: rejected`,
   `reason: "no person" | "more than one person" | "no face" | "looks like someone else (0.43)"`; passes get `kept`.
   Kept photos are EXIF-rotated, resized so the short side is 1024 (long side ≤ 2048, aspect kept) and captioned by
   Qwen2.5-VL as `<trigger>, <short scene description>`.
3. **Generated photos:** `n_gen = clamp(36 - kept_real, 12, 24)` from `KREA_DATASET_PROMPTS` as today (0 real → 24,
   the current behaviour; 24+ real → 12). They go through the content filter and `_one_person` as now, plus the same
   face check.
4. Dataset = kept real + kept generated; needs ≥ `AVATAR_DATASET_MIN` (16) in total, else the build fails with the
   counts. The manifest records `real_photos` and `generated_photos`; the Krea dataset and the Wan dataset (photos +
   stored clips) both use it.
5. The worker writes each photo's `status`/`reason` back to `avatar.json` (only while this job is still the current
   build), so the UIs show why a photo was rejected.

### Dependencies

`facenet-pytorch` (MIT) and its VGGFace2 weights (~110 MB) added to `worker/requirements.txt`, pinned, and mirrored
like the other models (docs/specs/2026-11-25-model-mirror.md).

## Acceptance Criteria

- When `POST /avatars/{name}/training-photos` adds photos to an avatar with 48, and the request has 3, the API shall
  return 400 `an avatar holds at most 50 training photos ('<name>' has 48)` and store nothing.
- When photos are added at 12:00:00 and more at 12:04:00, `retrain_at` shall be 12:14:00 and the waker shall queue
  exactly one rebuild, on its first run at or after 12:14:00.
- While a Krea or Wan build of the avatar is live, the waker shall not queue another and shall keep `retrain_at`.
- When a training photo shows two people, the build shall mark it `rejected` with reason `more than one person` and
  leave it out of the dataset.
- When a training photo's face has cosine similarity 0.43 to the reference vector, the build shall mark it `rejected`
  with reason `looks like someone else (0.43)`.
- When 10 real photos are kept, the build shall generate 24 photos; with 30 kept, 12; with 0, 24.
- If kept real + kept generated photos number fewer than 16, then the build shall fail with an error naming both
  counts, and the last ready LoRAs shall stay linked.
- The Krea and Wan datasets shall contain every kept real photo, captioned `<trigger>, ...`.
- When an avatar is deleted, its `training/` objects shall be deleted.
- The 12 reference images and keyframe generation shall behave exactly as before.

## Out of Scope

Raising the 12-reference-image cap; choosing which reference images drive keyframes; tuning LoRA steps by dataset
size; training on video clips the user provides; per-photo caption editing in the UI; backfilling existing avatars
(the first added photo triggers their rebuild anyway).

## Risks & Open Questions

- `SAME_PERSON_MIN = 0.6` is a starting value for VGGFace2 embeddings; check it on the existing avatars (mirabel,
  inbal) before relying on it, and log every similarity. (Needs spike.)
- The Krea trainer's handling of non-square images is unknown; if it needs squares, the resize becomes a centre crop
  to 1024×1024 around the detected face. (Needs spike.)
- Each automatic rebuild costs ~1-1.5 h of trainer GPU time; the 10-minute delay groups a batch, but adding photos
  one by one over hours triggers several rebuilds. (Deferred: accept, the user chose automatic.)
- Applying the face check to generated photos may drop more of them for avatars whose references vary; the error then
  says how many were kept, as today. (Resolved.)
- Real photos are of a real person and train a LoRA of them; that's the purpose and stays in the user's private
  bucket. (Resolved.)

## Testing

- Unit (tests/): training-photo add/remove/limit, `retrain_at` debounce and the retrain index, waker queuing one
  build and skipping while one is live, `n_gen` values, rejected reasons, avatar delete removing `training/`.
- Worker unit tests with the face model mocked: similarity threshold, no-face handling, dataset composition and
  manifest counts.
- Live check: add 15 real photos (2 of them of someone else, 1 group photo) to a test avatar; the 3 are rejected with
  the right reasons, one rebuild starts ~10 min later, and a Krea photo and a 30 s video with the new LoRAs keep the
  likeness at least as well as before.
