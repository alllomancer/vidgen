# A Krea 2 LoRA for every avatar

Jira: N/A
Date: 2026-11-24
Status: Approved
Last reviewed: 2026-11-24

## Problem

Krea 2 Turbo is text-to-image only: it can't take an avatar's photos, so Krea photos and Krea first
frames can't show a particular person. The spike of 2026-11-24 (decision log D9) showed a Krea 2
character LoRA trained from Qwen photos of an avatar keeps the identity on Turbo (`smka v1`). The user
wants this for every avatar without extra steps, and a local copy of every LoRA (Wan and Krea) on their
computer.

## Approach

Creating an avatar also starts a **Krea LoRA build** for it, unless turned off for that avatar:

1. **Photo set** (render box, Qwen-Image-Edit): 24 varied photos of the avatar from a fixed prompt set
   (framing, light, places, outfits), captioned `<avatar name>, <prompt>` (the trigger is the avatar's name, see below).
2. **Check** (render box): every photo goes through the same NSFW filter as Krea images (threshold 0.5);
   flagged photos are dropped, and the build stops if fewer than 16 remain.
3. **Train** (trainer machine, musubi-tuner `krea2_train_network.py` on Krea 2 RAW): rank/alpha 32,
   lr 1e-4, flow shift 2.5, 1024 px, 960 steps; Turbo samples at each checkpoint.
4. **Register** as a Krea LoRA named `avatar-<name>` with the **avatar's name as its trigger word** (e.g.
   avatar `smoke-a` -> trigger `smoke-a`) and link it to the avatar.

Once linked, the avatar works in Krea everywhere a person is allowed: Krea photos with that avatar, and
the Krea first frame of videos/story shots with that avatar. In prompts the person is called by the
avatar's name ("smoke-a reading on a park bench"); when the name is missing from the prompt it is put in
front, also in exact-prompt mode, since it names the person.

## Design

### Trigger and records
- Trigger = the avatar name verbatim (names are already `[a-z0-9_-]{1,32}`), stored in `lora.json` and on
  the avatar. Captions always start with it, so training and prompts use the same word.
- `avatar.json` gets `"krea_lora": {"state": "queued|photos|training|ready|failed|skipped", "lora":
  "avatar-<name>", "version": N, "trigger": "<name>", "job_ids": [...], "error": "..."}`.
- `POST /avatars` and avatar refresh (`vidgen avatar refresh`, Pick N) take `krea_lora: true|false`
  (default true; CLI `--no-krea-lora`, dashboard checkbox "Also train a Krea 2 LoRA (~1.5 h GPU)").
  Adding images to an avatar (`vidgen avatar add`) does not retrain automatically; `vidgen avatar
  krea-train NAME` (and a dashboard button) starts a new version on demand. Existing avatars are not
  backfilled automatically.
- A new avatar version (refresh) trains a new LoRA version; the link points at the newest ready one.

### Step 1-2: `avatar_photoset` job (render queue)
- New job type, queued by the API when an avatar is created with `krea_lora`. Worker: `models.make_photo`
  for each of the 24 prompts (fixed list in `worker/photos.py`, `KREA_DATASET_PROMPTS`, outfit-varied so
  the LoRA learns the face, not one outfit), 1024x1024, seeds base..base+23, avatar's first 3 images.
- Each photo is scored by the Krea NSFW filter (`krea_gen` filter, run as one batch in the Krea venv;
  no Krea generation). Flagged photos are dropped and counted; < 16 kept -> job fails ("the photo set was
  flagged by the content filter"), `krea_lora.state = failed`.
- Writes `datasets/<client>/avatar-<name>/v<N>/` (images + `.txt` captions + manifest, the layout train
  jobs already read) and then queues the `train` job itself (new worker permissions: PutItem on the jobs
  table, SendMessage on the train queue), so no client has to wait or chain steps.

### Step 3: `train` job with `kind: krea2` (trainer)
- `trainer.py` gains a Krea path next to Wan: downloads (once, kept on the 300 GB volume) Krea 2 RAW and
  Turbo, `Comfy-Org/Qwen-Image` VAE and `Comfy-Org/Qwen3-VL` 4B text encoder (~66 GB, HF token from SSM;
  RAW is gated: a missing licence acceptance fails the job with that message); caches latents and
  text-encoder outputs; trains with the spike's settings; resumes after spot interruption from the last
  saved state like Wan jobs.
- Uploads `loras/<client>/avatar-<name>/v<N>/krea2.safetensors` + `lora.json` (`models: ["krea2"]`,
  `source: "avatar"`, trigger, steps) and the Turbo samples, then sets `krea_lora.state = ready` and the
  version on the avatar. The upload checks (header, 6144 width) run on it like uploaded LoRAs.

### Using it
- `_validate_photos`: `model: krea2` with an avatar is allowed when that avatar's `krea_lora` is ready
  (else 400 "avatar X has no Krea 2 LoRA yet (state ...)"); the API adds the avatar's LoRA to
  `krea_loras` (weight 1.0 unless given) and the avatar name as trigger to params. Exact-prompt mode
  prepends only the name, and only when the prompt doesn't already contain it.
- `create_job` video/story: `image_model: krea2` with one avatar that has a ready LoRA -> Krea first frame
  with that LoRA and trigger (no Qwen keyframe). Two or more avatars in a shot keep the Qwen keyframe.
- Worker: `krea.generate` already takes LoRAs; the trigger is prefixed in `run_photos_krea`,
  `krea_first_frames` and `story_first_frames`.
- CLI/dashboard: `vidgen avatar list` and the Avatars panel show the Krea LoRA state; the photos and
  video forms offer Krea with an avatar when it is ready.

### Every LoRA downloaded locally
- Applies to all LoRAs, not only avatar ones: trained Wan LoRAs, uploaded Wan/Krea LoRAs and avatar Krea
  LoRAs. The watcher (`vidgen check`, every 45 s) keeps a local copy of each registered version's final
  files under `<download dir>/loras/<name>/v<N>/` (`krea2.safetensors`, or `t2v_high`/`t2v_low`), plus
  `lora.json`. Checkpoints are not downloaded (`vidgen lora pull NAME --checkpoints` gets them on demand).
- New API route `GET /loras/{name}/versions/{N}/files` returns 1-hour presigned GET links for those files
  (signed like every request; only the caller's own LoRAs). The watcher lists LoRAs at most every 10 min
  and only downloads versions it doesn't have (by size), streaming to `.part` then renaming, so a big file
  interrupted by sleep resumes on the next check.
- `vidgen lora pull [NAME]` does the same once; `vidgen lora list` marks versions with a local copy. A
  dashboard row shows "on this Mac". Disk use is reported (`vidgen lora list` total); Wan pairs are ~1.2 GB
  per version, Krea ~450 MB.

## Acceptance Criteria

- When `POST /avatars {name: "ava", ...}` is called without `krea_lora: false`, the system shall queue one
  `avatar_photoset` job and set `krea_lora.state = "queued"` on the avatar.
- When `krea_lora: false` is given, the system shall queue nothing and set `krea_lora.state = "skipped"`.
- When the photoset job runs, the worker shall render 24 photos at 1024x1024 from `KREA_DATASET_PROMPTS`
  with the avatar's images, write each with a caption starting with the trigger to
  `datasets/<client>/avatar-ava/v1/`, and queue a `train` job with `kind: krea2`.
- If the NSFW filter flags a photo (score >= 0.5), then it shall not be written to the dataset; if fewer
  than 16 photos remain, the job shall fail with "the photo set was flagged by the content filter" and
  no train job shall be queued.
- When a `krea2` train job finishes, `loras/<client>/avatar-ava/v1/krea2.safetensors` and `lora.json`
  (`models: ["krea2"]`, `source: "avatar"`) shall exist, the file shall pass `_check_krea2_lora`, and the
  avatar shall show `krea_lora.state = "ready"`, `version: 1`.
- If the token can't read `krea/Krea-2-Raw`, then the train job shall fail with a message naming the
  licence acceptance, and the avatar shall show `state: "failed"` with that error.
- While an avatar's Krea LoRA is ready, `photos {model: krea2, avatar: "ava"}` shall queue with
  `krea_loras` containing `avatar-ava` (trigger `ava`) and `ava` in front of each prompt that doesn't
  contain it; while it is not ready, it
  shall return 400 naming the state.
- When a video has `image_model: krea2` and one avatar with a ready LoRA, the first frame shall be made by
  Krea 2 with that LoRA (no Qwen keyframe); with two avatars, the Qwen keyframe shall be used.
- The content filter shall check every Krea image made with an avatar LoRA (threshold unchanged).
- When a LoRA version is registered (trained, uploaded or avatar), the next watcher check after the
  10-minute list interval shall download its final files and `lora.json` to
  `<download dir>/loras/<name>/v<N>/`; a version already present with matching sizes shall not be
  downloaded again.
- `GET /loras/{name}/versions/{N}/files` shall return presigned links for the caller's own LoRA only and
  404 for another client's or an unregistered version.
- If a download is interrupted, then the `.part` file shall be replaced on the next check and no partial
  file shall have the final name.

## Out of Scope

Syncing deletions (a LoRA removed from S3 stays locally); backfilling existing avatars automatically; training on uploaded (not generated) photos; Krea LoRAs for
multi-avatar shots; tuning steps per avatar; deleting old LoRA versions; changing the content filter.

## Risks & Open Questions

- Cost/time per avatar: ~20 min render box (photos) + ~70 min trainer (first Krea job also downloads
  ~66 GB). Every new avatar costs about $3 of spot GPU; the opt-out exists for that. (Resolved: default on,
  per user.)
- Explicit avatars: their photo sets are flagged by the filter, so they get no Krea LoRA. (Resolved:
  intended.)
- Likeness from one reference image (most generated avatars) depends on Qwen keeping the face across 24
  scenes; the spike used a 1-image avatar and it held. (Resolved for generated avatars; real-photo avatars
  untested.)
- Names as triggers: a name that is a common word (`star`, `b`) is a weaker trigger than a made-up
  token, since the model already knows the word; the LoRA still loads, so likeness may be a bit weaker for
  such names. (Accepted, per user: the trigger is the name.)
- Outfit bias (spike: blue shirt) - the prompt set varies outfits; check on the first real run. (Deferred.)
- Trainer disk: Wan (~60 GB) + Krea (~66 GB) models on 300 GB. (Resolved, fits.)
- Worker gets two new permissions (jobs PutItem, train-queue SendMessage). Alternative: the API chains
  steps on job completion; rejected because the API has no completion hook. (Resolved.)
- Real people: a likeness LoRA can place a person in any image; the dashboard checkbox text says to use
  it only for people who agreed. (Resolved.)

## Testing

Unit: LoRA files route (own/other client, unregistered), watcher sync (new version, already present,
interrupted `.part`, list interval); avatar create/refresh with and without `krea_lora`; photoset job (prompts, sizes, captions, dataset
layout, filter drop, < 16 failure, train job queued); trainer Krea command line, resume, upload and
avatar update (stubbed musubi/S3); Krea photos/video with a ready/not-ready avatar LoRA; CLI and dashboard
states. Live: create a generated avatar from a description, wait for `ready`, render Krea photos and a
Krea-first-frame video with it, compare likeness.

## Update: live check (2026-11-24)

- Deployed (2 API routes, worker settings.json, Lambda, worker/trainer IAM and code). Created the generated
  avatar `mirabel` (Krea candidates, picked 1, not default); its Krea LoRA build started by itself.
- Photo set on the render box 08:25-08:44 (24 photos, filter checked all 24); train job queued 08:46; the
  trainer built its Krea venv and downloaded the Krea models (first Krea job), trained 960 steps (~3.4 s/step),
  registered `avatar-mirabel` v1 and marked the avatar `ready` at ~09:48 (about 1 h 25 min end to end).
- `vidgen photos --model krea2 --avatar mirabel --raw --prompt "mirabel sitting at a cafe table by a rainy
  window, ..."`: 2 images, `loras: [user_avatar_mirabel_v1]`, 0 filtered; `vidgen video "mirabel walking along a
  misty harbour at dawn, ..." --image-model krea2 --avatar mirabel`: Krea first frame, 2.0 s with sound. Both keep
  the avatar's curly red hair, freckles and face.
- Local copy: `~/Movies/vidgen/loras/avatar-mirabel/v1/krea2.safetensors` (448 MB) + `lora.json`; the watcher
  had already copied the other 6 LoRA versions (5.7 GB total). `lora list` row for avatar LoRAs fixed.
