# Avatar media: photos and 2 s clips for every avatar

Jira: N/A
Date: 2026-11-24
Status: Approved
Last reviewed: 2026-11-24

## Problem

A new avatar gets candidate images and (by default) a Krea 2 LoRA built from 24 generated photos, but those photos
stay inside a training dataset in S3 and nothing shows how the avatar looks in motion. The user wants every avatar to
come with photos and a short clip, on their Mac, both to look at and as training material: the planned per-avatar Wan
LoRA (Part B, specced after the Wan LoRA spike) will train on the photos plus clips, since clips of the person moving
should help identity hold in video.

## Approach

Extend the existing `avatar_photoset` job (render box) so it always runs for a new avatar: it makes the 24 filtered
photos as today, queues the Krea LoRA training as today when the avatar has one, then renders four 2 s Wan clips from
four of the photos and stores photos and clips as the avatar's media. The watcher downloads media to
`<download dir>/avatars/<name>/`. A command and a dashboard button make media for existing avatars, reusing their
existing photo set when there is one.

## Design

### API (`lambda/api/handler.py`)
- `create_avatar`: always queues the `avatar_photoset` job; params gain `krea_lora: bool` (from the existing
  `krea_lora` flag, default true). With `krea_lora` false, `avatar.json` `krea_lora` stays `skipped` as today, and
  the job makes media only.
- `avatar.json` gains `media`: `{state: queued|photos|clips|ready|failed, job_id, created_at, photos: n,
  clips: n, preview: "clip_1.mp4", error?}`; `_write_avatar` preserves it like `krea_lora`. The same staleness rule as
  the Krea build applies: updates only land while `media.job_id` is this job and `created_at` matches.
- `POST /avatars/{name}/media`: queue a media-only job for an existing avatar (409 while one is queued or running;
  404 for an unknown avatar). Reuses the avatar's current Krea dataset photos when `krea_lora.ready` names one, else
  generates the photo set.
- `GET /avatars/{name}/media`: `{state, files: [{name, size, url}]}` with presigned GET URLs (1 h) for every media
  file; 404 when the avatar has no media.
- `GET /avatars` includes each avatar's `media` summary.

### Worker (`worker/worker.py`, `worker/models.py`, `worker/photos.py`)
- `avatar_photoset` job, in order:
  1. Photos: as today (24 Qwen photos, 1024x1024, content filter, at least 16 kept), or, for a media-only job on an
     avatar with a Krea dataset, download its `raw/` photos instead.
  2. Krea: when `krea_lora` is true, write the dataset and queue the train job exactly as today. A media-only job
     never queues training.
  3. Clips: four 2 s clips (33 frames at 16 fps, 832x832, Wan I2V, Lightning, no RIFE/upscale/sound/grain), each
     from a different kept photo (positions 0, n/4, n/2, 3n/4 of the kept list) with one of
     `photos.AVATAR_CLIP_MOTIONS`: turns their head slowly and smiles; takes a step towards the camera; looks around
     the room, then back at the camera; laughs softly and tilts their head. Prompt: avatar name + motion + "natural
     light, static camera". Seed: job base + clip index.
  4. Content filter on clips: the first, middle and last frame of each clip are scored with `krea.score_images`; a
     flagged clip is dropped. Zero clips left: `media.state = failed` with the reason; photos are still stored.
  5. Store: `avatars/<client>/<name>/media/photo_NN.png` (kept photos), `clip_N.mp4`, `media.json` (file list,
     prompts, motions, seeds, created_at); `avatar.json` `media` updated (`ready`, counts, preview = first clip).
- A failure in steps 3-5 sets `media.state = failed` and does not affect the Krea build (its training is already
  queued). A failure in step 1 fails both, as today.
- Progress: "photo N/24 for the avatar", "clip N/4 of the avatar".

### CLI / watcher (`cli/vidgen.py`, `cli/ui.py`, `cli/ui.html`)
- `vidgen avatar media NAME`: queues the media job; `vidgen avatar list` shows the media state.
- Watcher (`check_once`): `maybe_sync_avatar_media`, on the same 10-minute cadence and lock style as the LoRA sync:
  for each avatar with `media.state == ready`, download missing or changed files (by name and size, recorded in
  `.files.json`) into `<download dir>/avatars/<name>/`, via `.part` files; names from the server must be plain file
  names (`photo_NN.png`, `clip_N.mp4`, `media.json`), anything else is refused.
- Dashboard: the avatar list shows each avatar's media state, plays the preview clip (presigned URL via
  `/api/avatars/{name}/media`) and has a "Make media" button for avatars without media.

## Acceptance Criteria

- When an avatar is created (with or without `krea_lora`), the API shall queue one `avatar_photoset` job and set
  `avatar.json` `media.state` to `queued`.
- When the job finishes on a new avatar, `avatars/<client>/<name>/media/` shall hold the kept photos (16-24
  `photo_NN.png`), 1-4 `clip_N.mp4` of 33 frames at 16 fps, 832x832, and `media.json`; `media.state` shall be
  `ready` with `preview` = `clip_1.mp4`.
- When `krea_lora` is false, the job shall not write a Krea dataset or queue a train job.
- When `krea_lora` is true, the Krea train job shall be queued before the first clip is rendered.
- If a clip's first, middle or last frame is flagged by the content filter, then that clip shall not be stored; if
  no clip is left, then `media.state` shall be `failed` and the Krea build shall be unaffected.
- When `POST /avatars/{name}/media` is called for an avatar with a ready Krea LoRA, the job shall reuse that dataset's
  photos (no `make_photo` calls) and render only the clips; while a media job is queued or running, the API shall
  return 409.
- When the avatar is replaced or a newer media job starts, an older job shall not overwrite `media`.
- When the watcher runs and an avatar's media is ready, the files shall be downloaded to
  `<download dir>/avatars/<name>/`; a file name from the server that is not `photo_NN.png`, `clip_N.mp4` or
  `media.json` shall be refused and logged.

## Out of Scope

The per-avatar Wan LoRA (Part B: training on photos + clips, auto-attaching to video jobs), specced after the Wan LoRA
spike; clips with sound; clips longer than 2 s; picking which photos to animate; media for refused avatars (the
content filter decides as today).

## Risks & Open Questions

- Render time: ~6-8 min per avatar for the clips (I2V load plus 4 x ~1 min) on top of the photo set. (Resolved:
  acceptable per the user.)
- Square 832x832 is outside Wan's usual 16:9 sizes; clips may frame or move less naturally. (Needs live check; fall
  back to 704x1280 vertical if square looks wrong.)
- Clips rendered by Wan from Wan-adjacent photos may teach the later Wan LoRA Wan's own artefacts. (Deferred to
  Part B.)
- Disk on the Mac: ~35 MB of photos + ~4 MB of clips per avatar. (Resolved.)

## Testing

Unit: API (`create_avatar` queues the job with and without `krea_lora`, media routes, 409/404, `_write_avatar`
keeps `media`); worker (`avatar_photoset` order: train job queued before clips, krea false skips dataset, media-only
reuses dataset photos, clip filter drops flagged clips, zero clips fails media only, staleness, S3 keys and
`media.json`); CLI (watcher sync: downloads, `.files.json`, refuses bad names; `avatar media` command); dashboard in
the browser preview. Live: create one avatar and check photos and clips on the Mac; run `vidgen avatar media mirabel`.

## Live check (2026-11-24)

- mirabel (`vidgen avatar media --name mirabel`, ready Krea LoRA): her 24 dataset photos reused, 4 clips rendered and
  kept, ready in ~7 min; the Mac's watcher copied everything to `~/Movies/vidgen/avatars/mirabel/` on its own.
- inbal (no ready Krea LoRA): 24 new photos + 4 clips, ~40 min; copied to `~/Movies/vidgen/avatars/inbal/`.
- Clips: 832x832, 33 frames, 16 fps; square framing looks natural, the motions read well (smile, step, laugh, tilt).
- Problem found: about 12 of inbal's 24 photos show two or three women. Her 3 reference photos differ strongly (hair
  colour, light, distance; one has a partial second person at the edge), and Qwen-Image-Edit draws them as separate
  people. This affects every avatar photo set (also the Krea LoRA dataset), not just media. Follow-up: a one-person
  check (Qwen2.5-VL counts people per photo; photos without exactly one person are dropped), planned with the avatar
  Wan LoRA spec. inbal's in-flight Krea build was cancelled until then (D4).
