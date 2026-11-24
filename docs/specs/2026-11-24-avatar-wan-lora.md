# Avatar Wan LoRA and one-person photo sets

Jira: N/A
Date: 2026-11-24
Status: Approved
Last reviewed: 2026-11-24

## Problem

Long takes lose the avatar's likeness: in a 50 s take mirabel's short red curls turned auburn and her face changed.
Re-anchoring (`--anchor-every`) holds the face but jumps the head pose at anchored seams and costs ~6 min per anchor.
The Wan LoRA spike (docs/specs/2026-11-24-long-takes.md, live check 3) showed that a Wan character LoRA trained on the
avatar's 24 photos holds the likeness through 50 s with clean seams, trains in ~54 min and adds ~3 min to a take. The
user wants every new avatar to get one automatically.

The avatar media live check (docs/specs/2026-11-24-avatar-media.md) also showed that photo sets can show two or three
people when an avatar's reference images differ a lot (12 of inbal's 24 photos); any LoRA trained on such a set mixes
faces. That has to be fixed before more LoRAs train on photo sets.

## Approach

1. A one-person check on every avatar photo set (and the clips): the planner model counts the people in each photo;
   photos that don't show exactly one person are dropped, before any dataset, media or LoRA uses them.
2. After the avatar's media is stored, the same `avatar_photoset` job writes a Wan dataset (the kept photos + the kept
   2 s clips, captions pre-filled) and queues a Wan train job; the trainer trains it as today's Wan LoRAs (T2V,
   both experts) and registers it as the avatar's Wan LoRA.
3. Video jobs starring an avatar with a ready Wan LoRA get it attached automatically.

## Design

### One-person check (worker)
- `models.count_people(image_path) -> int`: Qwen2.5-VL (`_chat`) with the image and "How many people are visible in
  this photo? Count partial people too. Answer with a number only."; the first integer in the answer, else -1.
- `_make_photo_set` and `_dataset_photos` (reused Krea photos): after the content filter, photos with a count other
  than 1 are dropped; fewer than `AVATAR_DATASET_MIN` (16) left fails the photo set as today, with the reason ("only N
  of 24 photos show exactly one person; use reference images of just this person"). The result records
  `people_dropped`.
- Clips: the middle still of each clip is counted too; a clip that isn't exactly one person is dropped (as a flagged
  clip).
- The VLM loads once per job (swapped against the edit model as stories do); ~24 short answers, ~1-2 min.

### Wan dataset and training (worker, API, trainer)
- `avatar.json` gains `wan_lora` like `krea_lora`: `{state: queued|photos|training|ready|failed|skipped, lora,
  version, trigger, job_ids, ready: {lora, version}, error?}`; `_write_avatar` preserves it; image edits keep the link.
- LoRA name: `avatar-<name>-wan` when it fits 32 characters, else `avatar-<first 12>-<sha1 6>-wan`
  (`_avatar_wan_lora_name`); trigger = the avatar name.
- `create_avatar` gets `wan_lora` (default true; `--no-wan-lora` / dashboard checkbox to opt out); the photoset job's
  params gain `wan_lora: bool` and the Wan build's `lora`/`version`; `wan_lora.state = queued` with the same job id.
- In the `avatar_photoset` job, after the media is stored (clips may be 0): when `wan_lora` is true, write
  `datasets/<client>/<wan lora>/v<N>/` with `raw/` = the kept photos (`p_NN.png`) and kept clips (`c_N.mp4`), a
  manifest (`kind: character`, `source: avatar`), and `captions.json` pre-filled in the trainer's item names
  (photos `p{i:04d}.png`: "<trigger>, <scene>"; clips `s{i:04d}_00.mp4`: "<trigger>, <motion>, natural light"), then
  conditionally put and send a train job (id derived from the photoset job id, like the Krea one):
  `{model: "t2v", dataset, lora, version, trigger, avatar, avatar_created_at, steps: 1000, speed: "standard",
  kind: "character", ...}`; `wan_lora.state = training`. Media-only jobs (`POST /avatars/{name}/media`) never queue
  it. A Wan dataset/queue failure marks `wan_lora` failed and doesn't fail the job.
- Trainer, Wan path: when params carry `lora`/`version`, the dataset folder stays `params.dataset` but the LoRA is
  uploaded as `loras/<client>/<lora>/v<version>/` (not the dataset name); on success it sets the avatar's
  `wan_lora` ready (`ready = {lora, version}`) while the job is still in `wan_lora.job_ids` (staleness as for Krea);
  failure or cancel sets it failed. No automatic checkpoint previews for avatar builds (saves render-box time).
- On demand: `POST /avatars/{name}/wan-lora` (409 while busy, 404 unknown) queues a Wan build from the avatar's current
  media (its photos and clips are copied into a new dataset version, people check applied; no new photos or clips),
  `vidgen avatar wan-train --name N` and a dashboard "Train Wan LoRA" button. For avatars without media it returns 409
  "make media first".

### Using it (API, video jobs)
- A `video` job whose star (first avatar) has `wan_lora.ready` gets that LoRA first in `params.loras` at weight 1.0
  (pinned files, as `_resolve_loras` does), counted in the 2-Wan-LoRA limit (so one more user LoRA); not when the job
  sets `avatar_lora: false` (`--no-avatar-lora`) or already names that LoRA. The job summary shows it.
- Stories don't get it (shots without the star would get her face): a follow-up.

### CLI / dashboard / docs
- `avatar list` and the dashboard show the Wan LoRA state; RUNBOOK and README updated.

## Acceptance Criteria

- If a photo set has 12 of 24 photos with two people, then the 12 are dropped before the content-filtered set is used,
  and with fewer than 16 one-person photos left the photo set fails with the reason, for the Krea, media and Wan builds.
- When a reused Krea dataset (media-only job) has photos with more than one person, those photos shall not be stored
  as media.
- When an avatar is created (default flags), its photoset job shall queue the Krea train job, store the media, then
  write `datasets/<client>/avatar-<name>-wan/v1/` with the kept photos and clips and queue one Wan train job with
  `lora: avatar-<name>-wan`, `version: 1`, `steps: 1000`; `wan_lora.state` shall be `training`.
- When `wan_lora` is false, or the job is media-only, no Wan dataset or train job shall be written.
- When the trainer finishes an avatar Wan build, the LoRA shall be at `loras/<client>/avatar-<name>-wan/v1/` and the
  avatar's `wan_lora` shall be `ready` with `ready = {lora, version: 1}`, unless a newer build replaced the job.
- When a video job stars an avatar with a ready Wan LoRA, `params.loras[0]` shall be that LoRA at weight 1.0; with
  `avatar_lora: false` it shall not be attached; a user LoRA plus the avatar LoRA is accepted, two user LoRAs plus it
  return 400 naming the limit.
- When `POST /avatars/{name}/wan-lora` is called while a build is queued or training, the API shall return 409; for an
  avatar without media, 409 "make media first".

## Out of Scope

Wan LoRAs in stories; I2V LoRA training; replacing re-anchoring (it stays as an option); retraining existing avatars
automatically (mirabel and inbal get theirs on demand); deleting the spike LoRA `mirabel-wan-spike` (the user decides).

## Risks & Open Questions

- The VLM can miscount (reflections, posters, crowds in the background). (Needs live check on inbal's set: expect ~12
  dropped and a failed set, then a good set from a single clean reference.)
- Clips are Wan's own output: training on them may reinforce Wan artefacts. (Needs live check: compare the new LoRA
  with the photo-only spike on the same 50 s take.)
- Trainer queue: every new avatar adds ~1 h (Wan) after ~1 h (Krea) on the single trainer. (Resolved: acceptable per
  the user.)
- Weight 1.0 may over-stylise some scenes. (Deferred: `--lora avatar-<name>-wan:0.8` overrides via the user LoRA path.)
- Consent: a likeness LoRA can put the person in any video; the RUNBOOK rule (only people who agreed) applies.

## Testing

Unit: `count_people` parsing; photo sets with multi-person photos (dropped, failure below 16, reused datasets
filtered, clip middle frame); the Wan dataset (keys, captions keyed as the trainer names items, manifest) and train
job (params, conditional put, redelivery), skipped when `wan_lora` false or media-only; trainer avatar Wan upload path
and `wan_lora` state (staleness, failure, cancel); API `wan_lora` on create, `POST /wan-lora` (202/404/409), auto
attach and limits on video jobs; CLI/dashboard states. Live: rebuild inbal's photo set (people check), create one new
avatar end to end, and render the 50 s mirabel take with her automatic Wan LoRA.
