# Avatar media — decision log

## D1 — [2026-11-24] — Task 1: Every new avatar queues the photo-set job, with media state

**Status**: Accepted
**Decision**: Media is made by the existing `avatar_photoset` job (same job id as the Krea build when there is one); with `krea_lora` false, or when the avatar's LoRA name already holds a Wan LoRA, a media-only job is queued instead.
**Reason**: The photo set is already made for the Krea LoRA; making it twice would cost ~18 min of render time per avatar.
**Alternatives considered**: A separate `avatar_media` job type — duplicate photo generation or a cross-job dependency.

## D2 — [2026-11-24] — Task 2: Clips and stored media on the render box

**Status**: Accepted
**Decision**: Photos are stored before the clips render; all four clips render first and their 12 stills are filtered in one `score_images` call; staleness is checked before each clip.
**Reason**: Review: the spec keeps photos when every clip is flagged; each filter call unloads the GPU models, so per-clip filtering reloaded Wan I2V four times; a replaced job should stop using the GPU.
**Alternatives considered**: Filtering per clip (simpler, ~4 extra Wan loads per avatar).

## D3 — [2026-11-24] — Task 3: Media API and media for existing avatars

**Status**: Accepted
**Decision**: A redelivered Krea build whose train job already exists makes its unfinished media from its own dataset's photos; any photo-set failure on the Krea path also fails the media; a cancel is re-raised, never swallowed as a media error.
**Reason**: Review: media could stay in a busy state forever (redelivery after a spot stop, photo-set errors, no trainer), which would also make `POST /avatars/{name}/media` answer 409 forever.
**Alternatives considered**: Marking interrupted media failed and asking the user to retry — the photos are in the dataset, so finishing is cheap.

## D4 — [2026-11-24] — Task 6: Docs, deploy and live check

**Status**: Accepted
**Decision**: A photo-set job cancelled before it (re)starts marks the avatar's Krea build and media records it owns as failed ("cancelled").
**Reason**: Live: inbal's Krea build was cancelled while its message waited for redelivery (the worker restart for the deploy had killed it); the worker's before-start cancel only set the job status, which would have left `krea_lora.state` at "photos" and blocked any rebuild with 409.
**Alternatives considered**: An API-side reset route — the worker already owns these records and their staleness rules.

## D5 — [2026-11-24] — Wan LoRA Task 1: One-person check on photo sets and clips

**Status**: Accepted
**Decision**: Qwen2.5-VL counts people per photo (and per clip middle frame) after the content filter; only exactly-one-person photos are used anywhere; answers are parsed from digits, number words and "single/alone"; unreadable answers count as -1 (dropped).
**Reason**: Live: 12 of inbal's 24 photos showed two or three women (references that look very different); every LoRA trained on such a set mixes faces.
**Alternatives considered**: A face detector (new dependency, misses people seen from behind); one reference image only (loses likeness from the other views).

## D6 — [2026-11-24] — Wan LoRA Tasks 4-5: trainer avatar builds, auto-attach

**Status**: Accepted
**Decision**: Avatar Wan builds upload under `loras/<client>/avatar-<name>-wan/v<version>/` with the version picked by the API (monotonic: replacing an avatar drops only the `ready` link, never the version); the trainer re-checks it is still current after the first expert and before uploading, and a replaced build writes nothing; video jobs get the star's ready Wan LoRA pinned at 1.0, a broken link is skipped, never a 400.
**Reason**: Review: dropping the version on replacement let a still-training build of the previous person collide with the new person's version and overwrite it; a missing LoRA file must not block a user's video.
**Alternatives considered**: Version picked by the trainer (next free) — the avatar would not know which version to link until the upload.

## D7 — [2026-11-25] — The content filter scores on the CPU and no longer unloads the models

**Status**: Accepted (unit-tested; not yet measured on the render box)
**Decision**: `krea.score_images` no longer calls `models.release_all()`; score mode loads the filter on the CPU (`krea_gen.run_score`). Same filter, same threshold, still mandatory. Krea generation is unchanged (it still unloads the models and filters on the GPU).
**Reason**: Each filter call dropped the loaded model (~60 GB, ~5 min to reload) to run a small ViT classifier over 12-24 images. Avatar media paid for it twice: the edit model after the photo set and Wan I2V after the clips. This replaces the reason given in D2; the single `score_images` call for the clips stays (one filter process instead of four).
**Alternatives considered**: Keeping the filter on the GPU beside the loaded model — the VLM sits on the GPU and a pipeline mid-offload can hold most of the 48 GB, so a second process there can fail for lack of memory.

## D8 — [2026-11-25] — Offloaded models keep their RAM copy; smaller speed changes around it

**Status**: Accepted (unit-tested, and checked on a tiny two-expert Wan pipeline on a Mac GPU with diffusers 0.36.0; not yet run or timed on the render box)
**Decision**: The text encoder and transformers of the Wan and edit pipelines keep the RAM copy of their weights while they are on the GPU (`models.keep_ram_weights`, on by default, `KEEP_RAM_WEIGHTS=0` switches it off), so going back to RAM copies nothing. With it: long-take colour matching runs in 8 threads (same pixels); the sound model stays on the GPU across a take's chunks and a story's shots (`models.sound_session`); a story's shots are CRF 10 `veryfast` intermediates when subtitles are burned in (one slow encode instead of two); the next model's files are read into the page cache in the background (`models.warm_ahead`); every pipeline call and video logs where its time went (`pipeline_timing`, `video_timing`); a cancelled or failed pipeline call moves its components back to RAM.
**Reason**: Model offload copied each 28 GB Wan expert off the GPU twice per 5 s segment, and the ~56 GB edit model once per photo, although inference never changes the weights. RAM use stays at the between-runs level that `RESIDENT_LIMIT` is sized for; it no longer dips during a run. The weight keeping lives in a per-module subclass because accelerate deletes a `to` attribute set on the module whenever diffusers re-installs its hooks (after every call). The clean-up after a cancel: the pipeline skips its own, and the expert left on the GPU would not fit beside the next call's.
**Alternatives considered**: Untiled VAE decode (31-35% faster in published tests, but in fp32 it overflows this GPU at 1280x704; a half-precision decode needs a quality check on the box). Group offloading with streams and `torch.compile` (both need GPU measurements, and compiling costs minutes after every model load). A lookup table for the colour conversion (not bit-identical; threads alone gave 6.6x).
