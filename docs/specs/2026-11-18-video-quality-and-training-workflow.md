# Video quality, longer clips, variations, dashboard jobs, training workflow

Jira: N/A
Date: 2026-11-18
Status: Approved
Last reviewed: 2026-11-18

## Problem

Clips are 16 fps, 1280x704 and at most 5 s; choosing between takes means separate jobs; the
dashboard cannot submit work; and LoRA training is slow (~12.5 h for 1250 steps per expert),
captions are never reviewed, and checkpoint previews have to be queued by hand.

## Approach

Seven pieces in two phases (choices made 2026-11-18):

- Render side: (6) RIFE frame interpolation to 32 fps on by default, 2x upscale opt-in;
  (7) single takes up to 15 s by chaining 5 s segments; (8) `--variations`; (11) dashboard
  Create tab.
- Trainer side: (4) the trainer queues `lora_preview` itself when training finishes;
  (9) caption review before training; (10) `--speed fast` preset, default set from a timed run.

## Design

### 6. Smooth and sharp (worker post-processing)

- After Wan returns frames and before writing the mp4: RIFE doubles 81 -> 161 frames, written at
  32 fps, same duration. Weights `rife49.pth` (RIFE 4.9, MIT) from the ComfyUI-Frame-Interpolation
  GitHub release, sha256-pinned; the IFNet code is vendored from that project (MIT, based on
  Practical-RIFE) as `worker/rife_arch.py`. (Changed from 4.25: its architecture code only ships
  with the Google-Drive weights; 4.9 has a clean MIT source for both.)
- `--upscale`: Real-ESRGAN x2plus (BSD-3) via `spandrel` (MIT, no basicsr), 1280x704 ->
  2560x1408, frame by frame; weights from the official v0.2.1 release, sha256-pinned. Runs after RIFE.
- Params `fps` (16 | 32, default 32) and `upscale` (default false) on `video` and `story`;
  CLI `--fps 16`, `--upscale`. MMAudio sound is generated from the 16 fps clip (it only needs the
  motion) and muxed onto the final file.
- Spike first on the render box: install, timings per clip, peak memory. Expected ~10-20 s for
  RIFE, ~30 s for the upscale; if RIFE cannot be installed cleanly, fall back to ffmpeg
  `minterpolate` (worse, but no model).

### 7. Longer single takes (`video --seconds` up to 15)

- `ceil(seconds / 5)` segments of 81 frames. Segment 1 as today (T2V, or I2V from the avatar
  keyframe); each next segment is I2V from the previous segment's last frame with the same
  prompt and seed + n. Frames are joined dropping each segment's first frame (it repeats the
  previous last frame); the last segment is trimmed to the requested length.
- Sound: one MMAudio pass per segment, joined with a 0.2 s crossfade.
- `story` keeps 2-5 s per shot (shots already give length).

### 8. Variations (`video --variations 2-4`)

- N independent renders with seeds `seed, seed+1, ...` (random base when no seed); with an avatar
  each gets its own keyframe. Models load once. Result `video_1..N`; the CLI saves
  `<out>_1.mp4 ... <out>_N.mp4`. Not combined with `--seconds` > 5 (N x 3 segments is too long
  for one job); the API returns 400.

### 11. Dashboard Create tab

- New tab with three forms that submit through the existing signed CLI path (`ui.py` ->
  `vidgen.submit`, so the watcher downloads results):
  - Video: prompt, seconds 1-15, avatar (default / none / name), LoRA + weight (+ checkpoint),
    variations, 32 fps, upscale, sound.
  - Photos: avatar, where / wearing / doing / camera or a free prompt, count, aspect.
  - Story: text, shots, seconds per shot, style, voice, avatar, LoRA, subtitles, narration.
- Avatars come from `GET /avatars`, LoRAs from `GET /loras`.

### 4. Auto-preview after training

- Last trainer step: write a `lora_preview` job (same params the API builds: last 5 checkpoints,
  neutral scene + trigger word, weight 1.0, 2 s, 832x480, random seed) into the jobs table and
  the render queue; the waker starts the render box within a minute. The train result gets
  `preview_job_id`; the CLI watcher registers that job when it sees the training finish, so the
  clips download to `lora-preview-<id>/`.
- The trainer reads the render queue URL from `code/trainer/render.json` (written by Terraform;
  the trainer env file is only written at first boot). IAM: trainer gets `sqs:SendMessage` on
  the render queue and `dynamodb:PutItem` on the jobs table.
- An active uptime hold on the render box delays the preview until `vidgen wake` (unchanged).

### 9. Caption review

- Every train job uploads its captions after captioning:
  `datasets/<client>/<name>/captions.json` `{fingerprint, items: {segment: caption}, edited_at}`.
- `vidgen train NAME --captions-only` runs prepare + caption and stops (minutes, not hours).
- `vidgen dataset captions NAME [--edit]`: prints the captions, or writes them to a text file
  (`segment<TAB>caption`, one per line), opens `$EDITOR`, and uploads the result.
  API `GET` / `PUT /datasets/{name}/captions` (the PUT checks every segment exists, captions
  1-500 chars).
- Training writes stored captions as the segments' `.txt` when the dataset fingerprint matches,
  and captions only the rest (caption.py already skips existing `.txt`). A changed dataset
  re-captions and warns that edits were for an older version.

### 10. Faster training

- `vidgen train NAME --speed fast|standard`: fast trains at 512x288 (standard 832x480), both
  33 frames. Recorded in `lora.json` (`resolution`); `--continue` keeps the theme's resolution.
- A timed run (50 steps of each, one expert) sets the default and the time estimates in the
  CLI and dashboard; previews compare quality.

## Acceptance Criteria

- When a 5 s video renders with default settings, the system shall write 161 frames at 32 fps.
- When `--fps 16` is given, the system shall write 81 frames at 16 fps exactly as before.
- When `--upscale` is given, the output shall be 2560x1408.
- When `video --seconds 12` runs, the result shall be one 12 s clip made of 3 segments, each
  after the first starting from the previous segment's last frame.
- If `--seconds` is above 15 (video) or `--variations` is outside 2-4 or combined with
  `--seconds` > 5, then the API shall return 400.
- When `--variations 3 --seed 10` runs, the result shall hold 3 clips with seeds 10, 11, 12.
- When the Create tab submits a video, the job shall appear in History and download like a CLI job.
- When a train job finishes, a `lora_preview` job for its last 5 checkpoints shall be queued
  without any client action, and its id shall be in the train result.
- When `train NAME --captions-only` finishes, `captions.json` shall hold one caption per segment
  and no LoRA version shall be created.
- When captions were edited for the current dataset fingerprint, the next training shall use
  them unchanged.
- If an edited captions file names a segment that does not exist, then the API shall return 400.
- When `train NAME --speed fast` runs, musubi-tuner shall train at 512x288 and `lora.json`
  shall record it.

## Out of Scope

Image-to-video LoRA training, photo-to-video, saved outfits, a content guard (deferred real-person
pieces); longer `story` shots; H100 trainer; editing captions in the dashboard.

## Risks & Open Questions

- RIFE / spandrel installation on the render box venv (torch version). (Needs spike.)
- Drift across chained segments (faces, colours). (Accepted at 15 s; check in the live test.)
- MMAudio seams between segments. (Crossfade; check in the live test.)
- Deterministic segment names between caption-only and training runs of the same dataset.
  (Check in the trainer's prepare step; fingerprint guards it.)
- Fast preset quality for character themes (detail at 512x288). (Measured by the timed run and
  previews; the default may differ by kind.)
- No feature flag; `--fps 16` restores today's video output.

## Testing

Unit tests in `tests/` (`make test`; the earlier scratchpad tests were lost to a temp-folder cleanup); render-box spike
for RIFE/upscale; live: one default clip, one `--upscale`, one 12 s clip, one `--variations 2`,
a Create-tab submit, a `--captions-only` run, and the timed training run.

## Update: render time (2026-11-18)

Profile on the L40S (5 s clip, 1280x704, Lightning 4 steps, warm): 239 s = 4 denoising steps of
~40 s, ~32 s moving the two experts between CPU and GPU (they do not fit the 48 GB card together),
48 s fp32 VAE decode. RIFE to 32 fps: 3 s. Upscale: 0.23 s per 720p frame. Sound: 4 s warm.
Cold first clip: 456 s, the extra ~3.6 min being the experts paged in from the gp3 disk by one
reader at 315 MB/s (volume and instance allow 1000 MB/s).

- **Prefetch** (built): before loading a Wan model the worker reads its weight files with 8
  parallel readers into the page cache: 69 GB in 68 s (1014 MB/s). Cold start 469 s -> 329 s; the
  first clip then runs at warm speed. Local NVMe was rejected: it is wiped on every stop, so it
  would be filled from the same disk at each boot; the 124 GB of RAM caches the files instead.
- **Draft quality** (built): `--draft` / `quality: draft` renders at 832x480, then RIFE + 2x upscale
  to 1664x960: 126 s vs 242 s warm (1.9x). A different size gives a different scene for the same
  seed, so a draft checks a prompt but cannot be re-rendered as the same take at full quality.
- Not done (candidates): bf16 VAE decode (48 s), FP8 experts to avoid the CPU/GPU moves (~32 s).

## Update: live checks (2026-11-18)

- Video, all passed: default 5 s at 32 fps (248 s warm), `--upscale` 2560x1408, 12 s take (3 chained
  segments, 716 s, no visible joins), `--variations 2` (583 s), draft from the dashboard (136 s,
  1664x960). Each has the expected size, fps, length and sound.
- Captions-only: 241 captions in ~6 min, each with the trigger; edit, unknown-item rejection and
  restore checked through the API. The next training reused them (no captioning model load).
- Training speed (L40S, jerk dataset, 33 frames): standard 832x480 17.6 s/step steady, fast 512x288
  6.0 s/step (2.9x). Latent caching: ~15 min standard, ~7 min fast. A 1250-step training: ~12.5 h
  standard, ~4.5 h fast. Default stays standard until fast previews are judged.
- Auto-preview: a 100-step run has no checkpoint (saved every 250 steps), so the first test queued
  nothing; fixed by previewing the final weights when the final step has no checkpoint. Retest: the
  trainer queued the preview job itself and the watcher downloaded it.
