# Generated sound for videos

Jira: N/A
Date: 2026-11-07
Status: Approved
Last reviewed: 2026-11-07

## Problem

`vidgen video` clips are silent: Wan2.2-TI2V-5B renders frames only. `vidgen story` has
Kokoro narration but no ambient sound or effects, so stories feel empty between lines.

## Approach

Generate a sound track from each rendered clip with MMAudio (`large_44k_v2`), a
video-to-audio model that produces ambient sound and Foley synchronised to the frames and
steered by the clip's text prompt. Sound is on by default; `--no-sound` turns it off.

Licence: MMAudio weights are **CC-BY-NC 4.0** (checked on the Hugging Face model card,
2026-11-07). Videos with generated sound are for personal, non-commercial use only.

## Design

- **Model** (`worker/models.py`): `_load_audio()` loads MMAudio `large_44k_v2` in bf16 with
  its feature utils (CLIP, Synchformer, VAE). Checkpoint paths are pointed at
  `/opt/models/mmaudio/` and downloaded there on first use (`download_if_needed`). It goes
  through `_swap_to("audio", …)`, so it is the only model on the GPU while it runs.
  `generate_sound(video_path, out_path, prompt, seed)` writes a 44.1 kHz WAV covering the
  clip's duration.
- **Video jobs** (`worker/worker.py run_video`): after rendering, unless `sound` is false,
  generate sound and mux it into the MP4 (video stream copied, AAC 160 kb/s).
- **Story jobs** (`run_story`): keyframes → clips → one sound pass over all shots (one extra
  model swap per story) → concatenate the shot sounds. With narration, mix narration at full
  volume over sound at 30 % (`volume=0.3`, `amix … normalize=0`); without narration, use the
  sound alone. The sound track is uploaded as `sound.wav`.
- **API** (`lambda/api/handler.py`): video and story jobs accept `sound` (bool, default true).
- **CLI** (`cli/vidgen.py`): `--no-sound` on `video` and `story`.
- **Install**: `worker/requirements.txt` adds `torchaudio==2.5.1`, `torchcodec==0.1.1`,
  `tensordict==0.6.2` (pinned to match torch 2.5.1) and MMAudio from GitHub pinned to commit
  `974010a026c731054592d8f777218bd9d85a6c24`. The worker re-runs `pip install` on every
  start. `terraform/templates/user_data.sh.tpl` pre-downloads the weights on a fresh
  instance.
- **Smoke test** (`cli/smoke.py`): `video`, `avatar-vid` and `story` steps expect an audio
  stream (existing `expect_audio` check; skipped when `ffprobe` is not installed locally).

## Acceptance Criteria

- When a video job is submitted without `sound: false`, the system shall return an MP4 with
  one AAC audio stream whose duration is within 0.5 s of the video stream's.
- When a video job is submitted with `sound: false` (`--no-sound`), the system shall return
  an MP4 with no audio stream.
- When a story job is submitted with narration and sound, the system shall return a
  `story.mp4` with one audio stream containing narration mixed over sound, plus `sound.wav`
  and `narration.wav`.
- When a story job is submitted with `--no-narration` and sound on, the system shall return a
  `story.mp4` whose audio is the generated sound.
- When a story job is submitted with `--no-sound --no-narration`, the system shall return a
  silent `story.mp4`, as today.
- If `sound` is not a boolean, then the API shall treat it with Python truthiness, matching
  the existing `subtitles` and `narration` fields.
- The system shall load MMAudio at most once per story job.
- `make smoke` shall pass all steps, with the `video`, `avatar-vid` and `story` steps
  reporting an audio stream.

## Out of Scope

Background music, narration on single clips, user-supplied audio, sound for `describe` and
`avatar` jobs, per-shot volume control.

## Risks & Open Questions

- **Licence is non-commercial** (CC-BY-NC 4.0). Resolved: accepted by the user.
- **Dependency compatibility** with torch 2.5.1 / transformers 4.55.4 (MMAudio wants
  `torchcodec`, `tensordict`, `numpy<2.1`). Needs spike: `pip install --dry-run` on the
  instance must show torch unchanged before deploy.
- **Clips over 8 s**: MMAudio is trained on 8 s; 9–10 s clips may lose coherence at the end.
  Deferred: accept, revisit if it sounds wrong.
- **First-use download** of ~4 GB of weights from Hugging Face and GitHub releases on the
  existing instance adds a few minutes to the first sound job. Deferred: one-off.
- **`aws s3 sync` skips same-size files**: a future edit that keeps a file's size won't sync.
  Deferred: worked around manually; not part of this change.

## Testing

No unit-test harness exists in this project; verification is end-to-end on the real GPU:
`pip install --dry-run` spike, then `make smoke` (audio-stream checks), then a manual listen
to the smoke outputs.
