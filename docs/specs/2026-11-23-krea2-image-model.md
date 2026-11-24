# Krea 2 Turbo as a second image model

Jira: N/A
Date: 2026-11-23
Status: Approved
Last reviewed: 2026-11-23

## Problem

Images that need no particular person (avatar candidates, photos of places/things, the first
frame of videos without an avatar) come from models made for something else: a single Wan video
frame, or nothing at all (photos always need a person). The user asked to be able to choose
Krea 2 (open weights, June 2026) in the CLI and dashboard.

## Approach

Choices made 2026-11-23:

- Krea 2 **Turbo** (8 steps, CFG off), text-to-image only, offered where no person has to be kept:
  avatar candidates (`--model krea2`), photos without an avatar (`photos --no-avatar`), and the
  first frame of videos / story shots without avatars (`--image-model krea2`, then Wan
  image-to-video animates it). Avatar photos, series and keyframes stay on Qwen-Image-Edit; video
  stays on Wan.
- Its own venv on the render box (`/opt/krea-venv`), so the Wan/Qwen pipelines keep their tested
  diffusers 0.37.1 / transformers 4.55.4.
- The content filter the Krea 2 license requires (section 4.2) runs at generation on every Krea
  image, with no switch to turn it off (the user's own downstream filter can add to it). Timings
  for generation and the filter are recorded per job so its cost is visible.

## Design

- `worker/krea_gen.py` (runs in the Krea venv): prefetch the weights (8 readers), load
  `Krea2Pipeline` bf16 on the GPU (CPU offload if it does not fit), load
  `Falconsai/nsfw_image_detection` (Apache-2.0, pinned revision). Per image: generate, classify;
  nsfw score >= 0.5 -> re-roll once with seed + 1000003; flagged again -> not written, `filtered`.
  Writes per-image seed, score, re-rolled, generate/filter seconds.
- `worker/krea.py` (worker venv): first use creates the venv from `worker/krea_requirements.txt`
  (hash marker, so a changed file reinstalls) and downloads the pinned revision (diffusers folders
  only, ~36 GB; not the 26 GB single-file duplicate) with the HF token from SSM SecureString
  `/ai-video/hf_token`. `generate(items)` drops the worker's loaded models (`release_all`), runs
  the generator as a subprocess (cancellable between images), returns images (None = filtered) and
  stats (load, generate, filter seconds, filtered, re-rolled). Stats go into job results and the log.
- Jobs: avatar `model: wan|krea2` (896x1152 candidates; flagged ones dropped; all flagged -> job
  fails with the reason); photos `model: qwen|krea2` (`avatar: none` implies krea2; scenes as plain
  text + realism sentences + grain; no series, no `best`); video/story `image_model: wan|krea2`
  (video: only without an avatar; story: for the shots without avatars). A flagged first frame
  fails the job with the filter's reason.
- Pins: `krea/Krea-2-Turbo@98e0fe11`, filter `@96cb0d03`, venv torch 2.14.1+cu126, diffusers
  0.41.0, transformers 5.20.2, accelerate 1.15.0, safetensors 0.8.0, huggingface_hub 1.33.0 (the box
  has Python 3.10; diffusers 0.42+ / transformers 5.21+ need 3.11). Installed and imported on the
  box 2026-11-23 (6.6 GB venv, 83 s install). Status / Models card show the Krea repo, revision, filter.
- IAM: the worker may `ssm:GetParameter` that one parameter. The token is created by the user
  with the AWS CLI (never in Terraform state or this repo).
- CLI: `avatar refresh --describe … --model krea2`, `photos --no-avatar` (or `--model krea2`),
  `video|story --image-model krea2`. Dashboard: Photos avatar "no person: Krea 2", "First frame
  without an avatar" in the Video/Story Realism section, Avatar > Generate candidates (name,
  model, description, count) and **Pick N** in History.

## Acceptance Criteria

- When `avatar refresh --describe D --model krea2 --candidates 4` runs, Krea 2 shall render 4
  candidates (seeds base..base+3) and every image shall have been checked by the filter.
- If the filter flags an image twice, then it shall not be written or uploaded, and the job result
  shall count it as filtered; if all images of a job are flagged, the job shall fail with that reason.
- When `photos --no-avatar --prompt P` runs, the prompt shall contain no identity sentence and the
  photo shall come from Krea 2.
- If `photos --model krea2` names an avatar or upload, uses `--series` or `--best`, then the API
  shall return 400; if a video has an avatar and `image_model: krea2`, the API shall return 400.
- When `video --no-avatar --image-model krea2 --variations 2` runs, each take shall start from its
  own Krea 2 frame (one Krea run for both).
- Without these options, every job shall render exactly as before.
- Every Krea job result shall include load, generate and filter seconds.

## Out of Scope

Krea 2 Raw (fine-tuning base), Krea LoRAs / training, editing or reference images, a filter on the
Wan/Qwen outputs, turning the filter off.

## Risks & Open Questions

- Package versions with the box's driver and Python. (Needs spike: install + one image.)
- Memory: Krea (~36 GB) runs after the worker drops its models; the first job after a Krea job
  reloads Wan/Qwen. (Accepted; measured live.)
- License: commercial use only under $1M annual revenue (section 2.3); this runs on an employer
  AWS account. The user is responsible for licensing; noted in the RUNBOOK.
- Filter false positives (e.g. swimwear) drop images; threshold 0.5. (Measured live.)

## Testing

Unit tests: generator filter logic (pass, re-roll, drop), `krea.generate` results and stats,
avatar/photos/video wiring, API rules, CLI bodies, dashboard pick. Live: box install spike, one
avatar-candidates job, one no-avatar photo, one no-avatar video with a Krea first frame.

## Update: live checks (2026-11-23)

- Box install test: Python 3.10 on the box, so diffusers 0.41.0 / transformers 5.20.2 (pins updated
  to the box's `pip freeze`); torch 2.14.1+cu126 sees the L40S (driver 595). Venv 6.6 GB, 83 s.
- Avatar candidates `--model krea2 --candidates 4`: 128 s for the whole job including the first
  ~36 GB download; Krea load 12.6 s, 4 images 29.9 s (~7.5 s each at 896x1152), filter 0.1 s in
  total (~25 ms per image), none flagged. Sharp, studio-lit, true to the description; the 4 are
  very similar (low seed variety of the distilled model).
- Photos `--no-avatar` (a harbour prompt, a Rome location, film look, grain): both photorealistic,
  no person added, film colours.
- Video `--no-avatar --image-model krea2 --seconds 3 --fps 16`: Krea 2 first frame (black sand
  beach at sunset), Wan I2V animated it; 49 frames, 1280x704, keyframe saved next to the clip.
