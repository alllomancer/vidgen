# Helios as the default video model, render box on g7e only

Jira: N/A
Date: 2026-11-26
Status: Approved (decisions taken with the user 2026-11-26; details below follow from them)
Last reviewed: 2026-11-26

## Problem

The user wants clips of 30 s and more in one take. Wan 2.2 makes 5 s segments that are chained; Helios
(PKU-YuanGroup/Helios, Apache-2.0) makes a 30 s take in one run in about 3 minutes. Tests on 2026-11-26
(docs/specs/2026-11-26-long-clip-bakeoff.md and the runs after it) showed:

- At **640x384** Helios has no jump at its 33-frame chunk boundaries (1.0-1.3x the normal frame-to-frame
  change); at 512x512, 640x640 and 384x640 it jumps 5-7x. More steps per chunk did not change that.
- From a first frame, with the **low-noise half** of the avatar's Wan LoRA, the person holds over 30 s; without
  the LoRA she drifts; the high-noise half breaks the colours.
- Peak GPU memory ~45 GB at 640x384; the worker's stack runs on a g7e with torch 2.9.0+cu128, and the Krea
  venv with torch 2.14.1+cu130 (docs/specs/2026-11-25-render-on-g7e.md).

## Decisions (user, 2026-11-26)

1. `video` jobs use Helios by default; Wan stays selectable. Stories stay on Wan.
2. The render box is g7e only: spot `g7e.4xlarge`, then spot `g7e.8xlarge`, then on-demand `g7e.4xlarge`,
   then on-demand `g7e.8xlarge`. The trainer stays as it is.
3. Helios clips are delivered at their native 640x384 (upscale stays a per-job option).
4. Built and tested here; the user deploys.

## Design

### Worker: `worker/helios.py` (like `krea.py`: own venv, own process)

- **Environment** (`ensure_env`): `/opt/helios-venv` from `worker/helios_requirements.txt` (torch 2.10.0+cu128
  and Helios's pins), and the Helios repository at a pinned commit (`f1a8aadd…`) in `/opt/helios`. Third-party
  code that runs as root: pinned, bumped only after review. Rebuilt when the requirements file or the pin
  changes.
- **Weights** (`ensure_weights`): `BestWishYsh/Helios-Distilled` at a pinned revision (`b991c037…`) into
  `/opt/models/helios/<rev12>`, without `transformer_ode/` (not used; 57 GB): ~80 GB. From the model mirror
  first (unit `helios/<rev12>`), else Hugging Face, then queued for the mirror; `model_download` log line.
- **`generate(out_path, prompt, seconds, seed, image=None, lora=None, progress)`**: releases the worker's
  loaded models (as Krea does), runs Helios's own `infer_helios.py` in the venv with the settings that tested
  clean: 640x384, 24 fps, distilled 2-2-2 steps, guidance 1, `--is_amplify_first_chunk`,
  `--disable_flash_attention` (its attention library has no build for the g7e GPU), image-to-video when a
  first frame is given (resized to 640x384) and text-to-video otherwise, `--lora_path` when a LoRA file is
  given. Frames = seconds x 24 rounded up to whole chunks of 33; the result is cut to seconds x 24 frames.
  Returns the frames; a failure carries the last lines of Helios's output.
- **One LoRA**: Helios's script takes one. The job's Wan LoRA is used through its **low-noise file**.

### Worker: video jobs (`worker.py`)

- `params["video_model"] == "helios"`: the keyframe (avatar keyframe or Krea first frame) is made as today at
  1280x768 (the same 5:3 shape) and handed to Helios; without one, Helios runs text-to-video. The prompt gets
  the same realism / shot wording and natural-look sentences as Wan. Grain, upscale (to 1280x768) and sound
  work on the result as they do today; frame-rate doubling, colour matching, segments, beats, planning and
  face re-anchoring do not apply (one take, one prompt). Variations run Helios once per seed.
- Wan (`video_model == "wan"`) is unchanged.
- `mirror_backfill` also queues the Helios weights folder.

### API (`lambda/api/handler.py`)

- `video_model`: `helios` (default) or `wan`, on `video` jobs. Stories and photos refuse the key.
- With `helios`: size is 640x384 and fps 24 whatever was sent (the CLI always sends its defaults); `beats`,
  `plan_only` and `anchor_every` are refused with "use video model wan"; more than one Wan LoRA (the avatar's
  own counts) is refused with the choice named (`avatar_lora: false`, fewer LoRAs, or Wan); `seconds` defaults
  to 30 when not sent.
- History shows the model.

### CLI and dashboard

- `vidgen video --video-model helios|wan` (default: the server's, Helios). `--seconds` defaults to 30 with
  Helios, 5 with Wan. `vidgen plan` is Wan only (it writes beats).
- Dashboard Video form: a "Video model" select (Helios default / Wan). With Helios the beats, plan-first,
  re-anchor and smooth-frame-rate controls are disabled with a hint, and the length defaults to 30 s.

### Deployment (Terraform, first boot, switch script)

- `variables.tf`: `instance_type` defaults to `g7e.4xlarge` and must be a g7e type; `root_volume_gb` defaults
  to 750 (Helios adds ~80 GB; a first boot without the mirror holds two fp32 Wan downloads at once).
- Worker pins: `worker/requirements.txt` moves to the validated torch 2.9.0+cu128 set and
  `worker/krea_requirements.txt` to torch 2.14.1+cu130 (both passed on a g7e, 2026-11-26).
- First boot (`user_data.sh.tpl`): after the worker venv, the Helios environment and weights are prepared
  (same functions the worker uses), so the first job doesn't wait for them.
- IAM: the render box role may read/write `models/helios/*`.
- `cli/g7e_switch.py`: candidates in the user's order (spot 4xlarge, spot 8xlarge, on-demand 4xlarge,
  on-demand 8xlarge).

## Acceptance Criteria

API:

1. When a `video` job has no `video_model`, the system shall set `video_model` to `helios`, width 640, height
   384, fps 24, and seconds 30 unless `seconds` was sent.
2. When `video_model` is `wan`, the system shall validate the job exactly as before this change.
3. If `video_model` is anything else, or is sent on a story or photos job, then the system shall refuse it.
4. If a Helios job has `beats`, `plan_only` or `anchor_every`, then the system shall refuse it naming Wan.
5. If a Helios job would carry more than one Wan LoRA, then the system shall refuse it naming the options.

Worker:

6. When a Helios job runs, `helios.generate` shall be called once per take with the styled prompt, the first
   frame when there is one, the low-noise file of the job's Wan LoRA when there is one, and the job's seconds
   and seed; Wan's `generate_video` shall not be called.
7. `helios.generate` shall release the worker's loaded models first, ask for a whole number of 33-frame
   chunks covering seconds x 24, and return exactly seconds x 24 frames.
8. If Helios exits with an error, then `helios.generate` shall raise with the last lines of its output.
9. `helios.ensure_env` shall build the venv and check the repository out at the pinned commit only when the
   requirements file or the pin changed; `helios.ensure_weights` shall take the weights from the mirror when
   it has them, else Hugging Face (without `transformer_ode/`), then queue them for the mirror.
10. When `video_model` is `wan`, the worker shall render as before (existing tests pass unchanged).

CLI / deployment:

11. `vidgen video` shall send `video_model` only when `--video-model` is given, and `seconds` only when
    `--seconds` is given.
12. `terraform validate` shall pass; a non-g7e `instance_type` shall be refused by the variable's validation.
13. `g7e_switch.py` shall try spot 4xlarge, spot 8xlarge, on-demand 4xlarge, on-demand 8xlarge, in that order.
14. On a g7e, a Helios video of an avatar with its Wan LoRA shall render end to end through the worker's own
    code (first frame, Helios, sound), and a text-only Helios video shall render too.

## Out of Scope

- Stories on Helios; several avatars in a Helios take; per-segment prompts for Helios.
- LoRAs for part of a clip; LoRAs trained for Helios; the high-noise half.
- Sizes other than 640x384 (portrait and square jump).
- Moving the trainer to g7e. Deploying (the user runs `make g7e`).
- Helios's compile option and its Mid/Base checkpoints (only measured, not wired in, unless the test now
  running shows a clear gain without side effects).

## Risks & Open Questions

- **Python**: Helios asks for Python 3.11; the box has 3.10. Tests used 3.11 from the deadsnakes PPA. Open:
  whether 3.10 works; if not, first boot adds that PPA (a third-party package source on the box).
- **Text-to-video with Helios** was not run by us yet (criterion 14 covers it).
- **Helios keeps one person from the first frame; with two people in it one was dropped** (bake-off). Avatar
  videos with 2-3 avatars should use Wan; the API does not force that. (Accepted; documented.)
- **Upgrading the worker's torch** was validated for T2V, I2V, the edit model, MMAudio, the VLM and Krea on a
  g7e, not for RIFE/upscale/voice/captions or on the L40S; `make smoke` after deploying covers the rest.
- **g7e capacity**: on 2026-11-26 spot `g7e.4xlarge` never had capacity; on-demand 4xlarge usually did, once
  not; spot 8xlarge once did. The switch script stops without changing anything when nothing launches.
- **Cost**: on-demand g7e.4xlarge is ~$4.00/h against ~$1.80/h for the spot g6e today.
- **Licenses**: Helios code and weights are Apache-2.0. The LoRA reuse is of the user's own Wan LoRAs.
- Feature flag: `video_model: wan` per job; `HELIOS_DEFAULT`-style switches are not added (the API default
  is one constant).

## Testing

- `tests/test_helios_api.py`: criteria 1-5 against the handler.
- `tests/test_helios_worker.py`: criteria 6-10 with a stubbed subprocess and stubbed models.
- `tests/test_g7e_switch.py`: the new order (criterion 13); CLI flags (11).
- `terraform validate` (12).
- A temporary g7e box running the worker's own job code for criterion 14; results appended here.

## End-to-end results (2026-11-26, criterion 14: met)

On-demand `g7e.4xlarge`, the working tree's worker code, the worker venv built from `worker/requirements.txt`
as committed, `trainer/spike/validate_helios.py` calling the worker's own `run_video` (progress and uploads
kept local).

| Job | Result | Whole job | Helios render | Output |
|---|---|---|---|---|
| Liron dancing, phone look (`shot: pov`), 30 s, her Wan LoRA | ok | 1182 s, of which ~14 min were first-use downloads (Krea 36 GB, Helios 80 GB) | 175 s, 45.1 GB peak | 30.0 s, 640x384, 24 fps, AAC sound; chunk-boundary change 1.1x normal |
| Liron as a streaming session (`shot: stream`), 15 s | ok | 202 s | - | 15.0 s, 640x384, 24 fps, sound; 2.2x (a nearly still scene) |
| Text only (no avatar, no first frame), 10 s | ok | 90 s | - | 10.0 s, 640x384, 24 fps, sound; 1.1x |

- The path that ran: Krea first frame with the avatar's Krea LoRA (5 s), fitted to 640x384, Helios
  image-to-video with the low-noise file of her Wan LoRA, the phone look on the frames, MMAudio sound. The
  text-only job ran Helios text-to-video.
- **Python**: the image has `/usr/bin/python3.11`, and the Helios venv used it. No extra package source is
  needed. (Whether Helios runs on 3.10 was therefore not tested.)
- An earlier attempt failed at Helios's first line: the trimmed requirements lacked `pandas`. Fixed, and
  `ensure_env` now starts Helios's script once (`--help`) before marking the venv ready.
- The streaming wording put a **visible ring light** in the picture (a bright circle around her). Whether
  that is wanted is the user's call; the wording can say the light is off-camera.
- Earlier the same day a spot `g7e.8xlarge` running this test was reclaimed by AWS after five minutes.
