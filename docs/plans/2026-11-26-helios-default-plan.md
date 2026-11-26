# Helios as the default video model, g7e only — Implementation Plan

Spec: docs/specs/2026-11-26-helios-default.md
Workspace: current branch (avatar-media-and-wan-lora)
Jira: N/A

## Progress

- [x] Task 1: API accepts and defaults `video_model` — `30430a2`
- [x] Task 2: `worker/helios.py` (environment, weights, generate) — `30430a2`
- [x] Task 3: Worker video jobs route to Helios — `30430a2` (phone look on its frames: `461277c`)
- [x] Task 4: CLI flag and dashboard select — `30430a2` (CLI), `8a3e723` (dashboard, smoke test)
- [x] Task 5: Deployment: g7e only, pins, first boot, IAM, switch order — `8a3e723`
- [x] Task 6: Docs — `8185077`
- [x] Task 7: End-to-end on a g7e — results in the spec; fix found by it: `38ae490`

## Tasks

### Task 1: API accepts and defaults `video_model`

**What**: `VIDEO_MODELS = ("helios", "wan")`; `video` jobs default to `helios` with 640x384, 24 fps and 30 s
when `seconds` isn't sent; `wan` validates as before; Helios refuses `beats`, `plan_only`, `anchor_every` and
more than one Wan LoRA; stories and photos refuse the key; History summary shows the model.
**Files**: `lambda/api/handler.py`, `tests/test_helios_api.py` (new), existing API tests that submit videos
(they now ask for `wan` where they test Wan behaviour)
**Depends on**: none
**Verify**: `python3 -B tests/test_helios_api.py` and `python3 -B tests/test_api.py` (criteria 1-5)

### Task 2: `worker/helios.py` (environment, weights, generate)

**What**: `ensure_env` (venv from `worker/helios_requirements.txt`, Helios repo at the pinned commit),
`ensure_weights` (mirror first, else Hugging Face without `transformer_ode/`, marker, push, `model_download`),
`generate` (release models, run `infer_helios.py` with the tested settings, whole chunks, cut to length,
error with the output's tail). Pins: repo commit, model revision.
**Files**: `worker/helios.py` (new), `worker/helios_requirements.txt` (new), `tests/test_helios_worker.py` (new)
**Depends on**: none
**Verify**: `python3 -B tests/test_helios_worker.py` (criteria 7-9)

### Task 3: Worker video jobs route to Helios

**What**: `render_take` with `video_model == "helios"`: first frame as today at 1280x768, `helios.generate`
with the styled prompt and the Wan LoRA's low-noise file, then grain / upscale / write and sound as today;
no segments, beats, colour matching or frame doubling. Wan unchanged. `mirror_backfill` adds the Helios
folder.
**Files**: `worker/worker.py`, `tests/test_helios_worker.py`
**Depends on**: Task 2
**Verify**: `python3 -B tests/test_helios_worker.py` and `python3 -B tests/test_worker.py` (criteria 6, 10)

### Task 4: CLI flag and dashboard select

**What**: `vidgen video --video-model helios|wan`; `--seconds` sent only when given; `vidgen plan` sends
`wan`. Dashboard: "Video model" select on the Video form, Helios default; with Helios the beats, plan-first,
re-anchor and smooth controls are disabled and the length field shows 30.
**Files**: `cli/vidgen.py`, `cli/ui.html`, `tests/test_helios_api.py` (CLI part)
**Depends on**: Task 1
**Verify**: `python3 -B tests/test_helios_api.py` (criterion 11); the page script passes `node --check`

### Task 5: Deployment: g7e only, pins, first boot, IAM, switch order

**What**: `instance_type` default `g7e.4xlarge` with a g7e-only validation; `root_volume_gb` 750; worker pins
to the validated torch 2.9.0+cu128 set; Krea pins to torch 2.14.1+cu130; first boot prepares the Helios
environment and weights; render role gets `models/helios/*`; switch script order spot 4xlarge, spot 8xlarge,
on-demand 4xlarge, on-demand 8xlarge.
**Files**: `terraform/variables.tf`, `terraform/iam.tf`, `terraform/templates/user_data.sh.tpl`,
`terraform/terraform.tfvars.example`, `worker/requirements.txt`, `worker/krea_requirements.txt`,
`cli/g7e_switch.py`, `tests/test_g7e_switch.py`
**Depends on**: Task 2
**Verify**: `terraform -chdir=terraform validate`; `python3 -B tests/test_g7e_switch.py`; the template renders
and passes `bash -n` (criteria 12, 13)

### Task 6: Docs

**What**: RUNBOOK (video models, what Helios does and doesn't take, sizes, the g7e-only deployment and how
to switch, first boot), README sentence, decision log `docs/adr/helios-default.md`.
**Files**: `RUNBOOK.md`, `README.md`, `docs/adr/helios-default.md` (new)
**Depends on**: Tasks 1-5
**Verify**: every flag, file and setting named exists in the code (`grep`)

### Task 7: End-to-end on a g7e

**What**: a temporary g7e box with the working tree's worker code: the worker's own job path renders an
avatar video with Helios (first frame, her Wan LoRA, sound) and a text-only Helios video; Python 3.10 is tried
for the Helios venv first, 3.11 otherwise. Results appended to the spec.
**Files**: `trainer/spike/validate_helios.py` (new), `docs/specs/2026-11-26-helios-default.md`
**Depends on**: Tasks 2, 3, 5
**Verify**: both clips exist with the expected length and size (criterion 14)
