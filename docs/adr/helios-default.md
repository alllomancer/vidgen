# Helios as the default video model — decision log

Spec: docs/specs/2026-11-26-helios-default.md

## D1 — [2026-11-26] — Task 1: API accepts and defaults `video_model`

**Status**: Accepted
**Decision**: With Helios the API sets 640x384 and 24 fps whatever size and frame rate were sent, maps `draft` to standard, and defaults the length to 30 s only when `seconds` is absent.
**Reason**: Existing CLIs and the dashboard always send their Wan size and frame rate, so "refuse other sizes" would break every existing client; the tests showed other sizes jump at Helios's chunk boundaries.
**Alternatives considered**: Refusing non-640x384 sizes (breaks old clients); a per-client default model (one user).

## D2 — [2026-11-26] — Task 1: API accepts and defaults `video_model`

**Status**: Accepted (its one-LoRA part superseded by D9)
**Decision**: A Helios job with beats, `plan_only`, `anchor_every` or more than one Wan LoRA is refused with the way out named, instead of being switched to Wan or having the extras dropped.
**Reason**: Silently changing the model or ignoring a LoRA gives a result the user didn't ask for and can't explain.
**Alternatives considered**: Falling back to Wan automatically; using only the first LoRA and listing the rest as unused.

## D3 — [2026-11-26] — Task 2: `worker/helios.py`

**Status**: Accepted
**Decision**: The worker runs Helios's own `infer_helios.py` from a pinned checkout in its own venv and process, with exactly the flags that were tested, instead of building the pipeline in our code or using diffusers' Helios pipeline.
**Reason**: That script is what produced the clean clips; its model loading (attention backends, patched norms) is long and changes upstream; a separate process also frees the GPU when it ends.
**Alternatives considered**: Our own runner on their classes (needed for several LoRAs or a LoRA on part of a clip, not needed now); diffusers' `HeliosPyramidPipeline` (not tested by us).

## D4 — [2026-11-26] — Task 2: `worker/helios.py`

**Status**: Accepted
**Decision**: The Helios weights are fetched without `transformer_ode/` and the Helios venv leaves out its training-only packages.
**Reason**: `transformer_ode/` is 57 GB the tested path never loads (a full disk stopped two test boxes); deepspeed and mpi4py need system packages and are only for training.
**Alternatives considered**: The whole repository and requirements file as published.

## D5 — [2026-11-26] — Task 3: Worker video jobs route to Helios

**Status**: Accepted
**Decision**: The first frame for Helios is made at 1280x768 and fitted to 640x384, and Helios's frames go through the worker's own writer, so grain, the phone look and the 2x upscale work as for Wan.
**Reason**: Krea and the edit model are poor at 640x384; one writer keeps the encoder settings and effects in one place.
**Alternatives considered**: Keyframes at 640x384; shipping Helios's own mp4 unchanged.

## D6 — [2026-11-26] — Phone shot option

**Status**: Accepted
**Decision**: The phone shot (`shot: pov`) is made by wording plus a frame effect (`realism.PhoneLook`: slow shake, soft low resolution, grain) on both video models; its wording says the phone and the person filming are never in the picture.
**Reason**: The user wanted footage that looks shot on a phone; wording alone put a phone in the frame when the scene mentioned one, and neither model reliably produces shake or softness from text.
**Alternatives considered**: Wording only (first version, rejected by the user); a camera-shake LoRA.

## D7 — [2026-11-26] — Task 5: Deployment

**Status**: Accepted
**Decision**: The render box's `instance_type` must be a g7e (variable validation), the worker and Krea requirements move to the torch builds validated there, and the switch script no longer stops when the model mirror is empty.
**Reason**: The user wants g7e only; with that check the old g6e box can't be applied to any more, so the mirror can't be filled before the first move and the first boot has to download.
**Alternatives considered**: Keeping g6e possible behind a flag (two torch stacks to keep working); seeding the mirror from the g6e box first (needs a deploy to the box that is being replaced).

## D8 — [2026-11-26] — Task 5: Deployment

**Status**: Accepted
**Decision**: The switch script tries spot `g7e.4xlarge`, spot `g7e.8xlarge`, on-demand `g7e.4xlarge`, on-demand `g7e.8xlarge`.
**Reason**: The user's order: spot first in either size, then on-demand in either size.
**Alternatives considered**: The earlier order (spot 8xlarge, spot 4xlarge, on-demand 4xlarge).

## D9 — [2026-11-26] — After deployment: several LoRAs with Helios

**Status**: Accepted (supersedes the one-LoRA part of D2 and D3)
**Decision**: A Helios job takes all of the job's Wan LoRAs with their weights: one LoRA at weight 1 still runs Helios's script as it is; anything else runs the same script through `worker/helios_run.py`, which patches the pipeline's `set_adapters` to load the others and set every weight.
**Reason**: The user expected several LoRAs (the first live Helios job with an avatar and another LoRA was refused); a patch around the unchanged script keeps the tested path for the common case.
**Alternatives considered**: Our own pipeline runner (more code to keep in step with upstream); keeping the refusal.
