# LoRA training from your own videos and photos

Jira: N/A
Date: 2026-11-09
Status: Draft (spike: go)
Last reviewed: 2026-11-09

## Problem

Reference images (avatars, `--image`) give a person's look only approximately: identity
drifts across shots, and there is no way to teach the pipeline a style or a kind of
motion. Users want to supply their own videos (and photos) and have the generators learn
them, so `video`, `story` and `photos` reproduce that person, style or motion faithfully.

Full fine-tuning of 14B-parameter models is out of reach; **LoRA adapters** (small,
~100-600 MB files trained on top of the frozen base model) are the standard way and
stack with the Lightning LoRAs we already use.

## Approach

A separate, on-demand **training machine**, never the render box, so training never
blocks or slows rendering and can use a different GPU. It follows the same pattern as
the render box: queue-driven, spot, private subnet, stops itself when idle.

1. `vidgen dataset` uploads clips and photos to S3.
2. `vidgen train` queues a training job on a separate `train` queue.
3. The trainer box captions the data, trains the LoRA and uploads the `.safetensors` to S3.
4. `video` / `story` / `photos` jobs take `--lora NAME[:weight]`; the render worker loads it
   next to Lightning.

Trainer software: **musubi-tuner** (kohya-ss), pinned to a commit. It trains Wan2.2
T2V/I2V-A14B LoRAs per expert (high-noise and low-noise), and Qwen-Image / Qwen-Image-Edit
LoRAs, with fp8 base weights, latent/text-encoder caching and block swapping for smaller
GPUs. Alternatives are listed under Risks.

## Design

### What can be trained (LoRA "kind")

| Kind | Base model(s) | Data | Used by |
|---|---|---|---|
| `character` | Wan2.2 T2V-A14B and I2V-A14B (both experts each) | 10-40 clips of 2-5 s of one person + 20-50 photos, varied angles, light, clothes | `video`, `story` (with or without avatar keyframe) |
| `style` | Wan2.2 T2V-A14B | 30-100 clips in the target look | `video`, `story` |
| `motion` | Wan2.2 T2V-A14B and I2V-A14B | 20-60 clips showing the motion | `video`, `story` |
| `photo-character` (phase 2) | Qwen-Image-Edit-2509 | 20-50 photos of one person | `photos`, avatar keyframes |

**Themes.** Every dataset is one theme and trains one LoRA (one concept per LoRA trains
faster and more reliably, and themes can be retrained, deleted and mixed independently).
Themes are named by the user and grouped by kind in `vidgen lora list`. Several concepts
in one LoRA (each with its own trigger word) work but bleed into each other, so they are
not offered.

**How much data per theme** (guidance printed by `vidgen dataset show`; fewer than 5
usable items fails the job):

| Kind | Clips (2-5 s) | Extra | What makes it work |
|---|---|---|---|
| character | 10-40 | 20-50 photos | varied angles, light, clothes, backgrounds; the person clearly visible |
| style | 30-100 | - | many different subjects, all in the same look |
| motion | 20-60 | - | different people/places doing the same motion |
| place / object | 15-40 | photos | different times of day and angles |

Variety beats quantity: near-duplicate clips cause overfitting (the LoRA copies one
scene). No watermarks, text overlays or cuts inside a clip; longer videos are cut into
segments automatically.

Each LoRA gets a **trigger word** (default: `vgn_<name>`), which captions and prompts use
to call it up.

### Training machine

- **Terraform**: a new `training.tf`, off by default (`training_enabled = false`). It adds
  a second spot instance with stop on interruption, in the same VPC/private subnet found by
  tags, with SSM, no public IP, the same bucket, a `train` SQS queue (+ DLQ) and an
  instance role limited to `datasets/*`, `loras/*` and the train queue.
- **Instance**: default `p5.4xlarge` (1x H100 80 GB) if spot capacity and the org SCPs
  allow it in us-east-1; fallback `g6e.4xlarge` (L40S 48 GB, the render type) using fp8
  base weights + block swap at about 2-3x the time. The type is a variable; check
  availability with the same spot placement score command as the runbook.
- **Disk**: 500 GB gp3 at 1000 MB/s (base weights in fp8/bf16 + latent cache).
- **Waker**: the existing waker Lambda also starts the trainer when the train queue is
  non-empty; the trainer stops itself after 15 min idle (same code path as the render
  worker, separate systemd unit `ai-video-trainer`).
- **"Another machine" without AWS**: the trainer is a single script
  (`trainer/train.py --dataset DIR --kind character --name NAME`) that also runs on any
  Linux box with a 48 GB+ NVIDIA GPU (a workstation, RunPod, Lambda Labs); the user then
  uploads the result with `vidgen lora add NAME file_high.safetensors file_low.safetensors`.

### Datasets: upload (step 1, built 2026-11-10)

- **Storage**: `datasets/<client>/<name>/raw/<relative path>` (private bucket, SSE) plus
  `datasets/<client>/<name>/manifest.json`: `{name, kind, trigger, files: [{path, size,
  mtime}], file_count, total_bytes, created_at, updated_at, consent_at}`. The trainer trains
  from exactly the manifest's file list.
- **Upload is a sync** of one local folder: only files that are new or changed (path, size,
  mtime) are uploaded; files no longer in the folder are deleted from the dataset after the
  user confirms the list. Files go straight from the user's computer to S3 with presigned
  PUT URLs (one per file, 1 h); the API never proxies file bytes.
- **API**: `POST /datasets/{name}/uploads` `{files: [{path, size}]}` (max 100 per call) ->
  presigned PUT per file; `PUT /datasets/{name}` `{kind, trigger, consent, files}` checks every
  listed file is in S3 with that size, deletes objects not listed, writes the manifest;
  `GET /datasets`, `GET /datasets/{name}`, `DELETE /datasets/{name}` (all objects).
  Names: 1-32 chars `a-z 0-9 - _`. Paths: relative, no `..`, no leading `/` or `\`, no control
  characters, <= 512 chars; extensions mp4, mov, webm, mkv, png, jpg, jpeg, webp; 1 B - 2 GB.
- **Consent**: a `character` manifest is refused without `consent: true`; the time is
  stored as `consent_at`. The CLI and UI ask before the first upload.
- **Clients**: `vidgen dataset add NAME FOLDER --kind K [--trigger T] [--yes]`, `vidgen
  dataset list`, `vidgen dataset rm NAME`; the UI's Upload button runs the same sync with
  progress, and the theme card shows the last upload and "changed since upload".
- **Retention (changed from the first draft)**: no automatic expiry until training exists;
  datasets stay until `vidgen dataset rm` / Remove and show their upload date. The 30-day
  expiry for never-trained datasets ships with the trainer, when "trained" exists.

### Trainer: step 2 (built 2026-11-10)

What step 2 ships, and where it narrows the design above (approved 2026-11-10):

- **T2V only**: a train job trains the T2V-A14B low-noise and high-noise experts (in that
  order). I2V experts are deferred (`--models t2v,i2v` later); `character` themes are
  trained on T2V too.
- **No previews on the trainer**: the trainer has no diffusers pipeline. Checkpoints are
  kept every 250 steps; picking one happens in step 3 by rendering on the render box
  (`--lora name@750`).
- **Limits**: train jobs time out after `train_job_timeout_minutes` (960); the waker's
  uptime cap for the trainer is `train_max_uptime_hours` (24). The render box keeps its own.
- **Infrastructure** (`training_enabled`, default false): spot g6e.4xlarge (`train_instance_type`),
  300 GB gp3, same private subnet, SSM, own role (read `code/trainer/*` and `datasets/*`,
  write `loras/*`, the train queue, its jobs, stop itself), queue `ai-video-train` (+ DLQ,
  1 receive: a failed training is not retried automatically). First boot installs pinned
  musubi-tuner and downloads the ComfyUI T2V experts (fp16), Wan2.1 VAE, umT5 and Qwen2.5-VL.
- **Job** (`POST /jobs {"type": "train", "dataset": NAME, "steps": 100-3000 (default 1250)}`):
  the API checks the dataset manifest exists and records its fingerprint; the train queue
  gets the job and the waker starts the trainer.
- **Steps on the trainer**: prepare (manifest files; clips cut into 5 s segments, at most 12
  per source video, evenly spread; photos as single frames), caption (Qwen2.5-VL, trigger
  word first, then setting/action/camera/light), cache latents + text, train low then high
  (fp16 mixed precision, fp8 base, rank 32 / alpha 16, AdamW8bit lr 2e-4, timesteps
  [0, 875) and [875, 1000]), convert (`convert_lora.py --target other`), upload.
- **Spot interruption**: the job's work folder lives on the trainer's EBS
  (`/opt/trainer/jobs/<job_id>`), with saved state every 250 steps. When the restarted
  trainer gets the job again it skips finished phases and resumes training from the last
  state with the remaining steps; progress shows "resumed at 500".
- **Output**: `loras/<client>/<name>/v<N>/` with `t2v_low.safetensors`, `t2v_high.safetensors`
  (converted, final), `checkpoints/t2v_{low,high}-step<S>.safetensors` and `lora.json`
  `{name, kind, trigger, version, steps_per_expert, base: {t2v, comfy files}, dataset_fingerprint,
  segments, created_at, train_seconds}`; `loras/<client>/<name>/current.json` points at it.

### Data flow

```
vidgen dataset add NAME FOLDER         ──► S3 datasets/<client>/<NAME>/raw/... + manifest.json
vidgen train NAME --kind character      ──► API ──► DynamoDB job + SQS "train" ──► waker starts trainer
trainer: download ─► prepare ─► caption ─► cache ─► train high-noise ─► train low-noise ─► upload
                                                         S3 loras/<client>/<NAME>/{t2v,i2v}_{high,low}.safetensors + lora.json
vidgen video "vgn_dana walking in the rain" --lora dana:0.9  ──► render worker loads the LoRA next to Lightning
```

### Trainer steps (one `train` job)

1. **Prepare**: ffprobe every clip; cut into 81-frame (5 s at 16 fps) and 33-frame
   segments at scene changes; resize to training buckets (480x832, 832x480, 624x624;
   photos as single frames at about 1 MP); drop clips under 1 s or with no face detected
   (`character` kind).
2. **Caption**: Qwen2.5-VL-7B writes one caption per clip/photo, starting with the trigger
   word, describing action, setting, camera and light but **not** the person's fixed
   features (so they bind to the trigger word). Captions are saved as `.txt` next to each
   item; `vidgen dataset captions NAME --edit` lets the user fix them before training.
3. **Cache**: VAE latents and umT5 text embeddings (musubi-tuner cache scripts).
4. **Train**: per model (T2V, then I2V for `character`/`motion`), the high-noise expert
   (timesteps ≥ 875 for T2V, ≥ 900 for I2V) and the low-noise expert separately.
   Defaults: rank 32, alpha 16, AdamW8bit, lr 1e-4, batch 1, gradient checkpointing,
   fp8 base, about 2000 steps per expert (`--steps` overrides), save every 500 steps.
5. **Sample**: after each saved checkpoint, render 2 short 33-frame previews with fixed
   seeds (prompt: trigger word + 2 stock scenes) and upload them, so the user can pick a
   checkpoint that isn't overfitted.
6. **Upload**: final + intermediate `.safetensors`, previews, `lora.json` (name, kind,
   trigger word, base model + revision, steps, rank, dataset hash, created_at).

Progress: "captioning 12/40", "t2v high 1200/2000", and so on. Cancel stops between steps
and keeps the last saved checkpoint.

### Training further (versions)

A theme is never "finished": every training round makes a new **version** of its LoRA,
like avatar history, and any version can be continued, used or restored.

- **Layout**: `loras/<client>/<name>/v<N>/` holds that version's `.safetensors`, previews
  and `lora.json` (adds `version`, `parent_version`, `mode`, `steps_total`, dataset hash);
  `loras/<client>/<name>/current.json` points at the version renders use. Each version also
  keeps its **training state** (`state/`: optimizer, scheduler, step, RNG; musubi-tuner
  `--save_state`, ~0.9 GB per expert, measured) so it can be resumed exactly.
- **Two modes** (`vidgen train NAME --continue`):
  - `resume` (default when the dataset is unchanged): `--resume <state>` on the same
    dataset, e.g. 2000 -> 3000 with `--steps 1000`. musubi-tuner restarts its step counter
    on resume, so the trainer passes the remaining steps and tracks `steps_total` itself.
    For a LoRA that has not learned enough yet.
  - `extend` (default when clips were added since the version, or with `--mode extend`):
    a new run initialised from the version's weights (`--network_weights`) on the **whole**
    current dataset (old + new clips, so it does not forget the old ones), with a lower
    learning rate (default 5e-5) and `--steps` (default 1000). For adding angles, more
    examples, or fixing a weakness.
- **Constraints**: rank, kind, trigger word and models (t2v/i2v) are fixed per theme; a
  change needs a new theme. A version trained on a different base model revision cannot be
  continued (`lora.json` base check); after a base upgrade the theme is retrained from its
  dataset as v1 of a new lineage.
- **Picking and rollback**: new versions become current only when the job finishes;
  `vidgen lora history NAME` lists versions with steps, mode, date and previews;
  `vidgen lora use NAME --version N` (or `pick NAME --version N --step S` for an
  intermediate checkpoint) switches renders to it. Old versions are kept until `lora rm`.
- **Datasets of trained themes**: kept while the theme exists (needed for `resume` and to
  mix old + new data in `extend`); `vidgen dataset rm NAME` deletes them, after which only
  `extend` with newly added clips is possible (the CLI warns that the theme may drift
  toward the new clips).

### Using a LoRA when rendering

- API: `video` / `story` / `photos` accept `loras: [{"name": str, "weight": 0.0-1.5}]`
  (max 2). The API checks each exists for the client.
- Render worker: `pipe.load_lora_weights(..., adapter_name=name)` into `transformer`
  (high) and `transformer_2` (low), then `set_adapters([lightning, name], [1.0, w])`;
  unload after the job. Kept resident only as long as the Wan model stays loaded.
- The prompt must contain the trigger word; the CLI adds it when missing.
- LoRAs for a base model revision other than the deployed one are rejected
  (`lora.json` base check).

### CLI

```
vidgen dataset add NAME FOLDER --kind character|style|motion|place [--trigger T] [--yes]
vidgen dataset list | rm NAME          (show, captions: with the trainer)
vidgen train NAME [--kind ...] [--trigger WORD] [--steps N] [--rank R] [--models t2v,i2v]
vidgen train NAME --continue [--from-version N] [--mode resume|extend] [--steps N] [--lr LR]
vidgen lora list [--kind K] | show NAME | history NAME | use NAME --version N
vidgen lora pick NAME [--version N] --step S | add NAME FILES... | rm NAME
vidgen lora prune NAME --keep N        (drop training state of older versions)
vidgen video "..." --lora NAME[:WEIGHT]      (also story, photos in phase 2)
```

## Acceptance Criteria

- When `vidgen dataset add dana FOLDER --kind character` runs, the system shall upload every
  mp4, mov, webm, mkv, png, jpg, jpeg and webp file of 1 B - 2 GB under FOLDER to
  `datasets/<client>/dana/raw/` and print the counts; other files shall be skipped with a
  warning.
- When the same folder is uploaded again unchanged, the system shall upload 0 files; when one
  file changed (size or mtime), it shall upload only that file.
- When a file was removed from the folder, the system shall list it and delete it from the
  dataset only after confirmation (or `--yes`).
- If a `character` dataset is committed without consent, then the API shall return 400.
- If an upload path contains `..`, starts with `/` or `\`, or has another extension, then the
  API shall return 400.
- If a manifest lists a file that is missing in S3 or has a different size, then
  `PUT /datasets/{name}` shall return 400 naming the file.
- When `vidgen dataset rm dana` runs (confirmed), every object under `datasets/<client>/dana/`
  shall be deleted.
- When a `train` job for kind `character` finishes, S3 shall hold
  `loras/<client>/<name>/` with `t2v_high`, `t2v_low`, `i2v_high`, `i2v_low` `.safetensors`,
  `lora.json` and at least one preview per saved checkpoint.
- While a `train` job runs, the render instance shall not be started or used by it.
- When the train queue has been empty for 15 minutes, the trainer instance shall stop.
- If `training_enabled = false`, then `terraform plan` shall create no trainer resources.
- If a dataset has fewer than 5 usable items after preparation, then the job shall fail
  with error `dataset too small: N usable items (need >= 5)`.
- When a `video` job gives `--lora dana:0.9`, the render worker shall apply the LoRA at
  weight 0.9 on both experts together with Lightning, and the job result shall list the
  LoRAs used.
- If a requested LoRA does not exist or was trained on a different base revision, then
  the API shall return 400.
- When `vidgen lora add NAME high.safetensors low.safetensors --model t2v` runs (trained on
  another machine), the LoRA shall become usable exactly like one trained by the trainer.
- When `vidgen train dana --continue --steps 1000` runs on an unchanged dataset whose current
  version is v1 at 2000 steps, the system shall resume from v1's training state and produce
  v2 with `steps_total` 3000, `mode` resume and `parent_version` 1.
- When clips were added to the dataset since the current version, `vidgen train dana
  --continue` shall run in `extend` mode on all clips, initialised from the current
  version's weights, at learning rate 5e-5 unless `--lr` is given.
- While a continue job runs, renders shall keep using the previous current version; when it
  finishes, the new version shall become current.
- When `vidgen lora use dana --version 1` runs, subsequent renders with `--lora dana` shall
  use v1.
- If `--continue` asks for a different rank, kind or trigger word, or the version was
  trained on a different base revision, then the API shall return 400.

## Out of Scope

Full fine-tuning; training Kokoro voices or MMAudio; training on several people in one
LoRA; automatic hyperparameter search; a web UI; sharing LoRAs between clients;
`photo-character` (Qwen-Image-Edit) LoRAs and `--lora` on `photos` (phase 2).

## Risks & Open Questions

- **Training GPU availability** (Resolved 2026-11-09, read-only checks): only us-east-1 is
  usable (other regions are blocked by SCP). p5.4xlarge / p5.48xlarge / p4d spot placement
  score is 1/10 in every zone, so H100/A100 **spot is not viable**; the P-instance quota
  (768 vCPU on-demand and spot) allows **on-demand p5.4xlarge**. g6e.4xlarge spot scores 9/10
  at about $1.8-2.1/h. Default trainer: g6e.4xlarge spot (L40S, fp8 + block swap if
  needed); on-demand p5.4xlarge as the fast option.
- **Training time and cost** (Resolved by spike, L40S): 12.7 s/step at 832x480 / 17-33 frames,
  so ~7 h per 2000-step expert, ~14 h for a T2V theme, ~28 h for `character` (T2V + I2V);
  at g6e.4xlarge spot prices ($1.8-2.1/h) that is ~$13-15 per expert, ~$25-30 for a T2V
  theme, ~$50-60 for `character`. H100 not measured.
- **Trainer choice** (Deferred): musubi-tuner (recommended: native Wan2.2 two-expert
  support, low-VRAM modes), diffusion-pipe (pipeline-parallel, good for multi-GPU),
  ostris ai-toolkit (simplest config, also Qwen-Image-Edit). Pin a commit either way;
  their CLIs change often.
- **Diffusers compatibility** (Resolved by spike): after `convert_lora.py --target other`
  the LoRA loads with diffusers 0.36 `load_lora_weights` into all 400 modules of the expert.
- **Lightning + custom LoRA stacking** (Partly resolved): stacks and renders correctly with
  a visible effect; likeness strength at 4 steps still to be judged on a real person dataset.
- **Overfitting** (Resolved by design): intermediate checkpoints + previews + `lora pick`.
- **Consent and licensing** (Resolved by policy): Wan2.2 weights are Apache-2.0; training
  on a real person's likeness requires that person's consent. The CLI asks for
  confirmation on `character` datasets and the README states it.
- **Data security**: datasets contain biometric data (faces). Private bucket (no public
  access, SSE), a per-client prefix. Datasets of trained themes are kept while the theme
  exists (training further needs them); datasets never trained expire after 30 days once
  the trainer ships (until then no automatic expiry; see "Datasets: upload").
  `vidgen dataset rm` and `vidgen lora rm` delete for real. LoRAs never expire (like avatars).
- **Storage for training further** (Resolved by spike): ~0.9 GB of training state per expert
  per version, so a T2V theme costs ~1.8 GB and a `character` theme (4 experts) ~3.5 GB per
  version.
  `vidgen lora prune NAME --keep 3` drops the training state of older versions (their
  weights stay usable but can no longer be resumed, only extended).
- **Drift when extending** (Resolved by design): `extend` always trains on old + new clips,
  and the previous version stays one command away (`lora use --version`).
- **Feature flag**: `training_enabled` (Terraform) gates all trainer infrastructure;
  `--lora` on render jobs is additive and does nothing unless given.

## Testing

- Spike first (`trainer/spike/run_trial.sh`): 200-step LoRA on the T2V low-noise expert from
  the 12 most recent output clips, on the render box (L40S), load in diffusers next to
  Lightning, render the same prompt with and without it. Go/no-go for the design; results
  under "Spike results" below.
- Unit: dataset preparation (segmenting, bucket resize, file filtering), caption
  formatting (trigger word first), `lora.json` validation, API `loras` validation.
- Spike, part 2 (done, go): resume from saved state and extend from LoRA weights both work
  and load in diffusers; see "Spike results".
- Smoke (`vidgen smoke --training`, opt-in, long): a 6-clip synthetic dataset (Wan-rendered
  clips of `smoke-a`), 100 steps per expert, T2V only; checks the files and `lora.json`,
  then one 2 s `video` with `--lora smoke-lora`.
- Manual: compare a character LoRA render against the avatar-only render of the same
  prompt and seed for likeness.

## Spike results

Run 2026-11-09 on the render box (g6e.4xlarge, L40S 48 GB) with `trainer/spike/run_trial.sh`:
musubi-tuner `f8a1b037`, 12 output clips (1280x704, 2-5 s), generic captions with trigger
`vgn_trial`, LoRA rank 32 on the **T2V low-noise expert**, 200 steps. **Verdict: go.**

| Check | Result |
|---|---|
| Install (venv, torch 2.5.1 cu124, musubi-tuner) | 85 s; worker venv untouched |
| Weights (ComfyUI fp16 low-noise expert, Wan2.1 VAE, umT5) | 38 GB, 139 s |
| Cache latents / text (12 clips, buckets 832x480, 17/33 frames) | 38 s / 14 s |
| Training, fp16 mixed precision + `--fp8_base`, no block swap | **12.7 s/step**, 200 steps in 43 min, loss ~0.09 |
| Peak GPU memory | **32.8 GB** of 48 GB |
| `convert_lora.py --target other` | 2 s |
| Load in diffusers 0.36 next to Lightning (`load_into_transformer_2=True`) | converted file loads into all **400** low-noise LoRA modules; raw musubi file not needed |
| Render 2 s clip (Lightning, 4 steps), with / without LoRA | 334 s (first render after loading, includes warm-up) / 112 s |
| Visual effect | same composition and motion; softer, less saturated grade, toward the training clips' look (low-noise expert = detail and color) |

Lessons folded into the design:

- ComfyUI-format experts are **fp16**: musubi-tuner requires `--mixed_precision fp16` for them
  (bf16 fails its argument check).
- The LoRA must be converted with `convert_lora.py --target other` before diffusers loads it;
  add this as trainer step 6 ("Upload" stores the converted files).
- The worker role (by design) cannot list or read `outputs/`; the trainer role gets its own
  scoped access to `datasets/*` and `loras/*` (already in the design) and uploads with
  per-object puts.
- Time estimate (L40S): 2000 steps per expert = **~7 h**; T2V theme (2 experts) **~14 h**;
  `character` (T2V + I2V, 4 experts) **~28 h**. An H100 is expected to be 2-3x faster
  (not measured: no spot capacity). Defaults change to **1000-1500 steps per expert** and
  T2V-only unless `--models t2v,i2v`, with checkpoints every 250 steps to pick from.

### Spike part 2: training further (2026-11-09, `trainer/spike/run_part2.sh`) — go

| Check | Result |
|---|---|
| `--save_state` at step 60 | state folder **884 MB** for one expert (optimizer, scheduler, RNG) |
| `--resume` from it | state reloads (log "resume training from local state"); first loss **0.060** vs **0.172** from scratch |
| Step counter on resume | **restarts at 0**: `--max_train_steps 90` ran 90 *more* steps, not 30 |
| Extend: `--network_weights` from part 1's LoRA, 12 + 6 new clips, lr 5e-5, 100 steps | first loss **0.059** (from scratch: 0.172), last 0.087; 12.6 s/step |
| Both converted LoRAs in diffusers next to Lightning | resume **400**/400 modules, extend **400**/400; extended LoRA renders (333 s incl. warm-up) |

Lessons folded into the design:

- The trainer passes **remaining** steps on resume (`--max_train_steps = target - steps_total`)
  and keeps the running total itself in `lora.json` (`steps_total`); musubi-tuner does not
  continue its step counter.
- Training state is ~0.9 GB per expert, so ~1.8 GB per version for a T2V theme and ~3.5 GB
  for `character` (4 experts).

Still open: high-noise expert and I2V training; likeness test with real captions on a
person dataset.

## Update: rendering with `--lora` (2026-11-10)

Built for `video` and `story` (`photos` stays phase 2):

- API `loras: [{name, weight}]` (or `"name:weight,..."`), max 2, weight 0-1.5. Each is pinned at
  submit time to `current.json`'s version; its `lora.json` must list `t2v` and both
  `t2v_{high,low}.safetensors`, otherwise 400. Params carry name, version, weight, trigger and both keys.
- Worker downloads the files once per version to `/opt/models/loras/<client>/<name>/v<N>/`, loads
  them next to Lightning (`set_adapters([lightning, lightning_2, user, user_2], [1, 1, w, w])`), keeps
  them loaded while the next job asks for the same version, and drops them otherwise. The trigger
  word is prepended to clip prompts when missing; the result lists the LoRAs used.
- Avatar shots use I2V, and the trainer only makes T2V LoRAs: the T2V LoRA is loaded into I2V too.
  Whether that carries the motion well is checked with a side-by-side render (same seed, with and
  without the LoRA).
- No base-revision check yet: the trainer records a label, not a revision (both are Wan2.2-T2V-A14B).
