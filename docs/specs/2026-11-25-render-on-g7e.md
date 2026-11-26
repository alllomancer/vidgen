# Render box on g7e: worker ready for the 96 GB GPU, and a script that moves it there

Jira: N/A
Date: 2026-11-25
Status: Draft
Last reviewed: 2026-11-25

## Problem

The benchmark (docs/specs/2026-11-25-g7e-benchmark.md) shows a 5 s Wan segment in 104 s on a g7e with both
experts on the GPU, against ~240 s in tonight's production log on the g6e. Two things stand between the
render box and that:

- The worker can't run on a g7e: torch 2.5.1 has no kernels for its GPU (needs >= 2.7 built for CUDA 12.8),
  and the Krea venv's torch is a CUDA 12.6 build.
- Even there it would swap the experts on and off the GPU (`enable_model_cpu_offload`), which the benchmark
  puts at 30% of a segment.

And moving the box is manual today: pick a size and zone with capacity, edit `terraform.tfvars`, apply.

## Approach

1. Make the worker run on both GPUs with one set of pins, and keep a big model wholly on the GPU when the
   GPU has room (>= 80 GB).
2. Prove it on a temporary g7e box with a validation script that drives the production loaders, before
   anything may move the real box.
3. A script that finds g7e capacity by really launching (spot g7e.8xlarge, then spot g7e.4xlarge, else
   on-demand g7e.4xlarge), writes the choice to `terraform.tfvars` and applies.

## Design

### Worker pins (worker/requirements.txt, worker/krea_requirements.txt)

- Worker venv: torch 2.9.0+cu128 (the build the benchmark ran) with matching torchvision/torchaudio; the
  packages pinned "to keep torch at 2.5.1" (torchcodec, tensordict) move to the versions that go with it.
  The exact versions are whatever installs and passes validation on the trial box; they are written back
  as pins (`pip freeze` of that venv).
- Krea venv: stays as it is if its torch lists `sm_120` on the g7e; otherwise the nearest build that does.
- The box installs requirements at every worker start, so the same pins also reach a g6e. CUDA 12.8 builds
  run on the L40S (driver 595 on the box).

### Big models held on the GPU (worker/models.py)

- `GPU_HOLD`: true when the GPU has >= 80 GB (`GPU_HOLD=0|1` in `/etc/ai-video.env` overrides).
- With `GPU_HOLD`, `_offload(pipe)` moves the whole pipeline to the GPU instead of installing offload hooks,
  and `t2v`, `i2v` and `edit` join the VLM in `GPU_RESIDENT`: only one of them is loaded at a time, and
  switching reloads from disk (the page cache keeps recently used weights, and `warm_ahead` already reads
  the next model while the current one renders). Without `GPU_HOLD` nothing changes.
- Whatever assumes offload hooks (`_prompt_embeds`, `keep_ram_weights`, the clean-up in `_timed`) is made
  to work in both modes.
- MMAudio (a few GB) still moves to the GPU for a sound session next to the held Wan pipeline; Krea still
  runs after `release_all()` in its own process.

### Validation on a temporary g7e (trainer/spike/validate_g7e.py, run_g7e_validate.sh)

- Same kind of box as the benchmark (on-demand g7e.4xlarge, own tags, shuts itself down after 4 h,
  terminated at the end). Worker code from the working tree (uploaded to `code/g7e-trial/worker/`), venvs
  built from the new requirements files, weights from the model mirror when it has them, else Hugging Face.
- The script imports the worker's `models` / `krea` and runs, timing each: T2V segment (1280x704, 81
  frames), I2V segment from a generated frame, a keyframe/photo with the edit model, a Krea image with the
  content filter, MMAudio sound for the clip, a VLM description. It records seconds, peak GPU memory, and
  the versions, and fails loudly on the first error.
- Passing = every step produced its output with `GPU_HOLD` on and peak GPU memory under 96 GB.

### Switch script (cli/g7e_switch.py, `make g7e`)

1. **Preflight** (stops with a reason unless `--force`): AWS credentials; the worker pins are the validated
   ones (a marker line in requirements.txt); no job queued or running (SQS: 0 visible, 0 in flight); the
   model mirror has a manifest for the bf16 Wan copies, the edit and VLM cache folders and MMAudio (so the
   new disk restores in minutes, not ~45).
2. **Find capacity** by launching and terminating a probe instance (one-time spot, the project's AMI and
   the zone's private subnet, no instance profile, tags `Name=ai-video-probe`, `Project=ai-video-trial`,
   shutdown = terminate): spot `g7e.8xlarge` in us-east-1a/b/d, then spot `g7e.4xlarge` in those zones. A
   probe counts when it reaches `running`; it is terminated at once in every case.
3. **Else on-demand** `g7e.4xlarge`: the same probe on-demand, per zone.
4. **Apply**: `instance_type`, `on_demand`, `availability_zone` in `terraform.tfvars` (other lines kept;
   `train_availability_zone` keeps the trainer where it is), `make build`, `terraform plan`, then
   `terraform apply` of that plan. If the apply fails for capacity, the next candidate is tried; if all
   fail, `terraform.tfvars` is put back and nothing was replaced.
5. Prints what it chose, the hourly price, and how to watch first boot. `--dry-run` stops before step 4.

Replacing the instance destroys its disk; that is what the script is for, so it does not ask again
(decided 2026-11-25). Jobs sent while the new box boots wait in the queue.

## Acceptance Criteria

Worker:

1. While `GPU_HOLD` is off, the system shall load and offload models exactly as before (existing tests pass
   unchanged).
2. While `GPU_HOLD` is on, `_offload` shall move the pipeline to the GPU and install no offload hooks, and
   loading `t2v`, `i2v` or `edit` shall first drop any other of them and the VLM.
3. `GPU_HOLD` shall be true for a GPU of >= 80 GB and false below, and `GPU_HOLD=0` / `GPU_HOLD=1` shall
   override it.
4. On the trial g7e, the validation script shall complete all six steps with `GPU_HOLD` on.

Switch script:

5. When a job is queued or running, the script shall stop before probing with a message naming the reason.
6. When the spot probe for `g7e.8xlarge` reaches `running` in a zone, the script shall choose that size and
   zone with `on_demand = false` and shall not probe further.
7. If no 8xlarge probe succeeds and a `g7e.4xlarge` spot probe does, it shall choose that.
8. If no spot probe succeeds, it shall choose on-demand `g7e.4xlarge` in the first zone whose on-demand
   probe reaches `running`; if none does, it shall exit non-zero and leave `terraform.tfvars` unchanged.
9. Every probe instance shall be terminated, also when the script is interrupted or a later step fails.
10. The script shall change only `instance_type`, `on_demand` and `availability_zone` in `terraform.tfvars`.
11. If `terraform apply` fails, the script shall try the next candidate, and when none is left restore
    `terraform.tfvars`.
12. With `--dry-run`, the script shall print the choice and change nothing.

## Out of Scope

- The trainer (stays on g6e, in its zone).
- Faster VAE, fp8, torch.compile, the edit model's 4-step Lightning.
- Falling back from g7e to g6e automatically when a spot g7e is interrupted.
- Keeping more than one big model on the GPU.

## Risks & Open Questions

- **Dependency upgrade**: MMAudio, torchcodec, tensordict, kokoro or spandrel may not have a combination
  that installs with torch 2.9 on Python 3.10. (Open until the trial box; if one doesn't, that feature is
  reported, not silently dropped.)
- **The edit model held on the GPU** (~58 GB plus activations) and **Wan + MMAudio together** were not
  measured. (Open until validation.)
- **A probe is not a reservation**: capacity can be gone when Terraform launches seconds later; hence the
  retry on apply. A persistent spot box can also be stopped by AWS later, as today.
- **Spot g7e score is 1 of 10 in every zone today**, so the script will most likely choose on-demand
  ($4.00/h against ~$1.40-2.20 spot; the box still stops itself when idle).
- **The apply deploys everything in the working tree**, including Terraform and worker changes that are
  not committed yet. The script prints the plan summary before applying.
- **The mirror must be deployed and seeded first** (one `make plan && make apply` and one start of the
  render box); otherwise preflight stops, or with `--force` first boot downloads at ~135 MB/s and converts.
- Feature flag: `GPU_HOLD=0` on the box turns holding off.

## Testing

- `tests/test_worker.py` / a new `tests/test_gpu_hold.py`: criteria 1-3 with stubbed torch.
- `tests/test_g7e_switch.py`: the script against fake `aws` / `terraform` runners (criteria 5-12).
- The trial box run (criterion 4), results appended here.

## Validation results (2026-11-26, criterion 4: partly met)

Spot `g7e.8xlarge` in us-east-1b, the working tree's worker code, `GPU_HOLD` on (95 GB GPU), torch 2.9.0+cu128
(`sm_120` listed). Worker venv from the requirements with these lines changed; it installed without conflicts:

```
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.9.0+cu128
torchvision==0.24.0+cu128
torchaudio==2.9.0+cu128
torchcodec==0.8.0
tensordict==0.10.0
```

| Step (through the worker's own functions) | Result | First pass (with load) | Second pass | Peak GPU |
|---|---|---|---|---|
| T2V, 5 s at 1280x704, 32 fps + raw | ok | 318 s | 106 s | 74.3 GB |
| Keyframe with the edit model (Lightning 8) | ok | 87 s | 6.2 s | 65.6 GB |
| I2V, 5 s at 1280x704 | ok | 311 s | 115 s | 74.3 GB |
| MMAudio sound, next to the held I2V pipeline | ok | 61 s | - | 71.3 GB |
| VLM description | ok | 22 s | - | 65.6 GB |
| Krea 2 image | **failed** | - | - | - |

- Krea: `CUDA error: no kernel image is available for execution on the device`. The Krea venv's
  torch 2.14.1+cu126 has no kernels for this GPU; it needs a build that lists `sm_120`. Open: which build
  works with its diffusers 0.41 / transformers 5.20 pins on Python 3.10.
- So `worker/requirements.txt` is **not** changed yet: with the torch pin at 2.5.1 the switch script's
  preflight keeps refusing, which is right while Krea would be broken on a g7e.
- A keyframe takes 6 s with the edit model held on the GPU (the RUNBOOK's figure on the g6e is ~40 s per
  photo).
- Capacity seen while launching (2026-11-26 00:04 UTC): no on-demand `g7e.4xlarge` or `g7e.8xlarge` and no
  spot `g7e.4xlarge` in us-east-1a/b/d; spot `g7e.8xlarge` launched in us-east-1b.
