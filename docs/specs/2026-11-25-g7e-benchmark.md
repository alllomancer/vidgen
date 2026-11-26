# g7e benchmark: is a 96 GB GPU worth moving the render box to?

Jira: N/A
Date: 2026-11-25
Status: Approved
Last reviewed: 2026-11-25

## Problem

On the L40S (48 GB) the two 14B Wan experts (~28 GB each in bf16) don't fit together, so every 5 s segment
moves each of them on and off the GPU; the RUNBOOK says most of a 4-step Lightning render is that. A g7e
(RTX PRO 6000 Blackwell, 96 GB) could hold both. Moving the render box there needs the worker's stack on
torch >= 2.7 / CUDA 12.8 (torch 2.5.1 has no kernels for that GPU), so it should be measured first.

## Approach

A temporary on-demand `g7e.4xlarge`, separate from the render box, with its own venv and one benchmark
script. It is terminated at the end (and shuts itself down after 4 hours whatever happens). The same script
runs once on the existing g6e box with the production venv as the baseline.

## Design

- `trainer/spike/bench_wan.py`: builds the T2V pipeline the way `worker/models.py` `_load_wan` does (fp32
  VAE with tiling and slicing, bf16 experts, the pinned Lightning LoRAs, 4 steps, no CFG) and renders the
  production segment (1280x704, 81 frames) with a fixed prompt and seed: one warm-up, then `--runs` timed
  runs. `--mode offload` uses `enable_model_cpu_offload()` (what the deployed worker does);
  `--mode resident` keeps every component on the GPU. Per run it records seconds per denoising step, the
  rest (`finish`: decode and moving back), the total, and peak GPU memory; it writes one JSON file.
- `trainer/spike/run_g7e_bench.sh` (trial box, root): venv with torch 2.9.0+cu128 and the worker's
  diffusers/transformers/peft pins, downloads `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, runs `resident` then
  `offload`, uploads the JSON files and the log to `s3://<bucket>/outputs/trial-g7e/<timestamp>/`.
- g6e baseline: on the render box, worker service stopped for the run and started again afterwards (as
  `trainer/spike/run_trial.sh` does), `bench_wan.py --mode offload` with `/opt/ai-video/venv` and the local
  bf16 copy. The box is stopped again at the end.
- The trial box: newest DLAMI (the render box's image), the render box's subnet, security group and
  instance profile (S3 read `code/`, write `outputs/`, SSM), 300 GB gp3 at 1000 MB/s, encrypted, IMDSv2,
  tags `Name=ai-video-g7e-trial`, `Project=ai-video-trial` (so nothing of the project's own automation
  acts on it), shutdown behaviour `terminate`.

## Acceptance Criteria

1. The system shall produce, for g7e resident, g7e offload and g6e offload, the median seconds per 5 s
   segment split into denoising steps and the rest, and the peak GPU memory.
2. When the benchmark is finished or fails, the trial instance shall be terminated; if nobody terminates
   it, it shall shut itself down (and so terminate) 4 hours after boot.
3. The render box shall end in the state it started in (stopped, worker service enabled).

## Out of Scope

Moving the render box; upgrading the worker's torch; I2V, the edit model, Krea, sound; quality comparison
beyond "the clip rendered".

## Risks & Open Questions

- The g7e runs use torch 2.9 and the g6e run torch 2.5.1: the GPU and the library differ at once. The g7e
  offload run separates "faster GPU" from "no swapping". (Accepted.)
- `resident` may not fit 96 GB at 1280x704 with the fp32 VAE decode; the script then reports the error and
  the offload numbers still stand. (Open until run.)
- Cost: about $4/h on-demand, expected 2-3 hours. (Approved.)

## Testing

`python3 -m py_compile` and `bash -n` locally; the run itself is the test.

## Results (2026-11-25)

One 5 s segment, 1280x704, 81 frames, Lightning 4 steps. g7e: T2V, Lightning LoRAs only, torch 2.9.0+cu128,
median of 3 after a warm-up (`trainer/spike/bench_wan.py`; raw files in
`s3://<bucket>/outputs/trial-g7e/20261125-*/`).

| Run | Total | 4 steps | Rest (decode, moving back) | Per plain step | Peak GPU |
|---|---|---|---|---|---|
| g7e, both experts on the GPU | 104.3 s | 88.5 s | 15.9 s | 22.1 s | 74.5 GB |
| g7e, offload (as the worker does) | 148.6 s | 117.5 s | 30.9 s | 22.2 s | 36.1 GB |
| g6e, production worker log (not the benchmark, see below) | ~240 s | ~208 s | ~31 s | ~46 s | n/a |

- Both experts, the text encoder and the fp32 VAE decode fit 96 GB at production size (criterion 1, g7e).
- Swapping costs ~44 s of a 149 s segment on the g7e (30%): ~29 s inside the steps, ~15 s after them.
- The g6e row is from the render box's own `pipeline_timing` lines of a 30 s video rendered the same
  evening: I2V, with two user LoRAs besides Lightning, torch 2.5.1. It is not like-for-like (I2V and extra
  LoRAs make its steps slower than the benchmark's T2V), so "about 2x per step, about 2.3x per segment" is
  an indication, not a measurement. The planned g6e benchmark run was not done: the render box was
  rendering the user's jobs and the worker was not paused (criterion 1 open for g6e; criterion 3 holds, the
  box was only read).
- The trial box ran 21:31-22:10 UTC (us-east-1b; us-east-1a had no on-demand capacity) and was terminated
  (criterion 2). Hugging Face download on it: 118 GB in 874 s (135 MB/s).

### g6e baseline, same script (2026-11-25, cut short)

Run on the render box with the worker paused. Only the warm-up run finished: the user had queued jobs behind
it, so the benchmark was stopped after ~11 minutes and the worker restarted (criterion 3: the box is back in
service).

| | g6e (L40S, torch 2.5.1) | g7e (torch 2.9.0) |
|---|---|---|
| One plain denoising step (T2V, Lightning only) | 39.6 s | 22.1 s |

1.8x per step. The warm-up's swap steps (175 s, 149 s) and total (453 s) include first-use paging and are not
a measurement of a steady segment; the production log's ~240 s per I2V segment stays the practical g6e number.
