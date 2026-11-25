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
