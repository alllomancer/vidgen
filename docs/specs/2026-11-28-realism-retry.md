# Auto-retry until it looks real

Jira: N/A
Date: 2026-11-28
Status: Approved
Last reviewed: 2026-11-28

Approved in conversation: photos (one picture) and videos (5 s clips; the best clip is the result), up to 25 tries,
stop at score 0 or after 6 tries without a better score, each retry from the best attempt's suggested prompt.
Builds on docs/specs/2026-11-27-realism-check.md.

## Problem

The AI-look check scores one render and suggests a better prompt; trying that suggestion means rendering again by
hand, many times. The user wants the render box to keep retrying a single picture (or a short clip) with the same
avatar and LoRAs until the score is as low as it gets, within 25 tries.

## Approach

A job option `realism_retries` (1-25). The worker renders, runs the AI-look check, and renders again with a new seed
and the rewritten prompt of the best attempt so far, until the score is 0, 6 tries in a row bring no better score,
or the tries run out. Everything happens inside one job, so the avatar references and LoRAs stay as they were set
for the job. Every attempt is kept; the best one is the job's result.

## Design

- **Which jobs**: `photos` with one picture (count 1, at most one scene, no series) and `video` (one take, no
  variations, no plan first, no beats, no re-anchoring); `realism_check` must be on. A video with retries is a 5 s
  clip (`seconds` must be absent or 5). 400 otherwise.
- **The loop** (`realism_check.retry`, worker): try 1 renders the job as asked; try N > 1 renders with a new seed and
  the best attempt's suggestion (videos: its `prompt`; photos: its rewritten scene as `scenes: [{prompt}]`). After
  each render the check runs with the content filter as always; a skipped or failed check ends the loop (the
  attempt so far is kept). Stops: best score 0, `PATIENCE = 6` tries without a better score, or `realism_retries`
  tries.
- **Files**: each attempt's outputs are copied in S3 to `attempt_<n>_<name>` (worker: Get+Put on `outputs/*`), so
  later tries don't overwrite them. The job's result is the best attempt's result with its keys pointing at its
  copies (so `photo_1` / `video` is the best attempt), plus `attempt_<n>_key` for every attempt's main file, its
  `realism` verdict, and `realism_retry = {tries, best_try, stopped, attempts: [{try, score, signs, prompt, seed}]}`.
- **Progress**: every progress line of a try starts with "try 4/25 · best 3/10 · ". Each try's check is logged.
- **Local copy**: the watcher's `realism.json` also lists every try (score, signs, prompt, seed) under `tries`.
- **Clients**: dashboard and phone, in the Photos and Video forms, **Retry until it looks real (up to 25 tries)**
  (turns the AI-look check on; video forms switch to 5 s); CLI `--realism-retries N` on `photos` and `video`.
  History shows "best 2/10 after 9 tries"; the attempts download with their scores in `realism.json`.
- **Cost**: each try is a render plus a check, about 1-2 min on the g7e; 25 tries is up to ~40 min of GPU.

## Acceptance Criteria

- When a photos job with `realism_retries: 25` gets scores 7, 5, 6, 3, 4, 4, 4, 4, 4, 4 (6 without improving after
  try 4), the worker shall stop after try 10 and the job's `photo_1` shall be try 4's picture.
- When an attempt scores 0, the worker shall stop after it.
- Try N > 1 shall render with the best attempt's suggested prompt (scene) and a seed different from earlier tries,
  and with the job's avatar references and LoRAs unchanged.
- If an attempt's check is skipped (sexual content) or fails, then the worker shall stop and keep the best attempt so
  far (try 1 if it is the first).
- If a render fails on a later try (e.g. the content filter flags a rewritten Krea scene), or the job is cancelled or
  times out after a scored try, then the worker shall stop and keep the best attempt so far, with `stopped` saying
  why; a failure or cancel on try 1 fails or cancels the job as usual.
- `realism_retries` on a photos job with 2+ pictures, a series, a video with variations/plan first/beats/anchoring
  or `seconds` other than 5, a job without `realism_check`, or outside 1-25 shall return 400.
- Every attempt's main file shall be in the result as `attempt_<n>_key`, and `realism_retry.attempts` shall list each
  attempt's score.

## Out of Scope

Stories; several pictures or variations per retry run; rendering a longer video from the winning clip; changing
realism settings between tries (the suggestions are prompt-only).

## Testing

`tests/test_realism_retry.py`: the loop with fake render/check/copy (stops, prompts, seeds, files, skip), API
validation; page tests for the new controls.

## Update: a better critic and a loop that learns (2026-11-28)

The user found the improvement rate poor. Research (see the conversation's sources: artifact benchmarks show even
frontier models are weak at spotting artifacts, Qwen2.5-VL-7B among the weaker; iterative prompt-refinement papers
feed earlier attempts back to the rewriter) led to two changes, approved by the user:

- **The loop learns.** Each try's critique is three steps in one critic call: *judge* (deterministic: the ten
  checks and signs), *rewrite* (sampled, seeded per try, so it doesn't repeat itself; given the judge's findings and
  every earlier try's prompt, score and signs, marked best), and *compare* (the best try's pictures and this try's,
  side by side: which looks more real). The comparison decides whether a try beats the best; the score only when it
  says "same" or can't be read (`_better`). The attempts list records `compared`.
- **A stronger critic.** Qwen3-VL-32B-Instruct (Apache-2.0, ungated, revision `0cfaf48183f5`, ~66 GB bf16) runs in
  the Krea venv (it has transformers 5.20 with Qwen3-VL) through `vlm_judge.py`: weights from the model mirror or
  Hugging Face on first use (`vlm32.py`, mirror unit `qwen3-vl-32b/0cfaf48183f5`), the GPU freed first, one model
  load per try for all steps. `REALISM_CRITIC=qwen2.5-vl-7b` in `/etc/ai-video.env` goes back to the worker's own 7B
  model (`models.critic_local`). Cost: the 32B load and freeing the GPU add roughly 1-2 min per try (the render model
  loads again afterwards); measured in the live check.
