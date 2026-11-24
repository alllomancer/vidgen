# LoRA checkpoints and previews, photo shot files

Jira: N/A
Date: 2026-11-11
Status: Approved
Last reviewed: 2026-11-11

## Problem

- A training run saves checkpoints every 250 steps, but renders can only use the final weights of
  the newest version, and there is no way to see what a checkpoint does without spending renders
  on it by hand.
- The dashboard shows LoRAs but cannot choose or preview one.
- `photos --where/--wearing/--doing/--camera` makes one photo per job; a set of shots needs one
  command per photo.

## Approach

Four pieces, built in this order: shot files, checkpoint picking, previews, dashboard. Pieces that
render a real person (image-to-video LoRA training, photo-to-video, saved outfits) are out of scope
(user's choice, 2026-11-11).

## Design

### 1. Shot files (`photos --shots FILE`)

Plain text, one block per photo, blocks separated by blank lines, `#` comments:

```
where: on a rooftop terrace in Lisbon at golden hour
wearing: a light linen shirt and beige chinos
doing: leaning on the railing, smiling
camera: low angle, wide lens

where: a quiet library reading room
camera: from behind, over the shoulder
```

The CLI parses it into `{"shot": {...}}` scenes (the API format already deployed); unknown keys,
empty blocks and more than 8 photos are errors with the line number. Combines with `--prompt`,
`--location` and the single-shot flags (those come first).

### 2. Checkpoint picking

- `loras/<client>/<name>/current.json`: `{"version": N, "step": S | null}`; null = final weights.
- `PUT /loras/{name}/current` `{version, step?}`: 404 for an unknown version, 400 for a step whose
  two checkpoint files are not in that version's `lora.json` `files`. The API role gets PutObject on
  `loras/*/current.json` only.
- Per job: `loras: [{name, weight, version?, step?}]`; CLI `--lora NAME[@[vN][-]STEP][:WEIGHT]`, e.g.
  `jerk@750`, `jerk@v1`, `jerk@v1-750:0.8`. Without version/step the job uses `current.json`.
- Job params gain `step`; the worker adapter name includes it (`user_jerk_v1_s750`), so a different
  pick reloads. Files: `checkpoints/t2v_{high,low}-step000750.safetensors`.
- CLI `lora use NAME --version N [--step S]`; `lora list` marks the pick.

### 3. Previews (`lora preview`)

- New job type `lora_preview` on the render queue: `{name, version, steps: [..] (max 6, default all
  checkpoints, capped to the last 5), prompt, seed, seconds: 2}`, plus a baseline clip without the
  LoRA. Text-to-video only, 832x480, no sound, one seed for all clips.
- Default prompt: the trigger word plus a neutral stock scene; `--prompt` replaces the scene.
- Result: `preview_base.mp4`, `preview_s<step>.mp4`, downloaded into a folder like photos.
- CLI `lora preview NAME [--version N] [--steps 500,1000] [--prompt TEXT] [-o DIR]`.

### 4. Dashboard (Learning tab)

Each trained theme shows its versions and checkpoints with the current pick marked, a "Use" button
per checkpoint (calls `PUT /loras/{name}/current`) and a "Preview" button (queues `lora_preview`).

## Acceptance Criteria

- When a shot file has 2 blocks, `photos --shots` shall submit 2 `shot` scenes in file order.
- If a shot file has an unknown key, then the CLI shall exit with an error naming the line.
- When `PUT /loras/jerk/current` gets `{version: 1, step: 750}`, later jobs with `--lora jerk` shall
  use `checkpoints/t2v_{high,low}-step000750.safetensors` of v1.
- If the step has no checkpoint files, then the API shall return 400 and leave `current.json` as is.
- When a job gives `jerk@v1-500`, the job shall use v1 step 500 whatever `current.json` says.
- When a `lora_preview` job for 3 steps finishes, its result shall hold 4 clips (baseline + 3)
  rendered with the same seed and prompt.
- While no `step` is set, renders shall use the final weights exactly as before.

## Out of Scope

Image-to-video LoRA training, LoRAs on avatar videos, photo-to-video, saved outfits, a content
guard (all deferred with the real-person pieces); auto-queueing previews when training finishes.

## Risks & Open Questions

- A preview job loads the T2V model (~8 min cold) and renders up to 6 clips (~1 min each warm).
  (Accepted.)
- Checkpoint files of versions trained before this change use the same names. (Resolved: checked
  against v1's `lora.json`.)
- No feature flag: without new fields everything behaves as today.

## Testing

Unit tests (stubbed S3/GPU) for the shot parser, `PUT /current`, per-job pins, adapter names and the
preview runner; after deploy, `lora use`, one `lora preview` job, dashboard buttons in the browser.
