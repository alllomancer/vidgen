# Long-take quality: halo, planner sees the first frame, plan review

Jira: N/A
Date: 2026-11-25
Status: Approved
Last reviewed: 2026-11-25

## Problem

The 50 s live takes (docs/specs/2026-11-24-long-takes.md live checks 1-3, docs/specs/2026-11-24-avatar-wan-lora.md live
check) hold colour and, with the avatar Wan LoRA, likeness, but two problems remain:

1. A bright halo builds up around the subject over the take. Each 5 s segment starts from the previous segment's last
   frame; Wan I2V reproduces and slightly strengthens the edge contrast of the frame it starts from, so it compounds
   over 10 seams.
2. The planner writes the per-segment prompts from the user's text only (the first frame doesn't exist yet when it
   runs), so it invents things that aren't in the scene: strangers walking on the water, glitter particles,
   "touching the water" from a pier. The user also only sees the plan after an hour of rendering.

## Approach

1. Soften the frame each segment starts from (a light blur, ~0.8 px) before it conditions the next segment.
2. Make the first frame before planning, and give it to the planner as an image with the rule to keep to what is in
   it.
3. A "plan only" video job that makes the first frame and the plan and stops, returning editable beats; CLI
   `vidgen plan`, dashboard "Plan first". The user edits the beats and renders with them (and the same seed).

## Design

### Halo (worker/models.py)
- `CONDITION_BLUR = 0.8` (px, Gaussian, PIL `ImageFilter.GaussianBlur`); `soften(image, radius)`.
- In `generate_video`, for takes over 5 s, the conditioning image of every segment after the first is softened after
  any re-anchoring/colour transfer and before `_render_segment`. Segment 1's first frame and all output frames are
  unchanged. The blur radius is part of the resume key (old saved segments aren't spliced with new ones).

### First frame, then plan (worker/worker.py, worker/models.py)
- `run_video` for a single take: the first frame is made first, by a new `make_first_frame(job, params, cast, seed)`:
  the Krea first frame when `image_model` is `krea2` (no avatar, or the avatar's Krea LoRA), else the Qwen avatar
  keyframe when there is a cast, else none (text-to-video). `render_take` takes it instead of making the keyframe
  itself (same files, `keyframe.png`, same seed as today).
- `take_timeline` passes the first frame's path to `models.plan_segments(prompt, n, cast, seconds, image=path)`; the
  planner message then starts with the image and adds: "This is the opening frame of the take. Every prompt must keep
  to what is in this frame: the same people (no one else appears), place, light and camera; no new animals, objects,
  crowds or effects." Without a first frame the planner works from text as today.
- Variations (takes of up to 5 s) are unchanged.

### Plan only (API, worker, CLI, dashboard)
- API `video` param `plan_only: true`: only for takes over 15 s (the planner's range), not with `beats` or
  `variations` > 1 (400 naming the rule). Summary shows "plan".
- Worker: with `plan_only`, `run_video` makes the first frame and the plan and stops (no Wan, no sound). Result:
  `{plan_only: true, segment_prompts, beats: "0: ...\n5: ...\n...", seed, keyframe_key?}`. A planner failure fails the
  job ("the planner could not write a timeline; try again or write beats yourself") instead of falling back.
- CLI `vidgen plan "PROMPT" --seconds N [video options]`: submits a plan-only job and waits for it (progress lines;
  Ctrl-C leaves it to the watcher), then writes `<download dir>/<name>.beats.txt` (one `SECONDS: text` per line) and
  `<name>.keyframe.png`, prints the beats and the command to render them:
  `vidgen video "PROMPT" --seconds N --seed S --beats-file <name>.beats.txt`. The watcher saves the same two files for
  a plan job it downloads.
- Dashboard Video form: a "Plan first" button for takes over 15 s; it submits the plan-only job, waits (status line),
  then fills the Beats box and the Seed field and shows the first frame. "Make video" then renders with them.
- Rendering with the plan's seed makes the same first frame (same seed, same models); the beats replace the planner.

## Acceptance Criteria

- When a take is over 5 s, every segment after the first shall be conditioned on a copy of the previous last frame
  blurred with radius 0.8; segment 1 shall get the first frame unchanged and the output frames shall be unchanged.
- When a 50 s take has no beats, the first frame shall be made before the planner runs and passed to
  `plan_segments` as an image, and the planner's request shall contain the keep-to-the-frame rule.
- When a take has no first frame (Wan first frame, no avatar), `plan_segments` shall be called without an image.
- When a video job has `plan_only: true` and `seconds: 50`, the worker shall make the first frame and 10 segment
  prompts, render no video, and return `beats` with lines at 0, 5, ..., 45 and the `seed`; the job shall fail with
  the planner message if the planner fails.
- If `plan_only` is sent with `seconds` <= 15, with `beats`, or with `variations` > 1, then the API shall return 400
  naming the rule.
- When `vidgen plan "..." --seconds 50` finishes, `<name>.beats.txt` (10 lines `SECONDS: text`) and
  `<name>.keyframe.png` shall be in the download folder and the render command printed with the plan's seed.
- When the dashboard's "Plan first" job finishes, the Beats box and the Seed field shall hold the plan.

## Out of Scope

Planning for stories; automatic re-planning when the user edits the first frame; picking among several first
frames; motion smoothing across seams; changing the colour matching or re-anchoring.

## Risks & Open Questions

- The blur may soften fine detail (hair, fabric) over 10 segments. (Needs live check: the same 50 s take vs the last
  run; adjust `CONDITION_BLUR` between 0.5 and 1.2.)
- The planner can still invent despite the image and the rule. (Needs live check: no strangers/effects in the 50 s
  take.)
- Same seed, same first frame: GPU nondeterminism can change small details. (Deferred: acceptable; the plan only
  describes the scene.)
- Plan-only latency is dominated by the render box starting (~5-10 min when stopped). (Resolved: shown as progress.)

## Testing

Unit: `soften` and the conditioning in `generate_video` (segments 2+ blurred, segment 1 and outputs untouched, resume
key); run_video order (first frame before the planner, image passed, no image without a first frame), planner message
rule; plan-only path (no video/sound calls, result beats/seed, planner failure fails the job); API rules; CLI `plan`
(files, printed command; watcher saving plan files); dashboard in the browser preview. Live: the 50 s mirabel take
(plan first, render with the beats and seed) compared with the last run for halo, detail and invented content.
