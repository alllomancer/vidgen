# Long-take quality — Implementation Plan

Spec: docs/specs/2026-11-25-long-take-quality.md
Workspace: current branch (avatar-media-and-wan-lora)
Jira: N/A

## Progress

- [x] Task 1: Soften the frame each segment starts from — reviewed (anchored frames excepted, D13)
- [x] Task 2: First frame before planning, shown to the planner — reviewed
- [x] Task 3: Plan-only video jobs — reviewed; ';' blocker fixed (D14)
- [x] Task 4: CLI `vidgen plan` and plan files — reviewed; command quoting blocker fixed
- [x] Task 5: Dashboard "Plan first" — reviewed, browser-checked; CSP fixed for S3 previews
- [ ] Task 6: Docs, deploy and live check — docs done; deploy waiting for go-ahead

## Tasks

### Task 1: Soften the frame each segment starts from

**What**: `models.CONDITION_BLUR = 0.8`, `models.soften(image, radius)` (PIL GaussianBlur); in `generate_video`, takes
over 5 s soften the conditioning image of segments 2+ after re-anchoring/colour transfer, before `_render_segment`;
the radius joins the resume key.
**Files**: worker/models.py, tests/test_worker.py
**Depends on**: none
**Verify**: `make test`: segments 2..10 of a 50 s take get `soften(...)` of the previous last frame, segment 1 the first
frame unchanged; written frames unchanged; takes of 5 s not softened; changing the radius invalidates saved segments.

### Task 2: First frame before planning, shown to the planner

**What**: `make_first_frame(job, params, cast, seed)` (Krea frame for `krea2`, else the Qwen keyframe with a cast, else
None); `run_video` (single take) makes it before `take_timeline`, passes its path to the planner and the frame to
`render_take` (which no longer makes the keyframe itself when given one); `plan_segments(..., image=path)` puts the
image first and adds the keep-to-the-frame rule.
**Files**: worker/worker.py, worker/models.py, tests/test_worker.py
**Depends on**: Task 1
**Verify**: `make test`: call order first frame -> planner -> video; the planner gets the keyframe path and the rule; no
image without a first frame; keyframe file and result keys as before; variations unchanged.

### Task 3: Plan-only video jobs

**What**: API `plan_only` (bool; only seconds > 15, not with beats or variations > 1; summary "plan"); worker
`run_video` with `plan_only` makes the first frame and the plan, no video or sound; result `{plan_only, segment_prompts,
beats, seed, keyframe_key?}`; planner failure fails the job with the message.
**Files**: lambda/api/handler.py, worker/worker.py, tests/test_api.py, tests/test_worker.py
**Depends on**: Task 2
**Verify**: `make test`: API 400s; plan result for 50 s (beats lines 0..45, seed), no `generate_video`/sound calls;
planner failure raises with the message.

### Task 4: CLI `vidgen plan` and plan files

**What**: `vidgen plan "PROMPT" --seconds N [video flags]` submits a plan-only job and waits (progress; Ctrl-C leaves it
to the watcher), then writes `<name>.beats.txt` and `<name>.keyframe.png` and prints the beats and the render command
with `--seed` and `--beats-file`; `fetch_results` saves the same files for plan jobs.
**Files**: cli/vidgen.py, tests/test_trainer_cli.py
**Depends on**: Task 3
**Verify**: `make test`: body has `plan_only` and the video flags; beats file content; printed command; watcher saves
plan files.

### Task 5: Dashboard "Plan first"

**What**: Video form button "Plan first" shown above 15 s; submits plan-only, polls the job, then fills Beats and Seed
and shows the first frame; errors in the status line.
**Files**: cli/ui.html, cli/ui.py
**Depends on**: Task 3
**Verify**: browser preview with a stubbed job result: fields filled, image shown, no console errors.

### Task 6: Docs, deploy and live check

**What**: RUNBOOK/README, ADR, spec live check. Deploy after the go-ahead. Live: `vidgen plan` for the 50 s mirabel take,
render with the beats and seed; compare halo, detail and invented content with the last run.
**Files**: RUNBOOK.md, README.md, docs/adr/long-takes.md, docs/specs/2026-11-25-long-take-quality.md
**Depends on**: Task 1, Task 2, Task 3, Task 4, Task 5
**Verify**: `make test`; the live plan returns 10 beats that stay in the scene; the take shows less halo than the last
run and no invented people or effects.
