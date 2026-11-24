# Long single takes (up to 60 s) — Implementation Plan

Spec: docs/specs/2026-11-24-long-takes.md
Workspace: current folder (~/Documents/private/myproject; not a git repository, so no commits or SHAs)
Jira: N/A

## Progress

- [x] Task 1: 60 s cap end to end — reviewed (no git; snapshot diff)
- [x] Task 2: User beats become segment prompts — reviewed, advisories fixed (D3, D4)
- [x] Task 3: Automatic timeline from the planner — reviewed, advisories fixed (tolerant JSON, real length)
- [x] Task 4: Colour matching between segments — reviewed; seam-step blocker fixed with a ramped correction (D7)
- [x] Task 5: Resume from saved segments, segment progress — reviewed; hardened (save failures, versioned key, best-effort progress)
- [x] Task 6: Optional identity re-anchoring — reviewed; anchored frames colour-transferred (D12)
- [x] Task 7: Dashboard fields — reviewed, browser-verified, accessibility advisories fixed
- [x] Task 8: Docs, deploy and live 50 s check — deployed; live 50 s take OK for length, colour and seams; likeness drifts without re-anchoring

## Tasks

### Task 1: 60 s cap end to end

**What**: `MAX_TAKE_SECONDS = 60` in the API and worker; the API error reads "seconds must be 1-60 ...";
variations stay limited to takes of up to 5 s. CLI `--seconds` help and the submit message give a time
estimate (~5 min per 5 s segment plus sound) for takes over 15 s.
**Files**: lambda/api/handler.py, worker/models.py, cli/vidgen.py, tests/test_api.py
**Depends on**: none
**Verify**: `make test`: 50 s accepted, 61 s refused with "seconds must be 1-60", variations > 1 with 6 s refused;
the worker's chaining test renders 10 segments for 50 s (stubbed `_render_segment`) with 801 frames total.

### Task 2: User beats become segment prompts

**What**: API `beats` (string "SEC: text; SEC: text" or newline-separated, or list `[{at, prompt}]`):
parsed to a list, rules first at 0, increasing, `at` < seconds, 1-12 beats, 1-500 chars, only for takes over
5 s (400 naming the rule). Worker `models.segment_prompts(prompt, n, beats)` picks, per segment i, the last beat
with `at <= 5*i`; `generate_video` takes `segment_prompts` (one per segment) instead of one prompt, keeping
`with_triggers` and the look text per segment; the job result lists `segment_prompts`. CLI `--beats`,
`--beats-file`.
**Files**: lambda/api/handler.py, worker/models.py, worker/worker.py, cli/vidgen.py, tests/test_api.py, tests/test_worker.py
**Depends on**: Task 1
**Verify**: `make test`: "0: walks in; 10: sits down; 30: looks up" over 50 s gives segments 0-1/2-5/6-9 those
texts; each refusal returns 400; each segment's rendered prompt carries the trigger words; the CLI sends parsed
beats from `--beats` and `--beats-file`.

### Task 3: Automatic timeline from the planner

**What**: Without beats and with more than 3 segments, `segment_prompts` asks the planner model
(`models.plan_segments(prompt, n, cast)`, Qwen2.5-VL like `plan_story`) for n prompts that keep subject, place,
light and camera and move the action on; validated (n non-empty strings, each <= 500 chars), else every segment
uses the user's prompt (logged). The planner runs before the video model loads (as stories do).
**Files**: worker/models.py, worker/worker.py, tests/test_worker.py
**Depends on**: Task 2
**Verify**: `make test` with a stubbed planner: 10 prompts used in order and stored in the result; a wrong count, an
exception or an empty prompt falls back to the user's prompt for every segment; takes of up to 15 s don't call the
planner.

### Task 4: Colour matching between segments

**What**: `models.match_colours(frames, reference)`: per-channel LAB mean/std of the segment moved towards segment
1's (strength 0.7), applied to all frames of segments after the first, before the last frame conditions the next
segment; on for takes over 10 s.
**Files**: worker/models.py, tests/test_worker.py
**Depends on**: Task 1
**Verify**: `make test`: a shifted/darker segment's LAB stats move ~70% towards the reference; equal stats leave
frames unchanged; in the chaining test, the conditioning frame of segment 2 is the colour-matched one; takes of
10 s or less are not matched.

### Task 5: Resume from saved segments, segment progress

**What**: `generate_video` gets a `work_dir`: each finished segment (after colour matching) is saved as
`segment_NN.npz` plus `segments.json` (seed, prompts, count); on start it loads the finished ones and continues
from the last frame. The worker passes `/opt/ai-video/work/<job_id>/` for takes over 5 s and removes it when the job
ends (done, failed, cancelled). Progress "segment N/M".
**Files**: worker/models.py, worker/worker.py, tests/test_worker.py
**Depends on**: Task 4
**Verify**: `make test`: a run stopped after segment 6 of 10 and started again renders only segments 7-10 and the
frames equal an uninterrupted run (stubbed renderer); the folder is removed on done and on failure; progress
messages read "segment 1/10" ... "segment 10/10".

### Task 6: Optional identity re-anchoring

**What**: API `anchor_every` (int 2-6) only with exactly one avatar and seconds > 10 (else 400). Worker: before
segments k, 2k, ... the conditioning frame goes through `models.reanchor(frame, avatar_refs, w, h)`
(Qwen-Image-Edit: keep the frame, restore this person's face and hair), fitted to the take size. CLI
`--anchor-every N`.
**Files**: lambda/api/handler.py, worker/models.py, worker/worker.py, tests/test_api.py, tests/test_worker.py
**Depends on**: Task 5
**Verify**: `make test`: anchoring calls at segments 3, 6, 9 for k=3 over 50 s with the avatar's images; none
without `anchor_every`; API refusals for 0 or 2 avatars and for takes of 10 s or less.

### Task 7: Dashboard fields

**What**: Video form: seconds input max 60 (variations off above 5 s, as today); "Beats (optional)" textarea shown
above 5 s; "Re-anchor the avatar every N segments" shown above 10 s with one avatar; time estimate text.
`createBody` sends `beats` and `anchor_every`.
**Files**: cli/ui.html
**Depends on**: Task 6
**Verify**: browser preview: fields appear/disappear with seconds and avatar; the body carries `seconds: 50`,
`beats` and `anchor_every`; no console errors.

### Task 8: Docs, deploy and live 50 s check

**What**: RUNBOOK (long takes, beats format, anchoring, time, resume) and README; spec live-check section.
`make plan` / `make apply` after the user's go-ahead; restart the worker. Live: a 50 s take with an avatar and
automatic beats; compare the first and last segments for colour and likeness and check the seams.
**Files**: RUNBOOK.md, README.md, docs/specs/2026-11-24-long-takes.md
**Depends on**: Task 1, Task 2, Task 3, Task 4, Task 5, Task 6, Task 7
**Verify**: `make test` passes; the live job completes as 50 s with sound, the result lists 10 segment prompts; frames
from segment 1 and segment 10 compared side by side; watch host RAM on a 60 s take with fps 32 + upscale
(review estimate ~47 GB peak on top of the resident models).
