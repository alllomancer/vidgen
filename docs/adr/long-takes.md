# Long single takes — decision log

## D1 — [2026-11-24] — Task 2: User beats become segment prompts

**Status**: Superseded by D3
**Decision**: A beat's text replaces the take prompt for its segments (plus look text, avatar description and trigger words); it is not appended to the user's prompt.
**Reason**: The spec defines segment prompts as the beat text; appending would mix conflicting actions ("walks in" in the prompt, "sits down" in the beat). Scene continuity comes from the conditioning frame.
**Alternatives considered**: "prompt + beat" per segment — rejected for conflicting actions; the RUNBOOK asks for each beat to be a full description instead.

## D2 — [2026-11-24] — Task 2: User beats become segment prompts

**Status**: Accepted
**Decision**: The job result has `segment_prompts` only when the segments actually differ (beats, later the planner); a take that repeats its prompt has none.
**Reason**: Keeps results of ordinary takes unchanged; old CLIs and the dashboard ignore the key either way.
**Alternatives considered**: Always listing them — noise for 5-15 s takes.

## D3 — [2026-11-24] — Task 2: User beats become segment prompts

**Status**: Accepted (supersedes D1)
**Decision**: A beat's text replaces the take prompt for its segments, except that a take with no first image (no avatar, no Krea first frame) gets "prompt + beat" in segment 1.
**Reason**: Review: without a conditioning image, segment 1 is text-to-video and "walks in" alone loses the scene; later segments continue from the last frame, which carries the scene.
**Alternatives considered**: "prompt + beat" in every segment — conflicting actions; planner prompts are not prefixed (they already describe the scene).

## D4 — [2026-11-24] — Task 2: User beats become segment prompts

**Status**: Accepted
**Decision**: The API refuses beats that no segment would use: a beat after the last segment start, or two beats taking effect at the same segment (segments start every 5 s).
**Reason**: Review: "0: a; 48: b" over 50 s and "0: a; 2: b; 4: c" silently dropped beats; a 400 naming the rule is clearer than a take that ignores them.
**Alternatives considered**: Assigning beats by `at // 5` — would change the spec's "last beat with at <= 5*i" rule; warnings — nowhere visible for API/dashboard users.

## D5 — [2026-11-24] — Task 3: Automatic timeline from the planner

**Status**: Accepted
**Decision**: Planner failures (any exception or an invalid plan) are logged as `segment_plan_failed` and the take continues with the user's prompt; `JobCancelled` is re-raised.
**Reason**: Spec: the job must continue on planner failure; the VLM's `_chat` checks for cancel, which must still stop the job.
**Alternatives considered**: Failing the job — rejected by the spec.

## D6 — [2026-11-24] — Task 4: Colour matching between segments

**Status**: Accepted
**Decision**: LAB conversion is written in numpy (sRGB/D65 formulas), applied frame by frame on the CPU; statistics come from a 1/4 x 1/4 x 1/4 subsample; tiny corrections return the frames untouched; tested with real numpy in tests/test_colour.py (skipped where numpy is missing).
**Reason**: The worker has no OpenCV or scikit-image and adding one for one transform isn't worth a new pinned dependency; per-frame processing keeps memory at one frame of float LAB instead of several GB per segment.
**Alternatives considered**: OpenCV `cvtColor` (new dependency); torch on the GPU (faster, but the GPU holds Wan during the take). Cost to confirm live: roughly 20-30 s of CPU per segment.

## D7 — [2026-11-24] — Task 4: Colour matching between segments

**Status**: Accepted
**Decision**: The correction is measured on a segment's last quarter and ramps linearly from none on frame 0 to COLOUR_STRENGTH (0.7) on the last frame; the contrast gain is clamped to 0.7-1.4.
**Reason**: Review: one constant correction per segment, applied on top of an already corrected conditioning frame, moved frame 0 off the previous segment's last frame (a ~9-level step at every seam); unbounded gain amplified noise on flat frames.
**Alternatives considered**: Constant per-segment correction (the spec's wording) — seam steps; a lower strength (0.3-0.5) — left for the live check.

## D8 — [2026-11-24] — Task 5: Resume from saved segments, segment progress

**Status**: Accepted
**Decision**: Saved segments are reused only when a key of the take's settings matches (seed, frame count, size, first image or not, styled prompts, colour matching); a video job without a seed gets one derived from its job id; a planned timeline is saved next to the segments and reused on re-delivery.
**Reason**: Resuming with different settings would splice mismatched footage; a random seed or a re-run planner (bf16 sampling can differ) on re-delivery would never match the key, so resume would never happen.
**Alternatives considered**: Trusting the folder without a key — unsafe after a code or parameter change; storing the seed in the job record — an extra DynamoDB write for the same effect.

## D9 — [2026-11-24] — Task 5: Resume from saved segments, segment progress

**Status**: Accepted
**Decision**: The work folder is removed in `process()`'s `finally` (done, failed, cancelled) and, at worker start, any work folder untouched for 3 days is swept.
**Reason**: A spot stop kills the process without running `finally`, so the folder survives exactly when resume needs it; a job that is never delivered again (dead-lettered) would otherwise leave ~3 GB on the disk for good.
**Alternatives considered**: An S3 copy of the segments — survives instance loss, but the box keeps its disk on interruption ("stop" behaviour) and uploads would cost minutes per take.

## D10 — [2026-11-24] — Task 6: Optional identity re-anchoring

**Status**: Accepted
**Decision**: "Before segments k, 2k, ..." counts segments from 1 (k=3 re-anchors the frames that start segments 3, 6, 9); re-anchoring applies to takes whose first frame is an avatar keyframe or a Krea avatar frame; `anchor_every` is part of the resume key.
**Reason**: Matches the spec's acceptance criterion (segments 3, 6, 9 for k=3); a Krea-avatar take has the same one-avatar cast and drifts the same way; resuming without the anchor setting would splice differently conditioned footage.
**Alternatives considered**: 0-based counting (would anchor 4, 7, 10) — contradicts the spec.

## D11 — [2026-11-24] — Task 6: Optional identity re-anchoring

**Status**: Accepted
**Decision**: The one-avatar rule is checked in `create_job` after the avatars are resolved (the default avatar counts); range and length rules in `_validate`.
**Reason**: `_validate` does not know which avatars a job gets: with no `avatar` field the client's default avatar is used.
**Alternatives considered**: Requiring an explicit `avatar` with `anchor_every` — needlessly strict for the common default-avatar case.

## D12 — [2026-11-24] — Task 6: Optional identity re-anchoring

**Status**: Accepted
**Decision**: In colour-matched takes, a re-anchored frame is moved fully onto the LAB stats of the frame it replaced (`transfer_colours`), and each anchor uses seed + segment index.
**Reason**: Review: the image edit shifts saturation/contrast, and the next segment's ramped matching leaves frame 0 untouched, so an unmatched anchored frame would start a colour step at each anchored seam.
**Alternatives considered**: Matching the anchored frame to segment 1's stats — the previous frame is only 70% of the way there, so that would itself be a step.

## D13 — [2026-11-25] — Long-take quality Task 1: soften segment starts

**Status**: Accepted
**Decision**: The frame each later segment starts from is blurred 0.8 px (PIL Gaussian), never the first frame, a freshly re-anchored frame or the output frames; the radius is part of the resume key.
**Reason**: Live: a bright halo built up around the subject over 10 seams; Wan I2V reproduces and slightly strengthens the edge contrast of its start frame, so softening only that frame stops the compounding without touching what the viewer sees. Re-anchored frames come fresh from the image editor (no build-up), and blurring them would undo part of the face fix (review).
**Alternatives considered**: Starting from an earlier frame (the halo is in every frame of the segment); sharpening control in Wan (not exposed).

## D14 — [2026-11-25] — Long-take quality Tasks 2-3: plan with the first frame, plan-only jobs

**Status**: Accepted
**Decision**: Single takes make the first frame before planning and give it to the planner with a keep-to-the-frame rule; `plan_only` returns that plan as beats (";" turned into "," so the API's beat parser accepts them back), the seed and the frame, and a planner failure fails the job instead of falling back.
**Reason**: Live: the text-only planner invented strangers on the water and glitter; users only saw the plan after an hour. A plan-only job that silently fell back to the prompt would give the user nothing to review.
**Alternatives considered**: Planning after the first segment (too late to change the plan); a separate `plan` job type (duplicates video validation; `video` + `plan_only` reuses it).

## D15 — [2026-11-25] — Long-take quality: halo blur off

**Status**: Accepted (supersedes D13's default)
**Decision**: `CONDITION_BLUR` is 0 (off); the softening code stays, switchable, while the halo's cause is investigated.
**Reason**: Live check: with 0.8 px the halo at 47.5 s was as strong as without, and faces/coats showed over-sharpened texture; the edge-amplification explanation was wrong or incomplete.
**Alternatives considered**: A stronger blur (would soften detail further without evidence it addresses the cause).

## D16 — [2026-11-25] — Long-take quality: the halo is lighting, not chaining

**Status**: Accepted
**Decision**: No pipeline fix for the halo; `CONDITION_BLUR` stays off; the RUNBOOK advises side or front light for close shots of people at golden hour.
**Reason**: A front-lit 30 s take (6 chained segments, fixed framing) showed no outline and no build-up; a backlit one had the glow from the first frame, growing as Wan moved the camera in. Measurements across seams showed no step at the joins.
**Alternatives considered**: Post-processing to suppress glow (would fight intended golden-hour looks).
