# Long single takes (up to 60 s)

Jira: N/A
Date: 2026-11-24
Status: Approved
Last reviewed: 2026-11-24

## Problem

A single video take is limited to 15 s (`MAX_TAKE_SECONDS`): three 5 s Wan 2.2 segments, each rendered
image-to-video from the previous segment's last frame. The user wants a 50 s single clip. Chaining 10
segments with one prompt makes the action repeat or stall, colours and the person's face drift from segment
to segment, and a spot interruption 50 minutes into a take throws the whole render away.

## Approach

Keep the chained-segment pipeline and make it hold up over 10-12 segments:

1. Cap 60 s (API, worker, CLI, dashboard).
2. A timeline: one prompt per segment, written by the planner model from the user's prompt, or the user's
   own beats ("0: walks in; 10: sits down; 30: looks up").
3. Colour matching of every segment to the first.
4. Optional identity re-anchoring every N segments for takes with one avatar (off by default).
5. Finished segments saved on the box's disk so a re-delivered job resumes.

Already in place and unchanged: sound in overlapping 8 s windows (`sound_for_clip`), RIFE/upscale in RAM
(a 50 s take is ~800 frames at 16 fps, ~1,600 at 32 fps, ~17 GB after a 2x upscale; the box has 128 GB).

## Design

### API (`lambda/api/handler.py`, video jobs)
- `MAX_TAKE_SECONDS = 60`. Variations stay limited to takes of up to 5 s (unchanged).
- `beats`: optional string or list. String form `"0: walks in; 10: sits down; 30: looks up"` (`;` or newline
  separated, `SECONDS: text`); list form `[{"at": 0, "prompt": "..."}]`. Rules: first beat at 0, strictly
  increasing, each `at` < `seconds`, 1-12 beats, each text 1-500 chars; otherwise 400 naming the rule.
  Only for takes over 5 s.
- `anchor_every`: optional int 2-6, only with exactly one avatar and seconds > 10 (else 400).
- Both stored in params; the job summary shows `seconds`, `beats` count and `anchor_every`.

### Worker
- `models.MAX_TAKE_SECONDS = 60`.
- **Segment prompts** (`models.segment_prompts(prompt, n, beats, cast)`):
  - With beats: segment i (start `5*i` s) takes the last beat with `at <= 5*i`, prefixed by the take's
    look/realism text as today.
  - Without beats and n > 3: the planner model (Qwen2.5-VL, as for stories) writes n short prompts from the
    user's prompt: same subject, place, light and camera style; only the action moves on, continuing from
    the previous segment. Output validated (n non-empty strings); on any planner failure, every segment uses
    the user's prompt (logged, job continues).
  - n <= 3 without beats: the user's prompt for every segment (today's behaviour).
  - LoRA trigger words and the avatar name stay in every segment prompt (`with_triggers`).
  - The job result lists `segment_prompts`.
- **Colour matching** (`models.match_colours(frames, reference)`): per segment after the first, the mean and
  standard deviation of each LAB channel are moved towards those of segment 1's frames (strength 0.7,
  computed once per segment, applied to all its frames). The next segment is conditioned on the corrected
  last frame, so the correction never accumulates. `colour_match` is on for takes over 10 s.
- **Identity re-anchoring** (`anchor_every = k`): before segments k, 2k, ... the conditioning frame (previous
  last frame) goes through Qwen-Image-Edit with the avatar's first images and the instruction to keep the
  frame identical and restore this person's face and hair; the result is fitted to the take size. Off unless
  `anchor_every` is given.
- **Resume**: each finished segment's frames are written to `/opt/ai-video/work/<job_id>/segment_NN.npz`
  (16 fps frames after colour matching) plus `segments.json` (count, seed, prompts). On a re-delivered job
  the worker loads the finished segments and continues from the last one's final frame. The folder is
  removed when the job finishes (done, failed or cancelled). Disk: ~270 MB per segment at 1280x704, ~3 GB for
  a 60 s take, on the 500 GB root volume.
- **Progress**: "segment 4/10", then "smoothing", "adding sound" as today.

### CLI / dashboard
- `vidgen video "..." --seconds 50 [--beats "0: ...; 10: ..."] [--anchor-every 3]`; `--beats-file F`
  (one `SECONDS: text` per line).
- Dashboard Video form: seconds input max 60 (variations disabled above 5 s, as today), a "Beats (optional)"
  textarea shown above 5 s, and "Re-anchor the avatar every N segments" shown above 10 s with one avatar.
- Time estimate in the CLI output and dashboard: ~5 min per 5 s segment plus sound.

## Acceptance Criteria

- When a video job has `seconds: 50`, the API shall accept it, and when it has `seconds: 61`, the API shall
  return 400 "seconds must be 1-60".
- When `beats` is "0: walks in; 10: sits down; 30: looks up" with `seconds: 50`, the job params shall hold
  three beats at 0, 10 and 30 s; `segment_prompts` shall give segments 0-1 "walks in", 2-5 "sits down" and
  6-9 "looks up" (each with the look text).
- If beats don't start at 0, are not increasing, have an `at` >= seconds, number more than 12, or are given
  for a take of 5 s or less, then the API shall return 400 naming the rule.
- When a 50 s take has no beats, the worker shall ask the planner for 10 segment prompts and store them in
  the result; if the planner fails or returns the wrong count, every segment shall use the user's prompt and
  the job shall continue.
- The prompt of every segment shall contain the job's LoRA trigger words and, with an avatar, its name.
- When a take is over 10 s, every segment after the first shall be colour matched to segment 1 before its
  last frame conditions the next segment.
- When `anchor_every` is 3 on a take with one avatar, the conditioning frames of segments 3, 6 and 9 shall be
  re-anchored with the avatar's images; `anchor_every` with zero or two avatars shall return 400.
- If a 50 s job is re-delivered after segment 6 finished, then the worker shall render only segments 7-10
  and the result shall have 10 segments; the work folder shall be removed when the job ends.
- The progress shall read "segment N/10" while rendering.

## Out of Scope

New long-video models (FramePack, SkyReels-V2); smoothing motion across segment seams (one-frame
continuation can change the pace every 5 s); takes over 60 s; stories (shots stay up to 5 s); multi-avatar
re-anchoring.

## Risks & Open Questions

- Motion seams every 5 s: Wan I2V continues from one frame, so speed and direction can change at seams.
  (Needs live check; if clearly visible, spike FramePack-F1 / SkyReels-V2.)
- Re-anchoring may cause a visible jump at the anchored seams. (Needs live check; off by default.)
- Colour matching can flatten intended light changes (sunset over 50 s). Strength 0.7 and only above 10 s;
  beats that describe a light change could lower it later. (Deferred.)
- Render time: ~50-60 min for 50 s on the g6e.4xlarge; max uptime (8 h) and job timeout (6 h) are not hit.
  (Resolved.)
- Spot interruption: resume from saved segments. (Resolved.)

## Testing

Unit: API validation (cap, beats string/list, every refusal, anchor_every rules); `segment_prompts` (beats,
planner, planner failure, triggers); `match_colours` (moves stats towards the reference, identity when
equal); `generate_video` chaining with stubbed segments (prompt per segment, colour match applied before
conditioning, anchoring at k, 2k, resume from saved segments, cleanup); CLI `--beats`/`--beats-file`;
dashboard fields (browser preview). Live: one 50 s take with an avatar and automatic beats; compare first and
last segments for colour and likeness, check the seams.

## Changes during implementation

Recorded in docs/adr/long-takes.md: beats must fall on distinct segments (times in multiples of 5) and segment 1
also gets the prompt when the take has no first image (D3, D4); colour matching ramps from none at each seam to
0.7 on the segment's last frame, with clamped contrast (D7); resume uses a versioned settings key and a seed derived
from the job id (D8); re-anchored frames keep the colours of the frame they replace (D12).

## Live check (2026-11-24)

Job d7f73dd940b04d06858cf3fed5128133: `vidgen video "mirabel walks slowly along a wooden pier at golden hour, ..."
--seconds 50 --avatar mirabel` (automatic timeline, no re-anchoring), g6e.4xlarge, Lightning.

- Result: 50.03 s, 1280x704, 32 fps (1,601 frames), sound; render 54 min (start to finish on the box). The result
  lists 10 planner-written segment prompts (same subject and place, the action moves on; one implausible beat:
  "touching the water" from the pier).
- Colour: held. Sampled every 5 s: luma 144-148, chroma within about ±1, saturation 20.8-22.5; no colour step at the
  seams checked (5, 25, 45 s).
- Seams: frames 0.1 s either side of each checked seam match in pose, framing and light; no visible jump.
- Likeness: drifts. Segment 1 shows short red curls; by 25 s the hair is darker auburn and the face differs; the camera
  ends much closer than the prompt suggests. A bright halo around the coat builds up in the last segments.
  `--anchor-every` (off in this run) is the intended fix; it needs its own live check.
- Not measured: host RAM for a 60 s take with upscale (this run had no upscale).
