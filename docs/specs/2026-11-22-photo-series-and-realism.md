# Photo series in one location, realism options (photos, video, story)

Jira: N/A
Date: 2026-11-22
Status: Approved
Last reviewed: 2026-11-22

## Problem

`photos` places the avatar in a different (random or given) scene per photo; there is no way to
get a set of shots from one place, like a photo shoot. Photos also all use one fixed "pro 50mm"
look on the 8-step Lightning edit model, which skips CFG (so the anti-plastic-skin negative
prompt does nothing) and can look too clean.

## Approach

Choices made 2026-11-22:

- **Series**: N photos (2-8) in one place and outfit, pose and camera varied automatically (or
  one action per photo if given). Photos 2..N are anchored on photo 1: it is passed as an extra
  reference image so the place, light and outfit stay close.
- **Realism options** (single photos and series): look presets, best-quality mode, natural skin,
  candid, real film grain. Standard speed stays the default; best is opt-in.
- Added the same day at the user's request: the realism options on `video` and `story` too.

## Design

### API (`type: photos`)

- `series: {where (required), wearing?, doing?: [str, ...] (at most count), anchor?: bool (default true)}`.
  With `series`, `scenes` must be empty; `count` 2-8, default 4. Each text 1-500 chars.
- `look`: `pro` (default, today's wording) | `phone` | `candid` | `portrait` | `film`.
- `natural_skin`, `candid`, `grain`: booleans, default false.
- `quality`: `standard` (default) | `best` (named like video's `quality`).
- The job summary shows the series place, look and quality.

### Worker

- `photos.build_series(series, count, seed)`: photo 1 is a full-body, eye-level shot with the
  place clearly visible (a good anchor); the rest take the given actions in order, then
  seed-sampled entries from `SERIES_SHOTS` (pose + camera pairs, no repeats).
- Anchor: photos 2..N use the avatar's first 2 images plus photo 1 (the model takes at most 3).
  The prompt says the last image is an earlier photo from the same shoot: keep the place,
  lighting and outfit; new pose and camera as described. `anchor: false` = text only, 3 avatar images.
- `photo_prompt(..., look, natural_skin, candid, anchor_index)`: look sentence from `LOOKS`;
  natural skin adds visible pores / no retouching; candid adds "unposed, not looking at the camera".
- `quality: best`: the Lightning adapter is disabled for that call and the full 40-step model
  with true CFG 4.0 runs, so the negative prompt applies (~2.2 min per photo vs ~45 s).
- `grain`: after generation, mild luminance noise (sigma ~3.5/255, seeded) and a slight softening,
  applied to the PNG on the worker.

### Video and story

- Same params `look`, `natural_skin`, `candid`, `grain`; `quality` gains `best` next to
  `standard` / `draft` (draft and best exclude each other).
- `worker/realism.py`: video wording per look (`pro` adds nothing, so prompts are unchanged),
  appended to the Wan prompt and to the avatar keyframe scene (story: every shot and keyframe).
  The sound prompt stays the raw prompt.
- `quality: best` makes the avatar keyframes with the full 40-step edit model. Wan keeps Lightning:
  its full 40-step mode would be ~10x slower per clip.
- `grain`: a new seeded noise field per frame on the final frames (after RIFE and upscale),
  applied while writing the mp4; the 16 fps clip for the sound model is untouched.

### CLI and dashboard

- `vidgen photos --series --where … [--wearing …] [--doing … (repeatable)] [--no-anchor] --count N`;
  `--look`, `--natural-skin`, `--candid`, `--grain`, `--best` on photos, video and story.
- Dashboard Photos form: "Series in this place" checkbox (actions box, one per line), Look select,
  checkboxes Natural skin / Candid / Film grain / Best quality, and a time estimate. Video and
  Story forms get the same Realism section (Best = best keyframes).

## Acceptance Criteria

- When a series of 5 runs with anchor on, photo 1 shall use up to 3 avatar images and photos 2-5
  shall use up to 2 avatar images plus photo 1, all with the series place in the prompt.
- When `series.doing` has 2 actions and count is 4, photos 1-2 shall use those actions in order
  (photo 1 keeping the full-body, place-visible framing) and photos 3-4 shall use two distinct
  auto shots.
- If `series` has no `where`, is combined with `scenes`, has count below 2, or more actions than
  count, then the API shall return 400.
- If `look` or `quality` is not one of the listed values, then the API shall return 400.
- When `quality: best` is set, the edit call shall run 40 steps with true CFG 4.0 and the Lightning
  adapter disabled, and re-enable it afterwards.
- When `grain` is set, the saved photo shall differ from the raw output by seeded noise only
  (same seed gives the same result).
- When no realism option is set, the photo, video, keyframe and story prompts shall be exactly
  today's prompts.
- When a video has `look: film` and an avatar, the keyframe scene and the Wan prompt shall both end
  with the film sentence; with `grain`, the final clip (not the 16 fps sound clip) shall get grain.
- If a video or story has both `draft` and `best`, then the CLI and dashboard shall refuse it.

## Out of Scope

Real-photo upscaling of stills, face restoration models, per-photo looks inside one job, series
across several places.

## Risks & Open Questions

- Anchoring may pull identity towards photo 1's rendering of the person (2 avatar images instead
  of 3). Mitigation: `anchor: false`. (Check in the live test.)
- `disable_lora`/`enable_lora` on the Qwen edit pipeline with CPU offload. (Check live; fallback:
  set the adapter weight to 0.)
- Best quality is ~3x slower per photo; an 8-photo best series is ~18 min.

## Testing

Unit tests in `tests/` (prompt building, series shots, anchor references, API validation, CLI
flags, dashboard body). Live: a 4-photo anchored series fast, the same with `--best --look phone
--natural-skin --grain`, compared by eye.

## Update: live checks (2026-11-22)

Avatar smoke-b, 4-photo series at a Lisbon café terrace, seed 21, landscape.

- Fast series: 428 s including the edit-model load. Anchored series: the same place, light, outfit
  and face in all 4 photos.
- Same series with `--best --look phone --natural-skin --grain`: 716 s (~2.2 min per photo, so
  the 40-step mode ran; `disable_lora`/`enable_lora` worked with CPU offload, no errors). Colours
  are softer and less saturated than the fast run, with more natural light and skin; the grain is
  subtle.
- Video `--look film --grain --best`, 3 s: 424 s, 1280x704, 97 frames at 32 fps, warm film look,
  best keyframe, no errors.
- Found: the anchor is too strong. Photos 2-4 copy photo 1's background almost exactly and the
  camera barely moves ("wide shot" and "head and shoulders" come out from the same viewpoint;
  one photo kept the menu pose). The poses vary but the camera angles don't yet.
- Fix (same day): the anchor prompt now uses photo 1 only for the place's look, light and outfit
  and says not to copy its composition, camera, pose or props; shots seen from far away or another
  side (`SERIES_SHOTS` anchored = False) get no anchor image, only the text. Your own actions get
  varied cameras. Re-run, same seed: the waist-up and head-and-shoulders shots are real close-ups
  with the same terrace blurred behind; the wide shot shows the terrace from a different viewpoint
  and still reads as the same place. Identity and outfit hold in all 4. The "far, small in frame"
  shot still frames the person larger than asked.
