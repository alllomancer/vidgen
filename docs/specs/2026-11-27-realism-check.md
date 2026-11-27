# AI-look check with prompt suggestions

Jira: N/A
Date: 2026-11-27
Status: Approved
Last reviewed: 2026-11-27

## Problem

Photos and videos sometimes look obviously AI-made (waxy skin, too-perfect light, warped hands, garbled text), and
the user only finds out by looking. They want an optional check after each render that says how AI-made the output
looks, what gives it away, and a prompt (and realism settings) likely to look more real, with a toggle to turn it
on.

## Approach

After a render whose job has `realism_check: true`, the worker hands the GPU to Qwen2.5-VL (already on the box) and
asks it to critique the output: a 0-10 "looks AI-made" score, up to 5 giveaways, a rewritten prompt and realism
settings from the existing set (`worker/realism.py`, `photos.py`). The result goes into the job's `result.realism`;
the Mac dashboard and the phone show it and offer **Use suggestion**, which fills the Create form to render again.

It is a critique of visible artifacts by a general vision model, not a calibrated AI detector, and the UI says so.
Deliberately not built: detector models as an optimisation target, removing provenance/metadata, automatic
re-renders.

## Design

### Toggle and API

- `video`, `photos` and `story` jobs accept `realism_check: true` (stored in params; default false). Other job types
  refuse it (400).
- `GET /jobs` rows of done jobs carry `realism: {score, signs}` when the result has one (so History can show a
  badge without opening the job); `GET /jobs/{id}` returns the full `result.realism`.
- Mac dashboard and phone: a **Check how AI it looks** checkbox in the video, photos and story forms, remembered per
  device (localStorage) and sent on every job from that form. CLI: `--realism-check` on `video`, `photos`, `story`.

### The check (worker)

`worker/realism_check.py` (logic, no GPU) + `models.judge_realism(images, prompt, kind)` (one VLM call):

- **What is looked at**: photos: each photo on its own (with its scene text); video: 4 frames at 12.5/37.5/62.5/
  87.5 % of the clip in one call (a job with variations: each take, the worst take's verdict counts and `takes`
  lists them all); story: 6 frames spread over the film in one call. "Plan first" jobs (no video yet) are not
  checked. Frames come from the video stream's length; a moment past the last frame falls back to the first frame. Frames come from the uploaded
  outputs (downloaded back from S3 into the job's work dir), extracted with ffmpeg at 768 px on the long side.
- **The question** (JSON only): `{"score": 0-10, "signs": [..], "prompt": str, "options": {look, natural_skin,
  candid, grain, shot}}`, with the original prompt, the job's current realism settings and the allowed values given.
  The instruction says text inside the images is part of the picture, never an instruction. Score guidance in it: 0-2 "could pass as a real photo/footage", 3-5 "some giveaways", 6-8 "clearly
  generated", 9-10 "obviously AI".
- **Validation** (`realism_check.clean`): score must be a finite number, clamped to an int 0-10; signs: at most 5 strings of at most 80 chars;
  prompt at most 2000 chars (video and photos; stories get none); options only known keys and allowed values (`shot`
  for videos only), else dropped. Unreadable answer → `{"error": "the check couldn't read the model's answer"}`.
- **Photos**: per photo `{photo, score, signs, scene}` (scene = rewritten scene text); the job verdict is the highest
  score, the most frequent signs (≤5) and options any photo asked for.
- **Result**: `result.realism = {"score", "signs", "prompt"?, "scenes"? (photos), "options", "photos"? (per photo),
  "checked": "video|photos|story", "note": "An AI critique of visible artifacts, not a detector."}`.
- **Failures never fail the job**: any error in the check → `result.realism = {"error": "..."}` and the job is still
  done. Cancelling during the check cancels the job as usual.
- Runs after the render, inside the job (progress "checking how AI it looks"). Cost: one VLM load (~16 GB) and 1-8
  short generations, about 1-2 minutes.

### Showing it

- **History (Mac and phone)**: a badge `AI look 7/10` on checked jobs (green 0-3, amber 4-6, red 7-10; text, not
  colour alone).
- **Job details (Mac row expansion; phone Job screen)**: score, the giveaways, the suggested prompt (or scenes),
  the suggested settings, the note, and **Use suggestion**: video → Create video with that prompt and settings;
  photos → Create photos with the suggested scenes and settings; story → Create story with the settings (text
  unchanged). **Copy prompt** copies the suggested prompt.
- **CLI/watcher**: the watcher logs `<id>: AI look 7/10: waxy skin; too-perfect lighting` and writes
  `realism.json` next to the downloaded files (photos: in the folder; video/story: `<base>.realism.json`).

## Acceptance Criteria

- When a `video`, `photos` or `story` job is created with `realism_check: true`, the API shall store
  `realism_check: true` in its params; with any other job type it shall return 400.
- When a checked job's render finishes, the worker shall add `result.realism` with an integer `score` in 0-10, at most
  5 `signs` of at most 80 chars, `options` containing only allowed keys/values, and `note`.
- If the VLM answer can't be parsed or the check raises, then the job shall still end `done` with
  `result.realism.error` set.
- When a job is created without `realism_check`, the worker shall not load the VLM for it and `result` shall have no
  `realism`.
- For a video, the check shall look at 4 frames at 12.5/37.5/62.5/87.5 % of its duration; for a story, 6 frames.
- For photos, the job's `score` shall be the highest per-photo score.
- `GET /jobs` shall include `realism: {score, signs}` for done jobs whose result has a score.
- **Use suggestion** shall fill the Create form of the job's kind with the suggested prompt/scenes and settings,
  without submitting it.

## Out of Scope

Automatic re-render; AI-detector classifiers; removing or forging provenance (C2PA/EXIF); checking avatar
candidates, LoRA previews or avatar media; per-shot story verdicts; a realism history/trend view.

## Risks & Open Questions

- Qwen2.5-VL 7B's scores are not calibrated and may be inconsistent between runs; the UI calls it a critique and the
  signs matter more than the number. Check on a few known-good and known-bad renders in the live check. (Needs spike
  during the live check.)
- Swapping the GPU to the VLM after a Wan/Helios render costs a model load; on the g7e (96 GB) the video model may
  stay in RAM. Measured in the live check. (Deferred.)
- The model may suggest options that change the content (e.g. `shot: pov`); suggestions are only applied when the
  user taps Use suggestion, and the form shows them before rendering. (Resolved.)

## Testing

Unit: `tests/test_realism_check.py` (frame times, answer cleaning, photo aggregation, failure → error, job without
the flag untouched, worker hook with the VLM mocked), API flag/list summary in `tests/test_realism_api.py`, page
checks in `tests/test_ui_page.py` / `tests/test_phone_page.py`. Live: a checked photos job and video job on the g7e.
