# Subject photos: one person in many places

Jira: N/A
Date: 2026-11-09
Status: Approved
Last reviewed: 2026-11-09

## Problem

Users want photorealistic still photos of a specific person (a photo they supply, or a
saved avatar) in several different places, either places/scenes they describe or
random ones, without making a video.

## Approach

A new `photos` job type that runs the Qwen-Image-Edit-2509 model already used for avatar
keyframes, once per scene, with a photorealism prompt that keeps the person identical.
No new models, no infrastructure changes.

## Design

- **Input**: an uploaded image (`upload_key` from `POST /uploads`: png, jpg, jpeg, webp)
  or `avatar` (one saved avatar name). With neither, the client's default avatar is used;
  with no default either, the API returns 400.
- **Scenes**: `scenes` is an ordered list of `{"prompt": str}` (free-form scene: place,
  action, outfit details, mood, lighting) or `{"location": str}` (rendered as "in <X>,
  natural relaxed pose suited to the setting"). Each text is 1-500 chars.
- **Count**: `count` (1-8) is the total number of photos; default `len(scenes)` or 4 when
  no scenes are given. Extra photos beyond `scenes` are random locations drawn without
  replacement from a curated list of about 50 people-friendly settings (worker
  `photos.LOCATIONS`), skipping locations the user gave; `seed` makes the draw and the
  renders reproducible. `count < len(scenes)` and `len(scenes) > 8` are rejected with 400.
- **Aspect**: `aspect` = portrait (896x1152, default) | landscape (1152x896) |
  square (1024x1024).
- **Prompt** (per photo): keep-identity sentence (face, facial features, hair, skin tone,
  body shape, outfit) + scene + photo style (full-frame camera, 50 mm lens, natural light
  matching the scene, realistic skin texture, accurate shadows and perspective, sharp
  focus). Negative prompt: cartoon, anime, illustration, CGI, 3d render, painting,
  plastic/airbrushed skin, oversaturated, distorted face, deformed hands, extra
  fingers/limbs, blurry, watermark, text. 40 steps, true_cfg_scale 4.0, seed + i per photo
  (superseded 2026-11-10: see "Update: 2511 + Lightning" below).
- **Output**: `photo_<n>_key` per photo (downloads `photo_1` ... `photo_N`) and
  `result.scenes` = [{"photo": n, "source": "prompt"|"location"|"random", "text": str}].
  The CLI saves `photo_<n>.png` and `prompts.txt` into the output folder
  (`-o NAME`, default `photos-<job id>`).
- **Progress / cancel**: progress "photo i/N: <text>"; cancel works between denoising steps.
- **CLI**: `vidgen photos [--image FILE | --avatar NAME] [--prompt P]... [--location L]...
  [--prompts-file FILE] [--count N] [--aspect A] [--seed S] [-o DIR]`. `--prompt` and
  `--location` keep the order given; `--prompts-file` lines (blank lines and `#` comments
  skipped) are appended after them.
- **Smoke**: step `photos`: 3 photos of smoke-a (one prompt, one location, one random),
  queued right after avatar-vid so it shares that edit-model load.

## Acceptance Criteria

- When a `photos` job with 1 prompt, 1 location and count 3 finishes, the system shall
  return downloads `photo_1`, `photo_2`, `photo_3` (PNG) and `result.scenes` with sources
  prompt, location, random in that order.
- When `aspect` is portrait, landscape or square, each photo shall be 896x1152, 1152x896
  or 1024x1024 respectively.
- If `scenes` has more than 8 entries, `count` is outside 1-8 or less than `len(scenes)`,
  a scene text is empty or over 500 chars, or `aspect` is unknown, then the API shall
  return 400.
- If both `upload_key` and `avatar` are given, then the API shall return 400.
- If `upload_key` is not the client's own image upload, then the API shall return 400.
- When the same `seed` and inputs are given, the random locations shall be the same.
- The random fill shall never repeat a location within one job or repeat a location the
  user supplied.
- When `vidgen photos` finishes, the CLI shall save `photo_<n>.png` and `prompts.txt`
  (one line per photo: `photo_<n>.png [source] text`) in the output folder.

## Out of Scope

VLM-chosen locations, non-person subjects, more than one person per photo, outfit
changes beyond what a prompt asks for, the Qwen-Image Lightning speed-up, turning photos
into videos.

## Risks & Open Questions

- Identity drift (Resolved: accepted). Qwen-Image-Edit keeps identity well but not
  perfectly, same as keyframes; the README recommends a sharp, well-lit, single-person
  reference photo.
- A free-form prompt can contradict "keep the outfit identical" (Resolved: the prompt
  wins in practice; documented).
- Runtime (Resolved): about 1.5-2 min per photo + 1-2 min edit-model load on g6e.4xlarge;
  8 photos stay well inside the job timeout.
- No feature flag needed: new job type, no change to existing paths.

## Update: 2511 + Lightning (2026-11-10)

Photos and avatar keyframes moved from Qwen-Image-Edit-2509 at 40 steps to
Qwen-Image-Edit-2511 with the `lightx2v/Qwen-Image-Edit-2511-Lightning` 8-step LoRA
(Apache-2.0, pinned revision), true_cfg_scale 1.0, so the negative prompt no longer applies.
2511 needs diffusers >= 0.37 (`zero_cond_t`); the worker moved to diffusers 0.37.1.
`EDIT_LIGHTNING` = 8 (default) / 4 / 0 (full 40 steps with CFG) in `/etc/ai-video.env`.

Side-by-side on the L40S, job f8aa6a8f reproduced (avatar smoke-a, seed 3, 3 portrait scenes),
per photo after the first:

| Config | Per photo | Peak GPU |
|---|---|---|
| 2509, 40 steps (old) | ~137 s | 39 GB |
| 2511, 40 steps | ~132 s | 39 GB |
| 2511 + Lightning 8 (new default) | ~38 s | 40 GB |
| 2511 + Lightning 4 | ~31 s | 40 GB |

All four kept the person recognisable. 2511 gave more natural poses and lighting than 2509;
Lightning adds a little saturation and contrast, more at 4 steps. A 1-frame Wan T2V render on
diffusers 0.37.1 passed in the same test.

## Testing

- Local unit checks of the pure scene builder (`worker/photos.py`) and API validation.
- Smoke `photos` step on the real GPU: 3 PNGs downloaded, sizes 896x1152, sources
  prompt/location/random.
- Manual look at the photos for realism and identity.
