# Multi-image avatars

Jira: N/A
Date: 2026-11-10
Status: Approved
Last reviewed: 2026-11-10

## Problem

An avatar is one reference image. Everything the edit model knows about the person (face
from other angles, hair from behind, a sharp face when the reference is full-length) comes
from that single picture, so identity drifts in photos and avatar keyframes, and Wan then
animates whatever the keyframe shows. Users want to create an avatar from several images of
the same person and add more images to an existing avatar later.

## Approach

Multi-reference, no training: an avatar holds an ordered set of up to 12 images, and the edit
model (Qwen-Image-Edit-2511, which accepts up to 3 input images) gets the first 3 for photos
and single-avatar keyframes. A short side-by-side test on the render box comes first; if
3 references do not beat 1, the build stops there.

## Design

### Storage (S3, per client)

```
avatars/<client>/<name>/avatar.json
  {name, description, created_at,
   images: [{"key": "avatars/<client>/<name>/images/<id>.<ext>", "added_at": int}, ...],
   reference_key: images[0].key}          # kept so old code paths and history keep working
avatars/<client>/<name>/images/<id>.<ext>  # id = 8 hex chars
```

- Existing avatars (no `images`) read as a one-image set `[reference_key]`; no migration.
- Every change (create, add, remove, main) archives the current version to
  `history/<created_at>/` first, copying all of its images, so `avatar restore` undoes it.
- Limit: `MAX_AVATAR_IMAGES = 12`; same image types as today (png, jpg, jpeg, webp).

### API

| Route | Does |
|---|---|
| `POST /avatars` | as today, plus `upload_keys: [..]` (1-12) as an alternative to `upload_key` |
| `GET /avatars/{name}` | avatar with its images in order, each with a 1-hour preview URL |
| `POST /avatars/{name}/images` | `{upload_keys: [..]}` appends; 400 if the total would exceed 12 |
| `DELETE /avatars/{name}/images/{n}` | removes image n (1-based); 400 when it is the last one |
| `PUT /avatars/{name}/main` | `{index: n}` moves image n to position 1 |

Jobs: each entry in `params.avatars` gains `reference_keys` (the first 3 image keys, resolved
at submit time, so a re-run uses the same images); `reference_key` stays as image 1. Photos
jobs with an avatar get `reference_keys` the same way; `--image` uploads stay single-image.

### Worker

- `load_cast` returns `(name, [PIL images], description)`; falls back to `[reference_key]`.
- `make_photo(references, ...)` passes all references (max 3). With more than one, the
  identity sentence says "Images 1-N all show the same person; keep this exact person ...".
- `make_keyframe`: one avatar uses up to 3 of its images; 2-3 avatars use each one's image 1
  (the model takes at most 3 images per call).
- Wan is unchanged; it benefits through the better keyframe.

### CLI

- `avatar refresh --name N --image a.jpg --image b.jpg ... --describe "..."` (`--image` repeatable).
- `avatar add --name N FILE...` appends images.
- `avatar images [--name N]` lists images in order with preview links.
- `avatar remove --name N <n>` and `avatar main --name N <n>`.
- `avatar list` shows the image count per avatar.

## Acceptance Criteria

- When `POST /avatars` gets `upload_keys` with 3 of the caller's uploads, the system shall
  save an avatar whose `images` lists those 3 keys in order and whose `reference_key` is image 1.
- If `upload_keys` has 0 or more than 12 entries, or any key is not the caller's image upload,
  then the system shall return 400 and change nothing.
- When `POST /avatars/{name}/images` appends images, the system shall archive the previous
  version to history first and return the new image count.
- If an append would bring the avatar above 12 images, then the system shall return 400 and
  change nothing.
- If `DELETE /avatars/{name}/images/{n}` targets the only image, then the system shall return
  400; an out-of-range n shall return 404.
- When `PUT /avatars/{name}/main` gets `{index: 3}`, the system shall move image 3 to
  position 1 and set `reference_key` to it.
- When an avatar has no `images` field (created before this change), the system shall treat
  it as a one-image set in every route and job.
- When a video, story or photos job uses an avatar with 5 images, the system shall store the
  first 3 keys in that avatar's `reference_keys` in the job params.
- While a keyframe has 2 or 3 avatars, the worker shall pass exactly one image per avatar.
- When `avatar restore` brings back a version from before an add, the system shall restore
  that version's image set exactly.

## Out of Scope

Dashboard avatars tab; checking that all images show the same person; automatically picking
the best 3 images; multiple `--image` on `photos`; LoRA training from avatar images (the
photo-character LoRA in the LoRA training spec).

## Risks & Open Questions

- Do 3 references beat 1 on 2511 with Lightning, without duplicated people or blended
  outfits? (Resolved by spike, 2026-11-10: go.) Avatar smoke-a with 3 references (the
  full-length reference, a face close-up cropped from it, and an earlier 2511 photo of him),
  3 new scenes + 1 keyframe, same seeds. With 3 references the face matched the reference
  more closely (beard, face shape) in all 4 images; no duplicated people. One photo of 3
  (hiking) changed the trousers to shorts with 3 references. Per photo ~55 s vs ~50 s (+10%),
  keyframe 53 s vs 43 s. Caveat: the extra references came from the same generated
  character, so real photos taken at different times are the real test.
- Only the first 3 images are used, so order matters. (Resolved: documented; `main` and
  `remove` shape the set. Recommended set: full-length, face close-up, another angle.)
- More input images make each edit call slower (longer image token sequence). (Resolved:
  +10% per photo, +20% per keyframe in the spike.)
- No feature flag: one-image avatars behave exactly as before.
- The API role cannot delete S3 objects, so images removed from a set stay in
  `images/` (history also keeps a copy). (Resolved: accepted; a few MB per avatar.)

## Testing

- Spike: 1 vs 3 references for 3 photos and 1 keyframe, same seeds, timed.
- Local unit tests of the API routes against stubbed S3/DynamoDB (the existing
  test_api pattern): create with `upload_keys`, add, remove, main, limits, legacy avatars,
  `reference_keys` in job params.
- Worker: `load_cast` legacy fallback and per-shot reference selection.
- After deploy: create a 3-image avatar from the CLI, run `photos` and a 5 s avatar video.
