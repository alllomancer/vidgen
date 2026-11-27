# Upload an avatar's LoRAs

Jira: N/A
Date: 2026-11-27
Status: Approved
Last reviewed: 2026-11-27

Approved in conversation: Krea 2 and Wan LoRAs from one dialog; refuse while the avatar's build runs.

## Problem

Every avatar has two LoRAs, trained on the trainer: a Krea 2 LoRA (`avatar-<name>`, one file; Krea photos and first
frames) and a Wan LoRA (`avatar-<name>-wan`, one file or a high-noise + low-noise pair; used automatically in videos
starring the avatar). LoRAs trained elsewhere can't be attached to an avatar: Learning > Add a LoRA file refuses
`avatar-*` names (docs/specs/2026-11-23-lora-upload.md), and only the trainer marks an avatar's LoRA ready.

## Approach

An **Upload LoRAs** button on each avatar card (and `vidgen avatar lora`) uploads a Krea 2 file and/or a Wan file
(or pair) as the next version of that avatar's own LoRA names, checked like any uploaded LoRA, and marks them
ready on the avatar, exactly as a finished training run would.

## Design

- API (reuses the LoRA upload checks and storage; only the avatar's own names, only these routes):
  - `POST /avatars/{name}/loras/{kind}/uploads {files: single|pair, sizes}`, kind `krea` | `wan` -> presigned
    upload like `POST /loras/{name}/uploads`, for `avatar-<name>` (krea, one file) or `avatar-<name>-wan` (wan).
  - `PUT /avatars/{name}/loras/{kind}/versions/{version}` -> the usual checks (Krea 2 file must be a Krea 2 LoRA,
    Wan file(s) Wan 14B LoRAs), lora.json with trigger = the avatar name, current.json, then the avatar's
    `krea_lora` / `wan_lora` = `{state: ready, lora, version, trigger, ready: {lora, version}, source: upload}`.
  - 404 unknown avatar; 400 bad kind, `pair` for krea, or a file of the wrong kind; 409 while that kind's build for
    the avatar is live (`KREA_LORA_BUSY` / `WAN_LORA_BUSY` with a live job): "wait for it to finish".
- CLI: `vidgen avatar lora --name N [--krea F] [--wan F | --wan-high F --wan-low F]` (at least one).
- Dashboard: avatar card **Upload LoRAs** -> dialog: Krea 2 LoRA file, Wan LoRA file (or high-noise file) and an
  optional low-noise file, each with Choose… (native picker) and a path box; Upload uploads what is filled, one
  after the other, and refreshes the card. The local server uploads from the paths (the browser never reads them).

## Acceptance Criteria

- When a valid Krea 2 LoRA is uploaded for avatar `dana`, `avatar-dana` shall get the next version with trigger
  `dana`, and `dana`'s `krea_lora` shall be ready with that version.
- When a Wan pair is uploaded for `dana`, `avatar-dana-wan` shall get a version with both files and `wan_lora` shall
  be ready with it, so the next video starring `dana` uses it.
- If a Wan file is sent as the Krea one (or the reverse), then the API shall return 400 and change nothing on the
  avatar.
- While `dana`'s Krea (or Wan) build is live, uploading that kind shall return 409; the other kind still works.
- `POST /loras/{name}/uploads` shall still refuse `avatar-*` names.

## Out of Scope

Uploading LoRAs for avatars that don't exist yet; deleting single LoRA versions; the phone.

## Testing

`tests/test_avatar_lora_upload.py` (API, with headers stubbed), CLI argument checks, `tests/test_ui_page.py`.
