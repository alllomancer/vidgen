# Upload LoRA files (`vidgen lora add`, dashboard)

Jira: N/A
Date: 2026-11-23
Status: Approved
Last reviewed: 2026-11-23

## Problem

LoRAs only come from the trainer. A Wan LoRA made elsewhere (another machine, a download) can't
be used with `--lora`.

## Approach

Upload a `.safetensors` LoRA as a new version of a LoRA name, so it works exactly like a trained
one (`--lora NAME`, pickers, `lora use`). The API checks the file's header before the version
becomes usable. CLI `vidgen lora add`, dashboard Learning > Add a LoRA file.

## Design

- Two API calls, because the API (15 s timeout) can't copy GB-sized files:
  - `POST /loras/{name}/uploads {files: single|pair}` reserves version N (max existing + 1,
    counting unregistered ones) and returns presigned PUT URLs straight to
    `loras/<client>/<name>/v<N>/t2v_high.safetensors` (and `t2v_low` for a pair).
  - `PUT /loras/{name}/versions/{N} {trigger?, description?, make_current? (default true)}` checks
    each uploaded file and only then writes `lora.json` (`source: upload`, `single_file`), and
    `current.json` unless `make_current: false`. A version without lora.json is ignored everywhere.
- Checks (two ranged reads per file, no download): size <= 2 GB; a valid safetensors header
  (<= 100 MB); LoRA weights (`lora_A/B`, `lora_down/up`); no keys of other model families
  (`double_blocks`, `input_blocks`, `transformer_blocks`, ...); Wan attention blocks
  (`blocks.N.` or `blocks_N_` with `self_attn`/`cross_attn`/`attn1`/`attn2`/`ffn`); dtype F16/BF16/F32;
  a 5120-wide layer (Wan 14B; 1536/3072 = 1.3B/5B is refused with its own message).
- One file serves both Wan 2.2 experts (`_lora_keys` returns `t2v_high` twice); a pair maps high
  and low. The worker downloads by key, so one file is fetched once. A file that passes the checks
  but still fails to load fails the job with "LoRA … could not be loaded: …".
- IAM: the API may PutObject only `loras/*/t2v_high.safetensors`, `loras/*/t2v_low.safetensors`
  and `loras/*/lora.json` (plus the existing `current.json`).
- CLI `vidgen lora add NAME --file F | --high F --low F [--trigger W] [--description D] [--keep-current]`;
  streams the file (no full read into memory). `lora list` shows uploaded versions.
- Dashboard: Learning > Add a LoRA file: name, trigger, file (+ native Choose…), optional low-noise
  file, note, "use this version"; uploaded versions listed below. The local server uploads from the
  file path (the browser never sends the bytes).

## Acceptance Criteria

- When a valid single-file Wan 14B LoRA is added as `noir`, `--lora noir` shall render with that
  file on both experts.
- When a pair is added, the high file shall load into the high-noise expert and the low file into
  the low-noise one.
- If the file is not safetensors, not a LoRA, made for another model family, made for Wan
  1.3B/5B, has an unsupported dtype or is over 2 GB, then registering shall return 400 with that
  reason and the version shall stay unusable.
- While a version is reserved but not registered, `GET /loras` shall not list it.
- If a version already has lora.json, then registering it again shall return 400.

## Out of Scope

Deleting uploaded files (the API role cannot delete; rejected uploads stay in S3, invisible);
LoRAs for the edit/photo model; checkpoints for uploaded LoRAs; downloading from URLs.

## Risks & Open Questions

- Key layouts diffusers can't map despite passing the checks. (Worker error names the LoRA.)
- Rejected or abandoned uploads leave invisible objects. (Accepted; small.)
- Third-party LoRAs can carry anything, including real-person likenesses; same content rules as
  everything else here.

## Testing

Unit tests (reserve, register, every refusal, accepted layouts, single/pair keys, CLI, dashboard
validation). Live: re-upload the trained `jerk` v1 files as a pair under a new name, render a
short clip with it.

## Update: live check (2026-11-23)

- `vidgen lora add jerk-upload --high … --low … --trigger vgn_jerk` (the trained jerk v1 pair,
  2 x 585 MB): uploaded in 6 min 44 s from the laptop, passed the checks, registered as
  `jerk-upload` v1 (pair), listed as uploaded.
- A fake Flux LoRA file: refused with "made for a different model (Flux, SD, SDXL, Qwen-Image, ...),
  not Wan"; it stays unregistered (`zz-refuse-test`, invisible).
- `vidgen video "a person dancing in a sunny park…" --no-avatar --lora jerk-upload:0.9 --seconds 2
  --fps 16`: the worker loaded `user_jerk_upload_v1`, 33 frames at 16 fps, and the clip has the
  jerk dataset's look (handheld park dance video, even a watermark-like text artifact from the
  training clips), so the uploaded weights were applied.
