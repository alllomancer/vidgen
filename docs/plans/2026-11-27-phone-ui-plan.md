# Phone UI and shared job history — Implementation Plan

Spec: docs/specs/2026-11-27-phone-ui.md
Workspace: current branch (main)
Jira: N/A

## Progress

- [ ] Task 1: Job `meta` on the server (create, list, get, re-run)
- [ ] Task 2: CLI and dashboard send `meta`; pick falls back to server meta
- [ ] Task 3: Watcher adopts jobs from other devices (`jobs forget`, 7-day cut-off)
- [ ] Task 4: Unauthenticated `/m` routes serving a placeholder page
- [ ] Task 5: Browser SSHSIG signer with ssh-keygen parity test
- [ ] Task 6: Phone pairing screen and History
- [ ] Task 7: Phone Create (video, photos, story) and photo upload with S3 CORS
- [ ] Task 8: Phone Job screen (results, Save) and Avatars
- [ ] Task 9: "Pair a phone" in the dashboard, docs
- [ ] Task 10: Live check on the Android phone

## Tasks

### Task 1: Job `meta` on the server (create, list, get, re-run)

**What**: `POST /jobs` accepts optional `meta` with only `out_name` (plain name, ≤128 chars, no `/`, no leading
`.`), `avatar_name` (`AVATAR_NAME` rule), `krea_lora` (bool), `source` (`mac`|`cli`|`phone`); anything else → 400 and
nothing queued. Stored as the job item's `meta` (JSON string). `GET /jobs` rows and `GET /jobs/{id}` return `meta`
(`{}` when absent). Re-run copies `meta` with `out_name` → `<base>-rerun-<new id[:6]><ext>`.
**Files**: `lambda/api/handler.py`, `tests/test_job_meta.py` (new)
**Depends on**: none
**Verify**: `python3 -B tests/test_job_meta.py` (spec criteria 4, 5) and `python3 -B tests/test_api.py` still pass

### Task 2: CLI and dashboard send `meta`; pick falls back to server meta

**What**: `vidgen.py` create paths and `ui.py` `create_job`/`rerun`/`train_theme`/`preview_lora` send
`meta` (`source: cli` / `mac`, `out_name`, and for avatar jobs `avatar_name`/`krea_lora`). `ui.pick_candidate` and
`vidgen avatar pick` use `GET /jobs/{id}` `meta` when the local record has no `avatar_name`; `ui.rerun` uses the
server's re-run `out_name`.
**Files**: `cli/vidgen.py`, `cli/ui.py`, `tests/test_job_meta_clients.py` (new)
**Depends on**: Task 1
**Verify**: `python3 -B tests/test_job_meta_clients.py` (spec criterion 8 with the API stubbed); `make test`

### Task 3: Watcher adopts jobs from other devices (`jobs forget`, 7-day cut-off)

**What**: `check_once` calls `GET /jobs` once per pass and registers every `queued`/`running`/`done` job not in
`jobs.json`, created within `ADOPT_DAYS = 7` and not in `jobs.json`'s new `dismissed` list, with
`local_out(type, id, meta.out_name)` and `adopted: true`. A failed `GET /jobs` logs and continues with the local
jobs. New `vidgen jobs forget <id-prefix>` drops a local record and adds it to `dismissed`.
**Files**: `cli/vidgen.py`, `tests/test_watcher_adopt.py` (new)
**Depends on**: Task 2 (shares `cli/vidgen.py`)
**Verify**: `python3 -B tests/test_watcher_adopt.py` (spec criteria 6, 7: adopt + download, 8-day-old skipped,
dismissed skipped, GET /jobs failure doesn't stop downloads)

### Task 4: Unauthenticated `/m` routes serving a placeholder page

**What**: New `lambda/api/phone/` with `m.html`, `app.js`, `manifest.webmanifest`. The handler serves
`GET /m`, `GET /m/app.js`, `GET /m/manifest.webmanifest` before `_authenticate`, with the spec's CSP,
`Cache-Control: no-cache`, `nosniff`, `Referrer-Policy: no-referrer`; every other route still needs a signature.
`make build` copies `phone/` into the bundle; Terraform adds the three routes. The page at this point only shows
"vidgen" and a WebCrypto Ed25519 support check.
**Files**: `lambda/api/handler.py`, `lambda/api/phone/*` (new), `Makefile`, `terraform/api.tf`,
`tests/test_phone_routes.py` (new)
**Depends on**: none
**Verify**: `python3 -B tests/test_phone_routes.py` (spec criterion 1: 200 + headers without signature, `GET /jobs`
unsigned still 401, unknown `/m/x` 404); `make plan` shows 3 new routes and nothing else replaced

### Task 5: Browser SSHSIG signer with ssh-keygen parity test

**What**: `lambda/api/phone/sign.js` (ES module, no dependencies): key generation (`extractable: false`) and
IndexedDB storage, `publicKeyLine(comment)`, and `signedFetch(method, path, body)` that builds the canonical string,
SSHSIG v1 (namespace `vidgen-api`, sha512), armors it and sets the three `x-vidgen-*` headers like
`cli/vidgen.py` `sign()`. Served at `GET /m/sign.js` (route added like Task 4).
**Files**: `lambda/api/phone/sign.js` (new), `lambda/api/handler.py`, `terraform/api.tf`,
`tests/sign_parity.test.mjs` (new), `tests/test_sign_parity.py` (new: generates a test key with ssh-keygen, runs the
Node test, verifies with `_verify_sshsig`)
**Depends on**: Task 4
**Verify**: `python3 -B tests/test_sign_parity.py` (spec criteria 2, 10: byte-identical to `ssh-keygen -Y sign`,
accepted by `_verify_sshsig`, key not extractable); skipped with a message when `node` isn't installed

### Task 6: Phone pairing screen and History

**What**: `app.js` first run: create the key, ask for the comment, show the `ssh-ed25519 ... <comment>` line with
Copy and the "add to `authorized_ssh_public_keys`, `make apply`" steps; a 401 returns here ("This phone isn't paired
yet"). Once paired: History from `GET /jobs` (summary, status, progress and time left, Cancel, Re-run, Copy prompt),
refreshing every 10 s while visible. Calls `navigator.storage.persist()`; if not granted, shows "Install this page
(Add to Home screen) to keep this phone paired".
**Files**: `lambda/api/phone/app.js`, `lambda/api/phone/m.html`, `tests/test_phone_page.py` (new: static checks
like `tests/test_ui_page.py` — no inline scripts, only `/m/*` and signed API paths fetched)
**Depends on**: Task 5
**Verify**: `python3 -B tests/test_phone_page.py`; after `make plan && make apply`, open `/m` in desktop Chrome
with the phone key added: History matches `vidgen ui` (spec criterion 3)

### Task 7: Phone Create (video, photos, story) and photo upload with S3 CORS

**What**: Create screen with the dashboard's basic fields for video, photos and story, avatar picker from
`GET /avatars`, sending `meta {source: phone, out_name}`. Photo input: png/jpg/webp as picked, other decodable formats converted to JPEG 0.92 via canvas,
undecodable ones (HEIC) refused with the spec's message,
`POST /uploads` presigned PUT, then `upload_keys`. Terraform: `aws_s3_bucket_cors_configuration` with the API origin
only, PUT, `content-type`, `ETag`, 3600 s.
**Files**: `lambda/api/phone/app.js`, `terraform/storage.tf`, `tests/test_phone_page.py`
**Depends on**: Task 6
**Verify**: `python3 -B tests/test_phone_page.py`; `make plan` shows only the CORS resource added; from the phone,
a photos job with a camera-roll image is queued and appears in the Mac's History (spec criterion 9 checked by a
browser PUT from another origin failing preflight)

### Task 8: Phone Job screen (results, Save) and Avatars

**What**: Job screen re-fetches `GET /jobs/{id}` on open; photos and mp4s shown inline from presigned links, a
Save button per file using the new `saves` links (`GET /jobs/{id}` returns them, presigned with
`ResponseContentDisposition: attachment`). Avatars: list, set default, add reference photos (same upload
path as Task 7).
**Files**: `lambda/api/phone/app.js`, `lambda/api/handler.py`, `tests/test_phone_page.py`, `tests/test_job_meta.py`
**Depends on**: Task 7
**Verify**: `python3 -B tests/test_phone_page.py`; `python3 -B tests/test_job_meta.py` (`saves` links carry the attachment
disposition); on the phone a finished video plays and Save puts it in the gallery

### Task 9: "Pair a phone" in the dashboard, docs

**What**: Dashboard Settings → "Pair a phone": the `/m` URL as a QR code (generated in the page, no CDN) and the
Mac key's comment to reuse; `ui.py` adds `GET /api/phone` returning `{url, comment}` from the config. README
(phone section), RUNBOOK (pairing, revoking, installing the page so Chrome keeps the key), `CHANGELOG.md` dated entry.
**Files**: `cli/ui.py`, `cli/ui.html`, `README.md`, `RUNBOOK.md`, `CHANGELOG.md`
**Depends on**: Task 6
**Verify**: `make test` (`tests/test_ui_page.py` covers the new section); the QR opens `/m` on the phone

### Task 10: Live check on the Android phone

**What**: Pair the Android phone in Chrome and install the page; create a photos job from the phone; watch it on
both devices; Save a photo and find it in the gallery;
confirm the Mac watcher downloaded it as `adopted`; re-run from the Mac; revoke the key → 401 and pairing screen.
Record results in the spec's Live check section.
**Files**: `docs/specs/2026-11-27-phone-ui.md`
**Depends on**: Tasks 3, 8, 9
**Verify**: every step above observed; results written in the spec
