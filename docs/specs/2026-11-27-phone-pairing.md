# Phone pairing from the Mac dashboard

Jira: N/A
Date: 2026-11-27
Status: Approved
Last reviewed: 2026-11-27

Extends docs/specs/2026-11-27-phone-ui.md. Approved by the user in conversation ("do the fixes including the mac
pairing") after the design below was presented.

## Problem

Pairing a phone means pasting its public key into `authorized_ssh_public_keys` and running `make apply`, and again
whenever Chrome loses the key (cleared site data). The user wants to pair from the Mac dashboard instead.

## Approach

The phone asks to pair with an unsigned request carrying only its public key and gets a 6-digit code. The Mac
dashboard, signed with a Terraform key, lists waiting requests and approves the one whose code the phone shows. The
approved key is stored in S3 under the approver's client name, so the phone joins the same history with no name to
type. Only Terraform keys can approve or revoke. The Terraform list keeps working unchanged.

## Design

### Storage (S3, the data bucket)

- `auth/devices.json`: `{"<key blob base64>": {"id": <8 hex>, "client": <approver's client>, "name": <label>,
  "added_at": <unix>}}`.
- `auth/pairing/<code>.json`: one object per waiting request, `{"public_key": <blob base64>, "requested_at": <unix>,
  "agent": <printable-ASCII user agent, 160 chars>, "fingerprint": "SHA256:<10 chars>"}`; older than `PAIR_TTL = 600` s
  is ignored. Separate objects, so concurrent requests can't overwrite each other.
- `auth/pairing-open.json`: `{"until": <unix>, "client": ...}`: the pairing window the dashboard opens.

### API

- `POST /pairings/open` (signed, computer key only): opens the pairing window for 10 minutes; approving a phone closes
  it. The dashboard calls it when you press **Pair a phone**.
- `POST /pair` (unsigned): `{"public_key": "ssh-ed25519 AAAA..."}` (comment ignored). The blob must decode to an
  ssh-ed25519 key (type string + 32-byte key), else 400. Outside the window → 403 `pairing is closed: on your Mac, open
  vidgen ui → Phones → Pair a phone, then try again`. At most `PAIR_MAX_PENDING = 5` waiting requests: a 6th → 429.
  The same key asking again replaces its earlier request. Returns `{"code": "123456", "expires_at": ...,
  "fingerprint": "SHA256:..."}`; the code is 6 random digits (`secrets.randbelow`), unique among waiting requests. A
  key that is already authorized gets the same answer shape with a random code and nothing stored (no oracle for
  which keys are paired; its signed calls already work). Body over 1 KB → 413. Its own throttle: 1 rps, burst 5.
- `GET /pairings` (signed, Terraform key only): `{"pending": [{code, requested_at, agent, fingerprint}], "devices":
  [{id, name, client, added_at}], "open_until": <unix or null>}`; devices are filtered to the caller's client.
- `POST /pairings/{code}/approve` (signed, Terraform key only) `{"name": <label, optional>}`: moves the request into
  `devices.json` with `client` = the caller's client; unknown or expired code → 404. Name: 1-32 chars
  `[A-Za-z0-9 ._-]`, default `phone`.
- `DELETE /pairings/{code}` (signed, Terraform key only): declines a waiting request.
- `DELETE /devices/{id}` (signed, Terraform key only, own client's devices): revokes a device.
- A phone key calling any of the five signed routes → 403 `only a computer key can manage phones`.

### Authentication

`_authenticate` accepts a key from the Terraform list (as now) or from `devices.json`, read through a 30-second
in-process cache (so a revoke takes effect within ~30 s of the next warm call). The returned client name is the
device record's `client`. Everything else in the signature check is unchanged.

### Phone page

The pairing screen shows **Pair this phone**: it sends `POST /pair` and shows the code and the key fingerprint large,
with "On your Mac, open `vidgen ui` → Phones and approve this code" (and, when the window is closed, "press Pair a
phone there first"). It retries `GET /jobs` every 5 s while the code is valid and opens
History once approved; after 10 minutes it offers a new code. The Terraform steps (name + key line) stay below as
"Or add the key with Terraform".

### Mac dashboard

A **Phones** panel (Server tab): the phone page link (`<api_url>/m`) as a QR code to scan with the phone's camera
(made by `cli/qr.py`, standard library only, drawn as SVG) and as a link with Copy; waiting requests with code, time and
browser, **Approve** (with an optional name) and **Decline**; paired phones with **Remove**. It refreshes every 5 s
while a request is waiting. `ui.py` proxies `GET /api/pairings`, `POST /api/pairings/{code}/approve`,
`DELETE /api/pairings/{code}`, `DELETE /api/devices/{id}` and `GET /api/phone` (`{url}`).

### Also in this change

- Phone: a status strip on History from `GET /status` (render box off / starting / running N min, queued jobs).
- Phone: app icons in the manifest (192 and 512 px PNG, generated once and bundled) so Chrome offers an app install.

## Acceptance Criteria

- When `POST /pair` gets a valid ed25519 public key, the API shall return 200 with a 6-digit code and store the
  request; when it gets anything else (bad base64, wrong key type, wrong length, missing), it shall return 400.
- If 5 requests are waiting and a 6th key asks, then the API shall return 429; when a waiting request is older than
  600 s, it shall no longer count or be approvable.
- When a Terraform-key client approves a waiting code, a request signed by that phone key shall authenticate as the
  approver's client (its `GET /jobs` returns the approver's jobs).
- If a phone key calls `GET /pairings`, approve, decline or `DELETE /devices/{id}`, then the API shall return 403.
- When a device is revoked, requests signed by its key shall be rejected with 401 once the cache (30 s) expires.
- If a key that is already authorized asks to pair, then the API shall answer like a new request (200, a code) and
  store nothing.
- While the pairing window is closed, `POST /pair` shall return 403; `POST /pairings/open` (computer key) opens it for
  600 s, and approving a phone closes it.
- The unsigned `POST /pair` shall never grant access by itself: until approved, the phone's signed requests return 401.

## Out of Scope

Naming devices after approval; per-device permissions; pairing more than one client's phones from one Mac
(each Mac key approves into its own client).

## Risks & Open Questions

- `POST /pair` is unsigned and writes to S3. It only works for 10 minutes after the owner presses Pair a phone, with
  1 KB bodies, 5 waiting requests and a 1 rps throttle; a flood can only delay pairing, never grant access.
  (Resolved.)
- An attacker could submit their own key hoping the user approves the wrong code. The user approves only the code on
  their own phone's screen, and the dashboard shows each request's key fingerprint (also on the phone), time and
  browser. (Resolved.)
- `devices.json` writes are read-modify-write without a lock; approvals are rare and manual. (Deferred: accept.)
- A stolen unlocked phone signs as the user until revoked; Remove in the dashboard fixes it within ~30 s. (Resolved.)

## Testing

Unit tests (tests/test_pairing.py): `POST /pair` validation, limits, TTL, 409; approve/decline/revoke and the
Terraform-only rule; `_authenticate` with a device key (cache, revoke). `tests/test_phone_page.py` and
`tests/test_ui_page.py` extended for the new screens. Live: pair the Android phone from the dashboard, remove it,
see 401.
