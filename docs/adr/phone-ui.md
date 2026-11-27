# Phone UI — decision log

Spec: docs/specs/2026-11-27-phone-ui.md

## D1 — [2026-11-27] — Task 1: Job `meta` on the server

**Status**: Accepted
**Decision**: `meta.out_name` is validated with an ASCII allowlist (letters, digits, spaces, `. _ -`; starts with a letter or digit, doesn't end in `.` or a space; ≤128 chars), and a re-run replaces an existing `-rerun-<id6>` suffix and shortens the base to stay within 128.
**Reason**: The name comes from any paired device and becomes a file name on the Mac, so the security review showed a denylist (`/`, leading `.`) let through NUL/control/bidi characters, byte-long Unicode names and unbounded suffix growth.
**Alternatives considered**: Allowing Unicode names with a control/bidi-category filter: rejected as harder to get right for a small gain; the Mac watcher also re-checks the name (Task 3).
