# AI-look check — Implementation Plan

Spec: docs/specs/2026-11-27-realism-check.md
Workspace: current branch (main)
Jira: N/A

## Progress

- [ ] Task 1: API accepts `realism_check` and lists the verdict
- [ ] Task 2: Worker check (frames, VLM critique, cleaning, hook)
- [ ] Task 3: CLI flag and watcher output
- [ ] Task 4: Mac dashboard toggle, badge and Use suggestion
- [ ] Task 5: Phone toggle, badge and Use suggestion
- [ ] Task 6: Docs
- [ ] Task 7: Live check on the render box

## Tasks

### Task 1: API accepts `realism_check` and lists the verdict

**What**: `_realism_params` adds `realism_check: body.get("realism_check") is True` for video/story/photos; other
types sending it → 400. `list_jobs` adds `realism: {score, signs}` for done jobs whose stored result has a numeric
`realism.score`.
**Files**: `lambda/api/handler.py`, `tests/test_realism_api.py` (new)
**Depends on**: none
**Verify**: `python3 -B tests/test_realism_api.py`; `make test`

### Task 2: Worker check (frames, VLM critique, cleaning, hook)

**What**: `worker/realism_check.py`: frame times, frame extraction (ffmpeg), the VLM instruction, `clean()`, photo
aggregation, `check(job, params, result, workdir)` that downloads the outputs, judges and returns `result.realism`
(errors → `{"error"}`). `models.judge_realism(images, prompt, kind, current)` (one `_chat`). The dispatcher calls it
after the render when `params.realism_check` and the type is video/photos/story.
**Files**: `worker/realism_check.py` (new), `worker/models.py`, `worker/worker.py`, `tests/test_realism_check.py` (new)
**Depends on**: Task 1
**Verify**: `python3 -B tests/test_realism_check.py`; `make test`

### Task 3: CLI flag and watcher output

**What**: `--realism-check` on `video`, `photos`, `story`; the watcher logs the verdict and writes `realism.json`
next to the downloaded files.
**Files**: `cli/vidgen.py`, `tests/test_realism_cli.py` (new)
**Depends on**: Task 2
**Verify**: `python3 -B tests/test_realism_cli.py`; `make test`

### Task 4: Mac dashboard toggle, badge and Use suggestion

**What**: Checkbox in the video/photos/story forms (remembered), sent as `realism_check`; History badge; row details
with signs, suggestion, settings, note, **Use suggestion** and **Copy prompt**.
**Files**: `cli/ui.html`, `tests/test_ui_page.py`
**Depends on**: Task 1
**Verify**: `make test`

### Task 5: Phone toggle, badge and Use suggestion

**What**: Same on the phone: checkbox in the three Create forms (remembered), badge on History cards, verdict and
**Use suggestion** on the Job screen.
**Files**: `lambda/api/phone/app.js`, `lambda/api/phone/m.html`, `tests/test_phone_page.py`
**Depends on**: Task 1
**Verify**: `make test`

### Task 6: Docs

**What**: README, RUNBOOK (what the check does and doesn't, cost, toggle, CLI flag), CHANGELOG.
**Files**: `README.md`, `RUNBOOK.md`, `CHANGELOG.md`
**Depends on**: Tasks 3, 4, 5
**Verify**: docs name the controls that exist

### Task 7: Live check on the render box

**What**: After deploy: a checked photos job and a checked video job; record scores, signs, time added, and whether
Use suggestion improves the next render. Results in the spec.
**Files**: `docs/specs/2026-11-27-realism-check.md`
**Depends on**: Task 6
**Verify**: results written in the spec
