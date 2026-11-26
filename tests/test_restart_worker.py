"""Restart the render worker from the dashboard: the API leaves a request in the jobs table, and the worker,
between jobs, sees it and exits so that systemd starts it again (which pulls the latest worker code)."""
import json
import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install(os.path.join("lambda", "api"), "worker")
sys.modules["numpy"] = types.SimpleNamespace(stack=list, asarray=lambda x: x, uint8="uint8", concatenate=list)
import handler as h  # noqa: E402

NOW = 1_800_000_000
table = {}
h.jobs = types.SimpleNamespace(put_item=lambda Item: table.__setitem__(Item["job_id"], Item),
                               get_item=lambda Key: {"Item": table[Key["job_id"]]} if Key["job_id"] in table else {})
h.time = mock.MagicMock(time=lambda: NOW)

# ---- the machine is running: the request is recorded, and the answer says when it happens
h._instance_state = lambda: "running"
r = h.restart_worker()
body = json.loads(r["body"])
assert r["statusCode"] == 200 and body["requested"] is True and "between jobs" in body["note"], r
assert table[h.RESTART_KEY]["requested_at"] == NOW and table[h.RESTART_KEY]["client"] == "_system"
# ---- the machine is off or starting: nothing to restart, and it says the worker starts fresh anyway
for state in ("stopped", "pending", "stopping"):
    table.clear()
    h._instance_state = lambda s=state: s
    r = h.restart_worker()
    assert r["statusCode"] == 409 and state in json.loads(r["body"])["error"] and not table, r
# ---- the route
h._instance_state = lambda: "running"
h._authenticate = lambda event: "m" if hasattr(h, "_authenticate") else None
assert "POST /worker/restart" in open(os.path.join(_stubs.ROOT, "terraform", "api.tf")).read()
assert 'route == "POST /worker/restart"' in open(os.path.join(_stubs.ROOT, "lambda", "api", "handler.py")).read()

# ---- the worker: a request made after it started means "restart"; an older one doesn't
import worker  # noqa: E402

assert worker.RESTART_KEY == h.RESTART_KEY
row = {}
worker.jobs = types.SimpleNamespace(get_item=lambda Key: {"Item": row} if row else {})
worker.STARTED_AT = NOW
assert worker.restart_requested() is False  # no request at all
row.update(job_id=worker.RESTART_KEY, requested_at=NOW - 30)
assert worker.restart_requested() is False, "asked before this worker started: already done"
row.update(requested_at=NOW + 5)
assert worker.restart_requested() is True


def broken(Key):
    raise OSError("dynamodb down")


worker.jobs = types.SimpleNamespace(get_item=broken)
assert worker.restart_requested() is False, "a failed lookup never restarts the worker"

# ---- the main loop: between jobs it checks, reports, and exits with an error code (systemd restarts it)
worker.jobs = types.SimpleNamespace(get_item=lambda Key: {"Item": row})
reports, events = [], []
worker.report_ready = lambda percent, text: reports.append((percent, text))
worker.log = lambda **f: events.append(f.get("event"))
worker.sweep_work_dirs = worker.mirror_backfill = lambda *a, **k: None
polled = []
worker.sqs = types.SimpleNamespace(receive_message=lambda **kw: polled.append(1) or {"Messages": []})
try:
    worker.main()
    raise AssertionError("the worker did not exit")
except SystemExit as e:
    assert e.code == 1, e.code
assert polled == [], "it exits before taking another job"
assert "restart_requested" in events and reports[-1][0] < 100 and "estart" in reports[-1][1], (events, reports)

# ---- the CLI and the dashboard's local server know the route
assert '"/worker/restart"' in open(os.path.join(_stubs.ROOT, "cli", "vidgen.py")).read()
assert '"/api/worker/restart"' in open(os.path.join(_stubs.ROOT, "cli", "ui.py")).read()
assert 'id="restart-worker"' in open(os.path.join(_stubs.ROOT, "cli", "ui.html")).read()

print("restart worker tests passed")
