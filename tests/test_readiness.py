"""Render machine readiness: a percent and one line saying how far the machine is from taking a job.
The API works it out from the instance's state and the box's own report (status/render.json, written by
first boot and by the worker when it starts)."""
import datetime
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


def ts(seconds):
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)


def inst(state, launched=NOW - 600, created=None):
    """An instance as EC2 describes it; created: when its disk was attached (defaults to long before the launch)."""
    created = launched - 86400 if created is None else created
    return {"State": {"Name": state}, "LaunchTime": ts(launched),
            "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"AttachTime": ts(created)}}]}


def ready(i, report=None):
    r = h._readiness(i, report, NOW)
    assert set(r) == {"percent", "text"} and isinstance(r["percent"], int) and 0 <= r["percent"] <= 100, r
    assert r["text"] and len(r["text"]) <= 220 and "\n" not in r["text"], r
    return r


# ---- the machine is not running
assert ready(inst("stopped"))["percent"] == 0 and "send a job" in ready(inst("stopped"))["text"]
assert ready(inst("pending"))["percent"] == 5 and "tarting" in ready(inst("pending"))["text"]
assert ready(inst("stopping"))["percent"] == 0 and ready(inst("shutting-down"))["percent"] == 0
# a report from before it was stopped doesn't count while it is off
assert ready(inst("stopped"), {"percent": 100, "text": "Ready", "updated_at": NOW - 5})["percent"] == 0

# ---- running, and the box reported since this start: its own percent and line
r = ready(inst("running", launched=NOW - 300), {"percent": 62, "text": "First boot: downloading the sound model", "updated_at": NOW - 20})
assert r == {"percent": 62, "text": "First boot: downloading the sound model"}
r = ready(inst("running", launched=NOW - 300), {"percent": 100, "text": "Ready: the worker is up and takes jobs.", "updated_at": NOW - 200})
assert r["percent"] == 100 and r["text"].startswith("Ready")
# odd values from the box are kept in range and on one line
r = ready(inst("running", launched=NOW - 300), {"percent": 250, "text": "x" * 500 + "\nmore", "updated_at": NOW - 20})
assert r["percent"] == 100
assert ready(inst("running", launched=NOW - 300), {"percent": "soon", "text": "?", "updated_at": NOW - 20})["percent"] < 100

# ---- running, but the last report is from an earlier start (or there is none)
old = {"percent": 100, "text": "Ready", "updated_at": NOW - 4000}
# an ordinary start: the worker reports about a minute after the machine is up
r = ready(inst("running", launched=NOW - 60), old)
assert 30 <= r["percent"] < 100 and "worker" in r["text"], r
assert ready(inst("running", launched=NOW - 60), None)["percent"] == r["percent"]
# a first boot (the disk is as new as the launch): an estimate from the time so far, said to be one
r5 = ready(inst("running", launched=NOW - 300, created=NOW - 310), None)
r30 = ready(inst("running", launched=NOW - 1800, created=NOW - 1810), None)
r90 = ready(inst("running", launched=NOW - 5400, created=NOW - 5410), dict(old, updated_at=NOW - 9000))
assert 5 < r5["percent"] < r30["percent"] < r90["percent"] <= 95, (r5, r30, r90)
assert "First boot" in r5["text"] and "estimate" in r5["text"] and "5 of" in r5["text"], r5
assert r90["percent"] == 95, "an estimate never says ready"

# ---- GET /status carries it
h._instance = lambda instance_id=None: inst("running", launched=NOW - 300)
h._queue_depth = lambda url=None: 0
h._hold = lambda key=None: False
h._models = lambda: {}
h.TRAINER_INSTANCE_ID = ""
h.time = mock.MagicMock(time=lambda: NOW)
h._ready_report = lambda: {"percent": 100, "text": "Ready: the worker is up and takes jobs.", "updated_at": NOW - 100}
body = json.loads(h.status()["body"])
assert body["readiness"] == {"percent": 100, "text": "Ready: the worker is up and takes jobs."}, body
# the report can't be read: status still answers
h._ready_report = lambda: None
assert json.loads(h.status()["body"])["readiness"]["percent"] < 100

# ---- the worker reports when it starts
import worker  # noqa: E402

put = []
worker.s3 = types.SimpleNamespace(put_object=lambda **kw: put.append(kw))
worker.time = types.SimpleNamespace(time=lambda: NOW, sleep=lambda s: None)
worker.report_ready(100, "Ready: the worker is up and takes jobs.")
assert put[0]["Key"] == "status/render.json" and put[0]["Bucket"] == "b"
assert json.loads(put[0]["Body"]) == {"percent": 100, "text": "Ready: the worker is up and takes jobs.", "updated_at": NOW}
# a failed write never stops the worker


def broken(**kw):
    raise OSError("s3 down")


worker.s3 = types.SimpleNamespace(put_object=broken)
worker.report_ready(100, "Ready")
assert worker.READY_KEY == h.READY_KEY

print("readiness tests passed")
