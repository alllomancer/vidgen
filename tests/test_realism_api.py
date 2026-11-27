"""The AI-look check toggle in the API: realism_check on video/photos/story jobs only, and GET /jobs carrying the
verdict for History badges. docs/specs/2026-11-27-realism-check.md"""
import json
import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install(os.path.join("lambda", "api"))
import handler as h  # noqa: E402

h.s3 = _stubs.FakeS3()
h._s3_json = h.s3.json
h.time = mock.MagicMock(time=lambda: 1000)
queued = []
h._queue_job = lambda client, job_type, params, **kw: queued.append((job_type, params)) or h._resp(202, {})
h._resolve_avatars = lambda client, requested: []
h._job_loras = lambda *a: None
h._with_avatar_wan_lora = lambda *a: None
h._avatar_first_frame = lambda *a: None

# ---- video / story / photos: stored only when asked for
for body in ({"type": "video", "prompt": "a boat"}, {"type": "story", "story": "Once upon a time."}):
    queued.clear()
    assert h.create_job("m", dict(body, realism_check=True))["statusCode"] == 202
    assert queued[-1][1]["realism_check"] is True, queued
    h.create_job("m", dict(body))
    assert "realism_check" not in queued[-1][1], "absent unless asked (existing params unchanged)"
    h.create_job("m", dict(body, realism_check="yes"))
    assert "realism_check" not in queued[-1][1], "only a real true turns it on"
h._resolve_avatars = lambda client, requested: [{"name": "dana", "reference_key": "k", "reference_keys": ["k"]}]
queued.clear()
assert h.create_job("m", {"type": "photos", "avatar": "dana", "scenes": [{"prompt": "x"}],
                          "realism_check": True})["statusCode"] == 202
assert queued[-1][1]["realism_check"] is True

# ---- other job types refuse it
for body in ({"type": "write", "premise": "x"}, {"type": "avatar", "description": "d"}):
    queued.clear()
    r = h.create_job("m", dict(body, realism_check=True))
    assert r["statusCode"] == 400 and "realism_check" in json.loads(r["body"])["error"] and not queued, r

# ---- GET /jobs: the verdict for done jobs with a score
items = [{"job_id": "a" * 32, "client": "m", "type": "photos", "status": "done", "created_at": 3, "params": "{}",
          "result": json.dumps({"realism": {"score": 7, "signs": ["waxy skin", "too-perfect light"], "prompt": "p"}})},
         {"job_id": "b" * 32, "client": "m", "type": "video", "status": "done", "created_at": 2, "params": "{}",
          "result": json.dumps({"realism": {"error": "the check couldn't read the model's answer"}})},
         {"job_id": "c" * 32, "client": "m", "type": "video", "status": "done", "created_at": 1, "params": "{}",
          "result": "not json"},
         {"job_id": "d" * 32, "client": "m", "type": "video", "status": "running", "created_at": 0, "params": "{}"}]
h.jobs = types.SimpleNamespace(query=lambda **kw: {"Items": items})
h.Key = lambda name: types.SimpleNamespace(eq=lambda v: (name, v))
items[0]["result"] = json.dumps(dict(json.loads(items[0]["result"]), realism_retry={"tries": 9, "best_try": 4, "attempts": []}))
rows = {r["job_id"][0]: r for r in json.loads(h.list_jobs("m")["body"])["jobs"]}
assert rows["a"]["realism_retry"] == {"tries": 9, "best_try": 4} and "realism_retry" not in rows["b"]
assert rows["a"]["realism"] == {"score": 7, "signs": ["waxy skin", "too-perfect light"]}, rows["a"]
assert rows["b"]["realism"] == {"error": "the check couldn't read the model's answer"}, rows["b"]
assert "realism" not in rows["c"] and "realism" not in rows["d"]

print("test_realism_api: ok")
