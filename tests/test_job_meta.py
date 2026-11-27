"""Job `meta` (out_name, avatar_name, krea_lora, source) is stored with the job, returned by GET /jobs and
GET /jobs/{id}, and copied on re-run. docs/specs/2026-11-27-phone-ui.md"""
import json
import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install(os.path.join("lambda", "api"))
import handler as h  # noqa: E402

table = {}
sent = []


def _query(IndexName, KeyConditionExpression, ScanIndexForward, Limit):
    rows = sorted(table.values(), key=lambda i: i["created_at"], reverse=True)
    return {"Items": rows[:Limit]}


h.jobs = types.SimpleNamespace(put_item=lambda Item, **kw: table.__setitem__(Item["job_id"], Item),
                               get_item=lambda Key: {"Item": table[Key["job_id"]]} if Key["job_id"] in table else {},
                               query=_query)
h.Key = lambda name: types.SimpleNamespace(eq=lambda v: (name, v))
h.sqs = types.SimpleNamespace(send_message=lambda **kw: sent.append(kw))
h._set_hold = lambda *a: None
h._ensure_started = lambda *a: "running"
h.s3 = _stubs.FakeS3()
clock = [1000]
h.time = mock.MagicMock(time=lambda: clock[0])
body = lambda r: json.loads(r["body"])  # noqa: E731
WRITE = {"type": "write", "premise": "a lighthouse keeper finds a map"}

# ---- a job without meta: {} everywhere
r = h.create_job("m", dict(WRITE))
assert r["statusCode"] == 202, r
plain = body(r)["job_id"]
assert "meta" not in table[plain]
assert body(h.get_job("m", plain))["meta"] == {}
assert body(h.list_jobs("m"))["jobs"][0]["meta"] == {}

for empty in ({}, None):  # an empty or null meta stores nothing
    assert "meta" not in table[body(h.create_job("m", dict(WRITE, meta=empty)))["job_id"]], empty

# ---- a job with meta: stored and returned
clock[0] += 10
meta = {"out_name": "lighthouse.txt", "avatar_name": "mirabel", "krea_lora": False, "source": "phone"}
r = h.create_job("m", dict(WRITE, meta=meta))
assert r["statusCode"] == 202, r
jid = body(r)["job_id"]
assert json.loads(table[jid]["meta"]) == meta
assert "meta" not in json.loads(table[jid]["params"]), "meta is not part of params"
assert body(h.get_job("m", jid))["meta"] == meta
row = [j for j in body(h.list_jobs("m"))["jobs"] if j["job_id"] == jid][0]
assert row["meta"] == meta

# ---- invalid meta: 400 and nothing queued
for bad in [{"colour": "red"}, {"out_name": "a/b.mp4"}, {"out_name": ".hidden"}, {"out_name": "x" * 129},
            {"out_name": 5}, {"out_name": ""}, {"out_name": "a\x00b"}, {"out_name": "a\nb"}, {"out_name": "a\x1b[2Jb"},
            {"out_name": "evil\u202efdp.mp4"}, {"out_name": "a\uff0fb"}, {"out_name": "a:b"}, {"out_name": "foo."},
            {"out_name": "foo "}, {"out_name": "<b>x</b>"}, {"out_name": "..x"}, {"out_name": "a\\b"},
            {"out_name": "שלום.mp4"}, {"avatar_name": "Bad Name"}, {"avatar_name": "mirabel\n"}, {"krea_lora": "yes"},
            {"source": "tablet"}, "notadict"]:
    before, n_sent = dict(table), len(sent)
    r = h.create_job("m", dict(WRITE, meta=bad))
    assert r["statusCode"] == 400, (bad, r)
    assert table == before and len(sent) == n_sent, bad
assert h.create_job("m", dict(WRITE, meta={"out_name": "x" * 128}))["statusCode"] == 202
assert h.create_job("m", dict(WRITE, meta={"out_name": "my take 2.final.mp4"}))["statusCode"] == 202

# ---- re-run copies meta, out_name gets -rerun-<new id[:6]>
clock[0] += 10
r = h.rerun_job("m", jid)
assert r["statusCode"] == 202, r
new = body(r)["job_id"]
assert json.loads(table[new]["meta"]) == dict(meta, out_name=f"lighthouse-rerun-{new[:6]}.txt")
assert body(r)["meta"] == json.loads(table[new]["meta"]), "the 202 says the re-run's meta (its file name)"
r2 = h.rerun_job("m", new)  # a re-run of a re-run replaces the suffix instead of stacking it
new2 = body(r2)["job_id"]
assert json.loads(table[new2]["meta"])["out_name"] == f"lighthouse-rerun-{new2[:6]}.txt"
long_id = body(h.create_job("m", dict(WRITE, meta={"out_name": "y" * 124 + ".mp4"})))["job_id"]
for _ in range(3):
    long_id = body(h.rerun_job("m", long_id))["job_id"]
    name = json.loads(table[long_id]["meta"])["out_name"]
    assert len(name) <= h.MAX_OUT_NAME and name.endswith(f"-rerun-{long_id[:6]}.mp4") and h.OUT_NAME.fullmatch(name), name
table["bad"] = dict(table[plain], job_id="bad", meta='["not", "a dict"]')
assert body(h.get_job("m", "bad"))["meta"] == {} and h.rerun_job("m", "bad")["statusCode"] == 202
r = h.rerun_job("m", plain)
assert "meta" not in table[body(r)["job_id"]], "no meta to copy"
r = h.rerun_job("m", body(h.create_job("m", dict(WRITE, meta={"source": "cli"})))["job_id"])
assert json.loads(table[body(r)["job_id"]]["meta"]) == {"source": "cli"}

# ---- saves: the same files as downloads, presigned as attachments named after the job (the phone's Save)
signed = []
h.s3.generate_presigned_url = lambda op, Params, ExpiresIn: signed.append(Params) or f"https://signed/{Params['Key']}"
done = dict(table[jid], job_id="d" * 32, status="done", result=json.dumps({
    "video_key": "outputs/m/d/video.mp4", "subtitles_key": "outputs/m/d/subs.srt", "title": "x"}))
table[done["job_id"]] = done
r = body(h.get_job("m", done["job_id"]))
assert set(r["downloads"]) == set(r["saves"]) == {"video", "subtitles"}, r
by_key = {p["Key"]: p for p in signed if "ResponseContentDisposition" in p}
assert by_key["outputs/m/d/video.mp4"]["ResponseContentDisposition"] == 'attachment; filename="lighthouse.mp4"'
assert by_key["outputs/m/d/subs.srt"]["ResponseContentDisposition"] == 'attachment; filename="lighthouse.subtitles.srt"'
nameless = dict(done, job_id="e" * 32, meta=None)
nameless.pop("meta")
table[nameless["job_id"]] = nameless
signed.clear()
h.get_job("m", nameless["job_id"])
names = sorted(p["ResponseContentDisposition"] for p in signed if "ResponseContentDisposition" in p)
assert names == ['attachment; filename="vidgen-eeeeeeee.mp4"', 'attachment; filename="vidgen-eeeeeeee.subtitles.srt"'], names
assert "saves" not in body(h.get_job("m", plain)), "only for done jobs"

print("test_job_meta: ok")
