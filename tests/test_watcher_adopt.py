"""The watcher adopts jobs started on other devices (the phone): it downloads them like its own, skips ones older
than 7 days or forgotten with `vidgen jobs forget`, and only trusts a plain file name from the server.
docs/specs/2026-11-27-phone-ui.md"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("cli")
import vidgen  # noqa: E402

state = tempfile.mkdtemp()
downloads = tempfile.mkdtemp()
vidgen.STATE_DIR = state
vidgen.JOBS_FILE = os.path.join(state, "jobs.json")
vidgen.LOCK_FILE = os.path.join(state, "check.lock")
vidgen.DISMISSED_FILE = os.path.join(state, "dismissed.json")
vidgen.download_dir = lambda: downloads
NOW = 1_800_000_000
vidgen.time = types.SimpleNamespace(time=lambda: NOW, sleep=lambda s: None, strftime=lambda *a: "",
                                    localtime=lambda *a: None)
logged = []
vidgen.log = lambda msg: logged.append(msg)
vidgen.notify = lambda *a: None
DAY = 86400


def jid(n):
    return f"{n:032x}"


remote = {  # job id -> GET /jobs row
    jid(1): {"type": "photos", "status": "done", "created_at": NOW - 2 * DAY, "meta": {"out_name": "cafe", "source": "phone"}},
    jid(2): {"type": "video", "status": "done", "created_at": NOW - 8 * DAY, "meta": {}},  # too old
    jid(3): {"type": "video", "status": "running", "created_at": NOW - 60, "meta": {"out_name": "../../evil.mp4"}},
    jid(4): {"type": "avatar_photoset", "status": "running", "created_at": NOW - 60, "meta": {}},  # nothing to download
    jid(5): {"type": "video", "status": "failed", "created_at": NOW - 60, "meta": {}},  # nothing to download
    jid(6): {"type": "avatar", "status": "queued", "created_at": NOW - 60,
             "meta": {"avatar_name": "noa", "krea_lora": False, "out_name": "noa.png"}},
    jid(7): {"type": "video", "status": "done", "created_at": NOW - 60, "meta": {"out_name": "boat.mp4"}},
}
list_fails = [False]
downloaded = []


def api(method, path, body=None):
    if method == "GET" and path == "/jobs":
        if list_fails[0]:
            raise vidgen.ApiError("network down")
        return {"jobs": [dict(r, job_id=k) for k, r in remote.items()]}
    job_id = path.split("/")[2]
    r = remote.get(job_id) or {"status": "running", "type": "video"}
    return dict(r, job_id=job_id, downloads={})


vidgen.api = api
vidgen.fetch_results = lambda local, rem: downloaded.append(local["out"]) or [local["out"]]

with open(os.path.join(downloads, "boat.mp4"), "w") as f:  # a Mac file with the phone job's name
    f.write("mine")

vidgen.check_once()
jobs = vidgen.load_jobs()
assert jobs[jid(1)]["status"] == "downloaded" and jobs[jid(1)]["adopted"] is True, jobs[jid(1)]
assert jobs[jid(1)]["out"] == os.path.join(downloads, "cafe"), jobs[jid(1)]["out"]
assert jid(2) not in jobs, "older than 7 days"
assert jid(4) not in jobs and jid(5) not in jobs, "only jobs with something to download"
out3 = jobs[jid(3)]["out"]
assert os.path.dirname(out3) == downloads and "evil" not in out3, out3  # unsafe server name ignored
assert jobs[jid(3)]["status"] == "running"
assert jobs[jid(6)]["avatar_name"] == "noa" and jobs[jid(6)]["krea_lora"] is False
assert jobs[jid(7)]["out"] == os.path.join(downloads, f"boat-{jid(7)[:6]}.mp4"), "no clobbering a local file"
with open(os.path.join(downloads, "boat.mp4")) as f:
    assert f.read() == "mine"

# ---- a second pass doesn't re-adopt or re-download
n = len(downloaded)
vidgen.check_once()
assert len(downloaded) == n

# ---- forget: dropped locally and never adopted again
with contextlib.redirect_stdout(io.StringIO()):
    vidgen.forget_jobs([jid(3)])
assert jid(3) not in vidgen.load_jobs()
vidgen.check_once()
assert jid(3) not in vidgen.load_jobs(), "dismissed"
assert json.load(open(vidgen.DISMISSED_FILE))[jid(3)] == NOW

# ---- GET /jobs failing doesn't stop the local jobs
remote[jid(8)] = {"type": "video", "status": "done", "created_at": NOW - 60, "meta": {}}
vidgen.register_job(jid(8), "video", os.path.join(downloads, "local.mp4"))
list_fails[0] = True
vidgen.check_once()
assert vidgen.load_jobs()[jid(8)]["status"] == "downloaded"
assert any("network down" in m for m in logged), logged

# ---- same name twice in one pass, a pending local job with that name, and a video's side files
remote.clear()
list_fails[0] = False
vidgen.register_job(jid(20), "video", os.path.join(downloads, "dup.mp4"))  # Mac job, not downloaded yet
remote[jid(20)] = {"type": "video", "status": "running", "created_at": NOW - 60, "meta": {"out_name": "dup.mp4"}}
remote[jid(21)] = {"type": "video", "status": "running", "created_at": NOW - 60, "meta": {"out_name": "dup.mp4"}}
remote[jid(22)] = {"type": "video", "status": "running", "created_at": NOW - 60, "meta": {"out_name": "dup.mp4"}}
open(os.path.join(downloads, "side.srt"), "w").close()  # only a side file of an earlier "side" video is left
remote[jid(23)] = {"type": "video", "status": "running", "created_at": NOW - 60, "meta": {"out_name": "side.mp4"}}
remote[jid(24)] = {"type": "photos", "status": "running", "created_at": NOW - 60, "meta": {"out_name": "cafe"}}
os.makedirs(os.path.join(downloads, "cafe"), exist_ok=True)
vidgen.check_once()
jobs = vidgen.load_jobs()
outs = [jobs[jid(n)]["out"] for n in (20, 21, 22)]
assert len(set(outs)) == 3, outs
assert jobs[jid(23)]["out"] == os.path.join(downloads, f"side-{jid(23)[:6]}.mp4"), jobs[jid(23)]["out"]
assert jobs[jid(24)]["out"] == os.path.join(downloads, f"cafe-{jid(24)[:6]}"), jobs[jid(24)]["out"]

# ---- malformed rows are skipped and the local jobs still download
remote.clear()
vidgen.register_job(jid(30), "video", os.path.join(downloads, "ok.mp4"))
remote[jid(30)] = {"type": "video", "status": "done", "created_at": NOW - 60, "meta": {}}
good_rows = vidgen.api
vidgen.api = lambda m, p, b=None: ({"jobs": ["x", None, {"job_id": jid(31), "type": "video", "status": "done",
                                                          "created_at": "yesterday"},
                                             {"job_id": 5}, {"job_id": jid(32), "type": "video", "status": "done",
                                                             "created_at": True}]}
                                   if p == "/jobs" else good_rows(m, p, b))
vidgen.check_once()
jobs = vidgen.load_jobs()
assert jobs[jid(30)]["status"] == "downloaded" and jid(31) not in jobs and jid(32) not in jobs
vidgen.api = lambda m, p, b=None: [1, 2] if p == "/jobs" else good_rows(m, p, b)
vidgen.check_once()  # a non-object answer: no crash
vidgen.api = good_rows

# ---- forget: every id resolves or nothing changes; a corrupt dismissed.json is treated as empty
with open(vidgen.DISMISSED_FILE, "w") as f:
    f.write("[1, 2]")
before = vidgen.load_jobs()
try:
    vidgen.forget_jobs([jid(30), "zzzz"])
    raise AssertionError("expected ApiError")
except vidgen.ApiError:
    pass
assert vidgen.load_jobs() == before
with contextlib.redirect_stdout(io.StringIO()):
    vidgen.forget_jobs([jid(30)])
assert vidgen.load_dismissed() == {jid(30): NOW}

# ---- adopted_out: only plain names, always inside the download dir
for bad in ["../x.mp4", "/etc/passwd", "a/b.mp4", ".hidden", "a\x00b", None, ""]:
    p = vidgen.adopted_out("video", jid(9), bad)
    assert os.path.dirname(p) == downloads and p.endswith(".mp4"), (bad, p)

print("test_watcher_adopt: ok")
