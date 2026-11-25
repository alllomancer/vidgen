"""Trainer (captions, speed, auto-preview) and CLI (parsers, downloads, captions edit) tests."""
import contextlib
import io
import json
import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("trainer", "cli")
import trainer  # noqa: E402
import vidgen  # noqa: E402

# ---------------------------------------------------------------- trainer
store = {}


class S3:
    class exceptions:
        NoSuchKey = KeyError

    def get_object(self, Bucket, Key):
        if Key not in store:
            raise KeyError(Key)
        return {"Body": mock.MagicMock(read=lambda: store[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        store[Key] = Body


trainer.s3 = S3()
trainer.Progress = lambda job_id: (lambda *a, **k: None)
put_items, sent = [], []
trainer.jobs = mock.MagicMock(put_item=lambda Item: put_items.append(Item))
trainer.sqs = mock.MagicMock(send_message=lambda QueueUrl, MessageBody: sent.append((QueueUrl, MessageBody)))


def make_job(**params):
    jd = tempfile.mkdtemp()
    p = dict({"dataset": "moves", "steps": 1250, "fingerprint": "fp1"}, **params)
    return trainer.Job({"job_id": "train1", "client": "m", "params": json.dumps(p)}, jd), jd


job, jd = make_job()
items = os.path.join(jd, "items")
for n in ("s0001_00.mp4", "s0001_01.mp4", "p0002.png"):
    open(os.path.join(items, n), "w").write("x")
job.state.update(manifest={"trigger": "vgn_moves", "kind": "motion"}, segments=2, photos=1)
store["datasets/m/moves/captions.json"] = json.dumps(
    {"fingerprint": "fp1", "items": {"s0001_00.mp4": "edited caption", "gone.mp4": "x"}, "edited_at": 5}).encode()
assert job.apply_saved_captions(items) == 1
assert open(os.path.join(items, "s0001_00.txt")).read() == "edited caption"
open(os.path.join(items, "s0001_01.txt"), "w").write("vgn_moves new caption")
job.save_captions(items)
saved = json.loads(store["datasets/m/moves/captions.json"])
assert saved["items"] == {"s0001_00.mp4": "edited caption", "s0001_01.mp4": "vgn_moves new caption"}
assert saved["edited_at"] == 5 and saved["fingerprint"] == "fp1" and job.state["captions"] == 2
job2, jd2 = make_job(fingerprint="fp2")
open(os.path.join(jd2, "items", "s0001_00.mp4"), "w").write("x")
assert job2.apply_saved_captions(os.path.join(jd2, "items")) == 0, "changed dataset: captions are not reused"

fast, jdf = make_job(speed="fast")
fast.state["photos"] = 0
assert "resolution = [512, 288]" in open(fast.dataset_toml()).read()
std, jds = make_job()
std.state["photos"] = 0
assert "resolution = [832, 480]" in open(std.dataset_toml()).read()
# photo-only datasets (e.g. an avatar's photo set): no empty video dataset block for musubi
po, jdp = make_job()
po.state.update(photos=2, segments=0)
open(os.path.join(jdp, "items", "p0001.png"), "w").write("x")
open(os.path.join(jdp, "items", "p0001.txt"), "w").write("cap")
t = open(po.dataset_toml()).read()
assert "image_directory" in t and "video_directory" not in t, t
assert os.path.exists(os.path.join(jdp, "photos", "p0001.txt")), "captions move with their photos"
mixed, jdm = make_job()
mixed.state.update(photos=1, segments=3)
open(os.path.join(jdm, "items", "p0001.png"), "w").write("x")
t = open(mixed.dataset_toml()).read()
assert "image_directory" in t and "video_directory" in t

cap, jdc = make_job(captions_only=True)
ran = []
for phase in ("prepare", "caption", "cache", "train", "upload"):
    setattr(cap, phase, (lambda name: lambda *a: ran.append(name))(phase))
cap.state.update(segments=3, photos=0, captions=3)
r = cap.run_all()
assert r["captions_only"] and r["captions"] == 3 and ran == ["prepare", "caption"], (r, ran)

store["code/trainer/render.json"] = json.dumps({"queue_url": "https://sqs/render", "retention_days": 30}).encode()
pv, jdp = make_job()
pv.state["files"] = ["t2v_high.safetensors", "t2v_low.safetensors"] + [
    f"checkpoints/{e}-step{n:06d}.safetensors" for e in ("t2v_high", "t2v_low") for n in range(250, 1750, 250)]
pv.state["result"] = {"lora": "moves", "version": 2, "prefix": "loras/m/moves/v2/", "trigger": "vgn_moves"}
pv.queue_preview()
item = put_items[-1]
params = json.loads(item["params"])
assert item["type"] == "lora_preview" and item["client"] == "m" and item["auto_of"] == "train1"
assert [c["step"] for c in params["clips"]] == [500, 750, 1000, 1250, 1500], "last 5 checkpoints"
assert params["clips"][0]["high_key"] == "loras/m/moves/v2/checkpoints/t2v_high-step000500.safetensors"
assert sent[-1][0] == "https://sqs/render" and pv.state["result"]["preview_job_id"] == item["job_id"]
short, jds2 = make_job(steps=100)
short.state["files"] = ["t2v_high.safetensors", "t2v_low.safetensors"]
short.state["result"] = {"lora": "moves", "version": 3, "prefix": "loras/m/moves/v3/", "trigger": ""}
short.queue_preview()
sp = json.loads(put_items[-1]["params"])
assert [c["step"] for c in sp["clips"]] == [100] and sp["clips"][0]["high_key"] == "loras/m/moves/v3/t2v_high.safetensors"
del store["code/trainer/render.json"]
pv.state["result"].pop("preview_job_id")
pv.queue_preview()
assert "preview_error" in pv.state["result"], "a failed preview never fails the training"

# ---------------------------------------------------------------- trainer: Krea 2 avatar LoRAs
uploaded = {}
S3.upload_file = lambda self, path, Bucket, Key, ExtraArgs=None: uploaded.__setitem__(Key, open(path, "rb").read())
KP = dict(model="krea2", dataset="avatar-ava", lora="avatar-ava", version=1, trigger="ava", avatar="ava", steps=960)
kj, kjd = make_job(**KP)
ran = []
for phase in ("prepare", "caption", "krea_env", "krea_models", "krea_cache", "krea_train", "krea_upload", "krea_link",
              "cache", "queue_preview"):
    setattr(kj, phase, (lambda name: lambda *a: ran.append(name))(phase))
kj.state.update(result={"lora": "avatar-ava", "version": 1})
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "krea_lora": {"state": "training", "job_ids": ["pj", "train1"]}}).encode()
kj.run_all()
assert ran == ["prepare", "caption", "krea_env", "krea_models", "krea_cache", "krea_train", "krea_upload", "krea_link"], ran
# a build that is no longer the avatar's current one is skipped before any GPU work
ran.clear()
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "krea_lora": {"state": "queued", "job_ids": ["newer"]}}).encode()
kj.state.pop("krea_current", None)
assert "skipped" in kj.run_all() and ran == [], ran
# dataset: images only, at 1024
kj, kjd = make_job(**KP)
kj.state.update(photos=22, segments=0)
for n in ("p0001.png", "p0001.txt"):
    open(os.path.join(kjd, "items", n), "w").write("x")
toml = open(kj.krea_dataset_toml()).read()
assert "resolution = [1024, 1024]" in toml and "image_directory" in toml and "video_directory" not in toml
# training: the spike's command line; resumes from the last saved state
cmds = []


def fake_run(cmd, log_path, on_steps=None):
    cmds.append(cmd)
    out = cmd[cmd.index("--output_dir") + 1]
    open(os.path.join(out, cmd[cmd.index("--output_name") + 1] + ".safetensors"), "w").write("lora")


trainer.run = fake_run
os.makedirs(os.path.join(kjd, "out"), exist_ok=True)
kj.krea_train()
c = cmds[-1]
for flag, val in (("--network_module", "networks.lora_krea2"), ("--network_dim", "32"), ("--network_alpha", "32"),
                  ("--discrete_flow_shift", "2.5"), ("--learning_rate", "1e-4"), ("--max_train_steps", "960"),
                  ("--save_every_n_steps", "240"), ("--dit", trainer.KREA_RAW), ("--turbo_dit", trainer.KREA_TURBO),
                  ("--vae", trainer.KREA_VAE), ("--text_encoder", trainer.KREA_TE)):
    assert c[c.index(flag) + 1] == val, (flag, c[c.index(flag) + 1])
assert any(x.endswith("krea2_train_network.py") for x in c) and "--save_state" in c and "--resume" not in c
assert kj.state["krea2"]["final"] == "krea2_r0.safetensors"
kj2, kjd2 = make_job(**KP)
os.makedirs(os.path.join(kjd2, "out", "krea2_r0-step000480-state"))
kj2.state["krea2"] = {"runs": [{"run": 0, "base": 0}]}
kj2.krea_train()
c = cmds[-1]
assert c[c.index("--max_train_steps") + 1] == "480" and c[c.index("--resume") + 1].endswith("krea2_r0-step000480-state")
# upload: the file, lora.json (krea2, from an avatar), current.json, samples; version from the build
os.makedirs(os.path.join(kjd, "out", "sample"), exist_ok=True)
open(os.path.join(kjd, "out", "sample", "s_e000010_00.png"), "w").write("png")
kj.state["manifest"] = {"trigger": "ava", "kind": "character", "file_count": 22}
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "krea_lora": {"state": "training", "job_ids": ["pj", "train1"]}}).encode()
kj.krea_upload()
assert uploaded["loras/m/avatar-ava/v1/krea2.safetensors"] == b"lora"
meta = json.loads(store["loras/m/avatar-ava/v1/lora.json"])
assert meta["models"] == ["krea2"] and meta["source"] == "avatar" and meta["files"] == ["krea2.safetensors"]
assert meta["trigger"] == "ava" and meta["single_file"] is True and meta["avatar"] == "ava" and meta["steps"] == 960
assert json.loads(store["loras/m/avatar-ava/current.json"]) == {"version": 1}
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "krea_lora": {"job_ids": ["newer"]}}).encode()
store["loras/m/avatar-ava/current.json"] = json.dumps({"version": 7}).encode()
kj.krea_upload()
assert json.loads(store["loras/m/avatar-ava/current.json"]) == {"version": 7}, "a stale build never becomes current"
assert "loras/m/avatar-ava/v1/samples/s_e000010_00.png" in uploaded
# link: the avatar is marked ready with this version, keeping the link for later rebuilds
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "krea_lora": {
    "state": "training", "lora": "avatar-ava", "version": 1, "job_ids": ["pj", "train1"]}}).encode()
kj.krea_link()
kl = json.loads(store["avatars/m/ava/avatar.json"])["krea_lora"]
assert kl["state"] == "ready" and kl["ready"] == {"lora": "avatar-ava", "version": 1} and "error" not in kl, kl
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "krea_lora": {"state": "queued", "version": 2,
                                                                             "job_ids": ["newer"]}}).encode()
kj.krea_link()
assert json.loads(store["avatars/m/ava/avatar.json"])["krea_lora"]["state"] == "queued", "a stale job never overwrites"
# failure: the avatar shows failed with the reason (for this job only)
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "krea_lora": {"state": "training", "job_ids": ["train1"],
                                                                             "ready": {"lora": "avatar-ava", "version": 1}}}).encode()
trainer.krea_failed({"job_id": "train1", "client": "m", "params": json.dumps(KP)}, "boom")
kl = json.loads(store["avatars/m/ava/avatar.json"])["krea_lora"]
assert kl["state"] == "failed" and kl["error"] == "boom" and kl["ready"]["version"] == 1, kl
trainer.krea_failed({"job_id": "w1", "client": "m", "params": json.dumps({"dataset": "moves"})}, "x")  # Wan: no-op
# ---------------------------------------------------------------- trainer: avatar Wan LoRAs
WP = dict(model="t2v", dataset="avatar-ava-wan/v1", lora="avatar-ava-wan", version=1, trigger="ava", avatar="ava",
          steps=1000, avatar_created_at=77)
wj, wjd = make_job(**WP)
ran = []
for phase in ("prepare", "caption", "cache", "upload", "wan_link", "queue_preview"):
    setattr(wj, phase, (lambda name: lambda *a: ran.append(name))(phase))
wj.train = lambda expert: ran.append(expert)
wj.state.update(result={"lora": "avatar-ava-wan", "version": 1})
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "created_at": 77,
                                                 "wan_lora": {"state": "training", "job_ids": ["pw", "train1"]}}).encode()
wj.run_all()
assert ran == ["prepare", "caption", "cache", "t2v_low", "t2v_high", "upload", "wan_link"], "no auto previews for avatar builds"
ran.clear()
wj2, _ = make_job(**WP)
for phase in ("prepare", "caption", "cache", "upload", "wan_link"):
    setattr(wj2, phase, (lambda name: lambda *a: ran.append(name))(phase))
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "created_at": 77,
                                                 "wan_lora": {"state": "queued", "job_ids": ["newer"]}}).encode()
assert "skipped" in wj2.run_all() and ran == [], "a replaced build is skipped before any GPU work"
# upload: under the avatar's LoRA name and version (not the dataset name)
uj, ujd = make_job(**WP)
os.makedirs(os.path.join(ujd, "out"), exist_ok=True)
for e in trainer.plan.EXPERTS:
    open(os.path.join(ujd, "out", f"{e}.safetensors"), "wb").write(b"w")
    uj.state[e] = {"final": f"{e}.safetensors", "runs": []}
uj.state.update(manifest={"kind": "character", "trigger": "ava", "file_count": 28, "total_bytes": 1, "updated_at": 1},
                segments=4, photos=24)
uj.convert = lambda src, dst: open(dst, "wb").write(b"c")
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "created_at": 77,
                                                 "wan_lora": {"state": "training", "job_ids": ["train1"]}}).encode()
uploaded.clear()
uj.upload()
assert "loras/m/avatar-ava-wan/v1/t2v_high.safetensors" in uploaded and "loras/m/avatar-ava-wan/v1/t2v_low.safetensors" in uploaded
lj = json.loads(store["loras/m/avatar-ava-wan/v1/lora.json"])
assert lj["name"] == "avatar-ava-wan" and lj["version"] == 1 and lj["models"] == ["t2v"] and lj["avatar"] == "ava"
assert json.loads(store["loras/m/avatar-ava-wan/current.json"]) == {"version": 1}
assert uj.state["result"]["lora"] == "avatar-ava-wan" and not any(k.startswith("loras/m/avatar-ava-wan/v1/v") for k in uploaded)
# a build replaced while it trained uploads nothing (it could otherwise overwrite a newer build's files)
sj, sjd = make_job(**WP)
os.makedirs(os.path.join(sjd, "out"), exist_ok=True)
sj.state.update(uj.state)
sj.convert = uj.convert
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "created_at": 77,
                                                 "wan_lora": {"state": "queued", "job_ids": ["newer"]}}).encode()
uploaded.clear()
sj.upload()
assert not uploaded and sj.state["result"].get("skipped"), sj.state.get("result")
ran = []
sj3, _ = make_job(**WP)
for phase in ("prepare", "caption", "cache", "upload", "wan_link"):
    setattr(sj3, phase, (lambda name: lambda *a: ran.append(name))(phase))


def low_then_replaced(expert):
    ran.append(expert)
    store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "wan_lora": {"state": "queued", "job_ids": ["newer"]}}).encode()


sj3.train = low_then_replaced
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "created_at": 77,
                                                 "wan_lora": {"state": "training", "job_ids": ["train1"]}}).encode()
assert "skipped" in sj3.run_all() and ran == ["prepare", "caption", "cache", "t2v_low"], ran
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "created_at": 77,
                                                 "wan_lora": {"state": "training", "job_ids": ["train1"]}}).encode()
# link: ready with this version; a stale job never overwrites; failure marks the Wan build failed
uj.wan_link()
wl = json.loads(store["avatars/m/ava/avatar.json"])["wan_lora"]
assert wl["state"] == "ready" and wl["ready"] == {"lora": "avatar-ava-wan", "version": 1} and "error" not in wl, wl
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "wan_lora": {"state": "queued", "job_ids": ["newer"]}}).encode()
uj.wan_link()
assert json.loads(store["avatars/m/ava/avatar.json"])["wan_lora"]["state"] == "queued"
store["avatars/m/ava/avatar.json"] = json.dumps({"name": "ava", "wan_lora": {"state": "training", "job_ids": ["train1"],
                                                                            "ready": {"lora": "avatar-ava-wan", "version": 1}}}).encode()
trainer.krea_failed({"job_id": "train1", "client": "m", "params": json.dumps(WP)}, "out of memory")
wl = json.loads(store["avatars/m/ava/avatar.json"])["wan_lora"]
assert wl["state"] == "failed" and wl["error"] == "out of memory" and wl["ready"]["version"] == 1, wl
# models: a token without Krea 2 RAW access fails with the licence message
trainer._hf_token = lambda: "t"
trainer._hf_download = lambda repo, filename, local_dir, token: (_ for _ in ()).throw(PermissionError("403 gated"))
kj3, _ = make_job(**KP)
try:
    kj3.krea_models()
    raise AssertionError("no RAW access must fail")
except RuntimeError as e:
    assert "Krea-2-Raw" in str(e) and "license" in str(e), e

# ---------------------------------------------------------------- CLI
assert vidgen.parse_beats("0: walks in; 10: sits down") == [{"at": 0, "prompt": "walks in"},
                                                            {"at": 10, "prompt": "sits down"}]
bf = os.path.join(tempfile.mkdtemp(), "beats.txt")
open(bf, "w").write("# timeline\n0: walks in\n\n10: sits down; then waves\n")
assert vidgen.read_beats_file(bf) == [{"at": 0, "prompt": "walks in"}, {"at": 10, "prompt": "sits down; then waves"}]
open(bf, "w").write("0: a\nsoon: b\n")
_stubs.expect_error(vidgen.read_beats_file, bf, contains=":2:", exc=vidgen.ApiError)
_stubs.expect_error(vidgen.parse_beats, "walks in", contains="SECONDS: text", exc=vidgen.ApiError)
_stubs.expect_error(vidgen.read_beats_file, bf + ".missing", contains="cannot read", exc=vidgen.ApiError)
open(bf, "w", encoding="utf-8").write("0: café scène\n")
assert vidgen.read_beats_file(bf) == [{"at": 0, "prompt": "café scène"}]
est = vidgen.take_estimate(50)
assert "10 segments" in est and "55 min" in est, est
pl = vidgen.parse_loras
assert pl(["jerk@v1-750:0.8"]) == [{"name": "jerk", "weight": 0.8, "version": 1, "step": 750}]
assert pl(["Jerk:0.5,style@v2"]) == [{"name": "jerk", "weight": 0.5}, {"name": "style", "weight": 1.0, "version": 2}]
for bad in ["jerk@", "jerk@x", "jerk:abc", "@750", "jerk@v1-"]:
    _stubs.expect_error(pl, [bad], exc=vidgen.ApiError)
d = tempfile.mkdtemp()
f = os.path.join(d, "shots.txt")
open(f, "w").write("# set\nwhere: on a rooftop\ncamera: low angle\n\nwhere: a library\ndoing: reading\n")
assert vidgen.read_shots_file(f) == [{"shot": {"where": "on a rooftop", "camera": "low angle"}},
                                     {"shot": {"where": "a library", "doing": "reading"}}]
open(f, "w").write("where: x\nmood: y\n")
_stubs.expect_error(vidgen.read_shots_file, f, contains=":2:", exc=vidgen.ApiError)

# variations download as <out>_1.mp4, <out>_1.keyframe.png
got = []
vidgen.download = lambda url, path: got.append(path)
vidgen.fetch_results({"type": "video", "out": "/dl/market.mp4"}, {"downloads": {
    "video_1": "https://x/video_1.mp4", "video_2": "https://x/video_2.mp4", "keyframe_1": "https://x/keyframe_1.png"}})
assert sorted(got) == ["/dl/market_1.keyframe.png", "/dl/market_1.mp4", "/dl/market_2.mp4"], got
got.clear()
vidgen.fetch_results({"type": "video", "out": "/dl/one.mp4"}, {"downloads": {"video": "https://x/video.mp4"}})
assert got == ["/dl/one.mp4"]
# a name typed without an extension gets the file's own (story "m1" -> m1.mp4); a folder with that name is
# never the target (photos "test" folder, then a video named "test" -> test.mp4, then test-2.mp4)
dlx = tempfile.mkdtemp()
got.clear()
saved = vidgen.fetch_results({"type": "story", "out": os.path.join(dlx, "m1")}, {"downloads": {
    "video": "https://x/story.mp4", "narration": "https://x/narration.wav"}})
assert os.path.join(dlx, "m1.mp4") in got and os.path.join(dlx, "m1.narration.wav") in got, got
os.makedirs(os.path.join(dlx, "test"))
open(os.path.join(dlx, "test.mp4"), "w").write("x")  # an earlier clip with the name
got.clear()
vidgen.fetch_results({"type": "video", "out": os.path.join(dlx, "test")}, {"downloads": {"video": "https://x/video.mp4"}})
assert got == [os.path.join(dlx, "test.mp4")], got  # the same job's own file may be replaced
os.makedirs(os.path.join(dlx, "clip.mp4"))  # a folder exactly at the target name
got.clear()
vidgen.fetch_results({"type": "video", "out": os.path.join(dlx, "clip.mp4")}, {"downloads": {"video": "https://x/video.mp4"}})
assert got == [os.path.join(dlx, "clip-2.mp4")], got
assert vidgen.local_out("video", "j1", "m1").endswith("/m1.mp4") and vidgen.local_out("story", "j1", "s.mp4").endswith("/s.mp4")
# errors that can't fix themselves stop after 3 tries instead of re-downloading forever
assert vidgen.permanent_download_error(IsADirectoryError(21, "Is a directory"))
assert not vidgen.permanent_download_error(TimeoutError("The read operation timed out"))
jobs_store = {"t" * 32: {"type": "video", "out": "/dl/test", "status": "running", "submitted_at": 0}}
real = (vidgen.load_jobs, vidgen.save_jobs, vidgen.api, vidgen.fetch_results, vidgen.maybe_sync_loras, vidgen.locked,
        vidgen.maybe_sync_avatar_media)
vidgen.load_jobs = lambda: jobs_store
vidgen.save_jobs = lambda j: None
vidgen.maybe_sync_loras = lambda: None
vidgen.maybe_sync_avatar_media = lambda: None
vidgen.locked = __import__("contextlib").nullcontext
vidgen.api = lambda m, p, b=None: {"status": "done", "downloads": {"video": "https://x/v.mp4"}}
vidgen.fetch_results = lambda local, remote: (_ for _ in ()).throw(IsADirectoryError(21, "Is a directory"))
for _ in range(3):
    vidgen.check_once()
assert jobs_store["t" * 32]["status"] == "download_failed" and "Is a directory" in jobs_store["t" * 32]["error"]
vidgen.check_once()  # no longer pending: not downloaded again
jobs_store["u" * 32] = {"type": "video", "out": "/dl/u.mp4", "status": "running", "submitted_at": 0}
vidgen.fetch_results = lambda local, remote: (_ for _ in ()).throw(TimeoutError("timed out"))
for _ in range(5):
    vidgen.check_once()
assert jobs_store["u" * 32]["status"] == "running", "network errors keep retrying"
(vidgen.load_jobs, vidgen.save_jobs, vidgen.api, vidgen.fetch_results, vidgen.maybe_sync_loras, vidgen.locked,
 vidgen.maybe_sync_avatar_media) = real

# parallel transfers: multipart uploads (parts in parallel, completed in order; aborted on failure) and
# ranged parallel downloads (pieces reassembled exactly; servers without ranges get one stream)
import threading  # noqa: E402
blob = bytes(range(256)) * 4 + b"tail"  # 1028 bytes
src = os.path.join(tempfile.mkdtemp(), "big.safetensors")
open(src, "wb").write(blob)
puts, calls_mp, lock = {}, [], threading.Lock()


class FakeResp:
    def __init__(self, data=b"", headers=None, status=200):
        self.data, self.headers, self.status = data, headers or {}, status

    def read(self, n=-1):
        out, self.data = (self.data, b"") if n is None or n < 0 else (self.data[:n], self.data[n:])
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass


def fake_put(req, timeout=None):
    data = req.data if isinstance(req.data, bytes) else req.data.read()
    with lock:
        puts[req.full_url] = data
    return FakeResp(headers={"ETag": f'"etag-{req.full_url.rsplit("=", 1)[-1]}"'})


vidgen._urlopen = fake_put
vidgen.api = lambda m, p, b=None: calls_mp.append((m, p, b)) or {}
spec = {"key": "loras/m/big/v1/krea2.safetensors", "upload_id": "U1", "part_size": 300,
        "part_urls": [f"https://s3/part?n={i}" for i in range(1, 5)]}
vidgen.upload_multipart(spec, src)
assert b"".join(puts[f"https://s3/part?n={i}"] for i in range(1, 5)) == blob, "parts cover the file exactly"
assert calls_mp[-1][:2] == ("POST", "/uploads/complete")
assert calls_mp[-1][2]["parts"] == [{"part": i, "etag": f'"etag-{i}"'} for i in range(1, 5)]
attempts = []


def flaky_put(req, timeout=None):
    attempts.append(req.full_url)
    if req.full_url.endswith("n=3"):
        raise urllib.error.URLError("connection reset")
    return fake_put(req)


import urllib.error  # noqa: E402
vidgen._urlopen = flaky_put
calls_mp.clear()
_stubs.expect_error(vidgen.upload_multipart, spec, src, contains="part 3", exc=vidgen.ApiError)
assert attempts.count("https://s3/part?n=3") == vidgen.PART_RETRIES and calls_mp[-1][1] == "/uploads/abort"
_stubs.expect_error(vidgen.upload_multipart, dict(spec, part_urls=spec["part_urls"][:2]), src, contains="parts",
                    exc=vidgen.ApiError)
# lora add sends sizes and uses multipart when the API returns it
calls_mp.clear()
vidgen._urlopen = fake_put


def lora_api(m, p, b=None):
    calls_mp.append((m, p, b))
    if p.endswith("/uploads"):
        n = -(-b["sizes"]["high"] // 50)
        return {"version": 1, "model": "krea2", "urls": {}, "multipart": {"high": dict(
            spec, part_size=50, part_urls=[f"https://s3/part?n={i}" for i in range(1, n + 1)])}}
    return {"name": "moody", "version": 1, "model": "krea2", "single_file": True, "trigger": "", "current": True}


vidgen.api = lora_api
_hd = json.dumps({"lora_unet_blocks_0_attn_wq.lora_down.weight": {"dtype": "BF16", "shape": [32, 6144], "data_offsets": [0, 0]}}).encode()
kr2 = os.path.join(tempfile.mkdtemp(), "moody2.safetensors")
open(kr2, "wb").write(len(_hd).to_bytes(8, "little") + _hd + b"\0" * 64)
vidgen.lora_upload("moody", kr2)
assert calls_mp[0][2]["sizes"] == {"high": os.path.getsize(kr2)}
assert [c[1] for c in calls_mp] == ["/loras/moody/uploads", "/uploads/complete", "/loras/moody/versions/1"]
# downloads: ranges in parallel above the threshold
ranges = []


def fake_get(req, timeout=None):
    rng = req.headers.get("Range")
    if not rng:
        return FakeResp(blob)
    a, b2 = (int(x) for x in rng.split("=")[1].split("-"))
    with lock:
        ranges.append((a, b2))
    return FakeResp(blob[a:b2 + 1], {"Content-Range": f"bytes {a}-{b2}/{len(blob)}"}, status=206)


vidgen._urlopen = fake_get
vidgen.DOWNLOAD_PARALLEL_MIN, vidgen.DOWNLOAD_CHUNK = 100, 200
out = os.path.join(tempfile.mkdtemp(), "got.bin")
vidgen.download_file("https://s3/obj", out)
assert open(out, "rb").read() == blob and not os.path.exists(out + ".part")
assert len(ranges) >= 6, ranges  # the probe plus the pieces
vidgen._urlopen = lambda req, timeout=None: FakeResp(blob)  # no range support: one stream
out2 = os.path.join(tempfile.mkdtemp(), "got2.bin")
vidgen.download_file("https://s3/obj", out2)
assert open(out2, "rb").read() == blob
vidgen.DOWNLOAD_PARALLEL_MIN, vidgen.DOWNLOAD_CHUNK = 32 * 1024 ** 2, 16 * 1024 ** 2

# captions --edit: only changed lines are sent
calls = []
caps = {"items": {"s1.mp4": "vgn a", "s2.mp4": "vgn b"}, "trigger": "vgn", "current": True}
vidgen.api = lambda method, path, body=None: calls.append((method, path, body)) or (
    caps if method == "GET" else {"updated": len(body["items"]), "without_trigger": []})


def fake_editor(cmd):
    path = cmd[-1]
    text = open(path).read().replace("vgn b", "vgn b, slow motion")
    open(path, "w").write(text)


vidgen.subprocess = mock.MagicMock(call=fake_editor)
os.environ["EDITOR"] = "true"
vidgen.captions_command(__import__("types").SimpleNamespace(name="moves", edit=True))
assert calls[-1] == ("PUT", "/datasets/moves/captions", {"items": {"s2.mp4": "vgn b, slow motion"}}), calls[-1]
# photos: series and realism flags
NS = __import__("types").SimpleNamespace
photo_args = dict(no_avatar=False, model="qwen", series=False, where=None, wearing=None, doing=[], camera=None, scenes=[], prompts_file=None,
                  shots=None, count=None, aspect="portrait", seed=None, no_anchor=False, look="pro",
                  natural_skin=False, candid=False, grain=False, best=False)
b = vidgen.photos_body(NS(**dict(photo_args, series=True, where="a pier", doing=["a", "b"], look="film", best=True)))
assert b["series"] == {"where": "a pier", "anchor": True, "doing": ["a", "b"]} and b["count"] == 4 and b["scenes"] == []
assert b["quality"] == "best" and b["look"] == "film"
b = vidgen.photos_body(NS(**dict(photo_args, where="a pier", doing=["walking"], grain=True)))
assert b["scenes"] == [{"shot": {"where": "a pier", "doing": "walking"}}] and b["grain"] and "series" not in b
for bad, msg in [(dict(series=True), "--where"), (dict(series=True, where="x", camera="y"), "--series takes"),
                 (dict(where="x", doing=["a", "b"]), "needs --series")]:
    _stubs.expect_error(vidgen.photos_body, NS(**dict(photo_args, **bad)), contains=msg, exc=vidgen.ApiError)
assert vidgen.video_quality(NS(best=True, draft=False)) == "best"
_stubs.expect_error(vidgen.video_quality, NS(best=True, draft=True), contains="not both", exc=vidgen.ApiError)

# dashboard Train dialog: speed reaches the API
import ui  # noqa: E402
sent = []
vidgen.api = lambda m, p, b=None: sent.append(b) or {"job_id": "a" * 32}
vidgen.register_job = lambda *a, **k: None
ui.load_themes = lambda: {"jerk": {}}
ui.train_theme("jerk", 600, "fast")
ui.train_theme("jerk", 1250)
assert [b["speed"] for b in sent] == ["fast", "standard"] and sent[0]["steps"] == 600
_stubs.expect_error(ui.train_theme, "jerk", 600, "turbo", contains="speed")
# dashboard Create > Avatar: photos are checked, uploaded in order, then the avatar is created
import base64  # noqa: E402
png = b"\x89PNG\r\n\x1a\n" + b"x" * 20
jpg = b"\xff\xd8\xff\xe0" + b"x" * 20
uploaded, posted = [], []
vidgen.upload_file = lambda path, exts=None: uploaded.append((os.path.basename(path), open(path, "rb").read())) or \
    f"uploads/m/{len(uploaded)}.{path.rsplit('.', 1)[1]}"
vidgen.api = lambda m, p, b=None: posted.append((m, p, b)) or {"name": b["name"], "images": len(b["upload_keys"]),
                                                              "default": b["make_default"]}
enc = lambda d: base64.b64encode(d).decode()  # noqa: E731
r = ui.create_avatar({"name": "Dana", "description": "d", "images": [enc(jpg), enc(png)]})
assert [u[0] for u in uploaded] == ["01.jpg", "02.png"] and uploaded[0][1] == jpg, uploaded
assert posted[-1] == ("POST", "/avatars", {"name": "dana", "upload_keys": ["uploads/m/1.jpg", "uploads/m/2.png"],
                                           "description": "d", "make_default": False, "krea_lora": True}), posted[-1]
assert ui.create_avatar({"name": "dana", "images": [enc(png)], "make_default": True})["default"] is True
ui.create_avatar({"name": "dana", "images": [enc(png)], "krea_lora": False})
assert posted[-1][2]["krea_lora"] is False
n = len(uploaded)
for bad, msg in [({"name": "dana", "images": []}, "1-12"), ({"name": "dana", "images": [enc(png)] * 13}, "1-12"),
                 ({"name": "Bad Name!", "images": [enc(png)]}, "name"),
                 ({"name": "dana", "images": [enc(png), enc(b"GIF89a....")]}, "image 2"),
                 ({"name": "dana", "images": ["not base64!"]}, "image 1")]:
    _stubs.expect_error(ui.create_avatar, bad, contains=msg)
assert len(uploaded) == n, "nothing is uploaded when any photo is bad"
# write: .txt output path, story.txt downloads to it, dashboard loads it into the Story form
assert vidgen.local_out("write", "j" * 32, "lighthouse").endswith("/lighthouse.txt")
assert vidgen.local_out("write", "abc").endswith("/abc.txt")
got = []
vidgen.download = lambda url, path: got.append((url, path))
assert vidgen.fetch_results({"type": "write", "out": "/dl/l.txt"}, {"downloads": {"story": "https://x/story.txt"}}) == ["/dl/l.txt"]
assert got == [("https://x/story.txt", "/dl/l.txt")]
d = tempfile.mkdtemp()
story_path = os.path.join(d, "lighthouse.txt")
open(story_path, "w").write("The Keeper\n\nShe climbed.\n")
jid = "b" * 32
vidgen.load_jobs = lambda: {jid: {"type": "write", "status": "downloaded", "out": story_path},
                            "c" * 32: {"type": "video", "status": "downloaded", "out": story_path}}
assert ui.written_story(jid) == {"text": "The Keeper\n\nShe climbed.\n", "path": story_path}
for other in ("c" * 32, "d" * 32):
    _stubs.expect_error(ui.written_story, other, exc=KeyError)
# lora add: uploads to the reserved links, then registers; checks paths first
calls, puts = [], []
vidgen.api = lambda m, p, b=None: calls.append((m, p, b)) or (
    {"version": 3, "urls": {"high": "https://u/h", "low": "https://u/l"}} if m == "POST"
    else {"name": "noir", "version": 3, "single_file": b is None, "trigger": b["trigger"], "current": True})
vidgen.put_file = lambda url, path, label="": puts.append((url, os.path.basename(path)))
d = tempfile.mkdtemp()


def nosz(call):  # the upload request without the file sizes (newer CLIs send them)
    m, p, b = call
    return m, p, {k: v for k, v in (b or {}).items() if k != "sizes"}


def st_file(path, tensors):
    """A header-only .safetensors file (the CLI reads only the header to detect the kind)."""
    head = json.dumps({k: {"dtype": dt, "shape": shp, "data_offsets": [0, 0]} for k, (dt, shp) in tensors.items()}).encode()
    open(path, "wb").write(len(head).to_bytes(8, "little") + head + b"\0" * 64)
    return path


WAN_T = {"diffusion_model.blocks.0.self_attn.q.lora_down.weight": ("BF16", [32, 5120])}
KREA_T = {"lora_unet_blocks_0_attn_wq.lora_down.weight": ("BF16", [32, 6144]),
          "lora_unet_txtfusion_refiner_blocks_0_mlp_up.lora_up.weight": ("BF16", [6144, 32])}
hi, lo = st_file(os.path.join(d, "hi.safetensors"), WAN_T), st_file(os.path.join(d, "lo.safetensors"), WAN_T)
vidgen.lora_upload("Noir", hi, lo, "noir style")
assert nosz(calls[0]) == ("POST", "/loras/noir/uploads", {"files": "pair", "model": "t2v"}) and puts == [("https://u/h", "hi.safetensors"), ("https://u/l", "lo.safetensors")]
assert calls[1][:2] == ("PUT", "/loras/noir/versions/3") and calls[1][2]["trigger"] == "noir style"
calls.clear(); puts.clear()
vidgen.lora_upload("noir", hi)
assert nosz(calls[0])[2] == {"files": "single", "model": "t2v"} and len(puts) == 1
txt = os.path.join(d, "x.ckpt"); open(txt, "w").write("x")
junk = os.path.join(d, "junk.safetensors"); open(junk, "wb").write(b"x")
for bad, msg in [(txt, "safetensors"), (os.path.join(d, "missing.safetensors"), "not found"),
                 (junk, "not a .safetensors file")]:
    _stubs.expect_error(vidgen.lora_upload, "noir", bad, contains=msg, exc=vidgen.ApiError)
# Krea 2 files: detected from the header, reserved as model krea2; never as a pair
kr = st_file(os.path.join(d, "moody.safetensors"), KREA_T)
assert vidgen.lora_kind(vidgen.read_safetensors_header(kr)) == "krea2"
assert vidgen.lora_kind(vidgen.read_safetensors_header(hi)) == "t2v"
calls.clear(); puts.clear()
vidgen.lora_upload("moody", kr, trigger="mdy")
assert nosz(calls[0]) == ("POST", "/loras/moody/uploads", {"files": "single", "model": "krea2"}) and len(puts) == 1
calls.clear()
for pair in ((kr, hi), (hi, kr)):
    _stubs.expect_error(vidgen.lora_upload, "moody", *pair, contains="Krea 2 LoRAs are one file", exc=vidgen.ApiError)
assert calls == [], "refused before any API call"
vidgen.api = lambda m, p, b=None: ({"version": 1, "model": "krea2", "urls": {"high": "https://u/k"}} if m == "POST"
                                   else {"name": "moody", "version": 1, "model": "krea2", "single_file": True,
                                         "trigger": "", "current": True})
with mock.patch("builtins.print") as pr:
    vidgen.lora_command(NS(action="add", name="moody", file=kr, high=None, low=None, trigger=None,
                           description=None, keep_current=False))
said = " ".join(str(c.args[0]) for c in pr.call_args_list)
assert "Krea 2 LoRA" in said and "both experts" not in said, said
vidgen.api = lambda m, p, b=None: calls.append((m, p, b)) or (
    {"version": 3, "urls": {"high": "https://u/h", "low": "https://u/l"}} if m == "POST"
    else {"name": "noir", "version": 3, "single_file": b is None, "trigger": b["trigger"], "current": True})
calls.clear()
ui.add_lora({"name": "noir", "file": hi, "trigger": "t"})
assert nosz(calls[0]) == ("POST", "/loras/noir/uploads", {"files": "single", "model": "t2v"})
# local copies of every LoRA: lora pull / the watcher download missing versions to <download dir>/loras/
dl_root = tempfile.mkdtemp()
real_download_dir, real_state = vidgen.download_dir, vidgen.STATE_DIR
vidgen.download_dir = lambda: dl_root
vidgen.STATE_DIR = tempfile.mkdtemp()
remote_files = {("moody", 1): {"krea2.safetensors": b"K" * 40}, ("noir", 1): {"t2v_high.safetensors": b"H" * 30}}
gets, fetched = [], []
LIST = {"loras": [{"name": "moody", "current": 1, "versions": [{"version": 1, "models": ["krea2"], "source": "upload"}]},
                  {"name": "noir", "current": 1, "versions": [{"version": 1, "models": ["t2v"], "source": "upload"}]}]}


def fake_api(m, p, b=None):
    gets.append(p)
    if p == "/loras":
        return LIST
    _, _, name, _, ver, _ = p.split("/")
    files = remote_files[(name, int(ver))]
    return {"name": name, "version": int(ver), "model": "krea2" if name == "moody" else "t2v",
            "files": {f: {"url": f"https://d/{name}/{ver}/{f}", "size": len(data)} for f, data in files.items()},
            "checkpoints": {"checkpoints/t2v_low-step000250.safetensors": {"url": "https://d/ckpt", "size": 5}}
            if name == "noir" else {}, "lora_json": f"https://d/{name}/{ver}/lora.json"}


def fake_download(url, path):
    fetched.append(url)
    _, _, _, name, ver, fname = url.split("/", 5) if url.count("/") >= 5 else (None,) * 6
    data = b'{"version": 1}' if url.endswith("lora.json") else b"C" * 5 if url.endswith("ckpt") else \
        remote_files[(name, int(ver))][fname]
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)


vidgen.api, vidgen.download = fake_api, fake_download
got = vidgen.lora_sync()
assert os.path.getsize(os.path.join(dl_root, "loras/moody/v1/krea2.safetensors")) == 40
assert os.path.exists(os.path.join(dl_root, "loras/noir/v1/t2v_high.safetensors"))
assert os.path.exists(os.path.join(dl_root, "loras/moody/v1/lora.json")) and got == 2, got
assert not os.path.exists(os.path.join(dl_root, "loras/noir/v1/checkpoints")), "checkpoints only on request"
gets.clear(); fetched.clear()
assert vidgen.lora_sync() == 0 and fetched == [] and gets == ["/loras"], (gets, fetched)  # present: no links asked
# a size mismatch (e.g. an interrupted copy) is fetched again; a stray .part never counts as the file
with open(os.path.join(dl_root, "loras/moody/v1/krea2.safetensors"), "wb") as f:
    f.write(b"short")
open(os.path.join(dl_root, "loras/noir/v1/t2v_high.safetensors.part"), "wb").write(b"x")
gets.clear(); fetched.clear()
assert vidgen.lora_sync() == 1 and any(u.endswith("krea2.safetensors") for u in fetched), fetched
assert os.path.getsize(os.path.join(dl_root, "loras/moody/v1/krea2.safetensors")) == 40
fetched.clear()
vidgen.lora_sync(names=["noir"], checkpoints=True)
assert os.path.exists(os.path.join(dl_root, "loras/noir/v1/checkpoints/t2v_low-step000250.safetensors")), fetched
# the watcher syncs at most every 10 minutes
clock = [10_000.0]
vidgen.time.time = lambda: clock[0]
calls_sync = []
real_sync = vidgen.lora_sync
vidgen.lora_sync = lambda **kw: calls_sync.append(kw) or 0
vidgen.maybe_sync_loras()
vidgen.maybe_sync_loras()
clock[0] += 601
vidgen.maybe_sync_loras()
assert len(calls_sync) == 2, calls_sync
vidgen.lora_sync = lambda **kw: (_ for _ in ()).throw(vidgen.ApiError("network error: down"))
clock[0] += 601
vidgen.maybe_sync_loras()  # a failed sync is logged, never breaks the watcher
vidgen.lora_sync = real_sync
vidgen.time.time = __import__("time").time
assert vidgen.local_lora_versions() == {("moody", 1): 40, ("noir", 1): 35}, vidgen.local_lora_versions()
os.makedirs(os.path.join(dl_root, "loras/half/v1"))  # an incomplete copy (no record) isn't "on this Mac"
open(os.path.join(dl_root, "loras/half/v1/krea2.safetensors.part"), "wb").write(b"x")
open(os.path.join(dl_root, "loras/stray.txt"), "w").write("x")  # a stray file never breaks the listing
assert ("half", 1) not in vidgen.local_lora_versions()
# file names from the server that could escape the LoRA folder are refused, nothing is written
remote_files[("moody", 1)] = {"../../escape.safetensors": b"E"}
LIST["loras"][0]["versions"][0]["version"] = 1
os.remove(os.path.join(dl_root, "loras/moody/v1/.files.json"))
fetched.clear()
vidgen.lora_sync(names=["moody"])
assert not os.path.exists(os.path.join(dl_root, "escape.safetensors")) and fetched == [], fetched
remote_files[("moody", 1)] = {"krea2.safetensors": b"K" * 40}
# one version failing (server error, dropped connection) is logged; the others still sync
import http.client  # noqa: E402
real_dl = vidgen.download
vidgen.download = lambda url, path: (_ for _ in ()).throw(http.client.IncompleteRead(b"")) if "moody" in url else real_dl(url, path)
for rel in ("moody/v1/.files.json", "noir/v1/.files.json", "moody/v1/krea2.safetensors"):
    if os.path.exists(os.path.join(dl_root, "loras", rel)):
        os.remove(os.path.join(dl_root, "loras", rel))
fetched.clear()
vidgen.lora_sync()
assert not os.path.exists(os.path.join(dl_root, "loras/moody/v1/.files.json")), "a failed version is not marked complete"
assert os.path.exists(os.path.join(dl_root, "loras/noir/v1/.files.json"))
vidgen.download = real_dl
# the watcher survives any sync error
vidgen.lora_sync = lambda **kw: (_ for _ in ()).throw(KeyError("files"))
os.remove(os.path.join(vidgen.STATE_DIR, "lora_sync.json"))
vidgen.maybe_sync_loras()
vidgen.lora_sync = real_sync
# a second sync while one runs is skipped (lock), never writes the same files
import fcntl  # noqa: E402
held = open(os.path.join(vidgen.STATE_DIR, "lora_sync.lock"), "w")
fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
gets.clear()
assert vidgen.lora_sync() == 0 and gets == [], gets
fcntl.flock(held, fcntl.LOCK_UN)
held.close()
# avatar media: ready media is copied to <download dir>/avatars/<name>/, changed files only, bad names refused
media_files = {"photo_01.png": b"P" * 7, "clip_1.mp4": b"V" * 9}
AV = {"default": "mira", "avatars": [{"name": "mira", "media": {"state": "ready", "updated_at": 5}},
                                     {"name": "bob", "media": {"state": "clips"}},
                                     {"name": "Bad/../x", "media": {"state": "ready", "updated_at": 1}}]}
mgets, mfetched = [], []


def media_api(m, p, b=None):
    mgets.append(p)
    if p == "/avatars":
        return AV
    assert p == "/avatars/mira/media", p
    return {"state": "ready", "files": [{"name": n, "size": len(d), "url": f"https://m/{n}"} for n, d in media_files.items()]
            + [{"name": "media.json", "size": None, "url": "https://m/media.json"},
               {"name": "../evil.sh", "size": 1, "url": "https://m/evil"}]}


def media_download(url, path):
    mfetched.append(url)
    name = url.rsplit("/", 1)[1]
    with open(path + ".part", "wb") as f:
        f.write(media_files.get(name, b"{}"))
    os.replace(path + ".part", path)


vidgen.api, vidgen.download = media_api, media_download
assert vidgen.avatar_media_sync() == 1
md = os.path.join(dl_root, "avatars", "mira")
assert open(os.path.join(md, "clip_1.mp4"), "rb").read() == b"V" * 9 and os.path.exists(os.path.join(md, "media.json"))
assert "https://m/evil" not in mfetched and not os.path.exists(os.path.join(dl_root, "avatars", "evil.sh"))
assert not os.path.exists(os.path.join(dl_root, "avatars", "bob")), "media not ready: nothing copied"
mgets.clear(), mfetched.clear()
assert vidgen.avatar_media_sync() == 0 and mgets == ["/avatars"] and not mfetched, "complete copy: no links asked"
AV["avatars"][0]["media"]["updated_at"] = 6  # new media: fetched in full
media_files["clip_1.mp4"] = b"W" * 11
assert vidgen.avatar_media_sync() == 1 and "https://m/clip_1.mp4" in mfetched and "https://m/photo_01.png" in mfetched
os.remove(os.path.join(md, "clip_1.mp4"))  # same media, a file deleted here: only that file is fetched again
mfetched.clear()
assert vidgen.avatar_media_sync() == 1 and mfetched == ["https://m/clip_1.mp4"], mfetched
assert open(os.path.join(md, "clip_1.mp4"), "rb").read() == b"W" * 11
AV["avatars"][0]["media"]["updated_at"] = 7  # a remake: same sizes are fetched again, removed files go
media_files.pop("photo_01.png")
media_files["clip_1.mp4"] = b"X" * 11
mfetched.clear()
vidgen.avatar_media_sync()
assert open(os.path.join(md, "clip_1.mp4"), "rb").read() == b"X" * 11, "same size, new media: fetched again"
assert not os.path.exists(os.path.join(md, "photo_01.png")), "a file the server no longer lists is removed"
real_ams = vidgen.avatar_media_sync
calls_ams = []
vidgen.avatar_media_sync = lambda: calls_ams.append(1) or 0
vidgen.maybe_sync_avatar_media()
vidgen.maybe_sync_avatar_media()
assert calls_ams == [1], "at most every LORA_SYNC_EVERY seconds"
vidgen.avatar_media_sync = lambda: (_ for _ in ()).throw(vidgen.ApiError("network error: down"))
os.remove(os.path.join(vidgen.STATE_DIR, "avatar_media_sync.json"))
vidgen.maybe_sync_avatar_media()  # logged, never breaks the watcher
vidgen.avatar_media_sync = real_ams
vidgen.download = real_dl
posted_m = []
vidgen.api = lambda m, p, b=None: posted_m.append((m, p)) or {"name": "mira", "media": {"state": "queued"}}
vidgen.avatar_command(NS(action="media", args=[], name="Mira"))
assert posted_m == [("POST", "/avatars/mira/media")]
vidgen.api = lambda m, p, b=None: posted_m.append((m, p)) or {"name": "mira", "wan_lora": {"state": "queued", "lora": "avatar-mira-wan", "version": 1}}
vidgen.avatar_command(NS(action="wan-train", args=[], name="mira"))
assert posted_m[-1] == ("POST", "/avatars/mira/wan-lora")
vidgen.download_dir, vidgen.STATE_DIR = real_download_dir, real_state
# avatars: --no-krea-lora, krea-train, and the Krea LoRA state in avatar list
sent = []
vidgen.api = lambda m, p, b=None: sent.append((m, p, b)) or (
    {"avatars": [{"name": "ava", "description": "a sailor", "images": 1, "krea_lora": {"state": "training"}},
                 {"name": "old", "description": "", "images": 2, "krea_lora": {"state": "none"}}], "default": "ava"}
    if (m, p) == ("GET", "/avatars") else {"name": "ava", "krea_lora": {"state": "queued", "lora": "avatar-ava"}})
vidgen.upload_images = lambda images: ["uploads/m/x.png"]
vidgen.avatar_from_image("ava", "x.png", "a sailor", krea_lora=False)
assert sent[-1][1] == "/avatars" and sent[-1][2]["krea_lora"] is False
vidgen.avatar_from_image("ava", "x.png", "a sailor")
assert sent[-1][2]["krea_lora"] is True
with mock.patch("builtins.print") as pr:
    vidgen.avatar_command(NS(action="krea-train", name="ava", args=[], no_krea_lora=False))
assert sent[-1][:2] == ("POST", "/avatars/ava/krea-lora") and "queued" in str(pr.call_args_list)
_stubs.expect_error(vidgen.avatar_command, NS(action="krea-train", name=None, args=[], no_krea_lora=False),
                    contains="--name", exc=vidgen.ApiError)
with mock.patch("builtins.print") as pr:
    vidgen.avatar_command(NS(action="list", name=None, args=[], no_krea_lora=False))
rows = [str(c.args[0]) for c in pr.call_args_list]
assert any("ava" in r and "Krea LoRA: training" in r for r in rows) and any("old" in r and "Krea LoRA: none" in r for r in rows), rows
# pick keeps the choice made at refresh time (stored with the candidates job)
vidgen.load_jobs = lambda: {"j" * 32: {"type": "avatar", "avatar_name": "ava", "krea_lora": False}}
vidgen.resolve_job_id = lambda x: "j" * 32
vidgen.avatar_command(NS(action="pick", name=None, args=["jjjj", "2"], no_krea_lora=False))
assert sent[-1][2]["krea_lora"] is False and sent[-1][2]["candidate"] == 2
# lora list shows the kind of uploaded versions
vidgen.api = lambda m, p, b=None: {"loras": [
    {"name": "moody", "current": 1, "current_step": None,
     "versions": [{"version": 1, "created_at": 0, "trigger": "mdy", "source": "upload", "models": ["krea2"],
                   "checkpoints": [], "description": ""}]},
    {"name": "noir", "current": 1, "current_step": None,
     "versions": [{"version": 1, "created_at": 0, "trigger": "", "source": "upload", "models": ["t2v"],
                   "checkpoints": [], "description": ""}]},
    {"name": "avatar-mira", "current": 1, "current_step": None,
     "versions": [{"version": 1, "created_at": 0, "trigger": "mira", "source": "avatar", "models": ["krea2"],
                   "checkpoints": [], "kind": None, "steps_per_expert": None}]}]}
with mock.patch("builtins.print") as pr:
    vidgen.lora_command(NS(action="list", name=None))
listed = [str(c.args[0]) for c in pr.call_args_list]
assert any("moody" in s and "krea2" in s for s in listed) and any("noir" in s and "uploaded" in s for s in listed), listed
assert any("avatar-mira" in r and "krea2" in r and "avatar mira" in r and "None" not in r for r in listed), listed
_stubs.expect_error(ui.add_lora, {"name": "noir", "file": "rel/path.safetensors"}, contains="full path")
_stubs.expect_error(ui.add_lora, {"name": "Bad Name", "file": hi}, contains="name")
# Krea 2 from the CLI and dashboard
b = vidgen.photos_body(NS(**dict(photo_args, no_avatar=True, model="qwen", scenes=[{"prompt": "a harbour"}])))
assert b["model"] == "krea2" and b["avatar"] == "none" and b["scenes"] == [{"prompt": "a harbour"}]
assert "loras" not in b, "no --lora: nothing sent"
# photos --lora: Krea 2 LoRAs for Krea photos; refused locally with the qwen model
b = vidgen.photos_body(NS(**dict(photo_args, model="krea2", scenes=[{"prompt": "a harbour"}], lora=["moody:0.8"])))
assert b["loras"] == [{"name": "moody", "weight": 0.8}], b
b = vidgen.photos_body(NS(**dict(photo_args, no_avatar=True, scenes=[{"prompt": "a harbour"}], lora=["moody"])))
assert b["model"] == "krea2" and b["loras"] == [{"name": "moody", "weight": 1.0}], "--no-avatar implies Krea"
b = vidgen.photos_body(NS(**dict(photo_args, model="krea2", avatar="ready1", scenes=[{"prompt": "on a ferry"}])))
assert b["avatar"] == "ready1" and b["model"] == "krea2", "Krea photos of an avatar (through its Krea LoRA)"
b = vidgen.photos_body(NS(**dict(photo_args, model="krea2", scenes=[{"prompt": "a harbour"}])))
assert b["avatar"] == "none"
b = vidgen.photos_body(NS(**dict(photo_args, model="krea2", scenes=[{"prompt": "a red fox"}], raw=True)))
assert b["raw"] is True and b["model"] == "krea2"
assert "raw" not in vidgen.photos_body(NS(**dict(photo_args, model="krea2", scenes=[{"prompt": "a red fox"}])))
_stubs.expect_error(vidgen.photos_body, NS(**dict(photo_args, scenes=[{"prompt": "x"}], raw=True)),
                    contains="--model krea2", exc=vidgen.ApiError)
_stubs.expect_error(vidgen.photos_body, NS(**dict(photo_args, scenes=[{"prompt": "x"}], lora=["moody"])),
                    contains="--model krea2", exc=vidgen.ApiError)
b = vidgen.photos_body(NS(**dict(photo_args, no_avatar=False, model="qwen", where="a pier")))
assert "model" not in b and "avatar" not in b
sub = []
vidgen.submit = lambda body, out, extra=None: sub.append((body, extra)) or "job1"
real_ensure = vidgen.ensure_avatar
vidgen.ensure_avatar = lambda args: True
with mock.patch.object(sys, "argv", ["vidgen", "video", "a dog", "--seconds", "50", "--beats", "0: runs; 10: sits",
                                     "--anchor-every", "3", "--avatar", "dana"]):
    vidgen.main()
vb = sub[-1][0]
assert vb["seconds"] == 50 and vb["anchor_every"] == 3 and vb["beats"] == [{"at": 0, "prompt": "runs"},
                                                                          {"at": 10, "prompt": "sits"}], vb
with mock.patch.object(sys, "argv", ["vidgen", "video", "a dog"]):
    vidgen.main()
assert "anchor_every" not in sub[-1][0] and "beats" not in sub[-1][0] and "avatar_lora" not in sub[-1][0]
with mock.patch.object(sys, "argv", ["vidgen", "video", "a dog", "--no-avatar-lora"]):
    vidgen.main()
assert sub[-1][0]["avatar_lora"] is False
assert sub[-1][0]["image_model"] is None, "auto (default) leaves the choice to the API: Krea when possible, else Wan"
with mock.patch.object(sys, "argv", ["vidgen", "video", "a dog", "--image-model", "wan"]):
    vidgen.main()
assert sub[-1][0]["image_model"] == "wan"
# vidgen plan: a plan-only job; waits, saves <name>.beats.txt and the first frame, prints the render command
plan_dl = tempfile.mkdtemp()
real_api_p, real_dd_p, real_dl_p, real_reg_p = vidgen.api, vidgen.download_dir, vidgen.download, vidgen.register_job
real_sub_p, real_sleep_p, real_lj_p, real_sj_p = vidgen.submit, vidgen.time.sleep, vidgen.load_jobs, vidgen.save_jobs
plan_jobs, plan_posts = {}, []
vidgen.submit = lambda body, out, extra=None, quiet=False: plan_posts.append(body) or plan_jobs.__setitem__(
    "p" * 32, {"type": "video", "out": os.path.join(plan_dl, out + ".mp4"), "status": "queued"}) or "p" * 32
vidgen.load_jobs, vidgen.save_jobs = (lambda: plan_jobs), (lambda j: None)
vidgen.download_dir = lambda: plan_dl
polls = []


def plan_api(m, p, b=None):
    polls.append(p)
    if len(polls) < 2:
        return {"status": "running", "progress": "planning 10 segments"}
    return {"status": "done", "result": {"plan_only": True, "seed": 4, "beats": "0: she walks\n5: she stops"},
            "downloads": {"keyframe": "https://d/keyframe.png"}}


vidgen.api = plan_api
vidgen.download = lambda url, path: open(path, "w").write("png")
vidgen.time.sleep = lambda s: None
out_buf = io.StringIO()
with mock.patch.object(sys, "argv", ["vidgen", "plan", "mirabel on a pier", "--seconds", "50", "--avatar", "mirabel",
                                     "-o", "pier"]), contextlib.redirect_stdout(out_buf):
    vidgen.main()
assert plan_posts[-1]["plan_only"] is True and plan_posts[-1]["seconds"] == 50 and plan_posts[-1]["type"] == "video"
bf_p = os.path.join(plan_dl, "pier.beats.txt")
assert open(bf_p).read() == "0: she walks\n5: she stops\n" and os.path.exists(os.path.join(plan_dl, "pier.keyframe.png"))
printed = out_buf.getvalue()
assert "--seed 4" in printed and f"--beats-file {bf_p}" in printed and "--seconds 50" in printed and "0: she walks" in printed
assert plan_jobs["p" * 32]["status"] == "downloaded"
# the printed command is safe to paste: shell-quoted values, the options that shape the first frame, a real seed
pa = NS(prompt='café $HOME `id` "x"', seconds=50, no_avatar=False, avatar="mirabel", image_model="krea2", negative="",
        width=1280, height=704, best=False, draft=True, upscale=False, fps=32)
cmd = vidgen.plan_render_command(pa, 7, "/dl/my plans/pier.beats.txt")
import shlex  # noqa: E402
toks = shlex.split(cmd)
assert toks[:3] == ["vidgen", "video", 'café $HOME `id` "x"'], toks
assert toks[toks.index("--beats-file") + 1] == "/dl/my plans/pier.beats.txt" and "--draft" in toks and "--seed" in toks
_stubs.expect_error(vidgen.plan_render_command, pa, None, "/x.txt", contains="seed", exc=vidgen.ApiError)
vidgen.api, vidgen.download_dir, vidgen.download, vidgen.register_job = real_api_p, real_dd_p, real_dl_p, real_reg_p
vidgen.submit, vidgen.time.sleep, vidgen.load_jobs, vidgen.save_jobs = real_sub_p, real_sleep_p, real_lj_p, real_sj_p
vidgen.ensure_avatar = real_ensure
vidgen.avatar_generate("mira", "a sailor", 4, "krea2")
assert sub[-1][0] == {"type": "avatar", "description": "a sailor", "candidates": 4, "model": "krea2"}
posted = []
vidgen.api = lambda m, p, b=None: posted.append((m, p, b)) or {"job_id": "f" * 32, "name": "mira", "default": False}
regs = []
vidgen.register_job = lambda job_id, t, out, extra=None: regs.append((t, out, extra))
ui.create_job({"job": {"type": "avatar", "description": "a sailor", "model": "krea2"}, "avatar_name": "Mira"})
assert regs[-1][0] == "avatar" and regs[-1][2] == {"avatar_name": "mira", "krea_lora": True} and "avatar-mira-" in regs[-1][1]
ui.create_job({"job": {"type": "avatar", "description": "a sailor"}, "avatar_name": "mira", "krea_lora": False})
assert regs[-1][2]["krea_lora"] is False
_stubs.expect_error(ui.create_job, {"job": {"type": "avatar", "description": "x"}, "avatar_name": "Bad Name"}, contains="avatar name")
vidgen.load_jobs = lambda: {"f" * 32: {"type": "avatar", "status": "downloaded", "out": "/x.png", "avatar_name": "mira"}}
ui.pick_candidate({"job_id": "f" * 32, "candidate": 2})
assert posted[-1] == ("POST", "/avatars", {"name": "mira", "job_id": "f" * 32, "candidate": 2, "make_default": False,
                                           "krea_lora": True})
vidgen.load_jobs = lambda: {"f" * 32: {"type": "avatar", "status": "downloaded", "out": "/x.png", "avatar_name": "mira",
                                       "krea_lora": False}}
ui.pick_candidate({"job_id": "f" * 32, "candidate": 1})
assert posted[-1][2]["krea_lora"] is False, "the choice made when generating"
ui.avatar_krea_lora("mira")
assert posted[-1][:2] == ("POST", "/avatars/mira/krea-lora")
_stubs.expect_error(ui.avatar_krea_lora, "Bad Name", contains="avatar name")
ui.avatar_media("mira", "POST")
assert posted[-1][:2] == ("POST", "/avatars/mira/media")
ui.avatar_media("Mira", "GET")
assert posted[-1][:2] == ("GET", "/avatars/mira/media")
_stubs.expect_error(ui.avatar_media, "../x", "GET", contains="avatar name")
ui.avatar_wan_lora("mira")
assert posted[-1][:2] == ("POST", "/avatars/mira/wan-lora")
ui.get_job("a" * 32)
assert posted[-1][:2] == ("GET", "/jobs/" + "a" * 32)
_stubs.expect_error(ui.get_job, "../x", contains="job id")
_stubs.expect_error(ui.avatar_wan_lora, "Bad Name", contains="avatar name")
# the LoRA list marks versions copied to this computer
vidgen.api = lambda m, p, b=None: {"loras": [{"name": "moody", "current": 1, "versions": [{"version": 1}, {"version": 2}]}]}
vidgen.local_lora_versions = lambda: {("moody", 1): 40}
ll = ui.list_loras()
assert [v.get("local") for v in ll["loras"][0]["versions"]] == [True, False], ll
_stubs.expect_error(ui.pick_candidate, {"job_id": "0" * 32, "candidate": 1}, exc=KeyError)

# dashboard: cancel, avatar details, and the History previews of downloaded files
posted.clear()
vidgen.api = lambda m, p, b=None: posted.append((m, p, b)) or {"status": "cancelled"}
assert ui.cancel_job("c" * 32) == {"status": "cancelled"} and posted[-1][:2] == ("POST", "/jobs/" + "c" * 32 + "/cancel")
_stubs.expect_error(ui.cancel_job, "../status", contains="job id")
ui.get_avatar(" Mira ")
assert posted[-1][:2] == ("GET", "/avatars/mira")
_stubs.expect_error(ui.get_avatar, "mira/media", contains="avatar name")
pv = tempfile.mkdtemp()
for fname, data in (("photo_1.png", b"\x89PNG-bytes"), ("prompts.txt", b"a prompt"), ("clip.MP4", b"mp4-bytes")):
    with open(os.path.join(pv, fname), "wb") as fh:
        fh.write(data)
saved = [os.path.join(pv, n) for n in ("photo_1.png", "prompts.txt", "clip.MP4", "deleted.png")]
vidgen.load_jobs = lambda: {"d" * 32: {"type": "photos", "status": "downloaded", "out": pv, "files": saved},
                            "e" * 32: {"type": "photos", "status": "waiting", "out": pv, "files": saved}}
assert ui.job_file("d" * 32, 0) == (saved[0], "image/png")
assert ui.job_file("d" * 32, 2) == (saved[2], "video/mp4"), "the extension's case does not matter"
for jid_, n in (("d" * 32, 1),    # not an image or a video
                ("d" * 32, 3),    # listed by the watcher, deleted since
                ("d" * 32, 4), ("d" * 32, -1),  # outside the job's list
                ("e" * 32, 0),    # not downloaded yet
                ("0" * 32, 0)):   # not a job of this computer
    _stubs.expect_error(ui.job_file, jid_, n, exc=KeyError)

# ...and the same through the server: the token is needed, files stream with their type, cancel answers 200 or 202
from http.server import ThreadingHTTPServer  # noqa: E402
srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
srv.RequestHandlerClass = ui.make_handler(srv.server_port, "tok")
threading.Thread(target=srv.serve_forever, daemon=True).start()


def ui_request(method, path, token="tok"):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    conn.request(method, path, headers={"X-UI-Token": token} if token else {})
    r = conn.getresponse()
    got = (r.status, r.getheader("Content-Type"), r.read())
    conn.close()
    return got


files_url = "/api/jobs/" + "d" * 32 + "/files/"
assert ui_request("GET", files_url + "0") == (200, "image/png", b"\x89PNG-bytes")
assert ui_request("GET", files_url + "2")[:2] == (200, "video/mp4")
assert ui_request("GET", files_url + "0", token=None)[0] == 403, "files need the page's token"
assert ui_request("GET", files_url + "1")[0] == 404 and ui_request("GET", files_url + "3")[0] == 404
assert ui_request("GET", files_url + "x")[0] == 404 and ui_request("GET", "/api/jobs/zz/files/0")[0] == 404
assert ui_request("POST", "/api/jobs/" + "c" * 32 + "/cancel")[0] == 200
vidgen.api = lambda m, p, b=None: posted.append((m, p, b)) or {"status": "running", "cancel_requested": True}
assert ui_request("POST", "/api/jobs/" + "c" * 32 + "/cancel")[0] == 202, "a running job stops a little later"
vidgen.api = lambda m, p, b=None: posted.append((m, p, b)) or {"name": "mira", "images": [{"n": 1, "preview": "https://s3/x"}]}
status, _, body = ui_request("GET", "/api/avatars/mira")
assert status == 200 and json.loads(body)["images"][0]["preview"] == "https://s3/x" and posted[-1][:2] == ("GET", "/avatars/mira")
page_status, page_type, page = ui_request("GET", "/", token=None)
assert page_status == 200 and b"__UI_TOKEN__" not in page and b'const TOKEN = "tok"' in page
srv.shutdown()
srv.server_close()
print("trainer and cli tests passed")
