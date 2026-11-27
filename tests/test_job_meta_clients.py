"""The CLI and the dashboard send job `meta`, and avatar pick falls back to the server's meta for jobs started
on another device. docs/specs/2026-11-27-phone-ui.md"""
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
vidgen.download_dir = lambda: downloads
import ui  # noqa: E402

calls = []
server_jobs = {}
n = [0]


def api(method, path, body=None):
    calls.append((method, path, json.loads(json.dumps(body)) if body is not None else None))
    if method == "POST" and (path == "/jobs" or path.endswith("/rerun")):
        n[0] += 1
        job_id = f"{n[0]:032x}"
        meta = (body or {}).get("meta") or {}
        if path.endswith("/rerun"):
            old = server_jobs[path.split("/")[2]]
            meta = dict(old["meta"])
            if meta.get("out_name"):
                base, ext = os.path.splitext(meta["out_name"])
                meta["out_name"] = f"{base}-rerun-{job_id[:6]}{ext}"
            job_type = old["type"]
        else:
            job_type = body["type"]
        server_jobs[job_id] = {"type": job_type, "meta": meta}
        return {"job_id": job_id, "type": job_type, "status": "queued", "instance": "running", "meta": meta}
    m = path.startswith("/jobs/") and path.split("/")
    if method == "GET" and m and len(m) == 3:
        return dict(server_jobs[m[2]], job_id=m[2], status="done")
    return {}


vidgen.api = api
ui.vidgen.api = api


def last_post():
    return [c for c in calls if c[0] == "POST"][-1]


quiet = contextlib.redirect_stdout(io.StringIO())

# ---- CLI submit sends meta {source: cli, out_name}
with quiet:
    jid = vidgen.submit({"type": "video", "prompt": "a boat"}, "boat.mp4")
assert last_post()[2]["meta"] == {"source": "cli", "out_name": "boat.mp4"}, last_post()
assert vidgen.load_jobs()[jid]["out"] == os.path.join(downloads, "boat.mp4")
# a name with folders (CLI only) stays local and isn't sent
with quiet:
    vidgen.submit({"type": "video", "prompt": "a boat"}, "sub/boat.mp4")
assert last_post()[2]["meta"] == {"source": "cli"}, last_post()
with quiet:
    vidgen.submit({"type": "write", "premise": "x"}, None)
assert last_post()[2]["meta"] == {"source": "cli"}
# avatar candidates carry the avatar name and krea_lora
with quiet:
    av_cli = vidgen.avatar_generate("dana", "a woman", 2, krea_lora=False)
meta = last_post()[2]["meta"]
assert meta["source"] == "cli" and meta["avatar_name"] == "dana" and meta["krea_lora"] is False, meta

# ---- dashboard create sends meta {source: mac, ...}
ui.create_job({"job": {"type": "photos", "avatar": "dana", "scenes": [{"prompt": "x"}]}, "out": "cafe"})
assert last_post()[2]["meta"] == {"source": "mac", "out_name": "cafe"}, last_post()
ui.create_job({"job": {"type": "avatar", "description": "d", "candidates": 2}, "avatar_name": "lior", "krea_lora": True,
               "wan_lora": False})
meta = last_post()[2]["meta"]
assert meta["source"] == "mac" and meta["avatar_name"] == "lior" and meta["krea_lora"] is True and meta["wan_lora"] is False
assert meta["out_name"].startswith("avatar-lior-") and meta["out_name"].endswith(".png"), meta
ui.train_theme.__globals__["load_themes"] = lambda: {"moves": {}}
ui.train_theme("moves", 100)
assert last_post()[2]["meta"] == {"source": "mac", "out_name": "train-moves"}, last_post()
ui.preview_lora("moves", {"version": 1})
assert last_post()[2]["meta"] == {"source": "mac"}, last_post()

# ---- dashboard re-run saves under the server's re-run name
r = ui.rerun(jid)
assert vidgen.load_jobs()[r["job_id"]]["out"] == os.path.join(downloads, f"boat-rerun-{r['job_id'][:6]}.mp4")

# ---- pick: an avatar job with no local avatar_name (started elsewhere) uses the server's meta
phone_id = "f" * 32
server_jobs[phone_id] = {"type": "avatar", "meta": {"avatar_name": "noa", "krea_lora": False, "source": "phone"}}
vidgen.register_job(phone_id, "avatar", os.path.join(downloads, "x.png"), {"adopted": True})
ui.pick_candidate({"job_id": phone_id, "candidate": 2})
assert last_post() == ("POST", "/avatars", {"name": "noa", "job_id": phone_id, "candidate": 2, "make_default": False,
                                             "krea_lora": False, "wan_lora": True}), last_post()
server_jobs[phone_id]["meta"]["wan_lora"] = False  # made with no LoRAs at all: picking keeps that
ui.pick_candidate({"job_id": phone_id, "candidate": 2})
assert last_post()[2]["krea_lora"] is False and last_post()[2]["wan_lora"] is False
# the local record still wins when it has the name
ui.pick_candidate({"job_id": av_cli, "candidate": 1})
assert last_post()[2]["name"] == "dana" and last_post()[2]["krea_lora"] is False
# CLI pick, same fallback
args = types.SimpleNamespace(action="pick", name=None, no_krea_lora=False, loras=None, args=[phone_id, "1"])
with quiet:
    vidgen.avatar_command(args)
assert last_post()[2]["name"] == "noa" and last_post()[2]["krea_lora"] is False and last_post()[2]["wan_lora"] is False
args.loras = "both"  # --loras overrides what the job asked for
with quiet:
    vidgen.avatar_command(args)
assert last_post()[2]["krea_lora"] is True and last_post()[2]["wan_lora"] is True
for choice, want in vidgen.LORA_CHOICES.items():
    assert vidgen.lora_choice(types.SimpleNamespace(loras=choice, no_krea_lora=False)) == want
assert vidgen.lora_choice(types.SimpleNamespace(loras=None, no_krea_lora=True)) == (False, True)

print("test_job_meta_clients: ok")
