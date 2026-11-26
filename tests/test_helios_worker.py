"""worker/helios.py: Helios in its own venv and process (docs/specs/2026-11-26-helios-default.md, criteria 7-9),
and the video job that uses it (criteria 6, 10)."""
import io
import os
import sys
import tempfile
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker")
sys.modules["numpy"] = types.SimpleNamespace(stack=list, asarray=lambda x: x, uint8="uint8",
                                             concatenate=lambda parts: [f for p in parts for f in p])
import helios  # noqa: E402
import krea  # noqa: E402
import mirror  # noqa: E402
import models  # noqa: E402

# ---- pins and sizes
assert len(helios.REPO_COMMIT) == 40 and len(helios.MODEL_REVISION) == 40
assert helios.MODEL_UNIT == f"helios/{helios.MODEL_REVISION[:12]}" and helios.MODEL_DIR.endswith(helios.MODEL_REVISION[:12])
assert (helios.SIZE, helios.FPS, helios.CHUNK) == ((640, 384), 24, 33)
# whole 33-frame chunks that cover the length
assert [helios.frames_for(s) for s in (1, 5, 30, 60)] == [33, 132, 726, 1452]

# ---------------------------------------------------------------- environment (criterion 9)
root = tempfile.mkdtemp()
helios.VENV, helios.REPO_DIR = os.path.join(root, "venv"), os.path.join(root, "repo")
helios.REQUIREMENTS = os.path.join(root, "helios_requirements.txt")
open(helios.REQUIREMENTS, "w").write("torch==2.10.0\n")
ran = []


def run(cmd, check=True, **kw):
    ran.append(list(cmd))
    if cmd[:3] == [helios.PYTHON, "-m", "venv"]:
        os.makedirs(os.path.join(cmd[3], "bin"), exist_ok=True)
        open(os.path.join(cmd[3], "bin", "python"), "w").close()
    if cmd[:2] == ["git", "clone"]:
        os.makedirs(os.path.join(cmd[-1], ".git"))
    if "--help" in cmd:  # the check that Helios's script starts in this venv
        return mock.MagicMock(returncode=run.check_exit, stdout="", stderr="Traceback\nModuleNotFoundError: No module named 'pandas'\n")
    return mock.MagicMock(returncode=0)


run.check_exit = 0


helios.subprocess = types.SimpleNamespace(run=run, PIPE=-1, STDOUT=-2, Popen=None)
said = []
helios.ensure_env(said.append)
kinds = [" ".join(c[:3]) for c in ran]
assert any(c[:3] == [helios.PYTHON, "-m", "venv"] for c in ran), ran
assert any("pip" in c[0] and "-r" in c and helios.REQUIREMENTS in c for c in ran), ran
assert ["git", "clone"] in [c[:2] for c in ran] and any(c[:2] == ["git", "-C"] and "checkout" in c and helios.REPO_COMMIT in c for c in ran), kinds
assert said and "Helios" in said[0]
# ... and Helios's own script was started once in the new venv, so a missing package shows here, not on a job
assert any(c[0] == os.path.join(helios.VENV, "bin", "python") and "infer_helios.py" in c and "--help" in c for c in ran), ran
# nothing changed: nothing runs
ran.clear()
helios.ensure_env(said.append)
assert ran == [], ran
# the requirements changed: packages again, the repository is not cloned again but put on the pin
open(helios.REQUIREMENTS, "w").write("torch==2.10.1\n")
helios.ensure_env(said.append)
assert any("-r" in c for c in ran) and ["git", "clone"] not in [c[:2] for c in ran] and any("checkout" in c for c in ran), ran
# the pinned commit changed: the same
ran.clear()
helios.REPO_COMMIT = "0" * 40
helios.ensure_env(said.append)
assert any("checkout" in c and "0" * 40 in c for c in ran), ran

# the script doesn't start in the venv (a package is missing): an error that names it, and no "ready" marker
ran.clear()
helios.REPO_COMMIT = "1" * 40
run.check_exit = 1
try:
    helios.ensure_env(said.append)
    raise AssertionError("an incomplete environment was accepted")
except RuntimeError as e:
    assert "pandas" in str(e), e
run.check_exit = 0
ran.clear()
helios.ensure_env(said.append)
assert ran, "not marked ready: it is built again next time"

# ---------------------------------------------------------------- weights: mirror first (criterion 9)
helios.MODEL_DIR = os.path.join(root, "model")
pulls, pushes, hf = [], [], []
in_mirror = {"on": True}


def pull(unit, local_dir, progress=None):
    pulls.append(unit)
    if not in_mirror["on"]:
        return None
    os.makedirs(local_dir, exist_ok=True)
    return {"unit": unit, "source": "mirror", "bytes": 80_000_000_000, "seconds": 90.0, "mb_per_s": 880}


def snapshot_download(repo, revision=None, local_dir=None, allow_patterns=None, token=None):
    hf.append((repo, revision, tuple(allow_patterns or ())))
    os.makedirs(local_dir, exist_ok=True)
    open(os.path.join(local_dir, "model_index.json"), "w").write("{}")


mirror.pull = pull
mirror.push_async = lambda unit, local_dir, files=None: pushes.append(unit)
sys.modules["huggingface_hub"] = types.SimpleNamespace(snapshot_download=snapshot_download)
helios.ensure_weights(said.append)
assert pulls == [helios.MODEL_UNIT] and hf == [] and pushes == [] and os.path.exists(os.path.join(helios.MODEL_DIR, ".complete"))
pulls.clear()
helios.ensure_weights(said.append)
assert pulls == [], "the marker is there: nothing asked again"
# not mirrored: Hugging Face at the pinned revision, without the 57 GB transformer this path doesn't use
helios.MODEL_DIR = os.path.join(root, "model2")
in_mirror["on"] = False
helios.ensure_weights(said.append)
assert hf and hf[0][:2] == (helios.MODEL_REPO, helios.MODEL_REVISION), hf
assert "transformer/*" in hf[0][2] and not any("transformer_ode" in p for p in hf[0][2]), hf
assert pushes == [helios.MODEL_UNIT]

# ---------------------------------------------------------------- generate (criteria 7, 8)
helios.ensure_env = lambda progress=None: None
helios.ensure_weights = lambda progress=None: None
released, popen_calls = [], []
models.release_all = lambda: released.append(len(popen_calls))
models.check_cancel = lambda: None
models.fit = lambda image, w, h: types.SimpleNamespace(save=lambda path: open(path, "w").write(f"{image}@{w}x{h}"))


class Proc:
    def __init__(self, cmd, cwd=None, stdout=None, stderr=None, text=None, env=None):
        popen_calls.append({"cmd": list(cmd), "cwd": cwd, "env": env})
        self.cmd, self.returncode, self.killed = list(cmd), None, False
        if "--image_path" in cmd:  # read now: the work folder is removed when generate() returns
            popen_calls[-1]["image"] = open(cmd[cmd.index("--image_path") + 1]).read()
        out = cmd[cmd.index("--output_folder") + 1]
        self.stdout = io.StringIO(Proc.output)
        if Proc.exit == 0:
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, "0000_i2v_1.mp4"), "w").write(str(int(cmd[cmd.index("--num_frames") + 1])))

    def wait(self):
        self.returncode = Proc.exit
        return Proc.exit

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


Proc.exit, Proc.output = 0, "loading\nMax memory: 45.3 GB\n"
helios.subprocess = types.SimpleNamespace(Popen=Proc, PIPE=-1, STDOUT=-2, run=run)
# the fake clip "contains" as many frames as were asked for
sys.modules["imageio"] = types.SimpleNamespace(get_reader=lambda path: [f"frame{i}" for i in range(int(open(path).read()))])
helios.imageio = sys.modules["imageio"]

frames = helios.generate("she dances", 30, seed=7, image="<first frame>",
                         loras=[{"path": "/opt/models/loras/x/t2v_low.safetensors", "weight": 1.0}])
c = popen_calls[-1]["cmd"]
assert released == [0], "the worker's models leave the GPU before Helios starts"
assert c[0] == os.path.join(helios.VENV, "bin", "python") and "infer_helios.py" in c and popen_calls[-1]["cwd"] == helios.REPO_DIR


def arg(name, n=1):
    i = c.index(name)
    return c[i + 1] if n == 1 else c[i + 1:i + 1 + n]


assert arg("--sample_type") == "i2v" and arg("--num_frames") == "726" and arg("--fps") == "24"
assert (arg("--width"), arg("--height")) == ("640", "384") and arg("--seed") == "7" and arg("--prompt") == "she dances"
assert arg("--base_model_path") == helios.MODEL_DIR and arg("--transformer_path") == helios.MODEL_DIR
assert arg("--guidance_scale") == "1.0" and arg("--pyramid_num_inference_steps_list", 3) == ["2", "2", "2"]
assert all(flag in c for flag in ("--is_enable_stage2", "--is_amplify_first_chunk", "--disable_flash_attention"))
assert arg("--lora_path") == "/opt/models/loras/x/t2v_low.safetensors"
assert "HELIOS_LORAS" not in popen_calls[-1]["env"], "one LoRA at full weight: Helios's script as it is"
assert popen_calls[-1]["image"] == "<first frame>@640x384", "the first frame is fitted to Helios's size"
assert not os.path.exists(os.path.dirname(arg("--image_path"))), "the work folder is removed afterwards"
# several LoRAs, or one at another weight: our runner around the same script, which loads them all with their weights
two = [{"path": "/l/avatar_low.safetensors", "weight": 1.0}, {"path": "/l/moves_low.safetensors", "weight": 0.8}]
Proc.output = 'loading\n{"event": "helios_loras", "adapters": ["default", "lora2"], "weights": [1.0, 0.8]}\nMax memory: 45.3 GB\n'
import contextlib as _ctx  # noqa: E402

_printed = io.StringIO()
with _ctx.redirect_stdout(_printed):
    helios.generate("she dances", 5, seed=1, loras=two)
Proc.output = "loading\nMax memory: 45.3 GB\n"
c = popen_calls[-1]["cmd"]
# what the runner says it loaded reaches the worker's log line, so "both LoRAs were applied" can be read there
_event = [__import__("json").loads(l) for l in _printed.getvalue().splitlines() if '"event": "helios"' in l][-1]
assert _event["loras"] == 2 and _event["adapters"] == {"default": 1.0, "lora2": 0.8}, _event
assert c[2] == helios.RUNNER and os.path.basename(helios.RUNNER) == "helios_run.py" and os.path.exists(helios.RUNNER), c[:3]
assert popen_calls[-1]["cwd"] == helios.REPO_DIR and arg("--lora_path") == "/l/avatar_low.safetensors"
import json as _json  # noqa: E402

assert _json.loads(popen_calls[-1]["env"]["HELIOS_LORAS"]) == two
helios.generate("she dances", 5, seed=1, loras=[{"path": "/l/a.safetensors", "weight": 0.6}])
c = popen_calls[-1]["cmd"]
assert c[2] == helios.RUNNER and _json.loads(popen_calls[-1]["env"]["HELIOS_LORAS"]) == [{"path": "/l/a.safetensors", "weight": 0.6}]

# the runner's patch: when Helios's script switches its one LoRA on, the others are loaded and all get their weights
import helios_run  # noqa: E402


class FakePipe:
    def __init__(self):
        self.loaded, self.set = [], []

    def load_lora_weights(self, path, adapter_name=None):
        self.loaded.append((path, adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.set.append((list(names), list(adapter_weights)))


helios_run.patch(FakePipe, two)
pipe = FakePipe()
pipe.load_lora_weights("/l/avatar_low.safetensors", adapter_name="default")  # what infer_helios.py does
pipe.set_adapters(["default"], adapter_weights=[1.0])
assert pipe.loaded == [("/l/avatar_low.safetensors", "default"), ("/l/moves_low.safetensors", "lora2")], pipe.loaded
assert pipe.set == [(["default", "lora2"], [1.0, 0.8])], pipe.set
pipe.set_adapters(["default", "lora2"], adapter_weights=[0.5, 0.5])  # any later call passes through untouched
assert pipe.set[-1] == (["default", "lora2"], [0.5, 0.5]) and len(pipe.loaded) == 2

assert len(frames) == 30 * 24 == 720 and frames[0] == "frame0", "cut to the length asked for (726 were rendered)"
# text only: text-to-video, no image or LoRA arguments; no seed given, none passed
frames = helios.generate("a beach at dawn", 5)
c = popen_calls[-1]["cmd"]
assert arg("--sample_type") == "t2v" and arg("--num_frames") == "132" and len(frames) == 120
assert "--image_path" not in c and "--lora_path" not in c and "--seed" not in c and c[2] == "infer_helios.py"
# Helios fails: its last lines are in the error, and no frames come back (criterion 8)
Proc.exit, Proc.output = 1, "line1\nTraceback (most recent call last):\nRuntimeError: CUDA out of memory\n"
try:
    helios.generate("x", 5)
    raise AssertionError("a failed run returned frames")
except RuntimeError as e:
    assert "exit 1" in str(e) and "CUDA out of memory" in str(e), e
# a cancelled job stops the process
Proc.exit, Proc.output = 0, "a\nb\n"


class Stop(Exception):
    pass


def cancel():
    raise Stop()


models.check_cancel = cancel
procs = []
real_proc = Proc


def tracking(*a, **k):
    p = real_proc(*a, **k)
    procs.append(p)
    return p


helios.subprocess = types.SimpleNamespace(Popen=tracking, PIPE=-1, STDOUT=-2, run=run)
try:
    helios.generate("x", 5)
    raise AssertionError("the cancel was swallowed")
except Stop:
    pass
assert procs[-1].killed
models.check_cancel = lambda: None

# ---------------------------------------------------------------- the video job (criteria 6, 10)
import enhance  # noqa: E402
import worker  # noqa: E402

assert worker.HELIOS_FRAME == (1280, 768), "the first frame is made in Helios's 5:3 shape, at twice its size"
worker.progress = lambda *a: None
worker.upload = lambda path, client, job_id, name, ctype: f"outputs/{name}"
calls = {"helios": [], "wan": [], "keyframe": [], "written": [], "sound": [], "upscaled": 0, "timeline": 0}


class Frame:
    def save(self, path):
        open(path, "w").write("frame")


KEY = Frame()
helios.generate = lambda prompt, seconds, seed=None, image=None, loras=(), progress=None: (
    calls["helios"].append(dict(prompt=prompt, seconds=seconds, seed=seed, image=image, loras=list(loras))) or [f"f{i}" for i in range(seconds * 24)])
models.generate_video = lambda out, **kw: calls["wan"].append(kw) or out
models.make_keyframe = lambda cast, scene, width, height, seed, best=False: calls["keyframe"].append((scene, width, height)) or KEY
models.write_video = lambda frames, out, fps=16, grain=None, x264=None: calls["written"].append((len(frames), out, fps, grain))
enhance.upscale = lambda frames, check=None: calls.__setitem__("upscaled", calls["upscaled"] + 1) or [f"big-{f}" for f in frames]
worker.sound_for_clip = lambda path, prompt, seconds, seed, workdir, tag="sound": calls["sound"].append((path, seconds, seed)) or "/s.wav"
worker.add_audio = lambda v, a, out: out
worker.krea_first_frames = lambda job, params, cast, seeds: [None] * len(seeds)
worker.take_timeline = lambda *a, **k: calls.__setitem__("timeline", calls["timeline"] + 1) or ["beat"] * 6
worker.describe_cast = lambda cast: "liron is a woman with long brown hair."
CAST = [("liron", ["ref"], "")]
worker.load_cast = lambda params, workdir: CAST
models.set_loras([{"adapter": "user_avatar_liron_wan_v1", "high": "/l/t2v_high.safetensors",
                   "low": "/l/t2v_low.safetensors", "weight": 1.0, "trigger": "liron"}])
JOB = {"client": "m", "job_id": "j1"}
P = {"prompt": "dancing in a living room", "negative_prompt": "", "width": 640, "height": 384, "seconds": 30, "fps": 24,
     "upscale": False, "sound": True, "video_model": "helios", "shot": "pov", "seed": 7}

res = worker.run_video(JOB, dict(P), tempfile.mkdtemp())
assert calls["wan"] == [] and len(calls["helios"]) == 1, "Helios renders the take; Wan is not called"
hcall = calls["helios"][0]
assert hcall["seconds"] == 30 and hcall["seed"] == 7 and hcall["image"] is KEY
assert hcall["loras"] == [{"path": "/l/t2v_low.safetensors", "weight": 1.0}], "the Wan LoRA's low-noise file, at its weight"
# two Wan LoRAs on the job: both go to Helios, each with its weight
models.set_loras([{"adapter": "a", "high": "/l/a_high.safetensors", "low": "/l/a_low.safetensors", "weight": 1.0, "trigger": "liron"},
                  {"adapter": "b", "high": "/l/b_high.safetensors", "low": "/l/b_low.safetensors", "weight": 0.7, "trigger": ""}])
worker.run_video(JOB, dict(P, sound=False, seconds=5), tempfile.mkdtemp())
assert calls["helios"][-1]["loras"] == [{"path": "/l/a_low.safetensors", "weight": 1.0}, {"path": "/l/b_low.safetensors", "weight": 0.7}]
calls["helios"].pop(); calls["written"].pop(); calls["keyframe"].pop()
import realism  # noqa: E402

realism.Grain = lambda seed: ("grain", seed)  # the real one needs numpy
prompt = hcall["prompt"]
assert "dancing in a living room" in prompt and realism.SHOTS["pov"] in prompt and "liron is a woman" in prompt
assert models.NATURAL_STYLE in prompt and models.TRIPOD not in prompt, "the natural-look text, no tripod with the phone option"
assert calls["keyframe"] == [(realism.with_style("dancing in a living room", P), 1280, 768)], calls["keyframe"]
assert calls["written"][0][0] == 720 and calls["written"][0][2] == 24
assert isinstance(calls["written"][0][3], realism.PhoneLook), "the phone option also works on the finished frames"
assert calls["sound"] and calls["sound"][0][1:] == (30, 7) and calls["upscaled"] == 0
assert calls["timeline"] == 0, "one prompt for the whole take: no planner"
assert res["video_key"] == "outputs/video.mp4" and res["keyframe_key"] == "outputs/keyframe.png" and "segment_prompts" not in res
assert res["video_model"] == "helios"

# upscale and grain work on the result; no sound when it is off; no avatar and no first frame: text-to-video
for k in calls:
    calls[k] = 0 if isinstance(calls[k], int) else []
worker.load_cast = lambda params, workdir: []
models.set_loras([])
res = worker.run_video(JOB, dict(P, upscale=True, grain=True, sound=False, shot=None, seconds=5), tempfile.mkdtemp())
hcall = calls["helios"][0]
assert hcall["image"] is None and hcall["loras"] == [] and hcall["seconds"] == 5 and calls["keyframe"] == []
assert models.TRIPOD in hcall["prompt"], "a plain prompt keeps the tripod sentence"
assert calls["upscaled"] == 1 and calls["written"][0][0] == 120 and calls["written"][0][3] is not None and calls["sound"] == []
assert "keyframe_key" not in res

# variations: one Helios run per seed
for k in calls:
    calls[k] = 0 if isinstance(calls[k], int) else []
res = worker.run_video(JOB, dict(P, variations=3, sound=False, seconds=10), tempfile.mkdtemp())
assert [c["seed"] for c in calls["helios"]] == [7, 8, 9] and res["seeds"] == [7, 8, 9] and calls["wan"] == []

# Wan: as before (criterion 10)
for k in calls:
    calls[k] = 0 if isinstance(calls[k], int) else []
res = worker.run_video(JOB, dict(P, video_model="wan", width=1280, height=704, seconds=5, fps=32, sound=False), tempfile.mkdtemp())
assert calls["helios"] == [] and len(calls["wan"]) == 1 and calls["wan"][0]["width"] == 1280 and "video_model" not in res
# a job from before this change has no video_model: Wan
res = worker.run_video(JOB, {k: v for k, v in dict(P, width=1280, height=704, seconds=5, sound=False).items() if k != "video_model"},
                       tempfile.mkdtemp())
assert calls["helios"] == [] and len(calls["wan"]) == 2

# worker start: the Helios weights are queued for the mirror once they are complete
root2 = tempfile.mkdtemp()
helios.MODEL_DIR = os.path.join(root2, "helios", "abc")
os.makedirs(helios.MODEL_DIR)
queued = []
mirror.push_async = lambda unit, folder, files=None: queued.append(unit)
models.BF16_DIR = models.MMAUDIO_DIR = krea.MODEL_DIR = krea.FILTER_DIR = "/nonexistent"
worker.mirror_backfill(hf_home="/nonexistent")
assert helios.MODEL_UNIT not in queued
open(os.path.join(helios.MODEL_DIR, ".complete"), "w").close()
worker.mirror_backfill(hf_home="/nonexistent")
assert helios.MODEL_UNIT in queued

print("helios worker tests passed")
