"""API unit tests (stubbed S3/DynamoDB). Run: make test (or python3 -B tests/test_api.py)."""
import json
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install(os.path.join("lambda", "api"))
import handler as h  # noqa: E402

s3 = h.s3 = _stubs.FakeS3()
h._s3_json = s3.json
clock = [1000]
h.time = mock.MagicMock(time=lambda: clock[0])
body = lambda r: json.loads(r["body"])  # noqa: E731


def tick():
    clock[0] += 10


def uploads(*names):
    for n in names:
        s3.objects[f"uploads/m/{n}"] = n.encode()
    return [f"uploads/m/{n}" for n in names]


def content(keys):
    return [s3.objects[k].decode() for k in keys]


# ---------------------------------------------------------------- avatars (multi-image)
u = uploads("a.png", "b.png", "c.png")
r = h.create_avatar("m", {"name": "dana", "upload_keys": u, "description": "d"})
assert r["statusCode"] == 200 and body(r)["images"] == 3, body(r)
av = h._get_avatar("m", "dana")
assert content(h._image_keys(av)) == ["a.png", "b.png", "c.png"] and av["reference_key"] == h._image_keys(av)[0]
for bad in [{"upload_keys": []}, {"upload_keys": uploads(*[f"x{i}.png" for i in range(13)])},
            {"upload_keys": ["uploads/other/x.png"]}, {"upload_keys": ["uploads/m/x.mp4"]}]:
    snapshot = dict(s3.objects)
    assert h.create_avatar("m", dict({"name": "dana"}, **bad))["statusCode"] == 400, bad
    assert {k: v for k, v in s3.objects.items() if not k.startswith("uploads/")} == \
           {k: v for k, v in snapshot.items() if not k.startswith("uploads/")}, bad
tick()
v1 = av["created_at"]
assert body(h.add_avatar_images("m", "dana", {"upload_keys": uploads("d.png")}))["images"] == 4
assert content(h._image_keys(h._s3_json(f"avatars/m/dana/history/{v1}/avatar.json"))) == ["a.png", "b.png", "c.png"]
tick()
assert h.set_main_avatar_image("m", "dana", {"index": 3})["statusCode"] == 200
assert content(h._image_keys(h._get_avatar("m", "dana")))[0] == "c.png"
tick()
assert h.remove_avatar_image("m", "dana", "9")["statusCode"] == 404
assert body(h.remove_avatar_image("m", "dana", "2"))["images"] == 3
tick()
assert h.restore_avatar("m", "dana", {"version": str(v1)})["statusCode"] == 200
assert content(h._image_keys(h._get_avatar("m", "dana"))) == ["a.png", "b.png", "c.png"]
s3.put_json("avatars/m/old/avatar.json", {"name": "old", "description": "", "reference_key": "avatars/m/old/r.png",
                                         "created_at": 1})
assert h._image_keys(h._get_avatar("m", "old")) == ["avatars/m/old/r.png"], "legacy avatar = one image"
assert h._resolve_avatars("m", "dana")[0]["reference_keys"] == h._image_keys(h._get_avatar("m", "dana"))[:3]

# ---------------------------------------------------------------- photos (shot fields)
shot = {"where": "on a rooftop", "wearing": "a suit", "doing": "walking", "camera": "low angle"}
p = h._validate_photos("m", {"avatar": "dana", "scenes": [{"shot": shot}, {"prompt": "reading"}], "count": 3})
assert p["scenes"] == [{"shot": shot}, {"prompt": "reading"}] and p["reference_keys"]
for bad in [{"shot": {}}, {"shot": {"mood": "x"}}, {"shot": "x"}, {"shot": {"where": "x" * 5000}}]:
    _stubs.expect_error(h._validate_photos, "m", {"avatar": "dana", "scenes": [bad]})

# ---------------------------------------------------------------- photos: series and realism
p = h._validate_photos("m", {"avatar": "dana", "series": {"where": "a Lisbon café", "doing": ["reading", " "]},
                             "count": 5, "look": "phone", "grain": True, "quality": "best"})
assert p["series"] == {"where": "a Lisbon café", "doing": ["reading"], "anchor": True} and p["count"] == 5
assert (p["look"], p["grain"], p["natural_skin"], p["quality"]) == ("phone", True, False, "best")
p = h._validate_photos("m", {"avatar": "dana", "scenes": [{"prompt": "x"}]})
assert "series" not in p and (p["look"], p["quality"], p["grain"]) == ("pro", "standard", False)
assert h._validate_photos("m", {"avatar": "dana", "series": {"where": "x", "anchor": False}})["series"]["anchor"] is False
for bad, msg in [({"series": {"wearing": "x"}}, "where"), ({"series": {"where": "x"}, "scenes": [{"prompt": "y"}]}, "not both"),
                 ({"series": {"where": "x"}, "count": 1}, "count 2-8"),
                 ({"series": {"where": "x", "doing": ["a", "b", "c"]}, "count": 2}, "at most count"),
                 ({"series": {"where": "x", "doing": "a"}}, "list"), ({"series": {"where": "x", "camera": "y"}}, "series takes"),
                 ({"look": "anime"}, "look"), ({"quality": "draft"}, "quality")]:
    _stubs.expect_error(h._validate_photos, "m", dict({"avatar": "dana"}, **bad), contains=msg)
text, info = h._summary("photos", h._validate_photos("m", {"avatar": "dana", "series": {"where": "a pier"},
                                                            "count": 3, "candid": True}))
assert text == "series of 3 at a pier" and info["series"] and info["candid"] and "look" not in info, (text, info)

# ---------------------------------------------------------------- video and story params
_, v = h._validate("m", {"type": "video", "prompt": "a lighthouse"})
assert v["fps"] == 32 and v["upscale"] is False and v["variations"] == 1 and v["seconds"] == 5
_, v = h._validate("m", {"type": "video", "prompt": "x", "seconds": 15, "fps": 16, "upscale": True})
assert (v["seconds"], v["fps"], v["upscale"]) == (15, 16, True)
_, v = h._validate("m", {"type": "video", "prompt": "x", "variations": 4, "seconds": 5})
assert v["variations"] == 4
_, v = h._validate("m", {"type": "video", "prompt": "x", "seconds": 50})
assert v["seconds"] == 50 and "beats" not in v, "long single takes up to 60 s"
# beats: "SECONDS: text" separated by ; or newlines, or a list of {at, prompt}
BEATS3 = [{"at": 0, "prompt": "walks in"}, {"at": 10, "prompt": "sits down"}, {"at": 30, "prompt": "looks up"}]
_, v = h._validate("m", {"type": "video", "prompt": "x", "seconds": 50, "beats": "0: walks in; 10: sits down\n30: looks up"})
assert v["beats"] == BEATS3, v.get("beats")
_, v = h._validate("m", {"type": "video", "prompt": "x", "seconds": 50, "beats": BEATS3})
assert v["beats"] == BEATS3
assert h._summary("video", v)[1]["beats"] == 3 and "beats" not in h._summary("video", {"seconds": 5})[1]
_, v = h._validate("m", {"type": "video", "prompt": "x", "seconds": 50, "anchor_every": 3})
assert v["anchor_every"] == 3 and h._summary("video", v)[1]["anchor_every"] == 3
assert "anchor_every" not in h._validate("m", {"type": "video", "prompt": "x", "seconds": 50})[1]
for bad, msg in [({"anchor_every": 1}, "anchor_every must be 2-6"), ({"anchor_every": 7}, "2-6"),
                 ({"anchor_every": "x"}, "2-6"), ({"anchor_every": True}, "2-6"),
                 ({"anchor_every": 3, "seconds": 10}, "over 10 s")]:
    _stubs.expect_error(h._validate, "m", dict({"type": "video", "prompt": "x", "seconds": 50}, **bad), contains=msg)
_, v = h._validate("m", {"type": "video", "prompt": "x", "seconds": 50, "beats": "  "})
assert "beats" not in v, "empty beats are no beats"
for beats, secs, msg in [("5: a; 10: b", 50, "first beat must be at 0"),
                         ("0: a; 10: b; 10: c", 50, "increasing"),
                         ("0: a; 50: b", 50, "less than seconds"),
                         ("0: a; 48: b", 50, "no segment"),
                         ("0: a; 2: b; 4: c", 50, "same segment"),
                         ("; ".join(f"{i}: b{i}" for i in range(13)), 50, "1-12 beats"),
                         ("0: a", 5, "over 5 s"),
                         ("0: ; 10: b", 50, "1-500 chars"),
                         ("0: " + "x" * 501, 50, "1-500 chars"),
                         ("walks in", 50, "SECONDS: text"),
                         ([{"at": "soon", "prompt": "a"}], 50, "SECONDS: text"),
                         (7, 50, "string or a list")]:
    _stubs.expect_error(h._validate, "m", {"type": "video", "prompt": "x", "seconds": secs, "beats": beats}, contains=msg)
for bad, msg in [({"seconds": 61}, "seconds must be 1-60"), ({"seconds": 0}, "1-60"), ({"variations": 5}, "variations"),
                 ({"variations": 2, "seconds": 6}, "up to 5"), ({"fps": 24}, "fps")]:
    _stubs.expect_error(h._validate, "m", dict({"type": "video", "prompt": "x"}, **bad), contains=msg)
_, st = h._validate("m", {"type": "story", "story": "once", "upscale": True})
assert st["fps"] == 32 and st["upscale"] is True and "width" not in st
_, v = h._validate("m", {"type": "video", "prompt": "x", "quality": "draft", "width": 1280})
assert (v["width"], v["height"], v["upscale"], v["quality"]) == (832, 480, True, "draft")
_, st = h._validate("m", {"type": "story", "story": "once", "quality": "draft"})
assert (st["width"], st["height"], st["upscale"]) == (832, 480, True)
_stubs.expect_error(h._validate, "m", {"type": "video", "prompt": "x", "quality": "ultra"}, contains="quality")
_, v = h._validate("m", {"type": "video", "prompt": "x", "quality": "best", "look": "film", "natural_skin": True})
assert (v["quality"], v["look"], v["natural_skin"], v["candid"], v["grain"]) == ("best", "film", True, False, False)
assert "width" not in v or v["width"] != 832, "best keeps full size"
_, st = h._validate("m", {"type": "story", "story": "once", "grain": True, "look": "phone"})
assert st["grain"] is True and st["look"] == "phone"
_stubs.expect_error(h._validate, "m", {"type": "story", "story": "once", "look": "anime"}, contains="look")
assert h._summary("video", v)[1]["look"] == "film" and "look" not in h._summary("video", h._validate("m", {"type": "video", "prompt": "x"})[1])[1]

# ---------------------------------------------------------------- Krea 2: where it can be chosen
_, a = h._validate("m", {"type": "avatar", "description": "a sailor", "model": "krea2"})
assert a["model"] == "krea2" and h._validate("m", {"type": "avatar", "description": "x"})[1]["model"] == "wan"
_stubs.expect_error(h._validate, "m", {"type": "avatar", "description": "x", "model": "sd15"}, contains="model")
p = h._validate_photos("m", {"avatar": "none", "scenes": [{"prompt": "a harbour"}], "look": "film"})
assert (p["model"], p["subject"], p["look"]) == ("krea2", "none", "film") and "reference_keys" not in p
assert h._validate_photos("m", {"model": "krea2", "scenes": [{"prompt": "x"}]})["model"] == "krea2"
assert h._validate_photos("m", {"avatar": "dana", "scenes": [{"prompt": "x"}]})["model"] == "qwen"
for bad, msg in [({"model": "krea2", "avatar": "dana"}, "no Krea 2 LoRA"), ({"model": "krea2", "upload_key": "uploads/m/a.png"}, "from text"),
                 ({"model": "krea2", "series": {"where": "x"}}, "series needs an avatar"),
                 ({"model": "krea2", "quality": "best"}, "one speed"), ({"model": "sd15"}, "model"),
                 ({"model": "qwen", "avatar": "none"}, "need a person")]:
    _stubs.expect_error(h._validate_photos, "m", bad, contains=msg)
_, v = h._validate("m", {"type": "video", "prompt": "x", "image_model": "krea2"})
assert v["image_model"] == "krea2" and h._summary("video", v)[1]["first_frame"] == "krea2"
assert h._validate("m", {"type": "video", "prompt": "x"})[1]["image_model"] == "wan"
_stubs.expect_error(h._validate, "m", {"type": "story", "story": "x", "image_model": "flux"}, contains="image_model")
saved = (h._queue_job, h._resolve_loras, h._resolve_avatars)
h._queue_job = lambda client, job_type, params, rerun_of=None: {"statusCode": 202, "body": "{}"}
h._resolve_loras = lambda client, req: []
h._resolve_avatars = lambda client, req: [{"name": "dana"}] if req not in ("none",) else []
assert h.create_job("m", {"type": "video", "prompt": "x", "image_model": "krea2", "avatar": "dana"})["statusCode"] == 400
assert h.create_job("m", {"type": "video", "prompt": "x", "image_model": "krea2", "avatar": "none"})["statusCode"] == 202
assert h.create_job("m", {"type": "story", "story": "x", "image_model": "krea2", "avatar": "dana"})["statusCode"] == 202
# anchor_every: re-anchor the one avatar's face every N segments of a long take
h._resolve_avatars = lambda client, req: [] if req == "none" else [{"name": n} for n in str(req or "dana").split(",")]
assert h.create_job("m", {"type": "video", "prompt": "x", "seconds": 50, "anchor_every": 3, "avatar": "dana"})["statusCode"] == 202
for avatar in ("none", "dana,lee"):
    r = h.create_job("m", {"type": "video", "prompt": "x", "seconds": 50, "anchor_every": 3, "avatar": avatar})
    assert r["statusCode"] == 400 and "exactly one avatar" in r["body"], r
h._queue_job, h._resolve_loras, h._resolve_avatars = saved

# ---------------------------------------------------------------- write (story text from a premise)
_, w = h._validate("m", {"type": "write", "premise": "  a lighthouse keeper  ", "genre": "cozy", "seed": "7"})
assert w == {"premise": "a lighthouse keeper", "words": 800, "genre": "cozy", "seed": 7}, w
for bad, msg in [({"premise": ""}, "premise"), ({"premise": "x" * 2001}, "premise"), ({"premise": "x", "words": 50}, "words"),
                 ({"premise": "x", "words": 2001}, "words"), ({"premise": "x", "genre": "g" * 201}, "genre")]:
    _stubs.expect_error(h._validate, "m", dict({"type": "write"}, **bad), contains=msg)
assert h._summary("write", w) == ("a lighthouse keeper", {"words": 800, "genre": "cozy"})

# ---------------------------------------------------------------- LoRAs: resolve, pins, pick, previews
files = ["t2v_high.safetensors", "t2v_low.safetensors"] + [
    f"checkpoints/{e}-step{n:06d}.safetensors" for e in ("t2v_high", "t2v_low") for n in (250, 500, 750, 1000, 1250)]
s3.put_json("loras/m/jerk/current.json", {"version": 1})
s3.put_json("loras/m/jerk/v1/lora.json", {"version": 1, "models": ["t2v"], "trigger": "vgn_jerk", "files": files})
s3.put_json("loras/m/style/current.json", {"version": 1})
s3.put_json("loras/m/style/v1/lora.json", {"version": 1, "models": ["t2v"], "trigger": "", "files": files[:2],
                                          "steps_per_expert": 600})
r = h._resolve_loras("m", [{"name": "Jerk", "weight": 0.9}])[0]
assert (r["version"], r["step"], r["weight"], r["trigger"]) == (1, None, 0.9, "vgn_jerk")
assert r["high_key"] == "loras/m/jerk/v1/t2v_high.safetensors"
assert [x["name"] for x in h._resolve_loras("m", "jerk:0.5, style")] == ["jerk", "style"]
for bad, msg in [([{"name": "nope"}], "not found"), ([{"name": "jerk", "weight": 2}], "0.0-1.5"),
                 ([{"name": n} for n in ("jerk", "style", "x", "y", "z")], "at most 2"),
                 ([{"name": "jerk"}, {"name": "jerk"}], "twice"), ([{"name": "jerk", "step": 600}], "step 600"),
                 ([{"name": "style", "step": 500}], "no checkpoint")]:
    _stubs.expect_error(h._resolve_loras, "m", bad, contains=msg)
assert h._resolve_loras("m", [{"name": "jerk", "version": 1, "step": 500}])[0]["high_key"].endswith("step000500.safetensors")
# the star avatar's ready Wan LoRA is attached to video jobs first, at 1.0, within the 2-Wan-LoRA limit
s3.put_json("loras/m/avatar-dana-wan/current.json", {"version": 2})
s3.put_json("loras/m/avatar-dana-wan/v2/lora.json", {"version": 2, "models": ["t2v"], "trigger": "dana",
                                                     "files": ["t2v_high.safetensors", "t2v_low.safetensors"]})
saved_aw = (h._queue_job, h._resolve_avatars)
got_params = []
h._queue_job = lambda client, job_type, params, rerun_of=None: got_params.append(params) or {"statusCode": 202, "body": "{}"}
dana_wan = {"name": "dana", "wan_lora": {"state": "ready", "ready": {"lora": "avatar-dana-wan", "version": 2}}}
h._resolve_avatars = lambda client, req: [] if req == "none" else [dict(dana_wan)]
assert h.create_job("m", {"type": "video", "prompt": "x"})["statusCode"] == 202
pl = got_params[-1]["loras"]
assert pl[0]["name"] == "avatar-dana-wan" and pl[0]["version"] == 2 and pl[0]["weight"] == 1.0 and pl[0]["step"] is None, pl
assert pl[0]["high_key"] == "loras/m/avatar-dana-wan/v2/t2v_high.safetensors"
assert h._summary("video", got_params[-1])[1]["loras"][0].startswith("avatar-dana-wan")
h.create_job("m", {"type": "video", "prompt": "x", "loras": "jerk:0.5"})
assert [l["name"] for l in got_params[-1]["loras"]] == ["avatar-dana-wan", "jerk"], "one user LoRA beside it"
r = h.create_job("m", {"type": "video", "prompt": "x", "loras": "jerk, style"})
assert r["statusCode"] == 400 and "avatar" in body(r)["error"], "two user LoRAs plus the avatar's: limit named"
h.create_job("m", {"type": "video", "prompt": "x", "loras": "jerk", "avatar_lora": False})
assert [l["name"] for l in got_params[-1]["loras"]] == ["jerk"], "avatar_lora false: not attached"
h.create_job("m", {"type": "video", "prompt": "x", "loras": "avatar-dana-wan:0.7"})
assert [(l["name"], l["weight"]) for l in got_params[-1]["loras"]] == [("avatar-dana-wan", 0.7)], "named by the user: theirs"
h.create_job("m", {"type": "story", "story": "once upon a time"})
assert not got_params[-1].get("loras"), "stories don't get it"
h._resolve_avatars = lambda client, req: [dict(dana_wan, wan_lora={"state": "ready", "ready": {"lora": "avatar-dana-wan", "version": 9}})]
assert h.create_job("m", {"type": "video", "prompt": "x"})["statusCode"] == 202 and not got_params[-1].get("loras"), \
    "a broken link never fails the job"
h._queue_job, h._resolve_avatars = saved_aw
assert h.set_lora_current("m", "jerk", {"version": 1, "step": 750})["statusCode"] == 200
assert h._resolve_loras("m", [{"name": "jerk"}])[0]["step"] == 750
assert h._resolve_loras("m", [{"name": "jerk", "version": 1}])[0]["step"] is None, "pinned version = final"
for bad in [{"version": 1, "step": 600}, {"version": 9}, {"version": "x"}]:
    before = s3.json("loras/m/jerk/current.json")
    assert h.set_lora_current("m", "jerk", bad)["statusCode"] in (400, 404), bad
    assert s3.json("loras/m/jerk/current.json") == before
assert h.set_lora_current("m", "jerk", {"version": 1})["statusCode"] == 200
_, pv = h._validate("m", {"type": "lora_preview", "name": "jerk", "seed": 5})
assert [c["step"] for c in pv["clips"]] == [250, 500, 750, 1000, 1250] and pv["seed"] == 5
_, pv = h._validate("m", {"type": "lora_preview", "name": "jerk", "steps": [1000, 500, 500]})
assert [c["step"] for c in pv["clips"]] == [500, 1000]
_stubs.expect_error(h._validate, "m", {"type": "lora_preview", "name": "jerk", "steps": [600]})
_, pv = h._validate("m", {"type": "lora_preview", "name": "style"})
assert [c["step"] for c in pv["clips"]] == [600] and pv["clips"][0]["high_key"] == "loras/m/style/v1/t2v_high.safetensors"

# ---------------------------------------------------------------- uploaded LoRAs: reserve, check headers, register
h.ClientError = KeyError


def safetensors(tensors, pad=64):
    """A minimal .safetensors file: header only (the checks never read tensor data)."""
    head = json.dumps({k: {"dtype": d, "shape": shp, "data_offsets": [0, 0]} for k, (d, shp) in tensors.items()}).encode()
    return len(head).to_bytes(8, "little") + head + b"\0" * pad


WAN = {"diffusion_model.blocks.0.self_attn.q.lora_down.weight": ("BF16", [32, 5120]),
       "diffusion_model.blocks.0.self_attn.q.lora_up.weight": ("BF16", [5120, 32])}
r = h.lora_uploads("m", "Noir", {"files": "single"})
up = body(r)
assert r["statusCode"] == 200 and up["version"] == 1 and list(up["urls"]) == ["high"]
assert up["urls"]["high"].endswith("loras/m/noir/v1/t2v_high.safetensors")
assert h.list_loras("m") and "noir" not in [l["name"] for l in body(h.list_loras("m"))["loras"]], "pending = invisible"
assert h.register_lora_upload("m", "noir", "1", {})["statusCode"] == 400, "nothing uploaded yet"
s3.objects["loras/m/noir/v1/t2v_high.safetensors"] = safetensors(WAN)
r = h.register_lora_upload("m", "noir", "1", {"trigger": "noir style", "description": "from somewhere"})
assert r["statusCode"] == 200 and body(r)["single_file"] is True and body(r)["current"] is True, body(r)
meta = s3.json("loras/m/noir/v1/lora.json")
assert meta["source"] == "upload" and meta["files"] == ["t2v_high.safetensors"] and meta["trigger"] == "noir style"
assert s3.json("loras/m/noir/current.json") == {"version": 1}
res = h._resolve_loras("m", [{"name": "noir"}])[0]
assert res["high_key"] == res["low_key"] == "loras/m/noir/v1/t2v_high.safetensors" and res["trigger"] == "noir style"
assert h.register_lora_upload("m", "noir", "1", {})["statusCode"] == 400, "a version registers once"
listed = [l for l in body(h.list_loras("m"))["loras"] if l["name"] == "noir"][0]
assert listed["versions"][0]["source"] == "upload"
# a pair as v2 of the same name, kept off current
up = body(h.lora_uploads("m", "noir", {"files": "pair"}))
assert up["version"] == 2 and set(up["urls"]) == {"high", "low"}
s3.objects["loras/m/noir/v2/t2v_high.safetensors"] = safetensors(WAN)
s3.objects["loras/m/noir/v2/t2v_low.safetensors"] = safetensors(WAN)
r = body(h.register_lora_upload("m", "noir", "2", {"make_current": False}))
assert r["single_file"] is False and r["current"] is False and s3.json("loras/m/noir/current.json") == {"version": 1}
assert h._resolve_loras("m", [{"name": "noir", "version": 2}])[0]["low_key"].endswith("v2/t2v_low.safetensors")
# files that must be refused (each as a fresh version)
bad_files = [
    (b"not a safetensors file at all", "not a .safetensors"),
    (safetensors({"model.layers.0.weight": ("BF16", [5120, 5120])}), "not a LoRA"),
    (safetensors({"double_blocks.0.img_attn.qkv.lora_A.weight": ("BF16", [16, 3072])}), "different model"),
    (safetensors({"blocks.0.self_attn.q.lora_A.weight": ("BF16", [16, 1536])}), "smaller Wan"),
    (safetensors({"blocks.0.self_attn.q.lora_A.weight": ("I8", [16, 5120])}), "weight type"),
    (safetensors({"some.random.lora_A.weight": ("BF16", [16, 5120])}), "Wan LoRA"),
]
for data, msg in bad_files:
    v = body(h.lora_uploads("m", "junk", {}))["version"]
    s3.objects[f"loras/m/junk/v{v}/t2v_high.safetensors"] = data
    r = h.register_lora_upload("m", "junk", str(v), {})
    assert r["statusCode"] == 400 and msg in body(r)["error"], (msg, body(r))
    assert s3.json(f"loras/m/junk/v{v}/lora.json") is None
# other common layouts are accepted: musubi/kohya underscores, diffusers attn1/lora_A
for good in [{"lora_unet_blocks_0_self_attn_q.lora_down.weight": ("F16", [16, 5120])},
             {"transformer.blocks.3.attn1.to_q.lora_A.weight": ("F32", [8, 5120])}]:
    v = body(h.lora_uploads("m", "fmt", {}))["version"]
    s3.objects[f"loras/m/fmt/v{v}/t2v_high.safetensors"] = safetensors(good)
    assert h.register_lora_upload("m", "fmt", str(v), {})["statusCode"] == 200, good
big = body(h.lora_uploads("m", "huge", {}))["version"]
s3.objects[f"loras/m/huge/v{big}/t2v_high.safetensors"] = safetensors(WAN)
h.MAX_LORA_BYTES = 100
assert "limit" in body(h.register_lora_upload("m", "huge", str(big), {}))["error"]
h.MAX_LORA_BYTES = 2 * 1024 ** 3
assert h.lora_uploads("m", "Bad Name", {})["statusCode"] == 400
assert h.lora_uploads("m", "x", {"files": "three"})["statusCode"] == 400
assert h.register_lora_upload("m", "noir", "1", {"trigger": "a b c d"})["statusCode"] == 400

# ---------------------------------------------------------------- Krea 2 LoRA uploads
# Layout of a real Krea 2 LoRA (kera2_Unlocked_V3): musubi-flattened main blocks and text-fusion
# blocks, BF16 down/up + F32 alpha, 6144-wide (48 heads x 128).
KREA = {"lora_unet_blocks_0_attn_wq.lora_down.weight": ("BF16", [32, 6144]),
        "lora_unet_blocks_0_attn_wq.lora_up.weight": ("BF16", [6144, 32]),
        "lora_unet_blocks_0_attn_wq.alpha": ("F32", []),
        "lora_unet_blocks_0_mlp_gate.lora_down.weight": ("BF16", [32, 6144]),
        "lora_unet_txtfusion_refiner_blocks_0_attn_gate.lora_down.weight": ("BF16", [32, 2560])}
krea_header = json.loads(safetensors(KREA)[8:-64])
assert h._check_krea2_lora(krea_header) == 4
_stubs.expect_error(h._check_wan_lora, krea_header, contains="this is a Krea 2 LoRA", exc=ValueError)
_stubs.expect_error(h._check_krea2_lora, json.loads(safetensors(WAN)[8:-64]), contains="this is a Wan LoRA",
                    exc=ValueError)
r = h.lora_uploads("m", "moody", {"files": "single", "model": "krea2"})
up = body(r)
assert r["statusCode"] == 200 and up["version"] == 1 and list(up["urls"]) == ["high"], up
assert up["urls"]["high"].endswith("loras/m/moody/v1/krea2.safetensors") and up["model"] == "krea2"
s3.objects["loras/m/moody/v1/krea2.safetensors"] = safetensors(KREA)
r = h.register_lora_upload("m", "moody", "1", {"trigger": "mdy"})
assert r["statusCode"] == 200 and body(r)["model"] == "krea2", body(r)
meta = s3.json("loras/m/moody/v1/lora.json")
assert meta["models"] == ["krea2"] and meta["files"] == ["krea2.safetensors"] and meta["single_file"] is True
listed = [l for l in body(h.list_loras("m"))["loras"] if l["name"] == "moody"][0]
assert listed["versions"][0]["models"] == ["krea2"]
assert h.lora_uploads("m", "moody2", {"files": "pair", "model": "krea2"})["statusCode"] == 400, "Krea is one file"
assert h.lora_uploads("m", "moody2", {"model": "flux"})["statusCode"] == 400
# one kind per name
r = h.lora_uploads("m", "noir", {"model": "krea2"})
assert r["statusCode"] == 400 and "is a Wan LoRA" in body(r)["error"], body(r)
r = h.lora_uploads("m", "moody", {})
assert r["statusCode"] == 400 and "is a Krea 2 LoRA" in body(r)["error"], body(r)
# a Krea file under a Wan reservation (and the reverse) gets the specific message
v = body(h.lora_uploads("m", "mixup", {}))["version"]
s3.objects[f"loras/m/mixup/v{v}/t2v_high.safetensors"] = safetensors(KREA)
r = h.register_lora_upload("m", "mixup", str(v), {})
assert r["statusCode"] == 400 and "this is a Krea 2 LoRA" in body(r)["error"], body(r)
v = body(h.lora_uploads("m", "mixup2", {"model": "krea2"}))["version"]
s3.objects[f"loras/m/mixup2/v{v}/krea2.safetensors"] = safetensors(WAN)
r = h.register_lora_upload("m", "mixup2", str(v), {})
assert r["statusCode"] == 400 and "this is a Wan LoRA" in body(r)["error"], body(r)
# Krea files that must be refused
krea_bad = [
    (safetensors({"lora_unet_blocks_0_attn_wq.weight": ("BF16", [6144, 6144])}), "not a LoRA"),
    (safetensors({"double_blocks.0.img_attn.qkv.lora_A.weight": ("BF16", [16, 3072])}), "different model"),
    (safetensors({"lora_unet_blocks_0_attn_wq.lora_down.weight": ("BF16", [32, 3072])}), "6144"),
    (safetensors({"lora_unet_blocks_0_attn_wq.lora_down.weight": ("I8", [32, 6144])}), "weight type"),
    (safetensors({"some.random.lora_A.weight": ("BF16", [16, 6144])}), "Krea 2 LoRA"),
]
for data, msg in krea_bad:
    v = body(h.lora_uploads("m", "kjunk", {"model": "krea2"}))["version"]
    s3.objects[f"loras/m/kjunk/v{v}/krea2.safetensors"] = data
    r = h.register_lora_upload("m", "kjunk", str(v), {})
    assert r["statusCode"] == 400 and msg in body(r)["error"], (msg, body(r))
    assert s3.json(f"loras/m/kjunk/v{v}/lora.json") is None
# The CLI's own kind detection (it can't import handler.py) agrees with the server's checks
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cli"))
import vidgen  # noqa: E402

agreement = [KREA, WAN, {"lora_unet_blocks_0_self_attn_q.lora_down.weight": ("F16", [16, 5120])},
             {"transformer.blocks.3.attn1.to_q.lora_A.weight": ("F32", [8, 5120])},
             {"diffusion_model.blocks.0.attn.to_q.lora_A.weight": ("BF16", [16, 6144])},
             {"double_blocks.0.img_attn.qkv.lora_A.weight": ("BF16", [16, 3072])},
             {"model.layers.0.weight": ("BF16", [5120, 5120])}]
for const in ("LORA_KEY", "WAN_MARKERS", "KREA2_MODULE"):  # the copied rules stay identical
    assert getattr(vidgen, const).pattern == getattr(h, const).pattern, const
assert vidgen.MAX_SAFETENSORS_HEADER == h.MAX_SAFETENSORS_HEADER and vidgen.OTHER_MODEL_KEYS == h.OTHER_MODEL_KEYS
# another family's LoRA (diffusers Flux naming looks Krea-ish) is refused by the CLI before uploading
flux = json.loads(safetensors({"transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": ("BF16", [16, 3072])})[8:-64])
_stubs.expect_error(vidgen.lora_kind, flux, contains="different model", exc=vidgen.ApiError)
for tensors in agreement:
    hd = json.loads(safetensors(tensors)[8:-64])
    server = None
    for kind, check in (("krea2", h._check_krea2_lora), ("t2v", h._check_wan_lora)):
        try:
            check(hd)
            server = kind
        except ValueError:
            pass
    assert server is None or vidgen.lora_kind(hd) == server, (tensors, server, vidgen.lora_kind(hd))
# one kind per name also at register: a Krea reservation, then a Wan one under the same (still
# unregistered) name; whichever registers second is refused
v1 = body(h.lora_uploads("m", "race", {"model": "krea2"}))["version"]
s3.objects[f"loras/m/race/v{v1}/krea2.safetensors"] = safetensors(KREA)
v2 = body(h.lora_uploads("m", "race", {}))["version"]
s3.objects[f"loras/m/race/v{v2}/t2v_high.safetensors"] = safetensors(WAN)
assert h.register_lora_upload("m", "race", str(v2), {})["statusCode"] == 200
r = h.register_lora_upload("m", "race", str(v1), {})
assert r["statusCode"] == 400 and "is a Wan LoRA" in body(r)["error"], body(r)
assert s3.json(f"loras/m/race/v{v1}/lora.json") is None
# a version holding both kinds of file is refused, not silently registered as one
v = body(h.lora_uploads("m", "both", {"model": "krea2"}))["version"]
s3.objects[f"loras/m/both/v{v}/krea2.safetensors"] = safetensors(KREA)
s3.objects[f"loras/m/both/v{v}/t2v_high.safetensors"] = safetensors(WAN)
r = h.register_lora_upload("m", "both", str(v), {})
assert r["statusCode"] == 400 and "both" in body(r)["error"], body(r)
# nothing uploaded: one message naming both file kinds
v = body(h.lora_uploads("m", "empty", {"model": "krea2"}))["version"]
r = h.register_lora_upload("m", "empty", str(v), {})
assert r["statusCode"] == 400 and "no file uploaded" in body(r)["error"], body(r)
# Krea path: size limit and non-safetensors
v = body(h.lora_uploads("m", "kbig", {"model": "krea2"}))["version"]
s3.objects[f"loras/m/kbig/v{v}/krea2.safetensors"] = safetensors(KREA)
h.MAX_LORA_BYTES = 100
assert "limit" in body(h.register_lora_upload("m", "kbig", str(v), {}))["error"]
h.MAX_LORA_BYTES = 2 * 1024 ** 3
v = body(h.lora_uploads("m", "kbig", {"model": "krea2"}))["version"]
s3.objects[f"loras/m/kbig/v{v}/krea2.safetensors"] = b"nope"
assert "not a .safetensors" in body(h.register_lora_upload("m", "kbig", str(v), {}))["error"]
# malformed header entries are a 400, not a crash
for weird in ({"x.lora_A.weight": 1}, {"x.lora_A.weight": {"dtype": "BF16", "shape": 5}},
              {"x.lora_A.weight": {"dtype": "BF16", "shape": [[1]]}}):
    v = body(h.lora_uploads("m", "weird", {"model": "krea2"}))["version"]
    head = json.dumps(weird).encode()
    s3.objects[f"loras/m/weird/v{v}/krea2.safetensors"] = len(head).to_bytes(8, "little") + head + b"\0" * 64
    r = h.register_lora_upload("m", "weird", str(v), {})
    assert r["statusCode"] == 400 and "bad header" in body(r)["error"], (weird, body(r))
# a Krea version can be made current (no steps); a step is refused
assert s3.json("loras/m/moody/current.json") == {"version": 1}
v = body(h.lora_uploads("m", "moody", {"model": "krea2"}))["version"]
s3.objects[f"loras/m/moody/v{v}/krea2.safetensors"] = safetensors(KREA)
assert h.register_lora_upload("m", "moody", str(v), {"make_current": False})["statusCode"] == 200
assert s3.json("loras/m/moody/current.json") == {"version": 1}
r = h.set_lora_current("m", "moody", {"version": v})
assert r["statusCode"] == 200 and s3.json("loras/m/moody/current.json")["version"] == v, body(r)
assert h.set_lora_current("m", "moody", {"version": v, "step": 500})["statusCode"] == 400
h.set_lora_current("m", "moody", {"version": 1})

# ---------------------------------------------------------------- LoRA files for a local copy
r = h.lora_files("m", "moody", "1")
f = body(r)
assert r["statusCode"] == 200 and f["model"] == "krea2" and f["version"] == 1, f
assert list(f["files"]) == ["krea2.safetensors"] and f["files"]["krea2.safetensors"]["url"].endswith("moody/v1/krea2.safetensors")
assert f["files"]["krea2.safetensors"]["size"] == len(s3.objects["loras/m/moody/v1/krea2.safetensors"])
assert f["lora_json"].endswith("moody/v1/lora.json") and f["checkpoints"] == {}
fw = body(h.lora_files("m", "noir", "2"))  # an uploaded Wan pair
assert sorted(fw["files"]) == ["t2v_high.safetensors", "t2v_low.safetensors"] and fw["model"] == "t2v"
# a trained version: checkpoints listed apart from the final files
s3.put_json("loras/m/trained/v1/lora.json", {"version": 1, "models": ["t2v"], "files": [
    "t2v_high.safetensors", "t2v_low.safetensors", "checkpoints/t2v_low-step000250.safetensors"]})
for k in ("t2v_high.safetensors", "t2v_low.safetensors", "checkpoints/t2v_low-step000250.safetensors"):
    s3.objects[f"loras/m/trained/v1/{k}"] = b"w" * 8
fs = body(h.lora_files("m", "trained", "1"))
assert sorted(fs["files"]) == ["t2v_high.safetensors", "t2v_low.safetensors"], fs
assert list(fs["checkpoints"]) == ["checkpoints/t2v_low-step000250.safetensors"]
# a listed file S3 can't find is an error, never silently left out (the local copy would look complete)
del s3.objects["loras/m/trained/v1/t2v_low.safetensors"]
assert h.lora_files("m", "trained", "1")["statusCode"] == 502
# names in lora.json that could escape the version folder are refused
s3.put_json("loras/m/evil/v1/lora.json", {"version": 1, "models": ["t2v"], "files": ["../../x.safetensors"]})
assert h.lora_files("m", "evil", "1")["statusCode"] == 502
assert h.lora_files("m", "moody", "²")["statusCode"] == 400, "unicode digits are not a version"
assert h.lora_files("m", "moody", "9")["statusCode"] == 404, "unregistered version"
assert h.lora_files("other", "moody", "1")["statusCode"] == 404, "another client's LoRA"
assert h.lora_files("m", "Bad Name", "1")["statusCode"] == 400 and h.lora_files("m", "moody", "x")["statusCode"] == 400
v = body(h.lora_uploads("m", "pending", {"model": "krea2"}))["version"]
s3.objects[f"loras/m/pending/v{v}/krea2.safetensors"] = safetensors(KREA)
assert h.lora_files("m", "pending", str(v))["statusCode"] == 404, "reserved but unregistered"

# ---------------------------------------------------------------- multipart uploads for big files
MB = 1024 ** 2
r = body(h.lora_uploads("m", "bigone", {"files": "single", "model": "krea2", "sizes": {"high": 448 * MB}}))
mp = r["multipart"]["high"]
assert mp["key"] == f"loras/m/bigone/v{r['version']}/krea2.safetensors" and mp["part_size"] == h.MULTIPART_PART
assert len(mp["part_urls"]) == -(-448 * MB // h.MULTIPART_PART) and "part=1&" in mp["part_urls"][0], len(mp["part_urls"])
assert "high" not in r["urls"], "big files: multipart only"
small = body(h.lora_uploads("m", "smallone", {"files": "single", "model": "krea2", "sizes": {"high": 5 * MB}}))
assert "high" in small["urls"] and not small.get("multipart"), "small files: one PUT as before"
assert "high" in body(h.lora_uploads("m", "oldcli", {"files": "single", "model": "krea2"}))["urls"], "old CLIs: no sizes"
parts = [{"part": i + 1, "etag": f"e{i + 1}"} for i in range(len(mp["part_urls"]))]
r2 = h.uploads_complete("m", {"key": mp["key"], "upload_id": mp["upload_id"], "parts": parts})
assert r2["statusCode"] == 200 and s3.multipart[mp["upload_id"]]["state"] == "done"
assert s3.multipart[mp["upload_id"]]["parts"][0] == {"PartNumber": 1, "ETag": "e1"}
for bad in ({"key": "loras/other/x/v1/krea2.safetensors", "upload_id": "up9", "parts": parts},  # another client
            {"key": "avatars/m/ava/avatar.json", "upload_id": "up9", "parts": parts},          # not an upload path
            {"key": mp["key"], "upload_id": mp["upload_id"], "parts": []}):                    # no parts
    assert h.uploads_complete("m", bad)["statusCode"] == 400, bad
d = body(h.dataset_uploads("m", "clips", {"files": [{"path": "big.mp4", "size": 200 * MB}, {"path": "s.mp4", "size": MB}]}))
assert d["uploads"][0]["multipart"]["key"] == "datasets/m/clips/raw/big.mp4" and "put_url" not in d["uploads"][0]
assert "put_url" in d["uploads"][1] and "multipart" not in d["uploads"][1]
dmp = d["uploads"][0]["multipart"]
assert h.uploads_abort("m", {"key": dmp["key"], "upload_id": dmp["upload_id"]})["statusCode"] == 200
assert s3.multipart[dmp["upload_id"]]["state"] == "aborted"

# ---------------------------------------------------------------- every new avatar queues a Krea LoRA build
queued = []
h.jobs = mock.MagicMock(put_item=lambda Item: queued.append(Item))
h.sqs = mock.MagicMock()
h._ensure_started = lambda iid: "running"
r = h.create_avatar("m", {"name": "ava", "upload_keys": uploads("ava1.png"), "description": "a sailor"})
av = h._get_avatar("m", "ava")
assert r["statusCode"] == 200 and body(r)["krea_lora"]["state"] == "queued", body(r)
assert av["krea_lora"]["state"] == "queued" and av["krea_lora"]["trigger"] == "ava" and av["krea_lora"]["lora"] == "avatar-ava"
pj = [q for q in queued if q.get("type") == "avatar_photoset"]
assert len(pj) == 1 and av["krea_lora"]["job_ids"] == [pj[0]["job_id"]]
pp = json.loads(pj[0]["params"])
assert {k: v for k, v in pp.items() if k != "avatar_created_at"} == {"avatar": "ava", "lora": "avatar-ava", "version": 1, "trigger": "ava", "count": 24, "krea_lora": True,
                                                                     "wan_lora": True, "wan_lora_name": "avatar-ava-wan", "wan_version": 1}, pp
assert av["wan_lora"] == {"state": "queued", "lora": "avatar-ava-wan", "version": 1, "trigger": "ava", "job_ids": [pj[0]["job_id"]]}, av.get("wan_lora")
assert body(r)["wan_lora"]["state"] == "queued"
assert pp["avatar_created_at"] == av["created_at"]
assert av["media"] == {"state": "queued", "job_id": pj[0]["job_id"], "created_at": av["created_at"]}, av.get("media")
assert body(r)["media"]["state"] == "queued"
queued.clear()
r = h.create_avatar("m", {"name": "nolora", "upload_keys": uploads("n1.png"), "krea_lora": False})
nj = [q for q in queued if q.get("type") == "avatar_photoset"]
assert body(r)["krea_lora"]["state"] == "skipped" and len(nj) == 1, "no Krea LoRA, but media for every avatar"
npp = json.loads(nj[0]["params"])
assert npp["krea_lora"] is False and npp["avatar"] == "nolora" and npp["trigger"] == "nolora" and "lora" not in npp
assert npp["wan_lora"] is True and npp["wan_lora_name"] == "avatar-nolora-wan", "a Wan LoRA without a Krea one"
assert h._get_avatar("m", "nolora")["wan_lora"]["job_ids"] == [nj[0]["job_id"]]
queued.clear()
h.create_avatar("m", {"name": "nowan", "upload_keys": uploads("nw1.png"), "wan_lora": False})
nwp = json.loads([q for q in queued if q.get("type") == "avatar_photoset"][0]["params"])
assert nwp["wan_lora"] is False and "wan_lora_name" not in nwp and h._get_avatar("m", "nowan")["wan_lora"]["state"] == "skipped"
assert h.create_avatar("m", {"name": "x2", "upload_keys": uploads("x2.png"), "wan_lora": "yes"})["statusCode"] == 400
wl = h._avatar_wan_lora_name("b" * 32)
assert len(wl) <= 32 and wl.endswith("-wan") and h.DATASET_NAME.match(wl) and wl == h._avatar_wan_lora_name("b" * 32)
assert h._avatar_wan_lora_name("b" * 32) != h._avatar_wan_lora_name("b" * 31 + "c")
nav = h._get_avatar("m", "nolora")
assert nav["media"]["state"] == "queued" and nav["media"]["job_id"] == nj[0]["job_id"]
assert "job_ids" not in nav["krea_lora"]
assert h.create_avatar("m", {"name": "x1", "upload_keys": uploads("x1.png"), "krea_lora": "yes"})["statusCode"] == 400
# on demand: a new version
queued.clear()
r = h.avatar_krea_lora("m", "nolora", {})
assert r["statusCode"] == 202 and json.loads([q for q in queued if q.get("type")][0]["params"])["version"] == 1 and \
       h._get_avatar("m", "nolora")["krea_lora"]["state"] == "queued"
assert h.avatar_krea_lora("m", "ghost", {})["statusCode"] == 404
# avatar media: queue for an existing avatar (409 while one runs), list files with presigned URLs
queued.clear()
h._set_media("m", "nolora", {"state": "ready", "job_id": "old", "created_at": 1})
r = h.avatar_media_create("m", "nolora", {})
mj = [q for q in queued if q.get("type") == "avatar_photoset"]
assert r["statusCode"] == 202 and len(mj) == 1, r
mpp = json.loads(mj[0]["params"])
assert mpp["media_only"] is True and mpp["krea_lora"] is False and mpp["avatar"] == "nolora"
assert h._get_avatar("m", "nolora")["media"]["job_id"] == mj[0]["job_id"]
h.jobs.get_item = lambda Key: {"Item": {"job_id": Key["job_id"], "status": "queued"}}
assert h.avatar_media_create("m", "nolora", {})["statusCode"] == 409, "one media job at a time"
assert h.avatar_media_create("m", "ghost", {})["statusCode"] == 404
assert h.avatar_media_get("m", "ghost")["statusCode"] == 404
h._set_media("m", "nolora", {"state": "ready", "job_id": "x", "created_at": 1, "photos": 1, "clips": 1})
s3.put_json("avatars/m/nolora/media/media.json", {"photos": [{"file": "photo_01.png", "size": 10}],
                                                  "clips": [{"file": "clip_1.mp4", "size": 20}]})
g = body(h.avatar_media_get("m", "nolora"))
assert g["state"] == "ready" and [f["name"] for f in g["files"]] == ["photo_01.png", "clip_1.mp4", "media.json"], g
assert g["files"][1] == {"name": "clip_1.mp4", "size": 20, "url": "https://signed/avatars/m/nolora/media/clip_1.mp4"}
h._set_media("m", "nolora", {"state": "clips", "job_id": "y", "created_at": 1})
assert body(h.avatar_media_get("m", "nolora")) == {"state": "clips", "files": []}, "files only when ready"
lst = {a["name"]: a for a in body(h.list_avatars("m"))["avatars"]}
assert lst["nolora"]["media"]["state"] == "clips" and lst["old"]["media"] == {"state": "none"}
assert lst["old"]["wan_lora"] == {"state": "none"} and lst["nolora"]["wan_lora"]["state"] == "queued"
assert json.loads([q for q in queued if q.get("type") == "avatar_photoset"][-1]["params"])["wan_lora"] is False, \
    "media-only jobs never build a Wan LoRA"
# on-demand Wan LoRA build from the avatar's media
queued.clear()
assert h.avatar_wan_lora("m", "ghost", {})["statusCode"] == 404
h._set_media("m", "nolora", {"state": "clips", "job_id": "mm", "created_at": 1})
h._set_wan_lora("m", "nolora", {"state": "failed"})
r = h.avatar_wan_lora("m", "nolora", {})
assert r["statusCode"] == 409 and "media" in body(r)["error"], ("make media first", r)
h._set_media("m", "nolora", {"state": "ready", "job_id": "mm", "created_at": 1})
h._set_wan_lora("m", "nolora", {"state": "failed"})
r = h.avatar_wan_lora("m", "nolora", {})
wj = [q for q in queued if q.get("type") == "avatar_photoset"]
assert r["statusCode"] == 202 and len(wj) == 1, r
wpp = json.loads(wj[0]["params"])
assert wpp["wan_only"] is True and wpp["wan_lora"] is True and wpp["wan_lora_name"] == "avatar-nolora-wan" and wpp["krea_lora"] is False
assert h._get_avatar("m", "nolora")["wan_lora"]["state"] == "queued" and h._get_avatar("m", "nolora")["media"]["state"] == "ready"
h.jobs.get_item = lambda Key: {"Item": {"job_id": Key["job_id"], "status": "queued"}}
assert h.avatar_wan_lora("m", "nolora", {})["statusCode"] == 409, "one build at a time"
h.jobs.get_item = lambda Key: {"Item": {"job_id": Key["job_id"], "status": "failed"}}
assert h.avatar_wan_lora("m", "nolora", {})["statusCode"] == 202, "a build whose jobs ended is not busy"
h.jobs = mock.MagicMock(put_item=lambda Item: queued.append(Item))
# queueing failures leave no media stuck in "queued"
real_qj = h._queue_job
h._queue_job = lambda *a, **k: (_ for _ in ()).throw(ValueError("sqs down"))
h.create_avatar("m", {"name": "qfail", "upload_keys": uploads("q1.png"), "krea_lora": False})
assert h._get_avatar("m", "qfail")["media"]["state"] == "failed"
h.create_avatar("m", {"name": "qfail2", "upload_keys": uploads("q2.png")})
assert h._get_avatar("m", "qfail2")["media"]["state"] == "failed"
assert h._get_avatar("m", "qfail2")["wan_lora"]["state"] == "failed" and h._get_avatar("m", "qfail")["wan_lora"]["state"] == "failed"
h._set_media("m", "qfail", {"state": "failed"})
r = h.avatar_media_create("m", "qfail", {})
assert r["statusCode"] == 502 and h._get_avatar("m", "qfail")["media"]["state"] == "failed", "POST /media: no stuck queued"
h._queue_job = real_qj
# busy media whose job has ended (or vanished) does not block a new request
h.jobs = mock.MagicMock(put_item=lambda Item: queued.append(Item),
                        get_item=lambda Key: {"Item": {"job_id": Key["job_id"], "status": "failed"}})
h._set_media("m", "qfail", {"state": "clips", "job_id": "dead", "created_at": 1})
assert h.avatar_media_create("m", "qfail", {})["statusCode"] == 202, "the owning job ended: not busy"
h.jobs = mock.MagicMock(put_item=lambda Item: queued.append(Item),
                        get_item=lambda Key: {"Item": {"job_id": Key["job_id"], "status": "running"}})
assert h.avatar_media_create("m", "qfail", {})["statusCode"] == 409
h.jobs = mock.MagicMock(put_item=lambda Item: queued.append(Item))
# long names: the LoRA name stays within the 32-character limit, and is stable
long = "a" * 30 + "zz"
h.create_avatar("m", {"name": long, "upload_keys": uploads("l1.png")})
ln = h._get_avatar("m", long)["krea_lora"]["lora"]
assert len(ln) <= 32 and h.DATASET_NAME.match(ln) and ln == h._avatar_lora_name(long), ln
assert h._avatar_lora_name(long) != h._avatar_lora_name("a" * 30 + "zy")
# the build's job id is on the avatar before the job is sent (a warm worker reads it at once)
seen_at_send = []
h.sqs = mock.MagicMock(send_message=lambda QueueUrl, MessageBody: seen_at_send.append(
    (json.loads(MessageBody)["job_id"], (h._get_avatar("m", "early") or {}).get("krea_lora", {}).get("job_ids"))))
h.create_avatar("m", {"name": "early", "upload_keys": uploads("e1.png")})
jid, ids = [x for x in seen_at_send if x[1] is not None][-1]
assert ids == [jid], seen_at_send
h.sqs = mock.MagicMock()
# image edits and restore keep the Krea LoRA link
h._set_krea_lora("m", "ava", {"state": "ready", "lora": "avatar-ava", "version": 1, "trigger": "ava",
                              "ready": {"lora": "avatar-ava", "version": 1}, "job_ids": []})
tick()
h.add_avatar_images("m", "ava", {"upload_keys": uploads("ava2.png")})
assert h._get_avatar("m", "ava")["krea_lora"]["state"] == "ready", "add keeps the link"
assert h._get_avatar("m", "ava")["media"]["state"] == "queued", "image edits keep the media record"
assert h._get_avatar("m", "ava")["wan_lora"]["lora"] == "avatar-ava-wan", "and the Wan LoRA record"
h.create_avatar("m", {"name": "rep", "upload_keys": uploads("rep1.png"), "krea_lora": False})
h._set_wan_lora("m", "rep", dict(h._get_avatar("m", "rep")["wan_lora"], state="ready",
                                 ready={"lora": "avatar-rep-wan", "version": 1}))
h.create_avatar("m", {"name": "rep", "upload_keys": uploads("rep2.png"), "krea_lora": False})
rw = h._get_avatar("m", "rep")["wan_lora"]
assert "ready" not in rw and rw["state"] == "queued", "a replaced avatar drops the previous person's Wan LoRA"
assert rw["version"] == 2, "versions keep rising: an in-flight build of the previous person can't collide"
tick()
h.remove_avatar_image("m", "ava", "2")
assert h._get_avatar("m", "ava")["krea_lora"]["ready"]["version"] == 1, "remove keeps the link"
# a rebuild while one is in flight is refused; a new build keeps the last ready LoRA linked meanwhile
queued.clear()
r = h.avatar_krea_lora("m", "ava", {})
kl = h._get_avatar("m", "ava")["krea_lora"]
assert r["statusCode"] == 202 and kl["state"] == "queued" and kl["version"] == 2 and kl["ready"]["version"] == 1, kl
assert h.avatar_krea_lora("m", "ava", {})["statusCode"] == 409, "one build at a time"
assert json.loads([q for q in queued if q.get("type")][0]["params"])["avatar_created_at"] == h._get_avatar("m", "ava")["created_at"]
# --no-krea-lora on an existing avatar keeps its ready LoRA
h._set_krea_lora("m", "ava", dict(kl, state="ready", version=2, ready={"lora": "avatar-ava", "version": 2}))
tick()
h.create_avatar("m", {"name": "ava", "upload_keys": uploads("ava3.png"), "krea_lora": False})
kl = h._get_avatar("m", "ava")["krea_lora"]
assert kl["state"] == "skipped" and kl["ready"] == {"lora": "avatar-ava", "version": 2}, kl
# names starting with avatar- are reserved for avatar LoRAs
r = h.lora_uploads("m", "avatar-mine", {"model": "krea2"})
assert r["statusCode"] == 400 and "reserved" in body(r)["error"], body(r)
# a name that already holds a Wan LoRA: the Krea build is marked failed; the avatar still gets media only
s3.put_json("loras/m/avatar-wanclash/v1/lora.json", {"version": 1, "models": ["t2v"], "files": []})
queued.clear()
h.create_avatar("m", {"name": "wanclash", "upload_keys": uploads("w1.png")})
kl = h._get_avatar("m", "wanclash")["krea_lora"]
qd = [q for q in queued if q.get("type")]
assert kl["state"] == "failed" and "Wan" in kl["error"], kl
assert len(qd) == 1 and json.loads(qd[0]["params"])["krea_lora"] is False, "no Krea LoRA, but media"
# queueing fails: the avatar is still saved (and made default if asked), the build marked failed
h.jobs = mock.MagicMock(put_item=mock.MagicMock(side_effect=h.ClientError("throttled")))
r = h.create_avatar("m", {"name": "qfail", "upload_keys": uploads("q1.png"), "make_default": True})
assert r["statusCode"] == 200 and body(r)["default"] is True and body(r)["krea_lora"]["state"] == "failed", body(r)
h.jobs = mock.MagicMock(put_item=lambda Item: queued.append(Item))
h._put_json("avatars/m/_default.json", {"name": "dana"})
# the list shows the state; the job summary names the avatar
listed = {a["name"]: a for a in body(h.list_avatars("m"))["avatars"]}
assert listed["ava"]["krea_lora"]["state"] == "skipped" and listed["nolora"]["krea_lora"]["state"] == "queued"
assert "ava" in h._summary("avatar_photoset", pp)[0]

# ---------------------------------------------------------------- Krea photos / first frames with an avatar's LoRA
s3.put_json("loras/m/avatar-ready1/v1/lora.json", {"version": 1, "models": ["krea2"], "files": ["krea2.safetensors"],
                                                  "single_file": True, "trigger": "ready1", "source": "avatar"})
h.create_avatar("m", {"name": "ready1", "upload_keys": uploads("r1.png"), "krea_lora": False})
h._set_krea_lora("m", "ready1", {"state": "ready", "lora": "avatar-ready1", "version": 1, "trigger": "ready1",
                                 "ready": {"lora": "avatar-ready1", "version": 1}, "job_ids": []})
_, ap = h._validate("m", {"type": "photos", "model": "krea2", "avatar": "ready1", "scenes": [{"prompt": "on a ferry"}], "count": 1})
assert ap["avatar_lora"] == {"name": "avatar-ready1", "version": 1} and ap["avatar_trigger"] == "ready1"
h._job_loras("m", "photos", ap, None)
assert [l["name"] for l in ap["krea_loras"]] == ["avatar-ready1"] and ap["krea_loras"][0]["key"].endswith("avatar-ready1/v1/krea2.safetensors")
r = h.create_job("m", {"type": "photos", "model": "krea2", "avatar": "ready1", "scenes": [{"prompt": "x"}], "count": 1,
                       "raw": True, "loras": "moody:0.5"})
assert r["statusCode"] == 202 and {l["name"] for l in body(r)["loras"]} == {"avatar-ready1", "moody"}, body(r)
# the avatar's LoRA isn't added twice when also picked by hand; with it, one more Krea LoRA at most
s3.put_json("loras/m/avatar-ready1/current.json", {"version": 1})
s3.put_json("loras/m/extra/v1/lora.json", {"version": 1, "models": ["krea2"], "files": ["krea2.safetensors"], "trigger": ""})
s3.put_json("loras/m/extra/current.json", {"version": 1})
dp = {"model": "krea2", "avatar_lora": {"name": "avatar-ready1", "version": 1}}
h._job_loras("m", "photos", dp, "avatar-ready1,moody")
assert [l["name"] for l in dp["krea_loras"]] == ["avatar-ready1", "moody"], dp["krea_loras"]
_stubs.expect_error(h._job_loras, "m", "photos", {"model": "krea2", "avatar_lora": {"name": "avatar-ready1", "version": 1}},
                    "moody,extra", contains="one more Krea 2 LoRA")
vp3 = {"image_model": "krea2", "avatars": [{"name": "ready1"}], "krea_loras": h._resolve_loras("m", "avatar-ready1")}
h._avatar_first_frame("m", "video", vp3)
assert [l["name"] for l in vp3["krea_loras"]] == ["avatar-ready1"] and vp3["krea_avatar_lora"] == "avatar-ready1"
r = h.create_job("m", {"type": "photos", "model": "krea2", "avatar": "nolora", "scenes": [{"prompt": "x"}], "count": 1})
assert r["statusCode"] == 400 and "no Krea 2 LoRA" in body(r)["error"], body(r)
# video: one avatar with a ready LoRA -> a Krea first frame with it; otherwise the old rules
r = h.create_job("m", {"type": "video", "prompt": "walking", "image_model": "krea2", "avatar": "ready1"})
assert r["statusCode"] == 202, body(r)
assert h.create_job("m", {"type": "video", "prompt": "x", "image_model": "krea2", "avatar": "dana"})["statusCode"] == 400
vp2 = {"image_model": "krea2", "avatars": [{"name": "ready1"}]}
h._avatar_first_frame("m", "video", vp2)
assert vp2["krea_avatar"] == "ready1" and [l["name"] for l in vp2["krea_loras"]] == ["avatar-ready1"]
sp2 = {"image_model": "krea2", "avatars": [{"name": "ready1"}, {"name": "dana"}], "krea_loras": []}
h._avatar_first_frame("m", "story", sp2)
assert sp2["krea_avatar"] == "ready1", "story: shots with just this avatar get Krea frames"
sp3 = {"image_model": "krea2", "avatars": [{"name": "dana"}], "krea_loras": []}
h._avatar_first_frame("m", "story", sp3)
assert "krea_avatar" not in sp3

# ---------------------------------------------------------------- Krea 2 LoRAs on jobs: routed by kind
res = h._resolve_loras("m", "moody:0.8")[0]
assert res == {"name": "moody", "version": 1, "step": None, "weight": 0.8, "trigger": "mdy", "model": "krea2",
               "key": "loras/m/moody/v1/krea2.safetensors"}, res
assert h._resolve_loras("m", [{"name": "noir"}])[0]["model"] == "t2v"
_stubs.expect_error(h._resolve_loras, "m", [{"name": "moody", "step": 500}], contains="no checkpoint steps")
for nm in ("k2", "k3"):  # three Krea LoRAs: over the per-kind limit
    vv = body(h.lora_uploads("m", nm, {"model": "krea2"}))["version"]
    s3.objects[f"loras/m/{nm}/v{vv}/krea2.safetensors"] = safetensors(KREA)
    h.register_lora_upload("m", nm, str(vv), {})
_stubs.expect_error(h._resolve_loras, "m", "moody,k2,k3", contains="at most 2 Krea 2 LoRAs")
assert len(h._resolve_loras("m", "moody,k2,noir")) == 3, "2 Krea + 1 Wan is fine"
_stubs.expect_error(h._resolve_loras, "m", "noir,jerk,style", contains="at most 2 Wan LoRAs")
_stubs.expect_error(h._resolve_loras, "m", "moody:2.0", contains="weight")
# photos: Krea model takes Krea LoRAs only; qwen takes none
pp = {"model": "krea2"}
h._job_loras("m", "photos", pp, "moody:0.8")
assert pp["krea_loras"][0]["key"].endswith("moody/v1/krea2.safetensors") and "loras" not in pp
_stubs.expect_error(h._job_loras, "m", "photos", {"model": "krea2"}, "noir", contains="Wan LoRA 'noir' is for video")
_stubs.expect_error(h._job_loras, "m", "photos", {"model": "qwen"}, "moody", contains="--model krea2")
# video/story: Wan to Wan, Krea to the Krea first frame (needs image_model krea2)
vp = {"image_model": "krea2"}
h._job_loras("m", "video", vp, "moody,noir:0.5")
assert [l["name"] for l in vp["loras"]] == ["noir"] and [l["name"] for l in vp["krea_loras"]] == ["moody"]
_stubs.expect_error(h._job_loras, "m", "video", {"image_model": "wan"}, "moody", contains="needs --image-model krea2")
sp = {"image_model": "wan"}
h._job_loras("m", "story", sp, "noir")
assert [l["name"] for l in sp["loras"]] == ["noir"] and sp["krea_loras"] == []
# jobs without a Krea step refuse Krea LoRAs
for jt in ("avatar", "describe", "write"):
    _stubs.expect_error(h._job_loras, "m", jt, {}, "moody", contains="Krea 2 LoRA 'moody'")
    h._job_loras("m", jt, {}, "nonexistent")  # as before: other jobs ignore a loras field otherwise
    h._job_loras("m", jt, {}, 5)
_stubs.expect_error(h._job_loras, "m", "story", {"image_model": "wan"}, "moody", contains="needs --image-model krea2")
r = h.create_job("m", {"type": "photos", "model": "qwen", "avatar": "dana", "scenes": [{"prompt": "a street"}],
                       "count": 1, "loras": "moody"})
assert r["statusCode"] == 400 and "--model krea2" in body(r)["error"], body(r)
r = h.create_job("m", {"type": "avatar", "description": "a sailor", "loras": "moody"})
assert r["statusCode"] == 400 and "Krea 2 LoRA" in body(r)["error"], body(r)
assert "moody:1 (krea2)" in h._summary("video", vp)[1]["loras"] and "noir:0.5" in h._summary("video", vp)[1]["loras"]
r = h.create_job("m", {"type": "photos", "model": "krea2", "avatar": "none", "scenes": [{"prompt": "a street"}],
                       "count": 1, "loras": "moody:0.8"})
assert r["statusCode"] == 202 and body(r)["loras"] == [{"name": "moody", "version": 1, "step": None, "weight": 0.8,
                                                         "trigger": "mdy", "model": "krea2"}], body(r)
assert h.create_job("m", {"type": "video", "prompt": "x", "avatar": "none", "loras": "moody"})["statusCode"] == 400
assert "moody:0.8 (krea2)" in h._summary("photos", pp)[1]["loras"]
_stubs.expect_error(h._validate, "m", {"type": "lora_preview", "name": "moody"}, contains="Krea 2 LoRA")
# raw: Krea photos get the prompt exactly as typed (no look / realism / trigger additions)
_, rp = h._validate("m", {"type": "photos", "model": "krea2", "avatar": "none", "scenes": [{"prompt": "a red fox"}],
                          "count": 1, "raw": True})
assert rp["raw"] is True
_, rp = h._validate("m", {"type": "photos", "model": "krea2", "avatar": "none", "scenes": [{"prompt": "a red fox"}], "count": 1})
assert rp["raw"] is False
_stubs.expect_error(h._validate, "m", {"type": "photos", "avatar": "dana", "scenes": [{"prompt": "x"}], "count": 1,
                                       "raw": True}, contains="raw")
_stubs.expect_error(h._validate, "m", {"type": "photos", "model": "krea2", "avatar": "none", "scenes": [{"prompt": "x"}],
                                       "count": 1, "raw": "yes"}, contains="raw")
_, rp4 = h._validate("m", {"type": "photos", "model": "krea2", "avatar": "none", "count": 4, "raw": True,
                           "scenes": [{"prompt": "x"}]})
assert rp4["count"] == 4, "raw: extra photos repeat the prompts with new seeds"
_stubs.expect_error(h._validate, "m", {"type": "photos", "model": "krea2", "avatar": "none", "count": 1, "raw": True,
                                       "scenes": []}, contains="raw")  # raw needs at least one prompt
assert h._summary("photos", dict(rp, raw=True))[1]["raw"] is True
# Krea prompts may be long (up to 2000 chars); other scene text keeps the 500 limit
long_p = "a detailed scene " * 60  # ~1000 chars
_, lp = h._validate("m", {"type": "photos", "model": "krea2", "avatar": "none", "count": 1, "raw": True,
                          "scenes": [{"prompt": long_p}]})
assert lp["scenes"][0]["prompt"] == long_p.strip()
_stubs.expect_error(h._validate, "m", {"type": "photos", "model": "krea2", "avatar": "none", "count": 1,
                                       "scenes": [{"prompt": "x" * 2001}]}, contains="2000")
_stubs.expect_error(h._validate, "m", {"type": "photos", "avatar": "dana", "count": 1,
                                       "scenes": [{"prompt": long_p}]}, contains="500")
_stubs.expect_error(h._validate, "m", {"type": "photos", "model": "krea2", "avatar": "none", "count": 1,
                                       "scenes": [{"location": long_p}]}, contains="500")
# raw takes typed text only: person-form fields ("where" -> "in bedroom") would not be the exact prompt
_stubs.expect_error(h._validate, "m", {"type": "photos", "model": "krea2", "avatar": "none", "count": 1, "raw": True,
                                       "scenes": [{"shot": {"where": "bedroom"}}]}, contains="prompt")

# ---------------------------------------------------------------- training: speed, captions-only, captions
s3.put_json("datasets/m/moves/manifest.json", {"kind": "motion", "trigger": "vgn_moves", "file_count": 2,
                                              "updated_at": 1, "files": [{"path": "a.mp4", "size": 1}]})
h.TRAIN_QUEUE_URL = "tq"
_, t = h._validate("m", {"type": "train", "dataset": "moves", "speed": "fast", "captions_only": True})
assert t["speed"] == "fast" and t["captions_only"] is True and t["fingerprint"]
_, t = h._validate("m", {"type": "train", "dataset": "moves"})
assert t["speed"] == "standard" and t["captions_only"] is False
_stubs.expect_error(h._validate, "m", {"type": "train", "dataset": "moves", "speed": "turbo"}, contains="speed")
assert h.get_captions("m", "moves")["statusCode"] == 404
s3.put_json("datasets/m/moves/captions.json", {"fingerprint": t["fingerprint"], "trigger": "vgn_moves",
                                              "items": {"s0001_00.mp4": "vgn_moves a person in a park"}})
g = body(h.get_captions("m", "moves"))
assert g["current"] is True and g["items"]["s0001_00.mp4"].startswith("vgn_moves")
r = h.put_captions("m", "moves", {"items": {"s0001_00.mp4": "a person dancing on a beach"}})
assert r["statusCode"] == 200 and body(r)["without_trigger"] == ["s0001_00.mp4"]
assert s3.json("datasets/m/moves/captions.json")["items"]["s0001_00.mp4"] == "a person dancing on a beach"
assert s3.json("datasets/m/moves/captions.json")["edited_at"] == clock[0]
for bad in [{"items": {"nope.mp4": "x"}}, {"items": {"s0001_00.mp4": ""}}, {"items": {}}, {"items": "x"}]:
    assert h.put_captions("m", "moves", bad)["statusCode"] == 400, bad

# ---------------------------------------------------------------- routes are wired
h._authenticate = lambda *a: "m"
for route, params in [("GET /avatars/{name}", {"name": "dana"}), ("PUT /loras/{name}/current", {"name": "jerk"}),
                      ("GET /datasets/{name}/captions", {"name": "moves"}),
                      ("PUT /datasets/{name}/captions", {"name": "moves"}),
                      ("POST /loras/{name}/uploads", {"name": "routes"}),
                      ("PUT /loras/{name}/versions/{version}", {"name": "routes", "version": "1"})]:
    r = h.handler({"routeKey": route, "pathParameters": params, "headers": {}, "body": "{}"}, None)
    assert r["statusCode"] in (200, 400, 404), (route, r)
    assert "unknown route" not in r["body"].lower(), (route, r["body"])
print("api tests passed")
