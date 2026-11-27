"""Uploading an avatar's own Krea 2 / Wan LoRA: the next version of avatar-<name> / avatar-<name>-wan, checked like
any upload, then marked ready on the avatar. docs/specs/2026-11-27-avatar-lora-upload.md"""
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
h.ClientError = KeyError
h.time = mock.MagicMock(time=lambda: 1000)
body = lambda r: json.loads(r["body"])  # noqa: E731


def safetensors(tensors, pad=64):
    head = json.dumps({k: {"dtype": d, "shape": shp, "data_offsets": [0, 0]} for k, (d, shp) in tensors.items()}).encode()
    return len(head).to_bytes(8, "little") + head + b"\0" * pad


WAN = {"diffusion_model.blocks.0.self_attn.q.lora_down.weight": ("BF16", [32, 5120]),
       "diffusion_model.blocks.0.self_attn.q.lora_up.weight": ("BF16", [5120, 32])}
KREA = {"lora_unet_blocks_0_attn_wq.lora_down.weight": ("BF16", [32, 6144]),
        "lora_unet_blocks_0_attn_wq.lora_up.weight": ("BF16", [6144, 32])}

s3.put_json("avatars/m/dana/avatar.json", {"name": "dana", "description": "", "created_at": 1,
                                           "reference_key": "avatars/m/dana/r.png",
                                           "krea_lora": {"state": "failed", "error": "flagged"}})
live = set()
h._job_live = lambda job_id: job_id in live

# ---- Krea 2: one file as avatar-dana v1, ready on the avatar
r = h.avatar_lora_uploads("m", "Dana", "krea", {"files": "single", "sizes": {"high": 1000}})
assert r["statusCode"] == 200, body(r)
up = body(r)
assert up["model"] == "krea2" and up["urls"]["high"].endswith("loras/m/avatar-dana/v1/krea2.safetensors"), up
s3.objects["loras/m/avatar-dana/v1/krea2.safetensors"] = safetensors(KREA)
r = h.register_avatar_lora("m", "dana", "krea", "1")
assert r["statusCode"] == 200 and body(r)["trigger"] == "dana", body(r)
kl = h._get_avatar("m", "dana")["krea_lora"]
assert kl["state"] == "ready" and kl["ready"] == {"lora": "avatar-dana", "version": 1} and kl["source"] == "upload", kl
assert kl["error"] is None and s3.json("loras/m/avatar-dana/current.json") == {"version": 1}
assert s3.json("loras/m/avatar-dana/v1/lora.json")["trigger"] == "dana"

# ---- Wan: a pair as avatar-dana-wan v1
up = body(h.avatar_lora_uploads("m", "dana", "wan", {"files": "pair"}))
assert set(up["urls"]) == {"high", "low"} and up["urls"]["low"].endswith("loras/m/avatar-dana-wan/v1/t2v_low.safetensors")
for part in ("high", "low"):
    s3.objects[f"loras/m/avatar-dana-wan/v1/t2v_{part}.safetensors"] = safetensors(WAN)
r = h.register_avatar_lora("m", "dana", "wan", "1")
assert r["statusCode"] == 200 and body(r)["single_file"] is False, body(r)
wl = h._get_avatar("m", "dana")["wan_lora"]
assert wl["ready"] == {"lora": "avatar-dana-wan", "version": 1} and wl["version"] == 1 and wl["trigger"] == "dana"
# the next video starring dana picks it up
params = {"avatars": [h._get_avatar("m", "dana")], "loras": []}
h._with_avatar_wan_lora("m", params)
assert params["loras"][0]["name"] == "avatar-dana-wan" and params["loras"][0]["version"] == 1, params

# ---- the wrong kind of file in a slot: 400, avatar unchanged
before = h._get_avatar("m", "dana")
up = body(h.avatar_lora_uploads("m", "dana", "krea", {}))
s3.objects[f"loras/m/avatar-dana/v{up['version']}/krea2.safetensors"] = safetensors(WAN)
r = h.register_avatar_lora("m", "dana", "krea", str(up["version"]))
assert r["statusCode"] == 400 and h._get_avatar("m", "dana") == before, body(r)
up = body(h.avatar_lora_uploads("m", "dana", "wan", {}))
s3.objects[f"loras/m/avatar-dana-wan/v{up['version']}/t2v_high.safetensors"] = safetensors(KREA)
r = h.register_avatar_lora("m", "dana", "wan", str(up["version"]))
assert r["statusCode"] == 400 and h._get_avatar("m", "dana") == before, body(r)

# ---- refused: unknown avatar, bad kind, a Krea pair; while that kind's build is live (the other kind still works)
assert h.avatar_lora_uploads("m", "nobody", "krea", {})["statusCode"] == 404
assert h.avatar_lora_uploads("m", "dana", "flux", {})["statusCode"] == 400
assert h.avatar_lora_uploads("m", "dana", "krea", {"files": "pair"})["statusCode"] == 400
av = h._get_avatar("m", "dana")
av["krea_lora"] = dict(av["krea_lora"], state="training", job_ids=["t1"])
s3.put_json("avatars/m/dana/avatar.json", av)
live.add("t1")
r = h.avatar_lora_uploads("m", "dana", "krea", {})
assert r["statusCode"] == 409 and "wait for it" in body(r)["error"], body(r)
assert h.register_avatar_lora("m", "dana", "krea", "9")["statusCode"] == 409
assert h.avatar_lora_uploads("m", "dana", "wan", {})["statusCode"] == 200
live.clear()
assert h.avatar_lora_uploads("m", "dana", "krea", {})["statusCode"] == 200, "a finished (or dead) build doesn't block"

# ---- the generic upload still refuses avatar-* names
assert h.lora_uploads("m", "avatar-dana", {"files": "single"})["statusCode"] == 400

print("test_avatar_lora_upload: ok")
