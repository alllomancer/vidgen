"""CLI and dashboard side of uploading an avatar's LoRAs: the right endpoints in order, the file kinds checked here
before anything is uploaded, paths checked by the dashboard. docs/specs/2026-11-27-avatar-lora-upload.md"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("cli")
import vidgen  # noqa: E402

vidgen.STATE_DIR = tempfile.mkdtemp()
import ui  # noqa: E402

tmp = tempfile.mkdtemp()


def lora_file(name, tensors):
    head = json.dumps({k: {"dtype": d, "shape": s, "data_offsets": [0, 0]} for k, (d, s) in tensors.items()}).encode()
    path = os.path.join(tmp, name)
    open(path, "wb").write(len(head).to_bytes(8, "little") + head + b"\0" * 16)
    return path


WAN = {"diffusion_model.blocks.0.self_attn.q.lora_down.weight": ("BF16", [32, 5120])}
KREA = {"lora_unet_blocks_0_attn_wq.lora_down.weight": ("BF16", [32, 6144])}
krea, high, low = lora_file("k.safetensors", KREA), lora_file("h.safetensors", WAN), lora_file("l.safetensors", WAN)

calls, puts = [], []


def api(method, path, body=None):
    calls.append((method, path, body))
    if path.endswith("/uploads"):
        return {"version": 3, "urls": {"high": "https://s3/h", **({"low": "https://s3/l"} if body["files"] == "pair" else {})}}
    kind = "krea2" if "/loras/krea/" in path else "t2v"
    return {"name": "avatar-dana" + ("" if kind == "krea2" else "-wan"), "version": 3, "model": kind,
            "trigger": "dana", "avatar": "dana"}


vidgen.api = ui.vidgen.api = api
vidgen.put_file = lambda url, path, label: puts.append((url, os.path.basename(path)))

out = vidgen.avatar_lora_upload("Dana", krea, high, low)
assert [c[:2] for c in calls] == [("POST", "/avatars/dana/loras/krea/uploads"), ("PUT", "/avatars/dana/loras/krea/versions/3"),
                                  ("POST", "/avatars/dana/loras/wan/uploads"), ("PUT", "/avatars/dana/loras/wan/versions/3")], calls
assert calls[0][2]["model"] == "krea2" and calls[0][2]["files"] == "single" and calls[2][2]["files"] == "pair"
assert puts == [("https://s3/h", "k.safetensors"), ("https://s3/h", "h.safetensors"), ("https://s3/l", "l.safetensors")]
assert [r["model"] for r in out] == ["krea2", "t2v"]

# wrong kind in a slot: refused here, nothing reserved or uploaded
for args, msg in [((high,), "not a Krea 2 LoRA"), ((None, krea), "not a Wan LoRA"), ((None, high, krea), "not a Wan LoRA")]:
    calls.clear()
    puts.clear()
    try:
        vidgen.avatar_lora_upload("dana", *args)
        raise AssertionError(args)
    except vidgen.ApiError as e:
        assert msg in str(e) and not calls and not puts, (args, e, calls)
# a good Krea file with a bad Wan one: nothing at all is uploaded (no half-done result)
calls.clear()
puts.clear()
try:
    vidgen.avatar_lora_upload("dana", krea, krea)
    raise AssertionError("wrong Wan kind")
except vidgen.ApiError as e:
    assert "not a Wan LoRA" in str(e) and not calls and not puts, (e, calls)
for args in [(), (None, None, low)]:
    try:
        vidgen.avatar_lora_upload("dana", *args)
        raise AssertionError(args)
    except vidgen.ApiError:
        pass

# the dashboard route: full paths only; a local check comes back as the user's mistake (400), not a server error
calls.clear()
assert len(ui.add_avatar_loras("dana", {"krea": krea})["uploaded"]) == 1
for body in ({"krea": "k.safetensors"}, {"wan": "~nope/x"}):
    try:
        ui.add_avatar_loras("dana", body)
        raise AssertionError(body)
    except ValueError:
        pass
try:
    ui.add_avatar_loras("dana", {"krea": high})
    raise AssertionError("wrong kind")
except ValueError as e:
    assert "not a Krea 2 LoRA" in str(e)
try:
    ui.add_avatar_loras("Bad Name", {"krea": krea})
    raise AssertionError("bad name")
except ValueError:
    pass

print("test_avatar_lora_cli: ok")
