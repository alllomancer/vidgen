"""Phone pairing from the Mac dashboard: POST /pair (unsigned) -> a code; a computer (Terraform) key approves it; the
phone's key then signs as the approver's client; revoking stops it. docs/specs/2026-11-27-phone-pairing.md"""
import base64
import json
import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install(os.path.join("lambda", "api"))
import handler as h  # noqa: E402

s3 = h.s3 = _stubs.FakeS3()
h._s3_json = s3.json
clock = [1_800_000_000]
h.time = mock.MagicMock(time=lambda: clock[0])
h.nonces = types.SimpleNamespace(put_item=lambda **kw: None)
h.ClientError = KeyError
h.InvalidSignature = type("InvalidSignature", (Exception,), {})


def ssh_string(b):
    return len(b).to_bytes(4, "big") + b


def pub_blob(n):
    return ssh_string(b"ssh-ed25519") + ssh_string(bytes([n]) * 32)


MAC = pub_blob(1)
h._authorized = {MAC: "dan"}  # the Terraform list: the Mac key, comment "dan"
PHONE, OTHER = pub_blob(2), pub_blob(3)


def line(blob, comment="whatever"):
    return f"ssh-ed25519 {base64.b64encode(blob).decode()} {comment}"


def sshsig(blob):
    """An SSHSIG naming blob as the signer (the ed25519 check itself is stubbed in unit tests)."""
    sig = ssh_string(ssh_string(b"ssh-ed25519") + ssh_string(b"\0" * 64))
    raw = b"SSHSIG" + (1).to_bytes(4, "big") + ssh_string(blob) + ssh_string(b"vidgen-api") + ssh_string(b"") + \
        ssh_string(b"sha512") + sig
    return base64.b64encode(raw).decode()


nonce = [0]


def request(route, path, signer=None, body=None, params=None):
    raw = json.dumps(body) if body is not None else ""
    headers = {}
    if signer is not None:
        nonce[0] += 1
        armored = f"-----BEGIN SSH SIGNATURE-----\n{sshsig(signer)}\n-----END SSH SIGNATURE-----\n"
        headers = {"x-vidgen-timestamp": str(clock[0]), "x-vidgen-nonce": f"{nonce[0]:032x}",
                   "x-vidgen-signature": base64.b64encode(armored.encode()).decode()}
    event = {"routeKey": route, "rawPath": path, "headers": headers, "body": raw,
             "requestContext": {"http": {"method": route.split()[0], "userAgent": "Mozilla/5.0 (Linux; Android 15) Chrome/140"}},
             "pathParameters": params or {}}
    r = h.handler(event, None)
    return r["statusCode"], json.loads(r["body"]) if r.get("body") else {}


h.list_jobs = lambda client: h._resp(200, {"client": client})  # GET /jobs answers with who the caller is

# ---- not paired yet: the phone's signed calls are refused
assert request("GET /jobs", "/jobs", PHONE)[0] == 401
assert request("GET /jobs", "/jobs", MAC) == (200, {"client": "dan"})

# ---- POST /pair validation
for bad in [None, {}, {"public_key": 5}, {"public_key": "ssh-rsa AAAA"}, {"public_key": "ssh-ed25519 !!!"},
            {"public_key": line(ssh_string(b"ssh-ed25519") + ssh_string(b"x" * 31))},
            {"public_key": line(ssh_string(b"ssh-dss") + ssh_string(b"x" * 32))}]:
    st, b = request("POST /pair", "/pair", body=bad)
    assert st == 400, (bad, st, b)
st, b = request("POST /pair", "/pair", body={"public_key": "x" * 2000})
assert st == 413, (st, b)
# ---- closed until the dashboard opens pairing (a computer key)
st, b = request("POST /pair", "/pair", body={"public_key": line(PHONE)})
assert st == 403 and "Pair a phone" in b["error"], (st, b)
assert request("POST /pairings/open", "/pairings/open", PHONE)[0] == 401, "not paired yet"
st, b = request("POST /pairings/open", "/pairings/open", MAC)
assert st == 200 and b["open_until"] == clock[0] + 600, b

# an authorized key gets the same answer shape as a new one, and nothing is stored (no "is this key paired?" oracle)
st, b = request("POST /pair", "/pair", body={"public_key": line(MAC)})
assert st == 200 and len(b["code"]) == 6 and h._pairing_requests() == {}, (st, b)

st, b = request("POST /pair", "/pair", body={"public_key": line(PHONE)})
assert st == 200 and len(b["code"]) == 6 and b["code"].isdigit() and b["expires_at"] == clock[0] + 600, b
assert b["fingerprint"] == h.fingerprint(PHONE) and b["fingerprint"].startswith("SHA256:"), b
code = b["code"]
st, again = request("POST /pair", "/pair", body={"public_key": line(PHONE)})  # asking again replaces it
assert st == 200
pending = h._pairing_requests()
assert len(pending) == 1 and again["code"] in pending, pending
code = again["code"]
assert "Android" in pending[code]["agent"] and pending[code]["fingerprint"] == h.fingerprint(PHONE)

# ---- the pairing routes are for computer keys only
assert request("GET /pairings", "/pairings")[0] == 401
st, b = request("GET /pairings", "/pairings", MAC)
assert st == 200 and [p["code"] for p in b["pending"]] == [code] and b["devices"] == [], b
assert b["pending"][0]["fingerprint"] == h.fingerprint(PHONE) and b["open_until"] == clock[0] + 600
assert request("POST /pairings/{code}/approve", f"/pairings/{code}/approve", MAC, {}, {"code": "000000"})[0] == 404

# ---- approve: the phone now signs as "dan"
st, b = request("POST /pairings/{code}/approve", f"/pairings/{code}/approve", MAC, {"name": "Pixel 9"}, {"code": code})
assert st == 200 and b["device"]["name"] == "Pixel 9" and b["device"]["client"] == "dan", b
assert h._pairing_requests() == {}
assert request("GET /jobs", "/jobs", PHONE) == (200, {"client": "dan"})
assert request("GET /pairings", "/pairings", MAC)[1]["open_until"] is None, "approving closes the window"
assert request("POST /pair", "/pair", body={"public_key": line(OTHER)})[0] == 403
dev_id = b["device"]["id"]
# a phone can't manage phones
for route, path, params, body in [("GET /pairings", "/pairings", {}, None),
                                  ("POST /pairings/open", "/pairings/open", {}, None),
                                  ("POST /pairings/{code}/approve", "/pairings/123456/approve", {"code": "123456"}, {}),
                                  ("DELETE /pairings/{code}", "/pairings/123456", {"code": "123456"}, None),
                                  ("DELETE /devices/{id}", f"/devices/{dev_id}", {"id": dev_id}, None)]:
    st, b = request(route, path, PHONE, body, params)
    assert st == 403 and "computer key" in b["error"], (route, st, b)
assert [d["id"] for d in request("GET /pairings", "/pairings", MAC)[1]["devices"]] == [dev_id]
assert request("GET /pairings", "/pairings", MAC)[1]["devices"][0]["name"] == "Pixel 9"

# ---- bad names
request("POST /pairings/open", "/pairings/open", MAC)
assert request("POST /pair", "/pair", body={"public_key": line(PHONE)})[1]["code"] not in h._pairing_requests()
code2 = request("POST /pair", "/pair", body={"public_key": line(OTHER)})[1]["code"]
for bad in ["x" * 33, "a\nb", "<b>", 5]:
    assert request("POST /pairings/{code}/approve", f"/pairings/{code2}/approve", MAC, {"name": bad},
                   {"code": code2})[0] == 400, bad
# ---- decline
assert request("DELETE /pairings/{code}", f"/pairings/{code2}", MAC, None, {"code": code2})[0] == 200
assert code2 not in h._pairing_requests()
assert request("DELETE /pairings/{code}", f"/pairings/{code2}", MAC, None, {"code": code2})[0] == 404

# ---- limits: 5 waiting at most, and requests expire after 10 minutes
for n in range(10, 15):  # none waiting before this: five fill the list
    assert request("POST /pair", "/pair", body={"public_key": line(pub_blob(n))})[0] == 200
st, b = request("POST /pair", "/pair", body={"public_key": line(pub_blob(20))})
assert st == 429 and "too many" in b["error"], (st, b)
old = next(iter(h._pairing_requests()))
clock[0] += 601
assert h._pairing_requests() == {}
assert request("POST /pairings/{code}/approve", f"/pairings/{old}/approve", MAC, {}, {"code": old})[0] == 404
assert request("POST /pair", "/pair", body={"public_key": line(pub_blob(20))})[0] == 403, "the window closed too"
request("POST /pairings/open", "/pairings/open", MAC)
assert request("POST /pair", "/pair", body={"public_key": line(pub_blob(20))})[0] == 200
# a poisoned or odd object under auth/pairing/ is ignored, not a crash
s3.objects["auth/pairing/999999.json"] = b'["x"]'
s3.objects["auth/pairing/notacode.json"] = b"{}"
assert request("GET /pairings", "/pairings", MAC)[0] == 200

# ---- revoke: refused after the 30 s cache
assert request("DELETE /devices/{id}", "/devices/ffffffff", MAC, None, {"id": "ffffffff"})[0] == 404
assert request("DELETE /devices/{id}", f"/devices/{dev_id}", MAC, None, {"id": dev_id})[0] == 200
assert request("GET /pairings", "/pairings", MAC)[1]["devices"] == []
clock[0] += 31
assert request("GET /jobs", "/jobs", PHONE)[0] == 401

# ---- a device approved by another client's computer key isn't listed for, or removable by, this one
h._authorized[pub_blob(30)] = "noa"
request("POST /pairings/open", "/pairings/open", pub_blob(30))
code3 = request("POST /pair", "/pair", body={"public_key": line(pub_blob(31))})[1]["code"]
st, b = request("POST /pairings/{code}/approve", f"/pairings/{code3}/approve", pub_blob(30), {}, {"code": code3})
assert st == 200 and b["device"]["client"] == "noa" and b["device"]["name"] == "phone"
assert request("GET /pairings", "/pairings", MAC)[1]["devices"] == []
assert request("DELETE /devices/{id}", "/x", MAC, None, {"id": b["device"]["id"]})[0] == 404
clock[0] += 31
assert request("GET /jobs", "/jobs", pub_blob(31)) == (200, {"client": "noa"})

# ---- the device list unreadable (S3 error): phones are refused with 401, computer keys still work
real_get = s3.get_object
def failing_get(Bucket, Key, Range=None):
    if Key == "auth/devices.json":
        raise KeyError("AccessDenied")
    return real_get(Bucket, Key)
s3.get_object = failing_get
h._s3_json = lambda key: (_ for _ in ()).throw(KeyError("AccessDenied")) if key == "auth/devices.json" else s3.json(key)
clock[0] += 31
assert request("GET /jobs", "/jobs", pub_blob(31))[0] == 401
assert request("GET /jobs", "/jobs", MAC) == (200, {"client": "dan"})

print("test_pairing: ok")
