"""The phone page's signer (lambda/api/phone/sign.js) makes the same signature as `ssh-keygen -Y sign` and the CLI,
so the API accepts it unchanged. Needs node and ssh-keygen; skipped without node. docs/specs/2026-11-27-phone-ui.md"""
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGN_JS = os.path.join(ROOT, "lambda", "api", "phone", "sign.js")

if not shutil.which("node"):
    print("test_sign_parity: skipped (node not installed)")
    sys.exit(0)


def read_string(buf, off):
    n = int.from_bytes(buf[off:off + 4], "big")
    return buf[off + 4:off + 4 + n], off + 4 + n


def openssh_seed(path):
    """(seed, public key) of an unencrypted OpenSSH ed25519 private key file."""
    with open(path) as f:
        raw = base64.b64decode("".join(l for l in f.read().splitlines() if not l.startswith("-----")))
    assert raw.startswith(b"openssh-key-v1\0")
    off = len(b"openssh-key-v1\0")
    cipher, off = read_string(raw, off)
    assert cipher == b"none"
    _, off = read_string(raw, off)  # kdf name
    _, off = read_string(raw, off)  # kdf options
    off += 4  # number of keys
    _, off = read_string(raw, off)  # public key blob
    priv, _ = read_string(raw, off)
    o = 8  # two check ints
    _, o = read_string(priv, o)  # key type
    pub, o = read_string(priv, o)
    sk, o = read_string(priv, o)  # seed + public key
    return sk[:32], pub


tmp = tempfile.mkdtemp()
keyfile = os.path.join(tmp, "k")
subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f", keyfile], check=True)
seed, pub = openssh_seed(keyfile)
pkcs8 = bytes.fromhex("302e020100300506032b657004220420") + seed

cases = [("POST", "/jobs", 1_800_000_000, "0123456789abcdef0123456789abcdef", '{"type":"write","premise":"x"}'),
         ("GET", "/jobs", 1_800_000_123, "f" * 32, ""),
         ("GET", "/jobs/" + "a" * 32, 1_800_000_456, "ab" * 16, ""),
         ("DELETE", "/avatars/dana", 1_800_000_789, "cd" * 16, '{"loras": true}'),
         ("PUT", "/avatars/default", 1_800_000_999, "ef" * 16, "x" * 300000)]  # a big body
# Paths are signed the way fetch() sends them (query/fragment dropped, dots resolved): same as the API's rawPath.
WIRE = {"/jobs?x=1": "/jobs", "/jobs#a": "/jobs", "/m/../jobs": "/jobs"}
for method, path, ts, nonce, body in cases + [("GET", p, 1_800_001_000, "12" * 16, "") for p in WIRE]:
    out = json.loads(subprocess.run(
        ["node", os.path.join(ROOT, "tests", "sign_parity.mjs"), base64.b64encode(pkcs8).decode(),
         base64.b64encode(pub).decode(), method, path, str(ts), nonce, body],
        check=True, capture_output=True, text=True).stdout)
    h = out["headers"]
    assert h["x-vidgen-timestamp"] == str(ts) and h["x-vidgen-nonce"] == nonce, h

    # Same message as cli/vidgen.py sign() and the API's _authenticate.
    wire = WIRE.get(path, path)
    message = "\n".join(["vidgen-v1", method, wire, str(ts), nonce, hashlib.sha256(body.encode()).hexdigest()])
    ref = subprocess.run(["ssh-keygen", "-Y", "sign", "-q", "-f", keyfile, "-n", "vidgen-api"],
                         input=message.encode(), capture_output=True, check=True).stdout
    mine = base64.b64decode(h["x-vidgen-signature"])
    assert mine == ref, (method, path, mine, ref)  # ed25519 signatures are deterministic

    # And OpenSSH itself accepts it for this key and namespace.
    allowed = os.path.join(tmp, "allowed")
    with open(allowed, "w") as f:
        f.write(f"phone {out['line'].rsplit(' ', 1)[0]}\n")
    sig = os.path.join(tmp, "sig")
    with open(sig, "wb") as f:
        f.write(mine)
    subprocess.run(["ssh-keygen", "-Y", "verify", "-f", allowed, "-I", "phone", "-n", "vidgen-api", "-s", sig],
                   input=message.encode(), capture_output=True, check=True)

# The public key line is the key's own .pub line (with our comment).
with open(keyfile + ".pub") as f:
    assert out["line"] == " ".join(f.read().split()[:2]) + " phone", out["line"]

# The browser key is created non-extractable, and the signer drops the query string like the API's rawPath.
src = open(SIGN_JS).read()
assert re.search(r'generateKey\(\{ name: "Ed25519" \}, false, \["sign"\]\)', src), "private key must be non-extractable"
assert "new URL(path" in src and ".pathname" in src

print("test_sign_parity: ok")
