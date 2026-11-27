"""The phone page (lambda/api/phone/): no inline script or HTML injection points, every API call is signed, the
element ids the script uses exist, and the scripts parse. There is no browser test; this catches a broken or
loosened page. docs/specs/2026-11-27-phone-ui.md"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHONE = os.path.join(ROOT, "lambda", "api", "phone")
page = open(os.path.join(PHONE, "m.html")).read()
app = open(os.path.join(PHONE, "app.js")).read()
sign = open(os.path.join(PHONE, "sign.js")).read()

# ---- the page: scripts only from /m/, no inline handlers (the CSP would block them anyway)
scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", page, re.S)
assert scripts and all('src="/m/' in attrs and not body.strip() for attrs, body in scripts), scripts
assert not re.search(r"\son[a-z]+\s*=", page), "no inline event handlers"
assert 'name="viewport"' in page and 'rel="manifest"' in page

# ---- the script: data goes in as text, network only through signedFetch
for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
    assert banned not in app, f"app.js must not use {banned}"
assert not re.search(r"(?<![A-Za-z])fetch\(", app), "app.js calls the API only through signedFetch"
assert re.findall(r'^import .* from "([^"]+)";', app, re.M) == ["/m/sign.js"]

# ---- the API calls it makes are ones the phone is meant to make
ALLOWED = {("GET", "/jobs"), ("POST", "/jobs/{id}/cancel"), ("POST", "/jobs/{id}/rerun")}
calls = set()
for method, path in re.findall(r'call\("([A-Z]+)", [`"]([^`"]+)[`"]', app):
    calls.add((method, re.sub(r"\$\{[^}]+\}", "{id}", path)))
assert calls and calls <= ALLOWED, calls - ALLOWED

# ---- every element the script looks up is on the page
ids = set(re.findall(r'\bid="([^"]+)"', page))
used = set(re.findall(r'\$\("([^"]+)"\)', app))
assert used <= ids, used - ids
for view in re.findall(r'data-view="([^"]+)"', page):
    assert view in ids, f"nav button for a missing section: {view}"

# ---- key handling: created non-extractable, only the public half is shown
assert 'generateKey({ name: "Ed25519" }, false, ["sign"])' in sign
assert 'exportKey("raw", pair.publicKey)' in sign and 'exportKey("pkcs8"' not in sign and "exportKey(\"jwk\"" not in sign

node = shutil.which("node")
if node:
    for name, src in (("app", app), ("sign", sign)):
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
            f.write(src)
        try:
            r = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
            assert r.returncode == 0, (name, r.stderr[-800:])
        finally:
            os.unlink(f.name)
    print("test_phone_page: ok")
else:
    print("test_phone_page: ok (node not installed: script syntax not checked)")
