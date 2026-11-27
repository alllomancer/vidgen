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
# sign.js makes exactly two kinds of request: signed API calls (same origin) and the presigned S3 PUT
assert len(re.findall(r"await fetch\(", sign)) == len(re.findall(r"(?<![A-Za-z.])fetch\((?!\))", sign)) == 3, \
    "sign.js: signedFetch, putUpload and requestPairing only"
assert 'fetch("/pair", { method: "POST"' in sign and "public_key: publicKeyLine(key.pubRaw)" in sign
assert 'u.hostname.endsWith(".amazonaws.com")' in sign and 'u.protocol !== "https:"' in sign
assert re.findall(r'^import\b[^;]*?from "([^"]+)";', app, re.M | re.S) == ["/m/sign.js"]

# ---- the API calls it makes are ones the phone is meant to make
ALLOWED = {("GET", "/jobs"), ("POST", "/jobs/{id}/cancel"), ("POST", "/jobs/{id}/rerun"), ("POST", "/jobs"),
           ("POST", "/uploads"), ("GET", "/avatars"), ("GET", "/jobs/{id}"), ("PUT", "/avatars/default"),
           ("POST", "/avatars/{id}/images"), ("GET", "/status")}
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

# ---- Create: jobs carry meta {source: phone}; the client file-name rule is the API's; HEIC gets the spec's message
assert 'meta = { source: "phone" }' in app
api_src = open(os.path.join(ROOT, "lambda", "api", "handler.py")).read()
api_rule = re.search(r'OUT_NAME = re\.compile\(r"(.+?)" %', api_src).group(1) % (128 - 2)
assert f"const OUT_NAME = /^{api_rule}$/;" in app, "phone file-name rule must match the API's"
assert "set the camera to JPEG or pick another photo" in app
for form_field in ('name="prompt"', 'name="seconds"', 'name="avatar"', 'name="subject"', 'name="photo"',
                   'name="scenes"', 'name="count"', 'name="story"', 'name="subtitles"', 'name="narration"'):
    assert form_field in page, form_field
assert page.count('name="out"') == 3

# ---- Job screen: previews from downloads, Save from the attachment links, both only to the bucket
assert "j.saves" in app and "bucketLink(saves[name])" in app and 'u.hostname.endsWith(".amazonaws.com")' in app

# ---- AI-look check: the toggle in all three forms, remembered, sent as realism_check; verdict shown as text
assert page.count('name="realism_check" data-remember') == 3 and 'class="note suggested"' not in page
assert "body.realism_check = true" in app and "REMEMBER_CHECK" in app and 'id="job-verdict"' in page
assert "function useSuggestion(" in app and "showVerdict(j)" in app and "realismBadge(j.realism)" in app
assert page.count('name="realism_retry"') == 2 and "body.realism_retries = 25" in app and "function retryList(" in app
assert "Suggested settings" not in app and "allowedOptions" not in app, "the suggestion is the prompt; no settings ride along"

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
