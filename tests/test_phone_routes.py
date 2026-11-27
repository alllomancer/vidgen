"""The phone page: GET /m and GET /m/{file} are served without a signature, with fixed headers; every other route
still needs one. docs/specs/2026-11-27-phone-ui.md"""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install(os.path.join("lambda", "api"))
import handler as h  # noqa: E402


def call(route, raw_path, file=None):
    event = {"routeKey": route, "rawPath": raw_path, "headers": {},
             "requestContext": {"http": {"method": route.split()[0]}}}
    if file is not None:
        event["pathParameters"] = {"file": file}
    return h.handler(event, None)


S3 = "https://b.s3.amazonaws.com https://b.s3.us-east-1.amazonaws.com"  # bucket "b" in the test env
CSP = (f"default-src 'self'; img-src 'self' {S3}; media-src 'self' {S3}; connect-src 'self' {S3}; "
       "script-src 'self'; style-src 'self' 'unsafe-inline'; manifest-src 'self'; base-uri 'none'; "
       "form-action 'none'; object-src 'none'; worker-src 'none'; frame-ancestors 'none'")
EXPECT = {"/m": "text/html; charset=utf-8", "/m/app.js": "text/javascript; charset=utf-8",
          "/m/sign.js": "text/javascript; charset=utf-8",
          "/m/manifest.webmanifest": "application/manifest+json"}

for icon in ("icon-192.png", "icon-512.png"):  # binary: base64 through API Gateway
    r = call("GET /m/{file}", "/m/" + icon, icon)
    assert r["statusCode"] == 200 and r["isBase64Encoded"] is True and r["headers"]["content-type"] == "image/png", r
    assert base64.b64decode(r["body"]).startswith(b"\x89PNG"), icon
manifest = json.loads(call("GET /m/{file}", "/m/manifest.webmanifest", "manifest.webmanifest")["body"])
assert {i["sizes"] for i in manifest["icons"]} == {"192x192", "512x512"}, manifest

for path, ctype in EXPECT.items():
    file = path.split("/m/")[1] if "/m/" in path else None
    r = call("GET /m" if file is None else "GET /m/{file}", path, file)
    assert r["statusCode"] == 200, (path, r)
    hd = r["headers"]
    assert hd["content-type"] == ctype, (path, hd)
    assert hd["content-security-policy"] == CSP, hd
    assert hd["cache-control"] == "no-cache" and hd["x-content-type-options"] == "nosniff"
    assert hd["referrer-policy"] == "no-referrer" and hd["cross-origin-opener-policy"] == "same-origin", hd
    assert r["body"] and r.get("isBase64Encoded") is False, path
with open(os.path.join(h.PHONE_DIR, "m.html")) as f:
    assert call("GET /m", "/m")["body"] == f.read()

# unknown or sneaky file names: 404, nothing read outside the folder
for bad in ["x.js", "../handler.py", "..%2Fhandler.py", "m.html", ".hidden", "app.js/"]:
    r = call("GET /m/{file}", "/m/" + bad, bad)
    assert r["statusCode"] == 404 and r["headers"]["content-security-policy"] == CSP, (bad, r)

# the bundle without the page: a 500 that says so, not a crash
real = h.PHONE_DIR
h.PHONE_DIR = "/nonexistent"
r = call("GET /m", "/m")
assert r["statusCode"] == 500 and "not deployed" in r["body"], r
h.PHONE_DIR = real

# everything else still needs a signature
assert call("GET /jobs", "/jobs")["statusCode"] == 401
assert call("GET /status", "/status")["statusCode"] == 401

print("test_phone_routes: ok")
