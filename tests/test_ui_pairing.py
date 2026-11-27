"""The dashboard's Phones panel routes (cli/ui.py): codes and ids are checked before the signed call, methods map
to the right API call, and the host/token guard covers them. docs/specs/2026-11-27-phone-pairing.md"""
import io
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

calls = []
ui.vidgen.api = lambda method, path, body=None: calls.append((method, path, body)) or {"ok": True}
ui.vidgen.setting = lambda k, d=None: "https://abc.execute-api.us-east-1.amazonaws.com/" if k == "url" else d

TOKEN = "t" * 32
Handler = ui.make_handler(8765, TOKEN)


def hit(method, path, body=None, token=TOKEN, host="127.0.0.1:8765"):
    raw = json.dumps(body).encode() if body is not None else b""
    h = Handler.__new__(Handler)
    h.headers = {"Host": host, "X-UI-Token": token, "Content-Length": str(len(raw))}
    h.path, h.rfile, h.wfile = path, io.BytesIO(raw), io.BytesIO()
    sent = {}
    h.send_response = lambda code: sent.__setitem__("code", code)
    h.send_header = lambda *a: None
    h.end_headers = lambda: None
    h._route(method)
    return sent["code"], json.loads(h.wfile.getvalue() or b"{}")


st, body = hit("GET", "/api/phone")
assert st == 200 and body["url"] == "https://abc.execute-api.us-east-1.amazonaws.com/m", body
import qr  # noqa: E402
assert body["qr"] == qr.encode(body["url"]) and len(body["qr"]) == 33, "the QR code of exactly that link"
for method, path, body, want in [
        ("GET", "/api/pairings", None, ("GET", "/pairings", None)),
        ("POST", "/api/pairings", None, ("POST", "/pairings/open", None)),
        ("POST", "/api/pairings/123456/approve", {"name": " Pixel "}, ("POST", "/pairings/123456/approve", {"name": "Pixel"})),
        ("POST", "/api/pairings/123456/approve", ["x"], ("POST", "/pairings/123456/approve", {"name": ""})),
        ("DELETE", "/api/pairings/123456", None, ("DELETE", "/pairings/123456", None)),
        ("DELETE", "/api/devices/0a1b2c3d", None, ("DELETE", "/devices/0a1b2c3d", None))]:
    calls.clear()
    st, _ = hit(method, path, body)
    assert st == 200 and calls == [want], (method, path, st, calls)

for method, path in [("POST", "/api/pairings/12345/approve"), ("POST", "/api/pairings/1234567/approve"),
                     ("DELETE", "/api/pairings/%D9%A1%D9%A2%D9%A3%D9%A4%D9%A5%D9%A6"),
                     ("DELETE", "/api/pairings/١٢٣٤٥٦"), ("DELETE", "/api/devices/XYZ"),
                     ("DELETE", "/api/devices/0a1b2c3d4")]:
    calls.clear()
    st, _ = hit(method, path)
    assert st == 400 and not calls, (method, path, st)
for method, path in [("DELETE", "/api/pairings/123456/approve"), ("POST", "/api/pairings/123456"),
                     ("GET", "/api/devices/0a1b2c3d"), ("PUT", "/api/pairings")]:
    calls.clear()
    st, _ = hit(method, path)
    assert st == 404 and not calls, (method, path, st)
calls.clear()
assert hit("GET", "/api/pairings", token="wrong")[0] == 403 and not calls
assert hit("GET", "/api/pairings", host="evil.example:8765")[0] == 403 and not calls

print("test_ui_pairing: ok")
