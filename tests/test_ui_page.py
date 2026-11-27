"""The dashboard page (cli/ui.html): its script parses, and the controls the docs name are there.
The page has no browser test; this catches a broken script and a removed control."""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
page = open(os.path.join(ROOT, "cli", "ui.html")).read()
script = "\n".join(re.findall(r"<script>(.*?)</script>", page, re.S))
assert len(script) > 10000

for needed in ('data-copy-prompt=', 'data-use-prompt=', "function copyText(", "navigator.clipboard.writeText",
               'id="restart-worker"', 'id="ready-bar"', 'name="video_model"', 'name="shot" value="stream"',
               'name="shot" value="pov"', 'id="viewer"', 'id="realism-dialog"', 'id="realism-use"',
               "function realismBadge(", "function useRealism("):
    assert needed in page, f"missing from the dashboard: {needed}"
# copying is offered for the job kinds whose summary is the prompt as sent
assert re.search(r'function copyable\(j\) \{ return !!j\.summary && \["video", "photos", "avatar"\]\.includes\(j\.type\); \}', script)

# avatar cards: Upload LoRAs opens the dialog with the Krea 2, Wan and Wan low-noise paths
assert 'data-avatar-loras="${name}"' in script and 'id="avatar-lora-dialog"' in page
assert all(f'name="{n}"' in page for n in ("krea", "wan", "wan_low")) and "`${path}/loras`" in script

# a new avatar (generated or from photos): which LoRAs to train, including none (upload your own)
assert page.count('<select name="loras">') == 2 and page.count('<option value="none">None: I') == 2
assert script.count("...loraChoice(f)") == 2 and 'name="krea_lora"' not in page

# the Phones panel draws the link's QR code from ui.py's rows, as SVG elements (no HTML, no outside script)
assert 'id="phone-qr"' in page and "function drawQr(" in script and "drawQr(r.qr)" in script
assert "/^[01]+$/.test(r)" in script
# an <svg> has no .hidden property: the QR shows only if the attribute is removed
qr_fn = script[script.index("function drawQr("):script.index("api(\"GET\", \"/api/phone\")")]
assert 'svg.removeAttribute("hidden")' in qr_fn and "svg.hidden = false" not in qr_fn

# the AI-look toggle is in the video, photos and story forms, and sent with the job
assert page.count('name="realism_check" data-remember="realism_check"') == 3
assert 'realism_check: (on("realism_check") || retry) || undefined' in script
# retry until it looks real: in the video and photos forms only; turns the check on; one 5 s clip / one picture
assert page.count('name="realism_retry"') == 2 and "realism_retries: retry ? 25 : undefined" in script
assert "seconds: retry ? 5 : num(d.seconds)" in script and "count: retry ? 1 : num(d.count)" in script
assert "function retryHtml(" in script and "${esc(a.prompt" in script
# Use suggestion on photos clears what would override the prompt (Series, Where/Wearing/Doing/Camera)
use = script[script.index("function useRealism("):script.index("function rowHtml(")]
assert 'f.elements.series.checked = false; photoForm();' in use and '"where", "wearing", "doing", "camera", "actions"' in use
# the verdict is model output: every value goes through esc() (or Number()) before it is HTML
dialog = script[script.index("async function showRealism("):script.index("function useRealism(")]
assert "${esc(x)}" in dialog and "${esc(r.prompt)}" in dialog and "${esc(t)}" in dialog and "${esc(r.error)}" in dialog and "${esc(r.answer)}" in dialog

node = shutil.which("node")
if node:
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(script)
    try:
        r = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
    finally:
        os.unlink(f.name)
    print("ui page tests passed")
else:
    print("ui page tests passed (node not installed: the script's syntax was not checked)")
