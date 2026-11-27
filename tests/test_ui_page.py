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

# the AI-look toggle is in the video, photos and story forms, and sent with the job
assert page.count('name="realism_check" data-remember="realism_check"') == 3
assert 'realism_check: on("realism_check") || undefined' in script

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
