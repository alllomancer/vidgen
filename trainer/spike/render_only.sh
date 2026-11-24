#!/usr/bin/env bash
# Re-run only the trial's render test against an already trained LoRA.
#   render_only.sh <client> <trial out dir, e.g. /opt/trial/out-20261109-165330>
set -uo pipefail
CLIENT=${1:?client}
OUT=${2:?trial out dir}
set -a; . /etc/ai-video.env; set +a
DEST="s3://$BUCKET/outputs/$CLIENT/$(basename $OUT | sed 's/^out-/trial-/')"
exec > >(tee -a $OUT/render_only.log) 2>&1

finish() {
  echo "== finishing (exit $1)"
  for f in $OUT/render.json $OUT/with_lora.mp4 $OUT/baseline.mp4 $OUT/render_only.log; do
    [ -f "$f" ] && aws s3 cp "$f" "$DEST/$(basename "$f")" --only-show-errors
  done
  systemctl start ai-video-worker
  echo "results: $DEST/"
}
trap 'finish $?' EXIT

systemctl stop ai-video-worker
start=$(date +%s)
(cd /opt/ai-video/worker && /opt/ai-video/venv/bin/python -u /opt/trial/render_test.py $OUT)
rc=$?
echo "render-test $(( $(date +%s) - start ))s (exit $rc)"
exit $rc
