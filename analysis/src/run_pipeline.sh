#!/usr/bin/env bash
# Football Analytics - Full pipeline runner
#
# Usage:
#   bash /work/scripts/run_pipeline.sh
#   bash /work/scripts/run_pipeline.sh --from 4
#   bash /work/scripts/run_pipeline.sh --from 5
#   bash /work/scripts/run_pipeline.sh --from 2 --to 5
#   bash /work/scripts/run_pipeline.sh --from 5 --skip-keyframe-calib

set -euo pipefail

FROM_STAGE=2
TO_STAGE=99
SKIP_KF_CALIB=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM_STAGE="${2:?}"; shift 2 ;;
    --to) TO_STAGE="${2:?}"; shift 2 ;;
    --skip-keyframe-calib) SKIP_KF_CALIB=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() {
  echo ""
  echo "=========================================="
  echo "  $1"
  echo "=========================================="
  date
}

skip() { [ "$FROM_STAGE" -gt "$1" ] && echo "  (Stage $1 skipped)" && return 0 || return 1; }

skip 2 || { log "STAGE 2 - Person Tracking"; python /work/scripts/stage2_tracking/person_track.py; }
skip 3 || { log "STAGE 3 - Gameplay Filter"; python /work/scripts/stage3_filter/gameplay_filter.py; }
skip 4 || { log "STAGE 4 - Team Clustering"; python /work/scripts/stage4_clustering/clustering.py; }
skip 5 || { log "STAGE 5 - Ball Tracking"; python /work/scripts/stage5_ball/ball_tracking.py; }

if [ "$TO_STAGE" -le 5 ]; then
  echo "  (--to $TO_STAGE: pipeline stopped here)"
  exit 0
fi

mkdir -p /output/stage2_tracking
mkdir -p /output/stage3_filter
mkdir -p /output/stage4_clustering/crops
mkdir -p /output/stage4_clustering/_chunks
mkdir -p /output/stage5_ball
mkdir -p /output/stage7_possession
mkdir -p /output/stage6_field/{keyframes,keyframe_json,keyframe_projected,debug,logs,single_frame_runs}
mkdir -p /output/final

if [ "$SKIP_KF_CALIB" = 1 ]; then
  log "STAGE 6 PREP - Keyframe/calibration skipped (--skip-keyframe-calib)"
  if ! compgen -G "/output/stage6_field/keyframe_json/*.json" > /dev/null; then
    echo "Error: keyframe_json is empty; run full calibration first or remove --skip-keyframe-calib." >&2
    exit 1
  fi
  cd /work
else
  log "STAGE 6 PREP - Keyframe Extraction"
  ffmpeg -y -i /output/stage3_filter/gameplay.mp4 \
    -vf "select='not(mod(n,5))'" -vsync vfr \
    /output/stage6_field/keyframes/frame_%06d.png

  log "STAGE 6 - Batch Keyframe Calibration (PnLCalib)"
  cd /work/PnLCalib
  PYTHONPATH=/work/PnLCalib python /work/scripts/stage6_field/calibrate.py
  cd /work
fi

log "STAGE 6 - Homography Bank Build"
python - <<'PY'
import glob
import json
import math
from pathlib import Path

src_files = sorted(glob.glob("/output/stage6_field/keyframe_json/*.json"))
out_path = Path("/output/stage6_field/homography_bank.json")

bank = {
    "version": 2,
    "accept_rule": {"final_params_not_none": True, "rep_err_max": 8.0, "sanity_score_min": 0.5},
    "frames": [],
}
accepted = 0

for fp in src_files:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    final_params = data.get("final_params_dict")
    rep_err = data.get("rep_err")
    sanity = data.get("sanity_score", 0.0)
    ok = (
        final_params is not None
        and rep_err is not None
        and math.isfinite(rep_err)
        and float(rep_err) <= 8.0
        and float(sanity) > 0.5
    )
    if ok:
        accepted += 1
    bank["frames"].append(
        {
            "image_path": data.get("image_path"),
            "accepted": ok,
            "rep_err": rep_err,
            "sanity_score": sanity,
            "final_params_dict": final_params,
        }
    )

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(bank, f, ensure_ascii=False, indent=2)

print(f"Bank written: accepted={accepted}/{len(bank['frames'])}")
PY

log "STAGE 6 - Motion-Propagated Homography Map"
python /work/scripts/stage6_field/homography_motion.py

log "STAGE 6 - Refined Homography Rescue"
python /work/scripts/stage6_field/refine_homography.py

log "STAGE 6 - Projection [Pass 1 - initial labels]"
if [ ! -f /output/stage4_clustering/track_labels_corrected.json ]; then
  cp /output/stage4_clustering/track_labels.json \
     /output/stage4_clustering/track_labels_corrected.json
  echo "Bootstrap: track_labels_corrected.json created."
fi
python /work/scripts/stage6_field/projection.py

log "STAGE 4b - Goalkeeper Label Fix"
python /work/scripts/stage4_clustering/fix_goalkeepers.py

log "STAGE 6 - Projection [Pass 2 - corrected labels]"
python /work/scripts/stage6_field/projection.py

log "STAGE 7 - Possession"
python /work/scripts/stage7_possession/possession.py

log "FINAL - Summary JSON + Overlay Video"
python /work/scripts/stage6_field/finalize.py

echo ""
echo "=========================================="
echo "  PIPELINE COMPLETED"
echo "=========================================="
date
echo ""
echo "Final video   : /output/final/overlay.mp4"
echo "Final summary : /output/final/summary.json"
ls -lh /output/final/
