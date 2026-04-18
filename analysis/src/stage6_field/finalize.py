"""
Stage 6 — Final Output (Summary JSON + Overlay Video)

İki iş yapar:
  1. summary.json  — tüm stage istatistiklerini özetleyen JSON (hızlı)
  2. overlay.mp4   — gameplay videosu üzerine minimap + bbox overlay (yavaş)

Giriş : tüm stage çıktıları
Çıkış : /output/final/summary.json
         /output/final/overlay.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─── GİRİŞ DOSYALARI ─────────────────────────────────────────────────────────
VIDEO_PATH      = "/output/stage3_filter/gameplay.mp4"
GAMEPLAY_JSON   = "/output/stage3_filter/gameplay.json"
FILTER_REPORT   = "/output/stage3_filter/filter_report.json"
TRACK_LABELS    = "/output/stage4_clustering/track_labels.json"
TRACK_LABELS_C  = "/output/stage4_clustering/track_labels_corrected.json"
BALL_JSON       = "/output/stage5_ball/ball_tracks.json"
BANK_JSON       = "/output/stage6_field/homography_bank.json"
HMAP_JSON       = "/output/stage6_field/homography_map_refined.json"
POSSESSION_JSON = "/output/stage7_possession/possession.json"
PROJECTION_JSON = "/output/stage6_field/projection.json"

PERSON_TRACK_JSON = "/output/stage2_tracking/tracking.json"

# ─── ÇIKTI DOSYALARI ─────────────────────────────────────────────────────────
OUT_SUMMARY = "/output/final/summary.json"
OUT_VIDEO   = "/output/final/overlay.mp4"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-path", default=VIDEO_PATH)
    ap.add_argument("--gameplay-json", default=GAMEPLAY_JSON)
    ap.add_argument("--filter-report", default=FILTER_REPORT)
    ap.add_argument("--track-labels", default=TRACK_LABELS)
    ap.add_argument("--track-labels-corrected", default=TRACK_LABELS_C)
    ap.add_argument("--ball-json", default=BALL_JSON)
    ap.add_argument("--bank-json", default=BANK_JSON)
    ap.add_argument("--hmap-json", default=HMAP_JSON)
    ap.add_argument("--possession-json", default=POSSESSION_JSON)
    ap.add_argument("--projection-json", default=PROJECTION_JSON)
    ap.add_argument("--person-track-json", default=PERSON_TRACK_JSON)
    ap.add_argument("--out-summary", default=OUT_SUMMARY)
    ap.add_argument("--out-video", default=OUT_VIDEO)
    return ap.parse_args()

# ─── MİNİMAP AYARLARI ────────────────────────────────────────────────────────
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M  = 68.0
MINIMAP_W      = 420
MINIMAP_H      = 272
BOTTOM_MARGIN  = 24
PANEL_PAD      = 14
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# ─── YARDIMCI FONKSİYONLAR ───────────────────────────────────────────────────
def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def pitch_to_canvas(x, y, x0, y0, w, h):
    return (x0 + int((x / FIELD_LENGTH_M) * w),
            y0 + int((y / FIELD_WIDTH_M)  * h))

def draw_pitch(canvas, x0, y0, w, h):
    lc = (255, 255, 255)
    x1, y1 = x0+w, y0+h
    cv2.rectangle(canvas, (x0, y0), (x1, y1), lc, 2)
    mx = x0 + w//2
    cv2.line(canvas, (mx, y0), (mx, y1), lc, 2)
    cv2.circle(canvas, pitch_to_canvas(52.5, 34.0, x0, y0, w, h),
               int((9.15/FIELD_WIDTH_M)*h), lc, 2)
    for (xa, ya, xb, yb) in [
        (0.0, 13.84, 16.5,  54.16),
        (88.5,13.84, 105.0, 54.16),
        (0.0, 24.84,  5.5,  43.16),
        (99.5,24.84, 105.0, 43.16),
    ]:
        cv2.rectangle(canvas,
                      pitch_to_canvas(xa, ya, x0, y0, w, h),
                      pitch_to_canvas(xb, yb, x0, y0, w, h), lc, 2)

def alpha_rect(frame, x1, y1, x2, y2, color=(20,20,20), a=0.28):
    ov = frame.copy()
    cv2.rectangle(ov, (x1,y1), (x2,y2), color, -1)
    cv2.addWeighted(ov, a, frame, 1-a, 0, frame)

def label_color(label):
    return {"team_1": (255,0,0), "team_2": (0,140,255),
            "referee": (0,255,255)}.get(label, (180,180,180))

def arrow_above(frame, bbox, color=(0,0,255)):
    x1,y1,x2,_ = map(int, bbox)
    cx = (x1+x2)//2
    cv2.arrowedLine(frame, (cx, max(8,y1-48)), (cx, max(8,y1-8)), color, 4, tipLength=0.45)

# ─── JSON'LARI YÜKLEYİP SUMMARY YAZ ─────────────────────────────────────────
args = parse_args()
VIDEO_PATH = args.video_path
GAMEPLAY_JSON = args.gameplay_json
FILTER_REPORT = args.filter_report
TRACK_LABELS = args.track_labels
TRACK_LABELS_C = args.track_labels_corrected
BALL_JSON = args.ball_json
BANK_JSON = args.bank_json
HMAP_JSON = args.hmap_json
POSSESSION_JSON = args.possession_json
PROJECTION_JSON = args.projection_json
PERSON_TRACK_JSON = args.person_track_json
OUT_SUMMARY = args.out_summary
OUT_VIDEO = args.out_video

Path(OUT_SUMMARY).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)

step2      = load(PERSON_TRACK_JSON)
step3_gp   = load(GAMEPLAY_JSON)
step3_rep  = load(FILTER_REPORT)
step4      = load(TRACK_LABELS)
step5      = load(BALL_JSON)
poss       = load(POSSESSION_JSON)
bank       = load(BANK_JSON)
hmap       = load(HMAP_JSON)
proj       = load(PROJECTION_JSON)

track_counts = {}
for v in step4["tracks"].values():
    lbl = v["label"]
    track_counts[lbl] = track_counts.get(lbl, 0) + 1

bf         = step5["frames"]
pf         = proj["frames"]
b_vis      = sum(1 for fr in bf if bool(fr.get("ball", fr).get("visible", False)))
b_interp   = sum(1 for fr in bf if bool(fr.get("ball", fr).get("interpolated", False)))
b_xy       = sum(1 for fr in bf if fr.get("ball", fr).get("image_xy") is not None)
b_trusted  = sum(1 for fr in pf if fr["ball"].get("pitch_xy") is not None)
# Güven histogramı: Stage 5 (YOLO) confidence (enrich_ball yok)
bcf = lambda fr: float(fr.get("ball", fr).get("confidence") or 0.0)
b_high     = sum(1 for fr in bf if bcf(fr) >= 0.72)
b_med      = sum(1 for fr in bf if 0.45 <= bcf(fr) < 0.72)
b_low      = sum(1 for fr in bf if bcf(fr) < 0.45)
kf_accept  = sum(1 for fr in bank["frames"] if fr["accepted"])

summary = {
    "input_video":  "/input/input.mp4",
    "final_video":  OUT_VIDEO,
    "stage2": {
        "model_path":   step2.get("model_path"),
        "fps":          step2.get("fps"),
        "width":        step2.get("width"),
        "height":       step2.get("height"),
        "total_frames": len(step2.get("frames", [])),
    },
    "stage3": {
        "gameplay_video":  "/output/stage3_filter/gameplay.mp4",
        "gameplay_json":   GAMEPLAY_JSON,
        "report":          step3_rep,
        "kept_frames":     len(step3_gp.get("frames", [])),
    },
    "stage4": {
        "track_counts":          track_counts,
        "total_clustered_tracks": len(step4.get("tracks", {})),
    },
    "stage5": {
        "total_frames":              len(bf),
        "ball_visible_frames":       b_vis,
        "ball_interpolated_frames":  b_interp,
        "ball_has_xy_frames":        b_xy,
        "ball_trusted_pitch_frames": b_trusted,
        "ball_high_conf_frames":     b_high,
        "ball_med_conf_frames":      b_med,
        "ball_low_conf_frames":      b_low,
    },
    "stage6_field": {
        "accepted_keyframes":       kf_accept,
        "total_keyframes":          len(bank["frames"]),
        "homography_total_frames":  hmap.get("total_frames"),
        "projection_total_frames":  proj.get("total_frames"),
    },
    "stage7_possession": {
        "summary": poss.get("summary", {}),
    },
    "artifacts": {
        "person_tracking_json":    PERSON_TRACK_JSON,
        "gameplay_json":           GAMEPLAY_JSON,
        "track_labels_json":       TRACK_LABELS,
        "ball_json":               BALL_JSON,
        "possession_json":         POSSESSION_JSON,
        "homography_bank_json":    BANK_JSON,
        "frame_homography_map_json": HMAP_JSON,
        "projected_pitch_json":    PROJECTION_JSON,
    },
}

with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"  summary → {OUT_SUMMARY}")

# ─── OVERLAY VİDEO ───────────────────────────────────────────────────────────
labels_corrected = load(TRACK_LABELS_C)
id_to_label = {int(k): v["label"] for k, v in labels_corrected["tracks"].items()}

gp_frames   = step3_gp["frames"]
ball_frames = step5["frames"]
poss_frames = poss["frames"]
proj_frames = proj["frames"]

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Video açılamadı: {VIDEO_PATH}")

fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (video_w, video_h))
if not writer.isOpened():
    raise RuntimeError(f"Video yazılamadı: {OUT_VIDEO}")

# Minimap panel boyutları
panel_w  = MINIMAP_W + 2*PANEL_PAD
panel_h  = MINIMAP_H + 2*PANEL_PAD
panel_x1 = (video_w - panel_w) // 2
panel_y1 = video_h - panel_h - BOTTOM_MARGIN
panel_x2, panel_y2 = panel_x1 + panel_w, panel_y1 + panel_h
map_x0, map_y0 = panel_x1 + PANEL_PAD, panel_y1 + PANEL_PAD

total = min(len(gp_frames), len(ball_frames), len(poss_frames), len(proj_frames))

print(f"\nFINAL | Overlay Video — {total} frame")
for idx in tqdm(range(total), desc="FINAL | Overlay Video", unit="frame", ncols=90):
    ret, frame = cap.read()
    if not ret:
        break

    gf   = gp_frames[idx]
    bmet = ball_frames[idx].get("ball", ball_frames[idx])
    pmet = poss_frames[idx]
    prmt = proj_frames[idx]

    cur_player_id = pmet.get("current_player_id")

    # ── Oyuncu bbox'ları ──────────────────────────────────────────────────────
    for obj in gf["objects"]:
        tid = int(obj["track_id"])
        lbl = id_to_label.get(tid, "unknown")
        x1, y1, x2, y2 = map(int, obj["bbox_xyxy"])
        col = label_color(lbl)
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 3 if lbl in ("team_1","team_2") else 2)
        cv2.putText(frame, f"{lbl} | ID {tid}", (x1, max(18,y1-6)),
                    FONT, 0.55, col, 2, cv2.LINE_AA)
        if cur_player_id is not None and tid == int(cur_player_id):
            arrow_above(frame, obj["bbox_xyxy"])

    # ── Top ───────────────────────────────────────────────────────────────────
    if bmet.get("image_xy") is not None:
        bx, by   = map(int, bmet["image_xy"])
        ball_conf = float(bmet.get("confidence") or prmt["ball"].get("ball_confidence", 0.5))
        col = ((255,255,255) if ball_conf >= 0.72
               else (0,255,255) if ball_conf >= 0.45
               else (120,120,120))
        cv2.circle(frame, (bx,by), 6, col, -1)
        cv2.circle(frame, (bx,by), 10, (0,0,0), 2)

    # ── Minimap panel ─────────────────────────────────────────────────────────
    alpha_rect(frame, panel_x1, panel_y1, panel_x2, panel_y2)
    ov = frame.copy()
    cv2.rectangle(ov, (map_x0, map_y0),
                  (map_x0+MINIMAP_W, map_y0+MINIMAP_H), (40,120,40), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    draw_pitch(frame, map_x0, map_y0, MINIMAP_W, MINIMAP_H)

    # projected oyuncular
    for p in prmt["players"]:
        xy = p.get("pitch_xy")
        if xy is None or not (0 <= xy[0] <= FIELD_LENGTH_M and 0 <= xy[1] <= FIELD_WIDTH_M):
            continue
        cx, cy = pitch_to_canvas(xy[0], xy[1], map_x0, map_y0, MINIMAP_W, MINIMAP_H)
        col = (255,0,0) if p["label"] == "team_1" else (0,140,255)
        r = 6 if (cur_player_id is not None and int(p["track_id"]) == int(cur_player_id)) else 5
        cv2.circle(frame, (cx,cy), r, col, -1)

    # projected hakemler
    for r in prmt["referees"]:
        xy = r.get("pitch_xy")
        if xy is None or not (0 <= xy[0] <= FIELD_LENGTH_M and 0 <= xy[1] <= FIELD_WIDTH_M):
            continue
        cx, cy = pitch_to_canvas(xy[0], xy[1], map_x0, map_y0, MINIMAP_W, MINIMAP_H)
        cv2.circle(frame, (cx,cy), 4, (0,255,255), -1)

    # projected top
    bxy = prmt["ball"].get("pitch_xy_trusted") or prmt["ball"].get("pitch_xy")
    if bxy is not None and 0 <= bxy[0] <= FIELD_LENGTH_M and 0 <= bxy[1] <= FIELD_WIDTH_M:
        cx, cy = pitch_to_canvas(bxy[0], bxy[1], map_x0, map_y0, MINIMAP_W, MINIMAP_H)
        cv2.circle(frame, (cx,cy), 4, (255,255,255), -1)
        cv2.circle(frame, (cx,cy), 7, (0,0,0), 1)

    cv2.putText(frame, "2D Pitch Map",
                (panel_x1+10, panel_y1-8), FONT, 0.65, (255,255,255), 2, cv2.LINE_AA)
    writer.write(frame)

cap.release()
writer.release()

print(f"  overlay  → {OUT_VIDEO}  ({total} frames)")
print("DONE")
