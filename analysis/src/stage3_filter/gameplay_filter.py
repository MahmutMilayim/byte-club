import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# =========================
# DOSYA YOLLARI
# =========================
VIDEO_PATH = r"/input/input.mp4"
INPUT_JSON_PATH = r"/output/stage2_tracking/tracking.json"

REVIEW_VIDEO_PATH = r"/output/stage3_filter/review.mp4"
GAMEPLAY_VIDEO_PATH = r"/output/stage3_filter/gameplay.mp4"
GAMEPLAY_JSON_PATH = r"/output/stage3_filter/gameplay.json"
REPORT_JSON_PATH = r"/output/stage3_filter/filter_report.json"

# =========================
# EŞİKLER
# Bunlar ilk sağlam baseline değerleri
# Gerekirse sonra birlikte ayarlarız
# =========================
MIN_GREEN_RATIO = 0.22
MIN_PERSONS = 8
MAX_BBOX_HEIGHT_RATIO = 0.45
MERGE_GAP_FRAMES = 5
VERBOSE = True
LOG_EVERY_N_FRAMES = 50

# Sahadaki yeşili HSV uzayında kabaca yakalamak için
GREEN_LOWER = np.array([25, 35, 35], dtype=np.uint8)
GREEN_UPPER = np.array([95, 255, 255], dtype=np.uint8)

# =========================
# DOSYA KONTROLLERİ
# =========================
if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(f"Video bulunamadı: {VIDEO_PATH}")

if not Path(INPUT_JSON_PATH).exists():
    raise FileNotFoundError(f"JSON bulunamadı: {INPUT_JSON_PATH}")

Path(REVIEW_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(GAMEPLAY_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(GAMEPLAY_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(REPORT_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)

# =========================
# JSON YÜKLE
# =========================
with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
    tracking_data = json.load(f)

frames_info = tracking_data["frames"]
fps = float(tracking_data["fps"])
width = int(tracking_data["width"])
height = int(tracking_data["height"])
frame_count = int(tracking_data["frame_count"])

# =========================
# VİDEO AÇ
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video açılamadı.")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
review_writer = cv2.VideoWriter(REVIEW_VIDEO_PATH, fourcc, fps, (width, height))
if not review_writer.isOpened():
    raise RuntimeError("Review video oluşturulamadı.")

# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def compute_green_ratio(frame_bgr):
    # Üstteki skor tabelası / tribün etkisini azaltmak için
    y1 = int(height * 0.18)
    roi = frame_bgr[y1:height, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    green_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return green_ratio

def get_bbox_height_ratios(objects):
    ratios = []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        h = max(0.0, float(y2) - float(y1))
        ratios.append(h / height)
    return ratios

def draw_label_box(frame, text, org, bg_color, text_color=(255, 255, 255), font_scale=0.8, thickness=2):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y + 5), bg_color, -1)
    cv2.putText(frame, text, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

# =========================
# FRAME FRAME ANALİZ
# =========================
keep_flags = []
analysis_rows = []

frame_idx = 0
_total1 = min(len(frames_info), frame_count)
print(f"\nStage 3 | Gameplay Filter — Pass 1/2 (analiz) — {_total1} frame")
_pbar1 = tqdm(total=_total1, desc="Stage 3 | Analysis", unit="frame", ncols=90)

while True:
    ret, frame = cap.read()
    if not ret:
        _pbar1.close()
        break
    if frame_idx >= len(frames_info):
        _pbar1.close()
        break

    frame_info = frames_info[frame_idx]
    objects = frame_info["objects"]

    person_count = len(objects)
    green_ratio = compute_green_ratio(frame)
    bbox_height_ratios = get_bbox_height_ratios(objects)

    if len(bbox_height_ratios) > 0:
        # 90. persentil: tek yakın oyuncu tüm frame'i reddetmesin
        max_bbox_height_ratio = float(np.percentile(bbox_height_ratios, 90))
        median_bbox_height_ratio = float(np.median(bbox_height_ratios))
    else:
        max_bbox_height_ratio = 0.0
        median_bbox_height_ratio = 0.0

    # KEEP/REJECT kararı
    keep = (
        green_ratio >= MIN_GREEN_RATIO and
        person_count >= MIN_PERSONS and
        max_bbox_height_ratio <= MAX_BBOX_HEIGHT_RATIO
    )

    keep_flags.append(bool(keep))

    analysis_rows.append({
        "frame_index": frame_idx,
        "time_sec": round(frame_idx / fps, 3),
        "green_ratio": round(green_ratio, 4),
        "person_count": int(person_count),
        "max_bbox_height_ratio": round(float(max_bbox_height_ratio), 4),
        "median_bbox_height_ratio": round(float(median_bbox_height_ratio), 4),
        "keep": bool(keep)
    })

    # Review video üstüne bilgi bas
    review_frame = frame.copy()

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(review_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    status_text = "KEEP" if keep else "REJECT"
    status_color = (0, 180, 0) if keep else (0, 0, 255)

    draw_label_box(review_frame, f"STATUS: {status_text}", (20, 45), status_color)
    draw_label_box(review_frame, f"Frame: {frame_idx}", (20, 95), (70, 70, 70))
    draw_label_box(review_frame, f"Time: {frame_idx / fps:.2f}s", (20, 145), (70, 70, 70))
    draw_label_box(review_frame, f"Persons: {person_count}", (20, 195), (70, 70, 70))
    draw_label_box(review_frame, f"Green ratio: {green_ratio:.3f}", (20, 245), (70, 70, 70))
    draw_label_box(review_frame, f"Max bbox h ratio: {max_bbox_height_ratio:.3f}", (20, 295), (70, 70, 70))

    review_writer.write(review_frame)

    frame_idx += 1
    _pbar1.update(1)

cap.release()
review_writer.release()

# =========================
# KISA BOŞLUKLARI BİRLEŞTİR
# örn KEEP KEEP REJECT KEEP ise ortadaki kısa boşluk tolere edilsin
# =========================
smoothed_keep_flags = keep_flags.copy()
n = len(smoothed_keep_flags)

i = 0
while i < n:
    if smoothed_keep_flags[i]:
        i += 1
        continue

    start = i
    while i < n and not smoothed_keep_flags[i]:
        i += 1
    end = i - 1

    gap_len = end - start + 1
    prev_keep = (start - 1 >= 0 and smoothed_keep_flags[start - 1])
    next_keep = (end + 1 < n and smoothed_keep_flags[end + 1])

    if prev_keep and next_keep and gap_len <= MERGE_GAP_FRAMES:
        for j in range(start, end + 1):
            smoothed_keep_flags[j] = True

# analysis_rows içindeki keep değerini güncelle
for i in range(min(len(analysis_rows), len(smoothed_keep_flags))):
    analysis_rows[i]["keep_after_smoothing"] = bool(smoothed_keep_flags[i])

# =========================
# SEGMENTLERİ ÇIKAR
# =========================
segments = []
i = 0
while i < len(smoothed_keep_flags):
    if not smoothed_keep_flags[i]:
        i += 1
        continue

    start = i
    while i < len(smoothed_keep_flags) and smoothed_keep_flags[i]:
        i += 1
    end = i - 1

    segments.append({
        "start_frame": int(start),
        "end_frame": int(end),
        "start_time_sec": round(start / fps, 3),
        "end_time_sec": round(end / fps, 3),
        "length_frames": int(end - start + 1),
        "length_sec": round((end - start + 1) / fps, 3)
    })

# =========================
# SADECE KEEP OLAN FRAME'LERİ JSON'A AKTAR
# =========================
gameplay_only_frames = []
for idx, frame_info in enumerate(frames_info):
    if idx < len(smoothed_keep_flags) and smoothed_keep_flags[idx]:
        gameplay_only_frames.append({
            "original_frame_index": int(frame_info["frame_index"]),
            "time_sec": frame_info["time_sec"],
            "objects": frame_info["objects"]
        })

gameplay_json = {
    "source_video_path": VIDEO_PATH,
    "source_tracking_json": INPUT_JSON_PATH,
    "fps": fps,
    "width": width,
    "height": height,
    "source_frame_count": frame_count,
    "kept_frame_count": len(gameplay_only_frames),
    "segments": segments,
    "frames": gameplay_only_frames
}

with open(GAMEPLAY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(gameplay_json, f, ensure_ascii=False, indent=2)

# =========================
# RAPOR JSON
# =========================
report = {
    "thresholds": {
        "min_green_ratio": MIN_GREEN_RATIO,
        "min_persons": MIN_PERSONS,
        "max_bbox_height_ratio": MAX_BBOX_HEIGHT_RATIO,
        "merge_gap_frames": MERGE_GAP_FRAMES
    },
    "summary": {
        "source_frame_count": frame_count,
        "kept_frame_count": len(gameplay_only_frames),
        "kept_ratio": round(len(gameplay_only_frames) / frame_count, 4) if frame_count > 0 else 0.0,
        "segment_count": len(segments)
    },
    "segments": segments,
    "frame_analysis": analysis_rows
}

with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# =========================
# SADECE KEEP OLAN FRAME'LERDEN VİDEO ÜRET
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video ikinci geçişte açılamadı.")

gameplay_writer = cv2.VideoWriter(GAMEPLAY_VIDEO_PATH, fourcc, fps, (width, height))
if not gameplay_writer.isOpened():
    raise RuntimeError("Gameplay video oluşturulamadı.")

frame_idx = 0
_kept2 = sum(smoothed_keep_flags)
print(f"\nStage 3 | Gameplay Filter — Pass 2/2 (export) — {frame_count} frame → {_kept2} tutulacak")
with tqdm(total=frame_count, desc="Stage 3 | Export  ", unit="frame", ncols=90) as _pbar2:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(smoothed_keep_flags) and smoothed_keep_flags[frame_idx]:
            gameplay_writer.write(frame)
        frame_idx += 1
        _pbar2.update(1)

cap.release()
gameplay_writer.release()

print("\nBİTTİ.")
print(f"Review video:   {REVIEW_VIDEO_PATH}")
print(f"Gameplay video: {GAMEPLAY_VIDEO_PATH}")
print(f"Gameplay JSON:  {GAMEPLAY_JSON_PATH}")
print(f"Report JSON:    {REPORT_JSON_PATH}")
print(f"Toplam frame:   {frame_count}")
print(f"Keep frame:     {len(gameplay_only_frames)}")
print(f"Segment sayısı: {len(segments)}")
