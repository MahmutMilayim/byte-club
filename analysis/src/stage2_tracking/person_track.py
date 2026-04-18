import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# =========================
# AYARLAR
# =========================
VIDEO_PATH = r"/input/input.mp4"
MODEL_PATH = "/work/models/yolo26x.pt"
OUTPUT_VIDEO_PATH = r"/output/stage2_tracking/tracking.mp4"
OUTPUT_JSON_PATH = r"/output/stage2_tracking/tracking.json"

CONF_THRESHOLD = 0.18
IOU_THRESHOLD = 0.45
IMGSZ = 1280
LINE_THICKNESS = 2
FONT_SCALE = 0.6
VERBOSE = True

# =========================
# DOSYA KONTROLLERİ
# =========================
if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(f"Girdi videosu bulunamadı: {VIDEO_PATH}")

Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)

# =========================
# MODEL YÜKLE
# =========================
from pathlib import Path as _PathModelGuard
if not _PathModelGuard(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model bulunamadi: {MODEL_PATH}. Otomatik indirme kapali.")

model = YOLO(MODEL_PATH)

# =========================
# VİDEO BİLGİSİ
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video açılamadı.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

if fps <= 0:
    raise RuntimeError("FPS okunamadı.")

# MP4 çıktı için codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
if not writer.isOpened():
    raise RuntimeError("Çıktı videosu oluşturulamadı.")

# =========================
# JSON YAPISI
# =========================
export_data = {
    "video_path": VIDEO_PATH,
    "model_path": MODEL_PATH,
    "fps": fps,
    "width": width,
    "height": height,
    "frame_count": frame_count,
    "frames": []
}

# =========================
# TRACKING
# Sadece person class = 0
# =========================
results = model.track(
    source=VIDEO_PATH,
    tracker="/work/scripts/stage2_tracking/botsort_reid.yaml",
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    imgsz=IMGSZ,
    classes=[0],          # sadece person
    persist=True,
    stream=True,
    verbose=False,
)

frame_index = 0

print(f"\nStage 2 | Person Tracking — toplam {frame_count} frame")
for result in tqdm(results, total=frame_count, desc="Stage 2 | Person Tracking",
                   unit="frame", ncols=90, dynamic_ncols=False):
    frame = result.orig_img.copy()

    frame_info = {
        "frame_index": frame_index,
        "time_sec": round(frame_index / fps, 3),
        "objects": []
    }

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.array([-1] * len(boxes_xyxy), dtype=int)

        for box, conf, track_id in zip(boxes_xyxy, confs, track_ids):
            x1, y1, x2, y2 = box.tolist()

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            area = bbox_w * bbox_h

            # Gerçek ayak noktası: bbox alt kenarı değil, %88'i
            # (alt padding'i ekarte eder, projeksiyon hatasını ~2-3m azaltır)
            foot_x = (x1 + x2) / 2.0
            foot_y = y1 + (y2 - y1) * 0.88

            keep = True

            # Aşırı küçük kutuları ele
            if bbox_h < 14 or bbox_w < 6:
                keep = False

            # Görüntünün üst tarafındaki küçük kalabalıkları ele
            if foot_y < height * 0.30 and area < 1800:
                keep = False

            # Köşe tribünlerinde üst bölgede kalan küçük adayları ele
            if foot_y < height * 0.35 and (foot_x < width * 0.08 or foot_x > width * 0.92) and area < 2500:
                keep = False

            if not keep:
                continue

            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

            obj = {
                "track_id": int(track_id),
                "class_name": "person",
                "confidence": round(float(conf), 4),
                "bbox_xyxy": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2)
                ],
                "foot_point_image_xy": [
                    round(float(foot_x), 2),
                    round(float(foot_y), 2)
                ]
            }
            frame_info["objects"].append(obj)

            # Çizim
            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), LINE_THICKNESS)

            label = f"ID {track_id} | person | {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)

            text_bg_y1 = max(0, y1_i - th - 10)
            text_bg_y2 = max(0, y1_i)
            text_bg_x2 = min(width, x1_i + tw + 10)

            cv2.rectangle(frame, (x1_i, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 255, 0), -1)
            cv2.putText(
                frame,
                label,
                (x1_i + 5, max(15, y1_i - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            # Ayak noktası
            cv2.circle(frame, (int(foot_x), int(foot_y)), 4, (0, 0, 255), -1)

    # Sol üst köşeye frame bilgisi
    summary_text = f"Frame: {frame_index} | Time: {frame_index / fps:.2f}s | Persons: {len(frame_info['objects'])}"
    cv2.putText(
        frame,
        summary_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    export_data["frames"].append(frame_info)
    writer.write(frame)

    frame_index += 1

writer.release()

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(export_data, f, ensure_ascii=False, indent=2)

print("\nBİTTİ.")
print(f"Video çıktısı: {OUTPUT_VIDEO_PATH}")
print(f"JSON çıktısı:  {OUTPUT_JSON_PATH}")
