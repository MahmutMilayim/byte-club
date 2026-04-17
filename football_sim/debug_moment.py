"""
debug_moment.py
===============
Videonun belirli bir anini frame frame analiz eder.
Top ve oyuncu koordinatlarini, sut dedektorunun ne gordugunu gosterir.

Kullanim:
    python debug_moment.py --time 420    # 7. dakika = 420. saniye
    python debug_moment.py --time 420 --window 30  # 7. dak +/- 30 saniye
"""

import cv2
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils.calibration import Calibrator


def load_model(device):
    from ultralytics import YOLO
    path = cfg.DETECTOR_WEIGHTS
    print(f"Model yukleniyor: {path}")
    model = YOLO(path)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy, device=device, verbose=False, imgsz=cfg.IMGSZ)
    print("Hazir.\n")
    return model


def analyze_range(video_path, start_sec, end_sec, device):
    model = load_model(device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    calib = Calibrator(W, H)
    calib.initialize()

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"{'='*65}")
    print(f"  Analiz: {start_sec:.0f}s - {end_sec:.0f}s  ({end_frame-start_frame} frame)")
    print(f"  Saha kalibrasyonu: {'Manuel' if cfg.MANUAL_CALIB_POINTS else 'Otomatik'}")
    print(f"{'='*65}")
    print(f"  {'Sn':>6}  {'Frame':>6}  {'Top_px':>12}  {'Top_saha':>16}  {'Hiz_px/f':>9}  {'Durum'}")
    print(f"  {'-'*65}")

    prev_ball_px = None
    prev_frame   = None
    results_log  = []

    for fi in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # Her 3 frame'de bir isle
        if (fi - start_frame) % 3 != 0:
            continue

        t = fi / fps

        # Kalibrasyon guncelle
        if (fi - start_frame) % 90 == 0:
            calib.update(frame)

        # Tespit
        r = model.predict(
            source=frame, conf=0.2, verbose=False,
            device=device, imgsz=cfg.IMGSZ
        )

        ball_px  = None
        ball_fxy = None
        names    = r[0].names if r else {}

        if r and r[0].boxes is not None:
            for box in r[0].boxes:
                cls = names.get(int(box.cls[0].item()), "").lower()
                if "ball" in cls or cls == "sports ball":
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    cx = (x1+x2)/2
                    cy = (y1+y2)/2
                    ball_px  = (cx, cy)
                    ball_fxy = calib.pixel_to_field(cx, cy)
                    break

        # Hiz hesapla
        speed = 0.0
        if ball_px and prev_ball_px and prev_frame:
            dt = fi - prev_frame
            if dt > 0:
                speed = np.hypot(
                    ball_px[0]-prev_ball_px[0],
                    ball_px[1]-prev_ball_px[1]
                ) / dt

        # Gol bolgesi kontrolu
        goal_status = ""
        if ball_fxy:
            fx, fy = ball_fxy
            if fx <= cfg.GOAL_X_LEFT_MAX and cfg.GOAL_Y_MIN <= fy <= cfg.GOAL_Y_MAX:
                goal_status = "<<< SOL GOL BOLGESI!"
            elif fx >= cfg.GOAL_X_RIGHT_MIN and cfg.GOAL_Y_MIN <= fy <= cfg.GOAL_Y_MAX:
                goal_status = ">>> SAG GOL BOLGESI!"
            elif speed > cfg.SHOT_MIN_PIXEL_SPEED:
                goal_status = f"HIZLI (hiz={speed:.0f})"

        # Log
        ball_px_str  = f"({ball_px[0]:.0f},{ball_px[1]:.0f})" if ball_px else "YOK"
        ball_fxy_str = f"({ball_fxy[0]:.1f},{ball_fxy[1]:.1f})m" if ball_fxy else "YOK"
        speed_str    = f"{speed:.1f}" if speed > 0 else "-"

        line = f"  {t:>6.1f}s  {fi:>6}  {ball_px_str:>12}  {ball_fxy_str:>16}  {speed_str:>9}  {goal_status}"
        print(line)

        results_log.append({
            "time": round(t, 1),
            "frame": fi,
            "ball_px": list(ball_px) if ball_px else None,
            "ball_field": list(ball_fxy) if ball_fxy else None,
            "speed_px_per_frame": round(speed, 1),
            "goal_zone": bool(goal_status and "GOL" in goal_status),
        })

        if ball_px:
            prev_ball_px = ball_px
            prev_frame   = fi

    cap.release()

    # JSON kaydet
    out = Path("output/debug_moment.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results_log, indent=2), encoding="utf-8")

    print(f"\n{'='*65}")
    print(f"  Toplam frame analiz edildi : {len(results_log)}")
    print(f"  Top gorulen frame sayisi   : {sum(1 for r in results_log if r['ball_px'])}")
    print(f"  Gol bolgesi gorulen        : {sum(1 for r in results_log if r['goal_zone'])}")
    print(f"  Log kaydedildi             : {out}")
    print(f"{'='*65}\n")

    # Kalibrasyon kontrolu
    if cfg.MANUAL_CALIB_POINTS:
        print("  KALIBRASYON NOKTALARI:")
        for px, wd in zip(
            cfg.MANUAL_CALIB_POINTS["pixel"],
            cfg.MANUAL_CALIB_POINTS["world"]
        ):
            fx, fy = calib.pixel_to_field(px[0], px[1])
            print(f"    Piksel {px} -> Saha hesaplanan ({fx:.1f},{fy:.1f})m  |  Beklenen {wd}m")

    print("\n  GOL BOLGESI SINIRLARI (config.py):")
    print(f"    Sol kale : field_x <= {cfg.GOAL_X_LEFT_MAX}m  ve  {cfg.GOAL_Y_MIN} <= field_y <= {cfg.GOAL_Y_MAX}")
    print(f"    Sag kale : field_x >= {cfg.GOAL_X_RIGHT_MIN}m  ve  {cfg.GOAL_Y_MIN} <= field_y <= {cfg.GOAL_Y_MAX}")

    return results_log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--time",   type=float, required=True,
                   help="Analiz edilecek an (saniye). 7. dakika = 420")
    p.add_argument("--window", type=float, default=20,
                   help="Bu anin +/- kac saniyesini analiz et (default: 20)")
    p.add_argument("--device", default=cfg.DEVICE)
    args = p.parse_args()

    video = cfg.VIDEO_PATH
    if not Path(video).exists():
        print(f"Video bulunamadi: {video}")
        sys.exit(1)

    start = max(0, args.time - args.window)
    end   = args.time + args.window

    analyze_range(video, start, end, args.device)


if __name__ == "__main__":
    main()
