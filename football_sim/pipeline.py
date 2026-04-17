"""
pipeline.py
===========
Top icin yolo26x, oyuncu icin yolov8x - ayri modeller
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config as cfg
from utils.calibration     import Calibrator
from utils.team_classifier import TeamClassifier
from utils.shot_detector   import ShotDetector, PlayerSnap, BallSnap, ShotEvent
from utils.highlight_maker import make_highlights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_models(device):
    from ultralytics import YOLO
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    log.info(f"Top modeli yukleniyor  : {cfg.BALL_DETECTOR_WEIGHTS}")
    ball_model = YOLO(cfg.BALL_DETECTOR_WEIGHTS)
    ball_model.predict(source=dummy, device=device, verbose=False, imgsz=cfg.IMGSZ)

    log.info(f"Oyuncu modeli yukleniyor: {cfg.PLAYER_DETECTOR_WEIGHTS}")
    player_model = YOLO(cfg.PLAYER_DETECTOR_WEIGHTS)
    player_model.predict(source=dummy, device=device, verbose=False, imgsz=cfg.IMGSZ)

    log.info("   Her iki model hazir.\n")
    return ball_model, player_model


def detect_ball(model, frame, device):
    """Sadece top - dusuk conf, yolo26x."""
    r = model.predict(
        source=frame, conf=0.15, verbose=False,
        device=device, imgsz=cfg.IMGSZ,
    )
    if not r or r[0].boxes is None:
        return None
    names = r[0].names
    best, best_c = None, 0.0
    for box in r[0].boxes:
        cls = names.get(int(box.cls[0].item()), "").lower()
        if "ball" not in cls and cls != "sports ball":
            continue
        c = float(box.conf[0].item())
        if c > best_c:
            best_c = c
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            best = {"px": (x1+x2)/2, "py": (y1+y2)/2}
    return best


def detect_players(model, frame, device):
    """Sadece oyuncular - yolov8x."""
    r = model.predict(
        source=frame, conf=cfg.DETECTOR_CONF, verbose=False,
        device=device, imgsz=cfg.IMGSZ,
    )
    if not r or r[0].boxes is None:
        return []
    names   = r[0].names
    players = []
    for i, box in enumerate(r[0].boxes):
        cls = names.get(int(box.cls[0].item()), "").lower()
        if cls not in ("player","goalkeeper","oyuncu","kaleci","person"):
            continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        players.append({
            "tid": i,
            "px": (x1+x2)/2, "py": y2,
            "x1":x1,"y1":y1,"x2":x2,"y2":y2,
        })
    return players


def build_entry(ev: ShotEvent) -> dict:
    return {
        "frameIndex":  ev.frame_idx,
        "timeSeconds": ev.time_sec,
        "isGoal":      ev.is_goal,
        "shooter": {
            "playerId":   ev.shooter_id,
            "teamId":     "HOME" if ev.shooter_team == 0 else "AWAY",
            "targetGoal": ev.target_goal,
        },
        "players": [
            {"id": p.track_id,
             "teamId": "HOME" if p.team == 0 else "AWAY",
             "x": round(p.field_x, 2),
             "y": round(p.field_y, 2)}
            for p in ev.players if p.track_id >= 0
        ],
        "ball": {
            "visible": ev.ball.visible,
            "x": round(ev.ball.field_x, 2) if ev.ball.visible else None,
            "y": round(ev.ball.field_y, 2) if ev.ball.visible else None,
        },
    }


def run(args):
    ball_model, player_model = load_models(args.device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    skip  = int(cfg.SKIP_FIRST_SECONDS * fps)
    end   = total
    if args.end_min:
        end = min(total, skip + int(args.end_min * 60 * fps))

    n_frames = end - skip
    log.info(f"Video: {args.video}  {W}x{H} @ {fps:.0f}fps")
    log.info(f"Islenecek: {n_frames/fps/60:.0f} dk  FRAME_SKIP={cfg.FRAME_SKIP}")

    calib      = Calibrator(W, H)
    calib.initialize()
    team_clf   = TeamClassifier()
    shot_det   = ShotDetector(fps, W, H)

    all_events: list[ShotEvent] = []
    frame_idx  = skip
    n_proc     = 0
    team_seen  = 0
    log_every  = max(50, n_frames // 50)
    t0         = time.time()

    cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
    log.info(f"Ilk {cfg.SKIP_FIRST_SECONDS:.0f}s atlandi\n")
    log.info("Isleme basliyor...\n")

    try:
        while frame_idx < end:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - skip) % cfg.FRAME_SKIP != 0:
                frame_idx += 1
                continue

            n_proc += 1

            # Kalibrasyon
            if (frame_idx - skip) % (cfg.CALIB_EVERY_N_FRAMES * cfg.FRAME_SKIP) == 0:
                calib.update(frame)

            # ── TOP TESPIT (yolo26x, her frame) ──────────────────────────
            ball_det  = detect_ball(ball_model, frame, args.device)
            ball_snap = None
            if ball_det:
                fx, fy = calib.pixel_to_field(ball_det["px"], ball_det["py"])
                ball_snap = BallSnap(
                    visible=True, field_x=fx, field_y=fy,
                    px=ball_det["px"], py=ball_det["py"],
                )

            # ── OYUNCU TESPIT (yolov8x, sadece gerektiginde) ─────────────
            player_snaps: list[PlayerSnap] = []
            need_players = (
                shot_det.is_shot_candidate() or
                (team_seen < cfg.TEAM_SAMPLE_FRAMES and n_proc % 5 == 0)
            )

            if need_players:
                p_dets = detect_players(player_model, frame, args.device)

                if not team_clf.is_ready:
                    for p in p_dets:
                        team_clf.collect(frame, p["x1"],p["y1"],p["x2"],p["y2"])
                    team_seen += 1
                    if team_seen >= cfg.TEAM_SAMPLE_FRAMES:
                        team_clf.try_fit()

                for p in p_dets:
                    team = team_clf.predict(
                        frame, p["x1"],p["y1"],p["x2"],p["y2"]
                    ) if team_clf.is_ready else 0
                    fx, fy = calib.pixel_to_field(p["px"], p["py"])
                    player_snaps.append(PlayerSnap(
                        track_id=p["tid"], team=team,
                        field_x=fx, field_y=fy,
                        px=p["px"], py=p["py"],
                    ))

            # ── SUT TESPITI ───────────────────────────────────────────────
            ev = shot_det.update(frame_idx, player_snaps, ball_snap)
            if ev is not None:
                all_events.append(ev)

            # ── LOG ───────────────────────────────────────────────────────
            if n_proc % log_every == 0:
                el  = time.time() - t0
                spd = n_proc / max(el, 1)
                rem = (end - frame_idx) / cfg.FRAME_SKIP / max(spd, 1) / 60
                pct = (frame_idx - skip) / max(n_frames, 1) * 100
                g   = sum(1 for e in all_events if e.is_goal)
                s   = len(all_events) - g
                log.info(
                    f"  %{pct:4.1f}  {spd:.1f}f/s  "
                    f"kalan~{rem:.0f}dk  sut={s}  gol={g}"
                )

            frame_idx += 1

    except KeyboardInterrupt:
        log.warning(f"Durduruldu kare={frame_idx}")

    cap.release()
    el    = time.time() - t0
    goals = sum(1 for e in all_events if e.is_goal)
    shots = len(all_events) - goals
    log.info(f"\nBitti | {n_proc} frame | {el/60:.1f} dk | Sut={shots} Gol={goals}")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim_ev   = [e for e in all_events if not e.is_goal or args.include_goals]
    sim_json = [build_entry(e) for e in sim_ev]
    all_json = [build_entry(e) for e in all_events]

    sim_path = args.out or str(cfg.OUTPUT_DIR / "simulation_shots.json")
    all_path = str(cfg.OUTPUT_DIR / "all_events.json")

    Path(sim_path).write_text(
        json.dumps(sim_json, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(all_path).write_text(
        json.dumps(all_json, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"Simulasyon: {sim_path}  ({len(sim_json)} sut)")
    log.info(f"Tum olaylar: {all_path}  ({len(all_json)} olay)")

    if not args.no_video and all_json:
        try:
            vp = args.out_video or str(cfg.OUTPUT_DIR / "highlights.mp4")
            make_highlights(args.video, all_json, vp)
        except Exception as e:
            log.error(f"Highlight hatasi: {e}")

    return sim_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",         default=None)
    p.add_argument("--out",           default=None)
    p.add_argument("--out-video",     default=None)
    p.add_argument("--device",        default=cfg.DEVICE)
    p.add_argument("--end-min",       type=float, default=None)
    p.add_argument("--include-goals", action="store_true")
    p.add_argument("--no-video",      action="store_true")
    args = p.parse_args()

    if args.video is None:
        args.video = cfg.VIDEO_PATH
    if not Path(args.video).exists():
        log.error(f"Video bulunamadi: {args.video}")
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
