#!/usr/bin/env python3
"""
Stage 10 - High-precision shot input export via per-shot PnLCalib.

For each shot event:
  - search a small gameplay window around the shot anchor,
  - run PnLCalib on candidate frames,
  - score candidates using calibration quality + shooter/ball evidence,
  - project players and ball with the best per-shot homography,
  - export one JSON per shot plus an aggregate JSON.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ANALYTICS_JSON = "/output/stage8_analytics/analytics.json"
GAMEPLAY_JSON = "/output/stage3_filter/gameplay.json"
GAMEPLAY_VIDEO = "/output/stage3_filter/gameplay.mp4"
BALL_JSON = "/output/stage5_ball/ball_tracks.json"
TRACK_LABELS_JSON = "/output/stage4_clustering/track_labels_corrected.json"
OUT_JSON = "/output/stage10_setpiece_inputs/shot_inputs.json"
OUT_DEBUG_JSON = "/output/stage10_setpiece_inputs/shot_inputs_debug.json"
OUT_DEBUG_DIR = "/output/stage10_setpiece_inputs/shot_inputs_debug"

FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0
TEAM_ID_MAP = {"team_1": "HOME", "team_2": "AWAY"}
GOAL_MAP = {"left": "TOP", "right": "BOTTOM"}
CONTACT_AT_FOOT_PX = 45.0
CONTACT_STRONG_PX = 28.0


@dataclass
class CandidateResult:
    gameplay_frame_index: int
    raw_frame_index: int
    score: float
    rep_err: float | None
    ball_visible: bool
    shooter_visible: bool
    shooter_ball_image_dist: float | None
    same_team_ball_image_dist: float | None
    ball_pitch: tuple[float, float] | None
    shooter_pitch: tuple[float, float] | None
    infield_player_count: int
    debug_image_path: str
    players: list[dict]
    shooter: dict | None
    ball: dict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytics-json", default=ANALYTICS_JSON)
    ap.add_argument("--gameplay-json", default=GAMEPLAY_JSON)
    ap.add_argument("--gameplay-video", default=GAMEPLAY_VIDEO)
    ap.add_argument("--ball-json", default=BALL_JSON)
    ap.add_argument("--track-labels-json", default=TRACK_LABELS_JSON)
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-debug-json", default=OUT_DEBUG_JSON)
    ap.add_argument("--out-debug-dir", default=OUT_DEBUG_DIR)
    ap.add_argument("--search-radius", type=int, default=6)
    return ap.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_coord(value: float) -> float:
    return round(float(value), 3)


def normalize_homography(H):
    if H is None:
        return None
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return None
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def image_to_pitch_xy(H_img_to_pitch, x, y):
    H = normalize_homography(H_img_to_pitch)
    if H is None:
        return None
    pt = H @ np.array([float(x), float(y), 1.0], dtype=np.float64)
    if abs(pt[2]) < 1e-12:
        return None
    pt = pt / pt[2]
    if not np.isfinite(pt).all():
        return None
    return float(pt[0]), float(pt[1])


def calibrate_pitch_to_image_homography(P):
    P = np.asarray(P, dtype=np.float64)
    H = np.zeros((3, 3), dtype=np.float64)
    H[:, 0] = P[:, 0]
    H[:, 1] = P[:, 1]
    H[:, 2] = P[:, 3] - 52.5 * P[:, 0] - 34.0 * P[:, 1]
    return normalize_homography(H)


def to_friend_xy(pitch_xy):
    std_x = float(pitch_xy[0])
    std_y = float(pitch_xy[1])
    out_x = max(0.0, min(FIELD_WIDTH_M, std_y))
    out_y = max(0.0, min(FIELD_LENGTH_M, FIELD_LENGTH_M - std_x))
    return round_coord(out_x), round_coord(out_y)


def in_field(pitch_xy):
    if pitch_xy is None:
        return False
    x, y = pitch_xy
    return 0.0 <= x <= FIELD_LENGTH_M and 0.0 <= y <= FIELD_WIDTH_M


def dist(a, b):
    if a is None or b is None:
        return None
    return float(math.dist(a, b))


def import_calibration_module():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, "/work/PnLCalib")
    spec = importlib.util.spec_from_file_location(
        "stage6_calib",
        "/work/scripts/stage6_field/calibrate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_frame(cap, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def choose_team_player(objects: list[dict], labels_map: dict[int, str], team: str, track_id):
    team_objects = []
    exact = None
    for obj in objects:
        tid = int(obj["track_id"])
        label = labels_map.get(tid)
        if label != team:
            continue
        team_objects.append(obj)
        if track_id is not None and tid == int(track_id):
            exact = obj
    return exact, team_objects


def project_player(obj: dict, label: str, H_img_to_pitch):
    foot_xy = obj.get("foot_point_image_xy")
    if foot_xy is None:
        return None
    pitch_xy = image_to_pitch_xy(H_img_to_pitch, foot_xy[0], foot_xy[1])
    if not in_field(pitch_xy):
        return None
    fx, fy = to_friend_xy(pitch_xy)
    return {
        "track_id": int(obj["track_id"]),
        "label": label,
        "pitch_xy": pitch_xy,
        "friend_xy": (fx, fy),
        "foot_point_image_xy": [float(foot_xy[0]), float(foot_xy[1])],
    }


def project_ball(ball_meta: dict, H_img_to_pitch):
    image_xy = ball_meta.get("image_xy")
    visible = bool(ball_meta.get("visible")) and image_xy is not None
    if not visible:
        return {
            "visible": False,
            "pitch_xy": None,
            "friend_xy": None,
            "image_xy": None,
        }
    pitch_xy = image_to_pitch_xy(H_img_to_pitch, image_xy[0], image_xy[1])
    if not in_field(pitch_xy):
        return {
            "visible": False,
            "pitch_xy": None,
            "friend_xy": None,
            "image_xy": [float(image_xy[0]), float(image_xy[1])],
        }
    bx, by = to_friend_xy(pitch_xy)
    return {
        "visible": True,
        "pitch_xy": pitch_xy,
        "friend_xy": (bx, by),
        "image_xy": [float(image_xy[0]), float(image_xy[1])],
    }


def score_candidate(
    rep_err,
    frame_offset,
    ball_visible,
    shooter_visible,
    shooter_ball_image_dist,
    same_team_ball_image_dist,
    ball_pitch_dist_to_ref,
    infield_player_count,
):
    score = 0.0
    score += 8.0
    score -= min(float(rep_err or 30.0), 30.0) * 0.45
    score -= abs(frame_offset) * 0.35

    if ball_visible:
        score += 4.0
    if shooter_visible:
        score += 3.0

    if shooter_ball_image_dist is not None:
        if shooter_ball_image_dist <= CONTACT_AT_FOOT_PX:
            score += 14.0
            score += max(0.0, CONTACT_AT_FOOT_PX - shooter_ball_image_dist) / 2.5
        else:
            score += max(0.0, 120.0 - shooter_ball_image_dist) / 30.0
    elif same_team_ball_image_dist is not None:
        score += max(0.0, 110.0 - same_team_ball_image_dist) / 32.0
    else:
        score -= 4.0

    if ball_pitch_dist_to_ref is not None:
        score += max(0.0, 20.0 - ball_pitch_dist_to_ref) / 2.2

    score += min(infield_player_count, 18) * 0.12
    return float(score)


def build_target_goal(event: dict):
    raw_target = str(event.get("target_goal") or "").lower()
    if raw_target in GOAL_MAP:
        return GOAL_MAP[raw_target]
    raw_side = str(event.get("raw_side") or "").lower()
    if raw_side in GOAL_MAP:
        return GOAL_MAP[raw_side]
    return "TOP"


def main():
    args = parse_args()

    analytics = load_json(args.analytics_json)
    gameplay = load_json(args.gameplay_json)
    ball_tracks = load_json(args.ball_json)
    labels = load_json(args.track_labels_json)

    labels_map = {int(k): v.get("label") for k, v in labels.get("tracks", {}).items()}
    gameplay_frames = gameplay.get("frames", [])
    ball_frames = ball_tracks.get("frames", [])
    if len(gameplay_frames) != len(ball_frames):
        raise RuntimeError("gameplay and ball tracks are not aligned")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.out_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    per_entry_dir = out_json.parent / out_json.stem
    per_entry_dir.mkdir(parents=True, exist_ok=True)

    calib_mod = import_calibration_module()
    model, model_l = calib_mod.load_models()

    cap = cv2.VideoCapture(args.gameplay_video)
    if not cap.isOpened():
        raise RuntimeError(f"Gameplay video acilamadi: {args.gameplay_video}")

    shot_events = [
        event for event in analytics.get("events", [])
        if event.get("type") == "shot"
    ]
    shot_events.sort(key=lambda event: int(event["frame"]))

    aggregate_outputs = []
    debug_outputs = []

    for shot_index, event in enumerate(shot_events, start=1):
        anchor_idx = int(event["frame"])
        event_team = str(event.get("team") or "")
        event_player = event.get("player")
        event_ref_pitch = event.get("shot_xy") or event.get("event_xy")
        if event_ref_pitch is not None:
            event_ref_pitch = (float(event_ref_pitch[0]), float(event_ref_pitch[1]))

        candidates: list[CandidateResult] = []
        frame_lo = max(0, anchor_idx - args.search_radius)
        frame_hi = min(len(gameplay_frames) - 1, anchor_idx + args.search_radius)

        for cand_idx in range(frame_lo, frame_hi + 1):
            frame = read_frame(cap, cand_idx)
            if frame is None:
                continue

            raw_frame_index = int(gameplay_frames[cand_idx]["original_frame_index"])
            shot_dir = debug_dir / f"shot_{shot_index:03d}"
            shot_dir.mkdir(parents=True, exist_ok=True)
            img_path = shot_dir / f"candidate_gp_{cand_idx:06d}.png"
            cv2.imwrite(str(img_path), frame)

            calib_output, projected_img = calib_mod.run_one(img_path, model, model_l)
            if calib_output.get("final_params_dict") is None:
                continue

            P = calib_mod.projection_from_cam_params(calib_output["final_params_dict"])
            H_pitch_to_img = calibrate_pitch_to_image_homography(P)
            if H_pitch_to_img is None:
                continue
            try:
                H_img_to_pitch = normalize_homography(np.linalg.inv(H_pitch_to_img))
            except np.linalg.LinAlgError:
                continue
            if H_img_to_pitch is None:
                continue

            objects = gameplay_frames[cand_idx].get("objects", [])
            shooter_obj, team_objects = choose_team_player(objects, labels_map, event_team, event_player)
            projected_team_players = []
            projected_all_players = []
            for obj in objects:
                tid = int(obj["track_id"])
                label = labels_map.get(tid)
                if label not in TEAM_ID_MAP:
                    continue
                projected = project_player(obj, label, H_img_to_pitch)
                if projected is None:
                    continue
                projected_all_players.append(projected)
                if label == event_team:
                    projected_team_players.append(projected)

            ball_meta = ball_frames[cand_idx].get("ball", {})
            projected_ball = project_ball(ball_meta, H_img_to_pitch)

            shooter_projected = None
            shooter_visible = False
            if shooter_obj is not None:
                shooter_visible = True
                for projected in projected_team_players:
                    if projected["track_id"] == int(shooter_obj["track_id"]):
                        shooter_projected = projected
                        break

            if shooter_projected is None and projected_team_players:
                ball_pitch = projected_ball["pitch_xy"]
                if ball_pitch is not None:
                    shooter_projected = min(
                        projected_team_players,
                        key=lambda p: math.dist(p["pitch_xy"], ball_pitch),
                    )
                else:
                    shooter_projected = projected_team_players[0]

            shooter_ball_image_dist = None
            same_team_ball_image_dist = None
            if ball_meta.get("image_xy") is not None:
                ball_img_xy = (float(ball_meta["image_xy"][0]), float(ball_meta["image_xy"][1]))
                if shooter_obj is not None and shooter_obj.get("foot_point_image_xy") is not None:
                    shooter_ball_image_dist = dist(shooter_obj["foot_point_image_xy"], ball_img_xy)
                if team_objects:
                    same_team_ball_image_dist = min(
                        dist(obj.get("foot_point_image_xy"), ball_img_xy)
                        for obj in team_objects
                        if obj.get("foot_point_image_xy") is not None
                    )

            # Exact contact frame often has the ball blurred / momentarily unprojectable.
            # If the raw image says the ball is on the shooter's foot, pin the ball to the
            # shooter's projected foot so the exported setup reflects the actual strike moment.
            if (
                shooter_projected is not None
                and shooter_ball_image_dist is not None
                and shooter_ball_image_dist <= CONTACT_AT_FOOT_PX
                and not projected_ball["visible"]
            ):
                projected_ball = {
                    "visible": True,
                    "pitch_xy": shooter_projected["pitch_xy"],
                    "friend_xy": shooter_projected["friend_xy"],
                    "image_xy": ball_meta.get("image_xy"),
                    "estimated_from_shooter": True,
                }

            ball_pitch_dist_to_ref = dist(projected_ball["pitch_xy"], event_ref_pitch)

            score = score_candidate(
                rep_err=calib_output.get("rep_err"),
                frame_offset=(cand_idx - anchor_idx),
                ball_visible=projected_ball["visible"],
                shooter_visible=shooter_visible,
                shooter_ball_image_dist=shooter_ball_image_dist,
                same_team_ball_image_dist=same_team_ball_image_dist,
                ball_pitch_dist_to_ref=ball_pitch_dist_to_ref,
                infield_player_count=len(projected_all_players),
            )

            debug_img_path = shot_dir / f"candidate_gp_{cand_idx:06d}_projected.png"
            if projected_img is not None:
                cv2.imwrite(str(debug_img_path), projected_img)
            else:
                debug_img_path = img_path

            players_out = []
            shooter_track_id = int(shooter_projected["track_id"]) if shooter_projected is not None else None
            for projected in sorted(
                projected_all_players,
                key=lambda p: (TEAM_ID_MAP[p["label"]], p["track_id"]),
            ):
                if shooter_track_id is not None and projected["track_id"] == shooter_track_id:
                    continue
                px, py = projected["friend_xy"]
                players_out.append(
                    {
                        "id": int(projected["track_id"]),
                        "teamId": TEAM_ID_MAP[projected["label"]],
                        "x": px,
                        "y": py,
                    }
                )

            if shooter_projected is not None:
                sx, sy = shooter_projected["friend_xy"]
                shooter_out = {
                    "playerId": int(shooter_projected["track_id"]),
                    "teamId": TEAM_ID_MAP[shooter_projected["label"]],
                    "x": sx,
                    "y": sy,
                }
            else:
                shooter_out = None

            if projected_ball["visible"]:
                bx, by = projected_ball["friend_xy"]
                ball_out = {"visible": True, "x": bx, "y": by}
            else:
                ball_out = {"visible": False, "x": None, "y": None}

            candidates.append(
                CandidateResult(
                    gameplay_frame_index=cand_idx,
                    raw_frame_index=raw_frame_index,
                    score=score,
                    rep_err=calib_output.get("rep_err"),
                    ball_visible=projected_ball["visible"],
                    shooter_visible=shooter_visible,
                    shooter_ball_image_dist=shooter_ball_image_dist,
                    same_team_ball_image_dist=same_team_ball_image_dist,
                    ball_pitch=projected_ball["pitch_xy"],
                    shooter_pitch=shooter_projected["pitch_xy"] if shooter_projected is not None else None,
                    infield_player_count=len(projected_all_players),
                    debug_image_path=str(debug_img_path),
                    players=players_out,
                    shooter=shooter_out,
                    ball=ball_out,
                )
            )

        if not candidates:
            raise RuntimeError(f"No valid PnLCalib candidate for shot #{shot_index} (anchor gp {anchor_idx})")

        best = max(candidates, key=lambda cand: cand.score)

        output_item = {
            "frameIndex": int(best.raw_frame_index),
            "targetGoal": build_target_goal(event),
            "shooter": best.shooter,
            "players": best.players,
            "ball": best.ball,
        }
        aggregate_outputs.append(output_item)

        shot_path = per_entry_dir / f"shot_{shot_index:03d}.json"
        with open(shot_path, "w", encoding="utf-8") as f:
            json.dump(output_item, f, ensure_ascii=False, indent=2)

        debug_outputs.append(
            {
                "shot_index": shot_index,
                "anchor_gameplay_frame": anchor_idx,
                "anchor_raw_frame": int(gameplay_frames[anchor_idx]["original_frame_index"]),
                "selected_gameplay_frame": best.gameplay_frame_index,
                "selected_raw_frame": best.raw_frame_index,
                "selected_score": round(best.score, 4),
                "selected_rep_err": best.rep_err,
                "selected_ball_visible": best.ball_visible,
                "selected_shooter_visible": best.shooter_visible,
                "selected_shooter_ball_image_dist": round(best.shooter_ball_image_dist, 3)
                if best.shooter_ball_image_dist is not None
                else None,
                "selected_same_team_ball_image_dist": round(best.same_team_ball_image_dist, 3)
                if best.same_team_ball_image_dist is not None
                else None,
                "selected_infield_player_count": best.infield_player_count,
                "selected_debug_image_path": best.debug_image_path,
                "event": event,
                "candidates": [
                    {
                        "gameplay_frame_index": cand.gameplay_frame_index,
                        "raw_frame_index": cand.raw_frame_index,
                        "score": round(cand.score, 4),
                        "rep_err": cand.rep_err,
                        "ball_visible": cand.ball_visible,
                        "shooter_visible": cand.shooter_visible,
                        "shooter_ball_image_dist": round(cand.shooter_ball_image_dist, 3)
                        if cand.shooter_ball_image_dist is not None
                        else None,
                        "same_team_ball_image_dist": round(cand.same_team_ball_image_dist, 3)
                        if cand.same_team_ball_image_dist is not None
                        else None,
                        "ball_pitch": [round_coord(v) for v in cand.ball_pitch]
                        if cand.ball_pitch is not None
                        else None,
                        "shooter_pitch": [round_coord(v) for v in cand.shooter_pitch]
                        if cand.shooter_pitch is not None
                        else None,
                        "infield_player_count": cand.infield_player_count,
                        "debug_image_path": cand.debug_image_path,
                    }
                    for cand in sorted(candidates, key=lambda cand: cand.score, reverse=True)
                ],
            }
        )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_outputs, f, ensure_ascii=False, indent=2)

    debug_json = Path(args.out_debug_json)
    debug_json.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_json, "w", encoding="utf-8") as f:
        json.dump(debug_outputs, f, ensure_ascii=False, indent=2)

    cap.release()

    print("Stage 10 | Shot Input Export (PnLCalib)")
    print("  output =", out_json)
    print("  per_entry_dir =", per_entry_dir)
    print("  debug_json =", debug_json)
    print("  entries =", len(aggregate_outputs))


if __name__ == "__main__":
    main()
