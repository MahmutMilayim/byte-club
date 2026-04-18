import argparse
import json
import math
from collections import Counter
from pathlib import Path

from tqdm import tqdm

INPUT_JSON = "/output/stage6_field/projection.json"
EVENTS_JSON = "/output/stage8_analytics/team_spotting_raw.json"
OUT_JSON = "/output/stage7_possession/possession.json"

STRONG_PIXEL_CONTACT_PX = 24.0
MEDIUM_PIXEL_CONTACT_PX = 38.0
LOOSE_PIXEL_CONTACT_PX = 62.0

STRONG_PITCH_CONTACT_M = 1.35
MEDIUM_PITCH_CONTACT_M = 2.25
LOOSE_PITCH_CONTACT_M = 3.75

AIRBORNE_PIXEL_FAR_PX = 88.0
AIRBORNE_JUMP_M = 5.2
FAST_BALL_SPEED_MPS = 9.5

SAME_TEAM_CONFIRM_FRAMES = 2
OTHER_TEAM_CONFIRM_FRAMES = 3
MAX_AIRBORNE_HOLD_FRAMES = 14
MAX_OCCLUDED_HOLD_FRAMES = 10
MAX_LOOSE_HOLD_FRAMES = 4

MIN_STRONG_SCORE = 5.0
MIN_MEDIUM_SCORE = 2.9
CONTESTED_SCORE_DELTA = 0.6

DEAD_BALL_WINDOWS = {
    "OUT": (10, 24),
    "THROW IN": (12, 36),
    "FREE KICK": (14, 42),
    "GOAL": (18, 60),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", default=INPUT_JSON)
    ap.add_argument("--events-json", default=EVENTS_JSON)
    ap.add_argument("--out-json", default=OUT_JSON)
    return ap.parse_args()


def dist(a, b):
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def safe_xy(xy):
    if xy is None or len(xy) < 2:
        return None
    x, y = float(xy[0]), float(xy[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]


def ball_pitch_xy(ball):
    for key in ("pitch_xy_trusted", "pitch_xy_smoothed", "pitch_xy_raw", "pitch_xy"):
        xy = safe_xy(ball.get(key))
        if xy is not None:
            return xy, key
    return None, None


def load_dead_ball_mask(events_json_path, total_frames):
    mask = [False] * total_frames
    path = Path(events_json_path)
    if not path.is_file():
        return mask
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return mask

    for event in payload.get("events", []):
        label = str(event.get("label", "")).upper()
        if label not in DEAD_BALL_WINDOWS:
            continue
        frame = int(event.get("frame", 0))
        before, after = DEAD_BALL_WINDOWS[label]
        lo = max(0, frame - before)
        hi = min(total_frames, frame + after + 1)
        for idx in range(lo, hi):
            mask[idx] = True
    return mask


def candidate_score(player, ball, ball_xy, current_player_id, current_team_label):
    px_dist = dist(ball.get("image_xy"), player.get("foot_point_image_xy"))
    pitch_dist = dist(ball_xy, player.get("pitch_xy"))
    visible = bool(ball.get("visible", False))
    interpolated = bool(ball.get("interpolated", False))

    score = 0.0
    reasons = []

    if px_dist is not None:
        if px_dist <= STRONG_PIXEL_CONTACT_PX:
            score += 5.8
            reasons.append("px_strong")
        elif px_dist <= MEDIUM_PIXEL_CONTACT_PX:
            score += 4.0
            reasons.append("px_medium")
        elif px_dist <= LOOSE_PIXEL_CONTACT_PX:
            score += 1.2
            reasons.append("px_loose")
        else:
            score -= 2.4

    if pitch_dist is not None:
        if pitch_dist <= STRONG_PITCH_CONTACT_M:
            score += 4.4
            reasons.append("pitch_strong")
        elif pitch_dist <= MEDIUM_PITCH_CONTACT_M:
            score += 2.4
            reasons.append("pitch_medium")
        elif pitch_dist <= LOOSE_PITCH_CONTACT_M:
            score += 0.6
            reasons.append("pitch_loose")
        else:
            score -= 1.8

    if current_player_id is not None and int(player["track_id"]) == int(current_player_id):
        score += 0.65
        reasons.append("same_owner")
    if current_team_label and player["label"] == current_team_label:
        score += 0.18
        reasons.append("same_team")
    if not visible:
        score -= 0.35
    if interpolated:
        score -= 0.55

    return {
        "track_id": int(player["track_id"]),
        "team_label": player["label"],
        "score": round(float(score), 4),
        "pixel_dist_px": round(float(px_dist), 2) if px_dist is not None else None,
        "distance_m": round(float(pitch_dist), 3) if pitch_dist is not None else None,
        "reasons": reasons,
    }


def classify_ball_state(ball, best_candidate, second_candidate, prev_ball_xy):
    base_state = ball.get("ball_state") or "unknown"
    ball_xy, _ = ball_pitch_xy(ball)
    jump = dist(ball_xy, prev_ball_xy)
    nearest_px = ball.get("nearest_player_px")
    visible = bool(ball.get("visible", False))
    strong_contact = best_candidate is not None and (
        (best_candidate.get("pixel_dist_px") is not None and best_candidate["pixel_dist_px"] <= MEDIUM_PIXEL_CONTACT_PX)
        or (best_candidate.get("distance_m") is not None and best_candidate["distance_m"] <= MEDIUM_PITCH_CONTACT_M)
    )

    if ball.get("image_xy") is None:
        return "missing", jump

    if base_state in ("projection_outside", "weak_homography"):
        return base_state, jump

    if bool(ball.get("airborne_suspect", False)):
        if best_candidate and best_candidate.get("pixel_dist_px") is not None and best_candidate["pixel_dist_px"] <= STRONG_PIXEL_CONTACT_PX:
            return "controlled_ground", jump
        return "airborne", jump

    if visible and nearest_px is not None and float(nearest_px) > AIRBORNE_PIXEL_FAR_PX and not strong_contact:
        return "airborne", jump

    ball_speed = float(ball.get("_ball_speed_mps") or 0.0)
    if jump is not None and jump > AIRBORNE_JUMP_M and ball_speed >= FAST_BALL_SPEED_MPS and not strong_contact:
        return "airborne", jump

    if best_candidate is not None and best_candidate["score"] >= MIN_STRONG_SCORE:
        return "controlled_ground", jump

    if best_candidate is not None and best_candidate["score"] >= MIN_MEDIUM_SCORE:
        if second_candidate is not None and abs(best_candidate["score"] - second_candidate["score"]) <= CONTESTED_SCORE_DELTA:
            return "contested_ground", jump
        return "loose_ground", jump

    if visible and ball_xy is not None:
        return "loose_ground", jump

    return "free", jump


def should_switch_immediately(ball_state, candidate, current_player_id, current_team_label):
    if candidate is None:
        return False
    if current_player_id is not None and int(candidate["track_id"]) == int(current_player_id):
        return True
    px_dist = candidate.get("pixel_dist_px")
    pitch_dist = candidate.get("distance_m")
    if ball_state == "controlled_ground":
        if px_dist is not None and px_dist <= STRONG_PIXEL_CONTACT_PX:
            return True
        if pitch_dist is not None and pitch_dist <= STRONG_PITCH_CONTACT_M:
            return True
    if current_team_label and candidate["team_label"] == current_team_label and candidate["score"] >= (MIN_STRONG_SCORE + 0.7):
        return True
    return False


def main():
    args = parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data["frames"]
    dead_ball_mask = load_dead_ball_mask(args.events_json, len(frames))

    current_player_id = None
    current_team_label = None
    pending_player_id = None
    pending_team_label = None
    pending_count = 0
    owner_hold_frames = 0
    prev_ball_xy = None

    team_1_frames = 0
    team_2_frames = 0
    none_frames = 0
    contested_frames = 0
    state_counts = Counter()
    export_frames = []

    print(f"\nStage 7 | Possession - {len(frames)} frame")

    for frame_idx, fr in enumerate(
        tqdm(frames, desc="Stage 7 | Possession", unit="frame", ncols=90)
    ):
        players = [p for p in fr.get("players", []) if p.get("label") in ("team_1", "team_2")]
        ball = dict(fr.get("ball", {}))
        ball_xy, ball_xy_source = ball_pitch_xy(ball)
        ball["pitch_xy_selected"] = ball_xy
        ball["pitch_xy_selected_source"] = ball_xy_source

        if frame_idx > 0:
            prev_selected_xy = export_frames[-1]["ball"].get("pitch_xy_selected")
            step = dist(ball_xy, prev_selected_xy)
            ball["_ball_speed_mps"] = round(float(step or 0.0) * 25.0, 3)
        else:
            ball["_ball_speed_mps"] = 0.0

        candidates = []
        for player in players:
            cand = candidate_score(
                player,
                ball,
                ball_xy,
                current_player_id,
                current_team_label,
            )
            if cand["pixel_dist_px"] is None and cand["distance_m"] is None:
                continue
            candidates.append(cand)

        candidates.sort(
            key=lambda c: (
                c["score"],
                -(c["pixel_dist_px"] or 9999.0),
                -(c["distance_m"] or 9999.0),
            ),
            reverse=True,
        )
        best_candidate = candidates[0] if candidates else None
        second_candidate = candidates[1] if len(candidates) > 1 else None

        ball_play_state, ball_jump_m = classify_ball_state(
            ball, best_candidate, second_candidate, prev_ball_xy
        )
        if dead_ball_mask[frame_idx]:
            ball_play_state = "dead_ball"
        state_counts[ball_play_state] += 1

        assigned_now = False
        owner_mode = "none"
        owner_confidence = 0.0
        contested_possession = False
        transition = "none"

        if second_candidate is not None and best_candidate is not None:
            contested_possession = abs(best_candidate["score"] - second_candidate["score"]) <= CONTESTED_SCORE_DELTA

        if ball_play_state == "dead_ball":
            current_player_id = None
            current_team_label = None
            pending_player_id = None
            pending_team_label = None
            pending_count = 0
            owner_hold_frames = 0
            owner_mode = "dead_ball"

        elif ball_play_state == "controlled_ground" and best_candidate is not None:
            required_confirm = (
                SAME_TEAM_CONFIRM_FRAMES
                if current_team_label and best_candidate["team_label"] == current_team_label
                else OTHER_TEAM_CONFIRM_FRAMES
            )
            if should_switch_immediately(
                ball_play_state, best_candidate, current_player_id, current_team_label
            ):
                if current_player_id is None:
                    transition = "init_owner"
                elif int(best_candidate["track_id"]) != int(current_player_id):
                    transition = "switch_owner_fast"
                current_player_id = best_candidate["track_id"]
                current_team_label = best_candidate["team_label"]
                pending_player_id = None
                pending_team_label = None
                pending_count = 0
                owner_hold_frames = 0
                assigned_now = True
                owner_mode = "direct_control"
                owner_confidence = best_candidate["score"]
            else:
                if pending_player_id == best_candidate["track_id"]:
                    pending_count += 1
                else:
                    pending_player_id = best_candidate["track_id"]
                    pending_team_label = best_candidate["team_label"]
                    pending_count = 1

                if pending_count >= required_confirm:
                    if current_player_id is None:
                        transition = "confirmed_init"
                    elif int(best_candidate["track_id"]) != int(current_player_id):
                        transition = "confirmed_switch"
                    current_player_id = pending_player_id
                    current_team_label = pending_team_label
                    pending_player_id = None
                    pending_team_label = None
                    pending_count = 0
                    owner_hold_frames = 0
                    assigned_now = True
                    owner_mode = "confirmed_control"
                    owner_confidence = best_candidate["score"]
                else:
                    assigned_now = current_player_id is not None
                    owner_mode = "pending_control"
                    owner_confidence = best_candidate["score"]

        elif ball_play_state in ("airborne", "temporarily_occluded"):
            pending_player_id = None
            pending_team_label = None
            pending_count = 0
            if best_candidate is not None and best_candidate["score"] >= (MIN_STRONG_SCORE + 0.8):
                current_player_id = best_candidate["track_id"]
                current_team_label = best_candidate["team_label"]
                owner_hold_frames = 0
                assigned_now = True
                owner_mode = "air_receiver_override"
                owner_confidence = best_candidate["score"]
                transition = "switch_from_air"
            elif current_player_id is not None:
                owner_hold_frames += 1
                max_hold = MAX_AIRBORNE_HOLD_FRAMES if ball_play_state == "airborne" else MAX_OCCLUDED_HOLD_FRAMES
                if owner_hold_frames <= max_hold:
                    assigned_now = True
                    owner_mode = "airborne_hold" if ball_play_state == "airborne" else "occluded_hold"
                    owner_confidence = max(0.0, 2.2 - 0.14 * owner_hold_frames)
                else:
                    current_player_id = None
                    current_team_label = None
                    owner_mode = "release_after_air"
                    transition = "drop_after_air"
            else:
                owner_mode = "free_air"

        elif ball_play_state in ("loose_ground", "contested_ground", "free", "missing", "projection_outside", "weak_homography"):
            if best_candidate is not None and best_candidate["score"] >= (MIN_STRONG_SCORE + 0.5):
                current_player_id = best_candidate["track_id"]
                current_team_label = best_candidate["team_label"]
                owner_hold_frames = 0
                pending_player_id = None
                pending_team_label = None
                pending_count = 0
                assigned_now = True
                owner_mode = "recovered_control"
                owner_confidence = best_candidate["score"]
                transition = "recover_strong"
            elif current_player_id is not None:
                owner_hold_frames += 1
                if owner_hold_frames <= MAX_LOOSE_HOLD_FRAMES:
                    assigned_now = True
                    owner_mode = "loose_hold"
                    owner_confidence = max(0.0, 1.9 - 0.25 * owner_hold_frames)
                else:
                    current_player_id = None
                    current_team_label = None
                    owner_mode = "drop_loose"
                    transition = "drop_loose"
            else:
                owner_mode = "free_ground"
                pending_player_id = None
                pending_team_label = None
                pending_count = 0

        if current_team_label == "team_1":
            team_1_frames += 1
        elif current_team_label == "team_2":
            team_2_frames += 1
        else:
            none_frames += 1
        if contested_possession or ball_play_state == "contested_ground":
            contested_frames += 1

        export_frames.append(
            {
                "frame_index": frame_idx,
                "current_player_id": int(current_player_id) if current_player_id is not None else None,
                "current_team_label": current_team_label,
                "assigned": bool(assigned_now),
                "assignment_reason": owner_mode,
                "transition": transition,
                "owner_confidence": round(float(owner_confidence), 4),
                "ball_play_state": ball_play_state,
                "ball_jump_m": round(float(ball_jump_m), 3) if ball_jump_m is not None else None,
                "candidate": best_candidate,
                "candidate_secondary": second_candidate,
                "pending_player_id": int(pending_player_id) if pending_player_id is not None else None,
                "pending_team_label": pending_team_label,
                "pending_count": int(pending_count),
                "contested_possession": bool(contested_possession),
                "ball": ball,
            }
        )

        prev_ball_xy = ball_xy or prev_ball_xy

    total_frames = len(export_frames)
    out = {
        "source_projection_json": args.input_json,
        "source_events_json": args.events_json if Path(args.events_json).is_file() else None,
        "summary": {
            "total_frames": total_frames,
            "team_1_frames": team_1_frames,
            "team_2_frames": team_2_frames,
            "none_frames": none_frames,
            "contested_frames": contested_frames,
            "team_1_ratio": round(team_1_frames / total_frames, 4) if total_frames else 0.0,
            "team_2_ratio": round(team_2_frames / total_frames, 4) if total_frames else 0.0,
            "none_ratio": round(none_frames / total_frames, 4) if total_frames else 0.0,
            "contested_ratio": round(contested_frames / total_frames, 4) if total_frames else 0.0,
            "ball_state_counts": dict(state_counts),
            "dead_ball_frames": int(sum(1 for x in dead_ball_mask if x)),
        },
        "frames": export_frames,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("out =", args.out_json)
    print("summary =", out["summary"])


if __name__ == "__main__":
    main()
