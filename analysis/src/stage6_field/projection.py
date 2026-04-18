import argparse
import json
import math
import shutil
from pathlib import Path

from tqdm import tqdm

from field_utils import FIELD_LENGTH_M, FIELD_WIDTH_M, image_to_pitch_xy


GAMEPLAY_JSON_PATH = "/output/stage3_filter/gameplay.json"
TRACK_LABELS_JSON_PATH = "/output/stage4_clustering/track_labels_corrected.json"
BALL_JSON_PATH = "/output/stage5_ball/ball_tracks.json"
REFINED_HMAP_PATH = "/output/stage6_field/homography_map_refined.json"
BASE_HMAP_PATH = "/output/stage6_field/homography_map.json"
OUT_DIR = "/output/stage6_field"
OUT_JSON_NAME = "projection.json"
LEGACY_ALIAS_NAME = "projection_smoothed.json"

PLAYER_LOOSE_MARGIN_M = 3.0
PLAYER_CLAMP_MARGIN_M = 3.0
PLAYER_HOLD_FRAMES = 6
PLAYER_STEP_OK_M = 1.6
PLAYER_STEP_WARN_M = 4.5
PLAYER_STEP_DROP_M = 9.0
PLAYER_ALPHA_STABLE = 0.58
PLAYER_ALPHA_WARN = 0.28

REF_STEP_OK_M = 2.2
REF_ALPHA = 0.42

BALL_LOOSE_MARGIN_M = 4.0
BALL_TRUSTED_PLAYER_PX = 70.0
BALL_SUSPECT_PLAYER_PX = 105.0
BALL_HOLD_FRAMES = 8
BALL_GROUND_STEP_OK_M = 3.4
BALL_GROUND_STEP_WARN_M = 7.0
BALL_ALPHA_STABLE = 0.52
BALL_ALPHA_WARN = 0.24
BALL_TRACK_QUALITY_MIN = 0.22


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gameplay-json-path", default=GAMEPLAY_JSON_PATH)
    ap.add_argument("--track-labels-json-path", default=TRACK_LABELS_JSON_PATH)
    ap.add_argument("--ball-json-path", default=BALL_JSON_PATH)
    ap.add_argument("--hmap-path", default="")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--out-name", default=OUT_JSON_NAME)
    ap.add_argument("--legacy-alias-name", default=LEGACY_ALIAS_NAME)
    return ap.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dist(a, b):
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist_px(a, b):
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def inside_pitch(xy, margin=0.0):
    if xy is None:
        return False
    return -margin <= xy[0] <= FIELD_LENGTH_M + margin and -margin <= xy[1] <= FIELD_WIDTH_M + margin


def clamp_if_slight(xy, margin):
    if xy is None:
        return None
    x, y = float(xy[0]), float(xy[1])
    if x < -margin or x > FIELD_LENGTH_M + margin or y < -margin or y > FIELD_WIDTH_M + margin:
        return None
    return [min(max(0.0, x), FIELD_LENGTH_M), min(max(0.0, y), FIELD_WIDTH_M)]


def ema(prev_xy, cur_xy, alpha):
    if prev_xy is None:
        return cur_xy
    if cur_xy is None:
        return prev_xy
    return [
        (1.0 - alpha) * prev_xy[0] + alpha * cur_xy[0],
        (1.0 - alpha) * prev_xy[1] + alpha * cur_xy[1],
    ]


def stabilise_player(raw_xy, prev_state, projection_quality, step_ok, step_warn, alpha_stable, alpha_warn):
    reason = "missing_raw"
    trusted = False

    raw_clamped = clamp_if_slight(raw_xy, PLAYER_CLAMP_MARGIN_M)
    raw_loose = inside_pitch(raw_xy, PLAYER_LOOSE_MARGIN_M)

    if prev_state is None:
        if raw_clamped is not None:
            return raw_clamped, {"last_xy": raw_clamped, "hold_count": 0}, True, "init_valid"
        return None, {"last_xy": None, "hold_count": 1}, False, reason

    prev_xy = prev_state.get("last_xy")
    hold_count = int(prev_state.get("hold_count", 0))
    if prev_xy is None:
        if raw_clamped is not None:
            return raw_clamped, {"last_xy": raw_clamped, "hold_count": 0}, False, "recover_from_missing"
        return None, {"last_xy": None, "hold_count": hold_count + 1}, False, reason

    if raw_loose and raw_xy is not None and prev_xy is not None:
        target_xy = raw_clamped or raw_xy
        step = dist(prev_xy, raw_xy) or 0.0
        if step <= step_ok:
            filtered = ema(prev_xy, target_xy, alpha_stable)
            trusted = True
            reason = "stable_step"
        elif step <= step_warn:
            filtered = ema(prev_xy, target_xy, alpha_warn)
            trusted = projection_quality >= BALL_TRACK_QUALITY_MIN
            reason = "warning_step"
        elif step <= PLAYER_STEP_DROP_M and projection_quality >= 0.30 and raw_clamped is not None:
            filtered = ema(prev_xy, raw_clamped, 0.16)
            trusted = False
            reason = "large_step_but_kept"
        else:
            filtered = prev_xy if hold_count < PLAYER_HOLD_FRAMES else None
            reason = "reject_large_step"
    elif raw_clamped is not None and prev_xy is not None:
        filtered = ema(prev_xy, raw_clamped, 0.20)
        trusted = False
        reason = "slight_outside_clamped"
    else:
        filtered = prev_xy if hold_count < PLAYER_HOLD_FRAMES else None
        reason = "hold_previous"

    new_state = {
        "last_xy": filtered,
        "hold_count": 0 if filtered is not None and filtered != prev_xy else hold_count + 1,
    }
    return filtered, new_state, trusted, reason


def ball_gate(raw_xy, ball_info, filtered_players, projection_quality, last_trusted_xy, last_smoothed_xy):
    ball_img_xy = ball_info.get("image_xy")
    visible = bool(ball_info.get("visible"))
    interpolated = bool(ball_info.get("interpolated"))
    confidence = float(ball_info.get("confidence") or 0.0)

    nearest_player_px = None
    nearest_player_track_id = None
    nearest_player_pitch_dist = None
    for p in filtered_players:
        foot_px = p.get("foot_point_image_xy")
        if foot_px is None:
            continue
        d_px = dist_px(ball_img_xy, foot_px)
        if d_px is not None and (nearest_player_px is None or d_px < nearest_player_px):
            nearest_player_px = d_px
            nearest_player_track_id = p["track_id"]
        pxy = p.get("pitch_xy")
        d_pitch = dist(raw_xy, pxy)
        if d_pitch is not None and (nearest_player_pitch_dist is None or d_pitch < nearest_player_pitch_dist):
            nearest_player_pitch_dist = d_pitch

    raw_loose = inside_pitch(raw_xy, BALL_LOOSE_MARGIN_M)
    raw_clamped = clamp_if_slight(raw_xy, BALL_LOOSE_MARGIN_M)
    jump_from_trusted = dist(last_trusted_xy, raw_xy)

    trusted_ground = False
    airborne_suspect = False
    gate_reason = "ok"
    state = "ground_trusted"

    if ball_img_xy is None:
        state = "missing"
        gate_reason = "no_image_xy"
    elif not raw_loose:
        state = "projection_outside"
        gate_reason = "projection_outside"
        airborne_suspect = True
    elif projection_quality < BALL_TRACK_QUALITY_MIN:
        state = "weak_homography"
        gate_reason = "weak_homography"
    elif interpolated and nearest_player_px is not None and nearest_player_px > BALL_TRUSTED_PLAYER_PX:
        state = "temporarily_occluded"
        gate_reason = "interpolated_without_near_player"
    else:
        close_player = nearest_player_px is not None and nearest_player_px <= BALL_TRUSTED_PLAYER_PX
        pitch_consistent = nearest_player_pitch_dist is not None and nearest_player_pitch_dist <= 2.8
        stable_motion = jump_from_trusted is None or jump_from_trusted <= BALL_GROUND_STEP_OK_M

        if close_player or pitch_consistent or (confidence >= 0.72 and stable_motion):
            trusted_ground = True
            gate_reason = "ok"
        else:
            state = "ground_visible_untrusted"
            gate_reason = "far_from_players"
            airborne_suspect = True

    if trusted_ground and jump_from_trusted is not None and jump_from_trusted > BALL_GROUND_STEP_WARN_M:
        trusted_ground = False
        airborne_suspect = True
        state = "airborne_suspect"
        gate_reason = "jump_too_large"

    if nearest_player_px is not None and nearest_player_px > BALL_SUSPECT_PLAYER_PX and visible and not interpolated:
        airborne_suspect = True
        if not trusted_ground:
            state = "airborne_suspect"
            gate_reason = "visible_far_from_players"

    return {
        "raw_clamped": raw_clamped,
        "trusted_ground": trusted_ground,
        "airborne_suspect": airborne_suspect,
        "gate_reason": gate_reason,
        "state": state,
        "nearest_player_px": nearest_player_px,
        "nearest_player_track_id": nearest_player_track_id,
        "nearest_player_pitch_dist": nearest_player_pitch_dist,
        "jump_from_trusted": jump_from_trusted,
    }


def select_hmap_path(cli_value):
    if cli_value:
        return cli_value
    if Path(REFINED_HMAP_PATH).is_file():
        return REFINED_HMAP_PATH
    return BASE_HMAP_PATH


def main():
    args = parse_args()

    hmap_path = select_hmap_path(args.hmap_path)
    gameplay = load_json(args.gameplay_json_path)
    track_labels = load_json(args.track_labels_json_path)
    ball_tracks = load_json(args.ball_json_path)
    hmap = load_json(hmap_path)

    id_to_label = {int(k): v["label"] for k, v in track_labels["tracks"].items()}
    gameplay_frames = gameplay["frames"]
    ball_frames = ball_tracks["frames"]
    hmap_frames = hmap["frames"]

    player_states = {}
    referee_states = {}
    last_ball_trusted_xy = None
    last_ball_smoothed_xy = None
    ball_hold_count = 0

    frames_out = []
    print(f"\nStage 6 | Project Tracks - {len(hmap_frames)} frames")
    for hframe in tqdm(hmap_frames, desc="Stage 6 | Project", unit="frame", ncols=90):
        frame_index = int(hframe["frame_index"])
        gframe = gameplay_frames[frame_index]
        bframe = ball_frames[frame_index]
        H = hframe.get("H_img_to_pitch")
        quality = float(hframe.get("final_score") or 0.0)

        players_out = []
        referees_out = []
        filtered_players = []

        for obj in gframe["objects"]:
            tid = int(obj["track_id"])
            label = id_to_label.get(tid, "unknown")
            fx, fy = obj["foot_point_image_xy"]
            raw_xy = image_to_pitch_xy(H, fx, fy)

            if label in ("team_1", "team_2"):
                prev_state = player_states.get(tid)
                filtered_xy, new_state, trusted, reason = stabilise_player(
                    raw_xy,
                    prev_state,
                    quality,
                    PLAYER_STEP_OK_M,
                    PLAYER_STEP_WARN_M,
                    PLAYER_ALPHA_STABLE,
                    PLAYER_ALPHA_WARN,
                )
                player_states[tid] = new_state
                payload = {
                    "track_id": tid,
                    "label": label,
                    "bbox_xyxy": obj["bbox_xyxy"],
                    "foot_point_image_xy": obj["foot_point_image_xy"],
                    "pitch_xy_raw": raw_xy,
                    "pitch_xy": filtered_xy,
                    "projection_trusted": trusted,
                    "projection_reason": reason,
                }
                players_out.append(payload)
                filtered_players.append(payload)
            elif label == "referee":
                prev_state = referee_states.get(tid)
                filtered_xy, new_state, trusted, reason = stabilise_player(
                    raw_xy,
                    prev_state,
                    quality,
                    REF_STEP_OK_M,
                    PLAYER_STEP_WARN_M,
                    REF_ALPHA,
                    0.24,
                )
                referee_states[tid] = new_state
                referees_out.append({
                    "track_id": tid,
                    "label": label,
                    "bbox_xyxy": obj["bbox_xyxy"],
                    "foot_point_image_xy": obj["foot_point_image_xy"],
                    "pitch_xy_raw": raw_xy,
                    "pitch_xy": filtered_xy,
                    "projection_trusted": trusted,
                    "projection_reason": reason,
                })

        ball_info = bframe.get("ball", bframe)
        raw_ball_xy = None
        if ball_info.get("image_xy") is not None:
            bx, by = ball_info["image_xy"]
            raw_ball_xy = image_to_pitch_xy(H, bx, by)

        gate = ball_gate(raw_ball_xy, ball_info, filtered_players, quality, last_ball_trusted_xy, last_ball_smoothed_xy)
        ball_pitch_smoothed = last_ball_smoothed_xy
        ball_pitch_trusted = None
        ball_ground_trusted = False
        ball_possession_usable = False

        if gate["trusted_ground"] and gate["raw_clamped"] is not None:
            alpha = BALL_ALPHA_STABLE
            if gate["jump_from_trusted"] is not None and gate["jump_from_trusted"] > BALL_GROUND_STEP_OK_M:
                alpha = BALL_ALPHA_WARN
            ball_pitch_smoothed = ema(last_ball_smoothed_xy, gate["raw_clamped"], alpha)
            ball_pitch_trusted = ball_pitch_smoothed
            ball_ground_trusted = True
            ball_possession_usable = True
            last_ball_trusted_xy = ball_pitch_trusted
            last_ball_smoothed_xy = ball_pitch_smoothed
            ball_hold_count = 0
        else:
            if gate["raw_clamped"] is not None and quality >= BALL_TRACK_QUALITY_MIN:
                ball_pitch_smoothed = ema(last_ball_smoothed_xy, gate["raw_clamped"], 0.20)
                last_ball_smoothed_xy = ball_pitch_smoothed
                ball_hold_count = 0
            elif last_ball_smoothed_xy is not None and ball_hold_count < BALL_HOLD_FRAMES:
                ball_pitch_smoothed = last_ball_smoothed_xy
                ball_hold_count += 1
            else:
                ball_pitch_smoothed = None
                last_ball_smoothed_xy = None
                ball_hold_count = BALL_HOLD_FRAMES

        frames_out.append({
            "frame_index": frame_index,
            "source_keyframe_index": hframe.get("source_keyframe_index"),
            "source_original_frame_index": hframe.get("source_original_frame_index"),
            "homography_quality_score": quality,
            "homography_selection_source": hframe.get("selection_source"),
            "players": players_out,
            "referees": referees_out,
            "ball": {
                "image_xy": ball_info.get("image_xy"),
                "pitch_xy_raw": raw_ball_xy,
                "pitch_xy_smoothed": ball_pitch_smoothed,
                "pitch_xy_trusted": ball_pitch_trusted,
                "pitch_xy": ball_pitch_smoothed,
                "visible": ball_info.get("visible"),
                "interpolated": ball_info.get("interpolated"),
                "ball_confidence": float(ball_info.get("confidence") or 0.0),
                "ball_ground_trusted": ball_ground_trusted,
                "ball_possession_usable": ball_possession_usable,
                "airborne_suspect": gate["airborne_suspect"],
                "ball_possession_gate_reason": gate["gate_reason"],
                "ball_state": gate["state"],
                "nearest_player_px": gate["nearest_player_px"],
                "nearest_player_track_id": gate["nearest_player_track_id"],
                "nearest_player_pitch_dist_m": gate["nearest_player_pitch_dist"],
            },
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "field_length_m": FIELD_LENGTH_M,
            "field_width_m": FIELD_WIDTH_M,
            "source_hmap_path": hmap_path,
            "total_frames": len(frames_out),
            "frames": frames_out,
        }, f, ensure_ascii=False, indent=2)

    legacy_alias_path = out_dir / args.legacy_alias_name
    shutil.copyfile(out_path, legacy_alias_path)

    print("DONE")
    print("out =", out_path.as_posix())
    print("legacy_alias =", legacy_alias_path.as_posix())
    print("selected_hmap =", hmap_path)
    print("frames =", len(frames_out))


if __name__ == "__main__":
    main()
