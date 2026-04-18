from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

FIELD_L = 105.0

FIELD_W = 68.0

GOAL_WIDTH = 7.32

GOAL_HALF_W = GOAL_WIDTH / 2.0

MAX_TRACK_GAP_F = 3

MAX_PLAYER_SPEED_MS = 11.0

MIN_PLAYER_FRAMES = 125

POSSESSION_JSON = "/output/stage7_possession/possession.json"

PROJECTION_JSON = "/output/stage6_field/projection.json"

TRACK_LABELS_JSON = "/output/stage4_clustering/track_labels_corrected.json"

GAMEPLAY_JSON = "/output/stage3_filter/gameplay.json"

TEAM_SPOTTING_JSON = "/output/stage8_analytics/team_spotting_raw.json"

OUT_DIR = "/output/stage8_analytics"

CONF_THRESHOLDS = {
    "pass": 0.54,
    "drive": 0.58,
    "long_ball": 0.48,
    "header": 0.42,
    "out": 0.40,
    "cross": 0.42,
    "throw_in": 0.42,
    "shot": 0.34,
    "block": 0.36,
    "tackle": 0.40,
    "free_kick": 0.40,
    "goal": 0.30,
}

DEDUP_WINDOWS = {
    "pass": 14,
    "drive": 22,
    "long_ball": 20,
    "header": 16,
    "out": 24,
    "cross": 20,
    "throw_in": 26,
    "shot": 18,
    "block": 16,
    "tackle": 14,
    "free_kick": 28,
    "goal": 32,
}

POS_MEDIAN_WIN = 5

SPEED_MEDIAN_WIN = 5

GAP_BRIDGE_MAX = 2

SHOT_SPEED_MS_MIN = 13.0

SHOT_CLOSING_MS_MIN = 10.5

SHOT_ALIGN_MIN = 0.38

SHOT_MIN_PEAK_SCORE = 14.0

SHOT_MAX_EVENTS = 50

FINAL_THIRD_M = 35.0

BOX_DEPTH_M = 16.5

BOX_HALF_WIDTH_M = 20.16

PEAK_MIN_SEP_FRAMES = 55

PEAK_NEIGHBORHOOD = 3

POSSESSION_LOOKBACK_FRAMES = 15

INTERPOLATED_PENALTY = 0.35

MAX_SOURCE_FRAME_STEP = 3

GAMEPLAY_CUT_PAD_FRAMES = 3

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--possession-json", default=POSSESSION_JSON)
    ap.add_argument("--projection-json", default=PROJECTION_JSON)
    ap.add_argument("--track-labels-json", default=TRACK_LABELS_JSON)
    ap.add_argument("--gameplay-json", default=GAMEPLAY_JSON)
    ap.add_argument("--team-spotting-json", default=TEAM_SPOTTING_JSON)
    ap.add_argument("--out-dir", default=OUT_DIR)
    return ap.parse_args()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dist(a, b):
    if a is None or b is None:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ball_xy_from_projection(frame):
    ball = frame.get("ball", {})
    for key in ("pitch_xy_trusted", "pitch_xy_smoothed", "pitch_xy", "pitch_xy_raw"):
        xy = ball.get(key)
        if xy is not None and len(xy) >= 2 and np.isfinite(xy[0]) and np.isfinite(xy[1]):
            return [float(xy[0]), float(xy[1])]
    return None

def player_xy_at(proj_frames, frame_idx, track_id):
    if frame_idx < 0 or frame_idx >= len(proj_frames):
        return None
    for obj in proj_frames[frame_idx].get("players", []):
        if int(obj["track_id"]) != int(track_id):
            continue
        xy = obj.get("pitch_xy")
        if xy is None:
            return None
        return [float(xy[0]), float(xy[1])]
    return None

def nearest_team_player_to_ball(proj_frame, team=None):
    ball_xy = ball_xy_from_projection(proj_frame)
    if ball_xy is None:
        return None, None, None
    best_obj = None
    best_dist = None
    for obj in proj_frame.get("players", []):
        label = obj.get("label")
        if label not in ("team_1", "team_2"):
            continue
        if team is not None and label != team:
            continue
        xy = obj.get("pitch_xy")
        if xy is None:
            continue
        d = dist(ball_xy, xy)
        if d is None:
            continue
        if best_dist is None or d < best_dist:
            best_obj = obj
            best_dist = d
    if best_obj is None:
        return None, None, ball_xy
    return int(best_obj["track_id"]), best_obj.get("label"), ball_xy

def find_half_boundary_original_frame(track_labels, gameplay_frames):
    gk_fix = track_labels.get("goalkeeper_fix", {})
    boundary = gk_fix.get("half_boundary_original_frame")
    if boundary is not None:
        return int(boundary)
    if not gameplay_frames:
        return None
    return int(gameplay_frames[len(gameplay_frames) // 2].get("original_frame_index", 0))

def build_half_index(gameplay_frames, boundary_orig):
    original_indices = []
    half_by_frame = []
    for frame_idx, frame in enumerate(gameplay_frames):
        original_idx = frame.get("original_frame_index")
        if original_idx is None:
            original_idx = frame_idx
        original_idx = int(original_idx)
        original_indices.append(original_idx)
        half_idx = 1
        if boundary_orig is not None and original_idx >= boundary_orig:
            half_idx = 2
        half_by_frame.append(half_idx)
    return original_indices, half_by_frame

def infer_team_direction_by_half(proj_frames, half_by_frame):
    x_by_half = {
        1: {"team_1": [], "team_2": []},
        2: {"team_1": [], "team_2": []},
    }
    for frame_idx, frame in enumerate(proj_frames):
        half_idx = half_by_frame[frame_idx] if frame_idx < len(half_by_frame) else 1
        for obj in frame.get("players", []):
            label = obj.get("label")
            xy = obj.get("pitch_xy")
            if label not in ("team_1", "team_2") or xy is None:
                continue
            if not np.isfinite(xy[0]):
                continue
            x_by_half[half_idx][label].append(float(xy[0]))

    directions = {}
    for half_idx in (1, 2):
        team_1_x = x_by_half[half_idx]["team_1"]
        team_2_x = x_by_half[half_idx]["team_2"]
        if team_1_x and team_2_x:
            avg_1 = float(np.mean(team_1_x))
            avg_2 = float(np.mean(team_2_x))
            directions[half_idx] = {
                "team_1": "right" if avg_1 < avg_2 else "left",
                "team_2": "right" if avg_2 < avg_1 else "left",
            }
        elif half_idx == 2 and 1 in directions:
            directions[half_idx] = {
                "team_1": "left" if directions[1]["team_1"] == "right" else "right",
                "team_2": "left" if directions[1]["team_2"] == "right" else "right",
            }
        else:
            directions[half_idx] = {"team_1": "right", "team_2": "left"}
    return directions

def zone_of_half(x, team, half_idx, team_directions):
    direction = team_directions.get(half_idx, {}).get(team, "right")
    if direction == "right":
        if x < FIELD_L * 0.33:
            return "defensive_third"
        if x < FIELD_L * 0.67:
            return "middle_third"
        return "attacking_third"
    if x > FIELD_L * 0.67:
        return "defensive_third"
    if x > FIELD_L * 0.33:
        return "middle_third"
    return "attacking_third"

def goal_center_for(team, half_idx, team_directions):
    direction = team_directions.get(half_idx, {}).get(team, "right")
    return (FIELD_L, FIELD_W / 2.0) if direction == "right" else (0.0, FIELD_W / 2.0)

def xg_model(shot_xy, team, half_idx, team_directions):
    gc = goal_center_for(team, half_idx, team_directions)
    d = dist(shot_xy, gc)
    if d is None or d < 0.1:
        return 0.99
    angle = 2.0 * math.atan2(GOAL_HALF_W, d)
    raw = -1.5 + 2.1 * angle - 0.38 * math.log(max(1.0, d))
    xg = 1.0 / (1.0 + math.exp(-raw))
    return round(clamp(xg, 0.01, 0.97), 4)

def compute_player_stats(proj_frames, id_to_canonical, id_to_label, fps):
    player_stats = {}
    prev_player_xy = {}
    prev_player_fi = {}

    def get_or_init(cid, team):
        if cid not in player_stats:
            player_stats[cid] = {
                "team": team,
                "total_distance_m": 0.0,
                "top_speed_ms": 0.0,
                "frames_seen": 0,
                "positions": [],
            }
        return player_stats[cid]

    for fi, frame in enumerate(proj_frames):
        for obj in frame.get("players", []):
            tid = int(obj["track_id"])
            cid = int(id_to_canonical.get(tid, tid))
            label = id_to_label.get(tid, "unknown")
            if label not in ("team_1", "team_2"):
                continue
            xy = obj.get("pitch_xy")
            if xy is None or not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                continue
            st = get_or_init(cid, label)
            st["frames_seen"] += 1
            st["positions"].append([float(xy[0]), float(xy[1])])

            prev_xy = prev_player_xy.get(tid)
            prev_fi = prev_player_fi.get(tid, -999)
            if prev_xy is not None and (fi - prev_fi) <= MAX_TRACK_GAP_F:
                d_frame = dist(xy, prev_xy)
                speed_ms = d_frame * fps
                if speed_ms <= MAX_PLAYER_SPEED_MS:
                    st["total_distance_m"] += d_frame
                    st["top_speed_ms"] = max(st["top_speed_ms"], speed_ms)
            prev_player_xy[tid] = [float(xy[0]), float(xy[1])]
            prev_player_fi[tid] = fi

    filtered = {}
    for cid, st in player_stats.items():
        if st["frames_seen"] < MIN_PLAYER_FRAMES:
            continue
        pos = st.pop("positions", [])
        if pos:
            st["avg_position"] = [
                round(float(np.mean([p[0] for p in pos])), 1),
                round(float(np.mean([p[1] for p in pos])), 1),
            ]
        else:
            st["avg_position"] = None
        st["total_distance_m"] = round(float(st["total_distance_m"]), 1)
        st["top_speed_ms"] = round(float(st["top_speed_ms"]), 2)
        st["top_speed_kmh"] = round(float(st["top_speed_ms"]) * 3.6, 1)
        filtered[cid] = st
    return filtered

def build_zone_possession(poss_frames, proj_frames, half_by_frame, team_directions):
    counts = {"team_1": Counter(), "team_2": Counter()}
    total = Counter()
    for frame_idx, frame in enumerate(poss_frames):
        team = frame.get("current_team_label")
        if team not in ("team_1", "team_2"):
            continue
        ball_xy = ball_xy_from_projection(proj_frames[frame_idx])
        if ball_xy is None:
            continue
        half_idx = half_by_frame[frame_idx]
        zone = zone_of_half(ball_xy[0], team, half_idx, team_directions)
        counts[team][zone] += 1
        total[team] += 1

    out = {}
    for team in ("team_1", "team_2"):
        denom = max(int(total[team]), 1)
        out[team] = {
            "defensive_third_ratio": round(counts[team]["defensive_third"] / denom, 4),
            "middle_third_ratio": round(counts[team]["middle_third"] / denom, 4),
            "attacking_third_ratio": round(counts[team]["attacking_third"] / denom, 4),
        }
    return out

def build_team_stats(events, poss_summary, player_stats):
    xg_totals = {"team_1": 0.0, "team_2": 0.0}
    shot_counts = {"team_1": 0, "team_2": 0}
    pass_counts = {"team_1": 0, "team_2": 0}
    for event in events:
        team = event.get("team")
        if team not in ("team_1", "team_2"):
            continue
        if event["type"] in ("shot", "goal", "blocked_shot"):
            shot_counts[team] += 1
            xg_totals[team] += float(event.get("xG") or 0.0)
        if event["type"] in ("pass", "long_ball", "cross", "throw_in", "free_kick"):
            pass_counts[team] += 1

    team_distance = {"team_1": 0.0, "team_2": 0.0}
    for stats in player_stats.values():
        team = stats.get("team")
        if team in team_distance:
            team_distance[team] += float(stats.get("total_distance_m") or 0.0)

    return {
        "team_1": {
            "possession_ratio": round(float(poss_summary.get("team_1_ratio", 0.0)), 4),
            "shot_count": int(shot_counts["team_1"]),
            "xG_total": round(xg_totals["team_1"], 4),
            "pass_count": int(pass_counts["team_1"]),
            "total_distance_m": round(team_distance["team_1"], 1),
        },
        "team_2": {
            "possession_ratio": round(float(poss_summary.get("team_2_ratio", 0.0)), 4),
            "shot_count": int(shot_counts["team_2"]),
            "xG_total": round(xg_totals["team_2"], 4),
            "pass_count": int(pass_counts["team_2"]),
            "total_distance_m": round(team_distance["team_2"], 1),
        },
    }

def count_events(events):
    counts = Counter()
    for event in events:
        counts[event["type"]] += 1
    return dict(counts)

def infer_team_player_for_frame(frame_idx, poss_frames, proj_frames, id_to_canonical):
    lo = max(0, frame_idx - 10)
    hi = min(len(poss_frames), frame_idx + 11)
    team_votes = Counter()
    player_votes = Counter()
    for idx in range(lo, hi):
        team = poss_frames[idx].get("current_team_label")
        player = poss_frames[idx].get("current_player_id")
        if team in ("team_1", "team_2"):
            team_votes[team] += 1
            if player is not None:
                player_votes[(team, int(player))] += 1
    team = team_votes.most_common(1)[0][0] if team_votes else None
    player = None
    if team is not None:
        for (player_team, player_tid), _count in player_votes.most_common():
            if player_team == team:
                player = int(id_to_canonical.get(player_tid, player_tid))
                break
    if team is None:
        nearest_tid, nearest_team, _ = nearest_team_player_to_ball(proj_frames[frame_idx])
        team = nearest_team
        if nearest_tid is not None:
            player = int(id_to_canonical.get(nearest_tid, nearest_tid))
    elif player is None:
        nearest_tid, _nearest_team, _ = nearest_team_player_to_ball(proj_frames[frame_idx], team=team)
        if nearest_tid is not None:
            player = int(id_to_canonical.get(nearest_tid, nearest_tid))
    return team, player

def _rolling_median_1d(a: np.ndarray, win: int) -> np.ndarray:
    """Kenar tekrarlı basit rolling median; NaN korunur."""
    n = len(a)
    if win <= 1:
        return a.astype(float).copy()
    pad = win // 2
    x = np.pad(a.astype(float), (pad, pad), mode="edge")
    out = np.empty(n, dtype=float)
    for i in range(n):
        sl = x[i : i + win]
        out[i] = np.nanmedian(sl)
    return out

def _bridge_short_gaps(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray, max_gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kısa NaN boşluklarını lineer doldur (türev için)."""
    n = len(x)
    xi = x.astype(float).copy()
    yi = y.astype(float).copy()
    v = valid.astype(bool).copy()
    i = 0
    while i < n:
        if v[i]:
            i += 1
            continue
        j = i
        while j < n and not v[j]:
            j += 1
        gap = j - i
        if gap <= max_gap and i > 0 and j < n:
            for k in range(gap):
                t = (k + 1) / (gap + 1)
                xi[i + k] = xi[i - 1] * (1 - t) + xi[j] * t
                yi[i + k] = yi[i - 1] * (1 - t) + yi[j] * t
                v[i + k] = True
        i = j if j > i else i + 1
    return xi, yi, v

def compute_ball_kinematics(
    proj_frames: list[dict],
    fps: float,
) -> dict[str, Any]:
    """
    Her frame için pitch top konumu ve hız (m/s).
    Dönüş: x, y, vx_ms, vy_ms, speed_ms, valid, interpolated, visible
    """
    n = len(proj_frames)
    x = np.full(n, np.nan, dtype=float)
    y = np.full(n, np.nan, dtype=float)
    valid = np.zeros(n, dtype=bool)
    interp = np.zeros(n, dtype=bool)
    vis = np.zeros(n, dtype=bool)

    for i, fr in enumerate(proj_frames):
        b = fr.get("ball") or {}
        bxy = b.get("pitch_xy_trusted") or b.get("pitch_xy")
        if bxy and len(bxy) >= 2 and np.isfinite(bxy[0]) and np.isfinite(bxy[1]):
            x[i], y[i] = float(bxy[0]), float(bxy[1])
            valid[i] = True
        interp[i] = bool(b.get("interpolated", False))
        vis[i] = bool(b.get("visible", True))

    x_s = _rolling_median_1d(x, POS_MEDIAN_WIN)
    y_s = _rolling_median_1d(y, POS_MEDIAN_WIN)
    x_b, y_b, v_b = _bridge_short_gaps(x_s, y_s, valid, GAP_BRIDGE_MAX)

    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    spd = np.zeros(n, dtype=float)
    for i in range(1, n):
        if not v_b[i] or not v_b[i - 1]:
            continue
        vx[i] = (x_b[i] - x_b[i - 1]) * fps
        vy[i] = (y_b[i] - y_b[i - 1]) * fps
        spd[i] = math.hypot(vx[i], vy[i])

    # Ani sıçrama (homografi hatası) — hız sıfırla
    for i in range(1, n):
        if spd[i] > 55.0:
            vx[i] = vy[i] = spd[i] = 0.0

    hw = SPEED_MEDIAN_WIN // 2
    spd_med = np.copy(spd)
    for i in range(n):
        lo, hi = max(0, i - hw), min(n, i + hw + 1)
        spd_med[i] = float(np.median(spd[lo:hi])) if hi > lo else spd[i]

    return {
        "x": x_b,
        "y": y_b,
        "vx_ms": vx,
        "vy_ms": vy,
        "speed_ms": spd_med,
        "valid_mask": v_b,
        "interpolated": interp,
        "visible": vis,
    }

def build_gameplay_timeline_cut_mask(
    n: int,
    source_original_frame_index: Optional[Sequence[Optional[int]]],
    *,
    max_step: int = MAX_SOURCE_FRAME_STEP,
    pad: int = GAMEPLAY_CUT_PAD_FRAMES,
) -> np.ndarray:
    """
    True = bu gameplay karesi filtre birleşimi veya kaynak zaman sıçraması komşuluğunda;
    şut skoru ve hız bu karelerde kullanılmaz.
    """
    bad = np.zeros(n, dtype=bool)
    if not source_original_frame_index or n <= 0:
        return bad
    seq = list(source_original_frame_index)[:n]
    if len(seq) < n:
        seq.extend([None] * (n - len(seq)))
    for i in range(1, n):
        a, b = seq[i - 1], seq[i]
        if a is None or b is None:
            continue
        da, db = int(a), int(b)
        step = db - da
        if step < 0 or step > max_step:
            lo = max(0, i - pad)
            hi = min(n, i + pad + 1)
            bad[lo:hi] = True
    return bad

def apply_timeline_cut_to_kinematics(kin: dict[str, Any], cut_mask: np.ndarray) -> None:
    """Kesinti maskeli karelerde hız sıfırlanır (in-place)."""
    vx = kin["vx_ms"]
    vy = kin["vy_ms"]
    spd = kin["speed_ms"]
    n = len(cut_mask)
    for i in range(n):
        if cut_mask[i]:
            vx[i] = 0.0
            vy[i] = 0.0
            spd[i] = 0.0

def _goal_unit(ball_xy: tuple[float, float], goal_xy: tuple[float, float]) -> tuple[float, float]:
    dx = goal_xy[0] - ball_xy[0]
    dy = goal_xy[1] - ball_xy[1]
    L = math.hypot(dx, dy)
    if L < 0.25:
        return 0.0, 0.0
    return dx / L, dy / L

def _in_penalty_box_right(bx: float, by: float, field_l: float, field_w: float) -> bool:
    cy = field_w / 2.0
    return (field_l - BOX_DEPTH_M <= bx <= field_l) and (abs(by - cy) <= BOX_HALF_WIDTH_M)

def _in_penalty_box_left(bx: float, by: float, field_w: float) -> bool:
    cy = field_w / 2.0
    return (0.0 <= bx <= BOX_DEPTH_M) and (abs(by - cy) <= BOX_HALF_WIDTH_M)

def _shot_zone_ok(
    bx: float,
    by: float,
    target: str,
    field_l: float,
    field_w: float,
) -> bool:
    """target: 'left' veya 'right' kaleye şut için topun makul saha bölgesi."""
    ft = FINAL_THIRD_M
    if target == "right":
        if bx < field_l - ft:
            return False
        return True
    if target == "left":
        if bx > ft:
            return False
        return True
    return False

def detect_shots(
    proj_frames: list[dict],
    poss_frames: list[dict],
    *,
    fps: float,
    field_l: float,
    field_w: float,
    team_dir: dict[str, str],
    zone_of: Callable[[float, str], str],
    id_to_canonical: dict[int, int],
    source_original_frame_index: Optional[Sequence[Optional[int]]] = None,
) -> list[dict]:
    """
    Şut olayları listesi — analytics 'shot' event şemasına uyumlu alanlar.

    source_original_frame_index: gameplay.json `frames[].original_frame_index` ile aynı uzunlukta
    dizi (Stage 3 filtresi sahne atlamalarında indeksi sıçratır; verilmezse kesinti maskesi uygulanmaz).
    """
    n = len(proj_frames)
    kin = compute_ball_kinematics(proj_frames, fps)
    cut_mask = build_gameplay_timeline_cut_mask(n, source_original_frame_index)
    apply_timeline_cut_to_kinematics(kin, cut_mask)

    x, y = kin["x"], kin["y"]
    vx, vy = kin["vx_ms"], kin["vy_ms"]
    spd = kin["speed_ms"]
    vmask = kin["valid_mask"]
    interp = kin["interpolated"]

    gc_left = (0.0, field_w / 2.0)
    gc_right = (field_l, field_w / 2.0)

    score = np.zeros(n, dtype=float)

    for i in range(n):
        if cut_mask[i]:
            continue
        if not vmask[i] or not np.isfinite(x[i]) or not np.isfinite(y[i]):
            continue
        bxy = (float(x[i]), float(y[i]))
        vxi, vyi = float(vx[i]), float(vy[i])
        si = float(spd[i])
        if si < 1e-3:
            continue

        # Sağ kaleye şut (top sağa doğru genel olarak gidiyor)
        nr, ng = _goal_unit(bxy, gc_right)
        closing_r = vxi * nr + vyi * ng
        align_r = closing_r / si
        ok_r = (
            closing_r >= SHOT_CLOSING_MS_MIN
            and si >= SHOT_SPEED_MS_MIN
            and align_r >= SHOT_ALIGN_MIN
            and _shot_zone_ok(bxy[0], bxy[1], "right", field_l, field_w)
        )
        if ok_r:
            # Kanat / orta ayrımı: ceza sahası dışında çok lateral ise sıkılaştır
            if not _in_penalty_box_right(bxy[0], bxy[1], field_l, field_w):
                wide_y = bxy[1] < 14.0 or bxy[1] > (field_w - 14.0)
                if wide_y and (abs(vyi) > abs(vxi) * 1.08):
                    ok_r = False
            if ok_r:
                w = closing_r * (0.85 + 0.15 * min(1.0, si / 25.0))
                if interp[i]:
                    w *= INTERPOLATED_PENALTY
                score[i] = max(score[i], w)

        # Sol kaleye şut
        nl, ng2 = _goal_unit(bxy, gc_left)
        closing_l = vxi * nl + vyi * ng2
        align_l = closing_l / si
        ok_l = (
            closing_l >= SHOT_CLOSING_MS_MIN
            and si >= SHOT_SPEED_MS_MIN
            and align_l >= SHOT_ALIGN_MIN
            and _shot_zone_ok(bxy[0], bxy[1], "left", field_l, field_w)
        )
        if ok_l:
            if not _in_penalty_box_left(bxy[0], bxy[1], field_w):
                wide_y = bxy[1] < 14.0 or bxy[1] > (field_w - 14.0)
                if wide_y and (abs(vyi) > abs(vxi) * 1.08):
                    ok_l = False
            if ok_l:
                w = closing_l * (0.85 + 0.15 * min(1.0, si / 25.0))
                if interp[i]:
                    w *= INTERPOLATED_PENALTY
                score[i] = max(score[i], w)

    score[score < SHOT_MIN_PEAK_SCORE] = 0.0

    # Yerel maksimumlar
    peaks: list[int] = []
    for i in range(PEAK_NEIGHBORHOOD, n - PEAK_NEIGHBORHOOD):
        if cut_mask[i] or score[i] <= 0:
            continue
        if score[i] >= np.max(score[i - PEAK_NEIGHBORHOOD : i + PEAK_NEIGHBORHOOD + 1]) - 1e-9:
            if score[i] == np.max(score[i - PEAK_NEIGHBORHOOD : i + PEAK_NEIGHBORHOOD + 1]):
                peaks.append(i)

    # NMS — skora göre sıralı, minimum ayrım
    peaks.sort(key=lambda idx: float(score[idx]), reverse=True)
    picked: list[int] = []
    for idx in peaks:
        if cut_mask[idx]:
            continue
        if all(abs(idx - j) >= PEAK_MIN_SEP_FRAMES for j in picked):
            picked.append(idx)
    picked.sort()

    events: list[dict] = []
    for peak in picked:
        bx, by = float(x[peak]), float(y[peak])
        bxy = (bx, by)
        vxi, vyi = float(vx[peak]), float(vy[peak])
        si = float(spd[peak])

        nrx, nry = _goal_unit(bxy, gc_right)
        closing_r = vxi * nrx + vyi * nry
        nlx, nly = _goal_unit(bxy, gc_left)
        closing_l = vxi * nlx + vyi * nly

        if closing_r >= closing_l:
            target = "right"
            target_gc = gc_right
        else:
            target = "left"
            target_gc = gc_left

        # Bu kaleye saldıran takım (saha yönü — possession ile ezme)
        shooter_team: Optional[str] = None
        for t, d in team_dir.items():
            if (d == "right" and target == "right") or (d == "left" and target == "left"):
                shooter_team = t
                break

        if shooter_team is None:
            continue

        lo = max(0, peak - POSSESSION_LOOKBACK_FRAMES)
        players_hist = [
            poss_frames[j].get("current_player_id")
            for j in range(lo, peak + 1)
            if poss_frames[j].get("current_team_label") == shooter_team
            and poss_frames[j].get("current_player_id") is not None
        ]
        pid = Counter(players_hist).most_common(1)[0][0] if players_hist else None
        cid = id_to_canonical.get(int(pid), int(pid)) if pid is not None else None

        d_goal = math.hypot(bxy[0] - target_gc[0], bxy[1] - target_gc[1])

        events.append(
            {
                "frame": peak,
                "time_sec": round(peak / fps, 2),
                "team": shooter_team,
                "player": cid,
                "shot_xy": [round(bx, 1), round(by, 1)],
                "distance_m": round(d_goal, 1),
                "ball_speed_ms": round(si, 1),
                "closing_speed_ms": round(max(closing_r, closing_l), 1),
                "target_goal": target,
                "zone": zone_of(bx, shooter_team),
                "confidence": round(float(score[peak]), 3),
            }
        )

    events.sort(key=lambda e: float(e.get("confidence", 0.0)), reverse=True)
    events = events[:SHOT_MAX_EVENTS]
    events.sort(key=lambda e: int(e["frame"]))

    return events

def detect_tracking_shots(proj_frames, poss_frames, original_indices, half_by_frame, id_to_canonical, fps, team_directions):
    events = []
    ranges = []
    start = 0
    for frame_idx in range(1, len(half_by_frame)):
        if half_by_frame[frame_idx] != half_by_frame[frame_idx - 1]:
            ranges.append((half_by_frame[frame_idx - 1], start, frame_idx))
            start = frame_idx
    if half_by_frame:
        ranges.append((half_by_frame[-1], start, len(half_by_frame)))

    for half_idx, start_idx, end_idx in ranges:
        if end_idx - start_idx < 100:
            continue
        half_proj = proj_frames[start_idx:end_idx]
        half_poss = poss_frames[start_idx:end_idx]
        half_orig = original_indices[start_idx:end_idx]
        team_dir = team_directions.get(half_idx, {})

        def zone_of_local(x, team):
            return zone_of_half(x, team, half_idx, team_directions)

        shot_candidates = detect_shots(
            half_proj,
            half_poss,
            fps=fps,
            field_l=FIELD_L,
            field_w=FIELD_W,
            team_dir=team_dir,
            zone_of=zone_of_local,
            id_to_canonical=id_to_canonical,
            source_original_frame_index=half_orig,
        )

        for shot in shot_candidates:
            global_frame = int(start_idx + shot["frame"])
            shot_xy = tuple(shot["shot_xy"])
            events.append(
                {
                    "type": "shot",
                    "frame": global_frame,
                    "time_sec": round(global_frame / fps, 2),
                    "team": shot["team"],
                    "player": shot.get("player"),
                    "shot_xy": list(shot["shot_xy"]),
                    "distance_m": shot["distance_m"],
                    "ball_speed_ms": shot["ball_speed_ms"],
                    "closing_speed_ms": shot.get("closing_speed_ms"),
                    "target_goal": shot.get("target_goal"),
                    "zone": shot.get("zone", "unknown"),
                    "xG": xg_model(shot_xy, shot["team"], half_idx, team_directions),
                    "confidence": round(float(shot.get("confidence", 0.0)), 4),
                    "source": "tracking",
                    "half": int(half_idx),
                }
            )
    return events

def map_raw_label(label):
    mapping = {
        "PASS": "pass",
        "DRIVE": "drive",
        "HEADER": "header",
        "HIGH PASS": "long_ball",
        "OUT": "out",
        "CROSS": "cross",
        "THROW IN": "throw_in",
        "SHOT": "shot",
        "BALL PLAYER BLOCK": "block",
        "PLAYER SUCCESSFUL TACKLE": "tackle",
        "FREE KICK": "free_kick",
        "GOAL": "goal",
    }
    return mapping.get(str(label).upper(), str(label).lower().replace(" ", "_"))

def side_to_team_by_half(team_directions):
    out = {}
    for half_idx in (1, 2):
        t1_dir = team_directions.get(half_idx, {}).get("team_1", "right")
        if t1_dir == "right":
            out[half_idx] = {"left": "team_1", "right": "team_2"}
        else:
            out[half_idx] = {"left": "team_2", "right": "team_1"}
    return out

def nearest_shot_match(frame_idx, team, tracking_shots, window=20):
    best = None
    best_delta = None
    for shot in tracking_shots:
        if shot.get("team") != team:
            continue
        delta = abs(int(shot["frame"]) - int(frame_idx))
        if delta > window:
            continue
        if best is None or delta < best_delta or (
            delta == best_delta
            and float(shot.get("confidence", 0.0)) > float(best.get("confidence", 0.0))
        ):
            best = shot
            best_delta = delta
    return best

def lookup_half(frame_idx, half_by_frame):
    if not half_by_frame:
        return 1
    frame_idx = max(0, min(int(frame_idx), len(half_by_frame) - 1))
    return int(half_by_frame[frame_idx])

def resolve_actor(frame_idx, team, poss_frames, proj_frames, id_to_canonical):
    inferred_team, player = infer_team_player_for_frame(
        frame_idx, poss_frames, proj_frames, id_to_canonical
    )
    if inferred_team == team:
        return inferred_team, player

    best = None
    for idx in range(max(0, frame_idx - 8), min(len(proj_frames), frame_idx + 9)):
        track_id, label, _ = nearest_team_player_to_ball(proj_frames[idx], team=team)
        if track_id is None or label != team:
            continue
        ball_xy = ball_xy_from_projection(proj_frames[idx])
        if ball_xy is None:
            continue
        for obj in proj_frames[idx].get("players", []):
            if int(obj["track_id"]) != int(track_id):
                continue
            pxy = obj.get("pitch_xy")
            if pxy is None:
                continue
            d = (
                (float(pxy[0]) - float(ball_xy[0])) ** 2
                + (float(pxy[1]) - float(ball_xy[1])) ** 2
            ) ** 0.5
            cand = (d, abs(idx - frame_idx), int(id_to_canonical.get(track_id, track_id)))
            if best is None or cand < best:
                best = cand
    if best is None:
        return team, None
    return team, best[2]

def resolve_event_xy(frame_idx, team, proj_frames):
    frame_idx = max(0, min(int(frame_idx), len(proj_frames) - 1))
    for offset in (0, -1, 1, -2, 2, -4, 4, -6, 6):
        idx = frame_idx + offset
        if idx < 0 or idx >= len(proj_frames):
            continue
        ball_xy = ball_xy_from_projection(proj_frames[idx])
        if ball_xy is not None:
            return [round(float(ball_xy[0]), 1), round(float(ball_xy[1]), 1)]
        _, _, ball_xy = nearest_team_player_to_ball(proj_frames[idx], team=team)
        if ball_xy is not None:
            return [round(float(ball_xy[0]), 1), round(float(ball_xy[1]), 1)]
    return None

def passes_geometry_ok(event_type, event_xy, team, half_idx, team_directions):
    if event_xy is None:
        return False
    x, y = float(event_xy[0]), float(event_xy[1])
    zone = zone_of_half(x, team, half_idx, team_directions)
    if event_type == "cross":
        wing = y <= 15.0 or y >= (FIELD_W - 15.0)
        return wing and zone == "attacking_third"
    if event_type == "throw_in":
        return y <= 3.0 or y >= (FIELD_W - 3.0)
    if event_type == "out":
        return (
            y <= 2.2
            or y >= (FIELD_W - 2.2)
            or x <= 1.5
            or x >= (FIELD_L - 1.5)
        )
    return True

def dedupe_events(events):
    kept = []
    for event in sorted(
        events,
        key=lambda e: (
            float(e.get("confidence_spotting", 0.0)),
            float(e.get("confidence_validation", 0.0)),
        ),
        reverse=True,
    ):
        window = DEDUP_WINDOWS.get(event["type"], 16)
        duplicate = False
        for prev in kept:
            if prev["type"] != event["type"]:
                continue
            if prev.get("team") != event.get("team"):
                continue
            if abs(int(prev["frame"]) - int(event["frame"])) <= window:
                duplicate = True
                break
        if not duplicate:
            kept.append(event)
    kept.sort(key=lambda e: int(e["frame"]))
    return kept

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poss_data = load_json(args.possession_json)
    proj_data = load_json(args.projection_json)
    labels_data = load_json(args.track_labels_json)
    gameplay_data = load_json(args.gameplay_json)
    spotting_data = load_json(args.team_spotting_json)

    poss_frames = poss_data["frames"]
    proj_frames = proj_data["frames"]
    gameplay_frames = gameplay_data["frames"]
    raw_events = spotting_data.get("events", [])
    tracks = labels_data["tracks"]

    n = min(len(poss_frames), len(proj_frames), len(gameplay_frames))
    poss_frames = poss_frames[:n]
    proj_frames = proj_frames[:n]
    gameplay_frames = gameplay_frames[:n]
    fps = float(gameplay_data.get("fps", 25.0))

    id_to_label = {int(k): v.get("label", "unknown") for k, v in tracks.items()}
    id_to_canonical = {int(k): int(v.get("canonical_id", int(k))) for k, v in tracks.items()}

    half_boundary_orig = find_half_boundary_original_frame(labels_data, gameplay_frames)
    original_indices, half_by_frame = build_half_index(gameplay_frames, half_boundary_orig)
    team_directions = infer_team_direction_by_half(proj_frames, half_by_frame)
    side_map = side_to_team_by_half(team_directions)

    print("\nStage 8 v4 | Tracking shot validation...")
    tracking_shots = detect_tracking_shots(
        proj_frames,
        poss_frames,
        original_indices,
        half_by_frame,
        id_to_canonical,
        fps,
        team_directions,
    )

    print("Stage 8 v4 | Mapping official team spotting events...")
    events = []
    for raw_event in raw_events:
        frame_idx = int(raw_event.get("frame", 0))
        if frame_idx < 0 or frame_idx >= n:
            continue
        half_idx = lookup_half(frame_idx, half_by_frame)
        raw_side = str(raw_event.get("side") or "").lower()
        team = side_map.get(half_idx, {}).get(raw_side)
        if team not in ("team_1", "team_2"):
            continue

        event_type = map_raw_label(raw_event.get("label"))
        confidence = float(raw_event.get("confidence") or 0.0)
        if confidence < CONF_THRESHOLDS.get(event_type, 0.45):
            continue

        actor_team, actor_player = resolve_actor(
            frame_idx, team, poss_frames, proj_frames, id_to_canonical
        )
        if actor_team not in (None, team) and event_type in (
            "pass",
            "drive",
            "long_ball",
            "cross",
        ):
            continue

        event_xy = resolve_event_xy(frame_idx, team, proj_frames)
        if event_type in ("cross", "throw_in", "out") and not passes_geometry_ok(
            event_type, event_xy, team, half_idx, team_directions
        ):
            continue

        base_event = {
            "type": event_type,
            "frame": int(frame_idx),
            "time_sec": round(frame_idx / fps, 2),
            "team": team,
            "player": actor_player,
            "half": int(half_idx),
            "zone": zone_of_half(event_xy[0], team, half_idx, team_directions)
            if event_xy
            else "unknown",
            "event_xy": event_xy,
            "confidence_spotting": round(confidence, 4),
            "source": "sn_teamspotting",
            "raw_label": str(raw_event.get("label")).upper(),
            "raw_side": raw_side,
        }

        if event_type in ("shot", "goal", "block"):
            shot_match = nearest_shot_match(frame_idx, team, tracking_shots)
            if event_type == "shot" and shot_match is None:
                continue
            if shot_match is not None:
                shot_xy = shot_match.get("shot_xy") or event_xy
                base_event.update(
                    {
                        "shot_xy": shot_xy,
                        "distance_m": shot_match.get("distance_m"),
                        "ball_speed_ms": shot_match.get("ball_speed_ms"),
                        "closing_speed_ms": shot_match.get("closing_speed_ms"),
                        "target_goal": shot_match.get("target_goal"),
                        "xG": shot_match.get("xG"),
                        "zone": shot_match.get("zone", base_event["zone"]),
                        "confidence_validation": round(
                            float(shot_match.get("confidence", 0.0)), 4
                        ),
                    }
                )
            elif event_xy is not None:
                base_event.update(
                    {
                        "shot_xy": event_xy,
                        "xG": xg_model(tuple(event_xy), team, half_idx, team_directions),
                        "confidence_validation": 0.0,
                    }
                )
        else:
            base_event["confidence_validation"] = 0.0

        events.append(base_event)

    events = dedupe_events(events)

    print("Stage 8 v4 | Player stats / team summaries...")
    player_stats = compute_player_stats(proj_frames, id_to_canonical, id_to_label, fps)
    team_stats = build_team_stats(events, poss_data.get("summary", {}), player_stats)
    zone_possession = build_zone_possession(
        poss_frames, proj_frames, half_by_frame, team_directions
    )
    xg_totals = {
        "team_1": round(
            sum(
                float(e.get("xG") or 0.0)
                for e in events
                if e.get("team") == "team_1" and e["type"] in ("shot", "goal", "block")
            ),
            4,
        ),
        "team_2": round(
            sum(
                float(e.get("xG") or 0.0)
                for e in events
                if e.get("team") == "team_2" and e["type"] in ("shot", "goal", "block")
            ),
            4,
        ),
    }

    payload = {
        "fps": fps,
        "field_length_m": FIELD_L,
        "field_width_m": FIELD_W,
        "total_frames": n,
        "total_duration_sec": round(n / fps, 2) if fps > 0 else 0.0,
        "team_direction": {
            "half_1": team_directions.get(1, {}),
            "half_2": team_directions.get(2, {}),
            "half_boundary_original_frame": half_boundary_orig,
        },
        "goal_centers": {
            "half_1": {
                "team_1": list(goal_center_for("team_1", 1, team_directions)),
                "team_2": list(goal_center_for("team_2", 1, team_directions)),
            },
            "half_2": {
                "team_1": list(goal_center_for("team_1", 2, team_directions)),
                "team_2": list(goal_center_for("team_2", 2, team_directions)),
            },
        },
        "event_counts": count_events(events),
        "events": events,
        "team_stats": team_stats,
        "zone_possession": zone_possession,
        "xG": xg_totals,
        "player_stats": player_stats,
        "heatmaps": {"team1": False, "team2": False, "ball": False},
        "sources": {
            "possession_json": args.possession_json,
            "projection_json": args.projection_json,
            "track_labels_json": args.track_labels_json,
            "gameplay_json": args.gameplay_json,
            "team_spotting_json": args.team_spotting_json,
        },
        "debug": {
            "tracking_shot_count": len(tracking_shots),
            "team_spotting_raw_count": len(raw_events),
        },
    }

    out_path = out_dir / "analytics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("out =", out_path)
    print("event_counts =", payload["event_counts"])

if __name__ == "__main__":
    main()
