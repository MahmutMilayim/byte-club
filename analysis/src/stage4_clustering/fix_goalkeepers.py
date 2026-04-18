"""
Stage 4b - Goalkeeper relabeling with half-aware global slot assignment.

Only referee / light-goalkeeper style tracks are eligible for relabeling.
Regular outfield team kits remain protected. Team assignment is inferred from
strong goalkeeper-slot evidence across halves instead of fragile local votes.
"""

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

TRACK_LABELS_JSON = "/output/stage4_clustering/track_labels.json"
PROJECTION_JSON = "/output/stage6_field/projection.json"
OUT_JSON = "/output/stage4_clustering/track_labels_corrected.json"
GAMEPLAY_JSON = "/output/stage3_filter/gameplay.json"
GAMEPLAY_VIDEO = "/output/stage3_filter/gameplay.mp4"
OUT_VIDEO = "/output/stage4_clustering/gkfix_review.mp4"
HALF_BOUNDARY_CACHE = "/output/stage4_clustering/goalkeeper_half_boundary_cache.json"

LABEL_COLORS = {
    "team_1": (0, 165, 255),
    "team_2": (40, 40, 40),
    "referee": (255, 255, 0),
    "unknown": (0, 255, 0),
}

REFEREE_BLUE_MIN = 0.06
ORANGE_TEAM_STRONG = 0.10
TEAM2_GRAY_DARK_SUM = 0.50
TEAM2_ORANGE_CAP = 0.05

LIGHT_KIT_MIN_V = 158.0
LIGHT_KIT_MAX_ORANGE = 0.08
LIGHT_KIT_MAX_DARK_GRAY = 0.42

LEFT_GK_X_MAX = 20.0
RIGHT_GK_X_MIN = 85.0
MIN_SAMPLES = 30
NEAR_TEAM_RADIUS_M = 22.0

GK_Y_MIN = 22.0
GK_Y_MAX = 46.0
GK_ZONE_RATIO_MIN = 0.55
GK_X_RANGE_MAX = 18.0

OCR_SCOREBOARD_HEIGHT_FRAC = 0.14
OCR_SCOREBOARD_WIDTH_FRAC = 0.35
OCR_COARSE_STEP_SEC = 20.0
OCR_FINE_STEP_SEC = 1.0
HALFTIME_MIN_CLOCK_SEC = 45 * 60
HALFTIME_MAX_CLOCK_SEC = 50 * 60
HALFTIME_MIN_DROP_SEC = 30
HALFTIME_MIN_GAMEPLAY_FRAC = 0.30
HALFTIME_MAX_GAMEPLAY_FRAC = 0.80
TESSERACT_COMMON_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

GK_HALF_LOCK_MIN_ZONE_RATIO = 0.60
GK_HALF_LOCK_MIN_SAMPLES = 45
GK_DOMINANT_HALF_WEIGHT_RATIO = 1.25


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-labels-json", default=TRACK_LABELS_JSON)
    ap.add_argument("--projection-json", default=PROJECTION_JSON)
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--gameplay-json", default=GAMEPLAY_JSON)
    ap.add_argument("--gameplay-video", default=GAMEPLAY_VIDEO)
    ap.add_argument("--out-video", default=OUT_VIDEO)
    ap.add_argument("--half-boundary-cache", default=HALF_BOUNDARY_CACHE)
    return ap.parse_args()


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def safe_int(v):
    return int(round(float(v)))


def resolve_original_frame_index(frame_record, gameplay_frames):
    original_idx = frame_record.get("source_original_frame_index")
    if original_idx is not None:
        return int(original_idx)

    seq_idx = frame_record.get("frame_index")
    if seq_idx is None:
        return None
    seq_idx = int(seq_idx)
    if 0 <= seq_idx < len(gameplay_frames):
        original_idx = gameplay_frames[seq_idx].get("original_frame_index")
        if original_idx is not None:
            return int(original_idx)
    return None


def build_track_summary(points):
    if len(points) < MIN_SAMPLES:
        return None

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    median_x = sorted(xs)[len(xs) // 2]
    median_y = sorted(ys)[len(ys) // 2]

    in_left = sum(1 for x in xs if x < LEFT_GK_X_MAX)
    in_right = sum(1 for x in xs if x > RIGHT_GK_X_MIN)
    zone_ratio = max(in_left, in_right) / len(points)

    side_left = in_left >= in_right
    zone_xs = (
        [x for x in xs if x < LEFT_GK_X_MAX]
        if side_left
        else [x for x in xs if x > RIGHT_GK_X_MIN]
    )
    x_range = (max(zone_xs) - min(zone_xs)) if zone_xs else 999.0

    return {
        "count": len(points),
        "median_x": median_x,
        "median_y": median_y,
        "zone_ratio": zone_ratio,
        "x_range": x_range,
        "in_left": in_left,
        "in_right": in_right,
        "side_left": side_left,
    }


def is_gk_candidate(summary):
    if summary is None:
        return False
    if summary["zone_ratio"] < GK_ZONE_RATIO_MIN:
        return False
    if not (GK_Y_MIN <= summary["median_y"] <= GK_Y_MAX):
        return False
    if summary["x_range"] > GK_X_RANGE_MAX:
        return False
    return True


def _stats(tracks, tid):
    st = tracks.get(str(tid), {}).get("stats") or {}
    return st if isinstance(st, dict) else {}


def is_protected_outfield_player(stats):
    if not stats:
        return True
    orange = float(stats.get("orange_ratio", 0))
    gray = float(stats.get("gray_ratio", 0))
    dark = float(stats.get("dark_ratio", 0))

    if orange >= ORANGE_TEAM_STRONG:
        return True
    if orange <= TEAM2_ORANGE_CAP and (gray + dark) >= TEAM2_GRAY_DARK_SUM:
        return True
    return False


def is_referee_kit(stats):
    if not stats:
        return False
    orange = float(stats.get("orange_ratio", 0))
    blue = float(stats.get("blue_ratio", 0))
    return blue >= REFEREE_BLUE_MIN and orange < ORANGE_TEAM_STRONG


def is_light_goalkeeper_kit(stats):
    if not stats:
        return False
    if is_protected_outfield_player(stats):
        return False
    orange = float(stats.get("orange_ratio", 0))
    gray = float(stats.get("gray_ratio", 0))
    dark = float(stats.get("dark_ratio", 0))
    mean_v = float(stats.get("mean_v", 0))

    if mean_v < LIGHT_KIT_MIN_V:
        return False
    if orange > LIGHT_KIT_MAX_ORANGE:
        return False
    if (dark + gray) > LIGHT_KIT_MAX_DARK_GRAY:
        return False
    return True


def is_fix_eligible(tracks, tid):
    rec = tracks.get(str(tid), {})
    label = rec.get("label")
    stats = _stats(tracks, tid)

    if label == "referee":
        return True
    if not stats:
        return False
    if is_protected_outfield_player(stats):
        return False
    if is_referee_kit(stats):
        return True
    if label in ("team_1", "team_2") and is_light_goalkeeper_kit(stats):
        return True
    return False


def render_corrected_review(
    video_path, gameplay_frames, tracks_dict, out_path, changed_tids, fps_hint=None
):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: review video skipped, cannot open: {video_path}")
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or (float(fps_hint) if fps_hint else 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"  WARNING: review video cannot be written: {out_path}")
        return

    def tid_label(tid):
        rec = tracks_dict.get(str(tid)) or tracks_dict.get(tid)
        if not rec:
            return "unknown"
        return rec.get("label", "unknown")

    total_json_frames = len(gameplay_frames)
    fidx = 0
    changed_set = {int(x) for x in changed_tids}

    with tqdm(
        total=max(total_json_frames, 1),
        desc="Stage 4b | Review video",
        unit="frame",
        ncols=90,
    ) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fidx < total_json_frames:
                for obj in gameplay_frames[fidx].get("objects", []):
                    tid = int(obj["track_id"])
                    x1, y1, x2, y2 = map(safe_int, obj["bbox_xyxy"])
                    label = tid_label(tid)
                    color = LABEL_COLORS.get(label, (0, 255, 0))
                    thickness = 4 if tid in changed_set else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    marker = "*" if tid in changed_set else ""
                    text = f"ID {tid} | {label}{marker}"
                    (tw, th), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x1, max(0, y1 - th - 8)),
                        (x1 + tw + 8, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        text,
                        (x1 + 4, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                frame,
                "Stage 4b - track_labels_corrected (thick bbox = label changed)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            fidx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    print("  review video ->", out_path)


def _segment_gap_fallback(gameplay_data):
    frames = gameplay_data.get("frames", [])
    if len(frames) < 2:
        return None, {"source": "unavailable"}

    best_gap = None
    for idx in range(1, len(frames)):
        prev_orig = frames[idx - 1].get("original_frame_index")
        cur_orig = frames[idx].get("original_frame_index")
        if prev_orig is None or cur_orig is None:
            continue
        gap = int(cur_orig) - int(prev_orig)
        if gap <= 1:
            continue
        if best_gap is None or gap > best_gap[0]:
            best_gap = (gap, idx)

    if best_gap is None:
        return None, {"source": "unavailable"}

    boundary_gp_idx = int(best_gap[1])
    boundary_orig = int(frames[boundary_gp_idx]["original_frame_index"])
    return boundary_orig, {
        "source": "segment_gap_fallback",
        "boundary_gp_idx": boundary_gp_idx,
        "largest_gap_frames": int(best_gap[0]),
    }


def _configure_tesseract():
    try:
        import pytesseract
    except ImportError:
        return None

    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        for candidate in TESSERACT_COMMON_PATHS:
            if Path(candidate).exists():
                tesseract_path = candidate
                break
    if not tesseract_path:
        return None

    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    return pytesseract


def _scoreboard_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        : max(1, int(height * OCR_SCOREBOARD_HEIGHT_FRAC)),
        : max(1, int(width * OCR_SCOREBOARD_WIDTH_FRAC)),
    ]


def _ocr_variants(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return [
        cv2.resize(roi, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC),
        cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC),
        cv2.resize(
            cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1],
            None,
            fx=4.0,
            fy=4.0,
            interpolation=cv2.INTER_CUBIC,
        ),
        cv2.resize(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            None,
            fx=4.0,
            fy=4.0,
            interpolation=cv2.INTER_CUBIC,
        ),
    ]


def _parse_clock_records(text):
    if not text:
        return []

    raw = text
    cleaned = text.upper().replace(";", ":").replace("\n", " ")
    for src, dst in (("O", "0"), ("I", "1"), ("L", "1"), ("B", "8"), ("S", "5")):
        cleaned = cleaned.replace(src, dst)

    records = []

    def add_record(base_min, sec, extra_min=0):
        if not (0 <= base_min <= 140 and 0 <= sec < 60 and 0 <= extra_min <= 15):
            return
        total_sec = (base_min + extra_min) * 60 + sec
        records.append(
            {
                "total_sec": total_sec,
                "base_min": base_min,
                "sec": sec,
                "extra_min": extra_min,
                "raw": raw,
            }
        )

    for match in re.finditer(r"\+(\d{1,2})\s*(\d{2,3})[: ](\d{2})", cleaned):
        add_record(int(match.group(2)), int(match.group(3)), extra_min=int(match.group(1)))
    for match in re.finditer(r"(\d{1,3})\s*\+\s*(\d{1,2})[: ](\d{2})", cleaned):
        add_record(int(match.group(1)), int(match.group(3)), extra_min=int(match.group(2)))
    for match in re.finditer(r"(\d{1,3})[: ](\d{2})", cleaned):
        add_record(int(match.group(1)), int(match.group(2)), extra_min=0)

    collapsed = "".join(ch for ch in cleaned if ch.isdigit() or ch == "+")
    plus_match = re.search(r"\+(\d{1,2})(\d{2})(\d{2})", collapsed)
    if plus_match:
        add_record(
            int(plus_match.group(2)),
            int(plus_match.group(3)),
            extra_min=int(plus_match.group(1)),
        )

    digit_runs = re.findall(r"\d{3,5}", collapsed.replace("+", ""))
    for run in digit_runs:
        if len(run) == 3:
            add_record(int(run[0]), int(run[1:]), extra_min=0)
        elif len(run) == 4:
            add_record(int(run[:2]), int(run[2:]), extra_min=0)
        elif len(run) == 5:
            add_record(int(run[:2]), int(run[2:4]), extra_min=int(run[0]))

    dedup = {}
    for rec in records:
        key = (rec["total_sec"], rec["base_min"], rec["sec"], rec["extra_min"])
        dedup[key] = rec
    return list(dedup.values())


def _ocr_clock_records(cap, gp_idx, pytesseract_mod):
    if gp_idx < 0:
        return []

    cap.set(cv2.CAP_PROP_POS_FRAMES, gp_idx)
    ret, frame = cap.read()
    if not ret:
        return []

    roi = _scoreboard_roi(frame)
    variants = _ocr_variants(roi)
    configs = (
        "--psm 6 -c tessedit_char_whitelist=0123456789:+ ",
        "--psm 11 -c tessedit_char_whitelist=0123456789:+ ",
        "--psm 12 -c tessedit_char_whitelist=0123456789:+ ",
    )

    records = []
    for variant in variants:
        for cfg in configs:
            try:
                text = pytesseract_mod.image_to_string(variant, config=cfg).strip()
            except Exception:
                continue
            records.extend(_parse_clock_records(text))

    dedup = {}
    for rec in records:
        key = (rec["total_sec"], rec["base_min"], rec["sec"], rec["extra_min"])
        dedup[key] = rec
    return list(dedup.values())


def _classify_halftime_candidates(records):
    plus = sorted(
        rec["total_sec"]
        for rec in records
        if rec["extra_min"] > 0
        and "+" in rec["raw"]
        and HALFTIME_MIN_CLOCK_SEC <= rec["total_sec"] <= HALFTIME_MAX_CLOCK_SEC
    )
    plain = sorted(
        rec["total_sec"]
        for rec in records
        if rec["extra_min"] == 0
        and HALFTIME_MIN_CLOCK_SEC <= rec["total_sec"] <= HALFTIME_MAX_CLOCK_SEC
    )
    return plus, plain


def _load_half_boundary_cache(cache_path):
    if not cache_path:
        return None, None

    path = Path(cache_path)
    if not path.exists():
        return None, None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    boundary_orig = payload.get("half_boundary_original_frame")
    half_info = payload.get("half_boundary_info") or {}
    if boundary_orig is None:
        return None, None

    source = str(half_info.get("source") or "")
    if source and not source.startswith("scoreboard_clock_ocr"):
        return None, None

    try:
        boundary_orig = int(boundary_orig)
    except (TypeError, ValueError):
        return None, None

    half_info = dict(half_info)
    half_info["source"] = "scoreboard_clock_ocr_cached"
    return boundary_orig, half_info


def _save_half_boundary_cache(cache_path, boundary_orig, half_info):
    if not cache_path or boundary_orig is None or not half_info:
        return
    if half_info.get("source") != "scoreboard_clock_ocr":
        return

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "half_boundary_original_frame": int(boundary_orig),
        "half_boundary_info": dict(half_info),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_half_boundary(gameplay_video_path, gameplay_data, cache_path=None):
    fallback_boundary, fallback_info = _segment_gap_fallback(gameplay_data)
    gameplay_frames = gameplay_data.get("frames", [])
    if not gameplay_frames:
        return fallback_boundary, fallback_info

    cached_boundary, cached_info = _load_half_boundary_cache(cache_path)
    pytesseract_mod = _configure_tesseract()
    if pytesseract_mod is None:
        if cached_boundary is not None:
            return cached_boundary, cached_info
        fallback_info = dict(fallback_info)
        fallback_info["source"] = "segment_gap_no_tesseract"
        return fallback_boundary, fallback_info

    cap = cv2.VideoCapture(str(gameplay_video_path))
    if not cap.isOpened():
        if cached_boundary is not None:
            return cached_boundary, cached_info
        fallback_info = dict(fallback_info)
        fallback_info["source"] = "segment_gap_no_video"
        return fallback_boundary, fallback_info

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(gameplay_frames))
    coarse_step = max(1, int(round(fps * OCR_COARSE_STEP_SEC)))
    fine_step = max(1, int(round(fps * OCR_FINE_STEP_SEC)))

    coarse_samples = []
    for gp_idx in range(0, total_frames, coarse_step):
        records = _ocr_clock_records(cap, gp_idx, pytesseract_mod)
        plus, plain = _classify_halftime_candidates(records)
        if plus or plain:
            coarse_samples.append(
                {
                    "gp_idx": gp_idx,
                    "plus": plus,
                    "plain": plain,
                    "best_plus": max(plus) if plus else None,
                    "best_plain": min(plain) if plain else None,
                }
            )

    search_min_gp_idx = int(total_frames * HALFTIME_MIN_GAMEPLAY_FRAC)
    search_max_gp_idx = int(total_frames * HALFTIME_MAX_GAMEPLAY_FRAC)
    search_samples = [
        s for s in coarse_samples if search_min_gp_idx <= s["gp_idx"] <= search_max_gp_idx
    ]

    transition = None
    last_plus_sample = None
    for sample in search_samples:
        if sample["plus"]:
            last_plus_sample = sample
            continue
        if last_plus_sample and sample["plain"]:
            if sample["best_plain"] + HALFTIME_MIN_DROP_SEC <= last_plus_sample["best_plus"]:
                transition = (last_plus_sample, sample)
                break

    if transition is None:
        halftime_like = [s for s in search_samples if s["plus"] or s["plain"]]
        for prev, cur in zip(halftime_like, halftime_like[1:]):
            prev_total = prev["best_plus"] if prev["best_plus"] is not None else prev["best_plain"]
            cur_total = cur["best_plain"] if cur["best_plain"] is not None else cur["best_plus"]
            if prev_total is None or cur_total is None:
                continue
            if cur_total + HALFTIME_MIN_DROP_SEC <= prev_total:
                transition = (prev, cur)
                break

    if transition is None:
        cap.release()
        if cached_boundary is not None:
            return cached_boundary, cached_info
        fallback_info = dict(fallback_info)
        fallback_info["source"] = "segment_gap_ocr_no_transition"
        fallback_info["coarse_samples"] = len(coarse_samples)
        fallback_info["search_window"] = [search_min_gp_idx, search_max_gp_idx]
        return fallback_boundary, fallback_info

    coarse_prev, coarse_cur = transition
    search_start = max(0, coarse_prev["gp_idx"])
    search_end = min(total_frames - 1, coarse_cur["gp_idx"])
    last_plus_idx = coarse_prev["gp_idx"]
    last_plus_total = coarse_prev["best_plus"]
    first_plain_idx = coarse_cur["gp_idx"]
    first_plain_total = coarse_cur["best_plain"]

    for gp_idx in range(search_start, search_end + 1, fine_step):
        records = _ocr_clock_records(cap, gp_idx, pytesseract_mod)
        plus, plain = _classify_halftime_candidates(records)
        if plus:
            last_plus_idx = gp_idx
            last_plus_total = max(plus)
            continue
        if plain:
            best_plain = min(plain)
            if last_plus_total is None or best_plain + HALFTIME_MIN_DROP_SEC <= last_plus_total:
                first_plain_idx = gp_idx
                first_plain_total = best_plain
                break

    cap.release()

    boundary_gp_idx = first_plain_idx
    best_gap = None
    scan_start = max(1, last_plus_idx)
    scan_end = min(len(gameplay_frames) - 1, first_plain_idx + 1)
    for idx in range(scan_start, scan_end):
        prev_orig = gameplay_frames[idx - 1].get("original_frame_index")
        cur_orig = gameplay_frames[idx].get("original_frame_index")
        if prev_orig is None or cur_orig is None:
            continue
        gap = int(cur_orig) - int(prev_orig)
        if gap <= 1:
            continue
        if best_gap is None or gap > best_gap[0]:
            best_gap = (gap, idx)

    if best_gap is not None:
        boundary_gp_idx = int(best_gap[1])

    boundary_orig = int(gameplay_frames[boundary_gp_idx]["original_frame_index"])
    half_info = {
        "source": "scoreboard_clock_ocr",
        "coarse_samples": len(coarse_samples),
        "search_window": [search_min_gp_idx, search_max_gp_idx],
        "last_plus_gp_idx": int(last_plus_idx),
        "last_plus_total_sec": int(last_plus_total) if last_plus_total is not None else None,
        "first_plain_gp_idx": int(first_plain_idx),
        "first_plain_total_sec": int(first_plain_total) if first_plain_total is not None else None,
        "boundary_gp_idx": int(boundary_gp_idx),
        "largest_gap_frames": int(best_gap[0]) if best_gap is not None else None,
    }
    _save_half_boundary_cache(cache_path, boundary_orig, half_info)
    return boundary_orig, half_info


def expected_team_from_half_side(half_idx, side_left):
    if half_idx == 1:
        return "team_1" if side_left else "team_2"
    return "team_2" if side_left else "team_1"

def build_track_summaries(projection_frames, gameplay_frames, half_boundary_orig, tracks):
    track_points_all = defaultdict(list)
    track_points_by_half = {1: defaultdict(list), 2: defaultdict(list)}

    for fr in tqdm(
        projection_frames,
        desc="Stage 4b | Scan frames",
        unit="frame",
        ncols=90,
    ):
        original_idx = resolve_original_frame_index(fr, gameplay_frames)
        half_idx = 1
        if half_boundary_orig is not None and original_idx is not None and original_idx >= half_boundary_orig:
            half_idx = 2

        for obj in fr.get("players", []) + fr.get("referees", []):
            tid = int(obj["track_id"])
            if str(tid) not in tracks:
                continue
            xy = obj.get("pitch_xy")
            if xy is None or len(xy) < 2:
                continue
            point = [float(xy[0]), float(xy[1])]
            track_points_all[tid].append(point)
            track_points_by_half[half_idx][tid].append(point)

    summaries_all = {}
    summaries = {1: {}, 2: {}}
    for tid, points in track_points_all.items():
        summary = build_track_summary(points)
        if summary is not None:
            summaries_all[tid] = summary
    for half_idx in (1, 2):
        for tid, points in track_points_by_half[half_idx].items():
            summary = build_track_summary(points)
            if summary is not None:
                summaries[half_idx][tid] = summary
    return summaries_all, summaries

def collect_slot_leaders(half_summaries, eligible_ids):
    slots = {
        "half_1_left": [],
        "half_1_right": [],
        "half_2_left": [],
        "half_2_right": [],
    }
    for half_idx in (1, 2):
        for tid, summary in half_summaries[half_idx].items():
            if tid not in eligible_ids:
                continue
            if not is_gk_candidate(summary):
                continue
            side = "left" if summary["side_left"] else "right"
            score = float(summary["zone_ratio"]) * float(summary["count"])
            slots[f"half_{half_idx}_{side}"].append(
                {
                    "track_id": int(tid),
                    "score": round(score, 4),
                    "zone_ratio": round(float(summary["zone_ratio"]), 4),
                    "samples": int(summary["count"]),
                    "expected_team": expected_team_from_half_side(half_idx, side == "left"),
                    "median_x": summary["median_x"],
                    "median_y": summary["median_y"],
                    "x_range": round(float(summary["x_range"]), 4),
                }
            )
    for key in slots:
        slots[key].sort(
            key=lambda item: (item["score"], item["zone_ratio"], item["samples"]),
            reverse=True,
        )
    return slots

def infer_team_for_track(tid, half_summaries):
    half_debug = []
    half_records = []

    for half_idx in (1, 2):
        summary = half_summaries[half_idx].get(tid)
        if not is_gk_candidate(summary):
            continue
        team = expected_team_from_half_side(half_idx, bool(summary["side_left"]))
        samples = int(summary["count"])
        zone_ratio = float(summary["zone_ratio"])
        weight = zone_ratio * samples

        half_records.append(
            {
                "half": int(half_idx),
                "team": team,
                "zone_ratio": zone_ratio,
                "samples": samples,
                "weight": weight,
            }
        )
        half_debug.append(
            {
                "half": int(half_idx),
                "dominant_side": "left" if summary["side_left"] else "right",
                "expected_team": team,
                "zone_ratio": round(zone_ratio, 4),
                "samples": samples,
                "median_x": summary["median_x"],
                "median_y": summary["median_y"],
                "x_range": round(float(summary["x_range"]), 4),
                "weight": round(weight, 4),
            }
        )

    if not half_records:
        return None, {
            "mode": "insufficient_goalkeeper_evidence",
            "half_debug": half_debug,
        }

    if len(half_records) == 1:
        only = half_records[0]
        if only["zone_ratio"] >= GK_HALF_LOCK_MIN_ZONE_RATIO and only["samples"] >= GK_HALF_LOCK_MIN_SAMPLES:
            return only["team"], {
                "mode": "half_side_lock_single_half",
                "team_scores": {only["team"]: round(float(only["weight"]), 4)},
                "half_debug": half_debug,
            }
        return None, {
            "mode": "single_half_weak",
            "team_scores": {only["team"]: round(float(only["weight"]), 4)},
            "half_debug": half_debug,
        }

    teams_seen = {rec["team"] for rec in half_records}
    if len(teams_seen) == 1:
        team = half_records[0]["team"]
        total_weight = sum(rec["weight"] for rec in half_records)
        return team, {
            "mode": "half_side_lock_unanimous",
            "team_scores": {team: round(float(total_weight), 4)},
            "half_debug": half_debug,
        }

    ordered = sorted(
        half_records,
        key=lambda rec: (rec["weight"], rec["zone_ratio"], rec["samples"]),
        reverse=True,
    )
    best = ordered[0]
    runner = ordered[1]
    if best["weight"] >= GK_DOMINANT_HALF_WEIGHT_RATIO * runner["weight"]:
        return best["team"], {
            "mode": "half_side_lock_dominant_half",
            "team_scores": {
                best["team"]: round(float(best["weight"]), 4),
                runner["team"]: round(float(runner["weight"]), 4),
            },
            "half_debug": half_debug,
        }

    if best["zone_ratio"] >= GK_HALF_LOCK_MIN_ZONE_RATIO and best["samples"] >= GK_HALF_LOCK_MIN_SAMPLES:
        return best["team"], {
            "mode": "half_side_lock_best_half",
            "team_scores": {
                best["team"]: round(float(best["weight"]), 4),
                runner["team"]: round(float(runner["weight"]), 4),
            },
            "half_debug": half_debug,
        }

    return None, {
        "mode": "ambiguous_between_halves",
        "team_scores": {
            rec["team"]: round(float(rec["weight"]), 4)
            for rec in ordered[:2]
        },
        "half_debug": half_debug,
    }

def main():
    args = parse_args()

    with open(args.track_labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(args.projection_json, "r", encoding="utf-8") as f:
        proj = json.load(f)
    with open(args.gameplay_json, "r", encoding="utf-8") as f:
        gameplay = json.load(f)

    tracks = labels["tracks"]
    projection_frames = proj["frames"]
    gameplay_frames = gameplay.get("frames", [])

    half_boundary_orig, half_info = detect_half_boundary(
        args.gameplay_video,
        gameplay,
        cache_path=args.half_boundary_cache,
    )

    print(f"\nStage 4b | Goalkeeper Fix - {len(projection_frames)} projection frames")
    print(
        "  half boundary:",
        half_boundary_orig,
        "| source:",
        half_info.get("source", "unknown"),
    )

    track_summary_all, half_summaries = build_track_summaries(
        projection_frames,
        gameplay_frames,
        half_boundary_orig,
        tracks,
    )
    pitch_candidates = sorted(
        tid
        for tid in set(half_summaries[1].keys()) | set(half_summaries[2].keys())
        if any(is_gk_candidate(half_summaries[half_idx].get(tid)) for half_idx in (1, 2))
    )
    eligible_ids = [tid for tid in pitch_candidates if is_fix_eligible(tracks, tid)]
    slot_leaders = collect_slot_leaders(half_summaries, set(eligible_ids))

    corrected = json.loads(json.dumps(labels))
    changes = []

    for tid in tqdm(
        eligible_ids,
        desc="Stage 4b | Assign teams",
        unit="track",
        ncols=90,
    ):
        current_label = tracks[str(tid)]["label"]
        inferred_team, debug_info = infer_team_for_track(tid, half_summaries)
        if inferred_team not in ("team_1", "team_2"):
            continue
        if inferred_team == current_label:
            continue

        corrected["tracks"][str(tid)]["label"] = inferred_team
        stats = _stats(tracks, tid)
        summary_all = track_summary_all.get(tid)
        changes.append(
            {
                "track_id": int(tid),
                "from": current_label,
                "to": inferred_team,
                "old_label": current_label,
                "new_label": inferred_team,
                "inference_mode": debug_info["mode"],
                "team_scores": debug_info.get("team_scores", {}),
                "half_debug": debug_info.get("half_debug", []),
                "median_x": round(float(summary_all["median_x"]), 3) if summary_all else None,
                "median_y": round(float(summary_all["median_y"]), 3) if summary_all else None,
                "zone_ratio": round(float(summary_all["zone_ratio"]), 4) if summary_all else None,
                "x_range": round(float(summary_all["x_range"]), 3) if summary_all else None,
                "samples": int(summary_all["count"]) if summary_all else 0,
                "blue_ratio": round(float(stats.get("blue_ratio", 0.0)), 4),
                "orange_ratio": round(float(stats.get("orange_ratio", 0.0)), 4),
            }
        )

    corrected["goalkeeper_fix"] = {
        "version": "strict_half_slots",
        "half_boundary_original_frame": half_boundary_orig,
        "half_boundary_info": half_info,
        "pitch_candidates": pitch_candidates,
        "eligible_track_ids": eligible_ids,
        "slot_leaders": slot_leaders,
        "changes": changes,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(corrected, f, ensure_ascii=False, indent=2)

    changed_ids = {ch["track_id"] for ch in changes}
    try:
        fps_hint = gameplay.get("fps")
        render_corrected_review(
            args.gameplay_video,
            gameplay_frames,
            corrected["tracks"],
            args.out_video,
            changed_ids,
            fps_hint=fps_hint,
        )
    except Exception as exc:
        print(f"  WARNING: gkfix review video could not be produced: {exc}")

    print("DONE")
    print("out     =", args.out_json)
    print("review  =", args.out_video)
    print("eligible =", len(eligible_ids))
    print("changes =", len(changes))
    for ch in changes[:50]:
        print(
            f"  track {ch['track_id']:>5}  {ch['old_label']:>10} -> {ch['new_label']}"
            f"  ({ch['inference_mode']}  scores={ch['team_scores']})"
        )
    if len(changes) > 50:
        print(f"  ... +{len(changes) - 50} more changes")

if __name__ == "__main__":
    main()
