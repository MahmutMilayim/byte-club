#!/usr/bin/env python3
"""
Stage 9 - Summary video builder from shot, free kick, and scoreboard goals.

Rules:
  - shot:      start 9s before anchor, end 6s after anchor  -> 15s clip
  - free_kick: start 6s before anchor, end 6s after anchor  -> 12s clip
  - goal:      detect score increases on raw input.mp4 scoreboard;
               start 15s before goal anchor, end 25s after   -> 40s clip

Shot / free kick anchors come from canonical analytics gameplay-frame events and are
mapped back to raw input.mp4 through stage3 gameplay.json original_frame_index.

Goal anchors come from a scoreboard-change tracker that is robust to temporary
scoreboard disappearance:
  - only visible scoreboard samples are classified,
  - non-monotonic / impossible score jumps are ignored,
  - if the scoreboard disappears between old and new score states, the anchor is
    estimated between the last old-score frame and the first new-score frame,
  - nearby shot anchors are used as a refinement signal when available.

Goal clip ending is dynamic:
  - start remains 15 seconds before the goal anchor,
  - after the new score first becomes visible, the first sustained scoreboard
    disappearance is treated as replay start,
  - the clip ends when the scoreboard returns after that replay gap,
  - if no replay gap/return is found, the old +25s fallback is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

INPUT_VIDEO = "/input/input.mp4"
ANALYTICS_JSON = "/output/stage8_analytics/analytics.json"
GAMEPLAY_JSON = "/output/stage3_filter/gameplay.json"
OUT_DIR = "/output/stage9_summary"
OUT_VIDEO = "/output/stage9_summary/summary.mp4"
OUT_MANIFEST = "/output/stage9_summary/summary_segments.json"

# Bootstrap score exemplars for the current match / broadcast graphics.
DEFAULT_SCORE_TEMPLATES = (
    "5000=0-0",
    "15000=1-0",
    "75000=2-1",
    "125000=3-3",
    "153900=4-3",
)

FULL_SCOREBOARD_X = (0.07, 0.29)
FULL_SCOREBOARD_Y = (0.025, 0.115)
SCORE_ONLY_X = (0.15, 0.25)
SCORE_ONLY_Y = (0.02, 0.10)

SCOREBOARD_RED_RATIO_MIN = 0.08
SCOREBOARD_WHITE_RATIO_MIN = 0.02
MIN_DIGIT_PIXEL_SUM = 8

DIGIT_SIZE = (24, 32)  # width, height
DIGIT_DIFF_STRONG = 12
DIGIT_DIFF_SOFT = 20
DIGIT_MAX_ACCEPT = 180

SCORE_SAMPLE_HZ = 1.0
SCORE_SMOOTH_RADIUS = 2
MIN_STABLE_SCORE_SAMPLES = 4
MIN_STABLE_SCORE_SAMPLES_FINAL = 2
SAME_SCORE_MERGE_GAP_SEC = 12.0
DIRECT_GOAL_GAP_SEC = 2.5
SHOT_REFINE_WINDOW_SEC = 10.0
SHOT_REFINE_MAX_DISTANCE_SEC = 18.0
FINAL_RUN_EOF_MARGIN_SEC = 3.0
GOAL_REPLAY_SCAN_MAX_SEC = 70.0
GOAL_REPLAY_SAMPLE_HZ = 4.0
GOAL_REPLAY_SMOOTH_RADIUS = 1
GOAL_REPLAY_MIN_MISSING_SEC = 1.0
GOAL_REPLAY_MIN_RETURN_VISIBLE_SEC = 1.0

EVENT_CLIP_RULES = {
    "shot": (9.0, 6.0),
    "free_kick": (6.0, 6.0),
    "goal": (15.0, 25.0),
}


@dataclass
class ScoreRun:
    score: tuple[int, int]
    start_frame: int
    end_frame: int
    samples: int


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-video", default=INPUT_VIDEO)
    ap.add_argument("--analytics-json", default=ANALYTICS_JSON)
    ap.add_argument("--gameplay-json", default=GAMEPLAY_JSON)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--out-video", default=OUT_VIDEO)
    ap.add_argument("--out-manifest", default=OUT_MANIFEST)
    ap.add_argument(
        "--score-template",
        action="append",
        default=[],
        help="Bootstrap score template in FRAME=L-R form, e.g. 5000=0-0",
    )
    return ap.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_template_specs(values: list[str]) -> list[tuple[int, tuple[int, int]]]:
    specs: list[tuple[int, tuple[int, int]]] = []
    raw_values = values or list(DEFAULT_SCORE_TEMPLATES)
    for raw in raw_values:
        frame_part, score_part = raw.split("=")
        left_s, right_s = score_part.split("-")
        specs.append((int(frame_part), (int(left_s), int(right_s))))
    return specs


def read_frame_at(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def clip_bounds(anchor_frame: int, pre_sec: float, post_sec: float, fps: float, total_frames: int):
    pre_frames = int(round(pre_sec * fps))
    post_frames = int(round(post_sec * fps))
    start = max(0, int(anchor_frame) - pre_frames)
    end = min(int(total_frames), int(anchor_frame) + post_frames)
    return start, end


def gameplay_anchor_to_original(anchor_gp_frame: int | None, gameplay_frames: list[dict]) -> int | None:
    if anchor_gp_frame is None:
        return None
    idx = int(round(anchor_gp_frame))
    if 0 <= idx < len(gameplay_frames):
        original = gameplay_frames[idx].get("original_frame_index")
        if original is not None:
            return int(original)
    return None


def extract_scoreboard_rois(frame: np.ndarray):
    h, w = frame.shape[:2]
    full = frame[
        int(h * FULL_SCOREBOARD_Y[0]) : int(h * FULL_SCOREBOARD_Y[1]),
        int(w * FULL_SCOREBOARD_X[0]) : int(w * FULL_SCOREBOARD_X[1]),
    ]
    score = frame[
        int(h * SCORE_ONLY_Y[0]) : int(h * SCORE_ONLY_Y[1]),
        int(w * SCORE_ONLY_X[0]) : int(w * SCORE_ONLY_X[1]),
    ]
    return full, score


def scoreboard_is_visible(full_roi: np.ndarray) -> bool:
    hsv = cv2.cvtColor(full_roi, cv2.COLOR_BGR2HSV)
    red_ratio = (
        (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 50))
    ).mean()
    white_ratio = ((hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 160)).mean()
    return bool(
        red_ratio >= SCOREBOARD_RED_RATIO_MIN and white_ratio >= SCOREBOARD_WHITE_RATIO_MIN
    )


def extract_digit_masks(score_roi: np.ndarray) -> list[np.ndarray]:
    hsv = cv2.cvtColor(score_roi, cv2.COLOR_BGR2HSV)
    red_mask = (
        (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 40))
    ).astype(np.uint8) * 255
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon_mask = np.zeros(red_mask.shape, np.uint8)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
        if cv2.contourArea(contour) < 100:
            continue
        hull = cv2.convexHull(contour)
        cv2.drawContours(polygon_mask, [hull], -1, 255, -1)

    gray = cv2.cvtColor(score_roi, cv2.COLOR_BGR2GRAY)
    digit_mask = ((gray > 145) & (polygon_mask > 0)).astype(np.uint8) * 255
    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(digit_mask, 8)

    components: list[tuple[int, np.ndarray]] = []
    for comp_idx in range(1, n_comp):
        x, y, w, h, area = stats[comp_idx]
        if area < 120 or h < 18:
            continue
        crop = digit_mask[y : y + h, x : x + w]
        components.append((int(x), crop))

    components.sort(key=lambda item: item[0])
    outputs: list[np.ndarray] = []
    for _, crop in components[:2]:
        ch, cw = crop.shape
        side = max(ch, cw) + 8
        canvas = np.zeros((side, side), np.uint8)
        y0 = (side - ch) // 2
        x0 = (side - cw) // 2
        canvas[y0 : y0 + ch, x0 : x0 + cw] = crop
        norm = cv2.resize(canvas, DIGIT_SIZE, interpolation=cv2.INTER_NEAREST)
        outputs.append((norm > 0).astype(np.uint8))
    return outputs


def build_digit_templates(cap: cv2.VideoCapture, specs: list[tuple[int, tuple[int, int]]]):
    template_samples: dict[int, list[np.ndarray]] = {}

    for frame_index, score_pair in specs:
        frame = read_frame_at(cap, frame_index)
        if frame is None:
            continue
        _, score_roi = extract_scoreboard_rois(frame)
        digit_masks = extract_digit_masks(score_roi)
        if len(digit_masks) != 2:
            continue
        for digit, mask in zip(score_pair, digit_masks):
            template_samples.setdefault(int(digit), []).append(mask)

    required_digits = sorted({int(digit) for _, score_pair in specs for digit in score_pair})
    missing = [digit for digit in required_digits if digit not in template_samples]
    if missing:
        raise RuntimeError(
            f"Score digit templates could not be bootstrapped for digits: {missing}"
        )

    templates = {}
    for digit, samples in template_samples.items():
        stack = np.stack(samples, axis=0)
        templates[int(digit)] = (stack.sum(axis=0) >= (stack.shape[0] / 2)).astype(np.uint8)
    return templates


def classify_digit(mask: np.ndarray, templates: dict[int, np.ndarray]):
    scores = []
    for digit, template in templates.items():
        dist = int(np.count_nonzero(mask != template))
        scores.append((dist, int(digit)))
    scores.sort()
    return scores[0], scores[1]


def classify_score(frame: np.ndarray, templates: dict[int, np.ndarray]):
    full_roi, score_roi = extract_scoreboard_rois(frame)
    if not scoreboard_is_visible(full_roi):
        return None

    masks = extract_digit_masks(score_roi)
    if len(masks) != 2:
        return None

    digits = []
    total_best = 0
    for mask in masks:
        best, second = classify_digit(mask, templates)
        total_best += best[0]
        if best[0] > DIGIT_MAX_ACCEPT:
            return None
        digits.append(int(best[1]))
        if best[0] > DIGIT_DIFF_SOFT and (second[0] - best[0]) < 3:
            return None

    return {
        "score": (digits[0], digits[1]),
        "best_distance_sum": int(total_best),
    }


def sample_score_timeline(input_video_path: str, templates: dict[int, np.ndarray], fps: float):
    stride = max(1, int(round(fps / SCORE_SAMPLE_HZ)))
    samples = []

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video cannot be opened: {input_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = -1

    with tqdm(total=max(total_frames, 1), desc="Stage 9 | Scoreboard scan", unit="frame", ncols=90) as pbar:
        while True:
            ok = cap.grab()
            if not ok:
                break
            frame_index += 1
            pbar.update(1)
            if frame_index % stride != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            rec = classify_score(frame, templates)
            samples.append(
                {
                    "frame": int(frame_index),
                    "score": rec["score"] if rec else None,
                    "distance_sum": rec["best_distance_sum"] if rec else None,
                }
            )
    cap.release()
    return samples, stride


def smooth_score_samples(samples: list[dict]):
    smoothed = []
    radius = SCORE_SMOOTH_RADIUS
    for idx, sample in enumerate(samples):
        window_scores = [
            tuple(other["score"])
            for other in samples[max(0, idx - radius) : min(len(samples), idx + radius + 1)]
            if other["score"] is not None
        ]
        if not window_scores:
            score = None
        else:
            score = Counter(window_scores).most_common(1)[0][0]
        smoothed.append({"frame": int(sample["frame"]), "score": score})
    return smoothed


def sample_scoreboard_visibility(input_video_path: str, start_frame: int, end_frame: int, fps: float):
    stride = max(1, int(round(fps / GOAL_REPLAY_SAMPLE_HZ)))
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video cannot be opened: {input_video_path}")

    samples = []
    frame_index = max(0, int(start_frame))
    end_frame = max(frame_index, int(end_frame))
    while frame_index <= end_frame:
        frame = read_frame_at(cap, frame_index)
        if frame is None:
            break
        full_roi, _ = extract_scoreboard_rois(frame)
        samples.append(
            {
                "frame": int(frame_index),
                "visible": bool(scoreboard_is_visible(full_roi)),
            }
        )
        frame_index += stride
    cap.release()
    return samples, stride


def smooth_visibility_samples(samples: list[dict]):
    smoothed = []
    radius = GOAL_REPLAY_SMOOTH_RADIUS
    for idx, sample in enumerate(samples):
        window = samples[max(0, idx - radius) : min(len(samples), idx + radius + 1)]
        visible_votes = sum(1 for other in window if other["visible"])
        invisible_votes = len(window) - visible_votes
        smoothed.append(
            {
                "frame": int(sample["frame"]),
                "visible": visible_votes >= invisible_votes,
            }
        )
    return smoothed


def compress_visibility_runs(samples: list[dict], sample_stride: int):
    runs = []
    current = None
    for sample in samples:
        visible = bool(sample["visible"])
        frame = int(sample["frame"])
        if current is None:
            current = {
                "visible": visible,
                "start_frame": frame,
                "end_frame": frame,
                "samples": 1,
            }
            continue
        if visible == current["visible"] and frame - int(current["end_frame"]) <= sample_stride * 2:
            current["end_frame"] = frame
            current["samples"] = int(current["samples"]) + 1
        else:
            runs.append(current)
            current = {
                "visible": visible,
                "start_frame": frame,
                "end_frame": frame,
                "samples": 1,
            }
    if current is not None:
        runs.append(current)
    return runs


def refine_goal_end_by_replay_return(
    input_video_path: str,
    goal_transition: dict,
    fps: float,
    total_frames: int,
):
    fallback_end = int(goal_transition["end_input_frame"])
    search_start = max(
        int(goal_transition["anchor_input_frame"]),
        int(goal_transition["new_visible_score_frame"]),
    )
    search_end = min(
        int(total_frames) - 1,
        search_start + int(round(GOAL_REPLAY_SCAN_MAX_SEC * fps)),
    )
    if search_end <= search_start:
        return fallback_end, {
            "mode": "fallback_invalid_window",
            "search_start_frame": search_start,
            "search_end_frame": search_end,
        }

    samples, sample_stride = sample_scoreboard_visibility(
        input_video_path,
        search_start,
        search_end,
        fps,
    )
    if not samples:
        return fallback_end, {
            "mode": "fallback_no_samples",
            "search_start_frame": search_start,
            "search_end_frame": search_end,
        }

    smoothed = smooth_visibility_samples(samples)
    runs = compress_visibility_runs(smoothed, sample_stride)

    min_missing_frames = int(round(GOAL_REPLAY_MIN_MISSING_SEC * fps))
    min_return_frames = int(round(GOAL_REPLAY_MIN_RETURN_VISIBLE_SEC * fps))

    seen_visible = False
    replay_gap = None
    replay_return = None

    for run in runs:
        duration_frames = int(run["end_frame"]) - int(run["start_frame"]) + sample_stride
        if not seen_visible:
            if run["visible"] and duration_frames >= min_return_frames:
                seen_visible = True
            continue
        if replay_gap is None:
            if (not run["visible"]) and duration_frames >= min_missing_frames:
                replay_gap = run
            continue
        if run["visible"] and duration_frames >= min_return_frames:
            replay_return = run
            break

    if replay_gap is None or replay_return is None:
        return fallback_end, {
            "mode": "fallback_no_replay_return",
            "search_start_frame": search_start,
            "search_end_frame": search_end,
            "sample_stride": sample_stride,
            "runs": runs,
        }

    refined_end = max(
        int(goal_transition["start_input_frame"]) + 1,
        min(int(total_frames), int(replay_return["start_frame"])),
    )
    return refined_end, {
        "mode": "scoreboard_replay_return",
        "search_start_frame": search_start,
        "search_end_frame": search_end,
        "sample_stride": sample_stride,
        "replay_gap_start_frame": int(replay_gap["start_frame"]),
        "replay_gap_end_frame": int(replay_gap["end_frame"]),
        "scoreboard_return_frame": int(replay_return["start_frame"]),
        "runs": runs,
    }


def compress_score_runs(smoothed_samples: list[dict], sample_stride: int, fps: float, total_frames: int):
    runs: list[ScoreRun] = []
    current: ScoreRun | None = None

    for sample in smoothed_samples:
        score = sample["score"]
        if score is None:
            continue
        if current is None:
            current = ScoreRun(tuple(score), int(sample["frame"]), int(sample["frame"]), 1)
            continue

        if tuple(score) == current.score and int(sample["frame"]) - current.end_frame <= sample_stride * 3:
            current.end_frame = int(sample["frame"])
            current.samples += 1
        else:
            runs.append(current)
            current = ScoreRun(tuple(score), int(sample["frame"]), int(sample["frame"]), 1)

    if current is not None:
        runs.append(current)

    merged: list[ScoreRun] = []
    max_merge_gap = int(round(SAME_SCORE_MERGE_GAP_SEC * sample_stride / max(sample_stride, 1)))
    max_merge_gap_frames = max_merge_gap * sample_stride

    final_margin_frames = int(round(FINAL_RUN_EOF_MARGIN_SEC * fps))

    for run in runs:
        min_samples = MIN_STABLE_SCORE_SAMPLES
        if run.end_frame >= max(0, total_frames - final_margin_frames):
            min_samples = MIN_STABLE_SCORE_SAMPLES_FINAL
        if run.samples < min_samples:
            continue
        if (
            merged
            and merged[-1].score == run.score
            and run.start_frame - merged[-1].end_frame <= max_merge_gap_frames
        ):
            merged[-1].end_frame = run.end_frame
            merged[-1].samples += run.samples
        else:
            merged.append(run)
    return merged


def is_single_goal_increment(prev_score: tuple[int, int], next_score: tuple[int, int]) -> bool:
    dl = next_score[0] - prev_score[0]
    dr = next_score[1] - prev_score[1]
    return (dl, dr) in ((1, 0), (0, 1))


def detect_goal_segments(score_runs: list[ScoreRun], shot_anchor_frames: list[int], fps: float, total_frames: int):
    if not score_runs:
        return [], {"accepted_runs": [], "transitions": []}

    accepted_runs: list[ScoreRun] = [score_runs[0]]
    transitions = []

    for run in score_runs[1:]:
        current = accepted_runs[-1]
        if run.score == current.score:
            accepted_runs[-1] = ScoreRun(
                score=current.score,
                start_frame=current.start_frame,
                end_frame=run.end_frame,
                samples=current.samples + run.samples,
            )
            continue

        if not is_single_goal_increment(current.score, run.score):
            if run.score[0] <= current.score[0] and run.score[1] <= current.score[1]:
                continue
            continue

        gap_frames = run.start_frame - current.end_frame
        direct_gap = int(round(DIRECT_GOAL_GAP_SEC * fps))
        anchor_frame = int(run.start_frame)
        anchor_mode = "scoreboard_new_state"
        if gap_frames > direct_gap:
            anchor_frame = int(round((current.end_frame + run.start_frame) / 2))
            anchor_mode = "scoreboard_gap_midpoint"

        search_lo = max(0, current.end_frame - int(round(SHOT_REFINE_WINDOW_SEC * fps)))
        search_hi = min(total_frames - 1, run.start_frame + int(round(SHOT_REFINE_WINDOW_SEC * fps)))
        nearby_shots = [fr for fr in shot_anchor_frames if search_lo <= fr <= search_hi]
        if nearby_shots:
            best_shot = min(nearby_shots, key=lambda fr: abs(fr - anchor_frame))
            if abs(best_shot - anchor_frame) <= int(round(SHOT_REFINE_MAX_DISTANCE_SEC * fps)):
                anchor_frame = int(best_shot)
                anchor_mode = "scoreboard_refined_by_shot"

        start_frame, end_frame = clip_bounds(
            anchor_frame,
            EVENT_CLIP_RULES["goal"][0],
            EVENT_CLIP_RULES["goal"][1],
            fps,
            total_frames,
        )
        transitions.append(
            {
                "type": "goal",
                "score_before": list(current.score),
                "score_after": list(run.score),
                "anchor_input_frame": int(anchor_frame),
                "start_input_frame": int(start_frame),
                "end_input_frame": int(end_frame),
                "anchor_mode": anchor_mode,
                "previous_visible_score_frame": int(current.end_frame),
                "new_visible_score_frame": int(run.start_frame),
                "gap_frames": int(gap_frames),
                "gap_sec": round(gap_frames / fps, 3),
            }
        )
        accepted_runs.append(run)

    return transitions, {
        "accepted_runs": [
            {
                "score": list(run.score),
                "start_frame": int(run.start_frame),
                "end_frame": int(run.end_frame),
                "samples": int(run.samples),
            }
            for run in accepted_runs
        ],
        "all_runs": [
            {
                "score": list(run.score),
                "start_frame": int(run.start_frame),
                "end_frame": int(run.end_frame),
                "samples": int(run.samples),
            }
            for run in score_runs
        ],
        "transitions": transitions,
    }


def collect_event_segments(analytics: dict, gameplay_frames: list[dict], fps: float, total_frames: int):
    segments = []
    shot_anchor_frames = []

    events = sorted(
        (ev for ev in analytics.get("events", []) if ev.get("type") in ("shot", "free_kick")),
        key=lambda ev: (float(ev.get("time_sec", 0.0)), int(ev.get("frame", 0))),
    )

    counters = Counter()
    for ev in events:
        event_type = str(ev["type"])
        original_anchor = gameplay_anchor_to_original(ev.get("frame"), gameplay_frames)
        if original_anchor is None:
            continue
        counters[event_type] += 1
        pre_sec, post_sec = EVENT_CLIP_RULES[event_type]
        start_frame, end_frame = clip_bounds(original_anchor, pre_sec, post_sec, fps, total_frames)

        segment = {
            "id": f"{event_type}_{counters[event_type]:03d}",
            "type": event_type,
            "source": "analytics",
            "anchor_gameplay_frame": int(ev.get("frame", 0)),
            "anchor_input_frame": int(original_anchor),
            "start_input_frame": int(start_frame),
            "end_input_frame": int(end_frame),
            "clip_duration_sec": round((end_frame - start_frame) / fps, 3),
            "time_sec": round(original_anchor / fps, 3),
            "metadata": {
                "team": ev.get("team"),
                "player": ev.get("player"),
                "zone": ev.get("zone"),
                "confidence_spotting": ev.get("confidence_spotting"),
                "raw_label": ev.get("raw_label"),
                "xG": ev.get("xG"),
            },
        }
        segments.append(segment)
        if event_type == "shot":
            shot_anchor_frames.append(int(original_anchor))

    return segments, sorted(shot_anchor_frames)


def render_summary_video(input_video_path: str, segments: list[dict], out_video_path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(out_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video cannot be opened: {input_video_path}")

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Summary writer could not be opened: {out_video_path}")

    total_frames = sum(max(0, int(seg["end_input_frame"]) - int(seg["start_input_frame"])) for seg in segments)
    with tqdm(total=max(total_frames, 1), desc="Stage 9 | Render summary", unit="frame", ncols=90) as pbar:
        for segment in segments:
            start_frame = int(segment["start_input_frame"])
            end_frame = int(segment["end_input_frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx < end_frame:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(frame)
                frame_idx += 1
                pbar.update(1)

    writer.release()
    cap.release()


def main():
    args = parse_args()

    analytics = load_json(args.analytics_json)
    gameplay = load_json(args.gameplay_json)
    gameplay_frames = gameplay.get("frames", [])

    input_video_path = Path(args.input_video)
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Input video cannot be opened: {input_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    template_specs = parse_template_specs(args.score_template)
    templates = build_digit_templates(cap, template_specs)
    cap.release()

    event_segments, shot_anchor_frames = collect_event_segments(
        analytics,
        gameplay_frames,
        fps,
        total_frames,
    )

    score_samples, sample_stride = sample_score_timeline(str(input_video_path), templates, fps)
    smoothed_samples = smooth_score_samples(score_samples)
    score_runs = compress_score_runs(smoothed_samples, sample_stride, fps, total_frames)
    goal_segments, goal_debug = detect_goal_segments(score_runs, shot_anchor_frames, fps, total_frames)

    counters = Counter(seg["type"] for seg in event_segments)
    for goal_idx, goal in enumerate(goal_segments, start=1):
        counters["goal"] += 1
        refined_end_frame, replay_debug = refine_goal_end_by_replay_return(
            str(input_video_path),
            goal,
            fps,
            total_frames,
        )
        event_segments.append(
            {
                "id": f"goal_{goal_idx:03d}",
                "type": "goal",
                "source": "scoreboard_change",
                "anchor_gameplay_frame": None,
                "anchor_input_frame": int(goal["anchor_input_frame"]),
                "start_input_frame": int(goal["start_input_frame"]),
                "end_input_frame": int(refined_end_frame),
                "clip_duration_sec": round(
                    (int(refined_end_frame) - int(goal["start_input_frame"])) / fps,
                    3,
                ),
                "time_sec": round(int(goal["anchor_input_frame"]) / fps, 3),
                "metadata": {
                    "score_before": goal["score_before"],
                    "score_after": goal["score_after"],
                    "anchor_mode": goal["anchor_mode"],
                    "previous_visible_score_frame": goal["previous_visible_score_frame"],
                    "new_visible_score_frame": goal["new_visible_score_frame"],
                    "gap_sec": goal["gap_sec"],
                    "goal_post_mode": replay_debug["mode"],
                    "goal_post_fallback_end_frame": int(goal["end_input_frame"]),
                    "goal_post_search_start_frame": int(replay_debug["search_start_frame"]),
                    "goal_post_search_end_frame": int(replay_debug["search_end_frame"]),
                    "goal_post_scoreboard_return_frame": replay_debug.get("scoreboard_return_frame"),
                    "goal_post_replay_gap_start_frame": replay_debug.get("replay_gap_start_frame"),
                    "goal_post_replay_gap_end_frame": replay_debug.get("replay_gap_end_frame"),
                },
            }
        )
        goal["dynamic_end_debug"] = replay_debug
        goal["refined_end_input_frame"] = int(refined_end_frame)

    ordered_segments = sorted(
        event_segments,
        key=lambda seg: (int(seg["anchor_input_frame"]), {"shot": 0, "free_kick": 1, "goal": 2}.get(seg["type"], 9)),
    )

    timeline_frame = 0
    for idx, segment in enumerate(ordered_segments, start=1):
        duration_frames = max(0, int(segment["end_input_frame"]) - int(segment["start_input_frame"]))
        segment["order_index"] = idx
        segment["summary_start_frame"] = int(timeline_frame)
        segment["summary_end_frame"] = int(timeline_frame + duration_frames)
        timeline_frame += duration_frames

    render_summary_video(
        str(input_video_path),
        ordered_segments,
        args.out_video,
        fps,
        width,
        height,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_video": str(input_video_path),
        "analytics_json": str(args.analytics_json),
        "gameplay_json": str(args.gameplay_json),
        "fps": fps,
        "width": width,
        "height": height,
        "total_input_frames": total_frames,
        "summary_total_frames": int(timeline_frame),
        "summary_total_sec": round(timeline_frame / fps, 3),
        "score_template_specs": [
            {"frame": int(frame), "score": list(score)}
            for frame, score in template_specs
        ],
        "goal_detection": goal_debug,
        "segments": ordered_segments,
    }
    with open(args.out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nStage 9 | Summary Builder")
    print("  input   =", input_video_path)
    print("  output  =", args.out_video)
    print("  manifest=", args.out_manifest)
    print("  segments=", len(ordered_segments))
    print(
        "  counts  =",
        dict(Counter(seg["type"] for seg in ordered_segments)),
    )
    print("  summary_sec =", round(timeline_frame / fps, 3))


if __name__ == "__main__":
    main()
