import argparse
import json
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from field_utils import (
    build_bev_pitch_template,
    estimate_delta_h,
    extract_pitch_evidence,
    homography_disagreement_m,
    invert_homography,
    normalize_homography,
    refine_homography_bev_ecc,
    score_candidate,
)


VIDEO_PATH = "/output/stage3_filter/gameplay.mp4"
BASE_HMAP_PATH = "/output/stage6_field/homography_map.json"
CAMERA_CUTS_PATH = "/output/stage5_ball/camera_cuts.json"
OUT_DIR = "/output/stage6_field"
OUT_JSON_NAME = "homography_map_refined.json"

RESCUE_SCORE = 0.16
WIDE_RESCUE_SCORE = 0.14
TEMPORAL_SCORE_MARGIN = 0.015
TEMPORAL_DISAGREE_KEEP_M = 6.0
SEVERE_ANCHOR_DISAGREE_M = 9.0
ECC_MIN_SCORE = 0.12
ECC_MIN_LINE_PIXELS = 6000

NEAR_RADIUS = 35
NEAR_MAX_COUNT = 8
WIDE_RADIUS = 450
WIDE_MAX_COUNT = 14
LOCAL_RESCUE_RADIUS = 30
LOCAL_RESCUE_MAX_COUNT = 8


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-path", default=VIDEO_PATH)
    ap.add_argument("--base-hmap-path", default=BASE_HMAP_PATH)
    ap.add_argument("--camera-cuts-path", default=CAMERA_CUTS_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--out-name", default=OUT_JSON_NAME)
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-end", type=int, default=-1)
    ap.add_argument("--disable-ecc", action="store_true")
    return ap.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_homography(fr, key):
    H = fr.get(key)
    if H is None:
        return None
    return normalize_homography(H)


def frame_range_from_args(total_frames, frame_start, frame_end):
    start = max(0, int(frame_start))
    end = total_frames - 1 if frame_end is None or int(frame_end) < 0 else min(total_frames - 1, int(frame_end))
    if end < start:
        raise ValueError(f"Invalid frame range: {start}..{end}")
    return start, end


def extract_exact_anchors(base_frames):
    anchors = []
    seen = set()
    for fr in base_frames:
        frame_index = int(fr["frame_index"])
        src_index = fr.get("source_original_frame_index")
        H_img_to_pitch = as_homography(fr, "H_img_to_pitch")
        H_pitch_to_img = as_homography(fr, "H_pitch_to_img")
        if H_img_to_pitch is None or H_pitch_to_img is None:
            continue
        if src_index is None or int(src_index) != frame_index:
            continue
        if frame_index in seen:
            continue
        seen.add(frame_index)
        anchors.append({
            "frame_index": frame_index,
            "source_keyframe_index": fr.get("source_keyframe_index"),
            "source_original_frame_index": int(src_index),
            "source_image_path": fr.get("source_image_path"),
            "source_rep_err": fr.get("source_rep_err"),
            "source_sanity_score": fr.get("source_sanity_score"),
            "H_img_to_pitch": H_img_to_pitch,
            "H_pitch_to_img": H_pitch_to_img,
            "final_params_dict": fr.get("final_params_dict"),
        })
    anchors.sort(key=lambda x: x["frame_index"])
    if not anchors:
        raise RuntimeError("No exact anchors could be extracted from base homography map")
    return anchors


def choose_anchor_subset(anchor_candidates, frame_index, max_count):
    if len(anchor_candidates) <= max_count:
        return list(anchor_candidates)

    sorted_by_distance = sorted(anchor_candidates, key=lambda item: (abs(item["frame_index"] - frame_index), item["frame_index"]))
    keep_nearest = min(4, max_count)
    out = []
    used = set()

    for item in sorted_by_distance[:keep_nearest]:
        fid = item["frame_index"]
        if fid not in used:
            used.add(fid)
            out.append(item)

    remaining_slots = max_count - len(out)
    if remaining_slots <= 0:
        return out

    if remaining_slots == 1:
        middle = anchor_candidates[len(anchor_candidates) // 2]
        if middle["frame_index"] not in used:
            out.append(middle)
        return out[:max_count]

    for pos in np.linspace(0, len(anchor_candidates) - 1, remaining_slots):
        item = anchor_candidates[int(round(float(pos)))]
        fid = item["frame_index"]
        if fid in used:
            continue
        used.add(fid)
        out.append(item)
        if len(out) >= max_count:
            break

    if len(out) < max_count:
        for item in anchor_candidates:
            fid = item["frame_index"]
            if fid in used:
                continue
            used.add(fid)
            out.append(item)
            if len(out) >= max_count:
                break

    return out[:max_count]


class AnchorIndex:
    def __init__(self, anchors):
        self.anchors = anchors
        self.positions = [item["frame_index"] for item in anchors]

    def window(self, frame_index, radius):
        lo = bisect_left(self.positions, frame_index - radius)
        hi = bisect_right(self.positions, frame_index + radius)
        return self.anchors[lo:hi]

    def nearest(self, frame_index):
        idx = bisect_left(self.positions, frame_index)
        candidates = []
        if idx < len(self.anchors):
            candidates.append(self.anchors[idx])
        if idx > 0:
            candidates.append(self.anchors[idx - 1])
        if not candidates:
            raise RuntimeError("Anchor index is empty")
        return min(candidates, key=lambda item: abs(item["frame_index"] - frame_index))

    def nearby(self, frame_index, radius, max_count):
        items = self.window(frame_index, radius)
        if not items:
            return [self.nearest(frame_index)]
        return choose_anchor_subset(items, frame_index, max_count)


def build_candidate(label, H_img_to_pitch, H_pitch_to_img, source_anchor=None, blend_alpha=None):
    H_img_to_pitch = normalize_homography(H_img_to_pitch)
    H_pitch_to_img = normalize_homography(H_pitch_to_img)
    if H_img_to_pitch is None and H_pitch_to_img is not None:
        H_img_to_pitch = invert_homography(H_pitch_to_img)
    if H_pitch_to_img is None and H_img_to_pitch is not None:
        H_pitch_to_img = invert_homography(H_img_to_pitch)
    if H_img_to_pitch is None or H_pitch_to_img is None:
        return None

    payload = {
        "label": label,
        "H_img_to_pitch": H_img_to_pitch,
        "H_pitch_to_img": H_pitch_to_img,
        "blend_alpha": blend_alpha,
        "source_anchor": source_anchor,
    }
    if source_anchor is not None:
        payload.update({
            "source_keyframe_index": source_anchor.get("source_keyframe_index"),
            "source_original_frame_index": source_anchor.get("source_original_frame_index"),
            "source_image_path": source_anchor.get("source_image_path"),
            "source_rep_err": source_anchor.get("source_rep_err"),
            "source_sanity_score": source_anchor.get("source_sanity_score"),
            "final_params_dict": source_anchor.get("final_params_dict"),
        })
    else:
        payload.update({
            "source_keyframe_index": None,
            "source_original_frame_index": None,
            "source_image_path": None,
            "source_rep_err": None,
            "source_sanity_score": None,
            "final_params_dict": None,
        })
    return payload


def blend_homographies(H1, H2, alpha):
    H1 = normalize_homography(H1)
    H2 = normalize_homography(H2)
    if H1 is None:
        return H2
    if H2 is None:
        return H1
    return normalize_homography((1.0 - alpha) * H1 + alpha * H2)


def evaluate_candidate(candidate, evidence, frame_shape, base_H_img_to_pitch, prev_refined_H_img_to_pitch):
    metrics = score_candidate(evidence, candidate["H_pitch_to_img"], frame_shape)
    metrics["candidate_label"] = candidate["label"]
    metrics["source_original_frame_index"] = candidate.get("source_original_frame_index")
    metrics["source_keyframe_index"] = candidate.get("source_keyframe_index")
    metrics["blend_alpha"] = candidate.get("blend_alpha")
    metrics["selected_vs_base_disagree_m"] = homography_disagreement_m(
        candidate["H_img_to_pitch"], base_H_img_to_pitch, frame_shape[1], frame_shape[0]
    )
    metrics["selected_vs_prev_disagree_m"] = homography_disagreement_m(
        candidate["H_img_to_pitch"], prev_refined_H_img_to_pitch, frame_shape[1], frame_shape[0]
    )
    metrics["H_img_to_pitch"] = candidate["H_img_to_pitch"]
    metrics["H_pitch_to_img"] = candidate["H_pitch_to_img"]
    metrics["source_image_path"] = candidate.get("source_image_path")
    metrics["source_rep_err"] = candidate.get("source_rep_err")
    metrics["source_sanity_score"] = candidate.get("source_sanity_score")
    metrics["final_params_dict"] = candidate.get("final_params_dict")
    return metrics


def choose_best_candidate(scored_candidates, camera_cut_hint):
    best = max(scored_candidates, key=lambda item: (item["score"], item["line_precision"], item["field_green_ratio"]))
    if camera_cut_hint:
        return best, False

    propagated = next((item for item in scored_candidates if item["candidate_label"] == "propagated_prev_refined"), None)
    if propagated is None:
        return best, False

    disagree_to_prev = propagated.get("selected_vs_prev_disagree_m")
    if disagree_to_prev is not None and disagree_to_prev > TEMPORAL_DISAGREE_KEEP_M:
        return best, False

    if propagated["score"] + TEMPORAL_SCORE_MARGIN >= best["score"]:
        return propagated, propagated is not best
    return best, False


def frame_is_cut(frame_index, camera_cuts, segment_cuts):
    return frame_index in camera_cuts or frame_index in segment_cuts


def main():
    args = parse_args()

    base_hmap = load_json(args.base_hmap_path)
    base_frames = base_hmap["frames"]
    total_frames = int(base_hmap.get("total_frames") or len(base_frames))
    frame_start, frame_end = frame_range_from_args(total_frames, args.frame_start, args.frame_end)

    cuts_json = load_json(args.camera_cuts_path)
    camera_cuts = set(int(x) for x in cuts_json.get("camera_cuts", []))
    segment_cuts = set(int(x) for x in cuts_json.get("segment_cuts", []))

    anchors = extract_exact_anchors(base_frames)
    anchor_index = AnchorIndex(anchors)
    bev_template, S_pitch_to_bev = build_bev_pitch_template(bev_scale=10, pad=24)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {args.video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    ok, prev_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Frame could not be read: {frame_start}")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_shape = prev_frame.shape

    out_frames = []
    prev_refined_H_img_to_pitch = None
    prev_selected_anchor = None

    print(f"\nStage 6 | Refine Homography - frames {frame_start}..{frame_end}")
    for frame_index in tqdm(range(frame_start, frame_end + 1), desc="Stage 6 | Refine", unit="frame", ncols=90):
        if frame_index == frame_start:
            frame_bgr = prev_frame
            cur_gray = prev_gray
            flow_meta = {
                "tracked_points": None,
                "fb_good_points": None,
                "inliers": None,
                "inlier_ratio": None,
                "flow_confidence": None,
                "median_flow_px": None,
            }
            delta_prev_to_cur = None
        else:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            cur_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            delta_prev_to_cur, flow_meta = estimate_delta_h(prev_gray, cur_gray)

        base_fr = base_frames[frame_index]
        base_H_img_to_pitch = as_homography(base_fr, "H_img_to_pitch")
        base_H_pitch_to_img = as_homography(base_fr, "H_pitch_to_img")

        evidence = extract_pitch_evidence(frame_bgr)
        camera_cut_hint = frame_is_cut(frame_index, camera_cuts, segment_cuts)

        candidates = []
        base_source_anchor = {
            "source_keyframe_index": base_fr.get("source_keyframe_index"),
            "source_original_frame_index": base_fr.get("source_original_frame_index"),
            "source_image_path": base_fr.get("source_image_path"),
            "source_rep_err": base_fr.get("source_rep_err"),
            "source_sanity_score": base_fr.get("source_sanity_score"),
            "final_params_dict": base_fr.get("final_params_dict"),
        }
        base_candidate = build_candidate("base_current", base_H_img_to_pitch, base_H_pitch_to_img, source_anchor=base_source_anchor)
        if base_candidate is not None:
            candidates.append(base_candidate)

        propagated_H_img_to_pitch = None
        if prev_refined_H_img_to_pitch is not None and delta_prev_to_cur is not None:
            delta_inv = invert_homography(delta_prev_to_cur)
            if delta_inv is not None:
                propagated_H_img_to_pitch = normalize_homography(prev_refined_H_img_to_pitch @ delta_inv)
                propagated_candidate = build_candidate("propagated_prev_refined", propagated_H_img_to_pitch, None, source_anchor=prev_selected_anchor)
                if propagated_candidate is not None:
                    candidates.append(propagated_candidate)

        for anchor in anchor_index.nearby(frame_index, NEAR_RADIUS, NEAR_MAX_COUNT):
            cand = build_candidate("anchor_near", anchor["H_img_to_pitch"], anchor["H_pitch_to_img"], source_anchor=anchor)
            if cand is not None:
                candidates.append(cand)

        if propagated_H_img_to_pitch is not None:
            nearest_anchor = anchor_index.nearby(frame_index, NEAR_RADIUS, 1)[0]
            for alpha in (0.25, 0.50):
                blended = blend_homographies(propagated_H_img_to_pitch, nearest_anchor["H_img_to_pitch"], alpha)
                cand = build_candidate("blend_prop_anchor", blended, None, source_anchor=nearest_anchor, blend_alpha=alpha)
                if cand is not None:
                    candidates.append(cand)

        scored = []
        seen = set()
        for cand in candidates:
            key = (
                cand["label"],
                cand.get("source_original_frame_index"),
                round(float(cand.get("blend_alpha") or 0.0), 2),
            )
            if key in seen:
                continue
            seen.add(key)
            scored.append(evaluate_candidate(cand, evidence, frame_shape, base_H_img_to_pitch, prev_refined_H_img_to_pitch))

        if not scored:
            raise RuntimeError(f"No valid candidate at frame {frame_index}")

        base_scored = next((item for item in scored if item["candidate_label"] == "base_current"), None)
        best_initial = max(scored, key=lambda item: item["score"])
        base_anchor_disagree = base_fr.get("anchor_disagree_m")
        need_rescue = (
            camera_cut_hint or
            (base_anchor_disagree is not None and float(base_anchor_disagree) > TEMPORAL_DISAGREE_KEEP_M) or
            (base_fr.get("map_source") or "").startswith("anchor_fallback") or
            (best_initial["score"] < RESCUE_SCORE and (base_scored is None or (base_scored["score"] < RESCUE_SCORE)))
        )

        rescue_used = False
        wide_rescue_used = False
        need_wide_rescue = (
            need_rescue and (
                camera_cut_hint or
                (base_anchor_disagree is not None and float(base_anchor_disagree) > SEVERE_ANCHOR_DISAGREE_M) or
                best_initial["score"] < WIDE_RESCUE_SCORE or
                (base_fr.get("map_source") or "").startswith("anchor_fallback")
            )
        )
        if need_wide_rescue:
            rescue_used = True
            wide_rescue_used = True
            best_wide = None
            for anchor in anchor_index.nearby(frame_index, WIDE_RADIUS, WIDE_MAX_COUNT):
                cand = build_candidate("anchor_wide", anchor["H_img_to_pitch"], anchor["H_pitch_to_img"], source_anchor=anchor)
                if cand is None:
                    continue
                scored_cand = evaluate_candidate(cand, evidence, frame_shape, base_H_img_to_pitch, prev_refined_H_img_to_pitch)
                scored.append(scored_cand)
                if best_wide is None or scored_cand["score"] > best_wide["score"]:
                    best_wide = scored_cand

            if best_wide is not None:
                local_center = int(best_wide.get("source_original_frame_index") or frame_index)
                for anchor in anchor_index.nearby(local_center, LOCAL_RESCUE_RADIUS, LOCAL_RESCUE_MAX_COUNT):
                    cand = build_candidate("anchor_local_rescue", anchor["H_img_to_pitch"], anchor["H_pitch_to_img"], source_anchor=anchor)
                    if cand is None:
                        continue
                    scored.append(evaluate_candidate(cand, evidence, frame_shape, base_H_img_to_pitch, prev_refined_H_img_to_pitch))

        best_selected, temporal_preferred = choose_best_candidate(scored, camera_cut_hint)

        ecc_meta = {"ecc_success": False, "ecc_score": None, "ecc_displacement_m": None}
        final_candidate = best_selected
        if (
            not args.disable_ecc and
            evidence.line_pixel_count >= ECC_MIN_LINE_PIXELS and
            best_selected["score"] >= ECC_MIN_SCORE
        ):
            refined_H_img_to_pitch, ecc_meta = refine_homography_bev_ecc(
                best_selected["H_img_to_pitch"],
                evidence,
                frame_shape,
                bev_template,
                S_pitch_to_bev,
            )
            if refined_H_img_to_pitch is not None:
                refined_candidate = build_candidate(
                    "ecc_refined",
                    refined_H_img_to_pitch,
                    None,
                    source_anchor={
                        "source_keyframe_index": best_selected.get("source_keyframe_index"),
                        "source_original_frame_index": best_selected.get("source_original_frame_index"),
                        "source_image_path": best_selected.get("source_image_path"),
                        "source_rep_err": best_selected.get("source_rep_err"),
                        "source_sanity_score": best_selected.get("source_sanity_score"),
                        "final_params_dict": best_selected.get("final_params_dict"),
                    },
                )
                if refined_candidate is not None:
                    refined_scored = evaluate_candidate(refined_candidate, evidence, frame_shape, base_H_img_to_pitch, prev_refined_H_img_to_pitch)
                    if refined_scored["score"] >= best_selected["score"] + 0.01:
                        final_candidate = refined_scored

        prev_refined_H_img_to_pitch = final_candidate["H_img_to_pitch"]
        prev_selected_anchor = {
            "source_keyframe_index": final_candidate.get("source_keyframe_index"),
            "source_original_frame_index": final_candidate.get("source_original_frame_index"),
            "source_image_path": final_candidate.get("source_image_path"),
            "source_rep_err": final_candidate.get("source_rep_err"),
            "source_sanity_score": final_candidate.get("source_sanity_score"),
            "final_params_dict": final_candidate.get("final_params_dict"),
        }

        out_frames.append({
            "frame_index": frame_index,
            "source_keyframe_index": final_candidate.get("source_keyframe_index"),
            "source_original_frame_index": final_candidate.get("source_original_frame_index"),
            "source_image_path": final_candidate.get("source_image_path"),
            "source_rep_err": final_candidate.get("source_rep_err"),
            "source_sanity_score": final_candidate.get("source_sanity_score"),
            "H_img_to_pitch": final_candidate["H_img_to_pitch"].tolist(),
            "H_pitch_to_img": final_candidate["H_pitch_to_img"].tolist(),
            "final_params_dict": final_candidate.get("final_params_dict"),
            "base_map_source": base_fr.get("map_source"),
            "selection_source": final_candidate["candidate_label"],
            "selection_blend_alpha": final_candidate.get("blend_alpha"),
            "camera_cut_hint": camera_cut_hint,
            "rescue_search_used": rescue_used,
            "wide_rescue_used": wide_rescue_used,
            "temporal_preferred": temporal_preferred,
            "base_score": None if base_scored is None else float(base_scored["score"]),
            "final_score": float(final_candidate["score"]),
            "line_precision": float(final_candidate["line_precision"]),
            "line_recall": float(final_candidate["line_recall"]),
            "field_green_ratio": float(final_candidate["field_green_ratio"]),
            "evidence_line_pixel_count": int(evidence.line_pixel_count),
            "evidence_green_ratio": float(evidence.green_ratio),
            "selected_vs_base_disagree_m": final_candidate.get("selected_vs_base_disagree_m"),
            "selected_vs_prev_disagree_m": final_candidate.get("selected_vs_prev_disagree_m"),
            "base_anchor_disagree_m": base_anchor_disagree,
            "flow_confidence": flow_meta.get("flow_confidence"),
            "median_flow_px": flow_meta.get("median_flow_px"),
            "tracked_points": flow_meta.get("tracked_points"),
            "fb_good_points": flow_meta.get("fb_good_points"),
            "inliers": flow_meta.get("inliers"),
            "inlier_ratio": flow_meta.get("inlier_ratio"),
            "ecc_success": ecc_meta.get("ecc_success"),
            "ecc_score": ecc_meta.get("ecc_score"),
            "ecc_displacement_m": ecc_meta.get("ecc_displacement_m"),
        })

        prev_gray = cur_gray

    cap.release()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "source_base_hmap_path": args.base_hmap_path,
            "source_video_path": args.video_path,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "total_frames": len(out_frames),
            "exact_anchor_count": len(anchors),
            "parameters": {
                "rescue_score": RESCUE_SCORE,
                "wide_rescue_score": WIDE_RESCUE_SCORE,
                "near_radius": NEAR_RADIUS,
                "wide_radius": WIDE_RADIUS,
                "ecc_enabled": not args.disable_ecc,
            },
            "frames": out_frames,
        }, f, ensure_ascii=False, indent=2)

    avg_final_score = float(np.mean([fr["final_score"] for fr in out_frames])) if out_frames else 0.0
    avg_base_score = float(np.mean([fr["base_score"] for fr in out_frames if fr["base_score"] is not None])) if out_frames else 0.0
    rescue_count = sum(1 for fr in out_frames if fr["rescue_search_used"])
    print("DONE")
    print("out =", out_path.as_posix())
    print("frames =", len(out_frames))
    print("avg_base_score =", round(avg_base_score, 4))
    print("avg_final_score =", round(avg_final_score, 4))
    print("rescue_frames =", rescue_count)


if __name__ == "__main__":
    main()
