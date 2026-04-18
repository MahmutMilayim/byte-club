import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BANK_PATH = "/output/stage6_field/homography_bank.json"
VIDEO_PATH = "/output/stage3_filter/gameplay.mp4"
GAMEPLAY_JSON_PATH = "/output/stage3_filter/gameplay.json"
OUT_PATH = "/output/stage6_field/homography_map.json"

KEYFRAME_STRIDE = 5
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0

MAX_CORNERS = 500
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7
BLOCK_SIZE = 7

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

MIN_TRACKED_POINTS = 35
MIN_INLIERS = 20
RANSAC_THRESH = 3.0
FB_MAX_ERR = 1.5

# confidence-weighted blend
MIN_CONF_FOR_PROP = 0.18
MAX_CONF_FOR_PROP = 0.75
ANCHOR_BLEND_ALPHA_MIN = 0.08
ANCHOR_BLEND_ALPHA_MAX = 0.60

# adaptive relock
HIGH_MOTION_FLOW_PX = 14.0
VERY_HIGH_MOTION_FLOW_PX = 24.0
HIGH_DISAGREE_M = 4.5
VERY_HIGH_DISAGREE_M = 7.5
LOW_INLIER_RATIO = 0.45

# smarter relock
NEARBY_ANCHOR_WINDOW = 20  # frame bazında anchor arama penceresi
MAX_GRID_DISAGREE_M = 6.0

TEST_GRID_NORM = [
    (0.15, 0.22), (0.30, 0.22), (0.50, 0.22), (0.70, 0.22), (0.85, 0.22),
    (0.15, 0.40), (0.30, 0.40), (0.50, 0.40), (0.70, 0.40), (0.85, 0.40),
    (0.15, 0.58), (0.30, 0.58), (0.50, 0.58), (0.70, 0.58), (0.85, 0.58),
    (0.15, 0.76), (0.30, 0.76), (0.50, 0.76), (0.70, 0.76), (0.85, 0.76),
]

def normalize_H(H):
    if H is None:
        return None
    H = np.array(H, dtype=np.float64)
    if H.shape != (3, 3):
        return None
    if not np.isfinite(H).all():
        return None
    if abs(H[2, 2]) < 1e-9:
        return None
    H = H / H[2, 2]
    return H

def blend_H(H1, H2, alpha):
    H1 = normalize_H(H1)
    H2 = normalize_H(H2)
    if H1 is None:
        return H2
    if H2 is None:
        return H1
    H = (1 - alpha) * H1 + alpha * H2
    return normalize_H(H)

def world_to_image_with_P(P, x_field, y_field, z=0.0):
    X = x_field - FIELD_LENGTH_M / 2.0
    Y = y_field - FIELD_WIDTH_M / 2.0
    pt = np.array([X, Y, z, 1.0], dtype=np.float64)
    ip = P @ pt
    if abs(ip[-1]) < 1e-9:
        return None
    ip = ip / ip[-1]
    if not np.isfinite(ip[0]) or not np.isfinite(ip[1]):
        return None
    return float(ip[0]), float(ip[1])

def image_to_pitch_xy(H_img_to_pitch, x, y):
    H = normalize_H(H_img_to_pitch)
    if H is None:
        return None
    pt = np.array([[[float(x), float(y)]]], dtype=np.float64)
    try:
        out = cv2.perspectiveTransform(pt, H)
    except cv2.error:
        return None
    px = float(out[0, 0, 0])
    py = float(out[0, 0, 1])
    if not np.isfinite(px) or not np.isfinite(py):
        return None
    return px, py

def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params["x_focal_length"]
    y_focal_length = cam_params["y_focal_length"]
    principal_point = np.array(cam_params["principal_point"], dtype=np.float64)
    position_meters = np.array(cam_params["position_meters"], dtype=np.float64)
    rotation = np.array(cam_params["rotation_matrix"], dtype=np.float64)

    It = np.eye(4, dtype=np.float64)[:-1]
    It[:, -1] = -position_meters

    Q = np.array([
        [x_focal_length, 0, principal_point[0]],
        [0, y_focal_length, principal_point[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    P = Q @ (rotation @ It)
    return P

def compute_ground_homographies_from_P(P):
    pitch_pts = np.array([
        [0.0, 0.0],
        [105.0, 0.0],
        [0.0, 68.0],
        [105.0, 68.0],
        [52.5, 0.0],
        [52.5, 68.0],
        [52.5, 34.0],
        [11.0, 34.0],
        [94.0, 34.0],
        [16.5, 13.84],
        [16.5, 54.16],
        [88.5, 13.84],
        [88.5, 54.16],
    ], dtype=np.float64)

    image_pts = []
    valid_pitch_pts = []

    for x, y in pitch_pts:
        ip = world_to_image_with_P(P, x, y, z=0.0)
        if ip is not None:
            image_pts.append(ip)
            valid_pitch_pts.append((x, y))

    if len(image_pts) < 4:
        return None, None

    image_pts = np.array(image_pts, dtype=np.float64)
    valid_pitch_pts = np.array(valid_pitch_pts, dtype=np.float64)

    H_pitch_to_img, _ = cv2.findHomography(valid_pitch_pts, image_pts, method=0)
    H_img_to_pitch, _ = cv2.findHomography(image_pts, valid_pitch_pts, method=0)

    return normalize_H(H_pitch_to_img), normalize_H(H_img_to_pitch)

def median_homography_disagreement_m(H1, H2, width, height):
    H1 = normalize_H(H1)
    H2 = normalize_H(H2)
    if H1 is None or H2 is None:
        return None

    dists = []
    for xn, yn in TEST_GRID_NORM:
        x = xn * width
        y = yn * height
        p1 = image_to_pitch_xy(H1, x, y)
        p2 = image_to_pitch_xy(H2, x, y)
        if p1 is None or p2 is None:
            continue
        d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        if np.isfinite(d):
            dists.append(d)

    if not dists:
        return None
    return float(np.median(dists))

def estimate_delta_H(prev_gray, cur_gray):
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE
    )
    if prev_pts is None or len(prev_pts) < MIN_TRACKED_POINTS:
        return None, {
            "tracked_points": 0,
            "fb_good_points": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "flow_confidence": 0.0,
            "median_flow_px": 0.0,
        }

    cur_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, prev_pts, None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA
    )
    if cur_pts is None or status_fwd is None:
        return None, {
            "tracked_points": 0,
            "fb_good_points": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "flow_confidence": 0.0,
            "median_flow_px": 0.0,
        }

    back_pts, status_back, _ = cv2.calcOpticalFlowPyrLK(
        cur_gray, prev_gray, cur_pts, None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA
    )
    if back_pts is None or status_back is None:
        return None, {
            "tracked_points": 0,
            "fb_good_points": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "flow_confidence": 0.0,
            "median_flow_px": 0.0,
        }

    good_mask = (status_fwd.flatten() == 1) & (status_back.flatten() == 1)
    fwd_prev = prev_pts[good_mask].reshape(-1, 2)
    fwd_cur = cur_pts[good_mask].reshape(-1, 2)
    bwd_prev = back_pts[good_mask].reshape(-1, 2)

    tracked_points = len(fwd_prev)
    if tracked_points < MIN_TRACKED_POINTS:
        return None, {
            "tracked_points": tracked_points,
            "fb_good_points": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "flow_confidence": 0.0,
            "median_flow_px": 0.0,
        }

    fb_err = np.linalg.norm(fwd_prev - bwd_prev, axis=1)
    fb_mask = fb_err <= FB_MAX_ERR

    good_prev = fwd_prev[fb_mask]
    good_cur = fwd_cur[fb_mask]
    fb_good_points = len(good_prev)

    if fb_good_points < MIN_TRACKED_POINTS:
        return None, {
            "tracked_points": tracked_points,
            "fb_good_points": fb_good_points,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "flow_confidence": 0.0,
            "median_flow_px": 0.0,
        }

    flow_vec = good_cur - good_prev
    flow_mag = np.linalg.norm(flow_vec, axis=1)
    median_flow_px = float(np.median(flow_mag)) if len(flow_mag) else 0.0

    H_prev_to_cur, inlier_mask = cv2.findHomography(
        good_prev, good_cur,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH
    )
    if H_prev_to_cur is None or inlier_mask is None:
        return None, {
            "tracked_points": tracked_points,
            "fb_good_points": fb_good_points,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "flow_confidence": 0.0,
            "median_flow_px": median_flow_px,
        }

    inliers = int(inlier_mask.sum())
    inlier_ratio = float(inliers / max(fb_good_points, 1))

    if inliers < MIN_INLIERS:
        return None, {
            "tracked_points": tracked_points,
            "fb_good_points": fb_good_points,
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "flow_confidence": 0.0,
            "median_flow_px": median_flow_px,
        }

    # confidence: point coverage + inlier ratio birlikte
    point_score = min(1.0, fb_good_points / 120.0)
    flow_conf = 0.55 * inlier_ratio + 0.45 * point_score

    return normalize_H(H_prev_to_cur), {
        "tracked_points": tracked_points,
        "fb_good_points": fb_good_points,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "flow_confidence": float(flow_conf),
        "median_flow_px": median_flow_px,
    }

with open(BANK_PATH, "r", encoding="utf-8") as f:
    bank = json.load(f)
with open(GAMEPLAY_JSON_PATH, "r", encoding="utf-8") as f:
    gameplay = json.load(f)

accepted = {}
accepted_list = []
for i, fr in enumerate(bank["frames"]):
    if fr["accepted"] and fr["final_params_dict"] is not None:
        src_idx = i * KEYFRAME_STRIDE
        P = projection_from_cam_params(fr["final_params_dict"])
        H_pitch_to_img, H_img_to_pitch = compute_ground_homographies_from_P(P)
        if H_img_to_pitch is None:
            continue
        payload = {
            "keyframe_index": i,
            "source_original_frame_index": src_idx,
            "image_path": fr["image_path"],
            "rep_err": fr["rep_err"],
            "sanity_score": fr["sanity_score"],
            "final_params_dict": fr["final_params_dict"],
            "H_img_to_pitch": H_img_to_pitch,
            "H_pitch_to_img": H_pitch_to_img,
        }
        accepted[src_idx] = payload
        accepted_list.append(payload)

if not accepted_list:
    raise RuntimeError("Accepted anchor yok")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video acilamadi")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = len(gameplay["frames"])

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Ilk frame okunamadi")
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

def nearest_anchor(frame_idx):
    return min(accepted_list, key=lambda x: abs(x["source_original_frame_index"] - frame_idx))

def nearby_anchors(frame_idx):
    out = []
    for a in accepted_list:
        if abs(a["source_original_frame_index"] - frame_idx) <= NEARBY_ANCHOR_WINDOW:
            out.append(a)
    if not out:
        out.append(nearest_anchor(frame_idx))
    return out

frame_map = []

anchor0 = nearest_anchor(0)
current_H_img_to_pitch = anchor0["H_img_to_pitch"]
current_H_pitch_to_img = anchor0["H_pitch_to_img"]

frame_map.append({
    "frame_index": 0,
    "source_keyframe_index": anchor0["keyframe_index"],
    "source_original_frame_index": anchor0["source_original_frame_index"],
    "source_image_path": anchor0["image_path"],
    "source_rep_err": anchor0["rep_err"],
    "source_sanity_score": anchor0["sanity_score"],
    "H_img_to_pitch": current_H_img_to_pitch.tolist() if current_H_img_to_pitch is not None else None,
    "H_pitch_to_img": current_H_pitch_to_img.tolist() if current_H_pitch_to_img is not None else None,
    "final_params_dict": anchor0["final_params_dict"],
    "map_source": "anchor_exact_or_nearest",
    "tracked_points": None,
    "fb_good_points": None,
    "inliers": None,
    "inlier_ratio": None,
    "flow_confidence": None,
    "median_flow_px": None,
    "anchor_blend_alpha": None,
    "anchor_disagree_m": None,
})

print(f"\nStage 6 | Homography Motion — {total_frames} frame, {len(accepted_list)} anchor")
for frame_idx in tqdm(range(1, total_frames), desc="Stage 6 | Homography  ",
                      unit="frame", ncols=90):
    ret, cur_frame = cap.read()
    if not ret:
        break

    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    if frame_idx in accepted:
        a = accepted[frame_idx]
        current_H_img_to_pitch = a["H_img_to_pitch"]
        current_H_pitch_to_img = a["H_pitch_to_img"]
        map_source = "anchor_exact"
        flow_meta = {
            "tracked_points": None,
            "fb_good_points": None,
            "inliers": None,
            "inlier_ratio": None,
            "flow_confidence": None,
            "median_flow_px": None,
        }
        src = a
        anchor_blend_alpha = None
        anchor_disagree_m = None
    else:
        delta_H_prev_to_cur, flow_meta = estimate_delta_H(prev_gray, cur_gray)
        src = nearest_anchor(frame_idx)

        if delta_H_prev_to_cur is not None and current_H_img_to_pitch is not None:
            try:
                delta_inv = np.linalg.inv(delta_H_prev_to_cur)
                propagated = normalize_H(current_H_img_to_pitch @ delta_inv)
            except np.linalg.LinAlgError:
                propagated = None

            if propagated is not None:
                # en yakın birkaç anchor içinde en tutarlı olanı seç
                best_anchor = src
                best_disagree = None
                for cand in nearby_anchors(frame_idx):
                    disagree = median_homography_disagreement_m(
                        propagated, cand["H_img_to_pitch"], frame_width, frame_height
                    )
                    if disagree is None:
                        continue
                    if best_disagree is None or disagree < best_disagree:
                        best_disagree = disagree
                        best_anchor = cand

                src = best_anchor
                anchor_disagree_m = best_disagree

                conf = flow_meta["flow_confidence"]
                conf = max(0.0, min(1.0, conf))

                # confidence yüksekse anchor karışımı az, düşükse çok
                if conf <= MIN_CONF_FOR_PROP:
                    alpha = ANCHOR_BLEND_ALPHA_MAX
                elif conf >= MAX_CONF_FOR_PROP:
                    alpha = ANCHOR_BLEND_ALPHA_MIN
                else:
                    t = (conf - MIN_CONF_FOR_PROP) / (MAX_CONF_FOR_PROP - MIN_CONF_FOR_PROP)
                    alpha = ANCHOR_BLEND_ALPHA_MAX * (1 - t) + ANCHOR_BLEND_ALPHA_MIN * t

                # adaptive relock: hızlı kamera hareketi + düşük inlier ratio + yüksek disagreement
                median_flow_px = flow_meta["median_flow_px"] if flow_meta["median_flow_px"] is not None else 0.0
                inlier_ratio = flow_meta["inlier_ratio"] if flow_meta["inlier_ratio"] is not None else 0.0

                if anchor_disagree_m is not None and anchor_disagree_m > HIGH_DISAGREE_M:
                    alpha = max(alpha, 0.35)

                if anchor_disagree_m is not None and anchor_disagree_m > VERY_HIGH_DISAGREE_M:
                    alpha = max(alpha, 0.52)

                if median_flow_px > HIGH_MOTION_FLOW_PX:
                    alpha = max(alpha, 0.32)

                if median_flow_px > VERY_HIGH_MOTION_FLOW_PX:
                    alpha = max(alpha, 0.48)

                if inlier_ratio < LOW_INLIER_RATIO:
                    alpha = max(alpha, 0.42)

                # propagate-anchor disagreement çok yüksekse daha çok anchor'a yaslan
                if anchor_disagree_m is not None and anchor_disagree_m > MAX_GRID_DISAGREE_M:
                    alpha = max(alpha, 0.45)

                alpha = min(max(alpha, ANCHOR_BLEND_ALPHA_MIN), ANCHOR_BLEND_ALPHA_MAX)

                current_H_img_to_pitch = blend_H(propagated, src["H_img_to_pitch"], alpha)
                current_H_pitch_to_img = None
                try:
                    current_H_pitch_to_img = normalize_H(np.linalg.inv(current_H_img_to_pitch))
                except np.linalg.LinAlgError:
                    current_H_pitch_to_img = src["H_pitch_to_img"]

                map_source = "propagated_conf_blended_anchor"
                anchor_blend_alpha = float(alpha)
            else:
                current_H_img_to_pitch = src["H_img_to_pitch"]
                current_H_pitch_to_img = src["H_pitch_to_img"]
                map_source = "anchor_fallback_after_bad_propagation"
                anchor_blend_alpha = None
                anchor_disagree_m = None
        else:
            current_H_img_to_pitch = src["H_img_to_pitch"]
            current_H_pitch_to_img = src["H_pitch_to_img"]
            map_source = "anchor_fallback"
            anchor_blend_alpha = None
            anchor_disagree_m = None

    frame_map.append({
        "frame_index": frame_idx,
        "source_keyframe_index": src["keyframe_index"],
        "source_original_frame_index": src["source_original_frame_index"],
        "source_image_path": src["image_path"],
        "source_rep_err": src["rep_err"],
        "source_sanity_score": src["sanity_score"],
        "H_img_to_pitch": current_H_img_to_pitch.tolist() if current_H_img_to_pitch is not None else None,
        "H_pitch_to_img": current_H_pitch_to_img.tolist() if current_H_pitch_to_img is not None else None,
        "final_params_dict": src["final_params_dict"],
        "map_source": map_source,
        "tracked_points": flow_meta["tracked_points"],
        "fb_good_points": flow_meta["fb_good_points"],
        "inliers": flow_meta["inliers"],
        "inlier_ratio": flow_meta["inlier_ratio"],
        "flow_confidence": flow_meta["flow_confidence"],
        "median_flow_px": flow_meta["median_flow_px"],
        "anchor_blend_alpha": anchor_blend_alpha,
        "anchor_disagree_m": anchor_disagree_m,
    })

    prev_gray = cur_gray

cap.release()

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "keyframe_stride": KEYFRAME_STRIDE,
        "total_frames": len(frame_map),
        "accepted_keyframes": len(accepted_list),
        "nearby_anchor_window": NEARBY_ANCHOR_WINDOW,
        "frames": frame_map
    }, f, ensure_ascii=False, indent=2)

print("DONE")
print("accepted_keyframes =", len(accepted_list))
print("total_frames =", len(frame_map))
print("out =", OUT_PATH)
