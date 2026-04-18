from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0

PITCH_LINE_SEGMENTS = [
    ((0.0, 0.0), (105.0, 0.0)),
    ((0.0, 68.0), (105.0, 68.0)),
    ((0.0, 0.0), (0.0, 68.0)),
    ((105.0, 0.0), (105.0, 68.0)),
    ((52.5, 0.0), (52.5, 68.0)),
    ((16.5, 13.84), (16.5, 54.16)),
    ((88.5, 13.84), (88.5, 54.16)),
    ((0.0, 13.84), (16.5, 13.84)),
    ((0.0, 54.16), (16.5, 54.16)),
    ((88.5, 13.84), (105.0, 13.84)),
    ((88.5, 54.16), (105.0, 54.16)),
    ((0.0, 24.84), (5.5, 24.84)),
    ((5.5, 24.84), (5.5, 43.16)),
    ((5.5, 43.16), (0.0, 43.16)),
    ((105.0, 24.84), (99.5, 24.84)),
    ((99.5, 24.84), (99.5, 43.16)),
    ((99.5, 43.16), (105.0, 43.16)),
]

TEST_GRID_NORM = [
    (0.15, 0.22), (0.30, 0.22), (0.50, 0.22), (0.70, 0.22), (0.85, 0.22),
    (0.15, 0.40), (0.30, 0.40), (0.50, 0.40), (0.70, 0.40), (0.85, 0.40),
    (0.15, 0.58), (0.30, 0.58), (0.50, 0.58), (0.70, 0.58), (0.85, 0.58),
    (0.15, 0.76), (0.30, 0.76), (0.50, 0.76), (0.70, 0.76), (0.85, 0.76),
]

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


@dataclass
class FrameEvidence:
    green_mask: np.ndarray
    line_mask: np.ndarray
    line_pixel_count: int
    green_ratio: float


def normalize_homography(H):
    if H is None:
        return None
    H = np.array(H, dtype=np.float64)
    if H.shape != (3, 3):
        return None
    if not np.isfinite(H).all():
        return None
    if abs(H[2, 2]) < 1e-9:
        return None
    return H / H[2, 2]


def invert_homography(H):
    H = normalize_homography(H)
    if H is None:
        return None
    try:
        return normalize_homography(np.linalg.inv(H))
    except np.linalg.LinAlgError:
        return None


def image_to_pitch_xy(H_img_to_pitch, x, y):
    H = normalize_homography(H_img_to_pitch)
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
    return [px, py]


def pitch_to_image_xy(H_pitch_to_img, x, y):
    H = normalize_homography(H_pitch_to_img)
    if H is None:
        return None
    pt = np.array([[[float(x), float(y)]]], dtype=np.float64)
    try:
        out = cv2.perspectiveTransform(pt, H)
    except cv2.error:
        return None
    ix = float(out[0, 0, 0])
    iy = float(out[0, 0, 1])
    if not np.isfinite(ix) or not np.isfinite(iy):
        return None
    return [ix, iy]


def homography_disagreement_m(H1, H2, width, height):
    H1 = normalize_homography(H1)
    H2 = normalize_homography(H2)
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


def extract_pitch_evidence(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, (25, 25, 20), (100, 255, 255))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    green_mask = cv2.dilate(green_mask, np.ones((7, 7), np.uint8), iterations=1)

    white_mask = cv2.inRange(hsv, (0, 0, 145), (180, 100, 255))
    line_mask = cv2.bitwise_and(white_mask, cv2.dilate(green_mask, np.ones((9, 9), np.uint8), iterations=1))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    green_bool = green_mask > 0
    line_bool = line_mask > 0
    return FrameEvidence(
        green_mask=green_bool,
        line_mask=line_bool,
        line_pixel_count=int(line_bool.sum()),
        green_ratio=float(green_bool.mean()),
    )


def render_candidate_masks(H_pitch_to_img, frame_shape, line_thickness=4):
    H = normalize_homography(H_pitch_to_img)
    height, width = frame_shape[:2]

    line_mask = np.zeros((height, width), dtype=np.uint8)
    field_mask = np.zeros((height, width), dtype=np.uint8)
    if H is None:
        return line_mask > 0, field_mask > 0

    try:
        corners = np.array(
            [[0.0, 0.0], [FIELD_LENGTH_M, 0.0], [FIELD_LENGTH_M, FIELD_WIDTH_M], [0.0, FIELD_WIDTH_M]],
            dtype=np.float64,
        )
        polygon = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
        if np.isfinite(polygon).all():
            cv2.fillConvexPoly(field_mask, np.round(polygon).astype(np.int32), 255)

        for p1_pitch, p2_pitch in PITCH_LINE_SEGMENTS:
            seg = np.array([[p1_pitch, p2_pitch]], dtype=np.float64)
            pts = cv2.perspectiveTransform(seg, H)[0]
            if not np.isfinite(pts).all():
                continue
            p1 = tuple(np.round(pts[0]).astype(np.int32))
            p2 = tuple(np.round(pts[1]).astype(np.int32))
            if max(abs(p1[0]), abs(p1[1]), abs(p2[0]), abs(p2[1])) > 10000:
                continue
            cv2.line(line_mask, p1, p2, 255, line_thickness)
    except cv2.error:
        pass

    return line_mask > 0, field_mask > 0


def score_candidate(evidence: FrameEvidence, H_pitch_to_img, frame_shape):
    rendered_lines, rendered_field = render_candidate_masks(H_pitch_to_img, frame_shape)

    rendered_line_count = int(rendered_lines.sum())
    rendered_field_count = int(rendered_field.sum())
    if rendered_line_count == 0 or rendered_field_count == 0:
        return {
            "score": -1.0,
            "line_precision": 0.0,
            "line_recall": 0.0,
            "field_green_ratio": 0.0,
            "rendered_line_count": rendered_line_count,
            "rendered_field_count": rendered_field_count,
        }

    overlap = int((evidence.line_mask & rendered_lines).sum())
    line_precision = overlap / max(rendered_line_count, 1)
    line_recall = overlap / max(evidence.line_pixel_count, 1)
    field_green_ratio = float((evidence.green_mask & rendered_field).sum() / max(rendered_field_count, 1))

    score = (
        0.65 * line_precision +
        0.25 * field_green_ratio +
        0.10 * line_recall
    )

    return {
        "score": float(score),
        "line_precision": float(line_precision),
        "line_recall": float(line_recall),
        "field_green_ratio": float(field_green_ratio),
        "rendered_line_count": rendered_line_count,
        "rendered_field_count": rendered_field_count,
    }


def build_bev_pitch_template(bev_scale=10, pad=24):
    width = int(round(FIELD_LENGTH_M * bev_scale + 2 * pad))
    height = int(round(FIELD_WIDTH_M * bev_scale + 2 * pad))
    canvas = np.zeros((height, width), dtype=np.uint8)
    S = np.array(
        [[bev_scale, 0.0, pad], [0.0, bev_scale, pad], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rendered_lines, _ = render_candidate_masks(S, canvas.shape, line_thickness=max(1, int(round(0.35 * bev_scale))))
    canvas[rendered_lines] = 255
    canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
    return canvas, S


def refine_homography_bev_ecc(H_img_to_pitch, evidence: FrameEvidence, frame_shape, bev_template, S_pitch_to_bev):
    H = normalize_homography(H_img_to_pitch)
    if H is None:
        return None, {"ecc_success": False, "ecc_score": None, "ecc_displacement_m": None}

    line_mask = (evidence.line_mask.astype(np.uint8) * 255)
    H_img_to_bev = normalize_homography(S_pitch_to_bev @ H)
    if H_img_to_bev is None:
        return None, {"ecc_success": False, "ecc_score": None, "ecc_displacement_m": None}

    bev_height, bev_width = bev_template.shape[:2]
    warped = cv2.warpPerspective(line_mask, H_img_to_bev, (bev_width, bev_height))
    warped = cv2.GaussianBlur(warped, (5, 5), 0)
    if warped.max() < 1:
        return None, {"ecc_success": False, "ecc_score": None, "ecc_displacement_m": None}

    template_f = bev_template.astype(np.float32) / 255.0
    warped_f = warped.astype(np.float32) / 255.0
    warp_init = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)

    try:
        ecc_score, warp_update = cv2.findTransformECC(
            template_f,
            warped_f,
            warp_init,
            motionType=cv2.MOTION_HOMOGRAPHY,
            criteria=criteria,
            inputMask=bev_template,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None, {"ecc_success": False, "ecc_score": None, "ecc_displacement_m": None}

    warp_update = normalize_homography(warp_update)
    if warp_update is None:
        return None, {"ecc_success": False, "ecc_score": None, "ecc_displacement_m": None}

    displacement_px = []
    bev_corners = np.array(
        [
            [0.20 * bev_width, 0.20 * bev_height],
            [0.80 * bev_width, 0.20 * bev_height],
            [0.20 * bev_width, 0.80 * bev_height],
            [0.80 * bev_width, 0.80 * bev_height],
            [0.50 * bev_width, 0.50 * bev_height],
        ],
        dtype=np.float64,
    )
    moved = cv2.perspectiveTransform(bev_corners.reshape(-1, 1, 2), warp_update).reshape(-1, 2)
    for before, after in zip(bev_corners, moved):
        displacement_px.append(float(np.linalg.norm(before - after)))
    displacement_m = float(np.median(displacement_px) / max(float(S_pitch_to_bev[0, 0]), 1e-6))
    if displacement_m > 8.0:
        return None, {"ecc_success": False, "ecc_score": float(ecc_score), "ecc_displacement_m": displacement_m}

    H_img_to_bev_refined = normalize_homography(warp_update @ H_img_to_bev)
    H_bev_to_pitch = invert_homography(S_pitch_to_bev)
    if H_img_to_bev_refined is None or H_bev_to_pitch is None:
        return None, {"ecc_success": False, "ecc_score": float(ecc_score), "ecc_displacement_m": displacement_m}

    refined = normalize_homography(H_bev_to_pitch @ H_img_to_bev_refined)
    return refined, {
        "ecc_success": refined is not None,
        "ecc_score": float(ecc_score),
        "ecc_displacement_m": displacement_m,
    }


def estimate_delta_h(prev_gray, cur_gray):
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE,
    )
    if prev_pts is None or len(prev_pts) < MIN_TRACKED_POINTS:
        return None, _empty_flow_meta()

    cur_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        prev_pts,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
    )
    if cur_pts is None or status_fwd is None:
        return None, _empty_flow_meta()

    back_pts, status_back, _ = cv2.calcOpticalFlowPyrLK(
        cur_gray,
        prev_gray,
        cur_pts,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
    )
    if back_pts is None or status_back is None:
        return None, _empty_flow_meta()

    good_mask = (status_fwd.flatten() == 1) & (status_back.flatten() == 1)
    fwd_prev = prev_pts[good_mask].reshape(-1, 2)
    fwd_cur = cur_pts[good_mask].reshape(-1, 2)
    bwd_prev = back_pts[good_mask].reshape(-1, 2)

    tracked_points = len(fwd_prev)
    if tracked_points < MIN_TRACKED_POINTS:
        return None, _empty_flow_meta(tracked_points=tracked_points)

    fb_err = np.linalg.norm(fwd_prev - bwd_prev, axis=1)
    fb_mask = fb_err <= FB_MAX_ERR
    good_prev = fwd_prev[fb_mask]
    good_cur = fwd_cur[fb_mask]
    fb_good_points = len(good_prev)

    if fb_good_points < MIN_TRACKED_POINTS:
        return None, _empty_flow_meta(tracked_points=tracked_points, fb_good_points=fb_good_points)

    flow_mag = np.linalg.norm(good_cur - good_prev, axis=1)
    median_flow_px = float(np.median(flow_mag)) if len(flow_mag) else 0.0

    H_prev_to_cur, inlier_mask = cv2.findHomography(
        good_prev,
        good_cur,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
    )
    if H_prev_to_cur is None or inlier_mask is None:
        return None, _empty_flow_meta(
            tracked_points=tracked_points,
            fb_good_points=fb_good_points,
            median_flow_px=median_flow_px,
        )

    inliers = int(inlier_mask.sum())
    inlier_ratio = float(inliers / max(fb_good_points, 1))
    if inliers < MIN_INLIERS:
        return None, _empty_flow_meta(
            tracked_points=tracked_points,
            fb_good_points=fb_good_points,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            median_flow_px=median_flow_px,
        )

    point_score = min(1.0, fb_good_points / 120.0)
    flow_confidence = 0.55 * inlier_ratio + 0.45 * point_score
    meta = {
        "tracked_points": tracked_points,
        "fb_good_points": fb_good_points,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "flow_confidence": float(flow_confidence),
        "median_flow_px": median_flow_px,
    }
    return normalize_homography(H_prev_to_cur), meta


def _empty_flow_meta(
    tracked_points=0,
    fb_good_points=0,
    inliers=0,
    inlier_ratio=0.0,
    flow_confidence=0.0,
    median_flow_px=0.0,
):
    return {
        "tracked_points": tracked_points,
        "fb_good_points": fb_good_points,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "flow_confidence": flow_confidence,
        "median_flow_px": median_flow_px,
    }
