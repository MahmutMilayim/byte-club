import numpy as np
import cv2

def estimate_pitch_hsv(frames, k: int = 15) -> np.ndarray:
    k = min(k, len(frames))
    samples = []
    for f in frames[:k]:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        lower, upper = (25, 40, 40), (95, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
        if mask.mean() > 1:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            samples.append(hsv[mask > 0])
    if not samples:
        return np.median(
            cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0
        )
    samples = np.concatenate(samples, axis=0)
    return np.median(samples, axis=0)

def upper_body_mask(
    crop_bgr: np.ndarray,
    grass_hsv: np.ndarray,
    top_ratio: float = 0.55,
    dh: int = 12,
) -> np.ndarray:
    h0 = int(grass_hsv.reshape(-1, 3)[0][0])
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lowH, highH = max(0, h0 - dh), min(179, h0 + dh)
    grass_mask = cv2.inRange(hsv, (lowH, 40, 40), (highH, 255, 255))
    non_grass = cv2.bitwise_not(grass_mask)
    h, w = crop_bgr.shape[:2]
    upper = np.zeros((h, w), np.uint8)
    upper[: int(h * top_ratio), :] = 255
    mask = cv2.bitwise_and(non_grass, upper)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    return mask

def trim_outliers(colors: np.ndarray, method: str = "iqr", factor: float = 1.5) -> np.ndarray:
    if colors.size == 0:
        return colors
    X = colors.astype(np.float32, copy=True)
    if method == "iqr":
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        X = np.clip(X, lo, hi)
    return X

def get_left_team_label(x1_by_team: dict[int, list[float]], trim_ratio: float = 0.2) -> int:
    def trimmed_mean(xs: list[float]) -> float:
        if not xs:
            return np.nan
        xs = np.sort(np.asarray(xs, dtype=np.float32))
        n, k = len(xs), int(len(xs) * trim_ratio)
        if n >= 5 and k > 0 and 2 * k < n:
            xs = xs[k : n - k]
        return float(xs.mean()) if len(xs) else np.nan

    m0 = trimmed_mean(x1_by_team.get(0, []))
    m1 = trimmed_mean(x1_by_team.get(1, []))
    if np.isnan(m0) and np.isnan(m1):
        return 0
    if np.isnan(m0):
        return 1
    if np.isnan(m1):
        return 0
    return 0 if m0 <= m1 else 1
