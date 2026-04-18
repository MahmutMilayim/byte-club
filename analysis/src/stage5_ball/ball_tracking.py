"""
Stage 5 — Ball Tracking (YOLO + Jump-Suppressed + Review)

YOLO ile top tespiti, hareket/oyuncu yakınlığı/şekil skorlamasıyla
jump-suppressed takip yapar; çıktıyla birlikte detaylı review video üretir.

TrackNetV4 ensemble: geniş TN–YOLO eşleşme + güçlü çift bonus; çelişki cezası
yalnız TN bir YOLO ile teyitlendiğinde. TN-yalnız drift + daha uzun Kalman sürdürme
(MAX_MISS). Sıçrama kapısı miss≤1 iken (kopuklukta serbest yeniden yakalama).

Giriş : /output/stage3_filter/gameplay.mp4 + gameplay.json
         /output/stage4_clustering/track_labels.json
Çıkış : /output/stage5_ball/ball_tracks.json
         /output/stage5_ball/ball_review.mp4

Hard negative (YOLO sınıf 32 için yanlış pozitif kırpıntıları):
  COLLECT_BALL_HARD_NEGATIVES=1  →  hard_negatives/inbox/*.jpg + manifest.jsonl
  Manuel seçim: python .../hard_negative_review.py  → confirmed/ ve discarded/
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ─── DOSYA YOLLARI ────────────────────────────────────────────────────────────
VIDEO_PATH             = "/output/stage3_filter/gameplay.mp4"
GAMEPLAY_JSON_PATH     = "/output/stage3_filter/gameplay.json"
TRACK_LABELS_JSON_PATH = "/output/stage4_clustering/track_labels.json"
OUTPUT_JSON_PATH       = "/output/stage5_ball/ball_tracks.json"
OUTPUT_VIDEO_PATH      = "/output/stage5_ball/ball_review.mp4"
MODEL_PATH             = "/work/models/yolo26x.pt"

# Hard negative export (COLLECT_BALL_HARD_NEGATIVES=1 ile açılır)
HARD_NEGATIVES_DIR   = Path("/output/stage5_ball/hard_negatives")
HARD_NEG_MIN_CONF    = 0.12   # daha düşük güvenli tespitleri yazma
HARD_NEG_MAX_IMAGES  = 8000   # tek koşuda üst sınır
HARD_NEG_PAD_FRAC    = 0.45   # bbox çevresinde bağlam
HARD_NEG_MIN_SIDE    = 48     # kırpım minimum kenar (px)
HARD_NEG_GRID        = 12     # dedup: merkez quantize (px)

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
VERBOSE            = True
LOG_EVERY_N_FRAMES = 50

BALL_CLASS_ID  = 32       # COCO sports ball
CONF_THRESHOLD = 0.03
IOU_THRESHOLD  = 0.35
IMGSZ          = 1280     # 960 → 1280: küçük topu daha iyi yakalamak için

# ─── FİZİK KALMAN — YERÇEKIMI ÖNYARGISI ──────────────────────────────────────
# Broadcast kamera tipik ~20° yukarıdan bakar; topun image koordinatlarındaki
# düşey hareketi yakl. 0.3-0.5 px/frame² aşağı yönlü ivme içerir.
BALL_GRAVITY_PX = 0.35    # px/frame² aşağı yönlü ivme (ayarlanabilir)

# ─── TRACKNETV4 ENSEMBLE ──────────────────────────────────────────────────────
# Hedef: mümkün olduğunca az kayıp + az yanlış pozitif — tek kamerada ikisi tam
# birlikte garanti edilemez; bu blok önce TN+YOLO teyidini güçlendirir, sonra
# uzun interpolasyonla “boş” kareleri azaltır. Çelişki cezası yalnız TN bir YOLO
# ile eşleşince uygulanır (yanlış heatmap YOLO’yu susturmaz).
TRACKNET_JSON_PATH    = "/output/stage5b_tracknet/tracknet_detections.json"
TN_YOLO_MATCH_RADIUS  = 74.0   # Daha geniş = daha çok çift onay bonusu
TN_BONUS_SCORE        = 0.40   # Eşleşen adaya ek skor (TN+YOLO birlikte öne çıkar)
TN_SOLO_CONF_THRESH   = 0.46   # TN-yalnız (YOLO kaçırdı); düşük = daha çok kurtarma
TN_SOLO_FAKE_CONF     = 0.30   # Sentetik aday YOLO güveni (görselleştirme için)
TN_SOLO_MAX_DRIFT_PX     = 205.0   # TN solo: Kalman/merkeze izin verilen sapma
TN_SOLO_COLD_START_CONF  = 0.44    # İlk kilit: bir tık daha düşük eşik
TN_CONFLICT_TN_CONF      = 0.52    # (Sadece matched==True iken) uzak YOLO cezası
TN_CONFLICT_DIST_MULT    = 1.65
TN_CONFLICT_YOLO_PENALTY = 1.05
POST_PICK_MAX_JUMP_PX    = 305.0   # Kilitli izde TN’siz ani sıçramayı kes
POST_PICK_MAX_JUMP_TN_PX = 430.0   # TN teyitli yolda biraz daha serbest
# miss_count <= bu değerken sıçrama kapısı açık (0–1: kilit + ilk kaçırma karesi)
POST_PICK_GATE_MAX_MISS  = 1

MIN_BALL_W = 4;  MAX_BALL_W = 42
MIN_BALL_H = 4;  MAX_BALL_H = 42
MAX_ASPECT_RATIO = 2.3

TOP_REGION_BLOCK_RATIO = 0.12
MAX_MISS_FRAMES        = 13   # Kalman sürdürme: daha uzun “top yok” öncesi kopma
MOTION_DISTANCE_NORM   = 232.0   # Hareket uyumu biraz daha toleranslı
FOOT_DISTANCE_NORM     = 165.0
MAX_ALLOWED_JUMP_PX    = 300.0   # Skor içi sıçrama cezası ile uyum

STATIC_ARTIFACT_RADIUS_PX       = 18.0
STATIC_ARTIFACT_FRAMES          = 8
STATIC_ARTIFACT_PLAYER_DIST_MIN = 85.0
STATIC_ARTIFACT_PENALTY         = 2.0
STATIC_ARTIFACT_MAX_SPOTS       = 12

# ─── PENALTİ NOKTASI — KAMERA-KOMPANZASYONLU SERT DIŞLAMA ────────────────────
# Kamera her frame hareket edebilir; sabit alan noktaları piksel değil
# kamera-kompanze edilmiş konumda takip edilir.
CAM_FLOW_MAX_FEATURES  = 80    # LK optik flow arka plan noktası sayısı
CAM_FLOW_REFRESH_EVERY = 20    # Feature noktaları her N frame'de yenilenir
PSPOT_SUSPECT_RADIUS   = 14.0  # Bu yarıçap içi aynı nokta sayılır
PSPOT_CONFIRM_STREAK   = 12    # Onaylamak için gereken ardışık frame sayısı
PSPOT_SUSPECT_DECAY    = 25    # Bu kadar frame görülmezse şüpheli silinir
PSPOT_MAX_CONFIRMED    = 6     # Maksimum onaylı sabit nokta sayısı
PSPOT_EXCL_RADIUS      = 22.0  # Onaylı noktaya bu kadar yakın tespitler dışlanır
PSPOT_NEAR_PLAYER_PX   = 30.0  # Oyuncu bu kadar yakınsa dışlama kaldırılır (top üzerinde/yanında demek)
PSPOT_NEAR_BALL_PX     = 85.0  # Son top bu kadar yakınsa dışlama kaldırılır
PSPOT_FAR_PLAYER_MIN   = 80.0  # Şüpheli işaretlemek için oyuncu en az bu kadar uzak olmalı

SHAPE_WEIGHT          = 0.85
SHAPE_LOW_THRESHOLD   = 0.38
SHAPE_LOW_PENALTY     = 0.95
ASPECT_RATIO_HARD_LIMIT = 2.4
ASPECT_RATIO_SOFT_LIMIT = 1.7

ELONGATION_PENALTY_STRONG = 1.35
ELONGATION_PENALTY_SOFT   = 0.55
BORDER_TOUCH_PENALTY      = 0.95
FILL_RATIO_LOW_THRESHOLD  = 0.18
FILL_RATIO_LOW_PENALTY    = 0.75

# ─── STATIK ARTEFAKT DURUMU (modül düzeyi) ────────────────────────────────────
_suspect_static_spots   = []
_last_static_candidate  = None
_static_streak          = 0

# ── Penaltı noktası kamera-kompanzasyonlu takip durumu ────────────────────────
_prev_gray      = None
_bg_features    = None          # shape (N, 1, 2) float32
_spot_suspects  = []            # [{"cx":float,"cy":float,"streak":int,"no_see":int}]
_confirmed_spots = []           # [{"cx":float,"cy":float}]

# ─── TOP KALMAN FİLTRESİ (image koordinatları, piksel) ───────────────────────
class _BallKalman:
    """Constant-velocity Kalman filter — image pixel coordinates."""
    def __init__(self, cx, cy, vx=0., vy=0.):
        self._x = np.array([cx, cy, vx, vy], dtype=np.float64)
        self._P = np.diag([16., 16., 64., 64.])
        self._F = np.eye(4, dtype=np.float64)
        self._F[0, 2] = 1.; self._F[1, 3] = 1.
        self._H = np.zeros((2, 4), dtype=np.float64)
        self._H[0, 0] = 1.; self._H[1, 1] = 1.
        self._Q = np.diag([1., 1., 8., 8.])     # process noise
        self._R = np.diag([16., 16.])            # measurement noise (~4px error)

    def predict(self):
        self._x = self._F @ self._x
        self._x[3] += BALL_GRAVITY_PX   # fizik: aşağı yönlü ivme önyargısı
        self._P = self._F @ self._P @ self._F.T + self._Q
        return float(self._x[0]), float(self._x[1])

    def update(self, cx, cy):
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P
        return float(self._x[0]), float(self._x[1])

    @property
    def vel(self):
        return float(self._x[2]), float(self._x[3])

# ─── YARDIMCI FONKSİYONLAR ───────────────────────────────────────────────────
def _dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def round2(v, nd=2):
    return None if v is None else round(float(v), nd)

def _update_camera_flow(gray):
    """
    Arka plan optik akışıyla kameranın bu frame'deki ötme miktarını döndürür.
    Lucas-Kanade kullanır; medyan translasyon olarak tahmin verir.
    """
    global _prev_gray, _bg_features

    cam_dx, cam_dy = 0.0, 0.0

    if _prev_gray is not None and _bg_features is not None and len(_bg_features) >= 6:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            _prev_gray, gray, _bg_features, None,
            winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        if next_pts is not None:
            mask = status.ravel() == 1
            good_p = _bg_features[mask]
            good_n = next_pts[mask]
            if len(good_p) >= 4:
                cam_dx = float(np.median(good_n[:, 0, 0] - good_p[:, 0, 0]))
                cam_dy = float(np.median(good_n[:, 0, 1] - good_p[:, 0, 1]))

    _prev_gray = gray
    return cam_dx, cam_dy


def _refresh_bg_features(gray, seq_idx):
    """Her CAM_FLOW_REFRESH_EVERY frame'de arka plan feature noktalarını yeniler."""
    global _bg_features
    if seq_idx % CAM_FLOW_REFRESH_EVERY == 0 or _bg_features is None or len(_bg_features) < 10:
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=CAM_FLOW_MAX_FEATURES,
            qualityLevel=0.01, minDistance=20,
        )
        _bg_features = pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)


def _compensate_spots(cam_dx, cam_dy):
    """Tüm şüpheli ve onaylı noktaları kamera hareketi kadar kaydır."""
    for s in _spot_suspects:
        s["cx"] += cam_dx
        s["cy"] += cam_dy
    for s in _confirmed_spots:
        s["cx"] += cam_dx
        s["cy"] += cam_dy


def _update_spot_suspects(candidates):
    """
    Bu frame'deki adaylarla şüpheli nokta hafızasını günceller.
    Oyuncudan uzak, tekrar tekrar aynı yerde görülen tespitler onaylı
    sabit nokta (penaltı noktası vb.) olarak işaretlenir.
    """
    global _spot_suspects, _confirmed_spots

    # Tüm şüphelilerin no_see sayacını artır; eşiği aşanları sil
    for s in _spot_suspects:
        s["no_see"] += 1
    _spot_suspects = [s for s in _spot_suspects if s["no_see"] < PSPOT_SUSPECT_DECAY]

    for c in candidates:
        # Sadece oyuncudan yeterince uzak adayları şüpheli say
        pd = c.get("nearest_player_distance_px")
        if pd is not None and pd < PSPOT_FAR_PLAYER_MIN:
            continue

        cx, cy = c["cx"], c["cy"]

        # Zaten onaylı bir noktanın yakınındaysa tekrar işleme
        if any(_dist(cx, cy, s["cx"], s["cy"]) < PSPOT_SUSPECT_RADIUS
               for s in _confirmed_spots):
            continue

        # Mevcut şüpheli listesinde eşleşen var mı?
        matched = False
        for s in _spot_suspects:
            if _dist(cx, cy, s["cx"], s["cy"]) < PSPOT_SUSPECT_RADIUS:
                s["streak"] += 1
                s["no_see"] = 0
                # Konumu hafifçe güncelle (EMA)
                s["cx"] = s["cx"] * 0.85 + cx * 0.15
                s["cy"] = s["cy"] * 0.85 + cy * 0.15
                matched = True
                if s["streak"] >= PSPOT_CONFIRM_STREAK:
                    # Onaylı listeye ekle (zaten yoksa)
                    if not any(_dist(s["cx"], s["cy"], cs["cx"], cs["cy"]) < PSPOT_SUSPECT_RADIUS
                               for cs in _confirmed_spots):
                        _confirmed_spots = (_confirmed_spots + [{"cx": s["cx"], "cy": s["cy"]}]
                                            )[-PSPOT_MAX_CONFIRMED:]
                        print(f"\n  [Stage 5] ► Sabit nokta onaylandı: "
                              f"({s['cx']:.0f}, {s['cy']:.0f}) px")
                break

        if not matched:
            _spot_suspects.append({"cx": float(cx), "cy": float(cy),
                                   "streak": 1, "no_see": 0})


def _is_penalty_spot(cx, cy, player_dist, lc):
    """
    True dönerse bu aday onaylı bir sabit nokta (penaltı vb.) olduğu için
    dışlanmalıdır. Override: oyuncu yakınsa veya top o yöne gidiyorsa izin verilir.
    """
    if not _confirmed_spots:
        return False
    for s in _confirmed_spots:
        if _dist(cx, cy, s["cx"], s["cy"]) > PSPOT_EXCL_RADIUS:
            continue
        if player_dist is not None and player_dist < PSPOT_NEAR_PLAYER_PX:
            return False   # Penaltı vuruşu olabilir
        if lc is not None and _dist(lc[0], lc[1], s["cx"], s["cy"]) < PSPOT_NEAR_BALL_PX:
            return False   # Top zaten o noktaya yöneliyordu
        return True
    return False

def _far_from_players(c):
    d = c.get("nearest_player_distance_px")
    return d is not None and d >= STATIC_ARTIFACT_PLAYER_DIST_MIN

def _static_penalty(c):
    if not _far_from_players(c) or not _suspect_static_spots:
        return 0.0
    for sx, sy in _suspect_static_spots:
        if _dist(c["cx"], c["cy"], sx, sy) <= STATIC_ARTIFACT_RADIUS_PX:
            return STATIC_ARTIFACT_PENALTY
    return 0.0

def _update_static_memory(best):
    global _last_static_candidate, _static_streak, _suspect_static_spots
    if best is None or not _far_from_players(best):
        _last_static_candidate = None
        _static_streak = 0
        return
    if _last_static_candidate is None:
        _last_static_candidate = {"cx": best["cx"], "cy": best["cy"]}
        _static_streak = 1
    else:
        d = _dist(best["cx"], best["cy"],
                  _last_static_candidate["cx"], _last_static_candidate["cy"])
        _static_streak = (_static_streak + 1) if d <= STATIC_ARTIFACT_RADIUS_PX else 1
        _last_static_candidate = {"cx": best["cx"], "cy": best["cy"]}

    if _static_streak >= STATIC_ARTIFACT_FRAMES:
        spot = (float(best["cx"]), float(best["cy"]))
        existing = [s for s in _suspect_static_spots
                    if _dist(spot[0], spot[1], s[0], s[1]) > STATIC_ARTIFACT_RADIUS_PX]
        _suspect_static_spots = (existing + [spot])[-STATIC_ARTIFACT_MAX_SPOTS:]

def estimate_shape_score(frame, x1, y1, x2, y2):
    H, W = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 <= x1 or y2 <= y1:
        return {"shape_score": 0.0, "aspect_ratio": 999.0, "fill_ratio": 0.0, "border_touch": True}

    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch < 4 or cw < 4:
        return {"shape_score": 0.0, "aspect_ratio": 999.0, "fill_ratio": 0.0, "border_touch": True}

    aspect = max(cw, ch) / max(1.0, min(cw, ch))
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"shape_score": 0.0, "aspect_ratio": float(aspect), "fill_ratio": 0.0, "border_touch": False}

    cnt  = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if area <= 1.0 or peri <= 1.0:
        return {"shape_score": 0.0, "aspect_ratio": float(aspect), "fill_ratio": 0.0, "border_touch": False}

    circ       = max(0.0, min(1.0, 4 * 3.14159265 * area / max(peri * peri, 1e-6)))
    fill       = max(0.0, min(1.0, area / max(float(cw * ch), 1.0)))
    asp_score  = max(0.0, 1.0 - max(0.0, aspect - ASPECT_RATIO_SOFT_LIMIT) /
                     max(ASPECT_RATIO_HARD_LIMIT - ASPECT_RATIO_SOFT_LIMIT, 1e-6))
    rx, ry, rw, rh = cv2.boundingRect(cnt)
    border = (rx <= 0) or (ry <= 0) or (rx+rw >= cw-1) or (ry+rh >= ch-1)
    score  = float(max(0.0, min(1.0, 0.55*circ + 0.25*fill + 0.20*asp_score)))

    return {"shape_score": score, "aspect_ratio": float(aspect),
            "fill_ratio": float(fill), "border_touch": bool(border)}

def valid_candidate(x1, y1, x2, y2, frame_h):
    bw, bh = x2-x1, y2-y1
    if bw < MIN_BALL_W or bw > MAX_BALL_W: return False
    if bh < MIN_BALL_H or bh > MAX_BALL_H: return False
    if max(bw, bh) / max(min(bw, bh), 1e-6) > MAX_ASPECT_RATIO: return False
    if (y1+y2)/2.0 < frame_h * TOP_REGION_BLOCK_RATIO: return False
    return True

def nearest_player(ball_xy, objects, id_to_label):
    if ball_xy is None:
        return None, None, None
    bx, by = ball_xy
    best_d, best_tid, best_lab = 1e18, None, None
    for obj in objects:
        tid = int(obj["track_id"])
        if id_to_label.get(tid) not in ("team_1", "team_2"):
            continue
        fx, fy = obj["foot_point_image_xy"]
        d = _dist(bx, by, float(fx), float(fy))
        if d < best_d:
            best_d, best_tid, best_lab = d, tid, id_to_label[tid]
    return (best_tid, best_lab, float(best_d)) if best_tid is not None else (None, None, None)

def score_and_choose(candidates, last_center, last_vel, miss):
    if not candidates:
        _update_static_memory(None)
        return None
    best, best_sc = None, -1e18
    for c in candidates:
        conf_sc  = c["conf"] * 1.55
        mot_sc   = jump_p = 0.0
        if last_center is not None and miss <= MAX_MISS_FRAMES:
            px, py = last_center[0] + last_vel[0], last_center[1] + last_vel[1]
            d = _dist(c["cx"], c["cy"], px, py)
            mot_sc = max(0.0, 1.0 - d / MOTION_DISTANCE_NORM)
            if d > MAX_ALLOWED_JUMP_PX:
                jump_p = 0.8 if miss <= 1 else 0.45
        foot_sc  = max(0.0, 1.0 - c["nearest_player_distance_px"] / FOOT_DISTANCE_NORM) \
                   if c["nearest_player_distance_px"] is not None else 0.0
        size_b   = 0.18 if 16 <= c["bw"]*c["bh"] <= 900 else 0.0
        stat_p   = _static_penalty(c)
        shp      = float(c.get("shape_score", 0.0))
        shp_p    = SHAPE_LOW_PENALTY if shp < SHAPE_LOW_THRESHOLD else 0.0
        ar       = float(c.get("shape_aspect_ratio", 1.0))
        elong_p  = ELONGATION_PENALTY_STRONG if ar >= ASPECT_RATIO_HARD_LIMIT else \
                   (ELONGATION_PENALTY_SOFT if ar > ASPECT_RATIO_SOFT_LIMIT else 0.0)
        bord_p   = BORDER_TOUCH_PENALTY if c.get("shape_border_touch") else 0.0
        fill_p   = FILL_RATIO_LOW_PENALTY if float(c.get("shape_fill_ratio", 1.0)) < FILL_RATIO_LOW_THRESHOLD else 0.0
        far_p    = 0.8 if (c["nearest_player_distance_px"] or 0) > 220 and c["conf"] < 0.10 else 0.0

        tn_cnf = float(c.get("tn_conflict_penalty", 0.0))
        sc = (conf_sc + 1.00*mot_sc + 0.72*foot_sc + size_b + SHAPE_WEIGHT*shp
              - jump_p - stat_p - shp_p - elong_p - bord_p - fill_p - far_p - tn_cnf)

        c.update(selection_score=float(sc), jump_penalty=float(jump_p),
                 static_penalty=float(stat_p), shape_penalty=float(shp_p),
                 elongation_penalty=float(elong_p), border_touch_penalty=float(bord_p),
                 fill_ratio_penalty=float(fill_p), shape_score_used=float(shp))
        if sc > best_sc:
            best_sc, best = sc, c

    _update_static_memory(best)
    return best


def _bbox_iou_xyxy(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, xi2 - xi1), max(0.0, yi2 - yi1)
    inter = iw * ih
    a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a + b - inter
    return (inter / union) if union > 1e-6 else 0.0


def _hn_dedup_key(c):
    g = float(HARD_NEG_GRID)
    return (
        int(c["cx"] // g),
        int(c["cy"] // g),
        int((c["x2"] - c["x1"]) // max(4.0, g * 0.3)),
        int((c["y2"] - c["y1"]) // max(4.0, g * 0.3)),
    )


def _is_same_candidate_as_chosen(c, chosen):
    if chosen is None:
        return False
    return _bbox_iou_xyxy(
        c["x1"], c["y1"], c["x2"], c["y2"],
        chosen["x1"], chosen["y1"], chosen["x2"], chosen["y2"],
    ) >= 0.92


def save_hard_negative_crop(frame, c, seq_idx, orig_idx, reason, hn_state, manifest_f):
    """YOLO top sınıfı üretmiş ama takipçi seçmemiş kutu → JPEG + manifest satırı."""
    if hn_state["count"] >= HARD_NEG_MAX_IMAGES:
        return
    if float(c.get("conf", 0)) < HARD_NEG_MIN_CONF:
        return
    if c.get("tn_solo"):
        return
    dk = _hn_dedup_key(c)
    if dk in hn_state["dedup"]:
        return
    hn_state["dedup"].add(dk)

    H, W = frame.shape[:2]
    x1, y1, x2, y2 = float(c["x1"]), float(c["y1"]), float(c["x2"]), float(c["y2"])
    bw, bh = x2 - x1, y2 - y1
    pad_w = bw * HARD_NEG_PAD_FRAC
    pad_h = bh * HARD_NEG_PAD_FRAC
    xa = max(0, int(x1 - pad_w))
    ya = max(0, int(y1 - pad_h))
    xb = min(W, int(x2 + pad_w))
    yb = min(H, int(y2 + pad_h))
    if (xb - xa) < HARD_NEG_MIN_SIDE or (yb - ya) < HARD_NEG_MIN_SIDE:
        return

    crop = frame[ya:yb, xa:xb]
    reason_tag = reason.replace(" ", "_")
    fn = (
        f"hn_s{seq_idx:06d}_o{orig_idx:06d}_{hn_state['count']:05d}"
        f"_c{c['conf']:.2f}_{reason_tag}.jpg"
    )
    path = HARD_NEGATIVES_DIR / "inbox" / fn
    if not cv2.imwrite(str(path), crop):
        hn_state["dedup"].discard(dk)
        return

    hn_state["count"] += 1
    if manifest_f is not None:
        manifest_f.write(
            json.dumps(
                {
                    "file": fn,
                    "seq_frame_index": seq_idx,
                    "original_frame_index": orig_idx,
                    "yolo_conf": round(float(c["conf"]), 4),
                    "reason": reason,
                    "bbox_xyxy": [
                        round(float(c["x1"]), 2),
                        round(float(c["y1"]), 2),
                        round(float(c["x2"]), 2),
                        round(float(c["y2"]), 2),
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# ─── REVIEW VİDEO YARDIMCILARI ───────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_panel(frame, lines, x=16, y=16, width=800, line_h=26):
    hbox = 14 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+width, y+hbox), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    yy = y + 26
    for ln in lines:
        cv2.putText(frame, ln, (x+10, yy), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        yy += line_h

def draw_zoom(frame, cx, cy, half=90, size=220):
    H, W = frame.shape[:2]
    x1, y1 = max(0, cx-half), max(0, cy-half)
    x2, y2 = min(W, cx+half), min(H, cy+half)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    zoom = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    x0, y0 = W - size - 20, 20
    frame[y0:y0+size, x0:x0+size] = zoom
    cv2.rectangle(frame, (x0, y0), (x0+size, y0+size), (255, 255, 255), 2)
    cv2.putText(frame, "ZOOM", (x0+8, y0+20), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

# ─── GİRİŞ KONTROLLERİ ───────────────────────────────────────────────────────
for p in [VIDEO_PATH, GAMEPLAY_JSON_PATH, TRACK_LABELS_JSON_PATH, MODEL_PATH]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Bulunamadı: {p}")

Path(OUTPUT_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)

_collect_hn = os.environ.get("COLLECT_BALL_HARD_NEGATIVES", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
_hn_state = {"count": 0, "dedup": set()}
_hn_manifest = None
if _collect_hn:
    HARD_NEGATIVES_DIR.mkdir(parents=True, exist_ok=True)
    (HARD_NEGATIVES_DIR / "inbox").mkdir(parents=True, exist_ok=True)
    (HARD_NEGATIVES_DIR / "confirmed").mkdir(parents=True, exist_ok=True)
    (HARD_NEGATIVES_DIR / "discarded").mkdir(parents=True, exist_ok=True)
    _hn_manifest = open(HARD_NEGATIVES_DIR / "manifest.jsonl", "w", encoding="utf-8")
    print(
        f"  Hard negative toplama AÇIK → {HARD_NEGATIVES_DIR} "
        f"(max {HARD_NEG_MAX_IMAGES}, min_conf={HARD_NEG_MIN_CONF})"
    )

with open(GAMEPLAY_JSON_PATH, "r", encoding="utf-8") as f:
    gameplay = json.load(f)
with open(TRACK_LABELS_JSON_PATH, "r", encoding="utf-8") as f:
    label_data = json.load(f)

frames_meta = gameplay["frames"]
fps         = float(gameplay["fps"])
width       = int(gameplay["width"])
height      = int(gameplay["height"])
id_to_label = {int(k): v["label"] for k, v in label_data["tracks"].items()}

# ─── TRACKNETV4 TESPİTLERİ YÜKLE (varsa) ────────────────────────────────────
_tn_detections: list = []
_tn_active = False
if Path(TRACKNET_JSON_PATH).exists():
    with open(TRACKNET_JSON_PATH) as _f:
        _tn_data = json.load(_f)
    _tn_detections = _tn_data.get("frames", [])
    _tn_active = True
    print(f"  TrackNetV4 ensemble AKTİF — "
          f"{_tn_data.get('detected', '?')} / {_tn_data.get('total_frames', '?')} tespit")
else:
    print(f"  TrackNetV4 ensemble pasif (tracknet_detections.json yok)")
    print(f"  → Daha iyi sonuç için: make tracknet-prepare && make tracknet-train && make tracknet-infer")

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Video açılamadı: {VIDEO_PATH}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not writer.isOpened():
    raise RuntimeError(f"Review video oluşturulamadı: {OUTPUT_VIDEO_PATH}")

# ─── TRACK DURUMU ────────────────────────────────────────────────────────────
last_center  = None
last_bbox    = None
last_vel     = (0.0, 0.0)
miss_count   = 999
_ball_kf     = None   # BallKalman instance — None until first detection
export_frames = []

# ─── ANA DÖNGÜ ───────────────────────────────────────────────────────────────
print(f"\nStage 5 | Ball Tracking — {len(frames_meta)} frame")
_pbar5 = tqdm(total=len(frames_meta), desc="Stage 5 | Ball Tracking",
              unit="frame", ncols=90)
try:
  for seq_idx, frame_meta in enumerate(frames_meta):
    ret, frame = cap.read()
    if not ret:
        break

    orig_idx  = int(frame_meta["original_frame_index"])
    time_sec  = float(frame_meta["time_sec"])
    objects   = frame_meta["objects"]

    # ── Kamera hareketi tahmini (LK optik flow) ───────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _refresh_bg_features(gray, seq_idx)
    cam_dx, cam_dy = _update_camera_flow(gray)
    _compensate_spots(cam_dx, cam_dy)

    # ── YOLO tespiti ──────────────────────────────────────────────────────────
    results = model.predict(source=frame, classes=[BALL_CLASS_ID],
                            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                            imgsz=IMGSZ, verbose=False)[0]

    candidates = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box, conf in zip(results.boxes.xyxy.cpu().numpy(),
                             results.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.tolist()
            if not valid_candidate(x1, y1, x2, y2, height):
                continue
            cx, cy = (x1+x2)/2, (y1+y2)/2
            bw, bh = x2-x1, y2-y1
            ntid, nlab, ndist = nearest_player((cx, cy), objects, id_to_label)
            shp = estimate_shape_score(frame, x1, y1, x2, y2)
            candidates.append({
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "cx": float(cx), "cy": float(cy), "bw": float(bw), "bh": float(bh),
                "conf": float(conf),
                "nearest_player_track_id":   ntid,
                "nearest_player_team_label": nlab,
                "nearest_player_distance_px": ndist,
                "shape_score":        shp["shape_score"],
                "shape_aspect_ratio": shp["aspect_ratio"],
                "shape_fill_ratio":   shp["fill_ratio"],
                "shape_border_touch": shp["border_touch"],
            })

    # ── Kalman predict (her frame, her durumda) ──────────────────────────────
    if _ball_kf is not None:
        kf_pred_cx, kf_pred_cy = _ball_kf.predict()
        kf_vel = _ball_kf.vel
        score_center = (kf_pred_cx, kf_pred_cy)
        score_vel    = kf_vel
    else:
        score_center = last_center
        score_vel    = last_vel

    # ── Penaltı noktası hafızasını güncelle ve sert dışla ────────────────────
    _update_spot_suspects(candidates)
    spot_excluded = []
    filtered_cands = []
    for c in candidates:
        if _is_penalty_spot(c["cx"], c["cy"],
                            c["nearest_player_distance_px"],
                            score_center):
            spot_excluded.append(c)
        else:
            filtered_cands.append(c)
    candidates = filtered_cands

    # ── TrackNetV4 Ensemble ───────────────────────────────────────────────────
    tn_det = None
    if _tn_active and seq_idx < len(_tn_detections):
        tn_det = _tn_detections[seq_idx]   # None veya {"cx":...,"cy":...,"conf":...}

    if tn_det is not None:
        tn_cx, tn_cy, tn_conf = tn_det["cx"], tn_det["cy"], tn_det["conf"]
        tn_conf_f = float(tn_conf)
        # Mevcut YOLO adaylarına yakınlık bonusu ekle
        matched = False
        for c in candidates:
            if _dist(c["cx"], c["cy"], tn_cx, tn_cy) <= TN_YOLO_MATCH_RADIUS:
                c["tn_bonus"]      = TN_BONUS_SCORE
                c["selection_score"] = c.get("selection_score", 0.0) + TN_BONUS_SCORE
                matched = True
        # TN en az bir YOLO ile hemfikirse: TN’den uzak diğer YOLO kutularını cezalandır.
        # matched==False iken ASLA ceza verme — yanlış heatmap tekesi doğru YOLO’yu susturmasın.
        if matched and tn_conf_f >= TN_CONFLICT_TN_CONF:
            cdist = TN_YOLO_MATCH_RADIUS * TN_CONFLICT_DIST_MULT
            for c in candidates:
                if _dist(c["cx"], c["cy"], tn_cx, tn_cy) > cdist:
                    c["tn_conflict_penalty"] = TN_CONFLICT_YOLO_PENALTY
        # YOLO eşleşmedi; TN güçlüyse → sentetik aday (drift sınırı; soğuk başlangıçta daha düşük conf)
        if not matched:
            cold_start = last_center is None and _ball_kf is None
            solo_need = TN_SOLO_COLD_START_CONF if cold_start else TN_SOLO_CONF_THRESH
            if tn_conf_f >= solo_need:
                if cold_start:
                    allow_solo = True
                else:
                    ref = score_center if score_center is not None else last_center
                    allow_solo = (
                        ref is not None
                        and _dist(tn_cx, tn_cy, ref[0], ref[1]) <= TN_SOLO_MAX_DRIFT_PX
                    )
            else:
                allow_solo = False
            if allow_solo:
                hw = 14.0
                ntid2, nlab2, ndist2 = nearest_player((tn_cx, tn_cy), objects, id_to_label)
                fake_cand = {
                    "x1": tn_cx - hw, "y1": tn_cy - hw,
                    "x2": tn_cx + hw, "y2": tn_cy + hw,
                    "cx": float(tn_cx), "cy": float(tn_cy),
                    "bw": hw * 2, "bh": hw * 2,
                    "conf": float(TN_SOLO_FAKE_CONF),
                    "nearest_player_track_id":    ntid2,
                    "nearest_player_team_label":  nlab2,
                    "nearest_player_distance_px": ndist2,
                    "shape_score": 0.65,
                    "shape_aspect_ratio": 1.0, "shape_fill_ratio": 0.7,
                    "shape_border_touch": False,
                    "tn_solo":  True, "tn_conf": tn_conf_f,
                    "tn_bonus": TN_BONUS_SCORE,
                }
                candidates.append(fake_cand)

    chosen = score_and_choose(candidates, score_center, score_vel, miss_count)

    # Sıçrama kapısı: sadece ardışık kilitli iz (miss==0). Kopuklukta yeniden yakalamayı engellemez.
    if (
        chosen is not None
        and score_center is not None
        and miss_count <= POST_PICK_GATE_MAX_MISS
    ):
        d_gate = _dist(chosen["cx"], chosen["cy"], score_center[0], score_center[1])
        tn_sup = bool(chosen.get("tn_solo")) or float(chosen.get("tn_bonus", 0) or 0) >= (
            TN_BONUS_SCORE * 0.499
        )
        lim = POST_PICK_MAX_JUMP_TN_PX if tn_sup else POST_PICK_MAX_JUMP_PX
        if d_gate > lim:
            chosen = None

    # ── Hard negative: YOLO top dedi, seçici başka kutuyu seçti veya hiç seçmedi ─
    if _collect_hn:
        for c in candidates:
            if _is_same_candidate_as_chosen(c, chosen):
                continue
            tag = "not_chosen" if chosen is not None else "all_rejected"
            save_hard_negative_crop(
                frame, c, seq_idx, orig_idx, tag, _hn_state, _hn_manifest
            )
        for c in spot_excluded:
            save_hard_negative_crop(
                frame, c, seq_idx, orig_idx, "penalty_spot_excluded", _hn_state, _hn_manifest
            )

    # ── Sonucu kaydet ─────────────────────────────────────────────────────────
    ball_info = {
        "visible": False, "interpolated": False,
        "confidence": None, "image_xy": None, "bbox_xyxy": None,
        "nearest_player_track_id": None, "nearest_player_team_label": None,
        "nearest_player_distance_px": None, "candidate_count": len(candidates),
        "selected_candidate": None,
        "candidates": [
            {"bbox_xyxy": [round2(c["x1"]), round2(c["y1"]), round2(c["x2"]), round2(c["y2"])],
             "conf": round2(c["conf"], 4),
             "shape_score": round2(c.get("shape_score_used", c.get("shape_score")), 3),
             "selection_score": round2(c.get("selection_score"), 3)}
            for c in candidates
        ],
    }

    draw_color = (0, 0, 255)
    status_text = "BALL: none"

    if chosen is not None:
        # ── Kalman update + smooth pozisyon ──────────────────────────────────
        if _ball_kf is None:
            _ball_kf = _BallKalman(chosen["cx"], chosen["cy"])
            kx, ky = chosen["cx"], chosen["cy"]
        else:
            kx, ky = _ball_kf.update(chosen["cx"], chosen["cy"])

        last_vel    = _ball_kf.vel
        last_center = (kx, ky)
        last_bbox   = (chosen["x1"], chosen["y1"], chosen["x2"], chosen["y2"])
        miss_count  = 0

        ball_info.update(
            visible=True, interpolated=False,
            confidence=round2(chosen["conf"], 4),
            image_xy=[round2(kx), round2(ky)],
            image_xy_raw=[round2(chosen["cx"]), round2(chosen["cy"])],
            bbox_xyxy=[round2(chosen["x1"]), round2(chosen["y1"]),
                       round2(chosen["x2"]), round2(chosen["y2"])],
            nearest_player_track_id=chosen["nearest_player_track_id"],
            nearest_player_team_label=chosen["nearest_player_team_label"],
            nearest_player_distance_px=round2(chosen["nearest_player_distance_px"]),
            selected_candidate={
                k: round2(v, 3) if isinstance(v, float) else v
                for k, v in chosen.items()
            },
        )
        status_text = f"BALL: visible | conf={chosen['conf']:.2f} | cand={len(candidates)}"

        # çiz: seçilen aday
        x1i, y1i, x2i, y2i = map(lambda z: int(round(z)),
                                   [chosen["x1"], chosen["y1"], chosen["x2"], chosen["y2"]])
        cxi, cyi = int(round(kx)), int(round(ky))
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 0, 255), 3)
        cv2.circle(frame, (cxi, cyi), 9, (255, 255, 255), -1)
        cv2.circle(frame, (cxi, cyi), 16, (0, 0, 255), 3)
        draw_zoom(frame, cxi, cyi)

        shp_sc  = float(chosen.get("shape_score_used", 0.0))
        sel_sc  = float(chosen.get("selection_score", 0.0))
        shp_p   = float(chosen.get("shape_penalty", 0.0))
        jmp_p   = float(chosen.get("jump_penalty", 0.0))
        sta_p   = float(chosen.get("static_penalty", 0.0))
        cv2.putText(frame, f"shape={shp_sc:.2f} sel={sel_sc:.2f}",
                    (x1i, min(height-35, y2i+20)), _FONT, 0.55, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"shP={shp_p:.2f} jmpP={jmp_p:.2f} staP={sta_p:.2f}",
                    (x1i, min(height-10, y2i+42)), _FONT, 0.5, (0,0,255), 2, cv2.LINE_AA)

    elif last_center is not None and last_bbox is not None and miss_count < MAX_MISS_FRAMES:
        # ── Kalman ile daha doğru interpolasyon ──────────────────────────────
        if _ball_kf is not None:
            pred_x, pred_y = kf_pred_cx, kf_pred_cy
            last_vel = kf_vel
        else:
            last_vel = (last_vel[0] * 0.90, last_vel[1] * 0.90)
            pred_x   = last_center[0] + last_vel[0]
            pred_y   = last_center[1] + last_vel[1]

        bw   = last_bbox[2] - last_bbox[0]
        bh   = last_bbox[3] - last_bbox[1]
        pred = (pred_x, pred_y)
        pb   = (pred_x-bw/2, pred_y-bh/2, pred_x+bw/2, pred_y+bh/2)
        last_center, last_bbox = pred, pb
        miss_count += 1

        ntid, nlab, ndist = nearest_player(pred, objects, id_to_label)
        ball_info.update(
            visible=False, interpolated=True,
            image_xy=[round2(pred[0]), round2(pred[1])],
            bbox_xyxy=[round2(pb[0]), round2(pb[1]), round2(pb[2]), round2(pb[3])],
            nearest_player_track_id=ntid,
            nearest_player_team_label=nlab,
            nearest_player_distance_px=round2(ndist),
        )
        status_text = f"BALL: interpolated | miss={miss_count}"
        draw_color  = (255, 0, 255)

        cxi, cyi = int(round(pred[0])), int(round(pred[1]))
        cv2.rectangle(frame, tuple(map(lambda z: int(round(z)), [pb[0], pb[1]])),
                             tuple(map(lambda z: int(round(z)), [pb[2], pb[3]])), (255, 0, 255), 2)
        cv2.circle(frame, (cxi, cyi), 8, (255, 0, 255), 2)
    else:
        miss_count += 1
        # Uzun süre top yok → Kalman sıfırla (yeni sahneye geçiş olabilir)
        if miss_count > MAX_MISS_FRAMES * 3 and _ball_kf is not None:
            _ball_kf = None

    # diğer adayları koy
    for c in candidates:
        if c is chosen:
            continue
        cv2.rectangle(frame,
                      (int(c["x1"]), int(c["y1"])),
                      (int(c["x2"]), int(c["y2"])), (0, 255, 255), 1)
        cv2.putText(frame, f"{c.get('selection_score', 0.0):.2f}",
                    (int(c["x1"]), max(16, int(c["y1"])-4)),
                    _FONT, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # Penaltı noktası olarak dışlanan adayları turuncu ile göster
    for c in spot_excluded:
        cv2.rectangle(frame,
                      (int(c["x1"]) - 2, int(c["y1"]) - 2),
                      (int(c["x2"]) + 2, int(c["y2"]) + 2), (0, 128, 255), 2)
        cv2.putText(frame, "PSPOT",
                    (int(c["x1"]), max(16, int(c["y1"]) - 4)),
                    _FONT, 0.4, (0, 128, 255), 1, cv2.LINE_AA)

    # panel
    ntid_txt = ball_info["nearest_player_track_id"]
    excl_txt = f"  pspot_excl={len(spot_excluded)}" if spot_excluded else ""
    draw_panel(frame, [
        f"Stage 5 — Ball Tracking | seq={seq_idx} orig={orig_idx} t={time_sec:.2f}s",
        status_text,
        f"nearest_player=ID{ntid_txt} ({ball_info['nearest_player_team_label']}) "
        f"d={ball_info['nearest_player_distance_px']}px",
        f"candidates={len(candidates)}  miss={miss_count}{excl_txt}",
    ])

    export_frames.append({
        "seq_frame_index":        seq_idx,
        "original_frame_index":   orig_idx,
        "time_sec":               round2(time_sec, 3),
        "ball":                   ball_info,
    })
    writer.write(frame)
    _pbar5.update(1)
finally:
    _pbar5.close()
    if _hn_manifest is not None:
        _hn_manifest.close()
        if _collect_hn:
            print(f"  hard negatives: {_hn_state['count']} görsel → {HARD_NEGATIVES_DIR}")

cap.release()
writer.release()

# ─── JSON KAYDET ──────────────────────────────────────────────────────────────
output = {
    "source_video_path":            VIDEO_PATH,
    "source_gameplay_json_path":    GAMEPLAY_JSON_PATH,
    "source_track_labels_json_path": TRACK_LABELS_JSON_PATH,
    "model_path":                   MODEL_PATH,
    "ball_class_id":                BALL_CLASS_ID,
    "fps": fps, "width": width, "height": height,
    "frame_count":  len(export_frames),
    "frames":       export_frames,
}

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

visible_count = sum(1 for x in export_frames if x["ball"]["visible"])
interp_count  = sum(1 for x in export_frames if x["ball"]["interpolated"])

print("\nDONE")
print(f"  ball JSON   : {OUTPUT_JSON_PATH}")
print(f"  review video: {OUTPUT_VIDEO_PATH}")
print(f"  total frames: {len(export_frames)}")
print(f"  visible     : {visible_count}")
print(f"  interpolated: {interp_count}")
