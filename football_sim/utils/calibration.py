"""
utils/calibration.py
====================
Homografi hesaplama.
Elle verilen noktalar varsa onları kullanır (çok daha güvenilir).
Yoksa otomatik dener.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Optional
import logging
import config as cfg

log = logging.getLogger(__name__)

WORLD_4 = np.array([
    [0.0,   68.0],
    [105.0, 68.0],
    [105.0, 0.0 ],
    [0.0,   0.0 ],
], dtype=np.float32)


class Calibrator:

    def __init__(self, frame_w: int, frame_h: int):
        self.W = frame_w
        self.H = frame_h
        self.H_mat: Optional[np.ndarray] = None
        self._last_good: Optional[np.ndarray] = None
        self.is_calibrated = False
        self._auto_fail_count = 0
        self._initialized = False

    def initialize(self) -> None:
        """Başlangıçta bir kez çalışır — elle nokta varsa hemen kalibre eder."""
        if cfg.MANUAL_CALIB_POINTS is not None:
            self._from_manual()
        self._initialized = True

    def _from_manual(self) -> None:
        pts = cfg.MANUAL_CALIB_POINTS
        src = np.array(pts["pixel"], dtype=np.float32)
        dst = np.array(pts["world"], dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is not None:
            self.H_mat = H
            self._last_good = H.copy()
            self.is_calibrated = True
            log.info(f"  ✅ Manuel kalibrasyon yüklendi ({len(src)} nokta)")
        else:
            log.warning("  ⚠  Manuel noktalardan homografi hesaplanamadı")

    def update(self, frame: np.ndarray) -> None:
        """Her N frame'de çağrılır. Manuel varsa güncelleme yapmaz."""
        if not self._initialized:
            self.initialize()

        # Manuel kalibrasyon varsa otomatik deneme yapma
        if cfg.MANUAL_CALIB_POINTS is not None:
            return

        H = self._auto_detect(frame)
        if H is not None and self._sanity(H):
            self.H_mat = H
            self._last_good = H.copy()
            self.is_calibrated = True
            self._auto_fail_count = 0
        else:
            self._auto_fail_count += 1
            if self._last_good is not None:
                self.H_mat = self._last_good
            else:
                self._fallback()

    def pixel_to_field(self, px: float, py: float) -> tuple[float, float]:
        if self.H_mat is None:
            return self._linear(px, py)
        pt  = np.array([[[float(px), float(py)]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, self.H_mat)
        fx  = float(np.clip(dst[0][0][0], 0.0, cfg.FIELD_W))
        fy  = float(np.clip(dst[0][0][1], 0.0, cfg.FIELD_H))
        return round(fx, 2), round(fy, 2)

    # ── Otomatik tespit ────────────────────────────────────────────────────

    def _auto_detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Yeşil saha maskesi → en büyük kontur → 4 köşe → homografi."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1   = cv2.inRange(hsv, np.array([30,  30, 30]), np.array([90, 255, 255]))
        m2   = cv2.inRange(hsv, np.array([25,  20, 20]), np.array([95, 255, 200]))
        mask = cv2.bitwise_or(m1, m2)

        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.W * self.H * 0.15:
            return None

        hull = cv2.convexHull(largest)
        eps  = 0.02 * cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2).astype(np.float32)

        corners = self._to_four(poly)
        if corners is None:
            return None

        H, _ = cv2.findHomography(corners, WORLD_4, cv2.RANSAC, 5.0)
        return H

    @staticmethod
    def _to_four(pts: np.ndarray) -> Optional[np.ndarray]:
        if len(pts) < 4:
            return None
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _sanity(self, H: np.ndarray) -> bool:
        corners = np.array([
            [[0.0,        0.0       ]],
            [[float(self.W), 0.0   ]],
            [[float(self.W), float(self.H)]],
            [[0.0,        float(self.H)]],
        ], dtype=np.float32)
        try:
            world = cv2.perspectiveTransform(corners, H)
        except Exception:
            return False
        xs, ys = world[:, 0, 0], world[:, 0, 1]
        return (xs.min() > -25 and xs.max() < 130 and
                ys.min() > -25 and ys.max() < 93)

    def _fallback(self) -> None:
        src = np.array([
            [self.W * 0.05, self.H * 0.25],
            [self.W * 0.95, self.H * 0.25],
            [self.W * 0.95, self.H * 0.95],
            [self.W * 0.05, self.H * 0.95],
        ], dtype=np.float32)
        H, _ = cv2.findHomography(src, WORLD_4, 0)
        self.H_mat = H

    def _linear(self, px: float, py: float) -> tuple[float, float]:
        return round(px / self.W * cfg.FIELD_W, 2), round(py / self.H * cfg.FIELD_H, 2)
