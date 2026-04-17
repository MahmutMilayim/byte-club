"""
utils/team_classifier.py
"""
from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional
import logging
import config as cfg

log = logging.getLogger(__name__)


class TeamClassifier:

    def __init__(self):
        self.km: Optional[KMeans] = None
        self._samples: list[np.ndarray] = []
        self._fitted  = False
        self.frames_seen = 0

    def collect(self, frame: np.ndarray, x1, y1, x2, y2) -> None:
        if self._fitted:
            return
        feat = self._feat(frame, x1, y1, x2, y2)
        if feat is not None:
            self._samples.append(feat)

    def try_fit(self) -> bool:
        if self._fitted or len(self._samples) < 20:
            return False
        self.km = KMeans(n_clusters=2, n_init=20, random_state=42)
        self.km.fit(np.array(self._samples))
        self._fitted = True
        log.info(f"  ✅ TeamClassifier eğitildi ({len(self._samples)} örnek)")
        return True

    def predict(self, frame: np.ndarray, x1, y1, x2, y2) -> int:
        if not self._fitted or self.km is None:
            return 0
        feat = self._feat(frame, x1, y1, x2, y2)
        if feat is None:
            return 0
        return int(self.km.predict(feat.reshape(1, -1))[0])

    def _feat(self, frame: np.ndarray, x1, y1, x2, y2) -> Optional[np.ndarray]:
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0 or crop.shape[0] < 10:
            return None
        h = max(1, int(crop.shape[0] * 0.40))
        torso = crop[:h]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
        mask  = cv2.bitwise_not(green)
        pixels = hsv[mask > 0]
        if len(pixels) < 15:
            return None
        return np.array([
            pixels[:,0].mean(), pixels[:,0].std(),
            pixels[:,1].mean(), pixels[:,1].std(),
        ], dtype=np.float32)

    @property
    def is_ready(self) -> bool:
        return self._fitted
