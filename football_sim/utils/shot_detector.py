"""
utils/shot_detector.py
======================
Piksel hızı + yön tabanlı şut tespiti.
Kalibrasyon olmasa bile çalışır (piksel koordinatları kullanır).
Kalibrasyon varsa metre koordinatlarına çevirir → daha doğru.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import logging
import config as cfg

log = logging.getLogger(__name__)


@dataclass
class PlayerSnap:
    track_id: int
    team: int
    field_x: float
    field_y: float
    px: float   # piksel x
    py: float   # piksel y


@dataclass
class BallSnap:
    visible: bool
    field_x: float = 0.0
    field_y: float = 0.0
    px: float = 0.0
    py: float = 0.0


@dataclass
class ShotEvent:
    frame_idx:    int
    time_sec:     float
    shooter_id:   int
    shooter_team: int
    target_goal:  str
    is_goal:      bool
    players:      list[PlayerSnap]
    ball:         BallSnap


class ShotDetector:

    def __init__(self, fps: float, frame_w: int, frame_h: int):
        self.fps      = fps
        self.frame_w  = frame_w
        self.frame_h  = frame_h
        # (frame_idx, px, py, field_x, field_y)
        self._hist: deque[tuple] = deque(maxlen=12)
        self._last_shot_frame = -(cfg.SHOT_COOLDOWN_FRAMES * 10)
        self._pending: Optional[dict] = None
        self._pending_ttl = 0
        self.events: list[ShotEvent] = []

    def update(
        self,
        frame_idx: int,
        players: list[PlayerSnap],
        ball: Optional[BallSnap],
    ) -> Optional[ShotEvent]:

        # ── Bekleyen şut gol kontrolü ──────────────────────────────────────
        if self._pending is not None:
            self._pending_ttl -= 1
            result = self._check_goal(ball)
            if result is not None:
                self._pending = None
                self.events.append(result)
                return result
            if self._pending_ttl <= 0:
                ev = self._finalize(is_goal=False)
                self._pending = None
                self.events.append(ev)
                return ev

        if ball is None or not ball.visible:
            return None

        self._hist.append((frame_idx, ball.px, ball.py, ball.field_x, ball.field_y))

        if frame_idx - self._last_shot_frame < cfg.SHOT_COOLDOWN_FRAMES:
            return None

        if len(self._hist) < 5:
            return None

        candidate = self._analyze(frame_idx, players, ball)
        if candidate is not None:
            self._last_shot_frame = frame_idx
            self._pending = candidate
            self._pending_ttl = cfg.GOAL_CHECK_FRAMES

        return None

    def _analyze(
        self, frame_idx: int, players: list[PlayerSnap], ball: BallSnap
    ) -> Optional[dict]:
        hist = list(self._hist)

        # Son 5 frame'den hız hesapla
        f0, px0, py0, fx0, fy0 = hist[-5]
        f1, px1, py1, fx1, fy1 = hist[-1]

        dt_frames = f1 - f0
        if dt_frames <= 0:
            return None

        # ── Piksel hızı (her zaman çalışır) ───────────────────────────────
        dpx = px1 - px0
        dpy = py1 - py0
        pixel_speed = np.hypot(dpx, dpy) / dt_frames  # px/frame

        if not (cfg.SHOT_MIN_PIXEL_SPEED < pixel_speed < cfg.SHOT_MAX_PIXEL_SPEED):
            return None

        # ── Yön analizi ────────────────────────────────────────────────────
        # Kale sol tarafta mı sağda mı? Bu kameraya göre:
        # Kalenin pixel konumu frame genişliğine göre belirlenir
        # Sol kale: px < frame_w * 0.3
        # Sağ kale: px > frame_w * 0.7
        ball_px_ratio = ball.px / self.frame_w

        # Top yönü
        if dpx > 0:
            target_goal   = "RIGHT"
            goal_px_ratio = 1.0   # sağ kenar
        else:
            target_goal   = "LEFT"
            goal_px_ratio = 0.0   # sol kenar

        # Top kale bölgesine yönelik mi? (Y ekseni)
        # Kale ekranın ortasında dikey olarak — orta %60 bandı
        py_ratio = ball.py / self.frame_h
        if not (0.15 < py_ratio < 0.85):
            return None

        # ── Mesafe kontrolü (kalibrasyon varsa metre, yoksa piksel ratio) ─
        if hasattr(ball, 'field_x') and ball.field_x > 0:
            # Metre bazlı
            goal_x   = cfg.FIELD_W if target_goal == "RIGHT" else 0.0
            dist     = abs(ball.field_x - goal_x)
            if dist > cfg.SHOT_MAX_DIST_FROM_GOAL_M:
                return None
        else:
            # Piksel ratio bazlı — 40m ≈ sahの %38'i
            dist_ratio = abs(ball_px_ratio - goal_px_ratio)
            if dist_ratio > 0.75:
                return None

        # ── Y yönü kale direği arasında mı? ───────────────────────────────
        # Kale direkler frame yüksekliğinin yaklaşık %35-%65 arasında
        if not (0.25 < py_ratio < 0.75):
            return None

        # ── Şutçuyu bul ───────────────────────────────────────────────────
        shooter = self._nearest_player(ball, players)
        if shooter is None:
            return None

        return {
            "frame_idx":    frame_idx,
            "time_sec":     round(frame_idx / self.fps, 3),
            "shooter_id":   shooter.track_id,
            "shooter_team": shooter.team,
            "target_goal":  target_goal,
            "players":      players[:],
            "ball":         ball,
        }

    def _check_goal(self, ball: Optional[BallSnap]) -> Optional[ShotEvent]:
        if ball is None or not ball.visible or self._pending is None:
            return None

        target = self._pending["target_goal"]
        ball_b = self._pending["ball"]

        # Metre koordinatı varsa kullan
        if ball.field_x > 0:
            if target == "LEFT":
                is_goal = (ball.field_x <= cfg.GOAL_X_LEFT_MAX and
                           cfg.GOAL_Y_MIN <= ball.field_y <= cfg.GOAL_Y_MAX)
            else:
                is_goal = (ball.field_x >= cfg.GOAL_X_RIGHT_MIN and
                           cfg.GOAL_Y_MIN <= ball.field_y <= cfg.GOAL_Y_MAX)
        else:
            # Piksel bazlı kale kontrolü
            px_ratio = ball.px / self.frame_w
            py_ratio = ball.py / self.frame_h
            if target == "LEFT":
                is_goal = px_ratio < 0.08 and 0.30 < py_ratio < 0.70
            else:
                is_goal = px_ratio > 0.92 and 0.30 < py_ratio < 0.70

        if is_goal:
            return self._finalize(is_goal=True)
        return None

    def _finalize(self, is_goal: bool) -> ShotEvent:
        p  = self._pending
        ev = ShotEvent(
            frame_idx    = p["frame_idx"],
            time_sec     = p["time_sec"],
            shooter_id   = p["shooter_id"],
            shooter_team = p["shooter_team"],
            target_goal  = p["target_goal"],
            is_goal      = is_goal,
            players      = p["players"],
            ball         = p["ball"],
        )
        tag = "⚽ GOL!" if is_goal else "🦶 Şut"
        log.info(f"  {tag}  t={ev.time_sec:.1f}s  "
                 f"ID={ev.shooter_id}  hedef={ev.target_goal}")
        return ev

    @staticmethod
    def _nearest_player(ball: BallSnap, players: list[PlayerSnap]) -> Optional[PlayerSnap]:
        """Topa en yakın oyuncuyu döner — piksel mesafesi kullanır."""
        best, best_d = None, float("inf")
        for p in players:
            d = np.hypot(p.px - ball.px, p.py - ball.py)
            if d < best_d:
                best_d, best = d, p
        # Max ~120px uzaktaki oyuncu şutçu olamaz
        if best_d > 120:
            return None
        return best

    def is_shot_candidate(self) -> bool:
        """
        Son N frame'de top hizlandi mi? 
        Evet ise oyuncu tespiti de yap.
        """
        hist = list(self._hist)
        if len(hist) < 4:
            return False
        f0, px0, py0, fx0, fy0 = hist[-4]
        f1, px1, py1, fx1, fy1 = hist[-1]
        dt = f1 - f0
        if dt <= 0:
            return False
        speed = np.hypot(px1-px0, py1-py0) / dt
        return speed > cfg.SHOT_MIN_PIXEL_SPEED * 0.6
