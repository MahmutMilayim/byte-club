import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class ShotEvent:
    frame_idx: int
    time: float
    event_type: str
    team_id: Optional[int]
    shooter_id: Optional[int]
    target_side: str
    speed_px_s: float
    displacement_px: float
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    confidence: float

class ShotGoalDetector:
    def __init__(self, width: int = 1280, height: int = 720, fps: float = 25.0):
        self.width = width
        self.height = height
        self.fps = fps

        self._calibrated = False
        self._x_low = 0.0
        self._x_high = float(width)

        self._goal_x_eps = 20.0
        self._goal_y_min = int(height * 0.35)
        self._goal_y_max = int(height * 0.65)

        self._min_shot_speed = 250.0
        self._strong_shot_speed = 400.0
        self._min_progress = 40.0
        self._min_disp = 60.0

        self._goal_check_window = 10
        self._pos_window_start = 8
        self._pos_window_end = 10
        self._team_window = 20

    def detect(self, ball_segments: List[Dict[str, Any]], frames_data: List[Dict[str, Any]]) -> List[ShotEvent]:
        frame_map = {int(f["frame_idx"]): f for f in frames_data if "frame_idx" in f}

        if not self._calibrated:
            self._calibrate(frames_data, ball_segments)

        events: List[ShotEvent] = []

        for seg in ball_segments:
            start_frame = int(seg.get("start_time", -1))
            end_frame = int(seg.get("end_time", -1))
            if start_frame < 0 or end_frame < 0 or end_frame <= start_frame:
                continue

            start_owner = seg.get("start_owner", None)
            end_owner = seg.get("end_owner", None)
            if start_owner is None:
                continue

            start_pos = self._find_ball_pos_near(frame_map, start_frame, window=self._pos_window_start)
            end_pos = self._find_ball_pos_near(frame_map, end_frame, window=self._pos_window_end)
            if start_pos is None or end_pos is None:
                continue

            dx = end_pos[0] - start_pos[0]
            target_side = "RIGHT" if dx >= 0 else "LEFT"
            goal_x = self._x_high if target_side == "RIGHT" else self._x_low

            speed_pf = self._to_float(seg.get("average_speed", 0.0), default=0.0)
            speed_px_s = speed_pf * self.fps
            displacement_px = self._to_float(seg.get("displacement", 0.0), default=0.0)

            shooter_id = self._to_int(start_owner)
            shooter_team = self._get_team_id(shooter_id, start_frame, frame_map, window=self._team_window) if shooter_id is not None else None

            end_owner_id = self._to_int(end_owner)
            end_team = self._get_team_id(end_owner_id, end_frame, frame_map, window=self._team_window) if end_owner_id is not None else None

            if not self._is_potential_shot(
                start_pos=start_pos,
                end_pos=end_pos,
                goal_x=goal_x,
                target_side=target_side,
                speed_px_s=speed_px_s,
                displacement_px=displacement_px,
                start_owner=start_owner,
                end_owner=end_owner,
                start_team=shooter_team,
                end_team=end_team,
            ):
                continue

            goal_frame, goal_pos = self._goal_hit_near_end(frame_map, end_frame, goal_x, window=self._goal_check_window)
            is_goal = goal_frame is not None

            final_frame = goal_frame if is_goal else start_frame
            final_pos = goal_pos if is_goal else end_pos

            confidence = self._score_confidence(
                is_goal=is_goal,
                speed_px_s=speed_px_s,
                displacement_px=displacement_px,
                progress_px=self._progress_towards_goal(start_pos, end_pos, goal_x),
                end_pos=final_pos,
                goal_x=goal_x,
            )

            events.append(
                ShotEvent(
                    frame_idx=int(final_frame),
                    time=float(final_frame) / self.fps,
                    event_type="GOAL" if is_goal else "SHOT",
                    team_id=shooter_team,
                    shooter_id=shooter_id,
                    target_side=target_side,
                    speed_px_s=float(speed_px_s),
                    displacement_px=float(displacement_px),
                    start_pos=start_pos,
                    end_pos=final_pos,
                    confidence=float(confidence),
                )
            )

        return events

    def _goal_hit_near_end(
        self,
        frame_map: Dict[int, Dict[str, Any]],
        end_frame: int,
        goal_x: float,
        window: int = 10,
    ) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
        best_frame = None
        best_pos = None
        for i in range(end_frame - window, end_frame + 1):
            fr = frame_map.get(i)
            if fr is None:
                continue
            pos = self._get_ball_pos_from_frame(fr)
            if pos is None:
                continue
            if self._is_goal(pos, goal_x):
                best_frame = i
                best_pos = pos
        return best_frame, best_pos

    def _calibrate(self, frames_data: List[Dict[str, Any]], ball_segments: List[Dict[str, Any]]) -> None:
        xs: List[float] = []
        for fr in frames_data:
            pos = self._get_ball_pos_from_frame(fr)
            if pos is not None:
                xs.append(float(pos[0]))

        if len(xs) >= 30:
            self._x_low = self._quantile(xs, 0.02)
            self._x_high = self._quantile(xs, 0.98)
            if (self._x_high - self._x_low) < 50:
                self._x_low = 0.0
                self._x_high = float(self.width)
        else:
            self._x_low = 0.0
            self._x_high = float(self.width)

        span = max(1.0, self._x_high - self._x_low)
        self._goal_x_eps = max(12.0, span * 0.03)

        speeds: List[float] = []
        for seg in ball_segments:
            sp = seg.get("average_speed", None)
            if sp is None:
                continue
            v = self._to_float(sp, default=None)
            if v is None:
                continue
            speeds.append(v * self.fps)

        if len(speeds) >= 20:
            q50 = self._quantile(speeds, 0.50)
            q85 = self._quantile(speeds, 0.85)
            q93 = self._quantile(speeds, 0.93)
            self._min_shot_speed = max(150.0, q85, q50 * 1.8)
            self._strong_shot_speed = max(self._min_shot_speed * 1.25, q93)
        else:
            self._min_shot_speed = 250.0
            self._strong_shot_speed = 400.0

        self._min_progress = max(30.0, span * 0.06)
        self._min_disp = max(60.0, span * 0.10)

        self._calibrated = True

    def _find_ball_pos_near(
        self,
        frame_map: Dict[int, Dict[str, Any]],
        frame_idx: int,
        window: int = 8,
    ) -> Optional[Tuple[float, float]]:
        best = None
        best_dist = 10**9
        for i in range(frame_idx - window, frame_idx + window + 1):
            fr = frame_map.get(i)
            if fr is None:
                continue
            pos = self._get_ball_pos_from_frame(fr)
            if pos is None:
                continue
            d = abs(i - frame_idx)
            if d < best_dist:
                best = pos
                best_dist = d
        return best

    def _get_ball_pos_from_frame(self, frame: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        for t in frame.get("tracks", []):
            cls_name = str(t.get("cls", "")).lower()
            if cls_name == "ball" or "ball" in cls_name:
                bbox = t.get("bbox", None)
                if bbox and len(bbox) == 4:
                    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
                    cy = (float(bbox[1]) + float(bbox[3])) / 2.0
                    return (cx, cy)
        return None

    def _is_goal(self, pos: Tuple[float, float], goal_x: float) -> bool:
        x, y = pos
        in_y = self._goal_y_min <= y <= self._goal_y_max
        if goal_x <= (self._x_low + self._goal_x_eps):
            in_x = x <= (self._x_low + self._goal_x_eps)
        else:
            in_x = x >= (self._x_high - self._goal_x_eps)
        return in_x and in_y

    def _is_potential_shot(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        goal_x: float,
        target_side: str,
        speed_px_s: float,
        displacement_px: float,
        start_owner: Any,
        end_owner: Any,
        start_team: Optional[int],
        end_team: Optional[int],
    ) -> bool:
        if displacement_px < self._min_disp:
            return False

        if speed_px_s < self._min_shot_speed:
            return False

        progress = self._progress_towards_goal(start_pos, end_pos, goal_x)
        if progress < self._min_progress:
            return False

        dx = end_pos[0] - start_pos[0]
        if target_side == "RIGHT" and dx < 0:
            return False
        if target_side == "LEFT" and dx > 0:
            return False

        span = max(1.0, self._x_high - self._x_low)
        origin_dist = abs(goal_x - start_pos[0])
        near_final_third = origin_dist <= (0.55 * span)
        if not near_final_third and speed_px_s < self._strong_shot_speed:
            return False

        end_near_goal = abs(goal_x - end_pos[0]) <= (self._goal_x_eps * 1.8)

        if end_owner is None:
            return True

        if start_team is None or end_team is None:
            return end_near_goal

        if start_team != end_team:
            return True

        return end_near_goal

    def _progress_towards_goal(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        goal_x: float,
    ) -> float:
        d0 = abs(goal_x - start_pos[0])
        d1 = abs(goal_x - end_pos[0])
        return d0 - d1

    def _score_confidence(
        self,
        is_goal: bool,
        speed_px_s: float,
        displacement_px: float,
        progress_px: float,
        end_pos: Tuple[float, float],
        goal_x: float,
    ) -> float:
        span = max(1.0, self._x_high - self._x_low)

        denom = max(1.0, (self._strong_shot_speed - self._min_shot_speed))
        s_speed = min(1.0, max(0.0, (speed_px_s - self._min_shot_speed) / denom))
        s_disp = min(1.0, displacement_px / max(1.0, 0.25 * span))
        s_prog = min(1.0, progress_px / max(1.0, 0.20 * span))

        end_x, end_y = end_pos
        x_near = min(1.0, max(0.0, 1.0 - (abs(goal_x - end_x) / max(1.0, self._goal_x_eps * 3.0))))
        y_mid = 1.0 if (self._goal_y_min <= end_y <= self._goal_y_max) else 0.0

        base = 0.34 * s_speed + 0.18 * s_disp + 0.26 * s_prog + 0.17 * x_near + 0.05 * y_mid
        if is_goal:
            base = min(1.0, base + 0.25 * x_near + 0.10 * y_mid)

        return float(max(0.0, min(1.0, base)))

    def _get_team_id(self, player_id: Optional[int], frame_idx: int, frame_map: Dict[int, Dict[str, Any]], window: int = 20) -> Optional[int]:
        if player_id is None:
            return None

        teams: List[int] = []
        for i in range(frame_idx - window, frame_idx + window + 1):
            fr = frame_map.get(i)
            if fr is None:
                continue
            for t in fr.get("tracks", []):
                if t.get("track_id") == player_id:
                    team = t.get("team", None)
                    if team is not None:
                        try:
                            teams.append(int(team))
                        except Exception:
                            pass

        if not teams:
            return None

        counts: Dict[int, int] = {}
        for tm in teams:
            counts[tm] = counts.get(tm, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _quantile(self, values: List[float], q: float) -> float:
        if not values:
            return 0.0
        v = sorted(values)
        if q <= 0:
            return float(v[0])
        if q >= 1:
            return float(v[-1])
        pos = (len(v) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(v[lo])
        w = pos - lo
        return float(v[lo] * (1.0 - w) + v[hi] * w)

    def _to_int(self, x: Any) -> Optional[int]:
        if x is None:
            return None
        try:
            return int(x)
        except Exception:
            return None

    def _to_float(self, x: Any, default: Any = 0.0) -> Any:
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            return default
