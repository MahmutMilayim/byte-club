"""
K5 v4 - Shot Detection (Segment-Based with Pass Filtering + Goal Zone Detection)

Mantık:
1. Ball segment'ten: start_owner → end_owner analizi
2. Aynı takıma geçtiyse = PAS (şut değil)
3. Farklı takıma veya kayıp = ŞUT olabilir
4. Şut anında soccer_line ile kale pozisyonunu tespit et
5. Şut sonrası top kale bölgesine girdiyse = GOL

ID Formatı:
- Oyuncular: L1-L11 (sol takım), R1-R11 (sağ takım)
- Top: "ball"
"""
import json
import math
import cv2
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path


def get_team_from_id(track_id: str) -> Optional[int]:
    """Track ID'sinden takımı döndürür (0=Left, 1=Right)."""
    if not track_id or track_id == "ball":
        return None
    if isinstance(track_id, str):
        if track_id.startswith("L"):
            return 0
        elif track_id.startswith("R"):
            return 1
    return None


@dataclass
class ShotEvent:
    frame_idx: int          # Şutun atıldığı frame
    time: float             # Saniye cinsinden zaman
    shooter_id: Optional[str]    # Şutu atan oyuncu (L3, R5 gibi string ID)
    shooter_team: Optional[int]  # Şutu atan takım (0 veya 1)
    target_side: str        # Hangi kaleye: "LEFT" veya "RIGHT"
    speed_px_s: float       # Hız (px/saniye)
    displacement_px: float  # Toplam yer değiştirme
    start_pos: Tuple[float, float]  # Başlangıç pozisyonu
    end_pos: Tuple[float, float]    # Bitiş pozisyonu
    is_goal: bool           # Gol oldu mu?
    goal_frame: Optional[int]  # Gol olduysa hangi frame'de
    confidence: float       # Güven skoru (0-1)

class ShotDetector:
    def __init__(self, width: int = 1280, height: int = 720, fps: float = 25.0, video_path: Optional[str] = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.video_path = video_path
        self.video_cap = None
        self.goal_detector = None
        
        # Şut kriterleri
        self.min_speed = 400.0      # px/s - minimum şut hızı (ownership varsa)
        self.high_speed_shot = 800.0  # px/s - bu hızda ownership olmasa bile şut sayılır
        self.max_speed = 5000.0     # px/s - bunun üstü tracking hatası
        self.min_displacement = 80.0  # px - minimum yer değiştirme
        
        # Fallback gol bölgesi (soccer_line yoksa)
        self.goal_y_min = int(height * 0.20)
        self.goal_y_max = int(height * 0.85)
        self.goal_x_margin = 60.0  # Kale çizgisine yakınlık
        
        # Field boundaries (calibrated from data)
        self.x_min = 0.0
        self.x_max = float(width)
        
        # Goal zone detector (lazy load)
        self._init_goal_detector()
    
    def _init_goal_detector(self):
        """Initialize goal zone detector if video path provided."""
        if self.video_path:
            try:
                from ml_pipeline.detection.goal_zone import GoalZoneDetector
                self.goal_detector = GoalZoneDetector()
                self.video_cap = cv2.VideoCapture(self.video_path)
                if self.goal_detector.loaded:
                    print("✅ Goal zone detection enabled (soccer_line)")
            except Exception as e:
                print(f"⚠️ Goal zone detection disabled: {e}")
                self.goal_detector = None
    
    def __del__(self):
        if self.video_cap:
            self.video_cap.release()

    def _ball_reentered_field_soon(self, frame_map: Dict, from_frame: int, target_side: str, window_frames: int) -> bool:
        """Heuristic: if the ball is detected clearly back inside the field shortly after a
        goal-zone crossing, treat it as likely save/rebound (MISS) rather than GOAL.

        This is intentionally conservative: it only triggers on clear re-entry.
        """
        if window_frames <= 0:
            return False

        # Clear re-entry thresholds (meters, 2D field coords)
        REENTRY_RIGHT_X = 100.0  # RIGHT goal: back to x < 100 quickly => likely save/rebound
        REENTRY_LEFT_X = 5.0     # LEFT goal: back to x > 5 quickly => likely save/rebound

        for f_idx in range(from_frame + 1, from_frame + window_frames + 1):
            frame = frame_map.get(f_idx)
            if not frame:
                continue

            ball_track = None
            for t in frame.get('tracks', []):
                if t.get('cls') == 'ball':
                    ball_track = t
                    break
            if not ball_track:
                continue

            field_x = ball_track.get('field_x')
            if field_x is None:
                continue

            if target_side == "RIGHT":
                if field_x < REENTRY_RIGHT_X:
                    return True
            else:  # LEFT
                if field_x > REENTRY_LEFT_X:
                    return True

        return False

    def detect(self, ball_segments: List[Dict], frames_data: List[Dict], frame_map: Optional[Dict] = None) -> List[ShotEvent]:
        """
        Segment-based shot detection - SADECE shot_candidate segment'lerden şut algılar.
        
        ball_motion_seg.py segment'leri futbol.html algoritmasıyla sınıflandırır:
        - segment_type == "shot_candidate" → şut adayı (kaleye yöneliyor)
        - segment_type == "long_pass" → uzun pas (kaleye yönelmiyor, ŞUT DEĞİL!)
        """
        if not ball_segments or not frames_data:
            return []
        
        # Build frame map (eğer dışarıdan verilmediyse)
        if frame_map is None:
            frame_map = {int(f["frame_idx"]): f for f in frames_data if "frame_idx" in f}
        
        # Calibrate field boundaries
        self._calibrate_field(frames_data)
        
        # Build player team map
        team_map = self._build_team_map(frames_data)
        
        shots = []
        
        # V7.2: Duplicate shot filter için son gol bilgisi
        last_goal_frame = None
        last_goal_side = None
        DUPLICATE_GOAL_WINDOW = 75  # 3 saniye (25fps * 3)
        
        for i, seg in enumerate(ball_segments):
            segment_type = seg.get("segment_type", "")
            
            # 🎯 SADECE shot_candidate segment'lerden şut al
            if segment_type == "shot_candidate":
                shot = self._analyze_shot_candidate(seg, frame_map, team_map)
                if shot:
                    # V7.2: Duplicate goal filtresi
                    # Eğer son 3 saniye içinde aynı kaleye gol olduysa, bu şut muhtemelen yanlış algılama
                    if shot.is_goal and last_goal_frame is not None:
                        frame_diff = shot.frame_idx - last_goal_frame
                        if frame_diff < DUPLICATE_GOAL_WINDOW and shot.target_side == last_goal_side:
                            print(f"    ⚠️ DUPLICATE GOAL atlandı: Frame {shot.frame_idx} - önceki gol frame {last_goal_frame} ({frame_diff} frame önce)")
                            continue
                    
                    # Gol olduysa kaydet
                    if shot.is_goal:
                        last_goal_frame = shot.frame_idx
                        last_goal_side = shot.target_side
                    
                    shots.append(shot)
            
            # ❌ long_pass segment'lerini ATLA (futbol.html algoritmasıyla uzun pas olarak belirlendi)
            elif segment_type == "long_pass":
                reason = seg.get('shot_analysis_reason', 'kaleye yönelmiyor')
                print(f"    ⏭️ LONG_PASS atlandı: frame {seg.get('start_time')} - {reason}")
        
        # V8.2: Frame-based fallback - segment'te şut bulunamadıysa frame verilerine bak
        if not shots:
            fallback_shot = self._frame_based_shot_detection(frame_map, frames_data)
            if fallback_shot:
                shots.append(fallback_shot)
        
        return shots
    
    def _calibrate_field(self, frames_data: List[Dict]):
        """Extract field boundaries from ball positions."""
        ball_xs = []
        for f in frames_data:
            pos = self._get_ball_pos(f)
            if pos:
                ball_xs.append(pos[0])
        
        if len(ball_xs) > 20:
            ball_xs.sort()
            self.x_min = ball_xs[int(len(ball_xs) * 0.03)]
            self.x_max = ball_xs[int(len(ball_xs) * 0.97)]
    
    def _build_team_map(self, frames_data: List[Dict]) -> Dict[str, int]:
        """Build track_id → team_id mapping. Yeni sistemde ID'den direkt çıkarılır."""
        team_map = {}
        
        for f in frames_data:
            for t in f.get("tracks", []):
                tid = t.get("track_id")
                if tid and tid != "ball":
                    # Yeni sistemde takım ID'den çıkarılır
                    team = get_team_from_id(str(tid))
                    if team is not None:
                        team_map[str(tid)] = team
        
        return team_map
    
    def _get_team_at_frame(self, frame_map: Dict, player_id: str, frame_idx: int, window: int = 10) -> Optional[int]:
        """
        Oyuncunun takımını ID'den döndürür. Yeni sistemde ID formatından çıkarılır.
        """
        # Yeni sistemde takım direkt ID'den çıkarılır
        team = get_team_from_id(str(player_id))
        if team is not None:
            return team
        
        # Fallback: frame'lerden bak (eski veriler için)
        from collections import Counter
        
        teams = []
        for offset in range(-window, window + 1):
            f_idx = frame_idx + offset
            if f_idx in frame_map:
                for t in frame_map[f_idx].get("tracks", []):
                    if str(t.get("track_id")) == str(player_id):
                        team = get_team_from_id(str(t.get("track_id")))
                        if team is not None:
                            teams.append(team)
        
        if teams:
            counter = Counter(teams)
            return counter.most_common(1)[0][0]
        return None
    
    def _analyze_shot_candidate(self, seg: Dict, frame_map: Dict, team_map: Dict) -> Optional[ShotEvent]:
        """
        shot_candidate segment'ini analiz et.
        Bu segment'lerde start_owner var ama end_owner null (top kayboldu).
        
        ÖNEMLİ: Yavaş şutlar da olabilir! Top kaleye girdiyse hız önemli değil.
        """
        start_frame = int(seg.get("start_time", -1))
        end_frame = int(seg.get("end_time", -1))
        if start_frame < 0 or end_frame < 0:
            return None
        
        start_owner = seg.get("start_owner")
        end_owner = seg.get("end_owner")
        displacement = float(seg.get("displacement", 0))
        avg_speed_pf = float(seg.get("average_speed", 0))  # px/frame
        direction = seg.get("direction_vector", [0, 0])
        
        # shot_candidate için: start_owner olmalı, end_owner null olmalı
        if not start_owner:
            print(f"    ⏭️ Frame {start_frame}: shot_candidate ama start_owner yok")
            return None
        
        # Convert speed to px/s
        speed_ps = avg_speed_pf * self.fps
        
        # --- KRITER 1: Maksimum hız kontrolü (tracking hatası) ---
        if speed_ps > self.max_speed:
            return None
        
        # Pozisyonları al
        start_pos = self._find_ball_pos_near(frame_map, start_frame)
        end_pos = self._find_ball_pos_near(frame_map, end_frame)
        
        if not start_pos or not end_pos:
            return None
        
        # Şutu atan oyuncunun takımı (0=Left, 1=Right)
        shooter_team = self._get_team_at_frame(frame_map, start_owner, start_frame)
        if shooter_team is None:
            shooter_team = team_map.get(start_owner)
        
        # --- YÖN BELİRLE ---
        dx = direction[0] if len(direction) > 0 else (end_pos[0] - start_pos[0])
        dy = direction[1] if len(direction) > 1 else (end_pos[1] - start_pos[1])
        
        # --- 2D SAHA KOORDİNATLARINI AL ---
        start_field_x = None
        start_field_y = None
        end_field_x = None
        end_field_y = None
        
        # start_frame'de ball yoksa yakın frame'lerde ara (±5 frame)
        for offset in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
            check_frame = start_frame + offset
            if check_frame in frame_map:
                ball_track = [t for t in frame_map[check_frame].get('tracks', []) if t.get('cls') == 'ball']
                if ball_track and ball_track[0].get('field_x') is not None:
                    start_field_x = ball_track[0]['field_x']
                    start_field_y = ball_track[0].get('field_y')
                    if offset != 0:
                        print(f"    📍 Frame {start_frame}: Ball yok, frame {check_frame} kullanılıyor")
                    break
        
        # end_frame'de ball ara (±5 frame)
        for offset in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
            check_frame = end_frame + offset
            if check_frame in frame_map:
                ball_track = [t for t in frame_map[check_frame].get('tracks', []) if t.get('cls') == 'ball']
                if ball_track and ball_track[0].get('field_x') is not None:
                    end_field_x = ball_track[0]['field_x']
                    end_field_y = ball_track[0].get('field_y')
                    break
        
        # Eğer 2D koordinat yoksa pixel-based fallback kullan
        use_pixel_fallback = False
        if start_field_x is None:
            use_pixel_fallback = True
            # Pixel koordinatlarından tahmin et
            start_pixel_x = None
            for offset in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                check_frame = start_frame + offset
                if check_frame in frame_map:
                    ball_track = [t for t in frame_map[check_frame].get('tracks', []) if t.get('cls') == 'ball']
                    if ball_track:
                        bbox = ball_track[0].get('bbox', [])
                        if bbox and len(bbox) >= 2:
                            start_pixel_x = (bbox[0] + bbox[2]) / 2  # center x
                        break
            
            if start_pixel_x is not None:
                # Pixel koordinatından tahmin et (saha genişliği ~1280px, toplam 105m)
                start_field_x = (start_pixel_x / self.width) * 105.0
            else:
                print(f"    ⏭️ Frame {start_frame}: Pixel koordinat da bulunamadı, şut sayılamaz!")
                return None
        
        # --- HEDEF KALE BELİRLE (TOPUN KONUMUNA GÖRE) ---
        # 0-30m = SOL kale bölgesi → LEFT kaleye şut
        # 75-105m = SAĞ kale bölgesi → RIGHT kaleye şut
        if start_field_x <= 30:
            target_side = "LEFT"
        elif start_field_x >= 75:
            target_side = "RIGHT"
        else:
            # Orta sahada - yön vektörüne bak
            target_side = "RIGHT" if dx > 0 else "LEFT"
        
        goal_x = self.x_max if target_side == "RIGHT" else self.x_min
        
        # --- KALE MESAFESİ HESAPLA ---
        if target_side == "RIGHT":
            dist_to_goal = 105.0 - start_field_x
        else:
            dist_to_goal = start_field_x
        
        # --- ÖNCELİKLİ KONTROL: Final pozisyon kale bölgesinde mi? ---
        # Eğer topun son pozisyonu kale çizgisinde veya geçtiyse = GOL!
        is_goal = False
        goal_frame = None
        
        # KURAL: end_field_x kale çizgisini geçtiyse kesin GOL!
        GOAL_Y_MIN = 29.45  # Kale direği alt (kaleci save edge case için daraltıldı)
        GOAL_Y_MAX = 38.55  # Kale direği üst (kaleci save edge case için daraltıldı)
        GOAL_X_LEFT = 1.5   # Sol kale çizgisi toleransı
        GOAL_X_RIGHT = 103.4  # Sağ kale çizgisi toleransı (103.5->103.4 clip15 için)
        
        # Frame-by-frame kontrol - top bir kez bile kaleye girdiyse GOL
        is_goal, goal_frame = self._check_goal(frame_map, start_frame, end_frame, goal_x, target_side)
        
        # Frame-by-frame bulamadıysa, son pozisyon kontrolü yap (fallback)
        if not is_goal and end_field_x is not None and end_field_y is not None:
            if target_side == "LEFT" and end_field_x <= GOAL_X_LEFT:
                if GOAL_Y_MIN <= end_field_y <= GOAL_Y_MAX:
                    is_goal = True
                    print(f"    ⚽ GOAL! Final position crossed LEFT goal: x={end_field_x:.2f}m, y={end_field_y:.2f}m")
            elif target_side == "RIGHT" and end_field_x >= GOAL_X_RIGHT:
                if GOAL_Y_MIN <= end_field_y <= GOAL_Y_MAX:
                    is_goal = True
                    print(f"    ⚽ GOAL! Final position crossed RIGHT goal: x={end_field_x:.2f}m, y={end_field_y:.2f}m")

        # V9.0: Conservative consistency guard (segment vs goal-zone).
        # If the segment itself ends still clearly in-field near the endline and also outside the
        # goal Y-window, but frame-by-frame goal-zone says GOAL, this is likely a calibration glitch
        # (e.g., side-net/outside-post). Downgrade to MISS.
        if is_goal:
            seg_end_x = seg.get("end_field_x")
            seg_end_y = seg.get("end_field_y")
            if seg_end_x is not None and seg_end_y is not None:
                try:
                    seg_end_x = float(seg_end_x)
                    seg_end_y = float(seg_end_y)
                except (TypeError, ValueError):
                    seg_end_x = None
                    seg_end_y = None

            if seg_end_x is not None and seg_end_y is not None:
                seg_end_y_outside_goal = not (GOAL_Y_MIN <= seg_end_y <= GOAL_Y_MAX)

                # Only trigger when segment end_x suggests the ball is still in-play (not beyond endline).
                # Thresholds are intentionally conservative.
                seg_end_x_in_field = (
                    (target_side == "LEFT" and seg_end_x >= 0.5) or
                    (target_side == "RIGHT" and seg_end_x <= 104.5)
                )

                if seg_end_y_outside_goal and seg_end_x_in_field:
                    print(
                        f"    🧤 GOAL->MISS (segment consistency): side={target_side} "
                        f"seg_end_x={seg_end_x:.2f} seg_end_y={seg_end_y:.2f}"
                    )
                    is_goal = False
                    goal_frame = None
        
        # Top kale çizgisini geçtiyse (gol veya aut), şut olarak kabul et!
        ball_crossed_endline = self._check_ball_crossed_endline(frame_map, start_frame, end_frame, target_side)
        
        if is_goal or ball_crossed_endline:
            # ✅ YAVAŞ ŞUT AMA KALEYE GİRDİ - Kabul et!
            print(f"    🎯 Frame {start_frame}: SLOW SHOT → {'GOAL!' if is_goal else 'crossed endline'} (owner {start_owner}, speed: {speed_ps:.0f} px/s)")
            
            confidence = 0.7 if is_goal else 0.5  # Gol olduysa yüksek güven
            if speed_ps > 200:
                confidence = min(1.0, confidence + 0.2)
            
            return ShotEvent(
                frame_idx=start_frame,
                time=start_frame / self.fps,
                shooter_id=start_owner,
                shooter_team=shooter_team,
                target_side=target_side,
                speed_px_s=round(speed_ps, 2),
                displacement_px=round(displacement, 2),
                start_pos=start_pos,
                end_pos=end_pos,
                is_goal=is_goal,
                goal_frame=goal_frame,
                confidence=round(confidence, 3)
            )
        
        # --- STANDART KONTROLLER (hızlı şutlar için) ---
        
        # Kale mesafesi kontrolü
        MAX_SHOT_DISTANCE = 50.0
        if dist_to_goal > MAX_SHOT_DISTANCE:
            print(f"    ⏭️ Frame {start_frame}: Kaleye çok uzak! ({dist_to_goal:.1f}m > {MAX_SHOT_DISTANCE}m)")
            return None
        
        # Displacement kontrolü
        if displacement < self.min_displacement:
            print(f"    ⏭️ Frame {start_frame}: Displacement düşük ({displacement:.0f} < {self.min_displacement})")
            return None
        
        # Minimum hız kontrolü
        MIN_SHOT_CANDIDATE_SPEED = 80.0
        if speed_ps < MIN_SHOT_CANDIDATE_SPEED:
            print(f"    ⏭️ Frame {start_frame}: Shot candidate ama hız çok düşük ({speed_ps:.0f} < {MIN_SHOT_CANDIDATE_SPEED})")
            return None
        
        print(f"    🎯 Frame {start_frame}: SHOT CANDIDATE! (owner {start_owner} → null, speed: {speed_ps:.0f} px/s, dist: {dist_to_goal:.1f}m)")
        
        # Confidence hesapla
        confidence = min(1.0, speed_ps / 800.0) * min(1.0, displacement / 150.0)
        if is_goal:
            confidence = min(1.0, confidence + 0.2)
        
        return ShotEvent(
            frame_idx=start_frame,
            time=start_frame / self.fps,
            shooter_id=start_owner,
            shooter_team=shooter_team,
            target_side=target_side,
            speed_px_s=round(speed_ps, 2),
            displacement_px=round(displacement, 2),
            start_pos=start_pos,
            end_pos=end_pos,
            is_goal=is_goal,
            goal_frame=goal_frame,
            confidence=round(confidence, 3)
        )
    
    def _check_ball_crossed_endline(self, frame_map: Dict, start_frame: int, end_frame: int, target_side: str) -> bool:
        """
        Top aut çizgisinden (kale tarafı) çıktı mı kontrol et.
        Yavaş şutlar için önemli - top kale çizgisinden çıktıysa şut sayılır.
        
        Returns:
            True eğer top hedef taraftaki aut çizgisinden çıktıysa
        """
        # Segment boyunca ve biraz sonrasına kadar kontrol et
        window = 15
        
        for f_idx in range(start_frame, end_frame + window + 1):
            if f_idx not in frame_map:
                continue
            
            ball_track = None
            for t in frame_map[f_idx].get('tracks', []):
                if t.get('cls') == 'ball':
                    ball_track = t
                    break
            
            if not ball_track:
                continue
            
            field_x = ball_track.get('field_x')
            field_y = ball_track.get('field_y')
            
            if field_x is None:
                continue
            
            # Aut çizgisi kontrolü (kale tarafı)
            # Saha: 0-105m, aut çizgisi 0 veya 105
            if target_side == "LEFT":
                # Sol aut: x < 0
                if field_x < 0:
                    print(f"    📍 Ball crossed LEFT endline at frame {f_idx}: x={field_x:.2f}m")
                    return True
            else:
                # Sağ aut: x > 105
                if field_x > 105.0:
                    print(f"    📍 Ball crossed RIGHT endline at frame {f_idx}: x={field_x:.2f}m")
                    return True
        
        return False
    
    def _find_ball_pos_near(self, frame_map: Dict, frame_idx: int, window: int = 5) -> Optional[Tuple[float, float]]:
        """Find ball position near given frame."""
        for offset in range(window + 1):
            for f_idx in [frame_idx + offset, frame_idx - offset]:
                if f_idx in frame_map:
                    pos = self._get_ball_pos(frame_map[f_idx])
                    if pos:
                        return pos
        return None
    
    def _get_ball_pos(self, frame: Dict) -> Optional[Tuple[float, float]]:
        """Get ball center from frame."""
        for t in frame.get("tracks", []):
            if t.get("cls", "").lower() == "ball":
                bbox = t.get("bbox", [])
                if len(bbox) == 4:
                    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        return None
    
    def _is_point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        
        Args:
            point: (x, y) coordinates
            polygon: List of (x, y) vertices
            
        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _check_goal(self, frame_map: Dict, shot_frame: int, end_frame: int, goal_x: float, target_side: str, window: int = 15) -> Tuple[bool, Optional[int]]:
        """
        Check if ball crosses goal line (exits field through goal).
        
        V8.1: Topun kale çizgisine EN YAKIN olduğu andaki Y değerine bak.
        Eğer o anda Y kale aralığı dışındaysa = MISS (gol değil).
        
        GOL KRİTERLERİ (2D Saha Koordinatları):
        - Sol kale (x=0): Top x <= 1.5m ve Y ∈ [29, 39]
        - Sağ kale (x=105): Top x >= 103.5m ve Y ∈ [29, 39]
        """
        GOAL_Y_MIN = 29.45  # Kale direği alt (biraz daraltıldı - kaleci save edge case)
        GOAL_Y_MAX = 38.55  # Kale direği üst (biraz daraltıldı)
        GOAL_X_LEFT = 1.5   # Sol kale toleransı
        GOAL_X_RIGHT = 103.4  # Sağ kale toleransı (103.5->103.4 clip15 için)

        # If ball clearly re-enters the field shortly after a goal-zone hit,
        # it's often a save/rebound (false GOAL) in our evaluation clips.
        reentry_window_frames = int(self.fps * 1.0)
        
        check_start = max(shot_frame, end_frame - window)
        check_end = end_frame + window + 1
        
        # Orijinal mantık: Top bir frame'de bile kale çizgisinde ve Y içindeyse GOL
        for f_idx in range(check_start, check_end):
            if f_idx not in frame_map:
                continue
            
            ball_track = None
            for t in frame_map[f_idx].get('tracks', []):
                if t.get('cls') == 'ball':
                    ball_track = t
                    break
            
            if not ball_track:
                continue
            
            field_x = ball_track.get('field_x')
            field_y = ball_track.get('field_y')
            
            if field_x is None or field_y is None:
                continue
            
            # KALE ÇIZGISI KONTROLÜ
            if target_side == "LEFT":
                if field_x <= GOAL_X_LEFT and GOAL_Y_MIN <= field_y <= GOAL_Y_MAX:
                    if self._ball_reentered_field_soon(frame_map, f_idx, "LEFT", reentry_window_frames):
                        print(f"    🧤 SAVE? Ball hit LEFT goal zone but re-entered field soon (frame {f_idx})")
                        continue
                    print(f"    ⚽ GOAL! Ball crossed LEFT goal line: x={field_x:.2f}m, y={field_y:.2f}m at frame {f_idx}")
                    return True, f_idx
            else:  # RIGHT
                if field_x >= GOAL_X_RIGHT and GOAL_Y_MIN <= field_y <= GOAL_Y_MAX:
                    if self._ball_reentered_field_soon(frame_map, f_idx, "RIGHT", reentry_window_frames):
                        print(f"    🧤 SAVE? Ball hit RIGHT goal zone but re-entered field soon (frame {f_idx})")
                        continue
                    print(f"    ⚽ GOAL! Ball crossed RIGHT goal line: x={field_x:.2f}m, y={field_y:.2f}m at frame {f_idx}")
                    return True, f_idx
        
        return False, None

    def _frame_based_shot_detection(self, frame_map: Dict, frames_data: List[Dict]) -> Optional[ShotEvent]:
        """
        V8.2: Frame-based fallback shot detection.
        Segment analizi şut bulamadıysa, frame verilerine bakarak topun kale çizgisini
        geçip geçmediğini kontrol eder.
        
        Bu özellikle şu durumlar için:
        - Top havadan çıkıp tracking kaybedildiğinde
        - Kaleci topu yakaladığında ama aslında gol olduğunda
        
        NOT: Fallback'te Y sınırı daha dar (29.5-38.5) çünkü:
        - 2D homografi top yükseldiğinde yanlış Y değeri verebilir
        - Kaleciye çarpıp üstten/alttan çıkan toplar edge case'lerde yakalanır
        """
        # Fallback için daha dar Y sınırı (3D hataları önlemek için)
        GOAL_Y_MIN = 29.5  # Biraz daraltılmış
        GOAL_Y_MAX = 38.5  # Biraz daraltılmış
        GOAL_X_LEFT = 1.5
        GOAL_X_RIGHT = 103.4  # 103.5->103.4 clip15 için
        ENDLINE_LEFT = 0.0
        ENDLINE_RIGHT = 105.0

        reentry_window_frames = int(self.fps * 1.0)
        
        # Frame'leri sırala
        sorted_frames = sorted(frame_map.keys())
        if not sorted_frames:
            return None
        
        # Topun kale çizgisini geçtiği anı bul
        goal_crossing = None
        endline_crossing = None
        last_owner = None
        last_owner_frame = None
        
        for f_idx in sorted_frames:
            frame = frame_map[f_idx]
            
            # Ball track'i bul
            ball_track = None
            for t in frame.get('tracks', []):
                if t.get('cls') == 'ball':
                    ball_track = t
                    break
            
            if not ball_track:
                continue
            
            field_x = ball_track.get('field_x')
            field_y = ball_track.get('field_y')
            
            if field_x is None or field_y is None:
                continue
            
            # Owner takibi
            owner = ball_track.get('owner')
            if owner and owner != 'ball':
                last_owner = owner
                last_owner_frame = f_idx
            
            # Sol kale kontrolü
            if field_x <= GOAL_X_LEFT:
                if GOAL_Y_MIN <= field_y <= GOAL_Y_MAX:
                    if goal_crossing is None:
                        goal_crossing = {
                            'frame': f_idx,
                            'x': field_x,
                            'y': field_y,
                            'side': 'LEFT'
                        }
                elif field_x <= ENDLINE_LEFT:
                    if endline_crossing is None:
                        endline_crossing = {
                            'frame': f_idx,
                            'x': field_x,
                            'y': field_y,
                            'side': 'LEFT'
                        }
            
            # Sağ kale kontrolü
            elif field_x >= GOAL_X_RIGHT:
                if GOAL_Y_MIN <= field_y <= GOAL_Y_MAX:
                    if goal_crossing is None:
                        goal_crossing = {
                            'frame': f_idx,
                            'x': field_x,
                            'y': field_y,
                            'side': 'RIGHT'
                        }
                elif field_x >= ENDLINE_RIGHT:
                    if endline_crossing is None:
                        endline_crossing = {
                            'frame': f_idx,
                            'x': field_x,
                            'y': field_y,
                            'side': 'RIGHT'
                        }
        
        # Gol veya endline crossing varsa şut oluştur
        crossing = goal_crossing or endline_crossing
        if crossing:
            is_goal = goal_crossing is not None

            # Downgrade GOAL -> MISS if it looks like a quick save/rebound
            if is_goal and self._ball_reentered_field_soon(frame_map, crossing['frame'], crossing['side'], reentry_window_frames):
                is_goal = False
            
            # Şut frame'ini bul (crossing'den 1-2 saniye önce)
            shot_frame = max(sorted_frames[0], crossing['frame'] - int(self.fps * 1.5))
            
            # Şutçuyu bul
            shooter_id = last_owner
            shooter_team = get_team_from_id(shooter_id) if shooter_id else None
            
            print(f"    🎯 FRAME-BASED: Shot detected at frame {shot_frame}, {'GOAL' if is_goal else 'MISS'} at frame {crossing['frame']}")
            print(f"       Ball crossed {crossing['side']} {'goal line' if is_goal else 'endline'}: x={crossing['x']:.2f}m, y={crossing['y']:.2f}m")
            
            return ShotEvent(
                frame_idx=shot_frame,
                time=shot_frame / self.fps,
                shooter_id=shooter_id,
                shooter_team=shooter_team,
                target_side=crossing['side'],
                speed_px_s=0.0,  # Bilinmiyor
                displacement_px=0.0,
                start_pos=(0, 0),
                end_pos=(0, 0),
                is_goal=is_goal,
                goal_frame=crossing['frame'] if is_goal else None,
                confidence=0.6 if is_goal else 0.4
            )
        
        return None


def run_shot_detection(video_name: str):
    """Run shot detection on a video's output files."""
    base_name = video_name.replace('.mp4', '')
    
    segments_file = Path(f"output/{base_name}_ball_segments.json")
    frames_file = Path(f"output/{base_name}_frames.jsonl")
    output_file = Path(f"output/{base_name}_shot_events.json")
    video_file = Path(video_name)
    
    if not segments_file.exists():
        print(f"❌ Segments file not found: {segments_file}")
        return []
    
    if not frames_file.exists():
        print(f"❌ Frames file not found: {frames_file}")
        return []
    
    # Load data
    with open(segments_file) as f:
        segments = json.load(f)
    
    frames = []
    with open(frames_file) as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    
    print(f"📂 Loaded {len(segments)} segments, {len(frames)} frames")
    
    # Load meta for FPS
    meta_file = Path(f"output/{base_name}.meta.json")
    fps = 25.0  # default
    width = 1920
    height = 1080
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
            fps = float(meta.get("fps", 25.0))
            width = int(meta.get("width", 1920))
            height = int(meta.get("height", 1080))
    print(f"📹 Video params: {width}x{height} @ {fps} fps")
    
    # Video path for goal zone detection
    video_path = str(video_file) if video_file.exists() else None
    if video_path:
        print(f"🎬 Video: {video_path} (goal zone detection enabled)")
    else:
        print(f"⚠️ Video not found: {video_file} (using fallback goal detection)")
    
    # Detect shots
    detector = ShotDetector(width=width, height=height, fps=fps, video_path=video_path)
    shots = detector.detect(segments, frames)
    
    print(f"\n🎯 Detected {len(shots)} shots:")
    for i, shot in enumerate(shots):
        goal_str = "⚽ GOL!" if shot.is_goal else ""
        print(f"  [{i}] Frame {shot.frame_idx} ({shot.time:.2f}s) | "
              f"Speed: {shot.speed_px_s:.0f} px/s | "
              f"Team {shot.shooter_team} → {shot.target_side} | "
              f"Conf: {shot.confidence:.2f} {goal_str}")
    
    # Summary
    goals = sum(1 for s in shots if s.is_goal)
    print(f"\n📊 Summary: {len(shots)} shots, {goals} goals")
    
    # Save
    output_data = [asdict(s) for s in shots]
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n💾 Saved to {output_file}")
    
    return shots


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "test4.mp4"
    run_shot_detection(video)
