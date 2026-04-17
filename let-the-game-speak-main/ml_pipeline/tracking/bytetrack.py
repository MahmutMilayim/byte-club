from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics.trackers.byte_tracker import BYTETracker

from schemas.types import Track
from ml_pipeline.tracking.id_stabilizer import TeamBasedIDStabilizer


@dataclass
class RawTrack:
    """ByteTrack'ten gelen ham track (henüz takım bazlı ID atanmamış)."""
    track_id: int  # ByteTrack'ten gelen int ID
    cls: str
    bbox: tuple
    t: float
    score: Optional[float] = None
    _team: Optional[int] = None  # Takım sınıflandırmasından gelen bilgi


def _fixed_init_track(self, results, feats_keep):
    """
    Fixed init_track that handles CUDA tensors properly.
    Monkey-patched version of BYTETracker.init_track().
    """
    bboxes, scores, class_ids = results.xyxy, results.conf, results.cls
    
    # CRITICAL: Move tensors to CPU before numpy operations
    if torch.is_tensor(bboxes):
        bboxes = bboxes.cpu()
    if torch.is_tensor(scores):
        scores = scores.cpu()
    if torch.is_tensor(class_ids):
        class_ids = class_ids.cpu()
    
    # Convert to numpy arrays
    bboxes = np.asarray(bboxes)
    scores = np.asarray(scores)
    class_ids = np.asarray(class_ids)
    
    # Add track id column (required by original implementation)
    bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
    
    # Import STrack here to avoid circular import
    from ultralytics.trackers.bot_sort import STrack
    
    # Initialize tracks with proper format: [x1, y1, x2, y2, idx] -> [x, y, w, h, score, idx]
    tracks = []
    for i, (bbox, score, class_id) in enumerate(zip(bboxes, scores, class_ids)):
        # Convert xyxy to xywh
        x1, y1, x2, y2, idx = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        
        # STrack expects [x, y, w, h, score] or [x, y, w, h, score, idx]
        xywh = np.array([x, y, w, h, float(score), int(idx)])
        track = STrack(xywh, float(score), int(class_id))
        tracks.append(track)
    
    return tracks


# Monkey-patch the BYTETracker class BEFORE creating any instances
BYTETracker.init_track = _fixed_init_track


class FixedBYTETracker(BYTETracker):
    """
    ByteTracker with CUDA tensor fix.
    Ensures tensors are on CPU before numpy conversion.
    """
    def update(self, results, img=None):
        """Override to handle CUDA tensors properly."""
        # Move all box tensors to CPU if they're on CUDA
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            
            # Move each tensor attribute to CPU
            if hasattr(boxes, 'data') and torch.is_tensor(boxes.data):
                boxes.data = boxes.data.cpu()
            if hasattr(boxes, 'xyxy') and torch.is_tensor(boxes.xyxy):
                boxes.xyxy = boxes.xyxy.cpu()
            if hasattr(boxes, 'xywh') and torch.is_tensor(boxes.xywh):
                boxes.xywh = boxes.xywh.cpu()
            if hasattr(boxes, 'xyxyn') and torch.is_tensor(boxes.xyxyn):
                boxes.xyxyn = boxes.xyxyn.cpu()
            if hasattr(boxes, 'xywhn') and torch.is_tensor(boxes.xywhn):
                boxes.xywhn = boxes.xywhn.cpu()
            if hasattr(boxes, 'conf') and torch.is_tensor(boxes.conf):
                boxes.conf = boxes.conf.cpu()
            if hasattr(boxes, 'cls') and torch.is_tensor(boxes.cls):
                boxes.cls = boxes.cls.cpu()
        
        return super().update(results, img)


def make_default_bytetrack_args() -> SimpleNamespace:
    """
    ByteTrack için optimize edilmiş parametreler.
    
    ID Stabilitesi için Öneriler:
    - track_high_thresh: Yüksek tutulmalı (0.5+) - yanlış pozitifler ID karışıklığı yaratır
    - track_low_thresh: Düşük tutulmalı (0.1) - kısmi gizlenmelerde takibi sürdür
    - new_track_thresh: Yüksek tutulmalı (0.6+) - yeni ID kolayca verilmesin
    - track_buffer: Yüksek tutulmalı (60+) - 2+ saniye kaybolmalara tolerans
    - match_thresh: Yüksek tutulmalı (0.8+) - yanlış eşleşmeler ID switch yaratır
    """
    return SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.5,  # Oyuncular için yüksek güven
        track_low_thresh=0.1,   # Kısmi gizlenmelerde takibi sürdür
        new_track_thresh=0.6,   # ID switch'i önlemek için katı kural
        track_buffer=60,        # Oyuncular için 2 saniyelik hafıza (25fps)
        match_thresh=0.8,       # Yüksek eşleşme eşiği - yanlış eşleşmeleri önle
        fuse_score=True,
        gmc_method="none",      # Camera motion compensation (şimdilik kapalı)
        proximity_thresh=0.5,
        appearance_thresh=0.25,
    )


class ByteTrackTracker:
    """
    ÖZELLEŞTİRİLMİŞ TRACKER:
    - Oyuncular/Hakemler -> ByteTrack algoritmasına girer (Kalman Filter vs. uygulanır).
    - TOP -> ByteTrack'a SOKULMAZ. YOLO ne bulduysa direkt eklenir (Raw Detection).
    - ID STABILIZATION -> ByteTrack'ten gelen ID'ler stabilize edilerek ID switch'leri önlenir.
    """

    def __init__(
        self,
        fps: float,
        names: Dict[int, str],
        track_ball: bool = True,
        args: Optional[SimpleNamespace] = None,
        enable_id_stabilization: bool = True,
        stabilizer_max_disappeared: int = 30,
        stabilizer_iou_threshold: float = 0.3,
    ) -> None:
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.track_ball = track_ball
        self._names = {int(k): str(v).lower().strip() for k, v in names.items()}
        self._args = args or make_default_bytetrack_args()
        self._tracker = FixedBYTETracker(self._args, frame_rate=int(round(self.fps)))
        
        # ID Stabilization katmanı (Takım bazlı: L1-L11, R1-R11)
        self.enable_id_stabilization = enable_id_stabilization
        self._id_stabilizer = None
        if enable_id_stabilization:
            self._id_stabilizer = TeamBasedIDStabilizer(
                max_disappeared=stabilizer_max_disappeared,
                iou_threshold=stabilizer_iou_threshold,
                distance_threshold=200.0,
                frame_width=3840  # 4K varsayılan, vision.py'den güncellenebilir
            )
            print(f"[ByteTrackTracker] Team-Based ID Stabilization ENABLED (L1-L11, R1-R11)")
        
        # Top sınıfının ID'sini bulalım (genelde 'ball' veya 'sports ball')
        self.ball_cls_id = None
        # Hakem ve staff sınıflarını bul (tracking'e dahil edilmeyecek)
        self.referee_cls_ids = set()
        
        for k, v in self._names.items():
            v_lower = v.lower()
            if "ball" in v_lower:
                self.ball_cls_id = int(k)
            # Hakem ve staff sınıflarını belirle (tracking'e dahil edilmeyecek)
            if any(ref in v_lower for ref in ["ref", "referee", "main ref", "side ref", "linesman", "staff", "coach"]):
                self.referee_cls_ids.add(int(k))
        
        if self.referee_cls_ids:
            print(f"[ByteTrackTracker] Non-player classes excluded from tracking: {self.referee_cls_ids}")

    def _cls_id_from_track(self, st) -> Optional[int]:
        c = getattr(st, "cls", None)
        if c is None: return None
        if hasattr(c, 'numel'):
            c = c.item() if c.numel() == 1 else c[0].item()
        return int(c)

    def _role_from_cls_name(self, cls_name: str) -> str:
        n = cls_name.lower()
        if "ball" in n: return "ball"
        if "ref" in n or "linesman" in n: return "referee"  # Hakem
        if "staff" in n or "coach" in n: return "referee"  # Staff (referee gibi filtrele)
        if "player" in n or "goalkeeper" in n or "gk" in n: return "player"
        return "player"
    
    def _is_referee(self, cls_id: int) -> bool:
        """Sınıf ID'sinin hakem olup olmadığını kontrol et."""
        return cls_id in self.referee_cls_ids

    def step(self, result, frame_idx: int, team_classifier=None) -> List[Track]:
        """
        1. YOLO sonuçlarını al.
        2. Topu, Oyuncuları ve Hakemleri ayır.
        3. Sadece Oyuncuları Tracker'a ver (hakemler tracking'e dahil edilmez).
        4. Topu (en yüksek skorlu olanı) direkt listeye ekle.
        5. Takım sınıflandırması uygula.
        6. ID Stabilization uygula (L1-L11, R1-R11 formatı).
        
        NOT: Hakemler YOLO tarafından tespit edilir ancak ID atanmaz ve tracking'e dahil edilmez.
        
        Args:
            result: YOLO detection sonucu
            frame_idx: Frame numarası
            team_classifier: Opsiyonel takım sınıflandırıcı fonksiyonu
            
        Returns:
            Track listesi (string ID'li: L1, R5, ball gibi)
        """
        t = frame_idx / self.fps
        raw_tracks: List[RawTrack] = []
        
        if result is None or result.boxes is None:
            return []

        img = getattr(result, "orig_img", None)
        
        # --- 1. AYRIŞTIRMA (Top vs Diğerleri) ---
        all_boxes = result.boxes
        
        if self.ball_cls_id is not None:
            is_ball = (all_boxes.cls == self.ball_cls_id).cpu().numpy()
        else:
            is_ball = np.zeros(len(all_boxes), dtype=bool)

        # --- 2. TOP İŞLEME (MANUEL / RAW) ---
        if self.track_ball and np.any(is_ball):
            ball_indices = np.where(is_ball)[0]
            confs = all_boxes.conf[ball_indices].cpu().numpy()
            best_idx = ball_indices[np.argmax(confs)]
            
            best_box = all_boxes[best_idx]
            x1, y1, x2, y2 = map(float, best_box.xyxy[0].cpu().numpy())
            score = float(best_box.conf.cpu().numpy()[0])
            
            raw_tracks.append(
                RawTrack(
                    track_id=0,
                    cls="ball",
                    bbox=(x1, y1, x2, y2),
                    t=t,
                    score=score
                )
            )

        # --- 3. OYUNCU İŞLEME (BYTETRACK) - Hakemler hariç ---
        not_ball_indices = np.where(~is_ball)[0]
        
        # Hakemleri filtrele - tracking'e dahil etme
        cls_array = all_boxes.cls.cpu().numpy()
        player_mask = np.ones(len(not_ball_indices), dtype=bool)
        for i, idx in enumerate(not_ball_indices):
            cls_id = int(cls_array[idx])
            if self._is_referee(cls_id):
                player_mask[i] = False
        
        player_indices = not_ball_indices[player_mask]
        
        if len(player_indices) > 0:
            player_boxes = all_boxes[player_indices]
            _ = self._tracker.update(player_boxes, img=img)

            for st in getattr(self._tracker, "tracked_stracks", []):
                if not getattr(st, "is_activated", True): 
                    continue

                tlwh = getattr(st, "tlwh", None)
                if tlwh is None: 
                    continue
                
                x, y, w, h = map(float, tlwh[:4])
                x1, y1, x2, y2 = x, y, x + w, y + h

                cid = self._cls_id_from_track(st)
                cls_name = self._names.get(cid, "unknown") if cid is not None else "unknown"
                role = self._role_from_cls_name(cls_name)
                
                # Eğer hakem olarak sınıflandırıldıysa atla (ekstra güvenlik)
                if role == "referee":
                    continue

                score = float(getattr(st, "score", 0.0))
                tid = int(getattr(st, "track_id", -1))

                raw_tracks.append(
                    RawTrack(
                        track_id=tid,
                        cls=cls_name,  # Orijinal class ismi (Player-L, Player-R, GK-L, GK-R)
                        bbox=(x1, y1, x2, y2),
                        t=t,
                        score=score,
                    )
                )

        # --- 4. TAKIM SINIFLANDIRMASI (Opsiyonel) ---
        # team_classifier fonksiyonu dışarıdan gelecek (vision.py'den)
        raw_team_info: Dict[int, int] = {}
        if team_classifier is not None and img is not None:
            for rt in raw_tracks:
                if rt.cls != "ball":  # Top hariç tüm oyuncular için
                    team = team_classifier(img, rt.bbox)
                    if team is not None:
                        raw_team_info[rt.track_id] = team
                        rt._team = team

        # --- 5. ID STABILIZATION (Takım Bazlı) ---
        if self.enable_id_stabilization and self._id_stabilizer is not None:
            # Frame genişliğini güncelle
            if img is not None:
                self._id_stabilizer.frame_width = img.shape[1]
            
            tracks = self._id_stabilizer.update(raw_tracks, frame_idx, raw_team_info)
            
            if frame_idx % 100 == 0:
                stats = self._id_stabilizer.get_stats()
                print(f"[TeamIDStabilizer] Frame {frame_idx}: L={stats['left_team_count']}, R={stats['right_team_count']}, Total={stats['active_players']}")
        else:
            # Stabilization kapalıysa, RawTrack'leri Track'e dönüştür (geçici int ID ile)
            tracks = []
            for rt in raw_tracks:
                tracks.append(Track(
                    track_id=str(rt.track_id) if rt.cls != "ball" else "ball",
                    cls=rt.cls,
                    bbox=rt.bbox,
                    t=rt.t,
                    score=rt.score
                ))

        if frame_idx <= 5:
            print(f"[Tracker] frame {frame_idx}: {len(tracks)} objects (Ball + Players)")

        return tracks
    
    def set_field_transform(self, pixel_to_field_func):
        """Soccer_line'dan gelen pixel->field dönüşüm fonksiyonunu ID stabilizer'a geç."""
        if self._id_stabilizer is not None:
            self._id_stabilizer.set_field_transform(pixel_to_field_func)
            print("[ByteTrackTracker] Field transform set for goalkeeper positioning")