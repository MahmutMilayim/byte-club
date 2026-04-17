"""
Team-Based ID Stabilization System
==================================

Bu modül, ByteTrack'ten gelen track ID'lerini takım bazlı stabilize eder.

ID Formatı:
- Sol Takım: L1, L2, L3, ..., L11 (L1 = kaleci, rezerve)
- Sağ Takım: R1, R2, R3, ..., R11 (R1 = kaleci, rezerve)
- Top: "ball"

Kullanılan Stratejiler:
1. **Team-Based Assignment**: Her takımda max 11 oyuncu
2. **Goalkeeper Reserve**: L1/R1 sadece kalecilere atanır
3. **Temporal Smoothing**: Kısa süreli kaybolmalarda ID'yi hatırla
4. **Position Continuity**: Oyuncunun son pozisyonuna yakın olanı eşleştir

Kullanım:
    stabilizer = TeamBasedIDStabilizer()
    stable_tracks = stabilizer.update(tracks, frame_idx)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from schemas.types import Track, make_team_id, parse_team_id


@dataclass
class TrackedPlayer:
    """Takip edilen bir oyuncunun bilgileri."""
    team_id: str  # "L1", "R5" gibi
    team: int  # 0=Left, 1=Right
    number: int  # 1-11
    is_goalkeeper: bool
    last_bbox: Tuple[float, float, float, float]
    last_frame: int
    last_bytetrack_id: int
    disappeared_count: int = 0
    total_frames: int = 0
    confidence_history: List[float] = field(default_factory=list)
    # Renk kaybı toleransı için yeni alanlar
    color_missing_count: int = 0  # Renk kaç frame kayıp
    last_known_team: Optional[int] = None  # Son bilinen takım rengi


def bbox_iou(bbox1: Tuple[float, float, float, float], 
             bbox2: Tuple[float, float, float, float]) -> float:
    """İki bounding box arasındaki IoU (Intersection over Union) hesapla."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Bounding box'ın merkez koordinatlarını hesapla."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_distance(bbox1: Tuple[float, float, float, float], 
                  bbox2: Tuple[float, float, float, float]) -> float:
    """İki bounding box merkezi arası Euclidean mesafe."""
    cx1, cy1 = bbox_center(bbox1)
    cx2, cy2 = bbox_center(bbox2)
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def bbox_height(bbox: Tuple[float, float, float, float]) -> float:
    """Bounding box yüksekliği."""
    return bbox[3] - bbox[1]


class TeamBasedIDStabilizer:
    """
    Takım bazlı ID atama ve stabilizasyon sistemi.
    
    - Sol takım: L1-L11 (L1 kaleci)
    - Sağ takım: R1-R11 (R1 kaleci)
    - Her takımda max 11 oyuncu
    
    Parameters:
        max_disappeared: Kaç frame kaybolursa ID silinsin
        iou_threshold: IoU eşleştirme eşiği
        distance_threshold: Piksel cinsinden mesafe eşiği
        goalkeeper_position_threshold: Kaleci tespiti için x pozisyon eşiği (% olarak)
    """
    
    MAX_PLAYERS_PER_TEAM = 11
    GOALKEEPER_NUMBER = 1  # L1 ve R1 kalecilere rezerve
    
    def __init__(
        self,
        max_disappeared: int = 30,
        iou_threshold: float = 0.3,
        distance_threshold: float = 200.0,
        goalkeeper_position_threshold: float = 0.15,  # Ekranın %15'i
        frame_width: int = 3840
    ):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.goalkeeper_position_threshold = goalkeeper_position_threshold
        self.frame_width = frame_width
        
        # Soccer Line pixel_to_field fonksiyonu (kaleci saha pozisyonu için)
        self.pixel_to_field = None
        
        # Takım bazlı oyuncu takibi
        # team_id ("L1", "R5") -> TrackedPlayer
        self.players: Dict[str, TrackedPlayer] = {}
        
        # ByteTrack ID -> Team ID mapping
        self.bytetrack_to_team_id: Dict[int, str] = {}
        
        # Her takım için kullanılan numaralar (kaleci hariç: 2-11)
        self.used_numbers: Dict[int, Set[int]] = {0: set(), 1: set()}
        
        # Kaleci atandı mı?
        self.goalkeeper_assigned: Dict[int, bool] = {0: False, 1: False}
        
        # İstatistikler
        self.stats = {
            'total_reidentifications': 0,
            'total_new_tracks': 0,
            'left_team_count': 0,
            'right_team_count': 0
        }
    
    def update(self, tracks: List[Track], frame_idx: int, raw_team_info: Optional[Dict[int, int]] = None) -> List[Track]:
        """
        Yeni frame'deki track'leri al, takım bazlı ID ata ve döndür.
        
        Args:
            tracks: ByteTrack'ten gelen Track listesi (team bilgisi ile)
            frame_idx: Mevcut frame numarası
            raw_team_info: ByteTrack ID -> team (0/1) mapping (opsiyonel)
            
        Returns:
            Takım bazlı ID'lerle Track listesi
        """
        # Top ve oyuncuları ayır
        ball_tracks = [t for t in tracks if t.cls == "ball"]
        player_tracks = [t for t in tracks if t.cls != "ball"]
        
        # Topu "ball" ID'si ile döndür
        stabilized_ball = []
        for bt in ball_tracks:
            stabilized_ball.append(Track(
                track_id="ball",
                cls="ball",
                bbox=bt.bbox,
                t=bt.t,
                score=bt.score
            ))
        
        if not player_tracks:
            self._mark_disappeared(frame_idx)
            return stabilized_ball
        
        # Önce mevcut oyuncuları eşleştir
        matched, unmatched = self._match_tracks(player_tracks, frame_idx, raw_team_info)
        
        # Yeni oyuncular için ID ata
        for track in unmatched:
            self._register_new_player(track, frame_idx, raw_team_info)
        
        # Görünmeyen oyuncuları işaretle
        all_tracks = matched + unmatched
        seen_team_ids = {self.bytetrack_to_team_id.get(t.track_id) 
                         for t in all_tracks if t.track_id in self.bytetrack_to_team_id}
        self._mark_disappeared(frame_idx, seen_ids=seen_team_ids)
        
        # Takım bazlı ID'lerle Track oluştur
        stabilized_players = []
        for track in all_tracks:
            team_id = self.bytetrack_to_team_id.get(track.track_id)
            if team_id:
                stabilized_players.append(Track(
                    track_id=team_id,
                    cls=track.cls,
                    bbox=track.bbox,
                    t=track.t,
                    score=track.score
                ))
        
        # İstatistikleri güncelle
        self.stats['left_team_count'] = len([p for p in self.players.values() if p.team == 0])
        self.stats['right_team_count'] = len([p for p in self.players.values() if p.team == 1])
        
        return stabilized_ball + stabilized_players
    
    def _get_team_from_track(self, track, raw_team_info: Optional[Dict[int, int]]) -> Optional[int]:
        """
        Track'ten takım bilgisini al.
        
        Öncelik sırası:
        1. raw_team_info (renk sınıflandırmasından gelen)
        2. track._team attribute
        3. track.cls'den çıkar (Player-L, Player-R, GK-L, GK-R)
        """
        # 1. Önce raw_team_info'dan bak (en güvenilir - o frame'deki renk analizi)
        if raw_team_info and track.track_id in raw_team_info:
            return raw_team_info[track.track_id]
        
        # 2. Track'in kendi team bilgisine bak
        if hasattr(track, '_team') and track._team is not None:
            return track._team
        
        # 3. YOLO class isminden takım çıkar (Player-L, Player-R, GK-L, GK-R)
        # Bu, renk tespiti başarısız olduğunda fallback olarak kullanılır
        if hasattr(track, 'cls') and track.cls:
            cls_str = str(track.cls).lower()
            if '-l' in cls_str or cls_str.endswith(' l'):
                return 0  # Left team
            elif '-r' in cls_str or cls_str.endswith(' r'):
                return 1  # Right team
        
        return None
    
    def set_field_transform(self, pixel_to_field_func):
        """
        Soccer_line'dan gelen pixel->field dönüşüm fonksiyonunu ayarla.
        Bu fonksiyon (px, py) -> (field_x, field_y) dönüşümü yapar.
        field_x: 0-105m (sol kale=0, sağ kale=105)
        """
        self.pixel_to_field = pixel_to_field_func
    
    def _get_goalkeeper_team_by_field_position(self, bbox: Tuple[float, float, float, float]) -> Optional[int]:
        """
        Kalecinin saha koordinatına göre takımını belirle.
        Sol yarı saha (0-52.5m) -> L takımı kalecisi (team=0)
        Sağ yarı saha (52.5-105m) -> R takımı kalecisi (team=1)
        """
        if not hasattr(self, 'pixel_to_field') or self.pixel_to_field is None:
            # Saha dönüşümü yoksa piksel bazlı fallback
            cx = (bbox[0] + bbox[2]) / 2
            threshold_px = self.frame_width * 0.15  # %15
            if cx < threshold_px:
                return 0  # Sol tarafta -> L takımı kalecisi
            elif cx > (self.frame_width - threshold_px):
                return 1  # Sağ tarafta -> R takımı kalecisi
            return None
        
        # Ayak pozisyonu (bbox alt merkez)
        foot_x = (bbox[0] + bbox[2]) / 2
        foot_y = bbox[3]  # Alt kenar
        
        try:
            field_pos = self.pixel_to_field(foot_x, foot_y)
            if field_pos is None:
                return None
            
            field_x, field_y = field_pos
            
            # Sol yarı saha: 0-20m (kale önü)
            if field_x < 20:
                return 0  # L takımı kalecisi
            # Sağ yarı saha: 85-105m (kale önü)  
            elif field_x > 85:
                return 1  # R takımı kalecisi
            
        except Exception:
            pass
        
        return None
    
    def _is_goalkeeper_candidate(self, bbox: Tuple[float, float, float, float], team: int, cls: str = "") -> bool:
        """
        Kaleci adayı mı kontrol et.
        YOLO goalkeeper class'ı varsa True döner.
        NOT: Bu fonksiyon artık sadece pozisyon kontrolü için kullanılıyor.
        Kaleci ataması _register_new_player'da pozisyon bazlı yapılıyor.
        """
        cx = (bbox[0] + bbox[2]) / 2
        threshold_px = self.frame_width * self.goalkeeper_position_threshold
        
        if team == 0:  # Sol takım - kaleci sol tarafta
            return cx < threshold_px
        else:  # Sağ takım - kaleci sağ tarafta
            return cx > (self.frame_width - threshold_px)
    
    
    def _get_goalkeeper_team_by_position(self, bbox: Tuple[float, float, float, float]) -> Optional[int]:
        """
        Kalecinin takımını pozisyonuna göre belirle.
        Sol kale önündeki kaleci -> L takımı (0) -> L1
        Sağ kale önündeki kaleci -> R takımı (1) -> R1
        
        Returns:
            0 = Sol takım (L), 1 = Sağ takım (R), None = Kale bölgesinde değil
        """
        cx = (bbox[0] + bbox[2]) / 2
        threshold_px = self.frame_width * self.goalkeeper_position_threshold
        
        if cx < threshold_px:
            # Sol kale önünde -> Sol takım kalecisi (L1)
            return 0
        elif cx > (self.frame_width - threshold_px):
            # Sağ kale önünde -> Sağ takım kalecisi (R1)
            return 1
        else:
            # Orta sahada - kaleci değil
            return None
    
    def _get_goalkeeper_team_by_position(self, bbox: Tuple[float, float, float, float]) -> Optional[int]:
        """
        Kalecinin takımını pozisyonuna göre belirle.
        Sol kale önündeki kaleci -> L takımı (0) -> L1
        Sağ kale önündeki kaleci -> R takımı (1) -> R1
        
        Returns:
            0 = Sol takım (L), 1 = Sağ takım (R), None = Kale bölgesinde değil
        """
        cx = (bbox[0] + bbox[2]) / 2
        threshold_px = self.frame_width * self.goalkeeper_position_threshold
        
        if cx < threshold_px:
            # Sol kale önünde -> Sol takım kalecisi (L1)
            return 0
        elif cx > (self.frame_width - threshold_px):
            # Sağ kale önünde -> Sağ takım kalecisi (R1)
            return 1
        else:
            # Orta sahada - kaleci değil
            return None
    
    def _match_tracks(self, tracks: List, frame_idx: int, 
                      raw_team_info: Optional[Dict[int, int]]) -> Tuple[List, List]:
        """Mevcut track'leri kayıtlı oyuncularla eşleştir."""
        matched = []
        unmatched = []
        need_rematch = []  # Mapping silinen track'ler - yeniden eşleştir ama yeni ID verme
        
        # RENK KAYBI TOLERANSI: Kaç frame renk bilgisi olmadan ID korunsun
        COLOR_MISSING_TOLERANCE = 90  # 90 frame (yaklaşık 3 saniye) - daha toleranslı
        
        for track in tracks:
            bytetrack_id = track.track_id
            current_team = self._get_team_from_track(track, raw_team_info)
            
            # 1. Doğrudan mapping varsa kullan
            if bytetrack_id in self.bytetrack_to_team_id:
                team_id = self.bytetrack_to_team_id[bytetrack_id]
                if team_id in self.players:
                    player = self.players[team_id]
                    
                    # RENK BİLGİSİ YOKSA: Mevcut ID'yi koru, renk kaybı sayacını artır
                    if current_team is None:
                        player.color_missing_count += 1
                        # Tolerans içindeyse ID'yi koru
                        if player.color_missing_count <= COLOR_MISSING_TOLERANCE:
                            self._update_player(team_id, track, frame_idx, preserve_team=True)
                            matched.append(track)
                            continue
                        # Tolerans aşıldı ama bbox hala yakınsa ID'yi koru
                        elif bbox_distance(track.bbox, player.last_bbox) < 150:
                            self._update_player(team_id, track, frame_idx, preserve_team=True)
                            matched.append(track)
                            continue
                    
                    # RENK DOĞRULAMASI: Eğer renk var ve kayıtlı takımla uyuşmuyorsa
                    elif current_team is not None and player.team != current_team:
                        # Ancak renk yeni geldiyse ve oyuncu uzun süredir izleniyorsa
                        # bu yanlış okuma olabilir - bir kaç frame daha bekle
                        if player.total_frames > 50 and player.color_missing_count > 0:
                            # Renk yeni döndü ama farklı - muhtemelen yanlış okuma
                            # Birkaç frame daha doğru renk gelene kadar bekle
                            player.color_missing_count = max(0, player.color_missing_count - 1)
                            self._update_player(team_id, track, frame_idx, preserve_team=True)
                            matched.append(track)
                            continue
                        
                        # Takım rengi değişti = yanlış eşleştirme!
                        del self.bytetrack_to_team_id[bytetrack_id]
                        need_rematch.append(track)
                        continue
                    else:
                        # Takım uyuşuyor - devam et, renk kaybı sayacını sıfırla
                        player.color_missing_count = 0
                        player.last_known_team = current_team
                        self._update_player(team_id, track, frame_idx)
                        matched.append(track)
                        continue
            
            # 2. ByteTrack ID yeni - yeniden eşleştir veya yeni ID at
            #    SADECE AYNI TAKIMDAN ve çok yakın mesafedeki kayıp oyuncuları eşleştir
            best_match = self._find_best_match_strict(track, current_team, frame_idx)
            
            if best_match is not None:
                # Bu ByteTrack ID'yi mevcut oyuncuya bağla
                self.bytetrack_to_team_id[bytetrack_id] = best_match
                self._update_player(best_match, track, frame_idx)
                matched.append(track)
                
                self.stats['total_reidentifications'] += 1
            else:
                unmatched.append(track)
        
        # 3. Mapping silinen track'leri yeniden eşleştir (ama yeni ID verme!)
        for track in need_rematch:
            bytetrack_id = track.track_id
            current_team = self._get_team_from_track(track, raw_team_info)
            best_match = self._find_best_match_strict(track, current_team, frame_idx)
            
            if best_match is not None:
                self.bytetrack_to_team_id[bytetrack_id] = best_match
                self._update_player(best_match, track, frame_idx)
                matched.append(track)
                self.stats['total_reidentifications'] += 1
            # Eşleşme bulunamazsa hiçbir şey yapma - bu frame'de bu track yok sayılır
            # Yeni ID vermek yerine bekle, bir sonraki frame'de düzelir
        
        return matched, unmatched
    
    def _find_best_match_strict(self, track, team: Optional[int], frame_idx: int) -> Optional[str]:
        """
        Kayıp oyuncuyu mesafe ve IoU bazlı eşleştir.
        Çalım/occlusion durumlarında daha toleranslı davranır.
        Renk bilgisi varsa takım eşleşmesi de kontrol edilir.
        """
        best_team_id = None
        best_score = float('-inf')  # Birleşik skor (yüksek = iyi)
        
        # Mesafe eşikleri - çalım/occlusion için daha toleranslı
        STRICT_DISTANCE = 120.0  # 80 -> 120 piksel (4K için daha uygun)
        EXTENDED_DISTANCE = 180.0  # Uzun süreli occlusion için
        MAX_DISAPPEARED_SHORT = 8  # Kısa mesafe için max kayıp frame
        MAX_DISAPPEARED_LONG = 15  # Uzun mesafe için max kayıp frame (çalım süresi)
        
        # IoU eşiği - üst üste binen oyuncular için
        IOU_THRESHOLD = 0.1  # Düşük IoU bile yeterli olabilir occlusion sonrası
        
        for team_id, player in self.players.items():
            frames_missing = player.disappeared_count
            
            # Takım biliniyorsa, takım uyuşmalı
            if team is not None and player.team != team:
                continue
            
            # Mesafe hesapla
            dist = bbox_distance(track.bbox, player.last_bbox)
            
            # IoU hesapla - occlusion sonrası overlap kontrolü
            iou = bbox_iou(track.bbox, player.last_bbox)
            
            # Kayıp süresine göre mesafe toleransı
            # Kısa kayıp: sıkı mesafe
            # Uzun kayıp (çalım): daha geniş mesafe
            if frames_missing <= MAX_DISAPPEARED_SHORT:
                max_dist = STRICT_DISTANCE
            elif frames_missing <= MAX_DISAPPEARED_LONG:
                max_dist = EXTENDED_DISTANCE
            else:
                continue  # Çok uzun süredir kayıp, eşleştirme yapma
            
            # Mesafe veya IoU kontrolü
            # IoU > 0 ise mesafeyi göz ardı et (üst üste binmiş olabilirler)
            if dist > max_dist and iou < IOU_THRESHOLD:
                continue
            
            # Birleşik skor: IoU bonus + mesafe penalty
            # IoU yüksekse çok iyi, mesafe düşükse de iyi
            score = iou * 100 - dist * 0.5
            
            if score > best_score:
                best_score = score
                best_team_id = team_id
        
        return best_team_id
    
    def _get_next_player_number(self, team: int, is_goalkeeper: bool) -> Optional[int]:
        """Takım için bir sonraki kullanılabilir numarayı döndür."""
        if is_goalkeeper:
            if not self.goalkeeper_assigned[team]:
                self.goalkeeper_assigned[team] = True
                return self.GOALKEEPER_NUMBER
            return None  # Kaleci zaten atanmış
        
        # 2-11 arası ilk boş numarayı bul
        for num in range(2, self.MAX_PLAYERS_PER_TEAM + 1):
            if num not in self.used_numbers[team]:
                self.used_numbers[team].add(num)
                return num
        
        return None  # Takım dolu (11 oyuncu)
    
    def _register_new_player(self, track, frame_idx: int, raw_team_info: Optional[Dict[int, int]]):
        """Yeni oyuncu kaydı oluştur."""
        # Sadece class'ı açıkça 'goalkeeper' olan track'lere L1/R1 atanacak
        is_goalkeeper = (str(track.cls).lower() == "goalkeeper")
        team = None
        if is_goalkeeper:
            gk_team = self._get_goalkeeper_team_by_position(track.bbox)
            if gk_team is not None and not self.goalkeeper_assigned[gk_team]:
                team = gk_team
            else:
                # Kaleci kale bölgesinde değil veya zaten atanmışsa, normal oyuncu olarak işle
                is_goalkeeper = False
        if not is_goalkeeper:
            # Normal oyuncu - renk bazlı takım ataması
            team = self._get_team_from_track(track, raw_team_info)
        if team is None:
            # Pozisyon bazlı fallback
            cx = (track.bbox[0] + track.bbox[2]) / 2
            mid_x = self.frame_width / 2
            team = 0 if cx < mid_x else 1
            if frame_idx < 50 or frame_idx % 200 == 0:
                print(f"[TeamIDStabilizer] Position-based fallback: ByteTrack {track.track_id} -> team {'L' if team == 0 else 'R'} (x={cx:.0f}, mid={mid_x:.0f})")
        number = self._get_next_player_number(team, is_goalkeeper)
        if number is None:
            # Takım dolu veya kaleci zaten atanmış, kayıt yapma
            return
        team_id = make_team_id(team, number)
        self.players[team_id] = TrackedPlayer(
            team_id=team_id,
            team=team,
            number=number,
            is_goalkeeper=is_goalkeeper,
            last_bbox=track.bbox,
            last_frame=frame_idx,
            last_bytetrack_id=track.track_id,
            total_frames=1,
            confidence_history=[track.score] if track.score else [],
            color_missing_count=0,
            last_known_team=team
        )
        self.bytetrack_to_team_id[track.track_id] = team_id
        self.stats['total_new_tracks'] += 1
        prefix = "GK" if is_goalkeeper else "Player"
        if frame_idx < 50 or frame_idx % 100 == 0:
            print(f"[TeamIDStabilizer] New {prefix}: ByteTrack {track.track_id} -> {team_id}")
    
    def _update_player(self, team_id: str, track, frame_idx: int, preserve_team: bool = False):
        """Mevcut oyuncu bilgilerini güncelle.
        
        Args:
            team_id: Oyuncu ID'si (L3, R5 gibi)
            track: ByteTrack track objesi
            frame_idx: Frame numarası
            preserve_team: True ise renk kaybı durumunda mevcut takım bilgisi korunur
        """
        player = self.players[team_id]
        player.last_bbox = track.bbox
        player.last_frame = frame_idx
        player.last_bytetrack_id = track.track_id
        player.disappeared_count = 0
        player.total_frames += 1
        
        # Renk korunuyorsa (renk kaybı durumu) bilgi mesajı
        if preserve_team and frame_idx % 100 == 0:
            print(f"[TeamIDStabilizer] Preserving team for {team_id} (color missing: {player.color_missing_count} frames)")
        
        if track.score:
            player.confidence_history.append(track.score)
            if len(player.confidence_history) > 10:
                player.confidence_history.pop(0)
    
    def _mark_disappeared(self, frame_idx: int, seen_ids: Optional[Set[str]] = None):
        """Görünmeyen oyuncuları işaretle ve gerekirse sil."""
        if seen_ids is None:
            seen_ids = set()
        
        to_delete = []
        
        for team_id, player in self.players.items():
            if team_id not in seen_ids:
                player.disappeared_count += 1
                
                if player.disappeared_count > self.max_disappeared:
                    to_delete.append(team_id)
        
        for team_id in to_delete:
            player = self.players[team_id]
            
            # ByteTrack mapping'ini sil
            if player.last_bytetrack_id in self.bytetrack_to_team_id:
                del self.bytetrack_to_team_id[player.last_bytetrack_id]
            
            # Numarayı serbest bırak
            if player.number != self.GOALKEEPER_NUMBER:
                self.used_numbers[player.team].discard(player.number)
            else:
                self.goalkeeper_assigned[player.team] = False
            
            del self.players[team_id]
    
    def get_stats(self) -> dict:
        """İstatistikleri döndür."""
        return {
            **self.stats,
            'active_players': len(self.players),
            'left_goalkeeper': self.goalkeeper_assigned[0],
            'right_goalkeeper': self.goalkeeper_assigned[1]
        }
    
    def reset(self):
        """Tüm state'i sıfırla."""
        self.players.clear()
        self.bytetrack_to_team_id.clear()
        self.used_numbers = {0: set(), 1: set()}
        self.goalkeeper_assigned = {0: False, 1: False}
        self.stats = {
            'total_reidentifications': 0,
            'total_new_tracks': 0,
            'left_team_count': 0,
            'right_team_count': 0
        }


# Backward compatibility alias
IDStabilizer = TeamBasedIDStabilizer
