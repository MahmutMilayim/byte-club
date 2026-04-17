"""
2D Field Visualization Module
Gerçek video ile yan yana 2D saha görünümü oluşturur
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


class Field2DRenderer:
    """
    2D futbol sahası çizer ve oyuncuları/topu gösterir
    
    Saha boyutları: 105m x 68m
    Kale genişliği: 7.32m
    """
    
    # Saha boyutları (metre)
    FIELD_LENGTH = 105.0
    FIELD_WIDTH = 68.0
    GOAL_WIDTH = 7.32
    PENALTY_AREA_LENGTH = 16.5
    PENALTY_AREA_WIDTH = 40.32
    GOAL_AREA_LENGTH = 5.5
    GOAL_AREA_WIDTH = 18.32
    CENTER_CIRCLE_RADIUS = 9.15
    PENALTY_SPOT_DISTANCE = 11.0
    
    def __init__(self, field_width: int = 500, field_height: int = 320):
        """
        Args:
            field_width: 2D saha piksel genişliği
            field_height: 2D saha piksel yüksekliği
        """
        self.field_width = field_width
        self.field_height = field_height
        
        # Kale için margin
        self.margin = 3.0  # metre
        self.total_length = self.FIELD_LENGTH + 2 * self.margin
        
        # Ölçek faktörleri
        self.scale_x = field_width / self.total_length
        self.scale_y = field_height / self.FIELD_WIDTH
        
        # Boş saha template'i oluştur
        self.field_template = self._create_field_template()
    
    def _to_2d(self, fx: float, fy: float) -> Tuple[int, int]:
        """Saha koordinatını (metre) 2D piksele çevir"""
        x = int((fx + self.margin) * self.scale_x)
        y = int(fy * self.scale_y)
        return (x, y)
    
    def _create_field_template(self) -> np.ndarray:
        """Boş 2D saha template'i oluştur"""
        field = np.zeros((self.field_height, self.field_width, 3), dtype=np.uint8)
        field[:] = (34, 139, 34)  # Yeşil zemin
        
        white = (255, 255, 255)
        
        # Dış çizgiler
        cv2.rectangle(field, self._to_2d(0, 0), self._to_2d(105, 68), white, 2)
        
        # Orta çizgi
        cv2.line(field, self._to_2d(52.5, 0), self._to_2d(52.5, 68), white, 2)
        
        # Orta daire
        center = self._to_2d(52.5, 34)
        radius = int(self.CENTER_CIRCLE_RADIUS * self.scale_x)
        cv2.circle(field, center, radius, white, 2)
        cv2.circle(field, center, 3, white, -1)  # Merkez nokta
        
        # Sol ceza sahası (16.5m x 40.32m)
        cv2.rectangle(field, self._to_2d(0, 13.84), self._to_2d(16.5, 54.16), white, 2)
        
        # Sağ ceza sahası
        cv2.rectangle(field, self._to_2d(88.5, 13.84), self._to_2d(105, 54.16), white, 2)
        
        # Sol kale alanı (5.5m x 18.32m)
        cv2.rectangle(field, self._to_2d(0, 24.84), self._to_2d(5.5, 43.16), white, 2)
        
        # Sağ kale alanı
        cv2.rectangle(field, self._to_2d(99.5, 24.84), self._to_2d(105, 43.16), white, 2)
        
        # KALELER (7.32m genişlik)
        goal_top = 34 - self.GOAL_WIDTH / 2
        goal_bottom = 34 + self.GOAL_WIDTH / 2
        
        # Sol kale
        cv2.rectangle(field, self._to_2d(-2.5, goal_top), self._to_2d(0, goal_bottom), white, 2)
        cv2.rectangle(field, self._to_2d(-2.5, goal_top), self._to_2d(0, goal_bottom), (220, 220, 220), -1)
        
        # Sağ kale
        cv2.rectangle(field, self._to_2d(105, goal_top), self._to_2d(107.5, goal_bottom), white, 2)
        cv2.rectangle(field, self._to_2d(105, goal_top), self._to_2d(107.5, goal_bottom), (220, 220, 220), -1)
        
        # Penaltı noktaları
        cv2.circle(field, self._to_2d(11, 34), 3, white, -1)  # Sol
        cv2.circle(field, self._to_2d(94, 34), 3, white, -1)  # Sağ
        
        return field
    
    def pixel_to_field_simple(self, px: float, py: float, 
                               frame_width: int, frame_height: int) -> Optional[Tuple[float, float]]:
        """
        Basit piksel -> saha koordinat dönüşümü (homografi olmadan)
        Sadece yaklaşık bir eşleme yapar
        
        Bu fonksiyon kamera perspektifini dikkate almaz, 
        sadece lineer bir mapping yapar (demo amaçlı)
        """
        # Basit lineer mapping (gerçek projede homografi kullanılmalı)
        # Ekranın sol-sağ -> saha 0-105
        # Ekranın üst-alt -> saha 0-68
        
        field_x = (px / frame_width) * self.FIELD_LENGTH
        field_y = (py / frame_height) * self.FIELD_WIDTH
        
        return (field_x, field_y)
    
    def render(self, tracks: List[Dict], frame_width: int, frame_height: int,
               pixel_to_field_func=None) -> np.ndarray:
        """
        Tracks'leri 2D sahaya çiz
        
        Args:
            tracks: [{'track_id': 'L2', 'cls': 'player', 'bbox': [...], ...}, ...]
            frame_width: Video frame genişliği
            frame_height: Video frame yüksekliği
            pixel_to_field_func: Piksel->saha dönüşüm fonksiyonu (opsiyonel)
        """
        field = self.field_template.copy()
        
        for track in tracks:
            bbox = track.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = bbox
            cls_name = str(track.get('cls', '')).lower()
            track_id = str(track.get('track_id', '')).upper()
            
            # Ayak pozisyonu (bbox'ın alt merkezi)
            foot_x = (x1 + x2) / 2
            foot_y = y2  # Alt kenar
            
            # Top için merkez kullan
            if 'ball' in cls_name:
                foot_y = (y1 + y2) / 2
            
            # Piksel -> Saha koordinatı
            if pixel_to_field_func:
                field_pos = pixel_to_field_func(foot_x, foot_y)
            else:
                field_pos = self.pixel_to_field_simple(foot_x, foot_y, frame_width, frame_height)
            
            if field_pos is None:
                continue
            
            fx, fy = field_pos
            
            # Saha koordinatı -> 2D piksel
            x2d, y2d = self._to_2d(fx, fy)
            
            # Saha içinde mi?
            if not (0 <= x2d < self.field_width and 0 <= y2d < self.field_height):
                continue
            
            # Renk ve boyut belirle
            if 'ball' in cls_name or track_id == 'BALL':
                color = (0, 255, 255)  # Sarı - TOP
                size = 10
                border_color = (0, 0, 0)
            elif track_id.startswith('R'):
                # R = Right team = Kırmızı takım
                color = (0, 0, 255)  # Kırmızı (BGR)
                size = 8
                border_color = (255, 255, 255)
                if track_id == 'R1':
                    size = 12  # Kaleci büyük
            elif track_id.startswith('L'):
                # L = Left team = Mavi takım
                color = (255, 0, 0)  # Mavi (BGR)
                size = 8
                border_color = (255, 255, 255)
                if track_id == 'L1':
                    size = 12  # Kaleci büyük
            elif 'referee' in cls_name:
                color = (0, 0, 0)  # Siyah - HAKEM
                size = 6
                border_color = (0, 255, 255)  # Sarı border
            else:
                color = (128, 128, 128)  # Gri
                size = 6
                border_color = (255, 255, 255)
            
            # Çiz
            cv2.circle(field, (x2d, y2d), size + 2, border_color, -1)
            cv2.circle(field, (x2d, y2d), size, color, -1)
            
            # Top için ekstra vurgu
            if 'ball' in cls_name or track_id == 'BALL':
                cv2.circle(field, (x2d, y2d), size + 5, (0, 255, 255), 2)
            
            # Track ID yazısı (kaleci için)
            if track_id in ('L1', 'R1'):
                cv2.putText(field, 'GK', (x2d - 8, y2d - size - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return field
    
    def create_side_by_side(self, frame: np.ndarray, field_2d: np.ndarray) -> np.ndarray:
        """
        Video frame ve 2D sahayı yan yana birleştir
        
        Args:
            frame: Orijinal video frame
            field_2d: 2D saha görünümü
        """
        # 2D sahayı video yüksekliğine ölçekle
        target_height = frame.shape[0]
        scale = target_height / field_2d.shape[0]
        new_width = int(field_2d.shape[1] * scale)
        
        field_2d_resized = cv2.resize(field_2d, (new_width, target_height))
        
        # Yan yana birleştir
        combined = np.hstack([frame, field_2d_resized])
        
        return combined


def create_field_renderer(field_width: int = 500, field_height: int = 320) -> Field2DRenderer:
    """Field2DRenderer oluştur"""
    return Field2DRenderer(field_width=field_width, field_height=field_height)
