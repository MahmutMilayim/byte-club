"""
Field Transform - Piksel koordinatlarını 2D saha koordinatlarına dönüştür
Soccer Line'ın projection matrix'ini kullanır
"""
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List


class FieldTransform:
    """
    Piksel ↔ 2D Saha koordinat dönüşümü
    
    Saha koordinatları (metre):
    - X: 0-105 (sol kale = 0, sağ kale = 105)
    - Y: 0-68 (alt çizgi = 0, üst çizgi = 68)
    
    Sol kale merkezi: (0, 34)
    Sağ kale merkezi: (105, 34)
    Orta saha: (52.5, 34)
    """
    
    # Saha boyutları (metre)
    FIELD_LENGTH = 105.0  # X ekseni
    FIELD_WIDTH = 68.0    # Y ekseni
    
    # Kale pozisyonları
    LEFT_GOAL = (0, 34)
    RIGHT_GOAL = (105, 34)
    
    def __init__(self):
        self.P = None  # Projection matrix
        self.H = None  # Homography matrix (2D için)
        self.frame_size = None
        
    def set_projection_matrix(self, P: np.ndarray, frame_size: Tuple[int, int]):
        """
        Projection matrix'i ayarla
        P: 3x4 projection matrix
        frame_size: (width, height)
        """
        self.P = P
        self.frame_size = frame_size
        
        # Homography hesapla (z=0 düzlemi için)
        # P = [p1 p2 p3 p4] -> H = [p1 p2 p4] (z=0 için p3'ü çıkar)
        self.H = np.column_stack([P[:, 0], P[:, 1], P[:, 3]])
        
    def pixel_to_field(self, pixel_x: float, pixel_y: float) -> Optional[Tuple[float, float]]:
        """
        Piksel koordinatını 2D saha koordinatına dönüştür
        
        Args:
            pixel_x, pixel_y: Piksel koordinatları
            
        Returns:
            (field_x, field_y) metre cinsinden veya None (dönüşüm başarısızsa)
        """
        if self.H is None:
            return None
        
        try:
            # Homography'nin tersini al
            H_inv = np.linalg.inv(self.H)
            
            # Piksel -> normalize koordinat
            pixel = np.array([pixel_x, pixel_y, 1.0])
            world_h = H_inv @ pixel
            
            # Homojen koordinattan kartezyen koordinata
            if abs(world_h[2]) < 1e-6:
                return None
                
            world_x = world_h[0] / world_h[2]
            world_y = world_h[1] / world_h[2]
            
            # Saha merkezini offset olarak ekle (projection'da çıkarılmıştı)
            field_x = world_x + self.FIELD_LENGTH / 2
            field_y = world_y + self.FIELD_WIDTH / 2
            
            # Saha sınırları içinde mi kontrol et (biraz tolerans ile)
            margin = 10  # metre
            if not (-margin <= field_x <= self.FIELD_LENGTH + margin and
                    -margin <= field_y <= self.FIELD_WIDTH + margin):
                return None
            
            return (field_x, field_y)
            
        except np.linalg.LinAlgError:
            return None
    
    def field_to_pixel(self, field_x: float, field_y: float, z: float = 0) -> Optional[Tuple[int, int]]:
        """
        2D saha koordinatını piksele dönüştür
        
        Args:
            field_x, field_y: Saha koordinatları (metre)
            z: Yükseklik (varsayılan 0)
            
        Returns:
            (pixel_x, pixel_y) veya None
        """
        if self.P is None:
            return None
        
        # Saha merkezini çıkar (projection'da yapıldığı gibi)
        world_point = np.array([
            field_x - self.FIELD_LENGTH / 2,
            field_y - self.FIELD_WIDTH / 2,
            z,
            1.0
        ])
        
        # Project
        p = self.P @ world_point
        
        if abs(p[2]) < 1e-6:
            return None
            
        pixel_x = int(p[0] / p[2])
        pixel_y = int(p[1] / p[2])
        
        return (pixel_x, pixel_y)
    
    def get_goal_zones(self) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Kale bölgelerini döndür
        
        Returns:
            {
                'left': (x_min, y_min, x_max, y_max),
                'right': (x_min, y_min, x_max, y_max)
            }
        """
        # Kale boyutları: 7.32m genişlik, merkez y=34
        goal_half_width = 7.32 / 2
        
        return {
            'left': (0, 34 - goal_half_width, 5.5, 34 + goal_half_width),  # Kale alanı
            'right': (99.5, 34 - goal_half_width, 105, 34 + goal_half_width)
        }
    
    def is_in_goal_zone(self, field_x: float, field_y: float, goal: str = 'right') -> bool:
        """
        Koordinat kale bölgesinde mi?
        
        Args:
            field_x, field_y: Saha koordinatları
            goal: 'left' veya 'right'
        """
        zones = self.get_goal_zones()
        zone = zones.get(goal)
        
        if zone is None:
            return False
            
        x_min, y_min, x_max, y_max = zone
        return x_min <= field_x <= x_max and y_min <= field_y <= y_max
    
    def get_shot_direction(self, start_x: float, end_x: float) -> str:
        """
        Şut yönünü belirle
        
        Returns:
            'left' veya 'right' (hangi kaleye doğru)
        """
        if end_x < start_x:
            return 'left'
        else:
            return 'right'
    
    def transform_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Detection listesini 2D koordinatlara dönüştür
        
        Args:
            detections: [{'bbox': [x1,y1,x2,y2], 'class': 'player', ...}, ...]
            
        Returns:
            Aynı liste + 'field_pos': (x, y) eklenerek
        """
        result = []
        
        for det in detections:
            det_copy = det.copy()
            
            if 'bbox' in det:
                bbox = det['bbox']
                # Ayak pozisyonu (bbox'ın alt merkezi)
                foot_x = (bbox[0] + bbox[2]) / 2
                foot_y = bbox[3]  # Alt kenar
                
                field_pos = self.pixel_to_field(foot_x, foot_y)
                det_copy['field_pos'] = field_pos
                
            result.append(det_copy)
            
        return result


def create_field_transform_from_calibration(calibration_data: Dict) -> FieldTransform:
    """
    Kalibrasyon verisinden FieldTransform oluştur
    
    Args:
        calibration_data: Soccer Line'dan gelen kalibrasyon
    """
    transform = FieldTransform()
    
    if 'P' in calibration_data and 'frame_size' in calibration_data:
        transform.set_projection_matrix(
            calibration_data['P'],
            calibration_data['frame_size']
        )
    
    return transform
