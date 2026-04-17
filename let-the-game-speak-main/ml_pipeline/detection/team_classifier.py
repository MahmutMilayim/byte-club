"""
Team Classification Helper Functions
Extracted from original main.py for use in vision.py
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional


def get_grass_color(img: np.ndarray) -> Tuple[float, float, float]:
    """
    Görüntüdeki çim (yeşil) alanların ortalama BGR rengini döndürür.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]


def get_kits_colors(players: List[np.ndarray], grass_hsv: Optional[np.ndarray] = None, 
                   frame: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """
    Her oyuncu kırpığı için forma (üst gövde) rengini yaklaşıklar.
    
    Hareket bulanıklığı ve küçük görüntülere karşı dayanıklı versiyon:
    - Minimum piksel sayısı kontrolü
    - Merkez odaklı renk alımı
    - Laplace varyansı ile bulanıklık kontrolü
    """
    kits_colors = []
    
    if grass_hsv is None and frame is not None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
    
    for player_img in players:
        if player_img.size == 0:
            continue
        
        h, w = player_img.shape[:2]
        
        # MİNİMUM BOYUT KONTROLÜ: Daha düşük eşik - küçük oyuncuları da al
        if h < 10 or w < 5:
            continue
        
        # BULANIKLIK KONTROLÜ: Daha düşük eşik - bulanık görüntüleri de kabul et
        gray = cv2.cvtColor(player_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 15:  # Çok düşük eşik - sadece aşırı bulanıkları reddet
            continue
            
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
        
        # Çime yakın tonları maskele
        h_grass = int(grass_hsv[0, 0, 0]) if grass_hsv is not None else 50
        h_lo = max(h_grass - 15, 0)
        h_hi = min(h_grass + 15, 179)
        lower_green = np.array([h_lo, 30, 30])
        upper_green = np.array([h_hi, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Üst gövde odaklı maske (formanın olduğu alan)
        mask = cv2.bitwise_not(mask)
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        # Üst yarının merkez %60'ını al (kenarları hariç tut)
        top_third = h // 3
        center_margin = w // 5
        upper_mask[top_third:h // 2, center_margin:w - center_margin] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        
        # Yeterli piksel var mı kontrol et - daha düşük eşik
        non_zero = cv2.countNonZero(mask)
        if non_zero < 15:  # En az 15 piksel olmalı (daha toleranslı)
            continue
        
        # Maskelenen piksellerin BGR ortalaması
        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3], dtype=np.float32)
        kits_colors.append(kit_color)
    
    return kits_colors


def get_kits_classifier(kits_colors: List[np.ndarray]) -> KMeans:
    """
    Forma renklerine göre oyuncuları 2 kümeye (2 takıma) ayıracak KMeans modelini fit eder.
    """
    if len(kits_colors) < 2:
        base = np.array(kits_colors[0] if len(kits_colors) == 1 else [0, 0, 0], dtype=np.float32)
        X = np.vstack([base, base + 1.0]).astype(np.float32)
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(X)
        return kmeans
    
    X = np.asarray(kits_colors, dtype=np.float32)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(X)
    return kmeans


def classify_kits(kits_classifier: KMeans, kits_colors: List[np.ndarray]) -> np.ndarray:
    """
    Kit renkleri üzerinden KMeans modeli ile takım etiketi (0/1) tahmini yapar.
    """
    X = np.asarray(kits_colors, dtype=np.float32)
    team = kits_classifier.predict(X)
    return team
