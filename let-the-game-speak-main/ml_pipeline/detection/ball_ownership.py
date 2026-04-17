import numpy as np
from collections import deque
from collections import Counter

class BallOwnershipProcessor:
    # DİKKAT: Parametre ismi burada 'smoothing_window' olarak düzeltildi.
    def __init__(self, fps=25, smoothing_window=0.6, distance_threshold=150,
                 ball_speed_threshold=12.0, close_distance_threshold=70.0,
                 extreme_speed_threshold=25.0):
        """
        Args:
            fps: Video FPS değeri.
            smoothing_window: Saniye cinsinden yumuşatma penceresi (örn: 0.6 sn = 15 frame).
            distance_threshold: Piksel cinsinden maksimum sahiplik mesafesi (150px).
            ball_speed_threshold: Top bu hızın üstündeyse ve oyuncuya yakın değilse sahiplik atanmaz (piksel/frame).
                                  Artırıldı: 12 px/frame (önceden 8) - hızlı pasları yakalamak için.
            close_distance_threshold: Top bu mesafeden yakınsa, hızlı olsa bile sahiplik atanır (piksel).
                                      Artırıldı: 70px (önceden 40) - pas alan oyuncuları daha erken yakalamak için.
            extreme_speed_threshold: Top bu hızın üstündeyse (şut seviyesi) yakın olsa bile sahiplik atanmaz (piksel/frame).
                                     Şut gibi durumlarda kaleciye yanlış ownership verilmesini önler.
        """
        self.window_size = int(fps * smoothing_window)
        self.threshold = distance_threshold
        self.ball_speed_threshold = ball_speed_threshold
        self.close_distance_threshold = close_distance_threshold
        self.extreme_speed_threshold = extreme_speed_threshold
        # Son N frame'deki sahiplikleri tutar
        self.history = deque(maxlen=self.window_size)
        # Top pozisyon geçmişi (hız hesabı için)
        self.prev_ball_pos = None

    def calculate_distance(self, ball_box, player_box):
        """
        Top merkezi ile oyuncunun ayakları (alt orta nokta) arasındaki mesafeyi ölçer.
        Box formatı: [x1, y1, x2, y2]
        """
        bx = (ball_box[0] + ball_box[2]) / 2
        by = (ball_box[1] + ball_box[3]) / 2
        
        px = (player_box[0] + player_box[2]) / 2
        py = player_box[3] 
        
        return np.sqrt((bx - px)**2 + (by - py)**2)

    def calculate_ball_speed(self, ball_box):
        """
        Topun hızını hesaplar (piksel/frame).
        Önceki pozisyon ile şimdiki pozisyon arasındaki mesafe.
        """
        bx = (ball_box[0] + ball_box[2]) / 2
        by = (ball_box[1] + ball_box[3]) / 2
        current_pos = (bx, by)
        
        if self.prev_ball_pos is None:
            self.prev_ball_pos = current_pos
            return 0.0
        
        speed = np.sqrt((current_pos[0] - self.prev_ball_pos[0])**2 + 
                       (current_pos[1] - self.prev_ball_pos[1])**2)
        self.prev_ball_pos = current_pos
        return speed

    def update(self, tracks):
        """
        O anki frame'deki track listesini alır ve topun sahibini (ID) döndürür.
        Top hızlı hareket ediyorsa ve oyuncuya yakın değilse, sahiplik atanmaz.
        Ancak top oyuncuya çok yakınsa (ayağında gibi), hızlı olsa bile sahiplik atanır.
        """
        ball_track = None
        player_tracks = []

        for t in tracks:
            # ByteTrack class ismini güvenli şekilde al
            cls_name = str(getattr(t, 'cls', '')).lower() 
            
            if 'ball' in cls_name:
                ball_track = t
            elif 'player' in cls_name or 'keeper' in cls_name or 'goalkeeper' in cls_name:
                player_tracks.append(t)

        current_closest_id = None
        min_dist = float('inf')
        ball_speed = 0.0

        if ball_track and player_tracks:
            # Top hızını hesapla
            ball_speed = self.calculate_ball_speed(ball_track.bbox)
            
            for p in player_tracks:
                dist = self.calculate_distance(ball_track.bbox, p.bbox)
                if dist < min_dist:
                    min_dist = dist
                    current_closest_id = p.track_id

        final_raw_id = None
        
        if min_dist < self.threshold:
            # Top aşırı hızlı (şut seviyesi) - yakın olsa bile sahiplik atama
            # Bu, şut atıldığında kaleciye yanlış ownership verilmesini önler
            if ball_speed > self.extreme_speed_threshold:
                final_raw_id = None  # Şut seviyesinde hız, ownership yok
            # Top çok yakınsa (ayağında gibi) - hızlı olsa bile sahiplik ata
            elif min_dist < self.close_distance_threshold:
                final_raw_id = current_closest_id
            # Top hızlı ve uzakça ise - sahiplik atama (top üzerinden geçiyor)
            elif ball_speed > self.ball_speed_threshold:
                final_raw_id = None  # Top hızlı ve yakın değil, sahiplik yok
            else:
                # Normal durum - top yavaş veya orta mesafede
                final_raw_id = current_closest_id
        
        self.history.append(final_raw_id)

        if not self.history:
            return None, min_dist
        
        counts = Counter(self.history)
        most_common = counts.most_common(1)[0]
        
        smoothed_owner_id = most_common[0]
        
        return smoothed_owner_id, min_dist