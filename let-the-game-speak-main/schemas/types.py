from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union


BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


# ================= Team-Based ID Helpers =================

def parse_team_id(track_id: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Track ID'sinden takım ve numara çıkarır.
    
    Args:
        track_id: "L1", "R5", "ball" gibi string ID
        
    Returns:
        (team, number) - team: 0=Left, 1=Right, None=ball/unknown
                        number: 1-11 arası oyuncu numarası
    """
    if not track_id or track_id == "ball":
        return (None, None)
    
    if track_id.startswith("L") and track_id[1:].isdigit():
        return (0, int(track_id[1:]))
    elif track_id.startswith("R") and track_id[1:].isdigit():
        return (1, int(track_id[1:]))
    
    return (None, None)


def get_team_from_id(track_id: str) -> Optional[int]:
    """Track ID'sinden takımı döndürür (0=Left, 1=Right)."""
    team, _ = parse_team_id(track_id)
    return team


def make_team_id(team: int, number: int) -> str:
    """
    Takım ve numaradan ID oluşturur.
    
    Args:
        team: 0=Left, 1=Right
        number: 1-11 arası oyuncu numarası
        
    Returns:
        "L1", "R5" gibi string ID
    """
    prefix = "L" if team == 0 else "R"
    return f"{prefix}{number}"


@dataclass
class Track:
    """
    Tek bir obje izi (track) için kayıt.

    Attributes:
        track_id: Takım bazlı ID (L1-L11, R1-R11) veya "ball".
                  L1/R1 kalecilere rezerve.
        cls: Rol bilgisi (ör: "player", "goalkeeper", "ball").
        bbox: (x1, y1, x2, y2) formatında bounding box (piksel).
        t: Videonun başlangıcından itibaren saniye cinsinden zaman damgası.
        score: Takip edilen objenin confidence skoru (opsiyonel).
        field_x: 2D saha koordinatı X (metre, 0-105). None ise kalibrasyon yok.
        field_y: 2D saha koordinatı Y (metre, 0-68). None ise kalibrasyon yok.
        field_z: Top için yükseklik (metre, 0-2.44 kale yüksekliği). Sadece top için.
    """
    track_id: str  # "L1", "R5", "ball" gibi string ID
    cls: str
    bbox: BBox
    t: float
    score: Optional[float] = None
    field_x: Optional[float] = None  # 2D saha X koordinatı (metre)
    field_y: Optional[float] = None  # 2D saha Y koordinatı (metre)
    field_z: Optional[float] = None  # Top yüksekliği (metre, sadece ball için)
    
    @property
    def team(self) -> Optional[int]:
        """Track ID'sinden takımı döndürür (0=Left, 1=Right)."""
        return get_team_from_id(self.track_id)
    
    @property
    def player_number(self) -> Optional[int]:
        """Track ID'sinden oyuncu numarasını döndürür (1-11)."""
        _, number = parse_team_id(self.track_id)
        return number


@dataclass
class FrameRecord:
    """
    Tek bir video karesi için ID'li detection listesi.

    Attributes:
        frame_idx: 0-based frame index.
        t: Bu kareye karşılık gelen zaman (saniye).
        tracks: Bu karedeki tüm Track objeleri.
    """
    frame_idx: int
    t: float
    tracks: List[Track]
    
    ball_owner: Optional[str] = None       # Oyuncu ID'si (L3, R5 gibi)
    ball_owner_team: Optional[int] = None  # Takım ID'si (0 veya 1)

@dataclass
class BallPosition:
    """
    Temizlenmiş ve boşlukları doldurulmuş top pozisyon kaydı.
    Bu, Event Spotting modülünün okuyacağı nihai top izidir.
    """
    t: float    # zaman (saniye)
    x: float
    y: float


@dataclass
class TracksData:
    """
    Uçtan uca işlenmiş ve Event Spotting modülüne aktarılacak nihai veri şeması.
    Bu, ml_pipeline/io/writer.py tarafından JSON'a yazılacak ana objedir.
    """
    video: Dict[str, Union[float, int]]  # {"fps": 25.0, "w": 1280, "h": 720}
    teams: Dict[str, int]                # {"left_label": 0}
    frames: List[FrameRecord]            # Tüm ham/takip edilmiş FrameRecord verileri
    ball_track: List[BallPosition]       # Temizlenmiş ve doldurulmuş top izi


# ================= Feature Extraction Şemaları =================

@dataclass
class PlayerFeature:
    """
    Tek bir oyuncu için feature seti (event detection için).
    
    Attributes:
        track_id: Takım bazlı ID (L1-L11, R1-R11).
        x, y: Merkez pozisyon (bbox merkezinden hesaplanır).
        vx, vy: Hız bileşenleri (piksel/saniye), None ise hesaplanamadı.
    """
    track_id: str  # "L1", "R5" gibi string ID
    x: float
    y: float
    vx: Optional[float] = None
    vy: Optional[float] = None
    
    @property
    def team(self) -> Optional[int]:
        """Track ID'sinden takımı döndürür (0=Left, 1=Right)."""
        return get_team_from_id(self.track_id)


@dataclass
class BallFeature:
    """
    Top için feature seti.
    
    Attributes:
        x, y: Top merkez pozisyonu.
        speed: Toplam hız (piksel/saniye).
        vx, vy: Hız bileşenleri (piksel/saniye).
    """
    x: float
    y: float
    speed: float
    vx: float
    vy: float


@dataclass
class FrameFeatures:
    """
    Event detection için tek frame'in tüm feature'ları.
    
    Attributes:
        frame_idx: 0-based frame index.
        t: Zaman damgası (saniye).
        players: Bu frame'deki tüm oyuncular için feature'lar.
        ball: Top feature'ı (yoksa None).
    """
    frame_idx: int
    t: float
    players: List[PlayerFeature]
    ball: Optional[BallFeature] = None


# ================= Ball Segment Şemaları =================

from enum import Enum

class BallSegmentType(str, Enum):
    """
    Top hareket segmenti türleri.
    
    - pass: Aynı takım içinde farklı oyuncuya pas
    - dribble: Aynı oyuncu top sürme
    - turnover: Top kaybı - bir takımdan diğer takıma geçiş
    - shot_candidate: Şut adayı - top kayboldu/çıktı
    - unknown: Bilinmeyen hareket
    """
    PASS = "pass"
    DRIBBLE = "dribble"
    TURNOVER = "turnover"
    SHOT_CANDIDATE = "shot_candidate"
    UNKNOWN = "unknown"


@dataclass
class BallSegment:
    """
    Top hareket segmenti.
    
    Attributes:
        start_time: Segment başlangıç frame'i.
        end_time: Segment bitiş frame'i.
        start_owner: Topun başlangıçtaki sahibi (L1-L11, R1-R11).
        end_owner: Topun bitişindeki sahibi.
        segment_type: Segment türü (pass, dribble, turnover, shot_candidate, unknown).
        displacement: Toplam yer değiştirme (piksel).
        direction_vector: Hareket yönü vektörü [dx, dy].
        average_speed: Ortalama hız.
    """
    start_time: int
    end_time: int
    start_owner: Optional[str]
    end_owner: Optional[str]
    segment_type: str  # BallSegmentType değerlerinden biri
    displacement: float
    direction_vector: List[float]
    average_speed: float


@dataclass
class TurnoverStats:
    """
    Top kaybı istatistikleri.
    
    Attributes:
        total_turnovers: Toplam top kaybı sayısı.
        left_team_losses: Sol takımın top kaybı sayısı.
        right_team_losses: Sağ takımın top kaybı sayısı.
        turnovers_by_player: Oyuncu bazlı top kaybı sayıları.
        turnovers: Top kaybı segmentleri listesi.
    """
    total_turnovers: int
    left_team_losses: int
    right_team_losses: int
    turnovers_by_player: Dict[str, int]
    turnovers: List[BallSegment]