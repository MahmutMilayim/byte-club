#!/usr/bin/env python3
"""
Pass Detection (Segment-Based)

Bu modül ball_motion_seg.py'nin ürettiği segment çıktılarını kullanarak
pas tespiti yapar. Kendi sahiplik hesabı yapmaz, mevcut segment verisine güvenir.

Mantık:
1. ball_segments.json'dan segment_type="pass" olanları al
2. Takım kontrolü yap (sender ve receiver aynı takımdan mı?)
3. Ek filtreleme ve zenginleştirme uygula
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import json
from pathlib import Path


def get_team_from_id(track_id: str) -> Optional[int]:
    """Track ID'sinden takımı döndürür (0=Left, 1=Right)."""
    if not track_id:
        return None
    if isinstance(track_id, str):
        if track_id.startswith("L"):
            return 0
        elif track_id.startswith("R"):
            return 1
    return None


@dataclass
class PassEvent:
    """Pas olayı veri yapısı."""
    start_frame: int
    end_frame: int
    sender_id: str          # "L3", "R5" gibi string ID
    receiver_id: str        # "L3", "R5" gibi string ID
    team_id: int            # 0=Left, 1=Right
    displacement: float     # Piksel cinsinden mesafe
    duration: float         # Saniye cinsinden süre
    start_time: float = 0.0
    end_time: float = 0.0
    pass_type: str = "short"  # "short", "medium", "long_ball"
    confidence: float = 1.0
    avg_ball_speed: float = 0.0
    direction_vector: List[float] = field(default_factory=lambda: [0.0, 0.0])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "team_id": self.team_id,
            "displacement": self.displacement,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "pass_type": self.pass_type,
            "confidence": self.confidence,
            "avg_ball_speed": self.avg_ball_speed,
            "direction_vector": self.direction_vector
        }


def classify_pass_type(displacement: float) -> str:
    """Pas tipini mesafeye göre belirler."""
    if displacement >= 500:
        return "long_ball"
    elif displacement >= 200:
        return "medium"
    else:
        return "short"


def calculate_confidence(segment: Dict[str, Any], duration: float) -> float:
    """
    Pas güvenilirliğini hesaplar.
    
    Faktörler:
    - Displacement: Daha uzun pas = daha güvenilir
    - Duration: Mantıklı süre = daha güvenilir
    - Speed: Normal hız aralığı = daha güvenilir
    """
    score = 0.5
    
    displacement = segment.get("displacement", 0)
    avg_speed = segment.get("average_speed", 0)
    
    # Mesafe puanı
    if displacement >= 50:
        score += 0.1
    if displacement >= 100:
        score += 0.1
    if displacement >= 200:
        score += 0.1
    
    # Hız puanı (normal aralıkta mı?)
    if 2.0 <= avg_speed <= 20.0:
        score += 0.1
    
    # Süre puanı
    if 0.2 <= duration <= 3.0:
        score += 0.1
    
    return min(1.0, score)


def detect_passes_from_segments( 
    segments: List[Dict[str, Any]],
    fps: float = 25.0,
    min_displacement: float = 19.9,  # 30'dan 15'e düşürüldü - kısa paslar için
    min_duration_frames: int = 3,
    max_duration_seconds: float = 5.0,
    same_team_only: bool = True
) -> List[PassEvent]:
    """
    Ball segments'tan pas olaylarını çıkarır.
    
    Args:
        segments: ball_motion_seg.py'den gelen segment listesi
        fps: Video FPS değeri
        min_displacement: Minimum pas mesafesi (piksel)
        min_duration_frames: Minimum pas süresi (frame)
        max_duration_seconds: Maximum pas süresi (saniye)
        same_team_only: Sadece aynı takım içi pasları al
        
    Returns:
        PassEvent listesi
    """
    passes: List[PassEvent] = []
    
    for seg in segments:
        # 1. Sadece "pass" tipindeki segmentleri al
        if seg.get("segment_type") != "pass":
            continue
        
        start_owner = seg.get("start_owner")
        end_owner = seg.get("end_owner")
        
        # 2. Her iki tarafta da sahip olmalı
        if not start_owner or not end_owner:
            continue
        
        # 3. Farklı oyuncular olmalı (zaten segment_type="pass" ise öyle ama double-check)
        if start_owner == end_owner:
            continue
        
        # 4. Takım kontrolü
        sender_team = get_team_from_id(start_owner)
        receiver_team = get_team_from_id(end_owner)
        
        if sender_team is None or receiver_team is None:
            continue
        
        # Intercept edilmiş pasları da kabul et (segment_type="pass" + intercepted=True)
        is_intercepted = seg.get("intercepted", False)
        
        if same_team_only and sender_team != receiver_team and not is_intercepted:
            # Farklı takımlar arası geçiş = top kaybı, pas değil
            # AMA: intercepted=True ise yüksek hızlı pas, kabul et
            continue
        
        # 5. Frame ve süre bilgileri
        start_frame = seg.get("start_time", 0)  # Segment'te frame olarak tutulmuş
        end_frame = seg.get("end_time", 0)
        
        duration_frames = end_frame - start_frame
        duration_seconds = duration_frames / fps
        
        # 6. Filtreler
        displacement = seg.get("displacement", 0)
        
        if displacement < min_displacement:
            continue
        
        if duration_frames < min_duration_frames:
            continue
        
        # 6.5 ÇALIM/ID DEĞİŞİKLİĞİ FİLTRESİ
        # Aynı takımdan çok kısa mesafeli ve kısa süreli "pas"lar muhtemelen
        # çalım sırasında ID değişikliğidir, pas değil
        DRIBBLE_DISPLACEMENT_THRESHOLD = 80.0  # px
        DRIBBLE_DURATION_THRESHOLD = 0.35  # saniye (~8-9 frame)
        
        if sender_team == receiver_team:
            # Aynı takım içi - ID değişikliği olabilir mi kontrol et
            if displacement < DRIBBLE_DISPLACEMENT_THRESHOLD and duration_seconds < DRIBBLE_DURATION_THRESHOLD:
                # Çok kısa ve yakın - muhtemelen çalım sırasında ID kaybı
                continue
        
        if duration_seconds > max_duration_seconds:
            continue
        
        # 7. Pas tipini belirle
        pass_type = classify_pass_type(displacement)
        
        # 8. Güvenilirlik hesapla
        confidence = calculate_confidence(seg, duration_seconds)
        
        # 9. PassEvent oluştur
        pass_event = PassEvent(
            start_frame=start_frame,
            end_frame=end_frame,
            sender_id=start_owner,
            receiver_id=end_owner,
            team_id=sender_team,
            displacement=round(displacement, 2),
            duration=round(duration_seconds, 3),
            start_time=round(start_frame / fps, 3),
            end_time=round(end_frame / fps, 3),
            pass_type=pass_type,
            confidence=round(confidence, 3),
            avg_ball_speed=round(seg.get("average_speed", 0), 2),
            direction_vector=seg.get("direction_vector", [0.0, 0.0])
        )
        
        passes.append(pass_event)
    
    print(f"  ✓ {len(passes)} pas tespit edildi (segment-based)")
    return passes


def detect_passes(
    frames_data: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    fps: float = 25.0,
    min_displacement: float = 15.0,  # 30'dan 15'e düşürüldü - kısa paslar için
    min_duration_frames: int = 3
) -> List[PassEvent]:
    """
    Ana pas algılama fonksiyonu.
    
    events.py ile uyumlu interface.
    
    Args:
        frames_data: Frame kayıtları (kullanılmıyor ama interface uyumu için)
        segments: Ball motion segments
        fps: Video FPS
        min_displacement: Minimum pas mesafesi
        min_duration_frames: Minimum pas süresi
        
    Returns:
        PassEvent listesi
    """
    return detect_passes_from_segments(
        segments=segments,
        fps=fps,
        min_displacement=min_displacement,
        min_duration_frames=min_duration_frames
    )


# --- STANDALONE KULLANIM ---
def main():
    """Standalone test için."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pass Detection v3 (Segment-Based)")
    parser.add_argument("--segments", default="output/ball_segments.json", help="Ball segments dosyası")
    parser.add_argument("--output", default="output/passes_v3.json", help="Çıktı dosyası")
    parser.add_argument("--fps", type=float, default=25.0, help="Video FPS")
    parser.add_argument("--min-displacement", type=float, default=30.0, help="Minimum pas mesafesi")
    
    args = parser.parse_args()
    
    # Segments yükle
    with open(args.segments, 'r') as f:
        segments = json.load(f)
    
    print(f"📂 {len(segments)} segment yüklendi")
    
    # Pasları tespit et
    passes = detect_passes_from_segments(
        segments=segments,
        fps=args.fps,
        min_displacement=args.min_displacement
    )
    
    # Kaydet
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([p.to_dict() for p in passes], f, indent=2, ensure_ascii=False)
    
    print(f"💾 {len(passes)} pas kaydedildi: {output_path}")
    
    # Özet
    if passes:
        print("\n📊 Pas Özeti:")
        for p in passes:
            print(f"  [{p.start_frame}-{p.end_frame}] {p.sender_id} → {p.receiver_id} "
                  f"({p.pass_type}, {p.displacement}px, {p.duration}s)")


if __name__ == "__main__":
    main()
