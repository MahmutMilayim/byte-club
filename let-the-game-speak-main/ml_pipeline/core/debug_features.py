"""
Debug/Inspection tools for FrameFeatures.

İlk N frame'i human-readable formatta yazdırır, 
feature extraction'ın doğru çalıştığını kontrol etmek için.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import List

from schemas.types import FrameFeatures


def print_features_sample(
    features: List[FrameFeatures],
    n: int = 3,
    indent: int = 2
) -> None:
    """
    İlk N frame'in feature'larını JSON formatında yazdırır.
    
    Args:
        features: FrameFeatures listesi
        n: Kaç frame yazdırılacak (default: 3)
        indent: JSON indent seviyesi
    """
    sample = features[:n]
    
    print(f"\n{'='*60}")
    print(f"Feature Sample: İlk {len(sample)} frame")
    print(f"{'='*60}\n")
    
    for i, frame_feat in enumerate(sample):
        print(f"--- Frame {frame_feat.frame_idx} (t={frame_feat.t:.2f}s) ---")
        
        # dataclass → dict → JSON
        frame_dict = asdict(frame_feat)
        print(json.dumps(frame_dict, indent=indent, ensure_ascii=False))
        print()


def summarize_features(features: List[FrameFeatures]) -> None:
    """
    Feature listesi hakkında özet istatistikler yazdırır.
    
    Args:
        features: FrameFeatures listesi
    """
    if not features:
        print("⚠️  Feature listesi boş.")
        return
    
    total_frames = len(features)
    frames_with_ball = sum(1 for f in features if f.ball is not None)
    total_players = sum(len(f.players) for f in features)
    avg_players_per_frame = total_players / total_frames if total_frames else 0
    
    # Top hızı istatistikleri
    ball_speeds = [f.ball.speed for f in features if f.ball is not None]
    avg_ball_speed = sum(ball_speeds) / len(ball_speeds) if ball_speeds else 0
    max_ball_speed = max(ball_speeds) if ball_speeds else 0
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Özeti")
    print(f"{'='*60}")
    print(f"Toplam frame sayısı        : {total_frames}")
    print(f"Top tespit edilen frame    : {frames_with_ball} ({frames_with_ball/total_frames*100:.1f}%)")
    print(f"Toplam oyuncu tespiti      : {total_players}")
    print(f"Frame başına ort. oyuncu   : {avg_players_per_frame:.1f}")
    print(f"Ortalama top hızı          : {avg_ball_speed:.1f} px/s")
    print(f"Maksimum top hızı          : {max_ball_speed:.1f} px/s")
    print(f"{'='*60}\n")


def export_features_json(
    features: List[FrameFeatures],
    output_path: str,
    indent: int = 4
) -> None:
    """
    Feature'ları JSON dosyasına yazar.
    
    Args:
        features: FrameFeatures listesi
        output_path: Çıktı dosya yolu
        indent: JSON indent seviyesi
    """
    data = [asdict(f) for f in features]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    print(f"✅ {len(features)} frame feature'ı {output_path} konumuna yazıldı.")
