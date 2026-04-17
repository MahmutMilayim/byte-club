"""
Feature Extractor: FrameRecord → FrameFeatures

frames.jsonl'dan ham track verilerini okur, event detection için
işlenmiş feature'lara dönüştürür. Top hızı hesaplanır, oyuncu 
pozisyonları normalize edilir.

ID Formatı:
- Oyuncular: L1-L11 (sol takım), R1-R11 (sağ takım)
- Top: "ball"
"""

from __future__ import annotations

import json
from typing import List, Optional, Dict
from pathlib import Path

from schemas.types import (
    FrameRecord, 
    Track, 
    FrameFeatures, 
    PlayerFeature, 
    BallFeature,
    get_team_from_id
)
from ml_pipeline.detection.ball_interpolation import interpolate_ball_positions


def load_frames_jsonl(jsonl_path: str | Path) -> List[FrameRecord]:
    """
    frames.jsonl dosyasını okur ve FrameRecord listesi döndürür.
    
    Args:
        jsonl_path: .jsonl dosya yolu
        
    Returns:
        FrameRecord nesnelerinin listesi (frame_idx'e göre sıralı)
    """
    frames = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Track listesini yeniden oluştur
            tracks = [
                Track(
                    track_id=t['track_id'],
                    cls=t['cls'],
                    bbox=tuple(t['bbox']),
                    t=t['t'],
                    score=t.get('score')
                )
                for t in data['tracks']
            ]
            
            frame = FrameRecord(
                frame_idx=data['frame_idx'],
                t=data['t'],
                tracks=tracks
            )
            frames.append(frame)
    
    # Frame index'e göre sırala
    frames.sort(key=lambda f: f.frame_idx)
    return frames


def bbox_center(bbox: tuple) -> tuple[float, float]:
    """Bounding box merkezini hesaplar."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_velocity(
    pos_prev: tuple[float, float],
    pos_curr: tuple[float, float],
    dt: float
) -> tuple[float, float]:
    """
    İki pozisyon arasındaki hızı hesaplar (piksel/saniye).
    
    Args:
        pos_prev: Önceki (x, y)
        pos_curr: Şimdiki (x, y)
        dt: Zaman farkı (saniye)
        
    Returns:
        (vx, vy) hız bileşenleri
    """
    if dt <= 0:
        return (0.0, 0.0)
    
    vx = (pos_curr[0] - pos_prev[0]) / dt
    vy = (pos_curr[1] - pos_prev[1]) / dt
    return (vx, vy)


def extract_features(
    frames: List[FrameRecord],
    team_mapping: Optional[Dict[str, int]] = None
) -> List[FrameFeatures]:
    """
    Ham FrameRecord listesinden FrameFeatures listesi üretir.
    Top hızı frame-to-frame hesaplanır.
    
    STEP 1: Ball interpolation (missing detection'ları doldur)
    STEP 2: Feature extraction with velocity calculation
    
    Args:
        frames: FrameRecord listesi (sıralı)
        team_mapping: track_id -> team (0 veya 1) eşlemesi (opsiyonel).
                      Yeni sistemde takım ID'den otomatik çıkarılır.
        
    Returns:
        FrameFeatures listesi
    """
    # STEP 1: Interpolate missing ball detections with median smoothing
    # Median smoothing removes anomalous position spikes before interpolation
    frames = interpolate_ball_positions(
        frames, 
        max_gap=40,
        apply_smoothing=True,      # Enable spike detection
        smoothing_window=5,         # 5-frame median window
        smoothing_threshold=50.0    # 50px deviation threshold
    )
    
    # STEP 2: Extract features
    features_list = []
    
    # Önceki frame'deki pozisyonları sakla (hız hesabı için)
    prev_positions: Dict[str, tuple[float, float]] = {}
    prev_t: Optional[float] = None
    
    for frame in frames:
        players: List[PlayerFeature] = []
        ball_feature: Optional[BallFeature] = None
        
        for track in frame.tracks:
            cx, cy = bbox_center(track.bbox)
            
            if track.cls == 'ball' or track.track_id == 'ball':
                # Top için hız hesapla
                vx, vy = 0.0, 0.0
                if 'ball' in prev_positions and prev_t is not None:
                    dt = frame.t - prev_t
                    prev_pos = prev_positions['ball']
                    vx, vy = compute_velocity(prev_pos, (cx, cy), dt)
                
                speed = (vx**2 + vy**2) ** 0.5
                
                ball_feature = BallFeature(
                    x=cx,
                    y=cy,
                    speed=speed,
                    vx=vx,
                    vy=vy
                )
                
                # Top pozisyonunu sakla
                prev_positions['ball'] = (cx, cy)
                
            elif track.cls == 'player':
                # Oyuncu için hız hesapla (opsiyonel)
                vx, vy = None, None
                track_id_str = str(track.track_id)
                if track_id_str in prev_positions and prev_t is not None:
                    dt = frame.t - prev_t
                    prev_pos = prev_positions[track_id_str]
                    vx, vy = compute_velocity(prev_pos, (cx, cy), dt)
                
                # PlayerFeature oluştur - takım artık ID'den otomatik çıkarılıyor
                player_feat = PlayerFeature(
                    track_id=track.track_id,  # L3, R5 gibi string ID
                    x=cx,
                    y=cy,
                    vx=vx,
                    vy=vy
                )
                players.append(player_feat)
                
                # Oyuncu pozisyonunu sakla
                prev_positions[track_id_str] = (cx, cy)
        
        # FrameFeatures oluştur
        frame_feat = FrameFeatures(
            frame_idx=frame.frame_idx,
            t=frame.t,
            players=players,
            ball=ball_feature
        )
        features_list.append(frame_feat)
        
        prev_t = frame.t
    
    return features_list


def extract_features_from_file(
    jsonl_path: str | Path,
    team_mapping: Optional[Dict[int, int]] = None
) -> List[FrameFeatures]:
    """
    frames.jsonl dosyasından doğrudan feature extraction yapar.
    
    Args:
        jsonl_path: frames.jsonl dosya yolu
        team_mapping: track_id -> team mapping (opsiyonel)
        
    Returns:
        FrameFeatures listesi
    """
    frames = load_frames_jsonl(jsonl_path)
    return extract_features(frames, team_mapping)
