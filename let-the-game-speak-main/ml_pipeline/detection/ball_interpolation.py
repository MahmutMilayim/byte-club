"""
Ball position interpolation and smoothing for missing detections.

ByteTrack keeps tracks up to track_buffer frames without detection,
but sometimes ball disappears for 2-5 frames due to YOLO confidence.
This module provides:
1. Median smoothing to remove position spikes
2. Linear interpolation to fill short gaps
"""
from typing import List, Optional
import numpy as np
from schemas.types import FrameRecord, Track


def interpolate_ball_positions(
    frames: List[FrameRecord], 
    max_gap: int = 5,
    apply_smoothing: bool = False,
    smoothing_window: int = 5,
    smoothing_threshold: float = 50.0
) -> List[FrameRecord]:
    """
    Fill missing ball detections with linear interpolation and optional smoothing.
    
    Args:
        frames: List of FrameRecords from ByteTrack
        max_gap: Maximum gap size (frames) to interpolate
        apply_smoothing: If True, apply median smoothing to remove position spikes
        smoothing_window: Window size for median filter (must be odd)
        smoothing_threshold: Distance threshold (pixels) to detect spikes
        
    Returns:
        Modified frames list with interpolated/smoothed ball positions
    """
    # Step 1: Apply median smoothing if requested
    if apply_smoothing:
        frames = _apply_median_smoothing(frames, smoothing_window, smoothing_threshold)
    
    # Step 2: Find all frames with ball detection
    ball_frames = [(i, f) for i, f in enumerate(frames) if _has_ball(f)]
    
    if len(ball_frames) < 2:
        return frames  # Not enough data for interpolation
    
    # Process gaps between consecutive ball detections
    for idx in range(len(ball_frames) - 1):
        i1, f1 = ball_frames[idx]
        i2, f2 = ball_frames[idx + 1]
        gap_size = i2 - i1 - 1
        if gap_size == 0 or gap_size > max_gap:
            continue  # No gap or gap too large
        ball1 = _get_ball_track(f1)
        ball2 = _get_ball_track(f2)
        if ball1 is None or ball2 is None:
            continue
        for gap_idx in range(1, gap_size + 1):
            frame_idx = i1 + gap_idx
            t_ratio = gap_idx / (gap_size + 1)
            interp_bbox = (
                ball1.bbox[0] + t_ratio * (ball2.bbox[0] - ball1.bbox[0]),
                ball1.bbox[1] + t_ratio * (ball2.bbox[1] - ball1.bbox[1]),
                ball1.bbox[2] + t_ratio * (ball2.bbox[2] - ball1.bbox[2]),
                ball1.bbox[3] + t_ratio * (ball2.bbox[3] - ball1.bbox[3]),
            )
            interp_t = ball1.t + t_ratio * (ball2.t - ball1.t)
            interp_score = None
            if ball1.score is not None and ball2.score is not None:
                interp_score = (ball1.score + ball2.score) / 2.0
            interp_track = Track(
                track_id=ball1.track_id,
                cls='ball',
                bbox=interp_bbox,
                t=interp_t,
                score=interp_score
            )
            # --- SAHİPLİK ATAMASI ---
            # Eğer frame'de oyuncular varsa ve topun hızı düşükse, en yakın oyuncuya sahiplik ata
            # (Şut/pas anında sahiplik atama)
            players = [tr for tr in frames[frame_idx].tracks if tr.cls == 'player']
            def bbox_center(b):
                return ((b[0]+b[2])/2, (b[1]+b[3])/2)
            ball_cx, ball_cy = bbox_center(interp_bbox)
            min_dist = float('inf')
            nearest_player = None
            for p in players:
                px, py = bbox_center(p.bbox)
                dist = ((ball_cx - px)**2 + (ball_cy - py)**2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_player = p
            # Topun interpolasyon hızını hesapla (frame arası mesafe)
            prev_cx, prev_cy = bbox_center(ball1.bbox)
            next_cx, next_cy = bbox_center(ball2.bbox)
            total_dist = ((next_cx - prev_cx)**2 + (next_cy - prev_cy)**2) ** 0.5
            speed = total_dist / (gap_size + 1)
            # Şut/pas mantığı: Eğer hız yüksekse sahiplik atama
            BALL_SPEED_THRESHOLD = 8.0  # BallOwnershipProcessor ile uyumlu olmalı
            if nearest_player and speed < BALL_SPEED_THRESHOLD and min_dist < 120:
                # Sahiplik ata
                frames[frame_idx].ball_owner = nearest_player.track_id
            else:
                frames[frame_idx].ball_owner = None
            frames[frame_idx].tracks.append(interp_track)
    
    return frames


def _has_ball(frame: FrameRecord) -> bool:
    """Check if frame has ball detection."""
    return any(t.cls == 'ball' for t in frame.tracks)


def _get_ball_track(frame: FrameRecord) -> Optional[Track]:
    """Get ball track from frame."""
    for track in frame.tracks:
        if track.cls == 'ball':
            return track
    return None


def _apply_median_smoothing(
    frames: List[FrameRecord],
    window_size: int,
    threshold: float
) -> List[FrameRecord]:
    """
    Apply median filter to remove position spikes.
    
    Detects anomalous ball positions that deviate significantly from
    their local neighborhood and replaces them with median position.
    
    Args:
        frames: Input frames with ball detections
        window_size: Median filter window size (centered)
        threshold: Distance threshold (pixels) to classify as spike
        
    Returns:
        Frames with smoothed ball positions
    """
    half_window = window_size // 2
    
    # Collect all ball positions for analysis
    ball_indices = [i for i, f in enumerate(frames) if _has_ball(f)]
    
    if len(ball_indices) < 3:
        return frames  # Need at least 3 points for meaningful median
    
    # Process each ball detection
    for idx in ball_indices:
        frame = frames[idx]
        ball = _get_ball_track(frame)
        if ball is None:
            continue
        
        # Define window boundaries
        window_indices = [
            i for i in ball_indices
            if idx - half_window <= i <= idx + half_window
        ]
        
        if len(window_indices) < 3:
            continue  # Not enough neighbors
        
        # Collect window positions (center of bbox)
        window_positions = []
        for i in window_indices:
            b = _get_ball_track(frames[i])
            if b:
                center_x = (b.bbox[0] + b.bbox[2]) / 2
                center_y = (b.bbox[1] + b.bbox[3]) / 2
                window_positions.append((center_x, center_y))
        
        if len(window_positions) < 3:
            continue
        
        # Calculate median position
        median_x = np.median([p[0] for p in window_positions])
        median_y = np.median([p[1] for p in window_positions])
        
        # Current ball center
        ball_center_x = (ball.bbox[0] + ball.bbox[2]) / 2
        ball_center_y = (ball.bbox[1] + ball.bbox[3]) / 2
        
        # Check if current position is a spike
        distance = np.sqrt(
            (ball_center_x - median_x) ** 2 + 
            (ball_center_y - median_y) ** 2
        )
        
        if distance > threshold:
            # Spike detected! Replace with median position
            # Preserve bbox dimensions
            width = ball.bbox[2] - ball.bbox[0]
            height = ball.bbox[3] - ball.bbox[1]
            
            new_bbox = (
                median_x - width / 2,
                median_y - height / 2,
                median_x + width / 2,
                median_y + height / 2
            )
            
            # Update track in-place
            ball.bbox = new_bbox
    
    return frames
