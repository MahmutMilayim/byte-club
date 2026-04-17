"""
K5 - Shot + Goal Detection (Complete Pipeline)
Direct frame-level shot detection focusing on isolated >300px instantaneous velocities
"""
import json
import math
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2

def get_ball_pos_from_frame(frame: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Extract ball center (x, y) from frame tracks."""
    for track in frame.get("tracks", []):
        if track.get("cls", "").lower() == "ball":
            bbox = track.get("bbox", [])
            if len(bbox) == 4:
                x = (float(bbox[0]) + float(bbox[2])) / 2.0
                y = (float(bbox[1]) + float(bbox[3])) / 2.0
                return (x, y)
    return None

def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

def detect_shots_from_frames(frames_data: List[Dict[str, Any]], fps: float = 25.0) -> List[Dict[str, Any]]:
    """
    Detect shots by finding ISOLATED high-velocity kicks >300px.
    
    Key insight from youtube_clip analysis:
    - Frames 70-71 (pass): consecutive >300px frames
    - Frame 79 (shot): isolated >300px frame
    
    So a shot is an isolated high-velocity event, not part of a sequence.
    """
    shots = []
    sorted_frames = sorted(frames_data, key=lambda f: int(f["frame_idx"]))
    
    # First pass: identify all high-velocity frames (>300px, player has ball)
    high_velocity_frames = []
    
    for i in range(1, len(sorted_frames)):
        curr_pos = get_ball_pos_from_frame(sorted_frames[i])
        prev_pos = get_ball_pos_from_frame(sorted_frames[i-1])
        prev_owner = sorted_frames[i-1].get("ball_owner")
        
        if prev_pos and curr_pos and prev_owner is not None:
            frame_velocity = calculate_distance(prev_pos, curr_pos)
            
            if frame_velocity > 300:
                prev_frame_num = int(sorted_frames[i-1]["frame_idx"])
                curr_frame_num = int(sorted_frames[i]["frame_idx"])
                high_velocity_frames.append({
                    "idx": i,
                    "prev_frame": prev_frame_num,
                    "curr_frame": curr_frame_num,
                    "velocity": frame_velocity,
                    "prev_pos": prev_pos,
                    "curr_pos": curr_pos,
                    "owner": prev_owner
                })
    
    # Second pass: identify isolated high-velocity events (not part of consecutive sequences)
    for j, hv in enumerate(high_velocity_frames):
        # Check if consecutive with neighbors
        is_next_to_previous = (j > 0 and high_velocity_frames[j-1]["curr_frame"] == hv["prev_frame"])
        is_next_to_next = (j + 1 < len(high_velocity_frames) and hv["curr_frame"] == high_velocity_frames[j+1]["prev_frame"])
        
        # An isolated shot is NOT surrounded by other high-velocity events
        is_isolated = not is_next_to_previous and not is_next_to_next
        
        if not is_isolated:
            continue  # Skip sequences (passes)
        
        # Now trace this shot forward to find where the ball ends up
        start_frame = hv["prev_frame"]
        start_pos = hv["prev_pos"]
        curr_pos = hv["curr_pos"]
        
        # Look ahead up to 40 frames to find final position
        frame_idx = hv["idx"]
        end_pos = curr_pos
        end_frame = hv["curr_frame"]
        
        for k in range(frame_idx, min(frame_idx + 40, len(sorted_frames))):
            next_frame = sorted_frames[k]
            next_pos = get_ball_pos_from_frame(next_frame)
            next_frame_num = int(next_frame["frame_idx"])
            
            if next_pos:
                end_pos = next_pos
                end_frame = next_frame_num
                
                # Stop if ownership changes
                if next_frame.get("ball_owner") != hv["owner"]:
                    break
        
        # Calculate trajectory
        total_displacement = calculate_distance(curr_pos, end_pos)
        if total_displacement > 0:
            direction_x = (end_pos[0] - curr_pos[0]) / total_displacement
        else:
            direction_x = 0
        
        # Shot must be directed towards goal
        is_towards_goal = abs(direction_x) > 0.3
        
        if is_towards_goal and total_displacement > 150:
            target_side = "RIGHT" if direction_x > 0 else "LEFT"
            
            shots.append({
                "event_type": "SHOT",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": start_frame / fps,
                "end_time_s": end_frame / fps,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "target_side": target_side,
                "speed": round(hv["velocity"], 2),
                "displacement": round(total_displacement, 2),
                "confidence_raw": round(min(1.0, (hv["velocity"] / 400.0)), 3)
            })
    
    return shots

# ... [rest of the functions remain the same as original] ...
