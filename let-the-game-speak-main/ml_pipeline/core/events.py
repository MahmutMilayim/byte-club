"""
Events Module: Football Event Detection
Detects passes, shots, and goals from tracking data.

v3: Segment-based pass detection kullanır.
"""
import json
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from schemas.types import FrameRecord
from ml_pipeline.detection.pass_detection import detect_passes, PassEvent
from ml_pipeline.detection.shot_detection import ShotDetector, ShotEvent
from ml_pipeline.detection.ball_motion_seg import process_motion_segmentation


# =============================================================================
# ŞUT ZAMANI DÜZELTME - Top şutçudan yüksek hızla ayrılıp geri gelmiyorsa şut
# =============================================================================
# Şut detection algoritması bazen şutu erken tespit edebilir.
# Bu fonksiyon:
# 1. Tespit edilen şut frame'indeki topa en yakın oyuncuyu bulur (şutçu)
# 2. Sonraki frame'lerde şutçuyu ve topu takip eder
# 3. Top şutçudan yüksek hızla ayrılıp geri gelmiyorsa = gerçek şut
# =============================================================================

MIN_DEPARTURE_SPEED = 800    # px/s - Topun şutçudan ayrılma hızı (şut için minimum)
BALL_RETURN_THRESHOLD = 200  # px - Top şutçuya bu kadar yaklaşırsa "geri geldi" sayılır
DEPARTURE_DISTANCE = 150     # px - Top şutçudan bu kadar uzaklaşınca "ayrıldı" sayılır
CHECK_FRAMES_AFTER = 15      # frame - Ayrılmadan sonra kaç frame geri gelmediğini kontrol et


def refine_shot_time_from_frames(shot: Dict, frames_jsonl_path: str, fps: float) -> Dict:
    """
    Şut zamanını, topun şutçudan yüksek hızla ayrılıp geri gelmediği anı bularak düzeltir.
    
    Mantık:
    1. Her frame'de şutçunun ve topun konumunu takip et
    2. Top şutçudan yüksek hızla (>MIN_DEPARTURE_SPEED) uzaklaşıyor mu?
    3. Top sonraki CHECK_FRAMES_AFTER frame içinde geri geliyor mu?
    4. Geri gelmiyorsa = gerçek şut
    
    Args:
        shot: Şut verisi (frame_idx, time, ...)
        frames_jsonl_path: test_frames.jsonl dosyasının yolu
        fps: Video FPS değeri
        
    Returns:
        Düzeltilmiş şut verisi
    """
    original_time = shot.get("time", 0)
    original_frame = shot.get("frame_idx", 0)
    
    # Frames data'yı yükle
    frames_data = []
    try:
        with open(frames_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    frames_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"   ⚠️ Frames dosyası bulunamadı: {frames_jsonl_path}")
        return shot
    
    if not frames_data:
        return shot
    
    # Frame'leri index'e göre sırala ve map oluştur
    frames_data.sort(key=lambda x: x.get("frame_idx", 0))
    frame_map = {f.get("frame_idx"): f for f in frames_data}
    
    # Yardımcı fonksiyonlar
    def get_ball_pos(frame_data: dict):
        tracks = frame_data.get("tracks", [])
        for track in tracks:
            if track.get("track_id") == "ball":
                bbox = track.get("bbox", [])
                if len(bbox) >= 4:
                    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        return None
    
    def get_player_pos(frame_data: dict, player_id: str):
        tracks = frame_data.get("tracks", [])
        for track in tracks:
            if track.get("track_id") == player_id:
                bbox = track.get("bbox", [])
                if len(bbox) >= 4:
                    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        return None
    
    def get_closest_player(frame_data: dict, ball_pos: tuple):
        tracks = frame_data.get("tracks", [])
        closest = None
        min_dist = float('inf')
        for track in tracks:
            track_id = track.get("track_id", "")
            if track_id == "ball":
                continue
            bbox = track.get("bbox", [])
            if len(bbox) >= 4:
                px = (bbox[0] + bbox[2]) / 2
                py = (bbox[1] + bbox[3]) / 2
                dist = math.sqrt((px - ball_pos[0])**2 + (py - ball_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest = {"id": track_id, "x": px, "y": py, "dist": dist}
        return closest
    
    def ball_returns_to_player(start_frame: int, player_id: str, num_frames: int) -> bool:
        """Top belirtilen frame'lerden sonra oyuncuya geri dönüyor mu?"""
        for i in range(1, num_frames + 1):
            check_frame = start_frame + i
            if check_frame not in frame_map:
                continue
            
            fd = frame_map[check_frame]
            ball = get_ball_pos(fd)
            player = get_player_pos(fd, player_id)
            
            if ball and player:
                dist = math.sqrt((ball[0] - player[0])**2 + (ball[1] - player[1])**2)
                if dist < BALL_RETURN_THRESHOLD:
                    return True  # Top geri geldi
        return False  # Top geri gelmedi
    
    print(f"   📍 Orijinal şut tespiti: frame={original_frame} ({original_time:.2f}s)")
    
    # Her frame'de şutçu ve top takibi
    prev_ball_pos = None
    current_shooter = None  # Topa en yakın oyuncu (potansiyel şutçu)
    
    for fd in frames_data:
        frame_idx = fd.get("frame_idx", 0)
        
        # Sadece orijinal şut frame'inden sonrasına bak
        if frame_idx < original_frame:
            continue
        
        ball_pos = get_ball_pos(fd)
        if not ball_pos:
            prev_ball_pos = None
            continue
        
        # Topa en yakın oyuncuyu bul
        closest = get_closest_player(fd, ball_pos)
        
        if closest and closest["dist"] < 100:  # Oyuncu topa yakınsa (100px)
            current_shooter = closest
        
        if prev_ball_pos is not None and current_shooter is not None:
            # Top hızını hesapla
            dx = ball_pos[0] - prev_ball_pos[0]
            dy = ball_pos[1] - prev_ball_pos[1]
            ball_speed = math.sqrt(dx*dx + dy*dy) * fps  # px/s
            
            # Şutçudan uzaklığı hesapla
            shooter_pos = get_player_pos(fd, current_shooter["id"])
            if shooter_pos:
                dist_from_shooter = math.sqrt(
                    (ball_pos[0] - shooter_pos[0])**2 + 
                    (ball_pos[1] - shooter_pos[1])**2
                )
                
                # Koşullar:
                # 1. Top yüksek hızda hareket ediyor (>MIN_DEPARTURE_SPEED)
                # 2. Top şutçudan uzaklaşmış (>DEPARTURE_DISTANCE)
                # 3. Top geri gelmiyor
                if (ball_speed >= MIN_DEPARTURE_SPEED and 
                    dist_from_shooter >= DEPARTURE_DISTANCE):
                    
                    # Geri dönüş kontrolü
                    if not ball_returns_to_player(frame_idx, current_shooter["id"], CHECK_FRAMES_AFTER):
                        # Bu gerçek şut!
                        shot_frame = frame_idx - 1  # Hızlanmanın başladığı frame
                        new_time = round(shot_frame / fps, 2)
                        
                        print(f"   🎯 Gerçek şut bulundu: frame={shot_frame} ({new_time:.2f}s)")
                        print(f"      Şutçu: {current_shooter['id']}")
                        print(f"      Top hızı: {ball_speed:.0f} px/s")
                        print(f"      Şutçudan uzaklık: {dist_from_shooter:.0f} px")
                        print(f"      ✅ Top geri gelmedi ({CHECK_FRAMES_AFTER} frame kontrol edildi)")
                        
                        shot["frame_idx"] = shot_frame
                        shot["time"] = new_time
                        shot["speed_px_s"] = round(ball_speed, 1)
                        shot["shooter_id"] = current_shooter["id"]
                        shot["_refined"] = True
                        return shot
        
        prev_ball_pos = ball_pos
    
    # Şut bulunamadı - orijinal değerleri koru
    print(f"   ℹ️ Uygun şut bulunamadı, orijinal zaman korunuyor")
    return shot


def detect_events(frames: List[FrameRecord], metadata: Dict, 
                  video_path: Optional[str] = None,
                  output_dir: str = "./output") -> Dict[str, List]:
    """
    Detect football events (passes, shots, goals) from tracking data.
    
    Args:
        frames: List of FrameRecord objects from vision pipeline
        metadata: Video metadata (fps, width, height, etc.)
        video_path: Optional path to video for goal zone detection
        output_dir: Directory for intermediate outputs
        
    Returns:
        Dictionary with event lists:
        {
            "passes": [PassEvent, ...],
            "shots": [ShotEvent, ...],
            "goals": [ShotEvent, ...],  # Subset of shots where is_goal=True
            "segments": [dict, ...]      # Ball motion segments
        }
    """
    print("🎯 Event detection started...")
    
    fps = metadata.get("fps", 25.0)
    width = metadata.get("width", 1920)
    height = metadata.get("height", 1080)
    
    # Convert frames to dict format (for compatibility with existing detectors)
    frames_data = []
    for fr in frames:
        frame_dict = {
            "frame_idx": fr.frame_idx,
            "t": fr.t,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "cls": t.cls,
                    "bbox": list(t.bbox),
                    "t": t.t,
                    "score": t.score,
                    "team": t.team,
                    "field_x": getattr(t, 'field_x', None),
                    "field_y": getattr(t, 'field_y', None),
                    "field_z": getattr(t, 'field_z', None)
                }
                for t in fr.tracks
            ],
            "ball_owner": fr.ball_owner,
            "ball_owner_team": fr.ball_owner_team
        }
        frames_data.append(frame_dict)
    
    # 1. Ball Motion Segmentation
    print("  📊 Analyzing ball motion segments...")
    segments_file = Path(output_dir) / "ball_segments.json"
    
    # Save frames to JSONL temporarily
    temp_jsonl = Path(output_dir) / "temp_frames.jsonl"
    with open(temp_jsonl, 'w', encoding='utf-8') as f:
        for fd in frames_data:
            f.write(json.dumps(fd, ensure_ascii=False) + "\n")
    
    # Run segmentation
    process_motion_segmentation(str(temp_jsonl), str(segments_file))
    
    # Load segments
    with open(segments_file) as f:
        segments = json.load(f)
    
    print(f"    ✓ Found {len(segments)} ball motion segments")
    
    # 2. Pass Detection (Segment-based)
    print("  ⚽ Detecting passes (segment-based)...")
    passes = detect_passes(
        frames_data=frames_data,
        segments=segments,
        fps=fps,
        min_displacement=15.0,  # 30'dan 15'e düşürüldü - kısa paslar için
        min_duration_frames=3
    )
    
    # 3. Shot Detection
    print("  🎯 Detecting shots...")
    
    # Build frame map
    frame_map = {int(f["frame_idx"]): f for f in frames_data}
    
    detector = ShotDetector(
        width=width,
        height=height,
        fps=fps,
        video_path=video_path
    )
    
    shots = detector.detect(segments, frames_data, frame_map=frame_map)
    
    goals = [s for s in shots if s.is_goal]
    print(f"    ✓ Found {len(shots)} shots ({len(goals)} goals)")
    
    # Summary
    print(f"\n📊 Event Detection Summary:")
    print(f"   Segments: {len(segments)}")
    print(f"   Passes:   {len(passes)}")
    print(f"   Shots:    {len(shots)}")
    print(f"   Goals:    {len(goals)}")
    
    return {
        "passes": passes,
        "shots": shots,
        "goals": goals,
        "segments": segments
    }


def save_events(events: Dict[str, List], output_dir: str = "./output", video_name: str = "output",
                fps: float = 25.0):
    """
    Save detected events to JSON files.
    
    Args:
        events: Events dictionary from detect_events()
        output_dir: Output directory
        video_name: Base name for output files
        fps: Video FPS for shot time refinement
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    base_name = Path(video_name).stem
    
    # Save passes
    passes_file = output_dir / f"{base_name}_passes.json"
    if events.get("passes"):
        with open(passes_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(p) for p in events["passes"]], f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {passes_file}")
    elif passes_file.exists():
        passes_file.unlink()
        print(f"🗑️ Removed old: {passes_file}")
    
    # Save shots (or remove old file if no shots detected)
    # ŞUT ZAMANI DÜZELTME: Önce kaydet, sonra frames_data'dan düzelt
    shots_file = output_dir / f"{base_name}_shots.json"
    frames_jsonl = output_dir / f"{base_name}_frames.jsonl"
    
    if events.get("shots"):
        # Önce ham veriyi kaydet
        shots_data = [asdict(s) for s in events["shots"]]
        
        # Şut zamanlarını frames_data'dan düzelt
        if frames_jsonl.exists():
            print(f"🔄 Şut zamanları düzeltiliyor (frames_data'dan)...")
            refined_shots = []
            for shot in shots_data:
                refined_shot = refine_shot_time_from_frames(shot, str(frames_jsonl), fps)
                refined_shots.append(refined_shot)
            shots_data = refined_shots
        
        with open(shots_file, 'w', encoding='utf-8') as f:
            json.dump(shots_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {shots_file}")
    elif shots_file.exists():
        # No shots in this video - remove old file from previous video
        shots_file.unlink()
        print(f"🗑️ Removed old: {shots_file} (no shots in this video)")
    
    # Save segments
    segments_file = output_dir / f"{base_name}_segments.json"
    if events.get("segments"):
        with open(segments_file, 'w', encoding='utf-8') as f:
            json.dump(events["segments"], f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {segments_file}")
    elif segments_file.exists():
        segments_file.unlink()
        print(f"🗑️ Removed old: {segments_file}")
    
    # Save summary
    summary = {
        "total_passes": len(events.get("passes", [])),
        "total_shots": len(events.get("shots", [])),
        "total_goals": len(events.get("goals", [])),
        "total_segments": len(events.get("segments", [])),
        "pass_breakdown": _breakdown_passes(events.get("passes", [])),
        "shot_breakdown": _breakdown_shots(events.get("shots", []))
    }
    
    summary_file = output_dir / f"{base_name}_event_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {summary_file}")


def _breakdown_passes(passes: List) -> Dict:
    """Breakdown passes by type."""
    breakdown = {"short": 0, "medium": 0, "long_ball": 0}
    for p in passes:
        ptype = p.pass_type if hasattr(p, 'pass_type') else "short"
        breakdown[ptype] = breakdown.get(ptype, 0) + 1
    return breakdown


def _breakdown_shots(shots: List) -> Dict:
    """Breakdown shots by outcome."""
    breakdown = {"goals": 0, "missed": 0, "by_team": {0: 0, 1: 0}}
    for s in shots:
        if s.is_goal:
            breakdown["goals"] += 1
        else:
            breakdown["missed"] += 1
        
        team = s.shooter_team
        if team is not None:
            breakdown["by_team"][team] = breakdown["by_team"].get(team, 0) + 1
    
    return breakdown