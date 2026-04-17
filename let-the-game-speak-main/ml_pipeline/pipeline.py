"""
Main ML Pipeline: End-to-End Video Processing
Orchestrates vision, event detection, narrative, and speech synthesis.
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

from ml_pipeline.core.vision import detect_and_track
from ml_pipeline.core.events import detect_events, save_events
from ml_pipeline.core.hybrid_narrative import HybridNarrativeGenerator
from ml_pipeline.core.speech import SpeechGenerator


def run_pipeline(video_path: str, 
                 output_dir: str = "./output",
                 model_path: str = "./weights/last.pt",
                 enable_narrative: bool = False,
                 enable_speech: bool = False,
                 team_left: Optional[str] = None,
                 team_right: Optional[str] = None,
                 progress_callback=None) -> Dict[str, Any]:
    """
    Run the complete football commentary pipeline.
    
    Pipeline stages:
    1. Vision: Object detection + tracking + team classification
    1.5. 2D Calibration: Calculate field coordinates for all tracks
    2. Events: Pass/shot/goal detection from tracks (with 2D coords)
    3. 2D Field Video: Create visualization with event overlays
    4. Narrative: Turkish commentary generation (optional)
    5. Speech: Text-to-speech synthesis (optional)
    
    Args:
        video_path: Path to input video file
        output_dir: Directory for all outputs
        model_path: Path to YOLO model weights
        enable_narrative: Enable GPT-based narrative generation
        enable_speech: Enable TTS audio synthesis
        team_left: Optional left team name for commentary
        team_right: Optional right team name for commentary
        progress_callback: Optional callback function(stage, progress, details)
        
    Returns:
        Dictionary with pipeline results
    """
    
    def report_progress(stage: str, progress: int, details: str):
        """Report progress to callback if available"""
        if progress_callback:
            progress_callback(stage, progress, details)
        print(f"[{stage}] {progress}% - {details}")
    
    print("=" * 60)
    print("🎬 LET THE GAME SPEAK - Pipeline Started")
    print("=" * 60)
    print(f"📹 Input: {video_path}")
    print(f"📂 Output: {output_dir}")
    print()
    
    video_name = Path(video_path).name
    base_name = Path(video_path).stem
    frames_jsonl = Path(output_dir) / f"{base_name}_frames.jsonl"
    
    # Stage 1: Vision Processing
    print("▶️ STAGE 1: VISION PROCESSING")
    print("-" * 60)
    report_progress("vision", 5, "Starting YOLO detection and ByteTrack tracking...")
    
    frames, metadata = detect_and_track(
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir,
        progress_callback=lambda p, d: report_progress("vision", 5 + int(p * 0.30), d)
    )
    
    report_progress("vision", 35, f"Vision complete: {len(frames)} frames processed")
    print()
    
    # Stage 1.5: 2D Calibration - Calculate field coordinates BEFORE event detection
    print("▶️ STAGE 1.5: 2D FIELD CALIBRATION")
    print("-" * 60)
    report_progress("calibration", 40, "Running 2D field calibration...")
    
    # Initialize soccer_line processor for calibration
    import sys
    import os
    project_root = Path(__file__).parent.parent
    scripts_path = project_root / "scripts"
    soccer_line_path = project_root / "soccer_line"
    
    frames_jsonl_abs = Path(frames_jsonl).resolve()
    video_path_abs = Path(video_path).resolve()
    output_dir_abs = Path(output_dir).resolve()
    
    original_cwd = os.getcwd()
    calibration_success = False
    processor_temp = None
    
    try:
        os.chdir(soccer_line_path)
        sys.path.insert(0, str(scripts_path))
        sys.path.insert(0, str(soccer_line_path))
        
        from run_soccer_line import SoccerLineProcessor
        
        # Run calibration only (no video output)
        processor_temp = SoccerLineProcessor(device="cpu")
        processor_temp.process_video(
            input_path=str(video_path_abs),
            output_path=None,  # No video, just calibration
            sample_interval=1.0,
            frames_data=None
        )
        
        # Add 2D coordinates to frames
        print("📐 2D saha koordinatları hesaplanıyor...")
        if hasattr(processor_temp, 'last_calibrations') and processor_temp.last_calibrations:
            calibrations = processor_temp.last_calibrations
            calib_frames = processor_temp.last_calib_frames
            
            coords_added = 0
            for frame in frames:
                frame_num = frame.frame_idx - 1  # 0-indexed
                
                P = processor_temp.get_interpolated_P(frame_num, calibrations, calib_frames)
                if P is None:
                    continue
                
                for track in frame.tracks:
                    if len(track.bbox) < 4:
                        continue
                    
                    x1, y1, x2, y2 = track.bbox
                    cls_name = track.cls.lower() if track.cls else ''
                    
                    # Foot position for players, center for ball
                    foot_x = (x1 + x2) / 2
                    foot_y = y2 if 'ball' not in cls_name else (y1 + y2) / 2
                    
                    field_pos = processor_temp.pixel_to_field(P, foot_x, foot_y)
                    if field_pos is not None:
                        track.field_x = round(field_pos[0], 2)
                        track.field_y = round(field_pos[1], 2)
                        coords_added += 1
                        
                        if 'ball' in cls_name:
                            bbox_center_y = (y1 + y2) / 2
                            video_height = metadata.get('height', 720)
                            relative_y = bbox_center_y / video_height
                            if relative_y < 0.3:
                                track.field_z = round(2.44 * (0.3 - relative_y) / 0.3, 2)
                            else:
                                track.field_z = 0.0
            
            print(f"✅ 2D koordinatlar eklendi: {coords_added} track")
            calibration_success = True
        else:
            print("⚠️  Kalibrasyon bulunamadı, 2D koordinatlar eklenemedi")
            
    except Exception as e:
        print(f"⚠️  2D calibration failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(original_cwd)
    print()
    
    # Save frames to JSONL (with 2D coordinates)
    with open(frames_jsonl, 'w', encoding='utf-8') as f:
        for frame in frames:
            f.write(json.dumps(asdict(frame), ensure_ascii=False) + "\n")
    print(f"💾 Frames saved: {frames_jsonl}")
    print()
    
    # Stage 2: Event Detection (NOW with 2D coordinates!)
    print("▶️ STAGE 2: EVENT DETECTION")
    print("-" * 60)
    report_progress("events", 50, "Detecting passes, shots, and game events...")
    
    events = detect_events(
        frames=frames,
        metadata=metadata,
        video_path=video_path,
        output_dir=output_dir
    )
    
    # Save events (with shot time refinement)
    save_events(events, output_dir=output_dir, video_name=video_name, fps=metadata.get("fps", 25.0))
    report_progress("events", 60, f"Events detected: {len(events.get('segments', []))} segments, {len(events.get('passes', []))} passes")
    print()
    
    # Stage 2.5: 2D Field Video WITH Events
    print("▶️ STAGE 2.5: 2D FIELD VIDEO WITH EVENTS")
    print("-" * 60)
    try:
        os.chdir(soccer_line_path)
        
        from run_soccer_line import SoccerLineProcessor
        
        # Load frames data
        frames_data = {}
        with open(frames_jsonl_abs, 'r', encoding='utf-8') as f:
            for line in f:
                frame = json.loads(line)
                frames_data[frame['frame_idx']] = frame.get('tracks', [])
        
        # Convert events to dict format
        shots_list = events.get('shots', [])
        passes_list = events.get('passes', [])
        
        shots_dict_list = []
        for shot in shots_list:
            if hasattr(shot, '__dict__'):
                shot_dict = {
                    'frame_idx': getattr(shot, 'frame_idx', 0),
                    'time': getattr(shot, 'time', 0),
                    'shooter_id': getattr(shot, 'shooter_id', '?'),
                    'shooter_team': getattr(shot, 'shooter_team', None),
                    'target_side': getattr(shot, 'target_side', 'RIGHT'),
                    'is_goal': getattr(shot, 'is_goal', False),
                    'goal_frame': getattr(shot, 'goal_frame', None),
                    'start_pos': list(getattr(shot, 'start_pos', [0, 0])),
                    'end_pos': list(getattr(shot, 'end_pos', [0, 0])),
                }
                shots_dict_list.append(shot_dict)
            elif isinstance(shot, dict):
                shots_dict_list.append(shot)
        
        passes_dict_list = []
        for pas in passes_list:
            if hasattr(pas, '__dict__'):
                pas_dict = {
                    'frame': getattr(pas, 'start_frame', getattr(pas, 'frame', 0)),
                    'start_frame': getattr(pas, 'start_frame', 0),
                    'passer_id': getattr(pas, 'sender_id', getattr(pas, 'passer_id', '?')),
                    'receiver_id': getattr(pas, 'receiver_id', '?'),
                    'team': getattr(pas, 'team_id', ''),
                }
                passes_dict_list.append(pas_dict)
            elif isinstance(pas, dict):
                passes_dict_list.append(pas)
        
        events_dict = {
            'shots': shots_dict_list,
            'passes': passes_dict_list
        }
        
        print(f"📊 Events for 2D video: {len(shots_dict_list)} shots, {len(passes_dict_list)} passes")
        
        # Create 2D field video
        processor = SoccerLineProcessor(device="cpu")
        output_2d_events_path = output_dir_abs / f"{base_name}_2d_field_events.mp4"
        success = processor.process_video(
            input_path=str(video_path_abs),
            output_path=str(output_2d_events_path),
            sample_interval=1.0,
            frames_data=frames_data,
            events=events_dict,
            only_2d=True
        )
        
        os.chdir(original_cwd)
        if success and output_2d_events_path.exists():
            print(f"✅ 2D Field video with events: {output_2d_events_path}")
        else:
            print(f"⚠️  2D Field video oluşturulamadı: {output_2d_events_path}")
    except Exception as e:
        print(f"⚠️  2D Field video with events failed: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_cwd)
    print()
    
    # Stage 3: Timed Narrative Generation (Optional)
    narrative_data = None
    timed_commentaries = None
    video_duration = metadata.get('duration', len(frames) / metadata.get('fps', 25.0))
    
    if enable_narrative:
        print("▶️ STAGE 3: TIMED NARRATIVE GENERATION")
        print("-" * 60)
        report_progress("narrative", 65, "Generating Turkish commentary...")
        
        try:
            # Build events list from shots and passes
            events = []
            
            # Load raw segments first to get shot end_time (for goal timing)
            segments_path = Path(output_dir) / f"{base_name}_segments.json"
            shot_segments = {}
            if segments_path.exists():
                with open(segments_path, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                    for seg in segments:
                        if seg.get('segment_type') == 'shot_candidate':
                            # Map by start_time (frame) to end_time
                            shot_segments[seg.get('start_time', 0)] = seg.get('end_time', 0)
            
            # Load shots
            shots_path = Path(output_dir) / f"{base_name}_shots.json"
            if shots_path.exists():
                with open(shots_path, 'r', encoding='utf-8') as f:
                    shots = json.load(f)
                    for shot in shots:
                        shot_frame = shot.get('frame_idx', 0)
                        # Find matching segment end_time for accurate goal timing
                        # Shot might be detected at a different frame than segment start
                        end_time = None
                        for seg_start, seg_end in shot_segments.items():
                            # Shot frame should be within segment range
                            if seg_start <= shot_frame <= seg_end:
                                end_time = seg_end
                                break
                        # Fallback: use closest segment
                        if end_time is None and shot_segments:
                            closest = min(shot_segments.keys(), key=lambda x: abs(x - shot_frame))
                            end_time = shot_segments[closest]
                        
                        events.append({
                            'type': 'shot_candidate',
                            'frame': shot_frame,
                            'end_time': end_time,  # Add end_time for goal timing!
                            'player_id': shot.get('shooter_id', ''),
                            'goal': shot.get('is_goal', False),
                            'speed': shot.get('speed_px_s', 0)
                        })
            
            # Load passes
            passes_path = Path(output_dir) / f"{base_name}_passes.json"
            if passes_path.exists():
                with open(passes_path, 'r', encoding='utf-8') as f:
                    passes = json.load(f)
                    for p in passes:
                        events.append({
                            'type': 'pass',
                            'frame': p.get('start_frame', 0),
                            'start_owner': p.get('sender_id', ''),
                            'end_owner': p.get('receiver_id', ''),
                        })
            
            # Add dribbles from already loaded segments
            if segments_path.exists():
                # segments already loaded above for shot_segments
                for seg in segments:
                    if seg.get('segment_type') == 'dribble':
                        events.append({
                            'type': 'dribble',
                            'frame': seg.get('start_time', 0),
                            'start_owner': seg.get('start_owner', ''),
                                'distance': seg.get('displacement', 0)
                            })
            
            # Load metadata for context
            meta_path = Path(output_dir) / f"{base_name}.meta.json"
            context = {
                'fps': metadata.get('fps', 25.0),
                'duration': video_duration,
                'width': metadata.get('width', 1920),
                'height': metadata.get('height', 1080),
                'team_left': team_left or 'Sol Takım',
                'team_right': team_right or 'Sağ Takım',
            }
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    context['width'] = meta.get('width', context['width'])
                    context['height'] = meta.get('height', context['height'])
                    context['fps'] = meta.get('fps', context['fps'])
            
            # Load shots for context (for accurate shot timing in commentary)
            if shots_path.exists():
                context['shots'] = shots  # shots already loaded above
                print(f"📊 Shots loaded for commentary timing: {len(shots)} shots")
            
            # Load frames_data for zone/pressure detection
            frames_data_list = []
            if frames_jsonl.exists():
                print(f"📂 Loading frames data for zone detection...")
                with open(frames_jsonl, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            frame = json.loads(line)
                            frames_data_list.append(frame)
                        except json.JSONDecodeError:
                            continue
                print(f"   Loaded {len(frames_data_list)} frames")
                context['frames_data'] = frames_data_list
            
            # Sort events by frame
            events.sort(key=lambda x: x.get('frame', 0))
            
            print(f"📊 Video duration: {video_duration:.1f}s, FPS: {context['fps']}")
            print(f"📊 Events to narrate: {len(events)}")
            
            # Generate TIMED narrative using HybridNarrativeGenerator (template-based)
            gen = HybridNarrativeGenerator()
            timed_commentaries = gen.generate(events, context)
            
            print(f"🎙️ Generated {len(timed_commentaries)} timed commentary segments:")
            for tc in timed_commentaries:
                time_str = f"{tc['start_time']:.1f}s"
                tone_str = tc.get('tone', 'neutral')
                print(f"   [{time_str}] ({tone_str}) {tc['event_type']}: {tc['text'][:50]}...")
            
            # Also generate legacy single narrative for JSON
            narrative_text = " ".join([tc['text'] for tc in timed_commentaries])
            
            # Wrap in JSON structure with timing info
            narrative_data = {
                'narrative': narrative_text,
                'timed_segments': timed_commentaries,  # Already dicts from HybridNarrativeGenerator
                'events_count': len(events),
                'video_duration': video_duration,
                'shots': len([e for e in events if e.get('type') == 'shot_candidate']),
                'passes': len([e for e in events if e.get('type') == 'pass']),
                'goals': len([e for e in events if e.get('type') == 'shot_candidate' and e.get('goal')])
            }
            
            # Save narrative JSON
            narrative_path = Path(output_dir) / f"{base_name}_narrative.json"
            with open(narrative_path, 'w', encoding='utf-8') as f:
                json.dump(narrative_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Narrative saved: {narrative_path}")
            
            # ========== SAVE DETAILED COMMENTARY.JSON ==========
            # Her event tipi için ayrı kategoriler (fillers passes içine dahil)
            commentary_by_type = {
                'passes': [],  # Pas yorumları + fillers
                'shots': [],
                'goals': []
            }
            
            for tc in timed_commentaries:
                segment_info = tc.get('segment_info', {})
                outcome = segment_info.get('outcome', '')
                
                entry = {
                    'time': f"{tc['start_time']:.2f}s",
                    'text': tc['text'],
                    'tone': tc.get('tone', 'neutral'),
                    'duration': tc.get('duration', 0)
                }
                
                # Kategorize et
                if outcome == 'goal':
                    commentary_by_type['goals'].append(entry)
                elif outcome in ['shot', 'shot_saved', 'shot_wide', 'shot_post']:
                    commentary_by_type['shots'].append(entry)
                else:
                    # Filler'lar ve diğer tüm yorumlar passes'a gider
                    commentary_by_type['passes'].append(entry)
            
            # Commentary.json'ı güncelle
            import datetime
            commentary_data = {
                'video': str(video_path),
                'generated_at': datetime.datetime.now().isoformat(),
                'video_duration': f"{video_duration:.2f}s",
                'total_commentaries': len(timed_commentaries),
                'summary': {
                    'passes': len(commentary_by_type['passes']),
                    'shots': len(commentary_by_type['shots']),
                    'goals': len(commentary_by_type['goals'])
                },
                'commentaries_by_type': commentary_by_type,
                'all_commentaries': [
                    {
                        'time': f"{tc['start_time']:.2f}s",
                        'end_time': f"{tc['end_time']:.2f}s",
                        'text': tc['text'],
                        'tone': tc.get('tone', 'neutral'),
                        'event_type': tc.get('event_type', ''),
                        'outcome': tc.get('segment_info', {}).get('outcome', '')
                    }
                    for tc in timed_commentaries
                ]
            }
            
            commentary_path = Path(output_dir) / "commentary.json"
            with open(commentary_path, 'w', encoding='utf-8') as f:
                json.dump(commentary_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Commentary details saved: {commentary_path}")
            print(f"   📊 Passes: {len(commentary_by_type['passes'])}, Shots: {len(commentary_by_type['shots'])}, Goals: {len(commentary_by_type['goals'])}")
            print()
            
        except Exception as e:
            print(f"⚠️  Narrative generation failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    else:
        print("⏭️  STAGE 3: NARRATIVE GENERATION (Skipped)")
        print()
    
    # Stage 4: Synchronized Speech Synthesis (Optional)
    audio_path = None
    if enable_speech and narrative_data and timed_commentaries:
        print("▶️ STAGE 4: SYNCHRONIZED SPEECH SYNTHESIS")
        print("-" * 60)
        report_progress("speech", 75, "Generating voice synthesis...")
        
        try:
            use_mock_tts = os.getenv('USE_MOCK_TTS', 'false').lower() == 'true'
            use_batch_tts = os.getenv('USE_BATCH_TTS', 'true').lower() == 'true'  # Varsayılan: batch mode
            
            # TTS provider (Gemini default, ElevenLabs optional)
            from ml_pipeline.core.speech import SpeechGenerator
            speech_gen = SpeechGenerator(use_mock=use_mock_tts)
            
            audio_file = Path(output_dir) / f"{base_name}_narrative_audio.mp3"
            
            print(f"🎬 Creating synchronized audio ({video_duration:.1f}s video)")
            provider_label = getattr(speech_gen, "provider_label", "TTS")
            if getattr(speech_gen, "provider", "") == "gemini":
                voice_label = getattr(speech_gen, "voice_name", "Puck")
                model_label = getattr(speech_gen, "model", "gemini-tts")
                print(f"🎙️ Voice: {voice_label} ({provider_label}, model={model_label})")
            else:
                print(f"🎙️ Voice: Erman ({provider_label})")
            print(f"📢 {len(timed_commentaries)} commentary segments to synthesize")
            
            # Generate synchronized audio with timing
            if use_batch_tts:
                print(f"🚀 Mode: BATCH (tek API çağrısı)")
                audio_path = speech_gen.generate_timed_speech_batch(
                    timed_commentaries, 
                    video_duration, 
                    str(audio_file)
                )
            else:
                print(f"📡 Mode: SEQUENTIAL (her yorum için ayrı API çağrısı)")
                audio_path = speech_gen.generate_timed_speech(
                    timed_commentaries, 
                    video_duration, 
                    str(audio_file)
                )
            
            print(f"💾 Synchronized audio saved: {audio_path}")
            print()
            
        except Exception as e:
            print(f"⚠️  Speech synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    else:
        print("⏭️  STAGE 4: SPEECH SYNTHESIS (Skipped)")
        print()
    
    # Stage 5: Merge Commentary Audio with Background Music, then combine with Video
    result_video_path = None
    if enable_speech and audio_path and Path(audio_path).exists():
        print("▶️ STAGE 5: MERGING AUDIO WITH BACKGROUND MUSIC + VIDEO")
        print("-" * 60)
        report_progress("merge", 90, "Creating final video with mixed audio...")
        
        try:
            import subprocess
            import random
            
            result_video_path = Path(output_dir) / "result.mp4"
            
            print(f"🎬 Processing audio and video...")
            print(f"   📹 Video: {video_path}")
            print(f"   🔊 Commentary Audio: {audio_path}")
            
            # Check for background music file
            bg_music_path = Path(__file__).parent.parent / "commentary_templates" / "arka_plan.mp3"
            use_bg_music = bg_music_path.exists()
            
            if use_bg_music:
                print(f"   🎵 Background Music: {bg_music_path}")
                
                # Step 1: Mix commentary audio with background music (create intermediate file)
                mixed_audio_path = Path(output_dir) / f"{base_name}_narrative_audio_mixed.mp3"
                
                # Get background music duration using ffprobe
                try:
                    probe_cmd = [
                        'ffprobe', '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        str(bg_music_path)
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    bg_music_duration = float(probe_result.stdout.strip())
                    print(f"   🎵 Background music duration: {bg_music_duration:.1f}s")
                    
                    # If video is shorter than bg music, pick a random segment
                    if video_duration < bg_music_duration:
                        max_start = bg_music_duration - video_duration
                        bg_start_time = random.uniform(0, max_start)
                        print(f"   🎵 Using random segment: {bg_start_time:.1f}s - {bg_start_time + video_duration:.1f}s")
                    else:
                        bg_start_time = 0
                        print(f"   🎵 Using full background music (looping if needed)")
                except Exception as e:
                    print(f"   ⚠️  Could not get bg music duration: {e}")
                    bg_start_time = 0
                    bg_music_duration = 240  # Assume 4 min default
                
                # Step 1: Mix commentary audio with background music (create intermediate file)
                print(f"\n   📀 Step 1: Mixing commentary with background music...")
                
                # FFmpeg command: mix commentary + background music
                # Background music volume: 0.25 (25% - subtle but audible background)
                # Commentary volume: 1.0 (100% - main audio)
                mix_audio_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(audio_path),           # Commentary audio (already noise-reduced)
                    '-ss', str(bg_start_time),       # Start time for bg music
                    '-i', str(bg_music_path),        # Background music
                    '-filter_complex',
                    # Mix commentary (full volume) with background music (low volume)
                    '[0:a]volume=1.0[commentary];'
                    '[1:a]volume=0.65[bgmusic];'
                    '[commentary][bgmusic]amix=inputs=2:duration=first:dropout_transition=2',
                    '-b:a', '192k',
                    str(mixed_audio_path)
                ]
                
                mix_result = subprocess.run(mix_audio_cmd, capture_output=True, text=True)
                
                if mix_result.returncode == 0 and mixed_audio_path.exists():
                    mixed_size = mixed_audio_path.stat().st_size / 1024  # KB
                    print(f"   ✅ Mixed audio created: {mixed_audio_path} ({mixed_size:.1f} KB)")
                    print(f"      🎙️ Commentary + 🎵 Background music @ 25% volume")
                else:
                    print(f"   ⚠️  Audio mixing failed: {mix_result.stderr}")
                    mixed_audio_path = Path(audio_path)  # Fallback to original
                
                # Step 2: Merge video with mixed audio
                print(f"\n   🎬 Step 2: Merging video with mixed audio...")
                
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_path),           # Original video
                    '-i', str(mixed_audio_path),     # Mixed audio (commentary + bg music)
                    '-c:v', 'copy',                  # Copy video codec (no re-encoding)
                    '-c:a', 'aac',                   # Encode audio to AAC
                    '-map', '0:v:0',                 # Use video from first input
                    '-map', '1:a:0',                 # Use mixed audio from second input
                    '-shortest',                     # Match shortest duration
                    str(result_video_path)
                ]
            else:
                print(f"   ⚠️  Background music not found at: {bg_music_path}")
                print(f"   ⏭️  Proceeding without background music...")
                
                # FFmpeg command to merge video and audio (without bg music)
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_path),           # Original video
                    '-i', str(audio_path),           # Generated audio
                    '-c:v', 'copy',                  # Copy video codec (no re-encoding)
                    '-c:a', 'aac',                   # Encode audio to AAC
                    '-map', '0:v:0',                 # Use video from first input
                    '-map', '1:a:0',                 # Use audio from second input
                    '-shortest',                     # Match shortest duration
                    str(result_video_path)
                ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result_video_path.exists():
                file_size = result_video_path.stat().st_size / (1024 * 1024)  # MB
                print(f"\n✅ Final video created: {result_video_path} ({file_size:.2f} MB)")
                if use_bg_music:
                    print(f"   🎙️ Commentary track merged")
                    print(f"   🎵 Background music @ 25% volume")
                    print(f"   📹 Original video preserved")
            else:
                print(f"⚠️  FFmpeg failed: {result.stderr}")
                result_video_path = None
            print()
            
        except FileNotFoundError:
            print("⚠️  FFmpeg not found. Please install FFmpeg to merge video with audio.")
            print("   Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
            print()
        except Exception as e:
            print(f"⚠️  Video merge failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    else:
        print("⏭️  STAGE 5: VIDEO MERGE (Skipped - no audio generated)")
        print()
    
    # Build output summary
    output_files = {
        "video": str(Path(output_dir) / "track_vis_out.mp4"),
        "frames_jsonl": str(frames_jsonl),
        "metadata": str(Path(output_dir) / f"{base_name}.meta.json"),
        "passes": str(Path(output_dir) / f"{base_name}_passes.json"),
        "shots": str(Path(output_dir) / f"{base_name}_shots.json"),
        "segments": str(Path(output_dir) / f"{base_name}_segments.json"),
        "summary": str(Path(output_dir) / f"{base_name}_event_summary.json"),
    }
    
    if enable_narrative:
        output_files["narrative_json"] = str(Path(output_dir) / f"{base_name}_narrative.json")
    
    if enable_speech and audio_path:
        output_files["audio_clean"] = audio_path  # Clean commentary (noise-reduced)
        # Check for mixed audio (with background music)
        mixed_audio_file = Path(output_dir) / f"{base_name}_narrative_audio_mixed.mp3"
        if mixed_audio_file.exists():
            output_files["audio_mixed"] = str(mixed_audio_file)  # Commentary + Background music
    
    if result_video_path and Path(result_video_path).exists():
        output_files["result_video"] = str(result_video_path)
    
    # Print summary
    print("=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"📊 Processed {len(frames)} frames")
    
    if enable_narrative and narrative_data:
        print(f"⚽ Detected {narrative_data.get('passes', 0)} passes")
        print(f"🎯 Detected {narrative_data.get('shots', 0)} shots")
        print(f"🥅 Scored {narrative_data.get('goals', 0)} goals")
        print(f"💬 Generated narrative: {len(narrative_data.get('narrative', ''))} characters")
    
    if enable_speech and audio_path:
        file_size = Path(audio_path).stat().st_size if Path(audio_path).exists() else 0
        print(f"🔊 Clean commentary audio: {audio_path} ({file_size} bytes)")
        mixed_audio_file = Path(output_dir) / f"{base_name}_narrative_audio_mixed.mp3"
        if mixed_audio_file.exists():
            mixed_size = mixed_audio_file.stat().st_size
            print(f"🎵 Mixed audio (commentary + bg): {mixed_audio_file} ({mixed_size} bytes)")
    
    print()
    print("📂 Output files:")
    for key, path in output_files.items():
        if Path(path).exists():
            print(f"   ✓ {key}: {path}")
        else:
            print(f"   ⏭️  {key}: {path} (not generated)")
    print("=" * 60)
    
    return {
        "frames": frames,
        "metadata": metadata,
        "narrative": narrative_data,
        "audio_path": audio_path,
        "result_video": str(result_video_path) if result_video_path else None,
        "output_files": output_files
    }
