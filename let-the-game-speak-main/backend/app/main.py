"""
FastAPI Backend for Let The Game Speak
Handles video upload, ML pipeline processing, and results delivery
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
import json
from typing import Dict, Optional
from datetime import datetime
import os
import sys
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_pipeline.pipeline import run_pipeline

app = FastAPI(
    title="Let The Game Speak API",
    description="AI Football Commentary Generation API",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
project_root = Path(__file__).parent.parent.parent
UPLOAD_DIR = project_root / "backend" / "uploads"
OUTPUT_DIR = project_root / "backend" / "output"
WEIGHTS_PATH = project_root / "weights" / "last.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount output directory for static file serving
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

# In-memory job storage (replace with DB in production)
jobs: Dict[str, Dict] = {}


def load_existing_jobs():
    """Load existing jobs from disk on startup"""
    if not OUTPUT_DIR.exists():
        return
    
    for job_dir in OUTPUT_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        
        job_id = job_dir.name
        track_video = job_dir / "track_vis_out.mp4"
        
        # Only load completed jobs that have output video
        if track_video.exists():
            # Get file modification time as created_at fallback
            file_mtime = datetime.fromtimestamp(track_video.stat().st_mtime)
            
            jobs[job_id] = {
                "job_id": job_id,
                "status": "completed",
                "progress": 100,
                "output_dir": str(job_dir),
                "video_path": str(track_video),
                "created_at": file_mtime.isoformat(),
                "completed_at": file_mtime.isoformat(),
                "current_stage": "Complete",
                "stage_details": "Video analysis complete!"
            }
            print(f"✅ Loaded existing job: {job_id}")


# Load existing jobs on startup
load_existing_jobs()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Let The Game Speak API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "ok",
            "ml_pipeline": "ok",
            "storage": "ok"
        }
    }


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    team_left: Optional[str] = Form(None),
    team_right: Optional[str] = Form(None)
):
    """
    Upload video and start ML pipeline processing
    
    Returns:
        job_id: Unique identifier for tracking processing status
    """
    # Validate file
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    team_left = team_left.strip() if team_left else None
    team_right = team_right.strip() if team_right else None

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job-specific directories
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    video_filename = f"input{Path(file.filename).suffix}"
    video_path = job_upload_dir / video_filename
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize job status with detailed progress tracking
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0,
        "current_stage": "Initializing",
        "stage_details": "",
        "created_at": datetime.now().isoformat(),
        "video_filename": file.filename,
        "video_path": str(video_path),
        "output_dir": str(job_output_dir),
        "team_left": team_left,
        "team_right": team_right,
        "stages": {
            "upload": {"status": "completed", "progress": 100},
            "vision": {"status": "pending", "progress": 0},
            "calibration": {"status": "pending", "progress": 0},
            "events": {"status": "pending", "progress": 0},
            "narrative": {"status": "pending", "progress": 0},
            "speech": {"status": "pending", "progress": 0},
            "merge": {"status": "pending", "progress": 0}
        }
    }
    
    # Start background processing in a separate thread
    thread = threading.Thread(
        target=process_video, 
        args=(job_id, str(video_path), str(job_output_dir), team_left, team_right),
        daemon=True
    )
    thread.start()
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video uploaded successfully, processing started"
    }


def process_video(job_id: str, video_path: str, output_dir: str, team_left: Optional[str], team_right: Optional[str]):
    """
    Background task to process video through ML pipeline with real-time progress updates
    """
    try:
        # Progress callback function - updates job status in real-time
        def progress_callback(stage: str, progress: int, details: str):
            """Update job progress from pipeline"""
            print(f"📊 PROGRESS UPDATE: stage={stage}, progress={progress}%, details={details}")
            
            stage_map = {
                "vision": ("Vision Processing", "vision"),
                "calibration": ("2D Field Calibration", "calibration"),
                "events": ("Event Detection", "events"),
                "narrative": ("Narrative Generation", "narrative"),
                "speech": ("Speech Synthesis", "speech"),
                "merge": ("Video Merge", "merge")
            }
            
            if stage in stage_map:
                stage_name, stage_key = stage_map[stage]
                jobs[job_id]["current_stage"] = stage_name
                jobs[job_id]["stage_details"] = details
                jobs[job_id]["progress"] = progress
                jobs[job_id]["stages"][stage_key]["status"] = "processing"
                jobs[job_id]["stages"][stage_key]["progress"] = min(progress * 2, 100)  # Scale for stage
                
                # Mark previous stages as completed
                stage_order = ["upload", "vision", "calibration", "events", "narrative", "speech", "merge"]
                current_idx = stage_order.index(stage_key) if stage_key in stage_order else 0
                for i, s in enumerate(stage_order):
                    if i < current_idx:
                        jobs[job_id]["stages"][s]["status"] = "completed"
                        jobs[job_id]["stages"][s]["progress"] = 100
                
                print(f"   Jobs dict updated: progress={jobs[job_id]['progress']}, stage={jobs[job_id]['current_stage']}")
        
        # Mark upload as complete, start vision
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 5
        jobs[job_id]["current_stage"] = "Vision Processing"
        jobs[job_id]["stage_details"] = "Starting video analysis..."
        jobs[job_id]["stages"]["upload"]["status"] = "completed"
        jobs[job_id]["stages"]["upload"]["progress"] = 100
        jobs[job_id]["stages"]["vision"]["status"] = "processing"
        print(f"🎬 Starting ML pipeline for job {job_id}")
        
        # Run pipeline with progress callback
        results = run_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            model_path=str(WEIGHTS_PATH),
            enable_narrative=True,
            enable_speech=True,
            team_left=team_left,
            team_right=team_right,
            progress_callback=progress_callback
        )
        
        # Mark all stages as completed
        for stage_key in ["vision", "calibration", "events", "narrative", "speech", "merge"]:
            jobs[job_id]["stages"][stage_key]["status"] = "completed"
            jobs[job_id]["stages"][stage_key]["progress"] = 100
        
        # Update job with results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["current_stage"] = "Completed"
        jobs[job_id]["stage_details"] = "Video analysis complete!"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["results"] = results
        
        print(f"✅ Job {job_id} completed successfully")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["current_stage"] = "Failed"
        jobs[job_id]["stage_details"] = f"Error: {str(e)}"
        print(f"❌ Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get detailed job processing status with stage information
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "current_stage": job.get("current_stage", "Initializing"),
        "stage_details": job.get("stage_details", ""),
        "stages": job.get("stages", {}),
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "error": job.get("error")
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """
    Get full results for a completed job
    
    Returns:
        - videoUrl: URL to processed video with bounding boxes
        - events: Detected events (passes, shots, goals)
        - commentary: Generated Turkish commentary
        - statistics: Match statistics
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed yet (status: {job['status']})"
        )
    
    output_dir = Path(job["output_dir"])
    video_filename = Path(job["video_path"]).stem
    
    # Find output video - prioritize result.mp4 (with background music), fallback to track_vis_out.mp4
    result_video = output_dir / "result.mp4"
    track_video = output_dir / "track_vis_out.mp4"
    
    if result_video.exists():
        video_url = f"/files/{job_id}/result.mp4"
        print(f"📹 Serving final video with background music: {result_video}")
    elif track_video.exists():
        video_url = f"/files/{job_id}/track_vis_out.mp4"
        print(f"📹 Serving tracking video: {track_video}")
    else:
        video_url = None
        print(f"⚠️  No output video found")

    track_video_url = f"/files/{job_id}/track_vis_out.mp4" if track_video.exists() else None
    field_video = output_dir / f"{video_filename}_2d_field_events.mp4"
    field_video_url = f"/files/{job_id}/{field_video.name}" if field_video.exists() else None
    
    # Load events from segments and passes
    events = []
    segments_data = []
    
    # Load segments (includes passes, dribbles, etc.)
    segments_file = output_dir / "input_segments.json"
    if segments_file.exists():
        with open(segments_file, 'r', encoding='utf-8') as f:
            segments_data = json.load(f)
            for seg in segments_data:
                segment_type = seg.get("segment_type", "unknown")
                start_owner = seg.get("start_owner", "?")
                end_owner = seg.get("end_owner", "?")
                start_time = seg.get("start_time", 0) / 25.0  # Convert frame to seconds (assuming 25 fps)
                duration = (seg.get("end_time", 0) - seg.get("start_time", 0)) / 25.0
                
                # Determine team (L = team 0/home, R = team 1/away)
                team = "home" if start_owner.startswith("L") else "away"
                
                if segment_type == "pass":
                    events.append({
                        "id": len(events) + 1,
                        "type": "pass",
                        "time": start_time,
                        "duration": duration,
                        "team": team,
                        "description": f"🎯 Pas: {start_owner} → {end_owner}",
                        "player": start_owner,
                        "receiver": end_owner,
                        "displacement": round(seg.get("displacement", 0), 1),
                        "speed": round(seg.get("average_speed", 0), 1),
                        "importance": "medium"
                    })
                elif segment_type == "dribble":
                    events.append({
                        "id": len(events) + 1,
                        "type": "dribble",
                        "time": start_time,
                        "duration": duration,
                        "team": team,
                        "description": f"⚡ Dribling: {start_owner}",
                        "player": start_owner,
                        "displacement": round(seg.get("displacement", 0), 1),
                        "speed": round(seg.get("average_speed", 0), 1),
                        "importance": "low"
                    })
    
    # Load passes (for additional details)
    passes_file = output_dir / "input_passes.json"
    pass_details = {}
    if passes_file.exists():
        with open(passes_file, 'r', encoding='utf-8') as f:
            passes = json.load(f)
            for pas in passes:
                start_time = pas.get("start_time", 0)
                pass_details[start_time] = pas
    
    # Enhance pass events with additional details
    for event in events:
        if event["type"] == "pass" and event["time"] in pass_details:
            pas = pass_details[event["time"]]
            event["pass_type"] = pas.get("pass_type", "short")
            event["confidence"] = pas.get("confidence", 0)
    
    # Load shots
    shots_file = output_dir / "input_shots.json"
    if shots_file.exists():
        with open(shots_file, 'r', encoding='utf-8') as f:
            shots = json.load(f)
            for shot in shots:
                team = "home" if shot.get("shooter_team") == 0 else "away"
                events.append({
                    "id": len(events) + 1,
                    "type": "goal" if shot.get("is_goal") else "shot",
                    "time": shot.get("time", 0),
                    "team": team,
                    "description": f"⚽ GOL!" if shot.get("is_goal") else "💥 Şut!",
                    "player": shot.get("shooter_id", "?"),
                    "importance": "high"
                })
    
    # Sort events by time
    events.sort(key=lambda x: x["time"])
    
    # Load narrative
    narrative_text = "Maç analizi tamamlandı."
    narrative_data = None
    narrative_file = output_dir / f"{video_filename}_narrative.json"
    if narrative_file.exists():
        with open(narrative_file, 'r', encoding='utf-8') as f:
            narrative_data = json.load(f)
            narrative_text = narrative_data.get("narrative", narrative_text)
    
    # Audio URL
    audio_file = output_dir / f"{video_filename}_narrative_audio.mp3"
    audio_url = f"/files/{job_id}/{video_filename}_narrative_audio.mp3" if audio_file.exists() else None
    
    # Statistics from event summary
    summary_file = output_dir / "input_event_summary.json"
    stats = {
        "totalEvents": len(events),
        "goals": 0,
        "shots": 0,
        "passes": 0,
        "dribbles": 0
    }
    
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            stats.update({
                "totalSegments": summary.get("total_segments", 0),
                "goals": summary.get("total_goals", 0),
                "shots": summary.get("total_shots", 0),
                "passes": summary.get("total_passes", 0),
                "passBreakdown": summary.get("pass_breakdown", {}),
                "shotBreakdown": summary.get("shot_breakdown", {})
            })
    
    # Count event types from actual events
    event_counts = {"pass": 0, "dribble": 0, "shot": 0, "goal": 0}
    for event in events:
        event_type = event.get("type", "unknown")
        if event_type in event_counts:
            event_counts[event_type] += 1
    
    stats["dribbles"] = event_counts["dribble"]
    stats["totalEvents"] = len(events)
    
    return {
        "job_id": job_id,
        "video_url": video_url,
        "field_video_url": field_video_url,
        "track_video_url": track_video_url,
        "videoUrl": video_url,
        "fieldVideoUrl": field_video_url,
        "trackVideoUrl": track_video_url,
        "events": events,
        "segments": segments_data,
        "narrative": narrative_data,
        "commentary": {
            "text": narrative_text,
            "audioUrl": audio_url
        },
        "statistics": stats
    }


@app.get("/api/jobs")
async def list_jobs():
    """
    List all jobs
    """
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
                "video_filename": job.get("video_filename")
            }
            for job_id, job in jobs.items()
        ]
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    
    if job_upload_dir.exists():
        shutil.rmtree(job_upload_dir)
    
    if job_output_dir.exists():
        shutil.rmtree(job_output_dir)
    
    # Delete from memory
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
