"""
FastAPI Backend Server for Let The Game Speak
"""
import os
import sys
import uuid
import asyncio
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Let The Game Speak API",
    description="AI-powered football match commentary system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job storage (in-memory for now)
jobs = {}

class JobStatus(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: int
    current_stage: str
    stage_details: str
    stages: dict
    error: Optional[str] = None
    created_at: str
    video_path: Optional[str] = None
    output_path: Optional[str] = None


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    team_left: Optional[str] = Form(None),
    team_right: Optional[str] = Form(None)
):
    """Upload a video file for processing"""
    
    # Validate file type
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "current_stage": "Queued",
        "stage_details": "Waiting to start processing...",
        "stages": {
            "upload": {"status": "completed", "progress": 100},
            "vision": {"status": "pending", "progress": 0},
            "calibration": {"status": "pending", "progress": 0},
            "events": {"status": "pending", "progress": 0},
            "narrative": {"status": "pending", "progress": 0},
            "speech": {"status": "pending", "progress": 0},
            "merge": {"status": "pending", "progress": 0},
        },
        "error": None,
        "created_at": datetime.now().isoformat(),
        "video_path": str(upload_path),
        "output_path": None,
        "team_left": team_left,
        "team_right": team_right
    }
    
    # Start processing in a separate thread (non-blocking)
    thread = threading.Thread(target=process_video_sync, args=(job_id, str(upload_path), team_left, team_right))
    thread.start()
    
    return {"job_id": job_id, "message": "Upload successful, processing started"}


def process_video_sync(job_id: str, video_path: str, team_left: Optional[str] = None, team_right: Optional[str] = None):
    """Process video synchronously in a separate thread"""
    try:
        jobs[job_id]["status"] = "processing"
        
        # Import pipeline
        from ml_pipeline.pipeline import run_pipeline
        
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(exist_ok=True)
        
        # Update progress callback
        def update_progress(stage: str, progress: int, details: str = ""):
            stage_map = {
                "vision": "vision",
                "Vision Processing": "vision",
                "calibration": "calibration",
                "2D Calibration": "calibration",
                "events": "events",
                "Event Detection": "events",
                "narrative": "narrative",
                "AI Commentary": "narrative",
                "speech": "speech",
                "Voice Synthesis": "speech",
                "merge": "merge",
                "Final Merge": "merge",
            }
            
            stage_key = stage_map.get(stage, stage.lower())
            
            jobs[job_id]["current_stage"] = stage
            jobs[job_id]["stage_details"] = details
            jobs[job_id]["progress"] = progress
            
            if stage_key in jobs[job_id]["stages"]:
                jobs[job_id]["stages"][stage_key]["status"] = "processing"
                jobs[job_id]["stages"][stage_key]["progress"] = progress
        
        # Run pipeline
        result = run_pipeline(
            video_path=video_path,
            output_dir=str(output_dir),
            model_path="./weights/last.pt",
            enable_narrative=True,
            enable_speech=True,
            team_left=team_left,
            team_right=team_right,
            progress_callback=update_progress
        )
        
        # Mark all stages as completed
        for stage in jobs[job_id]["stages"]:
            jobs[job_id]["stages"][stage]["status"] = "completed"
            jobs[job_id]["stages"][stage]["progress"] = 100
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["current_stage"] = "Completed"
        jobs[job_id]["stage_details"] = "Processing complete!"
        jobs[job_id]["output_path"] = str(output_dir)
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["current_stage"] = "Failed"
        jobs[job_id]["stage_details"] = str(e)


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/api/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get the results of a completed job"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    output_dir = Path(job["output_path"])
    
    # Find output files
    results = {
        "job_id": job_id,
        "video_url": None,
        "field_video_url": None,
        "track_video_url": None,
        "narrative": None,
        "events": [],
        "segments": [],
        "statistics": {}
    }
    
    import json
    
    # Check for result video (main output with commentary)
    result_video = output_dir / "result.mp4"
    if result_video.exists():
        results["video_url"] = f"/api/files/{job_id}/result.mp4"
    
    # Check for tracking video (fallback)
    track_video = output_dir / "track_vis_out.mp4"
    if track_video.exists():
        results["track_video_url"] = f"/api/files/{job_id}/track_vis_out.mp4"
        if not results["video_url"]:
            results["video_url"] = results["track_video_url"]
    
    # Check for 2D field video
    field_video = output_dir / f"{job_id}_2d_field_events.mp4"
    if field_video.exists():
        results["field_video_url"] = f"/api/files/{job_id}/{job_id}_2d_field_events.mp4"
    
    # Load narrative
    narrative_file = output_dir / f"{job_id}_narrative.json"
    if narrative_file.exists():
        with open(narrative_file) as f:
            narrative_data = json.load(f)
            results["narrative"] = narrative_data
            
            # Convert timed_segments to events for frontend
            timed_segments = narrative_data.get('timed_segments', [])
            results["events"] = [{
                "id": i + 1,
                "type": seg.get('event_type', 'commentary'),
                "time": seg.get('start_time', 0),
                "end_time": seg.get('end_time', 0),
                "description": seg.get('text', ''),
                "tone": seg.get('tone', 'neutral')
            } for i, seg in enumerate(timed_segments)]
            
            # Build statistics from narrative data
            results["statistics"] = {
                "totalEvents": narrative_data.get('events_count', 0),
                "passes": narrative_data.get('passes', 0),
                "shots": narrative_data.get('shots', 0),
                "goals": narrative_data.get('goals', 0)
            }
    
    # Load segments data
    segments_file = output_dir / f"{job_id}_segments.json"
    if segments_file.exists():
        with open(segments_file) as f:
            results["segments"] = json.load(f)
    
    return results


@app.get("/api/files/{job_id}/{filename}")
async def serve_file(job_id: str, filename: str):
    """Serve output files"""
    
    file_path = OUTPUT_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)


# Serve static demo files
demo_dir = Path("./frontend/public/demo")
if demo_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(demo_dir)), name="demo")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)