"""
HoopSense AI — Analysis API Routes
Video upload, async processing, and result polling endpoints
"""
import os
import asyncio
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.models.schemas import AnalysisJobResponse, AnalysisResultResponse, AnalysisStatus
from app.services.analysis_service import AnalysisService

router = APIRouter()
logger = logging.getLogger(__name__)
_service = AnalysisService()


@router.post("/upload", response_model=AnalysisJobResponse, summary="Upload video for analysis")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a basketball video for shooting analysis.
    Returns session_id immediately; poll /status/{session_id} for results.
    """
    # Basic validation
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in settings.SUPPORTED_FORMATS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Use: {settings.SUPPORTED_FORMATS}")

    # Create session
    session_id = _service.create_session()

    # Save upload
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(settings.UPLOAD_DIR, f"{session_id}{ext}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_VIDEO_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max: {settings.MAX_VIDEO_SIZE_MB}MB")

    with open(save_path, "wb") as f:
        f.write(content)

    logger.info("Upload received: %s (%.1fMB) → session %s", file.filename, size_mb, session_id)

    # Queue background analysis
    background_tasks.add_task(_run_analysis, save_path, session_id)

    return AnalysisJobResponse(
        session_id=session_id,
        status=AnalysisStatus.PROCESSING,
        message=f"Video received ({size_mb:.1f}MB). Analysis started.",
        estimated_duration_s=max(10, int(size_mb * 3)),
    )


@router.get("/status/{session_id}", response_model=AnalysisResultResponse, summary="Poll analysis status")
async def get_status(session_id: str):
    """Poll for analysis results. Returns partial data while processing."""
    session = _service.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")

    return AnalysisResultResponse(
        session_id=session_id,
        status=session["status"],
        result=session.get("result"),
        error=session.get("error"),
        processing_time_ms=session.get("elapsed", 0),
    )


@router.post("/demo", response_model=AnalysisResultResponse, summary="Run demo analysis (no video needed)")
async def demo_analysis():
    """
    Run analysis on synthetic demo data.
    Returns a realistic prediction without requiring video upload.
    """
    session_id = _service.create_session()
    # Use a synthetic 'demo' path that triggers synthetic data generation
    result = await _service.analyze_video("__demo__", session_id)
    session = _service.get_session(session_id)
    return AnalysisResultResponse(
        session_id=session_id,
        status=AnalysisStatus.COMPLETED,
        result=result,
        processing_time_ms=session.get("elapsed", 500),
    )


async def _run_analysis(video_path: str, session_id: str):
    """Background task wrapper for async analysis pipeline."""
    try:
        await _service.analyze_video(video_path, session_id)
    except Exception as e:
        logger.error("Background analysis failed for %s: %s", session_id, e)
