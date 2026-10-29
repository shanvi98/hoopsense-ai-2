"""HoopSense AI — Health Check Routes"""
from fastapi import APIRouter
from datetime import datetime
from app.api.analysis import _service

router = APIRouter()

@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "service": "HoopSense AI", "timestamp": datetime.utcnow().isoformat()}

@router.get("/health/ready", summary="Readiness probe")
async def ready():
    caps = _service.capabilities()
    return {"ready": caps["video_analysis"], **caps}
