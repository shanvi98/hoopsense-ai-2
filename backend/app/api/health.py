"""HoopSense AI — Health Check Routes"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "service": "HoopSense AI", "timestamp": datetime.utcnow().isoformat()}

@router.get("/health/ready", summary="Readiness probe")
async def ready():
    return {"ready": True, "models_loaded": True}
