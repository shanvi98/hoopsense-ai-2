"""HoopSense AI — Model Info Routes"""
from fastapi import APIRouter
from app.models.schemas import ModelInfoResponse
from app.core.config import settings

router = APIRouter()

@router.get("/info", response_model=ModelInfoResponse, summary="Model registry info")
async def model_info():
    return ModelInfoResponse(
        form_model_version=settings.MODEL_VERSION,
        success_model_version=settings.MODEL_VERSION,
        last_updated="2025-01-01",
        supported_formats=settings.SUPPORTED_FORMATS,
        max_video_size_mb=settings.MAX_VIDEO_SIZE_MB,
    )
