"""
HoopSense AI — Core Configuration
Environment-based settings using Pydantic BaseSettings
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "HoopSense AI"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # API
    API_SECRET_KEY: str = "hoopsense-dev-secret-change-in-production"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # File paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "data", "uploads")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "data", "outputs")
    MODEL_DIR: str = os.path.join(BASE_DIR, "data", "models")
    PROCESSED_DIR: str = os.path.join(BASE_DIR, "data", "processed")

    # Video processing
    MAX_VIDEO_SIZE_MB: int = 500
    SUPPORTED_FORMATS: List[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    MAX_FRAMES_EXTRACT: int = 300
    TARGET_FPS: int = 30

    # MediaPipe
    MEDIAPIPE_DETECTION_CONFIDENCE: float = 0.7
    MEDIAPIPE_TRACKING_CONFIDENCE: float = 0.5
    MEDIAPIPE_MODEL_COMPLEXITY: int = 2  # 0=lite, 1=full, 2=heavy

    # YOLO
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.4
    YOLO_IOU_THRESHOLD: float = 0.45

    # ML Model
    FORM_MODEL_PATH: str = ""      # populated at runtime
    SUCCESS_MODEL_PATH: str = ""   # populated at runtime
    MODEL_VERSION: str = "1.0.0"

    # Feature engineering
    WINDOW_SIZE: int = 30           # frames for temporal features
    TRAJECTORY_POINTS: int = 50     # interpolated trajectory samples

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "hoopsense.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
