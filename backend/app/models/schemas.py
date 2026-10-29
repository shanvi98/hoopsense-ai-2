"""
HoopSense AI — Data Models & Schemas
Pydantic v2 models for request/response validation and internal data structures
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np


# ─── Enums ────────────────────────────────────────────────────────────────────

class AnalysisStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class ShotPhase(str, Enum):
    SETUP    = "setup"
    GATHER   = "gather"
    RELEASE  = "release"
    FOLLOW   = "follow_through"
    UNKNOWN  = "unknown"


class FormGrade(str, Enum):
    ELITE   = "Elite"
    GOOD    = "Good"
    AVERAGE = "Average"
    POOR    = "Poor"


# ─── Pose & Detection ─────────────────────────────────────────────────────────

class PoseLandmark(BaseModel):
    """Single MediaPipe pose landmark."""
    x: float
    y: float
    z: float
    visibility: float = 0.0


class PoseFrame(BaseModel):
    """All pose landmarks for a single frame."""
    frame_idx: int
    timestamp_ms: float
    landmarks: Dict[str, PoseLandmark]
    shot_phase: ShotPhase = ShotPhase.UNKNOWN
    confidence: float = 0.0


class BallDetection(BaseModel):
    """YOLOv8 basketball detection for a single frame."""
    frame_idx: int
    timestamp_ms: float
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class HoopDetection(BaseModel):
    """YOLOv8 hoop/basket detection."""
    frame_idx: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float


# ─── Feature Vector ───────────────────────────────────────────────────────────

class BiomechanicalFeatures(BaseModel):
    """Engineered features for ML inference."""
    # Joint angles (degrees)
    elbow_angle_release: float = Field(default=0.0, description="Elbow angle at release point")
    elbow_angle_min: float = Field(default=0.0, description="Minimum elbow angle during shot")
    knee_bend_max: float = Field(default=0.0, description="Maximum knee flexion during gather")
    wrist_release_angle: float = Field(default=0.0, description="Wrist extension at release")
    hip_shoulder_alignment: float = Field(default=0.0, description="Lateral alignment score")
    trunk_lean_angle: float = Field(default=0.0, description="Trunk forward lean at release")
    shooting_arm_extension: float = Field(default=0.0, description="Arm extension percentage")

    # Ball trajectory
    release_height_normalized: float = Field(default=0.0, description="Release height / player height")
    ball_arc_angle: float = Field(default=0.0, description="Estimated arc angle in degrees")
    ball_velocity_release: float = Field(default=0.0, description="Ball velocity at release (px/frame)")
    ball_velocity_peak: float = Field(default=0.0, description="Peak ball velocity during shot")
    trajectory_smoothness: float = Field(default=0.0, description="Smoothness of ball path (0-1)")

    # Timing & rhythm
    shot_duration_frames: int = Field(default=0, description="Total frames from gather to release")
    jump_height_normalized: float = Field(default=0.0, description="Jump height / player height")
    release_timing_score: float = Field(default=0.0, description="Release at peak jump (0-1)")
    rhythm_consistency: float = Field(default=0.0, description="Gather-to-release rhythm (0-1)")

    # Spatial
    player_distance_hoop: float = Field(default=0.0, description="Estimated player-to-hoop distance (px)")
    shot_angle_degrees: float = Field(default=0.0, description="Lateral shot angle")
    balance_score: float = Field(default=0.0, description="Left-right balance during shot (0-1)")
    follow_through_score: float = Field(default=0.0, description="Follow-through completeness (0-1)")


# ─── Predictions ──────────────────────────────────────────────────────────────

class FeedbackItem(BaseModel):
    """Single biomechanical feedback suggestion."""
    aspect: str
    rating: str          # "good" | "warning" | "error"
    message: str
    metric_value: Optional[float] = None
    benchmark_range: Optional[str] = None


class ShotPrediction(BaseModel):
    """Complete prediction result for a single shot."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    status: AnalysisStatus = AnalysisStatus.COMPLETED

    # Core scores
    form_score: float = Field(ge=0, le=100, description="Biomechanical form score 0-100")
    form_grade: FormGrade
    make_probability: float = Field(ge=0, le=1, description="Shot success probability 0-1")
    confidence: float = Field(ge=0, le=1, description="Model confidence in prediction")

    # Detailed features
    features: BiomechanicalFeatures
    feedback: List[FeedbackItem] = []

    # Trajectory data (for visualization)
    ball_trajectory: List[Dict[str, float]] = []    # [{x, y, frame}]
    pose_frames: List[Dict[str, Any]] = []           # sparse key frames
    hoop_position: Optional[Dict[str, float]] = None

    # Video metadata
    video_fps: float = 30.0
    video_duration_s: float = 0.0
    total_frames: int = 0
    shot_phase_frames: Dict[str, int] = {}

    # Model versioning
    model_version: str = "1.0.0"


# ─── API Request / Response ───────────────────────────────────────────────────

class AnalysisJobResponse(BaseModel):
    """Immediate response after video upload."""
    session_id: str
    status: AnalysisStatus
    message: str
    estimated_duration_s: int = 30


class AnalysisResultResponse(BaseModel):
    """Full analysis result returned to the frontend."""
    session_id: str
    status: AnalysisStatus
    result: Optional[ShotPrediction] = None
    error: Optional[str] = None
    processing_time_ms: int = 0


class ModelInfoResponse(BaseModel):
    """Model registry info."""
    form_model_version: str
    success_model_version: str
    last_updated: str
    supported_formats: List[str]
    max_video_size_mb: int
