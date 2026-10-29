"""
HoopSense AI — Pose Estimation Service
Full-body landmark extraction using MediaPipe Pose with shot phase classification
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from app.core.config import settings
from app.models.schemas import PoseLandmark, PoseFrame, ShotPhase

logger = logging.getLogger(__name__)

# MediaPipe landmark names indexed by ID (33 total)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_wrist", "left_index"), ("right_wrist", "right_index"),
    ("nose", "left_eye"), ("nose", "right_eye"),
]


class PoseEstimator:
    """
    MediaPipe-based full-body pose estimation.
    Raises if MediaPipe is unavailable; synthetic landmarks are only produced
    explicitly via _generate_synthetic_pose (demo mode).
    """

    def __init__(self):
        self._mp_pose = None
        self._pose_model = None
        self._available = False
        self._initialize()

    def _initialize(self):
        """Lazy-initialize MediaPipe to avoid import errors in environments without it."""
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            self._pose_model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                min_detection_confidence=settings.MEDIAPIPE_DETECTION_CONFIDENCE,
                min_tracking_confidence=settings.MEDIAPIPE_TRACKING_CONFIDENCE,
            )
            self._available = True
            logger.info("MediaPipe Pose initialized (complexity=%d)", settings.MEDIAPIPE_MODEL_COMPLEXITY)
        except ImportError:
            logger.warning("MediaPipe not installed — real video analysis disabled (demo mode only)")
        except Exception as e:
            logger.warning("MediaPipe init failed: %s — real video analysis disabled (demo mode only)", e)

    @property
    def available(self) -> bool:
        return self._available and self._pose_model is not None

    def process_frame(self, rgb_frame: np.ndarray, frame_idx: int, timestamp_ms: float) -> Optional[PoseFrame]:
        """Extract pose landmarks from a single RGB frame."""
        if not self.available:
            raise RuntimeError("MediaPipe is not available — cannot run pose estimation on real video")
        return self._process_with_mediapipe(rgb_frame, frame_idx, timestamp_ms)

    def _process_with_mediapipe(self, rgb: np.ndarray, idx: int, ts: float) -> Optional[PoseFrame]:
        """Run MediaPipe inference on a single frame."""
        try:
            results = self._pose_model.process(rgb)
            if not results.pose_landmarks:
                return None

            landmarks = {}
            total_visibility = 0.0
            for i, lm in enumerate(results.pose_landmarks.landmark):
                name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"lm_{i}"
                landmarks[name] = PoseLandmark(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(lm.visibility),
                )
                total_visibility += lm.visibility

            avg_confidence = total_visibility / max(len(landmarks), 1)
            phase = self._classify_phase(landmarks, idx)

            return PoseFrame(
                frame_idx=idx,
                timestamp_ms=ts,
                landmarks=landmarks,
                shot_phase=phase,
                confidence=avg_confidence,
            )
        except Exception as e:
            logger.debug("MediaPipe processing error on frame %d: %s", idx, e)
            return None

    def _generate_synthetic_pose(self, frame_idx: int, timestamp_ms: float) -> PoseFrame:
        """
        Generate anatomically-plausible synthetic pose for demo/testing.
        Simulates a basketball shooting motion over ~60 frames.
        """
        t = (frame_idx % 90) / 90.0  # normalized shot cycle [0, 1]

        # Shooting arm (right) traces an arc from low-gather to high-release
        gather_phase = t < 0.3
        rise_phase   = 0.3 <= t < 0.6
        release_phase = 0.6 <= t < 0.75
        follow_phase  = t >= 0.75

        # Base standing pose
        base = {
            "nose": (0.50, 0.15, 0.0), "left_eye": (0.52, 0.14, 0.0), "right_eye": (0.48, 0.14, 0.0),
            "left_shoulder": (0.58, 0.28, 0.0), "right_shoulder": (0.42, 0.28, 0.0),
            "left_hip": (0.56, 0.55, 0.0), "right_hip": (0.44, 0.55, 0.0),
            "left_knee": (0.57, 0.73, 0.0), "right_knee": (0.43, 0.73, 0.0),
            "left_ankle": (0.57, 0.90, 0.0), "right_ankle": (0.43, 0.90, 0.0),
        }

        # Animate shooting arm
        if gather_phase:
            s = t / 0.3
            elbow_y = 0.45 + 0.05 * s
            wrist_y = 0.50 + 0.05 * s
        elif rise_phase:
            s = (t - 0.3) / 0.3
            elbow_y = 0.45 - 0.15 * s
            wrist_y = 0.50 - 0.20 * s
        elif release_phase:
            s = (t - 0.6) / 0.15
            elbow_y = 0.30 - 0.10 * s
            wrist_y = 0.30 - 0.15 * s
        else:
            s = (t - 0.75) / 0.25
            elbow_y = 0.20 + 0.05 * s
            wrist_y = 0.15 + 0.10 * s

        # Knee bend during gather
        knee_bend = 0.0
        if gather_phase:
            knee_bend = 0.04 * (t / 0.3)
        elif rise_phase:
            knee_bend = 0.04 * (1.0 - (t - 0.3) / 0.3)

        base.update({
            "right_elbow": (0.38, elbow_y, -0.1),
            "right_wrist": (0.35, wrist_y, -0.2),
            "right_index": (0.34, wrist_y - 0.03, -0.25),
            "left_elbow": (0.62, 0.40, 0.05),
            "left_wrist": (0.64, 0.45, 0.0),
            "right_knee": (0.43, 0.73 + knee_bend, 0.0),
            "left_knee": (0.57, 0.73 + knee_bend, 0.0),
        })

        landmarks = {}
        for name, (x, y, z) in base.items():
            landmarks[name] = PoseLandmark(x=x, y=y, z=z, visibility=0.95)

        if gather_phase:
            phase = ShotPhase.GATHER
        elif rise_phase:
            phase = ShotPhase.SETUP
        elif release_phase:
            phase = ShotPhase.RELEASE
        else:
            phase = ShotPhase.FOLLOW

        return PoseFrame(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            landmarks=landmarks,
            shot_phase=phase,
            confidence=0.85,
        )

    def _classify_phase(self, landmarks: Dict[str, PoseLandmark], frame_idx: int) -> ShotPhase:
        """
        Rule-based shot phase classifier from pose landmarks.
        Uses wrist height relative to shoulder to detect release timing.
        """
        try:
            rs = landmarks.get("right_shoulder")
            rw = landmarks.get("right_wrist")
            rk = landmarks.get("right_knee")
            rh = landmarks.get("right_hip")

            if not all([rs, rw, rk, rh]):
                return ShotPhase.UNKNOWN

            wrist_above_shoulder = rw.y < rs.y
            knee_bent = rk.y < rh.y + 0.15  # knee significantly lower than hip

            if knee_bent and not wrist_above_shoulder:
                return ShotPhase.GATHER
            elif knee_bent and wrist_above_shoulder:
                return ShotPhase.RELEASE
            elif not knee_bent and wrist_above_shoulder:
                return ShotPhase.FOLLOW
            else:
                return ShotPhase.SETUP

        except Exception:
            return ShotPhase.UNKNOWN

    def process_batch(
        self,
        frames_rgb: List[Tuple[np.ndarray, int, float]],
    ) -> List[Optional[PoseFrame]]:
        """Process a batch of (rgb_array, frame_idx, timestamp_ms) tuples."""
        return [self.process_frame(rgb, idx, ts) for rgb, idx, ts in frames_rgb]

    @property
    def skeleton_connections(self) -> List[Tuple[str, str]]:
        return SKELETON_CONNECTIONS

    def close(self):
        """Release MediaPipe resources."""
        if self._pose_model:
            try:
                self._pose_model.close()
            except Exception:
                pass
