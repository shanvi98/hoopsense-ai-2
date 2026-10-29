"""
HoopSense AI — Video Processing Service
Frame extraction, preprocessing, and shot phase segmentation using OpenCV
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Extracted metadata from input video."""
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float
    codec: str


@dataclass
class ProcessedFrame:
    """Single processed video frame with metadata."""
    frame_idx: int
    timestamp_ms: float
    bgr: np.ndarray        # raw BGR frame
    rgb: np.ndarray        # RGB for MediaPipe
    gray: np.ndarray       # grayscale for motion detection
    resized: np.ndarray    # normalized 640x640 for YOLO


class VideoProcessor:
    """
    Handles video ingestion, frame extraction, and preprocessing.
    Supports MP4, MOV, AVI, MKV, WEBM formats.
    """

    # Target frame dimensions for YOLO inference
    YOLO_INPUT_SIZE = (640, 640)

    # Shot detection via motion energy
    MOTION_THRESHOLD = 15.0
    SHOT_COOLDOWN_FRAMES = 60   # minimum frames between detected shots

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None

    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Extract and validate video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        return VideoMetadata(
            path=video_path,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration_s=total_frames / fps if fps > 0 else 0,
            codec=codec.strip(),
        )

    def validate_video(self, video_path: str) -> Tuple[bool, str]:
        """Validate file format and size constraints."""
        path = Path(video_path)
        if path.suffix.lower() not in settings.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {path.suffix}. Use {settings.SUPPORTED_FORMATS}"

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > settings.MAX_VIDEO_SIZE_MB:
            return False, f"File too large: {size_mb:.1f}MB. Max: {settings.MAX_VIDEO_SIZE_MB}MB"

        return True, "OK"

    def extract_frames(
        self,
        video_path: str,
        max_frames: int = None,
        start_frame: int = 0,
        end_frame: int = None,
        step: int = 1,
    ) -> Generator[ProcessedFrame, None, None]:
        """
        Extract and preprocess frames as a lazy generator.
        Memory-efficient — yields one frame at a time.
        """
        max_frames = max_frames or settings.MAX_FRAMES_EXTRACT
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = end_frame or total

        # Seek to start
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        extracted = 0

        try:
            while cap.isOpened() and extracted < max_frames:
                ret, bgr = cap.read()
                if not ret:
                    break

                actual_idx = start_frame + frame_count
                if actual_idx >= end_frame:
                    break

                # Apply step stride
                if frame_count % step == 0:
                    yield self._preprocess_frame(bgr, actual_idx, actual_idx * 1000.0 / fps)
                    extracted += 1

                frame_count += 1

        finally:
            cap.release()

        logger.info(f"Extracted {extracted} frames from {Path(video_path).name}")

    def _preprocess_frame(self, bgr: np.ndarray, idx: int, ts_ms: float) -> ProcessedFrame:
        """Preprocess a single BGR frame for all downstream modules."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Letterbox resize for YOLO (preserves aspect ratio)
        resized = self._letterbox(bgr, self.YOLO_INPUT_SIZE)

        return ProcessedFrame(
            frame_idx=idx,
            timestamp_ms=ts_ms,
            bgr=bgr,
            rgb=rgb,
            gray=gray,
            resized=resized,
        )

    def _letterbox(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image with letterboxing to preserve aspect ratio."""
        h, w = img.shape[:2]
        th, tw = target_size
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        pad_x = (tw - new_w) // 2
        pad_y = (th - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return canvas

    def detect_shot_segments(self, video_path: str) -> List[Tuple[int, int]]:
        """
        Detect shot attempt segments via optical-flow motion energy.
        Returns list of (start_frame, end_frame) tuples.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        prev_gray = None
        motion_scores = []
        frame_idx = 0

        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_scores.append((frame_idx, float(np.mean(mag))))
            else:
                motion_scores.append((frame_idx, 0.0))

            prev_gray = gray
            frame_idx += 1

        cap.release()

        # Find high-motion windows (shot attempts)
        segments = []
        in_shot = False
        shot_start = 0
        cooldown = 0

        for fidx, score in motion_scores:
            if cooldown > 0:
                cooldown -= 1
                continue

            if not in_shot and score > self.MOTION_THRESHOLD:
                in_shot = True
                shot_start = max(0, fidx - int(fps * 0.5))  # 0.5s pre-roll

            elif in_shot and score < self.MOTION_THRESHOLD * 0.5:
                in_shot = False
                shot_end = min(len(motion_scores) - 1, fidx + int(fps * 1.0))  # 1s post-roll
                duration = (shot_end - shot_start) / fps
                if 0.5 <= duration <= 5.0:  # valid shot window
                    segments.append((shot_start, shot_end))
                    cooldown = self.SHOT_COOLDOWN_FRAMES

        if not segments:
            # Fallback: use entire video as single shot segment
            total = len(motion_scores)
            segments = [(0, total)]
            logger.warning("No shot segments detected via motion; using full video")

        logger.info(f"Detected {len(segments)} shot segment(s)")
        return segments

    def draw_skeleton_overlay(
        self,
        frame: np.ndarray,
        landmarks: dict,
        connections: List[Tuple[str, str]],
        ball_pos: Optional[Tuple[int, int]] = None,
        hoop_pos: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Draw pose skeleton, ball, and hoop overlays onto a frame copy."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        SKELETON_COLOR = (0, 255, 128)    # neon green
        JOINT_COLOR    = (255, 200, 0)    # amber
        BALL_COLOR     = (0, 140, 255)    # orange-blue
        HOOP_COLOR     = (255, 50, 50)    # red

        # Draw connections
        for (lm_a, lm_b) in connections:
            if lm_a in landmarks and lm_b in landmarks:
                a = landmarks[lm_a]
                b = landmarks[lm_b]
                if a.get("visibility", 0) > 0.4 and b.get("visibility", 0) > 0.4:
                    pt1 = (int(a["x"] * w), int(a["y"] * h))
                    pt2 = (int(b["x"] * w), int(b["y"] * h))
                    cv2.line(overlay, pt1, pt2, SKELETON_COLOR, 2, cv2.LINE_AA)

        # Draw joints
        for name, lm in landmarks.items():
            if lm.get("visibility", 0) > 0.4:
                pt = (int(lm["x"] * w), int(lm["y"] * h))
                cv2.circle(overlay, pt, 4, JOINT_COLOR, -1, cv2.LINE_AA)
                cv2.circle(overlay, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw ball
        if ball_pos:
            cv2.circle(overlay, ball_pos, 18, BALL_COLOR, 2, cv2.LINE_AA)
            cv2.circle(overlay, ball_pos, 4, BALL_COLOR, -1, cv2.LINE_AA)

        # Draw hoop
        if hoop_pos:
            x1, y1, x2, y2 = hoop_pos
            cv2.rectangle(overlay, (x1, y1), (x2, y2), HOOP_COLOR, 2, cv2.LINE_AA)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(overlay, (cx, cy), 4, HOOP_COLOR, -1)

        # Blend overlay for semi-transparency
        alpha = 0.85
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    @staticmethod
    def frames_to_video(
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
    ) -> str:
        """Write processed frames to an MP4 output video."""
        if not frames:
            raise ValueError("No frames to write")

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()
        return output_path
