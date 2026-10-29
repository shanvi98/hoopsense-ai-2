"""
HoopSense AI — Object Detection Service
YOLOv8-based detection and tracking of basketball, hoop, and player
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from app.models.schemas import BallDetection, HoopDetection
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    frame_idx: int
    timestamp_ms: float
    ball: Optional[BallDetection] = None
    hoop: Optional[HoopDetection] = None
    player_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class TrajectoryPoint:
    frame_idx: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.0


class ObjectDetector:
    """YOLOv8 detection + tracking for basketball scenes."""

    CLASS_PERSON      = 0
    CLASS_SPORTS_BALL = 32

    def __init__(self):
        self._model = None
        self._available = False
        self._prev_ball: Optional[BallDetection] = None
        self._initialize()

    def _initialize(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(settings.YOLO_MODEL)
            self._available = True
            logger.info("YOLOv8 initialized: %s", settings.YOLO_MODEL)
        except Exception as e:
            logger.warning("YOLO unavailable: %s — real video analysis disabled (demo mode only)", e)

    @property
    def available(self) -> bool:
        return self._available and self._model is not None

    def detect_frame(self, bgr_frame: np.ndarray, frame_idx: int, timestamp_ms: float) -> DetectionResult:
        if not self.available:
            raise RuntimeError("YOLO is not available — cannot run object detection on real video")
        return self._detect_with_yolo(bgr_frame, frame_idx, timestamp_ms)

    def _detect_with_yolo(self, bgr: np.ndarray, idx: int, ts: float) -> DetectionResult:
        try:
            results = self._model(bgr, conf=settings.YOLO_CONFIDENCE, iou=settings.YOLO_IOU_THRESHOLD, verbose=False)
            h, w = bgr.shape[:2]
            result = DetectionResult(frame_idx=idx, timestamp_ms=ts)

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                if cls_id == self.CLASS_SPORTS_BALL and result.ball is None:
                    vx = cx - self._prev_ball.x_center if self._prev_ball else 0.0
                    vy = cy - self._prev_ball.y_center if self._prev_ball else 0.0
                    ball = BallDetection(frame_idx=idx, timestamp_ms=ts,
                                        x_center=cx, y_center=cy, width=bw, height=bh,
                                        confidence=conf, velocity_x=vx, velocity_y=vy)
                    result.ball = ball
                    self._prev_ball = ball

                elif cls_id == self.CLASS_PERSON and result.player_bbox is None:
                    result.player_bbox = (x1/w, y1/h, x2/w, y2/h)

            if result.hoop is None:
                result.hoop = self._detect_hoop_heuristic(bgr, idx)

            return result
        except Exception as e:
            logger.debug("YOLO error frame %d: %s", idx, e)
            return DetectionResult(frame_idx=idx, timestamp_ms=ts)

    def _detect_hoop_heuristic(self, bgr: np.ndarray, idx: int) -> Optional[HoopDetection]:
        """Orange color segmentation to find hoop rim."""
        import cv2
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([25, 255, 255]))
        mask[int(h * 0.6):] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best, best_score = None, 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            score = area * (cw / max(ch, 1))
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)
        if best:
            x, y, bw, bh = best
            return HoopDetection(frame_idx=idx, x_center=(x+bw/2)/w,
                                  y_center=(y+bh/2)/h, width=bw/w, height=bh/h, confidence=0.5)
        return None

    def _generate_synthetic_detection(self, frame_idx: int, timestamp_ms: float, shape) -> DetectionResult:
        """Simulate parabolic ball trajectory for demo mode."""
        t = (frame_idx % 90) / 90.0
        if t < 0.5:
            arc_t = t / 0.5
            bx = 0.65 - 0.30 * arc_t
            by = 0.75 - 0.50 * arc_t + 0.35 * (arc_t ** 2)
        else:
            bx = 0.35 + 0.30 * ((t - 0.5) / 0.5)
            by = 0.60 + 0.15 * ((t - 0.5) / 0.5)

        ball = BallDetection(frame_idx=frame_idx, timestamp_ms=timestamp_ms,
                              x_center=bx, y_center=by, width=0.04, height=0.04,
                              confidence=0.85, velocity_x=-0.006 if t < 0.5 else 0.006,
                              velocity_y=-0.010 + 0.014 * t if t < 0.5 else 0.003)
        hoop = HoopDetection(frame_idx=frame_idx, x_center=0.28, y_center=0.22,
                              width=0.10, height=0.04, confidence=0.90)
        return DetectionResult(frame_idx=frame_idx, timestamp_ms=timestamp_ms,
                                ball=ball, hoop=hoop, player_bbox=(0.45, 0.30, 0.75, 0.95))

    def build_trajectory(self, detections: List[DetectionResult]) -> List[TrajectoryPoint]:
        """Build smoothed ball trajectory from detections."""
        raw_x, raw_y, raw_frames = [], [], []
        for det in detections:
            if det.ball and det.ball.confidence > 0.3:
                raw_x.append(det.ball.x_center)
                raw_y.append(det.ball.y_center)
                raw_frames.append(det.frame_idx)
        if len(raw_x) < 5:
            return [TrajectoryPoint(frame_idx=d.frame_idx, x=d.ball.x_center, y=d.ball.y_center, confidence=d.ball.confidence)
                    for d in detections if d.ball]
        try:
            from scipy.signal import savgol_filter
            win = min(11, len(raw_x) if len(raw_x) % 2 == 1 else len(raw_x) - 1)
            sx = savgol_filter(raw_x, win, 3)
            sy = savgol_filter(raw_y, win, 3)
        except ImportError:
            sx, sy = np.array(raw_x), np.array(raw_y)
        vx = np.gradient(sx)
        vy = np.gradient(sy)
        return [TrajectoryPoint(frame_idx=raw_frames[i], x=float(sx[i]), y=float(sy[i]),
                                 vx=float(vx[i]), vy=float(vy[i]), confidence=0.85)
                for i in range(len(sx))]

    def estimate_distance_to_hoop(self, player_bbox, hoop, frame_height: int) -> float:
        if not player_bbox or not hoop:
            return 15.0
        player_height_px = (player_bbox[3] - player_bbox[1]) * frame_height
        if player_height_px <= 0:
            return 15.0
        px_per_ft = player_height_px / 6.5
        dx_ft = abs((player_bbox[0] + player_bbox[2]) / 2 - hoop.x_center) * frame_height / px_per_ft
        return max(3.0, min(30.0, dx_ft))
