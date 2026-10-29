"""
HoopSense AI — Analysis Orchestration Service
Coordinates video processing → pose estimation → object detection → features → inference
Handles both real video files and __demo__ synthetic mode
"""
import time
import uuid
import logging
import os
from typing import Optional
from app.core.config import settings
from app.models.schemas import ShotPrediction, AnalysisStatus
from app.services.video_processor import VideoProcessor
from app.services.pose_estimator import PoseEstimator
from app.services.object_detector import ObjectDetector
from app.services.feature_engineer import FeatureEngineer
from app.services.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)
_sessions: dict = {}


class AnalysisService:

    def __init__(self):
        self._video_proc = VideoProcessor()
        self._pose_est   = PoseEstimator()
        self._detector   = ObjectDetector()
        self._features   = FeatureEngineer()
        self._inference  = InferenceEngine()

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        _sessions[sid] = {"status": AnalysisStatus.PENDING, "result": None, "error": None, "elapsed": 0}
        return sid

    def get_session(self, session_id: str) -> Optional[dict]:
        return _sessions.get(session_id)

    async def analyze_video(self, video_path: str, session_id: str) -> ShotPrediction:
        t_start = time.time()
        _sessions[session_id]["status"] = AnalysisStatus.PROCESSING

        try:
            if video_path == "__demo__":
                pose_frames, detections, fps, height, n_frames, duration = self._generate_demo_data()
            else:
                valid, msg = self._video_proc.validate_video(video_path)
                if not valid:
                    raise ValueError(msg)
                meta = self._video_proc.get_metadata(video_path)
                fps, height, n_frames, duration = meta.fps, meta.height, meta.total_frames, meta.duration_s
                segments = self._video_proc.detect_shot_segments(video_path)
                start_f, end_f = segments[0]
                frames = list(self._video_proc.extract_frames(
                    video_path, max_frames=settings.MAX_FRAMES_EXTRACT,
                    start_frame=start_f, end_frame=end_f
                ))
                pose_frames, detections = [], []
                for frame in frames:
                    pf = self._pose_est.process_frame(frame.rgb, frame.frame_idx, frame.timestamp_ms)
                    det = self._detector.detect_frame(frame.bgr, frame.frame_idx, frame.timestamp_ms)
                    if pf: pose_frames.append(pf)
                    detections.append(det)

            trajectory  = self._detector.build_trajectory(detections)
            features    = self._features.extract(pose_frames, detections, trajectory, fps, height)
            form_score, make_prob, confidence = self._inference.predict(features)
            form_grade  = InferenceEngine.score_to_grade(form_score)
            feedback    = self._inference.generate_feedback(features)

            ball_traj = [{"x": p.x, "y": p.y, "frame": p.frame_idx, "vx": p.vx, "vy": p.vy}
                         for p in trajectory]

            pose_sparse = []
            for pf in pose_frames[::5]:
                det = next((d for d in detections if d.frame_idx == pf.frame_idx), None)
                pose_sparse.append({
                    "frame_idx": pf.frame_idx,
                    "timestamp_ms": pf.timestamp_ms,
                    "phase": pf.shot_phase.value,
                    "landmarks": {k: {"x": v.x, "y": v.y, "visibility": v.visibility}
                                  for k, v in pf.landmarks.items()},
                    "ball": {"x": det.ball.x_center, "y": det.ball.y_center} if (det and det.ball) else None,
                })

            hoop_pos = None
            for det in detections:
                if det.hoop:
                    hoop_pos = {"x": det.hoop.x_center, "y": det.hoop.y_center,
                                "w": det.hoop.width, "h": det.hoop.height}
                    break

            phase_counts = {}
            for pf in pose_frames:
                phase_counts[pf.shot_phase.value] = phase_counts.get(pf.shot_phase.value, 0) + 1

            result = ShotPrediction(
                session_id=session_id,
                status=AnalysisStatus.COMPLETED,
                form_score=form_score,
                form_grade=form_grade,
                make_probability=make_prob,
                confidence=confidence,
                features=features,
                feedback=feedback,
                ball_trajectory=ball_traj,
                pose_frames=pose_sparse,
                hoop_position=hoop_pos,
                video_fps=fps,
                video_duration_s=duration,
                total_frames=n_frames,
                shot_phase_frames=phase_counts,
                model_version=settings.MODEL_VERSION,
            )

            elapsed = round((time.time() - t_start) * 1000)
            _sessions[session_id].update({"status": AnalysisStatus.COMPLETED, "result": result, "elapsed": elapsed})
            logger.info("Done: form=%.1f make=%.0f%% [%dms]", form_score, make_prob*100, elapsed)
            return result

        except Exception as e:
            logger.exception("Pipeline failed for %s", session_id)
            _sessions[session_id].update({"status": AnalysisStatus.FAILED, "error": str(e)})
            raise
        finally:
            if video_path != "__demo__" and os.path.exists(video_path):
                try: os.remove(video_path)
                except: pass

    def _generate_demo_data(self):
        """Generate synthetic shot sequence for demo mode (90 frames at 30fps)."""
        pose_frames, detections = [], []
        for i in range(90):
            pf  = self._pose_est.process_frame(None, i, i * 1000.0 / 30)
            # For demo, use pose estimator's synthetic mode
            if pf is None:
                pf = self._pose_est._generate_synthetic_pose(i, i * 1000.0 / 30)
            det = self._detector._generate_synthetic_detection(i, i * 1000.0 / 30, (720, 1280, 3))
            pose_frames.append(pf)
            detections.append(det)
        return pose_frames, detections, 30.0, 720, 90, 3.0
