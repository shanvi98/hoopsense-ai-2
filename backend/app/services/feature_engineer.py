"""
HoopSense AI — Feature Engineering
Computes 20 biomechanical features from pose + ball detections
"""
import numpy as np
import logging
from typing import List, Optional, Tuple
from app.models.schemas import BiomechanicalFeatures, PoseFrame, ShotPhase
from app.services.object_detector import DetectionResult, TrajectoryPoint

logger = logging.getLogger(__name__)


def _angle_between(a, b, c):
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-9: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))))


def _lm2d(landmarks, name):
    lm = landmarks.get(name)
    return np.array([lm.x, lm.y]) if lm else None


def _pts(landmarks, *names):
    pts = [_lm2d(landmarks, n) for n in names]
    return tuple(pts) if all(p is not None for p in pts) else None


class FeatureEngineer:

    def extract(self, pose_frames, detections, trajectory, video_fps=30.0, frame_height=720):
        if not pose_frames:
            return BiomechanicalFeatures()

        pose_by_frame = {pf.frame_idx: pf for pf in pose_frames}
        det_by_frame = {d.frame_idx: d for d in detections}

        release_frame = next((pf.frame_idx for pf in pose_frames if pf.shot_phase == ShotPhase.RELEASE), pose_frames[len(pose_frames)//2].frame_idx)
        gather_frame  = next((pf.frame_idx for pf in pose_frames if pf.shot_phase == ShotPhase.GATHER), pose_frames[0].frame_idx)
        follow_frame  = next((pf.frame_idx for pf in reversed(pose_frames) if pf.shot_phase == ShotPhase.FOLLOW), pose_frames[-1].frame_idx)

        rp = pose_by_frame.get(release_frame)
        gp = pose_by_frame.get(gather_frame)
        fp = pose_by_frame.get(follow_frame)

        arc, v_rel, v_peak, smooth = self._traj_features(trajectory, video_fps)

        rel_det = det_by_frame.get(release_frame)
        hoop = rel_det.hoop if rel_det else None
        bbox = rel_det.player_bbox if rel_det else None

        from app.services.object_detector import ObjectDetector
        dist = ObjectDetector().estimate_distance_to_hoop(bbox, hoop, frame_height)

        return BiomechanicalFeatures(
            elbow_angle_release   = round(self._elbow_angle(rp) if rp else 90.0, 2),
            elbow_angle_min       = round(self._elbow_min(pose_frames), 2),
            knee_bend_max         = round(self._knee_max(pose_frames), 2),
            wrist_release_angle   = round(self._wrist_angle(rp) if rp else 45.0, 2),
            hip_shoulder_alignment= round(self._hip_sh_align(rp) if rp else 0.85, 3),
            trunk_lean_angle      = round(self._trunk_lean(rp) if rp else 10.0, 2),
            shooting_arm_extension= round(self._arm_ext(rp) if rp else 0.75, 3),
            release_height_normalized = round(self._release_height(rp, gp) if (rp and gp) else 0.7, 3),
            ball_arc_angle        = round(arc, 2),
            ball_velocity_release = round(v_rel, 4),
            ball_velocity_peak    = round(v_peak, 4),
            trajectory_smoothness = round(smooth, 3),
            shot_duration_frames  = int(abs(release_frame - gather_frame)),
            jump_height_normalized= round(self._jump_height(pose_frames), 3),
            release_timing_score  = round(self._release_timing(pose_frames, release_frame), 3),
            rhythm_consistency    = round(self._rhythm(pose_frames, gather_frame, release_frame), 3),
            player_distance_hoop  = round(dist, 2),
            shot_angle_degrees    = round(self._shot_angle(bbox, hoop) if (bbox and hoop) else 0.0, 2),
            balance_score         = round(self._balance(pose_frames), 3),
            follow_through_score  = round(self._follow(fp) if fp else 0.65, 3),
        )

    def _elbow_angle(self, pf, side="right"):
        r = _pts(pf.landmarks, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        return _angle_between(*r) if r else 90.0

    def _elbow_min(self, frames):
        return float(np.min([self._elbow_angle(pf) for pf in frames])) if frames else 90.0

    def _knee_max(self, frames, side="right"):
        bends = []
        for pf in frames:
            r = _pts(pf.landmarks, f"{side}_hip", f"{side}_knee", f"{side}_ankle")
            if r: bends.append(180.0 - _angle_between(*r))
        return float(np.max(bends)) if bends else 0.0

    def _wrist_angle(self, pf, side="right"):
        r = _pts(pf.landmarks, f"{side}_elbow", f"{side}_wrist", f"{side}_index")
        return _angle_between(*r) if r else 45.0

    def _hip_sh_align(self, pf):
        sh = _pts(pf.landmarks, "left_shoulder", "right_shoulder")
        hp = _pts(pf.landmarks, "left_hip", "right_hip")
        if not sh or not hp: return 0.85
        sa = abs(np.degrees(np.arctan2(sh[0][1]-sh[1][1], sh[0][0]-sh[1][0])))
        ha = abs(np.degrees(np.arctan2(hp[0][1]-hp[1][1], hp[0][0]-hp[1][0])))
        return float(np.clip(1.0 - abs(sa-ha)/45.0, 0.0, 1.0))

    def _trunk_lean(self, pf):
        r = _pts(pf.landmarks, "left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if not r: return 10.0
        sh = (r[0]+r[1])/2; hp = (r[2]+r[3])/2
        return float(abs(np.degrees(np.arctan2(sh[0]-hp[0], hp[1]-sh[1]))))

    def _arm_ext(self, pf, side="right"):
        r = _pts(pf.landmarks, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        if not r: return 0.75
        return float(np.clip(np.linalg.norm(r[2]-r[0]) / (np.linalg.norm(r[1]-r[0])+np.linalg.norm(r[2]-r[1])+1e-9), 0.0, 1.0))

    def _release_height(self, rp, gp):
        rw = rp.landmarks.get("right_wrist"); gw = gp.landmarks.get("right_wrist")
        return float(np.clip((gw.y - rw.y) / 0.5, 0.0, 1.0)) if (rw and gw) else 0.7

    def _traj_features(self, traj, fps):
        if len(traj) < 3: return 45.0, 0.02, 0.03, 0.7
        vxs = np.array([p.vx for p in traj]); vys = np.array([p.vy for p in traj])
        speeds = np.sqrt(vxs**2 + vys**2)
        n = max(1, len(traj)//5)
        arc = float(abs(np.degrees(np.arctan2(-np.mean(vys[:n]), abs(np.mean(vxs[:n]))+1e-9))))
        v_rel = float(speeds[0]) if speeds.size else 0.02
        v_peak = float(np.max(speeds)) if speeds.size else 0.03
        if len(speeds) >= 4:
            jerk = np.diff(np.diff(speeds))
            smooth = float(np.clip(1.0 - np.std(jerk)/(np.mean(speeds)+1e-9)*5, 0.0, 1.0))
        else:
            smooth = 0.7
        return arc, v_rel, v_peak, smooth

    def _jump_height(self, frames):
        ays = [(pf.landmarks.get("left_ankle").y + pf.landmarks.get("right_ankle").y)/2
               for pf in frames if pf.landmarks.get("left_ankle") and pf.landmarks.get("right_ankle")]
        return float(np.clip(np.percentile(ays, 90) - np.min(ays), 0.0, 0.3)) if len(ays) > 2 else 0.05

    def _release_timing(self, frames, release_frame):
        if release_frame is None: return 0.5
        ays = [(pf.frame_idx, (pf.landmarks.get("left_ankle").y + pf.landmarks.get("right_ankle").y)/2)
               for pf in frames if pf.landmarks.get("left_ankle") and pf.landmarks.get("right_ankle")]
        if not ays: return 0.5
        peak_idx = min(ays, key=lambda t: t[1])[0]
        return float(np.clip(1.0 - abs(release_frame - peak_idx)/15.0, 0.0, 1.0))

    def _rhythm(self, frames, g, r):
        if g is None or r is None: return 0.7
        window = [pf for pf in frames if g <= pf.frame_idx <= r]
        wys = [pf.landmarks.get("right_wrist").y for pf in window if pf.landmarks.get("right_wrist")]
        if len(wys) < 4: return 0.7
        sc = int(np.sum(np.diff(np.sign(np.diff(wys))) != 0))
        return float(np.clip(1.0 - sc/6.0, 0.0, 1.0))

    def _shot_angle(self, bbox, hoop):
        cx = (bbox[0]+bbox[2])/2
        return float(abs(np.degrees(np.arctan2(abs(cx - hoop.x_center), abs(bbox[1] - hoop.y_center)+1e-9))))

    def _balance(self, frames):
        scores = []
        for pf in frames:
            r = _pts(pf.landmarks, "left_ankle", "right_ankle", "left_hip", "right_hip")
            if r: scores.append(1.0 - min(abs((r[0][0]+r[1][0])/2 - (r[2][0]+r[3][0])/2)*4, 1.0))
        return float(np.mean(scores)) if scores else 0.8

    def _follow(self, pf, side="right"):
        rsh = pf.landmarks.get(f"{side}_shoulder"); rw = pf.landmarks.get(f"{side}_wrist")
        if not (rsh and rw): return 0.65
        above = rsh.y - rw.y
        ext = self._arm_ext(pf, side)
        return float(np.clip((above*3 + ext)/4, 0.0, 1.0))

    def to_array(self, f: BiomechanicalFeatures):
        return np.array([f.elbow_angle_release, f.elbow_angle_min, f.knee_bend_max,
            f.wrist_release_angle, f.hip_shoulder_alignment, f.trunk_lean_angle,
            f.shooting_arm_extension, f.release_height_normalized, f.ball_arc_angle,
            f.ball_velocity_release, f.ball_velocity_peak, f.trajectory_smoothness,
            f.shot_duration_frames, f.jump_height_normalized, f.release_timing_score,
            f.rhythm_consistency, f.player_distance_hoop, f.shot_angle_degrees,
            f.balance_score, f.follow_through_score], dtype=np.float32)

    @staticmethod
    def feature_names():
        return ["elbow_angle_release","elbow_angle_min","knee_bend_max","wrist_release_angle",
                "hip_shoulder_alignment","trunk_lean_angle","shooting_arm_extension",
                "release_height_normalized","ball_arc_angle","ball_velocity_release",
                "ball_velocity_peak","trajectory_smoothness","shot_duration_frames",
                "jump_height_normalized","release_timing_score","rhythm_consistency",
                "player_distance_hoop","shot_angle_degrees","balance_score","follow_through_score"]
