"""
HoopSense AI — Inference Engine
XGBoost + Scikit-learn ensemble for form scoring and shot success prediction
"""
import numpy as np
import logging
import os
import pickle
from typing import Tuple, List, Dict, Optional
from app.models.schemas import BiomechanicalFeatures, FeedbackItem, FormGrade
from app.services.feature_engineer import FeatureEngineer
from app.core.config import settings

logger = logging.getLogger(__name__)

# Biomechanical benchmarks for feedback generation
BENCHMARKS = {
    "elbow_angle_release": (80, 110, "Elbow angle at release"),
    "knee_bend_max": (20, 60, "Knee bend during gather"),
    "wrist_release_angle": (30, 70, "Wrist extension at release"),
    "ball_arc_angle": (38, 55, "Shot arc angle"),
    "release_timing_score": (0.6, 1.0, "Release timing at jump peak"),
    "follow_through_score": (0.7, 1.0, "Follow-through completeness"),
    "balance_score": (0.7, 1.0, "Body balance during shot"),
    "rhythm_consistency": (0.6, 1.0, "Gather-to-release rhythm"),
    "shooting_arm_extension": (0.8, 1.0, "Arm extension at release"),
    "hip_shoulder_alignment": (0.7, 1.0, "Hip-shoulder alignment"),
}

FEEDBACK_MESSAGES = {
    "elbow_angle_release": {
        "low":  "Elbow angle too closed at release — aim for 80-110° for optimal power transfer",
        "good": "Great elbow angle at release — solid mechanical efficiency",
        "high": "Elbow is too open at release — tuck it in for better ball control",
    },
    "knee_bend_max": {
        "low":  "Not enough knee bend during gather — use your legs for more power",
        "good": "Good leg drive — knee bend is in the optimal range",
        "high": "Excessive knee bend adds unnecessary complexity — moderate squat is ideal",
    },
    "wrist_release_angle": {
        "low":  "Wrist isn't snapping fully — extend through the ball for better backspin",
        "good": "Clean wrist snap at release — good backspin mechanics",
        "high": "Over-flicking the wrist — keep it controlled for consistency",
    },
    "ball_arc_angle": {
        "low":  "Shot arc is too flat — aim higher (38-55°) for better hoop margin",
        "good": "Excellent shot arc — ideal angle for maximizing scoring window",
        "high": "Arc is too high — reduces power efficiency and wind resistance",
    },
    "release_timing_score": {
        "low":  "Releasing too early or late — sync your release with peak jump height",
        "good": "Perfect release timing at jump apex — maximizing shot height advantage",
        "high": "Release timing is optimal",
    },
    "follow_through_score": {
        "low":  "Incomplete follow-through — hold the 'gooseneck' until the ball hits the rim",
        "good": "Excellent follow-through — classic mechanics for consistent shooting",
        "high": "Follow-through looks solid",
    },
    "balance_score": {
        "low":  "Landing off-balance — stay centered over your feet throughout the shot",
        "good": "Balanced shot — feet aligned with hips throughout the motion",
        "high": "Balance score is excellent",
    },
    "rhythm_consistency": {
        "low":  "Jerky motion detected — aim for a smooth, uninterrupted gather-to-release",
        "good": "Fluid shot rhythm — consistent timing from gather to release",
        "high": "Rhythm is optimal",
    },
    "shooting_arm_extension": {
        "low":  "Arm not fully extending — reach through the shot for better arc",
        "good": "Full arm extension — maximizing release height",
        "high": "Arm extension is excellent",
    },
    "hip_shoulder_alignment": {
        "low":  "Hips and shoulders misaligned — square up to the basket for accuracy",
        "good": "Body aligned to the basket — solid shooting posture",
        "high": "Alignment is excellent",
    },
}


class InferenceEngine:
    """
    Two-model ensemble:
    - form_model: predicts biomechanical form score (0-100)
    - success_model: predicts shot make probability (0-1)
    Falls back to rule-based scoring when ML models aren't trained yet.
    """

    def __init__(self):
        self._form_model = None
        self._success_model = None
        self._scaler = None
        self._feature_eng = FeatureEngineer()
        self._model_version = settings.MODEL_VERSION
        self._load_models()

    def _load_models(self):
        form_path    = os.path.join(settings.MODEL_DIR, "form_model.pkl")
        success_path = os.path.join(settings.MODEL_DIR, "success_model.pkl")
        scaler_path  = os.path.join(settings.MODEL_DIR, "scaler.pkl")
        if os.path.exists(form_path):
            with open(form_path, "rb") as f:
                self._form_model = pickle.load(f)
            logger.info("Form model loaded from %s", form_path)
        if os.path.exists(success_path):
            with open(success_path, "rb") as f:
                self._success_model = pickle.load(f)
            logger.info("Success model loaded from %s", success_path)
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)

    def predict(self, features: BiomechanicalFeatures) -> Tuple[float, float, float]:
        """
        Returns (form_score 0-100, make_probability 0-1, confidence 0-1).
        """
        x = self._feature_eng.to_array(features).reshape(1, -1)

        if self._scaler:
            x = self._scaler.transform(x)

        if self._form_model and self._success_model:
            form_score    = float(np.clip(self._form_model.predict(x)[0], 0, 100))
            make_prob_raw = self._success_model.predict_proba(x)[0]
            make_prob     = float(make_prob_raw[1]) if len(make_prob_raw) > 1 else float(make_prob_raw[0])
            confidence    = 0.88
        else:
            form_score, make_prob = self._rule_based_score(features)
            confidence = 0.72  # lower confidence for heuristic scoring

        return round(form_score, 1), round(make_prob, 3), round(confidence, 3)

    def _rule_based_score(self, f: BiomechanicalFeatures) -> Tuple[float, float]:
        """
        Weighted rule-based biomechanical scoring.
        Used when ML models haven't been trained yet.
        """
        def score_range(val, lo, hi, weight):
            if lo <= val <= hi:
                return weight
            dist = min(abs(val - lo), abs(val - hi))
            penalty = dist / max(hi - lo, 1) * weight
            return max(0, weight - penalty)

        weights = {
            "elbow_angle_release":    (f.elbow_angle_release,   80,  110, 15),
            "knee_bend_max":          (f.knee_bend_max,          20,   60, 10),
            "wrist_release_angle":    (f.wrist_release_angle,    30,   70, 10),
            "ball_arc_angle":         (f.ball_arc_angle,         38,   55, 15),
            "release_timing_score":   (f.release_timing_score,  0.6,  1.0, 12),
            "follow_through_score":   (f.follow_through_score,  0.7,  1.0, 12),
            "balance_score":          (f.balance_score,         0.7,  1.0,  8),
            "rhythm_consistency":     (f.rhythm_consistency,    0.6,  1.0,  8),
            "shooting_arm_extension": (f.shooting_arm_extension,0.8, 1.0,  5),
            "hip_shoulder_alignment": (f.hip_shoulder_alignment,0.7, 1.0,  5),
        }

        form_score = sum(score_range(val, lo, hi, w) for val, lo, hi, w in weights.values())

        # Shot success probability from arc + release + distance
        arc_ok  = 1.0 if 38 <= f.ball_arc_angle <= 55 else max(0, 1 - abs(f.ball_arc_angle - 46.5)/20)
        dist_ok = float(np.clip(1.0 - (f.player_distance_hoop - 15)**2 / 200.0, 0.3, 1.0))
        form_ok = form_score / 100.0
        make_prob = float(np.clip((arc_ok * 0.35 + form_ok * 0.45 + dist_ok * 0.20) * 0.95, 0.05, 0.95))

        return float(form_score), make_prob

    def generate_feedback(self, features: BiomechanicalFeatures) -> List[FeedbackItem]:
        """Generate actionable biomechanical feedback sorted by impact."""
        feedback = []
        f_dict = features.model_dump()

        for key, (lo, hi, label) in BENCHMARKS.items():
            val = f_dict.get(key, 0.0)
            msgs = FEEDBACK_MESSAGES.get(key, {})

            if val < lo:
                rating, msg = "warning" if (lo - val) < (hi - lo) * 0.5 else "error", msgs.get("low", "")
            elif val > hi:
                rating, msg = "warning", msgs.get("high", "")
            else:
                rating, msg = "good", msgs.get("good", "")

            if msg:
                feedback.append(FeedbackItem(
                    aspect=label,
                    rating=rating,
                    message=msg,
                    metric_value=round(float(val), 3),
                    benchmark_range=f"{lo}–{hi}",
                ))

        # Sort: errors first, then warnings, then good
        order = {"error": 0, "warning": 1, "good": 2}
        feedback.sort(key=lambda x: order.get(x.rating, 3))
        return feedback

    @staticmethod
    def score_to_grade(score: float) -> FormGrade:
        if score >= 85: return FormGrade.ELITE
        if score >= 70: return FormGrade.GOOD
        if score >= 50: return FormGrade.AVERAGE
        return FormGrade.POOR
