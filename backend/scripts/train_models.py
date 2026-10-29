#!/usr/bin/env python3
"""
HoopSense AI — Model Training Script
Trains XGBoost form scorer + shot success classifier with evaluation metrics.

Usage:
  python scripts/train_models.py --data data/processed/shots.csv --output data/models/

Dataset CSV columns:
  elbow_angle_release, elbow_angle_min, knee_bend_max, wrist_release_angle,
  hip_shoulder_alignment, trunk_lean_angle, shooting_arm_extension,
  release_height_normalized, ball_arc_angle, ball_velocity_release,
  ball_velocity_peak, trajectory_smoothness, shot_duration_frames,
  jump_height_normalized, release_timing_score, rhythm_consistency,
  player_distance_hoop, shot_angle_degrees, balance_score, follow_through_score,
  form_score (0-100),  made_shot (0 or 1)
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, r2_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings
from app.ml.model_registry import register_model

FEATURE_COLS = [
    "elbow_angle_release","elbow_angle_min","knee_bend_max","wrist_release_angle",
    "hip_shoulder_alignment","trunk_lean_angle","shooting_arm_extension",
    "release_height_normalized","ball_arc_angle","ball_velocity_release",
    "ball_velocity_peak","trajectory_smoothness","shot_duration_frames",
    "jump_height_normalized","release_timing_score","rhythm_consistency",
    "player_distance_hoop","shot_angle_degrees","balance_score","follow_through_score",
]
FORM_COL    = "form_score"
SUCCESS_COL = "made_shot"


def generate_synthetic_dataset(n=2000, seed=42):
    """Generate realistic synthetic training data for bootstrapping."""
    rng = np.random.default_rng(seed)
    data = {}

    # Elbow angles: good shots cluster around 90-100°
    data["elbow_angle_release"]    = rng.normal(95, 15, n).clip(50, 150)
    data["elbow_angle_min"]        = rng.normal(75, 12, n).clip(40, 130)
    data["knee_bend_max"]          = rng.normal(35, 12, n).clip(5, 80)
    data["wrist_release_angle"]    = rng.normal(50, 12, n).clip(15, 90)
    data["hip_shoulder_alignment"] = rng.beta(5, 2, n)
    data["trunk_lean_angle"]       = rng.normal(12, 6, n).clip(0, 35)
    data["shooting_arm_extension"] = rng.beta(6, 2, n)
    data["release_height_normalized"] = rng.beta(4, 2, n)
    data["ball_arc_angle"]         = rng.normal(46, 8, n).clip(20, 80)
    data["ball_velocity_release"]  = rng.normal(0.025, 0.008, n).clip(0.005, 0.06)
    data["ball_velocity_peak"]     = rng.normal(0.035, 0.01, n).clip(0.01, 0.08)
    data["trajectory_smoothness"]  = rng.beta(5, 2, n)
    data["shot_duration_frames"]   = rng.normal(28, 8, n).clip(10, 60).astype(int)
    data["jump_height_normalized"] = rng.beta(3, 4, n) * 0.3
    data["release_timing_score"]   = rng.beta(4, 2, n)
    data["rhythm_consistency"]     = rng.beta(4, 2, n)
    data["player_distance_hoop"]   = rng.normal(15, 6, n).clip(3, 28)
    data["shot_angle_degrees"]     = rng.normal(5, 8, n).clip(0, 30)
    data["balance_score"]          = rng.beta(5, 2, n)
    data["follow_through_score"]   = rng.beta(4, 2, n)

    df = pd.DataFrame(data)

    # Form score: weighted combination of mechanics
    form = (
        np.clip(1 - abs(df.elbow_angle_release - 95) / 45, 0, 1) * 15 +
        np.clip(1 - abs(df.knee_bend_max - 35) / 30, 0, 1) * 10 +
        np.clip(1 - abs(df.wrist_release_angle - 50) / 35, 0, 1) * 10 +
        np.clip(1 - abs(df.ball_arc_angle - 46) / 20, 0, 1) * 15 +
        df.release_timing_score * 12 +
        df.follow_through_score * 12 +
        df.balance_score * 8 +
        df.rhythm_consistency * 8 +
        df.shooting_arm_extension * 5 +
        df.hip_shoulder_alignment * 5
    )
    df[FORM_COL] = (form + rng.normal(0, 3, n)).clip(0, 100)

    # Shot success probability
    arc_ok   = np.clip(1 - abs(df.ball_arc_angle - 46) / 20, 0, 1)
    dist_ok  = np.clip(1 - (df.player_distance_hoop - 15)**2 / 200, 0.2, 1)
    form_ok  = df[FORM_COL] / 100
    p_make   = (arc_ok * 0.35 + form_ok * 0.45 + dist_ok * 0.20).clip(0.05, 0.95)
    df[SUCCESS_COL] = rng.binomial(1, p_make).astype(int)

    return df


def train(data_path: str, output_dir: str, use_synthetic: bool = False, data_source: str = "unknown"):
    os.makedirs(output_dir, exist_ok=True)

    # Load or generate data
    if use_synthetic or not os.path.exists(data_path):
        print("⚠  No dataset found — generating synthetic training data (n=2000)")
        df = generate_synthetic_dataset()
        df.to_csv(data_path, index=False)
        print(f"   Saved synthetic dataset → {data_path}")
        data_source = "synthetic"
    else:
        df = pd.read_csv(data_path)
        print(f"Loaded dataset: {len(df)} samples from {data_path}")

    missing = [c for c in FEATURE_COLS + [FORM_COL, SUCCESS_COL] if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    X = df[FEATURE_COLS].values
    y_form    = df[FORM_COL].values
    y_success = df[SUCCESS_COL].values

    print(f"\nDataset summary:")
    print(f"  Samples: {len(df)}")
    print(f"  Form score: mean={y_form.mean():.1f}, std={y_form.std():.1f}")
    print(f"  Shot makes: {y_success.sum()} ({100*y_success.mean():.1f}%)")

    # Train/test split
    X_tr, X_te, yf_tr, yf_te, ys_tr, ys_te = train_test_split(
        X, y_form, y_success, test_size=0.2, random_state=42, stratify=y_success)

    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("\n✓ Scaler saved")

    # ── Form Score Regression (XGBoost) ───────────────────────────────────────
    print("\n── Training form score regressor (XGBoost) ─────")
    try:
        from xgboost import XGBRegressor
        form_model = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        print("  XGBoost not found — using sklearn GBR")
        form_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                max_depth=5, random_state=42)

    form_model.fit(X_tr_s, yf_tr)
    yf_pred = form_model.predict(X_te_s).clip(0, 100)

    mae  = mean_absolute_error(yf_te, yf_pred)
    r2   = r2_score(yf_te, yf_pred)
    print(f"  MAE: {mae:.2f} points | R²: {r2:.3f}")

    with open(os.path.join(output_dir, "form_model.pkl"), "wb") as f:
        pickle.dump(form_model, f)
    print("  ✓ Form model saved")

    # ── Shot Success Classifier (XGBoost) ─────────────────────────────────────
    print("\n── Training shot success classifier (XGBoost) ──")
    try:
        from xgboost import XGBClassifier
        success_model = XGBClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=len(ys_tr[ys_tr==0]) / max(len(ys_tr[ys_tr==1]), 1),
            random_state=42, n_jobs=-1, verbosity=0, eval_metric="logloss",
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        success_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                    max_depth=4, random_state=42)

    success_model.fit(X_tr_s, ys_tr)
    ys_pred  = success_model.predict(X_te_s)
    try:
        ys_proba = success_model.predict_proba(X_te_s)[:, 1]
        auc = roc_auc_score(ys_te, ys_proba)
    except:
        auc = 0.0

    acc = accuracy_score(ys_te, ys_pred)
    prec = precision_score(ys_te, ys_pred, zero_division=0)
    rec  = recall_score(ys_te, ys_pred, zero_division=0)
    f1   = f1_score(ys_te, ys_pred, zero_division=0)
    print(f"  Accuracy:  {100*acc:.1f}%")
    print(f"  Precision: {100*prec:.1f}%")
    print(f"  Recall:    {100*rec:.1f}%")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  ROC-AUC:   {auc:.3f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(ys_te, ys_pred)}")
    print(f"\n  Classification Report:\n{classification_report(ys_te, ys_pred)}")

    with open(os.path.join(output_dir, "success_model.pkl"), "wb") as f:
        pickle.dump(success_model, f)
    print("  ✓ Success model saved")

    # ── Feature Importance Plot ───────────────────────────────────────────────
    try:
        importances = form_model.feature_importances_
        fig, ax = plt.subplots(figsize=(10, 6))
        idx = np.argsort(importances)[::-1]
        ax.barh([FEATURE_COLS[i] for i in idx], importances[idx], color="#4F46E5")
        ax.set_xlabel("Feature Importance")
        ax.set_title("Form Score — Feature Importance (XGBoost)")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
        plt.close()
        print(f"\n  ✓ Feature importance plot saved")
    except Exception as e:
        print(f"  Plot skipped: {e}")

    register_model(
        version=settings.MODEL_VERSION,
        form_mae=mae, form_r2=r2,
        success_accuracy=acc, success_auc=auc,
        n_train_samples=len(X_tr),
        data_source=data_source,
    )
    if data_source != "real":
        print(f"\n⚠  Models trained on '{data_source}' data — predictions are not validated against real shots")

    print(f"\n✅ All models saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HoopSense AI models")
    parser.add_argument("--data",       default="data/processed/shots.csv")
    parser.add_argument("--output",     default="data/models/")
    parser.add_argument("--synthetic",  action="store_true", help="Force synthetic data generation")
    parser.add_argument("--data-source", choices=["real", "synthetic", "unknown"], default="unknown",
                        help="Provenance of --data CSV (ignored when synthetic data is generated)")
    args = parser.parse_args()
    train(args.data, args.output, args.synthetic, args.data_source)
