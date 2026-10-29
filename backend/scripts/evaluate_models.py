#!/usr/bin/env python3
"""
HoopSense AI — Model Evaluation Script
Generates comprehensive evaluation report with metrics, plots, and confusion matrices.
"""
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, mean_absolute_error, r2_score
)

def evaluate(model_dir: str, data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    with open(os.path.join(model_dir, "form_model.pkl"), "rb") as f:
        form_model = pickle.load(f)
    with open(os.path.join(model_dir, "success_model.pkl"), "rb") as f:
        success_model = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(data_path)
    FEATURES = [
        "elbow_angle_release","elbow_angle_min","knee_bend_max","wrist_release_angle",
        "hip_shoulder_alignment","trunk_lean_angle","shooting_arm_extension",
        "release_height_normalized","ball_arc_angle","ball_velocity_release",
        "ball_velocity_peak","trajectory_smoothness","shot_duration_frames",
        "jump_height_normalized","release_timing_score","rhythm_consistency",
        "player_distance_hoop","shot_angle_degrees","balance_score","follow_through_score",
    ]
    X = scaler.transform(df[FEATURES].values)
    yf = df["form_score"].values
    ys = df["made_shot"].values

    # Form regression metrics
    yf_pred = form_model.predict(X).clip(0, 100)
    mae = mean_absolute_error(yf, yf_pred)
    r2  = r2_score(yf, yf_pred)

    # Success classification metrics
    ys_pred  = success_model.predict(X)
    ys_proba = success_model.predict_proba(X)[:, 1]
    acc  = accuracy_score(ys, ys_pred)
    prec = precision_score(ys, ys_pred, zero_division=0)
    rec  = recall_score(ys, ys_pred, zero_division=0)
    f1   = f1_score(ys, ys_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(ys, ys_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n{'='*50}")
    print("HoopSense AI — Model Evaluation Report")
    print(f"{'='*50}")
    print(f"\nForm Score Regression:")
    print(f"  MAE : {mae:.2f} points")
    print(f"  R²  : {r2:.4f}")
    print(f"\nShot Success Classifier:")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  Precision : {prec*100:.1f}%")
    print(f"  Recall    : {rec*100:.1f}%")
    print(f"  F1        : {f1:.3f}")
    print(f"  ROC-AUC   : {roc_auc:.3f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(ys, ys_pred)}")

    # Plot evaluation dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("HoopSense AI — Model Evaluation", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#0d0d1a")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values(): spine.set_color("#334155")

    # 1. Form score scatter
    axes[0,0].scatter(yf, yf_pred, alpha=0.4, color="#f97316", s=15)
    axes[0,0].plot([0,100],[0,100], "w--", alpha=0.3)
    axes[0,0].set(xlabel="Actual", ylabel="Predicted", title=f"Form Score (MAE={mae:.2f}, R²={r2:.3f})")

    # 2. Confusion matrix heatmap
    cm = confusion_matrix(ys, ys_pred)
    im = axes[0,1].imshow(cm, cmap="YlOrRd")
    axes[0,1].set(xticks=[0,1], yticks=[0,1], xticklabels=["Miss","Make"],
                   yticklabels=["Miss","Make"], title="Confusion Matrix")
    for i in range(2):
        for j in range(2):
            axes[0,1].text(j, i, cm[i,j], ha="center", va="center", color="white", fontweight="bold", fontsize=14)

    # 3. ROC curve
    axes[1,0].plot(fpr, tpr, color="#f97316", lw=2, label=f"AUC={roc_auc:.3f}")
    axes[1,0].plot([0,1],[0,1],"w--",alpha=0.3)
    axes[1,0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[1,0].legend(facecolor="#1a1a2e", labelcolor="white")

    # 4. Predicted probability distribution
    axes[1,1].hist(ys_proba[ys==0], bins=30, alpha=0.6, color="#ef4444", label="Miss")
    axes[1,1].hist(ys_proba[ys==1], bins=30, alpha=0.6, color="#22c55e", label="Make")
    axes[1,1].set(xlabel="Predicted probability", ylabel="Count", title="Score Distribution")
    axes[1,1].legend(facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    outpath = os.path.join(output_dir, "evaluation_report.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Evaluation report saved → {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="data/models/")
    parser.add_argument("--data",   default="data/processed/shots.csv")
    parser.add_argument("--output", default="data/models/eval/")
    args = parser.parse_args()
    evaluate(args.models, args.data, args.output)
