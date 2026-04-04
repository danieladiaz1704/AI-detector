"""
backend/utils/evaluation.py
Enhanced evaluation utilities with ROC-AUC, confusion matrix plots,
and a combined multi-model comparison chart.
"""

import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────
def evaluate_classification(
    y_true,
    y_pred,
    model_name: str = "Model",
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true   : Ground-truth labels (0/1)
        y_pred   : Binary predictions (0/1)
        model_name: Label used in plots and prints
        y_proba  : Predicted probabilities for class 1 (enables ROC-AUC)

    Returns:
        Dict with accuracy, precision, recall, f1_score, roc_auc (if y_proba
        provided), confusion_matrix, and classification_report.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics: Dict[str, Any] = {
        "model_name":            model_name,
        "accuracy":              float(accuracy_score(y_true, y_pred)),
        "precision":             float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":                float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":              float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix":      confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "roc_auc":               None,
        "fpr":                   None,
        "tpr":                   None,
    }

    if y_proba is not None:
        y_proba = np.array(y_proba)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics["fpr"] = fpr.tolist()
        metrics["tpr"] = tpr.tolist()

    return metrics


# ─────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────
def print_evaluation(metrics: Dict[str, Any]) -> None:
    print(f"\n{'='*50}")
    print(f"  {metrics['model_name']} — Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1_score']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\n  Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")
    print("\n  Classification Report:")
    print(metrics["classification_report"])


# ─────────────────────────────────────────────────────────────
# Individual plots
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    save_dir: str,
) -> str:
    """Save a heatmap confusion matrix and return its file path."""
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").lower()
    path = os.path.join(save_dir, f"{safe_name}_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {path}")
    return path


def plot_roc_curve(
    y_true,
    y_proba,
    model_name: str,
    save_dir: str,
) -> str:
    """Save an individual ROC curve and return its file path."""
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").lower()
    path = os.path.join(save_dir, f"{safe_name}_roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve → {path}")
    return path


def plot_training_curves(
    history_dict: Dict[str, list],
    model_name: str,
    save_dir: str,
) -> None:
    """
    Save accuracy and loss curves from a Keras history dict or equivalent.

    Expected keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss'
    """
    os.makedirs(save_dir, exist_ok=True)
    safe = model_name.replace(" ", "_").lower()

    for metric, ylabel in [("accuracy", "Accuracy"), ("loss", "Loss")]:
        if metric not in history_dict:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history_dict[metric], label=f"Train {ylabel}")
        val_key = f"val_{metric}"
        if val_key in history_dict:
            ax.plot(history_dict[val_key], label=f"Val {ylabel}")
        ax.set_title(f"{ylabel} — {model_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{safe}_{metric}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {metric} curve → {path}")


# ─────────────────────────────────────────────────────────────
# Combined multi-model plots (call after all models are trained)
# ─────────────────────────────────────────────────────────────
def plot_combined_roc_curves(
    model_results: List[Dict[str, Any]],
    save_dir: str,
) -> str:
    """
    Overlay ROC curves for all models on a single plot.

    Each entry in model_results must contain:
        'model_name', 'fpr', 'tpr', 'roc_auc'
    Entries without fpr/tpr are skipped.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    colours = ["#3b82f6", "#8b5cf6", "#10b981", "#ef4444", "#f59e0b"]

    plotted = 0
    for i, r in enumerate(model_results):
        if r.get("fpr") is None or r.get("tpr") is None:
            continue
        colour = colours[i % len(colours)]
        auc    = r.get("roc_auc", 0)
        ax.plot(
            r["fpr"], r["tpr"],
            lw=2,
            color=colour,
            label=f"{r['model_name']} (AUC={auc:.3f})",
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("No ROC data to plot — skipping combined ROC chart.")
        return ""

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "combined_roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved combined ROC curves → {path}")
    return path


def plot_model_comparison_bar(
    model_results: List[Dict[str, Any]],
    save_dir: str,
) -> str:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1, and AUC
    across all models.
    """
    os.makedirs(save_dir, exist_ok=True)

    names    = [r["model_name"] for r in model_results]
    metrics  = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels   = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    colours  = ["#3b82f6", "#8b5cf6", "#10b981", "#ef4444", "#f59e0b"]

    x     = np.arange(len(names))
    width = 0.15
    offsets = np.linspace(-(len(metrics) - 1) / 2, (len(metrics) - 1) / 2, len(metrics)) * width

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 5))
    for i, (metric, label, colour) in enumerate(zip(metrics, labels, colours)):
        vals = [r.get(metric) or 0 for r in model_results]
        ax.bar(x + offsets[i], vals, width, label=label, color=colour, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.legend(loc="upper right", fontsize=9, ncol=5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison bar chart → {path}")
    return path