# backend/utils/evaluation.py

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_classification(y_true, y_pred, model_name: str = "Model") -> Dict[str, Any]:
    """
    Compute standard classification metrics for binary classification.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0)
    }

    return metrics


def print_evaluation(metrics: Dict[str, Any]) -> None:
    """
    Pretty print evaluation results.
    """
    print(f"\n===== {metrics['model_name']} Evaluation =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)
    print("\nClassification Report:")
    print(metrics["classification_report"])