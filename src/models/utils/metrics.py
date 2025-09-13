from __future__ import annotations
from typing import Sequence, Mapping, Any
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import pandas as pd
import numpy as np

ArrayLike1D = Sequence[int] | np.ndarray | pd.Series


def get_metrics(y_true: ArrayLike1D, 
                y_predictions : ArrayLike1D, 
                y_score: Sequence[float] | np.ndarray | pd.Series, 
                digits:int = 3) -> dict[str, Any]:
    """
    Computes a set of classification metrics for model evaluation.
    This function calculates the ROC AUC score, generates a classification report,
    and computes the confusion matrix for the given true labels, predicted labels,
    and prediction scores. It ensures all inputs are 1D arrays of matching length.
    Args:
        y_true (ArrayLike1D): Ground truth (correct) target values.
        y_predictions (ArrayLike1D): Predicted target values from the classifier.
        y_score (Sequence[float] | np.ndarray | pd.Series): Predicted scores or probabilities for ROC AUC calculation.
        digits (int, optional): Number of decimal places for the classification report. Defaults to 3.
    Returns:
        dict[str, Any]: Dictionary containing:
            - 'roc_auc': ROC AUC score (float)
            - 'clf_report': Classification report (str)
            - 'confusion': Confusion matrix (np.ndarray)
    Raises:
        ValueError: If input arrays are not 1D or their lengths do not match.
    # 
    """

    # Convert to np arrays and validate shapes
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_predictions)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true, y_pred, and y_score must be 1D.")
    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("Lengths must match.")

    return {
        'roc_auc' : roc_auc_score(y_true, y_score),
        'clf_report' : classification_report(y_true=y_true, y_pred=y_pred, digits=digits),
        'confusion' : confusion_matrix(y_true=y_true, y_pred=y_pred)
    }

def format_metrics(metrics: Mapping[str, Any]) -> str:
    """
    Build a nicely formatted string from a metrics dict like:
      {
        "roc_auc": float,
        "clf_report": str,          # sklearn.classification_report(...)
        "confusion": np.ndarray     # sklearn.confusion_matrix(...)
      }
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MODEL EVALUATION SUMMARY")
    lines.append("=" * 60)

    # ROC AUC
    roc = metrics.get("roc_auc", None)
    if roc is not None:
        try:
            lines.append(f"ROC AUC: {float(roc):.4f}")
        except Exception:
            lines.append(f"ROC AUC: {roc}")

    # accuracy if confusion matrix present
    cm = metrics.get("confusion", None)
    if cm is not None:
        try:
            cm = np.asarray(cm)
            acc = (np.trace(cm) / cm.sum()) if cm.size and cm.sum() > 0 else np.nan
            lines.append(f"Accuracy: {acc:.4f}")
        except Exception:
            pass

    lines.append("-" * 60)

    # Classification report
    report = metrics.get("clf_report", None)
    if isinstance(report, str) and report.strip():
        lines.append("Classification report")
        lines.append(report.rstrip())
    elif report is not None:
        try:
            df_rep = pd.DataFrame(report).T
            lines.append("Classification report")
            lines.append(df_rep.to_string(float_format=lambda x: f"{x:,.4f}"))
        except Exception:
            lines.append(f"Classification report: {report}")

    # Confusion matrix
    if cm is not None:
        try:
            cm = np.asarray(cm)
            lines.append("-" * 60)
            lines.append("Confusion matrix")
            # Pretty 2x2 with labels if binary; otherwise just show the array
            if cm.shape == (2, 2):
                df_cm = pd.DataFrame(cm,
                                     index=pd.Index(["Actual 0", "Actual 1"], name=""),
                                     columns=["Pred 0", "Pred 1"])
                lines.append(df_cm.to_string())
                # Also show TN/FP/FN/TP row for quick scanning
                tn, fp, fn, tp = cm.ravel()
                lines.append(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
            else:
                df_cm = pd.DataFrame(cm)
                lines.append(df_cm.to_string())
        except Exception:
            lines.append(f"Confusion matrix: {cm}")

    lines.append("=" * 60)
    return "\n".join(lines)


def print_metrics(metrics: Mapping[str, Any]) -> None:
    """Print the formatted metrics summary."""
    print(format_metrics(metrics))

def get_and_print_metrics(y_true: ArrayLike1D, 
                y_predictions : ArrayLike1D, 
                y_score: Sequence[float] | np.ndarray | pd.Series, 
                digits:int = 3) -> dict[str, Any]:
    
    metrics = get_metrics(y_true, y_predictions, y_score, digits)
    print_metrics(metrics)
    return metrics
    