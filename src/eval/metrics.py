from __future__ import annotations

from typing import Any, Dict, Sequence, Mapping
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime


def evaluate(model: Any,
             X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame,  y_true: pd.Series) -> Dict[str, Any]:
    
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    
    return get_metrics(y_true=y_true, y_predictions=pred, y_score=proba)

ArrayLike1D = Sequence[int] | np.ndarray | pd.Series


def get_metrics(y_true: ArrayLike1D, 
                y_predictions: ArrayLike1D, 
                y_score: Sequence[float] | np.ndarray | pd.Series,
                digits: int = 3) -> dict[str, Any]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_predictions)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true, y_pred, and y_score must be 1D.")
    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("Lengths must match.")

    return {
        'roc_auc': roc_auc_score(y_true, y_score),
        'clf_report': classification_report(y_true=y_true, y_pred=y_pred, digits=digits),
        'confusion': confusion_matrix(y_true=y_true, y_pred=y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score
    }

def get_multi_metrics_df(predictors: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ticker, predictor in predictors.items():
        row = flatten_metrics(predictor.metrics, ticker, predictor.model_name)
        rows.append(row)

    return pd.DataFrame(rows)

def flatten_metrics(metrics: dict, ticker: str, model: str) -> dict:
    cm = metrics["confusion"].ravel()
    tn, fp, fn, tp = cm
    acc = (tn + tp) / cm.sum()

    # classification_report with output_dict=True
    rep = classification_report(metrics["y_true"], metrics["y_pred"], output_dict=True)

    return {
        "ticker": ticker,
        "model": model,
        "roc_auc": metrics["roc_auc"],
        "accuracy": acc,
        "precision_0": rep["0"]["precision"],
        "recall_0": rep["0"]["recall"],
        "f1_0": rep["0"]["f1-score"],
        "support_0": rep["0"]["support"],
        "precision_1": rep["1"]["precision"],
        "recall_1": rep["1"]["recall"],
        "f1_1": rep["1"]["f1-score"],
        "support_1": rep["1"]["support"],
        "macro_f1": rep["macro avg"]["f1-score"],
        "weighted_f1": rep["weighted avg"]["f1-score"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


def format_metrics(metrics: Mapping[str, Any], duration: datetime ) -> str:
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
    lines.append(f"Model training duration: {duration.total_seconds()}")
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


def print_metrics(
        metrics: Mapping[str, Any], 
        duration: datetime,
        do_plot: bool = True,
        title: str | None = None) -> None:
    """Print the formatted metrics summary."""
    print(format_metrics(metrics, duration))
    if (do_plot):
        plot_classification_diagnostics(
            metrics=metrics,
            title=title
        )


def get_and_print_metrics(y_true: ArrayLike1D, 
                          y_predictions: ArrayLike1D, 
                          y_score: Sequence[float] | np.ndarray | pd.Series, 
                          duration: datetime,
                          digits: int = 3,
                          do_plot: bool = True,
                          title: str | None = None) -> dict[str, Any]:
    metrics = get_metrics(y_true, y_predictions, y_score, digits)
    print_metrics(metrics, duration)
    if do_plot:
        plot_classification_diagnostics(metrics, title=title)
    return metrics
    

def plot_classification_diagnostics(metrics: Mapping[str, Any], title: str | None = None) -> None:
    """
    Render ROC curve, Precision-Recall curve, and Confusion Matrix heatmap
    using values stored in the metrics dict.
    """
    y_true = np.asarray(metrics["y_true"])
    y_pred = np.asarray(metrics["y_pred"])
    y_score = np.asarray(metrics["y_score"], dtype=float)
    cm = np.asarray(metrics["confusion"])

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # Figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    if title:
        fig.suptitle(title, fontsize=14)

    # 1) ROC
    axs[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    axs[0].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].set_title("ROC Curve")
    axs[0].legend(loc="lower right")

    # 2) Precision-Recall
    axs[1].plot(recall, precision, lw=2)
    baseline = (y_true == 1).mean()
    axs[1].hlines(baseline, 0, 1, linestyles="--", colors="grey", lw=1, label=f"Baseline={baseline:.3f}")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_title("Precision–Recall Curve")
    axs[1].legend(loc="upper right")

    # 3) Confusion Matrix
    if cm.shape == (2, 2):
        df_cm = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[2])
        axs[2].set_title("Confusion Matrix")
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[2])
        axs[2].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.show()
