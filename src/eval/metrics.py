from __future__ import annotations

from typing import Any, Dict, Sequence, Mapping
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

ArrayLike1D = Sequence[int] | np.ndarray | pd.Series


def evaluate_on(
    model: Any,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame,  y_true: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)
    return get_metrics(y_true=y_true, y_predictions=pred, y_score=proba)


def evaluate(model: Any,
             X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame,  y_true: pd.Series) -> Dict[str, Any]:
    
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    
    return get_metrics(y_true=y_true, y_predictions=pred, y_score=proba)




def get_metrics(y_true: ArrayLike1D, 
                y_predictions: ArrayLike1D, 
                y_score: Sequence[float] | np.ndarray | pd.Series,
                digits: int = 3,
                clf_report: bool = True) -> dict[str, Any]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_predictions)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true, y_pred, and y_score must be 1D.")
    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("Lengths must match.")
    
    try:
        roc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc = float("nan")
    
    clf_txt = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        digits=digits,
        zero_division=0
    ) if clf_report else ""

    return {
        'roc_auc': roc,
        'clf_report': clf_txt,
        'confusion': confusion_matrix(y_true=y_true, y_pred=y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score
    }

def get_multi_metrics_df(predictors: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ticker, predictor in predictors.items():
        m = predictor.metrics
        if not m:
            continue

        # Prefer the proper held-out test metrics
        if "final_test" in m and isinstance(m["final_test"], dict):
            row = flatten_metrics(m["final_test"], ticker, predictor.model_name)
            rows.append(row)

        # If for some reason final_test is missing but old flat metrics exist, fall back
        elif "confusion" in m:
            row = flatten_metrics(m, ticker, predictor.model_name)
            rows.append(row)

        # Otherwise skip this ticker silently
    return pd.DataFrame(rows)


def flatten_metrics(metrics: dict, ticker: str, model: str) -> dict:
    cm = np.asarray(metrics.get("confusion"))
    if cm.size != 4:  # not 2x2
        tn = fp = fn = tp = np.nan
        acc = np.nan
    else:
        tn, fp, fn, tp = cm.ravel()
        acc = (tn + tp) / cm.sum() if cm.sum() > 0 else np.nan

    rep = classification_report(
        metrics.get("y_true"), metrics.get("y_pred"),
        output_dict=True, zero_division=0
    )

    return {
        "ticker": ticker,
        "model": model,
        "roc_auc": float(metrics.get("roc_auc", np.nan)),
        "accuracy": acc,
        "precision_0": rep.get("0", {}).get("precision", np.nan),
        "recall_0": rep.get("0", {}).get("recall", np.nan),
        "f1_0": rep.get("0", {}).get("f1-score", np.nan),
        "support_0": rep.get("0", {}).get("support", np.nan),
        "precision_1": rep.get("1", {}).get("precision", np.nan),
        "recall_1": rep.get("1", {}).get("recall", np.nan),
        "f1_1": rep.get("1", {}).get("f1-score", np.nan),
        "support_1": rep.get("1", {}).get("support", np.nan),
        "macro_f1": rep.get("macro avg", {}).get("f1-score", np.nan),
        "weighted_f1": rep.get("weighted avg", {}).get("f1-score", np.nan),
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

def print_wfv_and_test(metrics: Mapping[str, Any], duration) -> None:

    wfv = metrics["walk_forward"]
    testm = metrics["final_test"]

    print("=" * 60)
    print("WALK-FORWARD SUMMARY (mean ± std across folds)")
    if wfv["summary"]:
        for k, v in wfv["summary"].items():
            # pair mean/std lines
            if k.endswith("_mean"):
                base = k[:-5]
                mean = v
                std = wfv["summary"].get(f"{base}_std", 0.0)
                print(f"{base:>12}: {mean:.4f} ± {std:.4f}")
    else:
        print("No folds produced. Check min_train/horizon vs data length.")

    print("-" * 60)
    print("FINAL TEST EVALUATION (train = train+validate; test = held-out)")
    print(format_metrics(testm, duration))

    print("=" * 60)

def get_wfv_summary_df(predictors: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ticker, predictor in predictors.items():
        m = predictor.metrics or {}
        wf = m.get("walk_forward", {})
        summary = wf.get("summary", {})
        if not summary:
            continue
        row = {"ticker": ticker, "model": predictor.model_name}
        row.update(summary)
        rows.append(row)
    return pd.DataFrame(rows)

def get_wfv_folds_df(predictors: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for ticker, predictor in predictors.items():
        m = predictor.metrics or {}
        wf = m.get("walk_forward", {})
        ft = wf.get("folds_table", None)
        if ft is None or getattr(ft, "empty", True):
            continue
        frames.append(ft)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

