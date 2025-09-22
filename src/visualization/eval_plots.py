# src/visualization/eval_plots.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Iterable, Callable
import os

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss, roc_auc_score
)
from sklearn.calibration import calibration_curve


# ---------------------------
# Score extraction utilities
# ---------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically safer sigmoid
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def get_scores_and_proba(
    pipe,
    X,
    proba_strategy: str = "auto",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (scores, proba) where:
      - scores are continuous decision scores for ranking and thresholding
      - proba are class-1 probabilities or None if not available

    proba_strategy:
      - "auto": use predict_proba if available; else sigmoid(decision_function) if available; else None
      - "model": use predict_proba only; else None
      - "sigmoid": force sigmoid(decision_function) if available; else None
      - "none": never compute proba
    """
    scores = None
    proba = None

    has_dec = hasattr(pipe, "decision_function")
    has_proba = hasattr(pipe, "predict_proba")

    if has_dec:
        scores = pipe.decision_function(X)
    elif has_proba:
        # If there is no margin, use probability as a score
        proba_raw = pipe.predict_proba(X)
        scores = proba_raw[:, 1]
    else:
        # last resort: signed class; terrible, but prevents total failure
        scores = pipe.predict(X).astype(float)

    if proba_strategy == "none":
        return np.asarray(scores), None

    if proba_strategy == "model":
        if has_proba:
            proba = pipe.predict_proba(X)[:, 1]
        return np.asarray(scores), proba

    if proba_strategy == "sigmoid":
        if has_dec:
            proba = _sigmoid(scores)
        return np.asarray(scores), proba

    # auto
    if has_proba:
        proba = pipe.predict_proba(X)[:, 1]
    elif has_dec:
        proba = _sigmoid(scores)
    # else leave as None
    return np.asarray(scores), proba


# ---------------------------
# Dataframe builder
# ---------------------------

def build_eval_frame(
    pipe,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    proba_strategy: str = "auto",
) -> pd.DataFrame:
    scores, proba = get_scores_and_proba(pipe, X_test, proba_strategy=proba_strategy)
    pred0 = (scores >= 0).astype(int)

    return pd.DataFrame({
        "Date": X_test["Date"].values if "Date" in X_test.columns else pd.NaT,
        "ticker": X_test["__ticker__"].values if "__ticker__" in X_test.columns else "ALL",
        "y": y_test.values,
        "score": scores,
        "proba": proba if proba is not None else np.nan,
        "pred@0": pred0,
    })


# ---------------------------
# Threshold sweep
# ---------------------------

def sweep_thresholds(y_true: np.ndarray, scores: np.ndarray, grid: Optional[Iterable[float]] = None) -> pd.DataFrame:
    if grid is None:
        grid = np.quantile(scores, np.linspace(0.01, 0.99, 99))
        grid = np.unique(np.r_[grid, 0.0])  # ensure 0 in grid
    out = []
    for thr in grid:
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        youden = tpr - fpr
        out.append((thr, acc, f1, prec, tpr, fpr, youden))
    return pd.DataFrame(out, columns=["thr","acc","f1","precision","recall","fpr","youden"])


# ---------------------------
# Plot helpers
# ---------------------------

def plot_roc(df_eval: pd.DataFrame, savepath: Optional[Path] = None):
    fpr, tpr, _ = roc_curve(df_eval["y"], df_eval["score"])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Final Test ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=160)
        plt.close()
    else:
        plt.show()
    return roc_auc

def plot_pr(df_eval: pd.DataFrame):
    prec, rec, _ = precision_recall_curve(df_eval["y"], df_eval["score"])
    ap = average_precision_score(df_eval["y"], df_eval["score"])
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Final Test Precision-Recall")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    return ap

def plot_threshold_sweep(df_eval: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    sweep = sweep_thresholds(df_eval["y"].values, df_eval["score"].values)
    best_youden_thr = float(sweep.loc[sweep["youden"].idxmax(), "thr"])
    best_f1_thr = float(sweep.loc[sweep["f1"].idxmax(), "thr"])

    plt.figure(figsize=(7,5))
    plt.plot(sweep["thr"], sweep["acc"], label="accuracy")
    plt.plot(sweep["thr"], sweep["f1"], label="F1")
    plt.plot(sweep["thr"], sweep["precision"], label="precision")
    plt.plot(sweep["thr"], sweep["recall"], label="recall")
    plt.axvline(0.0, linestyle="--", label="thr=0.0")
    plt.axvline(best_youden_thr, linestyle=":", label=f"best Youden={best_youden_thr:.4f}")
    plt.axvline(best_f1_thr, linestyle="-.", label=f"best F1={best_f1_thr:.4f}")
    plt.xlabel("Decision threshold (on decision score)")
    plt.ylabel("Metric")
    plt.title("Threshold sweep on final test")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_youden_thr, best_f1_thr, sweep

def plot_calibration(df_eval: pd.DataFrame):
    if not np.isfinite(df_eval["proba"]).any():
        print("No probabilities available; skipping calibration.")
        return None, None, None
    prob_true, prob_pred = calibration_curve(df_eval["y"], df_eval["proba"], n_bins=10, strategy="quantile")
    brier = brier_score_loss(df_eval["y"], df_eval["proba"])
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker="o", label=f"Brier={brier:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return prob_pred, prob_true, brier

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5,4))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0","1"])
    plt.yticks(tick_marks, ["0","1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    return cm

def plot_gains(df_eval: pd.DataFrame):
    if not np.isfinite(df_eval["proba"]).any():
        # fallback to score ranking if no probas
        tmp = df_eval.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        tmp = df_eval.sort_values("proba", ascending=False).reset_index(drop=True)

    tmp["cum_positives"] = tmp["y"].cumsum()
    total_pos = tmp["y"].sum()
    pct = np.arange(1, len(tmp)+1) / len(tmp)
    lift = tmp["cum_positives"] / (total_pos + 1e-9)

    plt.figure(figsize=(6,5))
    plt.plot(pct, lift, label="Model cumulative capture")
    plt.plot([0,1],[0,1], linestyle="--", label="Random")
    plt.xlabel("Top x% by model score")
    plt.ylabel("Captured positives fraction")
    plt.title("Cumulative gains")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_per_ticker_ap(df_eval: pd.DataFrame):
    if "ticker" not in df_eval.columns or (df_eval["ticker"] == "ALL").all():
        print("No per-ticker info; skipping.")
        return None
    by_ticker = df_eval.groupby("ticker").apply(
        lambda d: average_precision_score(d["y"], d["score"])
    ).sort_values(ascending=False)
    plt.figure(figsize=(8, max(3, 0.25*len(by_ticker))))
    by_ticker.plot(kind="barh")
    plt.xlabel("Average Precision")
    plt.title("Per-ticker AP on final test")
    plt.tight_layout()
    plt.show()
    return by_ticker

def plot_rolling_accuracy(df_eval: pd.DataFrame, window: int = 50):
    if "Date" not in df_eval.columns or pd.isna(df_eval["Date"]).all():
        print("No dates available; skipping rolling accuracy.")
        return None
    dft = df_eval.sort_values("Date").copy()
    roll_acc = (dft["pred@0"] == dft["y"]).rolling(window).mean()
    plt.figure(figsize=(8,4))
    plt.plot(dft["Date"], roll_acc)
    plt.title(f"Rolling accuracy over time (window={window})")
    plt.xlabel("Date")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
    return roll_acc

# ---------------------------
# Convenience “do it all”
# ---------------------------

def run_full_eval_and_plots(
    pipe,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Optional[Path] = None,
    proba_strategy: str = "auto",
    save_predictions_as: str = "final_test_predictions.csv",
    save_roc_as: str = "final_test_roc.png",
):
    """
    Builds df_eval, renders the standard plots, saves a couple of artifacts.
    Returns df_eval and a dict of key stats.
    """
    out_dir = Path(out_dir) if out_dir is not None else None

    if not out_dir.exists():
        os.mkdir(out_dir)

    df_eval = build_eval_frame(pipe, X_test, y_test, proba_strategy=proba_strategy)

    roc_path = out_dir / save_roc_as if out_dir else None



    
    roc_auc_val = plot_roc(df_eval, savepath=roc_path)
    ap_val = plot_pr(df_eval)
    best_youden_thr, best_f1_thr, sweep = plot_threshold_sweep(df_eval)
    plot_calibration(df_eval)
    pred_0 = (df_eval["score"] >= 0.0).astype(int)
    plot_confusion(df_eval["y"].values, pred_0.values, "Confusion Matrix (thr=0.0)")
    pred_youden = (df_eval["score"] >= best_youden_thr).astype(int)
    plot_confusion(df_eval["y"].values, pred_youden.values, f"Confusion Matrix (Youden={best_youden_thr:.4f})")
    print("Report @ thr=0.0")
    print(classification_report(df_eval["y"], pred_0, digits=3))
    print("\nReport @ best Youden")
    print(classification_report(df_eval["y"], pred_youden, digits=3))
    plot_gains(df_eval)
    plot_per_ticker_ap(df_eval)
    plot_rolling_accuracy(df_eval, window=50)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_eval.to_csv(out_dir / save_predictions_as, index=False)

    stats = {
        "roc_auc": float(roc_auc_val),
        "average_precision": float(ap_val),
        "best_youden_thr": float(best_youden_thr),
        "best_f1_thr": float(best_f1_thr),
    }
    return df_eval, stats, sweep
