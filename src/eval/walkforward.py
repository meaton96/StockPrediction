from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from src.models.logistic_regression import basic_lr
from src.eval.metrics import get_metrics, flatten_metrics
#from src.models.registry import build_model


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    metrics: Dict[str, Any]

def _rolling_splits(
    n: int,
    min_train: int,
    horizon: int,
    step: int,
    expanding: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Produce sequential test windows of length `horizon`.
    Training has at least `min_train` samples. If expanding=True, training
    starts at 0 and grows; otherwise it's a fixed-size sliding window.
    """
    if min_train <= 0 or horizon <= 0:
        raise ValueError("min_train and horizon must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if min_train + horizon > n:
        # Not enough data to form even one fold
        return []

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train - 1  # inclusive

    while True:
        test_start = train_end + 1
        test_end = test_start + horizon - 1  # inclusive

        # stop if test window would exceed last valid index (n-1)
        if test_end >= n:
            break

        if expanding:
            train_idx = np.arange(0, train_end + 1, dtype=int)
        else:
            train_start = max(0, train_end - (min_train - 1))
            train_idx = np.arange(train_start, train_end + 1, dtype=int)

        test_idx = np.arange(test_start, test_end + 1, dtype=int)
        folds.append((train_idx, test_idx))

        # advance by step
        train_end = train_end + step

        # if the next train_end already reaches or passes n-1, no room for a new test window
        if train_end + 1 >= n:
            break

    return folds


def walk_forward_evaluate(
        model_name: str,
        C: float,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        min_train: int,
        horizon: int,
        step: int = None,
        expanding: bool = True,
    ) -> Dict[str, Any]:
    """
    Run walk-forward validation on X,y.
      - min_train: minimum number of samples in the initial training window
      - horizon: number of samples per test fold (e.g., 20 trading days ≈ 1 month)
      - step: how far to roll each time (default == horizon)
      - expanding: True = keep all history; False = sliding window of size min_train
    Returns:
      {
        "folds": [FoldResult as dict, ...],
        "summary": { mean/std of key metrics },
      }
    """
    if step is None:
        step = horizon

    n = len(X)
    splits = _rolling_splits(n, min_train=min_train, horizon=horizon, step=step, expanding=expanding)

    folds_out: List[FoldResult] = []
    for i, (tr_idx, te_idx) in enumerate(splits, start=1):
        model = basic_lr(C)
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        if y_te.nunique() < 2:
            # no positive or no negative in this fold; skip
            continue

        model.fit(X_tr, y_tr)
        proba_tr = model.predict_proba(X_tr)[:, 1]   # train probs
        proba_te = model.predict_proba(X_te)[:, 1]   # test probs

        # threshold matches prevalence
        p = float(y_tr.mean())
        thr = np.quantile(proba_tr, 1 - p) if 0 < p < 1 else 0.5
        preds = (proba_te >= thr).astype(int)

        

        m = get_metrics(y_true=y_te, y_predictions=preds, y_score=proba_te)

        folds_out.append(
            FoldResult(
                fold_id=i,
                train_start=X_tr.index[0],
                train_end=X_tr.index[-1],
                test_start=X_te.index[0],
                test_end=X_te.index[-1],
                n_train=len(X_tr),
                n_test=len(X_te),
                metrics=m,
            )
        )

    # Aggregate summary from flattened fold metrics
    flat_rows = []
    for fr in folds_out:
        row = flatten_metrics(fr.metrics, ticker="N/A", model=model_name)
        row['test_start'] = fr.test_start
        row['test_end'] = fr.test_end
        row['train_start'] = fr.train_start
        row['train_end'] = fr.train_end

        row.pop("ticker", None)
        row.pop("model", None)
        flat_rows.append(row)

    summary_df = pd.DataFrame(flat_rows) if flat_rows else pd.DataFrame()

    summary = {}
    if not summary_df.empty:
        
        for k in ["roc_auc", "accuracy", "precision_1", "recall_1", "f1_1", "macro_f1", "weighted_f1"]:
            if k in summary_df:
                summary[f"{k}_mean"] = float(summary_df[k].mean())
                summary[f"{k}_std"] = float(summary_df[k].std(ddof=1)) if len(summary_df) > 1 else 0.0

    return {
        "folds": [fr.__dict__ for fr in folds_out],
        "summary": summary,
        "folds_table": summary_df,  
    }