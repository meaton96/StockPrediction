# src/modeling/nested_cv.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Iterable, Tuple, Any, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

def _get_scores(pipe, X):
    # Try decision_function, fall back to predict_proba[:, 1]
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)
    proba = pipe.predict_proba(X)
    return proba[:, 1]

def nested_param_sweep(
    pairs: Iterable[Tuple[Any, Any, Any, Any]],
    num_cols_fn: Callable[[Any], Iterable[str]],
    make_pipeline: Callable[[Iterable[str], Any], Any],
    model_factory: Callable[[float], Any],
    param_grid: Iterable[float],
    scorer: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    inner_splits: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Generic nested-CV sweep for a single hyperparameter.
    - pairs: iterable of (Xtr, ytr, Xva, yva) outer folds
    - num_cols_fn: X -> list of numeric columns
    - make_pipeline: (num_cols, model) -> sklearn Pipeline
    - model_factory: param_value -> estimator to plug into pipeline
    - param_grid: iterable of floats (C, alpha, etc.)
    - scorer: (y_true, scores) -> float (default ROC AUC)
    - inner_splits: inner CV folds
    Returns: (inner_df, outer_df, final_param)
    """
    inner_rows, outer_rows = [], []

    for k, (Xtr, ytr, Xva, yva) in enumerate(pairs):
        num_cols = list(num_cols_fn(Xtr))
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=False)

        best_score, best_param = -np.inf, None

        for p in param_grid:
            fold_scores = []
            for inner_idx, (itrain, ival) in enumerate(inner.split(Xtr, ytr)):
                X_itr, X_iva = Xtr.iloc[itrain], Xtr.iloc[ival]
                y_itr, y_iva = ytr.iloc[itrain], ytr.iloc[ival]

                pipe = make_pipeline(num_cols, model_factory(p))
                pipe.fit(X_itr, y_itr)
                s = _get_scores(pipe, X_iva)
                m = scorer(y_iva, s)
                fold_scores.append(m)

                inner_rows.append({
                    "outer_fold": k,
                    "inner_split": inner_idx,
                    "param": p,
                    "metric": m,
                })

            mean_m = float(np.mean(fold_scores))
            if mean_m > best_score:
                best_score, best_param = mean_m, p

        # Refit on full outer-train with best param, evaluate on outer-val
        final_pipe = make_pipeline(num_cols, model_factory(best_param))
        final_pipe.fit(Xtr, ytr)
        s_val = _get_scores(final_pipe, Xva)

        outer_rows.append({
            "fold": k,
            "param": best_param,
            "inner_mean_metric": best_score,
            "val_auc": roc_auc_score(yva, s_val),
            "val_accuracy": ((s_val >= 0).astype(int) == yva).mean(),
            "n_val": len(Xva),
        })

    inner_df = pd.DataFrame(inner_rows)
    outer_df = pd.DataFrame(outer_rows)
    final_param = float(np.median(outer_df["param"]))
    return inner_df, outer_df, final_param
