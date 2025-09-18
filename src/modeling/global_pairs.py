from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np

from src.data_eng.types import MultiTickerFoldCollection

def _frame_with_target(X: pd.DataFrame, y: pd.Series, ticker: str) -> pd.DataFrame:
    """
    Produce a single DataFrame with features + 'Target' + '__ticker__' + 'Date' columns.
    We assume X and y already share the same DatetimeIndex (set upstream after splits).
    """
    if not isinstance(X.index, pd.DatetimeIndex) or not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("X and y must have DatetimeIndex and be aligned before framing.")
    df = X.copy()
    df["Target"] = y.values
    df["__ticker__"] = ticker
    df["Date"] = df.index  # keep Date as a plain column; RangeIndex for safety
    df = df.reset_index(drop=True)
    return df

def build_global_fold_pairs(
    collection: MultiTickerFoldCollection,
) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """
    For each aligned fold index k, concatenate train parts from all tickers into one big frame,
    and validation parts into another. No index-based alignment; we construct rows already aligned.
    Returns: [(X_train_k, y_train_k, X_val_k, y_val_k), ...]
    """
    K = min(len(tfb.bundle.folds) for tfb in collection.items)
    pairs: List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]] = []

    for k in range(K):
        train_frames, val_frames = [], []
        for tfb in collection.items:
            f = tfb.bundle.folds[k]
            if len(f.X_train) == 0 or len(f.X_val) == 0:
                continue
            train_frames.append(_frame_with_target(f.X_train, f.y_train, tfb.ticker))
            val_frames.append(_frame_with_target(f.X_val,   f.y_val,   tfb.ticker))

        if not train_frames or not val_frames:
            continue

        train_df = pd.concat(train_frames, ignore_index=True)
        val_df   = pd.concat(val_frames,   ignore_index=True)

        # Split back into X/y
        y_tr = train_df.pop("Target")
        y_va = val_df.pop("Target")
        X_tr = train_df
        X_va = val_df

        pairs.append((X_tr, y_tr, X_va, y_va))

    return pairs

def build_global_insample_and_test(
    collection: MultiTickerFoldCollection,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build a single in-sample frame (everything before validate_cutoff) and a single test frame
    (everything on/after validate_cutoff) across all tickers.

    We de-duplicate in-sample by ('__ticker__','Date') to avoid expanding-window overlaps.
    """
    ins_frames, test_frames = [], []

    for tfb in collection.items:
        # In-sample: concat each fold's train+val, then drop duplicates per (ticker, Date)
        per_ticker_ins = []
        for f in tfb.bundle.folds:
            per_ticker_ins.append(_frame_with_target(f.X_train, f.y_train, tfb.ticker))
            per_ticker_ins.append(_frame_with_target(f.X_val,   f.y_val,   tfb.ticker))
        if per_ticker_ins:
            ins_df_t = pd.concat(per_ticker_ins, ignore_index=True)
            ins_df_t = ins_df_t.drop_duplicates(subset=["__ticker__", "Date"], keep="last")
            ins_frames.append(ins_df_t)

        # Final test
        test_df_t = _frame_with_target(tfb.bundle.X_test, tfb.bundle.y_test, tfb.ticker)
        test_frames.append(test_df_t)

    if not ins_frames or not test_frames:
        raise ValueError("No in-sample or test frames available. Check your folds/test splits.")

    ins_df  = pd.concat(ins_frames,  ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    # Final split into X/y
    y_ins  = ins_df.pop("Target")
    y_test = test_df.pop("Target")
    X_ins  = ins_df
    X_test = test_df

    return X_ins, y_ins, X_test, y_test
