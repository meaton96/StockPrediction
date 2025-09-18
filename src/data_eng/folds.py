# src/data_eng/folds.py

from __future__ import annotations
from typing import List, Tuple, Optional
import pandas as pd
from pandas.tseries.frequencies import to_offset

from src.config import Config
from src.data_eng.get_data import get_X_y, read_csv
from src.data_eng.types import Fold, FoldBundle, TickerFoldBundle, MultiTickerFoldCollection






def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.set_index("Date", drop=True).sort_index()
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("Expected a Date column or a DatetimeIndex on the input.")
        out = out.sort_index()
    return out

def _purge_tail_for_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # If target was built with shift(-horizon), last `horizon` rows can have NaN Target.
    if horizon <= 0:
        return df
    return df.iloc[:-horizon] if len(df) > horizon else df.iloc[0:0]

def calendar_blocks(start: pd.Timestamp, end: pd.Timestamp, conf: Config) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Split [start, end) into contiguous calendar windows using conf.fold_len
    (e.g., '365D', 'Q', 'Y', '180D', '30D').
    """
    if start >= end:
        return []
    # Build edges with given frequency; ensure we include the final end
    edges = pd.date_range(start=start, end=end, freq=conf.fold_len).to_list()
    if not edges or edges[0] != start:
        edges = [start] + edges
    if edges[-1] < end:
        edges.append(end)
    # Produce half-open intervals
    return [(edges[i], edges[i+1]) for i in range(len(edges) - 1)]


def make_time_folds(
    df: pd.DataFrame,
    conf: Config
):
    """
    Build walk-forward folds on df for model selection using calendar-duration
    validation windows defined by conf.fold_len. The final test starts at conf.validate_cutoff.
    Supports 'expanding' and 'sliding' modes.
    Applies an embargo of max(conf.embargo_days, target.horizon) calendar days.
    """
    df = _ensure_datetime_index(df)

    horizon = int(conf.target.get('horizon', 1))
    embargo_days = int(conf.embargo_days) if conf.embargo_days is not None else horizon
    embargo = pd.Timedelta(days=embargo_days)

    # Final test split by time
    test_start = pd.Timestamp(conf.validate_cutoff)
    in_sample = df[df.index < test_start].copy()
    test = df[df.index >= test_start].copy()

    # Be explicit about tail purge for label creation
    in_sample = _purge_tail_for_horizon(in_sample, horizon)
    test = _purge_tail_for_horizon(test, horizon)

    # If essentially empty, fail early
    if in_sample.empty:
        raise ValueError("No in-sample rows available before validate_cutoff for fold construction.")

    # Build validation windows as calendar blocks over the *in-sample* range
    ins_start = in_sample.index.min()
    ins_end = in_sample.index.max()
    blocks = calendar_blocks(ins_start, ins_end, conf)

    folds = []
    for (val_start, val_end) in blocks:
        # Validation set: the block intersected with in-sample
        val_mask = (in_sample.index >= val_start) & (in_sample.index < val_end)
        val_df = in_sample.loc[val_mask]
        if val_df.empty:
            continue

        # Train window must end BEFORE val_start minus embargo
        train_end_time = val_start - embargo
        if conf.fold_mode == 'expanding':
            train_df = in_sample.loc[in_sample.index < train_end_time]
        elif conf.fold_mode == 'sliding':
            if not conf.sliding_train_years:
                raise ValueError("Provide sliding_train_years in Config for sliding mode.")
            train_start_time = train_end_time - pd.DateOffset(years=conf.sliding_train_years)
            train_df = in_sample.loc[(in_sample.index >= train_start_time) & (in_sample.index < train_end_time)]
        else:
            raise ValueError("fold_mode must be 'expanding' or 'sliding'")

        # Purge again for safety (if embargo chopped weirdly)
        train_df = _purge_tail_for_horizon(train_df, horizon)
        val_df = _purge_tail_for_horizon(val_df, horizon)

        # Skip tiny or empty folds
        if len(train_df) < 50 or len(val_df) < 5:
            continue

        X_tr, y_tr = get_X_y(train_df)
        X_va, y_va = get_X_y(val_df)

        # keep indices consistent
        X_tr.index = train_df.index; y_tr.index = train_df.index
        X_va.index = val_df.index;   y_va.index = val_df.index

        folds.append(
            {
                "train": train_df,
                "val": val_df,
                "X_train": X_tr, "y_train": y_tr,
                "X_val": X_va,   "y_val": y_va
            }
        )

    # Final test split features
    X_test, y_test = get_X_y(test)
    X_test.index = test.index
    y_test.index = test.index

    return folds, test, X_test, y_test



def load_fold_bundle(ticker: str, conf: Config) -> FoldBundle:
    df: pd.DataFrame = read_csv(ticker=ticker, conf=conf)
    df = _ensure_datetime_index(df)

    folds_raw, test, X_test, y_test = make_time_folds(df, conf)

    folds = [
        Fold(
            train=f["train"],
            val=f["val"],
            X_train=f["X_train"],
            y_train=f["y_train"],
            X_val=f["X_val"],
            y_val=f["y_val"]
        )
        for f in folds_raw
    ]

    return FoldBundle(
        folds=folds,
        test=test,
        X_test=X_test,
        y_test=y_test
    )

def load_multi_ticker_collection(conf: Config) -> MultiTickerFoldCollection:
    items: List[TickerFoldBundle] = []
    for t in conf.ticker_list:
        try:
            fb = load_fold_bundle(t, conf)
            if len(fb.folds) == 0:
                continue
            items.append(TickerFoldBundle(ticker=t, bundle=fb))
        except Exception as e:
            print(f"[WARN] Skipping {t}: {e}")
    if not items:
        raise RuntimeError("No tickers produced usable folds.")

    # Align fold counts across tickers to the minimum so folds correspond by calendar window
    min_folds = min(len(x.bundle.folds) for x in items)
    trimmed = []
    for x in items:
        trimmed.append(
            TickerFoldBundle(
                ticker=x.ticker,
                bundle=FoldBundle(
                    folds=x.bundle.folds[:min_folds],
                    test=x.bundle.test,
                    X_test=x.bundle.X_test,
                    y_test=x.bundle.y_test
                )
            )
        )
    return MultiTickerFoldCollection(items=trimmed)
