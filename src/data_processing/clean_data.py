import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta, datetime
from typing import cast
from pathlib import Path

def cleaning_pipeline(ticker: str, raw_path: Path) -> pd.DataFrame:
    df = load_csv_data(ticker, raw_path)
    print(f'pre-cleaning qc check on {ticker}')
    print(qc_report(df))
    de_dupe = clean_duplicates(df)
    de_nan = prune_nans(de_dupe)
    no_outliers, drop_log, qc = clean_outliers(de_nan, ticker)
    print(f'Outlier detection for {ticker}')
    print(drop_log)
    print(qc)
    return no_outliers

#raw_path = r'/content/drive/MyDrive/StockPricePredictor/data/raw'

def load_csv_data(ticker: str, raw_path: Path):
    # Load raw CSV (yfinance-style)
    df = pd.read_csv(f'{raw_path}/{ticker}.csv')

   # print(df.info())
    # Normalize column names to Title Case to avoid surprises
    df.columns = [c.strip().title() for c in df.columns]

    # Ensure Date is datetime index, sorted
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    # Keep only relevant columns that actually exist
    relevant = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in relevant if c in df.columns]]

    # Force numeric types
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def qc_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
        "dupe_dates": int(df.index.duplicated().sum()),
        "n_na_adjclose": int(df["Adj Close"].isna().sum()) if "Adj Close" in df.columns else None,
        "date_range": [str(df.index.min()) if len(df) else None,
                       str(df.index.max()) if len(df) else None],
    }

def clean_duplicates(df):
    if not df.index.is_unique:
      df = df[~df.index.duplicated(keep="last")]
    return df

def prune_nans(df):
    # Drop rows where Adj Close is missing (critical for returns)
    if "Adj Close" in df.columns:
        df = df.dropna(subset=["Adj Close"])

    # If all price fields are NaN on a row, drop it
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if price_cols:
        all_price_nan = df[price_cols].isna().all(axis=1)
        df = df[~all_price_nan]

    # Handle Volume NaNs: set to 0 (or uncomment next line to drop instead)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)
        # Strict alternative: df = df.dropna(subset=["Volume"])

    return df

def clean_outliers(
    df: pd.DataFrame,
    ticker: str,
    abs_threshold: float = 0.50,        # flag > ±50% daily move
    sigma_threshold: float = 5.0,       # flag > 5× rolling std
    sigma_window: int = 20,             # rolling window (trading days)
    keep_event_buffer_days: int = 1,    # keep within ±N days of corp/earnings events
    vol_spike_mult: float = 3.0,        # keep if Volume > mult × rolling median
    vol_window: int = 20,               # window for volume median
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Remove likely bad-tick outliers while preserving real events (splits, dividends, earnings)
    and high-volume shock days.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")
    if "Close" not in df.columns:
        raise ValueError("df must contain a 'Close' column.")

    df = df.copy().sort_index()
    # make index tz-naive for matching
    try:
        df.index = cast(pd.DatetimeIndex, df.index).tz_localize(None)
    except Exception:
        pass

    # Daily returns and rolling sigma
    ret = df["Close"].pct_change()
    rolling_sigma = ret.rolling(sigma_window, min_periods=max(5, sigma_window // 2)).std()

    # Candidate outliers by absolute or sigma rule
    abs_mask   = ret.abs() > abs_threshold
    sigma_mask = (rolling_sigma > 0) & (ret.abs() > sigma_threshold * rolling_sigma)
    candidate_mask = abs_mask | sigma_mask

    # Fetch corporate actions
    t = yf.Ticker(ticker)
    splits_idx    = getattr(t, "splits", pd.Series(dtype=float)).index if hasattr(t, "splits") else pd.DatetimeIndex([])
    dividends_idx = getattr(t, "dividends", pd.Series(dtype=float)).index if hasattr(t, "dividends") else pd.DatetimeIndex([])

    # Fetch earnings dates
    try:
        edf = cast(pd.DataFrame, t.get_earnings_dates(limit=1000))  # columns like: EPS Estimate, Reported EPS, Surprise(%)
        earnings_idx = pd.DatetimeIndex(edf.index)
    except Exception:
        earnings_idx = pd.DatetimeIndex([])

    def norm_idx(ix: pd.DatetimeIndex) -> pd.DatetimeIndex:
        try:
            return pd.DatetimeIndex(ix).tz_localize(None)
        except Exception:
            return pd.DatetimeIndex(ix)

    splits_idx    = norm_idx(cast(pd.DatetimeIndex, splits_idx))
    dividends_idx = norm_idx(cast(pd.DatetimeIndex, dividends_idx))
    earnings_idx  = norm_idx(cast(pd.DatetimeIndex, earnings_idx))

    # Expand events by ± buffer
    def expand(dates: pd.DatetimeIndex, k: int) -> set[pd.Timestamp]:
        s = set()
        for d in dates:
            for off in range(-k, k + 1):
                s.add(d + timedelta(days=off))
        return s

    keep_dates = expand(splits_idx, keep_event_buffer_days) \
               | expand(dividends_idx, keep_event_buffer_days) \
               | expand(earnings_idx, keep_event_buffer_days)

    in_event_window = df.index.to_series().isin(keep_dates)

    # Volume spike heuristic (saves genuine shock days not tagged as events)
    if "Volume" in df.columns:
        vol_med = df["Volume"].rolling(vol_window, min_periods=max(5, vol_window // 2)).median()
        vol_spike = df["Volume"] > (vol_med * vol_spike_mult)
    else:
        vol_spike = pd.Series(False, index=df.index)

    # Drop if candidate, not near an event, and not a volume spike
    drop_mask = candidate_mask & (~in_event_window) & (~vol_spike)

    # Build log
    drop_log = pd.DataFrame({
        "Return": ret.where(drop_mask).dropna(),
        "AbsThreshold": abs_threshold,
        "SigmaThreshold": sigma_threshold,
        "SigmaWindow": sigma_window,
        "NearEvent": in_event_window.where(drop_mask).dropna(),
        "VolSpike": vol_spike.where(drop_mask).dropna(),
    })

    df_clean = df.loc[~drop_mask].copy()

    qc = {
        "ticker": ticker,
        "rows_before": int(df.shape[0]),
        "rows_after": int(df_clean.shape[0]),
        "n_flagged": int(candidate_mask.sum()),
        "n_dropped": int(drop_mask.sum()),
        "n_kept_due_to_events": int((candidate_mask & in_event_window).sum()),
        "n_kept_due_to_vol_spike": int((candidate_mask & ~in_event_window & vol_spike).sum()),
        "n_splits": int(len(splits_idx)),
        "n_dividends": int(len(dividends_idx)),
        "n_earnings": int(len(earnings_idx)),
        "event_buffer_days": keep_event_buffer_days,
        "abs_threshold": abs_threshold,
        "sigma_threshold": sigma_threshold,
        "sigma_window": sigma_window,
        "vol_spike_mult": vol_spike_mult,
        "vol_window": vol_window,
    }

    return df_clean, drop_log, qc