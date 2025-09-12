# imports
import yfinance as yf
import json
from typing import cast, Any
import csv
import pandas as pd
from .write_data import dump_csv
from pathlib import Path





def fetch_ticker_data(ticker_list: Any, raw_path: Path):

    for ticker in ticker_list:
        fetch_and_save(ticker, raw_path)


def fetch_and_save(ticker: str, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = cast(pd.DataFrame, yf.download(ticker, start="1980-01-01", auto_adjust=True, progress=False))

    # 1) Ensure DatetimeIndex and name it 'Date' for CSV
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df.index.name = "Date"

    # 2) Flatten MultiIndex columns like ('Close','AAPL') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        # If there’s exactly one ticker level, drop it
        if len(df.columns.get_level_values(1).unique()) == 1:
            df.columns = [lvl0 for (lvl0, _lvl1) in df.columns]
        else:
            # Multiple tickers → keep TICKER_Close style for clarity
            df.columns = [f"{t}_{field}" for (field, t) in df.columns]

    # 3) Keep only adjusted OHLCV (auto_adjust=True already did the adjusting)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].sort_index()

    # 4) Write with index label so Date becomes a column in the CSV
    dump_csv(df, out_dir, ticker)
    return df
