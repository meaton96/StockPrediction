import pandas as pd
import numpy as np
from typing import Iterable


def make_features(
    ticker: str,
    df: pd.DataFrame,
    dropna_final: bool = True
) -> pd.DataFrame:
    """
    Build a standard feature set on top of cleaned OHLCV.
    Assumes df has columns: Open, High, Low, Close, Volume and a DatetimeIndex.
    """
    print(f'make features for {ticker}')
    out = df.copy()
    out = add_return_1d(out)
    out = add_lags(out, base_col="Return_1d", lags=(1, 2, 3, 5))
    out = add_sma_features(out, short=5, long=20)
    out = add_volatility(out, window=20, ret_col="Return_1d")
    out = add_rsi(out, window=14)
    out = add_macd(out, fast=12, slow=26, signal=9)
    out = add_bollinger(out, window=20, n_std=2.0)
    out = add_obv(out)
    out = add_intraday_range(out)
    out = add_gap(out)
  #  out = add_target_up_next_day(out)
    out = add_target_threshold(out)

    # Drop rows with NaNs produced by rolling windows
    if dropna_final:
        out = out.dropna().copy()

    return out

# one day % return
def add_return_1d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Return_1d"] = out["Close"].pct_change()
    return out

# add 4 a few days of lag for model memory
def add_lags(df: pd.DataFrame, base_col: str = "Return_1d", lags: Iterable[int] = (1, 2, 3, 5)) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{base_col}_lag{lag}"] = out[base_col].shift(lag)
    return out
# add simple moving averages
def add_sma_features(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    out = df.copy()
    out[f"SMA_{short}"] = out["Close"].rolling(short).mean()
    out[f"SMA_{long}"] = out["Close"].rolling(long).mean()
    # ratio only if both exist to avoid division warnings on initial rows
    out["SMA_ratio"] = out[f"SMA_{short}"] / out[f"SMA_{long}"]
    return out
#add intra dat volatility window
def add_volatility(df: pd.DataFrame, window: int = 20, ret_col: str = "Return_1d") -> pd.DataFrame:
    out = df.copy()
    out[f"Volatility_{window}"] = out[ret_col].rolling(window).std()
    return out
#add relative strength index
def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return out
#add moving average convergence-divergence
def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()
    return out
# add bollinger bands
def add_bollinger(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    ma = out["Close"].rolling(window).mean()
    sd = out["Close"].rolling(window).std()
    out["BB_upper"] = ma + n_std * sd
    out["BB_lower"] = ma - n_std * sd
    out["BB_width"] = (out["BB_upper"] - out["BB_lower"]) / ma.replace(0, np.nan)
    return out
# add OBV (vol spike signal)
def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sign = np.sign(out["Close"].diff().fillna(0))
    out["OBV"] = (sign * out["Volume"]).cumsum()
    return out
# add intra day price range
def add_intraday_range(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HL_range"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)
    return out
# add gap up/down price
def add_gap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CO_gap"] = (out["Close"] - out["Open"]) / out["Open"].replace(0, np.nan)
    return out

# compute target column, price up or down from previous close
def add_target_up_next_day(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    return out

def add_target_ternary(df: pd.DataFrame, horizon: int = 5, band: float = 0.005) -> pd.DataFrame:
    out = df.copy()
    fwd_return = out["Close"].shift(-horizon) / out["Close"] - 1
    out[f"Target_{horizon}d_tern"] = pd.Series(
        np.where(fwd_return > band, 1,
                 np.where(fwd_return < -band, -1, 0)),
        index=df.index
    )
    return out

def add_target_threshold(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.DataFrame:
    out = df.copy()
    fwd_return = out["Close"].shift(-horizon) / out["Close"] - 1
    out[f"Target"] = (fwd_return > threshold).astype(int)
    return out

def add_target(
    df: pd.DataFrame,
    *,
    price_col: str = "Close",
    horizon: int = 3,
    threshold: float | None = None,   # if None, threshold = 0 for binary; for ternary use neutral_band
    neutral_band: float = 0.005,      # 0.5% dead zone for ternary labels
    label_style: str = "binary",      # "binary" or "ternary"
    use_log_return: bool = False,
    dropna_tail: bool = True
) -> pd.DataFrame:
    """
    Add forward return and classification target.

    Returns:
      - R_{h}d[_log]: forward return over `horizon` days (pct or log)
      - Target_*:    labels per `label_style`
          binary: 1 if return > threshold (or >0 if threshold is None), else 0
          ternary: 1 if return > band, -1 if return < -band, else 0
    Notes:
      - Last `horizon` rows are NaN-labeled to avoid look-ahead bias.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if label_style not in {"binary", "ternary"}:
        raise ValueError("label_style must be 'binary' or 'ternary'")
    if price_col not in df.columns:
        raise KeyError(f"price_col '{price_col}' not found in df")

    out = df.copy()

    # forward return
    fwd_ratio = out[price_col].shift(-horizon) / out[price_col]
    if use_log_return:
        fwd_ret = np.log(fwd_ratio)
        rname = f"R_{horizon}d_log"
    else:
        fwd_ret = fwd_ratio - 1.0
        rname = f"R_{horizon}d"

    out[rname] = fwd_ret

    # labels
    if label_style == "binary":
        thr = 0.0 if threshold is None else float(threshold)
        tname = f"Target_{horizon}d_gt{thr:.2%}" if threshold is not None else f"Target_{horizon}d_up"
        labels = (out[rname] > thr).astype("Int64")  # nullable ints so tail NaNs are allowed
    else:
        band = float(neutral_band if threshold is None else threshold)
        tname = f"Target_{horizon}d_tern_b{band:.2%}"
        labels = pd.Series(np.where(out[rname] > band, 1,
                           np.where(out[rname] < -band, -1, 0)), index=out.index, dtype="Int64")

    # kill the tail to prevent leakage
    if horizon > 0:
        tail_idx = out.index[-horizon:]
        out.loc[tail_idx, [rname]] = np.nan
        out.loc[tail_idx, [tname]] = pd.NA

    out[tname] = labels

    if dropna_tail:
        out = out.dropna(subset=[rname, tname])

    return out

