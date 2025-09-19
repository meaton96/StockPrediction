import pandas as pd
import numpy as np
from typing import Iterable, Dict, Any


def make_features(
    ticker: str,
    df: pd.DataFrame,
    features: list[str],
    target: Dict[str, Any],
    dropna_final: bool = True
) -> pd.DataFrame:
    """
    Build a standard feature set on top of cleaned OHLCV.
    Assumes df has columns: Open, High, Low, Close, Volume and a DatetimeIndex.

    Features are determined by the features list, default adds all:
    ['r_1d, lag, sma, vix, rsi, macd, boll, range, gap']
    """
    print(f'make features for {ticker}')
    out = df.copy()
    # Ensure features is a list of strings, not a single comma-separated string
    if isinstance(features, str):
        features = [f.strip() for f in features.split(',')]
    elif features and isinstance(features[0], str) and ',' in features[0]:
        features = [f.strip() for f in features[0].split(',')]

    if 'r_1d' in features:
        out = add_return_1d(out)
    if ['lag', 'r_1d'] in features:
        out = add_lags(out, base_col="Return_1d", lags=(1, 2, 3, 5))
    if 'sma' in features:
        out = add_sma_features(out, short=5, long=20)
    if ['lag', 'r_1d', 'vix'] in features:
        out = add_volatility(out, window=20, ret_col="Return_1d")
    if 'rsi' in features:
        out = add_rsi(out, window=14)
    if 'macd' in features:
        out = add_macd(out, fast=12, slow=26, signal=9)
    if 'boll' in features:
        out = add_bollinger(out, window=20, n_std=2.0)
    if 'obv' in features:
        out = add_obv(out)
    if 'range' in features:
        out = add_intraday_range(out)
    if 'gap' in features:
        out = add_gap(out)

    _horizon = target.get('horizon', 5)
    _threshold = target.get('threshold', 0.01)
    out = add_target_threshold(out, _horizon, _threshold)

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

def add_target_threshold(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.DataFrame:
    out = df.copy()

    fwd = df["Close"].shift(-1) / df["Close"] - 1
    wins = []
    for k in range(1, 6):
        wins.append(df["Close"].shift(-k) / df["Close"] - 1)  # return in k days
    out["Target"] = (pd.concat(wins, axis=1).max(axis=1) > 0.01).astype(int)
  #  fwd_return = out["Close"].shift(-horizon) / out["Close"] - 1

  #  out[f"Target"] = (fwd_return > threshold).astype(int)
    return out

## Interaction Terms ##

def add_interaction_features(
    df: pd.DataFrame,
    *,
    return_col: str = "Return_1d",
    vol_col: str = "Volatility_20",
    sma_short_col: str = "SMA_5",
    sma_long_col: str = "SMA_20",
    sma_ratio_col: str = "SMA_ratio",
    rsi_col: str | None = None,        # auto-detect if None (e.g. "RSI_14")
    macd_col: str = "MACD",
    macd_sig_col: str = "MACD_signal",
    bb_width_col: str = "BB_width",
    obv_col: str = "OBV",
    gap_col: str = "CO_gap",
    lag_cols: tuple[str, str] = ("Return_1d_lag1", "Return_1d_lag2"),
    clip_z: float | None = 6.0          # winsorize extreme z-scores; set None to disable
) -> pd.DataFrame:
    """
    Add a small, opinionated set of interaction/regime features.
    Only creates a feature if its inputs exist.

    Adds (when possible):
      - trend_mom__sma_x_rsi              = SMA_ratio * RSI
      - mom_disagree__macd_minus_rsi      = MACD - RSI
      - move_vs_vol__ret_over_vol         = Return_1d / Volatility_20
      - vol_trend__bbwidth_x_smaratio     = BB_width * SMA_ratio
      - vol_cond__ret_x_vol               = Return_1d * Volatility_20
      - flow_price__obvZ_x_return         = zscore(OBV) * Return_1d
      - gap_flow__gap_x_volrel20          = CO_gap * (Volume / SMA20(Volume))
      - lag_cross__r1_lag1_x_lag2         = Return_1d_lag1 * Return_1d_lag2
      - lag_vol__r1_lag1_x_vol            = Return_1d_lag1 * Volatility_20
      - regime__rsi_lt30                  = 1{RSI < 30}
      - regime__rsi_gt70                  = 1{RSI > 70}
      - regime__above_sma20               = 1{Close > SMA_20}
      - regime__bands_widening_1d         = 1{BB_width.pct_change() > 0}
      - mom_agree__sign_macd_eq_lag1      = 1{sign(MACD) == sign(Return_1d_lag1)}
    """
    out = df.copy()

    def _has(*cols: str) -> bool:
        return all(c in out.columns for c in cols)

    def _z(x: pd.Series) -> pd.Series:
        mu = x.rolling(100, min_periods=20).mean()
        sd = x.rolling(100, min_periods=20).std()
        z = (x - mu) / sd.replace(0, np.nan)
        if clip_z is not None:
            z = z.clip(lower=-clip_z, upper=clip_z)
        return z

    # Auto-detect an RSI column if not supplied (e.g., "RSI_14")
    if rsi_col is None:
        rsi_candidates = [c for c in out.columns if c.upper().startswith("RSI_")]
        rsi_col = rsi_candidates[0] if rsi_candidates else None

    # 1) Trend vs momentum
    if _has(sma_ratio_col) and rsi_col and _has(rsi_col):
        out["trend_mom__sma_x_rsi"] = out[sma_ratio_col] * out[rsi_col]

    # 2) Momentum disagreement
    if rsi_col and _has(macd_col, rsi_col):
        out["mom_disagree__macd_minus_rsi"] = out[macd_col] - out[rsi_col]

    # 3) Move relative to volatility (normalized and multiplicative)
    if _has(return_col, vol_col):
        out["move_vs_vol__ret_over_vol"] = out[return_col] / out[vol_col].replace(0, np.nan)
        out["vol_cond__ret_x_vol"] = out[return_col] * out[vol_col]

    # 4) Volatility-trend coupling
    if _has(bb_width_col, sma_ratio_col):
        out["vol_trend__bbwidth_x_smaratio"] = out[bb_width_col] * out[sma_ratio_col]

    # 5) Price/volume pressure
    if _has(obv_col, return_col):
        out["flow_price__obvZ_x_return"] = _z(out[obv_col]) * out[return_col]

    # 6) Gap with relative volume
    if _has(gap_col, "Volume"):
        vol_ma20 = out["Volume"].rolling(20, min_periods=5).mean()
        out["gap_flow__gap_x_volrel20"] = out[gap_col] * (out["Volume"] / vol_ma20.replace(0, np.nan))

    # 7) Cross-lag interactions
    lag1, lag2 = lag_cols
    if _has(lag1, lag2):
        out["lag_cross__r1_lag1_x_lag2"] = out[lag1] * out[lag2]
    if _has(lag1, vol_col):
        out["lag_vol__r1_lag1_x_vol"] = out[lag1] * out[vol_col]

    # 8) Regime flags
    if rsi_col and _has(rsi_col):
        out["regime__rsi_lt30"] = (out[rsi_col] < 30).astype("Int8")
        out["regime__rsi_gt70"] = (out[rsi_col] > 70).astype("Int8")
    if _has("Close", sma_long_col):
        out["regime__above_sma20"] = (out["Close"] > out[sma_long_col]).astype("Int8")
    if _has(bb_width_col):
        out["regime__bands_widening_1d"] = (out[bb_width_col].pct_change() > 0).astype("Int8")

    # 9) Momentum agreement signal
    if _has(macd_col, "Return_1d_lag1"):
        macd_sign = np.sign(out[macd_col])
        r1_sign = np.sign(out["Return_1d_lag1"])
        out["mom_agree__sign_macd_eq_lag1"] = (macd_sign == r1_sign).astype("Int8")

    return out
    

