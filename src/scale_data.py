from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
from typing import Tuple, List


def scale_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = "standard"
) -> Tuple[pd.DataFrame, object]:
    """
    Return a copy of df with feature_cols scaled and the fitted scaler.
    scaler_type: 'standard' | 'minmax' | 'robust'
    Does not touch non-feature columns (e.g., Target).
    """
    X = df.copy()

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if scaler_type not in scalers:
        raise ValueError(f"scaler_type must be one of {list(scalers)}")

    scaler = scalers[scaler_type]
    # Fit on non-NaN rows for these columns
    X_features = X[feature_cols].values
    X_scaled_vals = scaler.fit_transform(X_features)

    X_scaled = X.copy()
    X_scaled.loc[:, feature_cols] = X_scaled_vals
    return X_scaled, scaler