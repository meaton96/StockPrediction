from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class DataBundle:
    # Raw frames
    train: pd.DataFrame
    validate: pd.DataFrame
    test: pd.DataFrame

    # Feature/target splits
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validate: pd.DataFrame
    y_validate: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series