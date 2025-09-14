from __future__ import annotations

from typing import Any
import pandas as pd

from src.models.utils.get_data import read_csv, prep_data, get_X_y
from src.data.bundle import DataBundle

def load_data_bundle(ticker: str) -> DataBundle:
    df: pd.DataFrame = read_csv(ticker=ticker)
    train, validate, test = prep_data(df)

    X_train, y_train = get_X_y(train)
    X_val,   y_val   = get_X_y(validate)
    X_test,  y_test  = get_X_y(test)

    return DataBundle(
        train=train, validate=validate, test=test,
        X_train=X_train, y_train=y_train,
        X_validate=X_val, y_validate=y_val,
        X_test=X_test, y_test=y_test
    )
