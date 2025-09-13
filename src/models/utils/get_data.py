from __future__ import annotations
import pandas as pd
from datetime import datetime, date
from typing import Tuple
from src.models.config import PROCESSED_DATA_PATH, TRAIN_CUTOFF, VALIDATE_CUTOFF




def read_csv(ticker: str, **kwargs) -> pd.DataFrame:
    """
    Reads a CSV file and returns a pandas DataFrame.

    Args:
        file_path (str | Path): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to pd.read_csv.

    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    if "parse_dates" not in kwargs:
        kwargs["parse_dates"] = ["Date"]
    return pd.read_csv(PROCESSED_DATA_PATH / f'{ticker}.csv', **kwargs)



def prep_data(
    df: pd.DataFrame,
    train_cutoff: str | datetime | date | pd.Timestamp = TRAIN_CUTOFF,
    validate_cutoff: str | datetime | date | pd.Timestamp = VALIDATE_CUTOFF,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/validate/test by date cutoffs.
      train:   Date <  train_cutoff
      validate: train_cutoff <= Date < validate_cutoff
      test:    Date >= validate_cutoff
    """
    if 'Date' not in df.columns:
        raise KeyError('Dataframe must have a "date" coulmn')

    out = df.copy()
    out['Date'] = pd.to_datetime(out['Date'], errors='raise')

    _train_c = pd.to_datetime(train_cutoff)
    _valid_c = pd.to_datetime(validate_cutoff)


    if _train_c >= _valid_c:
        raise ValueError(
            f"train_cutoff ({_train_c}) must be earlier than validate_cutoff ({_valid_c})."
        )

    train_df = out[out["Date"] < _train_c]
    valid_df = out[(out["Date"] >= _train_c) & (out["Date"] < _valid_c)]
    test_df  = out[out["Date"] >= _valid_c]

    return train_df, valid_df, test_df


def get_X_y(df: pd.DataFrame, y_col:str = 'Target', non_feature_cols:list = ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']) -> tuple[pd.DataFrame, pd.Series]:

    X = df.drop(columns=non_feature_cols)
    y = df[y_col]

    return X, y



