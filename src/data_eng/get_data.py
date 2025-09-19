from __future__ import annotations
import pandas as pd
from datetime import datetime, date
from typing import Tuple
from src.config import Config







def read_csv(ticker: str, conf: Config, **kwargs) -> pd.DataFrame:
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
    return pd.read_csv(conf.processed_data_path / f'{ticker}.csv', **kwargs)

def get_train_test_data(conf: Config
                        ) -> Tuple[pd.DataFrame , pd.Series,pd.DataFrame, pd.Series]:

    df = pd.DataFrame()

    for ticker in conf.ticker_list:
        df = pd.concat([df, read_csv(ticker, conf)])
    
    q = df['Date'] < conf.train_cutoff

    df_train = df[q]
    df_test = df[~q]

    X_train = df_train.drop(columns=['Target'])
    X_test = df_test.drop(columns=['Target'])

    y_train = df_train['Target']
    y_test = df_test['Target']

    return (X_train, y_train, X_test, y_test)
    

def prep_data(
    df: pd.DataFrame,
    conf: Config
    ):
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    else:
        raise KeyError('Dataframe must have a "Date" column or a DatetimeIndex')

    train_cutoff = pd.Timestamp(conf.train_cutoff)
    validate_cutoff = pd.Timestamp(conf.validate_cutoff)

    m_train = dates < train_cutoff
    m_val   = (dates >= train_cutoff) & (dates < validate_cutoff)
    m_test  = dates >= validate_cutoff

    train = df.loc[m_train].copy()
    validate = df.loc[m_val].copy()
    test = df.loc[m_test].copy()
    return train, validate, test


def get_X_y(df: pd.DataFrame, 
            y_col:str = 'Target', 
            non_feature_cols:list = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']
            ) -> tuple[pd.DataFrame, pd.Series]:

    X = df.drop(columns=non_feature_cols)
    y = df[y_col]

    return X, y



