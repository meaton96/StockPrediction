from .fetch_data import fetch_ticker_data
from pathlib import Path
from .clean_data import cleaning_pipeline
from .engineer_features import make_features
from .scale_data import scale_dataframe
from .write_data import dump_csvs
import json
import pandas as pd

ticker_list_path = Path('data/tickers.json')
raw_path = Path('data/raw')
out_path = Path('data/processed')

def main():
    print(f'begin fetching data from yfinance...')

    # load json file with list of ticker names
    with open(f'{ticker_list_path}', 'r') as file:
        ticker_list = json.load(file)['active']

    # fetch all tickers from yfinance
    fetch_ticker_data(ticker_list, raw_path)

    print(f'done fetching data')
    print(f'being data cleaning...')

    frames: dict[str, pd.DataFrame] = {}
    # read the csv data and run through cleaning pipeline
    for ticker in ticker_list:
        frames[ticker] = cleaning_pipeline(ticker, raw_path)

    print(f'done cleaning data')
    print(f'begin feature engineering')

    # build egineered features
    # one day % return
    # lag price data
    # 5 and 20 day SMA
    # volatility
    # 14 day RSI
    # MACD
    # Bollinger bands
    # On-Balance Volume
    # Intraday range
    # Gap up/down
    # target (prediction) column (price go up next day)
    for key, value in frames.items():
        frames[key] = make_features(key, value)

    print('Done egineering features')
    print('scaling data frames...')

    # get engineered column names
    feature_cols = [
    c for c in frames['AAPL'].columns
        if c not in {"Target"} and c not in {"Open","High","Low","Volume", 'Cost'}
    ]

    frame_scalers = {}
    # scale data before modeling
    for key, value in frames.items():
        scaled_df, scaler = scale_dataframe(value, feature_cols)
        frames[key] = scaled_df
        frame_scalers[key] = scaler

    print('Finished Scaling data')

    print("writing csvs...")
    dump_csvs(frames, out_path)

    

if __name__ == "__main__":
    main()