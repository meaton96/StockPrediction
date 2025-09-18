from .fetch_data import fetch_ticker_data
from .clean_data import cleaning_pipeline
from .engineer_features import make_features
from .write_data import dump_csvs
import pandas as pd
from src.config import Config






def run_pipeline(
    conf: Config
):

    print(f'begin fetching data from yfinance...')


    # fetch all tickers from yfinance
    print(conf.ticker_list)
    fetch_ticker_data(conf.ticker_list, conf.raw_path)

    print(f'done fetching data')
    print(f'being data cleaning...')

    frames: dict[str, pd.DataFrame] = {}
    # read the csv data and run through cleaning pipeline
    for ticker in conf.ticker_list:
        frames[ticker] = cleaning_pipeline(ticker, conf.raw_path)

    print(f'done cleaning data')
    print(f'begin feature engineering')

    # build engineered features
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
    # target (prediction) column: 5 day price horizon up 1%
    for key, value in frames.items():
        frames[key] = make_features(key, value, conf.features, conf.target)

    print('Done egineering features')

    print("writing csvs...")
    dump_csvs(frames, conf.processed_data_path)



# if __name__ == "__main__":
#     main()