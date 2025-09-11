import yfinance as yf
import json
import csv

data_path = 'data'
raw_path = 'raw'


with open(f'{data_path}/tickers.json', 'r') as file:
    ticker_list = json.load(file)

historical_data = {}
financials = {}
balance_sheets = {}
cashflows = {}
analyst_recs = {}


def fetch_ticker_data(ticker_list: list):
    for ticker in ticker_list:
        fetch_and_dump_ticker(ticker)


def fetch_and_dump_ticker(ticker: str):
    print(f"fetch and dump {ticker}")
    _ticker = yf.Ticker(ticker)
    _temp_ticker = _ticker.history(period="max")
    _temp_ticker.to_csv(f'{data_path}/{raw_path}/{ticker}.csv')
    print(f'finished fetching {ticker}')

fetch_ticker_data(ticker_list=ticker_list)





