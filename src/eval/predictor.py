from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime, timedelta

import pandas as pd

from src.models.utils.cli_args import parse_cli_args, CLIConfig
from src.data.loader import load_data_bundle
from src.models.registry import build_model
from src.eval.metrics import evaluate, print_metrics, get_multi_metrics_df
from src.models.config import DATA_DIR
import json



@dataclass
class StockPredictor:
    model_name: str
    ticker: str
    metrics: dict[str, Any] | None = None
    train_time: timedelta | None = None
 
    
    _model: Any | None = None

    def load(self) -> None:
        self._model = build_model(self.model_name)

    def run(self) -> Dict[str, Any]:
        if self._model is None:
            self.load()

        bundle = load_data_bundle(self.ticker)
        start_time = datetime.now()
        self.metrics = evaluate(
            model=self._model,
            X_train=bundle.X_train, y_train=bundle.y_train,
            X_test=bundle.X_validate, y_true=bundle.y_validate,
        )
        end_time = datetime.now()
        self.train_time = end_time - start_time
        return self.metrics
    
    def print_metrics(self) -> None:
        print_metrics(self.metrics, self.train_time, do_plot=True, title=f'{self.ticker} - {self.model_name}')

def main(argv: list[str] | None = None) -> None:
    cfg: CLIConfig = parse_cli_args(argv)

    with open(DATA_DIR / 'tickers.json', 'r') as f:
        ticker_list = json.load(f)['active']

    predictors: dict[str, StockPredictor] = {}


    for ticker in ticker_list:
        predictors[ticker] = StockPredictor(model_name=cfg.model, ticker=ticker)

    print(f'Built: {len(ticker_list)} predictors')
    print('Running.....')

    for ticker in ticker_list:
        print(f'evaluating {ticker} model')
        predictors[ticker].run()
        print(f'{ticker} done..')

    print(f'Finished evalulating models aggregating data')


    df = get_multi_metrics_df(predictors)

    df.to_csv(DATA_DIR / f'model_metrics/{cfg.model}.csv')


    # #predictor = StockPredictor(model_name=cfg.model)


    # start_time = datetime.now()
    # metrics = predictor.run()
    # end_time = datetime.now()

    # dur = end_time - start_time
    # print_metrics(metrics, dur, do_plot=True, title=f'{cfg.ticker} - {cfg.model}')

# if __name__ == "__main__":
#     main()
