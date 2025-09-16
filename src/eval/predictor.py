from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime, timedelta

import pandas as pd

# from src.models.utils.cli_args import parse_cli_args, CLIConfig
from src.data.loader import load_data_bundle
#from src.models.registry import build_model
from src.eval.metrics import evaluate_on, print_wfv_and_test, get_multi_metrics_df,plot_classification_diagnostics
from src.eval.walkforward import walk_forward_evaluate
from src.models.config import DATA_DIR, WF_MIN_TRAIN, WF_HORIZON, WF_STEP
from src.eval.walkforward import run_lr_model
import json



@dataclass
class StockPredictor:
    model_name: str
    ticker: str
    metrics: dict[str, Any] | None = None
 
    
    _model: Any | None = None

    # def load(self) -> None:
    #     self._model = basic_lr()

    def run(self) -> Dict[str, Any]:
        # if self._model is None:
        #     self.load()

        bundle = load_data_bundle(self.ticker)

        self.metrics = run_lr_model(
            bundle,
            self.model_name,
            self.ticker
        )
        return self.metrics
    
    def print_metrics(self) -> None:
        print_wfv_and_test(self.metrics, self.train_time)
        plot_classification_diagnostics(
            self.metrics["final_test"],
            title=f'{self.ticker} - {self.model_name} (TEST)')

def main(argv: list[str] | None = None) -> None:
    # cfg: CLIConfig = parse_cli_args(argv)

    with open(DATA_DIR / 'tickers.json', 'r') as f:
        ticker_list = json.load(f)['active']

    predictors: dict[str, StockPredictor] = {}


    for ticker in ticker_list:
        predictors[ticker] = StockPredictor(model_name='basic_lr', ticker=ticker)

    print(f'Built: {len(ticker_list)} predictors')
    print('Running.....')

    for ticker in ticker_list:
        print(f'evaluating Logistic Regression on {ticker}')
        predictors[ticker].run()
        print(f'{ticker} done..')

    print(f'Finished evalulating models aggregating data')

    df = get_multi_metrics_df(predictors)
    df.to_csv(DATA_DIR / f"model_metrics/lr_final_test.csv", index=False)

    from src.eval.metrics import get_wfv_summary_df, get_wfv_folds_df
    wfv_summary = get_wfv_summary_df(predictors)
    if not wfv_summary.empty:
        wfv_summary.to_csv(DATA_DIR / f"model_metrics/lr_wfv_summary.csv", index=False)

    wfv_folds = get_wfv_folds_df(predictors)
    if not wfv_folds.empty:
        wfv_folds.to_csv(DATA_DIR / f"model_metrics/lr_wfv_folds.csv", index=False)


    # df = get_multi_metrics_df(predictors)

    # df.to_csv(DATA_DIR / f'model_metrics/{cfg.model}.csv')


    # #predictor = StockPredictor(model_name=cfg.model)


    # start_time = datetime.now()
    # metrics = predictor.run()
    # end_time = datetime.now()

    # dur = end_time - start_time
    # print_metrics(metrics, dur, do_plot=True, title=f'{cfg.ticker} - {cfg.model}')

# if __name__ == "__main__":
#     main()
