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
from src.models.log_reg.logistic_regression import basic_lr_cv, basic_lr
import json



@dataclass
class StockPredictor:
    model_name: str
    ticker: str
    metrics: dict[str, Any] | None = None
    train_time: timedelta | None = None
 
    
    _model: Any | None = None

    # def load(self) -> None:
    #     self._model = basic_lr()

    def run(self) -> Dict[str, Any]:
        # if self._model is None:
        #     self.load()

        bundle = load_data_bundle(self.ticker)

        # Combine train + validate for rolling CV
        X_cv = pd.concat([bundle.X_train, bundle.X_validate])
        y_cv = pd.concat([bundle.y_train, bundle.y_validate])

        lr_cv = basic_lr_cv()

        lr_cv.fit(X_cv, y_cv)

        best_C = float(lr_cv.C_[0])
        

        
        min_train = WF_MIN_TRAIN
        horizon = WF_HORIZON
        step = WF_STEP

        start_time = datetime.now()
        wfv = walk_forward_evaluate(
            model_name=self.model_name,
            C=best_C,
            X=X_cv, y=y_cv,
            min_train=min_train,
            horizon=horizon,
            step=step,
            expanding=True,
        )
        

        # Final test: train on train+validate, evaluate on test exactly once
        final_test_metrics = evaluate_on(
            model=basic_lr(best_C),  # fresh model
            X_train=X_cv, y_train=y_cv,
            X_test=bundle.X_test, y_true=bundle.y_test
        )

        end_time = datetime.now()
        self.train_time = end_time - start_time

        # Store a compact dict for upstream code; you can expand this if you want
        self.metrics = {
            "walk_forward": {
                "folds": wfv["folds"],
                "summary": wfv["summary"],
            },
            "final_test": final_test_metrics,
        }
        folds_table = wfv["folds_table"].copy()
        if not folds_table.empty:
            folds_table.insert(0, "ticker", self.ticker)
            folds_table.insert(1, "model", self.model_name)
            # stash it so multi-ticker aggregator can write it later
            self.metrics["walk_forward"]["folds_table"] = folds_table
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
        print(f'evaluating Logrithmic Regression on {ticker}')
        predictors[ticker].run()
        print(f'{ticker} done..')

    print(f'Finished evalulating models aggregating data')

    df = get_multi_metrics_df(predictors)
    df.to_csv(DATA_DIR / f"model_metrics/Logrithmic Regression_final_test.csv", index=False)

    from src.eval.metrics import get_wfv_summary_df, get_wfv_folds_df
    wfv_summary = get_wfv_summary_df(predictors)
    if not wfv_summary.empty:
        wfv_summary.to_csv(DATA_DIR / f"model_metrics/Logrithmic Regression_wfv_summary.csv", index=False)

    wfv_folds = get_wfv_folds_df(predictors)
    if not wfv_folds.empty:
        wfv_folds.to_csv(DATA_DIR / f"model_metrics/Logrithmic Regression_wfv_folds.csv", index=False)


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
