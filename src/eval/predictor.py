from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime

import pandas as pd

from src.models.utils.cli_args import parse_cli_args, CLIConfig
from src.data.loader import load_data_bundle
from src.models.registry import build_model
from src.eval.metrics import evaluate, print_metrics

@dataclass
class StockPredictor:
    model_name: str
    ticker: str

    
    _model: Any | None = None

    def load(self) -> None:
        self._model = build_model(self.model_name)

    def run(self) -> Dict[str, Any]:
        if self._model is None:
            self.load()

        bundle = load_data_bundle(self.ticker)
        return evaluate(
            model=self._model,
            X_train=bundle.X_train, y_train=bundle.y_train,
            X_test=bundle.X_validate, y_true=bundle.y_validate,
        )

def main(argv: list[str] | None = None) -> None:
    cfg: CLIConfig = parse_cli_args(argv)
    predictor = StockPredictor(model_name=cfg.model, ticker=cfg.ticker)
    start_time = datetime.now()
    metrics = predictor.run()
    end_time = datetime.now()

    dur = end_time - start_time
    print_metrics(metrics, dur, do_plot=True, title=f'{cfg.ticker} - {cfg.model}')

# if __name__ == "__main__":
#     main()
