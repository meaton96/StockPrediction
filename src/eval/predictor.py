from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
    metrics = predictor.run()
    print_metrics(metrics)

# if __name__ == "__main__":
#     main()
