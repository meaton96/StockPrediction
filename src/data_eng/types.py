from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from typing import List

# @dataclass(frozen=True)
# class DataBundle:
#     # Raw frames
#     train: pd.DataFrame
#     validate: pd.DataFrame
#     test: pd.DataFrame

#     # Feature/target splits
#     X_train: pd.DataFrame
#     y_train: pd.Series
#     X_validate: pd.DataFrame
#     y_validate: pd.Series
#     X_test: pd.DataFrame
#     y_test: pd.Series


@dataclass(frozen=True)
class Fold:
    train: pd.DataFrame
    val: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series

@dataclass(frozen=True)
class FoldBundle:
    folds: List[Fold]
    # Final test, held out from all fold tuning
    test: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series

@dataclass(frozen=True)
class TickerFoldBundle:
    ticker: str
    bundle: FoldBundle

@dataclass(frozen=True)
class MultiTickerFoldCollection:
    items: List[TickerFoldBundle]

    def n_folds(self) -> int:
        # assume all tickers have same number of usable folds
        return min(len(tfb.bundle.folds) for tfb in self.items)

    def tickers(self) -> List[str]:
        return [tfb.ticker for tfb in self.items]