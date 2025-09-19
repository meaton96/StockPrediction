# src/config.py

from pathlib import Path
from typing import Dict, List, Optional

class Config:
    def __init__(
        self,
        features: Optional[List[str]] = None,
        add_int_features: Optional[bool] = True,
        target: Optional[Dict[str, float]] = None,
        ticker_list: Optional[List[str]] = None,
        # interpret validate_cutoff as the start of FINAL TEST
        train_cutoff: str = '2019-01-01',
        validate_cutoff: str = '2022-01-01',
        # new bits for folds
        fold_len:str='365D',
        fold_mode: str = 'expanding',   # 'expanding' or 'sliding'
        sliding_train_years: Optional[int] = None,  # required if fold_mode='sliding'
        embargo_days: Optional[int] = None,         # default: target['horizon']
        project_root: Optional[Path] = None
    ):
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data"
        self.processed_data_path = self.data_dir / "processed"
        self.features = features if features is not None else [
            'r_1d', 'lag', 'sma', 'vix', 'rsi', 'macd', 'boll', 'range', 'gap'
        ]
        self.target = target if target is not None else {'horizon': 5, 'threshold': 0.01}
        self.ticker_list = ticker_list if ticker_list is not None else [
            'AAPL', "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
            "TSM", "ORCL", "WMT", "JPM", "INTC", "UNH", "HD"
        ]
        self.train_cutoff = train_cutoff
        self.validate_cutoff = validate_cutoff  # use as final test start
        self.fold_len = fold_len
        self.fold_mode = fold_mode
        self.sliding_train_years = sliding_train_years
        self.embargo_days = embargo_days  # if None, we’ll fill from target.horizon
        self.raw_path = Path('../data/raw')
        self.add_int_features = add_int_features

    # def set_config(
    #     self,
    #     features: Optional[List[str]] = None,
    #     target: Optional[Dict[str, float]] = None,
    #     ticker_list: Optional[List[str]] = None,
    #     train_cutoff: Optional[str] = None,
    #     validate_cutoff: Optional[str] = None,
    #     fold_len: Optional[int] = None,
    #     fold_mode: Optional[str] = None,
    #     sliding_train_years: Optional[int] = None,
    #     embargo_days: Optional[int] = None
    # ) -> None:
    #     if features is not None: self.features = features
    #     if target is not None: self.target = target
    #     if ticker_list is not None: self.ticker_list = ticker_list
    #     if train_cutoff is not None: self.train_cutoff = train_cutoff
    #     if validate_cutoff is not None: self.validate_cutoff = validate_cutoff
    #     if fold_len is not None: self.fold_len = fold_len
    #     if fold_mode is not None: self.fold_mode = fold_mode
    #     if sliding_train_years is not None: self.sliding_train_years = sliding_train_years
    #     if embargo_days is not None: self.embargo_days = embargo_days

    def to_dict(self) -> Dict:
        return {
            'features': self.features,
            'interaction_features': self.add_int_features,
            'target': self.target,
            'ticker_list': self.ticker_list,
            'train_cutoff': self.train_cutoff,
            'validate_cutoff': self.validate_cutoff,
            'fold_len': self.fold_len,
            'fold_mode': self.fold_mode,
            'sliding_train_years': self.sliding_train_years,
            'embargo_days': self.embargo_days,
            'project_root': str(self.project_root),
            'data_dir': str(self.data_dir),
            'processed_data_path': str(self.processed_data_path)
            
        }
