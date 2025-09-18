
from pathlib import Path
from typing import Dict, List, Optional

class Config:
	def __init__(
		self,
		features: Optional[List[str]] = None,
		target: Optional[Dict[str, float]] = None,
		ticker_list: Optional[List[str]] = None,
		train_cutoff: str = '2019-01-01',
		validate_cutoff: str = '2022-01-01',
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
			'AAPL', "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "TSM", "ORCL", "WMT", "JPM",
			"INTC", "UNH", "HD"
		]
		self.train_cutoff = train_cutoff
		self.validate_cutoff = validate_cutoff
		self.raw_path = Path('../data/raw')

	def set_config(
		self,
		features: Optional[List[str]] = None,
		target: Optional[Dict[str, float]] = None,
		ticker_list: Optional[List[str]] = None,
		train_cutoff: Optional[str] = None,
		validate_cutoff: Optional[str] = None
	) -> None:
		"""
		Update config variables. Any parameter left as None will not overwrite the current value.
		"""
		if features is not None:
			self.features = features
		if target is not None:
			self.target = target
		if ticker_list is not None:
			self.ticker_list = ticker_list
		if train_cutoff is not None:
			self.train_cutoff = train_cutoff
		if validate_cutoff is not None:
			self.validate_cutoff = validate_cutoff

	def to_dict(self) -> Dict:
		"""Return config as a dictionary."""
		return {
			'features': self.features,
			'target': self.target,
			'ticker_list': self.ticker_list,
			'train_cutoff': self.train_cutoff,
			'validate_cutoff': self.validate_cutoff,
			'project_root': str(self.project_root),
			'data_dir': str(self.data_dir),
			'processed_data_path': str(self.processed_data_path)
		}


