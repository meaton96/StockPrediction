import pandas as pd
from src.config import Config
from pathlib import Path

def dump_csv_metrics(df: pd.DataFrame, file_name: str,  conf: Config):

    _path = Path(conf.data_dir / 'model_metrics' / f"{file_name}.csv")
    #print(f'writing csv:  {_path}')
    df.to_csv(_path, index=False)