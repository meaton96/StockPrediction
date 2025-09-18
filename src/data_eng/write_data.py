import csv
import pandas as pd
from pathlib import Path
import os

def dump_csvs(frames: dict[str, pd.DataFrame], out_dir: Path):

    for key, df in frames.items():
        dump_csv(df, out_dir, key)


def dump_csv(df: pd.DataFrame, out_dir: Path, label: str):
    if not out_dir.exists():
        os.mkdir(out_dir)
    _path = out_dir / f'{label}.csv'
    df.to_csv(_path, index=True, index_label="Date", date_format="%Y-%m-%d")
    print(f"saved: {_path}")