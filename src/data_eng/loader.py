# from __future__ import annotations

# from typing import Any
# import pandas as pd

# from src.data_eng.get_data import read_csv, prep_data, get_X_y
# from src.data_eng.types import DataBundle
# from src.config import Config


# def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     # if Date exists, use it; otherwise assume it’s already indexed by date
#     if "Date" in out.columns:
#         out["Date"] = pd.to_datetime(out["Date"])
#         out = out.set_index("Date", drop=True).sort_index()
#     else:
#         if not isinstance(out.index, pd.DatetimeIndex):
#             raise ValueError("Expected a Date column or a DatetimeIndex on the input.")
#     return out

# # def load_data_bundle(ticker: str, conf: Config):
# #     df: pd.DataFrame = read_csv(ticker=ticker,conf=conf)
# #   #  print(df.info())
# #     # Ensure base df has a proper DatetimeIndex
# #     df = _ensure_datetime_index(df)
# #    # print(df.info())

# #     train, validate, test = prep_data(df,conf)  

   
# #     train = _ensure_datetime_index(train)
# #     validate = _ensure_datetime_index(validate)
# #     test = _ensure_datetime_index(test)

# #     X_train, y_train = get_X_y(train)
# #     X_val,   y_val   = get_X_y(validate)
# #     X_test,  y_test  = get_X_y(test)

# #     # Make sure X/y inherit the split’s DatetimeIndex
# #     X_train.index = train.index
# #     y_train.index = train.index
# #     X_val.index   = validate.index
# #     y_val.index   = validate.index
# #     X_test.index  = test.index
# #     y_test.index  = test.index

# #     return DataBundle(
# #         train=train, validate=validate, test=test,
# #         X_train=X_train, y_train=y_train,
# #         X_validate=X_val, y_validate=y_val,
# #         X_test=X_test, y_test=y_test
# #     )