from src.modeling.eval import make_global_pipeline
import pandas as pd
from src.config import Config
# helpers
def num_cols_fn(X):
    return [c for c in X.columns if c not in ("__ticker__", "Date")]

def make_pipeline(num_cols, model):
    # reuse your existing preprocessor
    return make_global_pipeline(num_cols, model)

