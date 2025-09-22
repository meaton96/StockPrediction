from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

def make_global_linear_pipeline(numeric_cols, model):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["__ticker__"]),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", model)
    ])


def make_global_rf_pipeline(numeric_cols) -> Tuple[ColumnTransformer, RandomForestClassifier, Pipeline]:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            # no scaler for trees
        ]), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["__ticker__"]),
    ])
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",  # helpful for imbalanced up/down
        random_state=42,
        n_jobs=-1
    )
    return pre, rf, Pipeline([("pre", pre), ("clf", rf)])