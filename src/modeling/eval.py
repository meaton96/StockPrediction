from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def make_global_pipeline(numeric_cols, model):
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