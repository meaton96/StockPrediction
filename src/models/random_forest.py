from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def build_rf_search(
    n_splits: int = 5,
    random_state: int = 42,
    n_iter: int = 40,
    n_jobs: int = -1,
):
    """
    Returns a RandomizedSearchCV over RandomForestClassifier using TimeSeriesSplit.
    """
    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",      
        class_weight="balanced",  
        bootstrap=True,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    param_dist = {
        "n_estimators": np.linspace(200, 800, 7, dtype=int),
        "max_depth": [None, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5, 0.7, None],
        "bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=ts_cv,
        refit=True,               
        verbose=0,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return search

def rf_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp