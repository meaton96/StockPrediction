from __future__ import annotations
from typing import Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: Iterable[str],
    cat_cols: Iterable[str],
) -> List[str]:
    """
    Return the feature names AFTER the ColumnTransformer, in the order
    seen by the downstream estimator.

    Assumes:
      - numeric transformer outputs numeric_cols in the same order
      - categorical transformer is OneHotEncoder
      - your preprocessor has transformers named "num" and "cat"
    """
    names: List[str] = []

    # numeric block
    if "num" in preprocessor.named_transformers_:
        # Our numeric pipeline: imputer only, so names are unchanged
        names.extend(list(numeric_cols))

    # categorical block
    if "cat" in preprocessor.named_transformers_:
        cat = preprocessor.named_transformers_["cat"]
        # sklearn >= 1.0
        try:
            cat_names = list(cat.get_feature_names_out(cat_cols))
        except Exception:
            # older sklearn fallback
            cat_names = []
            for base in cat_cols:
                for v in getattr(cat, "categories_", [[]])[0]:
                    cat_names.append(f"{base}_{v}")
        names.extend(cat_names)

    return names


def _get_estimator_from_pipeline(pipe: Pipeline):
    return getattr(pipe, "named_steps", {}).get("clf", pipe)


def plot_rf_impurity_importance(
    pipe: Pipeline,
    feature_names: Iterable[str],
    top: int = 30,
):
    """
    Bar chart of impurity-based importances from a fitted RandomForest.
    """
    clf = _get_estimator_from_pipeline(pipe)
    if not hasattr(clf, "feature_importances_"):
        print("Estimator has no feature_importances_; skipping.")
        return None

    importances = pd.Series(clf.feature_importances_, index=list(feature_names))
    importances = importances.sort_values(ascending=False)

    k = min(top, len(importances))
    plt.figure(figsize=(8, max(3, 0.3 * k)))
    importances.head(k).iloc[::-1].plot(kind="barh")
    plt.xlabel("Impurity-based importance")
    plt.title("Random Forest feature importances")
    plt.tight_layout()
    plt.show()
    return importances


def plot_permutation_importance_on_test(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = 42,
    top: int = 30,
):
    """
    Permutation importance on the test set. Works with Pipeline + ColumnTransformer.
    """
    r = permutation_importance(
        pipe,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False)

    k = min(top, len(imp))
    plt.figure(figsize=(8, max(3, 0.3 * k)))
    imp.head(k).iloc[::-1].plot(kind="barh")
    plt.xlabel(f"Permutation importance ({scoring})")
    plt.title("Permutation importance on test")
    plt.tight_layout()
    plt.show()
    return imp, r


def plot_partial_dependence_top(
    pipe: Pipeline,
    X: pd.DataFrame,
    features: list[str] | list[tuple[str, str]],
    kind: str = "average",
    grid_resolution: int = 50,
):
    """
    Partial dependence plots for a list of feature names or pairs.

    Note: Passing original column names works when using a Pipeline + ColumnTransformer
    as long as those columns exist in X and are transformed inside the pipeline.
    """
    fig, ax = plt.subplots(figsize=(7, 4 * len(features)))
    PartialDependenceDisplay.from_estimator(
        pipe,
        X,
        features=features,
        kind=kind,
        grid_resolution=grid_resolution,
        ax=ax,
    )
    plt.tight_layout()
    plt.show()
