import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.models.utils.get_data import read_csv, prep_data, get_X_y
from src.models.utils.metrics import get_metrics, print_metrics
from typing import Any

def run_regression(ticker: str = 'AAPL', type: str = 'basic'):


    df = read_csv(ticker=ticker)

    

    frames : dict[str, pd.DataFrame] = {}

    frames['train'], frames['validate'], frames['test'] = prep_data(df)

    

    if (type == 'basic'):
        metrics = run_basic_regression(frames)
        print_metrics(metrics)


def run_basic_regression(frames: dict[str, pd.DataFrame]) -> dict[str: Any]:
    """
    # Trains a logistic regression model on the training data, evaluates it on the validation set,
    # and returns ROC AUC, classification report, and confusion matrix.
    """

    X_train, y_train = get_X_y(frames['train'])
    X_val, y_val = get_X_y(frames['validate'])
    

    reg_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=2000,
        n_jobs=None,
        random_state=42
    )

    reg_model.fit(X_train, y_train)

    proba_validate = reg_model.predict_proba(X_val)[:,1]
    pred_validate = (proba_validate >= 0.5).astype(int)

    return get_metrics(
        y_true=y_val,
        y_predictions=pred_validate,
        y_score=proba_validate)

def basic_lr_model() -> LogisticRegression:
    reg_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=2000,
        n_jobs=None,
        random_state=42
    )
    return reg_model


    

