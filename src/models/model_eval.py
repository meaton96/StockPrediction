from typing import Any
from src.models.utils.get_data import read_csv, prep_data, get_X_y
from src.models.log_reg.logistic_regression import basic_lr_model
from src.models.utils.metrics import get_and_print_metrics
import pandas as pd

def run(ticker: str = 'AAPL', model_str:str = 'basic_lr'):
    """
    Main function to run model evaluation.
    Loads data, selects model, and runs training and evaluation.
    """
    X_train, y_train, X_val, y_val, frames = get_and_prep_data(ticker)

    model: Any
    if model_str == 'basic_lr':
        model = basic_lr_model()

    eval_model(model, X_train, y_train, X_val, y_val)


def eval_model(model: Any, 
               X_train: pd.DataFrame, 
               y_train: pd.Series, 
               X_test: pd.DataFrame, 
               y_true: pd.Series) -> dict[str, Any]:
    """
    Fits the model on training data, makes predictions on test data,
    and prints evaluation metrics.
    """
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:,1]
    predictions = (probabilities >= 0.5).astype(int)

    return get_and_print_metrics(y_true=y_true, y_predictions=predictions, y_score=probabilities)


def get_and_prep_data(
        ticker: str = 'AAPL'
        ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, pd.DataFrame]]:
    """
    Loads and prepares data for training and validation.
    Returns feature and target splits for train and validation sets,
    along with all data frames.
    """
    df = read_csv(ticker=ticker)

    frames : dict[str, pd.DataFrame] = {}
    frames['train'], frames['validate'], frames['test'] = prep_data(df)

    X_train, y_train = get_X_y(frames['train'])
    X_val, y_val = get_X_y(frames['validate'])

    return X_train, y_train, X_val, y_val, frames





if __name__ == "__main__":
    run()

