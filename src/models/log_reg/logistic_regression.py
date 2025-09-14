
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit


def basic_lr() -> LogisticRegression:
    reg_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=2000,
        n_jobs=None,
        class_weight="balanced",
        random_state=42
    )
    return reg_model

def basic_lr_cv(
        n_splits: int = 5,
        Cs: list[float] = [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        scoring: str = 'roc_auc',
        penalty: str = 'l2',
        solver: str = 'lbfgs',
        max_iter: int = 5000
          ) -> LogisticRegressionCV:
    ts_cv = TimeSeriesSplit(n_splits=n_splits)
    
    return LogisticRegressionCV(
        Cs=Cs,
        cv=ts_cv,
        scoring=scoring,
        penalty=penalty,
        solver=solver,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42
    )


def lrcv_en(
        n_splits: int = 5,
        Cs: list[float] = [0.03, 0.1, 0.3, 1, 3, 10],
        l1_ratios: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
        scoring: str = 'roc_auc',
        penalty: str = 'elasticnet',
        solver: str = 'saga',
        max_iter: int = 8000
        ):
    
    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    return LogisticRegressionCV(
        n_jobs=-1,
        Cs=Cs,
        cv=ts_cv,
        l1_ratios=l1_ratios,
        scoring=scoring,
        penalty=penalty,
        class_weight="balanced",
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )