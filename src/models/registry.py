from __future__ import annotations

from typing import Callable, Dict, Any
from src.models.log_reg.logistic_regression import basic_lr, basic_lr_cv

# Constructor registry; add new models here
MODEL_BUILDERS: Dict[str, Callable[[], Any]] = {
    "basic_lr": basic_lr,
    "basic_lr_cv": basic_lr_cv,
}

def build_model(name: str) -> Any:
    try:
        return MODEL_BUILDERS[name]()
    except KeyError as e:
        known = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"Unknown model '{name}'. Known: {known}") from e
