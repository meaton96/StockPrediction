# from __future__ import annotations

# from typing import Callable, Dict, Any
# from src.models.log_reg.logistic_regression import basic_lr, basic_lr_cv, lrcv_en

# # Constructor registry; add new models here
# MODEL_BUILDERS: Dict[str, Callable[[], Any]] = {
#     "basic_lr": basic_lr,
#     "basic_lr_cv": basic_lr_cv,
#     "lrcv_en": lrcv_en,
# }
# MODELS = {
#     'basic_lr' : 'basic lbfgs solver with l2 penalty, 2000 iter',
#     'basic_lr_cv' : 'basic lbfgs solver with l2 penalty, cross validate, l2 penalty, roc_auc scoring',
#     'lrcv_en' : 'saga solver, elasticnet penalty, 8000 iter'
# }

# def build_model(name: str) -> Any:
#     try:
#         return MODEL_BUILDERS[name]()
#     except KeyError as e:
#         known = ", ".join(sorted(MODEL_BUILDERS))
#         raise ValueError(f"Unknown model '{name}'. Known: {known}") from e
