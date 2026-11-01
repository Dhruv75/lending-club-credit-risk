"""Model B: LightGBM gradient-boosted trees, with monotonic constraints and SHAP explanations.

Trees capture non-linear effects and interactions the scorecard cannot. Two safeguards keep them
honest:
  * early stopping on the validation period, so the number of trees is chosen out-of-time;
  * monotonic constraints on features whose direction is known (a higher FICO score can never
    raise the predicted risk), which makes the model easier to defend and more robust to noise.
"""

from __future__ import annotations

import inspect

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config

# +1: risk can only go up as the feature goes up; -1: risk can only go down.
MONOTONE = {
    "fico": -1,
    "dti": 1,
    "inq_last_6mths": 1,
    "revol_util": 1,
    "loan_to_income": 1,
}


class GBMModel:
    def __init__(self, cfg: Config, monotone: dict[str, int] | None = None):
        self.cfg = cfg
        self.monotone = MONOTONE if monotone is None else monotone

    def _params(self, columns: list[str]) -> dict:
        cfg = self.cfg
        return dict(
            objective="binary",
            metric="auc",
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_estimators=cfg.n_estimators,
            monotone_constraints=[self.monotone.get(c, 0) for c in columns],
            random_state=cfg.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train, X_valid: pd.DataFrame, y_valid) -> GBMModel:
        self.features_ = list(X_train.columns)
        self.model_ = lgb.LGBMClassifier(**self._params(self.features_))
        valid = (X_valid[self.features_], np.asarray(y_valid).astype(int))
        # LightGBM >= 4.7 prefers eval_X / eval_y; older versions only know eval_set.
        if "eval_X" in inspect.signature(self.model_.fit).parameters:
            eval_args = {"eval_X": (valid[0],), "eval_y": (valid[1],)}
        else:
            eval_args = {"eval_set": [valid]}
        self.model_.fit(
            X_train,
            np.asarray(y_train).astype(int),
            **eval_args,
            callbacks=[
                lgb.early_stopping(self.cfg.early_stopping_rounds, first_metric_only=True, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        self.best_iteration_ = int(self.model_.best_iteration_ or self.cfg.n_estimators)
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated probability of default."""
        return self.model_.predict_proba(X[self.features_])[:, 1]

    def shap_values(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """TreeSHAP contributions in log-odds, computed by LightGBM itself (no extra dependency).

        Each row's contributions plus the base value add up to the model's raw log-odds output.
        """
        contrib = self.model_.predict(X[self.features_], pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], columns=self.features_, index=X.index), contrib[:, -1]

    def gain_importance(self) -> pd.Series:
        booster = self.model_.booster_
        gain = booster.feature_importance(importance_type="gain")
        return pd.Series(gain, index=booster.feature_name()).sort_values(ascending=False)
