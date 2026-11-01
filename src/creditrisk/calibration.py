"""Calibration: make a predicted 10% PD mean that about 10 in 100 such borrowers default.

Ranking (AUC/Gini) decides who is riskier; calibration decides how much money to expect to lose,
which drives pricing, provisioning and the approval cutoff. Calibration is fitted on the
validation period, never on the test period.

  * "platt"    - logistic recalibration of the log-odds (intercept + slope). Monotonic, so it never
                 changes the ranking or the Gini, and with two parameters it cannot overfit.
  * "isotonic" - a free-form monotone step function. More flexible, but it creates ties.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "platt"):
        if method not in {"platt", "isotonic"}:
            raise ValueError("method must be 'platt' or 'isotonic'")
        self.method = method

    def fit(self, p_raw, y) -> Calibrator:
        y = np.asarray(y).astype(int)
        if self.method == "platt":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw).reshape(-1, 1), y)
            self.slope_ = float(self.model_.coef_[0, 0])
            self.intercept_ = float(self.model_.intercept_[0])
        else:
            self.model_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p_raw, y)
        return self

    def transform(self, p_raw) -> np.ndarray:
        if self.method == "platt":
            return self.model_.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
        return self.model_.predict(np.asarray(p_raw, dtype=float))
