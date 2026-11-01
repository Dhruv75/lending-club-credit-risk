"""Model A: the classic credit scorecard (WoE features + logistic regression + points).

Banks still use this because every prediction can be explained as a sum of points, the effect of
each feature is monotonic and signed the right way, and regulators can audit it line by line.

Scaling: score = offset + factor * ln(good:bad odds), with factor = PDO / ln 2 and
offset = base_score - factor * ln(base_odds). With the defaults, 600 points means 50:1 odds and
every 20 extra points doubles the odds of being a good borrower.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import Config
from .woe import WoEBinner


class Scorecard:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, y) -> Scorecard:
        cfg = self.cfg
        y = np.asarray(y).astype(int)
        self.binner_ = WoEBinner(cfg.n_bins, cfg.min_bin_share).fit(X, y)
        iv = self.binner_.iv_
        self.iv_ = iv
        self.dropped_: dict[str, str] = {}
        for f, v in iv.items():
            if v < cfg.min_iv:
                self.dropped_[f] = f"IV {v:.3f} below {cfg.min_iv} (no useful signal)"
            elif v > cfg.suspicious_iv:
                self.dropped_[f] = f"IV {v:.2f} above {cfg.suspicious_iv}: too good to be true, check for leakage"
        candidates = [f for f in iv.index if f not in self.dropped_]
        if not candidates:
            raise ValueError("No feature passed the Information Value filter")

        W = self.binner_.transform(X[candidates])
        corr = W.corr().abs()
        selected: list[str] = []
        for f in candidates:  # strongest first, skip near-duplicates of a stronger feature
            twin = next((s for s in selected if corr.loc[f, s] > cfg.max_woe_corr), None)
            if twin is None:
                selected.append(f)
            else:
                self.dropped_[f] = f"WoE correlation {corr.loc[f, twin]:.2f} with stronger feature '{twin}'"

        # With WoE = ln(%good/%bad), every coefficient for P(bad) must be negative. A positive one
        # means collinearity flipped the sign, which would make the points nonsensical.
        while True:
            model = LogisticRegression(C=1.0, max_iter=5000)
            model.fit(W[selected].to_numpy(), y)
            coef = pd.Series(model.coef_[0], index=selected)
            wrong = coef[coef >= 0]
            if wrong.empty or len(selected) == 1:
                break
            f = wrong.idxmax()
            selected.remove(f)
            self.dropped_[f] = "coefficient had the wrong sign once other features were in (collinearity)"

        self.features_ = selected
        self.model_ = model
        self.coef_ = coef
        self.intercept_ = float(model.intercept_[0])
        self.factor_ = cfg.pdo / np.log(2)
        self.offset_ = cfg.base_score - self.factor_ * np.log(cfg.base_odds)
        self.points_table_ = self._points_table()
        self.points_map_ = {
            f: dict(zip(g["bin"], g["points"])) for f, g in self.points_table_.groupby("feature", sort=False)
        }
        return self

    def _neutral_points(self) -> float:
        n = len(self.features_)
        return -(self.intercept_ / n) * self.factor_ + self.offset_ / n

    def _points_table(self) -> pd.DataFrame:
        n = len(self.features_)
        rows = []
        for f in self.features_:
            t = self.binner_.table(f).copy()
            t["points"] = -(self.coef_[f] * t["woe"] + self.intercept_ / n) * self.factor_ + self.offset_ / n
            t.insert(0, "feature", f)
            rows.append(t)
        return pd.concat(rows, ignore_index=True)

    def feature_points(self, X: pd.DataFrame) -> pd.DataFrame:
        labels = self.binner_.bin_labels(X[self.features_])
        neutral = self._neutral_points()
        return pd.DataFrame(
            {f: labels[f].map(self.points_map_[f]).astype(float).fillna(neutral) for f in self.features_},
            index=X.index,
        )

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.feature_points(X).sum(axis=1).to_numpy()

    def score_to_pd(self, score: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp((np.asarray(score) - self.offset_) / self.factor_))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.score_to_pd(self.score(X))

    def reason_codes(self, X: pd.DataFrame, top_n: int = 3) -> list[list[dict]]:
        """For each applicant, the features that cost the most points versus their best bin."""
        pts = self.feature_points(X)
        labels = self.binner_.bin_labels(X[self.features_])
        best = pd.Series({f: max(self.points_map_[f].values()) for f in self.features_})
        lost = best - pts  # broadcast over columns
        out = []
        for i in range(len(X)):
            row = lost.iloc[i].sort_values(ascending=False)
            out.append(
                [
                    {"feature": f, "bin": labels.iloc[i][f], "points_lost": float(v)}
                    for f, v in row.head(top_n).items()
                    if v > 0.5
                ]
            )
        return out
