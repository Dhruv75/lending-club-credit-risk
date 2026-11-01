"""Evaluation metrics used in credit risk.

  * AUC   - probability a random defaulter is scored riskier than a random non-defaulter.
  * Gini  - 2*AUC - 1, the number credit teams quote (0 = random, 1 = perfect).
  * KS    - largest gap between the score distributions of defaulters and non-defaulters.
  * Brier - mean squared error of the probabilities (calibration + sharpness).
  * ECE   - average gap between predicted and observed default rate across PD deciles.
Accuracy is deliberately absent: with ~15% defaults, "approve everyone" is 85% accurate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def decile_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Calibration by PD decile: mean predicted PD vs observed default rate."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    t = df.groupby("bin").agg(count=("y", "size"), mean_pd=("p", "mean"), default_rate=("y", "mean"))
    return t.reset_index()


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = decile_table(y, p, n_bins)
    return float(np.average(np.abs(t["mean_pd"] - t["default_rate"]), weights=t["count"]))


def lift_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Decile 1 = riskiest 10%. Shows how many of all defaults each band captures."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["decile"] = n_bins - pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    t = df.groupby("decile").agg(count=("y", "size"), defaults=("y", "sum"), mean_pd=("p", "mean"))
    t["default_rate"] = t["defaults"] / t["count"]
    t["cum_share_of_defaults"] = t["defaults"].cumsum() / t["defaults"].sum()
    t["lift"] = t["default_rate"] / df["y"].mean()
    return t.reset_index()


def bootstrap_gini_ci(y, p, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        stats.append(2 * roc_auc_score(y[i], p[i]) - 1)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarize(y, p, n_boot: int = 0, seed: int = 0, probabilities: bool = True) -> dict:
    """All headline metrics for one model on one dataset."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {
        "n": int(len(y)),
        "default_rate": float(y.mean()),
        "auc": auc(y, p),
        "gini": gini(y, p),
        "ks": ks(y, p),
    }
    if probabilities:
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        out |= {
            "mean_pd": float(p.mean()),
            "brier": float(brier_score_loss(y, pc)),
            "log_loss": float(log_loss(y, pc, labels=[0, 1])),
            "ece": expected_calibration_error(y, p),
        }
    if n_boot:
        out["gini_ci"] = list(bootstrap_gini_ci(y, p, n_boot, seed))
    return out
