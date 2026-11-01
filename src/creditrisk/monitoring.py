"""Stability monitoring: is the population the model scores today the one it was trained on?

Population Stability Index, PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%),
with bins fixed on the training data. Industry rule of thumb:
  < 0.10 stable,  0.10-0.25 moderate shift (investigate),  > 0.25 major shift (act).
Computed on a single feature it is often called a Characteristic Stability Index (CSI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import gini

EPS = 1e-4


def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    expected = pd.Series(expected)
    actual = pd.Series(actual)
    if pd.api.types.is_numeric_dtype(expected) and not isinstance(expected.dtype, pd.CategoricalDtype):
        e = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
        a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
        edges = np.unique(np.nanquantile(e, np.linspace(0, 1, n_bins + 1)[1:-1])) if np.isfinite(e).any() else []

        def bucket(v: np.ndarray) -> np.ndarray:
            b = np.searchsorted(edges, v, side="left").astype(float)
            b[np.isnan(v)] = -1  # missing is its own bucket
            return b

        e_lab, a_lab = pd.Series(bucket(e)), pd.Series(bucket(a))
    else:
        e_lab = expected.astype("object").fillna("Missing").astype(str)
        a_lab = actual.astype("object").fillna("Missing").astype(str)
    levels = sorted(set(e_lab.unique()) | set(a_lab.unique()), key=str)
    e_pct = e_lab.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy() + EPS
    a_pct = a_lab.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy() + EPS
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_label(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "major shift"


def feature_psi(train_X: pd.DataFrame, other_X: pd.DataFrame) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(train_X[c], other_X[c])} for c in train_X.columns]
    t = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    t["status"] = t["psi"].map(psi_label)
    return t


def vintage_table(issue_d: pd.Series, y, p, split: pd.Series, min_count: int = 300) -> pd.DataFrame:
    """Per issue quarter: volume, observed default rate, mean predicted PD and Gini."""
    df = pd.DataFrame(
        {
            "quarter": pd.to_datetime(issue_d).dt.to_period("Q").astype(str).to_numpy(),
            "y": np.asarray(y, dtype=float),
            "p": np.asarray(p, dtype=float),
            "split": np.asarray(split),
        }
    )
    rows = []
    for (q, s), g in df.groupby(["quarter", "split"], sort=True):
        has_both = g["y"].nunique() == 2
        rows.append(
            {
                "quarter": q,
                "split": s,
                "count": len(g),
                "default_rate": g["y"].mean(),
                "mean_pd": g["p"].mean(),
                "gini": gini(g["y"], g["p"]) if (len(g) >= min_count and has_both) else np.nan,
            }
        )
    return pd.DataFrame(rows)
