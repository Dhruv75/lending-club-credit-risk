"""Weight-of-Evidence (WoE) binning and Information Value (IV).

Convention used throughout: WoE = ln(%good / %bad).
  * A bin with WoE > 0 has proportionally more good borrowers than average (safer).
  * A bin with WoE < 0 is riskier than average.
IV = sum over bins of (%good - %bad) * WoE measures how well a feature separates goods from bads.
Rule of thumb: < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, > 0.5 suspicious.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING = "Missing"
OTHER = "Other"


def _fmt(v: float, digits: int = 4) -> str:
    if abs(v) >= 10_000:
        return f"{v:,.0f}"
    return f"{v:.{digits}g}"


def interval_labels(edges: np.ndarray) -> list[str]:
    """Labels for bins (-inf, e1], (e1, e2], ..., (ek, inf); precision grows until labels are unique."""
    for digits in (4, 6, 8, 12):
        labels, lo = [], "-inf"
        for e in edges:
            hi = _fmt(e, digits)
            labels.append(f"({lo}, {hi}]")
            lo = hi
        labels.append(f"({lo}, inf)")
        if len(set(labels)) == len(labels):
            return labels
    return [f"bin {i}" for i in range(len(edges) + 1)]


def woe_table(labels: list[str], counts, bads, smoothing: float = 0.5) -> tuple[pd.DataFrame, float]:
    """WoE and IV per bin, with additive smoothing so empty cells never produce log(0)."""
    counts = np.asarray(counts, dtype=float)
    bads = np.asarray(bads, dtype=float)
    goods = counts - bads
    k = len(counts)
    pct_good = (goods + smoothing) / (goods.sum() + smoothing * k)
    pct_bad = (bads + smoothing) / (bads.sum() + smoothing * k)
    woe = np.log(pct_good / pct_bad)
    iv = (pct_good - pct_bad) * woe
    table = pd.DataFrame(
        {
            "bin": labels,
            "count": counts.astype(int),
            "share": counts / counts.sum(),
            "bads": bads.astype(int),
            "bad_rate": bads / np.maximum(counts, 1),
            "woe": woe,
            "iv": iv,
        }
    )
    return table, float(iv.sum())


def _numeric_stats(edges: np.ndarray, v: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(edges, v, side="left")
    k = len(edges) + 1
    return np.bincount(idx, minlength=k).astype(float), np.bincount(idx, weights=y, minlength=k)


def _merge_small_bins(edges: np.ndarray, v: np.ndarray, y: np.ndarray, min_count: int) -> np.ndarray:
    while len(edges) > 0:
        cnt, bad = _numeric_stats(edges, v, y)
        small = np.flatnonzero(cnt < min_count)
        if len(small) == 0:
            break
        i = small[np.argmin(cnt[small])]
        rate = bad / np.maximum(cnt, 1)
        if i == 0:
            drop = 0
        elif i == len(cnt) - 1:
            drop = i - 1
        else:  # merge with the neighbour whose bad rate is closer
            drop = i - 1 if abs(rate[i] - rate[i - 1]) <= abs(rate[i] - rate[i + 1]) else i
        edges = np.delete(edges, drop)
    return edges


def _make_monotonic(edges: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Merge adjacent bins until the bad rate only moves in one direction (easy to explain)."""
    rho = pd.Series(v).corr(pd.Series(y), method="spearman")
    direction = 1.0 if (pd.isna(rho) or rho >= 0) else -1.0
    while len(edges) > 0:
        cnt, bad = _numeric_stats(edges, v, y)
        rate = bad / np.maximum(cnt, 1)
        violations = np.flatnonzero(np.diff(rate) * direction < 0)
        if len(violations) == 0:
            break
        edges = np.delete(edges, violations[0])
    return edges


@dataclass
class FeatureBinning:
    name: str
    kind: str  # "numeric" or "categorical"
    edges: np.ndarray | None = None  # numeric only: inner cut points
    groups: dict[str, str] | None = None  # categorical only: category -> bin label
    woe: dict[str, float] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    iv: float = 0.0

    def labels(self, x: pd.Series) -> np.ndarray:
        """Bin label for every value; unseen categories go to 'Other', NaN to 'Missing'."""
        if self.kind == "numeric":
            v = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
            names = np.array(interval_labels(self.edges), dtype=object)
            idx = np.minimum(np.searchsorted(self.edges, v, side="left"), len(names) - 1)
            out = names[idx]
            out[np.isnan(v)] = MISSING
            return out
        s = pd.Series(x).astype("object")
        isna = pd.isna(s).to_numpy()
        mapped = s.map(self.groups).to_numpy(dtype=object)
        out = np.where(pd.isna(mapped), OTHER, mapped).astype(object)
        out[isna] = MISSING
        return out

    def transform(self, x: pd.Series) -> np.ndarray:
        """WoE value for every row; a bin never seen in training gets 0 (neutral)."""
        lab = pd.Series(self.labels(x))
        return lab.map(self.woe).astype(float).fillna(0.0).to_numpy()


class WoEBinner:
    """Fit WoE bins on training data, then map any dataset onto them."""

    def __init__(self, n_bins: int = 10, min_bin_share: float = 0.05, monotonic: bool = True, smoothing: float = 0.5):
        self.n_bins = n_bins
        self.min_bin_share = min_bin_share
        self.monotonic = monotonic
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y) -> WoEBinner:
        y = np.asarray(y).astype(int)
        min_count = max(1, math.ceil(self.min_bin_share * len(y)))
        self.binnings_: dict[str, FeatureBinning] = {}
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                self.binnings_[c] = self._fit_numeric(c, X[c], y, min_count)
            else:
                self.binnings_[c] = self._fit_categorical(c, X[c], y, min_count)
        return self

    def _fit_numeric(self, name: str, x: pd.Series, y: np.ndarray, min_count: int) -> FeatureBinning:
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        ok = ~np.isnan(v)
        vo, yo = v[ok], y[ok]
        edges = np.array([], dtype=float)
        if len(vo) > 0:
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(vo, qs))
            edges = edges[edges < vo.max()]
            edges = _merge_small_bins(edges, vo, yo, min_count)
            if self.monotonic:
                edges = _make_monotonic(edges, vo, yo)
        cnt, bad = _numeric_stats(edges, vo, yo)
        labels, counts, bads = interval_labels(edges), list(cnt), list(bad)
        if (~ok).any():
            labels.append(MISSING)
            counts.append(float((~ok).sum()))
            bads.append(float(y[~ok].sum()))
        table, iv = woe_table(labels, counts, bads, self.smoothing)
        return FeatureBinning(name, "numeric", edges=edges, woe=dict(zip(table.bin, table.woe)), table=table, iv=iv)

    def _fit_categorical(self, name: str, x: pd.Series, y: np.ndarray, min_count: int) -> FeatureBinning:
        s = pd.Series(x).astype("object").reset_index(drop=True)
        isna = pd.isna(s).to_numpy()
        counts = s[~isna].value_counts()
        groups = {str(cat): (str(cat) if n >= min_count else OTHER) for cat, n in counts.items()}
        lab = np.where(isna, MISSING, s.astype(str).map(groups).fillna(OTHER).to_numpy(dtype=object))
        stats = pd.DataFrame({"bin": lab, "y": y}).groupby("bin")["y"].agg(["size", "sum"])
        stats = stats.assign(rate=stats["sum"] / stats["size"]).sort_values("rate")
        table, iv = woe_table(list(stats.index), stats["size"], stats["sum"], self.smoothing)
        return FeatureBinning(
            name, "categorical", groups=groups, woe=dict(zip(table.bin, table.woe)), table=table, iv=iv
        )

    # --- application -----------------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self.binnings_[c].transform(X[c]) for c in X.columns}, index=X.index)

    def bin_labels(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self.binnings_[c].labels(X[c]) for c in X.columns}, index=X.index)

    @property
    def iv_(self) -> pd.Series:
        return pd.Series({c: b.iv for c, b in self.binnings_.items()}).sort_values(ascending=False)

    def table(self, name: str) -> pd.DataFrame:
        return self.binnings_[name].table

    def all_tables(self) -> pd.DataFrame:
        return pd.concat([b.table.assign(feature=c) for c, b in self.binnings_.items()], ignore_index=True)
