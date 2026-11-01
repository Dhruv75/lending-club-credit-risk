"""From probabilities to money: expected loss, and an approval cutoff chosen on realised profit.

Expected loss per loan: EL = PD x LGD x EAD
  * PD  - probability of default (the model)
  * EAD - exposure at default: how much is still owed when the borrower stops paying
  * LGD - loss given default: the share of that exposure never recovered
LGD and EAD are estimated from defaulted loans in the TRAINING period. Their repayment columns
are post-origination, which is fine here: they are used to measure outcomes, not as features.

The cutoff is chosen on the validation period by back-testing: "had we approved only loans with
PD below t, what profit would those loans have actually made?" and then applied, unchanged, to
the test period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .leakage import TARGET


def estimate_loss_params(loans: pd.DataFrame) -> dict[str, float]:
    """Exposure-weighted EAD ratio and LGD from defaulted training loans."""
    d = loans[loans[TARGET] == 1]
    funded = d["funded_amnt"].fillna(d["loan_amnt"])
    ead = (funded - d["total_rec_prncp"].fillna(0)).clip(lower=0)
    net_recovery = (d["recoveries"].fillna(0) - d["collection_recovery_fee"].fillna(0)).clip(lower=0)
    ead_ratio = float(ead.sum() / funded.sum())
    lgd = float(np.clip(1 - net_recovery.sum() / ead.sum(), 0, 1))
    return {"ead_ratio": ead_ratio, "lgd": lgd, "n_defaults": int(len(d))}


def expected_loss(pd_: np.ndarray, loan_amnt: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return np.asarray(pd_) * params["lgd"] * params["ead_ratio"] * np.asarray(loan_amnt)


def realised_credit_loss(loans: pd.DataFrame) -> pd.Series:
    """Principal never recovered on defaulted loans (0 for loans that were repaid)."""
    funded = loans["funded_amnt"].fillna(loans["loan_amnt"])
    ead = (funded - loans["total_rec_prncp"].fillna(0)).clip(lower=0)
    net_recovery = (loans["recoveries"].fillna(0) - loans["collection_recovery_fee"].fillna(0)).clip(lower=0)
    return ((ead - net_recovery).clip(lower=0) * (loans[TARGET] == 1)).astype(float)


def realised_pnl(loans: pd.DataFrame) -> pd.Series:
    """Cash received minus cash lent (interest earned, minus principal lost). No discounting."""
    funded = loans["funded_amnt"].fillna(loans["loan_amnt"])
    return (loans["total_pymnt"].fillna(0) - loans["collection_recovery_fee"].fillna(0) - funded).astype(float)


def _ranked(risk: np.ndarray, seed: int = 0) -> np.ndarray:
    """Safest first; random tie-break so coarse scores (like 35 sub-grades) are treated fairly."""
    tiebreak = np.random.default_rng(seed).random(len(risk))
    return np.lexsort((tiebreak, np.asarray(risk, dtype=float)))


def profit_curve(risk, pnl, y, funded, rates=None) -> pd.DataFrame:
    """Approve the safest r% of loans for each approval rate r; report realised results."""
    rates = np.round(np.arange(0.01, 1.0001, 0.01), 2) if rates is None else np.asarray(rates)
    order = _ranked(risk)
    risk_s = np.asarray(risk, dtype=float)[order]
    cum_pnl = np.cumsum(np.asarray(pnl, dtype=float)[order])
    cum_bad = np.cumsum(np.asarray(y, dtype=float)[order])
    cum_funded = np.cumsum(np.asarray(funded, dtype=float)[order])
    n = len(order)
    rows = []
    for r in rates:
        k = int(np.clip(round(r * n), 1, n))
        rows.append(
            {
                "approval_rate": k / n,
                "risk_threshold": float(risk_s[k - 1]),
                "approved": k,
                "default_rate": float(cum_bad[k - 1] / k),
                "total_pnl": float(cum_pnl[k - 1]),
                "return_on_funded": float(cum_pnl[k - 1] / cum_funded[k - 1]),
            }
        )
    return pd.DataFrame(rows)


def choose_cutoff(pd_valid, pnl_valid, y_valid, funded_valid) -> dict:
    """PD threshold that maximised realised profit on the validation period."""
    curve = profit_curve(pd_valid, pnl_valid, y_valid, funded_valid)
    best = curve.loc[curve["total_pnl"].idxmax()]
    return {
        "pd_threshold": float(best["risk_threshold"]),
        "valid_approval_rate": float(best["approval_rate"]),
        "valid_total_pnl": float(best["total_pnl"]),
    }


def evaluate_policy(pd_test, pnl_test, y_test, funded_test, threshold: float, benchmark_risk=None) -> dict:
    """Apply the fixed PD cutoff to the test period and compare with simple alternatives."""
    pd_test = np.asarray(pd_test, dtype=float)
    pnl_test = np.asarray(pnl_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    funded_test = np.asarray(funded_test, dtype=float)
    approve = pd_test <= threshold
    k = int(approve.sum())
    res = {
        "approve_all": {
            "approval_rate": 1.0,
            "default_rate": float(y_test.mean()),
            "total_pnl": float(pnl_test.sum()),
            "return_on_funded": float(pnl_test.sum() / funded_test.sum()),
        },
        "model": {
            "approval_rate": k / len(pd_test),
            "default_rate": float(y_test[approve].mean()) if k else float("nan"),
            "total_pnl": float(pnl_test[approve].sum()),
            "return_on_funded": float(pnl_test[approve].sum() / funded_test[approve].sum()) if k else float("nan"),
            "defaults_avoided_share": float(y_test[~approve].sum() / y_test.sum()),
            "goods_rejected_share": float(((1 - y_test)[~approve]).sum() / (1 - y_test).sum()),
        },
    }
    res["model"]["pnl_gain_vs_all"] = res["model"]["total_pnl"] - res["approve_all"]["total_pnl"]
    if benchmark_risk is not None:
        # Same number of approvals, ranked by Lending Club's sub-grade instead: a like-for-like test.
        order = _ranked(benchmark_risk)[:k]
        res["subgrade_same_rate"] = {
            "approval_rate": k / len(pd_test),
            "default_rate": float(y_test[order].mean()) if k else float("nan"),
            "total_pnl": float(pnl_test[order].sum()),
            "return_on_funded": float(pnl_test[order].sum() / funded_test[order].sum()) if k else float("nan"),
        }
        res["model"]["pnl_gain_vs_subgrade"] = res["model"]["total_pnl"] - res["subgrade_same_rate"]["total_pnl"]
    return res
