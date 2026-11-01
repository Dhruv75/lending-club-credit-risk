import numpy as np
import pandas as pd

from creditrisk.calibration import Calibrator
from creditrisk.metrics import auc, bootstrap_gini_ci, decile_table, gini, ks, lift_table, summarize
from creditrisk.monitoring import psi, psi_label
from creditrisk.policy import choose_cutoff, estimate_loss_params, evaluate_policy, profit_curve, realised_pnl


def _scores(n=20_000, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-1.7 + z)))
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_perfect_separation():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(y, p) == 1.0 and gini(y, p) == 1.0 and ks(y, p) == 1.0


def test_random_scores_have_no_skill():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50_000)
    assert abs(gini(y, rng.random(50_000))) < 0.03


def test_tables_and_ci():
    y, p = _scores()
    assert decile_table(y, p)["count"].sum() == len(y)
    lt = lift_table(y, p)
    assert lt.loc[0, "default_rate"] == lt["default_rate"].max()
    assert np.isclose(lt["cum_share_of_defaults"].iloc[-1], 1.0)
    lo, hi = bootstrap_gini_ci(y, p, n_boot=50)
    assert lo < gini(y, p) < hi
    s = summarize(y, p)
    assert s["ece"] < 0.02  # p is the true probability, so it is well calibrated


def test_platt_keeps_ranking_and_fixes_level():
    y, p = _scores()
    shifted = np.clip(p * 0.5, 1e-6, 1)  # systematically under-predicts
    cal = Calibrator("platt").fit(shifted, y)
    fixed = cal.transform(shifted)
    assert np.isclose(auc(y, fixed), auc(y, shifted))
    assert abs(fixed.mean() - y.mean()) < abs(shifted.mean() - y.mean())
    iso = Calibrator("isotonic").fit(shifted, y)
    assert abs(iso.transform(shifted).mean() - y.mean()) < 0.01


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=20_000))
    assert psi(a, pd.Series(rng.normal(size=20_000))) < 0.01
    assert psi(a, pd.Series(rng.normal(1.0, 1, 20_000))) > 0.25
    cats = pd.Series(rng.choice(["x", "y"], 10_000, p=[0.5, 0.5]))
    assert psi(cats, pd.Series(rng.choice(["x", "y"], 10_000, p=[0.9, 0.1]))) > 0.25
    assert psi_label(0.05) == "stable" and psi_label(0.2) == "moderate shift" and psi_label(0.3) == "major shift"


def test_cutoff_avoids_loss_making_loans():
    rng = np.random.default_rng(2)
    n = 10_000
    pd_ = rng.random(n)
    y = (rng.random(n) < pd_).astype(int)
    pnl = np.where(y == 1, -600.0, 150.0)  # a default loses 4x what a repaid loan earns
    funded = np.full(n, 1_000.0)
    cut = choose_cutoff(pd_, pnl, y, funded)
    assert 0.1 < cut["pd_threshold"] < 0.3  # break-even PD is 150 / 750 = 0.2
    res = evaluate_policy(pd_, pnl, y, funded, cut["pd_threshold"], benchmark_risk=rng.random(n))
    assert res["model"]["total_pnl"] > res["approve_all"]["total_pnl"]
    assert res["model"]["total_pnl"] > res["subgrade_same_rate"]["total_pnl"]
    curve = profit_curve(pd_, pnl, y, funded, rates=[1.0])
    assert np.isclose(curve.loc[0, "total_pnl"], pnl.sum())


def test_loss_parameters_and_pnl_on_synthetic_loans(sample_loans):
    lp = estimate_loss_params(sample_loans)
    assert 0 < lp["lgd"] <= 1 and 0 < lp["ead_ratio"] <= 1
    pnl = realised_pnl(sample_loans)
    assert pnl[sample_loans["default"] == 0].median() > 0
    assert pnl[sample_loans["default"] == 1].median() < 0
