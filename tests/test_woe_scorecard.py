import numpy as np
import pandas as pd
import pytest

from creditrisk.config import Config
from creditrisk.woe import MISSING, WoEBinner


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1])
    logit = -1.6 + 1.0 * x + np.where(cat == "c", 0.8, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "cat": cat})
    X.loc[rng.choice(n, 500, replace=False), "x"] = np.nan
    return X, y


def test_riskier_bins_get_lower_woe(toy):
    X, y = toy
    t = WoEBinner().fit(X, y).table("x")
    assert np.corrcoef(t["bad_rate"], t["woe"])[0, 1] < -0.95


def test_information_value_separates_signal_from_noise(toy):
    X, y = toy
    iv = WoEBinner().fit(X, y).iv_
    assert iv["x"] > 0.3
    assert iv["noise"] < 0.02


def test_numeric_bins_are_monotonic_and_large_enough(toy):
    X, y = toy
    b = WoEBinner(min_bin_share=0.05).fit(X, y)
    t = b.table("x")
    body = t[t["bin"] != MISSING]
    assert body["bad_rate"].is_monotonic_increasing
    assert (body["share"] >= 0.05 - 1e-9).all()
    assert MISSING in set(t["bin"])


def test_unseen_and_missing_categories_are_handled(toy):
    X, y = toy
    b = WoEBinner().fit(X, y)
    new = pd.DataFrame({"x": [np.nan, 0.0], "noise": [0.0, 0.0], "cat": ["never-seen", None]})
    labels = b.bin_labels(new)
    assert labels.loc[0, "x"] == MISSING and labels.loc[1, "cat"] == MISSING
    assert np.isfinite(b.transform(new).to_numpy()).all()


def test_scorecard_points_reproduce_the_logistic_regression(pipeline_run, cfg, sample_loans):
    import joblib

    from creditrisk.features import apply_categories, build_features

    bundle = joblib.load(cfg.models_dir / "bundle.joblib")
    sc = bundle["scorecard"]
    X = apply_categories(build_features(sample_loans.head(500))[bundle["features"]], bundle["categories"])
    woe = sc.binner_.transform(X[sc.features_])
    lr = sc.model_.predict_proba(woe.to_numpy())[:, 1]
    assert np.allclose(sc.predict_proba(X), lr, atol=1e-9)
    assert (sc.coef_ < 0).all()


def test_score_scaling_matches_definition(pipeline_run, cfg):
    import joblib

    sc = joblib.load(cfg.models_dir / "bundle.joblib")["scorecard"]
    c = Config()
    pd_base = sc.score_to_pd(np.array([c.base_score]))[0]
    assert np.isclose((1 - pd_base) / pd_base, c.base_odds)
    pd_up = sc.score_to_pd(np.array([c.base_score + c.pdo]))[0]
    assert np.isclose((1 - pd_up) / pd_up, 2 * c.base_odds)
