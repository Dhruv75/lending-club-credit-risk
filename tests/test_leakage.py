import joblib
import pytest

from creditrisk import leakage
from creditrisk.features import build_features
from creditrisk.leakage import (
    ALLOWED_FEATURES,
    LENDER_DECISION,
    POST_ORIGINATION,
    LeakageError,
    check_features,
    find_leaks,
)


def test_reviewed_allowlist_is_clean():
    check_features(ALLOWED_FEATURES)


@pytest.mark.parametrize(
    "column", ["total_pymnt", "recoveries", "last_fico_range_high", "loan_status", "out_prncp", "settlement_amount"]
)
def test_post_origination_columns_are_blocked(column):
    with pytest.raises(LeakageError, match="after the loan was issued"):
        check_features(["fico", column])


@pytest.mark.parametrize("column", ["int_rate", "grade", "sub_grade", "installment", "funded_amnt"])
def test_lender_decision_columns_are_blocked(column):
    with pytest.raises(LeakageError, match="lender's own decision"):
        check_features([column])


@pytest.mark.parametrize("column", ["zip_code", "addr_state", "issue_d", "default", "emp_title"])
def test_identifiers_geography_and_target_are_blocked(column):
    with pytest.raises(LeakageError):
        check_features([column])


def test_unreviewed_feature_is_blocked_by_default():
    with pytest.raises(LeakageError, match="allowlist"):
        check_features(["shiny_new_feature"])


def test_lineage_check_catches_a_leak_hidden_in_an_engineered_feature(monkeypatch):
    monkeypatch.setitem(leakage.ENGINEERED_SOURCES, "payment_ratio", ("total_pymnt", "loan_amnt"))
    monkeypatch.setattr(leakage, "ALLOWED_FEATURES", leakage.ALLOWED_FEATURES | {"payment_ratio"})
    problems = find_leaks(["payment_ratio"])
    assert any("total_pymnt" in p for p in problems)


def test_feature_builder_only_emits_allowed_columns(sample_loans):
    X = build_features(sample_loans)
    assert set(X.columns) <= ALLOWED_FEATURES
    check_features(X.columns)


def test_trained_models_only_use_application_time_features(pipeline_run, cfg):
    bundle = joblib.load(cfg.models_dir / "bundle.joblib")
    for features in (bundle["features"], bundle["scorecard"].features_, bundle["gbm"].features_):
        check_features(features)
        assert not set(features) & (POST_ORIGINATION | LENDER_DECISION)
