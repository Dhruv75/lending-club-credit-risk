import pytest

from creditrisk.split import out_of_time_split


def test_splits_are_in_time_order_and_disjoint(sample_loans, cfg):
    idx = out_of_time_split(sample_loans["issue_d"], cfg)
    d = sample_loans["issue_d"]
    assert d.loc[idx["train"]].max() < d.loc[idx["valid"]].min()
    assert d.loc[idx["valid"]].max() < d.loc[idx["test"]].min()
    assert not set(idx["train"]) & set(idx["valid"]) and not set(idx["valid"]) & set(idx["test"])


def test_split_dates_must_increase(sample_loans, cfg):
    with pytest.raises(ValueError, match="increasing"):
        out_of_time_split(sample_loans["issue_d"], cfg.with_overrides(train_end="2016-06-01"))


def test_empty_split_is_reported(sample_loans, cfg):
    bad = cfg.with_overrides(train_end="2030-01-01", valid_end="2031-01-01", test_end="2032-01-01")
    with pytest.raises(ValueError, match="empty"):
        out_of_time_split(sample_loans["issue_d"], bad)
