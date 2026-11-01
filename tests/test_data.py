import numpy as np
import pandas as pd

from creditrisk.data import BAD_STATUSES, GOOD_STATUSES, read_population
from creditrisk.features import build_features, parse_emp_length, to_number

HEADER = (
    "id,loan_amnt,funded_amnt,term,int_rate,sub_grade,emp_length,home_ownership,annual_inc,verification_status,"
    "issue_d,loan_status,purpose,dti,earliest_cr_line,fico_range_low,fico_range_high,revol_util,total_pymnt,"
    "total_rec_prncp,recoveries,collection_recovery_fee"
)
ROWS = [
    # resolved, 36 months, matured -> kept
    "1,10000,10000, 36 months, 10.65%,B2,10+ years,RENT,24000,Verified,Dec-2011,Fully Paid,credit_card,27.65,"
    "Jan-1985,735,739,83.7%,11500,10000,0,0",
    "2,2500,2500, 36 months, 15.27%,C4,< 1 year,RENT,30000,Source Verified,Dec-2011,Charged Off,car,1.0,"
    "Apr-1999,740,744,9.4%,1000,450,120,10",
    # 60-month -> dropped
    "3,2400,2400, 60 months, 15.96%,C5,10+ years,RENT,12252,Not Verified,Dec-2011,Fully Paid,small_business,8.72,"
    "Nov-2001,735,739,98.5%,3000,2400,0,0",
    # still running -> dropped
    "4,5000,5000, 36 months, 7.9%,A4,3 years,RENT,36000,Source Verified,Dec-2017,Current,wedding,11.2,"
    "Nov-2004,730,734,28.3%,500,400,0,0",
]


def _write(tmp_path, with_prospectus_line: bool):
    lines = (
        ["Notes offered by Prospectus (https://www.lendingclub.com/info/prospectus.action)"]
        if with_prospectus_line
        else []
    )
    lines += [HEADER, *ROWS, '"Total amount funded in policy code 1: 19900",,,,,,,,,,,,,,,,,,,,,']
    path = tmp_path / ("lc_original.csv" if with_prospectus_line else "lc_kaggle.csv")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_loader_keeps_only_resolved_matured_36_month_loans(tmp_path, cfg):
    for prospectus in (True, False):
        df, funnel = read_population(_write(tmp_path, prospectus), cfg)
        assert df["id"].tolist() == ["1", "2"]
        assert df["default"].tolist() == [0, 1]
        assert funnel["all_loans"] == 4 and funnel["term_ok"] == 3 and funnel["resolved"] == 2


def test_loader_parses_percent_strings_and_dates(tmp_path, cfg):
    df, _ = read_population(_write(tmp_path, False), cfg)
    assert np.isclose(df.loc[0, "int_rate"], 10.65)
    assert np.isclose(df.loc[0, "revol_util"], 83.7)
    assert df.loc[0, "issue_d"] == pd.Timestamp("2011-12-01")
    assert df.loc[0, "earliest_cr_line"] == pd.Timestamp("1985-01-01")


def test_population_on_synthetic_file(sample_loans, cfg):
    assert set(sample_loans["loan_status"]) <= set(GOOD_STATUSES + BAD_STATUSES)
    assert sample_loans["default"].isin([0, 1]).all()
    bad = sample_loans["loan_status"].isin(BAD_STATUSES)
    assert (sample_loans.loc[bad, "default"] == 1).all() and (sample_loans.loc[~bad, "default"] == 0).all()
    snap = pd.Timestamp(cfg.snapshot_date)
    end_month = sample_loans["issue_d"].dt.year * 12 + sample_loans["issue_d"].dt.month + cfg.term_months
    assert (end_month <= snap.year * 12 + snap.month).all()


def test_emp_length_parsing():
    out = parse_emp_length(pd.Series(["< 1 year", "1 year", "10+ years", "n/a", None, "5 years"]))
    assert out.iloc[:3].tolist() == [0.0, 1.0, 10.0]
    assert out.iloc[3:5].isna().all()
    assert out.iloc[5] == 5.0


def test_to_number_handles_percent_strings():
    assert to_number(pd.Series([" 13.5%", "7", None, "abc"])).tolist()[:2] == [13.5, 7.0]


def test_feature_builder_cleans_bad_values():
    raw = pd.DataFrame(
        {
            "loan_amnt": [10_000, 10_000],
            "annual_inc": [50_000, 0],
            "dti": [20.0, 999.0],
            "fico_range_low": [700, 660],
            "fico_range_high": [704, 664],
            "issue_d": pd.to_datetime(["2015-06-01", "2015-06-01"]),
            "earliest_cr_line": pd.to_datetime(["2005-06-01", "2016-01-01"]),
            "home_ownership": ["ANY", "MORTGAGE"],
            "purpose": ["Credit_Card", "car"],
        }
    )
    X = build_features(raw)
    assert X.loc[0, "loan_to_income"] == 0.2 and np.isnan(X.loc[1, "loan_to_income"])
    assert np.isnan(X.loc[1, "dti"])
    assert X.loc[0, "fico"] == 702
    assert X.loc[0, "credit_history_months"] == 120 and np.isnan(X.loc[1, "credit_history_months"])
    assert X.loc[0, "home_ownership"] == "OTHER" and X.loc[0, "purpose"] == "credit_card"


def test_find_raw_file_looks_anywhere_under_data_raw(tmp_path):
    import pytest

    from creditrisk.config import Config
    from creditrisk.data import find_raw_file

    cfg = Config(raw_path=tmp_path / "raw" / "accepted_2007_to_2018Q4.csv.gz")
    nested = tmp_path / "raw" / "accepted_2007_to_2018q4.csv"  # Kaggle unzips into a folder with this name
    nested.mkdir(parents=True)
    (nested / "accepted_2007_to_2018Q4.csv").write_text("x")
    assert find_raw_file(cfg) == nested / "accepted_2007_to_2018Q4.csv"
    with pytest.raises(FileNotFoundError, match="wordsforthewise"):
        find_raw_file(Config(raw_path=tmp_path / "empty" / "accepted_2007_to_2018Q4.csv.gz"))
