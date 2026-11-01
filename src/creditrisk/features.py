"""Turn raw application fields into model features.

The same build_features() runs in training and in the app, so the model never sees a feature
computed one way in training and another way in production (training/serving skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .leakage import CATEGORICAL_FEATURES, INFORMATIVE_MISSING, NUMERIC_FEATURES

HOME_OWNERSHIP_LEVELS = ("MORTGAGE", "RENT", "OWN")
PASSTHROUGH_NUMERIC = (
    "inq_last_6mths",
    "delinq_2yrs",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "pub_rec",
    "pub_rec_bankruptcies",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "mort_acc",
    "acc_open_past_24mths",
    "bc_util",
    "tot_cur_bal",
)


def to_number(s: pd.Series) -> pd.Series:
    """Numeric conversion that tolerates strings such as ' 13.5%'."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    cleaned = s.astype("string").str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def parse_emp_length(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '3 years' -> 3, '10+ years' -> 10, 'n/a' -> missing."""
    text = s.astype("string").str.strip().str.lower()
    years = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce").astype("float64")
    under_one = text.str.startswith("<").fillna(False).astype(bool)
    return years.mask(under_one, 0.0)


def _month_index(d: pd.Series) -> pd.Series:
    d = pd.to_datetime(d, errors="coerce")
    return (d.dt.year * 12 + d.dt.month).astype("float64")


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.where(b > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return to_number(df[name])
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _text(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype("string").str.strip()
    return pd.Series(pd.NA, index=df.index, dtype="string")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw Lending Club columns -> the candidate feature matrix (see leakage.ALLOWED_FEATURES)."""
    X = pd.DataFrame(index=df.index)
    X["loan_amnt"] = _column(df, "loan_amnt")
    X["annual_inc"] = _column(df, "annual_inc")
    X["loan_to_income"] = _ratio(X["loan_amnt"], X["annual_inc"])
    dti = _column(df, "dti")
    X["dti"] = dti.where((dti >= 0) & (dti <= 100))  # the file uses -1 / 999 as "no usable value"
    X["fico"] = (_column(df, "fico_range_low") + _column(df, "fico_range_high")) / 2

    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        history = _month_index(df["issue_d"]) - _month_index(df["earliest_cr_line"])
        X["credit_history_months"] = history.where(history >= 0)
    else:
        X["credit_history_months"] = np.nan

    for name in PASSTHROUGH_NUMERIC:
        X[name] = _column(df, name)
    X["revol_bal_to_income"] = _ratio(X["revol_bal"], X["annual_inc"])
    X["emp_length_years"] = (
        parse_emp_length(df["emp_length"]) if "emp_length" in df.columns else pd.Series(np.nan, index=df.index)
    )

    home = _text(df, "home_ownership").str.upper()
    X["home_ownership"] = home.where(home.isin(HOME_OWNERSHIP_LEVELS) | home.isna(), "OTHER")
    X["verification_status"] = _text(df, "verification_status")
    X["purpose"] = _text(df, "purpose").str.lower()
    return X[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]


def select_available_features(
    train_X: pd.DataFrame, valid_X: pd.DataFrame, max_missing: float, max_shift: float
) -> tuple[list[str], dict[str, str]]:
    """Keep features that are actually populated, consistently over time.

    Some bureau fields only exist from 2012 onwards. In an out-of-time setup such a field is
    mostly missing in the older training years and fully populated later, so "missing" would
    secretly encode the issue date. Those features are dropped.
    """
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for c in train_X.columns:
        m_train = float(train_X[c].isna().mean())
        m_valid = float(valid_X[c].isna().mean())
        if m_train > max_missing and c not in INFORMATIVE_MISSING:
            dropped[c] = f"missing in {m_train:.0%} of training rows"
        elif abs(m_train - m_valid) > max_shift:
            dropped[c] = f"missing rate moved {m_train:.0%} -> {m_valid:.0%} (field availability changed)"
        elif train_X[c].nunique(dropna=True) <= 1:
            dropped[c] = "constant in training data"
        else:
            kept.append(c)
    return kept, dropped


def fit_categories(train_X: pd.DataFrame) -> dict[str, list[str]]:
    """Category levels seen in training; anything else becomes missing at scoring time."""
    return {
        c: sorted(str(v) for v in train_X[c].dropna().unique()) for c in train_X.columns if c in CATEGORICAL_FEATURES
    }


def apply_categories(X: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Give categorical columns a fixed pandas Categorical dtype (needed by LightGBM)."""
    X = X.copy()
    for c, levels in categories.items():
        if c in X.columns:
            values = X[c].astype("object").where(X[c].notna(), None)
            X[c] = pd.Categorical(values, categories=levels)
    return X
