"""Synthetic loans in the exact format of the Kaggle Lending Club file (accepted_2007_to_2018Q4).

FOR TESTS AND THE DEMO ONLY. The relationships below are invented to look plausible so that
every stage of the pipeline has something realistic to chew on. Numbers from a model trained on
this file say nothing about real lending and must never be reported as results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

YEAR_WEIGHTS = {
    2007: 0.3, 2008: 0.6, 2009: 1.0, 2010: 2.0, 2011: 3.0,
    2012: 5.0, 2013: 10.0, 2014: 16.0, 2015: 24.0, 2016: 22.0,
}  # fmt: skip
# Shift in default log-odds by vintage: a stressed 2007-08, a benign 2010-12, then rising risk.
YEAR_SHIFT = {
    2007: 0.40, 2008: 0.30, 2009: 0.05, 2010: -0.10, 2011: -0.05,
    2012: 0.00, 2013: 0.05, 2014: 0.18, 2015: 0.30, 2016: 0.35,
}  # fmt: skip
PURPOSES = {
    "debt_consolidation": 0.58, "credit_card": 0.21, "home_improvement": 0.06, "other": 0.05,
    "major_purchase": 0.022, "small_business": 0.013, "car": 0.011, "medical": 0.011,
    "moving": 0.007, "vacation": 0.006, "house": 0.005, "wedding": 0.004, "renewable_energy": 0.001,
}  # fmt: skip
PURPOSE_RISK = {
    "small_business": 0.55, "moving": 0.20, "medical": 0.15, "house": 0.15, "other": 0.12,
    "renewable_energy": 0.10, "debt_consolidation": 0.05, "vacation": 0.0, "home_improvement": -0.05,
    "major_purchase": -0.05, "wedding": -0.10, "car": -0.15, "credit_card": -0.15,
}  # fmt: skip
EMP_LENGTHS = {
    "< 1 year": 0.08, "1 year": 0.065, "2 years": 0.09, "3 years": 0.08, "4 years": 0.06,
    "5 years": 0.065, "6 years": 0.05, "7 years": 0.045, "8 years": 0.045, "9 years": 0.04,
    "10+ years": 0.33, None: 0.05,
}  # fmt: skip
HOME = {"MORTGAGE": 0.49, "RENT": 0.40, "OWN": 0.105, "OTHER": 0.003, "NONE": 0.001, "ANY": 0.001}
GRADE_SHARES = {"A": 0.17, "B": 0.29, "C": 0.28, "D": 0.15, "E": 0.07, "F": 0.03, "G": 0.01}
STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "MD", "MA", "AZ", "WA"]
JOBS = ["Teacher", "Manager", "Registered Nurse", "Owner", "Supervisor", "Sales", "Driver", "Engineer"]
SNAPSHOT = pd.Timestamp("2019-03-01")  # when the synthetic statuses are "observed"


def _choice(rng: np.random.Generator, options: dict, n: int) -> np.ndarray:
    keys = list(options)
    p = np.array(list(options.values()), dtype=float)
    return np.array(keys, dtype=object)[rng.choice(len(keys), size=n, p=p / p.sum())]


def _balance(principal: np.ndarray, r: np.ndarray, payment: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Outstanding principal after k scheduled monthly payments."""
    growth = (1 + r) ** k
    return np.clip(principal * growth - payment * (growth - 1) / r, 0, None)


def _month_str(year_month_index: np.ndarray) -> np.ndarray:
    years, months = np.divmod(year_month_index.astype(int) - 1, 12)
    parts = pd.DataFrame({"year": years, "month": months + 1, "day": np.ones_like(years)})
    return pd.to_datetime(parts).dt.strftime("%b-%Y").to_numpy()


def generate(n: int = 80_000, seed: int = 7) -> pd.DataFrame:
    """Return n synthetic loans (plus two junk footer rows, as in the original LC files)."""
    rng = np.random.default_rng(seed)
    years = np.array(list(YEAR_WEIGHTS))
    w = np.array(list(YEAR_WEIGHTS.values()))
    year = rng.choice(years, size=n, p=w / w.sum())
    month = np.where(year == 2007, rng.integers(6, 13, n), rng.integers(1, 13, n))
    issue_idx = year * 12 + month  # month index: year*12 + month
    term = np.where((year >= 2010) & (rng.random(n) < 0.27), 60, 36)

    # --- applicant profile ------------------------------------------------------------------
    fico_low = np.clip(660 + np.round(rng.gamma(2.0, 18.0, n) / 5) * 5, 660, 845)
    fico = fico_low + 2
    t = (issue_idx - (2011 * 12 + 1)) / 12.0  # years since Jan-2011, drives gentle population drift
    income = np.round(np.clip(np.exp(rng.normal(np.log(65_000) + 0.02 * t, 0.55, n)), 8_000, 1_500_000), -2)
    loan = np.round(np.clip(np.exp(rng.normal(np.log(12_000) + 0.05 * t, 0.6, n)), 1_000, 35_000) / 25) * 25
    dti = np.round(np.clip(rng.gamma(4.2, 4.3, n) + 0.45 * t, 0, 39.99), 2)
    inq = np.clip(rng.poisson(np.clip(0.95 - 0.06 * t, 0.4, None), n), 0, 8)
    delinq = rng.poisson(0.32, n)
    ever_delinq = (delinq > 0) | (rng.random(n) < 0.42)
    mths_delinq = np.where(delinq > 0, rng.integers(1, 24, n), rng.integers(1, 120, n)).astype(float)
    mths_delinq[~ever_delinq] = np.nan
    has_record = rng.random(n) < 0.16
    pub_rec = np.where(has_record, 1 + rng.poisson(0.2, n), 0)
    mths_record = np.where(has_record, rng.integers(1, 120, n), np.nan)
    bankruptcies = np.where(has_record & (rng.random(n) < 0.8), 1.0, 0.0)
    open_acc = 1 + rng.poisson(10.5, n)
    total_acc = open_acc + rng.poisson(13, n)
    revol_bal = np.round(np.clip(np.exp(rng.normal(np.log(11_000), 0.9, n)), 0, 800_000))
    revol_util = np.round(np.clip(rng.beta(2.2, 2.0, n) * 100, 0, 130), 1)
    history = np.clip(rng.gamma(3.5, 55, n) + (fico_low - 660) * 0.6, 36, 700).astype(int)
    emp = _choice(rng, EMP_LENGTHS, n)
    home = _choice(rng, HOME, n)
    purpose = _choice(rng, PURPOSES, n)
    p_verified = np.clip(0.25 + loan / 35_000 * 0.55, 0, 0.95)
    u = rng.random(n)
    verification = np.where(
        u < p_verified * 0.55, "Verified", np.where(u < p_verified, "Source Verified", "Not Verified")
    )
    mort_acc = np.where(home == "MORTGAGE", 1 + rng.poisson(1.5, n), rng.poisson(0.3, n)).astype(float)
    acc_24m = rng.poisson(4.2, n).astype(float)
    bc_util = np.round(np.clip(revol_util + rng.normal(0, 12, n), 0, 150), 1)
    tot_cur_bal = np.round(np.exp(rng.normal(np.log(40_000), 0.9, n)) * np.where(home == "MORTGAGE", 4.0, 1.0))
    # These bureau fields only exist for loans issued from Aug-2012 (an availability shift).
    newer = issue_idx >= 2012 * 12 + 8
    for arr in (mort_acc, acc_24m, bc_util, tot_cur_bal):
        arr[~newer] = np.nan

    # --- true default risk ----------------------------------------------------------------------
    lti = loan / income
    observable = (
        -2.50
        - 0.70 * (fico - 700) / 30
        + 0.28 * (dti - 18) / 8
        + 0.20 * np.minimum(inq, 6)
        + 0.10 * np.minimum(delinq, 4)
        + 0.90 * (revol_util / 100 - 0.5)
        + 2.5 * (np.minimum(lti, 0.6) - 0.2)
        - 0.12 * np.log(income / 65_000)
        + 0.25 * pd.isna(emp)
        + 0.08 * (emp == "< 1 year")
        + 0.22 * (home == "RENT")
        - 0.05 * (home == "OWN")
        + np.vectorize(PURPOSE_RISK.get)(purpose)
        + 0.15 * (verification != "Not Verified")
        + 0.25 * (pub_rec > 0)
        - 0.002 * (history - 180)
        + 0.45 * (term == 60)
    )
    lc_private = rng.normal(0, 0.30, n)  # what only Lending Club's own grading sees
    hidden = rng.normal(0, 0.50, n)  # what nobody observes
    shift = np.vectorize(YEAR_SHIFT.get)(year)
    p_default = 1 / (1 + np.exp(-(observable + lc_private + hidden + shift)))
    will_default = rng.random(n) < p_default

    # --- Lending Club's grade and price (relative within each vintage) ---------------------------
    lc_score = observable + lc_private + rng.normal(0, 0.30, n)
    cuts = np.cumsum(list(GRADE_SHARES.values()))[:-1]
    sub_index = np.zeros(n, dtype=int)
    for y in np.unique(year):
        m = year == y
        q = pd.Series(lc_score[m]).rank(pct=True).to_numpy()
        g = np.searchsorted(cuts, q)  # 0..6
        lo = np.concatenate([[0], cuts])[g]
        hi = np.concatenate([cuts, [1.0]])[g]
        within = np.clip(((q - lo) / (hi - lo) * 5).astype(int), 0, 4)
        sub_index[m] = g * 5 + within
    sub_grades = np.array([f"{g}{k}" for g in GRADE_SHARES for k in range(1, 6)], dtype=object)
    sub_grade = sub_grades[sub_index]
    grade = np.array([s[0] for s in sub_grade], dtype=object)
    int_rate = np.round(5.3 + sub_index * 0.72 + rng.normal(0, 0.25, n) + 0.6 * (year <= 2013), 2)
    r = int_rate / 1200
    installment = np.round(loan * r / (1 - (1 + r) ** (-term)), 2)

    # --- repayment history as observed at SNAPSHOT --------------------------------------------------
    snapshot_idx = SNAPSHOT.year * 12 + SNAPSHOT.month
    elapsed = snapshot_idx - issue_idx
    default_month = np.clip(np.round(rng.gamma(2.2, 7.5, n)), 3, term).astype(int)
    prepays = rng.random(n) < 0.35
    end_month = np.where(prepays, rng.integers(3, np.maximum(term - 1, 4), n), term)
    status = np.full(n, "Current", dtype=object)
    paid_k = np.minimum(elapsed, term).astype(int)  # payments made so far

    bad_done = will_default & (default_month + 5 <= elapsed)
    bad_late = will_default & (default_month <= elapsed) & ~bad_done
    good_done = ~will_default & (end_month <= elapsed)
    status[bad_done] = "Charged Off"
    status[bad_done & (default_month + 5 > elapsed - 1) & (rng.random(n) < 0.3)] = "Default"
    status[bad_late] = "Late (31-120 days)"
    status[good_done] = "Fully Paid"
    paid_k = np.where(will_default & (default_month <= elapsed), default_month, paid_k)
    paid_k = np.where(good_done, end_month, paid_k)
    old = issue_idx < 2010 * 12 + 11
    not_policy = old & (rng.random(n) < 0.04)
    status[not_policy & (status == "Fully Paid")] = "Does not meet the credit policy. Status:Fully Paid"
    status[not_policy & (status == "Charged Off")] = "Does not meet the credit policy. Status:Charged Off"

    remaining = _balance(loan, r, installment, paid_k)
    rec_prncp = loan - remaining
    rec_int = np.clip(installment * paid_k - rec_prncp, 0, None)
    finished = good_done
    rec_prncp = np.where(finished, loan, rec_prncp)  # prepayment clears the balance
    out_prncp = np.where(finished | bad_done, 0.0, remaining)
    charged = np.isin(status, ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"])
    recoveries = np.where(charged & (rng.random(n) < 0.75), remaining * rng.uniform(0.02, 0.25, n), 0.0)
    recovery_fee = recoveries * 0.18
    late_fee = np.where(will_default & (rng.random(n) < 0.3), rng.uniform(15, 60, n), 0.0)
    total_pymnt = rec_prncp + rec_int + late_fee + recoveries
    last_pymnt_idx = issue_idx + np.maximum(paid_k, 1)
    last_fico_low = np.where(
        will_default & (default_month <= elapsed),
        np.clip(np.round((fico_low - rng.normal(85, 60, n)) / 5) * 5, 300, 845),
        np.clip(np.round((fico_low + rng.normal(5, 35, n)) / 5) * 5, 500, 845),
    )

    df = pd.DataFrame(
        {
            "id": (68_000_000 + np.arange(n)).astype(str),
            "member_id": np.nan,
            "loan_amnt": loan,
            "funded_amnt": loan,
            "funded_amnt_inv": np.round(loan - np.where(rng.random(n) < 0.1, rng.uniform(0, 200, n), 0), 2),
            "term": np.where(term == 36, " 36 months", " 60 months"),
            "int_rate": int_rate,
            "installment": installment,
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_title": np.where(rng.random(n) < 0.9, np.array(JOBS, dtype=object)[rng.integers(0, 8, n)], None),
            "emp_length": emp,
            "home_ownership": home,
            "annual_inc": income,
            "verification_status": verification,
            "issue_d": _month_str(issue_idx),
            "loan_status": status,
            "pymnt_plan": "n",
            "url": "https://lendingclub.com/browse/loanDetail.action?loan_id=0",
            "desc": None,
            "purpose": purpose,
            "title": np.char.replace(purpose.astype(str), "_", " "),
            "zip_code": [f"{z:03d}xx" for z in rng.integers(10, 999, n)],
            "addr_state": np.array(STATES, dtype=object)[rng.integers(0, len(STATES), n)],
            "dti": dti,
            "delinq_2yrs": delinq.astype(float),
            "earliest_cr_line": _month_str(issue_idx - history),
            "fico_range_low": fico_low,
            "fico_range_high": fico_low + 4,
            "inq_last_6mths": inq.astype(float),
            "mths_since_last_delinq": mths_delinq,
            "mths_since_last_record": mths_record,
            "open_acc": open_acc.astype(float),
            "pub_rec": pub_rec.astype(float),
            "revol_bal": revol_bal,
            "revol_util": revol_util,
            "total_acc": total_acc.astype(float),
            "initial_list_status": np.where(rng.random(n) < 0.5, "w", "f"),
            "out_prncp": np.round(out_prncp, 2),
            "out_prncp_inv": np.round(out_prncp, 2),
            "total_pymnt": np.round(total_pymnt, 2),
            "total_pymnt_inv": np.round(total_pymnt, 2),
            "total_rec_prncp": np.round(rec_prncp, 2),
            "total_rec_int": np.round(rec_int, 2),
            "total_rec_late_fee": np.round(late_fee, 2),
            "recoveries": np.round(recoveries, 2),
            "collection_recovery_fee": np.round(recovery_fee, 2),
            "last_pymnt_d": _month_str(last_pymnt_idx),
            "last_pymnt_amnt": installment,
            "next_pymnt_d": None,
            "last_credit_pull_d": "Mar-2019",
            "last_fico_range_high": last_fico_low + 4,
            "last_fico_range_low": last_fico_low,
            "policy_code": 1.0,
            "application_type": "Individual",
            "acc_open_past_24mths": acc_24m,
            "bc_util": bc_util,
            "mort_acc": mort_acc,
            "pub_rec_bankruptcies": bankruptcies,
            "tot_cur_bal": tot_cur_bal,
            "hardship_flag": "N",
            "debt_settlement_flag": "N",
        }
    )
    # A few messy values that the real file also contains.
    messy = rng.choice(n, size=max(3, n // 5000), replace=False)
    df.loc[messy[: len(messy) // 3], "dti"] = 999.0
    df.loc[messy[len(messy) // 3 : 2 * len(messy) // 3], "annual_inc"] = 0.0
    df.loc[messy[2 * len(messy) // 3 :], "revol_util"] = np.nan
    footer = pd.DataFrame(
        {"id": ["Total amount funded in policy code 1: 1741781700", "Total amount funded in policy code 2: 0"]}
    )
    return pd.concat([df, footer], ignore_index=True)


def write_synthetic(path: str | Path, n: int = 80_000, seed: int = 7) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(n, seed).to_csv(path, index=False)
    return path
