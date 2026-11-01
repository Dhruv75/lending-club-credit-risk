"""Score one applicant with the saved bundle. Used by the Streamlit app (and testable without it).

Inputs use Lending Club's own column names and go through the same build_features() as training.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ROOT
from .features import apply_categories, build_features
from .leakage import FEATURE_DESCRIPTIONS
from .policy import expected_loss

BUNDLE_CANDIDATES = (ROOT / "models" / "bundle.joblib", ROOT / "runs" / "demo" / "models" / "bundle.joblib")

EXAMPLE_APPLICANT = {
    "loan_amnt": 12_000,
    "annual_inc": 65_000,
    "dti": 18.0,
    "fico_range_low": 690,
    "fico_range_high": 694,
    "earliest_cr_line": pd.Timestamp("2008-06-01"),
    "issue_d": pd.Timestamp("2026-10-01"),
    "inq_last_6mths": 1,
    "delinq_2yrs": 0,
    "mths_since_last_delinq": np.nan,
    "mths_since_last_record": np.nan,
    "pub_rec": 0,
    "pub_rec_bankruptcies": 0,
    "open_acc": 10,
    "total_acc": 22,
    "revol_bal": 11_000,
    "revol_util": 52.0,
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "verification_status": "Source Verified",
    "purpose": "debt_consolidation",
    "mort_acc": 0,
    "acc_open_past_24mths": 4,
    "bc_util": 55.0,
    "tot_cur_bal": 40_000,
}


def find_bundle() -> Path | None:
    """The bundle named by $CREDITRISK_BUNDLE, else the real-data model, else the demo model."""
    override = os.environ.get("CREDITRISK_BUNDLE")
    if override:
        return Path(override) if Path(override).exists() else None
    return next((p for p in BUNDLE_CANDIDATES if p.exists()), None)


def load_bundle(path: Path | None = None) -> dict:
    path = path or find_bundle()
    if path is None or not Path(path).exists():
        raise FileNotFoundError("No trained model found. Run the pipeline first (see README).")
    return joblib.load(path)


def score_applicant(bundle: dict, applicant: dict) -> dict:
    raw = pd.DataFrame([applicant])
    X = apply_categories(build_features(raw)[bundle["features"]], bundle["categories"])
    sc, gbm, cal = bundle["scorecard"], bundle["gbm"], bundle["calibrator"]

    pd_gbm = float(cal.transform(gbm.predict_raw(X))[0])
    score = float(sc.score(X)[0])
    pd_sc = float(sc.score_to_pd(np.array([score]))[0])
    shap, _ = gbm.shap_values(X)
    contrib = shap.iloc[0].sort_values(ascending=False)
    el = float(expected_loss(np.array([pd_gbm]), np.array([applicant["loan_amnt"]]), bundle["loss_params"])[0])
    approve = pd_gbm <= bundle["cutoff_pd"]
    return {
        "pd": pd_gbm,
        "scorecard_pd": pd_sc,
        "score": score,
        "decision": "Approve" if approve else "Decline",
        "cutoff_pd": float(bundle["cutoff_pd"]),
        "expected_loss": el,
        "reasons": [
            {**r, "description": FEATURE_DESCRIPTIONS.get(r["feature"], r["feature"])} for r in sc.reason_codes(X)[0]
        ],
        "shap": {k: float(v) for k, v in contrib.items()},
    }
