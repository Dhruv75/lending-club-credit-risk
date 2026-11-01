"""Score a single applicant.  Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works even without `pip install -e .`

from creditrisk.serving import EXAMPLE_APPLICANT, find_bundle, load_bundle, score_applicant  # noqa: E402

EMP_OPTIONS = ["< 1 year"] + [f"{i} year{'s' if i > 1 else ''}" for i in range(1, 10)] + ["10+ years", "n/a"]

st.set_page_config(page_title="Credit risk scorer", page_icon="📉", layout="wide")


@st.cache_resource
def _bundle(path: str) -> dict:
    return load_bundle(Path(path))


path = find_bundle()
if path is None:
    st.error(
        "No trained model found. Run `python -m creditrisk.pipeline --demo` for a synthetic demo model, "
        "or train on the real Lending Club file first (see README)."
    )
    st.stop()
bundle = _bundle(str(path))

st.title("Lending Club credit risk scorer")
st.caption(
    f"Trained on loans issued {bundle['trained_on']}. Probability of default (PD) over a 36-month loan, "
    "from a calibrated LightGBM model; reasons come from the WoE scorecard."
)
if bundle.get("synthetic"):
    st.warning("This model was trained on SYNTHETIC demo data, so its predictions mean nothing about real borrowers.")

with st.sidebar:
    st.header("Applicant")
    loan = st.number_input("Loan amount ($)", 1_000, 40_000, 12_000, step=500)
    income = st.number_input("Annual income ($)", 5_000, 2_000_000, 65_000, step=1_000)
    dti = st.slider("Debt-to-income, excluding this loan (%)", 0.0, 45.0, 18.0, 0.5)
    fico = st.slider("FICO score", 660, 850, 692)
    history_years = st.slider("Years since first credit line", 1, 50, 18)
    inq = st.slider("Credit inquiries, last 6 months", 0, 8, 1)
    delinq = st.slider("30+ day delinquencies, last 2 years", 0, 10, 0)
    months_delinq = st.number_input("Months since last delinquency (0 = never)", 0, 200, 0)
    revol_util = st.slider("Revolving utilisation (%)", 0.0, 120.0, 52.0, 0.5)
    revol_bal = st.number_input("Revolving balance ($)", 0, 1_000_000, 11_000, step=500)
    open_acc = st.slider("Open credit lines", 1, 60, 10)
    total_acc = st.slider("Total credit lines ever", 1, 120, 22)
    pub_rec = st.slider("Derogatory public records", 0, 10, 0)
    months_record = st.number_input("Months since last public record (0 = never)", 0, 200, 0)
    bankruptcies = st.slider("Public-record bankruptcies", 0, 5, 0)
    emp = st.selectbox("Employment length", EMP_OPTIONS, index=5)
    home = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    verification = st.selectbox("Income verification", ["Not Verified", "Source Verified", "Verified"], index=1)
    purposes = bundle["categories"].get("purpose") or ["debt_consolidation"]
    default_purpose = purposes.index("debt_consolidation") if "debt_consolidation" in purposes else 0
    purpose = st.selectbox("Purpose", purposes, index=default_purpose)

today = pd.Timestamp.today().normalize().replace(day=1)
applicant = {
    **EXAMPLE_APPLICANT,
    "loan_amnt": loan,
    "annual_inc": income,
    "dti": dti,
    "fico_range_low": fico - 2,
    "fico_range_high": fico + 2,
    "issue_d": today,
    "earliest_cr_line": today - pd.DateOffset(years=history_years),
    "inq_last_6mths": inq,
    "delinq_2yrs": delinq,
    "mths_since_last_delinq": months_delinq or np.nan,
    "mths_since_last_record": months_record or np.nan,
    "revol_util": revol_util,
    "revol_bal": revol_bal,
    "open_acc": open_acc,
    "total_acc": total_acc,
    "pub_rec": pub_rec,
    "pub_rec_bankruptcies": bankruptcies,
    "emp_length": None if emp == "n/a" else emp,
    "home_ownership": home,
    "verification_status": verification,
    "purpose": purpose,
}
res = score_applicant(bundle, applicant)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Probability of default", f"{res['pd']:.1%}")
c2.metric(
    "Scorecard points", f"{res['score']:.0f}", help="600 points = 50:1 odds of repaying; +20 points doubles the odds."
)
c3.metric(
    "Decision", res["decision"], help=f"Approve if PD <= {res['cutoff_pd']:.1%}, a cutoff chosen on validation data."
)
c4.metric("Expected loss", f"${res['expected_loss']:,.0f}", help="PD x loss given default x exposure at default.")

left, right = st.columns(2)
with left:
    st.subheader("Main reasons (scorecard)")
    if res["reasons"]:
        st.table(
            pd.DataFrame(
                [
                    {"Reason": r["description"], "Applicant's band": r["bin"], "Points lost": round(r["points_lost"])}
                    for r in res["reasons"]
                ]
            )
        )
    else:
        st.write("No feature costs this applicant meaningful points.")
with right:
    st.subheader("What moved the LightGBM prediction")
    shap = pd.Series(res["shap"]).sort_values(key=np.abs, ascending=False).head(8).sort_values()
    st.bar_chart(shap.rename("SHAP contribution (log-odds)"), horizontal=True)
    st.caption("Positive bars push the PD up; negative bars push it down.")
