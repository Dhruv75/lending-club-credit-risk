"""Leakage guard: which columns may become model inputs, and a check that fails loudly if a leak slips in.

The rule is "default deny". A feature is allowed only if it is on the reviewed allowlist below AND
none of the raw columns it is built from were unknown at application time. Post-origination
columns are still loaded, but only to compute outcomes (default flag, realised profit, loss given
default). They never enter the feature matrix.
"""

from __future__ import annotations

from collections.abc import Iterable

TARGET = "default"

# Known only after the loan was issued: repayments, recoveries, refreshed credit data,
# hardship and settlement programmes, and the status itself.
POST_ORIGINATION = frozenset(
    {
        "loan_status",
        "pymnt_plan",
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "deferral_term",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
    }
)

# Outputs of Lending Club's own underwriting and pricing decision. We are building the
# underwriting model, so feeding it the lender's verdict would be circular. sub_grade is
# kept aside as an external benchmark instead.
LENDER_DECISION = frozenset(
    {
        "grade",
        "sub_grade",
        "int_rate",
        "installment",
        "funded_amnt",
        "funded_amnt_inv",
        "initial_list_status",
        "disbursement_method",
    }
)

# Identifiers, free text, the split column, and geography (a proxy for protected attributes).
NOT_FEATURES = frozenset(
    {
        "id",
        "member_id",
        "url",
        "desc",
        "emp_title",
        "title",
        "zip_code",
        "addr_state",
        "policy_code",
        "issue_d",
        "term",
        TARGET,
    }
)

FORBIDDEN_SOURCES = POST_ORIGINATION | LENDER_DECISION | {TARGET}

# --- raw application-time columns we read from the file --------------------------------------
NUMERIC_RAW = (
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "pub_rec_bankruptcies",
    "mort_acc",
    "acc_open_past_24mths",
    "bc_util",
    "tot_cur_bal",
)
CATEGORICAL_RAW = ("emp_length", "home_ownership", "verification_status", "purpose")
DATE_RAW = ("earliest_cr_line",)

# Loaded only to measure outcomes (never features).
OUTCOME_COLUMNS = ("funded_amnt", "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee")
# Lending Club's own risk grade, used as the benchmark to beat.
BENCHMARK_COLUMNS = ("grade", "sub_grade", "int_rate")
# A leak that looks innocent: the same FICO score, but pulled after the loan was issued.
LEAK_DEMO_COLUMNS = ("last_fico_range_high", "last_fico_range_low")

# --- model features (the reviewed allowlist) ---------------------------------------------------
NUMERIC_FEATURES = (
    "loan_amnt",
    "annual_inc",
    "loan_to_income",
    "dti",
    "fico",
    "credit_history_months",
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
    "revol_bal_to_income",
    "emp_length_years",
    "mort_acc",
    "acc_open_past_24mths",
    "bc_util",
    "tot_cur_bal",
)
CATEGORICAL_FEATURES = ("home_ownership", "verification_status", "purpose")
ALLOWED_FEATURES = frozenset(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

# For these, "missing" means "never happened" (e.g. never delinquent), so a high missing
# rate is information rather than a data-quality problem.
INFORMATIVE_MISSING = frozenset({"mths_since_last_delinq", "mths_since_last_record"})

# Lineage of engineered features: which raw columns each one is computed from.
ENGINEERED_SOURCES: dict[str, tuple[str, ...]] = {
    "fico": ("fico_range_low", "fico_range_high"),
    "credit_history_months": ("earliest_cr_line", "issue_d"),
    "loan_to_income": ("loan_amnt", "annual_inc"),
    "revol_bal_to_income": ("revol_bal", "annual_inc"),
    "emp_length_years": ("emp_length",),
}

# Human-readable descriptions, used in the model card and the app.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "loan_amnt": "Amount requested",
    "annual_inc": "Self-reported annual income",
    "loan_to_income": "Loan amount / annual income",
    "dti": "Debt-to-income ratio (%), excluding this loan",
    "fico": "FICO score at application (mid-point of range)",
    "credit_history_months": "Months since the oldest credit line was opened",
    "inq_last_6mths": "Credit inquiries in the last 6 months",
    "delinq_2yrs": "30+ day delinquencies in the last 2 years",
    "mths_since_last_delinq": "Months since last delinquency (missing = never)",
    "mths_since_last_record": "Months since last public record (missing = never)",
    "pub_rec": "Derogatory public records",
    "pub_rec_bankruptcies": "Public-record bankruptcies",
    "open_acc": "Open credit lines",
    "total_acc": "Total credit lines ever",
    "revol_bal": "Revolving balance",
    "revol_util": "Revolving utilisation (%)",
    "revol_bal_to_income": "Revolving balance / annual income",
    "emp_length_years": "Years at current job (0 = under a year)",
    "mort_acc": "Mortgage accounts",
    "acc_open_past_24mths": "Accounts opened in the last 24 months",
    "bc_util": "Bankcard utilisation (%)",
    "tot_cur_bal": "Total current balance of all accounts",
    "home_ownership": "Home ownership (rent / own / mortgage)",
    "verification_status": "Whether income was verified",
    "purpose": "Stated purpose of the loan",
}


class LeakageError(ValueError):
    """Raised when a feature list contains a column that was not known at application time."""


def source_columns(feature: str) -> tuple[str, ...]:
    """Raw columns a feature is computed from (itself, for raw features)."""
    return ENGINEERED_SOURCES.get(feature, (feature,))


def find_leaks(features: Iterable[str]) -> list[str]:
    """Return one message per problem; an empty list means the feature set is clean."""
    problems: list[str] = []
    for feature in features:
        if feature in POST_ORIGINATION:
            problems.append(f"{feature}: only known after the loan was issued")
        elif feature in LENDER_DECISION:
            problems.append(f"{feature}: output of the lender's own decision (circular)")
        elif feature in NOT_FEATURES:
            problems.append(f"{feature}: identifier / free text / split column / geography")
        elif feature not in ALLOWED_FEATURES:
            problems.append(f"{feature}: not on the reviewed allowlist (add it to leakage.py after review)")
        for src in source_columns(feature):
            if src != feature and src in FORBIDDEN_SOURCES:
                problems.append(f"{feature}: built from '{src}', which is not known at application time")
    return problems


def check_features(features: Iterable[str]) -> None:
    """Fail loudly if any feature could leak information from after the application date."""
    problems = find_leaks(list(features))
    if problems:
        raise LeakageError("Leakage guard blocked these features:\n  - " + "\n  - ".join(problems))


def blocked_column_count() -> int:
    """How many raw columns the guard keeps out of the model."""
    return len(POST_ORIGINATION | LENDER_DECISION | NOT_FEATURES)
