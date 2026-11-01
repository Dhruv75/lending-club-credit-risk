"""Load the Lending Club file with DuckDB, define the target, and keep only loans whose outcome is known.

The Kaggle file (accepted_2007_to_2018Q4) has about 2.2M rows and 150 columns. Reading it with
pandas would need several GB of RAM, so DuckDB parses the CSV, casts types and filters the
population before anything reaches pandas. The filtered result is cached as Parquet.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

from .config import Config
from .leakage import (
    CATEGORICAL_RAW,
    DATE_RAW,
    LEAK_DEMO_COLUMNS,
    NUMERIC_RAW,
    OUTCOME_COLUMNS,
    TARGET,
)

log = logging.getLogger(__name__)

GOOD_STATUSES = ("Fully Paid", "Does not meet the credit policy. Status:Fully Paid")
BAD_STATUSES = ("Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off")
REQUIRED_COLUMNS = ("issue_d", "term", "loan_status", "loan_amnt")


def find_raw_file(cfg: Config) -> Path:
    """The configured file, or any accepted_2007_to_2018* file anywhere under data/raw/ (.csv.gz preferred)."""
    if cfg.raw_path.is_file():
        return cfg.raw_path
    candidates = [p for p in cfg.raw_path.parent.rglob("accepted_2007_to_2018*.csv*") if p.is_file()]
    if candidates:
        return sorted(candidates, key=lambda p: (p.suffix != ".gz", len(p.parts), p.name))[0]
    raise FileNotFoundError(
        f"Lending Club file not found at {cfg.raw_path}.\n"
        "Download the Kaggle dataset 'wordsforthewise/lending-club' and put "
        f"accepted_2007_to_2018Q4.csv.gz (or the unzipped .csv) in {cfg.raw_path.parent}/ - see README."
    )


def _lines_to_skip(path: Path) -> int:
    """Original LendingClub downloads start with a 'Notes offered by Prospectus' line before the header."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    return 0 if "loan_amnt" in first else 1


def _q(name: str) -> str:
    """Quote a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def _sql_list(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _month_index(column: str) -> str:
    return f"(year({column}) * 12 + month({column}))"


def read_population(path: Path, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Parse the raw file and return (loans with a known outcome, population funnel counts)."""
    path = Path(path)
    con = duckdb.connect()
    literal = path.as_posix().replace("'", "''")
    source = (
        f"read_csv('{literal}', header = true, all_varchar = true, skip = {_lines_to_skip(path)}, ignore_errors = true)"
    )
    available = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
    missing = [c for c in REQUIRED_COLUMNS if c not in available]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    numeric = list(dict.fromkeys(c for c in (*NUMERIC_RAW, *OUTCOME_COLUMNS, *LEAK_DEMO_COLUMNS, "int_rate")))
    strings = list(dict.fromkeys(c for c in (*CATEGORICAL_RAW, "grade", "sub_grade")))
    wanted = set(numeric) | set(strings) | set(DATE_RAW)
    absent = sorted(wanted - available)
    if absent:
        log.warning("Columns not in this file (filled with missing values): %s", ", ".join(absent))

    def num(c: str) -> str:
        if c not in available:
            return f"CAST(NULL AS DOUBLE) AS {_q(c)}"
        return f"TRY_CAST(NULLIF(trim(replace({_q(c)}, '%', '')), '') AS DOUBLE) AS {_q(c)}"

    def text(c: str) -> str:
        return f"NULLIF(trim({_q(c)}), '') AS {_q(c)}" if c in available else f"CAST(NULL AS VARCHAR) AS {_q(c)}"

    def date(c: str) -> str:
        if c not in available:
            return f"CAST(NULL AS DATE) AS {_q(c)}"
        return f"CAST(try_strptime(trim({_q(c)}), '%b-%Y') AS DATE) AS {_q(c)}"

    columns = [
        "NULLIF(trim(id), '') AS id" if "id" in available else "CAST(NULL AS VARCHAR) AS id",
        "CAST(try_strptime(trim(issue_d), '%b-%Y') AS DATE) AS issue_d",
        "TRY_CAST(regexp_extract(term, '(\\d+)', 1) AS INTEGER) AS term_months",
        "trim(loan_status) AS loan_status",
        *[num(c) for c in numeric],
        *[text(c) for c in strings],
        *[date(c) for c in DATE_RAW],
    ]
    con.execute(f"CREATE TEMP TABLE loans AS SELECT {', '.join(columns)} FROM {source}")

    snapshot = pd.Timestamp(cfg.snapshot_date)
    snapshot_index = snapshot.year * 12 + snapshot.month
    term = int(cfg.term_months)
    is_term = f"term_months = {term}"
    matured = f"issue_d IS NOT NULL AND {_month_index('issue_d')} + {term} <= {snapshot_index}"
    resolved = f"loan_status IN ({_sql_list(GOOD_STATUSES + BAD_STATUSES)})"

    funnel_row = con.execute(
        f"""
        SELECT
            count(*) FILTER (WHERE loan_amnt IS NOT NULL)                          AS all_loans,
            count(*) FILTER (WHERE {is_term})                                     AS term_ok,
            count(*) FILTER (WHERE {is_term} AND {matured})                       AS matured,
            count(*) FILTER (WHERE {is_term} AND {matured} AND {resolved})        AS resolved
        FROM loans
        """
    ).fetchone()
    funnel = dict(zip(("all_loans", "term_ok", "matured", "resolved"), map(int, funnel_row), strict=True))
    funnel["absent_columns"] = absent
    funnel["source_file"] = path.name

    df = con.execute(
        f"""
        SELECT * EXCLUDE (term_months)
        FROM loans
        WHERE {is_term} AND {matured} AND {resolved}
        ORDER BY issue_d, id
        """
    ).df()
    con.close()

    for c in ("issue_d", *DATE_RAW):
        df[c] = pd.to_datetime(df[c])
    df[TARGET] = df["loan_status"].isin(BAD_STATUSES).astype("int8")
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No loans left after filtering. Check that the file is the Lending Club 'accepted' file "
            f"and that snapshot_date ({cfg.snapshot_date}) and term_months ({term}) fit the data."
        )
    log.info(
        "Population: %s loans in file -> %s %s-month -> %s matured -> %s with a final outcome",
        f"{funnel['all_loans']:,}",
        f"{funnel['term_ok']:,}",
        term,
        f"{funnel['matured']:,}",
        f"{funnel['resolved']:,}",
    )
    return df, funnel


def load_loans(cfg: Config, path: Path | None = None, use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Read the population, re-using a Parquet cache keyed on the source file and population settings."""
    path = Path(path) if path else find_raw_file(cfg)
    stat = path.stat()
    key = f"{path.name.split('.')[0]}_{stat.st_size}_{cfg.term_months}m_{cfg.snapshot_date}"
    cache = cfg.processed_dir / f"{key}.parquet"
    meta = cache.with_suffix(".json")
    if use_cache and cache.exists() and meta.exists():
        log.info("Using cached population %s", cache.name)
        return pd.read_parquet(cache), json.loads(meta.read_text())

    log.info("Parsing %s with DuckDB (first run only; the result is cached)...", path.name)
    df, funnel = read_population(path, cfg)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    meta.write_text(json.dumps(funnel, indent=2))
    return df, funnel
