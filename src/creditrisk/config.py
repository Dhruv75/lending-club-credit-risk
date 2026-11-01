"""Single source of truth for paths, the population definition, split dates and model settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path, PurePath

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # --- paths ---------------------------------------------------------------------
    raw_path: Path = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
    processed_dir: Path = ROOT / "data" / "processed"
    models_dir: Path = ROOT / "models"
    reports_dir: Path = ROOT / "reports"

    # --- population ------------------------------------------------------------------
    term_months: int = 36
    # Loan statuses in the Kaggle file were captured around the end of 2018. A loan whose
    # scheduled end date falls after this snapshot has not had time to finish, so its final
    # outcome is unknown (right-censoring). Only loans that reached maturity are kept.
    snapshot_date: str = "2018-12-31"

    # --- out-of-time split, by issue date --------------------------------------------------
    start_date: str = "2007-01-01"
    train_end: str = "2014-07-01"  # train:  start_date <= issue_d < train_end
    valid_end: str = "2015-01-01"  # valid:  train_end  <= issue_d < valid_end
    test_end: str = "2016-01-01"  # test:   valid_end  <= issue_d < test_end (touched once)

    # --- feature availability ----------------------------------------------------------------
    max_missing_rate: float = 0.30  # drop a feature missing in more than 30% of training rows...
    max_missing_shift: float = 0.10  # ...or whose missing rate moves more than 10 points train -> valid

    # --- scorecard ---------------------------------------------------------------------------
    n_bins: int = 10
    min_bin_share: float = 0.05
    min_iv: float = 0.02  # below this a feature has no useful signal
    suspicious_iv: float = 0.50  # this strong on application data usually means leakage
    max_woe_corr: float = 0.70
    base_score: int = 600  # score that corresponds to base_odds
    base_odds: float = 50.0  # good:bad odds at base_score (600 points = 50:1)
    pdo: int = 20  # points to double the odds

    # --- LightGBM ------------------------------------------------------------------------------
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 200
    n_estimators: int = 2000
    early_stopping_rounds: int = 100

    # --- evaluation ----------------------------------------------------------------------------
    n_bootstrap: int = 200
    random_state: int = 42

    def with_overrides(self, **kwargs) -> Config:
        """Return a copy with every non-None keyword applied."""
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    # Saved models keep a copy of the config. Store paths as plain text, so a model trained on
    # Windows (WindowsPath) can still be loaded on Linux, e.g. on Streamlit Cloud.
    def __getstate__(self) -> dict:
        return {k: (str(v) if isinstance(v, PurePath) else v) for k, v in self.__dict__.items()}

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            object.__setattr__(self, k, Path(v) if k in PATH_FIELDS else v)


PATH_FIELDS = ("raw_path", "processed_dir", "models_dir", "reports_dir")
