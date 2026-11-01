"""Shared fixtures: a small synthetic Lending Club file and one pipeline run on it (session-scoped)."""

from __future__ import annotations

import pytest

from creditrisk.config import Config
from creditrisk.data import load_loans
from creditrisk.pipeline import run
from creditrisk.synthetic import write_synthetic


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("creditrisk")


@pytest.fixture(scope="session")
def cfg(workdir) -> Config:
    return Config(
        processed_dir=workdir / "processed",
        models_dir=workdir / "models",
        reports_dir=workdir / "reports",
        n_bootstrap=20,
        n_estimators=400,
    )


@pytest.fixture(scope="session")
def sample_file(workdir):
    return write_synthetic(workdir / "synthetic_lending_club.csv.gz", n=14_000, seed=3)


@pytest.fixture(scope="session")
def sample_loans(cfg, sample_file):
    df, _ = load_loans(cfg, sample_file)
    return df


@pytest.fixture(scope="session")
def pipeline_run(cfg, sample_file):
    return run(cfg, sample_file, synthetic=True)
