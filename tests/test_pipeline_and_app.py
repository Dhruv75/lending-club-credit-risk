import json
from pathlib import Path

import joblib
import pytest

from creditrisk.serving import EXAMPLE_APPLICANT, load_bundle, score_applicant

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_writes_every_output(pipeline_run, cfg):
    r = cfg.reports_dir
    assert (r / "metrics.json").exists() and (r / "model_card.md").exists()
    assert (cfg.models_dir / "bundle.joblib").exists()
    for rel in pipeline_run["figures"].values():
        assert (r / rel).exists()
    metrics = json.loads((r / "metrics.json").read_text())
    assert metrics["synthetic"] is True
    assert "SYNTHETIC DEMO RUN" in (r / "model_card.md").read_text()


def test_out_of_time_performance_is_sane(pipeline_run):
    t = pipeline_run["metrics"]["test"]
    for name in ("Scorecard", "LightGBM"):
        assert 0.2 < t[name]["gini"] < 0.8
        assert t[name]["brier"] < 0.25
    assert t["LightGBM"]["gini_ci"][0] < t["LightGBM"]["gini"] < t["LightGBM"]["gini_ci"][1]


def test_leakage_demo_shows_the_too_good_to_be_true_jump(pipeline_run):
    leak = pipeline_run["leakage_demo"]
    assert leak["test_auc"] > pipeline_run["metrics"]["test"]["LightGBM"]["auc"] + 0.05


def test_serving_scores_an_applicant(pipeline_run, cfg):
    bundle = load_bundle(cfg.models_dir / "bundle.joblib")
    res = score_applicant(bundle, EXAMPLE_APPLICANT)
    assert 0 < res["pd"] < 1 and 0 < res["scorecard_pd"] < 1
    assert res["decision"] in {"Approve", "Decline"}
    assert isinstance(res["reasons"], list)


def test_monotone_constraint_holds_for_fico(pipeline_run, cfg):
    bundle = joblib.load(cfg.models_dir / "bundle.joblib")
    pds = [score_applicant(bundle, {**EXAMPLE_APPLICANT, "fico_range_low": f, "fico_range_high": f + 4})["pd"]
           for f in range(660, 821, 20)]  # fmt: skip
    assert all(a >= b - 1e-12 for a, b in zip(pds, pds[1:]))


def test_streamlit_app_renders(pipeline_run, cfg, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CREDITRISK_BUNDLE", str(cfg.models_dir / "bundle.joblib"))
    at = testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Probability of default" in labels and "Decision" in labels


def test_saved_config_has_no_os_specific_paths():
    """A model trained on Windows must load on Linux (Streamlit Cloud): no WindowsPath in the pickle."""
    import pickle
    from pathlib import Path, PureWindowsPath

    from creditrisk.config import Config

    cfg = Config(models_dir=PureWindowsPath(r"C:\Users\x\models"))
    blob = pickle.dumps(cfg)
    assert b"WindowsPath" not in blob and b"PosixPath" not in blob
    back = pickle.loads(blob)
    assert isinstance(back.models_dir, Path) and back.term_months == cfg.term_months
