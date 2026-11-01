"""End-to-end pipeline.

    python -m creditrisk.pipeline --data data/raw/accepted_2007_to_2018Q4.csv.gz   # real data
    python -m creditrisk.pipeline --demo                                         # synthetic demo

Stages: load (DuckDB) -> target and population -> out-of-time split -> features -> leakage guard
-> scorecard + LightGBM -> calibration -> evaluation vs Lending Club's sub-grade -> profit-based
cutoff -> stability monitoring -> figures, tables, model card and a serving bundle.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import plots
from .calibration import Calibrator
from .config import ROOT, Config
from .data import load_loans
from .features import apply_categories, build_features, fit_categories, select_available_features
from .gbm import MONOTONE, GBMModel
from .leakage import LEAK_DEMO_COLUMNS, TARGET, blocked_column_count, check_features
from .metrics import auc, decile_table, lift_table, summarize
from .monitoring import feature_psi, psi, psi_label, vintage_table
from .policy import (
    choose_cutoff,
    estimate_loss_params,
    evaluate_policy,
    expected_loss,
    profit_curve,
    realised_credit_loss,
    realised_pnl,
)
from .report import write_report
from .scorecard import Scorecard
from .split import SPLITS, out_of_time_split, split_bounds
from .synthetic import write_synthetic

log = logging.getLogger("creditrisk")


class SubGradeBenchmark:
    """Lending Club's own grade turned into a PD: each sub-grade's default rate in training."""

    def fit(self, sub_grade: pd.Series, y, prior_strength: float = 20.0) -> SubGradeBenchmark:
        df = pd.DataFrame({"g": pd.Series(sub_grade).astype("object").to_numpy(), "y": np.asarray(y)})
        self.overall_ = float(df["y"].mean())
        stats = df.groupby("g")["y"].agg(["sum", "size"])
        smoothed = (stats["sum"] + prior_strength * self.overall_) / (stats["size"] + prior_strength)
        self.rates_ = smoothed.to_dict()
        return self

    def predict(self, sub_grade: pd.Series) -> np.ndarray:
        s = pd.Series(sub_grade).astype("object")
        return s.map(self.rates_).astype(float).fillna(self.overall_).to_numpy()


def leakage_demo(cfg: Config, df: pd.DataFrame, X: pd.DataFrame, idx: dict[str, pd.Index], y: pd.Series) -> dict:
    """A deliberately leaky model, built only to show what leakage looks like. Never saved or deployed."""
    cols = [c for c in LEAK_DEMO_COLUMNS if c in df.columns and df.loc[idx["train"], c].notna().mean() > 0.5]
    if not cols:
        return {}
    leaky_X = pd.concat([X, df[cols]], axis=1)
    model = GBMModel(cfg.with_overrides(n_estimators=600))
    model.fit(leaky_X.loc[idx["train"]], y.loc[idx["train"]], leaky_X.loc[idx["valid"]], y.loc[idx["valid"]])
    return {"columns": cols, "test_auc": auc(y.loc[idx["test"]], model.predict_raw(leaky_X.loc[idx["test"]]))}


def _config_dict(cfg: Config) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}


def run(cfg: Config, data_path: Path | None = None, synthetic: bool = False, use_cache: bool = True) -> dict:
    t0 = time.time()
    figures_dir, tables_dir = cfg.reports_dir / "figures", cfg.reports_dir / "tables"
    for d in (cfg.models_dir, figures_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Population ----------------------------------------------------------------------------
    df, funnel = load_loans(cfg, data_path, use_cache)
    idx = out_of_time_split(df["issue_d"], cfg)
    split_of = pd.Series(None, index=df.index, dtype="object")
    for name, ix in idx.items():
        split_of.loc[ix] = name
    in_window = split_of.notna()
    df, split_of = df.loc[in_window], split_of.loc[in_window]
    y = df[TARGET].astype(int)
    tr, va, te = (idx[s] for s in SPLITS)
    log.info("Splits: train %s | valid %s | test %s loans", f"{len(tr):,}", f"{len(va):,}", f"{len(te):,}")

    # 2. Features + leakage guard ------------------------------------------------------------------
    X_all = build_features(df)
    feats, dropped_availability = select_available_features(
        X_all.loc[tr], X_all.loc[va], cfg.max_missing_rate, cfg.max_missing_shift
    )
    check_features(feats)
    categories = fit_categories(X_all.loc[tr, feats])
    X = apply_categories(X_all[feats], categories)
    Xtr, Xva, Xte = X.loc[tr], X.loc[va], X.loc[te]
    ytr, yva, yte = y.loc[tr], y.loc[va], y.loc[te]
    log.info("%d candidate features kept, %d dropped for availability", len(feats), len(dropped_availability))

    # 3. Models -------------------------------------------------------------------------------------
    scorecard = Scorecard(cfg).fit(Xtr, ytr)
    check_features(scorecard.features_)
    log.info("Scorecard: %d features", len(scorecard.features_))
    gbm = GBMModel(cfg).fit(Xtr, ytr, Xva, yva)
    check_features(gbm.features_)
    log.info("LightGBM: early stopping at %d trees", gbm.best_iteration_)
    calibrator = Calibrator("platt").fit(gbm.predict_raw(Xva), yva)
    benchmark = None
    if df["sub_grade"].notna().mean() > 0.9:
        benchmark = SubGradeBenchmark().fit(df.loc[tr, "sub_grade"], ytr)

    preds: dict[str, dict[str, np.ndarray]] = {}
    for name, Xs in (("train", Xtr), ("valid", Xva), ("test", Xte)):
        raw = gbm.predict_raw(Xs)
        preds[name] = {
            "LightGBM": calibrator.transform(raw),
            "LightGBM (uncalibrated)": raw,
            "Scorecard": scorecard.predict_proba(Xs),
        }
        if benchmark is not None:
            preds[name]["LC sub-grade"] = benchmark.predict(df.loc[idx[name], "sub_grade"])

    # 4. Evaluation --------------------------------------------------------------------------------
    log.info("Evaluating (bootstrap CIs on the test period)...")
    metrics = {
        s: {
            m: summarize(y.loc[idx[s]], p, n_boot=cfg.n_bootstrap if s == "test" else 0, seed=cfg.random_state)
            for m, p in preds[s].items()
        }
        for s in SPLITS
    }
    leak = leakage_demo(cfg, df, X, idx, y)

    # 5. Money: loss parameters, cutoff, back-test ---------------------------------------------------
    loss_params = estimate_loss_params(df.loc[tr])
    pnl = realised_pnl(df)
    funded = df["funded_amnt"].fillna(df["loan_amnt"])
    cutoff = choose_cutoff(preds["valid"]["LightGBM"], pnl.loc[va], yva, funded.loc[va])
    policy = evaluate_policy(
        preds["test"]["LightGBM"], pnl.loc[te], yte, funded.loc[te], cutoff["pd_threshold"],
        preds["test"].get("LC sub-grade"),
    )  # fmt: skip
    el_test = float(expected_loss(preds["test"]["LightGBM"], df.loc[te, "loan_amnt"], loss_params).sum())
    realised_loss_test = float(realised_credit_loss(df.loc[te]).sum())
    curves = {"LightGBM": profit_curve(preds["test"]["LightGBM"], pnl.loc[te], yte, funded.loc[te])}
    if benchmark is not None:
        curves["LC sub-grade"] = profit_curve(preds["test"]["LC sub-grade"], pnl.loc[te], yte, funded.loc[te])

    # 6. Stability ---------------------------------------------------------------------------------------
    score_psi = {s: psi(preds["train"]["LightGBM"], preds[s]["LightGBM"]) for s in ("valid", "test")}
    fpsi = feature_psi(Xtr, Xte)
    p_all = pd.Series(np.nan, index=df.index)
    for s in SPLITS:
        p_all.loc[idx[s]] = preds[s]["LightGBM"]
    vintage = vintage_table(df["issue_d"], y, p_all, split_of)

    # 7. Explanations ------------------------------------------------------------------------------------
    sample = Xte.sample(min(len(Xte), 5000), random_state=cfg.random_state)
    shap_values, _ = gbm.shap_values(sample)
    shap_importance = shap_values.abs().mean().sort_values(ascending=False)
    lift = lift_table(yte, preds["test"]["LightGBM"])

    # 8. Figures and tables --------------------------------------------------------------------------------
    fig = {}
    fig["vintage"] = plots.vintage_chart(vintage, figures_dir / "vintage.png", synthetic)
    test_models = {k: preds["test"][k] for k in ("LightGBM", "Scorecard", "LC sub-grade") if k in preds["test"]}
    fig["roc"] = plots.roc_chart(
        yte, test_models, {k: metrics["test"][k]["gini"] for k in test_models}, figures_dir / "roc.png"
    )
    fig["ks"] = plots.ks_chart(yte, preds["test"]["LightGBM"], figures_dir / "ks.png")
    fig["calibration"] = plots.calibration_chart(
        {
            "LightGBM (calibrated)": decile_table(yte, preds["test"]["LightGBM"]),
            "LightGBM (uncalibrated)": decile_table(yte, preds["test"]["LightGBM (uncalibrated)"]),
            "Scorecard": decile_table(yte, preds["test"]["Scorecard"]),
        },
        figures_dir / "calibration.png",
    )
    fig["profit"] = plots.profit_chart(curves, policy["model"]["approval_rate"], figures_dir / "profit.png")
    fig["psi"] = plots.hbar_chart(
        fpsi.set_index("feature")["psi"], "Feature stability (PSI), train vs test", "PSI",
        figures_dir / "psi.png", thresholds=[(0.10, "0.10 investigate"), (0.25, "0.25 act")],
    )  # fmt: skip
    fig["shap"] = plots.hbar_chart(
        shap_importance, "What drives LightGBM: mean |SHAP|, test sample", "mean |SHAP| (log-odds)",
        figures_dir / "shap.png",
    )  # fmt: skip
    top4 = [f for f in scorecard.iv_.index if f in scorecard.features_][:4]
    fig["woe"] = plots.woe_panels(
        {f: scorecard.binner_.table(f) for f in top4}, scorecard.iv_.to_dict(), float(ytr.mean()),
        figures_dir / "woe_bins.png",
    )  # fmt: skip
    fig["score"] = plots.score_distribution(scorecard.score(Xte), yte, figures_dir / "score_distribution.png")
    if leak:
        fig["leakage"] = plots.leakage_chart(
            metrics["test"]["LightGBM"]["auc"], leak["test_auc"], figures_dir / "leakage.png"
        )

    iv_table = pd.DataFrame({"iv": scorecard.iv_})
    iv_table["status"] = ["used" if f in scorecard.features_ else scorecard.dropped_.get(f, "") for f in iv_table.index]
    iv_table.rename_axis("feature").to_csv(tables_dir / "information_value.csv")
    scorecard.points_table_.to_csv(tables_dir / "scorecard_points.csv", index=False)
    scorecard.binner_.all_tables().to_csv(tables_dir / "woe_bins_all.csv", index=False)
    lift.to_csv(tables_dir / "lift_test.csv", index=False)
    decile_table(yte, preds["test"]["LightGBM"]).to_csv(tables_dir / "calibration_test.csv", index=False)
    fpsi.to_csv(tables_dir / "feature_psi_test.csv", index=False)
    vintage.to_csv(tables_dir / "vintage.csv", index=False)
    pd.concat([c.assign(model=k) for k, c in curves.items()]).to_csv(tables_dir / "profit_curve_test.csv", index=False)

    # 9. Serving bundle ----------------------------------------------------------------------------------
    bounds = split_bounds(cfg)
    bundle = {
        "version": 1,
        "scorecard": scorecard,
        "gbm": gbm,
        "calibrator": calibrator,
        "features": feats,
        "categories": categories,
        "cutoff_pd": cutoff["pd_threshold"],
        "loss_params": loss_params,
        "base_rate": float(ytr.mean()),
        "synthetic": synthetic,
        "trained_on": f"{bounds['train'][0]:%b %Y} - {bounds['train'][1] - pd.Timedelta(days=1):%b %Y}",
    }
    joblib.dump(bundle, cfg.models_dir / "bundle.joblib")

    # 10. Report -------------------------------------------------------------------------------------------
    results = {
        "synthetic": synthetic,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": _config_dict(cfg),
        "funnel": funnel,
        "splits": {
            s: {
                "start": f"{bounds[s][0]:%Y-%m-%d}",
                "end": f"{bounds[s][1]:%Y-%m-%d}",
                "n": int(len(idx[s])),
                "default_rate": float(y.loc[idx[s]].mean()),
            }
            for s in SPLITS
        },
        "features": {
            "candidates": int(X_all.shape[1]),
            "after_availability": feats,
            "dropped_availability": dropped_availability,
            "scorecard": scorecard.features_,
            "scorecard_dropped": scorecard.dropped_,
            "lightgbm": gbm.features_,
            "iv": {k: float(v) for k, v in scorecard.iv_.items()},
            "blocked_columns": blocked_column_count(),
        },
        "lightgbm": {
            "best_iteration": gbm.best_iteration_,
            "monotone": {k: v for k, v in MONOTONE.items() if k in gbm.features_},
        },
        "calibration": {"method": "platt", "slope": calibrator.slope_, "intercept": calibrator.intercept_},
        "metrics": metrics,
        "leakage_demo": leak,
        "policy": {
            "loss_params": loss_params,
            "cutoff": cutoff,
            "test": policy,
            "expected_loss_test": el_test,
            "realised_loss_test": realised_loss_test,
        },
        "stability": {
            "score_psi": score_psi,
            "score_psi_label": {k: psi_label(v) for k, v in score_psi.items()},
            "feature_psi": fpsi.to_dict("records"),
        },
        "shap_importance": {k: float(v) for k, v in shap_importance.items()},
        "lift_test": lift.to_dict("records"),
        "calibration_test": decile_table(yte, preds["test"]["LightGBM"]).to_dict("records"),
        "scorecard_scaling": {"base_score": cfg.base_score, "base_odds": cfg.base_odds, "pdo": cfg.pdo},
          "figures": {k: Path(v).relative_to(cfg.reports_dir).as_posix() for k, v in fig.items()},
    }
    results["runtime_seconds"] = round(time.time() - t0, 1)
    write_report(results, cfg.reports_dir)
    log.info("Done in %.0fs. Model card: %s", results["runtime_seconds"], cfg.reports_dir / "model_card.md")
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the Lending Club PD models.")
    ap.add_argument("--data", type=Path, help="accepted_2007_to_2018Q4.csv(.gz); default: look in data/raw/")
    ap.add_argument("--demo", action="store_true", help="generate synthetic data and run on it (writes to runs/demo/)")
    ap.add_argument("--rows", type=int, default=80_000, help="rows of synthetic data for --demo")
    ap.add_argument("--synthetic", action="store_true", help="label outputs as synthetic")
    ap.add_argument("--out", type=Path, help="write models/ and reports/ under this folder")
    ap.add_argument("--no-cache", action="store_true", help="re-parse the raw file even if a cache exists")
    ap.add_argument("--bootstrap", type=int, help="bootstrap resamples for the Gini confidence interval")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    cfg = Config()
    data, out, synthetic = args.data, args.out, args.synthetic
    if args.demo:
        data = ROOT / "data" / "sample" / "synthetic_lending_club.csv.gz"
        if not data.exists() or args.no_cache:
            log.info("Writing %s synthetic loans to %s", f"{args.rows:,}", data)
            write_synthetic(data, args.rows)
        out = out or ROOT / "runs" / "demo"
        synthetic = True
    synthetic = synthetic or (data is not None and "synthetic" in Path(data).name)
    if out is not None:
        cfg = cfg.with_overrides(models_dir=Path(out) / "models", reports_dir=Path(out) / "reports")
    cfg = cfg.with_overrides(n_bootstrap=args.bootstrap)
    res = run(cfg, data, synthetic=synthetic, use_cache=not args.no_cache)

    t = res["metrics"]["test"]
    print("\nOut-of-time test period")
    for name in ("Scorecard", "LightGBM", "LC sub-grade"):
        if name in t:
            print(f"  {name:<14} Gini {t[name]['gini']:.3f}   KS {t[name]['ks']:.3f}   Brier {t[name]['brier']:.4f}")
    print(f"\nModel card: {cfg.reports_dir / 'model_card.md'}")
    if synthetic:
        print("NOTE: trained on SYNTHETIC data - these numbers are for testing only, do not report them.")


if __name__ == "__main__":
    main()
