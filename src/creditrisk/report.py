"""Write metrics.json and a human-readable model card (with resume bullets filled from this run)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .leakage import FEATURE_DESCRIPTIONS


def _clean(o):
    """Make results JSON-safe (numpy types, NaN -> null)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating | float):
        v = float(o)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _pct(v, digits=1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.{digits}f}%"


def _money(v: float) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e6:
        return f"{sign}${v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{sign}${v / 1e3:.1f}K"
    return f"{sign}${v:.0f}"


def resume_bullets(r: dict) -> list[str]:
    m = r["metrics"]["test"]
    sp = r["splits"]
    n_total = sum(s["n"] for s in sp.values())
    test_year = sp["test"]["start"][:4]
    sc, gb = m["Scorecard"]["gini"], m["LightGBM"]["gini"]
    bullets = []
    first = (
        f"Built an out-of-time-validated probability-of-default model on {n_total:,} Lending Club loans "
        f"(trained on {sp['train']['start'][:4]}-{int(sp['train']['end'][:4]) - (sp['train']['end'][5:] == '01-01')} "
        f"vintages, tested on unseen {test_year} loans): a WoE/IV scorecard and a monotone-constrained LightGBM "
        f"reached test Gini {sc:.2f} / {gb:.2f}"
    )
    if "LC sub-grade" in m:
        sg = m["LC sub-grade"]["gini"]
        if gb > sg + 0.005:
            first += f", beating Lending Club's own sub-grade ({sg:.2f})."
        elif gb >= sg - 0.005:
            first += f", matching Lending Club's own sub-grade ({sg:.2f})."
        else:
            first += f" vs {sg:.2f} for Lending Club's own sub-grade, which also uses data the public file lacks."
    else:
        first += "."
    bullets.append(first)

    leak = r.get("leakage_demo") or {}
    second = (
        f"Wrote a lineage-aware leakage guard that keeps {r['features']['blocked_columns']} post-origination, "
        "lender-decision and identifier columns out of the model"
    )
    if leak:
        second += (
            f" (one leaked post-issue FICO field inflated test AUC from {m['LightGBM']['auc']:.2f} "
            f"to {leak['test_auc']:.2f})"
        )
    second += f"; calibrated PDs to a test Brier score of {m['LightGBM']['brier']:.3f}."
    bullets.append(second)

    pol = r["policy"]["test"]
    roa_all, roa_model = pol["approve_all"]["return_on_funded"], pol["model"]["return_on_funded"]
    if roa_model is not None and roa_model - roa_all >= 0.001 and pol["model"]["approval_rate"] < 0.995:
        third = (
            f"Back-tested a profit-maximising approval cutoff chosen on a separate validation period: on unseen "
            f"{test_year} loans it approved {_pct(pol['model']['approval_rate'], 0)} of borrowers, avoided "
            f"{_pct(pol['model']['defaults_avoided_share'], 0)} of defaults and lifted realised return on "
            f"funded capital from {_pct(roa_all)} to {_pct(roa_model)}."
        )
        bullets.append(third)
    return bullets


def model_card(r: dict) -> str:
    L: list[str] = []
    add = L.append
    m, sp, f, pol = r["metrics"], r["splits"], r["features"], r["policy"]
    add("# Model card: Lending Club probability-of-default model\n")
    if r["synthetic"]:
        add(
            "> **SYNTHETIC DEMO RUN.** These numbers come from generated data that only mimics the Lending "
            "Club file format. They test the code, nothing else. Do not report them. Run the pipeline on the "
            "real Kaggle file to get real results.\n"
        )
    add(f"Generated {r['generated_at']} from `{r['funnel']['source_file']}` in {r['runtime_seconds']:.0f}s.\n")

    add("## 1. Population\n")
    fn = r["funnel"]
    in_window = sum(s["n"] for s in sp.values())
    add("| Step | Loans |\n|---|---:|")
    add(f"| All loans in the file | {fn['all_loans']:,} |")
    add(f"| {r['config']['term_months']}-month term | {fn['term_ok']:,} |")
    add(f"| Term ended before the status snapshot ({r['config']['snapshot_date']}) | {fn['matured']:,} |")
    add(f"| Final outcome known (fully paid or charged off) | {fn['resolved']:,} |")
    add(f"| Issued inside the study window | {in_window:,} |\n")

    add("## 2. Out-of-time split\n")
    add("| Split | Issued from | Issued before | Loans | Default rate |\n|---|---|---|---:|---:|")
    for name, s in sp.items():
        add(f"| {name} | {s['start']} | {s['end']} | {s['n']:,} | {_pct(s['default_rate'])} |")
    add("")

    add("## 3. Features\n")
    add(
        f"The leakage guard keeps {f['blocked_columns']} raw columns (post-origination, lender decision, "
        "identifiers, free text, geography) out of the model. Every input must be on a reviewed allowlist and "
        "the raw columns it is built from are checked too.\n"
    )
    if f["dropped_availability"]:
        add("Dropped because the field is not reliably available over time:\n")
        add("| Feature | Reason |\n|---|---|")
        for k, v in f["dropped_availability"].items():
            add(f"| `{k}` | {v} |")
        add("")
    add(f"**Scorecard** ({len(f['scorecard'])} features, ranked by Information Value):\n")
    add("| Feature | IV | Meaning |\n|---|---:|---|")
    for k in f["scorecard"]:
        add(f"| `{k}` | {f['iv'][k]:.3f} | {FEATURE_DESCRIPTIONS.get(k, '')} |")
    add("")
    if f["scorecard_dropped"]:
        add("Left out of the scorecard:\n")
        add("| Feature | Reason |\n|---|---|")
        for k, v in f["scorecard_dropped"].items():
            add(f"| `{k}` | {v} |")
        add("")
    mono = ", ".join(f"`{k}` ({'+' if v > 0 else '-'})" for k, v in r["lightgbm"]["monotone"].items())
    add(
        f"**LightGBM** uses all {len(f['lightgbm'])} features; early stopping on the validation period chose "
        f"{r['lightgbm']['best_iteration']} trees. Monotone constraints: {mono}.\n"
    )

    add("## 4. Performance\n")
    add("| Model | Train Gini | Valid Gini | Test Gini (95% CI) | Test KS | Test Brier | Test ECE |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for name in ("Scorecard", "LightGBM", "LightGBM (uncalibrated)", "LC sub-grade"):
        if name not in m["test"]:
            continue
        t = m["test"][name]
        ci = [float("nan") if c is None else c for c in (t.get("gini_ci") or [None, None])]
        label = "LC sub-grade (benchmark)" if name == "LC sub-grade" else name
        add(
            f"| {label} | {m['train'][name]['gini']:.3f} | {m['valid'][name]['gini']:.3f} | "
            f"{t['gini']:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) | {t['ks']:.3f} | {t['brier']:.4f} | {t['ece']:.4f} |"
        )
    add(
        "\nGini = 2 x AUC - 1. ECE = average gap between predicted and observed default rate across PD deciles. "
        "LightGBM rows use calibrated PDs unless marked otherwise.\n"
    )
    leak = r.get("leakage_demo") or {}
    if leak:
        add(
            f"**Leakage check.** Adding {', '.join(f'`{c}`' for c in leak['columns'])} (a FICO score pulled after "
            f"the loan was issued) lifts test AUC from {m['test']['LightGBM']['auc']:.3f} to "
            f"{leak['test_auc']:.3f}. That jump is what leakage looks like; the guard blocks these columns.\n"
        )

    add("## 5. Calibration and lift (test period, LightGBM)\n")
    cal = r["calibration"]
    tt = m["test"]["LightGBM"]
    add(
        f"Platt recalibration fitted on the validation period: slope {cal['slope']:.3f}, intercept "
        f"{cal['intercept']:.3f}. Mean predicted PD {_pct(tt['mean_pd'], 2)} vs observed default rate "
        f"{_pct(tt['default_rate'], 2)}.\n"
    )
    lift = r["lift_test"]
    add(
        f"The riskiest 10% of loans hold {_pct(lift[0]['cum_share_of_defaults'])} of all defaults "
        f"(lift {lift[0]['lift']:.2f}x); the riskiest 20% hold {_pct(lift[1]['cum_share_of_defaults'])}.\n"
    )

    add("## 6. Approval policy (back-test on the test period)\n")
    lp = pol["loss_params"]
    add(
        f"From {lp['n_defaults']:,} defaulted training loans: exposure at default = {_pct(lp['ead_ratio'])} of the "
        f"loan amount, loss given default = {_pct(lp['lgd'])}. Cutoff chosen on validation: approve if "
        f"PD <= {_pct(pol['cutoff']['pd_threshold'], 2)} (approval rate {_pct(pol['cutoff']['valid_approval_rate'], 0)} "
        "in validation).\n"
    )
    pt = pol["test"]
    add("| Policy on test loans | Approval rate | Default rate | Realised profit | Return on funded |")
    add("|---|---:|---:|---:|---:|")
    rows = [("Approve everyone", pt["approve_all"]), ("Model cutoff (PD threshold)", pt["model"])]
    if "subgrade_same_rate" in pt:
        rows.append(("LC sub-grade, same approval rate", pt["subgrade_same_rate"]))
    for label, s in rows:
        add(
            f"| {label} | {_pct(s['approval_rate'], 1)} | {_pct(s['default_rate'])} | {_money(s['total_pnl'])} | "
            f"{_pct(s['return_on_funded'], 2)} |"
        )
    add(
        f"\nThe cutoff turns away {_pct(pt['model']['defaults_avoided_share'])} of the loans that defaulted and "
        f"{_pct(pt['model']['goods_rejected_share'])} of the loans that were repaid. Expected loss on the test "
        f"book (PD x LGD x EAD) was {_money(pol['expected_loss_test'])} against {_money(pol['realised_loss_test'])} "
        "realised.\n"
    )

    add("## 7. Stability\n")
    st = r["stability"]
    add(
        f"Score PSI, train -> valid: {st['score_psi']['valid']:.3f} ({st['score_psi_label']['valid']}); "
        f"train -> test: {st['score_psi']['test']:.3f} ({st['score_psi_label']['test']}).\n"
    )
    add("| Feature | PSI train -> test | Status |\n|---|---:|---|")
    for row in st["feature_psi"][:6]:
        add(f"| `{row['feature']}` | {row['psi']:.3f} | {row['status']} |")
    add("")

    add("## 8. Figures\n")
    for key, path in r["figures"].items():
        add(f"![{key}]({path})\n")

    add("## 9. Resume bullets (numbers from this run)\n")
    if r["synthetic"]:
        add("_Synthetic run: these show the format only. Use the bullets from your real-data run._\n")
    for b in resume_bullets(r):
        add(f"- {b}")
    add("")

    add("## 10. Limitations\n")
    for item in (
        "Outcomes exist only for loans Lending Club approved, so the model never learns from rejected "
        "applicants (selection bias; reject inference is not attempted).",
        f"{r['config']['term_months']}-month loans only, and only vintages whose term ended before the status "
        "snapshot, so the newest data (2016+) is not used.",
        "Realised profit ignores the time value of money, servicing fees and funding costs.",
        "The sub-grade benchmark comes from Lending Club's own model, which saw information the public file "
        "does not include.",
        "No protected attributes are in the data, and geography is excluded as a likely proxy; a full "
        "fair-lending review would still be needed before any real use.",
    ):
        add(f"- {item}")
    return "\n".join(L) + "\n"


def write_report(results: dict, reports_dir: Path) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    clean = _clean(results)
    (reports_dir / "metrics.json").write_text(json.dumps(clean, indent=2))
    (reports_dir / "model_card.md").write_text(model_card(clean))
