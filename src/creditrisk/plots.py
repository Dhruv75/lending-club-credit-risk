"""Report figures (static PNGs for the model card and README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_COLORS = {"LightGBM": BLUE, "Scorecard": ORANGE, "LC sub-grade": AQUA}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
        }
    )


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _dot(ax, x, y, color, size=7):
    ax.plot(x, y, "o", ms=size, color=color, mec=SURFACE, mew=2, zorder=5)


def vintage_chart(vintage: pd.DataFrame, path: Path, synthetic: bool, min_count: int = 100) -> Path:
    _style()
    q = vintage.groupby("quarter", sort=True).agg(
        count=("count", "sum"), default_rate=("default_rate", "mean"), mean_pd=("mean_pd", "mean"),
        split=("split", "first"),
    )  # fmt: skip
    x = np.arange(len(q))
    splits = q["split"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    for ax in (ax1, ax2):
        for name, shade in (("valid", BAND), ("test", "#e8eef8")):
            idx = np.flatnonzero(splits == name)
            if len(idx):
                ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=shade, zorder=0, lw=0)
    ax1.bar(x, q["count"], width=0.7, color=BLUE, zorder=2)
    ax1.set_title("Loans issued per quarter")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v >= 1000 else f"{v:.0f}"))
    top = q["count"].max() * 1.22
    ax1.set_ylim(0, top)
    for name, label in (("train", "Train"), ("valid", "Valid"), ("test", "Test")):
        idx = np.flatnonzero(splits == name)
        if len(idx):
            ax1.text((idx[0] + idx[-1]) / 2, top * 0.97, label, ha="center", va="top", color=INK, fontsize=8.5)
    shown = (q["count"] >= min_count).to_numpy()
    dr = np.where(shown, q["default_rate"] * 100, np.nan)
    pdv = np.where(shown, q["mean_pd"] * 100, np.nan)
    ax2.plot(x, dr, color=BLUE, label="Observed default rate", zorder=3)
    ax2.plot(x, pdv, color=ORANGE, label="Mean predicted PD (LightGBM)", zorder=3)
    ax2.set_title("Observed default rate vs predicted PD, by issue quarter")
    ax2.set_ylabel("%")
    years = [str(v)[:4] for v in q.index]
    first_q = [i for i in range(len(years)) if i == 0 or years[i] != years[i - 1]]
    ax2.set_xticks(first_q)
    ax2.set_xticklabels([years[i] for i in first_q], fontsize=8.5)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    note = f"Quarters with fewer than {min_count} loans are not plotted."
    if synthetic:
        note = "Synthetic demo data. " + note
    fig.text(0.01, -0.01, note, ha="left", va="bottom", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, path)


def roc_chart(y, preds: dict[str, np.ndarray], ginis: dict[str, float], path: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, label="Random (Gini 0)")
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, MUTED), label=f"{name} (Gini {ginis[name]:.3f})")
    ax.set_xlabel("False positive rate (good borrowers flagged)")
    ax.set_ylabel("True positive rate (defaults caught)")
    ax.set_title("ROC curves, out-of-time test period")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def ks_chart(y, p, path: Path) -> Path:
    _style()
    y = np.asarray(y)
    p = np.asarray(p)
    grid = np.quantile(p, np.linspace(0, 1, 201))
    cdf_bad = np.searchsorted(np.sort(p[y == 1]), grid, side="right") / (y == 1).sum()
    cdf_good = np.searchsorted(np.sort(p[y == 0]), grid, side="right") / (y == 0).sum()
    gap = cdf_good - cdf_bad
    i = int(np.argmax(gap))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(grid * 100, cdf_good * 100, color=BLUE, label="Non-defaulters")
    ax.plot(grid * 100, cdf_bad * 100, color=ORANGE, label="Defaulters")
    ax.plot([grid[i] * 100] * 2, [cdf_bad[i] * 100, cdf_good[i] * 100], color=INK_2, lw=1.2)
    ax.annotate(
        f"KS = {gap[i]:.3f}", (grid[i] * 100, cdf_bad[i] * 100), xytext=(grid[i] * 100 + 2, cdf_bad[i] * 100 - 16),
        ha="left", va="center", color=INK, fontsize=9.5,
        arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8, shrinkA=2, shrinkB=2),
    )  # fmt: skip
    ax.set_xlabel("Predicted PD (%)")
    ax.set_ylabel("Cumulative share of group (%)")
    ax.set_title("KS: biggest gap between the two groups")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def calibration_chart(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    _style()
    colors = {"LightGBM (calibrated)": BLUE, "LightGBM (uncalibrated)": ORANGE, "Scorecard": AQUA}
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    hi = max(max(t["mean_pd"].max(), t["default_rate"].max()) for t in tables.values()) * 100 * 1.08
    ax.plot([0, hi], [0, hi], color=AXIS, lw=1, label="Perfect calibration")
    for name, t in tables.items():
        c = colors.get(name, MUTED)
        ax.plot(t["mean_pd"] * 100, t["default_rate"] * 100, color=c, label=name, lw=1.6)
        for xv, yv in zip(t["mean_pd"] * 100, t["default_rate"] * 100):
            _dot(ax, xv, yv, c, size=6)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Mean predicted PD in decile (%)")
    ax.set_ylabel("Observed default rate in decile (%)")
    ax.set_title("Calibration by PD decile, test period")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def profit_chart(curves: dict[str, pd.DataFrame], chosen_rate: float, path: Path) -> Path:
    _style()
    colors = {"LightGBM": BLUE, "LC sub-grade": AQUA}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, c in curves.items():
        ax.plot(c["approval_rate"] * 100, c["total_pnl"] / 1e6, color=colors.get(name, MUTED), label=name)
    model = curves["LightGBM"]
    j = int(np.argmin(np.abs(model["approval_rate"] - chosen_rate)))
    xv, yv = model["approval_rate"].iloc[j] * 100, model["total_pnl"].iloc[j] / 1e6
    _dot(ax, xv, yv, BLUE, size=8)
    ax.annotate(
        f"Cutoff chosen on validation:\napprove the safest {xv:.0f}%", (xv, yv), xytext=(0.97, 0.34),
        textcoords="axes fraction", ha="right", va="top", color=INK, fontsize=8.5,
        arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8, shrinkA=2, shrinkB=5),
    )  # fmt: skip
    end = model.iloc[-1]
    _dot(ax, end["approval_rate"] * 100, end["total_pnl"] / 1e6, MUTED, size=7)
    ax.annotate("Approve everyone", (100, end["total_pnl"] / 1e6), xytext=(0, -22), textcoords="offset points",
                ha="right", va="top", color=INK_2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=2, shrinkB=5))  # fmt: skip
    left = min(40.0, xv - 10)
    ax.set_xlim(left, 101)
    visible = [c[c["approval_rate"] * 100 >= left]["total_pnl"] / 1e6 for c in curves.values()]
    lo, hi = min(v.min() for v in visible), max(v.max() for v in visible)
    ax.set_ylim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.12)
    ax.set_xlabel("Approval rate (%) - safest loans approved first")
    ax.set_ylabel("Realised profit ($M)")
    ax.set_title("Realised profit vs approval rate, test period")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, path)


def hbar_chart(values: pd.Series, title: str, xlabel: str, path: Path, thresholds=None, fmt="{:.3f}") -> Path:
    _style()
    values = values.sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(values) + 1.2))
    y = np.arange(len(values))
    ax.barh(y, values.to_numpy(), height=0.62, color=BLUE, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(values.index, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    span = values.max() if len(values) else 1
    for yi, v in zip(y, values.to_numpy()):
        ax.text(v + span * 0.01, yi, fmt.format(v), va="center", color=INK_2, fontsize=8)
    for t, label in thresholds or []:
        ax.axvline(t, color=INK_2, lw=1, zorder=3)
        ax.text(t, len(values) - 0.3, f" {label}", color=INK_2, fontsize=8, va="bottom")
    ax.set_xlim(0, max(span * 1.15, max((t for t, _ in thresholds or []), default=0) * 1.15))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return _save(fig, path)


def woe_panels(tables: dict[str, pd.DataFrame], ivs: dict[str, float], overall_rate: float, path: Path) -> Path:
    _style()
    names = list(tables)[:4]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    for ax, name in zip(axes.ravel(), names):
        t = tables[name]
        x = np.arange(len(t))
        ax.bar(x, t["bad_rate"] * 100, width=0.62, color=BLUE, zorder=2)
        ax.axhline(overall_rate * 100, color=INK_2, lw=1, zorder=3)
        ax.text(len(t) - 0.5, overall_rate * 100, "average", color=INK_2, fontsize=7.5, ha="right", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in t["bin"]], rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{name}  (IV {ivs[name]:.3f})", fontsize=10)
        ax.set_ylabel("Default rate (%)")
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    fig.suptitle("Default rate by WoE bin, top features by Information Value", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)  # fmt: skip
    fig.tight_layout()
    return _save(fig, path)


def score_distribution(score, y, path: Path) -> Path:
    _style()
    score = np.asarray(score)
    y = np.asarray(y)
    bins = np.linspace(np.quantile(score, 0.005), np.quantile(score, 0.995), 45)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(score[y == 0], bins=bins, density=True, histtype="step", lw=2, color=BLUE, label="Non-defaulters")
    ax.hist(score[y == 1], bins=bins, density=True, histtype="step", lw=2, color=ORANGE, label="Defaulters")
    ax.set_xlabel("Scorecard points (higher = safer)")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by outcome, test period")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def leakage_chart(honest_auc: float, leaky_auc: float, path: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    labels = ["With post-issue FICO (leaky)", "Application data only (honest)"]
    vals = [leaky_auc, honest_auc]
    ax.barh([0, 1], vals, height=0.5, color=[ORANGE, BLUE], zorder=2)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"AUC {v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_title("Test AUC: one leaked column vs none")
    fig.tight_layout()
    return _save(fig, path)
