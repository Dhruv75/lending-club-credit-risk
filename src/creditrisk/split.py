"""Out-of-time split: train on older loans, tune on the next period, test on a later, untouched period.

A random split would let the model see loans from the same months it is tested on, which hides
how it behaves when the economy and the applicant mix move on. Lenders always score future
applicants, so the test set has to be in the future too.
"""

from __future__ import annotations

import pandas as pd

from .config import Config

SPLITS = ("train", "valid", "test")


def split_bounds(cfg: Config) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    edges = [pd.Timestamp(x) for x in (cfg.start_date, cfg.train_end, cfg.valid_end, cfg.test_end)]
    if not all(a < b for a, b in zip(edges, edges[1:])):
        raise ValueError("Split dates must be strictly increasing: start < train_end < valid_end < test_end")
    return {name: (lo, hi) for name, lo, hi in zip(SPLITS, edges[:-1], edges[1:])}


def out_of_time_split(issue_d: pd.Series, cfg: Config) -> dict[str, pd.Index]:
    """Row index of each split; loans outside [start_date, test_end) are left out."""
    d = pd.to_datetime(issue_d)
    out: dict[str, pd.Index] = {}
    for name, (lo, hi) in split_bounds(cfg).items():
        idx = d.index[(d >= lo) & (d < hi)]
        if len(idx) == 0:
            raise ValueError(
                f"The {name} split is empty: no loans issued in [{lo:%Y-%m-%d}, {hi:%Y-%m-%d}). "
                "Adjust the split dates in config.py to match your data."
            )
        out[name] = idx
    return out
