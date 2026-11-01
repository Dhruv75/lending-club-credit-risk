"""Write a synthetic file in the Lending Club format (for tests and the demo only).

python scripts/make_synthetic_data.py --rows 80000 --out data/sample/synthetic_lending_club.csv.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrisk.synthetic import write_synthetic  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/sample/synthetic_lending_club.csv.gz"))
    args = ap.parse_args()
    print(f"Wrote {write_synthetic(args.out, args.rows, args.seed)}")
