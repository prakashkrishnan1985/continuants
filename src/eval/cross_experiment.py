"""
Cross-experiment comparison plot generator.

Loads two (or more) completed runs and emits side-by-side figures
suitable for the paper's combined-analysis section.

Usage:

    python -m src.eval.cross_experiment \\
        --runs experiments/drift_study/runs/20260519-093632 \\
               experiments/drift_study/runs/20260519-125139 \\
        --labels "E1: templated" "E2: bitext" \\
        --out experiments/drift_study/cross_experiment_figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.efficiency import tool_calls_per_ticket
from src.eval.loader import load_run
from src.eval.plots import cross_experiment_efficiency_plot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run directories to compare.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Labels for each run (default: directory names).")
    p.add_argument("--out", default="experiments/drift_study/cross_experiment_figures",
                   help="Output directory for figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(p) for p in args.runs]
    labels = args.labels or [d.name for d in run_dirs]
    if len(labels) != len(run_dirs):
        print("ERROR: number of --labels must match number of --runs")
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    efficiency_by_experiment: dict[str, pd.DataFrame] = {}
    for label, run_dir in zip(labels, run_dirs):
        run = load_run(run_dir)
        eff = tool_calls_per_ticket(run)
        efficiency_by_experiment[label] = eff

    cross_experiment_efficiency_plot(
        efficiency_by_experiment,
        out_path=out_dir / "efficiency_cross_experiment.png",
    )

    print(json.dumps({
        "out_dir": str(out_dir),
        "labels": labels,
        "runs": [str(d) for d in run_dirs],
    }, indent=2))


if __name__ == "__main__":
    main()
