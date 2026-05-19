"""
Run the LLM judge against an existing experiment run.

Reads the probe interactions from a completed run, scores every probe
response, and writes:

  <run_dir>/analysis/probe_quality.csv
  <run_dir>/analysis/probe_quality_summary.csv

The summary aggregates by (arm, factor) to give a quick view of the
treatment-vs-control quality gap.

Usage:

    python -m src.eval.score_run /path/to/run_dir
    python -m src.eval.score_run --latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.factors import annotate_factors
from src.eval.llm_judge import score_probe_interactions
from src.eval.loader import discover_runs, load_run, probe_interactions_df
from src.probes.probe_bank import DEFAULT_PROBE_BANK


def _probe_bank_map() -> dict[str, dict]:
    return {
        p.probe_id: {
            "prompt": p.prompt,
            "probe_type": p.probe_type,
            "expected_answer": getattr(p, "expected_answer", None),
        }
        for p in DEFAULT_PROBE_BANK
    }


def score_run(run_dir: Path, max_rows: int | None = None) -> dict:
    run = load_run(run_dir)
    probe_df = probe_interactions_df(run)
    if probe_df.empty:
        return {"status": "no_probes", "run_dir": str(run_dir)}

    annotated = annotate_factors(probe_df)
    print(f"Scoring {len(annotated)} probe interactions from {run.run_id}...")
    scores = score_probe_interactions(
        annotated,
        probe_bank=_probe_bank_map(),
        max_rows=max_rows,
    )

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "probe_quality.csv", index=False)

    summary = (
        scores
        .groupby("arm")[["quality", "scope", "honesty", "tone"]]
        .agg(["mean", "median", "std", "count"])
    )
    summary.to_csv(out_dir / "probe_quality_summary.csv")

    judge_cost = scores.attrs.get("judge_cost_snapshot", {})
    print(f"\nJudge cost: ${judge_cost.get('estimated_usd', 0.0):.3f} "
          f"({judge_cost.get('api_calls', 0)} calls)")
    print(f"Scores written to {out_dir/'probe_quality.csv'}")
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "n_scored": int(len(scores)),
        "judge_cost": judge_cost,
        "out_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_dir", nargs="?", help="Run directory to score.")
    p.add_argument("--latest", action="store_true",
                   help="Score the most recently modified run.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap the number of probe responses to score (testing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.latest:
        runs = discover_runs(PROJECT_ROOT / "experiments")
        if not runs:
            print("No runs found.")
            sys.exit(1)
        run_dir = runs[-1]
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        print("Specify a run directory or use --latest.")
        sys.exit(2)

    result = score_run(run_dir, max_rows=args.max_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
