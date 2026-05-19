"""
Run the v2 pairwise judge on a completed experiment.

Reads probe interactions, finds every (pair, probe, sweep_index) where
both arms responded, runs pairwise judge, writes:

  <run_dir>/analysis/probe_quality_v2.csv          (long form, one row per dimension)
  <run_dir>/analysis/probe_quality_v2_summary.csv  (win rates per dimension)

Existing v1 outputs (probe_quality.csv) are preserved.

Usage:
  python -m src.eval.score_run_v2 --latest
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
from src.eval.llm_judge_v2 import pairwise_summary, score_paired_probes
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


def score_run_v2(run_dir: Path, max_rows: int | None = None) -> dict:
    run = load_run(run_dir)
    probe_df = probe_interactions_df(run)
    if probe_df.empty:
        return {"status": "no_probes", "run_dir": str(run_dir)}

    annotated = annotate_factors(probe_df)
    print(f"Pairwise-judging probe interactions from {run.run_id}...")
    scores = score_paired_probes(
        annotated,
        probe_bank=_probe_bank_map(),
        max_rows=max_rows,
    )
    if scores.empty:
        return {"status": "no_pairs", "run_dir": str(run_dir)}

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "probe_quality_v2.csv", index=False)

    summary = pairwise_summary(scores)
    summary.to_csv(out_dir / "probe_quality_v2_summary.csv", index=False)

    judge_cost = scores.attrs.get("judge_cost_snapshot", {})
    print(f"\nv2 judge cost: ${judge_cost.get('estimated_usd', 0.0):.3f} "
          f"({judge_cost.get('api_calls', 0)} calls)")
    print(f"Scores: {out_dir/'probe_quality_v2.csv'}")
    print(f"Summary: {out_dir/'probe_quality_v2_summary.csv'}")
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "n_pairs_compared": len(scores) // len(summary["dimension"]) if not summary.empty else 0,
        "judge_cost": judge_cost,
        "out_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--max-rows", type=int, default=None)
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

    result = score_run_v2(run_dir, max_rows=args.max_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
