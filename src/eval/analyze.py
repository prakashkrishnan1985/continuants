"""
Run the analysis pipeline against one or more experiment runs.

Usage:

    python -m src.eval.analyze /Users/prakash/continuants/experiments/drift_study/runs/<run-id>

Or simply:

    python -m src.eval.analyze --latest

Produces:

    <run_dir>/analysis/
        session_summary.csv
        probe_factors.csv
        treatment_vs_control.csv
        trajectories.csv
        trajectory_summary.csv
        efficiency_per_ticket.csv
        efficiency_summary.csv
        report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.compare import compare_all_factors
from src.eval.efficiency import efficiency_summary, tool_calls_per_ticket
from src.eval.factors import FACTOR_COLUMNS, annotate_factors
from src.eval.loader import discover_runs, load_run, probe_interactions_df, session_summary_df
from src.eval.plots import render_all
from src.eval.trajectory import trajectories, trajectory_summary


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.latest:
        runs_root = PROJECT_ROOT / "experiments"
        runs = discover_runs(runs_root)
        if not runs:
            print("No runs found.")
            sys.exit(1)
        return [runs[-1]]
    if not args.run_dirs:
        print("Specify one or more run directories, or use --latest.")
        sys.exit(2)
    return [Path(p) for p in args.run_dirs]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_report(out_dir: Path,
                  summary: pd.DataFrame,
                  comparison: pd.DataFrame,
                  traj_summary: pd.DataFrame,
                  eff_summary: pd.DataFrame) -> None:
    lines = ["# Analysis report\n"]
    lines.append("## Session summary\n")
    if not summary.empty:
        lines.append(summary.to_markdown(index=False))
    lines.append("\n\n## Treatment-vs-control comparison (paired)\n")
    if not comparison.empty:
        lines.append(comparison.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n## Within-arm drift trajectories (summary)\n")
    if not traj_summary.empty:
        lines.append(traj_summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n## Tool-call efficiency per ticket\n")
    if not eff_summary.empty:
        lines.append(eff_summary.to_markdown(index=False, floatfmt=".2f"))
    lines.append("\n")
    (out_dir / "report.md").write_text("\n".join(lines))


def analyze_run(run_dir: Path,
                bootstrap_resamples: int = 5000,
                rng_seed: int = 42) -> dict:
    run = load_run(run_dir)
    if not run.pairs:
        return {"run_dir": str(run_dir), "status": "no_pairs"}

    rng = np.random.default_rng(rng_seed)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = session_summary_df(run)
    _write_csv(summary, out_dir / "session_summary.csv")

    probe_df = probe_interactions_df(run)
    annotated = annotate_factors(probe_df)
    _write_csv(annotated, out_dir / "probe_factors.csv")

    if not annotated.empty:
        comparison = compare_all_factors(
            annotated,
            bootstrap_resamples=bootstrap_resamples,
            rng=rng,
        )
        _write_csv(comparison, out_dir / "treatment_vs_control.csv")

        traj_df = trajectories(annotated)
        _write_csv(traj_df, out_dir / "trajectories.csv")
        traj_summary = trajectory_summary(traj_df)
        _write_csv(traj_summary, out_dir / "trajectory_summary.csv")
    else:
        comparison = pd.DataFrame()
        traj_summary = pd.DataFrame()

    eff_df = tool_calls_per_ticket(run)
    _write_csv(eff_df, out_dir / "efficiency_per_ticket.csv")
    eff_summary = efficiency_summary(eff_df)
    _write_csv(eff_summary, out_dir / "efficiency_summary.csv")

    # Pick up judge scores if a previous run of src.eval.score_run produced them.
    quality_path = out_dir / "probe_quality.csv"
    quality_df = pd.read_csv(quality_path) if quality_path.exists() else None

    # Pick up pairwise judge summary if a previous run of score_run_v2 produced it.
    pairwise_path = out_dir / "probe_quality_v2_summary.csv"
    pairwise_df = pd.read_csv(pairwise_path) if pairwise_path.exists() else None

    # Always render plots; pass quality if present.
    figures = render_all(
        run_dir,
        annotated=annotated if not annotated.empty else pd.DataFrame(),
        comparison=comparison,
        efficiency=eff_df,
        quality=quality_df,
        pairwise=pairwise_df,
    )
    figures_dict = {k: str(v) if v else None for k, v in figures.items()}
    (out_dir / "figures_index.json").write_text(json.dumps(figures_dict, indent=2))

    _write_report(out_dir, summary, comparison, traj_summary, eff_summary)

    return {
        "run_dir": str(run_dir),
        "status": "ok",
        "n_pairs": len(run.pairs),
        "complete_pairs": sum(1 for p in run.pairs if p.is_complete()),
        "out_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_dirs", nargs="*", help="Run directories to analyze.")
    p.add_argument("--latest", action="store_true", help="Analyze only the most recent run.")
    p.add_argument("--bootstrap-resamples", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = _resolve_run_dirs(args)
    for run_dir in run_dirs:
        result = analyze_run(run_dir, bootstrap_resamples=args.bootstrap_resamples)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
