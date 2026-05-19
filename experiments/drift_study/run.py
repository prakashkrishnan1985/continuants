"""
Drift study experiment runner.

Runs N paired sessions (treatment + control) and saves all event logs
under experiments/drift_study/runs/<timestamp>/.

Requires `ANTHROPIC_API_KEY` in the environment.

Usage:

    python -m experiments.drift_study.run \\
        --pairs 3 \\
        --tickets-per-session 20 \\
        --probe-interval-seconds 600 \\
        --model claude-sonnet-4-6

For a quick smoke run:

    python -m experiments.drift_study.run --pairs 1 --tickets-per-session 4 --probe-interval-seconds 60
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.env import load_environment, require
from src.eval.experiment_runner import PairedSessionConfig, run_paired_session


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pairs", type=int, default=3,
                   help="Number of paired treatment+control sessions to run.")
    p.add_argument("--tickets-per-session", type=int, default=20)
    p.add_argument("--probe-interval-seconds", type=float, default=600.0,
                   help="Inject the probe bank every N seconds during a session.")
    p.add_argument("--seed-base", type=int, default=42,
                   help="Base seed; per-pair seeds are seed_base + i.")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--max-budget-usd-per-arm", type=float, default=0.0,
                   help="Abort an arm if its estimated spend exceeds this dollar amount. 0 disables.")
    p.add_argument("--ticket-source", default="templated",
                   choices=["templated", "bitext"],
                   help="Ticket stream source. 'templated' = synthetic in-code generator. "
                        "'bitext' = real customer-support exchanges from a local Bitext CSV "
                        "(requires data/bitext.csv).")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    load_environment()
    try:
        require("ANTHROPIC_API_KEY")
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    output_dir = Path(__file__).parent / "runs" / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {output_dir}")

    summaries = []
    for i in range(args.pairs):
        seed = args.seed_base + i
        cfg = PairedSessionConfig(
            output_dir=output_dir,
            n_tickets=args.tickets_per_session,
            probe_interval_seconds=args.probe_interval_seconds,
            seed=seed,
            model=args.model,
            max_budget_usd_per_arm=args.max_budget_usd_per_arm,
            ticket_source=args.ticket_source,
        )
        print(f"\nPair {i + 1}/{args.pairs}: seed={seed}")
        result = await run_paired_session(cfg)
        summaries.append(result)
        print(f"  Treatment: {result['treatment']}")
        print(f"  Control:   {result['control']}")
        print(f"  Elapsed:   {result['seconds_elapsed']:.1f}s")

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Drift study run: {output_dir.name}\n")
        f.write(f"Pairs: {args.pairs}\n")
        f.write(f"Tickets per session: {args.tickets_per_session}\n")
        f.write(f"Probe interval (sec): {args.probe_interval_seconds}\n")
        f.write(f"Model: {args.model}\n\n")
        for s in summaries:
            f.write(f"{s['run_id']}: treatment={s['treatment']} control={s['control']} "
                    f"elapsed={s['seconds_elapsed']:.1f}s\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
