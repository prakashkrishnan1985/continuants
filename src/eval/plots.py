"""
Figure generation for the paper.

All plots are produced from the same analysis DataFrames used elsewhere
in the pipeline. Each function takes a DataFrame and an output path,
writes a PNG, and returns the path. Style is deliberately plain: black-
and-white-readable, no unnecessary decoration, paper-friendly.

Functions:

  - trajectory_plot: factor-trajectory across sweeps, treatment vs
    control overlay, one panel per factor.
  - treatment_vs_control_bar: bar chart of mean factor values with
    95% bootstrap CIs.
  - efficiency_per_ticket_plot: tool calls per ticket index, treatment
    vs control.
  - quality_summary_plot: judge-scored quality dimensions, treatment
    vs control.
  - render_all: convenience wrapper that builds every figure for one run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.factors import FACTOR_COLUMNS


_ARM_COLOURS = {"treatment": "#1e88e5", "control": "#fb8c00"}
_ARM_MARKERS = {"treatment": "o", "control": "s"}


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def trajectory_plot(annotated: pd.DataFrame,
                    factors: Iterable[str] | None = None,
                    out_path: Path | None = None) -> Path | None:
    """
    Plot factor trajectory across sweep_index, treatment vs control.

    For each factor, averages across all pairs and probes within an arm
    at each sweep_index, plots the resulting curve with 1-SEM error bars.
    """
    factors = list(factors or FACTOR_COLUMNS)
    if annotated.empty or "sweep_index" not in annotated.columns:
        return None

    n = len(factors)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

    for idx, factor in enumerate(factors):
        ax = axes[idx // cols][idx % cols]
        for arm in ("treatment", "control"):
            sub = annotated[annotated["arm"] == arm]
            if sub.empty or factor not in sub.columns:
                continue
            grouped = (
                sub.groupby("sweep_index")[factor]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
            ax.errorbar(
                grouped["sweep_index"], grouped["mean"], yerr=grouped["sem"],
                marker=_ARM_MARKERS[arm], color=_ARM_COLOURS[arm],
                label=arm.capitalize(), capsize=3, linewidth=1.5,
            )
        ax.set_title(factor, fontsize=10)
        ax.set_xlabel("Sweep index (0=baseline → final)")
        ax.set_ylabel(factor)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Hide unused panels.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Drift trajectories: treatment vs control across probe sweeps", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def treatment_vs_control_bar(comparison_df: pd.DataFrame,
                             out_path: Path | None = None) -> Path | None:
    """
    Bar chart of treatment-minus-control mean differences per factor,
    with 95% bootstrap CIs.
    """
    if comparison_df.empty:
        return None
    df = comparison_df.dropna(subset=["mean_diff"]).copy()
    if df.empty:
        return None
    df = df.sort_values("mean_diff")

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["mean_diff"],
            xerr=[df["mean_diff"] - df["ci_lo"], df["ci_hi"] - df["mean_diff"]],
            color="#43a047", alpha=0.7, capsize=4)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["factor"])
    ax.set_xlabel("Treatment minus control (mean diff)")
    ax.set_title("Per-factor effect: treatment minus control, with 95% bootstrap CI")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def efficiency_per_ticket_plot(efficiency_df: pd.DataFrame,
                                out_path: Path | None = None) -> Path | None:
    """
    Plot tool calls per ticket index, treatment vs control, averaged
    across pairs.
    """
    if efficiency_df.empty:
        return None
    grouped = (
        efficiency_df
        .groupby(["arm", "ticket_index"])["tool_calls"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(8, 4))
    for arm in ("treatment", "control"):
        sub = grouped[grouped["arm"] == arm]
        if sub.empty:
            continue
        ax.errorbar(sub["ticket_index"], sub["mean"], yerr=sub["sem"],
                    marker=_ARM_MARKERS[arm], color=_ARM_COLOURS[arm],
                    label=arm.capitalize(), capsize=2, linewidth=1.5)
    ax.set_xlabel("Ticket index in session")
    ax.set_ylabel("Tool calls per ticket")
    ax.set_title("Tool-call efficiency over the session")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def quality_summary_plot(quality_df: pd.DataFrame,
                         out_path: Path | None = None) -> Path | None:
    """
    Plot mean judge scores per dimension, treatment vs control.
    """
    if quality_df.empty:
        return None
    dimensions = ["quality", "scope", "honesty", "tone"]
    available = [d for d in dimensions if d in quality_df.columns]
    if not available:
        return None

    arm_means = quality_df.groupby("arm")[available].agg(["mean", "sem"])

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(available))
    width = 0.35
    for i, arm in enumerate(("treatment", "control")):
        if arm not in arm_means.index:
            continue
        means = [arm_means.loc[arm, (d, "mean")] for d in available]
        sems = [arm_means.loc[arm, (d, "sem")] for d in available]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=sems,
               color=_ARM_COLOURS[arm], label=arm.capitalize(),
               alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(available)
    ax.set_ylabel("Mean judge score (0–5)")
    ax.set_title("LLM-judge quality scores by arm")
    ax.set_ylim(0, 5)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def pairwise_verdict_plot(pairwise_summary: pd.DataFrame,
                          out_path: Path | None = None,
                          title_suffix: str = "") -> Path | None:
    """
    Horizontal stacked bar chart per dimension: treatment wins, ties,
    control wins. Sorted by treatment win rate.

    Expects the output shape of `src.eval.llm_judge_v2.pairwise_summary`.
    """
    if pairwise_summary.empty:
        return None
    df = pairwise_summary.copy()
    for col in ("treatment", "control", "tied"):
        if col not in df.columns:
            df[col] = 0
    df = df.sort_values("treatment_win_rate_decisive", na_position="last")

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["treatment"], color=_ARM_COLOURS["treatment"], label="Treatment preferred")
    ax.barh(y, df["tied"], left=df["treatment"], color="#bdbdbd", label="Tied")
    ax.barh(y, df["control"], left=df["treatment"] + df["tied"],
            color=_ARM_COLOURS["control"], label="Control preferred")
    ax.set_yticks(y)
    ax.set_yticklabels(df["dimension"])
    ax.set_xlabel("Pairwise comparisons (count)")
    ax.set_title(f"Pairwise judge verdicts by dimension{(' — ' + title_suffix) if title_suffix else ''}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def new_guy_symptom_plot(annotated: pd.DataFrame,
                         factor: str = "response_length_words",
                         out_path: Path | None = None,
                         title_suffix: str = "") -> Path | None:
    """
    The signature figure: factor mean per sweep_index, treatment vs control.
    Demonstrates the diverging slopes that constitute the New Guy Symptom.
    """
    if annotated.empty or "sweep_index" not in annotated.columns:
        return None
    grouped = (
        annotated
        .groupby(["arm", "sweep_index"])[factor]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm in ("treatment", "control"):
        sub = grouped[grouped["arm"] == arm]
        if sub.empty:
            continue
        ax.errorbar(
            sub["sweep_index"], sub["mean"], yerr=sub["sem"],
            marker=_ARM_MARKERS[arm], color=_ARM_COLOURS[arm],
            label=arm.capitalize(), capsize=4, linewidth=2.0, markersize=8,
        )
    ax.set_xlabel("Probe sweep index (0 = baseline → final)")
    ax.set_ylabel(factor.replace("_", " ").capitalize())
    title = "New Guy Symptom: divergent within-session trajectories"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def efficiency_widening_plot(efficiency_df: pd.DataFrame,
                             out_path: Path | None = None,
                             title_suffix: str = "") -> Path | None:
    """
    Per-ticket tool-call curves with the treatment-control gap shaded.
    Shows whether the efficiency gap widens, narrows, or stays constant
    across the session.
    """
    if efficiency_df.empty:
        return None
    grouped = (
        efficiency_df
        .groupby(["arm", "ticket_index"])["tool_calls"]
        .mean()
        .unstack("arm")
    )
    if "treatment" not in grouped.columns or "control" not in grouped.columns:
        return None
    grouped = grouped.sort_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = grouped.index
    ax.plot(x, grouped["control"], marker="s", color=_ARM_COLOURS["control"],
            label="Control", linewidth=2)
    ax.plot(x, grouped["treatment"], marker="o", color=_ARM_COLOURS["treatment"],
            label="Treatment", linewidth=2)
    ax.fill_between(x, grouped["treatment"], grouped["control"],
                    alpha=0.15, color="#9e9e9e", label="Efficiency gap")
    ax.set_xlabel("Ticket index in session")
    ax.set_ylabel("Tool calls per ticket")
    title = "Tool-call efficiency: gap across session"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def cross_experiment_efficiency_plot(efficiency_by_experiment: dict[str, pd.DataFrame],
                                      out_path: Path | None = None) -> Path | None:
    """
    Side-by-side comparison of mean tool calls per ticket across
    experiments. `efficiency_by_experiment` maps a label
    (e.g., "E1: templated") to a `tool_calls_per_ticket` DataFrame.
    """
    if not efficiency_by_experiment:
        return None
    rows: list[dict] = []
    for label, df in efficiency_by_experiment.items():
        if df is None or df.empty:
            continue
        means = df.groupby("arm")["tool_calls"].mean()
        rows.append({"experiment": label,
                     "treatment": means.get("treatment", float("nan")),
                     "control": means.get("control", float("nan"))})
    if not rows:
        return None
    plot_df = pd.DataFrame(rows).set_index("experiment")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width / 2, plot_df["treatment"], width,
           color=_ARM_COLOURS["treatment"], label="Treatment")
    ax.bar(x + width / 2, plot_df["control"], width,
           color=_ARM_COLOURS["control"], label="Control")
    for i, exp in enumerate(plot_df.index):
        ratio = plot_df.loc[exp, "control"] / max(plot_df.loc[exp, "treatment"], 1e-9)
        ax.text(i, max(plot_df.loc[exp, "control"], plot_df.loc[exp, "treatment"]) * 1.02,
                f"{ratio:.1f}×", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index)
    ax.set_ylabel("Mean tool calls per ticket")
    ax.set_title("Cross-experiment efficiency: treatment vs control")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def render_all(run_dir: Path,
               annotated: pd.DataFrame,
               comparison: pd.DataFrame,
               efficiency: pd.DataFrame,
               quality: pd.DataFrame | None = None,
               pairwise: pd.DataFrame | None = None) -> dict[str, Path | None]:
    """Render every figure for one run and return the produced paths."""
    out_dir = run_dir / "analysis" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path | None] = {
        "trajectory": trajectory_plot(annotated, out_path=out_dir / "drift_trajectories.png"),
        "tvc_bar": treatment_vs_control_bar(comparison, out_path=out_dir / "treatment_vs_control_bar.png"),
        "efficiency": efficiency_per_ticket_plot(efficiency, out_path=out_dir / "efficiency_per_ticket.png"),
        "new_guy_symptom": new_guy_symptom_plot(annotated, out_path=out_dir / "new_guy_symptom.png"),
        "efficiency_widening": efficiency_widening_plot(efficiency, out_path=out_dir / "efficiency_widening.png"),
    }
    if quality is not None and not quality.empty:
        out["quality"] = quality_summary_plot(quality, out_path=out_dir / "quality_scores.png")
    if pairwise is not None and not pairwise.empty:
        out["pairwise_verdicts"] = pairwise_verdict_plot(pairwise, out_path=out_dir / "pairwise_verdicts.png")
    return out
