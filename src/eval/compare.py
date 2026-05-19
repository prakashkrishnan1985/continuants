"""
Treatment-vs-control statistical comparison.

For each behavioural factor, compare treatment-arm values to control-arm
values matched by (pair_id, probe_id, sweep_index). The natural test is
a paired comparison: same probe at same sweep position in the same pair,
treatment vs control.

We report:
  - n (number of matched probe interactions)
  - mean difference (treatment - control)
  - standard error
  - 95% CI from bootstrap (no normality assumption)
  - paired Wilcoxon signed-rank p-value (non-parametric)
  - Cohen's d (effect size, normalized by paired std)

Pre-registered analyses should reference this function explicitly so
the result is what was committed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.factors import FACTOR_COLUMNS


def _bootstrap_ci(diffs: np.ndarray, n_resamples: int = 5000, ci: float = 0.95,
                  rng: np.random.Generator | None = None) -> tuple[float, float]:
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(seed=42)
    means = rng.choice(diffs, size=(n_resamples, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return (float(lo), float(hi))


def paired_comparison(annotated: pd.DataFrame,
                      factor: str,
                      bootstrap_resamples: int = 5000,
                      rng: np.random.Generator | None = None) -> dict:
    """
    Compare treatment vs control on `factor`, matched by probe+sweep+pair.
    """
    if factor not in annotated.columns:
        raise KeyError(f"Factor {factor!r} not found in DataFrame")

    pivot = (
        annotated
        .groupby(["pair_id", "probe_id", "sweep_index", "arm"])[factor]
        .mean()  # in case there are duplicates
        .unstack("arm")
    )
    if "treatment" not in pivot.columns or "control" not in pivot.columns:
        return {
            "factor": factor,
            "n": 0,
            "mean_diff": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "wilcoxon_p": float("nan"),
            "cohens_d": float("nan"),
            "note": "Missing one arm; cannot pair.",
        }
    paired = pivot.dropna(subset=["treatment", "control"])
    if len(paired) < 2:
        return {
            "factor": factor,
            "n": int(len(paired)),
            "mean_diff": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "wilcoxon_p": float("nan"),
            "cohens_d": float("nan"),
            "note": "Too few paired observations.",
        }

    diffs = (paired["treatment"] - paired["control"]).to_numpy()
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    ci_lo, ci_hi = _bootstrap_ci(diffs, n_resamples=bootstrap_resamples, rng=rng)

    try:
        wilcoxon = stats.wilcoxon(diffs, zero_method="wilcox", correction=False,
                                  alternative="two-sided", method="auto")
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        # Wilcoxon fails if all diffs are zero
        wilcoxon_p = 1.0

    return {
        "factor": factor,
        "n": int(len(paired)),
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wilcoxon_p": wilcoxon_p,
        "cohens_d": cohens_d,
        "note": "",
    }


def compare_all_factors(annotated: pd.DataFrame,
                        factors: list[str] | None = None,
                        bootstrap_resamples: int = 5000,
                        rng: np.random.Generator | None = None) -> pd.DataFrame:
    factors = factors or FACTOR_COLUMNS
    return pd.DataFrame([
        paired_comparison(annotated, factor, bootstrap_resamples, rng)
        for factor in factors
    ])
