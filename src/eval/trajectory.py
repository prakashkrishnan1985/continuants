"""
Within-arm drift trajectory analysis.

For each (arm, pair_id, factor), regress the factor against sweep_index.
A non-zero slope is a drift signal: the agent's response on the same
probe changes systematically across sweeps.

Reports per (arm, pair, factor):
  - slope, intercept (OLS)
  - standard error
  - p-value vs slope=0
  - r-squared

The treatment-arm slopes are the central drift indicator. The control-arm
slopes are the baseline noise (control resets between tickets, so its
slope should be near zero modulo stochastic API noise).
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from src.eval.factors import FACTOR_COLUMNS


def _ols_slope(x: pd.Series, y: pd.Series) -> dict:
    nan_result = {
        "slope": float("nan"),
        "intercept": float("nan"),
        "stderr": float("nan"),
        "p_value": float("nan"),
        "r_squared": float("nan"),
        "n": int(len(x)),
    }
    if len(x) < 2 or x.nunique() < 2:
        return nan_result
    try:
        res = stats.linregress(x, y)
    except ValueError:
        # Identical inputs or other degenerate cases produce a NaN result.
        return nan_result
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "stderr": float(res.stderr),
        "p_value": float(res.pvalue),
        "r_squared": float(res.rvalue ** 2),
        "n": int(len(x)),
    }


def trajectories(annotated: pd.DataFrame,
                 factors: list[str] | None = None) -> pd.DataFrame:
    """
    For every (arm, pair_id, probe_id, factor), regress factor on sweep_index.
    Returns a long-form DataFrame.
    """
    factors = factors or FACTOR_COLUMNS
    rows: list[dict] = []
    grouped = annotated.groupby(["arm", "pair_id", "probe_id"])
    for (arm, pair_id, probe_id), grp in grouped:
        if grp["sweep_index"].nunique() < 2:
            continue
        for factor in factors:
            stats_row = _ols_slope(grp["sweep_index"].astype(float), grp[factor].astype(float))
            rows.append({
                "arm": arm,
                "pair_id": pair_id,
                "probe_id": probe_id,
                "factor": factor,
                **stats_row,
            })
    return pd.DataFrame(rows)


def trajectory_summary(traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (arm, factor): mean slope across pairs+probes, with sign distribution.
    """
    if traj_df.empty:
        return pd.DataFrame()

    grouped = traj_df.groupby(["arm", "factor"])
    base = grouped["slope"].agg(count="count", mean="mean", median="median", std="std").reset_index()
    pct_pos = grouped["slope"].apply(lambda s: (s > 0).mean()).rename("pct_positive").reset_index()
    pct_neg = grouped["slope"].apply(lambda s: (s < 0).mean()).rename("pct_negative").reset_index()
    merged = base.merge(pct_pos, on=["arm", "factor"]).merge(pct_neg, on=["arm", "factor"])
    return merged
