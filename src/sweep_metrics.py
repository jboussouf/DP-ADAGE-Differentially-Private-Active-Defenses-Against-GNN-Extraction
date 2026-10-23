"""
sweep_metrics.py
================
Metric computation functions for the DP-ADAGE epsilon-total sweep.

All functions operate on per-query log records (list of dicts) as produced
by SweepSession.run() in sweep_engine.py.

This module is ADDITIVE — it does not modify any existing source file.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Per-query accuracy helpers
# ──────────────────────────────────────────────────────────────────────────────

def rolling_accuracy(correct_flags: List[int], window: int = 10) -> np.ndarray:
    """
    Compute a rolling mean accuracy with a sliding window.

    For the first (window-1) entries the window is expanded (not truncated)
    so the curve begins at t=1 rather than t=window.

    Parameters
    ----------
    correct_flags : list of int (0 or 1)
    window : int, default 10

    Returns
    -------
    np.ndarray of float, shape (len(correct_flags),)
    """
    arr  = np.array(correct_flags, dtype=float)
    n    = len(arr)
    out  = np.empty(n)
    buf  = deque(maxlen=window)
    for i, v in enumerate(arr):
        buf.append(v)
        out[i] = float(np.mean(buf))
    return out


def accuracy_at_checkpoints(
    correct_flags: List[int],
    checkpoints: Tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200, 500),
) -> Dict[int, float]:
    """
    Return accuracy at specific query checkpoints.

    Parameters
    ----------
    correct_flags : list of int
    checkpoints : tuple of int

    Returns
    -------
    dict {t: accuracy} for each t <= len(correct_flags)
    """
    n = len(correct_flags)
    arr = np.array(correct_flags, dtype=float)
    result = {}
    for t in checkpoints:
        if t <= n:
            result[t] = float(arr[:t].mean())
    return result


def utility_retention(correct_flags: List[int], chance_level: float) -> float:
    """
    Normalized area under the rolling-10 accuracy curve, relative to chance.

    AUC  = trapz(rolling_acc) / len
    retention = (AUC - chance) / (1 - chance)

    Returns NaN if all rolling values equal chance (degenerate case).

    Parameters
    ----------
    correct_flags : list of int
    chance_level : float   (e.g. 1/7 for Cora)

    Returns
    -------
    float in [-inf, 1] or NaN
    """
    roll = rolling_accuracy(correct_flags, window=10)
    if len(roll) == 0:
        return float("nan")
    auc = float(np.trapz(roll) / len(roll))
    denom = 1.0 - chance_level
    if abs(denom) < 1e-12:
        return float("nan")
    retention = (auc - chance_level) / denom
    return retention


# ──────────────────────────────────────────────────────────────────────────────
# Session-level metrics
# ──────────────────────────────────────────────────────────────────────────────

def queries_to_floor(log_records: List[Dict[str, Any]]) -> Optional[int]:
    """
    Return the first t where floor_active == True.
    Returns None if the budget was never exhausted.
    """
    for rec in log_records:
        if rec.get("floor_active", False):
            return int(rec["t"])
    return None


def transport_availability(log_records: List[Dict[str, Any]]) -> float:
    """
    Fraction of records where status_code == 200 and no 'exception' field.

    This should be 1.0 at all caps — DP-ADAGE never blocks.
    """
    if not log_records:
        return float("nan")
    ok = sum(
        1 for r in log_records
        if r.get("status_code", 0) == 200 and "exception" not in r
    )
    return ok / len(log_records)


def informative_availability(log_records: List[Dict[str, Any]]) -> float:
    """
    Fraction of records where floor_active == False.

    This is the metric that collapses under tight budget caps.
    """
    if not log_records:
        return float("nan")
    ok = sum(1 for r in log_records if not r.get("floor_active", False))
    return ok / len(log_records)


def benign_budget_attenuation(
    log_records: List[Dict[str, Any]],
    epsilon_max: float = 10.0,
) -> float:
    """
    mean(eps_q) / epsilon_max over pre-floor queries.

    Returns NaN if there are no pre-floor queries.
    """
    pre_floor = [r["eps_q"] for r in log_records if not r.get("floor_active", False)]
    if not pre_floor:
        return float("nan")
    return float(np.mean(pre_floor)) / epsilon_max


def realized_noise_separation(
    benign_logs:   List[Dict[str, Any]],
    attacker_logs: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """
    sigma^2_attacker / sigma^2_benign over pre-floor queries.

    Returns (ratio, reason_string).
    ratio = NaN if either side has no pre-floor queries with sigma > 0.

    Parameters
    ----------
    benign_logs   : per-query logs for the benign policy
    attacker_logs : per-query logs for an attacker policy

    Returns
    -------
    (float, str)   ratio (NaN if not computable) and an explanatory string
    """
    def _pre_floor_sigmas(logs: List[Dict[str, Any]]) -> List[float]:
        return [
            float(r.get("sigma_or_rho", 0.0))
            for r in logs
            if not r.get("floor_active", False) and r.get("sigma_or_rho", 0.0) > 0.0
        ]

    b_sigmas = _pre_floor_sigmas(benign_logs)
    a_sigmas = _pre_floor_sigmas(attacker_logs)

    if not b_sigmas:
        return float("nan"), "NaN: no pre-floor benign queries with sigma>0"
    if not a_sigmas:
        return float("nan"), "NaN: no pre-floor attacker queries with sigma>0"

    b_var = float(np.mean([s ** 2 for s in b_sigmas]))
    a_var = float(np.mean([s ** 2 for s in a_sigmas]))

    if b_var < 1e-30:
        return float("nan"), "NaN: benign sigma^2 is effectively zero"

    ratio  = a_var / b_var
    reason = (
        f"sigma^2_attacker={a_var:.4f} / sigma^2_benign={b_var:.4f}"
        f" over {len(a_sigmas)} attacker and {len(b_sigmas)} benign pre-floor queries"
    )
    return ratio, reason


def strict_regime_coverage(
    log_records: List[Dict[str, Any]],
    epsilon_0:   float = 0.01,
) -> float:
    """
    Fraction of pre-floor queries where eps_q <= 1.1 * epsilon_0.

    Returns NaN if there are no pre-floor queries.
    """
    pre_floor = [r for r in log_records if not r.get("floor_active", False)]
    if not pre_floor:
        return float("nan")
    strict = sum(1 for r in pre_floor if r["eps_q"] <= 1.1 * epsilon_0)
    return strict / len(pre_floor)


# ──────────────────────────────────────────────────────────────────────────────
# Run-level summary
# ──────────────────────────────────────────────────────────────────────────────

def summarize_run(
    log_records:  List[Dict[str, Any]],
    chance_level: float,
    epsilon_max:  float = 10.0,
    epsilon_0:    float = 0.01,
    benign_logs:  Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute all metrics for a single session run.

    Parameters
    ----------
    log_records  : list of per-query dicts
    chance_level : float   (dataset-specific random guess baseline)
    epsilon_max  : float
    epsilon_0    : float
    benign_logs  : optional, for realized_noise_separation; if None, uses log_records
                   as the benign reference (only meaningful when comparing policies)

    Returns
    -------
    flat dict of all metrics
    """
    correct_flags = [
        int(r.get("predicted_label") == r.get("true_label"))
        for r in log_records
    ]
    roll = rolling_accuracy(correct_flags, window=10)

    qtf   = queries_to_floor(log_records)
    t_avail = transport_availability(log_records)
    i_avail = informative_availability(log_records)
    bba     = benign_budget_attenuation(log_records, epsilon_max)
    src     = strict_regime_coverage(log_records, epsilon_0)
    util_r  = utility_retention(correct_flags, chance_level)
    acc_ck  = accuracy_at_checkpoints(correct_flags)

    # Noise separation (needs a paired benign run)
    if benign_logs is not None and benign_logs is not log_records:
        ns_ratio, ns_reason = realized_noise_separation(benign_logs, log_records)
    else:
        ns_ratio, ns_reason = float("nan"), "NaN: no paired benign reference provided"

    metrics = {
        "utility_retention":        util_r,
        "queries_to_floor":         qtf,
        "transport_availability":   t_avail,
        "informative_availability": i_avail,
        "benign_budget_attenuation": bba,
        "realized_noise_separation": ns_ratio,
        "noise_separation_reason":  ns_reason,
        "strict_regime_coverage":   src,
        "rolling_acc_final":        float(roll[-1]) if len(roll) else float("nan"),
    }
    # Accuracy at checkpoints
    for t, acc in acc_ck.items():
        metrics[f"acc_at_t{t}"] = acc

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Cross-seed aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_seeds(per_seed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate a list of metric dicts (one per seed) into mean ± std.

    For every numeric key, produces {key}_mean, {key}_std, {key}_values.
    Non-numeric keys are passed through as-is (first seed's value).

    Parameters
    ----------
    per_seed_metrics : list of dicts

    Returns
    -------
    dict
    """
    if not per_seed_metrics:
        return {}

    all_keys = set()
    for d in per_seed_metrics:
        all_keys.update(d.keys())

    result: Dict[str, Any] = {}
    for key in sorted(all_keys):
        values = [d[key] for d in per_seed_metrics if key in d]
        # Try to convert to float array
        try:
            arr = np.array(values, dtype=float)
            valid = arr[~np.isnan(arr)]
            result[f"{key}_mean"]   = float(np.mean(valid))   if len(valid) else float("nan")
            result[f"{key}_std"]    = float(np.std(valid))    if len(valid) else float("nan")
            result[f"{key}_values"] = values
        except (TypeError, ValueError):
            # Non-numeric: pass through first value
            result[key] = values[0] if values else None

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Statistical comparison
# ──────────────────────────────────────────────────────────────────────────────

def paired_comparison(
    arm1_values:    List[float],
    arm2_values:    List[float],
    n_comparisons:  int = 1,
) -> Dict[str, Any]:
    """
    Holm-corrected paired t-test between two arms over seeds.

    Parameters
    ----------
    arm1_values   : list of floats (one per seed)
    arm2_values   : list of floats (one per seed)
    n_comparisons : int, total number of comparisons being made (for Holm correction)

    Returns
    -------
    dict with keys: t_stat, p_value, p_holm, reject_h0 (at alpha=0.05)
    """
    from scipy.stats import ttest_rel

    a1 = np.array(arm1_values, dtype=float)
    a2 = np.array(arm2_values, dtype=float)

    # Drop pairs where either is NaN
    mask = ~(np.isnan(a1) | np.isnan(a2))
    a1, a2 = a1[mask], a2[mask]

    if len(a1) < 2:
        return {
            "t_stat":    float("nan"),
            "p_value":   float("nan"),
            "p_holm":    float("nan"),
            "reject_h0": False,
            "n_pairs":   int(len(a1)),
            "note":      "Insufficient paired observations (< 2)",
        }

    t_stat, p_value = ttest_rel(a1, a2)
    # Holm correction: p_holm = min(1.0, p * n_comparisons)
    p_holm = min(1.0, float(p_value) * n_comparisons)

    return {
        "t_stat":    float(t_stat),
        "p_value":   float(p_value),
        "p_holm":    p_holm,
        "reject_h0": p_holm < 0.05,
        "n_pairs":   int(len(a1)),
    }
