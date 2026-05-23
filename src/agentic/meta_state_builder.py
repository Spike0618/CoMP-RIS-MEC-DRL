from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _finite_tail(vals: Iterable[Any], n: int) -> np.ndarray:
    arr = np.asarray([_to_float(v) for v in vals], dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return np.asarray([], dtype=np.float64)
    if n > 0 and arr.size > n:
        return arr[-n:]
    return arr


def _trend(vals: Iterable[Any], n: int) -> float:
    x = _finite_tail(vals, n)
    if x.size <= 1:
        return 0.0
    return float((x[-1] - x[0]) / max(1, x.size - 1))


def build_meta_state(
    *,
    history: Dict[str, List[float]],
    latest: Dict[str, Any],
    window: int = 4,
) -> Dict[str, Any]:
    """
     meta-controller 
    """
    w = int(max(2, window))
    out: Dict[str, Any] = {
        "paper_cost_now": _to_float(latest.get("paper_cost")),
        "paper_return_now": _to_float(latest.get("paper_return")),
        "vio_any_now": _to_float(latest.get("vio_any")),
        "coverage_now": _to_float(latest.get("coverage")),
        "clipfrac_now": _to_float(latest.get("clipfrac")),
        "kl_per_dim_now": _to_float(latest.get("kl_per_dim")),
        "explained_variance_now": _to_float(latest.get("explained_variance")),
        "comp_gain_ratio_now": _to_float(latest.get("comp_gain_ratio")),
        "ris_gain_ratio_now": _to_float(latest.get("ris_gain_ratio")),
        "eval_load_now": _to_float(latest.get("eval_load")),
        "train_load_now": _to_float(latest.get("train_load")),
        "traj_idle_ratio_now": _to_float(latest.get("traj_idle_ratio")),
        "traj_boundary_stick_frac_now": _to_float(latest.get("traj_boundary_stick_frac")),
        "traj_user_nn_dist_norm_now": _to_float(latest.get("traj_user_nn_dist_norm")),
        "traj_centroid_gap_norm_now": _to_float(latest.get("traj_centroid_gap_norm")),
        "traj_switchback_ratio_now": _to_float(latest.get("traj_switchback_ratio")),
        "window": int(w),
    }

    
    out["paper_cost_trend"] = _trend(history.get("paper_cost", []), w)
    out["paper_return_trend"] = _trend(history.get("paper_return", []), w)
    out["vio_any_trend"] = _trend(history.get("vio_any", []), w)
    out["coverage_trend"] = _trend(history.get("coverage", []), w)
    out["clipfrac_trend"] = _trend(history.get("clipfrac", []), w)
    out["kl_per_dim_trend"] = _trend(history.get("kl_per_dim", []), w)
    out["ev_trend"] = _trend(history.get("explained_variance", []), w)
    out["comp_gain_ratio_trend"] = _trend(history.get("comp_gain_ratio", []), w)
    out["ris_gain_ratio_trend"] = _trend(history.get("ris_gain_ratio", []), w)

    
    for key in (
        "paper_cost",
        "paper_return",
        "vio_any",
        "clipfrac",
        "kl_per_dim",
        "explained_variance",
    ):
        arr = _finite_tail(history.get(key, []), w)
        out[f"{key}_std_w"] = float(np.std(arr)) if arr.size > 0 else float("nan")

    return out
