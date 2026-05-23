from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _allow_set(actions: Iterable[str]) -> set:
    out = set()
    for x in actions:
        s = str(x).strip().lower()
        if s:
            out.add(s)
    return out


def _safe_action_set(raw: Any, default: Iterable[str]) -> set:
    """"""
    if isinstance(raw, (list, tuple, set)):
        out = _allow_set(raw)
        if len(out) > 0:
            return out
    return _allow_set(default)


def apply_safety_fallback(
    *,
    action_id: str,
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    allowed_actions: Iterable[str],
) -> Tuple[str, bool, str]:
    aid = str(action_id or "hold").strip().lower()
    allow = _allow_set(allowed_actions)
    safety = cfg.get("safety", {}) if isinstance(cfg, dict) else {}
    hard_fallback = str(safety.get("hard_fallback", "hold")).strip().lower() or "hold"
    if hard_fallback not in allow:
        hard_fallback = "hold"

    if aid not in allow:
        return hard_fallback, True, f"action_not_allowed:{aid}"

    ris_gate = safety.get("ris_boost_gate", {}) if isinstance(safety, dict) else {}
    
    gate_actions = _safe_action_set(
        ris_gate.get("actions", None) if isinstance(ris_gate, dict) else None,
        ("ris_boost", "comp_boost", "joint_boost"),
    )
    min_load = _to_float(ris_gate.get("min_load", 0.7), 0.7)
    max_vio = _to_float(ris_gate.get("max_vio_any", 0.95), 0.95)
    load = _to_float(state.get("eval_load_now", state.get("eval_load", 1.0)), 1.0)
    vio_any = _to_float(state.get("vio_any_now", state.get("vio_any", 0.0)), 0.0)
    if aid in gate_actions:
        if load < min_load:
            return hard_fallback, True, f"boost_gate_load:{aid}:{load:.4f}<{min_load:.4f}"
        if vio_any > max_vio:
            return hard_fallback, True, f"boost_gate_vio:{aid}:{vio_any:.4f}>{max_vio:.4f}"

    guard = safety.get("stability_guard_gate", {}) if isinstance(safety, dict) else {}
    clip_hi = _to_float(guard.get("clipfrac_high", 0.30), 0.30)
    kl_hi = _to_float(guard.get("kl_high", 0.004), 0.004)
    clip_hard = _to_float(guard.get("clipfrac_hard", clip_hi * 1.35), clip_hi * 1.35)
    kl_hard = _to_float(guard.get("kl_hard", kl_hi * 1.5), kl_hi * 1.5)
    idle_hi = _to_float(guard.get("idle_high", 0.65), 0.65)
    boundary_hi = _to_float(guard.get("boundary_high", 0.45), 0.45)
    user_gap_hi = _to_float(guard.get("user_nn_high", 0.55), 0.55)
    centroid_gap_hi = _to_float(guard.get("centroid_gap_high", 0.35), 0.35)

    clip_now = _to_float(state.get("clipfrac_now"), 0.0)
    kl_now = _to_float(state.get("kl_per_dim_now"), 0.0)
    idle_now = _to_float(state.get("traj_idle_ratio_now"), -1.0)
    boundary_now = _to_float(state.get("traj_boundary_stick_frac_now"), -1.0)
    user_gap_now = _to_float(state.get("traj_user_nn_dist_norm_now"), -1.0)
    centroid_gap_now = _to_float(state.get("traj_centroid_gap_norm_now"), -1.0)

    ppo_unstable_hard = (clip_now > clip_hard) or (kl_now > kl_hard)
    traj_bad = (
        (idle_now > idle_hi)
        or (boundary_now > boundary_hi)
        or (user_gap_now > user_gap_hi)
        or (centroid_gap_now > centroid_gap_hi)
    )

    # When trajectory is degraded but PPO diagnostics are stable, force a recovery action.
    if traj_bad and (not ppo_unstable_hard) and aid in ("hold", "stability_guard", "constraint_tight"):
        if "trajectory_recover" in allow:
            return "trajectory_recover", True, "traj_override:degraded"
        if "constraint_relax" in allow:
            return "constraint_relax", True, "traj_override:degraded"

    # Strict protection only for hard PPO instability; trajectory degradation is handled separately.
    if ppo_unstable_hard and aid in ("comp_boost", "ris_boost", "joint_boost", "constraint_relax", "trajectory_recover"):
        if "stability_guard" in allow:
            return "stability_guard", True, "guard_override:ppo_unstable"
        return hard_fallback, True, "guard_override:ppo_unstable"

    return aid, False, ""
