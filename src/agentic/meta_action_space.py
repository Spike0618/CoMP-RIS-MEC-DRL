from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


DEFAULT_ALLOWED_ACTIONS = (
    "hold",
    "trajectory_recover",
    "stability_guard",
    "constraint_tight",
    "constraint_relax",
    "comp_boost",
    "ris_boost",
    "joint_boost",
)


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _clip(x: float, lo: float, hi: float) -> Tuple[float, bool]:
    y = float(np.clip(float(x), float(lo), float(hi)))
    return y, bool(abs(y - float(x)) > 1e-12)


def allowed_actions(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    space = cfg.get("action_space", {}) if isinstance(cfg, dict) else {}
    allow = space.get("allow", None) if isinstance(space, dict) else None
    if isinstance(allow, (list, tuple, set)):
        out = []
        for x in allow:
            s = str(x).strip().lower()
            if s:
                out.append(s)
        if out:
            return tuple(dict.fromkeys(out))
    return tuple(DEFAULT_ALLOWED_ACTIONS)


def clamp_ranges(cfg: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    safety = cfg.get("safety", {}) if isinstance(cfg, dict) else {}
    clamp = safety.get("clamp", {}) if isinstance(safety, dict) else {}

    def _pair(key: str, lo: float, hi: float) -> Tuple[float, float]:
        raw = clamp.get(key, [lo, hi])
        try:
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                a = float(raw[0])
                b = float(raw[1])
                if b < a:
                    a, b = b, a
                return float(a), float(b)
        except Exception:
            pass
        return float(lo), float(hi)

    return {
        "lambda_v": _pair("lambda_v", 0.0, 20.0),
        "beta_comp_max": _pair("beta_comp_max", 0.0, 0.08),
        "beta_ris_max": _pair("beta_ris_max", 0.0, 0.15),
    }


def action_patch(
    action_id: str,
    current: Dict[str, float],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    aid = str(action_id or "hold").strip().lower()
    rng = clamp_ranges(cfg)
    safety = cfg.get("safety", {}) if isinstance(cfg, dict) else {}
    step_cfg = safety.get("max_abs_step", {}) if isinstance(safety, dict) else {}
    guard_cfg = safety.get("stability_guard_policy", {}) if isinstance(safety, dict) else {}

    dl = max(1e-6, _to_float(step_cfg.get("lambda_v", 1.2), 1.2))
    dbc = max(1e-6, _to_float(step_cfg.get("beta_comp_max", 0.01), 0.01))
    dbr = max(1e-6, _to_float(step_cfg.get("beta_ris_max", 0.02), 0.02))

    cur_lambda = _to_float(current.get("lambda_v", 0.0), 0.0)
    cur_bc = _to_float(current.get("beta_comp_max", 0.0), 0.0)
    cur_br = _to_float(current.get("beta_ris_max", 0.0), 0.0)

    l_lo, l_hi = rng["lambda_v"]
    bc_lo, bc_hi = rng["beta_comp_max"]
    br_lo, br_hi = rng["beta_ris_max"]

    lambda_guard_cap = float(
        np.clip(
            _to_float(guard_cfg.get("lambda_guard_cap", min(12.0, l_hi)), min(12.0, l_hi)),
            l_lo,
            l_hi,
        )
    )
    beta_comp_floor = float(
        np.clip(_to_float(guard_cfg.get("beta_comp_floor", max(0.015, bc_lo)), max(0.015, bc_lo)), bc_lo, bc_hi)
    )
    beta_ris_floor = float(
        np.clip(_to_float(guard_cfg.get("beta_ris_floor", max(0.025, br_lo)), max(0.025, br_lo)), br_lo, br_hi)
    )

    tgt_lambda = cur_lambda
    tgt_bc = cur_bc
    tgt_br = cur_br

    if aid == "hold":
        pass
    elif aid == "trajectory_recover":
        relax = 1.3 if cur_lambda > 0.75 * lambda_guard_cap else 1.0
        tgt_lambda = cur_lambda - relax * dl
        tgt_bc = max(cur_bc + 1.0 * dbc, beta_comp_floor)
        tgt_br = max(cur_br + 1.2 * dbr, beta_ris_floor)
    elif aid == "stability_guard":
        rise = 0.60 if cur_lambda < 0.80 * lambda_guard_cap else 0.25
        tgt_lambda = min(cur_lambda + rise * dl, lambda_guard_cap)
        tgt_bc = max(cur_bc * 0.92, beta_comp_floor)
        tgt_br = max(cur_br * 0.85, beta_ris_floor)
    elif aid == "constraint_tight":
        tgt_lambda = cur_lambda + dl
        tgt_bc = max(cur_bc, beta_comp_floor)
        tgt_br = max(cur_br, beta_ris_floor)
    elif aid == "constraint_relax":
        relax = 1.0 if cur_lambda <= 0.70 * lambda_guard_cap else 1.5
        tgt_lambda = cur_lambda - relax * dl
        tgt_bc = max(cur_bc, beta_comp_floor)
        tgt_br = max(cur_br, beta_ris_floor)
    elif aid == "comp_boost":
        tgt_bc = max(cur_bc + dbc, beta_comp_floor)
    elif aid == "ris_boost":
        tgt_br = max(cur_br + dbr, beta_ris_floor)
    elif aid == "joint_boost":
        tgt_bc = max(cur_bc + 0.7 * dbc, beta_comp_floor)
        tgt_br = max(cur_br + 0.7 * dbr, beta_ris_floor)
    else:
        aid = "hold"

    out: Dict[str, Any] = {
        "action_id": aid,
        "requested_action_id": str(action_id or "").strip().lower(),
        "patch": {},
        "delta": {},
        "clamped": {},
    }

    v, c = _clip(tgt_lambda, l_lo, l_hi)
    if abs(v - cur_lambda) > 1e-12:
        out["patch"]["lambda_v"] = float(v)
        out["delta"]["lambda_v"] = float(v - cur_lambda)
        out["clamped"]["lambda_v"] = bool(c)

    v, c = _clip(tgt_bc, bc_lo, bc_hi)
    if abs(v - cur_bc) > 1e-12:
        out["patch"]["beta_comp_max"] = float(v)
        out["delta"]["beta_comp_max"] = float(v - cur_bc)
        out["clamped"]["beta_comp_max"] = bool(c)

    v, c = _clip(tgt_br, br_lo, br_hi)
    if abs(v - cur_br) > 1e-12:
        out["patch"]["beta_ris_max"] = float(v)
        out["delta"]["beta_ris_max"] = float(v - cur_br)
        out["clamped"]["beta_ris_max"] = bool(c)

    return out
