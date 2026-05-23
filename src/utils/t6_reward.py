from __future__ import annotations

"""
PhaseT6 


-  T6 
- 
"""

from typing import Dict, Tuple

import numpy as np


def update_ema_mean_var(
    mean_prev: float,
    var_prev: float,
    value: float,
    beta: float = 0.95,
    eps: float = 1e-9,
) -> Tuple[float, float, float]:
    """ EMA """
    b = float(np.clip(beta, 0.0, 0.9999))
    x = float(value)
    m0 = float(mean_prev) if np.isfinite(mean_prev) else x
    v0 = float(var_prev) if np.isfinite(var_prev) else 1.0
    m1 = float(b * m0 + (1.0 - b) * x)
    second = float(b * (v0 + m0 * m0) + (1.0 - b) * (x * x))
    v1 = float(max(second - m1 * m1, eps))
    std = float(np.sqrt(v1 + eps))
    return m1, v1, std


def compute_t6_reward(
    *,
    policy_cost: float,
    ref_cost: float,
    cost_scale: float,
    potential_prev: float,
    potential_curr: float,
    gamma: float,
    beta_potential: float,
    lambda_value: float,
    violation_continuous: float,
    traj_bonus: float,
    traj_weight: float,
    main_weight: float = 1.0,
    eps: float = 1e-9,
) -> Dict[str, float]:
    """
     T6 

    
    r = - (c - c_ref) / (sigma + eps)
        + beta * (gamma*Phi(s') - Phi(s))
        - lambda * g
        + eta * r_traj
    """
    c = float(policy_cost)
    c_ref = float(ref_cost)
    sigma = float(max(abs(cost_scale), eps))
    g_t = float(np.clip(violation_continuous, 0.0, 1e9))
    lam = float(max(lambda_value, 0.0))
    eta = float(traj_weight)
    w_main = float(max(main_weight, 0.0))
    r_traj = float(traj_bonus)
    beta = float(beta_potential)
    gm = float(gamma)

    rel_cost_raw = float(c - c_ref)
    rel_cost_norm = float(rel_cost_raw / sigma)
    reward_main = float(-w_main * rel_cost_norm)
    reward_potential = float(beta * (gm * float(potential_curr) - float(potential_prev)))
    reward_constraint = float(-lam * g_t)
    reward_traj = float(eta * r_traj)
    reward_total = float(reward_main + reward_potential + reward_constraint + reward_traj)

    return {
        "reward_total": reward_total,
        "reward_main": reward_main,
        "reward_potential": reward_potential,
        "reward_constraint": reward_constraint,
        "reward_traj": reward_traj,
        "relative_cost_raw": rel_cost_raw,
        "relative_cost_norm": rel_cost_norm,
        "main_weight": w_main,
        "cost_scale": sigma,
        "policy_cost": c,
        "ref_cost": c_ref,
        "lambda_v_effective": lam,
        "violation_continuous": g_t,
    }
