"""
Phase Cscores_fixed C1  scores / C2 


- env.obs_flat 
-  score  delta  [-1, 1] PPO Actor  tanh 
- // scores_fixed 
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def _safe_getattr(obj: Any, name: str, default: Any) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _decode_obs_for_scores(cfg: Any, obs_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     obs_flat scores 

    
    -  obs_flat PhaseA/B QoS
    -  src/envs/comp_ris_env_joint.py:CompRISEnvJoint.obs_flat() 
        [ q(2M), w(2I), dist2_n(I*M), g_feat(I*M), Dn(I), Cn(I), fmax_n(M), ... ]
    """
    M = int(_safe_getattr(cfg, "M", 0) or 0)
    I = int(_safe_getattr(cfg, "I", 0) or 0)
    if M <= 0 or I <= 0:
        raise ValueError(f"M={M}, I={I}")

    obs = np.asarray(obs_raw, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.ndim != 2:
        raise ValueError(f"obs_rawndim={obs.ndim}")

    need = 2 * M + 2 * I + 2 * I * M + 2 * I + M
    if obs.shape[1] < need:
        raise ValueError(f"obs_flatgot={obs.shape[1]} need>={need}")

    p = 0
    p += 2 * M  # q
    p += 2 * I  # w

    dist2_n = obs[:, p : p + I * M].reshape(obs.shape[0], I, M)
    p += I * M

    g_feat = obs[:, p : p + I * M].reshape(obs.shape[0], I, M)
    p += I * M

    
    p += I  # Dn
    p += I  # Cn

    fmax_n = obs[:, p : p + M].reshape(obs.shape[0], M)
    return dist2_n, g_feat, fmax_n


def compute_scores_fixed_action(
    *,
    cfg: Any,
    obs_raw: np.ndarray,
    mode: str = "balanced",
    score_tanh_scale: float = 2.5,
) -> np.ndarray:
    """
     scores  delta 

    
    - cfg: JointEnvConfig
    - obs_raw:  (obs_dim,)  (B, obs_dim)  CollectorPreprocess  obs_raw
    - mode: fixed scores balanced/greedy_delay/greedy_energy
    - score_tanh_scale:  score  [-1,1] 

    
    - scores_act:  (score_dim,)  (B, score_dim)dtype=float32 [-1,1]
    """
    mode_n = str(mode or "balanced").strip().lower()
    if mode_n not in ("balanced", "greedy_delay", "greedy_energy"):
        raise ValueError(
            f"Invalid fixed score mode: {mode_n}. "
            "Expected one of: balanced, greedy_delay, greedy_energy"
        )

    dist2_n, g_feat, fmax_n = _decode_obs_for_scores(cfg, obs_raw)
    B, I, M = dist2_n.shape[0], dist2_n.shape[1], dist2_n.shape[2]

    
    L = float(_safe_getattr(cfg, "L", 1.0) or 1.0)
    Rc = float(_safe_getattr(cfg, "Rc", 0.0) or 0.0)
    H = float(_safe_getattr(cfg, "H", 0.0) or 0.0)
    Rc2_eff = max(Rc * Rc - H * H, 0.0)
    dist2 = dist2_n * max(L * L, 1e-12)
    covered = dist2 <= (Rc2_eff + 1e-12)

    
    
    
    
    g_est = np.expm1(np.asarray(g_feat, dtype=np.float64))
    logg = np.log(np.maximum(g_est, 1e-30))

    
    alpha_dist = 1.50
    base_score = logg - alpha_dist * dist2_n

    
    alpha_f_a = 0.40
    if mode_n == "greedy_delay":
        s_a = base_score + 0.80 * fmax_n.reshape(B, 1, M)
    elif mode_n == "greedy_energy":
        s_a = base_score - 0.80 * fmax_n.reshape(B, 1, M)
    else:
        s_a = base_score + alpha_f_a * fmax_n.reshape(B, 1, M)
    s_a = np.where(covered, s_a, s_a - 5.0)

    
    
    
    
    
    if mode_n == "balanced":
        try:
            temp = float(_safe_getattr(cfg, "assoc_soft_temp", 0.5) or 0.5)
            temp = float(max(temp, 1e-6))
            cap = 0.50 + np.clip(np.asarray(fmax_n, dtype=np.float64), 0.0, 1.0)  # (B,M)
            beta_load = 0.80
            for _ in range(2):
                x = s_a / temp
                x = x - np.max(x, axis=2, keepdims=True)
                p = np.exp(x)
                p = p / (np.sum(p, axis=2, keepdims=True) + 1e-12)  # (B,I,M)
                load = np.sum(p, axis=1)  # (B,M)
                load_norm = np.clip(load / (cap + 1e-6), 0.0, 10.0)
                s_a = s_a - beta_load * load_norm.reshape(B, 1, M)
        except Exception:
            
            pass
    
    s_a = s_a - np.mean(s_a, axis=2, keepdims=True)

    action_score_mode = str(_safe_getattr(cfg, "action_score_mode", "separate")).strip().lower()
    use_separate_z = action_score_mode != "shared"

    parts = [s_a.reshape(B, I * M)]
    if use_separate_z:
        
        alpha_f_z = 0.60
        if mode_n == "greedy_delay":
            s_z = logg + 0.90 * fmax_n.reshape(B, 1, M)
        elif mode_n == "greedy_energy":
            s_z = logg - 0.90 * fmax_n.reshape(B, 1, M)
        else:
            s_z = logg + alpha_f_z * fmax_n.reshape(B, 1, M)
        s_z = np.where(covered, s_z, s_z - 5.0)
        s_z = s_z - np.mean(s_z, axis=2, keepdims=True)
        parts.append(s_z.reshape(B, I * M))

    theta_mode = str(_safe_getattr(cfg, "theta_mode", "solver")).strip().lower()
    if theta_mode == "policy":
        
        theta_raw = np.ones((B, I), dtype=np.float64)
        parts.append(theta_raw.reshape(B, I))

    scores = np.concatenate(parts, axis=1)

    
    s = float(max(score_tanh_scale, 1e-6))
    scores_act = np.tanh(scores / s).astype(np.float32)

    
    if np.asarray(obs_raw).ndim == 1:
        return scores_act.reshape(-1)
    return scores_act
