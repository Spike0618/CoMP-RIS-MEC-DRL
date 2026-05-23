"""
CoMP+RIS Joint env


- greedy_delay
- greedy_energy
- balanced
- myopic_optimization


-  CompRISEnvJoint API
-  env.step(action) 
    - full: [delta(2M), s_a(I*M), s_z(I*M), ()theta_raw(I)]
    - hierarchical: [delta(2M), comp_score(I)] position_only+LayerC[delta(2M), comp_meta(2)]
- Windows 


-  env.obs_flat() + env.cfg PPO 
-  env  helper  env 
"""

from __future__ import annotations

from typing import Literal
import numpy as np

from src.envs.comp_ris_env_joint import CompRISEnvJoint
from src.algos.tianshou.scores_fixed import compute_scores_fixed_action

BaselineMode = Literal[
    "greedy_delay",
    "greedy_energy",
    "balanced",
    "myopic_optimization",
    "always_comp",
    "never_comp",
]
_BASELINE_WARN_ONCE_KEYS: set[str] = set()


def _baseline_warn_once(key: str, message: str) -> None:
    """"""
    k = str(key)
    if k in _BASELINE_WARN_ONCE_KEYS:
        return
    _BASELINE_WARN_ONCE_KEYS.add(k)
    print(f"[HB][baseline][WARN][{k}] {message}", flush=True)


def _pairwise_dist2(w: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = w[:, None, :] - q[None, :, :]
    return np.sum(d * d, axis=-1)


def _clip_delta(delta: np.ndarray, vmax_dt: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    out = np.zeros_like(delta, dtype=np.float64)
    for m in range(delta.shape[0]):
        d = delta[m]
        n = float(np.linalg.norm(d))
        if n <= vmax_dt + 1e-12:
            out[m] = d
        else:
            out[m] = d / max(n, 1e-12) * vmax_dt
    return out


def _repulse(delta: np.ndarray, q: np.ndarray, dmin: float, strength: float = 0.30) -> np.ndarray:
    """"""
    M = q.shape[0]
    nud = np.zeros_like(delta, dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            v = q[i] - q[j]
            dist = float(np.linalg.norm(v))
            if dist < max(dmin, 1e-6):
                u = v / max(dist, 1e-6)
                push = (max(dmin - dist, 0.0) / max(dmin, 1e-6)) * strength
                nud[i] += u * push
                nud[j] -= u * push
    return delta + nud


def _scores_from_binary(mask: np.ndarray, hi: float = 10.0, lo: float = -10.0) -> np.ndarray:
    return np.where(mask > 0, hi, lo).astype(np.float64)


def _move_towards_users(env: CompRISEnvJoint, a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    
    - UAV z
    -  z  a

     env.obs_flat() + env.cfg env.q/env.w 
    """
    cfg = env.cfg
    obs = env.obs_flat()
    qn, wn, *_ = _decode_obs_flat(cfg, obs)  
    q = np.asarray(qn, dtype=np.float64) * float(cfg.L)  
    w = np.asarray(wn, dtype=np.float64) * float(cfg.L)  

    M = q.shape[0]
    targets = q.copy()

    for m in range(M):
        idx = np.where(z[:, m] == 1)[0]
        if idx.size == 0:
            idx = np.where(a[:, m] == 1)[0]
        if idx.size > 0:
            targets[m] = np.mean(w[idx], axis=0)

    delta = targets - q
    delta = _repulse(delta, q, float(cfg.dmin))
    delta = _clip_delta(delta, float(cfg.Vmax) * float(cfg.dt))
    
    wall_safe = float(cfg.Vmax) * float(cfg.dt)
    delta = _clip_delta_to_bounds(delta, q, float(cfg.L), margin=wall_safe)
    return delta

def _decode_obs_flat(cfg, obs: np.ndarray):
    """
     CompRISEnvJoint.obs_flat() 

     src/envs/comp_ris_env_joint.py  obs_dim/obs_flat 
      [ q(2M),
        w(2I),
        dist2_n(I*M),
        g_feat(I*M),
        Dn(I),
        Cn(I),
        fmax_n(M),
        enable_comp, enable_ris,
        t_norm, load_scale,
        dist_to_walls(M*4), min_wall_dist(M),
        qos_summary(5),
        comp_gain_ratio(I),   # Phase T2 reviewper-user CoMP
        ...() ]
    """
    M, I = int(cfg.M), int(cfg.I)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    
    need = 2 * M + 2 * I + (I * M) + (I * M) + I + I + M + 2 + 2
    if obs.size < need:
        raise ValueError(f"obs_flat too short: got {obs.size}, need >= {need}")

    p = 0
    q = obs[p:p + 2 * M].reshape(M, 2); p += 2 * M
    w = obs[p:p + 2 * I].reshape(I, 2); p += 2 * I

    dist2_n = obs[p:p + I * M].reshape(I, M); p += I * M
    g_feat  = obs[p:p + I * M].reshape(I, M); p += I * M

    Dn = obs[p:p + I]; p += I
    Cn = obs[p:p + I]; p += I

    
    p += M
    
    p += 2

    t_norm = float(obs[p]); load_scale = float(obs[p + 1]); p += 2

    

    
    L2 = float(cfg.L) * float(cfg.L)
    dist2 = dist2_n * max(L2, 1e-12)
    return q, w, dist2_n, dist2, g_feat, Dn, Cn, t_norm, load_scale


def _safe_row_minmax_norm(x: np.ndarray) -> np.ndarray:
    """ min-max """
    arr = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    if arr.ndim != 2:
        return out
    for i in range(arr.shape[0]):
        row = arr[i]
        lo = float(np.min(row))
        hi = float(np.max(row))
        if hi - lo <= 1e-12:
            out[i] = 0.5
        else:
            out[i] = (row - lo) / (hi - lo)
    return out


def _safe_01_norm(x: np.ndarray) -> np.ndarray:
    """[0,1]"""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _mode_motion_profile(mode: str) -> dict:
    """ position_only """
    m = str(mode).strip().lower()
    if m == "greedy_delay":
        return {
            "assign_dist_w": 0.40,
            "assign_score_w": 0.60,
            "step_gain": 1.00,
            "repulse_strength": 0.10,
        }
    if m == "greedy_energy":
        return {
            "assign_dist_w": 0.92,
            "assign_score_w": 0.08,
            "step_gain": 0.45,
            "repulse_strength": 0.35,
        }
    if m == "myopic_optimization":
        return {
            "assign_dist_w": 0.58,
            "assign_score_w": 0.42,
            "step_gain": 0.90,
            "repulse_strength": 0.16,
        }
    return {
        "assign_dist_w": 0.72,
        "assign_score_w": 0.28,
        "step_gain": 0.82,
        "repulse_strength": 0.20,
    }


def _build_mode_user_weights(mode: str, Dn: np.ndarray, Cn: np.ndarray, dist2_n: np.ndarray) -> np.ndarray:
    """
    
    - delay /
    - energy 
    - balanced 
    """
    dn = _safe_01_norm(Dn)
    cn = _safe_01_norm(Cn)
    hard = _safe_01_norm(np.min(np.asarray(dist2_n, dtype=np.float64), axis=1))
    m = str(mode).strip().lower()
    if m == "greedy_delay":
        w = 1.0 + 1.50 * dn + 1.10 * cn + 0.70 * hard
    elif m == "greedy_energy":
        w = 1.0 + 0.20 * dn + 0.15 * cn + 0.10 * hard
    elif m == "myopic_optimization":
        
        w = 1.0 + 1.05 * dn + 0.95 * cn + 0.45 * hard
    else:
        w = 1.0 + 0.80 * dn + 0.60 * cn + 0.35 * hard
    return np.clip(w, 0.10, 5.00)


def _apply_myopic_score_adjustment(
    s_a: np.ndarray,
    s_z: np.ndarray,
    covered: np.ndarray,
    Dn: np.ndarray,
    Cn: np.ndarray,
    dist2_n: np.ndarray,
    g_feat: np.ndarray,
    load_scale: float,
    t_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
     fixed-scores 
    - Dn/Cn/
    - rollout
    """
    s_a0 = np.asarray(s_a, dtype=np.float64)
    s_z0 = np.asarray(s_z, dtype=np.float64)
    cov = np.asarray(covered, dtype=bool)
    dn = _safe_01_norm(Dn).reshape(-1, 1)
    cn = _safe_01_norm(Cn).reshape(-1, 1)
    dist_rank = _safe_row_minmax_norm(-np.asarray(dist2_n, dtype=np.float64))
    gain_rank = _safe_row_minmax_norm(np.asarray(g_feat, dtype=np.float64))

    
    load_w = float(np.clip(0.70 + 0.40 * float(load_scale), 0.70, 1.25))
    time_w = float(np.clip(1.05 - 0.20 * float(t_norm), 0.80, 1.05))
    pressure = 0.55 * dn + 0.45 * cn
    myopic_core = load_w * time_w * (0.48 * gain_rank + 0.34 * dist_rank + 0.18 * pressure)

    s_a1 = s_a0 + 1.25 * myopic_core
    s_z1 = s_z0 + 1.40 * myopic_core

    
    s_a1 = np.where(cov, s_a1, s_a1 - 6.0)
    s_z1 = np.where(cov, s_z1, s_z1 - 6.0)
    return s_a1, s_z1


def baseline_action(env, mode: str = "balanced") -> np.ndarray:
    """
     baseline
    -  env.obs_flat() + env.cfg
    -  env.step(action)  raw action 
      - action_score_mode='separate' [delta(2M), s_a(I*M), s_z(I*M), ()theta_raw(I)]
      - action_score_mode='shared'   [delta(2M), s_a(I*M),           ()theta_raw(I)]
      - action_space_mode='hierarchical' [delta(2M), comp_score(I)]
      - theta_mode='solver' theta_raw
    -  env / env  PPO 
    """
    
    global _BASELINE_BUDGET_LOGGED
    try:
        _BASELINE_BUDGET_LOGGED
    except NameError:
        _BASELINE_BUDGET_LOGGED = True
        print(
            f"[FAIRNESS] baseline mode={mode} rolloutO(M*I)",
            flush=True
        )
    mode = str(mode).strip().lower()
    allowed_modes = {
        "balanced",
        "greedy_delay",
        "greedy_energy",
        "myopic_optimization",
        "always_comp",
        "never_comp",
    }
    if mode not in allowed_modes:
        raise ValueError(f"Unknown baseline mode: {mode}. Allowed: {sorted(allowed_modes)}")

    cfg = env.cfg
    M, I = int(cfg.M), int(cfg.I)

    obs = env.obs_flat()
    qn, wn, dist2_n, dist2, g_feat, Dn, Cn, t_norm, load_scale = _decode_obs_flat(cfg, obs)

    
    Rc2_eff = float(cfg.Rc) * float(cfg.Rc) - float(cfg.H) * float(cfg.H)
    covered = dist2 <= max(Rc2_eff, 0.0) + 1e-12  # (I,M)

    action_score_mode = str(getattr(cfg, "action_score_mode", "separate")).strip().lower()
    action_space_mode = str(getattr(cfg, "action_space_mode", "full")).strip().lower()
    theta_mode = str(getattr(cfg, "theta_mode", "solver")).strip().lower()
    use_separate_z = action_score_mode != "shared"
    use_theta = theta_mode == "policy"

    
    
    
    if action_space_mode != "hierarchical":
        k_cfg = int(max(int(getattr(cfg, "K", 1)), 1))
        comp_on = bool(getattr(cfg, "enable_comp", True))
        if mode == "never_comp" and comp_on and k_cfg > 1:
            raise RuntimeError(
                "never_comp  full  enable_comp=1  K>1"
                " env.cfg.enable_comp=0  K=1"
            )
        if mode == "always_comp" and comp_on and k_cfg < 2:
            raise RuntimeError(
                "always_comp  full  K>=2"
                f" K={k_cfg}CoMP"
            )

    
    
    
    
    score_tail = None
    
    
    score_mode = mode if mode in ("balanced", "greedy_delay", "greedy_energy") else "balanced"
    allow_legacy_fallback = bool(getattr(cfg, "baseline_allow_legacy_score_fallback", False))
    try:
        score_tail = compute_scores_fixed_action(
            cfg=cfg,
            obs_raw=np.asarray(obs, dtype=np.float32),
            mode=str(score_mode),
            score_tanh_scale=float(getattr(cfg, "scores_fixed_tanh_scale", 2.5)),
        )
    except Exception as exc:
        if not allow_legacy_fallback:
            raise RuntimeError(
                f"baseline fixed-scores mode={score_mode}"
                f"{type(exc).__name__}: {exc}"
            ) from exc
        _baseline_warn_once(
            "baseline_scores_legacy_fallback",
            "fixed-scores "
            "",
        )
        score_tail = None

    if score_tail is not None:
        st = np.asarray(score_tail, dtype=np.float64).reshape(-1)
        p = 0
        if st.size >= p + I * M:
            s_a = st[p:p + I * M].reshape(I, M)
            p += I * M
        else:
            s_a = np.zeros((I, M), dtype=np.float64)
        if use_separate_z:
            if st.size >= p + I * M:
                s_z = st[p:p + I * M].reshape(I, M)
                p += I * M
            else:
                s_z = s_a.copy()
        else:
            s_z = s_a
        if use_theta and st.size >= p + I:
            theta_raw_fixed = np.asarray(st[p:p + I], dtype=np.float64).reshape(I,)
        else:
            theta_raw_fixed = np.ones((I,), dtype=np.float64)

        if mode == "myopic_optimization":
            s_a, s_z = _apply_myopic_score_adjustment(
                s_a=s_a,
                s_z=s_z,
                covered=covered,
                Dn=Dn,
                Cn=Cn,
                dist2_n=dist2_n,
                g_feat=g_feat,
                load_scale=load_scale,
                t_norm=t_norm,
            )
    else:
        
        base_score = g_feat - 0.1 * dist2_n
        fmax = np.asarray(cfg.fmax, dtype=np.float64).reshape(M,)
        fmin, fmaxv = float(np.min(fmax)), float(np.max(fmax))
        f_norm = (fmax - fmin) / max(fmaxv - fmin, 1e-12)
        alpha_f = 0.6
        if mode == "greedy_delay":
            s_a = base_score + alpha_f * f_norm[None, :]
        elif mode == "greedy_energy":
            s_a = base_score - alpha_f * f_norm[None, :]
        else:
            s_a = base_score
        s_a = np.where(covered, s_a, s_a - 5.0)
        if mode == "greedy_delay":
            s_z = g_feat + 0.8 * f_norm[None, :]
        elif mode == "greedy_energy":
            s_z = g_feat - 0.8 * f_norm[None, :]
        else:
            s_z = g_feat
        s_z = np.where(covered, s_z, s_z - 5.0)
        theta_raw_fixed = np.ones((I,), dtype=np.float64)

    
    vmax_dt = float(cfg.Vmax) * float(cfg.dt)
    q = qn * float(cfg.L)   
    w = wn * float(cfg.L)
    move_mode = mode if mode in ("greedy_delay", "balanced", "greedy_energy", "myopic_optimization") else "balanced"
    profile = _mode_motion_profile(move_mode)
    user_weights = _build_mode_user_weights(move_mode, Dn, Cn, dist2_n)
    s_a_row_n = _safe_row_minmax_norm(s_a)
    
    try:
        fmax = np.asarray(getattr(cfg, "fmax", np.ones((M,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    except Exception:
        fmax = np.ones((M,), dtype=np.float64)
    if fmax.size < M:
        fmax = np.pad(fmax, (0, M - fmax.size), mode="edge")
    fmax = np.asarray(fmax[:M], dtype=np.float64)
    f_lo = float(np.min(fmax))
    f_hi = float(np.max(fmax))
    if f_hi - f_lo <= 1e-12:
        f_norm = np.zeros((M,), dtype=np.float64)
    else:
        f_norm = (fmax - f_lo) / (f_hi - f_lo)
    if move_mode == "greedy_delay":
        uav_bias = -0.20 * f_norm
    elif move_mode == "greedy_energy":
        uav_bias = 0.20 * f_norm
    elif move_mode == "myopic_optimization":
        
        uav_bias = -0.12 * f_norm
    else:
        uav_bias = np.zeros((M,), dtype=np.float64)
    assign_metric = (
        float(profile["assign_dist_w"]) * np.asarray(dist2_n, dtype=np.float64)
        - float(profile["assign_score_w"]) * np.asarray(s_a_row_n, dtype=np.float64)
        + np.asarray(uav_bias, dtype=np.float64)[None, :]
    )

    covered_users = np.any(covered, axis=1)  # (I,)
    uncovered_idx = np.where(~covered_users)[0]
    targets = q.copy()

    if uncovered_idx.size > 0:
        pairs = []
        for i in uncovered_idx.tolist():
            for m in range(M):
                pairs.append((float(assign_metric[i, m]), int(m), int(i)))
        pairs.sort(key=lambda x: x[0])

        assigned_uavs = set()
        assigned_users = set()
        assign = {m: [] for m in range(M)}

        need = min(int(uncovered_idx.size), M)
        for _d2, m, i in pairs:
            if (m in assigned_uavs) or (i in assigned_users):
                continue
            assigned_uavs.add(m)
            assigned_users.add(i)
            assign[m].append(i)
            if len(assigned_users) >= need:
                break

        for i in uncovered_idx.tolist():
            if int(i) in assigned_users:
                continue
            m_star = int(np.argmin(assign_metric[int(i)]))
            assign[m_star].append(int(i))

        for m, users in assign.items():
            if not users:
                continue
            idx = np.asarray(users, dtype=np.int64)
            ww = np.asarray(user_weights[idx], dtype=np.float64)
            denom = float(np.sum(ww))
            if denom <= 1e-12:
                targets[m] = np.mean(w[idx], axis=0)
            else:
                targets[m] = np.sum(w[idx] * ww[:, None], axis=0) / denom
    else:
        owner = np.argmin(assign_metric, axis=1)  # (I,)
        for m in range(M):
            idx = np.where(owner == m)[0]
            if idx.size == 0:
                continue
            ww = np.asarray(user_weights[idx], dtype=np.float64)
            denom = float(np.sum(ww))
            if denom <= 1e-12:
                targets[m] = np.mean(w[idx], axis=0)
            else:
                targets[m] = np.sum(w[idx] * ww[:, None], axis=0) / denom

    
    if move_mode == "greedy_delay":
        targets = 0.00 * q + 1.00 * targets
    elif move_mode == "greedy_energy":
        targets = 0.80 * q + 0.20 * targets
    elif move_mode == "myopic_optimization":
        targets = 0.20 * q + 0.80 * targets

    delta = (targets - q) * float(profile["step_gain"])
    delta = _repulse(delta, q, float(cfg.dmin), strength=float(profile["repulse_strength"]))
    delta = _clip_delta(delta, vmax_dt)
    
    
    
    
    wall_safe = float(cfg.Vmax) * float(cfg.dt)
    delta = _clip_delta_to_bounds(delta, q, float(cfg.L), margin=wall_safe)

    
    
    delta_scale = max(float(cfg.Vmax) * float(cfg.dt), 1e-9)
    delta_q = np.clip(delta / delta_scale, -1.0, 1.0)

    if action_space_mode == "hierarchical":
        
        z4_action_mode = str(getattr(cfg, "z4_action_mode", "hierarchical")).strip().lower()
        if z4_action_mode == "position_only":
            
            
            if bool(getattr(cfg, "z4_comp_meta_enable", False)):
                comp_meta = np.zeros((2,), dtype=np.float64)
                return np.concatenate([delta_q.reshape(2 * M), comp_meta.reshape(2)], axis=0).astype(np.float32)
            return delta_q.reshape(2 * M).astype(np.float32)
        
        
        if mode == "always_comp":
            comp_score = np.ones((I,), dtype=np.float64)
            return np.concatenate([delta_q.reshape(2 * M), comp_score.reshape(I)], axis=0).astype(np.float32)
        if mode == "never_comp":
            comp_score = -np.ones((I,), dtype=np.float64)
            return np.concatenate([delta_q.reshape(2 * M), comp_score.reshape(I)], axis=0).astype(np.float32)
        mode_bias = {
            "greedy_delay": 0.55,
            "balanced": 0.15,
            "greedy_energy": -0.35,
            "myopic_optimization": 0.32,
        }.get(mode, 0.0)
        comp_score = np.full((I,), -1.0, dtype=np.float64)
        for i in range(I):
            cand = np.where(covered[i])[0]
            if cand.size < 2:
                comp_score[i] = -1.0
                continue
            s_row = np.asarray(s_a[i, cand], dtype=np.float64).reshape(-1)
            ord_i = np.argsort(s_row)[::-1]
            s1 = float(s_row[ord_i[0]])
            s2 = float(s_row[ord_i[1]])
            raw = float((s2 - s1) * 1.5 + mode_bias)
            comp_score[i] = float(np.clip(raw, -1.0, 1.0))
        return np.concatenate([delta_q.reshape(2 * M), comp_score.reshape(I)], axis=0).astype(np.float32)

    parts = [delta_q.reshape(2 * M), s_a.reshape(I * M)]
    if use_separate_z:
        parts.append(s_z.reshape(I * M))
    if use_theta:
        
        theta_raw = np.asarray(theta_raw_fixed, dtype=np.float64).reshape(I,)
        parts.append(theta_raw.reshape(I))

    out = np.concatenate(parts, axis=0).astype(np.float32)
    return out


def _clip_delta_to_bounds(delta: np.ndarray, q: np.ndarray, L: float, margin: float = 0.0) -> np.ndarray:
    """
     q + delta  [margin, L-margin]^2

    
    - baseline(q, L)
    - /envA(vio_wall)
    """
    delta = np.asarray(delta, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    L = float(L)
    margin = float(max(margin, 0.0))

    low = margin - q
    high = (L - margin) - q
    delta[:, 0] = np.clip(delta[:, 0], low[:, 0], high[:, 0])
    delta[:, 1] = np.clip(delta[:, 1], low[:, 1], high[:, 1])
    return delta
