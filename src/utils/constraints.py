from __future__ import annotations

from typing import Optional

import numpy as np


def project_speed(q_new: np.ndarray, q_prev: np.ndarray, Vmax: float, dt: float) -> np.ndarray:
    """
     UAV  ||q_new - q_prev|| <= Vmax * dt
    """
    q_new = np.asarray(q_new, dtype=np.float64)
    q_prev = np.asarray(q_prev, dtype=np.float64)
    max_step = float(Vmax) * float(dt)
    if max_step <= 0:
        return q_prev.copy()

    d = q_new - q_prev
    norm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    scale = np.minimum(1.0, max_step / norm)
    return q_prev + d * scale


def repair_collisions(
    q: np.ndarray,
    dmin: float,
    max_iter: int = 50,
    step: float = 0.5,
    rng: Optional[object] = None,
) -> tuple[np.ndarray, int]:
    """
     dmin  UAV 

    
        (q_repaired, num_pushes)
    """
    q = np.asarray(q, dtype=np.float64).copy()
    M = q.shape[0]
    dmin2 = float(dmin) ** 2
    pushes = 0

    
    
    
    rng_local = rng if rng is not None else np.random

    def _sample_unit_direction(rng_obj: object) -> np.ndarray:
        """
        
         np.random  seed 
        """
        vec: Optional[np.ndarray] = None

        if hasattr(rng_obj, "standard_normal"):
            try:
                vec = np.asarray(getattr(rng_obj, "standard_normal")(2), dtype=np.float64).reshape(-1)
            except Exception:
                vec = None
        if vec is None and hasattr(rng_obj, "normal"):
            fn = getattr(rng_obj, "normal")
            try:
                vec = np.asarray(fn(loc=0.0, scale=1.0, size=2), dtype=np.float64).reshape(-1)
            except Exception:
                try:
                    vec = np.asarray(fn(0.0, 1.0, 2), dtype=np.float64).reshape(-1)
                except Exception:
                    vec = None
        if vec is None and hasattr(rng_obj, "randn"):
            try:
                vec = np.asarray(getattr(rng_obj, "randn")(2), dtype=np.float64).reshape(-1)
            except Exception:
                vec = None
        if vec is None and hasattr(rng_obj, "random"):
            fn = getattr(rng_obj, "random")
            try:
                u = float(np.asarray(fn(), dtype=np.float64).reshape(-1)[0])
                ang = 2.0 * np.pi * u
                vec = np.asarray([np.cos(ang), np.sin(ang)], dtype=np.float64)
            except Exception:
                vec = None
        if vec is None and hasattr(rng_obj, "rand"):
            try:
                u = float(np.asarray(getattr(rng_obj, "rand")(), dtype=np.float64).reshape(-1)[0])
                ang = 2.0 * np.pi * u
                vec = np.asarray([np.cos(ang), np.sin(ang)], dtype=np.float64)
            except Exception:
                vec = None

        
        if vec is None or vec.size < 2 or (not np.all(np.isfinite(vec[:2]))):
            return np.asarray([1.0, 0.0], dtype=np.float64)
        vec2 = np.asarray(vec[:2], dtype=np.float64)
        return vec2 / (np.linalg.norm(vec2) + 1e-12)

    def _pair_unit_direction(i: int, j: int) -> np.ndarray:
        """
        (i,j)
        """
        code = (int(i + 1) * 73856093) ^ (int(j + 1) * 19349663)
        u = float((code % 3600) / 3600.0)
        ang = 2.0 * np.pi * u
        return np.asarray([np.cos(ang), np.sin(ang)], dtype=np.float64)

    def _num_colliding_pairs(x: np.ndarray) -> int:
        """
        
        """
        cnt = 0
        for ii in range(M):
            for jj in range(ii + 1, M):
                d = x[ii] - x[jj]
                if float(np.dot(d, d)) < dmin2:
                    cnt += 1
        return int(cnt)

    max_iter_eff = int(max(int(max_iter), 1))
    step_base = float(np.clip(float(step), 0.05, 1.0))

    for it in range(max_iter_eff):
        moved = False
        
        step_now = float(step_base + (1.0 - step_base) * (float(it) / float(max(max_iter_eff - 1, 1))))
        for i in range(M):
            for j in range(i + 1, M):
                diff = q[i] - q[j]
                dist2 = float(np.dot(diff, diff))
                if dist2 < dmin2:
                    dist = np.sqrt(dist2 + 1e-12)
                    
                    if dist <= 1e-6:
                        direction = _sample_unit_direction(rng_local)
                    else:
                        direction = diff / dist
                    push = (float(dmin) - dist) * float(step_now) * direction
                    if (not np.all(np.isfinite(push))) or (np.linalg.norm(push) <= 1e-15):
                        continue
                    q[i] = q[i] + push
                    q[j] = q[j] - push
                    pushes += 1
                    moved = True
        if not moved:
            break

    
    
    if _num_colliding_pairs(q) > 0:
        extra_iters = int(max(10, min(200, M * M * 4)))
        for _ in range(extra_iters):
            moved = False
            for i in range(M):
                for j in range(i + 1, M):
                    diff = q[i] - q[j]
                    dist2 = float(np.dot(diff, diff))
                    if dist2 >= dmin2:
                        continue
                    dist = np.sqrt(dist2 + 1e-12)
                    if dist <= 1e-6:
                        direction = _pair_unit_direction(i, j)
                    else:
                        direction = diff / dist
                    
                    push = 0.5 * (float(dmin) - dist) * direction
                    if (not np.all(np.isfinite(push))) or (np.linalg.norm(push) <= 1e-15):
                        continue
                    q[i] = q[i] + push
                    q[j] = q[j] - push
                    pushes += 1
                    moved = True
            if (not moved) or (_num_colliding_pairs(q) <= 0):
                break

    return q, pushes


def ensure_theta(theta: np.ndarray, theta_min: float) -> np.ndarray:
    """
     theta 
    -  theta >= theta_min  sum(theta)=1
    - n*theta_min >= 1
    """
    theta = np.asarray(theta, dtype=np.float64).copy()
    n = theta.size
    if n == 0:
        return theta

    theta_min = float(theta_min)
    if theta_min <= 0:
        s = theta.sum()
        if s <= 0:
            return np.full((n,), 1.0 / n, dtype=np.float64)
        return theta / s

    if n * theta_min >= 1.0 - 1e-12:
        return np.full((n,), 1.0 / n, dtype=np.float64)

    theta = np.maximum(theta, 0.0)
    s = theta.sum()
    if s <= 0:
        theta = np.full((n,), 1.0 / n, dtype=np.float64)
    else:
        theta = theta / s

    theta = np.maximum(theta, theta_min)
    surplus = theta.sum() - 1.0
    if surplus <= 1e-12:
        return theta / (theta.sum() + 1e-12)

    above = theta > theta_min + 1e-12
    if not np.any(above):
        return theta / (theta.sum() + 1e-12)

    reducible = theta[above] - theta_min
    reducible_sum = reducible.sum()
    if reducible_sum <= 1e-12:
        return theta / (theta.sum() + 1e-12)

    theta[above] = theta[above] - surplus * (reducible / reducible_sum)
    theta = np.maximum(theta, theta_min)
    theta = theta / (theta.sum() + 1e-12)
    return theta


def ensure_theta_feasible(theta: np.ndarray, theta_min: float) -> tuple[np.ndarray, bool]:
    """
     (theta_fixed, feasible_flag) feasible_flag  n*theta_min <= 1
    """
    theta = np.asarray(theta, dtype=np.float64)
    n = theta.size
    feasible = (n * float(theta_min) <= 1.0 + 1e-12)
    return ensure_theta(theta, theta_min), bool(feasible)


def alloc_theta_solver(
    C_cycles: np.ndarray,
    fmax: float,
    w_delay: float,
    w_energy: float,
    T_scale: float,
    E_scale: float,
    xi: float,
    theta_min: float,
) -> tuple[np.ndarray, bool]:
    """
     theta  + 

     C_cycles theta 
     ensure_theta 

    
    1) 
    2) 
       sum_i [ A_i / theta_i + B_i * theta_i^2 ]s.t. theta_i>=theta_min, sum theta_i=1
        A_i B_i 
    """
    C = np.asarray(C_cycles, dtype=np.float64)
    n = C.size
    if n == 0:
        return np.zeros((0,), dtype=np.float64), True

    theta_min = float(theta_min)
    feasible = (n * theta_min <= 1.0 + 1e-12)
    if not feasible:
        return np.full((n,), 1.0 / n, dtype=np.float64), False
    if n == 1:
        return np.ones((1,), dtype=np.float64), True

    C_pos = np.maximum(C, 1e-12)
    f_eff = max(float(fmax), 1e-12)
    a_delay = max(float(w_delay) / max(float(T_scale), 1e-12), 0.0)
    b_energy = max(float(w_energy) / max(float(E_scale), 1e-12), 0.0)

    
    A = a_delay * (C_pos / f_eff)
    B = b_energy * float(xi) * (f_eff ** 2) * C_pos

    
    a0 = max(a_delay, 1e-6)
    b0 = max(b_energy * float(xi) * (f_eff ** 2), 1e-6)
    score = np.sqrt(C_pos) / np.sqrt(a0 + b0 * C_pos)
    score = np.maximum(score, 1e-12)
    theta0 = ensure_theta(score / score.sum(), theta_min)

    def _obj(th: np.ndarray) -> float:
        th_safe = np.maximum(np.asarray(th, dtype=np.float64), 1e-12)
        val = float(np.sum(A / th_safe + B * (th_safe ** 2)))
        return val if np.isfinite(val) else float("inf")

    
    slack = float(max(1.0 - n * theta_min, 0.0))
    if slack <= 1e-12:
        return np.full((n,), 1.0 / n, dtype=np.float64), True

    
    if float(np.max(np.abs(A)) + np.max(np.abs(B))) <= 1e-18:
        return theta0, True

    y = np.maximum((theta0 - theta_min) / slack, 1e-12)
    y = y / (np.sum(y) + 1e-12)
    best_theta = theta0.copy()
    best_obj = _obj(best_theta)

    lr = 0.15
    for _ in range(24):
        theta_cur = theta_min + slack * y
        theta_cur = ensure_theta(theta_cur, theta_min)
        grad = -A / np.maximum(theta_cur ** 2, 1e-12) + 2.0 * B * theta_cur
        
        grad = grad - float(np.mean(grad))
        grad = np.clip(grad, -50.0, 50.0)

        y_new = y * np.exp(-lr * grad)
        s = float(np.sum(y_new))
        if (not np.isfinite(s)) or s <= 1e-12:
            lr *= 0.5
            if lr < 1e-4:
                break
            continue
        y_new = y_new / s
        theta_new = ensure_theta(theta_min + slack * y_new, theta_min)
        obj_new = _obj(theta_new)

        
        if obj_new <= best_obj + 1e-12:
            best_obj = obj_new
            best_theta = theta_new
            y = y_new
        else:
            lr *= 0.5
            if lr < 1e-4:
                break

    return best_theta, True


def ensure_topk_feasible(
    scores: np.ndarray,
    covered_mask: np.ndarray,
    K: int,
    dist2_row: np.ndarray,
    g_row: Optional[np.ndarray] = None,
    gain_mix: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """
     UAV  Top-K
    -  scores
    - scores + gain_mix * log1p(g)g  g_row
    -  UAV  UAVargmin dist2_row

    
        (chosen_indices, had_coverage)
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    covered_mask = np.asarray(covered_mask, dtype=bool).reshape(-1)
    dist2_row = np.asarray(dist2_row, dtype=np.float64).reshape(-1)
    K = int(max(1, K))

    cand = np.where(covered_mask)[0]
    if cand.size == 0:
        nearest = int(np.argmin(dist2_row))
        return np.array([nearest], dtype=np.int64), False

    s = scores[cand].copy()
    if g_row is not None and float(gain_mix) != 0.0:
        g_row = np.asarray(g_row, dtype=np.float64).reshape(-1)
        s = s + float(gain_mix) * np.log1p(np.maximum(g_row[cand], 0.0))

    order = np.argsort(s)[::-1]
    chosen = cand[order[: min(K, cand.size)]]
    return chosen.astype(np.int64), True


def enforce_min_coverage(
    a_mat: np.ndarray,
    z_mat: np.ndarray,
    covered: np.ndarray,
    dist2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    
    -  i  a_mat  1  UAV
    -  i  z_mat  1  UAV
    -  UAVargmin dist2[i]
    -  UAV z  1  a  1 
    """
    a = np.asarray(a_mat, dtype=np.int64).copy()
    z = np.asarray(z_mat, dtype=np.int64).copy()
    covered = np.asarray(covered, dtype=bool)
    dist2 = np.asarray(dist2, dtype=np.float64)

    I, M = a.shape
    for i in range(I):
        if a[i].sum() <= 0:
            cand = np.where(covered[i])[0]
            if cand.size == 0:
                m = int(np.argmin(dist2[i]))
                a[i, m] = 1
            else:
                a[i, int(cand[0])] = 1

        if z[i].sum() != 1:
            z[i, :] = 0
            sel = np.where(a[i] == 1)[0]
            m = int(sel[0]) if sel.size > 0 else int(np.argmin(dist2[i]))
            z[i, m] = 1

        m_star = int(np.argmax(z[i]))
        if a[i, m_star] != 1:
            a[i, m_star] = 1

    return a, z
