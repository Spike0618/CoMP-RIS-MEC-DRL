from __future__ import annotations

"""
/


- CoMP SNR 
- //
"""

import numpy as np


def gain_direct(q_m: np.ndarray, w_i: np.ndarray, H: float, beta0: float) -> float:
    """
    
    beta_d = beta0 / (H^2 + ||q-w||^2)
    """
    d2 = float(np.sum((q_m - w_i) ** 2))
    return float(beta0 / (H * H + d2 + 1e-12))


def gain_user_ris(w_i: np.ndarray, v: np.ndarray, beta0: float) -> float:
    """
     -> RIS 
    beta_ir = beta0 / ||w-v||^2
    """
    d2 = float(np.sum((w_i - v) ** 2))
    return float(beta0 / (d2 + 1e-12))


def gain_ris_uav(q_m: np.ndarray, v: np.ndarray, H: float, beta0: float) -> float:
    """
    RIS -> UAV 
    beta_rm = beta0 / (H^2 + ||q-v||^2)
    """
    d2 = float(np.sum((q_m - v) ** 2))
    return float(beta0 / (H * H + d2 + 1e-12))


def comp_gain(beta_d: float, beta_ir: float, beta_rm: float, N: int, eta: float) -> float:
    """
    RIS  + 
    g = ( sqrt(beta_d) + N * sqrt(eta * beta_ir * beta_rm) )^2
    """
    term = (np.sqrt(beta_d) + float(N) * np.sqrt(max(eta, 0.0) * beta_ir * beta_rm))
    return float(term * term)


def snr_from_comp(
    a_row: np.ndarray,
    p: float,
    g_vec: np.ndarray,
    N0: float,
    B: float,
    *,
    coherent: bool = True,
    coherence_boost: float = 1.0,
    power_mode: str = "per_uav",   # NEW: "per_uav" | "total"
) -> float:
    """
     CoMP  SNR

    
    - power_mode="per_uav"p  UAV  |S| 
    - power_mode="total"p  S 

    coherent
    - Truecoherence_boost 
    - False

    SNR
    """
    idx = np.flatnonzero(a_row > 0.0)
    if idx.size == 0:
        return 0.0

    denom = float(N0) * float(B)
    if denom <= 0.0:
        denom = 1e-12

    g_sel = np.asarray(g_vec, dtype=np.float64)[idx]
    
    
    
    w_sel = np.asarray(a_row, dtype=np.float64)[idx]
    w_sel = np.clip(w_sel, 0.0, None)
    if float(np.sum(w_sel)) <= 1e-12:
        w_sel = np.ones_like(g_sel, dtype=np.float64)
    S = int(idx.size)

    pmode = str(power_mode).strip().lower()
    if pmode not in ("per_uav", "total"):
        pmode = "per_uav"

    
    if pmode == "total":
        
        p_each = float(p) * (w_sel / float(np.sum(w_sel) + 1e-12))
    else:
        
        p_each = float(p) * w_sel

    
    if (not coherent) or (S == 1):
        return float(np.sum(np.asarray(p_each, dtype=np.float64) * g_sel) / denom)

    
    amps = np.sqrt(np.maximum(p_each * g_sel, 0.0))  
    sum_amp = float(np.sum(amps))
    sum_sq = float(np.sum(amps * amps))
    cross = max(sum_amp * sum_amp - sum_sq, 0.0)     

    boost = float(coherence_boost)
    if boost < 0.0:
        boost = 0.0

    num = sum_sq + boost * cross
    return float(num / denom)


def rate_from_snr(B: float, gamma: float) -> float:
    """
    R = B * log2(1 + gamma)
    """
    return float(B * np.log2(1.0 + max(gamma, 0.0)))
