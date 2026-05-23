from __future__ import annotations

import numpy as np


def clamp_positions(xy: np.ndarray, L: float) -> np.ndarray:
    """
     [0, L] x [0, L] 

    
        xy (..., 2) 
        L

    
         xy 
    """
    xy = np.asarray(xy, dtype=np.float64)
    out = xy.copy()
    out[..., 0] = np.clip(out[..., 0], 0.0, float(L))
    out[..., 1] = np.clip(out[..., 1], 0.0, float(L))
    return out


def pairwise_dist2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    

    
        A (NA, 2)
        B (NB, 2)

    
        dist2 (NA, NB)dist2[i, j] = ||A[i] - B[j]||^2
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if not (A.ndim == 2 and A.shape[1] == 2):
        raise ValueError(f"A  (NA,2) {A.shape}")
    if not (B.ndim == 2 and B.shape[1] == 2):
        raise ValueError(f"B  (NB,2) {B.shape}")
    diff = A[:, None, :] - B[None, :, :]
    return np.sum(diff * diff, axis=-1)


def clip_to_square(xy: np.ndarray, L: float) -> np.ndarray:
    """
    

     `clip_to_square` ImportError
    """
    return clamp_positions(xy, L)


