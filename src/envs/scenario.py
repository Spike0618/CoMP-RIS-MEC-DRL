from __future__ import annotations

"""
 UAV 


- 
- 
"""

import numpy as np


def init_positions(
    rng: np.random.RandomState,
    M: int,
    I: int,
    L: float,
    *,
    uav_init_mode: str = "random",
    uav_fixed_positions=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
     UAV 

    
    - uav_init_mode: "random" "fixed"
    - uav_fixed_positions: uav_init_mode="fixed"  (M,2) 
    """
    
    if uav_init_mode == "fixed" and uav_fixed_positions is not None:
        q0 = np.array(uav_fixed_positions, dtype=np.float64).reshape(M, 2)
    else:
        q0 = rng.uniform(0.0, L, size=(M, 2)).astype(np.float64)
    
    w = rng.uniform(0.0, L, size=(I, 2)).astype(np.float64)
    return q0, w


def sample_uavs_uniform(rng: np.random.RandomState, M: int, L: float) -> np.ndarray:
    return rng.uniform(0.0, L, size=(M, 2)).astype(np.float64)


def sample_users_uniform(rng: np.random.RandomState, I: int, L: float) -> np.ndarray:
    return rng.uniform(0.0, L, size=(I, 2)).astype(np.float64)

