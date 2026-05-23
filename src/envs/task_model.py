from __future__ import annotations

"""
 D  C


- 
-  load_scale 
"""

import numpy as np
from typing import Optional


class ConstantTaskModel:
    """
     D  C  load_scale 
    """
    def __init__(
        self,
        I: int,
        D_bits: float,
        C_cycles: float,
        load_scale: float = 1.0,
        load_alpha_D: float = 1.25,
        load_alpha_C: float = 1.15,
    ):
        self.I = int(I)
        self.D_bits = float(D_bits)
        self.C_cycles = float(C_cycles)
        self.load_scale = float(load_scale)
        self.load_alpha_D = float(load_alpha_D)
        self.load_alpha_C = float(load_alpha_C)

    def set_load_scale(self, s: float) -> None:
        self.load_scale = float(s)

    def sample(self, t: int = 0) -> tuple[np.ndarray, np.ndarray]:
        
        scale_D = (self.load_scale ** self.load_alpha_D) * (1.0 + 0.15 * (self.load_scale ** 2))
        scale_C = (self.load_scale ** self.load_alpha_C) * (1.0 + 0.10 * (self.load_scale ** 2))
        
        D = np.full((self.I,), self.D_bits, dtype=np.float64) * scale_D
        C = np.full((self.I,), self.C_cycles, dtype=np.float64) * scale_C
        return D, C

class UniformTaskModel:
    """
    
    - D ~ U(base*(1-jitter), base*(1+jitter))
    - C ~ U(base*(1-jitter), base*(1+jitter))

    load_scale
    """
    def __init__(
        self,
        rng: np.random.RandomState,
        I: int,
        D_bits_base: float,
        C_cycles_base: float,
        jitter: float = 0.3,
        load_scale: float = 1.0,
        load_alpha_D: float = 1.25,
        load_alpha_C: float = 1.15,
    ):
        self.rng = rng
        self.I = int(I)
        self.D0 = float(D_bits_base)
        self.C0 = float(C_cycles_base)
        self.jitter = float(jitter)
        self.load_scale = float(load_scale)
        self.load_alpha_D = float(load_alpha_D)
        self.load_alpha_C = float(load_alpha_C)

    def set_load_scale(self, s: float) -> None:
        self.load_scale = float(s)

    def sample(self, t: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        d_low, d_high = self.D0 * (1.0 - self.jitter), self.D0 * (1.0 + self.jitter)
        c_low, c_high = self.C0 * (1.0 - self.jitter), self.C0 * (1.0 + self.jitter)

        scale_D = (self.load_scale ** self.load_alpha_D) * (1.0 + 0.15 * (self.load_scale ** 2))
        scale_C = (self.load_scale ** self.load_alpha_C) * (1.0 + 0.10 * (self.load_scale ** 2))

        D = self.rng.uniform(d_low, d_high, size=(self.I,)).astype(np.float64) * scale_D
        C = self.rng.uniform(c_low, c_high, size=(self.I,)).astype(np.float64) * scale_C
        return D, C


class LogNormalTaskModel:
    """
    

    load_scale
    """
    def __init__(
        self,
        rng: np.random.RandomState,
        I: int,
        D_bits_base: float,
        C_cycles_base: float,
        jitter: float = 0.25,
        load_scale: float = 1.0,
        load_alpha_D: float = 1.25,
        load_alpha_C: float = 1.15,
        eps: float = 1e-12,
    ):
        self.rng = rng
        self.I = int(I)
        self.D0 = float(D_bits_base)
        self.C0 = float(C_cycles_base)
        self.jitter = float(jitter)
        self.load_scale = float(load_scale)
        self.load_alpha_D = float(load_alpha_D)
        self.load_alpha_C = float(load_alpha_C)
        self.eps = float(eps)

    def set_load_scale(self, s: float) -> None:
        self.load_scale = float(s)

    def sample(self, t: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        
        sigma = max(self.jitter, 0.0)
        
        mult_D = self.rng.lognormal(mean=0.0, sigma=sigma, size=(self.I,)) / (np.exp(0.5 * sigma * sigma) + self.eps)
        mult_C = self.rng.lognormal(mean=0.0, sigma=sigma, size=(self.I,)) / (np.exp(0.5 * sigma * sigma) + self.eps)

        scale_D = (self.load_scale ** self.load_alpha_D) * (1.0 + 0.15 * (self.load_scale ** 2))
        scale_C = (self.load_scale ** self.load_alpha_C) * (1.0 + 0.10 * (self.load_scale ** 2))

        D = (self.D0 * mult_D) * scale_D
        C = (self.C0 * mult_C) * scale_C
        return D, C
