from __future__ import annotations

"""
PhaseT6 PID-Lagrangian


-  lambda 
- / g_t[0,1]
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PidLagrangeController:
    """PID """

    kp: float = 0.05
    ki: float = 0.01
    kd: float = 0.00
    delta: float = 0.20
    lambda_init: float = 0.0
    lambda_min: float = 0.0
    lambda_max: float = 20.0
    integral_clip: float = 10.0
    
    violation_ema_beta: float = 0.8
    deadband: float = 0.0
    lambda_step_clip: float = 0.25
    integral_decay: float = 1.0
    freeze_after_frac: float = 1.0

    def __post_init__(self) -> None:
        self.kp = float(max(self.kp, 0.0))
        self.ki = float(max(self.ki, 0.0))
        self.kd = float(max(self.kd, 0.0))
        self.delta = float(max(self.delta, 0.0))
        self.lambda_min = float(self.lambda_min)
        self.lambda_max = float(max(self.lambda_max, self.lambda_min))
        self.integral_clip = float(max(self.integral_clip, 1e-6))
        self.violation_ema_beta = float(np.clip(self.violation_ema_beta, 0.0, 0.9999))
        self.deadband = float(max(self.deadband, 0.0))
        self.lambda_step_clip = float(max(self.lambda_step_clip, 0.0))
        self.integral_decay = float(np.clip(self.integral_decay, 0.0, 1.0))
        self.freeze_after_frac = float(np.clip(self.freeze_after_frac, 0.0, 1.0))
        self._lambda = float(np.clip(float(self.lambda_init), self.lambda_min, self.lambda_max))
        self._e_prev = 0.0
        self._e_int = 0.0
        self._g_ema = 0.0
        self._last_update_frozen = False

    @property
    def value(self) -> float:
        """"""
        return float(self._lambda)

    @property
    def violation_ema(self) -> float:
        """EMA"""
        return float(self._g_ema)

    @property
    def last_update_frozen(self) -> bool:
        """"""
        return bool(self._last_update_frozen)

    def reset(self, lambda_init: float | None = None) -> float:
        """ episode """
        if lambda_init is None:
            lam0 = float(self.lambda_init)
        else:
            lam0 = float(lambda_init)
        self._lambda = float(np.clip(lam0, self.lambda_min, self.lambda_max))
        self._e_prev = 0.0
        self._e_int = 0.0
        self._g_ema = 0.0
        self._last_update_frozen = False
        return float(self._lambda)

    def update(self, violation_value: float, progress: float | None = None) -> float:
        """
        

        
        e_t = g_t - delta
        lambda <- clip(lambda + kp*e + ki*sum_e + kd*(e-e_prev), [lambda_min, lambda_max])
        """
        g_t = float(max(violation_value, 0.0))
        self._g_ema = float(self.violation_ema_beta * self._g_ema + (1.0 - self.violation_ema_beta) * g_t)
        g_used = float(self._g_ema)

        prog = float(np.clip(float(progress), 0.0, 1.0)) if progress is not None else 0.0
        self._last_update_frozen = bool(prog >= self.freeze_after_frac)
        if self._last_update_frozen:
            return float(self._lambda)

        e_t = float(g_used - self.delta)
        if abs(e_t) <= self.deadband:
            e_t = 0.0
        self._e_int = float(
            np.clip(self.integral_decay * self._e_int + e_t, -self.integral_clip, self.integral_clip)
        )
        e_diff = float(e_t - self._e_prev)
        dlam = float(self.kp * e_t + self.ki * self._e_int + self.kd * e_diff)
        if self.lambda_step_clip > 0.0:
            dlam = float(np.clip(dlam, -self.lambda_step_clip, self.lambda_step_clip))
        lam = float(self._lambda + dlam)
        self._lambda = float(np.clip(lam, self.lambda_min, self.lambda_max))
        self._e_prev = float(e_t)
        return float(self._lambda)
