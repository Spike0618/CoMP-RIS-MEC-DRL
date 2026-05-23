"""
Gymnasium CompRISEnvJoint  gymnasium.EnvTianshouCollector/Trainer


- /
-  done env.step() done terminatedtruncatedFalse

 CompRISEnvJoint.step 
- raw action [-1, 1]
-  delta_q  [-1, 1]  (Vmax*dt) 
-  score/theta  [-1, 1]

 action_space  [-1, 1]  Box
 BasePolicy.map_action 
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.envs.comp_ris_env_joint import CompRISEnvJoint, JointEnvConfig


class CompRISEnvGym(gym.Env):
    """Gymnasium"""

    metadata = {"render_modes": []}

    def __init__(self, config: JointEnvConfig, seed: int = 0):
        super().__init__()
        
        self.config = copy.deepcopy(config)
        self.seed_value = int(seed)
        self.obs_clip_range = float(getattr(self.config, "obs_clip_range", 20.0))
        if (not np.isfinite(self.obs_clip_range)) or self.obs_clip_range <= 0.0:
            self.obs_clip_range = 20.0
        # Pending control signals are set by vectorized trainer backends that
        # cannot directly call env methods (e.g., via set_env_attr in workers).
        self._pending_meta_action: Optional[Dict[str, Any]] = None
        self._pending_load_scale: Optional[float] = None
        self._last_meta_apply_result: Optional[Dict[str, Any]] = None

        
        self.env = CompRISEnvJoint(self.config, seed=self.seed_value)

        
        self.M = int(getattr(self.config, "M"))
        self.I = int(getattr(self.config, "I"))
        self.obs_dim = int(self.env.obs_dim)
        self.act_dim = int(self.env.action_dim)

        
        obs_low = -np.full((self.obs_dim,), self.obs_clip_range, dtype=np.float32)
        obs_high = np.full((self.obs_dim,), self.obs_clip_range, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        
        
        low = -np.ones((self.act_dim,), dtype=np.float32)
        high = np.ones((self.act_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.size != self.obs_dim:
            raise ValueError(f"obs: got={int(arr.size)}, expected={int(self.obs_dim)}")
        if self.obs_clip_range > 0.0:
            arr = np.clip(arr, -self.obs_clip_range, self.obs_clip_range)
        return arr

    def _flush_pending_controls(self) -> None:
        pending_load = getattr(self, "_pending_load_scale", None)
        if pending_load is not None:
            try:
                self.set_load_scale(float(pending_load))
            except Exception:
                pass
            self._pending_load_scale = None

        pending_meta = getattr(self, "_pending_meta_action", None)
        if isinstance(pending_meta, dict) and pending_meta:
            try:
                self._last_meta_apply_result = self.apply_meta_action(dict(pending_meta))
            except Exception as e:
                self._last_meta_apply_result = {
                    "applied": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            self._pending_meta_action = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        if seed is not None:
            self.seed(seed)
        self._flush_pending_controls()
        obs = self.env.reset()
        return self._clip_obs(obs), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        
        self._flush_pending_controls()
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.size != self.act_dim:
            strict_dim_check = bool(getattr(self.config, "strict_action_dim_check", True))
            msg = f"action: got={int(act.size)}, expected={int(self.act_dim)}"
            if strict_dim_check:
                raise ValueError(msg)
        obs, reward, done, info = self.env.step(act)
        terminated = bool(done)
        truncated = False
        return self._clip_obs(obs), float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def seed(self, seed: Optional[int] = None):
        
        if seed is None:
            seed = self.seed_value
        self.seed_value = int(seed)
        
        if hasattr(self.env, "_seed_base"):
            try:
                self.env._seed_base = int(self.seed_value)
            except Exception:
                pass
        self.env.rng = np.random.RandomState(self.seed_value)
        if hasattr(self.env, "_noise_rng"):
            try:
                self.env._noise_rng = self.env.rng
            except Exception:
                pass
        
        task_model = getattr(self.env, "task_model", None)
        if task_model is not None and hasattr(task_model, "rng"):
            try:
                task_model.rng = self.env.rng
            except Exception:
                pass
        return [self.seed_value]

    
    def set_load_scale(self, scale: float) -> None:
        self.env.set_load_scale(float(scale))

    def apply_meta_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return self.env.apply_meta_action(action)

    def set_pending_load_scale(self, scale: float) -> None:
        self._pending_load_scale = float(scale)

    def set_pending_meta_action(self, action: Dict[str, Any]) -> None:
        if isinstance(action, dict) and action:
            self._pending_meta_action = dict(action)

    def enable_trace(self, enable: bool = True, keep_theta: bool = False, max_steps: Optional[int] = None) -> None:
        self.env.enable_trace(enable, keep_theta, max_steps)

    def get_trace(self) -> Optional[Dict]:
        return getattr(self.env, "_trace", None)
