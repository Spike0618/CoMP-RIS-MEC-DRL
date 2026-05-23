"""



-  Welford /
-  (obs_dim,)  (batch, obs_dim)
-  update=True  update=False 
-  clip
"""

from __future__ import annotations

import numpy as np


class ObsNormalizer:
    """
    x -> (x - mean) / (std + eps)

    
    - normalize(obs, update=True)
    - normalize(obs, update=False)
    """
    
    def __init__(
        self,
        obs_dim: int,
        clip: float = 10.0,
        eps: float = 1e-8,
        init_count: int = 0,
        var_warmup_prior_count: int = 32,
        var_warmup_prior_value: float = 1.0,
    ):
        """
        
        - obs_dim
        - clip [-clip, clip]<=0 
        - eps
        - init_count0 set_stats
        - var_warmup_prior_count32
        - var_warmup_prior_value1.0
        """
        self.obs_dim = int(obs_dim)
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim ={self.obs_dim}")
        self.clip = float(clip)
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError(f"eps ={self.eps}")

        
        self.count = int(init_count)
        if self.count < 0:
            raise ValueError(f"init_count ={self.count}")
        if self.count != 0:
            
            raise ValueError(
                "init_count 0 set_stats({count,mean,var})"
            )
        self.mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.var = np.ones(self.obs_dim, dtype=np.float64)
        self.M2 = np.zeros(self.obs_dim, dtype=np.float64)  
        
        self.var_warmup_prior_count = int(max(int(var_warmup_prior_count), 0))
        self.var_warmup_prior_value = float(max(float(var_warmup_prior_value), 1e-12))

    def _recompute_var(self) -> None:
        """
        (count, M2)
        """
        if self.count > 1:
            var_raw = self.M2 / max(self.count - 1, 1)
        else:
            var_raw = np.ones_like(self.var) * float(self.var_warmup_prior_value)
        var_raw = np.maximum(var_raw, 0.0)

        if self.count > 1 and self.var_warmup_prior_count > 0:
            eff_n = float(max(self.count - 1, 0))
            alpha = float(eff_n / (eff_n + float(self.var_warmup_prior_count)))
            prior = np.ones_like(var_raw) * float(self.var_warmup_prior_value)
            self.var = alpha * var_raw + (1.0 - alpha) * prior
        else:
            self.var = var_raw
        self.var = np.maximum(self.var, 0.0)

    def _merge_batch_stats(self, obs_batch: np.ndarray) -> None:
        """
        Welfordbatch
        """
        if obs_batch.ndim != 2 or obs_batch.shape[1] != self.obs_dim:
            raise ValueError(f"obs_batch (*,{self.obs_dim})={obs_batch.shape}")
        if obs_batch.shape[0] <= 0:
            return

        batch_mean = np.mean(obs_batch, axis=0)
        if obs_batch.shape[0] > 1:
            centered = obs_batch - batch_mean
            batch_M2 = np.sum(centered * centered, axis=0)
        else:
            batch_M2 = np.zeros((self.obs_dim,), dtype=np.float64)
        batch_count = int(obs_batch.shape[0])

        if self.count <= 0:
            self.count = batch_count
            self.mean = batch_mean
            self.M2 = batch_M2
            self._recompute_var()
            return

        total = int(self.count + batch_count)
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (float(batch_count) / float(max(total, 1)))
        self.M2 = self.M2 + batch_M2 + (delta * delta) * (float(self.count) * float(batch_count) / float(max(total, 1)))
        self.count = total
        self._recompute_var()
        
    def update(self, obs: np.ndarray) -> None:
        """
        

        obs 
        - (obs_dim,)
        - (batch, obs_dim)
        """
        obs = np.asarray(obs, dtype=np.float64)

        
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        elif obs.ndim != 2:
            raise ValueError(f"obs 12 shape={obs.shape}")
        if obs.shape[1] != self.obs_dim:
            raise ValueError(f"obs  obs_dim={self.obs_dim} shape={obs.shape}")

        
        mask = np.all(np.isfinite(obs), axis=1)
        if not np.any(mask):
            return
        obs_valid = obs[mask]
        self._merge_batch_stats(obs_valid)

    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """
        

        - update=True
        - update=False
        """
        obs = np.asarray(obs, dtype=np.float32)
        original_shape = obs.shape
        if obs.ndim == 1:
            if obs.shape[0] != self.obs_dim:
                raise ValueError(f"obs  obs_dim={self.obs_dim} shape={obs.shape}")
        elif obs.ndim == 2:
            if obs.shape[1] != self.obs_dim:
                raise ValueError(f"obs  obs_dim={self.obs_dim} shape={obs.shape}")
        else:
            raise ValueError(f"obs 12 shape={obs.shape}")

        
        if update:
            self.update(obs)

        
        mean32 = self.mean.astype(np.float32)
        std = np.sqrt(np.maximum(self.var, 0.0) + self.eps).astype(np.float32)
        
        obs_safe = np.where(np.isfinite(obs), obs, mean32)
        obs_norm = (obs_safe - mean32) / std

        
        if self.clip > 0:
            obs_norm = np.clip(obs_norm, -self.clip, self.clip)
        
        obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=0.0, neginf=0.0)

        return obs_norm.reshape(original_shape)
    
    def denormalize(self, obs_norm: np.ndarray) -> np.ndarray:
        """
        
        """
        obs_norm = np.asarray(obs_norm, dtype=np.float32)
        if obs_norm.ndim == 1:
            if obs_norm.shape[0] != self.obs_dim:
                raise ValueError(f"obs_norm  obs_dim={self.obs_dim} shape={obs_norm.shape}")
        elif obs_norm.ndim == 2:
            if obs_norm.shape[1] != self.obs_dim:
                raise ValueError(f"obs_norm  obs_dim={self.obs_dim} shape={obs_norm.shape}")
        else:
            raise ValueError(f"obs_norm 12 shape={obs_norm.shape}")
        std = np.sqrt(np.maximum(self.var, 0.0) + self.eps)
        obs = obs_norm * std.astype(np.float32) + self.mean.astype(np.float32)
        return obs
    
    def get_stats(self) -> dict:
        """/"""
        return {
            "count": int(self.count),
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "std": np.sqrt(self.var + self.eps),
        }
    
    def set_stats(self, stats: dict) -> None:
        """ checkpoint """
        if not isinstance(stats, dict):
            raise ValueError("stats ")
        if "mean" not in stats or "var" not in stats:
            raise ValueError("stats  mean  var")
        self.count = int(stats.get("count", 0))
        if self.count < 0:
            raise ValueError(f"count ={self.count}")
        mean = np.asarray(stats["mean"], dtype=np.float64).reshape(-1)
        var = np.asarray(stats["var"], dtype=np.float64).reshape(-1)
        if mean.shape[0] != self.obs_dim or var.shape[0] != self.obs_dim:
            raise ValueError(
                f"stats  obs_dim={self.obs_dim}"
                f" mean={mean.shape}, var={var.shape}"
            )
        if (not np.all(np.isfinite(mean))) or (not np.all(np.isfinite(var))):
            raise ValueError("stats NaN/Inf")
        var = np.maximum(var, 0.0)
        self.mean = mean
        
        self.M2 = var * max(self.count - 1, 0)
        self._recompute_var()

    def reset(self) -> None:
        """"""
        self.count = 0
        self.mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.var = np.ones(self.obs_dim, dtype=np.float64) * float(self.var_warmup_prior_value)
        self.M2 = np.zeros(self.obs_dim, dtype=np.float64)
