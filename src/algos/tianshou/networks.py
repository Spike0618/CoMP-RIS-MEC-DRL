"""
Tianshou - CoMP+RISActor-Critic

Tianshou PPO
PPO
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Any, Optional, Union, Sequence


from src.algos.tianshou.scores_fixed import compute_scores_fixed_action
from src.algos.common_init import init_linear, hidden_gain, actor_head_gain, critic_head_gain


class CompRISPreprocessNet(nn.Module):
    """
    
    
    MLP
    ActorCritic
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: nn.Module = nn.ReLU,
        use_layernorm: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.output_dim = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.device = device
        
        
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            input_dim = hidden_dim
        
        self.model = nn.Sequential(*layers)
        self.model.to(device)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """LayerNorm"""
        for mod in self.model.modules():
            if isinstance(mod, nn.Linear):
                init_linear(mod, gain=hidden_gain())
    
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        
        
        
            obs: shape (batch, obs_dim)
            state: RNN
            info: 
             
        
            (features, state): 
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        
        features = self.model(obs)
        return features, state


class CompRISActor(nn.Module):
    """
    Actor
    
    
    """
    
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_dim: int,
        delta_dim: Optional[int] = None,
        max_action: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
    ):
        super().__init__()
        self.preprocess = preprocess_net
        self.action_dim = action_dim
        
        
        self.delta_dim = int(delta_dim or 0)
        if self.delta_dim < 0 or self.delta_dim >= int(action_dim):
            self.delta_dim = 0
        
        self.comp_meta_dim: int = 0
        self.max_action = max_action
        self.device = device
        self.unbounded = unbounded
        self.conditioned_sigma = conditioned_sigma

        
        
        self.log_std_min: float = -5.0
        self.log_std_max: float = 1.0

        
        
        
        self.sigma_scale: float = 1.0
        self.sigma_scale_delta: float = 1.0
        self.sigma_scale_score: float = 1.0
        self.sigma_scale_comp_meta: float = 1.0
        self.sigma_min: float = 0.0
        
        
        self.mu = nn.Linear(preprocess_net.output_dim, action_dim)
        self.mu.to(device)
        
        
        if conditioned_sigma:
            self.sigma = nn.Linear(preprocess_net.output_dim, action_dim)
            self.sigma.to(device)
        else:
            
            
            log_std_init = -0.5  
            self.sigma = nn.Parameter(torch.full((action_dim,), float(log_std_init), device=device))
        self._reset_parameters()

        # =========================
        
        
        
        
        
        # =========================
        self.scores_mode: str = "learned"  # learned/fixed
        self.scores_fixed_method: str = "balanced"
        self.scores_fixed_tanh_scale: float = 2.5
        self.scores_fixed_sigma: float = 0.02  
        
        self.scores_fixed_require_obs_raw: bool = True
        self._scores_fixed_cfg: Any = None
        self._warned_once_keys: set[str] = set()

    def _reset_parameters(self) -> None:
        """"""
        init_linear(self.mu, gain=actor_head_gain())
        if isinstance(self.sigma, nn.Linear):
            init_linear(self.sigma, gain=actor_head_gain())

    def set_scores_fixed(
        self,
        *,
        cfg: Any,
        mode: str = "balanced",
        tanh_scale: float = 2.5,
        sigma_fixed: float = 0.02,
        require_obs_raw: bool = True,
    ) -> None:
        """ C1 scores_fixed """
        self._scores_fixed_cfg = cfg
        self.scores_fixed_method = str(mode or "balanced").strip().lower()
        self.scores_fixed_tanh_scale = float(max(tanh_scale, 1e-6))
        self.scores_fixed_sigma = float(max(sigma_fixed, 1e-6))
        self.scores_fixed_require_obs_raw = bool(require_obs_raw)

    def _warn_once(self, key: str, message: str) -> None:
        """"""
        k = str(key).strip().lower()
        if not k:
            k = f"warn_{len(self._warned_once_keys)}"
        if k in self._warned_once_keys:
            return
        self._warned_once_keys.add(k)
        print(f"[HB][actor][WARN] {message}", flush=True)

    def set_sigma_scale(self, scale: float, sigma_min: float = 0.0) -> None:
        """sigma"""
        self.set_sigma_scales(scale_delta=scale, scale_score=scale, sigma_min=sigma_min)

    def set_sigma_scales(self, *, scale_delta: float, scale_score: float, sigma_min: float = 0.0, scale_comp_meta: float | None = None) -> None:
        """sigmascorecomp_meta"""
        self.sigma_scale_delta = float(scale_delta)
        self.sigma_scale_score = float(scale_score)
        if scale_comp_meta is not None:
            self.sigma_scale_comp_meta = float(scale_comp_meta)
        else:
            self.sigma_scale_comp_meta = float(scale_score)
        
        self.sigma_scale = float(scale_score)
        self.sigma_min = float(max(sigma_min, 0.0))
    
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Any = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """
        
        
        
            obs: 
            state: RNN
            info: 
             
        
            ((mu, sigma), state): 
        """
        
        features, state = self.preprocess(obs, state)
        
        
        mu = self.mu(features)
        
        
        if self.conditioned_sigma:
            log_std = self.sigma(features)
            log_std = torch.clamp(log_std, min=float(self.log_std_min), max=float(self.log_std_max))
            sigma = torch.exp(log_std)
        else:
            log_std = torch.clamp(self.sigma, min=float(self.log_std_min), max=float(self.log_std_max))
            sigma = torch.exp(log_std).expand_as(mu)

        
        if int(self.delta_dim) > 0:
            
            scale = torch.ones_like(mu, dtype=sigma.dtype, device=sigma.device)
            cm_dim = int(self.comp_meta_dim)
            d_dim = int(self.delta_dim)
            total_dim = int(mu.shape[-1])
            scale[..., :d_dim] = float(self.sigma_scale_delta)
            if cm_dim > 0 and total_dim > d_dim + cm_dim:
                
                scale[..., d_dim:total_dim - cm_dim] = float(self.sigma_scale_score)
                scale[..., total_dim - cm_dim:] = float(self.sigma_scale_comp_meta)
            elif cm_dim > 0 and total_dim == d_dim + cm_dim:
                
                scale[..., d_dim:] = float(self.sigma_scale_comp_meta)
            else:
                
                scale[..., d_dim:] = float(self.sigma_scale_score)
            sigma = sigma * scale
        else:
            sigma = sigma * float(self.sigma_scale)
        if float(self.sigma_min) > 0.0:
            sigma = torch.clamp(sigma, min=float(self.sigma_min))
        
        
        if not self.unbounded:
            mu = self.max_action * torch.tanh(mu)

        
        if str(getattr(self, "scores_mode", "learned")).strip().lower() == "fixed":
            
            dd = int(getattr(self, "delta_dim", 0))
            if dd > 0 and int(self.action_dim) > dd and self._scores_fixed_cfg is not None:
                
                obs_raw = None
                if isinstance(info, dict) and ("obs_raw" in info):
                    obs_raw = info.get("obs_raw", None)
                elif hasattr(info, "obs_raw"):
                    obs_raw = getattr(info, "obs_raw", None)
                elif isinstance(info, (list, tuple)):
                    try:
                        obs_raw = np.stack([np.asarray(ii.get("obs_raw"), dtype=np.float32) for ii in info], axis=0)
                    except Exception:
                        obs_raw = None
                if obs_raw is None:
                    msg = "scores_mode=fixed info obs_rawCollectorPreprocess "
                    if bool(getattr(self, "scores_fixed_require_obs_raw", True)):
                        raise RuntimeError(msg)
                    self._warn_once("scores_fixed_obs_raw_missing", msg + " learned")
                    return (mu, sigma), state

                try:
                    scores_fixed = compute_scores_fixed_action(
                        cfg=self._scores_fixed_cfg,
                        obs_raw=np.asarray(obs_raw, dtype=np.float32),
                        mode=str(getattr(self, "scores_fixed_method", "balanced")),
                        score_tanh_scale=float(getattr(self, "scores_fixed_tanh_scale", 2.5)),
                    )
                except Exception as exc:
                    msg = f"scores_fixed {type(exc).__name__}: {exc}"
                    if bool(getattr(self, "scores_fixed_require_obs_raw", True)):
                        raise RuntimeError(msg) from exc
                    self._warn_once("scores_fixed_compute_fail", msg + "learned")
                    return (mu, sigma), state

                scores_fixed_t = torch.as_tensor(scores_fixed, device=mu.device, dtype=mu.dtype)
                if scores_fixed_t.ndim == 1:
                    scores_fixed_t = scores_fixed_t.unsqueeze(0)

                
                if scores_fixed_t.shape[0] != mu.shape[0]:
                    
                    if scores_fixed_t.shape[0] == 1:
                        scores_fixed_t = scores_fixed_t.repeat(mu.shape[0], 1)
                    else:
                        scores_fixed_t = scores_fixed_t[: mu.shape[0]]

                
                mu = mu.clone()
                sigma = sigma.clone()
                mu[..., dd:] = scores_fixed_t[..., : (mu.shape[-1] - dd)]

                sig_fix = float(getattr(self, "scores_fixed_sigma", 0.02))
                sig_fix = float(max(sig_fix, float(getattr(self, "sigma_min", 0.0))))
                sigma[..., dd:] = torch.clamp(torch.ones_like(mu[..., dd:]) * sig_fix, min=1e-6)
        # ===== [END Phase C] =====
        
        return (mu, sigma), state


class CompRISCritic(nn.Module):
    """
    Critic
    
    V(s)
    """
    
    def __init__(
        self,
        preprocess_net: nn.Module,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.preprocess = preprocess_net
        self.device = device
        
        
        self.value_head = nn.Linear(preprocess_net.output_dim, 1)
        self.value_head.to(device)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """"""
        init_linear(self.value_head, gain=critic_head_gain())
    
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Any = None,
    ) -> torch.Tensor:
        """
        
        
        
            obs: 
            state: RNN
            info: 
             
        
            value: shape (batch, 1)
        """
        
        features, _ = self.preprocess(obs, state)
        
        
        value = self.value_head(features)
        
        return value


def create_actor_critic(
    obs_dim: int,
    act_dim: int,
    delta_dim: Optional[int] = None,
    hidden_sizes: Sequence[int] = (256, 256),
    activation: nn.Module = nn.ReLU,
    use_layernorm: bool = True,
    max_action: float = 1.0,
    device: Union[str, torch.device] = "cpu",
    unbounded: bool = False,
    conditioned_sigma: bool = False,
    share_preprocess: bool = True,
) -> Tuple[CompRISActor, CompRISCritic]:
    """
    Actor-Critic
    
    
        obs_dim: 
        act_dim: 
        hidden_sizes: 
        activation: 
        use_layernorm: LayerNorm
        max_action: 
        device: 
        unbounded: 
        conditioned_sigma: 
         
    
        (actor, critic): ActorCritic
    """
    
    
    
    actor_pre = CompRISPreprocessNet(
        obs_dim=obs_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_layernorm=use_layernorm,
        device=device,
    )
    critic_pre = actor_pre if bool(share_preprocess) else CompRISPreprocessNet(
        obs_dim=obs_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_layernorm=use_layernorm,
        device=device,
    )
    
    
    actor = CompRISActor(
        preprocess_net=actor_pre,
        action_dim=act_dim,
        delta_dim=delta_dim,
        max_action=max_action,
        device=device,
        unbounded=unbounded,
        conditioned_sigma=conditioned_sigma,
    )
    
    
    critic = CompRISCritic(
        preprocess_net=critic_pre,
        device=device,
    )
    
    return actor, critic
