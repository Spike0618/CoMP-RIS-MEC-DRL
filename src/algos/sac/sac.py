"""
SAC 


-  `scripts/train_3drl_baselines.py` /
- 
"""

from __future__ import annotations
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from src.algos.common_init import init_linear, hidden_gain, actor_head_gain, critic_head_gain
from src.utils.torch_safe_io import safe_torch_load


class SACGaussianActor(nn.Module):
    """SAC Actor"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self._reset_parameters()

        self.log_std_min = -20
        self.log_std_max = 2

    def _reset_parameters(self) -> None:
        """ + """
        init_linear(self.fc1, gain=hidden_gain())
        init_linear(self.fc2, gain=hidden_gain())
        init_linear(self.mean, gain=actor_head_gain())
        init_linear(self.log_std, gain=actor_head_gain())

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor, std_scale: float = 1.0):
        """ (action, log_prob)"""
        mean, log_std = self.forward(obs)
        std = log_std.exp() * float(max(std_scale, 1e-6))

        
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class SACCritic(nn.Module):
    """SAC Critic Q """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        
        self.q1_fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.q1_fc2 = nn.Linear(hidden, hidden)
        self.q1_fc3 = nn.Linear(hidden, 1)

        
        self.q2_fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.q2_fc2 = nn.Linear(hidden, hidden)
        self.q2_fc3 = nn.Linear(hidden, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """ + """
        init_linear(self.q1_fc1, gain=hidden_gain())
        init_linear(self.q1_fc2, gain=hidden_gain())
        init_linear(self.q1_fc3, gain=critic_head_gain())
        init_linear(self.q2_fc1, gain=hidden_gain())
        init_linear(self.q2_fc2, gain=hidden_gain())
        init_linear(self.q2_fc3, gain=critic_head_gain())

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)

        
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2


class ReplayBuffer:
    """"""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx]
        )


class SACAgent:
    """
    SAC Agent 

     Tianshou PPOSAC 
    """

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            device: torch.device,
            lr: float = 3e-4,
            lr_actor: Optional[float] = None,
            lr_critic: Optional[float] = None,
            lr_alpha: Optional[float] = None,
            gamma: float = 0.99,
            tau: float = 0.005,
            alpha: float = 0.2,
            auto_tune_alpha: bool = True,
            hidden: int = 256,
            buffer_size: int = 100000,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.auto_tune_alpha = auto_tune_alpha
        self.hidden_dim = int(hidden)
        self.exploration_ref_std = 0.1
        self.exploration_std_scale = 1.0

        lr_actor_v = float(lr if lr_actor is None else lr_actor)
        lr_critic_v = float(lr if lr_critic is None else lr_critic)
        lr_alpha_v = float(lr if lr_alpha is None else lr_alpha)

        
        self.actor = SACGaussianActor(obs_dim, act_dim, hidden).to(device)

        self.critic = SACCritic(obs_dim, act_dim, hidden).to(device)
        self.critic_target = SACCritic(obs_dim, act_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor_v)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic_v)

        
        if auto_tune_alpha:
            self.target_entropy = -act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha_v)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=device)

        
        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)

    def set_exploration(self, noise_std: float) -> None:
        """
        
        - DDPG/TD3
        - SAC
        """
        std_now = float(max(noise_std, 0.0))
        ref = float(max(self.exploration_ref_std, 1e-6))
        self.exploration_std_scale = float(max(std_now / ref, 1e-3))

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """ actor """
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(obs)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(obs, std_scale=float(self.exploration_std_scale))
        return action.cpu().numpy()

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """ actor/critic"""
        if self.buffer.size < batch_size:
            return {}

        
        obs, act, rew, next_obs, done = self.buffer.sample(batch_size)

        obs_t = torch.from_numpy(obs).to(self.device)
        act_t = torch.from_numpy(act).to(self.device)
        rew_t = torch.from_numpy(rew).to(self.device)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)

        
        with torch.no_grad():
            next_act, next_log_prob = self.actor.sample(next_obs_t)
            target_q1, target_q2 = self.critic_target(next_obs_t, next_act)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = rew_t.unsqueeze(-1) + self.gamma * (1 - done_t.unsqueeze(-1)) * target_q

        current_q1, current_q2 = self.critic(obs_t, act_t)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        
        new_act, log_prob = self.actor.sample(obs_t)
        q1_new, q2_new = self.critic(obs_t, new_act)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        stats = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
            'alpha': self.alpha.item(),
        }

        
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            stats['alpha_loss'] = alpha_loss.item()

        
        self._soft_update(self.critic, self.critic_target)

        return stats

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """ target """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        """"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu() if self.auto_tune_alpha else None,
            'model_meta': {
                'hidden': int(self.hidden_dim),
                'exploration_ref_std': float(self.exploration_ref_std),
            },
        }, path)

    def load(self, path):
        """"""
        checkpoint = safe_torch_load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if self.auto_tune_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].to(self.device)
            self.alpha = self.log_alpha.exp()
        try:
            meta = checkpoint.get('model_meta', {}) or {}
            self.exploration_ref_std = float(max(float(meta.get('exploration_ref_std', 0.1)), 1e-6))
        except Exception:
            pass
