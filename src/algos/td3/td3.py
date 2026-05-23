"""
TD3 


-  `scripts/train_3drl_baselines.py` /
- 
"""

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.algos.common_init import init_linear, hidden_gain, actor_head_gain, critic_head_gain
from src.utils.torch_safe_io import safe_torch_load


class TD3Actor(nn.Module):
    """TD3 Actor"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, act_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """ + """
        init_linear(self.fc1, gain=hidden_gain())
        init_linear(self.fc2, gain=hidden_gain())
        init_linear(self.fc3, gain=actor_head_gain())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class TD3Critic(nn.Module):
    """TD3 Critic Q """

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

    def q1_forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_fc3(q1)


class ReplayBuffer:
    """off-policy"""

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


class TD3Agent:
    """
    TD3 Agent 

     Tianshou PPOTD3 
    """

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            device: torch.device,
            lr_actor: float = 1e-4,
            lr_critic: float = 1e-3,
            gamma: float = 0.99,
            tau: float = 0.005,
            policy_delay: int = 2,
            target_noise: float = 0.2,
            noise_clip: float = 0.5,
            hidden: int = 256,
            buffer_size: int = 100000,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.hidden_dim = int(hidden)

        
        self.actor = TD3Actor(obs_dim, act_dim, hidden).to(device)
        self.actor_target = TD3Actor(obs_dim, act_dim, hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(obs_dim, act_dim, hidden).to(device)
        self.critic_target = TD3Critic(obs_dim, act_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

        
        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)

        
        self.total_it = 0

        
        self.noise_std = 0.1

    def set_exploration(self, noise_std: float) -> None:
        """"""
        self.noise_std = float(max(noise_std, 0.0))

    def get_action(self, obs: torch.Tensor, add_noise: bool = True) -> np.ndarray:
        """ actor """
        with torch.no_grad():
            action = self.actor(obs)
            if add_noise:
                noise = torch.randn_like(action) * self.noise_std
                action = action + noise
                action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy()

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """ actor/critic"""
        if self.buffer.size < batch_size:
            return {}

        self.total_it += 1

        
        obs, act, rew, next_obs, done = self.buffer.sample(batch_size)

        obs_t = torch.from_numpy(obs).to(self.device)
        act_t = torch.from_numpy(act).to(self.device)
        rew_t = torch.from_numpy(rew).to(self.device)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)

        
        with torch.no_grad():
            
            noise = (torch.randn_like(act_t) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(next_obs_t) + noise).clamp(-1.0, 1.0)

            
            target_q1, target_q2 = self.critic_target(next_obs_t, next_act)
            target_q = torch.min(target_q1, target_q2)
            target_q = rew_t + self.gamma * (1 - done_t) * target_q.squeeze()

        
        current_q1, current_q2 = self.critic(obs_t, act_t)
        current_q1 = current_q1.squeeze()
        current_q2 = current_q2.squeeze()

        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        stats = {
            'critic_loss': critic_loss.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
        }

        
        if self.total_it % self.policy_delay == 0:
            
            actor_loss = -self.critic.q1_forward(obs_t, self.actor(obs_t)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            stats['actor_loss'] = actor_loss.item()

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
            'model_meta': {
                'hidden': int(self.hidden_dim),
                'policy_delay': int(self.policy_delay),
            },
        }, path)

    def load(self, path):
        """"""
        checkpoint = safe_torch_load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
