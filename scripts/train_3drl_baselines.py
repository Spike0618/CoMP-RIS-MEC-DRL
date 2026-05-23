"""
DDPG/SAC/TD3 


-  Tianshou PPO
-  baseline  ckpt
"""

from __future__ import annotations

import time
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from collections import deque

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.runio import make_run_dir, snapshot_cfgs, ensure_run_subdirs
from src.utils.seed import set_global_seed
from src.utils.torch_safe_io import safe_torch_load
from src.envs.comp_ris_env_joint import JointEnvConfig, CompRISEnvJoint
from src.algos.ddpg.ddpg import DDPGAgent
from src.algos.sac.sac import SACAgent
from src.algos.td3.td3 import TD3Agent
from src.utils.env_cfg import build_joint_env_cfg
from src.utils.hb import now_ts, fmt_hms

CKPT_DIR_NAME = "checkpoints"


def _linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    """"""
    s = int(max(step, 0))
    d = int(max(decay_steps, 1))
    if s >= d:
        return float(end)
    t = float(s) / float(d)
    return float(start + (end - start) * t)


class _A2CActorCritic(nn.Module):
    """A2Ctanh + """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(int(obs_dim), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(hidden))
        self.mean_head = nn.Linear(int(hidden), int(act_dim))
        self.value_head = nn.Linear(int(hidden), 1)
        self.log_std = nn.Parameter(torch.zeros(int(act_dim), dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        value = self.value_head(x)
        log_std = torch.clamp(self.log_std, -5.0, 2.0).unsqueeze(0).expand_as(mean)
        return mean, log_std, value


class A2CAgent:
    """A2Cbaseline"""

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            device: torch.device,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            hidden: int = 256,
            rollout_steps: int = 256,
            value_coef: float = 0.5,
            entropy_coef: float = 0.01,
            grad_clip: float = 1.0,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.hidden_dim = int(hidden)
        self.rollout_steps = int(max(16, rollout_steps))
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.grad_clip = float(max(0.0, grad_clip))
        self.lr_actor = float(lr_actor)
        self.lr_critic = float(lr_critic)
        self.exploration_ref_std = 0.1
        self.exploration_std_scale = 1.0

        self.actor_critic = _A2CActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=float(max(self.lr_actor, self.lr_critic)),
        )

        self._rollout: List[Dict[str, Any]] = []
        self._last_next_obs: np.ndarray | None = None
        self._last_done: bool = False

    def set_exploration(self, noise_std: float) -> None:
        """"""
        std_now = float(max(noise_std, 0.0))
        ref = float(max(self.exploration_ref_std, 1e-6))
        self.exploration_std_scale = float(max(std_now / ref, 1e-3))

    def _dist(self, mean: torch.Tensor, log_std: torch.Tensor) -> Normal:
        std = torch.exp(log_std) * float(max(self.exploration_std_scale, 1e-6))
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False, return_aux: bool = False):
        mean, log_std, value = self.actor_critic(obs)
        if bool(deterministic):
            raw = mean
            action = torch.tanh(raw)
            log_prob = torch.zeros((obs.shape[0],), device=obs.device, dtype=torch.float32)
        else:
            dist = self._dist(mean, log_std)
            raw = dist.rsample()
            action = torch.tanh(raw)
            log_prob = dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = torch.sum(log_prob, dim=-1)
        act_np = action.detach().cpu().numpy()
        if not bool(return_aux):
            return act_np
        aux = {
            "log_prob": float(log_prob.detach().cpu().numpy().reshape(-1)[0]),
            "value": float(value.detach().cpu().numpy().reshape(-1)[0]),
        }
        return act_np, aux

    def add_transition(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: bool,
            value: float,
            log_prob: float,
            next_obs: np.ndarray,
    ) -> None:
        self._rollout.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "reward": float(reward),
                "done": float(done),
                "value": float(value),
                "log_prob": float(log_prob),
            }
        )
        self._last_next_obs = np.asarray(next_obs, dtype=np.float32)
        self._last_done = bool(done)

    def _ready_to_update(self) -> bool:
        return len(self._rollout) >= int(self.rollout_steps) or bool(self._last_done)

    def _compute_gae(self, next_value: float) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._rollout)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        returns = np.zeros((n,), dtype=np.float32)
        adv = np.zeros((n,), dtype=np.float32)
        gae = 0.0
        nv = float(next_value)
        for t in reversed(range(n)):
            rew = float(self._rollout[t]["reward"])
            done = float(self._rollout[t]["done"])
            val = float(self._rollout[t]["value"])
            delta = rew + self.gamma * nv * (1.0 - done) - val
            gae = delta + self.gamma * self.gae_lambda * (1.0 - done) * gae
            adv[t] = float(gae)
            returns[t] = float(adv[t] + val)
            nv = val
        return returns, adv

    def update(self, _batch_size: int = 256) -> Dict[str, float]:
        if not self._ready_to_update():
            return {}
        if len(self._rollout) <= 0:
            return {}

        if bool(self._last_done) or (self._last_next_obs is None):
            next_value = 0.0
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(self._last_next_obs).float().unsqueeze(0).to(self.device)
                _, _, v = self.actor_critic(obs_t)
                next_value = float(v.squeeze(0).squeeze(0).item())

        returns_np, adv_np = self._compute_gae(next_value=next_value)
        if returns_np.size <= 0:
            self._rollout.clear()
            return {}

        obs_np = np.asarray([r["obs"] for r in self._rollout], dtype=np.float32)
        act_np = np.asarray([r["action"] for r in self._rollout], dtype=np.float32)

        obs_t = torch.from_numpy(obs_np).float().to(self.device)
        act_t = torch.from_numpy(act_np).float().to(self.device)
        ret_t = torch.from_numpy(returns_np).float().to(self.device)
        adv_t = torch.from_numpy(adv_np).float().to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-6)

        mean, log_std, value = self.actor_critic(obs_t)
        dist = self._dist(mean, log_std)

        act_clip = torch.clamp(act_t, -0.999999, 0.999999)
        raw_t = torch.atanh(act_clip)
        log_prob = dist.log_prob(raw_t) - torch.log(1.0 - act_clip.pow(2) + 1e-6)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1).mean()

        actor_loss = -(log_prob * adv_t.detach()).mean()
        critic_loss = F.mse_loss(value.squeeze(-1), ret_t)
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.grad_clip)
        self.optimizer.step()

        value_mean = float(value.detach().mean().item())
        self._rollout.clear()
        return {
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "value_mean": float(value_mean),
            "entropy": float(entropy.detach().item()),
        }

    def save(self, path: Path | str) -> None:
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "model_meta": {
                    "hidden": int(self.hidden_dim),
                    "algo": "a2c",
                },
            },
            path,
        )

    def load(self, path: Path | str) -> None:
        obj = safe_torch_load(path, map_location=self.device)
        if not isinstance(obj, dict) or "actor_critic" not in obj:
            raise RuntimeError(f"A2C checkpoint{path}")
        self.actor_critic.load_state_dict(obj["actor_critic"], strict=True)


# =========================

# =========================
def plot_3drl_training_curves(
        run_dir: Path,
        algo_name: str,
        reward_history: List[float],
        actor_loss_history: List[float],
        critic_loss_history: List[float],
        q_value_history: List[float],
        episode_history: List[int],
        update_episode_history: List[int],
        ema_span: int = 48,
):
    """ 3DRL """
    
    
    fig_dir = run_dir / "figs" / f"train{str(algo_name).upper()}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    episodes = np.array(episode_history, dtype=np.int64)
    update_episodes = np.array(update_episode_history, dtype=np.int64)

    rewards = np.array(reward_history, dtype=np.float32)
    actor_losses = np.array(actor_loss_history, dtype=np.float32)
    critic_losses = np.array(critic_loss_history, dtype=np.float32)
    q_values = np.array(q_value_history, dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{algo_name} Training Curves', fontsize=16, fontweight='bold')

    
    from src.utils.plot_smoothing import ema_smooth

    def _plot_raw_ema(ax, x, y, color, title, ylabel):
        x0 = np.asarray(x, dtype=np.float64).reshape(-1)
        y0 = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(min(x0.size, y0.size))
        x0 = x0[:n]
        y0 = y0[:n]
        mask = np.isfinite(x0) & np.isfinite(y0)
        x0 = x0[mask]
        y0 = y0[mask]
        if x0.size <= 0 or y0.size <= 0:
            return

        ax.plot(x0, y0, color=color, linewidth=1.2, alpha=0.32, label="raw", zorder=2)
        if y0.size < 3:
            ax.text(0.02, 0.96, "too few points", transform=ax.transAxes, fontsize=9, color="0.35", ha="left", va="top")
        else:
            span_eff = int(max(2, int(ema_span)))
            y_ema = ema_smooth(y0, span=span_eff, adjust=False)
            ax.plot(x0, y_ema, color=color, linewidth=3.0, alpha=0.95, label=f"EMA(span={int(span_eff)})", zorder=4)
        ax.set_xlabel('Training Episodes')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    
    ax = axes[0, 0]
    _plot_raw_ema(ax, episodes, rewards, 'b', 'Training Rollout: Total rewards (reward_total)', 'Total rewards')

    
    ax = axes[0, 1]
    if len(actor_losses) > 0:
        _plot_raw_ema(ax, update_episodes[:len(actor_losses)], actor_losses, 'r', 'Actor Loss', 'Actor Loss')

    
    ax = axes[1, 0]
    if len(critic_losses) > 0:
        _plot_raw_ema(ax, update_episodes[:len(critic_losses)], critic_losses, 'g', 'Critic Loss', 'Critic Loss')

    
    ax = axes[1, 1]
    if len(q_values) > 0:
        _plot_raw_ema(ax, update_episodes[:len(q_values)], q_values, 'm', 'Average Q-Value', 'Q-Value')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    save_path = fig_dir / f"{algo_name.lower()}_training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[HB][plot] Saved {algo_name} training curves to {save_path}")


# =========================

# =========================
def train_3drl(args, algo_name: str):
    """ DDPG/SAC/TD3"""
    
    run_name = f"{algo_name.lower()}_{args.run_name}" if args.run_name else f"{algo_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = make_run_dir(PROJECT_ROOT, subdir="runs/paper", ts_name=run_name)
    ensure_run_subdirs(run_dir)
    print(f"[HB][train] run_dir = {run_dir}")

    
    with open(args.env_yaml, "r", encoding="utf-8") as f:
        env_cfg_dict = yaml.safe_load(f)

    
    
    
    
    algo_upper = str(algo_name).upper()
    default_lr_actor = 3e-4 if algo_upper in ("SAC", "A2C") else 1e-4
    default_lr_critic = 3e-4 if algo_upper in ("SAC", "A2C") else 1e-3
    default_policy_delay = 2 if algo_upper == "TD3" else 1
    default_warmup_steps = 0 if algo_upper == "A2C" else 10000

    train_cfg = {
        'seed': 42,
        'device': 'cuda',
        'network': {
            
            'hidden_size': 512,
        },
        'train': {
            
            'total_steps': 1200450,
            'batch_size': 256,
            'lr_actor': default_lr_actor,
            'lr_critic': default_lr_critic,
            'gamma': 0.99,
            'tau': 0.005,
            'buffer_size': 100000,
            'warmup_steps': default_warmup_steps,
            'update_every': 1,
            
            'updates_per_step': 1,
            
            'policy_delay': default_policy_delay,
            
            'exploration_std_init': 0.10,
            'exploration_std_final': 0.02,
            'exploration_std_decay_steps': 300000,
            'save_every': 50000,
            'eval_every': 10000,
            'heartbeat_every': 2000,
        }
    }
    
    if getattr(args, "total_steps", None) is not None:
        train_cfg["train"]["total_steps"] = int(args.total_steps)
    if getattr(args, "seed", None) is not None:
        train_cfg["seed"] = int(args.seed)
    if getattr(args, "hidden_size", None) is not None:
        train_cfg.setdefault("network", {})["hidden_size"] = int(args.hidden_size)
    if getattr(args, "updates_per_step", None) is not None:
        train_cfg["train"]["updates_per_step"] = int(args.updates_per_step)
    if getattr(args, "policy_delay", None) is not None:
        train_cfg["train"]["policy_delay"] = int(args.policy_delay)
    if getattr(args, "exploration_std_init", None) is not None:
        train_cfg["train"]["exploration_std_init"] = float(args.exploration_std_init)
    if getattr(args, "exploration_std_final", None) is not None:
        train_cfg["train"]["exploration_std_final"] = float(args.exploration_std_final)
    if getattr(args, "exploration_std_decay_steps", None) is not None:
        train_cfg["train"]["exploration_std_decay_steps"] = int(args.exploration_std_decay_steps)

    
    snapshot_cfgs(run_dir, env_cfg_dict, train_cfg)
    (run_dir / "config_paths.txt").write_text(
        f"env_yaml: {args.env_yaml}\ntrain_yaml: <default_3drl.yaml (inline)>\n",
        encoding="utf-8",
    )

    
    seed = train_cfg.get('seed', 42)
    deterministic_seed = bool((train_cfg.get("train", {}) or {}).get("deterministic_seed", True))
    set_global_seed(seed, deterministic=deterministic_seed)
    print(f"[HB][train] seed={seed} deterministic_seed={int(deterministic_seed)}")

    
    device_str = train_cfg.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"[HB][train] Using device: {device}")

    
    env_cfg, env_cfg_dict_clean, dropped = build_joint_env_cfg(env_cfg_dict)
    print(f"[HB][train] env_cfg keys: kept={len(env_cfg_dict_clean)} dropped={len(dropped)}")
    if dropped:
        print(f"[HB][train] dropped keys (first 20): {dropped[:20]}")

    env = CompRISEnvJoint(env_cfg, seed=seed)

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    print(f"[HB][train] obs_dim={obs_dim}, act_dim={act_dim}")
    print(
        "[HB][train][fairness-env] "
        f"load_scale={float(getattr(env.cfg, 'load_scale', 1.0)):.3f} "
        f"comp_rule_threshold={float(getattr(env.cfg, 'comp_rule_threshold', float('nan'))):.4f} "
        f"z4_gap_enable={int(bool(getattr(env.cfg, 'z4_gap_enable', False)))} "
        f"z4_assoc_stage_enable={int(bool(getattr(env.cfg, 'z4_assoc_stage_enable', False)))} "
        f"assoc_window=({float(getattr(env.cfg, 'z4_assoc_stage_start_frac', 0.0)):.3f},"
        f"{float(getattr(env.cfg, 'z4_assoc_stage_end_frac', 1.0)):.3f})",
        flush=True,
    )
    
    
    
    

    
    train_params = train_cfg.get('train', {})
    net_params = train_cfg.get('network', {})
    total_steps = int(train_params.get('total_steps', 600000))
    batch_size = int(train_params.get('batch_size', 256))
    lr_actor = float(train_params.get('lr_actor', 1e-4))
    lr_critic = float(train_params.get('lr_critic', 1e-3))
    gamma = float(train_params.get('gamma', 0.99))
    tau = float(train_params.get('tau', 0.005))
    hidden_size = int(train_params.get('hidden_size', net_params.get('hidden_size', 512)))
    buffer_size = int(train_params.get('buffer_size', 100000))
    warmup_steps = int(train_params.get('warmup_steps', 10000))
    update_every = int(train_params.get('update_every', 1))
    updates_per_step = int(max(train_params.get('updates_per_step', 1), 1))
    policy_delay = int(max(train_params.get('policy_delay', 1), 1))
    exploration_std_init = float(train_params.get('exploration_std_init', 0.10))
    exploration_std_final = float(train_params.get('exploration_std_final', 0.02))
    exploration_std_decay_steps = int(max(train_params.get('exploration_std_decay_steps', max(1, total_steps // 2)), 1))
    save_every = int(train_params.get('save_every', 50000))
    plot_every = int(train_params.get('plot_every', max(1, save_every // 2)))
    heartbeat_every = int(train_params.get('heartbeat_every', 2000))
    ema_span = int(max(2, int(getattr(args, "ema_span", 96))))

    
    is_a2c = (str(algo_name).upper() == "A2C")
    if algo_name == 'DDPG':
        agent = DDPGAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            hidden=hidden_size,
            buffer_size=buffer_size,
        )
    elif algo_name == 'SAC':
        agent = SACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            lr=lr_actor,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            hidden=hidden_size,
            buffer_size=buffer_size,
        )
    elif algo_name == 'TD3':
        agent = TD3Agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            policy_delay=policy_delay,
            hidden=hidden_size,
            buffer_size=buffer_size,
        )
    elif algo_name == 'A2C':
        agent = A2CAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            hidden=hidden_size,
            rollout_steps=batch_size,
            value_coef=0.5,
            entropy_coef=0.01,
            grad_clip=1.0,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    
    reward_history = []
    actor_loss_history = []
    critic_loss_history = []
    q_value_history = []
    steps_history = []
    update_steps_history = []
    episode_history = []
    update_episode_history = []

    
    episode_rewards = deque(maxlen=100)
    episode_count = 0
    best_mean_reward = float("-inf")

    
    global_step = 0
    start_time = time.time()
    last_hb_time = start_time

    obs = env.reset()
    episode_reward = 0.0
    episode_steps = 0

    print(f"[HB][train] Starting {algo_name} training: total_steps={total_steps}")
    print(
        f"[HB][train][fairness] hidden={hidden_size} updates_per_step={updates_per_step} "
        f"lr_actor={lr_actor:.6g} lr_critic={lr_critic:.6g} policy_delay={policy_delay} "
        f"explore_std=({exploration_std_init:.4f}->{exploration_std_final:.4f}, decay={exploration_std_decay_steps})",
        flush=True,
    )

    def _save_ckpt_with_history(path: Path) -> None:
        """ckptagent.load"""
        try:
            agent.save(path)
            obj = safe_torch_load(path, map_location="cpu")
            if isinstance(obj, dict):
                obj["training_history"] = {
                    "episodes": list(episode_history),
                    "rewards": list(reward_history),
                }
                torch.save(obj, path)
        except Exception as e:
            print(f"[HB][train][{algo_name}][WARN] save_ckpt_with_history failed: {type(e).__name__}: {e}", flush=True)

    def _flush_artifacts(tag: str):
        ckpt_dir = run_dir / CKPT_DIR_NAME
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            _save_ckpt_with_history(ckpt_dir / "agent_latest.pt")
            _save_ckpt_with_history(ckpt_dir / f"agent_{tag}_step{global_step}.pt")
        except Exception as e:
            print(f"[HB][train][{algo_name}][WARN] flush ckpt failed: {e}")

        try:
            if reward_history:
                plot_3drl_training_curves(
                    run_dir, algo_name,
                    reward_history, actor_loss_history, critic_loss_history, q_value_history,
                    episode_history, update_episode_history, ema_span=ema_span
                )

        except Exception as e:
            print(f"[HB][train][{algo_name}][WARN] flush plot failed: {e}")

    try:
        while global_step < total_steps:
            
            if global_step % 500 == 0:
                pct = 100.0 * global_step / max(1, total_steps)
                print(f"[HB][{algo_name}] step={global_step}/{total_steps} ({pct:.2f}%)", end="\r", flush=True)

            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)

            a2c_aux = {"log_prob": 0.0, "value": 0.0}
            if global_step < warmup_steps:
                action = np.random.randn(act_dim).astype(np.float32)
                action = np.clip(action, -1, 1)
            else:
                if is_a2c:
                    action_out, a2c_aux = agent.get_action(obs_t, deterministic=False, return_aux=True)
                elif algo_name == "SAC":
                    action_out = agent.get_action(obs_t, deterministic=False)
                else:
                    action_out = agent.get_action(obs_t, add_noise=True)

                if isinstance(action_out, torch.Tensor):
                    action = action_out.detach().cpu().numpy()
                else:
                    action = np.asarray(action_out)

                action = action.squeeze(0).astype(np.float32, copy=False)

            
            action_env = np.asarray(action, dtype=np.float32).copy()
            next_obs, reward, done, info = env.step(action_env)

            
            if is_a2c:
                agent.add_transition(
                    obs=np.asarray(obs, dtype=np.float32),
                    action=np.asarray(action, dtype=np.float32),
                    reward=float(reward),
                    done=bool(done),
                    value=float(a2c_aux.get("value", 0.0)),
                    log_prob=float(a2c_aux.get("log_prob", 0.0)),
                    next_obs=np.asarray(next_obs, dtype=np.float32),
                )
            else:
                agent.buffer.add(obs, np.asarray(action, dtype=np.float32), reward, next_obs, float(done))

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            global_step += 1

            
            if global_step >= warmup_steps and hasattr(agent, "set_exploration"):
                explore_std_now = _linear_schedule(
                    exploration_std_init,
                    exploration_std_final,
                    global_step - warmup_steps,
                    exploration_std_decay_steps,
                )
                try:
                    agent.set_exploration(float(explore_std_now))
                except Exception:
                    pass

            
            if global_step >= warmup_steps and global_step % update_every == 0:
                update_stats = {}
                update_loops = 1 if is_a2c else int(updates_per_step)
                for _ in range(int(max(1, update_loops))):
                    stats_i = agent.update(batch_size)
                    if stats_i:
                        update_stats = stats_i

                if update_stats:
                    update_steps_history.append(global_step)
                    update_episode_history.append(episode_count)
                    
                    actor_loss_v = update_stats.get('actor_loss', np.nan)
                    actor_loss_history.append(float(actor_loss_v))
                    critic_loss_history.append(update_stats.get('critic_loss', 0.0))

                    if algo_name == 'DDPG':
                        q_value_history.append(update_stats.get('q_value', 0.0))
                    elif algo_name == 'SAC':
                        q_value_history.append((update_stats.get('q1_mean', 0.0) + update_stats.get('q2_mean', 0.0)) / 2)
                    elif algo_name == 'TD3':
                        q_value_history.append((update_stats.get('q1_mean', 0.0) + update_stats.get('q2_mean', 0.0)) / 2)
                    elif algo_name == 'A2C':
                        q_value_history.append(update_stats.get('value_mean', 0.0))

            
            if done or episode_steps >= env.cfg.T:
                episode_rewards.append(episode_reward)
                episode_count += 1

                reward_history.append(episode_reward)
                steps_history.append(global_step)
                episode_history.append(episode_count)

                obs = env.reset()
                episode_reward = 0.0
                episode_steps = 0

                
                mean_reward_now = float(np.mean(episode_rewards)) if episode_rewards else float("-inf")
                if np.isfinite(mean_reward_now) and mean_reward_now > float(best_mean_reward):
                    best_mean_reward = float(mean_reward_now)
                    ckpt_dir = run_dir / CKPT_DIR_NAME
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    best_path = ckpt_dir / "agent_best.pt"
                    _save_ckpt_with_history(best_path)
                    try:
                        with open(run_dir / "best_ckpt.json", "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "best_step": int(global_step),
                                    "best_episode": int(episode_count),
                                    "best_mean_reward_last_100": float(best_mean_reward),
                                    "best_ckpt_path": str(best_path),
                                },
                                f,
                                indent=2,
                                ensure_ascii=False,
                            )
                    except Exception as e:
                        print(f"[HB][train][{algo_name}][WARN]  best_ckpt.json : {type(e).__name__}: {e}", flush=True)

            
            if global_step % heartbeat_every == 0 or global_step >= total_steps:
                current_time = time.time()
                elapsed = current_time - start_time
                elapsed_since_hb = current_time - last_hb_time
                last_hb_time = current_time

                steps_per_sec = heartbeat_every / elapsed_since_hb if elapsed_since_hb > 0 else 0
                eta_seconds = (total_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0

                mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0

                print(f"\n{'=' * 80}")
                print(f"[HB][train][{algo_name}] Step {global_step}/{total_steps} ({100 * global_step / total_steps:.1f}%)")
                print(f"[HB][train][{algo_name}] Episodes: {episode_count}")
                print(f"[HB][train][{algo_name}] ---")
                print(
                    f"[HB][train][{algo_name}] Time: {fmt_hms(elapsed)} | Speed: {steps_per_sec:.1f} steps/s | ETA: {fmt_hms(eta_seconds)}")
                print(f"[HB][train][{algo_name}] ---")
                print(f"[HB][train][{algo_name}] Performance:")
                print(f"[HB][train][{algo_name}]   Mean Reward (last 100 ep): {mean_reward:.4f}")

                if actor_loss_history:
                    print(f"[HB][train][{algo_name}]   Actor Loss:  {actor_loss_history[-1]:.4f}")
                if critic_loss_history:
                    print(f"[HB][train][{algo_name}]   Critic Loss: {critic_loss_history[-1]:.4f}")
                if q_value_history:
                    print(f"[HB][train][{algo_name}]   Q-Value:     {q_value_history[-1]:.4f}")

                print(f"{'=' * 80}\n")

            
            if global_step % save_every == 0 or global_step >= total_steps:
                ckpt_dir = run_dir / CKPT_DIR_NAME
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                ckpt_path = ckpt_dir / f"agent_step{global_step}.pt"
                _save_ckpt_with_history(ckpt_path)
                print(f"[HB][train][{algo_name}] Saved checkpoint: {ckpt_path}")

                latest_path = ckpt_dir / "agent_latest.pt"
                _save_ckpt_with_history(latest_path)

            
            if global_step % plot_every == 0 or global_step >= total_steps:
                if reward_history:
                    plot_3drl_training_curves(
                        run_dir, algo_name,
                        reward_history, actor_loss_history, critic_loss_history, q_value_history,
                        episode_history, update_episode_history, ema_span=ema_span
                    )

    except KeyboardInterrupt:
        print(f"\n[HB][train][{algo_name}] KeyboardInterrupt. Flushing ckpt + figs...", flush=True)
        _flush_artifacts("interrupt")
        raise
    except Exception as e:
        print(f"\n[HB][train][{algo_name}] Exception: {type(e).__name__}: {e}. Flushing...", flush=True)
        _flush_artifacts("exception")
        raise
    finally:
        try:
            _flush_artifacts("finally")
        except Exception:
            pass
    
    final_ckpt = run_dir / CKPT_DIR_NAME / "agent_final.pt"
    (run_dir / CKPT_DIR_NAME).mkdir(parents=True, exist_ok=True)
    _save_ckpt_with_history(final_ckpt)
    print(f"[HB][train][{algo_name}] Saved final checkpoint: {final_ckpt}")

    
    meta_path = run_dir / "meta_train.json"
    with open(meta_path, 'w') as f:
        json.dump({
            'algo': algo_name,
            'seed': int(seed),
            'ema_span': int(ema_span),
            'total_steps': global_step,
            'total_episodes': episode_count,
            'obs_dim': obs_dim,
            'act_dim': act_dim,
            'hidden_size': int(hidden_size),
            'lr_actor': float(lr_actor),
            'lr_critic': float(lr_critic),
            'batch_size': int(batch_size),
            'warmup_steps': int(warmup_steps),
            'update_every': int(update_every),
            'updates_per_step': int(updates_per_step),
            'policy_delay': int(policy_delay),
            'exploration_std_init': float(exploration_std_init),
            'exploration_std_final': float(exploration_std_final),
            'exploration_std_decay_steps': int(exploration_std_decay_steps),
            'final_reward': reward_history[-1] if reward_history else 0.0,
            'mean_reward_last_100': np.mean(list(episode_rewards)) if episode_rewards else 0.0,
            'best_mean_reward_last_100': float(best_mean_reward) if np.isfinite(best_mean_reward) else None,
        }, f, indent=2)

    print(f"[HB][train][{algo_name}] Training complete! Total time: {fmt_hms(time.time() - start_time)}")




# =========================

# =========================
def main():
    parser = argparse.ArgumentParser(description='3DRL Baseline Training for CoMP-RIS')
    parser.add_argument('--algo', type=str, required=True, choices=['ddpg', 'sac', 'td3', 'a2c'],
                        help='Algorithm to train (ddpg/sac/td3/a2c)')
    parser.add_argument('--env-yaml', type=str, default='configs/PhaseZ4/env_phaseZ4_3drl.yaml',
                        help='Path to environment config (default: PhaseZ4 3DRL fair env)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Run name suffix (auto-generated if not provided)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed override for fair comparison')
    parser.add_argument('--ema-span', type=int, default=96,
                        help='EMA span for training curves (default: 96)')
    
    parser.add_argument('--total-steps', type=int, default=None,
                        help='Override total training steps for fair budget alignment')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Override hidden size for all 3DRL baselines')
    parser.add_argument('--updates-per-step', type=int, default=None,
                        help='Override updates per environment step (UTD ratio)')
    parser.add_argument('--policy-delay', type=int, default=None,
                        help='Override TD3 policy delay (classic default is 2)')
    parser.add_argument('--exploration-std-init', type=float, default=None,
                        help='Override exploration std initial value')
    parser.add_argument('--exploration-std-final', type=float, default=None,
                        help='Override exploration std final value')
    parser.add_argument('--exploration-std-decay-steps', type=int, default=None,
                        help='Override exploration std decay steps')

    args = parser.parse_args()

    algo_name = args.algo.upper()
    print(f"[HB][train] Training {algo_name} algorithm")

    train_3drl(args, algo_name)


if __name__ == "__main__":
    main()

