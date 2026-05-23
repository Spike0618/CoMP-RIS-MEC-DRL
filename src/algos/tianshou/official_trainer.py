"""
P4TianshouCollector + OnpolicyTrainer


-  scripts/train_ppo.py
- run
"""

from __future__ import annotations

import atexit
import csv
import json
import math
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer

from src.algos.tianshou.ppo_config import create_ppo_policy
from src.agentic.meta_controller import MetaController, create_meta_controller
from src.agentic.meta_state_builder import build_meta_state
from src.envs.comp_ris_env_gym import CompRISEnvGym
from src.envs.comp_ris_env_joint import CompRISEnvJoint, JointEnvConfig
from src.utils.hb import fmt_hms
from src.utils.obs_normalizer import ObsNormalizer


CKPT_DIR_NAME = "checkpoints"  
REPO_ROOT = Path(__file__).resolve().parents[3]
_WARN_ONCE_KEYS: set[str] = set()


OptimizerBundle = Union[torch.optim.Optimizer, Tuple[torch.optim.Optimizer, torch.optim.Optimizer]]


def _warn_once(key: str, message: str) -> None:
    """"""
    k = str(key)
    if k in _WARN_ONCE_KEYS:
        return
    _WARN_ONCE_KEYS.add(k)
    print(f"[HB][train][WARN][{k}] {message}", flush=True)


def _linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    """"""
    step = int(max(step, 0))
    decay_steps = int(max(decay_steps, 1))
    if step >= decay_steps:
        return float(end)
    t = float(step) / float(decay_steps)
    return float(start + (end - start) * t)


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """
     ties scipy.stats.rankdata(method="average") 
     numpy
    """
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n <= 0:
        return x.astype(np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty((n,), dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        xi = float(x[order[i]])
        while j < n and float(x[order[j]]) == xi:
            j += 1
        r = 0.5 * float(i + j - 1) + 1.0
        ranks[order[i:j]] = r
        i = j
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman  scipy """
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    y0 = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(min(x0.size, y0.size))
    if n <= 1:
        return float("nan")
    x0 = x0[:n]
    y0 = y0[:n]
    m = np.isfinite(x0) & np.isfinite(y0)
    if int(np.sum(m)) <= 1:
        return float("nan")
    xr = _rankdata_average(x0[m])
    yr = _rankdata_average(y0[m])
    xr = xr - float(np.mean(xr))
    yr = yr - float(np.mean(yr))
    den = float(np.sqrt(np.sum(xr * xr) * np.sum(yr * yr)))
    if den <= 1e-12:
        return float("nan")
    return float(np.sum(xr * yr) / den)


def _torch_load_ckpt(path: Any, map_location: Any, *, prefer_weights_only: bool = True) -> Any:
    """
     checkpoint  torch  weights_only 
    - prefer_weights_only=True
    - prefer_weights_only=False optimizer
    """
    path_s = str(path)

    def _try_load(flag: bool) -> Any:
        try:
            return torch.load(path_s, map_location=map_location, weights_only=flag)  # type: ignore[call-arg]
        except TypeError:
            return None
        except Exception:
            return None

    if bool(prefer_weights_only):
        obj = _try_load(True)
        if obj is not None:
            return obj
        obj = _try_load(False)
        if obj is not None:
            return obj
    else:
        obj = _try_load(False)
        if obj is not None:
            return obj
        obj = _try_load(True)
        if obj is not None:
            return obj

    try:
        return torch.load(path_s, map_location=map_location)
    except Exception:
        return None


def _try_load_obs_normalizer_from_ckpt(
    obs_normalizer: Optional[ObsNormalizer],
    ckpt_path: str,
    *,
    log_tag: str,
) -> None:
    """checkpointobs_normalizer"""
    if obs_normalizer is None:
        return
    try:
        ckpt_dir = Path(ckpt_path).resolve().parent
        ckpt_name = Path(ckpt_path).name.lower()
        ckpt_stem = Path(ckpt_path).stem.lower()
        step_target: Optional[int] = None
        for key in ("step_", "step-", "step"):
            pos = ckpt_name.find(key)
            if pos < 0:
                continue
            j = pos + len(key)
            digs: List[str] = []
            while j < len(ckpt_name) and ckpt_name[j].isdigit():
                digs.append(ckpt_name[j])
                j += 1
            if digs:
                step_target = int("".join(digs))
                break

        cand: List[Path] = []
        seen_npz: set[str] = set()

        def _add_npz(p: Path) -> None:
            try:
                k = str(p.resolve())
            except Exception:
                k = str(p)
            if (k not in seen_npz) and p.exists() and p.is_file():
                seen_npz.add(k)
                cand.append(p)

        if "best" in ckpt_name:
            _add_npz(ckpt_dir / "normalizer_best.npz")
        elif step_target is not None:
            _add_npz(ckpt_dir / f"normalizer_step{int(step_target)}.npz")
            step_le: List[Tuple[int, Path]] = []
            for q in ckpt_dir.glob("normalizer_step*.npz"):
                nm = q.name.lower()
                if not (nm.startswith("normalizer_step") and nm.endswith(".npz")):
                    continue
                num = nm[len("normalizer_step"):-len(".npz")]
                if not str(num).isdigit():
                    continue
                s = int(num)
                if s <= int(step_target):
                    step_le.append((s, q))
            if step_le:
                step_le.sort(key=lambda kv: kv[0])
                _add_npz(step_le[-1][1])

        tag = ckpt_stem
        for pref in ("ckpt_", "agent_"):
            if tag.startswith(pref) and len(tag) > len(pref):
                tag = tag[len(pref):]
                break
        if tag:
            _add_npz(ckpt_dir / f"normalizer_{tag}.npz")

        if "best" in ckpt_name:
            _add_npz(ckpt_dir / "normalizer_final.npz")
            _add_npz(ckpt_dir / "normalizer_latest.npz")
            _add_npz(ckpt_dir / "normalizer_traj_only.npz")
        elif "final" in ckpt_name:
            _add_npz(ckpt_dir / "normalizer_final.npz")
            _add_npz(ckpt_dir / "normalizer_latest.npz")
            _add_npz(ckpt_dir / "normalizer_traj_only.npz")
        elif step_target is not None:
            _add_npz(ckpt_dir / "normalizer_final.npz")
            _add_npz(ckpt_dir / "normalizer_latest.npz")
            _add_npz(ckpt_dir / "normalizer_traj_only.npz")
        else:
            _add_npz(ckpt_dir / "normalizer_latest.npz")
            _add_npz(ckpt_dir / "normalizer_traj_only.npz")
            _add_npz(ckpt_dir / "normalizer_final.npz")

        for npz_path in cand:
            data = np.load(npz_path, allow_pickle=False)
            obs_normalizer.set_stats(
                {
                    "count": int(data["count"]),
                    "mean": np.asarray(data["mean"], dtype=np.float64),
                    "var": np.asarray(data["var"], dtype=np.float64),
                }
            )
            print(f"[HB][train][{log_tag}] loaded obs_normalizer={npz_path}", flush=True)
            return
    except Exception as exc:
        print(f"[HB][train][{log_tag}][WARN] load obs_normalizer failed: {type(exc).__name__}: {exc}", flush=True)


def _split_optimizer_bundle(optimizer: OptimizerBundle) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
    """ (actor_optimizer, critic_optimizer_or_none)"""
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer, None
    if isinstance(optimizer, (tuple, list)) and len(optimizer) >= 2:
        opt_a, opt_c = optimizer[0], optimizer[1]
        if isinstance(opt_a, torch.optim.Optimizer) and isinstance(opt_c, torch.optim.Optimizer):
            return opt_a, opt_c
    raise TypeError(f"{type(optimizer).__name__}")


def _optimizer_export_state(optimizer: OptimizerBundle) -> Any:
    """state_dict{actor,critic}"""
    opt_a, opt_c = _split_optimizer_bundle(optimizer)
    if opt_c is None:
        return opt_a.state_dict()
    return {
        "actor": opt_a.state_dict(),
        "critic": opt_c.state_dict(),
    }


def _optimizer_load_state(optimizer: OptimizerBundle, state_obj: Any) -> None:
    """
    state_dict{actor,critic}

    
    -  actoractor critic
    -  actor critic
    """

    def _try_sync_lr_from_state(opt_dst: torch.optim.Optimizer, cand_state: Any) -> bool:
        """
         resume 
        """
        if not isinstance(cand_state, dict):
            return False
        pg_src = cand_state.get("param_groups")
        if not isinstance(pg_src, list) or len(pg_src) <= 0:
            return False
        try:
            lr_src = float(pg_src[0].get("lr", float("nan")))
        except Exception:
            return False
        if not np.isfinite(lr_src):
            return False
        for pg in opt_dst.param_groups:
            pg["lr"] = float(lr_src)
        return True

    opt_a, opt_c = _split_optimizer_bundle(optimizer)
    if opt_c is None:
        if isinstance(state_obj, dict) and ("actor" in state_obj or "critic" in state_obj):
            cands = []
            if "actor" in state_obj:
                cands.append(("actor", state_obj.get("actor")))
            if "critic" in state_obj:
                cands.append(("critic", state_obj.get("critic")))
            errs: List[str] = []
            for tag, cand in cands:
                try:
                    opt_a.load_state_dict(cand)
                    return
                except Exception as exc:
                    errs.append(f"{tag}:{type(exc).__name__}")
            for tag, cand in cands:
                if _try_sync_lr_from_state(opt_a, cand):
                    _warn_once(
                        "optimizer_load_single_from_dual_lr_only",
                        f"optimizer{tag}",
                    )
                    return
            raise ValueError(f"{', '.join(errs) if errs else ''}")
        try:
            opt_a.load_state_dict(state_obj)
        except Exception as exc:
            if _try_sync_lr_from_state(opt_a, state_obj):
                _warn_once(
                    "optimizer_load_single_lr_only",
                    "",
                )
                return
            raise exc
        return

    if isinstance(state_obj, dict) and ("actor" in state_obj or "critic" in state_obj):
        if "actor" in state_obj:
            opt_a.load_state_dict(state_obj["actor"])
        elif "critic" in state_obj:
            opt_a.load_state_dict(state_obj["critic"])

        if "critic" in state_obj:
            opt_c.load_state_dict(state_obj["critic"])
        elif "actor" in state_obj:
            
            opt_c.load_state_dict(state_obj["actor"])
        return

    
    try:
        opt_a.load_state_dict(state_obj)
    except Exception as exc:
        if _try_sync_lr_from_state(opt_a, state_obj):
            _warn_once(
                "optimizer_load_dual_actor_lr_only",
                "actor",
            )
        else:
            raise exc
    try:
        opt_c.load_state_dict(state_obj)
    except Exception:
        
        if _try_sync_lr_from_state(opt_c, state_obj):
            _warn_once(
                "optimizer_load_dual_critic_lr_only",
                "critic",
            )


def _optimizer_set_lr(optimizer: OptimizerBundle, *, lr_actor: float, lr_critic: Optional[float] = None) -> None:
    """ actor/critic"""
    opt_a, opt_c = _split_optimizer_bundle(optimizer)
    for pg in opt_a.param_groups:
        pg["lr"] = float(lr_actor)
    if opt_c is not None:
        lr_c = float(lr_actor if lr_critic is None else lr_critic)
        for pg in opt_c.param_groups:
            pg["lr"] = float(lr_c)


def _optimizer_actor_lr(optimizer: OptimizerBundle, default: float = float("nan")) -> float:
    """ actor """
    try:
        opt_a, _ = _split_optimizer_bundle(optimizer)
        if len(opt_a.param_groups) > 0:
            return float(opt_a.param_groups[0].get("lr", default))
    except Exception:
        pass
    return float(default)


def _optimizer_actor_only(optimizer: OptimizerBundle) -> torch.optim.Optimizer:
    """ actor BC"""
    opt_a, _ = _split_optimizer_bundle(optimizer)
    return opt_a


class CollectorPreprocess:
    """
    Collector.preprocess_fn
    - update=Trueupdate=False
    - infopaper_cost/vio/improvebuffer
    - reward info paper_costrewardLagrange
    """

    def __init__(
        self,
        *,
        obs_normalizer: Optional[ObsNormalizer],
        update_obs: bool,
        update_improve_z: bool = True,
        reward_override: str,
        
        
        
        
        paper_reward_map: str,
        paper_reward_bias: float,
        paper_reward_scale: float,
        paper_reward_tanh_s: float,
        paper_reward_clip_c: float,
        
        improve_mode: str,
        rel_improve_eps: float,
        improve_z_eps: float,
        paper_vio_gating_power: float,
        paper_vio_penalty: float,
        reward_clip_range: float,
        vio_scale: float,
        lagrange_enabled: bool,
        lagrange_init: float,
        
        
        
        
        
        
        enable_comp_ris_shaping: bool = False,
        comp_weight: float = 0.0,
        ris_weight: float = 0.0,
        shaping_ratio: float = 0.10,
        # H12 v2: per-step reward normalization (Welford online mean/std)
        reward_step_normalize: bool = False,
    ):
        self.obs_normalizer = obs_normalizer
        self.update_obs = bool(update_obs)
        
        self.update_improve_z = bool(update_improve_z)

        self.reward_override = str(reward_override or "env").strip().lower()
        if self.reward_override not in ("env", "paper"):
            self.reward_override = "env"

        
        self.paper_reward_map = str(paper_reward_map or "tanh").strip().lower()
        if self.paper_reward_map not in ("tanh", "clip"):
            self.paper_reward_map = "tanh"
        self.paper_reward_bias = float(paper_reward_bias)
        self.paper_reward_scale = float(paper_reward_scale)
        self.paper_reward_tanh_s = float(max(paper_reward_tanh_s, 1e-6))
        self.paper_reward_clip_c = float(max(paper_reward_clip_c, 1e-6))

        
        
        
        
        
        self.improve_mode = str(improve_mode or "paper_improve").strip().lower()
        if self.improve_mode not in ("paper_improve", "rel_improve", "improve_z", "neg_paper_cost"):
            self.improve_mode = "paper_improve"
        self.rel_improve_eps = float(max(rel_improve_eps, 0.0))
        self.improve_z_eps = float(max(improve_z_eps, 0.0))
        
        self._impz_n = 0
        self._impz_mean = 0.0
        self._impz_m2 = 0.0
        
        self.paper_vio_gating_power = float(max(paper_vio_gating_power, 0.0))
        
        
        
        
        self.paper_vio_penalty = float(max(paper_vio_penalty, 0.0))
        
        self.reward_clip_range = float(max(reward_clip_range, 0.0))
        
        self.vio_scale = float(max(vio_scale, 0.0))

        self.lagrange_enabled = bool(lagrange_enabled)
        self.lambda_v = float(max(lagrange_init, 0.0))

        
        self.enable_comp_ris_shaping = bool(enable_comp_ris_shaping)
        self.comp_weight = float(max(comp_weight, 0.0))
        self.ris_weight = float(max(ris_weight, 0.0))
        
        self.shaping_ratio = float(np.clip(float(shaping_ratio), 0.0, 0.20))
        if (self.comp_weight <= 0.0) and (self.ris_weight <= 0.0):
            
            self.enable_comp_ris_shaping = False

        
        
        
        self.reward_step_normalize = bool(reward_step_normalize)
        self._rew_norm_n: int = 0
        self._rew_norm_mean: float = 0.0
        self._rew_norm_m2: float = 0.0
        self._rew_norm_warmup: int = 100  

        
        
        
        
        self.reset_rollout_stats()

    def set_lambda_v(self, v: float) -> None:
        self.lambda_v = float(max(v, 0.0))

    def reset_rollout_stats(self) -> None:
        """collectcollect"""
        self._roll_paper_cost: List[float] = []
        self._roll_baseline_balanced: List[float] = []
        self._roll_baseline_greedy_delay: List[float] = []
        self._roll_baseline_greedy_energy: List[float] = []
        
        self._roll_paper_improve: List[float] = []
        
        self._roll_improve: List[float] = []
        self._roll_vio_cnt: List[float] = []
        self._roll_vio_metric: List[float] = []
        self._roll_coverage_fraction: List[float] = []
        self._roll_coverage_margin: List[float] = []
        self._roll_collision_margin: List[float] = []
        
        self._roll_reward_paper_step: List[float] = []
        self._roll_reward_paper_abs_step: List[float] = []
        self._roll_reward_paper_delta_step: List[float] = []
        self._roll_reward_paper_adv_step: List[float] = []
        self._roll_paper_delta_norm_step: List[float] = []
        self._roll_paper_adv_norm_step: List[float] = []
        self._roll_constraint_signal_step: List[float] = []
        self._roll_lambda_v_effective: List[float] = []
        self._roll_reward_constraint_step: List[float] = []
        self._roll_reward_misc_step: List[float] = []
        self._roll_reward_proximity_step: List[float] = []  
        self._roll_shaping_gate: List[float] = []
        self._roll_beta_comp_t: List[float] = []
        self._roll_beta_ris_t: List[float] = []
        self._roll_beta_train_frac: List[float] = []
        
        self._roll_z4_gap_raw_step: List[float] = []
        self._roll_z4_gap_step_clip: List[float] = []
        self._roll_z4_gap_step_ema: List[float] = []
        self._roll_z4_gap_step: List[float] = []
        self._roll_z4_gap_reward_step: List[float] = []
        self._roll_z4_gap_lambda_input: List[float] = []
        self._roll_z4_gap_lambda_sched: List[float] = []
        self._roll_z4_gap_lambda_cap: List[float] = []
        self._roll_z4_gap_lambda_eff: List[float] = []
        self._roll_z4_gap_grad_ratio_eff: List[float] = []
        self._roll_z4_gap_shape_guard_trigger: List[float] = []
        
        self._roll_t6_main_weight: List[float] = []
        self._roll_t6_potential_scale: List[float] = []
        self._roll_t6_traj_scale: List[float] = []
        self._roll_t6_anneal_ratio: List[float] = []
        self._roll_t6_violation_ema: List[float] = []
        self._roll_t6_pid_frozen: List[float] = []
        self._roll_comp_bonus_raw: List[float] = []
        self._roll_ris_bonus_raw: List[float] = []
        
        self._roll_reward_raw: List[float] = []
        
        self._roll_env_shaping: List[float] = []
        self._roll_total_shaping: List[float] = []
        self._roll_total_shaping_all: List[float] = []
        self._roll_comp_bonus: List[float] = []
        self._roll_ris_bonus: List[float] = []
        
        self._roll_comp_gain_ratio: List[float] = []
        self._roll_ris_gain_ratio: List[float] = []
        
        self._roll_comp_enable_rate: List[float] = []
        self._roll_service_switch_count: List[float] = []
        self._roll_theta_entropy: List[float] = []
        self._roll_comp_score_temp_runtime: List[float] = []
        self._roll_theta_mode_effective: List[str] = []
        
        self._roll_reward_total_step: List[float] = []
        self._roll_paper_cost_step: List[float] = []
        self._roll_contract_residual: List[float] = []
        self._roll_reward_total_ep_sum: List[float] = []
        self._roll_paper_cost_ep_sum: List[float] = []
        
        self._roll_reward_paper_ep_sum: List[float] = []
        self._roll_reward_shaping_ep_sum: List[float] = []
        self._roll_reward_constraint_ep_sum: List[float] = []
        self._roll_reward_misc_ep_sum: List[float] = []
        self._roll_reward_proximity_ep_sum: List[float] = []
        self._roll_paper_cost_ep_mean: List[float] = []
        self._roll_vio_cnt_ep_mean: List[float] = []
        self._roll_improve_ep_mean: List[float] = []
        self._roll_total_shaping_all_ep_mean: List[float] = []
        self._roll_comp_bonus_ep_mean: List[float] = []
        self._roll_ris_bonus_ep_mean: List[float] = []
        self._ep_running_reward_sum: Dict[int, float] = {}
        self._ep_running_paper_cost_sum: Dict[int, float] = {}
        self._ep_running_reward_paper_sum: Dict[int, float] = {}
        self._ep_running_reward_shaping_sum: Dict[int, float] = {}
        self._ep_running_reward_constraint_sum: Dict[int, float] = {}
        self._ep_running_reward_misc_sum: Dict[int, float] = {}
        self._ep_running_reward_proximity_sum: Dict[int, float] = {}
        self._ep_running_cost_sum: Dict[int, float] = {}
        self._ep_running_vio_sum: Dict[int, float] = {}
        self._ep_running_improve_sum: Dict[int, float] = {}
        self._ep_running_shaping_sum: Dict[int, float] = {}
        self._ep_running_comp_sum: Dict[int, float] = {}
        self._ep_running_ris_sum: Dict[int, float] = {}
        self._ep_running_step_cnt: Dict[int, int] = {}

    def get_rollout_stats(self) -> Dict[str, float]:
        """rollout0/NaN"""
        def _mean(x: List[float]) -> float:
            if not x:
                return float("nan")
            arr = np.asarray(x, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size <= 0:
                return float("nan")
            return float(np.mean(arr))

        def _p(x: List[float], q: float) -> float:
            if not x:
                return float("nan")
            arr = np.asarray(x, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size <= 0:
                return float("nan")
            return float(np.percentile(arr, q))

        def _max_abs(x: List[float]) -> float:
            if not x:
                return float("nan")
            arr = np.asarray(x, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size <= 0:
                return float("nan")
            return float(np.max(np.abs(arr)))

        cost_mean = _mean(self._roll_paper_cost)
        base_mean = _mean(self._roll_baseline_balanced)
        base_delay_mean = _mean(self._roll_baseline_greedy_delay)
        base_energy_mean = _mean(self._roll_baseline_greedy_energy)
        paper_improve_mean = _mean(self._roll_paper_improve)
        paper_improve_vs_balanced_mean = _mean(
            [
                float(b) - float(c)
                for b, c in zip(self._roll_baseline_balanced, self._roll_paper_cost)
                if np.isfinite(float(b)) and np.isfinite(float(c))
            ]
        )
        paper_improve_vs_greedy_delay_mean = _mean(
            [
                float(b) - float(c)
                for b, c in zip(self._roll_baseline_greedy_delay, self._roll_paper_cost)
                if np.isfinite(float(b)) and np.isfinite(float(c))
            ]
        )
        paper_improve_vs_greedy_energy_mean = _mean(
            [
                float(b) - float(c)
                for b, c in zip(self._roll_baseline_greedy_energy, self._roll_paper_cost)
                if np.isfinite(float(b)) and np.isfinite(float(c))
            ]
        )
        improve_mean = _mean(self._roll_improve)
        vio_cnt_mean = _mean(self._roll_vio_cnt)
        vio_metric_mean = _mean(self._roll_vio_metric)
        cov_frac_mean = _mean(self._roll_coverage_fraction)
        cov_margin_mean = _mean(self._roll_coverage_margin)
        col_margin_mean = _mean(self._roll_collision_margin)
        reward_paper_step_mean = _mean(self._roll_reward_paper_step)
        reward_paper_abs_step_mean = _mean(self._roll_reward_paper_abs_step)
        reward_paper_delta_step_mean = _mean(self._roll_reward_paper_delta_step)
        reward_paper_adv_step_mean = _mean(self._roll_reward_paper_adv_step)
        paper_delta_norm_step_mean = _mean(self._roll_paper_delta_norm_step)
        paper_adv_norm_step_mean = _mean(self._roll_paper_adv_norm_step)
        constraint_signal_step_mean = _mean(self._roll_constraint_signal_step)
        lambda_v_effective_mean = _mean(self._roll_lambda_v_effective)
        reward_constraint_step_mean = _mean(self._roll_reward_constraint_step)
        reward_misc_step_mean = _mean(self._roll_reward_misc_step)
        reward_proximity_step_mean = _mean(self._roll_reward_proximity_step)
        shaping_gate_mean = _mean(self._roll_shaping_gate)
        beta_comp_t_mean = _mean(self._roll_beta_comp_t)
        beta_ris_t_mean = _mean(self._roll_beta_ris_t)
        beta_train_frac_mean = _mean(self._roll_beta_train_frac)
        z4_gap_raw_step_mean = _mean(self._roll_z4_gap_raw_step)
        z4_gap_step_clip_mean = _mean(self._roll_z4_gap_step_clip)
        z4_gap_step_ema_mean = _mean(self._roll_z4_gap_step_ema)
        z4_gap_step_mean = _mean(self._roll_z4_gap_step)
        z4_gap_reward_step_mean = _mean(self._roll_z4_gap_reward_step)
        z4_gap_lambda_input_mean = _mean(self._roll_z4_gap_lambda_input)
        z4_gap_lambda_sched_mean = _mean(self._roll_z4_gap_lambda_sched)
        z4_gap_lambda_cap_mean = _mean(self._roll_z4_gap_lambda_cap)
        z4_gap_lambda_eff_mean = _mean(self._roll_z4_gap_lambda_eff)
        z4_gap_grad_ratio_eff_mean = _mean(self._roll_z4_gap_grad_ratio_eff)
        z4_gap_shape_guard_trigger_mean = _mean(self._roll_z4_gap_shape_guard_trigger)
        t6_main_weight_mean = _mean(self._roll_t6_main_weight)
        t6_potential_scale_mean = _mean(self._roll_t6_potential_scale)
        t6_traj_scale_mean = _mean(self._roll_t6_traj_scale)
        t6_anneal_ratio_mean = _mean(self._roll_t6_anneal_ratio)
        t6_violation_ema_mean = _mean(self._roll_t6_violation_ema)
        t6_pid_frozen_mean = _mean(self._roll_t6_pid_frozen)
        comp_bonus_raw_mean = _mean(self._roll_comp_bonus_raw)
        ris_bonus_raw_mean = _mean(self._roll_ris_bonus_raw)
        env_shaping_mean = _mean(self._roll_env_shaping)
        total_shaping_mean = _mean(self._roll_total_shaping)
        total_shaping_all_mean = _mean(self._roll_total_shaping_all)
        comp_bonus_mean = _mean(self._roll_comp_bonus)
        ris_bonus_mean = _mean(self._roll_ris_bonus)
        comp_gain_ratio_mean = _mean(self._roll_comp_gain_ratio)
        ris_gain_ratio_mean = _mean(self._roll_ris_gain_ratio)
        comp_enable_rate_mean = _mean(self._roll_comp_enable_rate)
        service_switch_count_mean = _mean(self._roll_service_switch_count)
        theta_entropy_mean = _mean(self._roll_theta_entropy)
        comp_score_temp_runtime_mean = _mean(self._roll_comp_score_temp_runtime)
        reward_total_step_mean = _mean(self._roll_reward_total_step)
        paper_cost_step_mean = _mean(self._roll_paper_cost_step)
        reward_total_ep_sum_mean = _mean(self._roll_reward_total_ep_sum)
        paper_cost_ep_sum_mean = _mean(self._roll_paper_cost_ep_sum)
        reward_paper_ep_sum_mean = _mean(self._roll_reward_paper_ep_sum)
        reward_shaping_ep_sum_mean = _mean(self._roll_reward_shaping_ep_sum)
        reward_constraint_ep_sum_mean = _mean(self._roll_reward_constraint_ep_sum)
        reward_misc_ep_sum_mean = _mean(self._roll_reward_misc_ep_sum)
        reward_proximity_ep_sum_mean = _mean(self._roll_reward_proximity_ep_sum)
        contract_abs_err_mean = _mean(self._roll_contract_residual)
        contract_abs_err_max = _max_abs(self._roll_contract_residual)
        ep_sum_n = int(min(len(self._roll_reward_total_ep_sum), len(self._roll_paper_cost_ep_sum)))

        
        reward_raw_mean = _mean(self._roll_reward_raw)
        reward_raw_p05 = _p(self._roll_reward_raw, 5.0)
        reward_raw_p50 = _p(self._roll_reward_raw, 50.0)
        reward_raw_p95 = _p(self._roll_reward_raw, 95.0)
        reward_raw_std = float("nan")
        clip_hit_frac = float("nan")
        try:
            arr = np.asarray(self._roll_reward_raw, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                reward_raw_std = float(np.std(arr))
                cr = float(getattr(self, "reward_clip_range", 0.0))
                if cr > 0.0:
                    clip_hit_frac = float(np.mean(np.abs(arr) > cr))
        except Exception:
            reward_raw_std = float("nan")
            clip_hit_frac = float("nan")

        vio_any_frac = float("nan")
        try:
            if self._roll_vio_cnt:
                vc = np.asarray(self._roll_vio_cnt, dtype=np.float64)
                vc = vc[np.isfinite(vc)]
                if vc.size > 0:
                    vio_any_frac = float(np.mean(vc > 0.0))
        except Exception:
            vio_any_frac = float("nan")

        theta_mode_effective = ""
        theta_mode_effective_agent_frac = float("nan")
        try:
            mode_tags = [str(x).strip().lower() for x in self._roll_theta_mode_effective if str(x).strip()]
            if mode_tags:
                uniq, cnt = np.unique(np.asarray(mode_tags, dtype=object), return_counts=True)
                j = int(np.argmax(cnt))
                theta_mode_effective = str(uniq[j])
                theta_mode_effective_agent_frac = float(np.mean(np.asarray(mode_tags) == "agent"))
        except Exception:
            theta_mode_effective = ""
            theta_mode_effective_agent_frac = float("nan")

        return {
            "paper_cost_mean": float(cost_mean),
            "baseline_balanced_paper_cost_mean": float(base_mean),
            "baseline_greedy_delay_paper_cost_mean": float(base_delay_mean),
            "baseline_greedy_energy_paper_cost_mean": float(base_energy_mean),
            "paper_improve_mean": float(paper_improve_mean),
            "paper_improve_vs_balanced_mean": float(paper_improve_vs_balanced_mean),
            "paper_improve_vs_greedy_delay_mean": float(paper_improve_vs_greedy_delay_mean),
            "paper_improve_vs_greedy_energy_mean": float(paper_improve_vs_greedy_energy_mean),
            "improve_mean": float(improve_mean),
            "vio_cnt_mean": float(vio_cnt_mean),
            "vio_metric_mean": float(vio_metric_mean),
            "coverage_fraction_mean": float(cov_frac_mean),
            "coverage_margin_mean": float(cov_margin_mean),
            "collision_margin_mean": float(col_margin_mean),
            
            "reward_paper_step_mean": float(reward_paper_step_mean) if np.isfinite(reward_paper_step_mean) else 0.0,
            "reward_paper_abs_step_mean": float(reward_paper_abs_step_mean) if np.isfinite(reward_paper_abs_step_mean) else 0.0,
            "reward_paper_delta_step_mean": float(reward_paper_delta_step_mean) if np.isfinite(reward_paper_delta_step_mean) else 0.0,
            "reward_paper_adv_step_mean": float(reward_paper_adv_step_mean) if np.isfinite(reward_paper_adv_step_mean) else 0.0,
            "paper_delta_norm_step_mean": float(paper_delta_norm_step_mean) if np.isfinite(paper_delta_norm_step_mean) else 0.0,
            "paper_adv_norm_step_mean": float(paper_adv_norm_step_mean) if np.isfinite(paper_adv_norm_step_mean) else 0.0,
            "constraint_signal_step_mean": float(constraint_signal_step_mean) if np.isfinite(constraint_signal_step_mean) else 0.0,
            "lambda_v_effective_mean": float(lambda_v_effective_mean) if np.isfinite(lambda_v_effective_mean) else 0.0,
            "reward_constraint_step_mean": float(reward_constraint_step_mean) if np.isfinite(reward_constraint_step_mean) else 0.0,
            "reward_misc_step_mean": float(reward_misc_step_mean) if np.isfinite(reward_misc_step_mean) else 0.0,
            "reward_proximity_step_mean": float(reward_proximity_step_mean) if np.isfinite(reward_proximity_step_mean) else 0.0,
            "shaping_gate_mean": float(shaping_gate_mean) if np.isfinite(shaping_gate_mean) else 0.0,
            "beta_comp_t_mean": float(beta_comp_t_mean) if np.isfinite(beta_comp_t_mean) else 0.0,
            "beta_ris_t_mean": float(beta_ris_t_mean) if np.isfinite(beta_ris_t_mean) else 0.0,
            "beta_train_frac_mean": float(beta_train_frac_mean) if np.isfinite(beta_train_frac_mean) else 0.0,
            "z4_gap_raw_step_mean": float(z4_gap_raw_step_mean) if np.isfinite(z4_gap_raw_step_mean) else 0.0,
            "z4_gap_step_clip_mean": float(z4_gap_step_clip_mean) if np.isfinite(z4_gap_step_clip_mean) else 0.0,
            "z4_gap_step_ema_mean": float(z4_gap_step_ema_mean) if np.isfinite(z4_gap_step_ema_mean) else 0.0,
            "z4_gap_step_mean": float(z4_gap_step_mean) if np.isfinite(z4_gap_step_mean) else 0.0,
            "z4_gap_reward_step_mean": float(z4_gap_reward_step_mean) if np.isfinite(z4_gap_reward_step_mean) else 0.0,
            "z4_gap_lambda_input_mean": float(z4_gap_lambda_input_mean) if np.isfinite(z4_gap_lambda_input_mean) else 0.0,
            "z4_gap_lambda_sched_mean": float(z4_gap_lambda_sched_mean) if np.isfinite(z4_gap_lambda_sched_mean) else 0.0,
            "z4_gap_lambda_cap_mean": float(z4_gap_lambda_cap_mean) if np.isfinite(z4_gap_lambda_cap_mean) else 0.0,
            "z4_gap_lambda_eff_mean": float(z4_gap_lambda_eff_mean) if np.isfinite(z4_gap_lambda_eff_mean) else 0.0,
            "z4_gap_grad_ratio_eff_mean": float(z4_gap_grad_ratio_eff_mean) if np.isfinite(z4_gap_grad_ratio_eff_mean) else 0.0,
            "z4_gap_shape_guard_trigger_mean": float(z4_gap_shape_guard_trigger_mean)
            if np.isfinite(z4_gap_shape_guard_trigger_mean)
            else 0.0,
            "t6_main_weight_mean": float(t6_main_weight_mean) if np.isfinite(t6_main_weight_mean) else 0.0,
            "t6_potential_scale_mean": float(t6_potential_scale_mean) if np.isfinite(t6_potential_scale_mean) else 0.0,
            "t6_traj_scale_mean": float(t6_traj_scale_mean) if np.isfinite(t6_traj_scale_mean) else 0.0,
            "t6_anneal_ratio_mean": float(t6_anneal_ratio_mean) if np.isfinite(t6_anneal_ratio_mean) else 0.0,
            "t6_violation_ema_mean": float(t6_violation_ema_mean) if np.isfinite(t6_violation_ema_mean) else 0.0,
            "t6_pid_frozen_mean": float(t6_pid_frozen_mean) if np.isfinite(t6_pid_frozen_mean) else 0.0,
            "comp_bonus_raw_mean": float(comp_bonus_raw_mean) if np.isfinite(comp_bonus_raw_mean) else 0.0,
            "ris_bonus_raw_mean": float(ris_bonus_raw_mean) if np.isfinite(ris_bonus_raw_mean) else 0.0,
            
            "env_shaping_mean": float(env_shaping_mean) if np.isfinite(env_shaping_mean) else 0.0,
            "total_shaping_mean": float(total_shaping_mean) if np.isfinite(total_shaping_mean) else 0.0,
            "total_shaping_all_mean": float(total_shaping_all_mean) if np.isfinite(total_shaping_all_mean) else 0.0,
            "comp_bonus_mean": float(comp_bonus_mean) if np.isfinite(comp_bonus_mean) else 0.0,
            "ris_bonus_mean": float(ris_bonus_mean) if np.isfinite(ris_bonus_mean) else 0.0,
            "comp_gain_ratio_mean": float(comp_gain_ratio_mean) if np.isfinite(comp_gain_ratio_mean) else 0.0,
            "ris_gain_ratio_mean": float(ris_gain_ratio_mean) if np.isfinite(ris_gain_ratio_mean) else 0.0,
            "comp_enable_rate_mean": float(comp_enable_rate_mean) if np.isfinite(comp_enable_rate_mean) else float("nan"),
            "service_switch_count_mean": float(service_switch_count_mean) if np.isfinite(service_switch_count_mean) else float("nan"),
            "theta_entropy_mean": float(theta_entropy_mean) if np.isfinite(theta_entropy_mean) else float("nan"),
            "comp_score_temp_runtime_mean": float(comp_score_temp_runtime_mean) if np.isfinite(comp_score_temp_runtime_mean) else float("nan"),
            "theta_mode_effective": str(theta_mode_effective),
            "theta_mode_effective_agent_frac": float(theta_mode_effective_agent_frac)
            if np.isfinite(theta_mode_effective_agent_frac)
            else float("nan"),
            "reward_total_step_mean": float(reward_total_step_mean) if np.isfinite(reward_total_step_mean) else 0.0,
            "paper_cost_step_mean": float(paper_cost_step_mean) if np.isfinite(paper_cost_step_mean) else 0.0,
            "reward_total_ep_sum_mean": float(reward_total_ep_sum_mean) if np.isfinite(reward_total_ep_sum_mean) else float("nan"),
            "paper_cost_ep_sum_mean": float(paper_cost_ep_sum_mean) if np.isfinite(paper_cost_ep_sum_mean) else float("nan"),
            "reward_paper_ep_sum_mean": float(reward_paper_ep_sum_mean) if np.isfinite(reward_paper_ep_sum_mean) else float("nan"),
            "reward_shaping_ep_sum_mean": float(reward_shaping_ep_sum_mean) if np.isfinite(reward_shaping_ep_sum_mean) else float("nan"),
            "reward_constraint_ep_sum_mean": float(reward_constraint_ep_sum_mean) if np.isfinite(reward_constraint_ep_sum_mean) else float("nan"),
            "reward_misc_ep_sum_mean": float(reward_misc_ep_sum_mean) if np.isfinite(reward_misc_ep_sum_mean) else float("nan"),
            "reward_proximity_ep_sum_mean": float(reward_proximity_ep_sum_mean) if np.isfinite(reward_proximity_ep_sum_mean) else float("nan"),
            "ep_sum_n": float(ep_sum_n),
            "reward_contract_abs_err_mean": float(contract_abs_err_mean) if np.isfinite(contract_abs_err_mean) else float("nan"),
            "reward_contract_abs_err_max": float(contract_abs_err_max) if np.isfinite(contract_abs_err_max) else float("nan"),
            "vio_any_frac": float(vio_any_frac),
            "reward_raw_mean": float(reward_raw_mean),
            "reward_raw_std": float(reward_raw_std),
            "reward_raw_p05": float(reward_raw_p05),
            "reward_raw_p50": float(reward_raw_p50),
            "reward_raw_p95": float(reward_raw_p95),
            "clip_hit_frac": float(clip_hit_frac),
        }

    def compute_paper_reward(
        self,
        *,
        paper_cost: np.ndarray,
        baseline_balanced: np.ndarray,
        improve_env: np.ndarray,
        vio_metric: np.ndarray,
        update_improve_z: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
         paper-rewardPhase C C0.2/C0.2b

        
        - reward_out/rewardLagrange
        - reward_rawrewardtelemetry
        - paper_improvecost_b - paper_cost
        - improve_metricpaper_improve / rel_improve / improve_z
        """
        paper_cost = np.asarray(paper_cost, dtype=np.float32).reshape(-1)
        baseline_balanced = np.asarray(baseline_balanced, dtype=np.float32).reshape(-1)
        improve_env = np.asarray(improve_env, dtype=np.float32).reshape(-1)
        vio_metric = np.asarray(vio_metric, dtype=np.float32).reshape(-1)

        
        base_ok = np.isfinite(baseline_balanced)
        paper_improve = np.where(base_ok, (baseline_balanced - paper_cost), improve_env).astype(np.float32)

        
        improve_metric = paper_improve.copy()
        if str(self.improve_mode) == "neg_paper_cost":
            
            improve_metric = (-paper_cost).astype(np.float32)
        elif str(self.improve_mode) == "rel_improve":
            eps = float(self.rel_improve_eps)
            denom = np.abs(np.asarray(baseline_balanced, dtype=np.float32))
            denom = np.where(np.isfinite(denom), denom, 0.0)
            improve_metric = (paper_improve / (denom + eps)).astype(np.float32)
        elif str(self.improve_mode) == "improve_z":
            if bool(update_improve_z):
                try:
                    vals = np.asarray(paper_improve, dtype=np.float64).reshape(-1)
                    for v in vals[np.isfinite(vals)]:
                        self._impz_n += 1
                        delta = float(v) - float(self._impz_mean)
                        self._impz_mean += delta / float(self._impz_n)
                        delta2 = float(v) - float(self._impz_mean)
                        self._impz_m2 += delta * delta2
                except Exception:
                    pass
            mu = float(self._impz_mean)
            var = float(self._impz_m2) / float(max(self._impz_n - 1, 1))
            sigma = float(math.sqrt(max(var, 0.0)))
            eps = float(self.improve_z_eps)
            improve_metric = ((paper_improve - mu) / (sigma + eps)).astype(np.float32)

        
        if str(self.paper_reward_map) == "clip":
            c = float(self.paper_reward_clip_c)
            improve_term = np.clip(float(self.paper_reward_scale) * improve_metric, -c, c)
        else:
            s = float(self.paper_reward_tanh_s)
            improve_term = float(self.paper_reward_scale) * np.tanh(improve_metric / max(s, 1e-6))

        
        gate = np.clip(1.0 - np.asarray(vio_metric, dtype=np.float32), 0.0, 1.0)
        p = float(self.paper_vio_gating_power)
        if p != 1.0:
            gate = np.power(gate, p)
        penalty = np.zeros_like(gate, dtype=np.float32)
        if float(self.paper_vio_penalty) > 0.0:
            penalty = float(self.paper_vio_penalty) * (1.0 - gate)

        reward_raw = (float(self.paper_reward_bias) + improve_term - penalty).astype(np.float32)

        
        if self.lagrange_enabled and float(self.lambda_v) > 0.0:
            reward_raw = (reward_raw - float(self.lambda_v) * float(self.vio_scale) * vio_metric).astype(np.float32)

        reward_out = reward_raw.copy()
        
        if float(self.reward_clip_range) > 0.0:
            cr = float(self.reward_clip_range)
            reward_out = (cr * np.tanh(reward_out / cr)).astype(np.float32)

        return reward_out.astype(np.float32), reward_raw.astype(np.float32), paper_improve.astype(np.float32), improve_metric.astype(np.float32)

    @staticmethod
    def _as_info_list(info: Any) -> List[Dict[str, Any]]:
        if info is None:
            return []
        if isinstance(info, dict):
            return [info]
        try:
            return list(info)
        except Exception:
            return []

    @staticmethod
    def _attach_obs_raw_to_info(info: Any, obs_raw: Any) -> Any:
        """
        Phase C obs_raw info 

        scores_fixed obs_flat obs_normalizer 
        CollectorPreprocess  obs_raw  info["obs_raw"] Actor  forward 
        """
        try:
            obs_arr = np.asarray(obs_raw, dtype=np.float32)
        except Exception:
            return info

        
        if isinstance(info, dict):
            out = dict(info)
            out["obs_raw"] = obs_arr.copy()
            return out

        
        try:
            info_list = list(info) if info is not None else []
        except Exception:
            return info

        if not info_list:
            
            return [{"obs_raw": obs_arr.copy()}] if obs_arr.ndim == 1 else [{"obs_raw": obs_arr[i].copy()} for i in range(obs_arr.shape[0])]

        if obs_arr.ndim == 1:
            obs_arr = np.repeat(obs_arr.reshape(1, -1), repeats=len(info_list), axis=0)
        elif obs_arr.ndim == 2 and obs_arr.shape[0] != len(info_list):
            
            if obs_arr.shape[0] == 1:
                obs_arr = np.repeat(obs_arr, repeats=len(info_list), axis=0)
            else:
                obs_arr = obs_arr[: len(info_list)]

        out_list: List[Dict[str, Any]] = []
        for i, ii in enumerate(info_list):
            if isinstance(ii, dict):
                d = dict(ii)
            else:
                try:
                    d = dict(ii)
                except Exception:
                    d = {}
            try:
                d["obs_raw"] = obs_arr[i].copy()
            except Exception:
                d["obs_raw"] = obs_arr.copy()
            out_list.append(d)
        return out_list

    def __call__(self, **kwargs) -> Dict[str, Any]:
        # resetCollector.reset_env/_reset_env_with_ids => obs/info/env_id
        if "rew" not in kwargs:
            obs = kwargs.get("obs", None)
            info = kwargs.get("info", None)
            if self.obs_normalizer is not None and obs is not None:
                obs_n = self.obs_normalizer.normalize(obs, update=self.update_obs)
                return {"obs": obs_n, "info": self._attach_obs_raw_to_info(info, obs)}
            if obs is not None:
                return {"info": self._attach_obs_raw_to_info(info, obs)}
            return {}

        # stepCollector.collect => obs_next/rew/terminated/truncated/info/env_id/act
        obs_next = kwargs.get("obs_next", None)
        rew = kwargs.get("rew", None)
        info = kwargs.get("info", None)
        if obs_next is None or rew is None:
            return {}

        if self.obs_normalizer is not None:
            obs_next_n = self.obs_normalizer.normalize(obs_next, update=self.update_obs)
        else:
            obs_next_n = obs_next

        infos = self._as_info_list(info)
        n = int(np.asarray(rew).shape[0]) if np.asarray(rew).ndim > 0 else 1
        if len(infos) != n and len(infos) > 0 and n > 1:
            
            infos = (infos * (n // len(infos) + 1))[:n]

        
        rd_set = set()
        try:
            for ii in infos:
                if isinstance(ii, dict):
                    rd_set.add(str(ii.get("reward_design", "")).strip().lower())
        except Exception:
            rd_set = set()
        is_phaseg_reward_total = bool(("reward_total_v1" in rd_set) or ("reward_total" in rd_set) or ("total_v1" in rd_set))

        def _get(ii: Dict[str, Any], k1: str, k2: str, default: float = 0.0) -> float:
            if k1 in ii:
                return float(ii.get(k1, default))
            return float(ii.get(k2, default))

        paper_cost_step = (
            np.asarray(
                [float(ii.get("paper_cost_step", _get(ii, "paper_cost", "cost", 0.0))) for ii in infos],
                dtype=np.float32,
            )
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        
        paper_cost = np.asarray(paper_cost_step, dtype=np.float32).reshape(-1)
        vio_cnt = (
            
            np.asarray(
                [
                    float(
                        ii.get(
                            "penalty_total",
                            ii.get(
                                "cost_constraint",
                                ii.get("violation_count", ii.get("vio_total_penalty", 0.0)),
                            ),
                        )
                    )
                    for ii in infos
                ],
                dtype=np.float32,
            )
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        improve = (
            np.asarray([_get(ii, "imp_paper_cost", "imp_paper_cost", 0.0) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )

        
        comp_gain_ratio = (
            np.asarray([float(ii.get("comp_gain_ratio", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        ris_gain_ratio = (
            np.asarray([float(ii.get("ris_gain_ratio", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        comp_users_fraction = (
            np.asarray([float(ii.get("comp_users_fraction", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        ris_users_fraction = (
            np.asarray([float(ii.get("ris_users_fraction", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        
        comp_enable_rate_step = (
            np.asarray([float(ii.get("comp_enable_rate", float("nan"))) for ii in infos], dtype=np.float32)
            if infos
            else np.full((n,), np.nan, dtype=np.float32)
        )
        service_switch_count_step = (
            np.asarray([float(ii.get("service_switch_count", float("nan"))) for ii in infos], dtype=np.float32)
            if infos
            else np.full((n,), np.nan, dtype=np.float32)
        )
        theta_entropy_step = (
            np.asarray([float(ii.get("theta_entropy", float("nan"))) for ii in infos], dtype=np.float32)
            if infos
            else np.full((n,), np.nan, dtype=np.float32)
        )
        comp_score_temp_runtime_step = (
            np.asarray([float(ii.get("comp_score_temp_runtime", float("nan"))) for ii in infos], dtype=np.float32)
            if infos
            else np.full((n,), np.nan, dtype=np.float32)
        )
        theta_mode_effective_step = (
            [str(ii.get("theta_mode_effective", "")).strip().lower() for ii in infos]
            if infos
            else []
        )
        env_shaping = (
            np.asarray(
                [float(ii.get("reward_shaping_total", ii.get("reward_potential", 0.0))) for ii in infos],
                dtype=np.float32,
            )
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        
        env_comp_step = (
            np.asarray([float(ii.get("reward_comp_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        env_ris_step = (
            np.asarray([float(ii.get("reward_ris_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        reward_paper_step = (
            np.asarray([float(ii.get("reward_paper_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        reward_paper_abs_step = (
            np.asarray([float(ii.get("reward_paper_abs_step", ii.get("reward_paper_step", 0.0))) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        reward_paper_delta_step = (
            np.asarray([float(ii.get("reward_paper_delta_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        reward_paper_adv_step = (
            np.asarray([float(ii.get("reward_paper_adv_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        paper_delta_norm_step = (
            np.asarray([float(ii.get("paper_delta_norm_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        paper_adv_norm_step = (
            np.asarray([float(ii.get("paper_adv_norm_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        constraint_signal_step = (
            np.asarray([float(ii.get("constraint_signal_step", ii.get("vio_any_step", 0.0))) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        lambda_v_effective_step = (
            np.asarray([float(ii.get("lambda_v_effective", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        reward_constraint_step = (
            np.asarray([float(ii.get("reward_constraint_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        reward_misc_step = (
            np.asarray([float(ii.get("reward_misc_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        
        reward_proximity_step = (
            np.asarray([float(ii.get("reward_proximity_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        shaping_gate = (
            np.asarray([float(ii.get("shaping_gate", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        beta_comp_t = (
            np.asarray([float(ii.get("beta_comp_t", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        beta_ris_t = (
            np.asarray([float(ii.get("beta_ris_t", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        beta_train_frac = (
            np.asarray([float(ii.get("beta_train_frac", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_raw_step = (
            np.asarray([float(ii.get("z4_gap_raw_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_step_clip = (
            np.asarray([float(ii.get("z4_gap_step_clip", ii.get("z4_gap_step", 0.0))) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_step_ema = (
            np.asarray([float(ii.get("z4_gap_step_ema", ii.get("z4_gap_step", 0.0))) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_step = (
            np.asarray([float(ii.get("z4_gap_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_reward_step = (
            np.asarray([float(ii.get("z4_gap_reward_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_lambda_input = (
            np.asarray([float(ii.get("z4_gap_lambda_input", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_lambda_sched = (
            np.asarray([float(ii.get("z4_gap_lambda_sched", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_lambda_cap = (
            np.asarray([float(ii.get("z4_gap_lambda_cap", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_lambda_eff = (
            np.asarray([float(ii.get("z4_gap_lambda_eff", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_grad_ratio_eff = (
            np.asarray([float(ii.get("z4_gap_grad_ratio_eff", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        z4_gap_shape_guard_trigger = (
            np.asarray([float(ii.get("z4_gap_shape_guard_trigger", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        t6_main_weight_step = (
            np.asarray([float(ii.get("t6_main_weight", 1.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.ones((n,), dtype=np.float32)
        )
        t6_potential_scale_step = (
            np.asarray([float(ii.get("t6_potential_scale", 1.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.ones((n,), dtype=np.float32)
        )
        t6_traj_scale_step = (
            np.asarray([float(ii.get("t6_traj_scale", 1.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.ones((n,), dtype=np.float32)
        )
        t6_anneal_ratio_step = (
            np.asarray([float(ii.get("t6_anneal_ratio", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        t6_violation_ema_step = (
            np.asarray([float(ii.get("t6_violation_ema", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        t6_pid_frozen_step = (
            np.asarray([float(ii.get("t6_pid_frozen", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        comp_bonus_raw_step = (
            np.asarray([float(ii.get("comp_bonus_raw_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        ris_bonus_raw_step = (
            np.asarray([float(ii.get("ris_bonus_raw_step", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        reward_shaping_step_env = (
            np.asarray(
                [float(ii.get("reward_shaping_step", ii.get("shaping_term", 0.0))) for ii in infos],
                dtype=np.float32,
            )
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        reward_total_step_env = (
            np.asarray(
                [float(ii.get("reward_total_step", np.asarray(rew, dtype=np.float32).reshape(-1)[i])) for i, ii in enumerate(infos)],
                dtype=np.float32,
            )
            if infos
            else np.asarray(rew, dtype=np.float32).reshape(-1)
        )
        paper_term_env = (
            np.asarray(
                [float(ii.get("paper_term", ii.get("reward_paper_step", 0.0))) for ii in infos],
                dtype=np.float32,
            )
            if infos
            else np.asarray(reward_paper_step, dtype=np.float32).reshape(-1)
        )
        misc_term_env = (
            np.asarray(
                [float(ii.get("misc_term", ii.get("reward_misc_step", 0.0))) for ii in infos],
                dtype=np.float32,
            )
            if infos
            else np.asarray(reward_misc_step, dtype=np.float32).reshape(-1)
        )
        penalty_term_env = (
            np.asarray(
                [
                    float(
                        ii.get(
                            "penalty_term",
                            max(-float(ii.get("reward_constraint_step", 0.0)), 0.0),
                        )
                    )
                    for ii in infos
                ],
                dtype=np.float32,
            )
            if infos
            else np.maximum(-np.asarray(reward_constraint_step, dtype=np.float32).reshape(-1), 0.0)
        )

        
        
        coverage_fraction = (
            np.asarray([float(ii.get("coverage_fraction", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        coverage_fraction = np.clip(coverage_fraction, 0.0, 1.0)
        coverage_margin = (
            np.asarray([float(ii.get("coverage_margin", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        collision_margin = (
            np.asarray([float(ii.get("collision_margin", 0.0)) for ii in infos], dtype=np.float32)
            if infos
            else np.zeros((n,), dtype=np.float32)
        )
        vio_metric = np.clip(
            (1.0 - coverage_fraction) + 0.5 * coverage_margin + 0.5 * collision_margin,
            0.0,
            1.0,
        ).astype(np.float32)

        env_reward = np.asarray(rew, dtype=np.float32).reshape(-1)
        reward_out = env_reward.copy()
        reward_raw_before_tanh = env_reward.copy()
        
        paper_improve_np = np.asarray(improve, dtype=np.float32).reshape(-1)
        improve_metric_np = paper_improve_np.copy()

        force_env_reward = bool(is_phaseg_reward_total)
        if self.reward_override == "paper" and (not force_env_reward):
            baseline_balanced = (
                np.asarray([float(ii.get("baseline_balanced_paper_cost", float("nan"))) for ii in infos], dtype=np.float32)
                if infos
                else np.full((n,), float("nan"), dtype=np.float32)
            )

            
            reward_out, reward_raw_before_tanh, paper_improve_np, improve_metric_np = self.compute_paper_reward(
                paper_cost=paper_cost,
                baseline_balanced=baseline_balanced,
                improve_env=improve,
                vio_metric=vio_metric,
                update_improve_z=bool(self.update_improve_z),
            )
        else:
            if bool(is_phaseg_reward_total):
                
                reward_out = env_reward.copy()
                
                if infos:
                    try:
                        rr = [float(ii.get("reward_raw", env_reward[i])) for i, ii in enumerate(infos)]
                        reward_raw_before_tanh = np.asarray(rr, dtype=np.float32).reshape(-1)
                    except Exception:
                        reward_raw_before_tanh = env_reward.copy()
                else:
                    reward_raw_before_tanh = env_reward.copy()
            else:
                
                if self.lagrange_enabled and float(self.lambda_v) > 0.0:
                    reward_out = (reward_out - float(self.lambda_v) * float(self.vio_scale) * vio_metric).astype(np.float32)
                reward_raw_before_tanh = reward_out.copy()
                if float(self.reward_clip_range) > 0.0:
                    cr = float(self.reward_clip_range)
                    reward_out = (cr * np.tanh(reward_out / cr)).astype(np.float32)

        
        
        
        
        
        
        comp_bonus = np.asarray(env_comp_step, dtype=np.float32).reshape(-1)
        ris_bonus = np.asarray(env_ris_step, dtype=np.float32).reshape(-1)
        total_shaping = (comp_bonus + ris_bonus).astype(np.float32)
        
        total_shaping_add = np.zeros_like(total_shaping, dtype=np.float32)
        try:
            
            if (not bool(is_phaseg_reward_total)) and bool(self.enable_comp_ris_shaping) and float(self.shaping_ratio) > 0.0:
                
                comp_sig = np.maximum(comp_gain_ratio - 1.0, 0.0) * np.clip(comp_users_fraction, 0.0, 1.0)
                ris_sig = np.maximum(ris_gain_ratio, 0.0) * np.clip(ris_users_fraction, 0.0, 1.0)

                
                wc = float(max(getattr(self, "comp_weight", 0.0), 0.0))
                wr = float(max(getattr(self, "ris_weight", 0.0), 0.0))
                comp_part = comp_sig * wc
                ris_part = ris_sig * wr
                den = comp_part + ris_part
                has_sig = den > 1e-12

                
                vio_any = np.asarray(vio_cnt, dtype=np.float32).reshape(-1) > 0.0
                gate = np.where(vio_any, 0.0, np.clip(1.0 - np.asarray(vio_metric, dtype=np.float32).reshape(-1), 0.0, 1.0))

                r_main = np.asarray(reward_out, dtype=np.float32).reshape(-1)
                total_target = float(self.shaping_ratio) * np.abs(r_main) * gate
                total_target = np.where(has_sig, total_target, 0.0).astype(np.float32)

                comp_bonus_add = np.where(has_sig, total_target * (comp_part / np.maximum(den, 1e-12)), 0.0).astype(np.float32)
                ris_bonus_add = np.where(has_sig, total_target * (ris_part / np.maximum(den, 1e-12)), 0.0).astype(np.float32)

                
                comp_bonus = (np.asarray(comp_bonus, dtype=np.float32).reshape(-1) + np.asarray(comp_bonus_add, dtype=np.float32).reshape(-1)).astype(np.float32)
                ris_bonus = (np.asarray(ris_bonus, dtype=np.float32).reshape(-1) + np.asarray(ris_bonus_add, dtype=np.float32).reshape(-1)).astype(np.float32)
                total_shaping = (comp_bonus + ris_bonus).astype(np.float32)

                
                total_shaping_add = (np.asarray(comp_bonus_add, dtype=np.float32).reshape(-1) + np.asarray(ris_bonus_add, dtype=np.float32).reshape(-1)).astype(np.float32)
                reward_raw_before_tanh = (np.asarray(reward_raw_before_tanh, dtype=np.float32).reshape(-1) + total_shaping_add).astype(np.float32)
                reward_out = reward_raw_before_tanh.copy()
                if float(self.reward_clip_range) > 0.0:
                    cr = float(self.reward_clip_range)
                    reward_out = (cr * np.tanh(reward_out / cr)).astype(np.float32)
        except Exception:
            
            comp_bonus = np.zeros_like(reward_out, dtype=np.float32)
            ris_bonus = np.zeros_like(reward_out, dtype=np.float32)
            total_shaping = np.zeros_like(reward_out, dtype=np.float32)
            total_shaping_add = np.zeros_like(reward_out, dtype=np.float32)
        # ===== [END Phase C C3] =====

        
        
        
        def _to_float(v: Any, default: float = 0.0) -> float:
            try:
                vv = float(v)
            except Exception:
                return float(default)
            return float(vv) if np.isfinite(vv) else float(default)

        def _arr64(x: Any, *, fill: Optional[float] = 0.0) -> np.ndarray:
            try:
                arr = np.asarray(x, dtype=np.float64).reshape(-1)
            except Exception:
                return np.asarray([], dtype=np.float64)
            if fill is None:
                return arr
            return np.where(np.isfinite(arr), arr, float(fill)).astype(np.float64, copy=False)

        def _resize64(x: Any, size: int, *, fill: float = 0.0) -> np.ndarray:
            n0 = int(max(size, 0))
            if n0 <= 0:
                return np.asarray([], dtype=np.float64)
            arr = _arr64(x, fill=fill)
            if arr.size == n0:
                return arr
            if arr.size <= 0:
                return np.full((n0,), float(fill), dtype=np.float64)
            arr = np.resize(arr, (n0,))
            return np.where(np.isfinite(arr), arr, float(fill)).astype(np.float64, copy=False)

        def _extend(dst: List[float], values: Any, *, keep_nan: bool = True) -> None:
            arr = _arr64(values, fill=None)
            if not keep_nan:
                arr = arr[np.isfinite(arr)]
            dst.extend([float(v) for v in arr.tolist()])

        n_i = int(max(n, 0))
        reward_step_arr = _resize64(reward_out, n_i, fill=0.0)
        paper_step_arr = _resize64(paper_cost_step, n_i, fill=0.0)
        vio_step_arr = _resize64(vio_cnt, n_i, fill=0.0)
        improve_step_arr = _resize64(improve_metric_np, n_i, fill=0.0)
        shaping_all_step_arr = _resize64(env_shaping, n_i, fill=0.0) + _resize64(total_shaping_add, n_i, fill=0.0)
        reward_shaping_step_env_arr = _resize64(reward_shaping_step_env, n_i, fill=0.0)
        reward_paper_step_arr = _resize64(reward_paper_step, n_i, fill=0.0)
        reward_constraint_step_arr = _resize64(reward_constraint_step, n_i, fill=0.0)
        reward_misc_step_arr = _resize64(reward_misc_step, n_i, fill=0.0)
        reward_proximity_step_arr = _resize64(reward_proximity_step, n_i, fill=0.0)
        comp_step_arr = _resize64(comp_bonus, n_i, fill=0.0)
        ris_step_arr = _resize64(ris_bonus, n_i, fill=0.0)
        reward_total_arr = _resize64(reward_total_step_env, n_i, fill=0.0)
        paper_term_arr = _resize64(paper_term_env, n_i, fill=0.0)
        shaping_term_arr = _resize64(reward_shaping_step_env, n_i, fill=0.0)
        penalty_term_arr = _resize64(penalty_term_env, n_i, fill=0.0)
        misc_term_arr = _resize64(misc_term_env, n_i, fill=0.0)

        
        reward_paper_contract_arr = _resize64(paper_term_arr, n_i, fill=0.0)
        reward_shaping_contract_arr = _resize64(shaping_term_arr, n_i, fill=0.0)
        reward_constraint_contract_arr = -_resize64(penalty_term_arr, n_i, fill=0.0)
        reward_misc_contract_arr = _resize64(misc_term_arr, n_i, fill=0.0)

        
        reward_log_step_arr = reward_total_arr if bool(is_phaseg_reward_total) else reward_step_arr
        _extend(self._roll_reward_total_step, reward_log_step_arr, keep_nan=False)
        _extend(self._roll_paper_cost_step, paper_step_arr, keep_nan=False)

        
        contract_rhs = paper_term_arr + shaping_term_arr - penalty_term_arr + misc_term_arr
        contract_abs = np.abs(reward_total_arr - contract_rhs)
        _extend(self._roll_contract_residual, contract_abs, keep_nan=False)

        
        done_raw = kwargs.get("done", None)
        if done_raw is None:
            terminated_raw = kwargs.get("terminated", None)
            truncated_raw = kwargs.get("truncated", None)
            if terminated_raw is not None or truncated_raw is not None:
                term = _resize64(
                    terminated_raw if terminated_raw is not None else np.zeros((n_i,), dtype=bool),
                    n_i,
                    fill=0.0,
                )
                trunc = _resize64(
                    truncated_raw if truncated_raw is not None else np.zeros((n_i,), dtype=bool),
                    n_i,
                    fill=0.0,
                )
                done_flags = (term > 0.5) | (trunc > 0.5)
            else:
                done_flags = np.zeros((n_i,), dtype=bool)
        else:
            done_flags = _resize64(done_raw, n_i, fill=0.0) > 0.5

        env_id_raw = kwargs.get("env_id", None)
        if env_id_raw is None:
            env_ids = np.arange(n_i, dtype=np.int64)
        else:
            env_ids = _resize64(env_id_raw, n_i, fill=0.0).astype(np.int64)

        for i in range(int(n_i)):
            eid = int(env_ids[i])
            self._ep_running_reward_sum[eid] = float(self._ep_running_reward_sum.get(eid, 0.0) + float(reward_log_step_arr[i]))
            self._ep_running_paper_cost_sum[eid] = float(self._ep_running_paper_cost_sum.get(eid, 0.0) + float(paper_step_arr[i]))
            self._ep_running_reward_paper_sum[eid] = float(
                self._ep_running_reward_paper_sum.get(eid, 0.0) + float(reward_paper_contract_arr[i])
            )
            self._ep_running_reward_shaping_sum[eid] = float(
                self._ep_running_reward_shaping_sum.get(eid, 0.0) + float(reward_shaping_contract_arr[i])
            )
            self._ep_running_reward_constraint_sum[eid] = float(
                self._ep_running_reward_constraint_sum.get(eid, 0.0) + float(reward_constraint_contract_arr[i])
            )
            self._ep_running_reward_misc_sum[eid] = float(
                self._ep_running_reward_misc_sum.get(eid, 0.0) + float(reward_misc_contract_arr[i])
            )
            self._ep_running_reward_proximity_sum[eid] = float(
                self._ep_running_reward_proximity_sum.get(eid, 0.0) + float(reward_proximity_step_arr[i])
            )
            self._ep_running_cost_sum[eid] = float(self._ep_running_cost_sum.get(eid, 0.0) + float(paper_step_arr[i]))
            self._ep_running_vio_sum[eid] = float(self._ep_running_vio_sum.get(eid, 0.0) + float(vio_step_arr[i]))
            self._ep_running_improve_sum[eid] = float(self._ep_running_improve_sum.get(eid, 0.0) + float(improve_step_arr[i]))
            self._ep_running_shaping_sum[eid] = float(self._ep_running_shaping_sum.get(eid, 0.0) + float(shaping_all_step_arr[i]))
            self._ep_running_comp_sum[eid] = float(self._ep_running_comp_sum.get(eid, 0.0) + float(comp_step_arr[i]))
            self._ep_running_ris_sum[eid] = float(self._ep_running_ris_sum.get(eid, 0.0) + float(ris_step_arr[i]))
            self._ep_running_step_cnt[eid] = int(self._ep_running_step_cnt.get(eid, 0) + 1)
            if bool(done_flags[i]):
                ep_cnt = int(self._ep_running_step_cnt.get(eid, 0))
                self._roll_reward_total_ep_sum.append(float(self._ep_running_reward_sum.get(eid, 0.0)))
                self._roll_paper_cost_ep_sum.append(float(self._ep_running_paper_cost_sum.get(eid, 0.0)))
                self._roll_reward_paper_ep_sum.append(float(self._ep_running_reward_paper_sum.get(eid, 0.0)))
                self._roll_reward_shaping_ep_sum.append(float(self._ep_running_reward_shaping_sum.get(eid, 0.0)))
                self._roll_reward_constraint_ep_sum.append(float(self._ep_running_reward_constraint_sum.get(eid, 0.0)))
                self._roll_reward_misc_ep_sum.append(float(self._ep_running_reward_misc_sum.get(eid, 0.0)))
                self._roll_reward_proximity_ep_sum.append(float(self._ep_running_reward_proximity_sum.get(eid, 0.0)))
                if ep_cnt > 0:
                    denom = float(ep_cnt)
                    self._roll_paper_cost_ep_mean.append(float(self._ep_running_cost_sum.get(eid, 0.0)) / denom)
                    self._roll_vio_cnt_ep_mean.append(float(self._ep_running_vio_sum.get(eid, 0.0)) / denom)
                    self._roll_improve_ep_mean.append(float(self._ep_running_improve_sum.get(eid, 0.0)) / denom)
                    self._roll_total_shaping_all_ep_mean.append(float(self._ep_running_shaping_sum.get(eid, 0.0)) / denom)
                    self._roll_comp_bonus_ep_mean.append(float(self._ep_running_comp_sum.get(eid, 0.0)) / denom)
                    self._roll_ris_bonus_ep_mean.append(float(self._ep_running_ris_sum.get(eid, 0.0)) / denom)
                self._ep_running_reward_sum[eid] = 0.0
                self._ep_running_paper_cost_sum[eid] = 0.0
                self._ep_running_reward_paper_sum[eid] = 0.0
                self._ep_running_reward_shaping_sum[eid] = 0.0
                self._ep_running_reward_constraint_sum[eid] = 0.0
                self._ep_running_reward_misc_sum[eid] = 0.0
                self._ep_running_reward_proximity_sum[eid] = 0.0
                self._ep_running_cost_sum[eid] = 0.0
                self._ep_running_vio_sum[eid] = 0.0
                self._ep_running_improve_sum[eid] = 0.0
                self._ep_running_shaping_sum[eid] = 0.0
                self._ep_running_comp_sum[eid] = 0.0
                self._ep_running_ris_sum[eid] = 0.0
                self._ep_running_step_cnt[eid] = 0

        _extend(self._roll_paper_cost, _resize64(paper_cost, n_i, fill=0.0), keep_nan=False)
        
        if infos:
            bb = [_to_float(ii.get("baseline_balanced_paper_cost", float("nan")), default=float("nan")) for ii in infos]
            _extend(self._roll_baseline_balanced, np.asarray(bb, dtype=np.float64).reshape(-1), keep_nan=True)
            bd = [_to_float(ii.get("baseline_greedy_delay_paper_cost", float("nan")), default=float("nan")) for ii in infos]
            _extend(self._roll_baseline_greedy_delay, np.asarray(bd, dtype=np.float64).reshape(-1), keep_nan=True)
            be = [_to_float(ii.get("baseline_greedy_energy_paper_cost", float("nan")), default=float("nan")) for ii in infos]
            _extend(self._roll_baseline_greedy_energy, np.asarray(be, dtype=np.float64).reshape(-1), keep_nan=True)
        else:
            self._roll_baseline_balanced.extend([float("nan")] * int(n_i))
            self._roll_baseline_greedy_delay.extend([float("nan")] * int(n_i))
            self._roll_baseline_greedy_energy.extend([float("nan")] * int(n_i))

        
        paper_improve_arr = _arr64(paper_improve_np, fill=None)
        if paper_improve_arr.size <= 0:
            paper_improve_arr = _arr64(improve, fill=None)
        _extend(self._roll_paper_improve, paper_improve_arr, keep_nan=True)

        improve_metric_arr = _arr64(improve_metric_np, fill=None)
        if improve_metric_arr.size <= 0:
            improve_metric_arr = _arr64(paper_improve_np, fill=None)
        _extend(self._roll_improve, improve_metric_arr, keep_nan=True)

        _extend(self._roll_vio_cnt, _resize64(vio_cnt, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_vio_metric, _resize64(vio_metric, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_coverage_fraction, _resize64(coverage_fraction, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_coverage_margin, _resize64(coverage_margin, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_collision_margin, _resize64(collision_margin, n_i, fill=0.0), keep_nan=True)
        
        _extend(self._roll_reward_paper_step, _resize64(reward_paper_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_reward_paper_abs_step, _resize64(reward_paper_abs_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_reward_paper_delta_step, _resize64(reward_paper_delta_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_reward_paper_adv_step, _resize64(reward_paper_adv_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_paper_delta_norm_step, _resize64(paper_delta_norm_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_paper_adv_norm_step, _resize64(paper_adv_norm_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_constraint_signal_step, _resize64(constraint_signal_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_lambda_v_effective, _resize64(lambda_v_effective_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_reward_constraint_step, _resize64(reward_constraint_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_reward_misc_step, _resize64(reward_misc_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_reward_proximity_step, _resize64(reward_proximity_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_shaping_gate, _resize64(shaping_gate, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_beta_comp_t, _resize64(beta_comp_t, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_beta_ris_t, _resize64(beta_ris_t, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_beta_train_frac, _resize64(beta_train_frac, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_raw_step, _resize64(z4_gap_raw_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_step_clip, _resize64(z4_gap_step_clip, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_step_ema, _resize64(z4_gap_step_ema, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_step, _resize64(z4_gap_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_reward_step, _resize64(z4_gap_reward_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_lambda_input, _resize64(z4_gap_lambda_input, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_lambda_sched, _resize64(z4_gap_lambda_sched, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_lambda_cap, _resize64(z4_gap_lambda_cap, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_lambda_eff, _resize64(z4_gap_lambda_eff, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_z4_gap_grad_ratio_eff, _resize64(z4_gap_grad_ratio_eff, n_i, fill=0.0), keep_nan=True)
        _extend(
            self._roll_z4_gap_shape_guard_trigger,
            _resize64(z4_gap_shape_guard_trigger, n_i, fill=0.0),
            keep_nan=True,
        )
        _extend(self._roll_t6_main_weight, _resize64(t6_main_weight_step, n_i, fill=1.0), keep_nan=True)
        _extend(self._roll_t6_potential_scale, _resize64(t6_potential_scale_step, n_i, fill=1.0), keep_nan=True)
        _extend(self._roll_t6_traj_scale, _resize64(t6_traj_scale_step, n_i, fill=1.0), keep_nan=True)
        _extend(self._roll_t6_anneal_ratio, _resize64(t6_anneal_ratio_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_t6_violation_ema, _resize64(t6_violation_ema_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_t6_pid_frozen, _resize64(t6_pid_frozen_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_comp_bonus_raw, _resize64(comp_bonus_raw_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_ris_bonus_raw, _resize64(ris_bonus_raw_step, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_reward_raw, _resize64(reward_raw_before_tanh, n_i, fill=0.0), keep_nan=True)
        
        _extend(self._roll_env_shaping, _resize64(env_shaping, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_total_shaping, _resize64(total_shaping, n_i, fill=0.0), keep_nan=True)
        _extend(
            self._roll_total_shaping_all,
            _resize64(env_shaping, n_i, fill=0.0) + _resize64(total_shaping_add, n_i, fill=0.0),
            keep_nan=True,
        )
        _extend(self._roll_comp_bonus, _resize64(comp_bonus, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_ris_bonus, _resize64(ris_bonus, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_comp_gain_ratio, _resize64(comp_gain_ratio, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_ris_gain_ratio, _resize64(ris_gain_ratio, n_i, fill=0.0), keep_nan=True)
        _extend(self._roll_comp_enable_rate, _resize64(comp_enable_rate_step, n_i, fill=np.nan), keep_nan=True)
        _extend(self._roll_service_switch_count, _resize64(service_switch_count_step, n_i, fill=np.nan), keep_nan=True)
        _extend(self._roll_theta_entropy, _resize64(theta_entropy_step, n_i, fill=np.nan), keep_nan=True)
        _extend(
            self._roll_comp_score_temp_runtime,
            _resize64(comp_score_temp_runtime_step, n_i, fill=np.nan),
            keep_nan=True,
        )
        if theta_mode_effective_step:
            self._roll_theta_mode_effective.extend([x for x in theta_mode_effective_step if str(x).strip()])

        # ===== H12 v2: per-step reward normalization =====
        
        
        if self.reward_step_normalize:
            rew_flat = reward_out.astype(np.float64).reshape(-1)
            for v in rew_flat:
                self._rew_norm_n += 1
                delta = v - self._rew_norm_mean
                self._rew_norm_mean += delta / self._rew_norm_n
                delta2 = v - self._rew_norm_mean
                self._rew_norm_m2 += delta * delta2
            if self._rew_norm_n >= self._rew_norm_warmup:
                var = self._rew_norm_m2 / max(self._rew_norm_n, 1)
                std = max(float(np.sqrt(var)), 1e-8)
                reward_out = ((rew_flat - self._rew_norm_mean) / std).astype(np.float32)

        return {
            "obs_next": obs_next_n,
            "rew": reward_out,
            
            "info": self._attach_obs_raw_to_info(info, obs_next),
            
            
            "paper_cost": paper_cost,
            "vio": vio_cnt,
            "vio_metric": vio_metric,
            "improve": improve,
            "env_reward": env_reward,
            "lambda_v": np.full_like(vio_cnt, float(self.lambda_v), dtype=np.float32),
        }


def _parse_list(val, cast_fn):
    """list/tuple/ndarray"""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        out = []
        for x in val:
            try:
                out.append(cast_fn(x))
            except Exception:
                continue
        return out
    s = str(val).strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(cast_fn(part))
        except Exception:
            continue
    return out


class CoMPOnpolicyTrainer(OnpolicyTrainer):
    """Trainer///"""

    def __init__(
        self,
        *,
        run_dir: Path,
        policy: PPOPolicy,
        optimizer: OptimizerBundle,
        train_collector: Collector,
        preprocess: CollectorPreprocess,
        env_cfg_eval: JointEnvConfig,
        obs_normalizer: Optional[ObsNormalizer],
        train_params: dict,
        eval_cfg: dict,
        meta_controller: Optional[MetaController],
        metrics_writer: csv.DictWriter,
        metrics_csv_f,
        total_steps_target: int,
        rollout_steps: int,
        train_epochs: int,
        batch_size: int,
        bc_warmstart_cfg: Optional[Dict[str, Any]] = None,
        bc_supervision_obs: Optional[np.ndarray] = None,
        bc_supervision_act: Optional[np.ndarray] = None,
        bc_supervision_cost: Optional[np.ndarray] = None,
        schedule_step_offset: int = 0,
        curriculum_step_offset: int = 0,
    ):
        max_epoch = int(math.ceil(float(total_steps_target) / float(max(1, rollout_steps))))

        super().__init__(
            policy=policy,
            train_collector=train_collector,
            test_collector=None,
            max_epoch=max_epoch,
            step_per_epoch=int(rollout_steps),
            repeat_per_collect=int(train_epochs),
            episode_per_test=int(eval_cfg.get("eval_episodes", 5)),
            batch_size=int(batch_size),
            step_per_collect=int(rollout_steps),
            
            
            train_fn=self._train_fn,
            test_fn=None,
            save_best_fn=None,
            save_checkpoint_fn=None,
            verbose=False,
            show_progress=False,
            test_in_train=False,
        )

        self.run_dir = run_dir
        self.optimizer = optimizer
        self._optimizer_dual = False
        self._lr_critic_ratio = 1.0
        try:
            _opt_a, _opt_c = _split_optimizer_bundle(self.optimizer)
            self._optimizer_dual = bool(_opt_c is not None)
        except Exception:
            self._optimizer_dual = False
        try:
            ratio_cfg = float(getattr(policy, "_lr_critic_ratio", 1.0))
        except Exception:
            ratio_cfg = 1.0
        if not np.isfinite(ratio_cfg) or ratio_cfg <= 0.0:
            ratio_cfg = 1.0
        self._lr_critic_ratio = float(ratio_cfg)
        self.preprocess = preprocess
        self.env_cfg_eval = env_cfg_eval
        # Keep immutable train-load reference. eval_det/fixed-eval may mutate env_cfg_eval.load_scale in-place.
        self.train_load_nominal = float(getattr(env_cfg_eval, "load_scale", float("nan")))
        self.obs_normalizer = obs_normalizer
        self.train_params = train_params
        
        self.log_returns_stats = bool(train_params.get("log_returns_stats", False))
        self.log_theta_comp_coupling = bool(train_params.get("log_theta_comp_coupling", False))
        self.eval_cfg = eval_cfg
        self.meta_controller = meta_controller
        self.meta_enabled = bool(meta_controller is not None and meta_controller.is_enabled())
        self.meta_source = (
            str(getattr(meta_controller, "source", "none")).strip().lower()
            if self.meta_enabled
            else "none"
        )
        meta_cfg_runtime = {}
        if isinstance(getattr(meta_controller, "cfg", None), dict):
            meta_cfg_runtime = (meta_controller.cfg.get("runtime", {}) or {})
            if not isinstance(meta_cfg_runtime, dict):
                meta_cfg_runtime = {}
        self.meta_strict_train_apply = bool(meta_cfg_runtime.get("strict_train_apply", True))
        self.strict_env_dispatch = bool(train_params.get("strict_env_dispatch", True))
        self.meta_best_after_action = bool(train_params.get("meta_best_after_action", True))
        self._meta_has_applied_action = False
        self._meta_best_reset_done = False
        self._meta_last_decision: Dict[str, Any] = {}

        self.metrics_writer = metrics_writer
        self.metrics_csv_f = metrics_csv_f

        self.total_steps_target = int(total_steps_target)
        self.rollout_steps = int(rollout_steps)
        
        self.schedule_step_offset = int(max(schedule_step_offset, 0))
        self.curriculum_step_offset = int(max(curriculum_step_offset, 0))

        self.save_every = int(train_params.get("save_every", 50000))
        
        
        eval_every_in = int(train_params.get("eval_every", 10000))
        eval_every_rec = int(max(2000, round(0.02 * float(max(self.total_steps_target, 1)))))
        if eval_every_in <= 0:
            self.eval_every = int(eval_every_rec)
        else:
            self.eval_every = int(eval_every_in)
            if int(self.eval_every) > int(eval_every_rec):
                self.eval_every = int(eval_every_rec)
                print(
                    f"[HB][train][PhaseF] eval_everycfg={int(eval_every_in)} -> rec={int(eval_every_rec)}",
                    flush=True,
                )

        self.next_save_step = int(self.save_every) if self.save_every > 0 else 10 ** 18
        self.next_eval_step = int(self.eval_every) if self.eval_every > 0 else 10 ** 18
        
        
        try:
            phase_c = (train_params.get("phase_c", {}) or {})
            if str(phase_c.get("scores_mode", "learned")).strip().lower() == "fixed":
                self.next_eval_step = 0
        except Exception:
            pass

        self.start_time = time.time()

        
        self.best_eval_paper_cost = float("inf")
        self.best_ckpt_step: Optional[int] = None
        self.best_ckpt_path: Optional[Path] = None

        
        self.reward_history: List[float] = []
        self.policy_loss_history: List[float] = []
        self.value_loss_history: List[float] = []
        self.entropy_history: List[float] = []
        self.kl_div_history: List[float] = []
        self.clipfrac_history: List[float] = []
        self.explained_variance_history: List[float] = []
        self.steps_history: List[int] = []
        
        self.train_episode: int = 0
        self.train_episode_history: List[int] = []
        
        self.train_episode_dense_history: List[int] = []
        self.reward_total_ep_dense_history: List[float] = []
        self.cost_ep_dense_history: List[float] = []
        self.vio_ep_dense_history: List[float] = []
        self.improve_ep_dense_history: List[float] = []
        self.shaping_ep_dense_history: List[float] = []
        self.comp_ep_dense_history: List[float] = []
        self.ris_ep_dense_history: List[float] = []
        self.policy_loss_ep_dense_history: List[float] = []
        self.value_loss_ep_dense_history: List[float] = []
        self.entropy_ep_dense_history: List[float] = []
        self.kl_ep_dense_history: List[float] = []
        self.clipfrac_ep_dense_history: List[float] = []
        self.explained_variance_ep_dense_history: List[float] = []
        self.lr_ep_dense_history: List[float] = []
        self.load_ep_dense_history: List[float] = []
        self.cost_history: List[float] = []
        self.vio_history: List[float] = []
        self.improve_history: List[float] = []

        self.eval_steps_history: List[int] = []
        self.eval_train_episode_history: List[int] = []
        self.eval_reward_history: List[float] = []
        self.eval_reward_ci_history: List[float] = []
        self.eval_paper_cost_history: List[float] = []
        self.eval_paper_cost_std_history: List[float] = []
        self.eval_paper_cost_ci_history: List[float] = []
        
        self.eval_paper_return_history: List[float] = []
        self.eval_paper_return_ci_history: List[float] = []
        self.eval_n_history: List[int] = []
        self.eval_vio_history: List[float] = []
        self.eval_vio_any_frac_history: List[float] = []
        self.eval_vio_any_frac_ci_history: List[float] = []
        self.eval_improve_history: List[float] = []
        self.eval_coverage_history: List[float] = []
        self.eval_comp_gain_ratio_history: List[float] = []
        self.eval_ris_gain_ratio_history: List[float] = []
        self.eval_det_paper_cost_ep_dense_history: List[float] = []
        self.eval_det_paper_cost_ci_ep_dense_history: List[float] = []
        self.eval_det_vio_any_frac_ep_dense_history: List[float] = []
        self.eval_det_improve_ep_dense_history: List[float] = []
        self._latest_eval_det_paper_cost: float = float("nan")
        self._latest_eval_det_paper_cost_ci: float = float("nan")
        self._latest_eval_det_vio_any_frac: float = float("nan")
        self._latest_eval_det_improve: float = float("nan")

        
        self._phaseg_reward_total_ep_samples: List[float] = []
        self._phaseg_paper_cost_ep_samples: List[float] = []
        self._phaseg_contract_abs_err_samples: List[float] = []
        self._phaseg_last_spearman: float = float("nan")
        self._phaseg_last_spearman_n: int = 0

        
        self._last_approx_kl: float = float("nan")
        self._last_clipfrac: float = float("nan")
        self._last_entropy: float = float("nan")
        self._last_value_loss: float = float("nan")

        
        self._sched_lr = float("nan")
        self._sched_ent = float("nan")
        self._sched_std_delta = float("nan")
        self._sched_std_score = float("nan")

        
        self._sched_lr, self._sched_ent, self._sched_std_delta, self._sched_std_score = self.update_schedules(0)

        
        bc_cfg = bc_warmstart_cfg or {}
        if not isinstance(bc_cfg, dict):
            bc_cfg = {}
        self.bc_warmstart_enable = bool(bc_cfg.get("enable", False))
        self.bc_warmstart_weight0 = float(max(float(bc_cfg.get("weight0", 0.0)), 0.0))
        self.bc_warmstart_decay_steps = int(max(int(bc_cfg.get("decay_steps", 0)), 1))
        self.bc_warmstart_min_weight = float(max(float(bc_cfg.get("min_weight", 0.0)), 0.0))
        
        self.bc_warmstart_cost_weight0 = float(max(float(bc_cfg.get("cost_weight0", 0.0)), 0.0))
        self.bc_warmstart_cost_decay_steps = int(
            max(int(bc_cfg.get("cost_decay_steps", self.bc_warmstart_decay_steps)), 1)
        )
        self.bc_warmstart_cost_min_weight = float(max(float(bc_cfg.get("cost_min_weight", 0.0)), 0.0))
        self.bc_warmstart_cost_scale = float(max(float(bc_cfg.get("cost_scale", 1.0)), 1e-9))
        self.bc_warmstart_cost_clip = float(max(float(bc_cfg.get("cost_clip", 2.0)), 1e-6))
        self.bc_warmstart_batch_size = int(max(int(bc_cfg.get("batch_size", self.batch_size)), 1))
        self.bc_warmstart_updates = int(max(int(bc_cfg.get("updates_per_rollout", 1)), 1))
        self.bc_warmstart_grad_clip = float(max(float(bc_cfg.get("grad_clip", 0.5)), 0.0))
        
        self.bc_warmstart_require_dataset = bool(bc_cfg.get("require_dataset", True))
        if bc_supervision_obs is None or bc_supervision_act is None:
            self.bc_supervision_obs = None
            self.bc_supervision_act = None
        else:
            self.bc_supervision_obs = np.asarray(bc_supervision_obs, dtype=np.float32)
            self.bc_supervision_act = np.asarray(bc_supervision_act, dtype=np.float32)
            if int(self.bc_supervision_obs.shape[0]) != int(self.bc_supervision_act.shape[0]):
                raise RuntimeError(
                    "BC"
                    f"obs_n={int(self.bc_supervision_obs.shape[0])}, act_n={int(self.bc_supervision_act.shape[0])}"
                )
        if bc_supervision_cost is None:
            self.bc_supervision_cost = None
        else:
            self.bc_supervision_cost = np.asarray(bc_supervision_cost, dtype=np.float32).reshape(-1)
            if (self.bc_supervision_obs is not None) and (
                int(self.bc_supervision_cost.size) != int(self.bc_supervision_obs.shape[0])
            ):
                print(
                    "[HB][train][BC][WARN] bc_supervision_costobs",
                    flush=True,
                )
        if bool(self.bc_warmstart_enable) and bool(self.bc_warmstart_require_dataset):
            if self.bc_supervision_obs is None or self.bc_supervision_act is None:
                raise RuntimeError(
                    " train.bc_warmstart.require_dataset=trueBC"
                    "balanced/greedy_delay/greedy_energy"
                )
        self._bc_rng = np.random.RandomState(int(train_params.get("seed", 42)) + 12017)
        self._bc_last_alpha = 0.0
        self._bc_last_cost_alpha = 0.0
        self._bc_last_loss_raw = float("nan")
        self._bc_last_cost_loss_raw = float("nan")
        self._bc_last_loss_weighted = float("nan")
        self._bc_last_cost_loss_weighted = float("nan")
        self._bc_last_updates = 0
        self._bc_dataset_missing_warned = False

        
        self._tail_stabilize_enable = bool(train_params.get("tail_stabilize_enable", False))
        self._tail_start_frac = float(np.clip(float(train_params.get("tail_start_frac", 0.75)), 0.0, 1.0))
        self._tail_lr_mult = float(np.clip(float(train_params.get("tail_lr_mult", 0.5)), 0.0, 1.0))
        self._tail_ent_mult = float(np.clip(float(train_params.get("tail_ent_mult", 0.5)), 0.0, 1.0))
        self._tail_std_mult = float(np.clip(float(train_params.get("tail_std_mult", 0.8)), 0.0, 1.0))
        self._tail_clip_ratio = float(max(float(train_params.get("tail_clip_ratio", 0.0)), 0.0))
        self._tail_target_kl = float(max(float(train_params.get("tail_target_kl", 0.0)), 0.0))
        self._base_eps_clip = float(getattr(self.policy, "_eps_clip", 0.2))
        self._base_target_kl = float(getattr(self.policy, "_target_kl_stop", 0.0))
        self._tail_active = False
        
        self._tail_assoc_hard_enable = bool(train_params.get("tail_assoc_hard_enable", False))
        self._tail_assoc_mode_base = str(train_params.get("tail_assoc_hard_base_mode", "soft")).strip().lower()
        if self._tail_assoc_mode_base not in ("hard", "soft"):
            self._tail_assoc_mode_base = "soft"
        self._tail_assoc_mode_applied: Optional[str] = None

        
        lag_cfg = train_params.get("lagrange_vio", {}) or {}
        if not isinstance(lag_cfg, dict):
            lag_cfg = {}
        self._lag_bucket_enable = bool(lag_cfg.get("load_bucket_enable", False))
        lag_edges = _parse_list(lag_cfg.get("load_bucket_edges", [0.7, 0.9]), float)
        lag_edges = sorted([float(np.clip(x, 0.0, 2.0)) for x in lag_edges])
        self._lag_bucket_edges = [float(x) for x in lag_edges]
        nb = int(max(len(self._lag_bucket_edges) + 1, 1))

        base_target = float(lag_cfg.get("target", 0.1))
        base_eta = float(lag_cfg.get("eta", 0.005))
        base_lam_min = float(max(float(lag_cfg.get("min", 0.0)), 0.0))
        base_lam_max = float(max(float(lag_cfg.get("max", 100.0)), base_lam_min))

        tg = _parse_list(lag_cfg.get("load_bucket_targets", []), float)
        et = _parse_list(lag_cfg.get("load_bucket_etas", []), float)
        mn = _parse_list(lag_cfg.get("load_bucket_mins", []), float)
        mx = _parse_list(lag_cfg.get("load_bucket_maxs", []), float)
        if len(tg) != nb:
            tg = [float(base_target)] * nb
        if len(et) != nb:
            et = [float(base_eta)] * nb
        if len(mn) != nb:
            mn = [float(base_lam_min)] * nb
        if len(mx) != nb:
            mx = [float(base_lam_max)] * nb

        self._lag_bucket_targets = [float(x) for x in tg]
        self._lag_bucket_etas = [float(max(float(x), 0.0)) for x in et]
        self._lag_bucket_mins = [float(max(float(x), 0.0)) for x in mn]
        self._lag_bucket_maxs = [
            float(max(float(mx_i), float(mn_i)))
            for mn_i, mx_i in zip(self._lag_bucket_mins, mx)
        ]
        self._lag_bucket_last_id = -1
        self._lag_target_current = float(base_target)
        self._lag_eta_current = float(base_eta)
        self._lag_lambda_min_current = float(base_lam_min)
        self._lag_lambda_max_current = float(base_lam_max)
        self._lag_bucket_last_load = float("nan")
        self._lag_bucket_logged = False
        if bool(self._lag_bucket_enable):
            edges_s = ",".join([f"{x:.2f}" for x in self._lag_bucket_edges]) if self._lag_bucket_edges else "-"
            t_s = ",".join([f"{x:.3f}" for x in self._lag_bucket_targets])
            e_s = ",".join([f"{x:.4f}" for x in self._lag_bucket_etas])
            m_s = ",".join([f"{a:.3f}:{b:.3f}" for a, b in zip(self._lag_bucket_mins, self._lag_bucket_maxs)])
            print(
                f"[HB][train][LAG] bucket=1 edges=[{edges_s}] target=[{t_s}] eta=[{e_s}] min:max=[{m_s}]",
                flush=True,
            )

        
        self._head_tune_enable = bool(train_params.get("head_tune_enable", False))
        self._ent_coef_delta_init = float(train_params.get("ent_coef_delta_init", train_params.get("ent_coef_init", train_params.get("ent_coef", 0.0))))
        self._ent_coef_delta_final = float(train_params.get("ent_coef_delta_final", train_params.get("ent_coef_final", 0.0)))
        self._ent_coef_delta_decay_steps = int(max(int(train_params.get("ent_coef_delta_decay_steps", train_params.get("ent_coef_decay_steps", max(1, self.total_steps_target // 2)))), 1))
        self._ent_coef_score_init = float(train_params.get("ent_coef_score_init", train_params.get("ent_coef_init", train_params.get("ent_coef", 0.0))))
        self._ent_coef_score_final = float(train_params.get("ent_coef_score_final", train_params.get("ent_coef_final", 0.0)))
        self._ent_coef_score_decay_steps = int(max(int(train_params.get("ent_coef_score_decay_steps", train_params.get("ent_coef_decay_steps", max(1, self.total_steps_target // 2)))), 1))
        self._clip_ratio_delta = float(max(float(train_params.get("clip_ratio_delta", train_params.get("clip_ratio", self._base_eps_clip))), 1e-4))
        self._clip_ratio_score = float(max(float(train_params.get("clip_ratio_score", train_params.get("clip_ratio", self._base_eps_clip))), 1e-4))
        self._tail_clip_ratio_delta = float(max(float(train_params.get("tail_clip_ratio_delta", 0.0)), 0.0))
        self._tail_clip_ratio_score = float(max(float(train_params.get("tail_clip_ratio_score", 0.0)), 0.0))
        self._value_clip_ratio = float(max(float(train_params.get("value_clip_ratio", train_params.get("clip_ratio", self._base_eps_clip))), 1e-4))
        self._tail_value_clip_ratio = float(max(float(train_params.get("tail_value_clip_ratio", 0.0)), 0.0))
        self._clip_delta_weight = float(max(float(train_params.get("clip_delta_weight", 0.0)), 0.0))
        self._clip_score_weight = float(max(float(train_params.get("clip_score_weight", 0.0)), 0.0))

        
        self._train_load_sampling_mode = str(train_params.get("train_load_sampling_mode", "fixed")).strip().lower()
        
        _cfg_loads_raw = _parse_list(train_params.get("train_load_sampling_loads", None), float)
        _default_sampling_enable = (
            self._train_load_sampling_mode in ("high_load_weighted", "weighted", "high_load_tiered", "tiered")
            or (self._train_load_sampling_mode == "fixed" and len(_cfg_loads_raw) > 1)
        )
        self._train_load_sampling_enable = bool(
            train_params.get("train_load_sampling_enable", _default_sampling_enable)
        )
        self._train_load_sampling_high_threshold = float(
            np.clip(float(train_params.get("train_load_sampling_high_threshold", 0.7)), 0.0, 1.0)
        )
        self._train_load_sampling_high_weight = float(max(float(train_params.get("train_load_sampling_high_weight", 2.0)), 1.0))
        tier_th = _parse_list(train_params.get("train_load_sampling_tier_thresholds", [0.7, 0.85, 0.95]), float)
        tier_w = _parse_list(train_params.get("train_load_sampling_tier_weights", [2.0, 2.8, 3.6]), float)
        if (not tier_th) or (len(tier_th) != len(tier_w)):
            tier_th = [0.7, 0.85, 0.95]
            tier_w = [2.0, 2.8, 3.6]
        pairs = sorted(
            [
                (float(np.clip(float(th), 0.0, 2.0)), float(max(float(ww), 1.0)))
                for th, ww in zip(tier_th, tier_w)
            ],
            key=lambda kv: kv[0],
        )
        self._train_load_sampling_tier_thresholds = [float(k) for k, _ in pairs]
        self._train_load_sampling_tier_weights = [float(v) for _, v in pairs]
        self._train_load_sampling_load_power = float(max(float(train_params.get("train_load_sampling_load_power", 0.0)), 0.0))
        cfg_loads = _parse_list(train_params.get("train_load_sampling_loads", None), float)
        eval_loads = _parse_list(eval_cfg.get("eval_loads", None), float)
        if cfg_loads:
            sample_loads = [float(x) for x in cfg_loads]
        elif eval_loads:
            sample_loads = [float(x) for x in eval_loads]
        else:
            sample_loads = [float(getattr(self.env_cfg_eval, "load_scale", 1.0))]
        sample_loads = [float(np.clip(x, 0.05, 2.0)) for x in sample_loads]
        sample_loads = sorted(list(dict.fromkeys([round(x, 6) for x in sample_loads])))
        self._train_load_sampling_loads = [float(x) for x in sample_loads]
        self._train_load_sampling_probs = self._build_train_load_sampling_probs(self._train_load_sampling_loads)
        self._load_rng = np.random.RandomState(int(train_params.get("seed", 42)) + 9187)
        self._load_sampling_logged = False
        self._train_load_sampling_force_switch = bool(train_params.get("train_load_sampling_force_switch", False))
        
        try:
            _nom = float(getattr(self, "train_load_nominal", float("nan")))
            if not np.isfinite(_nom):
                _nom = float(getattr(self.env_cfg_eval, "load_scale", 0.8))
        except Exception:
            _nom = 0.8
        self._curriculum_load_last = float(_nom)

        
        self._resume_enable: bool = False
        self._resume_env_step: int = 0
        self._resume_train_episode: int = 0
        self._resume_applied: bool = False

    def reset(self) -> None:  # type: ignore[override]
        """
        BaseTrainer.resetrun_dir env_step 

        Tianshou BaseTrainer.__iter__  reset() env_step=0
        
        - schedule0
        - train_metrics.csv  global_step 
        """
        super().reset()
        try:
            if bool(getattr(self, "_resume_enable", False)) and (not bool(getattr(self, "_resume_applied", False))):
                st = int(getattr(self, "_resume_env_step", 0) or 0)
                if st > 0:
                    self.env_step = int(st)
                    
                    self._sched_lr, self._sched_ent, self._sched_std_delta, self._sched_std_score = self.update_schedules(int(st))
                    
                    if int(self.save_every) > 0:
                        self.next_save_step = int((int(st) // int(self.save_every) + 1) * int(self.save_every))
                    if int(self.eval_every) > 0:
                        self.next_eval_step = int((int(st) // int(self.eval_every) + 1) * int(self.eval_every))
                ep = int(getattr(self, "_resume_train_episode", 0) or 0)
                if ep > 0:
                    self.train_episode = int(ep)
                self._resume_applied = True
        except Exception:
            pass

    def _train_fn(self, epoch: int, env_step: int) -> None:
        """Tianshoucollectlr/entropy/sigma_schedule"""
        self._sched_lr, self._sched_ent, self._sched_std_delta, self._sched_std_score = self.update_schedules(int(env_step))
        
        try:
            if bool(getattr(self, "_train_load_sampling_enable", False)):
                self._maybe_apply_load_sampling(int(env_step))
            else:
                self._maybe_apply_load_curriculum(int(env_step))
        except Exception as e:
            lvl = "FATAL" if bool(getattr(self, "strict_env_dispatch", False)) else "WARN"
            print(f"[HB][train][CUR][{lvl}] load curriculum failed: {type(e).__name__}: {e}", flush=True)
            if bool(getattr(self, "strict_env_dispatch", False)):
                raise
        
        try:
            self._maybe_apply_tail_association_mode(int(env_step))
        except Exception as e:
            lvl = "FATAL" if bool(getattr(self, "strict_env_dispatch", False)) else "WARN"
            print(f"[HB][train][TAIL][{lvl}] association_mode dispatch failed: {type(e).__name__}: {e}", flush=True)
            if bool(getattr(self, "strict_env_dispatch", False)):
                raise
        
        try:
            if hasattr(self.preprocess, "reset_rollout_stats"):
                self.preprocess.reset_rollout_stats()
        except Exception:
            pass

    def _resolve_lagrange_bucket(self, load_now: float, lag_cfg: Dict[str, Any]) -> Tuple[int, float, float, float, float]:
        """
        B5 Lagrange 

        bucket_id, target, eta, lam_min, lam_max
        """
        base_target = float(lag_cfg.get("target", 0.1))
        base_eta = float(max(float(lag_cfg.get("eta", 0.005)), 0.0))
        base_lam_min = float(max(float(lag_cfg.get("min", 0.0)), 0.0))
        base_lam_max = float(max(float(lag_cfg.get("max", 100.0)), base_lam_min))
        if not bool(getattr(self, "_lag_bucket_enable", False)):
            return 0, float(base_target), float(base_eta), float(base_lam_min), float(base_lam_max)

        edges = list(getattr(self, "_lag_bucket_edges", []) or [])
        idx = int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(load_now), side="right"))
        idx = int(max(0, min(idx, max(len(edges), 0))))

        tg = list(getattr(self, "_lag_bucket_targets", []) or [])
        et = list(getattr(self, "_lag_bucket_etas", []) or [])
        mn = list(getattr(self, "_lag_bucket_mins", []) or [])
        mx = list(getattr(self, "_lag_bucket_maxs", []) or [])

        target = float(tg[idx]) if idx < len(tg) else float(base_target)
        eta = float(et[idx]) if idx < len(et) else float(base_eta)
        lam_min = float(mn[idx]) if idx < len(mn) else float(base_lam_min)
        lam_max = float(mx[idx]) if idx < len(mx) else float(base_lam_max)
        lam_min = float(max(lam_min, 0.0))
        lam_max = float(max(lam_max, lam_min))
        return int(idx), float(target), float(max(eta, 0.0)), float(lam_min), float(lam_max)

    def _bc_action_cost_surrogate(self, mu: torch.Tensor) -> torch.Tensor:
        """
        

        
        - L1
        - scoresigmoidCoMP
        - 
        """
        if mu.ndim != 2:
            return torch.zeros((mu.shape[0],), dtype=mu.dtype, device=mu.device)

        d_dim = int(getattr(self.policy.actor, "delta_dim", 0))
        d_dim = int(max(min(d_dim, int(mu.shape[1])), 0))
        if d_dim > 0:
            move_term = torch.mean(torch.abs(mu[:, :d_dim]), dim=1)
        else:
            move_term = torch.zeros((mu.shape[0],), dtype=mu.dtype, device=mu.device)

        score_term = torch.zeros((mu.shape[0],), dtype=mu.dtype, device=mu.device)
        if int(mu.shape[1]) > d_dim:
            score_raw = mu[:, d_dim:]
            
            score_prob = torch.sigmoid(score_raw / 0.5)
            score_term = torch.mean(score_prob, dim=1)

        out = move_term + 0.35 * score_term
        out = out / float(self.bc_warmstart_cost_scale)
        clip_v = float(self.bc_warmstart_cost_clip)
        if clip_v > 0.0:
            out = torch.clamp(out, min=0.0, max=clip_v)
        return out

    def _run_bc_warmstart_update(self) -> Tuple[float, float, int]:
        """
        BCPPO
        - alpha  env_step 
        -  actor 
        -  (loss_raw, loss_weighted, updates) 
        """
        if not bool(self.bc_warmstart_enable):
            self._bc_last_alpha = 0.0
            self._bc_last_cost_alpha = 0.0
            self._bc_last_loss_raw = float("nan")
            self._bc_last_cost_loss_raw = float("nan")
            self._bc_last_loss_weighted = float("nan")
            self._bc_last_cost_loss_weighted = float("nan")
            self._bc_last_updates = 0
            return float("nan"), float("nan"), 0
        if self.bc_supervision_obs is None or self.bc_supervision_act is None:
            if bool(self.bc_warmstart_enable) and (not bool(self._bc_dataset_missing_warned)):
                print(
                    "[HB][train][BC][WARN] BCrolloutBC",
                    flush=True,
                )
                self._bc_dataset_missing_warned = True
            self._bc_last_alpha = 0.0
            self._bc_last_cost_alpha = 0.0
            self._bc_last_loss_raw = float("nan")
            self._bc_last_cost_loss_raw = float("nan")
            self._bc_last_loss_weighted = float("nan")
            self._bc_last_cost_loss_weighted = float("nan")
            self._bc_last_updates = 0
            return float("nan"), float("nan"), 0
        n = int(self.bc_supervision_obs.shape[0])
        if n <= 0:
            self._bc_last_alpha = 0.0
            self._bc_last_cost_alpha = 0.0
            self._bc_last_loss_raw = float("nan")
            self._bc_last_cost_loss_raw = float("nan")
            self._bc_last_loss_weighted = float("nan")
            self._bc_last_cost_loss_weighted = float("nan")
            self._bc_last_updates = 0
            return float("nan"), float("nan"), 0

        alpha = float(
            _linear_schedule(
                start=float(self.bc_warmstart_weight0),
                end=float(self.bc_warmstart_min_weight),
                step=int(self.env_step),
                decay_steps=int(self.bc_warmstart_decay_steps),
            )
        )
        alpha = float(max(alpha, float(self.bc_warmstart_min_weight)))

        
        cost_alpha = float(
            _linear_schedule(
                start=float(self.bc_warmstart_cost_weight0),
                end=float(self.bc_warmstart_cost_min_weight),
                step=int(self.env_step),
                decay_steps=int(self.bc_warmstart_cost_decay_steps),
            )
        )
        cost_alpha = float(max(cost_alpha, float(self.bc_warmstart_cost_min_weight)))
        has_cost_supervision = (
            self.bc_supervision_cost is not None
            and int(np.asarray(self.bc_supervision_cost).reshape(-1).size) == int(n)
            and cost_alpha > 1e-12
        )

        has_bc_supervision = bool(alpha > 1e-12)
        if not has_bc_supervision and (not has_cost_supervision):
            self._bc_last_alpha = 0.0
            self._bc_last_cost_alpha = float(cost_alpha if has_cost_supervision else 0.0)
            self._bc_last_loss_raw = float("nan")
            self._bc_last_cost_loss_raw = float("nan")
            self._bc_last_loss_weighted = float("nan")
            self._bc_last_cost_loss_weighted = float("nan")
            self._bc_last_updates = 0
            return float("nan"), float("nan"), 0

        self.policy.train()
        loss_raw_ema: Optional[float] = None
        loss_cost_raw_ema: Optional[float] = None
        updates_done = 0
        for _ in range(int(self.bc_warmstart_updates)):
            bs = int(min(self.bc_warmstart_batch_size, n))
            idx = self._bc_rng.randint(0, n, size=bs)
            dev = next(self.policy.actor.parameters()).device
            obs_np = np.asarray(self.bc_supervision_obs[idx], dtype=np.float32)
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev)
            act_t = torch.as_tensor(self.bc_supervision_act[idx], dtype=torch.float32, device=dev)
            (mu, _sigma), _ = self.policy.actor(obs_t, info={"obs_raw": obs_np})
            loss_raw = torch.mean((mu - act_t) ** 2)
            loss = torch.zeros((), dtype=mu.dtype, device=mu.device)
            if has_bc_supervision:
                loss = loss + float(alpha) * loss_raw

            if has_cost_supervision:
                cost_t = torch.as_tensor(self.bc_supervision_cost[idx], dtype=torch.float32, device=dev)
                cost_pred = self._bc_action_cost_surrogate(mu)
                loss_cost_raw = torch.mean((cost_pred - cost_t) ** 2)
                loss = loss + float(cost_alpha) * loss_cost_raw
            else:
                loss_cost_raw = torch.zeros((), dtype=mu.dtype, device=mu.device)

            actor_optim = _optimizer_actor_only(self.optimizer)
            actor_optim.zero_grad()
            loss.backward()
            if self.bc_warmstart_grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), float(self.bc_warmstart_grad_clip))
            actor_optim.step()

            lv = float(loss_raw.detach().cpu().item())
            loss_raw_ema = lv if loss_raw_ema is None else (0.9 * float(loss_raw_ema) + 0.1 * lv)
            lc = float(loss_cost_raw.detach().cpu().item())
            loss_cost_raw_ema = lc if loss_cost_raw_ema is None else (0.9 * float(loss_cost_raw_ema) + 0.1 * lc)
            updates_done += 1

        raw = float(loss_raw_ema if loss_raw_ema is not None else float("nan"))
        raw_cost = float(loss_cost_raw_ema if loss_cost_raw_ema is not None else float("nan"))
        weighted = float(alpha * raw) if np.isfinite(raw) else float("nan")
        weighted_cost = float(cost_alpha * raw_cost) if (np.isfinite(raw_cost) and has_cost_supervision) else float("nan")
        if np.isfinite(weighted_cost):
            weighted = float(weighted + weighted_cost)
        self._bc_last_alpha = float(alpha)
        self._bc_last_cost_alpha = float(cost_alpha if has_cost_supervision else 0.0)
        self._bc_last_loss_raw = float(raw)
        self._bc_last_cost_loss_raw = float(raw_cost)
        self._bc_last_loss_weighted = float(weighted)
        self._bc_last_cost_loss_weighted = float(weighted_cost)
        self._bc_last_updates = int(updates_done)
        return raw, weighted, int(updates_done)

    def _build_train_load_sampling_probs(self, loads: List[float]) -> List[float]:
        """"""
        if not loads:
            return [1.0]
        mode = str(getattr(self, "_train_load_sampling_mode", "fixed")).strip().lower()
        if mode not in ("high_load_weighted", "weighted", "high_load_tiered", "tiered"):
            n = int(len(loads))
            return [1.0 / float(max(n, 1))] * n
        hi_th = float(getattr(self, "_train_load_sampling_high_threshold", 0.7))
        hi_w = float(getattr(self, "_train_load_sampling_high_weight", 2.0))
        tiers = list(
            zip(
                list(getattr(self, "_train_load_sampling_tier_thresholds", []) or []),
                list(getattr(self, "_train_load_sampling_tier_weights", []) or []),
            )
        )
        load_power = float(max(float(getattr(self, "_train_load_sampling_load_power", 0.0)), 0.0))
        w = []
        for x in loads:
            xx = float(x)
            if mode in ("high_load_tiered", "tiered"):
                ww = 1.0
                for th, tw in tiers:
                    if xx >= float(th):
                        ww = max(float(ww), float(tw))
            else:
                ww = float(hi_w) if xx >= float(hi_th) else 1.0
            if load_power > 0.0:
                ww = float(ww) * float(max(xx, 1e-6) ** load_power)
            w.append(float(max(ww, 1e-9)))
        s = float(sum(w))
        if s <= 1e-12:
            n = int(len(loads))
            return [1.0 / float(max(n, 1))] * n
        return [float(v / s) for v in w]

    def _maybe_apply_load_sampling(self, env_step: int) -> None:
        """
        
        - collect
        - 
        """
        loads = list(getattr(self, "_train_load_sampling_loads", []) or [])
        probs = list(getattr(self, "_train_load_sampling_probs", []) or [])
        if not loads:
            return
        if len(probs) != len(loads):
            probs = self._build_train_load_sampling_probs(loads)
            self._train_load_sampling_probs = list(probs)
        pick = int(self._load_rng.choice(np.arange(len(loads)), p=np.asarray(probs, dtype=np.float64)))
        load_now = float(loads[pick])

        last = float(getattr(self, "_curriculum_load_last", float("nan")))
        if np.isfinite(last) and abs(load_now - last) < 1e-9:
            if bool(getattr(self, "_train_load_sampling_force_switch", False)) and len(loads) > 1:
                alt_ids = [i for i, lv in enumerate(loads) if abs(float(lv) - float(last)) >= 1e-9]
                if not alt_ids:
                    return
                alt_probs = np.asarray([float(probs[i]) for i in alt_ids], dtype=np.float64)
                prob_sum = float(np.sum(alt_probs))
                if (not np.isfinite(prob_sum)) or prob_sum <= 1e-12:
                    alt_probs = np.full(len(alt_ids), 1.0 / float(len(alt_ids)), dtype=np.float64)
                else:
                    alt_probs = alt_probs / prob_sum
                alt_pick = int(self._load_rng.choice(np.arange(len(alt_ids)), p=alt_probs))
                load_now = float(loads[int(alt_ids[alt_pick])])
            else:
                return
        setattr(self, "_curriculum_load_last", float(load_now))

        if not bool(getattr(self, "_load_sampling_logged", False)):
            setattr(self, "_load_sampling_logged", True)
            probs_s = ",".join([f"{float(p):.3f}" for p in probs])
            loads_s = ",".join([f"{float(x):.2f}" for x in loads])
            tiers = list(
                zip(
                    list(getattr(self, "_train_load_sampling_tier_thresholds", []) or []),
                    list(getattr(self, "_train_load_sampling_tier_weights", []) or []),
                )
            )
            tiers_s = ",".join([f"{float(th):.2f}:{float(tw):.2f}" for th, tw in tiers]) if tiers else "-"
            print(
                f"[HB][train][LOAD] sampling=1 mode={self._train_load_sampling_mode} "
                f"loads=[{loads_s}] probs=[{probs_s}] "
                f"high_th={float(self._train_load_sampling_high_threshold):.2f} "
                f"high_w={float(self._train_load_sampling_high_weight):.2f} "
                f"tiers={tiers_s} pwr={float(getattr(self, '_train_load_sampling_load_power', 0.0)):.2f} "
                f"force_switch={int(bool(getattr(self, '_train_load_sampling_force_switch', False)))}",
                flush=True,
            )

        dispatch = self._dispatch_train_env_method(
            method_name="set_load_scale",
            method_args=(float(load_now),),
            pending_attr="_pending_load_scale",
            pending_value=float(load_now),
        )
        eff = int(dispatch.get("applied", 0)) + int(dispatch.get("pending", 0))
        if eff <= 0:
            msg = (
                f"load_sampling dispatch failed: load={float(load_now):.6f} "
                f"path={dispatch.get('path', [])} errors={dispatch.get('errors', [])}"
            )
            if self.strict_env_dispatch:
                raise RuntimeError(msg)
            print(f"[HB][train][LOAD][WARN] {msg}", flush=True)

    def _maybe_apply_load_curriculum(self, env_step: int) -> None:
        """
        
        - train.enable_curriculum  train.curriculum
        - env_cfg.load_scale
        - eval.eval_loads 
        - 
        """
        p = self.train_params or {}
        enabled = bool(p.get("enable_curriculum", p.get("curriculum", False)))
        if not enabled:
            return

        
        load_start = float(getattr(self, "train_load_nominal", float("nan")))
        if not np.isfinite(load_start):
            load_start = float(getattr(self.env_cfg_eval, "load_scale", 1.0))

        
        eval_loads = _parse_list(self.eval_cfg.get("eval_loads", None), float)
        load_target = float(max(eval_loads)) if eval_loads else float(load_start)

        total_steps = int(self.total_steps_target)
        
        
        
        
        ramp_steps_cfg = int(p.get("curriculum_ramp_steps", 0) or 0)
        if ramp_steps_cfg > 0:
            ramp_steps = int(max(1, ramp_steps_cfg))
        else:
            ramp_ratio = float(p.get("curriculum_ramp_ratio", 1.0 / 3.0) or (1.0 / 3.0))
            ramp_ratio = float(np.clip(ramp_ratio, 1e-3, 1.0))
            ramp_steps = int(max(1, round(float(total_steps) * ramp_ratio)))
        ramp_steps = int(min(max(1, ramp_steps), max(1, total_steps)))
        env_step_eff = int(max(int(env_step), 0) + int(max(getattr(self, "curriculum_step_offset", 0), 0)))
        t = float(np.clip(float(env_step_eff) / float(max(ramp_steps, 1)), 0.0, 1.0))
        load_now = float(load_start + (load_target - load_start) * t)

        
        last = float(getattr(self, "_curriculum_load_last", float("nan")))
        if np.isfinite(last) and abs(load_now - last) < 1e-6:
            return
        setattr(self, "_curriculum_load_last", float(load_now))

        
        if not bool(getattr(self, "_curriculum_load_inited", False)):
            setattr(self, "_curriculum_load_inited", True)
            print(
                f"[HB][train][CUR] enable_curriculum=1 load_start={load_start:.3f} "
                f"load_target={load_target:.3f} ramp_steps={ramp_steps} total_steps={total_steps} "
                f"step_offset={int(getattr(self, 'curriculum_step_offset', 0))}",
                flush=True,
            )

        
        dispatch = self._dispatch_train_env_method(
            method_name="set_load_scale",
            method_args=(float(load_now),),
            pending_attr="_pending_load_scale",
            pending_value=float(load_now),
        )
        eff = int(dispatch.get("applied", 0)) + int(dispatch.get("pending", 0))
        if eff <= 0:
            msg = (
                f"load_curriculum dispatch failed: load={float(load_now):.6f} "
                f"path={dispatch.get('path', [])} errors={dispatch.get('errors', [])}"
            )
            if self.strict_env_dispatch:
                raise RuntimeError(msg)
            print(f"[HB][train][CUR][WARN] {msg}", flush=True)
        elif int(dispatch.get("pending", 0)) > 0 and (not bool(getattr(self, "_cur_pending_noted", False))):
            setattr(self, "_cur_pending_noted", True)
            print(
                f"[HB][train][CUR] pending_apply via workers/set_env_attr count={int(dispatch.get('pending', 0))}",
                flush=True,
            )

    def _maybe_apply_tail_association_mode(self, env_step: int) -> None:
        """
        
        -  base  soft
        -  hard
        """
        if not bool(getattr(self, "_tail_assoc_hard_enable", False)):
            return
        desired = "hard" if bool(getattr(self, "_tail_active", False)) else str(getattr(self, "_tail_assoc_mode_base", "soft"))
        if desired not in ("hard", "soft"):
            desired = "soft"
        if str(getattr(self, "_tail_assoc_mode_applied", "")) == desired:
            return

        dispatch = self._dispatch_train_env_method(
            method_name="set_association_mode",
            method_args=(str(desired),),
            pending_attr="_pending_association_mode",
            pending_value=str(desired),
        )
        eff = int(dispatch.get("applied", 0)) + int(dispatch.get("pending", 0))
        if eff <= 0:
            msg = (
                f"tail association dispatch failed: desired={desired} "
                f"path={dispatch.get('path', [])} errors={dispatch.get('errors', [])}"
            )
            if self.strict_env_dispatch:
                raise RuntimeError(msg)
            print(f"[HB][train][TAIL][WARN] {msg}", flush=True)
            return

        prev_mode = str(getattr(self, "_tail_assoc_mode_applied", "unset"))
        self._tail_assoc_mode_applied = str(desired)
        print(
            f"[HB][train][TAIL] association_mode switch: {prev_mode} -> {desired} "
            f"tail={1 if bool(getattr(self, '_tail_active', False)) else 0} step={int(env_step)}",
            flush=True,
        )

    def _dispatch_train_env_method(
        self,
        *,
        method_name: str,
        method_args: Tuple[Any, ...] = (),
        method_kwargs: Optional[Dict[str, Any]] = None,
        pending_attr: Optional[str] = None,
        pending_value: Any = None,
    ) -> Dict[str, Any]:
        """
         env_method/envs/workers 

        
        - applied: 
        - pending:  set_env_attr 
        - results: 
        - path: 
        - errors: 
        """
        out: Dict[str, Any] = {
            "applied": 0,
            "pending": 0,
            "results": [],
            "path": [],
            "errors": [],
        }
        kwargs = dict(method_kwargs) if isinstance(method_kwargs, dict) else {}
        env = getattr(self.train_collector, "env", None) if self.train_collector is not None else None
        if env is None:
            out["errors"].append("train_collector.env is None")
            return out

        
        if hasattr(env, "env_method"):
            try:
                res = env.env_method(str(method_name), *tuple(method_args), **kwargs)
                seq = list(res) if isinstance(res, (list, tuple)) else [res]
                out["results"].extend(seq)
                out["applied"] += int(len(seq))
                out["path"].append("env_method")
                return out
            except Exception as e:
                out["errors"].append(f"env_method:{type(e).__name__}:{e}")

        
        envs = getattr(env, "envs", None)
        if isinstance(envs, (list, tuple)):
            out["path"].append("envs")
            for ee in envs:
                try:
                    fn = getattr(ee, str(method_name), None)
                    if callable(fn):
                        out["results"].append(fn(*tuple(method_args), **kwargs))
                        out["applied"] += 1
                except Exception as e:
                    out["errors"].append(f"envs:{type(e).__name__}:{e}")

        
        workers = getattr(env, "workers", None)
        if isinstance(workers, (list, tuple)):
            out["path"].append("workers")
            for w in workers:
                ok = False
                
                ee = getattr(w, "env", None)
                if ee is not None:
                    try:
                        fn = getattr(ee, str(method_name), None)
                        if callable(fn):
                            out["results"].append(fn(*tuple(method_args), **kwargs))
                            out["applied"] += 1
                            ok = True
                    except Exception as e:
                        out["errors"].append(f"worker.env:{type(e).__name__}:{e}")
                
                if (not ok) and hasattr(w, "get_env_attr"):
                    try:
                        fn_any = w.get_env_attr(str(method_name))
                        if callable(fn_any):
                            out["results"].append(fn_any(*tuple(method_args), **kwargs))
                            out["applied"] += 1
                            ok = True
                    except Exception as e:
                        out["errors"].append(f"worker.get_env_attr:{type(e).__name__}:{e}")
                
                if (not ok) and pending_attr and hasattr(w, "set_env_attr"):
                    try:
                        w.set_env_attr(str(pending_attr), pending_value)
                        out["pending"] += 1
                        ok = True
                    except Exception as e:
                        out["errors"].append(f"worker.set_env_attr:{type(e).__name__}:{e}")

        
        if int(out["applied"]) <= 0 and int(out["pending"]) <= 0 and pending_attr and hasattr(env, "set_env_attr"):
            try:
                env.set_env_attr(str(pending_attr), pending_value)
                n = len(getattr(env, "workers", []) or []) or 1
                out["pending"] = int(n)
                out["path"].append("set_env_attr")
            except Exception as e:
                out["errors"].append(f"env.set_env_attr:{type(e).__name__}:{e}")

        return out

    def _meta_current_params(self) -> Dict[str, float]:
        return {
            "lambda_v": float(getattr(self.env_cfg_eval, "lambda_v", 0.0)),
            "beta_comp_max": float(getattr(self.env_cfg_eval, "beta_comp_max", 0.0)),
            "beta_ris_max": float(getattr(self.env_cfg_eval, "beta_ris_max", 0.0)),
        }

    def _apply_meta_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "applied": False,
            "results": [],
            "train_env_applied": 0,
            "train_env_pending": 0,
            "eval_cfg_applied": 0,
            "preprocess_applied": 0,
            "errors": [],
        }
        if not isinstance(patch, dict) or not patch:
            return out

        
        dispatch = self._dispatch_train_env_method(
            method_name="apply_meta_action",
            method_args=(dict(patch),),
            pending_attr="_pending_meta_action",
            pending_value=dict(patch),
        )
        out["results"] = list(dispatch.get("results", []))
        out["train_env_applied"] = int(dispatch.get("applied", 0))
        out["train_env_pending"] = int(dispatch.get("pending", 0))
        if dispatch.get("errors"):
            out["errors"].extend(list(dispatch.get("errors", [])))

        
        for k, v in patch.items():
            try:
                if hasattr(self.env_cfg_eval, str(k)):
                    setattr(self.env_cfg_eval, str(k), float(v))
                    out["eval_cfg_applied"] = int(out.get("eval_cfg_applied", 0)) + 1
            except Exception:
                continue

        
        if "lambda_v" in patch:
            try:
                if hasattr(self.preprocess, "set_lambda_v"):
                    self.preprocess.set_lambda_v(float(patch["lambda_v"]))
                    out["preprocess_applied"] = 1
            except Exception:
                pass

        out["applied"] = bool(int(out["train_env_applied"]) > 0 or int(out["train_env_pending"]) > 0)
        return out

    def _maybe_meta_step(
        self,
        *,
        eval_out: Dict[str, Any],
        train_episode_now: int,
        clipfrac_now: float,
        kl_per_dim_now: float,
        explained_variance_now: float,
    ) -> Dict[str, Any]:
        if not self.meta_enabled or self.meta_controller is None:
            return {}
        eval_index = int(len(self.eval_steps_history))
        if not self.meta_controller.should_trigger(eval_index):
            return {}

        try:
            window = int(
                max(
                    2,
                    float(
                        (
                            (self.meta_controller.cfg.get("state", {}) if isinstance(self.meta_controller.cfg, dict) else {})
                            .get("window", 4)
                        )
                    ),
                )
            )
        except Exception:
            window = 4

        coverage_hist = list(getattr(self, "eval_coverage_history", []))
        if len(coverage_hist) <= 0:
            coverage_hist = [float(eval_out.get("eval_det_coverage_fraction_mean", float("nan")))]
        comp_gain_hist = list(getattr(self, "eval_comp_gain_ratio_history", []))
        if len(comp_gain_hist) <= 0:
            comp_gain_hist = [float(eval_out.get("comp_gain_ratio", float("nan")))]
        ris_gain_hist = list(getattr(self, "eval_ris_gain_ratio_history", []))
        if len(ris_gain_hist) <= 0:
            ris_gain_hist = [float(eval_out.get("ris_gain_ratio", float("nan")))]

        hist = {
            "paper_cost": list(self.eval_paper_cost_history),
            "paper_return": list(self.eval_paper_return_history),
            "vio_any": list(self.eval_vio_any_frac_history),
            "coverage": coverage_hist,
            "clipfrac": list(self.clipfrac_history),
            "kl_per_dim": list(self.kl_div_history),
            "explained_variance": list(self.explained_variance_history),
            "comp_gain_ratio": comp_gain_hist,
            "ris_gain_ratio": ris_gain_hist,
        }
        latest = {
            "paper_cost": float(eval_out.get("eval_det_paper_cost_mean", float("nan"))),
            "paper_return": float(eval_out.get("eval_det_paper_return_mean", float("nan"))),
            "vio_any": float(eval_out.get("eval_det_vio_any_frac_mean", float("nan"))),
            "coverage": float(eval_out.get("eval_det_coverage_fraction_mean", float("nan"))),
            "clipfrac": float(clipfrac_now),
            "kl_per_dim": float(kl_per_dim_now),
            "explained_variance": float(explained_variance_now),
            "comp_gain_ratio": float(eval_out.get("comp_gain_ratio", float("nan"))),
            "ris_gain_ratio": float(eval_out.get("ris_gain_ratio", float("nan"))),
            "eval_load": float(eval_out.get("eval_det_loads_primary", float("nan"))),
            "train_load": float(getattr(self, "train_load_nominal", float("nan"))),
            "train_load_current": float(getattr(self, "_curriculum_load_last", getattr(self, "train_load_nominal", float("nan")))),
            "traj_idle_ratio": float(eval_out.get("traj_idle_ratio", float("nan"))),
            "traj_boundary_stick_frac": float(eval_out.get("traj_boundary_stick_frac", float("nan"))),
            "traj_user_nn_dist_norm": float(eval_out.get("traj_user_nn_dist_norm", float("nan"))),
            "traj_centroid_gap_norm": float(eval_out.get("traj_centroid_gap_norm", float("nan"))),
            "traj_switchback_ratio": float(eval_out.get("traj_switchback_ratio", float("nan"))),
        }
        state = build_meta_state(history=hist, latest=latest, window=window)
        cur = self._meta_current_params()
        context = {
            "env_step": int(self.env_step),
            "train_episode": int(train_episode_now),
            "eval_index": int(eval_index),
            "run_dir": str(self.run_dir),
        }

        decision = self.meta_controller.decide(state=state, current_params=cur, context=context)
        patch = decision.get("patch", {}) if isinstance(decision, dict) else {}
        apply_out = self._apply_meta_patch(patch if isinstance(patch, dict) else {})
        if isinstance(patch, dict) and patch:
            train_eff = int(apply_out.get("train_env_applied", 0)) + int(apply_out.get("train_env_pending", 0))
            if train_eff <= 0:
                msg = (
                    f"meta patch not delivered to train env: patch={patch} "
                    f"errors={apply_out.get('errors', [])}"
                )
                if self.meta_strict_train_apply:
                    raise RuntimeError(msg)
                print(f"[HB][meta][WARN] {msg}", flush=True)
            else:
                self._meta_has_applied_action = True
                
                if self.meta_best_after_action and (not bool(self._meta_best_reset_done)):
                    self.best_eval_paper_cost = float("inf")
                    self.best_ckpt_step = -1
                    self.best_ckpt_path = None
                    self._meta_best_reset_done = True
                    self._meta_skip_best_once = True
                    print("[HB][meta] reset best-ckpt window after first applied meta action", flush=True)
        if isinstance(decision, dict):
            decision["apply_result"] = apply_out
            decision["state"] = state
        self.meta_controller.append_decision(decision)
        self._meta_last_decision = dict(decision) if isinstance(decision, dict) else {}
        return self._meta_last_decision

    def update_schedules(self, env_step: int) -> Tuple[float, float, float, float]:
        """lr/entropy/sigma_scalecollect"""
        total_steps = int(self.total_steps_target)
        step_eff = int(max(int(env_step), 0) + int(max(getattr(self, "schedule_step_offset", 0), 0)))
        p = self.train_params

        
        ent_init = float(p.get("ent_coef_init", p.get("ent_coef", 0.0)))
        ent_final = float(p.get("ent_coef_final", 0.0))
        ent_decay = int(p.get("ent_coef_decay_steps", max(1, total_steps // 2)))
        ent_now = _linear_schedule(ent_init, ent_final, int(step_eff), int(ent_decay))
        ent_delta_now = float(ent_now)
        ent_score_now = float(ent_now)
        progress = float(np.clip(float(step_eff) / float(max(total_steps, 1)), 0.0, 1.0))
        tail_enable = bool(getattr(self, "_tail_stabilize_enable", False))
        tail_start_frac = float(getattr(self, "_tail_start_frac", 0.75))
        tail_active = bool(tail_enable and (progress >= float(tail_start_frac)))
        self._tail_active = bool(tail_active)
        if tail_active:
            ent_now = float(max(ent_final, float(ent_now) * float(getattr(self, "_tail_ent_mult", 0.5))))
        if bool(getattr(self, "_head_tune_enable", False)):
            ent_delta_now = float(
                _linear_schedule(
                    float(getattr(self, "_ent_coef_delta_init", ent_init)),
                    float(getattr(self, "_ent_coef_delta_final", ent_final)),
                    int(step_eff),
                    int(getattr(self, "_ent_coef_delta_decay_steps", ent_decay)),
                )
            )
            ent_score_now = float(
                _linear_schedule(
                    float(getattr(self, "_ent_coef_score_init", ent_init)),
                    float(getattr(self, "_ent_coef_score_final", ent_final)),
                    int(step_eff),
                    int(getattr(self, "_ent_coef_score_decay_steps", ent_decay)),
                )
            )
            if tail_active:
                ent_tail_mult = float(getattr(self, "_tail_ent_mult", 0.5))
                ent_delta_now = float(max(float(getattr(self, "_ent_coef_delta_final", ent_final)), ent_delta_now * ent_tail_mult))
                ent_score_now = float(max(float(getattr(self, "_ent_coef_score_final", ent_final)), ent_score_now * ent_tail_mult))

        
        
        
        
        std_init = float(p.get("exploration_std_scale_init", 1.0))
        std_final = float(p.get("exploration_std_scale_final", 0.2))
        std_decay = int(p.get("exploration_std_decay_steps", max(1, total_steps // 2)))
        
        
        if ("exploration_std_min" in p) and ("sigma_min" in p):
            v_new = float(p["exploration_std_min"])
            v_old = float(p["sigma_min"])
            if abs(v_new - v_old) > 1e-9:
                raise ValueError(
                    f"[train] exploration_std_min={v_new}  sigma_min={v_old} "
                    f" exploration_std_min sigma_min "
                    f"  sigma_min exploration_std_min"
                )
        std_min = float(p.get("exploration_std_min", p.get("sigma_min", 0.0)))
        if ("exploration_std_min" not in p) and ("sigma_min" in p):
            _warn_once(
                "exploration_std_min_alias",
                " train.sigma_min exploration_std_min  exploration_std_min",
            )

        std_delta_init = float(p.get("exploration_std_scale_delta_init", std_init))
        std_delta_final = float(p.get("exploration_std_scale_delta_final", std_final))
        std_delta_decay = int(p.get("exploration_std_delta_decay_steps", std_decay))
        std_score_init = float(p.get("exploration_std_scale_score_init", std_init))
        std_score_final = float(p.get("exploration_std_scale_score_final", std_final))
        std_score_decay = int(p.get("exploration_std_score_decay_steps", std_decay))
        
        std_cm_init = float(p.get("exploration_std_scale_comp_meta_init", std_score_init))
        std_cm_final = float(p.get("exploration_std_scale_comp_meta_final", std_score_final))
        std_cm_decay = int(p.get("exploration_std_comp_meta_decay_steps", std_score_decay))

        std_delta_now = _linear_schedule(std_delta_init, std_delta_final, int(step_eff), int(std_delta_decay))
        std_score_now = _linear_schedule(std_score_init, std_score_final, int(step_eff), int(std_score_decay))
        std_cm_now = _linear_schedule(std_cm_init, std_cm_final, int(step_eff), int(std_cm_decay))
        if tail_active:
            tail_std_mult = float(getattr(self, "_tail_std_mult", 0.8))
            std_delta_now = float(max(std_min, float(std_delta_now) * tail_std_mult))
            std_score_now = float(max(std_min, float(std_score_now) * tail_std_mult))
            std_cm_now = float(max(std_min, float(std_cm_now) * tail_std_mult))

        
        if bool(getattr(self, "_guard_freeze_updates", False)):
            ent_now = float(ent_final)
            try:
                self.policy._weight_ent = float(ent_now)
            except Exception:
                pass
            std_delta_now = float(std_delta_final)
            std_score_now = float(std_score_final)
            std_cm_now = float(std_cm_final)

        
        
        try:
            phase_c = p.get("phase_c", {}) or {}
            if bool(phase_c.get("enforce_sigma_delta_gt_score", False)):
                if float(std_score_now) >= float(std_delta_now):
                    
                    std_score_now = max(float(std_min), 0.80 * float(std_delta_now))
        except Exception:
            pass
        try:
            if hasattr(self.policy.actor, "set_sigma_scales"):
                self.policy.actor.set_sigma_scales(
                    scale_delta=float(std_delta_now),
                    scale_score=float(std_score_now),
                    sigma_min=float(std_min),
                    scale_comp_meta=float(std_cm_now),
                )
            else:
                
                self.policy.actor.set_sigma_scale(float(std_score_now), sigma_min=float(std_min))
        except Exception:
            pass

        
        lr = float(p.get("lr", 3e-5))
        min_lr = float(p.get("min_lr", lr))
        lr_anneal = bool(p.get("lr_anneal", True))
        warmup_cfg = int(p.get("lr_warmup_steps", 0))
        if warmup_cfg > 0:
            _warn_once(
                "lr_warmup_deprecated",
                f" lr_warmup_steps={warmup_cfg} warmup ",
            )

        if lr_anneal:
            denom = max(total_steps, 1)
            t = float(max(step_eff, 0)) / float(denom)
            t = float(np.clip(t, 0.0, 1.0))
            lr_now = lr + (min_lr - lr) * t
        else:
            lr_now = lr
        if tail_active:
            lr_now = float(max(min_lr, float(lr_now) * float(getattr(self, "_tail_lr_mult", 0.5))))

        if bool(getattr(self, "_guard_freeze_updates", False)):
            
            lr_now = float(min_lr)

        
        try:
            base_eps = float(getattr(self, "_base_eps_clip", getattr(self.policy, "_eps_clip", 0.2)))
            tail_clip_ratio = float(getattr(self, "_tail_clip_ratio", 0.0))
            eps_clip_now = float(base_eps)
            eps_clip_delta_now = float(max(float(getattr(self, "_clip_ratio_delta", base_eps)), 1e-6))
            eps_clip_score_now = float(max(float(getattr(self, "_clip_ratio_score", base_eps)), 1e-6))
            if tail_active and tail_clip_ratio > 0.0:
                eps_clip_now = float(min(base_eps, tail_clip_ratio))
                tail_delta = float(getattr(self, "_tail_clip_ratio_delta", 0.0))
                tail_score = float(getattr(self, "_tail_clip_ratio_score", 0.0))
                eps_clip_delta_now = float(min(eps_clip_delta_now, tail_delta if tail_delta > 0.0 else tail_clip_ratio))
                eps_clip_score_now = float(min(eps_clip_score_now, tail_score if tail_score > 0.0 else tail_clip_ratio))
            eps_clip_value_now = float(max(float(getattr(self, "_value_clip_ratio", base_eps)), 1e-6))
            if tail_active:
                tail_value = float(getattr(self, "_tail_value_clip_ratio", 0.0))
                if tail_value > 0.0:
                    eps_clip_value_now = float(min(eps_clip_value_now, tail_value))
            eps_clip_now = float(min(eps_clip_now, eps_clip_delta_now, eps_clip_score_now))
            self.policy._eps_clip = float(max(eps_clip_now, 1e-4))
            self.policy._eps_clip_delta = float(max(eps_clip_delta_now, 1e-4))
            self.policy._eps_clip_score = float(max(eps_clip_score_now, 1e-4))
            self.policy._eps_clip_value = float(max(eps_clip_value_now, 1e-4))
        except Exception:
            pass
        try:
            base_kl = float(getattr(self, "_base_target_kl", getattr(self.policy, "_target_kl_stop", 0.0)))
            tail_target_kl = float(getattr(self, "_tail_target_kl", 0.0))
            kl_now = float(base_kl)
            if tail_active and tail_target_kl > 0.0:
                if base_kl > 0.0:
                    kl_now = float(min(base_kl, tail_target_kl))
                else:
                    kl_now = float(tail_target_kl)
            self.policy._target_kl_stop = float(max(kl_now, 0.0))
        except Exception:
            pass
        try:
            self.policy._weight_ent = float(ent_now)
            self.policy._weight_ent_delta = float(ent_delta_now)
            self.policy._weight_ent_score = float(ent_score_now)
            self.policy._head_tune_enable = bool(getattr(self, "_head_tune_enable", False))
            self.policy._clip_delta_weight = float(getattr(self, "_clip_delta_weight", 0.0))
            self.policy._clip_score_weight = float(getattr(self, "_clip_score_weight", 0.0))
            self.policy._head_delta_dim = int(getattr(self.policy.actor, "delta_dim", 0))
            self.policy._tail_active = bool(tail_active)
        except Exception:
            pass

        lr_critic_now = float(lr_now) * float(max(getattr(self, "_lr_critic_ratio", 1.0), 1e-6))
        try:
            _optimizer_set_lr(
                self.optimizer,
                lr_actor=float(lr_now),
                lr_critic=float(lr_critic_now),
            )
        except Exception as exc:
            _warn_once("optimizer_set_lr_failed", f"{type(exc).__name__}: {exc}")

        return float(lr_now), float(ent_now), float(std_delta_now), float(std_score_now)

    def save_ckpt(self, tag: str) -> Path:
        ckpt_dir = self.run_dir / CKPT_DIR_NAME
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"ckpt_{tag}.pt"
        
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": _optimizer_export_state(self.optimizer),
                "history": self._build_ckpt_history(),
                "meta": {
                    "global_step": int(self.env_step),
                    "train_episode": int(getattr(self, "train_episode", 0) or 0),
                },
            },
            str(path),
        )

        
        try:
            if self.obs_normalizer is not None:
                st = self.obs_normalizer.get_stats()
                
                # - best: normalizer_best.npz
                # - step_XXXX: normalizer_stepXXXX.npz
                
                norm_name = f"normalizer_{tag}.npz"
                if str(tag).startswith("step_"):
                    step_str = str(tag)[len("step_"):].strip()
                    if step_str.isdigit():
                        norm_name = f"normalizer_step{step_str}.npz"
                np.savez(
                    ckpt_dir / norm_name,
                    count=int(st["count"]),
                    mean=np.asarray(st["mean"], dtype=np.float32),
                    var=np.asarray(st["var"], dtype=np.float32),
                )
                
                np.savez(
                    ckpt_dir / "normalizer_latest.npz",
                    count=int(st["count"]),
                    mean=np.asarray(st["mean"], dtype=np.float32),
                    var=np.asarray(st["var"], dtype=np.float32),
                )
        except Exception:
            pass
        return path

    def _build_ckpt_history(self) -> Dict[str, Any]:
        """
        checkpointhistory

        PhaseE
        - ep_paper_return_mean := -ep_paper_cost_meandet_evalseeds
        - ep_reward_meanlegacy
        """
        steps_eval = list(self.eval_steps_history)
        episodes_eval = list(getattr(self, "eval_train_episode_history", []))
        if len(episodes_eval) != len(steps_eval):
            episodes_eval = []
        ep_paper_cost_mean = list(self.eval_paper_cost_history)
        ep_paper_cost_ci = list(self.eval_paper_cost_ci_history)

        
        if self.eval_paper_return_history:
            ep_paper_return_mean = list(self.eval_paper_return_history)
        else:
            ep_paper_return_mean = [-float(x) for x in ep_paper_cost_mean]

        if self.eval_paper_return_ci_history:
            ep_paper_return_ci = list(self.eval_paper_return_ci_history)
        else:
            
            ep_paper_return_ci = list(ep_paper_cost_ci)

        return {
            
            "steps": steps_eval,
            "episodes": episodes_eval,
            "ep_paper_cost_mean": ep_paper_cost_mean,
            "ep_paper_cost_ci": ep_paper_cost_ci,
            "ep_paper_return_mean": ep_paper_return_mean,
            "ep_paper_return_ci": ep_paper_return_ci,
            
            "ep_reward_mean": list(self.eval_reward_history),
            "ep_reward_ci": list(self.eval_reward_ci_history),
            "eval_coverage_history": list(getattr(self, "eval_coverage_history", [])),
            "eval_comp_gain_ratio_history": list(getattr(self, "eval_comp_gain_ratio_history", [])),
            "eval_ris_gain_ratio_history": list(getattr(self, "eval_ris_gain_ratio_history", [])),
            
            "train_steps": list(self.steps_history),
            "train_episodes": list(self.train_episode_history) if len(self.train_episode_history) == len(self.steps_history) else [],
            "train_reward_mean": list(self.reward_history),
            "train_episodes_dense": list(self.train_episode_dense_history)
            if len(self.train_episode_dense_history) == len(self.reward_total_ep_dense_history)
            else [],
            "train_reward_total_ep_dense": list(self.reward_total_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.reward_total_ep_dense_history)
            else [],
            "train_cost_ep_dense": list(self.cost_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.cost_ep_dense_history)
            else [],
            "train_vio_ep_dense": list(self.vio_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.vio_ep_dense_history)
            else [],
            "train_improve_ep_dense": list(self.improve_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.improve_ep_dense_history)
            else [],
            "train_shaping_ep_dense": list(self.shaping_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.shaping_ep_dense_history)
            else [],
            "train_comp_ep_dense": list(self.comp_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.comp_ep_dense_history)
            else [],
            "train_ris_ep_dense": list(self.ris_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.ris_ep_dense_history)
            else [],
            "train_policy_loss_ep_dense": list(self.policy_loss_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.policy_loss_ep_dense_history)
            else [],
            "train_value_loss_ep_dense": list(self.value_loss_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.value_loss_ep_dense_history)
            else [],
            "train_entropy_ep_dense": list(self.entropy_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.entropy_ep_dense_history)
            else [],
            "train_kl_ep_dense": list(self.kl_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.kl_ep_dense_history)
            else [],
            "train_clipfrac_ep_dense": list(self.clipfrac_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.clipfrac_ep_dense_history)
            else [],
            "train_explained_variance_ep_dense": list(self.explained_variance_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.explained_variance_ep_dense_history)
            else [],
            "train_lr_ep_dense": list(self.lr_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.lr_ep_dense_history)
            else [],
            "train_load_ep_dense": list(self.load_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.load_ep_dense_history)
            else [],
            "eval_det_paper_cost_ep_dense": list(self.eval_det_paper_cost_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.eval_det_paper_cost_ep_dense_history)
            else [],
            "eval_det_paper_cost_ci_ep_dense": list(self.eval_det_paper_cost_ci_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.eval_det_paper_cost_ci_ep_dense_history)
            else [],
            "eval_det_vio_any_frac_ep_dense": list(self.eval_det_vio_any_frac_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.eval_det_vio_any_frac_ep_dense_history)
            else [],
            "eval_det_improve_ep_dense": list(self.eval_det_improve_ep_dense_history)
            if len(self.train_episode_dense_history) == len(self.eval_det_improve_ep_dense_history)
            else [],
            "clipfrac_history": list(self.clipfrac_history),
            "explained_variance_history": list(self.explained_variance_history),
        }

    def det_eval(self) -> Dict[str, float]:
        """mubest"""
        device = next(self.policy.parameters()).device  # type: ignore
        rd_eval = str(getattr(self.env_cfg_eval, "reward_design", "")).strip().lower()
        use_phaseg_ep_sum = bool(rd_eval in ("reward_total_v1", "reward_total", "total_v1", "v1"))
        eval_loads = _parse_list(self.eval_cfg.get("eval_loads", None), float)
        if not eval_loads:
            eval_loads = [float(getattr(self.env_cfg_eval, "load_scale", 1.0))]
        eval_seeds = _parse_list(self.eval_cfg.get("eval_seeds", None), int)
        if not eval_seeds:
            eval_seeds = [42, 43, 44]
        eval_ep_per = int(self.eval_cfg.get("eval_ep_per", self.eval_cfg.get("eval_episodes", 5)))

        det_returns: List[float] = []
        det_paper_cost: List[float] = []
        det_vio: List[float] = []
        det_vio_any_frac: List[float] = []
        det_vio_metric: List[float] = []
        det_cov_frac: List[float] = []
        det_base_balanced_paper: List[float] = []
        det_base_greedy_delay_paper: List[float] = []
        det_base_greedy_energy_paper: List[float] = []
        det_imp_vs_balanced: List[float] = []
        det_imp_vs_greedy_delay: List[float] = []
        det_imp_vs_greedy_energy: List[float] = []
        det_traj_idle_ratio: List[float] = []
        det_traj_boundary_stick_frac: List[float] = []
        det_traj_user_nn_dist_norm: List[float] = []
        det_traj_centroid_gap_norm: List[float] = []
        det_traj_switchback_ratio: List[float] = []
        total_eps = 0

        mCount = int(self.env_cfg_eval.M)
        L_eval = float(getattr(self.env_cfg_eval, "L", 1.0))
        boundary_band = 0.05 * float(L_eval)
        vnorm_denom = max(
            1e-9,
            float(getattr(self.env_cfg_eval, "Vmax", 1.0)) * float(getattr(self.env_cfg_eval, "dt", 1.0)),
        )

        
        
        
        eval_preprocess = CollectorPreprocess(
            obs_normalizer=self.obs_normalizer,
            update_obs=False,
            update_improve_z=False,
            reward_override=str(getattr(self.preprocess, "reward_override", "env")),
            paper_reward_map=str(getattr(self.preprocess, "paper_reward_map", "tanh")),
            paper_reward_bias=float(getattr(self.preprocess, "paper_reward_bias", 0.0)),
            paper_reward_scale=float(getattr(self.preprocess, "paper_reward_scale", 1.0)),
            paper_reward_tanh_s=float(getattr(self.preprocess, "paper_reward_tanh_s", 0.20)),
            paper_reward_clip_c=float(getattr(self.preprocess, "paper_reward_clip_c", 5.0)),
            improve_mode=str(getattr(self.preprocess, "improve_mode", "paper_improve")),
            rel_improve_eps=float(getattr(self.preprocess, "rel_improve_eps", 1e-3)),
            improve_z_eps=float(getattr(self.preprocess, "improve_z_eps", 1e-6)),
            paper_vio_gating_power=float(getattr(self.preprocess, "paper_vio_gating_power", 1.0)),
            paper_vio_penalty=float(getattr(self.preprocess, "paper_vio_penalty", 0.0)),
            reward_clip_range=float(getattr(self.preprocess, "reward_clip_range", 0.0)),
            vio_scale=float(getattr(self.preprocess, "vio_scale", 1.0)),
            lagrange_enabled=bool(getattr(self.preprocess, "lagrange_enabled", False)),
            lagrange_init=float(getattr(self.preprocess, "lambda_v", 0.0)),
            enable_comp_ris_shaping=bool(getattr(self.preprocess, "enable_comp_ris_shaping", False)),
            comp_weight=float(getattr(self.preprocess, "comp_weight", 0.0)),
            ris_weight=float(getattr(self.preprocess, "ris_weight", 0.0)),
            shaping_ratio=float(getattr(self.preprocess, "shaping_ratio", 0.10)),
        )
        
        try:
            eval_preprocess.set_lambda_v(float(getattr(self.preprocess, "lambda_v", 0.0)))
        except Exception:
            pass
        try:
            eval_preprocess._impz_n = int(getattr(self.preprocess, "_impz_n", 0))  # type: ignore[attr-defined]
            eval_preprocess._impz_mean = float(getattr(self.preprocess, "_impz_mean", 0.0))  # type: ignore[attr-defined]
            eval_preprocess._impz_m2 = float(getattr(self.preprocess, "_impz_m2", 0.0))  # type: ignore[attr-defined]
        except Exception:
            pass

        self.policy.eval()
        for load in eval_loads:
            for sd in eval_seeds:
                
                env = CompRISEnvJoint(dc_replace(self.env_cfg_eval), seed=int(sd))
                env.set_load_scale(float(load))
                for _ep in range(int(eval_ep_per)):
                    obs = env.reset()
                    
                    out0 = eval_preprocess(obs=obs, info={})
                    if "obs" in out0:
                        obs = out0["obs"]
                    info0 = out0.get("info", {})
                    try:
                        obs_raw = np.asarray(info0.get("obs_raw", obs), dtype=np.float32)
                    except Exception:
                        obs_raw = np.asarray(obs, dtype=np.float32)

                    done = False
                    ep_return = 0.0
                    ep_steps = 0
                    paper_sum = 0.0
                    vio_sum = 0.0
                    vio_any_steps = 0.0
                    vio_metric_sum = 0.0
                    cov_sum = 0.0
                    base_balanced_sum = 0.0
                    base_greedy_delay_sum = 0.0
                    base_greedy_energy_sum = 0.0
                    imp_vs_balanced_sum = 0.0
                    imp_vs_greedy_delay_sum = 0.0
                    imp_vs_greedy_energy_sum = 0.0
                    traj_idle_hits = 0.0
                    traj_idle_total = 0.0
                    traj_boundary_hits = 0.0
                    traj_boundary_total = 0.0
                    traj_user_nn_sum = 0.0
                    traj_user_nn_total = 0.0
                    traj_centroid_sum = 0.0
                    traj_centroid_total = 0.0
                    traj_switchback_hits = 0.0
                    traj_switchback_total = 0.0
                    q_prev = None
                    vel_prev = None
                    def _safe_float(v: Any, default: float = 0.0) -> float:
                        try:
                            vv = float(v)
                        except Exception:
                            return float(default)
                        return float(vv) if np.isfinite(vv) else float(default)

                    while not done:
                        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            
                            (mu, _sigma), _ = self.policy.actor(obs_t, info={"obs_raw": obs_raw})
                        act = mu.squeeze(0).detach().cpu().numpy().astype(np.float64)
                        
                        

                        obs2, r_env, done, info_env = env.step(act)
                        
                        out1 = eval_preprocess(obs_next=obs2, rew=r_env, info=info_env)
                        r_train = out1.get("rew", r_env)
                        try:
                            ep_return += float(np.asarray(r_train, dtype=np.float64).reshape(-1)[0])
                        except Exception:
                            ep_return += float(r_env)
                        ep_steps += 1
                        info = out1.get("info", info_env) or {}

                        paper_sum += _safe_float(
                            info.get("policy_paper_cost", info.get("paper_cost", info.get("cost", 0.0))),
                            default=0.0,
                        )
                        vio_step = _safe_float(
                            info.get(
                                "violation_count",
                                info.get(
                                    "penalty_total",
                                    info.get("cost_constraint", info.get("vio_total_penalty", 0.0)),
                                ),
                            ),
                            default=0.0,
                        )
                        vio_sum += float(vio_step)
                        if float(vio_step) > 0.0:
                            vio_any_steps += 1.0
                        cov = _safe_float(info.get("coverage_fraction", 0.0), default=0.0)
                        cov = float(np.clip(cov, 0.0, 1.0))
                        cm = float(max(_safe_float(info.get("coverage_margin", 0.0), default=0.0), 0.0))
                        lm = float(max(_safe_float(info.get("collision_margin", 0.0), default=0.0), 0.0))
                        vio_metric_sum += float(np.clip((1.0 - cov) + 0.5 * cm + 0.5 * lm, 0.0, 1.0))
                        cov_sum += cov
                        pol_cost = _safe_float(
                            info.get("policy_paper_cost", info.get("paper_cost", info.get("cost", 0.0))),
                            default=0.0,
                        )
                        b_bal = _safe_float(info.get("baseline_balanced_paper_cost", 0.0), default=0.0)
                        b_delay = _safe_float(info.get("baseline_greedy_delay_paper_cost", 0.0), default=0.0)
                        b_energy = _safe_float(info.get("baseline_greedy_energy_paper_cost", 0.0), default=0.0)
                        base_balanced_sum += b_bal
                        base_greedy_delay_sum += b_delay
                        base_greedy_energy_sum += b_energy
                        imp_vs_balanced_sum += float(b_bal - pol_cost)
                        imp_vs_greedy_delay_sum += float(b_delay - pol_cost)
                        imp_vs_greedy_energy_sum += float(b_energy - pol_cost)

                        obs = out1.get("obs_next", obs2)
                        try:
                            obs_raw = np.asarray(info.get("obs_raw", obs2), dtype=np.float32)
                        except Exception:
                            obs_raw = np.asarray(obs2, dtype=np.float32)

                        try:
                            q_now = np.asarray(getattr(env, "q", None), dtype=np.float64)
                            if q_now.ndim == 2 and q_now.shape[1] == 2 and q_now.shape[0] > 0:
                                M_now = int(q_now.shape[0])
                                d_left = q_now[:, 0]
                                d_right = float(L_eval) - q_now[:, 0]
                                d_bottom = q_now[:, 1]
                                d_top = float(L_eval) - q_now[:, 1]
                                d_min = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
                                traj_boundary_hits += float(np.sum(d_min <= float(boundary_band)))
                                traj_boundary_total += float(M_now)

                                w_now = np.asarray(getattr(env, "w", None), dtype=np.float64)
                                if w_now.ndim == 2 and w_now.shape[1] == 2 and w_now.shape[0] > 0:
                                    diff = q_now[:, None, :] - w_now[None, :, :]
                                    d = np.sqrt(np.sum(diff * diff, axis=2))
                                    d_nn = np.min(d, axis=1)
                                    traj_user_nn_sum += float(np.sum(d_nn) / max(float(L_eval), 1e-9))
                                    traj_user_nn_total += float(M_now)
                                    gap = np.linalg.norm(np.mean(q_now, axis=0) - np.mean(w_now, axis=0))
                                    traj_centroid_sum += float(gap / max(float(L_eval), 1e-9))
                                    traj_centroid_total += 1.0

                                if q_prev is not None and np.asarray(q_prev).shape == q_now.shape:
                                    vel = q_now - q_prev
                                    sp = np.linalg.norm(vel, axis=1) / float(vnorm_denom)
                                    traj_idle_hits += float(np.sum(sp < 0.08))
                                    traj_idle_total += float(M_now)

                                    if vel_prev is not None and np.asarray(vel_prev).shape == vel.shape:
                                        dot = np.sum(vel * vel_prev, axis=1)
                                        n0 = np.linalg.norm(vel_prev, axis=1)
                                        n1 = np.linalg.norm(vel, axis=1)
                                        valid = (n0 > 1e-9) & (n1 > 1e-9)
                                        if np.any(valid):
                                            cosv = dot[valid] / (n0[valid] * n1[valid] + 1e-12)
                                            traj_switchback_hits += float(np.sum(cosv < 0.0))
                                            traj_switchback_total += float(np.sum(valid))
                                    vel_prev = vel
                                else:
                                    vel_prev = None
                                q_prev = q_now.copy()
                        except Exception:
                            pass

                    det_returns.append(float(ep_return))
                    steps = float(max(1, ep_steps))
                    
                    
                    if bool(use_phaseg_ep_sum):
                        det_paper_cost.append(float(paper_sum))
                    else:
                        det_paper_cost.append(float(paper_sum / steps))
                    det_vio.append(float(vio_sum / steps))
                    det_vio_any_frac.append(float(vio_any_steps / steps))
                    det_vio_metric.append(float(vio_metric_sum / steps))
                    det_cov_frac.append(float(cov_sum / steps))
                    det_base_balanced_paper.append(float(base_balanced_sum / steps))
                    det_base_greedy_delay_paper.append(float(base_greedy_delay_sum / steps))
                    det_base_greedy_energy_paper.append(float(base_greedy_energy_sum / steps))
                    det_imp_vs_balanced.append(float(imp_vs_balanced_sum / steps))
                    det_imp_vs_greedy_delay.append(float(imp_vs_greedy_delay_sum / steps))
                    det_imp_vs_greedy_energy.append(float(imp_vs_greedy_energy_sum / steps))
                    det_traj_idle_ratio.append(float(traj_idle_hits / max(1.0, traj_idle_total)))
                    det_traj_boundary_stick_frac.append(float(traj_boundary_hits / max(1.0, traj_boundary_total)))
                    det_traj_user_nn_dist_norm.append(float(traj_user_nn_sum / max(1.0, traj_user_nn_total)))
                    det_traj_centroid_gap_norm.append(float(traj_centroid_sum / max(1.0, traj_centroid_total)))
                    det_traj_switchback_ratio.append(float(traj_switchback_hits / max(1.0, traj_switchback_total)))
                    total_eps += 1

        n_eval = int(total_eps)

        def _finite_arr(vals: List[float]) -> np.ndarray:
            arr = np.asarray(vals, dtype=np.float64).reshape(-1)
            return arr[np.isfinite(arr)]

        def _mean_std_ci(vals: List[float]) -> Tuple[float, float, float]:
            arr = _finite_arr(vals)
            n = int(arr.size)
            if n <= 0:
                return 0.0, 0.0, 0.0
            mu = float(np.mean(arr))
            std = float(np.std(arr))
            ci = float(1.96 * std / np.sqrt(max(1, n))) if n > 1 else 0.0
            return mu, std, ci

        
        ret_mean, ret_std, ret_ci = _mean_std_ci(det_returns)
        paper_mean, paper_std, paper_ci = _mean_std_ci(det_paper_cost)
        
        paper_return_mean = float(-paper_mean)
        paper_return_std = float(paper_std)
        paper_return_ci = float(paper_ci)

        vio_mean, _vio_std, _vio_ci = _mean_std_ci(det_vio)
        vio_any_mean, vio_any_std, vio_any_ci = _mean_std_ci(det_vio_any_frac)
        vio_metric_mean, _vmm_std, _vmm_ci = _mean_std_ci(det_vio_metric)
        cov_mean, _cov_std, _cov_ci = _mean_std_ci(det_cov_frac)
        base_bal_mean, _bb_std, _bb_ci = _mean_std_ci(det_base_balanced_paper)
        base_delay_mean, _bd_std, _bd_ci = _mean_std_ci(det_base_greedy_delay_paper)
        base_energy_mean, _be_std, _be_ci = _mean_std_ci(det_base_greedy_energy_paper)
        imp_bal_mean, _imp_bal_std, _imp_bal_ci = _mean_std_ci(det_imp_vs_balanced)
        imp_delay_mean, _imp_delay_std, _imp_delay_ci = _mean_std_ci(det_imp_vs_greedy_delay)
        imp_energy_mean, _imp_energy_std, _imp_energy_ci = _mean_std_ci(det_imp_vs_greedy_energy)
        traj_idle_mean, _ti_std, _ti_ci = _mean_std_ci(det_traj_idle_ratio)
        traj_boundary_mean, _tb_std, _tb_ci = _mean_std_ci(det_traj_boundary_stick_frac)
        traj_user_nn_mean, _tun_std, _tun_ci = _mean_std_ci(det_traj_user_nn_dist_norm)
        traj_centroid_mean, _tc_std, _tc_ci = _mean_std_ci(det_traj_centroid_gap_norm)
        traj_switchback_mean, _ts_std, _ts_ci = _mean_std_ci(det_traj_switchback_ratio)

        out = {
            
            "eval_det_is_valid": float(1.0 if int(n_eval) > 0 else 0.0),
            "eval_det_return_mean": float(ret_mean),
            "eval_det_return_std": float(ret_std),
            "eval_det_return_ci": float(ret_ci),
            "eval_det_return_ci95": float(ret_ci),
            "eval_det_paper_cost_mean": float(paper_mean),
            "eval_det_paper_cost_std": float(paper_std),
            "eval_det_paper_cost_ci": float(paper_ci),
            "eval_det_paper_cost_ci95": float(paper_ci),
            "eval_det_paper_return_mean": float(paper_return_mean),
            "eval_det_paper_return_std": float(paper_return_std),
            "eval_det_paper_return_ci": float(paper_return_ci),
            "eval_det_paper_return_ci95": float(paper_return_ci),
            "eval_det_n": int(n_eval),
            "eval_det_vio_mean": float(vio_mean),
            "eval_det_vio_any_frac_mean": float(vio_any_mean),
            "eval_det_vio_any_frac_std": float(vio_any_std),
            "eval_det_vio_any_frac_ci": float(vio_any_ci),
            "eval_det_vio_any_frac_ci95": float(vio_any_ci),
            "eval_det_vio_metric_mean": float(vio_metric_mean),
            "eval_det_coverage_fraction_mean": float(cov_mean),
            "eval_det_baseline_balanced_paper_cost_mean": float(base_bal_mean),
            "eval_det_baseline_greedy_delay_paper_cost_mean": float(base_delay_mean),
            "eval_det_baseline_greedy_energy_paper_cost_mean": float(base_energy_mean),
            
            "eval_det_imp_paper_cost_mean": float(imp_bal_mean),
            "eval_det_imp_vs_balanced_paper_cost_mean": float(imp_bal_mean),
            "eval_det_imp_vs_greedy_delay_paper_cost_mean": float(imp_delay_mean),
            "eval_det_imp_vs_greedy_energy_paper_cost_mean": float(imp_energy_mean),
            "traj_idle_ratio": float(traj_idle_mean),
            "traj_boundary_stick_frac": float(traj_boundary_mean),
            "traj_user_nn_dist_norm": float(traj_user_nn_mean),
            "traj_centroid_gap_norm": float(traj_centroid_mean),
            "traj_switchback_ratio": float(traj_switchback_mean),
            # PhaseG-v1.1+ metadata: lock aggregation/load context to avoid same-name different-mouth confusion.
            "eval_det_paper_cost_agg": "SUM" if bool(use_phaseg_ep_sum) else "MEAN",
            "eval_det_reward_design": str(rd_eval),
            "eval_det_train_load": float(getattr(self, "train_load_nominal", float("nan"))),
            "eval_det_loads_primary": float(eval_loads[0]) if eval_loads else float(getattr(self.env_cfg_eval, "load_scale", 1.0)),
            "eval_det_loads_count": int(len(eval_loads)),
        }
        self.policy.train()
        return out

    def _phaseg_eval_rollouts(
        self,
        *,
        env_cfg_eval: JointEnvConfig,
        eval_loads: List[float],
        eval_seeds: List[int],
        eval_ep_per: int,
    ) -> Dict[str, np.ndarray]:
        """
        PhaseG-v1.1 
        - reward_total_ep: _t reward_total_stepreward
        - paper_cost_ep:  _t paper_cost_step
        - comp/ris bonus: _t reward_comp_step / reward_ris_step
        """
        device = next(self.policy.parameters()).device  # type: ignore

        eval_preprocess = CollectorPreprocess(
            obs_normalizer=self.obs_normalizer,
            update_obs=False,
            update_improve_z=False,
            reward_override=str(getattr(self.preprocess, "reward_override", "env")),
            paper_reward_map=str(getattr(self.preprocess, "paper_reward_map", "tanh")),
            paper_reward_bias=float(getattr(self.preprocess, "paper_reward_bias", 0.0)),
            paper_reward_scale=float(getattr(self.preprocess, "paper_reward_scale", 1.0)),
            paper_reward_tanh_s=float(getattr(self.preprocess, "paper_reward_tanh_s", 0.20)),
            paper_reward_clip_c=float(getattr(self.preprocess, "paper_reward_clip_c", 5.0)),
            improve_mode=str(getattr(self.preprocess, "improve_mode", "paper_improve")),
            rel_improve_eps=float(getattr(self.preprocess, "rel_improve_eps", 1e-3)),
            improve_z_eps=float(getattr(self.preprocess, "improve_z_eps", 1e-6)),
            paper_vio_gating_power=float(getattr(self.preprocess, "paper_vio_gating_power", 1.0)),
            paper_vio_penalty=float(getattr(self.preprocess, "paper_vio_penalty", 0.0)),
            reward_clip_range=float(getattr(self.preprocess, "reward_clip_range", 0.0)),
            vio_scale=float(getattr(self.preprocess, "vio_scale", 1.0)),
            lagrange_enabled=bool(getattr(self.preprocess, "lagrange_enabled", False)),
            lagrange_init=float(getattr(self.preprocess, "lambda_v", 0.0)),
            enable_comp_ris_shaping=bool(getattr(self.preprocess, "enable_comp_ris_shaping", False)),
            comp_weight=float(getattr(self.preprocess, "comp_weight", 0.0)),
            ris_weight=float(getattr(self.preprocess, "ris_weight", 0.0)),
            shaping_ratio=float(getattr(self.preprocess, "shaping_ratio", 0.10)),
        )
        try:
            eval_preprocess.set_lambda_v(float(getattr(self.preprocess, "lambda_v", 0.0)))
        except Exception:
            pass
        try:
            eval_preprocess._impz_n = int(getattr(self.preprocess, "_impz_n", 0))  # type: ignore[attr-defined]
            eval_preprocess._impz_mean = float(getattr(self.preprocess, "_impz_mean", 0.0))  # type: ignore[attr-defined]
            eval_preprocess._impz_m2 = float(getattr(self.preprocess, "_impz_m2", 0.0))  # type: ignore[attr-defined]
        except Exception:
            pass

        ep_reward_total: List[float] = []
        ep_paper_cost: List[float] = []
        ep_comp_bonus: List[float] = []
        ep_ris_bonus: List[float] = []

        self.policy.eval()
        try:
            for load in eval_loads:
                for sd in eval_seeds:
                    
                    env = CompRISEnvJoint(dc_replace(env_cfg_eval), seed=int(sd))
                    env.set_load_scale(float(load))
                    for _ep in range(int(max(eval_ep_per, 1))):
                        obs = env.reset()
                        out0 = eval_preprocess(obs=obs, info={})
                        if "obs" in out0:
                            obs = out0["obs"]
                        info0 = out0.get("info", {})
                        try:
                            obs_raw = np.asarray(info0.get("obs_raw", obs), dtype=np.float32)
                        except Exception:
                            obs_raw = np.asarray(obs, dtype=np.float32)

                        done = False
                        reward_sum = 0.0
                        paper_sum = 0.0
                        comp_sum = 0.0
                        ris_sum = 0.0

                        while not done:
                            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
                            with torch.no_grad():
                                (mu, _sigma), _ = self.policy.actor(obs_t, info={"obs_raw": obs_raw})
                            act = mu.squeeze(0).detach().cpu().numpy().astype(np.float64)

                            obs2, r_env, done, info_env = env.step(act)
                            out1 = eval_preprocess(obs_next=obs2, rew=r_env, info=info_env)

                            r_train = out1.get("rew", r_env)
                            try:
                                reward_sum += float(np.asarray(r_train, dtype=np.float64).reshape(-1)[0])
                            except Exception:
                                reward_sum += float(r_env)

                            info = out1.get("info", info_env) or {}
                            pc_step = float(info.get("paper_cost_step", float("nan")))
                            if not np.isfinite(pc_step):
                                pc_step = float(info.get("policy_paper_cost", info.get("paper_cost", info.get("cost", 0.0))))
                            paper_sum += float(pc_step)
                            comp_sum += float(info.get("reward_comp_step", 0.0))
                            ris_sum += float(info.get("reward_ris_step", 0.0))

                            obs = out1.get("obs_next", obs2)
                            try:
                                obs_raw = np.asarray(info.get("obs_raw", obs2), dtype=np.float32)
                            except Exception:
                                obs_raw = np.asarray(obs2, dtype=np.float32)

                        ep_reward_total.append(float(reward_sum))
                        ep_paper_cost.append(float(paper_sum))
                        ep_comp_bonus.append(float(comp_sum))
                        ep_ris_bonus.append(float(ris_sum))
        finally:
            self.policy.train()

        return {
            "reward_total_ep": np.asarray(ep_reward_total, dtype=np.float64),
            "paper_cost_ep": np.asarray(ep_paper_cost, dtype=np.float64),
            "comp_bonus_ep": np.asarray(ep_comp_bonus, dtype=np.float64),
            "ris_bonus_ep": np.asarray(ep_ris_bonus, dtype=np.float64),
        }

    def run_phaseg_v11_selfcheck(
        self,
        *,
        output_path: Optional[Path] = None,
        corr_threshold: float = 0.20,
        corr_min_episodes: int = 50,
    ) -> Dict[str, Any]:
        """
        PhaseG-v1.1 
        1) reward 
        2) Spearman(reward_total_ep, -paper_cost_ep)seed>=50 episodes
        3) PPO approx_kl/clipfrac/entropy/value_loss
        4)  sanity-test
        5) RIS/CoMP toggle 
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        out_path = Path(output_path) if output_path is not None else (self.run_dir / "phaseg_v11_selfcheck.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _mean_std(x: np.ndarray) -> Tuple[float, float]:
            x0 = np.asarray(x, dtype=np.float64).reshape(-1)
            x0 = x0[np.isfinite(x0)]
            if x0.size <= 0:
                return float("nan"), float("nan")
            return float(np.mean(x0)), float(np.std(x0))

        
        contract_arr = np.asarray(self._phaseg_contract_abs_err_samples, dtype=np.float64).reshape(-1)
        contract_arr = contract_arr[np.isfinite(contract_arr)]
        contract_err_mean = float(np.mean(contract_arr)) if contract_arr.size > 0 else float("nan")
        contract_err_max = float(np.max(contract_arr)) if contract_arr.size > 0 else float("nan")
        contract_n = int(contract_arr.size)
        contract_pass = bool(np.isfinite(contract_err_mean) and np.isfinite(contract_err_max) and contract_err_mean <= 1e-4 and contract_err_max <= 1e-3)

        
        eval_loads = _parse_list(self.eval_cfg.get("eval_loads", None), float)
        if not eval_loads:
            eval_loads = [float(getattr(self.env_cfg_eval, "load_scale", 1.0))]
        corr_loads = [float(eval_loads[0])]
        eval_seeds = _parse_list(self.eval_cfg.get("eval_seeds", None), int)
        if not eval_seeds:
            eval_seeds = list(range(42, 52))
        eval_ep_cfg = int(self.eval_cfg.get("eval_ep_per", self.eval_cfg.get("eval_episodes", 1)))
        min_ep_per = int(math.ceil(float(max(corr_min_episodes, 1)) / float(max(len(corr_loads) * len(eval_seeds), 1))))
        corr_eval_ep_per = int(max(eval_ep_cfg, min_ep_per, 1))

        corr_eval = self._phaseg_eval_rollouts(
            env_cfg_eval=self.env_cfg_eval,
            eval_loads=corr_loads,
            eval_seeds=eval_seeds,
            eval_ep_per=corr_eval_ep_per,
        )
        corr_reward_ep = np.asarray(corr_eval.get("reward_total_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        corr_cost_ep = np.asarray(corr_eval.get("paper_cost_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        corr_n = int(min(corr_reward_ep.size, corr_cost_ep.size))
        corr_s = _spearman_corr(corr_reward_ep[:corr_n], -corr_cost_ep[:corr_n]) if corr_n > 1 else float("nan")
        corr_pass = bool(corr_n >= int(corr_min_episodes) and np.isfinite(corr_s) and float(corr_s) > float(corr_threshold))
        corr_reward_mean, corr_reward_std = _mean_std(corr_reward_ep)
        corr_cost_mean, corr_cost_std = _mean_std(corr_cost_ep)

        
        sens_seed = int(eval_seeds[0]) if eval_seeds else 42
        horizon = int(max(getattr(self.env_cfg_eval, "T", 1), 1))
        act_dim = int(max(getattr(self.policy, "_act_dim", 1), 1))

        
        act_seq_list: List[np.ndarray] = []
        try:
            env_sens = CompRISEnvJoint(dc_replace(self.env_cfg_eval), seed=int(sens_seed))
            obs_sens = env_sens.reset()
            obs_raw_sens = np.asarray(obs_sens, dtype=np.float32)
            done_sens = False
            step_sens = 0
            self.policy.eval()
            try:
                while (not done_sens) and (step_sens < horizon):
                    obs_t = torch.from_numpy(np.asarray(obs_sens, dtype=np.float32)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        (mu_sens, _), _ = self.policy.actor(obs_t, info={"obs_raw": obs_raw_sens})
                    act_sens = mu_sens.squeeze(0).detach().cpu().numpy().astype(np.float64).reshape(-1)
                    act_seq_list.append(np.asarray(act_sens, dtype=np.float64))
                    obs2_sens, _r_sens, done_sens, info_sens = env_sens.step(act_sens)
                    obs_sens = obs2_sens
                    try:
                        obs_raw_sens = np.asarray(info_sens.get("obs_raw", obs2_sens), dtype=np.float32)
                    except Exception:
                        obs_raw_sens = np.asarray(obs2_sens, dtype=np.float32)
                    step_sens += 1
            finally:
                self.policy.train()
        except Exception:
            act_seq_list = []

        if act_seq_list:
            act_seq = np.asarray(act_seq_list, dtype=np.float64).reshape(-1, act_dim)
        else:
            rng = np.random.default_rng(int(sens_seed) + 20260206)
            act_seq = np.asarray(rng.uniform(-0.6, 0.6, size=(horizon, act_dim)), dtype=np.float64)

        def _run_action_seq_cost_sum(actions: np.ndarray, seed_v: int) -> float:
            env = CompRISEnvJoint(dc_replace(self.env_cfg_eval), seed=int(seed_v))
            _ = env.reset()
            paper_sum_local = 0.0
            done_local = False
            t_local = 0
            while (not done_local) and (t_local < int(actions.shape[0])):
                act_local = np.asarray(actions[t_local], dtype=np.float64).reshape(-1)
                _, _r, done_local, info_local = env.step(act_local)
                pc = float(info_local.get("paper_cost_step", float("nan")))
                if not np.isfinite(pc):
                    pc = float(info_local.get("policy_paper_cost", info_local.get("paper_cost", info_local.get("cost", 0.0))))
                paper_sum_local += float(pc)
                t_local += 1
            return float(paper_sum_local)

        sens_delta = float(max(float(self.eval_cfg.get("phaseg_sens_delta", 0.05)), 1e-4))
        sens_scale = float(max(float(self.eval_cfg.get("phaseg_sens_scale", 1.05)), 1.0))
        sens_abs_thr = float(max(float(self.eval_cfg.get("phaseg_sens_diff_abs_min", 5e-2)), 0.0))
        sens_rel_thr = float(max(float(self.eval_cfg.get("phaseg_sens_diff_rel_min", 1e-3)), 0.0))
        sens_probe_limit = int(max(int(self.eval_cfg.get("phaseg_sens_probe_dims", 8)), 1))

        def _perturb_column(base_col: np.ndarray) -> Tuple[np.ndarray, float]:
            col = np.asarray(base_col, dtype=np.float64).reshape(-1)
            pert = np.clip(col * sens_scale, -1.0, 1.0)
            same_mask = np.abs(pert - col) < 1e-10
            if np.any(same_mask):
                direction = np.where(col[same_mask] >= 0.0, -1.0, 1.0)
                pert[same_mask] = np.clip(col[same_mask] + direction * sens_delta, -1.0, 1.0)
            same_mask2 = np.abs(pert - col) < 1e-10
            if np.any(same_mask2):
                direction2 = np.where(col[same_mask2] >= 0.0, -1.0, 1.0)
                pert[same_mask2] = np.clip(col[same_mask2] + direction2 * (2.0 * sens_delta), -1.0, 1.0)
            changed_frac = float(np.mean(np.abs(pert - col) > 1e-10)) if col.size > 0 else 0.0
            return pert, changed_frac

        sens_cost_a = _run_action_seq_cost_sum(act_seq, sens_seed)
        try:
            m_eval = int(max(getattr(self.env_cfg_eval, "M", 1), 1))
        except Exception:
            m_eval = 1
        delta_dim = int(max(min(2 * m_eval, act_dim), 1))
        sens_probe_dims: List[int] = []
        for d in range(delta_dim):
            sens_probe_dims.append(int(d))
            if len(sens_probe_dims) >= sens_probe_limit:
                break
        if len(sens_probe_dims) < sens_probe_limit and act_seq.size > 0:
            col_std = np.asarray(np.std(act_seq, axis=0), dtype=np.float64).reshape(-1)
            rank = np.argsort(-col_std)
            for d in rank.tolist():
                di = int(d)
                if di < 0 or di >= act_dim:
                    continue
                if di in sens_probe_dims:
                    continue
                sens_probe_dims.append(di)
                if len(sens_probe_dims) >= sens_probe_limit:
                    break
        if not sens_probe_dims:
            sens_probe_dims = [0]

        sens_probe_results: List[Dict[str, Any]] = []
        for dim_i in sens_probe_dims:
            act_seq_perturb_i = np.asarray(act_seq, dtype=np.float64).copy()
            if act_seq_perturb_i.size <= 0:
                continue
            try:
                pert_col_i, changed_frac_i = _perturb_column(act_seq_perturb_i[:, int(dim_i)])
                act_seq_perturb_i[:, int(dim_i)] = pert_col_i
            except Exception:
                continue
            sens_cost_b_i = _run_action_seq_cost_sum(act_seq_perturb_i, sens_seed)
            diff_abs_i = float(abs(sens_cost_a - sens_cost_b_i))
            diff_rel_i = float(diff_abs_i / max(abs(sens_cost_a), 1e-9))
            sens_probe_results.append(
                {
                    "dim": int(dim_i),
                    "paper_cost_ep_b": float(sens_cost_b_i),
                    "diff_abs": float(diff_abs_i),
                    "diff_rel": float(diff_rel_i),
                    "changed_frac": float(changed_frac_i),
                }
            )

        if sens_probe_results:
            sens_probe_results.sort(
                key=lambda x: (
                    float(x.get("diff_abs", 0.0)),
                    float(x.get("diff_rel", 0.0)),
                    float(x.get("changed_frac", 0.0)),
                ),
                reverse=True,
            )
            sens_best = sens_probe_results[0]
            sens_best_dim = int(sens_best.get("dim", 0))
            sens_cost_b = float(sens_best.get("paper_cost_ep_b", sens_cost_a))
            sens_diff_abs = float(sens_best.get("diff_abs", 0.0))
            sens_diff_rel = float(sens_best.get("diff_rel", 0.0))
            sens_changed_frac = float(sens_best.get("changed_frac", 0.0))
        else:
            act_seq_perturb = np.asarray(act_seq, dtype=np.float64).copy()
            if act_seq_perturb.size > 0:
                pert_col, sens_changed_frac = _perturb_column(act_seq_perturb[:, 0])
                act_seq_perturb[:, 0] = pert_col
            else:
                sens_changed_frac = 0.0
            sens_best_dim = 0
            sens_cost_b = _run_action_seq_cost_sum(act_seq_perturb, sens_seed)
            sens_diff_abs = float(abs(sens_cost_a - sens_cost_b))
            sens_diff_rel = float(sens_diff_abs / max(abs(sens_cost_a), 1e-9))
            sens_probe_results = [
                {
                    "dim": int(sens_best_dim),
                    "paper_cost_ep_b": float(sens_cost_b),
                    "diff_abs": float(sens_diff_abs),
                    "diff_rel": float(sens_diff_rel),
                    "changed_frac": float(sens_changed_frac),
                }
            ]
            sens_probe_dims = [int(sens_best_dim)]
        sensitivity_pass = bool(
            ((sens_diff_abs > sens_abs_thr) or (sens_diff_rel > sens_rel_thr))
            and (sens_changed_frac > 0.0)
        )
        sens_probe_topk = [
            {
                "dim": int(rec.get("dim", 0)),
                "diff_abs": float(rec.get("diff_abs", 0.0)),
                "diff_rel": float(rec.get("diff_rel", 0.0)),
                "changed_frac": float(rec.get("changed_frac", 0.0)),
            }
            for rec in sens_probe_results[: min(5, len(sens_probe_results))]
        ]

        
        toggle_seeds = list(eval_seeds)
        toggle_ep_per = int(max(int(self.eval_cfg.get("phaseg_toggle_ep_per", 1)), 1))

        
        obs_comp_lock = getattr(self.env_cfg_eval, "obs_enable_comp_flag", None)
        if obs_comp_lock is None:
            obs_comp_lock = 1.0 if bool(getattr(self.env_cfg_eval, "enable_comp", True)) else 0.0
        obs_ris_lock = getattr(self.env_cfg_eval, "obs_enable_ris_flag", None)
        if obs_ris_lock is None:
            obs_ris_lock = 1.0 if bool(getattr(self.env_cfg_eval, "enable_ris", True)) else 0.0

        ris_on_cfg = dc_replace(
            self.env_cfg_eval,
            enable_ris=True,
            obs_enable_comp_flag=float(obs_comp_lock),
            obs_enable_ris_flag=float(obs_ris_lock),
        )
        ris_off_cfg = dc_replace(
            self.env_cfg_eval,
            enable_ris=False,
            obs_enable_comp_flag=float(obs_comp_lock),
            obs_enable_ris_flag=float(obs_ris_lock),
        )
        comp_on_cfg = dc_replace(
            self.env_cfg_eval,
            enable_comp=True,
            obs_enable_comp_flag=float(obs_comp_lock),
            obs_enable_ris_flag=float(obs_ris_lock),
        )
        comp_off_cfg = dc_replace(
            self.env_cfg_eval,
            enable_comp=False,
            obs_enable_comp_flag=float(obs_comp_lock),
            obs_enable_ris_flag=float(obs_ris_lock),
        )

        ris_on_eval = self._phaseg_eval_rollouts(env_cfg_eval=ris_on_cfg, eval_loads=corr_loads, eval_seeds=toggle_seeds, eval_ep_per=toggle_ep_per)
        ris_off_eval = self._phaseg_eval_rollouts(env_cfg_eval=ris_off_cfg, eval_loads=corr_loads, eval_seeds=toggle_seeds, eval_ep_per=toggle_ep_per)
        comp_on_eval = self._phaseg_eval_rollouts(env_cfg_eval=comp_on_cfg, eval_loads=corr_loads, eval_seeds=toggle_seeds, eval_ep_per=toggle_ep_per)
        comp_off_eval = self._phaseg_eval_rollouts(env_cfg_eval=comp_off_cfg, eval_loads=corr_loads, eval_seeds=toggle_seeds, eval_ep_per=toggle_ep_per)

        ris_on_cost = np.asarray(ris_on_eval.get("paper_cost_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        ris_off_cost = np.asarray(ris_off_eval.get("paper_cost_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        ris_on_bonus = np.asarray(ris_on_eval.get("ris_bonus_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)

        comp_on_cost = np.asarray(comp_on_eval.get("paper_cost_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        comp_off_cost = np.asarray(comp_off_eval.get("paper_cost_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        comp_on_bonus = np.asarray(comp_on_eval.get("comp_bonus_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)

        ris_on_cost_mean, _ = _mean_std(ris_on_cost)
        ris_off_cost_mean, _ = _mean_std(ris_off_cost)
        ris_on_bonus_mean, _ = _mean_std(ris_on_bonus)
        ris_cost_diff_abs = float(abs(ris_on_cost_mean - ris_off_cost_mean)) if (np.isfinite(ris_on_cost_mean) and np.isfinite(ris_off_cost_mean)) else float("nan")
        ris_pair_n = int(min(ris_on_cost.size, ris_off_cost.size))
        try:
            ris_beta_max = float(getattr(self.env_cfg_eval, "beta_ris_max", 0.0))
        except Exception:
            ris_beta_max = 0.0
        ris_bonus_check_enabled = bool(abs(float(ris_beta_max)) > 1e-12)
        ris_bonus_pass = bool(
            np.isfinite(ris_on_bonus_mean)
            and (abs(ris_on_bonus_mean) > 1e-6)
        ) if ris_bonus_check_enabled else True
        ris_sign_consistency = float("nan")
        if ris_pair_n > 0:
            ris_diff = np.asarray(ris_off_cost[:ris_pair_n] - ris_on_cost[:ris_pair_n], dtype=np.float64)
            ris_sign_consistency = float(max(np.mean(ris_diff > 0.0), np.mean(ris_diff < 0.0)))
        ris_toggle_pass = bool(
            np.isfinite(ris_cost_diff_abs)
            and (ris_cost_diff_abs > 1.0)
            and np.isfinite(ris_sign_consistency)
            and (ris_sign_consistency >= 0.50)
            and bool(ris_bonus_pass)
        )

        comp_on_cost_mean, _ = _mean_std(comp_on_cost)
        comp_off_cost_mean, _ = _mean_std(comp_off_cost)
        comp_on_bonus_mean, _ = _mean_std(comp_on_bonus)
        comp_cost_diff_abs = float(abs(comp_on_cost_mean - comp_off_cost_mean)) if (np.isfinite(comp_on_cost_mean) and np.isfinite(comp_off_cost_mean)) else float("nan")
        comp_pair_n = int(min(comp_on_cost.size, comp_off_cost.size))
        try:
            comp_beta_max = float(getattr(self.env_cfg_eval, "beta_comp_max", 0.0))
        except Exception:
            comp_beta_max = 0.0
        comp_bonus_check_enabled = bool(abs(float(comp_beta_max)) > 1e-12)
        comp_bonus_pass = bool(
            np.isfinite(comp_on_bonus_mean)
            and (abs(comp_on_bonus_mean) > 1e-6)
        ) if comp_bonus_check_enabled else True
        comp_sign_consistency = float("nan")
        if comp_pair_n > 0:
            comp_diff = np.asarray(comp_off_cost[:comp_pair_n] - comp_on_cost[:comp_pair_n], dtype=np.float64)
            comp_sign_consistency = float(max(np.mean(comp_diff > 0.0), np.mean(comp_diff < 0.0)))
        comp_toggle_pass = bool(
            np.isfinite(comp_cost_diff_abs)
            and (comp_cost_diff_abs > 1.0)
            and np.isfinite(comp_sign_consistency)
            and (comp_sign_consistency >= 0.50)
            and bool(comp_bonus_pass)
        )

        
        ppo_diag = {
            "approx_kl": float(self._last_approx_kl),
            "clipfrac": float(self._last_clipfrac),
            "entropy": float(self._last_entropy),
            "value_loss": float(self._last_value_loss),
        }

        passed = bool(contract_pass and corr_pass and sensitivity_pass and ris_toggle_pass and comp_toggle_pass)
        out: Dict[str, Any] = {
            "timestamp": str(ts),
            "output_path": str(out_path),
            "pass": bool(passed),
            "contract": {
                "n_steps": int(contract_n),
                "abs_err_mean": float(contract_err_mean),
                "abs_err_max": float(contract_err_max),
                "pass": bool(contract_pass),
                "criteria": {"mean_le": 1e-4, "max_le": 1e-3},
            },
            "corr": {
                "method": "spearman",
                "n_episodes": int(corr_n),
                "threshold": float(corr_threshold),
                "min_episodes": int(corr_min_episodes),
                "value": float(corr_s),
                "pass": bool(corr_pass),
                "reward_total_ep_mean": float(corr_reward_mean),
                "reward_total_ep_std": float(corr_reward_std),
                "paper_cost_ep_mean": float(corr_cost_mean),
                "paper_cost_ep_std": float(corr_cost_std),
            },
            "action_sensitivity": {
                "seed": int(sens_seed),
                "horizon": int(horizon),
                "act_dim": int(act_dim),
                "delta": float(sens_delta),
                "scale": float(sens_scale),
                "probe_dims": [int(d) for d in sens_probe_dims],
                "best_probe_dim": int(sens_best_dim),
                "best_probe_changed_frac": float(sens_changed_frac),
                "paper_cost_ep_a": float(sens_cost_a),
                "paper_cost_ep_b": float(sens_cost_b),
                "diff_abs": float(sens_diff_abs),
                "diff_rel": float(sens_diff_rel),
                "probe_topk": sens_probe_topk,
                "pass": bool(sensitivity_pass),
                "criteria": {
                    "diff_abs_gt": float(sens_abs_thr),
                    "or_diff_rel_gt": float(sens_rel_thr),
                    "changed_frac_gt": 0.0,
                },
            },
            "toggle": {
                "seeds": [int(x) for x in toggle_seeds],
                "ep_per_seed": int(toggle_ep_per),
                "loads": [float(x) for x in corr_loads],
                "ris": {
                    "paper_cost_on_mean": float(ris_on_cost_mean),
                    "paper_cost_off_mean": float(ris_off_cost_mean),
                    "paper_cost_diff_abs": float(ris_cost_diff_abs),
                    "direction_consistency": float(ris_sign_consistency),
                    "ris_bonus_on_mean": float(ris_on_bonus_mean),
                    "criteria": {
                        "paper_cost_diff_abs_gt": 1.0,
                        "direction_consistency_ge": 0.50,
                        "bonus_abs_gt": 1e-6,
                        "bonus_check_enabled": bool(ris_bonus_check_enabled),
                    },
                    "pass": bool(ris_toggle_pass),
                },
                "comp": {
                    "paper_cost_on_mean": float(comp_on_cost_mean),
                    "paper_cost_off_mean": float(comp_off_cost_mean),
                    "paper_cost_diff_abs": float(comp_cost_diff_abs),
                    "direction_consistency": float(comp_sign_consistency),
                    "comp_bonus_on_mean": float(comp_on_bonus_mean),
                    "criteria": {
                        "paper_cost_diff_abs_gt": 1.0,
                        "direction_consistency_ge": 0.50,
                        "bonus_abs_gt": 1e-6,
                        "bonus_check_enabled": bool(comp_bonus_check_enabled),
                    },
                    "pass": bool(comp_toggle_pass),
                },
            },
            "ppo_diag": ppo_diag,
        }

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[HB][phaseg-v1.1][WARN] : {type(e).__name__}: {e}", flush=True)

        print(f"[HB][phaseg-v1.1] ts={ts} progress=final out={out_path}", flush=True)
        print(
            f"[HB][phaseg-v1.1] contract mean={contract_err_mean:.3e} max={contract_err_max:.3e} n={contract_n} pass={1 if contract_pass else 0}",
            flush=True,
        )
        print(
            f"[HB][phaseg-v1.1] spearman={corr_s:.4f} n={corr_n} gate>(thr={corr_threshold:.2f},n={corr_min_episodes}) "
            f"reward(m={corr_reward_mean:.4f},s={corr_reward_std:.4f}) paper_cost(m={corr_cost_mean:.4f},s={corr_cost_std:.4f}) pass={1 if corr_pass else 0}",
            flush=True,
        )
        print(
            f"[HB][phaseg-v1.1] action_sensitivity dim={sens_best_dim} probe={len(sens_probe_dims)} "
            f"changed={sens_changed_frac:.3f} diff_abs={sens_diff_abs:.6f} diff_rel={sens_diff_rel:.6f} "
            f"pass={1 if sensitivity_pass else 0}",
            flush=True,
        )
        print(
            f"[HB][phaseg-v1.1] toggle_ris diff={ris_cost_diff_abs:.6f} sign={ris_sign_consistency:.3f} bonus_on={ris_on_bonus_mean:.6f} pass={1 if ris_toggle_pass else 0}",
            flush=True,
        )
        print(
            f"[HB][phaseg-v1.1] toggle_comp diff={comp_cost_diff_abs:.6f} sign={comp_sign_consistency:.3f} bonus_on={comp_on_bonus_mean:.6f} pass={1 if comp_toggle_pass else 0}",
            flush=True,
        )
        print(
            f"[HB][phaseg-v1.1] ppo approx_kl={ppo_diag['approx_kl']:.6f} clipfrac={ppo_diag['clipfrac']:.4f} "
            f"entropy={ppo_diag['entropy']:.4f} value_loss={ppo_diag['value_loss']:.6f} pass={1 if passed else 0}",
            flush=True,
        )
        return out

    def _restore_best_for_stability(self) -> bool:
        """
        PhaseB fixed-eval  best checkpoint 

        
        - fixed-eval /obs_norm
        - PhaseB  best 

        
        - policy / optimizerckpt
        - obs_normalizernormalizer_best.npz
        
        - preprocess.update_obs / update_improve_z
        -  lr  0 update_schedules  _guard_freeze_updates 
        """
        if self.best_ckpt_path is None:
            return False

        device = next(self.policy.parameters()).device  # type: ignore
        ckpt = _torch_load_ckpt(self.best_ckpt_path, map_location=device, prefer_weights_only=False)
        if ckpt is None:
            print(f"[HB][guard][WARN] bestckpt {self.best_ckpt_path}", flush=True)
            return False
        if (not isinstance(ckpt, dict)) or ("policy" not in ckpt):
            return False

        try:
            self.policy.load_state_dict(ckpt["policy"])
            if "optimizer" in ckpt:
                try:
                    _optimizer_load_state(self.optimizer, ckpt["optimizer"])
                except Exception as e:
                    print(f"[HB][guard][WARN] optimizerpolicy: {type(e).__name__}: {e}", flush=True)
        except Exception:
            return False

        
        try:
            if self.obs_normalizer is not None:
                ckpt_dir = self.run_dir / CKPT_DIR_NAME
                npz = ckpt_dir / "normalizer_best.npz"
                if npz.exists():
                    data = np.load(str(npz))
                    stats = {
                        "count": int(data["count"]),
                        "mean": np.asarray(data["mean"], dtype=np.float64),
                        "var": np.asarray(data["var"], dtype=np.float64),
                    }
                    self.obs_normalizer.set_stats(stats)
        except Exception as e:
            print(f"[HB][guard][WARN] obs_normalizer: {type(e).__name__}: {e}", flush=True)

        
        try:
            if hasattr(self.preprocess, "update_obs"):
                setattr(self.preprocess, "update_obs", False)
            if hasattr(self.preprocess, "update_improve_z"):
                setattr(self.preprocess, "update_improve_z", False)
        except Exception as e:
            print(f"[HB][guard][WARN] preprocess: {type(e).__name__}: {e}", flush=True)
        setattr(self, "_guard_freeze_updates", True)
        return True

    def policy_update_fn(self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> None:
        assert self.train_collector is not None
        buf = self.train_collector.buffer

        
        lr_now = float(self._sched_lr) if np.isfinite(self._sched_lr) else float(_optimizer_actor_lr(self.optimizer, default=0.0))
        ent_now = float(self._sched_ent) if np.isfinite(self._sched_ent) else float(getattr(self.policy, "_weight_ent", 0.0))
        std_delta_now = float(self._sched_std_delta) if np.isfinite(self._sched_std_delta) else float(getattr(self.policy.actor, "sigma_scale_delta", 1.0))
        std_score_now = float(self._sched_std_score) if np.isfinite(self._sched_std_score) else float(getattr(self.policy.actor, "sigma_scale_score", getattr(self.policy.actor, "sigma_scale", 1.0)))
        std_now = float(std_score_now)  

        
        batch, indices = buf.sample(0)
        
        
        
        
        train_reward_step_mean = float(np.mean(batch.rew)) if hasattr(batch, "rew") else 0.0
        train_reward_step_std = float(np.std(batch.rew)) if hasattr(batch, "rew") else 0.0

        train_n_ep = 0
        train_len_mean = 0.0
        train_len_std = 0.0
        train_reward_mean = float(train_reward_step_mean)
        train_reward_std = float(train_reward_step_std)
        def _result_finite_value(key: str, default: float) -> float:
            if not isinstance(result, dict):
                return float(default)
            try:
                value = float(result.get(key, default))
            except Exception:
                return float(default)
            if not np.isfinite(value):
                return float(default)
            return float(value)

        train_n_ep = int(max(_result_finite_value("n/ep", 0.0), 0.0))
        if train_n_ep > 0:
            train_reward_mean = float(_result_finite_value("rew", train_reward_step_mean))
            train_reward_std = float(_result_finite_value("rew_std", train_reward_step_std))
            train_len_mean = float(_result_finite_value("len", 0.0))
            train_len_std = float(_result_finite_value("len_std", 0.0))

        
        
        
        
        if train_n_ep > 0 and float(train_len_mean) > 1e-9 and np.isfinite(float(train_reward_mean)):
            train_reward_step_mean = float(train_reward_mean) / float(train_len_mean)

        
        
        
        
        try:
            train_episode_prev = int(float(getattr(self, "train_episode", 0) or 0))
        except Exception:
            train_episode_prev = 0
        train_episode_prev = int(max(train_episode_prev, 0))
        self.train_episode = int(train_episode_prev + max(int(train_n_ep), 0))
        train_episode_now = int(getattr(self, "train_episode", 0) or 0)
        dense_eps_expected_this_rollout = np.asarray([], dtype=np.int64)
        if int(train_episode_now) > int(train_episode_prev):
            dense_eps_expected_this_rollout = np.arange(
                int(train_episode_prev) + 1,
                int(train_episode_now) + 1,
                dtype=np.int64,
            )

        
        
        # - ep_return_train_p50  [150, 450]episode_len80
        
        ep_return_p05 = float("nan")
        ep_return_p50 = float("nan")
        ep_return_p95 = float("nan")
        try:
            if isinstance(result, dict) and ("rews" in result):
                rews_arr = np.asarray(result.get("rews", np.asarray([])), dtype=np.float64).reshape(-1)
                rews_arr = rews_arr[np.isfinite(rews_arr)]
                if rews_arr.size > 0:
                    ep_return_p05 = float(np.percentile(rews_arr, 5))
                    ep_return_p50 = float(np.percentile(rews_arr, 50))
                    ep_return_p95 = float(np.percentile(rews_arr, 95))
        except Exception:
            ep_return_p05 = float("nan")
            ep_return_p50 = float("nan")
            ep_return_p95 = float("nan")

        
        action_oor_frac = 0.0
        action_oor_any_frac = 0.0
        action_sat_frac = 0.0
        action_sat_any_frac = 0.0
        try:
            act_arr = np.asarray(getattr(batch, "act", np.asarray([])), dtype=np.float64)
            if act_arr.ndim == 1:
                act_arr = act_arr.reshape(-1, 1)
            if act_arr.size > 0:
                oor = np.abs(act_arr) > 1.0
                sat = np.abs(act_arr) > 0.98  
                action_oor_frac = float(np.mean(oor))
                action_oor_any_frac = float(np.mean(np.any(oor, axis=1)))
                action_sat_frac = float(np.mean(sat))
                action_sat_any_frac = float(np.mean(np.any(sat, axis=1)))
        except Exception:
            action_oor_frac = 0.0
            action_oor_any_frac = 0.0
            action_sat_frac = 0.0
            action_sat_any_frac = 0.0

        
        actor_scores_mode = str(getattr(self.policy.actor, "scores_mode", "learned")).strip().lower()
        delta_dim_cfg = int(getattr(self.policy.actor, "delta_dim", 0))
        act_dim_effective = int(delta_dim_cfg) if (actor_scores_mode == "fixed" and delta_dim_cfg > 0) else int(getattr(self.policy, "_act_dim", 1))
        scores_fixed_mean = 0.0
        scores_fixed_std = 0.0
        try:
            act_arr2 = np.asarray(getattr(batch, "act", np.asarray([])), dtype=np.float64)
            if act_arr2.ndim == 1:
                act_arr2 = act_arr2.reshape(-1, 1)
            if act_arr2.size > 0 and act_arr2.shape[1] > int(delta_dim_cfg):
                score_seg = act_arr2[:, int(delta_dim_cfg):]
                scores_fixed_mean = float(np.mean(score_seg))
                scores_fixed_std = float(np.std(score_seg))
        except Exception:
            scores_fixed_mean = 0.0
            scores_fixed_std = 0.0

        
        
        
        
        
        info_obj = getattr(batch, "info", None)
        n_batch = int(np.asarray(getattr(batch, "rew", np.asarray([]))).reshape(-1).size)

        def _mean_from_info_batch(keys: Tuple[str, ...], default: float = 0.0) -> float:
            
            if isinstance(info_obj, Batch):
                for k in keys:
                    if hasattr(info_obj, k):
                        try:
                            arr = np.asarray(getattr(info_obj, k), dtype=np.float64).reshape(-1)
                            if arr.size > 0:
                                return float(np.mean(arr))
                        except Exception:
                            continue
                return float(default)

            
            if n_batch <= 0:
                return float(default)
            try:
                infos = np.asarray(info_obj, dtype=object).reshape(-1)
            except Exception:
                return float(default)

            vals: List[float] = []
            for ii in infos:
                for k in keys:
                    try:
                        if isinstance(ii, Batch) and hasattr(ii, k):
                            vals.append(float(getattr(ii, k)))
                            break
                        if isinstance(ii, dict) and (k in ii):
                            vals.append(float(ii.get(k, default)))
                            break
                    except Exception:
                        continue
            if not vals:
                return float(default)
            return float(np.mean(np.asarray(vals, dtype=np.float64)))

        def _arr_from_info_batch(key: str, default: float = 0.0) -> np.ndarray:
            """batch.info"""
            
            if isinstance(info_obj, Batch) and hasattr(info_obj, key):
                try:
                    arr = np.asarray(getattr(info_obj, key), dtype=np.float64).reshape(-1)
                    if arr.size > 0:
                        return arr
                except Exception:
                    pass

            
            if n_batch > 0:
                try:
                    infos = np.asarray(info_obj, dtype=object).reshape(-1)
                except Exception:
                    infos = np.asarray([], dtype=object)
                if infos.size > 0:
                    out = np.full((n_batch,), float(default), dtype=np.float64)
                    n = min(int(infos.size), int(n_batch))
                    for i in range(n):
                        ii = infos[i]
                        try:
                            if isinstance(ii, Batch) and hasattr(ii, key):
                                out[i] = float(getattr(ii, key))
                            elif isinstance(ii, dict) and (key in ii):
                                out[i] = float(ii.get(key, default))
                        except Exception:
                            continue
                    return out
            if n_batch <= 0:
                return np.asarray([], dtype=np.float64)
            return np.full((n_batch,), float(default), dtype=np.float64)

        
        
        
        roll = {}
        try:
            if hasattr(self.preprocess, "get_rollout_stats"):
                roll = self.preprocess.get_rollout_stats() or {}
        except Exception:
            roll = {}

        cost_mean = float(roll.get("paper_cost_mean", _mean_from_info_batch(("paper_cost", "cost"), 0.0)))
        vio_cnt_mean = float(
            roll.get(
                "vio_cnt_mean",
                _mean_from_info_batch(("violation_count", "vio_total_penalty", "penalty_total", "cost_constraint"), 0.0),
            )
        )
        improve_mean = float(roll.get("improve_mean", _mean_from_info_batch(("imp_paper_cost",), 0.0)))
        baseline_balanced_paper_cost_mean = float(
            roll.get("baseline_balanced_paper_cost_mean", _mean_from_info_batch(("baseline_balanced_paper_cost",), 0.0))
        )
        baseline_greedy_delay_paper_cost_mean = float(
            roll.get("baseline_greedy_delay_paper_cost_mean", _mean_from_info_batch(("baseline_greedy_delay_paper_cost",), 0.0))
        )
        baseline_greedy_energy_paper_cost_mean = float(
            roll.get("baseline_greedy_energy_paper_cost_mean", _mean_from_info_batch(("baseline_greedy_energy_paper_cost",), 0.0))
        )
        paper_improve_mean = float(
            roll.get("paper_improve_mean", float(baseline_balanced_paper_cost_mean) - float(cost_mean))
        )
        paper_improve_vs_balanced_mean = float(
            roll.get("paper_improve_vs_balanced_mean", float(baseline_balanced_paper_cost_mean) - float(cost_mean))
        )
        paper_improve_vs_greedy_delay_mean = float(
            roll.get("paper_improve_vs_greedy_delay_mean", float(baseline_greedy_delay_paper_cost_mean) - float(cost_mean))
        )
        paper_improve_vs_greedy_energy_mean = float(
            roll.get("paper_improve_vs_greedy_energy_mean", float(baseline_greedy_energy_paper_cost_mean) - float(cost_mean))
        )

        
        reward_total_step_mean = float(roll.get("reward_total_step_mean", float(train_reward_step_mean))) if roll else float(train_reward_step_mean)
        paper_cost_step_mean = float(roll.get("paper_cost_step_mean", float(cost_mean))) if roll else float(cost_mean)
        reward_total_ep_sum_mean = float(roll.get("reward_total_ep_sum_mean", float("nan"))) if roll else float("nan")
        paper_cost_ep_sum_mean = float(roll.get("paper_cost_ep_sum_mean", float("nan"))) if roll else float("nan")
        ep_sum_n = int(max(float(roll.get("ep_sum_n", 0.0)) if roll else 0.0, 0.0))
        reward_contract_abs_err_mean = float(roll.get("reward_contract_abs_err_mean", float("nan"))) if roll else float("nan")
        reward_contract_abs_err_max = float(roll.get("reward_contract_abs_err_max", float("nan"))) if roll else float("nan")

        if not np.isfinite(reward_total_ep_sum_mean):
            reward_total_ep_sum_mean = float(train_reward_mean)
        if not np.isfinite(paper_cost_ep_sum_mean):
            if np.isfinite(float(train_len_mean)) and float(train_len_mean) > 1e-9:
                paper_cost_ep_sum_mean = float(cost_mean) * float(train_len_mean)
            else:
                paper_cost_ep_sum_mean = float(cost_mean)

        
        dense_episode_len_before = int(len(self.train_episode_dense_history))
        dense_eps_this_rollout = np.asarray([], dtype=np.int64)
        try:
            ep_reward_now = np.asarray(getattr(self.preprocess, "_roll_reward_total_ep_sum", []), dtype=np.float64).reshape(-1)
            ep_cost_now = np.asarray(getattr(self.preprocess, "_roll_paper_cost_ep_sum", []), dtype=np.float64).reshape(-1)
            ep_cost_mean_now = np.asarray(getattr(self.preprocess, "_roll_paper_cost_ep_mean", []), dtype=np.float64).reshape(-1)
            ep_vio_mean_now = np.asarray(getattr(self.preprocess, "_roll_vio_cnt_ep_mean", []), dtype=np.float64).reshape(-1)
            ep_improve_mean_now = np.asarray(getattr(self.preprocess, "_roll_improve_ep_mean", []), dtype=np.float64).reshape(-1)
            ep_shaping_mean_now = np.asarray(getattr(self.preprocess, "_roll_total_shaping_all_ep_mean", []), dtype=np.float64).reshape(-1)
            ep_comp_mean_now = np.asarray(getattr(self.preprocess, "_roll_comp_bonus_ep_mean", []), dtype=np.float64).reshape(-1)
            ep_ris_mean_now = np.asarray(getattr(self.preprocess, "_roll_ris_bonus_ep_mean", []), dtype=np.float64).reshape(-1)

            dense_rewards = np.asarray([], dtype=np.float64)
            dense_cost = np.asarray([], dtype=np.float64)
            dense_vio = np.asarray([], dtype=np.float64)
            dense_improve = np.asarray([], dtype=np.float64)
            dense_shaping = np.asarray([], dtype=np.float64)
            dense_comp = np.asarray([], dtype=np.float64)
            dense_ris = np.asarray([], dtype=np.float64)
            n_pair = int(min(ep_reward_now.size, ep_cost_now.size))
            if n_pair > 0:
                er = ep_reward_now[:n_pair]
                ec = ep_cost_now[:n_pair]
                m_pair = np.isfinite(er) & np.isfinite(ec)
                if np.any(m_pair):
                    erv = np.asarray(er[m_pair], dtype=np.float64).reshape(-1)
                    ecv = np.asarray(ec[m_pair], dtype=np.float64).reshape(-1)
                    self._phaseg_reward_total_ep_samples.extend([float(x) for x in erv.tolist()])
                    self._phaseg_paper_cost_ep_samples.extend([float(x) for x in ecv.tolist()])
                    dense_rewards = erv
                    
                    dense_cost = np.asarray([], dtype=np.float64)

                    def _pick_dense_optional(arr: np.ndarray) -> np.ndarray:
                        if arr.size < n_pair:
                            return np.full((erv.size,), np.nan, dtype=np.float64)
                        out = np.asarray(arr[:n_pair], dtype=np.float64).reshape(-1)[m_pair]
                        if out.size != erv.size:
                            return np.full((erv.size,), np.nan, dtype=np.float64)
                        return np.asarray(np.where(np.isfinite(out), out, np.nan), dtype=np.float64).reshape(-1)

                    dense_vio = _pick_dense_optional(ep_vio_mean_now)
                    dense_improve = _pick_dense_optional(ep_improve_mean_now)
                    dense_shaping = _pick_dense_optional(ep_shaping_mean_now)
                    dense_comp = _pick_dense_optional(ep_comp_mean_now)
                    dense_ris = _pick_dense_optional(ep_ris_mean_now)
                    dense_cost_mean = _pick_dense_optional(ep_cost_mean_now)
                    if dense_cost_mean.size == erv.size and np.isfinite(dense_cost_mean).any():
                        dense_cost = dense_cost_mean
                    else:
                        
                        dense_cost = np.asarray(np.where(np.isfinite(ecv), ecv, np.nan), dtype=np.float64).reshape(-1)
            if dense_rewards.size <= 0 and isinstance(result, dict):
                try:
                    rews_now = np.asarray(result.get("rews", []), dtype=np.float64).reshape(-1)
                    rews_now = rews_now[np.isfinite(rews_now)]
                    if rews_now.size > 0:
                        n_take = int(min(rews_now.size, max(int(train_n_ep), 0)))
                        if n_take > 0:
                            dense_rewards = np.asarray(rews_now[:n_take], dtype=np.float64).reshape(-1)
                            dense_cost = np.asarray([], dtype=np.float64)
                            dense_vio = np.asarray([], dtype=np.float64)
                            dense_improve = np.asarray([], dtype=np.float64)
                            dense_shaping = np.asarray([], dtype=np.float64)
                            dense_comp = np.asarray([], dtype=np.float64)
                            dense_ris = np.asarray([], dtype=np.float64)
                except Exception:
                    pass
            if dense_rewards.size > 0:
                ep_end = int(train_episode_now)
                ep_start = int(ep_end - int(dense_rewards.size) + 1)
                if ep_start < 1:
                    drop = int(1 - ep_start)
                    old_n = int(dense_rewards.size)
                    if drop < old_n:
                        dense_rewards = dense_rewards[drop:]
                        if dense_cost.size == old_n:
                            dense_cost = dense_cost[drop:]
                        if dense_vio.size == old_n:
                            dense_vio = dense_vio[drop:]
                        if dense_improve.size == old_n:
                            dense_improve = dense_improve[drop:]
                        if dense_shaping.size == old_n:
                            dense_shaping = dense_shaping[drop:]
                        if dense_comp.size == old_n:
                            dense_comp = dense_comp[drop:]
                        if dense_ris.size == old_n:
                            dense_ris = dense_ris[drop:]
                        ep_start = 1
                    else:
                        dense_rewards = np.asarray([], dtype=np.float64)
                        dense_cost = np.asarray([], dtype=np.float64)
                        dense_vio = np.asarray([], dtype=np.float64)
                        dense_improve = np.asarray([], dtype=np.float64)
                        dense_shaping = np.asarray([], dtype=np.float64)
                        dense_comp = np.asarray([], dtype=np.float64)
                        dense_ris = np.asarray([], dtype=np.float64)
                if dense_rewards.size > 0:
                    dense_eps = np.arange(int(ep_start), int(ep_start + dense_rewards.size), dtype=np.int64)
                    if self.train_episode_dense_history:
                        last_ep = int(self.train_episode_dense_history[-1])
                        keep_dense = dense_eps > last_ep
                        old_n = int(dense_eps.size)
                        dense_eps = dense_eps[keep_dense]
                        dense_rewards = dense_rewards[keep_dense]
                        if dense_cost.size == old_n:
                            dense_cost = dense_cost[keep_dense]
                        if dense_vio.size == old_n:
                            dense_vio = dense_vio[keep_dense]
                        if dense_improve.size == old_n:
                            dense_improve = dense_improve[keep_dense]
                        if dense_shaping.size == old_n:
                            dense_shaping = dense_shaping[keep_dense]
                        if dense_comp.size == old_n:
                            dense_comp = dense_comp[keep_dense]
                        if dense_ris.size == old_n:
                            dense_ris = dense_ris[keep_dense]
                    if dense_eps.size > 0 and dense_rewards.size == dense_eps.size:
                        self.train_episode_dense_history.extend([int(v) for v in dense_eps.tolist()])
                        dense_eps_this_rollout = np.asarray(dense_eps, dtype=np.int64).reshape(-1)
                        self.reward_total_ep_dense_history.extend([float(v) for v in dense_rewards.tolist()])
                        if dense_cost.size == dense_eps.size:
                            self.cost_ep_dense_history.extend([float(v) for v in dense_cost.tolist()])
                        if dense_vio.size == dense_eps.size:
                            self.vio_ep_dense_history.extend([float(v) for v in dense_vio.tolist()])
                        if dense_improve.size == dense_eps.size:
                            self.improve_ep_dense_history.extend([float(v) for v in dense_improve.tolist()])
                        if dense_shaping.size == dense_eps.size:
                            self.shaping_ep_dense_history.extend([float(v) for v in dense_shaping.tolist()])
                        if dense_comp.size == dense_eps.size:
                            self.comp_ep_dense_history.extend([float(v) for v in dense_comp.tolist()])
                        if dense_ris.size == dense_eps.size:
                            self.ris_ep_dense_history.extend([float(v) for v in dense_ris.tolist()])
                        
                        _cur_load = float(getattr(self, "_curriculum_load_last", float("nan")))
                        if not np.isfinite(_cur_load):
                            _cur_load = float(getattr(self, "train_load_nominal", 0.8))
                        self.load_ep_dense_history.extend([_cur_load] * int(dense_eps.size))
            contract_now = np.asarray(getattr(self.preprocess, "_roll_contract_residual", []), dtype=np.float64).reshape(-1)
            if contract_now.size > 0:
                contract_now = contract_now[np.isfinite(contract_now)]
                if contract_now.size > 0:
                    self._phaseg_contract_abs_err_samples.extend([float(x) for x in contract_now.tolist()])

            
            keep_n = 50000
            if len(self._phaseg_reward_total_ep_samples) > keep_n:
                self._phaseg_reward_total_ep_samples = self._phaseg_reward_total_ep_samples[-keep_n:]
            if len(self._phaseg_paper_cost_ep_samples) > keep_n:
                self._phaseg_paper_cost_ep_samples = self._phaseg_paper_cost_ep_samples[-keep_n:]
            if len(self._phaseg_contract_abs_err_samples) > keep_n:
                self._phaseg_contract_abs_err_samples = self._phaseg_contract_abs_err_samples[-keep_n:]
        except Exception:
            pass
        if dense_eps_expected_this_rollout.size > 0:
            target_ep_start = int(dense_eps_expected_this_rollout[0])
            target_ep_end = int(dense_eps_expected_this_rollout[-1])
            last_dense_ep = int(self.train_episode_dense_history[-1]) if self.train_episode_dense_history else 0
            fill_ep_start = int(max(last_dense_ep + 1, target_ep_start))

            if fill_ep_start <= target_ep_end:
                fill_eps = np.arange(fill_ep_start, target_ep_end + 1, dtype=np.int64)
                n_fill = int(fill_eps.size)
                if n_fill > 0:
                    def _finite_or(v, default_v: float = 0.0) -> float:
                        try:
                            fv = float(v)
                            if np.isfinite(fv):
                                return fv
                        except Exception:
                            pass
                        return float(default_v)

                    fill_reward = _finite_or(reward_total_ep_sum_mean, _finite_or(train_reward_mean, 0.0))
                    fill_cost = _finite_or(roll.get("paper_cost_mean", cost_mean), _finite_or(cost_mean, 0.0))
                    fill_vio = _finite_or(roll.get("vio_cnt_mean", vio_cnt_mean), _finite_or(vio_cnt_mean, 0.0))
                    fill_improve = _finite_or(roll.get("improve_mean", improve_mean), _finite_or(improve_mean, 0.0))
                    fill_shaping = _finite_or(roll.get("total_shaping_all_mean", 0.0), 0.0)
                    fill_comp = _finite_or(roll.get("comp_bonus_mean", 0.0), 0.0)
                    fill_ris = _finite_or(roll.get("ris_bonus_mean", 0.0), 0.0)

                    self.train_episode_dense_history.extend([int(v) for v in fill_eps.tolist()])
                    self.reward_total_ep_dense_history.extend([float(fill_reward)] * n_fill)
                    self.cost_ep_dense_history.extend([float(fill_cost)] * n_fill)
                    self.vio_ep_dense_history.extend([float(fill_vio)] * n_fill)
                    self.improve_ep_dense_history.extend([float(fill_improve)] * n_fill)
                    self.shaping_ep_dense_history.extend([float(fill_shaping)] * n_fill)
                    self.comp_ep_dense_history.extend([float(fill_comp)] * n_fill)
                    self.ris_ep_dense_history.extend([float(fill_ris)] * n_fill)
                    
                    _fill_load = float(getattr(self, "_curriculum_load_last", float("nan")))
                    if not np.isfinite(_fill_load):
                        _fill_load = float(getattr(self, "train_load_nominal", 0.8))
                    self.load_ep_dense_history.extend([float(_fill_load)] * int(n_fill))

        dense_episode_len_after = int(len(self.train_episode_dense_history))
        if dense_episode_len_after > dense_episode_len_before:
            dense_eps_this_rollout = np.asarray(
                self.train_episode_dense_history[dense_episode_len_before:dense_episode_len_after],
                dtype=np.int64,
            ).reshape(-1)
        else:
            dense_eps_this_rollout = np.asarray([], dtype=np.int64)
        
        
        
        
        delta_scale_mean = _mean_from_info_batch(("delta_scale",), 0.0)
        vio_wall_arr = np.maximum(_arr_from_info_batch("vio_wall", 0.0), 0.0)
        vio_wall_mean = float(np.mean(vio_wall_arr)) if vio_wall_arr.size > 0 else 0.0
        vio_wall_any_frac = float(np.mean(vio_wall_arr > 0.0)) if vio_wall_arr.size > 0 else 0.0
        

        
        env_reward_mean = _mean_from_info_batch(("reward_total",), float(train_reward_step_mean))

        
        
        if "coverage_fraction_mean" in roll:
            coverage_fraction_arr = np.asarray([], dtype=np.float64)
            coverage_margin_arr = np.asarray([], dtype=np.float64)
            collision_margin_arr = np.asarray([], dtype=np.float64)
            coverage_fraction_mean = float(roll.get("coverage_fraction_mean", 0.0))
            coverage_margin_mean = float(roll.get("coverage_margin_mean", 0.0))
            collision_margin_mean = float(roll.get("collision_margin_mean", 0.0))
            vio_metric_mean = float(roll.get("vio_metric_mean", 0.0))
            vio_metric_arr = np.asarray([vio_metric_mean], dtype=np.float64)
        else:
            coverage_fraction_arr = np.clip(_arr_from_info_batch("coverage_fraction", 0.0), 0.0, 1.0)
            coverage_margin_arr = np.maximum(_arr_from_info_batch("coverage_margin", 0.0), 0.0)
            collision_margin_arr = np.maximum(_arr_from_info_batch("collision_margin", 0.0), 0.0)
            vio_metric_arr = np.clip(
                (1.0 - coverage_fraction_arr) + 0.5 * coverage_margin_arr + 0.5 * collision_margin_arr,
                0.0,
                1.0,
            )
            coverage_fraction_mean = float(np.mean(coverage_fraction_arr)) if coverage_fraction_arr.size > 0 else 0.0
            coverage_margin_mean = float(np.mean(coverage_margin_arr)) if coverage_margin_arr.size > 0 else 0.0
            collision_margin_mean = float(np.mean(collision_margin_arr)) if collision_margin_arr.size > 0 else 0.0
            vio_metric_mean = float(np.mean(vio_metric_arr)) if vio_metric_arr.size > 0 else 0.0

        # -----------------------------
        
        
        
        
        
        
        # -----------------------------
        
        violation_any_frac = float(roll.get("vio_any_frac", 0.0)) if roll else 0.0
        reward_raw_mean = float(roll.get("reward_raw_mean", 0.0)) if roll else 0.0
        reward_raw_std = float(roll.get("reward_raw_std", 0.0)) if roll else 0.0
        reward_raw_p05 = float(roll.get("reward_raw_p05", 0.0)) if roll else 0.0
        reward_raw_p50 = float(roll.get("reward_raw_p50", 0.0)) if roll else 0.0
        reward_raw_p95 = float(roll.get("reward_raw_p95", 0.0)) if roll else 0.0
        clip_hit_frac = float(roll.get("clip_hit_frac", 0.0)) if roll else 0.0

        # -----------------------------
        
        
        
        # -----------------------------
        comp_gain_ratio_mean = float(roll.get("comp_gain_ratio_mean", _mean_from_info_batch(("comp_gain_ratio",), 0.0))) if roll else 0.0
        ris_gain_ratio_mean = float(roll.get("ris_gain_ratio_mean", _mean_from_info_batch(("ris_gain_ratio",), 0.0))) if roll else 0.0
        env_shaping_mean = float(
            roll.get("env_shaping_mean", _mean_from_info_batch(("reward_shaping_total", "reward_potential"), 0.0))
        ) if roll else 0.0
        total_shaping_mean = float(roll.get("total_shaping_mean", 0.0)) if roll else 0.0
        total_shaping_all_mean = float(
            roll.get("total_shaping_all_mean", float(env_shaping_mean))
        ) if roll else float(env_shaping_mean)
        comp_bonus_mean = float(roll.get("comp_bonus_mean", 0.0)) if roll else 0.0
        ris_bonus_mean = float(roll.get("ris_bonus_mean", 0.0)) if roll else 0.0
        comp_ris_ratio = 0.0
        try:
            if abs(float(total_shaping_mean)) > 1e-12:
                comp_ris_ratio = float((float(comp_bonus_mean) + float(ris_bonus_mean)) / max(float(total_shaping_mean), 1e-12))
        except Exception:
            comp_ris_ratio = 0.0

        # -----------------------------
        
        # -----------------------------
        comp_enable_rate = float(roll.get("comp_enable_rate_mean", float("nan"))) if roll else float("nan")
        service_switch_count = float(roll.get("service_switch_count_mean", float("nan"))) if roll else float("nan")
        theta_entropy = float(roll.get("theta_entropy_mean", float("nan"))) if roll else float("nan")
        comp_score_temp_runtime = float(roll.get("comp_score_temp_runtime_mean", float("nan"))) if roll else float("nan")
        z4_comp_meta_enable_mean = float(roll.get("z4_comp_meta_enable_mean", _mean_from_info_batch(("z4_comp_meta_enable",), 0.0))) if roll else 0.0
        z4_comp_meta_warm_scale_mean = float(roll.get("z4_comp_meta_warm_scale_mean", _mean_from_info_batch(("z4_comp_meta_warm_scale",), 0.0))) if roll else 0.0
        z4_comp_meta_thr_effective_mean = float(roll.get("z4_comp_meta_thr_effective_mean", _mean_from_info_batch(("z4_comp_meta_thr_effective",), 0.0))) if roll else 0.0
        z4_comp_meta_temp_effective_mean = float(roll.get("z4_comp_meta_temp_effective_mean", _mean_from_info_batch(("z4_comp_meta_temp_effective",), 0.0))) if roll else 0.0
        z4_comp_meta_ctrl_thr_raw_mean = float(roll.get("z4_comp_meta_ctrl_thr_raw_mean", _mean_from_info_batch(("z4_comp_meta_ctrl_thr_raw",), 0.0))) if roll else 0.0
        z4_comp_meta_ctrl_temp_raw_mean = float(roll.get("z4_comp_meta_ctrl_temp_raw_mean", _mean_from_info_batch(("z4_comp_meta_ctrl_temp_raw",), 0.0))) if roll else 0.0
        z4_comp_meta_thr_delta_ema_mean = float(roll.get("z4_comp_meta_thr_delta_ema_mean", _mean_from_info_batch(("z4_comp_meta_thr_delta_ema",), 0.0))) if roll else 0.0
        z4_comp_meta_temp_scale_ema_mean = float(roll.get("z4_comp_meta_temp_scale_ema_mean", _mean_from_info_batch(("z4_comp_meta_temp_scale_ema",), 1.0))) if roll else 1.0
        z4_assoc_stage_enable_mean = float(_mean_from_info_batch(("z4_assoc_stage_enable",), 0.0))
        z4_assoc_policy_mix_mean = float(_mean_from_info_batch(("z4_assoc_policy_mix",), 1.0))
        z4_assoc_train_frac_mean = float(_mean_from_info_batch(("z4_assoc_train_frac",), 0.0))
        z4_assoc_stage_score_width_mean = float(_mean_from_info_batch(("z4_assoc_stage_score_width",), 0.0))
        z4_assoc_stage_comp_rule_thr_mean = float(_mean_from_info_batch(("z4_assoc_stage_comp_rule_thr",), 0.0))
        theta_mode_effective = str(roll.get("theta_mode_effective", "")) if roll else ""
        theta_mode_effective_agent_frac = float(roll.get("theta_mode_effective_agent_frac", float("nan"))) if roll else float("nan")
        if not bool(getattr(self, "log_theta_comp_coupling", False)):
            comp_enable_rate = float("nan")
            service_switch_count = float("nan")
            theta_entropy = float("nan")
            comp_score_temp_runtime = float("nan")
            z4_comp_meta_enable_mean = float("nan")
            z4_comp_meta_warm_scale_mean = float("nan")
            z4_comp_meta_thr_effective_mean = float("nan")
            z4_comp_meta_temp_effective_mean = float("nan")
            z4_comp_meta_ctrl_thr_raw_mean = float("nan")
            z4_comp_meta_ctrl_temp_raw_mean = float("nan")
            z4_comp_meta_thr_delta_ema_mean = float("nan")
            z4_comp_meta_temp_scale_ema_mean = float("nan")
            z4_assoc_stage_enable_mean = float("nan")
            z4_assoc_policy_mix_mean = float("nan")
            z4_assoc_train_frac_mean = float("nan")
            z4_assoc_stage_score_width_mean = float("nan")
            z4_assoc_stage_comp_rule_thr_mean = float("nan")
            theta_mode_effective = ""
            theta_mode_effective_agent_frac = float("nan")

        # -----------------------------
        
        
        # -----------------------------
        reward_paper_step_mean = float(roll.get("reward_paper_step_mean", 0.0)) if roll else 0.0
        reward_paper_abs_step_mean = float(roll.get("reward_paper_abs_step_mean", 0.0)) if roll else 0.0
        reward_paper_delta_step_mean = float(roll.get("reward_paper_delta_step_mean", 0.0)) if roll else 0.0
        reward_paper_adv_step_mean = float(roll.get("reward_paper_adv_step_mean", 0.0)) if roll else 0.0
        paper_delta_norm_step_mean = float(roll.get("paper_delta_norm_step_mean", 0.0)) if roll else 0.0
        paper_adv_norm_step_mean = float(roll.get("paper_adv_norm_step_mean", 0.0)) if roll else 0.0
        constraint_signal_step_mean = float(roll.get("constraint_signal_step_mean", 0.0)) if roll else 0.0
        lambda_v_effective_mean = float(roll.get("lambda_v_effective_mean", 0.0)) if roll else 0.0
        reward_constraint_step_mean = float(roll.get("reward_constraint_step_mean", 0.0)) if roll else 0.0
        reward_misc_step_mean = float(roll.get("reward_misc_step_mean", 0.0)) if roll else 0.0
        reward_proximity_step_mean = float(roll.get("reward_proximity_step_mean", 0.0)) if roll else 0.0
        shaping_gate_mean = float(roll.get("shaping_gate_mean", 0.0)) if roll else 0.0
        beta_comp_t_mean = float(roll.get("beta_comp_t_mean", 0.0)) if roll else 0.0
        beta_ris_t_mean = float(roll.get("beta_ris_t_mean", 0.0)) if roll else 0.0
        beta_train_frac_mean = float(roll.get("beta_train_frac_mean", 0.0)) if roll else 0.0
        z4_gap_raw_step_mean = float(roll.get("z4_gap_raw_step_mean", 0.0)) if roll else 0.0
        z4_gap_step_clip_mean = float(roll.get("z4_gap_step_clip_mean", 0.0)) if roll else 0.0
        z4_gap_step_ema_mean = float(roll.get("z4_gap_step_ema_mean", 0.0)) if roll else 0.0
        z4_gap_step_mean = float(roll.get("z4_gap_step_mean", 0.0)) if roll else 0.0
        z4_gap_reward_step_mean = float(roll.get("z4_gap_reward_step_mean", 0.0)) if roll else 0.0
        z4_gap_lambda_input_mean = float(roll.get("z4_gap_lambda_input_mean", 0.0)) if roll else 0.0
        z4_gap_lambda_sched_mean = float(roll.get("z4_gap_lambda_sched_mean", 0.0)) if roll else 0.0
        z4_gap_lambda_cap_mean = float(roll.get("z4_gap_lambda_cap_mean", 0.0)) if roll else 0.0
        z4_gap_lambda_eff_mean = float(roll.get("z4_gap_lambda_eff_mean", 0.0)) if roll else 0.0
        z4_gap_grad_ratio_eff_mean = float(roll.get("z4_gap_grad_ratio_eff_mean", 0.0)) if roll else 0.0
        z4_gap_shape_guard_trigger_mean = float(roll.get("z4_gap_shape_guard_trigger_mean", 0.0)) if roll else 0.0
        t6_main_weight_mean = float(roll.get("t6_main_weight_mean", 0.0)) if roll else 0.0
        t6_potential_scale_mean = float(roll.get("t6_potential_scale_mean", 0.0)) if roll else 0.0
        t6_traj_scale_mean = float(roll.get("t6_traj_scale_mean", 0.0)) if roll else 0.0
        t6_anneal_ratio_mean = float(roll.get("t6_anneal_ratio_mean", 0.0)) if roll else 0.0
        t6_violation_ema_mean = float(roll.get("t6_violation_ema_mean", 0.0)) if roll else 0.0
        t6_pid_frozen_mean = float(roll.get("t6_pid_frozen_mean", 0.0)) if roll else 0.0
        comp_bonus_raw_mean = float(roll.get("comp_bonus_raw_mean", 0.0)) if roll else 0.0
        ris_bonus_raw_mean = float(roll.get("ris_bonus_raw_mean", 0.0)) if roll else 0.0

        reward_shaping_step_mean = float(total_shaping_all_mean)
        
        reward_paper_ep = float(roll.get("reward_paper_ep_sum_mean", float("nan"))) if roll else float("nan")
        reward_shaping_ep = float(roll.get("reward_shaping_ep_sum_mean", float("nan"))) if roll else float("nan")
        reward_constraint_ep = float(roll.get("reward_constraint_ep_sum_mean", float("nan"))) if roll else float("nan")
        reward_misc_ep = float(roll.get("reward_misc_ep_sum_mean", float("nan"))) if roll else float("nan")
        if float(train_len_mean) > 1e-9 and np.isfinite(float(train_len_mean)):
            if not np.isfinite(reward_paper_ep):
                reward_paper_ep = float(reward_paper_step_mean) * float(train_len_mean)
            if not np.isfinite(reward_shaping_ep):
                reward_shaping_ep = float(reward_shaping_step_mean) * float(train_len_mean)
            if not np.isfinite(reward_constraint_ep):
                reward_constraint_ep = float(reward_constraint_step_mean) * float(train_len_mean)
            if not np.isfinite(reward_misc_ep):
                reward_misc_ep = float(reward_misc_step_mean) * float(train_len_mean)
        else:
            reward_paper_ep = 0.0 if not np.isfinite(reward_paper_ep) else float(reward_paper_ep)
            reward_shaping_ep = 0.0 if not np.isfinite(reward_shaping_ep) else float(reward_shaping_ep)
            reward_constraint_ep = 0.0 if not np.isfinite(reward_constraint_ep) else float(reward_constraint_ep)
            reward_misc_ep = 0.0 if not np.isfinite(reward_misc_ep) else float(reward_misc_ep)

        
        explained_variance = 0.0
        returns_mean = float("nan")
        returns_std = float("nan")
        value_bias_proxy = float("nan")
        b2 = None
        try:
            b2 = self.policy.process_fn(batch, buf, indices)
            y_true = b2.returns.detach().cpu().numpy().astype(np.float64)
            y_pred = b2.v_s.detach().cpu().numpy().astype(np.float64)
            var_y = float(np.var(y_true))
            returns_mean = float(np.mean(y_true)) if y_true.size > 0 else float("nan")
            returns_std = float(np.std(y_true)) if y_true.size > 0 else float("nan")
            value_bias_proxy = float(np.mean(y_pred) - np.mean(y_true)) if (y_true.size > 0 and y_pred.size > 0) else float("nan")
            if var_y > 1e-12:
                explained_variance = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-12))
        except Exception:
            explained_variance = 0.0
            returns_mean = float("nan")
            returns_std = float("nan")
            value_bias_proxy = float("nan")
        if not bool(getattr(self, "log_returns_stats", False)):
            returns_mean = float("nan")
            returns_std = float("nan")
            value_bias_proxy = float("nan")

        
        lag_cfg = self.train_params.get("lagrange_vio", {}) or {}
        if bool(lag_cfg.get("enabled", False)):
            
            load_now = float(getattr(self, "_curriculum_load_last", getattr(self, "train_load_nominal", 1.0)))
            if not np.isfinite(load_now):
                load_now = float(getattr(self, "train_load_nominal", 1.0))
            bucket_id, target, eta, lam_min, lam_max = self._resolve_lagrange_bucket(float(load_now), lag_cfg)
            
            lam_new = float(self.preprocess.lambda_v + float(eta) * (float(vio_metric_mean) - float(target)))
            lam_new = float(np.clip(lam_new, float(lam_min), float(lam_max)))
            self.preprocess.set_lambda_v(lam_new)

            self._lag_bucket_last_id = int(bucket_id)
            self._lag_target_current = float(target)
            self._lag_eta_current = float(eta)
            self._lag_lambda_min_current = float(lam_min)
            self._lag_lambda_max_current = float(lam_max)
            self._lag_bucket_last_load = float(load_now)
            if bool(getattr(self, "_lag_bucket_enable", False)):
                prev_id = int(getattr(self, "_lag_bucket_last_id_prev", -1))
                if prev_id != int(bucket_id):
                    print(
                        f"[HB][train][LAG] load={float(load_now):.2f} bucket={int(bucket_id)} "
                        f"target={float(target):.3f} eta={float(eta):.4f} "
                        f"range=[{float(lam_min):.3f},{float(lam_max):.3f}] "
                        f"lambda={float(lam_new):.3f}",
                        flush=True,
                    )
                self._lag_bucket_last_id_prev = int(bucket_id)

        
        losses = self.policy.update(0, buf, batch_size=int(self.batch_size), repeat=int(self.repeat_per_collect))
        
        bc_loss_raw, bc_loss_weighted, bc_updates = self._run_bc_warmstart_update()
        self.train_collector.reset_buffer(keep_statistics=True)

        
        def _loss_scalar_mean(name: str) -> float:
            vv = losses.get(name, None)
            if isinstance(vv, list):
                arr = np.asarray(vv, dtype=np.float64).reshape(-1)
                arr = arr[np.isfinite(arr)]
                return float(np.mean(arr)) if arr.size > 0 else float("nan")
            try:
                fv = float(vv)
                return float(fv) if np.isfinite(fv) else float("nan")
            except Exception:
                return float("nan")

        a1_head_clip_used_ratio = _loss_scalar_mean("a1/head_clip_used_ratio")
        a1_head_clip_ready = _loss_scalar_mean("a1/head_clip_ready")
        a1_head_ent_used_ratio = _loss_scalar_mean("a1/head_ent_used_ratio")

        
        step = max([1] + [len(v) for v in losses.values() if isinstance(v, list)])
        self.gradient_step += int(step)
        self.log_update_data(data, losses)

        
        policy_loss = float(np.mean(losses.get("loss/clip", [0.0]))) if isinstance(losses.get("loss/clip", None), list) else float(losses.get("loss/clip", 0.0))
        value_loss = float(np.mean(losses.get("loss/vf", [0.0]))) if isinstance(losses.get("loss/vf", None), list) else float(losses.get("loss/vf", 0.0))
        vf_clip_hit_frac = (
            float(np.mean(losses.get("loss/vf_clip_hit_frac", [0.0])))
            if isinstance(losses.get("loss/vf_clip_hit_frac", None), list)
            else float(losses.get("loss/vf_clip_hit_frac", 0.0))
        )

        
        act_dim_now = int(getattr(self.policy, "_act_dim", 1))
        logp_scale = float(getattr(self.policy, "_logp_scale", 1.0))
        logp_mode = str(getattr(self.policy, "_logp_mode", "sum")).strip().lower()
        logp_is_mean = 1.0 if logp_mode == "mean" else 0.0
        logp_act_dim = int(getattr(self.policy, "_logp_act_dim", act_dim_effective))
        logp_act_dim = int(max(1, min(int(logp_act_dim), int(act_dim_now))))

        entropy_sum = 0.0
        entropy_per_dim = 0.0
        kl_sum = 0.0
        kl_per_dim = 0.0
        clipfrac = 0.0
        clipfrac_per_dim = 0.0

        try:
            if b2 is not None:
                with torch.no_grad():
                    
                    dist_new = self.policy(b2).dist
                    logp_new = dist_new.log_prob(b2.act)
                    logp_old = b2.logp_old

                    
                    ratio = (logp_new - logp_old).exp()
                    eps_clip = float(getattr(self.policy, "_eps_clip", 0.2))
                    clipfrac = float(((ratio - 1.0).abs() > eps_clip).float().mean().cpu().item())

                    
                    kl_used = float((logp_old - logp_new).mean().cpu().item())
                    if not np.isfinite(kl_used):
                        kl_used = 0.0
                    kl_used = float(max(kl_used, 0.0))

                    
                    ent = dist_new.entropy()
                    if ent.ndim > 1:
                        ent = ent.sum(dim=-1)
                    entropy_sum = float(ent.mean().cpu().item())
                    if not np.isfinite(entropy_sum):
                        entropy_sum = 0.0

                    act_dim_full = int(max(1, int(getattr(b2.act, "shape", [act_dim_now])[-1])))
                    
                    
                    act_dim_used = int(max(1, min(int(logp_act_dim), int(act_dim_full))))

                    if logp_mode == "mean":
                        
                        entropy_per_dim = float(entropy_sum)
                        entropy_sum = float(entropy_per_dim * float(act_dim_used))
                    else:
                        entropy_sum = float(entropy_sum)
                        entropy_per_dim = float(entropy_sum / float(act_dim_used))

                    if logp_mode == "mean":
                        
                        kl_per_dim = float(kl_used)
                        kl_sum = float(kl_per_dim * float(act_dim_used))
                        clipfrac_per_dim = float(clipfrac)
                    else:
                        
                        kl_sum = float(kl_used)
                        kl_per_dim = float(kl_sum / float(max(act_dim_used, 1)))
                        
                        ratio_pd = ((logp_new - logp_old) / float(max(act_dim_used, 1))).exp()
                        clipfrac_per_dim = float(((ratio_pd - 1.0).abs() > eps_clip).float().mean().cpu().item())

                    act_dim_now = int(act_dim_full)
        except Exception:
            entropy_sum = 0.0
            entropy_per_dim = 0.0
            kl_sum = 0.0
            kl_per_dim = 0.0
            clipfrac = 0.0
            clipfrac_per_dim = 0.0

        
        entropy = float(entropy_sum)
        kl = float(kl_sum)

        
        log_std_mean = 0.0
        sigma_mean = 0.0
        sigma_delta_mean = 0.0
        sigma_score_mean = 0.0
        try:
            if hasattr(self.policy.actor, "sigma") and isinstance(self.policy.actor.sigma, torch.nn.Parameter):
                log_std_mean = float(self.policy.actor.sigma.detach().mean().cpu().item())
                sigma_mean = float(torch.exp(self.policy.actor.sigma.detach()).mean().cpu().item()) * float(getattr(self.policy.actor, "sigma_scale", 1.0))

                
                dd = int(getattr(self.policy.actor, "delta_dim", 0))
                base_sigma = torch.exp(self.policy.actor.sigma.detach()).detach().cpu().numpy().astype(np.float64).reshape(-1)
                if dd > 0 and base_sigma.size > dd:
                    sigma_delta_mean = float(np.mean(base_sigma[:dd])) * float(getattr(self.policy.actor, "sigma_scale_delta", 1.0))
                    if str(getattr(self.policy.actor, "scores_mode", "learned")).strip().lower() == "fixed":
                        sigma_score_mean = float(getattr(self.policy.actor, "scores_fixed_sigma", sigma_delta_mean))
                    else:
                        sigma_score_mean = float(np.mean(base_sigma[dd:])) * float(getattr(self.policy.actor, "sigma_scale_score", 1.0))
                else:
                    sigma_delta_mean = float(np.mean(base_sigma)) * float(getattr(self.policy.actor, "sigma_scale", 1.0))
                    sigma_score_mean = float(sigma_delta_mean)
        except Exception:
            log_std_mean = 0.0
            sigma_mean = 0.0
            sigma_delta_mean = 0.0
            sigma_score_mean = 0.0

        
        rolling_spearman = float("nan")
        rolling_spearman_n = int(
            min(len(self._phaseg_reward_total_ep_samples), len(self._phaseg_paper_cost_ep_samples))
        )
        if rolling_spearman_n >= 2:
            rr = np.asarray(self._phaseg_reward_total_ep_samples[-rolling_spearman_n:], dtype=np.float64)
            cc = np.asarray(self._phaseg_paper_cost_ep_samples[-rolling_spearman_n:], dtype=np.float64)
            rolling_spearman = _spearman_corr(rr, -cc)
        self._phaseg_last_spearman = float(rolling_spearman)
        self._phaseg_last_spearman_n = int(rolling_spearman_n)

        
        n_dense_stats = int(dense_eps_this_rollout.size) if dense_eps_this_rollout.size > 0 else 0
        def _dense_episode_transition(prev_val: float, curr_val: float, n_ep: int) -> List[float]:
            if int(n_ep) <= 0:
                return []
            pv = float(prev_val)
            cv = float(curr_val)
            if not np.isfinite(cv):
                
                cv = pv if np.isfinite(pv) else float("nan")
            if not np.isfinite(pv):
                return [float(cv)] * int(n_ep)
            if int(n_ep) == 1:
                return [float(cv)]
            return np.linspace(float(pv), float(cv), int(n_ep) + 1, dtype=np.float64)[1:].astype(np.float64).tolist()
        if n_dense_stats > 0:
            prev_policy = float(self.policy_loss_ep_dense_history[-1]) if self.policy_loss_ep_dense_history else float(policy_loss)
            prev_value = float(self.value_loss_ep_dense_history[-1]) if self.value_loss_ep_dense_history else float(value_loss)
            prev_entropy = float(self.entropy_ep_dense_history[-1]) if self.entropy_ep_dense_history else float(entropy_per_dim)
            prev_kl = float(self.kl_ep_dense_history[-1]) if self.kl_ep_dense_history else float(kl_per_dim)
            prev_clip = float(self.clipfrac_ep_dense_history[-1]) if self.clipfrac_ep_dense_history else float(clipfrac)
            prev_ev = (
                float(self.explained_variance_ep_dense_history[-1])
                if self.explained_variance_ep_dense_history
                else float(explained_variance)
            )
            prev_lr = float(self.lr_ep_dense_history[-1]) if self.lr_ep_dense_history else float(lr_now)

            self.policy_loss_ep_dense_history.extend(
                _dense_episode_transition(prev_policy, float(policy_loss), n_dense_stats)
            )
            self.value_loss_ep_dense_history.extend(
                _dense_episode_transition(prev_value, float(value_loss), n_dense_stats)
            )
            self.entropy_ep_dense_history.extend(
                _dense_episode_transition(prev_entropy, float(entropy_per_dim), n_dense_stats)
            )
            self.kl_ep_dense_history.extend(
                _dense_episode_transition(prev_kl, float(kl_per_dim), n_dense_stats)
            )
            self.clipfrac_ep_dense_history.extend(
                _dense_episode_transition(prev_clip, float(clipfrac), n_dense_stats)
            )
            self.explained_variance_ep_dense_history.extend(
                _dense_episode_transition(prev_ev, float(explained_variance), n_dense_stats)
            )
            self.lr_ep_dense_history.extend(
                _dense_episode_transition(prev_lr, float(lr_now), n_dense_stats)
            )

        
        self.reward_history.append(float(reward_total_ep_sum_mean))
        self.policy_loss_history.append(policy_loss)
        self.value_loss_history.append(value_loss)
        
        self.entropy_history.append(float(entropy_per_dim))
        self.kl_div_history.append(float(kl_per_dim))
        self.clipfrac_history.append(float(clipfrac))
        self.explained_variance_history.append(float(explained_variance))
        self.steps_history.append(int(self.env_step))
        self.train_episode_history.append(int(train_episode_now))
        self.cost_history.append(cost_mean)
        self.vio_history.append(vio_cnt_mean)
        self.improve_history.append(improve_mean)

        
        
        
        
        eval_out = {
            "eval_det_is_valid": 0.0,
            "eval_train_episode": float("nan"),
            "eval_det_return_mean": float("nan"),
            "eval_det_return_std": float("nan"),
            "eval_det_return_ci": float("nan"),
            "eval_det_return_ci95": float("nan"),
            "eval_det_paper_cost_mean": float("nan"),
            "eval_det_paper_cost_std": float("nan"),
            "eval_det_paper_cost_ci": float("nan"),
            "eval_det_paper_cost_ci95": float("nan"),
            "eval_det_paper_return_mean": float("nan"),
            "eval_det_paper_return_std": float("nan"),
            "eval_det_paper_return_ci": float("nan"),
            "eval_det_paper_return_ci95": float("nan"),
            "eval_det_n": 0,
            "eval_det_vio_mean": float("nan"),
            "eval_det_vio_any_frac_mean": float("nan"),
            "eval_det_vio_any_frac_std": float("nan"),
            "eval_det_vio_any_frac_ci": float("nan"),
            "eval_det_vio_any_frac_ci95": float("nan"),
            "eval_det_vio_metric_mean": float("nan"),
            "eval_det_coverage_fraction_mean": float("nan"),
            "eval_det_baseline_balanced_paper_cost_mean": float("nan"),
            "eval_det_baseline_greedy_delay_paper_cost_mean": float("nan"),
            "eval_det_baseline_greedy_energy_paper_cost_mean": float("nan"),
            "eval_det_imp_paper_cost_mean": float("nan"),
            "eval_det_imp_vs_balanced_paper_cost_mean": float("nan"),
            "eval_det_imp_vs_greedy_delay_paper_cost_mean": float("nan"),
            "eval_det_imp_vs_greedy_energy_paper_cost_mean": float("nan"),
            "eval_det_paper_cost_agg": "SUM",
            "eval_det_reward_design": str(getattr(self.env_cfg_eval, "reward_design", "")),
            "eval_det_train_load": float(getattr(self, "train_load_nominal", float("nan"))),
            "eval_det_loads_primary": float("nan"),
            "eval_det_loads_count": 0,
        }
        if int(self.env_step) >= int(self.next_eval_step):
            eval_out = self.det_eval()
            
            try:
                _is_valid = int(float(eval_out.get("eval_det_is_valid", 0.0) or 0.0))
                self._last_det_eval_step = int(self.env_step)
                self._last_det_eval_n = int(eval_out.get("eval_det_n", 0) or 0)
                self._last_det_eval_paper_return = float(eval_out.get("eval_det_paper_return_mean", float("nan")))
                print(
                    f"[HB][train][eval_det] step={int(self.env_step)} is_valid={_is_valid} n={self._last_det_eval_n} every={int(self.eval_every)} "
                    f"paper_return={self._last_det_eval_paper_return:.4f}",
                    flush=True,
                )
            except Exception:
                pass
            
            
            try:
                is_traj_only = bool(str(actor_scores_mode) == "fixed")
                guard_cfg = {}
                try:
                    _gc = (self.train_params or {}).get("phaseg_stability_guard", {})
                    if isinstance(_gc, dict):
                        guard_cfg = _gc
                except Exception:
                    guard_cfg = {}

                guard_enabled = bool(guard_cfg.get("enabled", True))
                guard_tol = float(guard_cfg.get("tol", 0.02))
                guard_windows = int(max(1, int(guard_cfg.get("eval_windows", 2))))
                guard_min_step = int(max(0, int(guard_cfg.get("min_step", 0))))
                guard_best_min_step = int(max(0, int(guard_cfg.get("best_min_step", 0))))
                guard_start = int(max(int(self.total_steps_target) - guard_windows * int(max(self.eval_every, 1)), 0))
                guard_start = int(max(guard_start, guard_min_step))
                best_step = int(self.best_ckpt_step) if self.best_ckpt_step is not None else -1
                if (
                    guard_enabled
                    and is_traj_only
                    and int(self.env_step) >= int(guard_start)
                    and int(best_step) >= int(guard_best_min_step)
                    and np.isfinite(self.best_eval_paper_cost)
                    and self.best_ckpt_path is not None
                ):
                    cost_now = float(eval_out.get("eval_det_paper_cost_mean", 0.0))
                    best = float(self.best_eval_paper_cost)
                    tol = float(max(0.0, guard_tol))  
                    if float(best) > 0.0 and float(cost_now) > float(best) * (1.0 + tol):
                        ok = self._restore_best_for_stability()
                        if ok:
                            print(
                                f"[HB][train][GUARD] step={int(self.env_step)} fixed-eval"
                                f"cost_now={cost_now:.4f} > best={best:.4f}*(1+{tol:.2f}) "
                                f"(start={int(guard_start)}, best_step={int(best_step)}) -> restore best & freeze",
                                flush=True,
                            )
                            eval_out = self.det_eval()
            except Exception:
                pass

            
            try:
                eval_out["eval_train_episode"] = float(train_episode_now)
            except Exception:
                eval_out["eval_train_episode"] = float("nan")
            self.eval_steps_history.append(int(self.env_step))
            self.eval_train_episode_history.append(int(train_episode_now))
            self.eval_reward_history.append(float(eval_out["eval_det_return_mean"]))
            self.eval_reward_ci_history.append(float(eval_out.get("eval_det_return_ci", 0.0)))
            self.eval_paper_cost_history.append(float(eval_out["eval_det_paper_cost_mean"]))
            self.eval_paper_cost_std_history.append(float(eval_out["eval_det_paper_cost_std"]))
            self.eval_paper_cost_ci_history.append(float(eval_out["eval_det_paper_cost_ci"]))
            self.eval_paper_return_history.append(float(eval_out.get("eval_det_paper_return_mean", -float(eval_out["eval_det_paper_cost_mean"]))))
            self.eval_paper_return_ci_history.append(float(eval_out.get("eval_det_paper_return_ci", float(eval_out.get("eval_det_paper_cost_ci", 0.0)))))
            self.eval_n_history.append(int(eval_out["eval_det_n"]))
            self.eval_vio_history.append(float(eval_out["eval_det_vio_mean"]))
            self.eval_vio_any_frac_history.append(float(eval_out.get("eval_det_vio_any_frac_mean", 0.0)))
            self.eval_vio_any_frac_ci_history.append(float(eval_out.get("eval_det_vio_any_frac_ci", 0.0)))
            self.eval_improve_history.append(float(eval_out["eval_det_imp_paper_cost_mean"]))
            self.eval_coverage_history.append(float(eval_out.get("eval_det_coverage_fraction_mean", float("nan"))))
            self.eval_comp_gain_ratio_history.append(float(eval_out.get("comp_gain_ratio", comp_gain_ratio_mean)))
            self.eval_ris_gain_ratio_history.append(float(eval_out.get("ris_gain_ratio", ris_gain_ratio_mean)))

            
            try:
                eval_out_meta = dict(eval_out)
                if "comp_gain_ratio" not in eval_out_meta:
                    eval_out_meta["comp_gain_ratio"] = float(comp_gain_ratio_mean)
                if "ris_gain_ratio" not in eval_out_meta:
                    eval_out_meta["ris_gain_ratio"] = float(ris_gain_ratio_mean)
                meta_decision = self._maybe_meta_step(
                    eval_out=eval_out_meta,
                    train_episode_now=int(train_episode_now),
                    clipfrac_now=float(clipfrac),
                    kl_per_dim_now=float(kl_per_dim),
                    explained_variance_now=float(explained_variance),
                )
                if isinstance(meta_decision, dict) and meta_decision:
                    print(
                        f"[HB][meta] step={int(self.env_step)} eval_idx={int(len(self.eval_steps_history))} "
                        f"src={self.meta_source} action={str(meta_decision.get('action_id', 'hold'))} "
                        f"fallback={1 if bool(meta_decision.get('fallback_flag', False)) else 0} "
                        f"patch={meta_decision.get('patch', {})}",
                        flush=True,
                    )
            except Exception as e:
                lvl = "FATAL" if bool(getattr(self, "meta_strict_train_apply", False)) else "WARN"
                print(f"[HB][meta][{lvl}] meta_step failed: {type(e).__name__}: {e}", flush=True)
                if bool(getattr(self, "meta_strict_train_apply", False)):
                    raise

            skip_best_once = bool(getattr(self, "_meta_skip_best_once", False))
            if skip_best_once:
                self._meta_skip_best_once = False
            if (not skip_best_once) and float(eval_out["eval_det_paper_cost_mean"]) > 0.0 and float(eval_out["eval_det_paper_cost_mean"]) < float(self.best_eval_paper_cost):
                self.best_eval_paper_cost = float(eval_out["eval_det_paper_cost_mean"])
                self.best_ckpt_step = int(self.env_step)
                self.best_ckpt_path = self.save_ckpt("best")
                try:
                    with open(self.run_dir / "best_ckpt.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "best_step": int(self.best_ckpt_step),
                                "best_paper_cost": float(self.best_eval_paper_cost),
                                "best_ckpt_path": str(self.best_ckpt_path),
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )
                except Exception:
                    pass

            self.next_eval_step += int(max(self.eval_every, 1))

        try:
            eval_valid = int(float(eval_out.get("eval_det_is_valid", 0.0) or 0.0))
            eval_n_now = int(eval_out.get("eval_det_n", 0) or 0)
            if eval_valid > 0 and eval_n_now > 0:
                self._latest_eval_det_paper_cost = float(eval_out.get("eval_det_paper_cost_mean", float("nan")))
                self._latest_eval_det_paper_cost_ci = float(eval_out.get("eval_det_paper_cost_ci", float("nan")))
                self._latest_eval_det_vio_any_frac = float(eval_out.get("eval_det_vio_any_frac_mean", float("nan")))
                self._latest_eval_det_improve = float(eval_out.get("eval_det_imp_paper_cost_mean", float("nan")))
        except Exception:
            pass

        if n_dense_stats > 0:
            prev_eval_cost = (
                float(self.eval_det_paper_cost_ep_dense_history[-1])
                if self.eval_det_paper_cost_ep_dense_history
                else float(self._latest_eval_det_paper_cost)
            )
            prev_eval_cost_ci = (
                float(self.eval_det_paper_cost_ci_ep_dense_history[-1])
                if self.eval_det_paper_cost_ci_ep_dense_history
                else float(self._latest_eval_det_paper_cost_ci)
            )
            prev_eval_vio = (
                float(self.eval_det_vio_any_frac_ep_dense_history[-1])
                if self.eval_det_vio_any_frac_ep_dense_history
                else float(self._latest_eval_det_vio_any_frac)
            )
            prev_eval_improve = (
                float(self.eval_det_improve_ep_dense_history[-1])
                if self.eval_det_improve_ep_dense_history
                else float(self._latest_eval_det_improve)
            )

            self.eval_det_paper_cost_ep_dense_history.extend(
                _dense_episode_transition(prev_eval_cost, float(self._latest_eval_det_paper_cost), n_dense_stats)
            )
            self.eval_det_paper_cost_ci_ep_dense_history.extend(
                _dense_episode_transition(prev_eval_cost_ci, float(self._latest_eval_det_paper_cost_ci), n_dense_stats)
            )
            self.eval_det_vio_any_frac_ep_dense_history.extend(
                _dense_episode_transition(prev_eval_vio, float(self._latest_eval_det_vio_any_frac), n_dense_stats)
            )
            self.eval_det_improve_ep_dense_history.extend(
                _dense_episode_transition(prev_eval_improve, float(self._latest_eval_det_improve), n_dense_stats)
            )

        
        if int(self.env_step) >= int(self.next_save_step):
            self.save_ckpt(f"step_{int(self.env_step)}")
            self.next_save_step += int(max(self.save_every, 1))

        
        train_stage = "train"
        trainable_parts = "delta+scores"
        if int(self.total_steps_target) <= 15000:
            train_stage = "A_canary"
        elif int(self.total_steps_target) <= 60000:
            train_stage = "A_probfix"
        elif str(actor_scores_mode) == "fixed":
            train_stage = "B_traj_only"
            trainable_parts = "delta"

        
        self._last_approx_kl = float(kl_per_dim)
        self._last_clipfrac = float(clipfrac)
        self._last_entropy = float(entropy_per_dim)
        self._last_value_loss = float(value_loss)

        meta_stats = self.meta_controller.stats() if (self.meta_controller is not None) else {}
        meta_last = self._meta_last_decision if isinstance(self._meta_last_decision, dict) else {}
        meta_action_id = str(meta_last.get("action_id", "hold"))
        meta_fallback_flag = 1.0 if bool(meta_last.get("fallback_flag", False)) else 0.0
        meta_latency_ms = float(meta_last.get("latency_ms", float("nan")))

        
        row = {
            "act_dim": int(act_dim_now),
            "logp_mode": str(logp_mode),
            "logp_scale": float(logp_scale),
            "logp_act_dim": int(logp_act_dim),
            "logp_is_mean": float(logp_is_mean),
            "selfcheck_enabled": 1.0 if bool(getattr(self.policy, "_selfcheck_enabled", True)) else 0.0,
            "uses_tanh_squash": 1.0,
            "uses_clamp": 0.0,
            "policy_dist": "SquashedGaussian",
            
            "improve_mode": str(getattr(self.preprocess, "improve_mode", "paper_improve")),
            "scores_mode": str(actor_scores_mode),
            "scores_mode_fixed": 1.0 if str(actor_scores_mode) == "fixed" else 0.0,
            "act_dim_effective": int(act_dim_effective),
            "train_stage": str(train_stage),
            "trainable_parts": str(trainable_parts),
            "scores_fixed_mean": float(scores_fixed_mean),
            "scores_fixed_std": float(scores_fixed_std),
            "action_oor_frac": float(action_oor_frac),
            "action_oor_any_frac": float(action_oor_any_frac),
            "action_sat_frac": float(action_sat_frac),
            "action_sat_any_frac": float(action_sat_any_frac),
            "global_step": int(self.env_step),
            "train_episode": int(train_episode_now),
            "update_count": int(len(self.steps_history)),
            "train_reward_mean": float(train_reward_mean),
            "train_reward_std": float(train_reward_std),
            "train_reward_step_mean": float(train_reward_step_mean),
            "train_reward_step_std": float(train_reward_step_std),
            # PhaseG-v1.1 canonical columns (step)
            "reward_total_step": float(reward_total_step_mean),
            "paper_cost_step": float(paper_cost_step_mean),
            "ep_return_train_p05": float(ep_return_p05) if np.isfinite(ep_return_p05) else float("nan"),
            "ep_return_train_p50": float(ep_return_p50) if np.isfinite(ep_return_p50) else float("nan"),
            "ep_return_train_p95": float(ep_return_p95) if np.isfinite(ep_return_p95) else float("nan"),
            
            
            "reward_train_mean": float(train_reward_mean),
            "reward_train_step_mean": float(train_reward_step_mean),
            "reward_train_raw_p05": float(reward_raw_p05),
            "reward_train_raw_p50": float(reward_raw_p50),
            "reward_train_raw_p95": float(reward_raw_p95),
            "reward_clip_hit_frac": float(clip_hit_frac),
            "paper_cost_mean": float(cost_mean),
            "vio_any_frac": float(violation_any_frac),
            
            
            
            "clipfrac_mean": float(clipfrac),
            "violation_any_frac": float(violation_any_frac),
            "reward_raw_mean": float(reward_raw_mean),
            "reward_raw_std": float(reward_raw_std),
            "reward_raw_p05": float(reward_raw_p05),
            "reward_raw_p50": float(reward_raw_p50),
            "reward_raw_p95": float(reward_raw_p95),
            "clip_hit_frac": float(clip_hit_frac),
            "train_len_mean": float(train_len_mean),
            "train_len_std": float(train_len_std),
            "train_n_ep": int(train_n_ep),
            "returns_mean": float(returns_mean),
            "returns_std": float(returns_std),
            "value_bias_proxy": float(value_bias_proxy),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "vf_clip_hit_frac": float(vf_clip_hit_frac),
            "bc_warmstart_alpha": float(self._bc_last_alpha),
            "bc_warmstart_cost_alpha": float(self._bc_last_cost_alpha),
            "bc_warmstart_loss_raw": float(self._bc_last_loss_raw),
            "bc_warmstart_cost_loss_raw": float(self._bc_last_cost_loss_raw),
            "bc_warmstart_loss_weighted": float(self._bc_last_loss_weighted),
            "bc_warmstart_cost_loss_weighted": float(self._bc_last_cost_loss_weighted),
            "bc_warmstart_updates": int(self._bc_last_updates),
            "tail_stabilize_active": 1.0 if bool(getattr(self, "_tail_active", False)) else 0.0,
            "head_tune_enable": 1.0 if bool(getattr(self.policy, "_head_tune_enable", False)) else 0.0,
            "a1_head_clip_used_ratio": float(a1_head_clip_used_ratio),
            "a1_head_clip_ready": float(a1_head_clip_ready),
            "a1_head_ent_used_ratio": float(a1_head_ent_used_ratio),
            "ppo_eps_clip_current": float(getattr(self.policy, "_eps_clip", float("nan"))),
            "ppo_eps_clip_delta_current": float(getattr(self.policy, "_eps_clip_delta", float("nan"))),
            "ppo_eps_clip_score_current": float(getattr(self.policy, "_eps_clip_score", float("nan"))),
            "ppo_value_clip_current": float(getattr(self.policy, "_eps_clip_value", float("nan"))),
            "ppo_target_kl_current": float(getattr(self.policy, "_target_kl_stop", float("nan"))),
            "explained_variance": float(explained_variance),
            "entropy": float(entropy),
            "entropy_sum": float(entropy_sum),
            "entropy_per_dim": float(entropy_per_dim),
            "kl": float(kl),
            "kl_per_dim": float(kl_per_dim),
            "clipfrac": float(clipfrac),
            "clipfrac_per_dim": float(clipfrac_per_dim),
            "lr": float(lr_now),
            "log_std_mean": float(log_std_mean),
            "sigma_mean": float(sigma_mean),
            "sigma_delta_mean": float(sigma_delta_mean),
            "sigma_score_mean": float(sigma_score_mean),
            "exploration_std_scale": float(std_now),
            "exploration_std_scale_delta": float(std_delta_now),
            "exploration_std_scale_score": float(std_score_now),
            "ent_coef": float(ent_now),
            "ent_coef_delta": float(getattr(self.policy, "_weight_ent_delta", ent_now)),
            "ent_coef_score": float(getattr(self.policy, "_weight_ent_score", ent_now)),
            **eval_out,
            "cost_mean": float(cost_mean),
            "vio_mean": float(vio_cnt_mean),
            "delta_scale_mean": float(delta_scale_mean),
            "vio_wall_mean": float(vio_wall_mean),
            "vio_wall_any_frac": float(vio_wall_any_frac),
            "vio_metric_mean": float(vio_metric_mean),
            "coverage_fraction_mean": float(coverage_fraction_mean),
            "coverage_margin_mean": float(coverage_margin_mean),
            "collision_margin_mean": float(collision_margin_mean),
            "baseline_balanced_paper_cost_mean": float(baseline_balanced_paper_cost_mean),
            "baseline_greedy_delay_paper_cost_mean": float(baseline_greedy_delay_paper_cost_mean),
            "baseline_greedy_energy_paper_cost_mean": float(baseline_greedy_energy_paper_cost_mean),
            "paper_improve_mean": float(paper_improve_mean),
            "paper_improve_vs_balanced_mean": float(paper_improve_vs_balanced_mean),
            "paper_improve_vs_greedy_delay_mean": float(paper_improve_vs_greedy_delay_mean),
            "paper_improve_vs_greedy_energy_mean": float(paper_improve_vs_greedy_energy_mean),
            "improve_mean": float(improve_mean),
            "env_total_mean": float(env_reward_mean),
            "lambda_v": float(lambda_v_effective_mean) if np.isfinite(float(lambda_v_effective_mean)) else float(getattr(self.preprocess, "lambda_v", 0.0)),
            "lag_bucket_id": float(getattr(self, "_lag_bucket_last_id", -1)),
            "lag_target_current": float(getattr(self, "_lag_target_current", float("nan"))),
            "lag_eta_current": float(getattr(self, "_lag_eta_current", float("nan"))),
            "lag_lambda_min_current": float(getattr(self, "_lag_lambda_min_current", float("nan"))),
            "lag_lambda_max_current": float(getattr(self, "_lag_lambda_max_current", float("nan"))),
            # PhaseG-v1.1 canonical columns (gain ratios + decomposition aliases)
            "comp_gain_ratio": float(comp_gain_ratio_mean),
            "ris_gain_ratio": float(ris_gain_ratio_mean),
            "shaping_term": float(reward_shaping_step_mean),
            "penalty_term": float(max(-float(reward_constraint_step_mean), 0.0)),
            "comp_gain_ratio_mean": float(comp_gain_ratio_mean),
            "ris_gain_ratio_mean": float(ris_gain_ratio_mean),
            "env_shaping_mean": float(env_shaping_mean),
            "total_shaping_mean": float(total_shaping_mean),
            "total_shaping_all_mean": float(total_shaping_all_mean),
            "comp_bonus_mean": float(comp_bonus_mean),
            "ris_bonus_mean": float(ris_bonus_mean),
            "comp_ris_ratio": float(comp_ris_ratio),
            "comp_enable_rate": float(comp_enable_rate),
            "service_switch_count": float(service_switch_count),
            "theta_entropy": float(theta_entropy),
            "comp_score_temp_runtime": float(comp_score_temp_runtime),
            "z4_comp_meta_enable_mean": float(z4_comp_meta_enable_mean),
            "z4_comp_meta_warm_scale_mean": float(z4_comp_meta_warm_scale_mean),
            "z4_comp_meta_thr_effective_mean": float(z4_comp_meta_thr_effective_mean),
            "z4_comp_meta_temp_effective_mean": float(z4_comp_meta_temp_effective_mean),
            "z4_comp_meta_ctrl_thr_raw_mean": float(z4_comp_meta_ctrl_thr_raw_mean),
            "z4_comp_meta_ctrl_temp_raw_mean": float(z4_comp_meta_ctrl_temp_raw_mean),
            "z4_comp_meta_thr_delta_ema_mean": float(z4_comp_meta_thr_delta_ema_mean),
            "z4_comp_meta_temp_scale_ema_mean": float(z4_comp_meta_temp_scale_ema_mean),
            "z4_assoc_stage_enable_mean": float(z4_assoc_stage_enable_mean),
            "z4_assoc_policy_mix_mean": float(z4_assoc_policy_mix_mean),
            "z4_assoc_train_frac_mean": float(z4_assoc_train_frac_mean),
            "z4_assoc_stage_score_width_mean": float(z4_assoc_stage_score_width_mean),
            "z4_assoc_stage_comp_rule_thr_mean": float(z4_assoc_stage_comp_rule_thr_mean),
            "theta_mode_effective": str(theta_mode_effective),
            "theta_mode_effective_agent_frac": float(theta_mode_effective_agent_frac),
            
            "reward_paper_step_mean": float(reward_paper_step_mean),
            "reward_paper_abs_step_mean": float(reward_paper_abs_step_mean),
            "reward_paper_delta_step_mean": float(reward_paper_delta_step_mean),
            "reward_paper_adv_step_mean": float(reward_paper_adv_step_mean),
            "paper_delta_norm_step_mean": float(paper_delta_norm_step_mean),
            "paper_adv_norm_step_mean": float(paper_adv_norm_step_mean),
            "constraint_signal_step_mean": float(constraint_signal_step_mean),
            "lambda_v_effective_mean": float(lambda_v_effective_mean),
            "reward_constraint_step_mean": float(reward_constraint_step_mean),
            "reward_shaping_step_mean": float(reward_shaping_step_mean),
            "reward_misc_step_mean": float(reward_misc_step_mean),
            "reward_proximity_step_mean": float(reward_proximity_step_mean),
            "shaping_gate_mean": float(shaping_gate_mean),
            "beta_comp_t_mean": float(beta_comp_t_mean),
            "beta_ris_t_mean": float(beta_ris_t_mean),
            "beta_train_frac_mean": float(beta_train_frac_mean),
            "z4_gap_raw_step_mean": float(z4_gap_raw_step_mean),
            "z4_gap_step_clip_mean": float(z4_gap_step_clip_mean),
            "z4_gap_step_ema_mean": float(z4_gap_step_ema_mean),
            "z4_gap_step_mean": float(z4_gap_step_mean),
            "z4_gap_reward_step_mean": float(z4_gap_reward_step_mean),
            "z4_gap_lambda_input_mean": float(z4_gap_lambda_input_mean),
            "z4_gap_lambda_sched_mean": float(z4_gap_lambda_sched_mean),
            "z4_gap_lambda_cap_mean": float(z4_gap_lambda_cap_mean),
            "z4_gap_lambda_eff_mean": float(z4_gap_lambda_eff_mean),
            "z4_gap_grad_ratio_eff_mean": float(z4_gap_grad_ratio_eff_mean),
            "z4_gap_shape_guard_trigger_mean": float(z4_gap_shape_guard_trigger_mean),
            "t6_main_weight_mean": float(t6_main_weight_mean),
            "t6_potential_scale_mean": float(t6_potential_scale_mean),
            "t6_traj_scale_mean": float(t6_traj_scale_mean),
            "t6_anneal_ratio_mean": float(t6_anneal_ratio_mean),
            "t6_violation_ema_mean": float(t6_violation_ema_mean),
            "t6_pid_frozen_mean": float(t6_pid_frozen_mean),
            "comp_bonus_raw_mean": float(comp_bonus_raw_mean),
            "ris_bonus_raw_mean": float(ris_bonus_raw_mean),
            "reward_paper_ep": float(reward_paper_ep),
            "reward_shaping_ep": float(reward_shaping_ep),
            "reward_constraint_ep": float(reward_constraint_ep),
            "reward_misc_ep": float(reward_misc_ep),
            
            "reward_contract_abs_err_mean": float(reward_contract_abs_err_mean),
            "reward_contract_abs_err_max": float(reward_contract_abs_err_max),
            "phaseg_spearman_corr_roll": float(rolling_spearman),
            "phaseg_spearman_n_roll": float(rolling_spearman_n),
            "ep_sum_n": float(ep_sum_n),
            
            "reward_total_ep": float(reward_total_ep_sum_mean),
            "paper_cost_ep": float(paper_cost_ep_sum_mean),
            "reward_total_ep_sum": float(reward_total_ep_sum_mean),
            "paper_cost_ep_sum": float(paper_cost_ep_sum_mean),
            "reward_total_ep_agg": "SUM",
            "paper_cost_ep_agg": "SUM",
            "reward_clip_order": "STEP_CLIP_THEN_EP_SUM",
            "phaseg_contract_version": "phaseg_v1.1_sum_stepclip",
            "train_load": float(getattr(self, "train_load_nominal", float("nan"))),
            "train_load_current": float(getattr(self, "_curriculum_load_last", getattr(self, "train_load_nominal", float("nan")))),
            "eval_load_primary": float(eval_out.get("eval_det_loads_primary", float("nan"))),
            "eval_load_count": float(eval_out.get("eval_det_loads_count", float("nan"))),
            
            "meta_enabled": 1.0 if self.meta_enabled else 0.0,
            "meta_source": str(self.meta_source),
            "meta_action_id": str(meta_action_id),
            "meta_fallback_flag": float(meta_fallback_flag),
            "meta_fallback_rate": float(meta_stats.get("meta_fallback_rate", 0.0)),
            "meta_timeout_rate": float(meta_stats.get("meta_timeout_rate", 0.0)),
            "meta_json_invalid_rate": float(meta_stats.get("meta_json_invalid_rate", 0.0)),
            "meta_decision_latency_ms": float(meta_latency_ms),
        }
        self.metrics_writer.writerow(row)
        self.metrics_csv_f.flush()

        
        if len(self.steps_history) == 1 or (len(self.steps_history) % 10 == 0) or int(self.env_step) >= int(self.total_steps_target):
            elapsed = float(time.time() - self.start_time)
            pct = 100.0 * float(self.env_step) / float(max(1, self.total_steps_target))
            eta = elapsed * (float(self.total_steps_target) / float(max(1, self.env_step)) - 1.0) if self.env_step > 0 else 0.0
            lam_now = (
                float(lambda_v_effective_mean)
                if np.isfinite(float(lambda_v_effective_mean))
                else float(getattr(self.preprocess, "lambda_v", 0.0))
            )
            
            try:
                last_eval_step = int(getattr(self, "_last_det_eval_step", -1) or -1)
                last_eval_n = int(getattr(self, "_last_det_eval_n", 0) or 0)
                last_eval_ret = float(getattr(self, "_last_det_eval_paper_return", float("nan")))
                last_eval_is_valid = 1 if (last_eval_n > 0 and np.isfinite(last_eval_ret)) else 0
            except Exception:
                last_eval_step = -1
                last_eval_n = 0
                last_eval_ret = float("nan")
                last_eval_is_valid = 0
            print(
                f"[HB][train][P4] step={int(self.env_step)}/{int(self.total_steps_target)} ({pct:.2f}%) "
                f"upd={len(self.steps_history)} EV={explained_variance:.3f} "
                f"cost={cost_mean:.4f} vio_cnt={vio_cnt_mean:.3f} wall={vio_wall_mean:.3f}({vio_wall_any_frac:.2f}) "
                f"scores={actor_scores_mode}/eff{act_dim_effective} s_m={scores_fixed_mean:.3f} s_s={scores_fixed_std:.3f} "
                f"vio_m={vio_metric_mean:.3f} vio_any={violation_any_frac:.2f} cov={coverage_fraction_mean:.3f} "
                f"shp={total_shaping_mean:.3f}(c={comp_bonus_mean:.3f},r={ris_bonus_mean:.3f}) "
                f"raw_q={reward_raw_p05:.2f}/{reward_raw_p50:.2f}/{reward_raw_p95:.2f} clip_hit={clip_hit_frac:.2f} "
                f"KL(sum)={kl_sum:.3f} KL/d={kl_per_dim:.4f} clipfrac/d={clipfrac_per_dim:.3f} "
                f"eval_det(v={last_eval_is_valid},n={last_eval_n},ret={last_eval_ret:.3f},step={last_eval_step}) "
                f"oor={action_oor_any_frac:.3f} sat={action_sat_any_frac:.3f} "
                f"ds={delta_scale_mean:.2f} lam={lam_now:.3f} lr={lr_now:.2e} ent={ent_now:.4g} "
                f"tail={1 if bool(getattr(self, '_tail_active', False)) else 0} "
                f"eps={float(getattr(self.policy, '_eps_clip', float('nan'))):.3f} "
                f"kl_tgt={float(getattr(self.policy, '_target_kl_stop', float('nan'))):.4f} "
                f"sig_d={sigma_delta_mean:.3f} sig_s={sigma_score_mean:.3f} "
                f"bc(a={float(self._bc_last_alpha):.4f},l={float(self._bc_last_loss_raw):.4f},u={int(self._bc_last_updates)}) "
                f"std_d={std_delta_now:.3f} std_s={std_score_now:.3f} "
                f"meta={self.meta_source}:{meta_action_id}/fb{int(meta_fallback_flag)} "
                f"(fr={float(meta_stats.get('meta_fallback_rate', 0.0)):.2f},to={float(meta_stats.get('meta_timeout_rate', 0.0)):.2f}) "
                f"t={fmt_hms(elapsed)} ETA={fmt_hms(eta)} run_dir={self.run_dir}",
                flush=True,
            )
            hb_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(
                f"[HB][train][G1.1] ts={hb_ts} step={int(self.env_step)} csv={self.run_dir / 'train_metrics.csv'} "
                f"agg=SUM reward_total_ep={reward_total_ep_sum_mean:.4f} paper_cost_ep={paper_cost_ep_sum_mean:.4f} ep_n={ep_sum_n} "
                f"train_load={float(getattr(self, 'train_load_nominal', float('nan'))):.2f} "
                f"eval_load={float(eval_out.get('eval_det_loads_primary', float('nan'))):.2f} "
                f"contract(mean={reward_contract_abs_err_mean:.3e},max={reward_contract_abs_err_max:.3e}) "
                f"spearman_roll={rolling_spearman:.4f}@n={rolling_spearman_n} "
                f"ppo(kl={kl_per_dim:.6f},clipfrac={clipfrac:.4f},ent={entropy_per_dim:.4f},"
                f"vf={value_loss:.6f},vfclip={vf_clip_hit_frac:.3f})",
                flush=True,
            )

            
            if int(self.env_step) <= 50000 and float(explained_variance) < 0.2 and b2 is not None:
                try:
                    def _stat(x: Any) -> Tuple[float, float, float, float]:
                        arr = np.asarray(x, dtype=np.float64).reshape(-1)
                        if arr.size <= 0:
                            return 0.0, 0.0, 0.0, 0.0
                        return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))

                    ret_m, ret_s, ret_min, ret_max = _stat(b2.returns.detach().cpu().numpy())
                    vs_m, vs_s, vs_min, vs_max = _stat(b2.v_s.detach().cpu().numpy())
                    adv_m, adv_s, adv_min, adv_max = _stat(b2.adv.detach().cpu().numpy())
                    print(
                        f"[HB][diag][P4] returns(m={ret_m:.3g},s={ret_s:.3g},min={ret_min:.3g},max={ret_max:.3g}) "
                        f"v_s(m={vs_m:.3g},s={vs_s:.3g},min={vs_min:.3g},max={vs_max:.3g}) "
                        f"adv(m={adv_m:.3g},s={adv_s:.3g},min={adv_min:.3g},max={adv_max:.3g})",
                        flush=True,
                    )
                except Exception:
                    pass

        
        if int(self.env_step) >= int(self.total_steps_target):
            self.stop_fn_flag = True


def run_official_training(
    *,
    run_dir: Path,
    env_cfg: JointEnvConfig,
    train_cfg: dict,
    device: torch.device,
    selfcheck_enabled: bool = True,
) -> Dict[str, Any]:
    """
    P4/meta
    """
    train_params = train_cfg.get("train", {}) or {}
    env_params = train_cfg.get("env", {}) or {}

    total_steps = int(train_params.get("total_steps", 300000))
    rollout_steps = int(train_params.get("rollout_steps", 2048))
    train_epochs = int(train_params.get("train_epochs", 6))
    batch_size = int(train_params.get("batch_size", 512))

    num_train_envs = max(1, int(env_params.get("num_train_envs", 1)))
    use_subproc = bool(env_params.get("use_subproc", False))

    
    
    try:
        if int(getattr(env_cfg, "train_total_steps", 0) or 0) <= 0:
            steps_per_env = int(math.ceil(float(total_steps) / float(max(num_train_envs, 1))))
            env_cfg = dc_replace(env_cfg, train_total_steps=int(max(steps_per_env, 1)))
            print(f"[HB][train][PhaseG] env_cfg.train_total_steps(per-env) = {int(getattr(env_cfg, 'train_total_steps', 0))}", flush=True)
    except Exception:
        pass

    if rollout_steps % num_train_envs != 0:
        print(
            f"[HB][train][P4][WARN] rollout_steps={rollout_steps}  num_train_envs={num_train_envs} "
            f"Tianshou Collector  <={num_train_envs-1} transition",
            flush=True,
        )

    seed = int(train_cfg.get("seed", 42))
    seed_base = int(seed)

    # -----------------------------
    
    # -----------------------------
    meta_inline = train_params.get("meta", {}) or {}
    if not isinstance(meta_inline, dict):
        meta_inline = {}
    meta_source_in = str(
        meta_inline.get("source", train_params.get("meta_source", ""))
    ).strip().lower()
    meta_enable_raw = meta_inline.get("enabled", train_params.get("meta_enable", None))
    meta_enable_in: Optional[bool]
    if meta_enable_raw is None:
        meta_enable_in = None
    else:
        meta_enable_in = bool(meta_enable_raw)
    meta_yaml_raw = str(
        meta_inline.get("meta_yaml", train_params.get("meta_yaml", ""))
    ).strip()
    meta_yaml_path: Optional[Path] = None
    if meta_yaml_raw:
        p = Path(meta_yaml_raw)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        meta_yaml_path = p

    meta_controller = create_meta_controller(
        run_dir=run_dir,
        seed=int(seed_base),
        meta_yaml=meta_yaml_path,
        inline_cfg=meta_inline,
        source_override=(meta_source_in if meta_source_in else None),
        enabled_override=meta_enable_in,
    )
    if meta_controller is not None:
        print(
            f"[HB][meta] enabled=1 source={meta_controller.source} yaml={str(meta_yaml_path) if meta_yaml_path else 'inline/default'} "
            f"allow={list(meta_controller.allow_actions)} warmup={meta_controller.warmup_evals} every={meta_controller.decision_every}",
            flush=True,
        )
    else:
        print("[HB][meta] enabled=0", flush=True)

    def _make_env(i: int):
        return CompRISEnvGym(config=env_cfg, seed=seed_base + 1000 * int(i))

    
    if use_subproc and num_train_envs > 1:
        venv = SubprocVectorEnv([lambda i=i: _make_env(i) for i in range(num_train_envs)])
    else:
        venv = DummyVectorEnv([lambda i=i: _make_env(i) for i in range(num_train_envs)])

    
    obs_space = venv.observation_space[0] if isinstance(venv.observation_space, list) else venv.observation_space
    act_space = venv.action_space[0] if isinstance(venv.action_space, list) else venv.action_space
    obs_dim = int(obs_space.shape[0])
    act_dim = int(act_space.shape[0])

    obs_normalizer = ObsNormalizer(obs_dim) if bool(train_params.get("obs_norm", True)) else None

    selfcheck_enabled = bool(selfcheck_enabled)

    # PPO policy
    net_cfg = train_cfg.get("network", {}) or {}
    hidden_sizes = tuple(net_cfg.get("hidden_sizes", [512, 512]))
    use_layernorm = bool(net_cfg.get("use_layernorm", True))
    share_preprocess = bool(net_cfg.get("share_preprocess", True))
    activation_name = str(net_cfg.get("activation", "relu")).strip().lower()
    if activation_name == "tanh":
        activation = torch.nn.Tanh
    elif activation_name == "gelu":
        activation = torch.nn.GELU
    else:
        activation = torch.nn.ReLU

    lr = float(train_params.get("lr", 3e-5))
    lr_critic_raw = train_params.get("lr_critic", None)
    lr_critic = float(lr_critic_raw) if lr_critic_raw is not None else None
    gamma = float(train_params.get("gamma", 0.99))
    lam = float(train_params.get("lam", 0.95))
    clip_ratio = float(train_params.get("clip_ratio", 0.15))
    target_kl = float(train_params.get("target_kl", 0.0))
    vf_coef = float(train_params.get("vf_coef", 0.5))
    ent_coef_init = float(train_params.get("ent_coef_init", train_params.get("ent_coef", 0.0)))
    max_grad_norm = float(train_params.get("max_grad_norm", 0.3))
    max_grad_norm_critic_raw = train_params.get("max_grad_norm_critic", None)
    max_grad_norm_critic = float(max_grad_norm_critic_raw) if max_grad_norm_critic_raw is not None else None
    vf_loss_clip_max = float(max(float(train_params.get("vf_loss_clip_max", 0.0)), 0.0))
    dual_optimizer_cfg = train_params.get("dual_optimizer", None)
    dual_optimizer = bool(dual_optimizer_cfg) if dual_optimizer_cfg is not None else (lr_critic is not None)
    value_clip_enable = bool(train_params.get("value_clip", False))
    adv_norm_enable = bool(train_params.get("advantage_normalization", True))
    recompute_adv_enable = bool(train_params.get("recompute_advantage", False))
    reward_norm = bool(train_params.get("reward_normalization", train_params.get("value_normalization", False)))
    max_batchsize = int(train_params.get("max_batchsize", 256))

    
    
    
    
    logp_mode_cfg = str(train_params.get("logp_mode", "auto")).strip().lower()
    if logp_mode_cfg == "auto":
        logp_mode = "mean" if int(act_dim) >= 32 else "sum"
    else:
        logp_mode = logp_mode_cfg

    # =========================
    
    
    # =========================
    phase_c_cfg = (train_params.get("phase_c", {}) or {})
    scores_mode = str(phase_c_cfg.get("scores_mode", "learned")).strip().lower()
    if scores_mode not in ("learned", "fixed"):
        scores_mode = "learned"
    scores_fixed_method = str(phase_c_cfg.get("scores_fixed_method", "balanced")).strip().lower()
    scores_fixed_tanh_scale = float(phase_c_cfg.get("scores_fixed_tanh_scale", 2.5))
    scores_fixed_sigma = float(phase_c_cfg.get("scores_fixed_sigma", 0.02))

    delta_dim_cfg = 2 * int(getattr(env_cfg, "M", 0))
    act_dim_effective = int(delta_dim_cfg) if (str(scores_mode) == "fixed" and int(delta_dim_cfg) > 0) else int(act_dim)

    policy, optimizer = create_ppo_policy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        delta_dim=int(delta_dim_cfg),
        logp_act_dim=int(act_dim_effective),
        logp_mode=str(logp_mode),
        hidden_sizes=hidden_sizes,
        use_layernorm=use_layernorm,
        share_preprocess=share_preprocess,
        activation=activation,
        lr=lr,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=lam,
        eps_clip=clip_ratio,
        target_kl=float(target_kl),
        kl_stop_mult=1.5,
        vf_coef=vf_coef,
        ent_coef=ent_coef_init,
        max_grad_norm=max_grad_norm,
        max_grad_norm_critic=max_grad_norm_critic,
        vf_loss_clip_max=vf_loss_clip_max,
        dual_optimizer=bool(dual_optimizer),
        value_clip=value_clip_enable,
        advantage_normalization=adv_norm_enable,
        recompute_advantage=recompute_adv_enable,
        reward_normalization=reward_norm,
        max_batchsize=max_batchsize,
        max_action=float(net_cfg.get("max_action", 1.0)),
        device=str(device),
        unbounded=bool(net_cfg.get("unbounded", False)),
        conditioned_sigma=bool(net_cfg.get("conditioned_sigma", False)),
    )
    
    policy.action_space = act_space
    
    try:
        setattr(policy, "_head_tune_enable", bool(train_params.get("head_tune_enable", False)))
        setattr(policy, "_clip_delta_weight", float(max(float(train_params.get("clip_delta_weight", 0.0)), 0.0)))
        setattr(policy, "_clip_score_weight", float(max(float(train_params.get("clip_score_weight", 0.0)), 0.0)))
        setattr(policy, "_head_delta_dim", int(delta_dim_cfg))
        setattr(policy, "_vf_loss_norm_returns", bool(train_params.get("vf_loss_norm_returns", False)))
        setattr(policy, "_vf_loss_norm_floor", float(max(float(train_params.get("vf_loss_norm_floor", 1.0)), 1e-6)))
        setattr(policy, "_vf_loss_norm_tail_only", bool(train_params.get("vf_loss_norm_tail_only", True)))
        setattr(policy, "_adv_norm_clip", float(max(float(train_params.get("adv_norm_clip", 0.0)), 0.0)))
        setattr(
            policy,
            "_adv_norm_clip_tail",
            float(max(float(train_params.get("adv_norm_clip_tail", train_params.get("adv_norm_clip", 0.0))), 0.0)),
        )
        setattr(policy, "_adv_norm_rollout_level", bool(train_params.get("adv_norm_rollout_level", False)))
    except Exception as exc:
        _warn_once("phase_t6_a1a2_inject", f"A1/A2PPO{type(exc).__name__}: {exc}")

    # =========================
    
    # =========================
    try:
        
        setattr(policy.actor, "scores_mode", str(scores_mode))
        if str(scores_mode) == "fixed" and hasattr(policy.actor, "set_scores_fixed"):
            policy.actor.set_scores_fixed(
                cfg=env_cfg,
                mode=str(scores_fixed_method),
                tanh_scale=float(scores_fixed_tanh_scale),
                sigma_fixed=float(scores_fixed_sigma),
            )
    except Exception as exc:
        _warn_once("phasec_scores_mode_apply", f"scores_mode {type(exc).__name__}: {exc}")
    
    
    try:
        z4_act_mode = str(getattr(env_cfg, "z4_action_mode", "hierarchical")).strip().lower()
        cm_dim = 2 if (z4_act_mode == "position_only" and bool(getattr(env_cfg, "z4_comp_meta_enable", False))) else 0
        policy.actor.comp_meta_dim = int(cm_dim)
    except Exception:
        pass
    print(
        f"[HB][train][PhaseC] scores_mode={scores_mode} act_dim_effective={act_dim_effective} "
        f"scores_fixed_method={scores_fixed_method} tanh_scale={scores_fixed_tanh_scale} sigma_fixed={scores_fixed_sigma}",
        flush=True,
    )
    print(
        f"[HB][train][A1A2A3] head_tune={1 if bool(train_params.get('head_tune_enable', False)) else 0} "
        f"value_clip={1 if bool(value_clip_enable) else 0} "
        f"vf_norm={1 if bool(train_params.get('vf_loss_norm_returns', False)) else 0} "
        f"load_mode={str(train_params.get('train_load_sampling_mode', 'fixed')).strip().lower()}",
        flush=True,
    )
    lag_cfg_runtime = train_params.get("lagrange_vio", {}) or {}
    bc_cfg_runtime = train_params.get("bc_warmstart", {}) or {}
    bc_methods_runtime = _parse_list((bc_cfg_runtime or {}).get("methods", []), str)
    print(
        f"[HB][train][B5B6] lag_enable={1 if bool((lag_cfg_runtime or {}).get('enabled', False)) else 0} "
        f"lag_bucket={1 if bool((lag_cfg_runtime or {}).get('load_bucket_enable', False)) else 0} "
        f"bc_mix={1 if bool((bc_cfg_runtime or {}).get('mix_enable', False) or len(bc_methods_runtime) > 1) else 0}",
        flush=True,
    )

    
    try:
        setattr(policy, "_selfcheck_enabled", bool(selfcheck_enabled))
        setattr(policy, "_selfcheck_every", int(10))
        setattr(policy, "_selfcheck_tol", float(1e-4))
    except Exception as exc:
        _warn_once("phasea_selfcheck_cfg", f"{type(exc).__name__}: {exc}")

    try:
        logp_scale = float(getattr(policy, "_logp_scale", 1.0))
        logp_mode_s = str(getattr(policy, "_logp_mode", "sum")).strip().lower()
        logp_is_mean = 1 if logp_mode_s == "mean" else 0
        policy_dist = "SquashedGaussian"
        print(
            f"[HB][train][P0] selfcheck_enabled={1 if selfcheck_enabled else 0} "
            f"policy_dist={policy_dist} uses_tanh_squash=True uses_clamp=False "
            f"logp_mode={logp_mode_s} logp_is_mean={logp_is_mean} logp_scale={logp_scale:.6g} "
            f"act_dim_effective={int(act_dim_effective)}",
            flush=True,
        )
        if not bool(selfcheck_enabled):
            print("[HB][train][P0] SELF-CHECK OFFrun", flush=True)
    except Exception as exc:
        _warn_once("phasea_selfcheck_log", f"{type(exc).__name__}: {exc}")

    # =========================
    
    # =========================
    misc_cfg = train_cfg.get("misc", {}) or {}
    resume_path = misc_cfg.get("resume_path", None)
    resume_path = str(resume_path).strip() if resume_path is not None else ""
    require_resume = bool(misc_cfg.get("require_resume", False))
    if require_resume and (not resume_path):
        raise RuntimeError(
            "misc.require_resume=1  misc.resume_path "
            " checkpoint"
        )
    resume_ckpt_step = 0
    resume_use_ckpt_step_offset = bool(train_params.get("resume_use_ckpt_step_offset", True))
    if resume_path:
        if require_resume and (not Path(resume_path).exists()):
            raise FileNotFoundError(f"misc.require_resume=1  checkpoint {resume_path}")
        try:
            
            ckpt_obj = _torch_load_ckpt(resume_path, map_location=device, prefer_weights_only=True)
            if isinstance(ckpt_obj, dict) and ("policy" in ckpt_obj):
                policy.load_state_dict(ckpt_obj["policy"])
                if "optimizer" in ckpt_obj:
                    try:
                        _optimizer_load_state(optimizer, ckpt_obj["optimizer"])
                    except Exception as exc:
                        
                        _warn_once("resume_optimizer_state", f"optimizerpolicy{type(exc).__name__}: {exc}")
                
                try:
                    cands: List[Any] = []
                    if "step" in ckpt_obj:
                        cands.append(ckpt_obj.get("step"))
                    if "global_step" in ckpt_obj:
                        cands.append(ckpt_obj.get("global_step"))
                    meta_obj = ckpt_obj.get("meta", None)
                    if isinstance(meta_obj, dict):
                        if "global_step" in meta_obj:
                            cands.append(meta_obj.get("global_step"))
                        if "step" in meta_obj:
                            cands.append(meta_obj.get("step"))
                    parsed = []
                    for vv in cands:
                        try:
                            parsed.append(int(float(vv)))
                        except Exception:
                            continue
                    if parsed:
                        resume_ckpt_step = int(max(max(parsed), 0))
                except Exception:
                    resume_ckpt_step = 0
                print(f"[HB][train][resume] loaded policy ckpt={resume_path}", flush=True)
                if resume_ckpt_step > 0:
                    print(
                        f"[HB][train][resume] ckpt_global_step={int(resume_ckpt_step)} "
                        f"use_step_offset={1 if resume_use_ckpt_step_offset else 0}",
                        flush=True,
                    )
            else:
                print(f"[HB][train][resume][WARN] ckpt{resume_path}", flush=True)

            
            _try_load_obs_normalizer_from_ckpt(
                obs_normalizer=obs_normalizer,
                ckpt_path=resume_path,
                log_tag="resume",
            )
        except Exception as e:
            print(f"[HB][train][resume][WARN] resume{type(e).__name__}: {e}", flush=True)

    
    policy_init_path = misc_cfg.get("policy_init_path", None)
    policy_init_path = str(policy_init_path).strip() if policy_init_path is not None else ""
    if resume_path and policy_init_path:
        raise RuntimeError(
            "misc.resume_path  misc.policy_init_path "
            ""
        )
    if policy_init_path:
        if not Path(policy_init_path).exists():
            raise FileNotFoundError(f"misc.policy_init_path {policy_init_path}")
        try:
            init_obj = _torch_load_ckpt(policy_init_path, map_location=device, prefer_weights_only=True)
            if isinstance(init_obj, dict) and ("policy" in init_obj):
                policy.load_state_dict(init_obj["policy"])
                print(f"[HB][train][policy_init] loaded policy ckpt={policy_init_path}", flush=True)
                _try_load_obs_normalizer_from_ckpt(
                    obs_normalizer=obs_normalizer,
                    ckpt_path=policy_init_path,
                    log_tag="policy_init",
                )
            else:
                raise RuntimeError(f"checkpoint {policy_init_path}")
        except Exception as exc:
            raise RuntimeError(
                f"[HB][train][FATAL] policy_init {type(exc).__name__}: {exc}"
            ) from exc

    # -----------------------------
    
    # -----------------------------
    
    
    
    pre_cfg = train_cfg.get("pretrain", {}) or {}
    bc_warm_cfg = train_params.get("bc_warmstart", {}) or {}
    if not isinstance(bc_warm_cfg, dict):
        bc_warm_cfg = {}
    bc_warm_enable = bool(bc_warm_cfg.get("enable", False))
    bc_warm_obs: Optional[np.ndarray] = None
    bc_warm_act: Optional[np.ndarray] = None
    bc_warm_cost: Optional[np.ndarray] = None

    def _bc_cost_surrogate_from_action(act: np.ndarray) -> float:
        """
         `_bc_action_cost_surrogate` 
        """
        a = np.asarray(act, dtype=np.float32).reshape(-1)
        if a.size <= 0:
            return 0.0
        d_dim = int(max(0, min(2 * int(getattr(env_cfg, "M", 0)), int(a.size))))
        move_term = float(np.mean(np.abs(a[:d_dim]))) if d_dim > 0 else 0.0
        score_term = 0.0
        if int(a.size) > d_dim:
            score = np.asarray(a[d_dim:], dtype=np.float32)
            score_prob = 1.0 / (1.0 + np.exp(-score / 0.5))
            score_term = float(np.mean(score_prob))
        out = float(move_term + 0.35 * score_term)
        cost_scale = float(max(float(bc_warm_cfg.get("cost_scale", 1.0)), 1e-9))
        cost_clip = float(max(float(bc_warm_cfg.get("cost_clip", 2.0)), 1e-6))
        out = float(out / cost_scale)
        if cost_clip > 0.0:
            out = float(np.clip(out, 0.0, cost_clip))
        return out

    def _resolve_bc_teacher_mix(
        *,
        cfg_block: Dict[str, Any],
        fallback_method: str,
    ) -> Tuple[List[str], List[float], str]:
        """
        B6BC(, , )
        """
        allowed = {"balanced", "greedy_delay", "greedy_energy", "always_comp", "never_comp"}
        fm = str(fallback_method or "balanced").strip().lower()
        if fm not in allowed:
            fm = "balanced"

        mix_enable = bool(cfg_block.get("mix_enable", False))
        method_token = str(cfg_block.get("method", fm)).strip().lower()
        raw_methods = cfg_block.get("methods", None)
        if (raw_methods is None) and method_token.startswith("mix:"):
            raw_methods = method_token.split(":", 1)[1]
            method_token = "mixed"

        methods: List[str] = []
        for m in _parse_list(raw_methods, str):
            mm = str(m).strip().lower()
            if mm in allowed:
                methods.append(mm)

        if method_token in allowed and (not bool(mix_enable)):
            methods = [method_token]
        elif (method_token in ("mixed", "mix", "hybrid")) or bool(mix_enable):
            if not methods:
                methods = ["balanced", "greedy_delay", "greedy_energy"]
        elif not methods:
            methods = [fm]

        
        methods = list(dict.fromkeys([str(m).strip().lower() for m in methods if str(m).strip()]))
        methods = [m for m in methods if m in allowed]
        if not methods:
            methods = [fm]

        raw_w = _parse_list(cfg_block.get("weights", None), float)
        if len(raw_w) != len(methods):
            probs = [1.0 / float(max(len(methods), 1))] * len(methods)
        else:
            ww = [float(max(float(x), 1e-9)) for x in raw_w]
            ss = float(sum(ww))
            probs = [float(x / ss) for x in ww] if ss > 1e-12 else [1.0 / float(max(len(ww), 1))] * len(ww)

        m_desc = ",".join(methods)
        p_desc = ",".join([f"{float(p):.3f}" for p in probs])
        desc = f"methods=[{m_desc}] probs=[{p_desc}]"
        return methods, probs, desc

    def _collect_bc_dataset(
        method: str,
        steps: int,
        seed_offset: int,
        tag: str,
        teacher_methods: Optional[List[str]] = None,
        teacher_probs: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ teacher obs, act_target, cost_target"""
        from src.algos.baseline import baseline_action  

        n_steps = int(max(int(steps), 1))
        env_bc = CompRISEnvJoint(env_cfg, seed=seed_base + int(seed_offset))
        obs = env_bc.reset()

        m_count = int(getattr(env_cfg, "M", 0))
        delta_scale = float(getattr(env_cfg, "Vmax", 1.0)) * float(getattr(env_cfg, "dt", 1.0))

        obs_buf = np.zeros((n_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_steps, act_dim), dtype=np.float32)
        cost_buf = np.zeros((n_steps,), dtype=np.float32)
        methods = [str(method).strip().lower()] if not teacher_methods else [str(m).strip().lower() for m in teacher_methods]
        methods = [m for m in methods if m]
        if not methods:
            methods = ["balanced"]
        probs = list(teacher_probs or [])
        if len(probs) != len(methods):
            probs = [1.0 / float(max(len(methods), 1))] * len(methods)
        else:
            ww = [float(max(float(x), 1e-9)) for x in probs]
            ss = float(sum(ww))
            probs = [float(x / ss) for x in ww] if ss > 1e-12 else [1.0 / float(max(len(ww), 1))] * len(ww)
        mix_rng = np.random.RandomState(seed_base + int(seed_offset) + 1317)
        method_counts: Dict[str, int] = {m: 0 for m in methods}

        t0 = time.time()
        for i in range(n_steps):
            if obs_normalizer is not None:
                obs_n = obs_normalizer.normalize(obs, update=True)
            else:
                obs_n = np.asarray(obs, dtype=np.float32)

            if len(methods) <= 1:
                method_now = str(methods[0])
            else:
                mi = int(mix_rng.choice(np.arange(len(methods)), p=np.asarray(probs, dtype=np.float64)))
                method_now = str(methods[mi])
            method_counts[method_now] = int(method_counts.get(method_now, 0) + 1)

            act_env = baseline_action(env_bc, mode=method_now)
            act_env = np.asarray(act_env, dtype=np.float32).reshape(-1)

            
            act_tgt = act_env.copy()
            if m_count > 0 and delta_scale > 1e-9:
                delta_part = act_tgt[: 2 * m_count]
                if float(np.nanmax(np.abs(delta_part))) > 1.5:
                    act_tgt[: 2 * m_count] = np.clip(delta_part / float(delta_scale), -1.0, 1.0)
                else:
                    act_tgt[: 2 * m_count] = np.clip(delta_part, -1.0, 1.0)
            if act_tgt.size > 2 * m_count:
                act_tgt[2 * m_count:] = np.tanh(act_tgt[2 * m_count:])

            obs_buf[i] = np.asarray(obs_n, dtype=np.float32).reshape(-1)
            act_buf[i] = np.asarray(act_tgt, dtype=np.float32).reshape(-1)
            cost_buf[i] = float(_bc_cost_surrogate_from_action(act_tgt))

            obs, _r, done, _info = env_bc.step(act_env)
            if bool(done):
                obs = env_bc.reset()

            if (i + 1) % 5000 == 0 or (i + 1) == n_steps:
                elapsed = float(time.time() - t0)
                eta = elapsed * (float(n_steps) / float(max(i + 1, 1)) - 1.0) if (i + 1) > 0 else 0.0
                m_desc = ",".join(methods)
                p_desc = ",".join([f"{float(pp):.3f}" for pp in probs])
                print(
                    f"[HB][{tag}][BC] collect {i+1}/{n_steps} "
                    f"t={fmt_hms(elapsed)} ETA={fmt_hms(eta)} methods=[{m_desc}] probs=[{p_desc}]",
                    flush=True,
                )

        cnt_desc = ",".join([f"{m}:{int(method_counts.get(m, 0))}" for m in methods])
        print(f"[HB][{tag}][BC] teacher_counts={cnt_desc}", flush=True)
        return obs_buf, act_buf, cost_buf

    def _validate_bc_dataset(
        obs_arr: Optional[np.ndarray],
        act_arr: Optional[np.ndarray],
        cost_arr: Optional[np.ndarray],
        *,
        tag: str,
    ) -> None:
        """
        BC
        """
        if obs_arr is None or act_arr is None:
            raise ValueError(f"{tag} BC")
        obs_np = np.asarray(obs_arr, dtype=np.float32)
        act_np = np.asarray(act_arr, dtype=np.float32)
        if obs_np.ndim != 2 or act_np.ndim != 2:
            raise ValueError(f"{tag} BCobs={obs_np.shape}, act={act_np.shape}")
        if int(obs_np.shape[0]) <= 0 or int(act_np.shape[0]) <= 0:
            raise ValueError(f"{tag} BCobs_n={obs_np.shape[0]}, act_n={act_np.shape[0]}")
        if int(obs_np.shape[0]) != int(act_np.shape[0]):
            raise ValueError(f"{tag} BCobs_n={obs_np.shape[0]}, act_n={act_np.shape[0]}")
        if not np.all(np.isfinite(obs_np)):
            raise ValueError(f"{tag} BCNaN/Inf")
        if not np.all(np.isfinite(act_np)):
            raise ValueError(f"{tag} BCNaN/Inf")
        if cost_arr is not None:
            cost_np = np.asarray(cost_arr, dtype=np.float32).reshape(-1)
            if int(cost_np.size) != int(obs_np.shape[0]):
                raise ValueError(f"{tag} BCcost_n={cost_np.size}, obs_n={obs_np.shape[0]}")
            if not np.all(np.isfinite(cost_np)):
                raise ValueError(f"{tag} BCNaN/Inf")

    
    if bool(pre_cfg.get("enabled", False)) and (not resume_path):
        try:
            method = str(pre_cfg.get("method", "balanced")).strip().lower()
            bc_steps = int(pre_cfg.get("steps", 20000))
            bc_epochs = int(pre_cfg.get("epochs", 3))
            bc_batch = int(pre_cfg.get("batch_size", 1024))
            pre_methods, pre_probs, pre_desc = _resolve_bc_teacher_mix(cfg_block=pre_cfg, fallback_method=method)
            bc_warm_obs, bc_warm_act, bc_warm_cost = _collect_bc_dataset(
                method=method,
                steps=bc_steps,
                seed_offset=9999,
                tag="pretrain",
                teacher_methods=pre_methods,
                teacher_probs=pre_probs,
            )
            _validate_bc_dataset(bc_warm_obs, bc_warm_act, bc_warm_cost, tag="pretrain")

            policy.train()
            rng = np.random.RandomState(seed_base + 2026)
            for ep in range(int(bc_epochs)):
                perm = rng.permutation(int(max(bc_warm_obs.shape[0], 1)))
                loss_ema = None
                t1 = time.time()
                for j in range(0, int(bc_warm_obs.shape[0]), int(max(bc_batch, 1))):
                    idx = perm[j : j + int(max(bc_batch, 1))]
                    obs_np = np.asarray(bc_warm_obs[idx], dtype=np.float32)
                    obs_t = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
                    act_t = torch.as_tensor(bc_warm_act[idx], device=device, dtype=torch.float32)
                    (mu, _sigma), _ = policy.actor(obs_t, info={"obs_raw": obs_np})
                    loss = torch.mean((mu - act_t) ** 2)

                    actor_optim = _optimizer_actor_only(optimizer)
                    actor_optim.zero_grad()
                    loss.backward()
                    actor_optim.step()

                    lv = float(loss.detach().cpu().item())
                    loss_ema = lv if loss_ema is None else (0.9 * float(loss_ema) + 0.1 * lv)

                print(
                    f"[HB][pretrain][BC] epoch {ep+1}/{bc_epochs} loss_ema={float(loss_ema or 0.0):.6f} "
                    f"t={fmt_hms(time.time() - t1)}",
                    flush=True,
                )

            print(
                f"[HB][pretrain][BC] done. steps={bc_steps} epochs={bc_epochs} method={method} {pre_desc}",
                flush=True,
            )
        except Exception as e:
            print(f"[HB][pretrain][BC][WARN] skipped: {type(e).__name__}: {e}", flush=True)

    
    if bc_warm_enable and (bc_warm_obs is None or bc_warm_act is None):
        try:
            ws_method = str(bc_warm_cfg.get("method", pre_cfg.get("method", "balanced"))).strip().lower()
            ws_steps = int(bc_warm_cfg.get("dataset_steps", pre_cfg.get("steps", 12000)))
            ws_methods, ws_probs, ws_desc = _resolve_bc_teacher_mix(cfg_block=bc_warm_cfg, fallback_method=ws_method)
            bc_warm_obs, bc_warm_act, bc_warm_cost = _collect_bc_dataset(
                method=ws_method,
                steps=ws_steps,
                seed_offset=19999,
                tag="warmstart",
                teacher_methods=ws_methods,
                teacher_probs=ws_probs,
            )
            _validate_bc_dataset(bc_warm_obs, bc_warm_act, bc_warm_cost, tag="warmstart")
            print(
                f"[HB][train][BC] warmstart dataset ready: n={int(bc_warm_obs.shape[0])} method={ws_method} {ws_desc}",
                flush=True,
            )
        except Exception as e:
            bc_warm_obs, bc_warm_act, bc_warm_cost = None, None, None
            print(f"[HB][train][BC][WARN] warmstart dataset skipped: {type(e).__name__}: {e}", flush=True)
    if bc_warm_enable and (bc_warm_obs is None or bc_warm_act is None):
        if bool(bc_warm_cfg.get("require_dataset", True)):
            raise RuntimeError(
                " train.bc_warmstart.require_dataset=true"
                "BC"
            )

    # preprocess_fn
    reward_override = str(train_params.get("reward_override", "env"))
    lag_cfg = train_params.get("lagrange_vio", {}) or {}

    
    paper_reward_cfg = train_params.get("paper_reward", {}) or {}
    paper_reward_map = str(paper_reward_cfg.get("map", "tanh")).strip().lower()
    paper_reward_bias = float(paper_reward_cfg.get("bias", 0.0))
    paper_reward_scale = float(paper_reward_cfg.get("scale", 1.0))
    paper_reward_tanh_s = float(paper_reward_cfg.get("tanh_s", 0.20))
    paper_reward_clip_c = float(paper_reward_cfg.get("clip_c", 5.0))
    
    paper_improve_mode = str(paper_reward_cfg.get("improve_mode", "paper_improve")).strip().lower()
    rel_improve_eps = float(paper_reward_cfg.get("rel_improve_eps", 1e-3))
    improve_z_eps = float(paper_reward_cfg.get("improve_z_eps", 1e-6))
    paper_vio_gating_power = float(paper_reward_cfg.get("vio_gating_power", 1.0))
    paper_vio_penalty = float(paper_reward_cfg.get("vio_penalty", 0.0))

    preprocess = CollectorPreprocess(
        obs_normalizer=obs_normalizer,
        update_obs=True,
        reward_override=reward_override,
        paper_reward_map=str(paper_reward_map),
        paper_reward_bias=float(paper_reward_bias),
        paper_reward_scale=float(paper_reward_scale),
        paper_reward_tanh_s=float(paper_reward_tanh_s),
        paper_reward_clip_c=float(paper_reward_clip_c),
        improve_mode=str(paper_improve_mode),
        rel_improve_eps=float(rel_improve_eps),
        improve_z_eps=float(improve_z_eps),
        paper_vio_gating_power=float(paper_vio_gating_power),
        paper_vio_penalty=float(paper_vio_penalty),
        reward_clip_range=float(getattr(env_cfg, "reward_clip_range", 0.0)),
        
        vio_scale=1.0,
        lagrange_enabled=bool(lag_cfg.get("enabled", False)),
        lagrange_init=float(lag_cfg.get("init", 0.0)),
        
        enable_comp_ris_shaping=bool(train_params.get("reward_shaping", False)),
        comp_weight=float(train_params.get("comp_weight", 0.0)),
        ris_weight=float(train_params.get("ris_weight", 0.0)),
        shaping_ratio=0.10,
        # H12 v2: per-step reward normalization
        reward_step_normalize=bool(train_params.get("reward_step_normalize", False)),
    )

    
    
    rd = str(getattr(env_cfg, "reward_design", "")).strip().lower()
    if str(reward_override).strip().lower() == "env":
        main_curve_desc = "reward_total_ep(env_return)"
    else:
        main_curve_desc = "paper_return_ep(=-paper_cost_ep)"

    print(
        f"[HB][train][E0] MainCurve={main_curve_desc} | TrainReward=train_reward_mean reward_override={reward_override} "
        f"paper_reward(map={paper_reward_map},A={paper_reward_bias},B={paper_reward_scale},s={paper_reward_tanh_s},"
        f"gate_p={paper_vio_gating_power},vio_penalty={paper_vio_penalty}) "
        f"improve_mode={paper_improve_mode}(rel_eps={rel_improve_eps:g},z_eps={improve_z_eps:g}) "
        f"reward_clip_range={float(getattr(env_cfg, 'reward_clip_range', 0.0))}"
        f" reward_step_normalize={bool(train_params.get('reward_step_normalize', False))}",
        flush=True,
    )

    buffer = VectorReplayBuffer(total_size=int(rollout_steps), buffer_num=int(num_train_envs))
    collector = Collector(policy, venv, buffer, preprocess_fn=preprocess)

    # metrics csv
    metrics_csv_path = Path(run_dir) / "train_metrics.csv"

    
    resume_existing_metrics = bool(resume_path) and metrics_csv_path.exists() and metrics_csv_path.stat().st_size > 0
    if (not resume_existing_metrics) and bool(resume_use_ckpt_step_offset) and int(resume_ckpt_step) > int(total_steps):
        raise RuntimeError(
            "[HB][train][FATAL] "
            f"resume_ckpt_step={int(resume_ckpt_step)} > total_steps={int(total_steps)}\n"
            " resume_use_ckpt_step_offset=1"
            " min_lrclipfrac  0\n"
            " train.resume_use_ckpt_step_offset  false"
            " total_steps "
        )
    resume_step_offset_effective = int(resume_ckpt_step) if (resume_use_ckpt_step_offset and int(resume_ckpt_step) > 0) else 0
    if resume_existing_metrics:
        
        resume_step_offset_effective = 0
    resume_env_step = 0
    resume_hist = {}
    resume_fieldnames: Optional[List[str]] = None
    if resume_existing_metrics:
        try:
            with open(metrics_csv_path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                resume_fieldnames = list(r.fieldnames) if r.fieldnames else []
                reward_history: List[float] = []
                policy_loss_history: List[float] = []
                value_loss_history: List[float] = []
                entropy_history: List[float] = []
                kl_div_history: List[float] = []
                clipfrac_history: List[float] = []
                explained_variance_history: List[float] = []
                steps_history: List[int] = []
                episodes_history: List[int] = []
                episodes_dense_history: List[int] = []
                reward_dense_history: List[float] = []
                policy_loss_dense_history: List[float] = []
                value_loss_dense_history: List[float] = []
                entropy_dense_history: List[float] = []
                kl_dense_history: List[float] = []
                clipfrac_dense_history: List[float] = []
                explained_variance_dense_history: List[float] = []
                lr_dense_history: List[float] = []
                cost_history: List[float] = []
                vio_history: List[float] = []
                improve_history: List[float] = []

                eval_steps_history: List[int] = []
                eval_episodes_history: List[int] = []
                eval_reward_history: List[float] = []
                eval_reward_ci_history: List[float] = []
                eval_paper_cost_history: List[float] = []
                eval_paper_cost_std_history: List[float] = []
                eval_paper_cost_ci_history: List[float] = []
                # PhaseEpaper_return_ep := -paper_cost_ep
                eval_paper_return_history: List[float] = []
                eval_paper_return_ci_history: List[float] = []
                eval_n_history: List[int] = []
                eval_vio_history: List[float] = []
                eval_vio_any_frac_history: List[float] = []
                eval_vio_any_frac_ci_history: List[float] = []
                eval_improve_history: List[float] = []
                eval_det_paper_cost_dense_history: List[float] = []
                eval_det_paper_cost_ci_dense_history: List[float] = []
                eval_det_vio_any_frac_dense_history: List[float] = []
                eval_det_improve_dense_history: List[float] = []
                latest_eval_det_paper_cost = float("nan")
                latest_eval_det_paper_cost_ci = float("nan")
                latest_eval_det_vio_any_frac = float("nan")
                latest_eval_det_improve = float("nan")

                train_episode_cum = 0
                prev_train_episode = 0
                for row in r:
                    try:
                        st = int(float(row.get("global_step", 0.0)))
                    except Exception:
                        continue
                    steps_history.append(int(st))
                    resume_env_step = int(st)

                    def _gf(k: str, default: float = 0.0) -> float:
                        try:
                            return float(row.get(k, default))
                        except Exception:
                            return float(default)

                    reward_now = _gf("train_reward_mean", _gf("train_reward_step_mean", 0.0))
                    policy_loss_now = _gf("policy_loss", 0.0)
                    value_loss_now = _gf("value_loss", 0.0)
                    entropy_now = _gf("entropy_per_dim", _gf("entropy", 0.0))
                    kl_now = _gf("kl_per_dim", 0.0)
                    reward_history.append(float(reward_now))
                    policy_loss_history.append(float(policy_loss_now))
                    value_loss_history.append(float(value_loss_now))
                    entropy_history.append(float(entropy_now))
                    kl_div_history.append(float(kl_now))
                    clipfrac_now = _gf("clipfrac", _gf("clipfrac_mean", 0.0))
                    explained_variance_now = _gf("explained_variance", 0.0)
                    lr_now = _gf("lr", 0.0)
                    clipfrac_history.append(float(clipfrac_now))
                    explained_variance_history.append(float(explained_variance_now))
                    cost_history.append(_gf("cost_mean", 0.0))
                    vio_history.append(_gf("vio_mean", 0.0))
                    improve_history.append(_gf("improve_mean", 0.0))

                    
                    te_v = None
                    try:
                        te_raw = row.get("train_episode", None)
                        if te_raw is not None and str(te_raw).strip() != "":
                            te_v = int(float(te_raw))
                    except Exception:
                        te_v = None
                    if te_v is not None and int(te_v) >= 0:
                        train_episode_cum = int(te_v)
                    else:
                        try:
                            train_episode_cum += int(max(_gf("train_n_ep", 0.0), 0.0))
                        except Exception:
                            pass
                    episodes_history.append(int(train_episode_cum))
                    n_new_ep = int(max(int(train_episode_cum) - int(prev_train_episode), 0))
                    if n_new_ep > 0:
                        ep_start = int(prev_train_episode) + 1
                        ep_end = int(prev_train_episode) + int(n_new_ep)
                        episodes_dense_history.extend(list(range(ep_start, ep_end + 1)))
                        reward_dense_history.extend([float(reward_now)] * n_new_ep)
                        policy_loss_dense_history.extend([float(policy_loss_now)] * n_new_ep)
                        value_loss_dense_history.extend([float(value_loss_now)] * n_new_ep)
                        entropy_dense_history.extend([float(entropy_now)] * n_new_ep)
                        kl_dense_history.extend([float(kl_now)] * n_new_ep)
                        clipfrac_dense_history.extend([float(clipfrac_now)] * n_new_ep)
                        explained_variance_dense_history.extend([float(explained_variance_now)] * n_new_ep)
                        lr_dense_history.extend([float(lr_now)] * n_new_ep)
                    prev_train_episode = int(train_episode_cum)

                    
                    n_eval = int(_gf("eval_det_n", 0.0))
                    if n_eval > 0:
                        eval_steps_history.append(int(st))
                        
                        ete_v = None
                        try:
                            ete_raw = row.get("eval_train_episode", None)
                            if ete_raw is not None and str(ete_raw).strip() != "":
                                ete_v = int(float(ete_raw))
                        except Exception:
                            ete_v = None
                        if ete_v is None or int(ete_v) < 0:
                            ete_v = int(train_episode_cum)
                        eval_episodes_history.append(int(ete_v))
                        eval_reward_history.append(_gf("eval_det_return_mean", 0.0))
                        eval_reward_ci_history.append(_gf("eval_det_return_ci", 0.0))
                        ec_mu = _gf("eval_det_paper_cost_mean", 0.0)
                        eval_paper_cost_history.append(float(ec_mu))
                        eval_paper_cost_std_history.append(_gf("eval_det_paper_cost_std", 0.0))
                        ec_ci = _gf("eval_det_paper_cost_ci", 0.0)
                        eval_paper_cost_ci_history.append(float(ec_ci))
                        
                        er_mu = _gf("eval_det_paper_return_mean", float("nan"))
                        if not np.isfinite(float(er_mu)):
                            er_mu = -float(ec_mu)
                        er_ci = _gf("eval_det_paper_return_ci", float("nan"))
                        if not np.isfinite(float(er_ci)):
                            er_ci = float(ec_ci)
                        eval_paper_return_history.append(float(er_mu))
                        eval_paper_return_ci_history.append(float(er_ci))
                        eval_n_history.append(int(n_eval))
                        eval_vio_history.append(_gf("eval_det_vio_mean", 0.0))
                        eval_vio_any_frac_now = _gf("eval_det_vio_any_frac_mean", 0.0)
                        eval_vio_any_frac_history.append(float(eval_vio_any_frac_now))
                        eval_vio_any_frac_ci_history.append(_gf("eval_det_vio_any_frac_ci", 0.0))
                        eval_improve_now = _gf("eval_det_imp_paper_cost_mean", 0.0)
                        eval_improve_history.append(float(eval_improve_now))
                        latest_eval_det_paper_cost = float(ec_mu)
                        latest_eval_det_paper_cost_ci = float(ec_ci)
                        latest_eval_det_vio_any_frac = float(eval_vio_any_frac_now)
                        latest_eval_det_improve = float(eval_improve_now)

                    if n_new_ep > 0:
                        eval_det_paper_cost_dense_history.extend([float(latest_eval_det_paper_cost)] * n_new_ep)
                        eval_det_paper_cost_ci_dense_history.extend([float(latest_eval_det_paper_cost_ci)] * n_new_ep)
                        eval_det_vio_any_frac_dense_history.extend([float(latest_eval_det_vio_any_frac)] * n_new_ep)
                        eval_det_improve_dense_history.extend([float(latest_eval_det_improve)] * n_new_ep)

            resume_hist = {
                "reward_history": reward_history,
                "policy_loss_history": policy_loss_history,
                "value_loss_history": value_loss_history,
                "entropy_history": entropy_history,
                "kl_div_history": kl_div_history,
                "clipfrac_history": clipfrac_history,
                "explained_variance_history": explained_variance_history,
                "steps_history": steps_history,
                "episodes_history": episodes_history,
                "episodes_dense_history": episodes_dense_history,
                "reward_dense_history": reward_dense_history,
                "policy_loss_dense_history": policy_loss_dense_history,
                "value_loss_dense_history": value_loss_dense_history,
                "entropy_dense_history": entropy_dense_history,
                "kl_dense_history": kl_dense_history,
                "clipfrac_dense_history": clipfrac_dense_history,
                "explained_variance_dense_history": explained_variance_dense_history,
                "lr_dense_history": lr_dense_history,
                "resume_train_episode": int(train_episode_cum),
                "cost_history": cost_history,
                "vio_history": vio_history,
                "improve_history": improve_history,
                "eval_steps_history": eval_steps_history,
                "eval_episodes_history": eval_episodes_history,
                "eval_reward_history": eval_reward_history,
                "eval_reward_ci_history": eval_reward_ci_history,
                "eval_paper_cost_history": eval_paper_cost_history,
                "eval_paper_cost_std_history": eval_paper_cost_std_history,
                "eval_paper_cost_ci_history": eval_paper_cost_ci_history,
                "eval_paper_return_history": eval_paper_return_history,
                "eval_paper_return_ci_history": eval_paper_return_ci_history,
                "eval_n_history": eval_n_history,
                "eval_vio_history": eval_vio_history,
                "eval_vio_any_frac_history": eval_vio_any_frac_history,
                "eval_vio_any_frac_ci_history": eval_vio_any_frac_ci_history,
                "eval_improve_history": eval_improve_history,
                "eval_det_paper_cost_dense_history": eval_det_paper_cost_dense_history,
                "eval_det_paper_cost_ci_dense_history": eval_det_paper_cost_ci_dense_history,
                "eval_det_vio_any_frac_dense_history": eval_det_vio_any_frac_dense_history,
                "eval_det_improve_dense_history": eval_det_improve_dense_history,
            }
            print(f"[HB][train][resume] append metrics.csv (resume_env_step={int(resume_env_step)})", flush=True)
        except Exception as e:
            resume_existing_metrics = False
            resume_env_step = 0
            resume_hist = {}
            resume_fieldnames = None
            print(f"[HB][train][resume][WARN] train_metrics{type(e).__name__}: {e}", flush=True)

    metrics_fieldnames = [
        
        "act_dim",
        "logp_mode",
        "logp_scale",
        "logp_act_dim",
        "logp_is_mean",
        "selfcheck_enabled",
        "uses_tanh_squash",
        "uses_clamp",
        "policy_dist",
        
        "improve_mode",
        
        "scores_mode",
        "scores_mode_fixed",
        "act_dim_effective",
        "train_stage",
        "trainable_parts",
        "scores_fixed_mean",
        "scores_fixed_std",
        
        "action_oor_frac",
        "action_oor_any_frac",
        "action_sat_frac",
        "action_sat_any_frac",
        "global_step",
        "train_episode",
        "update_count",
        "train_reward_mean",
        "train_reward_std",
        "train_reward_step_mean",
        "train_reward_step_std",
        "reward_total_step",
        "paper_cost_step",
        
        "ep_return_train_p05",
        "ep_return_train_p50",
        "ep_return_train_p95",
        
        "reward_train_mean",          
        "reward_train_step_mean",     
        "reward_train_raw_p05",       # = reward_raw_p05
        "reward_train_raw_p50",       # = reward_raw_p50
        "reward_train_raw_p95",       # = reward_raw_p95
        "reward_clip_hit_frac",       # = clip_hit_frac
        "paper_cost_mean",            # = cost_mean
        "vio_any_frac",               # = violation_any_frac
        "clipfrac_mean",              
        
        
        
        
        "violation_any_frac",
        "reward_raw_mean",
        "reward_raw_std",
        "reward_raw_p05",
        "reward_raw_p50",
        "reward_raw_p95",
        "clip_hit_frac",
        
        "train_len_mean",
        "train_len_std",
        "train_n_ep",
        "returns_mean",
        "returns_std",
        "value_bias_proxy",
        "policy_loss",
        "value_loss",
        "vf_clip_hit_frac",
        "bc_warmstart_alpha",
        "bc_warmstart_cost_alpha",
        "bc_warmstart_loss_raw",
        "bc_warmstart_cost_loss_raw",
        "bc_warmstart_loss_weighted",
        "bc_warmstart_cost_loss_weighted",
        "bc_warmstart_updates",
        "tail_stabilize_active",
        "head_tune_enable",
        "a1_head_clip_used_ratio",
        "a1_head_clip_ready",
        "a1_head_ent_used_ratio",
        "ppo_eps_clip_current",
        "ppo_eps_clip_delta_current",
        "ppo_eps_clip_score_current",
        "ppo_value_clip_current",
        "ppo_target_kl_current",
        "explained_variance",
        "entropy",
        "entropy_sum",
        "entropy_per_dim",
        "kl",
        "kl_per_dim",
        "clipfrac",
        "clipfrac_per_dim",
        "lr",
        "log_std_mean",
        "sigma_mean",
        "sigma_delta_mean",
        "sigma_score_mean",
        "exploration_std_scale",
        "exploration_std_scale_delta",
        "exploration_std_scale_score",
        "ent_coef",
        "ent_coef_delta",
        "ent_coef_score",
        
        "eval_det_is_valid",
        "eval_train_episode",
        "eval_det_return_mean",
        "eval_det_return_std",
        "eval_det_return_ci",
        "eval_det_return_ci95",
        "eval_det_paper_cost_mean",
        "eval_det_paper_cost_std",
        "eval_det_paper_cost_ci",
        "eval_det_paper_cost_ci95",
        
        "eval_det_paper_return_mean",
        "eval_det_paper_return_std",
        "eval_det_paper_return_ci",
        "eval_det_paper_return_ci95",
        "eval_det_n",
        "eval_det_vio_mean",
        "eval_det_vio_any_frac_mean",
        "eval_det_vio_any_frac_std",
        "eval_det_vio_any_frac_ci",
        "eval_det_vio_any_frac_ci95",
        "eval_det_vio_metric_mean",
        "eval_det_coverage_fraction_mean",
        "eval_det_baseline_balanced_paper_cost_mean",
        "eval_det_baseline_greedy_delay_paper_cost_mean",
        "eval_det_baseline_greedy_energy_paper_cost_mean",
        "eval_det_imp_paper_cost_mean",
        "eval_det_imp_vs_balanced_paper_cost_mean",
        "eval_det_imp_vs_greedy_delay_paper_cost_mean",
        "eval_det_imp_vs_greedy_energy_paper_cost_mean",
        "traj_idle_ratio",
        "traj_boundary_stick_frac",
        "traj_user_nn_dist_norm",
        "traj_centroid_gap_norm",
        "traj_switchback_ratio",
        "eval_det_paper_cost_agg",
        "eval_det_reward_design",
        "eval_det_train_load",
        "eval_det_loads_primary",
        "eval_det_loads_count",
        "cost_mean",
        "vio_mean",
        
        "delta_scale_mean",
        "vio_wall_mean",
        "vio_wall_any_frac",
        
        "vio_metric_mean",
        "coverage_fraction_mean",
        "coverage_margin_mean",
        "collision_margin_mean",
        "baseline_balanced_paper_cost_mean",
        "baseline_greedy_delay_paper_cost_mean",
        "baseline_greedy_energy_paper_cost_mean",
        "paper_improve_mean",
        "paper_improve_vs_balanced_mean",
        "paper_improve_vs_greedy_delay_mean",
        "paper_improve_vs_greedy_energy_mean",
        "improve_mean",
        "env_total_mean",
        "lambda_v",
        "lag_bucket_id",
        "lag_target_current",
        "lag_eta_current",
        "lag_lambda_min_current",
        "lag_lambda_max_current",
        "comp_gain_ratio",
        "ris_gain_ratio",
        "shaping_term",
        "penalty_term",
        
        "comp_gain_ratio_mean",
        "ris_gain_ratio_mean",
        "env_shaping_mean",
        "total_shaping_mean",
        "total_shaping_all_mean",
        "comp_bonus_mean",
        "ris_bonus_mean",
        "comp_ris_ratio",
        "comp_enable_rate",
        "service_switch_count",
        "theta_entropy",
        "comp_score_temp_runtime",
        "z4_comp_meta_enable_mean",
        "z4_comp_meta_warm_scale_mean",
        "z4_comp_meta_thr_effective_mean",
        "z4_comp_meta_temp_effective_mean",
        "z4_comp_meta_ctrl_thr_raw_mean",
        "z4_comp_meta_ctrl_temp_raw_mean",
        "z4_comp_meta_thr_delta_ema_mean",
        "z4_comp_meta_temp_scale_ema_mean",
        "z4_assoc_stage_enable_mean",
        "z4_assoc_policy_mix_mean",
        "z4_assoc_train_frac_mean",
        "z4_assoc_stage_score_width_mean",
        "z4_assoc_stage_comp_rule_thr_mean",
        "theta_mode_effective",
        "theta_mode_effective_agent_frac",
        
        "reward_paper_step_mean",
        "reward_paper_abs_step_mean",
        "reward_paper_delta_step_mean",
        "reward_paper_adv_step_mean",
        "paper_delta_norm_step_mean",
        "paper_adv_norm_step_mean",
        "constraint_signal_step_mean",
        "lambda_v_effective_mean",
        "reward_constraint_step_mean",
        "reward_shaping_step_mean",
        "reward_misc_step_mean",
        "reward_proximity_step_mean",  
        "shaping_gate_mean",
        "beta_comp_t_mean",
        "beta_ris_t_mean",
        "beta_train_frac_mean",
        "z4_gap_raw_step_mean",
        "z4_gap_step_clip_mean",
        "z4_gap_step_ema_mean",
        "z4_gap_step_mean",
        "z4_gap_reward_step_mean",
        "z4_gap_lambda_input_mean",
        "z4_gap_lambda_sched_mean",
        "z4_gap_lambda_cap_mean",
        "z4_gap_lambda_eff_mean",
        "z4_gap_grad_ratio_eff_mean",
        "z4_gap_shape_guard_trigger_mean",
        "t6_main_weight_mean",
        "t6_potential_scale_mean",
        "t6_traj_scale_mean",
        "t6_anneal_ratio_mean",
        "t6_violation_ema_mean",
        "t6_pid_frozen_mean",
        "comp_bonus_raw_mean",
        "ris_bonus_raw_mean",
        "reward_paper_ep",
        "reward_shaping_ep",
        "reward_constraint_ep",
        "reward_misc_ep",
        "reward_contract_abs_err_mean",
        "reward_contract_abs_err_max",
        "phaseg_spearman_corr_roll",
        "phaseg_spearman_n_roll",
        "ep_sum_n",
        "reward_total_ep",
        "paper_cost_ep",
        "reward_total_ep_sum",
        "paper_cost_ep_sum",
        "reward_total_ep_agg",
        "paper_cost_ep_agg",
        "reward_clip_order",
        "phaseg_contract_version",
        "train_load",
        "train_load_current",
        "eval_load_primary",
        "eval_load_count",
        
        "meta_enabled",
        "meta_source",
        "meta_action_id",
        "meta_fallback_flag",
        "meta_fallback_rate",
        "meta_timeout_rate",
        "meta_json_invalid_rate",
        "meta_decision_latency_ms",
    ]

    
    if resume_existing_metrics and resume_fieldnames is not None:
        need_cols = {
            "train_episode",
            "eval_train_episode",
            "reward_total_step",
            "paper_cost_step",
            "reward_total_ep",
            "paper_cost_ep",
            "shaping_term",
            "penalty_term",
            "comp_gain_ratio",
            "ris_gain_ratio",
            "traj_idle_ratio",
            "traj_boundary_stick_frac",
            "traj_user_nn_dist_norm",
            "traj_centroid_gap_norm",
            "traj_switchback_ratio",
            "a1_head_clip_used_ratio",
            "t6_main_weight_mean",
            "t6_potential_scale_mean",
            "t6_traj_scale_mean",
            "t6_anneal_ratio_mean",
            "returns_mean",
            "returns_std",
            "value_bias_proxy",
            "comp_enable_rate",
            "service_switch_count",
            "theta_entropy",
            "comp_score_temp_runtime",
            "theta_mode_effective",
            "theta_mode_effective_agent_frac",
        }
        if not need_cols.issubset(set(resume_fieldnames)):
            try:
                import shutil

                bak_path = metrics_csv_path.parent / f"{metrics_csv_path.name}.bak"
                if not bak_path.exists():
                    shutil.copyfile(str(metrics_csv_path), str(bak_path))

                rows_raw: List[Dict[str, str]] = []
                with open(metrics_csv_path, "r", encoding="utf-8", newline="") as f:
                    rr = csv.DictReader(f)
                    for rrow in rr:
                        rows_raw.append({str(k): str(v) for k, v in (rrow or {}).items() if k is not None})

                train_episode_cum = 0
                out_rows: List[Dict[str, str]] = []
                for rrow in rows_raw:
                    def _get_float(key: str, default: float = float("nan")) -> float:
                        try:
                            return float(rrow.get(key, default))
                        except Exception:
                            return float(default)

                    te = None
                    try:
                        te_raw = rrow.get("train_episode", "")
                        if str(te_raw).strip() != "":
                            te = int(float(te_raw))
                    except Exception:
                        te = None

                    if te is not None and int(te) >= 0:
                        train_episode_cum = int(te)
                    else:
                        try:
                            train_episode_cum += int(max(_get_float("train_n_ep", 0.0), 0.0))
                        except Exception:
                            pass

                    is_valid = _get_float("eval_det_is_valid", float("nan"))
                    n_eval = _get_float("eval_det_n", 0.0)
                    cost_mu = _get_float("eval_det_paper_cost_mean", float("nan"))
                    is_eval = (np.isfinite(is_valid) and is_valid >= 0.5) or (np.isfinite(n_eval) and n_eval > 0.0) or np.isfinite(cost_mu)

                    rrow["train_episode"] = str(int(train_episode_cum))
                    rrow["eval_train_episode"] = str(int(train_episode_cum)) if bool(is_eval) else "nan"
                    out_rows.append(rrow)

                tmp_path = metrics_csv_path.parent / f"{metrics_csv_path.name}.tmp"
                with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                    ww = csv.DictWriter(f, fieldnames=metrics_fieldnames)
                    ww.writeheader()
                    for rrow in out_rows:
                        ww.writerow({k: rrow.get(k, "nan") for k in metrics_fieldnames})

                tmp_path.replace(metrics_csv_path)
                resume_fieldnames = list(metrics_fieldnames)
                print(
                    f"[HB][train][resume] train_metrics.csv train_episode/eval_train_episode -> {metrics_csv_path}",
                    flush=True,
                )
            except Exception as e:
                resume_existing_metrics = False
                resume_env_step = 0
                resume_hist = {}
                resume_fieldnames = None
                print(
                    f"[HB][train][resume][WARN] train_metrics.csv{type(e).__name__}: {e}",
                    flush=True,
                )
    metrics_csv_f = open(metrics_csv_path, "a" if resume_existing_metrics else "w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_csv_f, fieldnames=metrics_fieldnames)
    if not resume_existing_metrics:
        metrics_writer.writeheader()
    metrics_csv_f.flush()
    atexit.register(lambda: metrics_csv_f.close())

    eval_cfg = train_cfg.get("eval", {}) or {}
    env_cfg_eval = dc_replace(env_cfg, association_mode="hard")
    if int(resume_step_offset_effective) > 0:
        print(
            f"[HB][train][resume] warm_start_step_offset={int(resume_step_offset_effective)} "
            f"(schedule+curriculum)",
            flush=True,
        )

    trainer = CoMPOnpolicyTrainer(
        run_dir=run_dir,
        policy=policy,
        optimizer=optimizer,
        train_collector=collector,
        preprocess=preprocess,
        env_cfg_eval=env_cfg_eval,
        obs_normalizer=obs_normalizer,
        train_params=train_params,
        eval_cfg=eval_cfg,
        meta_controller=meta_controller,
        metrics_writer=metrics_writer,
        metrics_csv_f=metrics_csv_f,
        total_steps_target=total_steps,
        rollout_steps=rollout_steps,
        train_epochs=train_epochs,
        batch_size=batch_size,
        bc_warmstart_cfg=bc_warm_cfg,
        bc_supervision_obs=bc_warm_obs,
        bc_supervision_act=bc_warm_act,
        bc_supervision_cost=bc_warm_cost,
        schedule_step_offset=int(resume_step_offset_effective),
        curriculum_step_offset=int(resume_step_offset_effective),
    )

    
    if resume_existing_metrics and int(resume_env_step) > 0:
        try:
            trainer._resume_enable = True
            trainer._resume_env_step = int(resume_env_step)
            trainer._resume_train_episode = int(resume_hist.get("resume_train_episode", 0) or 0)
        except Exception:
            pass

        try:
            
            trainer.reward_history = list(resume_hist.get("reward_history", []))
            trainer.policy_loss_history = list(resume_hist.get("policy_loss_history", []))
            trainer.value_loss_history = list(resume_hist.get("value_loss_history", []))
            trainer.entropy_history = list(resume_hist.get("entropy_history", []))
            trainer.kl_div_history = list(resume_hist.get("kl_div_history", []))
            trainer.clipfrac_history = list(resume_hist.get("clipfrac_history", []))
            trainer.explained_variance_history = list(resume_hist.get("explained_variance_history", []))
            trainer.steps_history = list(resume_hist.get("steps_history", []))
            trainer.train_episode_history = list(resume_hist.get("episodes_history", []))
            trainer.train_episode_dense_history = list(resume_hist.get("episodes_dense_history", []))
            trainer.reward_total_ep_dense_history = list(resume_hist.get("reward_dense_history", []))
            trainer.cost_ep_dense_history = list(resume_hist.get("cost_dense_history", []))
            trainer.vio_ep_dense_history = list(resume_hist.get("vio_dense_history", []))
            trainer.improve_ep_dense_history = list(resume_hist.get("improve_dense_history", []))
            trainer.shaping_ep_dense_history = list(resume_hist.get("shaping_dense_history", []))
            trainer.comp_ep_dense_history = list(resume_hist.get("comp_dense_history", []))
            trainer.ris_ep_dense_history = list(resume_hist.get("ris_dense_history", []))
            trainer.policy_loss_ep_dense_history = list(resume_hist.get("policy_loss_dense_history", []))
            trainer.value_loss_ep_dense_history = list(resume_hist.get("value_loss_dense_history", []))
            trainer.entropy_ep_dense_history = list(resume_hist.get("entropy_dense_history", []))
            trainer.kl_ep_dense_history = list(resume_hist.get("kl_dense_history", []))
            trainer.clipfrac_ep_dense_history = list(resume_hist.get("clipfrac_dense_history", []))
            trainer.explained_variance_ep_dense_history = list(resume_hist.get("explained_variance_dense_history", []))
            trainer.lr_ep_dense_history = list(resume_hist.get("lr_dense_history", []))
            trainer.train_episode = int(resume_hist.get("resume_train_episode", 0) or 0)
            trainer.cost_history = list(resume_hist.get("cost_history", []))
            trainer.vio_history = list(resume_hist.get("vio_history", []))
            trainer.improve_history = list(resume_hist.get("improve_history", []))

            trainer.eval_steps_history = list(resume_hist.get("eval_steps_history", []))
            trainer.eval_train_episode_history = list(resume_hist.get("eval_episodes_history", []))
            trainer.eval_reward_history = list(resume_hist.get("eval_reward_history", []))
            trainer.eval_reward_ci_history = list(resume_hist.get("eval_reward_ci_history", []))
            trainer.eval_paper_cost_history = list(resume_hist.get("eval_paper_cost_history", []))
            trainer.eval_paper_cost_std_history = list(resume_hist.get("eval_paper_cost_std_history", []))
            trainer.eval_paper_cost_ci_history = list(resume_hist.get("eval_paper_cost_ci_history", []))
            trainer.eval_paper_return_history = list(resume_hist.get("eval_paper_return_history", []))
            trainer.eval_paper_return_ci_history = list(resume_hist.get("eval_paper_return_ci_history", []))
            trainer.eval_n_history = list(resume_hist.get("eval_n_history", []))
            trainer.eval_vio_history = list(resume_hist.get("eval_vio_history", []))
            trainer.eval_vio_any_frac_history = list(resume_hist.get("eval_vio_any_frac_history", []))
            trainer.eval_vio_any_frac_ci_history = list(resume_hist.get("eval_vio_any_frac_ci_history", []))
            trainer.eval_improve_history = list(resume_hist.get("eval_improve_history", []))
            trainer.eval_coverage_history = list(resume_hist.get("eval_coverage_history", []))
            trainer.eval_comp_gain_ratio_history = list(resume_hist.get("eval_comp_gain_ratio_history", []))
            trainer.eval_ris_gain_ratio_history = list(resume_hist.get("eval_ris_gain_ratio_history", []))
            trainer.eval_det_paper_cost_ep_dense_history = list(resume_hist.get("eval_det_paper_cost_dense_history", []))
            trainer.eval_det_paper_cost_ci_ep_dense_history = list(resume_hist.get("eval_det_paper_cost_ci_dense_history", []))
            trainer.eval_det_vio_any_frac_ep_dense_history = list(resume_hist.get("eval_det_vio_any_frac_dense_history", []))
            trainer.eval_det_improve_ep_dense_history = list(resume_hist.get("eval_det_improve_dense_history", []))
            trainer._latest_eval_det_paper_cost = (
                float(trainer.eval_det_paper_cost_ep_dense_history[-1])
                if len(trainer.eval_det_paper_cost_ep_dense_history) > 0
                else float("nan")
            )
            trainer._latest_eval_det_paper_cost_ci = (
                float(trainer.eval_det_paper_cost_ci_ep_dense_history[-1])
                if len(trainer.eval_det_paper_cost_ci_ep_dense_history) > 0
                else float("nan")
            )
            trainer._latest_eval_det_vio_any_frac = (
                float(trainer.eval_det_vio_any_frac_ep_dense_history[-1])
                if len(trainer.eval_det_vio_any_frac_ep_dense_history) > 0
                else float("nan")
            )
            trainer._latest_eval_det_improve = (
                float(trainer.eval_det_improve_ep_dense_history[-1])
                if len(trainer.eval_det_improve_ep_dense_history) > 0
                else float("nan")
            )
        except Exception:
            pass

    trainer.run()

    # final ckpt & normalizer
    try:
        trainer.save_ckpt("final")
    except Exception as e:
        
        raise RuntimeError(
            f"[HB][train][FATAL] failed to save final checkpoint: {type(e).__name__}: {e}"
        ) from e
    
    
    try:
        need_best_fallback = bool(getattr(trainer, "meta_best_after_action", False)) and bool(
            getattr(trainer, "_meta_has_applied_action", False)
        ) and (getattr(trainer, "best_ckpt_path", None) is None or int(getattr(trainer, "best_ckpt_step", -1) or -1) < 0)
        if need_best_fallback:
            trainer.best_ckpt_step = int(getattr(trainer, "env_step", 0))
            if len(getattr(trainer, "eval_paper_cost_history", [])) > 0:
                try:
                    trainer.best_eval_paper_cost = float(trainer.eval_paper_cost_history[-1])
                except Exception:
                    trainer.best_eval_paper_cost = float("inf")
            trainer.best_ckpt_path = trainer.save_ckpt("best")
            print(
                f"[HB][meta] fallback best->final step={int(trainer.best_ckpt_step)} path={str(trainer.best_ckpt_path)}",
                flush=True,
            )
    except Exception as e:
        print(f"[HB][meta][WARN] finalize_best_fallback failed: {type(e).__name__}: {e}", flush=True)

    
    phaseg_v11_selfcheck: Dict[str, Any] = {}
    rd_now = str(getattr(env_cfg, "reward_design", "")).strip().lower()
    if rd_now in ("reward_total_v1", "reward_total", "total_v1", "v1"):
        try:
            phaseg_v11_selfcheck = trainer.run_phaseg_v11_selfcheck(
                output_path=run_dir / "phaseg_v11_selfcheck.json",
                corr_threshold=0.20,
                corr_min_episodes=50,
            )
        except Exception as e:
            phaseg_v11_selfcheck = {
                "pass": False,
                "reason": f"selfcheck_exception:{type(e).__name__}",
                "error": str(e),
            }
            print(f"[HB][phaseg-v1.1][WARN] selfcheck failed: {type(e).__name__}: {e}", flush=True)

    
    try:
        ckpt_dir = run_dir / CKPT_DIR_NAME
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        phase_ckpt_name = ""
        phase_norm_tag = ""
        if str(scores_mode) == "fixed":
            phase_ckpt_name = "traj_only.ckpt"
            phase_norm_tag = "traj_only"
        elif str(scores_mode) == "learned" and bool(resume_path):
            phase_ckpt_name = "joint_finetune.ckpt"
            phase_norm_tag = "joint_finetune"

        if phase_ckpt_name:
            phase_ckpt_path = ckpt_dir / phase_ckpt_name
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": _optimizer_export_state(optimizer),
                    "history": trainer._build_ckpt_history(),
                    "meta": {
                        "global_step": int(getattr(trainer, "env_step", 0) or 0),
                        "train_episode": int(getattr(trainer, "train_episode", 0) or 0),
                    },
                },
                str(phase_ckpt_path),
            )
            print(f"[HB][train][PhaseC] Saved PhaseC ckpt: {phase_ckpt_path}", flush=True)

            if obs_normalizer is not None and phase_norm_tag:
                st = obs_normalizer.get_stats()
                np.savez(
                    ckpt_dir / f"normalizer_{phase_norm_tag}.npz",
                    count=int(st["count"]),
                    mean=np.asarray(st["mean"], dtype=np.float32),
                    var=np.asarray(st["var"], dtype=np.float32),
                )
                
                np.savez(
                    ckpt_dir / "normalizer_latest.npz",
                    count=int(st["count"]),
                    mean=np.asarray(st["mean"], dtype=np.float32),
                    var=np.asarray(st["var"], dtype=np.float32),
                )

        
        
        if bool(train_params.get("reward_shaping", False)):
            shaping_ckpt_path = ckpt_dir / "shaping_on.ckpt"
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": _optimizer_export_state(optimizer),
                    "history": trainer._build_ckpt_history(),
                    "meta": {
                        "global_step": int(getattr(trainer, "env_step", 0) or 0),
                        "train_episode": int(getattr(trainer, "train_episode", 0) or 0),
                    },
                },
                str(shaping_ckpt_path),
            )
            print(f"[HB][train][PhaseC] Saved C3 ckpt: {shaping_ckpt_path}", flush=True)

            if obs_normalizer is not None:
                st = obs_normalizer.get_stats()
                np.savez(
                    ckpt_dir / "normalizer_shaping_on.npz",
                    count=int(st["count"]),
                    mean=np.asarray(st["mean"], dtype=np.float32),
                    var=np.asarray(st["var"], dtype=np.float32),
                )
                
                np.savez(
                    ckpt_dir / "normalizer_latest.npz",
                    count=int(st["count"]),
                    mean=np.asarray(st["mean"], dtype=np.float32),
                    var=np.asarray(st["var"], dtype=np.float32),
                )
    except Exception as e:
        print(f"[HB][train][PhaseC][WARN] phaseC ckpt{type(e).__name__}: {e}", flush=True)

    if obs_normalizer is not None:
        try:
            st = obs_normalizer.get_stats()
            ckpt_dir = run_dir / CKPT_DIR_NAME
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                ckpt_dir / "normalizer_final.npz",
                count=int(st["count"]),
                mean=np.asarray(st["mean"], dtype=np.float32),
                var=np.asarray(st["var"], dtype=np.float32),
            )
        except Exception:
            pass

    meta_stats: Dict[str, Any] = {}
    if meta_controller is not None:
        try:
            meta_stats = meta_controller.stats()
            meta_controller.save_stats(run_dir / "meta_action_stats.json")
        except Exception:
            meta_stats = {}

    return {
        "trainer": trainer,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden": int(hidden_sizes[-1]) if hidden_sizes else None,
        "obs_normalizer": obs_normalizer,
        "best_ckpt_step": trainer.best_ckpt_step,
        "best_paper_cost": trainer.best_eval_paper_cost,
        "best_ckpt_path": str(trainer.best_ckpt_path) if trainer.best_ckpt_path is not None else None,
        "total_steps": int(trainer.env_step),
        "total_updates": int(len(trainer.steps_history)),
        "total_episodes": int(getattr(trainer.train_collector, "collect_episode", 0)),
        "phaseg_v11_selfcheck": phaseg_v11_selfcheck,
        "meta_stats": meta_stats,
    }
