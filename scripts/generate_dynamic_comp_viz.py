#!/usr/bin/env python3
"""
 CoMP 


-  run_dir  checkpoint  PPO 
-  episode  CoMP 
- ///
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algos.tianshou.ppo_config import create_ppo_policy_from_config
from src.envs.comp_ris_env_joint import CompRISEnvJoint
from src.utils.dynamic_comp_viz import DynamicCoMPVisualizer
from src.utils.env_cfg import build_joint_env_cfg
from src.utils.obs_normalizer import ObsNormalizer


def _parse_bool_flag(v: int) -> bool:
    """ 0/1 """
    return bool(int(v) != 0)


def _set_global_seed(seed: int) -> None:
    """"""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _safe_torch_load(path: Path, map_location: Any) -> Any:
    """ PyTorch """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(str(path), map_location=map_location)
    except Exception:
        try:
            return torch.load(str(path), map_location=map_location, weights_only=False)  # type: ignore[call-arg]
        except TypeError:
            return torch.load(str(path), map_location=map_location)


def _resolve_ckpt_path(run_dir: Path) -> Path:
    """ checkpoint"""
    ckpt_dir = run_dir / "checkpoints"
    cands = [
        ckpt_dir / "ckpt_best.pt",
        ckpt_dir / "ckpt_final.pt",
        ckpt_dir / "ckpt_latest.pt",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"Checkpoint not found under: {ckpt_dir}")


def _resolve_cfg_paths(run_dir: Path) -> Tuple[Path, Path]:
    """ run_dir  env/train """
    env_yaml = run_dir / "configs" / "env_fixed.yaml"
    train_yaml = run_dir / "configs" / "train_tianshou_fixed.yaml"
    if not env_yaml.exists():
        raise FileNotFoundError(f"env yaml not found: {env_yaml}")
    if not train_yaml.exists():
        raise FileNotFoundError(f"train yaml not found: {train_yaml}")
    return env_yaml, train_yaml


def _merge_env_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """+ run """
    if not overrides:
        return dict(base)
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict):
            old = out.get(k, {})
            if not isinstance(old, dict):
                old = {}
            nv = dict(old)
            nv.update(v)
            out[k] = nv
        else:
            out[k] = v
    return out


def _build_viz_env_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """"""
    if not _parse_bool_flag(args.viz_showcase):
        return {}
    return {
        "scenario": {
            "user_mobility_mode": str(args.viz_user_mobility_mode),
            "user_mobility_prob": float(args.viz_user_mobility_prob),
            "user_speed": float(args.viz_user_speed),
            "user_heading_jitter": float(args.viz_user_heading_jitter),
        }
    }


def _to_int_list(x: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    """ hidden_sizes 2"""
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        try:
            return int(x[0]), int(x[1])
        except Exception:
            pass
    if isinstance(x, (int, float)):
        h = int(x)
        return h, h
    return int(default[0]), int(default[1])


def _flatten_ppo_policy_cfg(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    train/network create_ppo_policy_from_config 
     checkpoint 
    """
    train = train_cfg.get("train", {}) if isinstance(train_cfg, dict) else {}
    net = train_cfg.get("network", {}) if isinstance(train_cfg, dict) else {}
    if not isinstance(train, dict):
        train = {}
    if not isinstance(net, dict):
        net = {}

    h0, h1 = _to_int_list(net.get("hidden_sizes", net.get("hidden_size", [256, 256])), (256, 256))
    cfg: Dict[str, Any] = {
        "hidden_sizes": [h0, h1],
        "use_layernorm": bool(net.get("use_layernorm", True)),
        "share_preprocess": bool(net.get("share_preprocess", True)),
        "lr": float(train.get("lr", 1e-4)),
        "gamma": float(train.get("gamma", 0.99)),
        "gae_lambda": float(train.get("lam", train.get("gae_lambda", 0.95))),
        "eps_clip": float(train.get("clip_ratio", train.get("eps_clip", 0.2))),
        "target_kl": float(train.get("target_kl", 0.0)),
        "kl_stop_mult": float(train.get("kl_stop_mult", 1.5)),
        "vf_coef": float(train.get("vf_coef", 0.5)),
        "ent_coef": float(train.get("ent_coef", 0.01)),
        "max_grad_norm": float(train.get("max_grad_norm", 0.5)),
        "logp_mode": str(train.get("logp_mode", "sum")),
        "reward_normalization": bool(train.get("reward_normalization", train.get("value_normalization", False))),
        "deterministic_eval": True,
    }
    return cfg


def _load_obs_normalizer(ckpt_dir: Path, ckpt_path: Path, obs_dim: int) -> Optional[ObsNormalizer]:
    """ ckpt  normalizer"""
    name = str(ckpt_path.name).lower()
    cands = []
    if "best" in name:
        cands.append(ckpt_dir / "normalizer_best.npz")
    if "final" in name:
        cands.append(ckpt_dir / "normalizer_final.npz")
    cands.extend(
        [
            ckpt_dir / "normalizer_best.npz",
            ckpt_dir / "normalizer_final.npz",
            ckpt_dir / "normalizer.npz",
        ]
    )
    seen = set()
    for p in cands:
        k = str(p.resolve()) if p.exists() else str(p)
        if k in seen:
            continue
        seen.add(k)
        if not p.exists():
            continue
        try:
            data = np.load(p)
            norm = ObsNormalizer(obs_dim=obs_dim)
            norm.set_stats(
                {
                    "count": int(data["count"]),
                    "mean": data["mean"],
                    "var": data["var"],
                }
            )
            return norm
        except Exception:
            continue
    return None


def _policy_action(
    policy: Any,
    obs: np.ndarray,
    deterministic: bool,
    device: torch.device,
    obs_raw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """"""
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).float().unsqueeze(0).to(device)
    info = None
    if obs_raw is not None:
        
        info = {"obs_raw": np.asarray(obs_raw, dtype=np.float32)}
    with torch.no_grad():
        try:
            (mu, sigma), _ = policy.actor(obs_t, info=info)
        except TypeError:
            (mu, sigma), _ = policy.actor(obs_t)
        if deterministic:
            act_t = mu
        else:
            dist_fn = getattr(policy, "dist_fn", None)
            if callable(dist_fn):
                act_t = dist_fn(mu, sigma).sample()
            else:
                act_t = torch.distributions.Normal(mu, sigma).sample()
    act = act_t.squeeze(0).detach().cpu().numpy().reshape(-1)
    return act.astype(np.float32, copy=False)


def load_policy(run_dir: Path, device: str, env_overrides: Optional[Dict[str, Any]] = None):
    """ PPO  +  + obs normalizer"""
    ckpt_path = _resolve_ckpt_path(run_dir)
    env_yaml, train_yaml = _resolve_cfg_paths(run_dir)

    with open(env_yaml, "r", encoding="utf-8") as f:
        env_cfg_dict = yaml.safe_load(f) or {}
    with open(train_yaml, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f) or {}

    env_cfg_effective = _merge_env_dict(env_cfg_dict, env_overrides or {})
    env_cfg, _, _ = build_joint_env_cfg(env_cfg_effective)
    env = CompRISEnvJoint(env_cfg)

    obs_dim = int(env.obs_dim)
    act_dim = int(env.action_dim)
    policy_cfg = _flatten_ppo_policy_cfg(train_cfg if isinstance(train_cfg, dict) else {})

    policy, _ = create_ppo_policy_from_config(
        obs_dim=obs_dim,
        act_dim=act_dim,
        config=policy_cfg,
        device=device,
    )

    ckpt_obj = _safe_torch_load(ckpt_path, map_location=device)
    state_dict = ckpt_obj["policy"] if isinstance(ckpt_obj, dict) and ("policy" in ckpt_obj) else ckpt_obj
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to(torch.device(device))

    obs_normalizer = _load_obs_normalizer(run_dir / "checkpoints", ckpt_path, obs_dim)
    return policy, env, obs_normalizer, env_cfg, env_cfg_effective, ckpt_path, policy_cfg


def collect_trace(
    env: CompRISEnvJoint,
    policy: Any,
    obs_normalizer: Optional[ObsNormalizer],
    n_steps: int,
    deterministic: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """ episode  trace"""
    env.enable_trace(enable=True, keep_theta=True, max_steps=int(n_steps))

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for _ in range(int(n_steps)):
        obs_np = np.asarray(obs, dtype=np.float32)
        if obs_normalizer is not None:
            obs_in = obs_normalizer.normalize(obs_np, update=False)
        else:
            obs_in = obs_np
        action = _policy_action(
            policy,
            obs_in,
            deterministic=bool(deterministic),
            device=device,
            obs_raw=obs_np,
        )
        obs_next, _reward, done, _info = env.step(action)
        obs = obs_next
        if done:
            break
    return env.get_trace(clear=True)


def _normalize_assoc_matrix_for_stats(a_raw: Any, I: int, M: int) -> Optional[np.ndarray]:
    """ trace  (I, M) """
    if a_raw is None:
        return None
    try:
        arr = np.asarray(a_raw, dtype=np.float64)
    except Exception:
        return None
    if arr.size != int(I) * int(M):
        return None
    if arr.ndim == 2:
        if arr.shape == (I, M):
            return np.asarray(arr, dtype=np.float64)
        if arr.shape == (M, I):
            return np.asarray(arr.T, dtype=np.float64)
    return np.asarray(arr.reshape(I, M), dtype=np.float64)


def _infer_user_ring_labels(users_xy: np.ndarray, L: float) -> list[str]:
    """ inner/mid/outer outer """
    center = np.array([float(L) / 2.0, float(L) / 2.0], dtype=np.float64)
    d = np.linalg.norm(np.asarray(users_xy, dtype=np.float64) - center, axis=1)
    n = int(d.size)
    if n <= 0:
        return []
    if n < 3:
        return ["mid" for _ in range(n)]
    order = np.argsort(d).astype(np.int64)
    n_inner = int(np.floor(n / 3))
    n_outer = int(np.floor(n / 3))
    n_mid = int(n - n_inner - n_outer)
    if n_inner <= 0:
        n_inner = 1
        n_mid = max(n_mid - 1, 1)
    if n_outer <= 0:
        n_outer = 1
        n_mid = max(n_mid - 1, 1)
    labels = ["mid" for _ in range(n)]
    for idx in order[:n_inner]:
        labels[int(idx)] = "inner"
    for idx in order[n_inner + n_mid :]:
        labels[int(idx)] = "outer"
    return labels


def _calc_dynamic_comp_stats(trace: Dict[str, Any], L: float, M: int, I: int) -> Dict[str, Any]:
    """
    C8CoMP
    
    - CoMP
    - 
    - CoMP
    """
    frames = list(trace.get("frames", []) or [])
    users_xy = np.asarray(trace.get("w", np.zeros((I, 2), dtype=np.float64)), dtype=np.float64).reshape(int(I), 2)
    ring_labels = _infer_user_ring_labels(users_xy, float(L))
    T = int(len(frames))

    mask_matrix = np.zeros((int(I), max(T, 1)), dtype=np.int64)
    popcount_matrix = np.zeros((int(I), max(T, 1)), dtype=np.int64)
    assoc_threshold = 0.0

    for t, frame in enumerate(frames):
        a = _normalize_assoc_matrix_for_stats(frame.get("a", None), int(I), int(M))
        if a is None:
            continue
        for i in range(int(I)):
            mask = 0
            for m in range(int(M)):
                if float(a[i, m]) > float(assoc_threshold):
                    mask |= (1 << m)
            mask_matrix[i, t] = int(mask)
            popcount_matrix[i, t] = int(bin(mask).count("1"))

    if T <= 0:
        switch_counts = np.zeros((int(I),), dtype=np.int64)
        global_switch_flags = np.zeros((0,), dtype=np.int64)
        switch_user_ratio_t = np.zeros((0,), dtype=np.float64)
        comp_ratio_user = np.zeros((int(I),), dtype=np.float64)
        avg_serving_user = np.zeros((int(I),), dtype=np.float64)
        comp_active_ratio_t = np.zeros((0,), dtype=np.float64)
        avg_serving_t = np.zeros((0,), dtype=np.float64)
    else:
        if T >= 2:
            switched = (mask_matrix[:, 1:T] != mask_matrix[:, : T - 1])
            switch_counts = np.sum(switched, axis=1).astype(np.int64)
            global_switch_flags = np.any(switched, axis=0).astype(np.int64)
            switch_user_ratio_t = np.mean(switched, axis=0).astype(np.float64)
        else:
            switch_counts = np.zeros((int(I),), dtype=np.int64)
            global_switch_flags = np.zeros((0,), dtype=np.int64)
            switch_user_ratio_t = np.zeros((0,), dtype=np.float64)
        comp_ratio_user = np.mean(popcount_matrix[:, :T] >= 2, axis=1)
        avg_serving_user = np.mean(popcount_matrix[:, :T], axis=1)
        comp_active_ratio_t = np.mean(popcount_matrix[:, :T] >= 2, axis=0)
        avg_serving_t = np.mean(popcount_matrix[:, :T], axis=0)

    ring_stats: Dict[str, Dict[str, float]] = {}
    for ring in ("inner", "mid", "outer"):
        idx = [i for i, r in enumerate(ring_labels) if r == ring]
        if len(idx) <= 0:
            continue
        ring_stats[ring] = {
            "user_count": float(len(idx)),
            "avg_switch_count": float(np.mean(switch_counts[idx])) if len(idx) > 0 else 0.0,
            "avg_comp_ratio": float(np.mean(comp_ratio_user[idx])) if len(idx) > 0 else 0.0,
            "avg_serving_uavs": float(np.mean(avg_serving_user[idx])) if len(idx) > 0 else 0.0,
        }

    hist: Dict[str, float] = {}
    if T > 0:
        hist_counts = np.zeros((int(M) + 1,), dtype=np.float64)
        for k in range(int(M) + 1):
            hist_counts[k] = float(np.mean(popcount_matrix[:, :T] == k))
        for k in range(int(M) + 1):
            hist[str(k)] = float(hist_counts[k])

    global_switch_events = int(np.sum(global_switch_flags)) if global_switch_flags.size > 0 else 0
    global_switch_rate = float(global_switch_events / max(int(T - 1), 1))

    out: Dict[str, Any] = {
        "n_users": int(I),
        "n_frames": int(T),
        "comp_change_frequency": {
            "global_switch_events": int(global_switch_events),
            "global_switch_rate_per_step": float(global_switch_rate),
            "avg_switch_per_user": float(np.mean(switch_counts)) if switch_counts.size > 0 else 0.0,
            "p90_switch_per_user": float(np.percentile(switch_counts, 90)) if switch_counts.size > 0 else 0.0,
            "max_switch_per_user": float(np.max(switch_counts)) if switch_counts.size > 0 else 0.0,
        },
        "user_service_switch_counts": [int(x) for x in switch_counts.tolist()],
        "ring_pattern_diff": ring_stats,
        "comp_size_distribution": hist,
        "time_series": {
            "comp_active_ratio_t": [float(x) for x in comp_active_ratio_t.tolist()],
            "avg_serving_uavs_t": [float(x) for x in avg_serving_t.tolist()],
            "global_switch_flag_t": [int(x) for x in global_switch_flags.tolist()],
            "switch_user_ratio_t": [float(x) for x in switch_user_ratio_t.tolist()],
        },
    }
    return out


def _plot_dynamic_comp_stats(stats: Dict[str, Any], save_path: Path, title: str) -> None:
    """C8"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ts = stats.get("time_series", {}) if isinstance(stats, dict) else {}
    comp_active_ratio_t = np.asarray(ts.get("comp_active_ratio_t", []), dtype=np.float64).reshape(-1)
    switch_ratio_t = np.asarray(ts.get("switch_user_ratio_t", []), dtype=np.float64).reshape(-1)
    switch_flag_t = np.asarray(ts.get("global_switch_flag_t", []), dtype=np.float64).reshape(-1)
    switch_counts = np.asarray(stats.get("user_service_switch_counts", []), dtype=np.float64).reshape(-1)
    ring_stats = stats.get("ring_pattern_diff", {}) if isinstance(stats, dict) else {}
    comp_size_dist = stats.get("comp_size_distribution", {}) if isinstance(stats, dict) else {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.reshape(-1)

    
    if comp_active_ratio_t.size > 0:
        x = np.arange(int(comp_active_ratio_t.size))
        l1, = ax1.plot(x, comp_active_ratio_t * 100.0, color="#1f77b4", linewidth=2.2, label="CoMP(%)")
        ax1.set_ylim(0.0, 100.0)
        ax1.set_ylabel("CoMP(%)")
        l2 = None
        if switch_ratio_t.size > 0:
            if switch_ratio_t.size == max(comp_active_ratio_t.size - 1, 0):
                x_sw = np.arange(1, int(comp_active_ratio_t.size))
            else:
                x_sw = np.arange(int(switch_ratio_t.size))
            ax1b = ax1.twinx()
            l2, = ax1b.plot(
                x_sw,
                switch_ratio_t * 100.0,
                color="#d62728",
                linewidth=1.8,
                linestyle="--",
                alpha=0.9,
                label="(%)",
            )
            ax1b.set_ylim(0.0, 100.0)
            ax1b.set_ylabel("(%)")
        elif switch_flag_t.size > 0:
            
            if switch_flag_t.size == max(comp_active_ratio_t.size - 1, 0):
                x_sw = np.arange(1, int(comp_active_ratio_t.size))
            else:
                x_sw = np.arange(int(switch_flag_t.size))
            ax1b = ax1.twinx()
            l2, = ax1b.plot(
                x_sw,
                switch_flag_t * 100.0,
                color="#d62728",
                linewidth=1.6,
                linestyle="--",
                alpha=0.85,
                label="(0/1)",
            )
            ax1b.set_ylim(0.0, 100.0)
            ax1b.set_ylabel("(%)")
        if l2 is not None:
            ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best")
        else:
            ax1.legend([l1], [l1.get_label()], loc="best")
    ax1.set_title("", fontweight="bold")
    ax1.set_xlabel(" t")
    ax1.grid(True, alpha=0.25)

    
    if switch_counts.size > 0:
        idx = np.arange(1, int(switch_counts.size) + 1)
        ax2.bar(idx, switch_counts, color="#2ca02c", edgecolor="black", linewidth=0.7)
    ax2.set_title("", fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.grid(True, axis="y", alpha=0.25)

    
    rings = ["inner", "mid", "outer"]
    x3 = np.arange(len(rings), dtype=np.float64)
    ring_switch = np.asarray([float((ring_stats.get(r, {}) or {}).get("avg_switch_count", np.nan)) for r in rings], dtype=np.float64)
    ring_ratio = np.asarray([float((ring_stats.get(r, {}) or {}).get("avg_comp_ratio", np.nan)) * 100.0 for r in rings], dtype=np.float64)
    bar = ax3.bar(x3 - 0.18, np.nan_to_num(ring_switch, nan=0.0), width=0.36, color="#9467bd", label="")
    ax3.set_ylabel("")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(rings)
    ax3.set_title("", fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.25)
    ax3b = ax3.twinx()
    line, = ax3b.plot(x3 + 0.18, np.nan_to_num(ring_ratio, nan=0.0), color="#ff7f0e", marker="o", linewidth=1.8, label="CoMP(%)")
    ax3b.set_ylabel("CoMP(%)")
    ax3.legend([bar, line], ["", "CoMP(%)"], loc="best")

    
    if isinstance(comp_size_dist, dict) and len(comp_size_dist) > 0:
        ks = sorted([int(k) for k in comp_size_dist.keys()])
        vs = [float(comp_size_dist.get(str(k), 0.0)) * 100.0 for k in ks]
        ax4.bar(ks, vs, color="#17becf", edgecolor="black", linewidth=0.7)
        ax4.set_xticks(ks)
        ax4.set_xlabel("UAV |S|")
        ax4.set_ylabel("(%)")
    ax4.set_title("", fontweight="bold")
    ax4.grid(True, axis="y", alpha=0.25)

    summary = stats.get("comp_change_frequency", {}) if isinstance(stats, dict) else {}
    txt = (
        f": {float(summary.get('global_switch_rate_per_step', 0.0)):.3f}/step\n"
        f": {float(summary.get('avg_switch_per_user', 0.0)):.2f}\n"
        f"P90: {float(summary.get('p90_switch_per_user', 0.0)):.2f}"
    )
    ax4.text(
        0.03,
        0.98,
        txt,
        transform=ax4.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#777777", alpha=0.9),
    )

    fig.suptitle(f"CoMP{title}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_bundle_c8_summary(case_metas: list[Dict[str, Any]]) -> Dict[str, Any]:
    """C8"""
    cases: list[Dict[str, Any]] = []
    for meta in case_metas:
        c = meta.get("dynamic_comp_stats", {}) if isinstance(meta, dict) else {}
        f = c.get("comp_change_frequency", {}) if isinstance(c, dict) else {}
        cases.append(
            {
                "case_name": str(meta.get("case_name", "unknown")),
                "global_switch_rate_per_step": float(f.get("global_switch_rate_per_step", 0.0)),
                "avg_switch_per_user": float(f.get("avg_switch_per_user", 0.0)),
                "avg_comp_active_ratio_t": float(
                    np.mean(np.asarray((c.get("time_series", {}) or {}).get("comp_active_ratio_t", []), dtype=np.float64))
                )
                if isinstance(c, dict)
                else 0.0,
            }
        )
    return {"cases": cases}


def _plot_bundle_c8_summary(bundle_stats: Dict[str, Any], save_path: Path) -> None:
    """C8"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list((bundle_stats.get("cases", []) if isinstance(bundle_stats, dict) else []) or [])
    if len(rows) <= 0:
        return

    case_names = [str(r.get("case_name", "case")) for r in rows]
    switch_rate = np.asarray([float(r.get("global_switch_rate_per_step", 0.0)) for r in rows], dtype=np.float64)
    avg_switch = np.asarray([float(r.get("avg_switch_per_user", 0.0)) for r in rows], dtype=np.float64)
    comp_ratio = np.asarray([float(r.get("avg_comp_active_ratio_t", 0.0)) * 100.0 for r in rows], dtype=np.float64)
    x = np.arange(len(case_names), dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    w = 0.25
    b1 = ax1.bar(x - w, switch_rate, width=w, color="#1f77b4", label="(/step)")
    b2 = ax1.bar(x, avg_switch, width=w, color="#2ca02c", label="")
    ax1.set_xticks(x)
    ax1.set_xticklabels(case_names, rotation=0)
    ax1.set_ylabel("")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.set_title("CoMP", fontweight="bold")

    ax2 = ax1.twinx()
    l3, = ax2.plot(x + w, comp_ratio, color="#d62728", marker="o", linewidth=2.0, label="CoMP(%)")
    ax2.set_ylabel("CoMP(%)")

    ax1.legend([b1, b2, l3], ["(/step)", "", "CoMP(%)"], loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_single_case(
    *,
    case_name: str,
    run_dir: Path,
    output_dir: Path,
    seed: int,
    n_steps: int,
    device: str,
    deterministic: bool,
    env_overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """ CoMP  case"""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CoMP][{case_name}] : {run_dir}", flush=True)
    _set_global_seed(seed)

    policy, env, obs_normalizer, env_cfg, env_cfg_effective, ckpt_path, policy_cfg = load_policy(
        run_dir=run_dir,
        device=device,
        env_overrides=env_overrides or {},
    )
    dev = torch.device(device)

    print(f"[CoMP][{case_name}] ...", flush=True)
    trace = collect_trace(
        env=env,
        policy=policy,
        obs_normalizer=obs_normalizer,
        n_steps=int(n_steps),
        deterministic=bool(deterministic),
        device=dev,
    )

    print(f"[CoMP][{case_name}] ...", flush=True)
    ris_positions_cfg = getattr(env_cfg, "ris_positions", None)
    visualizer = DynamicCoMPVisualizer(
        trace=trace,
        L=env_cfg.L,
        M=env_cfg.M,
        I=env_cfg.I,
        v=env_cfg.v,
        ris_positions=ris_positions_cfg,
        enable_ris=env_cfg.enable_ris,
        enable_comp=env_cfg.enable_comp,
    )
    T = len(trace.get("frames", []))
    frame_indices = [T // 4, T // 2, (3 * T) // 4] if T > 0 else []
    if frame_indices:
        visualizer.create_key_frames(output_dir, frame_indices=frame_indices)
    visualizer.create_comp_timeline(output_dir / "comp_timeline.png")
    visualizer.create_user_focus_animation(output_dir / "user_focus_animation.gif", fps=10)
    visualizer.export_comp_metrics_csv(output_dir / "comp_metrics_by_ring.csv")

    
    c8_stats = _calc_dynamic_comp_stats(
        trace=trace,
        L=float(env_cfg.L),
        M=int(env_cfg.M),
        I=int(env_cfg.I),
    )
    c8_stats_path = output_dir / "dynamic_comp_stats.json"
    c8_fig_path = output_dir / "dynamic_comp_stats.png"
    with open(c8_stats_path, "w", encoding="utf-8") as f:
        json.dump(c8_stats, f, ensure_ascii=False, indent=2)
    _plot_dynamic_comp_stats(c8_stats, c8_fig_path, title=str(case_name))

    meta = {
        "case_name": str(case_name),
        "run_dir": str(run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": int(seed),
        "n_steps": int(n_steps),
        "device": str(device),
        "deterministic": bool(deterministic),
        "ckpt_path": str(ckpt_path.resolve()),
        "policy_cfg": dict(policy_cfg),
        "env_overrides": dict(env_overrides or {}),
        "n_frames": int(T),
        "num_ris_cfg": int(getattr(env_cfg, "num_ris", 1)),
        "ris_positions_cfg": (
            np.asarray(ris_positions_cfg, dtype=np.float64).reshape(-1, 2).tolist()
            if ris_positions_cfg is not None
            else [np.asarray(env_cfg.v, dtype=np.float64).reshape(2,).tolist()]
        ),
        "scenario_effective": dict((env_cfg_effective or {}).get("scenario", {}) or {}),
        "objective_effective": dict((env_cfg_effective or {}).get("objective", {}) or {}),
        "dynamic_comp_stats_json": str(c8_stats_path.resolve()),
        "dynamic_comp_stats_figure": str(c8_fig_path.resolve()),
        "dynamic_comp_stats": c8_stats,
    }
    with open(output_dir / "viz_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[CoMP][{case_name}] : {output_dir}", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=" CoMP ")
    parser.add_argument("--run_dir", type=str, required=True, help=" run ")
    parser.add_argument("--output_dir", type=str, default=None, help=" run_dir/figs/DynamicCoMP_Bundle evalCoMP")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--n_steps", type=int, default=80, help="episode ")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--deterministic", type=int, default=1, help="10")
    parser.add_argument("--viz_showcase", type=int, default=0, help="10")
    parser.add_argument("--viz_user_mobility_mode", type=str, default="slow", help="")
    parser.add_argument("--viz_user_speed", type=float, default=2.0, help="m/s")
    parser.add_argument("--viz_user_mobility_prob", type=float, default=1.0, help="")
    parser.add_argument("--viz_user_heading_jitter", type=float, default=0.15, help="")
    parser.add_argument("--three_case_bundle", type=int, default=1, help="10")
    parser.add_argument("--bundle_add_mixed_case", type=int, default=1, help=" mixed 10")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    if args.output_dir is None:
        
        out_name = "DynamicCoMP_Bundle evalCoMP" if _parse_bool_flag(args.three_case_bundle) else "dynamic_comp_viz evalCoMP2"
        output_dir = run_dir / "figs" / out_name
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if _parse_bool_flag(args.three_case_bundle):
        print(f"[CoMP] : {output_dir}", flush=True)
        slow_overrides = {
            "scenario": {
                "user_mobility_mode": str(args.viz_user_mobility_mode),
                "user_mobility_prob": float(args.viz_user_mobility_prob),
                "user_speed": float(args.viz_user_speed),
                "user_heading_jitter": float(args.viz_user_heading_jitter),
            }
        }
        cases = [
            {"case_name": "case1_static_det", "deterministic": True, "env_overrides": {}},
            {"case_name": "case2_slow_det", "deterministic": True, "env_overrides": slow_overrides},
            {"case_name": "case3_slow_stoch", "deterministic": False, "env_overrides": slow_overrides},
        ]
        if _parse_bool_flag(args.bundle_add_mixed_case):
            mixed_overrides = {
                "scenario": {
                    "user_mobility_mode": "mixed",
                    "user_mobility_prob": float(max(float(args.viz_user_mobility_prob), 0.60)),
                    "user_speed": float(max(float(args.viz_user_speed), 2.0)),
                    "user_heading_jitter": float(max(float(args.viz_user_heading_jitter), 0.25)),
                }
            }
            cases.append(
                {
                    "case_name": "case4_mixed_det",
                    "deterministic": True,
                    "env_overrides": mixed_overrides,
                }
            )
        metas = []
        for case in cases:
            metas.append(
                _run_single_case(
                    case_name=str(case["case_name"]),
                    run_dir=run_dir,
                    output_dir=output_dir / str(case["case_name"]),
                    seed=int(args.seed),
                    n_steps=int(args.n_steps),
                    device=str(args.device),
                    deterministic=bool(case["deterministic"]),
                    env_overrides=dict(case["env_overrides"]),
                )
            )
        bundle = {
            "mode": "three_case_bundle",
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "seed": int(args.seed),
            "n_steps": int(args.n_steps),
            "device": str(args.device),
            "cases": metas,
        }
        c8_bundle_stats = _build_bundle_c8_summary(metas)
        bundle["c8_bundle_stats"] = c8_bundle_stats
        c8_bundle_fig = output_dir / "bundle_c8_stats_comparison.png"
        _plot_bundle_c8_summary(c8_bundle_stats, c8_bundle_fig)
        bundle["c8_bundle_figure"] = str(c8_bundle_fig)
        with open(output_dir / "bundle_summary.json", "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)
        print(f"[CoMP] : {output_dir}", flush=True)
    else:
        env_overrides = _build_viz_env_overrides(args)
        _run_single_case(
            case_name="single_case",
            run_dir=run_dir,
            output_dir=output_dir,
            seed=int(args.seed),
            n_steps=int(args.n_steps),
            device=str(args.device),
            deterministic=_parse_bool_flag(args.deterministic),
            env_overrides=env_overrides,
        )
        print(f"[CoMP] : {output_dir}", flush=True)


if __name__ == "__main__":
    main()
