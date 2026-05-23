#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""



  1.  PPO (ARIA)  3DRL  (DDPG/SAC/TD3)  checkpoint  episode  reward 
  2.  +  seed CI95  seed
  3. 4  PPO 


  python scripts/plot_4algo_convergence.py --sigma 250 --out-dir figs/convergence_4algo
  python scripts/plot_4algo_convergence.py --sigma 200 --sigma 250 --sigma 300
  python scripts/plot_4algo_convergence.py --ppo-dirs runs/paper/seed1 runs/paper/seed2 --sigma 250
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.plot_smoothing import gaussian_smooth_with_band


# ============================================================

# ============================================================

_RUNS = _PROJECT_ROOT / "runs" / "paper"


_DEFAULT_ALGO_CONFIG = {
    "ppo": {
        "run_dirs": [_RUNS / "ppo_phaseZ4_step7_k3_r2_1200k_s3411_20260226_165956"],
        "ckpt_name": "ckpt_best.pt",
        "label": "ARIA (proposed)",
    },
    "ddpg": {
        "run_dirs": [_RUNS / "ddpg_phaseZ4_latestcfg_1200k_s3411_20260224_192300"],
        "ckpt_name": "agent_final.pt",
        "label": "DDPG",
    },
    "sac": {
        "run_dirs": [_RUNS / "sac_phaseZ4_latestcfg_1200k_s3411_20260224_192300"],
        "ckpt_name": "agent_final.pt",
        "label": "SAC",
    },
    "td3": {
        "run_dirs": [_RUNS / "td3_phaseZ4_latestcfg_1200k_s3411_20260224_192300"],
        "ckpt_name": "agent_final.pt",
        "label": "TD3",
    },
}


ALGO_COLORS = {
    "ppo":  "#2CA6A4",   
    "ddpg": "#5C8CC8",   
    "sac":  "#D96B3A",   
    "td3":  "#B5A522",   
}


ALGO_BAND_COLORS = {
    "ppo":  "#A3DCD9",
    "ddpg": "#B8CEE6",
    "sac":  "#F0C0A0",
    "td3":  "#DDD492",
}


# ============================================================

# ============================================================

def load_reward_from_ckpt(
    ckpt_path: Path,
    episode_length: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
     PPO  3DRL checkpoint  episode  reward 

    
    - PPO: obj["history"]["train_reward_total_ep_dense"]
    - 3DRL: obj["training_history"]["rewards"]

     (episodes, avg_rewards)avg_rewards  episode_length
    """
    import torch

    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(ckpt_path), map_location="cpu")

    if not isinstance(obj, dict):
        raise ValueError(f"checkpoint  dict : {ckpt_path}")

    rewards = None
    episodes = None

    
    hist = obj.get("history", {})
    if isinstance(hist, dict) and "train_reward_total_ep_dense" in hist:
        rewards = np.asarray(hist["train_reward_total_ep_dense"], dtype=np.float64)
        ep_raw = hist.get("train_episodes_dense", [])
        if len(ep_raw) > 0:
            episodes = np.asarray(ep_raw, dtype=np.float64)

    
    if rewards is None:
        th = obj.get("training_history", {})
        if isinstance(th, dict) and "rewards" in th:
            rewards = np.asarray(th["rewards"], dtype=np.float64)
            ep_raw = th.get("episodes", [])
            if len(ep_raw) > 0:
                episodes = np.asarray(ep_raw, dtype=np.float64)

    if rewards is None or rewards.size == 0:
        raise ValueError(f" {ckpt_path}  reward ")

    if episodes is None or episodes.size == 0:
        episodes = np.arange(1, rewards.size + 1, dtype=np.float64)

    
    n = min(episodes.size, rewards.size)
    episodes = episodes[:n]
    rewards = rewards[:n]

    
    mask = np.isfinite(rewards)
    episodes = episodes[mask]
    rewards = rewards[mask]

    
    avg_rewards = rewards / max(float(episode_length), 1.0)

    return episodes, avg_rewards


def load_algo_data(
    algo_key: str,
    run_dirs: List[Path],
    ckpt_name: str,
    episode_length: int = 80,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """
     seed 

     seed  mean  std seed  + std=None

     (episodes, mean_rewards, std_rewards_or_None, n_seeds)
    """
    curves: List[Tuple[np.ndarray, np.ndarray]] = []

    for rd in run_dirs:
        ckpt_path = rd / "checkpoints" / ckpt_name
        if not ckpt_path.exists():
            
            for alt in ["ckpt_final.pt", "ckpt_best.pt", "agent_final.pt", "agent_best.pt"]:
                alt_path = rd / "checkpoints" / alt
                if alt_path.exists():
                    ckpt_path = alt_path
                    break
        if not ckpt_path.exists():
            print(f"  [WARN] {algo_key}:  checkpoint: {rd}")
            continue

        try:
            ep, rw = load_reward_from_ckpt(ckpt_path, episode_length)
            print(f"  {algo_key} [{rd.name}]: {ep.size} episodes, "
                  f"reward range [{rw.min():.2f}, {rw.max():.2f}]")
            curves.append((ep, rw))
        except Exception as e:
            print(f"  [WARN] {algo_key} [{rd.name}]:  - {e}")

    if not curves:
        raise RuntimeError(f"{algo_key}: ")

    if len(curves) == 1:
        return curves[0][0], curves[0][1], None, 1

    
    min_len = min(v.size for _, v in curves)
    mat = np.stack([v[:min_len] for _, v in curves], axis=0)
    ep = curves[0][0][:min_len]
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0, ddof=1)
    return ep, mean, std, len(curves)


# ============================================================

# ============================================================

def _setup_style():
    """"""
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.1,
        "grid.color": "#D6D6D6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "font.size": 13,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "savefig.bbox": "tight",
        "savefig.dpi": 600,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#555555",
        "legend.fancybox": False,
        "axes.unicode_minus": False,
    })

    
    try:
        from matplotlib import font_manager
        preferred = ["Times New Roman", "Times", "DejaVu Serif"]
        installed = {f.name for f in font_manager.fontManager.ttflist}
        picked = [n for n in preferred if n in installed]
        plt.rcParams["font.family"] = picked if picked else ["DejaVu Serif"]
        plt.rcParams["mathtext.fontset"] = "stix"
    except Exception:
        plt.rcParams["font.family"] = ["DejaVu Serif"]

    return plt


def plot_4algo_convergence(
    algo_data: Dict[str, Dict[str, Any]],
    *,
    sigma: float = 250.0,
    rolling_window: int = 800,
    band_scale: float = 0.3,
    out_path: str = "convergence_4algo.png",
    figsize: Tuple[float, float] = (10.5, 7.0),
    dpi: int = 600,
    show_raw: bool = True,
    raw_alpha: float = 0.12,
    band_alpha: float = 0.20,
    x_label: str = "Training Episodes",
    y_label: str = "Average Episode Reward",
    ylim: Optional[Tuple[float, float]] = None,
) -> str:
    """
     4 

    
      algo_data: key=key"ppo","ddpg"value 
        "episodes": ndarray, "mean": ndarray, "std": ndarrayNone,
        "n_seeds": int, "label": str
      sigma:  sigma
      band_scale:  seed 
      show_raw:  raw 
    """
    plt = _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_facecolor("white")
    ax.grid(True, alpha=0.85)

    
    
    draw_order = ["ddpg", "td3", "sac", "ppo"]

    for zorder_base, algo_key in enumerate(draw_order):
        if algo_key not in algo_data:
            continue

        ad = algo_data[algo_key]
        ep = ad["episodes"]
        mean = ad["mean"]
        std = ad.get("std")
        n_seeds = ad.get("n_seeds", 1)
        label = ad["label"]

        c_line = ALGO_COLORS.get(algo_key, "#333333")
        c_band = ALGO_BAND_COLORS.get(algo_key, "#CCCCCC")
        lw = 3.5 if algo_key == "ppo" else 2.8
        z = 10 + zorder_base * 3

        if n_seeds > 1 and std is not None:
            
            from scipy.ndimage import gaussian_filter1d as _gf1d
            mean_smooth = _gf1d(mean, sigma=sigma, mode='nearest')
            ci_half = 1.96 * std / np.sqrt(n_seeds)
            ci_half_smooth = _gf1d(ci_half, sigma=sigma, mode='nearest')

            if show_raw:
                ax.plot(ep, mean, color=c_band, linewidth=0.6, alpha=raw_alpha, zorder=z)

            ax.fill_between(ep, mean_smooth - ci_half_smooth, mean_smooth + ci_half_smooth,
                            color=c_band, alpha=band_alpha, linewidth=0, zorder=z + 1)
            ax.plot(ep, mean_smooth, color=c_line, linewidth=lw,
                    alpha=0.95, label=f"{label} ({n_seeds}-seed)", zorder=z + 2)
        else:
            
            smoothed, lower, upper = gaussian_smooth_with_band(
                mean, sigma=sigma, rolling_window=rolling_window, band_scale=band_scale
            )

            if show_raw:
                ax.plot(ep, mean, color=c_band, linewidth=0.6, alpha=raw_alpha, zorder=z)

            ax.fill_between(ep, lower, upper,
                            color=c_band, alpha=band_alpha, linewidth=0, zorder=z + 1)
            ax.plot(ep, smoothed, color=c_line, linewidth=lw,
                    alpha=0.95, label=label, zorder=z + 2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    
    try:
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.get_offset_text().set_fontsize(12)
    except Exception:
        pass

    
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(handles[::-1], labels[::-1],
              loc="lower right", framealpha=0.92, edgecolor="#888888")

    
    if ylim is not None:
        ax.set_ylim(ylim)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot_4algo] : {out}")
    return str(out)


# ============================================================

# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    
    parser.add_argument("--ppo-dirs", nargs="+", type=str, default=None,
                        help="PPO (ARIA) run 1  seed")
    parser.add_argument("--ddpg-dirs", nargs="+", type=str, default=None)
    parser.add_argument("--sac-dirs", nargs="+", type=str, default=None)
    parser.add_argument("--td3-dirs", nargs="+", type=str, default=None)

    
    parser.add_argument("--sigma", nargs="+", type=float, default=[250.0],
                        help=" sigma")
    parser.add_argument("--rolling-window", type=int, default=800,
                        help=" 800")
    parser.add_argument("--band-scale", type=float, default=0.3,
                        help=" seed  0.3")

    
    parser.add_argument("--out-dir", type=str,
                        default="figs/convergence_4algo",
                        help="")
    parser.add_argument("--episode-length", type=int, default=80,
                        help="episode  T 80 Average Episode Reward")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-raw", action="store_true",
                        help=" raw ")
    parser.add_argument("--ylim", nargs=2, type=float, default=None,
                        help="y  --ylim -8 8")

    args = parser.parse_args()

    
    algo_configs = {}
    for key in ["ppo", "ddpg", "sac", "td3"]:
        cli_dirs = getattr(args, f"{key}_dirs", None)
        if cli_dirs:
            run_dirs = [Path(d) for d in cli_dirs]
        else:
            run_dirs = _DEFAULT_ALGO_CONFIG[key]["run_dirs"]
        algo_configs[key] = {
            "run_dirs": run_dirs,
            "ckpt_name": _DEFAULT_ALGO_CONFIG[key]["ckpt_name"],
            "label": _DEFAULT_ALGO_CONFIG[key]["label"],
        }

    
    print("=" * 60)
    print("...")
    print("=" * 60)

    algo_data: Dict[str, Dict[str, Any]] = {}
    for key, cfg in algo_configs.items():
        print(f"\n[{key.upper()}]  {len(cfg['run_dirs'])}  seed...")
        try:
            ep, mean, std, n_seeds = load_algo_data(
                key, cfg["run_dirs"], cfg["ckpt_name"], args.episode_length
            )
            algo_data[key] = {
                "episodes": ep,
                "mean": mean,
                "std": std,
                "n_seeds": n_seeds,
                "label": cfg["label"],
            }
        except Exception as e:
            print(f"  [ERROR] {key}: {e}")

    if not algo_data:
        print("[ERROR] ")
        sys.exit(1)

    
    min_ep_count = min(d["episodes"].size for d in algo_data.values())
    for key in algo_data:
        ad = algo_data[key]
        ad["episodes"] = ad["episodes"][:min_ep_count]
        ad["mean"] = ad["mean"][:min_ep_count]
        if ad["std"] is not None:
            ad["std"] = ad["std"][:min_ep_count]

    print(f"\n episode : {min_ep_count}")

    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sigma in args.sigma:
        suffix = "_clean" if args.no_raw else ""
        fname = f"convergence_4algo_sigma{int(sigma)}{suffix}.png"
        print(f"\n{'=' * 60}")
        print(f": sigma={sigma}")
        print(f"{'=' * 60}")

        plot_4algo_convergence(
            algo_data,
            sigma=sigma,
            rolling_window=args.rolling_window,
            band_scale=args.band_scale,
            out_path=str(out_dir / fname),
            dpi=args.dpi,
            show_raw=not args.no_raw,
            ylim=tuple(args.ylim) if args.ylim else None,
        )

    print(f"\n: {out_dir}")


if __name__ == "__main__":
    main()
