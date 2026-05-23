#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5-seedseedTCOM


  1.  run_dir  checkpoint  episode  reward / cost dense 
  2.  episode  episode  mean  std
  3. EMA  TCOM  mean  shaded band 


  - band  ci95 std
  -  intersect episode 
  - y  T Convergence  Average Episode Reward/Cost
  -  metric=reward  all_loads  7 
    span40 / span48 / span200 / span200_raw_ema / span396_raw_ema / span96_ci_only / span396_ci_only


  #  5-seed 
  python scripts/plot_multiseed_convergence.py ^
      --run-dirs runs/paper/seed1 runs/paper/seed2 runs/paper/seed3 runs/paper/seed4 runs/paper/seed5 ^
      --out-dir "figs/TCOM_Convergence evalreward" ^
      --ema-span 200 ^
      --metric reward
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

from src.utils.plot_smoothing import ema_smooth

# ============================================================

# ============================================================


SINGLE_ALGO_BAND_COLOR = "#9BC1FF"   
SINGLE_ALGO_LINE_COLOR = "#2D5AA0"   

# ============================================================

# ============================================================

def _load_dense_reward_from_ckpt(run_dir: Path, ckpt_name: str = "ckpt_final.pt",
                                  metric: str = "reward") -> Tuple[np.ndarray, np.ndarray]:
    """
     run_dir/checkpoints/<ckpt_name>  episode  dense 

     (episodes, values) 1-D ndarray
    metric: "reward"  train_reward_total_ep_dense
            "cost"    train_cost_ep_dense
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    
    ckpt_path = ckpt_dir / ckpt_name
    if not ckpt_path.exists():
        fallback = "ckpt_best.pt" if ckpt_name == "ckpt_final.pt" else "ckpt_final.pt"
        ckpt_path = ckpt_dir / fallback
    if not ckpt_path.exists():
        print(f"[WARN]  checkpoint: {ckpt_dir}")
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    import torch
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict):
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    
    hist = obj.get("history", obj)

    key_map = {
        "reward": "train_reward_total_ep_dense",
        "cost": "train_cost_ep_dense",
    }
    y_key = key_map.get(metric, f"train_{metric}_ep_dense")
    y_raw = hist.get(y_key, [])
    y = np.asarray(y_raw, dtype=np.float64).reshape(-1)

    
    
    ep_raw = hist.get("train_episodes_dense", [])
    if len(ep_raw) == 0:
        ep_raw = hist.get("episodes", hist.get("train_episodes", []))
    if len(ep_raw) == 0:
        ep_raw = np.arange(1, len(y) + 1, dtype=np.int64)
    ep = np.asarray(ep_raw, dtype=np.int64).reshape(-1)

    
    n = min(ep.size, y.size)
    ep, y = ep[:n], y[:n]

    
    mask = np.isfinite(y)
    return ep[mask], y[mask]


# ============================================================

# ============================================================

def aggregate_multiseed(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    align_mode: str = "intersect",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     (episodes, values)  mean  std

    align_mode:
      "truncate"  
      "intersect"   episode 

     (episodes, mean, std)
    """
    if not curves:
        empty = np.array([], dtype=np.float64)
        return np.array([], dtype=np.int64), empty, empty

    
    curves = [(e, v) for e, v in curves if e.size > 0 and v.size > 0]
    if not curves:
        empty = np.array([], dtype=np.float64)
        return np.array([], dtype=np.int64), empty, empty

    if align_mode == "truncate":
        min_len = min(v.size for _, v in curves)
        mat = np.stack([v[:min_len] for _, v in curves], axis=0)  # (N_seeds, min_len)
        ep = curves[0][0][:min_len]
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(min_len)
        return ep, mean, std

    
    ep_sets = [set(e.tolist()) for e, _ in curves]
    common = sorted(set.intersection(*ep_sets))
    if len(common) < 10:
        
        print("[plot_multiseed][WARN] intersect  truncate ")
        return aggregate_multiseed(curves, align_mode="truncate")

    ep_grid = np.array(common, dtype=np.int64)
    n_seeds = len(curves)
    mat = np.full((n_seeds, len(ep_grid)), np.nan, dtype=np.float64)
    for si, (ep, val) in enumerate(curves):
        ep_to_val = dict(zip(ep.tolist(), val.tolist()))
        for gi, e in enumerate(ep_grid):
            if e in ep_to_val:
                mat[si, gi] = ep_to_val[e]

    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=1) if n_seeds > 1 else np.zeros(len(ep_grid))
    return ep_grid, mean, std


# ============================================================

# ============================================================

def _setup_tcom_style():
    """ IEEE TCOM  plt """
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.0,
        "grid.color": "0.9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.dpi": 600,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "axes.unicode_minus": False,
    })
    
    try:
        from matplotlib import font_manager
        preferred = ["Times New Roman", "Times", "Microsoft YaHei", "SimHei", "DejaVu Serif"]
        installed = {f.name for f in font_manager.fontManager.ttflist}
        picked = [n for n in preferred if n in installed]
        plt.rcParams["font.family"] = picked if picked else ["DejaVu Serif"]
    except Exception:
        plt.rcParams["font.family"] = ["DejaVu Serif"]

    return plt


def _apply_episode_xaxis_sci(ax) -> None:
    """ episode  10^4"""
    try:
        from matplotlib.ticker import ScalarFormatter

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        formatter.set_useOffset(True)
        ax.xaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(9)
    except Exception:
        pass


def plot_multiseed_convergence(
    algo_data: List[Dict[str, Any]],
    *,
    out_path: str = "TCOM_Convergence.png",
    ema_span: int = 200,
    band_mode: str = "std",
    band_scale: float = 1.0,
    y_label: str = "Average Episode Reward",
    x_label: str = "Training Episodes",
    title: str = "",
    episode_length: int = 80,
    divide_by_ep_len: bool = True,
    figsize: Tuple[float, float] = (8.0, 3.8),
    dpi: int = 600,
    show_raw_mean: bool = True,
    raw_alpha: float = 0.18,
    raw_linewidth: float = 0.8,
    band_alpha: float = 0.22,
    line_linewidth: float = 2.4,
    line_alpha: float = 0.95,
    legend_loc: str = "lower right",
) -> str:
    """
     TCOM  seed 

    
      algo_data:  dict
        {
          "name": "PPO (proposed)",       # 
          "episodes": np.ndarray,         # episode 
          "mean": np.ndarray,             #  episode 
          "std": np.ndarray,              #  episode 
          "n_seeds": int,                 # seed 
          "color_idx": int,               # 0=, 1=, ...
        }
      band_mode: "std"  mean  band_scale  std
                 "ci95"  mean  1.96  std / n
                 "stderr"  mean  std / n
      divide_by_ep_len: True  Y  episode_lengthAverage Episode Reward

    
    """
    plt = _setup_tcom_style()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.set_facecolor("white")
    ax.grid(True, color="0.88", linestyle="--", linewidth=0.6, alpha=0.9)
    for sp in ax.spines.values():
        sp.set_color("#4D6A70")
        sp.set_linewidth(0.9)

    for algo in algo_data:
        ep = np.asarray(algo["episodes"], dtype=np.float64)
        mean = np.asarray(algo["mean"], dtype=np.float64)
        std = np.asarray(algo["std"], dtype=np.float64)
        n_seeds = int(algo.get("n_seeds", 1))
        name = str(algo.get("name", "PPO"))
        _ = int(algo.get("color_idx", 0))
        c_line = SINGLE_ALGO_LINE_COLOR
        c_band = SINGLE_ALGO_BAND_COLOR

        
        divisor = float(episode_length) if divide_by_ep_len and episode_length > 1 else 1.0
        mean_plot = mean / divisor
        std_plot = std / divisor

        
        if band_mode == "ci95":
            half = 1.96 * std_plot / max(np.sqrt(n_seeds), 1.0)
        elif band_mode == "stderr":
            half = std_plot / max(np.sqrt(n_seeds), 1.0)
        else:  # "std"
            half = band_scale * std_plot

        
        mean_ema = ema_smooth(mean_plot, span=ema_span, adjust=False)
        half_ema = ema_smooth(half, span=ema_span, adjust=False)

        
        if show_raw_mean:
            ax.plot(ep, mean_plot, color=c_band, linewidth=raw_linewidth,
                    alpha=raw_alpha, zorder=2)

        
        ax.fill_between(ep, mean_ema - half_ema, mean_ema + half_ema,
                        color=c_band, alpha=band_alpha, linewidth=0, zorder=3)

        
        label_str = f"{name} ({n_seeds}-seed mean)" if n_seeds > 1 else name
        ax.plot(ep, mean_ema, color=c_line, linewidth=line_linewidth,
                alpha=line_alpha, label=label_str, zorder=5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_episode_xaxis_sci(ax)
    if title:
        ax.set_title(title, pad=8, fontweight="bold")

    ax.legend(loc=legend_loc, framealpha=0.92, edgecolor="0.8")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot_multiseed] : {out}")
    return str(out)


# ============================================================

# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="5-seed TCOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    
    parser.add_argument("--run-dirs", nargs="+", type=str, default=None,
                        help=" run_dir  seed")
    parser.add_argument("--algo-name", type=str, default="PPO (proposed)",
                        help="")

    
    parser.add_argument("--out-dir", type=str, default="figs/TCOM_Convergence evalreward",
                        help="")
    parser.add_argument("--ema-span", type=int, default=200,
                        help="EMA  200")
    parser.add_argument("--metric", type=str, default="reward",
                        choices=["reward", "cost"],
                        help="reward  cost")
    parser.add_argument("--ckpt-name", type=str, default="ckpt_final.pt",
                        help="checkpoint ")
    parser.add_argument("--episode-length", type=int, default=80,
                        help="episode  T  Average Episode Reward/Cost")
    parser.add_argument("--no-divide", action="store_true",
                        help=" episode_length Total Rewards/Paper Cost")
    parser.add_argument("--band-mode", type=str, default="ci95",
                        choices=["std", "ci95", "stderr"],
                        help="band std / ci95 / stderr ci95")
    parser.add_argument("--band-scale", type=float, default=0.5,
                        help="band_mode=std  0.5 0.5")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--title", type=str, default="",
                        help="")
    parser.add_argument("--legend-loc", type=str, default="lower right")
    parser.add_argument("--align-mode", type=str, default="intersect",
                        choices=["truncate", "intersect"],
                        help=" seed  intersect")

    
    parser.add_argument("--extra-spans", type=str, default="48",
                        help=" EMA span '48,96'")
    parser.add_argument("--match-train-all-loads", type=int, default=1,
                        help="metric=reward  all_loads  7  1")

    args = parser.parse_args()

    
    if not args.run_dirs:
        parser.error(" --run-dirs")
    algo_groups: List[Dict[str, Any]] = [{"name": args.algo_name, "run_dirs": [Path(d) for d in args.run_dirs]}]

    
    divide_by_ep = not bool(args.no_divide)
    print(f"[plot_multiseed] : {args.metric}, EMA span: {args.ema_span}, "
          f"band: {args.band_mode}(scale={args.band_scale}), align={args.align_mode}, "
          f"divide_by_episode={divide_by_ep}")

    all_algo_data: List[Dict[str, Any]] = []

    for gi, group in enumerate(algo_groups):
        name = group["name"]
        run_dirs = group["run_dirs"]
        print(f"\n[{name}]  {len(run_dirs)}  seed...")

        curves = []
        for rd in run_dirs:
            ep, val = _load_dense_reward_from_ckpt(Path(rd), ckpt_name=args.ckpt_name,
                                                    metric=args.metric)
            if ep.size > 0:
                print(f"  {rd.name}: {ep.size} episodes, "
                      f"range [{val.min():.1f}, {val.max():.1f}]")
                curves.append((ep, val))
            else:
                print(f"  {rd.name}: [WARN] ")

        if not curves:
            print(f"  [ERROR] {name} ")
            continue

        ep_agg, mean_agg, std_agg = aggregate_multiseed(curves, align_mode=args.align_mode)
        print(f"  : {ep_agg.size} episodes, {len(curves)} seeds, "
              f"mean=[{mean_agg.min():.1f}, {mean_agg.max():.1f}]")

        all_algo_data.append({
            "name": name,
            "episodes": ep_agg,
            "mean": mean_agg,
            "std": std_agg,
            "n_seeds": len(curves),
            "color_idx": gi,
        })

    if not all_algo_data:
        print("[ERROR] ")
        sys.exit(1)

    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_label = "Average Episode Reward" if divide_by_ep else "Total Rewards"
    if args.metric == "cost":
        y_label = "Average Episode Cost" if divide_by_ep else "Paper Cost"

    
    
    
    use_match_profile = bool(int(args.match_train_all_loads) == 1) and (str(args.metric).strip().lower() == "reward")
    if use_match_profile:
        
        raw_specs = [
            (40, "Convergence_RewardTotal_span40.png"),
            (48, "Convergence_RewardTotal_span48.png"),
            (200, "Convergence_RewardTotal_span200.png"),
            (200, "Convergence_RewardTotal_span200_raw_ema.png"),
            (396, "Convergence_RewardTotal_span396_raw_ema.png"),
        ]
        for span, fname in raw_specs:
            plot_multiseed_convergence(
                all_algo_data,
                out_path=str(out_dir / fname),
                ema_span=int(span),
                band_mode=args.band_mode,
                band_scale=args.band_scale,
                y_label=y_label,
                title=args.title,
                episode_length=args.episode_length,
                divide_by_ep_len=divide_by_ep,
                dpi=args.dpi,
                legend_loc=args.legend_loc,
                show_raw_mean=True,
            )

        
        ci_only_specs = [
            (96, "Convergence_RewardTotal_span96_ci_only.png"),
            (396, "Convergence_RewardTotal_span396_ci_only.png"),
        ]
        for span, fname in ci_only_specs:
            plot_multiseed_convergence(
                all_algo_data,
                out_path=str(out_dir / fname),
                ema_span=int(span),
                band_mode=args.band_mode,
                band_scale=args.band_scale,
                y_label=y_label,
                title=args.title,
                episode_length=args.episode_length,
                divide_by_ep_len=divide_by_ep,
                dpi=args.dpi,
                legend_loc=args.legend_loc,
                show_raw_mean=False,
            )
    else:
        all_spans = [args.ema_span]
        if args.extra_spans:
            for s in args.extra_spans.split(","):
                s = s.strip()
                if not s:
                    continue
                try:
                    sval = int(s)
                except Exception:
                    print(f"[plot_multiseed][WARN] extra span : {s}")
                    continue
                if sval != args.ema_span:
                    all_spans.append(sval)

        for span in all_spans:
            suffix = f"_span{span}"
            fname = f"TCOM_Convergence_{args.metric}{suffix}.png"

            plot_multiseed_convergence(
                all_algo_data,
                out_path=str(out_dir / fname),
                ema_span=span,
                band_mode=args.band_mode,
                band_scale=args.band_scale,
                y_label=y_label,
                title=args.title,
                episode_length=args.episode_length,
                divide_by_ep_len=divide_by_ep,
                dpi=args.dpi,
                legend_loc=args.legend_loc,
            )

    print(f"\n[plot_multiseed] : {out_dir}")


if __name__ == "__main__":
    main()
