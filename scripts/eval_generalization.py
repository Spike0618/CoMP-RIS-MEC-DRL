#!/usr/bin/env python3
"""
H.13-S2  eval_policy_joint 


1. fixed env
2. random env
3.  eval_summary paper_cost / delay / energy 
4.  run_dir
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
import yaml
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.phaseg_summary import upsert_phaseg_eval_summary


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
})

_preferred_fonts = [
    "Microsoft YaHei",
    "SimSun",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
_installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [x for x in _preferred_fonts if x in _installed_fonts] or ["DejaVu Sans"]


METRICS = ["paper_cost", "mean_T_off", "mean_energy"]
METRIC_LABELS = {
    "paper_cost": "",
    "mean_T_off": " (s)",
    "mean_energy": " (J)",
}

METHOD_LABELS = {
    "ppo": "ARIA",
    "balanced": "",
    "greedy_delay": "",
    "greedy_energy": "",
    "myopic_optimization": "",
    "ddpg": "DDPG",
    "sac": "SAC",
    "td3": "TD3",
    "a2c": "A2C",
}

METHOD_COLORS = {
    "ppo": "#1CA9A6",
    "balanced": "#1F77B4",
    "greedy_delay": "#F2B632",
    "greedy_energy": "#F08A24",
    "myopic_optimization": "#7B68EE",
    "ddpg": "#4DA3FF",
    "sac": "#FF9E6D",
    "td3": "#C9A227",
    "a2c": "#6A994E",
}

DEFAULT_EVAL_LOADS_STR = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
SUPPORTED_USER_MOBILITY_MODES = {"static", "slow", "mixed"}


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _split_csv_items(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _split_csv_floats(text: str) -> List[float]:
    return [float(x) for x in _split_csv_items(text)]


def _is_ten_load_protocol(loads: List[float]) -> bool:
    arr = np.asarray(list(loads), dtype=np.float64).reshape(-1)
    if arr.size != 10:
        return False
    tgt = np.asarray([0.1 * i for i in range(1, 11)], dtype=np.float64)
    return bool(np.allclose(arr, tgt, atol=1e-9, rtol=0.0))


def _assert_ten_load_protocol(loads: List[float]) -> None:
    if not _is_ten_load_protocol(loads):
        raise ValueError(f"0.1~1.0={loads}")


def _align_random_env_action_contract(
    fixed_env_yaml: Path,
    random_env_yaml: Path,
    out_yaml: Path,
) -> Path:
    """
    fixed/randomcheckpoint

    Phase T checkpoint20Dyaml60D
    """
    with open(fixed_env_yaml, "r", encoding="utf-8") as f:
        fixed_cfg = yaml.safe_load(f) or {}
    with open(random_env_yaml, "r", encoding="utf-8") as f:
        random_cfg = yaml.safe_load(f) or {}

    fixed_obj = fixed_cfg.get("objective", {}) or {}
    random_obj = random_cfg.get("objective", {}) or {}
    fixed_shaping = fixed_cfg.get("shaping", {}) or {}
    random_shaping = random_cfg.get("shaping", {}) or {}

    
    objective_contract_keys = (
        "action_space_mode",
        "action_score_mode",
        "association_mode",
        "assoc_soft_temp",
        "comp_score_temp",
        "comp_score_eval_hard",
        "theta_mode",
        "theta_softmax_tau",
        "comp_temp_anneal_enable",
        "comp_temp_stage_0",
        "comp_temp_stage_1",
        "comp_temp_stage_2",
        "reward_design",
        "paper_cost_weight",
        "relative_baseline_mode",
        "relative_baseline_position_mode",
        "lambda_v",
        "vio_max_expected",
        "lambda_traj_guide",
        "use_fixed_normalization",
        "norm_update_interval",
        "reward_clip_range",
        "reward_ema_beta",
    )
    shaping_contract_keys = (
        "comp_gate_ratio",
        "weight_mode",
        "assoc_gain_mix",
        "use_fixed_normalization",
        "norm_update_interval",
        "norm_ema_beta",
        "norm_clip",
    )

    changed: List[str] = []
    for k in objective_contract_keys:
        if k not in fixed_obj:
            continue
        v_fixed = fixed_obj.get(k)
        if random_obj.get(k) != v_fixed:
            random_obj[k] = copy.deepcopy(v_fixed)
            changed.append(f"objective.{k}")
    for k in shaping_contract_keys:
        if k not in fixed_shaping:
            continue
        v_fixed = fixed_shaping.get(k)
        if random_shaping.get(k) != v_fixed:
            random_shaping[k] = copy.deepcopy(v_fixed)
            changed.append(f"shaping.{k}")

    
    scenario = random_cfg.get("scenario", {}) or {}
    mode_raw = str(scenario.get("user_mobility_mode", "static")).strip().lower()
    if mode_raw == "semi_random":
        scenario["user_mobility_mode"] = "mixed"
        changed.append("scenario.user_mobility_mode:semi_random->mixed")
    elif mode_raw not in SUPPORTED_USER_MOBILITY_MODES:
        scenario["user_mobility_mode"] = "mixed"
        changed.append(f"scenario.user_mobility_mode:{mode_raw}->mixed")

    if not changed:
        return random_env_yaml

    random_cfg["objective"] = random_obj
    random_cfg["shaping"] = random_shaping
    random_cfg["scenario"] = scenario
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(random_cfg, f, allow_unicode=True, sort_keys=False)
    print(
        f"[] random keys={changed} -> {out_yaml}",
        flush=True,
    )
    return out_yaml


def _build_default_random_env_from_fixed(
    fixed_env_yaml: Path,
    out_yaml: Path,
) -> Path:
    """
     random_env_yaml  fixed 

    
    - / _align_random_env_action_contract 
    - 
    """
    with open(fixed_env_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    sc = cfg.get("scenario", {}) or {}
    task = cfg.get("task", {}) or {}

    
    sc["user_mobility_mode"] = "mixed"
    sc["user_mobility_prob"] = float(max(float(sc.get("user_mobility_prob", 0.0)), 0.30))
    sc["user_speed"] = float(max(float(sc.get("user_speed", 0.0)), 0.50))
    sc["user_position_jitter"] = float(max(float(sc.get("user_position_jitter", 0.0)), 50.0))
    sc["user_position_refresh_interval"] = int(max(int(sc.get("user_position_refresh_interval", 1)), 1))

    task["task_arrival_jitter"] = float(max(float(task.get("task_arrival_jitter", 0.0)), 0.25))
    task["task_size_jitter"] = float(max(float(task.get("task_size_jitter", 0.0)), 0.30))

    cfg["scenario"] = sc
    cfg["task"] = task

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[] random_env_yamlfixed: {out_yaml}", flush=True)
    return out_yaml


def _enforce_phaseg_refresh_interval(
    env_yaml: Path,
    out_yaml: Path,
    *,
    tag: str,
) -> Path:
    """
    PhaseG user_position_refresh_interval=1
    """
    with open(env_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"env_yamldict{env_yaml}")

    changed = False
    scenario = cfg.get("scenario", {}) or {}
    if not isinstance(scenario, dict):
        scenario = {}

    if int(scenario.get("user_position_refresh_interval", 1)) != 1:
        scenario["user_position_refresh_interval"] = 1
        changed = True

    if int(cfg.get("user_position_refresh_interval", 1)) != 1:
        cfg["user_position_refresh_interval"] = 1
        changed = True

    cfg["scenario"] = scenario
    if not changed:
        return env_yaml

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(
        f"[] {tag} user_position_refresh_interval=1 -> {out_yaml}",
        flush=True,
    )
    return out_yaml


def _resolve_ppo_ckpt(run_dir: Path, ckpt_override: str) -> Path:
    """PPO checkpointfixed/random"""
    ckpt_override = str(ckpt_override or "").strip()
    if ckpt_override:
        p = Path(ckpt_override).resolve()
        if not p.exists():
            raise FileNotFoundError(f"ckpt_path: {p}")
        return p

    cand_dirs = [run_dir / "checkpoints", run_dir / "ckpts", run_dir / "ckpt"]
    cand_dirs = [d for d in cand_dirs if d.exists()]
    if not cand_dirs:
        raise FileNotFoundError(f"checkpoint: {run_dir}")

    
    for d in cand_dirs:
        p = d / "ckpt_best.pt"
        if p.exists():
            return p
    raise FileNotFoundError(
        f" ckpt_best.pt best checkpoint {cand_dirs}"
    )


def _run_joint_eval(
    run_dir: Path,
    eval_ts: str,
    env_yaml: Path,
    methods: List[str],
    eval_loads: List[float],
    n_seeds: int,
    ep_per: int,
    seed0: int,
    deterministic: bool,
    device: str,
    ckpt_path: Optional[Path],
) -> Path:
    """ eval_policy_joint  eval_dir"""
    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir)
    env["ENV_CFG"] = str(env_yaml)
    env["EVAL_METHODS"] = ",".join(methods)
    env["EVAL_NSEEDS"] = str(int(max(1, n_seeds)))
    env["EVAL_EP_PER"] = str(int(max(1, ep_per)))
    env["EVAL_SEED0"] = str(int(seed0))
    env["EVAL_DETERMINISTIC"] = "1" if deterministic else "0"
    env["EVAL_RESUME"] = "0"
    env["EVAL_PLOT_CI"] = "0"
    
    env["EVAL_PLOT_LOGY"] = "0"
    env["EVAL_PLOT_ZOOM"] = "0"
    if device.strip():
        env["EVAL_DEVICE"] = device.strip()
    if ckpt_path is not None:
        env["CKPT_PATH"] = str(ckpt_path)

    cmd = [
        sys.executable,
        "scripts/eval_policy_joint.py",
        "--run_dir",
        str(run_dir),
        "--eval_ts",
        eval_ts,
        "--eval_loads",
        ",".join(str(x) for x in eval_loads),
        "--strict_ten_loads",
        "1",
        "--require_explicit_ckpt",
        "1",
    ]
    if ckpt_path is not None:
        cmd.extend(["--ckpt_path", str(ckpt_path)])
    print("\n" + "=" * 88)
    print(f"[]  eval_ts={eval_ts}")
    print(f"[] env_yaml={env_yaml}")
    print(f"[] methods={methods} loads={eval_loads} n_seeds={n_seeds} ep_per={ep_per}")
    print(f"[] cmd={' '.join(cmd)}")
    print("=" * 88)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)

    eval_dir = run_dir / "evals" / eval_ts
    summary_path = eval_dir / f"eval_summary_{eval_ts}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f": {summary_path}")
    return eval_dir


def _load_summary(eval_dir: Path) -> Dict[str, object]:
    summary_path = eval_dir / f"eval_summary_{eval_dir.name}.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _extract_method_metrics(summary: Dict[str, object], methods: List[str], variant: str = "full") -> Dict[str, Dict[str, Dict[str, float]]]:
    """
     eval_summary 

    
    {
      method: {
        metric: {"mean":..., "ci_mean":..., "n_loads":...}
      }
    }
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    series = summary.get("series", {})
    if not isinstance(series, dict):
        return out

    for method in methods:
        key = f"{variant}|{method}"
        node = series.get(key, {})
        method_stats: Dict[str, Dict[str, float]] = {}
        if not isinstance(node, dict):
            node = {}

        for metric in METRICS:
            mnode = node.get(metric, {})
            if not isinstance(mnode, dict):
                mnode = {}
            y = mnode.get("y", [])
            e = mnode.get("e", [])
            y_list = [float(v) for v in y] if isinstance(y, list) else []
            e_list = [float(v) for v in e] if isinstance(e, list) else []
            method_stats[metric] = {
                "mean": _safe_mean(y_list),
                "ci_mean": _safe_mean(e_list),
                "n_loads": float(len(y_list)),
            }
        out[method] = method_stats
    return out


def _calc_degradation(
    fixed_stats: Dict[str, Dict[str, Dict[str, float]]],
    random_stats: Dict[str, Dict[str, Dict[str, float]]],
    methods: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """"""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in methods:
        out[method] = {}
        for metric in METRICS:
            fixed_v = float(fixed_stats.get(method, {}).get(metric, {}).get("mean", float("nan")))
            random_v = float(random_stats.get(method, {}).get(metric, {}).get("mean", float("nan")))
            if np.isfinite(fixed_v):
                denom = max(abs(fixed_v), 1e-9)
                deg = (random_v - fixed_v) / denom * 100.0 if np.isfinite(random_v) else float("nan")
            else:
                deg = float("nan")
            out[method][metric] = {
                "fixed": fixed_v,
                "random": random_v,
                "degradation_pct": float(deg),
            }
    return out


def _extract_method_metric_curves(
    summary: Dict[str, object],
    methods: List[str],
    variant: str = "full",
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """-"""
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    series = summary.get("series", {})
    if not isinstance(series, dict):
        return out

    for method in methods:
        out[method] = {}
        key = f"{variant}|{method}"
        node = series.get(key, {})
        if not isinstance(node, dict):
            node = {}
        for metric in METRICS:
            mnode = node.get(metric, {})
            if not isinstance(mnode, dict):
                mnode = {}
            x = [float(v) for v in (mnode.get("x", []) if isinstance(mnode.get("x", []), list) else [])]
            y = [float(v) for v in (mnode.get("y", []) if isinstance(mnode.get("y", []), list) else [])]
            out[method][metric] = {"x": x, "y": y}
    return out


def _bin_level(v: float, low_th: float, high_th: float) -> str:
    """ low/mid/high """
    if not np.isfinite(v):
        return "unknown"
    if v < low_th:
        return "low"
    if v < high_th:
        return "mid"
    return "high"


def _extract_scenario_cluster_profile(env_yaml: Path, tag: str) -> Dict[str, object]:
    """env/"""
    profile: Dict[str, object] = {
        "tag": str(tag),
        "env_yaml": str(env_yaml),
        "edge_user_ratio": float("nan"),
        "direct_blockage_prob": float("nan"),
        "edge_user_ratio_bin": "unknown",
        "direct_blockage_prob_bin": "unknown",
        "cluster_label": "unknown",
    }
    if not env_yaml.exists():
        return profile
    try:
        with open(env_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return profile
    if not isinstance(cfg, dict):
        return profile

    scenario = cfg.get("scenario", {}) or {}
    if not isinstance(scenario, dict):
        scenario = {}
    edge_ratio = float(scenario.get("t6_edge_user_ratio", np.nan))
    block_prob = float(scenario.get("t6_direct_blockage_prob", np.nan))
    edge_bin = _bin_level(edge_ratio, low_th=0.25, high_th=0.50)
    block_bin = _bin_level(block_prob, low_th=0.12, high_th=0.22)

    profile.update(
        {
            "edge_user_ratio": float(edge_ratio),
            "direct_blockage_prob": float(block_prob),
            "edge_user_ratio_bin": str(edge_bin),
            "direct_blockage_prob_bin": str(block_bin),
            "cluster_label": f"edge:{edge_bin}|block:{block_bin}",
        }
    )
    return profile


def _calc_load_band_degradation(
    fixed_curves: Dict[str, Dict[str, Dict[str, List[float]]]],
    random_curves: Dict[str, Dict[str, Dict[str, List[float]]]],
    methods: List[str],
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """
    
    - all/low/mid/high/worst_1.0
    - 
    """
    bands = [
        ("all", 0.0, 1.0),
        ("low", 0.1, 0.3),
        ("mid", 0.4, 0.6),
        ("high", 0.7, 1.0),
        ("worst_1.0", 1.0, 1.0),
    ]
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for method in methods:
        out[method] = {}
        for metric in METRICS:
            fx = np.asarray((fixed_curves.get(method, {}).get(metric, {}) or {}).get("x", []), dtype=np.float64).reshape(-1)
            fy = np.asarray((fixed_curves.get(method, {}).get(metric, {}) or {}).get("y", []), dtype=np.float64).reshape(-1)
            rx = np.asarray((random_curves.get(method, {}).get(metric, {}) or {}).get("x", []), dtype=np.float64).reshape(-1)
            ry = np.asarray((random_curves.get(method, {}).get(metric, {}) or {}).get("y", []), dtype=np.float64).reshape(-1)

            f_map = {round(float(l), 6): float(v) for l, v in zip(fx.tolist(), fy.tolist())}
            r_map = {round(float(l), 6): float(v) for l, v in zip(rx.tolist(), ry.tolist())}
            common_loads = sorted(set(f_map.keys()) & set(r_map.keys()))

            per_load: List[Dict[str, float]] = []
            for lk in common_loads:
                fv = float(f_map[lk])
                rv = float(r_map[lk])
                deg = (rv - fv) / max(abs(fv), 1e-9) * 100.0 if np.isfinite(fv) and np.isfinite(rv) else float("nan")
                per_load.append(
                    {
                        "load": float(lk),
                        "fixed": float(fv),
                        "random": float(rv),
                        "degradation_pct": float(deg),
                    }
                )

            band_stats: Dict[str, Dict[str, float]] = {}
            for name, lo, hi in bands:
                vals = [row for row in per_load if (float(row["load"]) >= float(lo) - 1e-9 and float(row["load"]) <= float(hi) + 1e-9)]
                fixed_mean = _safe_mean([float(v["fixed"]) for v in vals])
                random_mean = _safe_mean([float(v["random"]) for v in vals])
                if np.isfinite(fixed_mean) and np.isfinite(random_mean):
                    deg_mean = (random_mean - fixed_mean) / max(abs(fixed_mean), 1e-9) * 100.0
                else:
                    deg_mean = float("nan")
                band_stats[name] = {
                    "fixed_mean": float(fixed_mean),
                    "random_mean": float(random_mean),
                    "degradation_pct": float(deg_mean),
                }

            out[method][metric] = {
                "per_load_degradation": per_load,
                "band_degradation": band_stats,
            }
    return out


def _write_cluster_report_csv(
    out_path: Path,
    band_degradation: Dict[str, Dict[str, Dict[str, object]]],
    methods: List[str],
) -> None:
    """C9CSV"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "band", "fixed_mean", "random_mean", "degradation_pct"])
        for method in methods:
            for metric in METRICS:
                bands = (
                    (band_degradation.get(method, {}).get(metric, {}) or {}).get("band_degradation", {})
                    if isinstance(band_degradation.get(method, {}).get(metric, {}), dict)
                    else {}
                )
                if not isinstance(bands, dict):
                    continue
                for band_name, vals in bands.items():
                    if not isinstance(vals, dict):
                        continue
                    writer.writerow(
                        [
                            method,
                            metric,
                            str(band_name),
                            float(vals.get("fixed_mean", np.nan)),
                            float(vals.get("random_mean", np.nan)),
                            float(vals.get("degradation_pct", np.nan)),
                        ]
                    )


def _plot_cluster_breakdown(
    out_fig: Path,
    methods: List[str],
    scenario_profiles: Dict[str, Dict[str, object]],
    band_degradation: Dict[str, Dict[str, Dict[str, object]]],
) -> None:
    """C9 + """
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax_cluster, ax_curve = axes

    
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    colors: List[str] = []
    for tag, color in (("fixed", "#1f77b4"), ("random", "#d62728")):
        p = scenario_profiles.get(tag, {})
        x = float((p.get("edge_user_ratio", np.nan) if isinstance(p, dict) else np.nan))
        y = float((p.get("direct_blockage_prob", np.nan) if isinstance(p, dict) else np.nan))
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
            labels.append(f"{tag}: {str((p.get('cluster_label', 'unknown') if isinstance(p, dict) else 'unknown'))}")
            colors.append(color)
    if len(xs) > 0:
        ax_cluster.scatter(xs, ys, s=85, c=colors, alpha=0.85, edgecolors="black", linewidths=0.8)
        for x, y, txt in zip(xs, ys, labels):
            ax_cluster.text(x + 0.01, y + 0.005, txt, fontsize=8, ha="left", va="bottom")
    ax_cluster.set_xlabel("Edge User Ratio (t6_edge_user_ratio)")
    ax_cluster.set_ylabel("Direct Blockage Prob (t6_direct_blockage_prob)")
    ax_cluster.set_title("Scenario Cluster (Fixed vs Random)")
    ax_cluster.grid(True, alpha=0.25)

    
    target_method = "ppo" if "ppo" in methods else (methods[0] if methods else "")
    for metric, color in (("paper_cost", "#1f77b4"), ("mean_T_off", "#ff7f0e"), ("mean_energy", "#2ca02c")):
        rows = (
            (band_degradation.get(target_method, {}).get(metric, {}) or {}).get("per_load_degradation", [])
            if isinstance(band_degradation.get(target_method, {}).get(metric, {}), dict)
            else []
        )
        if not isinstance(rows, list) or len(rows) <= 0:
            continue
        rows = sorted(rows, key=lambda r: float((r or {}).get("load", 0.0)))
        x = np.asarray([float((r or {}).get("load", np.nan)) for r in rows], dtype=np.float64)
        y = np.asarray([float((r or {}).get("degradation_pct", np.nan)) for r in rows], dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if np.sum(m) <= 0:
            continue
        ax_curve.plot(x[m], y[m], marker="o", linewidth=1.8, color=color, label=METRIC_LABELS.get(metric, metric))
    ax_curve.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax_curve.set_xlabel("Load")
    ax_curve.set_ylabel("Degradation (%)")
    ax_curve.set_title(f"Load-wise Degradation ({METHOD_LABELS.get(target_method, target_method)})")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=400, bbox_inches="tight")
    plt.close()


def _build_required_degradation_fields(
    degradation: Dict[str, Dict[str, Dict[str, float]]],
    methods: List[str],
) -> Dict[str, Dict[str, float]]:
    """"""
    out: Dict[str, Dict[str, float]] = {}
    for method in methods:
        node = degradation.get(method, {}) if isinstance(degradation, dict) else {}
        out[method] = {
            "paper_cost_degradation_pct": float(
                (node.get("paper_cost", {}) or {}).get("degradation_pct", float("nan"))
            ),
            "delay_degradation_pct": float(
                (node.get("mean_T_off", {}) or {}).get("degradation_pct", float("nan"))
            ),
            "energy_degradation_pct": float(
                (node.get("mean_energy", {}) or {}).get("degradation_pct", float("nan"))
            ),
        }
    return out


def _plot_comparison(
    degradation: Dict[str, Dict[str, Dict[str, float]]],
    methods: List[str],
    fig_path: Path,
) -> None:
    """ vs """
    n_metrics = int(max(1, len(METRICS)))
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.2 * n_metrics, 4.8))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = list(np.asarray(axes).reshape(-1))
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.36
    legend_handles = None

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        fixed_vals: List[float] = []
        random_vals: List[float] = []
        colors: List[str] = []
        for m in methods:
            rec = degradation.get(m, {}).get(metric, {})
            fixed_vals.append(float(rec.get("fixed", np.nan)))
            random_vals.append(float(rec.get("random", np.nan)))
            colors.append(METHOD_COLORS.get(m, "#999999"))

        bar1 = ax.bar(x - width / 2.0, fixed_vals, width=width, color=colors, alpha=0.85, label="")
        bar2 = ax.bar(x + width / 2.0, random_vals, width=width, color=colors, alpha=0.45, hatch="//", label="")
        if legend_handles is None:
            legend_handles = (bar1[0], bar2[0])

        
        all_h = np.asarray(fixed_vals + random_vals, dtype=np.float64)
        all_h = all_h[np.isfinite(all_h)]
        if all_h.size > 0:
            y_max = float(np.max(all_h))
            ax.set_ylim(0.0, y_max * 1.14)

        for i, b in enumerate(bar2):
            h = float(b.get_height())
            rec = degradation.get(methods[i], {}).get(metric, {})
            d = float(rec.get("degradation_pct", np.nan))
            if np.isfinite(h) and np.isfinite(d):
                txt = f"{d:+.1f}%"
                c = "red" if d > 10.0 else ("#B06A00" if d > 0.0 else "green")
                ax.text(b.get_x() + b.get_width() / 2.0, h, txt, ha="center", va="bottom", fontsize=8, color=c)

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15, ha="right")
        ax.grid(True, alpha=0.25, axis="y")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["", ""],
            loc="upper center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(0.5, 1.02),
        )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close()


def _plot_extended_comparison_bundle(
    degradation: Dict[str, Dict[str, Dict[str, float]]],
    methods: List[str],
    out_dir: Path,
) -> Dict[str, str]:
    """
    
    1) 2) 
    3) PPO4) 
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, str] = {}
    if not methods:
        return out_paths

    x = np.arange(len(methods), dtype=np.float64)
    ref_method = "ppo" if "ppo" in methods else methods[0]

    def _collect(metric: str, key: str) -> np.ndarray:
        vals: List[float] = []
        for m in methods:
            rec = degradation.get(m, {}).get(metric, {})
            vals.append(float(rec.get(key, np.nan)))
        return np.asarray(vals, dtype=np.float64)

    def _annotate_bars(ax: Any, bars: Any, fmt: str = "{:.4f}") -> None:
        for b in bars:
            h = float(b.get_height())
            if not np.isfinite(h):
                continue
            ax.text(
                float(b.get_x()) + float(b.get_width()) / 2.0,
                h,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _set_xticks(ax: Any) -> None:
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15, ha="right")
        ax.grid(True, alpha=0.25, axis="y")

    
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.4 * len(METRICS), 4.8))
    axes = [axes] if len(METRICS) == 1 else list(np.asarray(axes).reshape(-1))
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        vals = _collect(metric, "fixed")
        bars = ax.bar(
            x,
            vals,
            width=0.68,
            color=[METHOD_COLORS.get(m, "#999999") for m in methods],
            alpha=0.88,
        )
        _annotate_bars(ax, bars, fmt="{:.4f}")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}")
        ax.set_ylabel("")
        _set_xticks(ax)
        ax.text(0.98, 0.96, "", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555555")
    plt.tight_layout()
    p = out_dir / "Generalization_Absolute_TrainEnv.png"
    plt.savefig(p, dpi=400, bbox_inches="tight")
    plt.close()
    out_paths["generalization_absolute_trainenv"] = str(p)

    
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.4 * len(METRICS), 4.8))
    axes = [axes] if len(METRICS) == 1 else list(np.asarray(axes).reshape(-1))
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        vals = _collect(metric, "random")
        bars = ax.bar(
            x,
            vals,
            width=0.68,
            color=[METHOD_COLORS.get(m, "#999999") for m in methods],
            alpha=0.62,
            hatch="//",
        )
        _annotate_bars(ax, bars, fmt="{:.4f}")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}")
        ax.set_ylabel("")
        _set_xticks(ax)
        ax.text(0.98, 0.96, "", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555555")
    plt.tight_layout()
    p = out_dir / "Generalization_Absolute_GeneralizationEnv.png"
    plt.savefig(p, dpi=400, bbox_inches="tight")
    plt.close()
    out_paths["generalization_absolute_randomenv"] = str(p)

    
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.4 * len(METRICS), 4.8))
    axes = [axes] if len(METRICS) == 1 else list(np.asarray(axes).reshape(-1))
    width = 0.36
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        fixed_vals = _collect(metric, "fixed")
        random_vals = _collect(metric, "random")
        ref_idx = int(methods.index(ref_method))
        ref_fixed = float(fixed_vals[ref_idx]) if fixed_vals.size > ref_idx else float("nan")
        ref_random = float(random_vals[ref_idx]) if random_vals.size > ref_idx else float("nan")
        if not np.isfinite(ref_fixed) or abs(ref_fixed) < 1e-12:
            gap_fixed = np.full_like(fixed_vals, np.nan)
        else:
            gap_fixed = (fixed_vals - ref_fixed) / abs(ref_fixed) * 100.0
        if not np.isfinite(ref_random) or abs(ref_random) < 1e-12:
            gap_random = np.full_like(random_vals, np.nan)
        else:
            gap_random = (random_vals - ref_random) / abs(ref_random) * 100.0

        b1 = ax.bar(x - width / 2.0, gap_fixed, width=width, color="#6C8EBF", label=f" {METHOD_LABELS.get(ref_method, ref_method)}")
        b2 = ax.bar(
            x + width / 2.0,
            gap_random,
            width=width,
            color="#D38C5F",
            label=f" {METHOD_LABELS.get(ref_method, ref_method)}",
        )
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        for bars in (b1, b2):
            for b in bars:
                h = float(b.get_height())
                if not np.isfinite(h):
                    continue
                ax.text(
                    float(b.get_x()) + float(b.get_width()) / 2.0,
                    h,
                    f"{h:+.1f}%",
                    ha="center",
                    va="bottom" if h >= 0.0 else "top",
                    fontsize=8,
                )
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}")
        ax.set_ylabel(" (%)")
        _set_xticks(ax)
    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    p = out_dir / "Generalization_GapToReference_Percent.png"
    plt.savefig(p, dpi=400, bbox_inches="tight")
    plt.close()
    out_paths["generalization_gap_to_reference_pct"] = str(p)

    
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.4 * len(METRICS), 4.8))
    axes = [axes] if len(METRICS) == 1 else list(np.asarray(axes).reshape(-1))
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        rows: List[Dict[str, float]] = []
        for m in methods:
            rec = degradation.get(m, {}).get(metric, {})
            rows.append({"method": m, "deg": float(rec.get("degradation_pct", np.nan))})
        rows_sorted = sorted(rows, key=lambda r: float(r["deg"]) if np.isfinite(float(r["deg"])) else float("inf"))
        xx = np.arange(len(rows_sorted), dtype=np.float64)
        vals = np.asarray([float(r["deg"]) for r in rows_sorted], dtype=np.float64)
        labels = [METHOD_LABELS.get(str(r["method"]), str(r["method"])) for r in rows_sorted]
        colors = [METHOD_COLORS.get(str(r["method"]), "#999999") for r in rows_sorted]
        bars = ax.bar(xx, vals, width=0.68, color=colors, alpha=0.88)
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        for b in bars:
            h = float(b.get_height())
            if not np.isfinite(h):
                continue
            ax.text(
                float(b.get_x()) + float(b.get_width()) / 2.0,
                h,
                f"{h:+.2f}%",
                ha="center",
                va="bottom" if h >= 0.0 else "top",
                fontsize=8,
            )
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}")
        ax.set_ylabel(" (%)")
        ax.set_xticks(xx)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    p = out_dir / "Generalization_Degradation_Ranking.png"
    plt.savefig(p, dpi=400, bbox_inches="tight")
    plt.close()
    out_paths["generalization_degradation_ranking"] = str(p)

    return out_paths


def _write_report_txt(
    out_path: Path,
    degradation: Dict[str, Dict[str, Dict[str, float]]],
    methods: List[str],
    fixed_eval_ts: str,
    random_eval_ts: str,
) -> None:
    """"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 88 + "\n")
        f.write("H.13-S2 \n")
        f.write("=" * 88 + "\n")
        f.write(f" eval_ts: {fixed_eval_ts}\n")
        f.write(f" eval_ts: {random_eval_ts}\n\n")

        f.write("degradation_pct = (random - fixed) / max(|fixed|, 1e-9) * 100%\n")
        f.write("paper_cost  <= 10% 10%~20% >20% \n\n")

        for method in methods:
            f.write(f"[{METHOD_LABELS.get(method, method)}]\n")
            for metric in METRICS:
                rec = degradation.get(method, {}).get(metric, {})
                fixed_v = float(rec.get("fixed", np.nan))
                random_v = float(rec.get("random", np.nan))
                deg = float(rec.get("degradation_pct", np.nan))
                f.write(
                    f"  - {metric:16s}: fixed={fixed_v:.6f}, random={random_v:.6f}, degradation={deg:+.2f}%\n"
                )
            f.write("\n")

        ppo_pc_deg = float(degradation.get("ppo", {}).get("paper_cost", {}).get("degradation_pct", np.nan))
        if np.isfinite(ppo_pc_deg):
            if ppo_pc_deg <= 10.0:
                judge = ""
            elif ppo_pc_deg <= 20.0:
                judge = ""
            else:
                judge = ""
            f.write(f"PPO paper_cost {judge}{ppo_pc_deg:+.2f}%\n")


def _write_report_csv(out_path: Path, degradation: Dict[str, Dict[str, Dict[str, float]]], methods: List[str]) -> None:
    """CSV"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "fixed", "random", "degradation_pct"])
        for method in methods:
            for metric in METRICS:
                rec = degradation.get(method, {}).get(metric, {})
                writer.writerow(
                    [
                        method,
                        metric,
                        float(rec.get("fixed", np.nan)),
                        float(rec.get("random", np.nan)),
                        float(rec.get("degradation_pct", np.nan)),
                    ]
                )


def _safe_remove_path(path: Path) -> None:
    """/"""
    try:
        if path.is_dir():
            import shutil
            shutil.rmtree(path, ignore_errors=False)
        elif path.exists():
            path.unlink()
    except Exception as e:
        print(f"[][WARN]  {path}: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="H.13-S2  eval_policy_joint")
    parser.add_argument("--run_dir", type=str, required=True, help=" run_dir")
    parser.add_argument("--eval_ts", type=str, default="", help="")
    parser.add_argument("--fixed_env_yaml", type=str, default="", help="env yaml run_dir/configs/env_fixed.yaml")
    parser.add_argument(
        "--random_env_yaml",
        type=str,
        default="",
        help="env yamlfixed",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="ppo,balanced,greedy_delay,greedy_energy,myopic_optimization,always_comp,never_comp",
        help="",
    )
    parser.add_argument(
        "--eval_loads",
        type=str,
        default=DEFAULT_EVAL_LOADS_STR,
        help=" 0.1~1.0  1.0",
    )
    parser.add_argument("--eval_nseeds", type=int, default=5, help="")
    parser.add_argument("--eval_seed0", type=int, default=42, help="")
    parser.add_argument("--eval_ep_per", type=int, default=3, help="(seed,load)episode")
    parser.add_argument("--deterministic", type=int, default=1, help="1=")
    parser.add_argument("--device", type=str, default="", help=" cuda/cpu")
    parser.add_argument("--ckpt_path", type=str, default="", help="PPO checkpoint")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir: {run_dir}")

    fixed_env_yaml = (
        Path(args.fixed_env_yaml).resolve()
        if args.fixed_env_yaml.strip()
        else (run_dir / "configs" / "env_fixed.yaml").resolve()
    )
    if not fixed_env_yaml.exists():
        raise FileNotFoundError(f"env: {fixed_env_yaml}")

    methods = _split_csv_items(args.methods)
    if not methods:
        raise ValueError("methods")
    loads = _split_csv_floats(args.eval_loads)
    if not loads:
        raise ValueError("eval_loads")
    _assert_ten_load_protocol(loads)

    ckpt_path: Path = _resolve_ppo_ckpt(run_dir, args.ckpt_path)
    print(f"[] checkpoint: {ckpt_path}")

    base_ts = args.eval_ts.strip() or f"gen_eval_{_now_ts()}"
    fixed_ts = f"{base_ts}_fixed"
    random_ts = f"{base_ts}_random"

    if args.random_env_yaml.strip():
        random_env_yaml = Path(args.random_env_yaml).resolve()
        if not random_env_yaml.exists():
            raise FileNotFoundError(f"env: {random_env_yaml}")
    else:
        random_env_yaml = _build_default_random_env_from_fixed(
            fixed_env_yaml=fixed_env_yaml,
            out_yaml=run_dir / "evals" / base_ts / "random_env_base.yaml",
        )

    random_env_effective = _align_random_env_action_contract(
        fixed_env_yaml=fixed_env_yaml,
        random_env_yaml=random_env_yaml,
        out_yaml=run_dir / "evals" / base_ts / "random_env_aligned.yaml",
    )
    fixed_env_effective = _enforce_phaseg_refresh_interval(
        env_yaml=fixed_env_yaml,
        out_yaml=run_dir / "evals" / base_ts / "fixed_env_phaseg.yaml",
        tag="fixed",
    )
    random_env_effective = _enforce_phaseg_refresh_interval(
        env_yaml=random_env_effective,
        out_yaml=run_dir / "evals" / base_ts / "random_env_phaseg.yaml",
        tag="random",
    )

    fixed_eval_dir = _run_joint_eval(
        run_dir=run_dir,
        eval_ts=fixed_ts,
        env_yaml=fixed_env_effective,
        methods=methods,
        eval_loads=loads,
        n_seeds=args.eval_nseeds,
        ep_per=args.eval_ep_per,
        seed0=args.eval_seed0,
        deterministic=bool(int(args.deterministic) == 1),
        device=args.device,
        ckpt_path=ckpt_path,
    )
    random_eval_dir = _run_joint_eval(
        run_dir=run_dir,
        eval_ts=random_ts,
        env_yaml=random_env_effective,
        methods=methods,
        eval_loads=loads,
        n_seeds=args.eval_nseeds,
        ep_per=args.eval_ep_per,
        seed0=args.eval_seed0,
        deterministic=bool(int(args.deterministic) == 1),
        device=args.device,
        ckpt_path=ckpt_path,
    )

    fixed_summary = _load_summary(fixed_eval_dir)
    random_summary = _load_summary(random_eval_dir)

    fixed_stats = _extract_method_metrics(fixed_summary, methods=methods, variant="full")
    random_stats = _extract_method_metrics(random_summary, methods=methods, variant="full")
    degradation = _calc_degradation(fixed_stats, random_stats, methods=methods)
    degradation_required = _build_required_degradation_fields(degradation, methods)
    fixed_curves = _extract_method_metric_curves(fixed_summary, methods=methods, variant="full")
    random_curves = _extract_method_metric_curves(random_summary, methods=methods, variant="full")
    band_degradation = _calc_load_band_degradation(fixed_curves, random_curves, methods=methods)
    scenario_profiles = {
        "fixed": _extract_scenario_cluster_profile(fixed_env_effective, tag="fixed"),
        "random": _extract_scenario_cluster_profile(random_env_effective, tag="random"),
    }

    out_eval_dir = run_dir / "evals" / base_ts
    out_fig_std_dir = run_dir / "figs" / "Generalization_Eval eval"
    out_eval_dir.mkdir(parents=True, exist_ok=True)
    out_fig_std_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = out_eval_dir / "generalization_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_eval_ts": base_ts,
                "fixed_eval_ts": fixed_ts,
                "random_eval_ts": random_ts,
                "run_dir": str(run_dir),
                "fixed_env_yaml": str(fixed_env_effective),
                "random_env_yaml": str(random_env_effective),
                "methods": methods,
                "eval_loads": loads,
                "eval_loads_protocol": "fixed_0.1_to_1.0_10pts",
                "eval_nseeds": int(args.eval_nseeds),
                "eval_ep_per": int(args.eval_ep_per),
                "deterministic": bool(int(args.deterministic) == 1),
                "ckpt_path": str(ckpt_path),
                "degradation": degradation,
                "degradation_required_fields": degradation_required,
                "scenario_profiles": scenario_profiles,
                "band_degradation": band_degradation,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    table_csv = out_eval_dir / "generalization_table.csv"
    _write_report_csv(table_csv, degradation, methods)

    cluster_json = out_eval_dir / "generalization_cluster_metrics.json"
    with open(cluster_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_eval_ts": base_ts,
                "fixed_eval_ts": fixed_ts,
                "random_eval_ts": random_ts,
                "scenario_profiles": scenario_profiles,
                "band_degradation": band_degradation,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    cluster_csv = out_eval_dir / "generalization_cluster_table.csv"
    _write_cluster_report_csv(cluster_csv, band_degradation, methods)

    summary_txt = out_eval_dir / "generalization_summary.txt"
    _write_report_txt(summary_txt, degradation, methods, fixed_ts, random_ts)

    fig_path = out_fig_std_dir / "Generalization_Comparison.png"
    _plot_comparison(degradation, methods, fig_path)
    cluster_fig = out_fig_std_dir / "Generalization_ClusterBreakdown.png"
    _plot_cluster_breakdown(
        out_fig=cluster_fig,
        methods=methods,
        scenario_profiles=scenario_profiles,
        band_degradation=band_degradation,
    )
    extra_fig_dir = out_fig_std_dir / "Generalization_ExtraBars"
    extra_fig_paths = _plot_extended_comparison_bundle(
        degradation=degradation,
        methods=methods,
        out_dir=extra_fig_dir,
    )
    try:
        artifacts: Dict[str, str] = {
            "generalization_metrics_json": str(metrics_json),
            "generalization_table_csv": str(table_csv),
            "generalization_summary_txt": str(summary_txt),
            "generalization_figure": str(fig_path),
            "generalization_cluster_metrics_json": str(cluster_json),
            "generalization_cluster_table_csv": str(cluster_csv),
            "generalization_cluster_figure": str(cluster_fig),
            "fixed_eval_ts": str(fixed_ts),
            "random_eval_ts": str(random_ts),
        }
        artifacts.update(extra_fig_paths)
        phaseg_summary_path = upsert_phaseg_eval_summary(
            run_dir=run_dir,
            role="gen",
            eval_ts=base_ts,
            ckpt_path=str(ckpt_path),
            artifacts=artifacts,
        )
        if phaseg_summary_path is not None:
            print(f"[phaseg] updated summary: {phaseg_summary_path}")
    except Exception as e:
        msg = str(e)
        if "[phaseg][FATAL]" in msg:
            raise
        print(f"[phaseg][WARN] failed to update summary: {type(e).__name__}: {e}")

    
    _safe_remove_path(run_dir / "evals" / fixed_ts)
    _safe_remove_path(run_dir / "evals" / random_ts)
    _safe_remove_path(run_dir / "figs" / fixed_ts)
    _safe_remove_path(run_dir / "figs" / random_ts)

    print("\n" + "=" * 88)
    print("[] ")
    print(f"[] fixed_eval_dir: {fixed_eval_dir}")
    print(f"[] random_eval_dir: {random_eval_dir}")
    print(f"[] metrics_json: {metrics_json}")
    print(f"[] table_csv: {table_csv}")
    print(f"[] cluster_json: {cluster_json}")
    print(f"[] cluster_csv: {cluster_csv}")
    print(f"[] summary_txt: {summary_txt}")
    print(f"[] figure: {fig_path}")
    print(f"[] cluster_figure: {cluster_fig}")
    print(f"[] extra_fig_dir: {extra_fig_dir}")
    for key in sorted(extra_fig_paths.keys()):
        print(f"[] {key}: {extra_fig_paths[key]}")
    print("=" * 88)


if __name__ == "__main__":
    main()
