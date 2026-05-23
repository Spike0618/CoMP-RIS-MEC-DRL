#!/usr/bin/env python3
"""
 cost_ref T7 / Phase Z


1. 
2.  `--loads 0.6,0.8,1.0` 
3. 
   - objective.<target_key>
   - objective.<target_key>_by_load
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algos.baseline import baseline_action
from src.envs.comp_ris_env_joint import CompRISEnvJoint
from src.utils.env_cfg import build_joint_env_cfg


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML  dict{path}")
    return obj


def _parse_loads_text(text: str) -> List[float]:
    vals: List[float] = []
    raw = str(text or "").strip()
    if not raw:
        return vals
    for tok in raw.split(","):
        s = str(tok).strip()
        if not s:
            continue
        try:
            x = float(s)
        except Exception:
            continue
        if np.isfinite(x):
            vals.append(float(np.clip(x, 0.05, 2.0)))
    vals = sorted(list(dict.fromkeys([round(float(v), 6) for v in vals])))
    return vals


def _parse_speed_levels_raw(v: Any) -> List[float]:
    out: List[float] = []
    if isinstance(v, (list, tuple)):
        for x in v:
            try:
                fx = float(x)
            except Exception:
                continue
            if np.isfinite(fx) and fx >= 0.0:
                out.append(float(fx))
    out = sorted(list(dict.fromkeys([round(float(v), 6) for v in out])))
    return out


def _run_balanced_rollouts(
    env: CompRISEnvJoint,
    episodes: int,
    *,
    progress_every: int = 20,
    progress_tag: str = "",
) -> Dict[str, float]:
    step_costs: List[float] = []
    ep_cost_means: List[float] = []
    total = int(max(episodes, 1))
    hb = int(max(progress_every, 1))
    for ep in range(total):
        _ = env.reset()
        done = False
        ep_cost_series: List[float] = []
        while not done:
            act = baseline_action(env, mode="balanced")
            _, _, done, info = env.step(act)
            c = float(info.get("paper_cost", np.nan))
            if np.isfinite(c):
                step_costs.append(c)
                ep_cost_series.append(c)
        if ep_cost_series:
            ep_cost_means.append(float(np.mean(np.asarray(ep_cost_series, dtype=np.float64))))
        done_now = int(ep + 1)
        if (done_now % hb == 0) or (done_now == total):
            print(
                f"[calibrate_cost_ref][HB]{progress_tag} progress={done_now}/{total} "
                f"step_samples={len(step_costs)}",
                flush=True,
            )

    step_arr = np.asarray(step_costs, dtype=np.float64)
    ep_arr = np.asarray(ep_cost_means, dtype=np.float64)
    return {
        "n_episodes": int(total),
        "n_step_samples": int(step_arr.size),
        "step_mean": float(np.mean(step_arr)) if step_arr.size > 0 else float("nan"),
        "step_std": float(np.std(step_arr)) if step_arr.size > 0 else float("nan"),
        "episode_mean": float(np.mean(ep_arr)) if ep_arr.size > 0 else float("nan"),
        "episode_std": float(np.std(ep_arr)) if ep_arr.size > 0 else float("nan"),
    }


def _write_back_cost_ref(
    env_yaml: Path,
    cost_ref: float,
    target_key: str,
    by_load_map: Dict[str, float] | None = None,
    by_speed_map: Dict[str, float] | None = None,
) -> None:
    cfg = _load_yaml(env_yaml)
    obj = cfg.get("objective", {}) or {}
    if not isinstance(obj, dict):
        obj = {}
    obj[str(target_key)] = float(cost_ref)
    if isinstance(by_load_map, dict) and by_load_map:
        obj[f"{str(target_key)}_by_load"] = {str(k): float(v) for k, v in by_load_map.items()}
    if isinstance(by_speed_map, dict) and by_speed_map:
        obj[f"{str(target_key)}_by_speed"] = {str(k): float(v) for k, v in by_speed_map.items()}
    cfg["objective"] = obj
    with open(env_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the cost reference used by the reward function.")
    parser.add_argument(
        "--env-yaml",
        type=str,
        default="configs/PhaseZ4/env_phaseZ4.yaml",
        help="Environment YAML path.",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of calibration episodes.")
    parser.add_argument("--seed", type=int, default=3411, help="Random seed.")
    parser.add_argument("--write-back", type=int, default=1, help="Write the calibrated cost reference back to the environment YAML.")
    parser.add_argument(
        "--target-key",
        type=str,
        default="auto",
        choices=["auto", "t7_cost_ref", "z_cost_ref"],
        help="Target objective key. auto selects the key from the reward design.",
    )
    parser.add_argument(
        "--loads",
        type=str,
        default="",
        help="Optional comma-separated load_scale list, e.g. 0.6,0.8,1.0.",
    )
    parser.add_argument("--speed-map-enable", type=int, default=1, help="Whether to calibrate per-speed references for mobile scenarios.")
    parser.add_argument("--speed-map-load", type=float, default=float('nan'), help="Load_scale used for speed-map calibration.")
    parser.add_argument("--speed-map-episodes", type=int, default=0, help="Episodes per speed-map item; <=0 reuses --episodes.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N episodes.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional output JSON path. Defaults to the environment config directory.",
    )
    args = parser.parse_args()

    env_yaml = Path(args.env_yaml)
    if not env_yaml.is_absolute():
        env_yaml = (PROJECT_ROOT / env_yaml).resolve()
    if not env_yaml.exists():
        raise FileNotFoundError(f"{env_yaml}")

    env_cfg_dict = _load_yaml(env_yaml)
    objective_cfg = env_cfg_dict.get("objective", {}) if isinstance(env_cfg_dict, dict) else {}
    if not isinstance(objective_cfg, dict):
        objective_cfg = {}
    reward_design = str(objective_cfg.get("reward_design", "")).strip().lower()
    if str(args.target_key).strip().lower() == "auto":
        target_key = "z_cost_ref" if reward_design in ("z_shifted_log", "z4_linear", "z5_saturating") else "t7_cost_ref"
    else:
        target_key = str(args.target_key).strip()

    cfg_load = float((env_cfg_dict.get("task", {}) or {}).get("load_scale", 1.0))
    loads = _parse_loads_text(str(args.loads))
    if not loads:
        loads = [round(float(np.clip(cfg_load, 0.05, 2.0)), 6)]
    speed_map_enable = bool(int(args.speed_map_enable) == 1)
    speed_map_load = float(args.speed_map_load)
    speed_map_episodes = int(args.speed_map_episodes)
    if speed_map_episodes <= 0:
        speed_map_episodes = int(args.episodes)
    speed_map_episodes = int(max(speed_map_episodes, 1))
    scenario_cfg = env_cfg_dict.get("scenario", {}) if isinstance(env_cfg_dict, dict) else {}
    if not isinstance(scenario_cfg, dict):
        scenario_cfg = {}
    speed_levels = _parse_speed_levels_raw(scenario_cfg.get("user_speed_levels", []))
    if not speed_levels:
        speed_levels = [round(float(max(float(scenario_cfg.get("user_speed", 0.0)), 0.0)), 6)]

    per_load_stats: List[Dict[str, Any]] = []
    per_speed_stats: List[Dict[str, Any]] = []
    by_load_step_mean: Dict[str, float] = {}
    by_speed_step_mean: Dict[str, float] = {}
    for ld in loads:
        cfg_i = deepcopy(env_cfg_dict)
        task_i = cfg_i.get("task", {}) or {}
        if not isinstance(task_i, dict):
            task_i = {}
        task_i["load_scale"] = float(ld)
        cfg_i["task"] = task_i

        env_cfg, _clean, dropped = build_joint_env_cfg(cfg_i)
        if dropped:
            print(f"[calibrate_cost_ref][WARN] load={ld:.3f} dropped keys: {dropped[:20]}", flush=True)
        env = CompRISEnvJoint(cfg=env_cfg, seed=int(args.seed))
        stats = _run_balanced_rollouts(
            env,
            episodes=int(args.episodes),
            progress_every=int(args.progress_every),
            progress_tag=f"[load={ld:.3f}]",
        )
        step_mean = float(stats.get("step_mean", float("nan")))
        if np.isfinite(step_mean):
            by_load_step_mean[f"{ld:.3f}"] = float(step_mean)
        per_load_stats.append(
            {
                "load": float(ld),
                "stats": stats,
            }
        )

    
    if speed_map_enable and len(speed_levels) > 1:
        speed_load_eff = float(cfg_load) if (not np.isfinite(speed_map_load)) else float(np.clip(speed_map_load, 0.05, 2.0))
        for spd in speed_levels:
            cfg_i = deepcopy(env_cfg_dict)
            task_i = cfg_i.get("task", {}) or {}
            if not isinstance(task_i, dict):
                task_i = {}
            task_i["load_scale"] = float(speed_load_eff)
            cfg_i["task"] = task_i

            sc_i = cfg_i.get("scenario", {}) or {}
            if not isinstance(sc_i, dict):
                sc_i = {}
            sc_i["user_speed"] = float(spd)
            sc_i["user_speed_levels"] = [float(spd)]
            sc_i["user_speed_probs"] = [1.0]
            cfg_i["scenario"] = sc_i

            env_cfg, _clean, dropped = build_joint_env_cfg(cfg_i)
            if dropped:
                print(f"[calibrate_cost_ref][WARN] speed={spd:.3f} dropped keys: {dropped[:20]}", flush=True)
            env = CompRISEnvJoint(cfg=env_cfg, seed=int(args.seed))
            stats = _run_balanced_rollouts(
                env,
                episodes=int(speed_map_episodes),
                progress_every=int(args.progress_every),
                progress_tag=f"[speed={spd:.3f}]",
            )
            step_mean = float(stats.get("step_mean", float("nan")))
            if np.isfinite(step_mean):
                by_speed_step_mean[f"{float(spd):.3f}"] = float(step_mean)
            per_speed_stats.append(
                {
                    "speed": float(spd),
                    "load": float(speed_load_eff),
                    "stats": stats,
                }
            )

    
    scalar_ref = float("nan")
    if by_load_step_mean:
        key_nom = f"{round(float(cfg_load), 3):.3f}"
        if key_nom in by_load_step_mean:
            scalar_ref = float(by_load_step_mean[key_nom])
        else:
            max_key = sorted(by_load_step_mean.keys(), key=lambda s: float(s))[-1]
            scalar_ref = float(by_load_step_mean[max_key])

    if bool(int(args.write_back) == 1) and np.isfinite(scalar_ref):
        map_for_write = by_load_step_mean if len(by_load_step_mean) > 1 else None
        speed_map_for_write = by_speed_step_mean if len(by_speed_step_mean) > 1 else None
        _write_back_cost_ref(
            env_yaml=env_yaml,
            cost_ref=float(scalar_ref),
            target_key=target_key,
            by_load_map=map_for_write,
            by_speed_map=speed_map_for_write,
        )
        if map_for_write or speed_map_for_write:
            print(
                f"[calibrate_cost_ref]  objective.{target_key}={scalar_ref:.6f} + "
                f"objective.{target_key}_by_load/by_speed  {env_yaml}",
                flush=True,
            )
        else:
            print(
                f"[calibrate_cost_ref]  objective.{target_key}={scalar_ref:.6f}  {env_yaml}",
                flush=True,
            )

    out_payload = {
        "env_yaml": str(env_yaml),
        "seed": int(args.seed),
        "reward_design": reward_design,
        "target_key": str(target_key),
        "loads": [float(x) for x in loads],
        "recommended_cost_ref": float(scalar_ref) if np.isfinite(scalar_ref) else float("nan"),
        
        "recommended_t7_cost_ref": float(scalar_ref) if np.isfinite(scalar_ref) else float("nan"),
        "recommended_cost_ref_by_load": {k: float(v) for k, v in by_load_step_mean.items()},
        "recommended_cost_ref_by_speed": {k: float(v) for k, v in by_speed_step_mean.items()},
        "per_load_stats": per_load_stats,
        "per_speed_stats": per_speed_stats,
        "speed_map_enable": bool(speed_map_enable),
        "speed_map_load": float(cfg_load if (not np.isfinite(speed_map_load)) else np.clip(speed_map_load, 0.05, 2.0)),
        "speed_map_episodes": int(speed_map_episodes),
    }
    if str(args.out_json).strip():
        out_json = Path(args.out_json.strip())
        if not out_json.is_absolute():
            out_json = (PROJECT_ROOT / out_json).resolve()
    else:
        out_json = env_yaml.parent / "t7_cost_ref_calibration.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
