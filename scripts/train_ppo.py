from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.utils.env_cfg import build_joint_env_cfg
from src.utils.runio import ensure_run_subdirs, make_run_dir, snapshot_cfgs
from src.utils.seed import set_global_seed
from src.utils.train_report import (
    compute_convergence_acceptance_from_csv,
    compute_h11_a_acceptance,
    compute_h11_cfg_dropped_keys_audit,
    compute_h11_knob_effect_audit,
    compute_h11_reward_balance,
    compute_h11_reward_shape_gate_dense,
    compute_phaseA_acceptance_from_csv,
    compute_phaseA_canary_acceptance_from_csv,
    compute_phaseB_acceptance_from_csv,
    compute_phaseZ_stage_acceptance_from_csv,
    compute_shaping_acceptance,
    plot_convergence_reward_total,
    plot_convergence_reward_total_ema_ci_only,
    plot_eval_metrics,
    plot_fixed_eval_curve,
    plot_metric1_robust_trend,
    plot_ppo_diagnostics_from_csv,
    plot_reward_decomposition,
    plot_training_curves,
    read_train_metrics_csv,
)


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _infer_algo_tag_from_cfg(cfg: dict) -> str:
    """
     run_name 
    - ppo_base
    -  agentic/meta amcp_ppo
    """
    try:
        train = (cfg.get("train", {}) or {}) if isinstance(cfg, dict) else {}
        if not isinstance(train, dict):
            return "ppo_base"

        
        meta_source = str(train.get("meta_source", "")).strip().lower()
        if meta_source in ("heuristic", "llm", "random"):
            return "amcp_ppo"

        
        for key in ("meta", "agentic", "meta_controller"):
            block = train.get(key, {}) or {}
            if not isinstance(block, dict):
                continue
            if _as_bool(block.get("enabled", block.get("enable", False))):
                return "amcp_ppo"
            src = str(block.get("source", block.get("meta_source", ""))).strip().lower()
            if src in ("heuristic", "llm", "random"):
                return "amcp_ppo"
    except Exception:
        pass
    return "ppo_base"


def _apply_meta_cli_overrides(train_cfg: dict, args: argparse.Namespace) -> None:
    train_cfg.setdefault("train", {})
    train = train_cfg.get("train", {}) or {}
    if not isinstance(train, dict):
        train = {}
        train_cfg["train"] = train

    meta = train.get("meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}

    
    
    def _default_meta_yaml() -> str:
        candidates = [
            PROJECT_ROOT / "configs" / "meta_controller.yaml",
        ]
        for p in candidates:
            try:
                if p.exists():
                    return str(p.resolve())
            except Exception:
                continue
        return str(candidates[0].resolve())

    meta_enable = getattr(args, "meta_enable", None)
    meta_source = getattr(args, "meta_source", None)
    meta_yaml = getattr(args, "meta_yaml", None)

    if meta_enable is not None:
        meta["enabled"] = bool(int(meta_enable) == 1)
    if meta_source is not None and str(meta_source).strip():
        meta["source"] = str(meta_source).strip().lower()
        if "enabled" not in meta:
            meta["enabled"] = True
    if meta_yaml is not None and str(meta_yaml).strip():
        p = Path(str(meta_yaml).strip())
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        meta["meta_yaml"] = str(p)

    if bool(meta.get("enabled", False)) and (not str(meta.get("meta_yaml", "")).strip()):
        meta["meta_yaml"] = _default_meta_yaml()

    
    if "source" in meta:
        train["meta_source"] = str(meta.get("source", "")).strip().lower()
    if "enabled" in meta:
        train["meta_enable"] = bool(meta.get("enabled", False))
    if "meta_yaml" in meta:
        train["meta_yaml"] = str(meta.get("meta_yaml", "")).strip()

    if meta:
        train["meta"] = meta
    train_cfg["train"] = train


def _apply_t6_cli_overrides(train_cfg: dict, args: argparse.Namespace) -> None:
    """
    Route behavior-cloning warm-start CLI overrides into train.bc_warmstart.
    """
    train_cfg.setdefault("train", {})
    train = train_cfg.get("train", {}) or {}
    if not isinstance(train, dict):
        train = {}
        train_cfg["train"] = train

    bc = train.get("bc_warmstart", {}) or {}
    if not isinstance(bc, dict):
        bc = {}

    if getattr(args, "bc_warmstart_enable", None) is not None:
        bc["enable"] = bool(int(args.bc_warmstart_enable) == 1)
    if getattr(args, "bc_warmstart_weight0", None) is not None:
        bc["weight0"] = float(args.bc_warmstart_weight0)
    if getattr(args, "bc_warmstart_decay_steps", None) is not None:
        bc["decay_steps"] = int(args.bc_warmstart_decay_steps)
    if getattr(args, "bc_warmstart_cost_weight0", None) is not None:
        bc["cost_weight0"] = float(args.bc_warmstart_cost_weight0)
    if getattr(args, "bc_warmstart_cost_decay_steps", None) is not None:
        bc["cost_decay_steps"] = int(args.bc_warmstart_cost_decay_steps)

    if bc:
        train["bc_warmstart"] = bc
    train_cfg["train"] = train


def _safe_float(v, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _sanitize_run_token(token: str, default: str = "run") -> str:
    """
     run_name 
    """
    s = str(token or "").strip()
    if not s:
        return str(default)
    
    bad_chars = '<>:"/\\|inf*'
    for ch in bad_chars:
        s = s.replace(ch, "_")
    s = s.replace(" ", "_")
    
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("._")
    return s if s else str(default)


def _maybe_auto_calibrate_z_cost_ref(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    env_cfg_dict: dict,
    train_cfg: dict,
) -> None:
    """
    Phase Z2 z_cost_ref balanced baseline

    
    1.  objective.<target_key>
    2.  objective.<target_key> + objective.<target_key>_by_load
    3. mobile objective.<target_key>_by_speed
    """
    if not isinstance(env_cfg_dict, dict):
        return
    obj = env_cfg_dict.get("objective", {}) or {}
    if not isinstance(obj, dict):
        return
    reward_design = str(obj.get("reward_design", "")).strip().lower()
    if reward_design not in ("z_shifted_log", "z4_linear", "z5_saturating"):
        return

    train_block = (train_cfg.get("train", {}) or {}) if isinstance(train_cfg, dict) else {}
    calib_cfg = train_block.get("z2_cost_ref_auto_calibrate", {}) or {}
    if not isinstance(calib_cfg, dict) or (not _as_bool(calib_cfg.get("enable", False))):
        return

    episodes = int(max(int(calib_cfg.get("episodes", 200)), 1))
    baseline_mode = str(calib_cfg.get("baseline_mode", "balanced")).strip().lower() or "balanced"
    if baseline_mode != "balanced":
        raise ValueError("z2_cost_ref_auto_calibrate  baseline_mode=balanced")
    target_key = str(calib_cfg.get("target_key", "z_cost_ref")).strip() or "z_cost_ref"
    write_back_env_yaml = bool(_as_bool(calib_cfg.get("write_back_env_yaml", False)))
    require_finite = bool(_as_bool(calib_cfg.get("require_finite", True)))
    only_if_placeholder = bool(_as_bool(calib_cfg.get("only_if_placeholder", False)))
    placeholder_value = float(calib_cfg.get("placeholder_value", 3.0))
    placeholder_tol = float(max(float(calib_cfg.get("placeholder_tol", 1e-6)), 0.0))
    progress_every = int(max(int(calib_cfg.get("progress_every", 20)), 1))
    seed = int(train_cfg.get("seed", env_cfg_dict.get("seed", 3411)))
    speed_map_enable = bool(_as_bool(calib_cfg.get("speed_map_enable", True)))
    speed_map_load = float(np.clip(float(calib_cfg.get("speed_map_load", np.nan)), 0.05, 2.0))
    speed_map_episodes = int(max(int(calib_cfg.get("speed_map_episodes", episodes)), 1))

    def _parse_loads(v: Any) -> list[float]:
        out: list[float] = []
        if isinstance(v, (list, tuple)):
            for x in v:
                try:
                    fx = float(x)
                except Exception:
                    continue
                if np.isfinite(fx):
                    out.append(float(np.clip(fx, 0.05, 2.0)))
        else:
            s = str(v or "").strip()
            if s:
                for tok in s.split(","):
                    tt = str(tok).strip()
                    if not tt:
                        continue
                    try:
                        fx = float(tt)
                    except Exception:
                        continue
                    if np.isfinite(fx):
                        out.append(float(np.clip(fx, 0.05, 2.0)))
        out = sorted(list(dict.fromkeys([round(float(x), 6) for x in out])))
        return out

    def _parse_speed_levels(v: Any) -> list[float]:
        out: list[float] = []
        if isinstance(v, (list, tuple)):
            for x in v:
                try:
                    fx = float(x)
                except Exception:
                    continue
                if np.isfinite(fx) and fx >= 0.0:
                    out.append(float(fx))
        out = sorted(list(dict.fromkeys([round(float(x), 6) for x in out])))
        return out

    cfg_task = env_cfg_dict.get("task", {}) or {}
    if not isinstance(cfg_task, dict):
        cfg_task = {}
    nominal_load = float(np.clip(float(cfg_task.get("load_scale", 1.0)), 0.05, 2.0))
    cfg_sc = env_cfg_dict.get("scenario", {}) or {}
    if not isinstance(cfg_sc, dict):
        cfg_sc = {}
    speed_levels = _parse_speed_levels(cfg_sc.get("user_speed_levels", []))
    if not speed_levels:
        speed_single = float(max(float(cfg_sc.get("user_speed", 0.0)), 0.0))
        speed_levels = [float(round(speed_single, 6))]

    calib_loads = _parse_loads(calib_cfg.get("loads", []))
    if not calib_loads:
        calib_loads = [float(round(nominal_load, 6))]

    current_ref = _safe_float(obj.get(target_key, float("nan")))
    if only_if_placeholder and np.isfinite(current_ref):
        has_map = isinstance(obj.get(f"{target_key}_by_load", None), dict) and len(obj.get(f"{target_key}_by_load", {}) or {}) > 0
        if len(calib_loads) > 1:
            if has_map:
                print(
                    f"[HB][z2_calib] skip auto-calibration: objective.{target_key}_by_load ",
                    flush=True,
                )
                return
        else:
            if abs(float(current_ref) - float(placeholder_value)) > float(placeholder_tol):
                print(
                    f"[HB][z2_calib] skip auto-calibration: objective.{target_key}={current_ref:.6f} ",
                    flush=True,
                )
                return

    from src.algos.baseline import baseline_action
    from src.envs.comp_ris_env_joint import CompRISEnvJoint

    by_load_ref: dict[str, float] = {}
    by_speed_ref: dict[str, float] = {}
    per_load_stats: list[dict] = []
    per_speed_stats: list[dict] = []
    t_start_all = time.time()
    for ld in calib_loads:
        env_cfg_dict_i = deepcopy(env_cfg_dict)
        task_i = env_cfg_dict_i.get("task", {}) or {}
        if not isinstance(task_i, dict):
            task_i = {}
        task_i["load_scale"] = float(ld)
        env_cfg_dict_i["task"] = task_i

        env_cfg_calib, _clean, dropped = build_joint_env_cfg(env_cfg_dict_i)
        dropped_show = [k for k in dropped if str(k).strip() != "seed"]
        if dropped_show:
            print(f"[HB][z2_calib][WARN] load={ld:.3f} dropped keys: {dropped_show[:20]}", flush=True)
        env = CompRISEnvJoint(cfg=env_cfg_calib, seed=seed)

        step_costs: list[float] = []
        ep_cost_means: list[float] = []
        t_start = time.time()
        for ep_idx in range(episodes):
            _ = env.reset()
            done = False
            ep_cost_series: list[float] = []
            while not done:
                act = baseline_action(env, mode="balanced")
                _, _, done, info = env.step(act)
                c = float(info.get("paper_cost", np.nan))
                if np.isfinite(c):
                    step_costs.append(c)
                    ep_cost_series.append(c)
            if ep_cost_series:
                ep_cost_means.append(float(np.mean(np.asarray(ep_cost_series, dtype=np.float64))))

            done_now = int(ep_idx + 1)
            if (done_now % progress_every == 0) or (done_now == episodes):
                elapsed = float(max(time.time() - t_start, 1e-9))
                speed = float(done_now / elapsed)
                eta = float((episodes - done_now) / max(speed, 1e-9))
                print(
                    f"[HB][z2_calib][load={ld:.3f}] progress={done_now}/{episodes} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s out={run_dir / 'z2_cost_ref_calibration.json'}",
                    flush=True,
                )

        step_arr = np.asarray(step_costs, dtype=np.float64)
        ep_arr = np.asarray(ep_cost_means, dtype=np.float64)
        step_mean = float(np.nanmean(step_arr)) if step_arr.size > 0 else float("nan")
        if np.isfinite(step_mean):
            by_load_ref[f"{ld:.3f}"] = float(step_mean)
        per_load_stats.append(
            {
                "load": float(ld),
                "step_samples": int(step_arr.size),
                "step_mean": float(step_mean),
                "step_std": float(np.nanstd(step_arr)) if step_arr.size > 0 else float("nan"),
                "episode_mean": float(np.nanmean(ep_arr)) if ep_arr.size > 0 else float("nan"),
                "episode_std": float(np.nanstd(ep_arr)) if ep_arr.size > 0 else float("nan"),
            }
        )

    
    if speed_map_enable and len(speed_levels) > 1 and np.isfinite(float(nominal_load)):
        speed_load = float(nominal_load) if (not np.isfinite(speed_map_load)) else float(speed_map_load)
        for spd in speed_levels:
            env_cfg_dict_i = deepcopy(env_cfg_dict)
            task_i = env_cfg_dict_i.get("task", {}) or {}
            if not isinstance(task_i, dict):
                task_i = {}
            task_i["load_scale"] = float(speed_load)
            env_cfg_dict_i["task"] = task_i

            sc_i = env_cfg_dict_i.get("scenario", {}) or {}
            if not isinstance(sc_i, dict):
                sc_i = {}
            sc_i["user_speed"] = float(spd)
            sc_i["user_speed_levels"] = [float(spd)]
            sc_i["user_speed_probs"] = [1.0]
            env_cfg_dict_i["scenario"] = sc_i

            env_cfg_calib, _clean, dropped = build_joint_env_cfg(env_cfg_dict_i)
            dropped_show = [k for k in dropped if str(k).strip() != "seed"]
            if dropped_show:
                print(f"[HB][z2_calib][WARN] speed={spd:.3f} dropped keys: {dropped_show[:20]}", flush=True)
            env = CompRISEnvJoint(cfg=env_cfg_calib, seed=seed)

            step_costs: list[float] = []
            ep_cost_means: list[float] = []
            t_start_spd = time.time()
            for ep_idx in range(speed_map_episodes):
                _ = env.reset()
                done = False
                ep_cost_series: list[float] = []
                while not done:
                    act = baseline_action(env, mode="balanced")
                    _, _, done, info = env.step(act)
                    c = float(info.get("paper_cost", np.nan))
                    if np.isfinite(c):
                        step_costs.append(c)
                        ep_cost_series.append(c)
                if ep_cost_series:
                    ep_cost_means.append(float(np.mean(np.asarray(ep_cost_series, dtype=np.float64))))
                done_now = int(ep_idx + 1)
                if (done_now % progress_every == 0) or (done_now == speed_map_episodes):
                    elapsed = float(max(time.time() - t_start_spd, 1e-9))
                    speed_eps = float(done_now / elapsed)
                    eta = float((speed_map_episodes - done_now) / max(speed_eps, 1e-9))
                    print(
                        f"[HB][z2_calib][speed={spd:.3f}] progress={done_now}/{speed_map_episodes} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s out={run_dir / 'z2_cost_ref_calibration.json'}",
                        flush=True,
                    )

            step_arr = np.asarray(step_costs, dtype=np.float64)
            ep_arr = np.asarray(ep_cost_means, dtype=np.float64)
            step_mean = float(np.nanmean(step_arr)) if step_arr.size > 0 else float("nan")
            if np.isfinite(step_mean):
                by_speed_ref[f"{spd:.3f}"] = float(step_mean)
            per_speed_stats.append(
                {
                    "speed": float(spd),
                    "load": float(speed_load),
                    "step_samples": int(step_arr.size),
                    "step_mean": float(step_mean),
                    "step_std": float(np.nanstd(step_arr)) if step_arr.size > 0 else float("nan"),
                    "episode_mean": float(np.nanmean(ep_arr)) if ep_arr.size > 0 else float("nan"),
                    "episode_std": float(np.nanstd(ep_arr)) if ep_arr.size > 0 else float("nan"),
                }
            )

    calibrated_ref = float("nan")
    if by_load_ref:
        nominal_key = f"{round(float(nominal_load), 3):.3f}"
        if nominal_key in by_load_ref:
            calibrated_ref = float(by_load_ref[nominal_key])
        else:
            max_key = sorted(by_load_ref.keys(), key=lambda s: float(s))[-1]
            calibrated_ref = float(by_load_ref[max_key])

    payload = {
        "enable": True,
        "episodes": int(episodes),
        "seed": int(seed),
        "baseline_mode": "balanced",
        "calib_loads": [float(x) for x in calib_loads],
        "target_key": str(target_key),
        "old_value": float(current_ref) if np.isfinite(current_ref) else None,
        "calibrated_value": float(calibrated_ref) if np.isfinite(calibrated_ref) else None,
        "calibrated_by_load": {k: float(v) for k, v in by_load_ref.items()},
        "calibrated_by_speed": {k: float(v) for k, v in by_speed_ref.items()},
        "per_load_stats": per_load_stats,
        "per_speed_stats": per_speed_stats,
        "speed_map_enable": bool(speed_map_enable),
        "speed_map_load": float(nominal_load if (not np.isfinite(speed_map_load)) else speed_map_load),
        "speed_map_episodes": int(speed_map_episodes),
        "elapsed_sec": float(max(time.time() - t_start_all, 0.0)),
    }
    (run_dir / "z2_cost_ref_calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    speed_map_required = bool(speed_map_enable and len(speed_levels) > 1)
    if require_finite and (
        (not np.isfinite(calibrated_ref))
        or (len(by_load_ref) < len(calib_loads))
        or (speed_map_required and (len(by_speed_ref) < len(speed_levels)))
    ):
        raise RuntimeError(
            "[HB][z2_calib][FATAL] z_cost_ref /"
        )

    if np.isfinite(calibrated_ref):
        obj[target_key] = float(calibrated_ref)
        has_load_map = len(by_load_ref) > 1
        has_speed_map = len(by_speed_ref) > 1
        if has_load_map:
            obj[f"{target_key}_by_load"] = {k: float(v) for k, v in by_load_ref.items()}
        if has_speed_map:
            obj[f"{target_key}_by_speed"] = {k: float(v) for k, v in by_speed_ref.items()}
        env_cfg_dict["objective"] = obj
        if has_load_map or has_speed_map:
            print(
                f"[HB][z2_calib] objective.{target_key}: {current_ref:.6f} -> {float(calibrated_ref):.6f}; "
                f"by_load={obj.get(f'{target_key}_by_load', {})}; "
                f"by_speed={obj.get(f'{target_key}_by_speed', {})}",
                flush=True,
            )
        else:
            print(
                f"[HB][z2_calib] objective.{target_key}: {current_ref:.6f} -> {float(calibrated_ref):.6f}",
                flush=True,
            )
        if write_back_env_yaml:
            with open(args.env_yaml, "w", encoding="utf-8") as fw:
                yaml.safe_dump(env_cfg_dict, fw, sort_keys=False, allow_unicode=True)
            print(f"[HB][z2_calib] {args.env_yaml}", flush=True)


def _t6_segment_stats(arr: np.ndarray) -> dict:
    """ episode  reward """
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out = {
        "n_points": int(n),
        "early_mean": float("nan"),
        "mid_mean": float("nan"),
        "late_mean": float("nan"),
        "max_drawdown_pct": float("nan"),
    }
    if n < 3:
        return out
    i1 = int(max(1, round(0.30 * n)))
    i2 = int(max(i1 + 1, round(0.75 * n)))
    i2 = min(i2, n - 1)
    early = x[:i1]
    mid = x[i1:i2]
    late = x[i2:]
    e = float(np.nanmean(early)) if early.size > 0 else float("nan")
    m = float(np.nanmean(mid)) if mid.size > 0 else float("nan")
    l = float(np.nanmean(late)) if late.size > 0 else float("nan")
    peak = float(np.nanmax(x)) if x.size > 0 else float("nan")
    dd = float((peak - l) / max(abs(peak), 1e-9) * 100.0) if np.isfinite(peak) and np.isfinite(l) else float("nan")
    out.update(
        {
            "early_mean": e,
            "mid_mean": m,
            "late_mean": l,
            "max_drawdown_pct": dd,
        }
    )
    return out


def _write_t6_artifacts(run_dir: Path, train_cfg: dict) -> None:
    """
     PhaseT6 
    - train_acceptance_t6.json
    - diag/reward_contract_t6.csv
    - diag/lagrange_trace_t6.csv
    """
    metrics_csv = run_dir / "train_metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f" train_metrics.csv{metrics_csv}")

    m = read_train_metrics_csv(metrics_csv)
    if not m:
        raise RuntimeError(" train_metrics.csv ")

    def _series(key: str) -> np.ndarray:
        return np.asarray(m.get(key, np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)

    train_episode = _series("train_episode")
    if train_episode.size <= 0:
        train_episode = np.arange(1, int(max(_series("paper_cost_ep").size, _series("reward_total_ep").size)) + 1, dtype=np.float64)

    paper_cost_ep = _series("paper_cost_ep")
    reward_total_ep = _series("reward_total_ep")
    reward_main = _series("reward_paper_step_mean")
    reward_potential = _series("reward_shaping_step_mean")
    reward_constraint = _series("reward_constraint_step_mean")
    reward_traj = _series("reward_misc_step_mean")
    violation_cont = _series("constraint_signal_step_mean")
    lambda_eff = _series("lambda_v_effective_mean")

    
    det_cost_ep = _series("eval_det_paper_cost_mean")
    det_valid_flag = _series("eval_det_is_valid")
    det_mask = np.isfinite(det_cost_ep)
    if det_valid_flag.size == det_cost_ep.size and np.isfinite(det_valid_flag).any():
        det_mask = det_mask & (det_valid_flag > 0.5)
    det_cost_valid = det_cost_ep[det_mask]
    metric1_window_size = 0
    if int(det_cost_valid.size) >= 2:
        metric1_window_size = int(max(1, round(0.05 * float(det_cost_valid.size))))
        metric1_window_size = int(min(max(metric1_window_size, 1), max(1, int(det_cost_valid.size) // 2)))
        paper_first = float(np.nanmean(det_cost_valid[:metric1_window_size]))
        paper_last = float(np.nanmean(det_cost_valid[-metric1_window_size:]))
        relative_drop_pct = float((paper_first - paper_last) / max(abs(paper_first), 1e-9) * 100.0)
    else:
        paper_first = float("nan")
        paper_last = float("nan")
        relative_drop_pct = float("nan")

    def _last_finite(arr: np.ndarray) -> float:
        x = np.asarray(arr, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        return float(x[-1]) if x.size > 0 else float("nan")

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        xa = np.asarray(a, dtype=np.float64).reshape(-1)
        xb = np.asarray(b, dtype=np.float64).reshape(-1)
        n = int(min(xa.size, xb.size))
        if n < 3:
            return float("nan")
        xa = xa[:n]
        xb = xb[:n]
        m = np.isfinite(xa) & np.isfinite(xb)
        xa = xa[m]
        xb = xb[m]
        if xa.size < 3:
            return float("nan")
        if float(np.nanstd(xa)) < 1e-12 or float(np.nanstd(xb)) < 1e-12:
            return float("nan")
        return float(np.corrcoef(xa, xb)[0, 1])

    reward_stats = _t6_segment_stats(reward_total_ep)
    bc_cfg = (((train_cfg or {}).get("train", {}) or {}).get("bc_warmstart", {}) or {})
    eval_det_imp = _last_finite(_series("eval_det_imp_paper_cost_mean"))
    train_load_current = _series("train_load_current")
    reward_cost_corr_ep = _safe_corr(reward_total_ep, paper_cost_ep)
    paper_load_corr_ep = _safe_corr(paper_cost_ep, train_load_current)
    reward_load_corr_ep = _safe_corr(reward_total_ep, train_load_current)
    
    v2_gate_v1_th = 0.0
    v2_gate_v2_th = 0.0

    metric1_source = "missing_train_metric1_contract_guard"
    metric1_n_points = 0
    metric1_ckpt = ""
    metric1_axis_source = "missing_train_metric1_contract_guard"
    metric1_guard_path = run_dir / "train_metric1_contract_guard.json"
    if metric1_guard_path.exists():
        try:
            m1_obj = json.loads(metric1_guard_path.read_text(encoding="utf-8"))
            metric1_source = str(m1_obj.get("metric1_source", metric1_source))
            metric1_n_points = int(m1_obj.get("n_points", 0) or 0)
            metric1_ckpt = str(m1_obj.get("checkpoint_used", "") or "")
            metric1_axis_source = str(m1_obj.get("episode_axis_source", metric1_axis_source))
            metric1_window_size = int(m1_obj.get("window_size", metric1_window_size) or metric1_window_size)
            g_first = float(m1_obj.get("paper_cost_first_5pct_mean", m1_obj.get("paper_cost_first_window_mean", float("nan"))))
            g_last = float(m1_obj.get("paper_cost_last_5pct_mean", m1_obj.get("paper_cost_last_window_mean", float("nan"))))
            g_drop = float(m1_obj.get("relative_drop_pct", float("nan")))
            if np.isfinite(g_first):
                paper_first = g_first
            if np.isfinite(g_last):
                paper_last = g_last
            if np.isfinite(g_drop):
                relative_drop_pct = g_drop
        except Exception:
            pass

    acceptance_t6 = {
        "paper_cost_first_5pct_mean": paper_first,
        "paper_cost_last_5pct_mean": paper_last,
        "paper_cost_first_10pct_mean": paper_first,
        "paper_cost_last_10pct_mean": paper_last,
        "paper_cost_first_window_mean": paper_first,
        "paper_cost_last_window_mean": paper_last,
        
        "paper_cost_first": paper_first,
        "paper_cost_last": paper_last,
        "relative_drop_pct": relative_drop_pct,
        "metric1_window_frac": 0.05,
        "metric1_window_size": int(metric1_window_size),
        "reward_total_ep_stats": reward_stats,
        "metric1_source": metric1_source,
        "metric1_axis_source": metric1_axis_source,
        "metric1_n_points": int(metric1_n_points),
        "metric1_checkpoint_used": metric1_ckpt,
        "eval_det_imp_paper_cost_mean": eval_det_imp,
        "reward_paper_cost_corr_ep": reward_cost_corr_ep,
        "paper_load_corr_ep": paper_load_corr_ep,
        "reward_load_corr_ep": reward_load_corr_ep,
        "v2_precheck_v1_threshold": v2_gate_v1_th,
        "v2_precheck_v2_threshold": v2_gate_v2_th,
        "v2_precheck_v1_pass": bool(np.isfinite(eval_det_imp) and (eval_det_imp > v2_gate_v1_th)),
        "v2_precheck_v2_pass": bool(np.isfinite(reward_cost_corr_ep) and (reward_cost_corr_ep < v2_gate_v2_th)),
        "lambda_v_effective_tail_mean": _safe_float(np.nanmean(lambda_eff[-min(20, lambda_eff.size):])) if lambda_eff.size > 0 else float("nan"),
        "violation_continuous_tail_mean": _safe_float(np.nanmean(violation_cont[-min(20, violation_cont.size):])) if violation_cont.size > 0 else float("nan"),
        "bc_warmstart": {
            "enable": bool(bc_cfg.get("enable", False)),
            "weight0": _safe_float(bc_cfg.get("weight0", float("nan"))),
            "decay_steps": int(bc_cfg.get("decay_steps", 0)) if str(bc_cfg.get("decay_steps", "")).strip() else 0,
        },
    }
    (run_dir / "train_acceptance_t6.json").write_text(
        json.dumps(acceptance_t6, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    diag_dir = run_dir / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    
    n = int(
        max(
            train_episode.size,
            paper_cost_ep.size,
            reward_total_ep.size,
            reward_main.size,
            reward_potential.size,
            reward_constraint.size,
            reward_traj.size,
            violation_cont.size,
        )
    )
    if n > 0:
        def _pad(arr: np.ndarray) -> np.ndarray:
            x = np.asarray(arr, dtype=np.float64).reshape(-1)
            if x.size >= n:
                return x[:n]
            if x.size == 0:
                return np.full((n,), np.nan, dtype=np.float64)
            tail = float(x[-1]) if np.isfinite(x[-1]) else float("nan")
            return np.pad(x, (0, n - x.size), mode="constant", constant_values=tail).astype(np.float64)

        ep = _pad(train_episode if train_episode.size > 0 else np.arange(1, n + 1, dtype=np.float64))
        pc = _pad(paper_cost_ep)
        rr = _pad(reward_total_ep)
        rm = _pad(reward_main)
        rp = _pad(reward_potential)
        rc = _pad(reward_constraint)
        rt = _pad(reward_traj)
        vg = _pad(violation_cont)
        with open(diag_dir / "reward_contract_t6.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "train_episode",
                    "paper_cost_ep",
                    "reward_total_ep",
                    "reward_main_mean",
                    "reward_potential_mean",
                    "reward_constraint_mean",
                    "reward_traj_mean",
                    "violation_continuous_mean",
                ],
            )
            writer.writeheader()
            for i in range(n):
                writer.writerow(
                    {
                        "train_episode": float(ep[i]),
                        "paper_cost_ep": float(pc[i]),
                        "reward_total_ep": float(rr[i]),
                        "reward_main_mean": float(rm[i]),
                        "reward_potential_mean": float(rp[i]),
                        "reward_constraint_mean": float(rc[i]),
                        "reward_traj_mean": float(rt[i]),
                        "violation_continuous_mean": float(vg[i]),
                    }
                )

    
    n_lag = int(max(train_episode.size, lambda_eff.size, violation_cont.size))
    if n_lag > 0:
        def _pad_l(arr: np.ndarray) -> np.ndarray:
            x = np.asarray(arr, dtype=np.float64).reshape(-1)
            if x.size >= n_lag:
                return x[:n_lag]
            if x.size == 0:
                return np.full((n_lag,), np.nan, dtype=np.float64)
            tail = float(x[-1]) if np.isfinite(x[-1]) else float("nan")
            return np.pad(x, (0, n_lag - x.size), mode="constant", constant_values=tail).astype(np.float64)

        ep = _pad_l(train_episode if train_episode.size > 0 else np.arange(1, n_lag + 1, dtype=np.float64))
        lam = _pad_l(lambda_eff)
        vio = _pad_l(violation_cont)
        with open(diag_dir / "lagrange_trace_t6.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "train_episode",
                    "lambda_v_effective_mean",
                    "violation_continuous_mean",
                ],
            )
            writer.writeheader()
            for i in range(n_lag):
                writer.writerow(
                    {
                        "train_episode": float(ep[i]),
                        "lambda_v_effective_mean": float(lam[i]),
                        "violation_continuous_mean": float(vio[i]),
                    }
                )


def _write_metric1_contract_guard(run_dir: Path) -> dict:
    """
    1
    -  train_metrics.csv  eval_det_paper_cost_mean
    -  Det-eval  fixed_eval_summary.csvvariant=full
    - 
    """
    guard = {
        "pass": False,
        "reason": "unknown",
        "source_contract_pass": False,
        "required_source": "train_metrics/eval_det_paper_cost_mean(valid_only)",
        "required_axis": "train_metrics/global_step",
        "metric1_source": "missing_train_metrics/eval_det_paper_cost_mean(valid_only)",
        "episode_axis_source": "missing_train_metrics/global_step",
        "checkpoint_used": "",
        "n_points": 0,
        "window_frac": 0.05,
        "window_size": 0,
        "paper_cost_first_5pct_mean": float("nan"),
        "paper_cost_last_5pct_mean": float("nan"),
        "paper_cost_first_10pct_mean": float("nan"),
        "paper_cost_last_10pct_mean": float("nan"),
        "paper_cost_first_window_mean": float("nan"),
        "paper_cost_last_window_mean": float("nan"),
        "paper_cost_first": float("nan"),
        "paper_cost_last": float("nan"),
        "relative_drop_pct": float("nan"),
        "relative_drop_direction": "unknown",
        "performance_pass_nonworse": False,
        "performance_pass_drop_positive": False,
        "episode_axis_monotonic": False,
        "fixed_eval_summary_used": "",
    }

    def _apply_metric1(cost: np.ndarray, axis: np.ndarray, source: str, axis_source: str) -> bool:
        cc = np.asarray(cost, dtype=np.float64).reshape(-1)
        xx = np.asarray(axis, dtype=np.float64).reshape(-1)
        if cc.size != xx.size:
            return False
        m = np.isfinite(cc) & np.isfinite(xx)
        cc = cc[m]
        xx = xx[m]
        if cc.size < 2:
            return False
        monotonic = bool(np.all(np.diff(xx) >= 0.0))
        if not monotonic:
            return False
        w = int(max(1, round(0.05 * float(cc.size))))
        w = int(min(max(w, 1), max(1, cc.size // 2)))
        first = float(np.nanmean(cc[:w]))
        last = float(np.nanmean(cc[-w:]))
        drop = float((first - last) / max(abs(first), 1e-9) * 100.0)
        guard.update(
            {
                "pass": True,
                "reason": "ok",
                "source_contract_pass": True,
                "metric1_source": str(source),
                "episode_axis_source": str(axis_source),
                "n_points": int(cc.size),
                "window_size": int(w),
                "paper_cost_first_5pct_mean": first,
                "paper_cost_last_5pct_mean": last,
                
                "paper_cost_first_10pct_mean": first,
                "paper_cost_last_10pct_mean": last,
                "paper_cost_first_window_mean": first,
                "paper_cost_last_window_mean": last,
                "paper_cost_first": first,
                "paper_cost_last": last,
                "relative_drop_pct": drop,
                "relative_drop_direction": ("decrease" if drop > 0.0 else ("flat" if abs(drop) <= 1e-9 else "increase")),
                "performance_pass_nonworse": bool(np.isfinite(drop) and (drop >= 0.0)),
                "performance_pass_drop_positive": bool(np.isfinite(drop) and (drop > 0.0)),
                "episode_axis_monotonic": True,
            }
        )
        return True

    
    metrics_csv = run_dir / "train_metrics.csv"
    if metrics_csv.exists():
        try:
            m = read_train_metrics_csv(metrics_csv)
            det = np.asarray(m.get("eval_det_paper_cost_mean", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
            det_valid = np.asarray(m.get("eval_det_is_valid", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
            axis = np.asarray(m.get("global_step", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
            if axis.size != det.size:
                axis = np.asarray(m.get("train_episode", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
            if axis.size != det.size:
                axis = np.arange(1, int(det.size) + 1, dtype=np.float64)

            mask = np.isfinite(det)
            if det_valid.size == det.size and np.isfinite(det_valid).any():
                mask = mask & (det_valid > 0.5)

            if _apply_metric1(
                cost=det[mask],
                axis=axis[mask],
                source="train_metrics/eval_det_paper_cost_mean(valid_only)",
                axis_source=("train_metrics/global_step" if np.asarray(m.get("global_step", np.asarray([], dtype=np.float64))).size == det.size else "train_metrics/train_episode_or_index"),
            ):
                guard["checkpoint_used"] = ""
        except Exception:
            pass

    
    if not bool(guard.get("pass", False)):
        eval_root = run_dir / "evals"
        cands: list[Path] = []
        try:
            cands = sorted(list(eval_root.glob("*/fixed_eval_summary*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            cands = []
        for p in cands:
            try:
                xs: list[float] = []
                ys: list[float] = []
                with p.open("r", encoding="utf-8", newline="") as f:
                    rr = csv.DictReader(f)
                    for row in rr:
                        v = str(row.get("variant", "full")).strip().lower()
                        if v != "full":
                            continue
                        try:
                            y = float(row.get("paper_cost_mean", float("nan")))
                        except Exception:
                            continue
                        if not np.isfinite(y):
                            continue
                        x = float("nan")
                        for key in ("eval_train_episode", "ckpt_step"):
                            try:
                                x = float(row.get(key, float("nan")))
                            except Exception:
                                x = float("nan")
                            if np.isfinite(x):
                                break
                        if not np.isfinite(x):
                            x = float(len(xs) + 1)
                        xs.append(float(x))
                        ys.append(float(y))
                if len(xs) < 2:
                    continue
                order = np.argsort(np.asarray(xs, dtype=np.float64))
                xx = np.asarray(xs, dtype=np.float64)[order]
                yy = np.asarray(ys, dtype=np.float64)[order]
                if _apply_metric1(
                    cost=yy,
                    axis=xx,
                    source=f"fixed_eval_summary/full@{p}",
                    axis_source="fixed_eval_summary/eval_train_episode_or_ckpt_step",
                ):
                    guard["reason"] = "ok_fallback_fixed_eval"
                    guard["fixed_eval_summary_used"] = str(p)
                    break
            except Exception:
                continue

    if not bool(guard.get("pass", False)):
        guard["reason"] = "missing_valid_det_eval_and_fixed_eval"

    out_path = run_dir / "train_metric1_contract_guard.json"
    out_path.write_text(json.dumps(guard, ensure_ascii=False, indent=2), encoding="utf-8")
    return guard

def _resolve_cfg_path(raw_path: str | None, default_rel: str) -> Path:
    text = str(raw_path).strip() if raw_path is not None else ""
    if not text:
        text = default_rel
    p = Path(text)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _display_cfg_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p.resolve())


def _resolve_optional_path(raw_path: str | None) -> str:
    text = str(raw_path).strip() if raw_path is not None else ""
    if not text:
        return ""
    p = Path(text)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return str(p)


def _parse_probe_loads(raw: str, fallback: str) -> str:
    """
     fixed-eval probe 
    
    """
    txt = str(raw or "").strip()
    if not txt:
        txt = str(fallback or "").strip()
    vals = []
    for part in txt.split(","):
        ss = str(part).strip()
        if not ss:
            continue
        try:
            vv = float(ss)
        except Exception as exc:
            raise ValueError(f"fixed_eval_probe_loads {ss}") from exc
        if (not np.isfinite(vv)) or vv <= 0.0:
            raise ValueError(f"fixed_eval_probe_loads {vv}")
        vals.append(float(vv))
    if not vals:
        raise ValueError("fixed_eval_probe_loads ")
    
    out = []
    seen = set()
    for v in vals:
        k = round(float(v), 6)
        if k in seen:
            continue
        seen.add(k)
        out.append(float(v))
    return ",".join([f"{v:g}" for v in out])


def _run_phasez_fixed_eval_probe_if_needed(
    *,
    run_dir: Path,
    train_cfg: dict,
    env_yaml: str,
    train_yaml: str,
    is_phasez_cfg: bool,
    total_steps_cfg: int,
) -> dict | None:
    """
    Phase Z1  G330k/100k  deterministic fixed-eval 
    """
    if not bool(is_phasez_cfg):
        return None
    train_params = (train_cfg.get("train", {}) or {}) if isinstance(train_cfg, dict) else {}
    probe_enable = bool(train_params.get("fixed_eval_probe_enable", False))
    if not probe_enable:
        return None

    
    if int(total_steps_cfg) < 24000:
        print(
            f"[HB][train][PhaseZ] fixed-eval probe  total_steps={int(total_steps_cfg)}<24000",
            flush=True,
        )
        return {
            "enabled": True,
            "ran": False,
            "reason": "steps_lt_24000",
        }

    stage_tag = "30k" if int(total_steps_cfg) <= 60000 else "100k"
    loads_key = "fixed_eval_probe_loads_30k" if stage_tag == "30k" else "fixed_eval_probe_loads_100k"
    raw_loads = str(train_params.get(loads_key, "")).strip()
    if not raw_loads:
        
        legacy_key = "fixed_eval_probe_loads_10k" if stage_tag == "30k" else "fixed_eval_probe_loads_50k"
        raw_loads = str(train_params.get(legacy_key, "")).strip()
    fallback_loads = "0.8" if stage_tag == "30k" else "0.6,0.8,1.0"
    loads_text = _parse_probe_loads(raw_loads, fallback_loads)

    ckpt_best = run_dir / "checkpoints" / "ckpt_best.pt"
    ckpt = ckpt_best
    if not ckpt.exists():
        raise FileNotFoundError(
            f"[HB][train][PhaseZ][FATAL] fixed-eval probe  best checkpoint{ckpt_best}"
        )

    eval_ts = f"phasez1_probe_{stage_tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "eval_policy_joint.py").resolve()),
        "--run_dir", str(run_dir),
        "--env-yaml", str(env_yaml),
        "--train-yaml", str(train_yaml),
        "--fixed_eval", "1",
        "--fixed_eval_ckpt", str(ckpt),
        "--ckpt_path", str(ckpt),
        "--fixed_eval_require_final_ckpt", "0",
        "--fixed_eval_variants", "full",
        "--seed_set", "3",
        "--episodes_per_seed", "1",
        "--deterministic", "1",
        "--eval_loads", str(loads_text),
        "--eval_ts", str(eval_ts),
    ]
    print(
        "[HB][train][PhaseZ] fixed-eval probe "
        f"stage={stage_tag} loads={loads_text} ckpt={ckpt} eval_ts={eval_ts}",
        flush=True,
    )
    t0 = datetime.now()
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    dt = (datetime.now() - t0).total_seconds()
    summary_csv = run_dir / "evals" / eval_ts / "fixed_eval_summary.csv"
    fig_path = run_dir / "figs" / f"{eval_ts} evalratio" / "FixedEval_PaperCost.png"
    payload = {
        "enabled": True,
        "ran": True,
        "stage": stage_tag,
        "eval_ts": eval_ts,
        "loads": loads_text,
        "ckpt": str(ckpt),
        "elapsed_sec": float(dt),
        "fixed_eval_summary_csv": str(summary_csv),
        "fixed_eval_fig": str(fig_path),
    }
    out_path = run_dir / "phasez_fixed_eval_probe.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HB][train][PhaseZ] fixed-eval probe {out_path}", flush=True)
    return payload


def _norm_rel_text(text: str | None) -> str:
    s = str(text or "").strip().replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s


def _first_existing_cfg(candidates: list[str]) -> Path | None:
    for rel in candidates:
        p = _resolve_cfg_path(rel, rel)
        if p.exists():
            return p
    return None


def _normalize_cfg_inputs(args: argparse.Namespace) -> None:
    
    
    
    
    train_default = "configs/PhaseZ4/train_step7_1200k.yaml"
    train_fallback = "configs/PhaseZ4/train_step3_50k.yaml"
    train_raw = str(getattr(args, "train_yaml", "") or "").strip()
    train_path = _resolve_cfg_path(train_raw, train_default)
    if not train_path.exists():
        train_raw_n = _norm_rel_text(train_raw)
        if train_raw_n in ("", _norm_rel_text(train_default)):
            auto_train = _first_existing_cfg([train_default, train_fallback])
            if auto_train is None:
                raise FileNotFoundError(
                    ""
                    f" { _resolve_cfg_path(train_default, train_default) }  { _resolve_cfg_path(train_fallback, train_fallback) }"
                )
            print(
                "[HB][train][WARN] "
                f" {_display_cfg_path(auto_train)}",
                flush=True,
            )
            train_path = auto_train
        else:
            raise FileNotFoundError(f"{train_path}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_cfg_probe = yaml.safe_load(f) or {}
    train_env_raw = ""
    if isinstance(train_cfg_probe, dict):
        env_block = train_cfg_probe.get("env", {}) or {}
        if isinstance(env_block, dict):
            train_env_raw = str(env_block.get("config_path", "")).strip()
    env_default = "configs/PhaseZ4/env_phaseZ4.yaml"
    env_fallback = "configs/PhaseZ4/env_phaseZ4.yaml"
    train_env_path = _resolve_cfg_path(train_env_raw, env_default)
    train_env_raw_n = _norm_rel_text(train_env_raw)
    if not train_env_path.exists():
        if train_env_raw_n in ("", _norm_rel_text(env_default)):
            auto_env = _first_existing_cfg([env_default, env_fallback])
            if auto_env is not None:
                train_env_path = auto_env
            elif train_env_raw_n:
                raise FileNotFoundError(f"train.env.config_path {train_env_path}")
        else:
            raise FileNotFoundError(f"train.env.config_path {train_env_path}")

    
    env_arg_raw = getattr(args, "env_yaml", None)
    if env_arg_raw is None:
        cli_env_raw = ""
    else:
        cli_env_raw = str(env_arg_raw).strip()
        if cli_env_raw.lower() in ("none", "null"):
            cli_env_raw = ""
    if cli_env_raw:
        env_path = _resolve_cfg_path(cli_env_raw, env_default)
        if not env_path.exists():
            raise FileNotFoundError(f"{env_path}")
        env_source = "cli"
    elif train_env_raw:
        env_path = train_env_path
        env_source = "train.env.config_path"
    else:
        auto_env = _first_existing_cfg([env_default, env_fallback])
        if auto_env is None:
            raise FileNotFoundError(
                ""
                f" { _resolve_cfg_path(env_default, env_default) }  { _resolve_cfg_path(env_fallback, env_fallback) }"
            )
        env_path = auto_env
        env_source = "default"

    
    if cli_env_raw and train_env_raw:
        cli_path = _resolve_cfg_path(cli_env_raw, "configs/env_fixed.yaml")
        if cli_path.resolve() != train_env_path.resolve():
            if bool(getattr(args, "allow_env_mismatch", False)):
                print(
                    "[HB][train][WARN] --env-yaml  train.env.config_path  CLI "
                    f" cli={_display_cfg_path(cli_path)} train_env={_display_cfg_path(train_env_path)}",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    "--env-yaml  train.env.config_path \n"
                    f"- cli_env_yaml: {_display_cfg_path(cli_path)}\n"
                    f"- train_env_config_path: {_display_cfg_path(train_env_path)}\n"
                    "\n"
                    "1)  --env-yaml train.env.config_path\n"
                    "2) \n"
                    "3)  --allow-env-mismatch "
                )

    if not env_path.exists():
        raise FileNotFoundError(f"{env_path}")

    args.train_yaml = str(train_path.resolve())
    args.env_yaml = str(env_path.resolve())
    args._cfg_train_display = _display_cfg_path(train_path)
    args._cfg_env_display = _display_cfg_path(env_path)
    args._cfg_env_source = env_source
    args._cfg_train_env_raw = train_env_raw


def train(args) -> None:
    """ Tianshou PPO """
    run_dir = make_run_dir(PROJECT_ROOT, subdir="runs/paper", ts_name=args.run_name)
    ensure_run_subdirs(run_dir)
    print(f"[HB][train] run_dir = {run_dir}", flush=True)

    
    with open(args.env_yaml, "r", encoding="utf-8") as f:
        env_cfg_dict = yaml.safe_load(f) or {}
    with open(args.train_yaml, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f) or {}
    
    if bool((train_cfg or {}).get("deprecated", False)):
        repl = (train_cfg or {}).get("replacement", [])
        raise RuntimeError(
            "[HB][train][FATAL]  train yaml "
            f"{repl if isinstance(repl, list) else ['configs/PhaseZ4/train_step3_50k.yaml', 'configs/PhaseZ4/train_step4_150k.yaml', 'configs/PhaseZ4/train_step5_300k.yaml', 'configs/PhaseZ4/train_step6_800k.yaml']}"
        )

    
    if getattr(args, "seed", None) is not None:
        seed_cli = int(args.seed)
        train_cfg["seed"] = seed_cli
        
        env_cfg_dict["seed"] = seed_cli

    
    train_cfg.setdefault("train", {})
    if args.save_every is not None:
        train_cfg["train"]["save_every"] = int(args.save_every)
    if args.plot_every is not None:
        train_cfg["train"]["plot_every"] = int(args.plot_every)
    if getattr(args, "total_steps", None) is not None:
        train_cfg["train"]["total_steps"] = int(args.total_steps)
    if getattr(args, "eval_every", None) is not None:
        train_cfg["train"]["eval_every"] = int(args.eval_every)
    if getattr(args, "eval_episodes", None) is not None:
        train_cfg.setdefault("eval", {})
        train_cfg["eval"]["eval_episodes"] = int(args.eval_episodes)
    
    if getattr(args, "resume_path", None) is not None:
        resume_raw = str(args.resume_path).strip()
        train_cfg.setdefault("misc", {})
        if resume_raw.lower() in ("", "none", "null"):
            train_cfg["misc"]["resume_path"] = ""
        else:
            resume_path_abs = _resolve_optional_path(resume_raw)
            if not Path(resume_path_abs).exists():
                raise FileNotFoundError(f"resume_path {resume_path_abs}")
            train_cfg["misc"]["resume_path"] = resume_path_abs
    
    if getattr(args, "policy_init_path", None) is not None:
        init_raw = str(args.policy_init_path).strip()
        train_cfg.setdefault("misc", {})
        if init_raw.lower() in ("", "none", "null"):
            train_cfg["misc"]["policy_init_path"] = ""
        else:
            init_path_abs = _resolve_optional_path(init_raw)
            if not Path(init_path_abs).exists():
                raise FileNotFoundError(f"policy_init_path {init_path_abs}")
            train_cfg["misc"]["policy_init_path"] = init_path_abs
    
    misc_cfg = train_cfg.get("misc", {}) if isinstance(train_cfg, dict) else {}
    if not isinstance(misc_cfg, dict):
        misc_cfg = {}
        train_cfg["misc"] = misc_cfg
    if bool(misc_cfg.get("forbid_resume", False)):
        resume_now = str(misc_cfg.get("resume_path", "") or "").strip()
        if resume_now:
            raise RuntimeError(
                "[HB][train][FATAL] misc.forbid_resume=1"
                f" resume_path={resume_now}"
            )
    _apply_meta_cli_overrides(train_cfg, args)
    _apply_t6_cli_overrides(train_cfg, args)
    _maybe_auto_calibrate_z_cost_ref(
        args=args,
        run_dir=run_dir,
        env_cfg_dict=env_cfg_dict,
        train_cfg=train_cfg,
    )

    
    snapshot_cfgs(run_dir, env_cfg_dict, train_cfg)
    env_display = str(getattr(args, "_cfg_env_display", args.env_yaml))
    train_display = str(getattr(args, "_cfg_train_display", args.train_yaml))
    env_source = str(getattr(args, "_cfg_env_source", "unknown"))
    train_env_raw = str(getattr(args, "_cfg_train_env_raw", "")).strip()
    config_lines = [
        f"env_yaml: {env_display}",
        f"train_yaml: {train_display}",
        f"env_source: {env_source}",
        f"env_yaml_resolved: {args.env_yaml}",
        f"train_yaml_resolved: {args.train_yaml}",
    ]
    if train_env_raw:
        config_lines.append(f"train_env_config_path: {train_env_raw}")
    (run_dir / "config_paths.txt").write_text(
        "\n".join(config_lines) + "\n",
        encoding="utf-8",
    )
    try:
        (run_dir / "config_resolution.json").write_text(
            json.dumps(
                {
                    "env_yaml": env_display,
                    "train_yaml": train_display,
                    "env_source": env_source,
                    "env_yaml_resolved": str(args.env_yaml),
                    "train_yaml_resolved": str(args.train_yaml),
                    "train_env_config_path": train_env_raw,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[HB][train][WARN] config_resolution.json {type(e).__name__}: {e}", flush=True)

    
    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), text=True)
        msg = [
            f"commit: {commit}",
            f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "dirty: " + ("yes" if status.strip() else "no"),
            "",
            "git status --porcelain:",
            status.rstrip(),
            "",
        ]
        (run_dir / "git_commit.txt").write_text("\n".join(msg), encoding="utf-8")
    except Exception as e:
        print(f"[HB][train][WARN] git_commit.txt {type(e).__name__}: {e}", flush=True)

    
    seed = int(train_cfg.get("seed", 42))
    train_block = train_cfg.get("train", {}) if isinstance(train_cfg, dict) else {}
    deterministic_seed = bool((train_block or {}).get("deterministic_seed", True))
    set_global_seed(seed, deterministic=deterministic_seed)
    print(
        f"[HB][train] seed={seed} deterministic_seed={int(deterministic_seed)}",
        flush=True,
    )

    
    device_str = str(train_cfg.get("device", "cuda"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[HB][train] Using device: {device}", flush=True)

    
    env_cfg, env_cfg_dict_clean, dropped = build_joint_env_cfg(env_cfg_dict)
    print(f"[HB][train] env_cfg keys: kept={len(env_cfg_dict_clean)} dropped={len(dropped)}", flush=True)
    if dropped:
        print(f"[HB][train] dropped keys (first 20): {dropped[:20]}", flush=True)
    
    
    if "action_space_mode" in set(dropped):
        raise RuntimeError(
            "[HB][train][FATAL] env config key 'action_space_mode' was dropped by build_joint_env_cfg. "
            ""
        )
    
    run_name_l = str(getattr(args, "run_name", "") or "").strip().lower()
    rd = str(getattr(env_cfg, "reward_design", "")).strip().lower()
    is_phaseg_cfg = rd in ("reward_total_v1", "reward_total", "total_v1", "v1")
    is_t6_cfg = rd == "t6_potential_lagrangian"
    if rd in ("z_shifted_log", "z4_linear", "z5_saturating"):
        dropped_set = set([str(k).strip() for k in dropped])
        phasez_required = {
            "obs_enable_prev_action",
            "z_cost_ref",
            "theta_mode",
            "coverage_floor_weight",
            "coverage_floor_threshold",
        }
        if rd == "z_shifted_log":
            phasez_required.update({
                "z_alpha",
                "z_beta",
                "z_gamma",
                "comp_score_temp",
                "reward_proximity_weight",
                "reward_proximity_weight_final",
                "reward_proximity_decay_start_step",
                "reward_proximity_decay_end_step",
                "reward_proximity_mode",
            })
        if rd == "z4_linear":
            phasez_required.update({
                "z4_reward_offset",
                "z4_reward_alpha",
                "z4_bonus_gamma",
                "z4_bonus_anchor",
                "z4_bonus_power",
                "z4_action_mode",
                "comp_rule_threshold",
                "reward_clip_range",
            })
        if rd == "z5_saturating":
            phasez_required.update({
                "z5_r_max",
                "z5_kappa",
                "z5_anchor_norm",
                "reward_proximity_weight",
                "reward_proximity_weight_final",
                "reward_proximity_decay_start_step",
                "reward_proximity_decay_end_step",
                "reward_proximity_mode",
                "reward_clip_range",
            })
        theta_mode_now = str(getattr(env_cfg, "theta_mode", "solver")).strip().lower()
        if theta_mode_now in ("agent", "policy"):
            phasez_required.add("theta_softmax_tau")
        if bool(getattr(env_cfg, "comp_temp_anneal_enable", False)):
            phasez_required.update({"comp_temp_stage_0", "comp_temp_stage_1", "comp_temp_stage_2"})
        phasez_dropped = sorted([k for k in phasez_required if k in dropped_set])
        if phasez_dropped:
            raise RuntimeError(
                "[HB][train][FATAL] PhaseZ  build_joint_env_cfg "
                f"{phasez_dropped}"
                "PhaseZ"
            )
    if rd in ("z_shifted_log", "z4_linear", "z5_saturating"):
        train_block_cfg = ((train_cfg or {}).get("train", {}) or {})
        probe_enable = bool(train_block_cfg.get("fixed_eval_probe_enable", False))
        if probe_enable:
            z_probe_keys = (
                "fixed_eval_probe_loads_30k",
                "fixed_eval_probe_loads_100k",
                "log_returns_stats",
                "log_theta_comp_coupling",
            )
            missing_probe_keys = [k for k in z_probe_keys if k not in train_block_cfg]
            if missing_probe_keys:
                raise RuntimeError(
                    "[HB][train][FATAL] PhaseZ fixed-eval probe "
                    f"{missing_probe_keys}"
                )
            raw_30k = str(train_block_cfg.get("fixed_eval_probe_loads_30k", "")).strip()
            if not raw_30k:
                raw_30k = str(train_block_cfg.get("fixed_eval_probe_loads_10k", "")).strip()
            raw_100k = str(train_block_cfg.get("fixed_eval_probe_loads_100k", "")).strip()
            if not raw_100k:
                raw_100k = str(train_block_cfg.get("fixed_eval_probe_loads_50k", "")).strip()
            _ = _parse_probe_loads(raw_30k, "0.8")
            _ = _parse_probe_loads(raw_100k, "0.6,0.8,1.0")
    if ("phaseg" in run_name_l) and (not is_phaseg_cfg):
        raise RuntimeError(
            f"run_name contains 'phaseg' but env.reward_design='{rd}'. "
            "This run would be evaluated as non-PhaseG (typically PhaseB). "
            "Set env.reward_design=reward_total_v1 or rename run_name."
        )
    if is_t6_cfg:
        
        
        train_block_cfg = ((train_cfg or {}).get("train", {}) or {})
        lag_cfg_t6 = (train_block_cfg.get("lagrange_vio", {}) or {})
        if bool(lag_cfg_t6.get("enabled", False)):
            lag_cfg_t6["enabled"] = False
            train_block_cfg["lagrange_vio"] = lag_cfg_t6
            train_cfg["train"] = train_block_cfg
            print(
                "[HB][train][T6][GUARD]  train.lagrange_vio.enabled=1"
                "0 t6_pid ",
                flush=True,
            )
        bc_cfg = (((train_cfg or {}).get("train", {}) or {}).get("bc_warmstart", {}) or {})
        print(
            "[HB][train][T6] reward_design=t6_potential_lagrangian "
            f"bc_enable={int(bool(bc_cfg.get('enable', False)))} "
            f"bc_weight0={bc_cfg.get('weight0', 'NA')} "
            f"bc_decay_steps={bc_cfg.get('decay_steps', 'NA')} "
            f"run_dir={run_dir}",
            flush=True,
        )

    
    from src.algos.tianshou.official_trainer import run_official_training

    print("[HB][train]  Collector+OnpolicyTrainer ", flush=True)
    res = run_official_training(
        run_dir=run_dir,
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        device=device,
        selfcheck_enabled=bool(not getattr(args, "no_selfcheck", False)),
    )
    trainer = res["trainer"]

    dense_eps = list(getattr(trainer, "train_episode_dense_history", []))
    dense_reward_total = list(getattr(trainer, "reward_total_ep_dense_history", []))
    dense_cost = list(getattr(trainer, "cost_ep_dense_history", []))
    dense_vio = list(getattr(trainer, "vio_ep_dense_history", []))
    dense_improve = list(getattr(trainer, "improve_ep_dense_history", []))
    dense_shaping = list(getattr(trainer, "shaping_ep_dense_history", []))
    dense_comp = list(getattr(trainer, "comp_ep_dense_history", []))
    dense_ris = list(getattr(trainer, "ris_ep_dense_history", []))
    dense_policy_loss = list(getattr(trainer, "policy_loss_ep_dense_history", []))
    dense_value_loss = list(getattr(trainer, "value_loss_ep_dense_history", []))
    dense_entropy = list(getattr(trainer, "entropy_ep_dense_history", []))
    dense_kl = list(getattr(trainer, "kl_ep_dense_history", []))
    dense_load = list(getattr(trainer, "load_ep_dense_history", []))
    dense_eval_cost = list(getattr(trainer, "eval_det_paper_cost_ep_dense_history", []))
    dense_eval_cost_ci = list(getattr(trainer, "eval_det_paper_cost_ci_ep_dense_history", []))
    dense_eval_vio_any = list(getattr(trainer, "eval_det_vio_any_frac_ep_dense_history", []))
    dense_eval_improve = list(getattr(trainer, "eval_det_improve_ep_dense_history", []))
    n_dense = int(len(dense_eps))

    use_dense_reward = bool(n_dense > 0 and len(dense_reward_total) == n_dense)
    use_dense_decomp = bool(
        n_dense > 0
        and len(dense_cost) == n_dense
        and len(dense_vio) == n_dense
        and len(dense_improve) == n_dense
        and len(dense_shaping) == n_dense
    )
    use_dense_train_metrics = bool(
        n_dense > 0
        and len(dense_policy_loss) == n_dense
        and len(dense_value_loss) == n_dense
        and len(dense_entropy) == n_dense
        and len(dense_kl) == n_dense
    )
    use_dense_eval_metrics = bool(
        n_dense > 0
        and len(dense_eval_cost) == n_dense
        and len(dense_eval_cost_ci) == n_dense
        and len(dense_eval_vio_any) == n_dense
        and len(dense_eval_improve) == n_dense
    )
    
    if use_dense_decomp:
        try:
            roll_cost = np.asarray(getattr(trainer, "cost_history", []), dtype=np.float64).reshape(-1)
            roll_cost = roll_cost[np.isfinite(roll_cost)]
            den_cost = np.asarray(dense_cost, dtype=np.float64).reshape(-1)
            den_cost = den_cost[np.isfinite(den_cost)]
            if roll_cost.size > 0 and den_cost.size > 0:
                med_roll = float(np.median(np.abs(roll_cost)))
                med_dense = float(np.median(np.abs(den_cost)))
                if med_roll > 1e-9 and med_dense > 20.0 * med_roll:
                    use_dense_decomp = False
                    print(
                        "[HB][train][plot][WARN] dense cost appears to be episode-SUM; "
                        "fallback to rollout-level decomposition for this run.",
                        flush=True,
                    )
        except Exception:
            pass

    
    zeros = [0.0 for _ in range(len(trainer.steps_history))]
    
    n_plot = int(len(getattr(trainer, "train_episode_history", [])))
    shaping_hist = [0.0 for _ in range(n_plot)]
    comp_hist = [0.0 for _ in range(n_plot)]
    ris_hist = [0.0 for _ in range(n_plot)]
    try:
        m_plot = read_train_metrics_csv(run_dir / "train_metrics.csv")

        def _pick_series(keys):
            for kk in keys:
                arr = np.asarray(m_plot.get(kk, np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
                if arr.size > 0 and np.isfinite(arr).any():
                    return arr
            return np.asarray([], dtype=np.float64)

        def _align_series(arr: np.ndarray, n: int) -> List[float]:
            if n <= 0:
                return []
            x = np.asarray(arr, dtype=np.float64).reshape(-1)
            if x.size <= 0:
                return [0.0 for _ in range(n)]
            if x.size < n:
                pad_val = float(x[-1]) if np.isfinite(x[-1]) else 0.0
                x = np.pad(x, (0, n - x.size), mode="constant", constant_values=pad_val)
            else:
                x = x[:n]
            x = np.where(np.isfinite(x), x, 0.0)
            return x.astype(np.float64).tolist()

        shaping_src = _pick_series(("total_shaping_all_mean", "total_shaping_mean", "reward_shaping_ep"))
        comp_src = _pick_series(("comp_bonus_mean",))
        ris_src = _pick_series(("ris_bonus_mean",))
        shaping_hist = _align_series(shaping_src, n_plot)
        comp_hist = _align_series(comp_src, n_plot)
        ris_hist = _align_series(ris_src, n_plot)

        sh_arr = np.asarray(shaping_hist, dtype=np.float64)
        if sh_arr.size > 0 and np.any(np.abs(sh_arr) > 1e-12):
            print(
                "[HB][train][plot] reward_decomposition  train_metrics.csv  TotalShaping/CoMP/RIS ",
                flush=True,
            )
        else:
            print("[HB][train][plot][WARN]  shaping reward_decomposition  0 ", flush=True)
    except Exception as e:
        print(f"[HB][train][plot][WARN]  shaping {type(e).__name__}: {e}", flush=True)

    train_reward_x_plot = dense_eps if use_dense_reward else list(getattr(trainer, "train_episode_history", []))
    train_reward_y_plot = dense_reward_total if use_dense_reward else list(getattr(trainer, "reward_history", []))
    if use_dense_reward:
        print(
            f"[HB][train][plot] training reward uses dense episode series: n={len(train_reward_x_plot)}",
            flush=True,
        )
    else:
        print(
            "[HB][train][plot][WARN] dense training reward missing; fallback to rollout-level reward_history",
            flush=True,
        )

    train_metric_x_plot = dense_eps if use_dense_train_metrics else list(getattr(trainer, "train_episode_history", []))
    train_policy_plot = dense_policy_loss if use_dense_train_metrics else list(getattr(trainer, "policy_loss_history", []))
    train_value_plot = dense_value_loss if use_dense_train_metrics else list(getattr(trainer, "value_loss_history", []))
    train_entropy_plot = dense_entropy if use_dense_train_metrics else list(getattr(trainer, "entropy_history", []))
    train_kl_plot = dense_kl if use_dense_train_metrics else list(getattr(trainer, "kl_div_history", []))
    if use_dense_train_metrics:
        print(
            f"[HB][train][plot] training_curves metrics use dense episode series: n={len(train_metric_x_plot)}",
            flush=True,
        )
    else:
        print(
            "[HB][train][plot][WARN] dense training metrics missing; fallback to rollout-level metrics",
            flush=True,
        )

    eval_x_plot = dense_eps if use_dense_eval_metrics else list(getattr(trainer, "eval_train_episode_history", []))
    eval_cost_plot = dense_eval_cost if use_dense_eval_metrics else list(getattr(trainer, "eval_paper_cost_history", []))
    eval_cost_ci_plot = dense_eval_cost_ci if use_dense_eval_metrics else list(getattr(trainer, "eval_paper_cost_ci_history", []))
    eval_vio_plot = dense_eval_vio_any if use_dense_eval_metrics else list(getattr(trainer, "eval_vio_any_frac_history", []))
    eval_improve_plot = dense_eval_improve if use_dense_eval_metrics else list(getattr(trainer, "eval_improve_history", []))
    eval_vio_ci_plot = list(getattr(trainer, "eval_vio_any_frac_ci_history", []))
    if use_dense_eval_metrics:
        print(
            f"[HB][train][plot] eval metrics use dense episode series: n={len(eval_x_plot)}",
            flush=True,
        )
    else:
        print(
            "[HB][train][plot][WARN] dense eval metrics missing; fallback to sparse det-eval points",
            flush=True,
        )

    if use_dense_decomp:
        decomp_x = dense_eps
        decomp_cost = dense_cost
        decomp_vio = dense_vio
        decomp_improve = dense_improve
        decomp_shaping = dense_shaping
        decomp_comp = dense_comp if len(dense_comp) == n_dense else None
        decomp_ris = dense_ris if len(dense_ris) == n_dense else None
        print(
            f"[HB][train][plot] reward_decomposition uses dense episode series: n={len(decomp_x)}",
            flush=True,
        )
    else:
        decomp_x = list(getattr(trainer, "train_episode_history", []))
        decomp_cost = list(getattr(trainer, "cost_history", []))
        decomp_vio = list(getattr(trainer, "vio_history", []))
        decomp_improve = list(getattr(trainer, "improve_history", []))
        decomp_shaping = list(shaping_hist)
        decomp_comp = list(comp_hist)
        decomp_ris = list(ris_hist)
        print(
            "[HB][train][plot][WARN] dense reward_decomposition missing; fallback to rollout-level decomposition",
            flush=True,
        )

    try:
        
        conv_parent_dir = run_dir / "figs" / "ConvergenceReward trainreward"
        conv_all_dir = conv_parent_dir / "all_loads"
        conv_parent_dir.mkdir(parents=True, exist_ok=True)
        conv_all_dir.mkdir(parents=True, exist_ok=True)
        deteval_out_dir = run_dir / "figs" / "DetEval_PaperCost trainpaper_cost"
        deteval_out_dir.mkdir(parents=True, exist_ok=True)

        plot_training_curves(
            run_dir=run_dir,
            reward_history=train_reward_y_plot,
            policy_loss_history=train_policy_plot,
            value_loss_history=train_value_plot,
            entropy_history=train_entropy_plot,
            kl_div_history=train_kl_plot,
            train_episode_history=train_metric_x_plot,
            reward_episode_history=train_reward_x_plot,
            eval_train_episode_history=getattr(trainer, "eval_train_episode_history", []),
            eval_reward_history=trainer.eval_reward_history,
        )
        plot_reward_decomposition(
            run_dir=run_dir,
            train_episode_history=decomp_x,
            cost_history=decomp_cost,
            vio_history=decomp_vio,
            improve_history=decomp_improve,
            shaping_history=decomp_shaping,
            comp_history=decomp_comp,
            ris_history=decomp_ris,
        )
        plot_eval_metrics(
            run_dir=run_dir,
            eval_train_episodes=eval_x_plot,
            eval_paper_cost=eval_cost_plot,
            eval_paper_cost_ci=eval_cost_ci_plot,
            eval_vio=eval_vio_plot,
            eval_improve=eval_improve_plot,
        )

        
        rd = str(getattr(env_cfg, "reward_design", "")).strip().lower()
        eval_cfg = train_cfg.get("eval", {}) if isinstance(train_cfg, dict) else {}
        eval_loads = eval_cfg.get("eval_loads", [1.0]) if isinstance(eval_cfg, dict) else [1.0]
        train_load = float(getattr(env_cfg, "load_scale", 1.0))
        horizon_t = int(getattr(env_cfg, "T", 0))
        eval_note = f"agg={'SUM' if rd in ('reward_total_v1', 'reward_total', 'total_v1', 'v1') else 'MEAN'}, train_load={train_load:.2f}, eval_loads={eval_loads}, T={horizon_t}"
        
        conv_train_eps = list(train_reward_x_plot)
        conv_train_rew = list(train_reward_y_plot)
        has_conv_reward = bool(len(conv_train_eps) > 0 and len(conv_train_rew) > 0)
        if has_conv_reward:
            conv_note = "reward_total_epepisode"
            if rd in ("reward_total_v1", "reward_total", "total_v1", "v1"):
                conv_note = "reward_total = reward_paper + reward_shaping + reward_constraint (+ misc)"
            
            for conv_span, conv_name in (
                (40, "Convergence_RewardTotal_span40.png"),
                (48, "Convergence_RewardTotal_span48.png"),
                
                (200, "Convergence_RewardTotal_span200_raw_ema.png"),
                
                (396, "Convergence_RewardTotal_span396_raw_ema.png"),
            ):
                plot_convergence_reward_total(
                    run_dir=run_dir,
                    train_episodes=conv_train_eps,
                    reward_total_mean=conv_train_rew,
                    reward_total_ci=None,
                    out_dir=conv_all_dir,
                    draw_ci=False,
                    ema_span=int(conv_span),
                    save_name=conv_name,
                    note=conv_note,
                    robust_display=True,
                    robust_clip_quantile=0.01,
                    robust_hampel_window=9,
                    robust_hampel_nsigma=3.5,
                    raw_stride=2,
                    episode_length=max(1, horizon_t),
                )
            for ci_span, ci_name in (
                (96, "Convergence_RewardTotal_span96_ci_only.png"),
                (200, "Convergence_RewardTotal_span200.png"),
                
                (396, "Convergence_RewardTotal_span396_ci_only.png"),
            ):
                plot_convergence_reward_total_ema_ci_only(
                    run_dir=run_dir,
                    train_episodes=conv_train_eps,
                    reward_total_mean=conv_train_rew,
                    out_dir=conv_all_dir,
                    ema_span=ci_span,
                    ci_window=ci_span,
                    ci_alpha=0.12,
                    save_name=ci_name,
                    note=f"EMA+CI only (span={ci_span})",
                    episode_length=max(1, horizon_t),
                )
            
            import numpy as _np
            _load_arr = _np.array(dense_load, dtype=_np.float64) if dense_load else _np.array([])
            _eps_arr = _np.array(conv_train_eps, dtype=_np.int64)
            _rew_arr = _np.array(conv_train_rew, dtype=_np.float64)
            if _load_arr.size == _eps_arr.size and _load_arr.size > 0:
                for _ld in (0.6, 0.8, 1.0):
                    _mask = _np.abs(_load_arr - _ld) < 0.01
                    if _mask.sum() < 10:
                        continue
                    _ld_dir = conv_parent_dir / f"load_{_ld:.1f}".replace(".", "")
                    _ld_dir.mkdir(parents=True, exist_ok=True)
                    _eps_ld = _np.arange(1, int(_mask.sum()) + 1)
                    _rew_ld = _rew_arr[_mask]
                    _ld_tag = str(_ld).replace(".", "")
                    for _raw_span in (48, 96, 200):
                        _raw_name = f"Convergence_RewardTotal_load{_ld_tag}_span{_raw_span}_raw_ema.png"
                        plot_convergence_reward_total(
                            run_dir=run_dir,
                            train_episodes=_eps_ld.tolist(),
                            reward_total_mean=_rew_ld.tolist(),
                            reward_total_ci=None,
                            out_dir=_ld_dir,
                            draw_ci=False,
                            ema_span=int(_raw_span),
                            save_name=_raw_name,
                            note=f"reward_total_ep (load={_ld:.1f}, 1 episode per point)",
                            robust_display=True,
                            robust_clip_quantile=0.01,
                            robust_hampel_window=9,
                            robust_hampel_nsigma=3.5,
                            raw_stride=1,
                            episode_length=max(1, horizon_t),
                        )
                    for _ci_span, _ci_name in (
                        (96, f"Convergence_RewardTotal_load{_ld:.1f}_span96.png".replace(".", "")),
                        (200, f"Convergence_RewardTotal_load{_ld:.1f}_span200.png".replace(".", "")),
                    ):
                        
                        _ci_name_fixed = f"Convergence_RewardTotal_load{str(_ld).replace('.','')}_span{_ci_span}.png"
                        plot_convergence_reward_total_ema_ci_only(
                            run_dir=run_dir,
                            train_episodes=_eps_ld.tolist(),
                            reward_total_mean=_rew_ld.tolist(),
                            out_dir=_ld_dir,
                            ema_span=_ci_span,
                            ci_window=_ci_span,
                            ci_alpha=0.12,
                            save_name=_ci_name_fixed,
                            note=f"load={_ld:.1f} only, EMA+CI (span={_ci_span})",
                            episode_length=max(1, horizon_t),
                        )
            
            try:
                _train_block = ((train_cfg or {}).get("train", {}) or {}) if isinstance(train_cfg, dict) else {}
                _sampling_on = bool(_train_block.get("train_load_sampling_enable", False))
                _cfg_loads = _train_block.get("train_load_sampling_loads", [])
                if not isinstance(_cfg_loads, (list, tuple)):
                    _cfg_loads = []
                _cfg_loads = [float(v) for v in _cfg_loads if _np.isfinite(float(v))]
                _cfg_loads = sorted(list(dict.fromkeys([round(float(v), 3) for v in _cfg_loads])))
                _tot_steps = int(float(_train_block.get("total_steps", 0) or 0))
                _roll_steps = int(float(_train_block.get("rollout_steps", 1) or 1))
                _roll_steps = int(max(_roll_steps, 1))
                _n_rollouts_est = int(max(1, round(float(max(_tot_steps, 1)) / float(_roll_steps))))
                _strict_long_run = bool(_n_rollouts_est >= 12)
                if _sampling_on and len(_cfg_loads) > 1 and _strict_long_run and _load_arr.size >= 200:
                    _missing = []
                    _counts = {}
                    for _ld in _cfg_loads:
                        _cnt = int((_np.abs(_load_arr - float(_ld)) < 0.01).sum())
                        _counts[f"{_ld:.1f}"] = int(_cnt)
                        if _cnt <= 0:
                            _missing.append(float(_ld))
                    _audit = {
                        "sampling_enable": True,
                        "configured_loads": [float(x) for x in _cfg_loads],
                        "dense_episode_count": int(_load_arr.size),
                        "estimated_rollouts": int(_n_rollouts_est),
                        "dense_load_counts": _counts,
                        "missing_loads": [float(x) for x in _missing],
                        "pass": bool(len(_missing) == 0),
                    }
                    _audit_path = run_dir / "train_load_coverage_audit.json"
                    with open(_audit_path, "w", encoding="utf-8") as _af:
                        json.dump(_audit, _af, ensure_ascii=False, indent=2)
                    if _missing:
                        raise RuntimeError(
                            "[HB][train][FATAL]  dense "
                            f"{_missing}counts={_counts}"
                        )
            except Exception as _e:
                if isinstance(_e, RuntimeError):
                    raise
                print(f"[HB][train][WARN] {type(_e).__name__}: {_e}", flush=True)
            plot_fixed_eval_curve(
                run_dir=run_dir,
                eval_train_episodes=eval_x_plot,
                reward_mean=trainer.eval_reward_history,
                reward_ci=getattr(trainer, "eval_reward_ci_history", []),
                paper_cost_mean=eval_cost_plot,
                paper_cost_ci=eval_cost_ci_plot,
                vio_any_frac_mean=eval_vio_plot,
                vio_any_frac_ci=eval_vio_ci_plot,
                out_dir=deteval_out_dir,
                quota_note=eval_note,
                ema_span=48,
                y_mode="paper_cost",
                save_name="DetEval_PaperCost.png",
                title_prefix="Det Eval",
            )
        else:
            raise RuntimeError(
                "[HB][train][FATAL_CONV_FALLBACK] missing reward_total episode curve; "
                "forbidden fallback to paper_return."
            )
        plot_ppo_diagnostics_from_csv(run_dir=run_dir)
    except Exception as e:
        print(f"[HB][train][WARN] {type(e).__name__}: {e}", flush=True)
        if "FATAL_CONV_FALLBACK" in str(e):
            raise

    
    meta_path = run_dir / "meta_train.json"
    try:
        best_ckpt_path = res.get("best_ckpt_path", None)
        if isinstance(best_ckpt_path, Path):
            best_ckpt_path = str(best_ckpt_path)

        meta = {
            "total_steps": int(res.get("total_steps", trainer.env_step)),
            "total_updates": int(res.get("total_updates", len(trainer.steps_history))),
            "total_episodes": int(res.get("total_episodes", getattr(trainer.train_collector, "collect_episode", 0))),
            "obs_dim": int(res.get("obs_dim", 0)),
            "act_dim": int(res.get("act_dim", 0)),
            "hidden": res.get("hidden", None),
            "final_reward": float(trainer.reward_history[-1]) if trainer.reward_history else 0.0,
            "best_ckpt_step": int(res.get("best_ckpt_step", None)) if res.get("best_ckpt_step", None) is not None else None,
            "best_paper_cost": float(res.get("best_paper_cost", None)) if res.get("best_paper_cost", None) is not None else None,
            "best_ckpt_path": best_ckpt_path,
            "meta_stats": res.get("meta_stats", {}),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[HB][train] Saved metadata: {meta_path}", flush=True)
    except Exception as e:
        print(f"[HB][train][WARN] meta_train.json {type(e).__name__}: {e}", flush=True)

    phasez_probe_result: dict | None = None
    try:
        total_steps_cfg_now = int(((train_cfg or {}).get("train", {}) or {}).get("total_steps", 0))
        phasez_probe_result = _run_phasez_fixed_eval_probe_if_needed(
            run_dir=run_dir,
            train_cfg=train_cfg,
            env_yaml=str(args.env_yaml),
            train_yaml=str(args.train_yaml),
            is_phasez_cfg=bool(rd in ("z_shifted_log", "z4_linear", "z5_saturating")),
            total_steps_cfg=int(total_steps_cfg_now),
        )
    except Exception as e:
        phasez_probe_result = {
            "enabled": bool(((train_cfg or {}).get("train", {}) or {}).get("fixed_eval_probe_enable", False)),
            "ran": False,
            "reason": f"exception:{type(e).__name__}",
            "error": str(e),
        }
        print(f"[HB][train][WARN] PhaseZ fixed-eval probe {type(e).__name__}: {e}", flush=True)

    
    try:
        metrics_csv_path = run_dir / "train_metrics.csv"
        conv_acc = compute_convergence_acceptance_from_csv(metrics_csv_path)
        phaseA_canary_acc = compute_phaseA_canary_acceptance_from_csv(metrics_csv_path)
        phaseA_acc = compute_phaseA_acceptance_from_csv(metrics_csv_path)
        phaseB_acc = compute_phaseB_acceptance_from_csv(metrics_csv_path)
        phaseZ_stage1_acc = compute_phaseZ_stage_acceptance_from_csv(metrics_csv_path, stage="stage1")
        phaseZ_stage2_acc = compute_phaseZ_stage_acceptance_from_csv(metrics_csv_path, stage="stage2")
        phaseZ_stage3_acc = compute_phaseZ_stage_acceptance_from_csv(metrics_csv_path, stage="stage3")
        phaseZ_stage4_acc = compute_phaseZ_stage_acceptance_from_csv(metrics_csv_path, stage="stage4")
        phaseZ_auto_acc = compute_phaseZ_stage_acceptance_from_csv(metrics_csv_path, stage="auto")
        phaseg_v11_acc = res.get("phaseg_v11_selfcheck", {})
        if not isinstance(phaseg_v11_acc, dict):
            phaseg_v11_acc = {}
        if not phaseg_v11_acc:
            
            try:
                phaseg_json = run_dir / "phaseg_v11_selfcheck.json"
                if phaseg_json.exists():
                    with open(phaseg_json, "r", encoding="utf-8") as f:
                        phaseg_v11_acc = json.load(f) or {}
            except Exception:
                phaseg_v11_acc = {}

        
        shaping_acc = compute_shaping_acceptance(
            shaping=zeros,
            comp=zeros,
            ris=zeros,
            improve=trainer.improve_history,
            alpha=0.08,
        )

        train_params = train_cfg.get("train", {}) or {}
        lag_cfg = train_params.get("lagrange_vio", {}) or {}
        total_steps_cfg = int(train_params.get("total_steps", 0))
        reward_override = str(train_params.get("reward_override", "env")).strip().lower()
        reward_design_cfg = str(getattr(env_cfg, "reward_design", "")).strip().lower()
        is_phaseg_cfg = bool(reward_design_cfg in ("reward_total_v1", "reward_total", "total_v1", "v1"))
        is_phasez_cfg = bool(reward_design_cfg in ("z_shifted_log", "z4_linear", "z5_saturating"))
        phasez_probe_enabled = bool(train_params.get("fixed_eval_probe_enable", False))
        phasez_probe_warning = False
        phasez_probe_reason = ""
        if is_phasez_cfg and phasez_probe_enabled:
            if isinstance(phasez_probe_result, dict):
                phasez_probe_reason = str(phasez_probe_result.get("reason", ""))
                probe_ran = bool(phasez_probe_result.get("ran", False))
                phasez_probe_warning = bool((not probe_ran) and (phasez_probe_reason != "steps_lt_24000"))
            else:
                phasez_probe_warning = True
                phasez_probe_reason = "missing_probe_result"
        phase_c_cfg = train_params.get("phase_c", {}) or {}
        scores_mode_cfg = str(phase_c_cfg.get("scores_mode", "learned")).strip().lower()
        is_traj_only_cfg = bool(scores_mode_cfg == "fixed")

        # Phase AP0
        
        
        
        use_phaseA_canary = bool(0 < total_steps_cfg <= 15000)
        use_phaseA_short = bool(15000 < total_steps_cfg <= 60000)

        
        use_phaseB = bool(phaseB_acc.get("reason", "") not in ("missing_train_metrics_csv", "too_few_points", "too_few_steps_for_phaseB"))

        phasez_target_stage = "stage4"
        phasez_target_acc: dict = phaseZ_stage4_acc
        if is_phasez_cfg:
            if 0 < total_steps_cfg <= 5000:
                phasez_target_stage = "stage1"
                phasez_target_acc = phaseZ_stage1_acc
            elif 5000 < total_steps_cfg <= 60000:
                phasez_target_stage = "stage2"
                phasez_target_acc = phaseZ_stage2_acc
            elif 60000 < total_steps_cfg <= 150000:
                phasez_target_stage = "stage3"
                phasez_target_acc = phaseZ_stage3_acc
            else:
                phasez_target_stage = "stage4"
                phasez_target_acc = phaseZ_stage4_acc
            pass_final = bool(phasez_target_acc.get("pass", False))
            phase_hint = f"PhaseZ_{phasez_target_stage}"
        elif is_phaseg_cfg:
            
            phaseg_v11_pass = bool(phaseg_v11_acc.get("pass", False))
            conv_reason = str(conv_acc.get("reason", "")).strip().lower()
            phaseg_curve_ready = bool(conv_reason not in ("missing_train_metrics_csv", "too_few_points"))
            pass_final = bool(phaseg_v11_pass and phaseg_curve_ready)
            phase_hint = "PhaseG-v1.1"
            if phaseg_v11_pass and (not phaseg_curve_ready):
                print(
                    "[HB][train][accept][WARN] PhaseG selfchecktrain_metrics"
                    "passfalse",
                    flush=True,
                )
        elif use_phaseA_canary:
            pass_final = bool(phaseA_canary_acc.get("pass", False))
            phase_hint = "PhaseA_canary"
        elif is_traj_only_cfg:
            pass_final = bool(phaseB_acc.get("pass", False))
            phase_hint = "PhaseB"
        elif use_phaseA_short:
            pass_final = bool(phaseA_acc.get("pass", False))
            phase_hint = "PhaseA"
        else:
            pass_final = bool(phaseB_acc.get("pass", False)) if use_phaseB else bool(conv_acc.get("pass", False))
            phase_hint = "PhaseB" if use_phaseB else "P10-B"

        acc = {
            "phase_hint": phase_hint,
            "cfg_reward_override": str(reward_override),
            "cfg_lagrange_enabled": bool(lag_cfg.get("enabled", False)),
            "phaseA_canary": phaseA_canary_acc,
            "phaseA": phaseA_acc,
            "phaseB": phaseB_acc,
            "phaseZ_stage1": phaseZ_stage1_acc,
            "phaseZ_stage2": phaseZ_stage2_acc,
            "phaseZ_stage3": phaseZ_stage3_acc,
            "phaseZ_stage4": phaseZ_stage4_acc,
            "phaseZ_auto": phaseZ_auto_acc,
            "phaseZ_target_stage": phasez_target_stage if is_phasez_cfg else None,
            "phaseZ_fixed_eval_probe": phasez_probe_result if is_phasez_cfg else None,
            "phaseZ_probe_warning": bool(phasez_probe_warning) if is_phasez_cfg else None,
            "phaseZ_probe_reason": phasez_probe_reason if is_phasez_cfg else None,
            "phaseG_v11": phaseg_v11_acc,
            "phaseG_curve_ready": bool(conv_acc.get("reason", "") not in ("missing_train_metrics_csv", "too_few_points")) if is_phaseg_cfg else None,
            "convergence": conv_acc,
            "shaping_legacy": shaping_acc,
            "pass": bool(pass_final),
        }
        acc_path = run_dir / "train_acceptance.json"
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump(acc, f, indent=2, ensure_ascii=False)
        print(f"[HB][train] Saved acceptance summary: {acc_path}", flush=True)

        
        try:
            _metrics_csv = run_dir / "train_metrics.csv"
            h11_bal = compute_h11_reward_balance(_metrics_csv)
            _bal_path = run_dir / "H11_reward_chain_balance.json"
            with open(_bal_path, "w", encoding="utf-8") as _bf:
                json.dump(h11_bal, _bf, indent=2, ensure_ascii=False)
            print(f"[HB][train] H11 balance audit: pass={h11_bal.get('pass')}, "
                  f"shares={h11_bal.get('shares', {})}", flush=True)
        except Exception as _e:
            print(f"[HB][train][WARN] H11 balance audit failed: {_e}", flush=True)

        
        try:
            drop_critical_keys = ["reward_proximity_weight", "beta_comp_max", "beta_ris_max", "beta_end_frac"]
            if str(reward_design_cfg).strip().lower() == "z_shifted_log":
                
                drop_critical_keys += [
                    "obs_enable_prev_action",
                    "z_alpha",
                    "z_beta",
                    "z_gamma",
                    "z_cost_ref",
                    "reward_proximity_weight_final",
                    "reward_proximity_decay_start_step",
                    "reward_proximity_decay_end_step",
                    "reward_proximity_mode",
                    "coverage_floor_weight",
                    "coverage_floor_threshold",
                ]
            if str(reward_design_cfg).strip().lower() == "z4_linear":
                
                drop_critical_keys += [
                    "obs_enable_prev_action",
                    "z_cost_ref",
                    "z4_reward_offset",
                    "z4_reward_alpha",
                    "z4_bonus_gamma",
                    "z4_bonus_anchor",
                    "z4_bonus_power",
                    "z4_action_mode",
                    "comp_rule_threshold",
                    "coverage_floor_weight",
                    "coverage_floor_threshold",
                    "reward_clip_range",
                ]
            if str(reward_design_cfg).strip().lower() == "z5_saturating":
                
                drop_critical_keys += [
                    "obs_enable_prev_action",
                    "z_cost_ref",
                    "z5_r_max",
                    "z5_kappa",
                    "z5_anchor_norm",
                    "reward_proximity_weight_final",
                    "reward_proximity_decay_start_step",
                    "reward_proximity_decay_end_step",
                    "reward_proximity_mode",
                    "coverage_floor_weight",
                    "coverage_floor_threshold",
                    "reward_clip_range",
                ]
            h11_drop = compute_h11_cfg_dropped_keys_audit(
                dropped_keys=dropped,
                critical_keys=drop_critical_keys,
            )
            _drop_path = run_dir / "H11_cfg_dropped_keys_audit.json"
            with open(_drop_path, "w", encoding="utf-8") as _df:
                json.dump(h11_drop, _df, indent=2, ensure_ascii=False)
            print(
                f"[HB][train] H11 dropped-keys audit: pass={h11_drop.get('pass')} dropped={h11_drop.get('dropped_count')}",
                flush=True,
            )
        except Exception as _e:
            h11_drop = {"pass": False, "reason": f"exception: {_e}"}
            print(f"[HB][train][WARN] H11 dropped-keys audit failed: {_e}", flush=True)

        
        try:
            h11_knob = compute_h11_knob_effect_audit(
                metrics_csv_path=metrics_csv_path,
                env_cfg_dict_clean=env_cfg_dict_clean,
                dropped_audit=h11_drop,
            )
            _knob_path = run_dir / "H11_knob_effect_audit.json"
            with open(_knob_path, "w", encoding="utf-8") as _kf:
                json.dump(h11_knob, _kf, indent=2, ensure_ascii=False)
            print(f"[HB][train] H11 knob audit: pass={h11_knob.get('pass')}", flush=True)
        except Exception as _e:
            print(f"[HB][train][WARN] H11 knob audit failed: {_e}", flush=True)

        
        try:
            h11_dense = compute_h11_reward_shape_gate_dense(
                metrics_csv_path=metrics_csv_path,
                run_dir=run_dir,
            )
            _dense_path = run_dir / "H11_reward_shape_gate_dense.json"
            with open(_dense_path, "w", encoding="utf-8") as _ff:
                json.dump(h11_dense, _ff, indent=2, ensure_ascii=False)
            print(
                f"[HB][train] H11 dense-shape audit: pass={h11_dense.get('pass')} reason={h11_dense.get('reason')}",
                flush=True,
            )
        except Exception as _e:
            print(f"[HB][train][WARN] H11 dense-shape audit failed: {_e}", flush=True)

        
        try:
            h11a_acc = compute_h11_a_acceptance(
                metrics_csv_path=metrics_csv_path,
                run_dir=run_dir,
                env_cfg_dict_clean=env_cfg_dict_clean,
                dropped_keys=dropped,
                total_steps_cfg=total_steps_cfg,
                smoke_max_steps=6000,
            )
            
            try:
                m1_plot = plot_metric1_robust_trend(run_dir=run_dir)
            except Exception as _m1e:
                m1_plot = {"ok": False, "reason": f"plot_exception: {_m1e}"}
                print(f"[HB][train][WARN] Metric1 robust trend plot failed: {_m1e}", flush=True)
            if isinstance(h11a_acc, dict):
                h11a_acc["metric1_robust_trend"] = m1_plot

            _h11a_path = run_dir / "H11_A_acceptance.json"
            with open(_h11a_path, "w", encoding="utf-8") as _af:
                json.dump(h11a_acc, _af, indent=2, ensure_ascii=False)
            print(
                f"[HB][train] H11-A acceptance: mode={h11a_acc.get('stage_mode')} pass={h11a_acc.get('pass')} reason={h11a_acc.get('reason')}",
                flush=True,
            )
        except Exception as _e:
            print(f"[HB][train][WARN] H11-A acceptance audit failed: {_e}", flush=True)

        
        try:
            env_reward_ema_beta = float(env_cfg_dict_clean.get("reward_ema_beta", 0.0))
            logp_mode = str(train_params.get("logp_mode", "auto"))
            phaseB_fixed = phaseB_acc.get("fixed_eval", {}) if isinstance(phaseB_acc, dict) else {}
            phaseB_fixed = phaseB_fixed if isinstance(phaseB_fixed, dict) else {}
            notes = [
                "# ",
                "",
                f"- run_dir: {str(run_dir)}",
                f"- time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- env.reward_ema_beta: {env_reward_ema_beta} reward  EMA ",
                f"- train.logp_mode: {logp_mode}PhaseA per-dim  PPO",
                f"- phase_hint: {phase_hint}",
                "",
                "## PhaseA CanaryA4.110k",
                f"- pass: {bool(phaseA_canary_acc.get('pass', False))}",
                f"- clipfrac_mean_tail: {float(phaseA_canary_acc.get('clipfrac_mean_tail', float('nan'))):.4f} (<0.35)",
                f"- kl_per_dim_tail: {float(phaseA_canary_acc.get('kl_per_dim_tail', float('nan'))):.6f} (not dead/explode)",
                f"- entropy_per_dim_tail: {float(phaseA_canary_acc.get('entropy_per_dim_tail', float('nan'))):.4f} (finite)",
                "",
                "## PhaseA A350kP0",
                f"- pass_basic: {bool(phaseA_acc.get('pass_basic', False))}PPOPhaseBv13 A5",
                f"- pass_strict: {bool(phaseA_acc.get('pass_strict', False))}rewardv13 A4.2",
                f"- pass: {bool(phaseA_acc.get('pass', False))}",
                f"- clipfrac_mean_late: {float(phaseA_acc.get('clipfrac_mean_late', float('nan'))):.4f} (in[0.05,0.25])",
                f"- kl_per_dim_late: {float(phaseA_acc.get('kl_per_dim_late', float('nan'))):.6f} (in[5e-4,5e-3])",
                f"- reward: pass={bool(phaseA_acc.get('reward_pass', False))}",
                f"-   R_early: {float(phaseA_acc.get('R_early', float('nan'))):.6f}",
                f"-   R_late : {float(phaseA_acc.get('R_late', float('nan'))):.6f}",
                f"-   rel_improve: {float(phaseA_acc.get('rel_improve', float('nan'))):.4f} (>=0.20)",
                f"-   abs_improve: {float(phaseA_acc.get('abs_improve', float('nan'))):.6f} (>=0.10*std_early)",
                f"-   std_early : {float(phaseA_acc.get('std_early', float('nan'))):.6f}",
                f"- anti-spoof: logp_is_mean_tail={float(phaseA_acc.get('logp_is_mean_tail', float('nan'))):.3f} "
                f"logp_scale_tail={float(phaseA_acc.get('logp_scale_tail', float('nan'))):.6f} "
                f"act_dim_effective_tail={float(phaseA_acc.get('act_dim_effective_tail', float('nan'))):.1f} "
                f"anti_spoof_pass={bool(phaseA_acc.get('anti_spoof_pass', False))} "
                f"reason={str(phaseA_acc.get('anti_spoof_reason', ''))}",
                "",
                "## PhaseB B4150ktraj_only",
                f"- pass: {bool(phaseB_acc.get('pass', False))}",
                f"- true10_pass: {bool(phaseB_acc.get('true10_pass', False))} "
                f"act_dim_eff_tail={float(phaseB_acc.get('act_dim_effective_tail', float('nan'))):.3f} "
                f"scores_fixed_tail={float(phaseB_acc.get('scores_mode_fixed_tail', float('nan'))):.3f}",
                f"- clipfrac_mean_late: {float(phaseB_acc.get('clipfrac_mean_late', float('nan'))):.4f} (in[0.05,0.25])",
                f"- kl_per_dim_late: {float(phaseB_acc.get('kl_per_dim_late', float('nan'))):.6f} (~[5e-4,5e-3])",
                f"- vio_pass: {bool(phaseB_acc.get('vio_pass', False))} "
                f"V_early={float(phaseB_acc.get('V_early', float('nan'))):.4f} "
                f"V_late={float(phaseB_acc.get('V_late', float('nan'))):.4f}",
                f"- fixed_eval: pass={bool(phaseB_acc.get('fixed_eval_pass', False))} "
                f"paper_return_pass={bool(phaseB_fixed.get('paper_return_pass', False))} "
                f"plateau_pass={bool(phaseB_fixed.get('paper_return_plateau_pass', False))} "
                f"CI_shrink(return)={bool(phaseB_fixed.get('paper_return_ci_shrink_pass', False))} "
                f"CI_shrink(cost)={bool(phaseB_fixed.get('paper_cost_ci_shrink_pass', False))}",
                f"- fixed_eval_pass: {bool(phaseB_acc.get('fixed_eval_pass', False))} "
                f"reason={str(phaseB_acc.get('fixed_eval_reason', ''))}",
                "",                "## PhaseZ 2k/30k/100k/500k",
                f"- target_stage: {phasez_target_stage if is_phasez_cfg else 'N/A'} total_steps ",
                f"- stage1_pass: {bool(phaseZ_stage1_acc.get('pass', False))} "
                f"reward_tail={float(phaseZ_stage1_acc.get('reward_tail_mean', float('nan'))):.4f} "
                f"clip_tail={float(phaseZ_stage1_acc.get('clipfrac_tail_mean', float('nan'))):.6f}",
                f"- stage2_pass: {bool(phaseZ_stage2_acc.get('pass', False))} "
                f"clip_tail={float(phaseZ_stage2_acc.get('clipfrac_tail_mean', float('nan'))):.6f} "
                f"value_tail={float(phaseZ_stage2_acc.get('value_loss_tail_mean', float('nan'))):.4f} "
                f"entropy_tail={float(phaseZ_stage2_acc.get('entropy_per_dim_tail_mean', float('nan'))):.4f}",
                f"- stage3_pass: {bool(phaseZ_stage3_acc.get('pass', False))} "
                f"reward_early={float(phaseZ_stage3_acc.get('reward_early_s3_mean', float('nan'))):.4f} "
                f"reward_late={float(phaseZ_stage3_acc.get('reward_late_s3_mean', float('nan'))):.4f} "
                f"cost_early={float(phaseZ_stage3_acc.get('paper_cost_early_s3_mean', float('nan'))):.4f} "
                f"cost_late={float(phaseZ_stage3_acc.get('paper_cost_late_s3_mean', float('nan'))):.4f}",
                f"- stage4_pass: {bool(phaseZ_stage4_acc.get('pass', False))} "
                f"reason={str(phaseZ_stage4_acc.get('reason', ''))}",
                "",
                "## PhaseZ step2/3/4 ",
                f"- step2_metric1(pass_non_worse): {bool((((phaseZ_stage2_acc.get('five_metrics', {}) or {}).get('metric1_paper_cost_trend', {}) or {}).get('pass_non_worse', False)))}",
                f"- step2_metric2(pass_non_collapse): {bool((((phaseZ_stage2_acc.get('five_metrics', {}) or {}).get('metric2_reward_shape', {}) or {}).get('pass_non_collapse', False)))}",
                f"- step2_metric5(pass_basic): {bool((((phaseZ_stage2_acc.get('five_metrics', {}) or {}).get('metric5_training_health', {}) or {}).get('pass_basic', False)))}",
                f"- step3_metric1(pass_non_worse): {bool((((phaseZ_stage3_acc.get('five_metrics', {}) or {}).get('metric1_paper_cost_trend', {}) or {}).get('pass_non_worse', False)))}",
                f"- step3_metric2(pass_upward): {bool((((phaseZ_stage3_acc.get('five_metrics', {}) or {}).get('metric2_reward_shape', {}) or {}).get('pass_upward', False)))}",
                f"- step3_metric5(pass_basic): {bool((((phaseZ_stage3_acc.get('five_metrics', {}) or {}).get('metric5_training_health', {}) or {}).get('pass_basic', False)))}",
                f"- step4_metric3(10loads_ready): {bool((((phaseZ_stage4_acc.get('five_metrics', {}) or {}).get('metric3_ablation_10loads', {}) or {}).get('ready', False)))}",
                f"- step4_metric4(dynamic_ready): {bool((((phaseZ_stage4_acc.get('five_metrics', {}) or {}).get('metric4_dynamic_comp_viz', {}) or {}).get('ready', False)))}",
                f"- phasez_probe_enabled: {bool(phasez_probe_enabled)}",
                f"- phasez_probe_warning: {bool(phasez_probe_warning)} reason={phasez_probe_reason}",
                f"- phasez_probe_payload: {json.dumps(phasez_probe_result, ensure_ascii=False) if isinstance(phasez_probe_result, dict) else 'null'}",
                "",
                "## Next",
                "-  clipfrac  logp_mode train_metrics.csv  logp_is_mean/logp_scale ",
                "-  KL/d lr/train_epochs/target_kl/max_grad_norm",
                "-  value_loss  train_metrics.csv  vf_clip_hit_frac1criticvf_loss_clip_max",
            ]
            (run_dir / "phase_notes.md").write_text("\n".join(notes), encoding="utf-8")
        except Exception as e:
            print(f"[HB][train][WARN] phase_notes.md {type(e).__name__}: {e}", flush=True)
    except Exception as e:
        print(f"[HB][train][WARN] acceptance summary {type(e).__name__}: {e}", flush=True)

    try:
        m1_guard = _write_metric1_contract_guard(run_dir=run_dir)
        print(
            "[HB][train][Metric1] "
            f"pass={int(bool(m1_guard.get('pass', False)))} "
            f"source={m1_guard.get('metric1_source', 'NA')} "
            f"axis={m1_guard.get('episode_axis_source', 'NA')} "
            f"drop={m1_guard.get('relative_drop_pct', float('nan'))} "
            f"dir={m1_guard.get('relative_drop_direction', 'unknown')} "
            f"perf_nonworse={int(bool(m1_guard.get('performance_pass_nonworse', False)))}",
            flush=True,
        )
    except Exception as e:
        print(f"[HB][train][WARN] 1{type(e).__name__}: {e}", flush=True)

    try:
        if is_t6_cfg:
            _write_t6_artifacts(run_dir=run_dir, train_cfg=train_cfg)
            print(
                f"[HB][train][T6] {run_dir / 'train_acceptance_t6.json'} "
                f"{run_dir / 'diag' / 'reward_contract_t6.csv'} "
                f"{run_dir / 'diag' / 'lagrange_trace_t6.csv'}",
                flush=True,
            )
    except Exception as e:
        print(f"[HB][train][WARN] PhaseT6{type(e).__name__}: {e}", flush=True)

    print(f"[HB][train] Training complete! run_dir={run_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CoMP-RIS UAV-MEC PPO policy.")
    parser.add_argument("--env-yaml", type=str, default=None, help="Environment YAML path. Defaults to the path declared by the train YAML.")
    parser.add_argument("--train-yaml", type=str, default="configs/PhaseZ4/train_step7_1200k.yaml", help="Training YAML path.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run directory name.")
    parser.add_argument("--seed", type=int, default=None, help="Override both environment and trainer seeds.")
    parser.add_argument("--save-every", type=int, default=None, help=" train.save_every")
    parser.add_argument("--plot-every", type=int, default=None, help=" train.plot_every=save_every//2")
    parser.add_argument("--total-steps", type=int, default=None, help="Override train.total_steps.")
    parser.add_argument("--eval-every", type=int, default=None, help=" train.eval_every")
    parser.add_argument("--eval-episodes", type=int, default=None, help=" eval.eval_episodes")
    parser.add_argument("--resume-path", type=str, default=None, help="Resume from a full training checkpoint; use none/null to disable.")
    parser.add_argument("--policy-init-path", type=str, default=None, help="Initialize policy weights without restoring optimizer state.")
    parser.add_argument("--bc-warmstart-enable", type=int, choices=[0, 1], default=None, help=" train.bc_warmstart.enable")
    parser.add_argument("--bc-warmstart-weight0", type=float, default=None, help=" train.bc_warmstart.weight0")
    parser.add_argument("--bc-warmstart-decay-steps", type=int, default=None, help=" train.bc_warmstart.decay_steps")
    parser.add_argument("--bc-warmstart-cost-weight0", type=float, default=None, help=" train.bc_warmstart.cost_weight0")
    parser.add_argument("--bc-warmstart-cost-decay-steps", type=int, default=None, help=" train.bc_warmstart.cost_decay_steps")
    parser.add_argument("--meta-enable", type=int, default=None, help="Enable the optional meta-controller: 1=yes, 0=no.")
    parser.add_argument("--meta-source", type=str, default=None, help="Optional meta-controller source: heuristic, random, or llm.")
    parser.add_argument("--meta-yaml", type=str, default=None, help="Optional meta-controller YAML path.")
    parser.add_argument(
        "--algo-tag",
        type=str,
        default="auto",
        help="Run-name prefix mode: auto, ppo_base, or amcp_ppo.",
    )
    parser.add_argument(
        "--no_selfcheck",
        action="store_true",
        help="Disable trainer self-checks. Intended for debugging only.",
    )
    parser.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help="Allow --env-yaml to differ from train.env.config_path.",
    )
    args = parser.parse_args()
    _normalize_cfg_inputs(args)
    print(
        "[HB][train] cfg_resolved: "
        f"env={str(getattr(args, '_cfg_env_display', args.env_yaml))} "
        f"train={str(getattr(args, '_cfg_train_display', args.train_yaml))} "
        f"source={str(getattr(args, '_cfg_env_source', 'unknown'))}",
        flush=True,
    )

    
    
    
    
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = "sNA"
        steps_k = "NAk"
        reward_override = "rwNA"
        paper_map = ""
        phase_tag = "train"
        algo_tag = str(getattr(args, "algo_tag", "auto") or "auto").strip().lower()
        run_name_prefix_cfg = ""
        run_phase_tag_cfg = ""
        
        reward_design = ""
        scores_mode = ""
        try:
            with open(args.train_yaml, "r", encoding="utf-8") as f:
                _cfg = yaml.safe_load(f) or {}
            _apply_meta_cli_overrides(_cfg, args)
            run_name_prefix_cfg = str(_cfg.get("run_name_prefix", "")).strip()
            run_phase_tag_cfg = str(_cfg.get("run_phase_tag", "")).strip()
            if algo_tag in ("", "auto"):
                algo_tag = _infer_algo_tag_from_cfg(_cfg)
            seed_v = _cfg.get("seed", None)
            if seed_v is not None:
                seed = f"s{int(seed_v)}"
            
            if getattr(args, "seed", None) is not None:
                seed = f"s{int(args.seed)}"
            _train = _cfg.get("train", {}) or {}
            total_steps_v = _train.get("total_steps", None)
            if getattr(args, "total_steps", None) is not None:
                total_steps_v = int(args.total_steps)
            if total_steps_v is not None:
                steps_i = int(total_steps_v)
                if steps_i >= 1000:
                    steps_k = f"{steps_i // 1000}k"
                else:
                    
                    steps_k = f"{steps_i}s"
                
                
                phase_tag = str(run_phase_tag_cfg or "train").strip()

            
            _pc = _train.get("phase_c", {}) or {}
            scores_mode = str(_pc.get("scores_mode", "learned")).strip().lower()
            reward_override = f"{str(_train.get('reward_override', 'env')).strip()}"
            if str(reward_override).strip().lower() == "paper":
                _pr = _train.get("paper_reward", {}) or {}
                paper_map = f"_{str(_pr.get('map', 'NA')).strip()}"
        except Exception:
            
            pass

        
        try:
            with open(args.env_yaml, "r", encoding="utf-8") as f:
                _env_cfg = yaml.safe_load(f) or {}
            _obj = _env_cfg.get("objective", {}) or {}
            reward_design = str(_obj.get("reward_design", "")).strip().lower()
        except Exception:
            reward_design = ""
        
        phase_tag = str(run_phase_tag_cfg or phase_tag or "train").strip()

        if run_name_prefix_cfg:
            prefix = _sanitize_run_token(run_name_prefix_cfg, default="ppo_base")
        else:
            if algo_tag not in ("ppo_base", "amcp_ppo"):
                algo_tag = "ppo_base"
            prefix = str(algo_tag)
        phase_tag = _sanitize_run_token(phase_tag, default="train")
        args.run_name = f"{prefix}_{phase_tag}_{reward_override}{paper_map}_{steps_k}_{seed}_{timestamp}"

    print(f"[HB][train] Run name: {args.run_name}", flush=True)
    train(args)


if __name__ == "__main__":
    main()
