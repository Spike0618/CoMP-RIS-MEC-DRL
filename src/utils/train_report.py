from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.plot_smoothing import compute_confidence_band, filter_valid_xy, ema_smooth


REWARD_TOTAL_RAW_COLOR = "#9BC1FF"
REWARD_TOTAL_EMA_COLOR = "#2D5AA0"


def ema_np(x: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    """EMA  alpha """
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=np.float64)
    out[0] = float(x[0])
    for i in range(1, x.size):
        out[i] = alpha * float(x[i]) + (1.0 - alpha) * float(out[i - 1])
    return out

def compute_shaping_acceptance(
    shaping: List[float],
    comp: List[float],
    ris: List[float],
    improve: List[float],
    alpha: float = 0.08,
) -> Dict[str, float]:
    """
    

    
    - TotalShaping / CoMP / RIS EPIC total_bonus 
    -  env_shaping TotalShaping 
    """
    out: Dict[str, float] = {
        "n_points": float(len(shaping)),
        "total_shaping_ema_end": float("nan"),
        "total_shaping_ema_early_mean": float("nan"),
        "total_shaping_ema_late_mean": float("nan"),
        "total_shaping_improve_ratio": float("nan"),
        "comp_ris_ratio_ema_end": float("nan"),
        "improve_ema_late_mean": float("nan"),
    }
    if len(shaping) < 5:
        return out

    s = ema_np(np.asarray(shaping, dtype=np.float64), alpha=alpha)
    c = ema_np(np.asarray(comp, dtype=np.float64), alpha=alpha)
    r = ema_np(np.asarray(ris, dtype=np.float64), alpha=alpha)
    imp = ema_np(np.asarray(improve, dtype=np.float64), alpha=alpha)

    n = int(s.size)
    w = max(5, int(round(0.2 * n)))
    w = min(w, n)

    early_mean = float(np.mean(s[:w]))
    late_mean = float(np.mean(s[-w:]))
    improve_ratio = float((late_mean - early_mean) / (abs(early_mean) + 1e-12))

    comp_ris_ratio_end = float((float(c[-1]) + float(r[-1])) / max(float(s[-1]), 1e-12))
    improve_late_mean = float(np.mean(imp[-w:]))

    out.update(
        {
            "total_shaping_ema_end": float(s[-1]),
            "total_shaping_ema_early_mean": early_mean,
            "total_shaping_ema_late_mean": late_mean,
            "total_shaping_improve_ratio": improve_ratio,
            "comp_ris_ratio_ema_end": comp_ris_ratio_end,
            "improve_ema_late_mean": improve_late_mean,
        }
    )
    return out


def ensure_train_metrics_episode_fields(metrics_csv_path: Path) -> bool:
    """
     train_metrics.csv  train_episode/eval_train_episode
    
    
    - train_episodeepisode train_n_ep steps
    - eval_train_episodedet-evalepisode nan
    -  .bak  False
    """

    metrics_csv_path = Path(metrics_csv_path)
    if not metrics_csv_path.exists():
        return False

    try:
        with open(metrics_csv_path, "r", encoding="utf-8", newline="") as f:
            rr = csv.DictReader(f)
            fieldnames = list(rr.fieldnames) if rr.fieldnames else []
    except Exception:
        return False

    need_train = "train_episode" not in fieldnames
    need_eval = "eval_train_episode" not in fieldnames
    if not (need_train or need_eval):
        return False

    try:
        rows: List[Dict[str, str]] = []
        with open(metrics_csv_path, "r", encoding="utf-8", newline="") as f:
            rr = csv.DictReader(f)
            for row in rr:
                if not isinstance(row, dict):
                    continue
                
                rows.append({str(k): str(v) for k, v in row.items() if k is not None})

        out_fields = list(fieldnames)
        if need_train:
            if "global_step" in out_fields:
                out_fields.insert(out_fields.index("global_step") + 1, "train_episode")
            else:
                out_fields.append("train_episode")
        if need_eval:
            if "eval_det_is_valid" in out_fields:
                out_fields.insert(out_fields.index("eval_det_is_valid") + 1, "eval_train_episode")
            else:
                out_fields.append("eval_train_episode")

        train_episode_cum = 0
        out_rows: List[Dict[str, str]] = []
        for row in rows:
            def _get_float(key: str, default: float = float("nan")) -> float:
                try:
                    return float(row.get(key, default))
                except Exception:
                    return float(default)

            te = None
            try:
                te_raw = row.get("train_episode", "")
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

            row["train_episode"] = str(int(train_episode_cum))

            
            ete = row.get("eval_train_episode", "")
            if need_eval or str(ete).strip() == "":
                is_valid = _get_float("eval_det_is_valid", float("nan"))
                n_eval = _get_float("eval_det_n", 0.0)
                cost_mu = _get_float("eval_det_paper_cost_mean", float("nan"))
                is_eval = (np.isfinite(is_valid) and is_valid >= 0.5) or (np.isfinite(n_eval) and n_eval > 0.0) or np.isfinite(cost_mu)
                row["eval_train_episode"] = str(int(train_episode_cum)) if bool(is_eval) else "nan"

            out_rows.append(row)

        
        bak_path = metrics_csv_path.parent / f"{metrics_csv_path.name}.bak"
        try:
            if not bak_path.exists():
                import shutil
                shutil.copyfile(str(metrics_csv_path), str(bak_path))
        except Exception:
            pass

        tmp_path = metrics_csv_path.parent / f"{metrics_csv_path.name}.tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            ww = csv.DictWriter(f, fieldnames=out_fields)
            ww.writeheader()
            for row in out_rows:
                ww.writerow({k: row.get(k, "nan") for k in out_fields})

        tmp_path.replace(metrics_csv_path)
        return True
    except Exception:
        return False


def read_train_metrics_csv(metrics_csv_path: Path) -> Dict[str, np.ndarray]:
    """ official_trainer  train_metrics.csv"""
    if not metrics_csv_path.exists():
        return {}

    
    try:
        ensure_train_metrics_episode_fields(metrics_csv_path)
    except Exception:
        pass

    cols: Dict[str, List[float]] = {}
    with open(metrics_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, [])
                try:
                    cols[k].append(float(v))
                except Exception:
                    cols[k].append(float("nan"))

    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def linreg_slope(x: np.ndarray, y: np.ndarray) -> float:
    """30%"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x0 = x - float(np.mean(x))
    denom = float(np.sum(x0 * x0))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x0 * (y - float(np.mean(y)))) / denom)


def compute_convergence_acceptance_from_csv(metrics_csv_path: Path) -> Dict[str, Any]:
    """
    Phase B/C  train_metrics.csv 

    
    - EV30% >= 0.30
    - reward30%>=0
    - value loss30%<=0
    - KL/clipfrac30%/

    
    """
    m = read_train_metrics_csv(metrics_csv_path)
    if not m or "global_step" not in m:
        return {"pass": False, "reason": "missing_train_metrics_csv", "metrics_csv": str(metrics_csv_path)}

    steps = m.get("global_step", np.asarray([], dtype=np.float64))
    n = int(steps.size)
    if n < 10:
        return {"pass": False, "reason": "too_few_points", "n_points": n, "metrics_csv": str(metrics_csv_path)}

    w_late = max(5, int(round(0.30 * n)))
    w_early = max(5, int(round(0.20 * n)))
    w_late = min(w_late, n)
    w_early = min(w_early, n)

    reward = m.get("train_reward_mean", np.full((n,), np.nan, dtype=np.float64))
    vloss = m.get("value_loss", np.full((n,), np.nan, dtype=np.float64))
    ev = m.get("explained_variance", np.full((n,), np.nan, dtype=np.float64))

    
    
    
    
    act_dim_arr = m.get("act_dim", np.full((n,), np.nan, dtype=np.float64))
    kl_sum = m.get("kl", np.full((n,), np.nan, dtype=np.float64))
    kl_per_dim = m.get("kl_per_dim", np.full((n,), np.nan, dtype=np.float64))
    clipfrac_pd = m.get("clipfrac_per_dim", np.full((n,), np.nan, dtype=np.float64))
    clipfrac = m.get("clipfrac", np.full((n,), np.nan, dtype=np.float64))

    
    if (not np.isfinite(kl_per_dim).any()) and np.isfinite(kl_sum).any() and np.isfinite(act_dim_arr).any():
        denom = np.maximum(act_dim_arr, 1.0)
        kl_per_dim = kl_sum / denom

    
    if not np.isfinite(clipfrac_pd).any():
        clipfrac_pd = clipfrac

    reward_ema = ema_np(np.asarray(reward, dtype=np.float64), alpha=0.08)
    vloss_ema = ema_np(np.asarray(vloss, dtype=np.float64), alpha=0.08)
    ev_ema = ema_np(np.asarray(ev, dtype=np.float64), alpha=0.08)

    early_reward_mean = float(np.nanmean(reward_ema[:w_early]))
    late_reward_mean = float(np.nanmean(reward_ema[-w_late:]))
    reward_slope_late = linreg_slope(steps[-w_late:], reward_ema[-w_late:])

    early_vloss_mean = float(np.nanmean(vloss_ema[:w_early]))
    late_vloss_mean = float(np.nanmean(vloss_ema[-w_late:]))
    vloss_slope_late = linreg_slope(steps[-w_late:], vloss_ema[-w_late:])

    late_ev_mean = float(np.nanmean(ev[-w_late:]))
    late_ev_p10 = float(np.nanquantile(ev[-w_late:], 0.10)) if np.isfinite(ev[-w_late:]).any() else float("nan")
    late_ev_ema_mean = float(np.nanmean(ev_ema[-w_late:]))

    late_kl_sum_mean = float(np.nanmean(kl_sum[-w_late:]))
    late_kl_sum_p95 = float(np.nanquantile(kl_sum[-w_late:], 0.95)) if np.isfinite(kl_sum[-w_late:]).any() else float("nan")
    late_kl_pd_mean = float(np.nanmean(kl_per_dim[-w_late:]))
    late_kl_pd_p95 = float(np.nanquantile(kl_per_dim[-w_late:], 0.95)) if np.isfinite(kl_per_dim[-w_late:]).any() else float("nan")

    late_clip_mean = float(np.nanmean(clipfrac[-w_late:]))
    late_clip_p95 = float(np.nanquantile(clipfrac[-w_late:], 0.95)) if np.isfinite(clipfrac[-w_late:]).any() else float("nan")
    late_clip_pd_mean = float(np.nanmean(clipfrac_pd[-w_late:]))
    late_clip_pd_p95 = float(np.nanquantile(clipfrac_pd[-w_late:], 0.95)) if np.isfinite(clipfrac_pd[-w_late:]).any() else float("nan")

    
    ev_pass = bool(np.isfinite(late_ev_mean) and (late_ev_mean >= 0.30))

    
    
    
    reward_slope_tol = 1e-3
    reward_pass = bool(np.isfinite(reward_slope_late) and (reward_slope_late >= -float(reward_slope_tol)))

    
    vloss_slope_tol = 1e-6
    vloss_pass = bool(np.isfinite(vloss_slope_late) and (vloss_slope_late <= float(vloss_slope_tol)))

    
    kl_warn = bool(np.isfinite(late_kl_pd_mean) and (late_kl_pd_mean > 0.02)) or bool(np.isfinite(late_kl_pd_p95) and (late_kl_pd_p95 > 0.05))
    clip_warn = bool(np.isfinite(late_clip_mean) and (late_clip_mean > 0.60)) or bool(np.isfinite(late_clip_p95) and (late_clip_p95 > 0.95))

    
    passed_basic = bool(ev_pass and reward_pass and vloss_pass)
    
    diag_pass = bool(np.isfinite(late_kl_pd_mean) and (late_kl_pd_mean <= 0.02)) and bool(np.isfinite(late_clip_mean) and (late_clip_mean <= 0.50))
    passed_strict = bool(passed_basic and diag_pass)

    return {
        "metrics_csv": str(metrics_csv_path),
        "n_points": int(n),
        "window_early": int(w_early),
        "window_late": int(w_late),
        "reward_ema_early_mean": float(early_reward_mean),
        "reward_ema_late_mean": float(late_reward_mean),
        "reward_slope_late": float(reward_slope_late),
        "value_loss_ema_early_mean": float(early_vloss_mean),
        "value_loss_ema_late_mean": float(late_vloss_mean),
        "value_loss_slope_late": float(vloss_slope_late),
        "ev_ema_late_mean": float(late_ev_mean),
        "ev_ema_late_p10": float(late_ev_p10),
        "ev_ema_late_mean_smoothed": float(late_ev_ema_mean),
        
        "kl_late_mean": float(late_kl_pd_mean),
        "kl_late_p95": float(late_kl_pd_p95),
        "clipfrac_late_mean": float(late_clip_mean),
        "clipfrac_late_p95": float(late_clip_p95),
        
        "kl_sum_late_mean": float(late_kl_sum_mean),
        "kl_sum_late_p95": float(late_kl_sum_p95),
        "kl_per_dim_late_mean": float(late_kl_pd_mean),
        "kl_per_dim_late_p95": float(late_kl_pd_p95),
        "clipfrac_per_dim_late_mean": float(late_clip_pd_mean),
        "clipfrac_per_dim_late_p95": float(late_clip_pd_p95),
        "ev_pass": bool(ev_pass),
        "reward_pass": bool(reward_pass),
        "value_loss_pass": bool(vloss_pass),
        "kl_warn": bool(kl_warn),
        "clipfrac_warn": bool(clip_warn),
        "diag_pass_strict": bool(diag_pass),
        "pass_basic": bool(passed_basic),
        "pass_strict": bool(passed_strict),
        "pass": bool(passed_basic),
    }


def compute_phaseA_acceptance_from_csv(metrics_csv_path: Path) -> Dict[str, Any]:
    """
    Phase AP0 50k steps_v13A4/A5

    PhaseA  PPO 
    - pass_basicclipfrac/KL  + anti-spoof  PhaseB v13 A5
    - pass_strict pass_basic reward v13 A4.2 

    
    - clipfrac_mean_late  [0.05, 0.25]
    - kl_per_dim_late    [5e-4, 5e-3]
    - reward_train_step_meanlate  early  20% 0.1std_early

    
    -  train_metrics.csv  update early/late / 20%5
    - reward_train_step_mean/clipfrac_mean
    """
    m = read_train_metrics_csv(metrics_csv_path)
    if not m or "global_step" not in m:
        return {"pass": False, "reason": "missing_train_metrics_csv", "metrics_csv": str(metrics_csv_path)}

    steps = np.asarray(m.get("global_step", []), dtype=np.float64).reshape(-1)
    n = int(steps.size)
    if n < 10:
        return {"pass": False, "reason": "too_few_points", "n_points": n, "metrics_csv": str(metrics_csv_path)}

    steps_end = int(np.nanmax(steps)) if np.isfinite(steps).any() else 0
    if steps_end < 45000:
        return {
            "pass": False,
            "reason": "too_few_steps_for_phaseA",
            "steps_end": int(steps_end),
            "n_points": int(n),
            "metrics_csv": str(metrics_csv_path),
        }

    w = max(5, int(round(0.20 * n)))
    w = min(w, n)

    
    reward_step = m.get(
        "reward_train_step_mean",
        m.get("train_reward_step_mean", m.get("train_reward_mean", np.full((n,), np.nan, dtype=np.float64))),
    )
    reward_step = np.asarray(reward_step, dtype=np.float64).reshape(-1)
    if reward_step.size != n:
        reward_step = np.resize(reward_step, (n,))

    R_early = float(np.nanmean(reward_step[:w]))
    R_late = float(np.nanmean(reward_step[-w:]))
    std_early = float(np.nanstd(reward_step[:w]))
    rel_improve = float((R_late - R_early) / (abs(R_early) + 1e-6))
    abs_improve = float(R_late - R_early)
    reward_pass = bool((rel_improve >= 0.20) or (abs_improve >= 0.10 * std_early))

    
    clipfrac_mean = m.get("clipfrac_mean", m.get("clipfrac", np.full((n,), np.nan, dtype=np.float64)))
    clipfrac_mean = np.asarray(clipfrac_mean, dtype=np.float64).reshape(-1)
    if clipfrac_mean.size != n:
        clipfrac_mean = np.resize(clipfrac_mean, (n,))
    clipfrac_late_mean = float(np.nanmean(clipfrac_mean[-w:]))
    clipfrac_pass = bool(np.isfinite(clipfrac_late_mean) and (0.05 <= clipfrac_late_mean <= 0.25))

    
    kl_per_dim = m.get("kl_per_dim", np.full((n,), np.nan, dtype=np.float64))
    kl_per_dim = np.asarray(kl_per_dim, dtype=np.float64).reshape(-1)
    if kl_per_dim.size != n:
        kl_per_dim = np.resize(kl_per_dim, (n,))

    
    if not np.isfinite(kl_per_dim).any():
        kl_sum = np.asarray(m.get("kl", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)
        act_dim = np.asarray(m.get("act_dim", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)
        if kl_sum.size != n:
            kl_sum = np.resize(kl_sum, (n,))
        if act_dim.size != n:
            act_dim = np.resize(act_dim, (n,))
        denom = np.maximum(act_dim, 1.0)
        kl_per_dim = kl_sum / denom

    kl_per_dim_late_mean = float(np.nanmean(kl_per_dim[-w:]))
    kl_pass = bool(np.isfinite(kl_per_dim_late_mean) and (5e-4 <= kl_per_dim_late_mean <= 5e-3))

    
    
    # - logp_scale1/act_dim_effective
    
    logp_is_mean_tail = float("nan")
    logp_scale_tail = float("nan")
    act_dim_effective_tail = float("nan")
    try:
        if "logp_is_mean" in m:
            logp_is_mean_tail = float(np.nanmean(np.asarray(m.get("logp_is_mean"), dtype=np.float64).reshape(-1)[-w:]))
        if "logp_scale" in m:
            logp_scale_tail = float(np.nanmean(np.asarray(m.get("logp_scale"), dtype=np.float64).reshape(-1)[-w:]))
        if "act_dim_effective" in m:
            act_dim_effective_tail = float(np.nanmean(np.asarray(m.get("act_dim_effective"), dtype=np.float64).reshape(-1)[-w:]))
    except Exception:
        pass

    anti_spoof_pass = True
    anti_spoof_reason = ""
    try:
        if not np.isfinite(act_dim_effective_tail):
            anti_spoof_pass = False
            anti_spoof_reason = "missing_act_dim_effective"
        else:
            act_eff = int(max(1, int(round(float(act_dim_effective_tail)))))
            if act_eff > 10:
                if (not np.isfinite(logp_is_mean_tail)) or (float(logp_is_mean_tail) < 0.5):
                    anti_spoof_pass = False
                    anti_spoof_reason = "logp_not_mean"
                exp_scale = 1.0 / float(act_eff)
                if (not np.isfinite(logp_scale_tail)) or (abs(float(logp_scale_tail) - float(exp_scale)) > 1e-6):
                    anti_spoof_pass = False
                    anti_spoof_reason = "logp_scale_mismatch"
    except Exception:
        anti_spoof_pass = False
        anti_spoof_reason = "anti_spoof_exception"

    passed_basic = bool(clipfrac_pass and kl_pass and anti_spoof_pass)
    passed_strict = bool(passed_basic and reward_pass)

    
    passed = bool(passed_basic)
    return {
        "metrics_csv": str(metrics_csv_path),
        "n_points": int(n),
        "steps_end": int(steps_end),
        "window": int(w),
        "pass_basic": bool(passed_basic),
        "pass_strict": bool(passed_strict),
        "reward_pass": bool(reward_pass),
        "R_early": float(R_early),
        "R_late": float(R_late),
        "std_early": float(std_early),
        "rel_improve": float(rel_improve),
        "abs_improve": float(abs_improve),
        "clipfrac_pass": bool(clipfrac_pass),
        "clipfrac_mean_late": float(clipfrac_late_mean),
        "kl_pass": bool(kl_pass),
        "kl_per_dim_late": float(kl_per_dim_late_mean),
        "logp_is_mean_tail": float(logp_is_mean_tail),
        "logp_scale_tail": float(logp_scale_tail),
        "act_dim_effective_tail": float(act_dim_effective_tail),
        "anti_spoof_pass": bool(anti_spoof_pass),
        "anti_spoof_reason": str(anti_spoof_reason),
        "pass": bool(passed),
    }


def compute_phaseA_canary_acceptance_from_csv(metrics_csv_path: Path) -> Dict[str, Any]:
    """
    Phase AP0canary 10k steps PPO 

     v13  A4.1
    - clipfrac_mean  <0.350.05~0.25
    - kl_per_dim early-stop0
    - entropy  NaN/Inf
    """
    m = read_train_metrics_csv(metrics_csv_path)
    if not m or "global_step" not in m:
        return {"pass": False, "reason": "missing_train_metrics_csv", "metrics_csv": str(metrics_csv_path)}

    steps = np.asarray(m.get("global_step", []), dtype=np.float64).reshape(-1)
    n = int(steps.size)
    if n < 3:
        return {"pass": False, "reason": "too_few_points", "n_points": int(n), "metrics_csv": str(metrics_csv_path)}

    steps_end = int(np.nanmax(steps)) if np.isfinite(steps).any() else 0
    if steps_end < 8000:
        return {
            "pass": False,
            "reason": "too_few_steps_for_canary",
            "steps_end": int(steps_end),
            "n_points": int(n),
            "metrics_csv": str(metrics_csv_path),
        }

    w = max(3, int(round(0.20 * n)))
    w = min(w, n)

    clipfrac_mean = m.get("clipfrac_mean", m.get("clipfrac", np.full((n,), np.nan, dtype=np.float64)))
    clipfrac_mean = np.asarray(clipfrac_mean, dtype=np.float64).reshape(-1)
    if clipfrac_mean.size != n:
        clipfrac_mean = np.resize(clipfrac_mean, (n,))
    clip_tail = float(np.nanmean(clipfrac_mean[-w:]))
    clip_ok = bool(np.isfinite(clip_tail) and (clip_tail < 0.35))

    kl_pd = np.asarray(m.get("kl_per_dim", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)
    if kl_pd.size != n:
        kl_pd = np.resize(kl_pd, (n,))
    kl_tail = float(np.nanmean(kl_pd[-w:]))
    
    kl_ok = bool(np.isfinite(kl_tail) and (1e-6 < kl_tail < 5e-2))

    ent_pd = np.asarray(m.get("entropy_per_dim", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)
    if ent_pd.size != n:
        ent_pd = np.resize(ent_pd, (n,))
    ent_tail = float(np.nanmean(ent_pd[-w:]))
    ent_ok = bool(np.isfinite(ent_tail))

    passed = bool(clip_ok and kl_ok and ent_ok)
    return {
        "metrics_csv": str(metrics_csv_path),
        "n_points": int(n),
        "steps_end": int(steps_end),
        "window": int(w),
        "clipfrac_mean_tail": float(clip_tail),
        "kl_per_dim_tail": float(kl_tail),
        "entropy_per_dim_tail": float(ent_tail),
        "clip_pass": bool(clip_ok),
        "kl_pass": bool(kl_ok),
        "entropy_pass": bool(ent_ok),
        "pass": bool(passed),
    }


def compute_phaseZ_stage_acceptance_from_csv(
    metrics_csv_path: Path,
    *,
    stage: str = "auto",
) -> Dict[str, Any]:
    """
    Phase Z stage1/2/3/4

    
    - stage1~2krewardclipfracreturns
    - stage2~30k +  + 1/2/5
    - stage3~100krewardcost+ 1/2/5
    - stage4~500kstage310CoMP
    """
    m = read_train_metrics_csv(metrics_csv_path)
    if not m or "global_step" not in m:
        return {"pass": False, "reason": "missing_train_metrics_csv", "metrics_csv": str(metrics_csv_path)}

    steps = np.asarray(m.get("global_step", []), dtype=np.float64).reshape(-1)
    n = int(steps.size)
    if n < 2:
        return {"pass": False, "reason": "too_few_points", "n_points": int(n), "metrics_csv": str(metrics_csv_path)}

    steps_end = int(np.nanmax(steps)) if np.isfinite(steps).any() else 0

    def _arr(key: str, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        raw = m.get(key, fallback if fallback is not None else np.full((n,), np.nan, dtype=np.float64))
        x = np.asarray(raw, dtype=np.float64).reshape(-1)
        if x.size != n:
            x = np.resize(x, (n,))
        return x

    reward_ep = _arr("reward_total_ep", fallback=_arr("train_reward_mean"))
    paper_cost_ep = _arr("paper_cost_ep", fallback=_arr("paper_cost_mean"))
    clipfrac = _arr("clipfrac_mean", fallback=_arr("clipfrac"))
    value_loss = _arr("value_loss")
    explained_variance = _arr("explained_variance")
    entropy_per_dim = _arr("entropy_per_dim", fallback=_arr("entropy"))
    violation_cont = _arr("constraint_signal_step_mean", fallback=_arr("vio_metric_mean"))
    returns_mean = _arr("returns_mean")
    returns_std = _arr("returns_std")
    value_bias_proxy = _arr("value_bias_proxy")
    comp_enable_rate = _arr("comp_enable_rate")
    service_switch_count = _arr("service_switch_count")
    theta_entropy = _arr("theta_entropy")
    theta_mode_effective_agent_frac = _arr("theta_mode_effective_agent_frac")
    reward_proximity_step = _arr("reward_proximity_step_mean")
    reward_main_step = _arr("reward_main_step_mean", fallback=_arr("reward_paper_step_mean"))

    stage_raw = str(stage or "auto").strip().lower()
    stage_map = {
        "stage1": "stage1", "s1": "stage1", "1": "stage1",
        "stage2": "stage2", "s2": "stage2", "2": "stage2",
        "stage3": "stage3", "s3": "stage3", "3": "stage3",
        "stage4": "stage4", "s4": "stage4", "4": "stage4",
    }
    if stage_raw == "auto":
        if steps_end < 24000:
            stage_now = "stage1"
        elif steps_end < 90000:
            stage_now = "stage2"
        elif steps_end < 450000:
            stage_now = "stage3"
        else:
            stage_now = "stage4"
    else:
        stage_now = stage_map.get(stage_raw, "stage1")

    min_steps_req = {"stage1": 1500, "stage2": 24000, "stage3": 90000, "stage4": 450000}
    min_steps = int(min_steps_req.get(stage_now, 0))
    if steps_end < min_steps:
        return {
            "pass": False,
            "reason": f"too_few_steps_for_{stage_now}",
            "stage": stage_now,
            "steps_end": int(steps_end),
            "min_steps_required": int(min_steps),
            "n_points": int(n),
            "metrics_csv": str(metrics_csv_path),
        }

    w = max(2, int(round(0.20 * n)))
    w = min(w, n)
    w_early_s3 = max(2, int(round(0.30 * n)))
    w_early_s3 = min(w_early_s3, n)
    w_late_s3 = max(2, int(round(0.25 * n)))
    w_late_s3 = min(w_late_s3, n)
    w_early_s2 = max(2, int(round(0.30 * n)))
    w_early_s2 = min(w_early_s2, n)

    reward_tail_mean = float(np.nanmean(reward_ep[-w:]))
    reward_tail_std = float(np.nanstd(reward_ep[-w:]))
    paper_cost_tail_mean = float(np.nanmean(paper_cost_ep[-w:]))
    clipfrac_tail = float(np.nanmean(clipfrac[-w:]))
    value_loss_tail = float(np.nanmean(value_loss[-w:]))
    entropy_tail = float(np.nanmean(entropy_per_dim[-w:]))
    ev_tail = float(np.nanmean(explained_variance[-w:]))
    violation_tail = float(np.nanmean(violation_cont[-w:]))
    returns_tail_mean = float(np.nanmean(returns_mean[-w:]))
    returns_tail_std = float(np.nanmean(returns_std[-w:]))
    value_bias_tail_mean = float(np.nanmean(value_bias_proxy[-w:]))
    comp_enable_tail = float(np.nanmean(comp_enable_rate[-w:]))
    comp_tail_lock_frac = float(np.nanmean((comp_enable_rate[-w:] <= 0.05) | (comp_enable_rate[-w:] >= 0.95)))
    switch_tail = float(np.nanmean(service_switch_count[-w:]))
    theta_entropy_tail = float(np.nanmean(theta_entropy[-w:]))
    theta_entropy_early = float(np.nanmean(theta_entropy[:w_early_s2]))
    theta_agent_frac_tail = float(np.nanmean(theta_mode_effective_agent_frac[-w:]))
    pm_mask = np.isfinite(reward_proximity_step) & np.isfinite(reward_main_step)
    pm_ratio = np.asarray([], dtype=np.float64)
    if np.any(pm_mask):
        px = np.abs(np.asarray(reward_proximity_step[pm_mask], dtype=np.float64))
        mx = np.abs(np.asarray(reward_main_step[pm_mask], dtype=np.float64))
        pm_ratio = np.asarray(px / np.maximum(mx, 1e-9), dtype=np.float64)
    pm_stage = _h11_stage_means(pm_ratio)
    pm_ratio_early = float(pm_stage.get("early", float("nan")))
    pm_ratio_mid = float(pm_stage.get("mid", float("nan")))
    pm_ratio_late = float(pm_stage.get("late", float("nan")))
    
    theta_mode_solver = bool(np.isfinite(theta_agent_frac_tail) and (theta_agent_frac_tail <= 0.05))

    reward_early_s3 = float(np.nanmean(reward_ep[:w_early_s3]))
    reward_late_s3 = float(np.nanmean(reward_ep[-w_late_s3:]))
    reward_gain_s3 = float(reward_late_s3 - reward_early_s3)
    cost_early_s3 = float(np.nanmean(paper_cost_ep[:w_early_s3]))
    cost_late_s3 = float(np.nanmean(paper_cost_ep[-w_late_s3:]))

    
    w_metric1 = max(2, int(round(0.10 * n)))
    w_metric1 = min(w_metric1, n)
    cost_first_10pct = float(np.nanmean(paper_cost_ep[:w_metric1]))
    cost_last_10pct = float(np.nanmean(paper_cost_ep[-w_metric1:]))
    relative_drop_pct = float((cost_first_10pct - cost_last_10pct) / max(abs(cost_first_10pct), 1e-9) * 100.0)
    metric1_non_worse = bool(np.isfinite(cost_first_10pct) and np.isfinite(cost_last_10pct) and (cost_last_10pct <= cost_first_10pct))
    metric1_drop_positive = bool(np.isfinite(relative_drop_pct) and (relative_drop_pct > 0.0))

    
    idx_30 = int(max(1, round(0.30 * n)))
    idx_75 = int(max(idx_30 + 1, round(0.75 * n)))
    idx_75 = min(idx_75, n - 1)
    reward_early_m2 = float(np.nanmean(reward_ep[:idx_30]))
    reward_mid_m2 = float(np.nanmean(reward_ep[idx_30:idx_75]))
    reward_late_m2 = float(np.nanmean(reward_ep[idx_75:]))
    reward_peak = float(np.nanmax(reward_ep)) if np.isfinite(reward_ep).any() else float("nan")
    reward_max_drawdown_pct = (
        float((reward_peak - reward_late_m2) / max(abs(reward_peak), 1e-9) * 100.0)
        if np.isfinite(reward_peak) and np.isfinite(reward_late_m2)
        else float("nan")
    )
    metric2_non_collapse = bool(
        np.isfinite(reward_late_m2)
        and np.isfinite(reward_early_m2)
        and (reward_late_m2 >= (reward_early_m2 - max(10.0, 0.35 * abs(reward_early_m2))))
    )
    metric2_upward = bool(np.isfinite(reward_late_m2) and np.isfinite(reward_early_m2) and (reward_late_m2 > reward_early_m2))

    
    metric5_checks = {
        "ev_tail_gt_0p15": bool(np.isfinite(ev_tail) and (ev_tail > 0.15)),
        "value_loss_tail_lt_15": bool(np.isfinite(value_loss_tail) and (value_loss_tail < 15.0)),
        "entropy_tail_gt_neg1p5": bool(np.isfinite(entropy_tail) and (entropy_tail > -1.5)),
        "clipfrac_tail_gt_0p005": bool(np.isfinite(clipfrac_tail) and (clipfrac_tail > 0.005)),
    }
    metric5_pass_basic = bool(all(metric5_checks.values()))

    
    run_dir = Path(metrics_csv_path).resolve().parent
    metric3_core_files = [
        run_dir / "figs" / "Ablation_10Loads eval" / "Ablation10_PaperCost.png",
        run_dir / "figs" / "Ablation_10Loads eval" / "Ablation10_Delay.png",
        run_dir / "figs" / "Ablation_10Loads eval" / "Ablation10_Energy.png",
        run_dir / "figs" / "Ablation_10Loads eval" / "Ablation10_Ct.png",
    ]
    metric3_core_ready = bool(all(p.exists() for p in metric3_core_files))
    metric3_eval_ready = bool(any((run_dir / "evals").glob("*/eval_summary_*.json"))) if (run_dir / "evals").exists() else False
    metric3_ready = bool(metric3_core_ready and metric3_eval_ready)
    metric4_ready = bool((run_dir / "figs" / "DynamicCoMP_Bundle evalCoMP" / "bundle_summary.json").exists())

    checks: Dict[str, bool] = {}
    if stage_now == "stage1":
        paper_cost_high_optional = bool(np.isfinite(paper_cost_tail_mean) and (paper_cost_tail_mean >= 1.2))
        checks = {
            "reward_total_finite_and_variable": bool(np.isfinite(reward_tail_mean) and np.isfinite(reward_tail_std) and (reward_tail_std > 1e-6)),
            "clipfrac_nonzero": bool(np.isfinite(clipfrac_tail) and (clipfrac_tail > 1e-6)),
            "returns_stats_available": bool(np.isfinite(returns_tail_mean) and np.isfinite(returns_tail_std) and (returns_tail_std > 1e-8)),
            "value_bias_proxy_available": bool(np.isfinite(value_bias_tail_mean)),
            "paper_cost_high_optional": bool(paper_cost_high_optional),
        }
        hard_keys = [k for k in checks.keys() if k != "paper_cost_high_optional"]
        pass_all = bool(all(checks[k] for k in hard_keys))
    elif stage_now == "stage2":
        theta_entropy_desc = bool(np.isfinite(theta_entropy_tail) and np.isfinite(theta_entropy_early) and (theta_entropy_tail < theta_entropy_early))
        theta_entropy_observable = bool(theta_mode_solver or np.isfinite(theta_entropy_tail))
        theta_entropy_gate = bool(theta_mode_solver or theta_entropy_desc or np.isfinite(theta_entropy_tail))
        checks = {
            "clipfrac_mean_late_gt_0p01": bool(np.isfinite(clipfrac_tail) and (clipfrac_tail > 0.01)),
            "reward_total_nonzero_variance": bool(np.isfinite(reward_tail_std) and (reward_tail_std > 1e-6)),
            "value_loss_late_lt_15": bool(np.isfinite(value_loss_tail) and (value_loss_tail < 15.0)),
            "entropy_per_dim_late_gt_neg1p5": bool(np.isfinite(entropy_tail) and (entropy_tail > -1.5)),
            "comp_enable_rate_not_locked": bool(np.isfinite(comp_enable_tail) and (comp_enable_tail > 0.05) and (comp_enable_tail < 0.95)),
            "comp_enable_rate_lock_frac_le_0p8": bool(np.isfinite(comp_tail_lock_frac) and (comp_tail_lock_frac <= 0.80)),
            "prox_main_ratio_early_le_0p45": bool(np.isfinite(pm_ratio_early) and (pm_ratio_early <= 0.45)),
            "prox_main_ratio_mid_le_0p35": bool(np.isfinite(pm_ratio_mid) and (pm_ratio_mid <= 0.35)),
            "prox_main_ratio_late_le_0p25": bool(np.isfinite(pm_ratio_late) and (pm_ratio_late <= 0.25)),
            "prox_main_ratio_descend": bool(
                np.isfinite(pm_ratio_early)
                and np.isfinite(pm_ratio_mid)
                and np.isfinite(pm_ratio_late)
                and (pm_ratio_early >= pm_ratio_mid - 1e-6)
                and (pm_ratio_mid >= pm_ratio_late - 1e-6)
            ),
            "service_switch_count_gt_0": bool(np.isfinite(switch_tail) and (switch_tail > 0.0)),
            "theta_entropy_observable": bool(theta_entropy_observable),
            "theta_entropy_descend_or_non_nan": bool(theta_entropy_gate),
            "metric1_paper_cost_non_worse": bool(metric1_non_worse),
            "metric2_reward_non_collapse": bool(metric2_non_collapse),
            "metric5_train_health_basic": bool(metric5_pass_basic),
        }
        pass_all = bool(all(checks.values()))
    elif stage_now == "stage3":
        checks = {
            "reward_total_gain_gt_50": bool(np.isfinite(reward_gain_s3) and (reward_gain_s3 > 50.0)),
            "explained_variance_late_gt_0p15": bool(np.isfinite(ev_tail) and (ev_tail > 0.15)),
            "paper_cost_late_lt_early": bool(np.isfinite(cost_late_s3) and np.isfinite(cost_early_s3) and (cost_late_s3 < cost_early_s3)),
            "violation_continuous_ema_lt_0p5": bool(np.isfinite(violation_tail) and (violation_tail < 0.50)),
            "metric1_paper_cost_non_worse": bool(metric1_non_worse),
            "metric2_reward_upward": bool(metric2_upward),
            "metric5_train_health_basic": bool(metric5_pass_basic),
        }
        pass_all = bool(all(checks.values()))
    else:
        checks = {
            "stage3_core_like_pass": bool(
                np.isfinite(reward_gain_s3) and (reward_gain_s3 > 50.0)
                and np.isfinite(ev_tail) and (ev_tail > 0.15)
                and np.isfinite(cost_late_s3) and np.isfinite(cost_early_s3) and (cost_late_s3 < cost_early_s3)
                and np.isfinite(violation_tail) and (violation_tail < 0.50)
            ),
            "metric1_drop_positive": bool(metric1_drop_positive),
            "metric2_reward_upward": bool(metric2_upward),
            "metric5_train_health_basic": bool(metric5_pass_basic),
            "metric3_ablation_ready": bool(metric3_ready),
            "metric4_dynamic_comp_ready": bool(metric4_ready),
        }
        pass_all = bool(all(checks.values()))

    failed = [k for k, v in checks.items() if not bool(v)]
    reason = "ok" if pass_all else f"failed_checks: {failed}"

    return {
        "metrics_csv": str(metrics_csv_path),
        "stage": stage_now,
        "n_points": int(n),
        "steps_end": int(steps_end),
        "window": int(w),
        "window_early_s2": int(w_early_s2),
        "window_early_s3": int(w_early_s3),
        "window_late_s3": int(w_late_s3),
        "reward_tail_mean": float(reward_tail_mean),
        "reward_tail_std": float(reward_tail_std),
        "paper_cost_tail_mean": float(paper_cost_tail_mean),
        "clipfrac_tail_mean": float(clipfrac_tail),
        "value_loss_tail_mean": float(value_loss_tail),
        "entropy_per_dim_tail_mean": float(entropy_tail),
        "ev_tail_mean": float(ev_tail),
        "violation_continuous_tail_mean": float(violation_tail),
        "returns_tail_mean": float(returns_tail_mean),
        "returns_tail_std": float(returns_tail_std),
        "value_bias_proxy_tail_mean": float(value_bias_tail_mean),
        "comp_enable_rate_tail_mean": float(comp_enable_tail),
        "comp_enable_rate_tail_lock_frac": float(comp_tail_lock_frac),
        "service_switch_count_tail_mean": float(switch_tail),
        "theta_entropy_early_s2_mean": float(theta_entropy_early),
        "theta_entropy_tail_mean": float(theta_entropy_tail),
        "theta_mode_effective_agent_frac_tail_mean": float(theta_agent_frac_tail),
        "theta_mode_solver_softpass": bool(theta_mode_solver),
        "prox_main_ratio_early_mean": float(pm_ratio_early),
        "prox_main_ratio_mid_mean": float(pm_ratio_mid),
        "prox_main_ratio_late_mean": float(pm_ratio_late),
        "reward_early_s3_mean": float(reward_early_s3),
        "reward_late_s3_mean": float(reward_late_s3),
        "reward_gain_s3_mean": float(reward_gain_s3),
        "paper_cost_early_s3_mean": float(cost_early_s3),
        "paper_cost_late_s3_mean": float(cost_late_s3),
        "five_metrics": {
            "metric1_paper_cost_trend": {
                "paper_cost_first_10pct_mean": float(cost_first_10pct),
                "paper_cost_last_10pct_mean": float(cost_last_10pct),
                "relative_drop_pct": float(relative_drop_pct),
                "pass_non_worse": bool(metric1_non_worse),
                "pass_drop_positive": bool(metric1_drop_positive),
            },
            "metric2_reward_shape": {
                "reward_early_mean": float(reward_early_m2),
                "reward_mid_mean": float(reward_mid_m2),
                "reward_late_mean": float(reward_late_m2),
                "max_drawdown_pct": float(reward_max_drawdown_pct),
                "pass_non_collapse": bool(metric2_non_collapse),
                "pass_upward": bool(metric2_upward),
            },
            "metric3_ablation_10loads": {
                "ready": bool(metric3_ready),
                "core_figs_ready": bool(metric3_core_ready),
                "eval_summary_ready": bool(metric3_eval_ready),
                "status": "ready" if metric3_ready else "pending_external_eval",
            },
            "metric4_dynamic_comp_viz": {
                "ready": bool(metric4_ready),
                "status": "ready" if metric4_ready else "pending_external_eval",
            },
            "metric5_training_health": {
                "checks": metric5_checks,
                "pass_basic": bool(metric5_pass_basic),
            },
        },
        "checks": checks,
        "pass": bool(pass_all),
        "reason": str(reason),
    }

def compute_phaseB_acceptance_from_csv(metrics_csv_path: Path) -> Dict[str, Any]:
    """
    Phase BP1traj_only_v13B4150k steps

    PhaseEP1.5
    - Total rewardspaper_return_ep := -paper_cost_ep
    -  PhaseB  FixedEvalCurve  `eval_det_paper_return_mean` cost 

    
    1) 10scores_mode=fixed  act_dim_effective==10 train_metrics.csv
    2) paper_return_ep late vs early 20%  0.1std_early 20% 
    3) vio_any_frac late  1.2early
    4) PPO clipfrac_mean  [0.05, 0.25]kl_per_dim  [5e-4, 5e-3]
    5) FixedEvalCurveCI95 paper_return / paper_cost / vio_any
    """
    m = read_train_metrics_csv(metrics_csv_path)
    if not m or "global_step" not in m:
        return {"pass": False, "reason": "missing_train_metrics_csv", "metrics_csv": str(metrics_csv_path)}

    steps = m.get("global_step", np.asarray([], dtype=np.float64))
    steps = np.asarray(steps, dtype=np.float64).reshape(-1)
    n = int(steps.size)
    if n < 10:
        return {"pass": False, "reason": "too_few_points", "n_points": n, "metrics_csv": str(metrics_csv_path)}

    steps_end = int(np.nanmax(steps)) if np.isfinite(steps).any() else 0
    
    if steps_end < 142500:
        return {
            "pass": False,
            "reason": "too_few_steps_for_phaseB",
            "steps_end": int(steps_end),
            "metrics_csv": str(metrics_csv_path),
        }

    w = max(5, int(round(0.20 * n)))
    w = min(w, n)

    
    act_dim_eff = np.asarray(m.get("act_dim_effective", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)
    scores_mode_fixed = np.asarray(m.get("scores_mode_fixed", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)
    if act_dim_eff.size != n:
        act_dim_eff = np.resize(act_dim_eff, (n,))
    if scores_mode_fixed.size != n:
        scores_mode_fixed = np.resize(scores_mode_fixed, (n,))

    act_dim_eff_tail = float(np.nanmean(act_dim_eff[-w:]))
    scores_fixed_tail = float(np.nanmean(scores_mode_fixed[-w:]))
    true10_pass = bool(np.isfinite(act_dim_eff_tail) and np.isfinite(scores_fixed_tail) and (abs(act_dim_eff_tail - 10.0) <= 0.1) and (scores_fixed_tail >= 0.9))

    
    vio_any = np.asarray(m.get("vio_any_frac", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)
    if vio_any.size != n:
        vio_any = np.resize(vio_any, (n,))
    V_early = float(np.nanmean(vio_any[:w]))
    V_late = float(np.nanmean(vio_any[-w:]))
    vio_pass = bool(np.isfinite(V_early) and np.isfinite(V_late) and (V_late <= 1.20 * V_early))

    
    clipfrac_mean = np.asarray(m.get("clipfrac_mean", m.get("clipfrac", np.full((n,), np.nan, dtype=np.float64))), dtype=np.float64).reshape(-1)
    if clipfrac_mean.size != n:
        clipfrac_mean = np.resize(clipfrac_mean, (n,))
    clip_tail = float(np.nanmean(clipfrac_mean[-w:]))
    clip_pass = bool(np.isfinite(clip_tail) and (0.05 <= clip_tail <= 0.25))

    kl_pd = np.asarray(m.get("kl_per_dim", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)
    if kl_pd.size != n:
        kl_pd = np.resize(kl_pd, (n,))
    kl_tail = float(np.nanmean(kl_pd[-w:]))
    kl_pass = bool(np.isfinite(kl_tail) and (5e-4 <= kl_tail <= 5e-3))

    
    eval_n = np.asarray(m.get("eval_det_n", np.zeros((n,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    if eval_n.size != n:
        eval_n = np.resize(eval_n, (n,))
    eval_mask = np.isfinite(eval_n) & (eval_n > 0.0)

    fixed_eval_pass = False
    fixed_eval_reason = ""
    fixed_eval_details: Dict[str, Any] = {}
    try:
        if int(np.sum(eval_mask)) < 3:
            fixed_eval_pass = False
            fixed_eval_reason = "too_few_fixed_eval_points"
        else:
            es = steps[eval_mask]
            ec = np.asarray(m.get("eval_det_paper_cost_mean", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)[eval_mask]
            ec_ci = np.asarray(m.get("eval_det_paper_cost_ci", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)[eval_mask]

            ev = np.asarray(m.get("eval_det_vio_any_frac_mean", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)[eval_mask]
            ev_ci = np.asarray(m.get("eval_det_vio_any_frac_ci", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)[eval_mask]

            
            er = np.asarray(m.get("eval_det_paper_return_mean", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)[eval_mask]
            er_ci = np.asarray(m.get("eval_det_paper_return_ci", np.full((n,), np.nan, dtype=np.float64)), dtype=np.float64).reshape(-1)[eval_mask]
            if (not np.isfinite(er).any()) and np.isfinite(ec).any():
                er = -ec
            if (not np.isfinite(er_ci).any()) and np.isfinite(ec_ci).any():
                er_ci = ec_ci

            ne = int(es.size)
            
            we = int(round(0.20 * ne))
            if ne >= 2:
                we = max(2, we)
            else:
                we = max(1, we)
            we = min(we, ne)

            er_early = float(np.nanmean(er[:we]))
            er_late = float(np.nanmean(er[-we:]))
            er_std_early = float(np.nanstd(er[:we]))
            er_rel = float((er_late - er_early) / (abs(er_early) + 1e-6))
            er_abs = float(er_late - er_early)
            er_pass = bool((er_rel >= 0.20) or (er_abs >= 0.10 * er_std_early))

            slope_late_e = linreg_slope(es[-we:], er[-we:]) if ne >= 2 else float("nan")

            
            slope_mid_e = float("nan")
            try:
                mid_i0 = int(np.floor(0.30 * ne))
                mid_i1 = int(np.ceil(0.70 * ne))
                if (mid_i1 - mid_i0) < 2:
                    mid_i0 = max(0, min(int(ne // 2) - 1, ne - 2))
                    mid_i1 = mid_i0 + 2
                if (mid_i1 - mid_i0) >= 2:
                    slope_mid_e = linreg_slope(es[mid_i0:mid_i1], er[mid_i0:mid_i1])
            except Exception:
                slope_mid_e = float("nan")

            
            if np.isfinite(slope_late_e) and np.isfinite(slope_mid_e):
                er_plateau = bool(abs(slope_late_e) <= max(0.5 * abs(slope_mid_e), 1e-6))
            else:
                er_plateau = True

            
            def _ci_shrink(ci: np.ndarray) -> bool:
                ci = np.asarray(ci, dtype=np.float64).reshape(-1)
                if ci.size < we:
                    return False
                c0 = float(np.nanmean(ci[:we]))
                c1 = float(np.nanmean(ci[-we:]))
                if (not np.isfinite(c0)) or (not np.isfinite(c1)):
                    return False
                if c0 <= 1e-9:
                    return True
                return bool(c1 <= 0.90 * c0)

            er_ci_pass = _ci_shrink(er_ci)
            ec_ci_pass = _ci_shrink(ec_ci)
            ev_ci_pass = _ci_shrink(ev_ci)

            
            
            fixed_eval_pass = bool(er_pass and er_plateau and er_ci_pass and ec_ci_pass)
            fixed_eval_reason = "" if fixed_eval_pass else "fixed_eval_failed"
            fixed_eval_details = {
                "n_points": int(ne),
                "window": int(we),
                "paper_return_pass": bool(er_pass),
                "paper_return_plateau_pass": bool(er_plateau),
                "paper_return_ci_shrink_pass": bool(er_ci_pass),
                "paper_cost_ci_shrink_pass": bool(ec_ci_pass),
                "vio_any_ci_shrink_pass": bool(ev_ci_pass),
                "R_early": float(er_early),
                "R_late": float(er_late),
                "std_early": float(er_std_early),
                "slope_mid": float(slope_mid_e),
                "slope_late": float(slope_late_e),
                "ci_early_mean": float(np.nanmean(er_ci[:we])),
                "ci_late_mean": float(np.nanmean(er_ci[-we:])),
            }
    except Exception:
        fixed_eval_pass = False
        fixed_eval_reason = "fixed_eval_exception"
        fixed_eval_details = {}

    passed = bool(true10_pass and vio_pass and clip_pass and kl_pass and fixed_eval_pass)
    out: Dict[str, Any] = {
        "metrics_csv": str(metrics_csv_path),
        "n_points": int(n),
        "window": int(w),
        "steps_end": int(steps_end),
        "true10_pass": bool(true10_pass),
        "act_dim_effective_tail": float(act_dim_eff_tail),
        "scores_mode_fixed_tail": float(scores_fixed_tail),
        "vio_pass": bool(vio_pass),
        "V_early": float(V_early),
        "V_late": float(V_late),
        "clip_pass": bool(clip_pass),
        "clipfrac_mean_late": float(clip_tail),
        "kl_pass": bool(kl_pass),
        "kl_per_dim_late": float(kl_tail),
        "fixed_eval_pass": bool(fixed_eval_pass),
        "fixed_eval_reason": str(fixed_eval_reason),
        "fixed_eval": fixed_eval_details,
        "pass": bool(passed),
    }
    if not bool(true10_pass):
        out["reason"] = "not_true_10d"
    elif not bool(fixed_eval_pass):
        out["reason"] = str(fixed_eval_reason or "fixed_eval_failed")
    elif not bool(passed):
        out["reason"] = "phaseB_failed"
    return out


def _get_plt():
    
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        try:
            matplotlib.use("Agg")
        except Exception:
            pass

    import matplotlib.pyplot as plt  

    return plt


def _apply_tcom_style(plt) -> None:
    """IEEE TCOM"""
    try:
        plt.rcParams.update(
            {
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
            }
        )
        
        try:
            from matplotlib import font_manager
            preferred_fonts = ["Times New Roman", "Times", "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Serif"]
            installed = {f.name for f in font_manager.fontManager.ttflist}
            picked = [name for name in preferred_fonts if name in installed]
            plt.rcParams["font.family"] = picked if picked else ["DejaVu Serif"]
        except Exception:
            plt.rcParams["font.family"] = ["DejaVu Serif"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


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


_DENSE_CKPT_KEYS: Tuple[str, ...] = (
    "train_reward_total_ep_dense",
    "train_cost_ep_dense",
    "train_vio_ep_dense",
    "train_improve_ep_dense",
    "train_shaping_ep_dense",
    "train_comp_ep_dense",
    "train_ris_ep_dense",
    "train_policy_loss_ep_dense",
    "train_value_loss_ep_dense",
    "train_entropy_ep_dense",
    "train_kl_ep_dense",
    "train_clipfrac_ep_dense",
    "train_explained_variance_ep_dense",
    "train_lr_ep_dense",
    "eval_det_paper_cost_ep_dense",
    "eval_det_paper_cost_ci_ep_dense",
    "eval_det_vio_any_frac_ep_dense",
    "eval_det_improve_ep_dense",
)
_DENSE_CKPT_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def _torch_load_ckpt_dict(torch_mod, ckpt_path: Path) -> Optional[Dict[str, Any]]:
    """ PyTorch  checkpoint  weights_only """
    try:
        obj = torch_mod.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch_mod.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict):
        return None
    return obj


def _episode_axis_is_dense(x: np.ndarray, *, min_unit_step_ratio: float = 0.85) -> bool:
    """Check whether episode axis is near one-episode-per-point."""
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    x0 = x0[np.isfinite(x0)]
    if x0.size < 3:
        return False
    x_i = np.asarray(np.rint(x0), dtype=np.int64).reshape(-1)
    d = np.diff(x_i)
    d = d[d > 0]
    if d.size <= 0:
        return False
    unit_ratio = float(np.mean(d == 1))
    return bool(unit_ratio >= float(min_unit_step_ratio))


def _collect_ckpt_candidates(run_dir: Path) -> List[Path]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return []

    cands: List[Path] = []
    
    for name in ("ckpt_final.pt", "final.pt", "ckpt_best.pt", "best.pt"):
        p = ckpt_dir / name
        if p.exists():
            cands.append(p)
    try:
        step_pts = list(ckpt_dir.glob("ckpt_step_*.pt")) + list(ckpt_dir.glob("step_*.pt"))
        step_pts = sorted(step_pts, key=lambda p: p.stat().st_mtime, reverse=True)
        cands.extend(step_pts[:8])
    except Exception:
        pass

    seen: set = set()
    uniq: List[Path] = []
    for p in cands:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def _load_dense_ckpt_bundle(run_dir: Path) -> Dict[str, np.ndarray]:
    """Load episode-dense history from checkpoint once per run_dir."""
    run_key = str(Path(run_dir).resolve())
    cached = _DENSE_CKPT_CACHE.get(run_key, None)
    if cached is not None:
        return cached

    out: Dict[str, np.ndarray] = {}
    ckpt_candidates = _collect_ckpt_candidates(Path(run_dir))
    if not ckpt_candidates:
        return out

    try:
        import torch
    except Exception:
        return out

    best_bundle: Dict[str, np.ndarray] = {}
    best_n = -1
    for ckpt_path in ckpt_candidates:
        try:
            obj = _torch_load_ckpt_dict(torch, ckpt_path)
            if obj is None:
                continue
            hist = obj.get("history", {}) or {}
            x_raw = np.asarray(hist.get("train_episodes_dense", []), dtype=np.int64).reshape(-1)
            if x_raw.size <= 0:
                continue
            m_x = np.isfinite(x_raw)
            x = np.asarray(x_raw[m_x], dtype=np.int64).reshape(-1)
            if x.size <= 0:
                continue

            bundle_now: Dict[str, np.ndarray] = {"episodes": x}
            for k in _DENSE_CKPT_KEYS:
                y_raw = np.asarray(hist.get(k, []), dtype=np.float64).reshape(-1)
                if y_raw.size != x_raw.size:
                    continue
                y = y_raw[m_x]
                if y.size != x.size:
                    continue
                if not np.isfinite(y).any():
                    continue
                bundle_now[k] = np.asarray(y, dtype=np.float64).reshape(-1)

            if int(x.size) > int(best_n) and len(bundle_now) > 1:
                best_n = int(x.size)
                best_bundle = bundle_now
        except Exception:
            continue

    out = best_bundle if best_bundle else {}
    
    if out:
        _DENSE_CKPT_CACHE[run_key] = out
    else:
        _DENSE_CKPT_CACHE.pop(run_key, None)
    return out


def _pick_metric1_dense_series(run_dir: Path) -> Tuple[np.ndarray, str]:
    """
    1
    1)  checkpoint history  train_cost_ep_dense
    2)  ckpt_final/finalrun 10%/10% episode ckpt_best/best
    3)  paper_cost_ep(SUM)
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    pref = [
        ckpt_dir / "ckpt_final.pt",
        ckpt_dir / "final.pt",
        ckpt_dir / "ckpt_best.pt",
        ckpt_dir / "best.pt",
    ]
    step_cands: List[Path] = []
    try:
        step_cands = sorted(list(ckpt_dir.glob("ckpt_step_*.pt")), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        step_cands = []
    cands = [p for p in pref if p.exists()] + step_cands[:4]
    if not cands:
        return np.asarray([], dtype=np.float64), "missing_checkpoint_history_dense/train_cost_ep_dense"

    try:
        import torch
    except Exception:
        return np.asarray([], dtype=np.float64), "missing_torch_for_checkpoint_history_dense"

    for p in cands:
        try:
            obj = _torch_load_ckpt_dict(torch, p)
            if obj is None:
                continue
            hist = obj.get("history", {}) or {}
            arr = np.asarray(hist.get("train_cost_ep_dense", []), dtype=np.float64).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 2:
                try:
                    rp = str(p.resolve())
                except Exception:
                    rp = str(p)
                return arr, f"checkpoint_history_dense/train_cost_ep_dense@{rp}"
        except Exception:
            continue
    return np.asarray([], dtype=np.float64), "missing_checkpoint_history_dense/train_cost_ep_dense"


def _pick_metric1_det_eval_series(metrics: Dict[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """
    1 Det-eval 
    """
    det = np.asarray(metrics.get("eval_det_paper_cost_mean", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
    if det.size <= 0:
        return np.asarray([], dtype=np.float64), "missing_train_metrics/eval_det_paper_cost_mean"
    mask = np.isfinite(det)
    det_valid = np.asarray(metrics.get("eval_det_is_valid", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
    if det_valid.size == det.size and np.isfinite(det_valid).any():
        mask = mask & (det_valid > 0.5)
    out = det[mask]
    out = out[np.isfinite(out)]
    if out.size < 2:
        return np.asarray([], dtype=np.float64), "too_few_valid_det_eval_points"
    return out, "train_metrics/eval_det_paper_cost_mean(valid_only)"


def _pick_metric1_fixed_eval_series(run_dir: Path) -> Tuple[np.ndarray, str]:
    """
    1fixed_eval_summary.csv  full 
    """
    eval_root = Path(run_dir) / "evals"
    cands: List[Path] = []
    try:
        cands = sorted(list(eval_root.glob("*/fixed_eval_summary*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        cands = []
    for p in cands:
        try:
            xs: List[float] = []
            ys: List[float] = []
            with p.open("r", encoding="utf-8", newline="") as f:
                rr = csv.DictReader(f)
                for row in rr:
                    if str(row.get("variant", "full")).strip().lower() != "full":
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
            if len(ys) < 2:
                continue
            order = np.argsort(np.asarray(xs, dtype=np.float64))
            yy = np.asarray(ys, dtype=np.float64)[order]
            yy = yy[np.isfinite(yy)]
            if yy.size >= 2:
                try:
                    rp = str(p.resolve())
                except Exception:
                    rp = str(p)
                return yy, f"fixed_eval_summary/full@{rp}"
        except Exception:
            continue
    return np.asarray([], dtype=np.float64), "missing_fixed_eval_summary/full"


def _metric1_window_drop(
    y: np.ndarray,
    *,
    window_frac: float = 0.10,
    min_window: int = 5,
) -> Dict[str, float]:
    """
    1
    
    """
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: Dict[str, float] = {
        "n_points": float(n),
        "window_frac": float(window_frac),
        "window_size": float(0),
        "paper_cost_first_5pct_mean": float("nan"),
        "paper_cost_last_5pct_mean": float("nan"),
        "paper_cost_first_10pct_mean": float("nan"),
        "paper_cost_last_10pct_mean": float("nan"),
        "paper_cost_first_window_mean": float("nan"),
        "paper_cost_last_window_mean": float("nan"),
        "paper_cost_first_point": float("nan"),
        "paper_cost_last_point": float("nan"),
        "relative_drop_pct": float("nan"),
    }
    if n < 2:
        return out

    w = int(max(int(min_window), round(float(window_frac) * float(n))))
    w = max(1, min(w, max(1, n // 2)))
    first_w = np.asarray(arr[:w], dtype=np.float64)
    last_w = np.asarray(arr[-w:], dtype=np.float64)
    first_mean = float(np.nanmean(first_w))
    last_mean = float(np.nanmean(last_w))
    denom = float(max(abs(first_mean), 1e-9))
    drop = float((first_mean - last_mean) / denom * 100.0)
    out.update(
        {
            "window_size": float(w),
            "paper_cost_first_5pct_mean": first_mean,
            "paper_cost_last_5pct_mean": last_mean,
            "paper_cost_first_10pct_mean": first_mean,
            "paper_cost_last_10pct_mean": last_mean,
            "paper_cost_first_window_mean": first_mean,
            "paper_cost_last_window_mean": last_mean,
            "paper_cost_first_point": float(arr[0]),
            "paper_cost_last_point": float(arr[-1]),
            "relative_drop_pct": drop,
        }
    )
    return out


def _metric1_trend_diag(
    y: np.ndarray,
    *,
    window_frac: float = 0.10,
    min_window: int = 5,
    max_slope_points: int = 128,
) -> Dict[str, float]:
    """
    1
    -  Theil-Sen EMA
    - 
    """
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: Dict[str, float] = {
        "n_points": float(n),
        "trend_slope_theil_sen_ema": float("nan"),
        "trend_intercept_ema": float("nan"),
        "ema_span_used": float("nan"),
        "late_rebound_pct": float("nan"),
        "late_mean": float("nan"),
        "trough_after_mid": float("nan"),
    }
    if n < 3:
        return out

    span = int(max(8, min(96, round(n * 0.08))))
    y_ema = np.asarray(ema_smooth(arr, span=span, adjust=False), dtype=np.float64).reshape(-1)
    if y_ema.size != n:
        y_ema = np.asarray(arr, dtype=np.float64)

    k = int(min(max_slope_points, n))
    idx = np.asarray(np.linspace(0, n - 1, k, dtype=np.int64), dtype=np.int64)
    idx = np.unique(np.clip(idx, 0, n - 1))
    xs = idx.astype(np.float64)
    ys = np.asarray(y_ema[idx], dtype=np.float64)
    if xs.size >= 2 and np.isfinite(ys).any():
        dx = xs[None, :] - xs[:, None]
        dy = ys[None, :] - ys[:, None]
        m = (dx > 0) & np.isfinite(dx) & np.isfinite(dy)
        if np.any(m):
            slopes = np.asarray(dy[m] / dx[m], dtype=np.float64).reshape(-1)
            slope = float(np.nanmedian(slopes)) if slopes.size > 0 else float("nan")
        else:
            slope = float("nan")
        if np.isfinite(slope):
            intercept = float(np.nanmedian(ys - slope * xs))
        else:
            intercept = float("nan")
    else:
        slope = float("nan")
        intercept = float("nan")

    w = int(max(int(min_window), round(float(window_frac) * float(n))))
    w = max(1, min(w, max(1, n // 2)))
    i_mid = int(min(n - 1, max(1, round(0.30 * n))))
    trough = float(np.nanmin(arr[i_mid:])) if i_mid < n else float(np.nanmin(arr))
    late_mean = float(np.nanmean(arr[-w:]))
    late_rebound = float((late_mean - trough) / max(abs(trough), 1e-9) * 100.0)

    out.update(
        {
            "trend_slope_theil_sen_ema": slope,
            "trend_intercept_ema": intercept,
            "ema_span_used": float(span),
            "late_rebound_pct": late_rebound,
            "late_mean": late_mean,
            "trough_after_mid": trough,
        }
    )
    return out


def _replace_sparse_with_dense_ckpt_series(
    run_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    dense_key: str,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Replace sparse rollout-level series with true episode-dense data from checkpoint history.
    """
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    y0 = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(min(x0.size, y0.size))
    if n <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), False
    x0 = x0[:n]
    y0 = y0[:n]

    if _episode_axis_is_dense(x0):
        return np.asarray(np.rint(x0), dtype=np.int64), np.asarray(y0, dtype=np.float64), False

    bundle = _load_dense_ckpt_bundle(Path(run_dir))
    x_dense = np.asarray(bundle.get("episodes", np.asarray([], dtype=np.int64)), dtype=np.int64).reshape(-1)
    y_dense = np.asarray(bundle.get(str(dense_key), np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
    if x_dense.size <= 0 or y_dense.size != x_dense.size:
        return np.asarray(np.rint(x0), dtype=np.int64), np.asarray(y0, dtype=np.float64), False

    x_valid = x0[np.isfinite(x0)]
    if x_valid.size > 0:
        hi = int(np.ceil(np.nanmax(x_valid)))
        m_range = x_dense <= hi
        if np.any(m_range):
            x_dense = x_dense[m_range]
            y_dense = y_dense[m_range]

    m = np.isfinite(x_dense) & np.isfinite(y_dense)
    if not np.any(m):
        return np.asarray(np.rint(x0), dtype=np.int64), np.asarray(y0, dtype=np.float64), False
    return np.asarray(x_dense[m], dtype=np.int64), np.asarray(y_dense[m], dtype=np.float64), True


def _plot_raw_and_ema(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    ema_color: Optional[str] = None,
    label_raw: str = "raw",
    span_cfg: int = 24,
    label_ema_prefix: str = "EMA",
    tcom_adaptive_span: bool = False,
    raw_alpha: float = 0.32,
    ema_alpha: float = 0.95,
    raw_linewidth: float = 1.2,
    ema_linewidth: float = 3.0,
    raw_stride: int = 1,
    y_for_ema: Optional[np.ndarray] = None,
) -> Optional[int]:
    """
     raw + EMA-only 

    - EMA
    - raw marker
    - EMA
    """
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    y0 = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(min(x0.size, y0.size))
    x0 = x0[:n]
    y0 = y0[:n]

    mask = np.isfinite(x0) & np.isfinite(y0)
    x0 = x0[mask]
    y0 = y0[mask]
    if x0.size <= 0 or y0.size <= 0:
        return None

    
    raw_stride_eff = int(max(1, int(raw_stride)))
    x_raw = x0[::raw_stride_eff]
    y_raw = y0[::raw_stride_eff]
    ax.plot(
        x_raw,
        y_raw,
        color=color,
        linewidth=float(raw_linewidth),
        alpha=float(raw_alpha),
        label=str(label_raw),
        zorder=2,
    )

    if y0.size < 3:
        ax.text(
            0.02,
            0.96,
            "too few points",
            transform=ax.transAxes,
            fontsize=9,
            color="0.35",
            ha="left",
            va="top",
        )
        return None

    
    span_base = int(max(3, int(span_cfg)))
    span_eff = int(span_base)
    if bool(tcom_adaptive_span):
        n_pts = int(y0.size)
        span_auto = int(round(0.08 * float(n_pts)))
        span_eff = int(max(span_base, min(96, span_auto)))
    span_eff = int(max(3, span_eff))
    y_ema_src = y0
    if y_for_ema is not None:
        y_alt = np.asarray(y_for_ema, dtype=np.float64).reshape(-1)
        if y_alt.size == y0.size:
            y_ema_src = y_alt
    y_ema = ema_smooth(y_ema_src, span=span_eff, adjust=False)
    c_ema = str(ema_color) if ema_color is not None else str(color)
    ax.plot(
        x0,
        y_ema,
        color=c_ema,
        linewidth=float(ema_linewidth),
        alpha=float(ema_alpha),
        label=f"{label_ema_prefix}(span={int(span_eff)})",
        zorder=4,
    )
    return int(span_eff)


def _robust_reward_display_series(
    y: np.ndarray,
    *,
    hampel_window: int = 9,
    hampel_nsigma: float = 3.5,
    clip_quantile: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    
    1) Hampel 
    2)  winsorize 
    """
    y0 = np.asarray(y, dtype=np.float64).reshape(-1)
    out = np.asarray(y0, dtype=np.float64).copy()
    n = int(out.size)
    meta: Dict[str, float] = {
        "n_points": float(n),
        "hampel_replaced": 0.0,
        "winsor_clipped": 0.0,
        "clip_quantile": float(max(0.0, min(0.20, float(clip_quantile)))),
    }
    if n <= 3:
        return out, meta

    
    win = int(max(3, int(hampel_window)))
    if win % 2 == 0:
        win += 1
    half = win // 2
    ns = float(max(1.0, float(hampel_nsigma)))
    replaced = 0
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        seg = out[l:r]
        seg = seg[np.isfinite(seg)]
        if seg.size < 3:
            continue
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med)))
        if not np.isfinite(mad) or mad <= 1e-12:
            continue
        sigma = 1.4826 * mad
        yi = float(out[i])
        if np.isfinite(yi) and abs(yi - med) > ns * sigma:
            out[i] = med
            replaced += 1
    meta["hampel_replaced"] = float(replaced)

    q = float(max(0.0, min(0.20, float(clip_quantile))))
    if q > 0.0:
        fin = out[np.isfinite(out)]
        if fin.size >= 10:
            lo, hi = np.quantile(fin, [q, 1.0 - q])
            before = np.asarray(out, dtype=np.float64).copy()
            out = np.clip(out, float(lo), float(hi))
            clipped = int(np.sum(np.isfinite(before) & np.isfinite(out) & (np.abs(before - out) > 1e-12)))
            meta["winsor_clipped"] = float(clipped)

    return out, meta


def _expand_rollout_series_to_episode(
    x: np.ndarray,
    y: np.ndarray,
    *,
    keep_non_finite_y: bool = False,
    mode: str = "hold",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand sparse rollout-level points to a strict 1-episode grid.

    
    -  episode hold
    -  `mode=\"linear\"`
    """
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    y0 = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(min(x0.size, y0.size))
    if n <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    x0 = x0[:n]
    y0 = y0[:n]
    if bool(keep_non_finite_y):
        m = np.isfinite(x0)
    else:
        m = np.isfinite(x0) & np.isfinite(y0)
    if not np.any(m):
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    x_i = np.asarray(np.rint(x0[m]), dtype=np.int64).reshape(-1)
    y_i = np.asarray(y0[m], dtype=np.float64).reshape(-1)
    if x_i.size <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    m_pos = x_i > 0
    if not np.any(m_pos):
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    x_i = x_i[m_pos]
    y_i = y_i[m_pos]

    order = np.argsort(x_i, kind="stable")
    x_i = np.asarray(x_i[order], dtype=np.int64).reshape(-1)
    y_i = np.asarray(y_i[order], dtype=np.float64).reshape(-1)

    x_u_list: List[int] = []
    y_u_list: List[float] = []
    for ep_now, val_now in zip(x_i.tolist(), y_i.tolist()):
        ep_int = int(ep_now)
        val_f = float(val_now)
        if x_u_list and ep_int == x_u_list[-1]:
            y_u_list[-1] = val_f
        else:
            x_u_list.append(ep_int)
            y_u_list.append(val_f)
    if not x_u_list:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    x_u = np.asarray(x_u_list, dtype=np.int64).reshape(-1)
    y_u = np.asarray(y_u_list, dtype=np.float64).reshape(-1)
    x_max = int(max(1, int(x_u[-1])))
    out_x = np.arange(1, x_max + 1, dtype=np.int64)

    mode_n = str(mode or "hold").strip().lower()
    if mode_n not in ("linear", "hold"):
        mode_n = "linear"

    if mode_n == "hold":
        idx = np.searchsorted(x_u, out_x, side="right") - 1
        idx = np.clip(idx, 0, x_u.size - 1)
        out_y = np.asarray(y_u[idx], dtype=np.float64).reshape(-1)
        return np.asarray(out_x, dtype=np.int64), np.asarray(out_y, dtype=np.float64)

    m_fin = np.isfinite(y_u)
    if not np.any(m_fin):
        if bool(keep_non_finite_y):
            return np.asarray(out_x, dtype=np.int64), np.full(out_x.shape, np.nan, dtype=np.float64)
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    x_fit = np.asarray(x_u[m_fin], dtype=np.float64).reshape(-1)
    y_fit = np.asarray(y_u[m_fin], dtype=np.float64).reshape(-1)
    if x_fit.size == 1:
        out_y = np.full(out_x.shape, float(y_fit[0]), dtype=np.float64)
    else:
        out_y = np.interp(
            np.asarray(out_x, dtype=np.float64),
            x_fit,
            y_fit,
            left=float(y_fit[0]),
            right=float(y_fit[-1]),
        )

    if bool(keep_non_finite_y) and np.any(~m_fin):
        x_nan = np.asarray(x_u[~m_fin], dtype=np.int64).reshape(-1)
        x_nan = x_nan[(x_nan >= 1) & (x_nan <= x_max)]
        if x_nan.size > 0:
            out_y[np.asarray(x_nan - 1, dtype=np.int64)] = np.nan

    return np.asarray(out_x, dtype=np.int64), np.asarray(out_y, dtype=np.float64)


def plot_convergence_total_rewards_paper_return(
    run_dir: Path,
    eval_train_episodes: List[int],
    paper_cost_mean: List[float],
    paper_cost_ci: Optional[List[float]] = None,
    out_dir: Optional[Path] = None,
    baseline_paper_cost_mean: Optional[List[float]] = None,
    draw_ci: bool = True,
    ema_span: int = 48,
    save_name: str = "Convergence_RewardTotal.png",
) -> None:
    """
    PhaseEP1.5Total rewards (Paper-return)
    
    -  `paper_cost_mean`  fixed-eval det-eval 
    - paper_return_ep := -paper_cost_ep
    """
    plt = _get_plt()
    _apply_tcom_style(plt)

    if (not eval_train_episodes) or (not paper_cost_mean):
        return

    x = np.asarray(eval_train_episodes, dtype=np.int64).reshape(-1)
    c = np.asarray(paper_cost_mean, dtype=np.float64).reshape(-1)
    if x.size == 0 or c.size == 0:
        return
    n = int(min(x.size, c.size))
    x = x[:n]
    c = c[:n]

    y = -c
    x, y = _expand_rollout_series_to_episode(x, y, keep_non_finite_y=True)
    if x.size <= 0 or y.size <= 0:
        return
    
    x_f, y_f, keep = filter_valid_xy(x, y, valid_mask=None, invalid_values=())
    if x_f.size <= 0 or y_f.size <= 0:
        return

    
    
    
    y_ci_f = None
    if draw_ci and paper_cost_ci is not None:
        ci0 = np.asarray(paper_cost_ci, dtype=np.float64).reshape(-1)[:n]
        ci0 = np.resize(ci0, (n,))
        _x_ci, ci0 = _expand_rollout_series_to_episode(
            np.asarray(eval_train_episodes, dtype=np.int64).reshape(-1)[:n],
            ci0,
            keep_non_finite_y=True,
        )
        if ci0.size != x.size:
            ci0 = np.resize(ci0, (x.size,))
        ci_f = ci0[keep]
        if ci_f.size == y_f.size and np.isfinite(ci_f).any():
            y_ci_f = ci_f

    fig_dir = out_dir if out_dir is not None else (run_dir / "figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.4))
    
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.35)
    
    span_eff = _plot_raw_and_ema(
        ax,
        x_f,
        y_f,
        color="#0072BD",
        label_raw="raw",
        span_cfg=int(ema_span),
        label_ema_prefix="EMA",
        tcom_adaptive_span=False,
    )
    
    if y_ci_f is not None:
        ax.fill_between(x_f, y_f - y_ci_f, y_f + y_ci_f, color="#0072BD", alpha=0.12, linewidth=0, label="95% CI", zorder=1)

    ax.set_title("Total rewards (Paper-return)", pad=8, fontweight="bold")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Total rewards")
    _apply_episode_xaxis_sci(ax)

    
    baseline_return = None
    try:
        if baseline_paper_cost_mean is not None:
            bc = np.asarray(baseline_paper_cost_mean, dtype=np.float64).reshape(-1)
            if bc.size > 0 and np.isfinite(bc).any():
                baseline_return = float(-np.nanmean(bc))
    except Exception:
        baseline_return = None

    if baseline_return is not None and np.isfinite(baseline_return):
        ax.axhline(baseline_return, color="0.35", linestyle="--", linewidth=1.2, label="baseline_return", zorder=0)

    ax.legend(loc="lower right", ncol=1)

    base_txt = f"{baseline_return:.4f}" if (baseline_return is not None and np.isfinite(baseline_return)) else "N/A"
    span_txt = f"{int(span_eff)}" if span_eff is not None else "N/A"
    ax.text(
        0.01,
        0.01,
        f"paper_return_ep = -paper_cost_ep\nEMA(span={span_txt})\nbaseline_return: {base_txt}",
        transform=ax.transAxes,
        fontsize=9,
        color="0.25",
        ha="left",
        va="bottom",
    )

    save_path = fig_dir / str(save_name)
    fig.savefig(str(save_path), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[HB][plot] Saved PhaseE main convergence plot to {save_path}", flush=True)


def plot_convergence_reward_total(
    run_dir: Path,
    train_episodes: List[int],
    reward_total_mean: List[float],
    reward_total_ci: Optional[List[float]] = None,
    *,
    out_dir: Optional[Path] = None,
    draw_ci: bool = True,
    ema_span: int = 48,
    save_name: str = "Convergence_RewardTotal.png",
    note: Optional[str] = None,
    robust_display: bool = False,
    robust_clip_quantile: float = 0.01,
    robust_hampel_window: int = 9,
    robust_hampel_nsigma: float = 3.5,
    raw_stride: int = 1,
    episode_length: int = 80,
) -> None:
    """
    PhaseGP1.8training rollout  env  reward_total_ep shaping
     episode_length > 1 y  Average Episode Rewardepisode_length
    """
    plt = _get_plt()
    _apply_tcom_style(plt)

    if (not train_episodes) or (not reward_total_mean):
        return

    x = np.asarray(train_episodes, dtype=np.int64).reshape(-1)
    y = np.asarray(reward_total_mean, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return

    n = int(min(x.size, y.size))
    x = x[:n]
    y = y[:n]
    x_dense, y_dense, used_dense = _replace_sparse_with_dense_ckpt_series(
        run_dir,
        x,
        y,
        dense_key="train_reward_total_ep_dense",
    )
    if bool(used_dense) and x_dense.size > 0 and y_dense.size > 0:
        x = x_dense
        y = y_dense
        print(
            f"[HB][plot] Convergence_RewardTotal uses checkpoint dense episode series: n={int(x.size)}",
            flush=True,
        )
    x, y = _expand_rollout_series_to_episode(x, y, keep_non_finite_y=True)
    if x.size <= 0 or y.size <= 0:
        return

    
    _ep_len = int(max(1, episode_length))
    if _ep_len > 1:
        y = y / float(_ep_len)

    
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return
    x_f = x[m]
    y_f = y[m]
    
    y_plot = np.asarray(y_f, dtype=np.float64).copy()
    robust_meta: Dict[str, float] = {}
    if bool(robust_display):
        y_plot, robust_meta = _robust_reward_display_series(
            y_plot,
            hampel_window=int(robust_hampel_window),
            hampel_nsigma=float(robust_hampel_nsigma),
            clip_quantile=float(robust_clip_quantile),
        )

    y_ci_f = None
    if draw_ci and reward_total_ci is not None:
        ci0 = np.asarray(reward_total_ci, dtype=np.float64).reshape(-1)[:n]
        ci0 = np.resize(ci0, (n,))
        _x_ci, ci0 = _expand_rollout_series_to_episode(
            np.asarray(train_episodes, dtype=np.int64).reshape(-1)[:n],
            ci0,
            keep_non_finite_y=True,
        )
        if ci0.size != x.size:
            ci0 = np.resize(ci0, (x.size,))
        ci_f = ci0[m]
        if ci_f.size == y_f.size and np.isfinite(ci_f).any():
            y_ci_f = ci_f

    fig_dir = out_dir if out_dir is not None else (run_dir / "figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.8))
    fig.patch.set_facecolor("white")
    
    ax.set_facecolor("white")
    ax.grid(True, color="#D5E2E6", linestyle="-", linewidth=0.7, alpha=0.68)
    for sp in ax.spines.values():
        sp.set_color("#4D6A70")
        sp.set_linewidth(0.9)

    
    n_plot = int(x_f.size)
    i_early = int(max(1, round(0.30 * n_plot)))
    i_mid = int(max(i_early + 1, round(0.75 * n_plot)))
    i_mid = min(i_mid, n_plot - 1)
    if n_plot >= 3:
        ax.axvspan(float(x_f[0]), float(x_f[i_early - 1]), color="#EAF5F7", alpha=0.0, zorder=0)
        ax.axvspan(float(x_f[i_early]), float(x_f[i_mid - 1]), color="#E2F0F3", alpha=0.0, zorder=0)
        ax.axvspan(float(x_f[i_mid]), float(x_f[-1]), color="#DAEBEF", alpha=0.0, zorder=0)

    
    _raw_stride_vis = int(max(1, int(raw_stride)))
    
    _raw_alpha_vis = 0.32
    if n_plot > 1200:
        
        _raw_stride_vis = int(max(_raw_stride_vis, int(np.ceil(float(n_plot) / 1200.0))))
        _raw_alpha_vis = float(max(0.11, 0.32 * min(1.0, 1200.0 / float(max(n_plot, 1)))))

    span_eff = _plot_raw_and_ema(
        ax,
        x_f,
        y_plot,
        color=REWARD_TOTAL_RAW_COLOR,
        ema_color=REWARD_TOTAL_EMA_COLOR,
        label_raw="raw",
        span_cfg=int(ema_span),
        label_ema_prefix="EMA",
        tcom_adaptive_span=False,
        raw_alpha=float(_raw_alpha_vis),
        ema_alpha=0.92,
        raw_linewidth=1.15,
        ema_linewidth=2.6,
        raw_stride=int(_raw_stride_vis),
        y_for_ema=y_plot,
    )
    if y_ci_f is not None:
        ax.fill_between(
            x_f,
            y_f - y_ci_f,
            y_f + y_ci_f,
            color=REWARD_TOTAL_RAW_COLOR,
            alpha=0.08,
            linewidth=0,
            label="95% CI",
            zorder=1,
        )

    
    y_late = y_f[i_mid:] if i_mid < n_plot else y_f
    y_late_mean = float(np.nanmean(y_late)) if y_late.size > 0 else float("nan")

    _y_label = "Average Episode Reward" if _ep_len > 1 else "Total rewards"
    _title_suffix = f" (avg over {_ep_len} steps)" if _ep_len > 1 else " (reward_total)"
    ax.set_title(f"Training Rollout: Average Episode Reward{_title_suffix}", pad=8, fontweight="bold")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel(_y_label)
    _apply_episode_xaxis_sci(ax)
    ax.legend(loc="lower right", ncol=1, frameon=False)

    span_txt = f"{int(span_eff)}" if span_eff is not None else "N/A"
    note_txt = (note or "").strip()
    if bool(robust_display):
        note_txt = (
            f"{note_txt}\nrobust-display: hampel(w={int(max(3, robust_hampel_window))}, nsigma={float(robust_hampel_nsigma):.1f}), "
            f"winsor_q={float(max(0.0, min(0.20, robust_clip_quantile))):.3f}, raw_stride={int(max(1, raw_stride))}"
        ).strip()
    
    if (not note_txt) or any(ord(ch) > 127 for ch in note_txt) or ("infinf" in note_txt):
        note_txt = "reward_total_ep (1 episode per point)"
    y_early_mean = float(np.nanmean(y_f[:i_early])) if i_early > 0 else float("nan")
    y_mid_mean = float(np.nanmean(y_f[i_early:i_mid])) if i_mid > i_early else float("nan")
    ax.text(
        0.01,
        0.01,
        f"{note_txt}\nEMA(span={span_txt})\nearly={y_early_mean:.3f}, mid={y_mid_mean:.3f}, late={y_late_mean:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        color="#33545B",
        ha="left",
        va="bottom",
    )

    save_path = fig_dir / str(save_name)
    fig.savefig(str(save_path), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[HB][plot] Saved PhaseG main convergence plot to {save_path}", flush=True)


def plot_convergence_reward_total_ema_ci_only(
    run_dir: Path,
    train_episodes: List[int],
    reward_total_mean: List[float],
    *,
    out_dir: Optional[Path] = None,
    ema_span: int = 96,
    ci_window: int = 96,
    ci_alpha: float = 0.12,
    save_name: str = "Convergence_RewardTotal_span96_ci_only.png",
    note: Optional[str] = None,
    episode_length: int = 80,
) -> None:
    """
    Convergence_RewardTotal  EMA + rolling CI raw
     episode_length > 1 y  Average Episode Rewardepisode_length
    """
    plt = _get_plt()
    _apply_tcom_style(plt)

    if (not train_episodes) or (not reward_total_mean):
        return

    x = np.asarray(train_episodes, dtype=np.int64).reshape(-1)
    y = np.asarray(reward_total_mean, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return

    n = int(min(x.size, y.size))
    x = x[:n]
    y = y[:n]
    x_dense, y_dense, used_dense = _replace_sparse_with_dense_ckpt_series(
        run_dir,
        x,
        y,
        dense_key="train_reward_total_ep_dense",
    )
    if bool(used_dense) and x_dense.size > 0 and y_dense.size > 0:
        x = x_dense
        y = y_dense
        print(
            f"[HB][plot] Convergence_RewardTotal EMA+CI uses checkpoint dense episode series: n={int(x.size)}",
            flush=True,
        )
    x, y = _expand_rollout_series_to_episode(x, y, keep_non_finite_y=True)
    if x.size <= 0 or y.size <= 0:
        return

    
    _ep_len2 = int(max(1, episode_length))
    if _ep_len2 > 1:
        y = y / float(_ep_len2)

    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return
    x_f = np.asarray(x[m], dtype=np.float64).reshape(-1)
    y_f = np.asarray(y[m], dtype=np.float64).reshape(-1)
    if x_f.size <= 2:
        return

    fig_dir = out_dir if out_dir is not None else (run_dir / "figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.8))
    fig.patch.set_facecolor("white")
    
    ax.set_facecolor("white")
    ax.grid(True, color="#D5E2E6", linestyle="-", linewidth=0.7, alpha=0.68)
    for sp in ax.spines.values():
        sp.set_color("#4D6A70")
        sp.set_linewidth(0.9)

    n_plot = int(x_f.size)
    i_early = int(max(1, round(0.30 * n_plot)))
    i_mid = int(max(i_early + 1, round(0.75 * n_plot)))
    i_mid = min(i_mid, n_plot - 1)
    if n_plot >= 3:
        ax.axvspan(float(x_f[0]), float(x_f[i_early - 1]), color="#EAF5F7", alpha=0.0, zorder=0)
        ax.axvspan(float(x_f[i_early]), float(x_f[i_mid - 1]), color="#E2F0F3", alpha=0.0, zorder=0)
        ax.axvspan(float(x_f[i_mid]), float(x_f[-1]), color="#DAEBEF", alpha=0.0, zorder=0)
    span_eff = int(max(3, int(ema_span)))
    y_ema = ema_smooth(y_f, span=span_eff, adjust=False)

    win = int(max(10, int(ci_window)))
    lo, hi = compute_confidence_band(y_f, window=win, confidence=0.95)
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    if lo.size != y_f.size:
        lo = np.resize(lo, (y_f.size,))
    if hi.size != y_f.size:
        hi = np.resize(hi, (y_f.size,))
    m_ci = np.isfinite(lo) & np.isfinite(hi)
    if np.any(m_ci):
        ax.fill_between(
            x_f[m_ci],
            lo[m_ci],
            hi[m_ci],
            color=REWARD_TOTAL_RAW_COLOR,
            alpha=float(max(0.02, min(0.60, ci_alpha))),
            linewidth=0,
            label="95% CI (rolling)",
            zorder=2,
        )

    ax.plot(
        x_f,
        y_ema,
        color=REWARD_TOTAL_EMA_COLOR,
        linewidth=2.8,
        alpha=0.96,
        label=f"EMA(span={int(span_eff)})",
        zorder=4,
    )

    _y_label2 = "Average Episode Reward" if _ep_len2 > 1 else "Total rewards"
    _title_suffix2 = f" (avg over {_ep_len2} steps)" if _ep_len2 > 1 else " (reward_total)"
    ax.set_title(f"Training Rollout: Average Episode Reward{_title_suffix2} - EMA+CI", pad=8, fontweight="bold")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel(_y_label2)
    _apply_episode_xaxis_sci(ax)
    ax.legend(loc="lower right", ncol=1, frameon=False)

    y_early_mean = float(np.nanmean(y_ema[:i_early])) if i_early > 0 else float("nan")
    y_mid_mean = float(np.nanmean(y_ema[i_early:i_mid])) if i_mid > i_early else float("nan")
    y_late_mean = float(np.nanmean(y_ema[i_mid:])) if i_mid < n_plot else float(np.nanmean(y_ema))

    note_txt = (note or "").strip()
    if (not note_txt) or any(ord(ch) > 127 for ch in note_txt) or ("infinf" in note_txt):
        note_txt = "EMA+CI only (no raw, no shift)"
    ax.text(
        0.01,
        0.01,
        (
            f"{note_txt}\nEMA(span={int(span_eff)}), CI=rolling95% (alpha={float(ci_alpha):.2f})\n"
            f"early={y_early_mean:.3f}, mid={y_mid_mean:.3f}, late={y_late_mean:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#33545B",
        ha="left",
        va="bottom",
    )

    save_path = fig_dir / str(save_name)
    fig.savefig(str(save_path), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[HB][plot] Saved Convergence_RewardTotal EMA+CI preview to {save_path}", flush=True)


def plot_training_curves(
    run_dir: Path,
    reward_history: List[float],
    policy_loss_history: List[float],
    value_loss_history: List[float],
    entropy_history: List[float],
    kl_div_history: List[float],
    train_episode_history: List[int],
    reward_episode_history: Optional[List[int]] = None,
    eval_train_episode_history: Optional[List[int]] = None,
    eval_reward_history: Optional[List[float]] = None,
    ema_span: int = 48,
) -> None:
    """=Training Episodesraw+EMA"""
    plt = _get_plt()

    fig_dir = run_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x_metrics = np.asarray(train_episode_history, dtype=np.int64).reshape(-1)
    x_reward = np.asarray(
        reward_episode_history if reward_episode_history is not None else train_episode_history,
        dtype=np.int64,
    ).reshape(-1)
    rewards = np.asarray(reward_history, dtype=np.float64).reshape(-1)
    policy_losses = np.asarray(policy_loss_history, dtype=np.float64).reshape(-1)
    value_losses = np.asarray(value_loss_history, dtype=np.float64).reshape(-1)
    entropies = np.asarray(entropy_history, dtype=np.float64).reshape(-1)
    kl_divs = np.asarray(kl_div_history, dtype=np.float64).reshape(-1)

    n_reward = int(min(x_reward.size, rewards.size))
    if n_reward <= 0:
        return
    x_reward = x_reward[:n_reward]
    rewards = rewards[:n_reward]
    x_reward_dense, rewards_dense, used_dense_reward = _replace_sparse_with_dense_ckpt_series(
        run_dir,
        x_reward,
        rewards,
        dense_key="train_reward_total_ep_dense",
    )
    if bool(used_dense_reward) and x_reward_dense.size > 0 and rewards_dense.size > 0:
        x_reward = x_reward_dense
        rewards = rewards_dense
        print(
            f"[HB][plot] training_curves reward pane uses checkpoint dense episode series: n={int(x_reward.size)}",
            flush=True,
        )
    x_reward, rewards = _expand_rollout_series_to_episode(
        x_reward,
        rewards,
        keep_non_finite_y=True,
        mode="hold",
    )
    if x_reward.size <= 0 or rewards.size <= 0:
        return

    def _series_with_metric_x(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(min(x_metrics.size, y.size))
        if n <= 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
        xs = np.asarray(x_metrics[:n], dtype=np.int64).reshape(-1)
        ys = np.asarray(y[:n], dtype=np.float64).reshape(-1)
        m = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(m):
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
        return np.asarray(xs[m], dtype=np.int64), np.asarray(ys[m], dtype=np.float64)

    def _metric_series_dense(y: np.ndarray, dense_key: str, pane_name: str) -> Tuple[np.ndarray, np.ndarray]:
        x0, y0 = _series_with_metric_x(y)
        if x0.size <= 0 or y0.size <= 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
        x_dense, y_dense, used_dense = _replace_sparse_with_dense_ckpt_series(
            run_dir,
            x0,
            y0,
            dense_key=dense_key,
        )
        if bool(used_dense) and x_dense.size > 0 and y_dense.size > 0:
            print(
                f"[HB][plot] training_curves {pane_name} pane uses checkpoint dense episode series: n={int(x_dense.size)}",
                flush=True,
            )
            return np.asarray(x_dense, dtype=np.int64), np.asarray(y_dense, dtype=np.float64)
        
        return _expand_rollout_series_to_episode(x0, y0, mode="hold")

    x_policy, y_policy = _metric_series_dense(policy_losses, "train_policy_loss_ep_dense", "policy_loss")
    x_value, y_value = _metric_series_dense(value_losses, "train_value_loss_ep_dense", "value_loss")
    x_entropy, y_entropy = _metric_series_dense(entropies, "train_entropy_ep_dense", "entropy")
    x_kl, y_kl = _metric_series_dense(kl_divs, "train_kl_ep_dense", "kl")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ARIA Training Curves", fontsize=16, fontweight="bold")

    def _finish(ax, title: str, ylabel: str) -> None:
        ax.set_xlabel("Training Episodes", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        _apply_episode_xaxis_sci(ax)
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(loc="best", fontsize=9)

    
    ax = axes[0, 0]
    if rewards.size >= 10:
        try:
            window = max(10, int(rewards.size * 0.05))
            lo, hi = compute_confidence_band(rewards, window=window, confidence=0.95)
            if lo.size == rewards.size and hi.size == rewards.size:
                ax.fill_between(
                    x_reward,
                    lo,
                    hi,
                    alpha=0.12,
                    color=REWARD_TOTAL_RAW_COLOR,
                    label="Rolling Band (95%)",
                    zorder=1,
                    linewidth=0,
                )
        except Exception:
            pass
    _plot_raw_and_ema(
        ax,
        x_reward,
        rewards,
        color=REWARD_TOTAL_RAW_COLOR,
        ema_color=REWARD_TOTAL_EMA_COLOR,
        label_raw="raw",
        span_cfg=int(ema_span),
        label_ema_prefix="EMA",
        tcom_adaptive_span=False,
    )
    _finish(ax, "Training Rollout: Total rewards (reward_total)", "Total rewards")

    # 2) policy loss
    ax = axes[0, 1]
    if x_policy.size > 0 and y_policy.size > 0:
        _plot_raw_and_ema(
            ax,
            x_policy,
            y_policy,
            color="#d62728",
            label_raw="raw",
            span_cfg=int(ema_span),
            label_ema_prefix="EMA",
            tcom_adaptive_span=False,
        )
    _finish(ax, "Policy Loss", "Policy Loss")

    # 3) value loss
    ax = axes[0, 2]
    if x_value.size > 0 and y_value.size > 0:
        _plot_raw_and_ema(
            ax,
            x_value,
            y_value,
            color="#2ca02c",
            label_raw="raw",
            span_cfg=int(ema_span),
            label_ema_prefix="EMA",
            tcom_adaptive_span=False,
        )
    _finish(ax, "Value Loss", "Value Loss")

    # 4) entropy
    ax = axes[1, 0]
    if x_entropy.size > 0 and y_entropy.size > 0:
        _plot_raw_and_ema(
            ax,
            x_entropy,
            y_entropy,
            color="#9467bd",
            label_raw="raw",
            span_cfg=int(ema_span),
            label_ema_prefix="EMA",
            tcom_adaptive_span=False,
        )
    _finish(ax, "Policy Entropy", "Entropy")

    # 5) KL divergence
    ax = axes[1, 1]
    if x_kl.size > 0 and y_kl.size > 0:
        _plot_raw_and_ema(
            ax,
            x_kl,
            y_kl,
            color="#17becf",
            label_raw="raw",
            span_cfg=int(ema_span),
            label_ema_prefix="EMA",
            tcom_adaptive_span=False,
        )
    _finish(ax, "KL Divergence", "KL Divergence")

    
    ax = axes[1, 2]
    _plot_raw_and_ema(
        ax,
        x_reward,
        rewards,
        color="#1f77b4",
        label_raw="raw",
        span_cfg=int(ema_span),
        label_ema_prefix="EMA",
        tcom_adaptive_span=False,
    )
    try:
        from src.utils.plot_smoothing import savitzky_golay_smooth

        sg = savitzky_golay_smooth(rewards)
        sg = np.asarray(sg, dtype=np.float64).reshape(-1)[: x_reward.size]
        ax.plot(x_reward, sg, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.85, label="Savitzky-Golay", zorder=3)
    except Exception:
        pass
    _finish(ax, "Smoothing Comparison", "Episode Reward")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = fig_dir / "training_curves.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[HB][plot] Saved publication-quality training curves to {save_path}")


def plot_reward_decomposition(
    run_dir: Path,
    train_episode_history: List[int],
    cost_history: List[float],
    vio_history: List[float],
    improve_history: List[float],
    shaping_history: List[float],
    comp_history: Optional[List[float]] = None,
    ris_history: Optional[List[float]] = None,
    ema_span: int = 48,
) -> None:
    """=Training Episodesraw+EMA"""
    plt = _get_plt()

    fig_dir = run_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(train_episode_history, dtype=np.int64).reshape(-1)
    costs = np.asarray(cost_history, dtype=np.float64).reshape(-1)
    vios = np.asarray(vio_history, dtype=np.float64).reshape(-1)
    improves = np.asarray(improve_history, dtype=np.float64).reshape(-1)
    shapings = np.asarray(shaping_history, dtype=np.float64).reshape(-1)
    comp = np.asarray(comp_history, dtype=np.float64).reshape(-1) if comp_history is not None else None
    ris = np.asarray(ris_history, dtype=np.float64).reshape(-1) if ris_history is not None else None

    n = int(min(x.size, costs.size, vios.size, improves.size, shapings.size))
    if n <= 0:
        return
    x = x[:n]
    x_raw = np.asarray(x, dtype=np.int64).reshape(-1)
    costs = costs[:n]
    vios = vios[:n]
    improves = improves[:n]
    shapings = shapings[:n]
    if comp is not None and comp.size != n:
        comp = comp[:n] if comp.size > n else None
    if ris is not None and ris.size != n:
        ris = ris[:n] if ris.size > n else None

    if not _episode_axis_is_dense(x):
        dense_bundle = _load_dense_ckpt_bundle(Path(run_dir))
        x_dense_all = np.asarray(dense_bundle.get("episodes", np.asarray([], dtype=np.int64)), dtype=np.int64).reshape(-1)
        if x_dense_all.size > 0:
            x_valid = np.asarray(x, dtype=np.float64).reshape(-1)
            x_valid = x_valid[np.isfinite(x_valid)]
            m_range = np.ones_like(x_dense_all, dtype=bool)
            if x_valid.size > 0:
                lo = int(max(1, np.floor(np.nanmin(x_valid))))
                hi = int(np.ceil(np.nanmax(x_valid)))
                m_now = (x_dense_all >= lo) & (x_dense_all <= hi)
                if np.any(m_now):
                    m_range = m_now
            x_dense = np.asarray(x_dense_all[m_range], dtype=np.int64).reshape(-1)
            used_dense_any = False

            def _dense_pick(key: str) -> Optional[np.ndarray]:
                arr = np.asarray(dense_bundle.get(key, np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
                if arr.size != x_dense_all.size:
                    return None
                arr = arr[m_range]
                if arr.size != x_dense.size:
                    return None
                if not np.isfinite(arr).any():
                    return None
                return np.asarray(arr, dtype=np.float64).reshape(-1)

            costs_d = _dense_pick("train_cost_ep_dense")
            if costs_d is not None:
                costs = costs_d
                used_dense_any = True
            vios_d = _dense_pick("train_vio_ep_dense")
            if vios_d is not None:
                vios = vios_d
                used_dense_any = True
            improves_d = _dense_pick("train_improve_ep_dense")
            if improves_d is not None:
                improves = improves_d
                used_dense_any = True
            shapings_d = _dense_pick("train_shaping_ep_dense")
            if shapings_d is not None:
                shapings = shapings_d
                used_dense_any = True
            comp_d = _dense_pick("train_comp_ep_dense")
            if comp_d is not None:
                comp = comp_d
                used_dense_any = True
            ris_d = _dense_pick("train_ris_ep_dense")
            if ris_d is not None:
                ris = ris_d
                used_dense_any = True

            if used_dense_any and x_dense.size > 0:
                x = x_dense
                x_raw = np.asarray(x_dense, dtype=np.int64).reshape(-1)
                print(
                    f"[HB][plot] reward_decomposition uses checkpoint dense episode series: n={int(x_dense.size)}",
                    flush=True,
                )

    x_cost, costs = _expand_rollout_series_to_episode(x, costs, keep_non_finite_y=True)
    x_vio, vios = _expand_rollout_series_to_episode(x, vios, keep_non_finite_y=True)
    x_imp, improves = _expand_rollout_series_to_episode(x, improves, keep_non_finite_y=True)
    x_sh, shapings = _expand_rollout_series_to_episode(x, shapings, keep_non_finite_y=True)
    if x_cost.size > 0:
        x = x_cost
    else:
        x = x_sh
    if x.size <= 0:
        return
    if x_vio.size != x.size:
        vios = np.resize(vios, (x.size,))
    if x_imp.size != x.size:
        improves = np.resize(improves, (x.size,))
    if x_sh.size != x.size:
        shapings = np.resize(shapings, (x.size,))
    if comp is not None:
        x_comp0 = x_raw[: comp.size] if comp.size <= x_raw.size else np.resize(x_raw, (comp.size,))
        _x_comp, comp = _expand_rollout_series_to_episode(x_comp0, comp, keep_non_finite_y=True)
        if comp.size != x.size:
            comp = np.resize(comp, (x.size,))
    if ris is not None:
        x_ris0 = x_raw[: ris.size] if ris.size <= x_raw.size else np.resize(x_raw, (ris.size,))
        _x_ris, ris = _expand_rollout_series_to_episode(x_ris0, ris, keep_non_finite_y=True)
        if ris.size != x.size:
            ris = np.resize(ris, (x.size,))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reward Decomposition", fontsize=16, fontweight="bold")

    def _plot_one(ax, y: np.ndarray, title: str, ylabel: str, color: str, *, label_prefix: Optional[str] = None) -> None:
        if label_prefix is None:
            _plot_raw_and_ema(
                ax,
                x,
                y,
                color=color,
                label_raw="raw",
                span_cfg=int(ema_span),
                label_ema_prefix="EMA",
                tcom_adaptive_span=False,
            )
        else:
            _plot_raw_and_ema(
                ax,
                x,
                y,
                color=color,
                label_raw=f"{label_prefix}(raw)",
                span_cfg=int(ema_span),
                label_ema_prefix=f"{label_prefix} EMA",
                tcom_adaptive_span=False,
            )
        ax.set_xlabel("Training Episodes", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        _apply_episode_xaxis_sci(ax)
        ax.legend(loc="best", fontsize=9)

    _plot_one(axes[0, 0], costs, "Cost Component", "Cost", "#d62728")
    _plot_one(axes[0, 1], vios, "Violation Component", "Violation Penalty", "#ff7f0e")
    _plot_one(axes[1, 0], improves, "Improvement Bonus", "Improvement", "#2ca02c")

    ax_sh = axes[1, 1]
    _plot_one(ax_sh, shapings, "Total Shaping Bonus (EPIC)", "Shaping Bonus", "#9467bd", label_prefix="TotalShaping")
    
    if comp is not None and ris is not None and comp.size == x.size and ris.size == x.size:
        comp_ris = np.asarray(comp + ris, dtype=np.float64).reshape(-1)
        _plot_raw_and_ema(
            ax_sh,
            x,
            comp_ris,
            color="#1F77B4",
            label_raw="CoMP+RIS(raw)",
            span_cfg=int(ema_span),
            label_ema_prefix="CoMP+RIS EMA",
            tcom_adaptive_span=False,
        )
        ax_sh.legend(loc="best", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = fig_dir / "reward_decomposition.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[HB][plot] Saved publication-quality reward decomposition to {save_path}")


def plot_ppo_diagnostics_from_csv(run_dir: Path, out_dir: Optional[Path] = None, ema_span: int = 48) -> None:
    """ train_metrics.csv  PPO =Training Episodesraw+EMA"""
    plt = _get_plt()

    metrics_csv_path = run_dir / "train_metrics.csv"
    m = read_train_metrics_csv(metrics_csv_path)
    if not m:
        return

    
    x = m.get("train_episode", np.asarray([], dtype=np.float64))
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size <= 0:
        n_ep = np.asarray(m.get("train_n_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        if n_ep.size <= 0:
            return
        x = np.cumsum(np.maximum(n_ep, 0.0)).astype(np.int64).astype(np.float64)
    if x.size <= 0:
        return

    
    
    
    kl_pd = m.get("kl_per_dim", m.get("kl", np.full_like(x, np.nan)))
    clip = m.get("clipfrac", m.get("clipfrac_mean", np.full_like(x, np.nan)))
    ent_pd = m.get("entropy_per_dim", m.get("entropy", np.full_like(x, np.nan)))

    vloss = m.get("value_loss", np.full_like(x, np.nan))
    ev = m.get("explained_variance", np.full_like(x, np.nan))
    lr = m.get("lr", np.full_like(x, np.nan))

    
    def _clip(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.size != x.size:
            y = np.resize(y, (x.size,))
        return y

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    kl_pd = _clip(kl_pd)
    clip = _clip(clip)
    ent_pd = _clip(ent_pd)
    vloss = _clip(vloss)
    ev = _clip(ev)
    lr = _clip(lr)

    def _dense_or_expand(y: np.ndarray, dense_key: str) -> Tuple[np.ndarray, np.ndarray]:
        x_dense, y_dense, used_dense = _replace_sparse_with_dense_ckpt_series(
            run_dir,
            x,
            y,
            dense_key=dense_key,
        )
        if bool(used_dense) and x_dense.size > 0 and y_dense.size > 0:
            xd = np.asarray(x_dense, dtype=np.int64)
            yd = np.asarray(y_dense, dtype=np.float64)
            
            
            x_valid = x[np.isfinite(x)] if hasattr(x, '__len__') else np.asarray([])
            if x_valid.size > 0:
                first_real_ep = int(np.rint(x_valid[0]))
                mask_before = xd < first_real_ep
                if np.any(mask_before):
                    yd[mask_before] = np.nan
            return xd, yd
        return _expand_rollout_series_to_episode(x, y)

    x_kl, y_kl = _dense_or_expand(kl_pd, "train_kl_ep_dense")
    x_clip, y_clip = _dense_or_expand(clip, "train_clipfrac_ep_dense")
    x_ent, y_ent = _dense_or_expand(ent_pd, "train_entropy_ep_dense")
    x_vl, y_vl = _dense_or_expand(vloss, "train_value_loss_ep_dense")
    x_ev, y_ev = _dense_or_expand(ev, "train_explained_variance_ep_dense")
    x_lr, y_lr = _dense_or_expand(lr, "train_lr_ep_dense")

    fig_dir = out_dir if out_dir is not None else (run_dir / "figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("ARIA Training Diagnostics", fontsize=16, fontweight="bold")

    def _plot(ax, x_s: np.ndarray, y_s: np.ndarray, title: str, ylabel: str, color: str):
        if x_s.size <= 0 or y_s.size <= 0:
            return
        _plot_raw_and_ema(
            ax,
            x_s,
            y_s,
            color=color,
            label_raw="raw",
            span_cfg=int(ema_span),
            label_ema_prefix="EMA",
            tcom_adaptive_span=False,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Episodes", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)

    _plot(axes[0, 0], x_kl, y_kl, "KL / dim", "kl_per_dim", "#1f77b4")
    _plot(axes[0, 1], x_clip, y_clip, "Clip Fraction", "clipfrac", "#ff7f0e")
    _plot(axes[0, 2], x_ent, y_ent, "Entropy / dim", "entropy_per_dim", "#9467bd")
    _plot(axes[1, 0], x_vl, y_vl, "Value Loss", "value_loss", "#2ca02c")
    _plot(axes[1, 1], x_ev, y_ev, "Explained Variance", "explained_variance", "#d62728")
    _plot(axes[1, 2], x_lr, y_lr, "Learning Rate", "lr", "#8c564b")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = fig_dir / "PPO_Diagnostics.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    
    try:
        alias_path = fig_dir / "ppo_diagnostics.png"
        if str(alias_path) != str(save_path):
            plt.savefig(str(alias_path), dpi=300, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print(f"[HB][plot] Saved PPO diagnostics to {save_path}")


def plot_eval_metrics(
    run_dir: Path,
    eval_train_episodes: List[int],
    eval_paper_cost: List[float],
    eval_paper_cost_ci: List[float],
    eval_vio: List[float],
    eval_improve: List[float],
    ema_span: int = 48,
) -> None:
    """=Training Episodesraw+EMA"""
    plt = _get_plt()

    if not eval_train_episodes:
        return
    fig_dir = run_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(eval_train_episodes, dtype=np.int64)
    y_cost = np.asarray(eval_paper_cost, dtype=np.float64)
    y_ci = np.asarray(eval_paper_cost_ci, dtype=np.float64)
    y_vio = np.asarray(eval_vio, dtype=np.float64)
    y_imp = np.asarray(eval_improve, dtype=np.float64)

    n = int(min(x.size, y_cost.size, y_vio.size, y_imp.size))
    if n <= 0:
        return
    x = x[:n]
    y_cost = y_cost[:n]
    y_vio = y_vio[:n]
    y_imp = y_imp[:n]
    if y_ci.size != n:
        y_ci = np.resize(y_ci, (n,))

    x_roll = np.asarray(x, dtype=np.float64).reshape(-1)

    def _dense_or_expand(y: np.ndarray, dense_key: str) -> Tuple[np.ndarray, np.ndarray]:
        x_dense, y_dense, used_dense = _replace_sparse_with_dense_ckpt_series(
            run_dir,
            x_roll,
            y,
            dense_key=dense_key,
        )
        if bool(used_dense) and x_dense.size > 0 and y_dense.size > 0:
            xd = np.asarray(x_dense, dtype=np.int64)
            yd = np.asarray(y_dense, dtype=np.float64)
            
            
            x_roll_valid = x_roll[np.isfinite(x_roll)]
            if x_roll_valid.size > 0:
                x_roll_pos = x_roll_valid[x_roll_valid > 0.0]
                if x_roll_pos.size > 0:
                    first_real_ep = int(np.rint(np.nanmin(x_roll_pos)))
                else:
                    first_real_ep = int(np.rint(np.nanmin(x_roll_valid)))
                mask_before = xd < first_real_ep
                if np.any(mask_before):
                    yd[mask_before] = np.nan
            return xd, yd
        return _expand_rollout_series_to_episode(x_roll, y, keep_non_finite_y=True)

    x_cost, y_cost = _dense_or_expand(y_cost, "eval_det_paper_cost_ep_dense")
    x_vio, y_vio = _dense_or_expand(y_vio, "eval_det_vio_any_frac_ep_dense")
    x_imp, y_imp = _dense_or_expand(y_imp, "eval_det_improve_ep_dense")
    x_ci, y_ci = _dense_or_expand(y_ci, "eval_det_paper_cost_ci_ep_dense")
    x = x_cost if x_cost.size > 0 else (x_vio if x_vio.size > 0 else x_imp)
    if x.size <= 0:
        return
    if y_cost.size != x.size:
        y_cost = np.resize(y_cost, (x.size,))
    if y_vio.size != x.size:
        y_vio = np.resize(y_vio, (x.size,))
    if y_imp.size != x.size:
        y_imp = np.resize(y_imp, (x.size,))
    if y_ci.size != x.size:
        y_ci = np.resize(y_ci, (x.size,))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Deterministic Eval (On-Train)", fontsize=14, fontweight="bold")

    ax = axes[0]
    if y_ci.size == y_cost.size and np.isfinite(y_ci).any():
        m = np.isfinite(x) & np.isfinite(y_cost) & np.isfinite(y_ci)
        if np.any(m):
            ax.fill_between(x[m], (y_cost - y_ci)[m], (y_cost + y_ci)[m], color="#D95319", alpha=0.12, label="95% CI", zorder=1, linewidth=0)
    _plot_raw_and_ema(
        ax,
        x,
        y_cost,
        color="#D95319",
        label_raw="paper_cost(raw)",
        span_cfg=int(ema_span),
        label_ema_prefix="paper_cost EMA",
        tcom_adaptive_span=False,
    )
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("paper_cost")
    ax.set_title("Paper Cost")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    _plot_raw_and_ema(
        ax,
        x,
        y_vio,
        color="#0072BD",
        label_raw="violation(raw)",
        span_cfg=int(ema_span),
        label_ema_prefix="violation EMA",
        tcom_adaptive_span=False,
    )
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("violations")
    ax.set_title("Violations")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)

    ax = axes[2]
    _plot_raw_and_ema(
        ax,
        x,
        y_imp,
        color="#00B050",
        label_raw="improve(raw)",
        span_cfg=int(ema_span),
        label_ema_prefix="improve EMA",
        tcom_adaptive_span=False,
    )
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("improve")
    ax.set_title("Improvement")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = fig_dir / "C_eval_metrics.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[HB][plot] Saved eval metrics to {save_path}")


def plot_fixed_eval_curve(
    run_dir: Path,
    eval_train_episodes: List[int],
    reward_mean: List[float],
    reward_ci: List[float],
    paper_cost_mean: List[float],
    paper_cost_ci: List[float],
    vio_any_frac_mean: List[float],
    vio_any_frac_ci: List[float],
    out_dir: Optional[Path] = None,
    quota_note: Optional[str] = None,
    ema_span: int = 48,
    save_name: str = "FixedEval_PaperCost.png",
    y_mode: str = "paper_cost",  # 'paper_return' | 'paper_cost'
    title_prefix: str = "Fixed Eval",
) -> None:
    """
    FixedEvalCurvePhaseBseedsCI95

    
    - 
    - CI95  det-eval seedsepisode 
    - paper_cost paper_return 
    -  <run_dir>/figs/<ts>/FixedEval_PaperCost.png out_dir 
    """
    plt = _get_plt()
    _apply_tcom_style(plt)

    if not eval_train_episodes:
        return

    x_roll = np.asarray(eval_train_episodes, dtype=np.float64).reshape(-1)
    y_c = np.asarray(paper_cost_mean, dtype=np.float64).reshape(-1)
    y_c_ci = np.asarray(paper_cost_ci, dtype=np.float64).reshape(-1)
    y_v = np.asarray(vio_any_frac_mean, dtype=np.float64).reshape(-1)
    y_v_ci = np.asarray(vio_any_frac_ci, dtype=np.float64).reshape(-1)

    n0 = int(min(x_roll.size, y_c.size, y_v.size))
    if n0 <= 0:
        return
    x_roll = x_roll[:n0]
    y_c = y_c[:n0]
    y_v = y_v[:n0]
    if y_c_ci.size != n0:
        y_c_ci = np.resize(y_c_ci, (n0,))
    else:
        y_c_ci = y_c_ci[:n0]
    if y_v_ci.size != n0:
        y_v_ci = np.resize(y_v_ci, (n0,))
    else:
        y_v_ci = y_v_ci[:n0]

    fig_dir = out_dir if out_dir is not None else (run_dir / "figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    
    y_mode_n = str(y_mode or "paper_cost").strip().lower()
    if y_mode_n not in ("paper_return", "paper_cost"):
        y_mode_n = "paper_cost"

    def _dense_or_expand(y: np.ndarray, dense_key: str) -> Tuple[np.ndarray, np.ndarray]:
        x_dense, y_dense, used_dense = _replace_sparse_with_dense_ckpt_series(
            run_dir,
            x_roll,
            y,
            dense_key=dense_key,
        )
        if bool(used_dense) and x_dense.size > 0 and y_dense.size > 0:
            xd = np.asarray(x_dense, dtype=np.int64)
            yd = np.asarray(y_dense, dtype=np.float64)
            
            
            x_roll_valid = x_roll[np.isfinite(x_roll)]
            if x_roll_valid.size > 0:
                x_roll_pos = x_roll_valid[x_roll_valid > 0.0]
                if x_roll_pos.size > 0:
                    first_real_ep = int(np.rint(np.nanmin(x_roll_pos)))
                else:
                    first_real_ep = int(np.rint(np.nanmin(x_roll_valid)))
                mask_before = xd < first_real_ep
                if np.any(mask_before):
                    yd[mask_before] = np.nan
            return xd, yd
        return _expand_rollout_series_to_episode(x_roll, y, keep_non_finite_y=True)

    x_cost, y_cost = _dense_or_expand(y_c, "eval_det_paper_cost_ep_dense")
    x_cost_ci, y_cost_ci = _dense_or_expand(y_c_ci, "eval_det_paper_cost_ci_ep_dense")
    x_vio, y_vio = _dense_or_expand(y_v, "eval_det_vio_any_frac_ep_dense")
    x = x_cost if x_cost.size > 0 else (x_vio if x_vio.size > 0 else x_cost_ci)
    if x.size <= 0:
        return
    if y_cost.size != x.size:
        y_cost = np.resize(y_cost, (x.size,))
    if y_cost_ci.size != x.size:
        y_cost_ci = np.resize(y_cost_ci, (x.size,))
    if y_vio.size != x.size:
        y_vio = np.resize(y_vio, (x.size,))
    if y_mode_n == "paper_cost":
        y_r = y_cost
    else:
        # paper_return = -paper_cost
        y_r = -y_cost
    y_r_ci = y_cost_ci

    
    
    x_f, y_f, keep = filter_valid_xy(x, y_r, valid_mask=None, invalid_values=())
    if x_f.size <= 0 or y_f.size <= 0:
        return

    ci0 = np.asarray(y_r_ci, dtype=np.float64).reshape(-1)
    if ci0.size != x.size:
        ci0 = np.resize(ci0, (x.size,))
    ci_f = ci0[keep]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.4))
    ax.grid(True, alpha=0.35)

    span_eff = _plot_raw_and_ema(
        ax,
        x_f,
        y_f,
        color="#0072BD",
        label_raw="raw",
        span_cfg=int(ema_span),
        label_ema_prefix="EMA",
        tcom_adaptive_span=False,
    )
    if ci_f.size == y_f.size and np.isfinite(ci_f).any():
        ax.fill_between(x_f, y_f - ci_f, y_f + ci_f, color="#0072BD", alpha=0.12, linewidth=0, label="95% CI", zorder=1)

    ttl_prefix = str(title_prefix or "Fixed Eval").strip() or "Fixed Eval"
    if y_mode_n == "paper_cost":
        ax.set_title(f"{ttl_prefix}: Paper cost", pad=8, fontweight="bold")
    else:
        ax.set_title(f"{ttl_prefix}: Total rewards (Paper-return)", pad=8, fontweight="bold")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("paper_cost" if y_mode_n == "paper_cost" else "Total rewards")
    _apply_episode_xaxis_sci(ax)
    ax.legend(loc="lower right", ncol=1)

    span_txt = f"{int(span_eff)}" if span_eff is not None else "N/A"
    if y_mode_n == "paper_cost":
        note = f"paper_cost_ep (fixed-eval)\nEMA(span={span_txt})"
    else:
        note = f"paper_return_ep = -paper_cost_ep\nEMA(span={span_txt})"
    if quota_note:
        note = note + f"\n{str(quota_note)}"
    ax.text(
        0.01,
        0.01,
        note,
        transform=ax.transAxes,
        fontsize=9,
        color="0.25",
        ha="left",
        va="bottom",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = fig_dir / str(save_name)
    plt.savefig(str(save_path), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[HB][plot] Saved FixedEvalCurve to {save_path}")


def plot_metric1_robust_trend(
    run_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    save_name: str = "Metric1_RobustTrend.png",
    window_frac: float = 0.10,
    min_window: int = 5,
    ema_span: int = 48,
) -> Dict[str, Any]:
    """
    1
     paper_cost EMA
    """
    costs, metric1_source = _pick_metric1_dense_series(Path(run_dir))
    out: Dict[str, Any] = {
        "ok": False,
        "reason": "unknown",
        "metric1_source": str(metric1_source),
        "save_path": "",
    }
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    costs = costs[np.isfinite(costs)]
    if costs.size < 2:
        out["reason"] = "too_few_metric1_points"
        return out

    drop = _metric1_window_drop(costs, window_frac=window_frac, min_window=min_window)
    diag = _metric1_trend_diag(costs, window_frac=window_frac, min_window=min_window)

    x = np.asarray(np.arange(1, costs.size + 1), dtype=np.int64)
    plt = _get_plt()
    _apply_tcom_style(plt)
    fig_dir = out_dir if out_dir is not None else (Path(run_dir) / "figs" / "Metric1_RobustTrend")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6))
    ax.grid(True, alpha=0.35)
    span_eff = _plot_raw_and_ema(
        ax,
        x,
        costs,
        color="#0072BD",
        label_raw="raw",
        span_cfg=int(ema_span),
        label_ema_prefix="EMA",
        tcom_adaptive_span=False,
    )

    w = int(drop.get("window_size", 0.0))
    if w > 0 and w < x.size:
        ax.axvspan(float(x[0]), float(x[w - 1]), color="#DCEFFD", alpha=0.30, label="")
        ax.axvspan(float(x[-w]), float(x[-1]), color="#FDE7D9", alpha=0.28, label="")

    slope = float(diag.get("trend_slope_theil_sen_ema", float("nan")))
    intercept = float(diag.get("trend_intercept_ema", float("nan")))
    if np.isfinite(slope) and np.isfinite(intercept):
        trend_y = slope * x.astype(np.float64) + intercept
        ax.plot(x, trend_y, color="#D95319", linewidth=1.4, linestyle="--", label="")

    ax.set_title("Metric1 Robust Trend (paper_cost)", pad=8, fontweight="bold")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("paper_cost")
    ax.legend(loc="upper right", ncol=1)

    drop_pct = float(drop.get("relative_drop_pct", float("nan")))
    rebound_pct = float(diag.get("late_rebound_pct", float("nan")))
    span_txt = f"{int(span_eff)}" if span_eff is not None else "N/A"
    note = (
        f"window={int(w)} ({float(drop.get('window_frac', 0.0)):.0%})\n"
        f"relative_drop_pct={drop_pct:.3f}%\n"
        f"slope(EMA)={slope:.6f}/ep\n"
        f"late_rebound_pct={rebound_pct:.3f}%\n"
        f"EMA(span={span_txt})"
    )
    ax.text(
        0.01,
        0.01,
        note,
        transform=ax.transAxes,
        fontsize=9,
        color="0.25",
        ha="left",
        va="bottom",
    )

    save_path = fig_dir / str(save_name)
    fig.savefig(str(save_path), dpi=600, bbox_inches="tight")
    plt.close(fig)

    out.update(
        {
            "ok": True,
            "reason": "ok",
            "save_path": str(save_path),
            "window_metrics": drop,
            "trend_metrics": diag,
        }
    )
    print(f"[HB][plot] Saved Metric1 robust trend plot to {save_path}", flush=True)
    return out


# ============================================================

# ============================================================

def compute_h11_reward_balance(
    metrics_csv_path: Path,
    *,
    sample_points: int = 5,
) -> Dict[str, Any]:
    """
     train_metrics.csv paper/proximity/shaping/guard

     sample_points 
    guard  = reward_constraint_step proximity

     dict  pass/reason/abs_mean/shares/thresholds
    """
    result: Dict[str, Any] = {
        "pass": False,
        "reason": "unknown",
        "n_points": 0,
        "abs_mean": {},
        "shares": {},
        "thresholds": {
            "paper_min": 0.35, "paper_max": 0.55,
            "proximity_min": 0.15, "proximity_max": 0.30,
            "shaping_min": 0.10, "shaping_max": 0.25,
            "guard_min": 0.05, "guard_max": 0.20,
        },
    }

    csv_path = Path(metrics_csv_path)
    if not csv_path.exists():
        result["reason"] = "missing_train_metrics_csv"
        return result

    
    rows: List[Dict[str, str]] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        result["reason"] = f"csv_read_error: {e}"
        return result

    if len(rows) < 3:
        result["reason"] = "too_few_points"
        return result

    
    n = len(rows)
    half_start = n // 2
    indices = np.linspace(half_start, n - 1, min(sample_points, n - half_start), dtype=int)
    indices = sorted(set(indices))

    paper_vals = []
    prox_vals = []
    shaping_vals = []
    guard_vals = []

    for idx in indices:
        row = rows[idx]
        try:
            
            p = abs(float(row.get("reward_paper_step_mean", row.get("reward_paper_step", 0.0))))
            x = abs(float(row.get("reward_proximity_step_mean", row.get("reward_proximity_step", 0.0))))
            s = abs(float(row.get("reward_shaping_step_mean", row.get("reward_shaping_step", 0.0))))
            g = abs(float(row.get("reward_constraint_step_mean", row.get("reward_constraint_step", 0.0))))
            paper_vals.append(p)
            prox_vals.append(x)
            shaping_vals.append(s)
            guard_vals.append(g)
        except (ValueError, TypeError):
            continue

    n_pts = len(paper_vals)
    result["n_points"] = n_pts
    if n_pts < 1:
        result["reason"] = "no_valid_sample_points"
        return result

    pm = float(np.mean(paper_vals))
    xm = float(np.mean(prox_vals))
    sm = float(np.mean(shaping_vals))
    gm = float(np.mean(guard_vals))
    total = pm + xm + sm + gm

    result["abs_mean"] = {"paper": pm, "proximity": xm, "shaping": sm, "guard": gm}

    if total < 1e-9:
        result["reason"] = "total_abs_near_zero"
        return result

    shares = {
        "paper": pm / total,
        "proximity": xm / total,
        "shaping": sm / total,
        "guard": gm / total,
    }
    result["shares"] = shares

    
    th = result["thresholds"]
    fails = []
    for layer in ("paper", "proximity", "shaping", "guard"):
        lo = th[f"{layer}_min"]
        hi = th[f"{layer}_max"]
        val = shares[layer]
        if val < lo or val > hi:
            fails.append(f"{layer}_share_out_of_range({val:.3f} not in [{lo},{hi}])")

    if fails:
        result["pass"] = False
        result["reason"] = "; ".join(fails)
    else:
        result["pass"] = True
        result["reason"] = "ok"

    return result


def _h11_safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _h11_tail_mean(a: np.ndarray, frac: float = 0.30) -> float:
    arr = np.asarray(a, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float("nan")
    w = int(max(1, round(float(frac) * float(arr.size))))
    w = min(w, int(arr.size))
    return float(np.mean(arr[-w:]))


def _h11_stage_means(a: np.ndarray) -> Dict[str, float]:
    """
     30%/45%/25%  early/mid/late
    """
    arr = np.asarray(a, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n <= 0:
        return {
            "n": 0,
            "early": float("nan"),
            "mid": float("nan"),
            "late": float("nan"),
        }

    i_early = int(max(1, round(0.30 * n)))
    i_mid = int(max(i_early + 1, round(0.75 * n)))
    i_mid = min(i_mid, n - 1) if n >= 2 else 1
    y_early = arr[:i_early]
    y_mid = arr[i_early:i_mid]
    y_late = arr[i_mid:]
    return {
        "n": int(n),
        "early": float(np.mean(y_early)) if y_early.size > 0 else float("nan"),
        "mid": float(np.mean(y_mid)) if y_mid.size > 0 else float("nan"),
        "late": float(np.mean(y_late)) if y_late.size > 0 else float("nan"),
    }


def _h11_average_rank(x: np.ndarray) -> np.ndarray:
    """
     Spearman 
    """
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(v.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64)

    order = np.argsort(v, kind="mergesort")
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        vi = v[order[i]]
        while j + 1 < n and float(v[order[j + 1]]) == float(vi):
            j += 1
        
        rank_val = 0.5 * (float(i + 1) + float(j + 1))
        ranks[order[i : j + 1]] = rank_val
        i = j + 1
    return ranks


def _h11_spearman(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=np.float64).reshape(-1)
    xb = np.asarray(b, dtype=np.float64).reshape(-1)
    n = int(min(xa.size, xb.size))
    if n <= 2:
        return float("nan")
    xa = xa[:n]
    xb = xb[:n]
    m = np.isfinite(xa) & np.isfinite(xb)
    xa = xa[m]
    xb = xb[m]
    if xa.size <= 2:
        return float("nan")
    ra = _h11_average_rank(xa)
    rb = _h11_average_rank(xb)
    sa = float(np.std(ra))
    sb = float(np.std(rb))
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_h11_cfg_dropped_keys_audit(
    dropped_keys: Sequence[str],
    *,
    critical_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    H11.8  env_cfg 
    """
    critical = list(critical_keys) if critical_keys is not None else [
        "reward_proximity_weight",
        "reward_proximity_weight_final",
        "reward_proximity_decay_start_step",
        "reward_proximity_decay_end_step",
        "reward_proximity_mode",
        "coverage_floor_weight",
        "coverage_floor_threshold",
        "beta_comp_max",
        "beta_ris_max",
        "beta_end_frac",
    ]
    dropped_norm = sorted(set([str(k).strip() for k in (dropped_keys or []) if str(k).strip()]))
    critical_set = set([str(k).strip() for k in critical if str(k).strip()])
    critical_dropped = [k for k in dropped_norm if k in critical_set]
    passed = bool(len(critical_dropped) == 0)
    return {
        "pass": passed,
        "dropped_count": int(len(dropped_norm)),
        "dropped_keys": dropped_norm,
        "critical_dropped_keys": critical_dropped,
        "reason": "ok" if passed else f"critical_keys_dropped: {critical_dropped}",
    }


def compute_h11_knob_effect_audit(
    metrics_csv_path: Path,
    *,
    env_cfg_dict_clean: Optional[Dict[str, Any]] = None,
    dropped_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    H11.8  +  + 
    """
    m = read_train_metrics_csv(Path(metrics_csv_path))
    cfg = dict(env_cfg_dict_clean or {})
    drop = dict(dropped_audit or {})

    prox_step = np.asarray(m.get("reward_proximity_step_mean", np.asarray([], dtype=np.float64)), dtype=np.float64)
    main_step = np.asarray(
        m.get(
            "reward_main_step_mean",
            m.get("reward_paper_step_mean", np.asarray([], dtype=np.float64)),
        ),
        dtype=np.float64,
    )
    beta_comp_t = np.asarray(m.get("beta_comp_t_mean", np.asarray([], dtype=np.float64)), dtype=np.float64)
    beta_ris_t = np.asarray(m.get("beta_ris_t_mean", np.asarray([], dtype=np.float64)), dtype=np.float64)
    lambda_eff = np.asarray(m.get("lambda_v_effective_mean", np.asarray([], dtype=np.float64)), dtype=np.float64)
    constraint_step = np.asarray(m.get("reward_constraint_step_mean", np.asarray([], dtype=np.float64)), dtype=np.float64)
    shaping_gate = np.asarray(m.get("shaping_gate_mean", np.asarray([], dtype=np.float64)), dtype=np.float64)
    comp_enable = np.asarray(
        m.get("comp_enable_rate", m.get("comp_enable_rate_mean", np.asarray([], dtype=np.float64))),
        dtype=np.float64,
    )

    cfg_reward_prox = _h11_safe_float(cfg.get("reward_proximity_weight", float("nan")))
    cfg_beta_comp = _h11_safe_float(cfg.get("beta_comp_max", float("nan")))
    cfg_beta_ris = _h11_safe_float(cfg.get("beta_ris_max", float("nan")))
    cfg_beta_end = _h11_safe_float(cfg.get("beta_end_frac", float("nan")))
    cfg_lambda_v = _h11_safe_float(cfg.get("lambda_v", float("nan")))
    cfg_reward_uncovered = _h11_safe_float(cfg.get("reward_uncovered_weight", float("nan")))

    
    prox_ratio_early_max = 0.45
    prox_ratio_mid_max = 0.35
    prox_ratio_late_max = 0.25
    comp_tail_min = 0.05
    comp_tail_max = 0.95
    comp_tail_lock_frac_max = 0.80

    n_pm = int(min(prox_step.size, main_step.size))
    prox_ratio = np.asarray([], dtype=np.float64)
    if n_pm > 0:
        px = np.asarray(prox_step[:n_pm], dtype=np.float64)
        mx = np.asarray(main_step[:n_pm], dtype=np.float64)
        mask_pm = np.isfinite(px) & np.isfinite(mx)
        if np.any(mask_pm):
            px = np.abs(px[mask_pm])
            mx = np.abs(mx[mask_pm])
            denom = np.maximum(mx, 1e-9)
            prox_ratio = np.asarray(px / denom, dtype=np.float64).reshape(-1)

    prox_ratio_stage = _h11_stage_means(prox_ratio)
    comp_tail = np.asarray(comp_enable, dtype=np.float64).reshape(-1)
    comp_tail = comp_tail[np.isfinite(comp_tail)]
    if comp_tail.size > 0:
        w_comp = int(max(1, round(0.30 * float(comp_tail.size))))
        w_comp = min(w_comp, int(comp_tail.size))
        comp_tail = np.asarray(comp_tail[-w_comp:], dtype=np.float64)
    comp_tail_mean = float(np.mean(comp_tail)) if comp_tail.size > 0 else float("nan")
    comp_tail_low_frac = float(np.mean(comp_tail <= comp_tail_min)) if comp_tail.size > 0 else float("nan")
    comp_tail_high_frac = float(np.mean(comp_tail >= comp_tail_max)) if comp_tail.size > 0 else float("nan")
    comp_tail_lock_frac = float(np.mean((comp_tail <= comp_tail_min) | (comp_tail >= comp_tail_max))) if comp_tail.size > 0 else float("nan")

    runtime = {
        "reward_proximity_step_tail_mean": _h11_tail_mean(prox_step),
        "reward_main_step_tail_mean": _h11_tail_mean(main_step),
        "prox_main_ratio_early_mean": _h11_safe_float(prox_ratio_stage.get("early", float("nan"))),
        "prox_main_ratio_mid_mean": _h11_safe_float(prox_ratio_stage.get("mid", float("nan"))),
        "prox_main_ratio_late_mean": _h11_safe_float(prox_ratio_stage.get("late", float("nan"))),
        "prox_main_ratio_tail_mean": _h11_tail_mean(prox_ratio),
        "prox_main_ratio_points": int(prox_ratio_stage.get("n", 0)),
        "beta_comp_t_tail_mean": _h11_tail_mean(beta_comp_t),
        "beta_ris_t_tail_mean": _h11_tail_mean(beta_ris_t),
        "lambda_v_effective_tail_mean": _h11_tail_mean(lambda_eff),
        "reward_constraint_step_tail_mean": _h11_tail_mean(constraint_step),
        "shaping_gate_tail_mean": _h11_tail_mean(shaping_gate),
        "comp_enable_rate_tail_mean": float(comp_tail_mean),
        "comp_enable_rate_tail_low_frac": float(comp_tail_low_frac),
        "comp_enable_rate_tail_high_frac": float(comp_tail_high_frac),
        "comp_enable_rate_tail_lock_frac": float(comp_tail_lock_frac),
    }

    eps = 1e-6
    prox_e = _h11_safe_float(runtime["prox_main_ratio_early_mean"], float("nan"))
    prox_m = _h11_safe_float(runtime["prox_main_ratio_mid_mean"], float("nan"))
    prox_l = _h11_safe_float(runtime["prox_main_ratio_late_mean"], float("nan"))
    prox_ratio_has_data = bool(int(runtime.get("prox_main_ratio_points", 0)) >= 3)
    checks = {
        "critical_keys_not_dropped": bool(drop.get("pass", True)),
        "reward_proximity_effective": (abs(cfg_reward_prox) <= eps)
        or (abs(_h11_safe_float(runtime["reward_proximity_step_tail_mean"], 0.0)) > eps),
        "prox_main_ratio_staged_bounded": (not prox_ratio_has_data)
        or (
            np.isfinite(prox_e)
            and np.isfinite(prox_m)
            and np.isfinite(prox_l)
            and (prox_e <= prox_ratio_early_max)
            and (prox_m <= prox_ratio_mid_max)
            and (prox_l <= prox_ratio_late_max)
        ),
        "prox_main_ratio_staged_descend": (not prox_ratio_has_data)
        or (
            np.isfinite(prox_e)
            and np.isfinite(prox_m)
            and np.isfinite(prox_l)
            and (prox_e >= prox_m - 1e-6)
            and (prox_m >= prox_l - 1e-6)
        ),
        "comp_enable_rate_not_locked": np.isfinite(comp_tail_mean)
        and (comp_tail_mean > comp_tail_min)
        and (comp_tail_mean < comp_tail_max),
        "comp_enable_rate_lock_frac_ok": np.isfinite(comp_tail_lock_frac)
        and (comp_tail_lock_frac <= comp_tail_lock_frac_max),
        "beta_comp_effective": (not np.isfinite(cfg_beta_comp))
        or (cfg_beta_comp <= eps)
        or (_h11_safe_float(runtime["beta_comp_t_tail_mean"], 0.0) > eps),
        "beta_ris_effective": (not np.isfinite(cfg_beta_ris))
        or (cfg_beta_ris <= eps)
        or (_h11_safe_float(runtime["beta_ris_t_tail_mean"], 0.0) > eps),
        "lambda_v_effective": (not np.isfinite(cfg_lambda_v))
        or (cfg_lambda_v <= eps)
        or (_h11_safe_float(runtime["lambda_v_effective_tail_mean"], 0.0) > eps)
        or (abs(_h11_safe_float(runtime["reward_constraint_step_tail_mean"], 0.0)) <= eps),
        
        "reward_uncovered_effective": (not np.isfinite(cfg_reward_uncovered))
        or (cfg_reward_uncovered <= eps)
        or (abs(_h11_safe_float(runtime["reward_constraint_step_tail_mean"], 0.0)) > eps),
        
        "beta_end_frac_effective": (not np.isfinite(cfg_beta_end))
        or (cfg_beta_end < 0.999)
        or (
            abs(_h11_safe_float(runtime["beta_comp_t_tail_mean"], 0.0) - cfg_beta_comp)
            <= max(1e-3, 0.05 * max(cfg_beta_comp, 0.0))
            and abs(_h11_safe_float(runtime["beta_ris_t_tail_mean"], 0.0) - cfg_beta_ris)
            <= max(1e-3, 0.05 * max(cfg_beta_ris, 0.0))
        ),
    }

    failed = [k for k, v in checks.items() if not bool(v)]
    passed = bool(len(failed) == 0)
    return {
        "pass": passed,
        "reason": "ok" if passed else f"check_failed: {failed}",
        "checks": checks,
        "dropped_audit": drop,
        "thresholds": {
            "prox_main_ratio_early_max": float(prox_ratio_early_max),
            "prox_main_ratio_mid_max": float(prox_ratio_mid_max),
            "prox_main_ratio_late_max": float(prox_ratio_late_max),
            "comp_enable_rate_tail_min": float(comp_tail_min),
            "comp_enable_rate_tail_max": float(comp_tail_max),
            "comp_enable_rate_tail_lock_frac_max": float(comp_tail_lock_frac_max),
        },
        "config": {
            "reward_proximity_weight": cfg_reward_prox,
            "beta_comp_max": cfg_beta_comp,
            "beta_ris_max": cfg_beta_ris,
            "beta_end_frac": cfg_beta_end,
            "lambda_v": cfg_lambda_v,
            "reward_uncovered_weight": cfg_reward_uncovered,
        },
        "runtime": runtime,
    }


def compute_h11_reward_shape_gate_dense(
    metrics_csv_path: Path,
    *,
    run_dir: Optional[Path] = None,
    require_dense_ckpt: bool = True,
    drawdown_ratio_max: float = 0.15,
    volatility_ratio_max: float = 0.85,
    sign_changes_per100_max: float = 45.0,
    spearman_min: float = 0.70,
) -> Dict[str, Any]:
    """
    H11.8  dense episode
    """
    out: Dict[str, Any] = {
        "pass": False,
        "reason": "unknown",
        "source": "train_metrics",
        "require_dense_ckpt": bool(require_dense_ckpt),
        "n_dense": 0,
        "reward_early": float("nan"),
        "reward_mid": float("nan"),
        "reward_late": float("nan"),
        "reward_peak": float("nan"),
        "reward_late_min": float("nan"),
        "late_drawdown": float("nan"),
        "late_drawdown_ratio": float("nan"),
        "late_volatility_ratio": float("nan"),
        "sign_changes": 0,
        "sign_changes_per100": float("nan"),
        "spearman_reward_vs_neg_cost": float("nan"),
        "mean_rollout_episodes": float("nan"),
        "dominant_period_episodes": float("nan"),
        "dominant_period_power_frac": float("nan"),
        "within_rollout_var_ratio": float("nan"),
        "between_rollout_var_ratio": float("nan"),
        "rollout_mean_sign_changes_per100": float("nan"),
        "thresholds": {
            "trend": "reward_early < reward_mid <= reward_late",
            "late_drawdown_ratio_max": float(drawdown_ratio_max),
            "late_volatility_ratio_max": float(volatility_ratio_max),
            "sign_changes_per100_max": float(sign_changes_per100_max),
            "spearman_reward_vs_neg_cost_min": float(spearman_min),
        },
        "checks": {},
        "diagnostic_warning": "",
    }

    metrics_csv_path = Path(metrics_csv_path)
    run_root = Path(run_dir) if run_dir is not None else metrics_csv_path.parent
    m = read_train_metrics_csv(metrics_csv_path)

    reward = np.asarray(m.get("reward_total_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
    cost = np.asarray(m.get("paper_cost_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)

    dense_loaded = False
    
    dense = _load_dense_ckpt_bundle(run_root)
    reward_d = np.asarray(dense.get("train_reward_total_ep_dense", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
    cost_d = np.asarray(dense.get("train_cost_ep_dense", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
    if reward_d.size > 0 and cost_d.size == reward_d.size:
        reward = reward_d
        cost = cost_d
        out["source"] = "checkpoint_history_dense"
        dense_loaded = True

    
    if bool(require_dense_ckpt) and (not dense_loaded):
        out["reason"] = "missing_checkpoint_dense"
        out["n_dense"] = 0
        return out

    n = int(min(reward.size, cost.size))
    if n < 20:
        out["reason"] = "too_few_dense_points"
        out["n_dense"] = n
        return out

    reward = reward[:n]
    cost = cost[:n]
    mask = np.isfinite(reward) & np.isfinite(cost)
    reward = reward[mask]
    cost = cost[mask]
    n = int(reward.size)
    out["n_dense"] = n
    if n < 20:
        out["reason"] = "too_few_valid_dense_points"
        return out

    i_early = int(max(1, round(0.30 * n)))
    i_mid = int(max(i_early + 1, round(0.75 * n)))
    i_mid = min(i_mid, n - 1)
    y_early = reward[:i_early]
    y_mid = reward[i_early:i_mid]
    y_late = reward[i_mid:]
    if y_early.size <= 0 or y_mid.size <= 0 or y_late.size <= 0:
        out["reason"] = "invalid_stage_split"
        return out

    reward_early = float(np.mean(y_early))
    reward_mid = float(np.mean(y_mid))
    reward_late = float(np.mean(y_late))
    reward_peak = float(np.max(reward))
    reward_late_min = float(np.min(y_late))
    late_drawdown = float(reward_peak - reward_late_min)
    denom = float(max(reward_peak - reward_early, 1e-9))
    late_drawdown_ratio = float(late_drawdown / denom)

    std_early = float(np.std(y_early))
    std_late = float(np.std(y_late))
    late_vol_ratio = float(std_late / max(std_early, 1e-9))

    d = np.diff(reward)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) > 1e-12]
    if d.size <= 1:
        sign_changes = 0
    else:
        s = np.sign(d)
        sign_changes = int(np.sum(s[1:] * s[:-1] < 0))
    sign_changes_per100 = float(100.0 * float(sign_changes) / max(1, n - 1))

    sp = float(_h11_spearman(reward, -cost))

    
    
    
    
    dominant_period = float("nan")
    dominant_power_frac = float("nan")
    within_var_ratio = float("nan")
    between_var_ratio = float("nan")
    rollout_mean_sign_changes_per100 = float("nan")
    mean_rollout_eps = float("nan")
    try:
        y0 = np.asarray(reward, dtype=np.float64)
        y0 = y0[np.isfinite(y0)]
        if y0.size >= 32:
            y1 = y0 - float(np.mean(y0))
            spec = np.abs(np.fft.rfft(y1)) ** 2
            if spec.size > 2:
                spec[0] = 0.0
                k = int(np.argmax(spec[1:]) + 1)
                if k > 0:
                    dominant_period = float(y0.size / float(k))
                    dominant_power_frac = float(spec[k] / max(float(np.sum(spec)), 1e-12))

        n_ep = np.asarray(m.get("train_n_ep", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        n_ep = n_ep[np.isfinite(n_ep)]
        n_ep = n_ep[n_ep > 0.0]
        if n_ep.size > 0:
            mean_rollout_eps = float(np.mean(n_ep))
            seg_n = np.asarray(np.rint(n_ep), dtype=np.int64)
            seg_n = seg_n[seg_n > 0]
            if seg_n.size > 0:
                bounds = np.cumsum(seg_n)
                bounds = bounds[bounds <= int(reward.size)]
                if bounds.size > 0:
                    starts = np.concatenate([np.asarray([0], dtype=np.int64), bounds[:-1]])
                    ends = bounds
                    seg_means: List[float] = []
                    within_num = 0.0
                    for a, b in zip(starts.tolist(), ends.tolist()):
                        if int(b) <= int(a):
                            continue
                        seg = reward[int(a):int(b)]
                        if seg.size <= 0:
                            continue
                        mu = float(np.mean(seg))
                        seg_means.append(mu)
                        within_num += float(np.sum((seg - mu) ** 2))
                    if len(seg_means) >= 1:
                        total_var = float(np.var(reward))
                        within_var = float(within_num / max(1, int(reward.size)))
                        between_var = float(max(total_var - within_var, 0.0))
                        within_var_ratio = float(within_var / max(total_var, 1e-12))
                        between_var_ratio = float(between_var / max(total_var, 1e-12))

                        seg_d = np.diff(np.asarray(seg_means, dtype=np.float64))
                        seg_d = seg_d[np.isfinite(seg_d)]
                        seg_d = seg_d[np.abs(seg_d) > 1e-12]
                        if seg_d.size <= 1:
                            rollout_mean_sign_changes_per100 = 0.0
                        else:
                            seg_s = np.sign(seg_d)
                            seg_changes = int(np.sum(seg_s[1:] * seg_s[:-1] < 0))
                            rollout_mean_sign_changes_per100 = float(
                                100.0 * float(seg_changes) / max(1, len(seg_means) - 1)
                            )
    except Exception as exc:
        out["diagnostic_warning"] = f"dense_structure_diag_failed: {exc}"

    trend_pass = bool(reward_early < reward_mid <= reward_late)
    drawdown_pass = bool(np.isfinite(late_drawdown_ratio) and late_drawdown_ratio <= float(drawdown_ratio_max))
    volatility_pass = bool(np.isfinite(late_vol_ratio) and late_vol_ratio <= float(volatility_ratio_max))
    smoothness_pass = bool(np.isfinite(sign_changes_per100) and sign_changes_per100 <= float(sign_changes_per100_max))
    spearman_pass = bool(np.isfinite(sp) and sp >= float(spearman_min))

    out.update(
        {
            "reward_early": reward_early,
            "reward_mid": reward_mid,
            "reward_late": reward_late,
            "reward_peak": reward_peak,
            "reward_late_min": reward_late_min,
            "late_drawdown": late_drawdown,
            "late_drawdown_ratio": late_drawdown_ratio,
            "late_volatility_ratio": late_vol_ratio,
            "sign_changes": int(sign_changes),
            "sign_changes_per100": sign_changes_per100,
            "spearman_reward_vs_neg_cost": sp,
            "mean_rollout_episodes": mean_rollout_eps,
            "dominant_period_episodes": dominant_period,
            "dominant_period_power_frac": dominant_power_frac,
            "within_rollout_var_ratio": within_var_ratio,
            "between_rollout_var_ratio": between_var_ratio,
            "rollout_mean_sign_changes_per100": rollout_mean_sign_changes_per100,
            "checks": {
                "trend_pass": trend_pass,
                "drawdown_pass": drawdown_pass,
                "volatility_pass": volatility_pass,
                "smoothness_pass": smoothness_pass,
                "spearman_pass": spearman_pass,
            },
        }
    )

    passed = bool(trend_pass and drawdown_pass and volatility_pass and smoothness_pass and spearman_pass)
    out["pass"] = passed
    out["reason"] = "ok" if passed else "threshold_failed"
    return out


def compute_h11_a_acceptance(
    metrics_csv_path: Path,
    *,
    run_dir: Optional[Path] = None,
    env_cfg_dict_clean: Optional[Dict[str, Any]] = None,
    dropped_keys: Optional[Sequence[str]] = None,
    total_steps_cfg: Optional[int] = None,
    smoke_max_steps: int = 6000,
) -> Dict[str, Any]:
    """
    H11-A 
    - smoke + 
    - short smoke  dense  + paper_cost  + PPO 
    """
    csv_path = Path(metrics_csv_path)
    run_root = Path(run_dir) if run_dir is not None else csv_path.parent

    out: Dict[str, Any] = {
        "pass": False,
        "reason": "unknown",
        "stage_mode": "short",
        "smoke_max_steps": int(smoke_max_steps),
        "total_steps_cfg": int(total_steps_cfg) if total_steps_cfg is not None else None,
        "checks": {},
        "metrics": {},
        "artifacts": {},
    }

    if total_steps_cfg is not None and int(total_steps_cfg) <= int(smoke_max_steps):
        out["stage_mode"] = "smoke"

    drop = compute_h11_cfg_dropped_keys_audit(
        dropped_keys or [],
        critical_keys=[
            "reward_proximity_weight",
            "reward_proximity_weight_final",
            "reward_proximity_decay_start_step",
            "reward_proximity_decay_end_step",
            "reward_proximity_mode",
            "coverage_floor_weight",
            "coverage_floor_threshold",
            "beta_comp_max",
            "beta_ris_max",
            "beta_end_frac",
        ],
    )
    knob = compute_h11_knob_effect_audit(
        metrics_csv_path=csv_path,
        env_cfg_dict_clean=env_cfg_dict_clean,
        dropped_audit=drop,
    )
    balance = compute_h11_reward_balance(csv_path)
    dense = compute_h11_reward_shape_gate_dense(csv_path, run_dir=run_root, require_dense_ckpt=True)

    m = read_train_metrics_csv(csv_path)
    clip_tail = _h11_tail_mean(np.asarray(m.get("clipfrac", m.get("clipfrac_mean", np.asarray([], dtype=np.float64))), dtype=np.float64))
    kl_tail = _h11_tail_mean(np.asarray(m.get("kl_per_dim", np.asarray([], dtype=np.float64)), dtype=np.float64))
    ev_tail = _h11_tail_mean(np.asarray(m.get("explained_variance", np.asarray([], dtype=np.float64)), dtype=np.float64))
    ppo_health_pass = bool(
        np.isfinite(clip_tail)
        and np.isfinite(kl_tail)
        and np.isfinite(ev_tail)
        and (0.05 <= float(clip_tail) <= 0.30)
        and (5e-4 <= float(kl_tail) <= 4e-3)
        and (float(ev_tail) >= 0.20)
    )

    
    costs, metric1_source = _pick_metric1_det_eval_series(m)
    if costs.size < 2:
        costs, metric1_source = _pick_metric1_fixed_eval_series(run_root)
    cost_drop = _metric1_window_drop(costs, window_frac=0.05, min_window=1)
    cost_trend = _metric1_trend_diag(costs, window_frac=0.05, min_window=1)
    paper_first = float(cost_drop.get("paper_cost_first_window_mean", float("nan")))
    paper_last = float(cost_drop.get("paper_cost_last_window_mean", float("nan")))
    relative_drop_pct = float(cost_drop.get("relative_drop_pct", float("nan")))
    cost_drop_pass = bool(np.isfinite(relative_drop_pct) and (relative_drop_pct >= 10.0))

    
    det_cost, det_source = _pick_metric1_dense_series(run_root)
    det_drop = _metric1_window_drop(det_cost, window_frac=0.05, min_window=1)
    det_first = float(det_drop.get("paper_cost_first_window_mean", float("nan")))
    det_last = float(det_drop.get("paper_cost_last_window_mean", float("nan")))
    det_drop_pct = float(det_drop.get("relative_drop_pct", float("nan")))
    det_main_cost, _det_main_source = _pick_metric1_det_eval_series(m)
    det_main_drop = _metric1_window_drop(det_main_cost, window_frac=0.05, min_window=1)
    det_main_first = float(det_main_drop.get("paper_cost_first_window_mean", float("nan")))
    det_main_last = float(det_main_drop.get("paper_cost_last_window_mean", float("nan")))
    det_main_drop_pct = float(det_main_drop.get("relative_drop_pct", float("nan")))

    knob_checks = dict(knob.get("checks", {}) or {})
    prox_guard_pass = bool(
        knob_checks.get("prox_main_ratio_staged_bounded", True)
        and knob_checks.get("prox_main_ratio_staged_descend", True)
    )
    comp_guard_pass = bool(
        knob_checks.get("comp_enable_rate_not_locked", True)
        and knob_checks.get("comp_enable_rate_lock_frac_ok", True)
    )

    checks = {
        "cfg_dropped_pass": bool(drop.get("pass", False)),
        "knob_effect_pass": bool(knob.get("pass", False)),
        "prox_guard_pass": bool(prox_guard_pass),
        "comp_gate_guard_pass": bool(comp_guard_pass),
        "reward_balance_pass": bool(balance.get("pass", False)),
        "dense_shape_pass": bool(dense.get("pass", False)),
        "paper_cost_drop_pass": bool(cost_drop_pass),
        "ppo_health_pass": bool(ppo_health_pass),
    }

    smoke_pass = bool(checks["cfg_dropped_pass"] and checks["reward_balance_pass"])
    short_pass = bool(
        smoke_pass
        and checks["prox_guard_pass"]
        and checks["comp_gate_guard_pass"]
        and checks["dense_shape_pass"]
        and checks["paper_cost_drop_pass"]
        and checks["ppo_health_pass"]
    )

    mode = str(out.get("stage_mode", "short"))
    if mode == "smoke":
        out["pass"] = smoke_pass
        out["reason"] = "ok" if smoke_pass else "smoke_threshold_failed"
    else:
        out["pass"] = short_pass
        out["reason"] = "ok" if short_pass else "short_threshold_failed"

    out["checks"] = checks
    out["metrics"] = {
        "clipfrac_late_mean": float(clip_tail) if np.isfinite(clip_tail) else float("nan"),
        "kl_per_dim_late_mean": float(kl_tail) if np.isfinite(kl_tail) else float("nan"),
        "explained_variance_late_mean": float(ev_tail) if np.isfinite(ev_tail) else float("nan"),
        "paper_cost_first_5pct_mean": paper_first,
        "paper_cost_last_5pct_mean": paper_last,
        "paper_cost_first_10pct_mean": paper_first,
        "paper_cost_last_10pct_mean": paper_last,
        "paper_cost_first_window_mean": paper_first,
        "paper_cost_last_window_mean": paper_last,
        
        "paper_cost_first": paper_first,
        "paper_cost_last": paper_last,
        "paper_cost_first_point": float(cost_drop.get("paper_cost_first_point", float("nan"))),
        "paper_cost_last_point": float(cost_drop.get("paper_cost_last_point", float("nan"))),
        "metric1_window_frac": float(cost_drop.get("window_frac", float("nan"))),
        "metric1_window_size": float(cost_drop.get("window_size", float("nan"))),
        "relative_drop_pct": relative_drop_pct,
        "metric1_trend_slope_theil_sen_ema": float(cost_trend.get("trend_slope_theil_sen_ema", float("nan"))),
        "metric1_late_rebound_pct": float(cost_trend.get("late_rebound_pct", float("nan"))),
        "metric1_trend_ema_span": float(cost_trend.get("ema_span_used", float("nan"))),
        "metric1_source": str(metric1_source),
        "diag_dense_metric1_source": str(det_source),
        "diag_dense_metric1_relative_drop_pct": det_drop_pct,
        "diag_dense_metric1_window_frac": float(det_drop.get("window_frac", float("nan"))),
        "diag_dense_metric1_window_size": float(det_drop.get("window_size", float("nan"))),
        "det_eval_paper_cost_first_5pct_mean": det_main_first,
        "det_eval_paper_cost_last_5pct_mean": det_main_last,
        "det_eval_paper_cost_first_10pct_mean": det_main_first,
        "det_eval_paper_cost_last_10pct_mean": det_main_last,
        "det_eval_paper_cost_first": det_main_first,
        "det_eval_paper_cost_last": det_main_last,
        "det_eval_paper_cost_first_window_mean": det_main_first,
        "det_eval_paper_cost_last_window_mean": det_main_last,
        "det_eval_metric1_window_frac": float(det_main_drop.get("window_frac", float("nan"))),
        "det_eval_metric1_window_size": float(det_main_drop.get("window_size", float("nan"))),
        "det_eval_relative_drop_pct": det_main_drop_pct,
        "det_eval_metric1_source": str(_det_main_source),
        "prox_main_ratio_early_mean": float(
            _h11_safe_float((knob.get("runtime", {}) or {}).get("prox_main_ratio_early_mean", float("nan")), float("nan"))
        ),
        "prox_main_ratio_mid_mean": float(
            _h11_safe_float((knob.get("runtime", {}) or {}).get("prox_main_ratio_mid_mean", float("nan")), float("nan"))
        ),
        "prox_main_ratio_late_mean": float(
            _h11_safe_float((knob.get("runtime", {}) or {}).get("prox_main_ratio_late_mean", float("nan")), float("nan"))
        ),
        "comp_enable_rate_tail_mean": float(
            _h11_safe_float((knob.get("runtime", {}) or {}).get("comp_enable_rate_tail_mean", float("nan")), float("nan"))
        ),
        "comp_enable_rate_tail_lock_frac": float(
            _h11_safe_float((knob.get("runtime", {}) or {}).get("comp_enable_rate_tail_lock_frac", float("nan")), float("nan"))
        ),
    }
    out["artifacts"] = {
        "H11_cfg_dropped_keys_audit": drop,
        "H11_knob_effect_audit": knob,
        "H11_reward_chain_balance": balance,
        "H11_reward_shape_gate_dense": dense,
        "smoke_pass": smoke_pass,
        "short_pass": short_pass,
    }
    return out


