from __future__ import annotations

"""



1)  `configs/env_fixed.yaml` scenario/ris/comm/compute/task/objective/shaping
2) train/eval/calibrate/baseline `JointEnvConfig` /
3) 


- build
-  `configs/env_fixed.yaml`  dropped 
-  `JointEnvConfig` 
"""

from dataclasses import fields as dc_fields
from typing import Any, Dict, List, Optional, Tuple


_CFG_META_CONFLICTS_KEY = "__cfg_conflicts__"


def _same_cfg_value(a: Any, b: Any) -> bool:
    """
    

    
    - 
    - list/tuple 
    - dict 
    """
    if type(a) is not type(b):
        
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) == float(b)
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_same_cfg_value(a[k], b[k]) for k in a.keys())
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_same_cfg_value(x, y) for x, y in zip(a, b))
    return a == b


def _assign_cfg_value(
    out: Dict[str, Any],
    key: str,
    value: Any,
    *,
    source: str,
    conflicts: List[str],
) -> None:
    """
    

    
    - 
    - 
    -  dropped 
    """
    if key not in out:
        out[key] = value
        return
    old = out[key]
    if _same_cfg_value(old, value):
        return
    conflicts.append(f"{key}:override_by_{source}")
    out[key] = value


def flatten_env_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
     `env_fixed.yaml`  `JointEnvConfig` 

    
    -  scenario/ris/comm/compute/task/objective/shaping
    -  JointEnvConfig 
    """
    if not isinstance(cfg, dict):
        return {}

    group_keys = ("scenario", "ris", "comm", "compute", "task", "objective", "shaping")
    has_group = any(k in cfg for k in group_keys)
    if not has_group:
        return dict(cfg)

    out: Dict[str, Any] = {}
    conflicts: List[str] = []

    
    for k, v in cfg.items():
        if k in group_keys:
            continue
        _assign_cfg_value(out, k, v, source="root", conflicts=conflicts)

    sc = cfg.get("scenario", {}) or {}
    _assign_cfg_value(out, "L", sc.get("region_size", sc.get("L")), source="scenario", conflicts=conflicts)
    sc_keys = (
        "M",
        "I",
        "T",
        "dt",
        "H",
        "Rc",
        "K",
        "Vmax",
        "dmin",
        "user_mobility_mode",
        "user_mobility_prob",
        "user_speed",
        "user_speed_sampling_scope",
        "user_speed_levels",
        "user_speed_probs",
        "user_heading_jitter",
        "user_reflect_boundary",
        "user_area_margin_frac",
        "user_fixed_positions",
        "user_position_jitter",
        "user_position_refresh_interval",
        "t6_edge_user_ratio",
        "t6_edge_band_frac",
        "uav_init_mode",
        "uav_fixed_positions",
        
        "noise_bucket_enable",
        "noise_bucket_mode",
        "noise_bucket_levels",
    )
    for k in sc_keys:
        if k in sc:
            _assign_cfg_value(out, k, sc[k], source="scenario", conflicts=conflicts)
    
    sc_consumed = {"region_size", "L", *sc_keys}
    for k, v in sc.items():
        if k not in sc_consumed:
            _assign_cfg_value(out, k, v, source="scenario", conflicts=conflicts)

    ris = cfg.get("ris", {}) or {}
    ris_keys = ("v", "N", "eta", "beta0", "enable_ris", "ris_boost", "num_ris", "ris_positions")
    for k in ris_keys:
        if k in ris:
            _assign_cfg_value(out, k, ris[k], source="ris", conflicts=conflicts)
    for k, v in ris.items():
        if k not in ris_keys:
            _assign_cfg_value(out, k, v, source="ris", conflicts=conflicts)

    comm = cfg.get("comm", {}) or {}
    comm_keys = (
        "B",
        "N0",
        "p",
        "R_min",
        "power_mode",
        "t6_interference_alpha",
        "t6_interference_jitter",
        "t6_direct_blockage_prob",
        "t6_direct_blockage_min_gain",
        "t6_direct_blockage_max_gain",
        "t6_direct_blockage_refresh_each_step",
        
        "enable_comp",
        "comp_coherent",
        "comp_coherence_boost",
        "obs_enable_comp_flag",
        "obs_enable_ris_flag",
    )
    for k in comm_keys:
        if k in comm:
            _assign_cfg_value(out, k, comm[k], source="comm", conflicts=conflicts)
    for k, v in comm.items():
        if k not in comm_keys:
            _assign_cfg_value(out, k, v, source="comm", conflicts=conflicts)

    comp = cfg.get("compute", {}) or {}
    comp_keys = (
        "fmax",
        "theta_min",
        "xi",
        "theta_mode",
        "theta_softmax_tau",
        
        "cpu_slot_time",
        "cpu_util_cap",
        "cpu_queue_beta",
        "cpu_energy_queue_gamma",
        "cpu_scale",
    )
    for k in comp_keys:
        if k in comp:
            _assign_cfg_value(out, k, comp[k], source="compute", conflicts=conflicts)
    for k, v in comp.items():
        if k not in comp_keys:
            _assign_cfg_value(out, k, v, source="compute", conflicts=conflicts)

    task = cfg.get("task", {}) or {}
    if "type" in task:
        _assign_cfg_value(out, "task_type", task["type"], source="task", conflicts=conflicts)
    task_keys = ("D_bits_base", "C_cycles_base", "jitter", "load_scale", "load_alpha_D", "load_alpha_C",
                 "task_arrival_jitter", "task_size_jitter")
    for k in task_keys:
        if k in task:
            _assign_cfg_value(out, k, task[k], source="task", conflicts=conflicts)
    task_consumed = {"type", *task_keys}
    for k, v in task.items():
        if k not in task_consumed:
            _assign_cfg_value(out, k, v, source="task", conflicts=conflicts)

    obj = cfg.get("objective", {}) or {}
    obj_keys = (
        "w_delay",
        "w_energy",
        "T_scale",
        "E_scale",
        "vio_penalty",
        "vio_penalty_mode",
        "violation_penalty_scale",
        "action_score_mode",
        "action_space_mode",
        "association_mode",
        "assoc_soft_temp",
        "comp_score_temp",
        "comp_temp_anneal_enable",
        "comp_temp_stage_0",
        "comp_temp_stage_1",
        "comp_temp_stage_2",
        "comp_score_eval_hard",
        "t6_hier_assoc_solver",
        "t6_hier_assoc_require_scipy",
        "t6_hier_assoc_rebalance_enable",
        "t6_hier_assoc_rebalance_rounds",
        "t6_hier_assoc_rebalance_weight",
        "t6_hier_assoc_rebalance_tol",
        "t6_hier_assoc_load_alpha",
        "t6_hier_assoc_use_task_weight",
        "t6_hier_compute_mode",
        "t6_hier_compute_load_weight",
        "t6_hier_compute_dist_weight",
        "t6_hier_compute_gain_weight",
        "t6_hier_compute_sort_demand_desc",
        "reward_design",
        
        "z_alpha",
        "z_beta",
        "z_gamma",
        "z_cost_ref",
        "z_cost_ref_by_load",
        "z_cost_ref_by_speed",
        "z5_r_max",
        "z5_kappa",
        "z5_anchor_norm",
        
        "z4_reward_offset",
        "z4_reward_alpha",
        "z4_bonus_gamma",
        "z4_bonus_anchor",
        "z4_bonus_power",
        "z4_bonus_deadzone",
        "z4_gap_enable",
        "z4_gap_lambda",
        "z4_gap_clip",
        "z4_gap_warmup_start_frac",
        "z4_gap_warmup_end_frac",
        "z4_gap_anneal_start_frac",
        "z4_gap_anneal_end_frac",
        "z4_gap_anneal_floor",
        "z4_gap_ema_enable",
        "z4_gap_ema_beta",
        "z4_gap_grad_ratio_max",
        "z4_gap_lambda_hard_cap",
        "z4_gap_shape_guard_enable",
        "z4_gap_shape_guard_min_steps",
        "z4_gap_shape_guard_tol",
        "z4_gap_shape_guard_patience",
        "z4_action_mode",
        "z4_assoc_stage_enable",
        "z4_assoc_stage_start_frac",
        "z4_assoc_stage_end_frac",
        "z4_assoc_stage_policy_min",
        "z4_assoc_stage_policy_max",
        "z4_assoc_stage_smoothstep",
        "z4_assoc_stage_score_clip",
        "comp_rule_threshold",
        "z4_comp_meta_enable",
        "z4_comp_meta_thr_delta_max",
        "z4_comp_meta_thr_min",
        "z4_comp_meta_thr_max",
        "z4_comp_meta_score_width",
        "z4_comp_meta_temp_scale_delta",
        "z4_comp_meta_temp_scale_min",
        "z4_comp_meta_temp_scale_max",
        "z4_comp_meta_ema_beta",
        "z4_comp_meta_warmup_start_frac",
        "z4_comp_meta_warmup_end_frac",
        
        "paper_reward_mode",
        "paper_reward_a",
        "paper_reward_b",
        "lambda_v",
        "lambda_smooth",
        "lambda_explore",
        "beta_comp_max",
        "beta_ris_max",
        "beta_warmup_frac",
        "beta_hold_frac",
        "beta_end_frac",
        "train_total_steps",
        "phaseg_v2_enable",
        "phaseg_v2_paper_abs_weight",
        "phaseg_v2_paper_delta_weight",
        "phaseg_v2_paper_adv_weight",
        "phaseg_v2_paper_adv_temp",
        "phaseg_v2_delta_clip",
        "phaseg_v2_cost_ref_init",
        "phaseg_v2_cost_ref_min",
        "phaseg_v2_cost_ref_ema_beta",
        "phaseg_v2_lambda_adapt",
        "phaseg_v2_lambda_lr",
        "phaseg_v2_lambda_target",
        "phaseg_v2_lambda_min",
        "phaseg_v2_lambda_max",
        
        "phaseh_cost_norm_enable",
        "phaseh_cost_norm_ref",
        "phaseh_cost_norm_floor",
        "phaseh_cost_norm_power",
        "phaseh_aux_constraint_scale",
        "phaseh_boundary_guard_frac",
        "phaseh_boundary_eval_band_frac",
        "phaseh_boundary_guard_dynamic",
        "phaseh_boundary_guard_low",
        "phaseh_boundary_guard_high",
        "phaseh_boundary_guard_ema_beta",
        "phaseh_reflect_on_wall",
        "phaseh_aux_traj_mix",
        "phaseh_aux_safety_mix",
        "phaseh_aux_gate_enable",
        "phaseh_aux_idle_target",
        "phaseh_aux_boundary_margin_target",
        "phaseh_aux_gate_power",
        "phaseh_comp_bonus_scale",
        "phaseh_ris_bonus_scale",
        "phaseh_shaping_soft_gate_mix",
        "phaseh_shaping_gate_floor",
        "phaseh_shaping_gate_simple",
        
        "phaseh_shaping_gate_bypass",
        "phaseh_ris_allow_negative",
        "phaseh_ris_baseline",
        "phaseh_traj_gain_coupling_enable",
        "phaseh_traj_comp_coupling_weight",
        "phaseh_traj_ris_coupling_weight",
        "phaseh_ris_coupling_baseline",
        "reward_boundary_potential_weight",
        "reward_boundary_potential_scale",
        "reward_paper_delta_scale",
        "reward_paper_cost_weight",
        "reward_violation_weight",
        "reward_wall_weight",
        "reward_wall_margin_weight",
        "reward_uncovered_weight",
        "reward_coverage_margin_weight",
        "reward_collision_margin_weight",
        "reward_movement_weight",
        "reward_movement_target_enable",
        "reward_movement_target",
        "reward_boundary_stick_weight",
        "reward_safe_clip_weight",
        "reward_user_nn_weight",
        "reward_user_centroid_weight",
        "reward_switchback_weight",
        "phaseh_traj_direct_enable",
        "phaseh_traj_direct_scale",
        "reward_potential_weight",
        "reward_step_bias",
        
        "reward_proximity_weight",
        "reward_proximity_weight_final",
        "reward_proximity_decay_start_step",
        "reward_proximity_decay_end_step",
        "reward_proximity_mode",
        "coverage_floor_weight",
        "coverage_floor_threshold",
        "reward_user_area_gap_weight",
        "obs_enable_comp_flag",
        "obs_enable_ris_flag",
        
        "obs_enable_rate_feedback",
        "obs_enable_cost_feedback",
        "obs_enable_prev_action",
        "use_fixed_normalization",
        "norm_update_interval",
        "reward_clip_range",
        "reward_ema_beta",
        "z_load_balance_alpha",
        "z_capacity_bias_alpha",
        
        "paper_cost_weight",
        "paper_cost_abs_weight",
        "lambda_comp_usage",
        "lambda_comp_switch",
        "comp_switch_max",
        "comp_switch_bonus_mode",
        "lambda_ris_usage",
        "lambda_velocity_penalty",
        "velocity_threshold",
        "velocity_penalty_mode",
        "lambda_boundary_penalty",
        "boundary_margin",
        "lambda_idle_penalty",
        "idle_threshold",
        "relative_baseline_mode",
        "relative_baseline_position_mode",
        "vio_max_expected",
        "lambda_traj_guide",
        "traj_serve_threshold",
        "improvement_episode_bonus_weight",
        "reward_scale_factor",
        
        "t6_beta_potential",
        "t6_ref_cost_mode",
        "t6_cost_scale_ema_beta",
        "t6_traj_bonus_weight",
        "t6_potential_gamma",
        "t6_main_weight_start",
        "t6_main_weight_end",
        "t6_potential_scale_start",
        "t6_potential_scale_end",
        "t6_traj_scale_start",
        "t6_traj_scale_end",
        "t6_anneal_start_frac",
        "t6_anneal_end_frac",
        "t6_pid_enable",
        "t6_pid_kp",
        "t6_pid_ki",
        "t6_pid_kd",
        "t6_pid_delta",
        "t6_pid_update_interval",
        "t6_lambda_min",
        "t6_lambda_max",
        "t6_pid_violation_ema_beta",
        "t6_pid_deadband",
        "t6_pid_lambda_step_clip",
        "t6_pid_integral_decay",
        "t6_pid_freeze_after_frac",
        "t6_metric_calib_enable",
        "t6_metric_calib_delay_ref",
        "t6_metric_calib_energy_ref",
        "t6_metric_calib_clip",
        "t6_metric_calib_mix",
    )
    for k in obj_keys:
        if k in obj:
            _assign_cfg_value(out, k, obj[k], source="objective", conflicts=conflicts)

    
    reward_blk = obj.get("reward", {}) if isinstance(obj, dict) else {}
    if isinstance(reward_blk, dict):
        t6_blk = reward_blk.get("t6", {})
        if isinstance(t6_blk, dict):
            t6_reward_map = {
                "beta_potential": "t6_beta_potential",
                "ref_cost_mode": "t6_ref_cost_mode",
                "cost_scale_ema_beta": "t6_cost_scale_ema_beta",
                "traj_bonus_weight": "t6_traj_bonus_weight",
                "potential_gamma": "t6_potential_gamma",
                "main_weight_start": "t6_main_weight_start",
                "main_weight_end": "t6_main_weight_end",
                "potential_scale_start": "t6_potential_scale_start",
                "potential_scale_end": "t6_potential_scale_end",
                "traj_scale_start": "t6_traj_scale_start",
                "traj_scale_end": "t6_traj_scale_end",
                "anneal_start_frac": "t6_anneal_start_frac",
                "anneal_end_frac": "t6_anneal_end_frac",
                "metric_calib_enable": "t6_metric_calib_enable",
                "metric_calib_delay_ref": "t6_metric_calib_delay_ref",
                "metric_calib_energy_ref": "t6_metric_calib_energy_ref",
                "metric_calib_clip": "t6_metric_calib_clip",
                "metric_calib_mix": "t6_metric_calib_mix",
            }
            for src_k, dst_k in t6_reward_map.items():
                if src_k in t6_blk:
                    _assign_cfg_value(out, dst_k, t6_blk[src_k], source="objective.reward.t6", conflicts=conflicts)

    
    cons_blk = obj.get("constraint", {}) if isinstance(obj, dict) else {}
    if isinstance(cons_blk, dict):
        pid_blk = cons_blk.get("pid", {})
        if isinstance(pid_blk, dict):
            t6_pid_map = {
                "enable": "t6_pid_enable",
                "kp": "t6_pid_kp",
                "ki": "t6_pid_ki",
                "kd": "t6_pid_kd",
                "delta": "t6_pid_delta",
                "update_interval": "t6_pid_update_interval",
                "lambda_min": "t6_lambda_min",
                "lambda_max": "t6_lambda_max",
                "violation_ema_beta": "t6_pid_violation_ema_beta",
                "deadband": "t6_pid_deadband",
                "lambda_step_clip": "t6_pid_lambda_step_clip",
                "integral_decay": "t6_pid_integral_decay",
                "freeze_after_frac": "t6_pid_freeze_after_frac",
            }
            for src_k, dst_k in t6_pid_map.items():
                if src_k in pid_blk:
                    _assign_cfg_value(out, dst_k, pid_blk[src_k], source="objective.constraint.pid", conflicts=conflicts)
    for k, v in obj.items():
        if k not in obj_keys and k not in ("reward", "constraint"):
            _assign_cfg_value(out, k, v, source="objective", conflicts=conflicts)

    
    root_reward = cfg.get("reward", {}) or {}
    if isinstance(root_reward, dict):
        root_t6 = root_reward.get("t6", {})
        if isinstance(root_t6, dict):
            t6_reward_map = {
                "beta_potential": "t6_beta_potential",
                "ref_cost_mode": "t6_ref_cost_mode",
                "cost_scale_ema_beta": "t6_cost_scale_ema_beta",
                "traj_bonus_weight": "t6_traj_bonus_weight",
                "potential_gamma": "t6_potential_gamma",
                "main_weight_start": "t6_main_weight_start",
                "main_weight_end": "t6_main_weight_end",
                "potential_scale_start": "t6_potential_scale_start",
                "potential_scale_end": "t6_potential_scale_end",
                "traj_scale_start": "t6_traj_scale_start",
                "traj_scale_end": "t6_traj_scale_end",
                "anneal_start_frac": "t6_anneal_start_frac",
                "anneal_end_frac": "t6_anneal_end_frac",
                "metric_calib_enable": "t6_metric_calib_enable",
                "metric_calib_delay_ref": "t6_metric_calib_delay_ref",
                "metric_calib_energy_ref": "t6_metric_calib_energy_ref",
                "metric_calib_clip": "t6_metric_calib_clip",
                "metric_calib_mix": "t6_metric_calib_mix",
            }
            for src_k, dst_k in t6_reward_map.items():
                if src_k in root_t6:
                    _assign_cfg_value(out, dst_k, root_t6[src_k], source="reward.t6", conflicts=conflicts)

    root_constraint = cfg.get("constraint", {}) or {}
    if isinstance(root_constraint, dict):
        root_pid = root_constraint.get("pid", {})
        if isinstance(root_pid, dict):
            t6_pid_map = {
                "enable": "t6_pid_enable",
                "kp": "t6_pid_kp",
                "ki": "t6_pid_ki",
                "kd": "t6_pid_kd",
                "delta": "t6_pid_delta",
                "update_interval": "t6_pid_update_interval",
                "lambda_min": "t6_lambda_min",
                "lambda_max": "t6_lambda_max",
                "violation_ema_beta": "t6_pid_violation_ema_beta",
                "deadband": "t6_pid_deadband",
                "lambda_step_clip": "t6_pid_lambda_step_clip",
                "integral_decay": "t6_pid_integral_decay",
                "freeze_after_frac": "t6_pid_freeze_after_frac",
            }
            for src_k, dst_k in t6_pid_map.items():
                if src_k in root_pid:
                    _assign_cfg_value(out, dst_k, root_pid[src_k], source="constraint.pid", conflicts=conflicts)

    shp = cfg.get("shaping", {}) or {}
    shp_keys = (
        "comp_gate_ratio",
        "comp_gate_snr_enable",
        "comp_gate_snr_margin",
        "weight_mode",
        "eff_wE_min",
        "eff_wT_min",
        "improve_alpha",
        "theta_mode",
        "theta_softmax_tau",
        "reward_cost_mode",
        "assoc_gain_mix",
        "use_fixed_normalization",
        "norm_update_interval",
        "reward_clip_range",
        "reward_ema_beta",
        "norm_ema_beta",
        "norm_clip",
        "dyn_beta",
        "dyn_eta",
        "dyn_w_min",
        "dyn_w_max",
        "dyn_margin",
        "dyn_update_every",
    )
    for k in shp_keys:
        if k in shp:
            _assign_cfg_value(out, k, shp[k], source="shaping", conflicts=conflicts)
    for k, v in shp.items():
        if k not in shp_keys:
            _assign_cfg_value(out, k, v, source="shaping", conflicts=conflicts)

    if conflicts:
        out[_CFG_META_CONFLICTS_KEY] = list(dict.fromkeys(conflicts))

    return {k: v for k, v in out.items() if v is not None}


def clean_joint_env_cfg(env_cfg_flat: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
     `JointEnvConfig` 
    """
    allowed = {f.name for f in dc_fields(_lazy_import_joint_env_config())}
    clean = {k: v for k, v in (env_cfg_flat or {}).items() if k in allowed and v is not None}
    dropped = sorted(set((env_cfg_flat or {}).keys()) - set(clean.keys()))
    return clean, dropped


def build_joint_env_cfg(
    cfg_env: Dict[str, Any],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    strict_conflicts: bool = True,
) -> Tuple["JointEnvConfig", Dict[str, Any], List[str]]:
    """
     yaml  dict  JointEnvConfig
    - env_cfgJointEnvConfig 
    - env_cfg_dict_clean
    - dropped_keys
    strict_conflicts=True 
    """
    flat = flatten_env_cfg(cfg_env)
    cfg_conflicts = flat.pop(_CFG_META_CONFLICTS_KEY, [])
    if bool(strict_conflicts) and isinstance(cfg_conflicts, list) and cfg_conflicts:
        show = ", ".join([str(x) for x in cfg_conflicts[:6]])
        extra = ""
        if len(cfg_conflicts) > 6:
            extra = f" ...(+{len(cfg_conflicts) - 6})"
        raise ValueError(
            ""
            f"{show}{extra}"
            ""
        )
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                flat[k] = v
    clean, dropped = clean_joint_env_cfg(flat)
    if isinstance(cfg_conflicts, list) and cfg_conflicts:
        dropped = sorted(set(dropped + [f"conflict::{str(x)}" for x in cfg_conflicts]))
    JointEnvConfig = _lazy_import_joint_env_config()
    return JointEnvConfig(**clean), clean, dropped


def _lazy_import_joint_env_config():
    """
    
    - src.envs.comp_ris_env_joint  src.utils.*
    """
    from src.envs.comp_ris_env_joint import JointEnvConfig  # local import

    return JointEnvConfig
