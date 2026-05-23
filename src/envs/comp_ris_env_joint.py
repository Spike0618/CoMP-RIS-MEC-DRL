from __future__ import annotations

"""
CoMP + RIS +  MEC 
"""

from dataclasses import dataclass, field
import hashlib
from typing import Optional, Any, Dict, List, Tuple
import numpy as np

from src.utils.geometry import clamp_positions, pairwise_dist2

try:
    
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except Exception:
    _scipy_linear_sum_assignment = None


def _clip_pos(q: np.ndarray, L: float) -> np.ndarray:
    return clamp_positions(q, L)


def _normalize_w(wT: float, wE: float, eps: float = 1e-12) -> Tuple[float, float]:
    wT = float(wT)
    wE = float(wE)
    s = wT + wE
    if s <= eps:
        return 0.5, 0.5
    return wT / s, wE / s

def _clamp_weights(wT: float, wE: float, wT_min: float = 0.0, wE_min: float = 0.0, eps: float = 1e-12):
    """wT+wE=1 (wT, wE, clamped_flag)"""
    wT = float(wT); wE = float(wE)
    wT_min = max(0.0, float(wT_min)); wE_min = max(0.0, float(wE_min))

    if wT_min <= eps and wE_min <= eps:
        wTn, wEn = _normalize_w(wT, wE, eps=eps)
        return wTn, wEn, False

    smin = wT_min + wE_min
    if smin >= 1.0 - 1e-9:
        
        wT = wT_min / max(smin, eps)
        wE = wE_min / max(smin, eps)
        return wT, wE, True

    
    wT, wE = _normalize_w(wT, wE, eps=eps)

    clamped = False
    wT2 = max(wT, wT_min)
    wE2 = max(wE, wE_min)
    if (wT2 != wT) or (wE2 != wE):
        clamped = True

    s = wT2 + wE2
    if s > 1.0:
        clamped = True
        wT2, wE2 = wT2 / s, wE2 / s

    
    if wT2 < wT_min:
        clamped = True
        wT2 = wT_min
        wE2 = 1.0 - wT2
    if wE2 < wE_min:
        clamped = True
        wE2 = wE_min
        wT2 = 1.0 - wE2

    wT2, wE2 = _normalize_w(wT2, wE2, eps=eps)
    return wT2, wE2, clamped

# ---- ----
from src.utils.constraints import (
    project_speed,
    repair_collisions,
    ensure_theta,
    ensure_theta_feasible,
    alloc_theta_solver,
    ensure_topk_feasible,
    enforce_min_coverage,
)

# ---- ----
from src.utils.channel import (
    gain_direct,
    gain_user_ris,
    gain_ris_uav,
    comp_gain,
    snr_from_comp,
    rate_from_snr,
)
from src.utils.t6_lagrangian import PidLagrangeController
from src.utils.t6_reward import compute_t6_reward, update_ema_mean_var

# ---- / ----
from src.envs.scenario import init_positions
from src.envs.task_model import ConstantTaskModel, UniformTaskModel, LogNormalTaskModel


# -------------------------

# -------------------------
class RunningNorm:
    """
    /
    """

    def __init__(self, ema_beta: float = 0.99, eps: float = 1e-8, clip: float = 5.0):
        self.beta = float(ema_beta)
        self.eps = float(eps)
        self.clip = float(clip)
        self._inited = False
        self.mean = 0.0
        self.var = 1.0

    def update(self, x: float) -> None:
        x = float(x)
        if not self._inited:
            self.mean = x
            self.var = 1.0
            self._inited = True
            return
        m = self.mean
        v = self.var
        beta = self.beta
        m_new = beta * m + (1.0 - beta) * x
        
        second = beta * (v + m * m) + (1.0 - beta) * (x * x)
        v_new = max(second - m_new * m_new, self.eps)
        self.mean = m_new
        self.var = v_new

    def std(self) -> float:
        return float(np.sqrt(self.var + self.eps))

    def norm(self, x: float) -> float:
        if not self._inited:
            return 0.0
        z = (float(x) - self.mean) / (self.std() + self.eps)
        if self.clip > 0:
            z = float(np.clip(z, -self.clip, self.clip))
        return z


@dataclass
class JointEnvConfig:
    # -----------------
    
    # -----------------
    L: Optional[float] = None
    M: Optional[int] = None
    I: Optional[int] = None
    T: Optional[int] = None
    dt: Optional[float] = None
    H: Optional[float] = None
    Rc: Optional[float] = None
    K: Optional[int] = None
    Vmax: Optional[float] = None
    dmin: Optional[float] = None
    # User mobility scenario:
    # - static: users stay fixed in one episode
    # - slow: users move with constant slow speed and boundary reflection
    # - mixed: sample slow/static per episode by user_mobility_prob
    user_mobility_mode: str = "static"  
    user_mobility_prob: float = 0.30
    user_speed: float = 0.30  # m/s; recommended range: 0.2 to 0.5
    
    
    
    user_speed_sampling_scope: str = "episode"  # 'episode' | 'per_user'
    
    user_speed_levels: Optional[List[float]] = None
    
    user_speed_probs: Optional[List[float]] = None
    user_heading_jitter: float = 0.0  # rad/step, 0 => straight line
    user_reflect_boundary: bool = True
    # Optional inner mobility box for users: users are sampled/move in [m, L-m]^2 where m = frac*L.
    # 0.0 keeps legacy behavior (full [0, L]^2).
    user_area_margin_frac: float = 0.0
    
    
    
    user_fixed_positions: Optional[List[List[float]]] = None
    
    user_position_jitter: float = 0.0  
    user_position_refresh_interval: int = 999999  
    
    t6_edge_user_ratio: float = 0.0
    t6_edge_band_frac: float = 0.15
    
    uav_init_mode: str = "random"  
    uav_fixed_positions: Optional[List[List[float]]] = None  # [[x1,y1], ..., [xM,yM]]
    
    noise_bucket_enable: bool = False
    noise_bucket_mode: str = "round_robin"  # "round_robin" | "random"
    noise_bucket_levels: Optional[List[float]] = None  
    
    task_arrival_jitter: float = 0.0  
    task_size_jitter: float = 0.0  

    # -----------------
    
    # -----------------
    v: Optional[np.ndarray] = None
    N: Optional[int] = None
    eta: Optional[float] = None
    beta0: Optional[float] = None
    enable_ris: bool = True
    
    
    
    ris_boost: float = 1.0
    
    num_ris: int = 1  
    ris_positions: Optional[List[List[float]]] = None  

    # -----------------
    
    # -----------------
    enable_comp: bool = True
    
    comp_coherent: bool = True
    comp_coherence_boost: float = 2.2
    
    
    
    power_mode: str = "per_uav"
    
    t6_interference_alpha: float = 0.0
    t6_interference_jitter: float = 0.0
    t6_direct_blockage_prob: float = 0.0
    t6_direct_blockage_min_gain: float = 0.35
    t6_direct_blockage_max_gain: float = 0.80
    t6_direct_blockage_refresh_each_step: bool = False
    
    
    
    
    comp_gate_ratio: float = 0.0
    
    
    z1_comp_gate_ratio_fallback: float = 0.35
    comp_gate_snr_enable: bool = False
    comp_gate_snr_margin: float = 0.0
    
    obs_enable_comp_flag: Optional[float] = None
    obs_enable_ris_flag: Optional[float] = None
    
    
    
    
    
    
    obs_enable_rate_feedback: bool = False
    obs_enable_cost_feedback: bool = False
    obs_enable_prev_action: bool = False

    # -----------------
    
    # -----------------
    B: float = 1.0
    N0: float = 1.0
    p: float = 1.0
    R_min: float = 0.0

    # -----------------
    # / MEC
    # -----------------
    fmax: Optional[np.ndarray] = field(default=None)
    theta_min: Optional[float] = None

    xi: float = 1.0e-28

    # CPU / load_scale
    cpu_slot_time: float = 1.0
    cpu_util_cap: float = 0.95
    cpu_queue_beta: float = 1.0
    cpu_energy_queue_gamma: float = 1.0
    cpu_scale: float = 1.0
    
    
    z_load_balance_alpha: float = 0.0
    z_capacity_bias_alpha: float = 0.0

    # -----------------
    # / aper_cost
    # -----------------
    w_delay: float = 0.5
    w_energy: float = 0.5
    T_scale: float = 1.0
    E_scale: float = 1.0
    vio_penalty: float = 1.0
    vio_penalty_mode: str = "pre"  # 'pre' | 'post'
    violation_penalty_scale: float = 0.1  

    # -----------------
    
    # -----------------
    improve_alpha: float = 0.1
    
    
    
    
    theta_mode: str = "solver"        # 'solver' | 'agent' | 'policy'
    
    theta_softmax_tau: float = 0.3
    reward_cost_mode: str = "p1"      # 'paper' | 'relative' | 'p1'
    # eparate= s_a _z hared= _a s_z
    action_score_mode: str = "separate"  # 'separate' | 'shared'
    
    
    
    action_space_mode: str = "full"  # 'full' | 'hierarchical'
    
    association_mode: str = "hard"  # 'hard' | 'soft'
    assoc_soft_temp: float = 0.50   # soft hard
    assoc_gain_mix: float = 0.05
    
    comp_score_temp: float = 0.5
    
    comp_temp_anneal_enable: bool = False
    comp_temp_stage_0: float = 0.25
    comp_temp_stage_1: float = 0.18
    comp_temp_stage_2: float = 0.12
    
    comp_score_eval_hard: bool = True
    
    t6_hier_assoc_solver: str = "greedy"  # greedy | hungarian
    
    
    t6_hier_assoc_require_scipy: bool = True
    
    t6_hier_assoc_rebalance_enable: bool = False
    t6_hier_assoc_rebalance_rounds: int = 2
    t6_hier_assoc_rebalance_weight: float = 1.0
    t6_hier_assoc_rebalance_tol: float = 0.05
    t6_hier_assoc_load_alpha: float = 0.10
    t6_hier_assoc_use_task_weight: bool = True
    t6_hier_compute_mode: str = "load_aware"  # primary | load_aware
    t6_hier_compute_load_weight: float = 1.00
    t6_hier_compute_dist_weight: float = 0.15
    t6_hier_compute_gain_weight: float = 0.10
    t6_hier_compute_sort_demand_desc: bool = True

    
    # - layered_v1 + + + +
    # - paper_delta_v2 paper_cost eward
    
    
    
    reward_design: str = "layered_v1"  # 'layered_v1' | 'paper_delta_v2' | 'reward_total_v1' | 't6_potential_lagrangian' | 't7_absolute' | 'z_shifted_log' | 'z4_linear' | 'z5_saturating'
    # T7reward_main = -t7_alpha * f(paper_cost / t7_cost_ref)
    
    t7_alpha: float = 2.0
    t7_cost_ref: float = 1.0
    t7_cost_transform: str = "linear"  # 'linear' | 'log'
    
    z_alpha: float = 3.0
    z_beta: float = 3.5
    z_gamma: float = 1.2
    z_cost_ref: float = 1.287
    
    
    z_cost_ref_by_load: Optional[Dict[str, float]] = None
    
    z_cost_ref_by_speed: Optional[Dict[str, float]] = None
    
    z5_r_max: float = 2.0
    z5_kappa: float = 15.0
    z5_anchor_norm: float = 1.15
    
    # r_t = (offset - cost/C_ref) + max(0, (anchor_cost - cost)/anchor_cost)^p
    z4_reward_offset: float = 1.75
    z4_reward_alpha: float = 1.5
    
    # anchor_cost = z_cost_ref * z4_bonus_anchor
    z4_bonus_gamma: float = 150.0
    z4_bonus_anchor: float = 1.15
    z4_bonus_power: float = 2.0
    
    z4_bonus_deadzone: float = 0.0
    
    z4_gap_enable: bool = False
    z4_gap_lambda: float = 0.0
    z4_gap_clip: float = 0.30
    
    z4_gap_warmup_start_frac: float = 0.02
    z4_gap_warmup_end_frac: float = 0.10
    z4_gap_anneal_start_frac: float = 0.70
    z4_gap_anneal_end_frac: float = 0.95
    z4_gap_anneal_floor: float = 0.35
    
    z4_gap_ema_enable: bool = True
    z4_gap_ema_beta: float = 0.85
    
    z4_gap_grad_ratio_max: float = 0.25
    z4_gap_lambda_hard_cap: float = 2.0
    
    z4_gap_shape_guard_enable: bool = True
    z4_gap_shape_guard_min_steps: int = 2000
    z4_gap_shape_guard_tol: float = 0.01
    z4_gap_shape_guard_patience: int = 6
    
    
    
    z4_action_mode: str = "hierarchical"
    
    
    z4_assoc_stage_enable: bool = False
    z4_assoc_stage_start_frac: float = 0.25
    z4_assoc_stage_end_frac: float = 0.65
    z4_assoc_stage_policy_min: float = 0.0
    z4_assoc_stage_policy_max: float = 1.0
    z4_assoc_stage_smoothstep: bool = True
    z4_assoc_stage_score_clip: float = 1.0
    
    comp_rule_threshold: float = 0.5
    
    
    z4_comp_meta_enable: bool = False
    z4_comp_meta_thr_delta_max: float = 0.10
    z4_comp_meta_thr_min: float = 0.05
    z4_comp_meta_thr_max: float = 0.70
    
    z4_comp_meta_score_width: float = 0.20
    
    z4_comp_meta_temp_scale_delta: float = 0.35
    z4_comp_meta_temp_scale_min: float = 0.65
    z4_comp_meta_temp_scale_max: float = 1.35
    
    z4_comp_meta_ema_beta: float = 0.80
    z4_comp_meta_warmup_start_frac: float = 0.00
    z4_comp_meta_warmup_end_frac: float = 0.08

    # -----------------
    
    # -----------------
    # paper_reward_step eward_paper = a - b * f(paper_cost)
    # - linear (x)=x
    
    paper_reward_mode: str = "log"  # 'log' | 'linear'
    paper_reward_a: float = 10.0
    paper_reward_b: float = 10.0
    # reward_constraint = -lambda_v * vio_signal
    lambda_v: float = 8.0
    
    vio_max_expected: float = 5.0
    
    lambda_traj_guide: float = 0.0
    
    traj_serve_threshold: float = 0.10
    
    lambda_smooth: float = 0.5      
    lambda_explore: float = 0.3     
    explore_move_ratio: float = 0.0  
    lambda_move: float = 0.0        
    
    paper_cost_weight: float = 1.0  
    
    paper_cost_abs_weight: float = 0.0
    
    
    
    reward_scale_factor: float = 1.0
    lambda_comp_usage: float = 0.0  
    lambda_comp_switch: float = 0.0  
    comp_switch_max: float = 10.0  
    comp_switch_bonus_mode: str = "linear"  
    lambda_ris_usage: float = 0.0  
    lambda_velocity_penalty: float = 0.0  
    velocity_threshold: float = 0.1  
    velocity_penalty_mode: str = "linear"  
    lambda_boundary_penalty: float = 0.0  
    boundary_margin: float = 30.0  
    lambda_idle_penalty: float = 0.0  
    idle_threshold: float = 0.05  

    
    # R = -paper_cost_weight * (paper_cost - baseline_balanced_cost) - lambda_v * vio_any
    
    relative_baseline_mode: str = "balanced"  
    
    relative_baseline_position_mode: str = "pre_move"  # 'pre_move' | 'post_move'

    
    running_mean_window: int = 100  
    improvement_step_weight: float = 0.1  
    improvement_episode_bonus_weight: float = 1.0  

    
    t6_beta_potential: float = 1.0
    t6_ref_cost_mode: str = "balanced"  # balanced | greedy_delay | greedy_energy | no_comp
    t6_cost_scale_ema_beta: float = 0.95
    t6_traj_bonus_weight: float = 0.10
    t6_potential_gamma: float = 0.99
    
    t6_main_weight_start: float = 1.0
    t6_main_weight_end: float = 1.3
    t6_potential_scale_start: float = 1.0
    t6_potential_scale_end: float = 0.4
    t6_traj_scale_start: float = 1.0
    t6_traj_scale_end: float = 0.3
    t6_anneal_start_frac: float = 0.10
    t6_anneal_end_frac: float = 0.80
    t6_pid_enable: bool = True
    t6_pid_kp: float = 0.05
    t6_pid_ki: float = 0.01
    t6_pid_kd: float = 0.00
    t6_pid_delta: float = 0.20
    
    t6_pid_update_interval: int = 2
    t6_lambda_min: float = 0.0
    t6_lambda_max: float = 20.0
    t6_pid_violation_ema_beta: float = 0.8
    t6_pid_deadband: float = 0.02
    t6_pid_lambda_step_clip: float = 0.25
    t6_pid_integral_decay: float = 1.0
    t6_pid_freeze_after_frac: float = 0.90
    
    t6_metric_calib_enable: bool = False
    t6_metric_calib_delay_ref: float = 1.0
    t6_metric_calib_energy_ref: float = 1.0
    t6_metric_calib_clip: float = 4.0
    t6_metric_calib_mix: float = 1.0

    # CoMP/RIS shaping eward_comp/ris = gate * beta(t) * bonus_raw
    beta_comp_max: float = 0.30
    beta_ris_max: float = 0.30
    # warmup(0~w)->max old(w~w+h)->max nneal(w+h~1)->end_frac*max
    beta_warmup_frac: float = 0.10
    beta_hold_frac: float = 0.50
    beta_end_frac: float = 0.0
    
    train_total_steps: int = 0
    # PhaseG-v2 +
    # reward_paper_step = w_abs * r_abs(paper_cost_t) + w_delta * clip((C_{t-1}-C_t)/C_ref, -clip, +clip)
    phaseg_v2_enable: bool = False
    phaseg_v2_paper_abs_weight: float = 1.0
    phaseg_v2_paper_delta_weight: float = 2.0
    phaseg_v2_paper_adv_weight: float = 0.5
    phaseg_v2_paper_adv_temp: float = 1.0
    phaseg_v2_delta_clip: float = 1.0
    phaseg_v2_cost_ref_init: float = 5.0
    phaseg_v2_cost_ref_min: float = 0.5
    phaseg_v2_cost_ref_ema_beta: float = 0.90
    # Optional paper-reward shaping controls.
    phaseh_cost_norm_enable: bool = False
    phaseh_cost_norm_ref: float = 0.6
    phaseh_cost_norm_floor: float = 0.2
    phaseh_cost_norm_power: float = 0.0
    # reward_constraint = -lambda_v_eff * constraint_signal
    phaseg_v2_lambda_adapt: bool = False
    phaseg_v2_lambda_lr: float = 0.01
    phaseg_v2_lambda_target: float = 0.35
    phaseg_v2_lambda_min: float = 0.0
    phaseg_v2_lambda_max: float = 20.0
    
    
    phaseh_aux_constraint_scale: float = 0.0
    
    phaseh_boundary_guard_frac: float = 0.0
    phaseh_boundary_eval_band_frac: float = 0.05
    phaseh_boundary_guard_min_from_eval_band: bool = True
    phaseh_boundary_guard_band_buffer: float = 1.0
    phaseh_boundary_guard_dynamic: bool = False
    
    phaseh_boundary_guard_low: float = 0.30
    phaseh_boundary_guard_high: float = 0.45
    phaseh_boundary_guard_ema_beta: float = 0.95
    
    phaseh_reflect_on_wall: bool = False
    phaseh_uav_operating_margin_frac: float = 0.0
    phaseh_uav_follow_user_area: bool = False
    
    phaseh_aux_traj_mix: float = 1.0
    phaseh_aux_safety_mix: float = 1.0
    phaseh_aux_gate_enable: bool = False
    phaseh_aux_idle_target: float = 0.60
    phaseh_aux_boundary_margin_target: float = 0.25
    phaseh_aux_gate_power: float = 1.0
    reward_paper_delta_scale: float = 200.0
    # paper_cost
    reward_paper_cost_weight: float = 0.0
    reward_violation_weight: float = 4.0
    
    reward_wall_weight: float = 0.0
    
    reward_wall_margin_weight: float = 0.0
    reward_uncovered_weight: float = 6.0
    
    reward_coverage_margin_weight: float = 0.0
    reward_collision_margin_weight: float = 3.0
    reward_movement_weight: float = 1.0
    reward_movement_target_enable: bool = False
    reward_movement_target: float = 0.35
    
    reward_boundary_stick_weight: float = 0.0
    
    reward_safe_clip_weight: float = 0.0
    reward_user_nn_weight: float = 0.0
    reward_user_centroid_weight: float = 0.0
    reward_user_area_gap_weight: float = 0.0
    reward_switchback_weight: float = 0.0
    
    phaseh_traj_direct_enable: bool = False
    phaseh_traj_direct_scale: float = 1.0
    phaseh_traj_direct_pressure_gain: float = 0.0
    phaseh_traj_direct_pressure_target: float = 0.35
    phaseh_comp_bonus_scale: float = 1.0
    phaseh_ris_bonus_scale: float = 1.0
    phaseh_shaping_soft_gate_mix: float = 0.0
    phaseh_shaping_gate_floor: float = 0.0
    
    phaseh_shaping_gate_simple: bool = False
    
    phaseh_shaping_gate_bypass: bool = False  
    phaseh_ris_allow_negative: bool = False   
    phaseh_ris_baseline: float = 0.15         
    
    phaseh_ris_quality_gate_enable: bool = False
    phaseh_ris_quality_gate_floor: float = 0.0
    phaseh_ris_quality_gate_power: float = 1.0
    phaseh_traj_gain_coupling_enable: bool = False  
    phaseh_traj_comp_coupling_weight: float = 0.0   
    phaseh_traj_ris_coupling_weight: float = 0.0    
    phaseh_ris_coupling_baseline: float = 0.15      
    reward_boundary_potential_weight: float = 0.0   
    reward_boundary_potential_scale: float = 0.1    
    reward_potential_weight: float = 2.0
    reward_step_bias: float = 0.0
    
    reward_proximity_weight: float = 0.0
    reward_proximity_weight_final: float = 0.0
    reward_proximity_decay_start_step: int = 0
    reward_proximity_decay_end_step: int = 0
    reward_proximity_mode: str = "delta"  
    coverage_floor_weight: float = 0.0
    coverage_floor_threshold: float = 0.8

    
    paper_cost_weight: float = 1.0  
    lambda_comp_usage: float = 0.0  
    lambda_comp_switch: float = 0.0  
    comp_switch_max: float = 10.0  
    comp_switch_bonus_mode: str = "linear"  
    lambda_ris_usage: float = 0.0  
    lambda_velocity_penalty: float = 0.0  
    velocity_threshold: float = 0.1  
    velocity_penalty_mode: str = "linear"  
    lambda_boundary_penalty: float = 0.0  
    boundary_margin: float = 30.0  
    lambda_idle_penalty: float = 0.0  
    idle_threshold: float = 0.05  

    
    
    
    use_fixed_normalization: bool = False
    norm_update_interval: int = 10
    reward_clip_range: float = 10.0
    reward_ema_beta: float = 0.0

    
    norm_ema_beta: float = 0.99
    norm_clip: float = 5.0
    
    obs_clip_range: float = 20.0
    
    strict_action_dim_check: bool = True
    
    strict_task_cache_slot: bool = True

    
    # - 'fixed' (w_delay, w_energy)
    
    # - 'auto' ynamic-improve + std-balancing
    weight_mode: str = "auto"

    
    dyn_beta: float = 0.98
    dyn_eta: float = 0.05
    dyn_w_min: float = 0.20
    dyn_w_max: float = 0.80
    dyn_margin: float = 0.02
    dyn_update_every: int = 1

    
    eff_wT_min: float = 0.10
    eff_wE_min: float = 0.30

    # -----------------
    
    # -----------------
    task_type: str = "uniform"        # 'constant' | 'uniform' | 'lognormal'
    D_bits_base: float = 8e5
    C_cycles_base: float = 1e9
    jitter: float = 0.2
    load_scale: float = 1.0
    load_alpha_D: float = 1.15
    load_alpha_C: float = 1.15
    def __post_init__(self):
        # ---- required fields check (extend this list if needed) ----
        required = [
            "theta_min",
            "fmax",
        ]
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            raise ValueError(
                f"[JointEnvConfig] Missing required fields in config: {missing}. "
                f"Please check your YAML / constructor inputs."
            )

        # ---- normalize types ----
        # fmax ist/tuple -> np.ndarray
        if self.fmax is not None and not isinstance(self.fmax, np.ndarray):
            self.fmax = np.asarray(self.fmax, dtype=np.float64)

        # ---- normalize enums ----
        mode = str(getattr(self, "action_space_mode", "full")).strip().lower()
        
        
        if mode == "position_only":
            mode = "hierarchical"
            self.z4_action_mode = "position_only"
        if mode not in ("full", "hierarchical"):
            raise ValueError(
                f"action_space_mode must be 'full'/'hierarchical'/'position_only', got: {self.action_space_mode}"
            )
        self.action_space_mode = mode

        theta_mode = str(getattr(self, "theta_mode", "solver")).strip().lower()
        if theta_mode not in ("solver", "agent", "policy"):
            raise ValueError(
                f"theta_mode must be 'solver'/'agent'/'policy', got: {self.theta_mode}"
            )
        self.theta_mode = theta_mode
        self.theta_softmax_tau = float(max(float(getattr(self, "theta_softmax_tau", 0.3)), 1e-6))

        
        
        
        speed_scope = str(getattr(self, "user_speed_sampling_scope", "episode")).strip().lower()
        if speed_scope not in ("episode", "per_user"):
            raise ValueError(
                "user_speed_sampling_scope must be 'episode' or 'per_user', "
                f"got: {self.user_speed_sampling_scope}"
            )
        self.user_speed_sampling_scope = speed_scope

        self.comp_score_temp = float(max(float(getattr(self, "comp_score_temp", 0.5)), 1e-6))
        self.comp_temp_stage_0 = float(max(float(getattr(self, "comp_temp_stage_0", 0.25)), 1e-6))
        self.comp_temp_stage_1 = float(max(float(getattr(self, "comp_temp_stage_1", 0.18)), 1e-6))
        self.comp_temp_stage_2 = float(max(float(getattr(self, "comp_temp_stage_2", 0.12)), 1e-6))
        self.z4_comp_meta_thr_delta_max = float(max(float(getattr(self, "z4_comp_meta_thr_delta_max", 0.10)), 0.0))
        self.z4_comp_meta_thr_min = float(np.clip(float(getattr(self, "z4_comp_meta_thr_min", 0.05)), 0.0, 1.0))
        self.z4_comp_meta_thr_max = float(np.clip(float(getattr(self, "z4_comp_meta_thr_max", 0.70)), 0.0, 1.0))
        if self.z4_comp_meta_thr_max < self.z4_comp_meta_thr_min:
            self.z4_comp_meta_thr_max = float(self.z4_comp_meta_thr_min)
        self.z4_comp_meta_score_width = float(max(float(getattr(self, "z4_comp_meta_score_width", 0.20)), 1e-3))
        self.z4_comp_meta_temp_scale_delta = float(max(float(getattr(self, "z4_comp_meta_temp_scale_delta", 0.35)), 0.0))
        self.z4_comp_meta_temp_scale_min = float(max(float(getattr(self, "z4_comp_meta_temp_scale_min", 0.65)), 1e-3))
        self.z4_comp_meta_temp_scale_max = float(max(float(getattr(self, "z4_comp_meta_temp_scale_max", 1.35)), self.z4_comp_meta_temp_scale_min))
        self.z4_comp_meta_ema_beta = float(np.clip(float(getattr(self, "z4_comp_meta_ema_beta", 0.80)), 0.0, 0.995))

class CompRISEnvJoint:
    """
    Joint-action environment (Phase2).
    """

    def __init__(self, cfg: JointEnvConfig, seed: int = 0, reward_mode: Optional[str] = None):
        self.cfg = cfg
        self.reward_mode = str(reward_mode or getattr(cfg, "reward_cost_mode", "p1")).strip().lower()
        
        self._seed_base = int(seed)
        self.rng = np.random.RandomState(self._seed_base)
        
        self._noise_rng = self.rng
        self._noise_bucket_id: int = -1
        self._noise_bucket_multiplier: float = 1.0
        # Diagnostic flag: full/no_comp gain can collapse under this setting.
        self._comp_total_noncoherent_mode = bool(
            bool(getattr(self.cfg, "enable_comp", True))
            and (str(getattr(self.cfg, "power_mode", "per_uav")).strip().lower() == "total")
            and (not bool(getattr(self.cfg, "comp_coherent", True)))
        )
        if self._comp_total_noncoherent_mode:
            print(
                "[HB][env][WARN] enable_comp=1 with power_mode=total and comp_coherent=0; "
                "full/no_comp gain may be strongly limited by physics.",
                flush=True,
            )

        task_type = str(getattr(cfg, "task_type", "uniform")).strip().lower()

        if task_type == "constant":
            self.task_model = ConstantTaskModel(
                cfg.I, cfg.D_bits_base, cfg.C_cycles_base,
                load_scale=cfg.load_scale,
                load_alpha_D=getattr(cfg, "load_alpha_D", 1.15),
                load_alpha_C=getattr(cfg, "load_alpha_C", 1.15),
            )
        elif task_type == "uniform":
            self.task_model = UniformTaskModel(
                self.rng, cfg.I, cfg.D_bits_base, cfg.C_cycles_base,
                jitter=cfg.jitter, load_scale=cfg.load_scale,
                load_alpha_D=getattr(cfg, "load_alpha_D", 1.15),
                load_alpha_C=getattr(cfg, "load_alpha_C", 1.15),
            )
        elif task_type == "lognormal":
            self.task_model = LogNormalTaskModel(
                self.rng, cfg.I, cfg.D_bits_base, cfg.C_cycles_base,
                jitter=cfg.jitter, load_scale=cfg.load_scale,
                load_alpha_D=getattr(cfg, "load_alpha_D", 1.15),
                load_alpha_C=getattr(cfg, "load_alpha_C", 1.15),
            )

        else:
            raise ValueError(f"Unknown task_type: {cfg.task_type} (normalized='{task_type}')")

        
        self.rn_T = RunningNorm(cfg.norm_ema_beta, clip=cfg.norm_clip)
        self.rn_E = RunningNorm(cfg.norm_ema_beta, clip=cfg.norm_clip)

        
        wT0, wE0 = _normalize_w(cfg.w_delay, cfg.w_energy)
        self.wT_dyn = float(wT0)
        self.wE_dyn = float(wE0)
        self._dyn_perf_T: Optional[float] = None
        self._dyn_perf_E: Optional[float] = None
        self._dyn_prev_T: Optional[float] = None
        self._dyn_prev_E: Optional[float] = None
        self._dyn_step: int = 0
        self._dyn_last_reason: str = "init"

        # Global training step (not reset each episode), used by beta schedule.
        self._train_step: int = 0
        
        self._comp_score_temp_runtime: float = float(max(float(getattr(self.cfg, "comp_score_temp", 0.5)), 1e-6))
        # EMA of boundary-stick ratio for dynamic guard logic.
        self._boundary_stick_ema: float = 0.0

        self.t = 0
        self.q: Optional[np.ndarray] = None
        self.q_prev: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.w_vel: Optional[np.ndarray] = None
        self._user_mobility_active: bool = False
        self._user_speed_eff: float = 0.0
        self._user_speed_level_idx: int = -1
        self._user_speed_level_idx_vec: Optional[np.ndarray] = None
        self._user_speed_eff_vec: Optional[np.ndarray] = None
        self._user_mobility_mask: Optional[np.ndarray] = None
        self._user_speed_levels_eff: List[float] = []
        self._user_speed_probs_eff: List[float] = []
        self._last_info: Dict = {}
        
        self._obs_prev_paper_cost: Optional[float] = None
        self._obs_prev_prev_paper_cost: Optional[float] = None
        
        
        self._task_next_D: Optional[np.ndarray] = None
        self._task_next_C: Optional[np.ndarray] = None
        self._task_next_slot: Optional[int] = None
        # =========================
        # [PATCH][PLOT] optional trace buffer for dynamic CoMP path animation
        
        # =========================
        self._trace_enabled: bool = False
        self._trace_keep_theta: bool = False
        self._trace_max_steps: Optional[int] = None
        self._trace_step: int = 0
        self._trace: Optional[Dict] = None
        # PhaseG-v2 state (reward shaping / dual variable)
        self._prev_potential: float = 0.0
        self._reward_ema: float = 0.0
        self._prev_paper_cost_step: Optional[float] = None
        self._paper_cost_ref_ema: float = float(max(float(getattr(self.cfg, "phaseg_v2_cost_ref_init", 1.0)), 1e-6))
        self._lambda_v_dyn: float = float(getattr(self.cfg, "lambda_v", 0.0))
        self._traj_prev_vel: Optional[np.ndarray] = None
        
        self._z4_gap_prev_main_no_gap: Optional[float] = None
        self._z4_gap_prev_main_with_gap: Optional[float] = None
        self._z4_gap_shape_bad_streak: int = 0
        self._z4_gap_ema: float = 0.0
        self._z4_gap_ema_ready: bool = False
        
        self._z4_comp_meta_thr_delta_ema: float = 0.0
        self._z4_comp_meta_temp_scale_ema: float = 1.0
        self._z4_comp_meta_ema_ready: bool = False
        
        self._z_cost_ref_by_load: Dict[float, float] = {}
        self._z_cost_ref_by_load_raw = getattr(self.cfg, "z_cost_ref_by_load", None)
        self._init_z_cost_ref_by_load_cache()
        
        self._z_cost_ref_by_speed: Dict[float, float] = {}
        self._z_cost_ref_by_speed_raw = getattr(self.cfg, "z_cost_ref_by_speed", None)
        self._init_z_cost_ref_by_speed_cache()
        self._warned_once_keys: set[str] = set()
        
        _solver_mode = str(getattr(self.cfg, "t6_hier_assoc_solver", "greedy")).strip().lower()
        _require_scipy = bool(getattr(self.cfg, "t6_hier_assoc_require_scipy", False))
        _reward_design = str(getattr(self.cfg, "reward_design", "")).strip().lower()
        _action_mode = str(getattr(self.cfg, "action_space_mode", "")).strip().lower()
        _is_t6_contract = bool((_reward_design == "t6_potential_lagrangian") and (_action_mode == "hierarchical"))
        if (_solver_mode == "hungarian") and (_scipy_linear_sum_assignment is None) and (_require_scipy or _is_t6_contract):
            raise RuntimeError(
                " hungarian  scipy"
                ""
            )
        if _is_t6_contract:
            _pid_iv = int(max(int(getattr(self.cfg, "t6_pid_update_interval", 2)), 1))
            if _pid_iv > 2:
                self._warn_once(
                    "t6_pid_update_interval_large",
                    f" t6_pid_update_interval={_pid_iv}12",
                )

        
        self.M = int(cfg.M)
        self.I = int(cfg.I)
        self.L = float(cfg.L)
        self.T = int(cfg.T)
        
        self._prev_comp_count: Optional[np.ndarray] = None  
        self._prev_comp_selection: Optional[np.ndarray] = None  
        
        self._prev_a_binary_for_switch_diag: Optional[np.ndarray] = None
        self._episode_counter: int = 0
        self._fixed_user_positions_cache: Optional[np.ndarray] = None
        self._fixed_user_positions_cache_key: Optional[str] = None
        self.a_binary = np.zeros((self.I, self.M), dtype=np.int32)
        self.comp_selection = np.zeros((self.M,), dtype=np.int32)

        
        self.cost_history: list = []  
        self.running_mean_window: int = int(getattr(cfg, "running_mean_window", 100))
        self.episode_cost_sum: float = 0.0
        self.episode_steps: int = 0
        self.improvement_step_weight: float = float(getattr(cfg, "improvement_step_weight", 0.1))
        self.improvement_episode_bonus_weight: float = float(
            getattr(cfg, "improvement_episode_bonus_weight", 1.0)
        )
        self.cost_prev: Optional[float] = None  

        
        self._t6_cost_ema_mean: float = 0.0
        self._t6_cost_ema_var: float = 1.0
        self._t6_cost_scale: float = 1.0
        self._t6_pid_step: int = 0
        self._t6_pid_controller: Optional[PidLagrangeController] = None
        if bool(getattr(self.cfg, "t6_pid_enable", True)):
            self._t6_pid_controller = PidLagrangeController(
                kp=float(getattr(self.cfg, "t6_pid_kp", 0.05)),
                ki=float(getattr(self.cfg, "t6_pid_ki", 0.01)),
                kd=float(getattr(self.cfg, "t6_pid_kd", 0.0)),
                delta=float(getattr(self.cfg, "t6_pid_delta", 0.20)),
                lambda_init=float(getattr(self.cfg, "lambda_v", 0.0)),
                lambda_min=float(getattr(self.cfg, "t6_lambda_min", 0.0)),
                lambda_max=float(getattr(self.cfg, "t6_lambda_max", 20.0)),
                violation_ema_beta=float(getattr(self.cfg, "t6_pid_violation_ema_beta", 0.8)),
                deadband=float(getattr(self.cfg, "t6_pid_deadband", 0.02)),
                lambda_step_clip=float(getattr(self.cfg, "t6_pid_lambda_step_clip", 0.25)),
                integral_decay=float(getattr(self.cfg, "t6_pid_integral_decay", 1.0)),
                freeze_after_frac=float(getattr(self.cfg, "t6_pid_freeze_after_frac", 0.90)),
            )
            self._lambda_v_dyn = float(self._t6_pid_controller.value)
        
        self._t6_direct_blockage_map: np.ndarray = np.ones((self.I, self.M), dtype=np.float64)

    def _warn_once(self, key: str, message: str) -> None:
        """"""
        k = str(key)
        if k in self._warned_once_keys:
            return
        self._warned_once_keys.add(k)
        print(f"[HB][env][WARN][{k}] {message}", flush=True)

    def _init_z_cost_ref_by_load_cache(self) -> None:
        """ z_cost_ref """
        self._z_cost_ref_by_load = {}
        raw = getattr(self, "_z_cost_ref_by_load_raw", None)
        if not isinstance(raw, dict):
            return
        for k, v in raw.items():
            try:
                lk = round(float(k), 6)
                rv = float(v)
            except Exception:
                continue
            if (not np.isfinite(lk)) or (not np.isfinite(rv)) or rv <= 0.0:
                continue
            self._z_cost_ref_by_load[lk] = float(rv)
        if len(self._z_cost_ref_by_load) <= 0:
            return
        pairs = ", ".join([f"{k:.3f}:{v:.6f}" for k, v in sorted(self._z_cost_ref_by_load.items())])
        print(f"[HB][env] z_cost_ref_by_load loaded: {pairs}", flush=True)

    def _resolve_z_cost_ref(self) -> Tuple[float, bool]:
        """
         z_cost_ref

        
        - ref_value:  z_cost_ref
        - from_map:  
        """
        base = float(max(float(getattr(self.cfg, "z_cost_ref", 1.0)), 1e-9))
        mp = getattr(self, "_z_cost_ref_by_load", {}) or {}
        if not isinstance(mp, dict) or len(mp) <= 0:
            return float(base), False
        load_now = float(getattr(self.cfg, "load_scale", 1.0))
        if not np.isfinite(load_now):
            return float(base), False
        probes = [
            round(float(load_now), 6),
            round(float(load_now), 4),
            round(float(load_now), 3),
            round(float(load_now), 2),
            round(float(load_now), 1),
        ]
        for p in probes:
            if p in mp:
                return float(max(float(mp[p]), 1e-9)), True
        for lk, rv in mp.items():
            if abs(float(lk) - float(load_now)) <= 1e-6:
                return float(max(float(rv), 1e-9)), True
        return float(base), False

    def _init_z_cost_ref_by_speed_cache(self) -> None:
        """ z_cost_ref """
        self._z_cost_ref_by_speed = {}
        raw = getattr(self, "_z_cost_ref_by_speed_raw", None)
        if not isinstance(raw, dict):
            return
        for k, v in raw.items():
            try:
                sk = round(float(k), 6)
                rv = float(v)
            except Exception:
                continue
            if (not np.isfinite(sk)) or (not np.isfinite(rv)) or rv <= 0.0:
                continue
            self._z_cost_ref_by_speed[sk] = float(rv)
        if len(self._z_cost_ref_by_speed) <= 0:
            return
        pairs = ", ".join([f"{k:.3f}:{v:.6f}" for k, v in sorted(self._z_cost_ref_by_speed.items())])
        print(f"[HB][env] z_cost_ref_by_speed loaded: {pairs}", flush=True)

    @staticmethod
    def _lookup_ref_from_map(mp: Dict[float, float], key_now: float) -> Tuple[float, bool]:
        """ round + """
        if (not isinstance(mp, dict)) or (len(mp) <= 0):
            return 1.0, False
        if not np.isfinite(float(key_now)):
            return 1.0, False
        probes = [
            round(float(key_now), 6),
            round(float(key_now), 4),
            round(float(key_now), 3),
            round(float(key_now), 2),
            round(float(key_now), 1),
        ]
        for p in probes:
            if p in mp:
                return float(max(float(mp[p]), 1e-9)), True
        for lk, rv in mp.items():
            if abs(float(lk) - float(key_now)) <= 1e-6:
                return float(max(float(rv), 1e-9)), True
        return 1.0, False

    def _resolve_z_cost_ref_with_speed(self) -> Tuple[float, bool, bool]:
        """
        / z_cost_ref
        
        - ref_value:  z_cost_ref
        - from_load_map:  
        - from_speed_map: 
        """
        base = float(max(float(getattr(self.cfg, "z_cost_ref", 1.0)), 1e-9))
        ref_load, hit_load = self._resolve_z_cost_ref()
        ref = float(ref_load)
        speed_now = float(getattr(self, "_user_speed_eff", getattr(self.cfg, "user_speed", 0.0)))
        
        
        idx_vec = getattr(self, "_user_speed_level_idx_vec", None)
        levels_eff = list(getattr(self, "_user_speed_levels_eff", []))
        if isinstance(idx_vec, np.ndarray) and idx_vec.size > 0 and len(levels_eff) > 0:
            try:
                idx_arr = np.asarray(idx_vec, dtype=np.int64).reshape(-1)
                idx_arr = idx_arr[idx_arr >= 0]
                if idx_arr.size > 0:
                    counts = np.bincount(idx_arr)
                    dom_idx = int(np.argmax(counts))
                    if 0 <= dom_idx < len(levels_eff):
                        speed_now = float(levels_eff[dom_idx])
            except Exception:
                pass
        ref_speed, hit_speed = self._lookup_ref_from_map(
            getattr(self, "_z_cost_ref_by_speed", {}) or {},
            float(speed_now),
        )
        if hit_speed:
            if hit_load:
                
                speed_ratio = float(ref_speed) / max(float(base), 1e-9)
                ref = float(ref) * float(max(speed_ratio, 1e-9))
            else:
                ref = float(ref_speed)
        return float(max(ref, 1e-9)), bool(hit_load), bool(hit_speed)

    def _resolve_comp_score_temp(self) -> float:
        """
         comp_score sigmoid 

        
        -  `comp_score_temp`
        -  `comp_temp_anneal_enable=1` 
          0~20k: stage_020k~40k: stage_140k+: stage_2
        """
        base_temp = float(max(float(getattr(self.cfg, "comp_score_temp", 0.5)), 1e-6))
        if not bool(getattr(self.cfg, "comp_temp_anneal_enable", False)):
            return float(base_temp)

        t0 = float(max(float(getattr(self.cfg, "comp_temp_stage_0", 0.25)), 1e-6))
        t1 = float(max(float(getattr(self.cfg, "comp_temp_stage_1", 0.18)), 1e-6))
        t2 = float(max(float(getattr(self.cfg, "comp_temp_stage_2", 0.12)), 1e-6))
        step_now = int(max(getattr(self, "_train_step", 0), 0))
        if step_now < 20000:
            return float(t0)
        if step_now < 40000:
            return float(t1)
        return float(t2)

    # --------- dims ---------
    @property
    def action_dim(self) -> int:
        M, I = int(self.cfg.M), int(self.cfg.I)
        action_space_mode = str(getattr(self.cfg, "action_space_mode", "full")).strip().lower()
        if action_space_mode == "hierarchical":
            
            z4_action_mode = str(getattr(self.cfg, "z4_action_mode", "hierarchical")).strip().lower()
            if z4_action_mode == "position_only":
                
                if bool(getattr(self.cfg, "z4_comp_meta_enable", False)):
                    return int((2 * M) + 2)
                return int(2 * M)
            
            return int((2 * M) + I)

        action_score_mode = str(getattr(self.cfg, "action_score_mode", "separate")).strip().lower()
        theta_mode = str(getattr(self.cfg, "theta_mode", "solver")).strip().lower()
        use_separate_z = action_score_mode != "shared"
        use_theta = theta_mode == "policy"

        dim = (2 * M) + (I * M)
        if use_separate_z:
            dim += I * M
        if use_theta:
            dim += I
        return int(dim)

    @property
    def action_space_dim(self) -> int:
        """Compatibility alias for trainers expecting env.action_space_dim."""
        return self.action_dim

    @property
    def act_dim(self) -> int:
        """Compatibility alias for trainers expecting env.act_dim."""
        return self.action_dim

    @property
    def obs_dim(self) -> int:
        """Observation dimension for flat observation vector."""
        M, I = self.cfg.M, self.cfg.I
        
        # - UAV : 2*M
        # - : 2*I
        # - : I*M
        # - : I*M
        # - Dn/Cn: I + I
        
        
        
        # ===== [ A 3] =====
        # - : M*4
        # - : M
        # ===== [END A 3] =====
        # ===== [ B 4] QoS/ / =====
        
        # ===== [END B 4] =====
        
        
        
        
        
        
        
        base_dim = 2 * M + 2 * I + I * M + I * M + I + I + M + 2 + 2 + M * 4 + M + 5 + I
        extra_dim = 0
        if bool(getattr(self.cfg, "obs_enable_rate_feedback", False)):
            extra_dim += int(I)
        if bool(getattr(self.cfg, "obs_enable_cost_feedback", False)):
            extra_dim += 2
        
        if bool(getattr(self.cfg, "obs_enable_prev_action", False)):
            z4_action_mode = str(getattr(self.cfg, "z4_action_mode", "hierarchical")).strip().lower()
            if z4_action_mode == "position_only":
                extra_dim += int(M) * 2  
                if bool(getattr(self.cfg, "z4_comp_meta_enable", False)):
                    extra_dim += 2  
            else:
                extra_dim += int(M) * 2 + int(I)  # delta_q(M*2) + comp_score(I) = 20D
        # ===== [END Phase Z] =====
        return int(base_dim + extra_dim)

    def set_load_scale(self, s: float) -> None:
        self.cfg.load_scale = float(s)
        self.task_model.set_load_scale(float(s))
        # load
        self._task_next_D = None
        self._task_next_C = None
        self._task_next_slot = None

    def set_association_mode(self, mode: str) -> str:
        """ association_mode hard/soft"""
        m = str(mode).strip().lower()
        if m not in ("hard", "soft"):
            raise ValueError(f"association_mode must be 'hard' or 'soft', got: {mode}")
        self.cfg.association_mode = m
        return m

    def _resolve_user_mobility_active(self) -> bool:
        mode = str(getattr(self.cfg, "user_mobility_mode", "static")).strip().lower()
        
        if mode == "semi_random":
            mode = "mixed"
        if mode == "static":
            return False
        speed_ready = bool(self._has_positive_user_speed_config())
        if mode == "slow":
            return speed_ready
        if mode == "mixed":
            p = float(np.clip(float(getattr(self.cfg, "user_mobility_prob", 0.30)), 0.0, 1.0))
            scope = str(getattr(self.cfg, "user_speed_sampling_scope", "episode")).strip().lower()
            if scope == "per_user":
                
                
                return bool(speed_ready and p > 1e-12)
            rng = self._get_noise_rng()
            return bool(speed_ready and (rng.rand() < p))
        return False

    def _resolve_user_speed_sampling_scope(self) -> str:
        scope = str(getattr(self.cfg, "user_speed_sampling_scope", "episode")).strip().lower()
        if scope not in ("episode", "per_user"):
            return "episode"
        return scope

    def _parse_user_speed_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """ (levels, probs)"""
        levels_raw = getattr(self.cfg, "user_speed_levels", None)
        probs_raw = getattr(self.cfg, "user_speed_probs", None)
        levels: List[float] = []
        probs: List[float] = []

        if isinstance(levels_raw, (list, tuple)):
            for v in levels_raw:
                try:
                    x = float(v)
                except Exception:
                    continue
                if np.isfinite(x) and x >= 0.0:
                    levels.append(float(x))

        if not levels:
            self._user_speed_levels_eff = []
            self._user_speed_probs_eff = []
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

        if isinstance(probs_raw, (list, tuple)) and len(probs_raw) == len(levels):
            for v in probs_raw:
                try:
                    p = float(v)
                except Exception:
                    p = 0.0
                if (not np.isfinite(p)) or p < 0.0:
                    p = 0.0
                probs.append(float(p))

        if len(probs) != len(levels) or float(np.sum(probs)) <= 1e-12:
            probs = [1.0 / float(len(levels))] * int(len(levels))
        else:
            s = float(np.sum(probs))
            probs = [float(p / s) for p in probs]

        self._user_speed_levels_eff = [float(x) for x in levels]
        self._user_speed_probs_eff = [float(x) for x in probs]
        return np.asarray(levels, dtype=np.float64), np.asarray(probs, dtype=np.float64)

    def _has_positive_user_speed_config(self) -> bool:
        """"""
        levels, _ = self._parse_user_speed_distribution()
        if int(levels.size) > 0:
            return bool(np.max(levels) > 1e-9)
        return bool(float(getattr(self.cfg, "user_speed", 0.0)) > 1e-9)

    def _sample_user_speed_for_episode(self) -> float:
        """ episode """
        levels, probs = self._parse_user_speed_distribution()
        self._user_speed_level_idx = -1
        if int(levels.size) > 0:
            rng = self._get_noise_rng()
            idx = int(rng.choice(int(levels.size), p=probs))
            self._user_speed_level_idx = idx
            return float(max(float(levels[idx]), 0.0))
        return float(max(float(getattr(self.cfg, "user_speed", 0.0)), 0.0))

    def _user_motion_bounds(self) -> Tuple[float, float]:
        """Return active user-motion box [lo, hi] along each axis."""
        L = float(max(float(getattr(self.cfg, "L", 0.0)), 1e-9))
        frac = float(np.clip(float(getattr(self.cfg, "user_area_margin_frac", 0.0)), 0.0, 0.49))
        margin = float(frac * L)
        lo = float(margin)
        hi = float(L - margin)
        if hi <= lo + 1e-6:
            return 0.0, float(L)
        return lo, hi

    def _get_noise_rng(self) -> np.random.RandomState:
        """ episode """
        rng = getattr(self, "_noise_rng", None)
        if isinstance(rng, np.random.RandomState):
            return rng
        return self.rng

    def _resolve_noise_bucket_profile(self) -> None:
        """
        LayerA episode easy/mid/hard
        
        """
        self._noise_bucket_id = -1
        self._noise_bucket_multiplier = 1.0
        if not bool(getattr(self.cfg, "noise_bucket_enable", False)):
            return

        raw_levels = getattr(self.cfg, "noise_bucket_levels", None)
        levels: List[float] = []
        if isinstance(raw_levels, (list, tuple)):
            for x in raw_levels:
                try:
                    xv = float(x)
                except Exception:
                    continue
                if np.isfinite(xv) and xv > 0.0:
                    levels.append(float(xv))
        if not levels:
            levels = [0.85, 1.0, 1.15]

        mode = str(getattr(self.cfg, "noise_bucket_mode", "round_robin")).strip().lower()
        n = int(len(levels))
        if mode == "random":
            idx = int(self.rng.randint(0, n))
        else:
            idx = int((int(max(self._episode_counter, 1)) - 1) % n)
        self._noise_bucket_id = int(idx)
        self._noise_bucket_multiplier = float(levels[idx])

    def _scale_exogenous_noise(self, base: float, *, lo: float, hi: float) -> float:
        """"""
        mult = float(getattr(self, "_noise_bucket_multiplier", 1.0))
        val = float(base) * float(mult)
        return float(np.clip(val, float(lo), float(hi)))

    def _eff_t6_edge_user_ratio(self) -> float:
        return self._scale_exogenous_noise(
            float(getattr(self.cfg, "t6_edge_user_ratio", 0.0)),
            lo=0.0,
            hi=1.0,
        )

    def _eff_user_position_jitter(self) -> float:
        return self._scale_exogenous_noise(
            float(getattr(self.cfg, "user_position_jitter", 0.0)),
            lo=0.0,
            hi=max(float(getattr(self.cfg, "L", 150.0)), 1e-6),
        )

    def _eff_task_arrival_jitter(self) -> float:
        return self._scale_exogenous_noise(
            float(getattr(self.cfg, "task_arrival_jitter", 0.0)),
            lo=0.0,
            hi=1.0,
        )

    def _eff_task_size_jitter(self) -> float:
        return self._scale_exogenous_noise(
            float(getattr(self.cfg, "task_size_jitter", 0.0)),
            lo=0.0,
            hi=1.0,
        )

    def _eff_t6_direct_blockage_prob(self) -> float:
        return self._scale_exogenous_noise(
            float(getattr(self.cfg, "t6_direct_blockage_prob", 0.0)),
            lo=0.0,
            hi=1.0,
        )

    def _eff_t6_interference_jitter(self) -> float:
        return self._scale_exogenous_noise(
            float(getattr(self.cfg, "t6_interference_jitter", 0.0)),
            lo=0.0,
            hi=1.0,
        )

    def _sample_user_positions(self, I: int) -> np.ndarray:
        """Sample users in the configured active box."""
        lo, hi = self._user_motion_bounds()
        rng = self._get_noise_rng()
        return rng.uniform(float(lo), float(hi), size=(int(I), 2)).astype(np.float64)

    def _apply_t6_edge_user_bias(self, users_xy: np.ndarray) -> np.ndarray:
        """
        PhaseT6-7
        baseline
        """
        w = np.asarray(users_xy, dtype=np.float64).reshape(int(self.cfg.I), 2).copy()
        ratio = float(self._eff_t6_edge_user_ratio())
        if ratio <= 1e-12:
            return w
        I = int(w.shape[0])
        if I <= 0:
            return w
        n_edge = int(max(1, round(ratio * float(I))))
        n_edge = int(min(max(n_edge, 0), I))
        if n_edge <= 0:
            return w
        L = float(max(float(getattr(self.cfg, "L", 1.0)), 1e-9))
        band_frac = float(np.clip(float(getattr(self.cfg, "t6_edge_band_frac", 0.15)), 1e-4, 0.49))
        band = float(np.clip(band_frac * L, 1e-6, 0.49 * L))
        rng = self._get_noise_rng()
        idx = rng.choice(I, size=n_edge, replace=False)
        for i in idx:
            side = int(rng.randint(0, 4))
            if side == 0:  
                x = float(rng.uniform(0.0, band))
                y = float(rng.uniform(0.0, L))
            elif side == 1:  
                x = float(rng.uniform(L - band, L))
                y = float(rng.uniform(0.0, L))
            elif side == 2:  
                x = float(rng.uniform(0.0, L))
                y = float(rng.uniform(0.0, band))
            else:  
                x = float(rng.uniform(0.0, L))
                y = float(rng.uniform(L - band, L))
            w[int(i), 0] = x
            w[int(i), 1] = y
        return w

    def _refresh_t6_direct_blockage_map(self) -> None:
        """
        PhaseT6-7
         direct RISRIS
        """
        I = int(self.cfg.I)
        M = int(self.cfg.M)
        prob = float(self._eff_t6_direct_blockage_prob())
        if prob <= 1e-12:
            self._t6_direct_blockage_map = np.ones((I, M), dtype=np.float64)
            return
        g_min = float(np.clip(float(getattr(self.cfg, "t6_direct_blockage_min_gain", 0.35)), 0.0, 1.0))
        g_max = float(np.clip(float(getattr(self.cfg, "t6_direct_blockage_max_gain", 0.80)), g_min, 1.0))
        rng = self._get_noise_rng()
        mask = (rng.rand(I, M) < prob)
        att = rng.uniform(g_min, g_max, size=(I, M)).astype(np.float64)
        out = np.ones((I, M), dtype=np.float64)
        out[mask] = att[mask]
        self._t6_direct_blockage_map = out

    def _init_user_velocity(self) -> None:
        if self.w is None:
            self.w_vel = None
            self._user_speed_eff = 0.0
            self._user_speed_level_idx = -1
            self._user_speed_level_idx_vec = None
            self._user_speed_eff_vec = None
            self._user_mobility_mask = None
            return
        I = int(self.w.shape[0])
        self._user_speed_level_idx_vec = np.full((I,), -1, dtype=np.int64)
        self._user_speed_eff_vec = np.zeros((I,), dtype=np.float64)
        self._user_mobility_mask = np.zeros((I,), dtype=bool)
        if not self._user_mobility_active:
            
            self._parse_user_speed_distribution()
            self._user_speed_level_idx = -1
            self.w_vel = np.zeros_like(self.w, dtype=np.float64)
            self._user_speed_eff = 0.0
            return
        rng = self._get_noise_rng()
        speed_scope = self._resolve_user_speed_sampling_scope()
        mode = str(getattr(self.cfg, "user_mobility_mode", "static")).strip().lower()
        if mode == "semi_random":
            mode = "mixed"
        move_prob = float(np.clip(float(getattr(self.cfg, "user_mobility_prob", 0.30)), 0.0, 1.0))

        if speed_scope == "per_user":
            levels, probs = self._parse_user_speed_distribution()
            if int(levels.size) > 0:
                idx_vec = rng.choice(int(levels.size), size=I, p=probs).astype(np.int64)
                speed_vec = levels[idx_vec].astype(np.float64)
            else:
                idx_vec = np.full((I,), -1, dtype=np.int64)
                speed_vec = np.full((I,), float(max(float(getattr(self.cfg, "user_speed", 0.0)), 0.0)), dtype=np.float64)

            if mode == "mixed":
                move_mask = (rng.rand(I) < move_prob)
            else:
                move_mask = np.ones((I,), dtype=bool)
            speed_vec = np.where(move_mask, speed_vec, 0.0).astype(np.float64)
            idx_vec_eff = np.where(move_mask, idx_vec, -1).astype(np.int64)

            ang = rng.uniform(0.0, 2.0 * np.pi, size=(I,))
            vel = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float64)
            self.w_vel = vel * speed_vec.reshape(-1, 1)

            self._user_speed_level_idx = -2  
            self._user_speed_level_idx_vec = idx_vec_eff
            self._user_speed_eff_vec = speed_vec
            self._user_mobility_mask = move_mask.astype(bool)
            self._user_speed_eff = float(np.mean(speed_vec)) if I > 0 else 0.0
            if float(np.max(speed_vec)) <= 1e-9:
                self.w_vel = np.zeros_like(self.w, dtype=np.float64)
                self._user_speed_eff = 0.0
            return

        speed = float(self._sample_user_speed_for_episode())
        self._user_speed_level_idx_vec = np.full((I,), int(self._user_speed_level_idx), dtype=np.int64)
        self._user_speed_eff_vec = np.full((I,), speed, dtype=np.float64)
        self._user_mobility_mask = np.ones((I,), dtype=bool)
        if speed <= 1e-9:
            self.w_vel = np.zeros_like(self.w, dtype=np.float64)
            self._user_speed_eff = 0.0
            self._user_mobility_mask = np.zeros((I,), dtype=bool)
            return
        ang = rng.uniform(0.0, 2.0 * np.pi, size=(I,))
        vel = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float64)
        self.w_vel = vel * speed
        self._user_speed_eff = speed

    def _advance_user_positions(self) -> None:
        if self.w is None:
            return
        if (not self._user_mobility_active) or (self.w_vel is None):
            return
        dt = float(max(float(getattr(self.cfg, "dt", 1.0)), 1e-9))
        lo, hi = self._user_motion_bounds()
        if hi <= lo + 1e-9:
            return

        # Optional heading jitter for slow-mobility scenario.
        jitter = float(max(float(getattr(self.cfg, "user_heading_jitter", 0.0)), 0.0))
        if jitter > 1e-12:
            rng = self._get_noise_rng()
            dphi = rng.normal(loc=0.0, scale=jitter, size=(int(self.w.shape[0]),))
            c = np.cos(dphi).reshape(-1, 1)
            s = np.sin(dphi).reshape(-1, 1)
            vx = self.w_vel[:, 0:1]
            vy = self.w_vel[:, 1:2]
            self.w_vel = np.concatenate([c * vx - s * vy, s * vx + c * vy], axis=1)

        w_new = self.w + self.w_vel * dt
        reflect = bool(getattr(self.cfg, "user_reflect_boundary", True))
        if reflect:
            low = w_new < float(lo)
            if np.any(low):
                w_new = np.where(low, 2.0 * float(lo) - w_new, w_new)
                self.w_vel[low] *= -1.0
            high = w_new > float(hi)
            if np.any(high):
                w_new = np.where(high, 2.0 * float(hi) - w_new, w_new)
                self.w_vel[high] *= -1.0
            w_new = np.clip(w_new, float(lo), float(hi))
        else:
            w_new = np.clip(w_new, float(lo), float(hi))
            hit = (w_new <= float(lo)) | (w_new >= float(hi))
            if np.any(hit):
                self.w_vel[hit] = 0.0
        self.w = w_new

    def _sample_task_with_jitter(self) -> Tuple[np.ndarray, np.ndarray]:
        """/"""
        D_raw, C_raw = self.task_model.sample()
        D = np.asarray(D_raw, dtype=np.float64).reshape(int(self.cfg.I),)
        C = np.asarray(C_raw, dtype=np.float64).reshape(int(self.cfg.I),)

        arrival_jitter = float(self._eff_task_arrival_jitter())
        size_jitter = float(self._eff_task_size_jitter())
        rng = self._get_noise_rng()

        if arrival_jitter > 1e-12:
            D_scale = rng.uniform(1.0 - arrival_jitter, 1.0 + arrival_jitter, size=D.shape).astype(np.float64)
            D = D * D_scale
        if size_jitter > 1e-12:
            C_scale = rng.uniform(1.0 - size_jitter, 1.0 + size_jitter, size=C.shape).astype(np.float64)
            C = C * C_scale

        D = np.maximum(D, 0.0)
        C = np.maximum(C, 0.0)
        return D, C

    def _build_fixed_user_cache_key(
        self,
        fixed_pos: np.ndarray,
        refresh_interval: int,
        jitter: float,
    ) -> str:
        """
        
        """
        arr = np.asarray(fixed_pos, dtype=np.float64).reshape(int(self.cfg.I), 2)
        h = hashlib.sha1()
        h.update(arr.tobytes())
        h.update(str(int(max(int(refresh_interval), 1))).encode("utf-8"))
        h.update(f"{float(jitter):.9g}".encode("utf-8"))
        h.update(f"{float(self.cfg.L):.9g}".encode("utf-8"))
        return h.hexdigest()

    def _is_task_cache_valid(self) -> bool:
        """
        
        """
        if self._task_next_D is None or self._task_next_C is None:
            return False
        try:
            D = np.asarray(self._task_next_D, dtype=np.float64).reshape(int(self.cfg.I),)
            C = np.asarray(self._task_next_C, dtype=np.float64).reshape(int(self.cfg.I),)
        except Exception:
            return False
        if (not np.all(np.isfinite(D))) or (not np.all(np.isfinite(C))):
            return False
        if np.any(D < -1e-12) or np.any(C < -1e-12):
            return False
        return True

    def _set_task_cache_for_slot(self, slot: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        slot
        """
        D, C = self._sample_task_with_jitter()
        self._task_next_D = np.asarray(D, dtype=np.float64).reshape(int(self.cfg.I),)
        self._task_next_C = np.asarray(C, dtype=np.float64).reshape(int(self.cfg.I),)
        self._task_next_slot = int(slot)
        return self._task_next_D.copy(), self._task_next_C.copy()

    def _ensure_task_cache(
        self,
        *,
        expected_slot: int,
        context: str,
        force_resample: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        
        - expected_slot: 
        - force_resample=True step
        """
        exp = int(max(int(expected_slot), 1))
        need = bool(force_resample) or (not self._is_task_cache_valid())
        if (not need) and (self._task_next_slot is not None) and (int(self._task_next_slot) != exp):
            if bool(getattr(self.cfg, "strict_task_cache_slot", True)):
                raise RuntimeError(
                    f"ctx={context}, cached={self._task_next_slot}, expected={exp}"
                    "reset/obs/step"
                )
            self._warn_once(
                f"task_cache_slot_mismatch_{str(context)}",
                f"ctx={context}, cached={self._task_next_slot}, expected={exp}",
            )
            need = True
        if need:
            return self._set_task_cache_for_slot(slot=exp)
        D = np.asarray(self._task_next_D, dtype=np.float64).reshape(int(self.cfg.I),).copy()
        C = np.asarray(self._task_next_C, dtype=np.float64).reshape(int(self.cfg.I),).copy()
        return D, C

    def apply_meta_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        meta-controller+
        """
        allowed = {
            "lambda_v": (0.0, 20.0),
            "beta_comp_max": (0.0, 0.08),
            "beta_ris_max": (0.0, 0.15),
            
        }
        out: Dict[str, Any] = {
            "applied": False,
            "patch": {},
            "clamped": {},
            "ignored": [],
        }
        if not isinstance(action, dict):
            out["ignored"] = ["non_dict"]
            return out

        for k, v in action.items():
            if k not in allowed:
                out["ignored"].append(str(k))
                continue
            lo, hi = allowed[k]
            try:
                vf = float(v)
            except Exception:
                out["ignored"].append(str(k))
                continue
            vc = float(np.clip(vf, lo, hi))
            out["clamped"][k] = bool(abs(vc - vf) > 1e-12)
            out["patch"][k] = float(vc)
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, float(vc))
                out["applied"] = True
            else:
                
                out["patch"].pop(k, None)
                out["clamped"].pop(k, None)

            
            if k == "lambda_v":
                self._lambda_v_dyn = float(vc)
                if self._t6_pid_controller is not None:
                    self._t6_pid_controller.reset(lambda_init=float(vc))

        return out

    def _boundary_eval_band(self) -> float:
        """EMA"""
        L = float(max(float(getattr(self.cfg, "L", 1.0)), 1e-9))
        frac = float(np.clip(float(getattr(self.cfg, "phaseh_boundary_eval_band_frac", 0.05)), 0.0, 0.49))
        return float(max(frac * L, 1e-9))

    def _boundary_stick_ratio(self, q_now: Optional[np.ndarray], band: Optional[float] = None) -> float:
        """ UAV """
        if q_now is None:
            return 0.0
        try:
            q_arr = np.asarray(q_now, dtype=np.float64).reshape(int(self.cfg.M), 2)
        except Exception as exc:
            self._warn_once("boundary_stick_ratio_parse", f"0.0{exc}")
            return 0.0
        if q_arr.size <= 0:
            return 0.0
        L = float(max(float(getattr(self.cfg, "L", 1.0)), 1e-9))
        if band is None:
            band = self._boundary_eval_band()
        band = float(max(float(band), 1e-9))
        d_left = q_arr[:, 0]
        d_right = L - q_arr[:, 0]
        d_down = q_arr[:, 1]
        d_up = L - q_arr[:, 1]
        d_min = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
        return float(np.mean(d_min <= band))

    def _update_boundary_stick_ema(self, q_now: Optional[np.ndarray]) -> None:
        """ EMA """
        hit_frac = float(self._boundary_stick_ratio(q_now, band=self._boundary_eval_band()))
        beta = float(np.clip(float(getattr(self.cfg, "phaseh_boundary_guard_ema_beta", 0.95)), 0.0, 0.999))
        if (not np.isfinite(float(getattr(self, "_boundary_stick_ema", 0.0)))) or self._train_step <= 1:
            self._boundary_stick_ema = float(hit_frac)
        else:
            self._boundary_stick_ema = float(
                beta * float(self._boundary_stick_ema) + (1.0 - beta) * float(hit_frac)
            )

    def _wall_safe_margin(self) -> float:
        """PhaseH"""
        base_safe = float(max(float(self.cfg.Vmax) * float(self.cfg.dt), 0.0))
        try:
            frac = float(getattr(self.cfg, "phaseh_boundary_guard_frac", 0.0))
        except Exception as exc:
            self._warn_once("wall_safe_margin_frac", f"phaseh_boundary_guard_frac 0.0{exc}")
            frac = 0.0
        frac = float(np.clip(frac, 0.0, 0.49))
        guard_safe = float(base_safe)
        if frac > 0.0:
            # Keep a tiny epsilon so d_min==0.05L does not stick on the threshold.
            guard_safe = float(max(guard_safe, float(frac * float(self.cfg.L) + 1e-6)))

        use_dynamic = bool(getattr(self.cfg, "phaseh_boundary_guard_dynamic", False))
        if (not use_dynamic) or guard_safe <= base_safe + 1e-12:
            return float(max(guard_safe, 0.0))

        low = float(np.clip(float(getattr(self.cfg, "phaseh_boundary_guard_low", 0.30)), 0.0, 0.95))
        high = float(np.clip(float(getattr(self.cfg, "phaseh_boundary_guard_high", 0.45)), low + 1e-6, 1.0))
        ema = float(np.clip(float(getattr(self, "_boundary_stick_ema", 0.0)), 0.0, 1.0))
        alpha = float(np.clip((ema - low) / max(high - low, 1e-6), 0.0, 1.0))
        wall_safe = float(base_safe + alpha * (guard_safe - base_safe))
        return float(max(wall_safe, 0.0))

    def _gate_comp_set_by_snr(
        self,
        g_row: np.ndarray,
        chosen_idx: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        CoMPSNR
        """
        chosen = np.asarray(chosen_idx, dtype=np.int64).reshape(-1)
        if chosen.size < 2:
            return chosen, False
        if not bool(getattr(self.cfg, "comp_gate_snr_enable", False)):
            return chosen, False

        M = int(self.cfg.M)
        g_arr = np.asarray(g_row, dtype=np.float64).reshape(M,)
        if g_arr.size != M:
            return chosen, False

        
        anchor_local = int(np.argmax(g_arr[chosen]))
        anchor_uav = int(chosen[anchor_local])

        p_tx = float(getattr(self.cfg, "p", getattr(self.cfg, "P", 1.0)))
        N0 = float(max(float(getattr(self.cfg, "N0", 1.0)), 1e-18))
        B = float(max(float(getattr(self.cfg, "B", 1.0)), 1e-18))
        coherent = bool(getattr(self.cfg, "comp_coherent", True))
        coherence_boost = float(max(float(getattr(self.cfg, "comp_coherence_boost", 1.0)), 0.0))
        power_mode = str(getattr(self.cfg, "power_mode", "per_uav")).strip().lower()
        if power_mode not in ("per_uav", "total"):
            power_mode = "per_uav"

        a_multi = np.zeros((M,), dtype=np.float64)
        a_multi[chosen] = 1.0
        gamma_multi = float(
            snr_from_comp(
                a_row=a_multi,
                p=p_tx,
                g_vec=g_arr,
                N0=N0,
                B=B,
                coherent=coherent,
                coherence_boost=coherence_boost,
                power_mode=power_mode,
            )
        )

        a_single = np.zeros((M,), dtype=np.float64)
        a_single[anchor_uav] = 1.0
        gamma_single = float(
            snr_from_comp(
                a_row=a_single,
                p=p_tx,
                g_vec=g_arr,
                N0=N0,
                B=B,
                coherent=False,
                coherence_boost=1.0,
                power_mode=power_mode,
            )
        )

        margin = float(np.clip(float(getattr(self.cfg, "comp_gate_snr_margin", 0.0)), 0.0, 5.0))
        target = float((1.0 + margin) * max(gamma_single, 1e-12))
        keep_multi = bool(np.isfinite(gamma_multi) and (gamma_multi >= target))
        if keep_multi:
            return chosen, False
        return np.asarray([anchor_uav], dtype=np.int64), True

    # --------- reset/obs ---------
    def reset(self) -> np.ndarray:
        self._episode_counter += 1
        self._resolve_noise_bucket_profile()
        
        
        
        _seed_mod = int(np.iinfo(np.uint32).max) + 1
        noise_seed_raw = int(
            self._seed_base
            + 1000003 * int(self._episode_counter)
            + 97 * int(max(self._noise_bucket_id, 0))
        )
        noise_seed = int(noise_seed_raw % _seed_mod)
        self._noise_rng = np.random.RandomState(noise_seed)
        if hasattr(self.task_model, "rng"):
            try:
                self.task_model.rng = self._noise_rng
            except Exception:
                pass
        self.t = 0
        rng_noise = self._get_noise_rng()
        q0, w = init_positions(
            rng_noise, self.cfg.M, self.cfg.I, self.cfg.L,
            uav_init_mode=str(getattr(self.cfg, "uav_init_mode", "random")),
            uav_fixed_positions=getattr(self.cfg, "uav_fixed_positions", None),
        )

        
        if self.cfg.user_fixed_positions is not None:
            fixed_pos = np.array(self.cfg.user_fixed_positions, dtype=np.float64)
            if fixed_pos.shape[0] != self.cfg.I:
                raise ValueError(
                    f"user_fixed_positions({fixed_pos.shape[0]})I({self.cfg.I})"
                )
            if fixed_pos.shape[1] != 2:
                raise ValueError(
                    f"user_fixed_positions[x, y]shape={fixed_pos.shape}"
                )
            refresh_interval = int(max(int(getattr(self.cfg, "user_position_refresh_interval", 1)), 1))
            jitter = float(self._eff_user_position_jitter())
            cache_key = self._build_fixed_user_cache_key(
                fixed_pos=fixed_pos,
                refresh_interval=refresh_interval,
                jitter=jitter,
            )
            need_refresh = self._fixed_user_positions_cache is None
            if (not need_refresh) and (self._fixed_user_positions_cache_key != cache_key):
                need_refresh = True
            if not need_refresh:
                need_refresh = ((self._episode_counter - 1) % refresh_interval) == 0
            if not need_refresh:
                try:
                    w_chk = np.asarray(self._fixed_user_positions_cache, dtype=np.float64).reshape(int(self.cfg.I), 2)
                    if (not np.all(np.isfinite(w_chk))) or np.any(w_chk < 0.0) or np.any(w_chk > float(self.cfg.L)):
                        need_refresh = True
                except Exception:
                    need_refresh = True
            if need_refresh:
                w_cached = fixed_pos.copy()
                
                if jitter > 1e-9:
                    rng = self._get_noise_rng()
                    jitter_offset = rng.uniform(-jitter, jitter, size=(self.cfg.I, 2))
                    w_cached = w_cached + jitter_offset
                    L = float(self.cfg.L)
                    w_cached = np.clip(w_cached, 0.0, L)
                self._fixed_user_positions_cache = w_cached
                self._fixed_user_positions_cache_key = str(cache_key)
            w = np.asarray(self._fixed_user_positions_cache, dtype=np.float64).copy()
            if refresh_interval > 1:
                self._warn_once(
                    "fixed_user_refresh_interval",
                    " user_position_refresh_interval 1",
                )
        # Optional mobility box: keep users away from hard boundaries.
        elif float(getattr(self.cfg, "user_area_margin_frac", 0.0)) > 1e-12:
            self._fixed_user_positions_cache = None
            self._fixed_user_positions_cache_key = None
            w = self._sample_user_positions(int(self.cfg.I))
        else:
            self._fixed_user_positions_cache = None
            self._fixed_user_positions_cache_key = None

        
        w = self._apply_t6_edge_user_bias(w)

        self.q = q0
        self.q_prev = q0.copy()
        self.w = w
        
        self._refresh_t6_direct_blockage_map()
        self._boundary_stick_ema = 0.0
        self._update_boundary_stick_ema(self.q)
        self._user_mobility_active = self._resolve_user_mobility_active()
        self._init_user_velocity()
        self._last_info = {}
        self._obs_prev_paper_cost = None
        self._obs_prev_prev_paper_cost = None
        self._prev_potential = 0.0
        self._prev_paper_cost_step = None
        self._z4_gap_prev_main_no_gap = None
        self._z4_gap_prev_main_with_gap = None
        self._z4_gap_shape_bad_streak = 0
        self._z4_gap_ema = 0.0
        self._z4_gap_ema_ready = False
        self._z4_comp_meta_thr_delta_ema = 0.0
        self._z4_comp_meta_temp_scale_ema = 1.0
        self._z4_comp_meta_ema_ready = False
        self._prev_prox_d_norm = None  
        self._paper_cost_ref_ema = float(max(float(getattr(self.cfg, "phaseg_v2_cost_ref_init", 1.0)), 1e-6))
        self._traj_prev_vel = None
        self._prev_comp_count = None
        self._prev_comp_selection = None
        self._prev_a_binary_for_switch_diag = None
        self.a_binary = np.zeros((int(self.cfg.I), int(self.cfg.M)), dtype=np.int32)
        self.comp_selection = np.zeros((int(self.cfg.M),), dtype=np.int32)
        
        wT0, wE0 = _normalize_w(float(self.cfg.w_delay), float(self.cfg.w_energy))
        self.wT_dyn = float(wT0)
        self.wE_dyn = float(wE0)
        self._dyn_perf_T = None
        self._dyn_perf_E = None
        self._dyn_prev_T = None
        self._dyn_prev_E = None
        self._dyn_step = 0
        self._dyn_last_reason = "reset"
        self._lambda_v_dyn = float(getattr(self.cfg, "lambda_v", 0.0))
        self._t6_cost_ema_mean = 0.0
        self._t6_cost_ema_var = 1.0
        self._t6_cost_scale = 1.0
        self._t6_pid_step = 0
        if self._t6_pid_controller is not None:
            self._lambda_v_dyn = float(self._t6_pid_controller.reset(lambda_init=float(getattr(self.cfg, "lambda_v", 0.0))))
        self._task_next_D = None
        self._task_next_C = None
        self._task_next_slot = None

        
        self.episode_cost_sum = 0.0
        self.episode_steps = 0
        self.cost_prev = None

        
        z4_action_mode = str(getattr(self.cfg, "z4_action_mode", "hierarchical")).strip().lower()
        if z4_action_mode == "position_only":
            _pa_dim = int(self.cfg.M) * 2  
            if bool(getattr(self.cfg, "z4_comp_meta_enable", False)):
                _pa_dim += 2  
        else:
            _pa_dim = int(self.cfg.M) * 2 + int(self.cfg.I)  # delta_q + comp_score = 20D
        self._prev_action_flat = np.zeros(_pa_dim, dtype=np.float64)
        # ===== [END Phase Z] =====

        
        self._ensure_task_cache(expected_slot=1, context="reset", force_resample=True)
        # [PATCH][PLOT]
        self._trace_reset()
        return self.obs_flat()

    # =========================
    # [PATCH][PLOT] trace APIs (used by eval animation; training unaffected)
    # =========================
    def enable_trace(self, enable: bool = True, keep_theta: bool = False, max_steps: Optional[int] = None) -> None:
        """Enable/disable per-step trace recording for animation."""
        self._trace_enabled = bool(enable)
        self._trace_keep_theta = bool(keep_theta)
        self._trace_max_steps = int(max_steps) if max_steps is not None else None
        self._trace_step = 0
        if self._trace_enabled:
            self._trace = {
                "L": float(self.cfg.L),
                "M": int(self.cfg.M),
                "I": int(self.cfg.I),
                "K": int(self.cfg.K),
                
                "Rc": float(getattr(self.cfg, "Rc", 0.0)),
                "H": float(getattr(self.cfg, "H", 0.0)),
                "v": None if (getattr(self.cfg, "v", None) is None) else np.asarray(self.cfg.v,
                                                                                    dtype=np.float64).reshape(2, ),
                "w": None,  # user positions (set on reset)
                "frames": [],  # list of {"t","q","a","z",(opt)"theta"}
            }
        else:
            self._trace = None

    def _trace_reset(self) -> None:
        if not self._trace_enabled:
            return
        if self._trace is None:
            self.enable_trace(True, keep_theta=self._trace_keep_theta, max_steps=self._trace_max_steps)
        self._trace_step = 0
        if self._trace is not None:
            self._trace["w"] = None if (self.w is None) else self.w.copy()
            self._trace["frames"] = []

    def get_trace(self, clear: bool = False) -> Optional[Dict]:
        tr = self._trace
        if clear:
            self._trace = None
            self._trace_enabled = False
        return tr

    def obs_flat(self) -> np.ndarray:
        if self.q is None or self.w is None:
            self.reset()

        dist2 = pairwise_dist2(self.q, self.w).T  # (I,M)
        dist2_n = dist2 / max(float(self.cfg.L) * float(self.cfg.L), 1e-9)

        g = self._compute_gains()  # (I,M)
        g_feat = np.log1p(np.maximum(g, 0.0))

        
        t_now = int(self.t)
        t_cap = int(max(int(self.cfg.T), 1))
        expected_slot = int(max(1, min(t_now + 1, t_cap)))
        D, C = self._ensure_task_cache(
            expected_slot=expected_slot,
            context="obs_flat",
            force_resample=False,
        )
        Dn = D / max(float(self.cfg.D_bits_base), 1e-9)
        Cn = C / max(float(self.cfg.C_cycles_base), 1e-9)

        
        try:
            fmax = np.asarray(getattr(self.cfg, "fmax"), dtype=np.float64).reshape(int(self.cfg.M),)
            fmin, fmaxv = float(np.min(fmax)), float(np.max(fmax))
            fmax_n = (fmax - fmin) / max(fmaxv - fmin, 1e-12)
        except Exception as exc:
            self._warn_once("obs_fmax_parse", f"fmax{exc}")
            fmax_n = np.zeros((int(self.cfg.M),), dtype=np.float64)

        
        
        enable_comp = 1.0 if bool(getattr(self.cfg, "enable_comp", True)) else 0.0
        enable_ris = 1.0 if bool(getattr(self.cfg, "enable_ris", True)) else 0.0
        try:
            obs_comp = getattr(self.cfg, "obs_enable_comp_flag", None)
            if obs_comp is not None:
                enable_comp = float(np.clip(float(obs_comp), 0.0, 1.0))
        except Exception as exc:
            self._warn_once("obs_comp_flag_parse", f"obs_enable_comp_flag {exc}")
        try:
            obs_ris = getattr(self.cfg, "obs_enable_ris_flag", None)
            if obs_ris is not None:
                enable_ris = float(np.clip(float(obs_ris), 0.0, 1.0))
        except Exception as exc:
            self._warn_once("obs_ris_flag_parse", f"obs_enable_ris_flag {exc}")

        # ===== [ A 3] =====
        
        L = float(self.cfg.L)
        M = int(self.cfg.M)
        dist_to_walls = np.zeros((M, 4), dtype=np.float64)  
        for m in range(M):
            x, y = self.q[m, 0], self.q[m, 1]
            dist_to_walls[m, 0] = x / L  
            dist_to_walls[m, 1] = (L - x) / L  
            dist_to_walls[m, 2] = y / L  
            dist_to_walls[m, 3] = (L - y) / L  

        
        min_wall_dist = np.min(dist_to_walls, axis=1)  # (M,)
        # ===== [END A 3] =====

        # ===== [ B 4] QoS/ / reward =====
        # 1) 1 AV
        Rc2_eff_local = float(self.cfg.Rc) * float(self.cfg.Rc) - float(self.cfg.H) * float(self.cfg.H)
        covered_mask = dist2 <= max(Rc2_eff_local, 0.0) + 1e-12
        coverage_fraction_obs = float(np.mean(np.any(covered_mask, axis=1))) if covered_mask.size > 0 else 0.0

        
        try:
            min_dist2 = np.min(dist2, axis=1)  # (I,)
            mean_min_dist = float(np.mean(np.sqrt(np.maximum(min_dist2, 0.0)))) / max(float(self.cfg.L), 1e-9)
        except Exception as exc:
            self._warn_once("obs_mean_min_dist", f"mean_min_dist0.0{exc}")
            mean_min_dist = 0.0

        
        qos_frac_est = 0.0
        mean_rate_ratio = 0.0
        try:
            g_best = np.max(np.asarray(g, dtype=np.float64), axis=1)  # (I,)
            snr_best = float(self.cfg.p) * g_best / max(float(self.cfg.N0), 1e-12)
            rate_best = float(self.cfg.B) * np.log2(1.0 + np.maximum(snr_best, 0.0))
            R_min = max(float(self.cfg.R_min), 1e-12)
            qos_frac_est = float(np.mean(rate_best >= R_min))
            mean_rate_ratio = float(np.mean(np.clip(rate_best / R_min, 0.0, 2.0))) / 2.0  # [0,1]
        except Exception as exc:
            self._warn_once("obs_qos_summary", f"QoS0.0{exc}")
            qos_frac_est = 0.0
            mean_rate_ratio = 0.0

        
        try:
            mean_log_gain_best = float(np.mean(np.log1p(np.maximum(np.max(g, axis=1), 0.0))))
        except Exception as exc:
            self._warn_once("obs_log_gain_best", f"log0.0{exc}")
            mean_log_gain_best = 0.0

        qos_summary = np.array(
            [
                float(np.clip(coverage_fraction_obs, 0.0, 1.0)),
                float(np.clip(mean_min_dist, 0.0, 1.0)),
                float(np.clip(qos_frac_est, 0.0, 1.0)),
                float(np.clip(mean_rate_ratio, 0.0, 1.0)),
                float(mean_log_gain_best),
            ],
            dtype=np.float64,
        )
        # ===== [END B 4] =====

        
        
        
        I_int = int(self.cfg.I)
        comp_gain_ratio = np.zeros((I_int,), dtype=np.float64)
        try:
            g_arr = np.asarray(g, dtype=np.float64)
            for i in range(I_int):
                g_row = g_arr[i]
                if covered_mask[i].any():
                    g_cand = g_row[covered_mask[i]]
                    if g_cand.size >= 2:
                        g_sorted = np.sort(g_cand)[::-1]
                        comp_gain_ratio[i] = float(g_sorted[1] / max(float(g_sorted[0]), 1e-12))
                    elif g_cand.size == 1:
                        comp_gain_ratio[i] = 0.0
        except Exception:
            comp_gain_ratio = np.zeros((I_int,), dtype=np.float64)
        

        
        
        rate_feedback = np.zeros((I_int,), dtype=np.float64)
        cost_feedback = np.zeros((2,), dtype=np.float64)  # [prev_cost_norm, delta_cost_norm]
        if bool(getattr(self.cfg, "obs_enable_rate_feedback", False)):
            try:
                last_rate = np.asarray(self._last_info.get("rate", np.zeros((I_int,), dtype=np.float64)), dtype=np.float64).reshape(I_int,)
                r_ref = float(max(float(getattr(self.cfg, "R_min", 1.0)) * 10.0, 1e-9))
                rate_feedback = np.clip(last_rate / r_ref, 0.0, 2.0)
            except Exception as exc:
                self._warn_once("obs_rate_feedback", f"rate{exc}")
                rate_feedback = np.zeros((I_int,), dtype=np.float64)

        if bool(getattr(self.cfg, "obs_enable_cost_feedback", False)):
            try:
                c_ref = float(max(float(getattr(self.cfg, "t7_cost_ref", 1.0)), 1e-9))
                c_prev = self._obs_prev_paper_cost
                c_prev_prev = self._obs_prev_prev_paper_cost
                if c_prev is not None and np.isfinite(float(c_prev)):
                    prev_cost_norm = float(c_prev) / c_ref
                else:
                    prev_cost_norm = 0.0
                if (
                    c_prev is not None
                    and c_prev_prev is not None
                    and np.isfinite(float(c_prev))
                    and np.isfinite(float(c_prev_prev))
                ):
                    delta_cost_norm = float(float(c_prev) - float(c_prev_prev)) / c_ref
                else:
                    delta_cost_norm = 0.0
                cost_feedback = np.asarray(
                    [
                        float(np.clip(prev_cost_norm, -5.0, 5.0)),
                        float(np.clip(delta_cost_norm, -5.0, 5.0)),
                    ],
                    dtype=np.float64,
                )
            except Exception as exc:
                self._warn_once("obs_cost_feedback", f"cost{exc}")
                cost_feedback = np.zeros((2,), dtype=np.float64)
        

        flat = np.concatenate(
            [
                (self.q / float(self.cfg.L)).reshape(-1),
                (self.w / float(self.cfg.L)).reshape(-1),
                dist2_n.reshape(-1),
                g_feat.reshape(-1),
                Dn.reshape(-1),
                Cn.reshape(-1),
                fmax_n.reshape(-1),
                np.array([enable_comp, enable_ris], dtype=np.float64),
                np.array([self.t / max(float(self.cfg.T), 1), float(self.cfg.load_scale)], dtype=np.float64),
                # ===== [ A 3] =====
                dist_to_walls.reshape(-1),  # (M*4,) UAV inf
                min_wall_dist.reshape(-1),  # (M,) UAV inf
                # ===== [END A 3] =====
                
                qos_summary.reshape(-1),
                # ===== [END B 4] =====
                
                comp_gain_ratio.reshape(-1),  
                
                
                
                rate_feedback.reshape(-1) if bool(getattr(self.cfg, "obs_enable_rate_feedback", False)) else np.zeros((0,), dtype=np.float64),
                cost_feedback.reshape(-1) if bool(getattr(self.cfg, "obs_enable_cost_feedback", False)) else np.zeros((0,), dtype=np.float64),
                
                
                np.clip(self._prev_action_flat, -1.0, 1.0).reshape(-1) if bool(getattr(self.cfg, "obs_enable_prev_action", False)) else np.zeros((0,), dtype=np.float64),
                # ===== [END Phase Z] =====
            ],
            axis=0,
        )
        
        obs_clip = float(getattr(self.cfg, "obs_clip_range", 20.0))
        if np.isfinite(obs_clip) and obs_clip > 0.0:
            flat = np.clip(flat, -obs_clip, obs_clip)
        return flat.astype(np.float32)

    # --------- action helpers ---------
    def random_action(self) -> np.ndarray:
        M, I = int(self.cfg.M), int(self.cfg.I)
        
        
        delta = self.rng.uniform(-1.0, 1.0, size=(M, 2)).astype(np.float64)

        action_space_mode = str(getattr(self.cfg, "action_space_mode", "full")).strip().lower()
        if action_space_mode == "hierarchical":
            z4_action_mode = str(getattr(self.cfg, "z4_action_mode", "hierarchical")).strip().lower()
            if z4_action_mode == "position_only":
                if bool(getattr(self.cfg, "z4_comp_meta_enable", False)):
                    comp_meta = self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float64)
                    return np.concatenate([delta.reshape(-1), comp_meta.reshape(-1)], axis=0)
                return delta.reshape(-1)
            comp_score = self.rng.uniform(-1.0, 1.0, size=(I,)).astype(np.float64)
            return np.concatenate([delta.reshape(-1), comp_score.reshape(-1)], axis=0)

        s_a = self.rng.normal(0.0, 1.0, size=(I, M)).astype(np.float64)

        action_score_mode = str(getattr(self.cfg, "action_score_mode", "separate")).strip().lower()
        theta_mode = str(getattr(self.cfg, "theta_mode", "solver")).strip().lower()
        use_separate_z = action_score_mode != "shared"
        use_theta = theta_mode == "policy"

        parts = [delta.reshape(-1), s_a.reshape(-1)]
        if use_separate_z:
            s_z = self.rng.normal(0.0, 1.0, size=(I, M)).astype(np.float64)
            parts.append(s_z.reshape(-1))
        if use_theta:
            theta_raw = np.abs(self.rng.normal(1.0, 0.5, size=(I,))).astype(np.float64) + 1e-3
            parts.append(theta_raw.reshape(-1))
        return np.concatenate(parts, axis=0)

    def _parse_action(self, action: np.ndarray):
        M, I = int(self.cfg.M), int(self.cfg.I)
        a = np.asarray(action, dtype=np.float64).reshape(-1)

        action_space_mode = str(getattr(self.cfg, "action_space_mode", "full")).strip().lower()
        if action_space_mode == "hierarchical":
            p0 = 0
            if a.size < 2 * M:
                a = np.zeros((2 * M,), dtype=np.float64)
            delta = a[p0:p0 + 2 * M].reshape(M, 2)
            p0 += 2 * M
            z4_action_mode = str(getattr(self.cfg, "z4_action_mode", "hierarchical")).strip().lower()
            if z4_action_mode == "position_only":
                comp_meta = np.zeros((2,), dtype=np.float64)
                if bool(getattr(self.cfg, "z4_comp_meta_enable", False)):
                    if a.size >= p0 + 2:
                        comp_meta = np.asarray(a[p0:p0 + 2], dtype=np.float64).reshape(2,)
                    comp_meta = np.clip(comp_meta, -1.0, 1.0)
                comp_score = np.zeros((I,), dtype=np.float64)
                theta_raw = np.ones((I,), dtype=np.float64)
                return delta, comp_score, None, theta_raw, comp_meta
            if a.size >= p0 + I:
                comp_score = a[p0:p0 + I].reshape(I)
            else:
                comp_score = np.zeros((I,), dtype=np.float64)
            theta_raw = np.ones((I,), dtype=np.float64)
            comp_meta = np.zeros((2,), dtype=np.float64)
            return delta, comp_score, None, theta_raw, comp_meta

        action_score_mode = str(getattr(self.cfg, "action_score_mode", "separate")).strip().lower()
        theta_mode = str(getattr(self.cfg, "theta_mode", "solver")).strip().lower()
        use_separate_z = action_score_mode != "shared"
        use_theta = theta_mode == "policy"

        
        
        p0 = 0

        if a.size < 2 * M:
            a = np.zeros((2 * M,), dtype=np.float64)
        delta = a[p0:p0 + 2 * M].reshape(M, 2)
        p0 += 2 * M

        
        if a.size >= p0 + I * M:
            s_a = a[p0:p0 + I * M].reshape(I, M)
            p0 += I * M
        else:
            s_a = np.zeros((I, M), dtype=np.float64)

        
        if use_separate_z:
            if a.size >= p0 + I * M:
                s_z = a[p0:p0 + I * M].reshape(I, M)
                p0 += I * M
            else:
                s_z = np.zeros((I, M), dtype=np.float64)
        else:
            # shared _a s_z
            s_z = s_a

        
        if use_theta:
            if a.size >= p0 + I:
                theta_raw = a[p0:p0 + I].reshape(I)
            else:
                theta_raw = np.ones((I,), dtype=np.float64)
            theta_raw = np.abs(theta_raw) + 1e-6
        else:
            theta_raw = np.ones((I,), dtype=np.float64)

        comp_meta = np.zeros((2,), dtype=np.float64)
        return delta, s_a, s_z, theta_raw, comp_meta

    def _solve_assignment_min_cost(self, cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hungarianscipy
        """
        c = np.asarray(cost, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError(f"assignment cost: shape={c.shape}")
        n_row, n_col = int(c.shape[0]), int(c.shape[1])
        if n_row <= 0 or n_col <= 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

        if _scipy_linear_sum_assignment is not None:
            r, col = _scipy_linear_sum_assignment(c)
            return np.asarray(r, dtype=np.int64), np.asarray(col, dtype=np.int64)

        solver_mode = str(getattr(self.cfg, "t6_hier_assoc_solver", "greedy")).strip().lower()
        reward_design = str(getattr(self.cfg, "reward_design", "")).strip().lower()
        action_mode = str(getattr(self.cfg, "action_space_mode", "")).strip().lower()
        is_t6_contract = bool((reward_design == "t6_potential_lagrangian") and (action_mode == "hierarchical"))
        if bool(getattr(self.cfg, "t6_hier_assoc_require_scipy", False)) or (solver_mode == "hungarian" and is_t6_contract):
            raise RuntimeError(
                " scipy  Hungarian  scipy"
                " scipy"
            )

        
        self._warn_once("hungarian_fallback", "scipyt6_hier_assoc_solver=hungarian")
        used: set[int] = set()
        rows: List[int] = []
        cols: List[int] = []
        for i in range(n_row):
            best_j = -1
            best_v = float("inf")
            for j in range(n_col):
                if j in used:
                    continue
                v = float(c[i, j])
                if (v < best_v - 1e-12) or (abs(v - best_v) <= 1e-12 and j < best_j):
                    best_v = v
                    best_j = j
            if best_j < 0:
                continue
            used.add(int(best_j))
            rows.append(int(i))
            cols.append(int(best_j))
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)

    def _hungarian_primary_assignment(
        self,
        g: np.ndarray,
        covered: np.ndarray,
        dist2: np.ndarray,
        task_weights: np.ndarray,
        use_task_weight: bool,
    ) -> np.ndarray:
        """
         + UAV
        """
        I, M = int(self.cfg.I), int(self.cfg.M)
        DIST_PENALTY_WEIGHT = 0.10
        l2 = float(max(float(getattr(self.cfg, "L", 1.0)) ** 2, 1e-9))
        task_weight_arr = np.asarray(task_weights, dtype=np.float64).reshape(I,)
        task_weight_arr = np.maximum(task_weight_arr, 1e-9)
        task_weight_ref = float(np.mean(task_weight_arr)) if float(np.mean(task_weight_arr)) > 1e-9 else 1.0

        
        fmax = np.asarray(getattr(self.cfg, "fmax"), dtype=np.float64).reshape(M,)
        fmax = np.maximum(fmax, 1e-9)
        cap_f = float(np.sum(fmax))
        cap_raw = (fmax / max(cap_f, 1e-9)) * float(I)
        cap = np.floor(cap_raw).astype(np.int64)
        if I >= M:
            cap = np.maximum(cap, 1)
        rem = int(max(I - int(np.sum(cap)), 0))
        if rem > 0:
            frac = np.asarray(cap_raw - cap, dtype=np.float64)
            ord_idx = np.argsort(frac)[::-1]
            for k in range(rem):
                cap[int(ord_idx[k % M])] += 1
        slot_uav: List[int] = []
        for m in range(M):
            slot_uav.extend([int(m)] * int(max(cap[m], 0)))
        if len(slot_uav) < I:
            for k in range(I - len(slot_uav)):
                slot_uav.append(int(k % M))
        slot_uav = slot_uav[: max(len(slot_uav), I)]

        S = int(len(slot_uav))
        high = 1e6
        cost = np.full((I, S), high, dtype=np.float64)
        g_arr = np.asarray(g, dtype=np.float64).reshape(I, M)
        d_arr = np.asarray(dist2, dtype=np.float64).reshape(I, M)
        cov = np.asarray(covered, dtype=bool).reshape(I, M)
        for i in range(I):
            task_weight_mul = float(task_weight_arr[i] / max(task_weight_ref, 1e-9)) if use_task_weight else 1.0
            for s, m in enumerate(slot_uav):
                base = -np.log1p(max(float(g_arr[i, int(m)]), 0.0))
                d_pen = DIST_PENALTY_WEIGHT * float(d_arr[i, int(m)] / l2)
                if cov[i, int(m)]:
                    cost[i, s] = float((base + d_pen) * task_weight_mul)
                else:
                    cost[i, s] = float(high + d_pen)

        rows, cols = self._solve_assignment_min_cost(cost)
        primary = np.full((I,), -1, dtype=np.int64)
        for rr, cc in zip(rows.tolist(), cols.tolist()):
            primary[int(rr)] = int(slot_uav[int(cc)])
        for i in range(I):
            if int(primary[i]) < 0:
                primary[i] = int(np.argmin(np.asarray(d_arr[i], dtype=np.float64)))
        return primary.astype(np.int64)

    def _rebalance_primary_assignment(
        self,
        primary: np.ndarray,
        covered: np.ndarray,
        dist2: np.ndarray,
        g: np.ndarray,
        task_weights: np.ndarray,
        load_alpha: float,
    ) -> np.ndarray:
        """
        Hungarian
        //UAV
        """
        if not bool(getattr(self.cfg, "t6_hier_assoc_rebalance_enable", False)):
            return np.asarray(primary, dtype=np.int64).reshape(int(self.cfg.I),)

        I, M = int(self.cfg.I), int(self.cfg.M)
        DIST_PENALTY_WEIGHT = 0.10
        rounds = int(max(int(getattr(self.cfg, "t6_hier_assoc_rebalance_rounds", 1)), 0))
        tol = float(max(float(getattr(self.cfg, "t6_hier_assoc_rebalance_tol", 0.0)), 0.0))
        rebalance_weight = float(max(float(getattr(self.cfg, "t6_hier_assoc_rebalance_weight", 1.0)), 0.0))
        w_load = float(max(float(load_alpha), 0.0) * rebalance_weight)
        if rounds <= 0 or M <= 1 or w_load <= 1e-12:
            return np.asarray(primary, dtype=np.int64).reshape(I,)

        prim = np.asarray(primary, dtype=np.int64).reshape(I,).copy()
        cov = np.asarray(covered, dtype=bool).reshape(I, M)
        d_arr = np.asarray(dist2, dtype=np.float64).reshape(I, M)
        g_arr = np.asarray(g, dtype=np.float64).reshape(I, M)
        task_weight_arr = np.asarray(task_weights, dtype=np.float64).reshape(I,)
        task_weight_arr = np.maximum(task_weight_arr, 1e-9)
        l2 = float(max(float(getattr(self.cfg, "L", 1.0)) ** 2, 1e-9))

        load_ref = float(np.sum(task_weight_arr) / max(M, 1))
        if load_ref <= 1e-12:
            return prim
        overload_th = float((1.0 + tol) * load_ref)

        for _ in range(rounds):
            load = np.zeros((M,), dtype=np.float64)
            for i in range(I):
                m = int(prim[i])
                if m < 0 or m >= M:
                    m = int(np.argmin(np.asarray(d_arr[i], dtype=np.float64)))
                    prim[i] = m
                load[m] += float(task_weight_arr[i])

            changed = 0
            for idx in np.argsort(task_weight_arr)[::-1].astype(np.int64):
                i = int(idx)
                cur = int(prim[i])
                if cur < 0 or cur >= M:
                    continue
                if float(load[cur]) <= overload_th:
                    continue
                cand = np.where(cov[i])[0]
                if cand.size <= 0:
                    continue

                cur_obj = (
                    np.log1p(max(float(g_arr[i, cur]), 0.0))
                    - DIST_PENALTY_WEIGHT * float(d_arr[i, cur] / l2)
                    - w_load * float(load[cur] / max(load_ref, 1e-9))
                )
                best_m = int(cur)
                best_obj = float(cur_obj)

                for m in np.asarray(cand, dtype=np.int64).tolist():
                    mm = int(m)
                    pred_load = float(load[mm] + (float(task_weight_arr[i]) if mm != cur else 0.0))
                    obj = (
                        np.log1p(max(float(g_arr[i, mm]), 0.0))
                        - DIST_PENALTY_WEIGHT * float(d_arr[i, mm] / l2)
                        - w_load * float(pred_load / max(load_ref, 1e-9))
                    )
                    if (obj > best_obj + 1e-12) or (abs(obj - best_obj) <= 1e-12 and mm < best_m):
                        best_obj = float(obj)
                        best_m = int(mm)

                if best_m != cur:
                    load[cur] = float(max(0.0, load[cur] - float(task_weight_arr[i])))
                    load[best_m] = float(load[best_m] + float(task_weight_arr[i]))
                    prim[i] = int(best_m)
                    changed += 1

            if changed <= 0:
                break
        return prim.astype(np.int64)

    def _build_assoc_from_comp_score(
        self,
        comp_score: np.ndarray,
        g: np.ndarray,
        covered: np.ndarray,
        dist2: np.ndarray,
        assoc_mode: str,
        k_eff: int,
        comp_gate_ratio: float,
        task_weights: Optional[np.ndarray] = None,
        comp_temp_override: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
        """
        Phase T comp_score 

        
        - a_mat: (I,M) hard=0/1soft=
        - primary_uav: (I,) UAVz
        - vio_no_cover: 
        - comp_gate_snr_candidates/pruned: SNR
        """
        I, M = int(self.cfg.I), int(self.cfg.M)
        a_dtype = np.int64 if str(assoc_mode) == "hard" else np.float64
        a_mat = np.zeros((I, M), dtype=a_dtype)
        primary_uav = np.full((I,), -1, dtype=np.int64)
        vio_no_cover = 0
        comp_gate_snr_candidates = 0
        comp_gate_snr_pruned = 0

        enable_comp = bool(getattr(self.cfg, "enable_comp", True))
        if comp_temp_override is not None and np.isfinite(float(comp_temp_override)) and float(comp_temp_override) > 0.0:
            temp = float(max(float(comp_temp_override), 1e-6))
        else:
            temp = float(self._resolve_comp_score_temp())
        self._comp_score_temp_runtime = float(temp)
        eval_hard = bool(getattr(self.cfg, "comp_score_eval_hard", True))
        solver_mode = str(getattr(self.cfg, "t6_hier_assoc_solver", "greedy")).strip().lower()
        if solver_mode not in ("greedy", "hungarian"):
            solver_mode = "greedy"
        assoc_load_alpha = float(max(float(getattr(self.cfg, "t6_hier_assoc_load_alpha", 0.0)), 0.0))
        use_task_weight = bool(getattr(self.cfg, "t6_hier_assoc_use_task_weight", True))
        comp_score_v = np.asarray(comp_score, dtype=np.float64).reshape(I,)
        l2 = float(max(float(getattr(self.cfg, "L", 1.0)) ** 2, 1e-9))
        
        
        comp_gate_ratio_eff = float(comp_gate_ratio)
        if comp_gate_ratio_eff <= 0.0:
            reward_design_now = str(getattr(self.cfg, "reward_design", "")).strip().lower()
            action_mode_now = str(getattr(self.cfg, "action_space_mode", "")).strip().lower()
            if reward_design_now == "z_shifted_log" and action_mode_now == "hierarchical":
                comp_gate_ratio_eff = float(max(float(getattr(self.cfg, "z1_comp_gate_ratio_fallback", 0.35)), 0.0))
                if comp_gate_ratio_eff > 0.0:
                    self._warn_once(
                        "z1_comp_gate_ratio_fallback",
                        f" comp_gate_ratio<=0 Z1  ratio={comp_gate_ratio_eff:.3f}",
                    )
        if task_weights is None:
            task_w = np.ones((I,), dtype=np.float64)
        else:
            task_w = np.asarray(task_weights, dtype=np.float64).reshape(I,)
            task_w = np.maximum(task_w, 1e-9)
        order_users = np.arange(I, dtype=np.int64)
        if use_task_weight:
            order_users = np.argsort(task_w)[::-1].astype(np.int64)
        load_ref = float(np.mean(task_w)) if float(np.mean(task_w)) > 1e-9 else 1.0
        uav_assoc_load = np.zeros((M,), dtype=np.float64)
        primary_fixed: Optional[np.ndarray] = None
        if solver_mode == "hungarian":
            primary_fixed = self._hungarian_primary_assignment(
                g=np.asarray(g, dtype=np.float64),
                covered=np.asarray(covered, dtype=bool),
                dist2=np.asarray(dist2, dtype=np.float64),
                task_weights=task_w,
                use_task_weight=use_task_weight,
            )
            primary_fixed = self._rebalance_primary_assignment(
                primary=primary_fixed,
                covered=np.asarray(covered, dtype=bool),
                dist2=np.asarray(dist2, dtype=np.float64),
                g=np.asarray(g, dtype=np.float64),
                task_weights=task_w,
                load_alpha=assoc_load_alpha,
            )

        for idx in order_users:
            i = int(idx)
            cand = np.where(covered[i])[0]
            base = np.log1p(np.maximum(np.asarray(g[i], dtype=np.float64), 0.0))
            load_pen = float(assoc_load_alpha) * (uav_assoc_load / max(load_ref, 1e-9))
            score = base - 0.1 * (np.asarray(dist2[i], dtype=np.float64) / l2) - load_pen
            if solver_mode == "hungarian" and primary_fixed is not None:
                primary = int(primary_fixed[i])
                if cand.size == 0:
                    vio_no_cover += 1
                
                if cand.size > 0 and int(primary) not in cand.tolist():
                    ord_cov = np.argsort(np.asarray(score[cand], dtype=np.float64))[::-1]
                    primary = int(cand[ord_cov[0]])
                if cand.size == 0:
                    primary = int(np.argmin(dist2[i]))
            else:
                if cand.size == 0:
                    nearest = int(np.argmin(dist2[i]))
                    primary_uav[i] = nearest
                    a_mat[i, nearest] = 1 if a_dtype == np.int64 else 1.0
                    uav_assoc_load[nearest] = float(uav_assoc_load[nearest] + task_w[i])
                    vio_no_cover += 1
                    continue
                order = np.argsort(np.asarray(score[cand], dtype=np.float64))[::-1]
                cand_sorted = cand[order]
                primary = int(cand_sorted[0])
            primary_uav[i] = primary
            a_mat[i, primary] = 1 if a_dtype == np.int64 else 1.0
            if 0 <= int(primary) < M:
                uav_assoc_load[int(primary)] = float(uav_assoc_load[int(primary)] + task_w[i])

            n_cand = int(np.asarray(cand, dtype=np.int64).size)
            if (not enable_comp) or int(k_eff) <= 1 or n_cand < 2:
                continue

            if solver_mode == "hungarian":
                sec_candidates = np.asarray(cand, dtype=np.int64)
                if sec_candidates.size <= 1:
                    continue
                sec_score = np.asarray(score[sec_candidates], dtype=np.float64)
                sec_order = np.argsort(sec_score)[::-1]
                cand_sorted_local = sec_candidates[sec_order].astype(np.int64)
            else:
                cand_sorted_local = np.asarray(cand_sorted, dtype=np.int64)

            
            extra_pool = [int(m) for m in cand_sorted_local.tolist() if int(m) != int(primary)]
            if len(extra_pool) <= 0:
                continue

            g_row = np.asarray(g[i], dtype=np.float64).reshape(-1)
            g_primary = float(max(g_row[int(primary)], 0.0))
            if float(comp_gate_ratio_eff) > 0.0:
                if g_primary <= 0.0:
                    continue
                extra_pool = [
                    int(m)
                    for m in extra_pool
                    if float(max(g_row[int(m)], 0.0)) >= float(comp_gate_ratio_eff) * g_primary
                ]
                if len(extra_pool) <= 0:
                    continue

            n_extra_max = int(min(max(int(k_eff) - 1, 0), len(extra_pool)))
            if n_extra_max <= 0:
                continue

            comp_prob = float(1.0 / (1.0 + np.exp(-float(comp_score_v[i]) / temp)))

            if str(assoc_mode) == "hard":
                hard_thr = (np.arange(n_extra_max, dtype=np.float64) + 1.0) / float(n_extra_max + 1)
                if bool(eval_hard):
                    n_extra_on = int(np.sum(float(comp_prob) > hard_thr))
                else:
                    n_extra_on = int(np.sum(float(comp_prob) >= hard_thr))
                n_extra_on = int(min(max(n_extra_on, 0), n_extra_max))
                if n_extra_on <= 0:
                    continue
                chosen = np.asarray([int(primary)] + extra_pool[:n_extra_on], dtype=np.int64)
                comp_gate_snr_candidates += 1
                chosen, pruned = self._gate_comp_set_by_snr(np.asarray(g[i], dtype=np.float64), chosen)
                if pruned:
                    comp_gate_snr_pruned += 1
                primary_uav[i] = int(chosen[0])
                if int(chosen[0]) != int(primary):
                    if 0 <= int(primary) < M:
                        uav_assoc_load[int(primary)] = float(max(0.0, uav_assoc_load[int(primary)] - task_w[i]))
                    if 0 <= int(chosen[0]) < M:
                        uav_assoc_load[int(chosen[0])] = float(uav_assoc_load[int(chosen[0])] + task_w[i])
                a_mat[i, :] = 0
                a_mat[i, chosen] = 1
                if chosen.size >= 2:
                    extra_idx = np.asarray(chosen[1:], dtype=np.int64).reshape(-1)
                    extra_load_share = float(0.5 * task_w[i] / max(int(extra_idx.size), 1))
                    for mm in extra_idx.tolist():
                        uav_assoc_load[int(mm)] = float(uav_assoc_load[int(mm)] + extra_load_share)
                continue

            
            
            if float(comp_prob) <= 1e-6:
                continue
            n_extra_on = int(max(1, int(np.ceil(float(comp_prob) * float(n_extra_max)))))
            n_extra_on = int(min(max(n_extra_on, 0), n_extra_max))
            if n_extra_on <= 0:
                continue
            chosen = np.asarray([int(primary)] + extra_pool[:n_extra_on], dtype=np.int64)
            comp_gate_snr_candidates += 1
            chosen, pruned = self._gate_comp_set_by_snr(np.asarray(g[i], dtype=np.float64), chosen)
            if pruned:
                comp_gate_snr_pruned += 1
                continue
            if chosen.size >= 1:
                primary_uav[i] = int(chosen[0])
                if int(chosen[0]) != int(primary):
                    if 0 <= int(primary) < M:
                        uav_assoc_load[int(primary)] = float(max(0.0, uav_assoc_load[int(primary)] - task_w[i]))
                    if 0 <= int(chosen[0]) < M:
                        uav_assoc_load[int(chosen[0])] = float(uav_assoc_load[int(chosen[0])] + task_w[i])
                    a_mat[i, :] = 0.0
                    a_mat[i, int(chosen[0])] = 1.0
                    primary = int(chosen[0])
            if chosen.size >= 2:
                
                extra_idx = np.asarray(chosen[1:], dtype=np.int64).reshape(-1)
                rank_decay = float(max(float(getattr(self.cfg, "comp_multilink_rank_decay", 0.35)), 0.0))
                rank_pos = np.arange(int(extra_idx.size), dtype=np.float64)
                w_raw = np.exp(-rank_decay * rank_pos)
                w_raw = w_raw / (float(np.sum(w_raw)) + 1e-12)
                w_extra = float(comp_prob) * w_raw
                a_mat[i, extra_idx] = w_extra.astype(np.float64)
                for jj, mm in enumerate(extra_idx.tolist()):
                    uav_assoc_load[int(mm)] = float(
                        uav_assoc_load[int(mm)] + float(w_extra[int(jj)]) * 0.5 * task_w[i]
                    )

        return a_mat, primary_uav, int(vio_no_cover), int(comp_gate_snr_candidates), int(comp_gate_snr_pruned)

    def _select_compute_uav_from_primary(self, primary_uav: np.ndarray, dist2: np.ndarray) -> np.ndarray:
        """Phase TUAV"""
        I, M = int(self.cfg.I), int(self.cfg.M)
        z_mat = np.zeros((I, M), dtype=np.int64)
        prim = np.asarray(primary_uav, dtype=np.int64).reshape(I,)
        for i in range(I):
            m = int(prim[i])
            if m < 0 or m >= M:
                m = int(np.argmin(dist2[i]))
            z_mat[i, m] = 1
        return z_mat

    def _select_compute_uav_hierarchical(
        self,
        primary_uav: np.ndarray,
        a_mat: np.ndarray,
        dist2: np.ndarray,
        g: np.ndarray,
        C: np.ndarray,
    ) -> np.ndarray:
        """
        PhaseT6
        - `primary`UAV
        - `load_aware`UAV
        """
        mode = str(getattr(self.cfg, "t6_hier_compute_mode", "load_aware")).strip().lower()
        if mode == "primary":
            return self._select_compute_uav_from_primary(primary_uav=primary_uav, dist2=dist2)

        I, M = int(self.cfg.I), int(self.cfg.M)
        z_mat = np.zeros((I, M), dtype=np.int64)
        fmax_arr = np.asarray(self.cfg.fmax, dtype=np.float64).reshape(M,)
        cpu_slot = float(max(float(getattr(self.cfg, "cpu_slot_time", 1.0)), 1e-9))
        l2 = float(max(float(getattr(self.cfg, "L", 1.0)) ** 2, 1e-9))
        w_load = float(max(float(getattr(self.cfg, "t6_hier_compute_load_weight", 1.0)), 0.0))
        w_dist = float(max(float(getattr(self.cfg, "t6_hier_compute_dist_weight", 0.15)), 0.0))
        w_gain = float(max(float(getattr(self.cfg, "t6_hier_compute_gain_weight", 0.10)), 0.0))
        sort_demand = bool(getattr(self.cfg, "t6_hier_compute_sort_demand_desc", True))
        c_arr = np.asarray(C, dtype=np.float64).reshape(I,)
        c_arr = np.maximum(c_arr, 0.0)
        dyn_load = np.zeros((M,), dtype=np.float64)

        order = np.arange(I, dtype=np.int64)
        if sort_demand:
            order = np.argsort(c_arr)[::-1].astype(np.int64)

        for idx in order:
            i = int(idx)
            cand = np.where(np.asarray(a_mat[i], dtype=np.float64) > 0.0)[0]
            if cand.size == 0:
                p = int(np.asarray(primary_uav, dtype=np.int64).reshape(I,)[i])
                cand = np.asarray([p], dtype=np.int64) if 0 <= p < M else np.asarray([int(np.argmin(dist2[i]))], dtype=np.int64)

            best_m = int(cand[0])
            best_obj = float("inf")
            for m in cand:
                mm = int(m)
                unit_load = float(c_arr[i] / max(float(fmax_arr[mm]) * cpu_slot, 1e-9))
                load_obj = float(w_load * (dyn_load[mm] + unit_load))
                dist_obj = float(w_dist * (float(dist2[i, mm]) / l2))
                gain_obj = float(-w_gain * np.log1p(max(float(g[i, mm]), 0.0)))
                obj = float(load_obj + dist_obj + gain_obj)
                if (obj < best_obj - 1e-12) or (abs(obj - best_obj) <= 1e-12 and mm < best_m):
                    best_obj = obj
                    best_m = mm

            z_mat[i, best_m] = 1
            dyn_load[best_m] = float(
                dyn_load[best_m] + float(c_arr[i] / max(float(fmax_arr[best_m]) * cpu_slot, 1e-9))
            )

        return z_mat

    
    def _weight_flags(self, weight_mode: str) -> Tuple[bool, bool, str]:
        """
        
        """
        wm = str(weight_mode or "auto").strip().lower()
        if wm in ("fixed",):
            return False, False, "fixed"
        if wm in ("auto_std", "std", "auto-scale"):
            return False, True, "auto_std"
        # P1 eight_mode='auto' dynamic-improve + std
        if wm in ("auto", "auto_improve", "auto_dyn", "dyn"):
            return True, True, "auto_improve"
        
        return True, True, "auto_improve"

    def _dyn_update(self, zT_pol: float, zE_pol: float) -> Tuple[float, float, float, float, str]:
        """
        
        """
        beta = float(getattr(self.cfg, "dyn_beta", 0.98))
        eta = float(getattr(self.cfg, "dyn_eta", 0.05))
        wmin = float(getattr(self.cfg, "dyn_w_min", 0.20))
        wmax = float(getattr(self.cfg, "dyn_w_max", 0.80))
        margin = float(getattr(self.cfg, "dyn_margin", 0.02))

        
        if self._dyn_perf_T is None or self._dyn_perf_E is None:
            self._dyn_perf_T = float(zT_pol)
            self._dyn_perf_E = float(zE_pol)
            self._dyn_prev_T = float(zT_pol)
            self._dyn_prev_E = float(zE_pol)
            self._dyn_last_reason = "init_ema"
            return self.wT_dyn, self.wE_dyn, 0.0, 0.0, self._dyn_last_reason

        
        prevT = float(self._dyn_perf_T)
        prevE = float(self._dyn_perf_E)
        self._dyn_perf_T = beta * prevT + (1.0 - beta) * float(zT_pol)
        self._dyn_perf_E = beta * prevE + (1.0 - beta) * float(zE_pol)

        
        upd_every = int(getattr(self.cfg, "dyn_update_every", 1))
        self._dyn_step += 1
        if upd_every > 1 and (self._dyn_step % upd_every != 0):
            self._dyn_last_reason = "hold"
            impT = float(self._dyn_prev_T - self._dyn_perf_T) if self._dyn_prev_T is not None else 0.0
            impE = float(self._dyn_prev_E - self._dyn_perf_E) if self._dyn_prev_E is not None else 0.0
            return self.wT_dyn, self.wE_dyn, impT, impE, self._dyn_last_reason

        
        impT = float((self._dyn_prev_T if self._dyn_prev_T is not None else self._dyn_perf_T) - self._dyn_perf_T)
        impE = float((self._dyn_prev_E if self._dyn_prev_E is not None else self._dyn_perf_E) - self._dyn_perf_E)

        
        
        reason = "equalize"
        d = impT - impE  # positive => T improved more; E lags
        if d > margin:
            
            self.wT_dyn = float(np.clip(self.wT_dyn - eta, wmin, wmax))
            reason = "E_lags"
        elif d < -margin:
            # -> wT
            self.wT_dyn = float(np.clip(self.wT_dyn + eta, wmin, wmax))
            reason = "T_lags"
        else:
            
            wT0, _wE0 = _normalize_w(self.cfg.w_delay, self.cfg.w_energy)
            self.wT_dyn = float(np.clip((1.0 - eta) * self.wT_dyn + eta * wT0, wmin, wmax))
            reason = "towards_base"

        self.wE_dyn = float(1.0 - self.wT_dyn)

        
        self._dyn_prev_T = float(self._dyn_perf_T)
        self._dyn_prev_E = float(self._dyn_perf_E)
        self._dyn_last_reason = reason
        return self.wT_dyn, self.wE_dyn, impT, impE, reason

    # ===== [STAGE 1+2 UPGRADE] =====
    def _t6_harmonized_cost(self, info: Dict[str, Any]) -> Dict[str, float]:
        """
        PhaseT6-8delay/energy/cost 
        T6paper_cost
        """
        mean_t = float(info.get("mean_T_off", 0.0))
        mean_e = float(info.get("mean_energy", 0.0))
        t_scale = float(max(float(getattr(self.cfg, "T_scale", 1.0)), 1e-9))
        e_scale = float(max(float(getattr(self.cfg, "E_scale", 1.0)), 1e-9))
        w_t, w_e = _normalize_w(float(getattr(self.cfg, "w_delay", 0.5)), float(getattr(self.cfg, "w_energy", 0.5)))

        delay_norm = float(mean_t / t_scale)
        energy_norm = float(mean_e / e_scale)
        legacy_cost = float(w_t * delay_norm + w_e * energy_norm)

        enable = bool(getattr(self.cfg, "t6_metric_calib_enable", False))
        if not enable:
            return {
                "cost": legacy_cost,
                "legacy_cost": legacy_cost,
                "delay_norm": delay_norm,
                "energy_norm": energy_norm,
                "delay_calib": delay_norm,
                "energy_calib": energy_norm,
                "calib_enable": 0.0,
            }

        d_ref = float(max(float(getattr(self.cfg, "t6_metric_calib_delay_ref", 1.0)), 1e-9))
        e_ref = float(max(float(getattr(self.cfg, "t6_metric_calib_energy_ref", 1.0)), 1e-9))
        c_clip = float(max(float(getattr(self.cfg, "t6_metric_calib_clip", 4.0)), 1e-6))
        mix = float(np.clip(float(getattr(self.cfg, "t6_metric_calib_mix", 1.0)), 0.0, 1.0))

        delay_calib = float(np.clip(delay_norm / d_ref, 0.0, c_clip))
        energy_calib = float(np.clip(energy_norm / e_ref, 0.0, c_clip))
        calib_cost = float(w_t * delay_calib + w_e * energy_calib)
        cost = float((1.0 - mix) * legacy_cost + mix * calib_cost)
        return {
            "cost": cost,
            "legacy_cost": legacy_cost,
            "delay_norm": delay_norm,
            "energy_norm": energy_norm,
            "delay_calib": delay_calib,
            "energy_calib": energy_calib,
            "calib_enable": 1.0,
        }

    def _compute_potential(self, info: Dict) -> float:
        """
        
        """
        potential = 0.0

        
        coverage_ratio = float(info.get('coverage_fraction', 0.0))
        potential += 2.0 * coverage_ratio  

        
        mean_rate = float(info.get('mean_rate', 0.0))
        R_min = float(self.cfg.R_min) if float(self.cfg.R_min) > 0 else 1e-9
        qos_ratio = min(mean_rate / R_min, 1.0)
        potential += 1.5 * qos_ratio  

        
        energy_per_bit = float(info.get('energy_per_bit', 1e9))
        
        energy_norm = 1.0 / (1.0 + energy_per_bit / 1e-6)
        potential += 0.5 * energy_norm  

        return potential

    def _current_reward_proximity_weight(self) -> float:
        """
         proximity 
        """
        w_start = float(getattr(self.cfg, "reward_proximity_weight", 0.0))
        if w_start <= 0.0:
            return 0.0
        w_final_cfg = float(getattr(self.cfg, "reward_proximity_weight_final", 0.0))
        
        w_final = w_start if w_final_cfg <= 0.0 else max(w_final_cfg, 0.0)
        s_step = int(max(int(getattr(self.cfg, "reward_proximity_decay_start_step", 0)), 0))
        e_step = int(max(int(getattr(self.cfg, "reward_proximity_decay_end_step", 0)), 0))
        if e_step <= s_step:
            return float(w_start)

        t_step = int(max(getattr(self, "_train_step", 0), 0))
        if t_step <= s_step:
            return float(w_start)
        if t_step >= e_step:
            return float(w_final)

        ratio = float((t_step - s_step) / max(float(e_step - s_step), 1e-9))
        w_eff = float(w_start + (w_final - w_start) * ratio)
        return float(max(w_eff, 0.0))

    def _compute_reward_proximity_step(self, dist2: np.ndarray) -> float:
        """
         proximity shaping delta/absolute 
        """
        prox_w = float(self._current_reward_proximity_weight())
        if prox_w <= 0.0:
            return 0.0

        prox_mode = str(getattr(self.cfg, "reward_proximity_mode", "delta")).strip().lower()
        dmin_per_uav = np.sqrt(np.maximum(np.min(dist2, axis=0), 0.0))  # (M,)
        mean_d_norm = float(np.mean(dmin_per_uav)) / max(float(self.cfg.L), 1e-9)

        if prox_mode == "delta":
            if self._prev_prox_d_norm is not None:
                
                prox_step = float(prox_w * (self._prev_prox_d_norm - mean_d_norm))
            else:
                prox_step = 0.0
            self._prev_prox_d_norm = mean_d_norm
            return float(prox_step)

        
        return float(-prox_w * mean_d_norm)

    def _compute_coverage_floor_penalty(self, coverage_fraction: float) -> float:
        """
         coverage floor 
        """
        floor_w = float(getattr(self.cfg, "coverage_floor_weight", 0.0))
        if floor_w <= 0.0:
            return 0.0
        floor_thr = float(getattr(self.cfg, "coverage_floor_threshold", 0.8))
        cov = float(np.clip(float(coverage_fraction), 0.0, 1.0))
        if cov >= floor_thr:
            return 0.0
        return float(-floor_w * (floor_thr - cov))

    def _train_progress_frac(self) -> float:
        """
        [0,1]LayerB
        """
        total_steps = int(max(getattr(self.cfg, "train_total_steps", 0), 0))
        if total_steps <= 0:
            total_steps = int(max(int(getattr(self.cfg, "T", 80)) * 200, 1))
        return float(np.clip(float(self._train_step) / float(max(total_steps, 1)), 0.0, 1.0))

    def _z4_gap_lambda_schedule(self, lambda_input: float) -> Tuple[float, float]:
        """
        LayerBlambda_gap warmup + late anneal
         (lambda_sched, train_frac)
        """
        lam_in = float(max(lambda_input, 0.0))
        frac = float(self._train_progress_frac())
        if lam_in <= 0.0:
            return 0.0, frac

        warm_s = float(np.clip(float(getattr(self.cfg, "z4_gap_warmup_start_frac", 0.02)), 0.0, 1.0))
        warm_e = float(np.clip(float(getattr(self.cfg, "z4_gap_warmup_end_frac", 0.10)), 0.0, 1.0))
        if warm_e <= warm_s + 1e-12:
            warm_scale = 1.0 if frac >= warm_e else 0.0
        elif frac <= warm_s:
            warm_scale = 0.0
        elif frac >= warm_e:
            warm_scale = 1.0
        else:
            warm_scale = float((frac - warm_s) / max(warm_e - warm_s, 1e-9))

        anneal_s = float(np.clip(float(getattr(self.cfg, "z4_gap_anneal_start_frac", 0.70)), 0.0, 1.0))
        anneal_e = float(np.clip(float(getattr(self.cfg, "z4_gap_anneal_end_frac", 0.95)), 0.0, 1.0))
        anneal_floor = float(np.clip(float(getattr(self.cfg, "z4_gap_anneal_floor", 0.35)), 0.0, 1.0))
        anneal_scale = 1.0
        if anneal_e > anneal_s + 1e-12:
            if frac >= anneal_e:
                anneal_scale = anneal_floor
            elif frac > anneal_s:
                x = float((frac - anneal_s) / max(anneal_e - anneal_s, 1e-9))
                anneal_scale = float(1.0 + (anneal_floor - 1.0) * x)

        lam_sched = float(lam_in * max(warm_scale, 0.0) * max(anneal_scale, 0.0))
        hard_cap = float(max(float(getattr(self.cfg, "z4_gap_lambda_hard_cap", 2.0)), 0.0))
        if hard_cap > 0.0:
            lam_sched = float(min(lam_sched, hard_cap))
        return float(max(lam_sched, 0.0)), frac

    def _z4_comp_meta_warmup_scale(self) -> Tuple[float, float]:
        """
        LayerCwarmup/
         (scale, train_frac)
        """
        frac = float(self._train_progress_frac())
        s = float(np.clip(float(getattr(self.cfg, "z4_comp_meta_warmup_start_frac", 0.00)), 0.0, 1.0))
        e = float(np.clip(float(getattr(self.cfg, "z4_comp_meta_warmup_end_frac", 0.08)), 0.0, 1.0))
        if e <= s + 1e-12:
            scale = 1.0 if frac >= e else 0.0
        elif frac <= s:
            scale = 0.0
        elif frac >= e:
            scale = 1.0
        else:
            scale = float((frac - s) / max(e - s, 1e-9))
        return float(np.clip(scale, 0.0, 1.0)), frac

    def _z4_assoc_stage_policy_mix(self) -> Tuple[float, float, str]:
        """
        LayerC-Plus

        
        - mix comp_score 0=1=
        - train_frac
        - mode_tagheuristic / blend / policy
        """
        frac = float(self._train_progress_frac())
        if not bool(getattr(self.cfg, "z4_assoc_stage_enable", False)):
            return 1.0, frac, "policy"

        start = float(np.clip(float(getattr(self.cfg, "z4_assoc_stage_start_frac", 0.25)), 0.0, 1.0))
        end = float(np.clip(float(getattr(self.cfg, "z4_assoc_stage_end_frac", 0.65)), 0.0, 1.0))
        if end < start:
            end = start
        pmin = float(np.clip(float(getattr(self.cfg, "z4_assoc_stage_policy_min", 0.0)), 0.0, 1.0))
        pmax = float(np.clip(float(getattr(self.cfg, "z4_assoc_stage_policy_max", 1.0)), pmin, 1.0))

        if frac <= start + 1e-12:
            mix = float(pmin)
        elif frac >= end - 1e-12:
            mix = float(pmax)
        else:
            x = float((frac - start) / max(end - start, 1e-9))
            if bool(getattr(self.cfg, "z4_assoc_stage_smoothstep", True)):
                x = float(np.clip(x, 0.0, 1.0))
                x = float(x * x * (3.0 - 2.0 * x))
            mix = float(pmin + (pmax - pmin) * x)

        mix = float(np.clip(mix, 0.0, 1.0))
        if mix <= 1e-6:
            mode_tag = "heuristic"
        elif mix >= 1.0 - 1e-6:
            mode_tag = "policy"
        else:
            mode_tag = "blend"
        return mix, frac, mode_tag

    def _phaseg_beta(self, beta_max: float) -> Tuple[float, float]:
        """
        PhaseGshaping (t) warmupholdanneal
        """
        beta_max = float(max(beta_max, 0.0))
        total_steps = int(max(getattr(self.cfg, "train_total_steps", 0), 0))
        if total_steps <= 0:
            
            total_steps = int(max(int(getattr(self.cfg, "T", 80)) * 200, 1))

        t = int(max(getattr(self, "_train_step", 0), 0))
        frac = float(np.clip(float(t) / float(max(total_steps, 1)), 0.0, 1.0))

        warm = float(np.clip(float(getattr(self.cfg, "beta_warmup_frac", 0.10)), 1e-6, 1.0))
        hold = float(np.clip(float(getattr(self.cfg, "beta_hold_frac", 0.50)), 0.0, 1.0))
        hold_end = float(np.clip(warm + hold, warm, 0.999))
        end_frac = float(np.clip(float(getattr(self.cfg, "beta_end_frac", 0.0)), 0.0, 1.0))
        beta_end = float(beta_max * end_frac)

        if beta_max <= 0.0:
            return 0.0, frac

        if frac <= warm:
            beta = beta_max * (frac / max(warm, 1e-6))
        elif frac <= hold_end:
            beta = beta_max
        else:
            denom = max(1.0 - hold_end, 1e-6)
            x = float(np.clip((frac - hold_end) / denom, 0.0, 1.0))
            beta = float(beta_max + (beta_end - beta_max) * x)

        beta = float(np.clip(beta, 0.0, beta_max))
        return beta, frac

    def _t6_anneal_ratio(self) -> float:
        """
        PhaseT6 r[0,1]
        - r=0: 
        - r=1: 
        """
        total_steps = int(max(getattr(self.cfg, "train_total_steps", 0), 0))
        if total_steps <= 0:
            total_steps = int(max(int(getattr(self.cfg, "T", 80)) * 200, 1))
        frac = float(np.clip(float(self._train_step) / float(max(total_steps, 1)), 0.0, 1.0))
        s = float(np.clip(float(getattr(self.cfg, "t6_anneal_start_frac", 0.10)), 0.0, 1.0))
        e = float(np.clip(float(getattr(self.cfg, "t6_anneal_end_frac", 0.80)), 0.0, 1.0))
        if e <= s + 1e-9:
            return 1.0 if frac >= e else 0.0
        return float(np.clip((frac - s) / (e - s), 0.0, 1.0))

    def _soft_clip_reward(self, reward_raw: float) -> float:
        """
        
        """
        reward_clip_range = float(getattr(self.cfg, "reward_clip_range", 10.0))
        if reward_clip_range > 0.0:
            return float(reward_clip_range * np.tanh(float(reward_raw) / float(reward_clip_range)))
        return float(reward_raw)

    def _apply_reward_ema(self, reward_value: float) -> float:
        """
        EMAreward_ema_beta=0 
        """
        reward_ema_beta = float(getattr(self.cfg, "reward_ema_beta", 0.0))
        reward_ema_beta = float(np.clip(reward_ema_beta, 0.0, 0.999))
        if not hasattr(self, "_reward_ema"):
            self._reward_ema = float(reward_value)
        if reward_ema_beta > 0.0:
            self._reward_ema = reward_ema_beta * float(self._reward_ema) + (1.0 - reward_ema_beta) * float(reward_value)
        else:
            self._reward_ema = float(reward_value)
        return float(self._reward_ema)

    def _postprocess_reward(self, reward_raw: float) -> Tuple[float, float, float]:
        """
         ->  -> EMA

        
        - reward_out: EMA
        - reward_scaled: 
        - reward_clipped: 
        """
        reward_scale_factor = float(getattr(self.cfg, "reward_scale_factor", 1.0))
        reward_scaled = float(reward_raw * reward_scale_factor) if reward_scale_factor != 1.0 else float(reward_raw)
        reward_clipped = self._soft_clip_reward(reward_scaled)
        reward_out = self._apply_reward_ema(reward_clipped)
        return float(reward_out), float(reward_scaled), float(reward_clipped)

    def _compute_traj_centroid_bonus(
        self,
        a_mat: np.ndarray,
        *,
        guide_weight: float,
        assoc_active_threshold: float = 0.10,
    ) -> float:
        """
        UAVdelta_q
        """
        weight = float(max(float(guide_weight), 0.0))
        if weight <= 0.0 or self.q_prev is None:
            return 0.0

        M_int, I_int = int(self.cfg.M), int(self.cfg.I)
        q_now = np.asarray(self.q, dtype=np.float64).reshape(M_int, 2)
        q_old = np.asarray(self.q_prev, dtype=np.float64).reshape(M_int, 2)
        w_now = np.asarray(self.w, dtype=np.float64).reshape(I_int, 2)
        a_np = np.asarray(a_mat, dtype=np.float64)
        serve_th = float(max(float(assoc_active_threshold), 0.0))
        vmax_dt = float(max(float(self.cfg.Vmax) * float(self.cfg.dt), 1e-9))

        bonus_sum = 0.0
        for m in range(M_int):
            served_users = np.where(a_np[:, m] > serve_th)[0]
            if served_users.size == 0:
                continue
            centroid = np.mean(w_now[served_users], axis=0)
            d_old = float(np.linalg.norm(q_old[m] - centroid))
            d_new = float(np.linalg.norm(q_now[m] - centroid))
            delta_d = float((d_old - d_new) / vmax_dt)
            bonus_sum += float(np.clip(delta_d, -1.0, 1.0))
        return float(weight * bonus_sum / max(float(M_int), 1.0))

    # --------- core dynamics ---------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.q is None or self.w is None:
            self.reset()

        self.t += 1
        self._train_step += 1
        
        if bool(getattr(self.cfg, "t6_direct_blockage_refresh_each_step", False)):
            self._refresh_t6_direct_blockage_map()
        reward_design = str(getattr(self.cfg, "reward_design", "layered_v1")).strip().lower()
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        expected_dim = int(self.action_dim)
        strict_dim_check = bool(getattr(self.cfg, "strict_action_dim_check", True))
        if action_arr.size != expected_dim:
            
            z4_action_mode = str(getattr(self.cfg, "z4_action_mode", "hierarchical")).strip().lower()
            action_space_mode_now = str(getattr(self.cfg, "action_space_mode", "full")).strip().lower()
            compat_layerc = bool(
                action_space_mode_now == "hierarchical"
                and z4_action_mode == "position_only"
                and bool(getattr(self.cfg, "z4_comp_meta_enable", False))
                and int(expected_dim) == int(2 * int(self.cfg.M) + 2)
                and int(action_arr.size) == int(2 * int(self.cfg.M))
            )
            if compat_layerc:
                action_arr = np.concatenate([action_arr, np.zeros((2,), dtype=np.float64)], axis=0)
                self._warn_once("layerc_action_dim_pad", "10D2D")
            msg = f"action: got={int(action_arr.size)}, expected={expected_dim}"
            if strict_dim_check and (not compat_layerc):
                raise ValueError(msg)
            if not compat_layerc:
                self._warn_once("action_dim_mismatch", msg + "/")
        
        delta_q, s_a, s_z, theta_raw, comp_meta = self._parse_action(action_arr)

        
        if bool(getattr(self.cfg, "obs_enable_prev_action", False)):
            M_int, I_int = int(self.cfg.M), int(self.cfg.I)
            z4_action_mode = str(getattr(self.cfg, "z4_action_mode", "hierarchical")).strip().lower()
            if z4_action_mode == "position_only":
                
                if bool(getattr(self.cfg, "z4_comp_meta_enable", False)):
                    self._prev_action_flat = np.concatenate(
                        [
                            delta_q.reshape(-1)[:M_int * 2],
                            np.asarray(comp_meta, dtype=np.float64).reshape(-1)[:2],
                        ]
                    ).astype(np.float64)
                else:
                    self._prev_action_flat = delta_q.reshape(-1)[:M_int * 2].astype(np.float64)
            else:
                self._prev_action_flat = np.concatenate([
                    delta_q.reshape(-1)[:M_int * 2],   
                    s_a.reshape(-1)[:I_int],            # (I,) comp_score [-1,1]
                ]).astype(np.float64)
        # ===== [END Phase Z] =====

        
        
        
        delta_scale = float(self.cfg.Vmax) * float(self.cfg.dt)  # inf2.0 * 1.0 = 12.0 inf inf
        delta_q_meters = delta_q * delta_scale
        # ===== [END A 1] =====

        # 1) move
        self.q_prev = self.q.copy()
        q_new = self.q + delta_q_meters
        q_new = project_speed(q_new, self.q_prev, self.cfg.Vmax, self.cfg.dt)
        safe_clip_projection_step = 0.0

        
        
        vio_wall = 0
        vio_wall_mask = np.zeros((int(self.cfg.M),), dtype=bool)
        L = float(self.cfg.L)
        reflect_on_wall = bool(getattr(self.cfg, "phaseh_reflect_on_wall", False))
        for m in range(int(self.cfg.M)):
            x, y = q_new[m, 0], q_new[m, 1]
            
            if x < 0.0 or x > L or y < 0.0 or y > L:
                if reflect_on_wall:
                    
                    xr = float(x)
                    yr = float(y)
                    if xr < 0.0:
                        xr = -xr
                    elif xr > L:
                        xr = 2.0 * L - xr
                    if yr < 0.0:
                        yr = -yr
                    elif yr > L:
                        yr = 2.0 * L - yr
                    q_new[m, 0] = float(xr)
                    q_new[m, 1] = float(yr)
                else:
                    # Fallback: revert to previous feasible position when reflection is disabled.
                    q_new[m] = self.q_prev[m].copy()
                vio_wall += 1
                vio_wall_mask[m] = True

        # A UAV
        # UAV [Vmax*dt, L-Vmax*dt]^2
        
        wall_safe = self._wall_safe_margin()
        if vio_wall > 0 and wall_safe > 1e-9:
            lo = float(wall_safe)
            hi = float(L - wall_safe)
            if hi > lo + 1e-6:
                for m in range(int(self.cfg.M)):
                    if not bool(vio_wall_mask[m]):
                        continue
                    q_new[m, 0] = float(np.clip(q_new[m, 0], lo, hi))
                    q_new[m, 1] = float(np.clip(q_new[m, 1], lo, hi))
        # ===== [END A 2] =====

        
        q_pre_repair = q_new.copy()

        vio_coll_pre = self._count_collisions(q_new)
        q_new, _ = repair_collisions(q_new, self.cfg.dmin, rng=self.rng)
        
        vio_coll_post = self._count_collisions(q_new)
        # io_wall
        q_new = _clip_pos(q_new, self.cfg.L)

        
        
        
        wall_safe = self._wall_safe_margin()
        if wall_safe > 1e-9:
            lo = float(wall_safe)
            hi = float(float(self.cfg.L) - wall_safe)
            if hi > lo + 1e-6:
                q_before_safe_clip = q_new.copy()
                q_new[:, 0] = np.clip(q_new[:, 0], lo, hi)
                q_new[:, 1] = np.clip(q_new[:, 1], lo, hi)
                proj = np.linalg.norm(q_before_safe_clip - q_new, axis=1)
                vmax_dt = float(max(float(self.cfg.Vmax) * float(self.cfg.dt), 1e-9))
                safe_clip_projection_step = float(np.mean(proj / vmax_dt))
        self.q = q_new
        self._update_boundary_stick_ema(self.q)

            # Mobile-user scenario: update user positions before link/coverage calculation.
        self._advance_user_positions()

        
        D, C = self._ensure_task_cache(
            expected_slot=int(self.t),
            context="step_pre",
            force_resample=False,
        )

        # 3) gains
        g = self._compute_gains()

        # 4) coverage
        Rc2_eff = self.cfg.Rc * self.cfg.Rc - self.cfg.H * self.cfg.H
        dist2 = pairwise_dist2(self.q, self.w).T  # (I,M)
        covered = dist2 <= max(Rc2_eff, 0.0) + 1e-12

        # 5) association a (Top-K)
        assoc_mode = str(getattr(self.cfg, "association_mode", "hard")).strip().lower()
        if assoc_mode not in ("hard", "soft"):
            assoc_mode = "hard"
        action_space_mode = str(getattr(self.cfg, "action_space_mode", "full")).strip().lower()
        if action_space_mode not in ("full", "hierarchical"):
            action_space_mode = "full"

        
        a_mat = np.zeros((self.cfg.I, self.cfg.M), dtype=(np.int64 if assoc_mode == "hard" else np.float64))
        vio_no_cover = 0
        comp_gate_ratio = float(getattr(self.cfg, "comp_gate_ratio", 0.0))
        enable_comp = bool(getattr(self.cfg, "enable_comp", True))
        K_eff = int(self.cfg.K) if enable_comp else 1
        comp_gate_snr_candidates = 0
        comp_gate_snr_pruned = 0
        primary_uav = None
        comp_temp_override: Optional[float] = None
        
        z4_comp_meta_enable_step = 0.0
        z4_comp_meta_ctrl_thr_raw = 0.0
        z4_comp_meta_ctrl_temp_raw = 0.0
        z4_comp_meta_warm_scale = 0.0
        z4_comp_meta_train_frac = 0.0
        z4_comp_meta_thr_base = float(np.clip(float(getattr(self.cfg, "comp_rule_threshold", 0.5)), 0.0, 1.0))
        z4_comp_meta_thr_delta_raw = 0.0
        z4_comp_meta_thr_delta_ema = float(getattr(self, "_z4_comp_meta_thr_delta_ema", 0.0))
        z4_comp_meta_thr_delta_eff = 0.0
        z4_comp_meta_thr_effective = float(z4_comp_meta_thr_base)
        z4_comp_meta_score_width = float(max(float(getattr(self.cfg, "z4_comp_meta_score_width", 0.20)), 1e-3))
        z4_comp_meta_temp_base = float(max(float(self._resolve_comp_score_temp()), 1e-6))
        z4_comp_meta_temp_scale_raw = 1.0
        z4_comp_meta_temp_scale_ema = float(getattr(self, "_z4_comp_meta_temp_scale_ema", 1.0))
        z4_comp_meta_temp_scale_eff = 1.0
        z4_comp_meta_temp_effective = float(z4_comp_meta_temp_base)
        
        z4_assoc_stage_enable_step = 0.0
        z4_assoc_policy_mix_step = 1.0
        z4_assoc_train_frac_step = 0.0
        z4_assoc_mode_effective = "policy"
        z4_assoc_stage_score_width = float(max(float(getattr(self.cfg, "z4_comp_meta_score_width", 0.20)), 1e-3))
        z4_assoc_stage_comp_rule_thr = float(np.clip(float(getattr(self.cfg, "comp_rule_threshold", 0.5)), 0.0, 1.0))

        if action_space_mode == "hierarchical":
            z4_action_mode = str(getattr(self.cfg, "z4_action_mode", "hierarchical")).strip().lower()
            if z4_action_mode == "position_only":
                
                
                comp_rule_thr_base = float(np.clip(float(getattr(self.cfg, "comp_rule_threshold", 0.5)), 0.0, 1.0))
                comp_rule_thr = float(comp_rule_thr_base)
                comp_temp_base = float(max(float(self._resolve_comp_score_temp()), 1e-6))
                comp_temp_effective = float(comp_temp_base)
                score_width = float(max(float(getattr(self.cfg, "z4_comp_meta_score_width", 0.20)), 1e-3))
                
                if bool(getattr(self.cfg, "z4_comp_meta_enable", False)):
                    ctrl = np.asarray(comp_meta, dtype=np.float64).reshape(-1)
                    if ctrl.size >= 2:
                        ctrl_thr = float(np.clip(ctrl[0], -1.0, 1.0))
                        ctrl_temp = float(np.clip(ctrl[1], -1.0, 1.0))
                        warm_scale, train_frac = self._z4_comp_meta_warmup_scale()
                        thr_delta_max = float(max(float(getattr(self.cfg, "z4_comp_meta_thr_delta_max", 0.10)), 0.0))
                        thr_delta_raw = float(ctrl_thr * thr_delta_max)
                        temp_scale_delta = float(max(float(getattr(self.cfg, "z4_comp_meta_temp_scale_delta", 0.35)), 0.0))
                        temp_scale_raw = float(1.0 + ctrl_temp * temp_scale_delta)
                        temp_scale_min = float(max(float(getattr(self.cfg, "z4_comp_meta_temp_scale_min", 0.65)), 1e-3))
                        temp_scale_max = float(max(float(getattr(self.cfg, "z4_comp_meta_temp_scale_max", 1.35)), temp_scale_min))
                        temp_scale_raw = float(np.clip(temp_scale_raw, temp_scale_min, temp_scale_max))

                        beta = float(np.clip(float(getattr(self.cfg, "z4_comp_meta_ema_beta", 0.80)), 0.0, 0.995))
                        if not bool(getattr(self, "_z4_comp_meta_ema_ready", False)):
                            self._z4_comp_meta_thr_delta_ema = float(thr_delta_raw)
                            self._z4_comp_meta_temp_scale_ema = float(temp_scale_raw)
                            self._z4_comp_meta_ema_ready = True
                        else:
                            self._z4_comp_meta_thr_delta_ema = float(
                                beta * float(self._z4_comp_meta_thr_delta_ema)
                                + (1.0 - beta) * float(thr_delta_raw)
                            )
                            self._z4_comp_meta_temp_scale_ema = float(
                                beta * float(self._z4_comp_meta_temp_scale_ema)
                                + (1.0 - beta) * float(temp_scale_raw)
                            )

                        thr_delta_eff = float(warm_scale * float(self._z4_comp_meta_thr_delta_ema))
                        temp_scale_eff = float(1.0 + warm_scale * (float(self._z4_comp_meta_temp_scale_ema) - 1.0))
                        temp_scale_eff = float(np.clip(temp_scale_eff, temp_scale_min, temp_scale_max))
                        thr_min = float(np.clip(float(getattr(self.cfg, "z4_comp_meta_thr_min", 0.05)), 0.0, 1.0))
                        thr_max = float(np.clip(float(getattr(self.cfg, "z4_comp_meta_thr_max", 0.70)), thr_min, 1.0))
                        comp_rule_thr = float(np.clip(comp_rule_thr_base + thr_delta_eff, thr_min, thr_max))
                        comp_temp_effective = float(max(comp_temp_base * temp_scale_eff, 1e-6))
                        comp_temp_override = float(comp_temp_effective)

                        z4_comp_meta_enable_step = 1.0
                        z4_comp_meta_ctrl_thr_raw = float(ctrl_thr)
                        z4_comp_meta_ctrl_temp_raw = float(ctrl_temp)
                        z4_comp_meta_warm_scale = float(warm_scale)
                        z4_comp_meta_train_frac = float(train_frac)
                        z4_comp_meta_thr_base = float(comp_rule_thr_base)
                        z4_comp_meta_thr_delta_raw = float(thr_delta_raw)
                        z4_comp_meta_thr_delta_ema = float(self._z4_comp_meta_thr_delta_ema)
                        z4_comp_meta_thr_delta_eff = float(thr_delta_eff)
                        z4_comp_meta_thr_effective = float(comp_rule_thr)
                        z4_comp_meta_score_width = float(score_width)
                        z4_comp_meta_temp_base = float(comp_temp_base)
                        z4_comp_meta_temp_scale_raw = float(temp_scale_raw)
                        z4_comp_meta_temp_scale_ema = float(self._z4_comp_meta_temp_scale_ema)
                        z4_comp_meta_temp_scale_eff = float(temp_scale_eff)
                        z4_comp_meta_temp_effective = float(comp_temp_effective)

                I_int = int(self.cfg.I)
                
                synthetic_comp_score = np.full(I_int, -1.0, dtype=np.float64)
                for i in range(I_int):
                    g_row = g[i]  
                    sorted_idx = np.argsort(g_row)[::-1]  
                    g_primary = float(g_row[sorted_idx[0]])
                    if len(sorted_idx) > 1 and g_primary > 1e-12:
                        g_second = float(g_row[sorted_idx[1]])
                        ratio_gap = float(g_second / g_primary - comp_rule_thr)
                        synthetic_comp_score[i] = float(np.clip(ratio_gap / max(score_width, 1e-9), -1.0, 1.0))
                comp_score = synthetic_comp_score
            else:
                comp_score_policy = np.asarray(s_a, dtype=np.float64).reshape(int(self.cfg.I),)
                if bool(getattr(self.cfg, "z4_assoc_stage_enable", False)):
                    
                    
                    comp_rule_thr = float(np.clip(float(getattr(self.cfg, "comp_rule_threshold", 0.5)), 0.0, 1.0))
                    score_width = float(max(float(getattr(self.cfg, "z4_comp_meta_score_width", 0.20)), 1e-3))
                    comp_score_heur = np.full(int(self.cfg.I), -1.0, dtype=np.float64)
                    for i in range(int(self.cfg.I)):
                        g_row = g[i]
                        sorted_idx = np.argsort(g_row)[::-1]
                        g_primary = float(g_row[sorted_idx[0]])
                        if len(sorted_idx) > 1 and g_primary > 1e-12:
                            g_second = float(g_row[sorted_idx[1]])
                            ratio_gap = float(g_second / g_primary - comp_rule_thr)
                            comp_score_heur[i] = float(np.clip(ratio_gap / max(score_width, 1e-9), -1.0, 1.0))
                    mix_policy, frac_stage, mode_stage = self._z4_assoc_stage_policy_mix()
                    comp_score = float(mix_policy) * comp_score_policy + (1.0 - float(mix_policy)) * comp_score_heur
                    score_clip = float(max(float(getattr(self.cfg, "z4_assoc_stage_score_clip", 1.0)), 0.0))
                    if score_clip > 0.0:
                        comp_score = np.clip(comp_score, -score_clip, score_clip)
                    z4_assoc_stage_enable_step = 1.0
                    z4_assoc_policy_mix_step = float(mix_policy)
                    z4_assoc_train_frac_step = float(frac_stage)
                    z4_assoc_mode_effective = str(mode_stage)
                    z4_assoc_stage_score_width = float(score_width)
                    z4_assoc_stage_comp_rule_thr = float(comp_rule_thr)
                else:
                    comp_score = comp_score_policy
            a_mat, primary_uav, vio_no_cover, comp_gate_snr_candidates, comp_gate_snr_pruned = (
                self._build_assoc_from_comp_score(
                    comp_score=comp_score,
                    g=g,
                    covered=covered,
                    dist2=dist2,
                    assoc_mode=assoc_mode,
                    k_eff=K_eff,
                    comp_gate_ratio=comp_gate_ratio,
                    task_weights=C,
                    comp_temp_override=comp_temp_override,
                )
            )
        else:
            for i in range(self.cfg.I):
                if assoc_mode == "hard":
                    chosen, had_cover = ensure_topk_feasible(
                        scores=s_a[i],
                        covered_mask=covered[i],
                        K=K_eff,
                        dist2_row=dist2[i],
                        g_row=g[i],
                        gain_mix=float(getattr(self.cfg, "assoc_gain_mix", 0.0)),
                    )
                    if not had_cover:
                        vio_no_cover += 1

                    # --- CoMP gating (keep only 1 UAV if 2nd link too weak) ---
                    if enable_comp and (comp_gate_ratio > 0.0) and len(chosen) >= 2:
                        gains = np.asarray(g[i, chosen], dtype=np.float64).reshape(-1)
                        order = np.argsort(gains)[::-1]
                        best_gain = float(gains[order[0]])
                        second_gain = float(gains[order[1]])
                        best_uav = int(chosen[order[0]])
                        if best_gain <= 0.0 or (second_gain < comp_gate_ratio * best_gain):
                            chosen = np.array([best_uav], dtype=np.int64)
                    if enable_comp and len(chosen) >= 2:
                        comp_gate_snr_candidates += 1
                        chosen, pruned = self._gate_comp_set_by_snr(g[i], chosen)
                        if pruned:
                            comp_gate_snr_pruned += 1

                    a_mat[i, chosen] = 1
                else:
                    
                    # - _eff AV /
                    
                    cand = np.where(covered[i])[0]
                    if cand.size == 0:
                        nearest = int(np.argmin(dist2[i]))
                        a_mat[i, nearest] = 1.0
                        vio_no_cover += 1
                        continue

                    s = np.asarray(s_a[i, cand], dtype=np.float64).reshape(-1)
                    gain_mix = float(getattr(self.cfg, "assoc_gain_mix", 0.0))
                    if gain_mix != 0.0:
                        s = s + gain_mix * np.log1p(np.maximum(np.asarray(g[i, cand], dtype=np.float64), 0.0))

                    order = np.argsort(s)[::-1]
                    k_take = int(min(max(1, K_eff), int(cand.size)))
                    chosen = cand[order[:k_take]].astype(np.int64)
                    s_chosen = s[order[:k_take]]

                    # softmax
                    temp = float(getattr(self.cfg, "assoc_soft_temp", 0.50))
                    temp = max(temp, 1e-6)
                    x = (s_chosen / temp) - float(np.max(s_chosen / temp))
                    w = np.exp(x)
                    w = w / (float(np.sum(w)) + 1e-12)

                    # CoMP gating AV hard
                    if enable_comp and (comp_gate_ratio > 0.0) and int(chosen.size) >= 2:
                        gains = np.asarray(g[i, chosen], dtype=np.float64).reshape(-1)
                        og = np.argsort(gains)[::-1]
                        best_gain = float(gains[og[0]])
                        second_gain = float(gains[og[1]])
                        best_uav = int(chosen[og[0]])
                        if best_gain <= 0.0 or (second_gain < comp_gate_ratio * best_gain):
                            chosen = np.array([best_uav], dtype=np.int64)
                            w = np.array([1.0], dtype=np.float64)
                    if enable_comp and int(chosen.size) >= 2:
                        comp_gate_snr_candidates += 1
                        chosen, pruned = self._gate_comp_set_by_snr(g[i], chosen)
                        if pruned:
                            comp_gate_snr_pruned += 1
                            w = np.array([1.0], dtype=np.float64)

                    a_mat[i, chosen] = w.astype(np.float64)

        
        if action_space_mode == "hierarchical" and primary_uav is not None:
            if str(reward_design).strip().lower() == "t6_potential_lagrangian":
                z_mat = self._select_compute_uav_hierarchical(
                    primary_uav=primary_uav,
                    a_mat=a_mat,
                    dist2=dist2,
                    g=g,
                    C=C,
                )
            else:
                z_mat = self._select_compute_uav_from_primary(primary_uav=primary_uav, dist2=dist2)
        else:
            z_mat = np.zeros((self.cfg.I, self.cfg.M), dtype=np.int64)
            fmax_arr = np.asarray(self.cfg.fmax, dtype=np.float64).reshape(int(self.cfg.M),)
            fmax_ref = float(max(np.max(fmax_arr), 1e-9))
            cpu_slot = float(max(float(getattr(self.cfg, "cpu_slot_time", 1.0)), 1e-9))
            z_load_balance_alpha = float(max(float(getattr(self.cfg, "z_load_balance_alpha", 0.0)), 0.0))
            z_capacity_bias_alpha = float(getattr(self.cfg, "z_capacity_bias_alpha", 0.0))
            
            z_dyn_load = np.zeros((self.cfg.M,), dtype=np.float64)
            for i in range(self.cfg.I):
                
                sel = np.where(np.asarray(a_mat[i], dtype=np.float64) > 0.0)[0]
                if sel.size == 0:
                    sel = np.array([int(np.argmin(dist2[i]))], dtype=np.int64)
                score = np.asarray(s_z[i, sel], dtype=np.float64).reshape(-1).copy()
                if abs(z_capacity_bias_alpha) > 1e-12:
                    cap_norm = np.clip(fmax_arr[sel] / fmax_ref, 1e-9, None)
                    score = score + float(z_capacity_bias_alpha) * np.log(cap_norm)
                if z_load_balance_alpha > 1e-12 and sel.size >= 2:
                    score = score - float(z_load_balance_alpha) * np.asarray(z_dyn_load[sel], dtype=np.float64)
                m_star = int(sel[int(np.argmax(score))])
                z_mat[i, m_star] = 1
                
                denom_m = max(float(fmax_arr[m_star]) * cpu_slot, 1e-9)
                z_dyn_load[m_star] = float(z_dyn_load[m_star] + float(C[i]) / denom_m)

        
        if assoc_mode == "hard":
            a_mat, z_mat = enforce_min_coverage(a_mat, z_mat, covered, dist2)
        else:
            
            
            for i in range(int(self.cfg.I)):
                if float(np.sum(a_mat[i])) <= 0.0:
                    cand = np.where(covered[i])[0]
                    if cand.size == 0:
                        m = int(np.argmin(dist2[i]))
                        a_mat[i, m] = 1.0
                    else:
                        a_mat[i, int(cand[0])] = 1.0
                if int(np.sum(z_mat[i])) != 1:
                    z_mat[i, :] = 0
                    sel = np.where(np.asarray(a_mat[i], dtype=np.float64) > 0.0)[0]
                    m = int(sel[0]) if sel.size > 0 else int(np.argmin(dist2[i]))
                    z_mat[i, m] = 1
                m_star = int(np.argmax(z_mat[i]))
                if float(np.asarray(a_mat[i], dtype=np.float64)[m_star]) <= 0.0:
                    a_mat[i, m_star] = 1.0
                    ssum = float(np.sum(np.asarray(a_mat[i], dtype=np.float64)))
                    if ssum > 0.0:
                        a_mat[i, :] = np.asarray(a_mat[i], dtype=np.float64) / ssum

        
        a_binary_now = (np.asarray(a_mat, dtype=np.float64) > 0.0).astype(np.int32)
        self.a_binary = a_binary_now
        self.comp_selection = np.sum(a_binary_now, axis=0).astype(np.int32)
        comp_enable_rate = 0.0
        z4_action_mode_now = str(getattr(self.cfg, "z4_action_mode", "hierarchical")).strip().lower()
        if action_space_mode == "hierarchical" and z4_action_mode_now != "position_only":
            
            score_vec = np.asarray(s_a, dtype=np.float64).reshape(int(self.cfg.I),)
            comp_enable_rate = float(np.mean(score_vec > 0.0)) if score_vec.size > 0 else 0.0
        else:
            
            s_size = np.sum(a_binary_now, axis=1).astype(np.float64)
            comp_enable_rate = float(np.mean(s_size >= 2.0)) if s_size.size > 0 else 0.0
        service_switch_count = 0.0
        prev_a_binary_diag = getattr(self, "_prev_a_binary_for_switch_diag", None)
        if prev_a_binary_diag is not None:
            try:
                prev_arr = np.asarray(prev_a_binary_diag, dtype=np.int32)
                if prev_arr.shape == a_binary_now.shape:
                    service_switch_count = float(np.sum(np.any(prev_arr != a_binary_now, axis=1)))
            except Exception:
                service_switch_count = 0.0

        # 7) theta
        theta = np.zeros((self.cfg.I, self.cfg.M), dtype=np.float64)
        vio_theta = 0
        theta_infeasible = 0

        theta_mode_cfg = str(getattr(self.cfg, "theta_mode", "solver")).strip().lower()
        if theta_mode_cfg not in ("policy", "solver", "agent"):
            raise ValueError(f"theta_mode must be 'solver'/'agent'/'policy', got: {theta_mode_cfg}")
        theta_mode_effective = str(theta_mode_cfg)
        if theta_mode_effective == "policy" and action_space_mode == "hierarchical":
            
            theta_mode_effective = "agent"
            self._warn_once(
                "theta_mode_policy_hierarchical_alias",
                " action_space_mode=hierarchical  theta_mode=policy agent ",
            )
        tau_theta_effective = float(max(float(getattr(self.cfg, "theta_softmax_tau", 0.3)), 1e-6))

        for m in range(self.cfg.M):
            users_m = np.where(z_mat[:, m] == 1)[0]
            if users_m.size == 0:
                continue

            if theta_mode_effective == "agent":
                if users_m.size == 1:
                    raw = np.ones((1,), dtype=np.float64)
                    feasible = True
                else:
                    if action_space_mode == "hierarchical":
                        score_vec = np.asarray(s_a, dtype=np.float64).reshape(int(self.cfg.I),)
                        priority = np.abs(score_vec[users_m]) + 1e-6
                    else:
                        score_mat = np.asarray(s_a, dtype=np.float64)
                        if score_mat.ndim == 2 and score_mat.shape[0] == int(self.cfg.I) and score_mat.shape[1] == int(self.cfg.M):
                            priority = np.abs(score_mat[users_m, int(m)]) + 1e-6
                        else:
                            priority = np.ones((users_m.size,), dtype=np.float64)
                    logits = priority / max(tau_theta_effective, 1e-9)
                    logits = logits - float(np.max(logits))
                    exp_logits = np.exp(logits)
                    raw = exp_logits / (float(np.sum(exp_logits)) + 1e-12)
                    raw = np.maximum(raw, float(max(float(self.cfg.theta_min), 1e-6)))
                    raw = raw / (float(np.sum(raw)) + 1e-12)
                    raw, feasible = ensure_theta_feasible(raw, self.cfg.theta_min)
            elif theta_mode_effective == "policy":
                raw = theta_raw[users_m].astype(np.float64)
                raw = raw / (raw.sum() + 1e-12)
                raw, feasible = ensure_theta_feasible(raw, self.cfg.theta_min)
            else:
                raw, feasible = alloc_theta_solver(
                    C_cycles=C[users_m].astype(np.float64),
                    fmax=float(self.cfg.fmax[m]),
                    w_delay=float(self.cfg.w_delay),
                    w_energy=float(self.cfg.w_energy),
                    T_scale=float(self.cfg.T_scale),
                    E_scale=float(self.cfg.E_scale),
                    xi=float(self.cfg.xi),
                    theta_min=float(self.cfg.theta_min),
                )

            if not feasible:
                theta_infeasible += 1
            theta[users_m, m] = raw

        
        theta_entropy_vals: List[float] = []
        for m in range(self.cfg.M):
            users_m = np.where(z_mat[:, m] == 1)[0]
            if users_m.size <= 1:
                continue
            pm = np.asarray(theta[users_m, m], dtype=np.float64).reshape(-1)
            pm = pm / (float(np.sum(pm)) + 1e-12)
            ent = -np.sum(pm * np.log(pm + 1e-12))
            ent_norm = float(ent / max(np.log(float(users_m.size)), 1e-12))
            theta_entropy_vals.append(float(np.clip(ent_norm, 0.0, 1.0)))
        theta_entropy = float(np.mean(theta_entropy_vals)) if theta_entropy_vals else 0.0

        for i in range(self.cfg.I):
            m_star = int(np.argmax(z_mat[i]))
            if theta[i, m_star] < self.cfg.theta_min - 1e-12:
                vio_theta += 1

        # =========================
        # [PATCH][PLOT] record per-step frame for animation (optional)
        # =========================
        if self._trace_enabled and (self._trace is not None):
            if self._trace.get("w", None) is None and (self.w is not None):
                self._trace["w"] = self.w.copy()
            if (self._trace_max_steps is None) or (self._trace_step < self._trace_max_steps):
                fr = {
                    "t": int(self.t),
                    "q": self.q.copy() if self.q is not None else None,
                    "w": self.w.copy() if self.w is not None else None,
                    "a": a_mat.copy(),
                    "z": z_mat.copy(),
                }
                if self._trace_keep_theta:
                    fr["theta"] = theta.copy()
                self._trace["frames"].append(fr)
                self._trace_step += 1

        
        info_policy = self._compute_metrics(D, C, g, a_mat, z_mat, theta, covered=covered)
        info_policy["comp_enable_rate"] = float(comp_enable_rate)
        info_policy["service_switch_count"] = float(service_switch_count)
        info_policy["theta_entropy"] = float(theta_entropy)
        info_policy["theta_mode_effective"] = str(theta_mode_effective)
        info_policy["theta_softmax_tau"] = float(tau_theta_effective)
        info_policy["comp_score_temp_runtime"] = float(getattr(self, "_comp_score_temp_runtime", self.cfg.comp_score_temp))
        
        info_policy["z4_comp_meta_enable"] = float(z4_comp_meta_enable_step)
        info_policy["z4_comp_meta_ctrl_thr_raw"] = float(z4_comp_meta_ctrl_thr_raw)
        info_policy["z4_comp_meta_ctrl_temp_raw"] = float(z4_comp_meta_ctrl_temp_raw)
        info_policy["z4_comp_meta_warm_scale"] = float(z4_comp_meta_warm_scale)
        info_policy["z4_comp_meta_train_frac"] = float(z4_comp_meta_train_frac)
        info_policy["z4_comp_meta_thr_base"] = float(z4_comp_meta_thr_base)
        info_policy["z4_comp_meta_thr_delta_raw"] = float(z4_comp_meta_thr_delta_raw)
        info_policy["z4_comp_meta_thr_delta_ema"] = float(z4_comp_meta_thr_delta_ema)
        info_policy["z4_comp_meta_thr_delta_eff"] = float(z4_comp_meta_thr_delta_eff)
        info_policy["z4_comp_meta_thr_effective"] = float(z4_comp_meta_thr_effective)
        info_policy["z4_comp_meta_score_width"] = float(z4_comp_meta_score_width)
        info_policy["z4_comp_meta_temp_base"] = float(z4_comp_meta_temp_base)
        info_policy["z4_comp_meta_temp_scale_raw"] = float(z4_comp_meta_temp_scale_raw)
        info_policy["z4_comp_meta_temp_scale_ema"] = float(z4_comp_meta_temp_scale_ema)
        info_policy["z4_comp_meta_temp_scale_eff"] = float(z4_comp_meta_temp_scale_eff)
        info_policy["z4_comp_meta_temp_effective"] = float(z4_comp_meta_temp_effective)
        
        info_policy["z4_assoc_stage_enable"] = float(z4_assoc_stage_enable_step)
        info_policy["z4_assoc_policy_mix"] = float(z4_assoc_policy_mix_step)
        info_policy["z4_assoc_train_frac"] = float(z4_assoc_train_frac_step)
        info_policy["z4_assoc_mode_effective"] = str(z4_assoc_mode_effective)
        info_policy["z4_assoc_stage_score_width"] = float(z4_assoc_stage_score_width)
        info_policy["z4_assoc_stage_comp_rule_thr"] = float(z4_assoc_stage_comp_rule_thr)
        
        # ===== [EPIC UPGRADE] hover_energy_ratio =====
        if self.q_prev is not None:
            movement_dist = np.linalg.norm(self.q - self.q_prev, axis=1)
            avg_movement = float(np.mean(movement_dist))
            max_movement = float(self.cfg.Vmax * self.cfg.dt)
            hover_energy_ratio = float(1.0 - np.clip(avg_movement / max(max_movement, 1e-9), 0.0, 1.0))
            info_policy['hover_energy_ratio'] = hover_energy_ratio
        info_policy["comp_gate_snr_candidates"] = float(comp_gate_snr_candidates)
        info_policy["comp_gate_snr_pruned"] = float(comp_gate_snr_pruned)
        info_policy["comp_gate_snr_pruned_frac"] = float(
            float(comp_gate_snr_pruned) / float(max(comp_gate_snr_candidates, 1))
        )
        # ===== [END EPIC UPGRADE] =====

        
        def _build_base_pack(g_ref: np.ndarray, covered_ref: np.ndarray, dist2_ref: np.ndarray,
                             fair_hierarchical: bool = False) -> Dict[str, Tuple[Dict, int]]:
            pack: Dict[str, Tuple[Dict, int]] = {}
            for bname in ("balanced", "greedy_delay", "greedy_energy"):
                a_b, z_b, th_b, vio_b = self._heuristic_decisions(
                    g_ref,
                    covered_ref,
                    dist2_ref,
                    mode=bname,
                    D=D,
                    C=C,
                    use_hierarchical_assoc=fair_hierarchical,
                )
                info_b = self._compute_metrics(D, C, g_ref, a_b, z_b, th_b, covered=covered_ref)
                pack[bname] = (info_b, int(vio_b))
            return pack

        base_pack = _build_base_pack(g, covered, dist2)

        
        relative_baseline_position_mode = str(
            getattr(self.cfg, "relative_baseline_position_mode", "pre_move")
        ).strip().lower()
        if relative_baseline_position_mode not in ("pre_move", "post_move"):
            raise ValueError(
                f"relative_baseline_position_mode {relative_baseline_position_mode} "
                f"(: pre_move/post_move)"
            )

        
        
        _fair_hier = (action_space_mode == "hierarchical" and reward_design == "tcom_relative_v1")

        relative_base_pack = _build_base_pack(g, covered, dist2, fair_hierarchical=_fair_hier) if _fair_hier else base_pack
        if reward_design == "tcom_relative_v1" and relative_baseline_position_mode == "pre_move":
            q_ref = np.asarray(self.q_prev if self.q_prev is not None else self.q, dtype=np.float64).reshape(int(self.cfg.M), 2)
            w_ref = np.asarray(self.w, dtype=np.float64).reshape(int(self.cfg.I), 2)
            g_ref = self._compute_gains(uavs_xy=q_ref, users_xy=w_ref)
            dist2_ref = pairwise_dist2(q_ref, w_ref).T
            covered_ref = dist2_ref <= max(Rc2_eff, 0.0) + 1e-12
            relative_base_pack = _build_base_pack(g_ref, covered_ref, dist2_ref, fair_hierarchical=_fair_hier)

        relative_baseline_mode_cfg = str(getattr(self.cfg, "relative_baseline_mode", "balanced")).strip().lower()
        if relative_baseline_mode_cfg not in relative_base_pack:
            allowed = ",".join(sorted(relative_base_pack.keys()))
            raise ValueError(
                f"relative_baseline_mode {relative_baseline_mode_cfg} "
                f"(: {allowed})"
            )
        relative_baseline_paper_cost_step = float(relative_base_pack[relative_baseline_mode_cfg][0]["paper_cost"])

        info_base, _vio_base_extra = base_pack["balanced"]

        # ---- Block-2 diagnostics: baseline-vs-policy deltas (for controllable training) ----
        base_T = float(info_base.get("mean_T_off", np.nan))
        base_E = float(info_base.get("mean_energy", np.nan))
        base_Ttx = float(info_base.get("mean_T_tx", np.nan))
        base_rate = float(info_base.get("mean_rate", np.nan))

        pol_T = float(info_policy.get("mean_T_off", np.nan))
        pol_E = float(info_policy.get("mean_energy", np.nan))
        pol_Ttx = float(info_policy.get("mean_T_tx", np.nan))
        pol_rate = float(info_policy.get("mean_rate", np.nan))

        imp_T = base_T - pol_T
        imp_E = base_E - pol_E
        imp_paper = float(info_base.get("paper_cost", np.nan)) - float(info_policy.get("paper_cost", np.nan))

        
        
        
        
        
        vio_total_pre = int(vio_coll_pre + vio_no_cover + vio_theta + theta_infeasible + vio_wall)
        vio_total_post = int(vio_coll_post + vio_no_cover + vio_theta + theta_infeasible + vio_wall)

        
        mode = str(getattr(self.cfg, "vio_penalty_mode", "pre")).strip().lower()
        if mode not in ("pre", "post"):
            mode = "pre"
        vio_coll_for_pen = int(vio_coll_post) if mode == "post" else int(vio_coll_pre)
        vio_total_penalty = int(vio_coll_for_pen + vio_no_cover + vio_theta + theta_infeasible + vio_wall)

        # ---- P1 reward: normalize T,E then combine with std-balancing + dynamic improve-based scheduling ----
        cost_policy_paper = float(info_policy["paper_cost"])
        cost_base_paper = float(info_base["paper_cost"])

        
        xT_pol = float(info_policy["mean_T_off"]) / max(float(self.cfg.T_scale), 1e-9)
        xE_pol = float(info_policy["mean_energy"]) / max(float(self.cfg.E_scale), 1e-9)
        xT_base = float(info_base["mean_T_off"]) / max(float(self.cfg.T_scale), 1e-9)
        xE_base = float(info_base["mean_energy"]) / max(float(self.cfg.E_scale), 1e-9)

        
        
        
        
        
        #
        
        
        
        #
        
        
        use_fixed_norm = bool(getattr(self.cfg, "use_fixed_normalization", False))
        if use_fixed_norm:
            
            zT_pol, zE_pol = float(xT_pol), float(xE_pol)
            zT_base, zE_base = float(xT_base), float(xE_base)
        else:
            
            norm_update_interval = int(getattr(self.cfg, "norm_update_interval", 10))
            if self.t % max(norm_update_interval, 1) == 0:
                self.rn_T.update(xT_pol)
                self.rn_E.update(xE_pol)
                self.rn_T.update(xT_base)
                self.rn_E.update(xE_base)

            zT_pol = float(self.rn_T.norm(xT_pol))
            zE_pol = float(self.rn_E.norm(xE_pol))
            zT_base = float(self.rn_T.norm(xT_base))
            zE_base = float(self.rn_E.norm(xE_base))
        # ===== [END P1 =====

        use_dyn, use_std, wm_norm = self._weight_flags(getattr(self.cfg, "weight_mode", "auto"))

        
        wT0, wE0 = _normalize_w(float(self.cfg.w_delay), float(self.cfg.w_energy))
        wT_use, wE_use = wT0, wE0

        dyn_impT = 0.0
        dyn_impE = 0.0
        dyn_reason = "off"

        # dynamic scheduling z-score
        if use_dyn:
            wT_use, wE_use, dyn_impT, dyn_impE, dyn_reason = self._dyn_update(zT_pol, zE_pol)
            wT_use, wE_use = _normalize_w(wT_use, wE_use)

        # std balancing
        if use_std:
            wT_eff = wT_use / (self.rn_T.std() + 1e-6)
            wE_eff = wE_use / (self.rn_E.std() + 1e-6)
            wT_eff, wE_eff = _normalize_w(wT_eff, wE_eff)
        else:
            wT_eff, wE_eff = wT_use, wE_use

        
        wT_eff, wE_eff, w_clamped = _clamp_weights(
            wT_eff, wE_eff,
            wT_min=float(getattr(self.cfg, "eff_wT_min", 0.0)),
            wE_min=float(getattr(self.cfg, "eff_wE_min", 0.0)),
        )

        
        wT_eff, wE_eff = _normalize_w(wT_eff, wE_eff)

        cost_policy_p1 = wT_eff * zT_pol + wE_eff * zE_pol
        cost_base_p1 = wT_eff * zT_base + wE_eff * zE_base

        
        if self.reward_mode == "paper":
            cost_policy = cost_policy_paper
            cost_base = cost_base_paper
        elif self.reward_mode == "relative":
            eps = 1e-12
            rel_T = float(info_policy["mean_T_off"]) / (float(info_base["mean_T_off"]) + eps)
            rel_E = float(info_policy["mean_energy"]) / (float(info_base["mean_energy"]) + eps)
            cost_policy = float(self.cfg.w_delay) * rel_T + float(self.cfg.w_energy) * rel_E
            cost_base = float(self.cfg.w_delay) * 1.0 + float(self.cfg.w_energy) * 1.0
        else:
            
            cost_policy = float(cost_policy_p1)
            cost_base = float(cost_base_p1)

        
        
        # 1) paper_cost
        
        
        
        
        paper_delta = 0.0
        collision_margin = 0.0
        coverage_margin = 0.0
        wall_margin = 0.0
        vio_rate_step = 0.0
        
        paper_cost_step = 0.0
        paper_cost_obj_step = 0.0
        paper_cost_norm_factor = 1.0
        paper_cost_ref_step = 0.0
        paper_delta_step = 0.0
        paper_delta_norm_step = 0.0
        paper_adv_step = 0.0
        paper_adv_norm_step = 0.0
        reward_paper_step = 0.0
        reward_paper_abs_step = 0.0
        reward_paper_delta_step = 0.0
        reward_paper_adv_step = 0.0
        reward_main_step = 0.0
        reward_traj_step = 0.0
        vio_any_step = 0.0
        violation_continuous_step = 0.0
        reward_constraint_step = 0.0
        constraint_signal_step = 0.0
        lambda_v_effective = float(getattr(self.cfg, "lambda_v", 0.0))
        t6_ref_cost_mode_used = ""
        reward_comp_step = 0.0
        reward_ris_step = 0.0
        reward_potential_step = 0.0
        reward_shaping_step = 0.0
        reward_misc_step = 0.0
        t6_violation_ema_step = 0.0
        t6_pid_frozen_step = 0.0
        t6_main_weight_step = 1.0
        t6_potential_scale_step = 1.0
        t6_traj_scale_step = 1.0
        t6_anneal_ratio_step = 0.0
        t6_cost_policy_raw_step = 0.0
        t6_cost_policy_obj_step = 0.0
        t6_cost_ref_raw_step = 0.0
        t6_cost_ref_obj_step = 0.0
        t6_delay_norm_step = 0.0
        t6_energy_norm_step = 0.0
        t6_delay_calib_step = 0.0
        t6_energy_calib_step = 0.0
        t6_metric_calib_enable_step = 0.0
        t7_alpha_step = 0.0
        t7_cost_ref_step = 0.0
        shaping_gate = 0.0
        beta_comp_t = 0.0
        beta_ris_t = 0.0
        beta_train_frac = 0.0
        comp_bonus_raw_step = 0.0
        ris_bonus_raw_step = 0.0
        
        reward_vio = 0.0
        reward_wall = 0.0
        reward_wall_margin = 0.0
        reward_safe_clip = 0.0
        reward_uncovered = 0.0
        reward_coverage_margin = 0.0
        reward_collision_margin = 0.0
        reward_movement = 0.0
        reward_boundary_stick = 0.0
        reward_user_nn = 0.0
        reward_user_centroid = 0.0
        reward_switchback = 0.0
        reward_traj_direct = 0.0
        reward_traj_gain_coupling = 0.0
        reward_constraint_bonus_spill = 0.0
        reward_proximity_step = 0.0
        reward_proximity_weight_effective_step = 0.0
        reward_coverage_floor_step = 0.0
        move_frac_step = 0.0
        traj_boundary_stick_step = 0.0
        traj_user_nn_step = 0.0
        traj_centroid_gap_step = 0.0
        traj_switchback_step = 0.0
        phaseh_aux_gate = 1.0
        phaseh_aux_term_traj = 0.0
        phaseh_aux_term_safety = 0.0
        shaping_gate_hard = 0.0
        shaping_gate_soft = 0.0
        comp_bonus_scale = 1.0
        ris_bonus_scale = 1.0
        ris_quality_gate = 1.0
        penalty_scale = float(getattr(self.cfg, "violation_penalty_scale", 1.0))
        
        z_cost_ref_runtime = float(max(float(getattr(self.cfg, "z_cost_ref", 1.0)), 1e-9))
        z_cost_ref_from_map = 0.0
        z_cost_ref_from_load_map = 0.0
        z_cost_ref_from_speed_map = 0.0
        
        z4_gap_cost_policy_step = 0.0
        z4_gap_cost_balanced_step = 0.0
        z4_gap_raw_step = 0.0
        z4_gap_step_clip = 0.0
        z4_gap_step_ema = 0.0
        z4_gap_step = 0.0
        z4_gap_reward_step = 0.0
        z4_gap_lambda_input = 0.0
        z4_gap_lambda_sched = 0.0
        z4_gap_lambda_cap = 0.0
        z4_gap_lambda_eff = 0.0
        z4_gap_grad_main = 0.0
        z4_gap_grad_ratio_eff = 0.0
        z4_gap_train_frac = 0.0
        z4_gap_shape_guard_trigger = 0.0
        z4_gap_shape_bad_streak = 0.0
        z4_gap_shape_slope_base = 0.0
        z4_gap_shape_slope_with = 0.0
        
        z4_comp_meta_enable_step = 0.0
        z4_comp_meta_ctrl_thr_raw = 0.0
        z4_comp_meta_ctrl_temp_raw = 0.0
        z4_comp_meta_warm_scale = 0.0
        z4_comp_meta_train_frac = 0.0
        z4_comp_meta_thr_base = float(np.clip(float(getattr(self.cfg, "comp_rule_threshold", 0.5)), 0.0, 1.0))
        z4_comp_meta_thr_delta_raw = 0.0
        z4_comp_meta_thr_delta_ema = float(getattr(self, "_z4_comp_meta_thr_delta_ema", 0.0))
        z4_comp_meta_thr_delta_eff = 0.0
        z4_comp_meta_thr_effective = float(z4_comp_meta_thr_base)
        z4_comp_meta_score_width = float(max(float(getattr(self.cfg, "z4_comp_meta_score_width", 0.20)), 1e-3))
        z4_comp_meta_temp_base = float(max(float(getattr(self.cfg, "comp_score_temp", 0.5)), 1e-6))
        z4_comp_meta_temp_scale_raw = 1.0
        z4_comp_meta_temp_scale_ema = float(getattr(self, "_z4_comp_meta_temp_scale_ema", 1.0))
        z4_comp_meta_temp_scale_eff = 1.0
        z4_comp_meta_temp_effective = float(z4_comp_meta_temp_base)
        
        z4_assoc_stage_enable_step = 0.0
        z4_assoc_policy_mix_step = 1.0
        z4_assoc_train_frac_step = 0.0
        z4_assoc_mode_effective = "policy"
        z4_assoc_stage_score_width = float(max(float(getattr(self.cfg, "z4_comp_meta_score_width", 0.20)), 1e-3))
        z4_assoc_stage_comp_rule_thr = float(np.clip(float(getattr(self.cfg, "comp_rule_threshold", 0.5)), 0.0, 1.0))

        
        if not hasattr(self, "_prev_potential"):
            self._prev_potential = 0.0
        prev_potential = float(self._prev_potential)
        current_potential = self._compute_potential(info_policy)
        gamma_shaping = 0.99
        potential_bonus = gamma_shaping * current_potential - prev_potential
        self._prev_potential = current_potential

        if reward_design in ("reward_total_v1", "reward_total", "total_v1", "v1"):
            # ===== PhaseG 1.8 eward/MDP eward_total = paper + shaping + constraint + misc =====
            
            
            
            

            
            paper_cost_step = float(cost_policy_paper)
            paper_cost_norm_factor = 1.0
            if bool(getattr(self.cfg, "phaseh_cost_norm_enable", False)):
                load_now = float(max(float(getattr(self.cfg, "load_scale", 1.0)), 1e-6))
                load_ref = float(max(float(getattr(self.cfg, "phaseh_cost_norm_ref", 0.6)), 1e-6))
                load_floor = float(max(float(getattr(self.cfg, "phaseh_cost_norm_floor", 0.2)), 1e-6))
                norm_power = float(max(float(getattr(self.cfg, "phaseh_cost_norm_power", 0.0)), 0.0))
                load_eff = float(max(load_now, load_floor))
                if norm_power > 1e-9:
                    paper_cost_norm_factor = float((load_ref / load_eff) ** norm_power)
            paper_cost_obj_step = float(paper_cost_step * paper_cost_norm_factor)
            cost_base_obj_step = float(float(cost_base_paper) * paper_cost_norm_factor)
            a = float(getattr(self.cfg, "paper_reward_a", 10.0))
            b = float(getattr(self.cfg, "paper_reward_b", 10.0))
            paper_mode = str(getattr(self.cfg, "paper_reward_mode", "log")).strip().lower()
            if paper_mode in ("linear", "lin"):
                reward_paper_abs_step = float(a - b * paper_cost_obj_step)
            else:
                # log1p(x) gives stronger early gradients and smoother late-stage behavior.
                reward_paper_abs_step = float(a - b * float(np.log1p(max(paper_cost_obj_step, 0.0))))

            phaseg_v2_enable = bool(getattr(self.cfg, "phaseg_v2_enable", False))
            if phaseg_v2_enable:
                prev_cost = self._prev_paper_cost_step
                if prev_cost is None or (not np.isfinite(float(prev_cost))):
                    prev_cost = float(paper_cost_obj_step)
                paper_delta_step = float(prev_cost - paper_cost_obj_step)
                self._prev_paper_cost_step = float(paper_cost_obj_step)

                ref_beta = float(np.clip(float(getattr(self.cfg, "phaseg_v2_cost_ref_ema_beta", 0.90)), 0.0, 0.999))
                ref_min = float(max(float(getattr(self.cfg, "phaseg_v2_cost_ref_min", 0.5)), 1e-6))
                ref_init = float(max(float(getattr(self.cfg, "phaseg_v2_cost_ref_init", 5.0)), ref_min))
                local_scale = float(max(abs(float(prev_cost)), abs(float(paper_cost_obj_step)), ref_min))
                if (not np.isfinite(self._paper_cost_ref_ema)) or (float(self._paper_cost_ref_ema) <= 0.0):
                    self._paper_cost_ref_ema = float(max(ref_init, local_scale))
                self._paper_cost_ref_ema = (
                    ref_beta * float(self._paper_cost_ref_ema)
                    + (1.0 - ref_beta) * float(local_scale)
                )
                paper_cost_ref_step = float(max(float(self._paper_cost_ref_ema), ref_min))

                delta_clip = float(max(float(getattr(self.cfg, "phaseg_v2_delta_clip", 1.0)), 1e-6))
                paper_delta_norm_step = float(
                    np.clip(float(paper_delta_step) / float(max(paper_cost_ref_step, 1e-6)), -delta_clip, delta_clip)
                    / delta_clip
                )
                reward_paper_delta_step = float(
                    float(getattr(self.cfg, "phaseg_v2_paper_delta_weight", 2.0)) * float(paper_delta_norm_step)
                )
                
                paper_adv_step = float(cost_base_obj_step - paper_cost_obj_step)
                paper_adv_scale = float(max(abs(float(cost_base_obj_step)), abs(float(paper_cost_obj_step)), ref_min))
                adv_temp = float(max(float(getattr(self.cfg, "phaseg_v2_paper_adv_temp", 1.0)), 1e-6))
                paper_adv_norm_step = float(np.tanh((float(paper_adv_step) / paper_adv_scale) / adv_temp))
                reward_paper_adv_step = float(
                    float(getattr(self.cfg, "phaseg_v2_paper_adv_weight", 0.5)) * float(paper_adv_norm_step)
                )
                reward_paper_step = float(
                    float(getattr(self.cfg, "phaseg_v2_paper_abs_weight", 1.0)) * float(reward_paper_abs_step)
                    + float(reward_paper_delta_step)
                    + float(reward_paper_adv_step)
                )
            else:
                self._prev_paper_cost_step = float(paper_cost_obj_step)
                paper_cost_ref_step = float(max(abs(paper_cost_obj_step), 1e-6))
                reward_paper_step = float(reward_paper_abs_step)

            
            cov_frac = float(np.clip(float(info_policy.get("coverage_fraction", 0.0)), 0.0, 1.0))
            vio_any_step = 1.0 if int(vio_total_penalty) > 0 else 0.0
            constraint_signal_step = float(np.clip(0.5 * float(vio_any_step) + 0.5 * (1.0 - cov_frac), 0.0, 1.0))

            lambda_v_base = float(getattr(self.cfg, "lambda_v", 0.0))
            lambda_v_effective = float(lambda_v_base)
            if phaseg_v2_enable and bool(getattr(self.cfg, "phaseg_v2_lambda_adapt", False)):
                target = float(np.clip(float(getattr(self.cfg, "phaseg_v2_lambda_target", 0.35)), 0.0, 1.0))
                lam_lr = float(max(float(getattr(self.cfg, "phaseg_v2_lambda_lr", 0.01)), 0.0))
                lam_min = float(getattr(self.cfg, "phaseg_v2_lambda_min", 0.0))
                lam_max = float(max(float(getattr(self.cfg, "phaseg_v2_lambda_max", 20.0)), lam_min))
                if (not np.isfinite(self._lambda_v_dyn)) or (float(self._lambda_v_dyn) < lam_min):
                    self._lambda_v_dyn = float(np.clip(lambda_v_base, lam_min, lam_max))
                self._lambda_v_dyn = float(
                    np.clip(float(self._lambda_v_dyn) + lam_lr * (constraint_signal_step - target), lam_min, lam_max)
                )
                lambda_v_effective = float(self._lambda_v_dyn)
            else:
                self._lambda_v_dyn = float(lambda_v_base)
                lambda_v_effective = float(lambda_v_base)
            reward_constraint_step = float(-lambda_v_effective * float(constraint_signal_step))

            # ---- Mobile-user reward_total_v1 ----
            
            
            reward_wall_weight = float(getattr(self.cfg, "reward_wall_weight", 0.0))
            reward_wall = -reward_wall_weight * float(vio_wall)

            wall_safe = self._wall_safe_margin()
            wall_safe = float(max(wall_safe, 1e-6))
            q_now = None
            try:
                q_now = np.asarray(self.q, dtype=np.float64).reshape(int(self.cfg.M), 2)
                d_left = q_now[:, 0]
                d_right = float(self.cfg.L) - q_now[:, 0]
                d_down = q_now[:, 1]
                d_up = float(self.cfg.L) - q_now[:, 1]
                d_min = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
                wall_margin = float(np.mean(np.clip((wall_safe - d_min) / wall_safe, 0.0, 1.0)))
                traj_boundary_stick_step = float(np.mean(d_min <= self._boundary_eval_band()))
            except Exception as exc:
                self._warn_once("reward_total_wall_margin", f"reward_total_v10.0{exc}")
                q_now = None
                wall_margin = 0.0
                traj_boundary_stick_step = 0.0
            reward_wall_margin_weight = float(getattr(self.cfg, "reward_wall_margin_weight", 0.0))
            reward_wall_margin = -reward_wall_margin_weight * float(wall_margin)
            reward_boundary_stick_weight = float(getattr(self.cfg, "reward_boundary_stick_weight", 0.0))
            reward_boundary_stick = -reward_boundary_stick_weight * float(traj_boundary_stick_step)
            reward_safe_clip_weight = float(getattr(self.cfg, "reward_safe_clip_weight", 0.0))
            reward_safe_clip = -reward_safe_clip_weight * float(np.clip(safe_clip_projection_step, 0.0, 2.0))

            
            reward_boundary_potential = 0.0
            boundary_potential_weight = float(getattr(self.cfg, "reward_boundary_potential_weight", 0.0))
            if boundary_potential_weight > 0.0 and q_now is not None:
                boundary_potential_scale = float(getattr(self.cfg, "reward_boundary_potential_scale", 0.1)) * float(self.cfg.L)
                boundary_potential_scale = max(boundary_potential_scale, 1e-6)
                boundary_potential = float(np.mean(np.exp(-d_min / boundary_potential_scale) - np.exp(-1.0)))
                reward_boundary_potential = -boundary_potential_weight * float(np.clip(boundary_potential, -1.0, 1.0))

            reward_constraint_step = float(reward_constraint_step + reward_boundary_stick + reward_safe_clip + reward_boundary_potential)
            # H4.C alignment: boundary-stick and safe-clip penalties are injected directly.

            reward_uncovered_weight = float(getattr(self.cfg, "reward_uncovered_weight", 6.0))
            reward_uncovered = -penalty_scale * reward_uncovered_weight * float(1.0 - cov_frac)

            coverage_margin = 0.0
            try:
                Rc2_eff_local = float(self.cfg.Rc) * float(self.cfg.Rc) - float(self.cfg.H) * float(self.cfg.H)
                Rc_eff = float(np.sqrt(max(Rc2_eff_local, 0.0)))
                if Rc_eff > 1e-9:
                    dmin_user = np.sqrt(np.min(dist2, axis=1))  # (I,)
                    margin = np.maximum(dmin_user - Rc_eff, 0.0) / max(Rc_eff, 1e-9)
                    coverage_margin = float(np.mean(margin))
            except Exception as exc:
                self._warn_once("reward_total_coverage_margin", f"reward_total_v10.0{exc}")
                coverage_margin = 0.0
            reward_coverage_margin_weight = float(getattr(self.cfg, "reward_coverage_margin_weight", 0.0))
            reward_coverage_margin = -penalty_scale * reward_coverage_margin_weight * float(coverage_margin)

            dmin_pair = float(self.cfg.dmin)
            M = int(self.cfg.M)
            margin_sum = 0.0
            pair_cnt = 0
            for i in range(M):
                for j in range(i + 1, M):
                    dist = float(np.linalg.norm(q_pre_repair[i] - q_pre_repair[j]))
                    margin = max(dmin_pair - dist, 0.0) / max(dmin_pair, 1e-9)
                    margin_sum += margin
                    pair_cnt += 1
            collision_margin = margin_sum / max(pair_cnt, 1)
            reward_collision_margin_weight = float(getattr(self.cfg, "reward_collision_margin_weight", 3.0))
            reward_collision_margin = -penalty_scale * reward_collision_margin_weight * float(collision_margin)

            hover_ratio = float(np.clip(float(info_policy.get("hover_energy_ratio", 0.0)), 0.0, 1.0))
            move_frac_step = float(np.clip(1.0 - hover_ratio, 0.0, 1.0))
            reward_movement_weight = float(getattr(self.cfg, "reward_movement_weight", 0.0))
            movement_target_enable = bool(getattr(self.cfg, "reward_movement_target_enable", False))
            if movement_target_enable:
                movement_target = float(np.clip(float(getattr(self.cfg, "reward_movement_target", 0.35)), 0.0, 1.0))
                movement_dev = float(abs(move_frac_step - movement_target))
                reward_movement = -reward_movement_weight * movement_dev
            else:
                # eward_movement_weight>0
                reward_movement = -reward_movement_weight * hover_ratio

            
            try:
                w_now = np.asarray(self.w, dtype=np.float64).reshape(-1, 2)
                if (q_now is not None) and (w_now.shape[0] > 0):
                    diff_qw = q_now[:, None, :] - w_now[None, :, :]
                    dist_qw = np.sqrt(np.sum(diff_qw * diff_qw, axis=2))
                    traj_user_nn_step = float(np.mean(np.min(dist_qw, axis=1)) / max(float(self.cfg.L), 1e-9))
                    traj_centroid_gap_step = float(
                        np.linalg.norm(np.mean(q_now, axis=0) - np.mean(w_now, axis=0)) / max(float(self.cfg.L), 1e-9)
                    )
            except Exception as exc:
                self._warn_once("reward_total_traj_metrics", f"reward_total_v10.0{exc}")
                traj_user_nn_step = 0.0
                traj_centroid_gap_step = 0.0

            traj_switchback_step = 0.0
            if self.q_prev is not None and q_now is not None:
                try:
                    vel_now = np.asarray(q_now - self.q_prev, dtype=np.float64)
                    vel_prev = self._traj_prev_vel
                    if vel_prev is not None and np.asarray(vel_prev).shape == vel_now.shape:
                        dot = np.sum(vel_now * vel_prev, axis=1)
                        n0 = np.linalg.norm(vel_prev, axis=1)
                        n1 = np.linalg.norm(vel_now, axis=1)
                        valid = (n0 > 1e-9) & (n1 > 1e-9)
                        if np.any(valid):
                            cosv = dot[valid] / (n0[valid] * n1[valid] + 1e-12)
                            traj_switchback_step = float(np.mean(cosv < 0.0))
                    self._traj_prev_vel = vel_now.copy()
                except Exception as exc:
                    self._warn_once("reward_total_switchback", f"reward_total_v10.0{exc}")
                    self._traj_prev_vel = None
                    traj_switchback_step = 0.0
            else:
                self._traj_prev_vel = None

            reward_user_nn_weight = float(getattr(self.cfg, "reward_user_nn_weight", 0.0))
            reward_user_nn = -reward_user_nn_weight * float(np.clip(traj_user_nn_step, 0.0, 2.0))
            reward_user_centroid_weight = float(getattr(self.cfg, "reward_user_centroid_weight", 0.0))
            reward_user_centroid = -reward_user_centroid_weight * float(np.clip(traj_centroid_gap_step, 0.0, 2.0))
            reward_switchback_weight = float(getattr(self.cfg, "reward_switchback_weight", 0.0))
            reward_switchback = -reward_switchback_weight * float(np.clip(traj_switchback_step, 0.0, 1.0))
            traj_direct_enable = bool(getattr(self.cfg, "phaseh_traj_direct_enable", False))
            traj_direct_scale = float(max(float(getattr(self.cfg, "phaseh_traj_direct_scale", 1.0)), 0.0))
            if traj_direct_enable and traj_direct_scale > 0.0:
                reward_traj_direct = float(
                    traj_direct_scale * (reward_user_nn + reward_user_centroid + reward_switchback)
                )
                reward_constraint_step = float(reward_constraint_step + reward_traj_direct)
            else:
                reward_traj_direct = 0.0

            aux_constraint_scale = float(np.clip(float(getattr(self.cfg, "phaseh_aux_constraint_scale", 0.0)), 0.0, 1.0))
            aux_traj_mix = float(np.clip(float(getattr(self.cfg, "phaseh_aux_traj_mix", 1.0)), 0.0, 2.0))
            aux_safety_mix = float(np.clip(float(getattr(self.cfg, "phaseh_aux_safety_mix", 1.0)), 0.0, 2.0))
            phaseh_aux_term_traj = float(reward_wall + reward_wall_margin + reward_movement)
            if not traj_direct_enable:
                phaseh_aux_term_traj = float(
                    phaseh_aux_term_traj + reward_user_nn + reward_user_centroid + reward_switchback
                )
            phaseh_aux_term_safety = float(reward_uncovered + reward_coverage_margin + reward_collision_margin)
            phaseh_aux_gate = 1.0
            if bool(getattr(self.cfg, "phaseh_aux_gate_enable", False)):
                idle_target = float(np.clip(float(getattr(self.cfg, "phaseh_aux_idle_target", 0.60)), 0.0, 0.99))
                idle_pressure = float(np.clip((float(hover_ratio) - idle_target) / max(1.0 - idle_target, 1e-6), 0.0, 1.0))
                bnd_target = float(np.clip(float(getattr(self.cfg, "phaseh_aux_boundary_margin_target", 0.25)), 1e-6, 1.0))
                bnd_pressure = float(np.clip(float(wall_margin) / bnd_target, 0.0, 1.0))
                gate_power = float(max(float(getattr(self.cfg, "phaseh_aux_gate_power", 1.0)), 1e-6))
                phaseh_aux_gate = float(max(idle_pressure, bnd_pressure) ** gate_power)
            reward_constraint_step = float(
                reward_constraint_step
                + aux_constraint_scale
                * phaseh_aux_gate
                * (aux_traj_mix * phaseh_aux_term_traj + aux_safety_mix * phaseh_aux_term_safety)
            )
            reward_constraint_bonus_spill = float(max(reward_constraint_step, 0.0))
            if reward_constraint_bonus_spill > 0.0:
                reward_constraint_step = float(reward_constraint_step - reward_constraint_bonus_spill)

            
            _gate_simple = bool(getattr(self.cfg, "phaseh_shaping_gate_simple", False))
            if _gate_simple:
                
                shaping_gate_hard = float(cov_frac)
                shaping_gate_soft = float(cov_frac)
                shaping_gate = float(cov_frac)
            else:
                shaping_gate_hard = float(cov_frac * (1.0 - float(vio_any_step)))
                shaping_gate_soft = float(cov_frac * np.clip(1.0 - float(constraint_signal_step), 0.0, 1.0))
                shaping_soft_mix = float(np.clip(float(getattr(self.cfg, "phaseh_shaping_soft_gate_mix", 0.0)), 0.0, 1.0))
                shaping_gate = float((1.0 - shaping_soft_mix) * shaping_gate_hard + shaping_soft_mix * shaping_gate_soft)
                shaping_gate_floor = float(np.clip(float(getattr(self.cfg, "phaseh_shaping_gate_floor", 0.0)), 0.0, 1.0))
                if shaping_gate_floor > 0.0:
                    shaping_gate = float(max(shaping_gate, shaping_gate_floor * cov_frac))

            beta_comp_t, beta_train_frac = self._phaseg_beta(float(getattr(self.cfg, "beta_comp_max", 0.0)))
            beta_ris_t, _ = self._phaseg_beta(float(getattr(self.cfg, "beta_ris_max", 0.0)))

            
            comp_gain_ratio_step = float(max(float(info_policy.get("comp_gain_ratio", 0.0)), 0.0))
            comp_users_fraction_step = float(info_policy.get("comp_users_fraction", 0.0))
            comp_bonus_scale = float(max(float(getattr(self.cfg, "phaseh_comp_bonus_scale", 1.0)), 0.0))
            comp_bonus_raw_step = float(
                max(comp_gain_ratio_step - 1.0, 0.0)
                * float(np.clip(comp_users_fraction_step, 0.0, 1.0))
                * comp_bonus_scale
            )

            
            ris_gain_ratio_step = float(np.clip(float(info_policy.get("ris_gain_ratio", 0.0)), 0.0, 1.0))
            ris_users_fraction_step = float(info_policy.get("ris_users_fraction", 0.0))
            ris_bonus_scale = float(max(float(getattr(self.cfg, "phaseh_ris_bonus_scale", 1.0)), 0.0))

            
            ris_allow_negative = bool(getattr(self.cfg, "phaseh_ris_allow_negative", False))
            if ris_allow_negative:
                ris_baseline = float(getattr(self.cfg, "phaseh_ris_baseline", 0.15))
                ris_bonus_raw_step = float(
                    (ris_gain_ratio_step - ris_baseline)
                    * float(np.clip(ris_users_fraction_step, 0.0, 1.0))
                    * ris_bonus_scale
                )
            else:
                ris_bonus_raw_step = float(
                    max(ris_gain_ratio_step, 0.0)
                    * float(np.clip(ris_users_fraction_step, 0.0, 1.0))
                    * ris_bonus_scale
                )
            ris_quality_gate = 1.0
            if bool(getattr(self.cfg, "phaseh_ris_quality_gate_enable", False)):
                gate_floor = float(np.clip(float(getattr(self.cfg, "phaseh_ris_quality_gate_floor", 0.0)), 0.0, 1.0))
                gate_power = float(max(float(getattr(self.cfg, "phaseh_ris_quality_gate_power", 1.0)), 1e-6))
                gate_raw = float(np.clip(float(paper_adv_norm_step), 0.0, 1.0))
                ris_quality_gate = float(max(gate_floor, gate_raw ** gate_power))

            
            shaping_gate_bypass = bool(getattr(self.cfg, "phaseh_shaping_gate_bypass", False))
            if shaping_gate_bypass:
                reward_comp_step = float(beta_comp_t * comp_bonus_raw_step)
                reward_ris_step = float(beta_ris_t * ris_bonus_raw_step * ris_quality_gate)
            else:
                reward_comp_step = float(shaping_gate * beta_comp_t * comp_bonus_raw_step)
                reward_ris_step = float(shaping_gate * beta_ris_t * ris_bonus_raw_step * ris_quality_gate)

            reward_potential_weight = float(getattr(self.cfg, "reward_potential_weight", 2.0))
            reward_potential_step = float(reward_potential_weight * float(potential_bonus))

            
            
            reward_traj_gain_coupling = 0.0
            traj_gain_coupling_enable = bool(getattr(self.cfg, "phaseh_traj_gain_coupling_enable", False))
            if traj_gain_coupling_enable:
                traj_comp_coupling_weight = float(getattr(self.cfg, "phaseh_traj_comp_coupling_weight", 0.0))
                traj_ris_coupling_weight = float(getattr(self.cfg, "phaseh_traj_ris_coupling_weight", 0.0))
                ris_coupling_baseline = float(getattr(self.cfg, "phaseh_ris_coupling_baseline", 0.15))

                
                reward_traj_comp_coupling = traj_comp_coupling_weight * max(comp_gain_ratio_step - 1.0, 0.0)

                
                reward_traj_ris_coupling = traj_ris_coupling_weight * (ris_gain_ratio_step - ris_coupling_baseline)

                reward_traj_gain_coupling = float(reward_traj_comp_coupling + reward_traj_ris_coupling)
            
            
            reward_shaping_step = float(
                reward_comp_step
                + reward_ris_step
                + reward_potential_step
                + reward_traj_gain_coupling
                + reward_constraint_bonus_spill
            )

            
            reward_misc_step = float(getattr(self.cfg, "reward_step_bias", 0.0))
            reward_proximity_weight_effective_step = float(self._current_reward_proximity_weight())
            reward_proximity_step = float(self._compute_reward_proximity_step(dist2))
            reward_misc_step = float(reward_misc_step + reward_proximity_step)

            
            reward_survival = float(reward_misc_step)
            reward_safety = float(reward_constraint_step)
            reward_performance = float(reward_paper_step)
            reward_improve = 0.0
            reward_potential = float(reward_potential_step)

            reward_total = float(reward_paper_step + reward_shaping_step + reward_constraint_step + reward_misc_step)
            reward_raw = float(reward_total)

        elif reward_design in ("paper_delta_v2", "paper_delta", "v2"):
            
            paper_delta = float(cost_base_paper) - float(cost_policy_paper)
            paper_delta_scale = float(getattr(self.cfg, "reward_paper_delta_scale", 200.0))
            
            paper_cost_weight = float(getattr(self.cfg, "reward_paper_cost_weight", 0.0))
            reward_cost_term = -paper_cost_weight * float(cost_policy_paper)
            reward_performance = paper_delta_scale * paper_delta + reward_cost_term

            # reward
            
            
            

            
            reward_violation_weight = float(getattr(self.cfg, "reward_violation_weight", 4.0))
            reward_vio = -penalty_scale * reward_violation_weight * float(vio_total_penalty)

            
            
            # - vio_wall
            
            reward_wall_weight = float(getattr(self.cfg, "reward_wall_weight", 0.0))
            reward_wall = -reward_wall_weight * float(vio_wall)

            # 2.6) vio_wall
            
            wall_safe = self._wall_safe_margin()
            wall_safe = float(max(wall_safe, 1e-6))
            try:
                q_now = np.asarray(self.q, dtype=np.float64).reshape(int(self.cfg.M), 2)
                d_left = q_now[:, 0]
                d_right = float(self.cfg.L) - q_now[:, 0]
                d_down = q_now[:, 1]
                d_up = float(self.cfg.L) - q_now[:, 1]
                d_min = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
                wall_margin = float(np.mean(np.clip((wall_safe - d_min) / wall_safe, 0.0, 1.0)))
                traj_boundary_stick_step = float(np.mean(d_min <= self._boundary_eval_band()))
            except Exception as exc:
                self._warn_once("paper_delta_wall_margin", f"paper_delta_v20.0{exc}")
                wall_margin = 0.0
                traj_boundary_stick_step = 0.0
            reward_wall_margin_weight = float(getattr(self.cfg, "reward_wall_margin_weight", 0.0))
            reward_wall_margin = -reward_wall_margin_weight * float(wall_margin)
            reward_boundary_stick_weight = float(getattr(self.cfg, "reward_boundary_stick_weight", 0.0))
            reward_boundary_stick = -reward_boundary_stick_weight * float(traj_boundary_stick_step)
            reward_safe_clip_weight = float(getattr(self.cfg, "reward_safe_clip_weight", 0.0))
            reward_safe_clip = -reward_safe_clip_weight * float(np.clip(safe_clip_projection_step, 0.0, 2.0))

            
            coverage_fraction = float(info_policy.get("coverage_fraction", 0.0))
            reward_uncovered_weight = float(getattr(self.cfg, "reward_uncovered_weight", 6.0))
            reward_uncovered = -penalty_scale * reward_uncovered_weight * float(
                1.0 - np.clip(coverage_fraction, 0.0, 1.0)
            )

            
            
            
            coverage_margin = 0.0
            try:
                Rc2_eff_local = float(self.cfg.Rc) * float(self.cfg.Rc) - float(self.cfg.H) * float(self.cfg.H)
                Rc_eff = float(np.sqrt(max(Rc2_eff_local, 0.0)))
                if Rc_eff > 1e-9:
                    dmin = np.sqrt(np.min(dist2, axis=1))  # (I,)
                    margin = np.maximum(dmin - Rc_eff, 0.0) / max(Rc_eff, 1e-9)
                    coverage_margin = float(np.mean(margin))
            except Exception as exc:
                self._warn_once("paper_delta_coverage_margin", f"paper_delta_v20.0{exc}")
                coverage_margin = 0.0
            reward_coverage_margin_weight = float(getattr(self.cfg, "reward_coverage_margin_weight", 0.0))
            reward_coverage_margin = -penalty_scale * reward_coverage_margin_weight * float(coverage_margin)

            
            dmin = float(self.cfg.dmin)
            M = int(self.cfg.M)
            margin_sum = 0.0
            pair_cnt = 0
            for i in range(M):
                for j in range(i + 1, M):
                    dist = float(np.linalg.norm(q_pre_repair[i] - q_pre_repair[j]))
                    margin = max(dmin - dist, 0.0) / max(dmin, 1e-9)
                    margin_sum += margin
                    pair_cnt += 1
            collision_margin = margin_sum / max(pair_cnt, 1)
            reward_collision_margin_weight = float(getattr(self.cfg, "reward_collision_margin_weight", 3.0))
            reward_collision_margin = -penalty_scale * reward_collision_margin_weight * float(collision_margin)

            
            vmax_dt = float(self.cfg.Vmax) * float(self.cfg.dt)
            move_frac = 0.0
            try:
                
                move_norm = np.linalg.norm(np.asarray(delta_q_meters, dtype=np.float64), axis=1)
                move_frac = float(np.mean(move_norm / max(vmax_dt, 1e-9)))
            except Exception as exc:
                self._warn_once("paper_delta_move_frac", f"paper_delta_v20.0{exc}")
                move_frac = 0.0
            reward_movement_weight = float(getattr(self.cfg, "reward_movement_weight", 1.0))
            reward_movement = -reward_movement_weight * float(move_frac)

            
            
            
            # - coverage_margin/collision_margin
            
            vio_rate_step = float(
                np.clip(
                    (1.0 - float(np.clip(coverage_fraction, 0.0, 1.0)))
                    + 0.5 * float(np.clip(coverage_margin, 0.0, 1.0))
                    + 0.5 * float(np.clip(collision_margin, 0.0, 1.0)),
                    0.0,
                    1.0,
                )
            )
            reward_performance = float(reward_performance) * float(1.0 - vio_rate_step)
            # ===== [END B 2] =====

            
            reward_potential_weight = float(getattr(self.cfg, "reward_potential_weight", 2.0))
            reward_potential = reward_potential_weight * float(potential_bonus)

            
            reward_survival = float(getattr(self.cfg, "reward_step_bias", 0.0))

            
            reward_safety = float(
                reward_vio
                + reward_wall
                + reward_wall_margin
                + reward_uncovered
                + reward_coverage_margin
                + reward_collision_margin
                + reward_movement
                + reward_boundary_stick
                + reward_safe_clip
            )
            reward_improve = 0.0
            reward_total = float(reward_survival + reward_safety + reward_performance + reward_improve + reward_potential)
            reward_raw = reward_total

        elif reward_design in ("tcom_simple_enhanced", "tcom_simple", "h13_s1"):
            
            
            
            
            
            

            
            paper_cost_step = float(cost_policy_paper)
            reward_performance = -paper_cost_step

            
            lambda_v = float(getattr(self.cfg, "lambda_v", 3.0))
            vio_any_step = float(vio_total_penalty > 1e-9)
            reward_constraint_step = -lambda_v * vio_any_step

            
            lambda_smooth = float(getattr(self.cfg, "lambda_smooth", 0.5))
            smoothness_penalty = 0.0
            if self._traj_prev_vel is not None:
                try:
                    vel_now = np.asarray(delta_q_meters, dtype=np.float64)  # (M, 2)
                    vel_prev = np.asarray(self._traj_prev_vel, dtype=np.float64)  # (M, 2)
                    
                    vel_change = np.linalg.norm(vel_now - vel_prev, axis=1)  # (M,)
                    vmax_dt = float(self.cfg.Vmax) * float(self.cfg.dt)
                    smoothness_penalty = float(np.mean(vel_change / max(vmax_dt, 1e-9)))
                except Exception as exc:
                    self._warn_once("tcom_smoothness", f"{exc}")
                    smoothness_penalty = 0.0
            
            self._traj_prev_vel = np.asarray(delta_q_meters, dtype=np.float64).copy()
            reward_smoothness = -lambda_smooth * smoothness_penalty

            
            lambda_explore = float(getattr(self.cfg, "lambda_explore", 0.3))
            explore_move_ratio = float(getattr(self.cfg, "explore_move_ratio", 0.0))
            exploration_bonus = 0.0
            try:
                
                
                dist2 = pairwise_dist2(self.q, self.w).T  
                min_dist = np.sqrt(np.min(dist2, axis=1))  
                avg_min_dist_norm = float(np.mean(min_dist / max(float(self.cfg.L), 1e-9)))
                position_bonus = 1.0 - avg_min_dist_norm

                
                movement_bonus = 0.0
                if explore_move_ratio > 1e-9:
                    speed_norm = np.linalg.norm(np.asarray(delta_q_meters, dtype=np.float64), axis=1)
                    vmax_dt = float(self.cfg.Vmax) * float(self.cfg.dt)
                    avg_speed_norm = float(np.mean(speed_norm / max(vmax_dt, 1e-9)))
                    movement_bonus = float(np.clip(avg_speed_norm, 0.0, 1.0))

                
                exploration_bonus = (1.0 - explore_move_ratio) * position_bonus + explore_move_ratio * movement_bonus
            except Exception as exc:
                self._warn_once("tcom_explore", f"{exc}")
                exploration_bonus = 0.0
            reward_exploration = lambda_explore * exploration_bonus

            
            lambda_move = float(getattr(self.cfg, "lambda_move", 0.0))
            reward_movement = 0.0
            if lambda_move > 1e-9:
                try:
                    speed_norm = np.linalg.norm(np.asarray(delta_q_meters, dtype=np.float64), axis=1)
                    vmax_dt = float(self.cfg.Vmax) * float(self.cfg.dt)
                    avg_speed = float(np.mean(speed_norm / max(vmax_dt, 1e-9)))
                    reward_movement = lambda_move * float(np.clip(avg_speed, 0.0, 1.0))
                except Exception as exc:
                    self._warn_once("tcom_move", f"{exc}")
                    reward_movement = 0.0

            
            reward_survival = 0.0
            reward_safety = reward_constraint_step
            reward_improve = 0.0
            reward_potential = 0.0
            reward_misc_step = reward_smoothness + reward_exploration + reward_movement
            reward_total = float(
                reward_performance + reward_constraint_step + reward_smoothness + reward_exploration + reward_movement
            )
            reward_raw = reward_total

            
            reward_vio = reward_constraint_step
            move_frac_step = smoothness_penalty
            traj_user_nn_step = avg_min_dist_norm if 'avg_min_dist_norm' in locals() else 0.0
            reward_proximity_step = 0.0  

        elif reward_design == "tcom_simple_v2":
            
            
            
            
            
            
            
            

            
            paper_cost_step = float(cost_policy_paper)
            reward_performance = -paper_cost_step

            
            lambda_v = float(getattr(self.cfg, "lambda_v", 3.0))
            vio_any_step = float(vio_total_penalty > 1e-9)
            reward_constraint_step = -lambda_v * vio_any_step

            
            lambda_comp_switch = float(getattr(self.cfg, "lambda_comp_switch", 0.15))
            comp_switch_max = float(getattr(self.cfg, "comp_switch_max", 10.0))
            comp_switch_bonus = 0.0
            if lambda_comp_switch > 1e-9:
                try:
                    
                    
                    if not hasattr(self, "_prev_comp_selection"):
                        self._prev_comp_selection = None

                    if self._prev_comp_selection is not None and hasattr(self, "comp_selection"):
                        
                        comp_now = np.asarray(self.comp_selection, dtype=np.int32)
                        comp_prev = np.asarray(self._prev_comp_selection, dtype=np.int32)
                        num_switches = float(np.sum(comp_now != comp_prev))
                        comp_switch_bonus = float(np.clip(num_switches / max(comp_switch_max, 1e-9), 0.0, 1.0))

                    
                    if hasattr(self, "comp_selection"):
                        self._prev_comp_selection = np.asarray(self.comp_selection, dtype=np.int32).copy()
                except Exception as exc:
                    self._warn_once("tcom_comp_switch", f"CoMP {exc}")
                    comp_switch_bonus = 0.0
            reward_comp_switch = lambda_comp_switch * comp_switch_bonus

            
            lambda_velocity_penalty = float(getattr(self.cfg, "lambda_velocity_penalty", 0.5))
            velocity_threshold = float(getattr(self.cfg, "velocity_threshold", 0.1))
            idle_penalty = 0.0
            if lambda_velocity_penalty > 1e-9:
                try:
                    
                    vel_now = np.asarray(delta_q_meters, dtype=np.float64)  # (M, 2)
                    speed_norm = np.linalg.norm(vel_now, axis=1)  # (M,)
                    vmax_dt = float(self.cfg.Vmax) * float(self.cfg.dt)
                    avg_speed_norm = float(np.mean(speed_norm / max(vmax_dt, 1e-9)))

                    
                    if avg_speed_norm < velocity_threshold:
                        idle_penalty = 1.0 - (avg_speed_norm / max(velocity_threshold, 1e-9))
                    else:
                        idle_penalty = 0.0
                except Exception as exc:
                    self._warn_once("tcom_velocity", f"{exc}")
                    idle_penalty = 0.0
            reward_velocity_penalty = -lambda_velocity_penalty * idle_penalty

            
            reward_survival = 0.0
            reward_safety = reward_constraint_step
            reward_improve = 0.0
            reward_potential = 0.0
            reward_misc_step = reward_comp_switch + reward_velocity_penalty
            reward_total = float(
                reward_performance + reward_constraint_step + reward_comp_switch + reward_velocity_penalty
            )
            reward_raw = reward_total

            
            reward_vio = reward_constraint_step
            move_frac_step = idle_penalty
            traj_user_nn_step = 0.0
            reward_proximity_step = comp_switch_bonus

        elif reward_design == "tcom_simple_v3":
            
            
            
            
            
            
            
            

            
            paper_cost_weight = float(getattr(self.cfg, "paper_cost_weight", 2.0))
            paper_cost_step = float(cost_policy_paper)
            reward_paper = -paper_cost_weight * paper_cost_step

            
            lambda_v = float(getattr(self.cfg, "lambda_v", 3.0))
            vio_any_step = float(vio_total_penalty > 1e-9)
            reward_constraint = -lambda_v * vio_any_step

            
            lambda_comp_usage = float(getattr(self.cfg, "lambda_comp_usage", 0.3))
            comp_usage_bonus = 0.0
            if lambda_comp_usage > 1e-9:
                try:
                    
                    if hasattr(self, "a_binary"):
                        a_binary = np.asarray(self.a_binary, dtype=np.int32)  # (I, M)
                        num_comp_per_uav = np.sum(a_binary, axis=0)  
                        num_using_comp = float(np.sum(num_comp_per_uav >= 2))  
                        comp_usage_bonus = num_using_comp / max(float(self.M), 1e-9)
                except Exception as exc:
                    self._warn_once("tcom_comp_usage", f"CoMP {exc}")
                    comp_usage_bonus = 0.0
            reward_comp_usage = lambda_comp_usage * comp_usage_bonus

            
            lambda_ris_usage = float(getattr(self.cfg, "lambda_ris_usage", 0.2))
            ris_usage_bonus = 0.0
            if lambda_ris_usage > 1e-9:
                try:
                    
                    ris_gain_ratio_step = float(np.clip(float(info_policy.get("ris_gain_ratio", 0.0)), 0.0, 1.0))
                    ris_users_fraction_step = float(np.clip(float(info_policy.get("ris_users_fraction", 0.0)), 0.0, 1.0))
                    ris_usage_bonus = float(ris_gain_ratio_step * ris_users_fraction_step)
                except Exception as exc:
                    self._warn_once("tcom_ris_usage", f"RIS {exc}")
                    ris_usage_bonus = 0.0
            reward_ris_usage = lambda_ris_usage * ris_usage_bonus

            
            lambda_comp_switch = float(getattr(self.cfg, "lambda_comp_switch", 0.8))
            comp_switch_max = float(getattr(self.cfg, "comp_switch_max", 8.0))
            comp_switch_bonus_mode = str(getattr(self.cfg, "comp_switch_bonus_mode", "exponential")).strip().lower()
            comp_switch_bonus = 0.0
            if lambda_comp_switch > 1e-9:
                try:
                    if not hasattr(self, "_prev_comp_selection"):
                        self._prev_comp_selection = None

                    if self._prev_comp_selection is not None and hasattr(self, "a_binary"):
                        
                        a_now = np.asarray(self.a_binary, dtype=np.int32)  # (I, M)
                        a_prev = np.asarray(self._prev_comp_selection, dtype=np.int32)
                        
                        num_switches = 0.0
                        for m in range(self.M):
                            if not np.array_equal(a_now[:, m], a_prev[:, m]):
                                num_switches += 1.0

                        
                        normalized = float(np.clip(num_switches / max(comp_switch_max, 1e-9), 0.0, 1.0))

                        
                        if comp_switch_bonus_mode == "exponential":
                            comp_switch_bonus = (np.exp(normalized) - 1.0) / (np.e - 1.0)
                        else:
                            comp_switch_bonus = normalized

                    
                    if hasattr(self, "a_binary"):
                        self._prev_comp_selection = np.asarray(self.a_binary, dtype=np.int32).copy()
                except Exception as exc:
                    self._warn_once("tcom_comp_switch_v3", f"CoMP {exc}")
                    comp_switch_bonus = 0.0
            reward_comp_switch = lambda_comp_switch * comp_switch_bonus

            
            lambda_velocity_penalty = float(getattr(self.cfg, "lambda_velocity_penalty", 2.0))
            velocity_threshold = float(getattr(self.cfg, "velocity_threshold", 0.15))
            velocity_penalty_mode = str(getattr(self.cfg, "velocity_penalty_mode", "quadratic")).strip().lower()
            velocity_penalty = 0.0
            if lambda_velocity_penalty > 1e-9:
                try:
                    vel_now = np.asarray(delta_q_meters, dtype=np.float64)  # (M, 2)
                    speed_norm = np.linalg.norm(vel_now, axis=1)  # (M,)
                    vmax_dt = float(self.cfg.Vmax) * float(self.cfg.dt)
                    avg_speed_norm = float(np.mean(speed_norm / max(vmax_dt, 1e-9)))

                    if avg_speed_norm < velocity_threshold:
                        penalty_raw = (velocity_threshold - avg_speed_norm) / max(velocity_threshold, 1e-9)
                        
                        if velocity_penalty_mode == "quadratic":
                            velocity_penalty = penalty_raw ** 2
                        else:
                            velocity_penalty = penalty_raw
                    else:
                        velocity_penalty = 0.0
                except Exception as exc:
                    self._warn_once("tcom_velocity_v3", f"{exc}")
                    velocity_penalty = 0.0
            reward_velocity = -lambda_velocity_penalty * velocity_penalty

            
            lambda_boundary_penalty = float(getattr(self.cfg, "lambda_boundary_penalty", 1.0))
            boundary_margin = float(getattr(self.cfg, "boundary_margin", 30.0))
            boundary_penalty = 0.0
            if lambda_boundary_penalty > 1e-9:
                try:
                    L = float(self.cfg.L)
                    penalties = []
                    for m in range(self.M):
                        x, y = self.q[m]
                        
                        dist_to_boundary = min(x, y, L - x, L - y)
                        if dist_to_boundary < boundary_margin:
                            penalty = 1.0 - (dist_to_boundary / boundary_margin)
                            penalties.append(penalty)
                        else:
                            penalties.append(0.0)
                    boundary_penalty = float(np.mean(penalties))
                except Exception as exc:
                    self._warn_once("tcom_boundary", f"{exc}")
                    boundary_penalty = 0.0
            reward_boundary = -lambda_boundary_penalty * boundary_penalty

            
            lambda_idle_penalty = float(getattr(self.cfg, "lambda_idle_penalty", 0.5))
            idle_threshold = float(getattr(self.cfg, "idle_threshold", 0.05))
            idle_penalty = 0.0
            if lambda_idle_penalty > 1e-9:
                try:
                    vel_now = np.asarray(delta_q_meters, dtype=np.float64)  # (M, 2)
                    speed_norm = np.linalg.norm(vel_now, axis=1)  # (M,)
                    vmax_dt = float(self.cfg.Vmax) * float(self.cfg.dt)
                    speed_normalized = speed_norm / max(vmax_dt, 1e-9)
                    
                    num_idle = float(np.sum(speed_normalized < idle_threshold))
                    idle_penalty = num_idle / max(float(self.M), 1e-9)
                except Exception as exc:
                    self._warn_once("tcom_idle", f"{exc}")
                    idle_penalty = 0.0
            reward_idle = -lambda_idle_penalty * idle_penalty

            
            reward_survival = 0.0
            reward_safety = reward_constraint
            reward_improve = 0.0
            reward_potential = 0.0
            reward_misc_step = (
                reward_comp_usage + reward_ris_usage + reward_comp_switch +
                reward_velocity + reward_boundary + reward_idle
            )
            reward_total = float(
                reward_paper + reward_constraint + reward_comp_usage + reward_ris_usage +
                reward_comp_switch + reward_velocity + reward_boundary + reward_idle
            )
            reward_raw = reward_total
            reward_performance = reward_paper

            
            reward_vio = reward_constraint
            move_frac_step = velocity_penalty
            traj_user_nn_step = boundary_penalty
            reward_proximity_step = comp_switch_bonus

        elif reward_design == "t6_potential_lagrangian":
            
            
            # r = -(c-c_ref)/sigma + beta*(gamma*Phi(s')-Phi(s)) - lambda*g + eta*r_traj
            paper_cost_policy_raw = float(cost_policy_paper)
            policy_cost_pack = self._t6_harmonized_cost(info_policy)
            paper_cost_policy_step = float(policy_cost_pack["cost"])
            t6_cost_policy_raw_step = float(paper_cost_policy_raw)
            t6_cost_policy_obj_step = float(paper_cost_policy_step)
            t6_delay_norm_step = float(policy_cost_pack["delay_norm"])
            t6_energy_norm_step = float(policy_cost_pack["energy_norm"])
            t6_delay_calib_step = float(policy_cost_pack["delay_calib"])
            t6_energy_calib_step = float(policy_cost_pack["energy_calib"])
            t6_metric_calib_enable_step = float(policy_cost_pack["calib_enable"])
            ref_mode = str(getattr(self.cfg, "t6_ref_cost_mode", "balanced")).strip().lower()
            t6_ref_cost_mode_used = str(ref_mode)

            if ref_mode in relative_base_pack:
                ref_info, _ = relative_base_pack[ref_mode]
            elif ref_mode == "no_comp":
                
                no_comp_pack = _build_base_pack(g, covered, dist2, fair_hierarchical=True)
                ref_info, _ = no_comp_pack["balanced"]
            else:
                self._warn_once("t6_ref_cost_mode", f" t6_ref_cost_mode={ref_mode} balanced")
                t6_ref_cost_mode_used = "balanced"
                ref_info, _ = relative_base_pack["balanced"]
            ref_cost_pack = self._t6_harmonized_cost(ref_info)
            paper_cost_ref_raw = float(ref_info["paper_cost"])
            paper_cost_ref_step = float(ref_cost_pack["cost"])
            t6_cost_ref_raw_step = float(paper_cost_ref_raw)
            t6_cost_ref_obj_step = float(paper_cost_ref_step)

            cov_frac = float(np.clip(float(info_policy.get("coverage_fraction", 0.0)), 0.0, 1.0))
            vio_max_expected = float(max(float(getattr(self.cfg, "vio_max_expected", 5.0)), 1.0))
            vio_ratio = float(np.clip(float(vio_total_penalty) / vio_max_expected, 0.0, 1.0))
            violation_continuous_step = float(np.clip(0.5 * vio_ratio + 0.5 * (1.0 - cov_frac), 0.0, 1.0))
            constraint_signal_step = float(violation_continuous_step)
            vio_any_step = 1.0 if int(vio_total_penalty) > 0 else 0.0

            
            rel_cost_raw_now = float(paper_cost_policy_step - paper_cost_ref_step)
            t6_beta = float(np.clip(float(getattr(self.cfg, "t6_cost_scale_ema_beta", 0.95)), 0.0, 0.9999))
            self._t6_cost_ema_mean, self._t6_cost_ema_var, self._t6_cost_scale = update_ema_mean_var(
                self._t6_cost_ema_mean,
                self._t6_cost_ema_var,
                rel_cost_raw_now,
                beta=t6_beta,
                eps=1e-9,
            )

            
            lambda_v_effective = float(getattr(self.cfg, "lambda_v", 0.0))
            if bool(getattr(self.cfg, "t6_pid_enable", True)) and (self._t6_pid_controller is not None):
                lambda_v_effective = float(self._t6_pid_controller.value)
                self._t6_pid_step += 1
                upd_interval = int(max(int(getattr(self.cfg, "t6_pid_update_interval", 4)), 1))
                if (self._t6_pid_step % upd_interval) == 0:
                    progress = float(np.clip(float(self.t) / float(max(int(self.cfg.T), 1)), 0.0, 1.0))
                    self._lambda_v_dyn = float(
                        self._t6_pid_controller.update(
                            violation_continuous_step,
                            progress=progress,
                        )
                    )
                else:
                    self._lambda_v_dyn = float(self._t6_pid_controller.value)
                t6_violation_ema_step = float(self._t6_pid_controller.violation_ema)
                t6_pid_frozen_step = 1.0 if bool(self._t6_pid_controller.last_update_frozen) else 0.0
            else:
                self._lambda_v_dyn = float(lambda_v_effective)

            
            t6_anneal_ratio_step = float(self._t6_anneal_ratio())
            main_w_s = float(getattr(self.cfg, "t6_main_weight_start", 1.0))
            main_w_e = float(getattr(self.cfg, "t6_main_weight_end", 1.3))
            pot_s = float(getattr(self.cfg, "t6_potential_scale_start", 1.0))
            pot_e = float(getattr(self.cfg, "t6_potential_scale_end", 0.4))
            traj_s = float(getattr(self.cfg, "t6_traj_scale_start", 1.0))
            traj_e = float(getattr(self.cfg, "t6_traj_scale_end", 0.3))
            t6_main_weight_step = float(main_w_s + (main_w_e - main_w_s) * t6_anneal_ratio_step)
            t6_potential_scale_step = float(pot_s + (pot_e - pot_s) * t6_anneal_ratio_step)
            t6_traj_scale_step = float(traj_s + (traj_e - traj_s) * t6_anneal_ratio_step)

            traj_bonus_raw = 0.0
            traj_weight_base = float(
                max(
                    float(getattr(self.cfg, "t6_traj_bonus_weight", getattr(self.cfg, "lambda_traj_guide", 0.0))),
                    0.0,
                )
            )
            traj_weight = float(max(traj_weight_base * max(t6_traj_scale_step, 0.0), 0.0))
            if traj_weight > 0.0 and self.q_prev is not None:
                traj_bonus_raw = self._compute_traj_centroid_bonus(
                    a_mat,
                    guide_weight=1.0,
                    assoc_active_threshold=float(getattr(self.cfg, "traj_serve_threshold", 0.10)),
                )

            beta_potential_base = float(getattr(self.cfg, "t6_beta_potential", 1.0))
            beta_potential_eff = float(max(beta_potential_base * max(t6_potential_scale_step, 0.0), 0.0))
            t6_out = compute_t6_reward(
                policy_cost=paper_cost_policy_step,
                ref_cost=paper_cost_ref_step,
                cost_scale=float(max(self._t6_cost_scale, 1e-9)),
                potential_prev=float(prev_potential),
                potential_curr=float(current_potential),
                gamma=float(np.clip(float(getattr(self.cfg, "t6_potential_gamma", 0.99)), 0.0, 1.0)),
                beta_potential=float(beta_potential_eff),
                lambda_value=float(lambda_v_effective),
                violation_continuous=float(violation_continuous_step),
                traj_bonus=float(traj_bonus_raw),
                traj_weight=float(traj_weight),
                main_weight=float(t6_main_weight_step),
            )

            reward_total = float(t6_out["reward_total"])
            reward_raw = float(reward_total)
            reward_main_step = float(t6_out["reward_main"])
            reward_traj_step = float(t6_out["reward_traj"])
            reward_paper_step = float(reward_main_step)
            reward_constraint_step = float(t6_out["reward_constraint"])
            reward_potential_step = float(t6_out["reward_potential"])
            reward_shaping_step = float(reward_potential_step)
            reward_misc_step = float(reward_traj_step)
            reward_survival = 0.0
            reward_safety = float(reward_constraint_step)
            reward_performance = float(reward_main_step)
            reward_improve = float(reward_traj_step)
            reward_potential = float(reward_potential_step)
            reward_vio = float(reward_constraint_step)
            paper_cost_step = float(paper_cost_policy_raw)
            paper_cost_obj_step = float(paper_cost_policy_step)
            paper_delta_step = float(t6_out["relative_cost_raw"])
            paper_delta_norm_step = float(t6_out["relative_cost_norm"])
            move_frac_step = 0.0
            traj_user_nn_step = 0.0
            reward_proximity_step = 0.0
            lambda_v_effective = float(t6_out["lambda_v_effective"])

        elif reward_design == "t7_absolute":
            
            
            #   r_t = -alpha * (c_t / c_ref) - lambda_t * g_t
            
            
            
            
            paper_cost_policy_step = float(cost_policy_paper)
            t7_alpha = float(max(float(getattr(self.cfg, "t7_alpha", 2.0)), 0.0))
            t7_cost_ref = float(max(float(getattr(self.cfg, "t7_cost_ref", 1.0)), 1e-9))
            if t7_cost_ref <= 1e-8:
                self._warn_once("t7_cost_ref_too_small", "t7_cost_ref 1.0")
                t7_cost_ref = 1.0
            t7_alpha_step = float(t7_alpha)
            t7_cost_ref_step = float(t7_cost_ref)

            cov_frac = float(np.clip(float(info_policy.get("coverage_fraction", 0.0)), 0.0, 1.0))
            vio_max_expected = float(max(float(getattr(self.cfg, "vio_max_expected", 5.0)), 1.0))
            vio_ratio = float(np.clip(float(vio_total_penalty) / vio_max_expected, 0.0, 1.0))
            violation_continuous_step = float(np.clip(0.5 * vio_ratio + 0.5 * (1.0 - cov_frac), 0.0, 1.0))
            constraint_signal_step = float(violation_continuous_step)
            vio_any_step = 1.0 if int(vio_total_penalty) > 0 else 0.0

            lambda_v_effective = float(getattr(self.cfg, "lambda_v", 0.0))
            if bool(getattr(self.cfg, "t6_pid_enable", True)) and (self._t6_pid_controller is not None):
                lambda_v_effective = float(self._t6_pid_controller.value)
                self._t6_pid_step += 1
                upd_interval = int(max(int(getattr(self.cfg, "t6_pid_update_interval", 2)), 1))
                if (self._t6_pid_step % upd_interval) == 0:
                    progress = float(np.clip(float(self.t) / float(max(int(self.cfg.T), 1)), 0.0, 1.0))
                    self._lambda_v_dyn = float(
                        self._t6_pid_controller.update(
                            violation_continuous_step,
                            progress=progress,
                        )
                    )
                else:
                    self._lambda_v_dyn = float(self._t6_pid_controller.value)
                t6_violation_ema_step = float(self._t6_pid_controller.violation_ema)
                t6_pid_frozen_step = 1.0 if bool(self._t6_pid_controller.last_update_frozen) else 0.0
            else:
                self._lambda_v_dyn = float(lambda_v_effective)

            
            t7_cost_transform = str(getattr(self.cfg, "t7_cost_transform", "linear")).strip().lower()
            cost_ratio = float(paper_cost_policy_step / max(t7_cost_ref, 1e-9))
            if t7_cost_transform == "log":
                
                reward_main_step = float(-t7_alpha * np.log(max(cost_ratio, 1e-9)))
            else:
                
                reward_main_step = float(-t7_alpha * cost_ratio)
            reward_constraint_step = float(-lambda_v_effective * violation_continuous_step)
            reward_total = float(reward_main_step + reward_constraint_step)
            reward_raw = float(reward_total)

            reward_paper_step = float(reward_main_step)
            reward_paper_abs_step = float(reward_main_step)
            reward_survival = 0.0
            reward_safety = float(reward_constraint_step)
            reward_performance = float(reward_main_step)
            reward_improve = 0.0
            reward_potential = 0.0
            reward_potential_step = 0.0
            reward_shaping_step = 0.0
            reward_misc_step = 0.0
            reward_vio = float(reward_constraint_step)

            paper_cost_step = float(paper_cost_policy_step)
            paper_cost_obj_step = float(paper_cost_policy_step)
            paper_cost_ref_step = float(t7_cost_ref)
            paper_delta_step = float(paper_cost_policy_step - t7_cost_ref)
            paper_delta_norm_step = float((paper_cost_policy_step - t7_cost_ref) / max(t7_cost_ref, 1e-9))
            t6_ref_cost_mode_used = "t7_absolute_ref"
            move_frac_step = 0.0
            traj_user_nn_step = 0.0
            reward_proximity_step = 0.0

        elif reward_design == "z_shifted_log":
            
            
            #   r_t = -  log(C_eff / C_ref) + 
            #   C_eff = paper_cost +   violation_continuous
            
            paper_cost_policy_step = float(cost_policy_paper)
            z_alpha = float(max(float(getattr(self.cfg, "z_alpha", 3.0)), 0.0))
            z_beta = float(getattr(self.cfg, "z_beta", 3.5))
            z_gamma = float(max(float(getattr(self.cfg, "z_gamma", 1.2)), 0.0))
            z_cost_ref, _z_cost_ref_from_load_map, _z_cost_ref_from_speed_map = self._resolve_z_cost_ref_with_speed()
            z_cost_ref_runtime = float(z_cost_ref)
            z_cost_ref_from_load_map = 1.0 if bool(_z_cost_ref_from_load_map) else 0.0
            z_cost_ref_from_speed_map = 1.0 if bool(_z_cost_ref_from_speed_map) else 0.0
            z_cost_ref_from_map = 1.0 if (bool(_z_cost_ref_from_load_map) or bool(_z_cost_ref_from_speed_map)) else 0.0

            
            cov_frac = float(np.clip(float(info_policy.get("coverage_fraction", 0.0)), 0.0, 1.0))
            vio_max_expected = float(max(float(getattr(self.cfg, "vio_max_expected", 5.0)), 1.0))
            vio_ratio = float(np.clip(float(vio_total_penalty) / vio_max_expected, 0.0, 1.0))
            violation_continuous_step = float(np.clip(vio_ratio, 0.0, 1.0))
            constraint_signal_step = float(violation_continuous_step)
            vio_any_step = 1.0 if int(vio_total_penalty) > 0 else 0.0

            
            c_eff = float(paper_cost_policy_step + z_gamma * violation_continuous_step)
            cost_ratio = float(c_eff / z_cost_ref)

            
            reward_main_step = float(-z_alpha * np.log(max(cost_ratio, 1e-9)) + z_beta)
            reward_constraint_step = 0.0  
            
            lambda_v_effective = 0.0

            reward_paper_step = float(reward_main_step)
            reward_paper_abs_step = float(reward_main_step)
            reward_proximity_weight_effective_step = float(self._current_reward_proximity_weight())
            reward_proximity_step = float(self._compute_reward_proximity_step(dist2))
            reward_coverage_floor_step = float(self._compute_coverage_floor_penalty(cov_frac))
            reward_misc_step = float(reward_proximity_step + reward_coverage_floor_step)
            reward_total = float(reward_main_step + reward_misc_step)
            reward_raw = float(reward_total)

            reward_survival = float(reward_misc_step)
            reward_safety = 0.0
            reward_performance = float(reward_main_step)
            reward_improve = 0.0
            reward_potential = 0.0
            reward_potential_step = 0.0
            reward_shaping_step = 0.0
            reward_vio = 0.0

            paper_cost_step = float(paper_cost_policy_step)
            paper_cost_obj_step = float(paper_cost_policy_step)
            paper_cost_ref_step = float(z_cost_ref)
            paper_delta_step = float(paper_cost_policy_step - z_cost_ref)
            paper_delta_norm_step = float((paper_cost_policy_step - z_cost_ref) / max(z_cost_ref, 1e-9))
            t6_ref_cost_mode_used = "z_shifted_log_ref"
            move_frac_step = 0.0
            traj_user_nn_step = 0.0
            # ===== [END Phase Z] =====

        elif reward_design == "z4_linear":
            
            
            
            
            paper_cost_policy_step = float(cost_policy_paper)
            z_cost_ref, _z_cost_ref_from_load_map, _z_cost_ref_from_speed_map = self._resolve_z_cost_ref_with_speed()
            z_cost_ref_runtime = float(z_cost_ref)
            z_cost_ref_from_load_map = 1.0 if bool(_z_cost_ref_from_load_map) else 0.0
            z_cost_ref_from_speed_map = 1.0 if bool(_z_cost_ref_from_speed_map) else 0.0
            z_cost_ref_from_map = 1.0 if (bool(_z_cost_ref_from_load_map) or bool(_z_cost_ref_from_speed_map)) else 0.0
            z4_offset = float(getattr(self.cfg, "z4_reward_offset", 1.75))
            z4_alpha = float(getattr(self.cfg, "z4_reward_alpha", 3.0))

            
            reward_linear_step = float(z4_alpha * (z4_offset - paper_cost_policy_step / z_cost_ref))

            # progressive bonus
            z4_gamma = float(getattr(self.cfg, "z4_bonus_gamma", 100.0))
            z4_anchor = float(getattr(self.cfg, "z4_bonus_anchor", 1.18))
            z4_power = float(getattr(self.cfg, "z4_bonus_power", 2.0))
            z4_deadzone = float(np.clip(float(getattr(self.cfg, "z4_bonus_deadzone", 0.0)), 0.0, 0.95))
            anchor_cost = z_cost_ref * z4_anchor
            impr_ratio = max(0.0, (anchor_cost - paper_cost_policy_step) / max(anchor_cost, 1e-9))
            
            if impr_ratio <= z4_deadzone:
                impr_ratio_eff = 0.0
            else:
                impr_ratio_eff = float((impr_ratio - z4_deadzone) / max(1.0 - z4_deadzone, 1e-9))
            reward_bonus_step = float(z4_gamma * (impr_ratio_eff ** z4_power))
            reward_main_no_gap = float(reward_linear_step + reward_bonus_step)

            
            z4_gap_cost_policy_step = float(paper_cost_policy_step)
            z4_gap_cost_balanced_step = float(base_pack["balanced"][0]["paper_cost"])
            if not np.isfinite(z4_gap_cost_balanced_step):
                z4_gap_cost_balanced_step = float(z4_gap_cost_policy_step)
            z4_gap_raw_step = float(
                (z4_gap_cost_balanced_step - z4_gap_cost_policy_step)
                / max(abs(z4_gap_cost_balanced_step), 1e-9)
            )
            if not np.isfinite(z4_gap_raw_step):
                z4_gap_raw_step = 0.0
            z4_gap_clip_cfg = float(max(float(getattr(self.cfg, "z4_gap_clip", 0.30)), 0.0))
            if z4_gap_clip_cfg > 0.0:
                z4_gap_step_clip = float(np.clip(z4_gap_raw_step, -z4_gap_clip_cfg, z4_gap_clip_cfg))
            else:
                z4_gap_step_clip = float(z4_gap_raw_step)

            
            z4_gap_ema_enable = bool(getattr(self.cfg, "z4_gap_ema_enable", True))
            z4_gap_ema_beta = float(np.clip(float(getattr(self.cfg, "z4_gap_ema_beta", 0.85)), 0.0, 0.99))
            if z4_gap_ema_enable:
                if (not bool(getattr(self, "_z4_gap_ema_ready", False))) or (not np.isfinite(float(getattr(self, "_z4_gap_ema", 0.0)))):
                    self._z4_gap_ema = float(z4_gap_step_clip)
                    self._z4_gap_ema_ready = True
                else:
                    self._z4_gap_ema = float(
                        z4_gap_ema_beta * float(self._z4_gap_ema)
                        + (1.0 - z4_gap_ema_beta) * float(z4_gap_step_clip)
                    )
                z4_gap_step_ema = float(self._z4_gap_ema)
            else:
                z4_gap_step_ema = float(z4_gap_step_clip)
                self._z4_gap_ema = float(z4_gap_step_ema)
                self._z4_gap_ema_ready = True
            if z4_gap_clip_cfg > 0.0:
                z4_gap_step = float(np.clip(z4_gap_step_ema, -z4_gap_clip_cfg, z4_gap_clip_cfg))
            else:
                z4_gap_step = float(z4_gap_step_ema)

            z4_gap_enable = bool(getattr(self.cfg, "z4_gap_enable", False))
            z4_gap_lambda_input = float(max(float(getattr(self.cfg, "z4_gap_lambda", 0.0)), 0.0))
            z4_gap_lambda_sched, z4_gap_train_frac = self._z4_gap_lambda_schedule(z4_gap_lambda_input)

            
            z4_gap_grad_main = float(abs(z4_alpha) / max(z_cost_ref, 1e-9))
            z4_gap_ratio_max = float(np.clip(float(getattr(self.cfg, "z4_gap_grad_ratio_max", 0.25)), 0.0, 1.0))
            z4_gap_lambda_cap = float(
                z4_gap_ratio_max * z4_gap_grad_main * max(abs(z4_gap_cost_balanced_step), 1e-9)
            )
            if not z4_gap_enable:
                z4_gap_lambda_eff = 0.0
            else:
                z4_gap_lambda_eff = float(min(z4_gap_lambda_sched, z4_gap_lambda_cap))
            z4_gap_reward_step = float(z4_gap_lambda_eff * z4_gap_step)

            
            shape_guard_enable = bool(getattr(self.cfg, "z4_gap_shape_guard_enable", True))
            shape_guard_min_steps = int(max(int(getattr(self.cfg, "z4_gap_shape_guard_min_steps", 2000)), 0))
            shape_guard_tol = float(max(float(getattr(self.cfg, "z4_gap_shape_guard_tol", 0.01)), 0.0))
            shape_guard_patience = int(max(int(getattr(self.cfg, "z4_gap_shape_guard_patience", 6)), 1))
            z4_gap_shape_guard_trigger = 0.0

            main_with_gap_candidate = float(reward_main_no_gap + z4_gap_reward_step)
            prev_main_no_gap = getattr(self, "_z4_gap_prev_main_no_gap", None)
            prev_main_with_gap = getattr(self, "_z4_gap_prev_main_with_gap", None)
            z4_gap_shape_slope_base = 0.0
            z4_gap_shape_slope_with = 0.0
            if (
                shape_guard_enable
                and z4_gap_lambda_eff > 0.0
                and self._train_step >= shape_guard_min_steps
                and prev_main_no_gap is not None
                and prev_main_with_gap is not None
                and np.isfinite(float(prev_main_no_gap))
                and np.isfinite(float(prev_main_with_gap))
            ):
                z4_gap_shape_slope_base = float(reward_main_no_gap - float(prev_main_no_gap))
                z4_gap_shape_slope_with = float(main_with_gap_candidate - float(prev_main_with_gap))
                if z4_gap_shape_slope_with < (z4_gap_shape_slope_base - shape_guard_tol):
                    self._z4_gap_shape_bad_streak = int(self._z4_gap_shape_bad_streak) + 1
                else:
                    self._z4_gap_shape_bad_streak = max(int(self._z4_gap_shape_bad_streak) - 1, 0)
                if int(self._z4_gap_shape_bad_streak) >= shape_guard_patience:
                    z4_gap_shape_guard_trigger = 1.0
                    z4_gap_lambda_eff = 0.0
                    z4_gap_reward_step = 0.0
                    self._z4_gap_shape_bad_streak = shape_guard_patience
            else:
                if (not z4_gap_enable) or (not shape_guard_enable):
                    self._z4_gap_shape_bad_streak = 0
                else:
                    self._z4_gap_shape_bad_streak = max(int(getattr(self, "_z4_gap_shape_bad_streak", 0)), 0)

            z4_gap_shape_bad_streak = float(int(getattr(self, "_z4_gap_shape_bad_streak", 0)))
            z4_gap_grad_ratio_eff = float(
                (z4_gap_lambda_eff / max(abs(z4_gap_cost_balanced_step), 1e-9))
                / max(z4_gap_grad_main, 1e-9)
            )
            reward_main_step = float(reward_main_no_gap + z4_gap_reward_step)
            self._z4_gap_prev_main_no_gap = float(reward_main_no_gap)
            self._z4_gap_prev_main_with_gap = float(reward_main_step)
            reward_constraint_step = 0.0
            lambda_v_effective = 0.0

            
            cov_frac = float(np.clip(float(info_policy.get("coverage_fraction", 0.0)), 0.0, 1.0))
            reward_coverage_floor_step = float(self._compute_coverage_floor_penalty(cov_frac))

            
            violation_continuous_step = 0.0
            constraint_signal_step = 0.0
            vio_any_step = 0.0

            reward_proximity_weight_effective_step = 0.0
            reward_proximity_step = 0.0
            reward_misc_step = float(reward_coverage_floor_step)
            reward_total = float(reward_main_step + reward_misc_step)
            reward_raw = float(reward_total)

            reward_paper_step = float(reward_main_step)
            reward_paper_abs_step = float(reward_main_step)
            reward_survival = float(reward_misc_step)
            reward_safety = 0.0
            reward_performance = float(reward_main_step)
            reward_improve = 0.0
            reward_potential = 0.0
            reward_potential_step = 0.0
            reward_shaping_step = 0.0
            reward_vio = 0.0

            paper_cost_step = float(paper_cost_policy_step)
            paper_cost_obj_step = float(paper_cost_policy_step)
            paper_cost_ref_step = float(z_cost_ref)
            paper_delta_step = float(paper_cost_policy_step - z_cost_ref)
            paper_delta_norm_step = float((paper_cost_policy_step - z_cost_ref) / max(z_cost_ref, 1e-9))
            t6_ref_cost_mode_used = "z4_linear_ref"
            move_frac_step = 0.0
            traj_user_nn_step = 0.0
            # ===== [END Phase Z4] =====

        elif reward_design == "z5_saturating":
            
            paper_cost_policy_step = float(cost_policy_paper)
            z_cost_ref, _z_cost_ref_from_load_map, _z_cost_ref_from_speed_map = self._resolve_z_cost_ref_with_speed()
            z_cost_ref_runtime = float(z_cost_ref)
            z_cost_ref_from_load_map = 1.0 if bool(_z_cost_ref_from_load_map) else 0.0
            z_cost_ref_from_speed_map = 1.0 if bool(_z_cost_ref_from_speed_map) else 0.0
            z_cost_ref_from_map = 1.0 if (bool(_z_cost_ref_from_load_map) or bool(_z_cost_ref_from_speed_map)) else 0.0

            z5_r_max = float(max(float(getattr(self.cfg, "z5_r_max", 2.0)), 0.0))
            z5_kappa = float(max(float(getattr(self.cfg, "z5_kappa", 15.0)), 0.0))
            z5_anchor = float(getattr(self.cfg, "z5_anchor_norm", 1.15))
            cost_ratio = float(paper_cost_policy_step / max(z_cost_ref, 1e-9))
            z5_delta = float(z5_anchor - cost_ratio)
            reward_main_step = float(z5_r_max * np.tanh(z5_kappa * z5_delta))
            reward_constraint_step = 0.0
            lambda_v_effective = 0.0

            cov_frac = float(np.clip(float(info_policy.get("coverage_fraction", 0.0)), 0.0, 1.0))
            vio_max_expected = float(max(float(getattr(self.cfg, "vio_max_expected", 5.0)), 1.0))
            vio_ratio = float(np.clip(float(vio_total_penalty) / vio_max_expected, 0.0, 1.0))
            violation_continuous_step = float(np.clip(vio_ratio, 0.0, 1.0))
            constraint_signal_step = float(violation_continuous_step)
            vio_any_step = 1.0 if int(vio_total_penalty) > 0 else 0.0

            reward_paper_step = float(reward_main_step)
            reward_paper_abs_step = float(reward_main_step)
            reward_proximity_weight_effective_step = float(self._current_reward_proximity_weight())
            reward_proximity_step = float(self._compute_reward_proximity_step(dist2))
            reward_coverage_floor_step = float(self._compute_coverage_floor_penalty(cov_frac))
            reward_misc_step = float(reward_proximity_step + reward_coverage_floor_step)
            reward_total = float(reward_main_step + reward_misc_step)
            reward_raw = float(reward_total)

            reward_survival = float(reward_misc_step)
            reward_safety = 0.0
            reward_performance = float(reward_main_step)
            reward_improve = 0.0
            reward_potential = 0.0
            reward_potential_step = 0.0
            reward_shaping_step = 0.0
            reward_vio = 0.0

            paper_cost_step = float(paper_cost_policy_step)
            paper_cost_obj_step = float(paper_cost_policy_step)
            paper_cost_ref_step = float(z_cost_ref)
            paper_delta_step = float(paper_cost_policy_step - z_cost_ref)
            paper_delta_norm_step = float((paper_cost_policy_step - z_cost_ref) / max(z_cost_ref, 1e-9))
            t6_ref_cost_mode_used = "z5_saturating_ref"
            move_frac_step = 0.0
            traj_user_nn_step = 0.0
            # ===== [END Phase Z5] =====

        elif reward_design == "tcom_relative_v1":
            
            
            
            
            #
            
            
            
            

            
            paper_cost_weight = float(getattr(self.cfg, "paper_cost_weight", 1.0))
            paper_cost_policy_step = float(cost_policy_paper)
            baseline_mode = relative_baseline_mode_cfg
            info_bl, _vio_bl = relative_base_pack[baseline_mode]
            paper_cost_baseline_step = float(info_bl["paper_cost"])

            
            relative_paper_cost = paper_cost_policy_step - paper_cost_baseline_step
            reward_paper_rel = -paper_cost_weight * relative_paper_cost

            
            
            
            paper_cost_abs_weight = float(getattr(self.cfg, "paper_cost_abs_weight", 0.0))
            reward_paper_abs = -paper_cost_abs_weight * paper_cost_policy_step if paper_cost_abs_weight > 0.0 else 0.0
            reward_paper = reward_paper_rel + reward_paper_abs

            
            
            
            lambda_v = float(getattr(self.cfg, "lambda_v", 3.0))
            vio_max_expected = float(max(float(getattr(self.cfg, "vio_max_expected", 5.0)), 1.0))
            vio_continuous = float(min(float(vio_total_penalty) / vio_max_expected, 1.0))
            reward_constraint = -lambda_v * vio_continuous

            
            reward_survival = 0.0
            reward_safety = reward_constraint

            
            
            
            lambda_traj_guide = float(getattr(self.cfg, "lambda_traj_guide", 0.0))
            traj_guide_bonus = self._compute_traj_centroid_bonus(
                a_mat,
                guide_weight=float(lambda_traj_guide),
                assoc_active_threshold=float(getattr(self.cfg, "traj_serve_threshold", 0.10)),
            )

            reward_improve = traj_guide_bonus
            reward_potential = 0.0
            reward_misc_step = 0.0
            reward_total = float(reward_paper + reward_constraint + traj_guide_bonus)
            reward_raw = reward_total
            reward_performance = reward_paper

            
            reward_vio = reward_constraint
            move_frac_step = 0.0
            traj_user_nn_step = 0.0
            reward_proximity_step = 0.0
            
            reward_paper_step = reward_paper
            reward_paper_abs_step = reward_paper_abs  
            reward_constraint_step = reward_constraint
            
            paper_cost_step = paper_cost_policy_step
            paper_cost_ref_step = paper_cost_baseline_step
            paper_delta_step = relative_paper_cost

        elif reward_design == "tcom_improvement_v1":
            
            
            
            #
            
            
            
            
            

            
            paper_cost_policy_step = float(cost_policy_paper)

            if self.cost_prev is not None and self.t > 0:
                
                improvement_step = (self.cost_prev - paper_cost_policy_step) / max(self.cost_prev, 1e-6)
                reward_improvement_step = self.improvement_step_weight * float(np.clip(improvement_step, -1.0, 1.0))
            else:
                reward_improvement_step = 0.0

            
            lambda_v = float(getattr(self.cfg, "lambda_v", 2.0))
            vio_max_expected = float(max(float(getattr(self.cfg, "vio_max_expected", 5.0)), 1.0))
            vio_continuous = float(min(float(vio_total_penalty) / vio_max_expected, 1.0))
            reward_constraint = -lambda_v * vio_continuous

            
            lambda_traj_guide = float(getattr(self.cfg, "lambda_traj_guide", 0.0))
            traj_guide_bonus = self._compute_traj_centroid_bonus(
                a_mat,
                guide_weight=float(lambda_traj_guide),
                assoc_active_threshold=float(getattr(self.cfg, "traj_serve_threshold", 0.10)),
            )

            # ---- Step reward ----
            reward_total = float(reward_improvement_step + reward_constraint + traj_guide_bonus)
            reward_raw = reward_total
            reward_performance = reward_improvement_step
            reward_survival = 0.0
            reward_safety = reward_constraint
            reward_improve = traj_guide_bonus
            reward_potential = 0.0
            reward_misc_step = 0.0

            
            self.episode_cost_sum += paper_cost_policy_step
            self.episode_steps += 1

            
            self.cost_prev = paper_cost_policy_step

            
            reward_vio = reward_constraint
            move_frac_step = 0.0
            traj_user_nn_step = 0.0
            reward_proximity_step = 0.0
            reward_paper_step = reward_improvement_step
            reward_paper_abs_step = 0.0
            reward_constraint_step = reward_constraint
            paper_cost_step = paper_cost_policy_step
            paper_cost_ref_step = 0.0  
            paper_delta_step = 0.0

        else:
            # ===== [ ] + =====
            REWARD_SCALE = 10.0

            reward_survival = 0.2 * REWARD_SCALE
            if vio_total_penalty == 0:
                reward_safety = 0.8 * REWARD_SCALE
            else:
                violation_scale = float(getattr(self.cfg, "violation_penalty_scale", 0.1))
                reward_safety = -0.1 * violation_scale * REWARD_SCALE * float(vio_total_penalty)

            reward_performance = -float(cost_policy) * REWARD_SCALE
            reward_improve = float(self.cfg.improve_alpha) * float(cost_base - cost_policy) * REWARD_SCALE

            POTENTIAL_WEIGHT = 0.3 * REWARD_SCALE
            reward_potential = POTENTIAL_WEIGHT * potential_bonus

            reward_total = reward_survival + reward_safety + reward_performance + reward_improve + reward_potential
            reward_raw = reward_total
        # ===== [END =====

        
        reward, reward_scaled, reward_clipped = self._postprocess_reward(float(reward_raw))

        done = self.t >= self.cfg.T

        
        if not done:
            self._ensure_task_cache(
                expected_slot=int(self.t) + 1,
                context="step_post",
                force_resample=True,
            )

        
        
        reward_components = {
            "survival": float(reward_survival),
            "safety": float(reward_safety),
            "performance": float(reward_performance),
            "improve": float(reward_improve),
            "potential": float(reward_potential),
            
            "paper_cost_step": float(paper_cost_step),
            "paper_cost_ref_step": float(paper_cost_ref_step),
            "paper_delta_step": float(paper_delta_step),
            "paper_delta_norm_step": float(paper_delta_norm_step),
            "paper_adv_step": float(paper_adv_step),
            "paper_adv_norm_step": float(paper_adv_norm_step),
            "reward_paper_step": float(reward_paper_step),
            "reward_main_step": float(reward_main_step),
            "z4_gap_cost_policy_step": float(z4_gap_cost_policy_step),
            "z4_gap_cost_balanced_step": float(z4_gap_cost_balanced_step),
            "z4_gap_raw_step": float(z4_gap_raw_step),
            "z4_gap_step_clip": float(z4_gap_step_clip),
            "z4_gap_step_ema": float(z4_gap_step_ema),
            "z4_gap_step": float(z4_gap_step),
            "z4_gap_reward_step": float(z4_gap_reward_step),
            "z4_gap_lambda_input": float(z4_gap_lambda_input),
            "z4_gap_lambda_sched": float(z4_gap_lambda_sched),
            "z4_gap_lambda_cap": float(z4_gap_lambda_cap),
            "z4_gap_lambda_eff": float(z4_gap_lambda_eff),
            "z4_gap_grad_main": float(z4_gap_grad_main),
            "z4_gap_grad_ratio_eff": float(z4_gap_grad_ratio_eff),
            "z4_gap_train_frac": float(z4_gap_train_frac),
            "z4_gap_shape_guard_trigger": float(z4_gap_shape_guard_trigger),
            "z4_gap_shape_bad_streak": float(z4_gap_shape_bad_streak),
            "z4_gap_shape_slope_base": float(z4_gap_shape_slope_base),
            "z4_gap_shape_slope_with": float(z4_gap_shape_slope_with),
            "z4_comp_meta_enable": float(z4_comp_meta_enable_step),
            "z4_comp_meta_ctrl_thr_raw": float(z4_comp_meta_ctrl_thr_raw),
            "z4_comp_meta_ctrl_temp_raw": float(z4_comp_meta_ctrl_temp_raw),
            "z4_comp_meta_warm_scale": float(z4_comp_meta_warm_scale),
            "z4_comp_meta_train_frac": float(z4_comp_meta_train_frac),
            "z4_comp_meta_thr_base": float(z4_comp_meta_thr_base),
            "z4_comp_meta_thr_delta_raw": float(z4_comp_meta_thr_delta_raw),
            "z4_comp_meta_thr_delta_ema": float(z4_comp_meta_thr_delta_ema),
            "z4_comp_meta_thr_delta_eff": float(z4_comp_meta_thr_delta_eff),
            "z4_comp_meta_thr_effective": float(z4_comp_meta_thr_effective),
            "z4_comp_meta_score_width": float(z4_comp_meta_score_width),
            "z4_comp_meta_temp_base": float(z4_comp_meta_temp_base),
            "z4_comp_meta_temp_scale_raw": float(z4_comp_meta_temp_scale_raw),
            "z4_comp_meta_temp_scale_ema": float(z4_comp_meta_temp_scale_ema),
            "z4_comp_meta_temp_scale_eff": float(z4_comp_meta_temp_scale_eff),
            "z4_comp_meta_temp_effective": float(z4_comp_meta_temp_effective),
            "z4_assoc_stage_enable": float(z4_assoc_stage_enable_step),
            "z4_assoc_policy_mix": float(z4_assoc_policy_mix_step),
            "z4_assoc_train_frac": float(z4_assoc_train_frac_step),
            "z4_assoc_stage_score_width": float(z4_assoc_stage_score_width),
            "z4_assoc_stage_comp_rule_thr": float(z4_assoc_stage_comp_rule_thr),
            "t7_alpha_step": float(t7_alpha_step),
            "t7_cost_ref_step": float(t7_cost_ref_step),
            "reward_paper_abs_step": float(reward_paper_abs_step),
            "reward_paper_delta_step": float(reward_paper_delta_step),
            "reward_paper_adv_step": float(reward_paper_adv_step),
            "reward_constraint_step": float(reward_constraint_step),
            "constraint_signal_step": float(constraint_signal_step),
            "violation_continuous": float(violation_continuous_step),
            "lambda_v_effective": float(lambda_v_effective),
            "lambda_v_next": float(self._lambda_v_dyn),
            "vio_any_step": float(vio_any_step),
            "reward_comp_step": float(reward_comp_step),
            "reward_ris_step": float(reward_ris_step),
            "reward_potential_step": float(reward_potential_step),
            "reward_traj_step": float(reward_traj_step),
            "reward_shaping_step": float(reward_shaping_step),
            "reward_misc_step": float(reward_misc_step),
            "reward_proximity_step": float(reward_proximity_step),
            "reward_proximity_weight_effective_step": float(reward_proximity_weight_effective_step),
            "reward_coverage_floor_step": float(reward_coverage_floor_step),
            "shaping_gate": float(shaping_gate),
            "beta_comp_t": float(beta_comp_t),
            "beta_ris_t": float(beta_ris_t),
            "beta_train_frac": float(beta_train_frac),
            "comp_bonus_raw_step": float(comp_bonus_raw_step),
            "ris_bonus_raw_step": float(ris_bonus_raw_step),
            "ris_quality_gate": float(ris_quality_gate),
            
            "vio": float(reward_vio),
            "wall": float(reward_wall),
            "wall_margin": float(wall_margin),
            "wall_margin_term": float(reward_wall_margin),
            "safe_clip_projection_step": float(safe_clip_projection_step),
            "safe_clip_term": float(reward_safe_clip),
            "uncovered": float(reward_uncovered),
            "coverage_margin": float(reward_coverage_margin),
            "collision_margin": float(reward_collision_margin),
            "movement": float(reward_movement),
            "movement_frac_step": float(move_frac_step),
            "traj_boundary_stick_step": float(traj_boundary_stick_step),
            "boundary_stick_term": float(reward_boundary_stick),
            "traj_user_nn_step": float(traj_user_nn_step),
            "traj_centroid_gap_step": float(traj_centroid_gap_step),
            "traj_switchback_step": float(traj_switchback_step),
            "user_nn_term": float(reward_user_nn),
            "user_centroid_term": float(reward_user_centroid),
            "switchback_term": float(reward_switchback),
            "traj_direct_term": float(reward_traj_direct),
            "traj_gain_coupling_term": float(reward_traj_gain_coupling),
            "constraint_bonus_spill_term": float(reward_constraint_bonus_spill),
            "phaseh_aux_gate": float(phaseh_aux_gate),
            "phaseh_aux_term_traj": float(phaseh_aux_term_traj),
            "phaseh_aux_term_safety": float(phaseh_aux_term_safety),
            "phaseh_boundary_stick_ema": float(getattr(self, "_boundary_stick_ema", 0.0)),
            "phaseh_wall_safe_margin": float(self._wall_safe_margin()),
            "shaping_gate_hard": float(shaping_gate_hard),
            "shaping_gate_soft": float(shaping_gate_soft),
            "phaseh_comp_bonus_scale": float(comp_bonus_scale),
            "phaseh_ris_bonus_scale": float(ris_bonus_scale),
            "penalty_scale": float(penalty_scale),
            "vio_rate_step": float(vio_rate_step),
            "paper_cost_term": float(-float(getattr(self.cfg, "reward_paper_cost_weight", 0.0)) * float(cost_policy_paper)),
            "raw": float(reward_raw),
            "scaled": float(reward_scaled),
            "clipped": float(reward_clipped),
            "ema": float(self._reward_ema),
            "z_cost_ref_runtime": float(z_cost_ref_runtime),
            "z_cost_ref_from_map": float(z_cost_ref_from_map),
            "z_cost_ref_from_load_map": float(z_cost_ref_from_load_map),
            "z_cost_ref_from_speed_map": float(z_cost_ref_from_speed_map),
            "total": float(reward_total),
        }

        
        violation_breakdown = {
            "collisions_pre": int(vio_coll_pre),
            "collisions_post": int(vio_coll_post),
            "wall": int(vio_wall),
            "no_cover": int(vio_no_cover),
            "theta_min": int(vio_theta),
            "theta_infeasible": int(theta_infeasible),
        }

        info = {
            **info_policy,
            "t": int(self.t),
            "load_scale": float(self.cfg.load_scale),
            "noise_bucket_id": int(getattr(self, "_noise_bucket_id", -1)),
            "noise_bucket_multiplier": float(getattr(self, "_noise_bucket_multiplier", 1.0)),
            "t6_edge_user_ratio_cfg": float(getattr(self.cfg, "t6_edge_user_ratio", 0.0)),
            "t6_edge_user_ratio_eff": float(self._eff_t6_edge_user_ratio()),
            "t6_edge_band_frac_cfg": float(getattr(self.cfg, "t6_edge_band_frac", 0.15)),
            "t6_direct_blockage_prob_cfg": float(getattr(self.cfg, "t6_direct_blockage_prob", 0.0)),
            "t6_direct_blockage_prob_eff": float(self._eff_t6_direct_blockage_prob()),
            "t6_direct_blockage_mean": float(np.mean(np.asarray(getattr(self, "_t6_direct_blockage_map", np.ones((1, 1))), dtype=np.float64))),
            "t6_interference_jitter_eff": float(self._eff_t6_interference_jitter()),
            "user_position_jitter_eff": float(self._eff_user_position_jitter()),
            "task_arrival_jitter_eff": float(self._eff_task_arrival_jitter()),
            "task_size_jitter_eff": float(self._eff_task_size_jitter()),
            "user_mobility_mode": str(getattr(self.cfg, "user_mobility_mode", "static")),
            "user_speed_sampling_scope": str(getattr(self.cfg, "user_speed_sampling_scope", "episode")),
            "user_mobility_active": int(bool(self._user_mobility_active)),
            "user_speed_eff": float(self._user_speed_eff),
            "user_speed_level_idx": int(getattr(self, "_user_speed_level_idx", -1)),
            "user_mobility_ratio": float(
                np.mean(np.asarray(getattr(self, "_user_mobility_mask", np.zeros((0,), dtype=bool)), dtype=np.float64))
            ) if getattr(self, "_user_mobility_mask", None) is not None else 0.0,
            "user_speed_eff_mean": float(
                np.mean(np.asarray(getattr(self, "_user_speed_eff_vec", np.zeros((0,), dtype=np.float64)), dtype=np.float64))
            ) if getattr(self, "_user_speed_eff_vec", None) is not None else float(self._user_speed_eff),
            "user_speed_eff_p90": float(
                np.percentile(np.asarray(getattr(self, "_user_speed_eff_vec", np.zeros((0,), dtype=np.float64)), dtype=np.float64), 90)
            ) if (
                isinstance(getattr(self, "_user_speed_eff_vec", None), np.ndarray)
                and int(np.asarray(getattr(self, "_user_speed_eff_vec")).size) > 0
            ) else float(self._user_speed_eff),
            "user_speed_level_idx_vec": (
                np.asarray(getattr(self, "_user_speed_level_idx_vec", np.zeros((0,), dtype=np.int64)), dtype=np.int64).tolist()
                if getattr(self, "_user_speed_level_idx_vec", None) is not None else []
            ),
            "user_speed_levels_eff": list(getattr(self, "_user_speed_levels_eff", [])),
            "user_speed_probs_eff": list(getattr(self, "_user_speed_probs_eff", [])),
            "phaseh_boundary_stick_ema": float(getattr(self, "_boundary_stick_ema", 0.0)),
            "phaseh_wall_safe_margin": float(self._wall_safe_margin()),
            "phaseh_aux_gate": float(phaseh_aux_gate),
            "phaseh_aux_term_traj": float(phaseh_aux_term_traj),
            "phaseh_aux_term_safety": float(phaseh_aux_term_safety),
            "traj_user_nn_step": float(traj_user_nn_step),
            "traj_centroid_gap_step": float(traj_centroid_gap_step),
            "traj_switchback_step": float(traj_switchback_step),
            "traj_boundary_stick_step": float(traj_boundary_stick_step),
            "safe_clip_projection_step": float(safe_clip_projection_step),
            "movement_frac_step": float(move_frac_step),
            "comp_total_noncoherent_mode": int(bool(getattr(self, "_comp_total_noncoherent_mode", False))),
            "comp_power_mode": str(getattr(self.cfg, "power_mode", "per_uav")),
            "comp_coherent_enabled": int(bool(getattr(self.cfg, "comp_coherent", True))),

            # ===== [ A] =====
            "delta_scale": float(delta_scale),  
            "vio_wall": int(vio_wall),  
            # ===== [END A] =====

            "vio_collisions_pre": int(vio_coll_pre),
            "vio_no_cover": int(vio_no_cover),
            "vio_theta": int(vio_theta),
            "vio_theta_infeasible": int(theta_infeasible),
            
            "violation_count": int(vio_total_penalty),
            "violation_count_pre": int(vio_total_pre),
            "violation_count_post": int(vio_total_post),
            "vio_collisions_post": int(vio_coll_post),
            "vio_total_penalty": int(vio_total_penalty),
            "vio_penalty_mode": str(mode),

            # reward
            "reward_survival": float(reward_survival),  
            "reward_safety": float(reward_safety),      
            "reward_performance": float(reward_performance),  
            "reward_improve": float(reward_improve),
            "reward_potential": float(reward_potential),  
            "reward_shaping_total": float(reward_shaping_step),  
            "reward_total": float(reward),
            "reward_raw": float(reward_raw),
            "reward_clipped": float(reward_clipped),
            "reward_ema": float(self._reward_ema),
            "reward_mode": self.reward_mode,
            "reward_design": str(reward_design),
            # PhaseG-v1.1 contract aliases:
            # reward_total_step (post-clip) = paper_term + shaping_term - penalty_term + misc_term
            # where misc_term absorbs clip residual to keep contract auditable under step-level clipping.
            "paper_term": float(reward_paper_step),
            "shaping_term": float(reward_shaping_step),
            "penalty_term": float(max(-float(reward_constraint_step), 0.0)),
            "misc_term": float(reward_misc_step + (float(reward) - float(reward_total))),
            "violation_continuous": float(violation_continuous_step),
            "t6_violation_ema": float(t6_violation_ema_step),
            "t6_pid_frozen": float(t6_pid_frozen_step),
            "t6_main_weight": float(t6_main_weight_step),
            "t6_potential_scale": float(t6_potential_scale_step),
            "t6_traj_scale": float(t6_traj_scale_step),
            "t6_anneal_ratio": float(t6_anneal_ratio_step),
            "t6_cost_policy_raw_step": float(t6_cost_policy_raw_step),
            "t6_cost_policy_obj_step": float(t6_cost_policy_obj_step),
            "t6_cost_ref_raw_step": float(t6_cost_ref_raw_step),
            "t6_cost_ref_obj_step": float(t6_cost_ref_obj_step),
            "t6_delay_norm_step": float(t6_delay_norm_step),
            "t6_energy_norm_step": float(t6_energy_norm_step),
            "t6_delay_calib_step": float(t6_delay_calib_step),
            "t6_energy_calib_step": float(t6_energy_calib_step),
            "t6_metric_calib_enable": float(t6_metric_calib_enable_step),
            "t6_ref_cost_mode_used": str(t6_ref_cost_mode_used),
            "t6_lambda_v_next": float(self._lambda_v_dyn),
            "t7_alpha": float(t7_alpha_step),
            "t7_cost_ref": float(t7_cost_ref_step),
            "z_cost_ref_runtime": float(z_cost_ref_runtime),
            "z_cost_ref_from_map": float(z_cost_ref_from_map),
            "z_cost_ref_from_load_map": float(z_cost_ref_from_load_map),
            "z_cost_ref_from_speed_map": float(z_cost_ref_from_speed_map),

            
            "reward_total_step": float(reward),
            "reward_paper_step": float(reward_paper_step),
            "reward_main_step": float(reward_main_step),
            "z4_gap_cost_policy_step": float(z4_gap_cost_policy_step),
            "z4_gap_cost_balanced_step": float(z4_gap_cost_balanced_step),
            "z4_gap_raw_step": float(z4_gap_raw_step),
            "z4_gap_step_clip": float(z4_gap_step_clip),
            "z4_gap_step_ema": float(z4_gap_step_ema),
            "z4_gap_step": float(z4_gap_step),
            "z4_gap_reward_step": float(z4_gap_reward_step),
            "z4_gap_lambda_input": float(z4_gap_lambda_input),
            "z4_gap_lambda_sched": float(z4_gap_lambda_sched),
            "z4_gap_lambda_cap": float(z4_gap_lambda_cap),
            "z4_gap_lambda_eff": float(z4_gap_lambda_eff),
            "z4_gap_grad_main": float(z4_gap_grad_main),
            "z4_gap_grad_ratio_eff": float(z4_gap_grad_ratio_eff),
            "z4_gap_train_frac": float(z4_gap_train_frac),
            "z4_gap_shape_guard_trigger": float(z4_gap_shape_guard_trigger),
            "z4_gap_shape_bad_streak": float(z4_gap_shape_bad_streak),
            "z4_gap_shape_slope_base": float(z4_gap_shape_slope_base),
            "z4_gap_shape_slope_with": float(z4_gap_shape_slope_with),
            "paper_cost_step": float(paper_cost_step),
            "paper_cost_obj_step": float(paper_cost_obj_step),
            "paper_cost_norm_factor": float(paper_cost_norm_factor),
            "paper_cost_ref_step": float(paper_cost_ref_step),
            "paper_delta_step": float(paper_delta_step),
            "paper_delta_norm_step": float(paper_delta_norm_step),
            "paper_adv_step": float(paper_adv_step),
            "paper_adv_norm_step": float(paper_adv_norm_step),
            "reward_paper_abs_step": float(reward_paper_abs_step),
            "reward_paper_delta_step": float(reward_paper_delta_step),
            "reward_paper_adv_step": float(reward_paper_adv_step),
            "reward_constraint_step": float(reward_constraint_step),
            "constraint_signal_step": float(constraint_signal_step),
            "violation_continuous_step": float(violation_continuous_step),
            "lambda_v_effective": float(lambda_v_effective),
            "lambda_v_next": float(self._lambda_v_dyn),
            "vio_any_step": float(vio_any_step),
            "reward_comp_step": float(reward_comp_step),
            "reward_ris_step": float(reward_ris_step),
            "reward_potential_step": float(reward_potential_step),
            "reward_traj_step": float(reward_traj_step),
            "reward_shaping_step": float(reward_shaping_step),
            "reward_misc_step": float(reward_misc_step),
            "reward_proximity_step": float(reward_proximity_step),
            "reward_proximity_weight_effective_step": float(reward_proximity_weight_effective_step),
            "reward_coverage_floor_step": float(reward_coverage_floor_step),
            "shaping_gate": float(shaping_gate),
            "shaping_gate_hard": float(shaping_gate_hard),
            "shaping_gate_soft": float(shaping_gate_soft),
            "beta_comp_t": float(beta_comp_t),
            "beta_ris_t": float(beta_ris_t),
            "beta_train_frac": float(beta_train_frac),
            "comp_bonus_raw_step": float(comp_bonus_raw_step),
            "ris_bonus_raw_step": float(ris_bonus_raw_step),
            "phaseh_ris_quality_gate": float(ris_quality_gate),
            "phaseh_comp_bonus_scale": float(comp_bonus_scale),
            "phaseh_ris_bonus_scale": float(ris_bonus_scale),
            "reward_user_nn": float(reward_user_nn),
            "reward_user_centroid": float(reward_user_centroid),
            "reward_switchback": float(reward_switchback),
            "reward_boundary_stick": float(reward_boundary_stick),
            "reward_safe_clip": float(reward_safe_clip),
            "reward_traj_direct": float(reward_traj_direct),
            "reward_traj_gain_coupling": float(reward_traj_gain_coupling),
            "reward_constraint_bonus_spill": float(reward_constraint_bonus_spill),
            # ===== [END PhaseG] =====
            "paper_delta": float(paper_delta),
            "collision_margin": float(collision_margin),
            "coverage_margin": float(coverage_margin),
            "wall_margin": float(wall_margin),
            "vio_rate_step": float(vio_rate_step),
            "weight_mode": wm_norm,
            
            "meta_beta_comp_max": float(getattr(self.cfg, "beta_comp_max", 0.0)),
            "meta_beta_ris_max": float(getattr(self.cfg, "beta_ris_max", 0.0)),

            # ===== =====
            "reward_components": reward_components,
            "violation_breakdown": violation_breakdown,
            "violation_scale_used": float(getattr(self.cfg, "violation_penalty_scale", 0.1)),

            
            "wT_base": float(wT0),
            "wE_base": float(wE0),
            "wT_dyn": float(wT_use),
            "wE_dyn": float(wE_use),
            "wT_eff": float(wT_eff),
            "wE_eff": float(wE_eff),
            "w_eff_clamped": bool(w_clamped),
            "eff_wT_min": float(getattr(self.cfg, "eff_wT_min", 0.0)),
            "eff_wE_min": float(getattr(self.cfg, "eff_wE_min", 0.0)),
            
            "dyn_impT": float(dyn_impT),
            "dyn_impE": float(dyn_impE),
            "dyn_reason": str(dyn_reason),

            "baseline_cost": float(cost_base),
            "baseline_paper_cost": float(cost_base_paper),
            "policy_paper_cost": float(cost_policy_paper),
            # -----------------------------
            # [P1/P4] Compatibility aliases (avoid train/eval mismatch)
            # -----------------------------
            "paper_cost": float(cost_policy_paper),  # alias
            "penalty_total": float(vio_total_penalty),  # alias
            "cost": float(cost_policy_paper),  # common key expected by some trainers
            "cost_constraint": float(vio_total_penalty),  # explicit constraint cost
            
            "baseline_mean_T_off": float(base_T),
            "baseline_mean_energy": float(base_E),
            "baseline_mean_T_tx": float(base_Ttx),
            "baseline_mean_rate": float(base_rate),

            "policy_mean_T_off": float(pol_T),
            "policy_mean_energy": float(pol_E),
            "policy_mean_T_tx": float(pol_Ttx),
            "policy_mean_rate": float(pol_rate),

            
            "imp_T": float(imp_T),
            "imp_E": float(imp_E),
            "imp_paper_cost": float(imp_paper),

            "p1_xT": float(xT_pol),
            "p1_xE": float(xE_pol),
            "p1_zT": float(zT_pol),
            "p1_zE": float(zE_pol),
            "p1_cost": float(cost_policy_p1),

            
            "baseline_balanced_paper_cost": float(base_pack["balanced"][0]["paper_cost"]),
            "baseline_greedy_delay_paper_cost": float(base_pack["greedy_delay"][0]["paper_cost"]),
            "baseline_greedy_energy_paper_cost": float(base_pack["greedy_energy"][0]["paper_cost"]),
            "z4_gap_enable": int(bool(getattr(self.cfg, "z4_gap_enable", False))),
            "relative_baseline_mode_used": str(relative_baseline_mode_cfg),
            "relative_baseline_position_mode_used": str(relative_baseline_position_mode),
            "relative_baseline_paper_cost": float(relative_baseline_paper_cost_step),
        }

        
        if done and reward_design == "tcom_improvement_v1":
            
            cost_episode_mean = self.episode_cost_sum / max(self.episode_steps, 1)

            
            if len(self.cost_history) < self.running_mean_window:
                
                baseline_cost = float(cost_base_paper)
            else:
                
                baseline_cost = float(np.mean(self.cost_history[-self.running_mean_window:]))

            
            improvement_episode = (baseline_cost - cost_episode_mean) / max(baseline_cost, 1e-6)

            
            bonus_w = float(max(self.improvement_episode_bonus_weight, 0.0))
            reward_episode_bonus = bonus_w * (improvement_episode / max(self.T, 1))

            
            reward = float(reward + reward_episode_bonus)

            
            info['episode_improvement_bonus'] = float(reward_episode_bonus)
            info['baseline_cost_running_mean'] = float(baseline_cost)
            info['cost_episode_mean'] = float(cost_episode_mean)
            info['improvement_episode'] = float(improvement_episode)
            info['improvement_episode_bonus_weight'] = float(bonus_w)
            info['reward_total'] = float(reward)
            info['reward_total_step'] = float(reward)
            info['misc_term'] = float(reward_misc_step + (float(reward) - float(reward_total)))
            if isinstance(info.get("reward_components", None), dict):
                rc = dict(info["reward_components"])
                rc["episode_bonus_step"] = float(reward_episode_bonus)
                rc["total"] = float(reward)
                info["reward_components"] = rc

            
            self.cost_history.append(cost_episode_mean)

            
            self.episode_cost_sum = 0.0
            self.episode_steps = 0
            self.cost_prev = None

        
        try:
            c_now = float(info.get("paper_cost", np.nan))
            if np.isfinite(c_now):
                self._obs_prev_prev_paper_cost = self._obs_prev_paper_cost
                self._obs_prev_paper_cost = float(c_now)
        except Exception:
            pass
        
        try:
            self._prev_a_binary_for_switch_diag = np.asarray(a_binary_now, dtype=np.int32).copy()
        except Exception:
            self._prev_a_binary_for_switch_diag = None

        self._last_info = info
        return self.obs_flat(), float(reward), bool(done), info

    # --------- internals ---------
    def _get_obs(self) -> Dict:
        assert self.q is not None and self.w is not None
        return {
            "uav_pos": (self.q / self.cfg.L).astype(np.float64),
            "user_pos": (self.w / self.cfg.L).astype(np.float64),
            "t_frac": np.array([self.t / max(self.cfg.T, 1)], dtype=np.float64),
            "load_scale": np.array([self.cfg.load_scale], dtype=np.float64),
        }

    def _compute_gains(
            self,
            rng=None,
            uavs_xy: Optional[np.ndarray] = None,
            users_xy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
         g IM
        H.13-S2: RISRIS
        """
        if self.q is None or self.w is None:
            self.reset()
        if rng is None:
            rng = self.rng

        I, M = int(self.cfg.I), int(self.cfg.M)

        if users_xy is None:
            users_xy = np.asarray(self.w, dtype=np.float64).reshape(I, 2)
        else:
            users_xy = np.asarray(users_xy, dtype=np.float64).reshape(I, 2)
        if uavs_xy is None:
            uavs_xy = np.asarray(self.q, dtype=np.float64).reshape(M, 2)
        else:
            uavs_xy = np.asarray(uavs_xy, dtype=np.float64).reshape(M, 2)

        H = float(self.cfg.H)
        beta0 = float(self.cfg.beta0)
        N = int(self.cfg.N)
        eta = float(self.cfg.eta)

        enable_ris = bool(getattr(self.cfg, "enable_ris", True))
        if not enable_ris:
            N = 0
            eta = 0.0

        ris_boost = float(getattr(self.cfg, "ris_boost", 1.0))
        if ris_boost < 0.0:
            ris_boost = 0.0

        g = np.zeros((I, M), dtype=np.float64)
        blockage_map = np.asarray(getattr(self, "_t6_direct_blockage_map", np.ones((I, M))), dtype=np.float64)
        if blockage_map.shape != (I, M):
            blockage_map = np.ones((I, M), dtype=np.float64)

        
        num_ris = int(getattr(self.cfg, "num_ris", 1))
        ris_positions = getattr(self.cfg, "ris_positions", None)

        if num_ris > 1 and ris_positions is not None and len(ris_positions) == num_ris:
            
            for m in range(M):
                q_m = uavs_xy[m]
                for i in range(I):
                    max_gain = 0.0
                    for ris_pos in ris_positions:
                        v = np.asarray(ris_pos, dtype=np.float64).reshape(2,)
                        beta_ir = gain_user_ris(users_xy[i], v, beta0) * ris_boost
                        beta_rm = gain_ris_uav(q_m, v, H, beta0) * ris_boost
                        beta_d = gain_direct(q_m, users_xy[i], H, beta0) * float(blockage_map[i, m])
                        gain = comp_gain(beta_d, beta_ir, beta_rm, N, eta)
                        max_gain = max(max_gain, gain)
                    g[i, m] = max_gain
        else:
            
            v = np.asarray(self.cfg.v, dtype=np.float64).reshape(2,)

            
            beta_ir = np.zeros((I,), dtype=np.float64)
            for i in range(I):
                beta_ir[i] = gain_user_ris(users_xy[i], v, beta0) * ris_boost

            for m in range(M):
                q_m = uavs_xy[m]
                beta_rm = gain_ris_uav(q_m, v, H, beta0) * ris_boost
                for i in range(I):
                    beta_d = gain_direct(q_m, users_xy[i], H, beta0) * float(blockage_map[i, m])
                    g[i, m] = comp_gain(beta_d, beta_ir[i], beta_rm, N, eta)

        return g

    def _count_collisions(self, q: np.ndarray) -> int:
        M = q.shape[0]
        cnt = 0
        d2 = float(self.cfg.dmin) ** 2
        for i in range(M):
            for j in range(i + 1, M):
                if np.sum((q[i] - q[j]) ** 2) < d2:
                    cnt += 1
        return cnt

    def _heuristic_decisions(
            self,
            g: np.ndarray,
            covered: np.ndarray,
            dist2: np.ndarray,
            mode: str = "balanced",
            D: Optional[np.ndarray] = None,
            C: Optional[np.ndarray] = None,
            use_hierarchical_assoc: bool = False,
    ):
        """
        

        use_hierarchical_assoc: Truehierarchical
            UAVCoMPcomp_score=0
            relative reward
        """
        mode = str(mode).strip().lower()
        if mode not in ("balanced", "greedy_delay", "greedy_energy"):
            raise ValueError(
                f"Invalid heuristic mode: {mode}. "
                "Expected one of: balanced, greedy_delay, greedy_energy"
            )

        I, M = self.cfg.I, self.cfg.M
        a = np.zeros((I, M), dtype=np.int64)
        vio_no_cover = 0

        lg = np.log1p(np.maximum(g, 0.0))
        
        dist_pen = dist2 / max(self.cfg.L * self.cfg.L, 1e-9)

        
        base_score = lg - 0.1 * dist_pen

        for i in range(I):
            cand = np.where(covered[i])[0]
            if cand.size == 0:
                vio_no_cover += 1
                nearest = int(np.argmin(dist2[i]))
                a[i, nearest] = 1
                continue
            s = base_score[i, cand]
            cand_sorted = cand[np.argsort(s)[::-1]]
            if use_hierarchical_assoc:
                
                
                chosen = cand_sorted[:1]
            else:
                chosen = cand_sorted[: min(self.cfg.K, cand_sorted.size)]
            a[i, chosen] = 1

        

        z = np.zeros((I, M), dtype=np.int64)

        
        f = np.asarray(self.cfg.fmax, dtype=np.float64).reshape(-1)
        if f.size != M:
            raise ValueError(f"cfg.fmax size mismatch: got {f.size}, expected M={M}")

        denom = float(f.max() - f.min())
        if denom < 1e-12:
            
            f_n = np.zeros_like(f)
        else:
            f_n = (f - f.min()) / (denom + 1e-12)

        for i in range(I):
            sel = np.where(a[i] == 1)[0]
            if sel.size == 0:
                sel = np.array([int(np.argmin(dist2[i]))], dtype=np.int64)

            if mode == "greedy_delay":
                
                score = lg[i, sel] + 0.5 * f_n[sel]
            elif mode == "greedy_energy":
                
                score = lg[i, sel] - 0.5 * f_n[sel]
            else:
                
                score = lg[i, sel] + 0.0 * f_n[sel]

            m_star = int(sel[int(np.argmax(score))])
            z[i, m_star] = 1

        
        theta = np.zeros((I, M), dtype=np.float64)
        for m in range(M):
            users_m = np.where(z[:, m] == 1)[0]
            if users_m.size == 0:
                continue
            raw = np.full((users_m.size,), 1.0 / float(users_m.size), dtype=np.float64)
            raw = ensure_theta(raw, self.cfg.theta_min)
            theta[users_m, m] = raw

        return a, z, theta, vio_no_cover

    # ===== PATCH A2: replace whole _compute_metrics to fix NameError (M/z/a) =====
    def _compute_metrics(self, D, C, g, a_mat, z_mat, theta, covered: Optional[np.ndarray] = None):
        """
        Compute per-user metrics given:
          D, C: (I,)
          g: (I, M)
          a_mat: (I, M)  association (Top-K)
          z_mat: (I, M)  compute assignment (one-hot preferred)
          theta: (I, M)  offload ratio aligned with z_mat (we read theta on the chosen compute UAV)
        """
        I = int(getattr(self.cfg, "I"))
        M = int(getattr(self.cfg, "M"))

        D = np.asarray(D, dtype=np.float64).reshape(I, )
        C = np.asarray(C, dtype=np.float64).reshape(I, )
        g = np.asarray(g, dtype=np.float64).reshape(I, M)
        a_mat = np.asarray(a_mat, dtype=np.float64).reshape(I, M)
        z_mat = np.asarray(z_mat, dtype=np.float64).reshape(I, M)
        theta = np.asarray(theta, dtype=np.float64).reshape(I, M)

        # ---- CPU queue / utilization model ----
        fmax = np.asarray(getattr(self.cfg, "fmax"), dtype=np.float64).reshape(M, )
        cpu_slot = float(getattr(self.cfg, "cpu_slot_time", 1.0))
        util_cap = float(getattr(self.cfg, "cpu_util_cap", 0.95))
        beta_q = float(getattr(self.cfg, "cpu_queue_beta", 1.0))
        eps = float(getattr(self.cfg, "eps", 1e-12))

        work = np.zeros((M,), dtype=np.float64)
        for m in range(M):
            users_m = np.where(z_mat[:, m] > 0.5)[0]
            if users_m.size == 0:
                continue
            denom = max(float(fmax[m]) * cpu_slot, eps)
            work[m] = float(np.sum(C[users_m])) / denom

        u = np.clip(work, 0.0, util_cap)
        q_factor = 1.0 / (1.0 - beta_q * u + eps)

        # ---- Wireless / compute constants ----
        B = float(getattr(self.cfg, "B"))
        N0 = float(getattr(self.cfg, "N0"))
        p_tx = float(getattr(self.cfg, "p", getattr(self.cfg, "P", 1.0)))
        R_min = float(getattr(self.cfg, "R_min", 1e-9))

        theta_min = float(getattr(self.cfg, "theta_min", 1e-6))
        cpu_scale = float(getattr(self.cfg, "cpu_scale", 1.0))
        xi = float(getattr(self.cfg, "xi", 1.0))
        cpu_energy_queue_gamma = float(getattr(self.cfg, "cpu_energy_queue_gamma", 0.0))

        coherent = bool(getattr(self.cfg, "comp_coherent", True))
        coherence_boost = float(getattr(self.cfg, "comp_coherence_boost", 1.0))

        # ---- allocate outputs (THIS fixes your unresolved references) ----
        gamma = np.zeros((I,), dtype=np.float64)
        rate = np.zeros((I,), dtype=np.float64)
        T_tx = np.zeros((I,), dtype=np.float64)
        E_tx = np.zeros((I,), dtype=np.float64)
        T_cp = np.zeros((I,), dtype=np.float64)
        E_cp = np.zeros((I,), dtype=np.float64)
        T_off = np.zeros((I,), dtype=np.float64)
        t6_interf_factor = np.ones((I,), dtype=np.float64)
        t6_interf_ratio = np.zeros((I,), dtype=np.float64)

        for i in range(I):
            # 1) CoMP SNR & rate
            enable_comp = bool(getattr(self.cfg, "enable_comp", True))
            power_mode = str(getattr(self.cfg, "power_mode", "per_uav")).strip().lower()
            if power_mode not in ("per_uav", "total"):
                power_mode = "per_uav"

            
            
            
            a_row = a_mat[i]
            coherent_i = bool(enable_comp and coherent)
            coherence_boost_i = float(coherence_boost) if enable_comp else 0.0
            power_mode_i = power_mode

            gamma[i] = snr_from_comp(
                a_row=a_row,
                p=float(p_tx),
                g_vec=g[i],
                N0=float(N0),
                B=float(B),
                coherent=bool(coherent_i),
                coherence_boost=float(coherence_boost_i),
                power_mode=str(power_mode_i),
            )
            
            interf_alpha = float(max(float(getattr(self.cfg, "t6_interference_alpha", 0.0)), 0.0))
            if interf_alpha > 1e-12:
                sel_mask = np.asarray(a_row, dtype=np.float64) > 0.0
                g_row = np.asarray(g[i], dtype=np.float64).reshape(-1)
                g_serv = float(np.sum(g_row[sel_mask])) if np.any(sel_mask) else float(np.max(g_row))
                g_other = float(np.sum(g_row[~sel_mask])) if np.any(~sel_mask) else 0.0
                ratio = float(max(g_other, 0.0) / max(g_serv, 1e-12))
                jitter = 1.0
                interf_jitter = float(self._eff_t6_interference_jitter())
                if interf_jitter > 1e-12:
                    rng = self._get_noise_rng()
                    jitter = float(rng.uniform(1.0 - interf_jitter, 1.0 + interf_jitter))
                fac = float(max(1.0 + interf_alpha * ratio * jitter, 1e-9))
                gamma[i] = float(gamma[i] / fac)
                t6_interf_factor[i] = fac
                t6_interf_ratio[i] = ratio
            rate[i] = max(rate_from_snr(float(B), float(gamma[i])), float(R_min))

            # 2) choose compute UAV
            if np.any(z_mat[i] > 0.5):
                m_star = int(np.argmax(z_mat[i]))
            else:
                
                
                idx = np.flatnonzero(a_mat[i] > 0.0)
                if idx.size > 0:
                    m_star = int(idx[int(np.argmax(g[i, idx]))])
                else:
                    m_star = int(np.argmax(g[i]))

            
            
            
            
            
            
            # - theta _user = theta * fmax
            th = float(theta[i, m_star]) if np.isfinite(theta[i, m_star]) else 0.0
            th = float(np.clip(th, float(theta_min), 1.0))

            
            
            off_bits = float(D[i])
            T_tx[i] = off_bits / max(float(rate[i]), 1e-12)
            
            S = int(np.sum(np.asarray(a_row, dtype=np.float64) > 0.0))
            if str(power_mode_i).strip().lower() == "per_uav":
                tx_power_total = float(p_tx) * float(max(S, 1))
            else:
                tx_power_total = float(p_tx)
            E_tx[i] = tx_power_total * float(T_tx[i])

            # 5) Compute time/energy PU theta
            
            off_cycles = float(C[i])
            f_user = max(float(cpu_scale) * float(fmax[m_star]) * float(th), 1e-12)
            T_cp[i] = (off_cycles / f_user) * float(q_factor[m_star])

            
            E_cp_i = float(xi) * (f_user ** 2) * off_cycles
            if cpu_energy_queue_gamma > 0.0:
                E_cp_i *= float(q_factor[m_star]) ** float(cpu_energy_queue_gamma)
            E_cp[i] = E_cp_i

            T_off[i] = T_tx[i] + T_cp[i]

        # ---- CoMP diagnostics (how often comp happens + how much SNR gain) ----
        
        S_size = np.sum(a_mat > 0.0, axis=1).astype(np.float64)  # per-user |S|
        mean_S_size = float(np.mean(S_size))
        comp_active_ratio = float(np.mean(S_size >= 2.0))

        # SNR CoMP
        denom = max(float(N0) * float(B), 1e-18)
        g_best = np.max(np.asarray(g, dtype=np.float64), axis=1)  # (I,)
        gamma_single_best = (float(p_tx) * g_best) / denom
        mean_gamma_single_best = float(np.mean(gamma_single_best))
        mean_gamma_comp = float(np.mean(gamma))
        # PhaseG-v1.1 definition lock:
        # comp_gain_ratio is a positive multiplier (>0), with 1.0 meaning parity to single-link baseline.
        comp_gain_ratio = float(max(mean_gamma_comp / (mean_gamma_single_best + 1e-12), 0.0))

        
        
        
        comp_users_count = int(np.sum(S_size >= 2.0))
        comp_users_fraction = float(comp_users_count / max(I, 1))
        avg_comp_set_size = mean_S_size  
        
        
        
        # PhaseG-v1.1 definition lock:
        # ris_gain_ratio is contribution ratio in [0, 1], NOT a >1 multiplier.
        
        
        # - g_total = (sqrt(beta_d) + N*sqrt(eta*beta_ir*beta_rm))^2
        # - g_direct_only = beta_d
        # - ris_part = max(g_total - g_direct_only, 0)
        enable_ris = bool(getattr(self.cfg, "enable_ris", True))
        ris_gain_ratio = 0.0
        ris_users_fraction = 0.0
        if enable_ris and (self.q is not None) and (self.w is not None):
            try:
                users_xy = np.asarray(self.w, dtype=np.float64).reshape(I, 2)
                uavs_xy = np.asarray(self.q, dtype=np.float64).reshape(M, 2)
                H = float(getattr(self.cfg, "H"))
                beta0 = float(getattr(self.cfg, "beta0"))

                g_direct = np.zeros((I, M), dtype=np.float64)
                for m in range(M):
                    q_m = uavs_xy[m]
                    for i in range(I):
                        g_direct[i, m] = gain_direct(q_m, users_xy[i], H, beta0)

                
                mask = (a_mat > 0.0)
                
                if np.any(mask):
                    g_sel = np.asarray(g, dtype=np.float64)[mask]
                    gd_sel = np.asarray(g_direct, dtype=np.float64)[mask]
                    ris_part = np.maximum(g_sel - gd_sel, 0.0)
                    ratio = ris_part / (np.maximum(g_sel, 1e-12))
                    ris_gain_ratio = float(np.mean(ratio)) if ratio.size > 0 else 0.0
                    ris_gain_ratio = float(np.clip(ris_gain_ratio, 0.0, 1.0))

                    
                    per_user = np.zeros((I,), dtype=np.float64)
                    for i in range(I):
                        idx = np.flatnonzero(mask[i])
                        if idx.size <= 0:
                            continue
                        g_i = np.asarray(g[i, idx], dtype=np.float64)
                        gd_i = np.asarray(g_direct[i, idx], dtype=np.float64)
                        rp_i = np.maximum(g_i - gd_i, 0.0)
                        per_user[i] = float(np.mean(rp_i / (np.maximum(g_i, 1e-12))))

                    
                    ris_users_fraction = float(np.mean(per_user > 0.05))
            except Exception as exc:
                self._warn_once("info_ris_gain_stats", f"RIS0.0{exc}")
                ris_gain_ratio = 0.0
                ris_users_fraction = 0.0
        
        
        
        uav_user_loads = np.sum(z_mat, axis=0).astype(np.float64)  # (M,) UAV
        
        
        uav_compute_loads = np.zeros(M, dtype=np.float64)
        for m in range(M):
            users_m = np.where(z_mat[:, m] > 0.5)[0]
            if users_m.size > 0:
                uav_compute_loads[m] = float(np.sum(C[users_m]))
        
        
        uav_loads = uav_compute_loads
        
        
        
        
        
        if covered is not None:
            try:
                covered_m = np.asarray(covered, dtype=bool).reshape(I, M)
            except Exception as exc:
                self._warn_once("info_covered_reshape", f"covereda_mat{exc}")
                covered_m = None
        else:
            covered_m = None

        if covered_m is None:
            
            
            covered_users_mask = np.any(a_mat > 0.0, axis=1)  # (I,)
            coverage_counts = np.sum(a_mat > 0.0, axis=1).astype(np.float64)  # (I,)
        else:
            covered_users_mask = np.any(covered_m, axis=1)  # (I,)
            coverage_counts = np.sum(covered_m, axis=1).astype(np.float64)  # (I,)

        coverage_fraction = float(np.mean(covered_users_mask))
        
        avg_coverage_count = float(np.mean(coverage_counts))
        
        
        
        total_bits = float(np.sum(D))
        total_energy = float(np.sum(E_tx + E_cp))
        energy_per_bit = float(total_energy / max(total_bits, 1e-12))
        
        
        
        hover_energy_ratio = 0.5  # step() inf
        
        # ===== [END EPIC UPGRADE] =====

        # ---- pack info (single source of truth for eval/train metrics parity) ----
        info = dict(
            gamma=gamma,
            rate=rate,
            T_tx=T_tx,
            E_tx=E_tx,
            T_cp=T_cp,
            E_cp=E_cp,
            T_off=T_off,

            mean_energy=float(np.mean(E_tx + E_cp)),
            mean_T_off=float(np.mean(T_off)),
            mean_T_tx=float(np.mean(T_tx)),
            mean_rate=float(np.mean(rate)),

            # --- NEW: CoMP visibility metrics ---
            mean_S_size=mean_S_size,
            comp_active_ratio=comp_active_ratio,
            mean_gamma_comp=mean_gamma_comp,
            mean_gamma_single_best=mean_gamma_single_best,
            comp_gain_ratio=comp_gain_ratio,
            
            # ===== [EPIC UPGRADE] =====
            # CoMP
            comp_users_fraction=comp_users_fraction,
            avg_comp_set_size=avg_comp_set_size,
            comp_users_count=comp_users_count,
            
            
            ris_gain_ratio=ris_gain_ratio,
            ris_users_fraction=ris_users_fraction,
            
            
            uav_loads=uav_loads.tolist(),  # list JSON inf
            uav_user_loads=uav_user_loads.tolist(),
            uav_compute_loads=uav_compute_loads.tolist(),
            
            
            coverage_fraction=coverage_fraction,
            avg_coverage_count=avg_coverage_count,
            
            
            energy_per_bit=energy_per_bit,
            hover_energy_ratio=hover_energy_ratio,
            total_bits=total_bits,
            total_energy=total_energy,
            t6_interference_factor_mean=float(np.mean(t6_interf_factor)),
            t6_interference_ratio_mean=float(np.mean(t6_interf_ratio)),
            # ===== [END EPIC UPGRADE] =====

            paper_cost=(
                    float(getattr(self.cfg, "w_delay", 0.5)) * (
                    float(np.mean(T_off)) / max(float(getattr(self.cfg, "T_scale", 1.0)), 1e-9)
            )
                    + float(getattr(self.cfg, "w_energy", 0.5)) * (
                            float(np.mean(E_tx + E_cp)) / max(float(getattr(self.cfg, "E_scale", 1.0)), 1e-9)
                    )
            ),
        )
        return info



