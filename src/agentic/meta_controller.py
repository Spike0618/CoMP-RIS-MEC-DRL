from __future__ import annotations

import json
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .meta_action_space import action_patch, allowed_actions
from .meta_fallback import apply_safety_fallback
from .meta_llm_client import OpenAICompatMetaClient


def _to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class MetaController:
    run_dir: Path
    cfg: Dict[str, Any]
    source: str
    rng: random.Random
    llm_client: Optional[OpenAICompatMetaClient]

    def __post_init__(self) -> None:
        self.enabled = _to_bool(self.cfg.get("enabled", False), False)
        self.source = str(self.source or self.cfg.get("source", "heuristic")).strip().lower()
        self.allow_actions = allowed_actions(self.cfg)
        runtime = self.cfg.get("runtime", {}) if isinstance(self.cfg, dict) else {}
        if not isinstance(runtime, dict):
            runtime = {}
        
        self.allow_llm_training = _to_bool(runtime.get("allow_llm_training", False), False)
        self.llm_fail_fallback = str(runtime.get("llm_fail_fallback", "heuristic")).strip().lower() or "heuristic"
        self.llm_force_temperature_zero = _to_bool(runtime.get("llm_force_temperature_zero", True), True)
        self.llm_guard_forced = _to_bool(self.cfg.get("_llm_guard_forced", False), False)
        if self.source == "llm" and (not self.allow_llm_training):
            self.source = "heuristic"
            self.llm_guard_forced = True

        trigger = self.cfg.get("trigger", {}) if isinstance(self.cfg, dict) else {}
        self.warmup_evals = int(max(0, _to_float(trigger.get("warmup_evals", 2), 2.0)))
        self.decision_every = int(max(1, _to_float(trigger.get("decision_every_evals", 1), 1.0)))

        safety = self.cfg.get("safety", {}) if isinstance(self.cfg, dict) else {}
        self.min_confidence = float(max(0.0, _to_float(safety.get("min_confidence", 0.60), 0.60)))

        logging_cfg = self.cfg.get("logging", {}) if isinstance(self.cfg, dict) else {}
        self.save_jsonl = _to_bool(logging_cfg.get("save_jsonl", True), True)
        self.jsonl_path = self.run_dir / str(logging_cfg.get("file", "meta_decisions.jsonl")).strip()

        llm_cfg = self.cfg.get("llm", {}) if isinstance(self.cfg, dict) else {}
        self.max_calls_per_hour = int(max(0, _to_float(llm_cfg.get("max_calls_per_hour", 0), 0.0)))
        self.max_tokens_per_call = int(max(1, _to_float(llm_cfg.get("max_tokens_per_call", 256), 256.0)))
        self.max_cost_per_run = float(max(0.0, _to_float(llm_cfg.get("max_cost_per_run_usd", 0.0), 0.0)))

        self.n_decisions = 0
        self.n_fallback = 0
        self.n_timeout = 0
        self.n_json_invalid = 0
        self.llm_calls = 0
        self.llm_tokens_in = 0
        self.llm_tokens_out = 0
        self.llm_cost_sum = 0.0
        self._llm_call_ts: list[float] = []
        self._stability_guard_streak = 0
        self._requested_action_counts: Dict[str, int] = {}
        self._final_action_counts: Dict[str, int] = {}
        self._fallback_reason_counts: Dict[str, int] = {}

    def is_enabled(self) -> bool:
        return bool(self.enabled)

    def should_trigger(self, eval_index: int) -> bool:
        if not self.is_enabled():
            return False
        ei = int(max(0, eval_index))
        if ei <= int(self.warmup_evals):
            return False
        return bool(((ei - int(self.warmup_evals)) % int(self.decision_every)) == 0)

    def _heuristic_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        clip_now = _to_float(state.get("clipfrac_now"), 0.0)
        kl_now = _to_float(state.get("kl_per_dim_now"), 0.0)
        vio_now = _to_float(state.get("vio_any_now"), 0.0)
        cov_now = _to_float(state.get("coverage_now"), 1.0)
        ev_now = _to_float(state.get("explained_variance_now"), 0.0)
        cost_tr = _to_float(state.get("paper_cost_trend"), 0.0)
        ris_gain = _to_float(state.get("ris_gain_ratio_now"), 0.0)
        comp_gain = _to_float(state.get("comp_gain_ratio_now"), 0.0)
        load = _to_float(state.get("eval_load_now"), 1.0)
        idle = _to_float(state.get("traj_idle_ratio_now"), 0.0)
        boundary = _to_float(state.get("traj_boundary_stick_frac_now"), 0.0)
        user_gap = _to_float(state.get("traj_user_nn_dist_norm_now"), 0.0)
        centroid_gap = _to_float(state.get("traj_centroid_gap_norm_now"), 0.0)

        safety = self.cfg.get("safety", {}) if isinstance(self.cfg, dict) else {}
        guard_gate = safety.get("stability_guard_gate", {}) if isinstance(safety, dict) else {}
        clip_hi = _to_float(guard_gate.get("clipfrac_high", 0.30), 0.30)
        kl_hi = _to_float(guard_gate.get("kl_high", 0.004), 0.004)
        clip_hard = _to_float(guard_gate.get("clipfrac_hard", clip_hi * 1.35), clip_hi * 1.35)
        kl_hard = _to_float(guard_gate.get("kl_hard", kl_hi * 1.5), kl_hi * 1.5)
        idle_hi = _to_float(guard_gate.get("idle_high", 0.65), 0.65)
        boundary_hi = _to_float(guard_gate.get("boundary_high", 0.45), 0.45)
        user_gap_hi = _to_float(guard_gate.get("user_nn_high", 0.55), 0.55)
        centroid_gap_hi = _to_float(guard_gate.get("centroid_gap_high", 0.35), 0.35)
        ev_lo = _to_float(guard_gate.get("ev_low", 0.05), 0.05)

        ppo_unstable = (clip_now > clip_hi) or (kl_now > kl_hi) or (ev_now < ev_lo and (clip_now > 0.20 or kl_now > 0.003))
        traj_bad = (idle > idle_hi) or (boundary > boundary_hi) or (user_gap > user_gap_hi) or (centroid_gap > centroid_gap_hi)
        gains_weak = (ris_gain < 0.12) or (comp_gain < 1.03)
        cost_worse = cost_tr > 0.0
        cost_stalled = bool(math.isfinite(cost_tr) and abs(cost_tr) <= 0.001)

        if ppo_unstable:
            aid = "stability_guard"
        elif traj_bad:
            if "trajectory_recover" in self.allow_actions and (cost_worse or cost_stalled or gains_weak):
                aid = "trajectory_recover"
            else:
                aid = "constraint_relax"
        elif vio_now > 0.95 or cov_now < 0.75:
            aid = "constraint_tight"
        elif cost_worse:
            if load >= 0.7 and ris_gain < 0.12:
                aid = "ris_boost"
            elif comp_gain < 1.02:
                aid = "comp_boost"
            else:
                aid = "joint_boost"
        elif cost_tr < -0.003 and vio_now < 0.50:
            if load >= 0.7 and gains_weak:
                aid = "joint_boost"
            else:
                aid = "constraint_relax"
        else:
            aid = "hold"

        return {"action_id": aid, "confidence": 0.74, "reason": "heuristic_rules"}

    def _stability_guard_override(self, action_id: str, state: Dict[str, Any]) -> tuple[str, bool, str]:
        aid = str(action_id or "hold").strip().lower() or "hold"
        if aid != "stability_guard":
            return aid, False, ""

        safety = self.cfg.get("safety", {}) if isinstance(self.cfg, dict) else {}
        guard_gate = safety.get("stability_guard_gate", {}) if isinstance(safety, dict) else {}
        guard_policy = safety.get("stability_guard_policy", {}) if isinstance(safety, dict) else {}
        max_streak = int(max(1, _to_float(guard_policy.get("max_consecutive", 2), 2.0)))

        clip_hi = _to_float(guard_gate.get("clipfrac_high", 0.30), 0.30)
        kl_hi = _to_float(guard_gate.get("kl_high", 0.004), 0.004)
        clip_hard = _to_float(guard_gate.get("clipfrac_hard", clip_hi * 1.35), clip_hi * 1.35)
        kl_hard = _to_float(guard_gate.get("kl_hard", kl_hi * 1.5), kl_hi * 1.5)
        idle_hi = _to_float(guard_gate.get("idle_high", 0.65), 0.65)
        boundary_hi = _to_float(guard_gate.get("boundary_high", 0.45), 0.45)
        user_gap_hi = _to_float(guard_gate.get("user_nn_high", 0.55), 0.55)
        centroid_gap_hi = _to_float(guard_gate.get("centroid_gap_high", 0.35), 0.35)

        clip_now = _to_float(state.get("clipfrac_now"), 0.0)
        kl_now = _to_float(state.get("kl_per_dim_now"), 0.0)
        idle_now = _to_float(state.get("traj_idle_ratio_now"), -1.0)
        boundary_now = _to_float(state.get("traj_boundary_stick_frac_now"), -1.0)
        user_gap_now = _to_float(state.get("traj_user_nn_dist_norm_now"), -1.0)
        centroid_gap_now = _to_float(state.get("traj_centroid_gap_norm_now"), -1.0)

        ppo_unstable = (clip_now > clip_hard) or (kl_now > kl_hard)
        traj_bad = (
            (idle_now > idle_hi)
            or (boundary_now > boundary_hi)
            or (user_gap_now > user_gap_hi)
            or (centroid_gap_now > centroid_gap_hi)
        )
        if self._stability_guard_streak < max_streak or ppo_unstable or (not traj_bad):
            return aid, False, ""

        if "trajectory_recover" in self.allow_actions:
            return "trajectory_recover", True, f"guard_streak_override:{self._stability_guard_streak}"
        if "constraint_relax" in self.allow_actions:
            return "constraint_relax", True, f"guard_streak_override:{self._stability_guard_streak}"
        return aid, False, ""

    def _random_action(self) -> Dict[str, Any]:
        acts = [a for a in self.allow_actions if a]
        if not acts:
            acts = ["hold"]
        aid = self.rng.choice(acts)
        
        conf = float(max(0.0, min(1.0, max(0.50, float(self.min_confidence)))))
        return {"action_id": str(aid), "confidence": conf, "reason": "random_meta"}

    @staticmethod
    def _bump_counter(bucket: Dict[str, int], key: str) -> None:
        k = str(key or "unknown").strip().lower() or "unknown"
        bucket[k] = int(bucket.get(k, 0)) + 1

    def _llm_budget_ok(self) -> bool:
        now = time.time()
        self._llm_call_ts = [t for t in self._llm_call_ts if (now - t) <= 3600.0]
        if self.max_calls_per_hour > 0 and len(self._llm_call_ts) >= self.max_calls_per_hour:
            return False
        if self.max_cost_per_run > 0 and self.llm_cost_sum >= self.max_cost_per_run:
            return False
        return True

    def _llm_action(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_client is None or (not self.llm_client.is_ready()):
            raise RuntimeError("llm_not_configured")
        if not self._llm_budget_ok():
            raise RuntimeError("llm_budget_exceeded")

        prompt_obj = {
            "task": "choose_meta_action",
            "allowed_actions": list(self.allow_actions),
            "state": state,
            "context": context,
        }
        prompt = json.dumps(prompt_obj, ensure_ascii=False, separators=(",", ":"))
        if self.max_tokens_per_call > 0:
            self.llm_client.max_tokens = int(min(int(self.llm_client.max_tokens), int(self.max_tokens_per_call)))
        out = self.llm_client.request_action(prompt)
        self.llm_calls += 1
        self._llm_call_ts.append(time.time())
        self.llm_tokens_in += int(out.get("token_in", 0) or 0)
        self.llm_tokens_out += int(out.get("token_out", 0) or 0)
        self.llm_cost_sum += float(out.get("cost_est", 0.0) or 0.0)
        return out

    def decide(
        self,
        *,
        state: Dict[str, Any],
        current_params: Dict[str, float],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        src = str(self.source).strip().lower()
        requested = {"action_id": "hold", "confidence": 0.0, "reason": "disabled"}
        timeout_flag = False
        json_invalid = False

        if src == "random":
            requested = self._random_action()
        elif src == "heuristic":
            requested = self._heuristic_action(state)
        elif src == "llm":
            try:
                requested = self._llm_action(state, context)
            except Exception as e:
                fail_mode = str(self.llm_fail_fallback).strip().lower()
                if fail_mode == "heuristic":
                    requested = self._heuristic_action(state)
                    requested["reason"] = f"llm_fail->heuristic:{type(e).__name__}"
                elif fail_mode == "random":
                    requested = self._random_action()
                    requested["reason"] = f"llm_fail->random:{type(e).__name__}"
                else:
                    requested = {"action_id": "hold", "confidence": 0.0, "reason": f"llm_fail:{type(e).__name__}"}
                msg = str(e).lower()
                if "timeout" in msg:
                    timeout_flag = True
                if "json" in msg:
                    json_invalid = True
        else:
            requested = self._heuristic_action(state)
            src = "heuristic"

        req_action = str(requested.get("action_id", "hold")).strip().lower() or "hold"
        conf = _to_float(requested.get("confidence", 0.0), 0.0)
        if conf < self.min_confidence:
            req_action = "hold"

        final_action, fb_flag, fb_reason = apply_safety_fallback(
            action_id=req_action,
            state=state,
            cfg=self.cfg,
            allowed_actions=self.allow_actions,
        )
        over_action, over_flag, over_reason = self._stability_guard_override(final_action, state)
        if over_flag:
            final_action = over_action
            fb_flag = True
            fb_reason = f"{fb_reason}|{over_reason}" if fb_reason else over_reason

        self._bump_counter(self._requested_action_counts, req_action)
        self._bump_counter(self._final_action_counts, str(final_action))
        if fb_flag:
            self._bump_counter(self._fallback_reason_counts, fb_reason if fb_reason else "fallback")

        patch_obj = action_patch(final_action, current_params, self.cfg)

        latency_ms = float((time.perf_counter() - t0) * 1000.0)
        self.n_decisions += 1
        if fb_flag:
            self.n_fallback += 1
        if timeout_flag:
            self.n_timeout += 1
        if json_invalid:
            self.n_json_invalid += 1
        if str(final_action).strip().lower() == "stability_guard":
            self._stability_guard_streak += 1
        else:
            self._stability_guard_streak = 0

        decision = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "source": src,
            "env_step": int(context.get("env_step", 0) or 0),
            "train_episode": int(context.get("train_episode", 0) or 0),
            "eval_index": int(context.get("eval_index", 0) or 0),
            "requested_action": str(requested.get("action_id", "hold")),
            "action_id": str(patch_obj.get("action_id", "hold")),
            "confidence": float(conf),
            "reason": str(requested.get("reason", "")),
            "fallback_flag": bool(fb_flag),
            "fallback_reason": str(fb_reason),
            "timeout_flag": bool(timeout_flag),
            "json_invalid_flag": bool(json_invalid),
            "patch": patch_obj.get("patch", {}),
            "delta": patch_obj.get("delta", {}),
            "clamped": patch_obj.get("clamped", {}),
            "current_params": {k: float(v) for k, v in (current_params or {}).items()},
            "latency_ms": float(latency_ms),
            "token_in": int(requested.get("token_in", 0) or 0),
            "token_out": int(requested.get("token_out", 0) or 0),
            "cost_est": float(requested.get("cost_est", 0.0) or 0.0),
        }
        return decision

    def append_decision(self, decision: Dict[str, Any]) -> None:
        if not self.save_jsonl:
            return
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(decision, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        n = int(max(1, self.n_decisions))
        return {
            "meta_enabled": bool(self.is_enabled()),
            "meta_source": str(self.source),
            "meta_decisions": int(self.n_decisions),
            "meta_fallback_count": int(self.n_fallback),
            "meta_timeout_count": int(self.n_timeout),
            "meta_json_invalid_count": int(self.n_json_invalid),
            "meta_fallback_rate": float(self.n_fallback / n),
            "meta_timeout_rate": float(self.n_timeout / n),
            "meta_json_invalid_rate": float(self.n_json_invalid / n),
            "meta_llm_calls": int(self.llm_calls),
            "meta_token_in": int(self.llm_tokens_in),
            "meta_token_out": int(self.llm_tokens_out),
            "meta_cost_est_usd": float(self.llm_cost_sum),
            "meta_guard_streak": int(self._stability_guard_streak),
            "meta_llm_guard_forced": bool(self.llm_guard_forced),
            "meta_requested_action_counts": {str(k): int(v) for k, v in sorted(self._requested_action_counts.items())},
            "meta_final_action_counts": {str(k): int(v) for k, v in sorted(self._final_action_counts.items())},
            "meta_fallback_reason_counts": {str(k): int(v) for k, v in sorted(self._fallback_reason_counts.items())},
        }

    def save_stats(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.stats(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def create_meta_controller(
    *,
    run_dir: Path,
    seed: int,
    meta_yaml: Optional[Path],
    inline_cfg: Optional[Dict[str, Any]] = None,
    source_override: Optional[str] = None,
    enabled_override: Optional[bool] = None,
) -> Optional[MetaController]:
    cfg_file: Dict[str, Any] = {}
    if meta_yaml is not None:
        cfg_file = _load_yaml(Path(meta_yaml))
    cfg_inline = deepcopy(inline_cfg) if isinstance(inline_cfg, dict) else {}
    cfg = _deep_merge(cfg_file, cfg_inline)

    if source_override is not None and str(source_override).strip():
        cfg["source"] = str(source_override).strip().lower()
    if enabled_override is not None:
        cfg["enabled"] = bool(enabled_override)

    enabled = _to_bool(cfg.get("enabled", False), False)
    source = str(cfg.get("source", "heuristic")).strip().lower() or "heuristic"
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
    allow_llm_training = _to_bool(runtime_cfg.get("allow_llm_training", False), False)
    llm_force_temperature_zero = _to_bool(runtime_cfg.get("llm_force_temperature_zero", True), True)
    if source == "llm" and (not allow_llm_training):
        cfg["_llm_guard_forced"] = True
        source = "heuristic"

    llm_client: Optional[OpenAICompatMetaClient] = None
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    if source == "llm":
        provider = str(llm_cfg.get("provider", "openai")).strip().lower()
        base_url = str(llm_cfg.get("base_url", "")).strip()
        model = str(llm_cfg.get("model", "")).strip()
        api_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
        api_key = str(llm_cfg.get("api_key", "")).strip() or str(os.environ.get(api_env, "")).strip()
        timeout_ms = int(max(100, _to_float(llm_cfg.get("timeout_ms", 8000), 8000.0)))
        temperature = float(max(0.0, _to_float(llm_cfg.get("temperature", 0.0), 0.0)))
        if llm_force_temperature_zero:
            temperature = 0.0
        max_tokens = int(max(16, _to_float(llm_cfg.get("max_tokens", 128), 128.0)))
        use_json_schema = _to_bool(llm_cfg.get("json_schema", True), True)
        llm_client = OpenAICompatMetaClient(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_ms=timeout_ms,
            temperature=temperature,
            max_tokens=max_tokens,
            use_json_schema=use_json_schema,
        )

    ctrl = MetaController(
        run_dir=Path(run_dir),
        cfg=cfg,
        source=source,
        rng=random.Random(int(seed) + 7919),
        llm_client=llm_client,
    )
    ctrl.enabled = bool(enabled)
    return ctrl if ctrl.is_enabled() else None
