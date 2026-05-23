"""
Tianshou PPO - PPO Policy

Tianshou PPO Policy

"""

import torch
from typing import Tuple, Union, Sequence, Optional
from tianshou.policy import PPOPolicy
from torch.distributions import Normal
import numpy as np

from src.algos.tianshou.networks import create_actor_critic, CompRISActor, CompRISCritic



OptimizerBundle = Union[torch.optim.Optimizer, Tuple[torch.optim.Optimizer, torch.optim.Optimizer]]


def _atanh(x: torch.Tensor) -> torch.Tensor:
    """atanhtorchatanh"""
    # 0.5 * (ln(1+x) - ln(1-x))
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class SquashedGaussian(torch.distributions.Distribution):
    """
    tanh-squashed Gaussianlogp

    cliplogpPhase3

    
    - loc(-1, 1)Actormutanh
    - locpre-tanhtanh
    - log_probtanhJacobianPPO ratio/KL/clipfrac
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-6,
        act_dim_effective: Optional[int] = None,
    ):
        self.eps = float(eps)
        self.act_dim = int(loc.shape[-1])
        eff = int(act_dim_effective) if act_dim_effective is not None else int(self.act_dim)
        eff = int(max(1, min(int(eff), int(self.act_dim))))
        
        
        self.act_dim_effective = int(eff)

        
        loc_bounded = torch.clamp(loc, -1.0 + self.eps, 1.0 - self.eps)
        pre_loc = _atanh(loc_bounded)
        self.pre_loc = pre_loc
        self.scale = scale
        
        self.base = Normal(pre_loc, scale)
        super().__init__(batch_shape=pre_loc.shape[:-1], event_shape=pre_loc.shape[-1:])

    def sample(self, sample_shape: torch.Size = torch.Size()):
        u = self.base.sample(sample_shape)
        return torch.tanh(u)

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        u = self.base.rsample(sample_shape)
        return torch.tanh(u)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        
        a = torch.clamp(value, -1.0 + self.eps, 1.0 - self.eps)
        u = _atanh(a)
        
        logp_u = self.base.log_prob(u)
        
        log_det = torch.log(1.0 - a.pow(2) + self.eps)
        logp_pd = logp_u - log_det

        
        if int(self.act_dim_effective) < int(self.act_dim):
            logp_pd = logp_pd[..., : int(self.act_dim_effective)]
        return torch.sum(logp_pd, dim=-1)

    def log_prob_per_dim(self, value: torch.Tensor) -> torch.Tensor:
        """log_probtanh JacobianPPO"""
        a = torch.clamp(value, -1.0 + self.eps, 1.0 - self.eps)
        u = _atanh(a)
        logp_u = self.base.log_prob(u)
        log_det = torch.log(1.0 - a.pow(2) + self.eps)
        logp_pd = logp_u - log_det
        if int(self.act_dim_effective) < int(self.act_dim):
            logp_pd = logp_pd[..., : int(self.act_dim_effective)]
        return logp_pd

    def entropy(self) -> torch.Tensor:
        
        
        base_ent_pd = self.base.entropy()  
        a_mean = torch.tanh(self.pre_loc)
        a_mean = torch.clamp(a_mean, -1.0 + self.eps, 1.0 - self.eps)
        log_det_pd = torch.log(1.0 - a_mean.pow(2) + self.eps)

        if int(self.act_dim_effective) < int(self.act_dim):
            base_ent_pd = base_ent_pd[..., : int(self.act_dim_effective)]
            log_det_pd = log_det_pd[..., : int(self.act_dim_effective)]

        return torch.sum(base_ent_pd, dim=-1) + torch.sum(log_det_pd, dim=-1)

    def entropy_per_dim(self) -> torch.Tensor:
        """entropy"""
        base_ent_pd = self.base.entropy()
        a_mean = torch.tanh(self.pre_loc)
        a_mean = torch.clamp(a_mean, -1.0 + self.eps, 1.0 - self.eps)
        log_det_pd = torch.log(1.0 - a_mean.pow(2) + self.eps)
        ent_pd = base_ent_pd + log_det_pd
        if int(self.act_dim_effective) < int(self.act_dim):
            ent_pd = ent_pd[..., : int(self.act_dim_effective)]
        return ent_pd

    def __getattr__(self, name: str):
        
        return getattr(self.base, name)


class PPOPolicyWithKLStop(PPOPolicy):
    """
    KLPPOPolicyPhase2/3

    Tianshou 0.5.1  PPOPolicy  target_kl (repeat/epochs)
    (act_dim=60) + KL/d
    - ratio/clipfrac
    - rewardEV

    epochapprox_klepoch
    TRPO
    """

    def __init__(self, *args, target_kl: float = 0.0, kl_stop_mult: float = 1.5, **kwargs):
        
        optim_critic = kwargs.pop("optim_critic", None)
        dual_optimizer_enabled = bool(kwargs.pop("dual_optimizer_enabled", False))
        grad_norm_actor = kwargs.pop("grad_norm_actor", None)
        grad_norm_critic = kwargs.pop("grad_norm_critic", None)
        vf_loss_clip_max = kwargs.pop("vf_loss_clip_max", 0.0)
        super().__init__(*args, **kwargs)
        self._target_kl_stop = float(target_kl)
        self._kl_stop_mult = float(kl_stop_mult)
        
        self._optim_actor = getattr(self, "optim", None)
        self._optim_critic = optim_critic
        self._dual_optimizer_enabled = bool(
            dual_optimizer_enabled
            and isinstance(self._optim_actor, torch.optim.Optimizer)
            and isinstance(self._optim_critic, torch.optim.Optimizer)
        )
        self._grad_norm_actor = float(
            grad_norm_actor if grad_norm_actor is not None else float(getattr(self, "_grad_norm", 0.0) or 0.0)
        )
        self._grad_norm_critic = float(
            grad_norm_critic if grad_norm_critic is not None else float(getattr(self, "_grad_norm", 0.0) or 0.0)
        )
        self._vf_loss_clip_max = float(max(float(vf_loss_clip_max), 0.0))

    def forward(self, batch, state=None, **kwargs):  # type: ignore[override]
        """
        Phase C C1  scores  score  mu score 

        
        - C1 (delta)scores  scores_fixed 
        - Actor.forward  score  mu/sigma  score  mu
          sigma fixed 
        """
        out = super().forward(batch, state=state, **kwargs)
        try:
            actor = getattr(self, "actor", None)
            mode = str(getattr(actor, "scores_mode", "learned")).strip().lower() if actor is not None else "learned"
            if mode == "fixed":
                dd = int(getattr(actor, "delta_dim", 0))
                if dd > 0 and hasattr(out, "act") and hasattr(out, "logits") and out.act is not None:
                    mu = out.logits[0] if isinstance(out.logits, tuple) else out.logits
                    if hasattr(out.act, "shape") and out.act.shape[-1] > dd:
                        act = out.act.clone()
                        act[..., dd:] = mu[..., dd:]
                        out.act = act
        except Exception:
            pass
        return out

    @staticmethod
    def _unwrap_squashed_dist(dist) -> tuple[Optional[SquashedGaussian], float]:
        """
         _ScaledDist  SquashedGaussian 
        """
        scale = 1.0
        cur = dist
        for _ in range(4):
            if isinstance(cur, SquashedGaussian):
                return cur, float(scale)
            if hasattr(cur, "base"):
                if hasattr(cur, "scale"):
                    try:
                        scale *= float(getattr(cur, "scale", 1.0))
                    except Exception:
                        pass
                cur = getattr(cur, "base")
                continue
            break
        return None, float(scale)

    def _split_log_prob(
        self,
        dist,
        act: torch.Tensor,
        delta_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        log_probdelta/scoreclip
        """
        base, scale = self._unwrap_squashed_dist(dist)
        if base is None:
            z = dist.log_prob(act)
            return z, torch.zeros_like(z), 0, 0
        logp_pd = base.log_prob_per_dim(act) * float(scale)
        eff = int(logp_pd.shape[-1]) if logp_pd.ndim >= 2 else 0
        dd = int(max(0, min(int(delta_dim), int(eff))))
        sd = int(max(0, int(eff - dd)))
        if dd > 0:
            lp_d = torch.sum(logp_pd[..., :dd], dim=-1)
        else:
            lp_d = torch.zeros((logp_pd.shape[0],), dtype=logp_pd.dtype, device=logp_pd.device)
        if sd > 0:
            lp_s = torch.sum(logp_pd[..., dd:], dim=-1)
        else:
            lp_s = torch.zeros((logp_pd.shape[0],), dtype=logp_pd.dtype, device=logp_pd.device)
        return lp_d, lp_s, int(dd), int(sd)

    def _split_entropy(
        self,
        dist,
        delta_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        delta/score
        """
        base, scale = self._unwrap_squashed_dist(dist)
        if base is None:
            z = dist.entropy()
            return z, torch.zeros_like(z), 0, 0
        ent_pd = base.entropy_per_dim() * float(scale)
        eff = int(ent_pd.shape[-1]) if ent_pd.ndim >= 2 else 0
        dd = int(max(0, min(int(delta_dim), int(eff))))
        sd = int(max(0, int(eff - dd)))
        if dd > 0:
            ent_d = torch.sum(ent_pd[..., :dd], dim=-1)
        else:
            ent_d = torch.zeros((ent_pd.shape[0],), dtype=ent_pd.dtype, device=ent_pd.device)
        if sd > 0:
            ent_s = torch.sum(ent_pd[..., dd:], dim=-1)
        else:
            ent_s = torch.zeros((ent_pd.shape[0],), dtype=ent_pd.dtype, device=ent_pd.device)
        return ent_d, ent_s, int(dd), int(sd)

    @staticmethod
    def _as_logits_pair(logits) -> tuple[Optional[object], Optional[object]]:
        """
        logits(mu, sigma)
        """
        if isinstance(logits, tuple) and len(logits) >= 2:
            return logits[0], logits[1]
        if isinstance(logits, list) and len(logits) >= 2:
            return logits[0], logits[1]
        if isinstance(logits, dict):
            mu_v = logits.get("mu", logits.get("loc", logits.get("mean", None)))
            sg_v = logits.get("sigma", logits.get("scale", logits.get("std", logits.get("stddev", None))))
            if (mu_v is not None) and (sg_v is not None):
                return mu_v, sg_v
        try:
            mu_v = getattr(logits, "mu", None)
            sg_v = getattr(logits, "sigma", None)
            if (mu_v is None) or (sg_v is None):
                mu_v = getattr(logits, "loc", getattr(logits, "mean", None))
                sg_v = getattr(logits, "scale", getattr(logits, "stddev", None))
            if (mu_v is not None) and (sg_v is not None):
                return mu_v, sg_v
        except Exception:
            pass
        return None, None

    def _dist_to_logits_pair(self, dist) -> tuple[Optional[object], Optional[object]]:
        """
        (mu, sigma)A1clip
        """
        if dist is None:
            return None, None
        try:
            base_squashed, _ = self._unwrap_squashed_dist(dist)
            if base_squashed is not None:
                mu_v = torch.tanh(base_squashed.pre_loc)
                sg_v = base_squashed.scale
                return mu_v, sg_v
        except Exception:
            pass
        try:
            cur = dist
            for _ in range(6):
                mu_v = getattr(cur, "loc", getattr(cur, "mean", None))
                sg_v = getattr(cur, "scale", getattr(cur, "stddev", None))
                if (mu_v is not None) and (sg_v is not None):
                    return mu_v, sg_v
                if hasattr(cur, "base_dist"):
                    cur = getattr(cur, "base_dist")
                    continue
                if hasattr(cur, "base"):
                    cur = getattr(cur, "base")
                    continue
                break
        except Exception:
            pass
        return None, None

    def _a1_warn_once(self, key: str, message: str) -> None:
        """
        A1
        """
        k = str(key).strip().lower()
        warned = getattr(self, "_a1_warned_keys", None)
        if not isinstance(warned, set):
            warned = set()
            setattr(self, "_a1_warned_keys", warned)
        if k in warned:
            return
        warned.add(k)
        print(f"[HB][ppo][A1][WARN] {message}", flush=True)

    def _ensure_headclip_logits(self, batch) -> tuple[bool, str]:
        """
        A1cliplogits
        - batchlogits
        - learn(mu,sigma)batch.logits
        """
        mu0, sg0 = self._as_logits_pair(getattr(batch, "logits", None))
        if (mu0 is not None) and (sg0 is not None):
            try:
                batch.logits_mu_old = torch.as_tensor(mu0).detach()
                batch.logits_sigma_old = torch.as_tensor(sg0).detach()
            except Exception:
                pass
            return True, "existing"

        chunk = int(max(int(getattr(self, "_a1_logits_chunk", 1024)), 64))
        mu_list = []
        sg_list = []
        try:
            with torch.no_grad():
                for mb in batch.split(chunk, merge_last=True):
                    out_mb = self(mb)
                    mu_b, sg_b = self._as_logits_pair(getattr(out_mb, "logits", None))
                    if (mu_b is None) or (sg_b is None):
                        mu_b, sg_b = self._dist_to_logits_pair(getattr(out_mb, "dist", None))
                    if (mu_b is None) or (sg_b is None):
                        return False, "recompute_invalid_logits"
                    mu_t = torch.as_tensor(mu_b).detach()
                    sg_t = torch.as_tensor(sg_b).detach()
                    mu_list.append(mu_t)
                    sg_list.append(sg_t)
            if len(mu_list) <= 0:
                return False, "recompute_empty"
            mu_cat = torch.cat(mu_list, dim=0)
            sg_cat = torch.cat(sg_list, dim=0)
            batch.logits = (mu_cat, sg_cat)
            
            batch.logits_mu_old = mu_cat
            batch.logits_sigma_old = sg_cat
            return True, "recompute"
        except Exception as exc:
            return False, f"recompute_failed:{type(exc).__name__}"

    def learn(self, batch, batch_size: int, repeat: int, **kwargs):  # type: ignore[override]
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        vf_clip_hit_fracs = []
        approx_kl_history: list[float] = []

        
        
        
        
        selfcheck_enabled = bool(getattr(self, "_selfcheck_enabled", True))
        selfcheck_every = int(getattr(self, "_selfcheck_every", 10))
        selfcheck_tol = float(getattr(self, "_selfcheck_tol", 1e-4))
        upd = int(getattr(self, "_selfcheck_update_count", 0)) + 1
        setattr(self, "_selfcheck_update_count", int(upd))
        do_selfcheck = bool(
            selfcheck_enabled and (upd == 1 or (selfcheck_every > 0 and (upd % max(selfcheck_every, 1) == 0)))
        )
        selfcheck_done = False

        
        head_tune_global = bool(getattr(self, "_head_tune_enable", False))
        delta_dim_global = int(getattr(self, "_head_delta_dim", getattr(self.actor, "delta_dim", 0)))
        a1_head_ready = True
        a1_head_source = "disabled"
        if head_tune_global and delta_dim_global > 0:
            a1_head_ready, a1_head_source = self._ensure_headclip_logits(batch)
            if not a1_head_ready:
                self._a1_warn_once(
                    "headclip_prepare_failed",
                    f"clip{a1_head_source}learnPPO",
                )

        a1_clip_total = 0
        a1_clip_used = 0
        a1_ent_total = 0
        a1_ent_used = 0

        for step in range(int(repeat)):
            if getattr(self, "_recompute_adv", False) and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)  # type: ignore[attr-defined]

            kl_epoch_vals: list[float] = []
            
            vf_scale_batch_std = float("nan")
            try:
                returns_all = getattr(batch, "returns", None)
                if isinstance(returns_all, torch.Tensor):
                    vf_scale_batch_std = float(returns_all.detach().std(unbiased=False).cpu().item())
                elif returns_all is not None:
                    _arr = np.asarray(returns_all, dtype=np.float64).reshape(-1)
                    if _arr.size > 0:
                        vf_scale_batch_std = float(np.nanstd(_arr))
            except Exception:
                vf_scale_batch_std = float("nan")

            
            adv_mean_rollout = float("nan")
            adv_std_rollout = float("nan")
            if getattr(self, "_norm_adv", False) and bool(getattr(self, "_adv_norm_rollout_level", False)):
                try:
                    adv_all = getattr(batch, "adv", None)
                    if isinstance(adv_all, torch.Tensor):
                        adv_mean_rollout = float(adv_all.detach().mean().cpu().item())
                        adv_std_rollout = float(adv_all.detach().std(unbiased=False).cpu().item())
                except Exception:
                    pass

            for minibatch in batch.split(int(batch_size), merge_last=True):
                
                dist = self(minibatch).dist

                
                if getattr(self, "_norm_adv", False):
                    if np.isfinite(adv_mean_rollout) and np.isfinite(adv_std_rollout):
                        
                        mean = torch.as_tensor(adv_mean_rollout, dtype=minibatch.adv.dtype, device=minibatch.adv.device)
                        std = torch.as_tensor(adv_std_rollout, dtype=minibatch.adv.dtype, device=minibatch.adv.device)
                    else:
                        
                        mean = minibatch.adv.mean()
                        std = minibatch.adv.std(unbiased=False)
                    std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
                    minibatch.adv = (minibatch.adv - mean) / (std + self._eps)
                    adv_clip = float(max(float(getattr(self, "_adv_norm_clip", 0.0)), 0.0))
                    if bool(getattr(self, "_tail_active", False)):
                        adv_clip = float(
                            max(float(getattr(self, "_adv_norm_clip_tail", adv_clip)), 0.0)
                        )
                    if adv_clip > 0.0:
                        minibatch.adv = torch.clamp(minibatch.adv, min=-adv_clip, max=adv_clip)

                
                logp_new = dist.log_prob(minibatch.act)
                logp_old = minibatch.logp_old

                
                if do_selfcheck and (not selfcheck_done) and int(step) == 0:
                    try:
                        diff = float((logp_new - logp_old).abs().mean().detach().cpu().item())
                    except Exception:
                        diff = float("nan")
                    if (not np.isfinite(diff)) or (diff > float(selfcheck_tol)):
                        raise RuntimeError(
                            f"[SELF-CHECK][FAIL] mean|logp_old-logp_recompute|={diff:.3e} > {float(selfcheck_tol):.3e}"
                            " rollout/update  dist/logp tanhlog-det-jacobian"
                            "updateatanhlogp/"
                        )
                    selfcheck_done = True

                ratio = (logp_new - logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)

                
                with torch.no_grad():
                    kl_val = float(torch.clamp((logp_old - logp_new).mean(), min=0.0).cpu().item())
                if np.isfinite(kl_val):
                    kl_epoch_vals.append(kl_val)

                
                head_tune_enable = bool(getattr(self, "_head_tune_enable", False))
                delta_dim_head = int(getattr(self, "_head_delta_dim", getattr(self.actor, "delta_dim", 0)))
                use_head_clip = False
                clip_loss = None
                if head_tune_enable and delta_dim_head > 0 and a1_head_ready:
                    a1_clip_total += 1
                    try:
                        dist_old = None
                        logits_old = getattr(minibatch, "logits", None)
                        mu_old, sg_old = self._as_logits_pair(logits_old)
                        if (mu_old is None) or (sg_old is None):
                            mu_old = getattr(minibatch, "logits_mu_old", None)
                            sg_old = getattr(minibatch, "logits_sigma_old", None)
                        if (mu_old is not None) and (sg_old is not None):
                            mu_old_t = torch.as_tensor(mu_old, device=minibatch.act.device, dtype=minibatch.act.dtype)
                            sg_old_t = torch.as_tensor(sg_old, device=minibatch.act.device, dtype=minibatch.act.dtype)
                            dist_old = self.dist_fn(mu_old_t, sg_old_t)
                        if dist_old is None:
                            a1_head_ready = False
                            self._a1_warn_once("headclip_logits_missing", "clipPPOminibatch.logits/")
                        if dist_old is not None:
                            lp_new_d, lp_new_s, dd_eff, sd_eff = self._split_log_prob(
                                dist=dist,
                                act=minibatch.act,
                                delta_dim=int(delta_dim_head),
                            )
                            lp_old_d, lp_old_s, dd_old, sd_old = self._split_log_prob(
                                dist=dist_old,
                                act=minibatch.act,
                                delta_dim=int(delta_dim_head),
                            )
                            if dd_eff > 0 and sd_eff > 0 and dd_old == dd_eff and sd_old == sd_eff:
                                eps_d = float(max(float(getattr(self, "_eps_clip_delta", self._eps_clip)), 1e-6))
                                eps_s = float(max(float(getattr(self, "_eps_clip_score", self._eps_clip)), 1e-6))

                                def _clip_obj(ratio_head: torch.Tensor, eps_head: float) -> torch.Tensor:
                                    rr = ratio_head.reshape(ratio_head.size(0), -1).transpose(0, 1)
                                    ss1 = rr * minibatch.adv
                                    ss2 = rr.clamp(1.0 - eps_head, 1.0 + eps_head) * minibatch.adv
                                    if getattr(self, "_dual_clip", None):
                                        c1 = torch.min(ss1, ss2)
                                        c2 = torch.max(c1, self._dual_clip * minibatch.adv)
                                        return -torch.where(minibatch.adv < 0, c2, c1).mean()
                                    return -torch.min(ss1, ss2).mean()

                                ratio_d = (lp_new_d - lp_old_d).exp().float()
                                ratio_s = (lp_new_s - lp_old_s).exp().float()
                                clip_d = _clip_obj(ratio_d, eps_d)
                                clip_s = _clip_obj(ratio_s, eps_s)

                                w_d = float(max(float(getattr(self, "_clip_delta_weight", 0.0)), 0.0))
                                w_s = float(max(float(getattr(self, "_clip_score_weight", 0.0)), 0.0))
                                if (w_d + w_s) <= 1e-12:
                                    den = float(max(int(dd_eff + sd_eff), 1))
                                    w_d = float(dd_eff) / den
                                    w_s = float(sd_eff) / den
                                else:
                                    den = float(max(w_d + w_s, 1e-12))
                                    w_d = float(w_d / den)
                                    w_s = float(w_s / den)
                                clip_loss = float(w_d) * clip_d + float(w_s) * clip_s
                                use_head_clip = True
                    except Exception as exc:
                        a1_head_ready = False
                        self._a1_warn_once("headclip_runtime_failed", f"clipPPO{type(exc).__name__}: {exc}")
                        use_head_clip = False
                elif head_tune_enable and delta_dim_head > 0:
                    a1_clip_total += 1

                if use_head_clip:
                    a1_clip_used += 1

                if not use_head_clip:
                    surr1 = ratio * minibatch.adv
                    surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * minibatch.adv
                    if getattr(self, "_dual_clip", None):
                        clip1 = torch.min(surr1, surr2)
                        clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                        clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                    else:
                        clip_loss = -torch.min(surr1, surr2).mean()

                # critic loss
                value = self.critic(minibatch.obs).flatten()
                
                vf_norm_enable = bool(getattr(self, "_vf_loss_norm_returns", False))
                if bool(getattr(self, "_vf_loss_norm_tail_only", False)) and (not bool(getattr(self, "_tail_active", False))):
                    vf_norm_enable = False
                if vf_norm_enable:
                    vf_scale_floor = float(max(float(getattr(self, "_vf_loss_norm_floor", 1.0)), 1e-6))
                    
                    
                    if np.isfinite(vf_scale_batch_std):
                        vf_scale = torch.as_tensor(vf_scale_batch_std, dtype=value.dtype, device=value.device)
                    else:
                        vf_scale = minibatch.returns.std(unbiased=False).detach()
                    vf_scale = torch.nan_to_num(
                        vf_scale,
                        nan=float(vf_scale_floor),
                        posinf=float(vf_scale_floor),
                        neginf=float(vf_scale_floor),
                    )
                    vf_scale = torch.clamp(vf_scale, min=vf_scale_floor)
                else:
                    vf_scale = torch.as_tensor(1.0, dtype=value.dtype, device=value.device)

                if getattr(self, "_value_clip", False):
                    eps_v = float(max(float(getattr(self, "_eps_clip_value", self._eps_clip)), 1e-6))
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(-eps_v, eps_v)
                    vf1 = ((minibatch.returns - value) / vf_scale).pow(2)
                    vf2 = ((minibatch.returns - v_clip) / vf_scale).pow(2)
                    vf_loss_per_sample = torch.max(vf1, vf2)
                else:
                    vf_loss_per_sample = ((minibatch.returns - value) / vf_scale).pow(2)
                
                vf_loss_clip_max = float(max(float(getattr(self, "_vf_loss_clip_max", 0.0)), 0.0))
                vf_clip_hit_frac = 0.0
                if vf_loss_clip_max > 0.0:
                    
                    vf_clip_hit_frac = float((vf_loss_per_sample > vf_loss_clip_max).float().mean().item())
                    vf_loss_per_sample = torch.clamp(vf_loss_per_sample, max=vf_loss_clip_max)
                vf_loss = vf_loss_per_sample.mean()

                
                ent_loss = dist.entropy().mean()
                ent_term = self._weight_ent * ent_loss
                if head_tune_enable and delta_dim_head > 0:
                    a1_ent_total += 1
                    try:
                        ent_d, ent_s, dd_eff, sd_eff = self._split_entropy(dist=dist, delta_dim=int(delta_dim_head))
                        if dd_eff > 0 and sd_eff > 0:
                            ent_d_m = ent_d.mean()
                            ent_s_m = ent_s.mean()
                            w_ent_d = float(max(float(getattr(self, "_weight_ent_delta", self._weight_ent)), 0.0))
                            w_ent_s = float(max(float(getattr(self, "_weight_ent_score", self._weight_ent)), 0.0))
                            ent_term = float(w_ent_d) * ent_d_m + float(w_ent_s) * ent_s_m
                            ent_loss = float(dd_eff / max(dd_eff + sd_eff, 1)) * ent_d_m + float(
                                sd_eff / max(dd_eff + sd_eff, 1)
                            ) * ent_s_m
                            a1_ent_used += 1
                    except Exception:
                        self._a1_warn_once("headent_runtime_failed", "")
                        ent_term = self._weight_ent * ent_loss

                loss_actor = clip_loss - ent_term
                loss_critic = self._weight_vf * vf_loss

                use_dual_optim = bool(getattr(self, "_dual_optimizer_enabled", False))
                optim_actor = getattr(self, "_optim_actor", None)
                optim_critic = getattr(self, "_optim_critic", None)

                if use_dual_optim and isinstance(optim_actor, torch.optim.Optimizer) and isinstance(optim_critic, torch.optim.Optimizer):
                    
                    optim_actor.zero_grad()
                    loss_actor.backward()
                    gn_actor = float(max(float(getattr(self, "_grad_norm_actor", 0.0)), 0.0))
                    if gn_actor > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=gn_actor)
                    optim_actor.step()

                    optim_critic.zero_grad()
                    loss_critic.backward()
                    gn_critic = float(max(float(getattr(self, "_grad_norm_critic", 0.0)), 0.0))
                    if gn_critic > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=gn_critic)
                    optim_critic.step()

                    loss = loss_actor + loss_critic
                else:
                    
                    loss = loss_actor + loss_critic
                    self.optim.zero_grad()
                    loss.backward()
                    if getattr(self, "_grad_norm", None):
                        torch.nn.utils.clip_grad_norm_(self._actor_critic.parameters(), max_norm=self._grad_norm)
                    self.optim.step()

                clip_losses.append(float(clip_loss.item()))
                vf_losses.append(float(vf_loss.item()))
                ent_losses.append(float(ent_loss.item()))
                losses.append(float(loss.item()))
                vf_clip_hit_fracs.append(float(vf_clip_hit_frac))

            
            if kl_epoch_vals:
                kl_epoch_mean = float(np.mean(kl_epoch_vals))
                approx_kl_history.append(kl_epoch_mean)
                if (self._target_kl_stop > 0.0) and (kl_epoch_mean > self._kl_stop_mult * self._target_kl_stop):
                    break

        out = {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
            "loss/vf_clip_hit_frac": vf_clip_hit_fracs,
        }
        if a1_clip_total > 0:
            out["a1/head_clip_used_ratio"] = [float(a1_clip_used) / float(max(a1_clip_total, 1))]
            out["a1/head_clip_ready"] = [1.0 if bool(a1_head_ready) else 0.0]
        if a1_ent_total > 0:
            out["a1/head_ent_used_ratio"] = [float(a1_ent_used) / float(max(a1_ent_total, 1))]
        
        if approx_kl_history:
            out["kl/epoch"] = approx_kl_history
        return out


def create_ppo_policy(
    obs_dim: int,
    act_dim: int,
    
    delta_dim: Optional[int] = None,
    
    logp_act_dim: Optional[int] = None,
    
    
    
    logp_mode: str = "sum",
    
    hidden_sizes: Sequence[int] = (256, 256),
    use_layernorm: bool = True,
    activation: torch.nn.Module = torch.nn.ReLU,
    
    lr: float = 1e-4,
    
    lr_critic: Optional[float] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    eps_clip: float = 0.2,
    
    target_kl: float = 0.0,
    kl_stop_mult: float = 1.5,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    
    max_grad_norm_critic: Optional[float] = None,
    
    vf_loss_clip_max: float = 0.0,
    
    dual_optimizer: Optional[bool] = None,
    
    max_action: float = 1.0,
    device: Union[str, torch.device] = "cuda",
    unbounded: bool = False,
    conditioned_sigma: bool = False,
    share_preprocess: bool = True,
    
    dual_clip: float = None,
    value_clip: bool = False,
    advantage_normalization: bool = True,
    recompute_advantage: bool = False,
    
    reward_normalization: bool = False,
    max_batchsize: int = 256,
    
    deterministic_eval: bool = True,
) -> Tuple[PPOPolicy, OptimizerBundle]:
    """
    Tianshou PPO Policy
    
    PPO Policy
    - Actor-Critic
    - 
    - PPO
    
    
        obs_dim: 
        act_dim: 
        hidden_sizes: 
        use_layernorm: LayerNorm
        activation: 
        lr: 
        gamma: 
        gae_lambda: GAE lambda
        eps_clip: PPO
        vf_coef: 
        ent_coef: 
        max_grad_norm: 
        max_action: 
        device: 
        unbounded: 
        conditioned_sigma: 
        share_preprocess: Actor/Critic
        dual_clip: 
        value_clip: 
        advantage_normalization: 
        recompute_advantage: 
         
    
        (policy, optimizer): PPO Policy
    """
    
    actor, critic = create_actor_critic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        delta_dim=delta_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_layernorm=use_layernorm,
        max_action=max_action,
        device=device,
        unbounded=unbounded,
        conditioned_sigma=conditioned_sigma,
        share_preprocess=bool(share_preprocess),
    )
    
    use_dual_optimizer = bool(dual_optimizer) if dual_optimizer is not None else (lr_critic is not None)
    if use_dual_optimizer and bool(share_preprocess):
        
        print("[HB][ppo][T8][WARN] share_preprocess=true ", flush=True)
        use_dual_optimizer = False

    optimizer: OptimizerBundle
    optimizer_for_policy: torch.optim.Optimizer
    optimizer_critic: Optional[torch.optim.Optimizer] = None
    lr_critic_now = float(lr_critic) if lr_critic is not None else float(lr * 3.0)
    if use_dual_optimizer:
        optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
        optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic_now)
        optimizer = (optimizer_actor, optimizer_critic)
        optimizer_for_policy = optimizer_actor
    else:
        
        params = list(actor.parameters()) + list(critic.parameters())
        params_dict = {id(p): p for p in params}
        params = list(params_dict.values())
        optimizer_single = torch.optim.Adam(params, lr=lr)
        optimizer = optimizer_single
        optimizer_for_policy = optimizer_single
    
    
    logp_mode_n = str(logp_mode or "sum").strip().lower()
    if logp_mode_n not in ("sum", "mean"):
        raise ValueError(f"logp_mode{logp_mode}sum/mean")

    
    logp_dim = int(logp_act_dim) if logp_act_dim is not None else int(act_dim)
    logp_dim = int(max(1, min(int(logp_dim), int(act_dim))))

    
    
    logp_scale = 1.0 / float(max(int(logp_dim), 1)) if logp_mode_n == "mean" else 1.0

    class _ScaledDist(torch.distributions.Distribution):
        """log_prob/entropyshape"""

        
        arg_constraints = {}

        has_rsample = True

        def __init__(self, base: torch.distributions.Distribution, scale: float):
            super().__init__(batch_shape=base.batch_shape, event_shape=base.event_shape)
            self.base = base
            self.scale = float(scale)
            try:
                self.has_rsample = bool(getattr(base, "has_rsample", True))
            except Exception:
                self.has_rsample = True

        def sample(self, sample_shape: torch.Size = torch.Size()):
            return self.base.sample(sample_shape)

        def rsample(self, sample_shape: torch.Size = torch.Size()):
            if hasattr(self.base, "rsample"):
                return self.base.rsample(sample_shape)  # type: ignore[attr-defined]
            return self.base.sample(sample_shape)

        def log_prob(self, value):
            return self.base.log_prob(value) * self.scale

        def entropy(self):
            return self.base.entropy() * self.scale

        def __getattr__(self, name: str):
            return getattr(self.base, name)

    def dist_fn(loc, scale):
        """
        Tianshouactortupledist_fn*logits

        
        - PGPolicy.forward  dist_fn(*logits) (mu, sigma) 
        -  dist_fn((mu, sigma)) Collector
        """
        
        base = SquashedGaussian(loc, scale, act_dim_effective=int(logp_dim))
        if logp_scale != 1.0:
            return _ScaledDist(base, logp_scale)
        return base
    
    
    policy = PPOPolicyWithKLStop(
        actor=actor,
        critic=critic,
        optim=optimizer_for_policy,
        optim_critic=optimizer_critic,
        dual_optimizer_enabled=bool(use_dual_optimizer),
        grad_norm_actor=float(max_grad_norm),
        grad_norm_critic=(float(max_grad_norm_critic) if max_grad_norm_critic is not None else float(max_grad_norm)),
        vf_loss_clip_max=float(vf_loss_clip_max),
        dist_fn=dist_fn,
        
        discount_factor=gamma,
        gae_lambda=gae_lambda,
        eps_clip=eps_clip,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        target_kl=float(target_kl),
        kl_stop_mult=float(kl_stop_mult),
        
        dual_clip=dual_clip,
        value_clip=value_clip,
        advantage_normalization=advantage_normalization,
        recompute_advantage=recompute_advantage,
        
        reward_normalization=bool(reward_normalization),
        max_batchsize=int(max_batchsize),
        
        action_space=None,  
        action_scaling=True,  
        
        action_bound_method="",  
        deterministic_eval=bool(deterministic_eval),
    )

    
    try:
        setattr(policy, "_logp_mode", str(logp_mode_n))
        setattr(policy, "_logp_scale", float(logp_scale))
        setattr(policy, "_act_dim", int(act_dim))
        setattr(policy, "_logp_act_dim", int(logp_dim))
        setattr(policy, "_dual_optimizer_enabled", bool(use_dual_optimizer))
        setattr(policy, "_grad_norm_actor", float(max_grad_norm))
        setattr(
            policy,
            "_grad_norm_critic",
            float(max_grad_norm_critic) if max_grad_norm_critic is not None else float(max_grad_norm),
        )
        setattr(policy, "_vf_loss_clip_max", float(max(vf_loss_clip_max, 0.0)))
        lr_base = float(max(lr, 1e-12))
        ratio = float(lr_critic_now / lr_base) if bool(use_dual_optimizer) else 1.0
        setattr(policy, "_lr_critic_ratio", float(max(ratio, 1e-6)))
    except Exception:
        pass
    
    return policy, optimizer


def create_ppo_policy_from_config(
    obs_dim: int,
    act_dim: int,
    config: dict,
    device: Union[str, torch.device] = "cuda",
) -> Tuple[PPOPolicy, OptimizerBundle]:
    """
    PPO Policy
    
    
        obs_dim: 
        act_dim: 
        config: 
        device: 
         
    
        (policy, optimizer): PPO Policy
    """
    return create_ppo_policy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        logp_mode=str(config.get("logp_mode", "sum")),
        hidden_sizes=tuple(config.get('hidden_sizes', [256, 256])),
        use_layernorm=config.get('use_layernorm', True),
        lr=config.get('lr', 1e-4),
        lr_critic=config.get("lr_critic", None),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        eps_clip=config.get('eps_clip', 0.2),
        target_kl=float(config.get("target_kl", 0.0)),
        kl_stop_mult=float(config.get("kl_stop_mult", 1.5)),
        vf_coef=config.get('vf_coef', 0.5),
        ent_coef=config.get('ent_coef', 0.01),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        max_grad_norm_critic=config.get("max_grad_norm_critic", None),
        vf_loss_clip_max=config.get("vf_loss_clip_max", 0.0),
        dual_optimizer=config.get("dual_optimizer", None),
        max_action=config.get('max_action', 1.0),
        device=device,
        unbounded=config.get('unbounded', False),
        conditioned_sigma=config.get('conditioned_sigma', False),
        share_preprocess=bool(config.get("share_preprocess", True)),
        dual_clip=config.get('dual_clip', None),
        value_clip=config.get('value_clip', False),
        advantage_normalization=config.get('advantage_normalization', True),
        recompute_advantage=config.get('recompute_advantage', False),
        reward_normalization=bool(config.get("reward_normalization", config.get("value_normalization", False))),
        max_batchsize=int(config.get("max_batchsize", 256)),
        deterministic_eval=bool(config.get("deterministic_eval", True)),
    )
