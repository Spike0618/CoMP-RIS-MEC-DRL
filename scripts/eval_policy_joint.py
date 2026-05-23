"""
 +  + 


- variantsfull / no_ris / no_comp
- methodsppo / balanced / greedy_delay / greedy_energy / myopic_optimization /
  always_comp / never_comp / ddpg / sac / td3 / a2c

 run 
- `<train_run_dir>/evals/<eval_ts>/eval_raw_<eval_ts>.csv`
- `<train_run_dir>/evals/<eval_ts>/eval_summary_<eval_ts>.json`
- `<train_run_dir>/evals/<eval_ts>/eval_meta_<eval_ts>.json`
- `<train_run_dir>/evals/<eval_ts>/trajectory_metrics_summary.csv`
- `<train_run_dir>/figs/<eval_ts> evalratio/Ablation10_PaperCost.png`
- `<train_run_dir>/figs/<eval_ts> evalratio/Ablation10_Delay.png`
- `<train_run_dir>/figs/<eval_ts> evalratio/Ablation10_Energy.png`
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_PaperCost.png`10
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_Delay.png`10
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_Energy.png`10
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_FullOnly_PaperCost.png` full
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_FullOnly_Delay.png` full
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_FullOnly_Energy.png` full
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_FullOnly_TaskD_PaperCost.png` full D
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_FullOnly_TaskD_Delay.png` full D
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_FullOnly_TaskD_Energy.png` full D
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_VariantBars_PaperCost.png`
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_VariantBars_Delay.png`
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_VariantBars_Energy.png`
- `<train_run_dir>/figs/Ablation_10Loads eval/Ablation10_Ct.png``C(t)`  full
- `<train_run_dir>/figs/Ablation_10Loads_Insights eval5/Ablation10_HighLoadCDF_Delay.png`
- `<train_run_dir>/figs/Ablation_10Loads_Insights eval5/Ablation10_GainDecomp_PaperCost.png`CoMP/RIS
- `<train_run_dir>/figs/Ablation_10Loads_Insights eval5/Ablation10_Pareto_DelayEnergy.png`-
- `<train_run_dir>/figs/Ablation_10Loads_Insights eval5/Ablation10_Behavior_vs_C.png`
- `<train_run_dir>/figs/Ablation_10Loads_Insights eval5/Ablation10_Generalization_Degradation.png`
CPU `C` `load_scale`  Mcycles
 `D`Task size MB full-only 


- RUN_DIR / OUT_DIR run_dir
- CKPT_PATH PPO checkpoint
- ENV_CFG / TRAIN_CFG yaml  run_dir 
- EVAL_METHODS
  `ppo,balanced,greedy_delay,greedy_energy,myopic_optimization,always_comp,never_comp,ddpg,sac,td3,a2c`
- DDPG_CKPT / SAC_CKPT / TD3_CKPT / A2C_CKPT ckpt
- EVAL_DEVICEcpu / cuda / cuda:0
- EVAL_SEED0, EVAL_NSEEDS
- EVAL_LOADS `0.1,0.2,...,1.0` `1.0`
- EVAL_STRICT_TEN_LOADS 0.1~1.0  1fixed-eval 
- EVAL_REQUIRE_EXPLICIT_CKPT PPO ckpt 1 best/final 
- EVAL_EP_PER episode  5
- EVAL_EP_PER_PROGRESSIVE 1
- EVAL_DETERMINISTICTrue/FalsePPO/3DRL 
- EVAL_HB_EVERY
- EVAL_TS eval_ts
- EVAL_PLOT_CI1/0 0 eval  CI95 
- EVAL_PLOT_LOGY1/0 1 log-y 
- EVAL_PLOT_ZOOM1/0 1 zoom 
- EVAL_PLOT_ZOOM_Qzoom  0.85 [0.50, 0.99]
- EVAL_RESUME1/0 1 eval_raw_<eval_ts>.csv 
- EVAL_FAIL_ON_METHOD_DROP1/0 1ckpt/
- EVAL_REQUIRE_CLEAN_OUTPUT1/0 1 eval_ts 
- EVAL_FORCE_OVERWRITE1/0 0 eval_ts  eval/fig 
- EVAL_RESUME_STRICT_GUARD1/0 1 raw //
- EVAL_FAIL_ON_INCOMPLETE_RAW1/0 1 raw /


-  run_dir
- PPO checkpoint  `checkpoints/` `ckpts/``ckpt/`
"""

from __future__ import annotations

import sys
import os
import re
import csv
import json
import time
import shutil
import argparse
from dataclasses import replace as dc_replace
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Tuple, Optional, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# -------------------------

# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


CKPT_DIR_NAME = "checkpoints"
_WARN_ONCE_KEYS: set[str] = set()
FIG_EVAL_TS_SUFFIX = " evalratio"
FIG_BUCKET_FIXED_EVAL = "FixedEval_PaperCost evalpaper_cost"
FIG_BUCKET_ABLATION10 = "Ablation_10Loads eval"
FIG_BUCKET_ABLATION10_INSIGHTS = "Ablation_10Loads_Insights eval5"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import yaml
except Exception as e:
    raise RuntimeError("Missing dependency: pyyaml. Please `pip install pyyaml` in your env.") from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _warn_once(key: str, message: str) -> None:
    """"""
    k = str(key)
    if k in _WARN_ONCE_KEYS:
        return
    _WARN_ONCE_KEYS.add(k)
    print(f"[HB][eval][WARN][{k}] {message}", flush=True)

# =========================

# =========================
def _apply_ieee_tcom_style() -> None:
    import matplotlib as mpl
    import logging
    from matplotlib import font_manager
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.dpi": 600,

        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "axes.unicode_minus": False,

        "lines.linewidth": 1.8,
        "lines.markersize": 6.0,

        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "legend.borderpad": 0.35,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.6,
    })

    preferred_fonts = [
        "Microsoft YaHei",
        "SimSun",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    picked = [x for x in preferred_fonts if x in installed]
    mpl.rcParams["font.family"] = picked if picked else ["DejaVu Sans"]


def _beautify_ax(ax) -> None:
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.grid(True, which="major")
    ax.grid(False, which="minor")
    ax.tick_params(top=True, right=True)
    ax.margins(x=0.02)


_TCOM_VARIANT_ORDER = ["full", "no_ris", "no_comp"]

_TCOM_METHOD_STYLE: Dict[str, Dict[str, Any]] = {
    "ppo":          {"color": "#1CA9A6", "marker": "x", "label": "ARIA", "linestyle": "-"},  # teal
    "balanced":     {"color": "#1F77B4", "marker": "*", "label": "",     "linestyle": "-"},  # blue
    "greedy_delay": {"color": "#F2B632", "marker": "o", "label": "",     "linestyle": "-"},  # gold
    "greedy_energy":{"color": "#F08A24", "marker": "^", "label": "",     "linestyle": "-"},  # orange
    "myopic_optimization": {"color": "#7B68EE", "marker": ">", "label": "", "linestyle": "-"},
    "always_comp":  {"color": "#B22222", "marker": "s", "label": "",     "linestyle": "--"}, # firebrick
    "never_comp":   {"color": "#555555", "marker": "h", "label": "",       "linestyle": "--"}, # gray
    "ddpg": {"color": "#4DA3FF", "marker": "d", "label": "DDPG", "linestyle": "-"},  # light blue
    "sac":  {"color": "#FF9E6D", "marker": "v", "label": "SAC",  "linestyle": "-"},  # salmon-orange
    "td3":  {"color": "#C9A227", "marker": "p", "label": "TD3",  "linestyle": "-"},  # mustard
    "a2c":  {"color": "#6A994E", "marker": "8", "label": "A2C",  "linestyle": "-"},  # green
}
_ALLOWED_EVAL_METHODS: Tuple[str, ...] = (
    "ppo",
    "balanced",
    "greedy_delay",
    "greedy_energy",
    "myopic_optimization",
    "always_comp",
    "never_comp",
    "ddpg",
    "sac",
    "td3",
    "a2c",
)
_apply_ieee_tcom_style()


def _infer_ppo_display_label_from_run_dir(train_run_dir: Path) -> str:
    """
     run_dir  PPO /summary method 
    
    1)  PPO_LABEL
    2) run_dir 
    3)  ARIA
    """
    env_label = os.environ.get("PPO_LABEL", "").strip()
    if env_label:
        return env_label

    name = str(train_run_dir.name).strip().lower()
    if "amcp_ppo" in name or "amcp-ppo" in name or name.startswith("amcp"):
        if "llmmeta" in name:
            return "AMCP-PPO (LLMMeta)"
        if "heuristicmeta" in name:
            return "AMCP-PPO (HeuristicMeta)"
        if "randommeta" in name:
            return "AMCP-PPO (RandomMeta)"
        return "AMCP-PPO"
    return "ARIA"


def _variant_display_label(variant: str) -> str:
    mapping = {
        "full": "",
        "no_ris": "RIS",
        "no_comp": "CoMP",
    }
    return str(mapping.get(str(variant).strip().lower(), str(variant)))


def _cpu_axis_label() -> str:
    return " CPU  C (Mcycles)"


def _method_display_label(method: str) -> str:
    st = _TCOM_METHOD_STYLE.get(str(method), {})
    lb = st.get("label", method)
    return str(lb)

# -------------------------

# -------------------------
from src.envs.comp_ris_env_joint import JointEnvConfig, CompRISEnvJoint
from src.algos.baseline import baseline_action
from src.utils.obs_normalizer import ObsNormalizer
from src.utils.env_cfg import build_joint_env_cfg
from src.utils.phaseg_summary import upsert_phaseg_eval_summary
from src.utils.seed import set_global_seed

# -------------------------

# -------------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _parse_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "t", "on"):
        return True
    if s in ("0", "false", "no", "n", "f", "off"):
        return False
    return default


def _parse_int(x: Any, default: int) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return int(default)

def _parse_loads(x: Optional[str]) -> List[float]:
    if not x:
        return [0.1 * i for i in range(1, 11)]
    out: List[float] = []
    for t in str(x).split(","):
        t = t.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def _build_ep_per_by_load(
    loads: Sequence[float],
    base_ep_per: int,
    *,
    progressive: bool = True,
) -> Dict[float, int]:
    """
    

    
    - progressive=False base_ep_per
    - progressive=True
       base=5  [3,3,4,4,5,5,6,6,7,7]
    """
    arr = np.asarray(list(loads), dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return {}

    base = int(max(1, int(base_ep_per)))
    uniq = np.asarray(sorted({float(v) for v in arr.tolist()}), dtype=np.float64)
    if uniq.size <= 1 or (not bool(progressive)):
        return {float(v): int(base) for v in uniq.tolist()}

    lo = int(max(1, base - 2))
    hi = int(max(lo, base + 2))
    out: Dict[float, int] = {}
    n = int(uniq.size)
    for i, ld in enumerate(uniq.tolist()):
        frac = float(i) / float(max(1, n - 1))
        ep = int(round(float(lo) + frac * float(hi - lo)))
        out[float(ld)] = int(max(1, ep))
    return out


def _stable_eval_seed(parts: Sequence[Any]) -> int:
    """
     Python hash 

    
    - deterministic=False
    -  resume  (variant/method/load/seed/ep) 
    """
    text = "|".join(str(x) for x in parts)
    h = 2166136261  # FNV-1a 32-bit offset basis
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int((h % 2147483646) + 1)


def _fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:d}m{s:02d}s"


def _load_key(x: Any) -> float:
    """
     0.3  0.30000000000000004 
    """
    v = _to_float(x, default=float("nan"))
    if not np.isfinite(v):
        return float("nan")
    return float(round(float(v), 6))


def _publish_key_plot(train_run_dir: Path, src_path: Path, bucket_name: str, *, dst_name: str) -> None:
    if not src_path.exists():
        return
    dst_dir = train_run_dir / "figs" / str(bucket_name)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / str(dst_name)
    shutil.copy2(str(src_path), str(dst_file))


def _is_ten_load_protocol(loads: Sequence[float]) -> bool:
    """ 0.1~1.0 """
    arr = np.asarray(list(loads), dtype=np.float64).reshape(-1)
    if arr.size != 10:
        return False
    target = np.asarray([0.1 * i for i in range(1, 11)], dtype=np.float64)
    return bool(np.allclose(arr, target, atol=1e-9, rtol=0.0))


def _should_publish_ablation10(eval_ts: str, loads: Sequence[float]) -> bool:
    """
    Ablation_10Loads eval
    -  0.1~1.0 
    - *_fixed / *_random /  gen
    """
    if not _is_ten_load_protocol(loads):
        return False
    ts = str(eval_ts).strip().lower()
    if ts.endswith("_fixed") or ts.endswith("_random"):
        return False
    if "gen" in ts:
        return False
    return True


def _to_float(x: Any, default: float = float("nan")) -> float:
    """ float default"""
    try:
        v = float(x)
    except Exception:
        return float(default)
    return float(v) if np.isfinite(v) else float(default)


def _pick_first_finite(src: Dict[str, Any], keys: Sequence[str], default: float = float("nan")) -> float:
    """"""
    for k in keys:
        if k not in src:
            continue
        v = _to_float(src.get(k), default=float("nan"))
        if np.isfinite(v):
            return float(v)
    return float(default)


def _acc_finite(sum_val: float, cnt_val: int, x: Any) -> Tuple[float, int]:
    """ NaN """
    v = _to_float(x, default=float("nan"))
    if not np.isfinite(v):
        return float(sum_val), int(cnt_val)
    return float(sum_val + v), int(cnt_val + 1)


def _progress_bar(k: int, total: int, width: int = 26) -> str:
    total = max(1, int(total))
    k = max(0, min(int(k), total))
    filled = int(width * (k / total))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _torch_load(path: Path, *, map_location: Any = "cpu") -> Any:
    """
     PyTorch  checkpoint 

    
    -  torch  weights_only  True FutureWarning
    -  weights_only=True ckpt 
    """
    path = Path(path)
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        
        return torch.load(str(path), map_location=map_location)
    except Exception:
        try:
            return torch.load(str(path), map_location=map_location, weights_only=False)  # type: ignore[call-arg]
        except TypeError:
            return torch.load(str(path), map_location=map_location)


def _select_cfg_paths(run_dir: Path) -> Tuple[Path, Path]:
    env_over = os.environ.get("ENV_CFG", "").strip()
    train_over = os.environ.get("TRAIN_CFG", "").strip()

    if env_over:
        env_p = Path(env_over)
        if not env_p.is_absolute():
            env_p = (REPO_ROOT / env_p).resolve()
    else:
        
        p0 = run_dir / "configs" / "env_fixed.yaml"
        p1 = run_dir / "env_fixed.yaml"
        env_p = p0 if p0.exists() else (p1 if p1.exists() else (REPO_ROOT / "configs" / "env_fixed.yaml"))

    if train_over:
        tr_p = Path(train_over)
        if not tr_p.is_absolute():
            tr_p = (REPO_ROOT / tr_p).resolve()
    else:
        p0 = run_dir / "configs" / "train_tianshou_fixed.yaml"
        p1 = run_dir / "train_tianshou_fixed.yaml"
        tr_p = p0 if p0.exists() else (p1 if p1.exists() else (REPO_ROOT / "configs" / "train_tianshou_fixed.yaml"))

    if not env_p.exists():
        raise FileNotFoundError(f"env_fixed.yaml not found: {env_p}")
    if not tr_p.exists():
        raise FileNotFoundError(f"train_tianshou_fixed.yaml not found: {tr_p}")
    return env_p, tr_p


def _find_latest_run_dir(base: Path) -> Path:
    """
     runs/paper  run 

    
    - YYYYMMDD-HHMMSS
    - 
    """
    base = base.resolve()
    if not base.exists():
        raise FileNotFoundError(f"runs base not found: {base}")

    ptr = base / "_latest_run.txt"
    pat = re.compile(r"^\d{8}-\d{6}$")

    def _valid_run(p: Path) -> bool:
        return p.exists() and p.is_dir()

    if ptr.exists():
        name = ptr.read_text(encoding="utf-8").strip()
        cand = (base / name).resolve()
        if _valid_run(cand):
            return cand

    cands = [p for p in base.iterdir() if p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"No run dirs under: {base}")

    ts_cands = [p for p in cands if pat.match(p.name)]
    if ts_cands:
        ts_cands.sort(key=lambda p: p.name)
        return ts_cands[-1]

    
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def _select_run_dir() -> Path:
    rd = (os.environ.get("RUN_DIR") or os.environ.get("OUT_DIR") or "").strip()
    if rd:
        p = Path(rd)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"RUN_DIR/OUT_DIR not found: {p}")
        return p

    base = (REPO_ROOT / "runs" / "paper").resolve()
    return _find_latest_run_dir(base)


def build_env_cfg(cfg_env: dict) -> JointEnvConfig:
    
    
    env_cfg, env_cfg_dict_clean, dropped = build_joint_env_cfg(cfg_env, overrides={"association_mode": "hard"})
    print(f"[HB][eval] env_cfg keys: kept={len(env_cfg_dict_clean)} dropped={len(dropped)}", flush=True)
    if dropped:
        print(f"[HB][eval] dropped keys (first 20): {dropped[:20]}", flush=True)
    
    if "action_space_mode" in set(dropped):
        raise RuntimeError(
            "[HB][eval][FATAL] env config key 'action_space_mode' was dropped by build_joint_env_cfg. "
            ""
        )
    return env_cfg


def make_variants(base_cfg: JointEnvConfig) -> Dict[str, JointEnvConfig]:
    full = base_cfg
    
    
    no_ris = dc_replace(
        base_cfg,
        enable_ris=False,
        eta=0.0,
        N=0,
        obs_enable_comp_flag=1.0,
        obs_enable_ris_flag=1.0,
    )
    # Keep variant flags visible as full in observation to avoid OOD input shift.
    # Ablation should isolate mechanism removal only.
    no_comp = dc_replace(
        base_cfg,
        enable_comp=False,
        obs_enable_comp_flag=1.0,
        obs_enable_ris_flag=1.0,
    )
    return {"full": full, "no_ris": no_ris, "no_comp": no_comp}


def _ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n <= 1:
        return 0.0
    return float(1.96 * np.std(x, ddof=1) / np.sqrt(n))


def _ensure_csv_header(path: Path, cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if not header:
            raise RuntimeError(f"[HB][eval][FATAL] raw csv header missing: {path}")
        got = [str(x).strip() for x in header]
        exp = [str(x).strip() for x in cols]
        if got != exp:
            raise RuntimeError(
                "[HB][eval][FATAL] raw csv header mismatch"
                f" expected={exp} got={got} file={path}"
            )
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)


def _append_csv_row(path: Path, cols: List[str], row: Dict[str, Any]) -> None:
    
    max_retry = 12
    for i in range(max_retry):
        try:
            with open(path, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow([row.get(c, "") for c in cols])
                f.flush()
            return
        except PermissionError:
            if i >= max_retry - 1:
                raise
            time.sleep(0.05 * float(i + 1))


def _load_done_keys(path: Path) -> set:
    """
     done keyvariant|method|load|seed|ep

    load  6 
    """
    done = set()
    if (not path.exists()) or path.stat().st_size == 0:
        return done
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for rr in reader:
                try:
                    v = str(rr["variant"])
                    m = str(rr["method"])
                    ld = float(rr["load"])
                    sd = int(float(rr["seed"]))
                    ep = int(float(rr["ep"]))
                    done.add(f"{v}|{m}|{ld:.6f}|{sd}|{ep}")
                except Exception:
                    continue
    except Exception:
        return done
    return done


def _scan_raw_scope(path: Path) -> Dict[str, Any]:
    """
     raw csv  resume 
    """
    out = {
        "rows": 0,
        "methods": set(),
        "variants": set(),
        "loads": set(),
        "bad_rows": 0,
    }
    if (not path.exists()) or path.stat().st_size <= 0:
        return out
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for rr in reader:
                try:
                    v = str(rr.get("variant", "")).strip()
                    m = str(rr.get("method", "")).strip()
                    ld = float(rr.get("load", float("nan")))
                    if (not v) or (not m) or (not np.isfinite(ld)):
                        out["bad_rows"] = int(out["bad_rows"]) + 1
                        continue
                    out["rows"] = int(out["rows"]) + 1
                    out["variants"].add(v)
                    out["methods"].add(m)
                    out["loads"].add(f"{ld:.6f}")
                except Exception:
                    out["bad_rows"] = int(out["bad_rows"]) + 1
    except Exception as e:
        raise RuntimeError(f"[HB][eval][FATAL] failed to scan raw csv: {path} ({type(e).__name__}: {e})") from e
    return out


def _clear_eval_outputs(eval_dir: Path, fig_dir: Path) -> None:
    """
     eval_ts 
    """
    for d in (eval_dir, fig_dir):
        if d.exists():
            shutil.rmtree(d, ignore_errors=False)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)


def _fairify_action(env: CompRISEnvJoint, act: Any) -> np.ndarray:
    """
    /
    -  float32 
    - pad/trim  env.action_dim
    -  NaN/Inf  0
    """
    a = np.asarray(act, dtype=np.float64).reshape(-1)
    need = int(env.action_dim)
    if a.size != need:
        a2 = np.zeros((need,), dtype=np.float64)
        n = min(a.size, need)
        a2[:n] = a[:n]
        a = a2
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a.astype(np.float32)



_EVAL_OBS_NORMALIZER: Optional[ObsNormalizer] = None


def _maybe_load_obs_normalizer(train_run_dir: Path, ckpt_path: Path, obs_dim: int, clip: float = 10.0) -> Optional[ObsNormalizer]:
    """
     ObsNormalizer

     checkpoint 
    - ckpt_best   normalizer_best normalizer_final latest 
    - ckpt_final  normalizer_final latest
    - ckpt_stepX  normalizer_stepX/Xstep final/latest
    -     normalizer_<tag> latest/final
    """
    cand_dirs = []
    d0 = train_run_dir / CKPT_DIR_NAME  
    d1 = train_run_dir / "ckpts"
    d2 = train_run_dir / "ckpt"
    if d0.exists():
        cand_dirs.append(d0)
    if d1.exists():
        cand_dirs.append(d1)
    if d2.exists():
        cand_dirs.append(d2)

    ckpt_name = ckpt_path.name.lower()
    is_best = "best" in ckpt_name
    is_final = "final" in ckpt_name

    cand_files: List[Path] = []
    seen_files: set[str] = set()

    def _add_candidate(p: Path) -> None:
        try:
            k = str(p.resolve())
        except Exception:
            k = str(p)
        if (k not in seen_files) and p.exists() and p.is_file():
            seen_files.add(k)
            cand_files.append(p)

    
    if is_best:
        for d in cand_dirs:
            _add_candidate(d / "normalizer_best.npz")

    
    step_target: Optional[int] = None
    m = re.search(r"step[_-]inf(\d+)", ckpt_path.name, flags=re.IGNORECASE)
    if m:
        step_target = int(m.group(1))
        for d in cand_dirs:
            _add_candidate(d / f"normalizer_step{int(step_target)}.npz")
        step_re = re.compile(r"normalizer_step(\d+)\.npz$", re.IGNORECASE)
        step_cands: List[Tuple[int, Path]] = []
        for d in cand_dirs:
            for q in d.glob("normalizer_step*.npz"):
                m2 = step_re.search(q.name)
                if not m2:
                    continue
                try:
                    step_cands.append((int(m2.group(1)), q))
                except Exception:
                    continue
        if step_cands:
            le = [kv for kv in step_cands if int(kv[0]) <= int(step_target)]
            if le:
                _add_candidate(sorted(le, key=lambda kv: kv[0])[-1][1])
            else:
                _add_candidate(sorted(step_cands, key=lambda kv: abs(int(kv[0]) - int(step_target)))[0][1])

    
    tag = ckpt_path.stem.lower()
    for pref in ("ckpt_", "agent_"):
        if tag.startswith(pref) and len(tag) > len(pref):
            tag = tag[len(pref):]
            break
    if tag and (not is_best) and (step_target is None):
        for d in cand_dirs:
            _add_candidate(d / f"normalizer_{tag}.npz")

    
    if is_best:
        
        for d in cand_dirs:
            _add_candidate(d / "normalizer_final.npz")
    elif is_final:
        for d in cand_dirs:
            _add_candidate(d / "normalizer_final.npz")
        for d in cand_dirs:
            _add_candidate(d / "normalizer_latest.npz")
    elif step_target is not None:
        for d in cand_dirs:
            _add_candidate(d / "normalizer_final.npz")
        for d in cand_dirs:
            _add_candidate(d / "normalizer_latest.npz")
    else:
        for d in cand_dirs:
            _add_candidate(d / "normalizer_latest.npz")
        for d in cand_dirs:
            _add_candidate(d / "normalizer_final.npz")

    for norm_path in cand_files:
        try:
            data = np.load(norm_path, allow_pickle=False)
            norm = ObsNormalizer(obs_dim=obs_dim, clip=float(clip))
            norm.set_stats({
                "count": int(data["count"]),
                "mean": data["mean"],
                "var": data["var"],
            })
            print(f"[HB][eval] loaded obs_normalizer: {norm_path} count={int(data['count'])}", flush=True)
            return norm
        except Exception as exc:
            _warn_once(
                f"eval_load_normalizer::{norm_path.name}",
                f"normalizer{type(exc).__name__}: {exc}",
            )

    return None


def _select_ppo_ckpt(train_run_dir: Path) -> Path:
    """
     run_dir  PPO checkpoint checkpoints/ckpts/ckpt/

    
    1)  CKPT_PATH 
    2) ckpt_best.pt
    3) ckpt_final.pt
    4) agent_final.pt
    5)  ckpt_*.pt
    6)  agent_step*.pt
    7) ckpt_latest.pt
    8)  *.pt
    """
    ckpt_over = os.environ.get("CKPT_PATH", "").strip()
    if ckpt_over:
        p = Path(ckpt_over)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"CKPT_PATH not found: {p}")
        return p

    cand_dirs = []
    for d in (train_run_dir / CKPT_DIR_NAME, train_run_dir / "ckpts", train_run_dir / "ckpt"):
        if d.exists():
            cand_dirs.append(d)
    if not cand_dirs:
        raise FileNotFoundError(f"No ckpt dir found under: {train_run_dir} (expected {CKPT_DIR_NAME}/, ckpts/, or ckpt/)")

    def _step_num(p: Path) -> int:
        m = re.search(r"step[_-]inf(\d+)", p.name, flags=re.IGNORECASE)
        return int(m.group(1)) if m else -1

    
    for d in cand_dirs:
        p = d / "ckpt_best.pt"
        if p.exists():
            return p

    
    for d in cand_dirs:
        p = d / "ckpt_final.pt"
        if p.exists():
            return p

    
    for d in cand_dirs:
        p = d / "agent_final.pt"
        if p.exists():
            return p

    
    pts = []
    for d in cand_dirs:
        pts.extend(list(d.glob("ckpt_*.pt")))
    pts.sort(key=lambda p: (_step_num(p), p.name))
    if pts:
        return pts[-1]

    
    pts = []
    for d in cand_dirs:
        pts.extend(list(d.glob("agent_step*.pt")))
    pts.sort(key=lambda p: (_step_num(p), p.name))
    if pts:
        return pts[-1]

    
    for d in cand_dirs:
        p = d / "ckpt_latest.pt"
        if p.exists():
            return p

    
    any_pts = []
    for d in cand_dirs:
        any_pts.extend(list(d.glob("*.pt")))
    any_pts.sort(key=lambda p: (_step_num(p), p.name))
    if any_pts:
        return any_pts[-1]

    raise FileNotFoundError(f"No .pt checkpoints found under: {cand_dirs}")


class _A2CActorCriticEval(nn.Module):
    """A2Ctanh + """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(int(obs_dim), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(hidden))
        self.mean_head = nn.Linear(int(hidden), int(act_dim))
        self.value_head = nn.Linear(int(hidden), 1)
        self.log_std = nn.Parameter(torch.zeros(int(act_dim), dtype=torch.float32))

    def _forward_feat(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._forward_feat(obs)
        mean = self.mean_head(feat)
        value = self.value_head(feat)
        log_std = torch.clamp(self.log_std, -5.0, 2.0).unsqueeze(0).expand_as(mean)
        return mean, log_std, value

    def sample_action(self, obs: torch.Tensor, deterministic: bool, std_scale: float = 1.0) -> torch.Tensor:
        mean, log_std, _v = self.forward(obs)
        if deterministic:
            return torch.tanh(mean)
        std = torch.exp(log_std) * float(max(std_scale, 1e-6))
        dist = Normal(mean, std)
        raw = dist.rsample()
        return torch.tanh(raw)


class _A2CEvalAgent:
    """A2CDDPG/SAC/TD3 get_action """

    def __init__(self, obs_dim: int, act_dim: int, device: torch.device, hidden: int = 256):
        self.device = device
        self.hidden_dim = int(hidden)
        self.actor_critic = _A2CActorCriticEval(obs_dim=int(obs_dim), act_dim=int(act_dim), hidden=int(hidden)).to(device)
        self.exploration_std_scale = 1.0

    def set_exploration(self, noise_std: float) -> None:
        
        self.exploration_std_scale = float(max(float(noise_std), 1e-4))

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            a = self.actor_critic.sample_action(
                obs=obs,
                deterministic=bool(deterministic),
                std_scale=float(self.exploration_std_scale),
            )
        return a.detach().cpu().numpy()

    def load(self, ckpt_path: str) -> None:
        obj = _torch_load(str(ckpt_path), map_location="cpu")
        if not isinstance(obj, dict):
            raise RuntimeError(f"A2C checkpointdict{ckpt_path}")
        if "actor_critic" in obj:
            self.actor_critic.load_state_dict(obj["actor_critic"], strict=True)
            return
        
        state = {}
        if isinstance(obj.get("actor", None), dict):
            for k, v in obj["actor"].items():
                state[f"{k}"] = v
        if isinstance(obj.get("critic", None), dict):
            for k, v in obj["critic"].items():
                state[f"{k}"] = v
        if "log_std" in obj:
            state["log_std"] = obj["log_std"]
        if state:
            self.actor_critic.load_state_dict(state, strict=False)
            return
        raise RuntimeError(f"A2C checkpoint actor_critic/actor/critic {ckpt_path}")


def _extract_run_seed_total_steps(run_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    """
     run_dir/configs/train_tianshou_fixed.yaml  meta_train.json  seed/total_steps
     seed
    """
    seed: Optional[int] = None
    total_steps: Optional[int] = None

    cfg_path = run_dir / "configs" / "train_tianshou_fixed.yaml"
    if cfg_path.exists():
        try:
            cfg_obj = _load_yaml(cfg_path) or {}
            if isinstance(cfg_obj, dict):
                if "seed" in cfg_obj:
                    seed = int(cfg_obj.get("seed"))
                train_block = cfg_obj.get("train", {}) or {}
                if isinstance(train_block, dict) and ("total_steps" in train_block):
                    total_steps = int(train_block.get("total_steps"))
                elif "total_steps" in cfg_obj:
                    total_steps = int(cfg_obj.get("total_steps"))
        except Exception as exc:
            _warn_once(
                "extract_run_seed_total_steps_cfg",
                f" train_tianshou_fixed.yaml  meta_train.json{type(exc).__name__}: {exc}",
            )

    meta_path = run_dir / "meta_train.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            if seed is None and ("seed" in meta):
                seed = int(meta.get("seed"))
            if total_steps is None and ("total_steps" in meta):
                total_steps = int(meta.get("total_steps"))
        except Exception as exc:
            _warn_once(
                "extract_run_seed_total_steps_meta",
                f" meta_train.json seed/total_steps {type(exc).__name__}: {exc}",
            )
    return seed, total_steps


def _extract_3drl_ckpt_quality(ckpt_path: Path) -> Dict[str, Any]:
    """
    3DRL checkpointrun/

    
    - warmupcheckpoint
    - summary
    """
    p = Path(ckpt_path).resolve()
    run_dir = p.parent.parent
    meta_path = run_dir / "meta_train.json"
    cfg_path = run_dir / "configs" / "train_tianshou_fixed.yaml"

    total_steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    seed: Optional[int] = None

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_obj = json.load(f) or {}
            if isinstance(meta_obj, dict):
                if "total_steps" in meta_obj:
                    total_steps = int(meta_obj.get("total_steps"))
                if "seed" in meta_obj:
                    seed = int(meta_obj.get("seed"))
        except Exception as exc:
            _warn_once(
                f"baseline_meta_parse::{p.name}",
                f"baseline meta_train.json{type(exc).__name__}: {exc}",
            )

    if cfg_path.exists():
        try:
            cfg_obj = _load_yaml(cfg_path) or {}
            if isinstance(cfg_obj, dict):
                if seed is None and ("seed" in cfg_obj):
                    seed = int(cfg_obj.get("seed"))
                train_block = cfg_obj.get("train", {}) or {}
                if isinstance(train_block, dict):
                    if total_steps is None and ("total_steps" in train_block):
                        total_steps = int(train_block.get("total_steps"))
                    if "warmup_steps" in train_block:
                        warmup_steps = int(train_block.get("warmup_steps"))
        except Exception as exc:
            _warn_once(
                f"baseline_cfg_parse::{p.name}",
                f"baseline train_tianshou_fixed.yaml{type(exc).__name__}: {exc}",
            )

    likely_untrained = False
    has_updates = None
    if (total_steps is not None) and (warmup_steps is not None):
        likely_untrained = bool(int(total_steps) <= int(warmup_steps))
        has_updates = bool(int(total_steps) > int(warmup_steps))

    return {
        "ckpt_path": str(p),
        "run_dir": str(run_dir),
        "seed": int(seed) if seed is not None else None,
        "total_steps": int(total_steps) if total_steps is not None else None,
        "warmup_steps": int(warmup_steps) if warmup_steps is not None else None,
        "likely_untrained": bool(likely_untrained),
        "has_updates": (bool(has_updates) if has_updates is not None else None),
    }


def _warn_curve_anomalies(
    series: Dict[str, Dict[str, Dict[str, List[float]]]],
    methods: Sequence[str],
) -> None:
    """
    
    """
    methods_l = [str(m).strip().lower() for m in methods]
    drl_methods = [m for m in methods_l if m in ("ddpg", "sac", "td3", "a2c")]

    
    if len(drl_methods) >= 2:
        for metric in ("paper_cost", "mean_T_off", "mean_energy"):
            pair_hits: List[str] = []
            for i in range(len(drl_methods)):
                for j in range(i + 1, len(drl_methods)):
                    m1 = drl_methods[i]
                    m2 = drl_methods[j]
                    k1 = f"full|{m1}"
                    k2 = f"full|{m2}"
                    y1 = np.asarray(((series.get(k1, {}).get(metric, {}) or {}).get("y", [])), dtype=np.float64).reshape(-1)
                    y2 = np.asarray(((series.get(k2, {}).get(metric, {}) or {}).get("y", [])), dtype=np.float64).reshape(-1)
                    if y1.size <= 0 or y2.size <= 0 or y1.shape != y2.shape:
                        continue
                    if np.allclose(y1, y2, atol=1e-12, rtol=0.0):
                        pair_hits.append(f"{m1}=={m2}")
            if pair_hits:
                _warn_once(
                    f"curve_identical::{metric}",
                    f"full{metric}{pair_hits}"
                    "baselinecheckpoint",
                )

def _auto_find_3drl_ckpt(
    algo: str,
    *,
    target_seed: Optional[int] = None,
    target_total_steps: Optional[int] = None,
    allow_crossrun_fallback: bool = True,
) -> Optional[Path]:
    """
     runs/paper  3DRL baseline checkpoint

    
    -  target_seed/target_total_steps seed total_steps  baseline run
    -  allow_crossrun_fallback=False None baseline
    -  fallback run + run  ckpt
    """
    algo = str(algo).strip().lower()
    root = REPO_ROOT / "runs" / "paper"
    if not root.exists():
        return None

    # item: (run_mtime, -local_rank, local_ckpt_mtime, ckpt_path, run_seed, run_total_steps)
    all_candidates: List[Tuple[float, int, float, Path, Optional[int], Optional[int]]] = []
    fair_candidates: List[Tuple[float, int, float, Path, Optional[int], Optional[int]]] = []

    for d in root.iterdir():
        if (not d.is_dir()) or (not d.name.lower().startswith(algo + "_")):
            continue
        try:
            run_mtime = float(d.stat().st_mtime)
        except Exception:
            run_mtime = -1.0

        local_best: Optional[Path] = None
        local_rank: int = 999
        local_mtime: float = -1.0
        local_dir_rank: int = 999

        ckpt_dirs = [d / CKPT_DIR_NAME, d / "ckpts", d / "ckpt"]
        for dir_rank, ckpt_dir in enumerate(ckpt_dirs):
            if not ckpt_dir.exists():
                continue

            
            
            
            cand: List[Tuple[int, Path]] = []
            p_final = ckpt_dir / "agent_final.pt"
            if p_final.exists():
                cand.append((0, p_final))
            for p_fin in ckpt_dir.glob("agent_finally_step*.pt"):
                cand.append((1, p_fin))

            for file_rank, p in cand:
                try:
                    mt = float(p.stat().st_mtime)
                except Exception:
                    continue
                local_score = (-int(file_rank), float(mt), -int(dir_rank))
                best_local_score = (-int(local_rank), float(local_mtime), -int(local_dir_rank))
                if (local_best is None) or (local_score > best_local_score):
                    local_best = p
                    local_rank = int(file_rank)
                    local_mtime = float(mt)
                    local_dir_rank = int(dir_rank)

        if local_best is None:
            continue

        run_seed, run_steps = _extract_run_seed_total_steps(d)
        item = (float(run_mtime), -int(local_rank), float(local_mtime), local_best, run_seed, run_steps)
        all_candidates.append(item)

        seed_ok = (target_seed is None) or (run_seed is not None and int(run_seed) == int(target_seed))
        step_ok = (target_total_steps is None) or (run_steps is not None and int(run_steps) == int(target_total_steps))
        if seed_ok and step_ok:
            fair_candidates.append(item)

    if fair_candidates:
        fair_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return fair_candidates[-1][3]

    if not allow_crossrun_fallback:
        return None

    if all_candidates:
        all_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return all_candidates[-1][3]
    return None

def _set_xlim_for_loads(ax, loads_arr: np.ndarray) -> None:
    """ xlim  Matplotlib """
    try:
        xmin = float(np.nanmin(loads_arr))
        xmax = float(np.nanmax(loads_arr))
    except Exception:
        return
    if (not np.isfinite(xmin)) or (not np.isfinite(xmax)):
        return
    if xmin == xmax:
        dx = max(1e-3, 0.05 * max(1.0, abs(xmin)))
        xmin -= dx
        xmax += dx
    ax.set_xlim(xmin, xmax)


def _required_cpu_mcycles_from_load(load: float, cfg: JointEnvConfig) -> float:
    """
     load_scale CPU CMcycles C_cycles 
     task_model  C 
      C = C_cycles_base * (load^alpha_C) * (1 + 0.10 * load^2)
    """
    l = float(max(float(load), 1e-9))
    c0_cycles = float(max(float(getattr(cfg, "C_cycles_base", 0.0)), 0.0))
    alpha_c = float(getattr(cfg, "load_alpha_C", 1.15))
    scale_c = float((l ** alpha_c) * (1.0 + 0.10 * (l ** 2)))
    c_cycles = float(c0_cycles * scale_c)
    return float(c_cycles / 1e6)


def _required_task_size_mb_from_load(load: float, cfg: JointEnvConfig) -> float:
    """
     load_scale  DMB D_bits 
     task_model  D 
      D = D_bits_base * (load^alpha_D) * (1 + 0.15 * load^2)
    """
    l = float(max(float(load), 1e-9))
    d0_bits = float(max(float(getattr(cfg, "D_bits_base", 0.0)), 0.0))
    alpha_d = float(getattr(cfg, "load_alpha_D", 1.25))
    scale_d = float((l ** alpha_d) * (1.0 + 0.15 * (l ** 2)))
    d_bits = float(d0_bits * scale_d)
    
    return float(d_bits / 8.0 / 1e6)


def _metric_axis_label(metric_key: str) -> str:
    """
    
    """
    mk = str(metric_key).strip()
    mapping = {
        "paper_cost": " (a.u.)",
        "mean_T_off": " (s)",
        "mean_energy": " (J)",
    }
    return str(mapping.get(mk, mk))


def _rankdata_average(x: Sequence[float]) -> np.ndarray:
    """
     scipy  Spearman 
    """
    arr = np.asarray(list(x), dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        v = float(arr[order[i]])
        while j + 1 < n and float(arr[order[j + 1]]) == v:
            j += 1
        
        rank_avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = float(rank_avg)
        i = j + 1
    return ranks


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """
     Spearman  NaN
    """
    xa = np.asarray(list(x), dtype=np.float64).reshape(-1)
    ya = np.asarray(list(y), dtype=np.float64).reshape(-1)
    n = int(min(xa.size, ya.size))
    if n < 3:
        return float("nan")
    xa = xa[:n]
    ya = ya[:n]
    m = np.isfinite(xa) & np.isfinite(ya)
    if int(np.sum(m)) < 3:
        return float("nan")
    xr = _rankdata_average(xa[m])
    yr = _rankdata_average(ya[m])
    std_x = float(np.std(xr))
    std_y = float(np.std(yr))
    if (not np.isfinite(std_x)) or (not np.isfinite(std_y)) or std_x <= 0.0 or std_y <= 0.0:
        return float("nan")
    r = float(np.corrcoef(xr, yr)[0, 1])
    return float(r) if np.isfinite(r) else float("nan")


def _audit_curve_non_decreasing_with_c(
    x_c: Sequence[float],
    y_vals: Sequence[float],
    *,
    tol_rel: float,
    spearman_min: float,
    max_violation_ratio: float,
) -> Dict[str, Any]:
    """
     C 
    """
    x = np.asarray(list(x_c), dtype=np.float64).reshape(-1)
    y = np.asarray(list(y_vals), dtype=np.float64).reshape(-1)
    n = int(min(x.size, y.size))
    if n <= 0:
        return {
            "n_points": 0,
            "pass": False,
            "reason": "empty_curve",
        }

    x = x[:n]
    y = y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return {
            "n_points": int(x.size),
            "pass": False,
            "reason": "insufficient_valid_points",
        }

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    dy = np.diff(y)
    max_abs_y = float(np.nanmax(np.abs(y))) if y.size > 0 else 0.0
    tol_abs = float(max(1e-9, abs(float(tol_rel)) * max(max_abs_y, 1e-9)))

    violations: List[Dict[str, Any]] = []
    for i, d in enumerate(dy):
        if np.isfinite(d) and float(d) < -tol_abs:
            y_from = float(y[i])
            y_to = float(y[i + 1])
            denom = max(max_abs_y, 1e-9)
            violations.append(
                {
                    "from_idx": int(i),
                    "to_idx": int(i + 1),
                    "from_x": float(x[i]),
                    "to_x": float(x[i + 1]),
                    "from_value": y_from,
                    "to_value": y_to,
                    "delta": float(y_to - y_from),
                    "delta_pct_of_max_abs": float((y_to - y_from) / denom * 100.0),
                }
            )

    violation_count = int(len(violations))
    total_edges = int(max(1, y.size - 1))
    violation_ratio = float(violation_count / total_edges)
    sp = _safe_spearman(x, y)
    low_vs_high_ok = bool(float(y[0]) <= float(y[-1]) + tol_abs)
    worst_drop = float(np.min(dy)) if dy.size > 0 and np.any(np.isfinite(dy)) else float("nan")
    worst_drop_pct = (
        float(worst_drop / max(max_abs_y, 1e-9) * 100.0)
        if np.isfinite(worst_drop)
        else float("nan")
    )
    monotonic_pass = bool(violation_ratio <= float(max_violation_ratio))
    spearman_pass = bool(np.isfinite(sp) and float(sp) >= float(spearman_min))
    passed = bool(monotonic_pass and spearman_pass and low_vs_high_ok)

    return {
        "n_points": int(y.size),
        "tol_rel": float(tol_rel),
        "tol_abs": float(tol_abs),
        "spearman": float(sp),
        "spearman_pass": bool(spearman_pass),
        "violation_count": int(violation_count),
        "violation_ratio": float(violation_ratio),
        "max_violation_ratio": float(max_violation_ratio),
        "worst_drop": float(worst_drop),
        "worst_drop_pct_of_max_abs": float(worst_drop_pct),
        "low_vs_high_ok": bool(low_vs_high_ok),
        "monotonic_pass": bool(monotonic_pass),
        "pass": bool(passed),
        "violations": violations[:20],
    }


def _build_c_axis_shape_audit(
    series: Dict[str, Dict[str, Dict[str, List[float]]]],
    methods: Sequence[str],
    loads: Sequence[float],
    x_axis_values: Sequence[float],
    *,
    target_variant: str = "full",
    metrics: Sequence[str] = ("paper_cost", "mean_T_off", "mean_energy"),
    tol_rel: float = 0.02,
    spearman_min: float = 0.60,
    max_violation_ratio: float = 0.25,
) -> Dict[str, Any]:
    """
     C 
    """
    vname = str(target_variant).strip().lower() or "full"
    metric_list = [str(m) for m in metrics]
    methods_l = [str(m).strip().lower() for m in methods]
    by_method: Dict[str, Any] = {}
    rank_rows: List[Dict[str, Any]] = []

    for method in methods_l:
        sname = f"{vname}|{method}"
        metric_obj: Dict[str, Any] = {}
        pass_count = 0
        total_count = 0
        for metric in metric_list:
            yv = np.asarray(((series.get(sname, {}).get(metric, {}) or {}).get("y", []),), dtype=object).reshape(-1)
            if yv.size == 1 and isinstance(yv[0], (list, tuple, np.ndarray)):
                y_list = np.asarray(yv[0], dtype=np.float64).reshape(-1).tolist()
            else:
                y_list = np.asarray(yv, dtype=np.float64).reshape(-1).tolist()
            audit = _audit_curve_non_decreasing_with_c(
                x_axis_values,
                y_list,
                tol_rel=float(tol_rel),
                spearman_min=float(spearman_min),
                max_violation_ratio=float(max_violation_ratio),
            )
            metric_obj[metric] = audit
            if bool(audit.get("pass", False)):
                pass_count += 1
            total_count += 1

        by_method[method] = {
            "pass_count": int(pass_count),
            "metric_count": int(total_count),
            "all_metrics_pass": bool(total_count > 0 and pass_count == total_count),
            "metrics": metric_obj,
        }
        rank_rows.append(
            {
                "method": method,
                "pass_count": int(pass_count),
                "metric_count": int(total_count),
                "pass_ratio": float(pass_count / max(total_count, 1)),
            }
        )

    rank_rows.sort(key=lambda d: (int(d["pass_count"]), float(d["pass_ratio"])), reverse=True)
    return {
        "variant": vname,
        "x_axis": _cpu_axis_label(),
        "expected_trend": "non_decreasing_with_C",
        "is_ten_load_protocol": bool(_is_ten_load_protocol(loads)),
        "thresholds": {
            "tol_rel": float(tol_rel),
            "spearman_min": float(spearman_min),
            "max_violation_ratio": float(max_violation_ratio),
        },
        "metrics": metric_list,
        "by_method": by_method,
        "method_rank_by_pass_count": rank_rows,
    }


def _save_c_axis_shape_audit_csv(eval_dir: Path, eval_ts: str, audit: Dict[str, Any]) -> Optional[Path]:
    """
     CSV
    """
    by_method = audit.get("by_method", {})
    if not isinstance(by_method, dict) or len(by_method) <= 0:
        return None
    out = eval_dir / f"shape_audit_{eval_ts}.csv"
    cols = [
        "variant",
        "method",
        "metric",
        "pass",
        "low_vs_high_ok",
        "spearman",
        "spearman_pass",
        "violation_count",
        "violation_ratio",
        "worst_drop",
        "worst_drop_pct_of_max_abs",
        "n_points",
        "tol_rel",
        "tol_abs",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        ww = csv.DictWriter(f, fieldnames=cols)
        ww.writeheader()
        for method, mobj in by_method.items():
            mm = mobj.get("metrics", {})
            if not isinstance(mm, dict):
                continue
            for metric, a in mm.items():
                if not isinstance(a, dict):
                    continue
                ww.writerow(
                    {
                        "variant": str(audit.get("variant", "full")),
                        "method": str(method),
                        "metric": str(metric),
                        "pass": int(bool(a.get("pass", False))),
                        "low_vs_high_ok": int(bool(a.get("low_vs_high_ok", False))),
                        "spearman": a.get("spearman", float("nan")),
                        "spearman_pass": int(bool(a.get("spearman_pass", False))),
                        "violation_count": int(a.get("violation_count", 0)),
                        "violation_ratio": a.get("violation_ratio", float("nan")),
                        "worst_drop": a.get("worst_drop", float("nan")),
                        "worst_drop_pct_of_max_abs": a.get("worst_drop_pct_of_max_abs", float("nan")),
                        "n_points": int(a.get("n_points", 0)),
                        "tol_rel": a.get("tol_rel", float("nan")),
                        "tol_abs": a.get("tol_abs", float("nan")),
                    }
                )
    return out


class _TianshouPpoAgentAdapter:
    """
    Tianshou PPOPPOAgent
      - act(obs, deterministic) -> (act_np, logp, v)
    """

    def __init__(self, policy: Any, device: str):
        self.policy = policy
        self.device = torch.device(device)
        try:
            self.policy.to(self.device)
        except Exception as exc:
            _warn_once("ppo_adapter_to_device", f"policy.to(device) {type(exc).__name__}: {exc}")
        try:
            self.policy.eval()
        except Exception as exc:
            _warn_once("ppo_adapter_eval", f"policy.eval() {type(exc).__name__}: {exc}")

    def act(self, obs: np.ndarray, deterministic: bool, obs_raw: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).float().unsqueeze(0).to(self.device)
        info = None
        
        if obs_raw is not None:
            try:
                info = {"obs_raw": np.asarray(obs_raw, dtype=np.float32)}
            except Exception:
                info = {"obs_raw": obs_raw}
        with torch.no_grad():
            (mu, sigma), _ = self.policy.actor(obs_t, info=info)
            if deterministic:
                act_t = mu
            else:
                
                try:
                    dist_fn = getattr(self.policy, "dist_fn", None)
                    if callable(dist_fn):
                        act_t = dist_fn(mu, sigma).sample()
                    else:
                        act_t = torch.distributions.Normal(mu, sigma).sample()
                        _warn_once("ppo_eval_distfn_missing", "policy.dist_fn  Normal ")
                except Exception as exc:
                    act_t = torch.distributions.Normal(mu, sigma).sample()
                    _warn_once("ppo_eval_distfn_fallback", f" Normal{type(exc).__name__}: {exc}")
            v_t = self.policy.critic(obs_t).squeeze(-1)
        act = act_t.squeeze(0).detach().cpu().numpy()
        v = float(v_t.squeeze(0).detach().cpu().item())
        return act, 0.0, v


def _load_ppo_agent(train_run_dir: Path, base_env_cfg: JointEnvConfig, seed0: int, cfg_train: dict, device: str) -> Tuple[Any, Path, int, int, Optional[int]]:
    """
     probe env  obs_dim/act_dim/M/I/K PPO checkpoint

     (agent, ckpt_path, obs_dim, act_dim, hidden)
    """
    probe_env = CompRISEnvJoint(base_env_cfg, seed=seed0, reward_mode=str(cfg_train.get("reward_mode", "p1")))
    obs_dim = int(probe_env.obs_flat().shape[0])
    act_dim = int(probe_env.action_dim)
    delta_dim = 2 * int(getattr(probe_env.cfg, "M", 0))

    ckpt_path = _select_ppo_ckpt(train_run_dir)
    ckpt_obj = _torch_load(ckpt_path, map_location="cpu")

    
    
    if isinstance(ckpt_obj, dict) and ("policy" in ckpt_obj) and ("ac" not in ckpt_obj) and ("actor" not in ckpt_obj):
        try:
            from src.algos.tianshou.ppo_config import create_ppo_policy  # type: ignore
        except Exception:
            from algos.tianshou.ppo_config import create_ppo_policy  # type: ignore

        net_cfg = cfg_train.get("network", {}) or {}
        hidden_sizes = tuple(net_cfg.get("hidden_sizes", [256, 256]))
        use_layernorm = bool(net_cfg.get("use_layernorm", True))
        share_preprocess = bool(net_cfg.get("share_preprocess", True))
        activation_name = str(net_cfg.get("activation", "relu")).strip().lower()
        if activation_name == "tanh":
            activation = torch.nn.Tanh
        elif activation_name == "gelu":
            activation = torch.nn.GELU
        else:
            activation = torch.nn.ReLU

        train_cfg = cfg_train.get("train", {}) or {}
        
        logp_mode = str(train_cfg.get("logp_mode", "auto")).strip().lower()
        if logp_mode == "auto":
            logp_mode = "mean" if int(act_dim) >= 32 else "sum"
        if logp_mode not in ("mean", "sum"):
            logp_mode = "sum"
        lr = float(train_cfg.get("lr", 1e-4))
        gamma = float(train_cfg.get("gamma", 0.99))
        gae_lambda = float(train_cfg.get("lam", 0.95))
        eps_clip = float(train_cfg.get("clip_ratio", 0.2))
        vf_coef = float(train_cfg.get("vf_coef", 0.5))
        ent_coef = float(train_cfg.get("ent_coef", 0.0))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))

        policy, _optimizer = create_ppo_policy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            delta_dim=int(delta_dim),
            logp_mode=str(logp_mode),
            hidden_sizes=hidden_sizes,
            use_layernorm=use_layernorm,
            share_preprocess=share_preprocess,
            activation=activation,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            eps_clip=eps_clip,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            max_action=float(net_cfg.get("max_action", 1.0)),
            device=device,
            unbounded=bool(net_cfg.get("unbounded", False)),
            conditioned_sigma=bool(net_cfg.get("conditioned_sigma", False)),
        )
        
        
        
        
        try:
            phase_c = (train_cfg.get("phase_c", {}) or {})
            scores_mode = str(phase_c.get("scores_mode", "learned")).strip().lower()
            scores_fixed_method = str(phase_c.get("scores_fixed_method", "balanced")).strip().lower()
            scores_fixed_tanh_scale = float(phase_c.get("scores_fixed_tanh_scale", 2.5))
            scores_fixed_sigma = float(phase_c.get("scores_fixed_sigma", 0.02))

            setattr(policy.actor, "scores_mode", str(scores_mode))
            if str(scores_mode) == "fixed" and hasattr(policy.actor, "set_scores_fixed"):
                policy.actor.set_scores_fixed(
                    cfg=base_env_cfg,
                    mode=str(scores_fixed_method),
                    tanh_scale=float(scores_fixed_tanh_scale),
                    sigma_fixed=float(scores_fixed_sigma),
                )
        except Exception as exc:
            _warn_once("eval_load_ppo_scores_mode", f"PPO scores_mode {type(exc).__name__}: {exc}")
        policy.load_state_dict(ckpt_obj["policy"])
        adapter = _TianshouPpoAgentAdapter(policy=policy, device=device)
        print(f"[HB][eval][ppo][tianshou] loaded ckpt={ckpt_path}", flush=True)
        return adapter, ckpt_path, obs_dim, act_dim, int(hidden_sizes[-1]) if hidden_sizes else None
    
    if isinstance(ckpt_obj, dict):
        keys = sorted(list(ckpt_obj.keys()))
        keys_show = ", ".join(keys[:30]) + (", ..." if len(keys) > 30 else "")
        raise RuntimeError(
            f"[HB][eval][FATAL]  PPO checkpoint {ckpt_path}keys={keys_show}"
            " scripts/train_ppo.py checkpoints/ckpt_*.pt"
        )
    raise RuntimeError(
        f"[HB][eval][FATAL]  PPO checkpoint {type(ckpt_obj).__name__}{ckpt_path}"
        " scripts/train_ppo.py checkpoints/ckpt_*.pt"
    )


def _load_ppo_agent_from_ckpt(
    train_run_dir: Path,
    ckpt_path: Path,
    base_env_cfg: JointEnvConfig,
    seed0: int,
    cfg_train: dict,
    device: str,
) -> Tuple[Any, Path, int, int, Optional[int]]:
    """
     checkpoint  PPO agentPhaseF fixed-eval curve

    
    -  `_load_ppo_agent`  ckpt_path 
    -  (agent, ckpt_path, obs_dim, act_dim, hidden)
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    probe_env = CompRISEnvJoint(base_env_cfg, seed=seed0, reward_mode=str(cfg_train.get("reward_mode", "p1")))
    obs_dim = int(probe_env.obs_flat().shape[0])
    act_dim = int(probe_env.action_dim)
    delta_dim = 2 * int(getattr(probe_env.cfg, "M", 0))

    ckpt_obj = _torch_load(ckpt_path, map_location="cpu")

    
    if isinstance(ckpt_obj, dict) and ("policy" in ckpt_obj) and ("ac" not in ckpt_obj) and ("actor" not in ckpt_obj):
        try:
            from src.algos.tianshou.ppo_config import create_ppo_policy  # type: ignore
        except Exception:
            from algos.tianshou.ppo_config import create_ppo_policy  # type: ignore

        net_cfg = cfg_train.get("network", {}) or {}
        hidden_sizes = tuple(net_cfg.get("hidden_sizes", [256, 256]))
        use_layernorm = bool(net_cfg.get("use_layernorm", True))
        share_preprocess = bool(net_cfg.get("share_preprocess", True))
        activation_name = str(net_cfg.get("activation", "relu")).strip().lower()
        if activation_name == "tanh":
            activation = torch.nn.Tanh
        elif activation_name == "gelu":
            activation = torch.nn.GELU
        else:
            activation = torch.nn.ReLU

        train_cfg = cfg_train.get("train", {}) or {}
        logp_mode = str(train_cfg.get("logp_mode", "auto")).strip().lower()
        if logp_mode == "auto":
            logp_mode = "mean" if int(act_dim) >= 32 else "sum"
        if logp_mode not in ("mean", "sum"):
            logp_mode = "sum"

        policy, _optimizer = create_ppo_policy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            delta_dim=int(delta_dim),
            logp_mode=str(logp_mode),
            hidden_sizes=hidden_sizes,
            use_layernorm=use_layernorm,
            share_preprocess=share_preprocess,
            activation=activation,
            lr=float(train_cfg.get("lr", 1e-4)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            gae_lambda=float(train_cfg.get("lam", 0.95)),
            eps_clip=float(train_cfg.get("clip_ratio", 0.2)),
            vf_coef=float(train_cfg.get("vf_coef", 0.5)),
            ent_coef=float(train_cfg.get("ent_coef", 0.0)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
            max_action=float(net_cfg.get("max_action", 1.0)),
            device=device,
            unbounded=bool(net_cfg.get("unbounded", False)),
            conditioned_sigma=bool(net_cfg.get("conditioned_sigma", False)),
        )
        
        try:
            phase_c = (train_cfg.get("phase_c", {}) or {})
            scores_mode = str(phase_c.get("scores_mode", "learned")).strip().lower()
            scores_fixed_method = str(phase_c.get("scores_fixed_method", "balanced")).strip().lower()
            scores_fixed_tanh_scale = float(phase_c.get("scores_fixed_tanh_scale", 2.5))
            scores_fixed_sigma = float(phase_c.get("scores_fixed_sigma", 0.02))

            setattr(policy.actor, "scores_mode", str(scores_mode))
            if str(scores_mode) == "fixed" and hasattr(policy.actor, "set_scores_fixed"):
                policy.actor.set_scores_fixed(
                    cfg=base_env_cfg,
                    mode=str(scores_fixed_method),
                    tanh_scale=float(scores_fixed_tanh_scale),
                    sigma_fixed=float(scores_fixed_sigma),
                )
        except Exception as exc:
            _warn_once("eval_load_ppo_ckpt_scores_mode", f"PPOckpt scores_mode {type(exc).__name__}: {exc}")
        policy.load_state_dict(ckpt_obj["policy"])
        adapter = _TianshouPpoAgentAdapter(policy=policy, device=device)
        print(f"[HB][eval][ppo][tianshou] loaded ckpt={ckpt_path}", flush=True)
        return adapter, ckpt_path, obs_dim, act_dim, int(hidden_sizes[-1]) if hidden_sizes else None

    if isinstance(ckpt_obj, dict):
        keys = sorted(list(ckpt_obj.keys()))
        keys_show = ", ".join(keys[:30]) + (", ..." if len(keys) > 30 else "")
        raise RuntimeError(
            f"[HB][eval][FATAL]  PPO checkpoint {ckpt_path}keys={keys_show}"
            " scripts/train_ppo.py checkpoints/ckpt_*.pt"
        )
    raise RuntimeError(
        f"[HB][eval][FATAL]  PPO checkpoint {type(ckpt_obj).__name__}{ckpt_path}"
        " scripts/train_ppo.py checkpoints/ckpt_*.pt"
    )


def _find_fixed_eval_ckpts(train_run_dir: Path) -> List[Tuple[int, Path]]:
    """
     fixed-eval curve  checkpoint  step 

    
    1) checkpoints/ckpt_step_*.pt ckpts/ckpt/
    2)  step ckpt best/final step

    [(ckpt_step, ckpt_path), ...]
    """
    train_run_dir = Path(train_run_dir)

    cand_dirs: List[Path] = []
    for d in (train_run_dir / CKPT_DIR_NAME, train_run_dir / "ckpts", train_run_dir / "ckpt"):
        if d.exists():
            cand_dirs.append(d)

    step_re = re.compile(r"ckpt_step_(\d+)\.pt$", re.IGNORECASE)
    found: Dict[int, Path] = {}
    for d in cand_dirs:
        for p in d.glob("ckpt_step_*.pt"):
            m = step_re.search(p.name)
            if not m:
                continue
            st = int(m.group(1))
            
            found.setdefault(st, p)

    out = sorted([(int(k), Path(v)) for k, v in found.items()], key=lambda kv: kv[0])
    if out:
        return out

    
    out2: List[Tuple[int, Path]] = []
    best_step: Optional[int] = None
    final_step: Optional[int] = None

    try:
        best_path = train_run_dir / "best_ckpt.json"
        if best_path.exists():
            with open(best_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            best_step = int(obj.get("best_step", -1))
    except Exception:
        best_step = None

    try:
        meta_path = train_run_dir / "meta_train.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            final_step = int(meta.get("total_steps", -1))
    except Exception:
        final_step = None

    for d in cand_dirs:
        p_best = d / "ckpt_best.pt"
        if p_best.exists() and best_step is not None and best_step > 0:
            out2.append((best_step, p_best))
            break

    for d in cand_dirs:
        p_final = d / "ckpt_final.pt"
        if p_final.exists() and final_step is not None and final_step > 0:
            out2.append((final_step, p_final))
            break

    if not out2:
        
        p0 = _select_ppo_ckpt(train_run_dir)
        out2.append((-1, p0))

    out2.sort(key=lambda kv: kv[0])
    return out2


def _resolve_fixed_eval_single_ckpt(train_run_dir: Path, ckpt_path: Path) -> List[Tuple[int, Path]]:
    """
     fixed-eval  checkpoint 

    
    - [(ckpt_step, ckpt_path)] ckpt_step 
    """
    p = Path(ckpt_path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"fixed_eval ckpt not found: {p}")

    ckpt_step = -1
    m = re.search(r"step[_-]inf(\d+)", p.name, flags=re.IGNORECASE)
    if m:
        try:
            ckpt_step = int(m.group(1))
        except Exception:
            ckpt_step = -1

    if ckpt_step <= 0:
        try:
            obj = _torch_load(p, map_location="cpu")
            if isinstance(obj, dict):
                meta = obj.get("meta", {}) or {}
                gs = meta.get("global_step", obj.get("global_step", None))
                if gs is not None:
                    ckpt_step = int(float(gs))
        except Exception:
            ckpt_step = -1

    if ckpt_step <= 0 and ("best" in p.name.lower()):
        try:
            best_json = Path(train_run_dir) / "best_ckpt.json"
            if best_json.exists():
                with open(best_json, "r", encoding="utf-8") as f:
                    obj = json.load(f) or {}
                bs = obj.get("best_step", None)
                if bs is not None:
                    ckpt_step = int(float(bs))
        except Exception:
            ckpt_step = -1

    if ckpt_step <= 0 and ("final" in p.name.lower()):
        try:
            meta_json = Path(train_run_dir) / "meta_train.json"
            if meta_json.exists():
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta_obj = json.load(f) or {}
                ts = meta_obj.get("total_steps", None)
                if ts is not None:
                    ckpt_step = int(float(ts))
        except Exception:
            ckpt_step = -1

    return [(int(ckpt_step), p)]

def _load_drl_agent(algo: str, ckpt_path: Path, obs_dim: int, act_dim: int, device: str):
    """
     3DRL DDPG/SAC/TD3/A2Ccheckpoint

     agent API
    - .load(str(ckpt_path))
    - .get_action(obs_t, ...) -> np.ndarray  torch.Tensor (1, act_dim)  (act_dim,)
    - .device
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DRL checkpoint not found: {ckpt_path}")

    dev = torch.device(device)
    algo = algo.lower().strip()
    ckpt_obj = _torch_load(str(ckpt_path), map_location="cpu")

    hidden_dim = None
    try:
        if isinstance(ckpt_obj, dict):
            meta = ckpt_obj.get("model_meta", {}) or {}
            if "hidden" in meta:
                hidden_dim = int(meta.get("hidden"))
            if hidden_dim is None:
                actor_sd = ckpt_obj.get("actor", {}) or {}
                w1 = actor_sd.get("fc1.weight", None)
                if hasattr(w1, "shape") and len(w1.shape) >= 1:
                    hidden_dim = int(w1.shape[0])
            if hidden_dim is None:
                ac_sd = ckpt_obj.get("actor_critic", {}) or {}
                w1 = ac_sd.get("fc1.weight", None)
                if hasattr(w1, "shape") and len(w1.shape) >= 1:
                    hidden_dim = int(w1.shape[0])
    except Exception:
        hidden_dim = None

    if hidden_dim is None or int(hidden_dim) <= 0:
        hidden_dim = 256

    if algo == "ddpg":
        from src.algos.ddpg.ddpg import DDPGAgent
        agent = DDPGAgent(obs_dim, act_dim, dev, hidden=int(hidden_dim))
    elif algo == "sac":
        from src.algos.sac.sac import SACAgent
        agent = SACAgent(obs_dim, act_dim, dev, hidden=int(hidden_dim))
    elif algo == "td3":
        from src.algos.td3.td3 import TD3Agent
        agent = TD3Agent(obs_dim, act_dim, dev, hidden=int(hidden_dim))
    elif algo == "a2c":
        agent = _A2CEvalAgent(obs_dim=obs_dim, act_dim=act_dim, device=dev, hidden=int(hidden_dim))
    else:
        raise ValueError(f"Unknown DRL algo: {algo}. Must be 'ddpg', 'sac', 'td3', or 'a2c'")

    agent.load(str(ckpt_path))
    
    if not hasattr(agent, "device"):
        try:
            agent.device = dev  # type: ignore
        except Exception as exc:
            _warn_once("eval_set_baseline_device", f"agentdevice{type(exc).__name__}: {exc}")

    
    for m_name in ("actor", "critic", "actor_target", "critic_target", "actor_critic"):
        mod = getattr(agent, m_name, None)
        if mod is not None and hasattr(mod, "eval"):
            try:
                mod.eval()
            except Exception as exc:
                _warn_once(
                    f"eval_set_baseline_mode::{algo}.{m_name}",
                    f" {algo}.{m_name}.eval() {type(exc).__name__}: {exc}",
                )

    print(f"[HB][eval][{algo.upper()}] Loaded from {ckpt_path} hidden={int(hidden_dim)}", flush=True)
    return agent


def _densify_episode_series(
    episodes: Union[List[float], np.ndarray],
    rewards: Union[List[float], np.ndarray],
    *,
    mode: str = "hold",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sparse history into strict 1-episode grid.

     hold
    """
    x0 = np.asarray(episodes, dtype=np.float64).reshape(-1)
    y0 = np.asarray(rewards, dtype=np.float64).reshape(-1)
    n = int(min(x0.size, y0.size))
    if n <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    x0 = x0[:n]
    y0 = y0[:n]

    m = np.isfinite(x0) & np.isfinite(y0)
    if not np.any(m):
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    x = np.asarray(np.rint(x0[m]), dtype=np.int64).reshape(-1)
    y = np.asarray(y0[m], dtype=np.float64).reshape(-1)
    m_pos = x > 0
    if not np.any(m_pos):
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    x = x[m_pos]
    y = y[m_pos]

    order = np.argsort(x, kind="stable")
    x = np.asarray(x[order], dtype=np.int64).reshape(-1)
    y = np.asarray(y[order], dtype=np.float64).reshape(-1)

    x_u: List[int] = []
    y_u: List[float] = []
    for ep_now, val_now in zip(x.tolist(), y.tolist()):
        ep_i = int(ep_now)
        val_f = float(val_now)
        if x_u and ep_i == x_u[-1]:
            y_u[-1] = val_f
        else:
            x_u.append(ep_i)
            y_u.append(val_f)
    if not x_u:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    x_anchor = np.asarray(x_u, dtype=np.int64).reshape(-1)
    y_anchor = np.asarray(y_u, dtype=np.float64).reshape(-1)
    out_x = np.arange(1, int(max(1, int(x_anchor[-1]))) + 1, dtype=np.int64)

    mode_n = str(mode or "hold").strip().lower()
    if mode_n == "hold":
        idx = np.searchsorted(x_anchor, out_x, side="right") - 1
        idx = np.clip(idx, 0, x_anchor.size - 1)
        out_y = np.asarray(y_anchor[idx], dtype=np.float64).reshape(-1)
        return np.asarray(out_x, dtype=np.int64), np.asarray(out_y, dtype=np.float64)

    if x_anchor.size == 1:
        out_y = np.full(out_x.shape, float(y_anchor[0]), dtype=np.float64)
    else:
        out_y = np.interp(
            np.asarray(out_x, dtype=np.float64),
            np.asarray(x_anchor, dtype=np.float64),
            np.asarray(y_anchor, dtype=np.float64),
            left=float(y_anchor[0]),
            right=float(y_anchor[-1]),
        )
    return np.asarray(out_x, dtype=np.int64), np.asarray(out_y, dtype=np.float64)


def _extract_training_history(ckpt_path: Path, algo: str) -> Dict[str, List[float]]:
    """
    

    
    - PPOepisode `train_reward_total_ep_dense`21 episode 1
       `ep_reward_mean` reward_total 
       `paper_return` 
    - 3DRL ckpt["training_history"]={"episodes":...,"rewards":...} 
    """
    algo = algo.lower().strip()
    try:
        ckpt = _torch_load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            return {"episodes": [], "rewards": []}

        if algo == "ppo":
            hist = ckpt.get("history", {}) or {}
            episodes_dense = hist.get("train_episodes_dense", []) or []
            reward_dense = hist.get("train_reward_total_ep_dense", []) or []
            if episodes_dense and reward_dense:
                x_dense, y_dense = _densify_episode_series(
                    episodes_dense,
                    [float(v) for v in reward_dense],
                    mode="hold",
                )
                if x_dense.size > 0 and y_dense.size > 0:
                    print(
                        "[HB][eval] PPO convergence uses dense per-episode reward_total from checkpoint history.",
                        flush=True,
                    )
                    return {"episodes": x_dense.astype(int).tolist(), "rewards": y_dense.astype(float).tolist()}

            steps = hist.get("steps", []) or []
            episodes = hist.get("episodes", []) or []
            
            
            rews = hist.get("ep_reward_mean", hist.get("rewards", [])) or []
            if not rews:
                print(
                    "[HB][eval][WARN] skip convergence history: reward_total episode series missing "
                    "(paper_return fallback disabled).",
                    flush=True,
                )
                return {"episodes": [], "rewards": []}

            ep_out = list(episodes)
            if (not ep_out) and steps:
                
                try:
                    run_dir = Path(ckpt_path).resolve().parent.parent
                    from src.utils.train_report import read_train_metrics_csv

                    m = read_train_metrics_csv(run_dir / "train_metrics.csv")
                    gs = np.asarray(m.get("global_step", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
                    te = np.asarray(m.get("train_episode", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
                    ete = np.asarray(m.get("eval_train_episode", np.full_like(gs, np.nan)), dtype=np.float64).reshape(-1)
                    if gs.size > 0 and te.size == gs.size:
                        ep_map: List[int] = []
                        for st in steps:
                            try:
                                s = float(st)
                            except Exception:
                                continue
                            idx = int(np.searchsorted(gs, s, side="right") - 1)
                            if idx < 0:
                                continue
                            v = float(ete[idx]) if (ete.size == gs.size and np.isfinite(ete[idx])) else float("nan")
                            if not np.isfinite(v):
                                v = float(te[idx])
                            if np.isfinite(v):
                                ep_map.append(int(v))
                        ep_out = ep_map
                except Exception:
                    ep_out = []

            x_dense, y_dense = _densify_episode_series(ep_out, rews, mode="hold")
            if x_dense.size > 0 and y_dense.size > 0:
                return {"episodes": x_dense.astype(int).tolist(), "rewards": y_dense.astype(float).tolist()}
            return {"episodes": list(ep_out), "rewards": list(rews)}

        if algo in ("ddpg", "sac", "td3", "a2c"):
            hist = ckpt.get("training_history", {}) or {}
            episodes = hist.get("episodes", []) or []
            rews = hist.get("rewards", hist.get("ep_reward_mean", [])) or []
            x_dense, y_dense = _densify_episode_series(episodes, rews, mode="hold")
            if x_dense.size > 0 and y_dense.size > 0:
                return {"episodes": x_dense.astype(int).tolist(), "rewards": y_dense.astype(float).tolist()}
            return {"episodes": list(episodes), "rewards": list(rews)}

        return {"episodes": [], "rewards": []}
    except Exception as e:
        print(f"[HB][eval][WARN] Failed to extract history from {ckpt_path}: {type(e).__name__}: {e}", flush=True)
        return {"episodes": [], "rewards": []}

def _plot_metric(
    fig_path: Path,
    loads: List[float],
    series: Dict[str, Dict[str, Any]],
    ylabel: str,
    yscale: str = "linear",
    zoom_quantile: Optional[float] = None,
    show_ci: bool = False,
    x_values: Optional[List[float]] = None,
    xlabel: str = " CPU  C (Mcycles)",
    show_variant_title: bool = True,
) -> None:
    loads_arr = np.asarray(loads, dtype=float).reshape(-1)
    if loads_arr.size == 0:
        return
    if x_values is None:
        x_arr = loads_arr
    else:
        x_arr = np.asarray(x_values, dtype=float).reshape(-1)
        if x_arr.size != loads_arr.size:
            _warn_once(
                "plot_metric_x_values_mismatch",
                f"x_valuesloadsloadslen(x)={int(x_arr.size)} len(loads)={int(loads_arr.size)}",
            )
            x_arr = loads_arr
    if x_arr.size == 0:
        return

    data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for sname, d in series.items():
        try:
            variant, method = sname.split("|", 1)
        except Exception:
            continue
        data.setdefault(variant, {})
        data[variant][method] = {
            "y": np.asarray(d["y"], dtype=float).reshape(-1),
            "e": np.asarray(d["e"], dtype=float).reshape(-1),
        }

    variants = [v for v in _TCOM_VARIANT_ORDER if v in data]
    if not variants:
        variants = sorted(list(data.keys()))
    n = len(variants)
    if n <= 0:
        return

    all_methods: List[str] = []
    for v in variants:
        for m in data[v].keys():
            if m not in all_methods:
                all_methods.append(m)
    method_order = [m for m in _TCOM_METHOD_STYLE.keys() if m in all_methods] + [m for m in all_methods if m not in _TCOM_METHOD_STYLE]

    fig_w = 7.2 if n == 3 else 2.6 * n
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 2.8), sharey=True)
    if n == 1:
        axes = [axes]

    all_y_values: List[np.ndarray] = []
    for ax, v in zip(axes, variants):
        _beautify_ax(ax)
        if bool(show_variant_title):
            ax.set_title(_variant_display_label(v), pad=6)
        for m in method_order:
            if m not in data[v]:
                continue
            st = _TCOM_METHOD_STYLE.get(m, {"color": "#000000", "marker": "o", "label": m, "linestyle": "-"})
            y = data[v][m]["y"]
            e = data[v][m]["e"]
            if y.size > 0:
                yv = y[np.isfinite(y)]
                if yv.size > 0:
                    all_y_values.append(yv)
            me = max(1, int(len(loads_arr) / 10))
            if bool(show_ci):
                ax.errorbar(
                    x_arr, y, yerr=e,
                    color=st["color"],
                    linestyle=st.get("linestyle", "-"),
                    marker=st["marker"],
                    markevery=me,
                    linewidth=1.9,
                    markersize=5.5,
                    markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                    markeredgecolor=st["color"],
                    capsize=3,
                    elinewidth=1.0,
                    capthick=1.0,
                    label=st.get("label", m),
                    zorder=3,
                )
            else:
                ax.plot(
                    x_arr, y,
                    color=st["color"],
                    linestyle=st.get("linestyle", "-"),
                    marker=st["marker"],
                    markevery=me,
                    linewidth=1.9,
                    markersize=5.5,
                    markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                    markeredgecolor=st["color"],
                    label=st.get("label", m),
                    zorder=3,
                )
        ax.set_xlabel(str(xlabel))
        _set_xlim_for_loads(ax, x_arr)

    yscale_norm = str(yscale).strip().lower()
    if yscale_norm == "log":
        if all_y_values:
            yy = np.concatenate(all_y_values)
            yy = yy[np.isfinite(yy) & (yy > 0.0)]
            if yy.size > 0:
                lo = float(np.nanquantile(yy, 0.02))
                hi = float(np.nanmax(yy))
                lo = max(lo * 0.8, 1e-8)
                hi = max(hi * 1.2, lo * 10.0)
                for ax in axes:
                    ax.set_yscale("log")
                    ax.set_ylim(lo, hi)
        ylabel_show = f"{ylabel} (log-y)"
    else:
        ylabel_show = ylabel

    if zoom_quantile is not None and all_y_values:
        try:
            q = float(zoom_quantile)
        except Exception:
            q = 0.85
        q = float(np.clip(q, 0.50, 0.99))
        yy = np.concatenate(all_y_values)
        yy = yy[np.isfinite(yy)]
        if yy.size > 1:
            lo = float(np.nanmin(yy))
            hi = float(np.nanquantile(yy, q))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = 0.08 * max(1e-9, hi - lo)
                lo2 = lo - pad
                if lo >= 0.0:
                    lo2 = max(0.0, lo2)
                hi2 = hi + pad
                for ax in axes:
                    
                    ax.set_yscale("linear")
                    ax.set_ylim(lo2, hi2)
                ylabel_show = f"{ylabel} (zoom Q{int(round(q * 100))})"

    axes[0].set_ylabel(ylabel_show)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 5),
        bbox_to_anchor=(0.5, 1.04),
        borderaxespad=0.12,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)
def _plot_ratio(
    fig_path: Path,
    loads: List[float],
    series: Dict[str, Dict[str, Dict[str, List[float]]]],
    metric: str,
    x_values: Optional[List[float]] = None,
    xlabel: str = " CPU  C (Mcycles)",
) -> None:
    from matplotlib.lines import Line2D

    loads_arr = np.asarray(loads, dtype=float).reshape(-1)
    if loads_arr.size == 0:
        return

    if x_values is None:
        x_arr = loads_arr
    else:
        x_arr = np.asarray(x_values, dtype=float).reshape(-1)
        if x_arr.size != loads_arr.size:
            _warn_once(
                "plot_ratio_x_values_mismatch",
                f"x_valuesloadsloadslen(x)={int(x_arr.size)} len(loads)={int(loads_arr.size)}",
            )
            x_arr = loads_arr
    if x_arr.size == 0:
        return

    methods = list(series.keys())
    method_order = [m for m in _TCOM_METHOD_STYLE.keys() if m in methods] + [m for m in methods if m not in _TCOM_METHOD_STYLE]

    fig = plt.figure(figsize=(4.4, 3.0))
    ax = plt.gca()
    _beautify_ax(ax)

    ymins, ymaxs = [], []
    for m in method_order:
        if m not in series:
            continue
        if ("full" not in series[m]) or (metric not in series[m]["full"]):
            continue
        y_full = np.asarray(series[m]["full"][metric], dtype=float).reshape(-1)
        if y_full.size != loads_arr.size:
            continue
        st = _TCOM_METHOD_STYLE.get(m, {"color": "#000000", "marker": "o", "label": m})
        me = max(1, int(len(loads_arr) / 10))

        if ("no_ris" in series[m]) and (metric in series[m]["no_ris"]):
            y_nr = np.asarray(series[m]["no_ris"][metric], dtype=float).reshape(-1)
            if y_nr.size == loads_arr.size:
                r = y_full / np.maximum(y_nr, 1e-12)
                ax.plot(x_arr, r, color=st["color"], linestyle="--",
                        marker=st["marker"], markevery=me, linewidth=1.8,
                        markersize=5.2,
                        markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                        markeredgecolor=st["color"], zorder=3)
                ymins.append(np.nanmin(r)); ymaxs.append(np.nanmax(r))

        if ("no_comp" in series[m]) and (metric in series[m]["no_comp"]):
            y_nc = np.asarray(series[m]["no_comp"][metric], dtype=float).reshape(-1)
            if y_nc.size == loads_arr.size:
                r = y_full / np.maximum(y_nc, 1e-12)
                ax.plot(x_arr, r, color=st["color"], linestyle=":",
                        marker=st["marker"], markevery=me, linewidth=1.8,
                        markersize=5.2,
                        markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                        markeredgecolor=st["color"], zorder=3)
                ymins.append(np.nanmin(r)); ymaxs.append(np.nanmax(r))

    ax.set_xlabel(str(xlabel))
    ax.set_ylabel(f"{_metric_axis_label(metric)}")
    _set_xlim_for_loads(ax, x_arr)

    if ymins and ymaxs:
        lo, hi = float(min(ymins)), float(max(ymaxs))
        pad = 0.08 * max(1e-6, hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    method_handles = []
    for m in method_order:
        st = _TCOM_METHOD_STYLE.get(m, {"color": "#000000", "marker": "o", "label": m})
        method_handles.append(Line2D(
            [0], [0], color=st["color"], marker=st["marker"], linestyle="-",
            markersize=6, markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
            markeredgecolor=st["color"], linewidth=1.6, label=st.get("label", m)
        ))

    ratio_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label=" / RIS"),
        Line2D([0], [0], color="black", linestyle=":",  linewidth=1.6, label=" / CoMP"),
    ]
    all_handles = method_handles + ratio_handles
    fig.legend(
        handles=all_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
        borderaxespad=0.12,
    )

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
    plt.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_full_only_teacher_style(
    fig_path: Path,
    loads: List[float],
    series: Dict[str, Dict[str, Any]],
    ylabel: str,
    *,
    x_values: Optional[List[float]] = None,
    xlabel: str = " CPU  C (Mcycles)",
    show_ci: bool = False,
) -> None:
    """
     full 
    - 
    - 
    -  variant 
    """
    loads_arr = np.asarray(loads, dtype=float).reshape(-1)
    if loads_arr.size == 0:
        return
    if x_values is None:
        x_arr = loads_arr
    else:
        x_arr = np.asarray(x_values, dtype=float).reshape(-1)
        if x_arr.size != loads_arr.size:
            _warn_once(
                "plot_full_only_x_values_mismatch",
                f"x_valuesloadsloadslen(x)={int(x_arr.size)} len(loads)={int(loads_arr.size)}",
            )
            x_arr = loads_arr
    if x_arr.size == 0:
        return

    
    curve_by_method: Dict[str, Dict[str, np.ndarray]] = {}
    for sname, d in series.items():
        try:
            variant, method = sname.split("|", 1)
        except Exception:
            continue
        if str(variant).strip().lower() != "full":
            continue
        curve_by_method[str(method)] = {
            "y": np.asarray(d.get("y", []), dtype=float).reshape(-1),
            "e": np.asarray(d.get("e", []), dtype=float).reshape(-1),
        }

    if not curve_by_method:
        return

    method_order = [m for m in _TCOM_METHOD_STYLE.keys() if m in curve_by_method] + [
        m for m in curve_by_method.keys() if m not in _TCOM_METHOD_STYLE
    ]

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.6))
    _beautify_ax(ax)
    me = max(1, int(len(loads_arr) / 10))

    for m in method_order:
        st = _TCOM_METHOD_STYLE.get(m, {"color": "#000000", "marker": "o", "label": m, "linestyle": "-"})
        y = curve_by_method[m]["y"]
        e = curve_by_method[m]["e"]
        if bool(show_ci):
            ax.errorbar(
                x_arr, y, yerr=e,
                color=st["color"],
                linestyle=st.get("linestyle", "-"),
                marker=st["marker"],
                markevery=me,
                linewidth=2.0,
                markersize=8.0,
                markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                markeredgecolor=st["color"],
                markeredgewidth=1.0,
                capsize=3,
                elinewidth=1.0,
                capthick=1.0,
                label=st.get("label", m),
                zorder=3,
            )
        else:
            ax.plot(
                x_arr, y,
                color=st["color"],
                linestyle=st.get("linestyle", "-"),
                marker=st["marker"],
                markevery=me,
                linewidth=2.0,
                markersize=8.0,
                markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                markeredgecolor=st["color"],
                markeredgewidth=1.0,
                label=st.get("label", m),
                zorder=3,
            )

    ax.set_xlabel(str(xlabel))
    ax.set_ylabel(str(ylabel))
    _set_xlim_for_loads(ax, x_arr)
    ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
    )

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_variant_grouped_bar(
    fig_path: Path,
    *,
    series: Dict[str, Dict[str, Dict[str, List[float]]]],
    methods: Sequence[str],
    metric_key: str,
    ylabel: str,
    variants_order: Sequence[str] = ("full", "no_comp", "no_ris"),
) -> None:
    """
    
    - full / no_comp / no_ris
    - 
    - 
    """
    def _lighten_color(hex_color: str, factor: float) -> Tuple[float, float, float]:
        """factor """
        c = str(hex_color).lstrip("#")
        if len(c) != 6:
            return (0.4, 0.4, 0.4)
        try:
            r = int(c[0:2], 16) / 255.0
            g = int(c[2:4], 16) / 255.0
            b = int(c[4:6], 16) / 255.0
        except Exception:
            return (0.4, 0.4, 0.4)
        a = float(np.clip(factor, 0.0, 1.0))
        return (r + (1.0 - r) * a, g + (1.0 - g) * a, b + (1.0 - b) * a)

    vlist = [str(v).strip().lower() for v in variants_order if str(v).strip()]
    if not vlist:
        return

    
    m_all = [str(m).strip().lower() for m in methods if str(m).strip()]
    m_order = [m for m in _TCOM_METHOD_STYLE.keys() if m in m_all] + [m for m in m_all if m not in _TCOM_METHOD_STYLE]
    if not m_order:
        return

    metric_mean: Dict[Tuple[str, str], float] = {}
    valid_methods: List[str] = []
    for m in m_order:
        has_any = False
        for v in vlist:
            node = ((series.get(f"{v}|{m}", {}) or {}).get(metric_key, {}) or {})
            y = np.asarray(node.get("y", []), dtype=np.float64).reshape(-1)
            y = y[np.isfinite(y)]
            if y.size > 0:
                metric_mean[(v, m)] = float(np.mean(y))
                has_any = True
        if has_any:
            valid_methods.append(m)
    if not valid_methods:
        return

    n_v = int(len(vlist))
    n_m = int(len(valid_methods))

    
    fig_w = float(max(5.6, 4.2 + 0.62 * float(n_m)))
    fig_h = 3.9
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    _beautify_ax(ax)

    
    group_gap = 1.35
    x_centers = np.arange(n_v, dtype=np.float64) * float(group_gap)

    cluster_w = 0.86
    
    if n_m == 1:
        bar_w = 0.78
    else:
        bar_w = float(min(0.18, max(0.06, cluster_w / float(n_m))))
    offsets = (np.arange(n_m, dtype=np.float64) - (float(n_m) - 1.0) / 2.0) * float(bar_w * 1.07)

    
    
    variant_lighten = {"full": 0.00, "no_comp": 0.18, "no_ris": 0.10}
    for j, m in enumerate(valid_methods):
        st = _TCOM_METHOD_STYLE.get(m, {"color": "#666666", "label": str(m)})
        base_color = str(st.get("color", "#666666"))
        added_label = False
        for i, v in enumerate(vlist):
            val = metric_mean.get((v, m), float("nan"))
            if not np.isfinite(val):
                continue
            xj = float(x_centers[i] + offsets[j])
            bar_color: Any
            if n_m == 1:
                bar_color = _lighten_color(base_color, float(variant_lighten.get(v, 0.0)))
            else:
                bar_color = base_color
            ax.bar(
                xj,
                float(val),
                width=float(bar_w),
                color=bar_color,
                edgecolor="#2A2A2A",
                linewidth=0.7,
                alpha=0.96,
                zorder=3,
                label=(st.get("label", m) if not added_label else None),
            )
            added_label = True

    ax.set_xticks(x_centers)
    ax.set_xticklabels([_variant_display_label(v) for v in vlist])
    ax.set_xlabel("")
    ax.set_ylabel(str(ylabel))

    
    y_lo, y_hi = ax.get_ylim()
    if np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo >= 0.0:
        ax.set_ylim(bottom=0.0)

    
    leg_ncol = int(min(4, max(1, len(valid_methods))))
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=leg_ncol,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
        borderaxespad=0.12,
    )

    
    x_pad = float(0.60 * group_gap)
    ax.set_xlim(float(x_centers[0] - x_pad), float(x_centers[-1] + x_pad))

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_highload_cdf(
    fig_path: Path,
    *,
    raw_rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    metric_key: str,
    metric_label: str,
    variant: str = "full",
    load_threshold: float = 0.8,
) -> None:
    """
    CDF
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.5))
    _beautify_ax(ax)

    plotted = 0
    x_lo = float("inf")
    x_hi = float("-inf")
    for method in methods:
        vals = []
        for rr in raw_rows:
            if str(rr.get("variant", "")).strip() != str(variant):
                continue
            if str(rr.get("method", "")).strip() != str(method):
                continue
            ld = _to_float(rr.get("load", np.nan))
            if (not np.isfinite(ld)) or (ld < float(load_threshold)):
                continue
            v = _to_float(rr.get(metric_key, np.nan))
            if np.isfinite(v):
                vals.append(float(v))
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 0:
            continue
        arr = np.sort(arr)
        cdf = np.arange(1, arr.size + 1, dtype=np.float64) / float(arr.size)
        st = _TCOM_METHOD_STYLE.get(str(method), {"color": "#000000", "marker": "o", "label": str(method), "linestyle": "-"})
        
        ax.step(
            arr, cdf,
            where="post",
            color=st["color"],
            linewidth=2.0,
            label=st.get("label", str(method)),
            zorder=3,
        )
        me = max(1, int(arr.size / 12))
        ax.plot(
            arr, cdf,
            color=st["color"],
            linestyle="None",
            marker=st.get("marker", "o"),
            markevery=me,
            markersize=5.0,
            markerfacecolor="white" if st.get("marker", "o") not in ["x", "+"] else st["color"],
            markeredgecolor=st["color"],
            markeredgewidth=1.0,
            zorder=4,
        )
        x_lo = min(x_lo, float(np.nanmin(arr)))
        x_hi = max(x_hi, float(np.nanmax(arr)))
        plotted += 1

    if plotted <= 0:
        plt.close(fig)
        return

    ax.set_xlabel(str(metric_label))
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    if np.isfinite(x_lo) and np.isfinite(x_hi) and (x_hi > x_lo):
        pad = 0.04 * (x_hi - x_lo)
        ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_title(f"High-load tail reliability ({variant}, load >= {float(load_threshold):.1f})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 4),
            bbox_to_anchor=(0.5, 1.03),
            borderaxespad=0.10,
        )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_gain_decomposition(
    fig_path: Path,
    *,
    loads: Sequence[float],
    x_values: Sequence[float],
    methods: Sequence[str],
    series: Dict[str, Dict[str, Dict[str, List[float]]]],
    metric_key: str = "paper_cost",
) -> None:
    """
    CoMP / RIS  no_comp / no_ris 
    """
    x_arr = np.asarray(list(x_values), dtype=np.float64).reshape(-1)
    l_arr = np.asarray(list(loads), dtype=np.float64).reshape(-1)
    if x_arr.size != l_arr.size or x_arr.size <= 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax in axes:
        _beautify_ax(ax)

    for method in methods:
        s_full = series.get(f"full|{method}", {})
        s_nr = series.get(f"no_ris|{method}", {})
        s_nc = series.get(f"no_comp|{method}", {})
        y_full = np.asarray(((s_full.get(metric_key, {}) or {}).get("y", [])), dtype=np.float64).reshape(-1)
        y_nr = np.asarray(((s_nr.get(metric_key, {}) or {}).get("y", [])), dtype=np.float64).reshape(-1)
        y_nc = np.asarray(((s_nc.get(metric_key, {}) or {}).get("y", [])), dtype=np.float64).reshape(-1)
        if y_full.size != x_arr.size:
            continue
        st = _TCOM_METHOD_STYLE.get(str(method), {"color": "#000000", "marker": "o", "label": str(method), "linestyle": "-"})
        me = max(1, int(x_arr.size / 10))
        if y_nc.size == x_arr.size:
            gain_comp = (y_nc - y_full) / np.maximum(np.abs(y_nc), 1e-9) * 100.0
            axes[0].plot(
                x_arr, gain_comp,
                color=st["color"], linestyle=st.get("linestyle", "-"), marker=st["marker"],
                markevery=me, linewidth=1.9, markersize=5.3,
                markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                markeredgecolor=st["color"], label=st.get("label", str(method)), zorder=3,
            )
        if y_nr.size == x_arr.size:
            gain_ris = (y_nr - y_full) / np.maximum(np.abs(y_nr), 1e-9) * 100.0
            axes[1].plot(
                x_arr, gain_ris,
                color=st["color"], linestyle=st.get("linestyle", "-"), marker=st["marker"],
                markevery=me, linewidth=1.9, markersize=5.3,
                markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                markeredgecolor=st["color"], label=st.get("label", str(method)), zorder=3,
            )

    axes[0].set_title("CoMP")
    axes[1].set_title("RIS")
    axes[0].set_xlabel(_cpu_axis_label())
    axes[1].set_xlabel(_cpu_axis_label())
    axes[0].set_ylabel(" (%)")
    _set_xlim_for_loads(axes[0], x_arr)
    _set_xlim_for_loads(axes[1], x_arr)
    h0, l0 = axes[0].get_legend_handles_labels()
    if h0:
        fig.legend(h0, l0, loc="upper center", ncol=min(len(l0), 4), bbox_to_anchor=(0.5, 1.04), borderaxespad=0.12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_delay_energy_pareto(
    fig_path: Path,
    *,
    methods: Sequence[str],
    series: Dict[str, Dict[str, Dict[str, List[float]]]],
) -> None:
    """
    Delay-Energy Pareto C 
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.7))
    _beautify_ax(ax)
    plotted = 0
    for method in methods:
        node = series.get(f"full|{method}", {})
        x = np.asarray(((node.get("mean_T_off", {}) or {}).get("y", [])), dtype=np.float64).reshape(-1)
        y = np.asarray(((node.get("mean_energy", {}) or {}).get("y", [])), dtype=np.float64).reshape(-1)
        n = int(min(x.size, y.size))
        if n <= 0:
            continue
        x = x[:n]
        y = y[:n]
        mask = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(mask)) <= 0:
            continue
        x = x[mask]
        y = y[mask]
        st = _TCOM_METHOD_STYLE.get(str(method), {"color": "#000000", "marker": "o", "label": str(method), "linestyle": "-"})
        ax.plot(
            x, y,
            color=st["color"], linestyle=st.get("linestyle", "-"), marker=st["marker"],
            linewidth=1.9, markersize=5.3,
            markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
            markeredgecolor=st["color"],
            label=st.get("label", str(method)), zorder=3,
        )
        plotted += 1
    if plotted <= 0:
        plt.close(fig)
        return
    ax.set_xlabel(" (s)")
    ax.set_ylabel(" (J)")
    ax.set_title("-")
    ax.legend(loc="upper left", ncol=1, frameon=True, framealpha=1.0, edgecolor="black", fancybox=False)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_behavior_vs_c(
    fig_path: Path,
    *,
    loads: Sequence[float],
    x_values: Sequence[float],
    methods: Sequence[str],
    grp: Dict[Tuple[str, str, float], List[Dict[str, Any]]],
    variant: str = "full",
) -> None:
    """
    |S(t)|CoMPCoMP  C 
    """
    x_arr = np.asarray(list(x_values), dtype=np.float64).reshape(-1)
    l_arr = np.asarray(list(loads), dtype=np.float64).reshape(-1)
    if x_arr.size != l_arr.size or x_arr.size <= 0:
        return

    metric_defs = [
        ("mean_S_size", " |S(t)|"),
        ("comp_active_ratio", "CoMP "),
        ("comp_gain_ratio", "CoMP "),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharex=True)
    for ax in axes:
        _beautify_ax(ax)

    for midx, (metric_key, ylab) in enumerate(metric_defs):
        ax = axes[midx]
        for method in methods:
            ys: List[float] = []
            es: List[float] = []
            for load in l_arr.tolist():
                rows = grp.get((str(variant), str(method), _load_key(load)), [])
                arr = np.asarray([rr.get(metric_key, np.nan) for rr in rows], dtype=np.float64)
                ys.append(float(np.nanmean(arr)) if arr.size else float("nan"))
                es.append(_ci95(arr) if arr.size else 0.0)
            y_arr = np.asarray(ys, dtype=np.float64)
            e_arr = np.asarray(es, dtype=np.float64)
            if int(np.sum(np.isfinite(y_arr))) <= 0:
                continue
            st = _TCOM_METHOD_STYLE.get(str(method), {"color": "#000000", "marker": "o", "label": str(method), "linestyle": "-"})
            me = max(1, int(x_arr.size / 10))
            ax.errorbar(
                x_arr, y_arr, yerr=e_arr,
                color=st["color"], linestyle=st.get("linestyle", "-"), marker=st["marker"],
                markevery=me, linewidth=1.7, markersize=4.8, capsize=2.5, elinewidth=0.9,
                markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                markeredgecolor=st["color"], label=st.get("label", str(method)), zorder=3,
            )
        ax.set_xlabel(_cpu_axis_label())
        ax.set_ylabel(ylab)
        _set_xlim_for_loads(ax, x_arr)

    h0, l0 = axes[0].get_legend_handles_labels()
    if h0:
        fig.legend(h0, l0, loc="upper center", ncol=min(len(l0), 4), bbox_to_anchor=(0.5, 1.04), borderaxespad=0.12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _find_latest_generalization_metrics_json(run_dir: Path) -> Optional[Path]:
    eval_root = Path(run_dir) / "evals"
    if not eval_root.exists():
        return None
    cands = list(eval_root.glob("*/generalization_metrics.json"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime_ns if p.exists() else -1)
    return cands[-1] if cands else None


def _plot_generalization_or_load_stress(
    fig_path: Path,
    *,
    run_dir: Path,
    methods: Sequence[str],
    loads: Sequence[float],
    x_values: Sequence[float],
    series: Dict[str, Dict[str, Dict[str, List[float]]]],
    env_cfg: JointEnvConfig,
) -> None:
    """
     generalization_metrics.json
    
    """
    x_arr = np.asarray(list(x_values), dtype=np.float64).reshape(-1)
    l_arr = np.asarray(list(loads), dtype=np.float64).reshape(-1)
    if x_arr.size != l_arr.size or x_arr.size <= 0:
        return

    metric_defs = [
        ("paper_cost", " (%)"),
        ("mean_T_off", " (%)"),
        ("mean_energy", " (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharex=True)
    for ax in axes:
        _beautify_ax(ax)

    gj = _find_latest_generalization_metrics_json(Path(run_dir))
    used_true_generalization = False
    if gj is not None and gj.exists():
        try:
            with open(gj, "r", encoding="utf-8") as f:
                gobj = json.load(f) or {}
            band_deg = gobj.get("band_degradation", {}) if isinstance(gobj, dict) else {}
            for midx, (mkey, ylab) in enumerate(metric_defs):
                ax = axes[midx]
                for method in methods:
                    node = ((band_deg.get(str(method), {}) or {}).get(str(mkey), {}) or {})
                    rows = node.get("per_load_degradation", []) if isinstance(node, dict) else []
                    if not isinstance(rows, list):
                        continue
                    ld_list: List[float] = []
                    dg_list: List[float] = []
                    for rr in rows:
                        if not isinstance(rr, dict):
                            continue
                        ld = _to_float(rr.get("load", np.nan))
                        dg = _to_float(rr.get("degradation_pct", np.nan))
                        if np.isfinite(ld) and np.isfinite(dg):
                            ld_list.append(float(ld))
                            dg_list.append(float(dg))
                    if len(ld_list) <= 0:
                        continue
                    x_mc = np.asarray([_required_cpu_mcycles_from_load(v, env_cfg) for v in ld_list], dtype=np.float64)
                    y = np.asarray(dg_list, dtype=np.float64)
                    order = np.argsort(x_mc)
                    x_mc = x_mc[order]
                    y = y[order]
                    st = _TCOM_METHOD_STYLE.get(str(method), {"color": "#000000", "marker": "o", "label": str(method), "linestyle": "-"})
                    ax.plot(
                        x_mc, y,
                        color=st["color"], linestyle=st.get("linestyle", "-"), marker=st["marker"],
                        linewidth=1.8, markersize=5.0,
                        markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                        markeredgecolor=st["color"], label=st.get("label", str(method)), zorder=3,
                    )
                ax.axhline(0.0, color="0.25", linestyle="--", linewidth=0.9, zorder=2)
                ax.set_xlabel(_cpu_axis_label())
                ax.set_ylabel(ylab)
            used_true_generalization = True
        except Exception as e:
            print(f"[HB][eval][WARN] parse generalization metrics failed: {type(e).__name__}: {e}", flush=True)

    if not used_true_generalization:
        
        for midx, (mkey, ylab) in enumerate(metric_defs):
            ax = axes[midx]
            for method in methods:
                node = series.get(f"full|{method}", {})
                y = np.asarray(((node.get(mkey, {}) or {}).get("y", [])), dtype=np.float64).reshape(-1)
                if y.size != x_arr.size:
                    continue
                finite = np.isfinite(y)
                if int(np.sum(finite)) <= 0:
                    continue
                idx0 = int(np.where(finite)[0][0])
                base = float(y[idx0])
                deg = (y - base) / max(abs(base), 1e-9) * 100.0
                st = _TCOM_METHOD_STYLE.get(str(method), {"color": "#000000", "marker": "o", "label": str(method), "linestyle": "-"})
                me = max(1, int(x_arr.size / 10))
                ax.plot(
                    x_arr, deg,
                    color=st["color"], linestyle=st.get("linestyle", "-"), marker=st["marker"],
                    markevery=me, linewidth=1.8, markersize=5.0,
                    markerfacecolor="white" if st["marker"] not in ["x", "+"] else st["color"],
                    markeredgecolor=st["color"], label=st.get("label", str(method)), zorder=3,
                )
            ax.axhline(0.0, color="0.25", linestyle="--", linewidth=0.9, zorder=2)
            ax.set_xlabel(_cpu_axis_label())
            ax.set_ylabel(ylab)
        print(
            "[HB][eval][WARN] generalization_metrics.json not found; "
            "use load-stress degradation fallback in Ablation10_Generalization_Degradation.png",
            flush=True,
        )

    h0, l0 = axes[0].get_legend_handles_labels()
    if h0:
        fig.legend(
            h0,
            l0,
            loc="upper center",
            ncol=min(len(l0), 4),
            bbox_to_anchor=(0.5, 1.015),
            borderaxespad=0.08,
        )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_fixed_eval_variant_overlay(
    fig_path: Path,
    curves_by_variant: Dict[str, Dict[str, List[float]]],
    variants_order: List[str],
    *,
    quota_note: Optional[str] = None,
    ema_span: int = 48,
) -> None:
    """
     fixed-eval full / no_ris / no_comp

    
    -  paper_cost
    -  train_episode ckpt_step
    -  raw+ EMA
    """
    span_eff = int(max(1, int(ema_span)))
    alpha = float(2.0 / (span_eff + 1.0))
    style_map = {
        "full": {"color": "#0072BD", "label": ""},
        "no_ris": {"color": "#D95319", "label": "RIS"},
        "no_comp": {"color": "#2CA02C", "label": "CoMP"},
    }

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.4))
    _beautify_ax(ax)

    plotted = 0
    for vname in variants_order:
        cobj = curves_by_variant.get(str(vname), {})
        steps = np.asarray(cobj.get("steps", []), dtype=np.float64).reshape(-1)
        episodes = np.asarray(cobj.get("episodes", []), dtype=np.float64).reshape(-1)
        y = np.asarray(cobj.get("cost_mean", []), dtype=np.float64).reshape(-1)
        if steps.size <= 0 or y.size <= 0:
            continue

        n = int(min(steps.size, y.size, episodes.size if episodes.size > 0 else steps.size))
        if n <= 0:
            continue
        steps = steps[:n]
        y = y[:n]
        if episodes.size > 0:
            episodes = episodes[:n]
        else:
            episodes = np.full((n,), np.nan, dtype=np.float64)

        
        ord_step = np.argsort(steps)
        steps = steps[ord_step]
        episodes = episodes[ord_step]
        y = y[ord_step]

        x = np.where(np.isfinite(episodes) & (episodes >= 0.0), episodes, steps)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        x = np.asarray(x[valid], dtype=np.float64)
        y = np.asarray(y[valid], dtype=np.float64)
        if x.size <= 0:
            continue

        
        ord_x = np.argsort(x)
        x = x[ord_x]
        y = y[ord_x]

        y_ema = y.copy()
        if y.size >= 2:
            try:
                from src.utils.plot_smoothing import exponential_moving_average

                y_ema = exponential_moving_average(y, alpha=alpha)
            except Exception:
                y_ema = y

        st = style_map.get(str(vname), {"color": "#444444", "label": str(vname)})
        ax.plot(
            x,
            y,
            color=st["color"],
            linewidth=1.2,
            alpha=0.35,
            linestyle="-",
            label=f'{st["label"]} raw',
            zorder=2,
        )
        ax.plot(
            x,
            y_ema,
            color=st["color"],
            linewidth=2.6,
            alpha=0.95,
            linestyle="-",
            label=f'{st["label"]} EMA(span={span_eff})',
            zorder=3,
        )
        plotted += 1

    if plotted <= 0:
        print(f"[HB][fixed_eval][WARN] overlay plot skipped: no valid points ({fig_path})", flush=True)
        plt.close(fig)
        return

    ax.set_title("", pad=8, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="lower right", ncol=1)

    variants_text = "".join([_variant_display_label(v) for v in variants_order])
    note = f"\n{variants_text}\nEMA(span={span_eff})"
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

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    plt.savefig(str(fig_path), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[HB][fixed_eval] overlay plot saved: {fig_path}", flush=True)


def _apply_baseline_mode_env_override(env: CompRISEnvJoint, method: str) -> None:
    """
     always_comp / never_comp  full/hierarchical 

    
    - never_comp: enable_comp=0, K=1
    - always_comp:  CoMP  K>=2 full 
    """
    m = str(method).strip().lower()
    if m not in ("always_comp", "never_comp"):
        return

    action_space_mode = str(getattr(env.cfg, "action_space_mode", "full")).strip().lower()
    k_now = int(max(int(getattr(env.cfg, "K", 1)), 1))
    comp_on = bool(getattr(env.cfg, "enable_comp", True))

    if m == "never_comp":
        try:
            setattr(env.cfg, "enable_comp", False)
            setattr(env.cfg, "K", 1)
        except Exception as exc:
            raise RuntimeError(
                f"[HB][eval][FATAL] never_comp {type(exc).__name__}: {exc}"
            ) from exc
        _warn_once(
            "baseline_never_comp_override",
            "never_comp  enable_comp=0, K=1",
        )
        return

    # always_comp
    if not comp_on:
        _warn_once(
            "baseline_always_comp_disabled_by_variant",
            "always_comp  enable_comp=0 no_comp ",
        )
        return

    if action_space_mode != "hierarchical" and k_now < 2:
        try:
            setattr(env.cfg, "K", 2)
        except Exception as exc:
            raise RuntimeError(
                f"[HB][eval][FATAL] always_comp K2{type(exc).__name__}: {exc}"
            ) from exc
        _warn_once(
            "baseline_always_comp_k_override",
            "always_comp  full  K  2",
        )


def run_one_episode(env: CompRISEnvJoint, method: str, agent: Any, deterministic: bool) -> Dict[str, Any]:
    """
     episode info
    H4.C
    - traj_idle_ratio
    - traj_boundary_stick_frac
    - traj_user_nn_dist_norm
    - traj_centroid_gap_norm
    - traj_switchback_ratio
    """
    method_n = str(method).strip().lower()
    _apply_baseline_mode_env_override(env, method_n)
    _ = env.reset()
    done = False

    
    ep_len = 0
    ep_paper_cost_sum = 0.0
    ep_penalty_sum = 0.0
    ep_vio_cnt_sum = 0.0
    
    ep_mean_rate_sum = 0.0
    ep_mean_T_tx_sum = 0.0
    ep_mean_T_off_sum = 0.0
    ep_mean_energy_sum = 0.0
    ep_mean_S_size_sum = 0.0
    ep_comp_active_ratio_sum = 0.0
    ep_comp_gain_ratio_sum = 0.0
    ep_r_cost_sum = 0.0
    ep_r_vio_sum = 0.0
    ep_r_imp_sum = 0.0

    
    traj_idle_hits = 0.0
    traj_idle_total = 0.0
    traj_boundary_hits = 0.0
    traj_boundary_total = 0.0
    traj_user_nn_sum = 0.0
    traj_user_nn_total = 0.0
    traj_centroid_sum = 0.0
    traj_centroid_total = 0.0
    traj_switchback_hits = 0.0
    traj_switchback_total = 0.0
    q_prev = None
    vel_prev = None
    try:
        L = float(getattr(env.cfg, "L", 1.0))
        boundary_band = 0.05 * float(L)
        vnorm_denom = max(1e-9, float(getattr(env.cfg, "Vmax", 1.0)) * float(getattr(env.cfg, "dt", 1.0)))
    except Exception:
        L = 1.0
        boundary_band = 0.05
        vnorm_denom = 1.0

    last_info: Dict[str, Any] = {}

    while not done:
        obs_flat = env.obs_flat()
        
        
        
        
        obs_in_ppo = obs_flat
        if _EVAL_OBS_NORMALIZER is not None:
            obs_in_ppo = _EVAL_OBS_NORMALIZER.normalize(obs_flat, update=False)

        if method_n == "ppo":
            assert agent is not None
            
            try:
                act, _logp, _v = agent.act(obs_in_ppo, deterministic=deterministic, obs_raw=obs_flat)
            except TypeError:
                act, _logp, _v = agent.act(obs_in_ppo, deterministic=deterministic)
            act = _fairify_action(env, act)

        elif method_n in ("ddpg", "td3", "sac", "a2c"):
            assert agent is not None
            
            obs_t = torch.from_numpy(np.asarray(obs_flat, dtype=np.float32)).float().unsqueeze(0).to(getattr(agent, "device", torch.device("cpu")))
            with torch.no_grad():
                if method_n in ("ddpg", "td3"):
                    
                    act_t = agent.get_action(obs_t, add_noise=(not bool(deterministic)))
                elif method_n == "a2c":
                    
                    act_t = agent.get_action(obs_t, deterministic=bool(deterministic))
                else:
                    act_t = agent.get_action(obs_t, deterministic=bool(deterministic))
            if isinstance(act_t, torch.Tensor):
                act = act_t.squeeze(0).detach().cpu().numpy()
            else:
                
                
                
                
                act_np = np.asarray(act_t, dtype=np.float32)
                if act_np.ndim >= 2 and int(act_np.shape[0]) == 1:
                    act_np = act_np[0]
                act = act_np.reshape(-1)
            
            act = _fairify_action(env, act)

        else:
            
            act = baseline_action(env, method_n)
            act = _fairify_action(env, act)

        _, _r, done, info = env.step(act)

        
        try:
            q_now = np.asarray(getattr(env, "q", None), dtype=np.float64)
            if q_now.ndim == 2 and q_now.shape[1] == 2 and q_now.shape[0] > 0:
                M_now = int(q_now.shape[0])
                # boundary stick
                d_left = q_now[:, 0]
                d_right = float(L) - q_now[:, 0]
                d_bottom = q_now[:, 1]
                d_top = float(L) - q_now[:, 1]
                d_min = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
                traj_boundary_hits += float(np.sum(d_min <= float(boundary_band)))
                traj_boundary_total += float(M_now)

                # user distance + centroid gap
                w_now = np.asarray(getattr(env, "w", None), dtype=np.float64)
                if w_now.ndim == 2 and w_now.shape[1] == 2 and w_now.shape[0] > 0:
                    diff = q_now[:, None, :] - w_now[None, :, :]
                    d = np.sqrt(np.sum(diff * diff, axis=2))
                    d_nn = np.min(d, axis=1)
                    traj_user_nn_sum += float(np.sum(d_nn) / max(float(L), 1e-9))
                    traj_user_nn_total += float(M_now)
                    gap = np.linalg.norm(np.mean(q_now, axis=0) - np.mean(w_now, axis=0))
                    traj_centroid_sum += float(gap / max(float(L), 1e-9))
                    traj_centroid_total += 1.0

                if q_prev is not None and np.asarray(q_prev).shape == q_now.shape:
                    vel = q_now - q_prev
                    sp = np.linalg.norm(vel, axis=1) / float(vnorm_denom)
                    traj_idle_hits += float(np.sum(sp < 0.08))
                    traj_idle_total += float(M_now)

                    if vel_prev is not None and np.asarray(vel_prev).shape == vel.shape:
                        dot = np.sum(vel * vel_prev, axis=1)
                        n0 = np.linalg.norm(vel_prev, axis=1)
                        n1 = np.linalg.norm(vel, axis=1)
                        valid = (n0 > 1e-9) & (n1 > 1e-9)
                        if np.any(valid):
                            cosv = dot[valid] / (n0[valid] * n1[valid] + 1e-12)
                            traj_switchback_hits += float(np.sum(cosv < 0.0))
                            traj_switchback_total += float(np.sum(valid))
                    vel_prev = vel
                else:
                    vel_prev = None
                q_prev = q_now.copy()
        except Exception as exc:
            _warn_once(
                "eval_episode_traj_metrics",
                f"{type(exc).__name__}: {exc}",
            )

        paper_cost = _pick_first_finite(info, ("policy_paper_cost", "paper_cost", "cost"), default=0.0)
        penalty_total = _pick_first_finite(info, ("vio_total_penalty", "penalty_total", "cost_constraint"), default=0.0)
        vio_cnt = _pick_first_finite(info, ("violation_count", "violations"), default=0.0)

        ep_paper_cost_sum += paper_cost
        ep_penalty_sum += penalty_total
        ep_vio_cnt_sum += vio_cnt

        ep_mean_rate_sum += _to_float(info.get("mean_rate", 0.0), default=0.0)
        ep_mean_T_tx_sum += _to_float(info.get("mean_T_tx", 0.0), default=0.0)
        ep_mean_T_off_sum += _to_float(info.get("mean_T_off", 0.0), default=0.0)
        ep_mean_energy_sum += _to_float(info.get("mean_energy", 0.0), default=0.0)
        ep_mean_S_size_sum += _to_float(info.get("mean_S_size", 0.0), default=0.0)
        ep_comp_active_ratio_sum += _to_float(info.get("comp_active_ratio", 0.0), default=0.0)
        ep_comp_gain_ratio_sum += _to_float(info.get("comp_gain_ratio", 0.0), default=0.0)

        ep_r_cost_sum += _to_float(info.get("reward_cost", 0.0), default=0.0)
        ep_r_vio_sum += _to_float(info.get("reward_vio", 0.0), default=0.0)
        ep_r_imp_sum += _to_float(info.get("reward_improve", 0.0), default=0.0)

        ep_len += 1
        last_info = dict(info)

    steps = max(ep_len, 1)
    paper_cost_ep_sum = float(ep_paper_cost_sum)
    paper_cost_ep_mean = float(ep_paper_cost_sum / steps)
    traj_idle_ratio = float(traj_idle_hits / max(1.0, traj_idle_total))
    traj_boundary_stick_frac = float(traj_boundary_hits / max(1.0, traj_boundary_total))
    traj_user_nn_dist_norm = float(traj_user_nn_sum / max(1.0, traj_user_nn_total))
    traj_centroid_gap_norm = float(traj_centroid_sum / max(1.0, traj_centroid_total))
    traj_switchback_ratio = float(traj_switchback_hits / max(1.0, traj_switchback_total))
    last_info.update({
        
        
        
        "paper_cost": paper_cost_ep_mean,
        
        "paper_cost_ep_sum": paper_cost_ep_sum,
        "paper_cost_ep_mean": paper_cost_ep_mean,
        
        "violation_count": ep_vio_cnt_sum / steps,
        "mean_rate": ep_mean_rate_sum / steps,
        "mean_T_tx": ep_mean_T_tx_sum / steps,
        "mean_T_off": ep_mean_T_off_sum / steps,
        "mean_energy": ep_mean_energy_sum / steps,
        "mean_S_size": ep_mean_S_size_sum / steps,
        "comp_active_ratio": ep_comp_active_ratio_sum / steps,
        "comp_gain_ratio": ep_comp_gain_ratio_sum / steps,
        "paper_cost_mean": paper_cost_ep_mean,
        "penalty_total_mean": ep_penalty_sum / steps,
        "violation_count_mean": ep_vio_cnt_sum / steps,
        "r_cost_mean": ep_r_cost_sum / steps,
        "r_vio_mean": ep_r_vio_sum / steps,
        "r_imp_mean": ep_r_imp_sum / steps,
        "ep_len": ep_len,
        "traj_idle_ratio": traj_idle_ratio,
        "traj_boundary_stick_frac": traj_boundary_stick_frac,
        "traj_user_nn_dist_norm": traj_user_nn_dist_norm,
        "traj_centroid_gap_norm": traj_centroid_gap_norm,
        "traj_switchback_ratio": traj_switchback_ratio,
    })
    return last_info


def _run_fixed_eval_curve(
    *,
    train_run_dir: Path,
    base_env_cfg: JointEnvConfig,
    cfg_train: dict,
    device: str,
    eval_ts: str,
    seed_set_size: int,
    episodes_per_seed: int,
    deterministic: bool,
    is_stochastic: bool,
    resume: bool,
    fixed_variants: Optional[List[str]] = None,
    fixed_eval_ckpt_path: Optional[Path] = None,
) -> None:
    """
    PhaseF fixed-eval curve with variant-aware reproducibility.
    """
    train_run_dir = Path(train_run_dir)
    eval_dir = train_run_dir / "evals" / eval_ts
    fig_dir = train_run_dir / "figs" / f"{eval_ts}{FIG_EVAL_TS_SUFFIX}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ppo_use_obs_norm = _parse_bool(os.environ.get("EVAL_PPO_USE_OBS_NORM", "1"), True)

    suffix = "_stoch" if bool(is_stochastic) else ""
    curve_csv = eval_dir / f"fixed_eval_curve{suffix}.csv"
    summary_csv = eval_dir / f"fixed_eval_summary{suffix}.csv"

    rd = str(getattr(base_env_cfg, "reward_design", "")).strip().lower()
    use_phaseg_ep_sum = bool(rd in ("reward_total_v1", "reward_total", "total_v1", "v1"))
    paper_cost_agg = "SUM" if bool(use_phaseg_ep_sum) else "MEAN"
    train_load = float(getattr(base_env_cfg, "load_scale", 1.0))
    contract_version = "phaseg_v1.1_sum_stepclip" if bool(use_phaseg_ep_sum) else "legacy_mean_or_non_phaseg"
    
    fig_name = f"FixedEval_PaperCost{suffix}.png"

    if fixed_eval_ckpt_path is not None:
        ckpts = _resolve_fixed_eval_single_ckpt(train_run_dir, Path(fixed_eval_ckpt_path))
        print(
            f"[HB][fixed_eval] single-ckpt mode: ckpt={ckpts[0][1]} step={int(ckpts[0][0])}",
            flush=True,
        )
    else:
        ckpts = _find_fixed_eval_ckpts(train_run_dir)
    if not ckpts:
        raise RuntimeError(f"[HB][fixed_eval][FATAL] No ckpts found under: {train_run_dir}")

    eval_cfg = cfg_train.get("eval", {}) or {}
    seeds_cfg = eval_cfg.get("eval_seeds", None)
    if isinstance(seeds_cfg, list) and seeds_cfg:
        seeds = [int(x) for x in seeds_cfg]
    else:
        seeds = [42 + i for i in range(int(max(1, seed_set_size)))]

    seed_set_size = int(max(1, seed_set_size))
    episodes_per_seed = int(max(1, episodes_per_seed))
    if len(seeds) < seed_set_size:
        start = int(max(seeds) + 1) if seeds else 42
        seeds = list(seeds) + [start + i for i in range(seed_set_size - len(seeds))]
    seeds = list(seeds[:seed_set_size])

    loads_cfg = eval_cfg.get("eval_loads", None)
    load0 = 1.0
    try:
        if isinstance(loads_cfg, list) and loads_cfg:
            load0 = float(loads_cfg[0])
    except Exception:
        load0 = 1.0

    variant_map = make_variants(base_env_cfg)
    req_variants = list(fixed_variants) if isinstance(fixed_variants, list) else ["full"]
    req_variants = [str(v).strip().lower() for v in req_variants if str(v).strip()]
    if not req_variants:
        req_variants = ["full"]
    req_variants = list(dict.fromkeys(req_variants))

    fixed_variants_norm: List[str] = []
    for v in req_variants:
        if v in variant_map:
            fixed_variants_norm.append(v)
    if not fixed_variants_norm:
        fixed_variants_norm = ["full"] if "full" in variant_map else [list(variant_map.keys())[0]]
    missing_variants = [v for v in req_variants if v not in variant_map]
    if missing_variants:
        print(
            f"[HB][fixed_eval][WARN] unknown variants ignored: {missing_variants}; "
            f"use={fixed_variants_norm}",
            flush=True,
        )

    total_traj = int(len(ckpts) * len(fixed_variants_norm) * len(seeds) * episodes_per_seed)
    print(
        f"[HB][fixed_eval] train_run_dir={train_run_dir} eval_ts={eval_ts} "
        f"variants={fixed_variants_norm} ckpts={len(ckpts)} "
        f"seeds={len(seeds)} eps/seed={episodes_per_seed} total_traj={total_traj} "
        f"deterministic={1 if deterministic else 0} paper_cost_agg={paper_cost_agg} "
        f"train_load={train_load:.2f} eval_load={load0:.2f} out={eval_dir}",
        flush=True,
    )

    curve_cols = [
        "ckpt_step",
        "eval_train_episode",
        "variant",
        "seed",
        "ep",
        "paper_cost_ep",
        "paper_return_ep",
        "vio_any",
        "traj_idle_ratio",
        "traj_boundary_stick_frac",
        "traj_user_nn_dist_norm",
        "traj_centroid_gap_norm",
        "traj_switchback_ratio",
        "mean_T_off_ep",
        "mean_energy_ep",
        "paper_cost_agg",
        "eval_load",
    ]
    sum_cols = [
        "ckpt_step",
        "eval_train_episode",
        "variant",
        "n_seeds",
        "episodes_per_seed",
        "total_traj",
        "deterministic",
        "paper_cost_mean",
        "paper_cost_std",
        "paper_cost_ci95",
        "paper_return_mean",
        "paper_return_std",
        "paper_return_ci95",
        "vio_any_mean",
        "vio_any_std",
        "vio_any_ci95",
        "traj_idle_ratio_mean",
        "traj_idle_ratio_std",
        "traj_idle_ratio_ci95",
        "traj_boundary_stick_frac_mean",
        "traj_boundary_stick_frac_std",
        "traj_boundary_stick_frac_ci95",
        "traj_user_nn_dist_norm_mean",
        "traj_user_nn_dist_norm_std",
        "traj_user_nn_dist_norm_ci95",
        "traj_centroid_gap_norm_mean",
        "traj_centroid_gap_norm_std",
        "traj_centroid_gap_norm_ci95",
        "traj_switchback_ratio_mean",
        "traj_switchback_ratio_std",
        "traj_switchback_ratio_ci95",
        "delay_mean",
        "delay_std",
        "delay_ci95",
        "energy_mean",
        "energy_std",
        "energy_ci95",
        "paper_cost_agg",
        "reward_design",
        "train_load",
        "eval_load_primary",
        "contract_version",
    ]

    do_resume = bool(resume) and curve_csv.exists() and summary_csv.exists() and (curve_csv.stat().st_size > 0) and (summary_csv.stat().st_size > 0)

    
    variant_curves: Dict[str, Dict[str, List[float]]] = {
        str(v): {
            "steps": [],
            "episodes": [],
            "cost_mean": [],
            "cost_ci": [],
            "vio_mean": [],
            "vio_ci": [],
        }
        for v in fixed_variants_norm
    }
    done_pairs: set = set()

    if do_resume:
        try:
            with open(curve_csv, "r", encoding="utf-8", newline="") as f:
                rr = csv.DictReader(f)
                fn = list(rr.fieldnames or [])
                required = {"ckpt_step", "eval_train_episode", "traj_idle_ratio", "variant", "mean_T_off_ep", "mean_energy_ep"}
                if not required.issubset(set(fn)):
                    do_resume = False
                    print(
                        "[HB][fixed_eval][WARN] resume disabled: legacy fixed_eval_curve header (missing required cols)",
                        flush=True,
                    )
        except Exception:
            do_resume = False

    if do_resume:
        try:
            with open(summary_csv, "r", encoding="utf-8", newline="") as f:
                rr = csv.DictReader(f)
                fn = list(rr.fieldnames or [])
                required = {"ckpt_step", "eval_train_episode", "traj_idle_ratio_mean", "variant", "delay_mean", "energy_mean"}
                if not required.issubset(set(fn)):
                    do_resume = False
                    print(
                        "[HB][fixed_eval][WARN] resume disabled: legacy fixed_eval_summary header (missing required cols)",
                        flush=True,
                    )
                else:
                    existing_variants: set = set()
                    for row in rr:
                        try:
                            st = int(float(row.get("ckpt_step", -1)))
                        except Exception:
                            continue
                        if st < 0:
                            continue
                        vname = str(row.get("variant", "full")).strip().lower() or "full"
                        existing_variants.add(vname)
                        if vname not in fixed_variants_norm:
                            continue
                        done_pairs.add((vname, int(st)))
                        if vname in variant_curves:
                            variant_curves[vname]["steps"].append(int(st))
                            try:
                                ep_val = int(float(row.get("eval_train_episode", float("nan"))))
                            except Exception:
                                ep_val = -1
                            variant_curves[vname]["episodes"].append(int(ep_val))
                            variant_curves[vname]["cost_mean"].append(float(row.get("paper_cost_mean", float("nan"))))
                            variant_curves[vname]["cost_ci"].append(float(row.get("paper_cost_ci95", float("nan"))))
                            variant_curves[vname]["vio_mean"].append(float(row.get("vio_any_mean", float("nan"))))
                            variant_curves[vname]["vio_ci"].append(float(row.get("vio_any_ci95", float("nan"))))
                    if not existing_variants.issubset(set(fixed_variants_norm)):
                        do_resume = False
        except Exception:
            do_resume = False

    if do_resume and done_pairs:
        try:
            with open(summary_csv, "r", encoding="utf-8", newline="") as f:
                rr = csv.DictReader(f)
                first = next(rr, None)
            if first is None:
                do_resume = False
            else:
                n_seeds_old = int(float(first.get("n_seeds", -1)))
                ep_old = int(float(first.get("episodes_per_seed", -1)))
                det_old = int(float(first.get("deterministic", -1)))
                agg_old = str(first.get("paper_cost_agg", "")).strip().upper() or "UNKNOWN"
                load_old = float(first.get("eval_load_primary", float("nan")))
                agg_now = str(paper_cost_agg).strip().upper()
                det_now = 1 if bool(deterministic) else 0
                same_load = bool(np.isfinite(load_old) and abs(load_old - float(load0)) <= 1e-9)
                if (n_seeds_old != int(len(seeds))) or (ep_old != int(episodes_per_seed)) or (det_old != det_now) or (agg_old != agg_now) or (not same_load):
                    print(
                        f"[HB][fixed_eval][WARN] resume disabled due quota mismatch: "
                        f"(old n_seeds={n_seeds_old}, ep/seed={ep_old}, det={det_old}, agg={agg_old}, eval_load={load_old}) "
                        f"vs (new n_seeds={len(seeds)}, ep/seed={episodes_per_seed}, det={det_now}, agg={agg_now}, eval_load={load0})",
                        flush=True,
                    )
                    do_resume = False
        except Exception:
            do_resume = False

    if do_resume:
        print(
            f"[HB][fixed_eval] RESUME enabled: done={len(done_pairs)}/{len(ckpts) * len(fixed_variants_norm)} "
            f"summary={summary_csv}",
            flush=True,
        )
    else:
        with open(curve_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(curve_cols)
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(sum_cols)
        variant_curves = {
            str(v): {
                "steps": [],
                "episodes": [],
                "cost_mean": [],
                "cost_ci": [],
                "vio_mean": [],
                "vio_ci": [],
            }
            for v in fixed_variants_norm
        }
        done_pairs = set()

    curve_f = open(curve_csv, "a", encoding="utf-8", newline="")
    curve_w = csv.writer(curve_f)
    summary_f = open(summary_csv, "a", encoding="utf-8", newline="")
    summary_w = csv.writer(summary_f)

    step_gs = None
    step_te = None
    try:
        from src.utils.train_report import read_train_metrics_csv

        mm = read_train_metrics_csv(train_run_dir / "train_metrics.csv")
        step_gs = np.asarray(mm.get("global_step", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        step_te = np.asarray(mm.get("train_episode", np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        if step_gs.size <= 0 or step_te.size != step_gs.size:
            step_gs = None
            step_te = None
    except Exception:
        step_gs = None
        step_te = None

    def _episode_at_ckpt(ckpt_step: int, ckpt_path: Path) -> int:
        try:
            obj = _torch_load(ckpt_path, map_location="cpu")
            if isinstance(obj, dict):
                meta = obj.get("meta", {}) or {}
                v = meta.get("train_episode", obj.get("train_episode", None))
                if v is not None:
                    return int(float(v))
        except Exception as exc:
            _warn_once("fixed_eval_ckpt_episode_parse", f"ckpttrain_episodeCSV{type(exc).__name__}: {exc}")
        if step_gs is not None and step_te is not None:
            try:
                idx = int(np.searchsorted(step_gs, float(ckpt_step), side="right") - 1)
                if 0 <= idx < int(step_te.size) and np.isfinite(step_te[idx]):
                    return int(step_te[idx])
            except Exception as exc:
                _warn_once("fixed_eval_csv_episode_map", f"ckpt_step->episode {type(exc).__name__}: {exc}")
        return -1

    k_done = int(len(done_pairs) * len(seeds) * episodes_per_seed)
    k = 0
    t0 = time.perf_counter()
    hb_every = int(max(50, total_traj // 50))
    try:
        for ckpt_step, ckpt_path in ckpts:
            pending_variants = [v for v in fixed_variants_norm if (v, int(ckpt_step)) not in done_pairs]
            if not pending_variants:
                continue

            eval_train_ep = int(_episode_at_ckpt(int(ckpt_step), Path(ckpt_path)))
            if eval_train_ep < 0:
                print(
                    f"[HB][fixed_eval][WARN] missing eval_train_episode: ckpt_step={int(ckpt_step)} ckpt={ckpt_path}",
                    flush=True,
                )

            seed0 = int(seeds[0]) if seeds else 42
            agent, used_ckpt, obs_dim, _act_dim, _hidden = _load_ppo_agent_from_ckpt(
                train_run_dir=train_run_dir,
                ckpt_path=ckpt_path,
                base_env_cfg=base_env_cfg,
                seed0=seed0,
                cfg_train=cfg_train,
                device=device,
            )
            global _EVAL_OBS_NORMALIZER
            try:
                clip = float((cfg_train.get("train", {}) or {}).get("obs_clip", 10.0))
            except Exception:
                clip = 10.0
            if ppo_use_obs_norm:
                _EVAL_OBS_NORMALIZER = _maybe_load_obs_normalizer(train_run_dir, used_ckpt, obs_dim, clip=clip)
            else:
                _EVAL_OBS_NORMALIZER = None
                print("[HB][fixed_eval] EVAL_PPO_USE_OBS_NORM=0PPORAW obs_flat()", flush=True)

            for vname in pending_variants:
                vcfg = variant_map[vname]
                seed_cost_means: List[float] = []
                seed_ret_means: List[float] = []
                seed_vio_means: List[float] = []
                seed_traj_idle_means: List[float] = []
                seed_traj_boundary_means: List[float] = []
                seed_traj_user_nn_means: List[float] = []
                seed_traj_centroid_means: List[float] = []
                seed_traj_switchback_means: List[float] = []
                seed_delay_means: List[float] = []
                seed_energy_means: List[float] = []

                for seed in seeds:
                    env = CompRISEnvJoint(vcfg, seed=int(seed), reward_mode=str(cfg_train.get("reward_mode", "p1")))
                    try:
                        env.set_load_scale(float(load0))  # type: ignore
                    except Exception:
                        try:
                            env.cfg.load_scale = float(load0)  # type: ignore
                        except Exception as exc:
                            raise RuntimeError(
                                f"[HB][fixed_eval][FATAL] load_scaleload={float(load0):.4f}"
                                f"{type(exc).__name__}: {exc}"
                            ) from exc

                    cost_sum = 0.0
                    ret_sum = 0.0
                    vio_sum = 0.0
                    traj_idle_sum = 0.0
                    traj_idle_cnt = 0
                    traj_boundary_sum = 0.0
                    traj_boundary_cnt = 0
                    traj_user_nn_sum = 0.0
                    traj_user_nn_cnt = 0
                    traj_centroid_sum = 0.0
                    traj_centroid_cnt = 0
                    traj_switchback_sum = 0.0
                    traj_switchback_cnt = 0
                    delay_sum = 0.0
                    delay_cnt = 0
                    energy_sum = 0.0
                    energy_cnt = 0

                    for ep in range(episodes_per_seed):
                        
                        action_seed = _stable_eval_seed(
                            (eval_ts, "fixed_eval", int(ckpt_step), str(vname), int(seed), int(ep))
                        )
                        set_global_seed(action_seed, deterministic=bool(deterministic))
                        info = run_one_episode(env, "ppo", agent, deterministic=bool(deterministic))
                        if bool(use_phaseg_ep_sum):
                            paper_cost_ep = _pick_first_finite(
                                info,
                                ("paper_cost_ep_sum", "paper_cost_sum", "paper_cost", "paper_cost_mean"),
                                default=0.0,
                            )
                        else:
                            paper_cost_ep = _pick_first_finite(
                                info,
                                ("paper_cost_ep_mean", "paper_cost", "paper_cost_mean"),
                                default=0.0,
                            )
                        paper_return_ep = -paper_cost_ep
                        vio_cnt = _pick_first_finite(info, ("violation_count", "penalty_total_mean"), default=0.0)
                        vio_any = 1.0 if np.isfinite(vio_cnt) and float(vio_cnt) > 0.0 else 0.0
                        traj_idle = _to_float(info.get("traj_idle_ratio", float("nan")))
                        traj_boundary = _to_float(info.get("traj_boundary_stick_frac", float("nan")))
                        traj_user_nn = _to_float(info.get("traj_user_nn_dist_norm", float("nan")))
                        traj_centroid = _to_float(info.get("traj_centroid_gap_norm", float("nan")))
                        traj_switchback = _to_float(info.get("traj_switchback_ratio", float("nan")))
                        
                        delay_ep = _pick_first_finite(info, ("mean_T_off", "mean_delay", "mean_T_tx"), default=float("nan"))
                        energy_ep = _pick_first_finite(info, ("mean_energy", "energy_per_bit"), default=float("nan"))

                        curve_w.writerow(
                            [
                                int(ckpt_step),
                                int(eval_train_ep),
                                str(vname),
                                int(seed),
                                int(ep),
                                float(paper_cost_ep),
                                float(paper_return_ep),
                                float(vio_any),
                                float(traj_idle),
                                float(traj_boundary),
                                float(traj_user_nn),
                                float(traj_centroid),
                                float(traj_switchback),
                                float(delay_ep),
                                float(energy_ep),
                                str(paper_cost_agg),
                                float(load0),
                            ]
                        )

                        cost_sum += float(paper_cost_ep)
                        ret_sum += float(paper_return_ep)
                        vio_sum += float(vio_any)
                        traj_idle_sum, traj_idle_cnt = _acc_finite(traj_idle_sum, traj_idle_cnt, traj_idle)
                        traj_boundary_sum, traj_boundary_cnt = _acc_finite(traj_boundary_sum, traj_boundary_cnt, traj_boundary)
                        traj_user_nn_sum, traj_user_nn_cnt = _acc_finite(traj_user_nn_sum, traj_user_nn_cnt, traj_user_nn)
                        traj_centroid_sum, traj_centroid_cnt = _acc_finite(traj_centroid_sum, traj_centroid_cnt, traj_centroid)
                        traj_switchback_sum, traj_switchback_cnt = _acc_finite(traj_switchback_sum, traj_switchback_cnt, traj_switchback)
                        delay_sum, delay_cnt = _acc_finite(delay_sum, delay_cnt, delay_ep)
                        energy_sum, energy_cnt = _acc_finite(energy_sum, energy_cnt, energy_ep)

                        k += 1
                        k_all = int(k_done + k)
                        if hb_every > 0 and (k_all % hb_every == 0 or k_all == 1 or k_all == total_traj):
                            elapsed = float(time.perf_counter() - t0)
                            avg = elapsed / max(1, k)
                            eta = avg * float(max(0, total_traj - k_all))
                            pct = 100.0 * float(k_all) / float(max(1, total_traj))
                            print(
                                f"[HB][fixed_eval] {k_all}/{total_traj} ({pct:.1f}%) "
                                f"variant={vname} ckpt_step={int(ckpt_step)} seed={int(seed)} ep={int(ep)} "
                                f"avg={avg:.3f}s ETA={_fmt_hms(eta)} curve={curve_csv}",
                                flush=True,
                            )
                            try:
                                curve_f.flush()
                            except Exception as exc:
                                _warn_once(
                                    "fixed_eval_curve_flush",
                                    f"fixed_eval CSV flush {type(exc).__name__}: {exc}",
                                )

                    seed_cost_means.append(cost_sum / float(max(1, episodes_per_seed)))
                    seed_ret_means.append(ret_sum / float(max(1, episodes_per_seed)))
                    seed_vio_means.append(vio_sum / float(max(1, episodes_per_seed)))
                    seed_traj_idle_means.append(traj_idle_sum / float(max(1, traj_idle_cnt)))
                    seed_traj_boundary_means.append(traj_boundary_sum / float(max(1, traj_boundary_cnt)))
                    seed_traj_user_nn_means.append(traj_user_nn_sum / float(max(1, traj_user_nn_cnt)))
                    seed_traj_centroid_means.append(traj_centroid_sum / float(max(1, traj_centroid_cnt)))
                    seed_traj_switchback_means.append(traj_switchback_sum / float(max(1, traj_switchback_cnt)))
                    seed_delay_means.append(delay_sum / float(max(1, delay_cnt)))
                    seed_energy_means.append(energy_sum / float(max(1, energy_cnt)))

                cost_arr = np.asarray(seed_cost_means, dtype=np.float64)
                ret_arr = np.asarray(seed_ret_means, dtype=np.float64)
                vio_arr = np.asarray(seed_vio_means, dtype=np.float64)
                traj_idle_arr = np.asarray(seed_traj_idle_means, dtype=np.float64)
                traj_boundary_arr = np.asarray(seed_traj_boundary_means, dtype=np.float64)
                traj_user_nn_arr = np.asarray(seed_traj_user_nn_means, dtype=np.float64)
                traj_centroid_arr = np.asarray(seed_traj_centroid_means, dtype=np.float64)
                traj_switchback_arr = np.asarray(seed_traj_switchback_means, dtype=np.float64)
                delay_arr = np.asarray(seed_delay_means, dtype=np.float64)
                energy_arr = np.asarray(seed_energy_means, dtype=np.float64)
                n_seed = int(cost_arr.size)

                def _mean_std_ci(a: np.ndarray) -> Tuple[float, float, float]:
                    a = np.asarray(a, dtype=np.float64).reshape(-1)
                    a = a[np.isfinite(a)]
                    n = int(a.size)
                    if n <= 0:
                        return float("nan"), float("nan"), float("nan")
                    mu = float(np.mean(a))
                    std = float(np.std(a, ddof=1)) if n > 1 else 0.0
                    ci = float(1.96 * std / np.sqrt(max(1, n))) if n > 1 else 0.0
                    return mu, std, ci

                c_mu, c_std, c_ci = _mean_std_ci(cost_arr)
                r_mu, r_std, r_ci = _mean_std_ci(ret_arr)
                v_mu, v_std, v_ci = _mean_std_ci(vio_arr)
                ti_mu, ti_std, ti_ci = _mean_std_ci(traj_idle_arr)
                tb_mu, tb_std, tb_ci = _mean_std_ci(traj_boundary_arr)
                tu_mu, tu_std, tu_ci = _mean_std_ci(traj_user_nn_arr)
                tc_mu, tc_std, tc_ci = _mean_std_ci(traj_centroid_arr)
                ts_mu, ts_std, ts_ci = _mean_std_ci(traj_switchback_arr)
                d_mu, d_std, d_ci = _mean_std_ci(delay_arr)
                e_mu, e_std, e_ci = _mean_std_ci(energy_arr)

                summary_w.writerow(
                    [
                        int(ckpt_step),
                        int(eval_train_ep),
                        str(vname),
                        int(n_seed),
                        int(episodes_per_seed),
                        int(n_seed * episodes_per_seed),
                        1 if bool(deterministic) else 0,
                        float(c_mu),
                        float(c_std),
                        float(c_ci),
                        float(r_mu),
                        float(r_std),
                        float(r_ci),
                        float(v_mu),
                        float(v_std),
                        float(v_ci),
                        float(ti_mu),
                        float(ti_std),
                        float(ti_ci),
                        float(tb_mu),
                        float(tb_std),
                        float(tb_ci),
                        float(tu_mu),
                        float(tu_std),
                        float(tu_ci),
                        float(tc_mu),
                        float(tc_std),
                        float(tc_ci),
                        float(ts_mu),
                        float(ts_std),
                        float(ts_ci),
                        float(d_mu),
                        float(d_std),
                        float(d_ci),
                        float(e_mu),
                        float(e_std),
                        float(e_ci),
                        str(paper_cost_agg),
                        str(rd),
                        float(train_load),
                        float(load0),
                        str(contract_version),
                    ]
                )
                try:
                    summary_f.flush()
                except Exception as exc:
                    _warn_once(
                        "fixed_eval_summary_flush",
                        f"fixed_eval CSV flush {type(exc).__name__}: {exc}",
                    )

                if vname in variant_curves:
                    variant_curves[vname]["steps"].append(int(ckpt_step))
                    variant_curves[vname]["episodes"].append(int(eval_train_ep))
                    variant_curves[vname]["cost_mean"].append(float(c_mu))
                    variant_curves[vname]["cost_ci"].append(float(c_ci))
                    variant_curves[vname]["vio_mean"].append(float(v_mu))
                    variant_curves[vname]["vio_ci"].append(float(v_ci))
                done_pairs.add((str(vname), int(ckpt_step)))
    finally:
        try:
            curve_f.close()
        except Exception as exc:
            _warn_once(
                "fixed_eval_curve_close",
                f"fixed_eval CSV close {type(exc).__name__}: {exc}",
            )
        try:
            summary_f.close()
        except Exception as exc:
            _warn_once(
                "fixed_eval_summary_close",
                f"fixed_eval CSV close {type(exc).__name__}: {exc}",
            )

    try:
        for vname in fixed_variants_norm:
            if vname not in variant_curves:
                continue
            arr_step = np.asarray(variant_curves[vname]["steps"], dtype=np.int64).reshape(-1)
            if arr_step.size <= 0:
                continue
            ord_idx = np.argsort(arr_step)
            variant_curves[vname]["steps"] = [variant_curves[vname]["steps"][i] for i in ord_idx.tolist()]
            variant_curves[vname]["episodes"] = [variant_curves[vname]["episodes"][i] for i in ord_idx.tolist()]
            variant_curves[vname]["cost_mean"] = [variant_curves[vname]["cost_mean"][i] for i in ord_idx.tolist()]
            variant_curves[vname]["cost_ci"] = [variant_curves[vname]["cost_ci"][i] for i in ord_idx.tolist()]
            variant_curves[vname]["vio_mean"] = [variant_curves[vname]["vio_mean"][i] for i in ord_idx.tolist()]
            variant_curves[vname]["vio_ci"] = [variant_curves[vname]["vio_ci"][i] for i in ord_idx.tolist()]

        quota_note = (
            f"variants={','.join(fixed_variants_norm)}, "
            f"seeds={len(seeds)}, eps/seed={episodes_per_seed}, total={len(seeds) * episodes_per_seed}, "
            f"deterministic={1 if deterministic else 0}, agg={paper_cost_agg}, "
            f"train_load={train_load:.2f}, eval_load={load0:.2f}"
        )
        _plot_fixed_eval_variant_overlay(
            fig_path=fig_dir / fig_name,
            curves_by_variant=variant_curves,
            variants_order=fixed_variants_norm,
            quota_note=quota_note,
            ema_span=48,
        )
        _publish_key_plot(
            train_run_dir=train_run_dir,
            src_path=fig_dir / fig_name,
            bucket_name=FIG_BUCKET_FIXED_EVAL,
            dst_name="FixedEval_PaperCost.png",
        )
    except Exception as e:
        print(f"[HB][fixed_eval][WARN] plot failed: {type(e).__name__}: {e}", flush=True)

    traj_summary_csv = eval_dir / "trajectory_metrics_summary.csv"
    try:
        with open(summary_csv, "r", encoding="utf-8", newline="") as f:
            rr = list(csv.DictReader(f))
        cols = [
            "ckpt_step",
            "eval_train_episode",
            "variant",
            "method",
            "load",
            "n_seeds",
            "episodes_per_seed",
            "traj_idle_ratio_mean",
            "traj_idle_ratio_std",
            "traj_idle_ratio_ci95",
            "traj_boundary_stick_frac_mean",
            "traj_boundary_stick_frac_std",
            "traj_boundary_stick_frac_ci95",
            "traj_user_nn_dist_norm_mean",
            "traj_user_nn_dist_norm_std",
            "traj_user_nn_dist_norm_ci95",
            "traj_centroid_gap_norm_mean",
            "traj_centroid_gap_norm_std",
            "traj_centroid_gap_norm_ci95",
            "traj_switchback_ratio_mean",
            "traj_switchback_ratio_std",
            "traj_switchback_ratio_ci95",
        ]
        with open(traj_summary_csv, "w", encoding="utf-8", newline="") as f:
            ww = csv.DictWriter(f, fieldnames=cols)
            ww.writeheader()
            for r in rr:
                ww.writerow(
                    {
                        "ckpt_step": int(float(r.get("ckpt_step", -1))),
                        "eval_train_episode": int(float(r.get("eval_train_episode", -1))),
                        "variant": str(r.get("variant", "full")).strip().lower() or "full",
                        "method": "ppo",
                        "load": float(r.get("eval_load_primary", load0)),
                        "n_seeds": int(float(r.get("n_seeds", len(seeds)))),
                        "episodes_per_seed": int(float(r.get("episodes_per_seed", episodes_per_seed))),
                        "traj_idle_ratio_mean": float(r.get("traj_idle_ratio_mean", "nan")),
                        "traj_idle_ratio_std": float(r.get("traj_idle_ratio_std", "nan")),
                        "traj_idle_ratio_ci95": float(r.get("traj_idle_ratio_ci95", "nan")),
                        "traj_boundary_stick_frac_mean": float(r.get("traj_boundary_stick_frac_mean", "nan")),
                        "traj_boundary_stick_frac_std": float(r.get("traj_boundary_stick_frac_std", "nan")),
                        "traj_boundary_stick_frac_ci95": float(r.get("traj_boundary_stick_frac_ci95", "nan")),
                        "traj_user_nn_dist_norm_mean": float(r.get("traj_user_nn_dist_norm_mean", "nan")),
                        "traj_user_nn_dist_norm_std": float(r.get("traj_user_nn_dist_norm_std", "nan")),
                        "traj_user_nn_dist_norm_ci95": float(r.get("traj_user_nn_dist_norm_ci95", "nan")),
                        "traj_centroid_gap_norm_mean": float(r.get("traj_centroid_gap_norm_mean", "nan")),
                        "traj_centroid_gap_norm_std": float(r.get("traj_centroid_gap_norm_std", "nan")),
                        "traj_centroid_gap_norm_ci95": float(r.get("traj_centroid_gap_norm_ci95", "nan")),
                        "traj_switchback_ratio_mean": float(r.get("traj_switchback_ratio_mean", "nan")),
                        "traj_switchback_ratio_std": float(r.get("traj_switchback_ratio_std", "nan")),
                        "traj_switchback_ratio_ci95": float(r.get("traj_switchback_ratio_ci95", "nan")),
                    }
                )
    except Exception as e:
        print(f"[HB][fixed_eval][WARN] trajectory summary write failed: {type(e).__name__}: {e}", flush=True)

    print(
        f"[HB][fixed_eval] DONE variants={fixed_variants_norm} curve={curve_csv} summary={summary_csv} fig_dir={fig_dir}",
        flush=True,
    )

def main(argv: Optional[List[str]] = None) -> None:
    _apply_ieee_tcom_style()

    # ---------- CLI for unified and fixed-eval evaluation ----------
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--run_dir", type=str, default=None, help="Training run directory.")
    parser.add_argument("--fixed_eval", type=int, default=0, help="Run fixed-eval convergence evaluation when set to 1.")
    parser.add_argument("--seed_set", type=int, default=None, help="Number of fixed-eval seeds.")
    parser.add_argument("--episodes_per_seed", type=int, default=None, help="Fixed-eval episodes per seed.")
    parser.add_argument("--deterministic", type=int, default=None, help="Use deterministic policy actions when set to 1.")
    parser.add_argument("--stochastic", type=int, default=0, help="Use stochastic policy actions when set to 1.")
    parser.add_argument(
        "--fixed_eval_ckpt",
        type=str,
        default="",
        help="Checkpoint path for fixed-eval. Defaults to the run's best checkpoint.",
    )
    parser.add_argument(
        "--fixed_eval_require_final_ckpt",
        type=int,
        choices=[0, 1],
        default=0,
        help="Require the final checkpoint for fixed-eval when set to 1; otherwise use best.",
    )
    parser.add_argument(
        "--fixed_eval_variants",
        type=str,
        default=None,
        help="fixed-eval variants, comma-separated (default: full,no_ris,no_comp), e.g. full,no_ris,no_comp",
    )
    parser.add_argument("--eval_ts", type=str, default=None, help="Evaluation timestamp/name used for output directories.")
    parser.add_argument(
        "--eval_loads",
        type=str,
        default="",
        help="Comma-separated load_scale values. Main evaluation should use 0.1 through 1.0.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Explicit PPO checkpoint path.",
    )
    parser.add_argument(
        "--require_explicit_ckpt",
        type=int,
        choices=[0, 1],
        default=None,
        help="Require an explicit checkpoint path when set to 1.",
    )
    parser.add_argument(
        "--strict_ten_loads",
        type=int,
        choices=[0, 1],
        default=None,
        help="Require the ten-load protocol from 0.1 to 1.0 when set to 1.",
    )
    args, _unknown = parser.parse_known_args(argv)

    
    methods_env = os.environ.get("EVAL_METHODS", "").strip()
    if methods_env:
        methods = [m.strip().lower() for m in methods_env.split(",") if m.strip()]
    else:
        
        methods = [
            "ppo",
            "balanced",
            "greedy_delay",
            "greedy_energy",
            "myopic_optimization",
            "always_comp",
            "never_comp",
            "ddpg",
            "sac",
            "td3",
        ]

    
    seen = set()
    methods = [m for m in methods if (m not in seen and not seen.add(m))]
    requested_methods = list(methods)
    unknown_methods = [m for m in methods if m not in _ALLOWED_EVAL_METHODS]
    if unknown_methods:
        allowed_text = ",".join(_ALLOWED_EVAL_METHODS)
        raise ValueError(
            f"[HB][eval][FATAL] Unknown methods in EVAL_METHODS: {unknown_methods}. "
            f"Allowed methods: {allowed_text}"
        )
    fail_on_method_drop = _parse_bool(os.environ.get("EVAL_FAIL_ON_METHOD_DROP", "1"), True)
    require_clean_output = _parse_bool(os.environ.get("EVAL_REQUIRE_CLEAN_OUTPUT", "1"), True)
    force_overwrite = _parse_bool(os.environ.get("EVAL_FORCE_OVERWRITE", "0"), False)
    resume_strict_guard = _parse_bool(os.environ.get("EVAL_RESUME_STRICT_GUARD", "1"), True)
    fail_on_incomplete_raw = _parse_bool(os.environ.get("EVAL_FAIL_ON_INCOMPLETE_RAW", "1"), True)

    
    if args.run_dir:
        p = Path(str(args.run_dir))
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        train_run_dir = p
    else:
        train_run_dir = _select_run_dir()

    
    ppo_display = _infer_ppo_display_label_from_run_dir(train_run_dir)
    _TCOM_METHOD_STYLE["ppo"] = {**_TCOM_METHOD_STYLE.get("ppo", {}), "label": ppo_display}

    env_yaml, train_yaml = _select_cfg_paths(train_run_dir)
    cfg_env = _load_yaml(env_yaml)
    cfg_train = _load_yaml(train_yaml)
    
    train_block = cfg_train.get("train", {}) if isinstance(cfg_train, dict) else {}
    ppo_train_seed = None
    ppo_train_total_steps = None
    try:
        if isinstance(cfg_train, dict) and ("seed" in cfg_train):
            ppo_train_seed = int(cfg_train.get("seed"))
    except Exception:
        ppo_train_seed = None
    try:
        if isinstance(train_block, dict) and ("total_steps" in train_block):
            ppo_train_total_steps = int(train_block.get("total_steps"))
    except Exception:
        ppo_train_total_steps = None
    allow_crossrun_baseline = _parse_bool(os.environ.get("EVAL_ALLOW_CROSSRUN_BASELINE", "0"), False)

    
    method_label_map = {str(m): _method_display_label(str(m)) for m in methods}

    
    eval_ts = (str(args.eval_ts).strip() if args.eval_ts else "") or (os.environ.get("EVAL_TS", "").strip()) or now_ts()
    device = os.environ.get("EVAL_DEVICE", str(cfg_train.get("device", "cpu"))).strip()

    seed0 = _parse_int(os.environ.get("EVAL_SEED0", "42"), 42)
    n_seeds = _parse_int(os.environ.get("EVAL_NSEEDS", "20"), 20)
    
    loads_raw = str(args.eval_loads).strip() if args.eval_loads is not None else ""
    if not loads_raw:
        loads_raw = os.environ.get("EVAL_LOADS", "").strip()
    loads = _parse_loads(loads_raw)
    strict_ten_loads = (
        int(args.strict_ten_loads)
        if args.strict_ten_loads is not None
        else _parse_int(os.environ.get("EVAL_STRICT_TEN_LOADS", "1"), 1)
    )
    if (int(args.fixed_eval) != 1) and bool(strict_ten_loads):
        if not _is_ten_load_protocol(loads):
            raise ValueError(f"[HB][eval][FATAL] strict_ten_loads=1 0.1~1.0{loads}")

    ckpt_path_raw = str(args.ckpt_path).strip()
    if not ckpt_path_raw:
        ckpt_path_raw = os.environ.get("CKPT_PATH", "").strip()
    explicit_ppo_ckpt: Optional[Path] = None
    if ckpt_path_raw:
        p = Path(ckpt_path_raw)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"[HB][eval][FATAL] ckpt_path not found: {p}")
        explicit_ppo_ckpt = p

    require_explicit_ckpt = (
        int(args.require_explicit_ckpt)
        if args.require_explicit_ckpt is not None
        else _parse_int(os.environ.get("EVAL_REQUIRE_EXPLICIT_CKPT", "1"), 1)
    )
    if ("ppo" in methods) and bool(require_explicit_ckpt) and (explicit_ppo_ckpt is None):
        raise RuntimeError("[HB][eval][FATAL] require_explicit_ckpt=1 --ckpt_path/CKPT_PATH")

    ep_per = _parse_int(os.environ.get("EVAL_EP_PER", "5"), 5)
    ep_per_progressive = _parse_bool(os.environ.get("EVAL_EP_PER_PROGRESSIVE", "1"), True)
    ep_per_by_load = _build_ep_per_by_load(loads, ep_per, progressive=bool(ep_per_progressive))
    if not ep_per_by_load:
        ep_per_by_load = {float(ld): int(max(1, ep_per)) for ld in loads}
    deterministic = _parse_bool(os.environ.get("EVAL_DETERMINISTIC", "True"), True)
    hb_every = _parse_int(os.environ.get("EVAL_HB_EVERY", "200"), 200)
    per_load_eps = [int(ep_per_by_load.get(_load_key(float(ld)), max(1, int(ep_per)))) for ld in loads]
    ep_per_min = int(min(per_load_eps)) if per_load_eps else int(max(1, int(ep_per)))
    ep_per_max = int(max(per_load_eps)) if per_load_eps else int(max(1, int(ep_per)))
    eval_quota = int(max(int(n_seeds), 1) * max(int(ep_per_min), 1))
    if eval_quota < 6:
        _warn_once(
            "low_stat_quota",
            f"n_seeds*min_ep_per={eval_quota}"
            "/>=155x3",
        )
    elif eval_quota < 15:
        _warn_once(
            "low_stat_quota_soft",
            f"n_seeds*min_ep_per={eval_quota}"
            ">=15",
        )
    
    
    
    
    ppo_obs_norm_raw = os.environ.get("EVAL_PPO_USE_OBS_NORM", "").strip()
    if ppo_obs_norm_raw:
        ppo_use_obs_norm = _parse_bool(ppo_obs_norm_raw, True)
    else:
        has_ppo = ("ppo" in methods)
        has_3drl_mix = any(m in methods for m in ("ddpg", "sac", "td3", "a2c"))
        ppo_use_obs_norm = True
        if has_ppo and has_3drl_mix:
            print(
                "[HB][eval][fairness] EVAL_PPO_USE_OBS_NORM PPO obs_norm=1"
                "RAW EVAL_PPO_USE_OBS_NORM=0",
                flush=True,
            )
    
    set_global_seed(seed0, deterministic=bool(deterministic))

    plot_ci = _parse_bool(os.environ.get("EVAL_PLOT_CI", "0"), False)
    plot_logy = _parse_bool(os.environ.get("EVAL_PLOT_LOGY", "1"), True)
    plot_zoom = _parse_bool(os.environ.get("EVAL_PLOT_ZOOM", "1"), True)
    try:
        plot_zoom_q = float(os.environ.get("EVAL_PLOT_ZOOM_Q", "0.85"))
    except Exception:
        plot_zoom_q = 0.85
    plot_zoom_q = float(np.clip(plot_zoom_q, 0.50, 0.99))
    resume = _parse_bool(os.environ.get("EVAL_RESUME", "1"), True)

    
    eval_dir = train_run_dir / "evals" / eval_ts
    fig_dir = train_run_dir / "figs" / f"{eval_ts}{FIG_EVAL_TS_SUFFIX}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_env_cfg = build_env_cfg(cfg_env)
    variants = make_variants(base_env_cfg)
    
    x_axis_cpu_mcycles = [_required_cpu_mcycles_from_load(float(ld), base_env_cfg) for ld in loads]
    
    x_axis_task_mb = [_required_task_size_mb_from_load(float(ld), base_env_cfg) for ld in loads]

    
    if int(args.fixed_eval) == 1:
        seed_set_size = int(args.seed_set) if args.seed_set is not None else 10
        ep_per_seed = int(args.episodes_per_seed) if args.episodes_per_seed is not None else 10
        fixed_variants_raw = (
            str(args.fixed_eval_variants).strip()
            if args.fixed_eval_variants is not None
            else os.environ.get("FIXED_EVAL_VARIANTS", "").strip()
        )
        if fixed_variants_raw:
            fixed_variants = [v.strip().lower() for v in fixed_variants_raw.split(",") if v.strip()]
            fixed_variants = list(dict.fromkeys(fixed_variants))
        else:
            
            fixed_variants = ["full", "no_ris", "no_comp"]
        
        det = True if args.deterministic is None else bool(int(args.deterministic) == 1)
        is_stoch = bool(int(args.stochastic) == 1) or (not bool(det))
        det = False if is_stoch else True
        fixed_eval_ckpt_raw = str(args.fixed_eval_ckpt).strip()
        if not fixed_eval_ckpt_raw:
            fixed_eval_ckpt_raw = os.environ.get("CKPT_PATH", "").strip()
        fixed_eval_ckpt_path: Optional[Path] = None
        if fixed_eval_ckpt_raw:
            p_ck = Path(fixed_eval_ckpt_raw)
            if not p_ck.is_absolute():
                p_ck = (REPO_ROOT / p_ck).resolve()
            fixed_eval_ckpt_path = p_ck
        
        if fixed_eval_ckpt_path is None:
            raise RuntimeError(
                "[HB][fixed_eval][FATAL] fixed-eval requires explicit checkpoint via "
                "--fixed_eval_ckpt or CKPT_PATH."
            )
        if fixed_eval_ckpt_path is not None:
            ck_name = str(fixed_eval_ckpt_path.name).strip().lower()
            require_final_ckpt = bool(int(args.fixed_eval_require_final_ckpt))
            if require_final_ckpt:
                if "final" not in ck_name:
                    raise RuntimeError(
                        "[HB][fixed_eval][FATAL] fixed-eval  final checkpoint"
                        f"={fixed_eval_ckpt_path}"
                    )
            else:
                if "best" not in ck_name:
                    raise RuntimeError(
                        "[HB][fixed_eval][FATAL] fixed-eval  best checkpoint"
                        " final --fixed_eval_require_final_ckpt 1"
                        f"={fixed_eval_ckpt_path}"
                    )

        _run_fixed_eval_curve(
            train_run_dir=train_run_dir,
            base_env_cfg=base_env_cfg,
            cfg_train=cfg_train,
            device=device,
            eval_ts=eval_ts,
            seed_set_size=seed_set_size,
            episodes_per_seed=ep_per_seed,
            deterministic=det,
            is_stochastic=is_stoch,
            resume=bool(resume),
            fixed_variants=fixed_variants,
            fixed_eval_ckpt_path=fixed_eval_ckpt_path,
        )
        try:
            suffix = "_stoch" if bool(is_stoch) else ""
            fixed_summary_csv = train_run_dir / "evals" / eval_ts / f"fixed_eval_summary{suffix}.csv"
            fixed_fig_path = train_run_dir / "figs" / f"{eval_ts}{FIG_EVAL_TS_SUFFIX}" / f"FixedEval_PaperCost{suffix}.png"
            phaseg_summary_path = upsert_phaseg_eval_summary(
                run_dir=train_run_dir,
                role="fixed",
                eval_ts=eval_ts,
                ckpt_path=(str(fixed_eval_ckpt_path) if fixed_eval_ckpt_path is not None else os.environ.get("CKPT_PATH", "").strip()),
                artifacts={
                    "fixed_eval_summary_csv": str(fixed_summary_csv),
                    "fixed_eval_fig": str(fixed_fig_path),
                },
            )
            if phaseg_summary_path is not None:
                print(f"[HB][phaseg] updated summary: {phaseg_summary_path}", flush=True)
        except Exception as e:
            msg = str(e)
            if "[phaseg][FATAL]" in msg:
                raise
            print(f"[HB][phaseg][WARN] failed to update summary after fixed-eval: {type(e).__name__}: {e}", flush=True)
        return

    
    ppo_agent = None
    ppo_ckpt = None
    obs_dim = None
    act_dim = None
    ppo_hidden = None

    if "ppo" in methods:
        if explicit_ppo_ckpt is not None:
            ppo_agent, ppo_ckpt, obs_dim, act_dim, ppo_hidden = _load_ppo_agent_from_ckpt(
                train_run_dir=train_run_dir,
                ckpt_path=explicit_ppo_ckpt,
                base_env_cfg=base_env_cfg,
                seed0=seed0,
                cfg_train=cfg_train,
                device=device,
            )
        else:
            ppo_agent, ppo_ckpt, obs_dim, act_dim, ppo_hidden = _load_ppo_agent(
                train_run_dir, base_env_cfg, seed0, cfg_train, device
            )
        
        global _EVAL_OBS_NORMALIZER
        try:
            clip = float((cfg_train.get("train", {}) or {}).get("obs_clip", 10.0))
        except Exception:
            clip = 10.0
        if ppo_use_obs_norm:
            _EVAL_OBS_NORMALIZER = _maybe_load_obs_normalizer(train_run_dir, ppo_ckpt, obs_dim, clip=clip)
        else:
            _EVAL_OBS_NORMALIZER = None
            print("[HB][eval] EVAL_PPO_USE_OBS_NORM=0PPORAW obs_flat()", flush=True)
        if ppo_use_obs_norm and _EVAL_OBS_NORMALIZER is None:
            print("[HB][eval][WARN] obs_normalizer not found; eval uses RAW obs_flat()", flush=True)
    else:
        
        probe_env = CompRISEnvJoint(base_env_cfg, seed=seed0, reward_mode=str(cfg_train.get("reward_mode", "p1")))
        obs_dim = int(probe_env.obs_flat().shape[0])
        act_dim = int(probe_env.action_dim)

    assert obs_dim is not None and act_dim is not None

    
    drl_agents: Dict[str, Any] = {}
    baseline_ckpt_quality: Dict[str, Dict[str, Any]] = {}
    require_3drl_final_ckpt = _parse_bool(os.environ.get("EVAL_3DRL_REQUIRE_FINAL_CKPT", "1"), True)
    for algo in ("ddpg", "sac", "td3", "a2c"):
        if algo not in methods:
            continue
        envvar = f"{algo.upper()}_CKPT"
        ckpt_str = os.environ.get(envvar, "").strip()
        if ckpt_str:
            ckpt_path = Path(ckpt_str)
        else:
            strict_fair_auto = ("ppo" in methods) and (not bool(allow_crossrun_baseline))
            auto = _auto_find_3drl_ckpt(
                algo,
                target_seed=ppo_train_seed,
                target_total_steps=ppo_train_total_steps,
                allow_crossrun_fallback=(not strict_fair_auto),
            )
            ckpt_path = auto if auto is not None else (REPO_ROOT / "runs" / "paper" / f"{algo}_latest" / CKPT_DIR_NAME / "agent_final.pt")
            if auto is not None:
                print(f"[HB][eval] auto-select {algo.upper()} ckpt: {ckpt_path}", flush=True)
                _auto_name_l = str(auto.name).lower()
                _auto_final_like = (_auto_name_l == "agent_final.pt") or _auto_name_l.startswith("agent_finally_step")
                if not bool(_auto_final_like):
                    _warn_once(
                        f"auto_ckpt_not_final::{algo}",
                        f"{algo.upper()}  {auto.name} {algo.upper()}_CKPT  "
                        "agent_final.pt / agent_finally_step*.pt ",
                    )
            elif strict_fair_auto:
                msg = (
                    f"no fair-matched {algo.upper()} ckpt found "
                    f"(need seed={ppo_train_seed}, total_steps={ppo_train_total_steps})"
                )
                if bool(fail_on_method_drop):
                    raise RuntimeError(
                        "[HB][eval][FATAL] " + msg +
                        "run EVAL_ALLOW_CROSSRUN_BASELINE=1"
                        " EVAL_FAIL_ON_METHOD_DROP=0"
                    )
                print(
                    f"[HB][eval][WARN] {msg}; remove from methods. "
                    "run EVAL_ALLOW_CROSSRUN_BASELINE=1",
                    flush=True,
                )
                methods = [m for m in methods if m != algo]
                continue
        if not ckpt_path.is_absolute():
            ckpt_path = (REPO_ROOT / ckpt_path).resolve()
        if not ckpt_path.exists():
            msg = f"{algo.upper()} checkpoint not found: {ckpt_path}"
            if bool(fail_on_method_drop):
                raise RuntimeError(
                    "[HB][eval][FATAL] " + msg +
                    "ckpt EVAL_FAIL_ON_METHOD_DROP=0"
                )
            print(f"[HB][eval][WARN] {msg} -> remove from methods", flush=True)
            methods = [m for m in methods if m != algo]
            continue
        _ckpt_name_l = str(ckpt_path.name).lower()
        _is_final_like = (_ckpt_name_l == "agent_final.pt") or _ckpt_name_l.startswith("agent_finally_step")
        if bool(require_3drl_final_ckpt) and (not bool(_is_final_like)):
            msg = (
                f"{algo.upper()} checkpoint  final{ckpt_path.name}"
                " agent_final.pt / agent_finally_step*.pt"
            )
            if bool(fail_on_method_drop):
                raise RuntimeError(
                    "[HB][eval][FATAL] " + msg +
                    "checkpoint EVAL_FAIL_ON_METHOD_DROP=0"
                )
            print(f"[HB][eval][WARN] {msg} -> remove from methods", flush=True)
            methods = [m for m in methods if m != algo]
            continue
        quality = _extract_3drl_ckpt_quality(ckpt_path)
        baseline_ckpt_quality[str(algo)] = quality
        ts = quality.get("total_steps", None)
        ws = quality.get("warmup_steps", None)
        if bool(quality.get("likely_untrained", False)):
            _warn_once(
                f"baseline_ckpt_untrained::{algo}",
                f"{algo.upper()} ckptwarmuptotal_steps={ts}, warmup_steps={ws}"
                "",
            )
        elif (ts is not None) and (int(ts) < 50000):
            _warn_once(
                f"baseline_ckpt_lowbudget::{algo}",
                f"{algo.upper()} total_steps={int(ts)}"
                ">=50kPPOcheckpoint",
            )
        try:
            drl_agents[algo] = _load_drl_agent(algo, ckpt_path, obs_dim, act_dim, device)
        except Exception as e:
            msg = f"Failed to load {algo.upper()} from {ckpt_path}: {type(e).__name__}: {e}"
            if bool(fail_on_method_drop):
                raise RuntimeError(
                    "[HB][eval][FATAL] " + msg +
                    "checkpoint"
                ) from e
            print(f"[HB][eval][WARN] {msg} -> remove from methods", flush=True)
            methods = [m for m in methods if m != algo]
    method_label_map = {str(m): _method_display_label(str(m)) for m in methods}
    dropped_methods = [m for m in requested_methods if m not in methods]
    if dropped_methods:
        msg = f"requested methods dropped during preflight: {dropped_methods}"
        if bool(fail_on_method_drop):
            raise RuntimeError("[HB][eval][FATAL] " + msg)
        _warn_once("method_drop", msg)

    # ---------- print startup HB ----------
    print(f"[HB][eval] t={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} train_run_dir={train_run_dir}", flush=True)
    print(f"[HB][eval] env_yaml={env_yaml}", flush=True)
    print(f"[HB][eval] train_yaml={train_yaml}", flush=True)
    print(f"[HB][eval] eval_dir={eval_dir}", flush=True)
    print(f"[HB][eval] fig_dir={fig_dir}", flush=True)
    print(f"[HB][eval] plot_ci={int(plot_ci)} plot_logy={int(plot_logy)} plot_zoom={int(plot_zoom)} zoom_q={plot_zoom_q:.2f}", flush=True)
    print(f"[HB][eval] methods={methods}", flush=True)
    print(f"[HB][eval] method_label_map={method_label_map}", flush=True)
    print(f"[HB][eval] variants={list(variants.keys())}", flush=True)
    print(f"[HB][eval] obs_dim={obs_dim} act_dim={act_dim} device={device} deterministic={deterministic}", flush=True)
    print(f"[HB][eval] global_seed={seed0}", flush=True)
    print(f"[HB][eval] x_axis_cpu_mcycles={x_axis_cpu_mcycles}", flush=True)
    print(
        f"[HB][eval] baseline_auto_match: seed={ppo_train_seed} total_steps={ppo_train_total_steps} "
        f"allow_crossrun={int(bool(allow_crossrun_baseline))}",
        flush=True,
    )
    if ppo_ckpt is not None:
        print(f"[HB][eval] ppo_ckpt={ppo_ckpt} hidden={ppo_hidden}", flush=True)
    has_3drl = any(m in methods for m in ("ddpg", "sac", "td3", "a2c"))
    if ("ppo" in methods) and has_3drl:
        if ppo_use_obs_norm and (_EVAL_OBS_NORMALIZER is not None):
            print(
                "[HB][eval][WARN] PPOobs3DRLRAW obs_flat"
                "RAW EVAL_PPO_USE_OBS_NORM=0",
                flush=True,
            )
        else:
            print("[HB][eval] RAW obs_flatPPOobs_norm", flush=True)
    ep_plan_log = {f"{float(ld):.1f}": int(ep_per_by_load.get(_load_key(float(ld)), int(max(1, ep_per)))) for ld in loads}
    print(
        f"[HB][eval] sweep seed0={seed0} n_seeds={n_seeds} loads={loads} "
        f"ep_per_base={ep_per} ep_per_range=[{ep_per_min},{ep_per_max}] "
        f"ep_per_by_load={ep_plan_log} hb_every={hb_every}",
        flush=True,
    )

    # ---------- raw CSV streaming + resume ----------
    raw_csv = eval_dir / f"eval_raw_{eval_ts}.csv"
    cols = [
        "variant", "method", "load", "seed", "ep",
        "mean_rate", "mean_T_tx", "mean_T_off", "mean_energy", "paper_cost",
        "violation_count",
        "mean_S_size", "comp_active_ratio", "comp_gain_ratio",
        "traj_idle_ratio", "traj_boundary_stick_frac", "traj_user_nn_dist_norm",
        "traj_centroid_gap_norm", "traj_switchback_ratio",
    ]
    total_eval_episodes = int(
        sum(int(ep_per_by_load.get(_load_key(float(ld)), int(max(1, ep_per)))) for ld in loads)
    )
    total = int(len(variants) * len(methods) * int(max(1, n_seeds)) * max(1, total_eval_episodes))
    k = 0
    ran = 0
    skipped = 0
    t_start = time.perf_counter()
    last50 = deque(maxlen=50)
    plot_only = _parse_bool(os.environ.get("EVAL_PLOT_ONLY", "0"), False)
    
    
    
    
    if plot_only and bool(force_overwrite):
        raise RuntimeError("[HB][eval][FATAL] EVAL_PLOT_ONLY=1  EVAL_FORCE_OVERWRITE=1 ")
    if bool(force_overwrite):
        print(f"[HB][eval] EVAL_FORCE_OVERWRITE=1 -> clear eval_dir/fig_dir for eval_ts={eval_ts}", flush=True)
        _clear_eval_outputs(eval_dir, fig_dir)
        resume = False
    elif (not plot_only) and (not bool(resume)) and bool(require_clean_output):
        has_eval_files = eval_dir.exists() and any(eval_dir.iterdir())
        has_fig_files = fig_dir.exists() and any(fig_dir.iterdir())
        if has_eval_files or has_fig_files:
            raise RuntimeError(
                "[HB][eval][FATAL] non-resume "
                f" eval_dir={eval_dir} fig_dir={fig_dir}"
                " EVAL_FORCE_OVERWRITE=1"
            )
    if (not plot_only) and bool(resume) and bool(resume_strict_guard) and raw_csv.exists() and raw_csv.stat().st_size > 0:
        scope = _scan_raw_scope(raw_csv)
        req_mset = set(str(m) for m in methods)
        req_vset = set(str(v) for v in variants.keys())
        req_lset = set(f"{float(ld):.6f}" for ld in loads)
        bad_methods = sorted(list(set(scope["methods"]) - req_mset))
        bad_variants = sorted(list(set(scope["variants"]) - req_vset))
        bad_loads = sorted(list(set(scope["loads"]) - req_lset))
        if bad_methods or bad_variants or bad_loads:
            raise RuntimeError(
                "[HB][eval][FATAL] resume strict guard failedraw"
                f" bad_methods={bad_methods} bad_variants={bad_variants} bad_loads={bad_loads} raw={raw_csv}"
            )
        if int(scope.get("bad_rows", 0)) > 0:
            raise RuntimeError(
                "[HB][eval][FATAL] resume strict guard failedraw csv contains malformed rows. "
                f"bad_rows={int(scope.get('bad_rows', 0))} raw={raw_csv}"
            )
        print(
            f"[HB][eval] resume strict guard pass: rows={int(scope.get('rows', 0))} "
            f"methods={sorted(list(scope['methods']))} variants={sorted(list(scope['variants']))}",
            flush=True,
        )
    if plot_only:
        if (not raw_csv.exists()) or (raw_csv.stat().st_size <= 0):
            raise RuntimeError(
                f"[HB][eval][FATAL] EVAL_PLOT_ONLY=1  raw_csv{raw_csv}"
            )
        print(f"[HB][eval] EVAL_PLOT_ONLY=1 -> skip sweep, reuse raw={raw_csv}", flush=True)
    else:
        _ensure_csv_header(raw_csv, cols)
        done_keys = _load_done_keys(raw_csv) if resume else set()

        
        for vname, vcfg in variants.items():
            for method in methods:
                for load in loads:
                    ep_per_load = int(ep_per_by_load.get(_load_key(float(load)), int(max(1, ep_per))))
                    for si in range(n_seeds):
                        seed = seed0 + si
                        for ep in range(ep_per_load):
                            k += 1
                            key = f"{vname}|{method}|{float(load):.6f}|{int(seed)}|{int(ep)}"
                            if key in done_keys:
                                skipped += 1
                                if hb_every > 0 and (k % hb_every == 0 or k == 1 or k == total):
                                    pct = 100.0 * k / max(total, 1)
                                    wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    bar = _progress_bar(k, total)
                                    elapsed = time.perf_counter() - t_start
                                    avg_all = (elapsed / max(1, ran)) if ran > 0 else 0.0
                                    avg50 = (sum(last50) / max(1, len(last50))) if len(last50) else avg_all
                                    eta = (total - k) * avg_all if avg_all > 0 else 0.0
                                    print(
                                        f"[HB][eval] t={wall} {bar} {k}/{total} ({pct:.1f}%) "
                                        f"(resume) skip={skipped} ran={ran} "
                                        f"variant={vname} method={method} load={float(load):.3f} seed={seed} ep={ep} | "
                                        f"avg_all={avg_all:.3f}s avg50={avg50:.3f}s ETA={_fmt_hms(eta)} | raw={raw_csv}",
                                        flush=True
                                    )
                                continue

                            if hb_every > 0 and (k % hb_every == 0 or k == 1 or k == total):
                                pct = 100.0 * k / max(total, 1)
                                wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                bar = _progress_bar(k, total)
                                elapsed = time.perf_counter() - t_start
                                avg_all = (elapsed / max(1, ran)) if ran > 0 else 0.0
                                avg50 = (sum(last50) / max(1, len(last50))) if len(last50) else avg_all
                                eta = (total - k) * avg_all if avg_all > 0 else 0.0
                                print(
                                    f"[HB][eval] t={wall} {bar} {k}/{total} ({pct:.1f}%) "
                                    f"skip={skipped} ran={ran} "
                                    f"variant={vname} method={method} load={float(load):.3f} seed={seed} ep={ep} | "
                                    f"avg_all={avg_all:.3f}s avg50={avg50:.3f}s ETA={_fmt_hms(eta)} | raw={raw_csv}",
                                    flush=True
                                )

                            
                            
                            
                            ep_seed = int(seed + ep * 1000003)
                            env = CompRISEnvJoint(vcfg, seed=ep_seed, reward_mode=str(cfg_train.get("reward_mode", "p1")))
                            
                            try:
                                env.set_load_scale(float(load))  # type: ignore
                            except Exception:
                                try:
                                    env.cfg.load_scale = float(load)  # type: ignore
                                except Exception as exc:
                                    raise RuntimeError(
                                        f"[HB][eval][FATAL] load_scalevariant={vname}, method={method}, "
                                        f"load={float(load):.4f}{type(exc).__name__}: {exc}"
                                    ) from exc

                            if method == "ppo":
                                use_agent = ppo_agent
                            elif method in ("ddpg", "sac", "td3", "a2c"):
                                use_agent = drl_agents.get(method, None)
                            else:
                                use_agent = None

                            
                            action_seed = _stable_eval_seed((eval_ts, vname, method, float(load), int(seed), int(ep)))
                            set_global_seed(action_seed, deterministic=bool(deterministic))

                            t0 = time.perf_counter()
                            info = run_one_episode(env, method, use_agent, deterministic)
                            dt = time.perf_counter() - t0
                            last50.append(dt)

                            row = {
                                "variant": vname,
                                "method": method,
                                "load": float(load),
                                "seed": int(seed),
                                "ep": int(ep),

                                "mean_rate": _to_float(info.get("mean_rate", np.nan)),
                                "mean_T_tx": _to_float(info.get("mean_T_tx", np.nan)),
                                "mean_T_off": _to_float(info.get("mean_T_off", np.nan)),
                                "mean_energy": _to_float(info.get("mean_energy", np.nan)),
                                "paper_cost": _pick_first_finite(info, ("paper_cost", "paper_cost_mean"), default=float("nan")),
                                "violation_count": _pick_first_finite(
                                    info, ("violation_count", "violations", "violation_count_mean"), default=float("nan")
                                ),

                                "mean_S_size": _to_float(info.get("mean_S_size", np.nan)),
                                "comp_active_ratio": _to_float(info.get("comp_active_ratio", np.nan)),
                                "comp_gain_ratio": _to_float(info.get("comp_gain_ratio", np.nan)),
                                "traj_idle_ratio": _to_float(info.get("traj_idle_ratio", np.nan)),
                                "traj_boundary_stick_frac": _to_float(info.get("traj_boundary_stick_frac", np.nan)),
                                "traj_user_nn_dist_norm": _to_float(info.get("traj_user_nn_dist_norm", np.nan)),
                                "traj_centroid_gap_norm": _to_float(info.get("traj_centroid_gap_norm", np.nan)),
                                "traj_switchback_ratio": _to_float(info.get("traj_switchback_ratio", np.nan)),
                            }

                            _append_csv_row(raw_csv, cols, row)
                            done_keys.add(key)
                            ran += 1

    print(f"[HB][eval] sweep done. ran={ran} skipped={skipped} raw={raw_csv}", flush=True)

    # ---------- reload raw rows from CSV (robust resume) ----------
    raw_rows: List[Dict[str, Any]] = []
    raw_keys_seen: set[str] = set()
    duplicate_key_count = 0
    with open(raw_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for rr in reader:
            variant = str(rr.get("variant", "")).strip()
            method = str(rr.get("method", "")).strip()
            load = _to_float(rr.get("load", float("nan")))
            seed_f = _to_float(rr.get("seed", float("nan")))
            ep_f = _to_float(rr.get("ep", float("nan")))
            if (not variant) or (not method):
                continue
            if (not np.isfinite(load)) or (not np.isfinite(seed_f)) or (not np.isfinite(ep_f)):
                continue
            k_raw = f"{variant}|{method}|{float(load):.6f}|{int(seed_f)}|{int(ep_f)}"
            if k_raw in raw_keys_seen:
                duplicate_key_count += 1
                continue
            raw_keys_seen.add(k_raw)
            raw_rows.append({
                "variant": variant,
                "method": method,
                "load": float(load),
                "seed": int(seed_f),
                "ep": int(ep_f),

                "mean_rate": _to_float(rr.get("mean_rate", float("nan"))),
                "mean_T_tx": _to_float(rr.get("mean_T_tx", float("nan"))),
                "mean_T_off": _to_float(rr.get("mean_T_off", float("nan"))),
                "mean_energy": _to_float(rr.get("mean_energy", float("nan"))),
                "paper_cost": _to_float(rr.get("paper_cost", float("nan"))),
                "violation_count": _pick_first_finite(
                    rr, ("violation_count", "violations", "violation_count_mean"), default=float("nan")
                ),

                "mean_S_size": _to_float(rr.get("mean_S_size", float("nan"))),
                "comp_active_ratio": _to_float(rr.get("comp_active_ratio", float("nan"))),
                "comp_gain_ratio": _to_float(rr.get("comp_gain_ratio", float("nan"))),
                "traj_idle_ratio": _to_float(rr.get("traj_idle_ratio", float("nan"))),
                "traj_boundary_stick_frac": _to_float(rr.get("traj_boundary_stick_frac", float("nan"))),
                "traj_user_nn_dist_norm": _to_float(rr.get("traj_user_nn_dist_norm", float("nan"))),
                "traj_centroid_gap_norm": _to_float(rr.get("traj_centroid_gap_norm", float("nan"))),
                "traj_switchback_ratio": _to_float(rr.get("traj_switchback_ratio", float("nan"))),
            })

    if len(raw_rows) == 0:
        raise RuntimeError(f"[HB][eval][FATAL] raw_rows empty after reload. raw_csv={raw_csv}")
    if duplicate_key_count > 0:
        msg = f"raw csv contains duplicate keys: {duplicate_key_count} (deduped on load) raw={raw_csv}"
        if bool(fail_on_incomplete_raw):
            raise RuntimeError("[HB][eval][FATAL] " + msg)
        _warn_once("raw_duplicate_keys", msg)

    
    exp_cell: Dict[Tuple[str, str, float], int] = {}
    for vname in variants.keys():
        for method in methods:
            for ld in loads:
                lk = _load_key(float(ld))
                
                exp_ep = ep_per_by_load.get(lk, ep_per_by_load.get(str(lk), int(max(1, ep_per))))
                exp_cell[(str(vname), str(method), float(lk))] = int(max(1, int(n_seeds))) * int(exp_ep)
    act_cell: Dict[Tuple[str, str, float], int] = {}
    for rr in raw_rows:
        key = (str(rr["variant"]), str(rr["method"]), _load_key(float(rr["load"])))
        if key in exp_cell:
            act_cell[key] = int(act_cell.get(key, 0)) + 1
    miss_cells: List[str] = []
    for key, exp_n in exp_cell.items():
        act_n = int(act_cell.get(key, 0))
        if act_n != int(exp_n):
            vname, method, lk = key
            miss_cells.append(f"{vname}|{method}|load={lk}|act={act_n}|exp={int(exp_n)}")
    if miss_cells:
        preview = miss_cells[:20]
        msg = (
            f"raw completeness check failed: {len(miss_cells)} cells mismatch. "
            f"examples={preview}"
        )
        if bool(fail_on_incomplete_raw):
            raise RuntimeError("[HB][eval][FATAL] " + msg)
        _warn_once("raw_incomplete", msg)

    # ---------- aggregate summary ----------
    metrics = ["paper_cost", "mean_T_off", "mean_energy", "violation_count"]
    series: Dict[str, Dict[str, Dict[str, List[float]]]] = {}  # series_name -> metric -> {"x":[],"y":[],"e":[]}

    grp: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for r in raw_rows:
        key = (str(r["variant"]), str(r["method"]), _load_key(r["load"]))
        grp.setdefault(key, []).append(r)

    for vname in variants.keys():
        for method in methods:
            sname = f"{vname}|{method}"
            series[sname] = {m: {"x": [], "y": [], "e": []} for m in metrics}
            for load in loads:
                rows = grp.get((vname, method, _load_key(load)), [])
                for m in metrics:
                    x = np.asarray([rr.get(m, np.nan) for rr in rows], dtype=np.float64)
                    mu = float(np.nanmean(x)) if x.size else float("nan")
                    ci = _ci95(x) if x.size else 0.0
                    series[sname][m]["x"].append(float(load))
                    series[sname][m]["y"].append(mu)
                    series[sname][m]["e"].append(ci)
    _warn_curve_anomalies(series, methods)
    shape_target_variant = str(os.environ.get("EVAL_SHAPE_TARGET_VARIANT", "full")).strip().lower() or "full"
    try:
        shape_tol_rel = float(os.environ.get("EVAL_SHAPE_MONO_TOL_REL", "0.02"))
    except Exception:
        shape_tol_rel = 0.02
    try:
        shape_spearman_min = float(os.environ.get("EVAL_SHAPE_SPEARMAN_MIN", "0.60"))
    except Exception:
        shape_spearman_min = 0.60
    try:
        shape_max_violation_ratio = float(os.environ.get("EVAL_SHAPE_MAX_VIOLATION_RATIO", "0.25"))
    except Exception:
        shape_max_violation_ratio = 0.25
    shape_tol_rel = float(np.clip(abs(shape_tol_rel), 1e-6, 0.50))
    shape_spearman_min = float(np.clip(shape_spearman_min, -1.0, 1.0))
    shape_max_violation_ratio = float(np.clip(shape_max_violation_ratio, 0.0, 1.0))

    shape_audit = _build_c_axis_shape_audit(
        series=series,
        methods=methods,
        loads=loads,
        x_axis_values=x_axis_cpu_mcycles,
        target_variant=shape_target_variant,
        metrics=("paper_cost", "mean_T_off", "mean_energy"),
        tol_rel=shape_tol_rel,
        spearman_min=shape_spearman_min,
        max_violation_ratio=shape_max_violation_ratio,
    )
    shape_audit_path = eval_dir / f"shape_audit_{eval_ts}.json"
    with open(shape_audit_path, "w", encoding="utf-8") as f:
        json.dump(shape_audit, f, ensure_ascii=False, indent=2)
    shape_audit_csv = _save_c_axis_shape_audit_csv(eval_dir, eval_ts, shape_audit)
    print(f"[HB][eval] saved shape audit: {shape_audit_path}", flush=True)
    if shape_audit_csv is not None:
        print(f"[HB][eval] saved shape audit csv: {shape_audit_csv}", flush=True)

    ppo_audit = ((shape_audit.get("by_method", {}) or {}).get("ppo", {}) or {}).get("metrics", {})
    if isinstance(ppo_audit, dict):
        failed_metrics = [
            m for m, obj in ppo_audit.items()
            if isinstance(obj, dict) and (not bool(obj.get("pass", False)))
        ]
        if failed_metrics:
            _warn_once(
                "ppo_shape_fail",
                f"PPO{shape_target_variant}C: {failed_metrics}",
            )
    summary = {
        "eval_ts": eval_ts,
        "train_run_dir": str(train_run_dir),
        "env_yaml": str(env_yaml),
        "train_yaml": str(train_yaml),
        "device": device,
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "ppo_ckpt": str(ppo_ckpt) if ppo_ckpt is not None else None,
        "ppo_hidden": int(ppo_hidden) if ppo_hidden is not None else None,
        "variants": list(variants.keys()),
        "methods": methods,
        "methods_display": [method_label_map.get(str(m), str(m)) for m in methods],
        "method_label_map": method_label_map,
        "loads": loads,
        "n_seeds": n_seeds,
        "ep_per": int(ep_per),
        "ep_per_progressive": bool(ep_per_progressive),
        "ep_per_by_load": {f"{float(ld):.1f}": int(ep_per_by_load.get(_load_key(float(ld)), int(max(1, ep_per)))) for ld in loads},
        "deterministic": deterministic,
        "strict_ten_loads": bool(strict_ten_loads),
        "require_explicit_ckpt": bool(require_explicit_ckpt),
        "explicit_ckpt": str(explicit_ppo_ckpt) if explicit_ppo_ckpt is not None else None,
        "requested_methods": requested_methods,
        "fail_on_method_drop": bool(fail_on_method_drop),
        "resume_strict_guard": bool(resume_strict_guard),
        "require_clean_output": bool(require_clean_output),
        "force_overwrite": bool(force_overwrite),
        "fail_on_incomplete_raw": bool(fail_on_incomplete_raw),
        "raw_duplicate_key_count": int(duplicate_key_count),
        "metrics": metrics,
        "x_axis_metric": {
            "name": "required_cpu_cycles_c",
            "unit": "Mcycles",
            "x_label": "Required CPU cycles C (Mcycles)",
            "x_values_mcycles": x_axis_cpu_mcycles,
            "x_from_load_formula": "C_mcycles = C_cycles_base*(load_scale**load_alpha_C)*(1+0.10*load_scale^2)/1e6",
        },
        "x_axis_metric_task_size_d": {
            "name": "task_size_d",
            "unit": "MB",
            "x_label": "Task size D (MB)",
            "x_values_mb": x_axis_task_mb,
            "x_from_load_formula": "D_mb = D_bits_base*(load_scale**load_alpha_D)*(1+0.15*load_scale^2)/8e6",
        },
        "shape_audit_c_axis": shape_audit,
        "shape_audit_json": str(shape_audit_path),
        "shape_audit_csv": (str(shape_audit_csv) if shape_audit_csv is not None else None),
        "series": series,
        "raw_csv": str(raw_csv),
        "ppo_use_obs_norm": bool(ppo_use_obs_norm),
        "ppo_obs_norm_loaded": bool(_EVAL_OBS_NORMALIZER is not None),
        "cross_algo_obs_protocol": {
            "ppo": "obs_norm_if_enabled_else_raw",
            "ddpg": "raw",
            "sac": "raw",
            "td3": "raw",
            "a2c": "raw",
            "balanced": "raw",
            "greedy_delay": "raw",
            "greedy_energy": "raw",
            "myopic_optimization": "raw",
            "always_comp": "raw",
            "never_comp": "raw",
        },
        "baseline_auto_match": {
            "target_seed": ppo_train_seed,
            "target_total_steps": ppo_train_total_steps,
            "allow_crossrun_baseline": bool(allow_crossrun_baseline),
        },
        "require_3drl_final_ckpt": bool(require_3drl_final_ckpt),
        "baseline_ckpt_quality": baseline_ckpt_quality,
    }

    summary_path = eval_dir / f"eval_summary_{eval_ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[HB][eval] saved summary: {summary_path}", flush=True)

    meta_path = eval_dir / f"eval_meta_{eval_ts}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "eval_ts": eval_ts,
            "train_run_dir": str(train_run_dir),
            "env_yaml": str(env_yaml),
            "train_yaml": str(train_yaml),
            "ppo_ckpt": str(ppo_ckpt) if ppo_ckpt is not None else None,
            "ppo_display": ppo_display,
            "methods": methods,
            "method_label_map": method_label_map,
            "ppo_use_obs_norm": bool(ppo_use_obs_norm),
            "ppo_obs_norm_loaded": bool(_EVAL_OBS_NORMALIZER is not None),
            "strict_ten_loads": bool(strict_ten_loads),
            "require_explicit_ckpt": bool(require_explicit_ckpt),
            "explicit_ckpt": str(explicit_ppo_ckpt) if explicit_ppo_ckpt is not None else None,
            "requested_methods": requested_methods,
            "fail_on_method_drop": bool(fail_on_method_drop),
            "resume_strict_guard": bool(resume_strict_guard),
            "require_clean_output": bool(require_clean_output),
            "force_overwrite": bool(force_overwrite),
            "fail_on_incomplete_raw": bool(fail_on_incomplete_raw),
            "raw_duplicate_key_count": int(duplicate_key_count),
            "ep_per": int(ep_per),
            "ep_per_progressive": bool(ep_per_progressive),
            "ep_per_by_load": {f"{float(ld):.1f}": int(ep_per_by_load.get(_load_key(float(ld)), int(max(1, ep_per)))) for ld in loads},
            "require_3drl_final_ckpt": bool(require_3drl_final_ckpt),
            "baseline_ckpt_quality": baseline_ckpt_quality,
            "eval_dir": str(eval_dir),
            "fig_dir": str(fig_dir),
            "raw_csv": str(raw_csv),
            "shape_audit_json": str(shape_audit_path),
            "shape_audit_csv": (str(shape_audit_csv) if shape_audit_csv is not None else None),
            "x_axis_cpu_mcycles": x_axis_cpu_mcycles,
            "x_axis_task_mb": x_axis_task_mb,
        }, f, ensure_ascii=False, indent=2)
    print(f"[HB][eval] saved meta: {meta_path}", flush=True)
    
    
    _eval_ts_l = str(eval_ts).strip().lower()
    _allow_phaseg_main_upsert = bool(
        ("phaseg" in _eval_ts_l)
        and (_eval_ts_l.endswith("_main") or _eval_ts_l == "phaseg_main")
    )
    if _allow_phaseg_main_upsert:
        try:
            phaseg_ckpt_text = (str(ppo_ckpt) if ppo_ckpt is not None else os.environ.get("CKPT_PATH", "").strip())
            
            if not str(phaseg_ckpt_text).strip():
                raise RuntimeError(
                    "[phaseg][FATAL] missing checkpoint for phaseg main eval summary "
                    "(need PPO ckpt or CKPT_PATH)."
                )
            phaseg_summary_path = upsert_phaseg_eval_summary(
                run_dir=train_run_dir,
                role="main",
                eval_ts=eval_ts,
                ckpt_path=str(phaseg_ckpt_text),
                artifacts={
                    "eval_summary_json": str(summary_path),
                    "eval_meta_json": str(meta_path),
                    "eval_raw_csv": str(raw_csv),
                },
            )
            if phaseg_summary_path is not None:
                print(f"[HB][phaseg] updated summary: {phaseg_summary_path}", flush=True)
        except Exception as e:
            msg = str(e)
            if "[phaseg][FATAL]" in msg:
                raise
            print(f"[HB][phaseg][WARN] failed to update summary after main-eval: {type(e).__name__}: {e}", flush=True)

    
    traj_metrics = [
        "traj_idle_ratio",
        "traj_boundary_stick_frac",
        "traj_user_nn_dist_norm",
        "traj_centroid_gap_norm",
        "traj_switchback_ratio",
    ]
    traj_summary_csv = eval_dir / "trajectory_metrics_summary.csv"
    traj_cols = [
        "variant",
        "method",
        "load",
        "n",
        "traj_idle_ratio_mean",
        "traj_idle_ratio_std",
        "traj_idle_ratio_ci95",
        "traj_boundary_stick_frac_mean",
        "traj_boundary_stick_frac_std",
        "traj_boundary_stick_frac_ci95",
        "traj_user_nn_dist_norm_mean",
        "traj_user_nn_dist_norm_std",
        "traj_user_nn_dist_norm_ci95",
        "traj_centroid_gap_norm_mean",
        "traj_centroid_gap_norm_std",
        "traj_centroid_gap_norm_ci95",
        "traj_switchback_ratio_mean",
        "traj_switchback_ratio_std",
        "traj_switchback_ratio_ci95",
    ]
    try:
        with open(traj_summary_csv, "w", encoding="utf-8", newline="") as f:
            ww = csv.DictWriter(f, fieldnames=traj_cols)
            ww.writeheader()
            for (vname, method, load), rows in sorted(grp.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
                row_out: Dict[str, Any] = {
                    "variant": str(vname),
                    "method": str(method),
                    "load": float(load),
                    "n": int(len(rows)),
                }
                for tm in traj_metrics:
                    arr = np.asarray([rr.get(tm, np.nan) for rr in rows], dtype=np.float64)
                    arr = arr[np.isfinite(arr)]
                    if arr.size <= 0:
                        mu = float("nan")
                        sd = float("nan")
                        ci = float("nan")
                    else:
                        mu = float(np.mean(arr))
                        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                        ci = float(1.96 * sd / np.sqrt(max(1, arr.size))) if arr.size > 1 else 0.0
                    row_out[f"{tm}_mean"] = mu
                    row_out[f"{tm}_std"] = sd
                    row_out[f"{tm}_ci95"] = ci
                ww.writerow(row_out)
        print(f"[HB][eval] saved trajectory summary: {traj_summary_csv}", flush=True)
    except Exception as e:
        print(f"[HB][eval][WARN] failed to save trajectory summary: {type(e).__name__}: {e}", flush=True)

    
    plot_metric_map = {
        "paper_cost": "Ablation10_PaperCost.png",
        "mean_T_off": "Ablation10_Delay.png",
        "mean_energy": "Ablation10_Energy.png",
    }
    published_fig_names: List[str] = []
    for m, fig_name in plot_metric_map.items():
        curve_pack = {}
        for sname, md in series.items():
            curve_pack[sname] = {"x": md[m]["x"], "y": md[m]["y"], "e": md[m]["e"]}
        fig_path = fig_dir / fig_name
        _plot_metric(
            fig_path,
            loads,
            curve_pack,
            ylabel=_metric_axis_label(m),
            show_ci=plot_ci,
            x_values=x_axis_cpu_mcycles,
            xlabel=_cpu_axis_label(),
        )
        print(f"[HB][eval] fig {fig_path}", flush=True)
        published_fig_names.append(fig_name)
        if plot_logy:
            fig_log = fig_dir / fig_name.replace(".png", "_logy.png")
            _plot_metric(
                fig_log,
                loads,
                curve_pack,
                ylabel=_metric_axis_label(m),
                yscale="log",
                show_ci=plot_ci,
                x_values=x_axis_cpu_mcycles,
                xlabel=_cpu_axis_label(),
            )
            print(f"[HB][eval] fig {fig_log}", flush=True)
        if plot_zoom:
            fig_zoom = fig_dir / fig_name.replace(".png", "_zoom.png")
            _plot_metric(
                fig_zoom,
                loads,
                curve_pack,
                ylabel=_metric_axis_label(m),
                zoom_quantile=plot_zoom_q,
                show_ci=plot_ci,
                x_values=x_axis_cpu_mcycles,
                xlabel=_cpu_axis_label(),
            )
            print(f"[HB][eval] fig {fig_zoom}", flush=True)

    
    ratio_series: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for method in methods:
        ratio_series[method] = {}
        for vname in variants.keys():
            sname = f"{vname}|{method}"
            if sname not in series:
                continue
            ratio_series[method][vname] = {mm: series[sname][mm]["y"] for mm in metrics}

    ratio_metric_map = {
        "paper_cost": "Ablation10_ratio_PaperCost.png",
        "mean_T_off": "Ablation10_ratio_Delay.png",
        "mean_energy": "Ablation10_ratio_Energy.png",
    }
    for m, fig_name in ratio_metric_map.items():
        ratio_path = fig_dir / fig_name
        _plot_ratio(
            ratio_path,
            loads,
            ratio_series,
            metric=m,
            x_values=x_axis_cpu_mcycles,
            xlabel=_cpu_axis_label(),
        )
        print(f"[HB][eval] fig {ratio_path}", flush=True)

    if _should_publish_ablation10(eval_ts, loads):
        try:
            ablation_dir = train_run_dir / "figs" / FIG_BUCKET_ABLATION10
            ablation_dir.mkdir(parents=True, exist_ok=True)
            
            ablation_dir_ts = train_run_dir / "figs" / f"{eval_ts}_Ablation10"
            ablation_dir_ts.mkdir(parents=True, exist_ok=True)

            for fig_name in sorted(set(published_fig_names)):
                _publish_key_plot(
                    train_run_dir=train_run_dir,
                    src_path=fig_dir / fig_name,
                    bucket_name=FIG_BUCKET_ABLATION10,
                    dst_name=fig_name,
                )
                _publish_key_plot(
                    train_run_dir=train_run_dir,
                    src_path=fig_dir / fig_name,
                    bucket_name=f"{eval_ts}_Ablation10",
                    dst_name=fig_name,
                )

            
            full_only_name_map = {
                "paper_cost": "Ablation10_FullOnly_PaperCost.png",
                "mean_T_off": "Ablation10_FullOnly_Delay.png",
                "mean_energy": "Ablation10_FullOnly_Energy.png",
            }
            for metric_key, dst_name in full_only_name_map.items():
                full_curve_pack: Dict[str, Dict[str, List[float]]] = {}
                for method in methods:
                    sname = f"full|{method}"
                    if sname not in series:
                        continue
                    mm = series[sname].get(metric_key, {})
                    full_curve_pack[sname] = {
                        "x": list(mm.get("x", [])),
                        "y": list(mm.get("y", [])),
                        "e": list(mm.get("e", [])),
                    }
                if not full_curve_pack:
                    continue
                _plot_full_only_teacher_style(
                    ablation_dir / dst_name,
                    loads,
                    full_curve_pack,
                    ylabel=_metric_axis_label(metric_key),
                    show_ci=plot_ci,
                    x_values=x_axis_cpu_mcycles,
                    xlabel=_cpu_axis_label(),
                )
                print(f"[HB][eval] fig {ablation_dir / dst_name}", flush=True)
                try:
                    shutil.copy2(str(ablation_dir / dst_name), str(ablation_dir_ts / dst_name))
                except Exception as e_copy:
                    print(f"[HB][eval][WARN] snapshot copy failed: {type(e_copy).__name__}: {e_copy}", flush=True)

            
            full_only_d_name_map = {
                "paper_cost": "Ablation10_FullOnly_TaskD_PaperCost.png",
                "mean_T_off": "Ablation10_FullOnly_TaskD_Delay.png",
                "mean_energy": "Ablation10_FullOnly_TaskD_Energy.png",
            }
            for metric_key, dst_name in full_only_d_name_map.items():
                full_curve_pack_d: Dict[str, Dict[str, List[float]]] = {}
                for method in methods:
                    sname = f"full|{method}"
                    if sname not in series:
                        continue
                    mm = series[sname].get(metric_key, {})
                    full_curve_pack_d[sname] = {
                        "x": list(mm.get("x", [])),
                        "y": list(mm.get("y", [])),
                        "e": list(mm.get("e", [])),
                    }
                if not full_curve_pack_d:
                    continue
                _plot_full_only_teacher_style(
                    ablation_dir / dst_name,
                    loads,
                    full_curve_pack_d,
                    ylabel=_metric_axis_label(metric_key),
                    show_ci=plot_ci,
                    x_values=x_axis_task_mb,
                    xlabel="Task size D (MB)",
                )
                print(f"[HB][eval] fig {ablation_dir / dst_name}", flush=True)
                try:
                    shutil.copy2(str(ablation_dir / dst_name), str(ablation_dir_ts / dst_name))
                except Exception as e_copy:
                    print(f"[HB][eval][WARN] snapshot copy failed: {type(e_copy).__name__}: {e_copy}", flush=True)

            
            variant_bar_name_map = {
                "paper_cost": "Ablation10_VariantBars_PaperCost.png",
                "mean_T_off": "Ablation10_VariantBars_Delay.png",
                "mean_energy": "Ablation10_VariantBars_Energy.png",
            }
            for metric_key, dst_name in variant_bar_name_map.items():
                _plot_variant_grouped_bar(
                    ablation_dir / dst_name,
                    series=series,
                    methods=methods,
                    metric_key=metric_key,
                    ylabel=_metric_axis_label(metric_key),
                    variants_order=("full", "no_comp", "no_ris"),
                )
                print(f"[HB][eval] fig {ablation_dir / dst_name}", flush=True)
                try:
                    shutil.copy2(str(ablation_dir / dst_name), str(ablation_dir_ts / dst_name))
                except Exception as e_copy:
                    print(f"[HB][eval][WARN] snapshot copy failed: {type(e_copy).__name__}: {e_copy}", flush=True)

            
            ct_curve_pack: Dict[str, Dict[str, List[float]]] = {}
            for method in methods:
                sname = f"full|{method}"
                x_ct: List[float] = []
                y_ct: List[float] = []
                e_ct: List[float] = []
                for load in loads:
                    rows = grp.get(("full", method, _load_key(load)), [])
                    arr = np.asarray([rr.get("mean_S_size", np.nan) for rr in rows], dtype=np.float64)
                    mu = float(np.nanmean(arr)) if arr.size else float("nan")
                    ci = _ci95(arr) if arr.size else 0.0
                    x_ct.append(float(load))
                    y_ct.append(mu)
                    e_ct.append(ci)
                if np.any(np.isfinite(np.asarray(y_ct, dtype=np.float64))):
                    ct_curve_pack[sname] = {"x": x_ct, "y": y_ct, "e": e_ct}
            if ct_curve_pack:
                ct_path = ablation_dir / "Ablation10_Ct.png"
                _plot_full_only_teacher_style(
                    ct_path,
                    loads,
                    ct_curve_pack,
                    ylabel=" |S(t)|",
                    show_ci=plot_ci,
                    x_values=x_axis_cpu_mcycles,
                    xlabel=_cpu_axis_label(),
                )
                print(f"[HB][eval] fig {ct_path}", flush=True)
                try:
                    shutil.copy2(str(ct_path), str(ablation_dir_ts / "Ablation10_Ct.png"))
                except Exception as e_copy:
                    print(f"[HB][eval][WARN] snapshot copy failed: {type(e_copy).__name__}: {e_copy}", flush=True)

            
            insight_dir = train_run_dir / "figs" / FIG_BUCKET_ABLATION10_INSIGHTS
            insight_dir.mkdir(parents=True, exist_ok=True)
            insight_dir_ts = train_run_dir / "figs" / f"{eval_ts}_Ablation10_Insights"
            insight_dir_ts.mkdir(parents=True, exist_ok=True)
            try:
                fig_cdf = insight_dir / "Ablation10_HighLoadCDF_Delay.png"
                _plot_highload_cdf(
                    fig_cdf,
                    raw_rows=raw_rows,
                    methods=methods,
                    metric_key="mean_T_off",
                    metric_label=_metric_axis_label("mean_T_off"),
                    variant="full",
                    load_threshold=0.8,
                )
                if fig_cdf.exists():
                    print(f"[HB][eval] fig {fig_cdf}", flush=True)
                    try:
                        shutil.copy2(str(fig_cdf), str(insight_dir_ts / fig_cdf.name))
                    except Exception as e_copy:
                        print(f"[HB][eval][WARN] snapshot copy failed: {type(e_copy).__name__}: {e_copy}", flush=True)

                fig_gain = insight_dir / "Ablation10_GainDecomp_PaperCost.png"
                _plot_gain_decomposition(
                    fig_gain,
                    loads=loads,
                    x_values=x_axis_cpu_mcycles,
                    methods=methods,
                    series=series,
                    metric_key="paper_cost",
                )
                if fig_gain.exists():
                    print(f"[HB][eval] fig {fig_gain}", flush=True)
                    try:
                        shutil.copy2(str(fig_gain), str(insight_dir_ts / fig_gain.name))
                    except Exception as e_copy:
                        print(f"[HB][eval][WARN] snapshot copy failed: {type(e_copy).__name__}: {e_copy}", flush=True)

                fig_pareto = insight_dir / "Ablation10_Pareto_DelayEnergy.png"
                _plot_delay_energy_pareto(
                    fig_pareto,
                    methods=methods,
                    series=series,
                )
                if fig_pareto.exists():
                    print(f"[HB][eval] fig {fig_pareto}", flush=True)
                    try:
                        shutil.copy2(str(fig_pareto), str(insight_dir_ts / fig_pareto.name))
                    except Exception as e_copy:
                        print(f"[HB][eval][WARN] snapshot copy failed: {type(e_copy).__name__}: {e_copy}", flush=True)

                fig_beh = insight_dir / "Ablation10_Behavior_vs_C.png"
                _plot_behavior_vs_c(
                    fig_beh,
                    loads=loads,
                    x_values=x_axis_cpu_mcycles,
                    methods=methods,
                    grp=grp,
                    variant="full",
                )
                if fig_beh.exists():
                    print(f"[HB][eval] fig {fig_beh}", flush=True)
                    try:
                        shutil.copy2(str(fig_beh), str(insight_dir_ts / fig_beh.name))
                    except Exception as e_copy:
                        print(f"[HB][eval][WARN] snapshot copy failed: {type(e_copy).__name__}: {e_copy}", flush=True)

                fig_gen = insight_dir / "Ablation10_Generalization_Degradation.png"
                _plot_generalization_or_load_stress(
                    fig_gen,
                    run_dir=train_run_dir,
                    methods=methods,
                    loads=loads,
                    x_values=x_axis_cpu_mcycles,
                    series=series,
                    env_cfg=base_env_cfg,
                )
                if fig_gen.exists():
                    print(f"[HB][eval] fig {fig_gen}", flush=True)
                    try:
                        shutil.copy2(str(fig_gen), str(insight_dir_ts / fig_gen.name))
                    except Exception as e_copy:
                        print(f"[HB][eval][WARN] snapshot copy failed: {type(e_copy).__name__}: {e_copy}", flush=True)
            except Exception as e_feat:
                print(f"[HB][eval][WARN] enhanced ablation figs failed: {type(e_feat).__name__}: {e_feat}", flush=True)

            
            removed_dup = 0
            for fig_name in sorted(set(published_fig_names)):
                p_dup = fig_dir / fig_name
                if p_dup.exists():
                    try:
                        p_dup.unlink()
                        removed_dup += 1
                    except Exception as _e_rm:
                        print(f"[HB][eval][WARN] remove duplicate fig failed: {p_dup} {_e_rm}", flush=True)

            print(
                f"[HB][eval] Published {FIG_BUCKET_ABLATION10} bundle ({len(sorted(set(published_fig_names)))} figs) "
                f"+ full_only=3; removed_eval_ts_duplicates={removed_dup}; "
                f"snapshot_main={ablation_dir_ts} snapshot_insights={insight_dir_ts}",
                flush=True,
            )
        except Exception as e:
            print(f"[HB][eval][WARN] publish {FIG_BUCKET_ABLATION10} failed: {type(e).__name__}: {e}", flush=True)

    print("[HB][eval] DONE", flush=True)


if __name__ == "__main__":
    main()
