"""
Run  I/O /


-  run /
-  `configs/` 
- checkpoint  `checkpoints/`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def make_run_dir(project_root: Path, subdir: str = "runs/paper", ts_name: Optional[str] = None) -> Path:
    """
     <project_root>/<subdir>/<ts_name>  run_dir
    - configs/
    - checkpoints/ checkpoint
    - evals//
    - figs/

     <subdir>/_latest_run.txt 
    """
    base = project_root / Path(subdir)
    base.mkdir(parents=True, exist_ok=True)

    if ts_name is None:
        
        from src.utils.hb import now_ts
        ts_name = now_ts()

    run_dir = base / ts_name
    ensure_run_subdirs(run_dir)

    (base / "_latest_run.txt").write_text(run_dir.name, encoding="utf-8")
    return run_dir


def snapshot_cfgs(run_dir: Path, cfg_env: Dict[str, Any], cfg_train: Dict[str, Any]) -> None:
    """
     env/train  run_dir/configs/

     `configs/` run_dir 
    """
    cfg_dir = run_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    (cfg_dir / "env_fixed.yaml").write_text(
        yaml.safe_dump(cfg_env, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (cfg_dir / "train_tianshou_fixed.yaml").write_text(
        yaml.safe_dump(cfg_train, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def ensure_run_subdirs(run_dir: Path) -> None:
    """ run_dir  no-op"""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "evals").mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
