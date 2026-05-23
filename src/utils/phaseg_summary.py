from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def upsert_phaseg_eval_summary(
    *,
    run_dir: Path,
    role: str,
    eval_ts: str,
    ckpt_path: Optional[str] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
     PhaseG //

    
    ----
    run_dir:  run 
    role: main / fixed / gen
    eval_ts:  phaseg
    ckpt_path:  checkpoint
    artifacts: 
    """
    eval_ts_n = str(eval_ts).strip()
    if not eval_ts_n:
        return None
    if "phaseg" not in eval_ts_n.lower():
        return None

    role_n = str(role).strip().lower()
    if role_n not in ("main", "fixed", "gen"):
        return None

    run_dir = Path(run_dir)
    summary_path = run_dir / "evals" / "phaseg_eval_summary.json"
    obj = _load_json(summary_path)

    if not obj:
        obj = {
            "phaseg_ckpt_path": "",
            "phaseg_eval_ts": "",
            "phaseg_fixed_eval_ts": "",
            "phaseg_gen_eval_ts": "",
            "records": {},
            "updated_at": _now_ts(),
        }

    def _norm_ckpt_text(text: str) -> str:
        t = str(text).strip()
        if not t:
            return ""
        try:
            return str(Path(t).resolve())
        except Exception:
            return t

    ckpt_text = _norm_ckpt_text(str(ckpt_path or ""))
    if ckpt_text:
        
        
        existing_ckpt = _norm_ckpt_text(str(obj.get("phaseg_ckpt_path", "") or ""))
        if existing_ckpt and (existing_ckpt != ckpt_text):
            raise RuntimeError(
                "[phaseg][FATAL] checkpoint mismatch in phaseg_eval_summary: "
                f"existing={existing_ckpt} new={ckpt_text}"
            )
        obj["phaseg_ckpt_path"] = ckpt_text

    if role_n == "main":
        obj["phaseg_eval_ts"] = eval_ts_n
    elif role_n == "fixed":
        obj["phaseg_fixed_eval_ts"] = eval_ts_n
    else:
        obj["phaseg_gen_eval_ts"] = eval_ts_n

    records = obj.get("records", {})
    if not isinstance(records, dict):
        records = {}
    rec = records.get(eval_ts_n, {})
    if not isinstance(rec, dict):
        rec = {}
    rec["role"] = role_n
    rec["updated_at"] = _now_ts()
    if isinstance(artifacts, dict) and artifacts:
        rec_art = rec.get("artifacts", {})
        if not isinstance(rec_art, dict):
            rec_art = {}
        rec_art.update(artifacts)
        rec["artifacts"] = rec_art
    records[eval_ts_n] = rec
    obj["records"] = records
    obj["updated_at"] = _now_ts()

    _dump_json(summary_path, obj)

    
    ts_copy_path = run_dir / "evals" / eval_ts_n / "phaseg_eval_summary.json"
    try:
        _dump_json(ts_copy_path, obj)
    except Exception:
        pass

    return summary_path
