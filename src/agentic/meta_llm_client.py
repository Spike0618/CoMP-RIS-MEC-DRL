from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


class OpenAICompatMetaClient:
    """
     OpenAI-compatible LLM  meta action 
    """

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str,
        timeout_ms: int,
        temperature: float,
        max_tokens: int,
        use_json_schema: bool = True,
    ):
        self.provider = str(provider or "openai").strip().lower()
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.model = str(model or "").strip()
        self.api_key = str(api_key or "").strip()
        self.timeout_ms = int(max(100, timeout_ms))
        self.temperature = float(max(0.0, temperature))
        self.max_tokens = int(max(16, max_tokens))
        self.use_json_schema = bool(use_json_schema)

    def is_ready(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)

    def _url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def request_action(self, prompt: str) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("llm_client_not_ready")

        sys_msg = (
            "You are a strict controller. Output JSON only with keys: "
            "action_id, confidence, reason."
        )
        body: Dict[str, Any] = {
            "model": self.model,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": str(prompt)},
            ],
        }
        if self.use_json_schema:
            body["response_format"] = {"type": "json_object"}

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=self._url(),
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_ms / 1000.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"http_error:{e.code}:{txt[:240]}") from e
        except Exception as e:
            raise RuntimeError(f"request_error:{type(e).__name__}:{e}") from e
        latency_ms = float((time.perf_counter() - t0) * 1000.0)

        try:
            obj = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"response_not_json:{type(e).__name__}") from e

        choice = {}
        try:
            choice = (((obj.get("choices") or [])[0] or {}).get("message") or {})
        except Exception:
            choice = {}
        content = str(choice.get("content", "")).strip()
        if not content:
            raise RuntimeError("empty_content")

        
        if content.startswith("```"):
            content = content.strip("`")
            if "\n" in content:
                content = content.split("\n", 1)[1]
        content = content.strip()
        try:
            j = json.loads(content)
        except Exception as e:
            raise RuntimeError(f"content_json_parse_error:{type(e).__name__}") from e

        action_id = str(j.get("action_id", "")).strip().lower()
        confidence = _to_float(j.get("confidence", 0.0), 0.0)
        reason = str(j.get("reason", "")).strip()
        usage = obj.get("usage", {}) if isinstance(obj, dict) else {}
        token_in = int(usage.get("prompt_tokens", 0) or 0)
        token_out = int(usage.get("completion_tokens", 0) or 0)

        return {
            "action_id": action_id,
            "confidence": float(confidence),
            "reason": reason,
            "latency_ms": float(latency_ms),
            "token_in": int(token_in),
            "token_out": int(token_out),
            "cost_est": 0.0,  
        }
