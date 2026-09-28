"""Small HTTP client for Jev.

Two ways to reach it:

  gateway   - Vercel AI Gateway, POST {base}/evaluation-model. Same thing
              @ai-sdk/gateway does under experimental_evaluate(). Question
              types are choice/score/boolean and there's no confidence field
              in the answer, so we compute one from the probabilities.
  typesafe  - TypeSafe directly, POST {base}/v1/systemone. Here the yes/no
              type is called "noul" (and the answer field too).

Callers always say "boolean", the client renames it for typesafe.

Env vars: JEV_TRANSPORT, AI_GATEWAY_API_KEY, AI_GATEWAY_BASE_URL,
TYPESAFE_API_KEY, TYPESAFE_BASE_URL, JEV_MODEL, JEV_TIMEOUT_SECONDS.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

import requests

Transport = Literal["gateway", "typesafe"]

GATEWAY_DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v4/ai"
GATEWAY_DEFAULT_MODEL = "typesafe-ai/jev"
GATEWAY_PROTOCOL_VERSION = "0.0.1"
TYPESAFE_DEFAULT_BASE_URL = "https://api.typesafe.ai"
TYPESAFE_DEFAULT_MODEL = "jev-latest"

# $0.042 per 1M input tokens, output is free
JEV_INPUT_USD_PER_TOKEN = 0.042 / 1_000_000

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})


class JevError(RuntimeError):
    """Anything that went wrong talking to Jev."""


# question helpers


def choice(instructions: str, criteria: Mapping[str, str | None]) -> dict[str, Any]:
    """criteria is {option: description or None}"""
    if not criteria:
        raise ValueError("choice criteria must be non-empty")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: str, levels: list[str | None]) -> dict[str, Any]:
    """levels are 0-indexed, Jev wants between 2 and 10 of them"""
    if not 2 <= len(levels) <= 10:
        raise ValueError("score needs 2-10 levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def boolean(instructions: str, *, true: str | None = None, false: str | None = None) -> dict[str, Any]:
    """yes/no question, answer is P(true)"""
    question: dict[str, Any] = {"type": "boolean", "instructions": instructions}
    if true is not None or false is not None:
        question["criteria"] = {"true": true, "false": false}
    return question


@dataclass(frozen=True)
class JevAnswer:
    """Answer in one shape regardless of transport.

    For booleans confidence is just |2p - 1|, i.e. how far from a coin flip.
    """

    type: str
    choice: str | None = None
    score: float | None = None
    probability: float | None = None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class JevResult:
    answers: dict[str, JevAnswer]
    model: str
    input_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * JEV_INPUT_USD_PER_TOKEN


def distribution_confidence(probabilities: Mapping[str, float]) -> float:
    # 1 - normalized entropy. one option gets everything -> 1, flat -> 0
    values = [max(0.0, float(p)) for p in probabilities.values()]
    total = sum(values)
    n = len(values)
    if n < 2 or total <= 0:
        return 1.0 if n == 1 else 0.0
    entropy = -sum((p / total) * math.log(p / total) for p in values if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(n)))


def _normalize_answer(raw: Mapping[str, Any]) -> JevAnswer:
    kind = str(raw.get("type") or "")
    probs = {str(k): float(v) for k, v in dict(raw.get("probabilities") or {}).items()}
    if kind == "choice":
        conf = raw.get("confidence")
        return JevAnswer(
            type="choice",
            choice=str(raw["choice"]),
            confidence=float(conf) if conf is not None else distribution_confidence(probs),
            probabilities=probs,
        )
    if kind == "score":
        conf = raw.get("confidence")
        return JevAnswer(
            type="score",
            score=float(raw["score"]),
            confidence=float(conf) if conf is not None else distribution_confidence(probs),
            probabilities=probs,
        )
    if kind in ("boolean", "noul"):
        p = float(raw["probability"] if kind == "boolean" else raw["noul"])
        if not 0.0 <= p <= 1.0:
            raise JevError(f"boolean probability out of range: {p}")
        return JevAnswer(type="boolean", probability=p, confidence=abs(2 * p - 1))
    raise JevError(f"unknown answer type {kind!r}")


class JevClient:
    """Sync client, retries a couple of times on 429/5xx/529.

    `post` is only there so tests can swap out requests.post.
    """

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        post: Callable[..., Any] | None = None,
    ) -> None:
        env = os.environ
        self.transport: Transport = (transport or env.get("JEV_TRANSPORT") or "gateway").strip().lower()  # type: ignore[assignment]
        if self.transport not in ("gateway", "typesafe"):
            raise JevError(f"unknown JEV_TRANSPORT {self.transport!r}")
        if self.transport == "gateway":
            self.api_key = api_key or env.get("AI_GATEWAY_API_KEY", "")
            self.base_url = (base_url or env.get("AI_GATEWAY_BASE_URL") or GATEWAY_DEFAULT_BASE_URL).rstrip("/")
            self.model = model or env.get("JEV_MODEL") or GATEWAY_DEFAULT_MODEL
        else:
            self.api_key = api_key or env.get("TYPESAFE_API_KEY", "")
            self.base_url = (base_url or env.get("TYPESAFE_BASE_URL") or TYPESAFE_DEFAULT_BASE_URL).rstrip("/")
            self.model = model or env.get("JEV_MODEL") or TYPESAFE_DEFAULT_MODEL
        self.timeout = float(timeout if timeout is not None else env.get("JEV_TIMEOUT_SECONDS") or 10.0)
        self.max_retries = max(0, int(max_retries))
        self._post = post or requests.post

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _request(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> tuple[str, dict, dict]:
        if self.transport == "gateway":
            url = f"{self.base_url}/evaluation-model"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "ai-gateway-protocol-version": GATEWAY_PROTOCOL_VERSION,
                "ai-evaluation-model-specification-version": "4",
                "ai-model-id": self.model,
            }
            body = {"state": state, "questions": {k: dict(v) for k, v in questions.items()}}
            return url, headers, body
        url = f"{self.base_url}/v1/systemone"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        wire: dict[str, dict] = {}
        for key, q in questions.items():
            q = dict(q)
            if q.get("type") == "boolean":
                q["type"] = "noul"
            wire[key] = q
        return url, headers, {"model": self.model, "state": state, "questions": wire}

    def evaluate(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> JevResult:
        """Ask all questions about `state`. Raises JevError on any problem,
        including a response that's missing some of the answers."""
        if not self.api_key:
            key_env = "AI_GATEWAY_API_KEY" if self.transport == "gateway" else "TYPESAFE_API_KEY"
            raise JevError(f"Jev not configured: set {key_env}")
        if not questions:
            raise JevError("at least one question is required")
        url, headers, body = self._request(state, questions)

        started = time.monotonic()
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._post(url, json=body, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"network error: {type(exc).__name__}"
            else:
                status = int(getattr(resp, "status_code", 0))
                if status == 200:
                    return self._parse(resp, questions, started)
                last_error = f"HTTP {status}: {str(getattr(resp, 'text', ''))[:300]}"
                if status not in _RETRYABLE_STATUS:
                    break
                retry_after = _retry_after_seconds(resp)
                if retry_after is not None and attempt < self.max_retries:
                    time.sleep(min(retry_after, 5.0))
                    continue
            if attempt < self.max_retries:
                time.sleep(0.25 * (2**attempt))
        raise JevError(last_error)

    def _parse(self, resp: Any, questions: Mapping[str, Any], started: float) -> JevResult:
        try:
            payload = resp.json()
            raw_answers = payload["answers"]
            answers = {key: _normalize_answer(raw_answers[key]) for key in questions}
        except JevError:
            raise
        except Exception as exc:  # missing keys, bad json etc
            raise JevError(f"unparseable Jev response: {type(exc).__name__}: {exc}") from exc
        usage = payload.get("usage") or {}
        tokens = usage.get("inputTokens", usage.get("input_tokens", 0)) or 0
        return JevResult(
            answers=answers,
            model=str(payload.get("model") or self.model),
            input_tokens=int(tokens),
            latency_ms=(time.monotonic() - started) * 1000.0,
        )


def _retry_after_seconds(resp: Any) -> float | None:
    headers = getattr(resp, "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
