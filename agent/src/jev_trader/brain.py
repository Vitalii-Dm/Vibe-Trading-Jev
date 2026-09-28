"""The "brains" - look at features + position, say buy/sell/hold.

LLMBrain: one short prompt per decision via whatever LLM the project is set
up with (LANGCHAIN_PROVIDER / LANGCHAIN_MODEL_NAME, I use OpenRouter with
deepseek/deepseek-v4-flash-0731). Replies are cached on disk so re-running a
sim doesn't cost anything. If the reply doesn't parse we just hold.

RulesBrain: dumb SMA trend follower to compare against, no key needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

logger = logging.getLogger(__name__)

ACTIONS = ("buy", "sell", "hold")


@dataclass(frozen=True)
class Proposal:
    # size_pct = target position as % of the per-asset cap

    action: str
    conviction: float
    size_pct: float
    thesis: str
    invalidation: str = ""
    source: str = ""
    cost_usd: float = 0.0
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "conviction": self.conviction,
            "size_pct": self.size_pct,
            "thesis": self.thesis,
            "invalidation": self.invalidation,
            "source": self.source,
            "cost_usd": self.cost_usd,
            "parse_error": self.parse_error,
        }


def hold(reason: str, *, source: str, parse_error: str | None = None, cost_usd: float = 0.0) -> Proposal:
    return Proposal("hold", 0.0, 0.0, reason, source=source, parse_error=parse_error, cost_usd=cost_usd)


class Brain(Protocol):
    name: str

    def propose(self, features: Mapping[str, Any], position: Mapping[str, Any]) -> Proposal: ...


# baseline


class RulesBrain:
    """buy when the SMAs line up, sell when the trend breaks"""

    name = "rules"

    def propose(self, features: Mapping[str, Any], position: Mapping[str, Any]) -> Proposal:
        in_pos = bool(position.get("in_position"))
        up = features["sma20_above_sma50"] and features["sma20_slope_5bar_pct"] > 0 and features["dist_sma50_pct"] > 0
        overheated = features["rsi14"] > 78
        broken = (not features["sma20_above_sma50"]) or features["dist_sma50_pct"] < -1.0
        if not in_pos and up and not overheated:
            return Proposal("buy", 0.6, 100.0, "SMA20>SMA50, rising SMA20, price above SMA50.", "Close below SMA50.", self.name)
        if in_pos and broken:
            return Proposal("sell", 0.6, 0.0, "Trend broken: SMA20<SMA50 or price >1% below SMA50.", source=self.name)
        return hold("no rule fired", source=self.name)


# LLM

SYSTEM_PROMPT = """You are a disciplined crypto spot trader managing ONE anonymized asset.
Constraints: long or flat only (no shorting, no leverage). Fees+slippage are ~0.15% per side, so avoid churning.
You receive precomputed features of the last CLOSED bar and the current position.
Decide one action:
- "buy": open or add to the long (only with genuine evidence of upside).
- "sell": reduce or close the long (use when the thesis is invalidated or risk is rising).
- "hold": do nothing.
Respond with ONLY a JSON object, no prose, exactly these keys:
{"action": "buy"|"sell"|"hold", "conviction": 0.0-1.0, "size_pct": 0-100, "thesis": "<=60 words citing specific features", "invalidation": "<=25 words"}
size_pct is the TARGET position size as a percent of the allowed allocation (0 = flat, 100 = full)."""


def build_user_prompt(features: Mapping[str, Any], position: Mapping[str, Any]) -> str:
    return json.dumps({"features": dict(features), "position": dict(position)}, ensure_ascii=False, sort_keys=True)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_proposal(text: str, *, source: str, cost_usd: float = 0.0) -> Proposal:
    # anything off about the reply -> hold
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):] if "{" in raw else raw
    match = _JSON_OBJECT.search(raw)
    if not match:
        return hold("unparseable reply", source=source, parse_error="no JSON object", cost_usd=cost_usd)
    try:
        obj = json.loads(match.group(0))
        action = str(obj["action"]).strip().lower()
        if action not in ACTIONS:
            raise ValueError(f"bad action {action!r}")
        conviction = float(obj.get("conviction", 0.0))
        size_pct = float(obj.get("size_pct", 0.0))
        if not (0.0 <= conviction <= 1.0) or not (0.0 <= size_pct <= 100.0):
            raise ValueError("conviction/size_pct out of range")
        return Proposal(
            action,
            conviction,
            size_pct,
            str(obj.get("thesis", ""))[:600],
            str(obj.get("invalidation", ""))[:300],
            source,
            cost_usd,
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return hold("invalid reply", source=source, parse_error=str(exc)[:200], cost_usd=cost_usd)


class LLMBrain:
    """cache_dir=None turns the cache off. `chat` is a fake for tests,
    otherwise it goes through src.providers.chat.ChatLLM. Prices are $/1M
    tokens and only used for the cost estimate."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        cache_dir: Path | None = None,
        chat: Any = None,
        price_in: float = 0.021,
        price_out: float = 0.32,
        timeout: int = 60,
    ) -> None:
        self._chat = chat
        self._model_name = model_name
        self.cache_dir = cache_dir
        self.price_in = price_in
        self.price_out = price_out
        self.timeout = timeout
        self._llm = None
        self.name = "llm"

    def model_key(self) -> str:
        # goes into the cache key, otherwise changing model would reuse old answers
        import os

        provider = os.environ.get("LANGCHAIN_PROVIDER", "")
        model = self._model_name or os.environ.get("LANGCHAIN_MODEL_NAME", "")
        return f"{provider}/{model}"

    def _complete(self, messages: list[dict[str, str]]) -> str:
        if self._chat is not None:
            return self._chat(messages)
        if self._llm is None:
            from src.providers.chat import ChatLLM

            self._llm = ChatLLM(model_name=self._model_name)
            self.name = f"llm:{self._llm.model_name}"
        return self._llm.chat(messages, timeout=self.timeout).content or ""

    def _estimate_cost(self, prompt: str, reply: str) -> float:
        # rough, ~4 chars per token
        return (len(prompt) / 4 * self.price_in + len(reply) / 4 * self.price_out) / 1_000_000

    def propose(self, features: Mapping[str, Any], position: Mapping[str, Any]) -> Proposal:
        user = build_user_prompt(features, position)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        key = hashlib.sha256(f"{self.model_key()}\n{SYSTEM_PROMPT}\n{user}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.txt" if self.cache_dir else None
        if cache_file is not None and cache_file.exists():
            return parse_proposal(cache_file.read_text(encoding="utf-8"), source=f"{self.name}(cached)")
        try:
            reply = self._complete(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM brain failed: %s", exc)
            return hold("llm unavailable", source=self.name, parse_error=f"{type(exc).__name__}: {exc}"[:200])
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(reply, encoding="utf-8")
        return parse_proposal(reply, source=self.name, cost_usd=self._estimate_cost(SYSTEM_PROMPT + user, reply))
