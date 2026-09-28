"""Jev check before opening/adding to a long.

Rules (policy.py relies on these too):
- only buys come through here. Jev can say no or make the size smaller
  (size_multiplier is 0..1), it can't create a trade or make one bigger.
- if anything goes wrong with Jev (no key, timeout, bad response...) the
  answer is no. Better to miss a trade than to trade blind.
- sells / stop losses never touch this, so if Jev is down we can still get out.

We send it a short summary of the features (well under 2k tokens), not raw
candles - it's not meant to do maths on OHLC.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.jev.client import JevClient, JevError, boolean, choice, score

logger = logging.getLogger(__name__)

REGIMES: dict[str, str] = {
    "strong_uptrend": "Persistent rise: price above rising averages, positive multi-horizon returns.",
    "weak_uptrend": "Mild or early rise with mixed confirmation.",
    "range": "Sideways, mean-reverting, no directional edge.",
    "weak_downtrend": "Mild or early decline with mixed confirmation.",
    "strong_downtrend": "Persistent decline: price below falling averages, negative multi-horizon returns.",
    "disorderly": "Crash-like or chaotic: extreme volatility, gaps, or panic volume.",
}

SETUP_LEVELS: list[str | None] = [
    "Poor: no edge or clearly against the evidence.",
    "Weak: thin evidence, likely noise.",
    "Acceptable: modest evidence in favor.",
    "Good: several independent signals agree.",
    "Excellent: strong, consistent, multi-horizon confirmation.",
]


@dataclass(frozen=True)
class GateThresholds:
    # defaults picked by eye from a few sim runs, override via env

    min_coherence: float = 0.55
    max_risk_flag: float = 0.60
    min_setup_score: float = 1.8  # on the 0-4 scale
    blocked_regimes: tuple[str, ...] = ("strong_downtrend", "disorderly")
    min_regime_confidence: float = 0.35

    @classmethod
    def from_env(cls) -> "GateThresholds":
        env = os.environ

        def _f(name: str, default: float) -> float:
            try:
                return float(env.get(name, default))
            except ValueError:
                return default

        return cls(
            min_coherence=_f("JEV_MIN_COHERENCE", cls.min_coherence),
            max_risk_flag=_f("JEV_MAX_RISK_FLAG", cls.max_risk_flag),
            min_setup_score=_f("JEV_MIN_SETUP_SCORE", cls.min_setup_score),
            min_regime_confidence=_f("JEV_MIN_REGIME_CONFIDENCE", cls.min_regime_confidence),
        )


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    size_multiplier: float
    reasons: tuple[str, ...]
    answers: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "size_multiplier": round(self.size_multiplier, 4),
            "reasons": list(self.reasons),
            "answers": self.answers,
            "cost_usd": self.cost_usd,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


def _deny(reason: str, *, error: str | None = None, answers: dict | None = None, **acct: float) -> GateDecision:
    return GateDecision(False, 0.0, (reason,), answers or {}, error=error, **acct)


class JevEntryGate:
    """Asks Jev 4 questions about a buy and applies the thresholds."""

    def __init__(self, client: JevClient | None = None, thresholds: GateThresholds | None = None) -> None:
        self.client = client or JevClient()
        self.thresholds = thresholds or GateThresholds.from_env()

    @staticmethod
    def build_state(features: Mapping[str, Any], proposal: Mapping[str, Any], position: Mapping[str, Any]) -> str:
        return json.dumps(
            {
                "task": "Review a proposed LONG entry in a spot crypto asset (no shorting, no leverage).",
                "market_features": dict(features),
                "current_position": dict(position),
                "proposal": {
                    "action": proposal.get("action"),
                    "conviction": proposal.get("conviction"),
                    "thesis": str(proposal.get("thesis", ""))[:1200],
                    "invalidation": str(proposal.get("invalidation", ""))[:400],
                },
            },
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def questions() -> dict[str, dict[str, Any]]:
        return {
            "regime": choice("Classify the current market regime from market_features.", REGIMES),
            "thesis_coherent": boolean(
                "The proposal's thesis is consistent with market_features and actually supports buying now.",
                true="The cited evidence matches the features and points to upside.",
                false="The thesis contradicts the features, is vague, or argues against buying.",
            ),
            "setup_quality": score("Rate the quality of this long entry setup given the evidence.", SETUP_LEVELS),
            "risk_flag": boolean(
                "An elevated-risk condition (extreme volatility, crash momentum, severe overextension, "
                "or liquidity stress) makes opening a new long imprudent right now."
            ),
        }

    def review_entry(
        self,
        features: Mapping[str, Any],
        proposal: Mapping[str, Any],
        position: Mapping[str, Any],
    ) -> GateDecision:
        # never raises - any failure is just a deny
        if str(proposal.get("action", "")).lower() != "buy":
            return _deny("gate only reviews buy entries")
        try:
            result = self.client.evaluate(self.build_state(features, proposal, position), self.questions())
        except JevError as exc:
            logger.warning("jev gate unavailable, denying entry: %s", exc)
            return _deny("jev unavailable (fail-closed)", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.warning("jev gate crashed, denying entry: %s", exc, exc_info=True)
            return _deny("jev error (fail-closed)", error=f"{type(exc).__name__}: {exc}")

        a = result.answers
        t = self.thresholds
        acct = {"cost_usd": result.cost_usd, "latency_ms": result.latency_ms}
        summary = {
            "regime": a["regime"].choice,
            "regime_confidence": round(a["regime"].confidence, 3),
            "thesis_coherent_p": round(a["thesis_coherent"].probability or 0.0, 3),
            "setup_score": round(a["setup_quality"].score or 0.0, 3),
            "setup_confidence": round(a["setup_quality"].confidence, 3),
            "risk_flag_p": round(a["risk_flag"].probability or 0.0, 3),
        }

        reasons: list[str] = []
        if summary["regime"] in t.blocked_regimes and summary["regime_confidence"] >= t.min_regime_confidence:
            reasons.append(f"regime {summary['regime']} (conf {summary['regime_confidence']})")
        if summary["thesis_coherent_p"] < t.min_coherence:
            reasons.append(f"thesis incoherent (p={summary['thesis_coherent_p']})")
        if summary["risk_flag_p"] > t.max_risk_flag:
            reasons.append(f"risk flag (p={summary['risk_flag_p']})")
        if summary["setup_score"] < t.min_setup_score:
            reasons.append(f"weak setup (score={summary['setup_score']})")
        if reasons:
            return GateDecision(False, 0.0, tuple(reasons), summary, **acct)

        # smaller size for so-so setups / some risk, never above 1x
        quality = summary["setup_score"] / 4.0
        multiplier = quality * (1.0 - 0.5 * summary["risk_flag_p"]) * 1.25
        multiplier = max(0.25, min(1.0, multiplier))
        return GateDecision(True, multiplier, ("approved",), summary, **acct)
