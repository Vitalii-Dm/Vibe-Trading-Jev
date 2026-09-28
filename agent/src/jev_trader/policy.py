"""Turns (proposal, Jev, risk limits) into at most one order.

Order of checks matters:
1. stop loss first. If it's hit we sell everything, don't even ask the
   brain or Jev.
2. sells don't go through Jev. If Jev is down we still want to be able to
   get out. Can't sell more than we hold (spot, no shorts).
3. buys go through Jev if there is a gate. Jev can block or shrink the size,
   never make it bigger.
4. then the hard limits: max % per asset, trades per day, min order size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.jev.gate import GateDecision, JevEntryGate
from src.jev_trader.brain import Proposal


@dataclass(frozen=True)
class RiskConfig:
    # plain limits, no model involved

    max_alloc_pct: float = 30.0  # max position value per asset, % of equity
    min_buy_conviction: float = 0.55
    stop_loss_pct: float = 5.0  # forced exit below entry
    max_trades_per_day: int = 12
    max_drawdown_pct: float = 15.0  # flatten everything + halt, from equity peak
    min_trade_usd: float = 10.0

    @classmethod
    def from_env(cls) -> "RiskConfig":
        env = os.environ

        def _f(name: str, default: float) -> float:
            try:
                return float(env.get(name, default))
            except ValueError:
                return default

        return cls(
            max_alloc_pct=_f("JEV_TRADER_MAX_ALLOC_PCT", cls.max_alloc_pct),
            min_buy_conviction=_f("JEV_TRADER_MIN_CONVICTION", cls.min_buy_conviction),
            stop_loss_pct=_f("JEV_TRADER_STOP_LOSS_PCT", cls.stop_loss_pct),
            max_trades_per_day=int(_f("JEV_TRADER_MAX_TRADES_PER_DAY", cls.max_trades_per_day)),
            max_drawdown_pct=_f("JEV_TRADER_MAX_DRAWDOWN_PCT", cls.max_drawdown_pct),
            min_trade_usd=_f("JEV_TRADER_MIN_TRADE_USD", cls.min_trade_usd),
        )


@dataclass(frozen=True)
class AssetState:
    symbol: str
    price: float  # last closed-bar close
    qty: float  # base units held
    entry_price: float | None
    equity: float  # total account equity in quote currency
    trades_today: int = 0

    @property
    def position_value(self) -> float:
        return self.qty * self.price


@dataclass(frozen=True)
class Order:
    # buys are in USDT (notional), sells in coins (qty)

    side: str  # "buy" | "sell"
    notional_usd: float = 0.0
    qty: float = 0.0
    reason: str = ""
    forced: bool = False


@dataclass
class PolicyOutcome:
    order: Order | None
    reason: str
    gate: GateDecision | None = None
    vetoed_notional: float = 0.0  # what Jev stopped us buying, for the "what if" pnl
    notes: dict[str, Any] = field(default_factory=dict)


def position_view(state: AssetState, cfg: RiskConfig) -> dict[str, Any]:
    # what the brain/Jev see about the position, relative numbers only
    cap = state.equity * cfg.max_alloc_pct / 100.0
    pnl = (state.price / state.entry_price - 1) * 100 if state.entry_price and state.qty > 0 else 0.0
    return {
        "in_position": state.qty * state.price >= cfg.min_trade_usd,
        "position_pct_of_allowed": round(state.position_value / cap * 100, 1) if cap > 0 else 0.0,
        "unrealized_pnl_pct": round(pnl, 2),
    }


def decide(
    proposal: Proposal,
    state: AssetState,
    features: Mapping[str, Any],
    cfg: RiskConfig,
    gate: JevEntryGate | None,
) -> PolicyOutcome:
    held_value = state.position_value

    # 1. stop loss
    if state.qty > 0 and state.entry_price and held_value >= cfg.min_trade_usd:
        if state.price <= state.entry_price * (1 - cfg.stop_loss_pct / 100.0):
            return PolicyOutcome(Order("sell", qty=state.qty, reason="stop-loss", forced=True), "stop-loss")

    cap = state.equity * cfg.max_alloc_pct / 100.0

    # 2. sells
    if proposal.action == "sell":
        if held_value < cfg.min_trade_usd:
            return PolicyOutcome(None, "sell ignored: flat")
        target_value = cap * proposal.size_pct / 100.0
        # models sometimes send sell with size_pct=100 meaning "sell all of it".
        # if the target isn't actually below what we hold, just exit fully -
        # staying in by mistake is worse than getting out by mistake
        if target_value < cfg.min_trade_usd or target_value >= held_value - cfg.min_trade_usd:
            return PolicyOutcome(Order("sell", qty=state.qty, reason="brain exit"), "brain exit")
        sell_value = held_value - target_value
        if sell_value < cfg.min_trade_usd:
            return PolicyOutcome(None, "sell too small")
        qty = min(state.qty, sell_value / state.price)
        return PolicyOutcome(Order("sell", qty=qty, reason="brain trim"), "brain trim")

    if proposal.action != "buy":
        return PolicyOutcome(None, "hold")

    # 3. buys - limits first, Jev last since it costs money
    if proposal.conviction < cfg.min_buy_conviction:
        return PolicyOutcome(None, f"conviction {proposal.conviction:.2f} < {cfg.min_buy_conviction}")
    if state.trades_today >= cfg.max_trades_per_day:
        return PolicyOutcome(None, "daily trade cap reached")
    target_value = min(cap, cap * proposal.size_pct / 100.0)
    add = target_value - held_value
    if add < cfg.min_trade_usd:
        return PolicyOutcome(None, "already at target")

    decision: GateDecision | None = None
    if gate is not None:
        decision = gate.review_entry(features, proposal.to_dict(), position_view(state, cfg))
        if not decision.allowed:
            return PolicyOutcome(None, "jev veto: " + "; ".join(decision.reasons), decision, vetoed_notional=add)
        multiplier = max(0.0, min(1.0, decision.size_multiplier))
        sized = add * multiplier
        if sized < cfg.min_trade_usd:
            return PolicyOutcome(None, "jev-sized entry below minimum", decision, vetoed_notional=add)
        add = sized

    return PolicyOutcome(Order("buy", notional_usd=add, reason="brain buy"), "brain buy", decision)
