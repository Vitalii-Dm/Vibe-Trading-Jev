"""The paper/live loop for OKX spot.

Every bar close, for each symbol:
    features -> brain -> policy (stop loss, limits, Jev on buys) -> order

The bot only counts coins it bought itself (bot_qty in state.json). The OKX
demo account comes pre-funded with random coins and I didn't want it selling
those, or anything I hold myself. Sells are min(bot_qty, available). If you go
live use a separate sub-account anyway.

Safety stuff checked each cycle:
- kill switch (src.live.halt, "okx"): no brain, no Jev, no new buys. Stop
  losses still run on paper. On a live profile the mandate gate blocks
  everything while halted, stop losses too.
- drawdown from peak equity over the limit: sell the bot's positions, then
  trip the kill switch (even if some sells failed).
- trades/day and per-asset caps (in policy.py).
Everything goes to journal.jsonl so I can look back at what happened.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.jev.gate import JevEntryGate
from src.jev_trader.brain import Brain, hold
from src.jev_trader.brokers import OKXBroker
from src.jev_trader.features import compute_features
from src.jev_trader.policy import AssetState, RiskConfig, decide, position_view

logger = logging.getLogger(__name__)

_INTERVAL_SECONDS = {"15m": 900, "30m": 1800, "1H": 3600, "4H": 14400}
_BROKER_KEY = "okx"


@dataclass
class TraderState:
    # saved to state.json between cycles

    path: Path
    peak_equity: float = 0.0
    entries: dict[str, float] = field(default_factory=dict)
    bot_qty: dict[str, float] = field(default_factory=dict)
    trades_by_day: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "TraderState":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                path,
                float(data.get("peak_equity", 0.0)),
                dict(data.get("entries") or {}),
                {k: float(v) for k, v in (data.get("bot_qty") or {}).items()},
                dict(data.get("trades_by_day") or {}),
            )
        return cls(path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "peak_equity": self.peak_equity,
            "entries": self.entries,
            "bot_qty": self.bot_qty,
            "trades_by_day": self.trades_by_day,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def closed_bars(symbol: str, interval: str, lookback_days: int = 20) -> pd.DataFrame:
    # drop the last bar if it hasn't closed yet
    from backtest.loaders.okx import DataLoader

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    df = DataLoader().fetch([symbol], start.isoformat(), end.isoformat(), interval=interval).get(symbol)
    if df is None or df.empty:
        raise RuntimeError(f"no bars for {symbol}")
    seconds = _INTERVAL_SECONDS.get(interval, 3600)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    last_open = pd.Timestamp(df.index[-1])
    if last_open.tzinfo is not None:
        last_open = last_open.tz_convert("UTC").tz_localize(None)
    if last_open + pd.Timedelta(seconds=seconds) > now:
        df = df.iloc[:-1]
    return df


class Trader:
    def __init__(
        self,
        *,
        symbols: list[str],
        brain: Brain,
        gate: JevEntryGate | None,
        broker: OKXBroker,
        risk: RiskConfig,
        interval: str,
        state_dir: Path,
        quote_ccy: str = "USDT",
    ) -> None:
        self.symbols = symbols
        self.brain = brain
        self.gate = gate
        self.broker = broker
        self.risk = risk
        self.interval = interval
        self.quote = quote_ccy
        self.state = TraderState.load(state_dir / "state.json")
        self.journal = state_dir / "journal.jsonl"
        self.bar_hours = _INTERVAL_SECONDS.get(interval, 3600) / 3600

    def _log(self, event: dict[str, Any]) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
        logger.info("%s", event.get("event"))

    def _equity(self, balances: dict[str, dict[str, float]], prices: dict[str, float]) -> float:
        # USDT + only the bot's own coins
        eq = balances.get(self.quote, {}).get("equity", 0.0)
        for s in self.symbols:
            eq += self.state.bot_qty.get(s, 0.0) * prices.get(s, 0.0)
        return eq

    def _held(self, symbol: str, balances: dict[str, dict[str, float]]) -> float:
        base = symbol.split("-")[0]
        available = balances.get(base, {}).get("available", 0.0)
        return max(0.0, min(self.state.bot_qty.get(symbol, 0.0), available))

    def _record_fill(self, symbol: str, side: str, fill: Any, held: float, price: float) -> None:
        st = self.state
        if side == "buy":
            st.entries[symbol] = (st.entries.get(symbol, price) * held + price * fill.qty) / max(held + fill.qty, 1e-12)
            st.bot_qty[symbol] = st.bot_qty.get(symbol, 0.0) + fill.qty
            return
        remaining = max(0.0, st.bot_qty.get(symbol, 0.0) - fill.qty)
        if remaining * price < self.risk.min_trade_usd:  # dust, call it flat
            st.bot_qty.pop(symbol, None)
            st.entries.pop(symbol, None)
        else:
            st.bot_qty[symbol] = remaining

    def _flatten_and_halt(self, balances: dict[str, dict[str, float]], prices: dict[str, float], reason: str) -> None:
        from src.live.halt import trip_halt

        try:
            for s in self.symbols:
                held = self._held(s, balances)
                if held * prices[s] < self.risk.min_trade_usd:
                    continue
                try:
                    fill = self.broker.sell(s, held, prices[s])
                    if fill.ok:
                        self._record_fill(s, "sell", fill, held, prices[s])
                    self._log({"event": "drawdown_flatten", "symbol": s, "fill": {k: v for k, v in fill.__dict__.items() if k != "raw"}})
                except Exception as exc:  # noqa: BLE001 - one failing shouldn't stop the others
                    self._log({"event": "drawdown_flatten_error", "symbol": s, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            # halt has to come after the sells, on live it would block them
            trip_halt("jev-trader", reason, _BROKER_KEY)
            self.state.save()

    def run_once(self) -> list[dict[str, Any]]:
        from src.live.halt import halt_flag_set

        halted = halt_flag_set(_BROKER_KEY)
        bars = {s: closed_bars(s, self.interval) for s in self.symbols}
        prices = {s: float(df["close"].iloc[-1]) for s, df in bars.items()}
        balances = self.broker.balances()
        equity = self._equity(balances, prices)
        st = self.state
        day = datetime.now(timezone.utc).date().isoformat()
        st.trades_by_day = {day: st.trades_by_day.get(day, 0)}

        if halted:
            self._log({"event": "halted", "detail": "kill switch set, no entries (stop losses still run)"})
        else:
            st.peak_equity = max(st.peak_equity, equity)
            if equity < st.peak_equity * (1 - self.risk.max_drawdown_pct / 100.0):
                self._flatten_and_halt(balances, prices, f"max drawdown {self.risk.max_drawdown_pct}% breached")
                return []

        results = []
        for s in self.symbols:
            held = self._held(s, balances)
            state = AssetState(s, prices[s], held, st.entries.get(s), equity, st.trades_by_day[day])
            features = compute_features(bars[s], self.bar_hours)
            if halted:
                # halted: skip brain/Jev, but decide() can still trigger the stop loss
                proposal = hold("halted", source="kill-switch")
                outcome = decide(proposal, state, features, self.risk, None)
                if outcome.order is None:
                    continue
            else:
                proposal = self.brain.propose(features, position_view(state, self.risk))
                outcome = decide(proposal, state, features, self.risk, self.gate)
            record: dict[str, Any] = {
                "event": "decision",
                "symbol": s,
                "equity": round(equity, 2),
                "features": features,
                "proposal": proposal.to_dict(),
                "policy": outcome.reason,
                "gate": outcome.gate.to_dict() if outcome.gate else None,
            }
            order = outcome.order
            if order is not None:
                if order.side == "buy":
                    fill = self.broker.buy(s, order.notional_usd, prices[s])
                else:
                    fill = self.broker.sell(s, min(order.qty, held), prices[s])
                record["order"] = order.__dict__
                record["fill"] = {k: v for k, v in fill.__dict__.items() if k != "raw"}
                if fill.ok:
                    st.trades_by_day[day] += 1
                    self._record_fill(s, order.side, fill, held, prices[s])
            self._log(record)
            results.append(record)

        st.save()
        return results

    def run_forever(self) -> None:
        seconds = _INTERVAL_SECONDS.get(self.interval, 3600)
        while True:
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("cycle failed")
                self._log({"event": "cycle_error", "error": f"{type(exc).__name__}: {exc}"})
            now = time.time()
            # +20s so OKX has actually published the closed candle
            time.sleep(seconds - (now % seconds) + 20)
