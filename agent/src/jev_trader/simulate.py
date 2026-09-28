"""Replay history and compare the different setups ("arms") on the same bars.

Arms: buyhold (equal weight, all in), rules (SMA baseline), brain (LLM only)
and brain+jev (LLM with Jev checking the buys).

No peeking: decision at bar t only sees bars[:t+1], and the order fills at
open[t+1] with fee + slippage. Stop losses are checked on close[t] and also
fill at the next open.

For brain+jev, every time Jev blocks a buy we also record what that buy
would have made over the next veto_horizon bars (after fees). That way the
report shows if Jev's vetoes actually helped or just cost us money.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.jev.gate import JevEntryGate
from src.jev_trader.brain import Brain
from src.jev_trader.brokers import SimBroker
from src.jev_trader.features import MIN_BARS, compute_features
from src.jev_trader.policy import AssetState, RiskConfig, decide, position_view

logger = logging.getLogger(__name__)

_FEATURE_WINDOW = 200  # only pass the last N bars to compute_features, keeps it fast


@dataclass
class ArmResult:
    name: str
    equity_curve: list[float] = field(default_factory=list)
    trades: int = 0
    fees: float = 0.0
    brain_cost_usd: float = 0.0
    jev_cost_usd: float = 0.0
    jev_calls: int = 0
    jev_errors: int = 0
    vetoes: list[dict[str, Any]] = field(default_factory=list)
    halted_at: int | None = None
    exposure_bars: int = 0
    parse_errors: int = 0

    def metrics(self, bars_per_year: float) -> dict[str, Any]:
        eq = pd.Series(self.equity_curve, dtype=float)
        if len(eq) < 2:
            return {"arm": self.name}
        rets = eq.pct_change().dropna()
        peak = eq.cummax()
        sharpe = float(rets.mean() / rets.std() * math.sqrt(bars_per_year)) if rets.std() > 0 else 0.0
        out: dict[str, Any] = {
            "arm": self.name,
            "total_return_pct": round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2),
            "max_drawdown_pct": round(float(((eq - peak) / peak).min()) * 100, 2),
            "sharpe": round(sharpe, 2),
            "trades": self.trades,
            "fees": round(self.fees, 2),
            "exposure_pct": round(self.exposure_bars / max(1, len(eq)) * 100, 1),
            "brain_cost_usd": round(self.brain_cost_usd, 4),
            "jev_cost_usd": round(self.jev_cost_usd, 5),
            "jev_calls": self.jev_calls,
            "jev_errors": self.jev_errors,
            "parse_errors": self.parse_errors,
            "halted_at_bar": self.halted_at,
        }
        if self.vetoes:
            cf = [v["counterfactual_return_pct"] for v in self.vetoes if v["counterfactual_return_pct"] is not None]
            out["vetoes"] = len(self.vetoes)
            out["veto_avg_counterfactual_return_pct"] = round(sum(cf) / len(cf), 3) if cf else None
            out["veto_counterfactual_pnl_usd"] = round(
                sum(v["notional"] * v["counterfactual_return_pct"] / 100 for v in self.vetoes if v["counterfactual_return_pct"] is not None),
                2,
            )
        return out


def load_bars(symbols: list[str], start: str, end: str, interval: str) -> dict[str, pd.DataFrame]:
    # public OKX candles, trimmed to the timestamps all symbols have
    from backtest.loaders.okx import DataLoader

    frames = DataLoader().fetch(symbols, start, end, interval=interval)
    missing = [s for s in symbols if s not in frames]
    if missing:
        raise RuntimeError(f"no OKX data for {missing}")
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    aligned = {s: frames[s].loc[common].sort_index() for s in symbols}
    # last bar might still be open, drop it
    seconds = {"15m": 900, "30m": 1800, "1H": 3600, "1h": 3600, "4H": 14400, "4h": 14400, "1D": 86400}.get(interval, 3600)
    last_open = pd.Timestamp(common.max())
    if last_open.tzinfo is not None:
        last_open = last_open.tz_convert("UTC").tz_localize(None)
    if last_open + pd.Timedelta(seconds=seconds) > pd.Timestamp.now(tz="UTC").tz_localize(None):
        aligned = {s: df.iloc[:-1] for s, df in aligned.items()}
    return aligned


def _buyhold(bars: dict[str, pd.DataFrame], cash: float, fee: float, start_t: int) -> ArmResult:
    arm = ArmResult("buyhold")
    per = cash / len(bars)
    qty = {s: per * (1 - fee) / float(df["open"].iloc[start_t + 1]) for s, df in bars.items()}
    n = len(next(iter(bars.values())))
    for t in range(start_t, n - 1):
        if t == start_t:
            arm.equity_curve.append(cash)
            continue
        arm.equity_curve.append(sum(qty[s] * float(df["close"].iloc[t]) for s, df in bars.items()))
        arm.exposure_bars += 1
    arm.trades = len(bars)
    arm.fees = cash * fee
    return arm


def _run_arm(
    name: str,
    bars: dict[str, pd.DataFrame],
    brain: Brain,
    gate: JevEntryGate | None,
    risk: RiskConfig,
    *,
    cash: float,
    fee: float,
    slippage_bps: float,
    decide_every: int,
    veto_horizon: int,
    bar_hours: float,
    progress: Callable[[str], None] | None,
) -> ArmResult:
    arm = ArmResult(name)
    broker = SimBroker(cash, fee, slippage_bps)
    symbols = list(bars)
    n = len(bars[symbols[0]])
    start_t = MIN_BARS - 1
    peak = cash
    halted = False
    trades_by_day: dict[str, int] = {}
    index = bars[symbols[0]].index

    for t in range(start_t, n - 1):
        closes = {s: float(bars[s]["close"].iloc[t]) for s in symbols}
        equity = broker.equity(closes)
        arm.equity_curve.append(equity)
        if any(broker.qty.get(s, 0) * closes[s] >= risk.min_trade_usd for s in symbols):
            arm.exposure_bars += 1
        peak = max(peak, equity)
        next_open = {s: float(bars[s]["open"].iloc[t + 1]) for s in symbols}

        if halted:
            continue
        if equity < peak * (1 - risk.max_drawdown_pct / 100.0):
            for s in symbols:
                if broker.qty.get(s):
                    f = broker.sell(s, broker.qty[s], next_open[s])
                    arm.trades += int(f.ok)
            halted, arm.halted_at = True, t
            continue

        day = str(index[t].date())
        for s in symbols:
            state = AssetState(
                symbol=s,
                price=closes[s],
                qty=broker.qty.get(s, 0.0),
                entry_price=broker.entry.get(s),
                equity=equity,
                trades_today=trades_by_day.get(day, 0),
            )
            window = bars[s].iloc[max(0, t + 1 - _FEATURE_WINDOW): t + 1]
            features = compute_features(window, bar_hours)
            if (t - start_t) % decide_every == 0:
                proposal = brain.propose(features, position_view(state, risk))
                arm.brain_cost_usd += proposal.cost_usd
                arm.parse_errors += int(proposal.parse_error is not None)
            else:
                from src.jev_trader.brain import hold

                proposal = hold("off-cycle", source="sim")
            outcome = decide(proposal, state, features, risk, gate)
            if outcome.gate is not None:
                arm.jev_calls += 1
                arm.jev_cost_usd += outcome.gate.cost_usd
                arm.jev_errors += int(outcome.gate.error is not None)
            if outcome.vetoed_notional > 0:
                exit_t = min(n - 1, t + 1 + veto_horizon)
                entry_px = next_open[s]
                exit_px = float(bars[s]["open"].iloc[exit_t])
                cf = ((exit_px / entry_px) * (1 - fee) ** 2 - 1) * 100 if exit_t > t + 1 else None
                arm.vetoes.append({"bar": t, "symbol": s, "notional": outcome.vetoed_notional,
                                   "reason": outcome.reason, "counterfactual_return_pct": cf})
            order = outcome.order
            if order is None:
                continue
            fill = broker.buy(s, order.notional_usd, next_open[s]) if order.side == "buy" else broker.sell(s, order.qty, next_open[s])
            if fill.ok:
                arm.trades += 1
                trades_by_day[day] = trades_by_day.get(day, 0) + 1
        if progress and (t - start_t) % 100 == 0:
            progress(f"[{name}] bar {t - start_t}/{n - 1 - start_t} equity={equity:,.0f}")

    arm.fees = broker.fees_paid
    return arm


def run_simulation(
    *,
    symbols: list[str],
    start: str,
    end: str,
    interval: str = "1H",
    arms: list[str],
    brain_factory: Callable[[str], Brain],
    gate: JevEntryGate | None = None,
    risk: RiskConfig | None = None,
    cash: float = 10_000.0,
    fee: float = 0.001,
    slippage_bps: float = 5.0,
    decide_every: int = 1,
    veto_horizon: int = 24,
    bars: dict[str, pd.DataFrame] | None = None,
    out_dir: Path | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    risk = risk or RiskConfig.from_env()
    bars = bars or load_bars(symbols, start, end, interval)
    n = len(next(iter(bars.values())))
    if n < MIN_BARS + 2:
        raise RuntimeError(f"only {n} bars; need > {MIN_BARS + 1}")
    bar_hours = {"15m": 0.25, "30m": 0.5, "1H": 1.0, "1h": 1.0, "4H": 4.0, "4h": 4.0, "1D": 24.0}.get(interval, 1.0)
    bars_per_year = 24 * 365 / bar_hours

    results: list[ArmResult] = []
    started = time.monotonic()
    for arm_name in arms:
        if arm_name == "buyhold":
            results.append(_buyhold(bars, cash, fee, MIN_BARS - 1))
            continue
        if arm_name == "brain+jev" and gate is None:
            raise ValueError("brain+jev arm requires a Jev gate")
        brain = brain_factory(arm_name)
        results.append(
            _run_arm(
                arm_name, bars, brain, gate if arm_name == "brain+jev" else None, risk,
                cash=cash, fee=fee, slippage_bps=slippage_bps, decide_every=decide_every,
                veto_horizon=veto_horizon, bar_hours=bar_hours, progress=progress,
            )
        )

    idx = next(iter(bars.values())).index
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(bars),
        "interval": interval,
        "period": [str(idx[MIN_BARS - 1]), str(idx[-1])],
        "bars_traded": n - MIN_BARS,
        "cash": cash,
        "fee": fee,
        "slippage_bps": slippage_bps,
        "decide_every": decide_every,
        "risk": risk.__dict__,
        "elapsed_s": round(time.monotonic() - started, 1),
        "arms": [r.metrics(bars_per_year) for r in results],
        "vetoes": next((r.vetoes for r in results if r.name == "brain+jev"), []),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"sim-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["report_path"] = str(path)
    return report


def format_report(report: dict[str, Any]) -> str:
    cols = ["arm", "total_return_pct", "max_drawdown_pct", "sharpe", "trades", "fees", "exposure_pct",
            "brain_cost_usd", "jev_calls", "vetoes", "veto_counterfactual_pnl_usd"]
    lines = [
        f"Period {report['period'][0]} -> {report['period'][1]}  ({report['bars_traded']} bars, {report['interval']}, "
        f"{', '.join(report['symbols'])})",
        " | ".join(cols),
        " | ".join("---" for _ in cols),
    ]
    for arm in report["arms"]:
        lines.append(" | ".join(str(arm.get(c, "")) for c in cols))
    lines.append("veto_counterfactual_pnl_usd < 0 means the entries Jev blocked would have lost money (Jev helped).")
    return "\n".join(lines)
