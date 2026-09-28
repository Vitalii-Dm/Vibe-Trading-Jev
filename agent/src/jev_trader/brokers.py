"""Where orders actually go.

SimBroker - fake account in memory, charges fee + slippage. The simulator
fills it at the next bar's open.

OKXBroker - real OKX spot through the existing trading service, demo
(okx-paper-trade) unless you say otherwise. Keys come from env on every call,
nothing is saved to disk. Some OKX gotchas handled here:
  - size has to be rounded down to lotSz
  - buy fees come out of the base coin, so you can't sell exactly what you
    bought -> sells are capped at what's actually available
  - anything under minSz is dust, treat as flat

Note: paper profiles don't go through the mandate check in
service.place_order, that's why loop.py has its own caps + kill switch
check. Live still needs a committed mandate and we don't go around that.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fill:
    ok: bool
    side: str
    qty: float = 0.0
    price: float = 0.0
    fee: float = 0.0
    notional: float = 0.0
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


# sim


class SimBroker:
    # fees taken in quote on both buy and sell (simpler than real OKX)

    def __init__(self, cash: float = 10_000.0, fee_rate: float = 0.001, slippage_bps: float = 5.0) -> None:
        self.cash = float(cash)
        self.fee_rate = fee_rate
        self.slippage = slippage_bps / 10_000.0
        self.qty: dict[str, float] = {}
        self.entry: dict[str, float] = {}
        self.fees_paid = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices.get(s, 0.0) for s, q in self.qty.items())

    def buy(self, symbol: str, notional: float, price: float) -> Fill:
        notional = min(notional, self.cash / (1 + self.fee_rate))
        if notional <= 0:
            return Fill(False, "buy", error="insufficient cash")
        px = price * (1 + self.slippage)
        qty = notional / px
        fee = notional * self.fee_rate
        self.cash -= notional + fee
        self.fees_paid += fee
        prev = self.qty.get(symbol, 0.0)
        self.entry[symbol] = (self.entry.get(symbol, px) * prev + px * qty) / (prev + qty)
        self.qty[symbol] = prev + qty
        return Fill(True, "buy", qty, px, fee, notional)

    def sell(self, symbol: str, qty: float, price: float) -> Fill:
        held = self.qty.get(symbol, 0.0)
        qty = min(qty, held)  # spot: never short
        if qty <= 0:
            return Fill(False, "sell", error="nothing to sell")
        px = price * (1 - self.slippage)
        notional = qty * px
        fee = notional * self.fee_rate
        self.cash += notional - fee
        self.fees_paid += fee
        remaining = held - qty
        if remaining * px < 1e-6:
            self.qty.pop(symbol, None)
            self.entry.pop(symbol, None)
        else:
            self.qty[symbol] = remaining
        return Fill(True, "sell", qty, px, fee, notional)


# okx

OKX_PUBLIC = "https://www.okx.com"


def okx_env_overrides() -> dict[str, str]:
    mapping = {"api_key": "OKX_API_KEY", "api_secret": "OKX_API_SECRET", "passphrase": "OKX_PASSPHRASE"}
    return {k: os.environ[v] for k, v in mapping.items() if os.environ.get(v)}


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(math.floor(value / step + 1e-9) * step, decimals)


@dataclass(frozen=True)
class InstrumentRules:
    lot_size: float
    min_size: float


class OKXBroker:

    def __init__(self, profile_id: str | None = None, *, session_id: str = "jev-trader") -> None:
        self.profile_id = profile_id or os.environ.get("JEV_TRADER_OKX_PROFILE", "okx-paper-trade")
        self.session_id = session_id
        self._rules: dict[str, InstrumentRules] = {}

    @property
    def is_live(self) -> bool:
        return "live" in self.profile_id

    def _svc(self):
        from src.trading import service

        return service

    def configured(self) -> bool:
        return len(okx_env_overrides()) == 3

    def rules(self, symbol: str) -> InstrumentRules:
        if symbol not in self._rules:
            resp = requests.get(
                f"{OKX_PUBLIC}/api/v5/public/instruments",
                params={"instType": "SPOT", "instId": symbol},
                timeout=10,
            )
            resp.raise_for_status()
            data = (resp.json().get("data") or [{}])[0]
            self._rules[symbol] = InstrumentRules(float(data.get("lotSz") or 0), float(data.get("minSz") or 0))
        return self._rules[symbol]

    def balances(self) -> dict[str, dict[str, float]]:
        # {ccy: {equity, available}}, raises if okx returns an error
        snap = self._svc().get_account(self.profile_id, **okx_env_overrides())
        if str(snap.get("status")) != "ok":
            raise RuntimeError(f"okx account read failed: {snap.get('error')}")
        out: dict[str, dict[str, float]] = {}
        for row in (snap.get("account") or {}).get("details") or []:
            ccy = str(row.get("currency") or "").upper()
            if ccy:
                out[ccy] = {
                    "equity": float(row.get("equity") or 0.0),
                    "available": float(row.get("available") or 0.0),
                }
        return out

    def base_available(self, symbol: str) -> float:
        base = symbol.split("-")[0].upper()
        return self.balances().get(base, {}).get("available", 0.0)

    def buy(self, symbol: str, notional: float, price: float) -> Fill:
        notional = round(float(notional), 2)
        result = self._svc().place_order(
            symbol,
            self.profile_id,
            side="buy",
            notional=notional,
            order_type="market",
            session_id=self.session_id,
            **okx_env_overrides(),
        )
        return self._fill("buy", result, notional=notional, price=price)

    def sell(self, symbol: str, qty: float, price: float) -> Fill:
        rules = self.rules(symbol)
        available = self.base_available(symbol)
        size = floor_to_step(min(qty, available), rules.lot_size)
        if size <= 0 or size < rules.min_size:
            return Fill(False, "sell", error=f"dust/flat: available={available} minSz={rules.min_size}")
        result = self._svc().place_order(
            symbol,
            self.profile_id,
            side="sell",
            quantity=size,
            order_type="market",
            session_id=self.session_id,
            **okx_env_overrides(),
        )
        return self._fill("sell", result, qty=size, price=price)

    @staticmethod
    def _fill(side: str, result: dict[str, Any], *, notional: float = 0.0, qty: float = 0.0, price: float = 0.0) -> Fill:
        ok = isinstance(result, dict) and str(result.get("status", "")).lower() == "ok"
        err = None if ok else str((result or {}).get("error") or (result or {}).get("reason") or result)[:300]
        est_qty = qty or (notional / price if price else 0.0)
        return Fill(ok, side, est_qty, price, 0.0, notional or qty * price, err, dict(result or {}))
