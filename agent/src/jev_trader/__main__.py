"""python -m src.jev_trader {check,simulate,run}   (from agent/)

    # what's configured. --ping does one real LLM + Jev call
    python -m src.jev_trader check --ping

    # no keys needed, just public OKX candles
    python -m src.jev_trader simulate --arms buyhold,rules --days 30

    # full comparison, needs OPENROUTER_API_KEY + AI_GATEWAY_API_KEY
    python -m src.jev_trader simulate --arms buyhold,rules,brain,brain+jev --days 30

    # one cycle on OKX demo / keep running
    python -m src.jev_trader run --once
    python -m src.jev_trader run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_AGENT_DIR / ".env", override=False)


def _state_dir() -> Path:
    from src.config.paths import get_runtime_root

    return get_runtime_root() / "jev_trader"


def _symbols(raw: str) -> list[str]:
    return [s.strip().upper().replace("/", "-") for s in raw.split(",") if s.strip()]


def _make_brain(kind: str, cache: bool = True):
    from src.jev_trader.brain import LLMBrain, RulesBrain

    if kind == "rules":
        return RulesBrain()
    return LLMBrain(cache_dir=_state_dir() / "llm_cache" if cache else None)


def _make_gate():
    from src.jev.gate import JevEntryGate

    return JevEntryGate()


def cmd_check(args: argparse.Namespace) -> int:
    from src.jev.client import JevClient
    from src.jev_trader.brokers import OKXBroker, okx_env_overrides

    jev = JevClient()
    llm_provider = os.environ.get("LANGCHAIN_PROVIDER", "")
    llm_model = os.environ.get("LANGCHAIN_MODEL_NAME", "")
    broker = OKXBroker()
    status = {
        "llm": {"provider": llm_provider, "model": llm_model, "openrouter_key": bool(os.environ.get("OPENROUTER_API_KEY"))},
        "jev": {"transport": jev.transport, "model": jev.model, "base_url": jev.base_url, "key": jev.configured},
        "okx": {"profile": broker.profile_id, "live": broker.is_live, "keys": sorted(okx_env_overrides())},
        "state_dir": str(_state_dir()),
    }
    if args.ping:
        from src.jev.gate import JevEntryGate
        from src.jev_trader.brain import LLMBrain

        sample = {"rsi14": 55.0, "dist_sma50_pct": 2.1, "sma20_above_sma50": True, "return_24h_pct": 1.4,
                  "atr14_pct_of_price": 0.8, "sma20_slope_5bar_pct": 0.2}
        try:
            p = LLMBrain(cache_dir=None).propose(sample, {"in_position": False})
            status["llm"]["ping"] = p.to_dict()
        except Exception as exc:  # noqa: BLE001
            status["llm"]["ping_error"] = str(exc)
        g = JevEntryGate().review_entry(sample, {"action": "buy", "conviction": 0.7, "thesis": "Uptrend: price above rising SMA50, RSI 55."}, {"in_position": False})
        status["jev"]["ping"] = g.to_dict()
        if broker.configured():
            try:
                status["okx"]["balances"] = broker.balances()
            except Exception as exc:  # noqa: BLE001
                status["okx"]["error"] = str(exc)
    print(json.dumps(status, indent=2, default=str))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from src.jev_trader.simulate import format_report, run_simulation

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    end = datetime.now(timezone.utc).date() if not args.end else datetime.fromisoformat(args.end).date()
    start = end - timedelta(days=args.days) if not args.start else datetime.fromisoformat(args.start).date()
    gate = _make_gate() if "brain+jev" in arms else None
    # if the model isn't reachable every bar is a "hold" and the report is
    # garbage, so bail out early instead
    if gate is not None and not gate.client.configured:
        print("brain+jev arm needs Jev credentials (AI_GATEWAY_API_KEY or TYPESAFE_API_KEY).", file=sys.stderr)
        return 2
    if {"brain", "brain+jev"} & set(arms):
        probe = _make_brain("llm", cache=False).propose({"rsi14": 50.0, "dist_sma50_pct": 0.0}, {"in_position": False})
        if probe.thesis == "llm unavailable":
            print(f"LLM brain unreachable: {probe.parse_error}", file=sys.stderr)
            return 2
    report = run_simulation(
        symbols=_symbols(args.symbols),
        start=start.isoformat(),
        end=end.isoformat(),
        interval=args.interval,
        arms=arms,
        brain_factory=lambda arm: _make_brain("rules" if arm == "rules" else "llm", cache=not args.no_cache),
        gate=gate,
        cash=args.cash,
        fee=args.fee,
        slippage_bps=args.slippage_bps,
        decide_every=args.decide_every,
        out_dir=_state_dir() / "sims",
    )
    print(format_report(report))
    print(f"\nFull report: {report.get('report_path')}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from src.jev_trader.brokers import OKXBroker
    from src.jev_trader.loop import Trader
    from src.jev_trader.policy import RiskConfig

    broker = OKXBroker(args.profile)
    if not broker.configured():
        print("OKX keys missing: set OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE (demo keys for okx-paper-trade).", file=sys.stderr)
        return 2
    if broker.is_live and not args.i_understand_live:
        print("Refusing live profile without --i-understand-live (and a committed mandate).", file=sys.stderr)
        return 2
    trader = Trader(
        symbols=_symbols(args.symbols),
        brain=_make_brain(args.brain),
        gate=None if args.no_jev else _make_gate(),
        broker=broker,
        risk=RiskConfig.from_env(),
        interval=args.interval,
        state_dir=_state_dir() / broker.profile_id,
    )
    if args.once:
        for rec in trader.run_once():
            print(json.dumps({k: rec.get(k) for k in ("symbol", "policy", "proposal", "gate", "fill")}, default=str))
        return 0
    trader.run_forever()
    return 0


def cmd_halt(args: argparse.Namespace) -> int:
    from src.live.halt import clear_halt, halt_flag_set, trip_halt

    if args.cmd == "halt":
        trip_halt("jev-trader-cli", args.reason, "okx")
    else:
        clear_halt("okx")
    print(json.dumps({"okx_halted": halt_flag_set("okx")}))
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    logging.basicConfig(level=os.environ.get("JEV_TRADER_LOG_LEVEL", "WARNING"))
    parser = argparse.ArgumentParser(prog="jev_trader", description="Jev-gated LLM crypto spot trader")
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_symbols = os.environ.get("JEV_TRADER_SYMBOLS", "BTC-USDT,ETH-USDT,SOL-USDT")
    default_interval = os.environ.get("JEV_TRADER_INTERVAL", "1H")

    p = sub.add_parser("check", help="show configuration status")
    p.add_argument("--ping", action="store_true", help="make one real LLM and Jev call")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("simulate", help="historical replay comparing arms")
    p.add_argument("--symbols", default=default_symbols)
    p.add_argument("--interval", default=default_interval, choices=["15m", "30m", "1H", "4H"])
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--arms", default="buyhold,rules,brain,brain+jev")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--decide-every", type=int, default=1, help="call the brain every N bars")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("run", help="paper/live loop on OKX spot")
    p.add_argument("--symbols", default=default_symbols)
    p.add_argument("--interval", default=default_interval, choices=["15m", "30m", "1H", "4H"])
    p.add_argument("--profile", default=None, help="okx-paper-trade (default) or okx-live-trade")
    p.add_argument("--brain", default="llm", choices=["llm", "rules"])
    p.add_argument("--no-jev", action="store_true", help="disable the Jev entry gate")
    p.add_argument("--once", action="store_true")
    p.add_argument("--i-understand-live", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("halt", help="trip the OKX kill switch (loop stops placing orders)")
    p.add_argument("--reason", default="manual halt")
    p.set_defaults(fn=cmd_halt)
    p = sub.add_parser("resume", help="clear the OKX kill switch")
    p.set_defaults(fn=cmd_halt)

    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
