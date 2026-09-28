"""tests for jev client/gate + the trader (policy, parsing, brokers, sim)"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.jev.client import JevClient, JevError, boolean, choice, distribution_confidence, score
from src.jev.gate import GateThresholds, JevEntryGate
from src.jev_trader.brain import Proposal, RulesBrain, parse_proposal
from src.jev_trader.brokers import OKXBroker, SimBroker, floor_to_step
from src.jev_trader.features import MIN_BARS, compute_features
from src.jev_trader.policy import AssetState, RiskConfig, decide
from src.jev_trader.simulate import run_simulation


# fakes


class FakeResp:
    def __init__(self, status: int, payload=None, text: str = "", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


class RecordingPost:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


GOOD_GATEWAY_ANSWERS = {
    "regime": {"type": "choice", "choice": "strong_uptrend", "probabilities": {"strong_uptrend": 0.9, "range": 0.1}},
    "thesis_coherent": {"type": "boolean", "probability": 0.9},
    "setup_quality": {"type": "score", "score": 3.2, "probabilities": {"3": 0.8, "4": 0.2}},
    "risk_flag": {"type": "boolean", "probability": 0.1},
}


class StubClient:

    def __init__(self, answers=None, exc: Exception | None = None):
        self.answers = answers
        self.exc = exc
        self.calls = 0

    def evaluate(self, state, questions):
        self.calls += 1
        if self.exc:
            raise self.exc
        post = RecordingPost(FakeResp(200, {"answers": self.answers, "usage": {"inputTokens": 500}}))
        return JevClient(transport="gateway", api_key="k", post=post).evaluate(state, questions)


class ExplodingGate:
    # blows up if anything calls it

    def review_entry(self, *a, **k):
        raise AssertionError("gate must not be called for this path")


def buy(conv=0.8, size=100.0):
    return Proposal("buy", conv, size, "uptrend", "below sma50", "test")


def sell(size=0.0):
    return Proposal("sell", 0.8, size, "trend broken", source="test")


FEATURES = {"rsi14": 55.0}


# client


def test_gateway_request_shape_and_normalization():
    post = RecordingPost(FakeResp(200, {"answers": GOOD_GATEWAY_ANSWERS, "usage": {"inputTokens": 1000}}))
    client = JevClient(transport="gateway", api_key="secret", post=post)
    res = client.evaluate("state", JevEntryGate.questions())
    call = post.calls[0]
    assert call["url"] == "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
    assert call["headers"]["ai-model-id"] == "typesafe-ai/jev"
    assert call["headers"]["ai-evaluation-model-specification-version"] == "4"
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["json"]["questions"]["risk_flag"]["type"] == "boolean"
    assert "model" not in call["json"]
    assert res.answers["regime"].choice == "strong_uptrend"
    assert 0 < res.answers["regime"].confidence < 1
    assert res.answers["risk_flag"].probability == 0.1
    assert res.cost_usd == pytest.approx(1000 * 0.042 / 1e6)


def test_typesafe_transport_translates_boolean_to_noul():
    payload = {"answers": {"q": {"type": "noul", "noul": 0.8}}, "model": "jev-1.13.0"}
    post = RecordingPost(FakeResp(200, payload))
    client = JevClient(transport="typesafe", api_key="k", post=post)
    res = client.evaluate({"x": 1}, {"q": boolean("true?")})
    call = post.calls[0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["json"]["questions"]["q"]["type"] == "noul"
    assert call["json"]["model"] == "jev-latest"
    assert res.answers["q"].probability == 0.8 and res.model == "jev-1.13.0"


def test_client_fails_without_key_on_auth_error_and_on_partial_answers():
    with pytest.raises(JevError, match="not configured"):
        JevClient(transport="gateway", api_key="").evaluate("s", {"q": boolean("x")})
    post = RecordingPost(FakeResp(401, text="bad key"))
    with pytest.raises(JevError, match="401"):
        JevClient(transport="gateway", api_key="k", post=post, max_retries=3).evaluate("s", {"q": boolean("x")})
    assert len(post.calls) == 1  # 401 is not retried
    post = RecordingPost(FakeResp(200, {"answers": {}}))
    with pytest.raises(JevError, match="unparseable"):
        JevClient(transport="gateway", api_key="k", post=post).evaluate("s", {"q": boolean("x")})


def test_client_retries_overload(monkeypatch):
    monkeypatch.setattr("src.jev.client.time.sleep", lambda s: None)
    post = RecordingPost(FakeResp(529), FakeResp(200, {"answers": {"q": {"type": "boolean", "probability": 0.3}}}))
    res = JevClient(transport="gateway", api_key="k", post=post, max_retries=2).evaluate("s", {"q": boolean("x")})
    assert res.answers["q"].probability == 0.3 and len(post.calls) == 2


def test_question_builders_and_confidence():
    assert choice("c", {"a": None})["criteria"] == {"a": None}
    with pytest.raises(ValueError):
        score("s", ["only one"])
    assert distribution_confidence({"a": 1.0, "b": 0.0}) == pytest.approx(1.0)
    assert distribution_confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)


# gate


def test_gate_approves_with_multiplier_at_most_one():
    gate = JevEntryGate(StubClient(GOOD_GATEWAY_ANSWERS), GateThresholds())
    d = gate.review_entry(FEATURES, buy().to_dict(), {})
    assert d.allowed and 0.25 <= d.size_multiplier <= 1.0
    assert d.cost_usd > 0


@pytest.mark.parametrize(
    "patch,reason",
    [
        ({"regime": {"type": "choice", "choice": "disorderly", "probabilities": {"disorderly": 0.9, "range": 0.1}}}, "regime"),
        ({"thesis_coherent": {"type": "boolean", "probability": 0.2}}, "incoherent"),
        ({"risk_flag": {"type": "boolean", "probability": 0.9}}, "risk flag"),
        ({"setup_quality": {"type": "score", "score": 0.5, "probabilities": {"0": 0.5, "1": 0.5}}}, "weak setup"),
    ],
)
def test_gate_vetoes_on_thresholds(patch, reason):
    gate = JevEntryGate(StubClient({**GOOD_GATEWAY_ANSWERS, **patch}), GateThresholds())
    d = gate.review_entry(FEATURES, buy().to_dict(), {})
    assert not d.allowed and d.size_multiplier == 0.0
    assert any(reason in r for r in d.reasons)


def test_gate_fails_closed_on_jev_error_and_unexpected_exception():
    for exc in (JevError("timeout"), RuntimeError("boom")):
        d = JevEntryGate(StubClient(exc=exc)).review_entry(FEATURES, buy().to_dict(), {})
        assert not d.allowed and d.error


def test_gate_refuses_to_review_non_buy():
    client = StubClient(GOOD_GATEWAY_ANSWERS)
    d = JevEntryGate(client).review_entry(FEATURES, sell().to_dict(), {})
    assert not d.allowed and client.calls == 0


# policy

RISK = RiskConfig(max_alloc_pct=30.0, min_buy_conviction=0.55, stop_loss_pct=5.0, max_trades_per_day=5, min_trade_usd=10.0)


def test_sells_bypass_gate_and_never_exceed_holdings():
    state = AssetState("BTC-USDT", price=100.0, qty=20.0, entry_price=100.0, equity=10_000.0)
    out = decide(sell(0.0), state, FEATURES, RISK, ExplodingGate())
    assert out.order.side == "sell" and out.order.qty == 20.0
    out = decide(sell(50.0), state, FEATURES, RISK, ExplodingGate())  # trim to 50% of 3000 cap
    assert out.order.qty == pytest.approx(5.0) and out.order.qty <= state.qty


def test_stop_loss_is_forced_and_bypasses_gate():
    state = AssetState("BTC-USDT", price=94.0, qty=10.0, entry_price=100.0, equity=10_000.0)
    out = decide(buy(), state, FEATURES, RISK, ExplodingGate())
    assert out.order.side == "sell" and out.order.forced and out.order.qty == 10.0


def test_jev_outage_blocks_entries_but_exits_still_work():
    gate = JevEntryGate(StubClient(exc=JevError("down")))
    flat = AssetState("BTC-USDT", price=100.0, qty=0.0, entry_price=None, equity=10_000.0)
    out = decide(buy(), flat, FEATURES, RISK, gate)
    assert out.order is None and out.vetoed_notional == pytest.approx(3000.0)
    held = AssetState("BTC-USDT", price=100.0, qty=20.0, entry_price=100.0, equity=10_000.0)
    assert decide(sell(0.0), held, FEATURES, RISK, gate).order.side == "sell"


def test_jev_can_only_shrink_buys():
    flat = AssetState("BTC-USDT", price=100.0, qty=0.0, entry_price=None, equity=10_000.0)
    ungated = decide(buy(), flat, FEATURES, RISK, None).order.notional_usd
    gated = decide(buy(), flat, FEATURES, RISK, JevEntryGate(StubClient(GOOD_GATEWAY_ANSWERS))).order.notional_usd
    assert ungated == pytest.approx(3000.0) and 0 < gated <= ungated


def test_buy_caps_conviction_and_daily_limit():
    flat = AssetState("BTC-USDT", price=100.0, qty=0.0, entry_price=None, equity=10_000.0)
    assert decide(buy(conv=0.3), flat, FEATURES, RISK, None).order is None
    capped = AssetState("BTC-USDT", 100.0, 0.0, None, 10_000.0, trades_today=5)
    assert decide(buy(), capped, FEATURES, RISK, ExplodingGate()).order is None
    full = AssetState("BTC-USDT", 100.0, 30.0, 100.0, 10_000.0)
    assert decide(buy(), full, FEATURES, RISK, ExplodingGate()).order is None


# brain


def test_parse_proposal_strict():
    ok = parse_proposal('```json\n{"action":"BUY","conviction":0.7,"size_pct":60,"thesis":"t","invalidation":"i"}\n```', source="x")
    assert ok.action == "buy" and ok.size_pct == 60 and ok.parse_error is None
    for bad in ("no json here", '{"action":"short","conviction":0.5,"size_pct":10}', '{"action":"buy","conviction":3,"size_pct":10}'):
        p = parse_proposal(bad, source="x")
        assert p.action == "hold" and p.parse_error


# brokers


def test_sim_broker_fees_and_no_shorting():
    b = SimBroker(1000.0, fee_rate=0.001, slippage_bps=0)
    f = b.buy("X", 500.0, 10.0)
    assert f.ok and b.qty["X"] == pytest.approx(50.0) and b.cash == pytest.approx(499.5)
    f = b.sell("X", 999.0, 10.0)
    assert f.qty == pytest.approx(50.0) and "X" not in b.qty
    assert not b.sell("X", 1.0, 10.0).ok


def test_floor_to_step():
    assert floor_to_step(0.123456, 0.0001) == 0.1234
    assert floor_to_step(5.9, 1) == 5


def test_okx_sell_caps_at_available_and_rounds_down(monkeypatch):
    placed = {}

    class FakeService:
        @staticmethod
        def get_account(profile_id, **kw):
            return {"status": "ok", "account": {"details": [{"currency": "BTC", "equity": "0.0199", "available": "0.01987"}]}}

        @staticmethod
        def place_order(symbol, profile_id, **kw):
            placed.update(kw, symbol=symbol, profile=profile_id)
            return {"status": "ok", "order_id": "1"}

    broker = OKXBroker("okx-paper-trade")
    monkeypatch.setattr(broker, "_svc", lambda: FakeService)
    monkeypatch.setattr(broker, "rules", lambda s: type("R", (), {"lot_size": 0.0001, "min_size": 0.0001})())
    fill = broker.sell("BTC-USDT", 0.02, 80_000.0)  # bought 0.02, fee took some base
    assert fill.ok and placed["quantity"] == 0.0198 and placed["profile"] == "okx-paper-trade"
    monkeypatch.setattr(broker, "rules", lambda s: type("R", (), {"lot_size": 0.0001, "min_size": 0.1})())
    assert not broker.sell("BTC-USDT", 0.02, 80_000.0).ok  # dust, nothing to sell


# features / sim


def _bars(n=160, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
         "close": close, "volume": rng.uniform(100, 200, n)},
        index=idx,
    )


def test_features_anonymized_and_need_history():
    f = compute_features(_bars())
    assert all(isinstance(v, (int, float, bool)) for v in f.values())
    assert not any("price" == k or "symbol" in k or "date" in k for k in f)
    with pytest.raises(ValueError):
        compute_features(_bars(MIN_BARS - 1))


class RecordingBrain:
    name = "rec"

    def __init__(self):
        self.seen = []
        self.inner = RulesBrain()

    def propose(self, features, position):
        self.seen.append(dict(features))
        return self.inner.propose(features, position)


def test_simulator_has_no_lookahead():
    # mess up everything after t, nothing up to t should change
    bars = _bars(200)
    corrupted = bars.copy()
    cut = 170
    corrupted.iloc[cut + 1:, corrupted.columns.get_indexer(["high", "low", "close", "volume"])] *= 3.0
    # except open[cut+1], that's where the order at cut fills
    corrupted.iloc[cut + 2:, corrupted.columns.get_indexer(["open"])] *= 3.0

    runs = []
    for frame in (bars, corrupted):
        brain = RecordingBrain()
        run_simulation(symbols=["X"], start="", end="", arms=["rules"], brain_factory=lambda a: brain,
                       bars={"X": frame}, risk=RISK, progress=None)
        runs.append(brain.seen)
    decisions_through_cut = cut - (MIN_BARS - 1) + 1
    assert runs[0][:decisions_through_cut] == runs[1][:decisions_through_cut]
    assert runs[0][decisions_through_cut:] != runs[1][decisions_through_cut:]


def test_simulator_reports_all_arms_and_veto_counterfactuals():
    bars = {"X": _bars(220, seed=3)}

    class AlwaysBuy:
        name = "always"

        def propose(self, features, position):
            return Proposal("buy", 0.9, 100.0, "t", source="always")

    veto_gate = JevEntryGate(StubClient(exc=JevError("down")))
    report = run_simulation(symbols=["X"], start="", end="", arms=["buyhold", "rules", "brain", "brain+jev"],
                            brain_factory=lambda a: RulesBrain() if a == "rules" else AlwaysBuy(), gate=veto_gate,
                            bars=bars, risk=RISK, progress=None)
    arms = {a["arm"]: a for a in report["arms"]}
    assert set(arms) == {"buyhold", "rules", "brain", "brain+jev"}
    assert arms["brain+jev"]["trades"] == 0 and arms["brain+jev"]["vetoes"] > 0
    assert arms["brain"]["trades"] > 0
    assert all(not math.isnan(a["total_return_pct"]) for a in arms.values())


def test_llm_brain_caches_and_holds_on_failure(tmp_path):
    from src.jev_trader.brain import LLMBrain

    calls = []

    def chat(messages):
        calls.append(messages)
        return '{"action":"buy","conviction":0.8,"size_pct":50,"thesis":"rsi ok","invalidation":"x"}'

    brain = LLMBrain(cache_dir=tmp_path, chat=chat)
    first = brain.propose({"rsi14": 50.0}, {"in_position": False})
    second = brain.propose({"rsi14": 50.0}, {"in_position": False})
    assert first.action == second.action == "buy" and len(calls) == 1
    assert "symbol" not in calls[0][1]["content"].lower()

    def broken(messages):
        raise TimeoutError("provider down")

    p = LLMBrain(cache_dir=None, chat=broken).propose({"rsi14": 50.0}, {})
    assert p.action == "hold" and "TimeoutError" in p.parse_error


class FakeOKX:
    profile_id = "okx-paper-trade"

    def __init__(self, usdt=10_000.0, btc=0.0):
        self.bal = {"USDT": {"equity": usdt, "available": usdt}, "BTC": {"equity": btc, "available": btc}}
        self.orders = []

    def balances(self):
        return self.bal

    def buy(self, symbol, notional, price):
        from src.jev_trader.brokers import Fill

        self.orders.append(("buy", symbol, notional))
        return Fill(True, "buy", notional / price, price, 0.0, notional)

    def sell(self, symbol, qty, price):
        from src.jev_trader.brokers import Fill

        self.orders.append(("sell", symbol, qty))
        return Fill(True, "sell", qty, price, 0.0, qty * price)


def _trader(tmp_path, monkeypatch, broker, brain, halted=False):
    from src.jev_trader import loop

    monkeypatch.setattr(loop, "closed_bars", lambda s, i: _bars(160))
    monkeypatch.setattr("src.live.halt.halt_flag_set", lambda b=None: halted)
    tripped = []
    monkeypatch.setattr("src.live.halt.trip_halt", lambda by, reason, broker=None: tripped.append(reason))
    t = loop.Trader(symbols=["BTC-USDT"], brain=brain, gate=None, broker=broker, risk=RISK, interval="1H", state_dir=tmp_path)
    return t, tripped


class AlwaysBuyBrain:
    name = "always"

    def propose(self, features, position):
        return Proposal("buy", 0.9, 100.0, "t", source="always")


def test_loop_kill_switch_blocks_entries(tmp_path, monkeypatch):
    broker = FakeOKX()
    t, _ = _trader(tmp_path, monkeypatch, broker, AlwaysBuyBrain(), halted=True)
    t.run_once()
    assert broker.orders == []
    assert "halted" in (tmp_path / "journal.jsonl").read_text()


def test_loop_places_capped_buy_and_journals(tmp_path, monkeypatch):
    broker = FakeOKX()
    t, _ = _trader(tmp_path, monkeypatch, broker, AlwaysBuyBrain())
    recs = t.run_once()
    assert broker.orders == [("buy", "BTC-USDT", pytest.approx(3000.0))]  # 30% cap
    assert recs[0]["fill"]["ok"] and (tmp_path / "state.json").exists()


def test_loop_drawdown_flattens_bot_position_then_halts(tmp_path, monkeypatch):
    broker = FakeOKX(usdt=5_000.0, btc=50.0)
    t, tripped = _trader(tmp_path, monkeypatch, broker, AlwaysBuyBrain())
    t.state.peak_equity = 10_000.0
    t.state.bot_qty = {"BTC-USDT": 20.0}
    assert t.run_once() == []
    assert broker.orders == [("sell", "BTC-USDT", pytest.approx(20.0))] and tripped  # only the bot's 20, not the account's 50


def test_loop_drawdown_halts_even_if_sell_raises(tmp_path, monkeypatch):
    broker = FakeOKX(usdt=5_000.0, btc=1.0)
    broker.sell = lambda *a: (_ for _ in ()).throw(RuntimeError("okx down"))
    t, tripped = _trader(tmp_path, monkeypatch, broker, AlwaysBuyBrain())
    t.state.peak_equity, t.state.bot_qty = 10_000.0, {"BTC-USDT": 0.5}
    t.run_once()
    assert tripped


def test_loop_ignores_coins_it_did_not_buy(tmp_path, monkeypatch):
    class AlwaysSell:
        name = "sell"

        def propose(self, features, position):
            assert position["in_position"] is False
            return Proposal("sell", 0.9, 0.0, "t", source="sell")

    broker = FakeOKX(btc=2.0)  # user's own BTC / demo pre-funding
    t, _ = _trader(tmp_path, monkeypatch, broker, AlwaysSell())
    t.run_once()
    assert broker.orders == []


def test_loop_halt_still_runs_stop_loss_but_no_entries(tmp_path, monkeypatch):
    class NeverCalled:
        name = "x"

        def propose(self, *a):
            raise AssertionError("brain must not run while halted")

    broker = FakeOKX(btc=1.0)
    t, _ = _trader(tmp_path, monkeypatch, broker, NeverCalled(), halted=True)
    last = float(_bars(160)["close"].iloc[-1])
    t.state.bot_qty, t.state.entries = {"BTC-USDT": 1.0}, {"BTC-USDT": last * 1.2}  # 17% under entry
    recs = t.run_once()
    assert broker.orders == [("sell", "BTC-USDT", pytest.approx(1.0))] and recs[0]["policy"] == "stop-loss"
    assert "BTC-USDT" not in t.state.bot_qty


def test_ambiguous_full_size_sell_is_a_full_exit():
    state = AssetState("BTC-USDT", price=100.0, qty=30.0, entry_price=100.0, equity=10_000.0)  # at 100% of cap
    out = decide(sell(100.0), state, FEATURES, RISK, ExplodingGate())
    assert out.order is not None and out.order.qty == 30.0


def test_llm_cache_is_keyed_by_model(tmp_path, monkeypatch):
    from src.jev_trader.brain import LLMBrain

    calls = []
    chat = lambda m: calls.append(m) or '{"action":"hold","conviction":0.1,"size_pct":0,"thesis":"t"}'  # noqa: E731
    monkeypatch.setenv("LANGCHAIN_MODEL_NAME", "deepseek/deepseek-v4-flash-0731")
    LLMBrain(cache_dir=tmp_path, chat=chat).propose({"rsi14": 50.0}, {})
    monkeypatch.setenv("LANGCHAIN_MODEL_NAME", "openai/gpt-5.6-luna")
    LLMBrain(cache_dir=tmp_path, chat=chat).propose({"rsi14": 50.0}, {})
    assert len(calls) == 2
