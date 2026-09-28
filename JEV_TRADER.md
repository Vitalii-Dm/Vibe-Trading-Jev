# Jev Trader

An autonomous crypto spot trader built on Vibe-Trading. An LLM brain (DeepSeek by default) proposes trades, **Jev** (TypeSafe's System One model) gates every entry, and deterministic code enforces risk limits and executes on **OKX** (demo account first).

```
closed bar → features (anonymized) → LLM brain: buy / sell / hold
                                        │
                     sell / stop-loss ──┼──────────────► execute (never gated)
                                        │
                     buy ──► Jev gate ──► veto, or shrink size (×0.25–1.0) ──► caps ──► execute
```

## Safety invariants

- **Jev only reviews entries.** It can veto a buy or shrink it, but it can never start a trade or make one bigger.
- **Fail-closed.** If Jev is missing, times out, errors or returns partial answers, no entry is made.
- **Exits skip Jev.** Sells, the stop-loss and drawdown flattening never go through Jev, so a Jev outage can't trap you in a position. A `sell` whose target is at or above the current holding counts as a full exit.
- **Bot-owned positions.** The loop tracks only the coins it bought itself (in `state.json`). Demo pre-funding and your own coins are never sold, sized against, or counted in drawdown. Use a dedicated OKX sub-account for live.
- **Spot only.** The trader never sells more than it holds, and there's no leverage. The FCA bans crypto derivatives for UK retail.
- **Hard caps in code** (env `JEV_TRADER_*`): per-asset allocation, daily trade cap, stop-loss, max-drawdown flatten plus kill switch.
- **Paper profiles skip Vibe-Trading's live mandate gate.** So the loop checks the kill switch and applies its own caps. Live (`okx-live-trade`) still goes through the mandate gate, which needs a committed mandate, and also needs `--i-understand-live`.
- **Keys come from env** and are passed per call. They are never written to disk. Every cycle is journaled to `~/.vibe-trading/jev_trader/<profile>/journal.jsonl`.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e . python-okx
# fill in agent/.env (gitignored): OPENROUTER_API_KEY, AI_GATEWAY_API_KEY, OKX demo keys
cd agent
../.venv/bin/python -m src.jev_trader check --ping     # one real LLM + Jev call, OKX balances
```

## Simulate first

```bash
# no keys needed: buy-and-hold vs the rules baseline on public OKX candles
../.venv/bin/python -m src.jev_trader simulate --arms buyhold,rules --days 30

# the real test: does DeepSeek beat buy-and-hold, and does Jev help?
../.venv/bin/python -m src.jev_trader simulate --arms buyhold,rules,brain,brain+jev --days 30
```

- Decisions are made on the close of bar t and fill at the open of bar t+1, with a 0.1% taker fee and 5 bps slippage.
- LLM replies are cached by provider, model and prompt, so re-runs are free, and switching `LANGCHAIN_MODEL_NAME` makes fresh calls.
- A 30-day, 3-symbol run on 1H bars makes about 2k brain calls per LLM arm. That's well under $1, but the calls run one at a time and can take hours. For a first DeepSeek run, use `--days 14 --interval 4H` (~250 calls per arm) or `--decide-every 4`.
- The Jev thresholds are uncalibrated. Compare `vetoes` with `jev_calls` in the first `brain+jev` run. A veto rate near 0% or near 100% means the `JEV_MIN_*` / `JEV_MAX_*` values need tuning.
- For the gated arm, `veto_counterfactual_pnl_usd` is the P&L of the entries Jev blocked. A negative number means Jev saved money.
- Features contain no symbol, date or absolute price, to reduce the chance the model recognizes a period it memorized. Still prefer recent windows.

## Run on OKX demo

```bash
../.venv/bin/python -m src.jev_trader run --once   # one cycle, prints decisions
../.venv/bin/python -m src.jev_trader run          # loop: wakes ~20s after each bar closes
```

Useful options:
- `--no-jev` runs the brain alone for an A/B comparison.
- `--brain rules` uses the rules baseline instead of the LLM.
- **Kill switch:** `python -m src.jev_trader halt` stops the brain, Jev and all entries, and `resume` clears it.
  - **Paper:** forced stop-loss exits still run while halted.
  - **Live:** the mandate gate denies every order while halted, stop-losses included.
  - The loop flattens and then trips the switch on a max-drawdown breach, even if a sell fails.

## Costs (September 2026 prices)

| Part | Price | Typical cost |
|---|---|---|
| DeepSeek v4 flash via OpenRouter | $0.021 / $0.32 per 1M tokens (in / out) | ~$0.0002 per decision |
| Jev via Vercel AI Gateway | $0.042 per 1M input tokens, output free | ~$0.00005 per entry review |
| OKX spot | ~0.1% taker fee | This is the cost that matters |

## Code

- `agent/src/jev/client.py`: Jev client. Supports the Gateway `/v4/ai/evaluation-model` and TypeSafe `/v1/systemone` wire formats.
- `agent/src/jev/gate.py`: the four typed questions (regime, thesis coherence, setup quality, risk flag) and the thresholds.
- `agent/src/jev_trader/`: `features`, `brain`, `policy`, `brokers`, `simulate`, `loop` and the CLI in `__main__`.
- `agent/tests/test_jev_trader.py`: invariant, transport, parsing and no-lookahead tests.
