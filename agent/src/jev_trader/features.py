"""Features we show the LLM and Jev.

Only uses bars up to the last closed one. No symbol, no dates, no actual
prices on purpose - otherwise the LLM might just recognise e.g. "BTC in March
2024" and "remember" what happened next. So it's all % returns, distances
from averages, RSI etc.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# need this many bars before every feature has a value
MIN_BARS = 80


def _pct(a: float, b: float) -> float:
    return (a / b - 1.0) * 100.0 if b else 0.0


def _rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    g, l_ = float(gain.iloc[-1]), float(loss.iloc[-1])
    if l_ == 0:
        return 100.0 if g > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + g / l_)


def compute_features(bars: pd.DataFrame, bar_hours: float = 1.0) -> dict[str, Any]:
    """Features for the last row of `bars` (OHLCV, oldest first, last row =
    last closed bar). bar_hours is just for naming e.g. return_24h_pct."""
    if len(bars) < MIN_BARS:
        raise ValueError(f"need >= {MIN_BARS} bars, got {len(bars)}")
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float)
    last = float(close.iloc[-1])

    def ret(n: int) -> float:
        return round(_pct(last, float(close.iloc[-1 - n])), 2)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])
    log_ret = (close / prev_close).apply(math.log).dropna()
    bars_per_year = 24 * 365 / bar_hours
    vol_z_window = volume.iloc[-50:]
    vol_std = float(vol_z_window.std()) or 1.0
    hi72, lo72 = float(high.iloc[-72:].max()), float(low.iloc[-72:].min())

    h = lambda n: f"{int(n * bar_hours)}h"  # noqa: E731
    return {
        f"return_{h(1)}_pct": ret(1),
        f"return_{h(6)}_pct": ret(6),
        f"return_{h(24)}_pct": ret(24),
        f"return_{h(72)}_pct": ret(72),
        "dist_sma20_pct": round(_pct(last, float(sma20.iloc[-1])), 2),
        "dist_sma50_pct": round(_pct(last, float(sma50.iloc[-1])), 2),
        "sma20_slope_5bar_pct": round(_pct(float(sma20.iloc[-1]), float(sma20.iloc[-6])), 3),
        "sma20_above_sma50": bool(sma20.iloc[-1] > sma50.iloc[-1]),
        "rsi14": round(_rsi(close), 1),
        "atr14_pct_of_price": round(atr14 / last * 100.0, 3),
        "realized_vol_24bar_annualized_pct": round(float(log_ret.iloc[-24:].std()) * math.sqrt(bars_per_year) * 100.0, 1),
        "volume_zscore_vs_50bar": round((float(volume.iloc[-1]) - float(vol_z_window.mean())) / vol_std, 2),
        "drawdown_from_72bar_high_pct": round(_pct(last, hi72), 2),
        "position_in_72bar_range_0to1": round((last - lo72) / (hi72 - lo72), 3) if hi72 > lo72 else 0.5,
    }
