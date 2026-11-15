"""因子库：从日线序列算出可检验的技术/量价因子。

设计原则（这是整个系统能不能站住脚的关键）：
1. **所有因子只用 t 时刻及之前的数据**，严格禁止未来函数。
   每个因子的输入是 `bars[:i+1]`，输出是第 i 根 K 线收盘时能算出的值。
2. **因子必须可证伪**。每个因子都能算出它在历史上的横截面 IC
   （与未来收益的相关性）。IC 接近 0 的因子不进模型。
3. **不硬编码结论**。因子的"有效性"由 backtest.py 用真实数据检验，
   这里只负责把原始序列变成特征。

因子方向约定：
    因子值越大代表"越看多"。反向因子（如短期反转）由 BACKTEST
    里的 sign 参数处理，不在这里偷偷取负。
"""
from __future__ import annotations

import math
from typing import Callable

# ------------------------------------------------------------------ 基础工具


def _safe(x: float | None, default: float = 0.0) -> float:
    if x is None or not math.isfinite(x):
        return default
    return x


def pct_change(a: float, b: float) -> float:
    """(a/b - 1)，b<=0 时返回 0（避免除零炸出 inf）。"""
    if not b:
        return 0.0
    return a / b - 1.0


def sma(xs: list[float], n: int) -> float | None:
    if len(xs) < n or n <= 0:
        return None
    return sum(xs[-n:]) / n


def stdev(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(var, 0.0))


def rsi(closes: list[float], n: int = 14) -> float | None:
    """Wilder RSI。只用最近 n+1 根。

    Wilder 平滑：avg = (prev_avg * (n-1) + current) / n
    种子用前 n 个变动的简单平均，之后逐根递推。
    与快版 fastfactors._rsi 保持一致（业界标准口径）。
    """
    if len(closes) < n + 1:
        return None
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / n
    avg_loss = losses / n
    # Wilder 递推到最新一根
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (n - 1) + g) / n
        avg_loss = (avg_loss * (n - 1) + l) / n
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def atr(bars: list[dict], n: int = 14) -> float | None:
    """平均真实波幅（用 close 做前收）。"""
    if len(bars) < n + 1:
        return None
    trs: list[float] = []
    for i in range(len(bars) - n, len(bars)):
        b, prev = bars[i], bars[i - 1]
        trs.append(max(b["high"] - b["low"], abs(b["high"] - prev["close"]), abs(b["low"] - prev["close"])))
    return sum(trs) / n


def max_drawdown(xs: list[float]) -> float:
    """区间最大回撤（正数表示回撤幅度）。"""
    peak = xs[0] if xs else 0.0
    mdd = 0.0
    for x in xs:
        if x > peak:
            peak = x
        if peak > 0:
            mdd = max(mdd, (peak - x) / peak)
    return mdd


# ------------------------------------------------------------------ 因子定义
# 每个因子： (bars) -> float | None
# 输入是截至当前 K 线的完整历史切片，输出是该时刻可得的因子原始值。


def f_mom_20(bars: list[dict]) -> float | None:
    """20 日动量（跳过最近 1 日，避免当日反转污染）。"""
    if len(bars) < 22:
        return None
    return pct_change(bars[-2]["close"], bars[-22]["close"])


def f_mom_60(bars: list[dict]) -> float | None:
    """60 日动量。"""
    if len(bars) < 62:
        return None
    return pct_change(bars[-2]["close"], bars[-62]["close"])


def f_mom_120(bars: list[dict]) -> float | None:
    """120 日动量。"""
    if len(bars) < 122:
        return None
    return pct_change(bars[-2]["close"], bars[-122]["close"])


def f_rev_5(bars: list[dict]) -> float | None:
    """5 日反转：短期跌多的可能反弹（原始值为负，越负越看多，故取负）。"""
    if len(bars) < 7:
        return None
    return -pct_change(bars[-2]["close"], bars[-7]["close"])


def f_ma_dev_20(bars: list[dict]) -> float | None:
    """偏离 20 日均线：趋势强度的标准化度量。"""
    if len(bars) < 21:
        return None
    ma = sma([b["close"] for b in bars], 20)
    if not ma:
        return None
    return pct_change(bars[-1]["close"], ma)


def f_ma_dev_60(bars: list[dict]) -> float | None:
    if len(bars) < 61:
        return None
    ma = sma([b["close"] for b in bars], 60)
    if not ma:
        return None
    return pct_change(bars[-1]["close"], ma)


def f_vol_20(bars: list[dict]) -> float | None:
    """20 日年化波动率（风险度量）。"""
    if len(bars) < 21:
        return None
    rets = []
    for i in range(len(bars) - 20, len(bars)):
        rets.append(pct_change(bars[i]["close"], bars[i - 1]["close"]))
    return stdev(rets) * math.sqrt(252)


def f_low_vol(bars: list[dict]) -> float | None:
    """低波动因子：波动越低越好（原始值为负波动）。"""
    v = f_vol_20(bars)
    return None if v is None else -v


def f_rsi_14(bars: list[dict]) -> float | None:
    """RSI 中性化：太高回撤风险大，太低可能弱势 → 用 (50-rsi)/50 居中。"""
    closes = [b["close"] for b in bars]
    r = rsi(closes, 14)
    return None if r is None else (50.0 - r) / 50.0


def f_vol_ratio(bars: list[dict]) -> float | None:
    """量比：近 5 日均量 / 近 60 日均量，衡量资金关注度。"""
    if len(bars) < 61:
        return None
    v5 = sum(b["volume"] for b in bars[-5:]) / 5.0
    v60 = sum(b["volume"] for b in bars[-60:]) / 60.0
    if not v60:
        return None
    return v5 / v60 - 1.0


def f_turnover_trend(bars: list[dict]) -> float | None:
    """成交额趋势：近 10 日均额 / 近 60 日均额。"""
    if len(bars) < 61:
        return None
    a10 = sum(b.get("amount", 0.0) for b in bars[-10:]) / 10.0
    a60 = sum(b.get("amount", 0.0) for b in bars[-60:]) / 60.0
    if not a60:
        return None
    return a10 / a60 - 1.0


def f_drawdown(bars: list[dict]) -> float | None:
    """距 60 日高点的回撤（越接近 0 越强势）。"""
    if len(bars) < 61:
        return None
    seg = [b["close"] for b in bars[-60:]]
    hi = max(seg)
    if not hi:
        return None
    return seg[-1] / hi - 1.0


def f_atr_ratio(bars: list[dict]) -> float | None:
    """ATR / 收盘价：日内波动强度（风险）。"""
    a = atr(bars, 14)
    if a is None or not bars[-1]["close"]:
        return None
    return a / bars[-1]["close"]


def f_high_low(bars: list[dict]) -> float | None:
    """区间位置：(close - min_low) / (max_high - min_low)，经典 Williams %R 变体。"""
    if len(bars) < 60:
        return None
    seg = bars[-60:]
    hi = max(b["high"] for b in seg)
    lo = min(b["low"] for b in seg)
    if hi <= lo:
        return None
    return (bars[-1]["close"] - lo) / (hi - lo)


def f_gap_bias(bars: list[dict]) -> float | None:
    """跳空偏置：近 20 日 平均 (open - prev_close)/prev_close。"""
    if len(bars) < 21:
        return None
    vals = []
    for i in range(len(bars) - 20, len(bars)):
        prev = bars[i - 1]["close"]
        if prev:
            vals.append((bars[i]["open"] - prev) / prev)
    return sum(vals) / len(vals) if vals else None


def f_updown_ratio(bars: list[dict]) -> float | None:
    """近 20 日上涨日占比 - 0.5，衡量趋势一致性。"""
    if len(bars) < 21:
        return None
    ups = 0
    for i in range(len(bars) - 20, len(bars)):
        if bars[i]["close"] > bars[i - 1]["close"]:
            ups += 1
    return ups / 20.0 - 0.5


def f_maxdd_120(bars: list[dict]) -> float | None:
    """120 日最大回撤（负值越大越好，即回撤越浅）。"""
    if len(bars) < 121:
        return None
    return -max_drawdown([b["close"] for b in bars[-120:]])


# ------------------------------------------------------------------ 因子注册表
# name -> (函数, 方向, 说明)
#   方向 +1 表示"因子原始值越大越看多"；-1 表示越小越看多。
#   注意：f_rev_5 / f_low_vol / f_maxdd_120 内部已经带了负号，
#   这里方向仍记 +1，因为它们输出的是"越大越看多"的分数。
FACTORS: dict[str, tuple[Callable[[list[dict]], float | None], int, str]] = {
    "mom_20": (f_mom_20, +1, "20日动量（跳过当日）"),
    "mom_60": (f_mom_60, +1, "60日动量"),
    "mom_120": (f_mom_120, +1, "120日动量"),
    "rev_5": (f_rev_5, +1, "5日短期反转（取负号）"),
    "ma_dev_20": (f_ma_dev_20, +1, "偏离20日均线"),
    "ma_dev_60": (f_ma_dev_60, +1, "偏离60日均线"),
    "low_vol": (f_low_vol, +1, "20日低波动（取负号）"),
    "rsi_14": (f_rsi_14, +1, "RSI 中性化"),
    "vol_ratio": (f_vol_ratio, +1, "5日/60日量比"),
    "turnover_trend": (f_turnover_trend, +1, "成交额趋势"),
    "drawdown": (f_drawdown, +1, "距60日高点回撤"),
    "atr_ratio": (f_atr_ratio, +1, "ATR/价格（日内波动）"),
    "high_low": (f_high_low, +1, "60日区间位置"),
    "gap_bias": (f_gap_bias, +1, "跳空偏置"),
    "updown_ratio": (f_updown_ratio, +1, "20日涨跌一致性"),
    "maxdd_120": (f_maxdd_120, +1, "120日最大回撤（取负号）"),
}


def compute_factors(bars: list[dict]) -> dict[str, float]:
    """算出一个时点所有因子的值（只给有效值）。"""
    out: dict[str, float] = {}
    for name, (fn, _sign, _desc) in FACTORS.items():
        try:
            v = fn(bars)
        except (IndexError, ZeroDivisionError, TypeError, ValueError):
            v = None
        if v is not None and math.isfinite(v):
            out[name] = v
    return out


def factor_descriptions() -> dict[str, str]:
    return {n: d for n, (_f, _s, d) in FACTORS.items()}
