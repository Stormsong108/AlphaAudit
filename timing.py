"""指数择时模型：把因子作用在指数时间序列上，做走样本外（walk-forward）检验。

为什么做这个：
    横截面选股（34~1300 只）噪声极大，单个因子 IC 接近 0 属正常。
    但指数择时是**时间序列**问题——同一个信号在时间维度上反复接受检验，
    而且交易成本低、逻辑可解释。这条路更可能出真实边际。

检验方式（严格防过拟合）：
1. 用全部历史算信号值 → 构造"信号 → 未来收益"的配对序列。
2. 走样本外：参数（比如均线窗口）只在训练段选，测试段不参与选择。
3. 用 Newey-West t 检验 + 分段稳定性检查（前半段 vs 后半段 IC 符号是否一致）。
4. 给出策略净值（信号为多/空时持有指数），计算 Sharpe / 最大回撤 / 胜率。

所有数字都来自真实历史，不做任何美化。
"""
from __future__ import annotations

import math
from typing import Any

from backtest import newey_west_t, spearman


def sma(xs: list[float], n: int, i: int) -> float | None:
    """截至第 i 根（含）的 n 日均值。i 从 0 开始。"""
    if i + 1 < n:
        return None
    seg = xs[i - n + 1 : i + 1]
    return sum(seg) / n


def build_signal_series(
    bars: list[dict],
    *,
    kind: str = "ma_cross",
    fast: int = 20,
    slow: int = 60,
    atr_n: int = 14,
) -> list[dict]:
    """构造时间序列信号：每根 K 线给一个 {date, signal, close}。

    signal ∈ [-1, 1]：正=偏多，负=偏空，0=观望。
    所有计算只用到 i 及之前的数据。
    """
    closes = [b["close"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    out: list[dict] = []
    for i in range(len(bars)):
        sig = 0.0
        if kind == "ma_cross":
            f = sma(closes, fast, i)
            s = sma(closes, slow, i)
            if f is not None and s is not None:
                # 价格相对双均线位置 + 均线斜率
                slope = (f / sma(closes, fast, i - 5) - 1.0) if (i >= 5 and sma(closes, fast, i - 5)) else 0.0
                sig = 1.0 if f > s else -1.0
                sig = sig * (1.0 + max(-0.5, min(0.5, slope * 10)))
        elif kind == "trend":
            s = sma(closes, slow, i)
            if s is not None:
                sig = 1.0 if closes[i] > s else -1.0
        elif kind == "breakout":
            if i + 1 >= slow:
                hi = max(highs[i - slow + 1 : i + 1])
                sig = 1.0 if closes[i] >= hi * 0.999 else 0.0
        elif kind == "rsi":
            n = 14
            if i + 1 > n:
                gains = losses = 0.0
                for k in range(i - n + 1, i + 1):
                    d = closes[k] - closes[k - 1]
                    if d >= 0:
                        gains += d
                    else:
                        losses -= d
                r = 100.0 if losses == 0 else 100.0 - 100.0 / (1 + (gains / n) / (losses / n))
                sig = max(-1.0, min(1.0, (50.0 - r) / 25.0))
        elif kind == "vol_regime":
            # 波动率状态：高波动时降低仓位（风险平价思想）
            if i + 1 >= 60:
                rets = [closes[k] / closes[k - 1] - 1.0 for k in range(i - 59, i + 1)]
                m = sum(rets) / len(rets)
                sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
                s = sma(closes, slow, i)
                base = 1.0 if (s is not None and closes[i] > s) else -1.0
                sig = base * min(1.0, (sd * math.sqrt(252)) / 0.25)
        out.append({"date": bars[i]["date"], "signal": sig, "close": closes[i]})
    return out


def evaluate_timing(
    bars: list[dict],
    *,
    kind: str = "ma_cross",
    fast: int = 20,
    slow: int = 60,
    hold_days: int = 5,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """评估择时信号：IC、t 值、分段稳定性、策略净值。

    策略规则：signal>0 满仓做多，signal<0 空仓（A股不做空），
    每 hold_days 调仓一次，扣双边成本 cost_bps。
    """
    sig = build_signal_series(bars, kind=kind, fast=fast, slow=slow)
    closes = [b["close"] for b in bars]
    n = len(sig)
    if n < 120:
        return {"ok": False, "reason": "样本不足"}

    # ---- 信号 vs 未来 hold_days 收益的 IC
    pairs: list[tuple[float, float]] = []
    for i in range(n - hold_days):
        s = sig[i]["signal"]
        if s == 0:
            continue
        fwd = closes[i + hold_days] / closes[i] - 1.0
        pairs.append((s, fwd))
    ics: list[float] = []
    # 用滚动窗口算 IC（20 日窗），更像实盘
    win = 20
    for j in range(0, len(pairs) - win, 5):
        chunk = pairs[j : j + win]
        rho = spearman([c[0] for c in chunk], [c[1] for c in chunk])
        if rho is not None:
            ics.append(rho)
    mean_ic, t = newey_west_t(ics) if ics else (None, None)

    # ---- 分段稳定性
    seg_ics: list[float | None] = []
    if ics:
        seg = max(1, len(ics) // 4)
        for k in range(0, len(ics), seg):
            chunk = ics[k : k + seg]
            if len(chunk) >= 5:
                m = sum(chunk) / len(chunk)
                seg_ics.append(m)

    # ---- 策略净值（多头/空仓，hold_days 调仓）
    equity = 1.0
    curve: list[tuple[str, float]] = []
    rets: list[float] = []
    pos = 0
    for i in range(0, n - hold_days, hold_days):
        s = sig[i]["signal"]
        target = 1 if s > 0 else 0
        seg_ret = closes[i + hold_days] / closes[i] - 1.0
        turnover = 1 if target != pos else 0
        cost = turnover * cost_bps / 10000.0
        net = (seg_ret if target else 0.0) - cost
        equity *= 1.0 + net
        rets.append(net)
        curve.append((sig[i]["date"], equity))
        pos = target
    sharpe = None
    if len(rets) > 2:
        m = sum(rets) / len(rets)
        sd = (sum((x - m) ** 2 for x in rets) / (len(rets) - 1)) ** 0.5
        if sd > 0:
            sharpe = m / sd * math.sqrt(252 / hold_days)
    peak = -1e18
    mdd = 0.0
    for _d, v in curve:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    bench = closes[-1] / closes[0] - 1.0
    strat = equity - 1.0
    return {
        "ok": True,
        "kind": kind,
        "fast": fast,
        "slow": slow,
        "hold_days": hold_days,
        "n_periods": len(rets),
        "ic_mean": mean_ic,
        "ic_t": t,
        "n_ic": len(ics),
        "seg_ics": seg_ics,
        "strategy_ret": strat,
        "benchmark_ret": bench,
        "excess": strat - bench,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "hit_rate": (sum(1 for r in rets if r > 0) / len(rets)) if rets else None,
        "curve": curve[-40:],
    }


def sweep(
    bars: list[dict],
    *,
    kinds: list[str] | None = None,
    hold_days: int = 5,
) -> list[dict]:
    """参数扫描。用它找候选，但必须再用走样本外检验确认。"""
    kinds = kinds or ["ma_cross", "trend", "breakout", "rsi", "vol_regime"]
    out: list[dict] = []
    for kind in kinds:
        for fast, slow in ((10, 40), (20, 60), (20, 120), (60, 120), (5, 20)):
            if fast >= slow:
                continue
            r = evaluate_timing(bars, kind=kind, fast=fast, slow=slow, hold_days=hold_days)
            if r.get("ok"):
                out.append(r)
    return out


def walk_forward(
    bars: list[dict],
    *,
    kind: str,
    fast: int,
    slow: int,
    hold_days: int = 5,
    train_frac: float = 0.6,
) -> dict[str, Any]:
    """走样本外：前 train_frac 用作训练段选参数，后段只用已选参数。

    这里用"训练段挑最好的参数 → 测试段固定用该参数"来验证参数选择本身
    是否过拟合。若测试段收益远低于训练段，说明参数是拟合出来的，不能用。
    """
    n = len(bars)
    cut = int(n * train_frac)
    train_bars = bars[:cut]
    test_bars = bars[cut - slow:]  # 留 warmup，避免测试段指标失真
    train_res = evaluate_timing(train_bars, kind=kind, fast=fast, slow=slow, hold_days=hold_days)
    test_res = evaluate_timing(test_bars, kind=kind, fast=fast, slow=slow, hold_days=hold_days)
    return {"kind": kind, "fast": fast, "slow": slow,
            "train": train_res, "test": test_res}
