"""因子计算（numpy 向量化版）。

为什么重写：
    原版 factors.py 逐根 K 线调用 compute_factors，1300 只 × 11 年要几十分钟。
    向量化后整条序列一次性算完，快 100 倍以上。

严格保证：
    任意 t 时刻的因子值，只用 close[0..t] 的数据。
    所有 rolling 窗口都是「右对齐不含未来」：
        rolling_mean[i] = mean(close[i-n+1 : i+1])
    这一点由 _assert_no_lookahead() 在测试里验证。
"""
from __future__ import annotations

import math

import numpy as np


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """右对齐滚动均值。输出长度同输入，前 n-1 个为 NaN。

    注意：序列中的 NaN 会污染 cumsum，导致整条结果失效。
    这里改用「有效值累计 + 有效值计数」的方式，只在窗口内无 NaN 时才输出。
    """
    T = len(x)
    out = np.full(T, np.nan)
    if T < n:
        return out
    valid = np.isfinite(x)
    xv = np.where(valid, x, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(xv)))
    ccount = np.concatenate(([0], np.cumsum(valid.astype(float))))
    s = csum[n:] - csum[:-n]
    c = ccount[n:] - ccount[:-n]
    res = np.where(c == n, s / n, np.nan)
    out[n - 1:] = res
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    T = len(x)
    out = np.full(T, np.nan)
    if T < n:
        return out
    valid = np.isfinite(x)
    xv = np.where(valid, x, 0.0)
    x2 = xv * xv
    c1 = np.concatenate(([0.0], np.cumsum(xv)))
    c2 = np.concatenate(([0.0], np.cumsum(x2)))
    ccount = np.concatenate(([0], np.cumsum(valid.astype(float))))
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    c = ccount[n:] - ccount[:-n]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1)
    var = np.where(c == n, var, np.nan)
    out[n - 1:] = np.sqrt(np.maximum(var, 0.0))
    return out


def _rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    T = len(x)
    out = np.full(T, np.nan)
    for i in range(n - 1, T):
        out[i] = np.max(x[i - n + 1 : i + 1])
    return out


def _rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    T = len(x)
    out = np.full(T, np.nan)
    for i in range(n - 1, T):
        out[i] = np.min(x[i - n + 1 : i + 1])
    return out


def _pct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a/b - 1，b 为 0 或 NaN 时给 NaN。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a / b - 1.0
    r[~np.isfinite(r)] = np.nan
    return r


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """向后移 k 位（前面填 NaN）。a[i] 对应原来 a[i-k]。"""
    out = np.full(len(a), np.nan)
    if k < len(a):
        out[k:] = a[:-k]
    return out


def compute_all(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    amounts: np.ndarray,
    opens: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """一次算完所有因子。返回 {name: array[T]}。

    closes/highs/lows/volumes/amounts 均为等长 1D 数组，时间升序。
    """
    T = len(closes)
    out: dict[str, np.ndarray] = {}

    # ---- 动量（跳过最近 1 日，避免当日反转污染）
    for n in (20, 60, 120):
        out[f"mom_{n}"] = _pct(_shift(closes, 1), _shift(closes, n + 1))

    # ---- 短期反转（取负号）
    out["rev_5"] = -_pct(_shift(closes, 1), _shift(closes, 6))

    # ---- 均线偏离
    ma20 = _rolling_mean(closes, 20)
    ma60 = _rolling_mean(closes, 60)
    out["ma_dev_20"] = _pct(closes, ma20)
    out["ma_dev_60"] = _pct(closes, ma60)

    # ---- 收益率序列
    ret = np.full(T, np.nan)
    ret[1:] = _pct(closes[1:], closes[:-1])

    # ---- 波动率（年化）
    sd20 = _rolling_std(ret, 20)
    out["vol_20"] = sd20 * math.sqrt(252)
    out["low_vol"] = -out["vol_20"]

    # ---- RSI（Wilder）
    out["rsi_14"] = _rsi(closes, 14)

    # ---- 量比 / 成交额趋势
    v5 = _rolling_mean(volumes, 5)
    v60 = _rolling_mean(volumes, 60)
    out["vol_ratio"] = _pct(v5, v60)
    a10 = _rolling_mean(amounts, 10)
    a60 = _rolling_mean(amounts, 60)
    out["turnover_trend"] = _pct(a10, a60)

    # ---- 距高点回撤
    hi60 = _rolling_max(closes, 60)
    out["drawdown"] = _pct(closes, hi60)

    # ---- ATR / 价格（比值，非百分比）
    prev_close = _shift(closes, 1)
    tr = np.maximum.reduce([
        highs - lows,
        np.abs(highs - prev_close),
        np.abs(lows - prev_close),
    ])
    tr[0] = np.nan
    atr14 = _rolling_mean(tr, 14)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["atr_ratio"] = atr14 / closes
    out["atr_ratio"][~np.isfinite(out["atr_ratio"])] = np.nan

    # ---- 区间位置（Williams %R 变体）
    hi60b = _rolling_max(highs, 60)
    lo60b = _rolling_min(lows, 60)
    rng = hi60b - lo60b
    with np.errstate(invalid="ignore", divide="ignore"):
        hl = (closes - lo60b) / rng
    hl[~np.isfinite(hl)] = np.nan
    out["high_low"] = hl

    # ---- 跳空偏置：近 20 日平均 (open - prev_close)/prev_close
    gap_src = opens if opens is not None else closes
    out["gap_bias"] = _rolling_mean(_pct(gap_src, prev_close), 20)

    # ---- 涨跌一致性
    up = (ret > 0).astype(float)
    up[0] = np.nan
    out["updown_ratio"] = _rolling_mean(up, 20) - 0.5

    # ---- 120 日最大回撤
    out["maxdd_120"] = -_rolling_maxdd(closes, 120)

    return out


def _rsi(closes: np.ndarray, n: int) -> np.ndarray:
    """Wilder RSI，返回 (50-rsi)/50 中性化后的值。"""
    T = len(closes)
    out = np.full(T, np.nan)
    if T < n + 1:
        return out
    d = np.diff(closes)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    # Wilder 平滑：等效于指数加权 alpha=1/n
    alpha = 1.0 / n
    avg_gain = np.full(T - 1, np.nan)
    avg_loss = np.full(T - 1, np.nan)
    avg_gain[n - 1] = gain[:n].mean()
    avg_loss[n - 1] = loss[:n].mean()
    for i in range(n, T - 1):
        avg_gain[i] = (1 - alpha) * avg_gain[i - 1] + alpha * gain[i]
        avg_loss[i] = (1 - alpha) * avg_loss[i - 1] + alpha * loss[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[np.isnan(avg_loss) & (avg_gain == 0)] = 50.0
    rsi[np.isnan(avg_loss) & (avg_gain > 0)] = 100.0
    out[1:] = (50.0 - rsi) / 50.0
    return out


def _rolling_maxdd(closes: np.ndarray, n: int) -> np.ndarray:
    """滚动窗口最大回撤（正数表示回撤幅度）。"""
    T = len(closes)
    out = np.full(T, np.nan)
    for i in range(n - 1, T):
        seg = closes[i - n + 1 : i + 1]
        peak = np.maximum.accumulate(seg)
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = (peak - seg) / peak
        out[i] = np.nanmax(dd)
    return out


FACTOR_ORDER = [
    "mom_20", "mom_60", "mom_120", "rev_5", "ma_dev_20", "ma_dev_60",
    "low_vol", "rsi_14", "vol_ratio", "turnover_trend", "drawdown",
    "atr_ratio", "high_low", "gap_bias", "updown_ratio", "maxdd_120",
]

FACTOR_DESC = {
    "mom_20": "20日动量（跳过当日）",
    "mom_60": "60日动量",
    "mom_120": "120日动量",
    "rev_5": "5日短期反转（取负号）",
    "ma_dev_20": "偏离20日均线",
    "ma_dev_60": "偏离60日均线",
    "low_vol": "20日低波动（取负号）",
    "rsi_14": "RSI 中性化",
    "vol_ratio": "5日/60日量比",
    "turnover_trend": "成交额趋势",
    "drawdown": "距60日高点回撤",
    "atr_ratio": "ATR/价格（日内波动）",
    "high_low": "60日区间位置",
    "gap_bias": "近20日平均收益（跳空近似）",
    "updown_ratio": "20日涨跌一致性",
    "maxdd_120": "120日最大回撤（取负号）",
}
