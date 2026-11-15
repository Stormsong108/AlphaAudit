# -*- coding: utf-8 -*-
"""etf_lab 统计口径的回归测试。

抓的是一个**数字看着像那么回事、其实错了 25 倍**的错：
年化 Sharpe 的年化因子被除反了。

正确：Sharpe_ann = (mean_per_period × ann) / (sd_per_period × √ann)
                 = (mean / sd) × √ann
错写：mean / (sd × √ann)  ← 年化因子变成了除数

实测：ETF 通道 ann = 252/10 = 25.2，错公式报出 Sharpe 0.018，
真实值 0.44 —— 差 25.2 倍。而项目其余地方（ooslab.py /
execution_realism.py / market_timing.py / timing.py）都是对的，
只有 etf_lab.py 是例外，所以这个错不会被交叉比对发现，只能靠测试锁住。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import etf_lab as E  # noqa: E402


def _periods(gross: np.ndarray) -> list[dict]:
    return [{"date": f"2024-{1 + i // 28:02d}-{i % 28 + 1:02d}",
             "gross": float(g), "cost": 0.0,
             "n_held": 5, "n_sold": 1, "n_bought": 1}
            for i, g in enumerate(gross)]


def test_metrics_sharpe_is_annualized():
    """年化 Sharpe = (mean/sd) × √ann，年化因子必须**在分子**。"""
    rng = np.random.default_rng(3)
    gross = rng.normal(0.004, 0.010, 60)
    R = 10
    m = E.metrics(_periods(gross), R)
    ann = E.TRADING_DAYS / R
    sd = float(np.std(gross, ddof=1))
    expect = float(np.mean(gross) / sd * np.sqrt(ann))
    assert m["sharpe"] == pytest.approx(expect, rel=1e-9)
    # 反证：不能是「除 √ann」那个错公式
    wrong = float(np.mean(gross) / (sd * np.sqrt(ann)))
    assert abs(m["sharpe"] - wrong) > 1e-6


def test_metrics_sharpe_scales_with_ann():
    """同样的每期收益，R 越小（期数越多、年化次数越大）→ 年化 Sharpe 越大。"""
    rng = np.random.default_rng(5)
    gross = rng.normal(0.004, 0.010, 120)
    s5 = E.metrics(_periods(gross), 5)["sharpe"]
    s20 = E.metrics(_periods(gross), 20)["sharpe"]
    # √ann 比值：√(252/5) / √(252/20) = 2
    assert s5 == pytest.approx(s20 * 2.0, rel=1e-9)


def test_benchmark_sharpe_also_annualized():
    """基准的 Sharpe 用同一个公式（不能只修策略、漏了基准）。

    独立复刻基准的每期收益序列（只复用模块的滚动均值helper，
    复权/Sharpe 公式全部自己算），再核对 Sharpe。
    """
    rng = np.random.default_rng(9)
    T, N = 140, 5
    px = 10.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.010, (T, N)), axis=0))
    amt = np.full((T, N), 1.0e8)
    dates = [f"{2020 + i // 250:04d}-{1 + (i // 21) % 12:02d}-{1 + i % 28:02d}"
             for i in range(T)]
    panel = {"open": px, "amount": amt, "dates": dates}
    R = 10
    out = E.equal_weight_benchmark(panel, R, 0, T)
    assert out["n_periods"] > 0

    # ── 独立复刻每期收益 ──
    amt_ma = E._rolling_mean(amt, 20)
    rs = []
    i = 0
    while i + R < T:
        elig = (~np.isnan(px[i]) & ~np.isnan(px[i + R])
                & ~np.isnan(amt_ma[i - 1]) & (amt_ma[i - 1] >= 5e7))
        if int(elig.sum()) >= 2:
            rs.append(float(np.mean(px[i + R][elig] / px[i][elig] - 1.0)))
        i += R
    r = np.array(rs)
    ann = E.TRADING_DAYS / R
    sd = float(np.std(r, ddof=1))
    expect = float(np.mean(r) / sd * np.sqrt(ann))
    assert out["sharpe"] == pytest.approx(expect, rel=1e-9)
    wrong = float(np.mean(r) / (sd * np.sqrt(ann)))
    assert abs(out["sharpe"] - wrong) > 1e-6


def test_metrics_annualization_uses_periods_per_year():
    """年化收益 = 每期均值 × (252/R)，不是 × 252。"""
    rng = np.random.default_rng(11)
    gross = rng.normal(0.004, 0.010, 60)
    R = 10
    m = E.metrics(_periods(gross), R)
    ann = E.TRADING_DAYS / R
    assert m["net_annualized"] == pytest.approx(float(np.mean(gross)) * ann, rel=1e-9)
    assert abs(m["net_annualized"] - float(np.mean(gross)) * 252) > 1e-6
