# -*- coding: utf-8 -*-
"""ETF 回测执行层回归测试。

抓的是 run_strategy 里真实存在过的一个错：
    exit_px = px[i + R]        # ← 全市场一整行（N 只）
    valid = ~np.isnan(entry) & ~np.isnan(exit_px)   # entry 只有 k 只
后果有两种：
  a) k < N 时布尔掩码广播错位 → 直接 IndexError（k=1 时 valid 被拉成 N 维）；
  b) k == N 时不报错，但 entry 用 pick 顺序、exit_px 用列顺序，
     配对错位 → 算出一个「看起来合理」的错收益。

测试用「同集合但配对顺序不同」的两个方案构造出**不同的 gross**，
这样 b) 也一定会被抓出来。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import etf_lab as E  # noqa: E402

T = 60
N = 3
ENTRY_ROW = 21
EXIT_ROW = 26


def _panel(entry=None, exit_px=None) -> dict:
    """合成面板：默认三只 ETF 在 EXIT_ROW 有各自不同的价格。

    列 0 / 1 / 2 在 EXIT_ROW 的价格分别是 3 / 6 / 4，
    基准价分别是 1 / 2 / 4 —— 让「同集合不同配对」的均值确实不同。
    """
    op = np.ones((T, N))
    base = np.array([1.0, 2.0, 4.0])
    op[:] = base
    tgt = exit_px if exit_px is not None else np.array([3.0, 6.0, 4.0])
    op[EXIT_ROW] = tgt
    if entry is not None:
        op[ENTRY_ROW] = entry
    return {
        "close": op.copy(),
        "open": op.copy(),
        "high": op.copy(),
        "low": op.copy(),
        "volume": np.full((T, N), 1e6),
        "amount": np.full((T, N), 1e9),      # 远超 min_amount，流动性过滤不拦
        "dates": [f"2020-01-{i % 28 + 1:02d}" for i in range(T)],
        "codes": ["a", "b", "c"],
    }


def _score(top: int = 1) -> np.ndarray:
    s = np.zeros((T, N))
    s[:, top] = 1.0
    return s


def _run(panel, score, k, R=5, lo=ENTRY_ROW, hi=40, **kw):
    kw.setdefault("min_amount", 1e7)
    return E.run_strategy(panel, score, k, R, price_mode="open",
                          capital=1_000_000.0,
                          cost=False, lo=lo, hi=hi, **kw)


def test_k1_returns_the_picked_etf_not_the_market():
    """k=1：收益必须是被选中那一只自己的收益。"""
    p = _panel()
    res = _run(p, _score(top=1), k=1)
    assert res["periods"], "没有产生任何调仓期，说明面板构造有问题"
    p0 = res["periods"][0]
    assert p0["n_held"] == 1
    # 列1：2.0 → 6.0
    assert p0["gross"] == pytest.approx(6.0 / 2.0 - 1.0)


def test_k_all_pairs_by_picked_order_not_column_order():
    """k=N：entry 用 pick 顺序、exit 必须用**同一批 pick** 的顺序。

    pick 顺序 = [1, 0, 2]（列1 分数最高）
    正确：entry [2,1,4] → exit [6,3,4] → 收益 [2.0, 2.0, 0.0] → 均值 4/3
    错配：entry [2,1,4] → exit 按列顺序 [3,6,4] → 收益 [0.5, 5.0, 0.0] → 均值 1.8333
    """
    p = _panel()
    res = _run(p, _score(top=1), k=3)
    assert res["periods"]
    gross = res["periods"][0]["gross"]
    assert gross == pytest.approx(4.0 / 3.0), f"配对错位：gross={gross}"
    assert gross != pytest.approx(1.8333333), "gross 恰好等于列顺序错配的结果"


def test_pick_order_is_by_score_desc():
    """分数高的先买：列1 分数最高 → 第一个持仓必须是它。"""
    p = _panel()
    res = _run(p, _score(top=2), k=1)
    # 列2：4.0 → 4.0 → 0%
    assert res["periods"][0]["gross"] == pytest.approx(0.0)


def test_signal_uses_previous_close_no_lookahead():
    """入场价必须是 open[i]，不能是 open[i+1]。"""
    p = _panel()
    base = _run(p, _score(top=1), k=1)["periods"][0]["gross"]

    # 把 ENTRY_ROW+1 的价格打乱成极端值 —— 若代码误用 open[i+1] 当入场价，
    # 结果必然变化。
    p2 = _panel()
    p2["open"][ENTRY_ROW + 1, 1] = 999.0
    p2["close"][ENTRY_ROW + 1, 1] = 999.0
    alt = _run(p2, _score(top=1), k=1)["periods"][0]["gross"]
    assert alt == pytest.approx(base)


def test_exit_row_price_drives_the_result():
    """出场价必须是 open[i+R]。"""
    p = _panel(exit_px=np.array([3.0, 8.0, 4.0]))
    res = _run(p, _score(top=1), k=1)
    assert res["periods"][0]["gross"] == pytest.approx(8.0 / 2.0 - 1.0)


def test_nan_exit_price_drops_that_name():
    """出场价缺失的标的必须被剔出平均，而不是变成 NaN 污染整期。"""
    p = _panel()
    p["open"][EXIT_ROW, 1] = np.nan
    p["close"][EXIT_ROW, 1] = np.nan
    res = _run(p, _score(top=1), k=2)
    p0 = res["periods"][0]
    assert p0["n_held"] == 1, "出场价 NaN 的标的应被剔除"
    assert np.isfinite(p0["gross"])


def test_liquidity_filter_uses_only_prior_day():
    """流动性过滤只能用截至 t−1 的成交额均值（当日额 t 日尚不可知）。

    构造：除第 21 行外成交额都是 1e9，第 21 行（= 入场当日）为 0。
    · 正确（用 t−1 的 20 日均额 = 1e9）→ 列1 通过过滤，能买。
    · 错误（用当日均额 = 19/20×1e9 = 9.5e8）→ 低于门槛 9.6e8，列1 被滤掉。
    """
    p = _panel()
    p["amount"][ENTRY_ROW, 1] = 0.0
    res = _run(p, _score(top=1), k=1, min_amount=9.6e8)
    assert res["periods"][0]["n_held"] == 1, \
        "过滤误用了当日成交额 —— 那是前视偏差"
    assert res["periods"][0]["gross"] == pytest.approx(6.0 / 2.0 - 1.0)


def test_metrics_reports_three_return_calibers():
    """收益统计必须三口径并列，不能只给均值。"""
    periods = [
        {"date": "d0", "gross": 0.01, "cost": 0.0002, "n_held": 3, "n_sold": 1, "n_bought": 1},
        {"date": "d1", "gross": -0.02, "cost": 0.0003, "n_held": 3, "n_sold": 2, "n_bought": 2},
        {"date": "d2", "gross": 0.03, "cost": 0.0001, "n_held": 3, "n_sold": 0, "n_bought": 1},
    ]
    m = E.metrics(periods, R=5)
    for key in ("gross_annualized", "net_annualized",
                "net_winsor_mean_annualized", "net_median_annualized"):
        assert key in m, f"缺三口径统计项 {key}"
    assert m["gross_per_period"] == pytest.approx(np.mean([p["gross"] for p in periods]))
    assert m["net_per_period"] == pytest.approx(
        np.mean([p["gross"] - p["cost"] for p in periods]))
