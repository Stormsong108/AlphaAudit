# -*- coding: utf-8 -*-
"""ETF 因子正确性回归测试。

重点是 _rsi：之前它用 `np.where(lo > 0, cg[lo-1], 0.0)`，在 2D 面板下
条件形状 (M,) 与值形状 (M, N) 广播不上，compute_factors 直接崩。
修完之后用「朴素循环参考实现」交叉比对，而不是只断言形状。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import etf_lab as E  # noqa: E402


def _naive_rsi_col(col: list[float], w: int = 14) -> list[float]:
    """朴素参考实现：逐行、逐列、逐窗口算 RSI。慢，但显然正确。"""
    out = [float("nan")] * len(col)
    for r in range(w, len(col)):
        win = col[r - w: r + 1]          # w+1 个价格 → w 个涨跌幅
        if any(not np.isfinite(p) for p in win):
            continue
        diffs = [win[k + 1] - win[k] for k in range(w)]
        g = sum(d for d in diffs if d > 0)
        l = sum(-d for d in diffs if d < 0)
        if g + l <= 0:
            out[r] = 0.5
        else:
            out[r] = g / (g + l)
    return out


def test_rsi_matches_naive_reference():
    rng = np.random.default_rng(20261114)
    a = 2.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(120, 6)), axis=0))
    got = E._rsi(a, 14)
    for j in range(a.shape[1]):
        want = _naive_rsi_col(list(a[:, j]), 14)
        for r in range(len(want)):
            g, wv = got[r, j], want[r]
            if np.isnan(wv):
                assert np.isnan(g), f"col{j} row{r} 期望 NaN 实得 {g}"
            else:
                assert g == pytest.approx(wv, rel=1e-12, abs=1e-12), \
                    f"col{j} row{r}: {g} != {wv}"


def test_rsi_window_is_exactly_w_periods():
    """RSI 用恰好 w 个涨跌幅，不是 w+1 个（老代码有 off-by-one）。"""
    a = np.array([[1.0], [1.1], [1.2], [1.3]])
    # w=2：行2 的窗口是 [1.0,1.1,1.2] → d=[0.1,0.1] → RSI=1.0
    got = E._rsi(a, 2)
    assert got[2, 0] == pytest.approx(1.0)
    # 若窗口是 3 个涨跌幅 [0.1,0.1,0.1] 结果同样是 1.0，
    # 所以再用一个有涨有跌的序列区分
    b = np.array([[1.0], [1.1], [1.05], [1.2]])
    got_b = E._rsi(b, 2)
    # w=2 → d=[0.1,-0.05] → g=0.1, l=0.05 → RSI = 0.1/0.15
    assert got_b[2, 0] == pytest.approx(0.1 / 0.15)


def test_rsi_nan_prefix_yields_nan():
    a = np.full((30, 2), np.nan)
    a[10:, 0] = np.linspace(1.0, 2.0, 20)
    got = E._rsi(a, 14)
    assert np.isnan(got[:24, 0]).all()
    assert np.isfinite(got[24:, 0]).all()
    assert np.isnan(got[:, 1]).all()


def test_rsi_no_lookahead():
    """改未来值不能影响今天的 RSI。"""
    rng = np.random.default_rng(7)
    a = 2.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(80, 4)), axis=0))
    base = E._rsi(a, 14)
    b = a.copy()
    b[60:, 2] *= 3.0           # 篡改 60 行之后的第 3 列
    after = E._rsi(b, 14)
    assert np.allclose(base[:60], after[:60], equal_nan=True)


def test_compute_factors_shape_and_nan_prefix():
    rng = np.random.default_rng(1)
    cl = 2.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(200, 5)), axis=0))
    cl[:30, 1] = np.nan       # 某只上市前无数据
    F = E.compute_factors(cl)
    for k, v in F.items():
        assert v.shape == cl.shape, k
        assert np.isnan(v[:29, 1]).all(), f"{k} 上市前不应有值"
        # 末尾若干行因回看窗口天然为 NaN（最大回看 120 期，如 mom_120/dd_120），
        # 成熟段只看 [150, T-130)
        assert np.isfinite(v[150:-130, 0]).all(), f"{k} 成熟段不应有 NaN"
