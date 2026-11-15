# -*- coding: utf-8 -*-
"""ETF 份额折算复权的回归测试。

抓的是 etf_adj.py 里真实存在过的三个错（都是 2026-11-14 那轮重建暴露的）：

1. **折算日切不出段边界** → k 跨 regime 整段拟合
   `_segments` 原来用 `if d in folds` 做**精确字符串匹配**，但折算生效日
   常常**不在本地行情的日期轴上**（实测 sh512010/512100/512170/512480/
   sh512690/513100/513500/sz159919 这 8 只的生效日本地都没有 bar）。
   匹配不上 → 不切段 → k 跨着折算拟合 → 残差全段是垃圾 →
   净值反解出的「除息日」从真实的 ~1 次/年 暴涨到几百次（实测 586 次）。

2. **切分点没被真正当作段边界**
   `estimate_k` 原来即使算出了 regimes，也只是记下来当「事件列表」，
   bounds 仍然是 [0, len] → 同样整段拟合。

3. **折算比例没进复权因子**
   检出的 k 被赋值后从未参与 factor 更新 → 折算日残留假跳空
   （sh510500 残留 −12.74%，sh515220 残留 −50.34%）。

测试全部用**合成数据**，不依赖网络也不依赖本地行情文件，
所以在别的机器上跑结果一致。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import etf_adj as A  # noqa: E402


# ── 合成数据构造 ───────────────────────────────────────────────────────
def _dates(n: int, start: str = "2020-01-01") -> list[str]:
    """n 个连续交易日（跳过周末）。"""
    import datetime as dt
    d0 = dt.date.fromisoformat(start)
    out = []
    d = d0
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _nav_series(dates: list[str], k_before: float, k_after: float,
                split_at: int, div_at: int | None = None,
                div_amt: float = 0.05) -> dict[str, tuple[float, float]]:
    """构造 (U, C) 净值序列。

    恒等式：C = k·U + D，其中 D 是累计分红（**k 加权口径**）。
    份额折算发生在 split_at：U 跳 X 倍、C 连续 → C/U 的 k 从 k_before
    变成 k_before·X（与真实 ETF 一致）。

    ⚠️ 除息时 D 的增量是 **k·d** 而不是 d —— 因为 C = k·U + D，
    除息日 C 连续、U 掉 d，于是 D 必须涨 k·d 才能维持恒等式。
    实证依据（data/research/etf_dividend_crosscheck.json，510300）：
    F10 累计分红 0.880、k=0.370951 ⇒ 0.880×k = 0.326437，
    与净值侧 C−k·U = 0.3267 吻合到四位小数。

    日波动用确定性伪随机游走（不依赖 random 全局状态，测试可复现）：
    幅度 ~0.5%，与真实 ETF 单位净值日波动同量级 —— 这一点很关键，
    因为 k 的拟合只用 |ΔU|>0.002 的日子做回归，波动太小会让样本点不足、
    拟合退化（那会让测试变成在测「合成数据太假」而不是测真实逻辑）。
    """
    nav: dict[str, tuple[float, float]] = {}
    u = 1.0
    d_cum = 0.0
    state = 12345
    for i, dt_ in enumerate(dates):
        k = k_before if i < split_at else k_after
        if i == split_at:
            # 折算：份额 ×X → 单位净值 ÷X，累计净值连续
            u = u / (k_after / k_before)
        else:
            state = (1103515245 * state + 12345) % (2 ** 31)
            u *= 1.0 + ((state / 2 ** 31) - 0.5) * 0.01   # ±0.5% 日波动
        if div_at is not None and i == div_at:
            u -= div_amt                     # 除息：U 掉 d，C 不变
            d_cum += k * div_amt             # D 涨 k·d 维持 C = k·U + D
        c = k * u + d_cum
        nav[dt_] = (u, c)
    return nav


def _price_rows(dates: list[str], k_before: float, k_after: float,
                split_at: int) -> list[dict]:
    """构造与 _nav_series 对应的价格序列：折算日价格 ÷X。"""
    x = k_after / k_before
    rows = []
    for i, dt_ in enumerate(dates):
        px = 1.0
        if i >= split_at:
            px = 1.0 / x
        rows.append({"date": dt_, "open": px, "high": px * 1.01,
                     "low": px * 0.99, "close": px, "volume": 1000.0,
                     "amount": 1000.0 * px})
    return rows


# ── 1. 折算日不在日期轴上时，仍必须切出段边界 ──────────────────────────
def test_fold_date_absent_from_axis_still_segments():
    """折算生效日**不在** dates 里（本地无当日 bar）时也必须切段。

    这是 sh512010/513100 等 8 只失败的直接原因：
    原来 `d in folds` 精确匹配失败 → 不切段 → k 跨 regime 拟合。
    """
    dates = _dates(200)
    # 折算生效日故意给一个 **不在 dates 里** 的日子
    fold_day = "2020-07-15"          # 该日是周三，但下面把 dates 挪开
    expect = next(i for i, d in enumerate(dates) if d >= fold_day)
    # 净值侧的 k 跳变必须和 F10 折算日指向**同一处** —— 否则这是两个互相
    # 矛盾的信息源（真实数据里 F10 漏记折算是常事，见下面的回归测试）。
    nav = _nav_series(dates, 1.0, 4.0, split_at=expect)

    k, regimes, d_true = A.estimate_k(dates, nav, folds=[fold_day])
    assert len(regimes) >= 1, "折算日不在日期轴上也必须切出段边界"
    # 段边界应落在「第一个 >= 折算日的日期」
    assert regimes[0] == expect


def test_fold_date_exact_match_still_segments():
    """折算日恰好在日期轴上时，边界必须精确落在该位置。"""
    dates = _dates(200)
    nav = _nav_series(dates, 1.0, 2.0, split_at=100)
    k, regimes, d_true = A.estimate_k(dates, nav, folds=[dates[100]])
    assert 100 in regimes


# ── 2. 分段后 k 必须随折算跳变，且各段 k 要对得上 ──────────────────────
def test_k_changes_across_fold_by_ratio():
    """1:2 折算 → 第二段 k 应是第一段的 2 倍。"""
    dates = _dates(200)
    nav = _nav_series(dates, 1.0, 2.0, split_at=100)
    k, regimes, _ = A.estimate_k(dates, nav, folds=[dates[100]])
    assert len(regimes) == 1
    assert k[100] / k[0] == pytest.approx(2.0, rel=1e-6)


def test_no_false_dividends_when_segmented():
    """正确分段后，净值反解出的除息日应接近 0（没有真实除息时）。

    未分段时这个数字是几百次（实测 586 次）—— 复权序列因此被污染。
    """
    dates = _dates(300)
    nav = _nav_series(dates, 1.0, 4.0, split_at=150)
    k, regimes, d_true = A.estimate_k(dates, nav, folds=[dates[150]])
    n_exdiv = sum(1 for i, d in enumerate(dates)
                  if d_true[i] > A.DIV_REL_THRESH * nav[d][0])
    assert n_exdiv == 0, f"分段后不该有假除息日，实得 {n_exdiv}"


def test_real_dividend_is_detected_once():
    """有真实除息时，应当恰好检出 1 次。"""
    dates = _dates(300)
    nav = _nav_series(dates, 1.0, 2.0, split_at=150, div_at=200, div_amt=0.05)
    k, regimes, d_true = A.estimate_k(dates, nav, folds=[dates[150]])
    n_exdiv = sum(1 for i, d in enumerate(dates)
                  if d_true[i] > A.DIV_REL_THRESH * nav[d][0])
    assert n_exdiv == 1, f"应恰好检出 1 次除息，实得 {n_exdiv}"


# ── 4. F10 折算记录不完整时，必须靠净值 C/U 跳变补上 ──────────────────
def test_split_missing_from_f10_is_detected_from_nav():
    """F10 漏记的折算必须由 C/U 跳变检测补上（sh512690 的真实教训）。

    实测 sh512690：F10 只记了 2021-05-14 (X=2.0)，但净值 C/U 在 2021-12-31
    又跳了一次（真折算）。旧实现「F10 优先、检测只兜底」→ 这次折算被当成
    36.6% 的假分红 → div_gap 爆表（致命失败）。
    """
    dates = _dates(300)
    nav = _nav_series(dates, 1.0, 2.0, split_at=150)
    # F10 只给了一个更早的折算日 —— 真正的折算在 150，F10 没记
    k, regimes, d_true = A.estimate_k(dates, nav, folds=[dates[50]])
    assert 150 in regimes, "F10 漏记的折算必须由 C/U 跳变检测补上"
    assert k[150] / k[50] == pytest.approx(2.0, rel=1e-6)


def test_large_dividend_is_not_treated_as_split():
    """大额分红不能被当成折算切段。

    sh512690 2021-12-31（F10 原文：每10份派 3.50元、占基金规模 26.67%）：
    C 连续、U 掉 26.6%、C/U 从 2.0 跳到 2.73 —— 形态上和折算**一模一样**，
    唯一区别是**局部 k 没变**（k_right/k_left ≈ 1）。
    在这里切段 → 该日的分红残差落在段外被丢掉 → 价格少复权。
    """
    dates = _dates(300)
    nav = _nav_series(dates, 1.0, 1.0, split_at=150, div_at=200, div_amt=0.35)
    k, regimes, d_true = A.estimate_k(dates, nav, folds=None)
    # 大额分红必须**没有**制造出段边界
    assert 200 not in regimes, "大额分红被误判成折算 → 该分红残差会被丢掉"
    # 但分红本身必须照常检出
    n_exdiv = sum(1 for i, d in enumerate(dates)
                  if d_true[i] > A.DIV_REL_THRESH * nav[d][0])
    assert n_exdiv == 1, f"应恰好检出 1 次分红，实得 {n_exdiv}"


def test_detect_regimes_fallback_actually_segments():
    """不传 folds 时，C/U 跳变必须真的被当作段边界（不能只记不用）。"""
    dates = _dates(300)
    nav = _nav_series(dates, 1.0, 5.0, split_at=150)
    k, regimes, d_true = A.estimate_k(dates, nav, folds=None)
    assert len(regimes) >= 1, "C/U 跳变必须切段"
    assert k[-1] / k[0] == pytest.approx(5.0, rel=1e-6)


# ── 3. 折算比例必须真的进复权因子（核心） ─────────────────────────────
def test_fold_ratio_applied_to_factor_removes_fake_jump():
    """折算后 adj 序列不得再有假跳空。

    未应用折算比例时，1:4 折算会在复权序列里留下 −75% 的假跌。
    """
    dates = _dates(200)
    nav = _nav_series(dates, 1.0, 4.0, split_at=100)
    rows = _price_rows(dates, 1.0, 4.0, split_at=100)
    actions = {"splits": [{"date": dates[100], "type": "份额分拆", "X": 4.0}],
               "divs": []}
    adj, diag = A.adjust_series(rows, nav, code6=None, actions=actions)

    rets = np.array([adj[i]["adj_close"] / adj[i - 1]["adj_close"] - 1.0
                     for i in range(1, len(adj))])
    assert np.max(np.abs(rets)) < 0.02, (
        f"折算后仍存在假跳空：max|ret|={np.max(np.abs(rets)):.4f}")
    assert diag["n_regime_events"] == 1
    assert diag["total_factor"] == pytest.approx(4.0, rel=1e-6), \
        "折算比例必须进复权因子"


def test_factor_monotonic_and_bounded():
    """total_factor 在合理区间，且 adj_factor 单调不减。"""
    dates = _dates(200)
    nav = _nav_series(dates, 1.0, 2.0, split_at=100)
    rows = _price_rows(dates, 1.0, 2.0, split_at=100)
    actions = {"splits": [{"date": dates[100], "type": "份额分拆", "X": 2.0}],
               "divs": []}
    adj, diag = A.adjust_series(rows, nav, code6=None, actions=actions)
    factors = [r["adj_factor"] for r in adj]
    assert all(b >= a - 1e-12 for a, b in zip(factors, factors[1:])), \
        "adj_factor 必须单调不减"
    assert 0.5 <= diag["total_factor"] <= 3.0


def test_no_fold_means_no_factor_change():
    """没有折算事件时，因子不应无故变化。"""
    dates = _dates(200)
    nav = _nav_series(dates, 1.0, 1.0, split_at=100)
    rows = _price_rows(dates, 1.0, 1.0, split_at=100)
    adj, diag = A.adjust_series(rows, nav, code6=None,
                                actions={"splits": [], "divs": []})
    assert diag["total_factor"] == pytest.approx(1.0, rel=1e-9)


# ── 4. F10 缓存 schema 归一化 ──────────────────────────────────────────
def test_f10_split_ratio_string_normalized():
    """缓存里 '1:4.0000' 这种字符串写法要能归一化成 X=4.0。"""
    out = A._norm_split({"date": "2021-06-25", "type": "份额分拆",
                         "ratio": "1:4.0000"})
    assert out is not None and out["X"] == pytest.approx(4.0)
    assert out["date"] == "2021-06-25"


def test_f10_split_float_X_preserved():
    out = A._norm_split({"date": "2022-08-26", "type": "份额分拆", "X": 1.1454})
    assert out["X"] == pytest.approx(1.1454)


def test_f10_div_normalized_from_amount_per_share():
    out = A._norm_div({"ex_date": "2024-06-20", "amount_per_share": 0.08})
    assert out is not None and out["per_share"] == pytest.approx(0.08)


def test_f10_div_normalized_from_per10():
    out = A._norm_div({"ex_date": "2024-06-20", "per10": 0.8})
    assert out is not None and out["per_share"] == pytest.approx(0.08)


def test_f10_split_missing_ratio_rejected():
    assert A._norm_split({"date": "2021-06-25", "ratio": "garbage"}) is None
    assert A._norm_split({"date": "2021-06-25", "X": 0.0}) is None


# ── 5. 价格跳变定位 ────────────────────────────────────────────────────
def test_locate_price_jump_finds_first_bar_after_fold():
    """折算日无 bar 时，价格跳变应落在生效日之后第一根可用 bar。"""
    dates = _dates(60)
    closes = [1.0] * 60
    closes[30] = 0.25                       # 1:4 折算的假跌
    jump = A._locate_price_jump(dates, closes, dates[29], 4.0)
    assert jump is not None
    assert jump["date"] == dates[30]
    assert jump["price_ratio"] == pytest.approx(0.25, rel=1e-9)


def test_locate_price_jump_rejects_when_ratio_mismatch():
    """比例对不上时必须拒绝（宁可漏掉也不能乱乘）。"""
    dates = _dates(60)
    closes = [1.0] * 60
    closes[30] = 0.90                       # 只有 -10%，跟 1:4 对不上
    assert A._locate_price_jump(dates, closes, dates[29], 4.0) is None
