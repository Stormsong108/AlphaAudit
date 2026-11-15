"""数据质量守卫的测试。

这些测试锁住的是 2026-11-10 踩过的那个坑：
    前复权价穿零 → 负价格 → 用负价格当分母算收益 → 符号翻转、幅度离谱
    → 横截面均值、IC、分组收益全被少数坏点带偏。
守卫必须在**坏数据进入研究之前**就把路堵死。
"""
from __future__ import annotations

import numpy as np
import pytest

import dataquality


def _panel(close: np.ndarray, codes: list[str] | None = None) -> dict:
    T, N = close.shape
    return {
        "dates": [f"2020-01-{i + 1:02d}" for i in range(T)],
        "codes": codes or [f"0.{i:06d}" for i in range(N)],
        "close": close,
        "fwd": np.zeros_like(close),
        "amount": np.ones_like(close),
        "factors": {},
    }


def test_limit_of_reads_market_prefixed_codes():
    """codes 是 '0.002003' 这种带市场前缀的，必须取后 6 位判制度。"""
    assert dataquality.limit_of("0.002003") == 0.10
    assert dataquality.limit_of("1.600519") == 0.10
    assert dataquality.limit_of("0.300033") == 0.20   # 创业板
    assert dataquality.limit_of("0.301656") == 0.20
    assert dataquality.limit_of("1.688825") == 0.20   # 科创板
    assert dataquality.limit_of("0.430047") == 0.30   # 北交所


def test_clean_panel_passes():
    close = np.tile(np.array([[10.0], [10.1], [10.05], [10.2], [10.3]]), (1, 30))
    rep = dataquality.audit_panel(_panel(close))
    assert rep["ok"], rep["issues"]
    assert rep["issues"] == []
    # 干净面板不该抛异常
    dataquality.assert_panel_clean(_panel(close))


def test_non_positive_price_is_detected_and_blocks():
    """前复权穿零的负价格必须被拦下。"""
    close = np.full((40, 30), 10.0)
    close[10, 3] = -0.15      # 贵州茅台前复权首价的真实形态
    close[11, 3] = 0.5
    rep = dataquality.audit_panel(_panel(close))
    assert not rep["ok"]
    kinds = {i["kind"] for i in rep["issues"]}
    assert "non_positive_price" in kinds
    with pytest.raises(dataquality.DirtyPanelError):
        dataquality.assert_panel_clean(_panel(close))


def test_limit_breach_is_detected():
    """单日涨跌幅离谱到不可能，必须被报出来。

    注意两点：
      1. 超过容忍带只是**软告警**，且只有占比超过 1% 才进 issues ——
         除权除息日（分红/送股）让总回报高于价格制度上限是完全正常的。
      2. 占比是**逐股日**口径（400 天 × 30 只 = 12,000 个观测），
         不是逐日横截面口径。
    """
    close = np.full((400, 30), 10.0)
    close[::10, :] = 13.0        # 每 10 天一次 +30%，约占 10%
    rep = dataquality.audit_panel(_panel(close))
    assert not rep["ok"]
    kinds = {i["kind"] for i in rep["issues"]}
    assert "limit_breach" in kinds
    assert rep["soft_breach_rate"] > 0.01


def test_single_soft_breach_is_not_a_failure():
    """偶发一次除权日不算面板有问题。"""
    close = np.full((400, 30), 10.0)
    close[7, 7] = 13.0       # 只占 0.25%
    rep = dataquality.audit_panel(_panel(close))
    assert rep["ok"], rep["issues"]
    assert rep["soft_breach_rate"] > 0


def test_hundred_percent_move_is_a_hard_failure():
    """单日翻倍以上，任何分红都解释不了，必须硬失败。"""
    close = np.full((400, 30), 10.0)
    close[7, 7] = 25.0       # +150%
    rep = dataquality.audit_panel(_panel(close))
    assert not rep["ok"]
    assert "extreme_move" in {i["kind"] for i in rep["issues"]}
    with pytest.raises(dataquality.DirtyPanelError):
        dataquality.assert_panel_clean(_panel(close))


def test_chinext_20pct_move_is_not_a_breach():
    """创业板 ±20% 是合法的，不能误报。"""
    codes = [f"0.30{i:04d}" for i in range(30)]
    close = np.full((400, 30), 10.0)
    close[5, 7] = 11.9       # +19%，创业板合法
    rep = dataquality.audit_panel(_panel(close, codes))
    assert rep["ok"], rep["issues"]


def test_chi_next_20pct_move_is_a_breach():
    """同一只票涨 30% 就是越界了。"""
    codes = [f"0.30{i:04d}" for i in range(30)]
    close = np.full((400, 30), 10.0)
    close[::10, :] = 13.0
    rep = dataquality.audit_panel(_panel(close, codes))
    assert not rep["ok"]
    assert "limit_breach" in {i["kind"] for i in rep["issues"]}


def test_dividend_day_move_above_price_limit_is_allowed():
    """除权日：后复权涨幅可以超过价格制度上限。

    实测 000426（2015-03-26 除权）：裸价 +10.04%（真实涨停），
    后复权 +11.68%（含每股 0.194 元分红）—— 后者必须放行，
    否则每个高分红股都会被误判成坏数据。
    """
    close = np.full((400, 30), 10.0)
    close[5, 7] = 11.168     # +11.68%，主板，但对应一次除权
    rep = dataquality.audit_panel(_panel(close))
    assert rep["ok"], rep["issues"]


def test_new_listing_first_five_days_are_exempt():
    """新股上市后前 5 个交易日不设涨跌幅限制 → 不能判为异常。

    实测 688026 洁特生物上市初期 +108.54% 是合法的。
    """
    close = np.full((400, 30), 10.0)
    close[1, 7] = 25.0        # 该股第 2 个有效观测，+150%
    rep = dataquality.audit_panel(_panel(close))
    assert rep["ok"], rep["issues"]
    assert rep["new_listing_exempt"] == 1


def test_date_axis_threshold_is_relative_to_eligible_stocks():
    """日期轴阈值必须相对"当日已上市的标的数"，而不是当前股票池总数。

    回归 2026-11-10 的坑：阈值写死 0.30 × 全部股票数（1328 × 30% = 398），
    而 2015 年 A 股总共才上市约 400 只，正常交易日覆盖率 380~400 正好卡在
    阈值附近 → 15 个正常交易日被丢出轴 → 日期轴断层 →
    跨断层的涨幅被当成"单日涨幅"（+109% / +147% / +154% 假极端值），
    而且按轴位置算的"持有 5 日"实际变成了"持有 20 日"。
    """
    import factorlab

    def bar(d, c):
        return {"date": d, "open": c, "high": c, "low": c, "close": c,
                "volume": 1000.0, "amount": 10000.0}

    early = ["2020-01-%02d" % (i + 1) for i in range(10)]
    late = ["2020-03-%02d" % (i + 1) for i in range(90)]
    hist = {}
    for i in range(10):
        hist["0.0000%02d" % i] = [bar(d, 10.0) for d in early]
    for i in range(90):
        hist["0.0001%02d" % i] = [bar(d, 20.0) for d in late]

    mat = factorlab.build_matrix(hist, min_bars=0, forward_days=1)
    assert early[0] in mat["dates"], "早期交易日被错误地丢出日期轴"
    assert len(mat["dates"]) == 100, "日期轴应该包含全部 100 个交易日"
    # 轴必须连续：相邻轴行的间隔不能超过一个交易日
    assert mat["dates"] == sorted(mat["dates"])


def test_error_message_carries_actionable_hint():
    """报错必须给出处置建议，不能只说'有问题'。"""
    close = np.full((40, 30), 10.0)
    close[10, 3] = -0.15
    with pytest.raises(dataquality.DirtyPanelError) as exc:
        dataquality.assert_panel_clean(_panel(close), context="单元测试")
    msg = str(exc.value)
    assert "单元测试" in msg
    assert "fqt" in msg  # 指向后复权口径


def test_thin_cross_section_is_reported():
    """某一天只剩 5 只有效数据，要报出来。"""
    close = np.full((40, 30), 10.0)
    close[:, 5:] = np.nan
    rep = dataquality.audit_panel(_panel(close))
    kinds = {i["kind"] for i in rep["issues"]}
    assert "thin_cross_section" in kinds
