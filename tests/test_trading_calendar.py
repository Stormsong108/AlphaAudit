"""交易日判定测试。

核心约束：不硬编码节假日表，交易日状态必须从行情接口的 trade_date 反推。
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from trading_calendar import resolve_market_session, session_label  # noqa: E402


def _quotes(a_date: str, hk_date: str, us_date: str) -> dict:
    return {
        "indexes": [
            {"name": "上证指数", "region": "A股", "trade_date": a_date},
            {"name": "沪深300", "region": "A股", "trade_date": a_date},
            {"name": "恒生指数", "region": "港股", "trade_date": hk_date},
            {"name": "道琼斯工业指数", "region": "美股", "trade_date": us_date},
        ],
        "stocks": [],
    }


D = dt.date(2026, 11, 9)  # 周一


def test_trading_day_detected():
    s = resolve_market_session(_quotes("2026-11-09", "2026-11-09", "2026-11-07"), today=D)
    assert s["is_trading_day"] is True
    assert s["data_date"] == "2026-11-09"
    assert s["markets"]["A股"]["is_open"] is True
    # 美股当天还没开盘（美东时间还没到收盘），应标为休市/未开
    assert s["markets"]["美股"]["is_open"] is False
    assert s["markets"]["美股"]["lag_days"] == 2


def test_weekend_is_not_trading_day():
    # 周日：行情仍是上周五
    sunday = dt.date(2026, 11, 8)
    s = resolve_market_session(_quotes("2026-11-06", "2026-11-06", "2026-11-06"), today=sunday)
    assert s["is_trading_day"] is False
    assert s["data_date"] == "2026-11-06"
    assert "休市" in s["markets"]["A股"]["reason"]
    assert s["notes"], "休市必须留说明，避免把上一交易日价格当今天价格"
    assert "休市" in session_label(s)


def test_holiday_gap_detected():
    # 周三，但 A 股因节假日休市，行情停在周一
    wed = dt.date(2026, 11, 11)
    s = resolve_market_session(_quotes("2026-11-09", "2026-11-09", "2026-11-09"), today=wed)
    assert s["is_trading_day"] is False
    assert s["markets"]["A股"]["lag_days"] == 2
    assert s["markets"]["A股"]["is_open"] is False


def test_missing_trade_date_is_tolerated():
    q = _quotes("2026-11-09", "2026-11-09", "2026-11-07")
    q["indexes"].append({"name": "某停更指数", "region": "海外", "trade_date": None})
    s = resolve_market_session(q, today=D)
    assert s["is_trading_day"] is True
    assert "海外" not in s["markets"] or s["markets"]["海外"]["trade_date"]


def test_region_override_map():
    """接口 region 缺失时，用外部映射兜底。"""
    q = {
        "indexes": [
            {"name": "上证指数", "trade_date": "2026-11-09"},
            {"name": "恒生指数", "trade_date": "2026-11-09"},
        ],
        "stocks": [],
    }
    s = resolve_market_session(
        q, today=D, index_region_map={"上证指数": "A股", "恒生指数": "港股"}
    )
    assert s["markets"]["A股"]["is_open"] is True
    assert s["markets"]["港股"]["is_open"] is True


def test_takes_latest_date_per_market():
    """同一市场多个指数 trade_date 不一致时，取多数口径。"""
    q = {
        "indexes": [
            {"name": "上证指数", "region": "A股", "trade_date": "2026-11-09"},
            {"name": "沪深300", "region": "A股", "trade_date": "2026-11-06"},
        ],
        "stocks": [],
    }
    s = resolve_market_session(q, today=D)
    # 1:1 平票 → 取较早的（保守，宁可判休市也不把陈旧数据当今日价）
    assert s["data_date"] == "2026-11-06"
    assert s["markets"]["A股"]["is_open"] is False
    assert s["markets"]["A股"]["source_dates"] == ["2026-11-06", "2026-11-09"]
    assert "不一致" in s["markets"]["A股"]["note"]


def test_majority_vote_wins():
    """多数源认同的日期优先于单个异常源。"""
    q = {
        "indexes": [
            {"name": "上证指数", "region": "A股", "trade_date": "2026-11-09"},
            {"name": "沪深300", "region": "A股", "trade_date": "2026-11-09"},
            {"name": "上证50", "region": "A股", "trade_date": "2026-11-06"},
        ],
        "stocks": [],
    }
    s = resolve_market_session(q, today=D)
    assert s["data_date"] == "2026-11-09"
    assert s["is_trading_day"] is True


def test_us_market_not_open_when_data_is_previous_session():
    """真实场景：美股上周五收盘，东财 f297=20261106，不能判成今天开市。

    这是新浪/腾讯口径（北京时间次日清晨）会导致的经典误判 ——
    如果用「取最大日期」，2026-11-07 的时间戳会被当成今天。
    """
    q = {
        "indexes": [
            {"name": "道琼斯工业指数", "region": "美股", "trade_date": "2026-11-06"},
            {"name": "纳斯达克综合指数", "region": "美股", "trade_date": "2026-11-06"},
            {"name": "标普500", "region": "美股", "trade_date": "2026-11-06"},
        ],
        "stocks": [],
    }
    s = resolve_market_session(q, today=D)
    assert s["markets"]["美股"]["is_open"] is False
    assert s["markets"]["美股"]["lag_days"] == 3
    # A 股交易与否独立判定，不受美股影响
    assert s["markets"].get("A股") is None


def test_mixed_dates_same_market_flagged():
    """同一市场内部日期分歧要被记录，不能静默抹掉。"""
    q = {
        "indexes": [
            {"name": "道琼斯工业指数", "region": "美股", "trade_date": "2026-11-06"},
            {"name": "纳斯达克综合指数", "region": "美股", "trade_date": "2026-11-06"},
            {"name": "标普500", "region": "美股", "trade_date": "2026-11-09"},
        ],
        "stocks": [],
    }
    s = resolve_market_session(q, today=D)
    assert s["markets"]["美股"]["trade_date"] == "2026-11-06"
    assert "2026-11-09" in s["markets"]["美股"]["source_dates"]


def test_session_label_without_session():
    assert "未知" in session_label(None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
