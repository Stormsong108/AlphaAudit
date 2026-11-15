"""交易日判定。

设计原则：**不硬编码节假日表**。
节假日安排每年由交易所公告、且常有临时调整，静态表必然过期。
这里改为从行情接口自身返回的 `trade_date` 反推：

  - 行情接口报的 trade_date == 今天 → 今天该市场在交易
  - 行情接口报的 trade_date < 今天  → 今天该市场休市
    （无论原因是周末、法定节假日还是临时休市，接口比静态表更准）

判定结果同时用于：
  1. 日报正文/HTML 里标注「今日是否交易日、数据对应哪一交易日」，避免
     读者把上一交易日的收盘价误当成今天的价格；
  2. 快照里固化 market_session，供复盘时对齐价格与日期。

对每个市场单独判定，因为 A 股 / 港股 / 美股的日历互不相同。
"""
from __future__ import annotations

import datetime as _dt
from typing import Any


def _parse_date(v: Any) -> _dt.date | None:
    """把接口返回的日期字符串解析成 date；解析不了返回 None。"""
    if not v or not isinstance(v, str):
        return None
    s = v.strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _weekday_cn(d: _dt.date) -> str:
    return "一二三四五六日"[d.weekday()]


def resolve_market_session(
    quotes: dict,
    *,
    today: _dt.date | None = None,
    index_region_map: dict[str, str] | None = None,
) -> dict:
    """从行情采集结果反推各市场的交易状态。

    参数
    ----
    quotes: collect_quotes() 的返回值
    today: 判断的基准日期，默认取本机当天
    index_region_map: {指数名: 区域}，用于把指数归到具体市场；
                       缺省时从 quotes["indexes"][i]["region"] 读取

    返回
    ----
    {
      "today": "2026-11-09",
      "is_trading_day": True,          # A 股是否交易日（日报主市场）
      "markets": {
         "A股": {"trade_date": "2026-11-09", "is_open": True, "reason": "..."},
         ...
      },
      "data_date": "2026-11-09",       # 行情对应的交易日
      "notes": ["..."],
    }
    """
    today = today or _dt.date.today()
    indexes = quotes.get("indexes") or []

    # 区域归属：优先用传入映射（稳定，不依赖接口字段），否则用接口的 region
    regions: dict[str, str] = {}
    if index_region_map:
        regions.update(index_region_map)
    for row in indexes:
        name = row.get("name")
        if name and name not in regions and row.get("region"):
            regions[name] = row["region"]

    # 每个区域取「交易日」：用**一致投票**而不是最大值。
    # 理由：不同来源对「美股这一周最后一个交易日」的日期口径可能不同
    # （北京时间 vs 美东时间），直接取最大值会让一个错源把整个市场
    # 判定成「今天开市」。取出现次数最多的日期；平票时取较早的
    # （保守：宁可判休市，也不要把陈旧数据当成今日价）。
    votes: dict[str, dict[str, int]] = {}
    raw_dates: dict[str, list[str]] = {}
    for row in indexes:
        name = row.get("name") or ""
        region = regions.get(name) or row.get("region") or "未知市场"
        d = _parse_date(row.get("trade_date"))
        if d is None:
            continue
        iso = d.isoformat()
        votes.setdefault(region, {})
        votes[region][iso] = votes[region].get(iso, 0) + 1
        raw_dates.setdefault(region, []).append(iso)

    latest: dict[str, _dt.date] = {}
    raw_date: dict[str, str] = {}
    disagree: dict[str, list[str]] = {}
    for region, tally in votes.items():
        best = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
        top_count = best[0][1]
        winners = [d for d, c in best if c == top_count]
        chosen = winners[0]  # 平票取较早的
        latest[region] = _dt.date.fromisoformat(chosen)
        raw_date[region] = chosen
        if len(tally) > 1:
            disagree[region] = sorted(tally)

    markets: dict[str, dict] = {}
    for region, d in sorted(latest.items()):
        is_open = d == today
        lag = (today - d).days
        if is_open:
            reason = f"今日交易日（数据日期 {d.isoformat()}）"
        elif today.weekday() >= 5 and lag <= 3:
            reason = (
                f"今日为周{_weekday_cn(today)}，该市场休市；"
                f"最近交易日 {d.isoformat()}（{lag} 天前）"
            )
        else:
            reason = (
                f"该市场今日休市（周末或法定节假日）；"
                f"最近交易日 {d.isoformat()}（{lag} 天前）"
            )
        markets[region] = {
            "trade_date": d.isoformat(),
            "is_open": is_open,
            "lag_days": lag,
            "reason": reason,
        }
        if region in disagree:
            markets[region]["source_dates"] = disagree[region]
            markets[region]["note"] = (
                f"该市场各指数报告的交易日不一致 {disagree[region]}，"
                f"按多数口径取 {d.isoformat()}"
            )

    cn = markets.get("A股")
    is_trading_day = bool(cn and cn["is_open"])

    notes: list[str] = []
    if cn and not cn["is_open"]:
        notes.append(
            f"今日 A 股休市，全部 A 股价格为最近交易日 {cn['trade_date']} 收盘价，"
            "不可当作今日价格解读。"
        )
    for region, m in markets.items():
        if region != "A股" and not m["is_open"]:
            notes.append(f"{region} 今日休市，价格对应 {m['trade_date']}。")

    return {
        "today": today.isoformat(),
        "today_weekday": f"周{_weekday_cn(today)}",
        "is_trading_day": is_trading_day,
        "markets": markets,
        "data_date": cn["trade_date"] if cn else "",
        "notes": notes,
    }


def session_label(session: dict | None) -> str:
    """给日报用的一行状态摘要。"""
    if not session:
        return "交易日状态未知"
    today = session.get("today", "")
    wd = session.get("today_weekday", "")
    if session.get("is_trading_day"):
        return f"{today}（{wd}）· A 股交易日"
    cn = (session.get("markets") or {}).get("A股") or {}
    return (
        f"{today}（{wd}）· A 股休市，行情对应 {cn.get('trade_date', '—')} 收盘"
    )
