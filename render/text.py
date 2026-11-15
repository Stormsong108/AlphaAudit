"""日报渲染：把三个采集器的结果拼成飞书可读文本。

格式遵循宋老师偏好（feishu-user-communication skill）：
纯文字 + 缩进 2 空格 + 关键数字加粗，不用 emoji / 表格 / 代码块。
"""
from __future__ import annotations

import time
from typing import Any

from trading_calendar import session_label

BAR = "─" * 28


# ---------------------------------------------------------------- 小工具
def _f(v: Any, nd: int = 2, dash: str = "—") -> str:
    """格式化浮点，None → 占位符。"""
    if v is None:
        return dash
    if isinstance(v, (int, float)):
        return f"{v:,.{nd}f}"
    return str(v)


def _pct(v: Any, nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:+.{nd}f}%"


def _bold(s: str) -> str:
    return f"**{s}**"


def _clean_title(s: Any, limit: int = 58) -> str:
    """标题清洗：压掉换行（原始摘要里带 \\n 会把版面撑碎），再截断。"""
    if not s:
        return "—"
    t = " ".join(str(s).split())  # 所有空白（含换行）压成单空格
    t = t.replace("**", "")  # 防止破坏 markdown 加粗语法
    if len(t) > limit:
        t = t[: limit - 1] + "…"
    return t


def _region_block(title: str, rows: list[dict]) -> list[str]:
    """一个指数/品种分区的行。"""
    if not rows:
        return []
    out = [title + ":"]
    for r in rows:
        name = r.get("name") or "—"
        price = _f(r.get("price"))
        chg = _bold(_pct(r.get("pct")))
        line = f"  {name} {price} {chg}"
        # PE / PB 有值时补在后面
        extra = []
        if r.get("pe") is not None:
            extra.append(f"PE {_f(r.get('pe'))}")
        if r.get("pb") is not None:
            extra.append(f"PB {_f(r.get('pb'))}")
        if extra:
            line += "  " + " ".join(extra)
        out.append(line)
    out.append("")
    return out


def _sort_by_pct(rows: list[dict], desc: bool = True) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (r.get("pct") is None, -(r.get("pct") or 0) if desc else (r.get("pct") or 0)),
    )


# ---------------------------------------------------------------- 主渲染
def render_daily(
    quotes: dict,
    events: dict,
    futures: dict,
    *,
    products: dict | None = None,
    advice: dict | None = None,
    review: dict | None = None,
    session: dict | None = None,
    title: str = "财经日报",
    now: str | None = None,
) -> str:
    """产出日报正文（飞书 post / markdown 可直接用）。"""
    now = now or time.strftime("%Y-%m-%d %H:%M")
    L: list[str] = []

    L.append(f"{title} {now}")
    L.append("")

    # ---------------------------------------------------- 0. 交易日状态
    if session:
        L.append(f"交易日状态: {session_label(session)}")
        for region, m in (session.get("markets") or {}).items():
            L.append(f"  {region}: {m.get('trade_date', '—')} · "
                     f"{'开市' if m.get('is_open') else '休市'}")
        for n in session.get("notes") or []:
            L.append(f"  说明: {n}")
        L.append("")

    # ---------------------------------------------------- 1. 大盘概览
    idx = quotes.get("indexes") or []
    if idx:
        L.append("大盘概览:")
        L.append("")
        for r in idx:
            name = r.get("name") or "—"
            region = r.get("region") or ""
            price = _f(r.get("price"))
            chg = _bold(_pct(r.get("pct")))
            amt = r.get("amount")
            tail = f"  成交 {amt/1e8:,.0f}亿" if amt else ""
            L.append(f"  {name} {price} {chg}{tail}")
        L.append("")

    # ---------------------------------------------------- 2. 自选股
    stocks = quotes.get("stocks") or []
    by_theme: dict[str, list[dict]] = {}
    for s in stocks:
        by_theme.setdefault(s.get("theme") or "其他", []).append(s)

    if by_theme:
        L.append("自选股池:")
        L.append("")
        for theme in ("大盘蓝筹", "科技成长", "红利低波", "港股互联网"):
            rows = by_theme.get(theme)
            if not rows:
                continue
            L.append(f"  {theme}")
            for r in _sort_by_pct(rows):
                name = r.get("name") or "—"
                price = _f(r.get("price"))
                chg = _bold(_pct(r.get("pct")))
                sector = r.get("sector") or ""
                L.append(f"    {name} {price} {chg}  {sector}")
            L.append("")

    # ---------------------------------------------------- 3. 期货
    cn = futures.get("cn_futures") or []
    gl = futures.get("global_futures") or []
    if cn or gl:
        L.append("期货:")
        L.append("")
        if cn:
            L.append("  国内")
            for r in _sort_by_pct(cn):
                L.append(f"    {r.get('name')} {_f(r.get('price'))} {_bold(_pct(r.get('pct')))}")
            L.append("")
        if gl:
            L.append("  外盘")
            for r in _sort_by_pct(gl):
                L.append(f"    {r.get('name')} {_f(r.get('price'))} {_bold(_pct(r.get('pct')))}")
            L.append("")

    # ---------------------------------------------------- 4. 外汇
    fx = futures.get("fx") or []
    mid = futures.get("fx_mid")
    if fx or mid:
        L.append("外汇:")
        L.append("")
        if mid:
            L.append(f"  {mid.get('name')} {_f(mid.get('price'), 4)}")
        for r in fx:
            L.append(f"  {r.get('name')} {_f(r.get('price'), 4)} {_bold(_pct(r.get('pct')))}")
        L.append("")

    # ---------------------------------------------------- 5. 宏观
    macro = events.get("macro") or []
    if macro:
        L.append("宏观数据:")
        L.append("")
        for m in macro:
            label = m.get("label") or m.get("report")
            unit = m.get("unit") or ""
            val = m.get("value")
            mom = m.get("mom")
            period = m.get("period") or ""
            if val is None:
                line = f"  {label} 待更新"
            else:
                line = f"  {label} {_f(val)}{unit}"
                if mom is not None:
                    line += f"  环比 {_pct(mom)}"
            if period:
                line += f"  ({period})"
            L.append(line)
        L.append("")

    # ---------------------------------------------------- 6. 理财产品
    prods = (products or {}).get("products") or []
    if prods:
        money = [p for p in prods if p.get("kind") == "money"]
        nav = [p for p in prods if p.get("kind") != "money"]
        L.append("理财产品:")
        L.append("")
        if money:
            L.append("  货币基金")
            for p in money:
                sdy = p.get("seven_day_yield")
                mc = p.get("million_copies")
                line = f"    {p.get('name')}"
                if sdy is not None:
                    line += f"  7日年化 {_f(sdy)}%"
                if mc is not None:
                    line += f"  万份收益 {_f(mc, 4)}"
                if p.get("ret_1y") is not None:
                    line += f"  近1年 {_pct(p.get('ret_1y'))}"
                L.append(line)
            L.append("")
        if nav:
            L.append("  其他基金")
            for p in nav:
                line = f"    {p.get('name')} 净值 {_f(p.get('nav'), 4)}"
                if p.get("pct") is not None:
                    line += f"  {_bold(_pct(p.get('pct')))}"
                if p.get("ret_1y") is not None:
                    line += f"  近1年 {_pct(p.get('ret_1y'))}"
                L.append(line)
            L.append("")

    # ---------------------------------------------------- 7. 今日事件
    evs = events.get("events") or []
    high = [e for e in evs if e.get("importance") == "high"]
    verified = [e for e in high if (e.get("cross") or {}).get("status") == "verified"]
    L.append("重点事件:")
    L.append("")
    # 双源验证的排前面，其余按时间倒序
    verified.sort(key=lambda e: e.get("time") or "", reverse=True)
    for e in verified:
        t = (e.get("time") or "")[5:16]
        L.append(f"  [{t}] {_clean_title(e.get('title'))}")
    L.append("")
    # 其余重要快讯：去重后限量，避免刷屏
    seen: set[str] = set()
    others: list[dict] = []
    for e in sorted(high, key=lambda e: e.get("time") or "", reverse=True):
        if e in verified:
            continue
        key = _clean_title(e.get("title"), 40)
        if key in seen:
            continue
        seen.add(key)
        others.append(e)
    if others:
        L.append(f"  其他重要快讯（{len(others)} 条，显示前 12）:")
        for e in others[:12]:
            t = (e.get("time") or "")[5:16]
            L.append(f"  [{t}] {_clean_title(e.get('title'))}")
        L.append("")

    # ---------------------------------------------------- 7.5 模拟组合建议
    if advice:
        alloc = advice.get("allocation") or {}
        st = advice.get("market_state") or {}
        L.append("模拟组合建议（研究性，不构成投资建议）:")
        L.append("")
        L.append(f"  大盘温度 {_f(st.get('temperature'), 1)}（{st.get('label')}）")
        L.append(
            f"  权益 {_f(alloc.get('equity_pct'), 1)}%"
            f"  理财 {_f(alloc.get('product_pct'), 1)}%"
            f"  现金 {_f(alloc.get('cash_pct'), 1)}%"
        )
        L.append("")
        for p_ in advice.get("portfolio") or []:
            if p_.get("kind") == "stock":
                tp = " / ".join(
                    f"{t_['gain_pct']:+.0f}%→{t_['price']}"
                    for t_ in (p_.get("targets") or [])
                )
                line = f"    {p_.get('name')} 权重 {_f(p_.get('weight_pct'), 1)}%"
                line += f"  现价 {_f(p_.get('price'))}"
                if p_.get("stop_price") is not None:
                    line += f"  止损 {_f(p_.get('stop_pct'), 0)}%→{_f(p_.get('stop_price'))}"
                if tp:
                    line += f"  目标 {tp}"
                if p_.get("call"):
                    line += f"  {p_.get('call')}"
                L.append(line)
            else:
                L.append(
                    f"    {p_.get('name')} 权重 {_f(p_.get('weight_pct'), 1)}%"
                    f"  {p_.get('category')}"
                )
        L.append("")
        L.append("  以上为研究性模拟组合，不构成投资建议，未执行任何真实交易。")
        L.append("")

    # ---------------------------------------------------- 7.6 历史判断复盘
    if review and review.get("available"):
        rst = review.get("stats") or {}
        L.append("历史判断与模拟建议复盘:")
        L.append("")
        L.append(f"  {review.get('note')}")
        if rst.get("hit_rate") is not None:
            L.append(
                f"  方向命中率 {_f(rst.get('hit_rate'), 4)}"
                f"（{rst.get('hits')}/{rst.get('judged')}）"
            )
        if rst.get("stop_trigger_rate") is not None:
            L.append(f"  止损触发率 {_f(rst.get('stop_trigger_rate'), 4)}")
        if rst.get("target_hit_rate") is not None:
            L.append(f"  目标达成率 {_f(rst.get('target_hit_rate'), 4)}")
        if rst.get("stability_avg_jaccard") is not None:
            L.append(f"  建议稳定性 Jaccard {_f(rst.get('stability_avg_jaccard'), 4)}")
        for e in (review.get("evaluations") or [])[-10:]:
            L.append(
                f"    {e.get('day')} {e.get('name')} {e.get('call')}"
                f"  参考 {_f(e.get('entry'))} 实际 {_f(e.get('exit'))}"
                f" {_pct(e.get('ret_pct'))}"
            )
        L.append("")

    # ---------------------------------------------------- 8. 数据核验
    stats = events.get("stats") or {}
    srcs = events.get("sources") or {}
    qsrcs = quotes.get("sources") or {}
    psrcs = (products or {}).get("sources") or {}

    def _cnt(d: dict) -> str:
        if not isinstance(d, dict):
            return "—"
        ok = sum(1 for v in d.values() if isinstance(v, dict) and v.get("ok"))
        return f"{ok}/{len(d)}"

    conflicts = (
        (quotes.get("conflicts") or [])
        + (futures.get("conflicts") or [])
        + ((products or {}).get("conflicts") or [])
    )
    prods_all = (products or {}).get("products") or []
    prod_verified = sum(
        1 for p in prods_all if (p.get("cross") or {}).get("status") == "verified"
    )
    L.append("数据核验:")
    L.append("")
    L.append(f"  行情源 {_cnt(qsrcs)}  事件源 {_cnt(srcs)}  产品源 {_cnt(psrcs)}")
    L.append(f"  事件 {stats.get('events_total', 0)} 条，双源验证 {stats.get('events_verified', 0)} 条，重要 {stats.get('events_high', 0)} 条")
    if prods_all:
        L.append(
            f"  理财 {len(prods_all)} 只，净值双源验证 {prod_verified} 只"
            f"（货币基金无单位净值，只标单源）"
        )
    if conflicts:
        L.append(f"  口径冲突 {len(conflicts)} 处（已标注，未静默取舍）:")
        for c in conflicts[:3]:
            L.append(f"    {c.get('name')}: {c.get('note')}")
    else:
        L.append("  无口径冲突")
    L.append("")
    L.append("本报告数据来自公开行情接口，双源交叉核验；所有交易内容均为模拟研究，不构成投资建议。")

    return "\n".join(L).rstrip() + "\n"


def render_html(
    quotes: dict,
    events: dict,
    futures: dict,
    *,
    products: dict | None = None,
    title: str = "财经日报",
    now: str | None = None,
) -> str:
    """归档用 HTML（深色主题，单文件自包含）。"""
    now = now or time.strftime("%Y-%m-%d %H:%M")
    text = render_daily(quotes, events, futures, products=products, title=title, now=now)
    body = text.replace("**", "")
    esc = (
        body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    pre = esc.replace("  ", "&nbsp;&nbsp;")
    return (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>"
        "body{background:#0b0e14;color:#e6eaf2;font-family:-apple-system,'PingFang SC',sans-serif;"
        "padding:32px;line-height:1.85;max-width:900px;margin:0 auto}"
        "pre{white-space:pre-wrap;word-break:break-word;font-family:inherit;font-size:15px;margin:0}"
        "</style></head><body><pre>" + pre + "</pre></body></html>"
    )
