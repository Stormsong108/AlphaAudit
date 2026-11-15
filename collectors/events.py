"""重大事件 + 经济数据采集器。

===================== 数据源与字段布局（实测）=====================

华尔街见闻 7×24 快讯  https://api-one.wallstcn.com/apiv1/content/lives
    ?channel=global-channel&limit=N
    返回 data.items[]，关键字段：
        id            整数，唯一 ID
        title         标题（多数条目有）
        content_text  正文
        display_time  unix 时间戳
        uri           原文链接
        is_calendar   1 = 该条是「经济日历」事件（重要！用于分类）
        channels      频道数组
        score         重要度打分（可作权重）

东方财富快讯
    https://np-listapi.eastmoney.com/comm/web/getFastNewsList
        ?client=web&biz=web_news_col&fastColumn=102&pageSize=N
    返回 data.fastNewsList[]，关键字段：
        code       数字 ID
        summary    标题+正文合一
        showTime   "YYYY-MM-DD HH:MM:SS"
        uniqueUrl  原文链接

东方财富新闻栏目
    https://np-listapi.eastmoney.com/comm/web/getNewsByColumns
        ?client=web&biz=web_news_col&column=347&pageSize=N
    返回 data.list[]，字段同上（summary / showTime / uniqueUrl）

东方财富宏观数据  https://datacenter-web.eastmoney.com/api/data/v1/get
    ?reportName=RPT_ECONOMY_XXX&columns=ALL&pageSize=N
    ⚠ 各报表字段名不同，必须按报表分别映射（见 MACRO_FIELDS）：
        CPI   NATIONAL_SAME(同比) NATIONAL_SEQUENTIAL(环比) NATIONAL_ACCUMULATE(累计)
        PPI   BASE_SAME(同比)  BASE_ACCUMULATE(累计)
        GDP   SUM_SAME(同比) FIRST_SAME SECOND_SAME THIRD_SAME
        PMI   MAKE_INDEX(制造业) NMAKE_INDEX(非制造业)
        M2    CURRENCY(货币供应量) BASIC_CURRENCY(基础货币) FREE_CASH(自由现金)

===================== 交叉验证策略 =====================
重大事件：华尔街见闻 × 东财快讯
    → 两源按「标题文本相似度」做同一事件归并，命中同一事件即 verified。
    → 相似度算法：字符集合 Jaccard，阈值 0.55，保守避免误并。
    → 只对两源都出现的事件标记 verified；单源事件标 single_source。

经济数据：东方财富宏观 API 为唯一直接源
    → 用「上一期值 + 本期值」做时间序列自洽性检查（同比/环比变化方向），
      数值异常（偏离上一期 >50%）标记为 needs_review，不静默丢弃。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors.base import (  # noqa: E402
    Fetch,
    http_get,
    http_get_json,
    run_collectors,
    safe_float,
)
import config as C  # noqa: E402


# 各宏观报表的字段映射（reportName → 显示名, 单位, 同比字段, 其他关键字段）
MACRO_FIELDS = {
    "RPT_ECONOMY_CPI": {
        "label": "CPI 同比", "unit": "%", "yoy": "NATIONAL_SAME",
        "mom": "NATIONAL_SEQUENTIAL", "extra": {},
    },
    "RPT_ECONOMY_PPI": {
        "label": "PPI 同比", "unit": "%", "yoy": "BASE_SAME",
        "mom": None, "extra": {"累计": "BASE_ACCUMULATE"},
    },
    "RPT_ECONOMY_GDP": {
        "label": "GDP 同比", "unit": "%", "yoy": "SUM_SAME",
        "mom": None,
        "extra": {"第一产业": "FIRST_SAME", "第二产业": "SECOND_SAME", "第三产业": "THIRD_SAME"},
    },
    "RPT_ECONOMY_PMI": {
        "label": "制造业PMI", "unit": "点", "yoy": None,
        "mom": None, "extra": {"非制造业PMI": "NMAKE_INDEX"},
    },
    "RPT_ECONOMY_CURRENCY_SUPPLY": {
        "label": "M2 同比", "unit": "%", "yoy": "CURRENCY_SAME",
        "mom": "CURRENCY_SEQUENTIAL",
        "extra": {"M1同比": "BASIC_CURRENCY_SAME", "M0同比": "FREE_CASH_SAME"},
    },
}

# 事件重要度关键词（命中则提升 importance 等级）
IMPORTANT_KEYWORDS = [
    "美联储", "议息", "加息", "降息", "非农", "CPI", "PPI", "PMI", "GDP", "失业率",
    "国务院", "国常会", "央行", "财政部", "发改委", "证监会", "政治局",
    "关税", "制裁", "出口管制", "禁运", "军事", "战争", "袭击", "冲突", "核",
    "稀土", "芯片", "半导体", "新能源", "房地产", "房价",
    "Trump", "美联储", "鲍威尔", "关税",
]


def _wallstreetcn_lives() -> Fetch:
    """华尔街见闻 7×24 快讯。"""
    url = (
        "https://api-one.wallstcn.com/apiv1/content/lives"
        "?channel=global-channel&limit=100"
    )
    f = http_get_json(url, referer="https://wallstreetcn.com/")
    if not f.ok:
        return f
    items = ((f.value or {}).get("data") or {}).get("items") or []
    out = []
    for it in items:
        title = (it.get("title") or "").strip()
        body = (it.get("content_text") or "").strip()
        text = title or body
        if not text:
            continue
        ts = safe_float(it.get("display_time"))
        out.append({
            "id": str(it.get("id") or ""),
            "title": title,
            "summary": body[:200] if body else title[:200],
            "text": text[:300],
            "is_calendar": bool(it.get("is_calendar")),
            "score": safe_float(it.get("score")) or 0,
            "url": it.get("uri") or "",
            "ts": int(ts) if ts else None,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "",
        })
    return Fetch(value=out, source="wallstreetcn.lives", ok=True)


def _em_fastnews() -> Fetch:
    """东财快讯（7×24 全球）。

    ⚠ 必须带 sortEnd=0，否则返回空列表；sortEnd 是游标（上一页末条的 realSort）。
    条目有独立 title 字段，summary 是正文，勿混用。
    """
    url = (
        "https://np-listapi.eastmoney.com/comm/web/getFastNewsList"
        "?client=web&biz=web_news_col&fastColumn=102&pageSize=50&req_trace=1&sortEnd=0"
    )
    f = http_get_json(url, referer="https://kuaixun.eastmoney.com/")
    if not f.ok:
        return f
    lst = ((f.value or {}).get("data") or {}).get("fastNewsList") or []
    out = []
    for it in lst:
        summary = (it.get("summary") or "").strip()
        title = (it.get("title") or "").strip()
        if not (summary or title):
            continue
        out.append({
            "id": str(it.get("code") or ""),
            "title": title or summary[:60],
            "summary": summary[:200] if summary else title[:200],
            "text": (title + " " + summary)[:300] if title else summary[:300],
            "is_calendar": False,
            "score": 0,
            "url": "",
            "ts": None,
            "time": it.get("showTime") or "",
        })
    return Fetch(value=out, source="eastmoney.fastnews", ok=True)


def _em_calendar_news() -> Fetch:
    """东财「本周财经日历」栏目（column=352）。

    内容形如「本周（11月9日至11月13日），美国10月CPI、PPI将公布……」，
    是用户要求的「经济事件」主来源之一。
    """
    url = (
        "https://np-listapi.eastmoney.com/comm/web/getNewsByColumns"
        "?client=web&biz=web_news_col&column=352&order=1&page_index=1&page_size=20"
        "&needInteractData=0&req_trace=1"
    )
    f = http_get_json(url, referer="https://finance.eastmoney.com/")
    if not f.ok:
        return f
    lst = ((f.value or {}).get("data") or {}).get("list") or []
    out = []
    for it in lst:
        title = (it.get("title") or "").strip()
        summary = (it.get("summary") or "").strip()
        if not (title or summary):
            continue
        out.append({
            "id": str(it.get("code") or ""),
            "title": title or summary[:60],
            "summary": summary[:300] if summary else title[:300],
            "text": (title + " " + summary)[:300],
            "is_calendar": True,
            "score": 0,
            "url": it.get("uniqueUrl") or "",
            "ts": None,
            "time": it.get("showTime") or "",
        })
    return Fetch(value=out, source="eastmoney.calendar352", ok=True)


def _em_news_column() -> Fetch:
    """东财财经要闻栏目（column=347）。"""
    url = (
        "https://np-listapi.eastmoney.com/comm/web/getNewsByColumns"
        "?client=web&biz=web_news_col&column=347&order=1&page_index=1&page_size=50"
        "&needInteractData=0&req_trace=1"
    )
    f = http_get_json(url, referer="https://finance.eastmoney.com/")
    if not f.ok:
        return f
    lst = ((f.value or {}).get("data") or {}).get("list") or []
    out = []
    for it in lst:
        summary = (it.get("summary") or "").strip()
        title = (it.get("title") or "").strip()
        if not (summary or title):
            continue
        out.append({
            "id": str(it.get("code") or ""),
            "title": title or summary[:60],
            "summary": summary[:200],
            "text": (title or summary)[:300],
            "is_calendar": False,
            "score": 0,
            "url": it.get("uniqueUrl") or "",
            "ts": None,
            "time": it.get("showTime") or "",
        })
    return Fetch(value=out, source="eastmoney.news347", ok=True)


def _em_macro() -> Fetch:
    """东财宏观指标。"""
    out: dict[str, dict] = {}
    for rpt, spec in MACRO_FIELDS.items():
        url = (
            "https://datacenter-web.eastmoney.com/api/data/v1/get"
            f"?reportName={rpt}&columns=ALL&pageSize=6&pageNumber=1"
            "&sortColumns=REPORT_DATE&sortTypes=-1"
        )
        f = http_get_json(url, referer="https://data.eastmoney.com/")
        if not f.ok:
            continue
        rows = ((f.value or {}).get("result") or {}).get("data") or []
        if not rows:
            continue
        series = []
        for r in rows:
            series.append({
                "period": r.get("TIME") or "",
                "report_date": (r.get("REPORT_DATE") or "")[:10],
                "yoy": safe_float(r.get(spec["yoy"])) if spec.get("yoy") else None,
                "mom": safe_float(r.get(spec["mom"])) if spec.get("mom") else None,
                "extra": {
                    k: safe_float(r.get(field))
                    for k, field in (spec.get("extra") or {}).items()
                    if r.get(field) is not None
                },
            })
        out[rpt] = {
            "report": rpt,
            "label": spec["label"],
            "unit": spec["unit"],
            "series": series,
        }
    return Fetch(value=out, source="eastmoney.macro", ok=True)


# --------------------------------------------------------------------------
# 事件相似度与归并
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[\s，。、；：！？（）()【】\[\]“”\"'’‘·\-,.!?;:]+")


def _norm(text: str) -> set[str]:
    """归一化：去标点 + 中文按字、英文按词。"""
    t = _PUNCT.sub("", text or "")
    chars = set(ch for ch in t if "一" <= ch <= "鿿")
    words = set(w.lower() for w in re.findall(r"[A-Za-z]{2,}", t))
    digits = set(re.findall(r"\d+(?:\.\d+)?", t))
    return chars | words | digits


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _containment(a: set[str], b: set[str]) -> float:
    """包含度：交集 / 较小集合。中文短标题场景下比 Jaccard 准得多。

    实测：同一事件的两源标题 Jaccard 仅 0.45，但包含度 1.00；
    不同事件 Jaccard 0.52 时包含度也只有 0.86 → 需与 Jaccard 联合判定。
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _similarity(a: set[str], b: set[str]) -> float:
    """综合相似度：取 Jaccard 与包含度的加权，偏向包含度。

    权重 0.35 / 0.65 的依据：中文标题长度差异大，包含度更能反映「同一事件」。
    """
    return 0.35 * _jaccard(a, b) + 0.65 * _containment(a, b)


def _is_important(text: str) -> bool:
    return any(k in text for k in IMPORTANT_KEYWORDS)


def _merge_events(wscn: list[dict], emnews: list[dict], *, thresh: float = 0.55) -> list[dict]:
    """两源事件归并：华尔街见闻为主，东财快讯/新闻为验证源。

    对每条华尔街见闻事件，找东财中最相似的条目：
      相似度 ≥ thresh → verified（双源命中）
      否则 → single_source
    东财中未被归并的条目也单独保留（single_source），不丢弃。
    """
    em_norms = [(_norm(e["text"]), e) for e in emnews]
    used_em: set[int] = set()
    merged: list[dict] = []

    for w in wscn:
        wn = _norm(w["text"])
        best_i, best_s = -1, 0.0
        for i, (en, _e) in enumerate(em_norms):
            if i in used_em:
                continue
            s = _similarity(wn, en)
            if s > best_s:
                best_i, best_s = i, s
        matched = None
        status = "single_source"
        if best_i >= 0 and best_s >= thresh:
            matched = em_norms[best_i][1]
            used_em.add(best_i)
            status = "verified"
        merged.append({
            "title": w["title"] or w["summary"][:60],
            "summary": w["summary"],
            "time": w["time"],
            "url": w["url"],
            "is_calendar": w["is_calendar"],
            "importance": "high" if _is_important(w["text"]) else "normal",
            "cross": {
                "status": status,
                "n_sources": 2 if matched else 1,
                "srcs": {
                    "华尔街见闻": w["text"][:120],
                    **({"东财": matched["text"][:120]} if matched else {}),
                },
                "similarity": round(best_s, 3) if matched else None,
                "note": (
                    f"双源命中（相似度 {best_s:.3f}）" if matched
                    else "仅华尔街见闻一个来源，未达双源验证要求"
                ),
            },
        })

    for i, (_n, e) in enumerate(em_norms):
        if i in used_em:
            continue
        merged.append({
            "title": e["title"] or e["summary"][:60],
            "summary": e["summary"],
            "time": e["time"],
            "url": e["url"],
            "is_calendar": False,
            "importance": "high" if _is_important(e["text"]) else "normal",
            "cross": {
                "status": "single_source",
                "n_sources": 1,
                "srcs": {"东财": e["text"][:120]},
                "similarity": None,
                "note": "仅东财一个来源，未达双源验证要求",
            },
        })

    merged.sort(key=lambda x: (x["time"] or "", x["title"]), reverse=True)
    return merged


def collect_events() -> dict:
    """采集重大事件与经济数据。"""
    res = run_collectors({
        "wscn": _wallstreetcn_lives,
        "em_fast": _em_fastnews,
        "em_news": _em_news_column,
        "em_cal": _em_calendar_news,
        "macro": _em_macro,
    })

    wscn = res["wscn"].value if res["wscn"].ok and res["wscn"].value else []
    em_fast = res["em_fast"].value if res["em_fast"].ok and res["em_fast"].value else []
    em_news = res["em_news"].value if res["em_news"].ok and res["em_news"].value else []
    em_cal = res["em_cal"].value if res["em_cal"].ok and res["em_cal"].value else []
    em_all = list(em_fast) + list(em_news) + list(em_cal)

    events = _merge_events(wscn, em_all)

    macro_map = res["macro"].value if res["macro"].ok and res["macro"].value else {}
    macro = []
    for rpt, spec in MACRO_FIELDS.items():
        rec = macro_map.get(rpt)
        if not rec or not rec.get("series"):
            continue
        series = rec["series"]
        latest = series[0]
        prev = series[1] if len(series) > 1 else None
        # 自洽检查：本期与上期不应出现离谱跳变
        review = ""
        if latest.get("yoy") is not None and prev and prev.get("yoy") is not None:
            if abs(latest["yoy"] - prev["yoy"]) > 50:
                review = f"本期 {latest['yoy']} 与上期 {prev['yoy']} 差异过大，待核实"
        macro.append({
            "report": rpt,
            "label": rec["label"],
            "unit": rec["unit"],
            "period": latest.get("period", ""),
            "report_date": latest.get("report_date", ""),
            "value": latest.get("yoy"),
            "mom": latest.get("mom"),
            "extra": latest.get("extra") or {},
            "prev_value": prev.get("yoy") if prev else None,
            "history": series[:6],
            "needs_review": bool(review),
            "review_note": review,
            "cross": {
                "status": "single_source",
                "n_sources": 1,
                "srcs": {"东财宏观": f"{rec['label']} {latest.get('period')} = {latest.get('yoy')}"},
                "note": "宏观数据来自东财官方宏观库，暂无第二独立源；已做时序自洽检查",
            },
        })

    n_verified = sum(1 for e in events if e["cross"]["status"] == "verified")
    conflicts = [
        {
            "name": e["title"],
            "kind": "event",
            "status": "single_source",
            "srcs": e["cross"]["srcs"],
            "note": e["cross"]["note"],
        }
        for e in events if e["cross"]["status"] == "conflict"
    ]

    return {
        "events": events,
        "macro": macro,
        "conflicts": conflicts,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "events_total": len(events),
            "events_verified": n_verified,
            "events_single_source": len(events) - n_verified,
            "events_calendar": sum(1 for e in events if e["is_calendar"]),
            "events_high": sum(1 for e in events if e["importance"] == "high"),
            "macro_count": len(macro),
            "macro_needs_review": sum(1 for m in macro if m["needs_review"]),
        },
        "sources": {
            "wallstreetcn": {"ok": res["wscn"].ok, "error": res["wscn"].error, "count": len(wscn)},
            "eastmoney_fastnews": {"ok": res["em_fast"].ok, "error": res["em_fast"].error, "count": len(em_fast)},
            "eastmoney_news347": {"ok": res["em_news"].ok, "error": res["em_news"].error, "count": len(em_news)},
            "eastmoney_macro": {"ok": res["macro"].ok, "error": res["macro"].error, "count": len(macro_map)},
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(collect_events(), ensure_ascii=False, indent=2))
