"""理财产品采集：基金池（货币 / 债 / 混合 / 指数 / QDII）。

三个数据源，职责分明：
  1. 东财 pingzhongdata（fund.eastmoney.com/pingzhongdata/<code>.js）
     —— 阶段收益率（近1年/6月/3月/1月）+ 货币基金的 7 日年化、每万份收益
  2. 东财 f10/lsjz（api.fund.eastmoney.com/f10/lsjz）
     —— 非货币基金的最新单位净值、累计净值、日涨跌、申赎状态
  3. 新浪基金（stock.finance.sina.com.cn/fundInfo/api/openapi.php）
     —— 非货币基金净值的**独立第二源**，用于交叉验证（口径同东财）

货币基金没有"单位净值"概念（东财 lsjz 返回的是每万份收益，FundType=005），
所以货币基金只走 pingzhongdata，新浪不覆盖 → 标 single_source。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from collectors.base import Fetch, http_get, http_get_json, run_collectors, safe_float

import config

# 新浪基金净值接口（与东财完全独立的另一家数据商）
SINA_NAV_URL = (
    "https://stock.finance.sina.com.cn/fundInfo/api/openapi.php/"
    "CaihuiFundInfoService.getNav?symbol={code}&date=&page=1&pageSize=3"
)

# 货币基金类别关键词：这类基金没有单位净值
MONEY_MARKERS = ("货币基金", "货币", "现金管理")


def _is_money(category: str) -> bool:
    return any(m in (category or "") for m in MONEY_MARKERS)


# ------------------------------------------------------------------ 东财 pingzhongdata
def _parse_pingzhong(text: str) -> dict[str, Any]:
    """解析 pingzhongdata JS 变量。只取需要的字段，不执行 JS。"""
    out: dict[str, Any] = {}

    m = re.search(r'var fS_name\s*=\s*"([^"]*)"', text)
    if m:
        out["name"] = m.group(1)

    # 阶段收益率（百分数，%）
    for key in ("syl_1n", "syl_6y", "syl_3y", "syl_1y"):
        mm = re.search(r'var %s\s*=\s*"([-\d.]*)"' % key, text)
        if mm and mm.group(1) != "":
            out[key] = safe_float(mm.group(1))

    # 7 日年化（货币基金核心指标）
    mm = re.search(r"Data_sevenDaysYearIncome\s*=\s*\[(.*?)\];", text, re.S)
    if mm:
        pts = re.findall(r"\[(\d+),([-\d.]+)\]", mm.group(1))
        if pts:
            out["seven_day_yield"] = safe_float(pts[-1][1])
            out["seven_day_date"] = time.strftime(
                "%Y-%m-%d", time.localtime(int(pts[-1][0]) / 1000)
            )

    # 每万份收益（货币基金）
    mm = re.search(r"Data_millionCopiesIncome\s*=\s*\[(.*?)\];", text, re.S)
    if mm:
        pts = re.findall(r"\[(\d+),([-\d.]+)\]", mm.group(1))
        if pts:
            out["million_copies"] = safe_float(pts[-1][1])
            out["million_copies_date"] = time.strftime(
                "%Y-%m-%d", time.localtime(int(pts[-1][0]) / 1000)
            )
    return out


def _fetch_pingzhong(codes: list[str]) -> Fetch:
    out: dict[str, Any] = {}
    for code in codes:
        f = http_get(
            f"https://fund.eastmoney.com/pingzhongdata/{code}.js",
            referer="https://fund.eastmoney.com/",
            encoding="utf-8",
            timeout=20,
        )
        if not f.ok or not f.value:
            continue
        parsed = _parse_pingzhong(f.value)
        if parsed:
            out[code] = parsed
    return Fetch(value=out, source="eastmoney.pingzhongdata", ok=bool(out))


# ------------------------------------------------------------------ 东财 f10/lsjz
def _fetch_lsjz(codes: list[str]) -> Fetch:
    out: dict[str, Any] = {}
    for code in codes:
        f = http_get_json(
            f"https://api.fund.eastmoney.com/f10/lsjz?fundCode={code}&pageIndex=1&pageSize=2",
            referer="https://fundf10.eastmoney.com/",
            timeout=20,
        )
        if not f.ok or not f.value:
            continue
        data = (f.value or {}).get("Data") or {}
        rows = data.get("LSJZList") or []
        if not rows:
            continue
        r = rows[0]
        out[code] = {
            "fund_type": data.get("FundType"),
            "date": r.get("FSRQ"),
            "nav": safe_float(r.get("DWJZ")),
            "cum_nav": safe_float(r.get("LJJZ")),
            "pct": safe_float(r.get("JZZZL")),
            "buy_status": r.get("SGZT"),
            "sell_status": r.get("SHZT"),
        }
    return Fetch(value=out, source="eastmoney.f10.lsjz", ok=bool(out))


# ------------------------------------------------------------------ 新浪基金（独立第二源）
def _fetch_sina(codes: list[str]) -> Fetch:
    out: dict[str, Any] = {}
    for code in codes:
        f = http_get_json(
            SINA_NAV_URL.format(code=code),
            referer="https://finance.sina.com.cn/",
            timeout=20,
        )
        if not f.ok or not f.value:
            continue
        result = (f.value or {}).get("result") or {}
        data = (result.get("data") or {})
        rows = data.get("data") or []
        if not rows:
            continue
        r = rows[0]
        out[code] = {
            "date": (r.get("fbrq") or "")[:10],
            "nav": safe_float(r.get("jjjz")),
            "cum_nav": safe_float(r.get("ljjz")),
        }
    return Fetch(value=out, source="sina.fund.getNav", ok=bool(out))


# ------------------------------------------------------------------ 交叉验证
def _cross_nav(em: dict, sina: dict) -> dict:
    """净值双源比对。口径相同（单位净值），差异 <0.05% 视为一致。"""
    a = em.get("nav")
    b = sina.get("nav")
    srcs = {"东财": a}
    if b is not None:
        srcs["新浪"] = b
    if a is None or b is None:
        return {
            "srcs": srcs,
            "src_names": list(srcs),
            "n_sources": len([v for v in srcs.values() if v is not None]),
            "status": "single_source",
            "note": "仅一个来源，未达双源验证要求",
        }
    diff = abs(a - b) / max(abs(a), 1e-9) * 100
    ok = diff <= 0.05
    return {
        "srcs": srcs,
        "src_names": list(srcs),
        "n_sources": 2,
        "diff_pct": round(diff, 4),
        "status": "verified" if ok else "conflict",
        "note": (
            f"2 源一致（东财={a}、新浪={b}，差异 {diff:.3f}%）"
            if ok
            else f"2 源分歧 {diff:.3f}%（东财={a}、新浪={b}）"
        ),
    }


# ------------------------------------------------------------------ 主入口
def collect_products() -> dict:
    """返回 {products: [...], conflicts: [...], sources: {...}}"""
    pool = list(config.PRODUCTS)
    all_codes = [c for c, _n, _cat in pool]
    money_codes = [c for c, _n, cat in pool if _is_money(cat)]
    nav_codes = [c for c, _n, cat in pool if not _is_money(cat)]

    res = run_collectors(
        {
            "pingzhong": lambda: _fetch_pingzhong(all_codes),
            "lsjz": lambda: _fetch_lsjz(nav_codes),
            "sina": lambda: _fetch_sina(nav_codes),
        }
    )
    pz = res["pingzhong"].value if res["pingzhong"].ok else {}
    ls = res["lsjz"].value if res["lsjz"].ok else {}
    sn = res["sina"].value if res["sina"].ok else {}

    products: list[dict] = []
    conflicts: list[dict] = []

    for code, name, category in pool:
        p = pz.get(code) or {}
        money = _is_money(category)
        row: dict[str, Any] = {
            "code": code,
            "name": name,
            "category": category,
            "kind": "money" if money else "nav",
            "src_name": (p or {}).get("name") or name,
            "ret_1y": p.get("syl_1n"),
            "ret_6m": p.get("syl_6y"),
            "ret_3m": p.get("syl_3y"),
            "ret_1m": p.get("syl_1y"),
        }

        if money:
            row["nav"] = None
            row["pct"] = None
            row["seven_day_yield"] = p.get("seven_day_yield")
            row["seven_day_date"] = p.get("seven_day_date")
            row["million_copies"] = p.get("million_copies")
            row["date"] = p.get("million_copies_date") or p.get("seven_day_date")
            row["cross"] = {
                "srcs": {"东财": p.get("seven_day_yield")},
                "src_names": ["东财"],
                "n_sources": 1,
                "status": "single_source",
                "note": "货币基金 7 日年化仅天天基金披露，无独立第二源可比对",
            }
        else:
            em = ls.get(code) or {}
            sina = sn.get(code) or {}
            row["nav"] = em.get("nav")
            row["cum_nav"] = em.get("cum_nav")
            row["pct"] = em.get("pct")
            row["date"] = em.get("date") or sina.get("date")
            row["buy_status"] = em.get("buy_status")
            row["sell_status"] = em.get("sell_status")
            row["cross"] = _cross_nav(em, sina)
            if row["cross"]["status"] == "conflict":
                conflicts.append({"name": f"{name} 净值", **row["cross"]})

        products.append(row)

    return {
        "products": products,
        "conflicts": conflicts,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "pingzhong": {"ok": res["pingzhong"].ok, "source": res["pingzhong"].source},
            "lsjz": {"ok": res["lsjz"].ok, "source": res["lsjz"].source},
            "sina": {"ok": res["sina"].ok, "source": res["sina"].source},
        },
    }
