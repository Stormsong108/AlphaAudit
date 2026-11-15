"""行情采集：A股 / 港股 / 美股指数与个股。

交叉验证策略：
- 指数和个股：东财 push2 为主源，新浪 hq 为副源。
- 两个源的价格差异超过容忍度时，标记 conflict 并记录双方数值。
"""
from __future__ import annotations

import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors.base import (  # noqa: E402
    Fetch,
    em_quote,
    http_get,
    http_get_json,
    parse_sina_hq,
    run_collectors,
    safe_float,
)
import config as C  # noqa: E402


def _em_all() -> Fetch:
    """东财一次拉全部指数 + 个股。"""
    ids = [s[0] for s in C.INDEXES] + [s[0] for s in C.ALL_STOCKS]
    f = em_quote(ids)
    if f.ok and f.value is not None:
        f.value = {r["code"]: r for r in f.value if r.get("code")}
    return f


def _sina_all() -> Fetch:
    """新浪一次拉全部标的，作为副源。"""
    codes = []
    for sid, name, *_ in C.INDEXES:
        if sid.startswith("1."):
            codes.append("sh" + sid.split(".")[1])
        elif sid.startswith("0."):
            codes.append("sz" + sid.split(".")[1])
    for sid, name, *_ in C.ALL_STOCKS:
        if sid.startswith("1."):
            codes.append("sh" + sid.split(".")[1])
        elif sid.startswith("0."):
            codes.append("sz" + sid.split(".")[1])
    # 去重
    codes = list(dict.fromkeys(codes))
    url = "https://hq.sinajs.cn/list=" + ",".join(codes)
    f = http_get(url, referer="https://finance.sina.com.cn/", encoding="gbk")
    if not f.ok:
        return f
    raw = parse_sina_hq(f.value)
    out = {}
    for code, arr in raw.items():
        if len(arr) < 4:
            continue
        out[code] = {
            "name": arr[0],
            "price": safe_float(arr[3]),
            "prev_close": safe_float(arr[2]),
            "open": safe_float(arr[1]),
            "date": arr[30] if len(arr) > 30 else "",
            "time": arr[31] if len(arr) > 31 else "",
        }
    f.value = out
    return f


def _sina_hk_us() -> Fetch:
    """新浪港股/美股指数/海外指数行情。"""
    codes = []
    for sid, name, *_ in C.HK_INTERNET:
        codes.append("rt_hk" + sid.split(".")[1])
    # 海外指数副源：美股三大 + 纳指100 + 恒生 + 日经 + VIX
    codes += [
        "gb_dji",
        "gb_inx",
        "gb_ixic",
        "gb_ndx",
        "rt_hkHSI",
        "znb_NKY",
        "znb_VIX",
    ]
    url = "https://hq.sinajs.cn/list=" + ",".join(codes)
    f = http_get(url, referer="https://finance.sina.com.cn/", encoding="gbk")
    if not f.ok:
        return f
    raw = parse_sina_hq(f.value)
    out = {}
    for code, arr in raw.items():
        if len(arr) < 4:
            continue
        if code.startswith("rt_hk"):
            # rt_hk: 英文名, 中文名, 开, 昨收, 高, 低, 现价, 涨跌, 涨跌幅, ...
            raw_date = arr[17] if len(arr) > 17 else ""
            out[code] = {
                "kind": "rt_hk",
                "name": arr[1] if len(arr) > 1 else "",
                "open": safe_float(arr[2]),
                "prev_close": safe_float(arr[3]),
                "high": safe_float(arr[4]),
                "low": safe_float(arr[5]),
                "price": safe_float(arr[6]),
                "pct": safe_float(arr[8]),
                "date": raw_date.replace("/", "-") if raw_date else "",
            }
        elif code.startswith("gb_"):
            # gb_: 中文名, 现价, 涨跌幅, 时间, 涨跌额, 最高, 最低, 开盘, ...
            raw_time = arr[3] if len(arr) > 3 else ""
            out[code] = {
                "kind": "gb",
                "name": arr[0],
                "price": safe_float(arr[1]),
                "pct": safe_float(arr[2]),
                "time": raw_time,
                "date": raw_time[:10] if raw_time else "",
                "chg": safe_float(arr[4]) if len(arr) > 4 else None,
            }
        elif code.startswith("znb_"):
            # znb_: 名称, 现价, 涨跌额, 涨跌幅, 时间, 时间戳, 日期, ...
            out[code] = {
                "kind": "znb",
                "name": arr[0],
                "price": safe_float(arr[1]),
                "chg": safe_float(arr[2]),
                "pct": safe_float(arr[3]),
                "time": arr[4] if len(arr) > 4 else "",
                "date": arr[6] if len(arr) > 6 else "",
            }
    f.value = out
    return f


# 海外指数：东财 secid -> 新浪 code
OVERSEAS_SINA_MAP = {
    "100.DJIA": "gb_dji",
    "100.SPX": "gb_inx",
    "100.NDX": "gb_ixic",       # 新浪 gb_ixic 是纳斯达克综合指数
    "100.NDX100": "gb_ndx",     # 新浪 gb_ndx 是纳斯达克100
    "100.HSI": "rt_hkHSI",
    "100.N225": "znb_NKY",
    "100.VIX": "znb_VIX",
}

# 海外指数副源：腾讯行情 code（新浪之外的第二主体，时间戳新鲜）
OVERSEAS_TENCENT_MAP = {
    "100.DJIA": "usDJI",
    "100.NDX": "usIXIC",
    "100.NDX100": "usNDX",
    "100.SPX": "usINX",
    "100.HSI": "hkHSI",
}


def _tencent_quotes() -> Fetch:
    """腾讯行情：海外指数副源 + 港股个股副源。"""
    codes = list(OVERSEAS_TENCENT_MAP.values())
    for sid, *_ in C.HK_INTERNET:
        codes.append("r_hk" + sid.split(".")[1])
    url = "https://qt.gtimg.cn/q=" + ",".join(codes)
    f = http_get(url, referer="https://gu.qq.com/", encoding="gbk")
    if not f.ok:
        return f
    out = {}
    for line in (f.value or "").split("\n"):
        if '="' not in line:
            continue
        key = line.split("=")[0].replace("v_", "").strip()
        val = line.split('="')[1].rstrip('";')
        p = val.split("~")
        if len(p) < 32:
            continue
        out[key] = {
            "name": p[1],
            "price": safe_float(p[3]),
            "prev_close": safe_float(p[4]),
            "pct": safe_float(p[32]) if len(p) > 32 else None,
            "time": p[30],
            "date": p[30][:10] if p[30] else "",
        }
    return Fetch(value=out, source="tencent.gtimg", ok=True)


def _cboe_vix() -> Fetch:
    """CBOE 官方延迟报价：VIX 的权威独立源。"""
    url = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VIX.json"
    f = http_get_json(url, referer="https://www.cboe.com/")
    if not f.ok:
        return f
    d = (f.value or {}).get("data") or {}
    px = safe_float(d.get("current_price"))
    prev = safe_float(d.get("prev_day_close"))
    out = {
        "100.VIX": {
            "name": d.get("symbol") or "^VIX",
            "price": px,
            "prev_close": prev,
            "pct": safe_float(d.get("price_change_percent")),
            "time": (d.get("last_trade_time") or "").replace("T", " "),
            "date": (d.get("last_trade_time") or "")[:10],
            "error": "",
        }
    }
    return Fetch(value=out, source="cboe.delayed_quotes", ok=True)


def collect_quotes() -> dict:
    """返回 {indexes: [...], stocks: [...], conflicts: [...], sources: {...}}"""
    res = run_collectors(
        {
            "em": _em_all,
            "sina": _sina_all,
            "sina_hk": _sina_hk_us,
            "tencent": _tencent_quotes,
            "cboe": _cboe_vix,
        }
    )
    em = res["em"]
    sina = res["sina"]
    sina_hk = res["sina_hk"]
    tencent = res["tencent"]
    cboe = res["cboe"]

    em_map: dict[str, dict] = em.value if em.ok and em.value else {}
    sina_map: dict[str, dict] = sina.value if sina.ok and sina.value else {}
    hk_map: dict[str, dict] = sina_hk.value if sina_hk.ok and sina_hk.value else {}
    tc_map: dict[str, dict] = tencent.value if tencent.ok and tencent.value else {}
    cb_map: dict[str, dict] = cboe.value if cboe.ok and cboe.value else {}

    conflicts: list[dict] = []

    def build_index_row(sid, name, region):
        code = sid.split(".")[1]
        emrow = em_map.get(code, {})
        sina_key = None
        if sid.startswith("1."):
            sina_key = "sh" + code
        elif sid.startswith("0."):
            sina_key = "sz" + code
        elif sid in OVERSEAS_SINA_MAP:
            sina_key = OVERSEAS_SINA_MAP[sid]
        srow = sina_map.get(sina_key) if sina_key else None
        if srow is None and sina_key:
            srow = hk_map.get(sina_key)
        tc_code = OVERSEAS_TENCENT_MAP.get(sid)
        trow = tc_map.get(tc_code) if tc_code else None
        crow = cb_map.get(sid)

        px_em = emrow.get("price")
        px_sina = (srow or {}).get("price")
        px_tc = (trow or {}).get("price")
        px_cb = (crow or {}).get("price")

        # 多源集合：A股=东财+新浪；港股/美股=东财+新浪+腾讯；VIX=新浪+CBOE
        prices = {"东财": px_em}
        if px_sina is not None:
            prices["新浪"] = px_sina
        if px_tc is not None:
            prices["腾讯"] = px_tc
        if px_cb is not None:
            prices["CBOE"] = px_cb

        # 主源价格：东财优先，缺失时依次回退
        price = px_em
        if price is None:
            price = px_tc if px_tc is not None else px_sina
        if price is None:
            price = px_cb
        pct = emrow.get("pct")
        if pct is None and trow:
            pct = trow.get("pct")
        if pct is None and crow:
            pct = crow.get("pct")
        chg = emrow.get("chg")
        if chg is None and trow and price and trow.get("prev_close"):
            chg = round(price - trow["prev_close"], 4)

        # 交易日优先用东财 f297（交易所口径），交叉源的日期只用于记录差异。
        # 美股尤其重要：新浪/腾讯给的是北京时间的次日清晨，会把上周五的收盘
        # 标成"今天"，导致休市日被误判成开市日。
        em_date = (emrow.get("trade_date") or "").strip()
        alt_date = (
            (srow or {}).get("date")
            or (trow or {}).get("date")
            or (crow or {}).get("date")
            or ""
        )
        trade_date = em_date or alt_date or (emrow.get("time", "")[:10])
        date_conflict = ""
        if em_date and alt_date and em_date != alt_date[:10]:
            date_conflict = (
                f"交易日口径分歧：东财 {em_date} vs 交叉源 {alt_date[:10]}"
            )
        cross = _cross_multi(prices, name, "index")
        if cross.get("status") == "conflict":
            conflicts.append(cross)
        return {
            "secid": sid,
            "code": code,
            "name": name,
            "region": region,
            "price": price,
            "pct": pct,
            "chg": chg,
            "amount": emrow.get("amount"),
            "volume": emrow.get("volume"),
            "pe": emrow.get("pe"),
            "pb": emrow.get("pb"),
            "mktcap": emrow.get("mktcap"),
            "trade_date": trade_date,
            "quote_time": emrow.get("time") or (trow or {}).get("time") or (crow or {}).get("time") or "",
            "date_conflict": date_conflict,
            "cross": cross,
        }

    def build_stock_row(sid, name, sector, theme):
        code = sid.split(".")[1]
        emrow = em_map.get(code, {})
        sina_key = ("sh" if sid.startswith("1.") else "sz") + code
        srow = sina_map.get(sina_key)
        hkrow = hk_map.get("rt_hk" + code)
        trow = tc_map.get("r_hk" + code) if sid.startswith("116.") else None
        px_em = emrow.get("price")
        px_sina = (hkrow or {}).get("price") if sid.startswith("116.") else (srow or {}).get("price")
        px_tc = (trow or {}).get("price")
        prices = {"东财": px_em}
        if px_sina is not None:
            prices["新浪"] = px_sina
        if px_tc is not None:
            prices["腾讯"] = px_tc
        cross = _cross_multi(prices, name, "stock")
        if cross.get("status") == "conflict":
            conflicts.append(cross)
        price = px_em
        if price is None:
            price = px_tc if px_tc is not None else px_sina
        pct = emrow.get("pct")
        if pct is None and trow:
            pct = trow.get("pct")
        # 个股同样优先用东财 f297 的交易所交易日（口径与指数一致）
        em_date = (emrow.get("trade_date") or "").strip()
        alt_date = (
            (srow or {}).get("date")
            or (trow or {}).get("date")
            or (hkrow or {}).get("date")
            or ""
        )
        alt_date = alt_date.replace("/", "-") if alt_date else ""
        trade_date = em_date or alt_date
        date_conflict = ""
        if em_date and alt_date and em_date != alt_date[:10]:
            date_conflict = f"交易日口径分歧：东财 {em_date} vs 交叉源 {alt_date[:10]}"
        return {
            "secid": sid,
            "code": code,
            "name": name,
            "sector": sector,
            "theme": theme,
            "market": "HK" if sid.startswith("116.") else "CN",
            "price": price,
            "pct": pct,
            "chg": emrow.get("chg"),
            "turnover": emrow.get("turnover"),
            "amount": emrow.get("amount"),
            "pe": emrow.get("pe"),
            "pb": emrow.get("pb"),
            "mktcap": emrow.get("mktcap"),
            "trade_date": trade_date,
            "quote_time": emrow.get("time") or (trow or {}).get("time") or "",
            "date_conflict": date_conflict,
            "cross": cross,
        }

    indexes = [build_index_row(*i) for i in C.INDEXES]
    stocks = [build_stock_row(*s) for s in C.ALL_STOCKS]

    # 交易日口径分歧单独列出：价格一致但日期不同，是真实存在的口径冲突，
    # 不能因为「价格对得上」就当没发生。
    for row in indexes + stocks:
        if row.get("date_conflict"):
            conflicts.append(
                {
                    "name": row.get("name"),
                    "kind": "trade_date",
                    "status": "conflict",
                    "src_names": ["东财", "交叉源"],
                    "n_sources": 2,
                    "note": row["date_conflict"],
                }
            )

    return {
        "indexes": indexes,
        "stocks": stocks,
        "conflicts": conflicts,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "eastmoney": {"ok": em.ok, "error": em.error, "count": len(em_map)},
            "sina_cn": {"ok": sina.ok, "error": sina.error, "count": len(sina_map)},
            "sina_hk_us": {"ok": sina_hk.ok, "error": sina_hk.error, "count": len(hk_map)},
            "tencent": {"ok": tencent.ok, "error": tencent.error, "count": len(tc_map)},
            "cboe": {"ok": cboe.ok, "error": cboe.error, "count": len(cb_map)},
        },
    }


def sina_hk_map_date(hkrow):
    if hkrow and hkrow.get("date"):
        return hkrow["date"]
    return ""


def _cross_multi(prices: dict[str, float | None], name: str, kind: str) -> dict:
    """多源交叉验证。

    prices: {源名: 价格}，None / 0 视为该源不可用。
    - ≥2 个有效源且最大相对差异 ≤ 容忍度 → verified
    - ≥2 个有效源但差异超容忍度 → conflict（保留各方数值）
    - 仅 1 个有效源 → single_source
    """
    avail = {k: v for k, v in prices.items() if v not in (None, 0)}
    base = {
        "name": name,
        "kind": kind,
        "srcs": {k: prices.get(k) for k in prices},
        "src_names": sorted(avail.keys()),
        "n_sources": len(avail),
    }
    if not avail:
        base.update({"status": "no_data", "note": "所有源均不可用"})
        return base
    if len(avail) == 1:
        only = list(avail.items())[0]
        base.update(
            {
                "status": "single_source",
                "src_a": only[1],
                "src_b": None,
                "note": f"仅 {only[0]} 一个来源可得，未达双源验证要求",
            }
        )
        return base
    vals = list(avail.values())
    lo, hi = min(vals), max(vals)
    diff_pct = abs(hi - lo) / max(abs(hi), abs(lo)) * 100
    base["src_a"] = vals[0]
    base["src_b"] = vals[1]
    base["diff_pct"] = round(diff_pct, 3)
    base["max_price"] = hi
    base["min_price"] = lo
    if diff_pct <= 0.35:
        base["status"] = "verified"
        base["note"] = f"{len(avail)} 源一致（{'/'.join(sorted(avail))}，差异 {diff_pct:.2f}%）"
    else:
        base["status"] = "conflict"
        detail = "、".join(f"{k}={v}" for k, v in sorted(avail.items()))
        base["note"] = f"{len(avail)} 源分歧 {diff_pct:.2f}%（{detail}），以东财为准并标注存疑"
    return base


def _cross(px_a, px_b, name, kind):
    """两个源比对（兼容旧调用）。"""
    return _cross_multi({"东财": px_a, "新浪": px_b}, name, kind)


if __name__ == "__main__":
    import json

    d = collect_quotes()
    print(json.dumps(d, ensure_ascii=False, indent=2)[:6000])
