"""期货与外汇采集器。

白名单机制：只覆盖 config.py 里列出的高流动性重点品种，不做全市场扫描。

===================== 各数据源的原生字段布局（实测钉死）=====================

新浪 nf_（国内期货，具体合约月，44 字段）
    [0]名称 [1]时间HHMMSS [2]开盘 [3]最高 [4]最低 [5]昨收(夜盘恒为0)
    [6]买价 [7]卖价 [8]最新价 [9]结算价(夜盘为0) [10]昨结算价
    [11]买量 [12]卖量 [13]持仓量 [14]成交量 [15]交易所 [16]品种 [17]日期
    注意：[5]昨收在夜盘时段恒为 0.000，不可用于算涨跌幅；
          对标基准是 [10]昨结算价（与东财 zjsj 完全一致）。

新浪 hf_（外盘期货连续，15 字段）
    [0]最新价 [1]涨跌幅%(新浪恒为空) [2]开盘 [3]最高 [4]最低
    [5]最低 [6]时间HHMMSS [7]昨收 [8]开盘(?) [9]买量 [10]持仓量
    [11]卖量 [12]日期 [13]中文名 [14]持仓量
    实测反推（用东财 f18/昨结算 反算，5/5 品种精确吻合）：
        [0]=最新价  [2]=买价  [3]=卖价  [4]=最高  [5]=最低
        [6]=时间    [7]=昨收  [8]=开盘  [12]=日期 [13]=中文名
    旧代码曾把 [5]最低价当昨收 → 原油涨跌幅失真；本次已修正。

新浪 fx_（外汇即期，18 字段）
    [0]时间HHMMSS [1]现价 [2]买价 [3]卖价 [4]成交量 [5]昨收
    [6]最高 [7]最低 [8]开盘 [9]中文名 [10]涨跌幅% [11]涨跌额
    [12]振幅% [13]备注 [14]52周高 [15]52周低 [16]空 [17]日期
    实测校验：calc_pct=(现价-昨收)/昨收 与 [10] 完全吻合（7/7 品种）。

东财 futsseapi（国内期货 / LME）
    dm=合约代码  name=名称  p=最新价  zjsj=昨结算价  zdf=涨跌幅
    zde=涨跌额  o=开 h=高 l=低  vol=成交量  ccl=持仓量
    market: 113=上期所 114=大商所 142=能源中心 109=LME 101=COMEX 102=NYMEX

东财 push2 ulist.np（外汇 / 外盘期货）
    f2=最新价  f3=涨跌幅%  f4=涨跌额  f18=昨收  f124=时间戳
    ⚠ f59 是"小数位缩放因子"，为 0 表示"无需缩放"。
      旧代码把 f59=0 当成"4位小数"再除以 10^4 → 昨收缩小一万倍（英镑/日元
      prev_close=0.0212）。fltt=2 时 f18 已是真实小数，绝不能再缩放。

腾讯 hf_（外盘期货，与新浪同源镜像，仅补 [1]涨跌幅）
    ⚠ 价格与新浪逐位相同，不构成独立第二源，只能作参考，不计入交叉验证。

================== 交叉验证策略 ==================
国内期货：新浪具体合约月(nf_RB2701) × 东财同合约月(rb2701)
    → 实测 12/12 品种「最新价/最高/最低/昨结算/持仓/成交」逐字段精确相等。
    用「同一合约月」配对，而非主连对主连（主连指向的合约月不同会产生假冲突）。

外盘期货：新浪 hf_ × 东财 101/102/109
    → hf_CL↔102.CL00Y、hf_GC↔101.GC00Y、hf_SI↔101.SI00Y
       hf_CAD↔109.LCPT、hf_NID↔109.LNKT
    两源均含独立的昨收/最高/最低，用于校验。

外汇：新浪 fx_ × 东财 push2 ×（在岸人民币另加中国货币网中间价）
    → fx_susdcny(在岸即期) 无严格同标的东财代码，故在岸人民币以
      东财 120.USDCNYC(中间价) + 中国货币网(CFETS 官方中间价) 交叉，
      并与新浪 fx_susdcny 标注为「口径不同（中间价 vs 即期）」。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors.base import (  # noqa: E402
    Fetch,
    http_get,
    http_get_json,
    parse_sina_hq,
    run_collectors,
    safe_float,
)
import config as C  # noqa: E402


# --------------------------------------------------------------------------
# 东方财富国内期货 / LME / 外盘期货：secid 映射
# --------------------------------------------------------------------------
# 东财合约代码（dm）→ 交易所 market
EM_CN_MARKET = {
    "RB": "113", "AU": "113", "AG": "113", "CU": "113", "AL": "113", "NI": "113",
    "I": "114", "J": "114", "M": "114", "C": "114",
    "SC": "142", "LU": "142",
}

# 东财主连 dm（用于取「哪个具体合约月最活跃」，进而定位新浪同月代码）
EM_MAIN_DM = {
    "RB": "rbm", "AU": "aum", "AG": "agm", "CU": "cum", "AL": "alm", "NI": "nim",
    "I": "im", "J": "jm", "M": "mm", "C": "cm",
    "SC": "scm", "LU": "lum",
}

# 东财 dm 前缀 → 新浪品种代码（大写）
EM_DM_PREFIX_TO_SINA = {
    "rb": "RB", "au": "AU", "ag": "AG", "cu": "CU", "al": "AL", "ni": "NI",
    "i": "I", "j": "J", "m": "M", "c": "C", "sc": "SC", "lu": "LU",
}

# 外盘期货：新浪 hf_ 代码 → 东财 secid
EM_GLOBAL_SECID = {
    "hf_CL": "102.CL00Y",   # NYMEX原油
    "hf_GC": "101.GC00Y",   # COMEX黄金
    "hf_SI": "101.SI00Y",   # COMEX白银
    "hf_CAD": "109.LCPT",   # 综合铜03 ≈ LME铜
    "hf_NID": "109.LNKT",   # 综合镍03 ≈ LME镍
}

# 外汇：新浪 fx_ 代码 → 东财 secid（严格同标的）
EM_FX_SECID = {
    "fx_seurusd": "119.EURUSD",
    "fx_susdjpy": "119.USDJPY",
    "fx_sgbpjpy": "119.GBPJPY",
    "fx_saudusd": "119.AUDUSD",
    "fx_susdhkd": "119.USDHKD",
    "fx_susdcnh": "133.USDCNH",   # 离岸人民币（与新浪 fx_susdcnh 同标的）
}

# 人民币中间价：中国货币网 currency → 新浪/东财对照
CM_CURENCIES = ["USD/CNY", "EUR/CNY", "100JPY/CNY", "HKD/CNY", "GBP/CNY"]


def _sina_futures_fx() -> Fetch:
    """新浪：国内期货(nf_) + 外盘(hf_) + 外汇(fx_) 一次取回。"""
    codes = []
    for code, *_ in C.CN_FUTURES:
        codes.append(code)
    for code, *_ in C.GLOBAL_FUTURES:
        codes.append(code)
    for code, *_ in C.FX:
        codes.append(code)
    url = "https://hq.sinajs.cn/list=" + ",".join(codes)
    f = http_get(url, referer="https://finance.sina.com.cn/", encoding="gbk")
    if not f.ok:
        return f
    raw = parse_sina_hq(f.value)
    out: dict[str, dict] = {}
    for code, arr in raw.items():
        if len(arr) < 4:
            continue
        if code.startswith("nf_"):
            # nf_: [0]名称 [1]时间 [2]开 [3]高 [4]低 [5]昨收(夜盘0)
            #      [6]买 [7]卖 [8]最新 [9]结算 [10]昨结算 [13]持仓 [14]成交 [17]日期
            out[code] = {
                "kind": "cn_futures",
                "name": arr[0],
                "price": safe_float(arr[8]),
                "settle": safe_float(arr[9]),
                "prev_settle": safe_float(arr[10]),
                "open": safe_float(arr[2]),
                "high": safe_float(arr[3]),
                "low": safe_float(arr[4]),
                "prev_close": safe_float(arr[5]),
                "bid": safe_float(arr[6]),
                "ask": safe_float(arr[7]),
                "position": safe_float(arr[13]) if len(arr) > 13 else None,
                "volume": safe_float(arr[14]) if len(arr) > 14 else None,
                "exchange": arr[15] if len(arr) > 15 else "",
                "time": arr[1],
                "date": arr[17] if len(arr) > 17 else "",
            }
        elif code.startswith("hf_"):
            # hf_: [0]最新 [1]涨跌幅%(新浪空) [2]买 [3]卖 [4]最高 [5]最低
            #      [6]时间 [7]昨收 [8]开盘 [12]日期 [13]中文名
            # 关键修正：[7]才是昨收（曾误用 [5]最低价）
            out[code] = {
                "kind": "global_futures",
                "name": arr[13] if len(arr) > 13 else code,
                "price": safe_float(arr[0]),
                "pct": safe_float(arr[1]),
                "bid": safe_float(arr[2]),
                "ask": safe_float(arr[3]),
                "high": safe_float(arr[4]),
                "low": safe_float(arr[5]),
                "prev_close": safe_float(arr[7]) if len(arr) > 7 else None,
                "open": safe_float(arr[8]) if len(arr) > 8 else None,
                "time": arr[6] if len(arr) > 6 else "",
                "date": arr[12] if len(arr) > 12 else "",
            }
        elif code.startswith("fx_s"):
            # fx_: [0]时间 [1]现价 [2]买 [3]卖 [5]昨收 [6]高 [7]低 [8]开
            #      [9]名称 [10]涨跌幅% [11]涨跌额 [17]日期
            out[code] = {
                "kind": "fx",
                "name": arr[9] if len(arr) > 9 else code,
                "price": safe_float(arr[1]) if len(arr) > 1 else None,
                "bid": safe_float(arr[2]) if len(arr) > 2 else None,
                "ask": safe_float(arr[3]) if len(arr) > 3 else None,
                "prev_close": safe_float(arr[5]) if len(arr) > 5 else None,
                "high": safe_float(arr[6]) if len(arr) > 6 else None,
                "low": safe_float(arr[7]) if len(arr) > 7 else None,
                "open": safe_float(arr[8]) if len(arr) > 8 else None,
                "pct": safe_float(arr[10]) if len(arr) > 10 else None,
                "chg": safe_float(arr[11]) if len(arr) > 11 else None,
                "time": arr[0] if arr else "",
                "date": arr[17] if len(arr) > 17 else "",
            }
    return Fetch(value=out, source="sina.hq.futures_fx", ok=True)


def _em_futures() -> Fetch:
    """东方财富国内期货 + LME：返回 (主连记录, 具体合约月记录, 活跃月映射)。

    接口：futsseapi.eastmoney.com/list/{market}
    market: 113=上期所 114=大商所 142=能源中心 109=LME
    """
    by_market: dict[str, list[str]] = {}
    for sina_code, *_ in C.CN_FUTURES:
        sym = sina_code.replace("nf_", "").replace("0", "").upper()
        mkt = EM_CN_MARKET.get(sym)
        if mkt:
            by_market.setdefault(mkt, []).append(sym)

    main_rows: dict[str, dict] = {}
    spec_rows: dict[str, dict] = {}
    active_month: dict[str, str] = {}
    errors: list[str] = []

    for mkt, syms in by_market.items():
        url = (
            f"https://futsseapi.eastmoney.com/list/{mkt}"
            "?orderBy=vol&sort=desc&pageSize=400&pageIndex=0"
            "&field=dm,sc,name,p,zde,zdf,zjsj,o,h,l,vol,ccl,cje"
        )
        f = http_get_json(url, referer="https://quote.eastmoney.com/")
        if not f.ok:
            errors.append(f"mkt{mkt}:{f.error}")
            continue
        lst = (f.value or {}).get("list") or []
        by_dm = {str(x.get("dm")): x for x in lst}
        for sym in syms:
            dm_main = EM_MAIN_DM.get(sym)
            x = by_dm.get(dm_main) if dm_main else None
            if not x:
                continue
            main_rows[sym] = _em_fut_row(x)
            # 找成交最大的具体合约月（与新浪 nf_<SYM><YYMM> 对标）
            best = None
            for cand in lst:
                cdm = str(cand.get("dm") or "")
                if not cdm.startswith(sym.lower()):
                    continue
                suffix = cdm[len(sym.lower()):]
                if len(suffix) != 4 or not suffix.isdigit():
                    continue
                if best is None or (cand.get("vol") or 0) > (best.get("vol") or 0):
                    best = cand
            if best:
                bdm = str(best.get("dm"))
                spec_rows[sym] = _em_fut_row(best)
                active_month[sym] = bdm

    if not main_rows and errors:
        return Fetch(value={}, source="eastmoney.futures", ok=False, error="; ".join(errors))
    return Fetch(
        value={"main": main_rows, "spec": spec_rows, "active_month": active_month},
        source="eastmoney.futures",
        ok=True,
    )


def _em_fut_row(x: dict) -> dict:
    return {
        "kind": "cn_futures",
        "name": x.get("name"),
        "dm": x.get("dm"),
        "price": safe_float(x.get("p")),
        "pct": safe_float(x.get("zdf")),
        "chg": safe_float(x.get("zde")),
        "prev_settle": safe_float(x.get("zjsj")),
        "open": safe_float(x.get("o")),
        "high": safe_float(x.get("h")),
        "low": safe_float(x.get("l")),
        "volume": safe_float(x.get("vol")),
        "position": safe_float(x.get("ccl")),
        "date": "",
        "time": "",
    }


def _em_futures_spec() -> Fetch:
    """按活跃合约月，从新浪取「同一合约月」的 nf_ 代码，交给 _sina 做精确对标。"""
    # 依赖 _em_futures 的结果，故在 collect 阶段串联执行
    return Fetch(value={}, source="sina.hq.cn_futures_spec", ok=True)


def _em_global_futures() -> Fetch:
    """东财外盘期货：101=COMEX 102=NYMEX 109=LME。"""
    out: dict[str, dict] = {}
    for secid in EM_GLOBAL_SECID.values():
        market = secid.split(".")[0]
        code = secid.split(".")[1]
        url = (
            f"https://futsseapi.eastmoney.com/list/{market}"
            "?orderBy=vol&sort=desc&pageSize=300&pageIndex=0"
            "&field=dm,sc,name,p,zde,zdf,zjsj,o,h,l,vol,ccl,cje"
        )
        f = http_get_json(url, referer="https://quote.eastmoney.com/")
        if not f.ok:
            continue
        lst = (f.value or {}).get("list") or []
        x = next((i for i in lst if str(i.get("dm")) == code), None)
        if not x:
            continue
        out[secid] = _em_fut_row(x)
    return Fetch(value=out, source="eastmoney.futures.global", ok=True)


def _em_fx() -> Fetch:
    """东财外汇，作为外汇的第二源。

    关键修正：fltt=2 时 f18 已是真实小数，f59=0 表示「无需缩放」。
    旧代码 `dec = int(f59 or 4)` 在 f59=0 时落到 4，把昨收除以 10^4，
    导致英镑/日元 prev_close=0.0212、澳元/美元 6.9e-05 的量级错误。
    """
    secids = list(EM_FX_SECID.values()) + ["120.USDCNYC"]
    url = (
        "https://push2.eastmoney.com/api/qt/ulist.np/get"
        f"?fltt=2&secids={','.join(secids)}&fields=f1,f2,f3,f4,f12,f13,f14,f18,f59,f124"
    )
    f = http_get_json(url, referer="https://quote.eastmoney.com/")
    if not f.ok:
        return f
    diff = (f.value or {}).get("data", {}).get("diff") or []
    by_code = {d.get("f12"): d for d in diff}
    out: dict[str, dict] = {}
    for sina_code, secid in EM_FX_SECID.items():
        d = by_code.get(secid.split(".")[1])
        if not d:
            continue
        out[sina_code] = _em_fx_row(d)
    # 人民币中间价单独存（口径=中间价，与即期不同）
    d = by_code.get("USDCNYC")
    if d:
        out["mid_cny"] = _em_fx_row(d)
    return Fetch(value=out, source="eastmoney.push2.fx", ok=True)


def _em_fx_row(d: dict) -> dict:
    px = safe_float(d.get("f2"))
    prev = safe_float(d.get("f18"))
    pct = safe_float(d.get("f3"))
    chg = safe_float(d.get("f4"))
    ts = safe_float(d.get("f124"))
    return {
        "kind": "fx",
        "name": d.get("f14"),
        "price": px,
        "prev_close": prev,      # 不做任何缩放（fltt=2 已是真实值）
        "pct": pct,
        "chg": chg,
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "",
        "date": time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "",
    }


def _chinamoney_mid() -> Fetch:
    """中国货币网（CFETS 官方）人民币中间价 —— 独立第三源。

    接口返回 records: [{"date": "2026-11-06", "values": ["6.7168", ...]}, ...]
    values 顺序对应请求的 currency 列表。
    """
    url = (
        "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew"
        f"?startDate=2026-09-01&endDate={time.strftime('%Y-%m-%d')}"
        f"&currency={','.join(CM_CURENCIES)}&pageNum=1&pageSize=10"
    )
    f = http_get_json(url, referer="https://www.chinamoney.com.cn/")
    if not f.ok:
        return f
    records = (f.value or {}).get("records") or []
    if not records:
        return Fetch(value={}, source="chinamoney.mid", ok=False, error="records empty")
    # 取最新一条（records 按日期倒序）
    latest = records[0]
    date = latest.get("date") or ""
    vals = latest.get("values") or []
    out: dict[str, dict] = {}
    for cur, v in zip(CM_CURENCIES, vals):
        px = safe_float(v)
        if px is None:
            continue
        out[cur] = {
            "kind": "fx_mid",
            "currency": cur,
            "price": px,
            "date": date,
            "name": f"{cur} 中间价",
        }
    return Fetch(value=out, source="chinamoney.mid", ok=True)


def _tencent_fx() -> Fetch:
    """腾讯：外盘期货参考源。

    ⚠ 腾讯 hf_ 与新浪 hf_ 同源镜像（价格逐位相同），不构成独立第二源，
    仅用于补新浪缺失的 [1] 涨跌幅，不计入交叉验证的独立源计数。
    """
    codes = list(EM_GLOBAL_SECID.keys())
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
        p = val.split(",")
        if len(p) < 13:
            continue
        out[key] = {
            "name": p[13] if len(p) > 13 else key,
            "price": safe_float(p[0]),
            "pct": safe_float(p[1]),
            "bid": safe_float(p[2]),
            "ask": safe_float(p[3]),
            "high": safe_float(p[4]),
            "low": safe_float(p[5]),
            "prev_close": safe_float(p[7]),
            "open": safe_float(p[8]),
            "time": p[6],
            "date": p[12][:10] if p[12] else "",
            "same_source_as": "sina.hq",
        }
    return Fetch(value=out, source="tencent.gtimg.futures_fx", ok=True)


def _cross(prices: dict, name: str, *, tol: float = 0.15, note: str = "") -> dict:
    """多源交叉验证（价格口径）。保留各方数值，冲突不静默消除。"""
    valid = {k: v for k, v in prices.items() if v not in (None, 0)}
    base = {
        "name": name,
        "srcs": dict(prices),
        "src_names": sorted(valid.keys()),
        "n_sources": len(valid),
    }
    if note:
        base["note"] = note
    if len(valid) == 0:
        base.update({"status": "no_data", "note": note or "所有源均不可用"})
    elif len(valid) == 1:
        only = list(valid.items())[0]
        base.update({
            "status": "single_source",
            "src_a": only[1],
            "note": note or f"仅 {only[0]} 一个来源可得，未达双源验证要求",
        })
    else:
        vals = list(valid.values())
        lo, hi = min(vals), max(vals)
        diff = abs(hi - lo) / max(abs(hi), abs(lo)) * 100
        base["diff_pct"] = round(diff, 3)
        base["src_a"] = vals[0]
        base["src_b"] = vals[1]
        base["max_price"] = hi
        base["min_price"] = lo
        if diff <= tol:
            base["status"] = "verified"
            detail = "、".join(f"{k}={v}" for k, v in sorted(valid.items()))
            base["note"] = note or f"{len(valid)} 源一致（{detail}，差异 {diff:.3f}%）"
        else:
            base["status"] = "conflict"
            detail = "、".join(f"{k}={v}" for k, v in sorted(valid.items()))
            base["note"] = (
                note + "；" if note else ""
            ) + f"{len(valid)} 源分歧 {diff:.3f}%（{detail}）"
    return base


def _cross_fields(rows: list[tuple[str, dict, str]], name: str, *, tol: float = 0.02) -> dict:
    """按「同一合约月」多字段交叉验证。

    rows: [(源名, 该源记录, 品种代码), ...]
    比对 price / prev_settle / high / low 四个字段，逐字段算最大相对差。
    """
    srcs = {}
    fields: dict[str, dict] = {}
    for src, row, _sym in rows:
        srcs[src] = row.get("price")
    checked = []
    max_diff = 0.0
    for fname in ("price", "prev_settle", "high", "low"):
        vals = [(src, row.get(fname)) for src, row, _ in rows]
        nums = [v for _s, v in vals if v not in (None, 0)]
        if len(nums) < 2:
            continue
        lo, hi = min(nums), max(nums)
        d = abs(hi - lo) / max(abs(hi), abs(lo)) * 100
        fields[fname] = {
            "vals": {s: v for s, v in vals},
            "diff_pct": round(d, 4),
            "match": d <= tol,
        }
        checked.append(fname)
        max_diff = max(max_diff, d)
    n_src = len({s for s, _r, _sym in rows})
    base = {
        "name": name,
        "srcs": srcs,
        "src_names": sorted({s for s, _r, _sym in rows}),
        "n_sources": n_src,
        "fields": fields,
        "max_field_diff_pct": round(max_diff, 4),
        "checked_fields": checked,
    }
    if n_src < 2:
        base.update({"status": "single_source", "note": "仅一个来源可得，未达双源验证要求"})
    elif not checked:
        base.update({"status": "no_data", "note": "无可比对字段"})
    elif all(f["match"] for f in fields.values()):
        base.update({
            "status": "verified",
            "note": f"{n_src} 源同合约月{'/'.join(checked)} 字段全部一致（最大差 {max_diff:.4f}%）",
        })
    else:
        bad = [k for k, f in fields.items() if not f["match"]]
        base.update({
            "status": "conflict",
            "note": f"{n_src} 源同合约月在 {','.join(bad)} 字段分歧（最大差 {max_diff:.4f}%）",
        })
    return base


def collect_futures_fx() -> dict:
    """采集期货与外汇重点品种，返回统一结构。"""
    # 第一轮：并发抓主源
    res = run_collectors({
        "sina": _sina_futures_fx,
        "em": _em_futures,
        "em_global": _em_global_futures,
        "em_fx": _em_fx,
        "tencent": _tencent_fx,
        "cm_mid": _chinamoney_mid,
    })

    em = res["em"]
    em_map: dict = em.value if em.ok and em.value else {}
    em_main: dict = em_map.get("main") or {}
    em_spec: dict = em_map.get("spec") or {}
    active_month: dict = em_map.get("active_month") or {}

    # 第二轮：按活跃合约月取新浪同月 nf_ 代码，做精确对标
    sina_map: dict[str, dict] = res["sina"].value if res["sina"].ok and res["sina"].value else {}
    spec_codes: list[str] = []
    sym_to_spec_code: dict[str, str] = {}
    for sina_code, name, sector in C.CN_FUTURES:
        sym = sina_code.replace("nf_", "").replace("0", "").upper()
        dm = active_month.get(sym)
        if not dm:
            continue
        suffix = dm[len(sym.lower()):]
        if len(suffix) != 4 or not suffix.isdigit():
            continue
        spec_code = f"nf_{sym}{suffix}"
        sym_to_spec_code[sym] = spec_code
        spec_codes.append(spec_code)

    if spec_codes:
        url = "https://hq.sinajs.cn/list=" + ",".join(spec_codes)
        f = http_get(url, referer="https://finance.sina.com.cn/", encoding="gbk")
        if f.ok:
            raw = parse_sina_hq(f.value)
            for code, arr in raw.items():
                if len(arr) < 18:
                    continue
                sina_map[code] = {
                    "kind": "cn_futures",
                    "name": arr[0],
                    "price": safe_float(arr[8]),
                    "settle": safe_float(arr[9]),
                    "prev_settle": safe_float(arr[10]),
                    "open": safe_float(arr[2]),
                    "high": safe_float(arr[3]),
                    "low": safe_float(arr[4]),
                    "prev_close": safe_float(arr[5]),
                    "bid": safe_float(arr[6]),
                    "ask": safe_float(arr[7]),
                    "position": safe_float(arr[13]),
                    "volume": safe_float(arr[14]),
                    "exchange": arr[15],
                    "time": arr[1],
                    "date": arr[17],
                }

    emg_map: dict = res["em_global"].value if res["em_global"].ok and res["em_global"].value else {}
    emfx_map: dict = res["em_fx"].value if res["em_fx"].ok and res["em_fx"].value else {}
    tc_map: dict = res["tencent"].value if res["tencent"].ok and res["tencent"].value else {}
    cm_map: dict = res["cm_mid"].value if res["cm_mid"].ok and res["cm_mid"].value else {}

    conflicts: list[dict] = []

    def build_cn(code, name, sector):
        sym = code.replace("nf_", "").replace("0", "").upper()
        main_row = sina_map.get(code) or {}
        spec_code = sym_to_spec_code.get(sym)
        spec_sina = sina_map.get(spec_code) if spec_code else None
        spec_em = em_spec.get(sym)

        # 交叉验证用「同合约月」两源
        rows = []
        if spec_sina:
            rows.append(("新浪", spec_sina, sym))
        if spec_em:
            rows.append(("东财", spec_em, sym))
        cross = _cross_fields(rows, name) if len(rows) >= 2 else _cross(
            {"新浪": main_row.get("price")}, name
        )
        if cross.get("status") == "conflict":
            conflicts.append(cross)

        price = (spec_sina or {}).get("price") or main_row.get("price")
        prev_settle = (spec_sina or {}).get("prev_settle") or main_row.get("prev_settle")
        pct = None
        if price and prev_settle:
            pct = round((price - prev_settle) / prev_settle * 100, 3)
        if pct is None and spec_em:
            pct = spec_em.get("pct")
        dm = active_month.get(sym, "")
        return {
            "code": code,
            "name": name,
            "sector": sector,
            "kind": "cn_futures",
            "price": price,
            "pct": pct,
            "chg": round(price - prev_settle, 4) if (price and prev_settle) else None,
            "settle": (spec_sina or {}).get("settle") or main_row.get("settle"),
            "prev_settle": prev_settle,
            "open": (spec_sina or {}).get("open") or main_row.get("open"),
            "high": (spec_sina or {}).get("high") or main_row.get("high"),
            "low": (spec_sina or {}).get("low") or main_row.get("low"),
            "prev_close": main_row.get("prev_close"),
            "volume": (spec_sina or {}).get("volume") or main_row.get("volume"),
            "position": (spec_sina or {}).get("position") or main_row.get("position"),
            "contract": dm,
            "contract_label": (spec_sina or {}).get("name") or (spec_em or {}).get("name") or "",
            "time": (spec_sina or {}).get("time") or main_row.get("time") or "",
            "date": (spec_sina or {}).get("date") or main_row.get("date") or "",
            "cross": cross,
        }

    def build_global(code, name):
        s = sina_map.get(code) or {}
        t = tc_map.get(code) or {}
        secid = EM_GLOBAL_SECID.get(code)
        e = emg_map.get(secid) if secid else None
        # 独立源判定（实测结论）：
        #   腾讯 hf_ 与新浪 hf_ 逐位同源（腾讯 [1]涨跌幅 = 新浪 [0]/[7] 自算值，5/5 吻合）
        #   → 两者合并计为「新浪/腾讯」一个源；东财是真正独立的第二源。
        prices = {"新浪/腾讯": s.get("price")}
        if e:
            prices["东财"] = e.get("price")
        cross = _cross(prices, name, tol=0.35)
        if cross.get("status") == "conflict":
            conflicts.append(cross)
        price = s.get("price")
        if price is None and e:
            price = e.get("price")
        # 昨收口径：东财 push2 的 f18 对 LME 不自洽（f18 与 zjsj 分歧，且按 f18
        # 反算的涨跌幅与其自身 f3 不符），故以 futsseapi 的 zjsj（昨结算）为基准。
        em_prev = (e or {}).get("prev_settle")
        sina_prev = s.get("prev_close")
        prev = sina_prev if sina_prev is not None else em_prev
        pct = None
        if price and prev:
            pct = round((price - prev) / prev * 100, 3)
        if pct is None and e:
            pct = e.get("pct")
        if pct is None and t:
            pct = t.get("pct")
        # 昨收交叉：新浪/腾讯 [7] vs 东财 zjsj
        prev_cross = None
        if sina_prev is not None and em_prev is not None:
            prev_cross = _cross(
                {"新浪/腾讯": sina_prev, "东财": em_prev},
                f"{name} 昨收", tol=0.35,
            )
            prev_cross["note"] = (
                prev_cross.get("note", "")
                + "（新浪/腾讯取 hf_[7]；东财取 futsseapi 昨结算 zjsj，"
                  "push2 的 f18 对 LME 品种与 zjsj 口径不一致，已排除）"
            )
            if prev_cross.get("status") == "conflict":
                conflicts.append(prev_cross)
        return {
            "code": code,
            "name": name,
            "sector": "外盘",
            "kind": "global_futures",
            "price": price,
            "pct": pct,
            "chg": round(price - prev, 4) if (price and prev) else None,
            "settle": (e or {}).get("prev_settle"),
            "open": s.get("open") or (e or {}).get("open"),
            "high": s.get("high") or (e or {}).get("high"),
            "low": s.get("low") or (e or {}).get("low"),
            "prev_close": prev,
            "volume": (e or {}).get("volume"),
            "position": (e or {}).get("position"),
            "contract": (e or {}).get("dm", ""),
            "contract_label": (e or {}).get("name", ""),
            "time": s.get("time") or (e or {}).get("time") or "",
            "date": s.get("date") or (e or {}).get("date") or "",
            "ref_tencent_price": t.get("price"),
            "cross": cross,
            "prev_cross": prev_cross,
        }

    def build_fx(code, name):
        s = sina_map.get(code) or {}
        e = emfx_map.get(code) or {}
        prices = {"新浪": s.get("price")}
        if e:
            prices["东财"] = e.get("price")
        note = ""
        if code == "fx_susdcny":
            note = "在岸人民币：东财为离岸/CNH口径或中间价，与新浪即期口径不同"
        cross = _cross(prices, name, tol=0.15, note=note)
        if cross.get("status") == "conflict":
            conflicts.append(cross)
        price = s.get("price")
        if price is None and e:
            price = e.get("price")
        prev = s.get("prev_close")
        if prev is None and e:
            prev = e.get("prev_close")
        pct = s.get("pct")
        if pct is None and e:
            pct = e.get("pct")
        if pct is None and price and prev:
            pct = round((price - prev) / prev * 100, 4)
        return {
            "code": code,
            "name": name,
            "sector": "外汇",
            "kind": "fx",
            "price": price,
            "pct": pct,
            "chg": s.get("chg") if s.get("chg") is not None else (e.get("chg") if e else None),
            "bid": s.get("bid"),
            "ask": s.get("ask"),
            "high": s.get("high"),
            "low": s.get("low"),
            "open": s.get("open"),
            "prev_close": prev,
            "time": s.get("time") or (e.get("time") if e else "") or "",
            "date": s.get("date") or (e.get("date") if e else "") or "",
            "cross": cross,
        }

    cn_futures = [build_cn(*x) for x in C.CN_FUTURES]
    global_futures = [build_global(*x) for x in C.GLOBAL_FUTURES]
    fx = [build_fx(*x) for x in C.FX]

    # 人民币中间价：东财 + 中国货币网 双源
    mid_em = emfx_map.get("mid_cny") or {}
    mid_cm = cm_map.get("USD/CNY") or {}
    mid_cross = _cross(
        {"东财": mid_em.get("price"), "中国货币网": mid_cm.get("price")},
        "美元/人民币中间价", tol=0.05,
    )
    if mid_cross.get("status") == "conflict":
        conflicts.append(mid_cross)
    mid = {
        "code": "mid_cny",
        "name": "美元/人民币中间价",
        "sector": "外汇",
        "kind": "fx_mid",
        "price": mid_em.get("price") or mid_cm.get("price"),
        "prev_close": None,
        "pct": mid_em.get("pct"),
        "date": mid_cm.get("date") or mid_em.get("date") or "",
        "eastmoney": mid_em.get("price"),
        "chinamoney": mid_cm.get("price"),
        "cross": mid_cross,
    }

    return {
        "cn_futures": cn_futures,
        "global_futures": global_futures,
        "fx": fx,
        "fx_mid": mid,
        "conflicts": conflicts,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "sina": {"ok": res["sina"].ok, "error": res["sina"].error, "count": len(sina_map)},
            "eastmoney_futures": {"ok": em.ok, "error": em.error, "count": len(em_map.get("main") or {})},
            "eastmoney_global": {"ok": res["em_global"].ok, "error": res["em_global"].error, "count": len(emg_map)},
            "eastmoney_fx": {"ok": res["em_fx"].ok, "error": res["em_fx"].error, "count": len(emfx_map)},
            "tencent": {"ok": res["tencent"].ok, "error": res["tencent"].error, "count": len(tc_map),
                        "note": "与新浪同源镜像，仅供参考，不计独立源"},
            "chinamoney_mid": {"ok": res["cm_mid"].ok, "error": res["cm_mid"].error, "count": len(cm_map)},
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(collect_futures_fx(), ensure_ascii=False, indent=2))
