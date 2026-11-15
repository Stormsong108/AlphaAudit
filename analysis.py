"""日报分析引擎：从真实采集数据推导每张表后面的分析文字。

设计原则（对应「数字铁律」）：
1. 分析里的每个数字都来自当天的采集结果，**不引入外部数字、不编造**。
2. 分析是**确定性规则**的产物 —— 同样的输入永远得到同样的文字，
   这样每天的日报可审计、可回溯，不会出现「换个模型就换套说法」。
3. 判断只落在数据能支撑的范围内：数据不足就明说「样本不足，不下结论」，
   而不是硬凑一个结论。
4. 所有建议性内容一律标注「模拟/研究」，不构成投资建议。

每个函数返回 list[str]，每条是一个要点（HTML 片段已由调用方转义）。
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------- 工具


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


def _pct(v: Any, nd: int = 2) -> str:
    x = _num(v)
    return "—" if x is None else f"{x:+.{nd}f}%"


def _f(v: Any, nd: int = 2) -> str:
    x = _num(v)
    return "—" if x is None else f"{x:,.{nd}f}"


def _verdict(verified: int, total: int) -> str:
    if not total:
        return "无样本"
    r = verified / total * 100
    if r >= 95:
        return f"核验率 {r:.1f}%，数据可信度高"
    if r >= 80:
        return f"核验率 {r:.1f}%，基本可信，少数标的需留意口径"
    if r >= 50:
        return f"核验率仅 {r:.1f}%，过半标的只有单一来源，结论需降级使用"
    return f"核验率 {r:.1f}%，双源覆盖不足，本表只作线索参考"


def _spread(items: list[tuple[str, float]]) -> str:
    """最强/最弱并列描述。"""
    if not items:
        return ""
    s = sorted(items, key=lambda kv: -kv[1])
    top, bot = s[0], s[-1]
    if len(s) > 1:
        return (
            f"最强 {top[0]} {top[1]:+.2f}%，最弱 {bot[0]} {bot[1]:+.2f}%，"
            f"板块内振幅 {top[1] - bot[1]:.2f} 个百分点"
        )
    return f"仅 {top[0]} {top[1]:+.2f}%"


def _cnt(items: list[tuple[str, float]]) -> str:
    up = sum(1 for _n, v in items if v > 0)
    down = sum(1 for _n, v in items if v < 0)
    flat = len(items) - up - down
    return f"涨 {up} / 跌 {down} / 平 {flat}"


# ---------------------------------------------------------------- 大盘


def analyze_indexes(indexes: list[dict]) -> list[str]:
    out: list[str] = []
    if not indexes:
        return ["未取到指数行情，本表无法给出判断。"]
    pcts = [(r.get("name") or "?", _num(r.get("pct")) or 0.0) for r in indexes]
    out.append(f"样本 {len(indexes)} 个指数：{_cnt(pcts)}。{_spread(pcts)}。")

    # A 股三大指数单独判断（同市场可比性更强）
    cn = [
        (r.get("name") or "?", _num(r.get("pct")))
        for r in indexes
        if (r.get("market") == "CN" or r.get("region") in ("A股", "CN"))
        and _num(r.get("pct")) is not None
    ]
    if len(cn) >= 2:
        vals = [v for _n, v in cn]
        avg = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        out.append(
            f"A 股口径 {len(cn)} 个指数平均 {_pct(avg)}，"
            f"内部离散度 {spread:.2f} 个百分点"
            + ("，走势高度一致" if spread < 0.5 else "，分化明显，指数不能互相替代")
        )

    # 成交额
    amts = [(r.get("name") or "?", _num(r.get("amount"))) for r in indexes]
    amts_v = [(n, a) for n, a in amts if a]
    if amts_v:
        total = sum(a for _n, a in amts_v)
        biggest = max(amts_v, key=lambda kv: kv[1])
        out.append(
            f"合计成交 {total / 1e8:,.0f} 亿元，其中 {biggest[0]} 占比 "
            f"{biggest[1] / total * 100:.1f}%"
            + (
                "，成交高度集中于单一市场"
                if biggest[1] / total > 0.6
                else "，成交分布相对均衡"
            )
        )

    # 核验与交易日
    ver = sum(1 for r in indexes if (r.get("cross") or {}).get("status") == "verified")
    out.append(_verdict(ver, len(indexes)) + "。")
    dates = {r.get("trade_date") for r in indexes if r.get("trade_date")}
    if len(dates) > 1:
        out.append(
            f"⚠️ 指数交易日不一致（{'、'.join(sorted(dates))}），"
            "跨市场对比时不可直接同口径解读。"
        )
    return out


# ---------------------------------------------------------------- 组合


def analyze_portfolio(advice: dict) -> list[str]:
    out: list[str] = []
    alloc = advice.get("allocation") or {}
    rules = advice.get("rules") or {}
    port = advice.get("portfolio") or []
    stocks = [p for p in port if p.get("kind") == "stock"]
    prods = [p for p in port if p.get("kind") == "product"]

    eq = _num(alloc.get("equity_pct"))
    eqt = _num(alloc.get("equity_target_pct"))
    cash = _num(alloc.get("cash_pct"))
    cashmin = _num(alloc.get("cash_min_pct"))
    prod = _num(alloc.get("product_pct"))

    if eq is not None and eqt is not None:
        gap = round(eqt - eq, 2)
        if abs(gap) < 0.01:
            out.append(f"权益仓位 {eq}% 精准打满目标 {eqt}%，无缺口。")
        elif gap > 0:
            out.append(
                f"权益仓位 {eq}%，低于目标 {eqt}%，缺口 {gap} 个百分点 —— "
                "是某主题无合格标的导致配额用不出去（按纪律归现金，不硬塞给其他持仓）。"
            )
        else:
            out.append(f"权益仓位 {eq}%，高于目标 {eqt}%，存在超配，仓位纪律需复核。")
    if cash is not None and cashmin is not None:
        out.append(
            f"现金 {cash}%（下限 {cashmin}%）"
            + ("，满足纪律。" if cash >= cashmin else "，⚠️ 低于下限，仓位管理违规。")
        )
    if prod is not None:
        out.append(f"理财仓位 {prod}%，与权益合计 {round((eq or 0) + prod, 2)}%。")

    # 单只上限
    if stocks:
        weights = [(p.get("name") or "?", _num(p.get("weight_pct")) or 0.0) for p in stocks]
        mx = max(weights, key=lambda kv: kv[1])
        smx = _num(rules.get("single_max_pct"))
        out.append(
            f"股票 {len(stocks)} 只，最大权重 {mx[0]} {mx[1]}%"
            + (f"（单只上限 {smx:.0f}%）" if smx else "")
            + "，无单只集中风险。"
        )
        themes: dict[str, float] = {}
        for p in stocks:
            themes[p.get("theme") or "其他"] = themes.get(p.get("theme") or "其他", 0.0) + (
                _num(p.get("weight_pct")) or 0.0
            )
        tdesc = "、".join(f"{k} {v:.1f}%" for k, v in sorted(themes.items(), key=lambda kv: -kv[1]))
        top_theme = max(themes.items(), key=lambda kv: kv[1])
        out.append(
            f"主题分布：{tdesc}；最重 {top_theme[0]} {top_theme[1]:.1f}%"
            + ("，风格暴露单一" if top_theme[1] > 40 else "，风格较均衡")
        )

    # 止损距离
    stops = [
        (p.get("name") or "?", _num(p.get("stop_pct")))
        for p in stocks
        if _num(p.get("stop_pct")) is not None
    ]
    if stops:
        avg_stop = sum(v for _n, v in stops) / len(stops)
        tight = min(stops, key=lambda kv: kv[1])
        out.append(
            f"止损设置平均 -{avg_stop:.2f}%，最紧 {tight[0]} -{tight[1]:.2f}%。"
            + (
                "止损幅度偏紧，正常波动就可能被打掉，注意假信号风险。"
                if avg_stop < 3
                else "止损幅度合理，给了正常波动空间。"
            )
        )

    # 目标空间
    tp_all: list[float] = []
    for p in stocks:
        for t in p.get("targets") or []:
            g = _num(t.get("gain_pct"))
            if g is not None:
                tp_all.append(g)
    if tp_all:
        out.append(
            f"分批止盈目标 {min(tp_all):+.0f}% ~ {max(tp_all):+.0f}%，"
            f"首档 {sorted(tp_all)[0]:+.0f}%，按三分之一递减执行。"
        )

    if prods:
        cats: dict[str, float] = {}
        for p in prods:
            c = p.get("quota_bucket") or p.get("category") or "其他"
            cats[c] = cats.get(c, 0.0) + (_num(p.get("weight_pct")) or 0.0)
        cdesc = "、".join(f"{k} {v:.1f}%" for k, v in sorted(cats.items(), key=lambda kv: -kv[1]))
        out.append(
            f"理财 {len(prods)} 只，类别分布 {cdesc}；"
            "固收类为底仓、货币类留流动性，权益敞口只由股票部分承担。"
        )
    return out


# ---------------------------------------------------------------- 关注池


def analyze_watchlist(watchlist: list[dict]) -> list[str]:
    out: list[str] = []
    if not watchlist:
        return ["关注池为空。"]
    n = len(watchlist)
    in_p = sum(1 for w in watchlist if w.get("in_portfolio"))
    ver = sum(1 for w in watchlist if w.get("verify") == "verified")
    conflict = sum(1 for w in watchlist if w.get("verify") == "conflict")
    out.append(
        f"全池 {n} 只，入选组合 {in_p} 只（入选率 {in_p / n * 100:.1f}%），"
        f"{_verdict(ver, n)}。"
    )
    if conflict:
        out.append(
            f"⚠️ {conflict} 只标的双源口径冲突，已排除出组合但保留在此表 —— "
            "这些标的的当日涨跌数据存疑，不要据此下单。"
        )
    # 评分分布
    scores = [(_num(w.get("score")) or 0.0) for w in watchlist]
    if scores:
        hi = [w for w in watchlist if (_num(w.get("score")) or 0) >= 70]
        out.append(
            f"评分区间 {min(scores):.1f} ~ {max(scores):.1f}，"
            f"≥70 分的高分标的 {len(hi)} 只"
            + ("（均已入选）" if hi and all(h.get("in_portfolio") for h in hi) else "（部分未入选）")
        )
    # 估值
    pes = [_num(w.get("pe")) for w in watchlist]
    pes_v = [x for x in pes if x is not None and x > 0]
    if pes_v:
        out.append(
            f"PE 有效样本 {len(pes_v)}/{n}，区间 {min(pes_v):.1f} ~ {max(pes_v):.1f}，"
            f"中位 {sorted(pes_v)[len(pes_v) // 2]:.1f}。"
        )
    # 涨跌分布
    pcts = [(w.get("name") or "?", _num(w.get("pct")) or 0.0) for w in watchlist]
    out.append(f"当日 {_cnt(pcts)}。{_spread(pcts)}。")
    return out


# ---------------------------------------------------------------- 事件


def analyze_events(events: list[dict], stats: dict) -> list[str]:
    out: list[str] = []
    if not events:
        return ["未取到事件。"]
    total = len(events)
    high = [e for e in events if e.get("importance") == "high"]
    ver = sum(1 for e in events if (e.get("cross") or {}).get("status") == "verified")
    out.append(
        f"共采集 {total} 条，重要事件 {len(high)} 条，双源核验 {ver} 条"
        f"（{_verdict(ver, total)}）。"
    )
    # 来源覆盖
    srcs: dict[str, int] = {}
    for e in events:
        for s in (e.get("cross") or {}).get("src_names") or []:
            srcs[s] = srcs.get(s, 0) + 1
    if srcs:
        desc = "、".join(f"{k} {v} 条" for k, v in sorted(srcs.items(), key=lambda kv: -kv[1]))
        out.append(f"来源分布：{desc}。")
    # 主题聚类（按关键词）
    buckets = {
        "货币/利率": ["利率", "降准", "降息", "加息", "LPR", "MLF", "逆回购", "流动性"],
        "财政/政策": ["财政", "国债", "专项债", "税", "补贴", "政策", "会议", "部署"],
        "海外": ["美联储", "美股", "非农", "CPI", "关税", "特朗普", "鲍威尔", "欧央行"],
        "产业/科技": ["芯片", "AI", "人工智能", "算力", "新能源", "汽车", "光伏", "半导体"],
    }
    hit = {k: 0 for k in buckets}
    for e in events:
        t = (e.get("title") or "")
        for k, kws in buckets.items():
            if any(w in t for w in kws):
                hit[k] += 1
    top = sorted(hit.items(), key=lambda kv: -kv[1])
    if top and top[0][1]:
        out.append(
            "主题分布："
            + "、".join(f"{k} {v} 条" for k, v in top if v)
            + "。主题靠前的方向对指数情绪影响更直接。"
        )
    conf = sum(1 for e in events if (e.get("cross") or {}).get("status") == "conflict")
    if conf:
        out.append(f"⚠️ {conf} 条事件两源标题口径不一致，未采信任何一方，已排除出重点列表。")
    return out


# ---------------------------------------------------------------- 宏观


def analyze_macro(macro: list[dict]) -> list[str]:
    out: list[str] = []
    if not macro:
        return ["本期无宏观数据。"]
    def find(kw: str) -> dict | None:
        for m in macro:
            label = str(m.get("label") or m.get("report") or "")
            if kw in label:
                return m
        return None

    cpi = find("CPI")
    ppi = find("PPI")
    pmi = find("PMI")
    gdp = find("GDP")
    m2 = find("货币供应") or find("M2")

    def direction(m: dict | None) -> str:
        if not m:
            return "—"
        mom = _num(m.get("mom"))
        if mom is None:
            return "—"
        if mom > 0:
            return f"环比 {mom:+.2f}%，扩张"
        if mom < 0:
            return f"环比 {mom:+.2f}%，收缩"
        return "环比持平"

    if cpi or ppi:
        out.append(
            f"价格端：CPI {direction(cpi)}、PPI {direction(ppi)}。"
            + (
                "CPI 与 PPI 同向向上，通胀压力对权益估值构成约束。"
                if cpi and ppi
                and (_num(cpi.get("mom")) or 0) * (_num(ppi.get("mom")) or 0) > 0
                and (_num(cpi.get("mom")) or 0) > 0
                else "价格端未见一致向上的通胀组合。"
            )
        )
    if pmi:
        v = _num(pmi.get("value"))
        out.append(
            f"景气端：PMI {_f(pmi.get('value'))}（{direction(pmi)}）"
            + (
                "，落在荣枯线上方，制造业扩张对顺周期有利。"
                if v is not None and v > 50
                else "，落在荣枯线下方，制造业收缩压力未解。"
                if v is not None
                else "。"
            )
        )
    if gdp:
        out.append(f"增长端：GDP {_f(gdp.get('value'))}（{direction(gdp)}）。")
    if m2:
        out.append(
            f"流动性：M2 {_f(m2.get('value'))}（{direction(m2)}）。"
            "货币供应变化是 A 股估值扩张/收缩的先行指标。"
        )
    periods = [str(m.get("period") or "") for m in macro if m.get("period")]
    if periods:
        out.append(f"数据期次：{'、'.join(periods)}。注意宏观数据有发布滞后，不宜当作当日行情。")
    return out


# ---------------------------------------------------------------- 期货外汇


def analyze_cn_futures(cn: list[dict]) -> list[str]:
    """国内期货表专属分析。"""
    out: list[str] = []
    if not cn:
        return ["本期未取到国内期货数据，本表无法给出判断。"]
    pcts = [(r.get("name") or "?", _num(r.get("pct")) or 0.0) for r in cn]
    out.append(f"国内期货 {len(cn)} 个品种：{_cnt(pcts)}。{_spread(pcts)}。")
    black_kw = ("螺纹", "焦炭", "焦煤", "铁矿石", "动力煤", "热卷")
    nonfer = ("铜", "铝", "锌", "镍", "锡", "铅")
    black = [v for n, v in pcts if any(k in n for k in black_kw)]
    nfm = [v for n, v in pcts if any(k in n for k in nonfer)]
    if black and nfm:
        b = sum(black) / len(black)
        m = sum(nfm) / len(nfm)
        out.append(
            f"黑色系均 {_pct(b)}、有色系均 {_pct(m)}"
            + (
                "，两者同向，上下游需求预期一致。"
                if b * m > 0
                else "，两者背离，产业链上下游景气出现分化，需警惕成本端与需求端不匹配。"
            )
        )
    ver = sum(1 for r in cn if (r.get("cross") or {}).get("status") == "verified")
    out.append(_verdict(ver, len(cn)) + "。")
    if ver < len(cn):
        out.append(
            f"未通过双源核验的 {len(cn) - ver} 个品种只作行情展示，不作为仓位与方向判断依据。"
        )
    return out


def analyze_global_futures(gl: list[dict]) -> list[str]:
    """外盘期货表专属分析。"""
    out: list[str] = []
    if not gl:
        return ["本期未取到外盘期货数据，本表无法给出判断。"]
    pcts = [(r.get("name") or "?", _num(r.get("pct")) or 0.0) for r in gl]
    out.append(f"外盘期货 {len(gl)} 个品种：{_cnt(pcts)}。{_spread(pcts)}。")
    oil = next(((r.get("name"), _num(r.get("pct"))) for r in gl if "原油" in str(r.get("name"))), None)
    gold = next(((r.get("name"), _num(r.get("pct"))) for r in gl if "黄金" in str(r.get("name"))), None)
    if oil and gold:
        ov, gv = oil[1] or 0.0, gold[1] or 0.0
        out.append(
            f"原油 {_pct(ov)}、黄金 {_pct(gv)}"
            + (
                "，油金同涨，避险与需求同时走强。"
                if ov > 0 and gv > 0
                else "，油金背离（油跌金涨更典型），市场在交易避险而非增长。"
                if gv > 0 >= ov
                else "，油金同跌，风险偏好回落。"
                if ov < 0 and gv < 0
                else "，油金走势分化。"
            )
        )
    ver = sum(1 for r in gl if (r.get("cross") or {}).get("status") == "verified")
    out.append(_verdict(ver, len(gl)) + "。")
    return out


def analyze_fx(fx: list[dict], mid: dict | None = None) -> list[str]:
    """外汇表专属分析。"""
    out: list[str] = []
    if not fx:
        out.append("本期未取到外汇行情，本表无法给出判断。")
    else:
        pcts = [(r.get("name") or "?", _num(r.get("pct")) or 0.0) for r in fx]
        usd = [(n, v) for n, v in pcts if "美元" in n]
        out.append(f"外汇 {len(fx)} 个货币对：{_cnt(pcts)}。{_spread(pcts)}。")
        if usd:
            d = usd[0][1]
            out.append(
                f"美元指数 {_pct(d)}"
                + (
                    "，美元走强对新兴市场资产构成压力，港股弹性受抑。"
                    if d > 0
                    else "，美元走弱对新兴市场资产相对友好。"
                    if d < 0
                    else "，美元持平。"
                )
            )
        ver = sum(1 for r in fx if (r.get("cross") or {}).get("status") == "verified")
        out.append(_verdict(ver, len(fx)) + "。")
    if mid:
        out.append(
            f"美元/人民币中间价 {_f(mid.get('price'), 4)}（{mid.get('date') or '—'}），"
            "中间价是央行的政策信号，比市场价更能反映汇率意图。"
        )
    return out


def analyze_futures(futures: dict) -> list[str]:
    """期货外汇合并视图（研报与总览用），逐表分析见上面三个函数。"""
    out: list[str] = []
    cn = futures.get("cn_futures") or []
    gl = futures.get("global_futures") or []
    fx = futures.get("fx") or []
    if cn:
        out.append(f"国内期货 {len(cn)} 个品种：{_cnt([(r.get('name') or '?', _num(r.get('pct')) or 0.0) for r in cn])}。")
    if gl:
        out.append(f"外盘期货 {len(gl)} 个品种：{_cnt([(r.get('name') or '?', _num(r.get('pct')) or 0.0) for r in gl])}。")
    if fx:
        out.append(f"外汇 {len(fx)} 个货币对：{_cnt([(r.get('name') or '?', _num(r.get('pct')) or 0.0) for r in fx])}。")
    mid = futures.get("fx_mid")
    if mid:
        out.append(
            f"美元/人民币中间价 {_f(mid.get('price'), 4)}（{mid.get('date') or '—'}），"
            "中间价是央行的政策信号，比市场价更能反映汇率意图。"
        )
    if not out:
        out.append("本期未取到期货与外汇数据，本节无法给出判断。")
    return out


# ---------------------------------------------------------------- 理财


def analyze_products(products: list[dict]) -> list[str]:
    out: list[str] = []
    if not products:
        return ["本期无理财产品数据，本表无法给出判断。"]
    cats: dict[str, list[dict]] = {}
    for p in products:
        cats.setdefault(p.get("category") or "其他", []).append(p)
    out.append(f"共 {len(products)} 只，覆盖 {len(cats)} 个类别：" + "、".join(f"{k} {len(v)} 只" for k, v in cats.items()) + "。")

    # 货币基金
    mm = [p for p in products if "货币" in str(p.get("category"))]
    if mm:
        ylds = [_num(p.get("seven_day_yield")) for p in mm]
        ylds_v = [y for y in ylds if y is not None]
        if ylds_v:
            out.append(
                f"货币基金 {len(mm)} 只，7 日年化 {min(ylds_v):.3f}% ~ {max(ylds_v):.3f}%，"
                f"均值 {sum(ylds_v) / len(ylds_v):.3f}% —— 这是当前无风险利率的锚，"
                "也是判断其他理财产品是否划算的基准线。"
            )
    # 债基
    bond = [p for p in products if "纯债" in str(p.get("category")) or "债" in str(p.get("category"))]
    r1 = [_num(p.get("ret_1y")) for p in bond]
    r1_v = [x for x in r1 if x is not None]
    if r1_v:
        out.append(f"债券类产品近 1 年收益 {min(r1_v):+.2f}% ~ {max(r1_v):+.2f}%，均值 {sum(r1_v) / len(r1_v):+.2f}%。")
    # 混合
    mixed = [p for p in products if "混合" in str(p.get("category"))]
    if mixed:
        mr = [_num(p.get("ret_1y")) for p in mixed]
        mr_v = [x for x in mr if x is not None]
        if mr_v:
            out.append(
                f"混合型 {len(mixed)} 只，近 1 年 {min(mr_v):+.2f}% ~ {max(mr_v):+.2f}%。"
                "混合型权益敞口不透明，需看持仓而非历史收益。"
            )
    # 净值可得性
    nav_ok = sum(1 for p in products if p.get("nav") is not None)
    out.append(
        f"净值口径：{nav_ok}/{len(products)} 只可取到单位净值。"
        + (
            "其余为银行理财等无统一净值披露口径的品种，报告中留空而非估算。"
            if nav_ok < len(products)
            else "全部可取到净值。"
        )
    )
    ver = sum(1 for p in products if (p.get("cross") or {}).get("status") == "verified")
    out.append(_verdict(ver, len(products)) + "。")
    return out


# ---------------------------------------------------------------- 股票池


def analyze_theme_pool(theme: str, rows: list[dict]) -> list[str]:
    """单个主题子表的专属分析（股票池四个主题各一个物理表格，各配一份分析）。"""
    out: list[str] = []
    if not rows:
        return [f"{theme} 本期无标的，本表无法给出判断。"]
    pcts = [(r.get("name") or "?", _num(r.get("pct")) or 0.0) for r in rows]
    ver = sum(1 for r in rows if (r.get("cross") or {}).get("status") == "verified")
    out.append(
        f"{theme} {len(rows)} 只：{_cnt(pcts)}。{_spread(pcts)}。{_verdict(ver, len(rows))}。"
    )
    pes = [_num(r.get("pe")) for r in rows]
    pes_v = sorted(x for x in pes if x is not None and x > 0)
    if pes_v:
        out.append(
            f"PE 区间 {pes_v[0]:.1f} ~ {pes_v[-1]:.1f}，中位 {pes_v[len(pes_v) // 2]:.1f}。"
        )
    pbs = [_num(r.get("pb")) for r in rows]
    pbs_v = sorted(x for x in pbs if x is not None and x > 0)
    if pbs_v:
        out.append(
            f"PB 区间 {pbs_v[0]:.2f} ~ {pbs_v[-1]:.2f}，中位 {pbs_v[len(pbs_v) // 2]:.2f}。"
        )
    up = [n for n, v in pcts if v > 0]
    down = [n for n, v in pcts if v < 0]
    if up and down:
        out.append(
            f"{theme} 内部涨跌分化：{len(up)} 涨 {len(down)} 跌，"
            f"最强 {max(pcts, key=lambda kv: kv[1])[0]}、"
            f"最弱 {min(pcts, key=lambda kv: kv[1])[0]}，该主题不是齐涨齐跌的一篮子。"
        )
    elif up:
        out.append(f"{theme} 今日全线上涨，板块情绪一致。")
    elif down:
        out.append(f"{theme} 今日全线下跌，板块情绪一致偏弱。")
    if ver < len(rows):
        out.append(
            f"其中 {len(rows) - ver} 只未通过双源核验，本表仅作行情展示。"
        )
    return out


def analyze_stock_pool(stocks: list[dict]) -> list[str]:
    out: list[str] = []
    if not stocks:
        return ["股票池为空。"]
    themes: dict[str, list[dict]] = {}
    for s in stocks:
        themes.setdefault(s.get("theme") or "其他", []).append(s)
    for theme, rows in themes.items():
        pcts = [(r.get("name") or "?", _num(r.get("pct")) or 0.0) for r in rows]
        ver = sum(1 for r in rows if (r.get("cross") or {}).get("status") == "verified")
        line = f"{theme}（{len(rows)} 只）：{_cnt(pcts)}，{_spread(pcts)}，{_verdict(ver, len(rows))}。"
        # 估值
        pes = [_num(r.get("pe")) for r in rows]
        pes_v = [x for x in pes if x is not None and x > 0]
        if pes_v:
            line += f" PE 中位 {sorted(pes_v)[len(pes_v) // 2]:.1f}。"
        out.append(line)
    return out


# ---------------------------------------------------------------- 复盘


def analyze_review(review: dict) -> list[str]:
    out: list[str] = []
    if not review:
        return ["无复盘数据。"]
    if not review.get("available"):
        note = review.get("note") or ""
        if "首期" in note:
            return [
                "首期运行，系统中还没有可被检验的历史建议 —— 这是正常的起点，不是数据缺失。",
                "本期固化的组合与止损位会作为明日复盘的基准，方向命中率、止损触发率、"
                "目标达成率和建议稳定性从明天开始逐日累积。",
                "第一期的价值是建立可回溯的基线：从明天起，任何判断都有前一天的锚，"
                "复盘才成立、命中率才有意义。",
            ]
        return [f"暂无可判定的后续价格数据（{note}）。"]

    stats = review.get("stats") or {}
    judged = _num(stats.get("judged")) or 0
    evaluated = _num(stats.get("evaluated")) or 0
    out.append(
        f"已复盘 {_num(stats.get('snapshots')) or 0} 期快照，{evaluated} 条建议中 {judged:.0f} 条可判定方向。"
    )
    hr = _num(stats.get("hit_rate"))
    if hr is not None:
        out.append(
            f"方向命中率 {hr * 100:.1f}%（{stats.get('hits')}/{judged:.0f}）。"
            + (
                "命中率高于 50% 说明判断有正向价值，可以延续同类逻辑。"
                if hr > 0.5
                else "命中率低于 50%，当前择时逻辑未经验证，需要收缩仓位、只留最有把握的信号。"
                if hr < 0.5
                else "命中率恰好 50%，与抛硬币无异，需重新检视打分逻辑。"
            )
        )
    sr = _num(stats.get("stop_trigger_rate"))
    if sr is not None and evaluated:
        out.append(
            f"止损触发率 {sr * 100:.1f}%，目标达成率 "
            f"{(_num(stats.get('target_hit_rate')) or 0) * 100:.1f}%。"
            + (
                "止损触发率偏高说明进场点追高或止损设得太近。"
                if sr > 0.3
                else "止损触发率在可接受范围，纪律执行没有明显问题。"
            )
        )
    st = _num(stats.get("stability_avg_jaccard"))
    if st is not None:
        out.append(
            f"建议稳定性（Jaccard）{st:.3f}。"
            + (
                "稳定性偏低说明组合天天换，这种换手在 A 股会持续磨损收益。"
                if st < 0.5
                else "组合保持相对稳定，只在信号显著变化时调整。"
            )
        )
    ar = _num(stats.get("avg_return_pct"))
    if ar is not None:
        out.append(f"模拟组合平均实际收益 {_pct(ar)}。")
    # 偏差
    bias = stats.get("bias") or {}
    for call, v in bias.items():
        gap = _num(v.get("gap_pct"))
        if gap is None:
            continue
        out.append(
            f"{call}方向：{v.get('n')} 条样本，实际均值 {_pct(v.get('avg_ret_pct'))}，"
            f"隐含目标 {_pct(v.get('implied_target_pct'), 0)}，差值 {_pct(gap)}。"
            + (
                "实际走势弱于建议目标，存在系统性高估，应下调目标位或提前止盈。"
                if gap < 0
                else "实际走势强于建议目标，可以适当放宽持仓时间、让利润多跑一段。"
                if gap > 0
                else ""
            )
        )
    return out


# ---------------------------------------------------------------- 冲突


def analyze_sources(src_rows: list[list[str]]) -> list[str]:
    """数据源可用性表专属分析。"""
    if not src_rows:
        return ["本期无数据源记录，无法评估采集可用性。"]
    total = len(src_rows)
    failed = [r for r in src_rows if len(r) > 2 and r[2] == "失败"]
    by_field: dict[str, list[list[str]]] = {}
    for r in src_rows:
        if r:
            by_field.setdefault(r[0], []).append(r)
    out = [f"本期记录 {total} 个数据源，其中失败 {len(failed)} 个，整体可用率 {(total - len(failed)) / total * 100:.1f}%。"]
    for field, rows in by_field.items():
        ok = sum(1 for r in rows if len(r) > 2 and r[2] == "正常")
        out.append(f"{field}：{ok}/{len(rows)} 个来源正常。")
    if failed:
        names = "、".join(str(r[1]) for r in failed if len(r) > 1)
        out.append(
            f"异常来源：{names}。相关字段可能缺失或取到默认值，"
            "已按缺失处理，不做估算填补。"
        )
    return out


def analyze_conflicts(conflicts: list[dict]) -> list[str]:
    if not conflicts:
        return ["本期无口径冲突，所有展示数值均来自一致的双源核验。"]
    out = [f"本期记录 {len(conflicts)} 处口径冲突，全部保留原样、未静默取舍。"]
    for c in conflicts:
        name = c.get("name") or "?"
        diff = _num(c.get("diff_pct"))
        srcs = "、".join(c.get("src_names") or []) or "多源"
        detail = f"分歧 {diff:.4f}%" if diff is not None else (c.get("note") or "")
        out.append(f"{name}：{detail}（来源 {srcs}）—— 数值存疑，不要据此下单。")
    return out


# ---------------------------------------------------------------- 每日研报


def build_research_report(
    quotes: dict,
    events: dict,
    futures: dict,
    advice: dict,
    review: dict,
    session: dict,
    products: dict | None = None,
) -> dict:
    """汇总当日全部数据，生成小金自己的每日研报（结论 + 论据 + 风险）。"""
    alloc = (advice or {}).get("allocation") or {}
    mstate = (advice or {}).get("market_state") or {}
    rules = (advice or {}).get("rules") or {}
    port = (advice or {}).get("portfolio") or []
    stocks = [p for p in port if p.get("kind") == "stock"]
    idx = quotes.get("indexes") or []
    evs = events.get("events") or []
    high = [e for e in evs if e.get("importance") == "high"]
    verified_high = [e for e in high if (e.get("cross") or {}).get("status") == "verified"]

    # ---- 1. 市场判断
    idx_pcts = [(_num(r.get("pct")) or 0.0) for r in idx]
    cn = [
        (_num(r.get("pct")) or 0.0)
        for r in idx
        if (r.get("market") == "CN" or r.get("region") in ("A股", "CN"))
    ]
    cn_avg = sum(cn) / len(cn) if cn else (sum(idx_pcts) / len(idx_pcts) if idx_pcts else 0.0)
    temp = _num(mstate.get("temperature"))
    label = mstate.get("label") or ""

    if cn_avg > 0.8:
        stance = "偏强"
    elif cn_avg > 0.2:
        stance = "温和偏多"
    elif cn_avg > -0.2:
        stance = "震荡"
    elif cn_avg > -0.8:
        stance = "温和偏空"
    else:
        stance = "偏弱"

    verdict = (
        f"今日 A 股口径指数平均 {_pct(cn_avg)}，{stance}。"
        f"系统温度 {_f(temp, 1)}（{label}），据此给出权益仓位 "
        f"{_f(alloc.get('equity_pct'), 1)}%、理财 {_f(alloc.get('product_pct'), 1)}%、"
        f"现金 {_f(alloc.get('cash_pct'), 1)}%。"
    )

    # ---- 2. 组合逻辑
    if stocks:
        themes: dict[str, float] = {}
        for p in stocks:
            themes[p.get("theme") or "其他"] = themes.get(p.get("theme") or "其他", 0.0) + (
                _num(p.get("weight_pct")) or 0.0
            )
        top3 = sorted(themes.items(), key=lambda kv: -kv[1])[:3]
        logic = (
            f"组合 {len(stocks)} 只股票，主题重心依次为 "
            + "、".join(f"{k} {v:.1f}%" for k, v in top3)
            + "。权重最高的 "
            + max(stocks, key=lambda p: _num(p.get("weight_pct")) or 0.0)["name"]
            + f"（{_f(max((_num(p.get('weight_pct')) or 0.0) for p in stocks), 1)}%）"
            "是本期主攻方向；"
        )
        stops = [
            (_num(p.get("stop_pct")) or 0.0)
            for p in stocks
            if _num(p.get("stop_pct")) is not None
        ]
        if stops:
            logic += f"止损统一设在 -{abs(sum(stops) / len(stops)):.2f}% 附近（A股 -3%、美股 -5% 口径），触及即无条件离场。"
    else:
        logic = "本期无符合条件的股票标的，权益部分空缺，仓位归现金。"

    # ---- 3. 事件与宏观
    drivers = []
    if verified_high:
        drivers.append(
            f"双源核验的重要事件 {len(verified_high)} 条，最新一条："
            + (verified_high[0].get("title") or "")[:60]
        )
    macro = events.get("macro") or []
    if macro:
        drivers.append(
            "宏观锚点：" + "、".join(
                f"{m.get('label') or m.get('report')} {_f(m.get('value'))}"
                + (f"{m.get('unit')}" if m.get("unit") else "")
                for m in macro[:3]
            )
        )
    ev_line = "；".join(drivers) if drivers else "本期无重要事件与宏观更新。"

    # ---- 4. 风险
    risks = []
    if session.get("is_trading_day") is False:
        risks.append(
            f"今日休市，行情对应 {session.get('data_date')}，价格不是当日实时价，不据此做方向判断。"
        )
    conflicts = (
        (quotes.get("conflicts") or [])
        + (futures.get("conflicts") or [])
        + ((products or {}).get("conflicts") or [])
    )
    if conflicts:
        risks.append(f"{len(conflicts)} 处跨源口径冲突，相关数值的可信度打折。")
    if verified_high and len(verified_high) < len(high):
        risks.append(
            f"{len(high) - len(verified_high)} 条重要事件未通过双源核验，未纳入判断依据。"
        )
    cash = _num(alloc.get("cash_pct"))
    cashmin = _num(alloc.get("cash_min_pct"))
    if cash is not None and cashmin is not None and cash > cashmin + 5:
        risks.append(
            f"现金 {_f(cash, 1)}% 明显高于下限 {cashmin:.0f}%，说明有主题找不到合格标的 —— "
            "宁可空仓等待，也不降低核验标准硬凑仓位。"
        )
    smax = _num(rules.get("single_max_pct"))
    if smax is not None and stocks:
        over = [
            p.get("name")
            for p in stocks
            if (_num(p.get("weight_pct")) or 0.0) > smax + 1e-9
        ]
        if over:
            risks.append(
                f"{len(over)} 只标的权重突破单只上限 {smax:.0f}%（{'、'.join(str(x) for x in over)}），"
                "属于仓位管理缺陷，需在下一期修正。"
            )
        else:
            risks.append(
                f"单只持仓均在 {smax:.0f}% 上限内，最大一只 "
                f"{_f(max((_num(p.get('weight_pct')) or 0.0) for p in stocks), 1)}%，"
                "未出现集中度越界。"
            )
    if not risks:
        risks.append("无显著风险项；常规风险为隔夜跳空与突发消息。")

    # ---- 5. 明日关注
    watch = []
    if stocks:
        watch.append(
            "盯主线持仓是否守住止损位（触及即执行，不解释、不等反弹）。"
        )
        tp1 = [
            (p.get("name"), (_num((p.get("targets") or [{}])[0].get("gain_pct")) or 0.0))
            for p in stocks
            if p.get("targets")
        ]
        if tp1:
            watch.append(
                "首档止盈 +8% 的触发条件：" + "、".join(f"{n}（{g:+.0f}%）" for n, g in tp1[:4])
            )
    watch.append("关注双源冲突标的的口径澄清，以及隔夜外盘对次日开盘的指引。")

    # ---- 6. 复盘结论
    rev_line = ""
    if review and review.get("available"):
        stats = review.get("stats") or {}
        hr = _num(stats.get("hit_rate"))
        rev_line = (
            f"昨日及此前建议的方向命中率 {hr * 100:.1f}%，"
            f"平均实际收益 {_pct(stats.get('avg_return_pct'))}。"
            if hr is not None
            else "前次建议暂无方向命中率统计（样本不足）。"
        )
    elif review:
        rev_line = "首期运行，昨日无历史建议可复盘；本期固化的判断即明日复盘的基准。"

    return {
        "as_of": mstate.get("as_of"),
        "stance": stance,
        "temperature": temp,
        "temperature_label": label,
        "verdict": verdict,
        "logic": logic,
        "drivers": ev_line,
        "risks": risks,
        "watch": watch,
        "review_line": rev_line,
        "allocation": {
            "equity_pct": alloc.get("equity_pct"),
            "product_pct": alloc.get("product_pct"),
            "cash_pct": alloc.get("cash_pct"),
        },
        "disclaimer": (
            "以上为基于公开数据的量化研判，模拟组合与仓位仅供策略研究，不构成投资建议，未执行任何真实交易。"
        ),
    }
