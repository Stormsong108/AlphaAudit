"""模拟组合建议引擎（研究性质）。

铁律：
1. 全部输出为研究/模拟，不构成投资建议，不执行任何真实交易。
2. 只使用采集到的真实字段；字段缺失就标注缺失，绝不臆造价格或估值。
3. 双源核验出现 conflict 的标的，一律不进模拟组合（只留在关注清单里说明原因）。
4. 权重、止损、目标位全部来自 config.RISK，任何一条建议都能复算。
"""
from __future__ import annotations

import time
from typing import Any

import config as C

DISCLAIMER = (
    "以下为研究性模拟组合，仅用于方法演示与事后复盘，不构成投资建议，"
    "不执行任何真实交易。"
)

# 权益内部的主题配额（合计 1.0）
THEME_QUOTA = {
    "大盘蓝筹": 0.34,
    "科技成长": 0.26,
    "红利低波": 0.24,
    "港股互联网": 0.16,
}

# 理财产品（非权益）内部配额 —— 覆盖用户要求的四类
PRODUCT_QUOTA = {
    "货币基金": 0.20,
    "固收+": 0.25,   # 银行理财固定收益类（登记口径）+ 公募二级债基（净值可双源核验）
    "纯债基金": 0.35,
    "混合基金": 0.20,
}

# 实际类别 → 配额键。未列出的类别（如「权益基金」）不占非权益仓位。
PRODUCT_CATEGORY_ALIAS = {
    "货币基金": "货币基金",
    "银行理财固收+": "固收+",
    "固收+基金": "固收+",
    "纯债基金": "纯债基金",
    "混合基金": "混合基金",
}

MAX_PER_THEME = 3

# 单只持仓的最小权重（低于这个数没有实际意义）
MIN_POSITION_PCT = 3.0


# ------------------------------------------------------------------ 打分
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _momentum_score(pct: float | None) -> float | None:
    """动量分 0-50：-3% 得 0，+3% 得 50。"""
    if pct is None:
        return None
    return _clamp((pct + 3.0) / 6.0 * 50.0, 0.0, 50.0)


def _valuation_score(pe: float | None) -> tuple[float, str]:
    """估值分 0-30：PE 越低分越高。无 PE 给中位分并注明。"""
    if pe is None:
        return 12.0, "无PE数据，估值按中位计"
    if pe <= 0:
        return 0.0, "PE为负（亏损），估值不给分"
    if pe <= 15:
        return 30.0, f"PE {pe:.1f}，低估区间"
    if pe <= 25:
        return 22.0, f"PE {pe:.1f}，合理区间"
    if pe <= 40:
        return 14.0, f"PE {pe:.1f}，偏高"
    return 8.0, f"PE {pe:.1f}，高估区间"


def _liquidity_score(amount: float | None) -> tuple[float, str]:
    """流动性分 0-10：按当日成交额（东财 f6，单位元）。"""
    if not amount:
        return 4.0, "无成交额数据"
    yi = amount / 1e8
    if yi >= 50:
        return 10.0, f"成交额 {yi:.0f}亿，流动性充裕"
    if yi >= 20:
        return 8.0, f"成交额 {yi:.0f}亿，流动性良好"
    if yi >= 5:
        return 5.0, f"成交额 {yi:.0f}亿，流动性一般"
    return 2.0, f"成交额 {yi:.0f}亿，流动性偏弱"


def _verify_score(cross: dict | None) -> tuple[float, str]:
    st = (cross or {}).get("status")
    if st == "verified":
        return 10.0, "双源核验通过"
    if st == "conflict":
        return 0.0, "双源存在口径冲突"
    if st == "single_source":
        return 4.0, "仅单源，未达双源要求"
    return 4.0, "无核验记录"


def score_stock(row: dict) -> dict:
    """给一只个股打分，返回分数与可读依据。"""
    pct = row.get("pct")
    mom = _momentum_score(pct)
    if mom is None:
        mom = 15.0
        mom_note = "无涨跌幅数据，动量按中位计"
    else:
        mom_note = f"当日涨跌 {pct:+.2f}%"
    val, val_note = _valuation_score(row.get("pe"))
    liq, liq_note = _liquidity_score(row.get("amount"))
    ver, ver_note = _verify_score(row.get("cross"))
    total = round(mom + val + liq + ver, 1)
    return {
        "score": total,
        "breakdown": {
            "momentum": round(mom, 1),
            "valuation": val,
            "liquidity": liq,
            "verify": ver,
        },
        "notes": [mom_note, val_note, liq_note, ver_note],
    }


# ------------------------------------------------------------------ 大盘温度
def _index_pct(indexes: list[dict], name: str) -> float | None:
    for r in indexes:
        if r.get("name") == name:
            return r.get("pct")
    return None


def _index_price(indexes: list[dict], name: str) -> float | None:
    for r in indexes:
        if r.get("name") == name:
            return r.get("price")
    return None


def market_state(indexes: list[dict]) -> dict:
    """用指数涨跌 + VIX 水平算一个 0-100 的温度分。"""
    sh = _index_pct(indexes, "上证指数")
    hs300 = _index_pct(indexes, "沪深300")
    ndx = _index_pct(indexes, "纳斯达克100")
    vix = _index_price(indexes, "VIX恐慌指数")

    def _pct_score(v: float | None, span: float) -> float:
        if v is None:
            return 50.0
        return _clamp((v + span) / (2 * span) * 100.0, 0.0, 100.0)

    def _vix_score(v: float | None) -> float:
        if v is None:
            return 50.0
        if v <= 14:
            return 85.0
        if v <= 18:
            return 70.0
        if v <= 22:
            return 50.0
        if v <= 28:
            return 30.0
        return 15.0

    a_score = _pct_score(sh, 2.0)
    h_score = _pct_score(hs300, 2.0)
    u_score = _pct_score(ndx, 2.0)
    v_score = _vix_score(vix)
    temp = round(0.30 * a_score + 0.15 * h_score + 0.25 * u_score + 0.30 * v_score, 1)

    if temp < 35:
        label = "谨慎"
    elif temp < 55:
        label = "中性偏谨慎"
    elif temp < 70:
        label = "均衡"
    elif temp < 85:
        label = "偏积极"
    else:
        label = "积极"

    inputs = []
    for nm, v in (("上证指数", sh), ("沪深300", hs300), ("纳斯达克100", ndx)):
        inputs.append(f"{nm} {v:+.2f}%" if v is not None else f"{nm} 数据缺失")
    inputs.append(f"VIX {vix:.2f}" if vix is not None else "VIX 数据缺失")

    return {
        "temperature": temp,
        "label": label,
        "inputs": inputs,
        "components": {
            "上证指数分": round(a_score, 1),
            "沪深300分": round(h_score, 1),
            "纳指100分": round(u_score, 1),
            "VIX分": round(v_score, 1),
        },
    }


def equity_budget_pct(temperature: float) -> float:
    """权益总仓位上限（占总资金）。温度越低仓位越轻。"""
    risk = C.RISK
    cap = risk["total_max_pct"]
    if temperature < 35:
        target = 40.0
    elif temperature < 55:
        target = 55.0
    elif temperature < 70:
        target = 65.0
    elif temperature < 85:
        target = 75.0
    else:
        target = 80.0
    return round(min(target, float(cap)), 1)


# ------------------------------------------------------------------ 组合
def _stop_for(market: str) -> float:
    risk = C.RISK
    if market == "HK":
        return float(risk["stop_loss_us"])
    return float(risk["stop_loss_a"])


def _allocate_units(
    units_total: int,
    scores: list[float],
    cap_u: int,
    floor_u: int,
) -> list[int]:
    """在「单项 ≤ cap_u、单项 ≥ floor_u」的约束下，按分数比例分配整数权重。

    单位是 0.01%（整数「分位」），所以各项之和可以精确等于预算 ——
    逐项 round(2) 会留下 0.01~0.05 的残差，残差会让「现金 ≥ 下限」
    在最后一位上不成立（实测 19.99% 而不是 20%）。

    做法：先按比例取连续解，再用二分求一个水位 λ，使得
        Σ clip(λ·score_i, floor_u, cap_u) == units_total
    最后按「最大余数法」把整数部分补齐到精确等于 units_total。

    若 cap_u 卡死导致预算用不完（Σcap_u < units_total），返回顶格结果，
    缺口由现金承担 —— 绝不能为了凑满预算而突破单只上限。
    """
    n = len(scores)
    if n == 0:
        return []
    if units_total <= 0:
        return [0] * n
    if cap_u < floor_u:
        floor_u = cap_u
    # 下限不能高到让 n·floor 超过预算，否则没有任何可行解
    floor_u = max(0, min(floor_u, units_total // n))
    s = [max(float(x), 1e-9) for x in scores]

    def total_at(lam: float) -> float:
        return float(sum(min(cap_u, max(floor_u, lam * sc)) for sc in s))

    hi = 1.0
    reachable = False
    for _ in range(200):
        if total_at(hi) >= units_total:
            reachable = True
            break
        hi *= 2.0
    if not reachable:
        # 单只上限卡死：全部顶格，缺口归现金
        return [cap_u] * n

    lo = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if total_at(mid) >= units_total:
            hi = mid
        else:
            lo = mid

    vals = [min(cap_u, max(floor_u, hi * sc)) for sc in s]
    ints = [int(v // 1) for v in vals]
    rem = units_total - sum(ints)

    if rem > 0:
        order = sorted(range(n), key=lambda i: (-(vals[i] - ints[i]), -s[i]))
        for i in order:
            if rem == 0:
                break
            if ints[i] < cap_u:
                ints[i] += 1
                rem -= 1
        # 极端情况下全部顶格仍有剩余：从最大的项里匀出来
        while rem > 0:
            idx = max(range(n), key=lambda i: ints[i])
            if ints[idx] <= floor_u:
                break
            ints[idx] -= 1
            rem -= 1
    elif rem < 0:
        for i in sorted(range(n), key=lambda i: -ints[i]):
            if rem == 0:
                break
            if ints[i] > floor_u:
                ints[i] -= 1
                rem += 1
    return ints


def _allocate_pct(
    total_pct: float,
    scores: list[float],
    *,
    cap: float,
    floor: float,
) -> list[float]:
    """按分数比例分配权重（百分比），单项 ∈ [floor, cap]，合计精确等于 total_pct。"""
    units = _allocate_units(
        int(round(total_pct * 100)),
        scores,
        int(round(cap * 100)),
        int(round(floor * 100)),
    )
    return [u / 100.0 for u in units]


def build_portfolio(stocks: list[dict], equity_pct: float) -> list[dict]:
    """按主题配额选股并分配权重。

    权重分配用整数「分位」（0.01%）做，保证：
      - 同一主题内权重之和精确等于该主题预算（除非被单只上限截断）
      - 单只不超过 single_max_pct，也不低于 MIN_POSITION_PCT
      - 主题缺合格标的时，它的配额空缺 → 归现金，不会挪给别的持仓
    """
    risk = C.RISK
    single_max = float(risk["single_max_pct"])
    by_theme: dict[str, list[dict]] = {}
    for s in stocks:
        scored = score_stock(s)
        by_theme.setdefault(s.get("theme") or "其他", []).append({**s, "_score": scored})

    positions: list[dict] = []
    for theme, quota in THEME_QUOTA.items():
        rows = by_theme.get(theme) or []
        # 只保留核验通过的（verified），conflict 一律剔除
        usable = [r for r in rows if (r.get("cross") or {}).get("status") == "verified"]
        usable.sort(key=lambda r: -r["_score"]["score"])
        picked = usable[:MAX_PER_THEME]
        if not picked:
            continue
        theme_budget = equity_pct * quota
        # 整数分位分配：合计精确等于 theme_budget，单只 ∈ [MIN_POSITION_PCT, single_max]
        weights = _allocate_pct(
            theme_budget,
            [r["_score"]["score"] for r in picked],
            cap=single_max,
            floor=MIN_POSITION_PCT,
        )
        for r, w in zip(picked, weights):
            price = r.get("price")
            if price is None or price <= 0:
                continue
            stop_pct = _stop_for(r.get("market") or "CN")
            tp = risk["take_profit"]
            positions.append(
                {
                    "kind": "stock",
                    "code": r.get("code"),
                    "name": r.get("name"),
                    "theme": theme,
                    "market": r.get("market"),
                    "sector": r.get("sector"),
                    "price": price,
                    "pct": r.get("pct"),
                    "weight_pct": w,
                    "stop_pct": stop_pct,
                    "stop_price": round(price * (1 + stop_pct / 100.0), 3),
                    "targets": [
                        {"gain_pct": g, "price": round(price * (1 + g / 100.0), 3),
                         "sell_ratio": round(rr, 3)}
                        for g, rr in tp
                    ],
                    "score": r["_score"]["score"],
                    "rationale": "；".join(r["_score"]["notes"]),
                    "verify": (r.get("cross") or {}).get("status"),
                    "n_sources": (r.get("cross") or {}).get("n_sources"),
                }
            )
    positions.sort(key=lambda p: (-p["weight_pct"], p["name"]))
    return positions


def _rescale_to_budget(positions: list[dict], budget_pct: float) -> None:
    """已废弃：曾经的实现把「主题缺标的导致仓位填不满」的缺口也归并到
    最大持仓上，导致单只权重被推到 42%（远超 single_max_pct）。

    现在改为：每个主题在自己的配额内做整数分位分配（见 _allocate_pct），
    主题配额之和精确等于 equity_pct；某主题缺合格标的时，它的配额空缺
    自然归现金，不会挪给别的持仓，也不需要事后归并。
    """
    return None


def build_product_portfolio(products: list[dict], budget_pct: float) -> list[dict]:
    """理财产品按类别配额分配。"""
    out: list[dict] = []
    by_cat: dict[str, list[dict]] = {}
    for p in products:
        cat = PRODUCT_CATEGORY_ALIAS.get(p.get("category") or "")
        if not cat:
            continue
        by_cat.setdefault(cat, []).append(p)

    for cat, quota in PRODUCT_QUOTA.items():
        rows = by_cat.get(cat) or []
        if not rows:
            continue
        cat_budget = budget_pct * quota
        # 每类按近1年收益排序（缺失的排后面），取前两只
        rows = sorted(
            rows,
            key=lambda r: (r.get("ret_1y") is None, -(r.get("ret_1y") or 0)),
        )
        picked = rows[:2]
        if not picked:
            continue
        # 整数分位分配：同类两只的权重之和精确等于该类预算
        shares = _allocate_pct(
            cat_budget,
            [1.0] * len(picked),
            cap=cat_budget,
            floor=0.0,
        )
        for p, w in zip(picked, shares):
            row = {
                "kind": "product",
                "code": p.get("code"),
                "name": p.get("name"),
                "category": p.get("category") or cat,
                "quota_bucket": cat,
                "weight_pct": w,
                "nav": p.get("nav"),
                "ret_1y": p.get("ret_1y"),
                "ret_6m": p.get("ret_6m"),
                "seven_day_yield": p.get("seven_day_yield"),
                "date": p.get("date"),
                "verify": (p.get("cross") or {}).get("status"),
                "n_sources": (p.get("cross") or {}).get("n_sources"),
                "rationale": p.get("rationale") or "",
                "stop_pct": None,
                "stop_price": None,
                "targets": [],
            }
            out.append(row)
    return out


def build_watchlist(stocks: list[dict], portfolio: list[dict]) -> list[dict]:
    """全池关注清单：每一只都给出分数、核验状态、是否入选。"""
    in_portfolio = {p["code"] for p in portfolio if p.get("kind") == "stock"}
    rows: list[dict] = []
    for s in stocks:
        scored = score_stock(s)
        rows.append(
            {
                "code": s.get("code"),
                "name": s.get("name"),
                "theme": s.get("theme"),
                "sector": s.get("sector"),
                "market": s.get("market"),
                "price": s.get("price"),
                "pct": s.get("pct"),
                "pe": s.get("pe"),
                "pb": s.get("pb"),
                "score": scored["score"],
                "verify": (s.get("cross") or {}).get("status"),
                "n_sources": (s.get("cross") or {}).get("n_sources"),
                "in_portfolio": s.get("code") in in_portfolio,
                "notes": scored["notes"],
            }
        )
    rows.sort(key=lambda r: (-r["score"], r.get("name") or ""))
    return rows


def build_advice(
    quotes: dict,
    products: dict | None,
    *,
    now: str | None = None,
) -> dict:
    """产出完整的研究性模拟组合建议。"""
    now = now or time.strftime("%Y-%m-%d %H:%M:%S")
    indexes = quotes.get("indexes") or []
    stocks = quotes.get("stocks") or []
    prods = (products or {}).get("products") or []

    state = market_state(indexes)
    equity_pct = equity_budget_pct(state["temperature"])
    cash_min = float(C.RISK["cash_min_pct"])
    product_budget = round(max(0.0, 100.0 - equity_pct - cash_min), 1)

    equity_positions = build_portfolio(stocks, equity_pct)
    product_positions = build_product_portfolio(prods, product_budget)

    # 权重落地后重算实际仓位，缺口归现金
    actual_equity = round(sum(p["weight_pct"] for p in equity_positions), 2)
    actual_product = round(sum(p["weight_pct"] for p in product_positions), 2)
    actual_cash = round(100.0 - actual_equity - actual_product, 2)

    for p in equity_positions:
        pct = p.get("pct")
        if p["score"] >= 60 and (pct or 0) > 0:
            p["call"] = "偏多"
        elif p["score"] <= 35 and (pct or 0) < 0:
            p["call"] = "偏空"
        else:
            p["call"] = "中性"

    watchlist = build_watchlist(stocks, equity_positions)

    allocation = {
        "equity_pct": actual_equity,
        "product_pct": actual_product,
        "cash_pct": actual_cash,
        "equity_target_pct": equity_pct,
        "product_target_pct": product_budget,
        "cash_min_pct": cash_min,
        "by_theme": {},
        "by_category": {},
    }
    for p in equity_positions:
        allocation["by_theme"][p["theme"]] = round(
            allocation["by_theme"].get(p["theme"], 0.0) + p["weight_pct"], 2
        )
    for p in product_positions:
        allocation["by_category"][p["category"]] = round(
            allocation["by_category"].get(p["category"], 0.0) + p["weight_pct"], 2
        )

    return {
        "as_of": now,
        "disclaimer": DISCLAIMER,
        "market_state": state,
        "allocation": allocation,
        "portfolio": equity_positions + product_positions,
        "watchlist": watchlist,
        "rules": dict(C.RISK),
        "n_equity": len(equity_positions),
        "n_product": len(product_positions),
        "n_watchlist": len(watchlist),
    }
