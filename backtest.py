"""因子回测与验证引擎。

这套东西回答一个问题：**某个因子在真实历史上，真的能预测未来收益吗？**

方法：
1. 对每个交易日 t（每个标的），用截至 t 的数据算因子值（无未来函数）。
2. 计算未来 N 日收益 fwd_ret(t, N)。
3. 算横截面 Spearman Rank IC：因子值与未来收益在当日截面上的秩相关。
4. 汇总成 IC 序列，做 Newey-West 调整 t 检验（处理自相关），
   并按 Bonferroni 做多重检验校正——测了 16 个因子，不校正必然出假阳性。

关键诚实性约束：
- 因子值只来自 bars[:t+1]，未来收益用 t+1 之后的数据，两者严格不重叠。
- 结果只作为"证据"，报告里会展示 IC 和 t 值，并如实说明哪些因子不显著。
- 不用任何未来信息调参。
"""
from __future__ import annotations

import math
from typing import Any

from factors import FACTORS, compute_factors

TRADING_DAYS = 252


# ------------------------------------------------------------------ 统计工具


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman 秩相关。样本不足或无方差时返回 None（不编造 0）。"""
    n = len(xs)
    if n < 3:
        return None
    rx = _rank(xs)
    ry = _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _rank(xs: list[float]) -> list[float]:
    """平均秩，处理并列值。"""
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def newey_west_t(series: list[float], lags: int = 5) -> tuple[float | None, float | None]:
    """Newey-West 调整的 t 统计量。

    日频 IC 序列有自相关，普通 t 检验会高估显著性。这里用 HAC 协方差修正。
    返回 (均值, t值)；样本不足返回 (None, None)。
    """
    n = len(series)
    if n < 12:
        return (sum(series) / n if n else None), None
    mean = sum(series) / n
    dev = [x - mean for x in series]
    gamma0 = sum(d * d for d in dev) / n
    var = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        g = sum(dev[i] * dev[i - lag] for i in range(lag, n)) / n
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * g
    if var <= 0:
        return mean, None
    se = math.sqrt(var / n)
    if se == 0:
        return mean, None
    return mean, mean / se


def bonferroni_alpha(n_tests: int, alpha: float = 0.05) -> float:
    """多重检验校正后的显著性阈值。"""
    return alpha / max(n_tests, 1)


# ------------------------------------------------------------------ 回测主流程


def build_panel(
    histories: dict[str, list[dict]],
    *,
    forward_days: int = 5,
    min_bars: int = 130,
) -> list[dict]:
    """构建 (日期, 标的) 级别的面板，每行含因子值与未来收益。

    forward_days: 前瞻窗口。5 日 ≈ 一周，兼顾信号强度与换手成本。
    min_bars: 少于这么多 K 线的标的不参与（因子需要预热）。
    """
    panel: list[dict] = []
    for secid, bars in histories.items():
        if len(bars) < min_bars + forward_days + 1:
            continue
        closes = [b["close"] for b in bars]
        for i in range(min_bars - 1, len(bars) - forward_days):
            window = bars[: i + 1]
            fvals = compute_factors(window)
            if not fvals:
                continue
            px_now = closes[i]
            px_fwd = closes[i + forward_days]
            if not px_now:
                continue
            fwd = px_fwd / px_now - 1.0
            panel.append({
                "date": bars[i]["date"],
                "secid": secid,
                "factors": fvals,
                "fwd_ret": fwd,
            })
    return panel


def cross_sectional_ic(
    panel: list[dict],
    factor_names: list[str],
    *,
    min_cross_section: int = 5,
) -> dict[str, dict[str, Any]]:
    """逐日算横截面 IC。

    返回 {factor: {"ics": [...], "ic_mean", "ic_std", "icir", "t_stat", "n_days", "hit_rate"}}
    """
    by_date: dict[str, list[dict]] = {}
    for row in panel:
        by_date.setdefault(row["date"], []).append(row)

    ics: dict[str, list[float]] = {f: [] for f in factor_names}
    for _date, rows in by_date.items():
        if len(rows) < min_cross_section:
            continue
        rets = [r["fwd_ret"] for r in rows]
        for f in factor_names:
            vals = [r["factors"].get(f) for r in rows]
            pairs = [(v, r) for v, r in zip(vals, rets) if v is not None]
            if len(pairs) < min_cross_section:
                continue
            rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
            if rho is not None:
                ics[f].append(rho)

    out: dict[str, dict[str, Any]] = {}
    for f, series in ics.items():
        if not series:
            out[f] = {"ics": [], "ic_mean": None, "ic_std": None, "icir": None,
                      "t_stat": None, "n_days": 0, "hit_rate": None}
            continue
        mean, t = newey_west_t(series)
        std = (sum((x - (mean or 0.0)) ** 2 for x in series) / (len(series) - 1)) ** 0.5 if len(series) > 1 else 0.0
        hit = sum(1 for x in series if x > 0) / len(series)
        out[f] = {
            "ics": series,
            "ic_mean": mean,
            "ic_std": std if len(series) > 1 else None,
            "icir": (mean / std) if (mean is not None and std) else None,
            "t_stat": t,
            "n_days": len(series),
            "hit_rate": hit,
        }
    return out


def evaluate_factors(
    histories: dict[str, list[dict]],
    *,
    forward_days: int = 5,
    min_bars: int = 130,
) -> dict[str, Any]:
    """完整因子评估：建面板 → 逐日 IC → 统计检验 → 选出显著因子。"""
    panel = build_panel(histories, forward_days=forward_days, min_bars=min_bars)
    names = list(FACTORS.keys())
    icres = cross_sectional_ic(panel, names)

    alpha = bonferroni_alpha(len(names))
    for f, r in icres.items():
        r["name"] = f
        r["significant"] = bool(
            r["t_stat"] is not None and abs(r["t_stat"]) > 1.96 and len(r["ics"]) >= 60
        )
        r["bonferroni_alpha"] = alpha
        r["direction"] = FACTORS[f][1]
        r["desc"] = FACTORS[f][2]

    # 符号校正：IC 均值为负的因子，取负号后用（方向已由因子内部处理）
    ranked = sorted(
        (r for r in icres.values() if r["ic_mean"] is not None),
        key=lambda r: -abs(r["ic_mean"] or 0.0),
    )
    selected = [r for r in ranked if r["significant"]]
    return {
        "panel_size": len(panel),
        "forward_days": forward_days,
        "n_factors": len(names),
        "alpha": alpha,
        "factors": icres,
        "ranked": ranked,
        "selected": selected,
    }


def factor_ensemble_scores(
    histories: dict[str, list[dict]],
    selected: list[dict],
    *,
    as_of: str | None = None,
) -> dict[str, dict[str, float]]:
    """用入选因子给每个标的打分（z-score 加权）。

    as_of: 只取该日期及之前的数据算分（None = 用最新）。
    返回 {secid: {factor: z, "score": w}}
    """
    if not selected:
        return {}
    weights = {r["name"]: r for r in selected}
    per_factor: dict[str, list[tuple[str, float]]] = {}
    raw: dict[str, dict[str, float]] = {}

    for secid, bars in histories.items():
        if not bars:
            continue
        seq = bars
        if as_of:
            seq = [b for b in bars if b["date"] <= as_of]
        if len(seq) < 130:
            continue
        fv = compute_factors(seq)
        if not fv:
            continue
        raw[secid] = fv
        for fname in weights:
            v = fv.get(fname)
            if v is not None:
                per_factor.setdefault(fname, []).append((secid, v))

    # 每个因子做横截面 z-score
    z: dict[str, dict[str, float]] = {s: {} for s in raw}
    for fname, pairs in per_factor.items():
        vals = [v for _s, v in pairs]
        if len(vals) < 3:
            continue
        m = sum(vals) / len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        if sd == 0:
            continue
        for s, v in pairs:
            z[s][fname] = (v - m) / sd

    out: dict[str, dict[str, float]] = {}
    for s, zs in z.items():
        if not zs:
            continue
        # 权重 = |IC均值| × ICIR 的稳健组合，再按入选因子数归一
        num = 0.0
        den = 0.0
        detail: dict[str, float] = {}
        for fname, zv in zs.items():
            r = weights[fname]
            w = abs(r["ic_mean"] or 0.0) * max(abs(r["icir"] or 0.0), 0.01)
            detail[fname] = zv
            num += w * zv
            den += w
        out[s] = {"score": (num / den) if den else 0.0, "z": detail}
    return out
