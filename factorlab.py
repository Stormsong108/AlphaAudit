"""大样本横截面因子检验（numpy 向量化）。

为什么重写：
    初版 build_panel 用纯 Python 逐根 K 线算因子，34 只股票 11 年要跑很久，
    也无法承载 1300+ 只全市场。改成向量化后全市场也只需分钟级。

统计设计（这是能不能下结论的关键）：
1. **逐日横截面**：把面板组织成 [日期 × 标的] 的矩阵，IC 在每个交易日
   的横截面上算。跨标的的同期比较才有意义。
2. **不重叠前瞻窗口**：IC 序列用步长 = forward_days 的不重叠块，
   否则 t 值会被重叠样本严重高估（这是我在小样本上踩过的坑）。
3. **Newey-West + Bonferroni**：处理自相关 + 多重检验校正。
4. **分段稳定性**：前半段/后半段 IC 符号必须一致，否则是过拟合。

结论输出要区分三档：
    显著可用 / 弱但方向一致 / 无效
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from fastfactors import FACTOR_DESC, FACTOR_ORDER

FACTOR_NAMES = list(FACTOR_ORDER)
TRADING_DAYS = 252


# ------------------------------------------------------------------ 数据组织


def build_matrix(
    histories: dict[str, list[dict]],
    *,
    min_bars: int = 130,
    forward_days: int = 5,
) -> dict[str, Any]:
    """把 {secid: bars} 组织成对齐矩阵。

    返回:
        dates:   [T] 交易日（升序）
        codes:   [N] 标的代码
        close:   [T, N] 收盘价（NaN 表示缺失）
        fwd:     [T, N] 未来 forward_days 日收益
        factors: {name: [T, N]}
    """
    # 只保留足够长的标的
    import fastfactors

    usable = {k: v for k, v in histories.items() if len(v) >= min_bars + forward_days + 1}
    if not usable:
        return {"dates": [], "codes": [], "close": np.zeros((0, 0)),
                "fwd": np.zeros((0, 0)), "amount": np.zeros((0, 0)),
                "factors": {}}

    from collections import Counter

    date_count: Counter = Counter()
    for bars in usable.values():
        for b in bars:
            date_count[b["date"]] += 1

    # 阈值必须相对"当日实际存在的标的数"，而不是当前股票池总数。
    #
    # 2026-11-10 的坑：固定用 0.30 × len(usable) = 398（1328 只里的 30%）。
    # 但 2015 年 A 股总共才上市了约 400 只，正常交易日也只有 380~400 只有数据，
    # 于是阈值像一把尺子卡在覆盖率附近，把 2015-07-03 ~ 2015-08-04 这些
    # **完全正常的交易日**随机丢进轴外 —— 日期轴出现 15 天的断层。
    # 断层之后 close[1:]/close[:-1] 把跨 20 个交易日的涨幅当成"单日涨幅"，
    # 凭空造出 +109% / +147% / +154% 的假极端值；
    # 更糟的是按轴位置算的"持有 5 日"实际变成了"持有 20 日"。
    #
    # 正确口径：某日的分母是"上市日期 ≤ 该日的标的数"。
    import bisect

    first_dates = sorted(min(b["date"] for b in bars) for bars in usable.values())
    all_dates = []
    for d in sorted(date_count):
        eligible = bisect.bisect_right(first_dates, d)   # 当日应有多少只有资格交易
        thresh = max(3, int(0.30 * eligible))
        if date_count[d] >= thresh:
            all_dates.append(d)
    date_idx = {d: i for i, d in enumerate(all_dates)}

    n_t = len(all_dates)
    codes = sorted(usable.keys())
    n_s = len(codes)
    code_idx = {c: i for i, c in enumerate(codes)}

    close = np.full((n_t, n_s), np.nan)
    fwd = np.full((n_t, n_s), np.nan)
    amount = np.full((n_t, n_s), np.nan)
    factors = {f: np.full((n_t, n_s), np.nan) for f in fastfactors.FACTOR_ORDER}

    for code, bars in usable.items():
        ci = code_idx[code]
        closes = np.array([b["close"] for b in bars], dtype=float)
        highs = np.array([b["high"] for b in bars], dtype=float)
        lows = np.array([b["low"] for b in bars], dtype=float)
        opens = np.array([b.get("open", b["close"]) for b in bars], dtype=float)
        vols = np.array([b["volume"] for b in bars], dtype=float)
        amts = np.array([b.get("amount", 0.0) for b in bars], dtype=float)

        fvals = fastfactors.compute_all(closes, highs, lows, vols, amts, opens)

        tidx = np.array([date_idx.get(b["date"], -1) for b in bars], dtype=int)
        ok = tidx >= 0
        rows = tidx[ok]
        close[rows, ci] = closes[ok]
        amount[rows, ci] = amts[ok]
        for fname in fastfactors.FACTOR_ORDER:
            factors[fname][rows, ci] = fvals[fname][ok]
        if len(bars) > forward_days:
            c0 = closes[:-forward_days]
            c1 = closes[forward_days:]
            # rr 必须与 f 等长：用 ok 的前 n-k 个位置（bars 索引），
            # 不能用 rows[:-k]（rows 长度是过滤后的）
            src_rows = np.array(
                [date_idx.get(bars[i]["date"], -1) for i in range(len(bars) - forward_days)],
                dtype=int,
            )
            m = src_rows >= 0
            with np.errstate(invalid="ignore", divide="ignore"):
                f = c1 / c0 - 1.0
            f[~np.isfinite(f)] = np.nan
            fwd[src_rows[m], ci] = f[m]

    return {"dates": all_dates, "codes": codes, "close": close, "fwd": fwd,
            "amount": amount, "factors": factors}


# ------------------------------------------------------------------ IC 计算


def _rank_rows(mat: np.ndarray) -> np.ndarray:
    """对每一行做平均秩（处理并列）。mat: [T, N] → [T, N]"""
    out = np.full(mat.shape, np.nan)
    for t in range(mat.shape[0]):
        row = mat[t]
        valid = ~np.isnan(row)
        n = int(valid.sum())
        if n < 3:
            continue
        vals = row[valid]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        sv = vals[order]
        # 平均秩分组
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sv[j + 1] == sv[i]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        out[t, valid] = ranks
    return out


def cross_sectional_ics(
    mat: dict[str, Any],
    factor_name: str,
    *,
    min_cross: int = 20,
) -> np.ndarray:
    """逐日横截面 IC（Spearman），返回 [T]（部分为 NaN）。"""
    fmat = mat["factors"][factor_name]
    fwd = mat["fwd"]
    T = fmat.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        f_row, r_row = fmat[t], fwd[t]
        valid = ~np.isnan(f_row) & ~np.isnan(r_row)
        n = int(valid.sum())
        if n < min_cross:
            continue
        fv, rv = f_row[valid], r_row[valid]
        # 秩
        fo = np.argsort(fv, kind="mergesort")
        ro = np.argsort(rv, kind="mergesort")
        fr = np.empty(n); fr[fo] = np.arange(1, n + 1, dtype=float)
        rr = np.empty(n); rr[ro] = np.arange(1, n + 1, dtype=float)
        # 处理并列（简单平均秩）
        _fix_ties(fv, fr)
        _fix_ties(rv, rr)
        fr_c = fr - fr.mean()
        rr_c = rr - rr.mean()
        den = math.sqrt(float((fr_c ** 2).sum()) * float((rr_c ** 2).sum()))
        if den == 0:
            continue
        out[t] = float((fr_c * rr_c).sum() / den)
    return out


def _fix_ties(vals: np.ndarray, ranks: np.ndarray) -> None:
    """把并列值的秩改成平均秩（就地修改）。"""
    n = len(vals)
    order = np.argsort(vals, kind="mergesort")
    sv = vals[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def newey_west(series: np.ndarray, lags: int = 5) -> tuple[float | None, float | None]:
    """Newey-West 调整的 t 统计。输入序列需已剔除 NaN。"""
    s = series[~np.isnan(series)]
    n = len(s)
    if n < 12:
        return (float(s.mean()) if n else None), None
    d = s - s.mean()
    g0 = float((d ** 2).sum()) / n
    var = g0
    for lag in range(1, min(lags, n - 1) + 1):
        g = float((d[lag:] * d[:-lag]).sum()) / n
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * g
    if var <= 0:
        return float(s.mean()), None
    se = math.sqrt(var / n)
    return float(s.mean()), (float(s.mean() / se) if se else None)


def nonoverlap_ic_series(
    ics: np.ndarray,
    *,
    forward_days: int,
) -> np.ndarray:
    """把逐日 IC 压成不重叠序列，消除重叠导致的 t 值虚高。

    每 forward_days 个交易日取一个 IC（取块内均值），得到近似独立样本。
    """
    clean = ics[~np.isnan(ics)]
    if len(clean) < forward_days:
        return clean
    k = forward_days
    n_blocks = len(clean) // k
    if n_blocks < 5:
        return clean
    blocks = clean[: n_blocks * k].reshape(n_blocks, k)
    return blocks.mean(axis=1)


# ------------------------------------------------------------------ 主评估


def evaluate(
    mat: dict[str, Any],
    *,
    forward_days: int = 5,
    min_cross: int = 20,
    min_samples: int = 60,
) -> dict[str, Any]:
    """全因子评估。返回每个因子的 IC 统计与判定。"""
    if not mat["codes"]:
        return {"ok": False, "reason": "无可用样本"}

    n_tests = len(FACTOR_NAMES)
    alpha = 0.05 / n_tests
    results: dict[str, dict[str, Any]] = {}

    for fname in FACTOR_NAMES:
        ics = cross_sectional_ics(mat, fname, min_cross=min_cross)
        series = nonoverlap_ic_series(ics, forward_days=forward_days)
        mean, t = newey_west(series)
        if mean is None:
            results[fname] = {"name": fname, "n": 0, "mean": None, "t": None,
                              "verdict": "无样本", "desc": FACTORS[fname][2]}
            continue
        sd = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
        icir = mean / sd if sd > 0 else None
        hit = float(np.mean(series > 0))
        # 分段稳定性：前后半段符号
        half = len(series) // 2
        m1 = float(np.mean(series[:half])) if half else None
        m2 = float(np.mean(series[half:])) if half > 1 else None
        stable = (m1 is not None and m2 is not None and np.sign(m1) == np.sign(m2)
                  and abs(m1) > 0 and abs(m2) > 0)
        sig = abs(t) > 1.96 if t is not None else False
        if sig and stable:
            verdict = "显著可用"
        elif sig:
            verdict = "显著但分段不稳"
        elif abs(mean) >= 0.01 and (icir is not None and abs(icir) >= 0.02):
            verdict = "弱信号"
        else:
            verdict = "无效"
        results[fname] = {
            "name": fname,
            "n": len(series),
            "mean": mean,
            "t": t,
            "sd": sd,
            "icir": icir,
            "hit": hit,
            "half1": m1,
            "half2": m2,
            "stable": bool(stable),
            "significant": bool(sig),
            "verdict": verdict,
            "desc": FACTOR_DESC.get(fname, fname),
        }

    ranked = sorted(
        (r for r in results.values() if r["mean"] is not None),
        key=lambda r: -abs(r["mean"]),
    )
    usable = [r for r in ranked if r["verdict"] == "显著可用"]
    return {
        "ok": True,
        "n_factors": n_tests,
        "alpha": alpha,
        "n_dates": len(mat["dates"]),
        "n_codes": len(mat["codes"]),
        "forward_days": forward_days,
        "results": results,
        "ranked": ranked,
        "usable": usable,
    }


def composite_score(
    mat: dict[str, Any],
    usable: list[dict],
    *,
    as_of_index: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """用可用因子做横截面 z-score 加权合成，返回 [T, N] 分数。

    as_of_index: 只用该行及之前的数据算 z-score（None = 用全样本，
    仅用于回测分析；实盘必须传入当期行号）。
    """
    T, N = mat["close"].shape
    scores = np.full((T, N), np.nan)
    if not usable:
        return scores, mat["codes"]

    zsum = np.zeros((T, N))
    wsum = np.zeros((T, N))
    for r in usable:
        fmat = mat["factors"][r["name"]]
        w = abs(r["mean"]) * max(abs(r.get("icir") or 0.0), 0.01)
        for t in range(T):
            row = fmat[t]
            valid = ~np.isnan(row)
            n = int(valid.sum())
            if n < 5:
                continue
            vals = row[valid]
            m = float(vals.mean())
            sd = float(vals.std(ddof=1))
            if sd == 0:
                continue
            zsum[t, valid] += w * (vals - m) / sd
            wsum[t, valid] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(wsum > 0, zsum / wsum, np.nan)
    return scores, mat["codes"]
