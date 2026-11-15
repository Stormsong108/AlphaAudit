# -*- coding: utf-8 -*-
"""ETF 面板数据质量 + 复权质量校验。

沿用个股通道纪律（dataquality.py / audit_panel）：
  坏数据必须让 audit 失败，不允许「缺了照样跑」。

检查项
------
A. 原始价格
   1. 价格必须为正
   2. OHLC 自洽：low ≤ min(open,close) ≤ max(open,close) ≤ high
   3. 内部缺口受控（A股长假 ≤ 12 自然日）
   4. 成交额存在且为正
B. 净值对齐
   5. 净值与价格的日收益序列在**最佳滞后**下的相关系数
      （QDII 的净值按境外收盘计算，必然滞后 1 日；滞后不是数据错误，
        但必须显式测出来，不能默认对齐）
C. 复权质量（关键）
   6. 除息日：复权**前**价格收益应显著低于净值收益，差值 ≈ 当日分红；
      复权**后**两者应基本一致（残差 ≈ 0）。
      若复权后仍偏离 → 复权失败，该 ETF 必须剔除，不能带病进策略。

产出：data/research/etf_dataquality.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etf_adj as A  # noqa: E402
import etf_lab as E  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, "etf_dataquality.json")
MAX_GAP_DAYS = 12
DIV_GAP_TOL_BP = 60.0      # 复权后除息日与净值的残差容忍（60bp）
MAX_LAG = 2

# 致命问题前缀：出现即代表**整块 ETF 面板不可信**，研究必须失败关闭，
# 不能「剔除掉坏的就当没事」—— 那等于悄悄改变研究宇宙，
# 而被剔掉的往往正是主流宽基/行业 ETF（实测纳指ETF、医药ETF、新能源车ETF…），
# 剔除会让 ETF 池产生系统性选择偏差。
FATAL_PREFIXES = (
    "non_positive_price",
    "ohlc_inconsistent",
    "internal_gap_gt_",
    "div_gap_after_adj_too_large",
    "extreme_adj_jump_gt_25pct",
)


def best_lag_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """返回 (最佳相关系数, 对应滞后)。滞后定义为 x 相对 y 偏移。"""
    best, bl = float("nan"), 0
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag < 0:
            a, b = x[-lag:], y[:len(y) + lag]
        elif lag > 0:
            a, b = x[:len(x) - lag], y[lag:]
        else:
            a, b = x, y
        n = min(len(a), len(b))
        if n < 60:
            continue
        a, b = a[-n:], b[-n:]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(c):
            continue
        if not np.isfinite(best) or c > best:
            best, bl = c, lag
    return best, bl


def audit_one(sym: str, name: str) -> dict[str, Any]:
    issues: list[str] = []
    raw = json.load(open(os.path.join(E.ETF_DIR, f"{sym}.json"), encoding="utf-8"))
    adjp = os.path.join(E.ADJ_DIR, f"{sym}.json")
    adj = json.load(open(adjp, encoding="utf-8")) if os.path.exists(adjp) else raw
    nav = A.load_nav(sym[2:])
    n = len(raw)

    bad_px = sum(1 for r in raw if not (r["open"] > 0 and r["close"] > 0
                                        and r["high"] > 0 and r["low"] > 0))
    if bad_px:
        issues.append(f"non_positive_price:{bad_px}")
    bad_ohlc = sum(1 for r in raw if not (r["low"] <= min(r["open"], r["close"])
                                         and max(r["open"], r["close"]) <= r["high"]))
    if bad_ohlc:
        issues.append(f"ohlc_inconsistent:{bad_ohlc}")
    ds = [dt.date.fromisoformat(r["date"]) for r in raw]
    gaps = [(b - a).days for a, b in zip(ds, ds[1:])]
    max_gap = max(gaps) if gaps else 0
    n_gap = sum(1 for g in gaps if g > MAX_GAP_DAYS)
    if n_gap:
        issues.append(f"internal_gap_gt_{MAX_GAP_DAYS}d:{n_gap}")
    n_amt = sum(1 for r in raw if not (r.get("amount") or 0) > 0)
    warnings: list[str] = []
    if n_amt:
        # 少量缺成交额的交易日：流动性过滤会把这些 ETF-日直接排除（保守、无前视），
        # 不构成数据错误；超过 10 天才算硬失败。
        msg = f"amount_missing:{n_amt}"
        (issues if n_amt > 10 else warnings).append(msg)

    # ── 净值对齐 + 复权质量 ──────────────────────────────────
    #
    # ⚠️ 三条收益序列必须在**同一批相邻交易日**上比较。
    # 原始价格行里可能有净值缺失的日子（实测 sh513500 有 30 天：净值更新
    # 滞后于二级市场）。如果「价格收益」按全部价格行算、「净值收益」只按
    # 有净值的行算，再按尾部对齐比较，两条序列会**错位最多 30 格** ——
    # 相关系数直接被算废（sh513500 因此只有 0.23，同一批交易日重算是 0.55）。
    # 复权质量门禁建在这个数上，错了等于门禁失效。
    rows_nav = [r for r in raw if r["date"] in nav]
    nav_dates = [r["date"] for r in rows_nav]
    nav_ret = np.array([nav[nav_dates[i]][1] / nav[nav_dates[i - 1]][1] - 1.0
                        for i in range(1, len(nav_dates))])
    raw_ret = np.array([rows_nav[i]["close"] / rows_nav[i - 1]["close"] - 1.0
                        for i in range(1, len(rows_nav))])
    adj_close = {r["date"]: r.get("adj_close") for r in adj}
    _ac = [adj_close.get(d) for d in nav_dates]
    if any(v is None or not np.isfinite(v) or v <= 0 for v in _ac):
        issues.append("adj_close_missing_for_nav_dates")
        adj_ret = raw_ret.copy()
    else:
        adj_ret = np.array([_ac[i] / _ac[i - 1] - 1.0 for i in range(1, len(_ac))])
    c_raw, lag_raw = best_lag_corr(raw_ret, nav_ret)
    c_adj, lag_adj = best_lag_corr(adj_ret, nav_ret)

    # ── D. 复权后价格跳变（份额折算/合并必须被消掉）────────────
    # ETF 二级市场涨跌幅限制 ±10%，真实单日行情**不可能**跳 25% 以上。
    # 出现这种跳变 = 复权因子漏掉了某次份额折算 / 份额合并，
    # 该 ETF 价格序列里存在假涨假跌 —— 带病进策略会造出假 edge
    # （实测：纳指ETF 2022-01-14 单日 -80%，直接把年化拖成 -122%）。
    #
    # 为什么原来漏了：adj_vs_nav_corr 这类**相关性**指标对单点极值不敏感
    # —— 1970 根里只有一处假跌，整体相关性照样 > 0.5，判 PASS。
    # 所以必须有一条**逐点**的硬门禁。
    JUMP_THR = 0.25
    extreme: list[tuple[str, float]] = []
    prev = None
    for r in adj:
        v = r.get("adj_close")
        if (prev is not None and v is not None
                and np.isfinite(prev) and np.isfinite(v)
                and prev > 0 and v > 0):
            rr = float(v / prev - 1.0)
            if abs(rr) > JUMP_THR:
                extreme.append((r["date"], rr))
        prev = v
    if extreme:
        issues.append(f"extreme_adj_jump_gt_25pct:{len(extreme)}")

    # 除息日残差：复权前应 ≈ −d/U，复权后应 ≈ 0
    #
    # ⚠️ 这里必须和 etf_adj.adjust_series 用**完全一致**的口径：
    #   同一批日期轴（本地行情 ∩ 净值）、同一批 F10 折算日做切段。
    # 原实现跑在「全部净值日期」轴上且不传 folds → k 跨折算整段拟合 →
    # 残差全段是垃圾 → 报出几百个假除息日，div_gap 指标完全失去意义
    # （实测 sh512170 报 852 个除息日，实际只有 1~2 次）。
    _have = [r["date"] for r in rows_nav]
    _folds = [s["date"] for s in A.load_f10_actions(sym[2:]).get("splits") or []] or None
    k, regimes, resid = A.estimate_k(_have, nav, folds=_folds)
    dmap = {d: float(resid[i]) for i, d in enumerate(_have)}
    # 下标用 rows_nav 的位置（与上面三条收益序列同一根轴）
    exdiv_idx = [i for i, r in enumerate(rows_nav)
                 if dmap.get(r["date"], 0.0)
                 > A.DIV_REL_THRESH * max(nav[r["date"]][0], 1e-9)]
    gap_before, gap_after = [], []
    for i in exdiv_idx:
        # ⚠️ 差分数组的下标必须**减 1**：ret[j] = rows[j+1]/rows[j] − 1，
        # 所以「除息日 i 当天」的收益在 ret 数组里的下标是 i−1，不是 i。
        # 原实现直接用 i，量到的是**除息次日**的缺口 —— sh512690 因此报出
        # −90bp（其实只是次日的正常波动），真正的除息缺口 −2589bp 被整个漏掉，
        # 「复权后残差 ≈ 0」这条硬门禁等于形同虚设。
        j = i - 1
        if j < 0 or j >= len(nav_ret) or j >= len(raw_ret) or j >= len(adj_ret):
            continue
        gap_before.append((raw_ret[j] - nav_ret[j]) * 1e4)
        gap_after.append((adj_ret[j] - nav_ret[j]) * 1e4)
    mb = float(np.mean(gap_before)) if gap_before else float("nan")
    ma = float(np.mean(gap_after)) if gap_after else float("nan")
    if gap_after and abs(ma) > DIV_GAP_TOL_BP:
        issues.append(f"div_gap_after_adj_too_large:{ma:.0f}bp")
    if np.isfinite(c_adj) and c_adj < 0.5:
        issues.append(f"adj_vs_nav_corr_low:{c_adj:.3f}")

    # ── 致命 vs 软性：决定「剔除这只」还是「整条研究失败关闭」────
    # 致命 = 数据本身是坏的（假跳空/价格非正/OHLC 自相矛盾/复权残差过大）
    #        → 这类品种一出现，说明**整块面板**不可信，必须让研究失败关闭，
    #          不能「剔除掉坏的就当没事」—— 那会悄悄改变研究宇宙。
    # 软性 = 相关性偏低等（多半是 QDII 净值滞后口径问题）→ 剔除该只并披露。
    fatal = [i for i in issues if i.startswith(FATAL_PREFIXES)]
    soft = [i for i in issues if not i.startswith(FATAL_PREFIXES)]

    return {
        "code": sym, "name": name, "ok": not issues, "issues": issues, "warnings": warnings,
        "fatal_issues": fatal, "soft_issues": soft,
        "n": n, "first": raw[0]["date"], "last": raw[-1]["date"],
        "max_gap_days": max_gap, "n_gap": n_gap,
        "non_positive_price": bad_px, "ohlc_inconsistent": bad_ohlc, "amount_missing": n_amt,
        "n_exdiv": len(exdiv_idx), "n_regime": len(regimes),
        "raw_vs_nav_corr": c_raw, "raw_lag": lag_raw,
        "adj_vs_nav_corr": c_adj, "adj_lag": lag_adj,
        "div_gap_before_bp": mb, "div_gap_after_bp": ma,
        "n_div_checked": len(gap_after),
        "n_extreme_adj_jump": len(extreme),
        "extreme_adj_jumps": extreme[:20],
    }


def main() -> None:
    manifest = {}
    if os.path.exists(E.MANIFEST):
        manifest = json.load(open(E.MANIFEST, encoding="utf-8"))
    files = sorted(f for f in os.listdir(E.ETF_DIR) if f.endswith(".json"))
    results = []
    print(f"{'sym':<11}{'n':>6}{'exdiv':>7}{'reg':>5}{'rawCorr':>9}{'lag':>5}"
          f"{'adjCorr':>9}{'gapBef':>9}{'gapAft':>9}  issues")
    for fn in files:
        sym = fn[:-5]
        name = (manifest.get(sym) or {}).get("name", sym)
        r = audit_one(sym, name)
        results.append(r)
        print(f"{sym:<11}{r['n']:>6}{r['n_exdiv']:>7}{r['n_regime']:>5}"
              f"{r['raw_vs_nav_corr']:>9.4f}{r['raw_lag']:>5}"
              f"{r['adj_vs_nav_corr']:>9.4f}{r['div_gap_before_bp']:>9.0f}"
              f"{r['div_gap_after_bp']:>9.0f}  {';'.join(r['issues'])}", flush=True)

    ok = sum(1 for r in results if r["ok"])
    fatal = [r for r in results if r["fatal_issues"]]
    soft_only = [r for r in results if r["soft_issues"] and not r["fatal_issues"]]
    out = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_files": len(results), "n_ok": ok, "n_bad": len(results) - ok,
        "n_fatal": len(fatal),
        "fatal_codes": [r["code"] for r in fatal],
        "soft_excluded_codes": [r["code"] for r in soft_only],
        "audit_ok": ok == len(results) and len(results) > 0,
        "fatal_free": not fatal,
        "checks": ["non_positive_price", "ohlc_consistency", f"internal_gap<={MAX_GAP_DAYS}d",
                   "amount_present", "nav_alignment_best_lag", "div_gap_after_adj<=60bp",
                   "adj_vs_nav_corr>=0.5", "extreme_adj_jump<=25%"],
        "div_gap_tol_bp": DIV_GAP_TOL_BP,
        "results": results,
    }
    os.makedirs(E.OUT_DIR, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w", encoding="utf-8"), indent=1)
    print(f"\n=== {ok}/{len(results)} 通过 → {OUT_PATH} ===")
    print(f"    致命失败 {len(fatal)} 只（研究必须失败关闭）: "
          f"{[r['code'] for r in fatal][:12]}")
    print(f"    仅软性问题（可剔除并披露）{len(soft_only)} 只: "
          f"{[r['code'] for r in soft_only][:12]}")


if __name__ == "__main__":
    main()
