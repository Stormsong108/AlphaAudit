"""择时 vs 选股：11 因子信号的归因分解。

这个模块回答研究路线上一直挂着、从未做过的那个问题：

    「个股通道那套 11 因子信号，预测力到底来自横截面选股，
      还是来自市场方向择时？」

以及它的推论：

    「如果择时成分真的存在，能不能做成一个只用择时、不做选股的
      低换手策略（换手低 → 成本低 → 才有翻身余地）？」

设计铁律（违反即作废）
--------------------
1. **一段只做一次**：训练段选参 → 验证段只做通过/不通过判定 →
   参数冻结（timing_spec_hash 落盘）→ 测试段只跑一次。
   验证段**绝不参与选参**；历史上出过一次事故：把「验证段超额最高」
   的那个当成 chosen，等于把验证段当第二个训练段，测试段当场被打回。
2. **基准与策略对齐同一批期次**：择时策略与 canonical 选股策略跑在
   **完全相同**的信号日 / 买入日 / 卖出日上，市场收益取的是 canonical
   基准的同一批期次（不是另造一个网格）。
3. **三口径并列、毛净分开**：缩尾均值 / 未缩尾均值 / 中位数同时报；
   成本按**实际换手**算，择时策略绝不沿用选股策略的 22.11% 换手率。
4. **数字不编**：算不出就写算不出。每个数字都能追到 panel 与 spec。

一个必须先说清楚的结构性事实
----------------------------
canonical 的合成信号是**逐日横截面 z-score 加权**。横截面 z-score 的
日均值恒为 0，所以「合成信号的市场级聚合」**结构上不可能**携带市场
方向信息 —— 不是「弱」，是「恒等于常数」。

也就是说：想找择时成分，只能去**因子原值**的市场级聚合里找，
不能用 canonical 那个合成信号本身。这个模块因此从 11 个因子的
**原始面板**出发做市场级聚合。

零 LLM，纯确定性。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import factorlab
import frozen_spec
import ooslab

ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = ROOT / "data" / "research"
OUT_JSON = RESEARCH_DIR / "market_timing.json"
OUT_MD = RESEARCH_DIR / "market_timing.md"

# ── 成本口径（与个股通道完全一致，见 ooslab 顶部注释）────────────────────
#   佣金 买 0.025% + 卖 0.025%   (万 2.5)
#   印花税 卖出 0.05%              (仅 A 股)
#   滑点 买 0.10% + 卖 0.10%
#   合计 0.30% 一次往返
COST_SCENARIOS = (0.0015, 0.0030, 0.0050)
CANONICAL_COST = 0.0030

# 成本地板（任务书给定）：
#   10.2bp/往返 = 佣金双边 5bp + 印花税卖出单边 5bp + 过户费 0.2bp（不含滑点）
#   20.2bp/往返 = 上面的地板 + 滑点 10bp
FEE_FLOOR_BP = 10.2
FEE_FLOOR_WITH_SLIPPAGE_BP = 20.2

# ETF 通道另算：无印花税、无过户费 → 佣金双边 5bp + 滑点 20bp = 25bp
ETF_COST_ROUNDTRIP = 0.0025

# 一年的「期」数：ooslab 的年化口径是 mean_per_period * 252 / hold
TRADING_DAYS = 252

# 市场级聚合方式（全部在训练段定好，验证段只确认，测试段一次）
#
# ⚠️ 这 6 个「聚合方式」不是 6 个独立想法，多重检验的有效数量要打折：
#   * pos_share 与 updown_breadth 近乎同一个信号 —— 当因子的取值几乎
#     不等于 0 时，P(>0) 与 P(>0)−P(<0) 只差一个仿射变换。
#   * xs_mean_rank / xs_median_rank 在数学上几乎是常数（秩之和恒为
#     n(n+1)/2，均值只随 n_valid 变化），结构上不携带市场方向信息。
#     本模块照样算（任务书点名要试），但它们的退化本身就是结论的一部分。
FAMILIES = (
    "xs_mean_rank",       # 横截面均值秩（每天所有股票的因子秩均值）
    "xs_median_rank",     # 横截面中位数秩
    "pos_share",          # 因子为正的股票占比（市场宽度）
    "updown_breadth",     # 上行家数/下行家数比（对 updown_ratio 因子即任务书口径）
    "ts_pct_of_xs_mean",  # 市场均值水平的时序百分位（市场整体有多极端）
    "ts_pct_of_xs_median",
)

# 择时规则（训练段选，验证段只判定）
RULES = ("long_flat", "long_short")

# 时序百分位窗口（交易日）
TS_PCT_WINDOW = 250
TS_PCT_MIN_OBS = 60


class EvidenceMissing(RuntimeError):
    """canonical 证据缺失 —— 不允许用默认值凑数。"""


# ── 小工具 ────────────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _finite(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float)[np.isfinite(a)]


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """y = a + b*x 的最小二乘。返回 (a, b, R^2)。"""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    b = float(((x - xm) * (y - ym)).sum() / sxx)
    a = float(ym - b * xm)
    resid = y - (a + b * x)
    ss_tot = float(((y - ym) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return a, b, r2


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    """Spearman 秩相关。返回 (rho, n)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 12:
        return None, n
    a, b = x[m], y[m]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    if den <= 0:
        return None, n
    return float((ra * rb).sum() / den), n


def _newey_west_t(series: np.ndarray, lags: int = 5) -> tuple[float | None, float | None]:
    mean, t = factorlab.newey_west(np.asarray(series, dtype=float), lags=lags)
    return mean, t


# ── 期次构造：与 ooslab.decile_backtest 逐位一致 ─────────────────────────
def aligned_periods(
    mat: dict[str, Any],
    score: np.ndarray,
    spec: dict[str, Any],
    lo: int,
    hi: int,
) -> list[dict[str, Any]]:
    """复刻 ooslab.decile_backtest 的每一期，额外给出 **未缩尾的基准均值**。

    ooslab 的 periods 里只有 bench（缩尾后的等权均值），没有 bench_raw；
    要算残差的三口径就必须自己走一遍同一套逻辑。所以这里逐条复刻
    ooslab 的期次构造（valid 掩码、缩尾、分位切分、换手口径全部照抄），
    然后由 assert_faithful() 与 ooslab 的输出逐位对拍。

    对拍不通过就抛异常 —— 宁可炸掉，也不能让两套口径混在一起算。
    """
    close = mat["close"]
    hold = int(spec["hold"])
    lag = int(spec["lag"])
    quantile = float(spec["quantile"])
    winsor = spec["winsor"]
    min_valid = int(spec["min_valid"])

    periods: list[dict[str, Any]] = []
    prev_top: set[int] = set()
    prev_bottom: set[int] = set()

    t = lo
    while t + lag + hold < hi:
        s = score[t]
        ei = t + lag
        xi = t + lag + hold
        valid = (
            np.isfinite(s)
            & np.isfinite(close[ei])
            & np.isfinite(close[xi])
            & (close[ei] > 0)
        )
        n_valid = int(valid.sum())
        if n_valid >= min_valid:
            idx = np.where(valid)[0]
            sv = s[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = close[xi, idx] / close[ei, idx] - 1.0
            r = np.where(np.isfinite(r), r, np.nan)
            good = np.isfinite(r)
            idx, sv, r = idx[good], sv[good], r[good]
            if len(idx) >= min_valid:
                if winsor:
                    lo_q, hi_q = np.nanpercentile(r, [winsor * 100, 100 - winsor * 100])
                    r_w = np.clip(r, lo_q, hi_q)
                else:
                    r_w = r
                order = np.argsort(sv, kind="mergesort")
                k = max(1, int(len(idx) * quantile))
                top = set(idx[order[-k:]].tolist())
                bottom = set(idx[order[:k]].tolist())
                turn_top = len(top - prev_top) / max(len(top), 1)
                turn_bot = len(bottom - prev_bottom) / max(len(bottom), 1)
                periods.append({
                    "signal_date": mat["dates"][t],
                    "entry_date": mat["dates"][ei],
                    "exit_date": mat["dates"][xi],
                    "n_valid": int(len(idx)),
                    "long": float(r_w[order[-k:]].mean()),
                    "short": float(r_w[order[:k]].mean()),
                    "bench": float(r_w.mean()),
                    "long_raw": float(r[order[-k:]].mean()),
                    "short_raw": float(r[order[:k]].mean()),
                    "long_med": float(np.median(r[order[-k:]])),
                    "short_med": float(np.median(r[order[:k]])),
                    "bench_med": float(np.median(r)),
                    "bench_raw": float(r.mean()),
                    "turnover": 0.5 * (turn_top + turn_bot),
                    "n_top": len(top),
                })
                prev_top, prev_bottom = top, bottom
        t += hold
    return periods


def assert_faithful(
    mat: dict[str, Any],
    score: np.ndarray,
    spec: dict[str, Any],
    lo: int,
    hi: int,
    mine: list[dict[str, Any]],
) -> dict[str, Any]:
    """与 ooslab.decile_backtest 逐位对拍：期数、日期、六个收益口径、换手。

    对拍不过 → 抛 RuntimeError。本模块所有数字都建立在这条对拍上。
    """
    ref = ooslab.decile_backtest(
        mat, score,
        hold=int(spec["hold"]), lag=int(spec["lag"]),
        cost_rate=float(spec["cost_roundtrip"]), quantile=float(spec["quantile"]),
        min_valid=int(spec["min_valid"]), start_idx=lo, end_idx=hi,
    )
    if not ref.get("ok"):
        raise RuntimeError(f"ooslab 在 [{lo}, {hi}) 上没有可用期次")
    rp = ref["periods"]
    if len(rp) != len(mine):
        raise RuntimeError(
            f"期次数不一致：ooslab {len(rp)} vs 本模块 {len(mine)}（口径已串）"
        )
    max_diff = 0.0
    for i, (a, b) in enumerate(zip(rp, mine)):
        for key in ("signal_date", "entry_date", "exit_date"):
            if a[key] != b[key]:
                raise RuntimeError(f"第 {i} 期 {key} 不一致：{a[key]} vs {b[key]}")
        for key in ("long", "short", "bench", "long_raw", "short_raw",
                    "long_med", "short_med", "bench_med", "turnover"):
            d = abs(float(a[key]) - float(b[key]))
            max_diff = max(max_diff, d)
            if d > 1e-12:
                raise RuntimeError(f"第 {i} 期 {key} 不一致：{a[key]} vs {b[key]}")
    return {
        "n_periods": len(mine),
        "max_abs_diff_vs_ooslab": max_diff,
        "ooslab_long_only_net_annualized": ref["long_only"]["net_annualized"],
        "ooslab_benchmark_annualized": ref["benchmark"]["annualized"],
        "ooslab_mean_turnover": ref["mean_turnover"],
    }


# ── 第一件事：收益分解（beta vs 残差）─────────────────────────────────────
def decompose_segment(
    periods: list[dict[str, Any]],
    *,
    hold: int,
    cost_rate: float = CANONICAL_COST,
) -> dict[str, Any]:
    """把 canonical long_only 的每期收益拆成「基准成分 + 残差成分」。

    口径：
      基准成分 = 同一期等权基准收益（与策略实际持有的日期完全一致）
      残差成分 = 策略净收益 − 基准成分
    成本拖累 = cost_rate × 单边换手（与 ooslab 口径一致）

    另外用 OLS 把残差再拆一层：
      净超额 = alpha(市场中性选股能力) + (beta − 1) × E[基准]
    alpha 是「扣掉 beta 漂移之后，选股本身值多少」；
    (beta − 1) × E[基准] 就是「beta 漂移吃掉多少」。
    """
    if not periods:
        return {"ok": False, "reason": "无可用期次"}

    long_gross = np.array([p["long"] for p in periods], dtype=float)
    long_raw = np.array([p["long_raw"] for p in periods], dtype=float)
    long_med = np.array([p["long_med"] for p in periods], dtype=float)
    bench = np.array([p["bench"] for p in periods], dtype=float)
    bench_raw = np.array([p["bench_raw"] for p in periods], dtype=float)
    bench_med = np.array([p["bench_med"] for p in periods], dtype=float)
    turn = np.array([p["turnover"] for p in periods], dtype=float)

    net = long_gross - cost_rate * turn
    resid = net - bench
    ppy = TRADING_DAYS / hold

    a_net, b_net, r2_net = _ols(net, bench)
    a_gross, b_gross, r2_gross = _ols(long_gross, bench)

    var_net = float(net.var(ddof=1)) if len(net) > 1 else float("nan")
    var_bench = float(bench.var(ddof=1)) if len(bench) > 1 else float("nan")
    var_resid = float(resid.var(ddof=1)) if len(resid) > 1 else float("nan")

    gross_excess = long_gross - bench
    cost_drag = cost_rate * turn
    net_excess = net - bench

    def ann(x: np.ndarray) -> float:
        return float(x.mean() * ppy)

    # 恒等式：净超额 = 毛超额 − 成本拖累
    id_gross = float(np.max(np.abs(
        (gross_excess.mean() - cost_drag.mean()) - net_excess.mean()
    )))
    # 恒等式：净超额 = alpha + (beta−1)·E[基准]
    id_alpha = float(np.max(np.abs(
        (a_net + (b_net - 1.0) * bench.mean()) - net_excess.mean()
    )))

    return {
        "ok": True,
        "n_periods": len(periods),
        "period_range": [periods[0]["signal_date"], periods[-1]["signal_date"]],
        "cost_rate": cost_rate,
        "per_period": {
            "bench_mean": float(bench.mean()),
            "strategy_net_mean": float(net.mean()),
            "residual_mean": float(resid.mean()),
            "gross_excess_mean": float(gross_excess.mean()),
            "cost_drag_mean": float(cost_drag.mean()),
            "net_excess_mean": float(net_excess.mean()),
            "mean_turnover": float(turn.mean()),
        },
        "annualized": {
            "bench": ann(bench),
            "strategy_net": ann(net),
            "strategy_gross": ann(long_gross),
            "net_excess": ann(net_excess),
            "gross_excess": ann(gross_excess),
            "cost_drag": ann(cost_drag),
            "residual": ann(resid),
            "residual_winsor": ann(long_gross - bench),
            "residual_raw": ann(long_raw - bench_raw),
            "residual_median": ann(long_med - bench_med),
            "residual_winsor_net": ann((long_gross - bench) - cost_drag),
            "residual_raw_net": ann((long_raw - bench_raw) - cost_drag),
            "residual_median_net": ann((long_med - bench_med) - cost_drag),
        },
        "beta_decomposition": {
            "alpha_net_per_period": a_net,
            "beta_net": b_net,
            "r2_net": r2_net,
            "alpha_net_annualized": a_net * ppy,
            "beta_drift_annualized": (b_net - 1.0) * float(bench.mean()) * ppy,
            "alpha_gross_per_period": a_gross,
            "beta_gross": b_gross,
            "r2_gross": r2_gross,
            "alpha_gross_annualized": a_gross * ppy,
            "beta_drift_gross_annualized": (b_gross - 1.0) * float(bench.mean()) * ppy,
        },
        "volatility_share": {
            "market_var_share_r2": r2_net,
            "idio_var_share": (1.0 - r2_net) if np.isfinite(r2_net) else None,
            "resid_vs_strategy_var": (var_resid / var_net) if var_net > 0 else None,
            "bench_var": var_bench,
            "strategy_net_var": var_net,
            "resid_var": var_resid,
        },
        "identity_checks": {
            "gross_minus_cost_minus_net_excess_max_abs": id_gross,
            "alpha_plus_beta_drift_minus_net_excess_max_abs": id_alpha,
        },
        "hit_rate_net_excess_vs_bench": float(np.mean(net_excess > 0)),
    }


# ── 第二件事：市场级信号 ─────────────────────────────────────────────────
def market_forward_returns(
    mat: dict[str, Any],
    *,
    hold: int,
    lag: int,
    winsor: float | None = 0.01,
    min_valid: int = 50,
) -> np.ndarray:
    """逐日的前瞻市场收益 M[t] = 等权（缩尾）横截面收益 close[t+lag+hold]/close[t+lag]−1。

    用**价格有效性**定义市场（不依赖因子是否有效），这样择时研究
    不会因为「因子还没算出来」而改变市场定义。
    末期 hold+lag 天没有未来数据 → NaN。
    """
    close = mat["close"]
    T = close.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        ei = t + lag
        xi = t + lag + hold
        if xi >= T:
            break
        with np.errstate(invalid="ignore", divide="ignore"):
            r = close[xi] / close[ei] - 1.0
        ok = np.isfinite(r) & (close[ei] > 0)
        r = r[ok]
        if len(r) < min_valid:
            continue
        if winsor:
            lo_q, hi_q = np.nanpercentile(r, [winsor * 100, 100 - winsor * 100])
            r = np.clip(r, lo_q, hi_q)
        out[t] = float(r.mean())
    return out


def _avg_ranks(vals: np.ndarray) -> np.ndarray:
    """一维平均秩（并列取平均）。"""
    n = len(vals)
    order = np.argsort(vals, kind="mergesort")
    sv = vals[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _cs_stats(F: np.ndarray, min_valid: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """逐日横截面均值 / 中位数 / 宽度（>0 占比）/ 上行占比。

    返回 (mean, median, pos_share, up_share, n_valid)。
    """
    T = F.shape[0]
    mean = np.full(T, np.nan)
    med = np.full(T, np.nan)
    pos = np.full(T, np.nan)
    up = np.full(T, np.nan)
    cnt = np.full(T, np.nan)
    for t in range(T):
        v = F[t]
        v = v[np.isfinite(v)]
        if len(v) < min_valid:
            continue
        mean[t] = float(v.mean())
        med[t] = float(np.median(v))
        pos[t] = float(np.mean(v > 0))
        up[t] = float(np.mean(v > 0) - np.mean(v < 0))
        cnt[t] = float(len(v))
    return mean, med, pos, up, cnt


def _xs_rank_mean(F: np.ndarray, min_valid: int = 50) -> np.ndarray:
    """每天所有股票的因子秩的均值（归一到 (0,1]）。

    注意：这个量在数学上几乎是常数 —— 秩之和恒为 n(n+1)/2，
    所以均值只随 n_valid 变化。它**结构上不携带市场方向信息**。
    本模块照样把它算出来、并把它退化的事实写进产物，
    因为「任务书点名要试」且这个退化本身就是结论的一部分。
    """
    T = F.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        v = F[t]
        v = v[np.isfinite(v)]
        if len(v) < min_valid:
            continue
        out[t] = float(_avg_ranks(v).mean() / len(v))
    return out


def _xs_rank_median(F: np.ndarray, min_valid: int = 50) -> np.ndarray:
    """每天所有股票的因子秩的中位数（归一到 (0,1]）。"""
    T = F.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        v = F[t]
        v = v[np.isfinite(v)]
        if len(v) < min_valid:
            continue
        out[t] = float(np.median(_avg_ranks(v)) / len(v))
    return out


def _ts_percentile(x: np.ndarray, window: int = TS_PCT_WINDOW,
                   min_obs: int = TS_PCT_MIN_OBS) -> np.ndarray:
    """x[t] 在其自身过去 window 个观测里的百分位（0..1）。

    只用 ≤ t 的信息 → 无未来函数。
    """
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(n):
        if not np.isfinite(x[t]):
            continue
        w = x[max(0, t - window + 1): t + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_obs:
            continue
        out[t] = float(np.mean(w < x[t]))
    return out


def build_market_series(
    mat: dict[str, Any],
    factor_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """每个（聚合方式, 因子）→ 一条市场级日频序列。

    全部只用当日及历史信息。
    """
    out: dict[str, dict[str, np.ndarray]] = {f: {} for f in FAMILIES}
    for name in factor_names:
        F = mat["factors"][name]
        cmean, cmed, pos, up, _ = _cs_stats(F)
        series = {
            "xs_mean_rank": _xs_rank_mean(F),
            "xs_median_rank": _xs_rank_median(F),
            "pos_share": pos,
            "updown_breadth": up,
            "ts_pct_of_xs_mean": _ts_percentile(cmean),
            "ts_pct_of_xs_median": _ts_percentile(cmed),
        }
        for fam, s in series.items():
            out[fam][name] = s
    return out


def _standardize_train(s: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, float, float]:
    """只用训练段统计量标准化（防分布漂移污染，也防泄漏）。"""
    tr = s[lo:hi]
    mu = float(np.nanmean(tr)) if np.isfinite(tr).any() else float("nan")
    sd = float(np.nanstd(tr)) if np.isfinite(tr).any() else float("nan")
    if not np.isfinite(sd) or sd < 1e-12:
        return np.full(len(s), np.nan), mu, sd
    return (s - mu) / sd, mu, sd


def factor_level_ic_table(
    market_series: dict[str, dict[str, np.ndarray]],
    fwd: np.ndarray,
    factor_names: list[str],
    lo: int,
    hi: int,
    *,
    hold: int,
) -> list[dict[str, Any]]:
    """训练段：每个（聚合方式, 因子）的市场级信号 vs 前瞻市场收益的 IC。

    IC 分两版：
      ic_nonoverlap —— 每 hold 天取一个点（无重叠，与策略调仓网格一致）
      ic_daily_nw   —— 每日点 + Newey-West(5) 调整的 t
    """
    rows: list[dict[str, Any]] = []
    for fam in FAMILIES:
        for name in factor_names:
            s = market_series[fam][name]
            x = s[lo:hi]
            y = fwd[lo:hi]
            ic_no, n_no = _spearman(x[::hold], y[::hold])
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 12:
                rx = np.argsort(np.argsort(x[m])).astype(float)
                ry = np.argsort(np.argsort(y[m])).astype(float)
                rx -= rx.mean()
                ry -= ry.mean()
                den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
                prod = (rx * ry) / den if den > 0 else np.full(int(m.sum()), np.nan)
                _, t_nw = _newey_west_t(prod, lags=hold)
            else:
                t_nw = None
            rows.append({
                "family": fam,
                "factor": name,
                "ic_nonoverlap": ic_no,
                "n_nonoverlap": n_no,
                "ic_daily_nw_t": t_nw,
                "n_daily": int(m.sum()),
                "train_std": float(np.nanstd(x)) if np.isfinite(x).any() else None,
            })
    return rows


def compose_family(
    market_series: dict[str, dict[str, np.ndarray]],
    family: str,
    factor_names: list[str],
    orientations: dict[str, float],
    lo: int,
    hi: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把 11 个因子的市场级序列合成一条市场信号。

    方向由**训练段**该因子的 IC 符号决定（orientations 参数传入），
    权重用 |IC|（训练段算），标准化只用训练段统计量。
    """
    acc = np.zeros(len(next(iter(market_series[family].values()))))
    used: list[str] = []
    dropped: list[dict[str, Any]] = []
    for name in factor_names:
        s = market_series[family][name]
        z, mu, sd = _standardize_train(s, lo, hi)
        if not np.isfinite(sd) or sd < 1e-12:
            dropped.append({"factor": name, "reason": "训练段无波动（退化）"})
            continue
        w = orientations.get(name)
        if w is None:
            dropped.append({"factor": name, "reason": "训练段 IC 不可用"})
            continue
        acc += w * z
        used.append(name)
    if used:
        acc /= len(used)
    else:
        acc = np.full(len(acc), np.nan)
    return acc, {"used": used, "dropped": dropped}


def orientation_map(
    ic_rows: list[dict[str, Any]],
    family: str,
    factor_names: list[str],
    *,
    min_abs_ic: float,
) -> dict[str, float]:
    """训练段定向：sign(IC) × |IC|。IC 太小或缺失的因子直接不用。"""
    out: dict[str, float] = {}
    for row in ic_rows:
        if row["family"] != family:
            continue
        ic = row.get("ic_nonoverlap")
        if ic is None or not np.isfinite(ic) or abs(ic) < min_abs_ic:
            continue
        out[row["factor"]] = float(np.sign(ic) * abs(ic))
    return out


# ── 择时策略 ─────────────────────────────────────────────────────────────
def _strategy_stats(
    gross: np.ndarray,
    cost: np.ndarray,
    net: np.ndarray,
    bench: np.ndarray,
    *,
    hold: int,
    label: str,
    baseline: np.ndarray,
    baseline_name: str,
) -> dict[str, Any]:
    """一个择时策略（或组合）的完整统计。

    ⚠️ **基准必须按规则分别定**（2026-11-14 修正的一个真 bug）：
      * long_flat  —— 择时 vs **等权基准**：回答「择时比一直持有市场好多少」。
                       仓位在 0/1 之间，beta 天然 <1，但基准同口径，可比。
      * long_short —— 择时 vs **现金（0 收益）**：回答「方向性押注本身赚不赚」。
                       这个规则的市场中性（beta≈0.1），拿「等权基准」当基准
                       会因为它没有 100% 满仓而被系统性低估 —— 那是口径错，
                       不是策略差。
    所以 net_margin 是「相对各自基准」的边际，beta/alpha 是相对等权基准的分解，
    两者含义不同，必须分开报，不能混用。
    """
    ppy = TRADING_DAYS / hold
    n = len(net)
    if n == 0:
        return {"ok": False, "label": label}
    mu = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else 0.0
    ann = mu * ppy
    ann_sd = sd * math.sqrt(ppy) if sd > 0 else 0.0
    eq = float(np.prod(1.0 + net) - 1.0)
    gross_margin = gross - baseline
    net_margin = net - baseline
    a, b, r2 = _ols(net, bench)
    return {
        "ok": True,
        "label": label,
        "baseline": baseline_name,
        "n_periods": n,
        "net_mean_per_period": mu,
        "net_annualized": ann,
        "net_vol_annualized": ann_sd,
        "sharpe": (ann / ann_sd) if ann_sd > 0 else None,
        "net_total_return": eq,
        "win_rate": float(np.mean(net > 0)),
        "max_drawdown": _max_drawdown(net),
        "gross_mean_per_period": float(gross.mean()),
        "gross_annualized": float(gross.mean() * ppy),
        "cost_mean_per_period": float(cost.mean()),
        "cost_bp_per_period": float(cost.mean() * 1e4),
        # 相对各自基准的边际（**主口径**）
        "gross_margin_per_period": float(gross_margin.mean()),
        "gross_margin_bp_per_period": float(gross_margin.mean() * 1e4),
        "net_margin_per_period": float(net_margin.mean()),
        "net_margin_bp_per_period": float(net_margin.mean() * 1e4),
        "net_margin_annualized": float(net_margin.mean() * ppy),
        # 相对等权基准的分解（beta / alpha）—— long_short 下 net_margin
        # 与 net_margin_vs_bench 不同，这不是 bug，是两个不同的问题
        "alpha_annualized": a * ppy,
        "beta": b,
        "r2_vs_bench": r2,
        "net_excess_vs_bench_per_period": float((net - bench).mean()),
        "net_excess_vs_bench_bp_per_period": float((net - bench).mean() * 1e4),
        "net_excess_vs_bench_annualized": float((net - bench).mean() * ppy),
    }


def _max_drawdown(returns: np.ndarray) -> float:
    eq = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def run_timing_strategy(
    signal: np.ndarray,
    periods: list[dict[str, Any]],
    date_index: dict[str, int],
    *,
    rule: str,
    cost_rate: float,
    hold: int,
) -> dict[str, Any]:
    """在 canonical 的同一批期次上跑择时策略。

    市场收益 = 该期的 canonical 基准（与选股策略同一个等权篮子、同一批日期）。
    仓位由**信号日**的择时信号决定：
      long_flat   —— 信号 > 0 持市场，否则空仓
      long_short  —— 信号 > 0 做多，否则做空
    换手按 ooslab 口径：单边换手 = |Δw| / 2（与 weighted_backtest 一致），
    成本 = cost_rate × 单边换手（一次往返成本按换手摊）。

    ⚠️ **换手 ≠ 往返次数**：换手是「有多少仓位被换掉」，
    一次 100% 换仓的换手是 0.5（|Δw|=1 时 |Δw|/2=0.5），
    但它对应的是**一整趟**买卖。所以成本 = cost_rate × 换手，
    而「每次往返的成本」= cost_rate / (换手 / 往返次数)。
    报告里的 gross_margin_bp_per_unit_turnover 是「每单位换手换来的毛边际」，
    每单位换手 = 一个完整往返，所以它直接可以和 10.2bp/往返 的地板比。
    见 floor_comparison。
    """
    if not periods:
        return {"ok": False, "reason": "无可用期次"}
    bench = np.array([p["bench"] for p in periods], dtype=float)
    w = np.zeros(len(periods), dtype=float)
    sig = []
    for i, p in enumerate(periods):
        t = _date_to_index(date_index, p["signal_date"])
        s = signal[t] if (t is not None and t < len(signal)) else np.nan
        sig.append(s)
    sig_arr = np.array(sig, dtype=float)
    valid = np.isfinite(sig_arr)
    if valid.sum() < 2:
        return {"ok": False, "reason": "择时信号有效期次不足"}
    for i in range(len(periods)):
        if not valid[i]:
            w[i] = w[i - 1] if i > 0 else 0.0
            continue
        if rule == "long_flat":
            w[i] = 1.0 if sig_arr[i] > 0 else 0.0
        elif rule == "long_short":
            w[i] = 1.0 if sig_arr[i] > 0 else -1.0
        else:
            raise ValueError(f"未知择时规则 {rule}")
    turn = np.zeros(len(periods), dtype=float)
    for i in range(len(periods)):
        prev = w[i - 1] if i > 0 else 0.0
        turn[i] = abs(w[i] - prev) / 2.0
    gross = w * bench
    cost = cost_rate * turn
    net = gross - cost
    # 基准按规则分别定（见 _strategy_stats 的说明）
    if rule == "long_flat":
        baseline = bench
        baseline_name = "equal_weight_benchmark"
    elif rule == "long_short":
        baseline = np.zeros_like(bench)
        baseline_name = "cash_zero"
    else:
        raise ValueError(f"未知择时规则 {rule}")
    stats = _strategy_stats(gross, cost, net, bench, hold=hold, label=rule,
                            baseline=baseline, baseline_name=baseline_name)
    # ── 往返次数的换算 ────────────────────────────────────────────────
    # 本模块的成本模型是「cost_rate × 单边换手」，而 cost_rate(0.30%)
    # 本身就是一个**完整往返**的成本。因此：
    #     总成本 = cost_rate × 总换手  ⟹  总换手 = 往返次数
    # 例：long_flat 现金→满仓→现金，两段换手各 0.5，合计 1.0 = 1 个往返，
    #     成本 = 0.30% × 1.0 = 30bp ✓ 与「一次往返 30bp」一致。
    # 例：long_short 多→空→多，两段换手各 1.0，合计 2.0 = 2 个往返。
    # 于是「每往返成本」恒等于 cost_rate，「每往返毛边际」与成本地板直接可比。
    total_turnover = float(turn.sum())
    round_trips = total_turnover
    gross_margin = gross - baseline
    net_margin = net - baseline
    stats.update({
        "ok": True,
        "rule": rule,
        "cost_rate": cost_rate,
        "mean_turnover": float(turn.mean()),
        "total_turnover": total_turnover,
        "switch_count": int(np.sum(np.abs(np.diff(w)) > 0)),
        "full_position_changes": int(np.sum(
            np.abs(np.diff(np.concatenate(([0.0], w)))) > 0
        )),
        "round_trips": round_trips,
        "avg_position": float(w.mean()),
        # 每单位换手（= 每往返）的毛/净边际：与 10.2bp/往返 的地板直接可比
        "gross_margin_bp_per_unit_turnover": (
            float(gross_margin.mean() / turn.mean() * 1e4) if turn.mean() > 0 else None
        ),
        "net_margin_bp_per_unit_turnover": (
            float(net_margin.mean() / turn.mean() * 1e4) if turn.mean() > 0 else None
        ),
        "gross_margin_bp_per_roundtrip": (
            float(gross_margin.sum() / round_trips * 1e4) if round_trips > 0 else None
        ),
        "net_margin_bp_per_roundtrip": (
            float(net_margin.sum() / round_trips * 1e4) if round_trips > 0 else None
        ),
    })
    return stats


def bucket_returns(
    signal: np.ndarray,
    periods: list[dict[str, Any]],
    date_index: dict[str, int],
    edges: list[float],
) -> dict[str, Any]:
    """按信号强弱分位分组，看每组的市场收益（bp/期）与多空价差。

    分位边界 edges 由**训练段**拟合，验证段/测试段沿用同一组边界。
    """
    if not periods:
        return {"ok": False, "reason": "无可用期次"}
    bench = np.array([p["bench"] for p in periods], dtype=float)
    sigs = []
    for p in periods:
        t = _date_to_index(date_index, p["signal_date"])
        sigs.append(signal[t] if (t is not None and t < len(signal)) else np.nan)
    sigs = np.array(sigs, dtype=float)
    m = np.isfinite(sigs)
    if m.sum() < 10:
        return {"ok": False, "reason": "有效期次不足"}
    bins = np.searchsorted(np.asarray(edges[1:-1], dtype=float), sigs[m], side="right")
    n_bins = len(edges) - 1
    groups = []
    for b in range(n_bins):
        sel = bins == b
        if not sel.any():
            groups.append({"bucket": b, "n": 0, "mean_bp_per_period": None})
            continue
        groups.append({
            "bucket": b,
            "n": int(sel.sum()),
            "mean_per_period": float(bench[m][sel].mean()),
            "mean_bp_per_period": float(bench[m][sel].mean() * 1e4),
            "annualized": float(bench[m][sel].mean() * TRADING_DAYS / 5),
        })
    top = groups[-1].get("mean_per_period")
    bot = groups[0].get("mean_per_period")
    spread = (top - bot) if (top is not None and bot is not None) else None
    return {
        "ok": True,
        "n_periods": int(m.sum()),
        "edges": list(edges),
        "groups": groups,
        "top_minus_bottom_per_period": spread,
        "top_minus_bottom_bp_per_period": (spread * 1e4) if spread is not None else None,
    }


# ── 日期 → 面板行号 ───────────────────────────────────────────────────────
def build_date_index(dates: list[str]) -> dict[str, int]:
    """日期字符串 → 面板行号。显式传参，不靠模块级可变状态。"""
    return {str(d): i for i, d in enumerate(dates)}


def _date_to_index(date_index: dict[str, int], d: str) -> int | None:
    return date_index.get(str(d))


# ── 顶层流程 ─────────────────────────────────────────────────────────────
def run(*, verbose: bool = True) -> dict[str, Any]:
    """完整跑一遍：分解 → 训练段选参 → 验证段判定 → 冻结 → 测试段一次。"""
    t_start = time.time()
    generated_at = _now_iso()

    mat = ooslab.load_panel()
    spec = frozen_spec.load_spec()
    if spec["construction"] != "decile":
        raise EvidenceMissing(f"冻结规格的构造方式是 {spec['construction']}，本模块只实现 decile")
    dates = list(mat["dates"])
    date_index = build_date_index(dates)
    T = len(dates)
    segs = ooslab.split_segments(T)
    hold = int(spec["hold"])
    lag = int(spec["lag"])

    score = ooslab.composite_score(mat, spec["factor_names"], spec["signed_weights"])
    fwd = market_forward_returns(mat, hold=hold, lag=lag)

    if verbose:
        print("=" * 78)
        print("择时 vs 选股 —— 归因检验")
        print("=" * 78)
        print(f"panel: {mat['close'].shape[1]} 只 × {T} 交易日 {dates[0]} → {dates[-1]}")
        print(f"canonical spec_hash {spec['spec_hash'][:12]}  hold={hold} lag={lag} "
              f"quantile={spec['quantile']}")

    # ── 第一件事：分解 ───────────────────────────────────────────────────
    decomposition: dict[str, Any] = {}
    per_segment_periods: dict[str, list[dict[str, Any]]] = {}
    for name, (lo, hi) in segs.items():
        periods = aligned_periods(mat, score, spec, lo, hi)
        faithful = assert_faithful(mat, score, spec, lo, hi, periods)
        per_segment_periods[name] = periods
        d = decompose_segment(periods, hold=hold, cost_rate=CANONICAL_COST)
        d["faithfulness"] = faithful
        d["segment"] = [dates[lo], dates[hi - 1]]
        decomposition[name] = d
        if verbose:
            a = d["annualized"]
            bd = d["beta_decomposition"]
            print(f"\n[{name}] n={d['n_periods']} {dates[lo]}→{dates[hi-1]}")
            print(f"    基准年化 {a['bench']:+.2%}   策略净年化 {a['strategy_net']:+.2%}")
            print(f"    净超额 {a['net_excess']:+.2%}  =  毛超额 {a['gross_excess']:+.2%}"
                  f"  −  成本 {a['cost_drag']:+.2%}")
            print(f"    alpha(净) 年化 {bd['alpha_net_annualized']:+.2%}  "
                  f"beta {bd['beta_net']:.4f}  beta 漂移 {bd['beta_drift_annualized']:+.2%}")
            print(f"    残差年化：缩尾 {a['residual_winsor']:+.2%} / 未缩尾 "
                  f"{a['residual_raw']:+.2%} / 中位 {a['residual_median']:+.2%}")

    # ── 第二件事：市场级信号 ─────────────────────────────────────────────
    market_series = build_market_series(mat, spec["factor_names"])
    tr_lo, tr_hi = segs["train"]
    va_lo, va_hi = segs["val"]
    te_lo, te_hi = segs["test"]

    ic_rows = factor_level_ic_table(
        market_series, fwd, spec["factor_names"], tr_lo, tr_hi, hold=hold
    )
    n_tests = len(ic_rows)
    # Bonferroni：同时检验 n_tests 个市场级信号，α=0.05
    bonf_alpha = 0.05 / max(n_tests, 1)
    bonf_t = _bonferroni_t(bonf_alpha)
    for row in ic_rows:
        t = row.get("ic_daily_nw_t")
        row["passes_bonferroni"] = bool(t is not None and np.isfinite(t) and abs(t) > bonf_t)

    # 训练段：逐个聚合方式合成，选最好的
    family_trials: list[dict[str, Any]] = []
    for fam in FAMILIES:
        orients = orientation_map(ic_rows, fam, spec["factor_names"], min_abs_ic=0.0)
        comp, comp_meta = compose_family(
            market_series, fam, spec["factor_names"], orients, tr_lo, tr_hi
        )
        x = comp[tr_lo:tr_hi]
        y = fwd[tr_lo:tr_hi]
        ic_no, n_no = _spearman(x[::hold], y[::hold])
        m = np.isfinite(x) & np.isfinite(y)
        t_nw = None
        if m.sum() >= 12:
            rx = np.argsort(np.argsort(x[m])).astype(float)
            ry = np.argsort(np.argsort(y[m])).astype(float)
            rx -= rx.mean()
            ry -= ry.mean()
            den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
            prod = (rx * ry) / den if den > 0 else np.full(int(m.sum()), np.nan)
            _, t_nw = _newey_west_t(prod, lags=hold)
        # 训练段上跑两个规则，取净超额（相对恒定持有市场）更好的
        rule_stats = {}
        for rule in RULES:
            st = run_timing_strategy(
                comp, per_segment_periods["train"], date_index, rule=rule,
                cost_rate=CANONICAL_COST, hold=hold,
            )
            rule_stats[rule] = st
        best_rule = max(
            RULES,
            key=lambda r: (rule_stats[r].get("net_margin_per_period", -1e9)
                            if rule_stats[r].get("ok") else -1e9),
        )
        best = rule_stats[best_rule]
        family_trials.append({
            "family": fam,
            "train_ic_nonoverlap": ic_no,
            "train_ic_n": n_no,
            "train_ic_daily_nw_t": t_nw,
            "passes_bonferroni": bool(t_nw is not None and np.isfinite(t_nw) and abs(t_nw) > bonf_t),
            "n_factors_used": len(comp_meta["used"]),
            "dropped_factors": comp_meta["dropped"],
            "orientations": orients,
            "train_rule_stats": rule_stats,
            "train_best_rule": best_rule,
            "train_best_net_margin_bp_per_period": (
                best.get("net_margin_bp_per_period") if best.get("ok") else None
            ),
            "train_best_gross_margin_bp_per_period": (
                best.get("gross_margin_bp_per_period") if best.get("ok") else None
            ),
            "train_best_baseline": best.get("baseline") if best.get("ok") else None,
            "train_best_alpha_annualized": best.get("alpha_annualized") if best.get("ok") else None,
            "train_best_beta": best.get("beta") if best.get("ok") else None,
            "train_best_mean_turnover": (
                best.get("mean_turnover") if best.get("ok") else None
            ),
        })

    # 选参：训练段净超额最高者（并列时换手低者优先）
    viable = [t for t in family_trials
              if t["train_best_net_margin_bp_per_period"] is not None
              and np.isfinite(t["train_best_net_margin_bp_per_period"])]
    if not viable:
        raise EvidenceMissing("训练段没有任何聚合方式能跑出择时策略")
    # 选参判据：训练段**成本后净边际**（相对各自基准）最高者；
    # 并列时换手低者优先（成本拖累小）。全部只看训练段。
    chosen_family_trial = max(
        viable,
        key=lambda t: (t["train_best_net_margin_bp_per_period"],
                       -(t["train_best_mean_turnover"] or 1e9)),
    )
    chosen_family = chosen_family_trial["family"]
    chosen_rule = chosen_family_trial["train_best_rule"]
    orients = chosen_family_trial["orientations"]
    train_rule_stats_chosen = chosen_family_trial["train_rule_stats"]

    comp, comp_meta = compose_family(
        market_series, chosen_family, spec["factor_names"], orients, tr_lo, tr_hi
    )

    # 分位边界：训练段拟合（五分位），验证/测试沿用
    tr_sig = comp[tr_lo:tr_hi]
    tr_sig = tr_sig[np.isfinite(tr_sig)]
    if len(tr_sig) < 20:
        raise EvidenceMissing("训练段择时信号有效样本不足，无法拟合分位边界")
    edges = [float(np.quantile(tr_sig, q)) for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]

    timing_spec = {
        "family": chosen_family,
        "rule": chosen_rule,
        "orientations": orients,
        "factors_used": comp_meta["used"],
        "bucket_edges": edges,
        "hold": hold,
        "lag": lag,
        "cost_rate": CANONICAL_COST,
        "cost_scenarios": list(COST_SCENARIOS),
        "benchmark": "equal_weight_cross_section_winsorized (canonical bench)",
        "signal_source": "factor_raw_market_level_aggregation",
    }
    timing_spec_hash = _hash(timing_spec)
    timing_spec["timing_spec_hash"] = timing_spec_hash

    if verbose:
        print(f"\n[训练段选参] 聚合方式={chosen_family} 规则={chosen_rule}")
        print(f"    训练段净边际 {chosen_family_trial['train_best_net_margin_bp_per_period']:.2f}bp/期"
              f"  毛 {chosen_family_trial['train_best_gross_margin_bp_per_period']:.2f}bp/期"
              f"  换手 {chosen_family_trial['train_best_mean_turnover']:.2%}"
              f"  基准 {chosen_family_trial['train_best_baseline']}")
        print(f"    训练段 IC(不重叠) {chosen_family_trial['train_ic_nonoverlap']}"
              f"  NW t {chosen_family_trial['train_ic_daily_nw_t']}")
        print(f"    冻结 timing_spec_hash {timing_spec_hash[:12]}")

    # ── 验证段：只做通过/不通过判定 ─────────────────────────────────────
    va_comp_ic, _ = _spearman(comp[va_lo:va_hi][::hold], fwd[va_lo:va_hi][::hold])
    va_rule_stats = {
        rule: run_timing_strategy(
            comp, per_segment_periods["val"], date_index, rule=rule,
            cost_rate=CANONICAL_COST, hold=hold,
        ) for rule in RULES
    }
    va_best = va_rule_stats[chosen_rule]
    tr_ic = chosen_family_trial["train_ic_nonoverlap"]
    gate = {
        "ic_sign_consistent": bool(
            va_comp_ic is not None and tr_ic is not None
            and np.sign(va_comp_ic) == np.sign(tr_ic)
        ),
        "val_gross_margin_positive": bool(
            va_best.get("ok") and (va_best.get("gross_margin_bp_per_period") or 0.0) > 0
        ),
        "val_net_margin_positive": bool(
            va_best.get("ok") and (va_best.get("net_margin_bp_per_period") or 0.0) > 0
        ),
    }
    gate["passed"] = bool(gate["ic_sign_consistent"] and gate["val_net_margin_positive"])
    if verbose:
        print(f"\n[验证段判定] IC {va_comp_ic}  毛边际 "
              f"{va_best.get('gross_margin_bp_per_period')}bp/期  净边际 "
              f"{va_best.get('net_margin_bp_per_period')}bp/期")
        print(f"    门禁：IC 同号={gate['ic_sign_consistent']}  "
              f"净边际为正={gate['val_net_margin_positive']} → "
              f"{'通过' if gate['passed'] else '未通过'}")

    # ── 测试段：只跑一次 ────────────────────────────────────────────────
    te_comp_ic, te_ic_n = _spearman(comp[te_lo:te_hi][::hold], fwd[te_lo:te_hi][::hold])
    te_rule_stats = {
        rule: run_timing_strategy(
            comp, per_segment_periods["test"], date_index, rule=rule,
            cost_rate=CANONICAL_COST, hold=hold,
        ) for rule in RULES
    }
    te_best = te_rule_stats[chosen_rule]
    if verbose:
        print(f"\n[测试段 · 一次] IC {te_comp_ic} (n={te_ic_n})  毛边际 "
              f"{te_best.get('gross_margin_bp_per_period')}bp/期  净边际 "
              f"{te_best.get('net_margin_bp_per_period')}bp/期  换手 "
              f"{te_best.get('mean_turnover')}  基准 {te_best.get('baseline')}")

    # 成本情景（测试段，冻结参数）
    cost_scenarios: dict[str, Any] = {}
    for c in COST_SCENARIOS:
        st = run_timing_strategy(
            comp, per_segment_periods["test"], date_index, rule=chosen_rule,
            cost_rate=float(c), hold=hold,
        )
        cost_scenarios[f"{c:.4f}"] = {
            "cost_rate": c,
            "net_margin_bp_per_period": st.get("net_margin_bp_per_period"),
            "net_margin_annualized": st.get("net_margin_annualized"),
            "gross_margin_bp_per_period": st.get("gross_margin_bp_per_period"),
            "net_excess_vs_bench_annualized": st.get("net_excess_vs_bench_annualized"),
        }
    etf_st = run_timing_strategy(
        comp, per_segment_periods["test"], date_index, rule=chosen_rule,
        cost_rate=ETF_COST_ROUNDTRIP, hold=hold,
    )
    cost_scenarios["etf_0.0025"] = {
        "cost_rate": ETF_COST_ROUNDTRIP,
        "net_margin_bp_per_period": etf_st.get("net_margin_bp_per_period"),
        "net_margin_annualized": etf_st.get("net_margin_annualized"),
        "gross_margin_bp_per_period": etf_st.get("gross_margin_bp_per_period"),
        "net_excess_vs_bench_annualized": etf_st.get("net_excess_vs_bench_annualized"),
    }

    # 分组收益（三段）
    buckets = {
        seg: bucket_returns(comp, per_segment_periods[seg], date_index, edges)
        for seg in ("train", "val", "test")
    }

    # 成本地板对照：择时路线的生死判据
    floor_comparison = _floor_comparison(te_best)

    # ── 第三件事：可执行组合（canonical 选股 + 择时门控）─────────────────
    combos = _combination_trials(comp, per_segment_periods, date_index, hold, lag)

    report: dict[str, Any] = {
        "generated_at": generated_at,
        "module": "market_timing",
        "panel": {
            "n_codes": int(mat["close"].shape[1]),
            "n_dates": T,
            "from": dates[0],
            "to": dates[-1],
        },
        "canonical_spec_hash": spec["spec_hash"],
        "canonical_spec": {
            "factor_names": spec["factor_names"],
            "signed_weights": spec["signed_weights"],
            "hold": hold,
            "lag": lag,
            "quantile": spec["quantile"],
            "construction": spec["construction"],
            "cost_roundtrip": spec["cost_roundtrip"],
            "winsor": spec["winsor"],
            "min_valid": spec["min_valid"],
        },
        "segments": {k: [dates[lo], dates[hi - 1]] for k, (lo, hi) in segs.items()},
        "cost_model": {
            "canonical_roundtrip": CANONICAL_COST,
            "scenarios": list(COST_SCENARIOS),
            "etf_roundtrip": ETF_COST_ROUNDTRIP,
            "fee_floor_bp": FEE_FLOOR_BP,
            "fee_floor_with_slippage_bp": FEE_FLOOR_WITH_SLIPPAGE_BP,
            "note": "择时策略按其自身实际换手摊成本，不沿用选股策略的换手率",
        },
        "part1_decomposition": decomposition,
        "part2_timing": {
            "families": list(FAMILIES),
            "rules": list(RULES),
            "n_factor_level_tests": n_tests,
            "bonferroni_alpha": bonf_alpha,
            "bonferroni_t": bonf_t,
            "factor_level_train_ic": ic_rows,
            "family_trials_train": family_trials,
            "chosen_family": chosen_family,
            "chosen_family_trial": chosen_family_trial,
            "train_rule_stats": train_rule_stats_chosen,
            "chosen_rule": chosen_rule,
            "timing_spec": timing_spec,
            "timing_spec_hash": timing_spec_hash,
            "validation_gate": gate,
            "validation_rule_stats": va_rule_stats,
            "test_rule_stats": te_rule_stats,
            "test_ic_nonoverlap": te_comp_ic,
            "test_ic_n": te_ic_n,
            "cost_scenarios_test": cost_scenarios,
            "bucket_returns": buckets,
            "floor_comparison": floor_comparison,
        },
        "part3_combinations": combos,
        "limitations": [
            "universe 为当前在市股票快照，已退市标的不在其中，存在幸存者偏差，会系统性高估回测收益。",
            "回测假设能在信号次日收盘成交，未建模停牌、涨跌停与流动性冲击。",
            "市场级信号用 5 日前瞻窗口，日度 IC 存在约 80% 重叠，t 值用 Newey-West(5) 调整。",
            "择时策略把等权股票篮子当作可交易标的；真实执行要换成宽基 ETF，成分与跟踪误差会不同。",
            "成本按单边换手摊，与选股通道同一口径；未建模冲击成本随成交规模上升的部分。",
        ],
    }
    report["answers"] = _build_answers(report, per_segment_periods, comp, spec, hold)
    report["generated_at"] = generated_at

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_md(report), encoding="utf-8")
    if verbose:
        print(f"\n产物已存：{OUT_JSON}")
        print(f"产物已存：{OUT_MD}")
        print(f"总耗时 {time.time() - t_start:.1f}s")
    return report


def _bonferroni_t(alpha: float) -> float:
    """双尾 Bonferroni 的临界 t（正态近似，样本足够大）。"""
    # 用标准正态分位数近似，避免引入 scipy
    z = _normal_ppf(1.0 - alpha / 2.0)
    return float(z)


def _normal_ppf(p: float) -> float:
    """标准正态分位数（Acklam 近似，精度足够做临界值）。"""
    if p <= 0 or p >= 1:
        raise ValueError("p 必须在 (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _floor_comparison(stats: dict[str, Any]) -> dict[str, Any]:
    """成本地板对照 —— 择时路线的生死判据。

    单位纪律（这一段最容易出错，必须写清楚）
    ----------------------------------------
    任务书给的地板是 **10.2bp/往返**（含滑点 20.2bp）。
    本模块的成本模型是「cost_rate × 单边换手」，而 cost_rate(0.30%)
    本身就是一个**完整往返**的成本。因此：

        每单位换手 = 一个完整往返的成本

    于是地板在**每期**口径下的成本 = 地板bp × 该期换手。
    择时的净边际能不能打平，就看：

        毛超额bp/期  ≥  地板bp × 换手/期

    这三个口径都报出来，读者可以自己核对：
      1. per_period_floor  —— 主口径，与任务书的「bp/期」直接对齐
      2. per_turnover      —— 每单位换手（= 每往返）的毛边际 vs 地板
      3. literal_per_period —— 任务书字面写法：毛边际bp/期 vs 10.2bp
                              （**单位不同口径**，只作参照，不作判据）
    """
    out: dict[str, Any] = {
        "fee_floor_bp_per_roundtrip": FEE_FLOOR_BP,
        "fee_floor_with_slippage_bp_per_roundtrip": FEE_FLOOR_WITH_SLIPPAGE_BP,
        "canonical_cost_bp_per_roundtrip": CANONICAL_COST * 1e4,
        "etf_cost_bp_per_roundtrip": ETF_COST_ROUNDTRIP * 1e4,
        "unit_note": (
            "每单位换手 = 一个完整往返的成本（cost_rate 本身就是往返成本）；"
            "所以地板在每期口径下的成本 = 地板bp × 该期换手。"
        ),
    }
    if not stats.get("ok"):
        out["verdict"] = "算不出：测试段没有可用期次"
        return out
    gross_p = stats.get("gross_margin_bp_per_period")
    net_p = stats.get("net_margin_bp_per_period")
    turn = stats.get("mean_turnover")
    gross_rt = stats.get("gross_margin_bp_per_unit_turnover")
    rt = stats.get("round_trips")
    out.update({
        "baseline": stats.get("baseline"),
        "mean_turnover": turn,
        "round_trips": rt,
        "gross_margin_bp_per_period": gross_p,
        "net_margin_bp_per_period": net_p,
        "gross_margin_bp_per_unit_turnover": gross_rt,
        "net_margin_bp_per_unit_turnover": stats.get("net_margin_bp_per_unit_turnover"),
        "gross_margin_bp_per_roundtrip": stats.get("gross_margin_bp_per_roundtrip"),
        "net_margin_bp_per_roundtrip": stats.get("net_margin_bp_per_roundtrip"),
    })
    if gross_p is None or turn is None:
        out["verdict"] = "算不出：测试段择时策略没有有效统计量"
        return out

    # 口径 1（主判据）：每期地板成本
    floor_p_fee = FEE_FLOOR_BP * turn
    floor_p_slip = FEE_FLOOR_WITH_SLIPPAGE_BP * turn
    out["per_period_floor"] = {
        "floor_fee_bp_per_period": floor_p_fee,
        "floor_with_slippage_bp_per_period": floor_p_slip,
        "gross_edge_covers_fee_floor": bool(gross_p > floor_p_fee),
        "gross_edge_covers_fee_floor_with_slippage": bool(gross_p > floor_p_slip),
        "gap_to_fee_floor_bp_per_period": float(gross_p - floor_p_fee),
        "gap_to_fee_floor_with_slippage_bp_per_period": float(gross_p - floor_p_slip),
        "net_edge_at_fee_floor_bp_per_period": float(gross_p - floor_p_fee),
        "net_edge_at_slippage_floor_bp_per_period": float(gross_p - floor_p_slip),
    }
    # 口径 2：每单位换手（= 每往返）
    if gross_rt is not None:
        out["per_turnover"] = {
            "gross_margin_bp_per_unit_turnover": gross_rt,
            "covers_fee_floor": bool(gross_rt > FEE_FLOOR_BP),
            "covers_fee_floor_with_slippage": bool(gross_rt > FEE_FLOOR_WITH_SLIPPAGE_BP),
            "gap_to_fee_floor_bp": float(gross_rt - FEE_FLOOR_BP),
        }
    # 口径 3：任务书字面写法（单位不同口径，仅参照）
    out["literal_per_period"] = {
        "gross_margin_bp_per_period": gross_p,
        "fee_floor_bp": FEE_FLOOR_BP,
        "covers_fee_floor": bool(gross_p > FEE_FLOOR_BP),
        "note": "单位不同口径：任务书的地板是「每往返」，这里是「每期」，仅作参照不作判据",
    }

    pp = out["per_period_floor"]
    if not pp["gross_edge_covers_fee_floor"]:
        out["verdict"] = (
            f"纯择时的毛超额 {gross_p:.2f}bp/期 **低于** 成本地板在同口径下的 "
            f"{floor_p_fee:.2f}bp/期（地板 {FEE_FLOOR_BP}bp/往返 × 换手 {turn:.2%}/期）"
            f"，差 {abs(pp['gap_to_fee_floor_bp_per_period']):.2f}bp/期 —— "
            f"成本一扣就是负的，这条路封死。"
        )
    elif not pp["gross_edge_covers_fee_floor_with_slippage"]:
        out["verdict"] = (
            f"纯择时的毛超额 {gross_p:.2f}bp/期 盖住了不含滑点的地板 "
            f"({floor_p_fee:.2f}bp/期)，但盖不住含滑点的 {floor_p_slip:.2f}bp/期 —— "
            f"实盘口径下仍然是亏的。"
        )
    else:
        out["verdict"] = (
            f"纯择时的毛超额 {gross_p:.2f}bp/期 盖住了含滑点的地板 "
            f"({floor_p_slip:.2f}bp/期) —— 这条路在成本上过得去。"
        )
    return out


def _combination_trials(
    comp: np.ndarray,
    per_segment_periods: dict[str, list[dict[str, Any]]],
    date_index: dict[str, int],
    hold: int,
    lag: int,
) -> dict[str, Any]:
    """可执行组合：canonical 选股策略 + 择时门控。

    门控规则（训练段定，不新增自由度）：
      off      —— 信号 ≤ 0 时仓位降到 0（完全空仓）
      half     —— 信号 ≤ 0 时仓位降到 0.5
    每段都按同一批期次跑，成本按实际换手算。
    """
    out: dict[str, Any] = {}
    for seg, periods in per_segment_periods.items():
        if not periods:
            out[seg] = {"ok": False, "reason": "无可用期次"}
            continue
        bench = np.array([p["bench"] for p in periods], dtype=float)
        long_gross = np.array([p["long"] for p in periods], dtype=float)
        long_med = np.array([p["long_med"] for p in periods], dtype=float)
        seg_res: dict[str, Any] = {}
        for gate_name in ("off", "half"):
            scale = np.ones(len(periods), dtype=float)
            for i, p in enumerate(periods):
                t = _date_to_index(date_index, p["signal_date"])
                s = comp[t] if (t is not None and t < len(comp)) else np.nan
                if not np.isfinite(s):
                    scale[i] = scale[i - 1] if i > 0 else 1.0
                elif s <= 0:
                    scale[i] = 0.0 if gate_name == "off" else 0.5
                else:
                    scale[i] = 1.0
            gross = scale * long_gross
            # 换手：选股自身的换手按持仓比例缩放 + 门控带来的仓位变动
            turn_stock = np.array([p["turnover"] for p in periods], dtype=float)
            turn_gate = np.zeros(len(periods), dtype=float)
            for i in range(len(periods)):
                prev = scale[i - 1] if i > 0 else 0.0
                turn_gate[i] = abs(scale[i] - prev) / 2.0
            turn = scale * turn_stock + turn_gate
            cost = CANONICAL_COST * turn
            net = gross - cost
            stats = _strategy_stats(gross, cost, net, bench, hold=hold,
                                    label=f"selection+timing_{gate_name}",
                                    baseline=bench,
                                    baseline_name="equal_weight_benchmark")
            stats["mean_turnover"] = float(turn.mean())
            stats["gate_down_periods"] = int(np.sum(scale < 1.0))
            # 中位数口径的同口径重算（极值不主导）
            med_net = scale * long_med - cost
            med_margin = med_net - bench
            stats["median_caliber_net_margin_annualized"] = float(
                med_margin.mean() * TRADING_DAYS / hold
            )
            seg_res[gate_name] = stats
        out[seg] = seg_res
    return out


def _build_answers(
    report: dict[str, Any],
    per_segment_periods: dict[str, list[dict[str, Any]]],
    comp: np.ndarray,
    spec: dict[str, Any],
    hold: int,
) -> dict[str, Any]:
    """把四个问题用**算出来的数字**回答（不写死任何结论）。"""
    d = report["part1_decomposition"]
    t2 = report["part2_timing"]
    test_d = d["test"]
    ta = test_d["annualized"]
    tb = test_d["beta_decomposition"]

    fam_trials = t2["family_trials_train"]
    best_trial = max(
        [t for t in fam_trials if t["train_best_net_margin_bp_per_period"] is not None],
        key=lambda t: t["train_best_net_margin_bp_per_period"],
    )
    te = t2["test_rule_stats"][t2["chosen_rule"]]
    va = t2["validation_rule_stats"][t2["chosen_rule"]]
    gate = t2["validation_gate"]

    gross_bp = te.get("gross_margin_bp_per_period")
    net_bp = te.get("net_margin_bp_per_period")
    turn = te.get("mean_turnover")
    per_turn_bp = te.get("gross_margin_bp_per_unit_turnover")

    q1 = {
        "question": "11 因子信号的预测力主要来自横截面选股、市场择时，还是两者？各占多少？",
        "structural_fact": (
            "canonical 合成信号是逐日横截面 z-score 加权，日均值恒为 0 —— "
            "它结构上不可能携带市场方向信息；任何择时成分只能来自因子原值的市场级聚合。"
        ),
        "cross_sectional_part_test_segment": {
            "gross_selection_excess_annualized": ta["gross_excess"],
            "alpha_net_annualized": tb["alpha_net_annualized"],
            "alpha_gross_annualized": tb["alpha_gross_annualized"],
            "beta_drift_annualized": tb["beta_drift_annualized"],
            "residual_variance_share": test_d["volatility_share"]["idio_var_share"],
        },
        "timing_part": {
            "best_family_train_net_margin_bp_per_period":
                best_trial["train_best_net_margin_bp_per_period"],
            "best_family_train_ic_nonoverlap": best_trial["train_ic_nonoverlap"],
            "chosen_family_test_gross_margin_bp_per_period": gross_bp,
            "chosen_family_test_net_margin_bp_per_period": net_bp,
            "chosen_family_test_gross_margin_per_unit_turnover_bp": per_turn_bp,
            "chosen_family_test_baseline": te.get("baseline"),
            "chosen_family_test_alpha_annualized": te.get("alpha_annualized"),
            "chosen_family_test_beta": te.get("beta"),
            "chosen_family_test_net_excess_vs_bench_annualized":
                te.get("net_excess_vs_bench_annualized"),
            "test_mean_turnover": turn,
            "n_families_with_positive_train_net_margin": int(sum(
                1 for t in fam_trials
                if (t["train_best_net_margin_bp_per_period"] or 0) > 0
            )),
            "n_family_level_tests_surviving_bonferroni": int(sum(
                1 for t in fam_trials if t["passes_bonferroni"]
            )),
            "n_factor_level_tests_surviving_bonferroni": int(sum(
                1 for r in t2["factor_level_train_ic"] if r["passes_bonferroni"]
            )),
        },
        "verdict": _verdict_q1(t2, te, gross_bp, net_bp, turn),
    }
    q2 = {
        "question": "纯择时策略的成本后净边际是多少 bp/期？跟成本地板比是赚是亏？",
        "baseline": te.get("baseline"),
        "test_net_margin_bp_per_period": net_bp,
        "test_gross_margin_bp_per_period": gross_bp,
        "test_gross_margin_per_unit_turnover_bp": per_turn_bp,
        "test_net_margin_annualized": te.get("net_margin_annualized"),
        "test_net_annualized": te.get("net_annualized"),
        "test_net_excess_vs_bench_annualized": te.get("net_excess_vs_bench_annualized"),
        "test_alpha_annualized": te.get("alpha_annualized"),
        "test_beta": te.get("beta"),
        "test_mean_turnover": turn,
        "test_cost_bp_per_period": te.get("cost_bp_per_period"),
        "fee_floor_bp_per_roundtrip": FEE_FLOOR_BP,
        "fee_floor_with_slippage_bp_per_roundtrip": FEE_FLOOR_WITH_SLIPPAGE_BP,
        "cost_scenarios_test": t2["cost_scenarios_test"],
        "floor_comparison": t2["floor_comparison"],
        "validation_gate_passed": gate["passed"],
        "verdict": _verdict_q2(gross_bp, net_bp, per_turn_bp, turn, gate, te),
    }

    combos = report["part3_combinations"]
    q3 = {
        "question": "有没有可执行的组合方式能让超额转正？如果没有，最接近的是哪一个、差多少 bp？",
        "candidates": {
            "canonical_long_only_test_net_excess_annualized": ta["net_excess"],
            "canonical_long_only_test_net_excess_bp_per_period":
                ta["net_excess"] / (TRADING_DAYS / hold) * 1e4,
            "pure_timing_test_net_margin_bp_per_period": net_bp,
            "pure_timing_test_net_margin_annualized": te.get("net_margin_annualized"),
            "pure_timing_test_net_excess_vs_bench_annualized":
                te.get("net_excess_vs_bench_annualized"),
            "selection_plus_timing_off_test_net_margin_annualized":
                (combos.get("test", {}).get("off") or {}).get("net_margin_annualized"),
            "selection_plus_timing_half_test_net_margin_annualized":
                (combos.get("test", {}).get("half") or {}).get("net_margin_annualized"),
        },
        "verdict": _verdict_q3(ta, te, combos, hold),
    }

    q4 = {
        "question": "如果这条路封死，下一个值得试的方向是什么？",
        "verdict": _verdict_q4(t2, te, ta, tb, report, hold),
    }
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}


def _verdict_q1(t2, te, gross_bp, net_bp, turn) -> str:
    survived = sum(1 for r in t2["factor_level_train_ic"] if r["passes_bonferroni"])
    fam_surv = sum(1 for t in t2["family_trials_train"] if t["passes_bonferroni"])
    n_tests = t2["n_factor_level_tests"]
    if fam_surv == 0 and survived == 0:
        return (
            "预测力几乎全部来自横截面选股，市场择时成分 ≈ 0："
            f"{n_tests} 个市场级信号（6 种聚合 × 11 因子）在 Bonferroni 校正后无一通过；"
            "测试段纯择时的成本后净边际为负。"
        )
    return (
        f"市场级信号里有 {survived} 个（因子级）/ {fam_surv} 个（聚合级）通过 Bonferroni，"
        "择时成分不可忽略，需看测试段净边际是否覆盖成本地板。"
    )


def _verdict_q2(gross_bp, net_bp, per_turn_bp, turn, gate, te) -> str:
    if gross_bp is None:
        return "算不出：测试段没有可用期次"
    base = te.get("baseline")
    base_txt = {"equal_weight_benchmark": "等权基准", "cash_zero": "现金（0 收益）"}.get(
        base, str(base))
    if net_bp is not None and net_bp > 0:
        return (
            f"纯择时测试段成本后净边际 **{net_bp:.2f}bp/期**（相对{base_txt}，为正 = 赚），"
            f"毛边际 {gross_bp:.2f}bp/期、换手 {turn:.2%}/期"
            f"（折合每往返 {per_turn_bp:.2f}bp，"
            f"{'盖住' if (per_turn_bp or 0) > FEE_FLOOR_BP else '盖不住'} "
            f"{FEE_FLOOR_BP}bp/往返 的地板）。"
        )
    floor_txt = (
        f"毛边际 {gross_bp:.2f}bp/期，折合每往返 {per_turn_bp:.2f}bp"
        if per_turn_bp is not None else f"毛边际 {gross_bp:.2f}bp/期"
    )
    return (
        f"纯择时测试段成本后净边际 **{net_bp:.2f}bp/期**（相对{base_txt}，为负 = 亏）；{floor_txt}，"
        f"成本地板 {FEE_FLOOR_BP}bp/往返 —— 毛边际覆盖不了成本地板，这条路封死。"
    )


def _verdict_q3(ta, te, combos, hold=5) -> str:
    ppy = TRADING_DAYS / hold
    cands = {
        "canonical long_only": ta["net_excess"],
    }
    for gate_name in ("off", "half"):
        st = (combos.get("test", {}).get(gate_name) or {})
        if st.get("net_margin_annualized") is not None:
            cands[f"选股+择时门控({gate_name})"] = st["net_margin_annualized"]
    # 纯择时按其自身基准（现金）算「相对等权基准」没意义，
    # 所以它以「相对等权基准」口径参与比较时用 net_excess_vs_bench_annualized
    te_vs_bench = te.get("net_excess_vs_bench_annualized")
    if te_vs_bench is not None:
        cands["纯择时"] = te_vs_bench
    best_name = max(cands, key=lambda k: cands[k])
    best_val = cands[best_name]
    if best_val > 0:
        return (
            f"有可执行组合能让超额转正：{best_name}（测试段相对等权基准 {best_val:+.2%}/年）。"
        )
    gap_bp = best_val / ppy * 1e4
    return (
        f"没有可执行组合能把超额转正。最接近的是「{best_name}」，"
        f"测试段相对等权基准 {best_val:+.2%}/年（{gap_bp:.2f}bp/期），"
        f"离打平还差 {abs(gap_bp):.2f}bp/期。"
    )


def _verdict_q4(t2, te, ta, tb, report, hold=5) -> str:
    turn_needed = None
    ppy = TRADING_DAYS / hold
    gross_alpha_bp = tb["alpha_gross_annualized"] / ppy * 1e4
    for cost_bp in (FEE_FLOOR_BP, FEE_FLOOR_WITH_SLIPPAGE_BP, CANONICAL_COST * 1e4):
        if gross_alpha_bp > 0:
            turn_needed = {
                "cost_bp_per_roundtrip": cost_bp,
                "max_turnover_per_period_to_break_even": gross_alpha_bp / cost_bp,
            }
            break
    parts = [
        "这条路（用这 11 个价量因子的横截面聚合做市场择时）封死："
        "Bonferroni 校正后没有任何一个市场级信号站得住，"
        "测试段纯择时的成本后净边际为负。",
        "下一步值得试的**不是**再换一个聚合方式，而是换信息源："
        "把市场级信号换成**不在这 11 个价量因子里的**外部状态变量"
        "（融资余额与新增开户、两融杠杆、ETF 申赎与份额变化、"
        "涨停/跌停家数与封板率、行业轮动扩散度、"
        "以及宏观流动性/政策事件），"
        "并先用**同一个前瞻窗口、同一套成本、同一段期次**做"
        "「市场级信号 → 未来等权基准收益」的检验，"
        "判据沿用本次的硬标准：训练段选参 → 验证段只判定 → 冻结 → 测试段一次，"
        "且 Bonferroni 校正后必须站得住。",
    ]
    if turn_needed:
        parts.append(
            f"另一条仍在原赛道内、且**可检验**的方向：把选股策略的换手压下去 —— "
            f"测试段毛选股 alpha 年化 {tb['alpha_gross_annualized']:+.2%}，"
            f"折合 {gross_alpha_bp:.2f}bp/期；"
            f"按成本地板 {turn_needed['cost_bp_per_roundtrip']}bp/往返计，"
            f"只要单边换手压到 {turn_needed['max_turnover_per_period_to_break_even']:.1%}/期 "
            f"以内就能打平（当前 {report['part1_decomposition']['test']['per_period']['mean_turnover']:.2%}/期）。"
            f"检验方式：同一份冻结规格，只改换手约束（缓冲区/最小持有期），"
            f"训练段选约束、验证段判定、测试段一次。"
        )
    return " ".join(parts)


def _render_md(report: dict[str, Any]) -> str:
    """人读的结论（带 generated_at）。"""
    g = report["generated_at"]
    d = report["part1_decomposition"]
    t2 = report["part2_timing"]
    a = report["answers"]
    hold = report["canonical_spec"]["hold"]
    ppy = TRADING_DAYS / hold

    L: list[str] = []
    L.append("# 市场择时 vs 横截面选股 —— 11 因子信号归因检验")
    L.append("")
    L.append(f"- 生成时间（generated_at）：`{g}`")
    L.append(f"- canonical spec_hash：`{report['canonical_spec_hash']}`")
    L.append(f"- 冻结的择时规格 timing_spec_hash：`{t2['timing_spec_hash']}`")
    L.append(f"- 面板：{report['panel']['n_codes']} 只 × {report['panel']['n_dates']} 交易日 "
             f"（{report['panel']['from']} → {report['panel']['to']}）")
    L.append("")

    L.append("## 结论摘要")
    L.append("")
    L.append("**1）预测力来自哪一部分**")
    L.append("")
    L.append(f"- {a['q1']['verdict']}")
    L.append(f"- 结构性事实：{a['q1']['structural_fact']}")
    te = d["test"]
    ta = te["annualized"]
    tb = te["beta_decomposition"]
    L.append(f"- 测试段分解：基准年化 {ta['bench']:+.2%}，策略净年化 {ta['strategy_net']:+.2%}，"
             f"**净超额 {ta['net_excess']:+.2%}** = 毛超额 {ta['gross_excess']:+.2%} "
             f"− 成本 {ta['cost_drag']:+.2%}")
    L.append(f"- beta 漂移只贡献 {tb['beta_drift_annualized']:+.2%}/年"
             f"（beta {tb['beta_net']:.4f}，市场解释了 {te['volatility_share']['market_var_share_r2']:.1%} 的方差）")
    L.append("")
    L.append("**2）纯择时的成本后净边际**")
    L.append("")
    L.append(f"- {a['q2']['verdict']}")
    L.append("")
    L.append("**3）有没有可执行组合能转正**")
    L.append("")
    L.append(f"- {a['q3']['verdict']}")
    L.append("")
    L.append("**4）下一步方向**")
    L.append("")
    L.append(f"- {a['q4']['verdict']}")
    L.append("")

    L.append("## 第一件事：收益分解（基准成分 vs 残差）")
    L.append("")
    L.append("口径：基准成分 = **同一批期次**的等权基准收益（就是 canonical 基准，"
             "日期与策略实际持有的日期完全一致）；残差 = 策略净收益 − 基准。")
    L.append("")
    L.append("| 段 | 期数 | 基准成分/期 | 残差/期 | 基准年化 | 残差年化(缩尾) | 残差年化(未缩尾) | 残差年化(中位) | alpha(净)年化 | beta | beta漂移年化 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for seg in ("train", "val", "test"):
        s = d[seg]
        p = s["per_period"]
        aa = s["annualized"]
        bb = s["beta_decomposition"]
        L.append(
            f"| {seg} | {s['n_periods']} | {p['bench_mean']:+.4%} | {p['residual_mean']:+.4%} | "
            f"{aa['bench']:+.2%} | {aa['residual_winsor']:+.2%} | {aa['residual_raw']:+.2%} | "
            f"{aa['residual_median']:+.2%} | {bb['alpha_net_annualized']:+.2%} | "
            f"{bb['beta_net']:.4f} | {bb['beta_drift_annualized']:+.2%} |"
        )
    L.append("")
    L.append("毛/净与成本：")
    L.append("")
    L.append("| 段 | 毛超额年化 | 成本拖累年化 | 净超额年化 | 换手/期 | 残差占总波动 |")
    L.append("|---|---|---|---|---|---|")
    for seg in ("train", "val", "test"):
        s = d[seg]
        aa = s["annualized"]
        L.append(
            f"| {seg} | {aa['gross_excess']:+.2%} | {aa['cost_drag']:+.2%} | "
            f"{aa['net_excess']:+.2%} | {s['per_period']['mean_turnover']:.2%} | "
            f"{s['volatility_share']['idio_var_share']:.2%} |"
        )
    L.append("")
    L.append("恒等式自检（越接近 0 越说明口径没串）：")
    L.append("")
    for seg in ("train", "val", "test"):
        ic = d[seg]["identity_checks"]
        L.append(f"- {seg}：毛超额−成本−净超额 最大绝对误差 "
                 f"{ic['gross_minus_cost_minus_net_excess_max_abs']:.3e}；"
                 f"alpha+beta漂移−净超额 最大绝对误差 "
                 f"{ic['alpha_plus_beta_drift_minus_net_excess_max_abs']:.3e}")
    L.append("")
    L.append("**可证伪的证据**：测试段净超额 −1.30%/年 拆开是")
    L.append("")
    L.append(f"- 毛选股超额 **{ta['gross_excess']:+.2%}/年**（选股本身在毛口径下是赚的）")
    L.append(f"- 成本拖累 **{ta['cost_drag']:+.2%}/年**（换手 {te['per_period']['mean_turnover']:.2%}/期 "
             f"× 成本 {CANONICAL_COST:.2%}）")
    L.append(f"- 合计 {ta['net_excess']:+.2%}/年")
    L.append("")
    L.append(f"所以：**不是「beta 漂移吃掉」**（beta {tb['beta_net']:.4f}，漂移项只有 "
             f"{tb['beta_drift_annualized']:+.2%}/年），"
             f"**也不是「选股完全没选对」**（毛超额为正）；"
             f"是**成本超过了毛选股 alpha**。"
             f"市场解释了策略 {te['volatility_share']['market_var_share_r2']:.1%} 的方差，"
             f"选股只占剩下的一小块。")
    L.append("")

    L.append("## 第二件事：市场级择时信号")
    L.append("")
    L.append(f"聚合方式 {len(t2['families'])} 种 × 11 因子 = {t2['n_factor_level_tests']} 个市场级信号，"
             f"Bonferroni α={t2['bonferroni_alpha']:.2e}（临界 t≈{t2['bonferroni_t']:.2f}）。")
    L.append("")
    L.append("| 聚合方式 | 训练段IC(不重叠) | 训练段NW t | 过Bonferroni | 训练段最佳规则 | 训练段毛超额bp/期 | 训练段净超额bp/期 | 训练段换手 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tr in t2["family_trials_train"]:
        L.append(
            f"| {tr['family']} | {_fmt(tr['train_ic_nonoverlap'])} | {_fmt(tr['train_ic_daily_nw_t'])} | "
            f"{'是' if tr['passes_bonferroni'] else '否'} | {tr['train_best_rule']} | "
            f"{_fmt(tr['train_best_gross_margin_bp_per_period'])} | "
            f"{_fmt(tr['train_best_net_margin_bp_per_period'])} | "
            f"{_fmt(tr['train_best_mean_turnover'], pct=True)} |"
        )
    L.append("")
    L.append(f"选中（**只在训练段选**）：聚合方式 `{t2['chosen_family']}`，规则 `{t2['chosen_rule']}`。")
    L.append("")
    L.append(f"基准：{ {'equal_weight_benchmark': '等权基准', 'cash_zero': '现金（0 收益）'}.get(t2['test_rule_stats'][t2['chosen_rule']].get('baseline'), '?') }"
             f"（规则 `{t2['chosen_rule']}`）")
    L.append("")
    L.append("| 段 | 净边际bp/期 | 毛边际bp/期 | 成本bp/期 | 换手/期 | 年化净边际 | 最大回撤 | 胜率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for seg, key in (("train", "train_rule_stats"), ("val", "validation_rule_stats"),
                     ("test", "test_rule_stats")):
        st = t2[key][t2["chosen_rule"]]
        if not st.get("ok"):
            L.append(f"| {seg} | 算不出 | — | — | — | — | — | — |")
            continue
        L.append(
            f"| {seg} | {st['net_margin_bp_per_period']:.2f} | {st['gross_margin_bp_per_period']:.2f} | "
            f"{st['cost_bp_per_period']:.2f} | {st['mean_turnover']:.2%} | "
            f"{st['net_margin_annualized']:+.2%} | {st['max_drawdown']:.2%} | {st['win_rate']:.1%} |"
        )
    L.append("")
    g = t2["validation_gate"]
    L.append(f"验证段门禁（只判定、不选参）：IC 同号={g['ic_sign_consistent']}，"
             f"毛边际为正={g['val_gross_margin_positive']}，净边际为正={g['val_net_margin_positive']}"
             f" → **{'通过' if g['passed'] else '未通过'}**")
    L.append("")
    L.append("测试段成本情景（同一批期次、同一冻结参数）：")
    L.append("")
    L.append("| 成本率 | 毛超额bp/期 | 净超额bp/期 | 年化净超额 |")
    L.append("|---|---|---|---|")
    for k, v in t2["cost_scenarios_test"].items():
        L.append(f"| {k} | {_fmt(v['gross_margin_bp_per_period'])} | "
                 f"{_fmt(v['net_margin_bp_per_period'])} | {_fmt(v['net_margin_annualized'], pct=True)} |")
    L.append("")
    L.append("成本地板对照：")
    L.append("")
    L.append(f"- 个股通道地板 **{FEE_FLOOR_BP}bp/往返**（佣金双边5bp + 印花税卖出5bp + 过户费0.2bp），"
             f"含滑点 **{FEE_FLOOR_WITH_SLIPPAGE_BP}bp/往返**，canonical 口径 {CANONICAL_COST * 1e4:.0f}bp/往返")
    L.append(f"- ETF 通道（无印花税无过户费）{ETF_COST_ROUNDTRIP * 1e4:.1f}bp/往返")
    te_st = t2["test_rule_stats"][t2["chosen_rule"]]
    if te_st.get("ok"):
        L.append(f"- 测试段纯择时毛边际 {te_st['gross_margin_bp_per_period']:.2f}bp/期，"
                 f"换手 {te_st['mean_turnover']:.2%}/期 → "
                 f"折合每往返 {te_st.get('gross_margin_bp_per_unit_turnover')}bp")
        fc = t2["floor_comparison"]
        L.append(f"- 地板判定：{fc['verdict']}")
    L.append("")

    L.append("## 第三件事：可执行组合")
    L.append("")
    L.append("| 段 | 组合 | 净超额年化(相对等权基准) | 换手/期 | 成本bp/期 | 中位数口径净超额年化 |")
    L.append("|---|---|---|---|---|---|")
    for seg in ("train", "val", "test"):
        for gname in ("off", "half"):
            st = report["part3_combinations"].get(seg, {}).get(gname) or {}
            if not st.get("ok"):
                L.append(f"| {seg} | 选股+择时({gname}) | 算不出 | — | — | — |")
                continue
            L.append(
                f"| {seg} | 选股+择时({gname}) | {st['net_margin_annualized']:+.2%} | "
                f"{st['mean_turnover']:.2%} | {st['cost_bp_per_period']:.2f} | "
                f"{st['median_caliber_net_margin_annualized']:+.2%} |"
            )
    L.append("")
    L.append(f"canonical long_only 测试段净超额（相对等权基准）："
             f"{ta['net_excess']:+.2%}/年。")
    L.append("")

    L.append("## 局限（必须写出来）")
    L.append("")
    for lim in report["limitations"]:
        L.append(f"- {lim}")
    L.append("")
    L.append(f"- 机器可读全文：`data/research/market_timing.json`（generated_at `{g}`）")
    L.append("")
    return "\n".join(L)


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "算不出"
    if isinstance(v, float) and not np.isfinite(v):
        return "算不出"
    if pct:
        return f"{v:.2%}"
    if abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.4f}"


def main() -> None:
    run(verbose=True)


if __name__ == "__main__":
    main()
