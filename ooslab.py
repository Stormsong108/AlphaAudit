"""样本外验证实验室（Out-of-Sample Lab）。

这个模块回答三个问题，每个问题都必须用「当时看不见的数据」回答：

    Q1  成本后还剩多少？    因子 IC 是毛的，A 股交易成本可能吃掉全部边际。
    Q2  时间外还成立吗？    因子发现期与检验期必须严格分开，测试段一次都不许回头调参。
    Q3  概率准吗？          输出的是「上涨概率」，所以必须过 Brier / 校准曲线 / 基线对比。

不可妥协的设计约束：
    * 成交假设：信号日 t 收盘算因子 → t+1 收盘买入 → t+1+hold 收盘卖出。
      绝不用信号日收盘价成交（那是拿当天才知道的信息做当天成交）。
    * 因子发现只用训练段；测试段只用来打分，不用来选因子、不用来调参。
    * **因子必须按发现期 IC 的符号定向**。IC 为负的因子（高分→未来跌），
      做多时要用它的相反方向。用错符号会得到「多空亏钱、多头跑输基准」
      这种看起来像因子失效、其实是代码错误的结论。
    * 前瞻窗口长度（持有期）也只在训练段选，选完整个实验全程锁定。
    * 所有数字都要能从 data/research/*.json 复现。
    * 任何环节失败都必须留下失败样本，不允许静默跳过。

成本模型（A 股口径，一次完整往返）：
    佣金   买 0.025% + 卖 0.025%   (万 2.5)
    印花税 卖出 0.05%              (仅 A 股)
    滑点   买 0.10% + 卖 0.10%
    ------------------------------------------
    合计   0.30% 一次往返
    基准用 0.30%，同时做 0.15% / 0.50% 敏感性测试。

已知局限（必须写在报告里，不能装作没有）：
    * 幸存者偏差：universe 是当前在市的股票快照，已退市的标的不在其中，
      会系统性高估回测收益。
    * 停牌与涨跌停未建模：回测假设能在信号次日收盘成交。
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import factorlab
import dataquality

ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = ROOT / "data" / "research"

# 一次完整往返的交易成本（含佣金 + 印花税 + 滑点）
COST_ROUNDTRIP = 0.0030
COST_SCENARIOS = (0.0015, 0.0030, 0.0050)

# 候选前瞻窗口（交易日）。选哪个只在训练段决定。
HORIZON_CANDIDATES = (5, 10, 20)

# 默认随机种子：抽样时用固定种子，保证结果可复现。
# 同一份代码 + 同一份数据 + 同一个种子 = 逐位相同的结果。
RANDOM_SEED = 20261111


# ------------------------------------------------------------------ 基础工具


def make_forward(mat: dict[str, Any], horizon: int) -> np.ndarray:
    """从收盘价矩阵算前瞻收益矩阵 [T, N]。

    fwd[t, i] = close[t + horizon, i] / close[t, i] - 1
    最后 horizon 行没有未来数据，置 NaN。
    """
    close = mat["close"]
    T, N = close.shape
    fwd = np.full((T, N), np.nan)
    if T <= horizon:
        return fwd
    with np.errstate(invalid="ignore", divide="ignore"):
        f = close[horizon:] / close[:-horizon] - 1.0
    f[~np.isfinite(f)] = np.nan
    fwd[: T - horizon, :] = f
    return fwd


def cs_zscore(mat: np.ndarray, *, min_valid: int = 20) -> np.ndarray:
    """逐行（逐日）横截面 z-score。

    mat: [T, N] → [T, N]，每日独立标准化，不跨日期。
    只有横截面同一时刻的比较才有意义，所以归一化必须逐日做。
    """
    mat = np.asarray(mat, dtype=float)
    valid = np.isfinite(mat)
    cnt = valid.sum(axis=1, keepdims=True)
    x = np.where(valid, mat, 0.0)
    s = x.sum(axis=1, keepdims=True)
    mean = np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0)
    d = np.where(valid, mat - mean, 0.0)
    denom = np.maximum(cnt - 1, 1)
    ssd = np.sqrt((d ** 2).sum(axis=1, keepdims=True) / denom)
    z = np.divide(d, ssd, out=np.zeros_like(d), where=ssd > 0)
    return np.where(valid & (cnt >= min_valid), z, np.nan)


def composite_score(
    mat: dict[str, Any],
    factor_names: list[str],
    weights: list[float] | None = None,
) -> np.ndarray:
    """用指定因子做横截面 z-score 加权合成，返回 [T, N]。

    权重默认等权。**权重的正负号决定因子的方向**，必须由发现期 IC 的符号
    决定（IC<0 的因子给负权重），否则会把反向因子当正向信号用。

    调用方必须只传「在发现期选出的因子」，否则就把未来信息泄漏进检验期了。
    """
    T, N = mat["close"].shape
    if not factor_names:
        return np.full((T, N), np.nan)
    if weights is None:
        weights = [1.0] * len(factor_names)
    zsum = np.zeros((T, N))
    wsum = np.zeros((T, N))
    for name, w in zip(factor_names, weights):
        if name not in mat["factors"]:
            continue
        z = cs_zscore(mat["factors"][name])
        ok = np.isfinite(z)
        zsum[ok] += w * z[ok]
        wsum[ok] += abs(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, zsum / wsum, np.nan)


def auc(probs: np.ndarray, labels: np.ndarray) -> float | None:
    """秩 AUC（Mann-Whitney U 的等价形式）。

    并列值按平均秩处理会有微小偏差，这里不做修正，
    但在本样本量（十万级）下偏差远小于统计噪声。
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    n = len(y)
    if n == 0:
        return None
    npos = int(y.sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return float((ranks[y > 0.5].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# ------------------------------------------------------------------ 策略检验


def decile_backtest(
    mat: dict[str, Any],
    score: np.ndarray,
    *,
    hold: int = 5,
    lag: int = 1,
    cost_rate: float = COST_ROUNDTRIP,
    quantile: float = 0.1,
    min_valid: int = 50,
    start_idx: int = 0,
    end_idx: int | None = None,
    winsor: float | None = 0.01,
) -> dict[str, Any]:
    """分组策略回测（成本后）。

    流程：
        信号日 t   → 用 score[t] 排序选股（score[t] 只含 ≤t 的信息）
        买入日 t+lag → 以 close[t+lag] 成交
        卖出日 t+lag+hold → 以 close[t+lag+hold] 成交

    统计口径（为什么要有三个）：
        2026-11-10 的教训 —— 面板里若有极端收益（复牌缺口、除权穿零），
        **单只股票的异常就能把分组均值彻底带偏**：实测同一份数据，
        十分位**中位数**价差 +0.71%（与 IC 完全一致），**均值**价差 −0.19%。
        所以这里同时给三个口径，缺一不可：
          * long_only        —— 横截面 1%/99% 缩尾后的均值（主口径，可执行性最接近真实）
          * long_only_raw    —— 未缩尾的均值（披露极值拖累有多大）
          * long_only_median —— 组内中位数（抗极值，与 IC 方向应一致）

    产出：
        long_only   多头组（最高分位）净值，扣一次往返成本
        long_short  多空组合（最高分位 - 最低分位），扣两腿成本
        benchmark   同期全市场等权
        turnover    单边换手率（成本按 turnover 摊）
        breakeven   多空超额被打平的临界成本
    """
    close = mat["close"]
    T, N = close.shape
    end_idx = T if end_idx is None else min(end_idx, T)

    periods: list[dict[str, Any]] = []
    prev_top: set[int] = set()
    prev_bottom: set[int] = set()

    t = start_idx
    while t + lag + hold < end_idx:
        s = score[t]
        ei = t + lag          # 买入日
        xi = t + lag + hold   # 卖出日
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
                # 横截面缩尾：把每期内的极端个股收益压到 1%/99% 分位
                if winsor:
                    lo_q, hi_q = np.nanpercentile(r, [winsor * 100, 100 - winsor * 100])
                    r_w = np.clip(r, lo_q, hi_q)
                else:
                    r_w = r
                order = np.argsort(sv, kind="mergesort")
                k = max(1, int(len(idx) * quantile))
                top = set(idx[order[-k:]].tolist())
                bottom = set(idx[order[:k]].tolist())
                long_ret = float(r_w[order[-k:]].mean())
                short_ret = float(r_w[order[:k]].mean())
                bench_ret = float(r_w.mean())
                long_raw = float(r[order[-k:]].mean())
                short_raw = float(r[order[:k]].mean())
                long_med = float(np.median(r[order[-k:]]))
                short_med = float(np.median(r[order[:k]]))
                bench_med = float(np.median(r))
                # 单边换手：新买 + 卖出 占组合比例
                turn_top = len(top - prev_top) / max(len(top), 1)
                turn_bot = len(bottom - prev_bottom) / max(len(bottom), 1)
                periods.append({
                    "signal_date": mat["dates"][t],
                    "entry_date": mat["dates"][ei],
                    "exit_date": mat["dates"][xi],
                    "n_valid": int(len(idx)),
                    "long": long_ret,
                    "short": short_ret,
                    "bench": bench_ret,
                    "long_raw": long_raw,
                    "short_raw": short_raw,
                    "long_med": long_med,
                    "short_med": short_med,
                    "bench_med": bench_med,
                    "turnover": 0.5 * (turn_top + turn_bot),
                    "n_top": len(top),
                })
                prev_top, prev_bottom = top, bottom
        t += hold

    if not periods:
        return {"ok": False, "reason": "无可用调仓期"}

    long_gross = np.array([p["long"] for p in periods])
    short_gross = np.array([p["short"] for p in periods])
    bench = np.array([p["bench"] for p in periods])
    long_raw = np.array([p["long_raw"] for p in periods])
    short_raw = np.array([p["short_raw"] for p in periods])
    long_med = np.array([p["long_med"] for p in periods])
    short_med = np.array([p["short_med"] for p in periods])
    bench_med = np.array([p["bench_med"] for p in periods])
    turn = np.array([p["turnover"] for p in periods])

    def stats(gross: np.ndarray, cost_mult: float) -> dict[str, float]:
        """gross 序列 → 扣成本后的统计量。cost_mult = 成本腿数（多头 1，多空 2）。"""
        net = gross - cost_mult * cost_rate * turn
        n = len(net)
        mu = float(net.mean())
        sd = float(net.std(ddof=1)) if n > 1 else 0.0
        ann = mu * (252.0 / hold)
        ann_sd = sd * math.sqrt(252.0 / hold) if sd > 0 else 0.0
        sharpe = ann / ann_sd if ann_sd > 0 else None
        eq = float(np.prod(1.0 + net) - 1.0)
        return {
            "n": n,
            "net_mean_per_period": mu,
            "net_annualized": ann,
            "net_vol_annualized": ann_sd,
            "sharpe": sharpe,
            "net_total_return": eq,
            "win_rate": float(np.mean(net > 0)),
            "std": sd,
        }

    # 多空超额（毛）→ 盈亏平衡成本
    ls_gross_excess = long_gross - short_gross
    bench_excess = long_gross - bench
    mean_turn = float(turn.mean())
    breakeven = None
    if mean_turn > 0:
        breakeven = float(ls_gross_excess.mean() / (2.0 * mean_turn))
    bench_breakeven = None
    if mean_turn > 0:
        bench_breakeven = float(bench_excess.mean() / mean_turn)

    return {
        "ok": True,
        "n_periods": len(periods),
        "hold": hold,
        "lag": lag,
        "cost_rate": cost_rate,
        "quantile": quantile,
        "mean_turnover": mean_turn,
        "long_only": stats(long_gross, 1.0),
        "long_short": stats(ls_gross_excess, 2.0),
        "benchmark": {
            "n": len(bench),
            "mean_per_period": float(bench.mean()),
            "annualized": float(bench.mean() * 252.0 / hold),
            "total_return": float(np.prod(1.0 + bench) - 1.0),
            "win_rate": float(np.mean(bench > 0)),
        },
        "gross": {
            "long_mean": float(long_gross.mean()),
            "short_mean": float(short_gross.mean()),
            "ls_excess_mean": float(ls_gross_excess.mean()),
            "long_vs_bench_mean": float(bench_excess.mean()),
        },
        "winsor": winsor,
        # 未缩尾均值：披露极值把结果拖偏了多少
        "long_only_raw": stats(long_raw, 1.0),
        "long_short_raw": stats(long_raw - short_raw, 2.0),
        # 中位数口径：抗极值，应与秩 IC 同向
        "long_only_median": stats(long_med, 1.0),
        "long_short_median": stats(long_med - short_med, 2.0),
        "benchmark_median": {
            "n": len(bench_med),
            "mean_per_period": float(bench_med.mean()),
            "annualized": float(bench_med.mean() * 252.0 / hold),
            "total_return": float(np.prod(1.0 + bench_med) - 1.0),
            "win_rate": float(np.mean(bench_med > 0)),
        },
        "breakeven_cost_roundtrip": breakeven,
        "breakeven_cost_long_only": bench_breakeven,
        "hit_rate_vs_bench": float(np.mean(bench_excess > 0)),
        "hit_rate_vs_bench_median": float(np.mean(long_med > bench_med)),
        "periods": periods,
    }


def weighted_backtest(
    mat: dict[str, Any],
    score: np.ndarray,
    *,
    hold: int = 5,
    lag: int = 1,
    cost_rate: float = COST_ROUNDTRIP,
    start_idx: int = 0,
    end_idx: int | None = None,
    long_only: bool = True,
    min_valid: int = 50,
) -> dict[str, Any]:
    """连续权重组合回测（成本后）。

    为什么不只用十分位：
        样本外实验发现，合成信号在测试段的**秩相关 IC 是正的且显著**，
        但「最高十分位 − 最低十分位」的价差是**负的** ——
        alpha 集中在分布中段，极端值那一头反而最嘈杂。
        连续权重让仓位跟着信号强度走，而不是靠尾部硬切一刀。

    多头：w_i ∝ max(score_i, 0)，归一化到和为 1。
    多空：w_i ∝ score_i，正负都做。
    换手：sum|w_new − w_old| / 2（多头口径，即单边换手）。
    成本：cost_rate × 换手（换掉的那部分要走一遍完整往返）。
    """
    close = mat["close"]
    T, N = close.shape
    end_idx = T if end_idx is None else min(end_idx, T)

    periods: list[dict[str, Any]] = []
    prev_w = np.zeros(N)

    t = start_idx
    while t + lag + hold < end_idx:
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
                if long_only:
                    raw = np.maximum(sv, 0.0)
                else:
                    raw = sv
                tot = float(raw.sum())
                if abs(tot) > 1e-12:
                    w = np.zeros(N)
                    w[idx] = raw / tot
                    port = float((w[idx] * r).sum())
                    gross_bench = float(r.mean())
                    turnover = float(np.abs(w - prev_w).sum() / 2.0)
                    periods.append({
                        "signal_date": mat["dates"][t],
                        "entry_date": mat["dates"][ei],
                        "exit_date": mat["dates"][xi],
                        "n_valid": int(len(idx)),
                        "long": port,
                        "short": 0.0,
                        "bench": gross_bench,
                        "turnover": turnover,
                        "n_top": int((w > 0).sum()),
                    })
                    prev_w = w
        t += hold

    if not periods:
        return {"ok": False, "reason": "无可用调仓期"}

    gross = np.array([p["long"] for p in periods])
    bench = np.array([p["bench"] for p in periods])
    turn = np.array([p["turnover"] for p in periods])

    def stats(net: np.ndarray) -> dict[str, float]:
        n = len(net)
        mu = float(net.mean())
        sd = float(net.std(ddof=1)) if n > 1 else 0.0
        ann = mu * (252.0 / hold)
        ann_sd = sd * math.sqrt(252.0 / hold) if sd > 0 else 0.0
        return {
            "n": n,
            "net_mean_per_period": mu,
            "net_annualized": ann,
            "net_vol_annualized": ann_sd,
            "sharpe": (ann / ann_sd) if ann_sd > 0 else None,
            "net_total_return": float(np.prod(1.0 + net) - 1.0),
            "win_rate": float(np.mean(net > 0)),
        }

    net = gross - cost_rate * turn
    bm = bench
    return {
        "ok": True,
        "n_periods": len(periods),
        "hold": hold,
        "lag": lag,
        "cost_rate": cost_rate,
        "construction": "score_weighted_long" if long_only else "score_weighted_ls",
        "mean_turnover": float(turn.mean()),
        "long_only": stats(net),
        "long_short": stats(net - bm) if not long_only else {},
        "benchmark": {
            "n": len(bm),
            "mean_per_period": float(bm.mean()),
            "annualized": float(bm.mean() * 252.0 / hold),
            "total_return": float(np.prod(1.0 + bm) - 1.0),
            "win_rate": float(np.mean(bm > 0)),
        },
        "gross": {
            "long_mean": float(gross.mean()),
            "long_vs_bench_mean": float((gross - bench).mean()),
        },
        "breakeven_cost_long_only": (
            float((gross - bench).mean() / turn.mean()) if turn.mean() > 0 else None
        ),
        "hit_rate_vs_bench": float(np.mean((gross - bench) > 0)),
        "periods": periods,
    }


def select_construction(
    mat: dict[str, Any],
    seg: tuple[int, int],
    factor_names: list[str],
    weights: list[float],
    *,
    horizons=HORIZON_CANDIDATES,
    quantiles=(0.10, 0.20, 0.33, 0.50),
    lag: int = 1,
    cost_rate: float = COST_ROUNDTRIP,
    min_valid: int = 50,
) -> dict[str, Any]:
    """在**指定区间（必须是训练段）**上挑选策略构造方式。

    一次扫的组合数 = 窗口数 × 分位档数 × 2（分组 / 连续权重），
    全部在训练段完成；选完之后测试段只跑一次，不回头调。

    判据：多头组合**成本后**的 Sharpe 最大者为主，
    净收益为正为硬门槛，换手低者优先（成本拖累小）。
    """
    lo, hi = seg
    configs: list[dict[str, Any]] = []
    for h in horizons:
        score = composite_score(mat, factor_names, weights)
        for q in quantiles:
            r = decile_backtest(
                mat, score, hold=h, lag=lag, cost_rate=cost_rate, quantile=q,
                min_valid=min_valid, start_idx=lo, end_idx=hi,
            )
            if not r.get("ok"):
                configs.append({
                    "horizon": h, "quantile": q, "construction": "decile",
                    "ok": False, "reason": r.get("reason"),
                })
                continue
            configs.append({
                "horizon": h, "quantile": q, "construction": "decile",
                "ok": True,
                "long_only_annualized": r["long_only"]["net_annualized"],
                "long_only_sharpe": r["long_only"]["sharpe"],
                "long_only_vs_bench": r["long_only"]["net_annualized"] - r["benchmark"]["annualized"],
                "long_short_annualized": r["long_short"]["net_annualized"],
                "mean_turnover": r["mean_turnover"],
                "n_periods": r["n_periods"],
            })
        for lo_only in (True, False):
            r = weighted_backtest(
                mat, score, hold=h, lag=lag, cost_rate=cost_rate,
                long_only=lo_only, min_valid=min_valid, start_idx=lo, end_idx=hi,
            )
            if not r.get("ok"):
                continue
            cf = {
                "horizon": h, "quantile": None,
                "construction": "score_weighted_long" if lo_only else "score_weighted_ls",
                "ok": True,
                "long_only_annualized": r["long_only"]["net_annualized"],
                "long_only_sharpe": r["long_only"]["sharpe"],
                "long_only_vs_bench": r["long_only"]["net_annualized"] - r["benchmark"]["annualized"],
                "long_short_annualized": (r.get("long_short") or {}).get("net_annualized"),
                "mean_turnover": r["mean_turnover"],
                "n_periods": r["n_periods"],
            }
            configs.append(cf)

    viable = [c for c in configs if c.get("ok") and (c.get("long_only_sharpe") or -9) > 0
              and (c.get("long_only_annualized") or -9) > 0]
    pool = viable or [c for c in configs if c.get("ok")]
    if not pool:
        return {"chosen": None, "configs": configs, "n_configs": len(configs)}

    def key(c: dict[str, Any]) -> tuple:
        return (
            c.get("long_only_sharpe") or -9,
            c.get("long_only_annualized") or -9,
            -(c.get("mean_turnover") or 0),
        )

    best = max(pool, key=key)
    return {
        "chosen": best,
        "configs": configs,
        "n_configs": len(configs),
        "n_viable": len(viable),
        "criterion": "训练段多头成本后 Sharpe 最大（净收益须为正），并列时换手低者优先",
    }


def diagnose_signal(
    mat: dict[str, Any],
    score: np.ndarray,
    seg: tuple[int, int],
    *,
    horizon: int,
    lag: int = 1,
    quantile: float = 0.10,
) -> dict[str, Any]:
    """诊断：合成信号的「秩相关 IC」与「尾部价差」是不是一致。

    这是整个研究里最关键的一次交叉检验：
        * IC 衡量的是**整体单调性**（Spearman，全分布）；
        * 尾部价差衡量的是**极端分位**的收益差。
    两者可以不一致 —— 如果 alpha 在分布中段，IC 为正但十分位价差为负。
    只看 IC 会高估策略可行��，只看尾部会误判因子失效。
    """
    lo, hi = seg
    fwd = make_forward(mat, horizon)
    sub = {
        "dates": mat["dates"][lo:hi],
        "codes": mat["codes"],
        "close": mat["close"][lo:hi],
        "fwd": fwd[lo:hi],
        "amount": mat["amount"][lo:hi],
        "factors": {"score": score[lo:hi]},
    }
    ics = factorlab.cross_sectional_ics(sub, "score", min_cross=30)
    ics = ics[~np.isnan(ics)]
    ic_mean, ic_t = factorlab.newey_west(ics)

    spreads: list[float] = []
    spreads_med: list[float] = []
    longs: list[float] = []
    t0 = lo
    while t0 + lag + horizon < hi:
        s = score[t0]
        ei, xi = t0 + lag, t0 + lag + horizon
        v = np.isfinite(s) & np.isfinite(mat["close"][ei]) & np.isfinite(mat["close"][xi])
        ix = np.where(v)[0]
        if len(ix) >= 100:
            sv = s[ix]
            r = mat["close"][xi, ix] / mat["close"][ei, ix] - 1.0
            ok = np.isfinite(r)
            sv, r = sv[ok], r[ok]
            o = np.argsort(sv, kind="mergesort")
            k = max(1, int(len(sv) * quantile))
            spreads.append(float(r[o[-k:]].mean() - r[o[:k]].mean()))
            # 中位数口径：极端个股收益不再主导分组统计
            spreads_med.append(float(np.median(r[o[-k:]]) - np.median(r[o[:k]])))
            longs.append(float(r[o[-k:]].mean()))
        t0 += horizon

    sp = np.asarray(spreads)
    spm = np.asarray(spreads_med)
    lg = np.asarray(longs)
    consistent = None
    if ic_mean is not None and len(sp):
        consistent = bool((ic_mean > 0) == (float(sp.mean()) > 0))
    consistent_med = None
    if ic_mean is not None and len(spm):
        consistent_med = bool((ic_mean > 0) == (float(spm.mean()) > 0))
    return {
        "ic_mean": ic_mean,
        "ic_t": ic_t,
        "ic_n": int(len(ics)),
        "tail_quantile": quantile,
        "tail_spread_mean": float(sp.mean()) if len(sp) else None,
        "tail_positive_ratio": float(np.mean(sp > 0)) if len(sp) else None,
        "tail_spread_median": float(spm.mean()) if len(spm) else None,
        "tail_median_positive_ratio": float(np.mean(spm > 0)) if len(spm) else None,
        "tail_n": int(len(sp)),
        "tail_long_mean": float(lg.mean()) if len(lg) else None,
        "n_periods": int(len(sp)),
        "ic_tail_consistent": consistent,
        "ic_tail_consistent_median": consistent_med,
        "note": (
            "IC 为正但尾部价差为负 → alpha 在分布中段，极端分位最嘈杂，"
            "不能靠十分位收割"
            if consistent is False else None
        ),
    }


def run_config(
    mat: dict[str, Any],
    score: np.ndarray,
    cfg: dict[str, Any],
    *,
    lag: int = 1,
    cost_rate: float = COST_ROUNDTRIP,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> dict[str, Any]:
    """按选定的构造方式跑一次回测。"""
    hold = int(cfg.get("horizon") or 5)
    if cfg.get("construction") == "decile":
        return decile_backtest(
            mat, score, hold=hold, lag=lag, cost_rate=cost_rate,
            quantile=float(cfg.get("quantile") or 0.10),
            start_idx=start_idx, end_idx=end_idx,
        )
    return weighted_backtest(
        mat, score, hold=hold, lag=lag, cost_rate=cost_rate,
        long_only=(cfg.get("construction") == "score_weighted_long"),
        start_idx=start_idx, end_idx=end_idx,
    )


def cost_sensitivity(
    mat: dict[str, Any],
    score: np.ndarray,
    *,
    scenarios=COST_SCENARIOS,
    **kw: Any,
) -> list[dict[str, Any]]:
    """在不同成本假设下重跑策略，看结论稳不稳。"""
    out = []
    for c in scenarios:
        r = decile_backtest(mat, score, cost_rate=c, **kw)
        if not r.get("ok"):
            out.append({"cost_rate": c, "ok": False, "reason": r.get("reason")})
            continue
        out.append({
            "cost_rate": c,
            "ok": True,
            "long_only_annualized": r["long_only"]["net_annualized"],
            "long_only_sharpe": r["long_only"]["sharpe"],
            "long_short_annualized": r["long_short"]["net_annualized"],
            "long_short_sharpe": r["long_short"]["sharpe"],
            "long_only_total_return": r["long_only"]["net_total_return"],
            "long_short_total_return": r["long_short"]["net_total_return"],
            "long_only_vs_bench_annualized": (
                r["long_only"]["net_annualized"] - r["benchmark"]["annualized"]
            ),
            "hit_rate_vs_bench": r["hit_rate_vs_bench"],
        })
    return out


# ------------------------------------------------------------------ 时间切分


def split_segments(n_dates: int, *, train_frac=0.5, val_frac=0.25) -> dict[str, tuple[int, int]]:
    """按时间切三段：训练 / 验证 / 测试。

    金融数据绝不能随机打乱，所以这里只做按时间顺序的切分。
    """
    tr_end = int(n_dates * train_frac)
    va_end = int(n_dates * (train_frac + val_frac))
    return {
        "train": (0, tr_end),
        "val": (tr_end, va_end),
        "test": (va_end, n_dates),
    }


def ic_series_on_segment(
    mat: dict[str, Any],
    factor_name: str,
    seg: tuple[int, int],
    *,
    horizon: int,
    min_cross: int = 20,
) -> np.ndarray:
    """只在一段时间区间内算该因子的 IC 序列（不重叠压缩）。"""
    lo, hi = seg
    fwd = make_forward(mat, horizon)
    sub = {
        "dates": mat["dates"][lo:hi],
        "codes": mat["codes"],
        "close": mat["close"][lo:hi],
        "fwd": fwd[lo:hi],
        "amount": mat["amount"][lo:hi],
        "factors": {k: v[lo:hi] for k, v in mat["factors"].items()},
    }
    ics = factorlab.cross_sectional_ics(sub, factor_name, min_cross=min_cross)
    return factorlab.nonoverlap_ic_series(ics, forward_days=horizon)


def discover_factors(
    mat: dict[str, Any],
    seg: tuple[int, int],
    *,
    horizon: int,
    min_cross: int = 20,
    max_factors: int | None = None,
) -> list[dict[str, Any]]:
    """在指定（通常是训练）区间内做 Bonferroni 校正后选出显著因子。

    这是「发现」环节，只能用发现期的数据。
    """
    picked: list[dict[str, Any]] = []
    for name in factorlab.FACTOR_NAMES:
        if name not in mat["factors"]:
            continue  # 该面板没有这个因子，跳过而不是崩溃
        series = ic_series_on_segment(
            mat, name, seg, horizon=horizon, min_cross=min_cross
        )
        if len(series) < 20:
            continue
        mean, t = factorlab.newey_west(series)
        if mean is None or t is None:
            continue
        sd = float(np.std(series, ddof=1))
        icir = mean / sd if sd > 0 else 0.0
        half = len(series) // 2
        m1 = float(np.mean(series[:half])) if half else 0.0
        m2 = float(np.mean(series[half:])) if half > 1 else 0.0
        stable = np.sign(m1) == np.sign(m2) and m1 != 0 and m2 != 0
        if abs(t) > 1.96 and stable:
            picked.append({
                "name": name,
                "mean": mean,
                "t": t,
                "icir": icir,
                "half1": m1,
                "half2": m2,
                "n": len(series),
                "desc": factorlab.FACTOR_DESC.get(name, name),
            })
    picked.sort(key=lambda r: -abs(r["mean"]))
    if max_factors is not None:
        picked = picked[:max_factors]
    return picked


def select_horizon(
    mat: dict[str, Any],
    seg: tuple[int, int],
    *,
    candidates=HORIZON_CANDIDATES,
    min_cross: int = 20,
) -> dict[str, Any]:
    """在训练段选前瞻窗口长度。

    判据：哪个窗口下能挖出最多的 Bonferroni 显著因子（并列时取 |IC| 合计大的）。
    这个决定只在训练段做一次，之后整个实验（发现/检验/策略/概率）全程锁定。
    """
    trials: list[dict[str, Any]] = []
    for h in candidates:
        found = discover_factors(mat, seg, horizon=h, min_cross=min_cross)
        total_ic = sum(abs(d["mean"]) for d in found)
        trials.append({
            "horizon": h,
            "n_significant": len(found),
            "sum_abs_ic": total_ic,
            "factors": [d["name"] for d in found],
        })
    best = max(trials, key=lambda t: (t["n_significant"], t["sum_abs_ic"]))
    return {"chosen": best["horizon"], "trials": trials, "candidates": list(candidates)}


def validate_factors_oos(
    mat: dict[str, Any],
    discovered: list[dict[str, Any]],
    seg: tuple[int, int],
    *,
    horizon: int,
    min_cross: int = 20,
) -> list[dict[str, Any]]:
    """把发现期选出的因子拿到检验期打分。

    这一步是「时间外」的真正检验：检验期数据在因子发现时从未被看过。
    """
    out: list[dict[str, Any]] = []
    for d in discovered:
        if d["name"] not in mat["factors"]:
            continue
        series = ic_series_on_segment(
            mat, d["name"], seg, horizon=horizon, min_cross=min_cross
        )
        if len(series) < 20:
            out.append({
                "name": d["name"], "desc": d["desc"],
                "discovered_mean": d["mean"], "discovered_t": d["t"],
                "oos_mean": None, "oos_t": None, "oos_icir": None,
                "oos_n": len(series), "oos_hit": None,
                "same_sign": None, "oos_verdict": "检验期样本不足",
            })
            continue
        mean, t = factorlab.newey_west(series)
        sd = float(np.std(series, ddof=1))
        icir = mean / sd if sd > 0 else 0.0
        same_sign = bool(np.sign(mean) == np.sign(d["mean"]))
        if t is None:
            verdict = "检验期样本不足"
        elif same_sign and abs(t) > 1.96:
            verdict = "时间外通过"
        elif same_sign:
            verdict = "方向一致但不显著"
        else:
            verdict = "方向反转（失效）"
        out.append({
            "name": d["name"],
            "desc": d["desc"],
            "discovered_mean": d["mean"],
            "discovered_t": d["t"],
            "oos_mean": mean,
            "oos_t": t,
            "oos_icir": icir,
            "oos_n": len(series),
            "oos_hit": float(np.mean(series > 0)),
            "same_sign": same_sign,
            "oos_verdict": verdict,
        })
    return out


# ------------------------------------------------------------------ 概率模型


def build_probability_dataset(
    mat: dict[str, Any],
    factor_names: list[str],
    *,
    horizon: int,
    label: str = "rel",
    max_per_date: int = 300,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """构造 (日期, 标的) 级别的概率建模数据集。

    特征：每个因子逐日横截面 z-score。
    标签：
        "abs" → 前瞻收益 > 0
        "rel" → 前瞻收益 > 当日横截面中位数（纯 alpha，不含市场 beta）
    抽样：每个交易日最多取 max_per_date 只（固定种子，可复现），
          避免百万级样本把内存打爆，且不改变分布。
    """
    if not factor_names:
        return {"ok": False, "reason": "无因子"}
    T, N = mat["close"].shape
    fwd = make_forward(mat, horizon)

    zcols = [cs_zscore(mat["factors"][n]) for n in factor_names]
    rng = np.random.default_rng(seed)

    rows_d: list[int] = []
    rows_s: list[int] = []
    rows_y: list[float] = []
    for i in range(T):
        mask = np.isfinite(fwd[i]) & np.isfinite(zcols[0][i])
        for z in zcols[1:]:
            mask &= np.isfinite(z[i])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        if len(idx) > max_per_date:
            pick = rng.choice(len(idx), size=max_per_date, replace=False)
            idx = idx[np.sort(pick)]
        r = fwd[i, idx]
        if label == "abs":
            y = (r > 0).astype(float)
        else:
            med = np.median(r)
            y = (r > med).astype(float)
        rows_d.extend([i] * len(idx))
        rows_s.extend(idx.tolist())
        rows_y.extend(y.tolist())

    if not rows_d:
        return {"ok": False, "reason": "无有效样本"}
    di = np.asarray(rows_d, dtype=int)
    si = np.asarray(rows_s, dtype=int)
    y = np.asarray(rows_y, dtype=float)
    X = np.stack([z[di, si] for z in zcols], axis=1).astype(np.float64)

    return {
        "ok": True,
        "X": X,
        "y": y,
        "date_idx": di,
        "stock_idx": si,
        "dates": mat["dates"],
        "factor_names": factor_names,
        "label": label,
        "horizon": horizon,
    }


def standardize_by_train(X: np.ndarray, n_train: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只用训练段统计量标准化特征（防止分布漂移污染，也防泄漏）。"""
    mu = X[:n_train].mean(axis=0)
    sd = X[:n_train].std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd, mu, sd


def run_probability_test(
    mat: dict[str, Any],
    factor_names: list[str],
    *,
    horizon: int,
    label: str = "rel",
    max_per_date: int = 300,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """概率模型完整时间外评估。

    训练段 → 验证段（拟合校准器）→ 测试段（一次性打分）。
    测试段的模型参数在校准之后就冻结，再也不许动。
    """
    import probmodel

    ds = build_probability_dataset(
        mat, factor_names, horizon=horizon, label=label,
        max_per_date=max_per_date, seed=seed,
    )
    if not ds.get("ok"):
        return {"ok": False, "reason": ds.get("reason")}

    X, y = ds["X"], ds["y"]
    n = len(y)
    # 行按 (日期, 标的) 排序，天然时间有序 → 可以直接按行号切时间
    tr_end = int(n * 0.5)
    va_end = int(n * 0.75)
    Xs, _, _ = standardize_by_train(X, tr_end)
    Xtr, ytr = Xs[:tr_end], y[:tr_end]
    Xva, yva = Xs[tr_end:va_end], y[tr_end:va_end]
    Xte, yte = Xs[va_end:], y[va_end:]

    model = probmodel.LogisticRegression(l2=1.0, max_iter=200)
    model.fit(Xtr, ytr)
    raw_va = model.predict_proba(Xva)
    raw_te = model.predict_proba(Xte)

    a, b = probmodel.fit_platt_scaler(raw_va, yva)
    iso = probmodel.fit_isotonic(raw_va, yva)
    platt_te = 1.0 / (1.0 + np.exp(-(a * raw_te + b)))
    iso_te = probmodel.apply_isotonic(iso, raw_te)

    def metrics(probs: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
        if len(truth) == 0:
            return {}
        eps = 1e-12
        p = np.clip(probs, eps, 1 - eps)
        base_rate = float(np.mean(truth))
        base_pred = np.full(len(truth), base_rate)
        return {
            "n": int(len(truth)),
            "brier": float(np.mean((p - truth) ** 2)),
            "brier_baseline": float(np.mean((base_pred - truth) ** 2)),
            "auc": auc(probs, truth),
            "hit_rate": float(np.mean((probs > 0.5) == (truth > 0.5))),
            "baseline_hit_rate": float(max(base_rate, 1 - base_rate)),
            "avg_prob": float(np.mean(probs)),
            "actual_rate": base_rate,
        }

    # 校准曲线（测试段，分 10 桶）
    cal = _calibration_curve(iso_te, yte, bins=10)

    return {
        "ok": True,
        "label": label,
        "horizon": horizon,
        "factor_names": factor_names,
        "n_total": n,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
        "raw": metrics(raw_te, yte),
        "platt": metrics(platt_te, yte),
        "isotonic": metrics(iso_te, yte),
        "calibration_curve": cal,
        "coef": [float(c) for c in model.coef_],
        "intercept": float(model.intercept_),
        "converged": bool(model.converged_),
    }


def _calibration_curve(probs: np.ndarray, truth: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    """校准曲线：把预测概率分桶，看每桶的实际频率。"""
    if len(probs) == 0:
        return []
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi) if i < bins - 1 else (probs >= lo) & (probs <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        out.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n": n,
            "avg_pred": float(np.mean(probs[m])),
            "actual_rate": float(np.mean(truth[m])),
        })
    return out


# ------------------------------------------------------------------ 面板加载


def _kline_fingerprint() -> str:
    """K 线缓存目录的指纹。

    没有指纹的话，换了数据源/换了复权口径之后，
    旧的 panel.npz 还会被直接读回来 —— 改了半天其实还在用旧数据。
    这里用（文件数, 总字节, 最新 mtime）做指纹，源数据一变缓存自动失效重建。
    """
    import hashlib
    import history

    files = [p for p in history.HIST_DIR.glob("*.json")]
    total = 0
    newest = 0.0
    for p in files:
        try:
            st = p.stat()
        except OSError:
            continue
        total += st.st_size
        newest = max(newest, st.st_mtime)
    h = hashlib.sha256()
    h.update(f"{len(files)}:{total}:{newest:.3f}".encode())
    return h.hexdigest()


def load_panel(
    *,
    min_bars: int = 130,
    forward_days: int = 5,
    cache: Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """构建面板；可选地缓存到 npz，避免重复 20 秒的重建。

    strict=True 时面板必须先通过 dataquality 体检，
    否则直接抛 DirtyPanelError —— 脏数据不该静默进入研究。
    """
    import history

    cache = cache or (ROOT / "data" / "panel" / "panel.npz")
    fp = _kline_fingerprint()
    panel: dict[str, Any] | None = None
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if "dates" in z and "codes" in z:
                cached_fp = str(z["fingerprint"]) if "fingerprint" in z else ""
                # 注意：cached_fp 为空（老版本缓存没有指纹字段）也必须判为失效。
                # 写成 `if cached_fp and cached_fp != fp` 会让空指纹短路放行，
                # 结果换口径之后旧面板照样被读回来 —— 改了半天还在用旧数据。
                if cached_fp != fp:
                    raise ValueError(
                        f"K 线源数据已变化或缓存无指纹（缓存 {cached_fp[:8]!r} "
                        f"≠ 当前 {fp[:8]!r}），面板缓存失效，需重建")
                dates = [str(d) for d in z["dates"]]
                codes = [str(c) for c in z["codes"]]
                factors = {k: z[f"f_{k}"] for k in factorlab.FACTOR_NAMES if f"f_{k}" in z}
                if len(factors) == len(factorlab.FACTOR_NAMES):
                    panel = {
                        "dates": dates,
                        "codes": codes,
                        "close": z["close"],
                        "fwd": z["fwd"],
                        "amount": z["amount"] if "amount" in z else np.full(z["close"].shape, np.nan),
                        "factors": factors,
                    }
        except Exception:
            panel = None  # 缓存损坏/过期就重建

    if panel is not None:
        # 守卫放在 try 外面：体检不过必须大声炸掉，不能被缓存读取的
        # except 吞掉后当成"缓存坏了"去重建 —— 那等于放脏数据进来。
        if strict:
            dataquality.assert_panel_clean(panel, context="panel 缓存")
        return panel

    hist: dict[str, list[dict]] = {}
    for fn in os.listdir(history.HIST_DIR):
        if not fn.endswith(".json"):
            continue
        sid = fn[:-5].replace("_", ".")
        try:
            rows = history.load_kline(sid)
        except Exception:
            continue
        if len(rows) > 200:
            hist[sid] = rows
    mat = factorlab.build_matrix(hist, min_bars=min_bars, forward_days=forward_days)

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dates": np.array(mat["dates"], dtype=object).astype("U16"),
            "codes": np.array(mat["codes"], dtype=object).astype("U16"),
            "close": mat["close"].astype(np.float32),
            "fwd": mat["fwd"].astype(np.float32),
            "amount": mat["amount"].astype(np.float32),
            "fingerprint": np.array(fp),
        }
        for k, v in mat["factors"].items():
            payload[f"f_{k}"] = v.astype(np.float32)
        np.savez_compressed(cache, **payload)
    except Exception:
        pass
    if strict:
        dataquality.assert_panel_clean(mat, context="新建面板")
    return mat


# ------------------------------------------------------------------ 主流程


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="样本外验证实验室")
    ap.add_argument("--hold", type=int, default=0, help="0 = 由训练段自动选择")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--cost", type=float, default=COST_ROUNDTRIP)
    ap.add_argument("--max-per-date", type=int, default=300)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--audit-only", action="store_true",
                    help="只做数据质量体检，不跑因子/回测/概率实验")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("样本外验证实验室  ——  成本后 / 时间外 / 概率校准")
    print("=" * 78)

    cache = None if args.no_cache else (ROOT / "data" / "panel" / "panel.npz")
    mat = load_panel(cache=cache)
    T, N = mat["close"].shape
    print(f"panel: {N} 只 × {T} 交易日  {mat['dates'][0]} → {mat['dates'][-1]}  ({time.time()-t0:.1f}s)")

    if args.audit_only:
        import dataquality

        rep = dataquality.audit_panel({
            "close": mat["close"], "codes": mat["codes"], "dates": mat["dates"],
        })
        out_dir = ROOT / "data" / "research"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "audit_latest.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n体检结果：{'通过' if rep.get('ok') else '未通过'}")
        for k in ("n_codes", "n_dates", "from", "to", "non_positive_rate",
                  "soft_breach_rate", "hard_breach_rate", "new_listing_exempt",
                  "thin_days", "thin_rate", "coverage_min", "coverage_median"):
            if k in rep:
                print(f"  {k}: {rep[k]}")
        for issue in rep.get("issues", []):
            print(f"  [!] {issue.get('kind')} × {issue.get('count')} —— {issue.get('hint')}")
        print(f"\n体检报告已存：{out}")
        if not rep.get("ok"):
            raise SystemExit(2)
        return

    segs = split_segments(T)
    print("\n时间切分：")
    for k, (lo, hi) in segs.items():
        print(f"  {k:<6} {mat['dates'][lo]} → {mat['dates'][hi-1]}  ({hi-lo} 天)")

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "panel": {
            "n_codes": N, "n_dates": T,
            "from": mat["dates"][0], "to": mat["dates"][-1],
        },
        "config": {
            "lag": args.lag, "cost_roundtrip": args.cost,
            "max_per_date": args.max_per_date, "seed": args.seed,
        },
        "segments": {k: [mat["dates"][lo], mat["dates"][hi - 1]] for k, (lo, hi) in segs.items()},
        "limitations": [
            "universe 为当前在市股票快照，已退市标的不在内，存在幸存者偏差，会系统性高估回测收益。",
            "回测假设能在信号次日收盘成交，未建模停牌、涨跌停与流动性冲击。",
            "IC 序列做 Newey-West(lags=5) 调整，但 5 日前瞻窗口仍有约 80% 重叠，有效样本数被高估。",
        ],
    }

    # ---- 选前瞻窗口（只在训练段决定）
    print("\n[0] 训练段选择前瞻窗口...")
    hs = select_horizon(mat, segs["train"])
    horizon = args.hold if args.hold > 0 else hs["chosen"]
    report["horizon_selection"] = hs
    report["horizon"] = horizon
    report["config"]["hold"] = horizon
    for tr in hs["trials"]:
        print(f"    {tr['horizon']:>3} 日窗口 → 显著因子 {tr['n_significant']:>2} 个  "
              f"|IC| 合计 {tr['sum_abs_ic']:.4f}")
    print(f"    选定：{horizon} 日（全程锁定）")

    # ---- 发现期选因子
    print("\n[1] 训练段发现因子（Bonferroni α=0.00313）...")
    discovered = discover_factors(mat, segs["train"], horizon=horizon)
    report["discovered"] = discovered
    print(f"    发现 {len(discovered)} 个：{[d['name'] for d in discovered]}")

    # ---- 时间外检验
    print("\n[2] 测试段时间外检验（这段数据在发现时从未被看过）...")
    oos = validate_factors_oos(mat, discovered, segs["test"], horizon=horizon)
    report["oos_validation"] = oos
    passed = [o for o in oos if o["oos_verdict"] == "时间外通过"]
    reversed_ = [o for o in oos if o["oos_verdict"] == "方向反转（失效）"]
    print(f"    {len(passed)}/{len(oos)} 时间外通过；{len(reversed_)} 个方向反转")
    for o in oos:
        print(f"      {o['name']:<15} 发现 {o['discovered_mean']:>8.4f} → 测试 {o['oos_mean']:>8.4f} "
              f"(t={o['oos_t']:.2f})  {o['oos_verdict']}")

    # ---- 全样本对照（说明为什么必须做时间外）
    print("\n[3] 全样本对照（不做时间外的话会看到什么）...")
    mat["fwd"] = make_forward(mat, horizon)
    full = factorlab.evaluate(mat, forward_days=horizon)
    report["full_sample"] = {
        "alpha": full["alpha"],
        "usable": [r["name"] for r in full["usable"]],
        "results": {r["name"]: r for r in full["ranked"]},
    }
    print(f"    全样本显著可用 {len(full['usable'])}/16 个")

    # ---- 成本后策略检验
    factor_names = [d["name"] for d in discovered]
    if factor_names:
        # 方向校正：权重符号 = 发现期 IC 的符号。IC<0 的因子必须给负权重。
        signed_weights = [1.0 if d["mean"] > 0 else -1.0 for d in discovered]
        test_lo, test_hi = segs["test"]

        print("\n[4] 合成信号诊断（测试段：IC 与尾部价差是否一致）...")
        score = composite_score(mat, factor_names, signed_weights)
        diag = diagnose_signal(mat, score, segs["test"], horizon=horizon, lag=args.lag)
        report["signal_diagnostic"] = diag
        print(f"    合成信号测试段逐日 IC: {diag['ic_mean']:+.4f} (t={diag['ic_t']:.2f}, n={diag['ic_n']})")
        print(f"    毛十分位价差: {diag['tail_spread_mean']:+.4%}  为正比例 {diag['tail_positive_ratio']:.1%}")
        if diag.get("ic_tail_consistent") is False:
            print("    ⚠ IC 为正但尾部价差为负 → alpha 在分布中段，极端分位最嘈杂，"
                  "十分位策略不可用")

        print("\n[5] 训练段选择策略构造方式（窗口 × 分位 × 连续权重）...")
        sel = select_construction(
            mat, segs["train"], factor_names, signed_weights,
            lag=args.lag, cost_rate=args.cost,
        )
        report["construction_selection"] = sel
        chosen = sel.get("chosen")
        if chosen:
            print(f"    扫了 {sel['n_configs']} 个组合（合格 {sel.get('n_viable', 0)} 个）")
            print(f"    训练段最优：窗口 {chosen['horizon']} 日 / "
                  f"{chosen['construction']} / 分位 {chosen.get('quantile')} → "
                  f"多头 Sharpe {chosen.get('long_only_sharpe'):.2f}、"
                  f"年化 {chosen.get('long_only_annualized'):+.2%}")
            print(f"    判据：{sel.get('criterion')}")
        else:
            print("    训练段没有任何可用的构造方式")

        print("\n[6] 测试段一次性评估（构造方式已在训练段锁定）...")
        strat = {}
        if chosen:
            strat = run_config(
                mat, score, chosen, lag=args.lag, cost_rate=args.cost,
                start_idx=test_lo, end_idx=test_hi,
            )
        report["strategy_test_segment"] = strat
        report["strategy_config"] = chosen
        if strat.get("ok"):
            bm = strat["benchmark"]["annualized"]
            print(f"    测试期 {strat['n_periods']} 个调仓期，平均单边换手 {strat['mean_turnover']:.1%}")
            print(f"    多头年化(净) {strat['long_only']['net_annualized']:+.2%}  "
                  f"Sharpe {strat['long_only']['sharpe']:.2f}")
            ls_ann = (strat.get("long_short") or {}).get("net_annualized")
            print(f"    多空年化(净) {ls_ann:+.2%}" if ls_ann is not None else "    多空: n/a")
            print(f"    基准年化 {bm:+.2%}   多头超额 "
                  f"{strat['long_only']['net_annualized'] - bm:+.2%}")
            be = strat.get("breakeven_cost_long_only")
            print(f"    多头打平临界成本 {be:.4%}" if be is not None else "    多头打平临界成本 n/a")

        # 固定构造（5 日十分位）作为对照：同一套因子的「传统做法」长什么样
        fixed = {"horizon": horizon, "quantile": 0.10, "construction": "decile"}
        ctrl_fixed = run_config(
            mat, score, fixed, lag=args.lag, cost_rate=args.cost,
            start_idx=test_lo, end_idx=test_hi,
        )
        report["strategy_fixed_decile"] = ctrl_fixed
        if ctrl_fixed.get("ok"):
            print(f"    [对照组·固定 5 日十分位] 多头年化 "
                  f"{ctrl_fixed['long_only']['net_annualized']:+.2%}  "
                  f"多空年化 {ctrl_fixed['long_short']['net_annualized']:+.2%}")

        # 对照组：不做方向校正（等权正权重），用来暴露符号错误
        naive_score = composite_score(mat, factor_names)
        naive = decile_backtest(
            mat, naive_score, hold=horizon, lag=args.lag, cost_rate=args.cost,
            start_idx=test_lo, end_idx=test_hi,
        )
        report["strategy_control_unsigned"] = naive
        if naive.get("ok"):
            print(f"    [对照组·未做方向校正] 多头年化 {naive['long_only']['net_annualized']:+.2%}  "
                  f"多空年化 {naive['long_short']['net_annualized']:+.2%}  "
                  "← 用来暴露方向用错时的样子")

        print("\n[7] 成本敏感性（测试期，按选定构造）...")
        if chosen:
            sens = []
            for c in COST_SCENARIOS:
                r = run_config(
                    mat, score, chosen, lag=args.lag, cost_rate=c,
                    start_idx=test_lo, end_idx=test_hi,
                )
                if not r.get("ok"):
                    sens.append({"cost_rate": c, "ok": False, "reason": r.get("reason")})
                    continue
                sens.append({
                    "cost_rate": c, "ok": True,
                    "long_only_annualized": r["long_only"]["net_annualized"],
                    "long_only_sharpe": r["long_only"]["sharpe"],
                    "long_short_annualized": (r.get("long_short") or {}).get("net_annualized"),
                    "long_short_sharpe": (r.get("long_short") or {}).get("sharpe"),
                    "long_only_total_return": r["long_only"]["net_total_return"],
                    "long_short_total_return": (r.get("long_short") or {}).get("net_total_return"),
                    "long_only_vs_bench_annualized": (
                        r["long_only"]["net_annualized"] - r["benchmark"]["annualized"]
                    ),
                    "hit_rate_vs_bench": r["hit_rate_vs_bench"],
                })
            report["cost_sensitivity"] = sens
            for s in sens:
                if s.get("ok"):
                    print(f"    成本 {s['cost_rate']:.2%} → 多头年化 {s['long_only_annualized']:+.2%}  "
                          f"多头 Sharpe {s['long_only_sharpe']:.2f}  "
                          f"多头超额 {s['long_only_vs_bench_annualized']:+.2%}")

        # ---- 概率模型
        print("\n[8] 概率模型（时间外：Brier / AUC / 校准曲线）...")
        for label in ("rel", "abs"):
            pr = run_probability_test(
                mat, factor_names, horizon=horizon, label=label,
                max_per_date=args.max_per_date, seed=args.seed,
            )
            report[f"probability_{label}"] = pr
            if not pr.get("ok"):
                print(f"    label={label}: {pr.get('reason')}")
                continue
            iso = pr["isotonic"]
            print(f"    label={label}  测试样本 {iso['n']:,}")
            print(f"      Brier {iso['brier']:.5f}  (基线 {iso['brier_baseline']:.5f})  "
                  f"AUC {iso['auc']:.4f}")
            print(f"      命中率 {iso['hit_rate']:.4f}  (基线 {iso['baseline_hit_rate']:.4f})  "
                  f"平均预测 {iso['avg_prob']:.4f}  实际 {iso['actual_rate']:.4f}")

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RESEARCH_DIR / f"oos_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = RESEARCH_DIR / "oos_latest.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已存：{latest}")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
