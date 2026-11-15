"""alpha_lab —— 尾部集中 + 低换手的策略构造实验台。

## 为什么需要这个模块

诊断（probe_ic_gap2.py，2026-11-12）用**正确符号权重**重算后得到两个硬事实：

1. **alpha 集中在尾部，不在中部。**
   用 bivariate normal 近似从 IC 反推多空价差，再和回测实测价差对比：

   | 分位 | 理论价差 | 实测价差 | 实测/理论 |
   |------|---------|---------|----------|
   | q=0.10 | +0.105% | +0.101% | **1.00x** |
   | q=0.20 | +0.189% | +0.098% | 0.52x |
   | q=0.50 | +0.431% | +0.104% | **0.24x** |

   尾部 10% 的价差与理论**完全吻合**；分位放到 50% 时实测只剩理论的 1/4。
   也就是说：持有全市场一半的股票 = 把 edge 稀释掉 4 倍。
   「买最高分 50%」（canonical 构造）等于把 90% 的仓位放在噪声上。

2. **换手是第二个瓶颈。**
   q=0.10 换手 46.6%/期，q=0.50 换手 22.8%/期 —— 尾部集中会把换手翻倍。
   在 0.30% 往返成本下，尾部策略的成本拖累会吃掉全部毛超额。

   这两个事实是**对立**的：alpha 在尾部，但尾部换手高。

## 本模块的解法

**缓冲区再平衡（buffered rebalancing）**，也称 hysteresis / 迟滞带：

- 买入门槛严格：只有进入 top-`entry_q` 才买
- 卖出门槛宽松：跌出 top-`exit_q`（exit_q > entry_q）才卖

于是一个股票在 top-entry_q 与 top-exit_q 之间时**继续持有**，不换手。
这在保留尾部 alpha 的同时大幅压低换手 —— 换手从"每次调仓重排前 k 名"
变成"只换掉掉出宽阈值的少数几只"。

配套还提供：
- `momentum_buffer_backtest`：缓冲区多空组合
- `tiered_weight_backtest`：按分数分层给权重（更接近连续权重但更可控换手）
- `realistic_cost_sensitivity`：按真实交易成本（佣金+印花税+过户费+冲击）算净超额
- 全部零 LLM、确定性、固定种子

## 纪律

- 所有选参（entry_q / exit_q / hold / 因子子集）只能在**训练段**做，
  验证段只做确认，测试段只跑一次。
- 收益统计同时给：普通均值 / 缩尾均值 / 中位数，不能用稳健统计掩盖坏数据。
- 同日个股样本高度相关，必须同时报告按交易日聚合结果。
- 换手口径必须与 ooslab.decile_backtest 逐位一致（对称口径 0.5×(多头腿+空头腿)），
  否则成本拖累不可比。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# 与 ooslab 保持一致的默认参数（跨模块引用，避免两处各写一份漂移）
DEFAULT_COST_ROUNDTRIP = 0.003
DEFAULT_LAG = 1
DEFAULT_MIN_VALID = 50
DEFAULT_WINSOR = 0.01


def _winsorize(r: np.ndarray, winsor: float) -> np.ndarray:
    """横截面缩尾：把每期内的极端个股收益压到 winsor / 1-winsor 分位。"""
    if not winsor:
        return r
    lo_q, hi_q = np.nanpercentile(r, [winsor * 100, (1 - winsor) * 100])
    return np.clip(r, lo_q, hi_q)


def _period_stats(
    gross: np.ndarray,
    cost_mult: float,
    cost_rate: float,
    turn: np.ndarray,
    hold: int,
) -> dict[str, Any]:
    """gross 序列 → 扣成本后统计量。

    年化口径：回测以「持有 hold 个交易日」为一期，
    所以 periods_per_year = 252 / hold，**不是 252**。
    （2026-11-12 踩过：直接 ×252 让年化虚高 hold 倍。）
    """
    net = gross - cost_mult * cost_rate * turn
    n = len(net)
    if n == 0:
        return {"ok": False}
    mu = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else 0.0
    ppy = 252.0 / hold
    ann = mu * ppy
    ann_sd = sd * math.sqrt(ppy) if sd > 0 else 0.0
    sharpe = ann / ann_sd if ann_sd > 0 else None
    return {
        "ok": True,
        "n": n,
        "net_mean_per_period": mu,
        "net_annualized": ann,
        "net_vol_annualized": ann_sd,
        "sharpe": sharpe,
        "net_total_return": float(np.prod(1.0 + net) - 1.0),
        "win_rate": float(np.mean(net > 0)),
        "std": sd,
    }


def _aggregate_periods(
    periods: list[dict[str, Any]],
    *,
    hold: int,
    lag: int,
    cost_rate: float,
    quantile_label: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把逐期记录聚合成标准结果字典（口径与 ooslab.decile_backtest 对齐）。"""
    if not periods:
        return {"ok": False, "reason": "无可用调仓期"}

    long_g = np.array([p["long"] for p in periods])
    short_g = np.array([p.get("short", np.nan) for p in periods])
    bench = np.array([p["bench"] for p in periods])
    long_raw = np.array([p["long_raw"] for p in periods])
    short_raw = np.array([p.get("short_raw", np.nan) for p in periods])
    long_med = np.array([p["long_med"] for p in periods])
    short_med = np.array([p.get("short_med", np.nan) for p in periods])
    bench_med = np.array([p["bench_med"] for p in periods])
    turn = np.array([p["turnover"] for p in periods])

    has_short = bool(np.isfinite(short_g).all())

    ls_gross = long_g - short_g if has_short else None
    bench_excess = long_g - bench

    mean_turn = float(turn.mean())
    breakeven_ls = None
    breakeven_long = None
    if mean_turn > 0:
        if has_short:
            breakeven_ls = float(ls_gross.mean() / (2.0 * mean_turn))
        breakeven_long = float(bench_excess.mean() / mean_turn)

    out: dict[str, Any] = {
        "ok": True,
        "n_periods": len(periods),
        "hold": hold,
        "lag": lag,
        "cost_rate": cost_rate,
        "quantile_label": quantile_label,
        "mean_turnover": mean_turn,
        "long_only": _period_stats(long_g, 1.0, cost_rate, turn, hold),
        "long_only_raw": _period_stats(long_raw, 1.0, cost_rate, turn, hold),
        "long_only_median": _period_stats(long_med, 1.0, cost_rate, turn, hold),
        "benchmark": {
            "n": len(bench),
            "mean_per_period": float(bench.mean()),
            "annualized": float(bench.mean() * (252.0 / hold)),
            "total_return": float(np.prod(1.0 + bench) - 1.0),
            "win_rate": float(np.mean(bench > 0)),
        },
        "benchmark_median": {
            "n": len(bench_med),
            "mean_per_period": float(bench_med.mean()),
            "annualized": float(bench_med.mean() * (252.0 / hold)),
            "total_return": float(np.prod(1.0 + bench_med) - 1.0),
            "win_rate": float(np.mean(bench_med > 0)),
        },
        "gross": {
            "long_mean": float(long_g.mean()),
            "bench_mean": float(bench.mean()),
            "long_vs_bench_mean": float(bench_excess.mean()),
        },
        "breakeven_cost_long_only": breakeven_long,
        "hit_rate_vs_bench": float(np.mean(bench_excess > 0)),
        "hit_rate_vs_bench_median": float(np.mean(long_med > bench_med)),
        "periods": periods,
    }
    if has_short:
        out["long_short"] = _period_stats(ls_gross, 2.0, cost_rate, turn, hold)
        out["long_short_raw"] = _period_stats(long_raw - short_raw, 2.0, cost_rate, turn, hold)
        out["long_short_median"] = _period_stats(long_med - short_med, 2.0, cost_rate, turn, hold)
        out["gross"]["short_mean"] = float(short_g.mean())
        out["gross"]["ls_excess_mean"] = float(ls_gross.mean())
        out["breakeven_cost_roundtrip"] = breakeven_ls
    if extra:
        out.update(extra)
    return out


def buffered_backtest(
    mat: dict[str, Any],
    score: np.ndarray,
    *,
    hold: int = 5,
    lag: int = DEFAULT_LAG,
    entry_q: float = 0.10,
    exit_q: float = 0.20,
    cost_rate: float = DEFAULT_COST_ROUNDTRIP,
    min_valid: int = DEFAULT_MIN_VALID,
    winsor: float = DEFAULT_WINSOR,
    start_idx: int = 0,
    end_idx: int | None = None,
    long_only: bool = True,
) -> dict[str, Any]:
    """缓冲区再平衡回测（hysteresis）。

    每期调仓时：
      1. 按 score 排序，取前 `entry_q` 作为**目标买入集合**
      2. 已在持仓且仍处于前 `exit_q` 的股票 → **继续持有**（不卖）
      3. 空出的仓位用目标集合里还没持有的、分数最高的补上
      4. 持仓里跌出前 `exit_q` 的 → 卖出

    效果：换手从「每期重排」降到「只换掉掉出宽阈值的少数几只」，
    同时保留尾部 alpha。

    `exit_q` 越大 → 换手越低，但持有的股票质量越松（拖累 alpha）。
    两者之间存在最优点，**必须在训练段上选**。

    参数：
      entry_q   买入门槛分位（越小越集中）
      exit_q    卖出门槛分位（必须 >= entry_q，否则退化成普通十分位）
      long_only True = 只做多头（对基准）；False = 多空
    """
    close = mat["close"]
    T, N = close.shape
    end_idx = T if end_idx is None else min(end_idx, T)
    if exit_q < entry_q:
        return {"ok": False, "reason": f"exit_q({exit_q}) < entry_q({entry_q})"}

    periods: list[dict[str, Any]] = []
    prev_hold: set[int] = set()      # 多头持仓
    prev_short: set[int] = set()     # 空头持仓

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
        if int(valid.sum()) >= min_valid:
            idx = np.where(valid)[0]
            sv = s[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = close[xi, idx] / close[ei, idx] - 1.0
            r = np.where(np.isfinite(r), r, np.nan)
            good = np.isfinite(r)
            idx, sv, r = idx[good], sv[good], r[good]
            if len(idx) >= min_valid:
                r_w = _winsorize(r, winsor)
                order = np.argsort(sv, kind="mergesort")
                n_valid = len(idx)

                k_entry = max(1, int(n_valid * entry_q))
                k_exit = max(k_entry, int(n_valid * exit_q))

                # 排名表：rank[i] = 该股在当期横截面的名次（0 = 最高分）
                rank_of = np.empty(n_valid, dtype=int)
                rank_of[order] = np.arange(n_valid)

                pos_in_rank = {int(idx[j]): int(rank_of[j]) for j in range(n_valid)}

                # --- 目标集合 ---
                target_long = set(int(x) for x in idx[order[-k_entry:]])
                # 空头：分数最低的一端（对称处理）
                target_short = set(int(x) for x in idx[order[:k_entry]])

                # --- 多头：缓冲区逻辑 ---
                keep_long = {c for c in prev_hold
                             if c in pos_in_rank and pos_in_rank[c] < k_exit}
                held = set(keep_long)
                need = k_entry - len(held)
                if need > 0:
                    # 从目标集合里分数最高的、还没持有的补进来
                    cands = [c for c in idx[order[-k_entry:]] if int(c) not in held]
                    for c in cands:
                        if need <= 0:
                            break
                        held.add(int(c))
                        need -= 1
                # 宽阈值内但不在目标集合的，**不主动买**（保持不换手）

                # --- 空头：对称 ---
                keep_short = {c for c in prev_short
                              if c in pos_in_rank and pos_in_rank[c] >= n_valid - k_exit}
                short_held = set(keep_short)
                need_s = k_entry - len(short_held)
                if need_s > 0:
                    cands = [c for c in idx[order[:k_entry]] if int(c) not in short_held]
                    for c in cands:
                        if need_s <= 0:
                            break
                        short_held.add(int(c))
                        need_s -= 1

                if held and (long_only or short_held):
                    held_idx = np.array(sorted(held))
                    short_idx = np.array(sorted(short_held))
                    pos = {int(v): j for j, v in enumerate(idx)}

                    def grp(subset: set[int], arr: np.ndarray) -> np.ndarray:
                        if not subset:
                            return np.array([])
                        return arr[np.array([pos[c] for c in subset])]

                    lr = grp(held, r)
                    lrw = grp(held, r_w)
                    lr_med = grp(held, r)
                    long_ret = float(lrw.mean())
                    long_raw = float(lr.mean())
                    long_med = float(np.median(lr_med))
                    bench_ret = float(r_w.mean())
                    bench_med = float(np.median(r))

                    turn_top = len(held - prev_hold) / max(len(held), 1)
                    rec: dict[str, Any] = {
                        "signal_date": mat["dates"][t],
                        "entry_date": mat["dates"][ei],
                        "exit_date": mat["dates"][xi],
                        "n_valid": int(n_valid),
                        "n_top": len(held),
                        "long": long_ret,
                        "long_raw": long_raw,
                        "long_med": long_med,
                        "bench": bench_ret,
                        "bench_med": bench_med,
                        "turnover": turn_top,
                    }
                    if not long_only:
                        sr_ = grp(short_held, r)
                        srw = grp(short_held, r_w)
                        turn_bot = len(short_held - prev_short) / max(len(short_held), 1)
                        rec["short"] = float(srw.mean())
                        rec["short_raw"] = float(sr_.mean())
                        rec["short_med"] = float(np.median(sr_))
                        # 与 ooslab 对齐：对称口径 0.5×(多头腿 + 空头腿)
                        rec["turnover"] = 0.5 * (turn_top + turn_bot)
                    periods.append(rec)
                    prev_hold = held
                    prev_short = short_held if not long_only else set()
        t += hold

    return _aggregate_periods(
        periods, hold=hold, lag=lag, cost_rate=cost_rate,
        quantile_label=f"entry_q={entry_q:.2f}/exit_q={exit_q:.2f}",
        extra={
            "entry_q": entry_q,
            "exit_q": exit_q,
            "long_only_mode": long_only,
            "winsor": winsor,
        },
    )


def tiered_weight_backtest(
    mat: dict[str, Any],
    score: np.ndarray,
    *,
    hold: int = 5,
    lag: int = DEFAULT_LAG,
    top_q: float = 0.20,
    cost_rate: float = DEFAULT_COST_ROUNDTRIP,
    min_valid: int = DEFAULT_MIN_VALID,
    winsor: float = DEFAULT_WINSOR,
    start_idx: int = 0,
    end_idx: int | None = None,
    weight_power: float = 2.0,
) -> dict[str, Any]:
    """分层权重回测：只买 top_q，但权重随分数强度非线性递减。

    动机：诊断显示 alpha 集中在尾部。top_q 里分数最高的那一小撮应该配更重的仓，
    这样：
      - 组合收益更贴近「尾部 alpha」而不是「尾部 + 中部噪声」
      - 同时因为权重平滑，单只股票的小幅排名变化不再触发整仓调换 → 换手下降

    权重：w_i ∝ (rank_score)^weight_power，其中 rank_score 是 top_q 内的分位分数
    （最高分 = 1，最低分 = 0）。归一化到和为 1。
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
        if int(valid.sum()) >= min_valid:
            idx = np.where(valid)[0]
            sv = s[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = close[xi, idx] / close[ei, idx] - 1.0
            r = np.where(np.isfinite(r), r, np.nan)
            good = np.isfinite(r)
            idx, sv, r = idx[good], sv[good], r[good]
            if len(idx) >= min_valid:
                r_w = _winsorize(r, winsor)
                order = np.argsort(sv, kind="mergesort")
                k = max(1, int(len(idx) * top_q))
                sel = order[-k:]
                sel_scores = sv[sel]
                lo, hi = sel_scores.min(), sel_scores.max()
                span = (hi - lo) if hi > lo else 1.0
                frac = (sel_scores - lo) / span          # [0,1]，最高分=1
                w = frac ** weight_power
                w = w / w.sum()
                ret_w = float((w * r_w[sel]).sum())
                ret_raw = float((w * r[sel]).sum())
                med_w = float((w * np.median(r[sel])).sum())

                wfull = np.zeros(N)
                wfull[sel] = w
                turnover = float(np.abs(wfull - prev_w).sum() / 2.0)

                periods.append({
                    "signal_date": mat["dates"][t],
                    "entry_date": mat["dates"][ei],
                    "exit_date": mat["dates"][xi],
                    "n_valid": int(len(idx)),
                    "n_top": int(k),
                    "long": ret_w,
                    "long_raw": ret_raw,
                    "long_med": med_w,
                    "bench": float(r_w.mean()),
                    "bench_med": float(np.median(r)),
                    "turnover": turnover,
                })
                prev_w = wfull
        t += hold

    return _aggregate_periods(
        periods, hold=hold, lag=lag, cost_rate=cost_rate,
        quantile_label=f"tiered_top_q={top_q:.2f}/power={weight_power}",
        extra={"top_q": top_q, "weight_power": weight_power, "winsor": winsor},
    )


def day_level_aggregate(
    result: dict[str, Any],
    *,
    hold: int,
) -> dict[str, Any]:
    """按调仓期聚合的收益序列 → 换算成「每交易日」口径。

    同日个股样本高度相关（一天 1300 只股票共享同一市场冲击），
    所以 t 检验必须按**调仓期**而不是按个股样本做。
    这个函数给出按期聚合后的胜率与均值，供报告使用。
    """
    if not result.get("ok"):
        return {"ok": False}
    per = result["long_only"]
    return {
        "ok": True,
        "n_periods": per["n"],
        "periods_per_year": 252.0 / hold,
        "net_mean_per_period": per["net_mean_per_period"],
        "net_annualized": per["net_annualized"],
        "win_rate_per_period": per["win_rate"],
        "sharpe": per["sharpe"],
        "note": "按调仓期聚合，非按个股样本；t 检验须用此口径",
    }


def realistic_cost_breakdown(
    *,
    commission_rate: float = 0.00025,
    stamp_duty: float = 0.0005,
    transfer_fee: float = 0.00001,
    slippage_per_side: float = 0.0005,
) -> dict[str, Any]:
    """真实 A 股单次往返成本拆解（佣金+印花税+过户费+冲击）。

    参数含义：
      commission_rate  单边佣金费率（万2.5 = 0.025% = 0.00025）
      stamp_duty       卖出单边印花税（0.05% = 0.0005，2023-08-28 起减半）
      transfer_fee     单边过户费（0.001% = 0.00001）
      slippage_per_side 单边滑点/冲击（0.05% = 0.0005）

    返回单边与往返成本，供成本敏感性分析使用。
    注意：这些默认值是**行业通用量级**，正式研究应以
    data/research/transaction_cost_basis.md 的核实值为准。
    """
    one_way = commission_rate + transfer_fee + slippage_per_side
    roundtrip = 2.0 * one_way + stamp_duty
    return {
        "one_way": one_way,
        "roundtrip": roundtrip,
        "components": {
            "commission_roundtrip": 2.0 * commission_rate,
            "stamp_duty_sell_only": stamp_duty,
            "transfer_fee_roundtrip": 2.0 * transfer_fee,
            "slippage_roundtrip": 2.0 * slippage_per_side,
        },
    }


def cost_sensitivity(
    result: dict[str, Any],
    *,
    hold: int,
    cost_rates: tuple[float, ...] = (0.0005, 0.0010, 0.0015, 0.0020, 0.0030),
) -> list[dict[str, Any]]:
    """按不同往返成本率重算多头净超额，看方法对成本假设有多敏感。"""
    if not result.get("ok"):
        return []
    periods = result["periods"]
    long_g = np.array([p["long"] for p in periods])
    turn = np.array([p["turnover"] for p in periods])
    bench = np.array([p["bench"] for p in periods])
    out = []
    for cr in cost_rates:
        net = long_g - cr * turn
        mu = float(net.mean())
        sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        ppy = 252.0 / hold
        ann = mu * ppy
        ann_sd = sd * math.sqrt(ppy) if sd > 0 else 0.0
        excess = float((long_g - bench).mean() * ppy)
        out.append({
            "cost_rate": cr,
            "net_annualized": ann,
            "net_sharpe": (ann / ann_sd) if ann_sd > 0 else None,
            "excess_vs_bench_annualized": excess,
            "win_rate": float(np.mean(net > 0)),
        })
    return out
