"""成交真实性检验：把「按收盘价成交」换成「按真实可成交价成交」，看超额还剩多少。

为什么必须做这个检验：
    整个研究有一个最乐观的假设 —— 回测按 **close[t+lag]** 成交，
    也就是"信号次日收盘价全部买到"。现实里：

      1. 你在收盘价形成之前不知道最终收盘价，却假设自己按收盘价成交；
      2. 涨停板你根本买不到（不是你想买就能买）；
      3. 一次性买 600 多只股票不可能都在同一个价格上成交。

    如果换成更现实的成交价之后超额就消失，那这套方法的"历史有效"只是
    成交假设的产物，不是信号本身的本事。

检验设计（一个函数，两个价格矩阵）：
    run_periodic(price_matrix) —— 与 ooslab.decile_backtest 完全同构，
    唯一区别是入场/出场价来自传入的矩阵。

    用 **close 矩阵**跑一次，必须逐位复现 canonical OOS 的数字
    （这是新代码忠实性的证明）；
    再用 **open 矩阵**跑一次，得到现实成交口径下的结果。

    如果 close 那一次对不上 canonical，说明这段代码写错了，
    那么 open 那次的结论也不可信 —— 所以这一步是硬闸门，不是可选检查。

成交规则（三条，全部可在账本里追溯）：
    R1 理论成交：入场 close[t+lag]，出场 close[t+lag+hold]（= canonical）
    R2 开盘成交：入场 open[t+lag]，出场 open[t+lag+hold]
       —— 信号在 t 日收盘后算出，t+1 日开盘才能动手，且开盘价含隔夜跳空
    R3 涨停不可买：入场日若该股收盘涨幅已达涨停板，视为买不到，
       从多头篮子中剔除并按剩余股票重新等权。
       涨停判定按板块分：主板 ±10%，创业板(300/301)/科创板(688/689) ±20%。

零 LLM，纯确定性。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import ooslab
import panel_open
from frozen_spec import load_spec

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "research" / "execution_realism.json"

# 涨停判定阈值（按板块）
LIMIT_MAIN = 0.098      # 主板 ±10%，留 0.2% 容差（价格四舍五入）
LIMIT_GROWTH = 0.198    # 创业板/科创板 ±20%
LIMIT_ST = 0.048        # ST/退市股 ±5%
BLOCK_RATIO = 0.5       # 一字板判定：最高=最低=收盘


def _load_st_flags(codes: list[str]) -> np.ndarray:
    """标记 ST / *ST / 退市股 —— 这些股票的涨跌幅限制是 ±5%，不是 ±10%/±20%。

    漏判的后果是方向性的：ST 股按 10% 阈值判，会把大量 5% 封板股
    误判成「没涨停」→ 该剔除的没剔除 → R3 结果偏乐观。
    """
    uni_path = ROOT / "data" / "universe.json"
    flags = np.zeros(len(codes), dtype=bool)
    if not uni_path.exists():
        return flags
    try:
        uni = json.loads(uni_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return flags
    # universe 用 secid（1.688825），面板用 code（0.000001）—— 统一成后缀匹配
    name_by_suffix: dict[str, str] = {}
    for secid, name in uni.items():
        if not isinstance(name, str):
            continue
        suffix = secid.split(".")[-1] if "." in secid else secid
        name_by_suffix[suffix] = name
    for i, c in enumerate(codes):
        suffix = c.split(".")[-1] if "." in c else c
        name = name_by_suffix.get(suffix, "")
        up = name.upper()
        if "ST" in up or "退" in name:
            flags[i] = True
    return flags


def _limit_rates(codes: list[str]) -> np.ndarray:
    """每只股票的**名义**涨跌幅限制（不是带容差的阈值）。

    主板 ±10%，创业板(300/301)/科创板(688/689) ±20%，ST/退市股 ±5%。

    ⚠️ 代码形如 '0.300001' / '1.688825'：**前面是市场前缀，后面才是 6 位代码**
    （与 dataquality.limit_of 同一约定）。所以板块必须从**后 6 位数字**判断。
    写成 `c.split(".")[0]` 会取到市场前缀（'0'/'1'），创业板/科创板永远
    判不出来 —— 后果是 20% 的票按 10% 阈值判涨停，大量本来买得到的票
    被当成"涨停买不到"剔掉，R3 结果被系统性做偏。
    """
    out = np.full(len(codes), 0.10, dtype=float)
    for i, c in enumerate(codes):
        digits = "".join(ch for ch in str(c) if ch.isdigit())[-6:]
        if digits.startswith(("300", "301", "688", "689")):
            out[i] = 0.20
    st = _load_st_flags(codes)
    out[st] = 0.05
    return out


def _limit_up_mask(
    close: np.ndarray,
    open_: np.ndarray,
    codes: list[str],
    idx: int,
    prev_idx: int,
) -> np.ndarray:
    """第 idx 日的涨停标记（True = 涨停买不到）。

    ⚠️ 口径铁律一：只能用**开盘那一刻已经知道的信息**判断，即
    `open[idx]` 相对 `close[idx-1]`（昨收）的涨幅。

    为什么不能用当日收盘涨幅判断：
        你是按**开盘价**买入的。收盘价在成交之后才形成，
        用收盘涨幅筛掉"涨停股"= 用未来信息决定今天买不买 = 前视偏差。
        那样算出来的超额会虚假地好看（因为偷偷避开了后面才知道的坏结果）。

    正确做法：开盘价已经封在涨停板（open 相对昨收涨幅 ≥ 阈值），
    说明**开盘就没法买到** —— 这才是真实存在的成交约束。

    ⚠️ 口径铁律二：用**四舍五入后的实际涨停价**判定，不用「涨幅 ≥ 10%」。
        A股涨停价 = 昨收 ×(1+幅度) 四舍五入到分。昨收 3.03 元时，
        涨停价 3.33 元，实际涨幅 9.90% —— 按「≥9.8%」判会误杀正常股票，
        按「≥10%」判会漏掉真涨停。必须先算涨停价再比。

    ⚠️ 口径铁律三：拿不到昨收（停牌/新上市）时**不判涨停**。
        未知 ≠ 涨停。宁可少剔几只，也不能凭空剔掉正常能买的股票。
    """
    prev = close[prev_idx]
    cur_open = open_[idx]
    rates = _limit_rates(codes)
    ok = np.isfinite(prev) & (prev > 0) & np.isfinite(cur_open) & (cur_open > 0)
    # 实际涨停价（四舍五入到分）；容差 1e-9 吸收浮点误差
    limit_price = np.round(prev * (1.0 + rates), 2)
    return ok & (cur_open >= limit_price - 1e-9)


def run_periodic(
    mat: dict[str, Any],
    score: np.ndarray,
    entry: np.ndarray,
    exit_: np.ndarray,
    *,
    hold: int,
    lag: int,
    cost_rate: float,
    quantile: float,
    min_valid: int,
    winsor: float,
    start_idx: int,
    end_idx: int | None,
    block_limit_up: bool = False,
    prev_close: np.ndarray | None = None,
    open_ref: np.ndarray | None = None,
) -> dict[str, Any]:
    """分组回测，入场/出场价由参数决定。

    与 ooslab.decile_backtest 同构，差别只有两点：
      - 价格矩阵是入参（可以换成 open）
      - 可选剔除涨停股（R3）
    """
    dates = mat["dates"]
    codes = mat["codes"]
    close = mat["close"]
    T, N = close.shape
    end_idx = T if end_idx is None else min(end_idx, T)

    periods: list[dict[str, Any]] = []
    prev_top: set[int] = set()
    prev_bottom: set[int] = set()

    t = start_idx
    while t + lag + hold < end_idx:
        s = score[t]
        ei = t + lag
        xi = t + lag + hold
        valid = (
            np.isfinite(s)
            & np.isfinite(entry[ei]) & np.isfinite(exit_[xi])
            & (entry[ei] > 0)
        )
        n_valid = int(valid.sum())
        if n_valid >= min_valid:
            idx = np.where(valid)[0]
            sv = s[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = exit_[xi, idx] / entry[ei, idx] - 1.0
            r = np.where(np.isfinite(r), r, np.nan)
            good = np.isfinite(r)
            idx, sv, r = idx[good], sv[good], r[good]

            blocked = 0
            if block_limit_up and prev_close is not None and open_ref is not None:
                lu = _limit_up_mask(close, open_ref, codes, ei, ei - 1)
                blocked_mask = lu[idx]
                blocked = int(blocked_mask.sum())
                if blocked:
                    keep = ~blocked_mask
                    idx, sv, r = idx[keep], sv[keep], r[keep]
                    if len(idx) < min_valid:
                        t += hold
                        continue

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
                long_ret = float(r_w[order[-k:]].mean())
                bench_ret = float(r_w.mean())
                long_raw = float(r[order[-k:]].mean())
                long_med = float(np.median(r[order[-k:]]))
                bench_med = float(np.median(r))
                # 换手口径必须与 ooslab.decile_backtest 逐位一致：
                # 单边换手 = 0.5 × (多头腿换手 + 空头腿换手)。
                # 多头策略只收一条腿的成本（cost_mult=1），但换手仍是
                # 对称口径 —— 这里若只算多头腿，忠实性闸门会直接报错。
                turn_top = len(top - prev_top) / max(len(top), 1)
                turn_bot = len(bottom - prev_bottom) / max(len(bottom), 1)
                periods.append({
                    "signal_date": dates[t],
                    "entry_date": dates[ei],
                    "exit_date": dates[xi],
                    "n_valid": int(len(idx)),
                    "n_top": len(top),
                    "blocked_limit_up": blocked,
                    "long": long_ret,
                    "bench": bench_ret,
                    "long_raw": long_raw,
                    "long_med": long_med,
                    "bench_med": bench_med,
                    "turnover": 0.5 * (turn_top + turn_bot),
                })
                prev_top, prev_bottom = top, bottom
        t += hold

    if not periods:
        return {"ok": False, "reason": "无可用调仓期"}

    long_g = np.array([p["long"] for p in periods])
    bench = np.array([p["bench"] for p in periods])
    long_raw = np.array([p["long_raw"] for p in periods])
    long_med = np.array([p["long_med"] for p in periods])
    bench_med = np.array([p["bench_med"] for p in periods])
    turn = np.array([p["turnover"] for p in periods])

    net = long_g - cost_rate * turn
    n = len(net)
    mu = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else 0.0
    ann = mu * (252.0 / hold)
    ann_sd = sd * math.sqrt(252.0 / hold) if sd > 0 else 0.0
    eq = float(np.prod(1.0 + net) - 1.0)

    bench_mu = float(bench.mean())
    bench_ann = bench_mu * (252.0 / hold)

    raw_net = long_raw - cost_rate * turn
    med_net = long_med - cost_rate * turn

    return {
        "ok": True,
        "n_periods": n,
        "hold": hold,
        "lag": lag,
        "cost_rate": cost_rate,
        "quantile": quantile,
        "mean_turnover": float(turn.mean()),
        "long_only": {
            "net_mean_per_period": mu,
            "net_annualized": ann,
            "net_vol_annualized": ann_sd,
            "sharpe": (ann / ann_sd) if ann_sd > 0 else None,
            "net_total_return": eq,
            "win_rate": float(np.mean(net > 0)),
            "std": sd,
        },
        "benchmark": {
            "mean_per_period": bench_mu,
            "annualized": bench_ann,
            "total_return": float(np.prod(1.0 + bench) - 1.0),
        },
        "gross": {
            "long_mean": float(long_g.mean()),
            "long_vs_bench_mean": float((long_g - bench).mean()),
            "cost_drag_mean": float((cost_rate * turn).mean()),
        },
        "long_only_raw": {
            "net_mean_per_period": float(raw_net.mean()),
            "net_annualized": float(raw_net.mean()) * (252.0 / hold),
            "net_total_return": float(np.prod(1.0 + raw_net) - 1.0),
        },
        "long_only_median": {
            "net_mean_per_period": float(med_net.mean()),
            "net_annualized": float(med_net.mean()) * (252.0 / hold),
            "net_total_return": float(np.prod(1.0 + med_net) - 1.0),
        },
        "benchmark_median": {
            "mean_per_period": float(bench_med.mean()),
            "annualized": float(bench_med.mean()) * (252.0 / hold),
            "total_return": float(np.prod(1.0 + bench_med) - 1.0),
        },
        "excess_annualized": ann - bench_ann,
        "hit_rate_vs_bench": float(np.mean(long_g - bench > 0)),
        "blocked_limit_up_total": int(sum(p["blocked_limit_up"] for p in periods)),
        "periods": periods,
    }


# 超额判零阈值：年化超额绝对值小于它就当作 0，避免拿噪声去除噪声
_EXCESS_EPS = 1e-12


def _excess_sign(x: float) -> int:
    """超额符号：+1 正 / 0 零 / -1 负。"""
    if x > _EXCESS_EPS:
        return 1
    if x < -_EXCESS_EPS:
        return -1
    return 0


def _pct(x: float) -> str:
    """年化超额的百分比写法，带符号（用于结论文案）。"""
    return f"{x:+.2%}"


def build_verdict(
    *,
    close_excess: float,
    open_excess: float,
) -> dict[str, Any]:
    """按 close / open 两个口径的超额，生成 verdict 的数值与结论文案。

    为什么不能只看 open 口径的符号就下结论：
        这段文案最早是按「close 口径超额为正」写的，所以 open 为负就断言
        「原来的正超额是成交假设的产物」。但实际数据里 close 口径本身就是负的
        （-2.03%），open 更负（-3.63%）—— 这不是「由正转负」，而是「本来就是负」。
        把「本来就是负」说成「成交假设的产物」，是把结论归错了因：
        真正的事实是边际在任何成交口径下都不成立。

    保留比例（excess_retained）只在两个口径都为正时才有意义：
        负 ÷ 负 会得到大于 1 的假比率（-0.036 / -0.020 = 1.79），读成
        「超额保留 179%」纯属胡说；符号不一致时分母分子不同号，比率更没法解释。
        所以这两种情况一律置 None，并在文案里写明「保留比例不可比」。
    """
    s1 = _excess_sign(close_excess)
    s2 = _excess_sign(open_excess)

    # 只有同为正才给保留比例；分母接近 0 一律不给
    retained: float | None = None
    if s1 > 0 and s2 > 0 and abs(close_excess) > _EXCESS_EPS:
        retained = open_excess / close_excess

    if s1 > 0 and s2 > 0:
        if retained is not None and retained < 1.0:
            conclusion = (
                f"换成开盘成交后超额仍为正（{_pct(close_excess)} → {_pct(open_excess)}）"
                f"—— 信号本身站得住，但幅度缩水，保留 {retained:.0%}"
            )
        else:
            conclusion = (
                f"换成开盘成交后超额仍为正（{_pct(close_excess)} → {_pct(open_excess)}）"
                "—— 信号本身站得住，幅度没有缩水"
            )
    elif s1 > 0 and s2 < 0:
        conclusion = (
            f"换成开盘成交后超额转负（{_pct(close_excess)} → {_pct(open_excess)}）"
            "—— 原来的正超额是成交假设的产物"
        )
    elif s1 < 0 and s2 < 0:
        conclusion = (
            f"两个口径下超额都为负（收盘 {_pct(close_excess)} → 开盘 {_pct(open_excess)}）"
            "—— 边际不成立；两个口径符号一致，保留比例不可比"
        )
    elif s1 < 0 and s2 > 0:
        conclusion = (
            f"收盘口径下超额为负（{_pct(close_excess)}），换成开盘成交后反而转正"
            f"（{_pct(open_excess)}）—— 罕见，需核对两套面板样本是否一致，保留比例不可比"
        )
    elif s1 == 0 and s2 == 0:
        conclusion = "两个口径下超额都是 0 —— 没有边际可谈"
    elif s1 == 0 and s2 < 0:
        conclusion = (
            f"收盘口径超额为 0、开盘口径转负（{_pct(open_excess)}）"
            "—— 边际不成立，保留比例不可比"
        )
    elif s1 == 0 and s2 > 0:
        conclusion = (
            f"收盘口径超额为 0、开盘口径转正（{_pct(open_excess)}）"
            "—— 罕见，需核对两套面板样本是否一致，保留比例不可比"
        )
    elif s1 > 0 and s2 == 0:
        conclusion = (
            f"换成开盘成交后超额归零（{_pct(close_excess)} → 0.00%）"
            "—— 边际不成立，保留比例不可比"
        )
    else:  # s1 < 0 and s2 == 0
        conclusion = (
            f"收盘口径下超额为负（{_pct(close_excess)}）、换成开盘成交后归零"
            "—— 边际不成立，保留比例不可比"
        )

    return {
        "close_excess": close_excess,
        "open_excess": open_excess,
        "close_excess_positive": s1 > 0,
        "open_excess_positive": s2 > 0,
        "excess_retained": retained,
        "conclusion": conclusion,
    }


def build_r3_conclusion(
    *,
    close_excess: float,
    open_excess: float,
    r3_excess: float,
) -> str:
    """R3（开盘成交 + 涨停买不到）的结论文案。

    同样不能无条件说「超额转负」：如果 open 口径本身就是负的，R3 不是
    「由正转负」，而是「本来就负、剔除涨停后更负」。两种说法归因完全不同。
    """
    s_open = _excess_sign(open_excess)
    s_r3 = _excess_sign(r3_excess)

    if s_r3 < 0 and s_open < 0:
        return (
            f"再考虑涨停买不到，超额仍为负并进一步恶化"
            f"（开盘 {_pct(open_excess)} → {_pct(r3_excess)}）—— 实盘不可行"
        )
    if s_r3 < 0:
        return (
            f"再考虑涨停买不到，超额转负（{_pct(open_excess)} → {_pct(r3_excess)}）"
            "—— 实盘不可行"
        )
    if s_open < 0:
        return (
            f"再考虑涨停买不到，超额转正（开盘 {_pct(open_excess)} → {_pct(r3_excess)}）"
            "—— 罕见，需核对剔除口径"
        )
    if r3_excess < close_excess:
        return (
            f"再考虑涨停买不到，超额仍为正（{_pct(r3_excess)}），"
            f"但已低于理论成交口径（{_pct(close_excess)}）"
        )
    return (
        f"再考虑涨停买不到，超额仍为正（{_pct(r3_excess)}），"
        f"不低于理论成交口径（{_pct(close_excess)}）"
    )


def _match_canonical(
    res: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    """close 口径必须逐位复现 canonical —— 忠实性硬闸门。"""
    canon = raw.get("strategy_test_segment") or {}
    if not canon.get("ok"):
        return {"ok": False, "reason": "canonical 策略段不可用，无法校验"}
    checks = {
        "n_periods": (res["n_periods"], canon.get("n_periods")),
        "net_annualized": (res["long_only"]["net_annualized"],
                           (canon.get("long_only") or {}).get("net_annualized")),
        "sharpe": (res["long_only"]["sharpe"],
                   (canon.get("long_only") or {}).get("sharpe")),
        "mean_turnover": (res["mean_turnover"], canon.get("mean_turnover")),
        "benchmark_annualized": (res["benchmark"]["annualized"],
                                 (canon.get("benchmark") or {}).get("annualized")),
    }
    out: dict[str, Any] = {"ok": True, "checks": {}, "worst_rel_err": 0.0}
    for k, (mine, theirs) in checks.items():
        if mine is None or theirs is None:
            out["checks"][k] = {"mine": mine, "canonical": theirs, "ok": False}
            out["ok"] = False
            continue
        if isinstance(mine, (int,)) and isinstance(theirs, (int,)):
            err = 0.0 if mine == theirs else abs(mine - theirs) / max(abs(theirs), 1e-12)
        else:
            err = abs(float(mine) - float(theirs)) / max(abs(float(theirs)), 1e-12)
        out["checks"][k] = {
            "mine": mine, "canonical": theirs,
            "rel_err": err, "ok": err <= 1e-9,
        }
        out["worst_rel_err"] = max(out["worst_rel_err"], err)
        if err > 1e-9:
            out["ok"] = False
    return out


def run(*, verbose: bool = True) -> dict[str, Any]:
    """跑完整检验：忠实性闸门 + 三种成交口径。"""
    spec = load_spec()
    raw = json.loads(Path(spec["source"]).read_text(encoding="utf-8"))

    mat = ooslab.load_panel()
    segs = ooslab.split_segments(len(mat["dates"]))
    test_lo, test_hi = segs["test"]

    score = ooslab.composite_score(
        mat, spec["factor_names"], spec["signed_weights"]
    )

    kw = dict(
        hold=spec["hold"], lag=spec["lag"], cost_rate=spec["cost_roundtrip"],
        quantile=spec["quantile"], min_valid=spec["min_valid"],
        winsor=spec["winsor"], start_idx=test_lo, end_idx=test_hi,
    )

    # ---- R1 理论成交（close）：忠实性闸门
    r1 = run_periodic(mat, score, mat["close"], mat["close"], **kw)
    if not r1.get("ok"):
        raise RuntimeError(f"R1 理论成交跑不出来: {r1.get('reason')}")
    fidelity = _match_canonical(r1, raw)

    # ---- R2 开盘成交（open）
    op = panel_open.load_open_panel(panel=mat, verbose=verbose)
    r2 = run_periodic(mat, score, op["open"], op["open"], **kw)
    r2_ok = bool(r2.get("ok"))

    # ---- R3 开盘成交 + 开盘封涨停不可买
    r3 = run_periodic(
        mat, score, op["open"], op["open"],
        block_limit_up=True, prev_close=mat["close"], open_ref=op["open"], **kw,
    )
    r3_ok = bool(r3.get("ok"))

    out: dict[str, Any] = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "spec_hash": spec["spec_hash"],
        "segment": {
            "test": [mat["dates"][test_lo], mat["dates"][test_hi - 1]],
            "n_dates": len(mat["dates"]),
            "n_codes": len(mat["codes"]),
        },
        "fidelity_gate": fidelity,
        "r1_close": r1,
        "r2_open": r2,
        "r3_open_limit_blocked": r3,
        "verdict": {},
    }

    # ---- 结论
    verdict: dict[str, Any] = {}
    if not fidelity["ok"]:
        verdict["status"] = "不可信"
        verdict["reason"] = (
            "close 口径没能复现 canonical 数字，本模块代码有误，"
            "因此 open 口径的结论不可采信"
        )
    else:
        if r2_ok:
            e1 = r1["excess_annualized"]
            e2 = r2["excess_annualized"]
            verdict["status"] = "已检验"
            # 结论文案与数值交给纯函数判定：文案必须跟着两个口径各自的
            # 符号走，不能只盯 open 一个口径（否则 close 本身为负时
            # 会把「本来就是负」误述成「由正转负」）。
            verdict.update(build_verdict(close_excess=e1, open_excess=e2))
        else:
            verdict["status"] = "不可用"
            verdict["reason"] = "开盘价面板覆盖不足，跑不出结果"

    if r3_ok and r2_ok:
        e3 = r3["excess_annualized"]
        verdict["r3_excess"] = e3
        verdict["blocked_limit_up_total"] = r3["blocked_limit_up_total"]
        # R3 文案同样要看 open 口径的符号：open 本来就为负时，
        # R3 不是「由正转负」，而是「本来就负、剔除涨停后更负」。
        verdict["r3_conclusion"] = build_r3_conclusion(
            close_excess=verdict.get("close_excess", 0.0),
            open_excess=r2["excess_annualized"],
            r3_excess=e3,
        )

    out["verdict"] = verdict
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print("\n=== 成交真实性检验 ===")
        print(f"忠实性闸门: {'通过' if fidelity['ok'] else '未通过'}"
              f"（最大相对误差 {fidelity['worst_rel_err']:.2e}）")
        for k, v in fidelity["checks"].items():
            print(f"  {k}: mine={v['mine']} canonical={v['canonical']} "
                  f"{'OK' if v['ok'] else 'MISMATCH'}")
        for name, r in (("R1 收盘成交(理论)", r1), ("R2 开盘成交", r2),
                        ("R3 开盘+涨停剔除", r3)):
            if not r.get("ok"):
                print(f"  {name}: 跑不出来 ({r.get('reason')})")
                continue
            print(f"  {name}: 多头年化 {r['long_only']['net_annualized']:+.2%}  "
                  f"超额 {r['excess_annualized']:+.2%}  "
                  f"换手 {r['mean_turnover']:.2%}  "
                  f"期数 {r['n_periods']}")
        if r3_ok:
            print(f"  R3 累计剔除涨停次数: {r3['blocked_limit_up_total']}")
        print(f"  结论: {verdict.get('r3_conclusion') or verdict.get('conclusion') or verdict.get('reason')}")
        print(f"  已存: {OUT}")
    return out


if __name__ == "__main__":
    run()
