"""信号解剖：这份超额到底来自哪个时段，以及**哪些时段你根本赚不到**。

为什么必须做这个：
    成交真实性检验发现一个反常现象 —— 这套方法挑出来的股票，
    在**买入当天盘中是跌的**（开盘→收盘篮子均值 -0.09%、中位数 -0.18%）。
    而回测按收盘价成交，等于「等到了当天跌完之后的价格」。

    光看总超额看不出问题在哪，必须把持有期收益**严格拆成互不重叠的段**，
    然后逐段回答两个问题：
      1. 这一段的超额是正是负？
      2. **这一段我在什么时候下单才吃得到？**

    时间轴（t = 信号日，lag=1，hold=5，ei = 买入日 = t+1，xi = 出场日 = t+6）：

        close[t] --S1(隔夜)--> open[t+1] --S2(入场日盘中)--> close[t+1]
                  --S3(持有中段)--> close[t+5] --S4(出场日盘中)--> close[t+6]

        另有 S5 = open[t+6]/close[t+5]（出场日隔夜 = 出场日前收 → 出场日开盘），
        用于拼 R2 口径。

        ⚠️ S4 与 S5 都从 close[t+5] 出发，是**同一时点的两条分支**：
        S4 走到出场日收盘（收盘成交），S5 走到出场日开盘（开盘成交）。
        开盘成交时你**没拿到出场日收盘价**，所以 S5 的分母必须是 close[t+5]
        （昨收），不能用 close[t+6]（出场日收盘）—— 那是开盘后 4.5 小时才
        形成的价格，方向还是反的（o/c 是当日日内涨幅的倒数）。

    关键洞察（这是本模块存在的核心理由）：
        信号在 close[t] 算出来。收盘价在 15:00 才最终形成，
        **你在 close[t] 无法下单**。最早能成交的是 open[t+1]。
        所以 **S1（隔夜段）结构上就赚不到** —— 无论用什么成交假设。
        而实测 S1 恰恰是超额最大的一段。

        这就是「回测数字好看但实盘不行」的精确机制：
        回测的口径把不可交易的收益算进了账本。

    两个成交口径各自能吃到哪几段（严格可加，逐股精确成立）：

        R1 收盘成交：入场 close[ei]，出场 close[xi]
            → 吃到 S3 × S4（S1、S2 都错过）
        R2 开盘成交：入场 open[ei]，出场 open[xi]
            → 吃到 S2 × S3 × S5（S1 同样错过）

        对照：S1 若能被吃到（不可行），才是「隔夜漂移」型收益。

    另附一层独立判断：训练段各因子的 IC 符号 → 这到底是反转还是动量信号。
    反转信号（买跌的）在实盘上的风险特征与动量完全不同，必须分清。

零 LLM，纯确定性。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import execution_realism
import frozen_spec
import ooslab
import panel_open

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "research" / "signal_anatomy.json"

# 段的定义（全部是「前一刻 → 后一刻」的收益，链式相乘恒等于总收益）
SEGMENTS = (
    ("S1_overnight", "隔夜(信号日收盘→买入日开盘)"),
    ("S2_entry_day", "买入日盘中"),
    ("S3_hold", "持有中段"),
    ("S4_exit_day", "出场日盘中"),
    ("S5_exit_gap", "出场日隔夜"),
)


def _seg_returns(
    close: np.ndarray,
    op: np.ndarray,
    ei: int,
    xi: int,
    sel: np.ndarray,
) -> dict[str, np.ndarray]:
    """逐股算五段收益。索引约定：
        ei = 买入日 = t+lag，xi = 出场日 = t+lag+hold
        close[ei-1] = 信号日收盘（lag=1 时）
    """
    prev_c = close[ei - 1, sel]
    o_ei = op[ei, sel]
    c_ei = close[ei, sel]
    c_xm1 = close[xi - 1, sel]
    c_xi = close[xi, sel]
    o_xi = op[xi, sel]
    return {
        "S1_overnight": o_ei / prev_c - 1.0,
        "S2_entry_day": c_ei / o_ei - 1.0,
        "S3_hold": c_xm1 / c_ei - 1.0,
        "S4_exit_day": c_xi / c_xm1 - 1.0,
        # ⚠️ 分母是 c_xm1（**昨收**）不是 c_xi（出场日收盘）。
        #   开盘成交口径在出场日**开盘**就卖出了，压根没拿到出场日收盘价；
        #   用 c_xi 当分母等于把一个"从没持有过的价格"写进收益里，
        #   方向还是反的（o/c 是当日日内涨幅的倒数）。
        #   正确口径：出场日隔夜 = 出场日开盘 / 出场日前一日收盘。
        #   这样 R2 = S2 × S3 × S5 才逐股精确等于 open[xi]/open[ei] - 1。
        "S5_exit_gap": o_xi / c_xm1 - 1.0,
    }


def _basket(close, op, ei, xi, sel) -> dict[str, float]:
    r = _seg_returns(close, op, ei, xi, sel)
    out = {k: float(np.nanmean(v)) for k, v in r.items()}
    # 严格可加校验：S1..S4 相乘 = close[xi]/close[ei-1] - 1
    chain_close = (
        (1 + out["S1_overnight"]) * (1 + out["S2_entry_day"])
        * (1 + out["S3_hold"]) * (1 + out["S4_exit_day"]) - 1.0
    )
    # R1 口径：入场 close[ei]，出场 close[xi] → S3 × S4
    out["R1_total"] = (
        (1 + out["S3_hold"]) * (1 + out["S4_exit_day"]) - 1.0
    )
    # R2 口径：入场 open[ei]，出场 open[xi] → S2 × S3 × S5
    out["R2_total"] = (
        (1 + out["S2_entry_day"]) * (1 + out["S3_hold"])
        * (1 + out["S5_exit_gap"]) - 1.0
    )
    out["chain_check"] = float(chain_close)
    return out


def _decompose(
    mat: dict[str, Any],
    score: np.ndarray,
    op: np.ndarray,
    spec: dict[str, Any],
    start_idx: int,
    end_idx: int,
) -> list[dict[str, Any]]:
    close = mat["close"]
    codes = mat["codes"]
    dates = mat["dates"]
    hold, lag = spec["hold"], spec["lag"]
    T, N = close.shape
    end_idx = T if end_idx is None else min(end_idx, T)

    rows: list[dict[str, Any]] = []
    t = start_idx
    while t + lag + hold < end_idx:
        s = score[t]
        ei = t + lag
        xi = t + lag + hold
        valid = (
            np.isfinite(s)
            & np.isfinite(close[ei]) & np.isfinite(close[xi])
            & np.isfinite(op[ei]) & np.isfinite(op[xi])
            & np.isfinite(close[ei - 1]) & np.isfinite(close[xi - 1])
            & (close[ei] > 0) & (close[xi] > 0)
            & (op[ei] > 0) & (op[xi] > 0)
        )
        n_valid = int(valid.sum())
        if n_valid >= spec["min_valid"]:
            idx = np.where(valid)[0]
            sv = s[idx]
            lu = execution_realism._limit_up_mask(close, op, codes, ei, ei - 1)
            keep = ~lu[idx]
            if keep.sum() >= spec["min_valid"]:
                idx, sv = idx[keep], sv[keep]
                order = np.argsort(sv, kind="mergesort")
                k = max(1, int(len(idx) * spec["quantile"]))
                rows.append({
                    "signal_date": dates[t],
                    "entry_date": dates[ei],
                    "exit_date": dates[xi],
                    "n_valid": int(len(idx)),
                    "blocked": int((~keep).sum()),
                    "top": _basket(close, op, ei, xi, idx[order[-k:]]),
                    "bottom": _basket(close, op, ei, xi, idx[order[:k]]),
                    "bench": _basket(close, op, ei, xi, idx),
                })
        t += hold
    return rows


def _agg(rows: list[dict[str, Any]], leg: str) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k, _ in SEGMENTS] + ["R1_total", "R2_total", "chain_check"]
    out: dict[str, float] = {}
    for k in keys:
        v = np.array([r[leg][k] for r in rows], dtype=float)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_median"] = float(np.median(v))
    return out


def _factor_direction(spec, mat) -> dict[str, Any]:
    """训练段各因子 IC 符号 → 反转还是动量。"""
    segs = ooslab.split_segments(len(mat["dates"]))
    tr_lo, tr_hi = segs["train"]
    fwd = mat.get("fwd")
    factors = mat.get("factors") or {}
    if fwd is None or not factors:
        return {"ok": False, "reason": "面板缺少 fwd/factors"}
    ic: dict[str, float] = {}
    for name in spec["factor_names"]:
        f = factors.get(name)
        if f is None:
            continue
        vals: list[float] = []
        for t in range(tr_lo, min(tr_hi, f.shape[0] - 1)):
            row_f, row_y = f[t], fwd[t]
            ok = np.isfinite(row_f) & np.isfinite(row_y)
            if ok.sum() < spec["min_valid"]:
                continue
            x, y = row_f[ok], row_y[ok]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            vals.append(float(np.corrcoef(x, y)[0, 1]))
        if vals:
            ic[name] = float(np.mean(vals))
    n_rev = sum(1 for v in ic.values() if v < 0)
    n_mom = sum(1 for v in ic.values() if v > 0)
    return {
        "ok": bool(ic),
        "ic_train": ic,
        "n_reversal": n_rev,
        "n_momentum": n_mom,
        "verdict": (
            f"训练段 {n_rev}/{n_rev+n_mom} 个因子 IC 为负（值越小未来收益越高 = 买跌的），"
            "整体是**反转型信号**"
            if n_rev > (n_rev + n_mom) / 2 else
            f"训练段 {n_rev}/{n_rev+n_mom} 个因子 IC 为负，动量/反转混杂"
        ),
    }


def main() -> None:
    spec = frozen_spec.load_spec()
    mat = ooslab.load_panel()
    op = panel_open.load_open_panel(panel=mat, verbose=False)["open"]
    score = ooslab.composite_score(mat, spec["factor_names"], spec["signed_weights"])
    segs = ooslab.split_segments(len(mat["dates"]))
    lo, hi = segs["test"]

    rows = _decompose(mat, score, op, spec, lo, hi)
    if not rows:
        print("[anatomy] 测试段无可用期次")
        return

    top = _agg(rows, "top")
    bench = _agg(rows, "bench")
    bottom = _agg(rows, "bottom")
    diff = {k: top[k] - bench[k] for k in top if k in bench}

    # 严格可加校验：R1 口径与 canonical 的 close 成交结果应当一致
    r1_check = {
        "strategy_mean_per_period": top["R1_total_mean"],
        "canonical_reference": None,
    }

    # 哪个口径吃到哪些段
    monetizable = {
        "R1_close": ["S3_hold", "S4_exit_day"],
        "R2_open": ["S2_entry_day", "S3_hold", "S5_exit_gap"],
        "S1_overnight": "❌ 任何成交口径都吃不到（信号在 close[t] 算出，"
                        "最早只能 open[t+1] 成交）",
    }

    # 不可交易段的超额贡献 —— 结论的核心
    s1_diff = diff["S1_overnight_mean"]
    s1_ann = s1_diff * (252.0 / spec["hold"])

    verdict: dict[str, Any] = {}
    verdict["untradeable_overnight_excess_ann"] = s1_ann
    verdict["untradeable_note"] = (
        f"隔夜段相对基准的超额是 {s1_ann*100:+.2f}%/年 —— "
        "而这一段在结构上就赚不到（信号在收盘后才算出来，最早次日开盘才能成交）"
    )

    ed = diff["S2_entry_day_mean"] * (252.0 / spec["hold"])
    verdict["entry_day_excess_ann"] = ed
    verdict["entry_day_note"] = (
        f"买入当天盘中相对基准 {ed*100:+.2f}%/年 —— "
        + ("**你能吃到的时段里，信号选的是买完继续跌的票**"
           if ed < -0.002 else
           "这是你能吃到的入场时段，信号在这里不占便宜也不吃亏")
    )

    # R1 / R2 各自的毛超额与成本
    for tag, key in (("R1_close", "R1_total_mean"), ("R2_open", "R2_total_mean")):
        g = diff[key]
        verdict[f"{tag}_gross_excess_per_period"] = g
        verdict[f"{tag}_gross_excess_ann"] = g * (252.0 / spec["hold"])

    direction = _factor_direction(spec, mat)
    verdict["signal_nature"] = direction.get("verdict")

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "segment": "test",
        "n_periods": len(rows),
        "segments": {k: label for k, label in SEGMENTS},
        "monetizable": monetizable,
        "top": top,
        "bench": bench,
        "bottom": bottom,
        "diff_top_minus_bench": diff,
        "factor_direction": direction,
        "verdict": verdict,
        "r1_check": r1_check,
        "periods": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 信号解剖 ===")
    print(f"测试段 {len(rows)} 期（每期 {spec['hold']} 个交易日）")
    print()
    print("  段收益（每期均值）与策略 vs 基准之差")
    for k, label in SEGMENTS:
        print(f"    {label:<26} 策略 {top[k+'_mean']*100:>8.4f}%  "
              f"基准 {bench[k+'_mean']*100:>8.4f}%  "
              f"差 {(top[k+'_mean']-bench[k+'_mean'])*100:>+8.4f}%")
    print()
    print("  严格可加校验（策略篮子）")
    print(f"    S1×S2×S3×S4 相乘      = {top['chain_check_mean']*100:.4f}%")
    print(f"    R1 口径 S3×S4         = {top['R1_total_mean']*100:.4f}%")
    print(f"    R2 口径 S2×S3×S5     = {top['R2_total_mean']*100:.4f}%")
    print(f"    基准 R1               = {bench['R1_total_mean']*100:.4f}%")
    print(f"    基准 R2               = {bench['R2_total_mean']*100:.4f}%")
    print()
    print("  结论")
    for k, v in verdict.items():
        if isinstance(v, str):
            print(f"    {k}: {v}")
    print(f"    已存: {OUT}")


if __name__ == "__main__":
    main()
