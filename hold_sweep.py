"""固定持有期实验：回答「拉长调仓周期压换手，能否翻转净超额为正？」

为什么需要这个实验（2026-11-12）
-------------------------------
`live_readiness.py` 算出的结论是：当前策略净超额 -2.03%（相对等权基准），
且维持 0.30% 成本时换手需从 22.08%/期 降到 8.67%/期（降 60.7%）才能追平基准。

「拉长持有期」是降低换手最直接的实盘手段（调仓越少，成本越低）。
但 `ooslab.py --hold N` **不能**用来做这个实验 —— 已定位的真实原因：

    horizon = args.hold if args.hold > 0 else hs["chosen"]   # ooslab.py:1253
    ...
    chosen = sel.get("chosen")           # 训练段 select_construction 选出
    strat = run_config(mat, score, chosen, ...)   # 用 chosen["horizon"]
    hold = int(cfg.get("horizon") or 5)  # run_config 从 cfg 读，**不是** args.hold

即：`--hold` 只影响**因子发现/IC 检验**用的前瞻收益窗口，
**策略回测的调仓周期仍由训练段选出的 chosen["horizon"] 决定**。
实测证据：--hold 10 与 --hold 20 的 strategy_test_segment 逐位相同
（long_only_annualized 0.18206070730641133 vs ...138，差 1e-17 浮点级）。

所以本模块直接调用底层回测函数，在**测试段**上固定 hold 扫描，
绕开训练段选择 —— 这才是有效的实验。

口径纪律
--------
- 因子集合固定用「发现期选出的那 11 个」，不重新发现，避免前视偏差。
- 方向权重固定用发现期 IC 符号。
- 每个结果都记录实际传入的 hold，并在输出里校验 `decile_backtest`
  回报的 hold 与请求一致 —— 参数没生效必须当场炸掉，不能静默出数。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "data" / "research"

# 固定持有期扫描点（交易日）
HOLD_GRID = (5, 10, 20, 40, 60)

# 成本情景（往返）
COSTS = (0.0015, 0.0030, 0.0050)

# 年化交易日
TRADING_DAYS = 252


def _load_oos() -> dict[str, Any]:
    p = RESEARCH / "oos_latest.json"
    if not p.exists():
        raise RuntimeError(f"缺少 {p}，请先跑 ooslab。")
    return json.loads(p.read_text(encoding="utf-8"))


def run_hold_sweep(
    *,
    hold_grid=HOLD_GRID,
    costs=COSTS,
    quantile: float | None = None,
) -> dict[str, Any]:
    """在测试段上跑固定持有期 × 成本情景的扫描。

    quantile=None 表示沿用训练段选出的构造方式（decile + 具体分位）；
    显式给定时跑纯十分位。
    """
    import ooslab

    raw = _load_oos()
    discovered = raw.get("discovered") or []
    if not discovered:
        raise RuntimeError("oos_latest.json 缺少 discovered 因子列表。")

    factor_names = [d["name"] for d in discovered]
    signed_weights = [1.0 if d["mean"] > 0 else -1.0 for d in discovered]

    # 面板 + 分段
    mat = ooslab.load_panel(cache=ROOT / "data" / "panel" / "panel.npz")
    segs = ooslab.split_segments(mat["close"].shape[0])
    test_lo, test_hi = segs["test"]

    # 合成信号：因子集合与权重全部来自发现期，测试段不重新发现
    score = ooslab.composite_score(mat, factor_names, signed_weights)

    # 构造方式：默认沿用训练段选出的
    cfg = raw.get("strategy_config") or {}
    construction = cfg.get("construction") or "decile"
    q = quantile if quantile is not None else (cfg.get("quantile") or 0.10)
    if construction != "decile" and quantile is not None:
        raise RuntimeError("指定 quantile 时必须用 decile 构造方式。")

    rows: list[dict[str, Any]] = []
    for hold in hold_grid:
        for cost in costs:
            run_cfg = {
                "horizon": hold,
                "construction": construction,
                "quantile": q,
            }
            r = ooslab.run_config(
                mat, score, run_cfg,
                lag=int((raw.get("config") or {}).get("lag") or 1),
                cost_rate=cost,
                start_idx=test_lo, end_idx=test_hi,
            )
            if not r.get("ok"):
                rows.append({
                    "hold": hold, "cost_rate": cost, "ok": False,
                    "reason": r.get("reason"),
                })
                continue

            # ── 参数生效校验：回测回报的 hold 必须等于请求的 hold ──
            # 不校验就会出现"参数没生效但照样出数"的假实验（真实踩过）。
            actual_hold = r.get("hold")
            if actual_hold is not None and int(actual_hold) != int(hold):
                raise AssertionError(
                    f"hold 参数未生效：请求 {hold}，回测实际用 {actual_hold}。"
                    "（这会让整个扫描失去意义，必须当场炸掉。）"
                )

            lo = r["long_only"]
            bm = r["benchmark"]
            gross = r.get("gross") or {}
            turn = float(r["mean_turnover"])

            gross_excess_pp = gross.get("long_vs_bench_mean")
            net_excess_pp = lo["net_mean_per_period"] - bm["mean_per_period"]
            cost_drag_pp = (gross_excess_pp - net_excess_pp
                            if gross_excess_pp is not None else None)

            rows.append({
                "ok": True,
                "hold": hold,
                "cost_rate": cost,
                "construction": construction,
                "quantile": q,
                "n_periods": r.get("n_periods"),
                "mean_turnover": turn,
                "long_only_annualized": lo["net_annualized"],
                "long_only_sharpe": lo["sharpe"],
                "long_only_win_rate": lo.get("win_rate"),
                "benchmark_annualized": bm["annualized"],
                "excess_annualized": lo["net_annualized"] - bm["annualized"],
                "gross_excess_per_period": gross_excess_pp,
                "net_excess_per_period": net_excess_pp,
                "cost_drag_per_period": cost_drag_pp,
                # 年化口径：per-period 值 × (252/hold)，与 ooslab 一致。
                # 用 252 会把「5 天一期」当成「一天一期」，虚高 hold 倍。
                "periods_per_year": TRADING_DAYS / float(r.get("hold") or hold),
                "cost_drag_annualized": (cost_drag_pp * (TRADING_DAYS / float(r.get("hold") or hold))
                                         if cost_drag_pp is not None else None),
                "gross_excess_annualized": (gross_excess_pp * (TRADING_DAYS / float(r.get("hold") or hold))
                                            if gross_excess_pp is not None else None),
                # 打平成本：毛超额/换手
                "breakeven_cost": (gross_excess_pp / turn
                                   if (gross_excess_pp is not None and turn) else None),
            })

    return {
        "construction": construction,
        "quantile": q,
        "factor_names": factor_names,
        "n_factors": len(factor_names),
        "test_segment": [mat["dates"][test_lo], mat["dates"][test_hi - 1]],
        "rows": rows,
    }


def analyze(result: dict[str, Any]) -> dict[str, Any]:
    """从扫描结果里提炼结论。

    关键问题：有没有某个 (hold, cost) 组合让 excess_annualized > 0？
    """
    rows = [r for r in result["rows"] if r.get("ok")]
    if not rows:
        return {"has_positive_excess": False, "best": None}

    positive = [r for r in rows if r["excess_annualized"] > 0]

    # 基准成本 0.30% 下换手最低且超额最高的 hold
    base = [r for r in rows if abs(r["cost_rate"] - 0.0030) < 1e-9]
    base.sort(key=lambda r: r["excess_annualized"], reverse=True)
    best_base = base[0] if base else None

    # 每个 hold 的最低打平成本要求
    by_hold: dict[int, float] = {}
    for r in base:
        h = r["hold"]
        by_hold[h] = min(by_hold.get(h, 9e9), r["breakeven_cost"] or 9e9)

    return {
        "has_positive_excess": bool(positive),
        "n_positive": len(positive),
        "best": best_base,
        "breakeven_cost_by_hold": by_hold,
        "turnover_by_hold": {
            r["hold"]: r["mean_turnover"] for r in base
        },
    }


def format_report(result: dict[str, Any], analysis: dict[str, Any]) -> str:
    L: list[str] = []
    q = result["quantile"]
    qtxt = "" if q is None else f" / 分位 {q}"
    L.append("【固定持有期扫描：拉长调仓周期能否翻转净超额】")
    L.append(f"  构造方式：{result['construction']}{qtxt}"
             f"  因子数：{result['n_factors']}")
    L.append(f"  测试段：{result['test_segment'][0]} → {result['test_segment'][1]}")
    L.append("")
    L.append("  一、基准成本 0.30% 下的换手与超额")
    for h in sorted(analysis["turnover_by_hold"]):
        row = next(r for r in result["rows"]
                   if r.get("ok") and r["hold"] == h
                   and abs(r["cost_rate"] - 0.0030) < 1e-9)
        L.append(f"    hold {h:>3} 日  换手 {row['mean_turnover']*100:>5.2f}%/期  "
                 f"多头年化 {row['long_only_annualized']*100:>+7.2f}%  "
                 f"Sharpe {row['long_only_sharpe']:>5.3f}  "
                 f"超额 {row['excess_annualized']*100:>+7.2f}%  "
                 f"打平成本 {row['breakeven_cost']*100:.4f}%")
    L.append("")
    L.append("  二、全扫描中超额为正的组合")
    if analysis["has_positive_excess"]:
        for r in result["rows"]:
            if r.get("ok") and r["excess_annualized"] > 0:
                L.append(f"    hold {r['hold']:>3} 日 / 成本 {r['cost_rate']*100:.2f}%  "
                         f"超额 {r['excess_annualized']*100:+.2f}%  "
                         f"换手 {r['mean_turnover']*100:.2f}%/期")
    else:
        L.append("    无 —— 任何（持有期, 成本）组合下都没能跑赢等权基准。")
    L.append("")
    L.append("  三、结论")
    if not analysis["has_positive_excess"]:
        L.append("    拉长调仓周期**不能**翻转结论：")
        L.append("    降换手省下的成本，小于拉长周期损失的选股 alpha。")
        L.append("    → 实盘若用这套方法，定位只能是「增强器/风控工具」，"
                 "不能当独立主力策略。")
    else:
        b = analysis["best"]
        L.append(f"    存在正超额组合，最优：hold {b['hold']} 日 / "
                 f"成本 {b['cost_rate']*100:.2f}%，超额 {b['excess_annualized']*100:+.2f}%")
    return "\n".join(L)


if __name__ == "__main__":
    res = run_hold_sweep()
    ana = analyze(res)
    out = RESEARCH / "hold_sweep_result.json"
    out.write_text(json.dumps({"result": res, "analysis": ana},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_report(res, ana))
    print()
    print(f"已存：{out}")
