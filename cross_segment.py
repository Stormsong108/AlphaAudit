"""hold=10 的 +0.49% 超额是真信号，还是样本内挑出来的噪声？

问题
----
hold_sweep.py 在**测试段**（2023-11-20 → 2026-11-10）扫出：
    hold 10 日 / 成本 0.30% → 相对等权基准超额 +0.49%
是全部 (hold, cost) 组合里唯一为正的。

但 hold=10 在**训练段**的构造方式选择中被落选（训练段最优是 hold=5），
说明 hold=10 是在**看过测试段之后**才被挑出来的。
这属于样本内挑选，+0.49% 完全可能是噪声。

验证方法
--------
把 hold=10 拿到**训练段**和**验证段**（测试段之前的数据）上独立跑一遍。
如果 hold=10 在训练段/验证段也跑赢基准，那它是真信号；
如果只有测试段跑赢，就是运气。

注意：hold_sweep 用的是 hold=5 的因子集合（发现期按 horizon=5 选出），
本模块沿用同一集合，只换持有期，保证唯一变量是 hold。

输出
----
- 每个区段的多头年化 / 基准年化 / 超额 / 换手
- 训练段、验证段、测试段三处的超额是否同号（这是最关键的判据）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "data" / "research"

# 检验的持有期（从 sweep 的正超额候选出发）
HOLDS = (5, 10, 20, 40)

# 检验的成本（0.30% 为主，另看 0.15%）
COSTS = (0.0015, 0.0030)

TRADING_DAYS = 252


def _load_oos() -> dict[str, Any]:
    p = RESEARCH / "oos_latest.json"
    if not p.exists():
        raise RuntimeError(f"缺少 {p}，请先跑 ooslab。")
    return json.loads(p.read_text(encoding="utf-8"))


def _run_segment(
    mat: dict[str, Any],
    score,
    *,
    hold: int,
    quantile: float,
    cost_rate: float,
    lag: int,
    start_idx: int,
    end_idx: int,
) -> dict[str, Any]:
    import ooslab

    r = ooslab.decile_backtest(
        mat, score, hold=hold, lag=lag, cost_rate=cost_rate,
        quantile=quantile, start_idx=start_idx, end_idx=end_idx,
    )
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("reason")}

    # 参数生效校验
    if r.get("hold") is not None and int(r["hold"]) != int(hold):
        raise AssertionError(
            f"hold 未生效：请求 {hold}，实际 {r['hold']}（会让检验失去意义）"
        )

    lo = r["long_only"]
    bm = r["benchmark"]
    gross = r.get("gross") or {}
    gross_excess_pp = gross.get("long_vs_bench_mean")
    net_excess_pp = lo["net_mean_per_period"] - bm["mean_per_period"]

    return {
        "ok": True,
        "hold": hold,
        "n_periods": r.get("n_periods"),
        "mean_turnover": float(r["mean_turnover"]),
        "long_only_annualized": lo["net_annualized"],
        "long_only_sharpe": lo["sharpe"],
        "benchmark_annualized": bm["annualized"],
        "excess_annualized": lo["net_annualized"] - bm["annualized"],
        "gross_excess_per_period": gross_excess_pp,
        "net_excess_per_period": net_excess_pp,
        "cost_drag_per_period": (gross_excess_pp - net_excess_pp
                                 if gross_excess_pp is not None else None),
        "breakeven_cost": (gross_excess_pp / r["mean_turnover"]
                           if (gross_excess_pp is not None and r["mean_turnover"]) else None),
    }


def cross_segment_check() -> dict[str, Any]:
    """在训练段 / 验证段 / 测试段分别跑各 hold，看超额是否跨段同号。"""
    import ooslab

    raw = _load_oos()
    discovered = raw.get("discovered") or []
    if not discovered:
        raise RuntimeError("oos_latest.json 缺少 discovered。")

    factor_names = [d["name"] for d in discovered]
    signed_weights = [1.0 if d["mean"] > 0 else -1.0 for d in discovered]

    mat = ooslab.load_panel(cache=ROOT / "data" / "panel" / "panel.npz")
    segs = ooslab.split_segments(mat["close"].shape[0])
    score = ooslab.composite_score(mat, factor_names, signed_weights)

    cfg = raw.get("strategy_config") or {}
    quantile = float(cfg.get("quantile") or 0.10)
    lag = int((raw.get("config") or {}).get("lag") or 1)

    seg_meta = {
        "train": segs["train"],
        "val": segs["val"],
        "test": segs["test"],
    }

    out: dict[str, Any] = {
        "factor_names": factor_names,
        "quantile": quantile,
        "segments": {
            k: [mat["dates"][lo], mat["dates"][hi - 1]] for k, (lo, hi) in seg_meta.items()
        },
        "rows": [],
    }

    for cost in COSTS:
        for hold in HOLDS:
            per_seg = {}
            for name, (lo_i, hi_i) in seg_meta.items():
                per_seg[name] = _run_segment(
                    mat, score, hold=hold, quantile=quantile, cost_rate=cost,
                    lag=lag, start_idx=lo_i, end_idx=hi_i,
                )
            excesses = {
                k: v["excess_annualized"] for k, v in per_seg.items() if v.get("ok")
            }
            signs = {k: (1 if v > 0 else (-1 if v < 0 else 0)) for k, v in excesses.items()}
            out["rows"].append({
                "cost_rate": cost,
                "hold": hold,
                "per_segment": per_seg,
                "excess_by_segment": excesses,
                "sign_by_segment": signs,
                "all_same_sign": len(set(signs.values())) == 1 and 0 not in signs.values(),
                "all_positive": all(v > 0 for v in excesses.values()) and excesses,
            })

    return out


def analyze(result: dict[str, Any]) -> dict[str, Any]:
    """提炼：hold=10 的正超额是稳定信号还是单段运气？"""
    rows = result["rows"]
    at_base = [r for r in rows if abs(r["cost_rate"] - 0.0030) < 1e-9]

    stable = [r for r in at_base if r["all_same_sign"]]
    stable_pos = [r for r in stable if all(v > 0 for v in r["excess_by_segment"].values())]

    return {
        "n_tested": len(at_base),
        "n_sign_stable": len(stable),
        "n_positive_all_segments": len(stable_pos),
        "stable_holds": [r["hold"] for r in stable],
        "robust_holds": [r["hold"] for r in stable_pos],
        "verdict": (
            "存在跨段稳定且全为正的持有期 —— 信号真实"
            if stable_pos else
            "没有任何持有期在三个区段都跑赢基准 —— 正超额是样本内挑选的噪声"
        ),
    }


def format_report(result: dict[str, Any], analysis: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("【跨区段稳健性检验：hold=10 的正超额是真的吗】")
    L.append(f"  因子数：{len(result['factor_names'])}  分位：{result['quantile']}")
    for k, v in result["segments"].items():
        L.append(f"  {k:<6} {v[0]} → {v[1]}")
    L.append("")
    L.append("  成本 0.30% 下，各持有期在三个区段的相对基准超额：")
    L.append("    hold   训练段        验证段        测试段        同号?")
    for r in result["rows"]:
        if abs(r["cost_rate"] - 0.0030) > 1e-9:
            continue
        ex = r["excess_by_segment"]
        cells = "  ".join(f"{ex.get(s, float('nan'))*100:>+8.2f}%" for s in ("train", "val", "test"))
        same = "是" if r["all_same_sign"] else "否"
        L.append(f"    {r['hold']:>3} 日  {cells}   {same}")
    L.append("")
    L.append("  换手率（测试段）：")
    for r in result["rows"]:
        if abs(r["cost_rate"] - 0.0030) > 1e-9:
            continue
        t = r["per_segment"].get("test", {})
        L.append(f"    hold {r['hold']:>3} 日  换手 {t.get('mean_turnover', 0)*100:.2f}%/期  "
                 f"期数 {t.get('n_periods')}")
    L.append("")
    L.append("  结论：")
    L.append(f"    {analysis['verdict']}")
    if analysis["robust_holds"]:
        L.append(f"    跨段稳健的持有期：{analysis['robust_holds']}")
    L.append(f"    （检验 {analysis['n_tested']} 个持有期，"
             f"跨段同号 {analysis['n_sign_stable']} 个，"
             f"三段全正 {analysis['n_positive_all_segments']} 个）")
    return "\n".join(L)


if __name__ == "__main__":
    res = cross_segment_check()
    ana = analyze(res)
    p = RESEARCH / "cross_segment_check.json"
    p.write_text(json.dumps({"result": res, "analysis": ana},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_report(res, ana))
    print()
    print(f"已存：{p}")
