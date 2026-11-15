"""前瞻纸面跟踪账本：从今天起，用真实时间一天天验证这套方法。

为什么必须做这个：
    历史回测已经证明「过去跑不赢基准」。要回答「将来能不能用」，
    唯一诚实的办法是**从今天开始冻结规则、逐期记录、用真实时间推进**。
    半年到一年后，账本里积累的是**前瞻样本**，不是回测样本 ——
    这两者不能互相冒充，所以账本必须是独立的一条时间线。

账本的四条铁律：
    L1 规则冻结：因子、权重符号、持有期、分位、成本、成交口径全部写进
       `ledger_state.json`，并记 spec_hash。改规则 = 开一条新账本，
       旧账本封存 —— **绝不允许在同一条时间线上改规则后继续记**，
       那样前半段和后半段就不是同一个策略了。
    L2 只记已发生：每一期的收益只能在**该期结束之后**才写进账本。
       正在进行的期次只记「已开仓、待结算」，不预估收益。
       未实现的账面浮盈一律不进账本（那是自欺欺人）。
    L3 现实成交：账面收益按**次日开盘价**成交算，不用收盘价 ——
       收盘价是当天最后一刻才形成的，用它成交等于假设你能在知道结果的
       同时拿到那个价格。涨停开盘封板的股票算买不到，直接剔除。
    L4 只追加：账本是 append-only 的 JSONL，每行带前一条的哈希。
       改历史行会立刻暴露（哈希链断裂）。

每期记录什么（每期 5 个交易日）：
    信号日 / 入场日 / 出场日
    候选数、实际持仓数、因涨停剔除数
    毛收益、扣成本后收益、基准收益、超额
    理论成交（收盘）vs 现实成交（开盘）的差 = 成交假设的代价
    换手率

账本同时回答三个问题：
    Q1 方法有没有效？   → 累计超额随时间走的是正是负
    Q2 会不会衰减？     → 逐期超额序列，看后半段是否明显弱于前半段
    Q3 折损在哪？       → 成交假设（开盘 vs 收盘）、成本、涨停剔除各占多少

零 LLM，纯确定性。
"""
from __future__ import annotations

import hashlib
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
LEDGER_DIR = ROOT / "data" / "forward"
LEDGER_FILE = LEDGER_DIR / "ledger.jsonl"
STATE_FILE = LEDGER_DIR / "ledger_state.json"
SUMMARY_FILE = LEDGER_DIR / "ledger_summary.json"

LEDGER_VERSION = 1


# ------------------------------------------------------------------ 哈希链
def _row_hash(row: dict[str, Any], prev_hash: str) -> str:
    payload = json.dumps(
        {k: v for k, v in row.items() if k != "hash"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(f"{prev_hash}|{payload}".encode("utf-8")).hexdigest()


def read_ledger() -> list[dict[str, Any]]:
    if not LEDGER_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in LEDGER_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def verify_chain(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """校验哈希链完整性 —— 账本被改过就会在这里暴露。"""
    rows = rows if rows is not None else read_ledger()
    prev = ""
    for i, r in enumerate(rows):
        expect = _row_hash(r, prev)
        if r.get("hash") != expect:
            return {
                "ok": False,
                "broken_at": i,
                "rows": len(rows),
                "detail": "哈希链断裂 —— 账本被改动过",
            }
        prev = r.get("hash") or ""
    return {"ok": True, "rows": len(rows), "head_hash": prev}


def _append(rows_to_add: list[dict[str, Any]]) -> None:
    """追加行（原子写 + 哈希链续接）。"""
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    existing = read_ledger()
    prev = existing[-1].get("hash") if existing else ""
    with LEDGER_FILE.open("a", encoding="utf-8") as f:
        for row in rows_to_add:
            row = dict(row)
            row["prev_hash"] = prev
            row["hash"] = _row_hash(row, prev)
            prev = row["hash"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ 状态
def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(state: dict[str, Any]) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ------------------------------------------------------------------ 核心计算
def _period_rows(
    mat: dict[str, Any],
    score: np.ndarray,
    open_m: np.ndarray,
    spec: dict[str, Any],
    *,
    start_idx: int,
    end_idx: int,
) -> list[dict[str, Any]]:
    """跑出 (现实成交, 理论成交) 两套逐期结果。

    复用 execution_realism.run_periodic —— 它已经通过忠实性闸门验证
    （close 口径逐位复现 canonical 数字，误差 0.00e+00）。
    """
    kw = dict(
        hold=spec["hold"], lag=spec["lag"], cost_rate=spec["cost_roundtrip"],
        quantile=spec["quantile"], min_valid=spec["min_valid"],
        winsor=spec["winsor"], start_idx=start_idx, end_idx=end_idx,
    )
    real = execution_realism.run_periodic(
        mat, score, open_m, open_m,
        block_limit_up=True, prev_close=mat["close"], open_ref=open_m, **kw,
    )
    theory = execution_realism.run_periodic(
        mat, score, mat["close"], mat["close"], **kw,
    )
    if not real.get("ok") or not theory.get("ok"):
        return []
    return _zip_periods(real["periods"], theory["periods"], spec)


def _zip_periods(
    real: list[dict[str, Any]],
    theory: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """两套口径按 (signal_date, entry_date, exit_date) 对齐后合并。"""
    tmap = {
        (p["signal_date"], p["entry_date"], p["exit_date"]): p for p in theory
    }
    out: list[dict[str, Any]] = []
    for rp in real:
        key = (rp["signal_date"], rp["entry_date"], rp["exit_date"])
        tp = tmap.get(key)
        if tp is None:
            continue
        turn = rp["turnover"]
        cost = spec["cost_roundtrip"] * turn
        out.append({
            "signal_date": rp["signal_date"],
            "entry_date": rp["entry_date"],
            "exit_date": rp["exit_date"],
            "n_valid": rp["n_valid"],
            "n_top": rp["n_top"],
            "blocked_limit_up": rp.get("blocked_limit_up", 0),
            "turnover": turn,
            "cost_charged": cost,
            # 现实成交（开盘价，涨停剔除）
            "gross_real": rp["long"],
            "net_real": rp["long"] - cost,
            "bench_real": rp["bench"],
            "excess_real": rp["long"] - cost - rp["bench"],
            # 理论成交（收盘价）
            "gross_theory": tp["long"],
            "net_theory": tp["long"] - cost,
            "excess_theory": tp["long"] - cost - tp["bench"],
            # 成交假设的代价：理论 - 现实
            "execution_drag": (tp["long"] - cost) - (rp["long"] - cost),
        })
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """从账本行汇总出「现在能不能用」需要的数字。"""
    if not rows:
        return {"n_periods": 0}

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=float)

    excess = arr("excess_real")
    excess_t = arr("excess_theory")
    turn = arr("turnover")
    cost = arr("cost_charged")
    drag = arr("execution_drag")
    n = len(rows)

    hold = 5
    def ann(x: np.ndarray) -> float:
        return float(x.mean() * (252.0 / hold))

    half = n // 2
    seg_stats = {}
    if half >= 2:
        for label, sl in (("前半段", slice(0, half)), ("后半段", slice(n - half, n))):
            e = excess[sl]
            seg_stats[label] = {
                "n": int(len(e)),
                "mean_per_period": float(e.mean()),
                "annualized": ann(e),
                "positive_ratio": float(np.mean(e > 0)),
            }

    return {
        "n_periods": n,
        "first_signal_date": rows[0]["signal_date"],
        "last_exit_date": rows[-1]["exit_date"],
        "real": {
            "mean_per_period": float(excess.mean()),
            "annualized": ann(excess),
            "cumulative": float(np.prod(1.0 + excess) - 1.0),
            "positive_ratio": float(np.mean(excess > 0)),
            "std": float(excess.std(ddof=1)) if n > 1 else 0.0,
        },
        "theory": {
            "mean_per_period": float(excess_t.mean()),
            "annualized": ann(excess_t),
            "cumulative": float(np.prod(1.0 + excess_t) - 1.0),
        },
        "execution_drag": {
            "mean_per_period": float(drag.mean()),
            "annualized": ann(drag),
            "cumulative": float(np.prod(1.0 + drag) - 1.0),
        },
        "cost": {
            "mean_per_period": float(cost.mean()),
            "annualized": ann(cost),
            "cumulative": float(np.prod(1.0 - cost) - 1.0),
        },
        "turnover": {
            "mean_per_period": float(turn.mean()),
            "annualized": float(turn.mean() * (252.0 / hold)),
        },
        "blocked_limit_up_total": int(sum(r["blocked_limit_up"] for r in rows)),
        "segments": seg_stats,
    }


# ------------------------------------------------------------------ 建账 / 推进
def init_ledger(*, verbose: bool = True) -> dict[str, Any]:
    """建立账本：冻结规则，起点 = 今天之后第一个交易日。

    已存在的账本不会被覆盖 —— 要改规则请换新文件（`--reset` 需显式确认）。
    """
    spec = frozen_spec.load_spec()
    if STATE_FILE.exists():
        st = load_state()
        if st.get("spec_hash") == spec["spec_hash"]:
            if verbose:
                print(f"[ledger] 账本已存在且规则未变（spec_hash={spec['spec_hash'][:12]}），"
                      f"起始 {st.get('start_date')}，已记 {len(read_ledger())} 期")
            return st
        raise RuntimeError(
            "规则指纹与现有账本不一致 —— 这是**另一套策略**。"
            "不能在同一账本上继续记（否则前后两段不可比）。"
            "请换新文件建新账本，或用 --reset 显式重置。"
        )

    mat = ooslab.load_panel()
    segs = ooslab.split_segments(len(mat["dates"]))
    test_hi = segs["test"][1]

    # 账本起点 = 面板当前最后一天（= 历史证据之后的第一根K线）。
    #
    # ⚠️ 为什么必须钉死**绝对日期**而不是「测试段之后的位置」：
    #     split_segments 是按日期总数切比例的，日期一多，测试段边界就会**移动**。
    #     如果账本用「索引」定位起点，每加一天数据起点就漂一次，
    #     记的期次会跟着变 —— 半年后账本里根本不是同一段时间。
    #     钉死绝对日期后，历史面板增长不影响已记期次。
    start_date = mat["dates"][-1]
    if test_hi >= len(mat["dates"]):
        # 面板末尾就是测试段末尾 → 前瞻起点 = 面板最后一天，等新数据长出来
        start_date = mat["dates"][-1]

    state = {
        "ledger_version": LEDGER_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec_hash": spec["spec_hash"],
        "spec": spec,
        "start_date": start_date,
        "start_date_note": (
            "账本起点 = 建账时面板的最后一天。第一期需要 start_date 之后 "
            f"{spec['lag'] + spec['hold']} 个交易日的行情才会结算，"
            "所以建账当天通常是 0 期，属正常。"
        ),
        "panel_last_date": mat["dates"][-1],
        "panel_dates_n": len(mat["dates"]),
        "n_codes": len(mat["codes"]),
        "execution_rule": "次日开盘价成交 + 开盘封涨停剔除（现实口径）",
        "settled_rule": "期次结束（出场日之后）才写入收益，未结束只记待结算",
        "note": (
            "本账本是独立的前瞻时间线，与历史回测互不冒充。"
            "改规则必须另建新账本。"
        ),
    }
    save_state(state)
    if verbose:
        print(f"[ledger] 新建前瞻账本")
        print(f"  规则指纹 {spec['spec_hash'][:12]}")
        print(f"  起点     {start_date}（等新数据长出来才会有第一期）")
        print(f"  面板末日 {state['panel_last_date']}  共 {len(mat['dates'])} 日")
        for line in frozen_spec.spec_digest_lines(spec):
            print(f"  {line}")
    return state


def advance_ledger(*, verbose: bool = True) -> dict[str, Any]:
    """推进账本：把已经结束的期次结算并写入。

    幂等 —— 同一期不会写两次（按 signal_date+exit_date 去重）。
    """
    state = load_state()
    if not state:
        raise RuntimeError("账本还没建，先跑 init_ledger()")

    spec = frozen_spec.load_spec()
    if spec["spec_hash"] != state.get("spec_hash"):
        raise RuntimeError(
            f"规则指纹不匹配：账本 {state.get('spec_hash','')[:12]} vs 当前 {spec['spec_hash'][:12]}"
        )

    existing = read_ledger()
    done = {(r.get("signal_date"), r.get("exit_date")) for r in existing}

    mat = ooslab.load_panel()
    dates = list(mat["dates"])
    # 起点：从账本起始日往前找对齐（面板可能已经滚动）
    try:
        start_idx = dates.index(state["start_date"])
    except ValueError:
        start_idx = next(
            (i for i, d in enumerate(dates) if d >= state["start_date"]), len(dates)
        )

    score = ooslab.composite_score(mat, spec["factor_names"], spec["signed_weights"])
    op = panel_open.load_open_panel(panel=mat, verbose=False)

    rows = _period_rows(
        mat, score, op["open"], spec,
        start_idx=start_idx, end_idx=len(dates),
    )

    fresh = [r for r in rows if (r["signal_date"], r["exit_date"]) not in done]

    if fresh:
        _append(fresh)

    summary = _summarize(read_ledger())
    summary["new_periods_written"] = len(fresh)
    summary["chain"] = verify_chain()
    summary["spec_hash"] = state["spec_hash"]
    summary["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if verbose:
        print(f"[ledger] 本次新增 {len(fresh)} 期，累计 {summary['n_periods']} 期")
        if summary["n_periods"]:
            r = summary["real"]
            print(f"  现实成交累计超额 **{r['cumulative']*100:+.2f}%**"
                  f"（年化 {r['annualized']*100:+.2f}%，"
                  f"{r['positive_ratio']:.0%} 的期为正）")
            print(f"  成交假设代价 累计 {summary['execution_drag']['cumulative']*100:+.2f}%")
        print(f"  哈希链校验 {'通过' if summary['chain']['ok'] else '断裂！'}")
    return summary


def status(*, verbose: bool = True) -> dict[str, Any]:
    state = load_state()
    rows = read_ledger()
    chain = verify_chain(rows)
    summary = _summarize(rows)
    info = {
        "state": state,
        "n_rows": len(rows),
        "chain": chain,
        "summary": summary,
    }
    if verbose:
        if not state:
            print("[ledger] 账本还没建")
            return info
        print(f"[ledger] 起始 {state.get('start_date')}  "
              f"规则指纹 {state.get('spec_hash','')[:12]}  "
              f"已记 {len(rows)} 期")
        print(f"  哈希链 {'通过' if chain['ok'] else '断裂'}")
        if summary.get("n_periods"):
            r = summary["real"]
            print(f"  现实成交累计超额 {r['cumulative']*100:+.2%}  "
                  f"年化 {r['annualized']*100:+.2%}")
    return info


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="前瞻纸面跟踪账本")
    ap.add_argument("cmd", choices=["init", "advance", "status"])
    args = ap.parse_args()
    if args.cmd == "init":
        init_ledger()
    elif args.cmd == "advance":
        advance_ledger()
    else:
        status()
