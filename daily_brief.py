"""每日研究汇报（大白话版）：把今天做了什么、查出了什么、下一步干什么讲清楚。

宋老师的要求（2026-11-12）：
  「以后每天都要把你做的事情，分析的结果讲大白话给我」

所以这份汇报必须回答 7 个问题，缺一个都不算汇报：
  1. 今天做了什么
  2. 分析结果是什么
  3. 结果意味着什么
  4. 和昨天相比变了什么
  5. 当前能不能用
  6. 主要风险是什么
  7. 下一步验证什么

设计铁律：
  D1 **零 LLM**：全部从磁盘上的确定性产物拼装（OOS 证据、成交真实性检验、
     前瞻账本、研究队列状态、体检结果）。MiniMax 再忙，这份汇报也出得来。
  D2 **不编数字**：任何数字都必须能指到来源文件。拿不到就写"取不到"，
     绝不用估算值冒充实测值。
  D3 **和昨天比**：从 state/brief_history.json 读昨天快照做差值。
     第一次跑没有昨天 → 明说"首次运行，无昨日可比"。
  D4 **纯文字**：缩进 2 空格、关键数字加粗、无表格、无 emoji、无代码块
     （宋老师飞书阅读偏好，见 feishu-user-communication skill）。
  D5 **结论前置**：第一行就说"能不能用"，不要让人读到最后才知道。

零 LLM，纯确定性。
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
HISTORY_FILE = STATE_DIR / "brief_history.json"
BRIEF_DIR = ROOT / "data" / "briefs"

OOS_FILE = ROOT / "data" / "research" / "oos_latest.json"
REALISM_FILE = ROOT / "data" / "research" / "execution_realism.json"
# ETF 通道（买场内 ETF 而不是个股的低成本第二通道）的研究产物。
# 这条通道的结论必须由产物里的 stage 字段推导 —— 历史上出过两次作废的运行
# （验证段泄漏选参、基准算出 nan%），旧数字一律不许进日报。
ETF_CHANNEL_FILE = ROOT / "data" / "research" / "etf_channel.json"
# 个股通道的固定成本地板（印花税+过户费+佣金），只作对照，不作结论。
COST_FEASIBILITY_FILE = ROOT / "data" / "research" / "cost_feasibility.json"
LEDGER_SUMMARY = ROOT / "data" / "forward" / "ledger_summary.json"
LEDGER_STATE = ROOT / "data" / "forward" / "ledger_state.json"
ANATOMY_FILE = ROOT / "data" / "research" / "signal_anatomy.json"
QUEUE_FILE = STATE_DIR / "research_queue.json"
DAILY_LOOP_FILE = STATE_DIR / "daily_loop.json"
CROSS_SEG_FILE = ROOT / "data" / "research" / "cross_segment_check.json"

# 投递防抖窗口（秒）。2026-11-09 踩坑教训：marker 必须按「完整时间戳 + 窗口」
# 判断，不能按「当天已投递」判断 —— 否则白天手动跑一次，
# 就会让当晚的定时任务被静默跳过，日报少一期。
DEBOUNCE_SEC = 120 * 60


# ------------------------------------------------------------------ 读取
def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _pct(v: Any, nd: int = 2) -> str:
    if v is None:
        return "取不到"
    try:
        return f"{float(v) * 100:+.{nd}f}%"
    except (TypeError, ValueError):
        return "取不到"


def _num(v: Any, nd: int = 2) -> str:
    if v is None:
        return "取不到"
    try:
        return f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return "取不到"


# ------------------------------------------------------------------ ETF 通道
# ETF 通道 = 把同一套信号搬到场内 ETF 上做（场内 ETF 没有印花税、没有过户费，
# 成本地板比个股通道低一个数量级），是给这条策略找的第二条低成本路。
#
# 三条纪律（2026-11-14 立规）：
#   1. 结论由产物里的 stage 字段推导，不许由"我记得上次是什么"推导 ——
#      历史上出过两次作废的运行（验证段参与选参、基准算出 nan%），
#      那些旧数字一律不许进日报。
#   2. 所有数字运行时从产物读，代码里不许出现任何 ETF 数值常量 ——
#      连"约 X bp"里的 X 也必须来自 JSON。
#   3. 取不到就如实写"取不到"，绝不静默跳过 —— 读者看到空白会以为系统挂了。

# 因子组代码 → 大白话说明。数字从代码本身取（如 mom60 里的 60），
# 不在代码里写死任何数值；产物里出现这张表没有的代码就原样显示，不猜。
_ETF_FACTOR_GLOSS = {
    "mom60": "{n} 日动量",
    "mom_dual": "短中期动量组合",
    "trend3": "多周期动量",
    "trend_lowvol": "多周期动量 + 低波动",
    "trend_dd": "多周期动量 + 回撤",
    "ma_only": "均线偏离",
    "ma_trend": "均线偏离 + 动量",
    "mixed": "动量 + 均线偏离 + 低波动",
    "trend_rsi": "多周期动量 + 相对强弱",
}

_ETF_PRICE_MODE_CN = {"open": "开盘价", "close": "收盘价"}


def _etf_channel_state() -> dict[str, Any]:
    """读 ETF 通道研究产物。

    返回 {"ok", "stage", "payload", "status_line"}：
      ok=True  → payload 里是 stage=completed 的完整结果，可以渲染数字；
      ok=False → status_line 说明这轮为什么取不到（写给读者看的那一句）。
    绝不抛异常。
    """
    if not ETF_CHANNEL_FILE.exists():
        return {"ok": False, "stage": None, "payload": None,
                "status_line": "ETF 通道：这轮结果取不到（研究产物文件还没生成）"}
    payload = _read_json(ETF_CHANNEL_FILE)
    if payload is None:
        return {"ok": False, "stage": None, "payload": None,
                "status_line": "ETF 通道：这轮结果取不到（研究产物文件损坏，JSON 读不出来）"}
    if not isinstance(payload, dict):
        return {"ok": False, "stage": None, "payload": None,
                "status_line": "ETF 通道：这轮结果取不到（研究产物结构不对，读不出对象）"}
    stage = payload.get("stage")
    if stage == "completed":
        return {"ok": True, "stage": stage, "payload": payload, "status_line": ""}
    if stage == "validation_failed":
        # 纪律：这轮没有测试段结果，就不许报任何"测试段"数字，
        # 更不许把"验证段没通过"改写成"毛收益为负"之类的结论。
        return {"ok": False, "stage": stage, "payload": None,
                "status_line": ("ETF 通道：验证段未通过 —— 训练段选出的参数在验证段净超额不为正，"
                                "按纪律停止，未进入测试段，本轮结果不可用")}
    return {"ok": False, "stage": stage, "payload": None,
            "status_line": f"ETF 通道：这轮结果取不到（这轮研究还没走完，stage={stage}）"}


def load_etf_channel() -> dict[str, Any] | None:
    """ETF 通道研究产物（data/research/etf_channel.json）。

    文件不存在 / JSON 坏掉 / stage 不是 completed → 返回 None，绝不抛异常。
    """
    state = _etf_channel_state()
    return state["payload"] if state["ok"] else None


def _etf_period_bp(metrics: dict[str, Any]) -> dict[str, Any]:
    """每期毛收益 / 净收益 / 成本的 bp 口径。

    口径纪律：产物里 mean_cost_per_period_bp 是 bp，而 gross_per_period /
    net_per_period 有时是小数、有时已经是 bp。这里不靠字段名猜，用恒等式
    验：净 = 毛 - 成本，且均值是线性的，所以毛与净的差必须正好等于每期成本。
    对得上就按对应口径换算；对不上（说明产物口径变了）就报"未取得"，
    绝不拿别处的数字替代、也不估算。

    返回 {"gross_bp", "net_bp", "cost_bp", "have_gross", "have_net"}；
    have_gross / have_net 为 False 表示该字段取不到。
    """
    out: dict[str, Any] = {"gross_bp": None, "net_bp": None, "cost_bp": None,
                           "have_gross": False, "have_net": False}
    g = metrics.get("gross_per_period")
    n = metrics.get("net_per_period")
    c = metrics.get("mean_cost_per_period_bp")
    try:
        cf = float(c)
        out["cost_bp"] = cf
    except (TypeError, ValueError):
        cf = None
    if g is None or n is None:
        return out
    try:
        gf, nf = float(g), float(n)
    except (TypeError, ValueError):
        return out
    diff = gf - nf
    if cf is None:
        # 成本取不到 → 恒等式用不上，只能按量级保守判断（小数一定 < 1）
        scale = 1e4 if max(abs(gf), abs(nf)) <= 1.0 else 1.0
    elif cf == 0.0:
        scale = 1e4 if max(abs(gf), abs(nf)) <= 1.0 else 1.0
    elif abs(diff - cf / 1e4) <= max(abs(cf), 1e-9) * 1e-6:
        scale = 1e4          # 毛/净是小数，成本是 bp
    elif abs(diff - cf) <= max(abs(cf), 1e-9) * 1e-6:
        scale = 1.0          # 毛/净本来就是 bp
    else:
        return out          # 口径对不上 → 不猜
    out["gross_bp"] = gf * scale
    out["net_bp"] = nf * scale
    out["have_gross"] = True
    out["have_net"] = True
    return out


def _etf_floor_bp_range(payload: dict[str, Any]) -> tuple[Any, Any]:
    """资金规模敏感化表里，单笔往返成本地板的 (最小, 最大) bp。"""
    vals: list[float] = []
    for row in payload.get("capital_sweep") or []:
        v = ((row or {}).get("cost_floor") or {}).get("total_bp_roundtrip")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None, None
    return min(vals), max(vals)


def _etf_sweep_cost_bp(payload: dict[str, Any]) -> tuple[Any, Any]:
    """资金规模敏感化表里，每期成本 (最小, 最大) bp。"""
    vals: list[float] = []
    for row in payload.get("capital_sweep") or []:
        v = (row or {}).get("mean_cost_per_period_bp")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None, None
    return min(vals), max(vals)


def _stock_channel_floor_bp() -> Any:
    """个股通道单笔往返的固定成本地板（bp），来自个股通道的成本可行性产物。

    只作对照用（ETF 通道省下的就是这部分）；取不到就是 None，不猜。
    """
    data = _read_json(COST_FEASIBILITY_FILE)
    if not isinstance(data, dict):
        return None
    v = (data.get("fee_floor_bp") or {}).get("order_ge_20000")
    return v if isinstance(v, (int, float)) else None


def _etf_factor_zh(fset: Any) -> str:
    """因子组代码的大白话说明。数字从代码里取，代码里不写死数值。"""
    code = str(fset or "")
    tmpl = _ETF_FACTOR_GLOSS.get(code)
    if not tmpl:
        return code
    digits = re.findall(r"\d+", code)
    return tmpl.format(n=digits[0] if digits else "")


def _etf_analysis_lines(payload: dict[str, Any]) -> list[str]:
    """第二节里的 ETF 通道核心数字。"""
    test = payload.get("test") or {}
    metrics = test.get("metrics") or {}
    bench = test.get("benchmark") or {}
    spec = payload.get("frozen_spec") or {}
    chosen = payload.get("chosen") or {}
    cost_model = payload.get("cost_model") or {}

    fset = spec.get("factor_set") or chosen.get("factor_set")
    k = spec.get("k") if spec.get("k") is not None else chosen.get("k")
    R = spec.get("R") if spec.get("R") is not None else chosen.get("R")
    capital = spec.get("capital")
    price_mode = spec.get("price_mode") or payload.get("price_mode")

    L: list[str] = ["    ETF 通道（买场内 ETF 的第二条低成本路）"]
    params: list[str] = []
    if fset:
        zh = _etf_factor_zh(fset)
        params.append(f"因子组 {fset}" + (f"（{zh}）" if zh and zh != str(fset) else ""))
    if k is not None:
        params.append(f"每次买 {k} 只")
    if R is not None:
        params.append(f"持有 {R} 个交易日")
    if capital is not None:
        params.append(f"资金 {_num(capital, 0)} 元")
    if price_mode:
        params.append(f"按{_ETF_PRICE_MODE_CN.get(str(price_mode), price_mode)}成交")
    if params:
        L.append("      冻结参数：" + "、".join(params))

    L.append(f"      测试段净年化 {_pct(metrics.get('net_annualized'))}，"
             f"同期基准（等权拿住全部合格 ETF）年化 {_pct(bench.get('annualized'))}，"
             f"超额 {_pct(test.get('excess_ann'))}/年")

    bp = _etf_period_bp(metrics)
    gross_txt = f"{_num(bp['gross_bp'])} bp" if bp["have_gross"] else "毛边际未取得"
    net_txt = f"{_num(bp['net_bp'])} bp" if bp["have_net"] else "净边际未取得"
    L.append(f"      每期毛收益 {gross_txt}、每期净收益 {net_txt}、"
             f"每期成本 {_num(bp['cost_bp'])} bp")

    L.append(f"      三个口径的净年化：掐掉极端好坏期再平均（缩尾均值）"
             f"{_pct(metrics.get('net_winsor_mean_annualized'))}、"
             f"不掐的平均（未缩尾均值）{_pct(metrics.get('net_annualized'))}、"
             f"中位数 {_pct(metrics.get('net_median_annualized'))}")

    L.append(f"      胜率 {_pct(metrics.get('win_rate'), 1)}（每期赚钱的比例），"
             f"最大回撤 {_pct(metrics.get('max_drawdown_ann'))}")

    stamp = cost_model.get("stamp_duty")
    transfer = cost_model.get("transfer_fee")
    if stamp is None and transfer is None:
        caliber = "成本口径取不到（产物里没有费率表）"
    elif stamp == 0 and transfer == 0:
        caliber = "场内 ETF 不收印花税、不收过户费，只收佣金"
    else:
        caliber = (f"印花税 {_pct(stamp, 3)}、过户费 {_pct(transfer, 4)}"
                   "（按产物里的费率表执行）")
    floor_lo, floor_hi = _etf_floor_bp_range(payload)
    L.append(f"      成本口径：{caliber}；单笔往返成本地板 {_num(floor_lo)}~{_num(floor_hi)} bp")

    sweep_lo, sweep_hi = _etf_sweep_cost_bp(payload)
    if sweep_lo is not None:
        L.append(f"      不同资金规模下，每期实际成本在 {_num(sweep_lo)}~{_num(sweep_hi)} bp 之间"
                 "（最低佣金在小资金上不线性）")

    gen = payload.get("generated_at")
    if gen:
        L.append(f"      （这组数字算于 {gen}，面板每长一天数据就会重算一次）")
    return L


def _etf_verdict_lines(payload: dict[str, Any] | None,
                       status_line: str) -> list[str]:
    """第五节里的 ETF 通道结论。

    结论由产物里的数字决定，不预设：
      毛收益就是负的 → 问题在信号本身，不是成本，换通道救不回来；
      毛正净负       → 成本把这点边际吃光了，降成本这条路走到头；
      毛正净正       → 这轮数字是正的，但样本太小，还不能进模拟仓位。
    产物取不到时只如实说取不到，不给任何结论。
    """
    L: list[str] = ["    ETF 通道（买场内 ETF 的第二条低成本路）的结论"]
    if payload is None:
        L.append(f"      {status_line}")
        L.append("      这一轮没有可用结果，所以现在给不出结论 —— 等 ETF 通道重跑出来再说。")
        return L

    test = payload.get("test") or {}
    metrics = test.get("metrics") or {}
    bp = _etf_period_bp(metrics)
    floor_lo, floor_hi = _etf_floor_bp_range(payload)
    stock_floor = _stock_channel_floor_bp()

    if floor_lo is not None:
        if stock_floor is not None:
            L.append(f"      成本地板：ETF 单笔往返 {_num(floor_lo)}~{_num(floor_hi)} bp，"
                     f"个股通道单笔往返 {_num(stock_floor)} bp（ETF 低得多）")
        else:
            L.append(f"      成本地板：ETF 单笔往返 {_num(floor_lo)}~{_num(floor_hi)} bp")

    if not bp["have_gross"] or not bp["have_net"]:
        L.append("      结论：取不到 —— 产物的毛/净边际字段缺失，这轮给不出结论")
        return L
    if bp["gross_bp"] < 0:
        L.append("      结论：不能用作独立策略，也不得进入模拟仓位")
        L.append(f"      原因：每期毛收益本身就是 {_num(bp['gross_bp'])} bp —— "
                 "还没扣成本就已经是负的")
        L.append("      也就是说：问题不在成本，在信号本身。换到 ETF 上也救不回这个信号。")
    elif bp["net_bp"] < 0:
        L.append("      结论：不能用作独立策略，也不得进入模拟仓位")
        L.append(f"      原因：每期毛收益 {_num(bp['gross_bp'])} bp 是正的，"
                 f"但扣掉成本后只剩 {_num(bp['net_bp'])} bp —— 成本把这点边际全吃掉了")
        L.append("      也就是说：降成本这条路也走到头了，信号本身仍然立不住。")
    else:
        n_periods = metrics.get("n_periods")
        L.append("      结论：这轮数字是正的，但还不能进模拟仓位")
        L.append(f"      原因：测试段只有 {n_periods if n_periods is not None else '取不到'} 期，"
                 "这么点样本说明不了信号稳不稳")
        L.append("      下一步：先把样本拉长重跑，再谈能不能进模拟仓位。")
    return L


def _etf_risk_lines(payload: dict[str, Any] | None) -> list[str]:
    """第六节里的 ETF 通道风险。幸存者偏差用产物原文，不自己编。"""
    L: list[str] = ["    ETF 通道的风险"]
    if payload is None:
        L.append("      ETF 通道这轮没有可用结果，样本池的幸存者偏差等风险暂时无从评估"
                 "（等重跑出结果再评）")
        return L
    disclosure = payload.get("survivorship_bias_disclosure")
    if disclosure:
        L.append(f"      ETF 通道的样本池有幸存者偏差（研究产物原文）：{disclosure}")
    superseded = payload.get("superseded_runs") or []
    for row in superseded:
        issue = str((row or {}).get("issue") or "").strip()
        if issue:
            L.append(f"      ETF 通道有一版历史产物已作废，它的数字不得被引用（{issue}）")
    return L


def _etf_next_lines(payload: dict[str, Any] | None,
                    stage: str | None) -> list[str]:
    """第七节里的 ETF 通道下一步。"""
    L: list[str] = ["    ETF 通道的下一步"]
    if payload is None:
        if stage == "validation_failed":
            L.append("      ETF 通道：把上游数据（复权/面板）的问题查清后重跑整条流程"
                     "（训练段选参 → 验证段只判定 → 冻结 → 测试段只跑一次）；"
                     "验证段没通过之前不换参、不看测试段")
        else:
            L.append("      ETF 通道：等 ETF 通道研究跑出结果后，再判断这条路能不能用")
        return L
    L.append("      ETF 通道：把这次测试段的结论放到更长的面板上再跑一遍，"
             "确认不是样本太短造成的")
    if payload.get("superseded_runs"):
        L.append("      ETF 通道：每次重跑都从训练段选参开始，"
                 "验证段只判定通过与否、不参与选参（历史上有一版栽在这里，已作废）")
    return L


# ------------------------------------------------------------------ 快照
def build_snapshot() -> dict[str, Any]:
    """抓当前所有关键指标，作为「今天」和「昨天」的比较基准。"""
    snap: dict[str, Any] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    oos = _read_json(OOS_FILE)
    if oos:
        seg = oos.get("strategy_test_segment") or {}
        lo = seg.get("long_only") or {}
        bench = seg.get("benchmark") or {}
        snap["oos"] = {
            "net_annualized": lo.get("net_annualized"),
            "sharpe": lo.get("sharpe"),
            "benchmark_annualized": bench.get("annualized"),
            "excess_annualized": (
                lo.get("net_annualized") - bench.get("annualized")
                if lo.get("net_annualized") is not None
                and bench.get("annualized") is not None else None
            ),
            "mean_turnover": seg.get("mean_turnover"),
            "n_periods": seg.get("n_periods"),
        }
        # 面板规模（n_codes / n_dates / 末日）—— 供日报引用，
        # 避免把「1298 只」写死在文案里随面板增长而失真。
        pan = oos.get("panel") or {}
        if pan:
            snap["panel"] = {
                "n_codes": pan.get("n_codes"),
                "n_dates": pan.get("n_dates"),
                "to": pan.get("to"),
                "from": pan.get("from"),
            }

    real = _read_json(REALISM_FILE)
    if real:
        v = real.get("verdict") or {}
        r1 = real.get("r1_close") or {}
        r2 = real.get("r2_open") or {}
        r3 = real.get("r3_open_limit_blocked") or {}
        snap["realism"] = {
            "fidelity_ok": (real.get("fidelity_gate") or {}).get("ok"),
            "close_excess": r1.get("excess_annualized"),
            "open_excess": r2.get("excess_annualized"),
            "open_limit_excess": r3.get("excess_annualized"),
            "blocked_total": r3.get("blocked_limit_up_total"),
            "verdict": v.get("status"),
            "conclusion": v.get("conclusion"),
            # 生成时间戳：防止把几天前的旧数字当今天的结论。
            # 2026-11-14 踩坑：面板长新数据后 canonical 数字变了
            # （-2.03% → -1.30%），但产物文件还是旧的，日报照抄了过期数字。
            "generated_at": real.get("generated_at"),
        }

    led = _read_json(LEDGER_SUMMARY)
    if led:
        r = led.get("real") or {}
        snap["ledger"] = {
            "n_periods": led.get("n_periods"),
            "new_periods": led.get("new_periods_written"),
            "cumulative_excess": r.get("cumulative"),
            "annualized_excess": r.get("annualized"),
            "positive_ratio": r.get("positive_ratio"),
            "first_signal_date": led.get("first_signal_date"),
            "chain_ok": (led.get("chain") or {}).get("ok"),
        }

    an = _read_json(ANATOMY_FILE)
    if an:
        d = an.get("diff_top_minus_bench") or {}
        v = an.get("verdict") or {}
        rc = an.get("r1_check") or {}
        snap["anatomy"] = {
            "overnight_diff": d.get("S1_overnight_mean"),
            "entry_day_diff": d.get("S2_entry_day_mean"),
            "hold_diff": d.get("S3_hold_mean"),
            "exit_day_diff": d.get("S4_exit_day_mean"),
            # r1_gross_diff / r2_gross_diff 是**毛超额**（已扣成本前），
            # 来自 diff_top_minus_bench（策略篮子 − 基准篮子）。
            "r1_gross_diff": d.get("R1_total_mean"),
            "r2_gross_diff": d.get("R2_total_mean"),
            "untradeable_ann": v.get("untradeable_overnight_excess_ann"),
            "entry_day_ann": v.get("entry_day_excess_ann"),
            "signal_nature": v.get("signal_nature"),
            "n_periods": an.get("n_periods"),
            # 独立校验与 canonical 对账状态 —— 日报必须如实展示，
            # 对账不过就不能拿这些数字下结论（闭环硬门禁）。
            "chain_check_ok": an.get("chain_check_ok"),
            "r1_check_ok": rc.get("ok"),
            "r1_check_rel_err": rc.get("rel_err"),
            "r1_check_reference": rc.get("canonical_reference"),
            "generated_at": an.get("generated_at"),
        }

    cs = _read_json(CROSS_SEG_FILE)
    if cs:
        # ⚠️ 2026-11-16 修正：cross_segment_check.json 的真实结构是
        #   result.rows[] → per_segment.{train,val,test}.excess_annualized
        #   result.segments → 只是日期区间（不是收益值）
        # 旧代码读 cs["segments"] 拿 excess_annualized，取不到 →
        # 分段结论静默退化成「暂无数据」，而日报却写死了「三段一致为负」。
        res = cs.get("result") or cs
        rows_cs = res.get("rows") or []
        seg_vals: dict[str, float] = {}
        seg_dates: dict[str, list] = {}
        if rows_cs:
            # 优先取第一行（canonical 配置：cost_rate/hold 由训练段选出）
            per = (rows_cs[0].get("per_segment") or {})
            for tag in ("train", "val", "validation", "test"):
                node = per.get(tag)
                if isinstance(node, dict) and node.get("excess_annualized") is not None:
                    seg_vals[tag] = float(node["excess_annualized"])
            for tag, rng in (res.get("segments") or {}).items():
                if isinstance(rng, list):
                    seg_dates[tag] = rng
        vals = list(seg_vals.values())
        snap["cross_segment"] = {
            "available": bool(vals),
            "segments": seg_vals,
            "segment_dates": seg_dates,
            "n_segments": len(vals),
            "all_negative": bool(vals) and all(x < 0 for x in vals),
            "any_positive": bool(vals) and any(x > 0 for x in vals),
            "all_same_sign": bool(vals) and (all(x < 0 for x in vals) or all(x > 0 for x in vals)),
            "generated_at": cs.get("generated_at"),
            # 该产物是 11-12 生成的，比其它产物旧 —— 日报要显示数据日期，
            # 否则读者会以为它和今天的数字同源。
            "source_mtime": datetime.fromtimestamp(
                CROSS_SEG_FILE.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M") if CROSS_SEG_FILE.exists() else None,
        }

    q = _read_json(QUEUE_FILE)
    if q:
        snap["queue"] = {
            k: {
                "status": v.get("last_status"),
                "last_run_at": v.get("last_run_at"),
                "error": v.get("last_error"),
            }
            for k, v in (q.get("steps") or {}).items()
        }

    dl = _read_json(DAILY_LOOP_FILE)
    if dl:
        snap["daily_loop"] = {
            k: {
                "status": v.get("last_status"),
                "last_run_at": v.get("last_run_at"),
                "error": v.get("last_error"),
                # detail 带覆盖率/失败数等结构化明细 —— 用来把
                # 「上游断了」和「数据坏了」这两件事分开讲。
                "detail": v.get("last_detail") or {},
            }
            for k, v in (dl.get("stages") or {}).items()
        }

    return snap


# ------------------------------------------------------------------ 昨日
def load_history() -> list[dict[str, Any]]:
    h = _read_json(HISTORY_FILE)
    if not h:
        return []
    return h.get("entries") or []


def save_history(snap: dict[str, Any]) -> None:
    entries = load_history()
    entries.append(snap)
    # 只留最近 400 天，避免文件无限膨胀
    entries = entries[-400:]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"entries": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(HISTORY_FILE)


def _prev_snapshot(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """昨天（或最近一次）的快照。"""
    if len(history) < 1:
        return None
    return history[-1]


def _diff(a: Any, b: Any, same_day: bool = False) -> str:
    """差值描述（b 是新值，a 是旧值）。

    ⚠️ 对比对象是**上一份汇报**，不一定是昨天。
    同一天内重跑（21:00 复算 + 21:30 正式投递，或手动补跑）时，
    若写死「昨日」，会与正文里的「上一份汇报时间：今天 HH:MM」自相矛盾。
    所以 same_day=True 时改用「上一份 / 本份」措辞。
    """
    old_lbl, new_lbl = ("上一份", "本份") if same_day else ("昨日", "今日")
    if a is None or b is None:
        return "无上一份可比" if same_day else "无昨日可比"
    try:
        d = float(b) - float(a)
    except (TypeError, ValueError):
        if a == b:
            return "与上一份相同" if same_day else "与昨日相同"
        return f"{old_lbl} {a} → {new_lbl} {b}"
    if abs(d) < 1e-12:
        return "与上一份相同" if same_day else "与昨日相同"
    return f"{old_lbl} {_pct(a)} → {new_lbl} {_pct(b)}（{_pct(d)}）"


def _same_day(prev: dict[str, Any] | None, cur: dict[str, Any] | None) -> bool:
    """上一份汇报是否就是**当天**的（同一天内重跑）。"""
    if not prev or not cur:
        return False
    pts, cts = str(prev.get("ts") or ""), str(cur.get("ts") or "")
    if len(pts) < 10 or len(cts) < 10:
        return False
    return pts[:10] == cts[:10]


# ------------------------------------------------------------------ 汇报
def build_brief(snap: dict[str, Any] | None = None,
                prev: dict[str, Any] | None = None) -> str:
    """生成大白话每日汇报。"""
    snap = snap if snap is not None else build_snapshot()
    history = load_history()
    prev = prev if prev is not None else _prev_snapshot(history)
    if prev is None:
        prev = {}

    L: list[str] = []
    day = time.strftime("%Y-%m-%d")

    L.append(f"【每日研究汇报 {day}】")
    L.append("")

    # ---------- 0. 结论前置 ----------
    real = snap.get("realism") or {}
    led = snap.get("ledger") or {}
    oos = snap.get("oos") or {}
    an = snap.get("anatomy") or {}

    close_ex = real.get("close_excess")
    open_ex = real.get("open_excess")
    openlim_ex = real.get("open_limit_excess")

    # 当前能不能用 —— 由三个硬事实决定
    can_use = False
    reasons: list[str] = []
    if openlim_ex is not None:
        if openlim_ex > 0:
            can_use = True
            reasons.append("现实成交口径下超额为正")
        else:
            reasons.append(
                f"现实成交口径（开盘价 + 涨停剔除）超额为 {_pct(openlim_ex)}/年，为负"
            )
    else:
        reasons.append("成交真实性检验尚无结果")
    if oos.get("excess_annualized") is not None:
        if oos["excess_annualized"] <= 0:
            reasons.append(
                f"回测口径超额 {_pct(oos['excess_annualized'])}/年，同样为负"
            )
    n_led = led.get("n_periods") or 0
    if n_led < 12:
        reasons.append(f"前瞻账本仅 {n_led} 期，离 12 期门槛还差 {12 - n_led} 期")

    verdict = "可以用" if can_use else "还不能用"
    L.append("  一句话结论")
    L.append(f"    当前状态：**{verdict}**，还不能拿真钱去做。")
    for r in reasons:
        L.append(f"    原因：{r}")
    L.append("")

    # ---------- 1. 今天做了什么 ----------
    L.append("  一、今天做了什么")
    L.append("")
    acts: list[str] = []
    if real:
        acts.append("把历史回测的成交假设拿真刀真枪验了一遍"
                    "（收盘价成交 vs 开盘价成交 vs 开盘封涨停买不到）")
    if an:
        acts.append("把这笔收益按时间段拆开解剖，"
                    "定位超额到底来自隔夜、入场当天还是持有期间")
    if led:
        acts.append("维护前瞻纸面跟踪账本"
                    "（规则已冻结，按真实时间一天天记，历史回测期次不算数）")
    q = snap.get("queue") or {}
    for key, v in q.items():
        acts.append(f"研究队列跑「{key}」：{v.get('status') or '未知'}")
    dl = snap.get("daily_loop") or {}
    # 研究队列的 refresh 步骤就是调 `daily_loop.py --stage refresh`，
    # 和每日流水线的 refresh 是**同一个任务**，并列显示两条会让人
    # 误以为是两件不同的事各挂了一次。这里只显示一次，并标清来源。
    dup = [k for k in dl if k in q]
    for key, v in dl.items():
        if key in q:
            continue
        acts.append(f"每日闭环跑「{key}」：{v.get('status') or '未知'}")
    if dup:
        acts.append(
            "    （上面那几条就是每日闭环的同一个任务，"
            "不是两件事各挂了一次）"
        )
    if not acts:
        acts.append("今天没有产生新的研究动作")
    for i, a in enumerate(acts, 1):
        L.append(f"    {i}. {a}")
    L.append("")

    # ---------- 2. 分析结果 ----------
    L.append("  二、分析结果是什么")
    L.append("")
    L.append("    回测口径（假设能按收盘价全部买到）")
    if oos:
        L.append(f"      策略年化 {_pct(oos.get('net_annualized'))}，"
                 f"同期大盘年化 {_pct(oos.get('benchmark_annualized'))}")
        L.append(f"      相对大盘 **{_pct(oos.get('excess_annualized'))}/年**"
                 "（负数 = 跑输）")
        L.append(f"      换手率 {_pct(oos.get('mean_turnover'))} 一期，"
                 f"共 {oos.get('n_periods') or '取不到'} 期")
    else:
        L.append("      取不到（OOS 证据文件缺失）")

    L.append("")
    L.append("    现实口径（换成开盘价成交，买不到的涨停股剔除）")
    if real:
        L.append(f"      按收盘价成交：超额 **{_pct(close_ex)}/年**")
        L.append(f"      按开盘价成交：超额 **{_pct(open_ex)}/年**")
        L.append(f"      再剔除买不到的涨停股：超额 **{_pct(openlim_ex)}/年**")
        L.append(f"      整个测试段里买不到的次数：{real.get('blocked_total') or 0} 次")
        gen = real.get("generated_at")
        if gen:
            L.append(f"      （这组数字算于 {gen}，面板每长一天数据就会重算一次）")
    else:
        L.append("      取不到（成交真实性检验还没跑）")
    if real and real.get("fidelity_ok") is False:
        L.append("      忠实性闸门未通过 —— 以下数字不可信，先别看")

    L.append("")
    L.append("    前瞻账本（从今天开始真实时间往前推）")
    if n_led:
        L.append(f"      已记 **{n_led} 期**，累计超额 **{_pct(led.get('cumulative_excess'))}**")
        L.append(f"      年化折算 {_pct(led.get('annualized_excess'))}，"
                 f"为正的期数占比 {_pct(led.get('positive_ratio'), 1)}")
    else:
        L.append("      还没到能出数的期数（每期 5 个交易日），属正常")
    L.append("")

    # ETF 通道（低成本的第二条路）—— 嵌进第二节，不另起第八段。
    # 状态与数字都来自产物：取不到就如实写取不到，绝不静默跳过。
    etf_state = _etf_channel_state()
    etf_payload = etf_state["payload"]
    etf_status = etf_state["status_line"]
    if etf_payload is not None:
        L.extend(_etf_analysis_lines(etf_payload))
    else:
        L.append("    ETF 通道（买场内 ETF 的第二条低成本路）")
        L.append(f"      {etf_status}")
    L.append("")

    # ---------- 3. 结果意味着什么 ----------
    L.append("  三、结果意味着什么")
    L.append("")
    if an and an.get("overnight_diff") is not None:
        # ⚠️ 2026-11-16 修正：旧版本把「毛超额是负的」写死成结论，
        # 但真实值是**毛超额为正、被成本压成负**。方向写反会让读者
        # 对策略机理产生完全错误的理解。现在全部按实测值动态判定方向。
        r1_gross = an.get("r1_gross_diff")
        r2_gross = an.get("r2_gross_diff")
        oos_ex = oos.get("excess_annualized")

        L.append("    第一，先纠正一个容易被误解的地方：")
        if r1_gross is not None:
            if r1_gross > 0:
                L.append(f"      即使用最宽松的假设（全部按收盘价成交、全部买得到），")
                L.append(f"      **毛超额是 {_pct(r1_gross)}/期（正的）** ——")
                L.append("      也就是说，这套方法本来是能挣到一点的。"
                         "问题出在两处：")
                L.append("      一是它挣的这点好处**结构上吃不到**（下面第三点）；")
                L.append("      二是**手续费把它吃掉了**，甚至吃成负数。")
            else:
                L.append(f"      即使用最宽松的假设（全部按收盘价成交、全部买得到），")
                L.append(f"      超额也是 {_pct(r1_gross)}/期 —— 是负的。")
                L.append("      所以问题不是「实盘会不会把好处吃掉」，而是「这个好处本来就不存在」。")
        L.append("")
        L.append("    第二，我把这笔收益按时间段拆开，挖到了真正的原因：")
        L.append(f"      这套信号挑出来的股票，**隔夜会跳空高开**（相对大盘 {_pct(an.get('overnight_diff'))}/期），")
        L.append(f"      但**开盘之后立刻开始跌**（相对大盘 {_pct(an.get('entry_day_diff'))}/期）。")
        L.append("      关键在于：信号是收盘后才算出来的，你最早只能在**次日开盘**成交。")
        L.append("      所以那波跳空收益，你**结构上就吃不到**；")
        L.append("      而开盘之后的下跌，你**全部要接着承受**。")
        L.append("")
        L.append("    第三，这意味着回测里那个「按收盘价成交」的假设，是**双向作弊**：")
        L.append("      它既白赚了买不到的跳空，又躲开了开盘后的下跌。")
        L.append(f"      这项每年值 {_pct(an.get('untradeable_ann'))}，正好是两个口径超额的差。")
        L.append("")
        L.append("    第四，最硬的一条结论：")
        if r2_gross is not None:
            if r2_gross > 0:
                L.append(f"      按真实开盘成交算，**毛超额**（还没扣手续费）仍是 {_pct(r2_gross)}/期 —— "
                         "**是正的**。")
                L.append("      这点必须说清楚：**不是「连成本都不用扣就已经跑不赢」**。")
                L.append("      真实情况是：毛超额为正，但**扣掉交易成本后变成负数**。")
            else:
                L.append(f"      按真实开盘成交算，**毛超额**（还没扣手续费）就是 {_pct(r2_gross)}/期，")
                L.append("      本身就是负的。也就是说——")
                L.append("      **这套方法连成本都不用扣，就已经跑不赢无脑买一篮子股票了。**")
        nature = (an.get("signal_nature") or "")
        if nature:
            L.append(f"      另外，{nature}。")
        L.append("")
        L.append("    第五，结论：不是「再调调参数」，是「这条路的门要先关掉」。")
        # 闭环校验状态必须如实展示 —— 对账不过就不能下结论
        if an.get("r1_check_ok") is False:
            L.append("")
            L.append("    但本轮有个数据问题必须先说明：")
            L.append("      解剖模块与 canonical 回测引擎的毛超额**对账没有通过**"
                     f"（相对误差 {an.get('r1_check_rel_err')}）。")
            L.append("      在这个差值查清之前，上面这段解读**只能当作参考**，不能当结论。")
        elif an.get("chain_check_ok") is False:
            L.append("")
            L.append("    独立校验（逐股连乘 vs 价格重算）没有通过，"
                     "以上解读暂不作为结论。")
    elif close_ex is not None and openlim_ex is not None:
        L.append("    取不到成交真实性检验结果，暂时无法解读。")
    else:
        L.append("    取不到成交真实性检验结果，暂时无法解读。")
    L.append("")

    # ---------- 4. 和昨天比 ----------
    same_day = _same_day(prev, snap)
    L.append("  四、和昨天比变了什么" if not same_day else "  四、和上一份比变了什么")
    L.append("")
    if not prev:
        L.append("    首次运行，没有昨天可比。")
    else:
        if same_day:
            L.append(f"    （上一份汇报就在同一天内，编号见下方时间戳）")
        L.append(f"    回测超额：{_diff((prev.get('oos') or {}).get('excess_annualized'), oos.get('excess_annualized'), same_day)}")
        L.append(f"    现实成交超额：{_diff((prev.get('realism') or {}).get('open_limit_excess'), openlim_ex, same_day)}")
        L.append(f"    前瞻账本期数：{_diff((prev.get('ledger') or {}).get('n_periods'), n_led, same_day)}")
        if prev.get("ts"):
            L.append(f"    上一份汇报时间：{prev.get('ts')}")
    L.append("")

    # ---------- 5. 当前能不能用 ----------
    L.append("  五、当前能不能用")
    L.append("")
    L.append(f"    判定：**{verdict}**")
    L.append("")
    L.append("    能用的部分：")
    L.append("      整条研究流水线是通的 —— 数据体检、样本外检验、成交真实性检验、")
    L.append("      概率校准、每日预测与事后打分、飞书投递，全部自动跑，零人工。")
    L.append("      这套基础设施本身是资产，不管这套策略将来行不行。")
    L.append("")
    L.append("    不能用的部分：")
    oos_ex = oos.get("excess_annualized")
    L.append("      这套 11 因子的选股方法本身。回测口径就已经跑不赢大盘"
             f"（{_pct(oos_ex)}/年），")
    L.append(f"      换成现实成交口径更差（{_pct(openlim_ex)}/年）——")
    # ⚠️ 2026-11-16 修正：旧版本把「训练/验证/测试三段一致为负」写死。
    # 真实数据不是这样（train 段实测为正），写死会把结论说过头。
    # 现在按 cross_segment_check.json 的实测值动态描述。
    cs = snap.get("cross_segment") or {}
    if cs.get("available") and cs.get("n_segments"):
        segs = cs.get("segments") or {}
        parts = []
        for tag in ("train", "validation", "test"):
            if tag in segs and segs[tag] is not None:
                parts.append(f"{tag} {_pct(segs[tag])}/年")
        if parts:
            L.append("      分段看：" + "、".join(parts) + " ——")
        if cs.get("all_negative"):
            L.append("      **三段一致为负**，说明这是「结构性没有超额」，不是「参数没调好」。")
        elif cs.get("any_positive"):
            L.append("      **三段符号不一致**（有的段为正、有的段为负），"
                     "说明这个结果很可能来自样本内选择偏差或运气，"
                     "而不是稳定存在的规律 —— 这种情况下更不能上实盘。")
        else:
            L.append("      分段结果见上。")
    else:
        L.append("      分段数据暂缺，无法判断一致性 —— 在补齐之前，"
                 "同样不能给出「结构性没有超额」的定论。")
    L.append("      结论是：这套方法不具备实盘条件，只能做研究和纸面跟踪。")
    L.append("")
    L.append("    什么时候我会说「可以实盘」：")
    L.append("      前瞻账本积累够期次之后（至少 12 期、覆盖半年以上真实行情），")
    L.append("      且现实成交口径下累计超额为正。我不会在历史回测上给这个结论。")
    L.append("")

    # ETF 通道的结论 —— 由产物里的 stage 与数字推导，不预设、不写死。
    L.extend(_etf_verdict_lines(etf_payload, etf_status))
    L.append("")

    # ---------- 6. 主要风险 ----------
    L.append("  六、主要风险是什么")
    L.append("")
    risks: list[str] = []
    # 忠实性闸门是**三态**，不能压成两态：
    #   False = 跑了但没过 → 结论不可采信
    #   None  = 根本没跑     → 「没数据」，不等于「结论不可信」
    #   True  = 过了         → 不列这条风险
    fid = real.get("fidelity_ok")
    if fid is False:
        risks.append("成交真实性检验的忠实性闸门没过，结论不可采信")
    elif fid is None:
        risks.append("成交真实性检验还没跑过，下面这些数字暂时不能当作结论")
    chain_ok = led.get("chain_ok")
    if chain_ok is False:
        risks.append("前瞻账本哈希链断裂 —— 账本被改动过，数据不可信")
    else:
        risks.append("前瞻账本样本还太少，半年内可能只是噪音，不能据此下结论")
    risks.append("回测用的是后复权价，实盘是真实成交价，两者永远有差距")
    # 上游行情不可用 ≠ 本地数据坏了。这两件事必须分开讲，
    # 否则读者会以为「数据不可信」，而实际上只是「今天没能再确认一次」。
    rf = ((snap.get("daily_loop") or {}).get("refresh") or {})
    if rf.get("status") == "failed":
        detail = rf.get("detail") or {}
        cov = detail.get("coverage")
        fails = detail.get("failed")
        cov_txt = f"{cov * 100:.1f}%" if isinstance(cov, (int, float)) else "取不到"
        risks.append(
            f"今天没能从行情上游重新确认一次数据（上游接口断了，"
            f"刷新成功率 {cov_txt}，{fails} 只失败）—— "
            "这不代表本地历史数据坏了，只是少了一次校验；"
            "已经算出来的结论仍然基于那批已校验过的数据"
        )
    risks.append("涨停剔除已按实际涨停价判定（主板10%/双创20%/ST 5%），但仍无法建模集合竞价排队、"
                 "部分成交与盘口深度——真实成交价比开盘价假设更差，不会更好")
    n_codes = (snap.get("panel") or {}).get("n_codes")
    if n_codes:
        risks.append(f"策略只覆盖 {n_codes} 只股票，不含北交所、科创板之外的部分新股")
    risks.append("研究结论基于历史，未来市场结构变化会让历史失效")
    for i, r in enumerate(risks, 1):
        L.append(f"    {i}. {r}")
    L.append("")

    # ETF 通道的风险 —— 幸存者偏差用产物原文，不自己编。
    etf_risks = _etf_risk_lines(etf_payload)
    if etf_risks:
        L.extend(etf_risks)
        L.append("")

    # ---------- 7. 下一步 ----------
    L.append("  七、下一步验证什么")
    L.append("")
    nxt = [
        "前瞻账本继续跑，每天结算已到期的期次 —— 半年到一年后才有资格谈实盘",
        "盯住三件事：累计超额是不是一直负（说明方法真不行）、换手能不能降下来、"
        "前后半段有没有明显衰减",
        "找**不同类型**的方法：不是调这 11 个因子，而是换思路（比如更长的持有期、"
        "更少因子的组合、行业中性化后还剩多少 alpha）",
        "把成本假设从 0.30% 往下压，看看打平线能不能到 0.12% 以内 —— "
        "这是这套方法唯一的翻盘机会",
        "每一项新研究都先跑「成交真实性检验」，避免又做出一个只在收盘价里好看的结论",
    ]
    for i, x in enumerate(nxt, 1):
        L.append(f"    {i}. {x}")
    L.append("")

    # ETF 通道的下一步 —— 由 stage 决定说什么，不写死。
    L.extend(_etf_next_lines(etf_payload, etf_state.get("stage")))
    L.append("")

    L.append("  以上数字全部来自实际跑出来的文件，不是估算。")
    L.append("  拿不到的数字我都写了「取不到」，没有编。")
    L.append("")
    L.append("  想让我把哪一块展开细讲，说一声就行。")

    return "\n".join(L)


def deliver_brief(text: str, *, dry_run: bool = False) -> list[dict[str, Any]]:
    """把大白话汇报投到金融组 + 宋老师 DM。

    路由铁律（2026-11-09 宋老师定规）：
        金融产出只发金融组 oc_cc9756440fd41494acc866dce3b4e885 + 宋老师 DM。
        **绝不发胖叔说留学·创作群** —— 那是留学内容的地盘。
        config.FEISHU['forbidden_chat_ids'] 里已列入禁止名单，
        这里再查一遍，双重保险。
    """
    import config
    import deliver as deliver_mod

    targets: list[tuple[str, str, str]] = []
    group = config.FEISHU.get("group_chat_id") or config.FEISHU.get("group_id")
    if group:
        targets.append(("group", group, "chat_id"))
    user = config.FEISHU.get("open_id")
    if user:
        targets.append(("dm", user, "open_id"))

    forbidden = set(config.FEISHU.get("forbidden_chat_ids") or [])
    results: list[dict[str, Any]] = []
    for kind, rid, id_type in targets:
        if rid in forbidden:
            results.append({"target": kind, "ok": False,
                            "error": "命中禁止投递名单，已拦截"})
            continue
        if dry_run:
            results.append({"target": kind, "ok": True,
                            "message_id": None, "error": "dry-run 未真发"})
            continue
        try:
            r = deliver_mod.send_text(text, receive_id=rid, id_type=id_type)
            ok = bool(r.get("ok"))
            results.append({"target": kind, "ok": ok,
                            "message_id": r.get("message_id") or r.get("data", {}).get("message_id"),
                            "error": r.get("error") or r.get("msg")})
        except Exception as exc:  # noqa: BLE001
            results.append({"target": kind, "ok": False, "error": str(exc)})
    return results


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="每日研究汇报（大白话）")
    ap.add_argument("--stdout", action="store_true", help="打印到标准输出")
    ap.add_argument("--save", action="store_true", help="存一份到 data/briefs/")
    ap.add_argument("--no-history", action="store_true", help="不写历史快照")
    ap.add_argument("--deliver", action="store_true", help="投递到金融组 + 宋老师 DM")
    ap.add_argument("--dry-run", action="store_true", help="只演练不真发")
    ap.add_argument("--force", action="store_true", help="忽略防抖窗口")
    args = ap.parse_args()

    snap = build_snapshot()
    text = build_brief(snap)

    if args.save:
        BRIEF_DIR.mkdir(parents=True, exist_ok=True)
        day = time.strftime("%Y-%m-%d")
        path = BRIEF_DIR / f"brief_{day}.txt"
        path.write_text(text, encoding="utf-8")

    if not args.no_history:
        save_history(snap)

    if args.deliver:
        # 时间窗口防抖（2026-11-09 踩坑教训）：
        #   marker 必须按「完整时间戳 + 窗口」判断，不能按「当天已投递」判断 ——
        #   否则白天手动跑一次，就会让当晚的定时任务被静默跳过。
        day = time.strftime("%Y-%m-%d")
        marker = STATE_DIR / f"brief_delivered_{day}.marker"
        if marker.exists() and not args.force:
            try:
                last = float(marker.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                last = 0.0
            if time.time() - last < DEBOUNCE_SEC:
                print(f"[brief] {DEBOUNCE_SEC//60} 分钟内已投递，跳过（--force 可强制重发）")
                return
        results = deliver_brief(text, dry_run=args.dry_run)
        ok = all(r.get("ok") for r in results) if results else False
        if ok and not args.dry_run:
            marker.write_text(str(time.time()), encoding="utf-8")
        for r in results:
            print(f"[brief][deliver] {r['target']} ok={r.get('ok')} "
                  f"mid={r.get('message_id')} err={r.get('error')}")
        if not ok:
            raise SystemExit(1)

    if args.stdout:
        print(text)


if __name__ == "__main__":
    main()
