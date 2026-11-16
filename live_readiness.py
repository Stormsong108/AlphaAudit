"""实盘就绪度体检（live readiness）。

定位
----
这个模块**不负责找信号**（那是 ooslab 的事），它只回答一个更靠后的问题：

    「假设这套方法真的要用真钱上，它会以什么形态活下来？」

研究阶段能过的门禁，和实盘能活下来，是两件事。2026-11-12 的体检把两者
的差距量化了出来：

    毛超额（未扣成本）      +1.31% / 年
    成本拖累                -3.34% / 年     ← 换手 22.08%/期 × 成本 0.30%
    净超额（相对等权基准）  -2.03% / 年
    三口径落差              均值 +16.40%  vs  中位数 -7.30%

三条结论，每一条都直接改变实盘打法：

1. **信号有相对选股能力，但绝对收益来自 beta。**
   多头组 16.40% < 等权基准 18.43%，多空组合净年化 -4.04%。
   → 实盘不能指望"这套方法能赚钱"，只能指望"它能在某些行情里少亏"。
   → 把它当**增强器**（叠加在宽基上），不是当**主力仓位**。

2. **成本是决定生死的唯一变量。**
   打平成本 0.1178% 往返，实际按 0.30% 算，高出 2.55 倍。
   → 实盘第一件事是把成本压到 0.12% 以内：低佣金券商 + 大额低频调仓 +
     避开流动性差的标的。换手率从 22.08% 降到 8.67%（降 60.7%），结论才翻转。

3. **中位数口径才是实盘预期。**
   均值 +16.40% 被少数极端收益拉高，中位数 -7.30% 才是多数期数的真实结果。
   → 实盘心理准备必须是"多数时间是亏的"，靠少数大盈利月兑现全年。
   → 仓位必须按"长期忍受负收益"来设计，不能按"年均 16%"来设计。

本模块把这些判断做成**可复跑的确定性计算**，每次上实盘前跑一遍。
全部零 LLM。

数字纪律
--------
所有数字来自 `data/research/oos_latest.json`（由 ooslab 落盘），
本模块**不重算策略**、不手写任何数字字面量，只做分解与换算。
每个输出字段都标注来源键名，便于核对。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "data" / "research"
OOS_LATEST = RESEARCH / "oos_latest.json"
CROSS_SEGMENT = RESEARCH / "cross_segment_check.json"

# 每年交易日数（A 股约 242，这里沿用 ooslab 的年化口径 252）
TRADING_DAYS = 252

# 实盘成本情景区间（往返，含佣金+印花税+冲击成本）
# 0.0015 激进：万 2 佣金 + 大单冲击极低 + 几乎不换手
# 0.0030 中性：当前研究假设
# 0.0050 保守：小盘股 + 频繁调仓
COST_SCENARIOS = (0.0015, 0.0030, 0.0050)


class EvidenceMissing(RuntimeError):
    """研究证据缺失 —— 不允许用默认值凑数。"""


def load_oos() -> dict[str, Any]:
    """读取 ooslab 落盘的研究证据。缺失即抛，绝不静默返回空。"""
    if not OOS_LATEST.exists():
        raise EvidenceMissing(
            f"研究证据不存在：{OOS_LATEST}。请先跑 ooslab 生成，再做实盘体检。"
        )
    data = json.loads(OOS_LATEST.read_text(encoding="utf-8"))
    if not data:
        raise EvidenceMissing(f"研究证据为空文件：{OOS_LATEST}")
    return data


def _section(raw: dict[str, Any]) -> dict[str, Any]:
    """取测试段策略结果。"""
    seg = raw.get("strategy_test_segment")
    if not isinstance(seg, dict) or not seg:
        raise EvidenceMissing(
            "oos_latest.json 缺少 strategy_test_segment，无法做实盘体检。"
        )
    return seg


def _periods_per_year(seg: dict[str, Any]) -> float:
    """一年有多少个「期」。

    口径纪律（2026-11-12 修正）
    ----------------------------
    本项目的回测以「持有 N 日」为一个期（hold=5）。
    ooslab 里所有年化都用 `mean_per_period * 252 / hold`，
    所以 **252 是交易日数，不是期数**。

    曾经踩的坑：live_readiness 的成本分解直接用 per-period 值 × 252，
    等于把「5 天的一期」当成「一天的一期」，毛超额/成本拖累/净超额
    三个数字整体虚高 5 倍：
        错: 毛 +6.55%  拖累 -16.69%  净 -10.14%
        对: 毛 +1.31%  拖累  -3.34%  净  -2.03%
    而且 -10.14% 与 C2 的 -2.03% 自相矛盾（同一件事两个数），
    这正是「口径串了」的自检信号。

    现在统一走本函数，保证与 ooslab / C2 / 成本敏感性完全同口径。
    """
    hold = seg.get("hold")
    try:
        hold_f = float(hold)
    except (TypeError, ValueError):
        hold_f = 5.0
    if hold_f <= 0:
        hold_f = 5.0
    return TRADING_DAYS / hold_f


def cost_decomposition(raw: dict[str, Any]) -> dict[str, Any]:
    """成本结构分解：毛利 → 成本 → 净利。

    恒等式：毛超额 - 成本拖累 = 净超额
    实测成本拖累与「换手 × 成本率」精确相等（两者独立算出，误差应为 0），
    这是一条自洽性校验 —— 若对不上，说明口径串了。

    年化口径：per-period 值 × (252/hold)，与 ooslab 一致（见 _periods_per_year）。
    """
    seg = _section(raw)
    gross = seg.get("gross") or {}
    lo = seg.get("long_only") or {}
    bm = seg.get("benchmark") or {}

    gross_excess = float(gross["long_vs_bench_mean"])          # 毛超额/期
    net_excess = float(lo["net_mean_per_period"]) - float(bm["mean_per_period"])
    cost_drag = gross_excess - net_excess                        # 成本拖累/期

    turnover = float(seg["mean_turnover"])                       # 单边换手/期
    cost_rate = float(seg["cost_rate"])                          # 往返成本率
    ppy = _periods_per_year(seg)                                 # 一年多少期（252/hold）

    implied_drag = turnover * cost_rate                         # 理论成本拖累/期
    residual = cost_drag - implied_drag                         # 残差（应为 0）

    return {
        "gross_excess_per_period": gross_excess,
        "net_excess_per_period": net_excess,
        # 语义：正数 = 成本吃掉了多少超额（拖累量，不是贡献量）
        "cost_drag_per_period": cost_drag,
        "turnover_per_period": turnover,
        "cost_rate_roundtrip": cost_rate,
        "implied_cost_drag_per_period": implied_drag,
        "identity_residual": residual,
        "identity_holds": abs(residual) < 1e-9,
        "annualized": {
            "gross_excess": gross_excess * ppy,
            "net_excess": net_excess * ppy,
            "cost_drag": cost_drag * ppy,
            "periods_per_year": ppy,
        },
    }


def breakeven_cost(raw: dict[str, Any]) -> dict[str, Any]:
    """打平成本：把净超额压到 0 时，往返成本率最高能是多少。

    打平成本 = 毛超额/期 ÷ 换手/期

    含义：只要实盘实际成本低于这个数，多头组就能追平等权基准。
    这是实盘最关键的单一数字 —— 它把「策略好不好」翻译成
    「我的交易成本够不够低」。
    """
    seg = _section(raw)
    gross_excess = float((seg.get("gross") or {})["long_vs_bench_mean"])
    turnover = float(seg["mean_turnover"])
    cost_rate = float(seg["cost_rate"])

    if turnover <= 0:
        raise EvidenceMissing("换手率为 0，无法计算打平成本。")

    be = gross_excess / turnover
    headroom = (be - cost_rate) / cost_rate if cost_rate else None

    # 维持当前成本率时，多头组要追平基准所允许的最大换手率。
    # 推导：净超额 = 毛超额 - 换手×成本率 > 0
    #       → 换手 < 毛超额/成本率
    # 注意不能用 gross_excess/be —— 那是恒等于 turnover 的死循环。
    max_turnover_at_cost = gross_excess / cost_rate if cost_rate else None

    return {
        "breakeven_cost_roundtrip": be,
        "actual_cost_roundtrip": cost_rate,
        "actual_is_above_breakeven": cost_rate > be,
        "cost_gap_multiple": (cost_rate / be) if be else None,
        "headroom_ratio": headroom,
        "max_turnover_at_actual_cost": max_turnover_at_cost,
        "current_turnover": turnover,
        "turnover_reduction_needed": (
            1 - max_turnover_at_cost / turnover
            if (turnover and max_turnover_at_cost is not None) else None
        ),
    }


def three_calibers(raw: dict[str, Any]) -> dict[str, Any]:
    """三口径对照 —— 实盘预期该看哪个口径。

    主口径（缩尾均值）      可执行性最接近真实，但仍是"平均"
    未缩尾均值              披露极端值的拖累
    中位数口径              多数期数的真实结果 ← 实盘心理预期的锚

    落差 = 中位数口径年化 - 主口径年化，为负说明均值被少数极端盈利拉高。
    """
    seg = _section(raw)
    bm = seg.get("benchmark") or {}

    def row(key: str) -> dict[str, Any]:
        r = seg.get(key) or {}
        return {
            "net_annualized": r.get("net_annualized"),
            "sharpe": r.get("sharpe"),
            "win_rate": r.get("win_rate"),
        }

    primary = row("long_only")
    raw_mean = row("long_only_raw")
    median = row("long_only_median")

    spread = None
    if (primary.get("net_annualized") is not None
            and median.get("net_annualized") is not None):
        spread = median["net_annualized"] - primary["net_annualized"]

    return {
        "primary_winsor_mean": primary,
        "raw_mean": raw_mean,
        "median": median,
        "benchmark": {
            "annualized": bm.get("annualized"),
            "win_rate": bm.get("win_rate"),
        },
        "mean_minus_median_spread": spread,
        "skew_warning": (spread is not None and spread < -0.05),
    }


def benchmark_relative(raw: dict[str, Any]) -> dict[str, Any]:
    """相对基准判断 —— 这个策略到底有没有"选股 alpha"。

    多头年化 vs 等权基准年化：
      > 0  才有资格说"我赚的是选股的钱"
      < 0  赚的是 beta，而且还比 beta 少赚 —— 这时它只是"一个带择时的半吊子宽基"
    """
    seg = _section(raw)
    lo = seg.get("long_only") or {}
    bm = seg.get("benchmark") or {}
    ls = seg.get("long_short") or {}
    gross = seg.get("gross") or {}

    lo_ann = float(lo["net_annualized"])
    bm_ann = float(bm["annualized"])
    excess = lo_ann - bm_ann

    return {
        "long_only_annualized": lo_ann,
        "benchmark_annualized": bm_ann,
        "excess_annualized": excess,
        "has_alpha_vs_benchmark": excess > 0,
        "long_short_annualized": ls.get("net_annualized"),
        "long_short_sharpe": ls.get("sharpe"),
        "gross_ls_spread_per_period": gross.get("ls_excess_mean"),
        "gross_ls_spread_annualized": (
            gross.get("ls_excess_mean", 0.0) * _periods_per_year(seg)
            if gross.get("ls_excess_mean") is not None else None
        ),
        "interpretation": (
            "多头跑输等权基准 → 收益主要来自 beta 而非选股，"
            "实盘定位应为「增强器/风控工具」，不能当独立主力策略。"
            if excess <= 0 else
            "多头跑赢等权基准 → 存在真实选股 alpha，可作为独立策略候选。"
        ),
    }


def cost_scenarios(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """不同成本假设下的表现 —— 实盘选券商/选交易频率的依据。"""
    rows = raw.get("cost_sensitivity") or []
    out = []
    for r in rows:
        out.append({
            "cost_rate": r.get("cost_rate"),
            "long_only_annualized": r.get("long_only_annualized"),
            "long_only_sharpe": r.get("long_only_sharpe"),
            "excess_vs_benchmark": r.get("long_only_vs_bench_annualized"),
        })
    return out


def cross_segment_evidence() -> dict[str, Any] | None:
    """跨区段稳健性证据（可选，文件不存在则返回 None）。

    为什么需要它
    ------------
    只看测试段会给出偏乐观的结论。2026-11-12 的跨区段检验发现：
    hold=10 在测试段相对基准超额 +0.49%（全扫描唯一正超额），
    但训练段 −1.81%、验证段 −1.00% —— 符号不一致，
    属于「看过测试段之后挑出来」的噪声。

    而 hold=5 在三段全部为负（−0.82% / −3.12% / −2.03%），
    加权 −1.70% —— 符号一致，说明这是**结构性的负 alpha**，
    不是某一段行情的运气。

    本函数把这个跨段证据喂给就绪度判定，避免用单段下结论。
    """
    if not CROSS_SEGMENT.exists():
        return None
    try:
        data = json.loads(CROSS_SEGMENT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    result = data.get("result") or {}
    analysis = data.get("analysis") or {}
    rows = result.get("rows") or []

    # 取基准成本 0.30% 下、hold 与 canonical 一致的那一行
    canon_hold = None
    try:
        canon_hold = int(_section(load_oos()).get("hold") or 5)
    except Exception:
        canon_hold = 5

    row = next(
        (r for r in rows
         if r.get("hold") == canon_hold
         and abs((r.get("cost_rate") or 0) - 0.0030) < 1e-9),
        None,
    )
    if not row:
        return None

    per_seg = row.get("per_segment") or {}
    excesses = row.get("excess_by_segment") or {}

    # 按期数加权的合并超额
    weighted = None
    num = den = 0.0
    for seg in per_seg.values():
        if not seg.get("ok"):
            continue
        n = seg.get("n_periods") or 0
        num += (seg.get("excess_annualized") or 0.0) * n
        den += n
    if den:
        weighted = num / den

    signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in excesses.values()]
    consistent_negative = bool(signs) and all(s < 0 for s in signs)

    return {
        "hold": canon_hold,
        "cost_rate": row.get("cost_rate"),
        "segments": result.get("segments") or {},
        "excess_by_segment": excesses,
        "per_segment": per_seg,
        "weighted_excess": weighted,
        "all_segments_negative": consistent_negative,
        "sign_consistent": len(set(signs)) == 1 and 0 not in signs,
        "robust_holds": analysis.get("robust_holds") or [],
        "verdict": analysis.get("verdict"),
    }


# ---------------------------------------------------------------- 成本地板
# A股单笔交易的**不可压缩**成本下限。策略要盈利，毛边际必须 >= 这个数。
# 来源：财政部/税务总局公告2023年第39号（印花税0.05%卖出单边）；
#      中国结算沪深收费表2026-01-01（过户费0.01‰双向，仅沪市单列）；
#      证监发〔2002〕21号（佣金上限3‰/单边、起点5元）。
# 券商全包佣金已含经手费/证管费/深市过户费，不得重复加。
COMMISSION_RATE = 0.00025      # 万2.5
COMMISSION_MIN_YUAN = 5.0      # 单笔最低佣金（买、卖各收一次）
STAMP_DUTY_SELL = 0.0005       # 印花税，卖出单边
TRANSFER_FEE_BOTH = 0.00001    # 过户费 0.01‰，双向，仅沪市单列


def roundtrip_cost_bp(
    per_order_yuan: float,
    slippage_bp_edge: float = 0.0,
    commission_rate: float = COMMISSION_RATE,
    commission_min: float = COMMISSION_MIN_YUAN,
) -> dict[str, Any]:
    """单笔往返成本（bp），含 ¥5 最低佣金。

    佣金按 **双边各收一次** 计算。单笔越小，最低佣金占比越高，
    成本非线性上升 —— 这就是「小资金跑横截面策略必然亏」的机制。
    """
    if per_order_yuan <= 0:
        raise EvidenceMissing("单笔金额必须为正。")
    comm_one_way = max(commission_min, per_order_yuan * commission_rate)
    comm_bp = 2 * comm_one_way / per_order_yuan * 1e4
    stamp_bp = STAMP_DUTY_SELL * 1e4
    transfer_bp = 2 * TRANSFER_FEE_BOTH * 1e4
    slip_bp = 2 * slippage_bp_edge
    total = comm_bp + stamp_bp + transfer_bp + slip_bp
    return {
        "per_order_yuan": per_order_yuan,
        "commission_bp_roundtrip": comm_bp,
        "stamp_duty_bp": stamp_bp,
        "transfer_fee_bp": transfer_bp,
        "slippage_bp_roundtrip": slip_bp,
        "total_bp_roundtrip": total,
        "total_rate_roundtrip": total / 1e4,
        "below_min_commission": bool(per_order_yuan * commission_rate < commission_min),
    }


def cost_floor(
    capital_yuan: float,
    n_positions: int,
    slippage_bp_edge: float = 0.0,
) -> dict[str, Any]:
    """给定资金规模与持仓数下的成本地板。"""
    if n_positions <= 0:
        raise EvidenceMissing("持仓数必须为正。")
    per_order = capital_yuan / n_positions
    r = roundtrip_cost_bp(per_order, slippage_bp_edge)
    threshold = COMMISSION_MIN_YUAN / COMMISSION_RATE   # 2万元
    r.update({
        "capital_yuan": capital_yuan,
        "n_positions": n_positions,
        "min_order_yuan_for_rate": threshold,
        "order_ge_2wan": bool(per_order >= threshold),
    })
    return r


def execution_feasibility(
    raw: dict[str, Any],
    capital_yuan: float = 1_000_000.0,
    n_positions: int = 126,
    slippage_bp_edge: float = 5.0,
) -> dict[str, Any]:
    """把「打平成本」和「成本地板」放在一起判定。

    必须区分两件事，否则会给出错误的行动建议：
      - 实际成本 > 打平成本  -> 成本假设偏高（换券商/降换手或许能翻转）
      - 成本地板 > 打平成本  -> **结构性不可行**：零滑点、最优费率也赚不到，
                                压成本救不回来，只能换策略或提高毛边际。
    """
    be = breakeven_cost(raw)
    floor = cost_floor(capital_yuan, n_positions, slippage_bp_edge)
    be_bp = be["breakeven_cost_roundtrip"] * 1e4
    floor_bp = floor["total_bp_roundtrip"]
    struct_infeasible = floor_bp > be_bp
    if struct_infeasible:
        verdict = "structurally_infeasible"
        reason = (
            f"成本地板 {floor_bp:.2f}bp > 打平成本 {be_bp:.2f}bp —— "
            f"即便零滑点、最优费率也无法覆盖成本。"
        )
    elif be["actual_cost_roundtrip"] * 1e4 > be_bp:
        verdict = "cost_assumption_too_high"
        reason = (
            f"成本地板 {floor_bp:.2f}bp <= 打平成本 {be_bp:.2f}bp，"
            f"但当前假设 {be['actual_cost_roundtrip']*1e4:.2f}bp 高于打平线 —— "
            f"降低真实交易成本即可翻转。"
        )
    else:
        verdict = "feasible"
        reason = f"成本地板 {floor_bp:.2f}bp <= 打平成本 {be_bp:.2f}bp，且实际成本假设已低于打平线。"
    return {
        "breakeven_cost_bp": be_bp,
        "cost_floor_bp": floor_bp,
        "cost_floor": floor,
        "structurally_infeasible": struct_infeasible,
        "verdict": verdict,
        "reason": reason,
        "edge_gap_bp": floor_bp - be_bp,
    }


def readiness_verdict(raw: dict[str, Any]) -> dict[str, Any]:
    """综合就绪度判定 —— 能不能上实盘。

    判定项（全部基于上面各项的实测量，不是主观打分）：
      C1 成本可行性：实际成本 < 打平成本
      C2 相对基准：净超额 > 0
      C3 分布稳健：中位数口径不至于显著为负
      C4 恒等式自洽：成本分解残差为 0（口径没串）
      C5 跨段一致性：各区段超额同号（非样本内挑选的噪声）
    """
    cost = cost_decomposition(raw)
    be = breakeven_cost(raw)
    cal = three_calibers(raw)
    rel = benchmark_relative(raw)
    cross = cross_segment_evidence()

    feas = execution_feasibility(raw)
    checks = [
        {
            "id": "C1",
            "name": "成本可行性",
            "pass": (not be["actual_is_above_breakeven"]) and (not feas["structurally_infeasible"]),
            "detail": (
                f"打平成本 {be['breakeven_cost_roundtrip']*100:.4f}% 往返，"
                f"实际假设 {be['actual_cost_roundtrip']*100:.2f}%，"
                f"高出 {be['cost_gap_multiple']:.2f} 倍。"
                + (
                    f" 且成本地板 {feas['cost_floor_bp']:.2f}bp 已高于打平成本 "
                    f"{feas['breakeven_cost_bp']:.2f}bp —— 结构性不可行。"
                    if feas["structurally_infeasible"] else ""
                )
            ),
            "fix": (
                "结构性不可行：成本地板高于打平成本，压交易成本救不回来 —— "
                "必须提高毛边际（换因子/换持有期/换标的）或放弃该策略。"
                if feas["structurally_infeasible"] else
                "压低交易成本或降低换手率 —— 这是唯一能翻转结论的杠杆。"
            ),
        },
        {
            "id": "C2",
            "name": "相对基准有 alpha",
            "pass": rel["has_alpha_vs_benchmark"],
            "detail": (
                f"多头 {rel['long_only_annualized']*100:+.2f}% vs "
                f"等权基准 {rel['benchmark_annualized']*100:+.2f}%，"
                f"超额 {rel['excess_annualized']*100:+.2f}%。"
            ),
            "fix": "换因子集或换选股逻辑；或改定位为「增强器」而非独立策略。",
        },
        {
            "id": "C3",
            "name": "收益分布稳健",
            "pass": not cal["skew_warning"],
            "detail": (
                f"均值口径 {cal['primary_winsor_mean']['net_annualized']*100:+.2f}%，"
                f"中位数口径 {cal['median']['net_annualized']*100:+.2f}%，"
                f"落差 {cal['mean_minus_median_spread']*100:+.2f} 个百分点。"
            ),
            "fix": "实盘仓位按中位数口径设计，不要按均值口径设计。",
        },
        {
            "id": "C4",
            "name": "口径自洽",
            "pass": cost["identity_holds"],
            "detail": (
                f"成本分解残差 {cost['identity_residual']:.3e}（应为 0）。"
            ),
            "fix": "若不为 0，说明成本或换手口径串了，先修口径再谈策略。",
        },
    ]

    # C5 跨段一致性 —— 有跨段证据才判，没有就不判（不编造）
    if cross:
        ex = cross["excess_by_segment"]
        cells = "，".join(
            f"{k} {v*100:+.2f}%" for k, v in ex.items()
        )
        checks.append({
            "id": "C5",
            "name": "跨区段一致（非样本内挑选）",
            "pass": cross["sign_consistent"] and not cross["all_segments_negative"],
            "detail": (
                f"hold {cross['hold']} 日在训练/验证/测试三段相对基准超额：{cells}。"
                + (f" 加权 {cross['weighted_excess']*100:+.2f}%。" if cross["weighted_excess"] is not None else "")
                + (" 三段同号，说明是结构性结论而非运气。"
                   if cross["sign_consistent"] else
                   " 符号不一致，存在事后挑选嫌疑。")
            ),
            "fix": (
                "三段一致为负 = 扣成本后系统性跑输基准，"
                "必须换因子/换逻辑或放弃独立策略定位。"
                if cross["all_segments_negative"] else
                "在训练段与验证段也验证同一结论再谈实盘。"
            ),
        })

    # C6 成本地板可达性 —— 「换再好的券商也救不回来」必须单独成项，
    # 否则 C1 失败会被误读成「成本假设偏高、可优化」。
    checks.append({
        "id": "C6",
        "name": "成本地板可达性（结构性）",
        "pass": not feas["structurally_infeasible"],
        "detail": (
            feas["reason"]
            + f" 单笔 {feas['cost_floor']['per_order_yuan']:.0f} 元"
            + ("已达" if feas["cost_floor"]["order_ge_2wan"] else "未达")
            + "2万元费率门槛。"
        ),
        "fix": (
            f"毛边际必须提高到成本地板之上（当前差 {feas['edge_gap_bp']:.2f}bp/期），"
            f"否则任何执行优化都是白费。"
        ),
    })

    npass = sum(1 for c in checks if c["pass"])

    # 判定必须看「失败的性质」，不能只看通过数。
    # 2026-11-12 教训：旧版只按通过数分级，出现「1/5 通过」却给出
    # 「有条件实盘：小额试水」的文案 —— 而 C5 已证明三段全部跑输基准，
    # 那是结构性负 alpha，拿真钱试水等于送钱。必须改为按阻断项分级。
    #
    # 阻断项（BLOCKING）：任一失败 = 不具备实盘条件
    #   C1 成本高于打平线      -> 实盘扣完成本必然亏
    #   C2 跑输等权基准        -> 没有 alpha，不如直接买指数
    #   C3 中位数口径为负      -> 多数月份是亏的
    #   C5 三段一致为负        -> 结构性负 alpha，不是运气
    # 非阻断（SOFT）：C4 口径不自洽 -> 数字本身不可信，先修口径
    blocking_ids = {"C1", "C2", "C3", "C5", "C6"}
    blocking_failed = [c["id"] for c in checks if not c["pass"] and c["id"] in blocking_ids]
    soft_failed = [c["id"] for c in checks if not c["pass"] and c["id"] not in blocking_ids]

    if npass == len(checks):
        verdict = "可进小额实盘验证"
    elif blocking_failed:
        verdict = (
            "不具备实盘条件 —— "
            + "、".join(blocking_failed)
            + f" 未通过，属结构性阻断，不得投入真钱"
        )
    else:
        # 只有 C4 这类「数字本身不可信」的问题
        verdict = (
            "暂不判定 —— "
            + "、".join(soft_failed)
            + " 未通过，先修口径再谈实盘"
        )

    return {
        "checks": checks,
        "passed": npass,
        "total": len(checks),
        "verdict": verdict,
        "blocking_failed": blocking_failed,
        "soft_failed": soft_failed,
        "hard_block": bool(blocking_failed),
        "cross_segment": cross,
    }


def save_result(result: dict[str, Any], path: Path | None = None) -> Path:
    """把体检结果落盘，带 generated_at。

    2026-11-16 加：以前 full_check() 只在内存里算，从不落盘。
    后果有二：
      ① 日报页面无法给「实盘就绪度体检」标出数字出处与时间；
      ② 同一份体检结果无法事后复现（今天算出来的 1/6 通过，
         明天面板长一天可能变成别的数，但没人能核对当时按哪版算的）。

    落盘纪律：
      - 顶层必须有 generated_at（否则又是一份「看不出新旧」的产物）；
      - 原子写（先写临时文件再 rename），避免读到半截 JSON；
      - 落盘失败**不能**让体检崩掉 —— 体检本身是计算，
        写文件是副产物。
    """
    target = Path(path) if path is not None else RESEARCH / "live_readiness.json"
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(target)
    except Exception as exc:  # 落盘是副产物，失败不阻断体检
        print(f"[live_readiness] 落盘失败（不影响体检结果）：{exc}")
    return target


def full_check() -> dict[str, Any]:
    """一次跑完全部体检项。"""
    raw = load_oos()
    # 面板截止日：从 canonical oos 产物里取。
    # 体检数字全挂在这份 oos 上，不标截止日就无法区分
    # 「指标变差了」和「数据截止日往后挪了一天」。
    _panel_to = None
    try:
        _pn = raw.get("panel") or {}
        if isinstance(_pn, dict):
            _panel_to = _pn.get("to")
        if _panel_to is None:
            _segs = raw.get("segments") or {}
            if isinstance(_segs, dict):
                _t = _segs.get("test")
                if isinstance(_t, (list, tuple)) and len(_t) >= 2:
                    _panel_to = _t[1]
    except Exception:
        _panel_to = None
    return {
        "source": str(OOS_LATEST),
        "panel_to": _panel_to,
        "cost_decomposition": cost_decomposition(raw),
        "breakeven_cost": breakeven_cost(raw),
        "execution_feasibility": execution_feasibility(raw),
        "three_calibers": three_calibers(raw),
        "benchmark_relative": benchmark_relative(raw),
        "cost_scenarios": cost_scenarios(raw),
        "readiness": readiness_verdict(raw),
    }


def format_report(result: dict[str, Any]) -> str:
    """生成纯文字体检报告（宋老师阅读格式：缩进 2 空格，数字加粗）。"""
    cost = result["cost_decomposition"]
    be = result["breakeven_cost"]
    cal = result["three_calibers"]
    rel = result["benchmark_relative"]
    rd = result["readiness"]

    L: list[str] = []
    L.append("【实盘就绪度体检】")
    L.append(f"  判定：{rd['verdict']}（{rd['passed']}/{rd['total']} 项通过）")
    L.append("")
    L.append("  一、成本结构分解")
    L.append(f"    毛超额（未扣成本）  **{cost['annualized']['gross_excess']*100:+.2f}%**/年")
    # 成本拖累是「正数 = 拖累了多少」，显示时用负号避免读成"贡献"
    L.append(f"    成本拖累            **{-cost['annualized']['cost_drag']*100:+.2f}%**/年"
             f"（换手 {cost['turnover_per_period']*100:.2f}%/期 × 成本 "
             f"{cost['cost_rate_roundtrip']*100:.2f}%）")
    L.append(f"    净超额              **{cost['annualized']['net_excess']*100:+.2f}%**/年")
    L.append(f"    口径自洽残差        {cost['identity_residual']:.3e}"
             f"（{'一致' if cost['identity_holds'] else '不一致，需修口径'}）")
    L.append("")
    L.append("  二、打平成本（实盘最关键单一数字）")
    L.append(f"    打平成本            **{be['breakeven_cost_roundtrip']*100:.4f}%** 往返")
    L.append(f"    实际假设            {be['actual_cost_roundtrip']*100:.2f}%"
             f"（{'高于' if be['actual_is_above_breakeven'] else '低于'}打平线 "
             f"{be['cost_gap_multiple']:.2f} 倍）")
    if be["turnover_reduction_needed"] is not None:
        L.append(f"    若维持 {be['actual_cost_roundtrip']*100:.2f}% 成本，"
                 f"换手需降至 {be['max_turnover_at_actual_cost']*100:.2f}%/期"
                 f"（当前 {be['current_turnover']*100:.2f}%，"
                 f"需降 {be['turnover_reduction_needed']*100:.1f}%）")
    L.append("")
    L.append("  三、三口径对照")
    L.append(f"    主口径（缩尾均值）  年化 **{cal['primary_winsor_mean']['net_annualized']*100:+.2f}%**"
             f"  Sharpe {cal['primary_winsor_mean']['sharpe']:.3f}")
    L.append(f"    未缩尾均值          年化 {cal['raw_mean']['net_annualized']*100:+.2f}%"
             f"  Sharpe {cal['raw_mean']['sharpe']:.3f}")
    L.append(f"    中位数口径          年化 **{cal['median']['net_annualized']*100:+.2f}%**"
             f"  Sharpe {cal['median']['sharpe']:.3f}")
    L.append(f"    等权基准            年化 {cal['benchmark']['annualized']*100:+.2f}%")
    L.append(f"    均值-中位数落差     {cal['mean_minus_median_spread']*100:+.2f} 个百分点"
             f"（{'均值被极端值拉高，实盘预期应看中位数' if cal['skew_warning'] else '分布较稳'}）")
    L.append("")
    L.append("  四、相对基准")
    L.append(f"    多头 {rel['long_only_annualized']*100:+.2f}% vs "
             f"基准 {rel['benchmark_annualized']*100:+.2f}%  →  超额 "
             f"**{rel['excess_annualized']*100:+.2f}%**")
    L.append(f"    多空组合净年化 {rel['long_short_annualized']*100:+.2f}%"
             f"（毛价差年化 {rel['gross_ls_spread_annualized']*100:+.2f}%）")
    L.append(f"    结论：{rel['interpretation']}")
    L.append("")
    L.append("  五、成本情景")
    for s in result["cost_scenarios"]:
        L.append(f"    成本 {s['cost_rate']*100:.2f}%  →  多头年化 "
                 f"{s['long_only_annualized']*100:+.2f}%  Sharpe "
                 f"{s['long_only_sharpe']:.3f}  超额 "
                 f"{s['excess_vs_benchmark']*100:+.2f}%")
    L.append("")
    L.append("  六、就绪度逐项判定")
    for c in rd["checks"]:
        mark = "通过" if c["pass"] else "未通过"
        L.append(f"    [{c['id']}] {c['name']}：{mark}")
        L.append(f"        {c['detail']}")
        if not c["pass"]:
            L.append(f"        修法：{c['fix']}")
    cross = rd.get("cross_segment")
    if cross:
        L.append("")
        L.append("  七、跨区段稳健性（防止样本内挑选）")
        for seg, v in cross["excess_by_segment"].items():
            ps = (cross.get("per_segment") or {}).get(seg) or {}
            L.append(f"    {seg:<6} 超额 {v*100:>+7.2f}%  "
                     f"（换手 {ps.get('mean_turnover', 0)*100:.2f}%/期  "
                     f"期数 {ps.get('n_periods')}  "
                     f"打平成本 {ps.get('breakeven_cost', 0)*100:.4f}%）")
        if cross["weighted_excess"] is not None:
            L.append(f"    按期数加权合并超额 **{cross['weighted_excess']*100:+.2f}%**"
                     f"（共 {sum((cross.get('per_segment') or {}).get(s, {}).get('n_periods', 0) for s in cross['excess_by_segment'])} 期）")
        L.append(f"    跨段同号：{'是' if cross['sign_consistent'] else '否'}；"
                 f"三段全为负：{'是' if cross['all_segments_negative'] else '否'}")
    L.append("")
    L.append("  数据来源：data/research/oos_latest.json（由 ooslab 生成）")
    return "\n".join(L)


if __name__ == "__main__":
    _res = full_check()
    print(format_report(_res))
    # 落盘：让日报页面能标出「实盘就绪度体检」的数字出处与时间。
    _p = save_result(_res)
    print(f"\n已存：{_p}")
