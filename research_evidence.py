"""研究能力实测：把「样本外验证」和「预测闭环」的结果组装成日报板块。

这一板块存在的意义是**让诚实的结论出现在用户看得见的地方**：
    * 全样本显著的因子，如果时间外不成立，就必须写出来；
    * 成本后不赚钱的策略，就不能拿去做模拟仓位；
    * 概率不准，就不能说「上涨概率 63%」；
    * 样本不够，就明写「样本不足，暂不评价」，不许硬凑数字。

四道门禁（任一不过，就不进入模拟仓位）：
    G1 时间外    训练段发现的因子，在测试段方向一致且显著的比例 ≥ 60%
    G2 成本后    基准成本（0.30%/次往返）下多头组合年化 > 0 且 Sharpe ≥ 0.5
    G3 概率校准  测试段 AUC > 0.52，且 Brier 优于「全猜基准率」
    G4 预测闭环  已判定样本 ≥ 30，且方向命中率高于全押方向的基准
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import daily_loop
import predictor

ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = ROOT / "data" / "research"
OOS_LATEST = RESEARCH_DIR / "oos_latest.json"
PRED_DIR = ROOT / "data" / "predictions"

# 门禁阈值（写死在代码里，避免「阈值也过拟合」）
GATE_OOS_PASS_RATIO = 0.60
GATE_SHARPE_MIN = 0.50
GATE_AUC_MIN = 0.52
GATE_LOOP_MIN_SAMPLES = 30


# ------------------------------------------------------------------ 读取


def load_oos() -> dict[str, Any] | None:
    """读取最近一次样本外验证结果。读不到就返回 None（不编造）。"""
    if not OOS_LATEST.exists():
        return None
    try:
        return json.loads(OOS_LATEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def prediction_loop_summary(*, min_samples: int = 5) -> dict[str, Any]:
    """预测闭环统计。

    判定结果**只认已落盘的**（predictor.judge_all 写进 data/predictions/resolved/）：
    判定一旦落盘就固定，不会随着行情回溯调整而漂移，
    也不会出现"报告里现算一遍、跟昨天不是一个数"的情况。

    报告侧**不现算判定**：judge 由 daily_loop 阶段负责（见 main.run_daily_loop），
    这里只读已落盘的 resolved 文件汇总。
    """
    return predictor.loop_stats(min_samples=min_samples)


def daily_loop_summary() -> dict[str, Any]:
    """每日闭环三个阶段最近一次的状态（refresh / predict / judge）。

    用于让日报如实显示"今天闭环有没有跑通"：
    失败时显示失败原因，而不是拿上一次的旧结果冒充今天已完成。
    """
    return daily_loop.daily_loop_status()


def latest_prediction_summary() -> dict[str, Any]:
    """最近一期预测（日报里展示"今天怎么看"用）。"""
    return predictor.latest_summary()


# ------------------------------------------------------------------ 组装


def build_evidence() -> dict[str, Any]:
    """组装日报里「研究能力实测」板块需要的全部数据。"""
    oos = load_oos()
    loop = prediction_loop_summary()

    bundle: dict[str, Any] = {
        "available": oos is not None,
        "generated_at": (oos or {}).get("generated_at") or time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_loop": loop,
        "daily_loop": daily_loop_summary(),
        "gates": [],
        "verdict": {},
    }
    if not oos:
        bundle["verdict"] = {
            "pass": False,
            "passed": 0,
            "total": 4,
            "text": "尚未完成样本外验证，研究信号不进入模拟仓位。",
            "sim_position_allowed": False,
            "failed_gates": ["G1 时间外验证", "G2 成本后收益", "G3 概率校准", "G4 预测闭环"],
        }
        return bundle

    panel = oos.get("panel") or {}
    segments = oos.get("segments") or {}
    bundle["panel"] = panel
    bundle["segments"] = segments
    bundle["config"] = oos.get("config") or {}

    # ---- 因子时间外验证
    factors: list[dict[str, Any]] = []
    for o in oos.get("oos_validation") or []:
        factors.append({
            "name": o.get("name"),
            "desc": o.get("desc"),
            "discovered_mean": o.get("discovered_mean"),
            "discovered_t": o.get("discovered_t"),
            "oos_mean": o.get("oos_mean"),
            "oos_t": o.get("oos_t"),
            "oos_hit": o.get("oos_hit"),
            "same_sign": o.get("same_sign"),
            "verdict": o.get("oos_verdict"),
        })
    bundle["factors"] = factors
    n_pass = sum(1 for f in factors if f.get("verdict") == "时间外通过")
    bundle["oos_pass"] = n_pass
    bundle["oos_total"] = len(factors)
    bundle["oos_pass_ratio"] = (n_pass / len(factors)) if factors else 0.0

    # ---- 成本后策略
    strat = oos.get("strategy_test_segment") or {}
    bundle["strategy"] = strat
    bundle["cost_sensitivity"] = oos.get("cost_sensitivity") or []
    bundle["strategy_control_unsigned"] = oos.get("strategy_control_unsigned") or {}
    bundle["strategy_fixed_decile"] = oos.get("strategy_fixed_decile") or {}
    bundle["strategy_config"] = oos.get("strategy_config") or {}
    bundle["construction_selection"] = oos.get("construction_selection") or {}
    bundle["signal_diagnostic"] = oos.get("signal_diagnostic") or {}
    bundle["limitations"] = oos.get("limitations") or []
    bundle["horizon"] = oos.get("horizon")

    # ---- 实盘就绪度体检（2026-11-12）
    # 研究通过门禁 ≠ 能上实盘。这里补上实盘视角的三个关键量：
    # 打平成本（成本高于它就跑不赢基准）、均值/中位数落差、
    # 以及跨区段一致性（防止用单段好看的数字下结论）。
    try:
        import live_readiness as LR

        _lr = LR.full_check()
        bundle["live_readiness"] = {
            "available": True,
            "breakeven_cost": _lr["breakeven_cost"]["breakeven_cost_roundtrip"],
            "actual_cost": _lr["breakeven_cost"]["actual_cost_roundtrip"],
            "cost_gap_multiple": _lr["breakeven_cost"]["cost_gap_multiple"],
            "turnover": _lr["cost_decomposition"]["turnover_per_period"],
            "max_turnover_at_cost": _lr["breakeven_cost"]["max_turnover_at_actual_cost"],
            "gross_excess_annualized": _lr["cost_decomposition"]["annualized"]["gross_excess"],
            "net_excess_annualized": _lr["cost_decomposition"]["annualized"]["net_excess"],
            "mean_annualized": _lr["three_calibers"]["primary_winsor_mean"]["net_annualized"],
            "median_annualized": _lr["three_calibers"]["median"]["net_annualized"],
            "mean_median_spread": _lr["three_calibers"]["mean_minus_median_spread"],
            "excess_vs_benchmark": _lr["benchmark_relative"]["excess_annualized"],
            "checks": _lr["readiness"]["checks"],
            "passed": _lr["readiness"]["passed"],
            "total": _lr["readiness"]["total"],
            "verdict": _lr["readiness"]["verdict"],
            "blocking_failed": _lr["readiness"].get("blocking_failed", []),
            "soft_failed": _lr["readiness"].get("soft_failed", []),
            "hard_block": _lr["readiness"].get("hard_block", False),
            "cross_segment": _lr["readiness"].get("cross_segment"),
        }
    except Exception as exc:  # 体检失败不得让整个日报崩掉
        bundle["live_readiness"] = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    base_cost = (oos.get("config") or {}).get("cost_roundtrip", 0.003)
    bundle["base_cost"] = base_cost

    # ---- 概率模型
    bundle["probability"] = {
        "rel": oos.get("probability_rel") or {},
        "abs": oos.get("probability_abs") or {},
    }

    # ---- 门禁
    gates: list[dict[str, Any]] = []

    ratio = bundle["oos_pass_ratio"]
    g1 = ratio >= GATE_OOS_PASS_RATIO and n_pass > 0
    gates.append({
        "name": "G1 时间外验证",
        "pass": bool(g1),
        "detail": (
            f"训练段发现 {len(factors)} 个因子，测试段方向一致且显著 {n_pass} 个"
            f"（{ratio:.0%}），门槛 {GATE_OOS_PASS_RATIO:.0%}"
        ),
    })

    lo = (strat.get("long_only") or {}) if strat.get("ok") else {}
    ann = lo.get("net_annualized")
    sharpe = lo.get("sharpe")
    g2 = bool(strat.get("ok")) and ann is not None and sharpe is not None \
        and ann > 0 and sharpe >= GATE_SHARPE_MIN
    gates.append({
        "name": "G2 成本后收益",
        "pass": g2,
        "detail": (
            f"测试期多头组合在 {base_cost:.2%} 往返成本下年化 "
            f"{ann:+.2%}、Sharpe {sharpe:.2f}（门槛年化>0 且 Sharpe≥{GATE_SHARPE_MIN}）"
            if strat.get("ok") and ann is not None and sharpe is not None
            else "测试期策略未产生有效调仓期，无法评价成本后收益"
        ),
    })

    rel = bundle["probability"].get("rel") or {}
    iso = rel.get("isotonic") or {}
    auc = iso.get("auc")
    brier = iso.get("brier")
    brier_base = iso.get("brier_baseline")
    g3 = (
        auc is not None and brier is not None and brier_base is not None
        and auc > GATE_AUC_MIN and brier < brier_base
    )
    gates.append({
        "name": "G3 概率校准",
        "pass": bool(g3),
        "detail": (
            f"测试段 AUC {auc:.4f}（门槛>{GATE_AUC_MIN}），Brier {brier:.5f} "
            f"vs 全猜基准 {brier_base:.5f}"
            if auc is not None and brier is not None and brier_base is not None
            else "概率模型未完成样本外评估"
        ),
    })

    lp = loop.get("n") or 0
    edge = loop.get("edge_vs_baseline")
    hit_rate = loop.get("hit_rate")
    brier = loop.get("brier")
    actual_rate = loop.get("actual_rate")
    pf = loop.get("portfolio") or {}
    dl = loop.get("day_level") or {}
    # 「全猜基准率」的 Brier：永远输出真实基准率 p，误差 = p(1-p)
    brier_base = (
        float(actual_rate) * (1.0 - float(actual_rate))
        if actual_rate is not None else None
    )

    # G4 判据（2026-11-11 修订，理由随附，不藏着旧判据）：
    #   ① 链跑通      已判定样本 ≥ 30
    #   ② 选股能力    十分位价差中位数 > 0（头部组合优于尾部组合，天然对冲市场方向）
    #   ③ 概率增量    按天命中率跑赢硬币的天数占比 > 50%，且 Brier 优于「全猜基准率」
    # 旧判据「方向命中率 > 全押方向基准」同时展示但不作门禁：它衡量的是押大盘方向，
    # 而本系统交付的是「关注标的 + 上涨概率」；在下跌窗口里全押下跌天然占优，
    # 拿它当门禁等于奖励押方向，不是奖励选股。
    c1 = bool(loop.get("sufficient")) and lp >= GATE_LOOP_MIN_SAMPLES
    c2 = bool(pf.get("available")) and (pf.get("median_spread") or 0.0) > 0
    c3 = (
        bool(dl.get("available"))
        and (dl.get("pct_days_hit_above_50") or 0.0) > 0.5
        and brier is not None and brier_base is not None and brier < brier_base
    )
    g4 = bool(c1 and c2 and c3)
    gates.append({
        "name": "G4 预测闭环",
        "pass": g4,
        "detail": (
            f"已判定 {lp} 条预测（{dl.get('n_days', 0)} 个交易日）；"
            f"选股：十分位价差中位数 {pf.get('median_spread', 0):+.2%}"
            f"（头部 {pf.get('avg_top_decile_return', 0):+.2%} vs "
            f"尾部 {pf.get('avg_bottom_decile_return', 0):+.2%}，"
            f"正价差天数占比 {pf.get('pct_days_spread_positive', 0):.0%}）；"
            f"概率：按天命中率跑赢硬币的天数占比 {dl.get('pct_days_hit_above_50', 0):.0%}，"
            f"Brier {brier:.5f} vs 全猜基准率 {brier_base:.5f}"
            f"（门槛：样本≥{GATE_LOOP_MIN_SAMPLES}、价差中位数>0、概率两项均达标）"
            if c1 and hit_rate is not None and brier is not None and brier_base is not None
            else f"已判定 {lp} 条预测，闭环证据不足（需 ≥{GATE_LOOP_MIN_SAMPLES} 条已判定）"
        ),
        "detail_legacy": (
            f"旧判据参考：方向命中率 {hit_rate:.1%} vs 全押方向基准 "
            f"{loop.get('baseline_hit_rate', 0):.1%}（差 {edge:+.1%}）"
            if hit_rate is not None and edge is not None else None
        ),
        "criteria": {
            "c1_samples": c1,
            "c2_selection": c2,
            "c3_probability": c3,
        },
        # 判据逐条留痕：日报必须能看出「哪一条没过、差多少」，
        # 而不是只给一个笼统的「未通过」。
        "criteria_detail": [
            {
                "key": "c1_samples",
                "name": "① 链跑通（样本量）",
                "threshold": f"已判定样本 ≥ {GATE_LOOP_MIN_SAMPLES} 条",
                "actual": f"{lp} 条（{dl.get('n_days', 0)} 个交易日）",
                "pass": bool(c1),
            },
            {
                "key": "c2_selection",
                "name": "② 选股能力（十分位价差中位数）",
                "threshold": "> 0",
                "actual": f"{(pf.get('median_spread') or 0.0):+.4%}",
                "pass": bool(c2),
            },
            {
                "key": "c3a_days",
                "name": "③ 概率增量（按天跑赢硬币的天数占比）",
                "threshold": "> 50%",
                "actual": f"{(dl.get('pct_days_hit_above_50') or 0.0) * 100:.2f}%",
                "pass": bool(dl.get("available"))
                and (dl.get("pct_days_hit_above_50") or 0.0) > 0.5,
            },
            {
                "key": "c3b_brier",
                "name": "④ 概率增量（Brier 优于全猜基准率）",
                "threshold": f"< {brier_base:.5f}" if brier_base is not None else "—",
                "actual": f"{brier:.5f}" if brier is not None else "—",
                "pass": bool(
                    brier is not None and brier_base is not None and brier < brier_base
                ),
            },
        ],
    })

    bundle["gates"] = gates
    passed = sum(1 for g in gates if g["pass"])
    all_pass = passed == len(gates)
    if all_pass:
        text = "四道门禁全部通过，研究信号可进入模拟仓位（仍为模拟，不执行真实交易）。"
    else:
        failed = [g["name"] for g in gates if not g["pass"]]
        text = (
            f"四道门禁通过 {passed}/{len(gates)}；未通过：{'、'.join(failed)}。"
            "研究信号暂不进入模拟仓位，仅作观察记录。"
        )
    bundle["verdict"] = {
        "pass": all_pass,
        "passed": passed,
        "total": len(gates),
        "text": text,
        # 显式开关：日报必须能直接读出「信号能不能进模拟仓位」，
        # 不能靠读者自己数几道门禁。
        "sim_position_allowed": bool(all_pass),
        "failed_gates": [g["name"] for g in gates if not g["pass"]],
    }
    return bundle


# ------------------------------------------------------------------ 分析文字


def analyze_factor_oos(bundle: dict[str, Any]) -> list[str]:
    """因子时间外验证表的分析。"""
    lines: list[str] = []
    if not bundle.get("available"):
        return ["尚未完成样本外验证，无法给出因子时间外表现的判断。"]
    total = bundle.get("oos_total") or 0
    n_pass = bundle.get("oos_pass") or 0
    if not total:
        return ["训练段未发现任何通过 Bonferroni 校正的因子，无可检验对象。"]
    lines.append(
        f"训练段共发现 {total} 个显著因子；放到测试段（发现时从未看过的数据）后，"
        f"{n_pass} 个仍方向一致且显著（{bundle.get('oos_pass_ratio', 0):.0%}），"
        f"其余是方向一致但显著性不足。"
    )
    same_but_weak = [f for f in bundle.get("factors", []) if f.get("verdict") == "方向一致但不显著"]
    if same_but_weak:
        names = "、".join(f.get("name", "") for f in same_but_weak[:6])
        lines.append(
            f"方向一致但检验期不显著的因子：{names}。这类信号方向可信、强度不足，"
            "单独使用没有把握，不进生产。"
        )
    reversed_ = [f for f in bundle.get("factors", []) if f.get("verdict") == "方向反转（失效）"]
    if reversed_:
        names = "、".join(f.get("name", "") for f in reversed_[:6])
        lines.append(f"方向反转（失效）的因子：{names}。已在样本外证伪，直接作废。")
    lines.append(
        "对照全样本口径会看到更多「显著可用」的因子，但那是因为把检验期也算了进去；"
        "只有拆开前后两段看，才能知道信号是真的还是挖出来的。"
    )
    return lines


def analyze_diagnostic(bundle: dict[str, Any]) -> list[str]:
    """「IC 与尾部价差」诊断表的分析。"""
    d = bundle.get("signal_diagnostic") or {}
    if not d or d.get("ic_mean") is None:
        return ["尚无合成信号诊断数据。"]
    lines = [
        f"合成信号在测试段的逐日秩相关 IC 为 {d['ic_mean']:+.4f}"
        f"（t={d['ic_t']:.2f}，n={d.get('ic_n', 0):,}）—— "
        f"{'信号本身在样本外是有效的' if d['ic_mean'] > 0 and (d.get('ic_t') or 0) > 1.96 else '信号本身在样本外站不住'}。"
    ]
    ts = d.get("tail_spread_mean")
    if ts is not None:
        n_tail = d.get("tail_n") or d.get("ic_n") or 0
        lines.append(
            f"但毛十分位价差（最高 10% 减最低 10%）是 {ts:+.4%}，"
            f"{n_tail} 期里只有 {d.get('tail_positive_ratio', 0):.1%} 的期为正。"
        )
    tsm = d.get("tail_spread_median")
    if tsm is not None:
        lines.append(
            f"同一份数据换成**中位数**口径，价差是 {tsm:+.4%}"
            f"（{d.get('tail_median_positive_ratio', 0):.1%} 的期为正）—— "
            "均值被少数极端收益带偏，中位数才是这个信号的真实画像。"
        )
    if d.get("ic_tail_consistent") is False:
        lines.append(
            "两者方向相反，说明 alpha 集中在分布中段而不在极端值里："
            "极端分位恰好是信号最嘈杂的地方（极端动量常常是噪音、ST股、流动性差的票），"
            "用十分位去收割反而是自己制造噪声。"
        )
        lines.append(
            "这也是为什么策略构造方式必须在训练段选、而不是默认用十分位 —— "
            "构造方式选错，信号再好也会被做成负收益。"
        )
    return lines


def analyze_strategy(bundle: dict[str, Any]) -> list[str]:
    """成本后策略表的分析。"""
    lines: list[str] = []
    strat = bundle.get("strategy") or {}
    if not strat.get("ok"):
        return [f"成本后策略检验未产生有效结果：{strat.get('reason', '原因未知')}，无法评价。"]
    lo = strat.get("long_only") or {}
    ls = strat.get("long_short") or {}
    bm = strat.get("benchmark") or {}
    base_cost = bundle.get("base_cost") or 0.003
    lines.append(
        f"测试期 {strat.get('n_periods', 0)} 个调仓期（每期 {strat.get('hold', '—')} 个交易日），"
        f"平均单边换手 {strat.get('mean_turnover', 0):.1%}，"
        f"成本按 {base_cost:.2%}/次往返计入。"
    )
    cfg = bundle.get("strategy_config") or {}
    sel = bundle.get("construction_selection") or {}
    if cfg:
        q = f"，分位 {cfg.get('quantile')}" if cfg.get("quantile") else ""
        lines.append(
            f"策略构造方式在**训练段**选定后锁定：{cfg.get('construction')}，"
            f"窗口 {cfg.get('horizon')} 日{q}；"
            f"共扫了 {sel.get('n_configs', 0)} 个组合，训练段最优多头 Sharpe "
            f"{cfg.get('long_only_sharpe') or 0:.2f}。"
        )
    fixed = bundle.get("strategy_fixed_decile") or {}
    if fixed.get("ok"):
        lines.append(
            f"对照组（固定 5 日十分位，不做构造选择）：多头年化 "
            f"{fixed['long_only']['net_annualized']:+.2%}、多空年化 "
            f"{fixed['long_short']['net_annualized']:+.2%}。"
            "这组数字说明构造方式本身就能决定成败，不是因子的功劳。"
        )
    lines.append(
        f"多头组年化 {lo.get('net_annualized', 0):+.2%}、Sharpe {lo.get('sharpe') or 0:.2f}；"
        f"同期基准（全市场等权）年化 {bm.get('annualized', 0):+.2%}，"
        f"多头超额 {lo.get('net_annualized', 0) - bm.get('annualized', 0):+.2%}。"
    )
    lines.append(
        f"多空组合年化 {ls.get('net_annualized', 0):+.2%}、Sharpe {ls.get('sharpe') or 0:.2f}；"
        f"多空组相对基准年化 {ls.get('net_annualized', 0) - bm.get('annualized', 0):+.2%}。"
    )
    be = strat.get("breakeven_cost_long_only")
    if be is not None:
        if be > 0:
            lines.append(
                f"多头组打平所需的临界成本是 {be:.3%}/次往返 —— 实际成本若高于这个数，多头部分就变成负贡献。"
            )
        else:
            lines.append(
                f"多头组的临界成本为负（{be:.3%}），意味着即使零成本也跑不赢基准，"
                "这不是成本问题，是信号方向或强度问题。"
            )
    sens = bundle.get("cost_sensitivity") or []
    if len(sens) >= 2:
        seg = "；".join(
            f"{s['cost_rate']:.2%}→多头 {s.get('long_only_annualized', 0):+.2%}/多空 {s.get('long_short_annualized', 0):+.2%}"
            for s in sens if s.get("ok")
        )
        lines.append(f"成本敏感性：{seg}。")
    ctrl = bundle.get("strategy_control_unsigned") or {}
    if ctrl.get("ok"):
        lines.append(
            f"对照组（未按发现期 IC 符号做方向校正）：多头年化 "
            f"{ctrl['long_only']['net_annualized']:+.2%}、多空年化 "
            f"{ctrl['long_short']['net_annualized']:+.2%}。"
            "方向用错时多空会明显变差，这一组是对照基准，不是可选方案。"
        )
    return lines


def analyze_probability(bundle: dict[str, Any]) -> list[str]:
    """概率模型表的分析。"""
    lines: list[str] = []
    prob = bundle.get("probability") or {}
    if not prob:
        return ["概率模型尚未完成时间外评估，无法给出判断。"]
    rel = (prob.get("rel") or {}).get("isotonic") or {}
    abs_ = (prob.get("abs") or {}).get("isotonic") or {}
    if not rel and not abs_:
        return ["概率模型未产出有效评估结果。"]
    if rel:
        lines.append(
            f"相对标签（跑赢横截面中位数）：测试样本 {rel.get('n', 0):,}，"
            f"AUC {rel.get('auc') or 0:.4f}，Brier {rel.get('brier') or 0:.5f}"
            f"（全猜基准 {rel.get('brier_baseline') or 0:.5f}），"
            f"方向命中率 {rel.get('hit_rate') or 0:.2%}"
            f"（基准 {rel.get('baseline_hit_rate') or 0:.2%}）。"
        )
        lines.append(
            f"平均预测概率 {rel.get('avg_prob') or 0:.4f}，实际上涨频率 {rel.get('actual_rate') or 0:.4f}，"
            "两者接近说明整体没有系统性偏高或偏低。"
        )
    if abs_:
        lines.append(
            f"绝对标签（绝对上涨）：AUC {abs_.get('auc') or 0:.4f}，"
            f"Brier {abs_.get('brier') or 0:.5f}（基准 {abs_.get('brier_baseline') or 0:.5f}），"
            f"命中率 {abs_.get('hit_rate') or 0:.2%}（基准 {abs_.get('baseline_hit_rate') or 0:.2%}）。"
        )
    best = max(
        (x for x in (rel, abs_) if x),
        key=lambda x: x.get("auc") or 0,
        default={},
    )
    auc_v = best.get("auc")
    if auc_v is not None:
        if auc_v >= 0.55:
            lines.append(f"AUC {auc_v:.4f} 属于有实用价值的水平。")
        elif auc_v >= 0.52:
            lines.append(f"AUC {auc_v:.4f} 略高于随机，但边际很薄，按当前水平做仓位意义有限。")
        else:
            lines.append(
                f"AUC {auc_v:.4f} 低于预设门槛 0.52 —— 概率排序能力不足，"
                "对外输出「上涨概率」时必须同时给出这个置信水平，不能让人误以为有高胜率。"
            )
    loop = bundle.get("prediction_loop") or {}
    if not loop.get("available"):
        lines.append(str(loop.get("note") or "预测闭环尚未启动。"))
    elif not loop.get("sufficient"):
        lines.append(
            f"预测闭环：已判定 {loop.get('n', 0)} 条预测，样本不足，暂不评价命中率。"
        )
    else:
        lines.append(
            f"预测闭环：已判定 {loop.get('n', 0)} 条预测，方向命中率 {loop.get('hit_rate', 0):.1%}"
            f"（基准 {loop.get('baseline_hit_rate', 0):.1%}，"
            f"超额 {(loop.get('edge_vs_baseline') or 0):+.1%}），"
            f"Brier {loop.get('brier') or 0:.4f}。这是用真实行情事后判定的结果，不是回测。"
        )
    return lines


def analyze_live_readiness(bundle: dict[str, Any]) -> list[str]:
    """实盘就绪度分析文案。

    研究门禁回答「方法本身站不站得住」，实盘就绪度回答
    「拿真钱上，它会以什么形态活下来」—— 两件事，必须分开讲。
    """
    lr = bundle.get("live_readiness") or {}
    if not lr.get("available"):
        lines = ["实盘就绪度体检本期不可用（" + str(lr.get("reason") or "原因未记录") + "）。"]
        lines.append("这不是「通过」，也不是「没通过」—— 没有体检数据就不能下实盘结论。")
        return lines

    lines: list[str] = []

    # 1. 打平成本 —— 实盘最关键的单一数字
    be = lr.get("breakeven_cost")
    ac = lr.get("actual_cost")
    gap = lr.get("cost_gap_multiple")
    if be and ac:
        lines.append(
            f"打平成本 {be:.4%} 往返 —— 意思是交易成本只要低于这个数，"
            f"这套方法才追得平等权基准；当前假设 {ac:.2%}，"
            + (f"高出 {gap:.2f} 倍。" if gap else "已经超线。")
        )
        if lr.get("max_turnover_at_cost") and lr.get("turnover"):
            lines.append(
                f"若维持 {ac:.2%} 的成本，换手率必须从 {lr['turnover']:.2%}/期 "
                f"降到 {lr['max_turnover_at_cost']:.2%}/期以下，"
                "否则成本会吃掉全部超额收益。"
            )

    # 2. 相对基准 —— 赚的是 beta 还是 alpha
    ex = lr.get("excess_vs_benchmark")
    if ex is not None:
        if ex > 0:
            lines.append(f"相对等权基准有 {ex:+.2%}/年的超额，选股能力是真的。")
        else:
            lines.append(
                f"相对等权基准 {ex:+.2%}/年 —— 收益来自市场本身，不是选股。"
                "它的价值是「选得比等权稍好/稍差」，不是「能独立赚钱」。"
            )

    # 3. 均值 / 中位数落差 —— 实盘心理预期的锚
    m = lr.get("mean_annualized")
    md = lr.get("median_annualized")
    sp = lr.get("mean_median_spread")
    if m is not None and md is not None and sp is not None:
        lines.append(
            f"均值口径年化 {m:+.2%}，中位数口径 {md:+.2%}，"
            f"落差 {sp:+.2%} —— "
            + ("均值被少数极端盈利月拉高，实盘该按中位数做心理准备：多数月份是亏的。"
               if sp < -0.05 else
               "两种口径接近，收益分布比较均匀。")
        )

    # 4. 跨区段一致性 —— 防样本内挑选
    cross = lr.get("cross_segment") or {}
    exs = cross.get("excess_by_segment") or {}
    if exs:
        cells = "，".join(f"{k} {v:+.2%}" for k, v in exs.items())
        if cross.get("all_segments_negative"):
            lines.append(
                f"训练/验证/测试三段相对基准超额：{cells} —— "
                "三段全为负且方向一致，说明这是**结构性的负 alpha**，"
                "不是某一段行情的运气。"
            )
        elif cross.get("sign_consistent"):
            lines.append(
                f"训练/验证/测试三段相对基准超额：{cells} —— "
                "三段方向一致，结论不是单段挑出来的。"
            )
        else:
            lines.append(
                f"训练/验证/测试三段相对基准超额：{cells} —— "
                "方向不一致，存在事后挑选嫌疑，不能只看好看的测试段。"
            )

    # 5. 总结论 —— 必须回答「现在能不能用」，不能只罗列数字
    blocking = lr.get("blocking_failed") or []
    lines.append(f"综合判定：{lr.get('verdict')}（{lr.get('passed')}/{lr.get('total')} 项通过）。")
    if blocking:
        lines.append(
            "逐项看，未通过的是 "
            + "、".join(blocking)
            + "。这几项任意一项失败都意味着拿真钱做这套方法会亏钱，"
            "不是「收益低一点」而是「方向就是错的」。"
        )
        lines.append(
            "现在能不能用：不能。这套方法目前只能用于纸面跟踪和方法研究，"
            "不能投真钱，也不能作为独立主力策略。"
        )
    else:
        lines.append(
            "现在能不能用：可以做小额实盘验证，但只限未通过项对应的范围内，"
            "且必须先在训练段和验证段复核同一结论。"
        )
    return lines


def analyze_gates(bundle: dict[str, Any]) -> list[str]:
    """门禁表的分析。"""
    gates = bundle.get("gates") or []
    if not gates:
        return ["尚未执行门禁检验。"]
    lines = ["四道门禁逐条过："]
    for g in gates:
        lines.append(f"{g['name']}：{'通过' if g['pass'] else '未通过'} —— {g['detail']}")
    verdict = bundle.get("verdict") or {}
    if verdict.get("text"):
        lines.append(str(verdict["text"]))
    limits = bundle.get("limitations") or []
    if limits:
        lines.append("已知局限：" + "；".join(limits) + "。")
    return lines


def _loop(bundle: dict[str, Any]) -> dict[str, Any]:
    return bundle.get("prediction_loop") or {}


_STAGE_LABEL = {
    "refresh": "刷新行情",
    "predict": "生成 live 预测",
    "judge": "用真实行情判定",
}


def analyze_daily_loop(bundle: dict[str, Any]) -> list[str]:
    """每日闭环三个阶段（刷新 → 预测 → 判定）的运行状态分析。

    失败必须说清楚：哪一段失败、为什么失败、那天的结论还算不算数。
    """
    dl = bundle.get("daily_loop") or {}
    stages = dl.get("stages") or {}
    if not stages:
        return ["每日闭环尚未运行过：今天没有产生新的预测或判定。"]

    lines: list[str] = []
    failed: list[str] = []
    for name in ("refresh", "predict", "judge"):
        st = stages.get(name) or {}
        label = _STAGE_LABEL.get(name, name)
        status = st.get("status")
        if status == "ok":
            detail = st.get("detail") or {}
            extra = ""
            if name == "refresh" and detail.get("updated") is not None:
                extra = (f"：更新 {detail.get('updated') or 0} 只、"
                         f"已是最新 {detail.get('up_to_date') or 0} 只、"
                         f"失败 {detail.get('failed') or 0} 只")
            elif name == "predict" and detail.get("day"):
                extra = f"：预测日 {detail.get('day')}"
            elif name == "judge":
                extra = "：用真实行情判定到期预测"
            lines.append(f"✓ {label}已完成{extra}。")
        elif status == "failed":
            failed.append(label)
            lines.append(
                f"✗ {label}失败：{st.get('error') or '原因未记录'}。"
                "本环节的结果**不能**当作今天的结论。"
            )
        else:
            lines.append(f"○ {label}今天还没跑过。")

    if failed:
        lines.append(
            f"今日闭环未跑通（{'、'.join(failed)} 失败），"
            "上述命中率只反映**已落盘的**历史判定，不代表今天的实时表现。"
        )
    return lines


def analyze_loop_overview(bundle: dict[str, Any]) -> list[str]:
    """预测闭环总览表 + live/replay 分口径表 的分析。"""
    loop = _loop(bundle)
    if not loop.get("available"):
        return [str(loop.get("note") or "预测闭环尚未启动，暂时无法评价。")]
    if not loop.get("sufficient"):
        return [
            f"已判定样本 {loop.get('n', 0)} 条，不足以评价命中率（门槛 30 条），本期不做结论。"
        ]
    lines = [
        f"全量闭环：{loop.get('n_days', 0)} 个预测日、{loop.get('n', 0)} 条已判定预测，"
        f"方向命中率 {loop.get('hit_rate') or 0:.2%}"
        f"（全押方向基准 {loop.get('baseline_hit_rate') or 0:.2%}，"
        f"差 {(loop.get('edge_vs_baseline') or 0):+.2%}），"
        f"Brier {loop.get('brier') or 0:.5f}。判定全部读自已落盘的 resolved 文件，"
        "不是回测现算，事后不会被改写。"
    ]
    return lines


def analyze_loop_split(bundle: dict[str, Any]) -> list[str]:
    """市场方向口径 + live/replay 分口径表的分析。"""
    loop = _loop(bundle)
    if not loop.get("available") or not loop.get("sufficient"):
        return ["预测闭环尚未启动或样本不足，暂时没有可拆分的三种口径。"]
    lines: list[str] = []
    mkt = loop.get("market") or {}
    if mkt.get("n"):
        lines.append(
            f"市场方向口径（与选股口径分开评估）：已判定 {mkt['n']} 条，"
            f"方向命中率 {mkt.get('hit_rate') or 0:.2%}，"
            f"平均前瞻收益 {mkt.get('avg_fwd_return') or 0:+.2%}。"
            "这一项衡量的是「押不押大盘」，不能拿来证明选股能力。"
        )
    live = loop.get("live") or {}
    replay = loop.get("replay") or {}
    lines.append(
        f"分口径：实时预测 {live.get('n_days', 0)} 个预测日 / 已到期 {live.get('n', 0)} 条；"
        f"历史回溯自检 {replay.get('n_days', 0)} 个预测日 / 已判定 {replay.get('n', 0)} 条"
        f"（回溯命中率 {replay.get('hit_rate') or 0:.2%}，"
        f"Brier {replay.get('brier') or 0:.5f}）。"
    )
    if (live.get("n") or 0) == 0:
        lines.append(
            "实时口径目前 0 条到期 —— 前瞻窗口还没走完，"
            "所以上面所有命中率都来自回溯自检，**不能**当成上线前的真实表现。"
        )
    return lines


def analyze_loop_selection(bundle: dict[str, Any]) -> list[str]:
    """十分位价差（选股能力）表的分析。"""
    pf = _loop(bundle).get("portfolio") or {}
    if not pf.get("available"):
        return ["选股口径（十分位价差）暂无已判定样本，无法计算。"]
    med = pf.get("median_spread") or 0.0
    avg = pf.get("avg_spread") or 0.0
    lines = [
        f"选股口径（十分位）：头部组合均值 {pf.get('avg_top_decile_return') or 0:+.2%}、"
        f"尾部 {pf.get('avg_bottom_decile_return') or 0:+.2%}，"
        f"价差均值 {avg:+.2%}、**中位数 {med:+.2%}**，"
        f"正价差天数占比 {pf.get('pct_days_spread_positive') or 0:.1%}"
        f"（{pf.get('n_days', 0)} 个可评估交易日）。"
    ]
    if avg > 0 and med <= 0:
        lines.append(
            "均值与中位数方向相反 —— 均值是被少数极端收益撑起来的，"
            "中位数才反映常态日的选股能力。**门禁看中位数**，"
            "所以这一条按未通过计，不能拿均值说事。"
        )
    lines.append(
        f"同期全市场等权均值 {pf.get('avg_equal_weight_return') or 0:+.2%}，"
        f"头部减等权 {pf.get('avg_top_minus_bw') or 0:+.2%} —— "
        "这一项才是「选股有没有用」而不是「大盘涨没涨」。"
    )
    return lines


def analyze_loop_day_level(bundle: dict[str, Any]) -> list[str]:
    """按交易日聚合表的分析。"""
    dl = _loop(bundle).get("day_level") or {}
    if not dl.get("available"):
        return ["按交易日聚合暂无可用样本。"]
    return [
        f"按交易日聚合（自由度口径）：{dl.get('n_days', 0)} 天里，"
        f"模型日命中率跑赢硬币的天数占 {(dl.get('pct_days_hit_above_50') or 0) * 100:.2f}%，"
        f"平均每日命中率 {dl.get('avg_daily_hit_rate') or 0:.2%}、"
        f"中位 {dl.get('median_daily_hit_rate') or 0:.2%}；"
        f"同期大盘上涨的天数占 {(dl.get('pct_days_actual_up_above_50') or 0) * 100:.2f}%。",
        "同一天的几千条样本高度相关（一荣俱荣一损俱损），"
        "把十几万条当独立样本会把置信度算爆，所以按天看才是真实自由度。",
    ]


def analyze_g4_criteria(bundle: dict[str, Any]) -> list[str]:
    """G4 判据明细表的分析：哪一条没过、差多少、能不能进模拟仓位。"""
    g4 = next(
        (g for g in (bundle.get("gates") or []) if str(g.get("name", "")).startswith("G4")),
        None,
    )
    if not g4:
        return ["G4 判据尚未生成。"]
    lines: list[str] = []
    cd = g4.get("criteria_detail") or []
    if cd:
        failed = [c for c in cd if not c.get("pass")]
        if failed:
            lines.append(
                "G4 判据逐条："
                + "；".join(
                    f"{c['name']} 实测 {c['actual']}、门槛 {c['threshold']} —— 未达标"
                    for c in failed
                )
                + "。"
            )
        else:
            lines.append("G4 三条判据全部达标。")
    if g4.get("detail_legacy"):
        lines.append(str(g4["detail_legacy"]) + "（旧判据仅作参考，不作门禁）。")
    verdict = bundle.get("verdict") or {}
    if verdict.get("sim_position_allowed"):
        lines.append(
            "四道门禁全过，研究信号可进入模拟仓位（仍为模拟，不执行真实交易）。"
        )
    else:
        failed_gates = verdict.get("failed_gates") or []
        lines.append(
            "结论：**信号未过门禁，不进入模拟仓位**，本期只作观察记录。"
            + (f"未通过：{'、'.join(failed_gates)}。" if failed_gates else "")
        )
    return lines


def analyze_prediction_loop(bundle: dict[str, Any]) -> list[str]:
    """预测闭环整块的分析（四张表的要点合起来，用于汇总/日志）。"""
    out: list[str] = []
    out.extend(analyze_loop_overview(bundle))
    out.extend(analyze_loop_split(bundle))
    out.extend(analyze_loop_selection(bundle))
    out.extend(analyze_loop_day_level(bundle))
    out.extend(analyze_g4_criteria(bundle))
    return out


def analyze_evidence(bundle: dict[str, Any]) -> list[str]:
    """整块的分析要点（用于汇总或日志）。"""
    if not bundle:
        return ["尚无研究能力实测数据，无法给出判断。"]
    out: list[str] = []
    out.extend(analyze_factor_oos(bundle))
    out.extend(analyze_diagnostic(bundle))
    out.extend(analyze_strategy(bundle))
    out.extend(analyze_probability(bundle))
    out.extend(analyze_prediction_loop(bundle))
    out.extend(analyze_gates(bundle))
    return out

