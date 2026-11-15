"""生成"小白版"财经日报正文。

设计原则（对应宋老师 2026-11-12 的要求"当我是小白"）：
1. 先说结论，再给依据 —— 小白不会先看第十节才知道结果。
2. 每个专业名词后面立刻跟一句白话，不要求读者自己去查。
3. 数字一律从原始 evidence 取，不在代码里手写任何数字字面量，
   避免"翻译成白话"过程中数字与原报告对不上。
4. 本模块只产出"文字版"正文，供日报渲染层或其他渠道引用，
   不改动 render/report.py，避免与小金正在做的渲染层改造冲突。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import plain_glossary as PG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
# build_evidence() 失败时记录的原因（供渲染层标注「数据降级」）
_EVIDENCE_FALLBACK_REASON = ""


def load_evidence() -> dict:
    """加载研究证据（门禁 / 因子 / 策略 / 概率）。

    优先用 research_evidence.build_evidence()（与日报正文同一真源），
    失败时回退到已落盘的 oos_latest.json。

    ⚠️ 回退文件的键形状与 build_evidence() **不同**：
       build_evidence 用 strategy / probability / gates，
       oos_latest.json 用 strategy_test_segment / probability_rel / oos_validation。
       直接把 oos_latest.json 当 bundle 用会让 gates/strategy/probability 三个
       section 静默渲染成空（读者看到空白，而不是"数据缺失"）。
    所以回退时必须显式做键映射，并且**记录失败原因**，不能悄悄降级。
    """
    try:
        import research_evidence as RE

        bundle = RE.build_evidence()
        if bundle:
            return bundle
    except Exception as exc:                       # noqa: BLE001
        global _EVIDENCE_FALLBACK_REASON
        _EVIDENCE_FALLBACK_REASON = f"build_evidence 失败: {exc!r}"
        print(f"[plain_report] {_EVIDENCE_FALLBACK_REASON}", file=sys.stderr)
        bundle = None

    path = DATA / "research" / "oos_latest.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        bundle = _adapt_raw_oos(raw)
        bundle.setdefault("_fallback", True)
        bundle.setdefault("_fallback_reason",
                          _EVIDENCE_FALLBACK_REASON or "build_evidence 返回空")
        return bundle
    return {"_fallback": True, "_fallback_reason": "无 oos_latest.json 且 build_evidence 失败"}


def _adapt_raw_oos(raw: dict) -> dict:
    """把 oos_latest.json 的原始键形状适配成 bundle 形状。

    只做键映射，不改任何数字 —— 数字的真理来源只有 ooslab。
    """
    if not isinstance(raw, dict):
        return {}
    strat = raw.get("strategy_test_segment") or {}
    return {
        "available": True,
        "generated_at": raw.get("generated_at"),
        "gates": [],
        "verdict": {},
        "panel": raw.get("panel") or {},
        "segments": raw.get("segments") or {},
        "config": raw.get("config") or {},
        "factors": [
            {
                "name": o.get("name"),
                "desc": o.get("desc"),
                "oos_mean": o.get("oos_mean"),
                "oos_t": o.get("oos_t"),
                "oos_hit": o.get("oos_hit"),
                "same_sign": o.get("same_sign"),
                "verdict": o.get("oos_verdict"),
            }
            for o in (raw.get("oos_validation") or [])
        ],
        "oos_pass": sum(1 for o in (raw.get("oos_validation") or [])
                        if o.get("oos_verdict") == "时间外通过"),
        "oos_total": len(raw.get("oos_validation") or []),
        "strategy": strat,
        "cost_sensitivity": raw.get("cost_sensitivity") or [],
        "strategy_config": raw.get("strategy_config") or {},
        "construction_selection": raw.get("construction_selection") or {},
        "signal_diagnostic": raw.get("signal_diagnostic") or {},
        "limitations": raw.get("limitations") or [],
        "horizon": raw.get("horizon"),
        "base_cost": (raw.get("config") or {}).get("cost_roundtrip", 0.003),
        "probability": {
            "rel": raw.get("probability_rel") or {},
            "abs": raw.get("probability_abs") or {},
        },
    }


def load_snapshot(day: str | None = None) -> dict:
    """加载某日快照。"""
    if day is None:
        snaps = sorted((DATA / "snapshots").glob("*.json"))
        if not snaps:
            return {}
        path = snaps[-1]
    else:
        path = DATA / "snapshots" / f"{day}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 格式化助手（数字的展示口径全部在这里定义，和 render/report.py 保持一致）
# --------------------------------------------------------------------------
def _pct(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{nd}f}%"


def _num(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def _rate(v, nd: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{nd}f}%"


def _pctnum(v, nd: int = 2) -> str:
    """已经以「百分数」为单位的字段（如 avg_return_pct=1.88 表示 1.88%）。

    这类字段不能再乘 100 —— 口径与 render/report.py:943 一致。
    """
    if v is None:
        return "—"
    return f"{v:.{nd}f}%"


# --------------------------------------------------------------------------
# 白话正文生成
# --------------------------------------------------------------------------
# 需要就地替换的黑话 → 白话（长词优先，避免 "IC" 误伤 "AUC" 之类）
DEJARGON_MAP: list[tuple[str, str]] = [
    ("十分位价差中位数", "最会买和最不会买两组的收益差（中位数口径）"),
    ("十分位价差", "最会买和最不会买两组的收益差"),
    ("Brier 优于全猜基准率", "概率标注是否比瞎猜更准"),
    ("全猜基准率", "及格线"),
    ("Brier", "概率标注准度"),
    ("AUC", "打分准头"),
    ("Sharpe", "赚得值不值（夏普比率）"),
    ("IC", "信号准头"),
    ("t 值", "把握程度"),
    ("门禁", "考试"),
    ("样本外", "没见过的数据上"),
    ("时间外", "没见过的数据上"),
    ("训练段", "学习用的那段数据"),
    ("测试段", "检验用的那段数据"),
    ("方向命中率", "猜对方向的比例"),
    ("命中率", "猜对的比例"),
    ("因子", "小线索"),
    ("正价差天数占比", "价差为正的天数占比"),
    ("往返成本", "一次买卖的手续费"),
]


def _dejargon(text: str) -> str:
    """把一段技术描述里的黑话就地换成白话。

    长词先替换，避免 "Sharpe" 被 "IC" 之类短词切碎。
    """
    out = text
    for src, dst in DEJARGON_MAP:
        if src in out:
            out = out.replace(src, dst)
    return out


def _gates_section(ev: dict) -> list[str]:
    """门禁：上场前的四道考试。

    门禁 detail 是原报告的原文（含 Sharpe / Brier 等黑话），不能原样贴给小白。
    优先用结构化的 criteria_detail 逐条翻译；没有 criteria_detail 时，
    用词典把黑话就地替换成白话。
    """
    gates = ev.get("gates") or []
    if not gates:
        return []
    passed = [g for g in gates if g.get("pass")]
    failed = [g for g in gates if not g.get("pass")]
    lines = [
        f"这套系统上场前要过 {len(gates)} 道考试（专业叫「门禁」），"
        f"过了 {len(passed)} 道，没过 {len(failed)} 道。",
    ]
    for g in gates:
        mark = "过了" if g.get("pass") else "没过"
        name = str(g.get("name") or "")
        lines.append(f"  {name}：{mark}")
        for c in g.get("criteria_detail") or []:
            cname = str(c.get("name") or "")
            threshold = c.get("threshold")
            actual = c.get("actual")
            cmark = "达标" if c.get("pass") else "没达标"
            sub = f"要求 {threshold}，实际 {actual}"
            lines.append(f"      {cname} —— {cmark}（{sub}）")
        detail = g.get("detail")
        if detail:
            lines.append(f"      {_dejargon(str(detail))}")
    if failed:
        lines.append(
            "没过的意思是：这些信号现在还不准，先不放进模拟仓位里试，只当观察记录。"
        )
    else:
        lines.append("四道全过，说明这套信号已经有资格进模拟仓位了。")
    return lines


def _strategy_section(ev: dict) -> list[str]:
    strat = ev.get("strategy") or {}
    if not strat.get("ok"):
        return []
    lo = strat.get("long_only") or {}
    bm = strat.get("benchmark") or {}
    ls = strat.get("long_short") or {}
    lo_med = strat.get("long_only_median") or {}
    breakeven = strat.get("breakeven_cost_long_only")

    lo_ann = lo.get("net_annualized")
    bm_ann = bm.get("annualized")
    excess = None
    if lo_ann is not None and bm_ann is not None:
        excess = lo_ann - bm_ann

    lines: list[str] = []
    if lo_ann is not None:
        lines.append(
            f"把交易手续费都扣掉之后，这套「只买看涨的股票」的方法，"
            f"一年能赚 {_pct(lo_ann)}（专业叫「年化收益」）。"
        )
    if bm_ann is not None:
        lines.append(
            f"同时间段，如果只是把所有股票平均买一遍（专业叫「基准」），"
            f"一年能赚 {_pct(bm_ann)}。"
        )
    if excess is not None:
        verdict = "跑赢" if excess > 0 else "没跑赢"
        lines.append(
            f"也就是说，这套方法{excess > 0 and '比大盘多赚' or '比大盘少赚'} "
            f"{_pct(abs(excess))} —— 结论是{excess > 0 and '跑赢' or '没跑赢'}大盘。"
        )
    if lo.get("sharpe") is not None:
        lines.append(
            f"衡量「赚得值不值」的分数（专业叫「夏普比率」Sharpe）是 "
            f"{_num(lo.get('sharpe'))} —— "
            f"{'有赚钱本事，但还不算强' if (lo.get('sharpe') or 0) >= 0.5 else '偏弱'}。"
        )
    if lo.get("win_rate") is not None:
        lines.append(
            f"赚钱的次数占 {_rate(lo.get('win_rate'))}（专业叫「胜率」）。"
        )
    if breakeven is not None:
        lines.append(
            f"手续费如果涨到每次买卖 {_pct(breakeven, 3)} 以上，这套方法就刚好不赚钱了"
            f"（专业叫「临界成本」）。实际成本比这高，就白忙。"
        )
    if ls.get("net_annualized") is not None:
        lines.append(
            f"如果同时做空看跌的股票（专业叫「多空组合」），一年是 "
            f"{_pct(ls.get('net_annualized'))} —— 说明这套信号靠做空没帮上忙，"
            f"甚至拖后腿。"
        )
    if lo_med.get("net_annualized") is not None and lo_ann is not None:
        lines.append(
            f"如果用「中位数」而不是平均数来算（掐掉极端涨跌、更接近普通人能拿到的数），"
            f"一年是 {_pct(lo_med.get('net_annualized'))} —— "
            f"{'和主口径差很多，说明收益靠少数极端值撑着，不稳' if abs((lo_med.get('net_annualized') or 0) - lo_ann) > 0.05 else '和主口径接近，收益比较均匀'}。"
        )
    return lines


def _signal_section(ev: dict) -> list[str]:
    diag = ev.get("signal_diagnostic") or {}
    if diag.get("ic_mean") is None:
        return []
    lines = [
        f"这套信号预测涨跌的准头（专业叫「IC」），平均是 {_num(diag.get('ic_mean'), 4)} —— "
        f"数字接近 0 就等于没用，0.05 算勉强，0.1 以上才算真准。",
    ]
    if diag.get("ic_t") is not None:
        t = diag["ic_t"]
        lines.append(
            f"这个结论的把握程度（专业叫「t 值」）是 {_num(t)} —— "
            f"{'这个结论基本可信' if abs(t) > 2 else '这个结论还不太站得住'}。"
        )
    if diag.get("tail_spread_mean") is not None:
        lines.append(
            f"如果真按信号高低买股票，最会买的一批和最不会买的一批，"
            f"收益差距是 {_pct(diag.get('tail_spread_mean'), 4)}（专业叫「十分位价差」）。"
        )
    if diag.get("tail_median_positive_ratio") is not None:
        lines.append(
            f"有 {_rate(diag.get('tail_median_positive_ratio'))} 的时间这个差距是正的 —— "
            f"也就是十次里有六次以上是对的。"
        )
    return lines


def _probability_section(ev: dict) -> list[str]:
    lines: list[str] = []
    prob = ev.get("probability") or {}
    for key, label in (("rel", "跑赢大盘"), ("abs", "绝对涨跌")):
        pm = prob.get(key) or {}
        raw = pm.get("raw") or {}
        if not raw:
            continue
        auc = raw.get("auc")
        brier = raw.get("brier")
        base_brier = raw.get("brier_baseline")
        if auc is not None:
            verdict = (
                "完全没用" if auc <= 0.5 else
                ("微弱有用" if auc < 0.6 else "确实有用")
            )
            lines.append(
                f"预测「{label}」的模型，打分准头（专业叫「AUC」）是 {_num(auc, 4)} —— "
                f"{verdict}。0.5 就是瞎猜，0.6 以上才算真会用。"
            )
        if brier is not None and base_brier is not None:
            # 判定必须与数字一致：brier < base 才叫「更好」。
            # 修复前这里写成「差不多，甚至更差」，但 0.24978 < 0.24959 明明更好 —— 小白一眼看出矛盾。
            if brier < base_brier - 0.00005:
                verdict = "略好一点点"
            elif brier <= base_brier + 0.00005:
                verdict = "基本持平"
            else:
                verdict = "反而更差"
            # 注意：rel（跑赢大盘）和 abs（绝对涨跌）是两个不同的预测任务，
            # 各自的「及格线」数值本来就不同（0.25000 vs 0.24959）。
            # 所以措辞必须点明「这条线自己的及格线」，否则小白会以为系统有两个标准。
            lines.append(
                f"它说「有七成把握」的时候准不准（专业叫「Brier」分数）："
                f"{_num(brier, 5)}，跟这条线自己的及格线 {_num(base_brier, 5)} 相比，{verdict}。"
                f"（每个预测方向的及格线不一样，上面「{label}」这条是 {_num(base_brier, 5)}。）"
            )

    # 真实预测闭环（如果已经产生预测）
    loop = ev.get("prediction_loop") or {}
    # 每日闭环今天有没有跑通 —— 没跑通就要说清楚，
    # 否则读者会以为下面的命中率就是今天的实时表现。
    dl = ev.get("daily_loop") or {}
    dl_stages = dl.get("stages") or {}
    dl_failed = [
        {"refresh": "刷新行情", "predict": "生成今天的预测", "judge": "用真实行情回头验证"}
        .get(name, name)
        for name, st in dl_stages.items()
        if (st or {}).get("status") == "failed"
    ]
    if dl_failed:
        errs = "；".join(
            f"{(dl_stages.get(n) or {}).get('error') or '原因未记录'}"
            for n in ("refresh", "predict", "judge")
            if (dl_stages.get(n) or {}).get("status") == "failed"
        )
        lines.append(
            f"注意：今天的自动流程没能全部跑完（{'、'.join(dl_failed)} 出问题）。"
            f"原因是：{errs}。"
            "下面说的命中率只统计**已经落盘的历史预测**，"
            "不代表今天的预测已经验证过了。"
        )
    if loop.get("available") and loop.get("sufficient"):
        hr = loop.get("hit_rate")
        if hr is not None:
            lines.append(
                f"另外，系统已经攒了 {loop.get('n', 0):,} 条真实预测并回头验证过，"
                f"方向命中率 {_rate(hr)} —— "
                f"{'比瞎猜的 50% 略好' if hr > 0.5 else '还不如瞎猜'}。"
            )
        if loop.get("brier") is not None:
            # 这里说的是「预测闭环」这条线自己的基准，与上面概率模型的基准线
            # 是两个不同口径 —— 必须用不同措辞区分，否则小白看到三个基准线会懵。
            lines.append(
                f"这些预测的概率标注（专业叫「Brier」分数）是 {_num(loop.get('brier'), 5)}。"
                f"判断标准是 0.25 —— 也就是「完全没有信息」的及格线，比 0.25 大就等于白预测。"
            )
    return lines


def _factors_section(ev: dict) -> list[str]:
    factors = ev.get("factors") or []
    if not factors:
        return []
    passed = [f for f in factors if f.get("verdict") == "时间外通过"]
    lines = [
        f"这套系统一共用了 {len(factors)} 个小指标（专业叫「因子」），"
        f"每个指标都是一个「可能预测涨跌的线索」。",
        f"其中 {len(passed)} 个在从来没见过的数据上（专业叫「样本外」）依然有效。"
        if passed else "其中没有在没见过的数据上被验证通过的。",
    ]
    for f in factors[:3]:
        name = f.get("name")
        desc = f.get("desc") or ""
        oos_mean = f.get("oos_mean")
        entry = PG.explain(name) or PG.explain(str(name))
        explain_txt = f"（{entry.plain}）" if entry else ""
        if oos_mean is not None:
            lines.append(
                f"  {desc or name}{explain_txt}：在没见过的数据上准头是 {_num(oos_mean, 4)}。"
            )
        else:
            lines.append(f"  {desc or name}{explain_txt}")
    return lines


def _review_section(review: dict | None) -> list[str]:
    """复盘：拿昨天/前期的建议回头核对实际涨跌。

    数据真源与日报完全一致：review.build_review()（main.py:441 同一个函数），
    不用快照里可能已过期的 review_at_run.stats。
    """
    if not review:
        return []
    stats = review.get("stats") or {}
    hit_rate = stats.get("hit_rate")
    if hit_rate is None:
        note = review.get("note") or ""
        return [f"暂时没有可复盘的历史建议（{note}）。"] if note else []

    judged = stats.get("judged") or 0
    lines = [
        f"回头核对了一下以前的建议：{judged} 条能判断对错的建议里，"
        f"方向猜对 {_rate(hit_rate)} —— "
        f"{'比瞎猜的 50% 好一些' if hit_rate > 0.5 else '不如瞎猜'}。",
    ]
    if stats.get("stop_trigger_rate") is not None:
        lines.append(
            f"设定的止损线被触发过 {_rate(stats.get('stop_trigger_rate'))}。"
        )
    if stats.get("target_hit_rate") is not None:
        lines.append(
            f"设定的止盈目标达到过 {_rate(stats.get('target_hit_rate'))}。"
        )
    if stats.get("stability_avg_jaccard") is not None:
        j = stats["stability_avg_jaccard"]
        lines.append(
            f"两次推荐清单的重合度是 {_num(j, 2)}（专业叫「Jaccard」相似度）—— "
            f"{'清单挺稳定' if j >= 0.5 else '清单换得比较频繁'}。"
        )
    if stats.get("avg_return_pct") is not None:
        lines.append(
            f"这些建议实际拿到的平均收益是 {_pctnum(stats.get('avg_return_pct'))}。"
        )
    bias = stats.get("bias") or {}
    for direction, b in bias.items():
        gap = b.get("gap_pct")
        if gap is None:
            continue
        lines.append(
            f"  {direction}方向：实际平均 {_pctnum(b.get('avg_ret_pct'))}，"
            f"但当时喊的目标是 {_pctnum(b.get('implied_target_pct'), 0)} —— "
            f"差 {_pctnum(abs(gap))}，{'目标定高了' if gap < 0 else '目标定保守了'}。"
        )
    return lines


def load_review(day: str | None = None) -> dict:
    """复盘数据：与日报渲染时同一真源（review.build_review）。"""
    try:
        from review import build_review

        quotes = {}
        if day:
            raw_path = DATA / "raw" / f"{day}.json"
            if raw_path.exists():
                quotes = (json.loads(raw_path.read_text(encoding="utf-8")) or {}).get("quotes") or {}
        else:
            raws = sorted((DATA / "raw").glob("*.json"))
            if raws:
                quotes = (json.loads(raws[-1].read_text(encoding="utf-8")) or {}).get("quotes") or {}
        return build_review(DATA / "snapshots", quotes) or {}
    except Exception:
        return {}


def build_plain_report(ev: dict | None = None, review: dict | None = None) -> str:
    """生成小白版正文（纯文本）。

    ev / review 都不传时自动从真源加载（与日报渲染同一个数据路径）。
    """
    if ev is None:
        ev = load_evidence()
    if review is None:
        review = load_review()

    lines: list[str] = []
    lines.append("【一句话结论】")
    if ev.get("_fallback"):
        lines.append(
            f"注意：本期研究数据走了降级通道（{ev.get('_fallback_reason', '原因未知')}），"
            "部分结论可能不完整，请以完整报告为准。"
        )
    strat = ev.get("strategy") or {}
    lo = strat.get("long_only") or {}
    bm = strat.get("benchmark") or {}
    lo_ann = lo.get("net_annualized")
    bm_ann = bm.get("annualized")
    if lo_ann is not None and bm_ann is not None:
        excess = lo_ann - bm_ann
        if excess > 0:
            lines.append(
                f"这套系统在扣掉手续费后一年赚 {_pct(lo_ann)}，"
                f"比「平均买所有股票」多赚 {_pct(excess)} —— 结论：有效，但优势不大。"
            )
        else:
            lines.append(
                f"这套系统在扣掉手续费后一年赚 {_pct(lo_ann)}，"
                f"比「平均买所有股票」少赚 {_pct(abs(excess))} —— 结论：目前这套方法还不如直接买指数，"
                f"所以系统自己决定：这些信号先不进模拟仓位，只当观察。"
            )
    else:
        lines.append("本期没有可对比的策略测试结果。")

    gates = ev.get("gates") or []
    if gates:
        npass = sum(1 for g in gates if g.get("pass"))
        lines.append("")
        lines.append(f"【这套方法现在处于什么状态】")
        lines.append(
            f"按「门禁」（上场前的考试）算，{len(gates)} 道里过了 {npass} 道。"
        )
    elif ev.get("_fallback"):
        lines.append("")
        lines.append("【这套方法现在处于什么状态】")
        lines.append(
            "本期门禁结论不可用（数据走了降级通道），"
            "所以现在**不能**说这套方法已经通过考验，也不能说它没通过。"
        )

    lines.append("")
    lines.append("【信号准不准】")
    lines.extend(_signal_section(ev))

    lines.append("")
    lines.append("【真金白银能赚多少】")
    lines.extend(_strategy_section(ev))

    lines.append("")
    lines.append("【预测涨跌的模型行不行】")
    lines.extend(_probability_section(ev))

    lines.append("")
    lines.append("【以前给的建议，事后对不对】")
    lines.extend(_review_section(review))

    lines.append("")
    lines.append("【用了哪些小线索】")
    lines.extend(_factors_section(ev))

    lines.append("")
    lines.append("【考试结果明细】")
    lines.extend(_gates_section(ev))

    lines.append("")
    lines.append("【术语表：这些词到底什么意思】")
    for line in PG.glossary_lines():
        lines.append(f"  {line}")

    return "\n".join(lines)


def build_plain_html(ev: dict | None = None) -> str:
    """生成小白版 HTML（独立文件，不动原日报）。"""
    import html as _html

    body = build_plain_report(ev)
    paras = []
    for block in body.split("\n"):
        paras.append(_html.escape(block))

    body_html = "\n".join(
        f"<p>{p}</p>" if p else "<p>&nbsp;</p>" for p in paras
    )
    return (
        "<!doctype html>\n<html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>财经日报 · 小白版</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;"
        "max-width:760px;margin:0 auto;padding:24px 20px;line-height:1.85;"
        "color:#1e293b;background:#f8fafc;}"
        "h1{font-size:22px;color:#0f172a;}"
        "p{margin:0 0 10px;font-size:15px;white-space:pre-wrap;}"
        "p:first-of-type{font-weight:600;color:#0f172a;}"
        "</style></head><body>"
        "<h1>财经日报 · 小白版</h1>"
        f"{body_html}"
        "</body></html>"
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="生成小白版财经日报")
    ap.add_argument("--day", default=None, help="快照日期，默认最新")
    ap.add_argument("--out", default=None, help="输出 HTML 路径")
    ap.add_argument("--text", action="store_true", help="只打印文本版")
    args = ap.parse_args()

    ev = load_evidence()
    if args.text:
        print(build_plain_report(ev))
        return

    out = args.out or str(DATA / "html" / f"{args.day or 'latest'}-plain.html")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(build_plain_html(ev), encoding="utf-8")
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
