"""白话层：把日报里的专业术语翻译成零基础读者也能看懂的话。

设计原则（不可违反）：
1. **不删任何数字**。日报的数字是审计证据 —— 白话层只做「加旁注」，
   不替换、不四舍五入、不覆盖原始数值。专业表原样保留。
2. **不编数字**。白话里的每一个数字都必须来自当次渲染的真实数据；
   数据缺失就写「暂无」，绝不臆造。
3. **确定性**。纯规则生成，与日报其余部分一样可复现、可审计、可 diff。
   本模块**不允许**调用任何 LLM。
4. **不改结论**。白话层不改变任何门禁判定、仓位建议、复盘结论，
   只解释这些结论「到底在说什么」。

为什么要这一层：日报里有大量量化术语（IC、Sharpe、Brier、十分位价差、
门禁、Jaccard…）。不解释，读者会以为报告在「装专业」；解释不清，
又会误读成「数字很大 = 很厉害」。这一层的存在，是让不懂量化的人也能
读出结论，并且读出结论的**分量**（而不是只看见数字）。

中文引号一律用「」—— 避免和 Python 字符串定界符打架。
"""
from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------ 小工具


def _esc(v: Any) -> str:
    s = "" if v is None else str(v)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _num(v: Any, nd: int = 1, dash: str = "暂无") -> str:
    """数值 → 人话里的数字。None 一律写「暂无」，不写 0（0 是有效结论）。"""
    if v is None:
        return dash
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int, float)):
        return f"{v:,.{nd}f}"
    return str(v)


def _pct(v: Any, nd: int = 1, dash: str = "暂无") -> str:
    """0.3333 → 33.3%。"""
    if v is None:
        return dash
    try:
        return f"{float(v) * 100:,.{nd}f}%"
    except (TypeError, ValueError):
        return dash


# ------------------------------------------------------------------ 术语对照表

# 术语 → 白话。顺序按「读者从上往下会遇到的顺序」排，不是字母序。
GLOSSARY: list[tuple[str, str]] = [
    # —— 第一层：看卡片和组合时就会撞上的词
    ("大盘温度", "系统给当前市场打的一个 0–100 分。分越高越适合进攻（多买股票），分越低越该防守（留现金）。"),
    ("权益仓位", "打算放在股票和股票型基金里的钱，占总资金的比例。"),
    ("理财仓位", "放在货币基金、债券基金、银行理财等波动较小的产品里的钱，占总资金的比例。"),
    ("现金", "不投进任何产品的钱，留着应急或等更好的机会。"),
    ("止损位", "事先定好的卖出线。股价跌到这条线就无条件卖出，不解释、不硬扛。"),
    ("目标位（分批止盈）", "事先定好的上涨方向卖出线。分三次执行：+8% 卖 1/3，+15% 再卖 1/3，+20% 卖完。"),
    ("双源核验", "同一个数据去两个互相独立的网站各取一次，两边对得上才采信。"),
    ("单源", "只在一个网站取到、找不到第二个来源对照的数据。报告会明确标出来，可信度打折。"),
    ("口径冲突", "两个来源对同一项给出不同的数或不同的说法。报告不偷偷挑一个，而是把冲突列出来。"),
    ("核验率", "一组数据里，通过双源核验的比例。比例越低，结论越要打折。"),
    ("评分", "关注标的清单里的综合打分 = 动量(0–50) + 估值(0–30) + 流动性(0–10) + 核验(0–10)。"),
    ("PE / PB", "市盈率 / 市净率，用来衡量贵不贵。PE 低不等于便宜，还要看行业和盈利质量。"),
    ("VIX", "标普 500 的恐慌指数。市场越慌它越高，是个反向指标。"),
    ("环比", "跟上一个统计期比（比如这个月跟上个月）。"),
    ("7 日年化", "货币基金最近 7 天的收益折算成一整年的参考值，不是承诺收益。"),
    ("固收+", "大部分资金投债券、少部分投股票的混合型基金，波动介于纯债和股票之间。"),
    ("前复权", "把分红送股的影响抹平，让历史价格和当前价格可比。"),

    # —— 第二层：看复盘时撞上的词
    ("不可变快照", "每天写下的组合、价格、止损位等原始记录，写完就锁定，之后不许回头修改。"),
    ("复盘", "拿后来真实发生的行情，去核对上一期给出的建议到底对不对。"),
    ("方向命中率", "上期说涨的建议里，后来真的涨了的比例。低于 50% 等于不如抛硬币。"),
    ("止损触发率", "上期给的建议里，后来真的跌到止损线的比例。"),
    ("目标达成率", "上期给的建议里，后来真的涨到第一档目标（+8%）的比例。"),
    ("建议稳定性（Jaccard）", "相邻两期推荐的股票名单有多少重合。1 = 完全没变，0 = 全换过。数值太低说明策略摇摆。"),
    ("偏差分析", "实际赚到的和当初承诺的目标差多少。差值为负，说明当初的目标定高了。"),
    ("隐含目标", "建议里默认承诺的收益目标（当前口径 +8%）。"),

    # —— 第三层：看研究能力实测时撞上的词
    ("样本外", "用算法从没见过的新数据去考试。在见过数据的成绩不算数。"),
    ("训练段", "用来发现因子规律的那段历史数据。"),
    ("测试段", "用来考试的那段历史数据 —— 找规律的时候从未看过它。"),
    ("前瞻窗口", "预测之后往后看几个交易日来验证结果。"),
    ("因子", "一个用来打分的数字指标，比如偏离 60 日均线、20 日涨跌幅。"),
    ("IC", "因子分数和未来涨跌的相关性。0 = 完全没用，1 = 完美。日常 0.05 就算有意义。"),
    ("t 值", "这个相关有多可能是巧合。绝对值大于 2 通常才算靠谱。"),
    ("十分位价差", "把所有股票按打分从高到低排成 10 档，最高的那档比最低的那档，每期多赚（或多亏）多少。"),
    ("均值口径", "用所有期次的平均收益来衡量。容易被少数暴涨暴跌的极端情况带偏。"),
    ("中位数口径", "把各期收益排好序取中间那个。更抗极端值干扰。"),
    ("缩尾均值", "先把极端值掐掉再求平均，避免个别妖股把结论带歪。"),
    ("成本后", "扣掉佣金、印花税和滑点之后的真实收益 —— 纸面赚的不等于能到手。"),
    ("换手率", "每期调仓时买卖的股票比例。换手越高，被手续费吃掉越多。"),
    ("年化", "把一段时间的收益折算成一整年，方便横向比较。"),
    ("Sharpe（夏普）", "每承担一份波动风险，换来多少超额收益。越高越好，0.5 算及格。"),
    ("胜率", "赚钱的期数占全部期数的比例。胜率高但赔钱多，照样可能亏。"),
    ("基准（全市场等权）", "什么都不选、平均买入所有股票的对照组。策略要跑赢它才算真有用。"),
    ("超额", "策略收益减去基准收益。差值为负 = 还不如什么都不干。"),
    ("多空组合", "看好的一边做多、看空的一边做空。"),
    ("临界成本", "手续费高到这个数时，策略刚好就不赚钱了。"),
    ("成本敏感性", "假设手续费是 0.15%、0.30%、0.50% 时，策略分别赚多少 —— 用来测策略对成本的脆弱程度。"),
    ("概率校准", "模型说上涨概率 70% 的时候，实际上是不是真的约七成会涨。"),
    ("Brier 分数", "概率预测的误差分，越小越好。0.25 大致相当于完全瞎猜。"),
    ("AUC", "模型把要涨的股票排在要跌的前面的能力。0.5 = 瞎猜，1 = 完美。"),
    ("命中率", "模型给的方向猜对的比例。"),
    ("门禁", "上模拟仓位前的四道硬性考试。任何一道不过，研究信号就不能用于仓位，只作观察。"),
    ("预测闭环", "每天真的产出预测，到期后用真实行情回头判对错，形成闭环 —— 区别于事后回测。"),
    ("实时预测 live", "当天真正跑出来的预测。"),
    ("历史回溯自检 replay", "用历史行情重跑的模拟预测。只用于检验算法，不代表实盘表现。"),
    ("市场方向", "只判断大盘涨跌，不涉及选股。"),
    ("选股能力", "在大盘涨跌之外，模型挑出来的股票是否比随机挑的更好。"),
    ("按交易日聚合", "同一天的几千条样本高度相关，不能当成几千次独立验证，所以按天算。"),
    ("自由度", "真正能算作独立验证的次数。"),
    ("幸存者偏差", "回测只用今天还在上市的股票，会漏掉当年已退市的公司，从而系统性高估历史收益。"),
    ("Newey-West", "一种统计修正方法，处理相邻日期数据互相重叠导致的置信度虚高。"),
]


# ------------------------------------------------------------------ 顶部导览

def build_guide(
    quotes: dict,
    advice: dict | None,
    review: dict | None,
    evidence: dict | None,
    research: dict | None,
) -> str:
    """报告最顶部的「30 秒看懂」。

    只回答四个问题：市场怎么样 / 我打算怎么配 / 上次战绩如何 /
    我的方法过关没有。每个数字都来自当次渲染的真实数据。
    """
    rows: list[tuple[str, str]] = []

    # ① 今天市场怎么样
    idx = (quotes or {}).get("indexes") or []
    a_idx = [r for r in idx if (r.get("region") or "") == "A股"]
    if a_idx:
        pcts = [r.get("pct") for r in a_idx if isinstance(r.get("pct"), (int, float))]
        if pcts:
            avg = sum(pcts) / len(pcts)
            best = max(a_idx, key=lambda r: r.get("pct") if isinstance(r.get("pct"), (int, float)) else -1e9)
            worst = min(a_idx, key=lambda r: r.get("pct") if isinstance(r.get("pct"), (int, float)) else 1e9)
            tone = "上涨" if avg > 0.05 else ("下跌" if avg < -0.05 else "基本走平")
            rows.append((
                "今天市场",
                f"A 股 {len(a_idx)} 个指数平均 {avg:+.2f}%，整体{tone}。"
                f"最强的是{best.get('name')}（{best.get('pct'):+.2f}%），"
                f"最弱的是{worst.get('name')}（{worst.get('pct'):+.2f}%）。",
            ))
    stocks = (quotes or {}).get("stocks") or []
    if stocks:
        up = sum(1 for s in stocks if isinstance(s.get("pct"), (int, float)) and s["pct"] > 0)
        down = sum(1 for s in stocks if isinstance(s.get("pct"), (int, float)) and s["pct"] < 0)
        flat = len(stocks) - up - down
        if up or down or flat:
            rows.append((
                "股票池",
                f"关注的 {len(stocks)} 只股票里，涨 {up} 只、跌 {down} 只、平 {flat} 只。",
            ))

    # ② 我打算怎么配钱
    alloc = ((advice or {}).get("allocation")) or {}
    mstate = ((advice or {}).get("market_state")) or {}
    if alloc:
        eq = _num(alloc.get("equity_pct"))
        pr = _num(alloc.get("product_pct"))
        ca = _num(alloc.get("cash_pct"))
        rows.append((
            "我怎么配钱",
            f"股票类 {eq}%、理财类 {pr}%、现金 {ca}%。"
            f"意思是：{eq}% 的钱买股票和股票型基金，{pr}% 买货币基金/债券基金这类波动小的，"
            f"{ca}% 留着不动等机会。",
        ))
    if mstate.get("temperature") is not None:
        rows.append((
            "为什么这么配",
            f"系统给当前市场打了 {_num(mstate.get('temperature'))} 分"
            f"（{mstate.get('label') or '—'}）。分数越高越该进攻，越低越该防守，分数直接决定上面那个配比。",
        ))
    rules = ((advice or {}).get("rules")) or {}
    if rules.get("stop_loss_a") is not None:
        rows.append((
            "纪律",
            f"每只股票跌 {_num(rules.get('stop_loss_a'))}% 就无条件卖出，不解释、不硬扛；"
            f"单只仓位不超过 {_num(rules.get('single_max_pct'), 0)}%，总仓位不超过 {_num(rules.get('total_max_pct'), 0)}%。",
        ))

    # ③ 组合里选了什么
    n_eq = (advice or {}).get("n_equity")
    n_pr = (advice or {}).get("n_product")
    if n_eq or n_pr:
        rows.append((
            "选了什么",
            f"股票 {_num(n_eq, 0)} 只、理财 {_num(n_pr, 0)} 只。",
        ))
    by_theme = alloc.get("by_theme") or {}
    if by_theme:
        items = sorted(by_theme.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else 0))
        top = "、".join(f"{k} {v}%" for k, v in items[:3] if isinstance(v, (int, float)))
        if top:
            rows.append(("钱的重心", f"按行业主题看，权重排在前面的是 {top}。"))

    # ④ 上次战绩
    rst = ((review or {}).get("stats")) or {}
    if rst.get("judged"):
        hr = rst.get("hit_rate")
        judged = _num(rst.get("judged"), 0)
        hits = _num(rst.get("hits"), 0)
        verdict = ""
        if hr is not None:
            if hr >= 0.55:
                verdict = "高于抛硬币（50%），说明这套判断暂时站得住。"
            elif hr >= 0.45:
                verdict = "在抛硬币上下浮动，还不能算验证通过。"
            else:
                verdict = "低于抛硬币（50%），说明这阶段择时没被验证，要收缩仓位、只做最有把握的。"
        extra = ""
        if rst.get("stop_trigger_rate") is not None:
            extra = f"止损触发率 {_pct(rst.get('stop_trigger_rate'))}"
        rows.append((
            "上次战绩",
            f"上期 {judged} 条能对账的建议里，方向猜对 {hits} 条，命中率 {_pct(hr)}。{verdict}"
            + (f"另外{extra}。" if extra else ""),
        ))

    # ⑤ 我的方法过关没有
    verdict = (evidence or {}).get("verdict") or {}
    gates = (evidence or {}).get("gates") or []
    if verdict and gates:
        passed = verdict.get("passed")
        total = verdict.get("total")
        allowed = verdict.get("sim_position_allowed")
        ok_names = [g.get("name") for g in gates if g.get("pass")]
        no_names = [g.get("name") for g in gates if not g.get("pass")]
        txt = f"四道门禁过了 {_num(passed, 0)} / {_num(total, 0)} 道。"
        if ok_names:
            txt += f"过了的是 {'、'.join(str(x) for x in ok_names)}。"
        if no_names:
            txt += f"没过的是 {'、'.join(str(x) for x in no_names)}。"
        txt += (
            "所以研究信号现在不进模拟仓位，只作观察记录 —— "
            "这是纪律，不是失败：门禁没过就不下注，比硬上强。"
            if allowed is False or (allowed is None and not verdict.get("pass"))
            else "所以研究信号可以进入模拟仓位观察。"
        )
        rows.append(("方法过关没", txt))

    # ⑥ 今天盯什么
    watch = (research or {}).get("watch") or []
    if watch:
        rows.append(("今天盯什么", str(watch[0])))

    if not rows:
        return ""

    items = "".join(
        f"<div class='gi'><div class='gn'>{_esc(k)}</div><div class='gx'>{_esc(v)}</div></div>"
        for k, v in rows
    )
    return (
        "<div class='guide'>"
        "<div class='gt'>30 秒看懂这份报告</div>"
        "<div class='gs'>下面这几条是全文结论的白话版。想看原始数据和推导过程，再往下翻。</div>"
        f"{items}"
        "</div>"
    )


# ------------------------------------------------------------------ 分节白话说明

_GATE_PLAIN = {
    "G1": ("换个时间段还灵吗", "先用 2015–2020 找到规律，再拿 2023–2026 从没见过的数据考试"),
    "G2": ("扣掉手续费还赚吗", "按 0.30% 往返成本算，年化和风险调整收益是否达标"),
    "G3": ("说七成涨是不是真七成", "模型给的概率和真实涨跌对不对得上"),
    "G4": ("每天真出预测、真回头判对错", "上线后的实际预测到期后有没有验证通过"),
}


def evidence_box(evidence: dict | None) -> str:
    """第十节开头的白话说明：这一节到底在考什么。"""
    if not evidence:
        return ""
    panel = evidence.get("panel") or {}
    segs = evidence.get("segments") or {}

    def _seg(k: str) -> str:
        v = segs.get(k) or []
        return f"{v[0]} → {v[1]}" if len(v) >= 2 else "—"

    rows = [
        "这一节回答一个更根本的问题：<b>我这套选股算法，到底是真有用，还是只是运气好？</b>",
        f"做法是把它当成一场考试：先用 {_seg('train')} 的历史数据"
        f"「学习」，再拿 {_seg('test')} 它从没见过的新数据「考试」。"
        f"全市场 {panel.get('n_codes', '—')} 只股票 × {panel.get('n_dates', '—')} 个交易日，"
        f"考场上每道题都按固定规则算分。",
    ]
    gates = evidence.get("gates") or []
    if gates:
        lines = []
        for g in gates:
            name = str(g.get("name") or "")
            key = name.split(" ")[0]
            plain = _GATE_PLAIN.get(key)
            mark = "通过" if g.get("pass") else "未通过"
            if plain:
                lines.append(f"{name} —— 问的是「{plain[0]}」（{plain[1]}）：{mark}。")
            else:
                lines.append(f"{name}：{mark}。")
        rows.append("<b>四道门禁分别是：</b><br>" + "<br>".join(_esc(x) for x in lines))
    verdict = evidence.get("verdict") or {}
    if verdict:
        allowed = verdict.get("sim_position_allowed")
        if allowed is False or (allowed is None and not verdict.get("pass")):
            rows.append(
                "<b>结论：没过的门禁就是不许下注。</b>研究信号不会进入模拟仓位，只作观察记录 —— "
                "等哪天四道全过，才谈得上拿它做仓位。"
            )
        else:
            rows.append("<b>结论：四道门禁都过了，研究信号可以进入模拟仓位观察。</b>")

    body = "".join(f"<p>{x}</p>" for x in rows)
    return f"<div class='plainbox'><div class='pt'>这一节在考什么（大白话）</div>{body}</div>"


def review_box(review: dict | None) -> str:
    """第九节开头的白话说明：复盘在做什么。"""
    if not review:
        return ""
    rst = review.get("stats") or {}
    rows = [
        "这一节在做一件事：<b>拿我上一期给出的建议，用后来真实发生的行情去对一遍账。</b>",
        "比如上期说「某某股票偏多、涨 8% 分批止盈、跌破某价就止损」，"
        "后来它到底涨了没？跌到止损位了没？账对完之后，才知道我这套判断到底靠不靠谱。",
    ]
    if rst.get("judged"):
        hr = rst.get("hit_rate")
        conclusion = ""
        if hr is not None:
            if hr >= 0.55:
                conclusion = "目前命中率高于抛硬币，这套判断暂时站得住。"
            elif hr >= 0.45:
                conclusion = "目前命中率在抛硬币上下浮动，还不能算验证通过。"
            else:
                conclusion = "目前命中率低于抛硬币，说明这阶段择时没被验证，接下来会收缩仓位、只做最有把握的信号。"
        rows.append(
            f"这一期对账的结果：{_num(rst.get('judged'), 0)} 条能判定的建议里对了 "
            f"{_num(rst.get('hits'), 0)} 条，方向命中率 {_pct(hr)}。{conclusion}"
        )
    else:
        rows.append("目前还没有能对账的历史建议，从今天起，每天的组合都会被记下来，逐日累积。")

    body = "".join(f"<p>{x}</p>" for x in rows)
    return f"<div class='plainbox'><div class='pt'>这一节在做什么（大白话）</div>{body}</div>"


def glossary_html() -> str:
    """术语对照表 —— 报告最后一节，供不懂量化的读者随时查。"""
    rows = [
        f"<tr><td class='gt'>{_esc(term)}</td><td>{_esc(desc)}</td></tr>"
        for term, desc in GLOSSARY
    ]
    return (
        "<h2>十二、术语白话对照表</h2>"
        "<p class='note'>上面各节出现的专业词，这里逐个用大白话解释一遍。"
        "看不懂某个词，先来这里查。</p>"
        "<table><thead><tr><th>术语</th><th>大白话解释</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
        "<div class='analysis'><b>怎么用这张表：</b>"
        "它是全文的字典，不是结论 —— "
        "看到不认识的词先回来查，再回去读那一节的本意，"
        "不要只挑对自己有利的那个解释。"
        f"本表共收录 {len(GLOSSARY)} 个术语。</div>"
    )
