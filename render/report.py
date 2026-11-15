"""结构化 HTML 日报渲染（单文件自包含）。

约束：
- 单文件：不引用任何外部 CSS/JS/图片，离线双击可看。
- 渲染层不造数：所有数字都来自采集结果，缺数据显示占位符。
- 建议类内容一律带「研究性/模拟」标注与免责声明。
"""
from __future__ import annotations

import time
from typing import Any

import analysis as A

# render/plain.py 是本包内的同级模块，必须用相对导入。
# 写成 `import plain as P` 会去找顶层 plain 模块 —— 但工作区根目录
# 没有 plain.py（白话版拆成了 plain_report.py + plain_glossary.py），
# 导致 `import render` 直接 ModuleNotFoundError，连带 3 个测试文件收集失败。
from . import plain as P

CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;padding:28px 22px 60px;background:#0b0e14;color:#e6eaf2;
 font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Helvetica Neue',Arial,sans-serif;
 font-size:14px;line-height:1.7}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px;letter-spacing:.5px}
.sub{color:#8b96a8;font-size:13px;margin-bottom:18px}
.banner{background:#2a1f10;border:1px solid #7a5a20;color:#f0c674;padding:10px 14px;
 border-radius:8px;margin:0 0 20px;font-size:13px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 24px}
.card{flex:1 1 190px;background:#121722;border:1px solid #1e2636;border-radius:10px;padding:12px 14px}
.card .k{color:#8b96a8;font-size:12px;margin-bottom:6px}
.card .v{font-size:20px;font-weight:600;letter-spacing:.3px}
.card .d{color:#6f7a8c;font-size:12px;margin-top:4px}
h2{font-size:16px;margin:28px 0 10px;padding-left:10px;border-left:3px solid #3b82f6;color:#cbd5e1}
table{width:100%;border-collapse:collapse;background:#0f1420;border:1px solid #1e2636;
 border-radius:8px;overflow:hidden;margin-bottom:8px;table-layout:auto}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #1a2130;white-space:nowrap;
 vertical-align:top}
th{background:#151c28;color:#8b96a8;font-weight:500;font-size:12px}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.up{color:#f87171}.down{color:#4ade80}.flat{color:#94a3b8}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;border:1px solid}
.t-ok{color:#4ade80;border-color:#2d5a3d;background:#132a1d}
.t-1{color:#fbbf24;border-color:#5a4a1d;background:#2a2210}
.t-x{color:#f87171;border-color:#5a2d2d;background:#2a1414}
.note{color:#6f7a8c;font-size:12px;margin:6px 0 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
ul{margin:6px 0 0;padding-left:20px}
li{margin:2px 0}
footer{margin-top:34px;padding-top:14px;border-top:1px solid #1e2636;color:#6f7a8c;font-size:12px}
code{background:#161d29;padding:1px 5px;border-radius:4px;font-size:12px}
/* 分析块：紧跟表格，让"看数→看结论"不脱节 */
.analysis{background:#0f1420;border:1px solid #1e2636;border-left:3px solid #3b82f6;
 border-radius:8px;padding:11px 14px;margin:0 0 26px}
.analysis .at{color:#7dd3fc;font-size:12px;font-weight:600;letter-spacing:.4px;margin-bottom:6px}
.analysis ul{margin:0;padding-left:18px;color:#c3ccdb;font-size:13px}
.analysis li{margin:3px 0;line-height:1.65}
/* 每日研报 */
.research{background:#101726;border:1px solid #2b3a55;border-radius:10px;padding:16px 18px;margin:0 0 24px}
.research .rt{font-size:15px;font-weight:600;color:#e8eefb;margin-bottom:4px}
.research .rs{font-size:12px;color:#8b96a8;margin-bottom:12px}
.research .rk{color:#7dd3fc;font-size:12px;font-weight:600;margin:11px 0 4px;letter-spacing:.3px}
.research p{margin:0 0 2px;color:#c3ccdb;font-size:13px;line-height:1.75}
.research ul{margin:2px 0 0;padding-left:20px;color:#c3ccdb;font-size:13px}
.research li{margin:3px 0;line-height:1.7}
.research .rd{border-top:1px solid #1e2636;margin-top:13px;padding-top:10px;color:#6f7a8c;font-size:12px}
/* 白话层：让不懂量化的人也能读懂结论 */
.guide{background:#101726;border:1px solid #2b3a55;border-radius:10px;padding:16px 18px;margin:0 0 22px}
.guide .gt{font-size:15px;font-weight:600;color:#e8eefb;margin-bottom:4px}
.guide .gs{font-size:12px;color:#8b96a8;margin-bottom:11px}
.guide .gi{display:flex;gap:10px;margin:7px 0;align-items:flex-start}
.guide .gn{color:#7dd3fc;font-size:12px;font-weight:600;white-space:nowrap;
 min-width:74px;padding-top:1px}
.guide .gx{color:#c3ccdb;font-size:13px;line-height:1.75}
@media(max-width:560px){.guide .gi{display:block}.guide .gn{margin-bottom:2px}}
/* 分节白话说明 */
.plainbox{background:#0f1420;border:1px solid #1e2636;border-left:3px solid #7dd3fc;
 border-radius:8px;padding:12px 15px;margin:0 0 18px}
.plainbox .pt{color:#7dd3fc;font-size:12px;font-weight:600;letter-spacing:.4px;margin-bottom:7px}
.plainbox p{color:#c3ccdb;font-size:13px;line-height:1.8;margin:0 0 7px}
.plainbox p:last-child{margin-bottom:0}
.plainbox b{color:#e8eefb}
table td.gt{color:#7dd3fc;font-weight:500;white-space:nowrap}
"""


def esc(v: Any) -> str:
    s = "" if v is None else str(v)
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def f(v: Any, nd: int = 2, dash: str = "—") -> str:
    if v is None:
        return dash
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int, float)):
        return f"{v:,.{nd}f}"
    return esc(v)


def p(v: Any, nd: int = 2) -> str:
    if v is None:
        return '<span class="flat">—</span>'
    try:
        x = float(v)
    except (TypeError, ValueError):
        return esc(v)
    cls = "up" if x > 0 else ("down" if x < 0 else "flat")
    return f'<span class="{cls}">{x:+.{nd}f}%</span>'


def num(v: Any, nd: int = 4) -> str:
    """普通小数（不带 % 号）：IC、t 值、Sharpe、Brier、AUC。"""
    if v is None:
        return '<span class="flat">—</span>'
    try:
        x = float(v)
    except (TypeError, ValueError):
        return esc(v)
    cls = "up" if x > 0 else ("down" if x < 0 else "flat")
    return f'<span class="{cls}">{x:.{nd}f}</span>'


def pct(v: Any, nd: int = 2) -> str:
    """比率按百分比显示（自动 ×100，带正负号）：年化收益、超额。"""
    if v is None:
        return '<span class="flat">—</span>'
    try:
        x = float(v) * 100.0
    except (TypeError, ValueError):
        return esc(v)
    cls = "up" if x > 0 else ("down" if x < 0 else "flat")
    return f'<span class="{cls}">{x:+.{nd}f}%</span>'


def rate(v: Any, nd: int = 2) -> str:
    """比率按百分比显示（自动 ×100，不带正负号）：命中率、胜率。"""
    if v is None:
        return '<span class="flat">—</span>'
    try:
        x = float(v) * 100.0
    except (TypeError, ValueError):
        return esc(v)
    cls = "up" if x > 0 else ("down" if x < 0 else "flat")
    return f'<span class="{cls}">{x:.{nd}f}%</span>'


def cls_p(v: Any) -> str:
    """涨跌幅单元格（自带配色 class）。"""
    return p(v)


def tag(status: Any) -> str:
    m = {
        "verified": ("t-ok", "双源核验"),
        "single_source": ("t-1", "单源"),
        "conflict": ("t-x", "有冲突"),
    }
    c, label = m.get(status, ("t-1", "未核验" if status is None else str(status)))
    return f'<span class="tag {c}">{esc(label)}</span>'


def table(headers: list[str], rows: list[list[str]], *, numeric: list[int] | None = None) -> str:
    """渲染表格。

    numeric 里的列是数值列：<td> 右对齐、<th> 也必须右对齐，
    否则表头和数字列会天然错开一行（这是之前排版乱的根因）。

    结构校验：每行的单元格数必须等于表头列数，否则表头与数据必然错位。
    这里直接抛错而不是静默补齐 —— 排版错误必须在生成阶段暴露，不能拖到投递后才发现。
    """
    numeric = numeric or []
    ncol = len(headers)
    for i, r in enumerate(rows):
        if len(r) != ncol:
            raise ValueError(
                f"表格列数不一致：表头 {ncol} 列（{headers}），"
                f"第 {i + 1} 行 {len(r)} 个单元格（{r}）"
            )
    for i in numeric:
        if not (0 <= i < ncol):
            raise ValueError(f"numeric 列号 {i} 越界，表头只有 {ncol} 列")
    if not rows:
        return '<div class="note">（本期无数据）</div>'
    th = "".join(
        f'<th class="num">{esc(h)}</th>' if i in numeric else f"<th>{esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for r in rows:
        tds = []
        for i, cell in enumerate(r):
            c = ' class="num"' if i in numeric else ""
            tds.append(f"<td{c}>{cell}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (
        f"<table><thead><tr>{th}</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def analysis(title: str, lines: list[str]) -> str:
    """表格后的分析块：标题 + 要点列表。"""
    if not lines:
        return ""
    lis = "".join(f"<li>{ln}</li>" for ln in lines)
    return (
        f'<div class="analysis"><div class="at">小金分析 · {esc(title)}</div>'
        f"<ul>{lis}</ul></div>"
    )


def section(title: str, body: str, note: str = "") -> str:
    n = f'<p class="note">{esc(note)}</p>' if note else ""
    return f"<h2>{esc(title)}</h2>{body}{n}"


def card(k: str, v: str, d: str = "") -> str:
    dd = f'<div class="d">{esc(d)}</div>' if d else ""
    return f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{v}</div>{dd}</div>'


def _index_rows(indexes: list[dict]) -> list[list[str]]:
    return [
        [
            esc(r.get("name")),
            esc(r.get("region")),
            f(r.get("price")),
            cls_p(r.get("pct")),
            f((r.get("amount") or 0) / 1e8, 0) + "亿" if r.get("amount") else "—",
            esc(r.get("trade_date") or "—"),
            tag((r.get("cross") or {}).get("status")),
        ]
        for r in indexes
    ]


def _stock_rows(stocks: list[dict], limit: int | None = None) -> list[list[str]]:
    rows = []
    for r in stocks[:limit] if limit else stocks:
        rows.append(
            [
                esc(r.get("name")),
                esc(r.get("theme")),
                esc(r.get("sector")),
                f(r.get("price")),
                cls_p(r.get("pct")),
                f(r.get("pe")),
                f(r.get("pb")),
                esc(r.get("trade_date") or "—"),
                tag((r.get("cross") or {}).get("status")),
            ]
        )
    return rows


def _portfolio_rows(portfolio: list[dict]) -> list[list[str]]:
    rows = []
    for p_ in portfolio:
        if p_.get("kind") == "stock":
            targets = " / ".join(
                f"{t['gain_pct']:+.0f}%→{t['price']}" for t in (p_.get("targets") or [])
            )
            rows.append(
                [
                    esc(p_.get("name")),
                    "股票 · " + esc(p_.get("theme")),
                    f(p_.get("weight_pct"), 1) + "%",
                    f(p_.get("price")),
                    f"{p_.get('stop_pct'):+.0f}%→{p_.get('stop_price')}" if p_.get("stop_price") else "—",
                    targets or "—",
                    esc(p_.get("call") or "—"),
                    tag(p_.get("verify")),
                    esc(p_.get("rationale")),
                ]
            )
        else:
            rows.append(
                [
                    esc(p_.get("name")),
                    "理财 · " + esc(p_.get("category")),
                    f(p_.get("weight_pct"), 1) + "%",
                    f(p_.get("nav"), 4) if p_.get("nav") is not None else "—",
                    "—",
                    (f"7日年化 {f(p_.get('seven_day_yield'), 3)}%" if p_.get("seven_day_yield") is not None
                     else (f"近1年 {p(p_.get('ret_1y'))}" if p_.get("ret_1y") is not None else "—")),
                    "—",
                    tag(p_.get("verify")),
                    esc(p_.get("rationale")),
                ]
            )
    return rows


def _evidence_section(evidence: dict | None) -> str:
    """「研究能力实测」板块：因子时间外 / 成本后 / 概率校准 / 预测闭环。

    每个表后面必须紧跟一段基于本表真实数据的分析（回归测试会检查这一点）。
    """
    if not evidence:
        return ""
    import research_evidence as RE

    parts: list[str] = []
    panel = evidence.get("panel") or {}
    segs = evidence.get("segments") or {}

    def _seg(k: str) -> str:
        v = segs.get(k) or []
        if len(v) < 2:
            return "—"
        return f"{v[0]} → {v[1]}"

    parts.append(
        "<p class='note'>样本：全市场 "
        + esc(panel.get("n_codes", "—"))
        + " 只 × "
        + esc(panel.get("n_dates", "—"))
        + " 个交易日（"
        + esc(panel.get("from", "—"))
        + " → "
        + esc(panel.get("to", "—"))
        + "）；训练段 "
        + esc(_seg("train"))
        + "，测试段 "
        + esc(_seg("test"))
        + "；前瞻窗口 "
        + esc(evidence.get("horizon", "—"))
        + " 个交易日。因子发现只用训练段，测试段在发现时从未被看过。</p>"
    )

    # ---- 每日闭环运行状态（刷新 → 预测 → 判定）
    # 必须显式渲染：闭环失败时读者要看到"今天没跑通"，而不是以为
    # 下面的命中率就是今天的实时表现。
    dl = evidence.get("daily_loop") or {}
    dl_stages = dl.get("stages") or {}
    if dl_stages:
        dl_rows: list[list[str]] = []
        for name in ("refresh", "predict", "judge"):
            st = dl_stages.get(name) or {}
            status = st.get("status")
            if status == "ok":
                mark = '<span class="tag t-ok">完成</span>'
                note = "本次运行成功"
                if name == "predict":
                    note = "本次运行成功" + (
                        f"（预测日 {esc(st.get('detail', {}).get('day'))}）"
                        if (st.get("detail") or {}).get("day") else ""
                    )
            elif status == "failed":
                mark = '<span class="tag t-x">失败</span>'
                note = esc(st.get("error") or "原因未记录")
            else:
                mark = '<span class="tag t-1">未运行</span>'
                note = "今天还没跑过"
            dl_rows.append([esc(RE._STAGE_LABEL.get(name, name)), mark, note])
        banner = ""
        if dl.get("failed"):
            banner = (
                "<div class='banner' style='background:#2a1414;border:1px solid #5a2d2d;color:#f87171'>"
                "今日每日闭环未跑通。下文的命中率只反映<strong>已落盘的历史判定</strong>，"
                "不代表今天的实时表现。</div>"
            )
        parts.append(
            "<h3 style='font-size:13px;color:#94a3b8;margin:18px 0 6px'>"
            "每日闭环运行状态（刷新行情 → 生成 live 预测 → 用真实行情判定）</h3>"
            + banner
            + table(["环节", "状态", "说明"], dl_rows)
            + analysis("每日闭环", RE.analyze_daily_loop(evidence))
        )

    # ---- 门禁
    gates = evidence.get("gates") or []
    if gates:
        gate_rows = [
            [esc(g.get("name")), "通过" if g.get("pass") else "未通过", esc(g.get("detail"))]
            for g in gates
        ]
        parts.append(table(["门禁", "结果", "说明"], gate_rows))
        parts.append(analysis("研究门禁", RE.analyze_gates(evidence)))

    # ---- 实盘就绪度（2026-11-12）
    # 门禁说「方法站不站得住」，就绪度说「拿真钱上会怎样」。分开讲。
    lr = evidence.get("live_readiness") or {}
    if lr.get("available"):
        lr_rows = [
            ["打平成本（往返）", pct(lr.get("breakeven_cost"), 4),
             "交易成本低于此值才追得平等权基准"],
            ["当前成本假设（往返）", pct(lr.get("actual_cost"), 2),
             f"高出打平线 {lr.get('cost_gap_multiple'):.2f} 倍"
             if lr.get("cost_gap_multiple") else "—"],
            ["平均单边换手", rate(lr.get("turnover"), 2), "每期换手比例"],
            ["维持当前成本时允许的最高换手", rate(lr.get("max_turnover_at_cost"), 2),
             "超过这个数，成本就吃掉全部超额"],
            ["毛超额（未扣成本）", pct(lr.get("gross_excess_annualized"), 2), "信号本身的选股能力"],
            ["净超额（已扣成本）", pct(lr.get("net_excess_annualized"), 2), "扣成本后还剩多少"],
            ["相对等权基准超额", pct(lr.get("excess_vs_benchmark"), 2),
             "为负说明赚的是市场 beta，不是选股 alpha"],
            ["年化 · 均值口径", pct(lr.get("mean_annualized"), 2), "被极端盈利月拉高"],
            ["年化 · 中位数口径", pct(lr.get("median_annualized"), 2), "实盘心理预期的锚"],
            ["均值-中位数落差", pct(lr.get("mean_median_spread"), 2), "落差越大越要按中位数做预期"],
        ]
        parts.append(
            "<h3 style='font-size:13px;color:#94a3b8;margin:18px 0 6px'>"
            "实盘就绪度（拿真钱上会怎样）</h3>"
            + table(["指标", "数值", "含义"], lr_rows, numeric=[1])
        )
        parts.append(analysis("实盘就绪度", RE.analyze_live_readiness(evidence)))

        # 逐项判定
        checks = lr.get("checks") or []
        if checks:
            ck_rows = [
                [
                    esc(c.get("id")),
                    esc(c.get("name")),
                    '<span class="tag t-ok">通过</span>' if c.get("pass")
                    else '<span class="tag t-x">未通过</span>',
                    esc(c.get("detail")),
                ]
                for c in checks
            ]
            parts.append(
                "<h3 style='font-size:13px;color:#94a3b8;margin:18px 0 6px'>"
                "实盘判定逐项（通过 "
                f"{lr.get('passed')}/{lr.get('total')}）</h3>"
                + table(["编号", "判定项", "结果", "说明"], ck_rows)
            )
            parts.append(analysis("实盘判定", RE.analyze_live_readiness(evidence)))
    elif lr:
        parts.append(
            "<h3 style='font-size:13px;color:#94a3b8;margin:18px 0 6px'>"
            "实盘就绪度（本期不可用）</h3>"
            + analysis("实盘就绪度", RE.analyze_live_readiness(evidence))
        )

    # ---- 因子时间外
    factors = evidence.get("factors") or []
    if factors:
        fac_rows = []
        for fa in factors:
            fac_rows.append([
                esc(fa.get("name")),
                esc(fa.get("desc") or ""),
                num(fa.get("discovered_mean"), 4),
                num(fa.get("discovered_t"), 2),
                num(fa.get("oos_mean"), 4),
                num(fa.get("oos_t"), 2),
                esc(fa.get("verdict") or ""),
            ])
        parts.append(
            "<h2 style='font-size:14px;border-left-color:#475569'>因子时间外验证</h2>"
            + table(
                ["因子", "含义", "发现期 IC", "发现期 t", "测试期 IC", "测试期 t", "判定"],
                fac_rows,
                numeric=[2, 3, 4, 5],
            )
        )
        parts.append(analysis("因子时间外验证", RE.analyze_factor_oos(evidence)))

    # ---- 合成信号诊断：IC 与尾部价差
    diag = evidence.get("signal_diagnostic") or {}
    if diag.get("ic_mean") is not None:
        cons = diag.get("ic_tail_consistent")
        drows = [
            ["合成信号 · 测试段逐日 IC", num(diag.get("ic_mean"), 4)],
            ["IC 的 t 值", num(diag.get("ic_t"), 2)],
            ["IC 样本数", f"{diag.get('ic_n', 0):,}"],
            ["毛十分位价差 · 均值口径", pct(diag.get("tail_spread_mean"), 4)],
            ["　└ 为正的比例", rate(diag.get("tail_positive_ratio"), 1)],
            ["毛十分位价差 · 中位数口径", pct(diag.get("tail_spread_median"), 4)],
            ["　└ 为正的比例", rate(diag.get("tail_median_positive_ratio"), 1)],
            ["两者方向是否一致（均值口径）", "一致" if cons else ("相反" if cons is False else "—")],
            ["两者方向是否一致（中位口径）",
             "一致" if diag.get("ic_tail_consistent_median") else
             ("相反" if diag.get("ic_tail_consistent_median") is False else "—")],
        ]
        parts.append(
            "<h2 style='font-size:14px;border-left-color:#475569'>合成信号诊断（IC vs 尾部价差）</h2>"
            + "<p class='note'>IC 衡量整体单调性，尾部价差衡量极端分位的收益差。"
              "两者可以不一致 —— 不一致就说明不能靠分位收割。"
              "均值口径会被少数极端收益压平，所以必须同时看中位数口径："
              "中位数与 IC 同向、均值反向时，结论是"
              "<b>信号在秩关系上成立，但被极端值扭曲了收益分布</b>，"
              "而不是「信号无效」。</p>"
            + table(["指标", "数值"], drows, numeric=[1])
        )
        parts.append(analysis("合成信号诊断", RE.analyze_diagnostic(evidence)))

    # ---- 成本后策略
    strat = evidence.get("strategy") or {}
    if strat.get("ok"):
        lo = strat.get("long_only") or {}
        ls = strat.get("long_short") or {}
        bm = strat.get("benchmark") or {}
        lo_raw = strat.get("long_only_raw") or {}
        lo_med = strat.get("long_only_median") or {}
        srows = [
            ["多头组（最高分位 · 横截面缩尾均值）", pct(lo.get("net_annualized"), 4), num(lo.get("sharpe"), 2),
             pct(lo.get("net_total_return"), 4), rate(lo.get("win_rate"), 4),
             pct(lo.get("net_annualized", 0) - bm.get("annualized", 0), 4)],
            ["多头组（未缩尾均值口径）", pct(lo_raw.get("net_annualized"), 4), num(lo_raw.get("sharpe"), 2),
             pct(lo_raw.get("net_total_return"), 4), rate(lo_raw.get("win_rate"), 4),
             pct(lo_raw.get("net_annualized", 0) - bm.get("annualized", 0), 4)],
            ["多头组（中位数口径）", pct(lo_med.get("net_annualized"), 4), num(lo_med.get("sharpe"), 2),
             pct(lo_med.get("net_total_return"), 4), rate(lo_med.get("win_rate"), 4),
             pct(lo_med.get("net_annualized", 0) - bm.get("annualized", 0), 4)],
            ["多空组合", pct(ls.get("net_annualized"), 4), num(ls.get("sharpe"), 2),
             pct(ls.get("net_total_return"), 4), rate(ls.get("win_rate"), 4),
             pct(ls.get("net_annualized", 0) - bm.get("annualized", 0), 4)],
            ["基准（全市场等权）", pct(bm.get("annualized"), 4), "—",
             pct(bm.get("total_return"), 4), rate(bm.get("win_rate"), 4), "0.0000"],
        ]
        parts.append(
            "<h2 style='font-size:14px;border-left-color:#475569'>成本后策略（测试期）</h2>"
            + "<p class='note'>"
            + esc(f"每 {strat.get('hold', '—')} 个交易日调仓一次，"
                  f"共 {strat.get('n_periods', 0)} 期；平均单边换手 "
                  f"{strat.get('mean_turnover', 0):.1%}；"
                  f"成本按 {evidence.get('base_cost', 0.003):.2%}/次往返计入（佣金+印花税+滑点）。"
                  "多头组给出三个口径：横截面缩尾均值（主口径，最接近可执行）、"
                  "未缩尾均值（披露极值拖累）、中位数（抗极值，与秩 IC 应同向）；"
                  "三者的差距就是极端收益对结论的影响幅度。")
            + "</p>"
            + table(
                ["组合", "年化（净）", "Sharpe", "累计收益", "胜率", "相对基准年化"],
                srows,
                numeric=[1, 2, 3, 4, 5],
            )
        )
        parts.append(analysis("成本后策略", RE.analyze_strategy(evidence)))

        sens = evidence.get("cost_sensitivity") or []
        if len(sens) >= 2:
            sens_rows = [
                [
                    num(s.get("cost_rate"), 4),
                    pct(s.get("long_only_annualized"), 4),
                    num(s.get("long_only_sharpe"), 2),
                    pct(s.get("long_short_annualized"), 4),
                    num(s.get("long_short_sharpe"), 2),
                    pct(s.get("long_only_vs_bench_annualized"), 4),
                ]
                for s in sens if s.get("ok")
            ]
            if sens_rows:
                parts.append(
                    "<h2 style='font-size:14px;border-left-color:#475569'>成本敏感性</h2>"
                    + table(
                        ["单次往返成本", "多头年化", "多头 Sharpe", "多空年化", "多空 Sharpe", "多头超额"],
                        sens_rows,
                        numeric=[0, 1, 2, 3, 4, 5],
                    )
                )
                parts.append(analysis("成本敏感性", RE.analyze_strategy(evidence)))

    # ---- 概率模型
    prob = evidence.get("probability") or {}
    prows = []
    for key, label in (("rel", "相对（跑赢横截面中位数）"), ("abs", "绝对（绝对上涨）")):
        for cal_key, cal_label in (("raw", "未校准"), ("platt", "Platt"), ("isotonic", "保序")):
            m = ((prob.get(key) or {}).get(cal_key)) or {}
            if not m:
                continue
            prows.append([
                esc(label), esc(cal_label),
                f"{m.get('n', 0):,}",
                num(m.get("brier"), 5),
                num(m.get("brier_baseline"), 5),
                num(m.get("auc"), 4),
                rate(m.get("hit_rate"), 4),
                rate(m.get("baseline_hit_rate"), 4),
            ])
    if prows:
        parts.append(
            "<h2 style='font-size:14px;border-left-color:#475569'>概率模型（时间外）</h2>"
            + table(
                ["标签", "校准方式", "测试样本", "Brier", "基准 Brier", "AUC", "命中率", "基准命中率"],
                prows,
                numeric=[3, 4, 5, 6, 7],
            )
        )
        parts.append(analysis("概率模型", RE.analyze_probability(evidence)))

    # ---- 预测闭环（G4 四件套：总览 / 口径拆分 / 选股价差 / 按天聚合 / 判据明细）
    loop = evidence.get("prediction_loop") or {}
    verdict = evidence.get("verdict") or {}
    parts.append(
        "<h2 style='font-size:14px;border-left-color:#475569'>预测闭环（真实行情事后判定）</h2>"
    )
    if verdict:
        # sim_position_allowed 缺失时退回 verdict.pass —— 不能因为一个字段没写就当成"放行"
        allowed = verdict.get("sim_position_allowed")
        if allowed is None:
            allowed = bool(verdict.get("pass"))
        allowed = bool(allowed)
        style = (
            "background:#12241a;border:1px solid #2d5a3d;color:#4ade80"
            if allowed
            else "background:#2a1414;border:1px solid #5a2d2d;color:#f87171"
        )
        parts.append(
            f"<div style='{style};padding:10px 14px;border-radius:8px;"
            "margin:0 0 16px;font-size:13px'>"
            f"<b>模拟仓位放行：{'是' if allowed else '否'}</b> · "
            f"{esc(verdict.get('text') or '')}</div>"
        )

    # ① 闭环总览
    lrows: list[list[str]] = []
    if loop.get("available"):
        lrows = [
            ["已判定样本", f"{loop.get('n', 0):,}"],
            ["覆盖交易日数", f"{loop.get('n_days', 0)}"],
            ["方向命中率", rate(loop.get("hit_rate"), 4)],
            ["基准命中率（全押方向）", rate(loop.get("baseline_hit_rate"), 4)],
            ["相对基准超额", num(loop.get("edge_vs_baseline"), 4)],
            ["Brier 分数", num(loop.get("brier"), 5)],
            ["平均预测概率", num(loop.get("avg_prob"), 4)],
            ["实际上涨频率", num(loop.get("actual_rate"), 4)],
        ]
    else:
        lrows = [["状态", "尚未产生已到期的预测，预测闭环暂未启动，暂不评价命中率。"]]
    parts.append(
        "<h3 style='font-size:13px;color:#94a3b8;margin:14px 0 6px'>① 闭环总览</h3>"
        + table(["指标", "数值"], lrows, numeric=[1] if loop.get("available") else None)
    )
    parts.append(analysis("闭环总览", RE.analyze_loop_overview(evidence)))

    # ② 口径拆分：市场方向 / 实时 / 回溯 —— 三个来源分开，不许混报
    live = loop.get("live") or {}
    replay = loop.get("replay") or {}
    mkt = loop.get("market") or {}
    if loop.get("available"):
        split_rows = [
            [
                "市场方向（独立于选股）", "—", f"{mkt.get('n', 0):,}",
                rate(mkt.get("hit_rate"), 2), "—", "—", "—",
                pct(mkt.get("avg_fwd_return"), 4),
            ],
            [
                "实时预测 live", f"{live.get('n_days', 0)}", f"{live.get('n', 0):,}",
                rate(live.get("hit_rate"), 2), rate(live.get("baseline_hit_rate"), 2),
                num(live.get("edge_vs_baseline"), 4), num(live.get("brier"), 5),
                pct(live.get("avg_fwd_return"), 4),
            ],
            [
                "历史回溯自检 replay", f"{replay.get('n_days', 0)}", f"{replay.get('n', 0):,}",
                rate(replay.get("hit_rate"), 2), rate(replay.get("baseline_hit_rate"), 2),
                num(replay.get("edge_vs_baseline"), 4), num(replay.get("brier"), 5),
                pct(replay.get("avg_fwd_return"), 4),
            ],
        ]
        parts.append(
            "<h3 style='font-size:13px;color:#94a3b8;margin:18px 0 6px'>"
            "② 口径拆分（市场方向 / 实时 / 回溯）</h3>"
            + table(
                ["口径", "预测日数", "已判定样本", "方向命中率", "基准命中率",
                 "相对基准超额", "Brier", "平均前瞻收益"],
                split_rows,
                numeric=[1, 2, 3, 4, 5, 6, 7],
            )
        )
        parts.append(analysis("口径拆分", RE.analyze_loop_split(evidence)))

    # ③ 选股能力：十分位价差（对冲掉市场方向）
    pf = loop.get("portfolio") or {}
    if pf.get("available"):
        prows = [
            ["可评估交易日数", f"{pf.get('n_days', 0)}"],
            ["头部组合（最高分位）平均前瞻收益", pct(pf.get("avg_top_decile_return"), 4)],
            ["尾部组合（最低分位）平均前瞻收益", pct(pf.get("avg_bottom_decile_return"), 4)],
            ["十分位价差 · 均值口径", pct(pf.get("avg_spread"), 4)],
            ["十分位价差 · 中位数口径（门禁口径）", pct(pf.get("median_spread"), 4)],
            ["全市场等权平均前瞻收益", pct(pf.get("avg_equal_weight_return"), 4)],
            ["头部减等权", pct(pf.get("avg_top_minus_bw"), 4)],
            ["正价差天数占比", rate(pf.get("pct_days_spread_positive"), 2)],
        ]
        parts.append(
            "<h3 style='font-size:13px;color:#94a3b8;margin:18px 0 6px'>"
            "③ 选股能力（十分位价差，对冲掉市场方向）</h3>"
            + table(["指标", "数值"], prows, numeric=[1])
        )
        parts.append(analysis("选股能力", RE.analyze_loop_selection(evidence)))

    # ④ 按交易日聚合：同日样本高度相关，自由度按天算
    dl = loop.get("day_level") or {}
    if dl.get("available"):
        drows = [
            ["可聚合交易日数", f"{dl.get('n_days', 0)}"],
            ["平均每日命中率", rate(dl.get("avg_daily_hit_rate"), 4)],
            ["中位每日命中率", rate(dl.get("median_daily_hit_rate"), 4)],
            ["跑赢硬币的天数占比", rate(dl.get("pct_days_hit_above_50"), 2)],
            ["大盘上涨的天数占比", rate(dl.get("pct_days_actual_up_above_50"), 2)],
            ["平均每日实际上涨率", rate(dl.get("avg_daily_actual_rate"), 4)],
        ]
        parts.append(
            "<h3 style='font-size:13px;color:#94a3b8;margin:18px 0 6px'>"
            "④ 按交易日聚合（自由度口径）</h3>"
            + table(["指标", "数值"], drows, numeric=[1])
        )
        parts.append(analysis("按交易日聚合", RE.analyze_loop_day_level(evidence)))

    # ⑤ G4 判据明细：哪一条没过、差多少
    g4 = next(
        (g for g in (evidence.get("gates") or []) if str(g.get("name", "")).startswith("G4")),
        None,
    )
    cd = (g4 or {}).get("criteria_detail") or []
    if cd:
        crows = [
            [
                esc(c.get("name")),
                esc(c.get("threshold")),
                esc(c.get("actual")),
                '<span class="tag t-ok">达标</span>' if c.get("pass")
                else '<span class="tag t-x">未达标</span>',
            ]
            for c in cd
        ]
        parts.append(
            "<h3 style='font-size:13px;color:#94a3b8;margin:18px 0 6px'>"
            "⑤ G4 判据明细（哪一条没过、差多少）</h3>"
            + table(["判据", "门槛", "实测", "结果"], crows)
        )
        parts.append(analysis("G4 判据", RE.analyze_g4_criteria(evidence)))

    return "".join(parts)


def render_html(
    quotes: dict,
    events: dict,
    futures: dict,
    *,
    products: dict | None = None,
    advice: dict | None = None,
    review: dict | None = None,
    session: dict | None = None,
    research: dict | None = None,
    evidence: dict | None = None,
    brief_text: str | None = None,
    brief_error: str | None = None,
    title: str = "财经日报",
    now: str | None = None,
) -> str:
    """产出单文件 HTML 日报。"""
    now = now or time.strftime("%Y-%m-%d %H:%M")
    prods = (products or {}).get("products") or []
    evs = events.get("events") or []
    stats = events.get("stats") or {}
    qsrcs = quotes.get("sources") or {}
    esrcs = events.get("sources") or {}
    psrcs = (products or {}).get("sources") or {}
    session = session or {}

    def cnt(d: dict) -> str:
        if not isinstance(d, dict):
            return "—"
        ok = sum(1 for v in d.values() if isinstance(v, dict) and v.get("ok"))
        return f"{ok}/{len(d)}"

    # ---------------------------------------------------------- 卡片
    alloc = (advice or {}).get("allocation") or {}
    mstate = (advice or {}).get("market_state") or {}
    conflicts = (
        (quotes.get("conflicts") or [])
        + (futures.get("conflicts") or [])
        + ((products or {}).get("conflicts") or [])
    )
    cards = "".join(
        [
            card("大盘温度", f"{mstate.get('temperature', '—')}", mstate.get("label", "")),
            card("权益仓位", f"{alloc.get('equity_pct', '—')}%", f"上限 {alloc.get('equity_target_pct', '—')}%"),
            card("理财仓位", f"{alloc.get('product_pct', '—')}%", f"现金 {alloc.get('cash_pct', '—')}%"),
            card("事件双源", f"{stats.get('events_verified', 0)}/{stats.get('events_total', 0)}", "重要事件已核验"),
            card("数据源", f"{cnt(qsrcs)}/{cnt(esrcs)}/{cnt(psrcs)}", "行情/事件/理财"),
            card("口径冲突", f"{len(conflicts)}", "已标注未静默取舍"),
            card(
                "交易日状态",
                "交易日" if session.get("is_trading_day") else "休市",
                f"行情对应 {session.get('data_date') or '—'}",
            ),
        ]
    )

    parts: list[str] = []

    # ---------------------------------------------------------- 白话导览
    # 放在最前面（卡片之后、正文之前）：不懂量化的人先看这几条，
    # 就能读出全文结论，再决定要不要往下看推导过程。
    guide_html = P.build_guide(quotes, advice, review, evidence, research)

    # ---------------------------------------------------------- 大盘
    idx_rows = _index_rows(quotes.get("indexes") or [])
    parts.append(
        section(
            "一、大盘概览",
            table(
                ["指数", "市场", "点位", "涨跌", "成交额", "交易日", "核验"],
                idx_rows,
                numeric=[2, 3, 4],
            ),
        )
        + analysis(
            "大盘概览",
            A.analyze_indexes(quotes.get("indexes") or []),
        )
    )

    # ---------------------------------------------------------- 建议（前置）
    if advice:
        st = advice.get("market_state") or {}
        comp = st.get("components") or {}
        parts.append(
            section(
                "二、模拟组合建议（研究性，非投资建议）",
                f'<div class="banner">{esc(advice.get("disclaimer", ""))}</div>'
                + table(
                    [
                        "标的", "类别", "建议权重", "参考价", "止损位", "目标位（分批）",
                        "方向", "核验", "依据",
                    ],
                    _portfolio_rows(advice.get("portfolio") or []),
                    numeric=[2, 3, 4],
                )
                + (
                    "<p class='note'>仓位结构：权益 "
                    + f(alloc.get("equity_pct"), 1)
                    + "%（目标 "
                    + f(alloc.get("equity_target_pct"), 1)
                    + "%）、理财 "
                    + f(alloc.get("product_pct"), 1)
                    + "%（目标 "
                    + f(alloc.get("product_target_pct"), 1)
                    + "%）、现金 "
                    + f(alloc.get("cash_pct"), 1)
                    + "%（不少于 "
                    + f(alloc.get("cash_min_pct"), 0)
                    + "%）。单只上限 "
                    + f((advice.get("rules") or {}).get("single_max_pct"), 0)
                    + "%，总仓位上限 "
                    + f((advice.get("rules") or {}).get("total_max_pct"), 0)
                    + "%。</p>"
                ),
                note="大盘温度 " + f(st.get("temperature"), 1) + "（" + str(st.get("label")) + "）；构成："
                + "、".join(f"{k} {v}" for k, v in comp.items()),
            )
            + analysis("模拟组合建议", A.analyze_portfolio(advice))
        )

        wl = advice.get("watchlist") or []
        parts.append(
            section(
                "三、关注标的清单（全池评分）",
                table(
                    ["标的", "主题", "板块", "现价", "涨跌", "PE", "PB", "评分", "核验", "入选"],
                    [
                        [
                            esc(w.get("name")),
                            esc(w.get("theme")),
                            esc(w.get("sector")),
                            f(w.get("price")),
                            cls_p(w.get("pct")),
                            f(w.get("pe")),
                            f(w.get("pb")),
                            f(w.get("score"), 1),
                            tag(w.get("verify")),
                            "是" if w.get("in_portfolio") else "否",
                        ]
                        for w in wl
                    ],
                    numeric=[3, 4, 5, 6, 7],
                ),
                note="评分 = 动量(0-50) + 估值(0-30) + 流动性(0-10) + 核验(0-10)；双源冲突标的不进组合。",
            )
            + analysis("关注标的清单", A.analyze_watchlist(wl))
        )

    # ---------------------------------------------------------- 事件
    high = [e for e in evs if e.get("importance") == "high"]
    verified = [e for e in high if (e.get("cross") or {}).get("status") == "verified"]
    ev_rows = [
        [
            esc((e.get("time") or "")[5:16]),
            esc(e.get("title")),
            esc(", ".join((e.get("cross") or {}).get("src_names") or [])),
            tag((e.get("cross") or {}).get("status")),
            esc(e.get("uri") or ""),
        ]
        for e in verified[:40]
    ]
    parts.append(
        section(
            "四、重点事件（双源核验）",
            table(["时间", "事件", "来源", "核验", "链接"], ev_rows),
            note=f"共采集 {len(evs)} 条，其中重要 {len(high)} 条、双源核验 {len(verified)} 条。",
        )
        + analysis("重点事件", A.analyze_events(evs, stats))
    )

    # ---------------------------------------------------------- 宏观
    macro = events.get("macro") or []
    parts.append(
        section(
            "五、宏观数据",
            table(
                ["指标", "数值", "单位", "环比", "期次"],
                [
                    [
                        esc(m.get("label") or m.get("report")),
                        f(m.get("value")),
                        esc(m.get("unit") or ""),
                        p(m.get("mom")),
                        esc(m.get("period") or ""),
                    ]
                    for m in macro
                ],
                numeric=[1, 3],
            ),
        )
        + analysis("宏观数据", A.analyze_macro(macro))
    )

    # ---------------------------------------------------------- 期货外汇
    cn = futures.get("cn_futures") or []
    gl = futures.get("global_futures") or []
    fx = futures.get("fx") or []
    mid = futures.get("fx_mid")
    parts.append(
        section(
            "六、重点期货与外汇",
            "<div class='grid2'>"
            + table(
                ["国内期货", "最新", "涨跌", "基准"],
                [
                    [
                        esc(r.get("name")),
                        f(r.get("price")),
                        cls_p(r.get("pct")),
                        esc(r.get("basis_label") or r.get("prev_close_label") or "昨结算"),
                    ]
                    for r in cn
                ],
                numeric=[1, 2],
            )
            + analysis("国内期货", A.analyze_cn_futures(cn))
            + table(
                ["外盘期货", "最新", "涨跌"],
                [[esc(r.get("name")), f(r.get("price")), cls_p(r.get("pct"))] for r in gl],
                numeric=[1, 2],
            )
            + analysis("外盘期货", A.analyze_global_futures(gl))
            + table(
                ["外汇", "最新", "涨跌", "核验"],
                [
                    [
                        esc(r.get("name")),
                        f(r.get("price"), 4),
                        cls_p(r.get("pct")),
                        tag((r.get("cross") or {}).get("status")),
                    ]
                    for r in fx
                ],
                numeric=[1, 2],
            )
            + analysis("外汇", A.analyze_fx(fx, mid))
            + (
                "<p class='note'>美元/人民币中间价（中国货币网）："
                + f((mid or {}).get("price"), 4)
                + "，日期 "
                + esc((mid or {}).get("date") or "—")
                + "</p>"
                if mid
                else ""
            )
            + "</div>",
        )
    )
    prods_rows = []
    for pr in prods:
        prods_rows.append(
            [
                esc(pr.get("name")),
                esc(pr.get("category")),
                f(pr.get("nav"), 4) if pr.get("nav") is not None else "—",
                cls_p(pr.get("pct")),
                f(pr.get("seven_day_yield"), 3) + "%" if pr.get("seven_day_yield") is not None else "—",
                p(pr.get("ret_1y")),
                tag((pr.get("cross") or {}).get("status")),
                esc(pr.get("rationale") or (pr.get("cross") or {}).get("note") or ""),
            ]
        )
    parts.append(
        section(
            "七、国内理财产品",
            table(
                ["产品", "类别", "净值", "日涨跌", "7日年化", "近1年", "核验", "说明"],
                prods_rows,
                numeric=[2, 3, 4, 5],
            ),
            note="货币基金无单位净值口径；银行理财采用中国理财网登记信息，无统一公开净值接口，不臆造净值。",
        )
        + analysis("国内理财产品", A.analyze_products(prods))
    )

    # ---------------------------------------------------------- 自选股池
    stocks = quotes.get("stocks") or []
    by_theme: dict[str, list[dict]] = {}
    for s in stocks:
        by_theme.setdefault(s.get("theme") or "其他", []).append(s)
    pool_html = ""
    for theme in ("大盘蓝筹", "科技成长", "红利低波", "港股互联网"):
        rows = by_theme.get(theme) or []
        if not rows:
            continue
        pool_html += (
            f"<h2 style='font-size:14px;border-left-color:#475569'>{esc(theme)}</h2>"
            + table(
                ["标的", "主题", "板块", "现价", "涨跌", "PE", "PB", "交易日", "核验"],
                _stock_rows(rows),
                numeric=[3, 4, 5, 6],
            )
            + analysis(theme, A.analyze_theme_pool(theme, rows))
        )
    parts.append(
        section("八、股票池（四类）", pool_html)
        + analysis("股票池整体", A.analyze_stock_pool(stocks))
    )

    # ---------------------------------------------------------- 复盘
    if review:
        rst = review.get("stats") or {}
        rev_rows = [
            [
                esc(e.get("day")),
                esc(e.get("name")),
                esc(e.get("call")),
                f(e.get("entry")),
                f(e.get("exit")),
                cls_p(e.get("ret_pct")),
                ("命中" if e.get("hit") else "未命中") if e.get("hit") is not None else "不可判定",
                "触发" if e.get("stop_hit") else "未触发",
                "达到" if e.get("tp_hit") else "未达到",
            ]
            for e in (review.get("evaluations") or [])[-30:]
        ]
        stat_rows = [
            ["可判定建议", f(rst.get("judged"), 0) if rst.get("judged") is not None else "—"],
            [
                "方向命中",
                (
                    f"{rst.get('hits')}（{f(rst.get('hit_rate'), 4)}）"
                    if rst.get("hit_rate") is not None
                    else "—"
                ),
            ],
            ["止损触发率", f(rst.get("stop_trigger_rate"), 4) if rst.get("stop_trigger_rate") is not None else "—"],
            ["目标达成率", f(rst.get("target_hit_rate"), 4) if rst.get("target_hit_rate") is not None else "—"],
            ["建议稳定性(Jaccard)", f(rst.get("stability_avg_jaccard"), 4) if rst.get("stability_avg_jaccard") is not None else "—"],
            ["平均实际收益", f(rst.get("avg_return_pct"), 2) + "%" if rst.get("avg_return_pct") is not None else "—"],
        ]
        bias_rows = [
            [k, f(v.get("n"), 0), f(v.get("avg_ret_pct"), 2) + "%", f(v.get("implied_target_pct"), 1) + "%", f(v.get("gap_pct"), 2) + "%"]
            for k, v in (rst.get("bias") or {}).items()
        ]

        # 首期没有可复盘的历史建议时，不摆一张全「—」的死表：
        # 改成说明复盘机制 + 展示已固化的今日基线，让读者知道明天开始就会逐条判定。
        first_run = not rev_rows
        if first_run:
            body = (
                "<div class='note'>"
                + esc(review.get("note") or "首期运行，暂无历史建议可复盘")
                + "</div>"
                + "<h2 style='font-size:14px;border-left-color:#475569'>复盘机制说明</h2>"
                + "<ul class='note'>"
                "<li>每一期的组合、参考价、止损位、目标位都写进<b>不可变快照</b>，当天之后不再改写。</li>"
                "<li>次日及以后每个交易日，用真实收盘价回测上一期的建议：方向对不对、止损有没有被打掉、目标有没有到。</li>"
                "<li>「方向命中率」「止损触发率」「目标达成率」「建议稳定性(Jaccard)」四个指标逐日累积。</li>"
                "<li>若实际走势系统性弱于建议目标，差值会出现在下方「偏差分析」里，并据此下调目标位或收紧止损 —— "
                "只改未来的参数，不回头篡改已经写下的结论。</li>"
                "</ul>"
            )
        else:
            body = (
                "<p class='note'>"
                + esc(review.get("note") or "")
                + "</p>"
                + table(["统计项", "值"], stat_rows)
                + (
                    "<h2 style='font-size:14px;border-left-color:#475569'>偏差分析</h2>"
                    + table(["方向", "样本", "实际均值", "隐含目标", "差值"], bias_rows)
                    if bias_rows
                    else ""
                )
                + (
                    "<h2 style='font-size:14px;border-left-color:#475569'>逐条判定</h2>"
                    + table(
                        ["建议日", "标的", "方向", "参考价", "实际价", "实际涨跌", "方向判定", "止损", "目标"],
                        rev_rows,
                        numeric=[3, 4, 5],
                    )
                )
            )
        parts.append(
            section(
                "九、昨日判断与模拟建议复盘",
                P.review_box(review) + body,
                note="复盘只针对本系统此前给出的研究性判断与模拟建议，不追溯改写历史结论。",
            )
            + analysis("复盘", A.analyze_review(review))
        )

    # ---------------------------------------------------------- 研究能力实测
    if evidence:
        parts.append(
            section(
                "十、研究能力实测（样本外 / 成本后 / 概率校准）",
                _evidence_section(evidence),
                note="本板块的数字全部来自 data/research 的样本外实验，"
                     "可复现；不通过门禁的信号不进入模拟仓位。",
            )
        )

    # ---------------------------------------------------------- 冲突与核验
    conflict_rows = [
        [esc(c.get("name")), esc(c.get("note") or c.get("diff_pct")), esc(", ".join(c.get("src_names") or []))]
        for c in conflicts
    ]
    src_rows = []
    for group, label in ((qsrcs, "行情"), (esrcs, "事件"), (psrcs, "理财产品")):
        for name, meta in (group or {}).items():
            if not isinstance(meta, dict):
                continue
            src_rows.append(
                [
                    esc(label),
                    esc(name),
                    "正常" if meta.get("ok") else "失败",
                    esc(meta.get("error") or meta.get("source") or ""),
                ]
            )
    parts.append(
        section(
            "十一、数据核验与口径冲突",
            table(["领域", "来源", "状态", "说明"], src_rows)
            + analysis("数据源可用性", A.analyze_sources(src_rows))
            + ("<h2 style='font-size:14px;border-left-color:#475569'>口径冲突</h2>"
               + table(["对象", "说明", "涉及来源"], conflict_rows)
               + analysis("口径冲突", A.analyze_conflicts(conflicts))
               if conflict_rows
               else "<p class='note'>本期无口径冲突。</p>"),
        )
    )

    body = "".join(parts)
    banner = (advice or {}).get("disclaimer") or (
        "本报告数据来自公开行情接口，双源交叉核验；所有交易内容均为模拟研究，不构成投资建议，不执行真实交易。"
    )

    # ---------------------------------------------------------- 每日研报
    research_html = ""
    if research:
        def _rk(label: str, key: str) -> str:
            v = research.get(key)
            if not v:
                return ""
            if isinstance(v, list):
                items = "".join(f"<li>{esc(x)}</li>" for x in v)
                return f"<div class='rk'>{esc(label)}</div><ul>{items}</ul>"
            return f"<div class='rk'>{esc(label)}</div><p>{esc(v)}</p>"

        research_html = (
            "<h2>每日研报 · 小金观点</h2>"
            "<div class='research'>"
            f"<div class='rt'>{esc(research.get('stance') or '')}｜"
            f"温度 {f(research.get('temperature'), 1)}"
            f"（{esc(research.get('temperature_label') or '')}）</div>"
            f"<div class='rs'>研判基于当日全部采集数据，经双源核验后生成</div>"
            + _rk("市场判断", "verdict")
            + _rk("组合逻辑", "logic")
            + _rk("驱动因素", "drivers")
            + _rk("风险提示", "risks")
            + _rk("明日关注", "watch")
            + _rk("前次复盘", "review_line")
            + f"<div class='rd'>{esc(research.get('disclaimer') or '')}</div>"
            "</div>"
        )

    # ---------------------------------------------------------- 交易日说明
    session_block = ""
    if session:
        mk = session.get("markets") or {}
        rows = [
            [
                esc(region),
                esc(m.get("trade_date", "—")),
                "开市" if m.get("is_open") else "休市",
                esc(m.get("reason", "")),
            ]
            for region, m in mk.items()
        ]
        notes = "".join(f"<li>{esc(n)}</li>" for n in (session.get("notes") or []))
        session_block = (
            "<h2 style='font-size:14px;border-left-color:#475569'>交易日状态</h2>"
            + table(["市场", "数据对应交易日", "状态", "说明"], rows)
            + (f"<ul class='note'>{notes}</ul>" if notes else "")
        )

    # ---------------------------------------------------------- 每日研究汇报（大白话）
    # 2026-11-14 起：七段式研究汇报不再单独发一条飞书消息，
    # 而是嵌进同一份 HTML —— 「每天 21:30 只发一份 HTML」是既定铁律。
    brief_block = ""
    if brief_text:
        # 白话正文按段落切；标题行（不含句末标点且较短）单独成行加粗。
        paras: list[str] = []
        for raw in brief_text.splitlines():
            line = raw.rstrip()
            if not line:
                continue
            paras.append(f"<p>{esc(line)}</p>")
        brief_block = (
            "<h2 style='font-size:15px;border-left-color:#7dd3fc'>每日研究汇报（大白话 · 七段式）</h2>"
            "<div class='plainbox'>"
            + "".join(paras)
            + "</div>"
        )
    elif brief_error:
        brief_block = (
            "<h2 style='font-size:15px;border-left-color:#7dd3fc'>每日研究汇报（大白话 · 七段式）</h2>"
            "<div class='plainbox'>"
            f"<p>{esc(brief_error)}</p>"
            "</div>"
        )

    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)} {esc(now)}</title><style>{CSS}</style></head><body><div class='wrap'>"
        f"<h1>{esc(title)}</h1>"
        f"<div class='sub'>生成时间 {esc(now)} · 数据采集 → 双源核验 → 模拟组合建议 → 落盘快照 → 复盘</div>"
        f"<div class='banner'>{esc(banner)}</div>"
        f"<div class='cards'>{cards}</div>"
        f"{guide_html}"
        f"{session_block}"
        f"{research_html}"
        f"{brief_block}"
        f"{body}"
        f"{P.glossary_html()}"
        "<footer>本文件由小金自动化流水线生成，为研究性记录。"
        "所有价格与涨跌幅来自公开行情接口并经交叉核验；"
        "模拟组合、仓位与止损位仅用于策略研究演示，不构成投资建议，未执行任何真实交易。"
        "历史结论以每日不可变快照固化，不做事后改写。</footer>"
        "</div></body></html>"
    )
