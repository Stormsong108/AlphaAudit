"""研报版日报的回归测试。

锁住用户明确提出的四点：
1. 每个表格的表头列数 == 每行单元格数（数据与表头必须对齐，不能只靠 CSS）
2. 每个表格后面都有基于该表真实数据的独立分析
3. 首期无历史时，输出明确空态说明，不能出现「可判定建议 0.00」这种死表
4. 每日研报（research）同时进入 HTML 和不可变快照
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import analysis as A  # noqa: E402
from advice import build_advice  # noqa: E402
from render import render_html  # noqa: E402

VERIFIED = {"status": "verified", "sources": ["sina", "eastmoney"]}
UNVERIFIED = {"status": "single", "sources": ["sina"]}


class _TableCollector(HTMLParser):
    """收集每个 <table> 的表头数、每行单元格数，以及表格后面紧邻的块类型。"""

    def __init__(self):
        super().__init__()
        self.tables: list[dict] = []
        self._cur: dict | None = None
        self._in_head = False
        self._row_cells = 0
        self._in_td = False
        self.blocks: list[str] = []          # 顶层块序列，元素名
        self._stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "table":
            self._cur = {"ncol": 0, "rows": [], "head_classes": [], "cell_classes": []}
            self.tables.append(self._cur)
            self._in_head = False
        elif tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self._row_cells = 0
            self._row_head_classes = []
        elif tag in ("td", "th"):
            self._row_cells += 1
            if self._cur is not None:
                self._row_head_classes.append(cls)
        elif tag == "div" and "analysis" in cls.split():
            self.blocks.append("analysis")
        elif tag == "div" and cls in ("note", "banner"):
            self.blocks.append(cls)

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_head = False
        elif tag == "tr":
            if self._cur is not None:
                if self._in_head:
                    self._cur["ncol"] = self._row_cells
                    self._cur["head_classes"] = self._row_head_classes
                else:
                    self._cur["rows"].append(self._row_cells)
                    self._cur["cell_classes"].append(self._row_head_classes)
        elif tag == "table":
            self._cur = None


def _stock(code, name, theme, price, pct, cross=None, sector="银行"):
    return {
        "code": code, "name": name, "theme": theme, "sector": sector,
        "region": "A股", "price": price, "pct": pct, "pe": 12.5, "pb": 1.4,
        "amount": 3.2e9, "trade_date": "2026-11-09",
        "cross": cross if cross is not None else VERIFIED,
    }


def _quotes(stocks):
    return {
        "indexes": [{
            "name": "上证指数", "region": "A股", "market": "CN", "price": 3200.0,
            "pct": 0.85, "amount": 4.5e11, "trade_date": "2026-11-09", "cross": VERIFIED,
        }],
        "stocks": stocks, "conflicts": [], "sources": {"sina": 2, "eastmoney": 2},
    }


def _events():
    return {
        "events": [{
            "title": "央行开展逆回购操作 净投放 2000 亿元", "importance": "high",
            "cross": VERIFIED, "time": "2026-11-09 09:00", "source": "华尔街见闻",
            "uri": "https://example.com/a",
        }, {
            "title": "某公司三季报净利同比增长 20%", "importance": "normal",
            "cross": UNVERIFIED, "time": "2026-11-09 08:00", "source": "东财",
            "uri": "https://example.com/b",
        }],
        "macro": [
            {"label": "CPI", "value": 0.3, "mom": 0.1, "period": "2026-10", "unit": "%"},
            {"label": "PMI", "value": 50.6, "mom": 0.2, "period": "2026-10", "unit": ""},
        ],
        "conflicts": [], "sources": {"wscn": 1, "eastmoney": 2},
        "stats": {"events_verified": 1, "events_total": 2},
    }


def _futures():
    return {
        "cn_futures": [
            {"name": "螺纹钢", "pct": 1.2, "price": 3300.0, "cross": VERIFIED},
            {"name": "沪铜", "pct": -0.8, "price": 78000.0, "cross": VERIFIED},
        ],
        "global_futures": [
            {"name": "WTI原油", "pct": 0.5, "price": 72.0, "cross": VERIFIED},
            {"name": "COMEX黄金", "pct": -0.3, "price": 2650.0, "cross": VERIFIED},
        ],
        "fx": [{"name": "美元指数", "pct": -0.2, "price": 104.2, "cross": VERIFIED}],
        "fx_mid": {"price": 7.1234, "date": "2026-11-09"},
        "conflicts": [], "sources": {"sina": 3, "eastmoney": 3},
    }


def _products():
    return {
        "products": [
            {"name": "货币基金A", "category": "货币基金", "seven_day_yield": 1.85,
             "ret_1y": 2.1, "cross": VERIFIED, "note": "—"},
            {"name": "纯债基金B", "category": "纯债基金", "ret_1y": 3.4,
             "cross": VERIFIED, "note": "—"},
        ],
        "sources": {"fund": 2}, "conflicts": [],
    }


def _full_html():
    stocks = [
        _stock("S1", "工商银行", "大盘蓝筹", 6.2, 0.8),
        _stock("S2", "宁德时代", "科技成长", 210.0, 2.1),
        _stock("S3", "长江电力", "红利低波", 28.5, -0.3),
        _stock("S4", "腾讯控股", "港股互联网", 380.0, 1.4, sector="互联网"),
    ]
    quotes = _quotes(stocks)
    advice = build_advice(quotes, _products())
    review = {
        "available": False,
        "note": "首期运行，暂无历史建议可复盘",
        "stats": {}, "evaluations": [],
    }
    research = A.build_research_report(
        quotes, _events(), _futures(), advice, review,
        {"is_trading_day": True, "data_date": "2026-11-09"}, _products(),
    )
    html = render_html(
        quotes, _events(), _futures(),
        products=_products(), advice=advice, review=review, research=research,
        session={"is_trading_day": True, "data_date": "2026-11-09"},
    )
    return research, html


# ------------------------------------------------------- 1. 表头与数据列对齐
def test_every_table_header_matches_row_width():
    """每个 <table> 的 th 数必须等于每行 td 数 —— 排版错位在生成阶段就要炸掉。"""
    _research, html = _full_html()
    p = _TableCollector()
    p.feed(html)
    assert p.tables, "HTML 里应该有表格"
    for t in p.tables:
        assert t["rows"], "表格有表头但没有数据行"
        for i, ncell in enumerate(t["rows"]):
            assert ncell == t["ncol"], (
                f"列数错位：表头 {t['ncol']} 列，第 {i + 1} 行 {ncell} 个单元格"
            )


def test_numeric_columns_have_matching_th_and_td_class():
    """数值列的表头和数据必须都用 num 类（右对齐），否则数字列会歪。"""
    _research, html = _full_html()
    p = _TableCollector()
    p.feed(html)
    for t in p.tables:
        if not t["rows"]:
            continue
        heads = t["head_classes"]
        first = t["cell_classes"][0]
        for i, (h, c) in enumerate(zip(heads, first)):
            assert ("num" in h) == ("num" in c), f"第 {i} 列表头与数据的对齐类不一致"


def test_table_helper_raises_on_column_mismatch():
    """table() 必须对列数不一致直接抛错，而不是静默补齐。"""
    from render.report import table

    with pytest.raises(ValueError):
        table(["A", "B"], [["1", "2", "3"]])
    with pytest.raises(ValueError):
        table(["A", "B"], [["1"]], numeric=[5])


# ------------------------------------------------------- 2. 每表后有分析
def test_every_table_is_followed_by_analysis():
    """每个物理表格后面必须紧跟一个分析块 —— 逐表分析，一个都不许漏。"""
    _research, html = _full_html()
    # 找出每个 </table> 之后、下一个 <table> 之前，是否出现 class="analysis"
    for i, m in enumerate(re.finditer(r"</table>(.*?)(?=<table|</body)", html, re.S)):
        seg = m.group(1)
        assert "class='analysis'" in seg or 'class="analysis"' in seg, (
            f"第 {i + 1} 个表格后面没有紧跟分析块"
        )


def test_analysis_block_has_points():
    """分析块必须有非空要点。"""
    _research, html = _full_html()
    blocks = re.findall(r'<div class=.analysis.>(.*?)</ul>', html, re.S)
    assert len(blocks) >= 10, f"分析块太少：{len(blocks)}"
    for b in blocks:
        assert "<li>" in b and "<li></li>" not in b
        assert "（本期无数据）" not in b


def test_analysis_text_uses_only_real_data():
    """分析里的数字必须来自当日数据，不能凭空出现。"""
    idx = _quotes([])["indexes"]
    lines = A.analyze_indexes(idx)
    assert lines
    assert "+0.85%" in " ".join(lines)


def test_analysis_states_no_data_instead_of_guessing():
    """无数据时明确说明不下结论，不返回空也不编造。"""
    for lines in (
        A.analyze_indexes([]),
        A.analyze_products([]),
        A.analyze_cn_futures([]),
        A.analyze_global_futures([]),
        A.analyze_fx([]),
        A.analyze_theme_pool("大盘蓝筹", []),
    ):
        assert lines, "无数据时应给出明确说明，而不是空列表"
        assert any("无法给出判断" in x or "为空" in x for x in lines)


# ------------------------------------------------------- 3. 首日空态
def test_first_day_review_shows_explicit_empty_state():
    """首期无历史：必须显示明确空态说明，不能出现「可判定建议 0.00」死表。"""
    _research, html = _full_html()
    assert "复盘" in html
    assert "可判定建议 0.00" not in html
    assert "首期" in html or "暂无历史" in html


def test_first_day_review_has_no_placeholder_rows():
    """首期复盘区不应出现成排的「—」占位行。"""
    _research, html = _full_html()
    seg = html.split("九、昨日判断与模拟建议复盘")
    if len(seg) > 1:
        body = seg[1].split("<h2", 1)[0]
        assert body.count("—") <= 2, "首期复盘区不应堆砌占位符"


# ------------------------------------------------------- 5. 研究能力实测板块


def _evidence():
    """一份结构完整的样本外实验结果，用于渲染研究板块。"""
    return {
        "available": True,
        "generated_at": "2026-11-10 23:50:00",
        "horizon": 5,
        "base_cost": 0.003,
        "panel": {"n_codes": 1298, "n_dates": 2863, "from": "2015-01-05", "to": "2026-11-10"},
        "segments": {"train": ["2015-01-05", "2020-12-10"], "test": ["2023-11-24", "2026-11-10"]},
        "factors": [
            {"name": "mom_20", "desc": "20日动量", "discovered_mean": -0.0346,
             "discovered_t": -5.1, "oos_mean": -0.0352, "oos_t": -2.47,
             "oos_hit": 0.39, "same_sign": True, "verdict": "时间外通过"},
            {"name": "rsi_14", "desc": "RSI", "discovered_mean": 0.0307,
             "discovered_t": 4.9, "oos_mean": 0.0328, "oos_t": 2.41,
             "oos_hit": 0.61, "same_sign": True, "verdict": "时间外通过"},
        ],
        "signal_diagnostic": {
            "ic_mean": 0.0401, "ic_t": 3.02, "ic_n": 711,
            "tail_spread_mean": -0.000765, "tail_positive_ratio": 0.479,
            "n_periods": 142, "ic_tail_consistent": False,
        },
        "strategy": {
            "ok": True, "n_periods": 142, "hold": 5, "mean_turnover": 0.479,
            "long_only": {"net_annualized": 0.1338, "sharpe": 0.35,
                          "net_total_return": 0.61, "win_rate": 0.53},
            "long_short": {"net_annualized": -0.1833, "sharpe": -0.58,
                           "net_total_return": -0.71, "win_rate": 0.44},
            "benchmark": {"annualized": 0.2182, "total_return": 0.88, "win_rate": 0.61},
            "breakeven_cost_long_only": -0.0005,
        },
        "strategy_config": {"horizon": 5, "quantile": 0.33, "construction": "decile",
                            "long_only_sharpe": 0.62},
        "construction_selection": {"n_configs": 18, "n_viable": 5,
                                   "criterion": "训练段多头成本后 Sharpe 最大"},
        "strategy_fixed_decile": {
            "ok": True,
            "long_only": {"net_annualized": 0.0938},
            "long_short": {"net_annualized": -0.2433},
        },
        "cost_sensitivity": [
            {"cost_rate": 0.0015, "ok": True, "long_only_annualized": 0.17,
             "long_only_sharpe": 0.44, "long_short_annualized": -0.1109,
             "long_short_sharpe": -0.3, "long_only_vs_bench_annualized": -0.0482},
            {"cost_rate": 0.003, "ok": True, "long_only_annualized": 0.1338,
             "long_only_sharpe": 0.35, "long_short_annualized": -0.1833,
             "long_short_sharpe": -0.58, "long_only_vs_bench_annualized": -0.0844},
        ],
        "probability": {
            "rel": {"isotonic": {"n": 209775, "brier": 0.24973, "brier_baseline": 0.25,
                                 "auc": 0.5175, "hit_rate": 0.5122,
                                 "baseline_hit_rate": 0.5, "avg_prob": 0.503,
                                 "actual_rate": 0.5}},
            "abs": {},
        },
        "prediction_loop": {"available": False, "n": 0, "sufficient": False,
                            "note": "尚未产生任何预测记录，预测闭环暂未启动，暂不评价命中率。"},
        "gates": [
            {"name": "G1 时间外验证", "pass": False, "detail": "6/11 = 54.5%，门槛 60%"},
            {"name": "G2 成本后收益", "pass": False, "detail": "Sharpe 0.35 < 0.50"},
        ],
        "verdict": {"pass": False, "passed": 0, "total": 4,
                    "text": "四道门禁通过 0/4；未通过：G1 时间外验证、G2 成本后收益。研究信号暂不进入模拟仓位。"},
        "limitations": ["universe 为当前在市股票快照，存在幸存者偏差。"],
    }


def test_evidence_section_renders_with_aligned_tables():
    """研究板块的每个表头列数必须等于数据行单元格数。"""
    import analysis as A
    from advice import build_advice
    from render import render_html

    quotes = _quotes([_stock("S1", "工商银行", "大盘蓝筹", 6.2, 0.8)])
    advice = build_advice(quotes, _products())
    html = render_html(
        quotes, _events(), _futures(),
        products=_products(), advice=advice,
        review={"available": False, "note": "首期运行", "stats": {}, "evaluations": []},
        research=A.build_research_report(
            quotes, _events(), _futures(), advice,
            {"available": False, "note": "首期", "stats": {}, "evaluations": []},
            {"is_trading_day": True, "data_date": "2026-11-09"}, _products(),
        ),
        evidence=_evidence(),
        session={"is_trading_day": True, "data_date": "2026-11-09"},
    )
    assert "研究能力实测" in html
    p = _TableCollector()
    p.feed(html)
    assert p.tables
    for t in p.tables:
        assert t["rows"], "表格有表头但没有数据行"
        for i, ncell in enumerate(t["rows"]):
            assert ncell == t["ncol"], f"列数错位：表头 {t['ncol']} 列，第 {i+1} 行 {ncell} 个单元格"


def test_evidence_section_has_analysis_after_every_table():
    """研究板块里每个表后面都要有分析块。"""
    import analysis as A
    from advice import build_advice
    from render import render_html

    quotes = _quotes([_stock("S1", "工商银行", "大盘蓝筹", 6.2, 0.8)])
    advice = build_advice(quotes, _products())
    html = render_html(
        quotes, _events(), _futures(),
        products=_products(), advice=advice,
        review={"available": False, "note": "首期运行", "stats": {}, "evaluations": []},
        research=A.build_research_report(
            quotes, _events(), _futures(), advice,
            {"available": False, "note": "首期", "stats": {}, "evaluations": []},
            {"is_trading_day": True, "data_date": "2026-11-09"}, _products(),
        ),
        evidence=_evidence(),
        session={"is_trading_day": True, "data_date": "2026-11-09"},
    )
    seg = html.split("十、研究能力实测")
    assert len(seg) > 1, "缺少研究能力实测板块"
    body = seg[1]
    for i, m in enumerate(re.finditer(r"</table>(.*?)(?=<table|</body)", body, re.S)):
        s = m.group(1)
        assert "class='analysis'" in s or 'class="analysis"' in s, (
            f"研究板块第 {i+1} 个表后面没有分析块"
        )


def test_evidence_gates_fail_is_shown_honestly():
    """门禁没过必须在 HTML 里写出来，不能只报好看的部分。"""
    import analysis as A
    from advice import build_advice
    from render import render_html

    quotes = _quotes([_stock("S1", "工商银行", "大盘蓝筹", 6.2, 0.8)])
    advice = build_advice(quotes, _products())
    html = render_html(
        quotes, _events(), _futures(),
        products=_products(), advice=advice,
        review={"available": False, "note": "首期运行", "stats": {}, "evaluations": []},
        research=A.build_research_report(
            quotes, _events(), _futures(), advice,
            {"available": False, "note": "首期", "stats": {}, "evaluations": []},
            {"is_trading_day": True, "data_date": "2026-11-09"}, _products(),
        ),
        evidence=_evidence(),
        session={"is_trading_day": True, "data_date": "2026-11-09"},
    )
    assert "未通过" in html
    assert "暂不进入模拟仓位" in html


# ------------------------------------------------------- 6. 每日研报
def test_research_report_structure():
    """研报必须有判断、逻辑、驱动、风险、关注、复盘结论。"""
    research, _html = _full_html()
    for key in ("stance", "verdict", "logic", "drivers", "risks", "watch", "review_line"):
        assert key in research, f"研报缺字段 {key}"
    assert research["stance"]
    assert research["risks"]
    assert "不构成投资建议" in research.get("disclaimer", "")


def test_research_appears_in_html():
    _research, html = _full_html()
    assert "每日研报" in html
    assert "小金观点" in html
    assert "research" in html


def test_research_is_deterministic():
    """同样的输入必须得到同样的研报文字 —— 便于审计与复盘。"""
    a, _ = _full_html()
    b, _ = _full_html()
    assert a == b


def test_research_persisted_into_snapshot(tmp_path, monkeypatch):
    """研报属于当日判断，必须与组合一起固化进不可变快照。"""
    import json

    import main as main_mod

    monkeypatch.setattr(main_mod, "SNAPSHOT_DIR", tmp_path)
    bundle = {"quotes": {"stocks": []}, "events": {}, "futures": {}, "products": {}}
    research = {"stance": "震荡", "verdict": "测试"}
    p = main_mod.save_immutable_snapshot(
        bundle, {"portfolio": []}, {"available": False}, "2026-11-01", None, research
    )
    saved = json.loads(p.read_text(encoding="utf-8"))
    assert saved["research"]["stance"] == "震荡"
    assert saved["immutable"] is True


# ------------------------------------------------------- 7. G4 四件套


def _evidence_with_loop():
    """带完整闭环数据的证据：live/replay/选股/按天/判据都有。"""
    ev = _evidence()
    ev["prediction_loop"] = {
        "available": True, "n_days": 197, "n": 196238, "sufficient": True,
        "hit_rate": 0.5160, "baseline_hit_rate": 0.5106, "edge_vs_baseline": 0.0054,
        "brier": 0.25058, "avg_prob": 0.4842, "actual_rate": 0.4894,
        "live": {"n_days": 1, "n": 0, "sufficient": False,
                 "hit_rate": None, "brier": None, "baseline_hit_rate": None,
                 "edge_vs_baseline": None},
        "replay": {"n_days": 196, "n": 196238, "hit_rate": 0.5160,
                   "brier": 0.25058, "baseline_hit_rate": 0.5106,
                   "edge_vs_baseline": 0.0054, "avg_fwd_return": 0.0066},
        "market": {"n": 194, "hit_rate": 0.5412, "avg_fwd_return": 0.0070},
        "portfolio": {
            "available": True, "n_days": 194,
            "avg_top_decile_return": 0.00701, "avg_bottom_decile_return": 0.0059,
            "avg_spread": 0.00110, "median_spread": -0.000897,
            "avg_equal_weight_return": 0.00707, "avg_top_minus_bw": -0.000057,
            "pct_days_spread_positive": 0.4897,
        },
        "day_level": {
            "available": True, "n_days": 194,
            "avg_daily_hit_rate": 0.5158, "median_daily_hit_rate": 0.5176,
            "pct_days_hit_above_50": 0.5464, "pct_days_actual_up_above_50": 0.4742,
            "avg_daily_actual_rate": 0.4894, "daily": [],
        },
    }
    ev["gates"] = ev["gates"] + [{
        "name": "G4 预测闭环", "pass": False,
        "detail": "价差中位数 -0.09%",
        "criteria_detail": [
            {"key": "c1_samples", "name": "① 链跑通（样本量）",
             "threshold": "已判定样本 ≥ 30 条", "actual": "196238 条", "pass": True},
            {"key": "c2_selection", "name": "② 选股能力（十分位价差中位数）",
             "threshold": "> 0", "actual": "-0.0897%", "pass": False},
            {"key": "c3a_days", "name": "③ 概率增量（按天跑赢硬币的天数占比）",
             "threshold": "> 50%", "actual": "54.64%", "pass": True},
            {"key": "c3b_brier", "name": "④ 概率增量（Brier 优于全猜基准率）",
             "threshold": "< 0.24989", "actual": "0.25058", "pass": False},
        ],
    }]
    ev["verdict"] = {
        "pass": False, "passed": 2, "total": 4,
        "text": "四道门禁通过 2/4；未通过：G2 成本后收益、G4 预测闭环。",
        "sim_position_allowed": False,
        "failed_gates": ["G2 成本后收益", "G4 预测闭环"],
    }
    return ev


def _render_with(evidence):
    from advice import build_advice
    from render import render_html

    quotes = _quotes([_stock("S1", "工商银行", "大盘蓝筹", 6.2, 0.8)])
    advice = build_advice(quotes, _products())
    return render_html(
        quotes, _events(), _futures(),
        products=_products(), advice=advice,
        review={"available": False, "note": "首期运行", "stats": {}, "evaluations": []},
        research=A.build_research_report(
            quotes, _events(), _futures(), advice,
            {"available": False, "note": "首期", "stats": {}, "evaluations": []},
            {"is_trading_day": True, "data_date": "2026-11-09"}, _products(),
        ),
        evidence=evidence,
        session={"is_trading_day": True, "data_date": "2026-11-09"},
    )


def test_g4_four_pieces_all_present():
    """G4 四件套（总览/口径拆分/选股价差/按天聚合/判据明细）必须都渲染出来。"""
    html = _render_with(_evidence_with_loop())
    for piece in ("① 闭环总览", "② 口径拆分", "③ 选股能力", "④ 按交易日聚合", "⑤ G4 判据明细"):
        assert piece in html, f"缺 G4 板块：{piece}"


def test_g4_shows_live_and_replay_separately():
    """实时预测与历史回溯必须分列 —— 混报会让人把回测当成实盘。"""
    html = _render_with(_evidence_with_loop())
    assert "实时预测 live" in html
    assert "历史回溯自检 replay" in html
    assert "市场方向（独立于选股）" in html


def test_g4_shows_selection_spread_and_day_level():
    """选股价差（门禁口径）与按天聚合必须都有数字。"""
    html = _render_with(_evidence_with_loop())
    assert "十分位价差 · 中位数口径（门禁口径）" in html
    assert "跑赢硬币的天数占比" in html
    assert "-0.0897%" in html


def test_g4_criteria_show_which_failed():
    """判据明细必须逐条写出门槛、实测、达标与否。"""
    html = _render_with(_evidence_with_loop())
    assert "未达标" in html
    assert "达标" in html
    assert "① 链跑通（样本量）" in html
    assert "② 选股能力（十分位价差中位数）" in html


def test_sim_position_gate_is_explicit_when_gates_fail():
    """门禁没过，日报必须显式写「不放行」，不能靠读者自己数。"""
    html = _render_with(_evidence_with_loop())
    assert "模拟仓位放行：否" in html
    assert "不进入模拟仓位" in html


def test_snapshot_keeps_full_evidence(tmp_path, monkeypatch):
    """不可变快照必须同时保留完整研究证据与采集口径来源。

    回归：以前 payload 里写了两个 "evidence" 键，后者把前者整个覆盖掉，
    快照里只剩 sources/conflicts，门禁与判定证据全丢。
    """
    import json

    import main as main_mod

    monkeypatch.setattr(main_mod, "SNAPSHOT_DIR", tmp_path)
    bundle = {
        "quotes": {"stocks": [], "sources": {"sina": 1}},
        "events": {"sources": {"wscn": 1}},
        "futures": {"sources": {"sina": 1}},
        "products": {"sources": {"fund": 1}},
    }
    evidence = {
        "gates": [{"name": "G4 预测闭环", "pass": False}],
        "verdict": {"pass": False, "passed": 2, "total": 4,
                    "sim_position_allowed": False},
        "prediction_loop": {"available": True, "n": 196238},
    }
    p = main_mod.save_immutable_snapshot(
        bundle, {"portfolio": []}, {"available": False}, "2026-11-02", None, None, evidence
    )
    saved = json.loads(p.read_text(encoding="utf-8"))
    ev = saved["evidence"]
    assert ev["verdict"]["passed"] == 2, "门禁判定没进快照"
    assert ev["gates"], "门禁明细没进快照"
    assert ev["prediction_loop"]["n"] == 196238, "闭环样本数没进快照"
    assert ev["sources"]["quotes"] == {"sina": 1}, "采集来源没进快照"
