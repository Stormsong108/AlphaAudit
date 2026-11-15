"""实盘就绪度接入日报的端到端测试。

真实证据（build_evidence()）→ 渲染层（render_html）→ HTML 检查。
不 mock 证据，因为 mock 会掩盖「渲染层拿到的是旧 schema」这类问题
（2026-11-12 就发生过：render/report.py 顶层 import plain 导致整个
render 包无法导入，3 个测试文件直接 collection error）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import analysis as A  # noqa: E402
import research_evidence as RE  # noqa: E402
from advice import build_advice  # noqa: E402
from render import render_html  # noqa: E402


@pytest.fixture(scope="module")
def real_evidence():
    ev = RE.build_evidence()
    if not ev.get("live_readiness", {}).get("available"):
        pytest.skip(f"live_readiness 不可用：{ev.get('live_readiness', {}).get('reason')}")
    return ev


@pytest.fixture(scope="module")
def rendered(real_evidence):
    import test_report_quality as T

    quotes = T._quotes([T._stock("S1", "工商银行", "大盘蓝筹", 6.2, 0.8)])
    products = T._products()
    advice = build_advice(quotes, products)
    review = {"available": False, "note": "首期运行", "stats": {}, "evaluations": []}
    session = {"is_trading_day": True, "data_date": "2026-11-09"}

    html = render_html(
        quotes, T._events(), T._futures(),
        products=products, advice=advice,
        review=review,
        research=A.build_research_report(
            quotes, T._events(), T._futures(), advice, review, session, products,
        ),
        evidence=real_evidence,
        session=session,
    )
    return html


def test_live_readiness_section_rendered(rendered):
    """实盘就绪度表与判定表都必须出现在 HTML 里。"""
    assert "实盘就绪度（拿真钱上会怎样）" in rendered
    assert "实盘判定逐项" in rendered


def test_all_tables_have_analysis_blocks(rendered):
    """所有表格后必须有分析块（新增的两张表也算）。"""
    missing = []
    for i, m in enumerate(
        re.finditer(r"</table>(.*?)(?=<table|</body)", rendered, re.S)
    ):
        seg = m.group(1)
        if "class='analysis'" not in seg and 'class="analysis"' not in seg:
            missing.append(i + 1)
    assert not missing, f"第 {missing} 个表后面没有分析块"


def test_rendered_numbers_match_source(rendered, real_evidence):
    """渲染层不得改数 —— 关键数字必须与 live_readiness 源一致。"""
    import live_readiness as LR

    full = LR.full_check()
    expected = [
        f"{full['breakeven_cost']['breakeven_cost_roundtrip']:.4%}",
        f"{full['breakeven_cost']['actual_cost_roundtrip']:.2%}",
        f"{full['cost_decomposition']['turnover_per_period']:.2%}",
    ]
    for text in expected:
        assert text in rendered, f"{text} 未出现在 HTML（渲染层可能改了数）"


def test_all_check_ids_rendered(rendered, real_evidence):
    """每个判定项编号都要渲染出来。"""
    for c in real_evidence["live_readiness"]["checks"]:
        assert c["id"] in rendered, f"判定项 {c['id']} 未渲染"


def test_cross_segment_info_present(rendered, real_evidence):
    """跨段稳健性信息必须出现在 HTML 里。"""
    if not real_evidence["live_readiness"].get("cross_segment"):
        pytest.skip("无跨段证据")
    assert "结构性" in rendered or "train" in rendered


def test_live_readiness_failure_does_not_crash_report():
    """体检失败时日报必须仍能渲染，且明说不可用（不能假装通过）。"""
    import render.report as RR

    orig = RE.build_evidence
    try:
        # 让 live_readiness 不可用
        RE.build_evidence = lambda: {
            "available": True,
            "live_readiness": {"available": False, "reason": "模拟故障"},
            "gates": [],
            "factors": [],
        }
        import test_report_quality as T

        quotes = T._quotes([T._stock("S1", "工商银行", "大盘蓝筹", 6.2, 0.8)])
        products = T._products()
        advice = build_advice(quotes, products)
        review = {"available": False, "note": "首期运行", "stats": {}, "evaluations": []}
        session = {"is_trading_day": True, "data_date": "2026-11-09"}
        html = render_html(
            quotes, T._events(), T._futures(),
            products=products, advice=advice, review=review,
            research=A.build_research_report(
                quotes, T._events(), T._futures(), advice, review, session, products,
            ),
            evidence={
                "available": True,
                "live_readiness": {"available": False, "reason": "模拟故障"},
                "gates": [], "factors": [],
            },
            session=session,
        )
        assert "实盘就绪度" in html
        assert "不可用" in html
        # 不能出现"通过"的误导性表述
        assert "模拟故障" in html
    finally:
        RE.build_evidence = orig


def test_analyze_live_readiness_handles_unavailable():
    """体检不可用时，文案必须说清楚「既不是通过也不是没通过」。"""
    lines = RE.analyze_live_readiness({
        "live_readiness": {"available": False, "reason": "文件缺失"},
    })
    txt = " ".join(lines)
    assert "不可用" in txt
    assert "不是「通过」" in txt or "不是「没通过」" in txt
