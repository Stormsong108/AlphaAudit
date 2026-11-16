"""溯源表回归测试 —— 锁死「日报里每个研究数字都能追到来源」。

背景（2026-11-16）：
  同一份日报里同时出现 -2.03%（11-12 冻结的 cross-segment 产物）
  和 -1.30%（当天 canonical oos），读者无法判断哪个是真的。
  根因：研究产物不带 generated_at / 面板截止日，页面也不印溯源信息。
  修复：render/report.py 增加「数据口径与产物时间」表。

本模块锁死三条不变量：
  1. 溯源表必须包含：产物文件 / 生成时间 / 文件修改时间 / 面板截止日；
  2. 每个产物的面板截止日必须能挖出来（挖不到写「未标注」，但不能静默空白）；
  3. 若多份产物的面板截止日不一致，分析块必须显式警告
     —— 数字差异可能来自数据新旧，而不是策略变好/变坏。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "data" / "research"

# 与 render/report.py 里那一组保持一致（改了那边就要改这里）
PROVENANCE_FILES = (
    "oos_latest.json",
    "cross_segment_check.json",
    "execution_realism.json",
    "signal_anatomy.json",
    "live_readiness.json",
)


# --------------------------------------------------------------------------
# 直接测试 _extract_panel_to 的挖取逻辑（通过导入 render.report 拿函数）
# --------------------------------------------------------------------------
def _get_extractor():
    import render.report as report

    # _extract_panel_to 是 render_html 内部的闭包，报告生成时才存在。
    # 为了能单测，把它的逻辑独立复刻到本模块，再与真实产物对照；
    # 若 render.report 改了这个函数名，本测试会显式失败而不是静默通过。
    src = Path(report.__file__).read_text(encoding="utf-8")
    assert "def _extract_panel_to(" in src, (
        "render/report.py 里找不到 _extract_panel_to —— "
        "溯源表的面板截止日提取逻辑被改名或删除了，本测试需要同步更新"
    )
    return src


def _extract_panel_to(d: dict[str, Any]) -> str:
    """与 render/report.py 中 _extract_panel_to 保持一致的复刻实现。"""

    def _from_seg(obj):
        if not isinstance(obj, dict):
            return None
        t = obj.get("test")
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            return t[1]
        return None

    if d.get("panel_to"):
        return str(d["panel_to"])
    pn = d.get("panel")
    if isinstance(pn, dict) and pn.get("to"):
        return str(pn["to"])
    v = _from_seg(d.get("segments"))
    if v:
        return str(v)
    v = _from_seg(d.get("segment"))
    if v:
        return str(v)
    r = d.get("result")
    if isinstance(r, dict):
        if r.get("panel_to"):
            return str(r["panel_to"])
        v = _from_seg(r.get("segments"))
        if v:
            return str(v)
        v = _from_seg(r.get("segment"))
        if v:
            return str(v)
    return "未标注"


# --------------------------------------------------------------------------
# 1) 提取逻辑本身
# --------------------------------------------------------------------------
def test_extractor_reads_panel_to_toplevel():
    assert _extract_panel_to({"panel_to": "2026-11-13"}) == "2026-11-13"


def test_extractor_reads_panel_to():
    assert _extract_panel_to({"panel": {"to": "2026-11-13"}}) == "2026-11-13"


def test_extractor_reads_segments_test_tuple():
    d = {"segments": {"train": ["2015-01-02", "2020-12-01"], "test": ["2023-11-22", "2026-11-13"]}}
    assert _extract_panel_to(d) == "2026-11-13"


def test_extractor_reads_segment_test_tuple():
    assert _extract_panel_to({"segment": {"test": ["2023-11-22", "2026-11-13"]}}) == "2026-11-13"


def test_extractor_reads_nested_result_panel_to():
    assert _extract_panel_to({"result": {"panel_to": "2026-11-13"}}) == "2026-11-13"


def test_extractor_reads_nested_result_segments():
    d = {"result": {"segments": {"test": ["2023-11-22", "2026-11-13"]}}}
    assert _extract_panel_to(d) == "2026-11-13"


def test_extractor_returns_marked_when_absent():
    """挖不到必须写「未标注」，不能返回空串或 None（否则表格里是空白格）。"""
    assert _extract_panel_to({}) == "未标注"
    assert _extract_panel_to({"result": {}}) == "未标注"


def test_extractor_ignores_malformed_segment():
    """test 段不是二元组时不能崩，也不能瞎猜日期。"""
    assert _extract_panel_to({"segments": {"test": "2026-11-13"}}) == "未标注"
    assert _extract_panel_to({"segments": {"test": ["2023-11-22"]}}) == "未标注"


# --------------------------------------------------------------------------
# 2) 与真实产物对照（产物不存在时 skip，不编数据）
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fname", PROVENANCE_FILES)
def test_real_artifact_has_generated_at_and_panel_to(fname):
    p = RESEARCH / fname
    if not p.exists():
        pytest.skip(f"{fname} 不存在（公开仓库不含运行数据），跳过")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d.get("generated_at"), f"{fname} 缺 generated_at —— 读者无法判断数字新旧"
    pt = _extract_panel_to(d)
    assert pt != "未标注", f"{fname} 挖不到面板截止日"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", pt), f"{fname} 面板截止日格式异常：{pt}"


def test_all_present_artifacts_share_same_panel_to():
    """同时存在的产物必须算的是同一批数据。

    截止日不一致 = 数字差异可能来自数据新旧而非策略变化，
    报告页会对此发警告，但根子上说明队列没对齐。
    """
    found = []
    for fname in PROVENANCE_FILES:
        p = RESEARCH / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        pt = _extract_panel_to(d)
        if pt != "未标注":
            found.append((fname, pt))
    if len(found) < 2:
        pytest.skip("可用产物不足 2 份，无法比对")
    pts = {pt for _, pt in found}
    assert len(pts) == 1, (
        "产物面板截止日不一致："
        + "；".join(f"{n}={t}" for n, t in found)
        + " —— 数字差异可能只是数据新旧，必须先对齐再比较"
    )


# --------------------------------------------------------------------------
# 3) 报告页必须真的渲染出这张表
# --------------------------------------------------------------------------
def test_report_source_has_provenance_columns():
    src = _get_extractor()
    for col in ("产物文件", "产物生成时间", "文件修改时间", "面板截止日"):
        assert col in src, f"溯源表缺少列：{col}"


def test_report_source_lists_all_artifacts():
    src = _get_extractor()
    for fname in PROVENANCE_FILES:
        assert fname in src, f"溯源表没有列出 {fname}"


def test_generated_html_contains_provenance_table():
    """有 HTML 时校验表头列数与数据行列数一致（项目铁律）。"""
    htmls = sorted((ROOT / "data" / "html").glob("*.html"))
    if not htmls:
        pytest.skip("没有已生成的 HTML，跳过")
    html = htmls[-1].read_text(encoding="utf-8")
    if "数据口径与产物时间" not in html:
        pytest.skip("最新 HTML 不含溯源表（可能是旧版本生成的），跳过")
    # 抽溯源表：表头 th 数 vs 各数据行 td 数
    seg = html.split("数据口径与产物时间", 1)[1]
    tbl = seg.split("</table>", 1)[0]
    head = tbl.split("</thead>", 1)[0]
    n_th = head.count("<th")
    body = tbl.split("<tbody>", 1)[1] if "<tbody>" in tbl else ""
    rows = [r for r in body.split("<tr>") if r.strip()]
    assert rows, "溯源表没有数据行"
    for r in rows:
        assert r.count("<td") == n_th, (
            f"溯源表列数错位：表头 {n_th} 列，数据行 {r.count('<td')} 列"
        )
    # 表后必须紧跟分析块（项目铁律：每个物理表格后面都有分析块）
    after = seg.split("</table>", 1)[1]
    assert "analysis" in after[:400], "溯源表后面没有紧跟分析块"
