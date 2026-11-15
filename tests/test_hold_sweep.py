"""hold_sweep / cross_segment 的回归测试。

这两个模块回答的是同一个实盘问题：
    「拉长调仓周期压换手，能不能把净超额翻正？」

背景与坑
--------
1. `ooslab.py --hold N` **不能**做这个实验。已定位：
   - `horizon = args.hold if args.hold > 0 else hs["chosen"]`（只影响因子发现）
   - 策略回测用的是训练段选出的 `chosen["horizon"]`（恒为 5）
   实测：--hold 10 与 --hold 20 的 strategy_test_segment 逐位相同。
   → 必须直接调底层回测函数在测试段上扫。

2. **换手率不随 hold 延长而下降**，反而上升：
   hold 5→60，换手 22.08% → 53.58%。
   原因：hold 越长每期持仓数越少，单只标的进出更频繁。

3. hold=10 在测试段超额 +0.49%（唯一正超额），
   但训练段 −1.81%、验证段 −1.00% —— **符号不一致，是样本内挑选的噪声**。

4. hold=5 三段全负（−0.82% / −3.12% / −2.03%），加权 −1.70%
   —— **符号一致，结构性的负 alpha**。

本测试文件锁住上述结论，防止后续改动悄悄把它们推翻。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cross_segment as CS  # noqa: E402
import hold_sweep as HS  # noqa: E402


@pytest.fixture(scope="module")
def sweep_result():
    p = ROOT / "data" / "research" / "hold_sweep_result.json"
    if not p.exists():
        pytest.skip("hold_sweep_result.json 不存在，先跑 hold_sweep.py")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cross_result():
    p = ROOT / "data" / "research" / "cross_segment_check.json"
    if not p.exists():
        pytest.skip("cross_segment_check.json 不存在，先跑 cross_segment.py")
    return json.loads(p.read_text(encoding="utf-8"))


# ── hold_sweep ──────────────────────────────────────────────────────────────


def test_sweep_rows_have_every_hold(sweep_result):
    """扫描结果必须覆盖所有请求的持有期。"""
    holds = {r["hold"] for r in sweep_result["result"]["rows"]}
    assert set(HS.HOLD_GRID).issubset(holds)


def test_sweep_records_actual_hold(sweep_result):
    """每行必须记录实际使用的 hold —— 参数没生效要能看出来。"""
    for r in sweep_result["result"]["rows"]:
        assert "hold" in r
        if r.get("ok"):
            assert r["hold"] in HS.HOLD_GRID


def test_turnover_rises_with_hold(sweep_result):
    """锁住关键反直觉事实：换手率随 hold 延长而**上升**，不是下降。

    这条如果哪天变成下降，说明回测逻辑变了，本模块的结论要全部重估。
    """
    rows = [r for r in sweep_result["result"]["rows"]
            if r.get("ok") and abs(r["cost_rate"] - 0.0030) < 1e-9]
    rows.sort(key=lambda r: r["hold"])
    turns = [r["mean_turnover"] for r in rows]
    assert turns == sorted(turns), (
        f"换手率不再随 hold 单调上升（历史实测 hold5→60: 22.08%→53.58%），"
        f"当前: {[(r['hold'], round(r['mean_turnover'], 4)) for r in rows]}"
    )


def test_sweep_excess_identity(sweep_result):
    """毛超额 - 成本拖累 = 净超额（每行独立校验）。"""
    for r in sweep_result["result"]["rows"]:
        if not r.get("ok"):
            continue
        g = r["gross_excess_per_period"]
        n = r["net_excess_per_period"]
        c = r["cost_drag_per_period"]
        if g is None or n is None or c is None:
            continue
        assert abs((g - c) - n) < 1e-12, f"hold={r['hold']} 成本={r['cost_rate']} 口径不自洽"


def test_sweep_breakeven_matches_turnover(sweep_result):
    """打平成本 × 换手 = 毛超额。"""
    for r in sweep_result["result"]["rows"]:
        if not r.get("ok"):
            continue
        be = r.get("breakeven_cost")
        if be is None:
            continue
        assert abs(be * r["mean_turnover"] - r["gross_excess_per_period"]) < 1e-12


# ── cross_segment ──────────────────────────────────────────────────────────


def test_cross_segment_covers_three_segments(cross_result):
    """必须覆盖训练/验证/测试三段。"""
    segs = cross_result["result"]["segments"]
    assert set(segs) >= {"train", "val", "test"}


def test_cross_segment_has_consistent_negative_for_hold5(cross_result):
    """锁住核心结论：hold=5 在三段全部为负。

    这是「结构性的负 alpha」与「某段行情运气」的分水岭。
    """
    rows = cross_result["result"]["rows"]
    row = next(r for r in rows
               if r["hold"] == 5 and abs(r["cost_rate"] - 0.0030) < 1e-9)
    ex = row["excess_by_segment"]
    assert set(ex) >= {"train", "val", "test"}
    for seg, v in ex.items():
        assert v < 0, f"{seg} 段超额变成 {v*100:+.2f}%（不再是负）—— 结论需重估"
    assert row["all_same_sign"] is True


def test_cross_segment_hold10_is_not_robust(cross_result):
    """锁住：hold=10 的正超额是样本内挑选的噪声。

    它在测试段为正，但训练/验证段为负 —— 符号不一致。
    """
    rows = cross_result["result"]["rows"]
    row = next(r for r in rows
               if r["hold"] == 10 and abs(r["cost_rate"] - 0.0030) < 1e-9)
    ex = row["excess_by_segment"]
    assert ex["test"] > 0, "测试段不再是正超额"
    assert ex["train"] < 0 and ex["val"] < 0
    assert row["all_same_sign"] is False


def test_cross_segment_analysis_finds_no_robust_hold(cross_result):
    """没有任何持有期在三个区段都跑赢基准。"""
    ana = cross_result["analysis"]
    assert ana["n_positive_all_segments"] == 0
    assert ana["robust_holds"] == []
    assert "噪声" in ana["verdict"]


def test_cross_segment_weighted_excess_negative(cross_result):
    """按期数加权的合并超额必须为负。"""
    rows = cross_result["result"]["rows"]
    row = next(r for r in rows
               if r["hold"] == 5 and abs(r["cost_rate"] - 0.0030) < 1e-9)
    num = den = 0.0
    for seg in row["per_segment"].values():
        n = seg["n_periods"]
        num += seg["excess_annualized"] * n
        den += n
    weighted = num / den
    assert weighted < 0, f"加权超额变成 {weighted*100:+.2f}%"
    # 历史实测 -1.70%，容差 0.5 个点
    assert abs(weighted - (-0.0170)) < 0.005


def test_analyze_detects_no_robust_hold(cross_result):
    """analyze() 的判定逻辑：没有跨段全正的持有期。

    注意：analyze() 接收的是 JSON 里的 "result" 部分，不是整个文件。
    """
    ana = CS.analyze(cross_result["result"])
    assert ana["n_positive_all_segments"] == 0
    assert ana["robust_holds"] == []
    assert "噪声" in ana["verdict"]


# ── 渲染层导入断裂（2026-11-12 修复的 bug）────────────────────────────────
# render/report.py 曾写 `import plain as P`，但 plain.py 在 render/ 包内，
# 顶层没有同名模块 → `import render` 直接 ModuleNotFoundError，
# 连带 3 个测试文件收集失败（131 passed 变成 collection error）。
# 这条测试锁住包内相对导入的正确性。


def test_render_package_imports_cleanly():
    """import render 必须成功（不能有 ModuleNotFoundError）。"""
    import importlib

    m = importlib.import_module("render")
    assert hasattr(m, "render_html")
    assert hasattr(m, "render_daily")


def test_render_report_uses_relative_import_for_plain():
    """render/report.py 必须用相对导入拿同级 plain 模块。"""
    src = (ROOT / "render" / "report.py").read_text(encoding="utf-8")
    assert "from . import plain as P" in src, (
        "render/report.py 必须用 `from . import plain as P`；"
        "顶层 plain 模块不存在，写成 `import plain as P` 会让整个 render 包无法导入"
    )
    assert not any(
        line.strip() in ("import plain as P", "import plain")
        for line in src.splitlines()
    ), "存在顶层 `import plain`，会与包内模块冲突"


def test_every_table_has_analysis_block():
    """每张表后面必须紧跟分析块（术语表也必须有）。"""
    import re

    import test_report_quality as T

    _research, html = T._full_html()
    missing = []
    for i, m in enumerate(
        re.finditer(r"</table>(.*?)(?=<table|</body)", html, re.S)
    ):
        seg = m.group(1)
        if "class='analysis'" not in seg and 'class="analysis"' not in seg:
            missing.append(i + 1)
    assert not missing, f"第 {missing} 个表后面没有紧跟分析块"
