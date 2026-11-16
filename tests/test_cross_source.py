"""跨来源对账门禁的回归测试。

背景（2026-11-16）：
  日报同时出现 -2.03%（11-12 冻结的 cross-segment 产物）和 -1.30%（当天 canonical OOS），
  方向还相反。当时靠人眼交叉比对发现 —— 但人眼只比对一次。

  本模块把「同一指标多来源必须一致」从人的纪律变成代码的门禁。

本文件锁死的不变量：
  1. 取值路径互相独立 —— 每个来源自己从产物文件里挖，
     不能共用中间变量（否则一致性检验退化成自证，永远通过）。
  2. 口径必须显式匹配 —— cross_segment 的 rows 必须按 cost_rate + hold 选行，
     禁止取 rows[0]（那是成本 0.15% 那行，会把 +0.3753% 当成 canonical）。
  3. 值不一致 → 标 INCONSISTENT，且日报必须**拒绝出结论**（不显示任何数字）。
  4. 找不到 canonical 行 → MISSING，不许兜底。
  5. 浮点噪声不算不一致（-0.012962048552470597 vs -0.0129620485524706）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cross_source as cs

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "data" / "research"


# ------------------------------------------------------------------
# 1. 容差判定
# ------------------------------------------------------------------
def test_close_identical():
    assert cs._close(0.5, 0.5)


def test_close_float_noise():
    """浮点噪声必须判为一致：两个来源各算一遍，末位差 1e-19 是正常的。"""
    a = -0.012962048552470597
    b = -0.0129620485524706
    assert a != b, "前提：这两个值在浮点层面确实不相等"
    assert cs._close(a, b)


def test_close_rejects_real_mismatch():
    """真实口径错位必须判为不一致。"""
    assert not cs._close(-0.0129620485524706, -0.020277)
    assert not cs._close(0.14779151747007227, 0.14779151747007228 * 1.001)


def test_close_both_zero():
    assert cs._close(0.0, 0.0)
    assert cs._close(-0.0, 0.0)


def test_close_small_values_use_abs_tol():
    """两边都接近 0 时，纯相对容差会把噪声放大 —— ABS_TOL 兜底。"""
    assert cs._close(1e-15, -1e-15)


# ------------------------------------------------------------------
# 2. 各来源取值路径（独立性的核心）
# ------------------------------------------------------------------
def _oos(overrides: dict | None = None) -> dict:
    d = {
        "strategy_test_segment": {
            "hold": 5,
            "cost_rate": 0.003,
            "n_periods": 143,
            "mean_turnover": 0.22110322298759919,
            "long_only": {"net_annualized": 0.14779151747007227, "sharpe": 0.5079523251702476},
            "benchmark": {"annualized": 0.16075356602254287},
        }
    }
    if overrides:
        d["strategy_test_segment"].update(overrides)
    return d


def test_from_oos_excess_is_difference_of_two_fields():
    """OOS 的超额是 long_only.net_annualized - benchmark.annualized 现场算的，
    不是从一个现成字段抄的 —— 这样才能与别处的独立算法对账。"""
    v = cs._from_oos(_oos(), "excess_annualized")
    assert v == pytest.approx(0.14779151747007227 - 0.16075356602254287, rel=1e-12)


def test_from_oos_missing_field_returns_none():
    d = _oos()
    d["strategy_test_segment"]["long_only"] = {}
    assert cs._from_oos(d, "excess_annualized") is None


def test_from_oos_unknown_key_returns_none():
    assert cs._from_oos(_oos(), "not_a_metric") is None


def test_from_cross_segment_selects_row_by_cost_and_hold():
    """必须按 cost_rate + hold 显式选行。

    rows 顺序是 for cost in (0.0015, 0.0030): for hold in (5,10,20,40)
    → rows[0] 是成本 0.15% 那行，取它会把 +0.3753%（正）当成 canonical。
    """
    d = {"result": {"rows": [
        {"cost_rate": 0.0015, "hold": 5,
         "per_segment": {"test": {"excess_annualized": 0.003753}}},   # 错的行
        {"cost_rate": 0.0030, "hold": 5,
         "per_segment": {"test": {"excess_annualized": -0.0129620485524706}}},  # canonical
        {"cost_rate": 0.0030, "hold": 20,
         "per_segment": {"test": {"excess_annualized": -0.0288}}},
    ]}}
    v = cs._from_cross_segment(d, "excess_annualized", 0.003, 5)
    assert v == pytest.approx(-0.0129620485524706, rel=1e-12)
    assert v != pytest.approx(0.003753, rel=1e-6), "绝对不能取到 rows[0]"


def test_from_cross_segment_missing_row_returns_none():
    """找不到 canonical 行 → MISSING，不许兜底取第一行。"""
    d = {"result": {"rows": [
        {"cost_rate": 0.0015, "hold": 5, "per_segment": {"test": {"excess_annualized": 0.003753}}},
    ]}}
    assert cs._from_cross_segment(d, "excess_annualized", 0.003, 5) is None


def test_from_cross_segment_accepts_result_or_top_level():
    """结构兼容：result 包裹或顶层都行。"""
    rows = [{"cost_rate": 0.003, "hold": 5, "per_segment": {"test": {"excess_annualized": -0.01}}}]
    assert cs._from_cross_segment({"result": {"rows": rows}}, "excess_annualized", 0.003, 5) == -0.01
    assert cs._from_cross_segment({"rows": rows}, "excess_annualized", 0.003, 5) == -0.01


def test_from_realism_rejects_wrong_cost_or_hold():
    d = {"r1_close": {"cost_rate": 0.0015, "hold": 5, "excess_annualized": 0.003}}
    assert cs._from_realism(d, "excess_annualized", 0.003, 5) is None
    d2 = {"r1_close": {"cost_rate": 0.003, "hold": 20, "excess_annualized": -0.0288}}
    assert cs._from_realism(d2, "excess_annualized", 0.003, 5) is None


def test_from_readiness_uses_benchmark_relative():
    d = {"result": {"benchmark_relative": {"excess_annualized": -0.0129620485524706}}}
    assert cs._from_readiness(d, "excess_annualized") == pytest.approx(-0.0129620485524706, rel=1e-12)


def test_internal_reconciliation_reads_cost_decomposition():
    d = {"result": {"cost_decomposition": {"annualized": {"net_excess": -0.012962048552470597}}}}
    assert cs._from_readiness_cost_decomposition(d) == pytest.approx(-0.012962048552470597, rel=1e-12)


# ------------------------------------------------------------------
# 3. 单位格式化（夏普是比值、期数是计数 —— 不能统一 ×100）
# ------------------------------------------------------------------
def test_fmt_sharpe_is_ratio_not_percent():
    assert "%" not in cs._fmt_value("sharpe", 0.5079523251702476)


def test_fmt_n_periods_is_count():
    out = cs._fmt_value("n_periods", 143)
    assert "%" not in out
    assert "143" in out


def test_fmt_excess_is_percent():
    assert cs._fmt_value("excess_annualized", -0.0129620485524706).endswith("%")


# ------------------------------------------------------------------
# 4. reconcile() 整体行为
# ------------------------------------------------------------------
def _write_artifacts(research_dir: Path, oos_excess: float, cs_excess: float,
                     realism_excess: float, readiness_excess: float) -> None:
    """写四份产物到 research_dir，其中三份一致、一份不同。

    注意：research_dir 直接就是产物目录（fixture 已指向 tmp/research），
    这里不要再套一层 "research" —— 套了就会写到 fixture 看不到的地方。
    """
    r = research_dir
    r.mkdir(parents=True, exist_ok=True)
    (r / "oos_latest.json").write_text(json.dumps({
        "strategy_test_segment": {
            "hold": 5, "cost_rate": 0.003, "n_periods": 143,
            "mean_turnover": 0.22,
            "long_only": {"net_annualized": 0.14779151747007227, "sharpe": 0.5079},
            "benchmark": {"annualized": 0.16075356602254287},
        }
    }), encoding="utf-8")
    (r / "cross_segment_check.json").write_text(json.dumps({
        "result": {"rows": [{"cost_rate": 0.003, "hold": 5,
                             "per_segment": {"test": {"excess_annualized": cs_excess}}}]}
    }), encoding="utf-8")
    (r / "execution_realism.json").write_text(json.dumps({
        "r1_close": {"cost_rate": 0.003, "hold": 5, "n_periods": 143, "mean_turnover": 0.22,
                     "excess_annualized": realism_excess,
                     "long_only": {"net_annualized": 0.14779151747007227, "sharpe": 0.5079},
                     "benchmark": {"annualized": 0.16075356602254287}}
    }), encoding="utf-8")
    (r / "live_readiness.json").write_text(json.dumps({
        "result": {
            "benchmark_relative": {"excess_annualized": readiness_excess,
                                   "long_only_annualized": 0.14779151747007227,
                                   "benchmark_annualized": 0.16075356602254287},
            "cost_decomposition": {"annualized": {"net_excess": readiness_excess},
                                   "turnover_per_period": 0.22},
        }
    }), encoding="utf-8")


@pytest.fixture()
def patched(tmp_path, monkeypatch):
    """把 cross_source 的产物路径指向临时目录。"""
    r = tmp_path / "research"
    monkeypatch.setattr(cs, "RESEARCH", r)
    monkeypatch.setattr(cs, "OOS_FILE", r / "oos_latest.json")
    monkeypatch.setattr(cs, "CROSS_SEG_FILE", r / "cross_segment_check.json")
    monkeypatch.setattr(cs, "REALISM_FILE", r / "execution_realism.json")
    monkeypatch.setattr(cs, "READINESS_FILE", r / "live_readiness.json")
    monkeypatch.setattr(cs, "SOURCE_FILES", {
        "oos_latest.json": r / "oos_latest.json",
        "cross_segment_check.json": r / "cross_segment_check.json",
        "execution_realism.json": r / "execution_realism.json",
        "live_readiness.json": r / "live_readiness.json",
    })
    return r


def test_reconcile_all_consistent(patched):
    _write_artifacts(patched, 0, -0.0129620485524706, -0.0129620485524706, -0.0129620485524706)
    res = cs.reconcile()
    assert res["consistent"] is True
    assert res["bad_metrics"] == []
    assert "可以出结论" in res["verdict"]


def test_reconcile_detects_mismatch(patched):
    """cross_segment 用旧面板的 -2.03%，其余是 -1.30% → 必须判不一致。"""
    _write_artifacts(patched, 0, -0.020277, -0.0129620485524706, -0.0129620485524706)
    res = cs.reconcile()
    assert res["consistent"] is False
    bad_keys = {b["key"] for b in res["bad_metrics"]}
    assert "excess_annualized" in bad_keys
    assert "拒绝出结论" in res["verdict"]
    m = next(x for x in res["metrics"] if x["key"] == "excess_annualized")
    assert m["status"] == "INCONSISTENT"
    assert m["spread"] == pytest.approx(abs(-0.020277 - (-0.0129620485524706)), rel=1e-9)


def test_reconcile_detects_internal_mismatch(patched):
    """体检模块内部两处算法不一致也必须报出来（2026-11-12 抓到 5 倍口径 bug 的地方）。"""
    _write_artifacts(patched, 0, -0.0129620485524706, -0.0129620485524706, -0.0129620485524706)
    p = patched / "live_readiness.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["result"]["cost_decomposition"]["annualized"]["net_excess"] = -0.1014  # 5 倍错
    p.write_text(json.dumps(d), encoding="utf-8")
    res = cs.reconcile()
    assert res["consistent"] is False
    assert res["internal_reconciliation"]["status"] == "INCONSISTENT"


def test_reconcile_missing_artifact_is_soft_not_hard(patched):
    """缺一份非关键来源 → 软告警，不阻断。

    语义分层（重要）：
      硬阻断 = 值互相矛盾（INCONSISTENT），或**关键指标**取不到/没人对过账
      软告警 = 非关键指标缺来源，或来源覆盖不全（值仍一致）
    「宁可缺数据，不给错数据」针对的是**矛盾**，不是「少一路交叉验证」。
    """
    _write_artifacts(patched, 0, -0.0129620485524706, -0.0129620485524706, -0.0129620485524706)
    (patched / "execution_realism.json").unlink()
    res = cs.reconcile()
    # 关键指标还有 3 个来源且一致 → 不阻断
    m = next(x for x in res["metrics"] if x["key"] == "excess_annualized")
    assert m["status"] == "CONSISTENT"
    assert "execution_realism.json" in m["missing_sources"]
    assert res["consistent"] is True
    # 但必须记为软告警，不能悄悄吞掉
    soft = res["soft_issues"]
    assert any("execution_realism.json" in (s.get("missing_sources") or [])
               or "execution_realism.json" in str(s) for s in soft)


def test_reconcile_key_metric_single_source_is_hard(patched):
    """关键指标只剩 1 个来源 → 硬阻断（没人跟它对过账，不能当事实呈现）。

    注意：2 个来源互相一致 = 真交叉验证，可以通过；
    1 个来源 = 「没人跟它对过账」，不能当事实呈现。
    """
    _write_artifacts(patched, 0, -0.0129620485524706, -0.0129620485524706, -0.0129620485524706)
    (patched / "cross_segment_check.json").unlink()
    (patched / "execution_realism.json").unlink()
    (patched / "live_readiness.json").unlink()
    res = cs.reconcile()
    m = next(x for x in res["metrics"] if x["key"] == "excess_annualized")
    assert m["status"] == "UNVERIFIED"
    assert res["consistent"] is False
    assert any(b["key"] == "excess_annualized" for b in res["bad_metrics"])


def test_reconcile_two_consistent_sources_passes(patched):
    """2 个独立来源互相一致 = 真交叉验证，应当通过。"""
    _write_artifacts(patched, 0, -0.0129620485524706, -0.0129620485524706, -0.0129620485524706)
    (patched / "cross_segment_check.json").unlink()
    (patched / "live_readiness.json").unlink()
    res = cs.reconcile()
    m = next(x for x in res["metrics"] if x["key"] == "excess_annualized")
    assert m["status"] == "CONSISTENT"
    assert len(m["values"]) == 2
    assert res["consistent"] is True


def test_reconcile_key_metric_all_missing_is_hard(patched):
    """关键指标一个来源都取不到 → 硬阻断。"""
    _write_artifacts(patched, 0, -0.0129620485524706, -0.0129620485524706, -0.0129620485524706)
    (patched / "cross_segment_check.json").unlink()
    (patched / "execution_realism.json").unlink()
    (patched / "live_readiness.json").unlink()
    (patched / "oos_latest.json").unlink()
    res = cs.reconcile()
    m = next(x for x in res["metrics"] if x["key"] == "excess_annualized")
    assert m["status"] == "MISSING"
    assert res["consistent"] is False


def test_reconcile_single_source_is_unverified(patched):
    """只有一个来源时不能叫「一致」—— 未经交叉验证。"""
    _write_artifacts(patched, 0, -0.0129620485524706, -0.0129620485524706, -0.0129620485524706)
    (patched / "cross_segment_check.json").unlink()
    (patched / "execution_realism.json").unlink()
    (patched / "live_readiness.json").unlink()
    res = cs.reconcile()
    m = next(x for x in res["metrics"] if x["key"] == "excess_annualized")
    assert m["status"] == "UNVERIFIED"


# ------------------------------------------------------------------
# 5. 真实产物对账（公开仓库无运行数据时 skip）
# ------------------------------------------------------------------
def test_real_artifacts_reconcile():
    if not (RESEARCH / "oos_latest.json").exists():
        pytest.skip("缺少运行时数据，跳过")
    res = cs.reconcile()
    bad = [(m["key"], m["status"]) for m in res["metrics"]
           if m["status"] in ("INCONSISTENT", "MISSING")]
    assert not bad, f"真实产物对账不通过：{bad}"
    assert res["consistent"] is True


def test_real_canonical_excess_matches_oos():
    """对账通过的 canonical 超额必须与 OOS 现场算的一致。"""
    if not (RESEARCH / "oos_latest.json").exists():
        pytest.skip("缺少运行时数据，跳过")
    res = cs.reconcile()
    m = next(x for x in res["metrics"] if x["key"] == "excess_annualized")
    oos = json.loads((RESEARCH / "oos_latest.json").read_text(encoding="utf-8"))
    seg = oos["strategy_test_segment"]
    expected = seg["long_only"]["net_annualized"] - seg["benchmark"]["annualized"]
    assert m["values"]["oos_latest.json"] == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------------------
# 6. 研究队列接线
# ------------------------------------------------------------------
def test_queue_has_cross_source_step():
    """对账必须是研究队列的一步 —— 否则产物会冻结在旧值，跟 oos 错位。"""
    import research_queue as rq
    steps = rq.STEPS if hasattr(rq, "STEPS") else rq.DEFAULT_STEPS
    keys = [s["key"] for s in steps]
    assert "cross_source" in keys
    assert len(keys) == len(set(keys)), "步骤 key 必须唯一"


def test_queue_step_dependencies_all_exist():
    """所有前置步骤都必须真实存在（字符串或列表都要检查）。"""
    import research_queue as rq
    steps = rq.STEPS if hasattr(rq, "STEPS") else rq.DEFAULT_STEPS
    keys = [s["key"] for s in steps]
    for s in steps:
        dep = s.get("requires")
        deps = [dep] if isinstance(dep, str) else list(dep or [])
        for one in deps:
            assert one in keys, f"{s['key']} 的前置 {one} 不存在于队列"


def test_queue_cross_source_requires_all_compared_artifacts():
    """跨来源对账必须等 4 份被对账的产物全部刷新完。

    少任何一份 → 可能拿「旧产物 vs 新产物」对账 → 误报口径不一致，
    把本来正常的日报挡掉。这种误报会让人忽略真正的告警。
    """
    import research_queue as rq
    steps = rq.STEPS if hasattr(rq, "STEPS") else rq.DEFAULT_STEPS
    step = next(s for s in steps if s["key"] == "cross_source")
    dep = step["requires"]
    deps = [dep] if isinstance(dep, str) else list(dep)
    for needed in ("cross_segment", "realism", "readiness"):
        assert needed in deps, f"cross_source 必须等 {needed} 完成"
