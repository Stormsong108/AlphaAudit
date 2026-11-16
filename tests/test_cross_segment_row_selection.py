"""跨区段三段数字的**行选择**回归测试。

2026-11-16 血泪：日报里同一个「测试段」出现两个数字。

    -2.03%  ← cross_segment_check.json（11-12 冻结产物，面板截止 11-10）
    -1.30%  ← oos_latest.json（当天 canonical，面板截止 11-13）

根因有两条，本文件分别锁死：

    R1 **产物会过期**：cross_segment_check.json 曾不在研究队列里，
       面板天天长新数据它却不动 → 它的数字永远是几周前的。
       锁法：产物必须写 generated_at，且队列里必须有它。

    R2 **行会选错**：cross_segment_check.json 里同时存着
       成本 0.15% 与 0.30% 两套行（rows 顺序是
       `for cost in (0.0015, 0.0030): for hold in (5,10,20,40)`），
       所以 `rows[0]` 是 **0.15%** 那行，不是 canonical 的 0.30%。
       旧代码写死 rows[0] → 日报三段数字取自 0.15% 口径，
       其测试段超额是 **+0.3753%（正数）**，与 canonical 的 -1.2962%
       方向相反 → 同一个测试段第四个数字。
       锁法：必须按 canonical 口径（cost=0.30% + canonical hold）选行，
       且选出来的测试段超额必须与 oos_latest 的 canonical 超额一致。

一致性铁律：**同一个量在系统里被几处独立算法算出来，
交叉比对，不一致就是口径串了。**
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import daily_brief as DB  # noqa: E402


# ── 造一份与真实产物同构的 cross_segment_check.json ──────────────────────

def _per(train: float, val: float, test: float) -> dict:
    return {
        "train": {"excess_annualized": train},
        "val": {"excess_annualized": val},
        "test": {"excess_annualized": test},
    }


def _cs_payload() -> dict:
    """rows 顺序刻意保持与 cross_segment.py 一致：先 cost 后 hold。"""
    return {
        "generated_at": "2026-11-16 12:15:49",
        "result": {
            "factor_names": ["mom_20"],
            "quantile": 0.5,
            "segments": {
                "train": ["2015-01-02", "2020-12-01"],
                "val": ["2020-12-02", "2023-11-21"],
                "test": ["2023-11-22", "2026-11-13"],
            },
            "rows": [
                # cost=0.15% —— 旧代码错取的就是这一行（rows[0]）
                {"cost_rate": 0.0015, "hold": 5,
                 "per_segment": _per(0.009102, -0.021073, 0.003753)},
                {"cost_rate": 0.0015, "hold": 10,
                 "per_segment": _per(-0.006189, 0.001057, 0.016651)},
                {"cost_rate": 0.0030, "hold": 5,
                 "per_segment": _per(-0.008163, -0.037793, CANON_TEST_EXCESS)},
                {"cost_rate": 0.0030, "hold": 10,
                 "per_segment": _per(-0.018131, -0.010844, 0.004745)},
            ],
        },
        "analysis": {"all_segments_negative": True},
    }


# 用全精度（真实产物里就是全精度）—— 断言 tol=1e-9，fixture 不能预先四舍五入。
CANON_TEST_EXCESS = -0.0129620485524706


@pytest.fixture
def cs_env(tmp_path, monkeypatch):
    monkeypatch.setattr(DB, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(DB, "HISTORY_FILE", tmp_path / "state" / "brief_history.json")
    monkeypatch.setattr(DB, "BRIEF_DIR", tmp_path / "briefs")
    monkeypatch.setattr(DB, "OOS_FILE", tmp_path / "oos_latest.json")
    monkeypatch.setattr(DB, "REALISM_FILE", tmp_path / "execution_realism.json")
    monkeypatch.setattr(DB, "LEDGER_SUMMARY", tmp_path / "ledger_summary.json")
    monkeypatch.setattr(DB, "LEDGER_STATE", tmp_path / "ledger_state.json")
    monkeypatch.setattr(DB, "ANATOMY_FILE", tmp_path / "signal_anatomy.json")
    monkeypatch.setattr(DB, "QUEUE_FILE", tmp_path / "research_queue.json")
    monkeypatch.setattr(DB, "DAILY_LOOP_FILE", tmp_path / "daily_loop.json")
    monkeypatch.setattr(DB, "CROSS_SEG_FILE", tmp_path / "cross_segment_check.json")
    return tmp_path


def _write_oos(env: Path, hold: int = 5) -> None:
    """canonical oos_latest.json：测试段超额 = CANON_TEST_EXCESS。

    hold 写在 strategy_test_segment.hold（真实 schema，live_readiness 读这里）
    和 config.hold 两处都放，避免只测到某一条读取路径。
    """
    (env / "oos_latest.json").write_text(json.dumps({
        "generated_at": "2026-11-16 03:29:55",
        "config": {"hold": hold, "cost_roundtrip": 0.003},
        "panel": {"to": "2026-11-13"},
        "strategy_test_segment": {
            "hold": hold,
            "n_periods": 143,
            "long_only": {"net_annualized": 0.14779151747007227},
            "benchmark": {"annualized": 0.16075356602254287},
        },
    }, ensure_ascii=False), encoding="utf-8")


def _write_cs(env: Path) -> None:
    (env / "cross_segment_check.json").write_text(
        json.dumps(_cs_payload(), ensure_ascii=False), encoding="utf-8")


# ── R2：行选择 ─────────────────────────────────────────────────────────────

def test_picks_canonical_cost_row_not_rows_zero(cs_env):
    """必须选 cost=0.30% + canonical hold 那一行，不是 rows[0]。

    rows[0] 是 cost=0.15%，其测试段超额是 +0.3753%（正数）——
    与 canonical 的 -1.2962% 方向相反。
    """
    _write_oos(cs_env)
    _write_cs(cs_env)
    snap = DB.build_snapshot()
    cs = snap["cross_segment"]
    assert cs["row_cost_rate"] == 0.0030
    assert cs["row_hold"] == 5
    assert cs["segments"]["test"] == pytest.approx(CANON_TEST_EXCESS)
    # 明确断言没有取到 rows[0]
    assert cs["segments"]["test"] != pytest.approx(0.003753)


def test_cross_segment_test_matches_canonical_oos_excess(cs_env):
    """铁律：同一量在两处独立算出，必须一致。

    cross_segment 的测试段超额 vs oos_latest 的 canonical 超额，
    不一致就说明口径串了，日报不得出结论。
    """
    _write_oos(cs_env)
    _write_cs(cs_env)
    snap = DB.build_snapshot()
    cs_test = snap["cross_segment"]["segments"]["test"]
    oos_ex = snap["oos"]["excess_annualized"]
    assert cs_test == pytest.approx(oos_ex, abs=1e-9)


def test_respects_canonical_hold_from_oos_latest(cs_env):
    """canonical hold 不是 5 时，也要选对应那一行（不写死 5）。"""
    _write_oos(cs_env, hold=10)
    _write_cs(cs_env)
    snap = DB.build_snapshot()
    cs = snap["cross_segment"]
    assert cs["row_hold"] == 10
    assert cs["row_cost_rate"] == 0.0030
    assert cs["segments"]["test"] == pytest.approx(0.004745)


def test_no_canonical_row_yields_unavailable_not_wrong_row(cs_env):
    """找不到 canonical 行时**宁可缺数据，不给错数据**。

    兜底取 rows[0]（0.15% 口径）= 用错口径，
    会让日报写出方向相反的测试段数字。
    """
    _write_oos(cs_env, hold=99)  # canonical hold=99，产物里没有这行
    _write_cs(cs_env)
    snap = DB.build_snapshot()
    cs = snap["cross_segment"]
    assert cs["available"] is False
    assert not cs.get("segments")
    assert cs.get("row_cost_rate") is None


# ── R1：产物会过期 ─────────────────────────────────────────────────────────

def test_snapshot_carries_generated_at_of_cross_segment(cs_env):
    """产物必须带 generated_at —— 否则日报无法判断数字是哪天的。"""
    _write_oos(cs_env)
    _write_cs(cs_env)
    snap = DB.build_snapshot()
    assert snap["cross_segment"]["generated_at"] == "2026-11-16 12:15:49"


def test_snapshot_carries_generated_at_of_oos(cs_env):
    _write_oos(cs_env)
    _write_cs(cs_env)
    snap = DB.build_snapshot()
    assert snap["oos"]["generated_at"] == "2026-11-16 03:29:55"


def test_cross_segment_is_in_research_queue():
    """cross_segment 必须在研究队列里，且依赖 oos。

    不在队列 → 产物冻结在某一版面板 → 日报里出现过期数字。
    """
    import research_queue as RQ
    steps = {s["key"]: s for s in RQ.STEPS}
    assert "cross_segment" in steps, "cross_segment 不在研究队列里，产物会冻结"
    assert steps["cross_segment"].get("requires") == "oos"


def test_cross_segment_script_writes_generated_at():
    """cross_segment.py 写盘时必须带 generated_at。"""
    src = (ROOT / "cross_segment.py").read_text(encoding="utf-8")
    assert '"generated_at"' in src, "cross_segment.py 写盘未带 generated_at"
