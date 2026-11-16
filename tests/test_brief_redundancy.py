"""日报里「因子冗余」那一段的回归测试。

铁律：**正文里的数字必须来自产物，不能写死在代码里**。
只要产物里的数字变了，正文必须跟着变 —— 否则读者看到的就是过期数字
（本项目已经吃过一次亏：日报内部同时出现 -2.03% 和 -1.30%）。

另外，跨来源对账门禁不通过时，日报整篇不出数字 —— 这里一并锁死。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import daily_brief

ROOT = Path(__file__).resolve().parents[1]


def _write_redundancy(tmp_path: Path, monkeypatch, eff_dim: float,
                      test_excess: float, full_excess: float,
                      flips: bool = False) -> Path:
    payload = {
        "ok": True,
        "generated_at": "2026-11-16 13:39:00",
        "panel_to": "2026-11-13",
        "canonical_factors": [f"f{i}" for i in range(11)],
        "redundancy": {"effective_dimension": eff_dim},
        "full_set": {"test": {
            "turnover": 0.2211,
            "excess_net_winsor_annualized": full_excess,
            "excess_net_median_annualized": 0.0664,
        }},
        "frozen": {
            "k": 3,
            "factor_names": ["ma_dev_60", "mom_60", "turnover_trend"],
            "spec": {"spec_hash": "deadbeef"},
            "val_gate_passed": True,
            "val": {"excess_net_winsor_annualized": -0.0093},
            "test": {
                "turnover": 0.1542,
                "excess_net_winsor_annualized": test_excess,
                "excess_net_median_annualized": 0.0548,
                "excess_net_raw_annualized": 0.0166,
            },
        },
        "improvement_vs_full": {
            "val_excess_annualized_delta": 0.0285,
            "test_excess_annualized_delta": 0.0136,
        },
        "verdict": {"flips_on_test": flips, "conclusion": "测试用"},
    }
    p = tmp_path / "factor_redundancy.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(daily_brief, "REDUNDANCY_FILE", p)
    return p


def test_brief_shows_redundancy_numbers_from_artifact(tmp_path, monkeypatch):
    _write_redundancy(tmp_path, monkeypatch, eff_dim=2.5,
                      test_excess=0.0006, full_excess=-0.0130)
    snap = daily_brief.build_snapshot()
    assert snap["redundancy"]["available"] is True
    assert snap["redundancy"]["effective_dimension"] == 2.5

    text = daily_brief.build_brief(snap, prev={})
    assert "有效维度只有 2.50" in text
    assert "0.2211" in text and "0.1542" in text
    assert "ma_dev_60" in text
    # 数字不能写死：产物值变了正文必须跟着变
    assert "2.50" in text


def test_brief_numbers_follow_the_artifact(tmp_path, monkeypatch):
    """换一份产物，正文数字必须跟着换。"""
    _write_redundancy(tmp_path, monkeypatch, eff_dim=7.77,
                      test_excess=0.0123, full_excess=-0.0130)
    text = daily_brief.build_brief(daily_brief.build_snapshot(), prev={})
    assert "有效维度只有 7.77" in text
    assert "2.50" not in text
    assert "+1.23%" in text


def test_brief_says_cost_saved_but_no_alpha_when_not_flipped(tmp_path, monkeypatch):
    _write_redundancy(tmp_path, monkeypatch, eff_dim=3.56,
                      test_excess=0.0006, full_excess=-0.0130, flips=False)
    text = daily_brief.build_brief(daily_brief.build_snapshot(), prev={})
    assert "省得出成本，省不出 alpha" in text
    assert "测试段这个区间本项目已经反复看过多次" in text


def test_brief_says_passed_when_flipped(tmp_path, monkeypatch):
    _write_redundancy(tmp_path, monkeypatch, eff_dim=3.56,
                      test_excess=0.0500, full_excess=-0.0130, flips=True)
    text = daily_brief.build_brief(daily_brief.build_snapshot(), prev={})
    assert "这一轮算过线了" in text
    assert "省得出成本，省不出 alpha" not in text


def test_brief_omits_section_when_no_artifact(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(daily_brief, "REDUNDANCY_FILE", missing)
    snap = daily_brief.build_snapshot()
    assert "redundancy" not in snap
    text = daily_brief.build_brief(snap, prev={})
    assert "另外做了一个新研究" not in text


def test_brief_gated_when_reconciliation_inconsistent(tmp_path, monkeypatch):
    """对账不通过 → 整篇不出数字，冗余那段的数字也不能漏出去。"""
    _write_redundancy(tmp_path, monkeypatch, eff_dim=3.56,
                      test_excess=0.0006, full_excess=-0.0130)
    snap = daily_brief.build_snapshot()
    snap["reconciliation"] = {
        "available": True,
        "consistent": False,
        "bad_metrics": [{"key": "excess", "label": "测试段净超额",
                         "status": "INCONSISTENT", "magnitude": "明显"}],
    }
    text = daily_brief.build_brief(snap, prev={})
    assert "口径对账未通过" in text
    assert "有效维度只有" not in text
    assert "0.1542" not in text
