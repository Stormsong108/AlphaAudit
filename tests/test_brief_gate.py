"""日报正文的对账门禁 —— 端到端行为测试。

背景（2026-11-16）：
  日报同时出现 -2.03%（11-12 冻结的 cross-segment 产物）和 -1.30%（当天 canonical OOS），
  方向相反。当时靠人眼交叉比对发现。

  cross_source.py 把对账做成机器判定；daily_brief.py 负责在门禁不通过时
  **不给出任何具体数字**。

本文件锁死的不变量（正文层，不是模块层）：
  1. 对账不通过 → 正文里**不能出现任何具体百分比数字**
     （只说"口径对账未通过"，列出对不上的指标名）
  2. 对账通过 → 正文恢复正常，并显式说明"口径已对账"
  3. 对账产物缺失 → 不能静默当作"通过"，也不能崩
     （没门禁 ≠ 有矛盾，但也不能装作验证过了）
  4. 门禁必须在结论之前触发 —— 不能先写数字再补一句免责声明

第 1 条是最关键的：**挑一个来源显示 = 编造确定性**。
数字本身可能都是真的，但读者无法知道该信哪个 → 呈现即为误导。
"""

from __future__ import annotations

import json
import re

import pytest

import daily_brief as db


# 一份最小可用的快照：有 oos / realism / ledger，但数字会被门禁覆盖。
def _snap(reconciliation: dict | None) -> dict:
    return {
        "oos": {"excess_annualized": -0.0129620485524706,
                "long_only_annualized": 0.14779151747007227,
                "benchmark_annualized": 0.16075356602254287},
        "realism": {"close_excess": -0.0129, "open_excess": -0.0267,
                    "open_limit_excess": -0.0267},
        "ledger": {"n_periods": 0},
        "reconciliation": reconciliation or {},
    }


def _xs(consistent: bool, bad_metrics: list | None = None) -> dict:
    return {
        "available": True,
        "consistent": consistent,
        "verdict": ("各来源数值一致，可以出结论" if consistent
                    else "存在对不上的指标 —— 拒绝出结论，先查口径"),
        "generated_at": "2026-11-16 12:58:58",
        "canonical": {"cost_rate": 0.003, "hold": 5, "quantile": 0.5},
        "bad_metrics": bad_metrics or [],
        "internal_bad": False,
        "excess_annualized": -0.0129620485524706 if consistent else None,
    }


def _bad(label: str = "测试段净超额（相对等权基准，年化）") -> list:
    """模拟 bad_metrics —— 只带定性信息（与 build_snapshot 产出的形状一致）。"""
    return [{"key": "excess_annualized", "label": label,
             "status": "INCONSISTENT", "magnitude": "明显"}]


# ------------------------------------------------------------------
# 1. 对账不通过 → 正文零数字
# ------------------------------------------------------------------
def test_gate_blocks_when_inconsistent():
    """核心不变量：对账不通过时，正文里不能出现任何具体百分比数字。"""
    snap = _snap(_xs(False, _bad()))
    text = db.build_brief(snap=snap, prev={})

    assert "口径对账未通过" in text
    assert "暂不出结论" in text
    # 关键断言：正文里不能出现任何百分比数字
    pcts = re.findall(r"[-+]?\d+\.?\d*\s*%", text)
    assert not pcts, f"门禁未通过时正文仍出现百分比数字：{pcts}"
    # 也不能出现具体的年化数字
    assert "-1.30%" not in text
    assert "-1.296205%" not in text


def test_gate_lists_the_offending_metrics():
    """必须点名是哪个指标对不上，否则读者不知道该查什么。"""
    snap = _snap(_xs(False, [
        {"key": "excess_annualized", "label": "测试段净超额（年化）",
         "status": "INCONSISTENT", "magnitude": "明显"},
        {"key": "sharpe", "label": "测试段夏普", "status": "INCONSISTENT",
         "magnitude": "明显"},
    ]))
    text = db.build_brief(snap=snap, prev={})
    assert "测试段净超额（年化）" in text
    assert "测试段夏普" in text


def test_gate_reports_internal_mismatch():
    """体检模块内部两处算法不一致也必须触发门禁。"""
    xs = _xs(False, [{"key": "excess_annualized", "label": "测试段净超额",
                      "status": "INCONSISTENT", "magnitude": "明显"}])
    xs["internal_bad"] = True
    snap = _snap(xs)
    text = db.build_brief(snap=snap, prev={})
    assert "口径对账未通过" in text
    assert "两处独立算法" in text


def test_gate_message_explains_why_not_numbers():
    """必须解释为什么不出数字 —— 否则看起来像故障，不像纪律。"""
    snap = _snap(_xs(False, _bad("x")))
    text = db.build_brief(snap=snap, prev={})
    assert "不给出任何具体数字" in text
    assert "口径串了" in text or "口径" in text


def test_gate_states_not_that_strategy_changed():
    """要区分「数据自相矛盾」和「策略变好变坏」—— 否则读者误判。"""
    snap = _snap(_xs(False, _bad("x")))
    text = db.build_brief(snap=snap, prev={})
    assert "不代表策略变好或变坏" in text


# ------------------------------------------------------------------
# 2. 对账通过 → 正常出结论
# ------------------------------------------------------------------
def test_gate_passes_when_consistent():
    snap = _snap(_xs(True))
    text = db.build_brief(snap=snap, prev={})
    assert "口径对账未通过" not in text
    assert "口径已对账" in text
    # 正常路径下数字应该出现
    assert "-1.30%" in text


def test_consistent_path_still_reports_reasons():
    """通过门禁不代表策略可用 —— 不可用的理由还得照说。"""
    snap = _snap(_xs(True))
    text = db.build_brief(snap=snap, prev={})
    assert "还不能用" in text
    assert "-2.67%" in text  # 现实成交口径
    assert "12" in text       # 前瞻账本还差 12 期


# ------------------------------------------------------------------
# 3. 对账产物缺失 → 不崩、不装作验证过
# ------------------------------------------------------------------
def test_missing_reconciliation_does_not_crash():
    """对账产物还没生成时，日报照常出，不能崩。"""
    snap = _snap(None)
    text = db.build_brief(snap=snap, prev={})
    assert text  # 有输出
    assert "口径对账未通过" not in text


def test_missing_reconciliation_not_claimed_as_verified():
    """缺对账产物 ≠ 对账通过 —— 不能写"口径已对账"。"""
    snap = _snap(None)
    text = db.build_brief(snap=snap, prev={})
    assert "口径已对账" not in text


# ------------------------------------------------------------------
# 4. 门禁位置：必须在结论之前
# ------------------------------------------------------------------
def test_gate_message_precedes_any_numbers():
    """门禁提示必须出现在正文靠前位置，不能埋在末尾当免责声明。"""
    snap = _snap(_xs(False, _bad("x")))
    text = db.build_brief(snap=snap, prev={})
    gate_pos = text.index("口径对账未通过")
    assert gate_pos < 400, "门禁提示出现得太晚 —— 读者已经先看到结论了"


def test_gate_blocks_before_usage_section():
    """门禁不通过时，不应出现「今天做了什么」这类正常内容 ——
    说明是提前 return 了，不是结尾补一句。"""
    snap = _snap(_xs(False, _bad("x")))
    text = db.build_brief(snap=snap, prev={})
    assert "一、今天做了什么" not in text


# ------------------------------------------------------------------
# 5. 快照层：reconciliation 字段的构造
# ------------------------------------------------------------------
def test_snapshot_reconciliation_shape():
    """build_snapshot 读到的对账产物要转成正确形状。"""
    from pathlib import Path
    import cross_source as cs

    if not (Path(cs.RESEARCH) / "oos_latest.json").exists():
        pytest.skip("缺少运行时数据，跳过")
    res = cs.reconcile()
    snap = db.build_snapshot()
    rec = snap.get("reconciliation") or {}
    assert rec.get("available") is True
    assert "consistent" in rec
    assert "verdict" in rec
    assert "generated_at" in rec
    if rec["consistent"]:
        # 通过时必须能取到 canonical 超额供正文引用
        assert rec["excess_annualized"] is not None


# ------------------------------------------------------------------
# 6. 真正端到端：产物文件 → build_snapshot → build_brief
# ------------------------------------------------------------------
# 前面所有用例都直接注入 snap，绕过了 build_snapshot —— 那条路径没验到。
# 这里用 monkeypatch 换掉对账产物文件的读取，验证真实链路。
def _fake_cross_src(tmp_path, monkeypatch, consistent: bool) -> None:
    """造一份对账产物文件，让 build_snapshot 读到它。"""
    f = tmp_path / "cross_source_check.json"
    metrics = [
        {"key": "excess_annualized", "label": "测试段净超额（年化）",
         "status": "CONSISTENT" if consistent else "INCONSISTENT",
         "spread": 0.0 if consistent else -0.0073,
         "values": {"oos_latest.json": -0.0129620485524706}},
        {"key": "sharpe", "label": "测试段夏普", "status": "CONSISTENT",
         "spread": 0.0, "values": {"oos_latest.json": 0.5079}},
    ]
    payload = {
        "generated_at": "2026-11-16 12:58:58",
        "canonical": {"cost_rate": 0.003, "hold": 5, "quantile": 0.5},
        "consistent": consistent,
        "verdict": ("各来源数值一致，可以出结论" if consistent
                    else "存在对不上的指标 —— 拒绝出结论，先查口径"),
        "metrics": metrics,
        "internal_reconciliation": {"status": "CONSISTENT"},
        "bad_metrics": [] if consistent else [{
            "key": "excess_annualized", "label": "测试段净超额（年化）",
            "status": "INCONSISTENT", "spread": -0.0073,
        }],
    }
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(db, "CROSS_SRC_FILE", f)


def test_end_to_end_inconsistent_gives_zero_numbers(tmp_path, monkeypatch):
    """端到端：产物文件说对不上 → 日报正文一个百分比数字都没有。

    这是最接近真实运行的路径：build_snapshot 读产物 → build_brief 出正文。
    """
    _fake_cross_src(tmp_path, monkeypatch, consistent=False)
    snap = db.build_snapshot()
    text = db.build_brief(snap=snap, prev={})

    assert "口径对账未通过" in text
    pcts = re.findall(r"[-+]?\d+\.?\d*\s*%", text)
    assert not pcts, f"端到端路径仍出现百分比数字：{pcts}"


def test_end_to_end_consistent_gives_numbers(tmp_path, monkeypatch):
    """端到端：对账通过 → 正文正常给数字。"""
    _fake_cross_src(tmp_path, monkeypatch, consistent=True)
    snap = db.build_snapshot()
    text = db.build_brief(snap=snap, prev={})
    assert "口径对账未通过" not in text
    assert "口径已对账" in text


def test_end_to_end_snapshot_has_no_spread_number(tmp_path, monkeypatch):
    """快照里也不能留 spread 数值 —— 会被误引用。"""
    _fake_cross_src(tmp_path, monkeypatch, consistent=False)
    snap = db.build_snapshot()
    rec = snap.get("reconciliation") or {}
    blob = json.dumps(rec, ensure_ascii=False)
    assert "-0.0073" not in blob and "-0.00730000" not in blob
    for b in rec.get("bad_metrics") or []:
        assert "spread" not in b
        assert b.get("magnitude")
