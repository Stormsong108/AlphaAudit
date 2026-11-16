"""因子冗余分析的回归测试。

本模块回答的问题：11 个因子是不是其实只有少数几个独立信号？
用更少的因子能不能把换手降下来，从而把成本缺口补上？

本文件锁死的不变量：

1. **看的是「相对基准的超额」，不是「多头绝对收益」**。
   ooslab 的 ``long_only.net_annualized`` 是多头组合自己的收益，里面全是 beta；
   策略有没有 alpha 只看 excess = long − bench。
   第一版实现误用了绝对收益，结论方向完全反了 —— 这里锁死口径。

2. **翻盘判据不能是「抗成本 > 实际成本」这种一刀切的比值**。
   换手小的时候这个比值天然大，净额可能只有零点几个 bp，
   是刀刃上的判定。必须三口径全正 + 胜率过半 + 幅度超过门槛才算过线。

3. **按口径选行，不能按索引取第一行**。
   cross_segment_check.json 里 cost=0.15% 的行排在最前面，
   取第一行会拿到另一套口径的数字（本项目已踩过一次）。

4. **同一个量在几处独立算法算出，必须逐位一致**。
   本模块重算的 canonical 三段超额，要和 cross_segment.py
   （另一条代码路径）的产物逐位相同；不一致就拒绝出结论。

5. **合成口径不能自创**：_composite 必须和 ooslab.composite_score 逐位一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import factor_redundancy as fr
import ooslab

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "data" / "research"


# ------------------------------------------------------------------
# 1. 平均秩（含并列、NaN）
# ------------------------------------------------------------------
def test_rank_row_simple():
    v = np.array([3.0, 1.0, 2.0])
    r = fr._rank_row(v)
    assert list(r) == [3.0, 1.0, 2.0]


def test_rank_row_ties_get_average_rank():
    """并列值必须取平均秩，否则相关性会偏。"""
    v = np.array([10.0, 20.0, 20.0, 30.0])
    r = fr._rank_row(v)
    assert list(r) == [1.0, 2.5, 2.5, 4.0]


def test_rank_row_keeps_nan():
    v = np.array([1.0, np.nan, 3.0])
    r = fr._rank_row(v)
    assert np.isnan(r[1])
    assert r[0] == 1.0 and r[2] == 2.0


def test_rank_row_all_nan():
    v = np.array([np.nan, np.nan])
    r = fr._rank_row(v)
    assert np.all(np.isnan(r))


# ------------------------------------------------------------------
# 2. 超额三口径
# ------------------------------------------------------------------
def _fake_periods():
    return [
        {"long": 0.02, "short": 0.0, "bench": 0.01,
         "long_raw": 0.03, "short_raw": 0.0, "long_med": 0.015, "bench_med": 0.005,
         "turnover": 0.10},
        {"long": -0.01, "short": 0.0, "bench": 0.00,
         "long_raw": -0.02, "short_raw": 0.0, "long_med": -0.005, "bench_med": 0.0,
         "turnover": 0.20},
    ]


def test_excess_calibers_are_excess_not_absolute():
    """关键口径：算的是 (long − bench) − cost×turn，不是多头绝对收益。"""
    out = fr._excess_calibers(_fake_periods(), cost_rate=0.003, hold=5)
    # 缩尾口径：期1 (0.02-0.01) - 0.003*0.10 = 0.01 - 0.0003 = 0.0097
    #           期2 (-0.01-0.00) - 0.003*0.20 = -0.01 - 0.0006 = -0.0106
    assert out["excess_net_winsor_per_period"] == pytest.approx((0.0097 - 0.0106) / 2)
    # 中位数口径：期1 (0.015-0.005)-0.0003 = 0.0097；期2 (-0.005-0)-0.0006 = -0.0056
    assert out["excess_net_median_per_period"] == pytest.approx((0.0097 - 0.0056) / 2)
    # 未缩尾口径：期1 (0.03-0.01)-0.0003 = 0.0197；期2 (-0.02-0)-0.0006 = -0.0206
    assert out["excess_net_raw_per_period"] == pytest.approx((0.0197 - 0.0206) / 2)


def test_excess_calibers_annualization_uses_hold_not_252():
    """年化必须用 252/hold，不能用 252（hold=5 时差 5 倍）。"""
    out = fr._excess_calibers(_fake_periods(), cost_rate=0.003, hold=5)
    assert out["excess_net_winsor_annualized"] == pytest.approx(
        out["excess_net_winsor_per_period"] * 252.0 / 5
    )
    out10 = fr._excess_calibers(_fake_periods(), cost_rate=0.003, hold=10)
    assert out10["excess_net_winsor_annualized"] == pytest.approx(
        out10["excess_net_winsor_per_period"] * 252.0 / 10
    )


def test_excess_calibers_hit_rate():
    out = fr._excess_calibers(_fake_periods(), cost_rate=0.003, hold=5)
    assert out["excess_net_winsor_hit_rate"] == pytest.approx(0.5)


# ------------------------------------------------------------------
# 3. 合成口径必须和 ooslab.composite_score 逐位一致
# ------------------------------------------------------------------
def test_composite_matches_ooslab_bitwise():
    rng = np.random.default_rng(7)
    T, N = 12, 40
    factors = {
        "a": rng.normal(size=(T, N)),
        "b": rng.normal(size=(T, N)),
        "c": rng.normal(size=(T, N)),
    }
    factors["a"][:, :3] = np.nan          # 制造缺失，检验「有效因子数」归一
    mat = {"factors": factors, "close": rng.normal(size=(T, N)),
           "dates": [f"d{i}" for i in range(T)], "codes": [f"c{i}" for i in range(N)]}
    names = ["a", "b", "c"]
    signs = {"a": 1.0, "b": -1.0, "c": 1.0}
    zc = fr._zscore_cache(mat, names)
    mine = fr._composite(zc, names, signs)
    ref = ooslab.composite_score(mat, names, [signs[n] for n in names])
    assert np.allclose(mine, ref, rtol=0, atol=0, equal_nan=True)


# ------------------------------------------------------------------
# 4. 冻结规则
# ------------------------------------------------------------------
def _step(k, set_, net, gross, turnover, be=0.001):
    return {"k": k, "set": list(set_), "excess_net_mean_per_period": net,
            "gross_per_period": gross, "turnover": turnover,
            "breakeven_cost": be, "excess_net_winsor_annualized": net * 50.4}


def test_choose_frozen_picks_max_net_excess():
    paths = {
        "gross": [_step(1, ["a"], 0.0001, 0.001, 0.20), _step(2, ["a", "b"], 0.0003, 0.002, 0.25)],
        "net": [_step(1, ["c"], 0.0005, 0.003, 0.30)],
    }
    got = fr._choose_frozen(paths)
    assert got["set"] == ["c"]
    assert got["path"] == "net"


def test_choose_frozen_tie_breaks_on_lower_turnover():
    """并列时换手低的优先（事先定好的次序，不看结果再改）。"""
    paths = {
        "net": [
            _step(1, ["a"], 0.0005, 0.003, 0.30),
            _step(2, ["b"], 0.0005, 0.003, 0.10),
        ],
    }
    got = fr._choose_frozen(paths)
    assert got["set"] == ["b"]
    assert got["turnover"] == 0.10


def test_choose_frozen_returns_none_when_empty():
    assert fr._choose_frozen({}) is None


# ------------------------------------------------------------------
# 5. spec_hash
# ------------------------------------------------------------------
def test_spec_hash_ignores_meta_keys():
    spec = {"a": 1, "b": [1, 2]}
    h1 = fr.spec_hash(spec)
    spec2 = dict(spec, spec_hash="x", frozen_at="2026-01-01", selected_by="me")
    assert fr.spec_hash(spec2) == h1


def test_spec_hash_changes_with_content():
    h1 = fr.spec_hash({"a": 1})
    h2 = fr.spec_hash({"a": 2})
    assert h1 != h2


# ------------------------------------------------------------------
# 6. 按口径选行（绝不能取 rows[0]）
# ------------------------------------------------------------------
def test_cross_segment_reference_selects_by_cost_and_hold(tmp_path, monkeypatch):
    """rows[0] 是 cost=0.15% 那行 —— 按索引取会拿到另一套口径。"""
    payload = {
        "result": {
            "rows": [
                {"cost_rate": 0.0015, "hold": 5, "per_segment": {"train": {"excess_annualized": 111.0}}},
                {"cost_rate": 0.0015, "hold": 10, "per_segment": {"train": {"excess_annualized": 222.0}}},
                {"cost_rate": 0.0030, "hold": 5, "per_segment": {"train": {"excess_annualized": 333.0}}},
                {"cost_rate": 0.0030, "hold": 10, "per_segment": {"train": {"excess_annualized": 444.0}}},
            ]
        }
    }
    monkeypatch.setattr(fr, "RESEARCH", tmp_path)
    (tmp_path / "cross_segment_check.json").write_text(
        json.dumps(payload), encoding="utf-8")
    got = fr._cross_segment_reference(0.0030, 5)
    assert got["train"]["excess_annualized"] == 333.0
    got2 = fr._cross_segment_reference(0.0030, 10)
    assert got2["train"]["excess_annualized"] == 444.0


def test_cross_segment_reference_returns_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "RESEARCH", tmp_path)
    assert fr._cross_segment_reference(0.0030, 5) == {}


# ------------------------------------------------------------------
# 7. 集成：需要运行时数据
# ------------------------------------------------------------------
def _require(path: str):
    if not (ROOT / path).exists():
        pytest.skip(f"{path} 不存在，先跑对应的研究脚本")


@pytest.fixture(scope="module")
def analysis():
    """整个模块只跑一次 analyze()（约 15 秒），后面的用例共用结果。"""
    _require("data/research/oos_latest.json")
    _require("data/research/cross_segment_check.json")
    out = fr.analyze()
    assert out.get("ok"), out.get("reason")
    return out


def test_analyze_reproduces_canonical_three_segments(analysis):
    """重算的 canonical 三段超额必须与 cross_segment.py 的产物逐位一致。

    这是「同一个量在几处独立算法算出」的交叉比对 ——
    不一致就是口径串了，当场拒绝出结论。
    """
    out = analysis
    assert out["cross_check"]["ok"], out["cross_check"]
    assert out["segment_check"]["ok"], out["segment_check"]
    for seg in ("train", "val", "test"):
        row = out["segment_check"]["rows"].get(seg)
        assert row is not None, f"{seg} 段没有参与交叉校验"
        assert row["ok"], f"{seg} 段与 cross_segment_check.json 不一致：{row}"


def test_analyze_reduces_turnover(analysis):
    """降换手这个动作本身必须有效：子集换手显著低于全量。"""
    sub = analysis["frozen"]["test"]
    full = analysis["full_set"]["test"]
    assert sub["turnover"] < full["turnover"]


def test_analyze_reports_three_calibers(analysis):
    """三口径必须并列输出，不能只给一个数。"""
    for seg in ("val", "test"):
        m = analysis["frozen"][seg]
        for key in ("excess_net_winsor_annualized",
                    "excess_net_median_annualized",
                    "excess_net_raw_annualized"):
            assert key in m, f"{seg} 段缺 {key}"


def test_frozen_spec_is_written_and_hash_matches(analysis):
    """冻结件必须带 spec_hash，且 hash 能从规格本身重算出来。"""
    spec = analysis["frozen"]["spec"]
    assert spec["spec_hash"] == fr.spec_hash(spec)
    assert spec["selection_segment"] == "train"
    assert 1 <= len(spec["factor_names"]) <= 11
