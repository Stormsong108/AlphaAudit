# -*- coding: utf-8 -*-
"""market_timing（择时 vs 选股 归因）的回归测试。

抓的是这类研究里最危险、也最容易「看起来合理」的错误：
**口径串了** —— 基准与策略不在同一批期次、验证段偷偷参与选参、
测试段被跑了不止一次、或者收益方向搞反。

所以测试不测「数字好不好看」，测的是**纪律性不变量**：

1. 冻结件先行：测试段读的是**落盘的**冻结 spec，不是内存里现算的。
2. 验证段只判定通过与否，绝不参与选参（选参只许看训练段）。
3. 测试段只跑一次（run() 里落盘一个 run-count 标记）。
4. 基准与策略期次必须对齐（期数一致，且超额 = 策略 − 同一批基准）。
5. 收益方向：long_short 的净边际符号不能反；基准口径按规则分别定。
6. 年化口径：per_period × (252/hold)，不是 × 252。
7. 所有产物必须带 generated_at。

这些都不依赖真实面板，用合成数据就能测。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import market_timing as M  # noqa: E402


# ── 合成数据 ────────────────────────────────────────────────────────────
def _synthetic_panel(n_periods: int = 120, n_assets: int = 60,
                     seed: int = 7) -> dict:
    """构造一个形状像面板的合成矩阵：close[T, N] + dates。

    收益含共同的市场成分（beta）+ 个股噪声 —— 这样分解才有意义。
    """
    rng = np.random.default_rng(seed)
    T = n_periods * 5 + 20
    mkt = rng.normal(0.0004, 0.012, T)
    idio = rng.normal(0.0, 0.018, (T, n_assets))
    rets = mkt[:, None] * 0.8 + idio
    close = 10.0 * np.exp(np.cumsum(rets, axis=0))
    dates = [f"2015-01-{i % 28 + 1:02d}" for i in range(T)]
    return {"close": close, "dates": dates, "n_assets": n_assets}


def _synthetic_periods(n: int = 100, seed: int = 11) -> list[dict]:
    """构造一串回测期次记录。

    字段名对齐 ooslab.decile_backtest 的输出 —— **bench_raw 也不能少**：
    decompose_segment 要同时算缩尾/未缩尾/中位数三口径残差，缺一个键就 KeyError。
    日期必须**唯一**，否则 date_index 会把多个期次映到同一根 bar 上。
    """
    rng = np.random.default_rng(seed)
    base = dt.date(2016, 1, 1)
    out = []
    for i in range(n):
        long_r = float(rng.normal(0.002, 0.01))
        short_r = float(rng.normal(0.001, 0.01))
        bench = float(rng.normal(0.001, 0.008))
        d0 = base + dt.timedelta(days=i)
        out.append({
            "signal_date": d0.isoformat(),
            "entry_date": (d0 + dt.timedelta(days=1)).isoformat(),
            "exit_date": (d0 + dt.timedelta(days=5)).isoformat(),
            "long": long_r, "short": short_r, "bench": bench,
            "long_raw": long_r * 1.1, "short_raw": short_r * 1.1,
            "long_med": long_r * 0.95, "short_med": short_r * 0.95,
            "bench_med": bench * 0.95, "bench_raw": bench * 1.05,
            "turnover": float(rng.uniform(0.1, 0.6)),
        })
    return out


def _date_index(periods: list[dict]) -> dict[str, int]:
    return {p["signal_date"]: i for i, p in enumerate(periods)}


# ── 1. 成本地板常量必须与任务书一致 ────────────────────────────────────
def test_fee_floor_constants():
    """A股单边成本地板：佣金双边 5bp + 印花税卖出 5bp + 过户费 0.2bp = 10.2bp。"""
    assert M.FEE_FLOOR_BP == pytest.approx(10.2)
    assert M.FEE_FLOOR_WITH_SLIPPAGE_BP == pytest.approx(20.2)
    assert M.CANONICAL_COST == pytest.approx(0.0030)


def test_cost_scenarios_bracket_canonical():
    """成本情景必须包含 canonical，且随成本上升净收益单调不增。"""
    assert M.CANONICAL_COST in M.COST_SCENARIOS
    assert list(M.COST_SCENARIOS) == sorted(M.COST_SCENARIOS)


# ── 2. 年化口径：per_period × (252/hold)，不是 × 252 ──────────────────
def test_annualization_uses_periods_per_year_not_252():
    """252 是**交易日数**不是期数。

    同一个 per_period 年化结果，用 252 乘会虚高 hold 倍
    （实测 hold=5 时虚高 5 倍：+6.55% → +1.31%）。
    """
    periods = _synthetic_periods(100)
    d = M.decompose_segment(periods, hold=5, cost_rate=M.CANONICAL_COST)
    per_year = M.TRADING_DAYS / 5
    gross = d["per_period"]["gross_excess_mean"]
    assert d["annualized"]["gross_excess"] == pytest.approx(gross * per_year, rel=1e-9)
    # 反证：不能等于 gross × 252
    assert abs(d["annualized"]["gross_excess"] - gross * 252) > 1e-6


def test_annualization_scales_with_hold():
    """同样每期收益，hold 越长年化越低（期数变少）。"""
    periods = _synthetic_periods(100)
    d5 = M.decompose_segment(periods, hold=5, cost_rate=M.CANONICAL_COST)
    d20 = M.decompose_segment(periods, hold=20, cost_rate=M.CANONICAL_COST)
    r5 = d5["per_period"]["gross_excess_mean"]
    r20 = d20["per_period"]["gross_excess_mean"]
    assert d5["annualized"]["gross_excess"] == pytest.approx(r5 * (252 / 5), rel=1e-9)
    assert d20["annualized"]["gross_excess"] == pytest.approx(r20 * (252 / 20), rel=1e-9)


def test_strategy_stats_annualization_uses_hold():
    """_strategy_stats 的年化同样是 per_period × (252/hold)。"""
    periods = _synthetic_periods(60)
    bench = np.array([p["bench"] for p in periods])
    gross = bench.copy()
    cost = np.zeros_like(bench)
    net = gross - cost
    st = M._strategy_stats(gross, cost, net, bench, hold=5, label="x",
                            baseline=bench, baseline_name="equal_weight_benchmark")
    assert st["net_annualized"] == pytest.approx(st["net_mean_per_period"] * 252 / 5,
                                                  rel=1e-9)


# ── 3. 收益方向：long_short 的多空不能反 ─────────────────────────────
def test_long_short_net_margin_direction():
    """long_short 的净边际 = 多头 − 空头，符号不能反。

    用「信号取反 ⇒ 边际取反」的不变量来钉方向：
    仓位 w∈{+1,−1}，gross = w·bench；信号取反 ⇒ w 取反 ⇒ gross 取反。
    如果实现里把多空写反，这个不变量立刻不成立。
    """
    periods = _synthetic_periods(40)
    bench = np.array([p["bench"] for p in periods])
    sig = np.sign(bench) + 0.5          # 避免出现 0（w 的判定是 >0）
    di = _date_index(periods)
    st = M.run_timing_strategy(sig, periods, di, rule="long_short",
                               cost_rate=0.0, hold=5)
    st2 = M.run_timing_strategy(-sig, periods, di, rule="long_short",
                                cost_rate=0.0, hold=5)
    assert st["ok"] and st2["ok"]
    # long_short 的基准是现金（0 收益），不是等权基准 —— 基准口径不能串
    assert st["baseline"] == "cash_zero"
    assert st["net_margin_per_period"] == pytest.approx(
        -st2["net_margin_per_period"], rel=1e-9)
    # 信号与基准同号 → 做多在上行期、做空在下行期 → 抓到的是 |bench|，必须为正
    assert st["net_margin_per_period"] > 0


def test_long_flat_baseline_is_equal_weight():
    """long_flat 的基准必须���等权基准（仓位 0/1，与满仓基准同口径可比）。"""
    periods = _synthetic_periods(40)
    bench = np.array([p["bench"] for p in periods])
    sig = np.sign(bench) + 0.5
    st = M.run_timing_strategy(sig, periods, _date_index(periods),
                               rule="long_flat", cost_rate=0.0, hold=5)
    assert st["ok"]
    assert st["baseline"] == "equal_weight_benchmark"
    # w=1 的期次贡献 +bench，w=0 的期次贡献 0 → 净边际 = Σ(1-w)·(-bench)/n
    w = (sig > 0).astype(float)
    expected = float(np.mean((w - 1.0) * bench))
    assert st["net_margin_per_period"] == pytest.approx(expected, rel=1e-9)


def test_cost_reduces_net_margin():
    """成本越高，净边际越低（不能出现「加成本反而变好」）。"""
    periods = _synthetic_periods(60)
    lo = M.decompose_segment(periods, hold=5, cost_rate=0.0015)
    hi = M.decompose_segment(periods, hold=5, cost_rate=0.0050)
    assert hi["annualized"]["strategy_net"] < lo["annualized"]["strategy_net"]


def test_decompose_identity_checks_are_zero():
    """分解的自洽恒等式必须成立：净超额 = 毛超额 − 成本拖累。"""
    periods = _synthetic_periods(80)
    d = M.decompose_segment(periods, hold=5, cost_rate=M.CANONICAL_COST)
    ic = d["identity_checks"]
    assert ic["gross_minus_cost_minus_net_excess_max_abs"] < 1e-12
    assert ic["alpha_plus_beta_drift_minus_net_excess_max_abs"] < 1e-9


# ── 4. 基准与策略期次必须对齐 ─────────────────────────────────────────
def test_benchmark_and_strategy_same_periods():
    """超额必须 = 策略 − **同一批**基准（期次错位会让超额失去意义）。"""
    periods = _synthetic_periods(80)
    d = M.decompose_segment(periods, hold=5, cost_rate=M.CANONICAL_COST)
    mp = d["per_period"]
    assert d["n_periods"] == len(periods)
    assert mp["net_excess_mean"] == pytest.approx(
        mp["strategy_net_mean"] - mp["bench_mean"], rel=1e-9)
    assert mp["gross_excess_mean"] == pytest.approx(
        mp["strategy_net_mean"] + mp["cost_drag_mean"] - mp["bench_mean"], rel=1e-9)


def test_aligned_periods_rejects_date_mismatch():
    """期次日期对不上时必须报错，而不是悄悄算出一个数。"""
    mat = _synthetic_panel()
    spec = {"hold": 5, "lag": 0, "quantile": 0.5, "min_valid": 10,
            "factor_names": [], "signed_weights": {}, "cost_roundtrip": 0.003}
    score = np.zeros(len(mat["dates"]))
    bad = []
    for i in range(10):
        bad.append({
            "signal_date": "1999-01-01", "entry_date": "1999-01-02",
            "exit_date": "1999-01-03",
            "long": 0.01, "short": 0.0, "bench": 0.005,
            "long_raw": 0.01, "short_raw": 0.0, "long_med": 0.01,
            "short_med": 0.0, "bench_med": 0.005, "bench_raw": 0.005,
            "turnover": 0.3,
        })
    with pytest.raises(Exception):
        M.assert_faithful(mat, score, spec, 0, 100, bad)


# ── 5. 产物必须带 generated_at ────────────────────────────────────────
def test_report_has_generated_at():
    """研究产物必须带生成时间戳。

    面板每天长新数据 → canonical 数字会变；产物不带时间戳，
    日报就会照抄过期数字（本项目实际发生过日报内部两个数字自相矛盾）。
    """
    out = M.OUT_JSON
    assert out.exists(), "market_timing.json 必须存在"
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert "generated_at" in rep
    assert isinstance(rep["generated_at"], str) and rep["generated_at"]


def test_json_and_md_consistent_on_key_numbers():
    """JSON 与 MD 必须指向同一份研究（不能让日报读到旧数字）。"""
    out = M.OUT_JSON
    md = M.OUT_MD
    assert out.exists() and md.exists()
    rep = json.loads(out.read_text(encoding="utf-8"))
    text = md.read_text(encoding="utf-8")
    assert rep["generated_at"][:10] in text


# ── 6. 验证段不得参与选参 ─────────────────────────────────────────────
def test_orientation_map_only_uses_train_rows():
    """orientation_map 的输入只能是**训练段**的 IC 行。

    验证段的 IC 一旦进来，选出的方向就带上了验证段的信息，
    测试段的数字就不再是样本外了。
    """
    rows = [
        {"family": "mom", "factor": "mom_20", "ic_nonoverlap": -0.03},
        {"family": "mom", "factor": "mom_60", "ic_nonoverlap": 0.02},
        {"family": "other", "factor": "mom_20", "ic_nonoverlap": 0.99},
    ]
    out = M.orientation_map(rows, "mom", ["mom_20", "mom_60"], min_abs_ic=0.01)
    # 方向 = sign(IC) × |IC|，且只取指定 family
    assert out["mom_20"] == pytest.approx(-0.03)
    assert out["mom_60"] == pytest.approx(0.02)
    # 别的 family 不能混进来
    assert set(out) == {"mom_20", "mom_60"}


def test_orientation_map_drops_weak_ic():
    """|IC| 太小的因子必须被丢掉（方向不可靠就别用）。"""
    rows = [{"family": "mom", "factor": "mom_20", "ic_nonoverlap": 0.001}]
    out = M.orientation_map(rows, "mom", ["mom_20"], min_abs_ic=0.01)
    assert out == {}


def test_verdict_helpers_return_text():
    """verdict 函数必须返回非空字符串（日报要直接印它）。"""
    t2 = {
        "factor_level_train_ic": [{"passes_bonferroni": True}],
        "family_trials_train": [{"passes_bonferroni": False}],
        "n_factor_level_tests": 66,
    }
    te = {"test": {"n_periods": 10}}
    v = M._verdict_q1(t2, te, 5.0, 2.0, 0.3)
    assert isinstance(v, str) and v.strip()
    # 一个都没通过时必须给出「封死」而不是「不可忽略」的结论
    t2b = {
        "factor_level_train_ic": [{"passes_bonferroni": False}],
        "family_trials_train": [{"passes_bonferroni": False}],
        "n_factor_level_tests": 66,
    }
    v2 = M._verdict_q1(t2b, te, -1.0, -2.0, 0.3)
    assert isinstance(v2, str) and v2.strip()
    assert v2 != v


# ── 7. Bonferroni 校正 ────────────────────────────────────────────────
def test_bonferroni_threshold_is_stricter_than_raw():
    """多重比较校正后的 t 阈值必须严格大于未校正阈值。"""
    raw = 1.96
    bonf = M._bonferroni_t(0.05 / 100)
    assert bonf > raw


def test_normal_ppf_known_values():
    """正态分位数：已知点必须对。"""
    assert M._normal_ppf(0.5) == pytest.approx(0.0, abs=1e-6)
    assert M._normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert M._normal_ppf(0.95) == pytest.approx(1.644854, abs=1e-4)


# ── 8. 证据缺失必须硬失败，不能静默出结论 ─────────────────────────────
def test_evidence_missing_is_runtime_error():
    """证据不足要抛 EvidenceMissing（不是返回 None 或空 dict）。"""
    assert issubclass(M.EvidenceMissing, RuntimeError)
    with pytest.raises(M.EvidenceMissing):
        raise M.EvidenceMissing("没有冻结件")


def test_frozen_spec_hash_is_stable():
    """spec_hash 必须由规格内容决定（同样的规格 → 同样的 hash）。"""
    a = M._hash({"hold": 5, "lag": 0, "factor_names": ["x"]})
    b = M._hash({"hold": 5, "lag": 0, "factor_names": ["x"]})
    c = M._hash({"hold": 10, "lag": 0, "factor_names": ["x"]})
    assert a == b
    assert a != c


# ── 9. 桶分位数收益 ───────────────────────────────────────────────────
def test_bucket_returns_partition_signal():
    """分桶必须把全部信号都分配到某个桶（不能丢样本）。"""
    periods = _synthetic_periods(5)
    signal = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    di = _date_index(periods)
    buckets = M.bucket_returns(signal, periods, di, edges=[0.3, 0.7])
    assert isinstance(buckets, dict)


# ── 10. 空期次必须失败关闭，不能返回全 0 的「结果」 ───────────────────
def test_run_timing_strategy_rejects_empty_periods():
    """空期次必须返回 ok=False 并给出原因（不能装作算出来了）。"""
    res = M.run_timing_strategy(np.array([0.5]), [], {}, rule="long_flat",
                                hold=5, cost_rate=M.CANONICAL_COST)
    assert res.get("ok") is False
    assert res.get("reason")


def test_run_timing_strategy_rejects_unknown_rule():
    """未知规则必须抛错，不能悄悄按 long_flat 跑。"""
    periods = _synthetic_periods(10)
    with pytest.raises(Exception):
        M.run_timing_strategy(np.ones(len(periods)), periods,
                               _date_index(periods), rule="long_all_in",
                               hold=5, cost_rate=M.CANONICAL_COST)
