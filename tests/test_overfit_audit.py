# -*- coding: utf-8 -*-
"""过拟合审计工具的正式回归测试。

锁定三类东西：
  1. **公式数值**：PSR / E[max SR] / DSR / MinTRL / PBO 的绝对值。
     这些值由独立实现与公开参考实现在多组参数下逐位对齐（|d| = 0.000e+00），
     这里硬编码作为回归锁 —— 任何人改动公式都会立刻被这条测试挡住。
  2. **语义不变量**：PurgedKFold 的核心性质是「训练样本的信息集与测试段
     信息集不得相交」。这是 AFML 7.4 的硬约束，**不能靠改断言来通过**。
  3. **踩过的坑**：np.where 的布尔掩码必须显式加 [:, None]。

跑法： cd market_daily && /opt/homebrew/bin/python3 -m pytest tests/test_overfit_audit.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import overfit_audit as M  # noqa: E402


# ── 1. PSR ───────────────────────────────────────────────────────────────
def test_psr_known_values():
    """PSR 绝对值回归锁（与公开参考实现逐位对齐）。"""
    assert M.probabilistic_sharpe_ratio(0.10, 0.0, 250, 0.0, 3.0) == pytest.approx(
        0.9422612667196991, abs=1e-15)
    assert M.probabilistic_sharpe_ratio(-0.08, 0.0, 60, 1.1, 8.0) == pytest.approx(
        0.2789014776893116, abs=1e-15)
    # 基准不为零时应低于「对零」情形
    assert (M.probabilistic_sharpe_ratio(0.10, 0.05, 250, 0.0, 3.0)
            < M.probabilistic_sharpe_ratio(0.10, 0.0, 250, 0.0, 3.0))


def test_psr_is_monotone_in_sharpe_and_sample_size():
    base = M.probabilistic_sharpe_ratio(0.05, 0.0, 200, 0.0, 3.0)
    assert M.probabilistic_sharpe_ratio(0.10, 0.0, 200, 0.0, 3.0) > base
    assert M.probabilistic_sharpe_ratio(0.05, 0.0, 800, 0.0, 3.0) > base


def test_psr_rejects_degenerate_input():
    assert math.isnan(M.probabilistic_sharpe_ratio(0.1, 0.0, 1, 0.0, 3.0))


# ── 2. E[max SR] ─────────────────────────────────────────────────────────
def test_expected_max_sharpe_known_value():
    # 0.157459830134575 —— 与 GPL-3.0 参考实现逐位对齐（|d|=0.000e+00）
    assert M.expected_max_sharpe(0.01, 10, full=False) == pytest.approx(
        0.157459830134575, abs=1e-15)


def test_expected_max_sharpe_has_no_kurtosis_term():
    """E[max SR] 只有 Eq.(1) 一个式子：偏度/峰度**不参与**。

    ⚠️ 历史 bug：早期版本多了一个 ``(g4-1)/4·√V·z1²`` 项，
    实测把 E[max SR] 撑到 3.77（而 36 个试验的实测 Sharpe 最大仅 0.12），
    把 DSR 压成 0 → 得出「彻底过拟合」的错误结论。
    这条测试把「峰度不影响 E[max SR]」锁死。

    峰度修正在 PSR 的**分母**里（``probabilistic_sharpe_ratio``），不在这里。
    """
    base = M.expected_max_sharpe(0.01, 36)
    for ku in (3.0, 8.36, 30.0):
        for sk in (-0.18, 0.0, 1.5):
            assert M.expected_max_sharpe(0.01, 36, skew=sk, kurtosis=ku) == pytest.approx(
                base, rel=1e-12)


def test_expected_max_sharpe_cannot_exceed_observed_trial_max():
    """不变量：E[max SR] 不应远大于实测 trial Sharpe 的最大值。

    这是抓「自相矛盾」最直接的判据：若 E[max SR] = 3.77 而所有 trial
    的实测 Sharpe 都 ≤ 0.12，说明方差口径或公式错了 —— 数学上不可能。
    """
    rng = np.random.default_rng(7)
    for n_trials in (36, 144):
        trials = rng.normal(0.0, 0.10, size=n_trials)   # 真实 Sharpe 都在 ±0.3 内
        v = float(np.var(trials, ddof=1))
        emax = M.expected_max_sharpe(v, n_trials)
        observed_max = float(np.max(trials))
        # E[max SR] 允许高于实测最大（那是「期望」，本来就可能超过某次实测），
        # 但不应高出一个数量级。
        assert emax < max(observed_max, 0.0) * 10 + 0.5, (
            f"E[max SR]={emax:.4f} 相对实测最大 {observed_max:.4f} 明显不合理")


def test_expected_max_sharpe_grows_with_trials_and_variance():
    a = M.expected_max_sharpe(0.01, 10, full=False)
    b = M.expected_max_sharpe(0.01, 100, full=False)
    c = M.expected_max_sharpe(0.04, 10, full=False)
    assert b > a          # 试验越多，期望最大 Sharpe 越高（惩罚越重）
    assert c > a          # 方差越大，惩罚越重
    assert M.expected_max_sharpe(0.01, 1, full=False) == 0.0


# ── 3. DSR ───────────────────────────────────────────────────────────────
def test_dsr_fallback_path_known_values():
    """不传 trial_sharpes 时的兜底路径（variance_source 必须是 fallback）。"""
    r = np.random.default_rng(5).normal(0.0004, 0.011, size=400)
    res = M.deflated_sharpe_ratio(r, n_trials=64, full_formula=False)
    assert res.variance_source == "fallback_single_series"
    assert res.deflation_benchmark_sr0 == pytest.approx(
        0.11872763205361316, abs=1e-14)
    assert res.deflated_sharpe_ratio == pytest.approx(
        0.0035349686326745645, abs=1e-14)


def test_dsr_known_values_seeded():
    for seed, psr, dsr in [
        (7, 0.06913296216900108, 1.1095425037616084e-07),
        (13, 0.9850756244395971, 0.05405681522962407),
        (29, 0.7952281370797994, 0.0037329516807787437),
    ]:
        r = np.random.default_rng(seed).normal(0.0006, 0.012, size=750)
        trials = list(np.random.default_rng(seed + 100).normal(0.0, 0.05, size=100))
        res = M.deflated_sharpe_ratio(r, n_trials=100, trial_sharpes=trials,
                                      full_formula=False)
        assert res.psr_vs_zero == pytest.approx(psr, abs=1e-14)
        assert res.deflated_sharpe_ratio == pytest.approx(dsr, abs=1e-14)
        assert res.variance_source == "trial_sharpes"


def test_dsr_more_trials_makes_harder_to_pass():
    """诚实计数铁律：试验次数变多 → SR0 抬高 → DSR 下降（更难通过）。"""
    r = np.random.default_rng(3).normal(0.0008, 0.010, size=500)
    trials = list(np.random.default_rng(11).normal(0.0, 0.05, size=200))
    few = M.deflated_sharpe_ratio(r, n_trials=10, trial_sharpes=trials, full_formula=False)
    many = M.deflated_sharpe_ratio(r, n_trials=200, trial_sharpes=trials, full_formula=False)
    assert many.deflation_benchmark_sr0 > few.deflation_benchmark_sr0
    assert many.deflated_sharpe_ratio < few.deflated_sharpe_ratio


def test_dsr_threshold_flag():
    r = np.random.default_rng(9).normal(0.002, 0.010, size=600)
    res = M.deflated_sharpe_ratio(r, n_trials=5, threshold=0.95)
    assert res.passed == (res.deflated_sharpe_ratio >= 0.95)


# ── 4. MinTRL ────────────────────────────────────────────────────────────
def test_min_track_record_length_known_values():
    for sr, bench, sk, ku, conf, expect in [
        (0.10, 0.0, 0.0, 3.0, 0.95, 272.90711713658885),
        (0.05, 0.03, -0.5, 6.0, 0.95, 6955.092159354611),
        (0.20, 0.10, 0.1, 4.0, 0.99, 547.6013375364881),
    ]:
        got = M.minimum_track_record_length(sr, bench, sk, ku, conf)
        assert got == pytest.approx(expect, rel=1e-12)


def test_min_trl_infinite_when_not_better_than_benchmark():
    assert M.minimum_track_record_length(0.05, 0.05, 0.0, 3.0, 0.95) == float("inf")


# ── 5. 多重检验阈值 ──────────────────────────────────────────────────────
def test_multiple_testing_threshold_ordering():
    """Holm / BHY 都是**逐步**方法：名次越靠后（策略越差），门槛越松。

    ⚠️ 之前这���测试写成 b > h > y 是错的，方向反了。实测（n=128, se=0.05）：

        bonferroni  恒为 0.167968（对所有名次一致）
        holm   rank1=0.167968 → rank128=0.082243（递减 = 门槛放松）
        bhy    rank1=0.190078 → rank128=0.117868（递减）

    Holm 第 1 档与 Bonferroni **相同**（都是 alpha/N）；
    BHY 第 1 档比 Bonferroni **更严**（alpha/(N·C(N))），
    到 rank ≈ C(N) ≈ 5.4 之后才反超并变松 —— 这是 BH step-up 的固有性质。
    """
    b = M.multiple_testing_threshold(0.05, 128, method="bonferroni")
    h1 = M.multiple_testing_threshold(0.05, 128, method="holm", rank=1)
    h2 = M.multiple_testing_threshold(0.05, 128, method="holm", rank=2)
    h128 = M.multiple_testing_threshold(0.05, 128, method="holm", rank=128)
    y1 = M.multiple_testing_threshold(0.05, 128, method="bhy", rank=1)
    y2 = M.multiple_testing_threshold(0.05, 128, method="bhy", rank=2)
    y128 = M.multiple_testing_threshold(0.05, 128, method="bhy", rank=128)

    # Bonferroni 对所有名次一致
    assert M.multiple_testing_threshold(0.05, 128, method="bonferroni", rank=64) == b
    # Holm：rank1 == Bonferroni；名次越靠后门槛越松（值递减）
    assert h1 == pytest.approx(b, rel=1e-12)
    assert h2 < h1
    assert h128 < h2
    # BHY：rank1 比 Bonferroni 更严；名次越靠后越松；rank10 起反超 Bonferroni
    assert y1 > b
    assert y2 < y1
    assert y128 < y2
    y10 = M.multiple_testing_threshold(0.05, 128, method="bhy", rank=10)
    assert y10 < b

    with pytest.raises(ValueError):
        M.multiple_testing_threshold(0.05, 128, method="unknown")
    with pytest.raises(ValueError):
        M.multiple_testing_threshold(0.05, 128, method="holm", rank=0)
    with pytest.raises(ValueError):
        M.multiple_testing_threshold(0.05, 128, method="holm", rank=129)


# ── 6. PBO（CSCV）────────────────────────────────────────────────────────
def test_pbo_known_values():
    for seed, T, N, S, pbo, n_splits, median_logit in [
        (1, 600, 20, 8, 0.24285714285714285, 70, 0.6931471805599452),
        (42, 800, 12, 10, 0.6071428571428571, 252, -0.15415067982725822),
        (99, 400, 30, 6, 0.55, 20, -0.4616297005951241),
    ]:
        Mx = np.random.default_rng(seed).normal(0, 1, size=(T, N))
        res = M.probability_of_backtest_overfitting(Mx, n_blocks=S)
        assert res.pbo == pytest.approx(pbo, abs=1e-14)
        assert res.n_splits == n_splits
        assert res.median_logit == pytest.approx(median_logit, abs=1e-14)
        assert res.overfit == (pbo > 0.5)


def test_pbo_null_anchor_is_median_half_over_many_draws():
    """纯噪声下 PBO 的**中位数**应≈0.5（理论锚点）。

    ⚠️ 不能用单次抽样做区间断言：实测单次 PBO 的 sd ≈ 0.21，
    单个种子的 PBO 落在 0.30~0.70 之外完全正常（那是抽样噪声，
    不是实现错误）。所以这里对**多次重复的中位数**断言，
    且容差按中位数的抽样标准误定（实测 120 次重复时 SE ≈ 0.024）。

    实测基线（/tmp/diag_overfit.py，120 次重复）：
        N=12  median=0.5071 (0.27 SE)
        N=20  median=0.5000 (0.00 SE)
        N=30  median=0.5143 (0.61 SE)
    """
    pooled = []
    for N in (12, 20, 30):
        v = np.array([
            M.probability_of_backtest_overfitting(
                np.random.default_rng(5000 + s).normal(0, 1, size=(600, N)),
                n_blocks=8).pbo
            for s in range(120)
        ])
        med = float(np.median(v))
        # 容差 = 4 × 中位数抽样 SE（1.2533·sd/√n），n=120 时 ≈ 4×0.024
        se = 1.2533 * float(v.std(ddof=1)) / np.sqrt(v.size)
        assert abs(med - 0.5) < 4.0 * se, (
            f"N={N} 的 PBO 中位数 {med:.4f} 偏离 0.5 超过 4 个抽样标准误 "
            f"(SE≈{se:.4f})")
        pooled.extend(v.tolist())
    a = np.array(pooled)
    med = float(np.median(a))
    se = 1.2533 * float(a.std(ddof=1)) / np.sqrt(a.size)
    assert abs(med - 0.5) < 4.0 * se, (
        f"合并后的 PBO 中位数 {med:.4f} 偏离 0.5 超过 4 个抽样标准误 (SE≈{se:.4f})")


def test_pbo_detects_a_genuinely_better_column():
    """单列有真实 edge 时 PBO 应显著低于 0.5 —— 证明 PBO 真的有判别力。

    反向对照：所有列同漂移（不存在「样本内最优」这一说）时 PBO ≈ 0.5，
    这是**正确行为**，不是 bug。
    """
    # edge = 0.0015/期、sd = 0.01 → 真实 Sharpe 约 0.15
    v = np.array([
        M.probability_of_backtest_overfitting(
            _with_edge(np.random.default_rng(7000 + s).normal(0, 0.01, size=(600, 20)),
                       col=0, edge=0.0015),
            n_blocks=8).pbo
        for s in range(40)
    ])
    assert float(np.median(v)) < 0.25, f"PBO 中位数 {np.median(v):.4f} 未识别出真实 edge"

    # 反向：全部列同漂移 → 没有最优列 → PBO ≈ 0.5
    v2 = np.array([
        M.probability_of_backtest_overfitting(
            np.random.default_rng(9000 + s).normal(0.0005, 0.01, size=(600, 20)),
            n_blocks=8).pbo
        for s in range(40)
    ])
    assert 0.40 < float(np.median(v2)) < 0.60


def _with_edge(Mx: np.ndarray, col: int, edge: float) -> np.ndarray:
    Mx = Mx.copy()
    Mx[:, col] += edge
    return Mx


def test_pbo_rejects_bad_shapes():
    with pytest.raises(ValueError):
        M.probability_of_backtest_overfitting(np.zeros((10, 1)), n_blocks=4)
    with pytest.raises(ValueError):
        M.probability_of_backtest_overfitting(np.zeros((10, 5)), n_blocks=5)   # 奇数块
    with pytest.raises(ValueError):
        M.probability_of_backtest_overfitting(np.zeros((10, 5)), n_blocks=20)  # 块 > T


# ── 7. PurgedKFold —— 语义不变量（不可通过放宽断言来通过）────────────────
@pytest.mark.parametrize("n,n_splits,hold,embargo", [
    (100, 5, 5, 0),
    (100, 5, 5, 5),
    (240, 6, 10, 4),
    (180, 4, 3, 18),
    (60, 3, 1, 0),
    (120, 5, 20, 1),
])
def test_purged_kfold_no_information_leak(n, n_splits, hold, embargo):
    """核心不变量：训练样本的信息集与测试段信息集**不得相交**。

    这是 AFML 7.4 的硬约束 —— 若为让测试通过而削弱它，等于放开门禁。
    """
    pkf = M.PurgedKFold(n_splits=n_splits, n_obs=n, hold=hold, embargo=embargo)
    folds = pkf.split()
    assert len(folds) == n_splits
    for f in folds:
        te_lo, te_hi = int(f.test[0]), int(f.test[-1])
        info_end = int(pkf.label_end[f.test].max())
        # 训练样本的信息集 [i, label_end[i]] 与 [te_lo, info_end] 必须不相交
        bad = (pkf.label_end[f.train] >= te_lo) & (f.train <= info_end)
        assert int(bad.sum()) == 0, f"fold{f.fold} 存在 {int(bad.sum())} 个信息集相交的训练样本"
        # 测试集与训练集不相交
        assert np.intersect1d(f.test, f.train).size == 0
        # embargo 段不得进训练集
        if f.embargoed.size:
            assert np.isin(f.embargoed, f.train).sum() == 0
    assert pkf.audit()["violations"] == 0


def test_purged_kfold_purges_on_both_sides():
    """purge 必须两侧都做：测试段**之后**、起始时间仍落在测试信息集内的样本也要剔。

    这正是本轮纠正的认知点 —— 只清尾端 = 泄漏仍在。
    """
    n, hold = 100, 5
    pkf = M.PurgedKFold(n_splits=5, n_obs=n, hold=hold)
    f = pkf.split()[2]                     # 中间折，两侧都有训练样本
    te_lo, te_hi = int(f.test[0]), int(f.test[-1])
    info_end = int(pkf.label_end[f.test].max())
    # 测试段之后、但起始 <= info_end 的样本，必须被 purge
    head_side = np.arange(te_hi + 1, info_end + 1)
    assert head_side.size > 0
    assert np.isin(head_side, f.train).sum() == 0
    assert np.isin(head_side, f.purged).all()


def test_purged_kfold_embargo_removes_immediately_after_test():
    pkf = M.PurgedKFold(n_splits=5, n_obs=100, hold=1, embargo=7)
    for f in pkf.split():
        te_hi = int(f.test[-1])
        expect = np.arange(te_hi + 1, min(te_hi + 1 + 7, 100))
        assert np.array_equal(f.embargoed, expect)


def test_purged_kfold_test_folds_are_contiguous_and_cover_all():
    n = 120
    pkf = M.PurgedKFold(n_splits=6, n_obs=n, hold=3)
    seen = []
    for f in pkf.split():
        d = np.diff(f.test)
        assert (d == 1).all(), "测试段必须是连续块（不洗牌）"
        seen.extend(f.test.tolist())
    assert sorted(seen) == list(range(n))


def test_purged_kfold_rejects_bad_params():
    with pytest.raises(ValueError):
        M.PurgedKFold(n_splits=1, n_obs=10)
    with pytest.raises(ValueError):
        M.PurgedKFold(n_splits=20, n_obs=10)
    with pytest.raises(ValueError):
        M.PurgedKFold(n_splits=2, n_obs=10, embargo=-1)
    # label_end[i] 必须 >= i（信息集不能倒流）
    with pytest.raises(ValueError):
        M.PurgedKFold(n_splits=2, n_obs=10, label_end=[0, 1, 2, 3, 4, 5, 6, 7, 0, 9])
    # label_end 长度必须等于 n_obs
    with pytest.raises(ValueError):
        M.PurgedKFold(n_splits=2, n_obs=10, label_end=[0, 1, 2])


# ── 8. 前向链式 helper（与 PurgedKFold 是不同用途，别混用）───────────────
def test_purged_train_bounds_forward_chaining():
    """前向链式：训练段整体在下一段之前，只需清尾端。"""
    lo, hi = M.purged_train_bounds(0, 99, 100, hold=5, embargo=0)
    assert (lo, hi) == (0, 94)          # 99-5=94，剔掉最后 5 期
    lo, hi = M.purged_train_bounds(0, 99, 100, hold=5, embargo=2)
    assert (lo, hi) == (0, 92)
    # 训练段太短时退化为空
    lo, hi = M.purged_train_bounds(0, 3, 4, hold=5)
    assert lo == hi


def test_segment_overlap_days():
    assert M.segment_overlap_days(100, 100, 5) == 0
    assert M.segment_overlap_days(100, 98, 5) > 0


def test_label_end_indices():
    assert np.array_equal(M.label_end_indices(4, 3), [2, 3, 4, 5])


# ── 9. 年化换算口径 ─────────────────────────────────────────────────────
def test_annualised_to_per_period():
    assert M.annualised_to_per_period(1.0, 252) == pytest.approx(1.0 / math.sqrt(252))
    assert M.annualised_to_per_period(0.5, 252) == pytest.approx(
        M.annualised_to_per_period(1.0, 252) * 0.5)


def test_sharpe_per_period_matches_manual():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.013])
    assert M.sharpe_per_period(r) == pytest.approx(
        (r.mean() - 0.0) / r.std(ddof=1), rel=1e-14)


def test_skewness_kurtosis_on_normal_sample():
    r = np.random.default_rng(1).normal(0, 1, size=200000)
    sk, ku = M.skewness_kurtosis(r)
    assert abs(sk) < 0.02
    assert abs(ku - 3.0) < 0.05


# ── 10. 踩过的坑：np.where 布尔掩码必须显式 [:, None] ───────────────────
def test_where_broadcast_requires_explicit_none_axis():
    """条件形状 (M,) 对上值形状 (M, N) → np.where 会广播到最后一轴而报错。

    正确写法是显式 [:, None] 保持 (M, 1) 逐行广播。
    """
    M_, N_ = 5, 3
    cond = np.array([True, False, True, True, False])
    vals = np.arange(M_ * N_, dtype=float).reshape(M_, N_)
    with pytest.raises(ValueError):
        np.where(cond, vals, np.nan)
    out = np.where(cond[:, None], vals, np.nan)
    assert out.shape == (M_, N_)
    assert np.isnan(out[1]).all()
    assert np.isnan(out[4]).all()
    assert not np.isnan(out[0]).any()
