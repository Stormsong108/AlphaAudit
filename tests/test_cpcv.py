# -*- coding: utf-8 -*-
"""CPCV（AFML 12.4）的正式回归测试。

锁定四类东西：

1. **原文数值**：切分数 C(N,k)（Eq.39）、路径数 φ[N,k]=(k/N)·C(N,k)=C(N−1,k−1)
   （Eq.40）、分组大小规则（12.4.1）、路径均值方差公式（Eq.43a/43b）。
2. **原文图**：Figure 12.1（哪些切分测哪些组）与 Figure 12.2（路径分配矩阵）
   的**逐格**一致 —— 数值取自 AFML PDF（州立图书馆数字馆藏）的一手解析，
   不是二手博客的转述。路径分配写错一个格子，CPCV 的整条路径就是错的，
   所以这里逐格锁死。
3. **语义不变量**：训练样本的信息集与**任一**测试组的信息集不得相交。
   这是 CPCV 特有的检查点 —— 按首尾一个连续区间检查会漏掉中间组两侧的泄漏。
4. **三套 purge 语义互不串味**：purged_train_bounds（前向链式）、
   PurgedKFold（对称单块）、CombinatorialPurgedCV（对称多组）。

跑法： cd market_daily && /opt/homebrew/bin/python3 -m pytest tests/test_cpcv.py -q
"""
from __future__ import annotations

import math
import os
import sys
from itertools import combinations

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import overfit_audit as M  # noqa: E402


# ── 一手来源：AFML Figure 12.1 / 12.2 ────────────────────────────────────
# Figure 12.1：哪些切分测哪些组（切分编号 1 基，字典序枚举 S1..S15）
FIG12_1 = {
    1: {1, 2}, 2: {1, 3}, 3: {1, 4}, 4: {1, 5}, 5: {1, 6},
    6: {2, 3}, 7: {2, 4}, 8: {2, 5}, 9: {2, 6},
    10: {3, 4}, 11: {3, 5}, 12: {3, 6},
    13: {4, 5}, 14: {4, 6}, 15: {5, 6},
}

# Figure 12.2：路径分配（路径 1..5 → 组 → 负责它的切分编号）
FIG12_2 = {
    1: {1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
    2: {1: 2, 2: 6, 3: 6, 4: 7, 5: 8, 6: 9},
    3: {1: 3, 2: 7, 3: 10, 4: 10, 5: 11, 6: 12},
    4: {1: 4, 2: 8, 3: 11, 4: 13, 5: 13, 6: 14},
    5: {1: 5, 2: 9, 3: 12, 4: 14, 5: 15, 6: 15},
}


# ── 1. 计数公式（Eq.39 / Eq.40）───────────────────────────────────────────
@pytest.mark.parametrize("n,k,expected_splits,expected_paths", [
    (6, 2, 15, 5),      # AFML Fig 12.1：S1..S15，右侧标注 5 条路径
    (8, 2, 28, 7),
    (10, 2, 45, 9),
    (12, 3, 220, 55),
    (16, 2, 120, 15),
    (20, 2, 190, 19),
    (10, 5, 252, 126),
])
def test_cpcv_counts_match_afml(n, k, expected_splits, expected_paths):
    """切分数 = C(N,k)，路径数 = C(N−1,k−1) = (k/N)·C(N,k)。"""
    assert M.cpcv_split_count(n, k) == expected_splits
    assert M.cpcv_path_count(n, k) == expected_paths
    # Eq.40 的两个写法必须相等
    assert M.cpcv_path_count(n, k) == pytest.approx(
        (k / n) * M.cpcv_split_count(n, k))
    # 恒等式 C(N,k)·k = φ·N
    assert M.cpcv_split_count(n, k) * k == M.cpcv_path_count(n, k) * n


def test_cpcv_path_count_degenerate_cases():
    """k=1 → φ=C(N−1,0)=1（退化成单一路径）；k=N → φ=1（无训练数据）。"""
    assert M.cpcv_path_count(10, 1) == 1
    assert M.cpcv_path_count(10, 10) == 1


def test_cpcv_rejects_bad_params():
    with pytest.raises(ValueError):
        M.cpcv_path_count(6, 0)
    with pytest.raises(ValueError):
        M.cpcv_path_count(6, 7)
    with pytest.raises(ValueError):
        M.cpcv_group_bounds(5, 6)
    with pytest.raises(ValueError):
        M.cpcv_group_bounds(10, 1)


# ── 2. 分组规则（12.4.1）─────────────────────────────────────────────────
def test_cpcv_group_bounds_follow_afml_rule():
    """第 1..N−1 组各 ⌊T/N⌋ 个，第 N 组吃余数。"""
    bounds = M.cpcv_group_bounds(100, 6)
    assert bounds[0] == (0, 15)
    assert bounds[1] == (16, 31)
    assert bounds[2] == (32, 47)
    assert bounds[3] == (48, 63)
    assert bounds[4] == (64, 79)
    assert bounds[-1] == (80, 99)          # 余数 100 − 16·5 = 20
    # 前 N−1 组等长，最后一组吃余数
    sizes = [hi - lo + 1 for lo, hi in bounds]
    assert sizes[:-1] == [16] * 5
    assert sizes[-1] == 20
    # 覆盖全部观测，无重叠无空隙
    assert bounds[0][0] == 0 and bounds[-1][1] == 99
    for (lo1, hi1), (lo2, _) in zip(bounds, bounds[1:]):
        assert hi1 + 1 == lo2


def test_cpcv_group_bounds_uneven_division():
    """T 不能被 N 整除时，余数全部进第 N 组。"""
    bounds = M.cpcv_group_bounds(10, 4)
    assert bounds == [(0, 1), (2, 3), (4, 5), (6, 9)]
    assert sum(hi - lo + 1 for lo, hi in bounds) == 10


# ── 3. 原文图的逐格一致性（Figure 12.1 / 12.2）────────────────────────────
def test_cpcv_path_assignment_matches_afml_figure_12_2_cell_by_cell():
    """路径分配矩阵必须与 AFML Figure 12.2 逐格一致。"""
    assign = M.cpcv_path_assignment(6, 2)      # 0-based
    assert len(assign) == 5                   # φ[6,2] = 5
    for p in range(5):
        for g in range(6):
            got = assign[p][g] + 1            # 转 1 基切分编号
            assert got == FIG12_2[p + 1][g + 1], (
                f"路径{p+1} 组{g+1}: 实现={got} 原文={FIG12_2[p+1][g+1]}")


def test_cpcv_path_assignment_only_points_at_splits_containing_the_group():
    """Fig 12.2 的每个数字必须指向一个**确实含该组**的切分（Fig 12.1）。"""
    assign = M.cpcv_path_assignment(6, 2)
    for p in range(5):
        for g in range(6):
            split_no = assign[p][g] + 1
            assert (g + 1) in FIG12_1[split_no]


def test_cpcv_path_assignment_uses_every_split_exactly_k_times():
    """每个切分被恰好 k 条路径使用 —— 原文均匀分布性质的直接推论。"""
    from collections import Counter
    n, k = 6, 2
    assign = M.cpcv_path_assignment(n, k)
    cnt = Counter(assign[p][g] for p in range(len(assign)) for g in range(n))
    assert set(cnt.values()) == {k}
    assert sum(cnt.values()) == M.cpcv_path_count(n, k) * n


def test_cpcv_split_enumeration_is_lexicographic():
    """切分枚举序必须是组合的字典序 —— Fig 12.1 的 S1..S15 依赖这个顺序。"""
    splits = list(combinations(range(6), 2))
    assert splits[0] == (0, 1)
    assert splits[4] == (0, 5)
    assert splits[5] == (1, 2)
    assert splits[-1] == (4, 5)
    assert len(splits) == M.cpcv_split_count(6, 2)


# ── 4. 语义不变量：信息集不得相交 ─────────────────────────────────────────
@pytest.mark.parametrize("n_groups,k_test", [(6, 2), (5, 2), (4, 1), (6, 3)])
@pytest.mark.parametrize("hold,embargo", [(1, 0), (3, 2), (5, 1)])
def test_cpcv_no_information_leak(n_groups, k_test, hold, embargo):
    """训练样本的信息集不得与**任一**测试组的信息集相交。"""
    cv = M.CombinatorialPurgedCV(
        n_groups=n_groups, n_obs=120, k_test=k_test, hold=hold, embargo=embargo)
    aud = cv.audit()
    assert aud["violations"] == 0
    assert aud["ok"] is True
    assert aud["folds"] == M.cpcv_split_count(n_groups, k_test)


def test_cpcv_purges_on_both_sides_of_every_test_group():
    """中间测试组的两侧都要 purge —— 只按首尾连续区间做会漏中间组。"""
    n_obs, n_groups, k_test, hold = 120, 6, 2, 4
    cv = M.CombinatorialPurgedCV(
        n_groups=n_groups, n_obs=n_obs, k_test=k_test, hold=hold, embargo=0)
    for f in cv.split():
        for g in f.test_groups:
            lo, hi = cv.group_bounds[g]
            info_end = int(cv.label_end[cv._group_index(g)].max())
            # 与该组信息集 [lo, info_end] 相交的训练样本，一个都不许留
            overlap = (cv.label_end[f.train] >= lo) & (f.train <= info_end)
            assert int(overlap.sum()) == 0


def test_cpcv_train_excludes_test_and_is_complement_plus_purge():
    """训练集 ⊆ 全样本 − 测试集，且被 purge/embargo 掉的确实不在训练集。"""
    cv = M.CombinatorialPurgedCV(
        n_groups=6, n_obs=120, k_test=2, hold=3, embargo=2)
    allidx = set(range(120))
    for f in cv.split():
        assert not (set(f.train.tolist()) & set(f.test.tolist()))
        assert set(f.train.tolist()) | set(f.test.tolist()) <= allidx
        if f.purged.size:
            assert not (set(f.purged.tolist()) & set(f.train.tolist()))
        if f.embargoed.size:
            assert not (set(f.embargoed.tolist()) & set(f.train.tolist()))


def test_cpcv_embargo_applies_after_each_test_group():
    """每个测试组之后都要有 embargo 段。"""
    cv = M.CombinatorialPurgedCV(
        n_groups=6, n_obs=120, k_test=2, hold=1, embargo=3)
    for f in cv.split():
        for g in f.test_groups:
            _lo, hi = cv.group_bounds[g]
            after = set(range(hi + 1, min(hi + 4, 120)))
            assert after <= set(f.embargoed.tolist()) | set(f.test.tolist())


def test_cpcv_test_groups_may_be_non_adjacent():
    """CPCV 的测试组可以不相邻（这正是它与 PurgedKFold 的差别）。"""
    cv = M.CombinatorialPurgedCV(
        n_groups=6, n_obs=120, k_test=2, hold=1, embargo=0)
    combos = [f.test_groups for f in cv.split()]
    # 0-based 组号：(0,1) 相邻，(0,3) 不相邻
    assert (0, 3) in combos
    assert (0, 1) in combos
    assert (2, 4) in combos
    # 测试组确实可以不连续（把测试组对应的观测下标合起来看是否连续）
    def _combo_obs(c):
        out = set()
        for g in c:
            out |= set(cv._group_index(g).tolist())
        return out

    assert any(not _is_contiguous(_combo_obs(c)) for c in combos)


def test_cpcv_paths_cover_all_groups_exactly_once():
    """每条路径 = N 个组各一段，覆盖全样本。"""
    cv = M.CombinatorialPurgedCV(
        n_groups=6, n_obs=120, k_test=2, hold=3, embargo=1)
    paths = cv.paths()
    assert len(paths) == M.cpcv_path_count(6, 2)
    for p in paths:
        groups = [g for g, _ in p["segments"]]
        assert groups == list(range(6))          # 每组恰好一次，按时间序
        for _g, split_idx in p["segments"]:
            assert 0 <= split_idx < M.cpcv_split_count(6, 2)


# ── 5. Eq.43a / 43b：路径均值方差 ─────────────────────────────────────────
def test_cpcv_path_mean_variance_matches_eq43a():
    """σ²[μ] = φ⁻¹σ²(1 + (φ−1)ρ̄)，且 φ⁻¹σ² ≤ σ²[μ] < σ²。

    ρ̄ 由**路径逐期收益序列**的两两相关估出（不是由 Sharpe 标量序列估）。
    """
    rng = np.random.default_rng(20261115)
    phi, T = 9, 120
    # 构造有共同成分的 φ 条路径 → 路径间正相关
    common = rng.normal(0.0, 1.0, size=T)
    pr = np.array([common + rng.normal(0.0, 0.5, size=T) for _ in range(phi)])
    y = np.array([float(np.mean(r) / (np.std(r) + 1e-12)) for r in pr])

    r = M.cpcv_path_mean_variance(y, path_returns=pr)
    sigma2 = float(np.var(y, ddof=1))

    # 独立复算 ρ̄：标准化后取外积，再**除以 T**（z 每行单位方差 → z·z' = T·corr）
    z = ((pr - pr.mean(axis=1, keepdims=True))
         / pr.std(axis=1, ddof=0, keepdims=True))
    corr = (z @ z.T) / float(T)
    iu = np.triu_indices(phi, k=1)
    rho = float(np.mean(corr[iu]))

    # 结构性不变量：ρ̄ 必须落在 [-1, 1]，否则 Eq.43b 的上下界必然被破坏
    assert -1.0 <= rho <= 1.0

    assert r["sigma2_paths"] == pytest.approx(sigma2)
    assert r["rho_bar"] == pytest.approx(rho)
    assert r["var_of_mean"] == pytest.approx(
        (sigma2 / phi) * (1 + (phi - 1) * rho))
    # Eq.43b
    assert r["lower_bound"] <= r["var_of_mean"] <= r["upper_bound"]
    assert r["within_bounds"] is True
    assert r["degenerate"] is False
    assert r["rho_available"] is True


def test_cpcv_path_mean_variance_without_returns_is_unavailable_not_optimistic():
    """⚠️ 没有 path_returns 时**不得**默认 ρ̄=0。

    ρ̄=0 等于假设 φ 条路径完全独立，会把 var_of_mean 低估成 φ⁻¹σ² ——
    这是最乐观的假设，会系统性高估证据强度。正确做法是标为不可用。
    """
    rng = np.random.default_rng(7)
    y = rng.normal(0.4, 0.2, size=9)
    r = M.cpcv_path_mean_variance(y)
    assert r["rho_available"] is False
    assert np.isnan(r["rho_bar"])
    assert np.isnan(r["var_of_mean"])
    assert r["within_bounds"] is False
    # 上下界仍然如实给出（σ²/φ 和 σ²）
    assert r["lower_bound"] == pytest.approx(float(np.var(y, ddof=1)) / 9)
    assert r["upper_bound"] == pytest.approx(float(np.var(y, ddof=1)))


def test_cpcv_path_mean_variance_identical_paths_have_no_averaging_benefit():
    """路径完全相同（σ²=0）时，均值方差 = 0，不因多条路径而变小。

    ⚠️ 退化情形：corrcoef 在恒定序列上是 0/0 = NaN，
    实现必须显式处理，不能让 rho_bar=NaN 把 var_of_mean 污染成 NaN。
    """
    y = np.full(5, 0.3)
    r = M.cpcv_path_mean_variance(y)
    assert r["sigma2_paths"] == pytest.approx(0.0)
    assert r["var_of_mean"] == pytest.approx(0.0)
    assert r["degenerate"] is True
    assert not np.isnan(r["var_of_mean"])


def test_cpcv_path_mean_variance_rejects_mismatched_returns():
    with pytest.raises(ValueError):
        M.cpcv_path_mean_variance([0.1, 0.2, 0.3],
                                  path_returns=np.zeros((2, 10)))


def test_cpcv_path_mean_variance_requires_two_paths():
    with pytest.raises(ValueError):
        M.cpcv_path_mean_variance([0.3])


# ── 6. 三套 purge 语义不能串味 ───────────────────────────────────────────
def test_cpcv_is_not_purged_train_bounds_semantics():
    """purged_train_bounds 是前向链式；CPCV 必须是对称多组。

    取一个**中间**测试组做例子：此时训练集同时包含测试段之前和之后的样本。
    （首组/末组做例子会掩盖这个差别 —— 前向链式在那种情形下也只训后面。）
    """
    cv = M.CombinatorialPurgedCV(
        n_groups=6, n_obs=120, k_test=2, hold=3, embargo=0)
    mid = next(f for f in cv.split() if f.test_groups == (2, 3))
    assert mid.train.min() < mid.test.min()
    assert mid.train.max() > mid.test.max()

    # 前向链式：训练段整体位于下一段之前 → 训练集不可能越过测试段
    lo, hi = 0, 60
    cut_lo, cut_hi = M.purged_train_bounds(lo, hi, 60, hold=3, embargo=0)
    assert cut_hi <= hi
    assert not (cut_lo < 60 < cut_hi)


def test_cpcv_and_purged_kfold_differ_on_same_data():
    """同一份数据上，CPCV 与 PurgedKFold 的切分结构必须不同。"""
    cv = M.CombinatorialPurgedCV(
        n_groups=6, n_obs=120, k_test=2, hold=3, embargo=0)
    pk = M.PurgedKFold(n_splits=6, n_obs=120, hold=3, embargo=0)
    cpcv_tests = [set(f.test.tolist()) for f in cv.split()]
    pk_tests = [set(f.test.tolist()) for f in pk.split()]
    assert cpcv_tests != pk_tests
    # CPCV 的测试集可能不连续；PurgedKFold 的测试集必连续
    assert any(not _is_contiguous(t) for t in cpcv_tests)
    assert all(_is_contiguous(t) for t in pk_tests)


def _is_contiguous(s: set[int]) -> bool:
    if not s:
        return False
    ss = sorted(s)
    return ss == list(range(ss[0], ss[-1] + 1))


# ── 7. 通用健壮性 ────────────────────────────────────────────────────────
def test_cpcv_rejects_bad_constructor_params():
    with pytest.raises(ValueError):
        M.CombinatorialPurgedCV(n_groups=1, n_obs=100, k_test=1)
    with pytest.raises(ValueError):
        M.CombinatorialPurgedCV(n_groups=5, n_obs=100, k_test=6)
    with pytest.raises(ValueError):
        M.CombinatorialPurgedCV(n_groups=5, n_obs=100, k_test=2, embargo=-1)
    with pytest.raises(ValueError):
        M.CombinatorialPurgedCV(n_groups=5, n_obs=3, k_test=2)


def test_cpcv_audit_reports_shape():
    cv = M.CombinatorialPurgedCV(
        n_groups=6, n_obs=120, k_test=2, hold=3, embargo=1)
    aud = cv.audit()
    assert aud["groups"] == 6
    assert aud["k_test"] == 2
    assert aud["folds"] == 15
    assert aud["violations"] == 0
