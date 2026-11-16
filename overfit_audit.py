"""过拟合审计工具箱：PSR / DSR / E[max SR] / MinTRL / PBO(CSCV) / Purged-Embargo。

本模块是**独立实现**，公式来自公开发表的学术论文（不是从任何第三方代码库拷贝）：

  * Bailey, D. H. & Lopez de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting and Non-Normality." Journal of Portfolio Management, 40(5).
  * Bailey, D. H. & Lopez de Prado, M. (2012).
    "The Sharpe Ratio Efficient Frontier." Journal of Risk, 15(2).
  * Bailey, D. H., Borwein, J., Lopez de Prado, M. & Zhu, Q. J. (2017).
    "The Probability of Backtest Overfitting." Journal of Computational
    Finance, 20(4), 39-69.
  * Lopez de Prado, M. (2018). "Advances in Financial Machine Learning",
    Chapter 7 (Cross-Validation in Finance).

口径铁律（踩过的坑写在这里）：
  1. 本模块所有 Sharpe 都是**每期（per-period）**单位，不是年化。
     把年化 Sharpe 传进来，概率公式会算错（se 被放大 √252 倍）。
     要年化请用 `annualised_to_per_period()` 先换算。
  2. `deflated_sharpe_ratio` 的 `n_trials` 必须是**诚实的多次试验数**：
     每一个参数网格点、每一个因子变体、每一次口径切换都要算进去。
     少报 = 把过拟合当成不存在。
  3. `expected_max_sharpe` 的方差必须来自**真实的试验集合**。
     只给单条收益序列做兜底估计会把 DSR 变成宽松版本，本模块会显式标记
     `variance_source='fallback_single_series'`，调用方必须自己承担这个风险。
  4. PBO 要求 N（策略列数）足够大才有意义（本项目用 128 个网格点）。
     N=2 时 PBO 几乎必然是 0.5 或 0，无信息量。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm

__all__ = [
    "EULER_MASCHERONI",
    "sharpe_per_period",
    "skewness_kurtosis",
    "annualised_to_per_period",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "DeflatedSharpeResult",
    "deflated_sharpe_ratio",
    "minimum_track_record_length",
    "multiple_testing_threshold",
    "PBOResult",
    "probability_of_backtest_overfitting",
    "purged_train_bounds",
    "segment_overlap_days",
    "PurgedFold",
    "PurgedKFold",
    "CPCVSplit",
    "CombinatorialPurgedCV",
    "cpcv_group_bounds",
    "cpcv_split_count",
    "cpcv_path_count",
    "cpcv_path_assignment",
    "cpcv_path_mean_variance",
]

EULER_MASCHERONI = 0.5772156649015328606


# --------------------------------------------------------------------------- #
# 基本量
# --------------------------------------------------------------------------- #
def sharpe_per_period(returns: np.ndarray, benchmark: float = 0.0) -> float:
    """每期 Sharpe（ddof=1）。benchmark 是每期无风险/基准收益。"""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((r.mean() - benchmark) / sd)


def skewness_kurtosis(returns: np.ndarray) -> tuple[float, float]:
    """样本偏度 g1 与**非超额**峰度 g2（正态 = 3.0）。"""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n < 4:
        return 0.0, 3.0
    m = r.mean()
    s = r.std(ddof=0)
    if s == 0:
        return 0.0, 3.0
    z = (r - m) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


def annualised_to_per_period(sharpe_annual: float, periods_per_year: int = 252) -> float:
    """年化 Sharpe → 每期 Sharpe。"""
    return sharpe_annual / math.sqrt(periods_per_year)


# --------------------------------------------------------------------------- #
# PSR —— Probabilistic Sharpe Ratio
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """P[真实 Sharpe > benchmark_sr]，已对非正态（偏度/峰度）做修正。

    Sharpe 估计量的标准误（Lopez de Prado 2012）::

        se(SR) = sqrt( (1 - g1*SR + (g2-1)/4 * SR^2) / (n - 1) )

    其中 g1 = 偏度，g2 = 非超额峰度。SR 必须是**每期**单位。
    """
    if n_obs < 2 or math.isnan(observed_sr):
        return float("nan")
    denom = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2
    denom = max(denom, 1e-12)
    se = math.sqrt(denom / (n_obs - 1))
    z = (observed_sr - benchmark_sr) / se
    return float(norm.cdf(z))


# --------------------------------------------------------------------------- #
# E[max SR] —— 多次试验下的期望最大 Sharpe（DSR 的惩罚项）
# --------------------------------------------------------------------------- #
def expected_max_sharpe(
    sr_variance_across_trials: float,
    n_trials: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0,
    full: bool = True,
) -> float:
    """N 次独立试验中 Sharpe 最大值的期望（用作 DSR 的基准 SR0）。

    Bailey & Lopez de Prado (2014) 论文 Eq.(1)::

        E[max SR] ≈ sqrt(V) * [ (1-gamma)*Z^-1(1-1/N)
                                + gamma     *Z^-1(1-1/(N*e)) ]

    其中 gamma = 欧拉-马斯克若尼常数，V = **各试验 Sharpe 估计的方差**
    （论文原文："{SR} is the variance across the trials' estimated SR"）。

    ⚠️ 口径铁律（2026-11-15 实测纠错）：
    本函数**只有这一个式子**。偏度/峰度修正出现在 **PSR 的分母**
    （``probabilistic_sharpe_ratio`` 里的 ``1 - g1*SR + (g2-1)/4*SR^2``），
    **不在** E[max SR] 里。早期版本这里多了一个
    ``(g4-1)/4 * sqrt(V) * Z^-1(1-1/N)^2`` 项，导致：
      - 实测 E[max SR] = 3.77，而 36 个试验的实测 Sharpe 最大仅 0.12
        —— E[max SR] 比所有观测值还大 30 倍，**数学上不可能成立**；
      - DSR 被这个虚高的 SR0 压成 0，得出「彻底过拟合」的错误结论。

    ``skew`` / ``kurtosis`` / ``benchmark_sr`` / ``full`` 参数保留只为
    **向后兼容**（旧调用点不会因此崩溃），但**不参与计算**。
    """
    if n_trials < 1:
        return 0.0
    if n_trials == 1:
        return 0.0
    v = max(float(sr_variance_across_trials), 0.0)
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return float(math.sqrt(v) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


# --------------------------------------------------------------------------- #
# DSR —— Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
@dataclass
class DeflatedSharpeResult:
    observed_sr: float
    deflation_benchmark_sr0: float
    psr_vs_zero: float
    deflated_sharpe_ratio: float
    n_obs: int
    n_trials: int
    skew: float
    kurtosis: float
    sr_variance_across_trials: float
    variance_source: str
    threshold: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "observed_sr_per_period": self.observed_sr,
            "deflation_benchmark_sr0": self.deflation_benchmark_sr0,
            "psr_vs_zero": self.psr_vs_zero,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "n_obs": self.n_obs,
            "n_trials": self.n_trials,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "sr_variance_across_trials": self.sr_variance_across_trials,
            "variance_source": self.variance_source,
            "threshold": self.threshold,
            "passed": self.passed,
        }


def deflated_sharpe_ratio(
    strategy_returns: np.ndarray,
    n_trials: int,
    *,
    trial_sharpes: Sequence[float] | None = None,
    sr_variance_across_trials: float | None = None,
    benchmark_sr: float = 0.0,
    threshold: float = 0.95,
    full_formula: bool = True,
) -> DeflatedSharpeResult:
    """对**被选中的**最优策略计算 DSR。

    参数
    ----
    strategy_returns : 被选中策略的每期收益
    n_trials         : 选它之前试过的**全部**独立配置数（诚实计数！）
    trial_sharpes    : 所有试验的每期 Sharpe（用来估计跨试验方差）
    sr_variance_across_trials : 直接给定跨试验方差（优先于 trial_sharpes）
    benchmark_sr     : 基准每期 Sharpe（0 = 只问"是不是正 alpha"）
    threshold        : DSR 通过线（默认 0.95）

    没有给 trial_sharpes 也没给方差时，会退化到单序列方差并标记
    `variance_source='fallback_single_series'` —— 那是个**宽松下界**，
    调用方必须自己判断能不能接受。
    """
    r = np.asarray(strategy_returns, dtype=float)
    r = r[~np.isnan(r)]
    n = int(r.size)
    sr = sharpe_per_period(r, benchmark_sr)
    skew, kurt = skewness_kurtosis(r)

    variance_source = "explicit"
    if sr_variance_across_trials is None:
        if trial_sharpes is not None and len(trial_sharpes) > 1:
            ts = np.asarray(list(trial_sharpes), dtype=float)
            ts = ts[~np.isnan(ts)]
            sr_variance_across_trials = float(np.var(ts, ddof=1)) if ts.size > 1 else 0.0
            variance_source = "trial_sharpes"
        else:
            denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
            sr_variance_across_trials = max(denom, 1e-12) / max(n - 1, 1)
            variance_source = "fallback_single_series"

    sr0 = expected_max_sharpe(
        sr_variance_across_trials, n_trials,
        skew=skew, kurtosis=kurt, benchmark_sr=benchmark_sr, full=full_formula,
    )
    psr0 = probabilistic_sharpe_ratio(sr, sr0, n, skew, kurt)
    psr_zero = probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt)

    return DeflatedSharpeResult(
        observed_sr=sr,
        deflation_benchmark_sr0=sr0,
        psr_vs_zero=psr_zero,
        deflated_sharpe_ratio=psr0,
        n_obs=n,
        n_trials=int(n_trials),
        skew=skew,
        kurtosis=kurt,
        sr_variance_across_trials=float(sr_variance_across_trials),
        variance_source=variance_source,
        threshold=threshold,
        passed=bool(psr0 >= threshold),
    )


def minimum_track_record_length(
    observed_sr: float,
    benchmark_sr: float,
    skew: float,
    kurtosis: float,
    confidence: float = 0.95,
) -> float:
    """要证明 observed_sr 显著优于 benchmark_sr，至少需要多少个观测期。

    Bailey & Lopez de Prado (2012)::

        MinTRL = 1 + (1 - g1*SR + (g2-1)/4*SR^2) * ( Z_alpha / (SR - SR*) )^2
    """
    if observed_sr <= benchmark_sr:
        return float("inf")
    z = float(norm.ppf(confidence))
    num = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2
    return float(1.0 + max(num, 1e-12) * (z / (observed_sr - benchmark_sr)) ** 2)


def multiple_testing_threshold(
    sr_standard_error: float,
    n_trials: int,
    alpha: float = 0.05,
    method: str = "bonferroni",
    rank: int = 1,
) -> float:
    """多重检验修正后，选中策略每期 Sharpe 至少要达到的水平。

    * ``bonferroni``：单侧显著性 alpha/N，最保守，对所有排名一致。
    * ``holm``：逐步降序（step-down）。第 ``rank`` 名的门槛是
      ``alpha / (N - rank + 1)`` —— 排名第 1 时与 Bonferroni **相同**
      （Holm 的第一档就是 alpha/N），名次越靠后门槛越松。
    * ``bhy``：Benjamini-Hochberg-Yekutieli，控制 FDR。最松。

    参数
    ----
    rank : 选中策略在 N 次试验中的名次（1 = 最好/被选中的那个）。

    返回值是**每期 Sharpe 的门槛**（不是概率）。
    """
    if n_trials < 1:
        return 0.0
    if rank < 1 or rank > n_trials:
        raise ValueError(f"rank 必须在 1..{n_trials} 之间")
    se = float(sr_standard_error)
    if method == "bonferroni":
        z = float(norm.ppf(1.0 - alpha / n_trials))
    elif method == "holm":
        z = float(norm.ppf(1.0 - alpha / (n_trials - rank + 1)))
    elif method == "bhy":
        c_n = sum(1.0 / i for i in range(1, n_trials + 1))
        z = float(norm.ppf(1.0 - alpha * rank / (n_trials * c_n)))
    else:
        raise ValueError(f"未知的多次检验方法: {method}")
    return z * se


# --------------------------------------------------------------------------- #
# PBO —— Probability of Backtest Overfitting（CSCV）
# --------------------------------------------------------------------------- #
def _sharpe_columns(block: np.ndarray) -> np.ndarray:
    """(T, N) 区块逐列的每期 Sharpe。"""
    mu = np.nanmean(block, axis=0)
    sd = np.nanstd(block, axis=0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    return mu / sd


@dataclass
class PBOResult:
    pbo: float
    n_splits: int
    n_strategies: int
    n_blocks: int
    logits: list[float] = field(default_factory=list)
    oos_ranks: list[float] = field(default_factory=list)
    is_best_oos_perf: list[float] = field(default_factory=list)
    median_logit: float = float("nan")
    overfit: bool = False

    def to_dict(self) -> dict:
        return {
            "pbo": self.pbo,
            "n_splits": self.n_splits,
            "n_strategies": self.n_strategies,
            "n_blocks": self.n_blocks,
            "median_logit": self.median_logit,
            "overfit": self.overfit,
        }


def probability_of_backtest_overfitting(
    perf_matrix: np.ndarray,
    n_blocks: int = 16,
    metric: Callable[[np.ndarray], np.ndarray] = _sharpe_columns,
    overfit_threshold: float = 0.5,
) -> PBOResult:
    """CSCV 估计「样本内最优配置在样本外还剩多少」的过拟合概率。

    参数
    ----
    perf_matrix : (T, N) 每期收益矩阵。行=时间，列=竞争配置。
    n_blocks    : 连续分块数 S（必须为偶数）。S=16 → C(16,8)=12870 个切分。
    metric      : 区块 → N 个绩效分数。默认每期 Sharpe。

    做法：把 T 行切成 S 个连续块，任取 S/2 块作样本内、其余作样本外；
    找样本内最好的配置 n*，算它在样本外所有配置中的相对排名 w，
    映射成 logit = ln(w/(1-w))；PBO = logit ≤ 0 的切分占比。

    PBO 高 = "样本内看起来最好的那个，出了样本基本就是随机水平" → 过拟合。
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("perf_matrix 必须是二维 (T x N)")
    T, N = M.shape
    if N < 2:
        raise ValueError("至少要 2 个配置才能评估 PBO")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks (S) 必须为偶数")
    if n_blocks > T:
        raise ValueError("n_blocks 不能超过观测期数")

    block_rows = np.array_split(np.arange(T), n_blocks)
    logits: list[float] = []
    oos_ranks: list[float] = []
    is_best_oos: list[float] = []

    for is_blocks in combinations(range(n_blocks), n_blocks // 2):
        is_set = set(is_blocks)
        is_rows = np.concatenate([block_rows[b] for b in range(n_blocks) if b in is_set])
        oos_rows = np.concatenate([block_rows[b] for b in range(n_blocks) if b not in is_set])

        is_perf = metric(M[is_rows])
        oos_perf = metric(M[oos_rows])
        if np.all(np.isnan(is_perf)):
            continue
        n_star = int(np.nanargmax(is_perf))

        valid = ~np.isnan(oos_perf)
        n_valid = int(valid.sum())
        if n_valid < 2 or np.isnan(oos_perf[n_star]):
            continue
        # 相对排名 w ∈ (0,1)：IS 最优者比多少比例的策略 OOS 表现差
        beaten = float(np.sum(oos_perf[valid] <= oos_perf[n_star]))
        w = beaten / (n_valid + 1.0)
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(w / (1.0 - w))))
        oos_ranks.append(w)
        is_best_oos.append(float(oos_perf[n_star]))

    pbo = float(np.mean([1.0 if lam <= 0 else 0.0 for lam in logits])) if logits else float("nan")
    med = float(np.median(logits)) if logits else float("nan")
    return PBOResult(
        pbo=pbo,
        n_splits=len(logits),
        n_strategies=int(N),
        n_blocks=int(n_blocks),
        logits=logits,
        oos_ranks=oos_ranks,
        is_best_oos_perf=is_best_oos,
        median_logit=med,
        overfit=bool(pbo > overfit_threshold),
    )


# --------------------------------------------------------------------------- #
# Purged / Embargo —— 时间序列分段防泄漏
# --------------------------------------------------------------------------- #
def segment_overlap_days(
    train_hi: int,
    next_lo: int,
    hold: int,
) -> int:
    """训练段末端有多少天的持仓窗口会伸进下一段。

    信号在 close[t] 算出、open[t+1] 成交、持有 hold 天 → 训练段最后一期
    的持仓窗口延伸到 train_hi + hold - 1。若 next_lo <= train_hi + hold - 1，
    就有重叠。
    """
    if next_lo >= train_hi:
        return 0
    end = train_hi + hold - 1
    if end < next_lo:
        return 0
    return int(end - next_lo + 1)


def purged_train_bounds(
    train_lo: int,
    train_hi: int,
    next_lo: int,
    hold: int,
    embargo: int = 0,
) -> tuple[int, int]:
    """把训练段末端 purge 掉，返回可安全使用的 (lo, hi)。

    purge 规则：训练段最后 ``hold`` 期，其持仓窗口会越过 train_hi 进入
    下一段，必须剔除。embargo 额外再往后砍 ``embargo`` 期（序列相关导致
    的前向泄漏）。

    这是本项目「训练段选参 → 验证段只确认」链条里防止**边界泄漏**的
    硬约束：宁可少用一点训练数据，也不能让验证期的信息回流到训练期。
    """
    cut = train_hi - hold - embargo
    if cut <= train_lo:
        return train_lo, train_lo
    return train_lo, cut


# --------------------------------------------------------------------------- #
# Purged K-Fold —— AFML 7.4 的「信息集相交」语义
# --------------------------------------------------------------------------- #
@dataclass
class PurgedFold:
    """一次 purged k-fold 切分的全部信息（可审计，不只是 train/test）。"""

    fold: int
    train: np.ndarray
    test: np.ndarray
    purged: np.ndarray
    embargoed: np.ndarray

    def to_dict(self) -> dict:
        return {
            "fold": int(self.fold),
            "n_train": int(self.train.size),
            "n_test": int(self.test.size),
            "n_purged": int(self.purged.size),
            "n_embargoed": int(self.embargoed.size),
        }


def label_end_indices(n_obs: int, hold: int) -> np.ndarray:
    """第 i 个样本的标签信息集终点（闭区间右端）。

    信息集 = ``[i, i + hold - 1]``：第 i 期建仓、持有 hold 期，
    标签在第 ``i + hold - 1`` 期才最终确定。
    """
    if hold < 1:
        raise ValueError("hold 必须 >= 1")
    return np.arange(int(n_obs)) + int(hold) - 1


class PurgedKFold:
    """AFML 7.4 的 Purged K-Fold：purge（信息集相交）+ embargo（尾部再剔除）。

    ⚠️ 与 :func:`purged_train_bounds` 的区别，**别混用**：

    * ``purged_train_bounds`` 是**前向链式**切分 —— 训练段整体位于下一段
      之前，只需清理训练段**尾端**（持仓窗口伸进下一段的那几期）。
    * ``PurgedKFold`` 是**对称 K 折** —— 训练集 = 测试集之外**全部**样本，
      于是 purge 必须在**两侧**都做：
        1. 尾部：训练集中标签窗口伸进测试段的样本；
        2. 首部：测试段之后、但起始时间仍落在测试段标签窗口内的样本。
      只做尾部那一侧 = 泄漏仍在（AFML 7.4 明确要求 purge 是信息集相交，
      不是按索引方向一刀切）。

    参数
    ----
    n_splits  : 折数（>= 2）
    n_obs     : 样本总数
    hold      : 标签持有期（期数），信息集 = [i, i+hold-1]
    embargo   : 测试段**之后**额外剔除的样本数（期）
    label_end : 可选的显式标签终点数组（闭区间右端）；给了则覆盖 hold
    """

    def __init__(self, n_splits: int, n_obs: int, hold: int = 1,
                 embargo: int = 0, label_end: Sequence[int] | None = None):
        self.n_splits = int(n_splits)
        self.n_obs = int(n_obs)
        self.embargo = int(embargo)
        if self.n_splits < 2:
            raise ValueError("n_splits 必须 >= 2")
        if self.n_splits > self.n_obs:
            raise ValueError("n_splits 不能超过样本总数")
        if self.embargo < 0:
            raise ValueError("embargo 不能为负")
        if label_end is None:
            self.label_end = label_end_indices(self.n_obs, hold)
        else:
            le = np.asarray(label_end, dtype=int)
            if le.shape != (self.n_obs,):
                raise ValueError("label_end 长度必须等于 n_obs")
            if np.any(le < np.arange(self.n_obs)):
                raise ValueError("label_end[i] 必须 >= i")
            self.label_end = le

    def split(self) -> "list[PurgedFold]":
        """生成全部折。测试段是连续块（不洗牌），与 AFML 12.4 的分组一致。"""
        idx = np.arange(self.n_obs)
        folds: list[PurgedFold] = []
        for k, block in enumerate(np.array_split(idx, self.n_splits)):
            test = np.asarray(block, dtype=int)
            test_lo, test_hi = int(test[0]), int(test[-1])
            info_end = int(self.label_end[test].max())

            keep = np.ones(self.n_obs, dtype=bool)
            keep[test] = False

            # PURGE：训练样本的信息集 [i, label_end[i]] 与测试段信息集
            # [test_lo, info_end] 有交集 → 剔除（**两侧都剔**）。
            overlap = (self.label_end >= test_lo) & (idx <= info_end)
            purged = idx[overlap & keep]
            keep &= ~overlap

            # EMBARGO：测试段之后紧邻的样本（抵消序列相关的前向泄漏）。
            e_hi = min(test_hi + 1 + self.embargo, self.n_obs)
            emb_range = np.arange(test_hi + 1, e_hi)
            embargoed = emb_range[keep[emb_range]] if emb_range.size else emb_range
            keep[emb_range] = False

            folds.append(PurgedFold(
                fold=k, train=idx[keep], test=test,
                purged=purged, embargoed=embargoed,
            ))
        return folds

    def audit(self) -> dict:
        """自检：任何一折的训练集都不得与测试段信息集相交。

        返回违规数；**必须为 0** 才算这个切分器可用。
        """
        violations = 0
        for f in self.split():
            if f.train.size == 0:
                violations += 1
                continue
            te_lo, te_hi = int(f.test[0]), int(f.test[-1])
            info_end = int(self.label_end[f.test].max())
            # 训练样本的信息集与测试段信息集相交 → 泄漏
            bad = (self.label_end[f.train] >= te_lo) & (f.train <= info_end)
            violations += int(bad.sum())
            # embargo 段不得出现在训练集
            if f.embargoed.size:
                violations += int(np.isin(f.embargoed, f.train).sum())
        return {"folds": self.n_splits, "violations": int(violations),
                "ok": violations == 0}


# --------------------------------------------------------------------------- #
# CPCV —— AFML 12.4 的「组合式净化交叉验证」
# --------------------------------------------------------------------------- #
# 来源（公开书籍原文，非二手转述）：
#   López de Prado, M. (2018). Advances in Financial Machine Learning.
#   Wiley. 第 12 章「Backtesting through Cross-Validation」，12.4 节
#   「The Combinatorial Purged Cross-Validation Method」。
#   本项目的实现按该章的公式与图独立完成，未复制任何第三方代码。
#
# 原文关键内容（用于对照，不入库正文）：
#   Eq.39  切分数        C(N,k) = ∏_{i=0}^{k-1}(N-i) / k!
#   Eq.40  路径数        φ[N,k] = (k/N)·C(N,k) = C(N-1,k-1)
#   12.4.1 分组          T 个观测分成 N 组，不打乱；第 1..N-1 组各 ⌊T/N⌋ 个，
#                        第 N 组吃余数 T − ⌊T/N⌋(N−1)
#   Fig.12.1/12.2 路径分配 见下 cpcv_path_assignment 的注释
#   Eq.43a 路径均值方差  σ²[μ_i] = φ⁻¹σ_i²(1 + (φ−1)ρ̄_i)
#   Eq.43b 上下界        φ⁻¹σ_i² ≤ σ²[μ_i] < σ_i²
#
# ⚠️ 与本项目另两套 purge 语义的差别（**三者不能混用**）：
#   * purged_train_bounds —— 前向链式，训练段整体在下一段之前，只清训练段尾端。
#   * PurgedKFold         —— 对称 K 折，单个连续测试块，purge 在两侧做。
#   * CombinatorialPurgedCV —— 对称但测试集由 **k 个不相邻的组** 组成
#     （k 组可以从任意位置挑），训练集 = 其余 N−k 组。此时 purge 必须
#     **逐组**做：与任一测试组信息集相交的训练样本都要剔掉。
#     只按「首尾一个连续区间」做 purge 会漏掉中间测试组两侧的泄漏。
#
# 三套语义共同的不变量：**训练样本的信息集与测试段信息集不得相交**。
# CPCV 的 audit() 检查的就是这一条。

def cpcv_group_bounds(n_obs: int, n_groups: int) -> list[tuple[int, int]]:
    """AFML 12.4.1 的分组：T 个观测分成 N 组，不打乱。

    第 1..N-1 组各 ⌊T/N⌋ 个，第 N 组吃余数 T − ⌊T/N⌋(N−1)。
    返回 [(lo, hi), ...]，闭区间，每组非空。
    """
    n_obs = int(n_obs)
    n_groups = int(n_groups)
    if n_groups < 2:
        raise ValueError("n_groups 必须 >= 2")
    if n_obs < n_groups:
        raise ValueError("n_obs 不能小于 n_groups")
    base = n_obs // n_groups
    bounds: list[tuple[int, int]] = []
    start = 0
    for g in range(n_groups):
        size = base if g < n_groups - 1 else n_obs - base * (n_groups - 1)
        bounds.append((start, start + size - 1))
        start += size
    return bounds


def cpcv_split_count(n_groups: int, k_test: int) -> int:
    """AFML Eq.39：k 组大小测试集的切分数量 C(N,k)。"""
    return int(math.comb(int(n_groups), int(k_test)))


def cpcv_path_count(n_groups: int, k_test: int) -> int:
    """AFML Eq.40：φ[N,k] = (k/N)·C(N,k) = C(N−1,k−1)。

    原文推导：每种组合涉及 k 个被测组，被测组总数为 k·C(N,k)，
    且所有组合算完后被测组在 N 组上**均匀分布**，故路径数为
    φ[N,k] = (k/N)·C(N,k)。
    """
    n_groups = int(n_groups)
    k_test = int(k_test)
    if not 1 <= k_test <= n_groups:
        raise ValueError("k_test 必须在 1..n_groups")
    return int(math.comb(n_groups - 1, k_test - 1))


def cpcv_path_assignment(n_groups: int, k_test: int) -> list[list[int]]:
    """AFML Fig 12.1/12.2 的「测试组 → 路径」分配矩阵。

    返回 ``assign[path_idx][group_idx]`` = 该路径上这一组由哪个切分负责
    （切分编号 = 该切分的测试组元组在 ``combinations`` 枚举序中的下标）。

    原文的构造（Fig 12.2，N=6,k=2 的 5 条路径为例）：
      * 切分按组合的字典序枚举：S1=(G1,G2) S2=(G1,G3) … S15=(G5,G6)。
      * 每个组 Gi 恰好出现在 C(N−1,k−1) = φ 个切分里。
      * 按「Gi 参与的切分的枚举序」依次分配路径 1,2,…,φ。
        即 Fig 12.2 里 G1 在 S1→路径1、S2→路径2、S3→路径3、S4→路径4、S5→路径5；
        G2 在 S1→路径1、S6→路径2、S7→路径3、S8→路径4、S9→路径5。
      * 于是一条路径把 N 个组各取一段，拼成**一条覆盖全样本的收益路径**；
        每个切分被 φ 条路径中的 k 条使用（C(N,k)·k = φ·N 的恒等式）。

    这与 Fig 12.1 的「x 标记哪些切分测哪些组」互为逆运算：
    ``assign`` 给出的是 Fig 12.2 的数字矩阵。
    """
    n_groups = int(n_groups)
    k_test = int(k_test)
    if not 1 <= k_test <= n_groups:
        raise ValueError("k_test 必须在 1..n_groups")
    n_paths = cpcv_path_count(n_groups, k_test)
    splits = list(combinations(range(n_groups), k_test))

    # assign[p][g] = 负责组 g 在路径 p 上的切分下标
    assign = [[-1] * n_groups for _ in range(n_paths)]
    for g in range(n_groups):
        # 组 g 参与的切分，按枚举序
        g_splits = [i for i, s in enumerate(splits) if g in s]
        if len(g_splits) != n_paths:
            # 恒等式：组 g 参与 C(N−1,k−1) = φ 个切分
            raise AssertionError(
                f"组 {g} 参与 {len(g_splits)} 个切分，应为 φ={n_paths}")
        for p, si in enumerate(g_splits):
            assign[p][g] = si
    return assign


@dataclass
class CPCVSplit:
    """一次组合式切分的全部信息（可审计，不只是 train/test）。"""

    split: int
    test_groups: tuple[int, ...]
    train: np.ndarray
    test: np.ndarray
    purged: np.ndarray
    embargoed: np.ndarray

    def to_dict(self) -> dict:
        return {
            "split": int(self.split),
            "test_groups": [int(g) for g in self.test_groups],
            "n_train": int(self.train.size),
            "n_test": int(self.test.size),
            "n_purged": int(self.purged.size),
            "n_embargoed": int(self.embargoed.size),
        }


class CombinatorialPurgedCV:
    """AFML 12.4 的 CPCV：N 组 → C(N,k) 个组合切分 → φ[N,k] 条回测路径。

    语义与 :class:`PurgedKFold` 的关键差异：测试集由 **k 个不相邻的组**
    组成，训练集 = 其余 N−k 组。因此 purge 必须**逐测试组**做
    （与每个测试组信息集相交的训练样本都剔掉），embargo 也在每个
    测试组之后施加一次。

    参数
    ----
    n_groups : 组数 N（>= 2）
    n_obs    : 观测总数 T
    k_test   : 每折测试组数 k（1 <= k <= N）
    hold     : 标签持有期，��息集 = [i, i+hold−1]
    embargo  : 每个测试组之后额外剔除的观测数
    label_end : 可选的显式标签终点数组；给了则覆盖 hold
    """

    def __init__(self, n_groups: int, n_obs: int, k_test: int,
                 hold: int = 1, embargo: int = 0,
                 label_end: Sequence[int] | None = None):
        self.n_groups = int(n_groups)
        self.n_obs = int(n_obs)
        self.k_test = int(k_test)
        self.embargo = int(embargo)
        if self.n_groups < 2:
            raise ValueError("n_groups 必须 >= 2")
        if not 1 <= self.k_test <= self.n_groups:
            raise ValueError("k_test 必须在 1..n_groups")
        if self.n_obs < self.n_groups:
            raise ValueError("n_obs 不能小于 n_groups")
        if self.embargo < 0:
            raise ValueError("embargo 不能为负")
        if label_end is None:
            self.label_end = label_end_indices(self.n_obs, hold)
        else:
            le = np.asarray(label_end, dtype=int)
            if le.shape != (self.n_obs,):
                raise ValueError("label_end 长度必须等于 n_obs")
            if np.any(le < np.arange(self.n_obs)):
                raise ValueError("label_end[i] 必须 >= i")
            self.label_end = le
        self.group_bounds = cpcv_group_bounds(self.n_obs, self.n_groups)

    def _group_index(self, g: int) -> np.ndarray:
        lo, hi = self.group_bounds[g]
        return np.arange(lo, hi + 1, dtype=int)

    def split(self) -> "list[CPCVSplit]":
        """生成全部 C(N,k) 个组合切分。

        每个切分：测试集 = k 个组的并集；训练集 = 其余组；
        purge 按**每个测试组**分别做；embargo 在每个测试组之后各做一次。
        """
        idx = np.arange(self.n_obs, dtype=int)
        out: list[CPCVSplit] = []
        for si, combo in enumerate(combinations(range(self.n_groups), self.k_test)):
            test_groups = tuple(int(g) for g in combo)
            test_idx = np.concatenate([self._group_index(g) for g in test_groups])
            test_idx.sort()

            keep = np.ones(self.n_obs, dtype=bool)
            keep[test_idx] = False

            purged_parts: list[np.ndarray] = []
            embargo_parts: list[np.ndarray] = []
            for g in test_groups:
                gidx = self._group_index(g)
                g_lo, g_hi = int(gidx[0]), int(gidx[-1])
                # 该测试组的信息集 = [g_lo, max(label_end[组内])]
                info_end = int(self.label_end[gidx].max())

                # PURGE（逐测试组）：训练样本信息集 [i, label_end[i]] 与
                # 该测试组信息集 [g_lo, info_end] 相交 → 剔除。
                overlap = (self.label_end >= g_lo) & (idx <= info_end)
                purged_parts.append(idx[overlap & keep])
                keep &= ~overlap

                # EMBARGO：该测试组之后紧邻的观测（抵消序列相关的前向泄漏）。
                e_hi = min(g_hi + 1 + self.embargo, self.n_obs)
                emb = np.arange(g_hi + 1, e_hi, dtype=int)
                embargo_parts.append(emb[keep[emb]] if emb.size else emb)
                keep[emb] = False

            purged = (np.concatenate(purged_parts) if purged_parts
                      else np.array([], dtype=int))
            embargoed = (np.concatenate(embargo_parts) if embargo_parts
                         else np.array([], dtype=int))
            purged = np.unique(purged)
            embargoed = np.unique(embargoed)

            out.append(CPCVSplit(
                split=si, test_groups=test_groups,
                train=idx[keep], test=test_idx,
                purged=purged, embargoed=embargoed,
            ))
        return out

    def paths(self) -> "list[dict]":
        """按 AFML Fig 12.2 组装 φ[N,k] 条回测路径。

        每条路径 = N 个组各取一段，按组的时间顺序拼接 → 覆盖全样本的一条
        收益序列。返回 [{"path": p, "segments": [(group, split_idx), ...]}]
        """
        assign = cpcv_path_assignment(self.n_groups, self.k_test)
        out: list[dict] = []
        for p, row in enumerate(assign):
            out.append({
                "path": p,
                "segments": [(g, int(row[g])) for g in range(self.n_groups)],
            })
        return out

    def audit(self) -> dict:
        """自检：任何一折的训练集都不得与**任一**测试组的信息集相交。

        这是 CPCV 特有的检查点：若只按首尾一个连续区间做检查，
        中间测试组两侧的泄漏会被漏掉。
        """
        violations = 0
        for f in self.split():
            if f.train.size == 0:
                violations += 1
                continue
            for g in f.test_groups:
                gidx = self._group_index(g)
                g_lo = int(gidx[0])
                info_end = int(self.label_end[gidx].max())
                bad = (self.label_end[f.train] >= g_lo) & (f.train <= info_end)
                violations += int(bad.sum())
            if f.embargoed.size:
                violations += int(np.isin(f.embargoed, f.train).sum())
        return {
            "folds": len(self.split()),
            "groups": self.n_groups,
            "k_test": self.k_test,
            "violations": int(violations),
            "ok": violations == 0,
        }


def cpcv_path_mean_variance(
    path_sharpes: Sequence[float],
    path_returns: np.ndarray | None = None,
) -> dict:
    """AFML Eq.43a/43b：路径 Sharpe 序列的均值方差。

    原文（§12.5 逐字对照）：

        "CPCV derives the distribution of Sharpe ratios from a large number of
         paths, j = 1, …, φ, with mean E[{y_i,j}] = μ_i and variance
         σ²[{y_i,j}] = σ_i². The variance of the sample mean of CPCV paths is

              σ²[μ_i] = φ⁻²(φσ_i² + φ(φ−1)σ_i²ρ̄_i)
                      = φ⁻¹σ_i²(1 + (φ−1)ρ̄_i)              (Eq.43a)

         where σ_i² is the variance of the Sharpe ratios across paths for
         strategy i, and ρ̄_i is the average off-diagonal correlation among
         {y_i,j}. CPCV leads to fewer false discoveries than CV and WF,
         because ρ̄_i < 1 implies that the variance of the sample mean is
         lower than the variance of the sample,

              φ⁻¹σ_i² ≤ σ²[μ_i] < σ_i²                        (Eq.43b)"

    ⚠️ **ρ̄_i 怎么估？** 这是最容易做错的一步。

    原文说 ρ̄_i 是 "{y_i,j} 之间的平均非对角相关"。但 y_i,j 是**样本路径 j 上
    策略 i 的 Sharpe** —— 一个由有限样本估出来的**标量**。单条路径只给出
    一个标量，**无法**从 φ 个标量算出它们之间的相关（那只是把 φ 个数排成
    序列做自相关，语义上完全不对）。

    可操作的口径是：y_i,j 背后的真实量是「在路径 j 的样本上策略 i 的真实
    Sharpe」。两条路径 j、j' 的 Sharpe 之间的相关，可以通过**两条路径的
    逐期收益序列的相关系数**来估计 —— 路径重叠得越多，其 Sharpe 估得越像，
    相关越高。因此：

      * 给了 ``path_returns``（形状 (φ, T) 的逐期收益矩阵）→
        ρ̄_i = 所有非对角「路径逐期收益相关系数」的均值。**这是推荐口径**。
      * 没给 ``path_returns`` → 无法估计 ρ̄_i。此时**不做任何假设**，
        返回 ρ̄_i = NaN，并把 var_of_mean 标为不可用；调用方必须自己决定
        怎么办。**绝不能**默认取 0（那等于假设各路径完全独立，会把
        var_of_mean 低估成 φ⁻¹σ²，是最乐观的假设）。

    参数
    ----
    path_sharpes : 长度 φ 的路径 Sharpe 序列
    path_returns : 可选的 (φ, T) 逐期收益矩阵，用于估 ρ̄_i

    返回
    ----
    dict，含 σ²[μ_i]、ρ̄_i、Eq.43b 的上下界，以及 var_of_mean 是否可用。
    """
    y = np.asarray(list(path_sharpes), dtype=float)
    phi = int(y.size)
    if phi < 2:
        raise ValueError("路径数必须 >= 2 才能算路径间相关性")
    sd = float(np.std(y, ddof=1))
    sigma2 = sd * sd

    # 退化情形：所有路径 Sharpe 相同 → σ²=0，均值方差也是 0。
    if sigma2 <= 0.0:
        return {
            "phi": phi,
            "sigma2_paths": 0.0,
            "rho_bar": float("nan"),
            "var_of_mean": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "degenerate": True,
            "rho_available": False,
            "within_bounds": True,
        }

    rho_bar = float("nan")
    if path_returns is not None:
        pr = np.asarray(path_returns, dtype=float)
        if pr.shape[0] != phi:
            raise ValueError(
                f"path_returns 行数 {pr.shape[0]} 必须等于路径数 {phi}")
        # 逐期收益的相关矩阵：corr(j,j') = 两条路径逐期收益的 Pearson 相关。
        # 只用**两两都有值**的期次（NaN 掩掉），否则含 NaN 会污染整行。
        pr = pr[:, ~np.isnan(pr).any(axis=0)]
        if pr.shape[1] >= 2:
            # ⚠️ keepdims=True 必须写：np.std(pr, axis=1) 返回 (φ,)，不 keepdims
            # 时 (φ,) 与 (φ,T) 广播失败（这正是 np.where 布尔掩码那类坑的同款）。
            mu = pr.mean(axis=1, keepdims=True)
            sds = pr.std(axis=1, ddof=0, keepdims=True)
            ok = sds[:, 0] > 0
            if int(ok.sum()) >= 2:
                z = np.where(ok[:, None], (pr - mu) / np.where(sds > 0, sds, 1.0), 0.0)
                # ⚠️ z @ z.T 是 **T 倍**的相关系数，不是相关系数本身：
                #   z 的每行零均值、单位方差（ddof=0 → Σ z² = T），
                #   所以 corr(i,j) = (1/T)·Σ_t z(i,t)·z(j,t)，必须除以 T。
                #   忘记除 T 会让 ρ̄ 变成几十上百，Eq.43b 的上界
                #   σ²[μ] < σ² 立刻被违反（这是本实现曾被回归测试抓到的真 bug）。
                corr = (z @ z.T) / float(pr.shape[1])
                iu = np.triu_indices(phi, k=1)
                rho_bar = float(np.mean(corr[iu])) if iu[0].size else float("nan")
                if not np.isfinite(rho_bar) or abs(rho_bar) > 1.0 + 1e-9:
                    raise ValueError(
                        f"ρ̄_i 估计越界：{rho_bar!r}（必须在 [-1, 1] 内）")

    if np.isnan(rho_bar):
        # 没有可用的 ρ̄ 估计 → var_of_mean 不可用。
        # 不默认取 0：那等于假设路径完全独立，会把不确定性低估到最优假设。
        return {
            "phi": phi,
            "sigma2_paths": sigma2,
            "rho_bar": float("nan"),
            "var_of_mean": float("nan"),
            "lower_bound": float(sigma2 / phi),
            "upper_bound": float(sigma2),
            "degenerate": False,
            "rho_available": False,
            "within_bounds": False,
            "note": (
                "缺少 path_returns，ρ̄_i 无法估计 —— var_of_mean 不可用。"
                "不默认取 ρ̄=0（那是最乐观假设）。"
            ),
        }

    var_mean = (sigma2 / phi) * (1.0 + (phi - 1) * rho_bar)
    return {
        "phi": phi,
        "sigma2_paths": sigma2,
        "rho_bar": rho_bar,
        "var_of_mean": float(var_mean),
        "lower_bound": float(sigma2 / phi),
        "upper_bound": float(sigma2),
        "degenerate": False,
        "rho_available": True,
        "within_bounds": bool(
            (sigma2 / phi) - 1e-12 <= var_mean <= sigma2 + 1e-12),
    }
