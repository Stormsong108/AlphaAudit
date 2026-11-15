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
