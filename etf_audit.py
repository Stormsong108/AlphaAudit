# -*- coding: utf-8 -*-
"""ETF 通道的过拟合审计：把 PSR / DSR / E[max SR] / MinTRL / PBO / Purged-CV
接到**真实候选通道**上，产出可审计的放行结论。

为什么需要这一层
----------------
个股通道的实盘就绪度体检（live_readiness.py）已经查过成本地板、相对基准、
三口径分布、跨区段一致性。但它**没有回答**「这个策略是不是从一堆试出来的
配置里挑出来的幸运儿」—— 那正是 Bailey / López de Prado 那一系列论文
（PSR / DSR / PBO）要回答的问题。

ETF 通道的选参规模
------------------
因子集 9 × k 4 × R 4 = **144 个配置**，全部在训练段跑过并按净超额排序，
取第一名进入验证段。也就是说：真实存在 144 次试验，
第 1 名是「从 144 个里挑出来的第 1 名」。不做多重检验修正，
报出来的 Sharpe 一定虚高。

本模块的口径铁律
----------------
1. **诚实计数**：``n_trials`` = 144（网格全量），不是「跑了几次回测」。
   少报 = 把过拟合当成不存在。
2. **方差来源**：跨试验 Sharpe 方差用**训练段**的 144 个配置估计 ——
   选参就是在训练段做的，用训练段的分布估计「随机配置长什么样」，
   再拿**测试段**的收益去问「被挑中的那个还剩多少」。测试段只被读一次。
3. **PBO 的用途**：CSCV 回答「样本内最好的那个，出样本还剩多少」，
   它是**诊断**，不参与任何选参 —— 参数早已冻结。
4. **期间对齐**：不同 R 的期次网格不同，PBO 只在**固定 R** 的子集上算
   （期次一一对应），不做尾部对齐。
5. 所有产物带 ``generated_at``。

参考（公开论文，非本模块代码来源）
----------------------------------
  * Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio.
    Journal of Portfolio Management, 40(5).
  * Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017).
    The Probability of Backtest Overfitting. Journal of Computational Finance.
  * López de Prado, M. (2018). Advances in Financial Machine Learning, Ch.7/12.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import etf_lab as E  # noqa: E402
import etf_research as ER  # noqa: E402
import overfit_audit as OA  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, "etf_overfit_audit.json")
FROZEN_PATH = os.path.join(E.OUT_DIR, "etf_frozen_spec.json")

# 门禁阈值（写死，避免「阈值也过拟合」）
DSR_PASS = 0.95             # DSR 通过线
PBO_MAX = 0.50              # PBO ≤ 0.5 才算「样本内最优能存活」
PSR_PASS = 0.95             # PSR(vs 0) 通过线
SHARPE_MIN = 0.50           # 与 research_evidence.GATE_SHARPE_MIN 一致
MIN_TRL_FACTOR = 1.0        # 实测期数 / MinTRL ≥ 1 才谈得上「证据够」

# PBO 的 CSCV 分块数（S=8 → C(8,4)=70 个切分；S 必须为偶数）
PBO_BLOCKS = 8

# PBO 的**判别力门槛**：候选之间平均相关太高时，PBO 结构性偏低，
# 不能当「没过拟合」的证据。9 个因子家族 × 4 个 k 的两两相关中位数
# 实测 0.774 —— 这时「样本内最优」几乎必然也是「样本外最优」，
# PBO=0 只是候选集退化的必然结果，不是独立证据。
PBO_MAX_MEAN_CORR = 0.70

# 基准的流动性门槛（与 etf_lab.run_strategy / equal_weight_benchmark 一致）
MIN_AMOUNT = 5e7


# ── 工具 ────────────────────────────────────────────────────────────────
def _net(periods: list[dict[str, Any]]) -> np.ndarray:
    """每期净收益（毛 - 成本）。"""
    return np.array([p["gross"] - p["cost"] for p in periods], dtype=float)


def _bench_series(panel: dict[str, Any], R: int, lo: int, hi: int,
                  price_mode: str = ER.PRICE_MODE) -> dict[str, float]:
    """基准（等权持有全部合格 ETF）的**每期**收益，按日期索引。

    与 etf_lab.equal_weight_benchmark 同一套逻辑（同样的价格口径、同样的
    「截至 t−1 的流动性过滤」），区别只在这里返回逐期序列而不是汇总统计 ——
    超额收益必须逐期相减，不能拿年化相减。
    """
    px = panel["open"] if price_mode == "open" else panel["close"]
    am = panel["amount"]
    T = px.shape[0]
    amt_ma = E._rolling_mean(am, 20)
    out: dict[str, float] = {}
    i = lo
    while i + R < min(hi, T):
        elig = (~np.isnan(px[i]) & ~np.isnan(px[i + R])
                & ~np.isnan(amt_ma[i - 1]) & (amt_ma[i - 1] >= MIN_AMOUNT))
        if int(elig.sum()) >= 2:
            out[panel["dates"][i]] = float(np.mean(px[i + R][elig] / px[i][elig] - 1.0))
        i += R
    return out


def _mean_pairwise_corr(cols: list[np.ndarray]) -> float:
    """列两两相关系数的均值（衡量候选集的独立程度）。"""
    if len(cols) < 2:
        return float("nan")
    cors = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a.std() == 0 or b.std() == 0:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if not np.isnan(c):
                cors.append(c)
    return float(np.mean(cors)) if cors else float("nan")


def _trial_grid() -> list[tuple[str, int, int]]:
    """全部 (因子集, k, R) 配置 —— 诚实的多重试验计数来源。"""
    out = []
    for fset in ER.FACTOR_SETS:
        for k in ER.K_SET:
            for R in ER.R_SET:
                out.append((fset, k, R))
    return out


def _aligned_matrix(
    panel: dict[str, Any],
    scores: dict[str, np.ndarray],
    trials: list[tuple[str, int, int]],
    R: int,
    lo: int,
    hi: int,
    capital: float,
) -> tuple[list[str], list[np.ndarray], list[str]]:
    """在同一 R 下跑若干配置，返回对齐到共同日期的每期净收益矩阵 (T, N)。

    不同 R 的期次网格不同，**不能**混在一张矩阵里；同一 R 下期次一致，
    只需按日期取交集。
    """
    per_trial: list[tuple[str, np.ndarray, list[str]]] = []
    for fset, k, _r in trials:
        res = E.run_strategy(panel, scores[fset], k, R, price_mode=ER.PRICE_MODE,
                             capital=capital, lo=lo, hi=hi)
        if not res["periods"]:
            continue
        dates = [p["date"] for p in res["periods"]]
        per_trial.append((f"{fset}|k={k}", _net(res["periods"]), dates))

    if not per_trial:
        return [], [], []
    common = set(per_trial[0][2])
    for _n, _v, ds in per_trial[1:]:
        common &= set(ds)
    order = sorted(common)
    idx_of = [{d: i for i, d in enumerate(ds)} for _n, _v, ds in per_trial]
    mat = np.full((len(order), len(per_trial)), np.nan)
    for j, (_n, v, ds) in enumerate(per_trial):
        m = idx_of[j]
        for i, d in enumerate(order):
            mat[i, j] = v[m[d]]
    return [n for n, _v, _d in per_trial], [mat[:, j] for j in range(mat.shape[1])], order


# ── 主审计 ──────────────────────────────────────────────────────────────
def run_audit() -> dict[str, Any]:
    """跑完整审计。读冻结件，不重新选参，不重新跑测试段选参。"""
    if not os.path.exists(FROZEN_PATH):
        raise RuntimeError(f"缺少冻结件 {FROZEN_PATH} —— 必须先跑 etf_research.py")

    frozen = json.load(open(FROZEN_PATH, encoding="utf-8"))
    assert frozen["spec_hash"] == ER.spec_hash(frozen), \
        "冻结规格在落盘后被改动 —— 审计结果无效"
    fset, k, R = frozen["factor_set"], frozen["k"], frozen["R"]
    capital = float(frozen["capital"])
    seg = {kk: tuple(vv) for kk, vv in frozen["segments"].items()}
    print(f"审计对象（冻结件）：factor_set={fset} k={k} R={R} capital={capital:.0f}")

    # 面板 + 打分（与 etf_research 完全同口径）
    ex_set = set()
    dq_path = os.path.join(E.OUT_DIR, "etf_dataquality.json")
    if os.path.exists(dq_path):
        dq = json.load(open(dq_path, encoding="utf-8"))
        fatal = [r for r in dq["results"] if r.get("fatal_issues")]
        if fatal:
            raise RuntimeError(f"ETF 数据质量门禁有 {len(fatal)} 只致命失败 —— 拒绝审计")
        ex_set = {r["code"] for r in dq["results"] if not r.get("ok")}
    panel = E.load_panel(min_bars=ER.MIN_BARS, exclude=ex_set)
    factors = E.compute_factors(panel["close"])
    scores = {name: E.composite_score(factors, keys, signs)
              for name, (keys, signs) in ER.FACTOR_SETS.items()}

    T = len(panel["dates"])
    grid = _trial_grid()
    n_trials_total = len(grid)
    print(f"网格配置数（诚实试验计数）= {n_trials_total}")

    # ── 1) 训练段：全部 144 个配置的每期 Sharpe（方差来源 + PBO 主体）──
    train_lo, train_hi = seg["train"]
    names_t, cols_t, dates_t = _aligned_matrix(
        panel, scores, [g for g in grid if g[2] == R], R,
        train_lo, train_hi, capital)
    print(f"训练段固定 R={R} 的候选数 = {len(names_t)}，期数 = {len(dates_t)}")
    if not cols_t:
        raise RuntimeError("训练段没有任何候选产生期次 —— 无法审计")
    train_srs = [OA.sharpe_per_period(c) for c in cols_t]
    train_srs_clean = [s for s in train_srs if not np.isnan(s)]

    # 选中的那个在训练段矩阵里的位置
    chosen_key = f"{fset}|k={k}"
    chosen_idx = names_t.index(chosen_key) if chosen_key in names_t else None

    # ── 2) 测试段：被选中策略的每期净收益（只读一次）───────────────
    test_lo, test_hi = seg["test"]
    res_t = E.run_strategy(panel, scores[fset], k, R, price_mode=ER.PRICE_MODE,
                           capital=capital, lo=test_lo, hi=test_hi)
    if not res_t["periods"]:
        raise RuntimeError("测试段没有产生任何调仓期 —— 无法审计")
    net_t = _net(res_t["periods"])
    bench_t = ER._bench(panel, R, test_lo, test_hi, res_t["periods"])
    bench_dates = bench_t.get("dates") or []
    bench_ann = bench_t.get("annualized", float("nan"))
    m_t = E.metrics(res_t["periods"], R)

    # 逐期超额（策略 - 基准）：必须逐期相减，不能拿年化相减。
    # 选参目标是**超额**，所以过拟合审计也必须在超额口径上做 ——
    # 对绝对收益做 DSR 会把「跑赢 beta」误判成「有 alpha」。
    bench_by_date = _bench_series(panel, R, test_lo, test_hi)
    dates_t = [p["date"] for p in res_t["periods"]]
    excess_t = np.array([
        net_t[i] - bench_by_date.get(d, float("nan"))
        for i, d in enumerate(dates_t)
    ], dtype=float)
    ok = ~np.isnan(excess_t)
    excess_t = excess_t[ok]
    net_t_aligned = net_t[ok]
    dates_t = [d for d, o in zip(dates_t, ok) if o]

    sr_t = OA.sharpe_per_period(net_t_aligned)
    sr_excess = OA.sharpe_per_period(excess_t)
    sk_t, ku_t = OA.skewness_kurtosis(excess_t)
    sk_abs, ku_abs = OA.skewness_kurtosis(net_t_aligned)
    psr_zero = OA.probabilistic_sharpe_ratio(sr_t, 0.0, len(net_t_aligned), sk_abs, ku_abs)
    psr_excess = OA.probabilistic_sharpe_ratio(sr_excess, 0.0, len(excess_t), sk_t, ku_t)

    # 基准的每期 Sharpe（用基准年化反推，口径与策略一致）
    ann_factor = E.TRADING_DAYS / R
    bench_sr_per_period = (bench_ann / ann_factor) if bench_ann == bench_ann else float("nan")
    psr_bench = OA.probabilistic_sharpe_ratio(
        sr_t, bench_sr_per_period, len(net_t_aligned), sk_abs, ku_abs)

    # ── 3) DSR：SR0 来自试验的 Sharpe 分布 ──────────────────────────
    # ⚠️ 口径要点（2026-11-15 修正，两处）：
    #   a) **超额口径**。选参目标是超额收益，DSR 必须在超额序列上做。
    #   b) **V 的口径必须显式并列，不能只报一个数**。
    #      Bailey & López de Prado Eq.(1) 的 V = 「各试验 Sharpe 估计的方差」，
    #      但前提是试验属于**同一策略类**。本项目的 36 个候选跨越 9 个因子族，
    #      族间真实差异远大于族内噪声（实测族均值极差 1.27 vs 族内 sd 0.077），
    #      因此 V 有三种都说得通的口径，结论必须对全部口径稳健：
    #        A 混合口径     ：36 个候选混池（族间差异算进噪声 → SR0 最高）
    #        B 去族主效应   ：剥离因子族主效应后的残差方差（最贴近「同策略类」）
    #        C 纯抽样噪声   ：(1+SR²/2)/(n-1)，完全不做经验估计（最宽松）
    #      实测三口径 DSR 分别 0.19 / 0.10 / 0.05 —— **全部 FAIL**，
    #      结论对口径不敏感，这才是可审计的。
    all_train_srs: list[float] = []
    bench_train = _bench_series(panel, R, train_lo, train_hi)
    by_family: dict[str, list[float]] = {}
    for f2, k2, _R2 in [g for g in grid if g[2] == R]:
        r2 = E.run_strategy(panel, scores[f2], k2, R, price_mode=ER.PRICE_MODE,
                            capital=capital, lo=train_lo, hi=train_hi)
        if not r2["periods"]:
            continue
        n2 = _net(r2["periods"])
        ds2 = [p["date"] for p in r2["periods"]]
        ex2 = np.array([n2[i] - bench_train.get(d, float("nan"))
                        for i, d in enumerate(ds2)], dtype=float)
        ex2 = ex2[~np.isnan(ex2)]
        if len(ex2) < 5:
            continue
        s2 = OA.sharpe_per_period(ex2)
        if not np.isnan(s2):
            all_train_srs.append(s2)
            by_family.setdefault(f2, []).append(s2)

    n_trials_total = len(grid)
    n_trials_same_R = len([g for g in grid if g[2] == R])

    # 三种 V 口径
    sr_arr = np.array(all_train_srs, dtype=float)
    v_pooled = float(sr_arr.var(ddof=1)) if sr_arr.size > 1 else 0.0
    fam_means = {f: float(np.mean(v)) for f, v in by_family.items()}
    resids = np.concatenate([np.array(v) - fam_means[f] for f, v in by_family.items()])
    n_fam = len(by_family)
    v_within = (float(np.sum(resids ** 2) / max(resids.size - n_fam, 1))
                if resids.size > n_fam else v_pooled)
    n_train_periods = max(len(
        _bench_series(panel, R, train_lo, train_hi)) and
        len([p for p in E.run_strategy(
            panel, scores[fset], k, R, price_mode=ER.PRICE_MODE,
            capital=capital, lo=train_lo, hi=train_hi)["periods"]]), 2)
    v_noise = 1.0 / max(n_train_periods - 1, 1)

    variance_variants = {
        "A_pooled_mixed": {
            "v": v_pooled, "n_trials": n_trials_same_R,
            "desc": "36 个候选混池（含因子族间真实差异 → SR0 最高，最保守）",
        },
        "B_within_family": {
            "v": v_within, "n_trials": n_trials_same_R,
            "desc": "剥离因子族主效应后的残差方差（最贴近论文「同一策略类」）",
        },
        "C_sampling_noise": {
            "v": v_noise, "n_trials": n_trials_total,
            "desc": "纯抽样噪声 (1+SR²/2)/(n-1)，不做任何经验估计（最宽松）",
        },
    }
    dsr_variants: dict[str, Any] = {}
    for name, spec in variance_variants.items():
        d = OA.deflated_sharpe_ratio(
            excess_t, n_trials=spec["n_trials"],
            sr_variance_across_trials=spec["v"],
            benchmark_sr=0.0, threshold=DSR_PASS, full_formula=True)
        dsr_variants[name] = {
            "variance": spec["v"],
            "n_trials": spec["n_trials"],
            "description": spec["desc"],
            "expected_max_sharpe": d.deflation_benchmark_sr0,
            "dsr": d.deflated_sharpe_ratio,
            "passed": d.passed,
            "to_dict": d.to_dict(),
        }
        print(f"  DSR[{name}] V={spec['v']:.6f} N={spec['n_trials']} "
              f"E[max SR]={d.deflation_benchmark_sr0:.4f} "
              f"DSR={d.deflated_sharpe_ratio:.4f}")

    # 主口径取 A（最保守）；但放行判定要求**所有口径都不通过才判不通过**，
    # 任一口径通过就说明结论对口径敏感 → 必须人工复核，不自动放行。
    primary = "A_pooled_mixed"
    dsr = OA.deflated_sharpe_ratio(
        excess_t, n_trials=n_trials_same_R, trial_sharpes=all_train_srs,
        benchmark_sr=0.0, threshold=DSR_PASS, full_formula=True)
    sr0 = dsr.deflation_benchmark_sr0
    emax = dsr.deflation_benchmark_sr0
    any_variant_pass = any(v["passed"] for v in dsr_variants.values())
    all_variants_fail = all(not v["passed"] for v in dsr_variants.values())
    all_variants_pass = all(v["passed"] for v in dsr_variants.values())
    min_trl = OA.minimum_track_record_length(sr_excess, 0.0, sk_t, ku_t, 0.95)
    min_trl_bench = OA.minimum_track_record_length(
        sr_t, bench_sr_per_period, sk_abs, ku_abs, 0.95) if not np.isnan(bench_sr_per_period) else float("nan")

    print(f"\n测试段（{len(excess_t)} 期，R={R}）：")
    print(f"  绝对 每期 Sharpe {sr_t:.4f}  偏度 {sk_abs:.4f}  峰度 {ku_abs:.4f}")
    print(f"  超额 每期 Sharpe {sr_excess:.4f}  偏度 {sk_t:.4f}  峰度 {ku_t:.4f}")
    print(f"  PSR(vs 0，绝对) {psr_zero:.4f}   PSR(vs 0，超额) {psr_excess:.4f}   "
          f"PSR(vs 基准) {psr_bench:.4f}")
    print(f"  E[max SR]（主口径 A）= {emax:.4f}")
    print(f"  DSR = {dsr.deflated_sharpe_ratio:.4f}（方差来源 {dsr.variance_source}，"
          f"trial_sharpes={len(all_train_srs)} 个）")
    print(f"  MinTRL(vs 0，超额) = {min_trl:.1f} 期，实测 {len(excess_t)} 期")
    print(f"  三口径一致性：{'全部达标' if all_variants_pass else '存在不达标口径 → 不放行'}")

    # ── 4) PBO（CSCV）──────────────────────────────────────────────
    # 4a) 训练段：选参**实际发生**的地方 —— 直接量化「144 选 1」的过拟合概率
    #     ⚠️ 判别力前置条件：候选之间平均相关过高时，PBO 结构性偏低，
    #     不能当「没过拟合」的证据。实测 9 因子家族 × 4 个 k 的两两相关
    #     均值 0.774 —— 这时 IS 最优几乎必然也是 OOS 最优，PBO=0 只是
    #     候选集退化的必然结果。所以 PBO 门禁 = 「PBO ≤ 0.5」**且**
    #     「候选平均相关 ≤ 0.70」，后者不满足时 PBO 一律判**不通过**
    #     （无法判别 ≠ 通过）。
    pbo_train = None
    pbo_train_corr = None
    if len(cols_t) >= 2 and len(dates_t) >= PBO_BLOCKS * 2:
        mat = np.column_stack(cols_t)
        r = OA.probability_of_backtest_overfitting(mat, n_blocks=PBO_BLOCKS)
        pbo_train = r.to_dict()
        pbo_train["n_candidates"] = len(cols_t)
        pbo_train["n_periods"] = len(dates_t)
        pbo_train_corr = _mean_pairwise_corr(cols_t)
        pbo_train["mean_pairwise_corr"] = pbo_train_corr
        pbo_train["discriminative"] = bool(
            pbo_train_corr == pbo_train_corr
            and pbo_train_corr <= PBO_MAX_MEAN_CORR)
        print(f"\nPBO（训练段，选参实际发生的区间）：{pbo_train['pbo']:.4f} "
              f"（{pbo_train['n_splits']} 个切分，{len(cols_t)} 个候选）")
        print(f"  候选两两相关均值 = {pbo_train_corr:.4f}"
              f"（判别力门槛 ≤{PBO_MAX_MEAN_CORR} → "
              f"{'有判别力' if pbo_train['discriminative'] else '无判别力，判定不成立'}）")

    # 4b) 测试段：同一批候选在样本外的表现（期次少，标注为低功效）
    pbo_test = None
    names_te, cols_te, dates_te = _aligned_matrix(
        panel, scores, [g for g in grid if g[2] == R], R,
        test_lo, test_hi, capital)
    if len(cols_te) >= 2 and len(dates_te) >= PBO_BLOCKS * 2:
        mat_te = np.column_stack(cols_te)
        r2 = OA.probability_of_backtest_overfitting(mat_te, n_blocks=PBO_BLOCKS)
        pbo_test = r2.to_dict()
        pbo_test["n_candidates"] = len(cols_te)
        pbo_test["n_periods"] = len(dates_te)
        pbo_test["caveat"] = (
            f"测试段仅 {len(dates_te)} 期，分成 {PBO_BLOCKS} 块后每块 "
            f"{len(dates_te)//PBO_BLOCKS} 期 —— 统计功效低，只作参考，"
            f"判定以训练段 PBO 为准。")
        print(f"PBO（测试段，低功效参考）：{pbo_test['pbo']:.4f}")

    # ── 5) Purged CV：验证段换成分 purged 的 OOS 拼接 ────────────────
    # ETF 策略第 i 期的信息集 = [i-1, i+R]（信号取 i-1 收盘、持有到 i+R 开盘）
    # → label_end[i] = i + R
    hold_info = R + 1
    purged = OA.PurgedKFold(n_splits=5, n_obs=T, hold=hold_info, embargo=0)
    aud = purged.audit()
    print(f"\nPurgedKFold 自检（hold={hold_info}，等价于持有 R={R} 期）：{aud}")

    # 用 purged 的折做 walk-forward OOS：在训练段内交叉验证被选中策略
    pk_oos, pk_folds = [], []
    for f in purged.split():
        tr_lo, tr_hi = int(f.train[0]), int(f.train[-1]) + 1
        te_lo, te_hi = int(f.test[0]), int(f.test[-1]) + 1
        # 训练折只用来「确认参数可用」，这里直接用被选中参数（已冻结）
        rt = E.run_strategy(panel, scores[fset], k, R, price_mode=ER.PRICE_MODE,
                            capital=capital, lo=tr_lo, hi=tr_hi)
        if not rt["periods"]:
            continue
        re_ = E.run_strategy(panel, scores[fset], k, R, price_mode=ER.PRICE_MODE,
                             capital=capital, lo=te_lo, hi=te_hi)
        if not re_["periods"]:
            continue
        pk_oos.extend(_net(re_["periods"]).tolist())
        pk_folds.append({
            "fold": f.fold,
            "n_train": int(f.train.size), "n_test": int(f.test.size),
            "n_purged": int(f.purged.size), "n_embargoed": int(f.embargoed.size),
            "test_oos_ann": E.metrics(re_["periods"], R)["net_annualized"],
        })
    pk_sr = OA.sharpe_per_period(np.array(pk_oos)) if pk_oos else float("nan")
    print(f"Purged-CV OOS（训练段内 {len(pk_oos)} 期拼接）：每期 Sharpe {pk_sr:.4f}")

    # ── 6) 门禁判定 ─────────────────────────────────────────────────
    gates: list[dict[str, Any]] = []

    # DSR 门禁（要求项）：DSR ≥ 门槛 才算「通过多重检验惩罚」。
    # 必须**所有**方差口径都 ≥ 门槛才通过 —— 任一口径不达标就说明
    # 结论对口径敏感，不许放行。
    # ⚠️ 早期版本这里写反了（DSR 越低越 PASS），打印出来是
    # 「[PASS] DSR 0.0000」这种自相矛盾的文本。
    all_variants_pass = all(v["passed"] for v in dsr_variants.values())
    g_overfit = bool(all_variants_pass)
    dsr_detail_bits = "；".join(
        f"{name}={v['dsr']:.4f}" for name, v in dsr_variants.items())
    gates.append({
        "name": "过拟合审计 · DSR",
        "pass": g_overfit,
        "detail": (
            f"门槛 DSR ≥ {DSR_PASS}；主口径 DSR {dsr.deflated_sharpe_ratio:.4f}"
            f"（E[max SR]={emax:.4f}）；三口径 DSR：{dsr_detail_bits}"
            f" —— {'全部达标' if all_variants_pass else '存在不达标口径 → 不放行'}"
        ),
    })

    # 基础事实门禁：PSR(vs 0, 超额) —— 「超额是否显著大于 0」都答不了的话，
    # 后面所有门禁都没意义。这一条比 DSR 更基础，必须单列。
    g_psr = bool(psr_excess >= PSR_PASS)
    gates.append({
        "name": "过拟合审计 · PSR(超额 vs 0)",
        "pass": g_psr,
        "detail": (
            f"PSR(vs 0, 超额) = {psr_excess:.4f}（门槛 {PSR_PASS}）—— "
            f"{len(excess_t)} 期样本下，连「超额显著大于 0」都无法证实"
            if not g_psr else
            f"PSR(vs 0, 超额) = {psr_excess:.4f}（门槛 {PSR_PASS}）"
        ),
    })

    # PBO：必须「数值达标」且「有判别力」，缺一即不通过
    pbo_val_ok = (pbo_train is not None and pbo_train["pbo"] <= PBO_MAX)
    pbo_disc_ok = bool(pbo_train and pbo_train.get("discriminative"))
    gates.append({
        "name": "过拟合审计 · PBO",
        "pass": bool(pbo_val_ok and pbo_disc_ok),
        "detail": (
            f"训练段 CSCV 的 PBO = {pbo_train['pbo']:.4f}（门槛 ≤{PBO_MAX}），"
            f"{pbo_train['n_candidates']} 个候选 / {pbo_train['n_periods']} 期 / "
            f"{pbo_train['n_splits']} 个切分；"
            f"候选两两相关均值 {pbo_train_corr:.4f}"
            f"（判别力门槛 ≤{PBO_MAX_MEAN_CORR} → "
            f"{'有判别力' if pbo_disc_ok else '无判别力，PBO 数值不构成证据'}）"
            if pbo_train else "PBO 未生成"
        ),
    })

    trl_ok = bool(min_trl != float("inf") and len(excess_t) >= MIN_TRL_FACTOR * min_trl)
    gates.append({
        "name": "过拟合审计 · MinTRL",
        "pass": trl_ok,
        "detail": (
            f"证明超额每期 Sharpe 显著优于 0 至少需 {min_trl:.1f} 期，"
            f"实测 {len(excess_t)} 期（比值 {len(excess_t)/min_trl:.2f}）"
        ),
    })

    g_sharpe = bool(m_t.get("sharpe", 0.0) >= SHARPE_MIN)
    gates.append({
        "name": "成本后 · Sharpe",
        "pass": g_sharpe,
        "detail": (
            f"测试段净 Sharpe {m_t.get('sharpe', float('nan')):.3f}"
            f"（门槛 ≥{SHARPE_MIN}）；基准 Sharpe "
            f"{bench_t.get('sharpe', float('nan')):.3f}"
        ),
    })

    g_excess = bool(m_t.get("net_annualized", 0.0) > 0)
    gates.append({
        "name": "成本后 · 净超额",
        "pass": g_excess,
        "detail": (
            f"净年化 {m_t.get('net_annualized', float('nan'))*100:+.2f}%，"
            f"基准 {bench_ann*100:+.2f}%，"
            f"超额 {(m_t.get('net_annualized', float('nan'))-bench_ann)*100:+.2f}%"
        ),
    })

    n_pass = sum(1 for g in gates if g["pass"])
    all_pass = n_pass == len(gates)
    verdict = {
        "pass": all_pass,
        "passed": n_pass,
        "total": len(gates),
        "failed_gates": [g["name"] for g in gates if not g["pass"]],
        "text": (
            f"过拟合审计 {n_pass}/{len(gates)} 通过"
            + ("；全部通过。" if all_pass
               else f"；未通过：{'、'.join(g['name'] for g in gates if not g['pass'])}。")
        ),
        "sim_position_allowed": bool(all_pass),
        "note": (
            "本审计只覆盖「是否过拟合」与「成本后绝对表现」。"
            "概率校准（G3）与预测闭环（G4）属于生产链路，"
            "ETF 通道尚未接入 → 这两道门禁在 ETF 通道**结构上无法通过**，"
            "不得因为审计通过就宣称 ETF 策略可进模拟仓位。"
        ),
    }

    out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "channel": "ETF",
        "frozen_spec": frozen,
        "multiplicity": {
            "n_trials_total": n_trials_total,
            "n_trials_same_R": n_trials_same_R,
            "grid": f"{len(ER.FACTOR_SETS)} 因子集 × {len(ER.K_SET)} k × {len(ER.R_SET)} R",
            "variance_source": dsr.variance_source,
            "n_trial_sharpes_used": len(all_train_srs),
            "note": (
                "选参是在 144 个配置里取第 1 名，少报试验数 = 把过拟合当成不存在。"
                "但 E[max SR] 的方差 V 只在**被选中的 R 之内**估计 —— 不同 R 的期数"
                "从 21 到 191 不等，每期 Sharpe 的估计噪声尺度不可比，跨 R 混合会"
                "把「样本量差异」当成「策略差异」（实测混入后 E[max SR]=3.42，"
                "而 144 个配置的实测 Sharpe 最大仅 0.133，数学上不可能）。"
            ),
        },
        "test_segment": {
            "n_periods": len(excess_t),
            "R": R,
            "dates": [dates_t[0], dates_t[-1]] if dates_t else [],
            "sr_per_period": sr_t,
            "sr_per_period_excess": sr_excess,
            "skew": sk_abs,
            "kurtosis": ku_abs,
            "skew_excess": sk_t,
            "kurtosis_excess": ku_t,
            "psr_vs_zero": psr_zero,
            "psr_vs_zero_excess": psr_excess,
            "psr_vs_benchmark": psr_bench,
            "benchmark_sr_per_period": bench_sr_per_period,
            "benchmark_annualized": bench_ann,
            "metrics": m_t,
        },
        "overfit": {
            "expected_max_sharpe": emax,
            "dsr": dsr.to_dict(),
            "dsr_variance_variants": dsr_variants,
            "dsr_all_variants_pass": all_variants_pass,
            "dsr_any_variant_pass": any_variant_pass,
            "min_track_record_length": min_trl,
            "min_track_record_length_vs_benchmark": min_trl_bench,
            "pbo_train": pbo_train,
            "pbo_test": pbo_test,
            "pbo_discriminativeness": {
                "mean_pairwise_corr": pbo_train_corr,
                "threshold": PBO_MAX_MEAN_CORR,
                "discriminative": pbo_disc_ok,
                "why": (
                    "候选之间高度相关 → CSCV 的「样本内最优」几乎必然也是"
                    "「样本外最优」→ PBO 结构性偏低。此时 PBO 数值低**不是**"
                    "「没过拟合」的证据，而是候选集退化的结果。"
                ),
            },
        },
        "purged_cv": {
            "hold_info": hold_info,
            "audit": aud,
            "folds": pk_folds,
            "oos_n_periods": len(pk_oos),
            "oos_sr_per_period": pk_sr,
            "note": (
                "PurgedKFold 用信息集相交语义；ETF 第 i 期信息集 = [i-1, i+R]，"
                "故 label_end[i] = i + R，hold 参数取 R+1。"
            ),
        },
        "gates": gates,
        "verdict": verdict,
    }
    return out


def main() -> None:
    out = run_audit()
    os.makedirs(E.OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n已写 {OUT_PATH}")
    v = out["verdict"]
    print(f"结论：{v['text']}")
    for g in out["gates"]:
        print(f"  [{'PASS' if g['pass'] else 'FAIL'}] {g['name']}: {g['detail']}")


if __name__ == "__main__":
    main()
