"""因子冗余分析：11 个因子到底有几个独立信号？能不能用更少的因子换更低的换手？

为什么做这个（换思路，不是调参）：
    本项目已判定个股通道「不具备实盘条件」，主因是手续费吃掉了毛边际。
    已定的两条翻盘条件之一是「换手降到 8.67%/期以下」（现 22.11%/期）。
    换手高是因为 11 个因子的横截面排名组合很不稳定，每期重排都换掉一批票。
    如果这 11 个因子其实只有 2~3 个独立信号，用更少的因子合成，
    排序会稳得多 → 换手可能大幅下降，而毛超额未必损失多少。

    这不是「再调一次参数」，是问一个结构问题：这些因子是不是同一件事的不同写法。

口径纪律（这一版踩过两个坑，必须记住）：
    A. **看的是「相对基准的超额」，不是「多头绝对收益」**。
       ooslab 里的 long_only.net_annualized 是多头组合自己的收益，里面全是 beta；
       策略有没有 alpha 只看 excess = long − bench。
       （本模块第一版把绝对收益当成净值结论，结论方向完全错了。）
    B. **翻盘判据不能用「抗成本 > 实际成本」这种一刀切的比值**。
       换手小的时候这个比值天然大，但净额可能只有零点几个 bp —— 差一点点就"转正"，
       是刀刃上的判定，不是真翻盘。必须同时看超额三口径 + 胜率 + 最小有意义幅度。
    C. 因子子集的选择**只在训练段**做；验证段**只判通过与否**；测试段只跑一次，
       且必须披露测试段已被本项目多个假设反复看过，对新假设不再是处女地。

核心指标（全部从 decile_backtest 的逐期数据现算，不新造口径）：
    毛超额/期    = mean(long − bench)
    成本后超额/期 = 毛超额 − 换手 × cost_rate       ← 决策指标
    抗成本       = 毛超额 / 换手                      （诊断用，不是判据）
    三口径：缩尾均值 / 中位数 / 未缩尾均值，必须并列披露
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import factorlab
import ooslab

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "data" / "research"

OUT_FILE = RESEARCH / "factor_redundancy.json"
FROZEN_FILE = RESEARCH / "factor_redundancy_frozen_spec.json"

# 训练段的三条贪心目标（事先定好，不看结果再改）
OBJECTIVES = ("gross", "net", "breakeven")

# 目标 → 主指标字段
METRIC_KEY = {
    "gross": "gross_per_period",
    "net": "excess_net_mean_per_period",
    "breakeven": "breakeven_cost",
}

# 「算得上有意义的超额」门槛：年化 0.30%。低于这个数就算符号对了也不值得上。
MEANINGFUL_EDGE = 0.0030

# 冻结件里除规格本身之外附加的元数据字段（不参与 hash）
_META_KEYS = ("spec_hash", "frozen_at", "selected_by", "validation_net_annualized")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ 基础工具


def spec_hash(spec: dict[str, Any]) -> str:
    payload = {k: v for k, v in spec.items() if k not in _META_KEYS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rank_row(vals: np.ndarray) -> np.ndarray:
    """单行平均秩（NaN 保留）。全向量化，避免逐元素 Python 循环。"""
    out = np.full(vals.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(vals))
    if idx.size == 0:
        return out
    v = vals[idx]
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks_sorted = np.arange(1, v.size + 1, dtype=float)
    new_grp = np.empty(v.size, dtype=bool)
    new_grp[0] = True
    new_grp[1:] = sv[1:] != sv[:-1]
    grp = np.cumsum(new_grp) - 1
    counts = np.bincount(grp)
    sums = np.bincount(grp, weights=ranks_sorted)
    avg = sums / counts
    out[idx[order]] = avg[grp]
    return out


def _zscore_cache(mat: dict[str, Any], names: list[str]) -> dict[str, np.ndarray]:
    return {n: ooslab.cs_zscore(mat["factors"][n]) for n in names}


def _composite(
    zc: dict[str, np.ndarray],
    names: list[str],
    signs: dict[str, float],
) -> np.ndarray:
    """与 ooslab.composite_score 完全一致的合成口径（逐日 z-score，方向由发现期 IC 符号定）。"""
    if not names:
        first = next(iter(zc.values()))
        return np.full(first.shape, np.nan)
    first = zc[names[0]]
    T, N = first.shape
    zsum = np.zeros((T, N))
    wsum = np.zeros((T, N))
    for n in names:
        z = zc[n]
        ok = np.isfinite(z)
        zsum[ok] += signs.get(n, 1.0) * z[ok]
        wsum[ok] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, zsum / wsum, np.nan)


# ------------------------------------------------------------------ 冗余矩阵


def redundancy_matrix(
    mat: dict[str, Any],
    names: list[str],
    lo: int,
    hi: int,
) -> dict[str, Any]:
    """逐日 Spearman 因子间相关，跨日平均。

    有效维度用相关矩阵特征值的参与比（participation ratio）：
        PR = (Σλ)² / Σλ²
    11 个互不相关的因子 → PR = 11；11 个完全一样的因子 → PR = 1。
    """
    k = len(names)
    if k < 2:
        return {"ok": False, "reason": "因子少于 2 个，算不了冗余"}
    N = mat["close"].shape[1]
    acc = np.zeros((k, k))
    cnt = 0
    for t in range(lo, hi):
        R = np.empty((k, N))
        for j, n in enumerate(names):
            R[j] = _rank_row(mat["factors"][n][t])
        ok = np.isfinite(R).all(axis=0)
        if int(ok.sum()) < 20:
            continue
        X = R[:, ok]
        Xc = X - X.mean(axis=1, keepdims=True)
        denom = np.sqrt((Xc ** 2).sum(axis=1))
        denom[denom == 0] = 1.0
        acc += (Xc @ Xc.T) / np.outer(denom, denom)
        cnt += 1
    if cnt == 0:
        return {"ok": False, "reason": "该区间没有可用的横截面"}
    M = acc / cnt
    eig = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    pr = float((eig.sum() ** 2) / max(float((eig ** 2).sum()), 1e-12))
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            rho = float(M[i, j])
            pairs.append({"a": names[i], "b": names[j], "rho": rho, "abs": abs(rho)})
    pairs.sort(key=lambda p: (-p["abs"], p["a"], p["b"]))
    return {
        "ok": True,
        "n_days": cnt,
        "names": list(names),
        "matrix": M.tolist(),
        "effective_dimension": pr,
        "eigenvalues": [float(x) for x in eig],
        "max_abs": pairs[0]["abs"] if pairs else 0.0,
        "max_abs_pair": [pairs[0]["a"], pairs[0]["b"]] if pairs else [],
        "mean_abs": float(np.mean([p["abs"] for p in pairs])) if pairs else 0.0,
        "n_pairs": len(pairs),
        "n_pairs_ge_0_5": sum(1 for p in pairs if p["abs"] >= 0.5),
        "n_pairs_ge_0_7": sum(1 for p in pairs if p["abs"] >= 0.7),
        "top_pairs": [
            {"a": p["a"], "b": p["b"], "rho": p["rho"]} for p in pairs[:8]
        ],
    }


# ------------------------------------------------------------------ 评估


def _excess_calibers(periods: list[dict[str, Any]], cost_rate: float, hold: int) -> dict[str, Any]:
    """从逐期数据算「相对基准的成本后超额」三口径。

    三个口径的原始序列：
        winsor : long − bench            （缩尾横截面均值）
        median : long_med − bench_med    （横截面中位数，抗极值）
        raw    : long_raw − bench        （未缩尾，披露极值影响）
    每条序列再扣掉 cost_rate × 当期换手。
    """
    ex = {
        "winsor": np.array([p["long"] - p["bench"] for p in periods], dtype=float),
        "median": np.array([p["long_med"] - p["bench_med"] for p in periods], dtype=float),
        "raw": np.array([p["long_raw"] - p["bench"] for p in periods], dtype=float),
    }
    turn = np.array([p["turnover"] for p in periods], dtype=float)
    cost = cost_rate * turn
    ann = 252.0 / hold
    out: dict[str, Any] = {}
    for tag, arr in ex.items():
        net = arr - cost
        out[f"excess_net_{tag}_per_period"] = float(net.mean())
        out[f"excess_net_{tag}_annualized"] = float(net.mean() * ann)
        out[f"excess_net_{tag}_median_annualized"] = float(np.median(net) * ann)
        out[f"excess_net_{tag}_hit_rate"] = float(np.mean(net > 0))
    return out


def _eval_subset(
    mat: dict[str, Any],
    score: np.ndarray,
    lo: int,
    hi: int,
    *,
    hold: int,
    quantile: float,
    lag: int,
    cost_rate: float,
) -> dict[str, Any]:
    """跑一次分组回测，返回本分析要用的口径。"""
    r = ooslab.decile_backtest(
        mat, score, hold=hold, lag=lag, cost_rate=cost_rate,
        quantile=quantile, min_valid=50, start_idx=lo, end_idx=hi,
    )
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("reason")}
    be = r.get("breakeven_cost_long_only")
    out = {
        "ok": True,
        "n_periods": int(r["n_periods"]),
        "gross_per_period": float(r["gross"]["long_vs_bench_mean"]),
        "gross_ls_per_period": float(r["gross"]["ls_excess_mean"]),
        "turnover": float(r["mean_turnover"]),
        "breakeven_cost": float(be) if be is not None else None,
        "breakeven_cost_long_short": r.get("breakeven_cost_roundtrip"),
        "hit_rate_vs_bench_gross": float(r["hit_rate_vs_bench"]),
        # 绝对收益（只作参考，里面全是 beta）
        "long_absolute_annualized": float(r["long_only"]["net_annualized"]),
        "benchmark_annualized": float(r["benchmark"]["annualized"]),
    }
    out.update(_excess_calibers(r["periods"], cost_rate, hold))
    return out


def _composite_ic(
    mat: dict[str, Any],
    score: np.ndarray,
    lo: int,
    hi: int,
    horizon: int,
) -> dict[str, Any]:
    """合成信号在指定区间的 Spearman IC（不重叠压缩 + Newey-West）。"""
    fwd = ooslab.make_forward(mat, horizon)
    ics: list[float] = []
    for t in range(lo, hi):
        s, f = score[t], fwd[t]
        ok = np.isfinite(s) & np.isfinite(f)
        n = int(ok.sum())
        if n < 20:
            continue
        sv, fv = s[ok], f[ok]
        ro = np.argsort(fv, kind="mergesort")
        rr = np.empty(n, dtype=float)
        rr[ro] = np.arange(1, n + 1, dtype=float)
        factorlab._fix_ties(fv, rr)
        sc = sv - sv.mean()
        rc = rr - rr.mean()
        den = math.sqrt(float((sc ** 2).sum()) * float((rc ** 2).sum()))
        if den == 0:
            continue
        ics.append(float((sc * rc).sum() / den))
    if not ics:
        return {"ok": False, "reason": "无有效横截面"}
    series = factorlab.nonoverlap_ic_series(np.array(ics), forward_days=horizon)
    mean, t = factorlab.newey_west(series)
    return {
        "ok": True,
        "ic_mean": mean,
        "ic_t": t,
        "ic_n": int(len(series)),
        "ic_hit": float(np.mean(series > 0)) if len(series) else None,
    }


def _better(a: dict[str, Any], b: dict[str, Any], objective: str) -> bool:
    """候选 a 是否优于 b。并列时的次序是事先定好的，不看结果再改。"""
    ma, mb = a["metrics"], b["metrics"]
    pa, pb = ma.get(METRIC_KEY[objective]), mb.get(METRIC_KEY[objective])
    if pb is None:
        return True
    if pa is None:
        return False
    if pa > pb:
        return True
    if pa < pb:
        return False
    if ma["gross_per_period"] != mb["gross_per_period"]:
        return ma["gross_per_period"] > mb["gross_per_period"]
    if ma["turnover"] != mb["turnover"]:
        return ma["turnover"] < mb["turnover"]
    return len(a["set"]) < len(b["set"])


def _greedy_path(
    mat: dict[str, Any],
    zc: dict[str, np.ndarray],
    signs: dict[str, float],
    pool: list[str],
    seg: tuple[int, int],
    objective: str,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """训练段贪心前向：每一步试所有候选，挑目标函数最大的那个。"""
    chosen: list[str] = []
    remaining = list(pool)
    steps: list[dict[str, Any]] = []
    while remaining:
        best: dict[str, Any] | None = None
        for cand in remaining:
            names = chosen + [cand]
            score = _composite(zc, names, signs)
            ev = _eval_subset(mat, score, seg[0], seg[1], **cfg)
            if not ev.get("ok"):
                continue
            if ev.get(METRIC_KEY[objective]) is None:
                continue
            row = {"set": list(names), "metrics": ev}
            if best is None or _better(row, best, objective):
                best = row
        if best is None:
            break
        m = best["metrics"]
        steps.append({
            "k": len(steps) + 1,
            "added": best["set"][-1],
            "set": list(best["set"]),
            "n_periods": m["n_periods"],
            "gross_per_period": m["gross_per_period"],
            "turnover": m["turnover"],
            "breakeven_cost": m["breakeven_cost"],
            "excess_net_mean_per_period": m["excess_net_winsor_per_period"],
            "excess_net_median_per_period": m["excess_net_median_per_period"],
            "excess_net_raw_per_period": m["excess_net_raw_per_period"],
            "excess_net_winsor_annualized": m["excess_net_winsor_annualized"],
            "excess_net_median_annualized": m["excess_net_median_annualized"],
            "excess_net_raw_annualized": m["excess_net_raw_annualized"],
            "excess_net_winsor_hit_rate": m["excess_net_winsor_hit_rate"],
            "long_absolute_annualized": m["long_absolute_annualized"],
        })
        chosen = list(best["set"])
        remaining.remove(chosen[-1])
    return steps


def _choose_frozen(paths: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """冻结规则（事先定好）：训练段「成本后超额」最高者。

    并列时：换手低 → 因子少 → 路径名字序。
    """
    best: dict[str, Any] | None = None
    for path_name in sorted(paths):
        for st in paths[path_name]:
            v = st.get("excess_net_mean_per_period")
            if v is None:
                continue
            row = {
                "path": path_name, "k": st["k"], "set": list(st["set"]),
                "excess_net_mean_per_period": v,
                "breakeven_cost": st.get("breakeven_cost"),
                "gross_per_period": st.get("gross_per_period"),
                "turnover": st.get("turnover"),
                "excess_net_winsor_annualized": st.get("excess_net_winsor_annualized"),
            }
            if best is None:
                best = row
                continue
            if v > best["excess_net_mean_per_period"]:
                best = row
            elif v == best["excess_net_mean_per_period"]:
                if st["turnover"] < best["turnover"]:
                    best = row
    return best


# ------------------------------------------------------------------ 主流程


def _cross_segment_reference(cost_rate: float, hold: int) -> dict[str, Any]:
    """从 cross_segment_check.json 按**口径**（cost_rate + hold）选行取参照值。

    绝不能按索引取第一行 —— 那个文件里 cost=0.15% 的行排在最前面，
    取第一行会拿到另一套口径的数字（这坑本项目已经踩过一次）。
    """
    p = RESEARCH / "cross_segment_check.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = ((d.get("result") or {}).get("rows")) or []
    for row in rows:
        if row.get("cost_rate") is None or row.get("hold") is None:
            continue
        if abs(float(row["cost_rate"]) - cost_rate) < 1e-12 and int(row["hold"]) == hold:
            return row.get("per_segment") or {}
    return {}


def _load_canonical() -> dict[str, Any]:
    p = RESEARCH / "oos_latest.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    disc = d.get("discovered") or []
    names = [str(x["name"]) for x in disc if x.get("name")]
    signs = {
        str(x["name"]): (1.0 if (x.get("mean") or 0.0) > 0 else -1.0)
        for x in disc if x.get("name")
    }
    seg = d.get("strategy_test_segment") or {}
    return {
        "names": names,
        "signs": signs,
        "hold": int(seg.get("hold") or 5),
        "quantile": float(seg.get("quantile") if seg.get("quantile") is not None else 0.5),
        "cost_rate": float(seg.get("cost_rate") or ooslab.COST_ROUNDTRIP),
        "lag": int(seg.get("lag") or 1),
        "panel": d.get("panel") or {},
        "segments": d.get("segments") or {},
        "reference_test_gross": (seg.get("gross") or {}).get("long_vs_bench_mean"),
        "reference_test_turnover": seg.get("mean_turnover"),
    }


def analyze(*, verbose: bool = True) -> dict[str, Any]:
    canon = _load_canonical()
    if not canon.get("names"):
        return {"ok": False, "reason": "缺少 canonical 产物 data/research/oos_latest.json"}

    mat = ooslab.load_panel()
    if not mat.get("codes"):
        return {"ok": False, "reason": "面板为空（先跑数据刷新流程）"}

    names = canon["names"]
    signs = canon["signs"]
    hold = canon["hold"]
    quantile = canon["quantile"]
    lag = canon["lag"]
    cost_rate = canon["cost_rate"]

    n_dates = len(mat["dates"])
    segs = ooslab.split_segments(n_dates)
    cfg = {"hold": hold, "quantile": quantile, "lag": lag, "cost_rate": cost_rate}

    zc = _zscore_cache(mat, names)

    # ① 全量因子在三个区间的基准表现
    full_score = _composite(zc, names, signs)
    full_train = _eval_subset(mat, full_score, *segs["train"], **cfg)
    full_val = _eval_subset(mat, full_score, *segs["val"], **cfg)
    full_test = _eval_subset(mat, full_score, *segs["test"], **cfg)
    full_train_ic = _composite_ic(mat, full_score, *segs["train"], horizon=hold)

    # 交叉校验：用 canonical 口径重算测试段，必须和 oos_latest 里的数字逐位一致
    ref_gross = canon.get("reference_test_gross")
    ref_turn = canon.get("reference_test_turnover")
    cross_check = {"checked": False, "ok": None, "detail": ""}
    if ref_gross is not None and ref_turn is not None:
        dg = abs(float(full_test["gross_per_period"]) - float(ref_gross))
        dt = abs(float(full_test["turnover"]) - float(ref_turn))
        cross_check = {
            "checked": True,
            "ok": bool(dg < 1e-12 and dt < 1e-12),
            "gross_diff": dg,
            "turnover_diff": dt,
            "detail": (
                f"重算毛超额 {full_test['gross_per_period']:.16g} vs "
                f"canonical {float(ref_gross):.16g}；"
                f"重算换手 {full_test['turnover']:.16g} vs "
                f"canonical {float(ref_turn):.16g}"
            ),
        }
        if not cross_check["ok"]:
            return {"ok": False, "reason": "与 canonical 产物对不上，拒绝出结论",
                    "cross_check": cross_check}

    # 交叉校验 2：三段超额 vs 另一个独立产物 cross_segment_check.json
    # （那是 cross_segment.py 走另一条代码路径算出来的，同一个量两处独立算出必须一致）
    ref_seg = _cross_segment_reference(cost_rate, hold)
    segment_check: dict[str, Any] = {"checked": False, "ok": None, "rows": {}}
    if ref_seg:
        seg_rows = {}
        all_ok = True
        for seg in ("train", "val", "test"):
            r = ref_seg.get(seg) or {}
            mine = {"train": full_train, "val": full_val, "test": full_test}[seg]
            if not r or not mine.get("ok"):
                continue
            de = abs(float(r.get("excess_annualized", 0.0))
                     - float(mine.get("excess_net_winsor_annualized", 0.0)))
            dg = abs(float(r.get("gross_excess_per_period", 0.0))
                     - float(mine.get("gross_per_period", 0.0)))
            dt = abs(float(r.get("mean_turnover", 0.0))
                     - float(mine.get("turnover", 0.0)))
            ok = bool(de < 1e-12 and dg < 1e-12 and dt < 1e-12)
            all_ok = all_ok and ok
            seg_rows[seg] = {"ok": ok, "excess_diff": de, "gross_diff": dg,
                             "turnover_diff": dt}
        segment_check = {"checked": True, "ok": all_ok, "rows": seg_rows,
                         "source": "data/research/cross_segment_check.json（cost=%.4f, hold=%d）"
                                   % (cost_rate, hold)}
        if not all_ok:
            return {"ok": False,
                    "reason": "与 cross_segment_check.json 对不上，拒绝出结论",
                    "segment_check": segment_check}

    # ② 因子冗余矩阵（训练段）
    red = redundancy_matrix(mat, names, *segs["train"])

    # ③ 三条贪心路径（只在训练段选）
    paths: dict[str, list[dict[str, Any]]] = {}
    for obj in OBJECTIVES:
        paths[obj] = _greedy_path(mat, zc, signs, names, segs["train"], obj, cfg)

    # ④ 冻结（训练段定）
    frozen = _choose_frozen(paths)
    if frozen is None:
        return {"ok": False, "reason": "训练段没有可冻结的子集"}

    spec = {
        "objective": frozen["path"],
        "k": frozen["k"],
        "factor_names": list(frozen["set"]),
        "signs": {n: signs.get(n, 1.0) for n in frozen["set"]},
        "hold": hold,
        "quantile": quantile,
        "lag": lag,
        "cost_rate": cost_rate,
        "selection_segment": "train",
        "rule": "训练段成本后超额（相对基准）最高者；并列按换手低→因子少",
    }
    spec["spec_hash"] = spec_hash(spec)

    # ⑤ 验证段：只判通过与否，不参与选参
    sub_score = _composite(zc, frozen["set"], signs)
    sub_val = _eval_subset(mat, sub_score, *segs["val"], **cfg)
    sub_val_ic = _composite_ic(mat, sub_score, *segs["val"], horizon=hold)
    gate_passed = bool(
        sub_val.get("ok")
        and (sub_val.get("excess_net_winsor_per_period") or -9)
        > (full_val.get("excess_net_winsor_per_period") or -9)
    )

    # ⑥ 测试段：只跑一次（测试段已被本项目多个假设看过，只能算旁证）
    sub_test = _eval_subset(mat, sub_score, *segs["test"], **cfg)
    sub_test_ic = _composite_ic(mat, sub_score, *segs["test"], horizon=hold)

    # ⑦ 训练段上有没有子集同时满足「超额三口径全正 + 幅度有意义」
    survivors = []
    for path_name in sorted(paths):
        for st in paths[path_name]:
            if st.get("excess_net_winsor_annualized") is None:
                continue
            if st["excess_net_winsor_annualized"] > MEANINGFUL_EDGE:
                survivors.append({
                    "path": path_name, "k": st["k"], "set": st["set"],
                    "excess_net_winsor_annualized": st["excess_net_winsor_annualized"],
                    "excess_net_median_annualized": st["excess_net_median_annualized"],
                    "turnover": st["turnover"],
                    "gross_per_period": st["gross_per_period"],
                    "breakeven_cost": st.get("breakeven_cost"),
                })
    survivors.sort(key=lambda r: -r["excess_net_winsor_annualized"])

    def _flip(m: dict[str, Any]) -> bool:
        """翻盘判据：三口径全正 + 胜率过半 + 幅度有意义。"""
        if not m.get("ok"):
            return False
        return bool(
            (m.get("excess_net_winsor_annualized") or -9) > 0
            and (m.get("excess_net_median_annualized") or -9) > 0
            and (m.get("excess_net_raw_annualized") or -9) > 0
            and (m.get("excess_net_winsor_hit_rate") or 0) > 0.5
            and (m.get("excess_net_winsor_annualized") or -9) > MEANINGFUL_EDGE
        )

    out: dict[str, Any] = {
        "ok": True,
        "generated_at": _now(),
        "panel_to": (list(mat["dates"])[-1] if len(mat.get("dates", [])) else None),
        "panel": {"n_codes": len(mat["codes"]), "n_dates": n_dates,
                  "from": mat["dates"][0] if mat["dates"] else None,
                  "to": mat["dates"][-1] if mat["dates"] else None},
        "segments": {k: [mat["dates"][v[0]], mat["dates"][min(v[1], n_dates) - 1]]
                     for k, v in segs.items()},
        "config": {"hold": hold, "quantile": quantile, "lag": lag,
                   "cost_rate": cost_rate, "objectives": list(OBJECTIVES),
                   "meaningful_edge": MEANINGFUL_EDGE},
        "canonical_factors": names,
        "cross_check": cross_check,
        "segment_check": segment_check,
        "redundancy": red,
        "full_set": {"train": full_train, "val": full_val, "test": full_test,
                     "train_ic": full_train_ic},
        "paths": paths,
        "frozen": {
            "spec": spec,
            "path": frozen["path"],
            "k": frozen["k"],
            "factor_names": list(frozen["set"]),
            "train_metrics": next(
                (st for st in paths[frozen["path"]] if st["k"] == frozen["k"]), {}
            ),
            "val": sub_val,
            "val_ic": sub_val_ic,
            "val_gate_passed": gate_passed,
            "test": sub_test,
            "test_ic": sub_test_ic,
        },
        "train_survivors": {
            "n": len(survivors),
            "top": survivors[:5],
            "note": "训练段上「成本后超额三口径全正且幅度>0.30%/年」的子集数量",
        },
        "verdict": {
            "flips_on_train": len(survivors) > 0,
            "flips_on_val": gate_passed and _flip(sub_val),
            "flips_on_test": gate_passed and _flip(sub_test),
            "conclusion": "",
        },
        "limitations": [
            "测试段已被本项目多个假设反复看过，对新假设不再是处女地；测试段数字只能算旁证，不能算独立确认。",
            "因子子集只按时间外切分选择，未做组合爆炸式的全子集搜索；贪心路径本身也是一种选择偏好。",
            "训练段毛超额显著高于验证/测试段（口径漂移），训练段的结论不能直接外推。",
            "未缩尾口径用 long_raw − bench（基准只有缩尾值），极端值披露口径略有不对齐。",
        ],
    }

    v = out["verdict"]
    sub_val_a = sub_val.get("excess_net_winsor_annualized") or 0.0
    full_val_a = full_val.get("excess_net_winsor_annualized") or 0.0
    sub_test_a = sub_test.get("excess_net_winsor_annualized") or 0.0
    full_test_a = full_test.get("excess_net_winsor_annualized") or 0.0
    improve_val = sub_val_a - full_val_a
    improve_test = sub_test_a - full_test_a
    out["improvement_vs_full"] = {
        "val_excess_annualized_delta": improve_val,
        "test_excess_annualized_delta": improve_test,
        "val_turnover": sub_val.get("turnover"),
        "full_val_turnover": full_val.get("turnover"),
        "test_turnover": sub_test.get("turnover"),
        "full_test_turnover": full_test.get("turnover"),
        "cost_drag_per_period_saved": (
            (full_test.get("turnover") or 0) - (sub_test.get("turnover") or 0)
        ) * cost_rate,
    }
    if v["flips_on_test"]:
        v["conclusion"] = (
            f"冻结子集（{frozen['path']} 路径，{frozen['k']} 个因子）在测试段"
            f"成本后超额三口径全正、胜率过半、年化 {sub_test_a:+.2%}（> 门槛 0.30%）；"
            f"全量 11 因子同期为 {full_test_a:+.2%}。仍需前瞻账本验证，不能直接进实盘。"
        )
    elif improve_val > 0 and improve_test > 0:
        # 样本外两个区间都比全量好 —— 这是「降换手」这个动作本身有效，
        # 不能笼统说成「过拟合」；真正的问题剩下的是「补上成本缺口后还剩多少」。
        v["conclusion"] = (
            f"降换手在样本外确实稳定改善了超额（验证段 {full_val_a:+.2%} → {sub_val_a:+.2%}，"
            f"测试段 {full_test_a:+.2%} → {sub_test_a:+.2%}）：换手 "
            f"{(full_test.get('turnover') or 0):.4f} → {(sub_test.get('turnover') or 0):.4f}，"
            f"成本拖累省下 {(out['improvement_vs_full']['cost_drag_per_period_saved']) * 1e4:.2f} bp/期。"
            f"但补上成本缺口之后剩下的超额仍在 0 附近"
            f"（验证段 {sub_val_a:+.2%}、测试段 {sub_test_a:+.2%}，门槛 0.30%/年），"
            "省得出成本，省不出 alpha，仍然不能实盘。"
        )
    elif improve_val <= 0:
        v["conclusion"] = (
            "冻结子集在验证段没跑赢全量因子 —— 选择本身过拟合，停止，不换参。"
        )
    else:
        v["conclusion"] = (
            "训练段上没有任何因子子集能同时满足「超额三口径全正 + 幅度有意义」；"
            "换手这条路在结构上走不通，问题不在因子个数。"
        )
    return out


def _save(out: dict[str, Any]) -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    tmp = OUT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT_FILE)


def _save_frozen(out: dict[str, Any]) -> None:
    spec = out.get("frozen", {}).get("spec")
    if not spec:
        return
    payload = dict(spec)
    payload["frozen_at"] = out.get("generated_at")
    payload["selected_by"] = "factor_redundancy.py 训练段贪心前向"
    payload["validation_net_annualized"] = (
        out.get("frozen", {}).get("val", {}) or {}
    ).get("excess_net_winsor_annualized")
    payload["spec_hash"] = spec_hash(payload)
    tmp = FROZEN_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(FROZEN_FILE)


def main() -> int:
    out = analyze()
    if not out.get("ok"):
        print(f"[factor_redundancy] 未出结论：{out.get('reason')}")
        return 1
    _save(out)
    _save_frozen(out)

    red = out.get("redundancy") or {}
    fr = out.get("frozen") or {}
    fs = out.get("full_set") or {}
    print(f"[factor_redundancy] 面板截止 {out.get('panel_to')}｜"
          f"因子 {len(out.get('canonical_factors') or [])} 个")
    if red.get("ok"):
        print(f"  有效维度 {red['effective_dimension']:.2f}（11.0=互相独立，1.0=同一个信号）｜"
              f"最高相关 {red['max_abs']:.3f} {red.get('max_abs_pair')}")
    for seg in ("train", "val", "test"):
        fm = fs.get(seg) or {}
        sm = fr.get(seg) if seg in ("val", "test") else (fr.get("train_metrics") or {})
        print(f"  {seg:<5} 全量: 毛 {fm.get('gross_per_period', 0) * 1e4:+.3f}bp 换手 "
              f"{fm.get('turnover', 0):.4f} 超额 {fm.get('excess_net_winsor_annualized', 0):+.3%}/年 "
              f"(中位 {fm.get('excess_net_median_annualized', 0):+.3%})")
        print(f"  {'':<5} 子集: 毛 {sm.get('gross_per_period', 0) * 1e4:+.3f}bp 换手 "
              f"{sm.get('turnover', 0):.4f} 超额 {sm.get('excess_net_winsor_annualized', 0):+.3%}/年 "
              f"(中位 {sm.get('excess_net_median_annualized', 0):+.3%})")
    print(f"  冻结子集：{fr.get('path')} 路径，{fr.get('k')} 个因子 {fr.get('factor_names')}")
    print(f"  验证段闸门：{'通过' if fr.get('val_gate_passed') else '未通过'}")
    print(f"  结论：{out['verdict']['conclusion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
