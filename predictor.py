"""每日预测生产者 + 事后判定器（零 LLM，可复现）。

这个模块补上整个系统最关键的缺口：**闭环里「预测」这一环**。
此前 prediction.py 只有「判定」和「统计」，没有生产者 ——
预测目录永远是空的，G4 门禁在结构上就不可能通过。

三条硬约束（每一条都由代码强制，不是靠约定）：

1. **只用预测日之前的数据训练**
   训练样本、标准化统计量、校准器样本，全部来自 *严格早于预测日* 的日期。
   预测日当天的因子值只用于打分；它的未来收益正是要预测的对象，
   在预测时根本不存在。因子集合本身也在截断后的面板上重新发现，
   所以「选因子」这件事也不可能用到预测日之后的数据。

2. **先落盘，后判定**
   `predict` 先写出 data/predictions/{day}.json，已有记录时拒绝覆盖
   （预测不允许事后重做）。`judge` 才用真实行情回填 outcome，
   而且回填前必须通过指纹校验 —— 改了 prob_up 就会被当场拒绝。

3. **零 LLM**
   全部是确定性计算（固定随机种子），配额再紧也能跑。

预测分层：
    市场级    全市场等权指数 T+H 上涨概率
    个股级    面板内每只标的 T+H 绝对上涨概率（普查，不做选择性记录）
    关注标的  按合成信号排名前 top_n 只（规则事前固定，与结果无关）

统计口径用的是**全量普查**，不是挑出来的样本：
命中率相对基准的超额才经得起检验；挑样本会自己骗自己。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import factorlab
import ooslab
import prediction
import probmodel

ROOT = Path(__file__).resolve().parent
# 预测目录的**唯一所有者**是 prediction.PRED_DIR。
# 以前这里写死 `PRED_DIR = prediction.PRED_DIR`（import 时绑定成一个独立引用），
# 于是外部只改 prediction.PRED_DIR 时，predictor 仍然打在真实目录上：
# generate_predictions(force=True) 里的 path.unlink() 会把真实记录删掉
# （实测误删过 2019-06-19 一天的预测）。现在所有取目录的地方都实时读
# prediction.PRED_DIR，改一处就全链路一致。
SUPERSEDED_DIRNAME = "_superseded"


def _pred_dir() -> Path:
    return prediction.PRED_DIR


def _archive_existing(path: Path) -> Path | None:
    """force 覆盖前把旧记录改名归档 —— **绝不删除**。

    预测记录一旦落盘就是不可变证据。历史上这里直接 path.unlink()，
    结果 force 重算把真实记录删掉了（实测误删 2019-06-19 一天）。
    改名归档到 _superseded/ 子目录，既让新记录能落盘，又保留原始证据。
    放在子目录里也不会被 glob("*.json") 当成又一次预测重复统计。
    """
    if not path.exists():
        return None
    archive_dir = _pred_dir() / SUPERSEDED_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / (
        f"{path.stem}.superseded-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    path.replace(archived)
    return archived

# 闭环统计缓存：报告每次都要读全部预测记录（跨年可达百 MB），
# 重算一遍太慢。判定完写一份缓存，报告优先用新鲜的缓存。
LOOP_CACHE = ROOT / "data" / "research" / "loop_stats.json"

# 市场级预测的虚拟代码（不是真实标的，只是等权指数的代号）
MKT_TARGET = "MKT_EQW"
MKT_NAME = "全市场等权"

RANDOM_SEED = ooslab.RANDOM_SEED
# 训练/校准切分：前 70% 训练，后 30% 拟合校准器（都早于预测日）
TRAIN_FRAC = 0.70
# 每个交易日最多抽样多少只进入训练集（固定种子，可复现）
DEFAULT_MAX_PER_DATE = 200
# 最多用几个因子做概率模型特征
MAX_FACTORS = 8
# 关注标的数量
DEFAULT_TOP_N = 20
# 训练样本下限，低于此不出概率（样本不足就不给概率）
MIN_TRAIN_ROWS = 5000
MIN_MARKET_ROWS = 200


# ------------------------------------------------------------------ 名称


def load_names() -> dict[str, str]:
    """secid → 中文名。读不到就返回空字典（显示代码，不编名字）。"""
    for p in (ROOT / "data" / "universe.json"), Path("/tmp/universe.json"):
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict) and d:
                    return {str(k): str(v) for k, v in d.items()}
            except Exception:  # noqa: BLE001
                continue
    return {}


def display_name(code: str, names: dict[str, str]) -> str:
    return names.get(code) or code


# ------------------------------------------------------------------ 面板工具


def truncate_panel(mat: dict[str, Any], day_idx: int) -> dict[str, Any]:
    """把面板截断到预测日（含）。截断之后的一切计算都看不到未来。"""
    sl = slice(0, day_idx + 1)
    return {
        "dates": list(mat["dates"][sl]),
        "codes": list(mat["codes"]),
        "close": mat["close"][sl],
        "amount": mat["amount"][sl] if "amount" in mat else np.full(mat["close"][sl].shape, np.nan),
        "factors": {k: v[sl] for k, v in mat["factors"].items()},
    }


def equal_weight_index(mat: dict[str, Any]) -> dict[str, Any]:
    """全市场等权指数：每日横截面平均收益复利成净值。

    这就是回测里那个"全市场等权"基准的同一条曲线 ——
    市场级预测用它做标签，判定也用它算前瞻收益，两边口径必须一致。

    日收益用「每只股票当日相对前一日的收益」的横截面平均；
    某天没有涨跌停/停牌数据导致全截面缺失时，该日收益记 0（不编造）。
    """
    close = np.asarray(mat["close"], dtype=float)
    T = close.shape[0]
    if T < 2:
        return {"dates": list(mat["dates"]), "level": np.ones(T), "daily_ret": np.zeros(T)}
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = close[1:] / close[:-1] - 1.0
    rets[~np.isfinite(rets)] = np.nan
    cnt = np.sum(np.isfinite(rets), axis=1)
    tot = np.nansum(np.where(np.isfinite(rets), rets, 0.0), axis=1)
    daily = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    level = np.ones(T)
    level[1:] = np.cumprod(1.0 + daily)
    return {"dates": list(mat["dates"]), "level": level, "daily_ret": daily}


def market_feature_matrix(
    mat: dict[str, Any],
    mkt: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """市场级特征矩阵 [T, F]。

    特征 = 各因子当日的横截面均值（市场整体处在因子的什么位置）
          + 等权指数自身的 5 日 / 20 日动量。

    全是"当天收盘就能看到"的信息，没有任何未来函数。
    """
    feats: list[np.ndarray] = []
    names: list[str] = []
    for fname in factorlab.FACTOR_NAMES:
        f = mat["factors"].get(fname)
        if f is None:
            continue
        f = np.asarray(f, dtype=float)
        cnt = np.sum(np.isfinite(f), axis=1)
        tot = np.nansum(np.where(np.isfinite(f), f, 0.0), axis=1)
        feats.append(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan))
        names.append(f"cross_{fname}")
    level = np.asarray(mkt["level"], dtype=float)
    for lb in (5, 20):
        mom = np.full(len(level), np.nan)
        if len(level) > lb:
            with np.errstate(invalid="ignore", divide="ignore"):
                mom[lb:] = level[lb:] / level[:-lb] - 1.0
        feats.append(mom)
        names.append(f"idx_mom_{lb}")
    if not feats:
        return np.empty((len(mkt["dates"]), 0)), []
    return np.stack(feats, axis=1), names


def build_market_dataset(
    mat: dict[str, Any],
    mkt: dict[str, Any],
    *,
    horizon: int,
) -> dict[str, Any]:
    """市场级 (特征, 标签) 数据集：标签 = 等权指数 T+H 是否上涨。"""
    X, names = market_feature_matrix(mat, mkt)
    level = np.asarray(mkt["level"], dtype=float)
    T = len(level)
    fwd = np.full(T, np.nan)
    if T > horizon:
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd[: T - horizon] = level[horizon:] / level[: T - horizon] - 1.0
    valid = np.isfinite(X).all(axis=1) & np.isfinite(fwd)
    idx = np.where(valid)[0]
    return {
        "X": X[idx],
        "y": (fwd[idx] > 0).astype(float),
        "date_idx": idx,
        "fwd": fwd,
        "names": names,
    }


def rank_factors_train(
    mat: dict[str, Any],
    seg: tuple[int, int],
    *,
    horizon: int,
) -> list[dict[str, Any]]:
    """在指定区间（必须是训练段）给因子按 |IC| 排名。

    与 discover_factors 的区别：这里不要求显著，只排顺序。
    概率模型的特征集只需要在训练段里定下来，不需要每只都显著；
    但"显著的那些"优先用。
    """
    out: list[dict[str, Any]] = []
    for name in factorlab.FACTOR_NAMES:
        if name not in mat["factors"]:
            continue
        series = ooslab.ic_series_on_segment(
            mat, name, seg, horizon=horizon, min_cross=20
        )
        if len(series) < 20:
            continue
        mean, t = factorlab.newey_west(series)
        if mean is None:
            continue
        out.append({
            "name": name,
            "mean": float(mean),
            "t": float(t) if t is not None else None,
            "n": int(len(series)),
            "significant": bool(t is not None and abs(t) > 1.96),
        })
    out.sort(key=lambda r: -abs(r["mean"]))
    return out


# ------------------------------------------------------------------ 概率模型


def _fit_calibrated(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 200,
) -> dict[str, Any] | None:
    """训练 + 校准（保序回归），返回可直接用于打分的模型包。

    时间切分：前 TRAIN_FRAC 训练，后 30% 只用来拟合校准器。
    校准器见过验证段，测试段/未来数据它一次都没碰过。
    """
    n = len(y)
    if n < 50:
        return None
    tr_end = int(n * TRAIN_FRAC)
    if tr_end < 30 or n - tr_end < 20:
        return None
    Xs, mu, sd = ooslab.standardize_by_train(X, tr_end)
    model = probmodel.LogisticRegression(l2=l2, max_iter=max_iter)
    model.fit(Xs[:tr_end], y[:tr_end])
    raw_val = model.predict_proba(Xs[tr_end:])
    iso = probmodel.fit_isotonic(raw_val, y[tr_end:])
    return {
        "model": model,
        "iso": iso,
        "mu": mu,
        "sd": sd,
        "n_train": int(tr_end),
        "n_val": int(n - tr_end),
        "converged": bool(model.converged_),
    }


def _predict_with(bundle: dict[str, Any], X: np.ndarray) -> np.ndarray:
    Xs = (X - bundle["mu"]) / bundle["sd"]
    raw = bundle["model"].predict_proba(Xs)
    return probmodel.apply_isotonic(bundle["iso"], raw)


# ------------------------------------------------------------------ 生成预测


def _locked_horizon(default: int = 5) -> int:
    """沿用样本外实验锁定的前瞻窗口，保持全链路口径一致。"""
    p = ROOT / "data" / "research" / "oos_latest.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        h = d.get("horizon")
        if isinstance(h, int) and h > 0:
            return h
    except Exception:  # noqa: BLE001
        pass
    return default


def generate_predictions(
    *,
    day: str | None = None,
    horizon: int | None = None,
    max_per_date: int = DEFAULT_MAX_PER_DATE,
    seed: int = RANDOM_SEED,
    top_n: int = DEFAULT_TOP_N,
    panel: dict[str, Any] | None = None,
    force: bool = False,
    mode: str = "live",
) -> dict[str, Any]:
    """为某个交易日生成预测并落盘。返回结果摘要。

    mode="live"   实时预测：给当天用（生产口径）
    mode="replay" 回溯自检：用同一套因果流程在历史日期上重放，
                  用来端到端验证闭环能跑通。两种口径在统计里分开报。
    """
    t0 = time.time()
    mat = panel if panel is not None else ooslab.load_panel()
    dates = list(mat["dates"])
    if not dates:
        return {"ok": False, "reason": "面板为空"}

    day = day or dates[-1]
    if day not in dates:
        return {
            "ok": False,
            "reason": f"面板里没有 {day}（面板区间 {dates[0]} → {dates[-1]}）",
        }
    day_idx = dates.index(day)

    path = _pred_dir() / f"{day}.json"
    if path.exists() and not force:
        return {
            "ok": False,
            "day": day,
            "reason": f"{day} 已有预测记录，预测不允许事后重做（如确需重算请加 --force）",
            "path": str(path),
        }

    horizon = int(horizon or _locked_horizon(5))

    # ---- 截断：之后的一切计算都看不到预测日之后的数据
    trunc = truncate_panel(mat, day_idx)
    segs = ooslab.split_segments(len(trunc["dates"]))
    train_seg = segs["train"]

    # ---- 因子集合：在截断后的面板上重新发现（只用训练段）
    ranked = rank_factors_train(trunc, train_seg, horizon=horizon)
    sig = [r for r in ranked if r["significant"]][:MAX_FACTORS]
    if sig:
        factor_names = [r["name"] for r in sig]
        selection = "train_significant"
    else:
        factor_names = [r["name"] for r in ranked[:MAX_FACTORS]]
        selection = "train_top_by_abs_ic_fallback"
    if not factor_names:
        return {"ok": False, "reason": "训练段没有任何可用因子，无法建模"}
    weight_map = {r["name"]: (1.0 if r["mean"] > 0 else -1.0) for r in ranked}

    # ---- 个股级概率模型（标签 = T+H 绝对上涨）
    ds = ooslab.build_probability_dataset(
        trunc, factor_names, horizon=horizon, label="abs",
        max_per_date=max_per_date, seed=seed,
    )
    if not ds.get("ok"):
        return {"ok": False, "reason": f"概率数据集构建失败：{ds.get('reason')}"}
    if len(ds["y"]) < MIN_TRAIN_ROWS:
        return {
            "ok": False,
            "reason": f"训练样本仅 {len(ds['y'])} 条（需 ≥{MIN_TRAIN_ROWS}），样本不足不出概率",
        }
    stock_bundle = _fit_calibrated(ds["X"], ds["y"])
    if stock_bundle is None:
        return {"ok": False, "reason": "概率模型切分后训练/验证样本不足"}

    # ---- 预测日的个股特征
    zcols = [ooslab.cs_zscore(trunc["factors"][n]) for n in factor_names]
    feats = np.stack([z[day_idx] for z in zcols], axis=1)
    ok = np.isfinite(feats).all(axis=1)
    codes = list(trunc["codes"])
    codes_ok = [codes[i] for i in np.where(ok)[0]]
    probs = _predict_with(stock_bundle, feats[ok]) if codes_ok else np.array([])

    # ---- 合成信号排名（方向由训练段 IC 符号决定）
    weights = [weight_map.get(n, 1.0) for n in factor_names]
    score = ooslab.composite_score(trunc, factor_names, weights)
    s_day = np.asarray(score[day_idx], dtype=float)
    score_ok = np.isfinite(s_day)
    rank_order = np.argsort(-np.where(score_ok, s_day, -np.inf), kind="mergesort")
    rank_of: dict[int, int] = {}
    pos = 0
    for j in rank_order:
        if score_ok[j]:
            pos += 1
            rank_of[int(j)] = pos
    n_ranked = int(score_ok.sum())

    # ---- 市场级概率模型
    mkt = equal_weight_index(trunc)
    mds = build_market_dataset(trunc, mkt, horizon=horizon)
    mkt_prob = None
    mkt_bundle = None
    if len(mds["y"]) >= MIN_MARKET_ROWS:
        mkt_bundle = _fit_calibrated(mds["X"], mds["y"], max_iter=300)
    if mkt_bundle is not None:
        Xall, _names = market_feature_matrix(trunc, mkt)
        row = Xall[day_idx: day_idx + 1]
        if np.isfinite(row).all():
            mkt_prob = float(_predict_with(mkt_bundle, row)[0])

    # ---- 组装预测记录
    names = load_names()
    preds: list[dict[str, Any]] = []
    for code, p in zip(codes_ok, probs):
        ci = codes.index(code)
        rk = rank_of.get(ci)
        label = display_name(code, names)
        rec = prediction.make_prediction(
            day=day,
            horizon_days=horizon,
            target=f"{label} {code}",
            prob_up=float(p),
            reason=(
                f"横截面合成信号排名 {rk}/{n_ranked}"
                f"（发现期因子等权 z-score 合成）；"
                f"模型给出 T+{horizon} 交易日绝对上涨概率 {float(p):.2%}"
                if rk else
                f"合成信号无有效排名；模型给出 T+{horizon} 交易日绝对上涨概率 {float(p):.2%}"
            ),
            evidence={
                "composite_score": round(float(s_day[ci]), 4) if rk else None,
                "n_ranked": n_ranked,
            },
            extra={
                "target_code": code,
                "kind": "stock",
                "composite_rank": rk,
            },
        )
        preds.append(rec)

    if mkt_prob is not None:
        mom20 = None
        lvl = mkt["level"]
        if day_idx >= 20 and lvl[day_idx - 20]:
            mom20 = round(float(lvl[day_idx] / lvl[day_idx - 20] - 1.0), 4)
        preds.append(
            prediction.make_prediction(
                day=day,
                horizon_days=horizon,
                target=MKT_NAME,
                prob_up=mkt_prob,
                reason=(
                    f"全市场等权指数 T+{horizon} 交易日上涨概率 {mkt_prob:.2%}；"
                    f"等权 20 日动量 {mom20:+.2%}" if mom20 is not None else
                    f"全市场等权指数 T+{horizon} 交易日上涨概率 {mkt_prob:.2%}"
                ),
                evidence={
                    "n_train_rows": int(len(mds["y"])),
                    "idx_mom_20": mom20,
                },
                extra={
                    "target_code": MKT_TARGET,
                    "kind": "market",
                    "composite_rank": None,
                },
            )
        )

    if not preds:
        return {"ok": False, "reason": "没有生成任何预测记录"}

    meta = {
        "mode": mode,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "horizon": horizon,
        "seed": seed,
        "max_per_date": max_per_date,
        "train_frac": TRAIN_FRAC,
        "factor_selection": selection,
        "factor_names": factor_names,
        "factor_ic_train": {r["name"]: round(r["mean"], 5) for r in ranked[:MAX_FACTORS]},
        "n_codes": len(codes_ok),
        "n_ranked": n_ranked,
        "panel": {"from": trunc["dates"][0], "to": trunc["dates"][-1], "n_dates": len(trunc["dates"])},
        "stock_model": {
            "n_train": stock_bundle["n_train"],
            "n_val": stock_bundle["n_val"],
            "converged": stock_bundle["converged"],
            "label": "绝对上涨（fwd>0）",
        },
        "market_model": (
            {
                "n_train": mkt_bundle["n_train"],
                "n_val": mkt_bundle["n_val"],
                "converged": mkt_bundle["converged"],
                "prob_up": mkt_prob,
            } if mkt_bundle is not None else {"available": False}
        ),
        "note": (
            "训练样本、标准化统计量、校准器样本全部来自严格早于预测日的日期；"
            "预测日当天的因子值只用于打分，其未来收益在预测时不存在。"
        ),
    }
    if force and path.exists():
        archived = _archive_existing(path)
        if archived is not None:
            meta["superseded_previous"] = str(archived)
    prediction.save_predictions(day, preds, meta=meta)

    watch = sorted(
        [p for p in preds if p.get("kind") == "stock" and p.get("composite_rank")],
        key=lambda p: p["composite_rank"],
    )[:top_n]

    return {
        "ok": True,
        "day": day,
        "mode": mode,
        "horizon": horizon,
        "path": str(path),
        "n_predictions": len(preds),
        "n_stocks": len(codes_ok),
        "n_market": 1 if mkt_prob is not None else 0,
        "market_prob_up": mkt_prob,
        "watchlist": [
            {"code": p["target_code"], "name": p["target"].split(" ")[0],
             "rank": p["composite_rank"], "prob_up": p["prob_up"]}
            for p in watch
        ],
        "factor_names": factor_names,
        "factor_selection": selection,
        "elapsed_sec": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------------ 判定


def resolve_with_panel(
    preds: list[dict[str, Any]],
    mat: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """用面板行情判定预测。返回 (判定后的记录, 本次新判定条数)。

    判定只使用预测日**之后**的价格：entry 行 = 预测日，exit 行 = 预测日+h。
    预测日当天的收盘价是判定基准（不是收益的一部分），
    收益区间严格是 [预测日, 预测日+h]。
    """
    dates = list(mat["dates"])
    idx_of = {d: i for i, d in enumerate(dates)}
    close = np.asarray(mat["close"], dtype=float)
    codes = list(mat["codes"])
    code_idx = {c: i for i, c in enumerate(codes)}
    mkt = equal_weight_index(mat)
    level = np.asarray(mkt["level"], dtype=float)

    out: list[dict[str, Any]] = []
    newly = 0
    for p in preds:
        if p.get("resolved"):
            out.append(p)
            continue
        day = p.get("day")
        try:
            h = int(p.get("horizon_days", 5))
        except (TypeError, ValueError):
            out.append(p)
            continue
        ei = idx_of.get(day)
        if ei is None:
            out.append(p)
            continue
        xi = ei + h
        if xi >= len(dates):
            out.append(p)
            continue

        r: float | None = None
        if p.get("target_code") == MKT_TARGET:
            if np.isfinite(level[ei]) and np.isfinite(level[xi]) and level[ei] > 0:
                r = float(level[xi] / level[ei] - 1.0)
        else:
            ci = code_idx.get(p.get("target_code"))
            if ci is not None:
                p0 = float(close[ei, ci])
                p1 = float(close[xi, ci])
                if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                    r = p1 / p0 - 1.0
        if r is None:
            out.append(p)
            continue

        actual_up = bool(r > 0)
        predicted_up = float(p.get("prob_up", 0.5)) > 0.5
        q = dict(p)
        q["resolved"] = True
        q["outcome"] = {
            "fwd_return": round(r, 6),
            "actual_up": actual_up,
            "hit": bool(actual_up == predicted_up),
            "close_return_pct": round(r * 100, 3),
            "entry_date": dates[ei],
            "exit_date": dates[xi],
            "resolved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        newly += 1
        out.append(q)
    return out, newly


def judge_all(*, panel: dict[str, Any] | None = None) -> dict[str, Any]:
    """判定所有历史预测，把结果落到独立文件（预测文件永不改写）。

    回填前必须通过指纹校验：预测落盘后被改过的话，拒绝判定并报错 ——
    否则"事后改答案"就会被系统自己掩盖掉。

    已落盘的判定结果**不会被重算覆盖**：后复权价格会随新分红回溯调整，
    重算值可能与当初判定不同；差异会计数并披露，但以当初的判定为准。
    """
    t0 = time.time()
    if not _pred_dir().exists():
        return {"ok": False, "reason": "预测目录不存在"}
    files = sorted(_pred_dir().glob("*.json"))
    if not files:
        return {"ok": False, "reason": "没有任何预测记录"}

    mat = panel if panel is not None else ooslab.load_panel()

    total = 0
    resolved = 0
    newly = 0
    revised = 0
    tampered: list[str] = []
    for fn in files:
        try:
            payload = json.loads(fn.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            tampered.append(f"{fn.name}: 读不出来（{exc}）")
            continue
        ok, why = prediction.verify_fingerprint(payload)
        if not ok:
            tampered.append(f"{fn.name}: {why}")
            continue
        day = payload.get("day") or fn.stem
        preds = payload.get("predictions") or []
        total += len(preds)

        prev = prediction.load_resolutions(day)
        outcomes: dict[str, dict[str, Any]] = dict(prev.get("outcomes") or {})

        fresh, n_new = resolve_with_panel(preds, mat)
        n_rev = 0
        for p in fresh:
            if not p.get("resolved"):
                continue
            code = p.get("target_code") or p.get("target")
            oc = p["outcome"]
            old = outcomes.get(code)
            if old is not None:
                if abs(float(old.get("fwd_return", 0.0)) - float(oc.get("fwd_return", 0.0))) > 1e-9:
                    n_rev += 1
                continue
            outcomes[code] = oc
        revised += n_rev
        newly += n_new
        resolved += len(outcomes)

        if n_new or not prev:
            meta = dict(payload.get("meta") or {})
            meta["judged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta["prediction_fingerprint"] = (payload.get("meta") or {}).get("fingerprint")
            meta["n_predictions"] = len(preds)
            meta["n_revised_vs_saved"] = n_rev
            prediction.save_resolutions(day, outcomes, meta=meta)

    res = {
        "ok": not tampered,
        "n_files": len(files),
        "n_predictions": total,
        "n_resolved": resolved,
        "n_newly_resolved": newly,
        "n_revised_vs_saved": revised,
        "tampered": tampered,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    try:
        save_loop_cache(loop_stats(use_cache=False))
    except Exception as exc:  # noqa: BLE001
        res["cache_error"] = str(exc)
    return res


# ------------------------------------------------------------------ 读取


def list_prediction_days() -> list[str]:
    if not _pred_dir().exists():
        return []
    return sorted(p.stem for p in _pred_dir().glob("*.json"))


def load_day(day: str) -> dict[str, Any]:
    return prediction.load_predictions(day)


def latest_day() -> str | None:
    days = list_prediction_days()
    return days[-1] if days else None


def latest_summary(*, top_n: int = DEFAULT_TOP_N) -> dict[str, Any]:
    """最近一期预测的摘要（日报用）。"""
    day = latest_day()
    if not day:
        return {"available": False, "note": "尚未产生任何预测记录，预测闭环暂未启动。"}
    payload = load_day(day)
    preds = payload.get("predictions") or []
    res = prediction.load_resolutions(day)
    preds = prediction.apply_outcomes(preds, res.get("outcomes") or {})
    meta = payload.get("meta") or {}
    mkt = [p for p in preds if p.get("kind") == "market"]
    stocks = [p for p in preds if p.get("kind") == "stock"]
    stocks_ranked = sorted(
        [p for p in stocks if p.get("composite_rank")],
        key=lambda p: p["composite_rank"],
    )
    watch = stocks_ranked[:top_n]
    return {
        "available": True,
        "day": day,
        "horizon": meta.get("horizon"),
        "generated_at": meta.get("generated_at"),
        "n_predictions": len(preds),
        "n_stocks": len(stocks),
        "factor_names": meta.get("factor_names") or [],
        "factor_selection": meta.get("factor_selection"),
        "market": (
            {
                "prob_up": mkt[0]["prob_up"],
                "direction": mkt[0]["direction"],
                "reason": mkt[0]["reason"],
                "outcome": mkt[0].get("outcome"),
            } if mkt else None
        ),
        "watchlist": [
            {
                "code": p.get("target_code"),
                "name": str(p.get("target", "")).split(" ")[0],
                "rank": p.get("composite_rank"),
                "prob_up": p.get("prob_up"),
                "direction": p.get("direction"),
                "outcome": p.get("outcome"),
            } for p in watch
        ],
        "n_resolved": sum(1 for p in preds if p.get("resolved")),
        "n_pending": sum(1 for p in preds if not p.get("resolved")),
    }


def decile_metrics(stocks: list[dict], n_deciles: int = 10) -> dict[str, Any]:
    """按预测日的合成信号排名分组，算各十分位组合的前瞻收益。

    这是**选股能力**的口径：同一批预测里，信号排前面的组合
    是否比排后面的组合涨得多。跟"大盘涨没涨"无关，天然对冲掉市场方向。

    与 G2 的回测口径一致（同样按信号排序分十分位），所以两处可以互相印证：
    G2 说回测里有效，G4 说生产链上同样有效，才算闭环。
    """
    ranked = [
        p for p in stocks
        if p.get("composite_rank") and p.get("resolved") and p.get("outcome")
    ]
    if len(ranked) < n_deciles * 5:
        return {}
    ranked.sort(key=lambda p: p["composite_rank"])
    n = len(ranked)
    per = n // n_deciles
    groups: list[dict[str, Any]] = []
    for i in range(n_deciles):
        lo = i * per
        hi = lo + per if i < n_deciles - 1 else n
        seg = ranked[lo:hi]
        groups.append({
            "decile": i + 1,
            "n": len(seg),
            "avg_fwd_return": sum(p["outcome"]["fwd_return"] for p in seg) / len(seg),
        })
    all_ret = sum(p["outcome"]["fwd_return"] for p in ranked) / n
    return {
        "n": n,
        "n_deciles": n_deciles,
        "groups": groups,
        "top_decile_return": groups[0]["avg_fwd_return"],
        "bottom_decile_return": groups[-1]["avg_fwd_return"],
        "spread": groups[0]["avg_fwd_return"] - groups[-1]["avg_fwd_return"],
        "equal_weight_return": all_ret,
        "top_minus_bw": groups[0]["avg_fwd_return"] - all_ret,
    }


def portfolio_stats(preds: list[dict], n_deciles: int = 10) -> dict[str, Any]:
    """跨日聚合的组合层指标。"""
    by_day: dict[str, list[dict]] = {}
    for p in preds:
        if p.get("kind") == "market":
            continue
        by_day.setdefault(p.get("day", "?"), []).append(p)
    per_day: list[dict[str, Any]] = []
    for day, stocks in sorted(by_day.items()):
        m = decile_metrics(stocks, n_deciles=n_deciles)
        if m:
            m["day"] = day
            per_day.append(m)
    if not per_day:
        return {"available": False, "n_days": 0,
                "note": "还没有已判定的选股样本，组合层指标暂不可算。"}
    n = len(per_day)

    def _avg(key: str) -> float:
        return sum(d[key] for d in per_day) / n

    spreads = [d["spread"] for d in per_day]
    tops = [d["top_decile_return"] for d in per_day]
    return {
        "available": True,
        "n_days": n,
        "avg_top_decile_return": _avg("top_decile_return"),
        "avg_bottom_decile_return": _avg("bottom_decile_return"),
        "avg_spread": _avg("spread"),
        "median_spread": float(np.median(spreads)),
        "avg_equal_weight_return": _avg("equal_weight_return"),
        "avg_top_minus_bw": _avg("top_minus_bw"),
        "pct_days_spread_positive": sum(1 for s in spreads if s > 0) / n,
        "pct_days_top_positive": sum(1 for t in tops if t > 0) / n,
        "note": (
            "按信号排名分十分位后，头部组合与尾部组合的前瞻收益差。"
            "这是选股能力口径（对冲掉市场方向），与方向命中率口径互为补充。"
        ),
    }


def day_level_stats(preds: list[dict]) -> dict[str, Any]:
    """按天聚合的方向命中统计。

    同一天的几千条样本高度相关（一荣俱荣一损俱损），把 10 万条当成 10 万个
    独立样本会把置信度算爆。所以这里按天看：模型在多少天里跑赢硬币。
    """
    by_day: dict[str, list[dict]] = {}
    for p in preds:
        if p.get("kind") == "market":
            continue
        if p.get("resolved") and p.get("outcome"):
            by_day.setdefault(p.get("day", "?"), []).append(p)
    rows: list[dict[str, Any]] = []
    for day, ps in sorted(by_day.items()):
        n = len(ps)
        if n < 20:
            continue
        hits = sum(1 for p in ps if p["outcome"]["hit"])
        ups = sum(1 for p in ps if p["outcome"]["actual_up"])
        rows.append({
            "day": day,
            "n": n,
            "hit_rate": hits / n,
            "actual_rate": ups / n,
            "avg_fwd_return": sum(p["outcome"]["fwd_return"] for p in ps) / n,
        })
    if not rows:
        return {"available": False, "n_days": 0,
                "note": "还没有按天可聚合的已判定样本。"}
    n = len(rows)
    return {
        "available": True,
        "n_days": n,
        "avg_daily_hit_rate": sum(r["hit_rate"] for r in rows) / n,
        "median_daily_hit_rate": float(np.median([r["hit_rate"] for r in rows])),
        "pct_days_hit_above_50": sum(1 for r in rows if r["hit_rate"] > 0.5) / n,
        "pct_days_actual_up_above_50": sum(1 for r in rows if r["actual_rate"] > 0.5) / n,
        "avg_daily_actual_rate": sum(r["actual_rate"] for r in rows) / n,
        "daily": rows[-30:],
        "note": (
            "按天聚合后的方向命中率。样本同日相关，按天看更接近真实自由度。"
        ),
    }


def _load_loop_cache() -> dict[str, Any] | None:
    """读闭环统计缓存。缓存必须比最新的判定文件新，否则作废重算。"""
    if not LOOP_CACHE.exists():
        return None
    try:
        cached = json.loads(LOOP_CACHE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    cache_mtime = LOOP_CACHE.stat().st_mtime
    stale = False
    for d in (prediction.PRED_RESOLVED_DIR, _pred_dir()):
        if not d.exists():
            continue
        for fn in d.glob("*.json"):
            try:
                if fn.stat().st_mtime > cache_mtime:
                    stale = True
            except OSError:
                stale = True
    if stale:
        return None
    return cached.get("stats") if isinstance(cached, dict) else None


def save_loop_cache(stats: dict[str, Any]) -> Path:
    LOOP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats,
    }
    tmp = LOOP_CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(LOOP_CACHE)
    return LOOP_CACHE


def loop_stats(*, min_samples: int = 5, use_cache: bool = True) -> dict[str, Any]:
    """全量预测的闭环统计。

    **只用已落盘的判定结果**（不现算）：判定一旦落盘就固定，
    现算会随着后复权价格回溯调整而漂移。日报看到的永远是当初的判定。
    """
    if use_cache:
        cached = _load_loop_cache()
        if cached is not None:
            return cached
    days = list_prediction_days()
    if not days:
        return {"available": False, "n": 0, "sufficient": False,
                "note": "尚未产生任何预测记录，预测闭环暂未启动，暂不评价命中率。"}
    preds: list[dict[str, Any]] = []
    for d in days:
        payload = load_day(d)
        recs = payload.get("predictions") or []
        mode = (payload.get("meta") or {}).get("mode") or "live"
        res = prediction.load_resolutions(d)
        recs = prediction.apply_outcomes(recs, res.get("outcomes") or {})
        for p in recs:
            p = dict(p)
            p.setdefault("target_code", p.get("target"))
            p["mode"] = mode
            preds.append(p)
    stats = prediction.summarize(preds, min_samples=min_samples)
    by_target = prediction.by_target(preds, min_samples=min_samples)
    mkt_resolved = [p for p in preds if p.get("resolved") and p.get("kind") == "market"]

    # 实时预测与回溯自检分开统计：两者都是因果生成，但来源必须说清楚
    live = [p for p in preds if (p.get("mode") or "live") == "live"]
    replay = [p for p in preds if p.get("mode") == "replay"]
    live_stats = prediction.summarize(live, min_samples=min_samples)
    replay_stats = prediction.summarize(replay, min_samples=min_samples)
    live_days = sorted({p.get("day") for p in live if p.get("day")})
    replay_days = sorted({p.get("day") for p in replay if p.get("day")})

    return {
        "available": True,
        "n_days": len(days),
        "n_total": len(preds),
        "n": stats.get("n", 0),
        "sufficient": stats.get("sufficient", False),
        "hit_rate": stats.get("hit_rate"),
        "brier": stats.get("brier"),
        "baseline_hit_rate": stats.get("baseline_hit_rate"),
        "edge_vs_baseline": stats.get("edge_vs_baseline"),
        "avg_prob": stats.get("avg_prob"),
        "actual_rate": stats.get("actual_rate"),
        "avg_fwd_return": stats.get("avg_fwd_return"),
        "calibration": stats.get("calibration") or [],
        "by_target": by_target,
        "live": {
            "n_days": len(live_days),
            "days": live_days,
            "n": live_stats.get("n", 0),
            "sufficient": live_stats.get("sufficient", False),
            "hit_rate": live_stats.get("hit_rate"),
            "brier": live_stats.get("brier"),
            "baseline_hit_rate": live_stats.get("baseline_hit_rate"),
            "edge_vs_baseline": live_stats.get("edge_vs_baseline"),
            "note": live_stats.get("note", ""),
        },
        "replay": {
            "n_days": len(replay_days),
            "days": replay_days[:60],
            "n": replay_stats.get("n", 0),
            "sufficient": replay_stats.get("sufficient", False),
            "hit_rate": replay_stats.get("hit_rate"),
            "brier": replay_stats.get("brier"),
            "baseline_hit_rate": replay_stats.get("baseline_hit_rate"),
            "edge_vs_baseline": replay_stats.get("edge_vs_baseline"),
            "avg_fwd_return": replay_stats.get("avg_fwd_return"),
            "note": replay_stats.get("note", ""),
        },
        "market": {
            "n": len(mkt_resolved),
            "hit_rate": (
                sum(1 for p in mkt_resolved if p["outcome"]["hit"]) / len(mkt_resolved)
                if mkt_resolved else None
            ),
            "avg_fwd_return": (
                sum(p["outcome"]["fwd_return"] for p in mkt_resolved) / len(mkt_resolved)
                if mkt_resolved else None
            ),
        },
        "portfolio": portfolio_stats(preds),
        "day_level": day_level_stats(preds),
        "note": stats.get("note", ""),
    }


# ------------------------------------------------------------------ CLI


def main() -> int:
    ap = argparse.ArgumentParser(description="预测生产者 + 事后判定器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("predict", help="为某个交易日生成预测并落盘")
    p1.add_argument("--day", help="预测日（默认取面板最后一天）")
    p1.add_argument("--horizon", type=int, default=0, help="前瞻窗口（默认沿用样本外实验锁定值）")
    p1.add_argument("--max-per-date", type=int, default=DEFAULT_MAX_PER_DATE)
    p1.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p1.add_argument("--seed", type=int, default=RANDOM_SEED)
    p1.add_argument("--mode", choices=["live", "replay"], default="live",
                    help="live=实时预测；replay=回溯自检（历史日期重放）")
    p1.add_argument("--force", action="store_true", help="覆盖已有记录（原则上不允许）")

    p2 = sub.add_parser("judge", help="用真实行情判定历史预测并落盘")
    p2.add_argument("--panel", default="", help="可选：指定面板 npz（调试用）")

    p3 = sub.add_parser("replay", help="对一段历史区间批量回溯自检")
    p3.add_argument("--start", required=True, help="起始日（含）")
    p3.add_argument("--end", required=True, help="结束日（含）")
    p3.add_argument("--step", type=int, default=5, help="每隔几个交易日取一天")
    p3.add_argument("--horizon", type=int, default=0)
    p3.add_argument("--max-per-date", type=int, default=DEFAULT_MAX_PER_DATE)

    sub.add_parser("status", help="打印预测闭环状态")
    sub.add_parser("list", help="列出已有预测的交易日")

    args = ap.parse_args()

    if args.cmd == "predict":
        res = generate_predictions(
            day=args.day,
            horizon=args.horizon or None,
            max_per_date=args.max_per_date,
            seed=args.seed,
            top_n=args.top_n,
            force=args.force,
            mode=args.mode,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("ok") else 1

    if args.cmd == "replay":
        mat = ooslab.load_panel()
        dates = list(mat["dates"])
        lo = dates.index(args.start) if args.start in dates else 0
        hi = dates.index(args.end) if args.end in dates else len(dates) - 1
        if hi < lo:
            print(json.dumps({"ok": False, "reason": "end 早于 start"}, ensure_ascii=False))
            return 1
        picked = [dates[i] for i in range(lo, hi + 1, max(1, args.step))]
        results = []
        for d in picked:
            r = generate_predictions(
                day=d,
                horizon=args.horizon or None,
                max_per_date=args.max_per_date,
                panel=mat,
                mode="replay",
            )
            results.append({
                "day": d,
                "ok": r.get("ok"),
                "n": r.get("n_predictions"),
                "reason": r.get("reason"),
            })
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        n_ok = sum(1 for r in results if r["ok"])
        print(json.dumps({"ok": n_ok > 0, "n_days": len(results), "n_ok": n_ok},
                         ensure_ascii=False, indent=2))
        return 0 if n_ok else 1

    if args.cmd == "judge":
        res = judge_all()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("ok") else 1

    if args.cmd == "status":
        st = loop_stats()
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "list":
        for d in list_prediction_days():
            print(d)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
