"""预测记录与复盘闭环。

这是整个系统的核心闭环：
    每天生成【预测】 → 固化进快照 → 未来某日用真实数据【判定】 →
    统计【命中率 / 校准度 / 稳定性】→ 反过来约束明天的仓位与选股。

三个必须坚持的原则：
1. **预测必须先落盘再发生**。判定用的价格只能来自预测日之后的交易日，
   绝不能用预测当天的数据（那是自证）。
2. **预测要给出概率，不只是方向**。只有方向没有概率，就无法计算校准度
   （Brier score），也无法知道模型是"自信地错"还是"犹豫地错"。
3. **样本不足就不下结论**。前 5 个样本的命中率没有统计意义，
   报告里必须写"样本不足，暂不评价"。

预测标的分层：
    - 指数方向（A股大盘）：T+5 交易日沪深300 涨/跌
    - 主题方向：四个股票池主题的等权组合 T+5 涨/跌
    - 个股方向：组合内每只标的 T+5 涨/跌
    - 商品/外汇：只记录方向，不作为仓位依据
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

# 预测记录目录（与快照同源，便于一起固化）
PRED_DIR = Path(__file__).resolve().parent / "data" / "predictions"
# 判定结果目录。**与预测记录分开存放**：预测文件写一次就永久不可变，
# 判定结果另存一份，这样"预测先落盘、事后不可改"才是文件级的事实，
# 而不是靠大家自觉。
PRED_RESOLVED_DIR = PRED_DIR / "resolved"

# 预测记录的不可变字段：一旦落盘就永远不许改。
# 判定阶段只允许写 resolved / outcome；动这些字段等于事后改答案。
IMMUTABLE_FIELDS = (
    "day",
    "target",
    "target_code",
    "kind",
    "horizon_days",
    "prob_up",
    "direction",
    "reason",
    "composite_rank",
)


# ------------------------------------------------------------------ 预测生成


def make_prediction(
    *,
    day: str,
    horizon_days: int,
    target: str,
    prob_up: float,
    reason: str,
    evidence: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成一条预测记录。

    prob_up ∈ [0,1]，是"目标在 horizon_days 个交易日后上涨"的概率。
    概率必须来自可复现的计算，不得拍脑袋；没有依据时应给 0.5 并说明。

    extra 放目标代码、类别、排名等附加字段（会一并计入指纹）。
    """
    if not (0.0 <= prob_up <= 1.0):
        raise ValueError(f"prob_up 必须在 [0,1]，收到 {prob_up}")
    rec = {
        "day": day,
        "target": target,
        "horizon_days": horizon_days,
        "prob_up": round(float(prob_up), 4),
        "direction": "看多" if prob_up > 0.5 else ("看空" if prob_up < 0.5 else "中性"),
        "reason": reason,
        "evidence": evidence or {},
        "resolved": False,
        "outcome": None,
    }
    if extra:
        rec.update(extra)
    return rec


def fingerprint(preds: list[dict]) -> str:
    """预测记录的指纹：只覆盖不可变字段。

    判定阶段回填 outcome 不会改变指纹；如果有人事后改了 prob_up，
    指纹立刻对不上 —— 这比"约定不许改"可靠得多，是可执行的约束。
    """
    h = hashlib.sha256()
    for p in preds:
        payload = {k: p.get(k) for k in IMMUTABLE_FIELDS}
        h.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def verify_fingerprint(payload: dict[str, Any]) -> tuple[bool, str | None]:
    """校验落盘记录的指纹。返回 (是否可信, 不可信的原因)。"""
    meta = payload.get("meta") or {}
    expected = meta.get("fingerprint")
    if not expected:
        return False, "记录里没有指纹（老版本或被手工改过），拒绝回填判定结果"
    actual = fingerprint(payload.get("predictions") or [])
    if actual != expected:
        return False, (
            f"指纹不一致（落盘 {str(expected)[:12]} ≠ 当前 {actual[:12]}），"
            "预测记录在落盘后被改动过"
        )
    return True, None


def save_predictions(
    day: str,
    preds: list[dict],
    meta: dict[str, Any] | None = None,
) -> Path:
    """落盘预测记录。**只允许写一次** —— 已有记录时拒绝覆盖。

    meta 里会写进指纹；判定回填见 save_resolutions。
    """
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{day}.json"
    if path.exists():
        raise FileExistsError(f"{path} 已存在，预测记录不允许事后重做")
    m = dict(meta or {})
    m["fingerprint"] = fingerprint(preds)
    payload = {"day": day, "predictions": preds, "meta": m}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def save_resolutions(
    day: str,
    outcomes: dict[str, dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> Path:
    """落盘判定结果（写到独立目录，不动预测文件）。

    outcomes: {target_code: {"fwd_return":..., "actual_up":..., "hit":...}}
    允许覆盖 —— 随着时间推移，会有更多预测到期，判定结果需要更新。
    但预测文件本身永远不会被改写。
    """
    PRED_RESOLVED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_RESOLVED_DIR / f"{day}.json"
    m = dict(meta or {})
    m["n_resolved"] = len(outcomes)
    payload = {"day": day, "outcomes": outcomes, "meta": m}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load_resolutions(day: str) -> dict[str, Any]:
    path = PRED_RESOLVED_DIR / f"{day}.json"
    if not path.exists():
        return {"day": day, "outcomes": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"day": day, "outcomes": {}}


def apply_outcomes(
    preds: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """把已落盘的判定结果贴回预测记录（不改任何不可变字段）。"""
    if not outcomes:
        return preds
    out: list[dict[str, Any]] = []
    for p in preds:
        code = p.get("target_code") or p.get("target")
        oc = outcomes.get(code)
        if oc and not p.get("resolved"):
            q = dict(p)
            q["resolved"] = True
            q["outcome"] = dict(oc)
            out.append(q)
        else:
            out.append(p)
    return out


def load_predictions(day: str) -> dict[str, Any]:
    path = PRED_DIR / f"{day}.json"
    if not path.exists():
        return {"day": day, "predictions": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"day": day, "predictions": []}


# ------------------------------------------------------------------ 判定


def _fwd_return(prices: dict[str, float], code: str, start: str, horizon: int) -> float | None:
    """从 start 日往后 horizon 个交易日的收益率。价格表需按日期升序。"""
    dates = [d for d in sorted(prices) if d >= start]
    if len(dates) <= horizon:
        return None
    p0 = prices.get(dates[0])
    p1 = prices.get(dates[horizon])
    if not p0 or not p1:
        return None
    return p1 / p0 - 1.0


def resolve_predictions(
    predictions: list[dict],
    price_series: dict[str, dict[str, float]],
) -> list[dict]:
    """用真实行情判定预测。

    price_series: {target_code: {date: price}}，date 升序。
    判定只使用预测日之后的价格 —— 这是闭环不塌陷的关键。
    """
    out: list[dict] = []
    for p in predictions:
        if p.get("resolved"):
            out.append(p)
            continue
        series = price_series.get(p.get("target_code") or p.get("target"))
        if not series:
            continue
        r = _fwd_return(series, p.get("target"), p["day"], int(p.get("horizon_days", 5)))
        if r is None:
            continue
        actual_up = r > 0
        predicted_up = p["prob_up"] > 0.5
        q = dict(p)
        q["resolved"] = True
        q["outcome"] = {
            "fwd_return": round(r, 6),
            "actual_up": actual_up,
            "hit": bool(actual_up == predicted_up),
            "close_return_pct": round(r * 100, 3),
        }
        out.append(q)
    return out


# ------------------------------------------------------------------ 统计


def brier_score(probs: list[float], outcomes: list[bool]) -> float | None:
    """Brier 分数：预测概率与实际结果的均方误差。越小越好（0=完美，0.25=全是0.5）。"""
    if not probs or len(probs) != len(outcomes):
        return None
    return sum((p - (1.0 if o else 0.0)) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def calibration_bins(probs: list[float], outcomes: list[bool]) -> list[dict]:
    """把预测按概率分桶，看"说 70% 的时候，实际是不是 70% 涨"。

    这是判断模型是否"知道自己知道什么"的关键。
    """
    if not probs:
        return []
    buckets = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    out: list[dict] = []
    for lo, hi in buckets:
        sel = [
            (p, o) for p, o in zip(probs, outcomes)
            if (p >= lo and (p < hi or (hi > 1.0 and p <= 1.0)))
        ]
        if not sel:
            continue
        out.append({
            "range": f"{lo:.0%}-{min(hi, 1.0):.0%}",
            "n": len(sel),
            "avg_prob": sum(p for p, _ in sel) / len(sel),
            "actual_rate": sum(1 for _p, o in sel if o) / len(sel),
        })
    return out


def summarize(resolved: list[dict], *, min_samples: int = 5) -> dict[str, Any]:
    """汇总预测表现。样本不足时明确说"不评价"，不硬给数字。"""
    done = [p for p in resolved if p.get("resolved") and p.get("outcome")]
    if not done:
        return {
            "n": 0, "sufficient": False,
            "note": "尚无已到期的预测，暂不评价命中率。",
        }
    probs = [p["prob_up"] for p in done]
    outs = [bool(p["outcome"]["actual_up"]) for p in done]
    hits = sum(1 for p in done if p["outcome"]["hit"])
    n = len(done)
    hit_rate = hits / n
    bs = brier_score(probs, outs)
    bins = calibration_bins(probs, outs)
    # 基准：全押同一方向的命中率（多数方向），用于判断模型是否只是"猜对了大方向"
    base = max(sum(outs), n - sum(outs)) / n
    return {
        "n": n,
        "sufficient": n >= min_samples,
        "hit_rate": hit_rate,
        "brier": bs,
        "baseline_hit_rate": base,
        "edge_vs_baseline": hit_rate - base,
        "avg_prob": sum(probs) / n,
        "actual_rate": sum(outs) / n,
        "avg_fwd_return": sum(p["outcome"]["fwd_return"] for p in done) / n,
        "calibration": bins,
        "note": (
            f"已判定 {n} 条预测，方向命中率 {hit_rate:.1%}，"
            f"Brier {bs:.3f}（0.25=无信息量基准）。"
            if n >= min_samples
            else f"仅 {n} 条已判定，样本不足（需 ≥{min_samples}），暂不评价命中率。"
        ),
    }


def by_target(resolved: list[dict], *, min_samples: int = 5) -> dict[str, dict]:
    """按预测标的分组统计。"""
    groups: dict[str, list[dict]] = {}
    for p in resolved:
        if p.get("resolved") and p.get("outcome"):
            groups.setdefault(p.get("target", "?"), []).append(p)
    return {k: summarize(v, min_samples=min_samples) for k, v in groups.items()}
