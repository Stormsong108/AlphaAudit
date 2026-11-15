"""复盘：用真实行情检验上一期的模拟组合建议。

指标口径（全部基于已落盘的不可变快照 + 后续快照的真实收盘价）：
- 方向命中率：建议方向（偏多/偏空）与实际涨跌方向一致的比例
- 止损触发率：建议后实际收盘跌破止损位的比例
- 目标达成率：建议后实际收盘达到第一档目标（+8%）的比例
- 建议稳定性：相邻两期组合标的集合的 Jaccard 重合度
- 可能偏差：实际收益与建议隐含目标（第一档止盈）之间的系统性偏离
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

MAX_WINDOW = 5  # 每期建议最多向后观察的快照数


def load_snapshots(snapshot_dir: Path) -> list[dict]:
    """读取所有快照，按日期升序。损坏的文件跳过。"""
    out: list[dict] = []
    if not snapshot_dir.exists():
        return out
    for p in sorted(snapshot_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("day"):
            out.append(data)
    out.sort(key=lambda d: d["day"])
    return out


def _prices_by_code(snapshots: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """每个标的 -> [(日期, 价格)] 时间序列。

    只收**交易日**快照的价格：休市日的快照价格是上一个交易日的重复值，
    把它当独立观测点会稀释收益率（同一价格算成"零涨跌的一天"），
    也会让止损/目标触发的观察窗口失真。
    """
    series: dict[str, list[tuple[str, float]]] = {}
    for s in snapshots:
        session = s.get("market_session") or {}
        # 老快照没有 market_session 字段 → 无法判定，按交易日处理
        if session.get("is_trading_day") is False:
            continue
        for code, price in (s.get("prices") or {}).items():
            if isinstance(price, (int, float)):
                series.setdefault(code, []).append((s["day"], float(price)))
    return series


def _jaccard(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    union = a | b
    if not union:
        return None
    return round(len(a & b) / len(union), 4)


def build_review(
    snapshot_dir: Path,
    quotes: dict,
    *,
    lookback: int = 10,
    now: str | None = None,
) -> dict:
    """生成复盘结果。没有历史快照时返回 available=False。"""
    now = now or time.strftime("%Y-%m-%d %H:%M:%S")
    snaps = load_snapshots(snapshot_dir)
    series = _prices_by_code(snaps)

    # 当日行情并入价格序列末端（今天的快照可能还没写）
    today_prices = {
        s.get("code"): s.get("price")
        for s in (quotes.get("stocks") or [])
        if s.get("code") and isinstance(s.get("price"), (int, float))
    }
    today_day = time.strftime("%Y-%m-%d")
    # 休市日的实时价格是上一交易日收盘价的重复值，不能当成"新的收盘价"
    # 参与收益率计算，否则同一价格会被算成零涨跌的一天。
    quotes_session = quotes.get("market_session") or {}
    if quotes_session.get("is_trading_day") is False:
        today_prices = {}

    if not snaps:
        return {
            "available": False,
            "as_of": now,
            "note": "首期运行，尚无历史快照，无可复盘的历史建议",
            "evaluations": [],
            "stats": {},
        }

    # 稳定性：相邻两期组合重合度
    stability: list[float] = []
    for prev, cur in zip(snaps, snaps[1:]):
        a = {
            p["code"] for p in (prev.get("advice") or {}).get("portfolio", [])
            if p.get("kind") == "stock"
        }
        b = {
            p["code"] for p in (cur.get("advice") or {}).get("portfolio", [])
            if p.get("kind") == "stock"
        }
        j = _jaccard(a, b)
        if j is not None:
            stability.append(j)

    evaluations: list[dict] = []
    window = snaps[-lookback:]
    for idx, snap in enumerate(window):
        day = snap["day"]
        positions = [
            p for p in (snap.get("advice") or {}).get("portfolio", [])
            if p.get("kind") == "stock"
        ]
        for pos in positions:
            code = pos.get("code")
            entry = pos.get("price")
            if not code or not isinstance(entry, (int, float)) or entry <= 0:
                continue
            hist = [
                (d, p) for d, p in series.get(code, [])
                if d > day
            ]
            if not hist and today_day > day:
                cur = today_prices.get(code)
                if cur is not None:
                    hist = [(today_day, float(cur))]
            if not hist:
                continue
            hist = hist[:MAX_WINDOW]
            closes = [p for _d, p in hist]
            last_close = closes[-1]
            ret = (last_close / entry - 1.0) * 100.0
            stop = pos.get("stop_price")
            targets = pos.get("targets") or []
            tp1 = targets[0]["price"] if targets else None
            stop_hit = bool(stop is not None and any(c <= stop for c in closes))
            tp_hit = bool(tp1 is not None and any(c >= tp1 for c in closes))
            call = pos.get("call") or "中性"
            if call == "偏多":
                hit = ret > 0
            elif call == "偏空":
                hit = ret < 0
            else:
                hit = None
            evaluations.append(
                {
                    "day": day,
                    "code": code,
                    "name": pos.get("name"),
                    "call": call,
                    "entry": entry,
                    "exit": last_close,
                    "obs_days": len(closes),
                    "ret_pct": round(ret, 2),
                    "hit": hit,
                    "stop_hit": stop_hit,
                    "tp_hit": tp_hit,
                    "stop_price": stop,
                    "tp1_price": tp1,
                    "weight_pct": pos.get("weight_pct"),
                }
            )

    judged = [e for e in evaluations if e["hit"] is not None]
    hits = sum(1 for e in judged if e["hit"])
    stops = sum(1 for e in evaluations if e["stop_hit"])
    tps = sum(1 for e in evaluations if e["tp_hit"])

    by_call: dict[str, list[float]] = {}
    for e in judged:
        by_call.setdefault(e["call"], []).append(e["ret_pct"])

    avg_ret = round(
        sum(e["ret_pct"] for e in judged) / len(judged), 2
    ) if judged else None

    # 偏差：偏多/偏空组的实际均值 vs 建议隐含的第一档目标 +8%
    implied = 8.0
    bias = {}
    for call, rets in by_call.items():
        mean_ret = round(sum(rets) / len(rets), 2)
        bias[call] = {
            "n": len(rets),
            "avg_ret_pct": mean_ret,
            "implied_target_pct": implied if call == "偏多" else -implied,
            "gap_pct": round(mean_ret - (implied if call == "偏多" else -implied), 2),
        }

    stats = {
        "snapshots": len(snaps),
        "evaluated": len(evaluations),
        "judged": len(judged),
        "calls": len(judged),
        "hits": hits,
        "hit_rate": round(hits / len(judged), 4) if judged else None,
        "stop_trigger_rate": round(stops / len(evaluations), 4) if evaluations else None,
        "target_hit_rate": round(tps / len(evaluations), 4) if evaluations else None,
        "stability_avg_jaccard": round(sum(stability) / len(stability), 4) if stability else None,
        "stability_samples": len(stability),
        "avg_return_pct": avg_ret,
        "bias": bias,
    }

    return {
        "available": bool(evaluations),
        "as_of": now,
        "note": (
            f"已复盘 {len(snaps)} 期快照，{len(evaluations)} 条可判定建议"
            if evaluations
            else "已有历史快照，但暂无可判定的后续价格数据"
        ),
        "evaluations": evaluations,
        "stats": stats,
    }
