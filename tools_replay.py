#!/usr/bin/env python3
"""重放 2026-11-11 的日报渲染，逐块比对与已发布 HTML 的数字是否一致。

用途：给日报加"白话层"时，必须证明**只增加解释、不改动任何数字**。
本脚本用当时落盘的 raw 数据重建各板块，并与 data/html/2026-11-11.html 做数字级 diff。

用法：
  .venv/bin/python tools_replay.py 2026-11-11
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analysis  # noqa: E402
from advice import build_advice  # noqa: E402
from render.report import render_html  # noqa: E402
from review import build_review  # noqa: E402
from trading_calendar import resolve_market_session  # noqa: E402

RAW_DIR = HERE / "data" / "raw"
HTML_DIR = HERE / "data" / "html"
SNAPSHOT_DIR = HERE / "data" / "snapshots"


class _FrozenTime:
    """把 time.strftime 钉在发布那一刻，让复盘等依赖"今天"的逻辑可复现。"""

    def __init__(self, frozen: str) -> None:
        self._frozen = frozen
        self._real = time.strftime

    def strftime(self, fmt: str, *a, **kw) -> str:
        if "%Y-%m-%d %H:%M:%S" in fmt:
            return self._frozen
        if "%Y-%m-%d" in fmt:
            return self._frozen[:10]
        if "%H:%M" in fmt:
            return self._frozen[11:16]
        return self._real(fmt, *a, **kw)


def _numbers(html: str) -> list[str]:
    """抽出 HTML 里所有数字（含百分比），用于数字级比对。"""
    txt = re.sub(r"<[^>]+>", " ", html)
    return re.findall(r"[+-]?\d[\d,]*\.?\d*%?", txt)


def replay(day: str, frozen: str = "2026-11-11 21:31:00") -> str:
    raw = json.loads((RAW_DIR / f"{day}.json").read_text(encoding="utf-8"))
    quotes, events = raw["quotes"], raw["events"]
    futures, products = raw["futures"], raw["products"]

    real_time = time.strftime
    time.strftime = _FrozenTime(frozen).strftime
    try:
        session = resolve_market_session(quotes)
        quotes["market_session"] = session
        review = build_review(SNAPSHOT_DIR, quotes)
        advice = build_advice(quotes, products)
        research = analysis.build_research_report(
            quotes, events, futures, advice, review, session, products
        )
    finally:
        time.strftime = real_time

    import research_evidence

    evidence = research_evidence.build_evidence()

    return render_html(
        quotes, events, futures,
        products=products, advice=advice, review=review,
        session=session, research=research, evidence=evidence,
        title="财经日报", now=frozen[:16],
    )


def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else "2026-11-11"
    published = (HTML_DIR / f"{day}.html").read_text(encoding="utf-8")
    rebuilt = replay(day)

    out = HERE / "data" / "html" / f"{day}.replay.html"
    out.write_text(rebuilt, encoding="utf-8")

    pn, rn = _numbers(published), _numbers(rebuilt)
    print(f"[replay] 已写出 {out}")
    print(f"[replay] 数字个数：发布版 {len(pn)} / 重放版 {len(rn)}")
    if pn == rn:
        print("[replay] ✅ 数字序列完全一致 —— 重放保真")
        return 0

    # 逐项找第一个差异
    for i, (a, b) in enumerate(zip(pn, rn)):
        if a != b:
            print(f"[replay] ❌ 第 {i} 个数字不同：发布版 {a!r} vs 重放版 {b!r}")
            print(f"[replay]    上下文 发布版: {pn[max(0,i-6):i+6]}")
            print(f"[replay]    上下文 重放版: {rn[max(0,i-6):i+6]}")
            break
    else:
        print(f"[replay] ❌ 长度不同，尾部差异：发布版 {pn[len(rn):len(rn)+12]} / 重放版 {rn[len(pn):len(pn)+12]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
