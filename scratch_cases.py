# -*- coding: utf-8 -*-
"""特殊事件精确取证：159922(折算+同日分红)、512690(巨额分红)、159919(小幅折算)、511010/511260(极小k)、
512100(折算)、513100(折算)。"""
from __future__ import annotations
import json, os

REPO = "/Users/sst/.hermes/profiles/xiaojin/workspace/market_daily"
ETF = os.path.join(REPO, "data/history/kline_etf")
NAV = os.path.join(REPO, "data/history/nav_etf")
f10 = json.load(open(os.path.join(REPO, "data/research/etf_f10_actions.json"), encoding="utf-8"))

CASES = {
    "159922": ("2024-11-25", "2024-12-06"),
    "512690": ("2021-12-24", "2022-01-07"),
    "159919": ("2019-01-08", "2019-01-20"),
    "511010": ("2026-09-10", "2026-09-25"),
    "511260": ("2026-09-10", "2026-09-25"),
    "512100": ("2022-08-29", "2022-09-09"),
    "513100": ("2022-01-06", "2022-01-21"),
    "510300": ("2019-01-11", "2019-01-22"),
    "512480": ("2026-06-25", "2026-07-10"),
    "516160": ("2024-09-09", "2024-09-24"),
}

for code, (lo, hi) in CASES.items():
    sym = next(f[:-5] for f in os.listdir(ETF) if f.endswith(".json") and f[2:-5] == code)
    rows = json.load(open(os.path.join(ETF, f"{sym}.json"), encoding="utf-8"))
    nav = {r["date"]: (float(r["nav"]), float(r["cumnav"]))
           for r in json.load(open(os.path.join(NAV, f"{code}.json"), encoding="utf-8"))}
    act = f10[code]
    print(f"########## {code} {act.get('name','')}")
    print(f"  F10 splits: {[(s['date'], s['ratio']) for s in act['splits']]}")
    print(f"  F10 divs  : {[(d['ex_date'], d['per10'], d['reg_date']) for d in act['divs']]}")
    print("  --- 价格 ---")
    prev = None
    for r in rows:
        d = r["date"]
        if lo <= d <= hi:
            c = float(r["close"])
            ret = f"{(c/prev-1)*100:+7.3f}%" if prev else "      -"
            print(f"    {d} close={c:10.4f} ret={ret}")
            prev = c
        elif d < lo:
            prev = float(r["close"])
    print("  --- 净值 ---")
    nd = sorted(nav)
    prevn = None
    for d in nd:
        if lo <= d <= hi:
            u, c = nav[d]
            if prevn:
                pu, pc = prevn
                print(f"    {d} U={u:10.5f} dU={u-pu:+9.5f}  C={c:10.5f} dC={c-pc:+9.5f} "
                      f"dC/C={(c/pc-1)*100:+7.3f}% dU/U={(u/pu-1)*100:+7.3f}%")
            else:
                print(f"    {d} U={u:10.5f}            C={c:10.5f}")
            prevn = nav[d]
        elif d < lo:
            prevn = nav[d]
    print()
