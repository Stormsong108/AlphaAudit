# -*- coding: utf-8 -*-
"""F10 × NAV × 价格 三方交叉核对（探索用，输出到 scratch 文件）。"""
from __future__ import annotations
import json, os, sys
from datetime import date

REPO = "/Users/sst/.hermes/profiles/xiaojin/workspace/market_daily"
ETF = os.path.join(REPO, "data/history/kline_etf")
NAV = os.path.join(REPO, "data/history/nav_etf")
OUT = "/Users/sst/.hermes/profiles/xiaojin/cache/scratch/f10_xcheck.txt"


def load(p):
    return json.load(open(p, encoding="utf-8"))


f10 = load(os.path.join(REPO, "data/research/etf_f10_actions.json"))
lines = []
P = lines.append

for fn in sorted(os.listdir(ETF)):
    if not fn.endswith(".json"):
        continue
    sym = fn[:-5]
    code = sym[2:]
    rows = load(os.path.join(ETF, fn))
    nav = {}
    for r in load(os.path.join(NAV, f"{code}.json")):
        nav[r["date"]] = (float(r["nav"]), float(r["cumnav"]))
    pdates = [r["date"] for r in rows]
    pclose = [float(r["close"]) for r in rows]
    pds = [date.fromisoformat(d) for d in pdates]
    act = f10.get(code, {"splits": [], "divs": []})
    P(f"===== {sym} {act.get('name','')}  price {pdates[0]}..{pdates[-1]} ({len(rows)} bars)")

    # ── NAV C/U ratio jumps (independent fold detection) ──
    nd = sorted(nav)
    ratios = []
    for d in nd:
        u, c = nav[d]
        ratios.append(c / u if u > 0 else float("nan"))
    nav_folds = []
    for i in range(1, len(nd)):
        if ratios[i - 1] > 0 and abs(ratios[i] / ratios[i - 1] - 1) > 0.08:
            nav_folds.append((nd[i], ratios[i] / ratios[i - 1]))
    # ── F10 splits ──
    f10_splits = []
    for s in act["splits"]:
        X = float(s["ratio"].split(":")[1])
        f10_splits.append((s["date"], X))
    f10_splits.sort()

    P(f"  F10 splits: {[(d, round(X,6)) for d,X in f10_splits]}")
    P(f"  NAV folds : {[(d, round(r,6)) for d,r in nav_folds]}")

    # ── price jumps (|ret|>12%) ──
    pjumps = []
    for i in range(1, len(rows)):
        r = pclose[i] / pclose[i - 1] - 1.0
        if abs(r) > 0.12:
            pjumps.append((pdates[i], r))
    P(f"  price jumps>12%: {[(d, round(r,4)) for d,r in pjumps]}")

    # ── match F10 splits to price jump dates ──
    for d, X in f10_splits:
        # NAV fold ratio near this date
        navr = [r for (dd, r) in nav_folds if abs((date.fromisoformat(dd) - date.fromisoformat(d)).days) <= 5]
        P(f"  SPLIT F10 {d} X={X:.6f}  nav_fold_ratio={navr}")
        # find price dates within +15 trading days whose ratio matches 1/X
        idx0 = pdates.index(d) if d in pdates else None
        cands = []
        if idx0 is not None:
            for j in range(idx0, min(idx0 + 15, len(rows))):
                if j == 0:
                    continue
                ratio = pclose[j] / pclose[j - 1]
                implied = ratio * X
                cands.append((pdates[j], round(ratio, 6), round(implied - 1.0, 5)))
        else:
            P("    (fold date not a price date)")
        if cands:
            P("    price-date candidates (date, raw_ratio, implied_ret):")
            for c in cands[:8]:
                P(f"      {c[0]} ratio={c[1]:.6f} implied_ret={c[2]:+.4f}")
        # also: what does C do at the NAV fold date
        if d in nav:
            i = nd.index(d)
            if i > 0:
                P(f"    NAV@{d}: U {nav[nd[i-1]][0]:.4f}->{nav[d][0]:.4f} "
                  f"C {nav[nd[i-1]][1]:.4f}->{nav[d][1]:.4f} "
                  f"Uratio {nav[nd[i-1]][0]/nav[d][0]:.4f} Cratio {nav[nd[i-1]][1]/nav[d][1]:.5f}")

    # ── dividends: F10 ex_date vs NAV U-drop vs price gap ──
    for dv in sorted(act["divs"], key=lambda x: x["ex_date"]):
        ex = dv["ex_date"]
        if ex < pdates[0] or ex > pdates[-1]:
            P(f"  DIV {ex} amt/sh={dv['amount_per_share']:.4f}  (out of price window)")
            continue
        # NAV U drop around ex
        ui = None
        for k in range(1, len(nd)):
            if nd[k] == ex:
                ui = k
        navinfo = ""
        if ui is not None:
            navinfo = (f"NAV dU={nav[nd[ui]][0]-nav[nd[ui-1]][0]:+.5f} "
                       f"dC={nav[nd[ui]][1]-nav[nd[ui-1]][1]:+.5f}")
        else:
            navinfo = "NAV: no row on ex_date"
        # price gap around ex
        j = pdates.index(ex)
        gaps = []
        for jj in range(max(1, j - 1), min(len(rows), j + 3)):
            gaps.append(f"{pdates[jj]}:{pclose[jj]/pclose[jj-1]-1:+.4f}")
        P(f"  DIV {ex} amt/sh={dv['amount_per_share']:.4f} | {navinfo} | price gaps {gaps}")

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print(f"wrote {OUT} ({len(lines)} lines)")
