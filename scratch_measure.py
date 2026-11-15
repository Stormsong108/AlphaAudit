# -*- coding: utf-8 -*-
"""精确测量：折算/分红事件在价格与净值上的机械效应（探索用）。"""
from __future__ import annotations
import json, os
from datetime import date

REPO = "/Users/sst/.hermes/profiles/xiaojin/workspace/market_daily"
ETF = os.path.join(REPO, "data/history/kline_etf")
NAV = os.path.join(REPO, "data/history/nav_etf")
OUT = "/Users/sst/.hermes/profiles/xiaojin/cache/scratch/measure.txt"

f10 = json.load(open(os.path.join(REPO, "data/research/etf_f10_actions.json"), encoding="utf-8"))
L = []
P = L.append


def nav_map(code):
    out = {}
    for r in json.load(open(os.path.join(NAV, f"{code}.json"), encoding="utf-8")):
        out[r["date"]] = (float(r["nav"]), float(r["cumnav"]))
    return out


for fn in sorted(os.listdir(ETF)):
    if not fn.endswith(".json"):
        continue
    sym = fn[:-5]; code = sym[2:]
    rows = json.load(open(os.path.join(ETF, fn), encoding="utf-8"))
    nav = nav_map(code)
    pd_ = [r["date"] for r in rows]
    pc = [float(r["close"]) for r in rows]
    nd = sorted(nav)
    act = f10[code]
    P(f"===== {sym} {act.get('name','')}  price {pd_[0]}..{pd_[-1]}  nav {nd[0]}..{nd[-1]}")

    # ── 折算 ──
    for s in sorted(act["splits"], key=lambda x: x["date"]):
        fd = s["date"]; X = float(s["ratio"].split(":")[1])
        # 在 NAV 日网格上找折算点（U 跳变）
        fi = nd.index(fd) if fd in nd else None
        uratio = cratio = None
        if fi is not None and fi > 0:
            uratio = nav[nd[fi - 1]][0] / nav[nd[fi]][0]
            cratio = nav[nd[fi - 1]][1] / nav[nd[fi]][1]
        # 在价格网格上找跳变：score = |log(ratio) - log(1/X)|
        best = None
        lo = None
        for i, d in enumerate(pd_):
            if d < fd:
                continue
            if best is not None and len([x for x in pd_ if x < d]) > (len(pd_) - 1):
                break
            if i == 0:
                continue
            r = pc[i] / pc[i - 1]
            sc = abs(r * X - 1.0)
            # 只考虑折算日后 20 个交易日内
            if len([x for x in pd_ if x <= d]) - len([x for x in pd_ if x <= fd]) > 20:
                break
            if best is None or sc < best[1]:
                best = (d, sc, r, i)
        if best is None:
            P(f"  FOLD {fd} X={X}: no price candidate")
            continue
        d, sc, r, i = best
        P(f"  FOLD navdate={fd} X={X} navUratio={uratio if uratio is None else round(uratio,6)} "
          f"navCratio={cratio if cratio is None else round(cratio,5)}")
        P(f"    price jump {d}: prev_close={pc[i-1]:.4f} close={pc[i]:.4f} ratio={r:.6f} "
          f"implied(ratio*X-1)={r*X-1:+.4%} score={sc:.4f}")
        # 净值口径：跳变日对应的 C 收益（NAV 日期 d 及之后）
        for nd_ in [x for x in nd if fd <= x <= d][:6]:
            j = nd.index(nd_)
            if j > 0:
                P(f"      NAV {nd_}: U {nav[nd[j-1]][0]:.5f}->{nav[nd[j]][0]:.5f} "
                  f"C {nav[nd[j-1]][1]:.5f}->{nav[nd[j]][1]:.5f} "
                  f"dC/C={(nav[nd[j]][1]/nav[nd[j-1]][1]-1):+.4%} dU/U={(nav[nd[j]][0]/nav[nd[j-1]][0]-1):+.4%}")

    # ── 分红 ──
    for dv in sorted(act["divs"], key=lambda x: x["ex_date"]):
        ex = dv["ex_date"]; amt = dv["amount_per_share"]
        if ex < pd_[0] or ex > pd_[-1]:
            continue
        i = pd_.index(ex)
        pr = (pc[i] / pc[i - 1] - 1.0) if i > 0 else None
        # NAV 端 d_true = dC/k - dU，k 取该 regime
        P(f"  DIV ex={ex} f10_amt={amt:.4f}  price_ret={pr:+.4%} close_prev={pc[i-1]:.4f}")
        for k_ in (None,):
            j = nd.index(ex) if ex in nd else None
            if j is None or j == 0:
                P("    NAV: no row on ex_date")
                continue
            dU = nav[nd[j]][0] - nav[nd[j - 1]][0]
            dC = nav[nd[j]][1] - nav[nd[j - 1]][1]
            P(f"    NAV {ex}: dU={dU:+.5f} dC={dC:+.5f} U {nav[nd[j-1]][0]:.5f}->{nav[nd[j]][0]:.5f} "
              f"C {nav[nd[j-1]][1]:.5f}->{nav[nd[j]][1]:.5f}  dC/U={dC/nav[nd[j-1]][0]:+.5f}")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print(f"wrote {OUT} ({len(L)} lines)")
