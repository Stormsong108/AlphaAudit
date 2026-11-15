# -*- coding: utf-8 -*-
"""探索脚本（临时）：从 NAV 代数还原折算/分红事件，并检验与价格跳变日的对齐。"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etf_adj as A

ETF_DIR = "data/history/kline_etf"


def load(sym):
    rows = json.load(open(f"{ETF_DIR}/{sym}.json"))
    nav = A.load_nav(sym[2:])
    return rows, nav


def nav_events(nav):
    ds = sorted(nav)
    u = np.array([nav[d][0] for d in ds], float)
    c = np.array([nav[d][1] for d in ds], float)
    n = len(ds)
    g = np.ones(n); m = np.ones(n); dv = np.zeros(n)
    for i in range(1, n):
        g[i] = c[i] / c[i - 1]           # C 的日比值 = 基金总收益因子
        uA = u[i - 1] * g[i]              # 假设无折算无分红时 U 的"理论值"
        m[i] = uA / u[i]                  # 折算乘数（<1 表示份额合并/折算缩小）
        dv[i] = uA - u[i]                # 每份分红（正）
    return ds, u, c, g, m, dv


def main():
    files = sorted(f for f in os.listdir(ETF_DIR) if f.endswith(".json"))
    tot_fold = 0
    rows_out = []
    for fn in files:
        sym = fn[:-5]
        rows, nav = load(sym)
        if not nav:
            print(sym, "NO NAV"); continue
        ds, u, c, g, m, dv = nav_events(nav)
        pdates = [r["date"] for r in rows]
        pclose = np.array([r["close"] for r in rows], float)
        p0, p1 = pdates[0], pdates[-1]

        # NAV 日期落在价格窗口内但不是价格日
        inwin = [d for d in ds if p0 <= d <= p1]
        extra = [d for d in inwin if d not in set(pdates)]

        folds = [(ds[i], m[i]) for i in range(1, len(ds)) if abs(np.log(m[i])) > np.log(1.25)]
        divs = [(ds[i], dv[i], dv[i] / u[i - 1]) for i in range(1, len(ds))
                if abs(np.log(m[i])) < np.log(1.05) and dv[i] > 0.0015 * u[i - 1]]

        # 折算 → 价格跳变日匹配
        fold_match = []
        for (X, mm) in folds:
            if not (p0 <= X <= p1):
                fold_match.append((X, mm, None, None, None)); continue
            xi = pdates.index(X)
            lo = max(1, xi - 6); hi = min(len(pdates), xi + 16)
            best = None
            for j in range(lo, hi):
                ratio = pclose[j] / pclose[j - 1]
                sc = abs(ratio * mm - 1.0)
                if best is None or sc < best[0]:
                    best = (sc, j, ratio)
            if best:
                sc, j, ratio = best
                fold_match.append((X, mm, pdates[j], j - xi, ratio * mm - 1.0))
            else:
                fold_match.append((X, mm, None, None, None))
        tot_fold += len(folds)

        # 价格自身的巨跳（>25%）
        price_jumps = []
        for j in range(1, len(pdates)):
            rr = pclose[j] / pclose[j - 1] - 1.0
            if abs(rr) > 0.25:
                price_jumps.append((pdates[j], round(rr, 4)))

        rows_out.append((sym, p0, p1, len(folds), len(divs), fold_match, price_jumps, extra,
                         len(inwin)))
        print(f"{sym} nav[window]={len(inwin)} extra_nav_dates={len(extra)} "
              f"folds={len(folds)} divs={len(divs)} price_jumps={len(price_jumps)}")
        for X, mm, Y, off, r in fold_match:
            print(f"    FOLD nav={X} m={mm:.5f} -> price_day={Y} offset={off} implied_r={r}")
        if price_jumps:
            print(f"    PRICE_JUMPS {price_jumps}")
    print("TOTAL folds(all nav history):", tot_fold)


if __name__ == "__main__":
    main()
