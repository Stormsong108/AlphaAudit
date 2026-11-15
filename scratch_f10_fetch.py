# -*- coding: utf-8 -*-
"""抓取天天基金 F10 分红送配页（fhsp_{code}.html），解析分红与份额折算记录。

输出: data/research/etf_f10_actions.json
  {"510300": {"name":..., "divs":[{reg_date,ex_date,per10,pay_date,amount_per_share}], 
              "splits":[{date,type,ratio}]}, ...}
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
ETF_DIR = os.path.join(REPO, "data", "history", "kline_etf")
OUT = os.path.join(REPO, "data", "research", "etf_f10_actions.json")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def fetch(code: str) -> str | None:
    url = f"https://fundf10.eastmoney.com/fhsp_{code}.html"
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Referer": "https://fundf10.eastmoney.com/"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        print(f"  [FETCH-FAIL] {code}: {e}", flush=True)
        return None


def _cells(row_html: str) -> list[str]:
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.S)
    out = []
    for c in cells:
        c = re.sub(r"<[^>]+>", "", c)
        c = c.replace("\xa0", " ").replace("&nbsp;", " ")
        out.append(re.sub(r"\s+", " ", c).strip())
    return out


def parse(html: str) -> dict:
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.S)
    divs, splits = [], []
    mode = None
    for r in rows:
        c = _cells(r)
        if not any(c):
            continue
        if c[0] == "年份" and len(c) > 1:
            if "每10份分红" in c[1] or "分红" in "".join(c):
                mode = "div"
                continue
            if "拆分" in "".join(c) or "折算" in "".join(c):
                mode = "split"
                continue
        if mode == "div" and re.fullmatch(r"\d{4}年", c[0]) and len(c) >= 4:
            m = re.search(r"派现金([\d.]+)元", "".join(c))
            if not m:
                m = re.search(r"([\d.]+)元", "".join(c))
            amount = float(m.group(1)) if m else None
            divs.append({
                "year": c[0],
                "reg_date": c[1] if len(c) > 1 else None,
                "ex_date": c[2] if len(c) > 2 else None,
                "desc": c[3] if len(c) > 3 else None,
                "pay_date": c[4] if len(c) > 4 else None,
                "per10": amount,
                "amount_per_share": (amount / 10.0) if amount is not None else None,
            })
        elif mode == "split" and re.fullmatch(r"\d{4}年", c[0]) and len(c) >= 4:
            splits.append({
                "year": c[0], "date": c[1], "type": c[2], "ratio": c[3],
            })
    return {"divs": divs, "splits": splits}


def main() -> None:
    files = sorted(f for f in os.listdir(ETF_DIR) if f.endswith(".json"))
    out: dict[str, dict] = {}
    for i, fn in enumerate(files, 1):
        sym = fn[:-5]
        code = sym[2:]
        html = fetch(code)
        if html is None:
            out[code] = {"sym": sym, "error": "fetch_failed", "divs": [], "splits": []}
            continue
        name_m = re.findall(r"<title>(.*?)</title>", html, re.S)
        name = name_m[0].split("(")[0] if name_m else ""
        p = parse(html)
        p["sym"] = sym
        p["name"] = name
        out[code] = p
        print(f"[{i}/{len(files)}] {sym} {name} divs={len(p['divs'])} splits={len(p['splits'])}",
              flush=True)
        time.sleep(0.35)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"=== saved {OUT} ({len(out)} codes) ===")


if __name__ == "__main__":
    main()
