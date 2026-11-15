#!/usr/bin/env python3
"""日报验收清单（确定性检查，不联网、不改文件）。

输出逐条 PASS/FAIL + 证据，最后一行给总判定。
供看板上的核验 bot 直接读取，避免它去通读几十 KB 的 HTML。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAY = sys.argv[1] if len(sys.argv) > 1 else ""
HTML = HERE / "data" / "html" / f"{DAY}.html"
SNAP = HERE / "data" / "snapshots" / f"{DAY}.json"

checks: list[tuple[str, bool, str]] = []


def add(name: str, ok: bool, evidence: str) -> None:
    checks.append((name, bool(ok), evidence))


if not HTML.exists():
    print(f"FAIL: HTML 不存在 {HTML}")
    raise SystemExit(2)
html = HTML.read_text(encoding="utf-8")
snap = json.loads(SNAP.read_text(encoding="utf-8")) if SNAP.exists() else {}

# 1 单文件、无外链
ext = re.findall(r'(?:src|href)=["\'](https?://[^"\']+)', html)
ext = [u for u in ext if not u.startswith("https://open.feishu.cn")]
add("单文件自包含（无外部 CSS/JS/图片）", not ext, f"外链数 {len(ext)}" + (f" 例: {ext[:3]}" if ext else ""))

# 2 免责声明 + 模拟标注
need = ["不构成投资建议", "模拟"]
miss = [n for n in need if n not in html]
add("免责声明 + 研究/模拟标注", not miss, "缺失: " + ", ".join(miss) if miss else "已包含")

# 3 模拟组合表：权重/参考价/止损/目标
sec = re.search(r"二、模拟组合建议.*?(?=<h2>)", html, re.S)
seg = sec.group(0) if sec else ""
rows = seg.count("<tr>") - 1
has_stop = "止损" in seg
has_tgt = "目标位" in seg
has_w = "建议权重" in seg
add("模拟组合含 权重/参考价/止损位/目标位", rows > 0 and has_stop and has_tgt and has_w,
    f"组合行数 {rows}, 止损列 {has_stop}, 目标列 {has_tgt}, 权重列 {has_w}")

# 4 股票四类
themes = [t for t in ("大盘蓝筹", "科技成长", "红利低波", "港股互联网") if t in html]
add("股票池四类齐全", len(themes) == 4, f"命中 {themes}")

# 5 理财四类
cats = [c for c in ("货币基金", "固收+", "纯债基金", "混合基金") if c in html]
add("理财产品四类齐全", len(cats) == 4, f"命中 {cats}")

# 6 核验状态与冲突保留
has_status = ("双源核验" in html) or ("单源" in html)
conflict_sec = re.search(r"十、数据核验与口径冲突(.*?)</div>", html, re.S)
add("含核验状态标注", has_status, f"双源/单源标注 {has_status}")
add("口径冲突显式列出", "口径冲突" in html, f"章节存在: {'口径冲突' in html}")

# 7 快照含 advice
advice = snap.get("advice") or {}
portfolio = advice.get("portfolio") or []
watch = advice.get("watchlist") or []
add("快照存在且含 advice（判断/依据/关注标的/模拟组合）",
    bool(advice) and bool(portfolio) and bool(watch),
    f"snapshot {bool(snap)}, portfolio {len(portfolio)}, watchlist {len(watch)}, rules {len(advice.get('rules') or {})}")
add("快照标记为不可变", snap.get("immutable") is True, f"immutable={snap.get('immutable')}")

# 8 复盘区块
add("含历史复盘区块", "复盘" in html, f"{'复盘' in html}")

# 8b 交易日状态标注（数据对应哪个交易日必须写明，避免把上一交易日价格当今天价格）
add("含交易日状态标注", "交易日状态" in html, f"{'交易日状态' in html}")
sess = snap.get("market_session") or {}
add("快照固化 market_session", bool(sess), f"is_trading_day={sess.get('is_trading_day')}, data_date={sess.get('data_date')}")

# 9 仓位守恒
alloc = advice.get("allocation") or {}
if alloc:
    tot = (alloc.get("equity_pct", 0) + alloc.get("product_pct", 0) + alloc.get("cash_pct", 0))
    add("仓位合计 100% 且现金不低于下限",
        abs(tot - 100) < 0.5 and alloc.get("cash_pct", 0) >= alloc.get("cash_min_pct", 20) - 0.5,
        f"权益 {alloc.get('equity_pct')} + 理财 {alloc.get('product_pct')} + 现金 {alloc.get('cash_pct')} = {round(tot,2)}，现金下限 {alloc.get('cash_min_pct')}")

# 10 G4 四件套：五张表必须都在
g4_pieces = ("① 闭环总览", "② 口径拆分", "③ 选股能力", "④ 按交易日聚合", "⑤ G4 判据明细")
miss_pieces = [p for p in g4_pieces if p not in html]
add("G4 四件套五张表齐全", not miss_pieces,
    "缺失: " + ", ".join(miss_pieces) if miss_pieces else "全部存在")

# 11 模拟仓位放行状态必须显式
add("显式标注模拟仓位放行与否",
    ("模拟仓位放行：是" in html) or ("模拟仓位放行：否" in html),
    "未找到放行状态标注")

# 12 实时/回溯口径必须分开
add("实时与回溯口径分列",
    "实时预测 live" in html and "历史回溯自检 replay" in html,
    f"live={'实时预测 live' in html}, replay={'历史回溯自检 replay' in html}")

# 13 选股价差（门禁口径）+ 按天聚合
add("含十分位价差中位数（门禁口径）", "十分位价差 · 中位数口径（门禁口径）" in html,
    f"{'十分位价差 · 中位数口径（门禁口径）' in html}")
add("含按交易日聚合口径", "跑赢硬币的天数占比" in html,
    f"{'跑赢硬币的天数占比' in html}")

# 14 快照必须保留完整研究证据（门禁/判定/闭环）
snap_ev = snap.get("evidence") or {}
add("快照保留完整研究证据（门禁/判定/闭环）",
    bool(snap_ev.get("gates")) and bool(snap_ev.get("verdict")) and bool(snap_ev.get("prediction_loop")),
    f"gates={len(snap_ev.get('gates') or [])}, verdict={bool(snap_ev.get('verdict'))}, "
    f"loop={bool(snap_ev.get('prediction_loop'))}")
add("快照保留采集口径来源",
    bool(snap_ev.get("sources")) and (snap_ev.get("conflicts") is not None),
    f"sources={bool(snap_ev.get('sources'))}")

# 15 全 HTML 表格列数一致
#    校验规则：表头行的 th 数 == 每个数据行的 td 数。
#    表头行本身不算「不一致」—— 把表头当数据行去比，会误报。
bad_tables = []
for ti, tm in enumerate(re.finditer(r"<table>(.*?)</table>", html, re.S), start=1):
    seg = tm.group(1)
    ncol = None
    for ri, rm in enumerate(re.finditer(r"<tr>(.*?)</tr>", seg, re.S), start=1):
        row = rm.group(1)
        n_th = len(re.findall(r"<th[ >]", row))
        n_td = len(re.findall(r"<td[ >]", row))
        if n_th and not n_td:
            ncol = n_th            # 表头行
            continue
        if ncol is None:
            continue               # 没有表头，跳过（渲染层不会产生这种表）
        if n_td != ncol:
            bad_tables.append(f"表{ti}第{ri}行: 表头{ncol}列 vs {n_td}个单元格")
add("所有表格表头列数与数据行一致", not bad_tables,
    f"错位 {len(bad_tables)} 处" + (f" 例: {bad_tables[:3]}" if bad_tables else ""))

failed = 0
for name, ok, ev in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} —— {ev}")
    if not ok:
        failed += 1
print(f"CHECKS: {len(checks)}  FAILED: {failed}")
print("VERDICT:", "PASS" if failed == 0 else "FAIL")
