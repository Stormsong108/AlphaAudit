"""对已生成的 HTML 成品做结构审计：表头对齐 + 每表后有分析 + 首日空态 + 研报区。"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tq", ROOT / "tests" / "test_report_quality.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_TableCollector = _mod._TableCollector

day = sys.argv[1] if len(sys.argv) > 1 else "2026-11-10"
html_path = ROOT / "data" / "html" / f"{day}.html"
html = html_path.read_text(encoding="utf-8")

print(f"FILE: {html_path}  ({len(html.encode('utf-8'))} bytes)")

p = _TableCollector()
p.feed(html)
print(f"TABLES: {len(p.tables)}")

bad = []
for i, t in enumerate(p.tables):
    if not t["rows"]:
        bad.append((i, "no data rows"))
        continue
    for j, n in enumerate(t["rows"]):
        if n != t["ncol"]:
            bad.append((i, f"head={t['ncol']} row{j}={n}"))
print("STRUCTURE ERRORS:", bad if bad else "NONE")

# 数值列表头/数据对齐类一致
mism = []
for i, t in enumerate(p.tables):
    if not t["rows"]:
        continue
    heads, first = t["head_classes"], t["cell_classes"][0]
    for j, (h, c) in enumerate(zip(heads, first)):
        if ("num" in h) != ("num" in c):
            mism.append((i, j))
print("ALIGN-CLASS MISMATCH:", mism if mism else "NONE")

segs = re.findall(r"</table>(.*?)(?=<table|</body)", html, re.S)
miss = [i for i, s in enumerate(segs) if "analysis" not in s]
print(f"TABLES WITHOUT FOLLOWING ANALYSIS: {miss if miss else 'NONE'} (of {len(segs)})")
print("ANALYSIS BLOCKS:", html.count("class='analysis'"))

print("RESEARCH BLOCK:", "YES" if "每日研报" in html and "小金观点" in html else "NO")
print("TRADING-DAY NOTE:", "YES" if "交易日状态" in html or "休市" in html else "NO")
print("DEAD TABLE '可判定建议 0.00':", "PRESENT (BAD)" if "可判定建议 0.00" in html else "ABSENT (GOOD)")
print("LITERAL 'None':", "PRESENT (BAD)" if ">None<" in html or ">None " in html else "ABSENT (GOOD)")

# 抽一个分析块和研报区看内容
m = re.search(r"<div class='research'>(.*?)</div></div>", html, re.S)
if m:
    txt = re.sub(r"<[^>]+>", " ", m.group(1))
    txt = re.sub(r"\s+", " ", txt).strip()
    print("\n--- RESEARCH EXCERPT ---\n" + txt[:700])
