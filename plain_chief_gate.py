
#!/usr/bin/env python3
"""大副终审闸门 —— 小白版日报的 4 项硬检查。

用法：python3 plain_chief_gate.py     （exit 0 = 通过，exit 1 = 有缺陷）

为什么不用 pytest：实测 pytest 收集时模块导入路径与直接运行不一致，
会出现「直接调用会红、pytest 报绿」的假通过（r2：验收靠注入，不靠声明）。
本文件用独立进程跑，退出码才是真信号。
"""
import sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import plain_report as PR
import plain_glossary as PG

fails = []

# ① alias 必须真能解析（不能写成 "A「, 」B" 单字符串）
MUST = ["尾部价差", "毛十分位价差", "打平成本", "breakeven cost",
        "相对基准年化", "多头超额", "横截面缩尾均值", "winsor",
        "ma_dev_60", "ma_dev_20", "mom_20", "mom_60"]
miss = [t for t in MUST if PG.explain(t) is None]
if miss:
    fails.append(f"[1] alias 查不到（aliases 写成了单字符串）: {miss}")

# ② 源码里不得再出现「, 」手写分隔
src = (ROOT / "plain_glossary.py").read_text(encoding="utf-8")
bad = re.findall(r'aliases=\("([^"]*「, 」[^"]*)"\)', src)
if bad:
    fails.append(f"[2] aliases 仍是单字符串: {bad}")

# ③ Brier 基准线全篇唯一
plain = PR.build_plain_report(PR.load_evidence())
RX = re.compile(r"(?:瞎猜的基准线|全猜基准|完全瞎猜的基准线|全猜的基准率|"
                r"瞎猜基准率|完全瞎猜的基准率)[^0-9]{0,6}(\d\.\d{5})")
found = sorted(set(RX.findall(plain)))
if len(found) > 1:
    fails.append(f"[3] Brier 基准线不唯一（同一指标多个答案）: {found}")

# ④ 措辞必须与数字一致
for mm in re.finditer(r"(\d\.\d{5})[^。]{0,25}?基准线\s*(\d\.\d{5})([^。]{0,45})", plain):
    v, b, tail = float(mm.group(1)), float(mm.group(2)), mm.group(3)
    if "更差" in tail and v < b:
        fails.append(f"[4] 措辞与数字矛盾: {v} < {b} 但说「{tail.strip()}」")

if fails:
    print("FAIL")
    for f in fails:
        print("   ", f)
    sys.exit(1)
print("PASS —— 4 项硬检查全部通过")
sys.exit(0)
