"""大副终审新增的回归测试 —— 钉死三处实测缺陷。

来源：2026-11-12 大副终审实测
  ① alias 写成单字符串（"A「, 」B"），12 个别名全部查不到
  ② Brier 基准线在同一份报告里出现三个不同值（0.24959 / 0.25000 / 0.24989），
     且 0.24978 被描述成「比基准线 0.24959 差不多甚至更差」—— 与数字自相矛盾
  ③ 原有 6 个测试全绿却没抓到以上两项 —— 测试盲区

原则（SOUL r2）：验收靠注入，不靠声明。这几条在修复前必须会红。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plain_glossary as PG  # noqa: E402
import plain_report as PR  # noqa: E402

MUST_RESOLVE = [
    "尾部价差", "毛十分位价差",
    "打平成本", "breakeven cost",
    "相对基准年化", "多头超额",
    "横截面缩尾均值", "winsor",
    "ma_dev_60", "ma_dev_20",
    "mom_20", "mom_60",
]

# 措辞锚点：这些词后面跟着的 5 位小数才是「Brier 基准线」
_BASELINE_RE = re.compile(
    r"(?:瞎猜的基准线|全猜基准|完全瞎猜的基准线|全猜的基准率|"
    r"瞎猜基准率|完全瞎猜的基准率)[^0-9]{0,6}(\d\.\d{5})"
)


def test_aliases_actually_resolve():
    """别名必须能查到，不能写成 "A「, 」B" 这种单字符串。"""
    miss = [t for t in MUST_RESOLVE if PG.explain(t) is None]
    assert not miss, f"这些别名查不到（aliases 写成了单字符串而非 tuple）：{miss}"


def test_no_separator_leaks_in_alias_definitions():
    """词典源码里不允许再出现「, 」这种手写分隔符。"""
    src = (ROOT / "plain_glossary.py").read_text(encoding="utf-8")
    bad = re.findall(r'aliases=\("([^"]*「, 」[^"]*)"\)', src)
    assert not bad, f"aliases 仍用「, 」单字符串分隔：{bad}"


def test_brier_baseline_single_value_in_report():
    """Brier 基准线全篇只能有一个值。

    修复前实测：同一份报告里同时出现 0.24959 / 0.25000 / 0.24989 三个基准线，
    同一指标三个答案 —— 小白完全无法判断哪个是真的。
    """
    ev = PR.load_evidence()
    plain = PR.build_plain_report(ev)
    found = set(_BASELINE_RE.findall(plain))
    assert len(found) <= 1, (
        f"报告里出现多个 Brier 基准线 {sorted(found)} —— 同一指标必须只有一个答案"
    )


def test_brier_verdict_agrees_with_the_numbers():
    """结论措辞必须与数字一致。

    修复前实测：Brier=0.24978 被说成「比基准线 0.24959 差不多，甚至更差」，
    但 0.24978 < 0.24959，数字上明明更好 —— 小白一眼就能看出矛盾。
    """
    ev = PR.load_evidence()
    plain = PR.build_plain_report(ev)
    for m in re.finditer(r"(\d\.\d{5})[^。]{0,20}?基准线\s*(\d\.\d{5})([^。]{0,40})", plain):
        val, base, tail = float(m.group(1)), float(m.group(2)), m.group(3)
        says_worse = any(w in tail for w in ("更差", "差一点点", "差不多"))
        if says_worse and val < base:
            assert not any(w in tail for w in ("更差",)), (
                f"Brier {val} 小于基准 {base}，但结论却说「{tail.strip()}」—— 措辞与数字矛盾"
            )


def test_gate_detail_baseline_matches_evidence():
    """门禁明细里写死的基准线必须来自 evidence 真实字段。"""
    ev = PR.load_evidence()
    gates = ev.get("gates") or []
    prob = ev.get("probability") or {}
    rel_raw = ((prob.get("rel") or {}).get("raw") or {})
    real_base = rel_raw.get("brier_baseline")
    if real_base is None:
        return
    for g in gates:
        detail = str(g.get("detail") or "")
        m = re.search(r"vs\s*全猜基准\s*(\d\.\d{5})", detail)
        if not m:
            continue
        got = float(m.group(1))
        # 报告里做的是展示性四舍五入（5 位小数），真值是全精度浮点。
        # 容差必须按「展示精度」给（半个末位 = 5e-6），不能按 1e-9 ——
        # 后者会把正常的四舍五入误判成口径不一致。
        # 2026-11-14 实测差 3.8e-9，正是这个量级。
        assert abs(got - real_base) <= 5e-6 + 1e-9, (
            f"门禁明细基准线 {got:.5f} 与 evidence 真值 {real_base:.8f} "
            f"差 {abs(got - real_base):.2e}，超出展示精度容差"
        )
