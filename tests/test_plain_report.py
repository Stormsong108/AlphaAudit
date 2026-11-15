"""小白版日报的回归测试。

两个必须锁死的性质：
1. 数字防漂移 —— 小白版里的每个数字都必须能在原始数据里找到出处，
   防止"翻译成白话"过程中数字被改写（宋老师 11-09 明确提过"数字不对"）。
2. 术语覆盖 —— 原文出现的专业术语，在小白版里必须有对应的白话解释。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plain_glossary as PG  # noqa: E402
import plain_report as PR  # noqa: E402

# 小白版里禁止出现的黑话（这些必须有白话解释或已被替换掉）
BANNED_BARE = [
    "Sharpe", "Brier", "AUC", "Jaccard", "IC", "t 值",
    "十分位", "门禁", "样本外", "Newey-West", "缩尾", "中位数口径",
]


def test_glossary_has_plain_language_for_every_term():
    """词典里每一条都必须有白话解释，且不能是原样抄回术语本身。"""
    assert len(PG.TERMS) >= 20, "术语词典条目太少，覆盖不住日报里的黑话"
    for term, entry in PG.TERMS.items():
        assert entry.plain, f"术语 {term} 缺少白话解释"
        assert entry.plain != term, f"术语 {term} 的白话解释不能是原文"
        assert len(entry.plain) >= 8, f"术语 {term} 的白话解释太短，说不清"


def test_glossary_lookup_is_case_insensitive():
    """查找不区分大小写，且未收录的词返回 None 而不是 KeyError。"""
    assert PG.explain("Sharpe") is not None
    assert PG.explain("sharpe") is not None
    assert PG.explain("SHARPE") is not None
    assert PG.explain("这个词根本没收录") is None


def test_build_plain_sections_covers_key_numbers():
    """小白版必须包含原报告里的关键数字，且数字与原始数据一致。"""
    ev = PR.load_evidence()
    plain = PR.build_plain_report(ev)

    # 门禁通过数：原文 4 道门禁过 2
    assert "2" in plain and "4" in plain

    # 多头组年化必须出现在白话版里，且与证据文件逐位一致。
    # ⚠️ 不能写死具体数值：面板每天长新数据，canonical 年化就会变
    # （2026-11-14 实测 16.40% → 14.78%）。写死会让测试以
    # 「测试自己过期了」的形式突然失败 —— 那不是回归。
    # 断言的不变量是「值来自真源」，不是「值等于某个数字」。
    strat = (ev.get("strategy") or {})
    lo = strat.get("long_only") or {}
    ann = lo.get("net_annualized")
    assert ann is not None, "证据文件里没有多头年化"
    assert f"{ann * 100:.2f}%" in plain or f"{ann * 100:.1f}%" in plain

    # 基准年化必须出现 —— 小白最容易漏掉"没跑赢大盘"这个事实
    # 基准在 strategy.benchmark 里（不是顶层）
    bench = (ev.get("strategy") or {}).get("benchmark") or {}
    bann = bench.get("annualized")
    assert bann is not None, "证据文件里没有基准年化"
    assert f"{bann * 100:.2f}%" in plain or f"{bann * 100:.1f}%" in plain

    # 方向命中率必须出现，且必须等于复盘真源里的那个数。
    # 这里不能写死百分比：复盘窗口会随着新快照累积而滚动，
    # 写死的数字会在某一天突然失效 —— 那不是回归，是测试自己过期了。
    # 断言取"值来自真源"这个不变量，而不是某个具体数值。
    review = PR.load_review()
    hit_rate = (review.get("stats") or {}).get("hit_rate")
    assert hit_rate is not None, "复盘没有命中率，白话版也就无从显示"
    assert f"{hit_rate * 100:.1f}%" in plain


def test_plain_report_has_no_unexplained_jargon():
    """小白版正文里不应出现裸露的英文术语（术语表自身除外）。"""
    ev = PR.load_evidence()
    plain = PR.build_plain_report(ev)

    # 去掉术语表区域后再检查
    body = plain.split("术语表") [0] if "术语表" in plain else plain
    for term in BANNED_BARE:
        # 允许出现在括号解释里，如"夏普比率（Sharpe）"，但不允许无解释裸露
        for m in re.finditer(re.escape(term), body):
            window = body[max(0, m.start() - 30): m.end() + 30]
            assert re.search(r"[（(].{0,20}" + re.escape(term), window) or \
                re.search(re.escape(term) + r".{0,20}[)）]", window), \
                f"小白版出现未加解释的术语：{term}（上下文：{window!r}）"


def test_numbers_in_plain_report_trace_back_to_source():
    """小白版里的百分比数字必须能在原始数据里找到对应值。"""
    ev = PR.load_evidence()
    review = PR.load_review()
    plain = PR.build_plain_report(ev, review)

    # 收集原始数据里所有百分比形态的数值（转成报告里的展示形态）
    source_pcts = set()
    def collect(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            source_pcts.add(f"{obj * 100:.2f}")
            source_pcts.add(f"{obj * 100:.1f}")
            source_pcts.add(f"{obj * 100:.4f}")
            # 已是百分数单位的字段（如 avg_return_pct=1.88 表示 1.88%）
            source_pcts.add(f"{obj:.2f}")
            source_pcts.add(f"{obj:.1f}")
            source_pcts.add(f"{obj:.4f}")
            source_pcts.add(f"{abs(obj):.2f}")
            source_pcts.add(f"{abs(obj):.1f}")
            source_pcts.add(f"{abs(obj):.4f}")

    collect(ev)
    collect(review)

    # 小白版里出现的百分比
    for m in re.finditer(r"(\d+\.\d+)%", plain):
        val = m.group(1)
        # 允许 4 位以内精度；取最接近的原始表示
        ok = any(abs(float(val) - float(s)) < 0.011 for s in source_pcts)
        assert ok, f"小白版数字 {val}% 在原始数据里找不到出处（数字漂移）"


def test_plain_report_says_what_it_means_for_a_beginner():
    """小白版必须给出"这意味着什么"的判断，而不只是罗列指标。"""
    ev = PR.load_evidence()
    plain = PR.build_plain_report(ev)
    assert "结论" in plain or "什么意思" in plain or "大白话" in plain
    assert "跑赢" in plain or "没跑赢" in plain or "跑输" in plain


# ── 降级路径（2026-11-12 补）─────────────────────────────────────────────
# 背景：build_evidence() 曾瞬时抛异常，load_evidence() 静默回退到
# oos_latest.json —— 而那个文件的键形状完全不同（strategy_test_segment vs
# strategy），导致 gates/strategy/prob/factors 四个 section 静默渲染成空。
# 读者看到的是"空白"，而不是"数据缺失" —— 这是最危险的一类 bug。


def _force_build_evidence_failure(monkeypatch):
    """让 research_evidence.build_evidence 抛异常。"""
    import types

    bad = types.ModuleType("research_evidence")

    def boom():
        raise RuntimeError("模拟故障")

    bad.build_evidence = boom
    monkeypatch.setitem(sys.modules, "research_evidence", bad)


def test_fallback_never_silently_empties_sections(monkeypatch):
    """降级后 section 不能是空的（键必须被映射过来）。"""
    _force_build_evidence_failure(monkeypatch)
    ev = PR.load_evidence()

    assert ev.get("_fallback") is True
    assert ev.get("_fallback_reason"), "降级必须记录原因，不能静默"

    # 这些 section 在降级后仍必须有内容
    assert len(PR._strategy_section(ev)) > 0, "降级后策略 section 不能是空的"
    assert len(PR._signal_section(ev)) > 0, "降级后信号 section 不能是空的"
    assert len(PR._factors_section(ev)) > 0, "降级后因子 section 不能是空的"


def test_fallback_preserves_numbers_exactly(monkeypatch):
    """降级映射只改键，不改数字 —— 数字真理来源只有 ooslab。"""
    _force_build_evidence_failure(monkeypatch)
    ev = PR.load_evidence()
    raw = json.loads(
        (PR.DATA / "research" / "oos_latest.json").read_text(encoding="utf-8")
    )
    assert ev["strategy"]["long_only"]["net_annualized"] == (
        raw["strategy_test_segment"]["long_only"]["net_annualized"]
    )
    assert ev["signal_diagnostic"]["ic_mean"] == raw["signal_diagnostic"]["ic_mean"]


def test_fallback_is_visible_to_the_reader(monkeypatch):
    """降级必须在正文里显式告知读者，不能只写在 stderr。"""
    _force_build_evidence_failure(monkeypatch)
    ev = PR.load_evidence()
    plain = PR.build_plain_report(ev)
    assert "降级" in plain, "数据降级必须在正文里显式标注"
    # 门禁不可用时，不能说"通过了"也不能说"没通过"
    assert "不能" in plain


def test_glossary_covers_daily_report_vocabulary():
    """日报正文里实际出现的词，必须能在词典里查到。"""
    # 这些词都真实出现在 render/report.py 的日报卡片与正文里
    must_cover = [
        "大盘温度", "权益仓位", "理财仓位", "现金", "仓位",
        "止损", "止损触发率", "目标达成率", "隐含目标", "权重",
        "偏多", "偏空", "中性", "口径", "口径冲突", "口径拆分",
        "双源", "单源", "交叉核验", "交易日", "年化", "分位",
        "净值", "净值型", "固收", "货币基金", "混合基金",
        "数据源可用性", "显著性", "模拟组合", "可复现", "均值", "校准",
    ]
    missing = [t for t in must_cover if PG.explain(t) is None]
    assert not missing, f"词典缺这些日报真实用词：{missing}"


def test_glossary_aliases_are_real_separate_entries():
    """别名必须是独立的元组元素，不能把分隔符写进字符串。

    历史 bug：aliases=("毛十分位价差「, 」尾部价差",) —— 分隔符被当成
    字面量，于是"毛十分位价差"和"尾部价差"两个词都查不到。
    """
    for term, entry in PG.TERMS.items():
        for alias in entry.aliases:
            assert "「" not in alias and "」" not in alias, (
                f"术语 {term} 的别名 {alias!r} 里混进了分隔符"
            )
            assert "," not in alias, f"术语 {term} 的别名 {alias!r} 里混进了逗号"
            assert alias.strip(), f"术语 {term} 存在空别名"
            assert PG.explain(alias) is not None, (
                f"术语 {term} 的别名 {alias!r} 查不到自己"
            )
