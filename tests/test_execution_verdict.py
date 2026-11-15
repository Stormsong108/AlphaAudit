"""成交真实性检验「结论文案」的回归测试。

背景与坑
--------
`execution_realism.py` 的 `run()` 以前是这样判定结论的：

    verdict["excess_retained"] = (e2 / e1) if abs(e1) > 1e-12 else None
    if e2 < 0:
        verdict["conclusion"] = "换成开盘成交后超额转负 —— 原来的正超额是成交假设的产物"

这段逻辑**只看了 open 口径（e2）的符号**，却把 close 口径（e1）默认为正。
实际数据里 close 口径本身就是负的（-2.03%），open 更负（-3.63%）：
这不是「由正转负」，而是「两个口径下本来就是负」—— 边际在任何成交口径下
都不成立。文案说成「原来的正超额是成交假设的产物」属于归错因。

同一个 bug 还带出一句荒谬数字：负 ÷ 负 = 1.79，被解读成「超额保留 179%」。
保留比例只有在两个口径同为正时才有意义，否则必须置 null 并说明不可比。

本测试文件直接调用结论文案生成逻辑（`build_verdict` / `build_r3_conclusion`）
并注入受控的 R1/R2/R3 数值，锁死每种符号组合的文案与新字段，
不依赖 `data/research/execution_realism.json` 的静态内容。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import execution_realism as ER  # noqa: E402


# ── 文案分支：close / open 两个口径的符号组合 ──────────────────────────────


def test_close_and_open_both_negative_is_not_described_as_turning_negative():
    """两个口径都为负 → 结论是「边际不成立」，绝不能说成「由正转负」。

    这就是本次修复的主场景：真实数据 close=-2.03% / open=-3.63%，
    两个口径符号一致，边际本来就不成立。
    """
    v = ER.build_verdict(close_excess=-0.0203, open_excess=-0.0363)

    assert v["close_excess_positive"] is False
    assert v["open_excess_positive"] is False
    assert "两个口径下超额都为负" in v["conclusion"]
    assert "边际不成立" in v["conclusion"]
    # 「转负」暗示存在一个先正后负的过程，这里不存在
    assert "转负" not in v["conclusion"]
    # 「原来的正超额」是本 bug 的错误归因，必须消失
    assert "原来的正超额" not in v["conclusion"]
    assert "成交假设的产物" not in v["conclusion"]


def test_close_positive_open_negative_keeps_the_assumption_attribution():
    """close 为正、open 为负 → 这才是「正超额是成交假设的产物」。"""
    v = ER.build_verdict(close_excess=0.05, open_excess=-0.01)

    assert v["close_excess_positive"] is True
    assert v["open_excess_positive"] is False
    assert "换成开盘成交后超额转负" in v["conclusion"]
    assert "原来的正超额是成交假设的产物" in v["conclusion"]


def test_both_positive_says_edge_survives():
    """两个口径都为正 → 换成更真实的成交价后 edge 依然存在。"""
    v = ER.build_verdict(close_excess=0.08, open_excess=0.04)

    assert v["close_excess_positive"] is True
    assert v["open_excess_positive"] is True
    assert "换成开盘成交后超额仍为正" in v["conclusion"]
    assert "信号本身站得住" in v["conclusion"]
    # 同为正时才允许出现保留比例
    assert v["excess_retained"] == pytest.approx(0.5)


def test_close_negative_open_positive_has_explicit_wording():
    """close 为负、open 为正 → 罕见情况，必须给出明确文案而非静默。"""
    v = ER.build_verdict(close_excess=-0.01, open_excess=0.02)

    assert v["close_excess_positive"] is False
    assert v["open_excess_positive"] is True
    assert "反而转正" in v["conclusion"]
    assert "罕见" in v["conclusion"]


# ── excess_retained：只在同为正时给出 ──────────────────────────────────────


@pytest.mark.parametrize(
    "close_excess, open_excess",
    [
        (-0.0203, -0.0363),   # 符号一致但都是负 —— 真实数据的情形
        (0.05, -0.01),        # 符号不一致
        (-0.01, 0.02),        # 符号不一致（反方向）
        (0.0, 0.02),          # 分母为零
        (-0.02, 0.0),         # 分子为零
        (0.0, 0.0),           # 两个都是零
        (1e-15, 0.02),        # 分母接近 0（噪声除噪声）
    ],
)
def test_excess_retained_is_null_when_not_comparable(close_excess, open_excess):
    """分母非正、符号不一致、或分母接近 0 时，保留比例一律为 null。"""
    v = ER.build_verdict(close_excess=close_excess, open_excess=open_excess)
    assert v["excess_retained"] is None, (
        f"close={close_excess} open={open_excess} 时不该给保留比例，"
        f"实际给了 {v['excess_retained']}"
    )


def test_excess_retained_never_exceeds_unity_for_the_real_data():
    """真实数据（两个口径都为负）不得再出现「保留 179%」这种假比率。"""
    v = ER.build_verdict(close_excess=-0.02027668523515072,
                         open_excess=-0.0363016576355461)
    assert v["excess_retained"] is None
    # 反向验算：旧算法确实会算出 1.79，正是本次要消灭的荒谬值
    assert (-0.0363016576355461 / -0.02027668523515072) > 1.0


def test_excess_retained_present_only_when_both_positive():
    """同为正 → 保留比例存在且等于 open / close。"""
    v = ER.build_verdict(close_excess=0.04, open_excess=0.03)
    assert v["excess_retained"] == pytest.approx(0.75)
    assert 0.0 < v["excess_retained"] <= 1.0


# ── R3 文案：同样要看 open 口径的符号 ──────────────────────────────────────


def test_r3_negative_after_negative_open_says_it_stays_negative():
    """open 本来为负时，R3 不是「由正转负」，而是「仍为负并进一步恶化」。"""
    text = ER.build_r3_conclusion(
        close_excess=-0.0203, open_excess=-0.0363, r3_excess=-0.0373
    )
    assert "仍为负" in text
    assert "转负" not in text
    assert "实盘不可行" in text


def test_r3_turning_negative_from_positive_open():
    """open 为正、R3 转负 → 才是真正的「由正转负」。"""
    text = ER.build_r3_conclusion(
        close_excess=0.08, open_excess=0.05, r3_excess=-0.01
    )
    assert "超额转负" in text
    assert "实盘不可行" in text


def test_r3_still_positive_but_below_close_excess():
    """R3 仍为正但低于理论口径 → 文案要如实说明低于理论口径。"""
    text = ER.build_r3_conclusion(
        close_excess=0.08, open_excess=0.05, r3_excess=0.03
    )
    assert "超额仍为正" in text
    assert "低于理论成交口径" in text


# ── 静态产物一致性（次要防线，主防线是上面的纯函数测试）──────────────────


def test_shipped_artifact_matches_the_new_verdict_logic():
    """已落盘的 execution_realism.json 必须与新判定逻辑自洽。

    这不是把静态 JSON 当成被测对象，而是防止「代码改了、产物没重跑」
    导致下游 daily_brief 继续读到旧文案。
    """
    p = ROOT / "data" / "research" / "execution_realism.json"
    if not p.exists():
        pytest.skip("execution_realism.json 不存在，先跑 execution_realism.py")

    v = (json.loads(p.read_text(encoding="utf-8")) or {}).get("verdict") or {}
    if "close_excess" not in v:
        pytest.skip("产物里没有 verdict 数值字段（状态不可用），跳过")

    e1 = v["close_excess"]
    e2 = v["open_excess"]
    expected = ER.build_verdict(close_excess=e1, open_excess=e2)

    assert v["conclusion"] == expected["conclusion"], (
        "落盘产物的结论文案与判定逻辑不一致，需要重跑 execution_realism.py"
    )
    assert v.get("close_excess_positive") == expected["close_excess_positive"]
    assert v.get("open_excess_positive") == expected["open_excess_positive"]
    assert v.get("excess_retained") == expected["excess_retained"]
