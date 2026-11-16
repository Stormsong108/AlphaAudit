"""信号解剖（signal_anatomy）的回归测试 —— 最高优先级的全新模块。

这个模块存在的核心理由：回测按**收盘价**成交，而信号在 close[t] 才算出来，
最早只能 open[t+1] 成交。所以必须把持有期收益**严格拆成互不重叠的段**，
逐段回答"这一段的超额是正是负""这一段我吃不吃得到"。

必须锁死的性质（按重要性排序）：

1. **逐股严格可加** —— 对**每一只股票**，S1×S2×S3×S4 相乘必须**精确等于**
   整段收益 close[xi]/close[ei-1]-1。这是"拆解"成立的前提：
   如果段与段之间有重叠或缺口，相乘就对不上整段收益，拆出来的数字没有意义。

   ⚠️ 必须**逐股**验证，不能只在篮子层面验证：
   篮子层面 (1+mean S1)(1+mean S2)... ≠ mean(per-stock product)（Jensen 不等式）。
   逐股相等、篮子层面才可能近似；反过来不成立。

2. **R1 / R2 口径与段定义一致**：
     - R1 = 收盘成交（入 close[ei]，出 close[xi]）→ 只吃到 S3 × S4
     - R2 = 开盘成交（入 open[ei]，出 open[xi]）→ 只吃到 S2 × S3 × S5
   这两个口径是模块存在的意义（"S1 结构上吃不到"）。

3. **S5 的分母是昨收** —— 出场日隔夜 = 出场日开盘 / 出场**前**一日收盘。
   开盘成交时你**没拿到出场日收盘价**；拿它当分母等于把一个从没持有过的
   价格写进收益里，方向还是反的（o/c 是当日日内涨幅的倒数）。
   这条曾经写错（分母用了出场日收盘），会让 R2 口径整体偏离真值。

4. **S1 结构上不可得** —— 信号在收盘后才算出来，S1（信号日收盘→买入日开盘）
   无论用什么成交假设都吃不到。这是全模块最重要的结论。

5. **段的方向性** —— S3 是"持有中段"（close[ei]→close[xi-1]），
   S4 是"出场日盘中"（close[xi-1]→close[xi]），S5 是"出场日隔夜"
   （close[xi-1]→open[xi]）。索引写错一个，整段拆解就全错。

全部用**合成面板**测试（不加载真面板、不联网、不写任何生产产物）。
合成面板的构造刻意用"能手工算出精确答案"的数字，让断言是绝对值而不是
近似相等 —— 段收益是价格比值的乘积，任何浮点近似都会掩盖索引错误。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import signal_anatomy as SA  # noqa: E402


# ── 合成面板 ───────────────────────────────────────────────────────────────
#
# 价格刻意选成 **2 的幂次**（10, 20, 40, 80, ...），这样：
#   - 每一段的收益都是能被手算出来的精确二进制分数（0.5、1.0、3.0 …）
#   - 乘积链在浮点下**精确**成立，断言可以用 == 而不是 approx
#
# 布局（6 只股票 × 8 天）：
#   close[i] = 10 * 2**i      → 每只股票每天翻倍
#   open[i]  = close[i-1]     → 开盘价等于前一天的收盘价（无跳空）
#
# 于是对每只股票（ei = 买入日，xi = 出场日）：
#   S1 = open[ei]/close[ei-1] - 1 = 0              （无跳空）
#   S2 = close[ei]/open[ei]   - 1 = 1
#   S3 = close[xi-1]/close[ei] - 1 = 2**(xi-1-ei) - 1
#   S4 = close[xi]/close[xi-1] - 1 = 1
#   S5 = open[xi]/close[xi-1] - 1 = 0              （无跳空）
# 整段 close[xi]/close[ei-1] - 1 = 2**(xi-ei+1) - 1

N_DAYS = 8
N_CODES = 6


@pytest.fixture
def panel():
    """close[i] = 10*2**i，open[i] = close[i-1]。每只股票每天翻倍。"""
    close = np.array(
        [[10.0 * 2.0 ** i for _ in range(N_CODES)] for i in range(N_DAYS)],
        dtype=np.float64,
    )
    open_ = np.empty_like(close)
    open_[0, :] = close[0, :]          # 第一天开盘 == 第一天收盘
    open_[1:, :] = close[:-1, :]        # 其余天开盘 == 前一天收盘
    dates = [f"2026-01-{i + 1:02d}" for i in range(N_DAYS)]
    codes = [f"0.60000{i}" for i in range(N_CODES)]
    return {
        "dates": dates,
        "codes": codes,
        "close": close,
        "open": open_,
    }


# ── 1. 逐股严格可加（核心）─────────────────────────────────────────────────


def test_each_stock_chain_equals_whole_period_return(panel):
    """**逐股**：S1×S2×S3×S4 相乘 == close[xi]/close[ei-1] - 1（精确）。

    这是整个模块的地基。任何一段的索引写错（多一天/少一天/错一天），
    这个恒等式都会立刻破掉。
    """
    close, open_ = panel["close"], panel["open"]
    ei, xi = 2, 6
    sel = np.arange(N_CODES)

    seg = SA._seg_returns(close, open_, ei, xi, sel)
    chain = (
        (1 + seg["S1_overnight"])
        * (1 + seg["S2_entry_day"])
        * (1 + seg["S3_hold"])
        * (1 + seg["S4_exit_day"])
    ) - 1.0
    whole = close[xi, sel] / close[ei - 1, sel] - 1.0

    assert np.array_equal(chain, whole), (
        f"逐股链式可加性破了：\n{chain}\nvs 整段收益：\n{whole}"
    )


def test_chain_identity_holds_for_every_window(panel):
    """恒等式必须对**所有** (ei, xi) 组合都成立，不能只在挑的那一对上成立。"""
    close, open_ = panel["close"], panel["open"]
    sel = np.arange(N_CODES)

    for ei in range(1, N_DAYS):
        for xi in range(ei + 1, N_DAYS):
            seg = SA._seg_returns(close, open_, ei, xi, sel)
            chain = (
                (1 + seg["S1_overnight"])
                * (1 + seg["S2_entry_day"])
                * (1 + seg["S3_hold"])
                * (1 + seg["S4_exit_day"])
            ) - 1.0
            whole = close[xi, sel] / close[ei - 1, sel] - 1.0
            assert np.array_equal(chain, whole), (
                f"ei={ei} xi={xi} 时链式可加性破了"
            )


def test_segment_definitions_use_the_right_price_pairs(panel):
    """每一段必须用**它自己那两个端点**：索引错一天，恒等式可能仍过但语义全错。

    这里直接对合成数据断言每段的精确数值（价格是 10*2**i，比值可手算）。
    """
    close, open_ = panel["close"], panel["open"]
    ei, xi = 2, 6
    sel = np.arange(N_CODES)

    seg = SA._seg_returns(close, open_, ei, xi, sel)

    # open[i] == close[i-1] → S1（隔夜跳空）恒为 0
    assert np.array_equal(seg["S1_overnight"], np.zeros(N_CODES))
    # S2 = close[ei]/open[ei] - 1 = 翻倍 → +1
    assert np.array_equal(seg["S2_entry_day"], np.ones(N_CODES))
    # S3 = close[xi-1]/close[ei] - 1 = 2**(xi-1-ei) - 1
    assert np.array_equal(seg["S3_hold"], np.full(N_CODES, 2.0 ** (xi - 1 - ei) - 1))
    # S4 = close[xi]/close[xi-1] - 1 = 翻倍 → +1
    assert np.array_equal(seg["S4_exit_day"], np.ones(N_CODES))
    # S5 = open[xi]/close[xi-1] - 1 = 无跳空 → 0
    assert np.array_equal(seg["S5_exit_gap"], np.zeros(N_CODES))


def test_s5_denominator_is_yesterdays_close_not_exit_day_close(panel):
    """**S5 的分母必须是昨收（close[xi-1]），不是出场日收盘（close[xi]）。**

    开盘成交口径在出场日**开盘**就卖出 —— 你压根没拿到出场日收盘价。
    拿它当分母，等于把一个从没持有过的价格写进收益里，方向还是反的
    （o/c 是当日日内涨幅的倒数）。

    这条曾被写错过（分母用了 close[xi]），会让 R2 口径整体偏离真值，
    所以必须单独钉死。
    """
    close, open_ = panel["close"], panel["open"]
    ei, xi = 2, 6
    sel = np.arange(N_CODES)

    seg = SA._seg_returns(close, open_, ei, xi, sel)

    # 分母 = 出场前一日收盘
    assert np.array_equal(seg["S5_exit_gap"], open_[xi, sel] / close[xi - 1, sel] - 1.0)
    # 不是出场日收盘（那个值会给出完全不同的数）
    wrong = open_[xi, sel] / close[xi, sel] - 1.0
    assert not np.array_equal(seg["S5_exit_gap"], wrong), (
        "S5 仍在用出场日收盘当分母 —— 开盘成交口径里你拿不到这个价格"
    )


def test_s5_has_the_sign_of_an_overnight_gap(panel):
    """S5 是「昨收 → 出场日开盘」的隔夜跳空，方向与隔夜一致。

    构造一个出场日开盘跳空的场景：S5 必须为正（吃到了隔夜跳空）。
    """
    close = panel["close"].copy()
    open_ = panel["open"].copy()
    open_[6, :] = close[5, :] * 1.5          # 出场日开盘跳空 +50%
    panel2 = dict(panel, close=close, open=open_)

    seg = SA._seg_returns(close, open_, 2, 6, np.arange(N_CODES))
    assert np.allclose(seg["S5_exit_gap"], 0.5, atol=1e-15)


def test_segments_have_no_overlap_and_no_gap(panel):
    """持有期的段必须**首尾相接**：S2 起点=S3 起点，S3 终点=S4/S5 起点。

    用"把每段单独乘起来"反推：S2·S3·S4 必须等于 close[xi]/open[ei]-1
    （从买入日开盘到出场日收盘，中间不留缝也不重叠）。
    """
    close, open_ = panel["close"], panel["open"]
    ei, xi = 2, 6
    sel = np.arange(N_CODES)

    seg = SA._seg_returns(close, open_, ei, xi, sel)
    joined = (
        (1 + seg["S2_entry_day"])
        * (1 + seg["S3_hold"])
        * (1 + seg["S4_exit_day"])
    ) - 1.0
    assert np.allclose(joined, close[xi, sel] / open_[ei, sel] - 1.0, atol=1e-12)


# ── 2. R1 / R2 口径 ────────────────────────────────────────────────────────


def test_r1_equals_s3_times_s4(panel):
    """R1（收盘成交）== S3 × S4：入场 close[ei]，出场 close[xi]。"""
    close, open_ = panel["close"], panel["open"]
    ei, xi = 2, 6
    sel = np.arange(N_CODES)

    seg = SA._seg_returns(close, open_, ei, xi, sel)
    r1 = (1 + seg["S3_hold"]) * (1 + seg["S4_exit_day"]) - 1.0

    assert np.allclose(r1, close[xi, sel] / close[ei, sel] - 1.0, atol=1e-12), (
        "R1 口径必须等于 close[xi]/close[ei]-1（入场收盘、出场收盘）"
    )


def test_r2_equals_s2_times_s3_times_s5(panel):
    """R2（开盘成交）== S2 × S3 × S5：入场 open[ei]，出场 open[xi]。"""
    close, open_ = panel["close"], panel["open"]
    ei, xi = 2, 6
    sel = np.arange(N_CODES)

    seg = SA._seg_returns(close, open_, ei, xi, sel)
    r2 = (
        (1 + seg["S2_entry_day"])
        * (1 + seg["S3_hold"])
        * (1 + seg["S5_exit_gap"])
    ) - 1.0

    assert np.allclose(r2, open_[xi, sel] / open_[ei, sel] - 1.0, atol=1e-12), (
        "R2 口径必须等于 open[xi]/open[ei]-1（入场开盘、出场开盘）"
    )


def test_r1_and_r2_are_different_calibers(panel):
    """R1 ≠ R2：入场价不同（收盘 vs 开盘）→ 收益必然不同。

    这两条是**不同的成交假设**，如果它们算出同一个数，说明其中一条写错了。

    ⚠️ 合成面板是无跳空的（open[i] == close[i-1]），此时 R1 与 R2 恰好相等
    —— 那是数学上的必然，不是 bug。所以这里要**注入跳空**再比：
    入场日开盘跳空会改变 S2，但不改变 S3/S4 → R1 不动、R2 动。
    """
    close, open_ = panel["close"], panel["open"]
    ei, xi = 2, 6
    sel = np.arange(N_CODES)

    seg = SA._seg_returns(close, open_, ei, xi, sel)
    r1 = (1 + seg["S3_hold"]) * (1 + seg["S4_exit_day"]) - 1.0
    r2 = (
        (1 + seg["S2_entry_day"])
        * (1 + seg["S3_hold"])
        * (1 + seg["S5_exit_gap"])
    ) - 1.0

    # 入场日开盘跳空 +50%：只影响 S2（入场日盘中），R1 走 close→close 不受影响
    jumped = open_.copy()
    jumped[ei, :] = close[ei - 1, :] * 1.5
    seg2 = SA._seg_returns(close, jumped, ei, xi, sel)
    r1_after = (1 + seg2["S3_hold"]) * (1 + seg2["S4_exit_day"]) - 1.0
    r2_after = (
        (1 + seg2["S2_entry_day"])
        * (1 + seg2["S3_hold"])
        * (1 + seg2["S5_exit_gap"])
    ) - 1.0

    assert np.allclose(r1_after, r1, atol=1e-12), "R1 是 close→close，跳空不该影响它"
    assert not np.allclose(r2_after, r2, atol=1e-9), (
        "R2 是 open→open，入场日跳空必须改变它 —— 否则开盘价根本没被用上"
    )


def test_basket_reports_r1_r2_and_chain_check(panel):
    """_basket 必须同时给出 R1/R2/chain_check 三个口径。

    ⚠️ 2026-11-16 语义变更：chain_check 现在是**独立校验的最大绝对误差**
    （不是收益值）。旧实现把「篮子层面的 S1..S4 连乘」叫 chain_check，
    但那是拿被校验的均值自己算自己 —— 它无法发现聚合顺序错误。
    现在 chain_check = max|逐股连乘 − 直接价格重算|，恒应 ≈ 0。
    """
    close, open_ = panel["close"], panel["open"]
    ei, xi = 2, 6
    sel = np.arange(N_CODES)

    b = SA._basket(close, open_, ei, xi, sel)

    for key in ("S1_overnight", "S2_entry_day", "S3_hold", "S4_exit_day",
                "S5_exit_gap", "R1_total", "R2_total",
                "chain_check", "chain_check_r1", "chain_check_r2"):
        assert key in b, f"_basket 缺少 {key}"
        assert np.isfinite(b[key]), f"{key} 不是有限数"

    # chain_check 现在是误差量，必须 ≈ 0
    assert abs(b["chain_check"]) < 1e-9, (
        f"chain_check={b['chain_check']} 不是 0 —— 逐股连乘与价格重算不自洽"
    )
    assert abs(b["chain_check_r1"]) < 1e-9
    assert abs(b["chain_check_r2"]) < 1e-9

    # R1/R2 必须是**逐股连乘后取均值**（不是段均值相乘）
    # 合成面板里每只股票相同，所以两种算法恰好相等 —— 这里只锁结构，
    # 真正的 Jensen 回归测试见 test_basket_uses_per_stock_chaining_not_mean_of_means。
    assert b["R1_total"] == pytest.approx(
        (1 + b["S3_hold"]) * (1 + b["S4_exit_day"]) - 1.0, abs=1e-12)
    assert b["R2_total"] == pytest.approx(
        (1 + b["S2_entry_day"]) * (1 + b["S3_hold"]) * (1 + b["S5_exit_gap"]) - 1.0,
        abs=1e-12)


def test_basket_uses_per_stock_chaining_not_mean_of_means():
    """🚨 回归测试：篮子聚合必须是「先逐股连乘、再跨股平均」。

    旧实现是「先对每段跨股取均值、再把均值相乘」——
        (1+mean(S3)) × (1+mean(S4)) - 1
    这是 Jensen 误差：E[∏(1+sᵢ)] ≠ ∏(1+E[sᵢ])。
    实测这个错误让 R1 毛超额的**符号都反了**（日报曾写成负，实际为正）。

    本测试用**异质收益**的股票构造篮子：两只股票一段涨一段跌，
    这样「均值相乘」与「逐股连乘后平均」必然给出不同结果，
    任何退回旧实现的改动都会被这个断言抓住。
    """
    # 两只股票，收益完全相反 → 协方差为负 → Jensen 误差显著
    close = np.array([
        [100.0, 100.0],   # t=0
        [110.0,  90.0],   # t=1
        [121.0,  99.0],   # t=2  (ei=1 → close[1]=110/90)
        [133.1, 108.9],   # t=3
    ], dtype=np.float64)
    open_ = close.copy()   # 无跳空，S1=S5=0
    ei, xi = 1, 3
    sel = np.array([0, 1])

    b = SA._basket(close, open_, ei, xi, sel)

    # 逐股真值（手工算出）
    # 股票0: S3 = close[2]/close[1]-1 = 121/110-1 = 0.1
    #        S4 = close[3]/close[2]-1 = 133.1/121-1 = 0.1
    #        R1 = 1.1*1.1-1 = 0.21
    # 股票1: S3 = 99/90-1   = 0.1
    #        S4 = 108.9/99-1 = 0.1
    #        R1 = 0.21
    # 这里两只票同向，用反向收益更狠：改成股票1持有期下跌
    close2 = close.copy()
    close2[2, 1] = 81.0    # 90 → 81，S3 = -0.1
    close2[3, 1] = 72.9    # S4 = 72.9/81-1 = -0.1
    b2 = SA._basket(close2, open_, ei, xi, sel)

    r1_stock0 = (121.0 / 110.0) * (133.1 / 121.0) - 1.0        # = 0.21
    r1_stock1 = (81.0 / 90.0) * (72.9 / 81.0) - 1.0            # = -0.19
    expected = (r1_stock0 + r1_stock1) / 2.0                    # = 0.01

    wrong = ((1 + b2["S3_hold"]) * (1 + b2["S4_exit_day"])) - 1.0

    assert b2["R1_total"] == pytest.approx(expected, abs=1e-12), (
        f"R1_total={b2['R1_total']} 不是逐股连乘后平均的 {expected} —— "
        "说明退回了「段均值相乘」的 Jensen 错误实现"
    )
    assert b2["R1_total"] != pytest.approx(wrong, abs=1e-6), (
        "两种算法在本测试数据上恰好相同，测试失去鉴别力"
    )
    # 均值相乘会给出 0.21*(-0.19) 级别的偏差，明确不等于正确答案
    assert abs(wrong - expected) > 1e-3, (
        f"期望两种算法差异显著，实际 wrong={wrong} expected={expected}"
    )
    # 独立校验必须依然通过
    assert abs(b2["chain_check_r1"]) < 1e-9


def test_basket_chain_check_matches_whole_period_return(panel):
    """chain_check 现在是**独立误差**，必须 ≈ 0（而不是等于整段收益）。

    校验的独立性是关键：拿被校验的量自己算自己，等于没校验。
    """
    close, open_ = panel["close"], panel["open"]
    ei, xi = 2, 6
    sel = np.arange(N_CODES)

    b = SA._basket(close, open_, ei, xi, sel)
    whole = float(np.mean(close[xi, sel] / close[ei - 1, sel] - 1.0))

    # 误差量必须极小；整段收益本身是 2**(xi-ei+1)-1 = 31
    assert abs(b["chain_check"]) < 1e-9, (
        f"chain_check={b['chain_check']} 不是 0 —— 逐股连乘与价格重算不自洽"
    )
    assert whole > 1.0, "合成面板整段收益应远大于 1，确保校验不是碰巧通过"


# ── 3. S1 结构上不可得（本模块最重要的结论）───────────────────────────────


def test_s1_is_overnight_gap_from_signal_close_to_entry_open(panel):
    """S1 必须是"信号日收盘 → 买入日开盘"这一段（隔夜跳空）。

    索引口径：ei = 买入日 = t + lag，所以信号日收盘是 close[ei-1]。
    如果这里错成 close[ei]，S1 就变成了买入日盘中，结论整个反了。
    """
    close, open_ = panel["close"], panel["open"]
    ei, xi = 3, 6
    sel = np.arange(N_CODES)

    seg = SA._seg_returns(close, open_, ei, xi, sel)
    expected = open_[ei, sel] / close[ei - 1, sel] - 1.0

    assert np.allclose(seg["S1_overnight"], expected, atol=1e-12)


def test_s1_is_not_included_in_either_tradeable_caliber(panel):
    """S1 不得出现在 R1 或 R2 里 —— 它结构上吃不到。

    这是模块存在的核心理由：如果 S1 混进了某个可成交口径，
    "隔夜收益吃不到"这个结论就被自己的数据推翻了。

    做法：只改**信号日收盘**（close[ei-1]），S1 就变了，
    而 R1（close[ei]→close[xi]）与 R2（open[ei]→open[xi]）都不该变。
    """
    close, open_ = panel["close"], panel["open"]
    ei, xi = 3, 6
    sel = np.arange(N_CODES)

    seg = SA._seg_returns(close, open_, ei, xi, sel)
    r1 = (1 + seg["S3_hold"]) * (1 + seg["S4_exit_day"]) - 1.0
    r2 = (
        (1 + seg["S2_entry_day"])
        * (1 + seg["S3_hold"])
        * (1 + seg["S5_exit_gap"])
    ) - 1.0

    # 只把信号日收盘压低一半 → S1 变了（多出一段"没吃到的隔夜"）
    bumped = close.copy()
    bumped[ei - 1, :] = close[ei - 1, :] * 0.5
    seg2 = SA._seg_returns(bumped, open_, ei, xi, sel)

    assert not np.allclose(seg2["S1_overnight"], seg["S1_overnight"], atol=1e-9), (
        "改信号日收盘必须改变 S1 —— 否则 S1 根本没接在信号日收盘上"
    )
    r1_after = (1 + seg2["S3_hold"]) * (1 + seg2["S4_exit_day"]) - 1.0
    r2_after = (
        (1 + seg2["S2_entry_day"])
        * (1 + seg2["S3_hold"])
        * (1 + seg2["S5_exit_gap"])
    ) - 1.0

    assert np.allclose(r1_after, r1, atol=1e-12), "R1 被 S1 污染了"
    assert np.allclose(r2_after, r2, atol=1e-12), "R2 被 S1 污染了"


def test_segments_constant_definitions_are_stable():
    """段的 key 名与顺序是模块对外的契约，不能随手改名或改序。

    SEGMENTS 的顺序决定 _agg 产物里 mean/median 的键顺序，
    改名会让已落盘的 signal_anatomy.json 读不出对应关系。
    """
    assert [k for k, _ in SA.SEGMENTS] == [
        "S1_overnight", "S2_entry_day", "S3_hold", "S4_exit_day", "S5_exit_gap",
    ]
    for key, label in SA.SEGMENTS:
        assert label, f"{key} 缺少中文说明"


# ── 4. _decompose：期次切分与选股 ─────────────────────────────────────────


def _make_score(n_days, n_codes, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_days, n_codes))


def test_decompose_advances_by_hold_days(panel):
    """调仓期必须按 hold 步长推进（每期持仓 hold 天），不能每期只隔一天。

    这条锁住"持有期"这个参数真的生效了。
    """
    score = _make_score(N_DAYS, N_CODES)
    spec = {"hold": 2, "lag": 1, "quantile": 0.5, "min_valid": 2}

    rows = SA._decompose(panel, score, panel["open"], spec, 0, N_DAYS)

    assert len(rows) >= 2, "合成面板下应该至少切出 2 期"
    # 期次信号日之间的间隔必须是 hold 的整数倍
    idx_of = {d: i for i, d in enumerate(panel["dates"])}
    for prev, cur in zip(rows, rows[1:]):
        step = idx_of[cur["signal_date"]] - idx_of[prev["signal_date"]]
        assert step % spec["hold"] == 0, (
            f"调仓间隔 {step} 不是 hold={spec['hold']} 的整数倍"
        )


def test_decompose_dates_are_signal_entry_exit(panel):
    """每期必须记录 signal/entry/exit 三个日期，且与 lag/hold 对得上。"""
    score = _make_score(N_DAYS, N_CODES)
    spec = {"hold": 2, "lag": 1, "quantile": 0.5, "min_valid": 2}

    rows = SA._decompose(panel, score, panel["open"], spec, 0, N_DAYS)
    idx_of = {d: i for i, d in enumerate(panel["dates"])}

    assert rows, "应该切出至少一期"
    for r in rows:
        t = idx_of[r["signal_date"]]
        assert idx_of[r["entry_date"]] == t + spec["lag"], "入场日 = 信号日 + lag"
        assert idx_of[r["exit_date"]] == t + spec["lag"] + spec["hold"], (
            "出场日 = 信号日 + lag + hold"
        )


def test_decompose_accounting_is_consistent(panel):
    """n_valid + blocked 必须等于当期有效股票数。

    剔除涨停股后剩余的可交易股票数（n_valid）与被剔掉的数（blocked）
    加起来必须等于当期本来有效的股票数 —— 记账不平就意味着漏算或多算。
    """
    score = _make_score(N_DAYS, N_CODES, seed=3)
    spec = {"hold": 2, "lag": 1, "quantile": 0.5, "min_valid": 2}

    rows = SA._decompose(panel, score, panel["open"], spec, 0, N_DAYS)
    assert rows

    for r in rows:
        t = panel["dates"].index(r["signal_date"])
        valid = np.isfinite(score[t])
        assert r["n_valid"] >= spec["min_valid"]
        assert r["n_valid"] + r["blocked"] == int(valid.sum()), (
            "n_valid + blocked 必须等于当期有效股票数"
        )


def test_decompose_top_basket_is_the_highest_scores(panel):
    """top 篮子必须是分数最高的那一批，bottom 是最低的 —— 方向不能反。

    反了的话"超额为正/为负"的结论会整体翻转，是最危险的一类错误。

    做法：让**不同股票的价格路径不同**（按分数高低给不同的涨幅），
    于是"段收益"就能反查"这个篮子选的是不是高分那批"。
    """
    n_days, n_codes = N_DAYS, N_CODES
    # 每只股票有自己的日涨幅：分数越高（j 越大）→ 涨得越多
    growth = 1.0 + 0.05 * np.arange(n_codes)
    close = np.array(
        [[10.0 * growth[j] ** (i + 1) for j in range(n_codes)]
         for i in range(n_days)],
        dtype=np.float64,
    )
    open_ = np.empty_like(close)
    open_[0, :] = close[0, :]
    open_[1:, :] = close[:-1, :]
    panel2 = {
        "dates": [f"2026-01-{i + 1:02d}" for i in range(n_days)],
        "codes": [f"0.60000{i}" for i in range(n_codes)],
        "close": close,
        "open": open_,
    }

    # 分数与涨幅同向
    score = np.tile(np.arange(n_codes, dtype=float), (n_days, 1))
    spec = {"hold": 2, "lag": 1, "quantile": 0.5, "min_valid": 2}

    rows = SA._decompose(panel2, score, open_, spec, 0, n_days)
    assert rows

    for r in rows:
        t = panel2["dates"].index(r["signal_date"])
        s = score[t]
        valid = np.isfinite(s)
        sv = s[valid]
        k = max(1, int(len(sv) * spec["quantile"]))
        order = np.argsort(sv, kind="mergesort")
        top_idx = np.where(valid)[0][order[-k:]]
        bot_idx = np.where(valid)[0][order[:k]]

        # 用该篮子的股票重算 S3（持有中段），与模块报出的 top/bottom 对照
        ei = panel2["dates"].index(r["entry_date"])
        xi = panel2["dates"].index(r["exit_date"])
        seg_top = SA._seg_returns(close, open_, ei, xi, top_idx)
        seg_bot = SA._seg_returns(close, open_, ei, xi, bot_idx)

        assert r["top"]["S3_hold"] == pytest.approx(
            float(np.mean(seg_top["S3_hold"])), abs=1e-12), (
            "top 篮子不是分数最高的那批 —— 选股方向反了"
        )
        assert r["bottom"]["S3_hold"] == pytest.approx(
            float(np.mean(seg_bot["S3_hold"])), abs=1e-12), (
            "bottom 篮子不是分数最低的那批"
        )
        # 高分那批必须赚得更多（价格涨得更多）
        assert r["top"]["S3_hold"] > r["bottom"]["S3_hold"], (
            "top 篮子的段收益应高于 bottom —— 否则选股方向反了"
        )


def test_decompose_skips_periods_below_min_valid(panel):
    """有效股票数不足 min_valid 的期次必须**整期跳过**，不能出一期残缺结果。"""
    score = _make_score(N_DAYS, N_CODES, seed=5)
    spec = {"hold": 2, "lag": 1, "quantile": 0.5, "min_valid": N_CODES + 1}

    rows = SA._decompose(panel, score, panel["open"], spec, 0, N_DAYS)
    assert rows == [], "min_valid 高于股票数时不应产生任何期次"


def test_decompose_excludes_limit_up_stocks(panel):
    """开盘封涨停的股票必须被剔出篮子，并计入 blocked。

    这里构造一只"开盘 = 昨收×1.1"的股票（主板涨停），它必须被剔掉。
    """
    close = panel["close"].copy()
    open_ = panel["open"].copy()
    # 第 0 只股票在第 3 天开盘涨停（主板 10%）
    open_[3, 0] = close[2, 0] * 1.10
    panel2 = dict(panel, close=close, open=open_)

    score = _make_score(N_DAYS, N_CODES, seed=7)
    spec = {"hold": 2, "lag": 1, "quantile": 0.5, "min_valid": 2}

    rows = SA._decompose(panel2, score, open_, spec, 0, N_DAYS)
    assert rows

    # 找 signal_date 使 entry 落在第 3 天的那一期
    hit = [r for r in rows if panel2["dates"].index(r["entry_date"]) == 3]
    assert hit, "应该有一期的入场日是第 3 天"
    assert hit[0]["blocked"] == 1, "涨停股必须被计入 blocked"
    assert hit[0]["n_valid"] == N_CODES - 1


# ── 5. _agg：聚合口径 ─────────────────────────────────────────────────────


def test_agg_reports_mean_and_median_for_every_key(panel):
    """_agg 必须为每个段和每个口径同时给出 mean 和 median。"""
    score = _make_score(N_DAYS, N_CODES, seed=11)
    spec = {"hold": 2, "lag": 1, "quantile": 0.5, "min_valid": 2}

    rows = SA._decompose(panel, score, panel["open"], spec, 0, N_DAYS)
    agg = SA._agg(rows, "top")

    expected_keys = [k for k, _ in SA.SEGMENTS] + [
        "R1_total", "R2_total",
        "chain_check", "chain_check_r1", "chain_check_r2",
    ]
    for k in expected_keys:
        assert f"{k}_mean" in agg, f"聚合缺 {k}_mean"
        assert f"{k}_median" in agg, f"聚合缺 {k}_median"


def test_agg_mean_equals_manual_average(panel):
    """mean 必须等于手工求平均 —— 防止聚合时用错行（top/bench 串了）。"""
    score = _make_score(N_DAYS, N_CODES, seed=13)
    spec = {"hold": 2, "lag": 1, "quantile": 0.5, "min_valid": 2}

    rows = SA._decompose(panel, score, panel["open"], spec, 0, N_DAYS)
    agg = SA._agg(rows, "top")

    manual = float(np.mean([r["top"]["S3_hold"] for r in rows]))
    assert agg["S3_hold_mean"] == pytest.approx(manual, abs=1e-15)


def test_agg_of_empty_rows_is_empty_dict(panel):
    """没有期次时返回空字典，不得返回一堆 0（0 会被误读成"超额为零"）。"""
    assert SA._agg([], "top") == {}


# ── 6. _factor_direction：反转还是动量 ────────────────────────────────────


def _ic_mat(corr_sign: int, n_days: int = 40, n_codes: int = 60, seed: int = 17):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n_days, n_codes))
    y = corr_sign * f + rng.normal(scale=0.1, size=(n_days, n_codes))
    mat = {
        "dates": [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_days)],
        "fwd": y,
        "factors": {"f_test": f},
    }
    return mat


def test_factor_direction_reports_reversal_when_ic_negative():
    """发现期 IC 为负的因子 → 反转信号（买跌的）。

    构造：因子值与未来收益**负相关** → IC 为负 → 判定为反转。
    """
    mat = _ic_mat(-1)
    spec = {"factor_names": ["f_test"], "min_valid": 20}

    out = SA._factor_direction(spec, mat)

    assert out["ok"] is True
    assert out["ic_train"]["f_test"] < 0, "负相关因子的 IC 必须为负"
    assert out["n_reversal"] == 1 and out["n_momentum"] == 0
    assert "反转" in out["verdict"]


def test_factor_direction_reports_momentum_when_ic_positive():
    """发现期 IC 为正 → 动量信号，判定不能仍然说"反转"。"""
    mat = _ic_mat(+1, seed=19)
    spec = {"factor_names": ["f_test"], "min_valid": 20}

    out = SA._factor_direction(spec, mat)

    assert out["ok"] is True
    assert out["ic_train"]["f_test"] > 0
    assert out["n_momentum"] == 1 and out["n_reversal"] == 0
    # 正 IC 的判定文案是"动量/反转混杂"，不能写成"整体是反转型信号"
    assert "反转型信号" not in out["verdict"]


def test_factor_direction_without_fwd_is_not_ok():
    """面板缺 fwd/factors → 必须返回 ok=False，不得静默给出"反转"结论。"""
    out = SA._factor_direction(
        {"factor_names": ["f_test"], "min_valid": 20},
        {"dates": ["2020-01-01"]},
    )
    assert out["ok"] is False
    assert "fwd" in out["reason"] or "factors" in out["reason"]
