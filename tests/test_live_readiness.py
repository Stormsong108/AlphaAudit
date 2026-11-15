"""实盘就绪度体检的回归测试。

覆盖三类：
  1. 恒等式与数学正确性（打平成本、换手上限、成本分解残差）
  2. 口径纪律（数字必须来自 oos_latest.json，不得手写）
  3. 缺失证据时必须大声失败，不得静默凑默认值
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import live_readiness as LR  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return LR.load_oos()


# ── 恒等式与数学 ────────────────────────────────────────────────────────────


def test_cost_decomposition_identity_holds(raw):
    """毛超额 - 成本拖累 = 净超额，且成本拖累 = 换手×成本率。"""
    c = LR.cost_decomposition(raw)
    assert c["identity_holds"], f"口径残差 {c['identity_residual']}，说明口径串了"
    # 直接断言残差绝对值，不能只断言 identity_holds —— 否则把阈值
    # 从 1e-9 放松到 1e6 也测不出来（mutation test 已验证这个盲区）。
    assert abs(c["identity_residual"]) < 1e-12, (
        f"残差 {c['identity_residual']} 过大，口径已串"
    )
    # 独立算一遍成本拖累，验证不是自证
    expected = c["turnover_per_period"] * c["cost_rate_roundtrip"]
    assert abs(c["cost_drag_per_period"] - expected) < 1e-12
    # 净超额 = 毛超额 - 成本拖累
    assert abs(
        c["net_excess_per_period"]
        - (c["gross_excess_per_period"] - c["cost_drag_per_period"])
    ) < 1e-12
    # 标志位必须与残差一致（否则有人把阈值放松成 1e6 也测不出来）
    assert c["identity_holds"] == (abs(c["identity_residual"]) < 1e-9), (
        "identity_holds 与残差实际值不一致 —— 判定逻辑被改坏了"
    )


def test_breakeven_cost_is_consistent_with_turnover(raw):
    """打平成本 × 当前换手 = 毛超额。"""
    be = LR.breakeven_cost(raw)
    implied = be["breakeven_cost_roundtrip"] * be["current_turnover"]
    gross = LR.cost_decomposition(raw)["gross_excess_per_period"]
    assert abs(implied - gross) < 1e-12


def test_max_turnover_at_actual_cost_is_not_a_dead_loop(raw):
    """回归测试：max_turnover 不能恒等于 current_turnover。

    历史 bug：写成 gross_excess/be，而 be = gross_excess/turnover，
    结果 max_turnover 恒等于 turnover，输出"换手需降至当前换手"这种废话。
    """
    be = LR.breakeven_cost(raw)
    assert be["max_turnover_at_actual_cost"] is not None
    assert abs(
        be["max_turnover_at_actual_cost"] - be["current_turnover"]
    ) > 1e-9, "max_turnover 恒等于 current_turnover —— 死循环公式"
    # 最大允许换手 × 成本率 = 毛超额
    gross = LR.cost_decomposition(raw)["gross_excess_per_period"]
    assert abs(
        be["max_turnover_at_actual_cost"] * be["actual_cost_roundtrip"] - gross
    ) < 1e-12


def test_turnover_reduction_needed_is_consistent(raw):
    """需要的降幅 = 1 - max_turnover/current_turnover。"""
    be = LR.breakeven_cost(raw)
    expected = 1 - be["max_turnover_at_actual_cost"] / be["current_turnover"]
    assert abs(be["turnover_reduction_needed"] - expected) < 1e-12


def test_annualization_uses_periods_per_year_not_trading_days(raw):
    """年化换算必须用「一年多少期」(252/hold)，不是 252。

    2026-11-12 修正：旧版直接 ×252，等于把「5 天的一期」当「一天的一期」，
    毛超额/成本拖累/净超额三个数字整体虚高 5 倍（+6.55/-16.69/-10.14
    应为 +1.31/-3.34/-2.03），且与 C2 的 -2.03% 自相矛盾。

    注意：断言里用字面量 252 和字面量 5，不能写 LR.TRADING_DAYS / LR 里的
    hold —— 否则改了常量测试也跟着变，等于自证循环（mutation test 已验证）。
    """
    assert LR.TRADING_DAYS == 252, (
        f"年化交易日数被改成了 {LR.TRADING_DAYS}，必须保持 252"
    )
    seg = raw["strategy_test_segment"]
    assert seg["hold"] == 5, f"canonical hold 应为 5，实为 {seg['hold']}"
    ppy = 252.0 / 5.0  # 字面量，防自证循环

    c = LR.cost_decomposition(raw)
    assert abs(c["annualized"]["gross_excess"]
               - c["gross_excess_per_period"] * ppy) < 1e-12
    assert abs(c["annualized"]["net_excess"]
               - c["net_excess_per_period"] * ppy) < 1e-12
    assert abs(c["annualized"]["cost_drag"]
               - c["cost_drag_per_period"] * ppy) < 1e-12
    assert abs(c["annualized"]["periods_per_year"] - ppy) < 1e-12


def test_net_excess_annualized_matches_c2_excess(raw):
    """同一件事（净超额）不能有两个数。

    成本分解的 net_excess 年化，必须与「相对基准」算出的 excess_annualized
    相等 —— 两者都来自「多头净收益 - 基准收益」。
    2026-11-12 修正前前者是 -10.14%、后者是 -2.03%，差 5 倍，
    这种自相矛盾正是「口径串了」最可靠的信号。
    """
    c = LR.cost_decomposition(raw)
    rel = LR.benchmark_relative(raw)
    assert abs(c["annualized"]["net_excess"] - rel["excess_annualized"]) < 1e-9, (
        f"净超额口径不一致：成本分解 {c['annualized']['net_excess']:.6f} "
        f"vs 相对基准 {rel['excess_annualized']:.6f}"
    )


def test_cost_drag_annualized_is_negative_and_below_gross(raw):
    """成本拖累是「吃掉多少」，年化后必须为负且不超过毛超额。"""
    c = LR.cost_decomposition(raw)
    a = c["annualized"]
    assert a["cost_drag"] > 0, "成本拖累本身是拖累量，正数"
    assert abs(a["gross_excess"] - a["cost_drag"] - a["net_excess"]) < 1e-9, (
        "毛超额 - 成本拖累 必须等于 净超额"
    )
    assert a["net_excess"] < 0, "当前证据下净超额应为负"


# ── 口径纪律 ────────────────────────────────────────────────────────────────


def test_no_hardcoded_numbers_in_output(raw):
    """体检输出里的每个数字都必须能在 oos_latest.json 里找到对应。

    做法：取所有派生字段，反查它们是由哪些原始键算出来的 ——
    这里退一步做更实用的检查：输出中不得出现 JSON 里不存在的
    "凭空"指标字段名（防止把两个不同口径的数混在一起）。
    """
    res = LR.full_check()
    # 关键字段必须在
    for k in ("cost_decomposition", "breakeven_cost", "three_calibers",
              "benchmark_relative", "cost_scenarios", "readiness"):
        assert k in res
    # 三口径必须齐全，缺一不可（这是 2026-11-10 教训）
    cal = res["three_calibers"]
    for k in ("primary_winsor_mean", "raw_mean", "median", "benchmark"):
        assert k in cal and cal[k] is not None


def test_three_calibers_spread_math(raw):
    """均值-中位数落差 = 中位数 - 主口径。"""
    cal = LR.three_calibers(raw)
    expected = (cal["median"]["net_annualized"]
                - cal["primary_winsor_mean"]["net_annualized"])
    assert abs(cal["mean_minus_median_spread"] - expected) < 1e-12


def test_skew_warning_matches_spread_sign(raw):
    """落差显著为负时必须报警。"""
    cal = LR.three_calibers(raw)
    assert cal["skew_warning"] == (cal["mean_minus_median_spread"] < -0.05)


def test_cost_scenarios_match_source(raw):
    """成本情景必须与原始文件逐行一致，不得改数。"""
    src = raw.get("cost_sensitivity") or []
    out = LR.cost_scenarios(raw)
    assert len(out) == len(src)
    for a, b in zip(src, out):
        assert a["cost_rate"] == b["cost_rate"]
        assert a["long_only_annualized"] == b["long_only_annualized"]
        assert a["long_only_sharpe"] == b["long_only_sharpe"]


# ── 缺失证据必须失败 ──────────────────────────────────────────────────────


def test_missing_evidence_raises_not_silently_defaults(tmp_path, monkeypatch):
    """证据文件不存在时必须抛异常，不能返回默认值 0 或空 dict。"""
    monkeypatch.setattr(LR, "OOS_LATEST", tmp_path / "nope.json")
    with pytest.raises(LR.EvidenceMissing):
        LR.load_oos()


def test_missing_strategy_segment_raises(raw):
    """缺 strategy_test_segment 时必须抛。"""
    bad = dict(raw)
    bad["strategy_test_segment"] = {}
    with pytest.raises(LR.EvidenceMissing):
        LR.cost_decomposition(bad)


def test_empty_evidence_file_raises(tmp_path, monkeypatch):
    """空文件也算缺失。"""
    p = tmp_path / "empty.json"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(LR, "OOS_LATEST", p)
    with pytest.raises(LR.EvidenceMissing):
        LR.load_oos()


# ── 判定逻辑 ───────────────────────────────────────────────────────────────


def test_readiness_verdict_has_four_checks(raw):
    """C1-C4 四项判定必须齐全（无跨段证据时）。"""
    rd = LR.readiness_verdict(raw)
    ids = [c["id"] for c in rd["checks"]]
    assert ids[:4] == ["C1", "C2", "C3", "C4"]
    for c in rd["checks"]:
        assert c["name"]
        assert c["detail"]
        assert isinstance(c["pass"], bool)
        assert c["fix"], "未通过项必须给出修法"


# ── C5 跨区段一致性（2026-11-12 新增）────────────────────────────────────
# 背景：hold 扫描在测试段找到 hold=10 超额 +0.49%（唯一正超额），
# 但训练段 −1.81%、验证段 −1.00% —— 符号不一致，是事后挑选的噪声。
# 跨段检验把这类「单段好看」的结论挡在实盘门外。


def test_cross_segment_evidence_is_loaded():
    """跨段证据文件存在时必须被读进来。"""
    cross = LR.cross_segment_evidence()
    if cross is None:
        pytest.skip("cross_segment_check.json 不存在，跳过")
    assert cross["hold"] in (5, 10, 20, 40)
    assert set(cross["excess_by_segment"]) >= {"train", "val", "test"}


def test_cross_segment_weighted_excess_matches_segments():
    """加权合并超额必须等于各段按期数加权的平均。"""
    cross = LR.cross_segment_evidence()
    if cross is None:
        pytest.skip("cross_segment_check.json 不存在，跳过")
    num = den = 0.0
    for seg in cross["per_segment"].values():
        n = seg.get("n_periods") or 0
        num += seg["excess_annualized"] * n
        den += n
    assert abs(cross["weighted_excess"] - num / den) < 1e-12


def test_c5_exists_when_cross_evidence_available(raw):
    """有跨段证据时，判定项必须包含 C5。"""
    if LR.cross_segment_evidence() is None:
        pytest.skip("无跨段证据")
    rd = LR.readiness_verdict(raw)
    ids = [c["id"] for c in rd["checks"]]
    assert "C5" in ids


def test_c5_fails_when_all_segments_negative(raw):
    """三段一致为负时，C5 必须判未通过（不能因为「符号一致」就放行）。

    这是本条判定最容易写反的地方：符号一致说明结论可靠，
    但如果一致的方向是负的，说明是**稳定的负 alpha**，
    恰恰更不该上实盘。
    """
    cross = LR.cross_segment_evidence()
    if cross is None:
        pytest.skip("无跨段证据")
    rd = LR.readiness_verdict(raw)
    c5 = next(c for c in rd["checks"] if c["id"] == "C5")
    if cross["all_segments_negative"]:
        assert c5["pass"] is False
        assert "系统性跑输" in c5["fix"]


def test_missing_cross_evidence_does_not_crash(raw, tmp_path, monkeypatch):
    """跨段文件缺失时不能崩，也不能凭空造出 C5。"""
    monkeypatch.setattr(LR, "CROSS_SEGMENT", tmp_path / "nope.json")
    assert LR.cross_segment_evidence() is None
    rd = LR.readiness_verdict(raw)
    ids = [c["id"] for c in rd["checks"]]
    assert "C5" not in ids


def test_corrupt_cross_evidence_returns_none(tmp_path, monkeypatch):
    """跨段文件损坏时返回 None，不得抛异常中断体检。"""
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(LR, "CROSS_SEGMENT", p)
    assert LR.cross_segment_evidence() is None


def test_cost_drag_displayed_as_negative(raw):
    """成本拖累必须显示为负号（拖累 = 减少收益），不能显示成 + 号贡献。"""
    txt = LR.format_report(LR.full_check())
    line = next(l for l in txt.splitlines() if "成本拖累" in l)
    # 拖累值前必须是负号
    import re
    m = re.search(r"\*\*([-+][\d.]+)%\*\*/年", line)
    assert m, f"成本拖累行没有带符号的百分比：{line}"
    assert m.group(1).startswith("-"), (
        f"成本拖累显示为 {m.group(1)}，会被读成「贡献」而非「拖累」"
    )


def test_verdict_consistent_with_pass_count(raw):
    """判定文案必须与「阻断项是否失败」一致，不能自相矛盾。

    2026-11-12 修正：旧版只按通过数分级，1/5 通过时给出
    「有条件实盘：小额试水」—— 而 C5 已证明三段全部跑输基准，
    那是结构性负 alpha，拿真钱试水等于送钱。
    现在改为：C1/C2/C3/C5/C6 任一失败 = 不具备实盘条件。
    （2026-11-13 新增 C6 成本地板可达性：成本地板高于打平成本 = 结构性不可行。）
    """
    rd = LR.readiness_verdict(raw)
    npass = rd["passed"]
    blocking_failed = rd["blocking_failed"]
    assert set(blocking_failed) <= {"C1", "C2", "C3", "C5", "C6"}
    if npass == rd["total"]:
        assert "小额实盘" in rd["verdict"]
        assert not blocking_failed
    elif blocking_failed:
        assert "不具备实盘条件" in rd["verdict"]
        assert rd["hard_block"] is True
        # 阻断项必须逐个点名，不能含糊
        for cid in blocking_failed:
            assert cid in rd["verdict"]
    else:
        # 只剩 C4 这类「数字本身不可信」的问题
        assert "暂不判定" in rd["verdict"]
        assert rd["hard_block"] is False


def test_current_evidence_blocks_live_trading(raw):
    """当前真实证据下必须判定为「不具备实盘条件」，不得给出试水建议。

    这是一条针对具体结论的回归锁：若将来 C5 判定逻辑被改松，
    这条测试会立刻失败，防止日报又开始对外发「小额试水」。
    """
    rd = LR.readiness_verdict(raw)
    assert rd["hard_block"] is True, "当前 C5 三段全负，必须阻断实盘"
    assert "不具备实盘条件" in rd["verdict"]
    assert "试水" not in rd["verdict"]
    assert "有条件实盘" not in rd["verdict"]


def test_format_report_mentions_key_numbers(raw):
    """报告必须包含实盘最关键的三个数字。"""
    res = LR.full_check()
    txt = LR.format_report(res)
    be = res["breakeven_cost"]["breakeven_cost_roundtrip"]
    assert f"{be*100:.4f}" in txt, "报告必须给出打平成本"
    assert "换手" in txt
    assert "中位数" in txt
    # 纯文字格式：无表格、无代码块
    assert "```" not in txt
    assert "|---" not in txt


# ── 成本地板（2026-11-13 新增）─────────────────────────────────────────────
# 背景：旧 C1 只比较「假设成本 vs 打平成本」，失败时给出的建议是
# 「压低交易成本或降低换手率 —— 这是唯一能翻转结论的杠杆」。
# 但 A股单笔成本有**不可压缩的下限**：佣金万2.5双边5bp + 印花税卖出5bp
# + 沪市过户费0.2bp = 10.2bp（单笔≥2万时），加滑点后更高。
# 当这个地板已经高于打平成本时，压成本永远救不回来 —— 必须单独成项。


def test_roundtrip_cost_known_values():
    """单笔往返成本必须等于手算值（字面量，防自证循环）。"""
    # 单笔 2 万：佣金按万2.5 收费（20000×0.025%=5元）→ 双边 5bp
    # + 印花税 5bp（卖出单边）+ 过户费 0.2bp（双向 0.01‰）= 10.2bp
    r = LR.roundtrip_cost_bp(20_000.0)
    assert abs(r["total_bp_roundtrip"] - 10.2) < 1e-9
    # 单笔 1 万：触发 ¥5 最低佣金 → 双边 10bp + 5 + 0.2 = 15.2bp
    r = LR.roundtrip_cost_bp(10_000.0)
    assert abs(r["total_bp_roundtrip"] - 15.2) < 1e-9
    assert r["below_min_commission"] is True
    # 单笔 5 千 → 双边 20bp + 5.2 = 25.2bp
    r = LR.roundtrip_cost_bp(5_000.0)
    assert abs(r["total_bp_roundtrip"] - 25.2) < 1e-9
    # 单笔 2 千 → 双边 50bp + 5.2 = 55.2bp
    r = LR.roundtrip_cost_bp(2_000.0)
    assert abs(r["total_bp_roundtrip"] - 55.2) < 1e-9


def test_cost_floor_is_nonlinear_in_order_size():
    """单笔越小成本越高（¥5 最低佣金支配）—— 小资金跑横截面必亏的机制。"""
    costs = [LR.roundtrip_cost_bp(n)["total_bp_roundtrip"]
             for n in (2_000, 5_000, 10_000, 20_000, 100_000)]
    assert costs == sorted(costs, reverse=True), f"成本未随单笔增大而单调下降: {costs}"
    # 单笔 ≥2万后进入费率地板，不再下降
    assert abs(costs[-1] - costs[-2]) < 1e-9


def test_cost_floor_flags_2wan_threshold():
    """2万元是「不再被最低佣金吃掉」的门槛。"""
    assert LR.cost_floor(1_000_000.0, 126)["order_ge_2wan"] is False
    assert LR.cost_floor(3_000_000.0, 126)["order_ge_2wan"] is True


def test_execution_feasibility_flags_structural_infeasibility(raw):
    """canonical 配置的成本地板高于打平成本 → 结构性不可行。

    ⚠️ 这里不能写死打平成本的具体数值：面板每天长新数据，
    canonical 的毛超额/换手都会变，写死的数字会以「测试过期」
    的形式突然失败（2026-11-14 实测：11.78bp → 18.37bp）。
    断言的不变量是「成本地板 > 打平成本」和「判定为结构性不可行」，
    打平成本本身从真源算出来再对账。
    """
    feas = LR.execution_feasibility(raw)
    # 打平成本必须等于「毛超额/换手」，与 breakeven_cost 同一口径
    be = LR.breakeven_cost(raw)
    assert abs(feas["breakeven_cost_bp"] - be["breakeven_cost_roundtrip"] * 1e4) < 1e-6
    assert feas["cost_floor_bp"] > feas["breakeven_cost_bp"]
    assert feas["structurally_infeasible"] is True
    assert feas["verdict"] == "structurally_infeasible"


def test_structural_infeasibility_is_not_a_cost_assumption_problem(raw):
    """结构性不可行 ≠ 成本假设偏高。

    回归测试：曾经只比较「假设成本 vs 打平成本」，于是把
    「根本不可能盈利」误报成「去谈个更好的费率就能救」。
    """
    feas = LR.execution_feasibility(raw)
    assert feas["verdict"] != "cost_assumption_too_high"
    assert feas["edge_gap_bp"] > 0


def test_structural_infeasibility_holds_at_any_capital_scale(raw):
    """把资金规模放大到 1000万也救不回来 —— 地板不随规模下降。"""
    for cap in (1_000_000.0, 3_000_000.0, 10_000_000.0, 100_000_000.0):
        feas = LR.execution_feasibility(raw, capital_yuan=cap, n_positions=126)
        assert feas["structurally_infeasible"] is True, (
            f"资金 {cap} 时不再判定结构性不可行 —— 结论被资金规模意外翻转"
        )


def test_c6_exists_and_is_blocking(raw):
    """C6 必须存在，且失败时进入 blocking_failed。"""
    v = LR.readiness_verdict(raw)
    ids = [c["id"] for c in v["checks"]]
    assert "C6" in ids, f"缺少 C6 成本地板门禁，实际门禁: {ids}"
    c6 = next(c for c in v["checks"] if c["id"] == "C6")
    if not c6["pass"]:
        assert "C6" in v["blocking_failed"], "C6 失败但未列入阻断项"
        assert v["hard_block"] is True


def test_c1_fix_text_not_misleading_when_structurally_infeasible(raw):
    """结构性不可行时，C1 的修复建议不能是「压低交易成本」。"""
    v = LR.readiness_verdict(raw)
    c1 = next(c for c in v["checks"] if c["id"] == "C1")
    feas = LR.execution_feasibility(raw)
    if feas["structurally_infeasible"]:
        assert "压低交易成本" not in c1["fix"], (
            f"结构性不可行却仍建议压低交易成本: {c1['fix']}"
        )
        assert "结构性不可行" in c1["fix"]
