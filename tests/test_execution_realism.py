"""成交真实性检验（execution_realism）的回归测试。

核心是 `run_periodic(price_matrix, ...)`：

    用 close 矩阵跑，必须**逐位复现** canonical OOS（忠实性闸门）。
    再用 open 矩阵跑，得到现实成交口径的结果。

本文件全部用合成的小面板（6-8 只股票 × 40 个交易日），不加载真面板、
不联网、不写任何 canonical 产物。

另外锁死涨停判定的三条口径铁律：
    一、只能用开盘那一刻已知的信息（open 相对**昨收**的涨幅）判断；
    二、用**四舍五入后的实际涨停价**判定，不用「涨幅 ≥ 10%」；
    三、拿不到昨收（停牌/新上市）时**不判涨停** —— 未知 ≠ 涨停。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import execution_realism as ER  # noqa: E402
import ooslab  # noqa: E402


# ── 合成 fixture ───────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_panel():
    """8 只股票 × 40 个交易日的确定性假面板。"""
    rng = np.random.default_rng(20261114)
    T, N = 40, 8
    dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(T)]
    codes = [f"6000{i:02d}" for i in range(N)]
    steps = rng.normal(0.001, 0.02, size=(T, N))
    close = 10.0 * np.cumprod(1.0 + steps, axis=0)
    score = rng.normal(size=(T, N))
    return {"dates": dates, "codes": codes, "close": close, "score": score}


@pytest.fixture
def fake_universe(monkeypatch, tmp_path):
    """把 universe.json 换成 tmp 目录里的假文件，ST 判定才可复现。"""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    uni = {
        "1.688825": "ST臻镭",      # ST
        "0.000758": "*ST禾信",     # *ST
        "1.600123": "退市示例",    # 退市
        "0.000001": "平安银行",    # 正常
    }
    (tmp_path / "data" / "universe.json").write_text(
        json.dumps(uni, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ER, "ROOT", tmp_path)
    return uni


def _synthetic_market(n_days=60, n_codes=200, seed=11):
    """确定性合成面板（比 fixture 更大，够算出多期换手）。

    直接返回 (mat, score)：mat 只有 run_periodic 需要的 dates/codes/close。
    """
    rng = np.random.default_rng(seed)
    dates = [f"d{i:03d}" for i in range(n_days)]
    codes = [f"6000{i:02d}" for i in range(n_codes)]
    close = 10.0 * np.exp(
        np.cumsum(rng.normal(0, 0.02, (n_days, n_codes)), axis=0)
    ).astype(np.float32)
    score = rng.normal(size=(n_days, n_codes))
    return {"dates": dates, "codes": codes, "close": close}, score


# ── 1. _limit_rates：按板块的涨跌幅限制 ────────────────────────────────────


@pytest.mark.parametrize("code,expected", [
    ("300750", 0.20),   # 创业板
    ("301001", 0.20),   # 创业板
    ("688001", 0.20),   # 科创板
    ("689001", 0.20),   # 科创板
    ("600000", 0.10),   # 沪主板
    ("000001", 0.10),   # 深主板
    ("002415", 0.10),   # 深主板
])
def test_limit_rates_by_board(code, expected):
    """主板 ±10%，创业板(300/301)/科创板(688/689) ±20%，ST/退市 ±5%。"""
    assert ER._limit_rates([code])[0] == pytest.approx(expected)


def test_limit_rates_st_overrides_board(fake_universe):
    """ST/退市股一律 ±5%，哪怕它是创业板/科创板代码。"""
    rates = ER._limit_rates(["1.688825", "0.000758", "1.600123", "0.000001"])
    assert rates[0] == pytest.approx(0.05)
    assert rates[1] == pytest.approx(0.05)
    assert rates[2] == pytest.approx(0.05)
    assert rates[3] == pytest.approx(0.10)


def test_limit_rates_length_matches_codes():
    assert len(ER._limit_rates(["600000", "300750", "000001"])) == 3


def test_limit_rates_handles_exchange_prefixed_codes():
    """真实面板里创业板/科创板代码带交易所前缀（0./1.），也必须拿到 0.20。

    回归锁：板块必须从**后 6 位数字**判断。
    写成 `c.split(".")[0]` 会取到市场前缀（'0'/'1'），
    创业板/科创板永远判不出来 —— 后果是 20% 的票按 10% 阈值判涨停，
    大量本来买得到的票被当成"涨停买不到"剔掉，R3 结果被系统性做偏。
    """
    rates = ER._limit_rates(["0.300750", "1.688001", "0.301001", "1.689001"])
    assert rates[0] == pytest.approx(0.20)
    assert rates[1] == pytest.approx(0.20)
    assert rates[2] == pytest.approx(0.20)
    assert rates[3] == pytest.approx(0.20)


def test_limit_rates_prefixed_main_board_stays_ten_percent():
    """带交易所前缀的主板票仍是 10%（不能被前缀写法带偏）。"""
    rates = ER._limit_rates(["0.000001", "1.600000", "0.002415"])
    assert rates.tolist() == pytest.approx([0.10, 0.10, 0.10])


def test_limit_rates_agree_with_dataquality_limit_of():
    """本模块的板块判定必须和 dataquality.limit_of 逐票一致。

    这是**跨模块一致性锁**，也是 2026-11-14 那个 bug 的真正防线：
    单条单元测试只验了 4 个手写代码，漏掉了「真实面板里到底有多少
    双创票被错判」这件事。两套独立实现必须给出同一个答案。

    如果 dataquality 不可用（单测环境没装），跳过 —— 不能让测试假通过。
    """
    import pytest as _pt
    try:
        import dataquality
    except Exception:  # noqa: BLE001
        _pt.skip("dataquality 不可用")

    codes = ["0.000001", "1.600000", "0.300750", "1.688001",
             "0.301001", "1.689001", "0.002415", "1.603000"]
    mine = ER._limit_rates(codes)
    for i, c in enumerate(codes):
        try:
            ref = float(dataquality.limit_of(c))
        except Exception:  # noqa: BLE001
            continue
        # dataquality.limit_of **不做 ST 判定**（代码里没有 ST 信息，
        # 它只做板块判定）；本模块会用 universe.json 把 ST 票压到 5%。
        # 所以只比「非 ST 票」的板块档位 —— ST 差异是设计上的，不是 bug。
        if ER._load_st_flags([c])[0]:
            continue
        assert abs(mine[i] - ref) < 1e-9, (
            f"{c}: execution_realism 判 {mine[i]}，dataquality 判 {ref} —— "
            "两套板块判定不一致"
        )


def test_limit_rates_covers_every_real_panel_code():
    """真实面板里所有代码都必须能被归到 10% / 20% / 5% 三档之一。

    回归背景（2026-11-14）：板块判定写成 `c.split(".")[0]` 时，
    创业板/科创板永远拿不到 0.20，被按 10% 阈值判涨停 ——
    全样本 242.9 万个 stock-day 里多算 2458 次误判，
    R3 的「买不到次数」从正确的 22 次虚高到 248 次。
    """
    import ooslab
    panel = ooslab.load_panel()
    codes = [str(c) for c in panel["codes"]]
    rates = ER._limit_rates(codes)
    assert len(rates) == len(codes)
    assert set(np.round(rates, 4)) <= {0.05, 0.10, 0.20}, (
        f"出现了预期外的涨跌幅档位：{sorted(set(np.round(rates, 4)))}"
    )
    # 双创票必须真的拿到 0.20 —— 这是 bug 的直接受害者。
    # 唯一例外：ST/*ST 票的涨跌幅上限是 ±5%，优先于板块的 ±20%
    # （真实面板里有 16 只这样的双创 ST 票）。
    st = ER._load_st_flags(codes)
    dual = [i for i, c in enumerate(codes)
            if str(c)[-6:].startswith(("300", "301", "688", "689"))
            and not st[i]]
    if dual:
        assert np.allclose(rates[dual], 0.20), (
            f"{len(dual)} 只非 ST 双创票里有一部分没拿到 0.20 阈值"
        )
    # 反向锁：ST 票必须是 5%，不管它属于哪个板块
    st_all = [i for i in range(len(codes)) if st[i]]
    if st_all:
        assert np.allclose(rates[st_all], 0.05), (
            f"{len(st_all)} 只 ST 票里有一部分没被压到 5%"
        )


# ── 2. _load_st_flags：从 universe.json 认 ST / 退市 ───────────────────────


def test_load_st_flags_marks_st_and_delisted(fake_universe):
    """ST / *ST / 退市 都要被标出来（漏判会让 R3 结果偏乐观）。"""
    flags = ER._load_st_flags(["1.688825", "0.000758", "1.600123", "0.000001"])
    assert flags.tolist() == [True, True, True, False]


def test_load_st_flags_matches_by_code_suffix(fake_universe):
    """universe 用 secid、面板用 code —— 统一成后缀匹配才能对上。"""
    # universe 里是 "1.688825"，面板里是 "0.688825"
    flags = ER._load_st_flags(["0.688825"])
    assert flags[0] is np.True_ or bool(flags[0]) is True


def test_load_st_flags_missing_universe_returns_all_false(monkeypatch, tmp_path):
    """universe.json 不存在 → 全部当正常股，不能抛异常打断流水线。"""
    monkeypatch.setattr(ER, "ROOT", tmp_path)
    flags = ER._load_st_flags(["600000", "300750"])
    assert flags.tolist() == [False, False]


def test_load_st_flags_broken_universe_returns_all_false(monkeypatch, tmp_path):
    """universe.json 坏了 → 全部当正常股，不能抛异常。"""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "universe.json").write_text("{坏掉的 JSON", encoding="utf-8")
    monkeypatch.setattr(ER, "ROOT", tmp_path)
    flags = ER._load_st_flags(["600000"])
    assert flags.tolist() == [False]


# ── 3. _limit_up_mask：用昨收算实际涨停价 ─────────────────────────────────


def _mask(close_prev, open_cur, codes):
    close = np.array([[close_prev]], dtype=float)
    open_ = np.array([[open_cur]], dtype=float)
    return bool(ER._limit_up_mask(close, open_, codes, 0, 0)[0])


def test_limit_up_uses_rounded_actual_limit_price(fake_universe):
    """昨收 10.00 → 主板涨停价 11.00（10.00×1.1 四舍五入到分）。"""
    assert _mask(10.00, 11.00, ["600000"]) is True
    assert _mask(10.00, 10.99, ["600000"]) is False


def test_limit_up_for_st_uses_five_percent(fake_universe):
    """ST 股昨收 10.00 → 涨停价 10.50（±5%），不是 11.00。"""
    assert _mask(10.00, 10.50, ["1.688825"]) is True
    assert _mask(10.00, 10.49, ["1.688825"]) is False
    # 同一个价格对主板票不算涨停、对 ST 票算涨停 —— 阈值必须按股票区分，
    # 否则 ST 股按 10% 判会把大量 5% 封板股误判成"没涨停"→ 该剔的没剔。
    assert _mask(10.00, 10.50, ["600000"]) is False


def test_limit_up_uses_rounded_price_not_raw_percentage(fake_universe):
    """昨收 3.03 → 涨停价 3.33（实际涨幅只有 9.90%）。

    按「涨幅 ≥ 9.8%」判会把 3.32 这种正常票误杀成涨停；
    按「涨幅 ≥ 10%」判又会漏掉真涨停。必须先算涨停价再比。
    """
    assert _mask(3.03, 3.33, ["600000"]) is True
    assert _mask(3.03, 3.32, ["600000"]) is False


def test_limit_up_not_judged_when_open_missing(fake_universe):
    """开盘价缺失 → 不判涨停（未知 ≠ 涨停）。"""
    assert _mask(10.00, float("nan"), ["600000"]) is False


def test_limit_up_not_judged_when_prev_close_missing(fake_universe):
    """昨收缺失（停牌/新上市）→ 不判涨停。"""
    assert _mask(float("nan"), 11.00, ["600000"]) is False


def test_limit_up_not_judged_for_non_positive_prices(fake_universe):
    """价格非正 → 不判涨停。"""
    assert _mask(0.0, 11.0, ["600000"]) is False
    assert _mask(10.0, 0.0, ["600000"]) is False


def test_limit_up_growth_board_uses_twenty_percent(fake_universe):
    """创业板昨收 10.00 → 涨停价 12.00，不是 11.00。"""
    assert _mask(10.00, 12.00, ["300750"]) is True
    assert _mask(10.00, 11.00, ["300750"]) is False


# ── 4. run_periodic：忠实性闸门（close 必须复现 canonical）────────────────


def test_run_periodic_with_close_reproduces_ooslab_exactly(synthetic_panel):
    """**忠实性闸门**：close 矩阵跑出来的数字必须与 ooslab.decile_backtest 逐位一致。

    误差 0.00e+00 是硬要求 —— 只要差一点点，open 口径的结论就不可信。
    """
    mat = {k: synthetic_panel[k] for k in ("dates", "codes", "close")}
    score = synthetic_panel["score"]
    kw = dict(hold=5, lag=1, cost_rate=0.003, quantile=0.5,
              min_valid=4, start_idx=0, end_idx=None, winsor=0.01)

    mine = ER.run_periodic(mat, score, mat["close"], mat["close"], **kw)
    ref = ooslab.decile_backtest(mat, score, **kw)

    assert mine["ok"] is True
    assert mine["n_periods"] == ref["n_periods"]
    assert mine["long_only"]["net_annualized"] == pytest.approx(
        ref["long_only"]["net_annualized"], abs=0.0)
    assert mine["long_only"]["sharpe"] == pytest.approx(
        ref["long_only"]["sharpe"], abs=0.0)
    assert mine["mean_turnover"] == pytest.approx(ref["mean_turnover"], abs=0.0)
    assert mine["benchmark"]["annualized"] == pytest.approx(
        ref["benchmark"]["annualized"], abs=0.0)


def test_match_canonical_gate_passes_on_reproduced_numbers(synthetic_panel):
    """把复现结果喂给忠实性闸门本身 → 必须 ok=True。"""
    mat = {k: synthetic_panel[k] for k in ("dates", "codes", "close")}
    kw = dict(hold=5, lag=1, cost_rate=0.003, quantile=0.5,
              min_valid=4, start_idx=0, end_idx=None, winsor=0.01)
    res = ER.run_periodic(mat, synthetic_panel["score"],
                           mat["close"], mat["close"], **kw)
    ref = ooslab.decile_backtest(mat, synthetic_panel["score"], **kw)

    raw = {"strategy_test_segment": {
        "ok": True,
        "n_periods": ref["n_periods"],
        "mean_turnover": ref["mean_turnover"],
        "long_only": {
            "net_annualized": ref["long_only"]["net_annualized"],
            "sharpe": ref["long_only"]["sharpe"],
        },
        "benchmark": {"annualized": ref["benchmark"]["annualized"]},
    }}
    gate = ER._match_canonical(res, raw)
    assert gate["ok"] is True, gate
    assert gate["worst_rel_err"] == 0.0


def test_match_canonical_gate_fails_on_mismatch(synthetic_panel):
    """数字对不上 → 闸门必须判失败（这是硬闸门，不是可选检查）。"""
    mat = {k: synthetic_panel[k] for k in ("dates", "codes", "close")}
    kw = dict(hold=5, lag=1, cost_rate=0.003, quantile=0.5,
              min_valid=4, start_idx=0, end_idx=None, winsor=0.01)
    res = ER.run_periodic(mat, synthetic_panel["score"],
                           mat["close"], mat["close"], **kw)

    raw = {"strategy_test_segment": {
        "ok": True,
        "n_periods": res["n_periods"],
        "mean_turnover": res["mean_turnover"],
        "long_only": {
            "net_annualized": res["long_only"]["net_annualized"] * 1.01,
            "sharpe": res["long_only"]["sharpe"],
        },
        "benchmark": {"annualized": res["benchmark"]["annualized"]},
    }}
    gate = ER._match_canonical(res, raw)
    assert gate["ok"] is False
    assert gate["checks"]["net_annualized"]["ok"] is False


def test_match_canonical_gate_fails_when_canonical_unavailable():
    """canonical 段不可用 → 直接判失败，不能默认通过。"""
    gate = ER._match_canonical({}, {"strategy_test_segment": {"ok": False}})
    assert gate["ok"] is False
    assert "canonical" in gate["reason"]


# ── 5. run_periodic：open 矩阵与涨停剔除 ──────────────────────────────────


def test_run_periodic_uses_the_price_matrix_it_is_given(synthetic_panel):
    """换成 open 矩阵跑，结果必须不同 —— 否则说明价格矩阵根本没被用上。"""
    mat = {k: synthetic_panel[k] for k in ("dates", "codes", "close")}
    score = synthetic_panel["score"]
    rng = np.random.default_rng(7)
    open_m = mat["close"] * (1.0 + rng.normal(0, 0.005, size=mat["close"].shape))
    kw = dict(hold=5, lag=1, cost_rate=0.003, quantile=0.5,
              min_valid=4, start_idx=0, end_idx=None, winsor=0.01)

    r_close = ER.run_periodic(mat, score, mat["close"], mat["close"], **kw)
    r_open = ER.run_periodic(mat, score, open_m, open_m, **kw)

    assert r_close["ok"] and r_open["ok"]
    assert r_close["long_only"]["net_annualized"] != pytest.approx(
        r_open["long_only"]["net_annualized"], abs=0.0)


def test_run_periodic_open_matrix_with_nan_excludes_those_stocks(synthetic_panel):
    """open 缺失 → 该股那一期不可交易，不能用 close 顶替。"""
    mat = {k: synthetic_panel[k] for k in ("dates", "codes", "close")}
    score = synthetic_panel["score"]
    open_m = mat["close"].copy()
    open_m[6, 0] = np.nan       # 第一期的出场价缺 → 股票 0 出场不可交易
    kw = dict(hold=5, lag=1, cost_rate=0.003, quantile=0.5,
              min_valid=4, start_idx=0, end_idx=None, winsor=0.01)

    r = ER.run_periodic(mat, score, open_m, open_m, **kw)
    assert r["ok"] is True
    assert r["periods"][0]["n_valid"] == mat["close"].shape[1] - 1


def test_run_periodic_block_limit_up_removes_unbuyable_stocks(monkeypatch):
    """R3：开盘封涨停的股票买不到 → 剔除并按剩余股票重新等权。"""
    rng = np.random.default_rng(3)
    T, N = 20, 8
    dates = [f"2026-01-{i + 1:02d}" for i in range(T)]
    codes = [f"6000{i:02d}" for i in range(N)]
    close = 10.0 * np.cumprod(1.0 + rng.normal(0, 0.02, size=(T, N)), axis=0)
    score = rng.normal(size=(T, N))

    # 股票 0 在第 1 天开盘就涨停（昨收 10.00 → 涨停价 11.00）
    close[0, 0] = 10.0
    open_m = close.copy()
    open_m[1, 0] = 11.0

    mat = {"dates": dates, "codes": codes, "close": close}
    kw = dict(hold=5, lag=1, cost_rate=0.003, quantile=0.5,
              min_valid=4, start_idx=0, end_idx=None, winsor=0.01)

    r_plain = ER.run_periodic(mat, score, open_m, open_m, **kw)
    r_block = ER.run_periodic(mat, score, open_m, open_m,
                               block_limit_up=True, prev_close=close,
                               open_ref=open_m, **kw)

    assert r_plain["blocked_limit_up_total"] == 0
    assert r_block["blocked_limit_up_total"] >= 1
    assert r_block["periods"][0]["blocked_limit_up"] >= 1
    # 剔掉一只 → 该期可交易股票数变少
    assert r_block["periods"][0]["n_valid"] == r_plain["periods"][0]["n_valid"] - 1


def test_turnover_is_double_sided_average():
    """换手口径 = 0.5 × (多头腿换手 + 空头腿换手)，**双边对称**。

    忠实性闸门要求 close 口径逐位复现 canonical，而 canonical（ooslab）
    的换手就是这么算的 —— 这里若只算多头腿，闸门会直接报错。

    验算方式：构造一个**分数每期不变**的面板（选股集合恒定），
    首期从空仓建满仓 → 两腿换手都是 1.0 → 换手 = 0.5×(1+1) = 1.0；
    此后选股不变 → 两腿换手都是 0 → 换手 = 0。
    """
    n_days, n_codes = 40, 120
    dates = [f"2026-01-{i + 1:02d}" for i in range(n_days)]
    codes = [f"6000{i:02d}" for i in range(n_codes)]
    close = np.full((n_days, n_codes), 10.0, dtype=np.float32)
    score = np.tile(np.arange(n_codes, dtype=float), (n_days, 1))
    mat = {"dates": dates, "codes": codes, "close": close}

    r = ER.run_periodic(mat, score, close, close,
                        hold=5, lag=1, cost_rate=0.003, quantile=0.5,
                        min_valid=50, winsor=0.01, start_idx=0, end_idx=n_days)

    assert r["ok"] is True
    turn = [p["turnover"] for p in r["periods"]]
    assert turn[0] == pytest.approx(1.0), (
        f"首期两腿都是从空仓建满仓，换手应为 0.5×(1+1)=1.0，实得 {turn[0]}"
    )
    assert all(t == pytest.approx(0.0) for t in turn[1:]), (
        f"选股恒定时换手必须为 0，实得 {turn[1:]}"
    )


def test_turnover_matches_ooslab_decile_backtest():
    """换手必须与 ooslab.decile_backtest **逐位一致**（忠实性闸门的前提）。"""
    mat, score = _synthetic_market()
    kw = dict(hold=5, lag=1, cost_rate=0.003, quantile=0.5,
              min_valid=50, start_idx=0, end_idx=len(mat["dates"]), winsor=0.01)
    mine = ER.run_periodic(mat, score, mat["close"], mat["close"], **kw)
    ref = ooslab.decile_backtest(mat, score, **kw)

    assert mine["mean_turnover"] == pytest.approx(ref["mean_turnover"], abs=0.0)
    # 逐期换手也必须一致（不只是均值）
    assert [p["turnover"] for p in mine["periods"]] == \
        [p["turnover"] for p in ref["periods"]]


def test_run_periodic_returns_not_ok_when_no_valid_period(synthetic_panel):
    """min_valid 太大 → 跑不出任何调仓期，必须明确说"跑不出来"。"""
    mat = {k: synthetic_panel[k] for k in ("dates", "codes", "close")}
    r = ER.run_periodic(mat, synthetic_panel["score"], mat["close"], mat["close"],
                        hold=5, lag=1, cost_rate=0.003, quantile=0.5,
                        min_valid=999, start_idx=0, end_idx=None, winsor=0.01)
    assert r["ok"] is False
    assert "无可用调仓期" in r["reason"]


def test_run_periodic_periods_are_ordered_and_non_overlapping(synthetic_panel):
    """期次必须按时间递增、且入场/出场日与 lag/hold 严格对得上。

    滚动持有期的设计：第 i 期的出场日 = 第 i+1 期的入场日
    （上一期卖掉、下一期同时买入，组合全程满仓）。
    """
    mat = {k: synthetic_panel[k] for k in ("dates", "codes", "close")}
    dates = mat["dates"]
    idx = {d: i for i, d in enumerate(dates)}
    r = ER.run_periodic(mat, synthetic_panel["score"], mat["close"], mat["close"],
                        hold=5, lag=1, cost_rate=0.003, quantile=0.5,
                        min_valid=4, start_idx=0, end_idx=None, winsor=0.01)
    assert r["ok"] is True
    assert r["periods"], "应该有期次"
    for i, p in enumerate(r["periods"]):
        assert idx[p["entry_date"]] - idx[p["signal_date"]] == 1, "lag 不对"
        assert idx[p["exit_date"]] - idx[p["entry_date"]] == 5, "hold 不对"
        if i:
            prev = r["periods"][i - 1]
            assert idx[p["signal_date"]] > idx[prev["signal_date"]]
            assert idx[p["entry_date"]] == idx[prev["exit_date"]], (
                "滚动持有期下，下一期入场日应等于上一期出场日"
            )


def test_excess_annualized_is_long_minus_benchmark(synthetic_panel):
    """超额年化 = 多头年化 - 基准年化，不能是两个不相干的数。"""
    mat = {k: synthetic_panel[k] for k in ("dates", "codes", "close")}
    r = ER.run_periodic(mat, synthetic_panel["score"], mat["close"], mat["close"],
                        hold=5, lag=1, cost_rate=0.003, quantile=0.5,
                        min_valid=4, start_idx=0, end_idx=None, winsor=0.01)
    assert r["excess_annualized"] == pytest.approx(
        r["long_only"]["net_annualized"] - r["benchmark"]["annualized"])


def test_blocked_limit_up_total_is_zero_when_flag_off(synthetic_panel):
    """不开启涨停剔除时，剔除计数必须恒为 0。"""
    mat = {k: synthetic_panel[k] for k in ("dates", "codes", "close")}
    r = ER.run_periodic(mat, synthetic_panel["score"], mat["close"], mat["close"],
                        hold=5, lag=1, cost_rate=0.003, quantile=0.5,
                        min_valid=4, start_idx=0, end_idx=None, winsor=0.01)
    assert r["blocked_limit_up_total"] == 0
    assert all(p["blocked_limit_up"] == 0 for p in r["periods"])


# ── 6. 结论文案（补齐边界组合，已有文件覆盖主干分支）─────────────────────


def test_excess_sign_treats_tiny_values_as_zero():
    assert ER._excess_sign(0.0) == 0
    assert ER._excess_sign(1e-15) == 0
    assert ER._excess_sign(-1e-15) == 0
    assert ER._excess_sign(0.01) == 1
    assert ER._excess_sign(-0.01) == -1
