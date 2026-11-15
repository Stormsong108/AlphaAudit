"""前瞻纸面跟踪账本（forward_ledger）的回归测试。

账本四条铁律里，有三条必须用测试锁死：

    L1 规则冻结：spec_hash 不匹配时**必须抛错**，不能静默重置账本。
        改规则 = 开一条新账本；在同一条时间线上改规则后继续记，
        前后两段就不是同一个策略了。
    L4 只追加：账本是 append-only 的 JSONL，每行带前一条的哈希。
        改历史行会立刻暴露（哈希链断裂）。

另外锁死：
    - 哈希链能检出篡改（改一行内容 → verify_chain 必须 ok=False）
    - _summarize 的统计口径（累计超额 = 各期复利连乘）
    - _zip_periods 把「现实成交」和「理论成交」两套口径按日期对齐

测试全部在 pytest 临时目录里跑：账本文件、状态文件、摘要文件都被
monkeypatch 到 tmp_path，绝不碰 data/forward/ 下的真账本。
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
import forward_ledger as FL  # noqa: E402


# ── 合成 fixture ───────────────────────────────────────────────────────────


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    """把账本的所有路径重定向到 tmp_path，并屏蔽真数据读取。"""
    monkeypatch.setattr(FL, "LEDGER_DIR", tmp_path)
    monkeypatch.setattr(FL, "LEDGER_FILE", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(FL, "STATE_FILE", tmp_path / "ledger_state.json")
    monkeypatch.setattr(FL, "SUMMARY_FILE", tmp_path / "ledger_summary.json")
    # ST 判定依赖真 universe.json —— 换成空目录，判定结果确定（全 False）
    monkeypatch.setattr(ER, "ROOT", tmp_path)
    return tmp_path


def _fake_spec(hash_: str = "spec-hash-AAAA") -> dict:
    return {
        "factor_names": ["f_a", "f_b"],
        "signed_weights": [1.0, -1.0],
        "factor_means": {"f_a": 0.01, "f_b": -0.02},
        "hold": 5,
        "lag": 1,
        "quantile": 0.5,
        "construction": "decile",
        "cost_roundtrip": 0.003,
        "winsor": 0.01,
        "min_valid": 4,
        "benchmark": "equal_weight_cross_section_winsorized",
        "source": "<fake>",
        "source_generated_at": "2026-11-14 00:00:00",
        "spec_hash": hash_,
    }


@pytest.fixture
def fake_spec(monkeypatch):
    """替换 frozen_spec.load_spec：返回可控的规格（含可控的 spec_hash）。"""
    state = {"spec": _fake_spec()}

    def _load_spec(path=None):
        return dict(state["spec"])

    monkeypatch.setattr(FL.frozen_spec, "load_spec", _load_spec)
    return state


@pytest.fixture
def fake_panel(monkeypatch):
    """替换 ooslab.load_panel / composite_score / panel_open.load_open_panel。

    造 30 个交易日 × 24 只股票的确定性假面板，完全离线。
    股票数不能少于 20 —— ooslab.cs_zscore 要求横截面至少 20 个有效值，
    少于这个数整行 z-score 全是 NaN，跑不出任何期次。
    """
    rng = np.random.default_rng(20261114)
    T, N = 30, 24
    dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(T)]
    codes = [f"6000{i:02d}" for i in range(N)]
    close = 10.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, size=(T, N)), axis=0)
    factors = {
        "f_a": rng.normal(size=(T, N)),
        "f_b": rng.normal(size=(T, N)),
    }
    mat = {"dates": dates, "codes": codes, "close": close, "factors": factors}
    open_m = close * (1.0 + rng.normal(0, 0.005, size=(T, N)))

    monkeypatch.setattr(FL.ooslab, "load_panel", lambda *a, **k: mat)
    monkeypatch.setattr(
        FL.panel_open, "load_open_panel",
        lambda *a, **k: {"open": open_m, "dates": dates, "codes": codes,
                         "cached": False},
    )
    return {"mat": mat, "open": open_m}


def _row(i: int, excess: float = 0.01) -> dict:
    return {
        "signal_date": f"2026-01-{i + 1:02d}",
        "entry_date": f"2026-01-{i + 2:02d}",
        "exit_date": f"2026-01-{i + 7:02d}",
        "n_valid": 100,
        "n_top": 50,
        "blocked_limit_up": 0,
        "turnover": 0.2,
        "cost_charged": 0.0006,
        "gross_real": excess + 0.002,
        "net_real": excess,
        "bench_real": 0.0,
        "excess_real": excess,
        "gross_theory": excess + 0.004,
        "net_theory": excess + 0.002,
        "excess_theory": excess + 0.002,
        "execution_drag": 0.002,
    }


# ── 1. 哈希链：能检出篡改 ─────────────────────────────────────────────────


def test_row_hash_is_deterministic_and_ignores_hash_field():
    """同一行内容 → 同一个 hash；hash 字段本身不参与计算。"""
    r = _row(0)
    h1 = FL._row_hash(r, "prev")
    h2 = FL._row_hash(r, "prev")
    assert h1 == h2
    assert h1 != FL._row_hash(r, "other-prev"), "前一条的哈希必须参与计算"

    r2 = dict(r)
    r2["hash"] = "whatever"
    assert FL._row_hash(r2, "prev") == h1, "hash 字段自己不该参与计算"


def test_row_hash_changes_when_content_changes():
    """改一个数字 → hash 变（这就是篡改能被发现的原因）。"""
    base = FL._row_hash(_row(0), "")
    changed = FL._row_hash({**_row(0), "excess_real": 0.02}, "")
    assert base != changed


def test_append_builds_a_valid_chain(ledger_env):
    """_append 写出来的行必须自洽（verify_chain 通过）。"""
    FL._append([_row(0), _row(1)])
    rows = FL.read_ledger()
    assert len(rows) == 2
    assert FL.verify_chain(rows)["ok"] is True


def test_verify_chain_detects_tampered_content(ledger_env):
    """**改一行内容 → verify_chain 必须报断裂。**"""
    FL._append([_row(0), _row(1), _row(2)])
    assert FL.verify_chain()["ok"] is True

    # 直接改磁盘上的第一行的数字（不改 hash）
    lines = FL.LEDGER_FILE.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    obj["excess_real"] = 0.5          # 篡改：把这一期的超额改大
    lines[0] = json.dumps(obj, ensure_ascii=False)
    FL.LEDGER_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = FL.verify_chain()
    assert res["ok"] is False, "篡改被放过了 —— 哈希链形同虚设"
    assert res["broken_at"] == 0, "必须指出断在第几行"
    assert "断裂" in res["detail"]


def test_verify_chain_detects_tampered_hash(ledger_env):
    """把 hash 字段本身改掉 → 一样要检出。"""
    FL._append([_row(0), _row(1)])
    rows = FL.read_ledger()
    rows[1]["hash"] = "0" * 64
    res = FL.verify_chain(rows)
    assert res["ok"] is False
    assert res["broken_at"] == 1


def test_verify_chain_detects_deleted_row(ledger_env):
    """删掉中间一行 → 后面的 prev_hash 对不上 → 断裂。"""
    FL._append([_row(0), _row(1), _row(2)])
    rows = FL.read_ledger()
    del rows[1]
    res = FL.verify_chain(rows)
    assert res["ok"] is False
    assert res["broken_at"] == 1


def test_verify_chain_on_empty_ledger_is_ok(ledger_env):
    """空账本不是"坏"，是"还没记"。"""
    res = FL.verify_chain()
    assert res["ok"] is True
    assert res["rows"] == 0


def test_verify_chain_reports_head_hash(ledger_env):
    """链尾哈希要能被报出来（用于溯源）。"""
    FL._append([_row(0), _row(1)])
    res = FL.verify_chain()
    assert res["head_hash"] == FL.read_ledger()[-1]["hash"]


# ── 2. append-only 语义 ───────────────────────────────────────────────────


def test_append_only_never_rewrites_existing_lines(ledger_env):
    """**只追加**：追加新行不能改动已有行的任何一个字节。"""
    FL._append([_row(0)])
    first_line_before = FL.LEDGER_FILE.read_text(encoding="utf-8").splitlines()[0]

    FL._append([_row(1)])
    lines = FL.LEDGER_FILE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == first_line_before, "老行被改写了 —— 这不是 append-only"


def test_append_continues_the_hash_chain(ledger_env):
    """第二批追加的行必须接在第一批的链尾上。"""
    FL._append([_row(0)])
    h1 = FL.read_ledger()[-1]["hash"]
    FL._append([_row(1)])
    rows = FL.read_ledger()
    assert rows[1]["prev_hash"] == h1
    assert FL.verify_chain(rows)["ok"] is True


def test_read_ledger_skips_blank_and_corrupt_lines(ledger_env):
    """坏行不能把整个账本读崩（跳过即可，哈希链会照常报断裂）。"""
    FL._append([_row(0)])
    with FL.LEDGER_FILE.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("{这行不是合法 JSON\n")
    rows = FL.read_ledger()
    assert len(rows) == 1


def test_read_ledger_on_missing_file_returns_empty(ledger_env):
    assert FL.read_ledger() == []


# ── 3. L1 规则冻结：spec_hash 不匹配必须抛错 ──────────────────────────────


def test_init_ledger_raises_when_spec_hash_differs(ledger_env, fake_spec):
    """**规则指纹不一致 → 必须抛错，不能静默重置账本。**

    静默重置意味着：旧账本前半段被抹掉，或者更糟 —— 新规则接着旧账本记，
    前后两段不是同一个策略，结论全部作废。
    """
    FL.save_state({"spec_hash": "OLD-HASH", "start_date": "2026-01-01"})
    before = FL.STATE_FILE.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError) as ei:
        FL.init_ledger(verbose=False)

    assert "规则指纹" in str(ei.value) or "另一套策略" in str(ei.value)
    # 关键：状态文件不能被静默改写
    assert FL.STATE_FILE.read_text(encoding="utf-8") == before, (
        "状态文件被静默重置了 —— 必须抛错让人来决定"
    )


def test_init_ledger_reuses_existing_state_when_hash_matches(ledger_env,
                                                             fake_spec):
    """指纹一致 → 沿用现有账本，不重建、不覆盖。"""
    FL.save_state({"spec_hash": fake_spec["spec"]["spec_hash"],
                   "start_date": "2026-01-01"})
    before = FL.STATE_FILE.read_text(encoding="utf-8")
    st = FL.init_ledger(verbose=False)
    assert st["spec_hash"] == fake_spec["spec"]["spec_hash"]
    assert FL.STATE_FILE.read_text(encoding="utf-8") == before


def test_init_ledger_creates_state_with_frozen_rules(ledger_env, fake_spec,
                                                      fake_panel):
    """新建账本 → 冻结规则、记 spec_hash、起点钉在面板最后一天。"""
    st = FL.init_ledger(verbose=False)
    assert st["spec_hash"] == fake_spec["spec"]["spec_hash"]
    assert st["spec"]["hold"] == 5
    assert st["start_date"] == fake_panel["mat"]["dates"][-1]
    assert st["panel_last_date"] == fake_panel["mat"]["dates"][-1]
    assert st["n_codes"] == 24
    assert FL.STATE_FILE.exists()


def test_advance_ledger_raises_when_spec_hash_differs(ledger_env, fake_spec):
    """**推进账本时规则指纹对不上 → 必须抛错，不能继续记。**"""
    FL.save_state({"spec_hash": "OLD-HASH", "start_date": "2026-01-01"})

    with pytest.raises(RuntimeError) as ei:
        FL.advance_ledger(verbose=False)

    assert "规则指纹" in str(ei.value)
    # 不能有半截数据落盘
    assert not FL.LEDGER_FILE.exists() or FL.read_ledger() == []


def test_advance_ledger_requires_a_ledger(ledger_env, fake_spec, fake_panel):
    """账本还没建 → 明确报错让人先 init。"""
    with pytest.raises(RuntimeError) as ei:
        FL.advance_ledger(verbose=False)
    assert "init_ledger" in str(ei.value)


def _started_ledger(fake_spec, fake_panel) -> dict:
    """把账本状态写成"已建账、起点在面板开头"，这样推进才有期次可记。

    真实流程里 init_ledger 会把起点钉在面板最后一天（等新数据长出来），
    所以建账当天确实是 0 期 —— 那是另一条性质，单独有测试。
    """
    st = {
        "ledger_version": FL.LEDGER_VERSION,
        "spec_hash": fake_spec["spec"]["spec_hash"],
        "spec": fake_spec["spec"],
        "start_date": fake_panel["mat"]["dates"][0],
        "panel_last_date": fake_panel["mat"]["dates"][-1],
    }
    FL.save_state(st)
    return st


def test_advance_ledger_writes_rows_and_summary(ledger_env, fake_spec,
                                                 fake_panel):
    """推进账本 → 写入期次行、生成摘要、哈希链通过。"""
    _started_ledger(fake_spec, fake_panel)
    summary = FL.advance_ledger(verbose=False)

    assert summary["n_periods"] >= 1
    assert summary["chain"]["ok"] is True
    assert summary["spec_hash"] == fake_spec["spec"]["spec_hash"]
    assert FL.SUMMARY_FILE.exists()
    rows = FL.read_ledger()
    assert len(rows) == summary["n_periods"]
    for r in rows:
        assert "signal_date" in r and "exit_date" in r
        assert "excess_real" in r and "excess_theory" in r


def test_advance_ledger_is_idempotent(ledger_env, fake_spec, fake_panel):
    """同一期不会写两次（幂等）。"""
    _started_ledger(fake_spec, fake_panel)
    FL.advance_ledger(verbose=False)
    n1 = len(FL.read_ledger())
    summary2 = FL.advance_ledger(verbose=False)
    n2 = len(FL.read_ledger())

    assert n2 == n1, "同一期被写了两次"
    assert summary2["new_periods_written"] == 0


def test_freshly_created_ledger_has_zero_periods(ledger_env, fake_spec,
                                                  fake_panel):
    """建账当天就是 0 期 —— 账本起点钉在面板最后一天，等新数据长出来。

    这不是 bug，是设计：账本必须从"今天之后"开始记，
    不能拿历史回测期次冒充前瞻样本。
    """
    FL.init_ledger(verbose=False)
    summary = FL.advance_ledger(verbose=False)
    assert summary["n_periods"] == 0
    assert summary["new_periods_written"] == 0
    assert FL.read_ledger() == []


# ── 4. _summarize 统计口径 ────────────────────────────────────────────────


def test_summarize_of_empty_ledger_is_minimal(ledger_env):
    """空账本 → 只有 n_periods，不能编出别的数。"""
    s = FL._summarize([])
    assert s == {"n_periods": 0}


def test_summarize_cumulative_is_compounded_not_summed(ledger_env):
    """累计超额 = 各期 (1+超额) 连乘 - 1，不是简单相加。"""
    rows = [_row(0, 0.01), _row(1, 0.02)]
    s = FL._summarize(rows)
    expected = (1.01 * 1.02) - 1.0
    assert s["real"]["cumulative"] == pytest.approx(expected)
    assert s["real"]["cumulative"] != pytest.approx(0.03)


def test_summarize_mean_and_positive_ratio(ledger_env):
    rows = [_row(0, 0.01), _row(1, -0.02), _row(2, 0.03)]
    s = FL._summarize(rows)
    assert s["n_periods"] == 3
    assert s["real"]["mean_per_period"] == pytest.approx((0.01 - 0.02 + 0.03) / 3)
    assert s["real"]["positive_ratio"] == pytest.approx(2 / 3)
    assert s["first_signal_date"] == rows[0]["signal_date"]
    assert s["last_exit_date"] == rows[-1]["exit_date"]


def test_summarize_annualization_uses_252_over_hold(ledger_env):
    """年化 = 每期均值 × (252 / 5)。"""
    rows = [_row(0, 0.01), _row(1, 0.01)]
    s = FL._summarize(rows)
    assert s["real"]["annualized"] == pytest.approx(0.01 * (252.0 / 5.0))


def test_summarize_splits_half_when_enough_periods(ledger_env):
    """期数够多 → 给出前后半段对比（回答"会不会衰减"）。"""
    rows = [_row(i, 0.01 if i < 2 else -0.01) for i in range(4)]
    s = FL._summarize(rows)
    assert set(s["segments"]) == {"前半段", "后半段"}
    assert s["segments"]["前半段"]["mean_per_period"] > 0
    assert s["segments"]["后半段"]["mean_per_period"] < 0
    assert s["segments"]["后半段"]["n"] == 2


def test_summarize_omits_segments_when_too_few_periods(ledger_env):
    """期数太少 → 不给前后半段（样本不足不能下结论）。"""
    s = FL._summarize([_row(0, 0.01), _row(1, 0.02)])
    assert s["segments"] == {}


def test_summarize_counts_blocked_limit_up(ledger_env):
    """因涨停买不到被剔除的次数要累计进来（回答"折损在哪"）。"""
    rows = [_row(0), _row(1)]
    rows[0]["blocked_limit_up"] = 3
    rows[1]["blocked_limit_up"] = 4
    s = FL._summarize(rows)
    assert s["blocked_limit_up_total"] == 7


def test_summarize_std_is_zero_for_single_period(ledger_env):
    """只有一期时标准差没有统计意义 → 记 0，不能崩。"""
    s = FL._summarize([_row(0, 0.01)])
    assert s["real"]["std"] == 0.0


# ── 5. _zip_periods：两套口径按日期对齐 ───────────────────────────────────


def test_zip_periods_aligns_real_and_theory(ledger_env):
    """现实成交与理论成交按 (信号日, 入场日, 出场日) 对齐合并。"""
    spec = _fake_spec()
    real = [{
        "signal_date": "2026-01-01", "entry_date": "2026-01-02",
        "exit_date": "2026-01-09", "n_valid": 100, "n_top": 50,
        "blocked_limit_up": 2, "turnover": 0.2,
        "long": 0.011, "bench": 0.001,
    }]
    theory = [{
        "signal_date": "2026-01-01", "entry_date": "2026-01-02",
        "exit_date": "2026-01-09", "n_valid": 100, "n_top": 50,
        "blocked_limit_up": 0, "turnover": 0.2,
        "long": 0.013, "bench": 0.001,
    }]
    out = FL._zip_periods(real, theory, spec)
    assert len(out) == 1
    r = out[0]
    cost = spec["cost_roundtrip"] * 0.2
    assert r["net_real"] == pytest.approx(0.011 - cost)
    assert r["net_theory"] == pytest.approx(0.013 - cost)
    assert r["excess_real"] == pytest.approx(0.011 - cost - 0.001)
    assert r["execution_drag"] == pytest.approx(0.013 - cost - (0.011 - cost))
    assert r["blocked_limit_up"] == 2


def test_zip_periods_drops_unmatched_real_periods(ledger_env):
    """理论口径没有对应期次的现实期次 → 不合并（不能凭空造一个对照）。"""
    spec = _fake_spec()
    real = [{
        "signal_date": "2026-01-01", "entry_date": "2026-01-02",
        "exit_date": "2026-01-09", "n_valid": 100, "n_top": 50,
        "blocked_limit_up": 0, "turnover": 0.2,
        "long": 0.011, "bench": 0.001,
    }]
    assert FL._zip_periods(real, [], spec) == []


def test_zip_periods_ignores_extra_theory_periods(ledger_env):
    """理论口径多出来的期次不产生行。"""
    spec = _fake_spec()
    p = {
        "signal_date": "2026-01-01", "entry_date": "2026-01-02",
        "exit_date": "2026-01-09", "n_valid": 100, "n_top": 50,
        "blocked_limit_up": 0, "turnover": 0.2,
        "long": 0.011, "bench": 0.001,
    }
    other = {**p, "signal_date": "2026-02-01", "entry_date": "2026-02-02",
             "exit_date": "2026-02-09"}
    out = FL._zip_periods([p], [p, other], spec)
    assert len(out) == 1


# ── 6. status ─────────────────────────────────────────────────────────────


def test_status_reports_chain_and_row_count(ledger_env):
    """status 要把行数和哈希链状态一起报出来。"""
    FL._append([_row(0), _row(1)])
    info = FL.status(verbose=False)
    assert info["n_rows"] == 2
    assert info["chain"]["ok"] is True
    assert info["summary"]["n_periods"] == 2


def test_status_flags_a_broken_chain(ledger_env):
    """账本被改过 → status 必须报出链断裂。"""
    FL._append([_row(0)])
    lines = FL.LEDGER_FILE.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    obj["net_real"] = 99.0
    lines[0] = json.dumps(obj, ensure_ascii=False)
    FL.LEDGER_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    info = FL.status(verbose=False)
    assert info["chain"]["ok"] is False


def test_status_without_state_is_not_an_error(ledger_env):
    """账本还没建 → status 正常返回"没有"，不是抛异常。"""
    info = FL.status(verbose=False)
    assert info["state"] == {}
    assert info["n_rows"] == 0
