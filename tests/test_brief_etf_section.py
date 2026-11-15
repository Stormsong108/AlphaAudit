"""ETF 通道接进每日研究汇报（daily_brief）的回归测试。

这一组测试锁死的是 2026-11-14 立下的 ETF 通道纪律：

  1. **结论由 stage 推导**：ETF 通道的结论只能由产物里的 stage 字段 + 数字
     推导出来。历史上出过两次作废的运行（验证段参与选参、基准算出 nan%），
     那些旧数字一律不许进日报 —— 所以正文里不许出现任何硬编码的 ETF 数字
     或结论。
  2. **取不到就如实说**：文件缺失 / JSON 坏掉 / stage 不是 completed，
     一律不抛异常，并且正文里必须有一行说清楚为什么取不到（不是空白）。
  3. **validation_failed 不许报测试段数字**：这轮没进测试段，就没有
     "测试段净年化"这种数字可报 —— 也不许把"验证段没通过"改写成
     "毛收益为负"之类的结论。
  4. **数字防漂移**：正文里出现的 ETF 百分比必须能在 etf_channel.json 里
     找到出处（缩尾/未缩尾/中位数三个口径都要在）。

测试不依赖真实的 etf_channel.json（那个文件正在被重新生成，数字会变）——
用 monkeypatch 把路径指到 tmp_path，再写临时产物。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import daily_brief as DB  # noqa: E402


# ── fixture：把 ETF 相关路径重定向到 tmp_path ─────────────────────────────


@pytest.fixture
def etf_env(tmp_path, monkeypatch):
    """把 daily_brief 用到的 ETF 产物路径指到 tmp_path。

    同时把其余路径也重定向 —— 否则 build_brief 会读真产物、真历史，
    测试就不再是纯函数了。
    """
    monkeypatch.setattr(DB, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(DB, "HISTORY_FILE", tmp_path / "state" / "brief_history.json")
    monkeypatch.setattr(DB, "BRIEF_DIR", tmp_path / "briefs")
    monkeypatch.setattr(DB, "OOS_FILE", tmp_path / "oos_latest.json")
    monkeypatch.setattr(DB, "REALISM_FILE", tmp_path / "execution_realism.json")
    monkeypatch.setattr(DB, "LEDGER_SUMMARY", tmp_path / "ledger_summary.json")
    monkeypatch.setattr(DB, "LEDGER_STATE", tmp_path / "ledger_state.json")
    monkeypatch.setattr(DB, "ANATOMY_FILE", tmp_path / "signal_anatomy.json")
    monkeypatch.setattr(DB, "QUEUE_FILE", tmp_path / "research_queue.json")
    monkeypatch.setattr(DB, "DAILY_LOOP_FILE", tmp_path / "daily_loop.json")
    monkeypatch.setattr(DB, "CROSS_SEG_FILE", tmp_path / "cross_segment_check.json")
    monkeypatch.setattr(DB, "ETF_CHANNEL_FILE", tmp_path / "etf_channel.json")
    monkeypatch.setattr(DB, "COST_FEASIBILITY_FILE", tmp_path / "cost_feasibility.json")
    return tmp_path


def _write_etf(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "etf_channel.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _write_cost_feasibility(tmp_path: Path, floor_bp: float) -> Path:
    """个股通道的成本地板（只作对照，不是 ETF 数字）。"""
    path = tmp_path / "cost_feasibility.json"
    path.write_text(json.dumps({
        "fee_floor_bp": {"order_ge_20000": floor_bp},
    }, ensure_ascii=False), encoding="utf-8")
    return path


# ── 产物样本 ───────────────────────────────────────────────────────────────
# 数字刻意取得很"不像真的"，这样正文里出现它们就一定是从这份 JSON 读出来的。

GENERATED_AT = "2026-11-14T12:00:00"


def _completed_payload() -> dict:
    return {
        "generated_at": GENERATED_AT,
        "stage": "completed",
        "n_etf": 57,
        "price_mode": "open",
        "cost_model": {
            "commission_rate": 0.00005,
            "commission_min_yuan": 5.0,
            "stamp_duty": 0.0,
            "transfer_fee": 0.0,
            "slippage_bp_edge": 2.0,
        },
        "chosen": {"factor_set": "mom60", "k": 3, "R": 5},
        "frozen_spec": {
            "factor_set": "mom60", "k": 3, "R": 5,
            "capital": 1_000_000.0, "price_mode": "open",
            "spec_hash": "abc123", "frozen_at": GENERATED_AT,
        },
        "test": {
            "metrics": {
                "n_periods": 85,
                # 毛/净每期是小数（-0.00103 = -10.3bp），成本已经是 bp。
                # 恒等式 gross - net == cost/1e4 必须严格成立，否则日报
                # 会判"口径对不上"而拒绝渲染 —— 样本自己就要自洽。
                "gross_per_period": -0.00103,
                "gross_annualized": -0.0519,
                "net_per_period": -0.00132,
                "net_annualized": -0.0665,
                "net_winsor_mean_annualized": -0.0402,
                "net_median_annualized": -0.0388,
                "sharpe": -0.31,
                "win_rate": 0.4235,
                "median_period_net_bp": -13.2,
                "mean_cost_per_period_bp": 2.90,
                "mean_held": 3.0,
                "max_drawdown_ann": -0.1342,
            },
            "benchmark": {
                "n_periods": 85, "annualized": 0.0512,
                "sharpe": 0.42, "win_rate": 0.55,
                "dates": ["2024-01-03", "2026-11-13"],
            },
            "excess_ann": -0.1177,
        },
        "capital_sweep": [
            {"capital": 100_000.0, "mean_cost_per_period_bp": 19.4,
             "cost_floor": {"total_bp_roundtrip": 5.0}},
            {"capital": 1_000_000.0, "mean_cost_per_period_bp": 2.91,
             "cost_floor": {"total_bp_roundtrip": 1.0}},
            {"capital": 10_000_000.0, "mean_cost_per_period_bp": 1.86,
             "cost_floor": {"total_bp_roundtrip": 1.0}},
        ],
        "survivorship_bias_disclosure": (
            "ETF 池为当前存续活跃 ETF 名单，存在幸存者偏差；池内成员按上市日点时生效，"
            "不做向前回填。ETF 标的是宽基/行业指数，死亡率远低于个股，但偏差未消除。"
        ),
        "superseded_runs": [{
            "date": "2026-11-14",
            "issue": "旧实现按验证段超额选参（passed[0]），验证段被当第二个训练段",
        }],
    }


def _validation_failed_payload() -> dict:
    """验证段没通过的产物。

    故意在里面塞一份"测试段"数字 —— 真实产物里 test 是 null，
    但这一版更能证明纪律：stage 不对时，正文一个字都不许从它身上取。
    """
    return {
        "generated_at": GENERATED_AT,
        "stage": "validation_failed",
        "n_etf": 57,
        "chosen": {"factor_set": "mom60", "k": 3, "R": 5},
        "train_grid_top": [{"factor_set": "mom60", "k": 3, "R": 5,
                            "excess_train_ann": -0.0331}],
        "val_rows": [{"factor_set": "mom60", "k": 3, "R": 5,
                      "excess_val_ann": -0.0288}],
        "test": {
            "metrics": {"net_annualized": 0.8765, "gross_per_period": 0.1234,
                        "net_per_period": 0.1111, "mean_cost_per_period_bp": 123.4},
            "benchmark": {"annualized": 0.6543},
            "excess_ann": 0.2222,
        },
    }


# ── 正文里的 ETF 片段怎么切出来 ───────────────────────────────────────────
# 二/五段里 ETF 是一块（4 空格表头 + 6 空格明细）；六/七段里 ETF 是几行
# 6 空格开头的独立条目。统一规则：ETF 表头本身 + 它下面更深的缩进行，
# 以及任何以 6 空格 + "ETF" 开头的行。

ETF_HEADER_RE = re.compile(r"^ {4}ETF ")


def _etf_lines(text: str) -> list[str]:
    """切出简报正文里属于 ETF 通道的那些行。"""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ETF_HEADER_RE.match(ln):
            out.append(ln)
            i += 1
            # 表头下面缩进更深的行都是这一块的内容
            while i < len(lines) and lines[i].startswith("      ") and lines[i].strip():
                out.append(lines[i])
                i += 1
            continue
        if ln.startswith("      ETF "):
            out.append(ln)
        i += 1
    return out


def _pct_values(text: str) -> list[str]:
    """正文里出现的所有百分比显示串（不含正负号）。"""
    return [m.group(1) for m in re.finditer(r"(\d+\.\d+)%", text)]


def _bp_values(text: str) -> list[str]:
    """正文里出现的所有 bp 显示串。"""
    return [m.group(1) for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*bp", text)]


def _allowed_pct_values(payload: dict) -> set[str]:
    """产物里所有能当作百分比出处的**显示串**。

    与 tests/test_plain_report.py 同一套做法：不逐字比对，而是
    "这个百分比必须能在原始数据里找到出处"。这里比那边多一层 ——
    候选串按**显示精度**生成（胜率是 1 位小数，年化是 2 位小数），
    否则 0.4235 → "42.4%" 会被 42.35 卡在容差外。
    """
    out: set[str] = set()

    def collect(obj) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            f = float(obj)
            for mult in (100.0, 1.0):
                for prec in (1, 2, 4):
                    out.add(f"{f * mult:.{prec}f}")
                    out.add(f"{abs(f * mult):.{prec}f}")

    collect(payload)
    return out


def _allowed_bp_values(payload: dict, extra: tuple[float, ...] = ()) -> set[str]:
    """产物里所有能当作 bp 出处的显示串。

    毛/净每期在产物里是小数（要 ×1e4 才是 bp），成本字段已经是 bp，
    两种表示都收进来 —— 出处判定按"数值上能不能对上"来做，不靠猜口径。
    """
    out: set[str] = set()

    def collect(obj) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            f = float(obj)
            for mult in (1.0, 1e4):
                for prec in (1, 2, 4):
                    out.add(f"{f * mult:.{prec}f}")
                    out.add(f"{abs(f * mult):.{prec}f}")

    collect(payload)
    for f in extra:
        for prec in (1, 2, 4):
            out.add(f"{f:.{prec}f}")
            out.add(f"{abs(f):.{prec}f}")
    return out


# ── a. stage=completed：正文必须带齐核心数字 ─────────────────────────────


def test_completed_stage_shows_core_etf_numbers(etf_env):
    """ETF 通道跑完时，第二节必须给出冻结参数、测试段净年化、基准、超额。"""
    payload = _completed_payload()
    _write_etf(etf_env, payload)
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})

    # 冻结参数
    assert "冻结参数" in text
    assert "mom60" in text, "冻结参数里没有因子组"
    assert "每次买 3 只" in text, "冻结参数里没有持仓只数"
    assert "持有 5 个交易日" in text, "冻结参数里没有持有期"

    # 测试段净年化 / 基准年化 / 超额
    assert "-6.65%" in text, "第二段没有测试段净年化"
    assert "+5.12%" in text, "第二段没有基准年化"
    assert "-11.77%" in text, "第二段没有超额"

    # generated_at 必须和数字一起出现（防止日报内部新旧两个数字打架）
    assert GENERATED_AT in text, "ETF 数字旁边没有生成时间戳"


def test_completed_stage_shows_three_calibers_gross_and_net(etf_env):
    """三个口径（缩尾/未缩尾/中位数）并列，毛/净分开，成本 bp 单列。"""
    _write_etf(etf_env, _completed_payload())
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    etf = "\n".join(_etf_lines(text))

    # 三口径
    assert "-4.02%" in etf, "缺缩尾均值口径"
    assert "-6.65%" in etf, "缺未缩尾均值口径"
    assert "-3.88%" in etf, "缺中位数口径"
    # 毛 / 净 分开 + 成本
    assert "每期毛收益" in etf and "每期净收益" in etf and "每期成本" in etf
    assert "2.90 bp" in etf, "每期成本没有用 bp 口径"
    # 胜率 + 最大回撤
    assert "胜率" in etf and "最大回撤" in etf
    # 成本口径（无印花税无过户费）+ 成本地板
    assert "不收印花税" in etf and "不收过户费" in etf
    assert "成本地板" in etf
    # 单笔往返成本地板区间（来自 capital_sweep，不是写死的）
    assert "1.00~5.00 bp" in etf, "成本地板区间不是从产物读的"


def test_completed_stage_uses_generated_at_next_to_numbers(etf_env):
    """数字旁边必须带生成时间戳 —— 面板每天长数据，旧数字会失真。"""
    _write_etf(etf_env, _completed_payload())
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    etf = "\n".join(_etf_lines(text))
    assert GENERATED_AT in etf, "ETF 那一块没有 generated_at"


# ── b. 文件缺失：不抛异常，正文如实说取不到 ──────────────────────────────


def test_missing_etf_file_does_not_crash_and_says_unavailable(etf_env):
    """产物文件不存在 → 简报照常出，并且明说取不到（不是空白）。"""
    _write_cost_feasibility(etf_env, 10.2)
    assert not (etf_env / "etf_channel.json").exists()

    text = DB.build_brief({}, prev={})   # 不抛异常就是第一道关

    assert "ETF 通道" in text, "ETF 通道被静默跳过了 —— 读者会以为系统挂了"
    assert "取不到" in text, "没有明说为什么取不到"
    assert "研究产物文件还没生成" in text, "取不到的原因写得不够具体"
    # 七段式不能塌
    for title in ("一、今天做了什么", "二、分析结果是什么", "三、结果意味着什么",
                  "四、和昨天比变了什么", "五、当前能不能用", "六、主要风险是什么",
                  "七、下一步验证什么"):
        assert title in text, f"缺段落：{title}"


def test_missing_etf_file_gives_no_conclusion(etf_env):
    """取不到就不能给结论 —— 不能凭记忆硬写"ETF 通道救不回来"。"""
    _write_cost_feasibility(etf_env, 10.2)
    text = DB.build_brief({}, prev={})
    sec5 = text.split("五、当前能不能用")[1].split("六、主要风险是什么")[0]

    assert "救不回" not in sec5, "产物都没有就敢下结论"
    assert "不能用作独立策略" not in sec5, "产物都没有就敢下结论"


# ── c. JSON 损坏：同样不抛异常，同样如实说明 ──────────────────────────────


def test_broken_etf_json_does_not_crash_and_says_unavailable(etf_env):
    """产物文件坏了 → 简报照常出，并且说清楚是文件坏了。"""
    (etf_env / "etf_channel.json").write_text("{坏掉的 JSON", encoding="utf-8")
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})

    assert "取不到" in text, "JSON 坏了却没说明"
    assert "文件损坏" in text, "没说清楚是文件坏了"
    assert "二、分析结果是什么" in text, "结构塌了"


def test_broken_etf_json_is_not_read_as_a_result(etf_env):
    """坏 JSON 里就算塞了 stage=completed，也不能被当成有效结果。

    注意：文件必须是**语法坏**的（截断），不能只是"内容看着可疑"。
    """
    (etf_env / "etf_channel.json").write_text(
        '{"stage": "completed", "test": {"metrics": {"net_annualized": 0.8765',  # 截断
        encoding="utf-8",
    )
    _write_cost_feasibility(etf_env, 10.2)
    text = DB.build_brief({}, prev={})
    assert "87.65%" not in text, "坏 JSON 里的数字进了日报"
    assert "取不到" in text, "坏 JSON 没如实说明"


def test_unknown_stage_is_reported_as_not_ready(etf_env):
    """stage 是没见过的东西 → 如实说"这轮还没走完"，不许当 completed 处理。"""
    _write_etf(etf_env, {"generated_at": GENERATED_AT, "stage": "running",
                         "test": {"metrics": {"net_annualized": 0.8765}}})
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    assert "取不到" in text
    assert "还没走完" in text
    assert "87.65%" not in text, "未知 stage 的数字进了日报"


# ── d. validation_failed：不报测试段数字，只说验证段没通过 ────────────────


def test_validation_failed_reports_no_test_segment_numbers(etf_env):
    """验证段没通过 → 不得报任何测试段数字。

    样本里故意放了 net_annualized=0.8765 这类数字：只要它出现在正文里，
    就说明有测试段数字漏出去了。
    """
    _write_etf(etf_env, _validation_failed_payload())
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    etf = "\n".join(_etf_lines(text))

    assert "87.65%" not in text, "验证段没通过却报了测试段净年化"
    assert "测试段净年化" not in etf, "验证段没通过却出现了测试段口径"
    assert "净年化" not in etf, "验证段没通过却出现了净年化数字"
    assert "65.43%" not in text, "验证段没通过却报了基准年化"
    assert "22.22%" not in text, "验证段没通过却报了超额"
    assert "123.4 bp" not in text, "验证段没通过却报了成本 bp"


def test_validation_failed_says_validation_did_not_pass(etf_env):
    """必须明确写出"验证段未通过"以及按纪律停止的原因。"""
    _write_etf(etf_env, _validation_failed_payload())
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    etf = "\n".join(_etf_lines(text))

    assert "验证段未通过" in etf, "没有明说验证段未通过"
    assert "训练段选出的参数在验证段净超额不为正" in etf, "没写清楚为什么停"
    assert "按纪律停止，未进入测试段" in etf, "没写清楚按纪律停止了"
    assert "本轮结果不可用" in etf, "没写清楚本轮结果不可用"


def test_validation_failed_is_not_rewritten_as_a_gross_return_conclusion(etf_env):
    """不许把"验证段没通过"改写成"毛收益为负"这类结论。"""
    _write_etf(etf_env, _validation_failed_payload())
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    sec5 = text.split("五、当前能不能用")[1].split("六、主要风险是什么")[0]

    assert "毛收益本身就是" not in sec5, "把验证段没通过改写成了毛收益结论"
    assert "问题不在成本，在信号本身" not in sec5, "把验证段没通过改写成了信号结论"


def test_validation_failed_next_step_says_rerun_without_switching_params(etf_env):
    """下一步必须说清：验证段没通过之前不换参、不看测试段。"""
    _write_etf(etf_env, _validation_failed_payload())
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    sec7 = text.split("七、下一步验证什么")[1]
    assert "不换参" in sec7, "下一步没说清验证段没通过之前不换参"


# ── e. 数字防漂移：正文里的 ETF 数字必须能在产物里找到出处 ─────────────────


def test_etf_percentages_trace_back_to_source(etf_env):
    """正文里出现的每个 ETF 百分比，都必须能在 etf_channel.json 里找到出处。"""
    payload = _completed_payload()
    _write_etf(etf_env, payload)
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    etf = "\n".join(_etf_lines(text))
    allowed = _allowed_pct_values(payload)

    found = _pct_values(etf)
    assert found, "ETF 那一块一个百分比都没有 —— 说明根本没渲染数字"
    for val in found:
        assert val in allowed, f"ETF 数字 {val}% 在 etf_channel.json 里找不到出处（数字漂移）"


def test_etf_bp_values_trace_back_to_source(etf_env):
    """正文里的 ETF bp 数字（毛/净/成本/地板）也要能对上产物。"""
    payload = _completed_payload()
    _write_etf(etf_env, payload)
    stock_floor = 10.2
    _write_cost_feasibility(etf_env, stock_floor)

    text = DB.build_brief({}, prev={})
    etf = "\n".join(_etf_lines(text))
    # 个股通道地板来自另一份产物（只作对照），单独放行
    allowed = _allowed_bp_values(payload, extra=(stock_floor,))

    found = _bp_values(etf)
    assert found, "ETF 那一块一个 bp 数字都没有"
    for val in found:
        assert val in allowed, f"ETF 数字 {val} bp 在产物里找不到出处（数字漂移）"


def test_etf_section_survives_when_stock_floor_is_unavailable(etf_env):
    """个股通道对照产物取不到时，ETF 那一块照样出（只是少了对照）。"""
    _write_etf(etf_env, _completed_payload())
    # 不写 cost_feasibility.json

    text = DB.build_brief({}, prev={})
    assert "-6.65%" in text
    assert "10.20" not in text, "个股通道地板取不到却写了个数"


# ── 六/七段：幸存者偏差用产物原文，下一步按 stage 说话 ─────────────────────


def test_survivorship_bias_uses_source_wording_verbatim(etf_env):
    """幸存者偏差必须用产物里的原文，不许自己编一套说法。"""
    payload = _completed_payload()
    _write_etf(etf_env, payload)
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    assert payload["survivorship_bias_disclosure"] in text, "幸存者偏差没用产物原文"


def test_superseded_run_is_flagged_as_void(etf_env):
    """作废的历史运行必须在风险里点명，不得被当成有效结果。"""
    _write_etf(etf_env, _completed_payload())
    _write_cost_feasibility(etf_env, 10.2)

    text = DB.build_brief({}, prev={})
    assert "已作废" in text, "没提作废的历史运行"
    assert "不得被引用" in text


# ── 纪律本身：源码里不许写死任何 ETF 数字或结论 ───────────────────────────


def test_source_contains_no_hardcoded_etf_numbers():
    """源码里不许出现两次作废运行里的那些 ETF 数字。

    这条是"纪律的纪律"：只要代码里没有这些常量，日报就不可能引用它们。
    匹配用数字边界（前后不能紧挨别的数字/小数点），否则 "1298 只" 这种
    无关数字会被误判。
    """
    source = (ROOT / "daily_brief.py").read_text(encoding="utf-8")
    # 两次作废运行里的 ETF 数字（validation_leak + nan_bench）
    forbidden = [
        "-10.33", "-12.38", "-6.15", "+14.71", "1.86",
        "-11.91", "-122.67", "14.72", "-0.1237", "0.1471",
    ]
    for token in forbidden:
        pat = r"(?<![\d.])" + re.escape(token) + r"(?![\d])"
        assert not re.search(pat, source), (
            f"daily_brief.py 里写死了 {token} —— ETF 数字只能运行时读"
        )


def test_etf_conclusion_follows_the_data_not_a_hardcoded_verdict(etf_env):
    """ETF 结论必须跟着产物的数字走 —— 数字翻面，结论就得跟着翻。

    这是"不许写死结论"的行为级验证：把毛收益从负改成正，
    第五段的判断必须从"信号本身不行"翻成"数字转正但样本不够"。
    """
    payload = _completed_payload()
    _write_etf(etf_env, payload)
    _write_cost_feasibility(etf_env, 10.2)

    neg_text = DB.build_brief({}, prev={})
    neg_sec5 = neg_text.split("五、当前能不能用")[1].split("六、主要风险是什么")[0]
    assert "不能用作独立策略" in neg_sec5, "毛收益为负时结论不对"
    assert "问题不在成本，在信号本身" in neg_sec5, "毛收益为负时原因说错了"

    # 把毛/净翻成正的。恒等式 gross - net == cost/1e4 要继续成立，
    # 所以新毛 = 新净 + 成本（新净也翻正，新毛比新净大一个成本）。
    payload["test"]["metrics"]["net_per_period"] = 0.00103
    payload["test"]["metrics"]["gross_per_period"] = 0.00103 + 2.90 / 1e4
    _write_etf(etf_env, payload)

    pos_text = DB.build_brief({}, prev={})
    pos_sec5 = pos_text.split("五、当前能不能用")[1].split("六、主要风险是什么")[0]
    assert "不能用作独立策略" not in pos_sec5, "数字翻正了结论却没跟着翻 —— 结论写死了"
    assert "问题不在成本，在信号本身" not in pos_sec5, "数字翻正了原因却没跟着翻"
    assert "这轮数字是正的" in pos_sec5, "数字翻正时结论不对"


def test_load_etf_channel_returns_none_on_every_failure_mode(etf_env, monkeypatch):
    """load_etf_channel()：三种失败都返回 None，且不抛异常。"""
    assert DB.load_etf_channel() is None, "文件不存在时不该返回数据"

    (etf_env / "etf_channel.json").write_text("{坏掉的 JSON", encoding="utf-8")
    assert DB.load_etf_channel() is None, "JSON 坏了时不该返回数据"

    _write_etf(etf_env, _validation_failed_payload())
    assert DB.load_etf_channel() is None, "stage 不是 completed 时不该返回数据"

    _write_etf(etf_env, _completed_payload())
    payload = DB.load_etf_channel()
    assert payload is not None and payload["stage"] == "completed"
