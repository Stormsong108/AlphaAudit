"""每日大白话汇报（daily_brief）的回归测试。

设计铁律里最要紧的三条，用测试锁死：

    D1 **零 LLM**：全部从磁盘上的确定性产物拼装。
    D2 **不编数字**：任何数字都必须能指到来源文件。拿不到就写"取不到"，
       绝不用估算值冒充实测值。
    D3 **和昨天比**：从历史快照读昨天做差值。第一次跑没有昨天 →
       明说"首次运行"。

另外锁死：7 个段落一个都不能少（缺一个就不算汇报）。

测试全部把路径常量 monkeypatch 到 tmp_path：
不读真产物、不写真 state/brief_history.json、不碰真 data/briefs/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import daily_brief as DB  # noqa: E402


# ── 7 个段落标题（缺一个都不算汇报）────────────────────────────────────────

SECTION_TITLES = [
    "一、今天做了什么",
    "二、分析结果是什么",
    "三、结果意味着什么",
    "四、和昨天比变了什么",
    "五、当前能不能用",
    "六、主要风险是什么",
    "七、下一步验证什么",
]


# ── fixture：把全部路径重定向到 tmp_path ──────────────────────────────────


@pytest.fixture
def brief_env(tmp_path, monkeypatch):
    """把 daily_brief 用到的所有路径常量指向 tmp_path。

    真产物（data/research/*.json）只读不写；真历史（state/brief_history.json）
    完全不碰 —— 否则跑一次测试就把"昨天"改没了。
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
    return tmp_path


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def _full_snapshot() -> dict:
    return {
        "ts": "2026-11-14 09:00:00",
        "oos": {
            "net_annualized": 0.1478,
            "sharpe": 0.5079,
            "benchmark_annualized": 0.1608,
            "excess_annualized": -0.0130,
            "mean_turnover": 0.2211,
            "n_periods": 143,
        },
        "realism": {
            "fidelity_ok": True,
            "close_excess": -0.0130,
            "open_excess": -0.0264,
            "open_limit_excess": -0.0269,
            "blocked_total": 248,
            "verdict": "已检验",
            "conclusion": "两个口径下超额都为负",
            "generated_at": "2026-11-14 03:19:20",
        },
        "ledger": {
            "n_periods": 0,
            "new_periods": 0,
            "cumulative_excess": None,
            "annualized_excess": None,
            "positive_ratio": None,
            "first_signal_date": None,
            "chain_ok": True,
        },
        "anatomy": {
            "overnight_diff": 0.0041,
            "entry_day_diff": -0.0052,
            "hold_diff": -0.0011,
            "exit_day_diff": 0.0003,
            "r1_gross_diff": -0.0016,
            "r2_gross_diff": -0.0063,
            "untradeable_ann": -0.0199,
            "entry_day_ann": -0.0261,
            "signal_nature": "反转",
            "n_periods": 143,
        },
        "queue": {"步骤A": {"status": "成功", "last_run_at": "2026-11-14 08:00:00"}},
        "daily_loop": {"阶段B": {"status": "成功", "last_run_at": "2026-11-14 08:10:00"}},
    }


# ── 1. 7 段结构 ────────────────────────────────────────────────────────────


def test_brief_has_all_seven_sections(brief_env):
    """7 个问题一个都不能少（缺一个就不算汇报）。"""
    text = DB.build_brief(_full_snapshot(), prev={})
    for title in SECTION_TITLES:
        assert title in text, f"汇报缺少段落：{title}"


def test_brief_has_all_seven_sections_even_with_empty_snapshot(brief_env):
    """**一个数字都没有时，7 段结构依然完整** —— 结构不能因为缺数据就塌。"""
    text = DB.build_brief({}, prev={})
    for title in SECTION_TITLES:
        assert title in text, f"汇报缺少段落：{title}"


def test_brief_starts_with_the_verdict(brief_env):
    """D5 结论前置：第一行就说能不能用，不能让人读到最后才知道。"""
    text = DB.build_brief(_full_snapshot(), prev={})
    head = text.split("\n")[:8]
    assert any("当前状态" in ln for ln in head), "开头没有结论"
    assert "还不能用" in text or "可以用" in text


def test_brief_is_plain_text_without_tables_or_code_blocks(brief_env):
    """D4 纯文字：无表格、无代码块。"""
    text = DB.build_brief(_full_snapshot(), prev={})
    assert "```" not in text
    assert not any(ln.strip().startswith("|") for ln in text.splitlines())


def test_brief_lists_todays_actions(brief_env):
    """有研究动作时要说清楚今天做了什么。"""
    snap = _full_snapshot()
    text = DB.build_brief(snap, prev={})
    assert "步骤A" in text
    assert "阶段B" in text


def test_brief_says_no_action_when_nothing_happened(brief_env):
    """没有动作时明说"今天没有产生新的研究动作"，不能空着。"""
    text = DB.build_brief({}, prev={})
    assert "今天没有产生新的研究动作" in text


# ── 2. D2 不编数字：拿不到就写「取不到」 ──────────────────────────────────


def test_missing_oos_is_reported_as_unavailable(brief_env):
    """OOS 证据文件缺失 → 明说取不到，不能拿别处的数字凑。"""
    text = DB.build_brief({}, prev={})
    assert "取不到（OOS 证据文件缺失）" in text


def test_missing_realism_is_reported_as_unavailable(brief_env):
    """成交真实性检验还没跑 → 明说取不到。"""
    text = DB.build_brief({}, prev={})
    assert "取不到（成交真实性检验还没跑）" in text


def test_partial_numbers_render_as_unavailable_not_zero(brief_env):
    """**数字缺失时输出「取不到」，绝不用 0 冒充实测值。**

    这是最危险的一种错：0.00% 看起来像"实测就是零"，
    而真相是"根本没测出来"。
    """
    snap = _full_snapshot()
    snap["oos"]["excess_annualized"] = None
    snap["realism"]["open_limit_excess"] = None
    text = DB.build_brief(snap, prev={})

    assert "取不到" in text
    # 不能把缺失的超额显示成 0.00%
    assert "跑赢大盘 **+0.00%/年**" not in text
    assert "超额 **+0.00%/年**" not in text


def test_pct_and_num_helpers_return_unavailable_marker():
    """格式化助手：None / 非法值 → 一律「取不到」。"""
    assert DB._pct(None) == "取不到"
    assert DB._num(None) == "取不到"
    assert DB._pct("不是数字") == "取不到"
    assert DB._num("不是数字") == "取不到"
    assert DB._pct(0.0130) == "+1.30%"
    assert DB._num(1298) == "1,298.00"


def test_brief_never_invents_a_number_for_missing_ledger(brief_env):
    """账本还没有数 → 说"还没到能出数的期数"，不编累计超额。"""
    text = DB.build_brief(_full_snapshot(), prev={})
    assert "还没到能出数的期数" in text
    assert "累计超额 **取不到**" not in text


# ── 3. D3 和昨天比 ────────────────────────────────────────────────────────


def test_first_run_without_history_says_first_run(brief_env):
    """**首次运行、没有昨日快照 → 必须明说「首次运行」。**"""
    assert DB.load_history() == [], "前提：没有历史快照"
    text = DB.build_brief(_full_snapshot(), prev=None)
    assert "首次运行" in text
    assert "没有昨天可比" in text


def test_first_run_via_empty_prev_snapshot_says_first_run(brief_env):
    """显式传空快照 → 同样按"首次运行"处理。"""
    text = DB.build_brief(_full_snapshot(), prev={})
    assert "首次运行" in text


def test_with_history_shows_comparison_instead_of_first_run(brief_env):
    """有昨天 → 说清楚变了什么，不再出现"首次运行"。"""
    prev = _full_snapshot()
    prev["oos"]["excess_annualized"] = -0.0200
    prev["realism"]["open_limit_excess"] = -0.0300
    prev["ts"] = "2026-11-13 09:00:00"

    text = DB.build_brief(_full_snapshot(), prev=prev)
    assert "首次运行" not in text
    assert "回测超额" in text
    assert "现实成交超额" in text
    assert "2026-11-13 09:00:00" in text


def test_history_saved_on_disk_is_used_as_previous_snapshot(brief_env):
    """历史快照落到磁盘后，下一次汇报自动拿它做对比。"""
    prev = _full_snapshot()
    prev["oos"]["excess_annualized"] = -0.0200
    DB.save_history(prev)

    text = DB.build_brief(_full_snapshot(), prev=None)
    assert "首次运行" not in text
    assert "昨日" in text


def test_diff_helper_semantics():
    """_diff 的四种情形：没有昨天 / 没变 / 变了 / 新值缺失。"""
    assert DB._diff(None, 0.01) == "无昨日可比"
    assert DB._diff(0.01, None) == "无昨日可比"
    assert DB._diff(0.01, 0.01) == "与昨日相同"
    d = DB._diff(0.01, 0.02)
    assert "昨日" in d and "今日" in d and "+1.00%" in d


# ── 4. build_snapshot：从磁盘产物抓数 ─────────────────────────────────────


def test_build_snapshot_extracts_oos_numbers(brief_env):
    """OOS 产物里的数字要被抓出来，且超额 = 策略 - 基准。"""
    _write(brief_env / "oos_latest.json", {
        "strategy_test_segment": {
            "long_only": {"net_annualized": 0.14779151747007227,
                          "sharpe": 0.5079523251702476},
            "benchmark": {"annualized": 0.16075356602254287},
            "mean_turnover": 0.22110322298759919,
            "n_periods": 143,
        }
    })
    snap = DB.build_snapshot()
    assert snap["oos"]["net_annualized"] == pytest.approx(0.14779151747007227)
    assert snap["oos"]["benchmark_annualized"] == pytest.approx(0.16075356602254287)
    assert snap["oos"]["excess_annualized"] == pytest.approx(
        0.14779151747007227 - 0.16075356602254287)
    assert snap["oos"]["n_periods"] == 143


def test_build_snapshot_extracts_realism_numbers(brief_env):
    """成交真实性检验产物里的三套口径超额都要被抓出来。"""
    _write(brief_env / "execution_realism.json", {
        "fidelity_gate": {"ok": True},
        "r1_close": {"excess_annualized": -0.0129},
        "r2_open": {"excess_annualized": -0.0263},
        "r3_open_limit_blocked": {"excess_annualized": -0.0268,
                                  "blocked_limit_up_total": 248},
        "verdict": {"status": "已检验", "conclusion": "边际不成立"},
        "generated_at": "2026-11-14 03:19:20",
    })
    snap = DB.build_snapshot()
    assert snap["realism"]["fidelity_ok"] is True
    assert snap["realism"]["close_excess"] == pytest.approx(-0.0129)
    assert snap["realism"]["open_excess"] == pytest.approx(-0.0263)
    assert snap["realism"]["open_limit_excess"] == pytest.approx(-0.0268)
    assert snap["realism"]["blocked_total"] == 248
    assert snap["realism"]["generated_at"] == "2026-11-14 03:19:20"


def test_build_snapshot_omits_missing_sections_instead_of_guessing(brief_env):
    """产物文件不在 → 那一段整个不出现（而不是塞一份默认值）。"""
    snap = DB.build_snapshot()
    assert "oos" not in snap
    assert "realism" not in snap
    assert "ledger" not in snap
    assert "ts" in snap


def test_build_snapshot_survives_broken_json(brief_env):
    """产物文件坏了 → 跳过，不能让整份日报挂掉。"""
    (brief_env / "oos_latest.json").write_text("{坏掉的 JSON", encoding="utf-8")
    snap = DB.build_snapshot()
    assert "oos" not in snap


def test_build_snapshot_reads_ledger_summary(brief_env):
    """账本摘要里的累计超额与哈希链状态要被抓出来。"""
    _write(brief_env / "ledger_summary.json", {
        "n_periods": 3,
        "new_periods_written": 1,
        "first_signal_date": "2026-11-10",
        "chain": {"ok": True},
        "real": {"cumulative": 0.0123, "annualized": 0.103,
                 "positive_ratio": 0.667},
    })
    snap = DB.build_snapshot()
    assert snap["ledger"]["n_periods"] == 3
    assert snap["ledger"]["cumulative_excess"] == pytest.approx(0.0123)
    assert snap["ledger"]["chain_ok"] is True


# ── 5. 历史快照：只追加、封顶 400 ────────────────────────────────────────


def test_save_history_appends(brief_env):
    """每跑一次追加一条，历史快照按时间累积。"""
    DB.save_history({"ts": "2026-11-12 09:00:00"})
    DB.save_history({"ts": "2026-11-13 09:00:00"})
    entries = DB.load_history()
    assert len(entries) == 2
    assert entries[0]["ts"] == "2026-11-12 09:00:00"
    assert entries[-1]["ts"] == "2026-11-13 09:00:00"


def test_save_history_keeps_only_last_400(brief_env):
    """历史文件不能无限膨胀 —— 只留最近 400 条。"""
    for i in range(405):
        DB.save_history({"ts": f"d{i}"})
    entries = DB.load_history()
    assert len(entries) == 400
    assert entries[0]["ts"] == "d5", "留的不是最新的 400 条"


def test_history_file_is_valid_json_after_save(brief_env):
    """保存走的是"写临时文件再改名"，落盘的必须是合法 JSON。"""
    DB.save_history({"ts": "2026-11-14 09:00:00"})
    raw = json.loads((brief_env / "state" / "brief_history.json")
                     .read_text(encoding="utf-8"))
    assert isinstance(raw["entries"], list)
    assert len(raw["entries"]) == 1


def test_load_history_on_missing_file_returns_empty(brief_env):
    """历史文件还没建 → 返回空列表，不是报错。"""
    assert DB.load_history() == []


# ── 6. 忠实性闸门在日报里的呈现 ───────────────────────────────────────────


def test_brief_warns_when_fidelity_gate_failed(brief_env):
    """忠实性闸门没过 → 汇报里必须明说"以下数字不可信"。"""
    snap = _full_snapshot()
    snap["realism"]["fidelity_ok"] = False
    text = DB.build_brief(snap, prev={})
    assert "忠实性闸门未通过" in text
    assert "不可信" in text


def test_brief_does_not_warn_when_fidelity_passed(brief_env):
    """闸门通过 → 不加这句警告。"""
    text = DB.build_brief(_full_snapshot(), prev={})
    assert "忠实性闸门未通过" not in text


def test_brief_warns_when_ledger_chain_is_broken(brief_env):
    """账本哈希链断裂 → 风险清单里必须点명。"""
    snap = _full_snapshot()
    snap["ledger"]["chain_ok"] = False
    text = DB.build_brief(snap, prev={})
    assert "哈希链断裂" in text


# ── 7. 已修复缺陷的回归锁定 ───────────────────────────────────────────────
# 2026-11-14 已修（原为 xfail 记录）：
#   ① 第五段把 '-2.03%/年'、'-3.73%/年'、'1298 只股票' 写死在文案里
#   ② 忠实性闸门风险文案自相矛盾
#   ③ 三态闸门（False/None/True）被压成两态
# 现在改为正式断言 —— 缺陷再犯会被立刻抓住。


def test_section_five_numbers_match_the_snapshot(brief_env):
    """第五段引用的数字必须来自 snapshot，不能是写死的旧值。"""
    snap = _full_snapshot()
    text = DB.build_brief(snap, prev={})
    sec5 = text.split("五、当前能不能用")[1].split("六、主要风险是什么")[0]
    assert "-1.30%" in sec5, "第五段没有用 snapshot 里的回测口径超额"
    assert "-2.69%" in sec5, "第五段没有用 snapshot 里的现实成交口径超额"


def test_risk_line_about_fidelity_gate_is_not_self_contradictory(brief_env):
    """闸门状态的风险文案不能自相矛盾。"""
    snap = _full_snapshot()
    snap["realism"]["fidelity_ok"] = False
    text = DB.build_brief(snap, prev={})
    risks = text.split("六、主要风险是什么")[1].split("七、下一步验证什么")[0]
    assert "当前已通过" not in risks, "文案自相矛盾：说没过又说已通过"


def test_missing_fidelity_is_not_reported_as_gate_failed(brief_env):
    """检验**没跑**（fidelity_ok 缺失）≠ 检验**没过**。

    这会把「没数据」误报成「结论不可信」—— 两种情况的风险含义完全不同。
    """
    snap = _full_snapshot()
    del snap["realism"]["fidelity_ok"]
    text = DB.build_brief(snap, prev={})
    risks = text.split("六、主要风险是什么")[1].split("七、下一步验证什么")[0]
    assert "忠实性闸门没过" not in risks, (
        "检验还没跑就被说成「闸门没过」—— 这会把「没数据」误报成「结论不可信」"
    )


def test_section_five_has_no_hardcoded_stale_numbers(brief_env):
    """第五段不得出现写死的旧数字（写死 = 面板一变就失真）。"""
    snap = _full_snapshot()
    text = DB.build_brief(snap, prev={})
    sec5 = text.split("五、当前能不能用")[1].split("六、主要风险是什么")[0]
    for stale in ("-2.03%", "-3.73%", "1298 只"):
        assert stale not in sec5, (
            f"第五段仍写着写死的旧值 {stale} —— 违反 D2「不编数字」铁律"
        )


def test_brief_uses_no_emoji(brief_env):
    """宋老师偏好纯文字：日报正文不得出现 emoji。"""
    snap = _full_snapshot()
    text = DB.build_brief(snap, prev={})
    bad = [c for c in text if c in "⚠✅❌🚨📊🏆⚡️🔥"]
    assert not bad, f"日报正文出现了 emoji：{bad}"


def test_refresh_status_is_not_reported_twice_as_two_tasks(brief_env):
    """研究队列的 refresh 与每日闭环的 refresh 是同一个任务。

    并列显示两条 failed 会让人误以为是两件不同的事各挂了一次。
    """
    snap = _full_snapshot()
    snap.setdefault("queue", {})["refresh"] = {"status": "failed"}
    snap.setdefault("daily_loop", {})["refresh"] = {"status": "failed"}
    text = DB.build_brief(snap, prev={})
    sec1 = text.split("一、今天做了什么")[1].split("二、分析结果是什么")[0]
    n_mentions = sec1.count("「refresh」")
    assert n_mentions <= 1, (
        f"第一节把同一个 refresh 任务列了 {n_mentions} 次，"
        "读者会误以为是两件事各挂了一次"
    )
