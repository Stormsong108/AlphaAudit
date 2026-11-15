"""每日闭环接线测试：research_queue 步骤 + main.py 触发 + history 新鲜度。

三条不变量，各对应一个曾经的坑：

1. research_queue 的 refresh/predict/judge 三步必须**幂等**且**失败留痕**：
   失败要记进 state/research_queue.json，并且不能挡住后面的步骤。
2. main.py 在 build_evidence 之前触发 predict → judge；判定失败时，
   日报必须显示"失败"，不能拿旧数据冒充今天已完成。
3. history 的新鲜度基准是"最近一个已收盘的交易日"，不是写死的年份；
   缓存末根落后于该基准时必须走刷新路径。

这些测试全部离线：上游接口一律替换掉，绝不真连。
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import daily_loop  # noqa: E402
import history  # noqa: E402
import research_queue  # noqa: E402


# ------------------------------------------------------------------ 工具


def _write_state(st: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def _step(key: str) -> dict:
    return next(s for s in research_queue.STEPS if s["key"] == key)


# ------------------------------------------------------------------ 1. 队列


def test_queue_has_daily_loop_steps():
    """refresh / predict / judge 必须作为独立步骤接进队列。"""
    keys = [s["key"] for s in research_queue.STEPS]
    for k in ("refresh", "predict", "judge"):
        assert k in keys, f"队列缺少 daily_loop 步骤：{k}"
    # 每个步骤都得有题目、最小间隔、命令
    for s in research_queue.STEPS:
        assert s.get("title"), f"{s['key']} 缺标题"
        assert isinstance(s.get("min_hours"), (int, float))
        assert isinstance(s.get("cmd"), list) and s["cmd"]


def test_daily_loop_steps_are_ordered_and_dependent():
    """刷新 → 预测 → 判定 的依赖必须写明，否则刷新失败还会接着出预测。"""
    assert _step("refresh").get("requires") is None
    assert _step("predict").get("requires") == "refresh"
    assert _step("judge").get("requires") == "predict"


def test_step_failure_is_recorded_and_does_not_block_others(tmp_path, monkeypatch):
    """一步失败必须留痕（状态文件 + 冷却期），且不阻塞后面的步骤。"""
    state_path = tmp_path / "queue_state.json"
    logdir = tmp_path / "logs"
    monkeypatch.setattr(research_queue, "STATE", state_path)
    monkeypatch.setattr(research_queue, "LOGDIR", logdir)

    st = research_queue.load_state()
    fail_step = {
        "key": "boom",
        "title": "故意失败",
        "min_hours": 6.0,
        "cmd": [sys.executable, "-c", "raise SystemExit(3)"],
    }
    rec = research_queue.run_step(fail_step, st)

    assert rec["last_status"] == "failed"
    assert rec["last_exit_code"] == 3
    assert "exit 3" in rec["last_error"]
    # 失败要留冷却期，别把机器跑满
    assert rec.get("retry_after_at")
    # 状态必须落盘
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["steps"]["boom"]["last_status"] == "failed"
    # 日志必须留痕
    assert Path(rec["last_log"]).exists()


def test_failed_step_is_not_due_immediately(tmp_path, monkeypatch):
    """失败后的冷却期内不再重跑（上游不可用时不能硬刷）。"""
    state_path = tmp_path / "queue_state.json"
    monkeypatch.setattr(research_queue, "STATE", state_path)
    st = research_queue.load_state()
    # run_step 失败时写的是 now + 30 分钟
    retry_at = (datetime.now(research_queue.CST) + timedelta(minutes=30)).isoformat(
        timespec="seconds")
    st["steps"]["boom"] = {
        "last_status": "failed",
        "last_error": "上游不可用",
        "last_ok_at": None,
        "retry_after_at": retry_at,
    }
    ok, why = research_queue.due({"key": "boom", "min_hours": 6.0}, st)
    assert not ok
    assert "冷却" in why

    # 冷却期过了 → 重新到期（不能永久卡住）
    st["steps"]["boom"]["retry_after_at"] = (
        datetime.now(research_queue.CST) - timedelta(minutes=1)).isoformat(timespec="seconds")
    ok2, _ = research_queue.due({"key": "boom", "min_hours": 6.0}, st)
    assert ok2


def test_step_blocked_by_failed_prerequisite(tmp_path, monkeypatch):
    """前置步骤失败时，本步不跑 —— 不能拿旧行情冒充今天的闭环。"""
    state_path = tmp_path / "queue_state.json"
    monkeypatch.setattr(research_queue, "STATE", state_path)
    st = research_queue.load_state()
    st["steps"]["refresh"] = {"last_status": "failed", "last_error": "东财不可用"}

    ok, why = research_queue.due(_step("predict"), st)
    assert not ok
    assert "refresh" in why


def test_step_succeeds_and_becomes_not_due(tmp_path, monkeypatch):
    """成功一次后进入冷却（min_hours 内不再跑）—— 幂等，不空转。"""
    state_path = tmp_path / "queue_state.json"
    monkeypatch.setattr(research_queue, "STATE", state_path)
    # 日志也必须落在临时目录：测试不能往真实 logs/ 里写垃圾
    monkeypatch.setattr(research_queue, "LOGDIR", tmp_path / "logs")
    st = research_queue.load_state()
    rec = research_queue.run_step(
        {
            "key": "ok",
            "title": "成功",
            "min_hours": 6.0,
            "cmd": [sys.executable, "-c", "print('fine')"],
        },
        st,
    )
    assert rec["last_status"] == "ok"
    assert rec.get("last_ok_at")
    assert not rec.get("retry_after_at")
    ok, _ = research_queue.due({"key": "ok", "min_hours": 6.0}, st)
    assert not ok


def test_daily_loop_run_stage_is_idempotent_and_traces(tmp_path, monkeypatch):
    """daily_loop.run_stage：重复跑不覆盖已落盘的预测，且每次都留痕。"""
    state_path = tmp_path / "daily_loop_state.json"
    outdir = tmp_path / "out"
    monkeypatch.setattr(daily_loop, "STATE", state_path)
    monkeypatch.setattr(daily_loop, "OUTDIR", outdir)

    calls: list[int] = []

    def fake_predict(*, force: bool = False):
        calls.append(1)
        return {
            "stage": "predict",
            "ts": daily_loop.now_iso(),
            "ok": True,
            "skipped": bool(calls and len(calls) > 1),
            "day": "2026-11-12",
            "reason": "已有预测记录" if len(calls) > 1 else "已生成",
            "n_predictions": 20,
        }

    monkeypatch.setitem(daily_loop.STAGES, "predict", fake_predict)

    p1 = daily_loop.run_stage("predict")
    p2 = daily_loop.run_stage("predict")
    assert p1["ok"] and p2["ok"]
    # 两次都写了独立的结果文件
    assert len(list(outdir.glob("predict_*.json"))) == 2
    # 状态文件里记着最近一次
    st = json.loads(state_path.read_text(encoding="utf-8"))
    assert st["stages"]["predict"]["last_status"] == "ok"


def test_daily_loop_run_stage_failure_is_captured_not_raised(tmp_path, monkeypatch):
    """阶段抛异常时必须转成 ok=False，不能把调用方掀翻。"""
    state_path = tmp_path / "daily_loop_state.json"
    outdir = tmp_path / "out"
    monkeypatch.setattr(daily_loop, "STATE", state_path)
    monkeypatch.setattr(daily_loop, "OUTDIR", outdir)

    def boom(*, force: bool = False):
        raise RuntimeError("东财不可用")

    monkeypatch.setitem(daily_loop.STAGES, "predict", boom)
    payload = daily_loop.run_stage("predict")
    assert payload["ok"] is False
    assert "东财不可用" in payload["reason"]
    # 失败也要留痕
    st = json.loads(state_path.read_text(encoding="utf-8"))
    assert st["stages"]["predict"]["last_status"] == "failed"


def test_daily_loop_status_reflects_stage_failures(tmp_path, monkeypatch):
    """daily_loop_status：阶段失败 → ok=False；只是没轮上 → 不算失败。"""
    state_path = tmp_path / "daily_loop_state.json"
    monkeypatch.setattr(daily_loop, "STATE", state_path)
    st = {
        "stages": {
            "refresh": {"last_status": "ok"},
            "predict": {"last_status": "failed", "last_error": "训练样本不足"},
            "judge": {"last_status": "never_run"},
        }
    }
    _write_state(st, state_path)
    status = daily_loop.daily_loop_status()
    assert status["ok"] is False
    assert status["failed"] == ["predict"]
    assert status["stages"]["predict"]["error"] == "训练样本不足"
    assert status["stages"]["judge"]["status"] == "never_run"
    # 没跑过 ≠ 失败
    assert "judge" in status["missing"]

    # 全部成功 → ok=True
    st["stages"]["predict"] = {"last_status": "ok"}
    st["stages"]["judge"] = {"last_status": "ok"}
    _write_state(st, state_path)
    status = daily_loop.daily_loop_status()
    assert status["ok"] is True
    assert status["failed"] == []


# ------------------------------------------------------------------ 2. main.py


def test_plain_report_shows_daily_loop_failure(monkeypatch):
    """闭环失败必须在小白版里说清楚 —— 不能让读者误以为命中率是今天的。"""
    import plain_report as PR

    ev = PR.load_evidence()
    ev = dict(ev)
    ev["daily_loop"] = {
        "ok": False,
        "stages": {
            "refresh": {"status": "ok"},
            "predict": {"status": "failed", "error": "上游不可用"},
            "judge": {"status": "never_run"},
        },
    }
    plain = PR.build_plain_report(ev)
    assert "上游不可用" in plain
    assert "没" in plain and "跑完" in plain


def test_report_path_does_not_judge_inline():
    """报告侧只能读已落盘的判定，不能现算判定。

    这是"日报只消费已落盘判定结果"这条约束的回归护栏：
    research_evidence 里一旦有人图省事直接调 judge_all，
    报告里的命中率就会变成每次现算、跟昨天不是一个数。
    """
    import inspect

    import research_evidence

    src = inspect.getsource(research_evidence)
    # 去掉注释与文档串后再查：文档里提到 judge_all 是在解释它写了什么
    code = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    )
    assert "judge_all(" not in code, "research_evidence 不该自己调 judge_all（判定只由 daily_loop 落盘）"


def test_main_runs_daily_loop_before_evidence(monkeypatch):
    """main 必须在 build_evidence 之前跑 predict → judge。"""
    import main as main_mod

    order: list[str] = []

    def fake_predict(*, force: bool = False):
        order.append("predict")
        return {"stage": "predict", "ts": "t", "ok": True, "reason": "已生成",
                "day": "2026-11-12", "skipped": False, "n_predictions": 20}

    def fake_judge():
        order.append("judge")
        return {"stage": "judge", "ts": "t", "ok": True, "reason": "已判定"}

    monkeypatch.setitem(daily_loop.STAGES, "predict", fake_predict)
    monkeypatch.setitem(daily_loop.STAGES, "judge", fake_judge)

    import research_evidence

    real_build = research_evidence.build_evidence

    def spy_build(*a, **kw):
        order.append("build_evidence")
        return real_build(*a, **kw)

    monkeypatch.setattr(research_evidence, "build_evidence", spy_build)
    monkeypatch.setattr(main_mod, "run_daily_loop", main_mod.run_daily_loop)

    status = main_mod.run_daily_loop()
    assert order == ["predict", "judge"], f"闭环阶段顺序不对：{order}"
    assert "stages" in status


def test_main_daily_loop_failure_is_reported(monkeypatch):
    """判定失败时，日报拿到的是 ok=False + 原因，不是"今天已完成"。"""
    import main as main_mod

    def boom(*, force: bool = False):
        raise RuntimeError("上游不可用")

    monkeypatch.setitem(daily_loop.STAGES, "predict", boom)
    status = main_mod.run_daily_loop()
    assert status["ok"] is False
    # 失败信息必须能被人看到
    blob = json.dumps(status, ensure_ascii=False)
    assert "上游不可用" in blob


def test_evidence_bundle_carries_daily_loop_status(monkeypatch, tmp_path):
    """证据包必须带上闭环状态，渲染层才有东西可显示。"""
    import research_evidence

    st = {
        "stages": {
            "refresh": {"last_status": "ok", "last_detail": {"updated": 12}},
            "predict": {"last_status": "failed", "last_error": "上游不可用"},
            "judge": {"last_status": "never_run"},
        }
    }
    tmp = tmp_path / "dl.json"
    _write_state(st, tmp)
    monkeypatch.setattr(daily_loop, "STATE", tmp)

    bundle = research_evidence.build_evidence()
    assert bundle["daily_loop"]["ok"] is False
    assert bundle["daily_loop"]["stages"]["predict"]["error"] == "上游不可用"


def test_analyze_daily_loop_says_failure(monkeypatch, tmp_path):
    """分析文字必须明说哪一段失败、失败意味着什么。"""
    import research_evidence

    st = {
        "stages": {
            "refresh": {"last_status": "ok", "last_detail": {"updated": 5, "up_to_date": 1300, "failed": 0}},
            "predict": {"last_status": "failed", "last_error": "上游不可用"},
            "judge": {"last_status": "never_run"},
        }
    }
    tmp = tmp_path / "dl.json"
    _write_state(st, tmp)
    monkeypatch.setattr(daily_loop, "STATE", tmp)

    bundle = research_evidence.build_evidence()
    text = "\n".join(research_evidence.analyze_daily_loop(bundle))
    assert "上游不可用" in text
    assert "失败" in text
    # 失败必须说明"不能当作今天的结论"
    assert "不能" in text


# ------------------------------------------------------------------ 3. history


def test_latest_trade_date_is_not_a_hardcoded_year():
    """新鲜度基准必须是算出来的，不能是写死的年份阈值。"""
    src = Path(history.__file__).read_text(encoding="utf-8")
    # 只看代码行：文档里提到旧阈值是给人看的，不算。
    code = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith("#") and '"""' not in line
    )
    assert "20260101" not in code, "history.py 里还有写死的年份阈值"


def test_latest_trade_date_uses_market_session_when_available(tmp_path, monkeypatch):
    """有行情快照时，基准取接口自己报的 A 股交易日（不硬编码节假日表）。"""
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(parents=True)
    (snap_dir / "2026-11-10.json").write_text(
        json.dumps({
            "market_session": {
                "today": "2026-11-10",
                "markets": {"A股": {"trade_date": "2026-11-10", "is_open": True}},
            }
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(history, "SNAPSHOT_DIR", snap_dir)

    from datetime import date
    assert history.latest_trade_date(today=date(2026, 11, 12)) == date(2026, 11, 10)


def test_latest_trade_date_falls_back_to_weekday(tmp_path, monkeypatch):
    """没有快照时退回工作日推算，且收盘前今天不算数。"""
    from datetime import date, datetime, timedelta, timezone
    monkeypatch.setattr(history, "SNAPSHOT_DIR", tmp_path / "empty")
    cst = timezone(timedelta(hours=8))

    # 周六 → 往前退到周五
    sat = datetime(2026, 11, 14, 16, 0, tzinfo=cst)
    assert history.latest_trade_date(now=sat) == date(2026, 11, 13)
    # 周日 → 周五
    sun = datetime(2026, 11, 15, 16, 0, tzinfo=cst)
    assert history.latest_trade_date(now=sun) == date(2026, 11, 13)
    # 周一收盘前 → 基准是上周五（今天的 bar 还没出）
    mon_am = datetime(2026, 11, 16, 9, 0, tzinfo=cst)
    assert history.latest_trade_date(now=mon_am) == date(2026, 11, 13)


def test_is_fresh_triggers_refresh_when_cache_lags(tmp_path, monkeypatch):
    """缓存末根落后于最近交易日 → 判定为不新鲜 → 走刷新路径。"""
    from datetime import date, datetime, timedelta, timezone
    monkeypatch.setattr(history, "SNAPSHOT_DIR", tmp_path / "empty")
    cst = timezone(timedelta(hours=8))
    now = datetime(2026, 11, 12, 16, 0, tzinfo=cst)

    stale = [{"date": "2026-11-05", "close": 1.0}]
    assert history.is_fresh(stale, now=now) is False
    # 末根正好是最近交易日 → 新鲜
    fresh = [{"date": "2026-11-12", "close": 1.0}]
    assert history.is_fresh(fresh, now=now) is True


def test_merge_append_keeps_history_intact():
    """刷新只能追加 —— 后复权价会随分红回头重算，整段覆盖等于改写历史。"""
    local = [
        {"date": "2026-11-10", "close": 10.0},
        {"date": "2026-11-11", "close": 10.5},
    ]
    fresh = [
        {"date": "2026-11-11", "close": 10.5},
        {"date": "2026-11-12", "close": 10.8},
    ]
    merged, writable = history.merge_append(local, fresh)
    assert writable
    assert [r["date"] for r in merged] == ["2026-11-10", "2026-11-11", "2026-11-12"]
    # 历史段的收盘价一个字节都不能改
    assert merged[0] == local[0]
    assert merged[1] == local[1]


def test_merge_append_refuses_when_anchor_changed():
    """重叠段对不上（上游回头重算了后复权价）→ 拒绝写入，保留本地历史。"""
    local = [{"date": "2026-11-10", "close": 10.0}]
    fresh = [{"date": "2026-11-10", "close": 10.06}]  # 锚点变了
    merged, writable = history.merge_append(local, fresh)
    assert writable is False
    assert merged == local


def test_get_history_refreshes_when_cache_lags(tmp_path, monkeypatch):
    """缓存落后 → 真的联网拉了（刷新路径），而不是直接返回旧缓存。"""
    monkeypatch.setattr(history, "SNAPSHOT_DIR", tmp_path / "empty")
    monkeypatch.setattr(history, "HIST_DIR", tmp_path / "hist")
    cst = timezone(timedelta(hours=8))
    now = datetime(2026, 11, 12, 16, 0, tzinfo=cst)

    fetched: list[str] = []

    def fake_fetch(secid, beg="20150101", end="20500101"):
        fetched.append(secid)
        return [{"date": "2026-11-12", "open": 1, "close": 2, "high": 2, "low": 1,
                 "volume": 1, "amount": 1}]

    monkeypatch.setattr(history, "_fetch_kline", fake_fetch)

    # 先落一份"落后"的缓存
    (tmp_path / "hist").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hist" / "1_600036.json").write_text(
        json.dumps([{"date": "2026-01-05", "close": 1.0}]), encoding="utf-8")

    rows = history.get_history("1.600036", sleep=0, now=now)
    assert fetched == ["1.600036"], "缓存落后时必须走刷新路径"
    assert rows[-1]["date"] == "2026-11-12"


def test_get_history_uses_cache_when_fresh(tmp_path, monkeypatch):
    """缓存够新 → 直接返回，不联网（改了阈值也不能变成每次都重拉）。"""
    monkeypatch.setattr(history, "SNAPSHOT_DIR", tmp_path / "empty")
    monkeypatch.setattr(history, "HIST_DIR", tmp_path / "hist")
    cst = timezone(timedelta(hours=8))
    now = datetime(2026, 11, 12, 16, 0, tzinfo=cst)

    def boom(secid, beg="20150101", end="20500101"):
        raise AssertionError("缓存够新时不该联网")

    monkeypatch.setattr(history, "_fetch_kline", boom)
    (tmp_path / "hist").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hist" / "1_600036.json").write_text(
        json.dumps([{"date": "2026-11-12", "close": 2.0}]), encoding="utf-8")

    rows = history.get_history("1.600036", sleep=0, now=now)
    assert rows[-1]["date"] == "2026-11-12"
