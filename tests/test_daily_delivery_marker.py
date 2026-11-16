"""投递标记（delivered marker）的判定逻辑 —— 锁死「dry-run 不写标记」。

背景：
    2026-11-16 发现 `main.py --dry-run` 会误写 `delivered_{day}.marker`。
    原因：`deliver_html(dry_run=True)` 返回 `[{"target":"dry-run","ok":True}]`，
    而调用处直接用 `all(r.get("ok"))` 判定 → dry-run 被当成「投递成功」。
    后果：白天手动 dry-run 一次，当晚 21:30 的定时投递被
    「最近 N 分钟内已投递」**静默跳过** = 少发一期日报（用户看不到）。

这些测试**完全离线**：不联网、不写生产产物、不碰 state/ 下的真实 marker。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import main as M  # noqa: E402


class TestShouldMarkDelivered:
    """_should_mark_delivered 的判定矩阵。"""

    def test_dry_run_never_marks_delivered(self):
        """dry-run 返回的假 ok=True 结果，绝不能触发写标记。"""
        fake_dry = [{"target": "dry-run", "ok": True, "message_id": None}]
        assert M._should_mark_delivered(fake_dry, dry_run=True) is False, (
            "dry-run 写了投递标记 → 真实投递会被静默跳过 = 少发一期日报"
        )

    def test_real_success_marks_delivered(self):
        """真实投递全部成功 → 必须写标记。"""
        ok = [
            {"target": "group:oc_x", "ok": True, "message_id": "om_1"},
            {"target": "dm:ou_y", "ok": True, "message_id": "om_2"},
        ]
        assert M._should_mark_delivered(ok, dry_run=False) is True

    def test_partial_failure_does_not_mark_delivered(self):
        """一个目标失败 → 不能写标记（否则失败的那路永远不补发）。"""
        mixed = [
            {"target": "group:oc_x", "ok": True, "message_id": "om_1"},
            {"target": "dm:ou_y", "ok": False, "message_id": None},
        ]
        assert M._should_mark_delivered(mixed, dry_run=False) is False

    def test_empty_results_do_not_mark_delivered(self):
        """没有任何目标（例如投递目标被全部禁用）→ 不写标记。"""
        assert M._should_mark_delivered([], dry_run=False) is False

    def test_dry_run_false_with_empty_results_is_false(self):
        """空结果 + 非 dry-run → 仍然是 False（防止有人把 `results and` 改成恒真）。"""
        assert M._should_mark_delivered([], dry_run=False) is False


class TestDeliverHtmlDryRunShape:
    """dry-run 返回值形状的回归锁。

    如果哪天有人把 dry_run 分支改成返回 ok=False，本模块的
    「dry-run 不写标记」保护就变成双保险而不是唯一防线。
    """

    def test_deliver_html_dry_run_returns_single_fake_success(self):
        res = M.deliver_html("/nonexistent/never-used.html", dry_run=True)
        assert len(res) == 1
        assert res[0]["target"] == "dry-run"
        assert res[0]["ok"] is True
        assert res[0]["message_id"] is None
        # 关键：这个假 ok=True 正是历史 bug 的成因，必须被 _should_mark_delivered 拦下
        assert M._should_mark_delivered(res, dry_run=True) is False

    def test_deliver_text_dry_run_returns_single_fake_success(self):
        res = M.deliver("hello", dry_run=True)
        assert len(res) == 1
        assert res[0]["ok"] is True
        assert M._should_mark_delivered(res, dry_run=True) is False


class TestMarkerHelpersArePure:
    """_last_delivered_ts 的时间窗口语义（防抖）。

    ⚠️ marker 里存的是**格式化时间字符串** `"%Y-%m-%d %H:%M:%S"`（见
    `_mark_delivered`），不是 epoch 浮点。解析失败一律当作 0（= 没投过）。
    """

    def test_missing_marker_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(M, "STATE_DIR", tmp_path)
        assert M._last_delivered_ts("2026-11-16") == 0.0

    def test_marker_outside_debounce_window_is_stale(self, tmp_path, monkeypatch):
        import time as _time

        monkeypatch.setattr(M, "STATE_DIR", tmp_path)
        old = _time.time() - (M.DEBOUNCE_MIN * 60 + 600)
        stamp = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(old))
        (tmp_path / "delivered_2026-11-16.marker").write_text(stamp, encoding="utf-8")
        # _last_delivered_ts 返回的是**解析出的时间戳**（不是 0）；
        # 「过没过期」由 _already_delivered 判定。
        assert M._last_delivered_ts("2026-11-16") == pytest.approx(old, abs=1.0)
        assert M._already_delivered("2026-11-16") is False, (
            "超出防抖窗口的标记必须视为过期，否则定时投递会被跳过"
        )

    def test_marker_inside_debounce_window_is_fresh(self, tmp_path, monkeypatch):
        import time as _time

        monkeypatch.setattr(M, "STATE_DIR", tmp_path)
        now = _time.time()
        stamp = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(now))
        (tmp_path / "delivered_2026-11-16.marker").write_text(stamp, encoding="utf-8")
        assert M._last_delivered_ts("2026-11-16") == pytest.approx(now, abs=1.0)
        assert M._already_delivered("2026-11-16") is True, (
            "窗口内的标记必须判定为「已投递」，以防同一期内重复发"
        )

    def test_legacy_date_only_marker_is_treated_as_stale(self, tmp_path, monkeypatch):
        """旧格式（纯日期）= 当天零点 → 早已超出窗口 → 视为没投过。

        否则升级过 marker 格式的那天，白天跑过一次就可能把晚上的投递挡掉。
        """
        monkeypatch.setattr(M, "STATE_DIR", tmp_path)
        (tmp_path / "delivered_2026-11-16.marker").write_text("2026-11-16", encoding="utf-8")
        assert M._already_delivered("2026-11-16") is False, (
            "纯日期旧标记不得被判为「已投递」"
        )

    def test_unparseable_marker_is_treated_as_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(M, "STATE_DIR", tmp_path)
        (tmp_path / "delivered_2026-11-16.marker").write_text("garbage", encoding="utf-8")
        assert M._last_delivered_ts("2026-11-16") == 0.0
        assert M._already_delivered("2026-11-16") is False
