"""投递路由测试：确保金融内容只进金融组，绝不进胖叔说留学群。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import main  # noqa: E402

LIUXUE_CHAT = "oc_3b0f8a7e88d0f2ff10ef01702e24c908"
FINANCE_CHAT = "oc_cc9756440fd41494acc866dce3b4e885"


def test_config_targets_finance_group():
    assert config.FEISHU["group_chat_id"] == FINANCE_CHAT


def test_liuxue_chat_is_forbidden():
    assert LIUXUE_CHAT in config.FEISHU["forbidden_chat_ids"]


def test_liuxue_chat_not_a_delivery_target():
    """留学群不能出现在 deliver_to 里，也不能是 group_chat_id。"""
    assert LIUXUE_CHAT not in config.FEISHU["deliver_to"]
    assert config.FEISHU["group_chat_id"] != LIUXUE_CHAT


def test_deliver_refuses_forbidden_target(monkeypatch):
    """就算有人把 group_chat_id 改成留学群，deliver 也必须抛错拒绝投递。"""
    monkeypatch.setitem(config.FEISHU, "group_chat_id", LIUXUE_CHAT)
    monkeypatch.setitem(config.FEISHU, "deliver_to", ["group"])

    with pytest.raises(RuntimeError, match="禁止名单"):
        main.deliver("测试文本")


def test_deliver_does_not_hit_liuxue_chat(monkeypatch):
    """正常路径下，被投递的 chat_id 里不能出现留学群。"""
    sent: list[str] = []

    def fake_send(text, receive_id, id_type="chat_id"):
        sent.append(receive_id)
        return {"ok": True, "message_id": "mock", "chat_id": receive_id}

    monkeypatch.setattr(main, "_try_send", fake_send)
    monkeypatch.setitem(config.FEISHU, "deliver_to", ["group"])

    main.deliver("测试文本")

    assert sent == [FINANCE_CHAT]
    assert LIUXUE_CHAT not in sent


def test_dm_channel_preserved(monkeypatch):
    """DM 是既定的第二投递通道，不能因为修群路由被删掉。"""
    sent: list[str] = []

    def fake_send(text, receive_id, id_type="chat_id"):
        sent.append(receive_id)
        return {"ok": True, "message_id": "mock", "chat_id": receive_id}

    monkeypatch.setattr(main, "_try_send", fake_send)
    monkeypatch.setitem(config.FEISHU, "deliver_to", ["group", "dm"])

    main.deliver("测试文本")

    assert FINANCE_CHAT in sent
    assert config.FEISHU["open_id"] in sent


def test_debounce_window_not_whole_day(monkeypatch, tmp_path):
    """白天手动投递过，不应让当晚 21:30 的定时任务被静默跳过。"""
    monkeypatch.setattr(main, "STATE_DIR", tmp_path)
    monkeypatch.setattr(main, "DEBOUNCE_MIN", 120)

    import time as _time

    day = "2026-11-09"
    main._mark_delivered(day)

    # 刚投递完 → 120 分钟内应跳过（防刷屏）
    assert main._already_delivered(day) is True

    # 模拟 14 小时后（21:30 定时跑）→ 不应跳过
    real_time = _time.time
    try:
        _time.time = lambda: real_time() + 14 * 3600
        assert main._already_delivered(day) is False
    finally:
        _time.time = real_time


def test_old_marker_format_parsed(monkeypatch, tmp_path):
    """旧版 marker 只写了日期，也要能解析，不能让防抖失效或误判。"""
    monkeypatch.setattr(main, "STATE_DIR", tmp_path)
    marker = tmp_path / "delivered_2026-11-09.marker"
    marker.write_text("2026-11-09", encoding="utf-8")

    ts = main._last_delivered_ts("2026-11-09")
    assert ts > 0
