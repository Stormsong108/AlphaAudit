# -*- coding: utf-8 -*-
"""21:30 单份 HTML 铁律的回归测试。

背景：2026-11-14 之前，21:30 发一份 HTML 日报，22:30 又单独发一条七段式
大白话文本 —— 违反「每天北京时间 21:30 仅通过飞书发送一份 HTML，日报、
分析表、模拟组合建议和复盘均嵌入同一文件」的既定约束。

现在：七段式内容由 main.py 调 daily_brief.build_brief() 嵌入同一份 HTML，
brief cron 只复算数据、不再投递。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import render.report as R  # noqa: E402


def _html(**kw) -> str:
    base = dict(quotes={}, events={}, futures={})
    base.update(kw)
    return R.render_html(**base)


def test_brief_text_embedded_in_html():
    """七段式文本必须出现在 HTML 正文里。"""
    h = _html(brief_text="【每日研究汇报 2026-01-01】\n一句话结论\n当前状态：**还不能用**")
    assert "每日研究汇报（大白话" in h
    assert "当前状态" in h
    assert "还不能用" in h


def test_brief_error_shown_not_blank():
    """取不到时必须显示原因，绝不留白（空白会被误读成系统挂了）。"""
    h = _html(brief_error="每日研究汇报：这轮取不到（文件缺失）")
    assert "每日研究汇报（大白话" in h
    assert "这轮取不到" in h
    assert "文件缺失" in h


def test_no_brief_no_block():
    """既没有文本也没有错误时，不渲染空区块（避免出现无意义的空标题）。"""
    h = _html()
    assert "每日研究汇报（大白话" not in h


def test_brief_is_html_escaped():
    """白话正文里的尖括号必须转义，不能破坏 HTML 结构。"""
    h = _html(brief_text="<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in h
    assert "&lt;script&gt;" in h


def test_brief_script_does_not_deliver():
    """brief 的 cron 脚本不得再单独投递飞书 —— 违反「只发一份 HTML」。"""
    script = Path("/Users/sst/.hermes/profiles/xiaojin/scripts/market_daily_brief.sh")
    if not script.exists():          # 换机器时跳过，不让测试变脆
        return
    body = script.read_text(encoding="utf-8")
    assert "--deliver" not in body, "brief 脚本里仍有 --deliver，会多发一条飞书消息"
    assert "--save" in body


def test_main_passes_brief_into_render():
    """main.py 必须把 brief 传给 render_html，否则嵌入链路是断的。"""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "daily_brief.build_brief()" in src
    assert "brief_text=brief_text" in src
    assert "brief_error=brief_error" in src
