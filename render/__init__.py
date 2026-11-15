"""渲染层：把采集结果转成日报文本 / 结构化 HTML。

- render_daily：飞书可读纯文本（缩进 2 空格、关键数字加粗）
- render_html：结构化单文件 HTML（表格 + 卡片 + 模拟组合 + 复盘）
"""
from render.report import render_html  # noqa: F401
from render.text import render_daily  # noqa: F401

__all__ = ["render_daily", "render_html"]
