"""pytest 全局夹具。

**为什么需要这个文件**：本仓库是**公开**的，但仓库里**不含运行数据**
（行情面板 `data/history/kline/`、研究产物 `data/research/*.json`、预测落盘文件
等）—— 这些体积大、可由脚本重新生成，不适合进版本库。

后果是：克隆仓库后，部分依赖真实运行数据的测试会因为
``FileNotFoundError`` 而**失败**，而不是跳过。这会让公开仓库的 CI 一片红，
掩盖真正的问题。

本文件提供两样东西：

1. ``runtime_data_available`` —— 判定运行时数据是否就位的标记；
2. ``pytest_collection_modifyitems`` —— 自动给「需要运行时数据」的测试打 skip。

**原则**：只在**确定缺的是运行数据**时跳过；代码缺陷导致的失败必须照常暴露，
绝不允许用 skip 把真 bug 藏起来。所以这里的判定只认「路径不存在」
这一个信号，不吞任何异常。
"""

from __future__ import annotations

import os

import pytest

# 需要运行时数据才能跑的测试文件 → 各自依赖的路径（任一存在即认为数据就位）
_RUNTIME_DATA_FILES: dict[str, tuple[str, ...]] = {
    # 依赖真实行情面板（data/history/kline）
    "test_execution_realism.py": ("data/history/kline",),
    # 依赖 research 产物
    "test_market_timing.py": ("data/research/market_timing.json",),
    "test_plain_report.py": ("data/research/market_timing.json",),
    # 依赖预测落盘文件（注意：是 data/predictions/，不是 data/research/predictions/）
    "test_prediction_integrity.py": ("data/predictions",),
}


def _repo_root() -> str:
    # tests/ 的上一级即仓库根
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _has_runtime_data(paths: tuple[str, ...]) -> bool:
    root = _repo_root()
    for rel in paths:
        if not os.path.exists(os.path.join(root, rel)):
            return False
    return True


def pytest_collection_modifyitems(config, items):
    """缺运行时数据的测试文件 → 标记 skip。

    只按**文件名**匹配（而不是按测试函数名），粒度粗但可预测：
    一个测试文件里的用例通常共享同一批运行数据。
    """
    root = _repo_root()
    for item in items:
        fname = os.path.basename(str(item.fspath))
        paths = _RUNTIME_DATA_FILES.get(fname)
        if not paths:
            continue
        if _has_runtime_data(paths):
            continue
        missing = [p for p in paths
                   if not os.path.exists(os.path.join(root, p))]
        item.add_marker(
            pytest.mark.skip(
                reason=f"缺少运行时数据（{'、'.join(missing)}）—— "
                       f"公开仓库不含运行数据，请先运行数据刷新流程"
            )
        )
