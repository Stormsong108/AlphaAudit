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


# ── 生产产物保护护栏 ──────────────────────────────────────────────────
# 纪律：生产产物只能由**正式生成流程**更新；pytest 及其 fixture 不得写入。
# 曾实测踩坑：tests/test_daily_loop_wiring.py 的一个用例没挡住 judge 阶段，
# judge_all() → save_loop_cache() 悄悄改了 data/research/loop_stats.json
# 的 generated_at（数值没变，但仍是违规写入）。
#
# 这里做**全局兜底**：每个用例前后比对受保护产物的 (mtime, size, sha1)，
# 一旦被测试改动 → 该用例直接失败，并指出是哪个文件。
# 这比逐个用例去打补丁可靠 —— 新增用例若再犯同样的错，会立刻被抓住。

_PROTECTED_GLOBS = (
    "data/research/oos_latest.json",
    "data/research/execution_realism.json",
    "data/research/signal_anatomy.json",
    "data/research/loop_stats.json",
    "data/research/etf_overfit_audit.json",
    "data/research/cost_feasibility.json",
    "data/research/factor_redundancy.json",
    "data/research/factor_redundancy_frozen_spec.json",
    "data/forward/ledger.jsonl",
    "data/forward/ledger_state.json",
    "data/forward/ledger_summary.json",
    "state/brief_history.json",
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _protected_fingerprints() -> dict:
    """受保护产物的指纹快照。缺失文件不记（避免测试新建文件被判违规）。"""
    import hashlib

    out = {}
    for rel in _PROTECTED_GLOBS:
        fp = os.path.join(ROOT, rel)
        if not os.path.exists(fp):
            continue
        try:
            stt = os.stat(fp)
            with open(fp, "rb") as fh:
                h = hashlib.sha1(fh.read()).hexdigest()
        except OSError:
            continue
        out[rel] = (stt.st_mtime_ns, stt.st_size, h)
    return out


@pytest.fixture(autouse=True)
def _guard_protected_artifacts():
    """夹具：受保护生产产物在用例执行期间不得被改动。"""
    before = _protected_fingerprints()
    yield
    after = _protected_fingerprints()

    changed = []
    for rel in sorted(set(before) | set(after)):
        b = before.get(rel)
        a = after.get(rel)
        if b != a:
            changed.append(rel)
    assert not changed, (
        "测试写动了生产产物（违反「测试只读生产产物」纪律）："
        + "、".join(changed)
        + "。生产产物只能由正式生成流程更新；测试请把输出路径 monkeypatch 到 tmp_path。"
    )


def _test_needs_scipy(item) -> bool:
    """判断该用例是否依赖 scipy 的正态分布。

    判据：用例**直接调用** overfit_audit 里用到 ``norm`` 的那几个函数
    （PSR / E[max SR] / MinTRL / 多重检验）。DSR 内部会调 E[max SR] 与
    PSR，因此也一并算作依赖。

    只看「是否调用了这些函数」，不做更深的静态分析 —— 宁可漏判（用例照常
    跑、报 ScipyRequiredError）也不误判（把不依赖 scipy 的用例白白跳过）。
    """
    src = ""
    try:
        src = item.function.__doc__ or ""
        code = item.function.__code__
        names = code.co_names
    except AttributeError:  # pragma: no cover
        return False
    # co_names 里有 M.xxx / OA.xxx 的属性名 → 粗判：是否出现 norm 依赖符号
    norm_dependent = {
        "probabilistic_sharpe_ratio",
        "expected_max_sharpe",
        "minimum_track_record_length",
        "multiple_testing_threshold",
        "deflated_sharpe_ratio",
    }
    return bool(norm_dependent & set(names))


def _scipy_available() -> bool:
    """overfit_audit 是否能拿到 scipy 的正态分布对象。

    ``overfit_audit`` 的 PSR / DSR / E[max SR] / MinTRL / 多重检验门槛
    依赖 scipy 的 ``norm.cdf`` / ``norm.ppf``。这些数值与公开参考实现做过
    **逐位**交叉验证（|d| = 0），所以**不做**近似降级 —— 缺 scipy 时
    显式抛 ``ScipyRequiredError``，而不是悄悄换个算法算出略有偏差的数字。

    后果是：没装 scipy 的环境里，依赖 norm 的用例会抛错而不是失败，
    对公开仓库的克隆者就是「一片红」。本函数把这类用例标记为 skip。
    判定只认「scipy 确实不可用」这一个信号，不吞任何其它异常。
    """
    try:
        import overfit_audit as OA  # noqa: PLC0415
    except Exception:  # pragma: no cover - overfit_audit 自身坏了应当照常暴露
        return True
    return bool(getattr(OA, "SCIPY_AVAILABLE", True))


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

    # 缺 scipy → 依赖正态分位数的用例跳过（其余用例照常跑）
    if not _scipy_available():
        for item in items:
            if _test_needs_scipy(item):
                item.add_marker(
                    pytest.mark.skip(
                        reason="缺少 scipy（标准正态分位数/累积分布）—— "
                               "pip install scipy 后即可运行这些数值交叉验证用例"
                    )
                )
