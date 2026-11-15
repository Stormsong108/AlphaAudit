"""开盘价面板（panel_open）的回归测试。

三条必须锁死的口径铁律（来自模块 docstring）：

1. **与主面板逐格对齐** —— 同样的 dates、同样的 codes、同样的顺序。
   对不齐就是拿错了数据，宁可炸掉也不能错位。
2. **只保留正开盘价；缺失一律 NaN，不插值、不前向填充** ——
   缺失本身是信息（停牌/无数据），填平了就把摩擦藏起来了。
3. **缓存带 K 线指纹，指纹变了缓存失效，必须重建**（和主面板同一纪律）。

测试全部用合成的小面板 + monkeypatch 掉 history.load_kline /
ooslab._kline_fingerprint，既不加载真面板、也不碰真缓存文件
（缓存一律写到 pytest 临时目录）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import panel_open as PO  # noqa: E402


# ── 合成 fixture ───────────────────────────────────────────────────────────


@pytest.fixture
def panel():
    """3 只股票 × 4 个交易日的假面板。"""
    return {
        "dates": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
        "codes": ["0.000001", "0.000002", "1.600000"],
        "close": np.array([
            [10.0, 20.0, 30.0],
            [10.5, 20.5, 30.5],
            [11.0, 21.0, 31.0],
            [11.5, 21.5, 31.5],
        ], dtype=np.float32),
    }


@pytest.fixture
def fake_kline(monkeypatch):
    """替换 history.load_kline：按 code 返回假 K 线行。

    返回 (rows_by_code, calls)，calls 记录每次请求的 code，
    用来验证"缓存命中时不再回读 K 线"。
    """
    rows_by_code: dict[str, list[dict]] = {}
    calls: list[str] = []

    def _load_kline(code: str):
        calls.append(code)
        return list(rows_by_code.get(code, []))

    monkeypatch.setattr(PO.history, "load_kline", _load_kline)
    return rows_by_code, calls


@pytest.fixture
def fake_fingerprint(monkeypatch):
    """替换 K 线指纹：返回一个可控的字符串。"""
    state = {"fp": "fp-AAAA"}

    def _fp() -> str:
        return state["fp"]

    monkeypatch.setattr(PO.ooslab, "_kline_fingerprint", _fp)
    return state


@pytest.fixture
def cache(tmp_path):
    """所有 load_open_panel 调用都用这个临时缓存，绝不碰真缓存文件。"""
    return tmp_path / "open_panel.npz"


def _row(date, open_, close):
    return {"date": date, "open": open_, "close": close}


def _full_klines(panel, rows_by_code, open_map=None):
    """给 panel 里每只股票铺满 4 天 K 线（open_map 可覆盖开盘价）。"""
    for code in panel["codes"]:
        rows = []
        for i, d in enumerate(panel["dates"]):
            o = (open_map or {}).get((code, d), float(i + 1))
            rows.append(_row(d, o, float(i + 1)))
        rows_by_code[code] = rows
    return rows_by_code


# ── 1. 逐格对齐 ────────────────────────────────────────────────────────────


def test_open_panel_is_cell_by_cell_aligned_with_main_panel(panel, fake_kline,
                                                            fake_fingerprint,
                                                            cache):
    """open 矩阵的行=dates、列=codes，顺序必须与主面板完全一致。"""
    rows_by_code, _ = fake_kline
    rows_by_code["0.000001"] = [
        _row("2026-01-05", 1.0, 10.0),
        _row("2026-01-06", 2.0, 10.5),
        _row("2026-01-07", 3.0, 11.0),
        _row("2026-01-08", 4.0, 11.5),
    ]
    rows_by_code["0.000002"] = [
        _row("2026-01-05", 5.0, 20.0),
        _row("2026-01-06", 6.0, 20.5),
        _row("2026-01-07", 7.0, 21.0),
        _row("2026-01-08", 8.0, 21.5),
    ]
    rows_by_code["1.600000"] = [
        _row("2026-01-05", 9.0, 30.0),
        _row("2026-01-06", 9.5, 30.5),
        _row("2026-01-07", 9.9, 31.0),
        _row("2026-01-08", 9.95, 31.5),
    ]

    op = PO.load_open_panel(panel=panel, cache=cache, verbose=False)

    assert op["dates"] == panel["dates"]
    assert op["codes"] == panel["codes"]
    assert op["open"].shape == panel["close"].shape

    # 逐格核对：open[i, j] 必须是 code=j 在 date=i 的那一格
    expected = np.array([
        [1.0, 5.0, 9.0],
        [2.0, 6.0, 9.5],
        [3.0, 7.0, 9.9],
        [4.0, 8.0, 9.95],
    ], dtype=np.float32)
    assert np.allclose(op["open"], expected)


def test_open_panel_ignores_kline_dates_outside_the_panel(panel, fake_kline,
                                                          fake_fingerprint,
                                                          cache):
    """K 线里有面板之外的日期 → 直接忽略，不能错位写进去。"""
    rows_by_code, _ = fake_kline
    rows_by_code["0.000001"] = [
        _row("2025-12-31", 99.0, 99.0),   # 面板之前
        _row("2026-01-05", 1.0, 10.0),
        _row("2026-01-06", 2.0, 10.5),
        _row("2026-01-07", 3.0, 11.0),
        _row("2026-01-08", 4.0, 11.5),
        _row("2026-01-09", 88.0, 88.0),   # 面板之后
    ]
    op = PO.load_open_panel(panel=panel, cache=cache, verbose=False)
    assert op["open"][0, 0] == pytest.approx(1.0)
    assert np.all(np.isnan(op["open"][:, 1:]))  # 其他股票没有数据


# ── 2. 缺失一律 NaN：不插值、不前向填充 ───────────────────────────────────


def test_missing_open_stays_nan_and_is_not_forward_filled(panel, fake_kline,
                                                          fake_fingerprint,
                                                          cache):
    """中间某天缺开盘价 → 那一格必须是 NaN，且不能被前后填平。"""
    rows_by_code, _ = fake_kline
    # 0.000001 缺 2026-01-06
    rows_by_code["0.000001"] = [
        _row("2026-01-05", 1.0, 10.0),
        # 2026-01-06 缺失
        _row("2026-01-07", 3.0, 11.0),
        _row("2026-01-08", 4.0, 11.5),
    ]
    op = PO.load_open_panel(panel=panel, cache=cache, verbose=False)

    assert np.isnan(op["open"][1, 0]), "缺失的格子必须是 NaN"
    assert not np.isnan(op["open"][0, 0])
    assert not np.isnan(op["open"][2, 0])
    # 关键：不能被前一行"前向填充"
    assert op["open"][2, 0] == pytest.approx(3.0), "第二天被前向填充了"
    # 也不能被后一行"后向填充"
    assert op["open"][0, 0] == pytest.approx(1.0), "第一天被后向填充了"


def test_none_and_non_positive_open_are_dropped(panel, fake_kline,
                                                 fake_fingerprint, cache):
    """None / 0 / 负数开盘价 → 一律丢弃留 NaN（价格必须是正的）。"""
    rows_by_code, _ = fake_kline
    rows_by_code["0.000001"] = [
        _row("2026-01-05", 1.0, 10.0),
        _row("2026-01-06", None, 10.5),
        _row("2026-01-07", 0.0, 11.0),
        _row("2026-01-08", -2.0, 11.5),
    ]
    op = PO.load_open_panel(panel=panel, cache=cache, verbose=False)

    assert op["open"][0, 0] == pytest.approx(1.0)
    assert np.isnan(op["open"][1, 0])
    assert np.isnan(op["open"][2, 0])
    assert np.isnan(op["open"][3, 0])


def test_code_with_no_kline_history_is_all_nan(panel, fake_kline,
                                               fake_fingerprint, cache):
    """完全没有历史数据的股票 → 整列 NaN，不报错、不填值。"""
    rows_by_code, _ = fake_kline
    rows_by_code["0.000001"] = [_row("2026-01-05", 1.0, 10.0)]
    op = PO.load_open_panel(panel=panel, cache=cache, verbose=False)
    assert op["open"][0, 0] == pytest.approx(1.0)
    assert np.all(np.isnan(op["open"][:, 1:]))


# ── 3. 缓存指纹纪律 ────────────────────────────────────────────────────────


def test_first_call_builds_and_second_call_hits_cache(panel, fake_kline,
                                                      fake_fingerprint, cache):
    """指纹相同 → 第二次直接吃缓存，不再回读 K 线。"""
    rows_by_code, calls = fake_kline
    _full_klines(panel, rows_by_code)

    first = PO.load_open_panel(panel=panel, cache=cache, verbose=False)
    assert first["cached"] is False
    assert cache.exists(), "缓存文件应该被写出来"
    n_calls = len(calls)

    second = PO.load_open_panel(panel=panel, cache=cache, verbose=False)
    assert second["cached"] is True
    assert len(calls) == n_calls, "缓存命中时不该再回读 K 线"
    assert np.allclose(first["open"], second["open"])


def test_cache_is_invalidated_when_kline_fingerprint_changes(panel, fake_kline,
                                                             fake_fingerprint,
                                                             cache):
    """K 线指纹变了 → 缓存必须失效并重建（旧面板被直接读回来是最阴的坑）。"""
    rows_by_code, calls = fake_kline
    _full_klines(panel, rows_by_code)

    PO.load_open_panel(panel=panel, cache=cache, verbose=False)
    assert len(calls) == 3

    # 源数据变了：价格改了 + 指纹也变了
    fake_fingerprint["fp"] = "fp-BBBB"
    _full_klines(panel, rows_by_code, open_map={
        (code, d): 10.0 + i
        for code in panel["codes"]
        for i, d in enumerate(panel["dates"])
    })
    rebuilt = PO.load_open_panel(panel=panel, cache=cache, verbose=False)
    assert rebuilt["cached"] is False, "指纹变了还吃缓存 —— 旧数据被读回来了"
    assert rebuilt["open"][0, 0] == pytest.approx(10.0)
    assert len(calls) == 6  # 重建时重新回读了 K 线


def test_cache_without_fingerprint_field_is_invalid(panel, fake_kline,
                                                     fake_fingerprint, cache):
    """老版本缓存没有指纹字段 → 必须判失效（不能短路放行）。"""
    rows_by_code, calls = fake_kline
    _full_klines(panel, rows_by_code)

    np.savez_compressed(
        cache,
        dates=np.array(panel["dates"], dtype=object).astype("U16"),
        codes=np.array(panel["codes"], dtype=object).astype("U16"),
        open=np.full((4, 3), 7.0, dtype=np.float32),
        # 故意不写 fingerprint
    )
    op = PO.load_open_panel(panel=panel, cache=cache, verbose=False)
    assert op["cached"] is False, "没有指纹字段的缓存被放行了"
    assert op["open"][0, 0] == pytest.approx(1.0), "应该重建而不是读缓存"
    assert len(calls) == 3


def test_cache_with_different_dates_or_codes_is_invalid(panel, fake_kline,
                                                          fake_fingerprint,
                                                          cache):
    """缓存的 dates/codes 与主面板不一致 → 判失效（错位比没有更糟）。"""
    rows_by_code, _ = fake_kline
    _full_klines(panel, rows_by_code)
    PO.load_open_panel(panel=panel, cache=cache, verbose=False)

    # 只改 dates（顺序变），指纹保持不变
    other = dict(panel)
    other["dates"] = ["2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"]
    op = PO.load_open_panel(panel=other, cache=cache, verbose=False)
    assert op["cached"] is False, "dates 不一致还吃缓存 —— 格子错位了"


def test_corrupt_cache_is_rebuilt_silently(panel, fake_kline, fake_fingerprint,
                                            cache):
    """缓存文件坏了 → 重建，而不是抛异常把整条流水线打断。"""
    rows_by_code, _ = fake_kline
    _full_klines(panel, rows_by_code)

    cache.write_bytes(b"not a npz file at all")
    op = PO.load_open_panel(panel=panel, cache=cache, verbose=False)
    assert op["cached"] is False
    assert op["open"][0, 0] == pytest.approx(1.0)


# ── 4. 其它守卫 ────────────────────────────────────────────────────────────


def test_empty_panel_raises(panel, fake_kline, fake_fingerprint, cache):
    """主面板为空 → 直接炸，不能返回一个 0×0 的假面板。"""
    empty = {"dates": [], "codes": [], "close": np.zeros((0, 0), dtype=np.float32)}
    with pytest.raises(RuntimeError):
        PO.load_open_panel(panel=empty, cache=cache, verbose=False)


def test_verbose_mode_reports_coverage(panel, fake_kline, fake_fingerprint,
                                        cache, capsys):
    """verbose 模式要打印覆盖率，方便人工核对数据完整度。"""
    rows_by_code, _ = fake_kline
    rows_by_code["0.000001"] = [_row("2026-01-05", 1.0, 10.0)]
    PO.load_open_panel(panel=panel, cache=cache, verbose=True)
    out = capsys.readouterr().out
    assert "open_panel" in out
    assert "覆盖" in out


def test_real_cache_is_never_touched_by_tests():
    """回归防线：测试里所有 load_open_panel 调用都必须显式给 cache=。

    防止以后有人在测试里漏传 cache 参数、把真面板缓存
    data/panel/open_panel.npz 覆盖掉。
    """
    src = Path(__file__).read_text(encoding="utf-8")
    calls = [ln.strip() for ln in src.splitlines()
             if "load_open_panel(" in ln and "def " not in ln]
    assert calls, "测试文件里应该有 load_open_panel 的调用"
    for ln in calls:
        assert "cache=" in ln, f"调用没显式指定 cache：{ln}"
