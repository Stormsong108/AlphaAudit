"""daily_loop refresh 的腾讯回退路径测试（全部离线，绝不联网）。

覆盖的不变量：

1. **回退触发**：东财抛异常 / 返回空序列 → 换腾讯当数据源，不是直接判死。
2. **换算统一**：腾讯的价格必须换算回本地的复权锚点尺度，而且开/收/高/低
   **四个价格全换**（只换 close 会在边界留下假的价格跳变）。
3. **比值带**：重叠段比值带不稳（n < 20 或离散 > 1%）→ 整只判失败。
4. **边界漂移**：新 bar 实际比例与历史比值带的漂移超过 RATIO_TOL → 判失败。
5. **失败不落脏数据**：失败时不得返回 new_rows（调用方会把它 append 进历史）。
6. **来源如实记录**：primary_source 标明实际用的源；cross_source 只有真做了
   交叉校验才标 "passed"，没做标 "skipped"。
7. **成交额**：腾讯不给成交额，必须估算（不能写 0），且估算口径要能被回测验证。

这些测试全部离线：em_fetch / tx_fetch 一律替换成合成数据。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import daily_loop  # noqa: E402


# ------------------------------------------------------------------ 合成数据

SECID = "1.600036"
TX_CODE = "sh600036"
HIST_RATIO = 1.006      # 本地(东财 fqt=2) / 腾讯(hfq) 的常数比
VOL_SCALE = 0.01        # 腾讯给"股"、本地给"手" → 单位比 = 1/100
START = datetime(2026, 1, 5)


def _dates(n: int) -> list[str]:
    return [(START + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]


def _make_local(n: int = 60, base: float = 40.0) -> list[dict]:
    """合成本地历史：close 缓慢上行，volume/amount 自洽（amount = 手数×100×典型价）。"""
    rows = []
    for i, d in enumerate(_dates(n)):
        close = base * (1.0 + 0.002 * i)
        high = close * 1.01
        low = close * 0.99
        typ = (high + low + close) / 3.0
        vol = 100000.0 + 500.0 * i          # 单位：手
        rows.append({
            "date": d, "open": close * 0.998, "close": close,
            "high": high, "low": low, "volume": vol,
            "amount": vol * 100.0 * typ,
        })
    return rows


def _tx_from_local(local: list[dict], n_new: int = 2,
                   ratio: float = HIST_RATIO, ratio_tail: float | None = None) -> list[dict]:
    """把本地历史"翻译"成腾讯口径：close 除以 ratio，volume 除以 VOL_SCALE。

    ratio_tail 不为空时，最后 5 个重叠日的比例改成 ratio_tail —— 用来构造
    "比值带还算稳、但边界比例已经漂了"的场景。
    """
    out = []
    n = len(local)
    for i, r in enumerate(local):
        rr = ratio
        if ratio_tail is not None and i >= n - 5:
            rr = ratio_tail
        typ = (r["high"] + r["low"] + r["close"]) / 3.0
        out.append({
            "date": r["date"],
            "open": r["open"] / rr, "close": r["close"] / rr,
            "high": r["high"] / rr, "low": r["low"] / rr,
            "volume": r["volume"] / VOL_SCALE,
            "amount": 0.0,          # 腾讯不提供成交额（与真实接口一致）
        })
    # 新 bar：本地末根之后
    last = local[-1]
    for k in range(1, n_new + 1):
        close = last["close"] * (1.0 + 0.003 * k)
        high = close * 1.01
        low = close * 0.99
        typ = (high + low + close) / 3.0
        vol = last["volume"] * (1.0 + 0.01 * k)
        out.append({
            "date": (datetime.strptime(last["date"], "%Y-%m-%d")
                     + timedelta(days=k)).strftime("%Y-%m-%d"),
            "open": close * 0.999 / ratio, "close": close / ratio,
            "high": high / ratio, "low": low / ratio,
            "volume": vol / VOL_SCALE,
            "amount": 0.0,
        })
    return out


def _em_from_local(local: list[dict], n_new: int = 2) -> list[dict]:
    """东财口径：重叠段与本地逐日完全一致（OVERLAP_TOL=1e-6 的前提），新 bar 直接给本地尺度。"""
    rows = [{
        "date": r["date"], "open": r["open"], "close": r["close"],
        "high": r["high"], "low": r["low"], "volume": r["volume"],
        "amount": r["amount"],
    } for r in local]
    last = local[-1]
    for k in range(1, n_new + 1):
        close = last["close"] * (1.0 + 0.003 * k)
        high = close * 1.01
        low = close * 0.99
        typ = (high + low + close) / 3.0
        vol = last["volume"] * (1.0 + 0.01 * k)
        rows.append({
            "date": (datetime.strptime(last["date"], "%Y-%m-%d")
                     + timedelta(days=k)).strftime("%Y-%m-%d"),
            "open": close * 0.999, "close": close, "high": high, "low": low,
            "volume": vol, "amount": vol * 100.0 * typ,
        })
    return rows


def _patch(monkeypatch, em_rows=None, em_exc=None, tx_rows=None, tx_exc=None):
    def _em(secid, beg):
        if em_exc is not None:
            raise em_exc
        return list(em_rows or [])

    def _tx(code, count=daily_loop.TX_COUNT):
        if tx_exc is not None:
            raise tx_exc
        return list(tx_rows or [])

    monkeypatch.setattr(daily_loop, "em_fetch", _em)
    monkeypatch.setattr(daily_loop, "tx_fetch", _tx)


# ------------------------------------------------------------------ 1. 东财正常


def test_eastmoney_primary_path_unchanged(monkeypatch):
    """东财可用 → 走东财路径，primary_source=eastmoney，交叉校验通过标 passed。"""
    local = _make_local()
    em = _em_from_local(local)
    tx = _tx_from_local(local)
    _patch(monkeypatch, em_rows=em, tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    assert out["ok"] is True, out["reason"]
    assert out["primary_source"] == "eastmoney"
    assert out["stats"]["cross_source"] == "passed"
    assert out["new_rows"], "东财可用时应当拿到新 bar"
    # 东财路径不做任何价格换算：价格就是东财原值
    em_new = [r for r in em if r["date"] > local[-1]["date"]]
    assert len(out["new_rows"]) == len(em_new)
    for got, src in zip(out["new_rows"], em_new):
        assert got["close"] == pytest.approx(src["close"])
        assert got["open"] == pytest.approx(src["open"])
        assert got["amount"] == pytest.approx(src["amount"])


def test_eastmoney_primary_path_marks_cross_source_skipped_when_tx_down(monkeypatch):
    """东财可用但腾讯不可用 → 交叉校验没做，如实标 skipped（不能标 passed）。"""
    local = _make_local()
    em = _em_from_local(local)
    _patch(monkeypatch, em_rows=em, tx_exc=RuntimeError("腾讯挂了"))

    out = daily_loop.validate_secid(SECID, local)

    assert out["ok"] is True, out["reason"]
    assert out["primary_source"] == "eastmoney"
    assert out["stats"]["cross_source"] == "skipped"


# ------------------------------------------------------------------ 2. 腾讯回退成功


def test_tencent_fallback_used_when_eastmoney_raises(monkeypatch):
    """东财抛异常 → 换腾讯当数据源，四个价格全部换算到本地锚点尺度。"""
    local = _make_local()
    tx = _tx_from_local(local)
    _patch(monkeypatch, em_exc=RuntimeError("东财连接被重置"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    assert out["ok"] is True, out["reason"]
    assert out["primary_source"] == "tencent"
    # 腾讯路径没有第三个源可比 → 交叉校验是"没做"，不能标 passed
    assert out["stats"]["cross_source"] == "skipped"

    rows = out["new_rows"]
    assert len(rows) == 2
    tx_new = [r for r in tx if r["date"] > local[-1]["date"]]
    for got, src in zip(rows, tx_new):
        # 四个价格全换算（只换 close 会留下假跳变）
        assert got["open"] == pytest.approx(src["open"] * HIST_RATIO)
        assert got["close"] == pytest.approx(src["close"] * HIST_RATIO)
        assert got["high"] == pytest.approx(src["high"] * HIST_RATIO)
        assert got["low"] == pytest.approx(src["low"] * HIST_RATIO)
        # 成交量换回本地的"手"口径
        assert got["volume"] == pytest.approx(src["volume"] * VOL_SCALE)
        # 成交额必须估算出正值，不能是 0（0 会被下游当成"这天零成交额"）
        assert got["amount"] > 0
    # 日期严格递增、且严格晚于本地末根（只追加不改写）
    assert [r["date"] for r in rows] == sorted(r["date"] for r in rows)
    assert all(r["date"] > local[-1]["date"] for r in rows)


def test_tencent_fallback_amount_estimate_is_close_to_truth(monkeypatch):
    """估算成交额要贴近真实值（本地 amount 与典型价×手数自洽）。"""
    local = _make_local()
    tx = _tx_from_local(local)
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    assert out["ok"] is True, out["reason"]
    assert out["stats"]["amount_mode"] == "estimated"
    assert out["stats"]["amount_median_err"] < 0.01
    # 合成数据的真实成交额可复算，估算应当对得上
    for row in out["new_rows"]:
        typ = (row["high"] + row["low"] + row["close"]) / 3.0
        assert row["amount"] == pytest.approx(row["volume"] * 100.0 * typ, rel=0.02)


def test_tencent_fallback_used_when_eastmoney_returns_empty(monkeypatch):
    """东财返回空序列 → 走回退，不是直接判死。"""
    local = _make_local()
    tx = _tx_from_local(local)
    _patch(monkeypatch, em_rows=[], tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    assert out["ok"] is True, out["reason"]
    assert out["primary_source"] == "tencent"
    assert out["new_rows"]


def test_tencent_fallback_no_new_rows_is_ok(monkeypatch):
    """腾讯也没有更新数据 → 不写任何行（new_rows 为空），不报错。"""
    local = _make_local()
    tx = _tx_from_local(local, n_new=0)
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    assert out["ok"] is True, out["reason"]
    assert out["primary_source"] == "tencent"
    assert out["new_rows"] == []
    assert out["stats"]["cross_source"] == "skipped"


# ------------------------------------------------------------------ 3. 回退失败路径


def _assert_failed_without_rows(out, dict):
    assert out["ok"] is False, out.get("reason")
    assert not out.get("new_rows"), "失败时不得返回 new_rows（会被 append 进历史）"


def test_tencent_fallback_rejects_unstable_ratio_band(monkeypatch):
    """历史比值带离散 > 1% → 整只判失败，不写入。"""
    local = _make_local()
    # 比值在窗口内从 1.000 漂到 1.03 → 离散远超 1%
    tx = _tx_from_local(local, ratio=1.000)
    for i, r in enumerate(tx):
        if r["date"] < local[-1]["date"]:
            f = 1.0 + 0.03 * (i / max(1, len(tx) - 3))
            for k in ("open", "close", "high", "low"):
                r[k] /= f
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    _assert_failed_without_rows(out, out)
    assert "比值带不稳" in out["reason"]
    assert out["stats"]["cross_source"] == "skipped"


def test_tencent_fallback_rejects_too_short_overlap(monkeypatch):
    """重叠天数 < 20 → 比值带样本不足，判失败。"""
    local = _make_local(n=10)
    tx = _tx_from_local(local)
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    _assert_failed_without_rows(out, out)
    assert "比值带" in out["reason"] or "重叠" in out["reason"]


def test_tencent_fallback_rejects_boundary_ratio_drift(monkeypatch):
    """比值带整体还算稳，但边界比例漂了 → 判失败（漂移 > RATIO_TOL）。"""
    local = _make_local()
    # 最后 5 个重叠日比例漂 0.5%：比值带离散 ~0.5%（< 1%，过带），
    # 但边界比例与历史比值带偏离 ~0.44% > RATIO_TOL(0.2%) → 必须判失败
    tx = _tx_from_local(local, ratio=HIST_RATIO, ratio_tail=HIST_RATIO * 1.005)
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    _assert_failed_without_rows(out, out)
    assert "漂移" in out["reason"]
    assert out["stats"]["cross_source"] == "skipped"


def test_tencent_fallback_rejects_partial_local_last_bar(monkeypatch):
    """本地末根是盘中快照 → 边界锚点是错的 → 必须判失败，不写脏数据。

    本地历史最后一根可能是盘中抓的：收盘价是盘中价（不是当天收盘价），
    成交量只有全天的一小截。实测这类末根的边界比值会偏离历史比值带
    0.7%~1.5%（远超 RATIO_TOL）—— 按 hist_ratio 换算会在边界造出
    一个 1% 级别的假价格跳变，直接毁掉收益率。

    正确做法：拒绝这只标的（失败关闭），等本地历史被完整重抓后再来。
    """
    local = _make_local()
    # 模拟盘中快照：末根收盘价是盘中价（不是收盘价），成交量只有 8%
    local[-1] = dict(local[-1])
    local[-1]["close"] = local[-1]["close"] * 0.99
    local[-1]["volume"] = local[-1]["volume"] * 0.08
    local[-1]["amount"] = local[-1]["amount"] * 0.08
    # 腾讯那边是完整的当天数据
    tx = _tx_from_local(local[:-1])
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    assert out["ok"] is False, out.get("reason")
    assert not out.get("new_rows"), "失败时不得返回 new_rows"
    assert "漂移" in out["reason"] or "边界" in out["reason"]
    assert out["stats"]["cross_source"] == "skipped"


def test_tencent_fallback_fails_when_both_sources_down(monkeypatch):
    """东财和腾讯都挂 → 判失败，不返回任何新行。"""
    local = _make_local()
    _patch(monkeypatch,
           em_exc=RuntimeError("东财挂了"), tx_exc=RuntimeError("腾讯也挂了"))

    out = daily_loop.validate_secid(SECID, local)

    _assert_failed_without_rows(out, out)
    assert out["primary_source"] == "tencent"
    assert out["stats"]["cross_source"] == "skipped"


def test_tencent_fallback_fails_when_tencent_returns_empty(monkeypatch):
    """东财挂了、腾讯返回空序列 → 判失败，不返回新行。"""
    local = _make_local()
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=[])

    out = daily_loop.validate_secid(SECID, local)

    _assert_failed_without_rows(out, out)
    assert out["primary_source"] == "tencent"


def test_tencent_fallback_rejects_bad_amount_estimate(monkeypatch):
    """成交额估算口径对不上（回测偏差超阈值）→ 判失败，绝不写脏成交额。"""
    local = _make_local()
    tx = _tx_from_local(local)
    # 把腾讯的成交量口径搞乱（单位比不稳）→ 成交额估算必然对不上
    for i, r in enumerate(tx):
        r["volume"] *= (1.0 + 0.5 * (i % 3))
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    _assert_failed_without_rows(out, out)
    assert out["primary_source"] == "tencent"


def test_tencent_fallback_rejects_price_jump(monkeypatch):
    """换算后价格跳变离谱 → 判失败（换算系数不可信）。"""
    local = _make_local()
    tx = _tx_from_local(local)
    for r in tx:
        if r["date"] > local[-1]["date"]:
            for k in ("open", "close", "high", "low"):
                r[k] *= 1.5          # 凭空造一个 50% 的跳变
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    _assert_failed_without_rows(out, out)
    assert out["primary_source"] == "tencent"


# ------------------------------------------------------------------ 4. 常量没被放松


def test_fallback_thresholds_not_loosened():
    """回退路径的门槛必须与东财路径同量级，不能比原来松。"""
    assert daily_loop.RATIO_TOL == 0.002
    assert daily_loop.OVERLAP_TOL == 1e-6
    assert daily_loop.TX_FALLBACK_MIN_N >= 20
    assert daily_loop.TX_FALLBACK_MAX_SPREAD <= 0.01


def test_tencent_fallback_requires_ratio_band_of_at_least_20_days(monkeypatch):
    """比值带必须至少有 20 个重叠日（与东财路径同门槛）。"""
    local = _make_local(n=60)
    tx = _tx_from_local(local)
    # 只留 19 个重叠日
    keep = {r["date"] for r in tx[:19]}
    tx = [r for r in tx if r["date"] in keep or r["date"] > local[-1]["date"]]
    _patch(monkeypatch, em_exc=RuntimeError("东财挂了"), tx_rows=tx)

    out = daily_loop.validate_secid(SECID, local)

    _assert_failed_without_rows(out, out)
    assert out["stats"].get("hist_n", 0) < 20
