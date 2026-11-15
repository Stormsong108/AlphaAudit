"""样本外实验室的防泄漏 / 成本核算测试。

这些测试是整个研究链路的护栏：一旦失败，说明结论不可信。
"""
from __future__ import annotations

import numpy as np
import pytest

import ooslab


def make_panel(T=120, N=40, seed=7, drift=0.0):
    """合成一个可控面板：随机游走价格 + 随机因子分数。"""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.02, size=(T, N))
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    score = rng.normal(size=(T, N))
    fwd = np.full((T, N), np.nan)
    fwd[:-5] = close[5:] / close[:-5] - 1.0
    dates = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(T)]
    return {
        "dates": dates,
        "codes": [f"C{i:04d}" for i in range(N)],
        "close": close,
        "fwd": fwd,
        "amount": np.full((T, N), 5e7),
        "factors": {"mom_20": score},
    }


# ------------------------------------------------------------------ z-score


def test_cs_zscore_rowwise():
    m = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, -5.0, 0.0, 5.0],
    ])
    z = ooslab.cs_zscore(m, min_valid=2)
    for row in z:
        assert abs(row.mean()) < 1e-9
        assert abs(row.std(ddof=1) - 1.0) < 1e-9


def test_cs_zscore_respects_min_valid():
    m = np.full((1, 5), np.nan)
    m[0, 0] = 1.0
    z = ooslab.cs_zscore(m, min_valid=3)
    assert np.all(np.isnan(z))


def test_cs_zscore_is_rowwise_not_cross_time():
    """第二行整体放大，不应影响第一行的 z 值。"""
    m = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    z = ooslab.cs_zscore(m, min_valid=2)
    assert abs(z[0, 0] - z[1, 0]) < 1e-9


# ------------------------------------------------------------------ 前瞻收益


def test_make_forward_matches_manual():
    """fwd[t] 必须等于 close[t+h]/close[t]-1，最后 h 行无未来数据。"""
    mat = make_panel(T=30, N=5)
    h = 3
    fwd = ooslab.make_forward(mat, h)
    close = mat["close"]
    for t in (0, 7, 26):
        for i in range(5):
            assert abs(fwd[t, i] - (close[t + h, i] / close[t, i] - 1.0)) < 1e-12
    assert np.all(np.isnan(fwd[30 - h:]))


def test_make_forward_uses_no_future_price():
    """改动 t+h 之后的价格，不得影响 fwd[t]（fwd[t] 只用到 t 和 t+h 两个价格）。"""
    mat = make_panel(T=30, N=5)
    base = ooslab.make_forward(mat, 5)
    mat2 = make_panel(T=30, N=5)
    mat2["close"][10:, :] *= 3.0        # 只影响 t+5 >= 10 即 t >= 5 的前瞻收益
    alt = ooslab.make_forward(mat2, 5)
    assert np.allclose(base[:5], alt[:5], equal_nan=True)
    assert not np.allclose(base[5:], alt[5:], equal_nan=True)


# ------------------------------------------------------------------ 因子方向


def test_negative_weight_flips_composite_ranking():
    """IC 为负的因子必须给负权重，否则会把反向信号当正向信号用。"""
    mat = make_panel()
    up = ooslab.composite_score(mat, ["mom_20"], [1.0])
    down = ooslab.composite_score(mat, ["mom_20"], [-1.0])
    assert np.allclose(up, -down, equal_nan=True)
    i0, i1 = 0, 1
    assert up[50, i0] > up[50, i1]
    assert down[50, i0] < down[50, i1]


def test_direction_corrected_beats_unsigned_on_reversed_panel():
    """合成一个「高分→未来跌」的面板：方向校正后多头必须赚钱，未校正必须亏。"""
    T, N = 300, 60
    rng = np.random.default_rng(5)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(T, N)), axis=0))
    fwd = np.full((T, N), np.nan)
    fwd[:-5] = close[5:] / close[:-5] - 1.0
    factor = -fwd * 2.0 + rng.normal(0, 0.01, size=(T, N))   # IC 为负
    factor[-5:] = np.nan
    mat = {
        "dates": [f"d{i:04d}" for i in range(T)],
        "codes": [f"C{i}" for i in range(N)],
        "close": close, "fwd": fwd, "amount": np.full((T, N), 1e8),
        "factors": {"mom_20": factor},
    }
    score_ok = ooslab.composite_score(mat, ["mom_20"], [-1.0])   # 方向校正
    score_bad = ooslab.composite_score(mat, ["mom_20"], [1.0])   # 符号用错
    ok = ooslab.decile_backtest(mat, score_ok, hold=5, lag=1, cost_rate=0.003, min_valid=10)
    bad = ooslab.decile_backtest(mat, score_bad, hold=5, lag=1, cost_rate=0.003, min_valid=10)
    assert ok["ok"] and bad["ok"]
    assert ok["long_only"]["net_annualized"] > 0
    assert ok["long_only"]["net_annualized"] > bad["long_only"]["net_annualized"]
    assert bad["long_only"]["net_annualized"] < 0


# ------------------------------------------------------------------ 合成


def test_composite_score_no_lookahead():
    """改某一天的因子值，只能影响那一天的合成分数。"""
    mat = make_panel()
    mat["factors"]["mom_20"] = np.random.default_rng(1).normal(size=mat["close"].shape)
    s1 = ooslab.composite_score(mat, ["mom_20"])
    mat2 = make_panel()
    mat2["factors"]["mom_20"] = mat["factors"]["mom_20"].copy()
    # 只污染第 50 天的其中一只标的（整行加常数会被 z-score 归一化掉，
    # 那样的改动本来就不影响合成分数，测不出东西）
    mat2["factors"]["mom_20"][50, 0] += 100.0
    s2 = ooslab.composite_score(mat2, ["mom_20"])
    affected = np.where(np.nanmax(np.abs(s1 - s2), axis=1) > 1e-9)[0]
    assert affected.tolist() == [50]


# ------------------------------------------------------------------ 成本核算


def test_zero_cost_equals_gross():
    mat = make_panel()
    score = ooslab.composite_score(mat, ["mom_20"])
    r = ooslab.decile_backtest(mat, score, hold=5, lag=1, cost_rate=0.0, min_valid=10)
    assert r["ok"]
    per = r["periods"]
    gross_mean = float(np.mean([p["long"] for p in per]))
    assert abs(r["long_only"]["net_mean_per_period"] - gross_mean) < 1e-12


def test_cost_is_deducted_by_turnover():
    mat = make_panel()
    score = ooslab.composite_score(mat, ["mom_20"])
    cost = 0.003
    r = ooslab.decile_backtest(mat, score, hold=5, lag=1, cost_rate=cost, min_valid=10)
    assert r["ok"]
    per = r["periods"]
    turn = np.array([p["turnover"] for p in per])
    gross = np.array([p["long"] for p in per])
    net = gross - cost * turn
    assert abs(r["long_only"]["net_mean_per_period"] - float(net.mean())) < 1e-12
    assert r["long_only"]["net_mean_per_period"] < gross.mean()


def test_flat_market_gross_zero_but_net_negative():
    """价格完全不动时，毛收益为 0，扣成本后必为负。"""
    T, N = 60, 30
    mat = {
        "dates": [f"2021-01-{i % 28 + 1:02d}" for i in range(T)],
        "codes": [f"C{i}" for i in range(N)],
        "close": np.full((T, N), 10.0),
        "fwd": np.zeros((T, N)),
        "amount": np.full((T, N), 1e8),
        "factors": {"mom_20": np.random.default_rng(3).normal(size=(T, N))},
    }
    score = ooslab.composite_score(mat, ["mom_20"])
    r = ooslab.decile_backtest(mat, score, hold=5, lag=1, cost_rate=0.003, min_valid=10)
    assert r["ok"]
    assert abs(r["gross"]["long_mean"]) < 1e-12
    assert r["long_only"]["net_mean_per_period"] < 0


# ------------------------------------------------------------------ 防未来函数


def test_entry_is_after_signal_date():
    """成交日必须严格晚于信号日——不能用信号日收盘价成交。"""
    mat = make_panel()
    score = ooslab.composite_score(mat, ["mom_20"])
    r = ooslab.decile_backtest(mat, score, hold=5, lag=1, min_valid=10)
    assert r["ok"]
    di = {d: i for i, d in enumerate(mat["dates"])}
    for p in r["periods"]:
        assert di[p["entry_date"]] > di[p["signal_date"]]
        assert di[p["exit_date"]] - di[p["entry_date"]] == 5


def test_future_prices_cannot_affect_past_signal_periods():
    """把 81 天之后的价格整体打乱，之前调仓期的收益不应改变。"""
    mat = make_panel()
    score = ooslab.composite_score(mat, ["mom_20"])
    base = ooslab.decile_backtest(mat, score, hold=5, lag=1, cost_rate=0.0, min_valid=10)

    mat2 = make_panel()
    mat2["factors"]["mom_20"] = mat["factors"]["mom_20"].copy()
    rng = np.random.default_rng(99)
    mat2["close"][81:, :] *= rng.uniform(0.4, 1.8, size=(mat2["close"].shape[0] - 81, 1))
    alt = ooslab.decile_backtest(mat2, score, hold=5, lag=1, cost_rate=0.0, min_valid=10)

    di = {d: i for i, d in enumerate(mat["dates"])}

    def early(r):
        return [(p["signal_date"], p["long"], p["short"], p["bench"])
                for p in r["periods"] if di[p["exit_date"]] <= 80]

    assert early(base) == early(alt)
    assert len(early(base)) > 0


# ------------------------------------------------------------------ 时间切分


def test_split_segments_are_ordered_and_cover_all():
    segs = ooslab.split_segments(1000)
    assert segs["train"][0] == 0
    assert segs["train"][1] == segs["val"][0]
    assert segs["val"][1] == segs["test"][0]
    assert segs["test"][1] == 1000


def test_discover_then_validate_detects_direction_reversal():
    """在训练段发现正 IC 的因子，放到符号相反的测试段应判定失效。"""
    T, N = 800, 60
    rng = np.random.default_rng(11)
    dates = [f"d{i:04d}" for i in range(T)]
    # 训练段(0-400)：因子 ≈ 未来收益（强正相关）；测试段(600-800)：因子 ≈ -未来收益
    # T=800 是为了让测试段(200 天)压缩成的不重叠 IC 序列有 39 个块，
    # 超过 validate_factors_oos 的 20 个块最低门槛。
    fwd = np.full((T, N), np.nan)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(T, N)), axis=0))
    fwd[:-5] = base[5:] / base[:-5] - 1.0
    noise = rng.normal(0, 0.008, size=(T, N))
    factor = np.full((T, N), np.nan)
    # 加噪声是必要的：完全等于前瞻收益会让 IC 序列恒为 1.0，
    # 方差为 0 时 t 统计无定义（newey_west 返回 None），测不到东西。
    factor[:400] = fwd[:400] + noise[:400]
    factor[400:] = -fwd[400:] * 3.0 + noise[400:]
    mat = {
        "dates": dates, "codes": [f"C{i}" for i in range(N)],
        "close": base, "fwd": fwd, "amount": np.full((T, N), 1e8),
        "factors": {"mom_20": factor},
    }
    segs = ooslab.split_segments(T)
    disc = ooslab.discover_factors(mat, segs["train"], horizon=5)
    assert any(d["name"] == "mom_20" for d in disc)
    oos = ooslab.validate_factors_oos(mat, disc, segs["test"], horizon=5)
    rec = [o for o in oos if o["name"] == "mom_20"][0]
    assert rec["same_sign"] is False
    assert "反转" in rec["oos_verdict"]


# ------------------------------------------------------------------ AUC


def test_auc_perfect_and_random():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert abs(ooslab.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) - 1.0) < 1e-9
    assert abs(ooslab.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) - 0.0) < 1e-9


def test_auc_single_class_returns_none():
    assert ooslab.auc(np.array([0.1, 0.2]), np.array([1.0, 1.0])) is None


# ------------------------------------------------------------------ 概率数据集


def test_probability_dataset_respects_label_modes():
    mat = make_panel(T=80, N=30)
    ds = ooslab.build_probability_dataset(mat, ["mom_20"], horizon=5, label="abs", max_per_date=100)
    assert ds["ok"]
    assert set(np.unique(ds["y"])).issubset({0.0, 1.0})
    assert ds["X"].shape[1] == 1
    assert len(ds["y"]) == ds["X"].shape[0]


def test_probability_dataset_rows_are_time_ordered():
    """行必须按日期升序，否则按行号切时间就切错了。"""
    mat = make_panel(T=60, N=20)
    ds = ooslab.build_probability_dataset(mat, ["mom_20"], horizon=5, max_per_date=50)
    assert ds["ok"]
    d = ds["date_idx"]
    assert np.all(np.diff(d) >= 0)


def test_standardize_uses_train_only():
    X = np.array([[0.0], [0.0], [100.0], [100.0]])
    Z, mu, sd = ooslab.standardize_by_train(X, 2)
    assert abs(mu[0]) < 1e-9
    assert abs(sd[0] - 1.0) < 1e-9
    assert Z[2, 0] > 50


# ------------------------------------------------------------------ 面板缓存指纹


def _write_stale_cache(path, fingerprint=None, with_factors=True):
    """造一个"旧面板"缓存：close 里放一个负价格，一眼就能认出来。"""
    import factorlab

    payload = {
        "dates": np.array(["2020-01-01"], dtype="U16"),
        "codes": np.array(["0.000001"], dtype="U16"),
        "close": np.array([[-1.0]], dtype=np.float32),
        "fwd": np.array([[0.0]], dtype=np.float32),
        "amount": np.array([[1.0]], dtype=np.float32),
    }
    if with_factors:
        # 因子齐全，否则 load_panel 会按"缓存不完整"重建，测不到指纹逻辑
        for k in factorlab.FACTOR_NAMES:
            payload[f"f_{k}"] = np.zeros((1, 1), dtype=np.float32)
    if fingerprint is not None:
        payload["fingerprint"] = np.array(fingerprint)
    np.savez_compressed(path, **payload)


def test_cache_without_fingerprint_is_rebuilt(tmp_path, monkeypatch):
    """老缓存没有指纹字段 → 必须重建，不能直接读回来。

    回归 2026-11-10 的坑：判断写成 `if cached_fp and cached_fp != fp`，
    空指纹会短路放行 —— 换复权口径之后旧面板照样被读回来，
    于是"改了半天还在用旧数据"，而且旧面板里是负价格。
    """
    import factorlab
    import history

    cache = tmp_path / "panel.npz"
    _write_stale_cache(cache, fingerprint=None)
    assert "fingerprint" not in np.load(cache)

    monkeypatch.setattr(history, "HIST_DIR", tmp_path)
    monkeypatch.setattr(ooslab, "_kline_fingerprint", lambda: "CURRENT_FP")
    fresh = {
        "dates": ["2020-01-01"],
        "codes": ["0.000001"],
        "close": np.array([[7.0]], dtype=np.float32),
        "fwd": np.array([[0.0]], dtype=np.float32),
        "amount": np.array([[1.0]], dtype=np.float32),
        "factors": {},
    }
    monkeypatch.setattr(factorlab, "build_matrix", lambda *a, **k: fresh)

    panel = ooslab.load_panel(cache=cache, strict=False)
    assert float(panel["close"][0, 0]) == 7.0, "旧缓存被读回来了"


def test_cache_with_stale_fingerprint_is_rebuilt(tmp_path, monkeypatch):
    """指纹对不上 → 必须重建。"""
    import factorlab
    import history

    cache = tmp_path / "panel.npz"
    _write_stale_cache(cache, fingerprint="OLD_FP")

    monkeypatch.setattr(history, "HIST_DIR", tmp_path)
    monkeypatch.setattr(ooslab, "_kline_fingerprint", lambda: "NEW_FP")
    fresh = {
        "dates": ["2020-01-01"],
        "codes": ["0.000001"],
        "close": np.array([[9.0]], dtype=np.float32),
        "fwd": np.array([[0.0]], dtype=np.float32),
        "amount": np.array([[1.0]], dtype=np.float32),
        "factors": {},
    }
    monkeypatch.setattr(factorlab, "build_matrix", lambda *a, **k: fresh)

    panel = ooslab.load_panel(cache=cache, strict=False)
    assert float(panel["close"][0, 0]) == 9.0


def test_cache_with_matching_fingerprint_is_reused(tmp_path, monkeypatch):
    """指纹一致 → 复用缓存，不重建（省掉 20 秒重建）。"""
    import history

    cache = tmp_path / "panel.npz"
    _write_stale_cache(cache, fingerprint="SAME_FP")
    monkeypatch.setattr(history, "HIST_DIR", tmp_path)
    monkeypatch.setattr(ooslab, "_kline_fingerprint", lambda: "SAME_FP")

    def _boom(*a, **k):
        raise AssertionError("指纹一致却重建了面板")

    import factorlab
    monkeypatch.setattr(factorlab, "build_matrix", _boom)
    panel = ooslab.load_panel(cache=cache, strict=False)
    assert float(panel["close"][0, 0]) == -1.0
