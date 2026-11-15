# -*- coding: utf-8 -*-
"""ETF 低成本通道研究。

为什么另开一条通道
------------------
个股通道的**不可压缩成本地板**是 10.2bp/往返（单笔 ≥2 万时：佣金万2.5双边 5bp
+ 印花税卖出单边 5bp + 沪市过户费 0.2bp），加滑点后 20.2bp。
而 canonical 策略的毛边际只有 2.60bp/期（多头）/ 5.24bp/期（多空），
差 8~16bp —— 不是调参能补上的量级。

场内基金（ETF/LOF）按券商官方费率表（国新证券《证券交易佣金收取标准》、
中金财富佣金公示）确认：**无印花税、无过户费**，只收佣金（佣金含经手费 0.004%，
不得重复加）。因此成本地板低一个数量级：
    佣金万0.5 + 单笔 ≥10 万 → 往返 1bp；单笔 ≥2 万 → 往返 5bp。
这让「毛边际 2.6~5.2bp」的现有信号**第一次有机会变成净正收益**。

纪律（与个股通道完全一致）
--------------------------
训练段选参 → 验证段只确认不改参 → 参数冻结 → 测试段只跑一次。
收益统计三口径并列（缩尾均值/未缩尾均值/中位数），不看均值掩盖坏数据。
所有统计零 LLM、固定随机种子。

已知偏差（必须披露）
--------------------
ETF 池是「今天仍存续」的活跃 ETF 名单 —— 存在**幸存者偏差**。
但 ETF 跟踪的是宽基/行业指数，标的不死则基金不灭，死亡率远低于个股；
且池内成员按**上市日**逐日生效（点时成员资格），不向前回填。
本模块把这一偏差写进输出，不隐藏。
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
ETF_DIR = os.path.join(REPO, "data", "history", "kline_etf")
ADJ_DIR = os.path.join(REPO, "data", "history", "kline_etf_adj")
OUT_DIR = os.path.join(REPO, "data", "research")
OUT_PATH = os.path.join(OUT_DIR, "etf_channel.json")
MANIFEST = "/Users/sst/.hermes/profiles/xiaojin/cache/scratch/etf_manifest.json"

# ── ETF 成本参数（出处见文件头）─────────────────────────────────────────
COMMISSION_RATE = 0.00005   # 万0.5（券商可谈档位；法定上限 3‰，起点 5 元）
COMMISSION_MIN = 5.0        # 单笔最低佣金，买卖各收一次
STAMP_DUTY = 0.0            # 场内基金不征印花税
TRANSFER_FEE = 0.0          # 场内基金不收过户费（官方费率表未列此项）
SLIPPAGE_BP_EDGE = 2.0      # 单边滑点

TRADING_DAYS = 252


# ── 数据装载 ────────────────────────────────────────────────────────────
def load_panel(min_bars: int = 120, adjusted: bool = True,
               exclude: set[str] | None = None) -> dict[str, Any]:
    """把 ETF 日线装成 (T, N) 面板。上市日之前一律 NaN，不回填。

    adjusted=True 时读 data/history/kline_etf_adj（除息复权后的价格），
    用 adj_open / adj_close；成交额不受复权影响，仍用原始 close×volume。
    exclude：数据质量审计未通过的品种，直接剔除，不进策略。
    """
    src = ADJ_DIR if adjusted else ETF_DIR
    if not os.path.isdir(src):
        src = ETF_DIR
    exclude = exclude or set()
    files = sorted(f for f in os.listdir(src) if f.endswith(".json"))
    series: dict[str, dict[str, tuple[float, float, float]]] = {}
    for fn in files:
        code = fn[:-5]
        if code in exclude:
            continue
        rows = json.load(open(os.path.join(src, fn), encoding="utf-8"))
        d: dict[str, tuple[float, float, float]] = {}
        for r in rows:
            try:
                o = float(r.get("adj_open") or r["open"])
                c = float(r.get("adj_close") or r["close"])
                d[r["date"]] = (o, c, float(r.get("amount") or 0.0))
            except (KeyError, TypeError, ValueError):
                continue
        if len(d) >= min_bars:
            series[code] = d
    if not series:
        raise RuntimeError(f"ETF 目录无可用数据: {ETF_DIR}")

    alld = sorted({d for s in series.values() for d in s})
    idx = {d: i for i, d in enumerate(alld)}
    T, N = len(alld), len(series)
    codes = sorted(series)
    op = np.full((T, N), np.nan)
    cl = np.full((T, N), np.nan)
    am = np.full((T, N), np.nan)
    for j, code in enumerate(codes):
        for d, (o, c, a) in series[code].items():
            i = idx[d]
            op[i, j], cl[i, j], am[i, j] = o, c, a
    names = {}
    if os.path.exists(MANIFEST):
        for secid, meta in json.load(open(MANIFEST, encoding="utf-8")).items():
            names[secid.replace(".", "_")] = meta.get("name", secid)
    return {
        "dates": alld, "codes": codes,
        "names": [names.get(c, c) for c in codes],
        "open": op, "close": cl, "amount": am,
    }


# ── 秩工具 ──────────────────────────────────────────────────────────────
def pd_ranks(vals: np.ndarray) -> np.ndarray:
    """平均秩（1..n），并列取均值。"""
    n = len(vals)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    sv = vals[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _rank_rows(mat: np.ndarray) -> np.ndarray:
    """逐日横截面秩，归一到 [-0.5, 0.5]；有效数 <3 的行保持 NaN。"""
    out = np.full_like(mat, np.nan)
    for i in range(mat.shape[0]):
        row = mat[i]
        v = ~np.isnan(row)
        if int(v.sum()) < 3:
            continue
        out[i, v] = pd_ranks(row[v]) - 0.5
    return out


# ── 滚动统计（向量化）──────────────────────────────────────────────────
def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.nan_to_num(a, nan=0.0), axis=0)
    n = np.cumsum(~np.isnan(a), axis=0)
    out = np.full_like(a, np.nan)
    if a.shape[0] > w:
        out[w:] = (c[w:] - c[:-w]) / np.maximum(n[w:] - n[:-w], 1)
    out[n < w] = np.nan
    return out


def _rolling_std(a: np.ndarray, w: int) -> np.ndarray:
    m = _rolling_mean(a, w)
    m2 = _rolling_mean(a * a, w)
    v = np.maximum(m2 - m * m, 0.0)
    out = np.sqrt(v)
    out[np.isnan(m)] = np.nan
    return out


def _rolling_max(a: np.ndarray, w: int) -> np.ndarray:
    """窗口含 NaN 则结果 NaN（上市前的前缀 NaN 会自动传染）。"""
    T = a.shape[0]
    out = np.full_like(a, np.nan)
    if T < w:
        return out
    win = np.lib.stride_tricks.sliding_window_view(a, w, axis=0)
    out[w - 1:] = np.max(win, axis=2)
    return out


def _rsi(a: np.ndarray, w: int = 14) -> np.ndarray:
    """RSI(w)：截至当日收盘的 w 期相对强弱，逐列独立。

    窗口 = d[r-w .. r-1]（r 与 r-1 之间恰好 w 个涨跌幅）。
    只用截至当行的信息；窗口内价格有缺失（上市前/缺口）则输出 NaN。

    注：这里的布尔掩码必须显式加 [:, None] —— 2D 面板下条件形状 (M,)
    对不上值形状 (M, N)，np.where 会把条件广播到最后一轴从而报错。
    """
    T = a.shape[0]
    out = np.full_like(a, np.nan)
    if T < w + 1:
        return out
    d = np.diff(a, axis=0)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    cg = np.cumsum(gain, axis=0)
    cl_ = np.cumsum(loss, axis=0)
    rows = np.arange(w, T)          # 有效输出行
    hi = rows - 1                   # cg/cl 下标上界（cg 长度 T-1）
    lo = rows - w                   # 窗口起点（d 的下标）
    has_prefix = lo > 0             # 窗口起点之前是否还有 d 可作前缀
    idx = np.maximum(lo - 1, 0)
    base_g = np.where(has_prefix[:, None], cg[idx], 0.0)
    base_l = np.where(has_prefix[:, None], cl_[idx], 0.0)
    sg = cg[hi] - base_g
    sl = cl_[hi] - base_l
    denom = sg + sl
    r = np.where(denom > 0, sg / np.where(denom > 0, denom, 1.0), 0.5)
    # 窗口内价格必须全为有限值（a[lo .. rows]，共 w+1 个）
    cf = np.cumsum(np.isfinite(a).astype(np.int64), axis=0)
    cnt = cf[rows] - np.where(has_prefix[:, None], cf[idx], 0)
    # 逐元素有效性：该 (行, 列) 的窗口内 w+1 个价格必须全是有限值。
    # 注意保持 (M, N) 形状 —— 不能用 np.all(axis=1)，那会把列折叠成行，
    # 导致「一列上市前无数据」把其它列一起判死。
    win_ok = cnt == (w + 1)
    out[rows] = np.where(win_ok, r, np.nan)
    return out


def compute_factors(cl: np.ndarray) -> dict[str, np.ndarray]:
    """逐 ETF 的时序因子，全部只用截至当日的信息（无前视）。"""
    F: dict[str, np.ndarray] = {}

    def shift(a: np.ndarray, k: int) -> np.ndarray:
        out = np.full_like(a, np.nan)
        if k < a.shape[0]:
            out[:-k] = a[k:]
        return out

    for w in (20, 60, 120):
        F[f"mom_{w}"] = cl / shift(cl, w) - 1.0
    for w in (20, 60):
        F[f"vol_{w}"] = _rolling_std(np.log(np.where(cl > 0, cl, np.nan)), w)
    F["ma_dev_60"] = cl / _rolling_mean(cl, 60) - 1.0
    for w in (60, 120):
        F[f"dd_{w}"] = cl / _rolling_max(cl, w) - 1.0
    F["rsi_14"] = _rsi(cl, 14)
    return F


# ── 组合打分 ────────────────────────────────────────────────────────────
def composite_score(factors: dict[str, np.ndarray], keys: list[str],
                     signs: dict[str, float] | None = None) -> np.ndarray:
    """多因子**秩**等权平均（先秩化再平均，避免量纲主导）。"""
    signs = signs or {}
    acc = None
    for k in keys:
        r = _rank_rows(factors[k]) * signs.get(k, 1.0)
        acc = r if acc is None else acc + r
    if acc is None:
        raise ValueError("keys 为空")
    return acc / len(keys)


# ── 成本模型 ────────────────────────────────────────────────────────────
def etf_roundtrip_cost_rate(per_order_yuan: float,
                            slippage_bp_edge: float = SLIPPAGE_BP_EDGE) -> dict[str, float]:
    """单笔往返成本（比例）。含 ¥5 最低佣金，无印花税、无过户费。"""
    if per_order_yuan <= 0:
        raise ValueError("per_order_yuan 必须为正")
    comm_one = max(COMMISSION_MIN, per_order_yuan * COMMISSION_RATE)
    comm_bp = 2 * comm_one / per_order_yuan * 1e4
    slip_bp = 2 * slippage_bp_edge
    total_bp = comm_bp + slip_bp + (STAMP_DUTY + TRANSFER_FEE) * 1e4
    return {
        "commission_bp_roundtrip": comm_bp,
        "slippage_bp_roundtrip": slip_bp,
        "total_bp_roundtrip": total_bp,
        "total_rate_roundtrip": total_bp / 1e4,
    }


def rebalance_cost(capital: float, n_held: int, n_sold: int, n_bought: int,
                   slippage_bp_edge: float = SLIPPAGE_BP_EDGE) -> float:
    """一次调仓的实际成本（金额）。

    每次卖出 s 只、买入 b 只，单笔名义金额 ≈ capital / n_held。
    佣金按**双边各收一次**、每笔不足 5 元按 5 元收；
    滑点按名义金额 × 单边 bp（买卖各一次）。
    """
    if n_held <= 0:
        return 0.0
    per = capital / n_held
    one_comm = max(COMMISSION_MIN, per * COMMISSION_RATE)
    n_orders = n_sold + n_bought
    return n_orders * one_comm + n_orders * per * (slippage_bp_edge * 1e-4)


# ── 回测 ────────────────────────────────────────────────────────────────
def run_strategy(panel: dict[str, Any], score: np.ndarray, k: int, R: int,
                 *, price_mode: str = "open", capital: float = 1_000_000.0,
                 min_amount: float = 5e7, cost: bool = True,
                 slippage_bp_edge: float = SLIPPAGE_BP_EDGE,
                 lo: int = 1, hi: int | None = None) -> dict[str, Any]:
    """因子分最高的 k 只 ETF 轮动。

    price_mode='open' ：信号用 t-1 收盘算，t 日开盘买入，t+R 日开盘卖出
    price_mode='close'：信号用 t-1 收盘算，t 日收盘买入，t+R 日收盘卖出
    """
    cl, op, am = panel["close"], panel["open"], panel["amount"]
    T = cl.shape[0]
    if hi is None:
        hi = T
    px = op if price_mode == "open" else cl
    sig = np.full_like(score, np.nan)
    sig[1:] = score[:-1]
    # 流动性过滤必须用**截至 t−1 的**成交额均值：t 日开盘下单时，当日成交额尚未知，
    # 用当日额过滤 = 前视偏差。
    amt_ma = _rolling_mean(am, 20)

    periods: list[dict[str, Any]] = []
    prev_held: np.ndarray | None = None
    i = lo
    while i + R < hi:
        s, row_px, row_amt = sig[i], px[i], amt_ma[i - 1]
        ok = ~np.isnan(s) & ~np.isnan(row_px) & ~np.isnan(row_amt) & (row_amt >= min_amount)
        if int(ok.sum()) < 2:
            i += R
            continue
        idxs = np.where(ok)[0]
        order = idxs[np.argsort(-s[idxs], kind="mergesort")]
        pick = order[:k]
        entry = row_px[pick]
        # 卖出价必须取**同一批 pick** 的价格：px[i+R] 是全市场一整行，
        # 与 entry（长度 k）做布尔掩码会广播错位 —— k≠N 时直接崩，
        # k 恰好等于 N 时会算出「全市场平均收益」冒充策略收益。
        exit_px = px[i + R][pick]
        valid = ~np.isnan(entry) & ~np.isnan(exit_px)
        if int(valid.sum()) == 0:
            i += R
            continue
        pick = pick[valid]
        gross = float(np.mean(exit_px[valid] / entry[valid] - 1.0))
        if prev_held is None or len(prev_held) == 0:
            n_sold, n_bought = 0, len(pick)
        else:
            ps, ns = set(prev_held.tolist()), set(pick.tolist())
            n_sold, n_bought = len(ps - ns), len(ns - ps)
        c = 0.0
        if cost and len(pick) > 0:
            c = rebalance_cost(capital, len(pick), n_sold, n_bought, slippage_bp_edge) / capital
        periods.append({"date": panel["dates"][i], "gross": gross, "cost": c,
                        "n_held": len(pick), "n_sold": n_sold, "n_bought": n_bought})
        prev_held = pick
        i += R
    return {"periods": periods, "R": R, "k": k}


def metrics(periods: list[dict[str, Any]], R: int) -> dict[str, Any]:
    """三口径收益统计（缩尾均值 / 未缩尾均值 / 中位数）。"""
    if not periods:
        return {"n_periods": 0}
    g = np.array([p["gross"] for p in periods], dtype=float)
    c = np.array([p["cost"] for p in periods], dtype=float)
    net = g - c
    ann = TRADING_DAYS / R
    sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    lo, hi = np.quantile(net, [0.05, 0.95])
    return {
        "n_periods": int(len(net)),
        "gross_per_period": float(np.mean(g)),
        "gross_annualized": float(np.mean(g) * ann),
        "net_per_period": float(np.mean(net)),
        "net_annualized": float(np.mean(net) * ann),
        "net_winsor_mean_annualized": float(np.mean(np.clip(net, lo, hi)) * ann),
        "net_median_annualized": float(np.median(net) * ann),
        # 年化 Sharpe = (每期均值 × 年化次数) / (每期标准差 × √年化次数)
        #              = (mean / sd) × √ann
        # ⚠️ 原来写成 mean/(sd×√ann)，把年化因子**除反了** —— 报出来的 Sharpe
        # 小了 ann 倍（实测 ann=25.2 时报 0.018，真实值 0.44）。项目其余地方
        # （ooslab.py / execution_realism.py / market_timing.py / timing.py）
        # 用的都是正确写法，这里是唯一的例外。
        "sharpe": float(np.mean(net) / sd * np.sqrt(ann)) if sd > 0 else 0.0,
        "win_rate": float(np.mean(net > 0)),
        "median_period_net_bp": float(np.median(net) * 1e4),
        "mean_cost_per_period_bp": float(np.mean(c) * 1e4),
        "mean_held": float(np.mean([p["n_held"] for p in periods])),
        "max_drawdown_ann": _max_dd(net, ann),
    }


def _max_dd(net: np.ndarray, ann: float) -> float:
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())


def equal_weight_benchmark(panel: dict[str, Any], R: int, lo: int, hi: int,
                           price_mode: str = "open",
                           min_amount: float = 5e7,
                           dates_filter: list[str] | None = None) -> dict[str, Any]:
    """等权持有全部合格 ETF（点时成员资格 + 截至 t−1 的流动性）。

    dates_filter 不为 None 时，只保留这些日期上的期次。
    用途：把基准和策略对齐到**同一批期次**。策略会因流动性不足 /
    横截面排名不足跳过一些期次（训练段实测 183 期 vs 基准 192 期），
    基准若不跟着跳过，超额就是拿两套不同样本的年化相减 —— 口径串了。
    """
    px = panel["open"] if price_mode == "open" else panel["close"]
    am = panel["amount"]
    T = px.shape[0]
    amt_ma = _rolling_mean(am, 20)
    rs, dates = [], []
    # 起点必须与 run_strategy 一致（i = lo），否则两条时间网格错开，
    # 基准日期集与策略日期集不相交 → 对齐后全空 → 基准变 NaN。
    i = lo
    while i + R < min(hi, T):
        elig = (~np.isnan(px[i]) & ~np.isnan(px[i + R])
                & ~np.isnan(amt_ma[i - 1]) & (amt_ma[i - 1] >= min_amount))
        if int(elig.sum()) >= 2:
            rs.append(float(np.mean(px[i + R][elig] / px[i][elig] - 1.0)))
            dates.append(panel["dates"][i])
        i += R
    if dates_filter is not None:
        keep = set(dates_filter)
        pairs = [(d, v) for d, v in zip(dates, rs) if d in keep]
        rs = [v for _, v in pairs]
        dates = [d for d, _ in pairs]
    if not rs:
        return {"n_periods": 0}
    r = np.array(rs)
    ann = TRADING_DAYS / R
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return {"n_periods": len(r), "annualized": float(np.mean(r) * ann),
            "sharpe": float(np.mean(r) / sd * np.sqrt(ann)) if sd > 0 else 0.0,
            "win_rate": float(np.mean(r > 0)), "dates": [dates[0], dates[-1]]}


def split(n: int, train_frac: float = 0.5, val_frac: float = 0.25) -> dict[str, tuple[int, int]]:
    a = int(n * train_frac)
    b = a + int(n * val_frac)
    return {"train": (0, a), "val": (a, b), "test": (b, n)}


def filter_periods(periods: list[dict[str, Any]], dates: list[str],
                   lo: int, hi: int) -> list[dict[str, Any]]:
    dset = set(dates[lo:hi])
    return [p for p in periods if p["date"] in dset]
