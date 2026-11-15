# -*- coding: utf-8 -*-
"""ETF 除息 + 份额折算复权（v2：F10 权威锚点，份额折算真正进复权因子）。

为什么必须做
------------
新浪给的是**不复权**价格。ETF 的两类机械事件会让价格序列凭空跳空：

  1. 现金分红除息日：价格机械性下跌 d/前收盘，与市场无关。
  2. 份额折算日（F10「份额折算 / 份额分拆」）：份额数 ×X ⇒ 单位净值 ÷X ⇒
     价格 ÷X。实测单日跳空可达 ±200%，而 ETF 二级市场涨跌幅限制 ±10%，
     真实行情不可能跳 25% 以上 —— 所以这些跳空**全是机械的**，必须消掉。

若直接用不复权价做横截面排序，会把「刚折算/刚除息的 ETF」标成跌幅最大的
候选 —— **凭空造出一个反转信号**。实测：纳指ETF 2022-01-14 单日 −80% 的假跌，
直接把策略年化从 +15% 拖到 −122%。

事件锚点：F10 分红送配页（权威）
--------------------------------
`https://fundf10.eastmoney.com/fhsp_{code6}.html` 的表格同时给出：

  - **份额折算**：折算日 + 「1:X」（X = 折算后份额 / 折算前份额）
    ⇒ 价格除以 X。
  - **现金分红**：权益登记日 + 除息日 + 每 10 份派现金额
    ⇒ 每股分红 = 每10份金额 / 10，除息日价格机械下跌该金额。

NAV 代数（交叉校验源，不是唯一依据）
----------------------------------
净值数据满足恒等式：

        C(t) = k · ( U(t) + D(t) )

  C = 累计净值, U = 单位净值, D = 累计每份分红, k = 份额倍数（折算造成）

  - **k 在份额折算时跳变**（C/U 比值跳变 > 8%），一段 regime 内 k 恒定。
  - **C 的日涨跌幅 = 基金总收益率**（含分红再投资），C 在折算日连续。
  - 除息日：U 跌 d 而 D 涨 d ⇒ C 几乎不动 ⇒ C/U **小幅上跳**（实测 < 2%）。

于是每日分红可反解（注意是 **除以 k**，不是乘 k）：

        d_true(t) = ΔC(t)/k − ΔU(t) = ( ΔC(t) − k·ΔU(t) ) / k

  实测 17 笔分红事件：d_true / F10 官方金额 的比值中位数 1.001，17/17 落在
  [0.97, 1.03]。**份额折算的倍数直接就是 k 的跳变幅度**，两者必须一致。

两源冲突怎么办
--------------
  - 折算：**以 F10 为准**（比例 + 日期）。NAV 代数只用来交叉校验并在 F10 缺失时兜底。
  - 分红：NAV 反解 d_true 为主，F10 官方金额为准；两者不一致时用 F10 并**报告**。
  - 债基/货币型（k 极小，如 511010 k=0.0087）NAV 代数不可靠 ⇒ **直接用 F10 金额**。

折算落在哪一天
--------------
NAV 折算日（基金记账日）与**价格跳变日**（交易所除权日）**不是同一天**，
实测偏移 1~3 个交易日（含长假）。做法：以 F10 折算日为候选，在价格序列里
按**比例匹配**（跳变后价格 ×X ≈ 1，即隐含真实日收益在 ±20% 内）定位真正的
跳变日，复权调整落在那一天。绝不写死「折算日 + 1 天」。

复权公式
--------
对每个交易日 t：

        r_adj(t) = ( close(t) + div(t) ) · X(t) / close(t−1) − 1
        factor(t) = factor(t−1) · (1 + r_adj(t)) / (1 + r_price(t))
        adj_close(t) = close(t) · factor(t)，adj_open(t) = open(t) · factor(t)

  - 普通日：X=1, div=0 ⇒ r_adj = r_price（不动）。
  - 除息日：X=1 ⇒ r_adj = r_price + div/close(t−1)（补回机械除息缺口）。
  - 折算日：X=折算倍数 ⇒ 价格被机械除以 X，factor 同比例乘 X，折算前后 adj 价格连续。
  - 折算日恰逢除息（实测 sz159922 2024-12-02）：div 是折算后的每股口径，
    公式统一成立。

校验（不通过就不许继续跑策略）
----------------------------
  1. 复权后 close→close 收益 与 净值总收益（C 的日变化率）的相关系数
  2. 每年检出的除息次数（应在 0~3 次/年）
  3. 累计 factor 量级（复权后总量应 ≥ 份额倍数累计比 k_last/k_first）
  4. 复权后不应残留 > 25% 的单日跳空（真实行情不可能，ETF 限涨 ±10%）
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
NAV_DIR = os.path.join(REPO, "data", "history", "nav_etf")
ETF_DIR = os.path.join(REPO, "data", "history", "kline_etf")
ADJ_PATH = os.path.join(REPO, "data", "history", "kline_etf_adj")
F10_PATH = os.path.join(REPO, "data", "research", "etf_f10_actions.json")
F10_URL = "https://fundf10.eastmoney.com/fhsp_{code6}.html"

REGIME_JUMP = 0.08       # C/U 跳变 > 8% 判为份额折算（除息只造成 ~1~2%）
K_WINDOW = 250          # k 的滚动估计窗口（保留常量以兼容旧调用）
DIV_REL_THRESH = 0.0015  # d/U > 15bp 判为除息（远高于 NAV 四舍五入噪声 ~1bp）

FOLD_WINDOW = 10        # 折算日之后多少个交易日内寻找价格跳变日
FOLD_IMPLIED_MAX = 0.25  # 跳变日隐含真实日收益的绝对值上限（ETF 限涨 ±10~20%）
FOLD_JUMP_MIN = 0.02    # 价格跳变的最小幅度（2%），排除噪声日
DIV_RATIO_TOL = 0.03    # d_true / F10 金额 的相对容差（超出则以 F10 为准）
TINY_K = 0.05           # k 低于此值 ⇒ NAV 反解不可靠（债基/货币型），直接用 F10
MIN_SEG_POINTS = 60     # 每段至少要这么多个样本才允许作为独立 regime 段


# ── 净值 ────────────────────────────────────────────────────────────────
def load_nav(code6: str) -> dict[str, tuple[float, float]]:
    p = os.path.join(NAV_DIR, f"{code6}.json")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in json.load(open(p, encoding="utf-8")):
        try:
            out[r["date"]] = (float(r["nav"]), float(r["cumnav"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ── F10 分红送配页（权威事件源）────────────────────────────────────────
def _f10_fetch(code6: str) -> dict[str, Any] | None:
    """抓取并解析 F10 分红送配页。返回 None 表示抓取失败。"""
    req = urllib.request.Request(
        F10_URL.format(code6=code6),
        headers={"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/120.0 Safari/537.36"),
                 "Referer": "https://fundf10.eastmoney.com/"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            html = resp.read().decode("utf-8", "replace")
    except Exception:                                    # noqa: BLE001
        return None
    return _f10_parse(html)


def _f10_parse(html: str) -> dict[str, Any]:
    """解析 F10 页面里的分红明细表与份额折算表。"""
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.S)
    divs: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    mode: str | None = None
    for row in rows:
        cells = []
        for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S):
            text = re.sub(r"<[^>]+>", "", cell)
            cells.append(re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip())
        if not any(cells):
            continue
        if cells[0] == "年份" and len(cells) > 1:
            head = "".join(cells)
            if "分红" in head:
                mode = "div"
                continue
            if "拆分" in head or "折算" in head:
                mode = "split"
                continue
        if mode == "div" and re.fullmatch(r"\d{4}年", cells[0]) and len(cells) >= 4:
            m = re.search(r"([\d.]+)元", "".join(cells))
            per10 = float(m.group(1)) if m else None
            divs.append({"reg_date": cells[1], "ex_date": cells[2],
                         "per10": per10,
                         "per_share": (per10 / 10.0) if per10 is not None else None})
        elif mode == "split" and re.fullmatch(r"\d{4}年", cells[0]) and len(cells) >= 4:
            ratio = cells[3]
            m = re.match(r"^1\s*[:：]\s*([\d.]+)$", ratio.replace(" ", ""))
            if m:
                splits.append({"date": cells[1], "type": cells[2], "X": float(m.group(1))})
    return {"divs": divs, "splits": splits}


def refresh_f10_actions(codes: list[str], pause: float = 0.35) -> dict[str, Any]:
    """抓取 F10 并写入缓存（低频、带间隔，不并发轰炸）。"""
    cache: dict[str, Any] = {}
    if os.path.exists(F10_PATH):
        try:
            cache = json.load(open(F10_PATH, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    for i, code in enumerate(codes, 1):
        got = _f10_fetch(code)
        if got is None:
            print(f"[F10] {code} 抓取失败，保留旧缓存", flush=True)
            continue
        prev = cache.get(code, {})
        cache[code] = {"divs": got["divs"], "splits": got["splits"],
                       "name": prev.get("name", "")}
        print(f"[F10] {i}/{len(codes)} {code} divs={len(got['divs'])} "
              f"splits={len(got['splits'])}", flush=True)
        time.sleep(pause)
    os.makedirs(os.path.dirname(F10_PATH), exist_ok=True)
    json.dump(cache, open(F10_PATH, "w"), ensure_ascii=False, indent=1)
    return cache


def _norm_split(sp: dict[str, Any]) -> dict[str, Any] | None:
    """把折算记录归一化成 {'date','type','X'}。

    F10 页面与缓存里有两种写法：
      - {'date': '2022-08-26', 'type': '份额分拆', 'X': 1.1454}      （本模块解析器）
      - {'date': '2022-08-26', 'type': '份额分拆', 'ratio': '1:1.1454'}（交叉校验缓存）
    比例语义统一为 **X = 折算后份额 / 折算前份额**，故价格要除以 X。
    """
    d = sp.get("date") or sp.get("split_date")
    if not d:
        return None
    x = sp.get("X")
    if x is None:
        raw = sp.get("ratio") or sp.get("ratio_raw") or ""
        m = re.match(r"^\s*1\s*[:：]\s*([\d.]+)\s*$", str(raw).replace(" ", ""))
        if not m:
            return None
        x = float(m.group(1))
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not (x > 0):
        return None
    return {"date": d, "type": sp.get("type") or sp.get("split_type") or "",
            "X": x}


def _norm_div(dv: dict[str, Any]) -> dict[str, Any] | None:
    """把分红记录归一化成 {'ex_date','per_share'}（每股口径，元/份）。"""
    ex = dv.get("ex_date") or dv.get("ex_div_date")
    if not ex:
        return None
    amt = dv.get("per_share")
    if amt is None:
        amt = dv.get("amount_per_share")
    if amt is None and dv.get("per10") is not None:
        amt = float(dv["per10"]) / 10.0
    if amt is None:
        return None
    try:
        amt = float(amt)
    except (TypeError, ValueError):
        return None
    return {"ex_date": ex, "per_share": amt}


def load_f10_actions(code6: str) -> dict[str, Any]:
    """读取 F10 缓存里的分红/折算记录。缓存缺失返回空 dict。

    缓存文件由不同工具写入过（分红双源校验 / 本模块解析器），
    字段名不统一 —— 这里统一归一化，避免 KeyError 直接崩掉整轮重建。
    """
    if not os.path.exists(F10_PATH):
        return {"divs": [], "splits": []}
    try:
        cache = json.load(open(F10_PATH, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"divs": [], "splits": []}
    entry = cache.get(code6) or {}
    divs = [d for d in (_norm_div(x) for x in (entry.get("divs") or [])) if d]
    splits = [s for s in (_norm_split(x) for x in (entry.get("splits") or [])) if s]
    return {"divs": divs, "splits": splits}


# ── k（份额倍数）估计 ──────────────────────────────────────────────────
# 判定「候选边界是真折算还是只是分红」的阈值：
# 局部 k 比值 k_right/k_left 超过它 → 认为是份额折算（k 真的变了）；
# 否则认为只是分红（C 连续、D 跳变，斜率不变）。
SPLIT_K_RATIO = 1.10


def _local_k_ratio(u: np.ndarray, c: np.ndarray, b: int,
                   win: int = 60) -> float | None:
    """候选边界 b 两侧的局部 k 之比（k_right / k_left），判不出返回 None。

    边界那一对差分 (b-1, b) 被**两侧窗口都排除**，所以：
      * 真折算 → k_right / k_left = X（份额折算比例，>1）
      * 真分红 → k_right / k_left ≈ 1（C 连续、D 跳变，斜率不变）

    这个区分是必需的：在 (U, C) 序列里，「份额折算」和「大额分红」长得
    **一模一样** —— 都是 ΔC=0、ΔU<0。唯一能分开它们的是 **k 有没有变**
    （折算改斜率，分红只改截距）。
    """
    lo = max(0, b - win)
    hi = min(len(u), b + win)
    if b - lo < 30 or hi - b < 30:
        return None
    kl, _ = _fit_k_segment(u[lo:b], c[lo:b])
    kr, _ = _fit_k_segment(u[b:hi], c[b:hi])
    if not (np.isfinite(kl) and np.isfinite(kr)) or kl <= 0 or kr <= 0:
        return None
    return float(kr / kl)


def detect_regimes(ratio: np.ndarray) -> list[int]:
    """返回份额折算事件的下标（C/U 跳变 > 8%）。"""
    idx = []
    for i in range(1, len(ratio)):
        if ratio[i - 1] > 0 and abs(ratio[i] / ratio[i - 1] - 1.0) > REGIME_JUMP:
            idx.append(i)
    return idx


def _fit_k_segment(u: np.ndarray, c: np.ndarray) -> tuple[float, np.ndarray]:
    """在单个 regime 段内稳健拟合 C = k·U + D，返回 (k, 残差序列)。

    残差 = ΔC − k·ΔU = k · (每日分红 d_true)。
    注意：**每日分红（每股口径）= 残差 / k**，不是残差本身。

    为什么用**整段回归**而不是滚动中位数：C/U 会随 U 缓慢漂移，
    250 日滚动中位数滞后约 125 天，对一个 1.4e-4/日的漂移就产生 δ≈0.03 的
    k 误差 → 残差噪声被放大到 40bp，完全淹没真实除息信号。
    整段 OLS 对线性漂移是无偏的。

    ⚠️ 除息日是**强离群点**：除息日 C 连续、U 掉 d，所以
    ΔC=0 而 ΔU=−d，(du,dc) 这一点偏离直线 dc=k·du 很远。
    如果用 OLS 做**初值**，这一个点会把 k 带偏（实测能偏到真实值的 1/14），
    接着残差筛选又把正常日子全部误杀 → 整段 k 崩掉、假除息日暴增。
    所以初值改用**比值中位数** dc/du（除息日该比值≈0，是中位数的天然离群点），
    初值稳健后再用 OLS 迭代精修 —— 精修这一步保留了「对线性漂移无偏」的性质。
    """
    du = np.diff(u)
    dc = np.diff(c)
    if len(du) == 0:
        return 1.0, np.array([])
    mask = np.abs(du) > 0.002
    if mask.sum() < 30:
        mask = np.ones(len(du), dtype=bool)
    # 稳健初值：比值中位数（对除息日这类离群点不敏感）
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(mask, dc / np.where(du != 0, du, np.nan), np.nan)
    ratios = ratios[np.isfinite(ratios)]
    k = float(np.median(ratios)) if ratios.size else 1.0
    if not np.isfinite(k) or k <= 0:
        k = 1.0
    resid = dc - k * du
    # 迭代：先用当前 k 筛掉离群点（含除息日），再在干净集合上做 OLS 精修。
    # 顺序不能反 —— 先 OLS 再筛的话，除息日这个强离群点已经污染了 k，
    # 筛出来的集合是错的（实测整段只剩 0 个有效点）。
    for _ in range(6):
        rel = np.abs(resid) / np.maximum(u[:-1], 1e-9)
        clean = mask & (rel < 0.002)
        if clean.sum() < 30:
            break
        den = float(np.sum(du[clean] ** 2))
        if den <= 0:
            break
        k_new = float(np.sum(dc[clean] * du[clean]) / den)
        if not np.isfinite(k_new) or k_new <= 0:
            break
        if abs(k_new - k) <= 1e-12 * max(1.0, abs(k)):
            k = k_new
            resid = dc - k * du
            break
        k = k_new
        resid = dc - k * du
    return k, resid


def _segments(dates: list[str], folds: list[str] | None) -> tuple[list[int], str]:
    """给定折算日（date 字符串列表）→ 段边界下标。

    折算日**不一定**出现在 dates 里 —— dates 是「本地行情有 K 线的日期」，
    而 F10 折算生效日有时正好落在停牌/未上市/数据缺失的日子
    （实测 sh512010/512100/512170/512480/512690/513100/513500/sz159919
    这 8 只的生效日本地都没有 bar）。
    所以这里**不能做精确字符串匹配**，必须映射到
    「第一个 >= 折算日的日期」，否则折算永远切不出段边界。
    """
    if not folds:
        return [], "nav"
    bounds = [0]
    for f in sorted(set(folds)):
        idx = None
        for i, d in enumerate(dates):
            if d >= f:
                idx = i
                break
        if idx is not None and idx > 0:
            bounds.append(idx)
    bounds.append(len(dates))
    return sorted(set(bounds)), "f10"


def estimate_k(dates: list[str], nav: dict[str, tuple[float, float]],
               folds: list[str] | None = None,
               ) -> tuple[np.ndarray, list[int], np.ndarray]:
    """返回 (逐日 k 序列, 折算事件下标, 逐日分红 d_true)。

    分段：份额折算日切段（优先用 F10 折算日；没有就用 C/U 跳变兜底），
    每段独立拟合 k。残差 ΔC − k·ΔU 再**除以 k** 得到每股口径分红。

    ⚠️ 切分点必须真正用作**段边界**。如果只把它当「事件列表」记一下、
    却仍然整段一起拟合 k，那么跨越折算的那一段会把两个不同 k 的 regime
    混在一起拟合，残差在全段都是垃圾 → 净值反解出的「除息日」从真实的
    ~1 次/年 暴涨到几百次（实测 586 次），复权序列直接被污染。
    """
    u = np.array([nav[d][0] for d in dates], dtype=float)
    c = np.array([nav[d][1] for d in dates], dtype=float)
    ok = (u > 0) & (c > 0)
    ratio = np.where(ok, c / np.where(ok, u, 1.0), np.nan)
    bounds, _src = _segments(dates, folds)
    # F10 折算记录**不完整**：F10 主要记「份额分拆」，会漏掉「份额折算/合并」。
    # 实测 sh512690：F10 只有 2021-05-14 (X=2.0)，但净值 C/U 在 2021-12-31
    # 又从 2.0 跳到 2.73（真实份额折算，F10 没记）——只信 F10 边界的话，
    # 这次折算会被当成 36.6% 的假分红，div_gap 爆表 → 致命失败。
    # 所以边界 = F10 ∪ C/U 跳变检测。
    #
    # ⚠️ 但跳变点**还要再判一次是不是分红**：在 (U, C) 序列里折算和分红
    # 长得一模一样，唯一区别是 k 变没变（见 _local_k_ratio）。
    # 把分红日当边界切段 → 该日的分红残差落在段外被**丢掉** → 价格少复权。
    f10_set = set(bounds)
    extra: list[int] = []
    for b in detect_regimes(ratio):
        if b in f10_set or not (0 < b < len(dates)):
            continue
        r = _local_k_ratio(u, c, b)
        if r is None:
            # 判不出（样本不足）→ 看跳变幅度：>20% 只可能是折算
            # （分红不会让 C/U 跳 20%，那需要 20% 的现金分红）
            if (np.isfinite(ratio[b]) and np.isfinite(ratio[b - 1])
                    and ratio[b - 1] > 0
                    and abs(ratio[b] / ratio[b - 1] - 1.0) > 0.20):
                extra.append(b)
        elif r > SPLIT_K_RATIO:
            extra.append(b)
    bounds = sorted(set([0, len(dates)] + list(bounds) + extra))
    # 丢弃会让某一段短到无法拟合的切分点（< MIN_SEG_POINTS 个样本）
    bounds = [b for b in bounds if 0 <= b <= len(dates)]
    kept = [bounds[0]]
    for b in bounds[1:-1]:
        if b - kept[-1] >= MIN_SEG_POINTS and len(dates) - b >= MIN_SEG_POINTS:
            kept.append(b)
    kept.append(len(dates))
    bounds = sorted(set(kept))
    if len(bounds) < 2:
        bounds = [0, len(dates)]
    regimes = list(bounds[1:-1])
    k_arr = np.ones(len(dates))
    resid_arr = np.zeros(len(dates))          # ΔC − k·ΔU（k 加权口径）
    for bi in range(len(bounds) - 1):
        lo, hi = bounds[bi], bounds[bi + 1]
        if hi - lo < 30:
            k_arr[lo:hi] = (np.nanmedian(ratio[lo:hi])
                            if np.any(~np.isnan(ratio[lo:hi])) else 1.0)
            continue
        k, resid = _fit_k_segment(u[lo:hi], c[lo:hi])
        k_arr[lo:hi] = k
        resid_arr[lo + 1:hi] = resid           # resid[i] 对应 Δ 在 i→i+1
    with np.errstate(divide="ignore", invalid="ignore"):
        d_true = np.where(k_arr > 0, resid_arr / k_arr, 0.0)
    return k_arr, regimes, d_true


# ── 事件定位 ───────────────────────────────────────────────────────────
def _locate_price_jump(pdates: list[str], pclose: list[float],
                       start_date: str, X: float) -> dict[str, Any] | None:
    """在折算日之后定位**价格序列里真正跳变的那一天**。

    判据：跳变日满足 close(t)/close(t−1) ≈ 1/X，即隐含真实日收益
    (ratio·X − 1) 落在 ±25% 内（ETF 限涨 ±10~20%）；非跳变日 ratio ≈ 1，
    隐含收益 ≈ X−1（对 X=2 是 +100%），两者相差一个数量级。
    """
    idx0 = None
    for i, d in enumerate(pdates):
        if d >= start_date:
            idx0 = i
            break
    if idx0 is None:
        return None
    best: dict[str, Any] | None = None
    for j in range(idx0, min(idx0 + FOLD_WINDOW + 1, len(pdates))):
        if j <= 0:
            continue
        ratio = pclose[j] / pclose[j - 1]
        implied = ratio * X - 1.0
        score = abs(implied)
        if best is None or score < best["score"]:
            best = {"date": pdates[j], "index": j, "price_ratio": ratio,
                    "implied_ret": implied, "score": score,
                    "prev_date": pdates[j - 1], "prev_close": pclose[j - 1],
                    "close": pclose[j]}
    if best is None:
        return None
    if abs(best["price_ratio"] - 1.0) < FOLD_JUMP_MIN:
        return None                                  # 没有真正的价格跳变
    if abs(best["implied_ret"]) > FOLD_IMPLIED_MAX:
        return None                                  # 比例对不上，不认
    return best


def _first_price_date_on_or_after(pdates: list[str], d: str) -> int | None:
    for i, x in enumerate(pdates):
        if x >= d:
            return i
    return None


def collect_events(rows: list[dict[str, Any]], nav: dict[str, tuple[float, float]],
                   code6: str | None,
                   actions: dict[str, Any] | None = None,
                   ) -> dict[str, Any]:
    """收集复权事件：份额折算（比例 + 价格跳变日）与现金分红（日期 + 每股金额）。

    F10 为主，NAV 代数兜底并交叉校验。
    """
    ds = [r["date"] for r in rows]
    pclose = [float(r["close"]) for r in rows]
    have = [d for d in ds if d in nav]
    f10 = actions if actions is not None else (load_f10_actions(code6) if code6 else
                                              {"divs": [], "splits": []})
    f10_splits = sorted(f10.get("splits") or [], key=lambda s: s["date"])
    f10_divs = sorted(f10.get("divs") or [], key=lambda s: s.get("ex_date") or "")

    # NAV 侧的 regime（兜底 + 交叉校验）
    nav_k, nav_regimes, d_true = estimate_k(have, nav,
                                             folds=[s["date"] for s in f10_splits]
                                             if f10_splits else None)
    kmap = dict(zip(have, nav_k))
    dmap = dict(zip(have, d_true))

    # ── 份额折算事件 ──
    folds: list[dict[str, Any]] = []
    used_ratio: set[str] = set()
    for sp in f10_splits:
        X = float(sp["X"])
        jump = _locate_price_jump(ds, pclose, sp["date"], X)
        folds.append({"nav_date": sp["date"], "X": X, "type": sp.get("type", ""),
                      "price_date": jump["date"] if jump else None,
                      "price_index": jump["index"] if jump else None,
                      "price_ratio": jump["price_ratio"] if jump else None,
                      "implied_ret": jump["implied_ret"] if jump else None,
                      "nav_ratio": (nav_k[have.index(sp["date"]) + 1]
                                    / nav_k[have.index(sp["date"])]
                                    if sp["date"] in have
                                    and have.index(sp["date"]) + 1 < len(have) else None),
                      "source": "f10"})
        if sp["date"] in used_ratio:
            continue
        used_ratio.add(sp["date"])

    if not folds:                                   # F10 缺失 → NAV 兜底
        u = np.array([nav[d][0] for d in have], dtype=float)
        c = np.array([nav[d][1] for d in have], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(u > 0, c / np.where(u > 0, u, 1.0), np.nan)
        for i in detect_regimes(ratio):
            d0, d1 = have[i - 1], have[i]
            X = float(nav[d0][1] / nav[d0][0]) / float(nav[d1][1] / nav[d1][0])
            if not np.isfinite(X) or X <= 0:
                continue
            jump = _locate_price_jump(ds, pclose, d0, X)
            folds.append({"nav_date": d0, "X": X, "type": "nav",
                          "price_date": jump["date"] if jump else None,
                          "price_index": jump["index"] if jump else None,
                          "price_ratio": jump["price_ratio"] if jump else None,
                          "implied_ret": jump["implied_ret"] if jump else None,
                          "nav_ratio": X, "source": "nav_fallback"})

    # ── 分红事件 ──
    divs: list[dict[str, Any]] = []
    for dv in f10_divs:
        ex = dv.get("ex_date")
        amt = dv.get("per_share")
        if not ex or amt is None:
            continue
        idx = _first_price_date_on_or_after(ds, ex)
        if idx is None:
            continue
        nav_amt = dmap.get(ex)
        k_here = kmap.get(ex)
        tiny_k = (k_here is not None and 0 < k_here < TINY_K)
        ratio = (nav_amt / amt) if (nav_amt and amt and nav_amt > 0) else None
        if tiny_k or nav_amt is None or nav_amt <= 0 or ratio is None or abs(ratio - 1.0) > DIV_RATIO_TOL:
            use_amt, src = amt, ("f10_tiny_k" if tiny_k else "f10_override")
        else:
            use_amt, src = float(nav_amt), "nav"
        divs.append({"ex_date": ex, "price_date": ds[idx], "price_index": idx,
                     "f10_per_share": amt, "nav_amt": nav_amt,
                     "ratio": ratio, "amount": use_amt, "source": src})

    # NAV 反解到、但 F10 没有登记的分红（兜底）
    f10_ex = {d["ex_date"] for d in divs}
    for d in have:
        if d in f10_ex:
            continue
        amt = dmap.get(d, 0.0)
        if amt <= 0:
            continue
        u_prev = nav.get(d)
        if not u_prev or u_prev[0] <= 0:
            continue
        if amt / u_prev[0] <= DIV_REL_THRESH:
            continue
        idx = _first_price_date_on_or_after(ds, d)
        if idx is None:
            continue
        divs.append({"ex_date": d, "price_date": ds[idx], "price_index": idx,
                     "f10_per_share": None, "nav_amt": amt, "ratio": None,
                     "amount": amt, "source": "nav_only"})

    return {"folds": folds, "divs": divs, "k": nav_k, "d_true": dmap,
            "have": have, "f10_source": "f10" if f10_splits else "nav_fallback"}


# ── 复权 ────────────────────────────────────────────────────────────────
def adjust_series(rows: list[dict[str, Any]], nav: dict[str, tuple[float, float]],
                  code6: str | None = None,
                  actions: dict[str, Any] | None = None,
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """返回 (复权后的行, 诊断信息)。"""
    if not rows or not nav:
        return rows, {"status": "no_nav", "n_exdiv": 0}

    ds = [r["date"] for r in rows]
    ev = collect_events(rows, nav, code6, actions)
    fold_by_index: dict[int, float] = {}
    for f in ev["folds"]:
        if f.get("price_index") is not None:
            fold_by_index[int(f["price_index"])] = float(f["X"])
    div_by_index: dict[int, float] = {}
    for d in ev["divs"]:
        if d.get("price_index") is not None:
            div_by_index[int(d["price_index"])] = float(d["amount"])

    out: list[dict[str, Any]] = []
    factor = 1.0
    prev_close: float | None = None
    n_exdiv = 0
    div_events: list[dict[str, Any]] = []
    nav_missing = 0
    for i, r in enumerate(rows):
        close = float(r["close"])
        if prev_close is None or prev_close <= 0:
            rr = dict(r)
            rr["adj_factor"] = factor
            rr["adj_close"] = close * factor
            rr["adj_open"] = float(r["open"]) * factor
            out.append(rr)
            prev_close = close
            continue
        r_price = close / prev_close - 1.0
        X = fold_by_index.get(i, 1.0)
        div = div_by_index.get(i, 0.0)
        # r_adj = (close + div)·X / close_prev − 1
        r_adj = (close + div) * X / prev_close - 1.0
        if div > 0:
            n_exdiv += 1
            div_events.append({"date": r["date"], "div": round(div, 6),
                               "r_price": round(r_price, 5), "r_adj": round(r_adj, 5),
                               "X": X})
        if (1.0 + r_price) > 0:
            factor *= (1.0 + r_adj) / (1.0 + r_price)
        rr = dict(r)
        rr["adj_factor"] = factor
        rr["adj_close"] = close * factor
        rr["adj_open"] = float(r["open"]) * factor
        out.append(rr)
        prev_close = close
        if r["date"] not in nav:
            nav_missing += 1

    years = max(1e-9, (len(rows) / 243.0))
    k_arr = ev["k"]
    k_first = float(k_arr[0]) if len(k_arr) else None
    k_last = float(k_arr[-1]) if len(k_arr) else None
    k_ratio = (k_last / k_first) if (k_first and k_last and k_first > 0) else None
    diag = {
        "status": "ok",
        "n_exdiv": n_exdiv,
        "exdiv_per_year": round(n_exdiv / years, 3),
        "n_regime_events": len(ev["folds"]),
        "regime_dates": [f["price_date"] for f in ev["folds"] if f.get("price_date")],
        "fold_events": [{k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in f.items() if k != "price_index"} for f in ev["folds"]],
        "div_events": div_events[-30:],
        "div_crosscheck": [{"date": d["ex_date"], "f10": d["f10_per_share"],
                            "nav": d["nav_amt"],
                            "ratio": (round(d["ratio"], 4) if d["ratio"] else None),
                            "source": d["source"]} for d in ev["divs"]],
        "nav_missing_days": nav_missing,
        "total_factor": factor,
        "k_first": round(k_first, 6) if k_first else None,
        "k_last": round(k_last, 6) if k_last else None,
        "k_ratio": round(k_ratio, 6) if k_ratio else None,
        "event_source": ev["f10_source"],
    }
    return out, diag


def nav_return_series(rows: list[dict[str, Any]], nav: dict[str, tuple[float, float]],
                      ) -> np.ndarray:
    """净值**总收益**序列（C 累计净值的日变化率），用于校验复权结果。

    为什么用 C 而不是 U：
      - C 在**份额折算日连续**（U 会跳 X 倍），折算日的假跳空不会污染相关性；
      - C 的日涨跌幅 = 基金总收益率（含分红再投资），正是复权序列要对齐的目标。
    U 的日变化率会把折算跳空当成 −80% 的行情，任何相关性指标都会被单点极值毁掉。
    """
    rets = []
    prev = None
    for r in rows:
        nv = nav.get(r["date"])
        if nv is None:
            continue
        c = nv[1]
        if prev is not None and prev > 0:
            rets.append(c / prev - 1.0)
        prev = c
    return np.array(rets, dtype=float)


def build(force: bool = False) -> dict[str, Any]:
    os.makedirs(ADJ_PATH, exist_ok=True)
    files = sorted(f for f in os.listdir(ETF_DIR) if f.endswith(".json"))
    diags: dict[str, Any] = {}
    written = 0
    for fn in files:
        sym = fn[:-5]
        code6 = sym[2:]
        outp = os.path.join(ADJ_PATH, fn)
        if os.path.exists(outp) and not force:
            diags[sym] = {"status": "cached"}
            written += 1
            continue
        rows = json.load(open(os.path.join(ETF_DIR, fn), encoding="utf-8"))
        nav = load_nav(code6)
        adj, diag = adjust_series(rows, nav, code6=code6)
        a_ret = np.array([adj[i]["adj_close"] / adj[i - 1]["adj_close"] - 1.0
                          for i in range(1, len(adj))], dtype=float)
        n_ret = nav_return_series(rows, nav)
        m = min(len(a_ret), len(n_ret))
        corr = float(np.corrcoef(a_ret[-m:], n_ret[-m:])[0, 1]) if m > 30 else float("nan")
        diag["adj_vs_nav_ret_corr"] = corr
        diag["n_bars"] = len(adj)
        diag["n_extreme_neg"] = int(np.sum(a_ret < -0.08)) if len(a_ret) else 0
        diag["max_adj_factor"] = float(max(r.get("adj_factor", 1.0) for r in adj))
        diag["max_abs_adj_ret"] = (float(np.max(np.abs(a_ret))) if len(a_ret) else None)
        json.dump(adj, open(outp, "w"), ensure_ascii=False)
        diags[sym] = diag
        written += 1
        print(f"[ADJ] {sym} bars={len(adj)} exdiv={diag['n_exdiv']}({diag['exdiv_per_year']}/y) "
              f"fold={diag['n_regime_events']} corr={corr:.4f} "
              f"factor={diag['total_factor']:.4f} maxF={diag['max_adj_factor']:.4f} "
              f"neg8={diag['n_extreme_neg']}", flush=True)
    os.makedirs(os.path.join(REPO, "data", "research"), exist_ok=True)
    json.dump(diags, open(os.path.join(REPO, "data", "research", "etf_adjustment.json"),
                          "w"), ensure_ascii=False, indent=1)
    print(f"=== 复权完成 {written} 只 → {ADJ_PATH} ===")
    return diags


if __name__ == "__main__":
    import sys
    if "--refresh-f10" in sys.argv:
        codes = sorted(f[2:-5] for f in os.listdir(ETF_DIR) if f.endswith(".json"))
        refresh_f10_actions(codes)
    build(force=True)
