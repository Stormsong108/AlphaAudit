#!/usr/bin/env python3
"""每日 live 闭环：增量刷新行情 → 生成 live 预测 → 到期后独立判定。

三个阶段各自独立、各自落盘、各自留痕：
    refresh  增量刷新后复权日线（东财为主源、腾讯为交叉校验源；只追加，不改写历史）
    predict  生成当天的 live 预测（预测落盘即不可改写，已存在就跳过）
    judge    用真实行情判定到期的预测（结果落独立文件，预测文件永不改写）

铁律（踩过的坑都在这里）：
1. 上游不可用 → 失败关闭。宁可保留旧数据，也不用"看起来新鲜"的数据凑数。
2. 只追加不覆盖：新 bar 的日期必须严格大于本地末根日期，历史段一个字节都不改。
3. 重叠段必须与本地逐日一致（容差内），不一致就拒绝写入 —— 说明锚点或口径变了。
4. 新 bar 必须通过第二源交叉校验：与腾讯后复权的比值必须落在历史比值带内。
   两家后复权锚点不同（实测比值约 1.006 的常数），比值漂移就是污染信号。
5. 腾讯只能校验、不能直接写入：它的锚点与东财不同，直接拼接会在边界制造
   约 0.6% 的价格跳变，直接毁掉收益率。
6. 全程零 LLM 调用，固定随机种子，可重复。

用法：
    python daily_loop.py --stage refresh --dry-run     # 只抓只验不写
    python daily_loop.py --stage refresh --limit 20     # 先在少量标的上试
    python daily_loop.py --stage predict
    python daily_loop.py --stage judge
    python daily_loop.py --stage all
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from collectors.base import http_get_json  # noqa: E402

CST = timezone(timedelta(hours=8))
STATE = ROOT / "state" / "daily_loop.json"
OUTDIR = ROOT / "data" / "daily_loop"
UNIVERSE = ROOT / "data" / "universe.json"
HIST_DIR = ROOT / "data" / "history" / "kline"

EM_HOST = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TX_HOST = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
# 腾讯同一套接口有多个域名：主域名被 WAF 挡（HTTP 501）时换备用域名，
# 返回结构一致。任何一个可用即可 —— 拿不到数据才叫失败。
TX_HOSTS = (
    TX_HOST,
    "https://ifzq.gtimg.cn/appstock/app/fqkline/get",
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
)

# 校验容差（相对）
OVERLAP_TOL = 1e-6      # 重叠段与本地必须逐日一致（后复权价是精确值）
RATIO_TOL = 0.002       # 新 bar 与第二源的比值，必须落在历史比值带 ±0.2% 内
MIN_COVERAGE = 0.5      # 低于这个刷新覆盖率，整阶段判失败（但已验证的写入保留）
TX_COUNT = 800          # 腾讯单次最多约 800 根（实测 1200+ 反而只给 640）
OVERLAP_DAYS = 60       # 向东财多要 60 天做重叠校验
WORKERS = 8             # 并发抓取线程数

# 腾讯回退路径（东财网络层不可用时）的额外阈值 —— 与上面同量级，只收紧不放宽
TX_FALLBACK_MIN_N = 20          # 历史比值带的最少重叠天数
TX_FALLBACK_MAX_SPREAD = 0.01   # 历史比值带的最大离散度
TX_FALLBACK_VOL_OUTLIER = 0.10  # 成交量单位比允许的异常日占比上限
TX_FALLBACK_AMT_TOL = 0.10      # 成交额估算回测允许的中位相对偏差
TX_FALLBACK_AMT_TEST_DAYS = 21  # 成交额估算器回测用多少个重叠日
TX_FALLBACK_AMT_MIN_SAMPLES = 10  # 回测至少要有多少个有效样本


# ------------------------------------------------------------------ 工具


def now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now(CST).strftime("%Y%m%d_%H%M%S")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"stages": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    st["updated_at"] = now_iso()
    # 临时文件名带 pid：research_queue 与日报可能同时写这个状态文件，
    # 共用同一个 .tmp 名会互相抢文件（一方 replace 时另一方已删）。
    tmp = STATE.with_name(f"{STATE.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def write_result(stage: str, payload: dict) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # 文件名带微秒：同一秒内跑同一阶段（队列 + 日报）不会互相覆盖结果文件。
    stamp = datetime.now(CST).strftime("%Y%m%d_%H%M%S_%f")
    p = OUTDIR / f"{stage}_{stamp}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def secid_safe(secid: str) -> str:
    return secid.replace(".", "_")


def load_universe() -> dict[str, str]:
    try:
        data = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def to_tx_code(secid: str) -> str:
    """东财 secid → 腾讯代码。非 A 股返回空串。"""
    if "." not in secid:
        return ""
    market, code = secid.split(".", 1)
    if market == "1":
        return "sh" + code
    if market == "0":
        if code[:1] in ("4", "8"):
            return "bj" + code
        return "sz" + code
    return ""


# ------------------------------------------------------------------ 抓取


def em_fetch(secid: str, beg: str) -> list[dict]:
    """东财后复权日线。失败抛异常（调用方负责降级）。"""
    url = (
        f"{EM_HOST}?secid={secid}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57"
        f"&klt=101&fqt=2&beg={beg}&end=20500101"
    )
    f = http_get_json(url, referer="https://quote.eastmoney.com/", timeout=25, retries=2)
    if not f.ok:
        raise RuntimeError(f"东财抓取失败 {secid}: {f.error}")
    data = f.value.get("data") or {}
    rows: list[dict] = []
    for k in data.get("klines") or []:
        p = k.split(",")
        if len(p) < 7:
            continue
        try:
            o, c, h, l = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        if not (o > 0 and c > 0 and h > 0 and l > 0):
            continue          # 非正价格一律不收：坏数据比没数据危险
        rows.append({
            "date": p[0], "open": o, "close": c, "high": h, "low": l,
            "volume": float(p[5]), "amount": float(p[6]),
        })
    return rows


def tx_fetch(tx_code: str, count: int = TX_COUNT) -> list[dict]:
    """腾讯后复权日线（交叉校验源 / 东财不可用时的回退数据源）。

    域名逐个试：主域名被 WAF 挡（HTTP 501）时换备用域名，返回结构一致。
    """
    errs: list[str] = []
    f = None
    for host in TX_HOSTS:
        url = f"{host}?param={tx_code},day,,,{count},hfq"
        f = http_get_json(url, referer="https://gu.qq.com/", timeout=25, retries=1)
        if f.ok:
            break
        errs.append(f"{host}: {f.error}")
    if f is None or not f.ok:
        raise RuntimeError(f"腾讯抓取失败 {tx_code}: {'; '.join(errs)}")
    bars = _tx_pick(f.value)
    rows: list[dict] = []
    for b in bars:
        if len(b) < 5:
            continue
        try:
            o, c, h, l = float(b[1]), float(b[2]), float(b[3]), float(b[4])
        except (ValueError, TypeError):
            continue
        if not (o > 0 and c > 0 and h > 0 and l > 0):
            continue
        rows.append({
            "date": str(b[0]), "open": o, "close": c, "high": h, "low": l,
            "volume": float(b[5]) if len(b) > 5 else 0.0, "amount": 0.0,
        })
    return rows


def _tx_pick(node: Any) -> list:
    """腾讯返回结构有 dict / list 两种形态，这里统一取出 bars 数组。"""
    if isinstance(node, list):
        for it in node:
            b = _tx_pick(it)
            if b:
                return b
        return []
    if not isinstance(node, dict):
        return []
    for k in ("hfqday", "qfqday", "day"):
        v = node.get(k)
        if isinstance(v, list) and v and isinstance(v[0], list):
            return v
    for v in node.values():
        b = _tx_pick(v)
        if b:
            return b
    return []


# ------------------------------------------------------------------ 校验


def _overlap_ratio(local: dict[str, float], other: dict[str, float]) -> tuple[float, float, int]:
    """两条序列在重叠区间的比值均值与最大离散度，以及重叠天数。"""
    common = sorted(set(local) & set(other))
    if not common:
        return 0.0, 1.0, 0
    ratios = [local[d] / other[d] for d in common if other[d]]
    if not ratios:
        return 0.0, 1.0, 0
    mean = sum(ratios) / len(ratios)
    spread = (max(ratios) - min(ratios)) / mean if mean else 1.0
    return mean, spread, len(common)


def _fail_tencent(out: dict[str, Any], stage: str, reason: str, **extra: Any) -> dict[str, Any]:
    """腾讯回退路径的统一失败出口：不返回任何新行，绝不写入。

    刻意**不**设置 out["new_rows"]：失败时调用方拿不到任何行可写，
    避免将来有人把脏数据 append 进历史。
    """
    out["ok"] = False
    out["stage"] = stage
    out["reason"] = reason
    # 腾讯路径里腾讯就是主源，没有第三个源可比 —— 交叉校验是"没做"，不是"通过"。
    out["stats"] = {"cross_source": "skipped", **extra}
    return out


def _local_complete_flags(local_rows: list[dict], min_share: float = 0.5) -> list[bool]:
    """标记本地历史里哪些 bar 是"完整交易日"（不是盘中抓的半截 bar）。

    盘中快照的成交量只有正常日的一小截 —— 用它算锚点比值会得到一个假的
    偏离，把好标的误判成"口径漂移"。判据：成交量 ≥ 前 20 日成交量中位数的
    min_share 倍。
    """
    vols = [float(r.get("volume") or 0.0) for r in local_rows]
    flags: list[bool] = []
    for i in range(len(local_rows)):
        prior = [v for v in vols[max(0, i - 20):i] if v > 0]
        if not prior or vols[i] <= 0:
            flags.append(False)
        else:
            flags.append(vols[i] >= min_share * statistics.median(prior))
    return flags


def _pick_amount_ref(local_rows: list[dict], tx_vol: dict[str, float],
                     vol_scale: float, tol: float = 0.2) -> dict | None:
    """挑一根可作成交额估算基准的本地 bar。

    基准必须是**完整交易日**：盘中抓的半截 bar（收盘价是盘中价、成交量只有
    正常日的一小截）拿来当基准，会把新 bar 的成交额整体缩掉一个量级。

    判据用**跨源成交量比对**而不是"跟自己的历史比"：本地那天的量与腾讯
    同一天的量之比，应当落在标定出的单位比 vol_scale 附近；偏离超过 tol
    说明那天是残缺 bar。自比（跟前 20 日中位数比）会误伤天生低成交的标的。
    """
    for r in reversed(local_rows):
        d = r.get("date")
        v = float(r.get("volume") or 0.0)
        a = float(r.get("amount") or 0.0)
        if not (a > 0 and v > 0 and d in tx_vol and tx_vol[d] > 0):
            continue
        if abs(v / (tx_vol[d] * vol_scale) - 1.0) <= tol:
            return r
    return None


def _typ(row: dict) -> float:
    """典型价 (high+low+close)/3 —— 比收盘价更接近当日成交均价。

    缺 high/low 时退回收盘价（估算退化但不算错）。
    """
    c = float(row.get("close") or 0.0)
    h = float(row.get("high") or c)
    l = float(row.get("low") or c)
    return (h + l + c) / 3.0


def _amount_estimate(ref_amount: float, ref_vol: float, ref_typ: float,
                     new_vol: float, new_typ: float) -> float:
    """成交额估算：基准 bar 的成交额 × 量比 × 典型价比。

    依据：成交额 ≈ 成交量 × 当日均价，而典型价是均价的代理。
    本地 close 是东财 fqt=2 **后复权**价、amount 是**未复权**真实成交额，
    两者的比值（复权因子）只在除权除息日跳变 —— 估算基准与新 bar 相隔越近
    越准，所以基准取本地最近的一根。
    """
    if ref_vol <= 0 or ref_typ <= 0:
        return 0.0
    return ref_amount * (new_vol / ref_vol) * (new_typ / ref_typ)


def _validate_via_tencent(secid: str, local_rows: list[dict],
                           em_reason: str) -> dict[str, Any]:
    """东财不可用时，用腾讯当数据源。

    口径要点（与东财路径同量级，只收紧不放宽）：
      1. 腾讯后复权与东财 fqt=2 的锚点不同（实测逐标的约 1.006 的常数差），
         直接拼接会在边界制造约 0.6% 的价格跳变。所以先用本地历史与腾讯在
         重叠段上的比值带标定换算系数 hist_ratio，再把新 bar 的四个价格
         （开/收/高/低）**全部**换算回本地锚点尺度。
      2. 比值带不稳（重叠天数 < 20 或离散 > 1%）→ 整只判失败，不写入。
      3. 边界漂移（重叠段末端的实际比值与历史比值带偏离 > RATIO_TOL）
         → 整只判失败，不写入。
      4. 成交量/成交额是绝对量，不做价格换算。但腾讯返回的成交量单位随
         接口字段而变（实测科创板给"股"、主板给"手"），所以用重叠段实测
         标定单位比 vol_scale，再折算回本地"手"的口径。
      5. 成交额腾讯不提供，用本地基准 bar 的成交额按「量比 × 典型价比」估算，
         并先在重叠段上用同一个公式回测（中位偏差 > 10% 就整只判失败）。
         不写 amount=0：那会被下游当成"这天零成交额"，污染流动性判断。
    """
    last_date = local_rows[-1]["date"]
    beg = (
        datetime.strptime(last_date, "%Y-%m-%d") - timedelta(days=OVERLAP_DAYS)
    ).strftime("%Y-%m-%d")
    out: dict[str, Any] = {
        "secid": secid, "last_date": last_date, "primary_source": "tencent",
    }

    # --- 取腾讯
    tx_code = to_tx_code(secid)
    if not tx_code:
        return _fail_tencent(out, "tencent_primary",
                             f"{em_reason}；且非 A 股标的，腾讯回退不可用")
    try:
        tx_rows = tx_fetch(tx_code)
    except Exception as exc:  # noqa: BLE001
        return _fail_tencent(out, "tencent_primary",
                             f"{em_reason}；腾讯也不可用：{exc}")
    if not tx_rows:
        return _fail_tencent(out, "tencent_primary",
                             f"{em_reason}；腾讯返回空序列")

    local_close = {r["date"]: float(r["close"]) for r in local_rows}
    local_vol = {r["date"]: float(r.get("volume") or 0.0) for r in local_rows}
    local_amt = {r["date"]: float(r.get("amount") or 0.0) for r in local_rows}
    local_typ = {r["date"]: _typ(r) for r in local_rows}
    # 排序保证新 bar 按日期递增（只追加、不改写的硬要求）
    tx_rows = sorted(tx_rows, key=lambda r: r["date"])
    tx_close = {r["date"]: float(r["close"]) for r in tx_rows}
    tx_vol = {r["date"]: float(r.get("volume") or 0.0) for r in tx_rows}
    tx_typ = {r["date"]: _typ(r) for r in tx_rows}
    tx_dates = [r["date"] for r in tx_rows]

    # --- 标定窗口：与东财路径同口径，只看本地末根往前 OVERLAP_DAYS 天
    calib = sorted(d for d in (set(local_close) & set(tx_close)) if d >= beg)
    if not calib:
        return _fail_tencent(out, "tencent_overlap",
                             f"{em_reason}；腾讯与本地没有重叠日期，无法标定锚点")
    # 本地历史里"完整交易日"的集合 —— 盘中抓的半截 bar 会同时污染比值带
    # 的离散度和边界比值，必须排除，否则好标的被误杀。
    complete = set(
        r["date"] for r, ok in zip(local_rows, _local_complete_flags(local_rows))
        if ok
    )
    calib_clean = [d for d in calib if d in complete]
    if len(calib_clean) < TX_FALLBACK_MIN_N:
        calib_clean = calib      # 完整日不够就退回用全部（更保守）
    r_mean, r_spread, r_n = _overlap_ratio(
        {d: local_close[d] for d in calib_clean}, {d: tx_close[d] for d in calib_clean})
    if r_n < TX_FALLBACK_MIN_N or r_spread > TX_FALLBACK_MAX_SPREAD:
        return _fail_tencent(
            out, "tencent_ratio",
            f"{em_reason}；历史比值带不稳（n={r_n} 离散 {r_spread:.4%}），"
            f"两源口径对不上，拒绝写入",
            hist_ratio=r_mean, hist_spread=r_spread, hist_n=r_n)
    hist_ratio = r_mean

    # --- 新 bar（严格晚于本地末根）
    #     先看有没有东西要补：没有新 bar 就没有任何写入，风险为零，
    #     不该因为边界比值对不上而把标的判失败（那会平白拉低覆盖率）。
    new_tx = [r for r in tx_rows if r["date"] > last_date]
    if not new_tx:
        out["ok"] = True
        out["new_rows"] = []
        out["stats"] = {"overlap_days": len(calib), "new": 0,
                        "cross_source": "skipped"}
        out["reason"] = f"{em_reason}；腾讯也没有更新数据，本地已是最新"
        return out

    # --- 边界一致性：新 bar 继承的比例 vs 历史比值带
    #     新 bar 落在重叠段之后，它继承的比例就是**边界日**（本地末根）的
    #     实测比值；这个比值与历史比值带的偏离超过 RATIO_TOL，说明边界处
    #     口径变了（锚点换过、或本地末根本身是盘中抓的半截 bar），
    #     按 hist_ratio 换算会在边界造出一个假的价格跳变 —— 拒绝写入。
    #
    #     这条与东财路径的"重叠段必须逐日一致"是同一个用意：东财路径靠
    #     OVERLAP_TOL 逐日比，这里没有逐日可比的对象（腾讯锚点本就不同），
    #     所以退化成"边界日比值必须落在历史比值带内"。
    if last_date not in tx_close:
        return _fail_tencent(
            out, "tencent_overlap",
            f"{em_reason}；腾讯没有覆盖本地末根 {last_date}，无法校验边界口径，拒绝写入")
    r_seam = local_close[last_date] / tx_close[last_date]
    drift = abs(r_seam - hist_ratio) / hist_ratio if hist_ratio else 1.0
    if drift > RATIO_TOL:
        return _fail_tencent(
            out, "tencent_ratio",
            f"{em_reason}；边界比值 {r_seam:.6f} 与历史比值带 {hist_ratio:.6f} "
            f"漂移 {drift:.4%} 超过 {RATIO_TOL:.4%}（本地末根可能是盘中快照），"
            f"拒绝写入",
            hist_ratio=hist_ratio, hist_spread=r_spread, hist_n=r_n, drift=drift)

    # --- 成交量单位标定（腾讯给"股"还是"手"，逐标的实测）
    rv = [local_vol[d] / tx_vol[d] for d in calib_clean
          if local_vol.get(d) and tx_vol.get(d)]
    if len(rv) < TX_FALLBACK_MIN_N:
        return _fail_tencent(out, "tencent_ratio",
                             f"{em_reason}；重叠段成交量样本不足（{len(rv)}），"
                             f"无法标定单位，拒绝写入")
    vol_scale = statistics.median(rv)
    outlier_rate = sum(1 for x in rv if abs(x / vol_scale - 1) > 0.2) / len(rv)
    if outlier_rate > TX_FALLBACK_VOL_OUTLIER:
        return _fail_tencent(
            out, "tencent_ratio",
            f"{em_reason}；成交量单位比不稳（异常日占比 {outlier_rate:.1%}），拒绝写入",
            vol_scale=vol_scale, vol_outlier_rate=outlier_rate)

    # --- 成交额估算的基准 bar（必须是完整交易日）
    ref = _pick_amount_ref(local_rows, tx_vol, vol_scale)
    if ref is None:
        return _fail_tencent(out, "tencent_amount",
                             f"{em_reason}；本地历史没有可用的成交额基准"
                             f"（完整交易日的成交额），无法估算成交额，拒绝写入")
    ref_date = ref["date"]
    ref_vol = float(ref["volume"])
    ref_amount = float(ref["amount"])
    ref_typ = _typ(ref)

    # --- 成交额估算器自检：在重叠段上用**同一个公式**回测
    errs: list[float] = []
    test_days = calib_clean[-TX_FALLBACK_AMT_TEST_DAYS - 1:]
    for prev, d in zip(test_days, test_days[1:]):
        a_prev, a_d = local_amt.get(prev, 0.0), local_amt.get(d, 0.0)
        if not (a_prev > 0 and a_d > 0 and local_vol.get(prev)):
            continue
        pred = _amount_estimate(
            a_prev, local_vol[prev], local_typ[prev],
            tx_vol[d] * vol_scale, tx_typ[d] * hist_ratio,
        )
        if pred <= 0:
            continue
        errs.append(abs(pred / a_d - 1.0))
    if len(errs) < TX_FALLBACK_AMT_MIN_SAMPLES:
        return _fail_tencent(out, "tencent_amount",
                             f"{em_reason}；成交额估算回测样本不足（{len(errs)}），"
                             f"无法确认口径，拒绝写入")
    amt_err = statistics.median(errs)
    if amt_err > TX_FALLBACK_AMT_TOL:
        return _fail_tencent(
            out, "tencent_amount",
            f"{em_reason}；成交额估算中位偏差 {amt_err:.2%} 超过 "
            f"{TX_FALLBACK_AMT_TOL:.0%}，口径对不上，拒绝写入",
            amount_median_err=amt_err, amount_samples=len(errs))

    # --- 换算并生成新 bar
    try:
        from dataquality import move_tolerance, price_limits
        tol = float(move_tolerance(price_limits([secid]))[0])
    except Exception:  # noqa: BLE001
        tol = 0.35
    local_last_close = local_close[last_date]
    new_rows: list[dict] = []
    prev_close = local_last_close
    prev_date = last_date
    for r in new_tx:
        d = r["date"]
        o = float(r["open"]) * hist_ratio
        c = float(r["close"]) * hist_ratio
        h = float(r["high"]) * hist_ratio
        l = float(r["low"]) * hist_ratio
        if not (o > 0 and c > 0 and h > 0 and l > 0):
            return _fail_tencent(out, "tencent_ratio",
                                 f"{em_reason}；{d} 换算后价格非正，拒绝写入")
        # 日线内部一致性（低价 ≤ 收开 ≤ 高价）
        if not (l <= min(o, c) and h >= max(o, c)):
            return _fail_tencent(out, "tencent_ratio",
                                 f"{em_reason}；{d} 换算后日线内部不一致，拒绝写入")
        # 涨跌幅容忍带：按间隔的交易日数复利放宽（本地可能落后多天）
        steps = max(1, bisect.bisect_right(tx_dates, d) - bisect.bisect_right(tx_dates, prev_date))
        allowed = (1.0 + tol) ** steps - 1.0
        if abs(c / prev_close - 1.0) > allowed:
            return _fail_tencent(
                out, "tencent_ratio",
                f"{em_reason}；{d} 换算后日涨幅 {c / prev_close - 1:+.2%} "
                f"超出容忍带 ±{allowed:.2%}，拒绝写入")
        v_new = tx_vol.get(d, 0.0) * vol_scale
        amount_new = _amount_estimate(ref_amount, ref_vol, ref_typ, v_new,
                                      _typ({"high": h, "low": l, "close": c}))
        if not (amount_new > 0):
            return _fail_tencent(out, "tencent_amount",
                                 f"{em_reason}；{d} 估算成交额非正，拒绝写入")
        new_rows.append({
            "date": d, "open": o, "close": c, "high": h, "low": l,
            "volume": v_new, "amount": amount_new,
        })
        prev_close = c
        prev_date = d

    out["ok"] = True
    out["new_rows"] = new_rows
    out["stats"] = {
        "overlap_days": len(calib), "new": len(new_rows),
        "new_from": new_rows[0]["date"], "new_to": new_rows[-1]["date"],
        "hist_ratio": hist_ratio, "hist_spread": r_spread, "hist_n": r_n,
        "boundary_ratio": r_seam, "drift": drift, "vol_scale": vol_scale,
        "amount_median_err": amt_err, "amount_samples": len(errs),
        "amount_mode": "estimated", "amount_ref_date": ref_date,
        # 腾讯路径没有第三个源可比：交叉校验没做，如实记 skipped。
        "cross_source": "skipped",
    }
    out["reason"] = (
        f"新增 {len(new_rows)} 根（{new_rows[0]['date']} → {new_rows[-1]['date']}），"
        f"腾讯回退（东财不可用：{em_reason}）"
    )
    return out


def validate_secid(secid: str, local_rows: list[dict]) -> dict[str, Any]:
    """抓一条标的的新数据并做全套校验。

    返回 dict：ok / reason / new_rows / stats / primary_source。
    不写盘 —— 写盘由调用方在整批通过后统一做。

    东财是主源；东财网络层不可用（抛异常）或返回空序列时，自动回退到腾讯
    作为数据源（腾讯的价格会按历史比值带换算回本地的复权锚点尺度）。
    """
    last_date = local_rows[-1]["date"]
    beg = (
        datetime.strptime(last_date, "%Y-%m-%d") - timedelta(days=OVERLAP_DAYS)
    ).strftime("%Y-%m-%d")

    out: dict[str, Any] = {"secid": secid, "last_date": last_date}

    # --- 主源：东财
    try:
        em_rows = em_fetch(secid, beg)
    except Exception as exc:  # noqa: BLE001
        em_rows = []
        em_reason = f"东财不可用：{exc}"
    else:
        em_reason = "东财返回空序列"
    if not em_rows:
        # 东财挂了 → 换腾讯当数据源（不是判死，是换源）
        return _validate_via_tencent(secid, local_rows, em_reason)

    out["primary_source"] = "eastmoney"
    local_map = {r["date"]: float(r["close"]) for r in local_rows}
    em_map = {r["date"]: float(r["close"]) for r in em_rows}

    # --- 重叠段必须逐日一致（否则锚点/口径变了）
    overlap = sorted(set(local_map) & set(em_map))
    if not overlap:
        out["ok"] = False
        out["stage"] = "overlap"
        out["reason"] = "与本地没有重叠日期，无法验证口径"
        out["stats"] = {"cross_source": "skipped"}
        return out
    for d in overlap:
        base = local_map[d]
        if base and abs(em_map[d] - base) / base > OVERLAP_TOL:
            out["ok"] = False
            out["stage"] = "overlap"
            out["reason"] = (
                f"重叠日 {d} 不一致：本地 {base:.6f} vs 东财 {em_map[d]:.6f}"
            )
            out["stats"] = {"cross_source": "skipped"}
            return out

    # --- 新 bar（严格晚于本地末根）
    new_rows = [r for r in em_rows if r["date"] > last_date]
    if not new_rows:
        out["ok"] = True
        out["new_rows"] = []
        out["stats"] = {"overlap_days": len(overlap), "new": 0,
                        "cross_source": "skipped"}
        out["reason"] = "已是最新，无新增"
        return out

    # --- 第二源交叉校验：腾讯后复权
    tx_code = to_tx_code(secid)
    ratio_ok = None
    if tx_code:
        try:
            tx_rows = tx_fetch(tx_code)
            tx_map = {r["date"]: float(r["close"]) for r in tx_rows}
            # 历史比值带（在重叠段上算）
            hist_local = {d: local_map[d] for d in overlap if d in tx_map}
            hist_tx = {d: tx_map[d] for d in overlap if d in tx_map}
            r_mean, r_spread, r_n = _overlap_ratio(hist_local, hist_tx)
            if r_n < 20 or r_spread > 0.01:
                ratio_ok = False
                out["reason"] = (
                    f"历史比值不稳（n={r_n} 离散 {r_spread:.4%}），两源口径对不上"
                )
            else:
                # 新 bar 的比值必须落在历史比值带内
                new_local = {r["date"]: float(r["close"]) for r in new_rows}
                new_tx = {d: tx_map[d] for d in new_local if d in tx_map}
                n_mean, n_spread, n_n = _overlap_ratio(new_local, new_tx)
                if n_n < 1:
                    ratio_ok = None
                    out["reason"] = "腾讯没有覆盖新日期，跳过第二源校验"
                else:
                    drift = abs(n_mean - r_mean) / r_mean if r_mean else 1.0
                    ratio_ok = drift <= RATIO_TOL
                    out["stats"] = {
                        "hist_ratio": r_mean, "hist_spread": r_spread, "hist_n": r_n,
                        "new_ratio": n_mean, "new_n": n_n, "drift": drift,
                        "cross_source": "passed" if ratio_ok else "failed",
                    }
                    if not ratio_ok:
                        out["ok"] = False
                        out["stage"] = "cross_source"
                        out["reason"] = (
                            f"第二源比值漂移 {drift:.4%}"
                            f"（历史 {r_mean:.6f} → 新增 {n_mean:.6f}），疑似口径污染"
                        )
                        return out
        except Exception as exc:  # noqa: BLE001
            ratio_ok = None
            out["reason"] = f"第二源不可用（不影响写入）：{exc}"

    out["ok"] = True
    out["new_rows"] = new_rows
    out["stats"] = {
        "overlap_days": len(overlap),
        "new": len(new_rows),
        "new_from": new_rows[0]["date"],
        "new_to": new_rows[-1]["date"],
        "cross_source": "passed" if ratio_ok else ("skipped" if ratio_ok is None else "failed"),
    }
    out["reason"] = f"新增 {len(new_rows)} 根（{new_rows[0]['date']} → {new_rows[-1]['date']}）"
    return out


# ------------------------------------------------------------------ 阶段：refresh


def stage_refresh(*, limit: int | None = None, dry_run: bool = False,
                  min_coverage: float = MIN_COVERAGE) -> dict[str, Any]:
    universe = load_universe()
    if not universe:
        return {"ok": False, "reason": "universe.json 读不到，没有标的可刷新"}

    secids = list(universe.keys())
    if limit:
        secids = secids[:limit]

    results: list[dict[str, Any]] = []
    updated: list[tuple[str, list[dict], list[dict]]] = []
    skipped: list[str] = []
    failed: list[dict[str, Any]] = []
    no_baseline: list[str] = []

    # ---- 取本地基线（顺序读本地盘，快）
    baselines: list[tuple[str, list[dict]]] = []
    for secid in secids:
        path = HIST_DIR / f"{secid_safe(secid)}.json"
        try:
            local_rows = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            local_rows = []
        if not local_rows:
            no_baseline.append(secid)
            continue
        baselines.append((secid, local_rows))

    base_map = dict(baselines)

    # ---- 抓取 + 校验（并发；只读网络与内存，不写盘）
    def _work(item: tuple[str, list[dict]]) -> dict[str, Any]:
        secid, local_rows = item
        try:
            return validate_secid(secid, local_rows)
        except Exception as exc:  # noqa: BLE001
            return {"secid": secid, "ok": False, "reason": f"异常：{exc!r}"}

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(_work, baselines):
            done += 1
            results.append(r)
            if not r.get("ok"):
                failed.append(r)
                continue
            if r.get("new_rows"):
                secid = r["secid"]
                updated.append((secid, base_map[secid], r["new_rows"]))
            else:
                skipped.append(r["secid"])
            if done % 100 == 0:
                print(f"  已扫 {done}/{len(baselines)}  更新 {len(updated)}  "
                      f"失败 {len(failed)}  {time.time() - t0:.0f}s", flush=True)

    # 覆盖率口径：**成功抓取并通过校验**的标的占已扫描标的的比例。
    # 不能用"新增了多少"当覆盖率：收盘后重复刷新时没有新 bar，但上游完全正常，
    # 那样算出来是 0%，会把"上游可用"误判成"上游整体不可用"。
    scanned = len(baselines)
    validated = len(updated) + len(skipped)
    coverage = (validated / scanned) if scanned else 0.0
    # 实际用的源（东财主源 / 腾讯回退）—— 回退生效时一眼能看出来
    by_source: dict[str, int] = {}
    for r in results:
        src = str(r.get("primary_source") or "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    payload: dict[str, Any] = {
        "stage": "refresh",
        "ts": now_iso(),
        "dry_run": dry_run,
        "universe": len(universe),
        "scanned": len(secids),
        "validated": validated,
        "updated": len(updated),
        "up_to_date": len(skipped),
        "failed": len(failed),
        "no_baseline": len(no_baseline),
        "coverage": coverage,
        "min_coverage": min_coverage,
        "by_source": by_source,
        "failures": failed[:50],
        "updated_list": [
            {"secid": s, "n": len(rows), "from": rows[0]["date"], "to": rows[-1]["date"]}
            for s, _old, rows in updated[:200]
        ],
    }

    # 覆盖率过低 → 判失败（多半是上游整体不可用），但已逐只验证通过的写入保留。
    # 个别标的失败不算整批不可信 —— 它们各自被拦下不写入，失败清单里有名字。
    payload["ok"] = coverage >= min_coverage
    if coverage < min_coverage:
        payload["reason"] = (
            f"刷新覆盖率 {coverage:.1%} 低于门槛 {min_coverage:.0%}，"
            f"（失败 {len(failed)} 只），本轮结果不可信"
        )
    else:
        payload["reason"] = (
            f"更新 {len(updated)} 只 / 已是最新 {len(skipped)} 只 / 失败 {len(failed)} 只"
        )

    if not dry_run and updated:
        # 整批提交：每只都已单独验证过，写入用 tmp + replace 保证原子
        for secid, local_rows, new_rows in updated:
            merged = list(local_rows) + list(new_rows)
            path = HIST_DIR / f"{secid_safe(secid)}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        payload["written"] = len(updated)

    return payload


# ------------------------------------------------------------------ 阶段：predict / judge


def stage_predict(*, force: bool = False) -> dict[str, Any]:
    import predictor

    t0 = time.time()
    res = predictor.generate_predictions(mode="live", force=force)
    reason = str(res.get("reason") or "")
    skipped = "已有预测记录" in reason
    payload = {
        "stage": "predict",
        "ts": now_iso(),
        "ok": bool(res.get("ok")) or skipped,
        "skipped": skipped,
        "day": res.get("day"),
        "n_predictions": res.get("n_predictions"),
        "reason": reason or ("已生成" if res.get("ok") else "未知原因"),
        "duration_sec": round(time.time() - t0, 1),
        "raw": res,
    }
    return payload


def stage_judge() -> dict[str, Any]:
    import predictor

    t0 = time.time()
    res = predictor.judge_all()
    payload = {
        "stage": "judge",
        "ts": now_iso(),
        "ok": bool(res.get("ok")),
        "reason": res.get("reason") or "",
        "duration_sec": round(time.time() - t0, 1),
        "raw": res,
    }
    return payload


def run_stage(name: str, **kwargs: Any) -> dict[str, Any]:
    """跑一个阶段并留痕（结果文件 + 状态文件）。

    与命令行 --stage 走同一条路径，所以队列调它和日报调它，
    落盘位置、失败语义、状态记录完全一致。

    失败关闭：任何异常都转成 ok=False 的 payload，不往上抛 ——
    上游不可用时必须"记为失败"，而不是把整个调用方掀翻。
    """
    fn = STAGES.get(name)
    if fn is None:
        raise KeyError(f"未知阶段：{name}")
    try:
        payload = fn(**kwargs)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "stage": name,
            "ts": now_iso(),
            "ok": False,
            "reason": f"异常：{exc!r}",
        }
    p = write_result(name, payload)
    payload["path"] = str(p)
    record(load_state(), name, payload)
    return payload


# ------------------------------------------------------------------ 调度


STAGES = {
    "refresh": stage_refresh,
    "predict": stage_predict,
    "judge": stage_judge,
}


def record(st: dict, name: str, payload: dict) -> None:
    rec = st.setdefault("stages", {}).setdefault(name, {})
    rec["last_run_at"] = payload.get("ts")
    rec["last_status"] = "ok" if payload.get("ok") else "failed"
    if payload.get("ok"):
        rec["last_ok_at"] = payload.get("ts")
        rec["last_error"] = None
    else:
        rec["last_error"] = payload.get("reason") or "未知原因"
    rec["last_result"] = payload.get("path")
    rec["last_detail"] = {
        k: v for k, v in payload.items()
        if k in ("updated", "up_to_date", "failed", "coverage", "n_predictions",
                 "day", "skipped", "duration_sec", "cross_source")
    }
    save_state(st)


def daily_loop_status(st: dict | None = None) -> dict[str, Any]:
    """汇总三个阶段最近一次的状态，供日报正文读取。

    返回 {"ok": bool, "stages": {name: {...}}, "failed": [name], "missing": [name]}。
    ok = 没有任何阶段失败过 —— "还没跑过"不算失败（那是没轮上），
    但日报要能区分这两者：没跑过 vs 跑失败了。
    """
    st = load_state() if st is None else st
    stages: dict[str, Any] = {}
    failed: list[str] = []
    missing: list[str] = []
    for name in STAGES:
        rec = (st.get("stages") or {}).get(name) or {}
        status = rec.get("last_status") or "never_run"
        stages[name] = {
            "status": status,
            "ran_at": rec.get("last_run_at"),
            "ok_at": rec.get("last_ok_at"),
            "error": rec.get("last_error"),
            "detail": rec.get("last_detail") or {},
        }
        if status == "failed":
            failed.append(name)
        elif status != "ok":
            missing.append(name)
    return {
        "ok": not failed,
        "updated_at": st.get("updated_at"),
        "stages": stages,
        "failed": failed,
        "missing": missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="每日 live 闭环")
    ap.add_argument("--stage", choices=["refresh", "predict", "judge", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="refresh 只扫前 N 只（调试用）")
    ap.add_argument("--dry-run", action="store_true", help="只抓只验不写盘")
    ap.add_argument("--force", action="store_true", help="predict 允许覆盖已有记录（原则上不允许）")
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    args = ap.parse_args()

    names = list(STAGES) if args.stage == "all" else [args.stage]
    overall_ok = True

    for name in names:
        print(f"[daily_loop] === {name} ===", flush=True)
        if name == "refresh":
            payload = run_stage(name, limit=args.limit, dry_run=args.dry_run,
                                min_coverage=args.min_coverage)
        elif name == "predict":
            payload = run_stage(name, force=args.force)
        else:
            payload = run_stage(name)
        print(f"[daily_loop] {name}: ok={payload.get('ok')} "
              f"{payload.get('reason') or ''}", flush=True)
        if not payload.get("ok"):
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
