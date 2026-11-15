"""历史日线数据层：从东方财富拉取日 K，落盘缓存。

为什么需要它：
    没有历史数据，任何"预测"都是故事。要判断一个信号有没有预测力，
    唯一诚实的办法是拿它跑一遍历史，看前瞻收益是不是真的被它解释。

数据源：push2his.eastmoney.com/api/qt/stock/kline/get
字段（fields2=f51..f61）：日期,开,收,高,低,成交量(手),成交额(元),
                          振幅%,涨跌幅%,涨跌额,换手率%
    —— 派生字段（振幅/涨跌幅/换手）我们不直接采信，
       全部由原始 OHLCV 自行重算，避免源方口径污染。

缓存：data/history/kline/{secid_safe}.json
    secid_safe = 1_600036 这种形式（把 '.' 换成 '_'），避免路径穿越。
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.request
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HIST_DIR = ROOT / "data" / "history" / "kline"
# 主流程落盘的不可变快照：里面固化了 market_session（行情接口自己报的交易日）。
SNAPSHOT_DIR = ROOT / "data" / "snapshots"

CST = timezone(timedelta(hours=8))
# A股收盘时间（本地时间）。收盘前，当天的日线还不存在。
CLOSE_TIME = dt_time(15, 5)
# 重叠段一致性容差（相对）。后复权价是精确值，差异即口径变化。
_OVERLAP_TOL = 1e-6

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_CTX = ssl.create_default_context()  # 默认 TLS 校验，证书校验不得关闭


def secid_safe(secid: str) -> str:
    return secid.replace(".", "_")


def _fetch_kline(secid: str, beg: str = "20150101", end: str = "20500101") -> list[dict]:
    """拉取单个标的的日 K，返回 [{date, open, high, low, close, volume, amount}]。

    复权口径固定为 **fqt=2（后复权）**，不要改回 fqt=1（前复权）。

    为什么必须是后复权：
        东财的前复权价以"最新价"为锚往回折算，对长期高分红股
        （贵州茅台、中国神华、五粮液等），历史段的前复权价会**穿过零点变成负数**。
        2026-11-10 实测：贵州茅台前复权首价 -117.62、中国神华有 551 根非正价格、
        五粮液最大单日变动 5850%、中兴通讯 1200%。
        用负价格当分母算收益，**符号会翻转、幅度离谱**（-0.15 → +0.5 算成 -433%），
        少数这样的坏点就能把横截面均值、IC、分组收益全部带偏。

        后复权以首日价为锚往后累加，**恒为正**，单日变动落在制度上限之内。
        同一批标的实测：中国神华 8.36%、贵州茅台 9.30%、五粮液 9.58%、中兴通讯 8.38%。
    """
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={secid}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57"
        f"&klt=101&fqt=2&beg={beg}&end={end}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Referer": "https://quote.eastmoney.com/"})
    with urllib.request.urlopen(req, timeout=20, context=_CTX) as resp:
        payload = json.loads(resp.read().decode("utf-8", "replace"))
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    rows: list[dict] = []
    for k in klines:
        parts = k.split(",")
        if len(parts) < 7:
            continue
        try:
            try:
                o, c, h, l = (float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4]))
            except (TypeError, ValueError):
                continue
            # 非正价格一律不落盘：坏数据比没有数据更危险。
            # （后复权不该出现负价；出现了说明源方口径变了或代码错了。）
            if not (o > 0 and c > 0 and h > 0 and l > 0):
                continue
            rows.append({
                "date": parts[0],
                "open": o,
                "close": c,
                "high": h,
                "low": l,
                "volume": float(parts[5]),
                "amount": float(parts[6]),
            })
        except (TypeError, ValueError):
            continue
    return rows


def load_kline(secid: str) -> list[dict]:
    """读缓存；不存在返回空列表。"""
    path = HIST_DIR / f"{secid_safe(secid)}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _parse_date(v: Any) -> date | None:
    """把 "2026-11-12" / "20261112" 解析成 date；解析不了返回 None。"""
    if not v or not isinstance(v, str):
        return None
    s = v.strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def latest_trade_date(*, today: date | None = None, now: datetime | None = None) -> date:
    """最近一个"应该已经有收盘 bar"的交易日（YYYY-MM-DD）。

    这是缓存新鲜度的基准，替代掉原先写死的年份阈值：
    写死年份意味着缓存只要进了那一年就永远不再刷新 —— 数据看着是新的，
    实际可能停在几个月前。

    口径与 trading_calendar 一致：**不硬编码节假日表**。
      1. 优先用行情接口自己报的交易日 —— 快照里的 market_session 里
         A 股的 trade_date 就是接口口径的最近交易日；
      2. 没有快照可依时，退回"最近一个工作日"（周一~周五）推算，
         再按收盘时间修正：收盘前，今天的 bar 还不该有。

    注意这里给的是"数据截止到哪一天算新鲜"，不是"今天星期几"。
    """
    now = now or datetime.now(CST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=CST)
    else:
        now = now.astimezone(CST)
    today = today or now.date()

    # 1) 行情接口自己报的交易日（快照里固化），最准
    snap = _latest_snapshot_session()
    if snap:
        cn = (snap.get("markets") or {}).get("A股") or {}
        td = _parse_date(cn.get("trade_date"))
        if td and td <= today:
            # 快照里的 A 股交易日就是接口口径的"最近一个已收盘交易日"，
            # 但它是快照当时的状态；只接受不晚于今天的值。
            return td

    # 2) 退回工作日推算
    d = today
    if now.time() < CLOSE_TIME:
        # 还没收盘 → 今天还没有 bar，基准往前退一天
        d = d - timedelta(days=1)
    while d.weekday() >= 5:  # 周末不是交易日
        d = d - timedelta(days=1)
    return d


def _latest_snapshot_session() -> dict | None:
    """读最近一份快照里固化的 market_session（没有就 None）。"""
    try:
        snaps = sorted(SNAPSHOT_DIR.glob("*.json"))
    except OSError:
        return None
    for fn in reversed(snaps):
        try:
            data = json.loads(fn.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sess = data.get("market_session")
        if isinstance(sess, dict) and sess.get("markets"):
            return sess
    return None


def is_fresh(cached: list[dict], **kw: Any) -> bool:
    """缓存是否够新（末根 bar 已覆盖到最近交易日）。

    够新就直接返回缓存，不联网 —— 这是"缓存优先"的前提，
    绝不能因为改了阈值就变成每次都联网重拉。
    """
    if not cached:
        return False
    last = _parse_date((cached[-1] or {}).get("date"))
    if last is None:
        return False
    return last >= latest_trade_date(**kw)


def save_kline(secid: str, rows: list[dict]) -> None:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    path = HIST_DIR / f"{secid_safe(secid)}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# 拉取时多要一段用于重叠校验（与 daily_loop 的 OVERLAP_DAYS 一致）。
OVERLAP_DAYS = 60


def merge_append(local: list[dict], fresh: list[dict]) -> tuple[list[dict], bool]:
    """把新抓到的行情并到本地缓存后面：**只追加，历史段一个字节都不改**。

    返回 (合并后的行, 是否可以写入)。

    为什么不能整段覆盖：
        后复权价以上市首日为锚，但**分红送股会回头重算整段历史**。
        整段覆盖等于让上游悄悄改写已经落盘的历史 ——
        面板、样本外结果会跟着漂移，而且事后无从察觉。

    重叠段与本地对不上（锚点变了）时拒绝写入：宁可保留旧数据，
    也不用"看起来更新"的数据覆盖历史。

    与 daily_loop.py 的铁律一致（只追加 + 重叠校验）。
    """
    if not local:
        return list(fresh), True
    if not fresh:
        return list(local), False

    local_map = {r["date"]: r for r in local}
    fresh_map = {r["date"]: r for r in fresh}
    overlap = sorted(set(local_map) & set(fresh_map))
    for d in overlap[-30:]:
        base = float(local_map[d].get("close") or 0.0)
        new = float(fresh_map[d].get("close") or 0.0)
        if base and abs(new - base) / base > _OVERLAP_TOL:
            return list(local), False

    last = local[-1]["date"]
    newer = [r for r in fresh if r["date"] > last]
    if not newer:
        return list(local), False
    return list(local) + newer, True


def get_history(
    secid: str,
    *,
    refresh: bool = False,
    min_date: str = "20150101",
    sleep: float = 0.12,
    now: datetime | None = None,
) -> list[dict]:
    """取历史日线：优先缓存，过期或缺失则联网拉取。

    返回按日期升序的列表；每条含 date/open/high/low/close/volume/amount。

    新鲜度基准是"最近一个已收盘的交易日"（见 latest_trade_date），
    不是写死的年份 —— 写死会让缓存进了那一年之后就再也不刷新。

    已有缓存时只做**增量追加**（见 merge_append），不整段覆盖：
    后复权价会随分红送股回头重算，整段覆盖等于让上游改写已落盘的历史。
    """
    cached = load_kline(secid)
    if not refresh and is_fresh(cached, now=now):
        return cached

    # 已有缓存 → 只拉尾部（多要一段做重叠校验），增量追加
    if cached:
        last = _parse_date(cached[-1].get("date"))
        beg = (
            (last - timedelta(days=OVERLAP_DAYS)).strftime("%Y%m%d")
            if last else min_date
        )
    else:
        beg = min_date

    try:
        rows = _fetch_kline(secid, beg=beg)
    except Exception as exc:  # 网络问题不能拖垮整批
        if cached:
            return cached
        raise RuntimeError(f"{secid} 历史行情获取失败且无缓存: {exc}") from exc

    if rows:
        merged, writable = merge_append(cached, rows)
        if writable:
            save_kline(secid, merged)
        rows = merged
    time.sleep(sleep)
    return rows


def get_many(secids: list[str], *, refresh: bool = False) -> dict[str, list[dict]]:
    """批量取历史，失败的标的用空列表占位（不中断整批）。"""
    out: dict[str, list[dict]] = {}
    for s in secids:
        try:
            out[s] = get_history(s, refresh=refresh)
        except Exception:
            out[s] = []
    return out


def to_series(rows: list[dict]) -> tuple[list[str], list[float]]:
    """转成 (日期列表, 收盘价列表)，方便做序列运算。"""
    dates = [r["date"] for r in rows]
    closes = [float(r["close"]) for r in rows]
    return dates, closes


def stats(rows: list[dict]) -> dict[str, Any]:
    """一段历史的覆盖情况，用于报告里如实说明样本范围。"""
    if not rows:
        return {"bars": 0, "from": None, "to": None}
    return {
        "bars": len(rows),
        "from": rows[0]["date"],
        "to": rows[-1]["date"],
    }
