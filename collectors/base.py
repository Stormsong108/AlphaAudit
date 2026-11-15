"""底层 HTTP / 解析工具。

设计原则：
1. 零第三方依赖（只用 stdlib），任何机器上都能跑。
2. 每个抓取都返回 (value, source, ok, error) 四元组，方便上层做交叉验证。
3. 任何单个源失败都不能让整条流水线挂掉。
"""
from __future__ import annotations

import gzip
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

# 生产链路使用系统默认 TLS 校验（证书链 + 主机名）。
# 早期临时诊断用的 CERT_NONE 不得进入生产采集路径。
_SSL_CTX = ssl.create_default_context()


@dataclass
class Fetch:
    """一次抓取的结果。"""

    value: Any = None
    source: str = ""
    ok: bool = False
    error: str = ""
    ts: float = field(default_factory=time.time)

    @property
    def fetched_at(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))


def http_get(
    url: str,
    *,
    referer: str | None = None,
    encoding: str = "utf-8",
    timeout: int = 15,
    retries: int = 2,
    headers: dict[str, str] | None = None,
) -> Fetch:
    """GET 一个 URL。失败自动重试（指数退避）。"""
    last_err = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", UA)
            req.add_header("Accept", "*/*")
            req.add_header("Accept-Encoding", "gzip, deflate")
            if referer:
                req.add_header("Referer", referer)
            for k, v in (headers or {}).items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                text = raw.decode(encoding, errors="replace")
                return Fetch(value=text, source=url, ok=True)
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (404, 403, 401):
                break  # 不重试无意义
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(1.2 * (attempt + 1))
    return Fetch(value=None, source=url, ok=False, error=last_err or "unknown")


def http_get_json(url: str, **kw) -> Fetch:
    """GET 并解析 JSON。"""
    f = http_get(url, **kw)
    if not f.ok:
        return Fetch(value=None, source=url, ok=False, error=f.error)
    try:
        return Fetch(value=json.loads(f.value), source=url, ok=True)
    except json.JSONDecodeError as e:
        return Fetch(value=None, source=url, ok=False, error=f"JSON parse: {e}")


def parse_sina_hq(text: str) -> dict[str, list[str]]:
    """解析新浪 hq_str_xxx="a,b,c,..." 格式（GBK 解码后）。"""
    out: dict[str, list[str]] = {}
    for m in re.finditer(r'hq_str_([A-Za-z0-9_]+)="([^"]*)"', text or ""):
        out[m.group(1)] = m.group(2).split(",")
    return out


def safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None or v == "" or v == "-":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def em_quote(
    secids: list[str],
    fields: str = "f2,f3,f4,f5,f6,f8,f9,f12,f13,f14,f18,f20,f23,f124,f297",
) -> Fetch:
    """东方财富批量行情。secids 形如 1.000001 / 0.399001 / 100.NDX。

    f297 = 交易所口径的交易日（YYYYMMDD），f124 = 最后更新时间戳。
    两者都要：f297 判断「今天这个市场开没开市」，f124 判断数据新鲜度。
    只有 f124 时，美股在周末/假期会拿到下一个交易日的日期而无法判断休市。
    """
    url = (
        "https://push2.eastmoney.com/api/qt/ulist.np/get"
        f"?fltt=2&secids={','.join(secids)}&fields={fields}"
    )
    f = http_get_json(url, referer="https://quote.eastmoney.com/")
    if not f.ok:
        return f
    diff = (f.value or {}).get("data", {}).get("diff") or []
    rows = []
    for d in diff:
        ts = safe_float(d.get("f124"))
        raw_date = str(d.get("f297") or "").strip()
        trade_date = ""
        if len(raw_date) == 8 and raw_date.isdigit():
            trade_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        rows.append(
            {
                "code": d.get("f12"),
                "name": d.get("f14"),
                "price": safe_float(d.get("f2")),
                "pct": safe_float(d.get("f3")),
                "chg": safe_float(d.get("f4")),
                "turnover": safe_float(d.get("f8")),
                "amount": safe_float(d.get("f6")),
                "volume": safe_float(d.get("f5")),
                "pe": safe_float(d.get("f9")),
                "pb": safe_float(d.get("f23")),
                "mktcap": safe_float(d.get("f20")),
                "ts": int(ts) if ts else None,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "",
                "trade_date": trade_date,
            }
        )
    return Fetch(value=rows, source="eastmoney.push2", ok=True)


def run_collectors(tasks: dict[str, Callable[[], Fetch]]) -> dict[str, Fetch]:
    """并发跑一组抓取任务。"""
    from concurrent.futures import ThreadPoolExecutor

    out: dict[str, Fetch] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(tasks)))) as ex:
        futs = {ex.submit(fn): name for name, fn in tasks.items()}
        for fut, name in futs.items():
            try:
                out[name] = fut.result()
            except Exception as e:  # noqa: BLE001
                out[name] = Fetch(value=None, source=name, ok=False, error=str(e))
    return out
