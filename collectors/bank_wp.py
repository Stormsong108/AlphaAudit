"""银行理财产品采集：中国理财网（银行业理财登记托管中心）登记清单。

为什么单独用这个源：
- 银行理财没有公募那样的统一净值接口，权威登记口径在中国理财网披露平台。
- 这里只取登记事实（产品名、管理人、运作方式、风险等级、登记编码），
  不臆造净值或收益率；净值口径缺失时在 cross.note 里写明。

分类口径（按登记名称 + 风险等级关键词，规则写死可复算）：
- 名称含「固定收益类」或「固收」→ 银行理财固收+
- 名称含「现金管理」→ 现金管理类
- 名称含「权益」「股票」→ 权益类
- 其余 → 稳健类
"""
from __future__ import annotations

import time
from typing import Any

from collectors.base import Fetch, http_get_json

ONSALE_URL = "https://www.chinawealth.com.cn/wealthprod/onsaleProduct_EN.json"

FIXED_INCOME_MARKERS = ("固定收益类", "固收")
CASH_MARKERS = ("现金管理",)
EQUITY_MARKERS = ("权益", "股票", "资本市场")


def _classify(name: str) -> str:
    n = name or ""
    if any(m in n for m in FIXED_INCOME_MARKERS):
        return "银行理财固收+"
    if any(m in n for m in CASH_MARKERS):
        return "现金管理类"
    if any(m in n for m in EQUITY_MARKERS):
        return "权益类"
    return "稳健类"


def _fetch_onsale() -> Fetch:
    f = http_get_json(ONSALE_URL, referer="https://www.chinawealth.com.cn/", timeout=20)
    if not f.ok or not isinstance(f.value, list):
        return Fetch(value=None, source=ONSALE_URL, ok=False, error=f.error or "非列表响应")
    rows: list[dict[str, Any]] = []
    for it in f.value:
        name = (it.get("prodName") or "").strip()
        if not name:
            continue
        rows.append(
            {
                "code": it.get("prodRegCode") or "",
                "name": name,
                "org": it.get("orgName") or "",
                "mode": it.get("prodOperateModeName") or "",
                "risk": it.get("prodRiskLevelName") or "",
                "category": _classify(name),
                "detail_url": it.get("detailurl") or "",
            }
        )
    return Fetch(value=rows, source="chinawealth.onsaleProduct", ok=bool(rows))


def collect_bank_wp() -> dict:
    """返回 {products: [...], sources: {...}}，产品口径为登记信息，无净值。"""
    f = _fetch_onsale()
    rows = f.value if f.ok and f.value else []
    products: list[dict[str, Any]] = []
    for r in rows:
        products.append(
            {
                "code": r["code"],
                "name": r["name"],
                "category": r["category"],
                "kind": "bank_wp",
                "org": r["org"],
                "mode": r["mode"],
                "risk": r["risk"],
                "nav": None,
                "pct": None,
                "ret_1y": None,
                "ret_6m": None,
                "date": time.strftime("%Y-%m-%d"),
                "rationale": (
                    f"登记来源：中国理财网；管理人 {r['org']}；运作方式 {r['mode']}；"
                    f"风险等级 {r['risk']}；分类按登记名称关键词判定"
                ),
                "cross": {
                    "srcs": {"中国理财网登记": r["code"]},
                    "src_names": ["中国理财网登记"],
                    "n_sources": 1,
                    "status": "single_source",
                    "note": "银行理财登记信息为单一权威源；公开接口无统一净值口径，本期不列净值与收益率",
                },
            }
        )
    return {
        "products": products,
        "sources": {
            "chinawealth": {
                "ok": f.ok,
                "error": f.error,
                "count": len(products),
                "source": "chinawealth.onsaleProduct",
            }
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
