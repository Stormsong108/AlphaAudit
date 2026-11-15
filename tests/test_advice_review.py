"""模拟组合建议 / 复盘 / HTML 渲染 的回归测试。

锁住的规则：
1. 四类股票主题分类正确，港股池不含银行/通信股，无重复公司
2. 理财产品四类齐全（货币 / 固收+ / 纯债 / 混合），权益类不占非权益仓位
3. 模拟组合的硬约束：单只 ≤ single_max、总仓位 ≤ total_max、现金 ≥ cash_min
4. 双源冲突的标的不进组合
5. 止损价 / 目标价计算正确
6. 复盘的命中率、止损触发率、目标达成率计算正确
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from advice import build_advice, score_stock  # noqa: E402
from collectors.bank_wp import _classify  # noqa: E402
from render import render_html  # noqa: E402
from review import build_review  # noqa: E402


# ------------------------------------------------------------------ 主题分类
def test_hk_pool_has_no_bank_or_telecom():
    """港股互联网池里不能混进银行股、通信股。"""
    for code, name, sector, theme in config.HK_INTERNET:
        assert theme == "港股互联网"
        assert sector not in ("银行", "通信"), f"{name} 不该在港股互联网池"


def test_no_duplicate_company_across_pools():
    """同一公司不能同时出现在两个主题里（会重复暴露）。"""
    names = [n for pool in (config.LARGE_CAP, config.TECH_GROWTH, config.DIVIDEND, config.HK_INTERNET) for _c, n, _s, _t in pool]
    dup = {n for n in names if names.count(n) > 1}
    assert not dup, f"重复标的: {dup}"


def test_all_four_themes_present():
    themes = {t for pool in (config.LARGE_CAP, config.TECH_GROWTH, config.DIVIDEND, config.HK_INTERNET) for _c, _n, _s, t in pool}
    assert themes == {"大盘蓝筹", "科技成长", "红利低波", "港股互联网"}


# ------------------------------------------------------------------ 理财分类
def test_product_four_categories_present():
    cats = {c for _code, _n, c in config.PRODUCTS}
    for need in ("货币基金", "固收+基金", "纯债基金", "混合基金"):
        assert need in cats, f"缺类别 {need}"


def test_bank_wp_classification():
    assert _classify("某银行固定收益类理财产品") == "银行理财固收+"
    assert _classify("某某固收增强封闭式") == "银行理财固收+"
    assert _classify("某某现金管理类") == "现金管理类"
    assert _classify("某某权益类混合") == "权益类"
    assert _classify("某某稳健精选") == "稳健类"


# ------------------------------------------------------------------ 建议引擎
def _mk_stock(code, name, theme, price, pct, cross, market="CN", pe=None, amount=None):
    return {
        "code": code, "name": name, "theme": theme, "sector": "测试", "market": market,
        "price": price, "pct": pct, "pe": pe, "pb": None, "amount": amount,
        "cross": cross,
    }


VERIFIED = {"status": "verified", "n_sources": 2, "src_names": ["a", "b"]}
CONFLICT = {"status": "conflict", "n_sources": 2, "src_names": ["a", "b"], "note": "口径不一致"}


def _quotes(stocks):
    return {"indexes": [], "stocks": stocks, "conflicts": [], "sources": {}}


def test_conflict_stock_excluded_from_portfolio():
    stocks = [
        _mk_stock("S1", "冲突股", "大盘蓝筹", 10.0, 5.0, CONFLICT),
        _mk_stock("S2", "正常股", "大盘蓝筹", 20.0, 3.0, VERIFIED),
    ]
    a = build_advice(_quotes(stocks), {"products": []})
    codes = [p["code"] for p in a["portfolio"] if p["kind"] == "stock"]
    assert "S1" not in codes, "双源冲突标的不得进模拟组合"
    assert "S2" in codes


def test_position_limits_respected():
    stocks = [
        _mk_stock(f"S{i}", f"股{i}", "大盘蓝筹", 10.0 + i, 4.0, VERIFIED)
        for i in range(12)
    ]
    a = build_advice(_quotes(stocks), {"products": []})
    stocks_p = [p for p in a["portfolio"] if p["kind"] == "stock"]
    for p in stocks_p:
        assert p["weight_pct"] <= config.RISK["single_max_pct"], "单只超上限"
    alloc = a["allocation"]
    assert alloc["equity_pct"] <= config.RISK["total_max_pct"], "总仓位超上限"
    assert alloc["cash_pct"] >= config.RISK["cash_min_pct"] - 0.05, "现金低于下限"
    assert abs(
        alloc["equity_pct"] + alloc["product_pct"] + alloc["cash_pct"] - 100.0
    ) < 0.5, "仓位没凑到 100"


def test_stop_and_target_prices():
    stocks = [_mk_stock("S1", "测试股", "科技成长", 100.0, 2.0, VERIFIED)]
    a = build_advice(_quotes(stocks), {"products": []})
    p = [x for x in a["portfolio"] if x["kind"] == "stock"][0]
    assert p["stop_pct"] == config.RISK["stop_loss_a"]
    assert p["stop_price"] == pytest.approx(100.0 * (1 - 0.03), abs=1e-6)
    for t in p["targets"]:
        assert t["price"] == pytest.approx(100.0 * (1 + t["gain_pct"] / 100.0), abs=1e-6)


def test_hk_position_uses_wider_stop():
    stocks = [_mk_stock("H1", "港股", "港股互联网", 200.0, 1.0, VERIFIED, market="HK")]
    a = build_advice(_quotes(stocks), {"products": []})
    p = [x for x in a["portfolio"] if x["kind"] == "stock"][0]
    assert p["stop_pct"] == config.RISK["stop_loss_us"]


def test_score_penalises_conflict():
    good = score_stock(_mk_stock("A", "好", "大盘蓝筹", 10.0, 3.0, VERIFIED))
    bad = score_stock(_mk_stock("A", "差", "大盘蓝筹", 10.0, 3.0, CONFLICT))
    assert good["score"] > bad["score"]


# ------------------------------------------------------------------ 复盘
def _write_snapshot(d: Path, day: str, prices: dict, positions: list[dict]):
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "day": day,
        "generated_at": day + " 21:30:00",
        "immutable": True,
        "prices": prices,
        "advice": {"portfolio": positions},
    }
    (d / f"{day}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_review_hit_rate_and_triggers(tmp_path):
    positions = [
        {
            "kind": "stock", "code": "X1", "name": "甲", "call": "偏多", "price": 100.0,
            "stop_price": 97.0, "targets": [{"gain_pct": 8, "price": 108.0}], "weight_pct": 10.0,
        },
        {
            "kind": "stock", "code": "X2", "name": "乙", "call": "偏多", "price": 100.0,
            "stop_price": 97.0, "targets": [{"gain_pct": 8, "price": 108.0}], "weight_pct": 10.0,
        },
    ]
    _write_snapshot(tmp_path, "2026-11-01", {"X1": 100.0, "X2": 100.0}, positions)
    _write_snapshot(tmp_path, "2026-11-02", {"X1": 105.0, "X2": 96.0}, [])

    r = build_review(tmp_path, {"stocks": []})
    assert r["available"] is True
    st = r["stats"]
    assert st["judged"] == 2
    assert st["hits"] == 1
    assert st["hit_rate"] == 0.5
    assert st["stop_trigger_rate"] == 0.5   # 乙 96 <= 97 触发止损
    assert st["target_hit_rate"] == 0.0    # 都没到 108


def test_review_empty_when_no_history(tmp_path):
    r = build_review(tmp_path, {"stocks": []})
    assert r["available"] is False
    assert r["stats"] == {}


def test_snapshot_is_immutable(tmp_path, monkeypatch):
    import main as main_mod

    monkeypatch.setattr(main_mod, "SNAPSHOT_DIR", tmp_path)
    bundle = {"quotes": {"stocks": [{"code": "S1", "price": 10.0}]}, "events": {}, "futures": {}, "products": {}}
    advice = {"portfolio": [], "note": "第一版"}
    p1 = main_mod.save_immutable_snapshot(bundle, advice, {"available": False}, "2026-11-01")
    advice2 = {"portfolio": [], "note": "偷偷改写的版本"}
    p2 = main_mod.save_immutable_snapshot(bundle, advice2, {"available": False}, "2026-11-01")
    assert p1 == p2
    saved = json.loads(p1.read_text(encoding="utf-8"))
    assert saved["advice"]["note"] == "第一版", "已固化快照不能被改写"


# ------------------------------------------------------------------ HTML
def test_html_contains_required_sections():
    stocks = [_mk_stock("S1", "测试股", "大盘蓝筹", 10.0, 1.0, VERIFIED)]
    advice = build_advice(_quotes(stocks), {"products": []})
    html = render_html(
        {"indexes": [], "stocks": stocks, "conflicts": [], "sources": {}},
        {"events": [], "macro": [], "conflicts": [], "sources": {}, "stats": {}},
        {"cn_futures": [], "global_futures": [], "fx": [], "fx_mid": None, "conflicts": [], "sources": {}},
        products={"products": [], "sources": {}},
        advice=advice,
        review={"available": False, "note": "首期", "stats": {}, "evaluations": []},
    )
    assert "模拟组合建议" in html
    assert "复盘" in html
    assert "不构成投资建议" in html
    assert "<table" in html
