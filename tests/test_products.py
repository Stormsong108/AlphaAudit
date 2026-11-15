"""产品采集器回归测试。

锁住 2026-11-09 核对的字段真相：
1. config.PRODUCTS 的代码与东财基金库登记名称一致（防止再次写错代码）
2. 货币基金没有单位净值，只有 7 日年化 + 万份收益
3. 非货币基金净值双源（东财 + 新浪）一致 → verified
4. pingzhongdata 解析：阶段收益率 + 7 日年化 + 每万份收益

不联网的部分用固定样本；联网的部分做真实比对（网络不通则 skip）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from collectors.products import (  # noqa: E402
    _is_money,
    _parse_pingzhong,
    collect_products,
)
import config  # noqa: E402

# 2026-11-09 东财 fundsuggest 实测登记（代码 → 真实名称、类型）
EXPECTED_REGISTRY = {
    "000198": ("天弘余额宝货币", "货币型-普通货币"),
    "000009": ("易方达天天理财货币A", "货币型-普通货币"),
    "000620": ("易方达现金增利货币A", "货币型-普通货币"),
    "952100": ("国泰海通现金管家货币", "货币型-普通货币"),
    "000032": ("易方达信用债债券A", "债券型-长债"),
    "000171": ("易方达裕丰回报债券A", "债券型-混合二级"),
    "000216": ("华安黄金ETF联接A", "指数型-其他"),
    "110022": ("易方达消费行业股票", "股票型"),
    "270042": ("广发纳斯达克100ETF联接人民币(QDII)A", "指数型-海外股票"),
    "161725": ("招商中证白酒指数(LOF)A", "指数型-股票"),
    "005827": ("易方达蓝筹精选混合", "混合型-偏股"),
    "007300": ("国联安中证半导体ETF联接A", "指数型-股票"),
}


# ------------------------------------------------------------------ config 正确性
def test_product_codes_match_registry():
    """config.PRODUCTS 的代码必须与东财实际登记一致，且不重复。"""
    codes = [c for c, _n, _cat in config.PRODUCTS]
    assert len(codes) == len(set(codes)), "产品代码重复"
    assert set(codes) == set(EXPECTED_REGISTRY), (
        f"产品池代码与东财登记不一致: 差异 "
        f"{set(codes) ^ set(EXPECTED_REGISTRY)}"
    )


def test_product_names_match_registry():
    """config 里写的名称必须与东财登记名称一致（防止张冠李戴）。"""
    for code, name, _cat in config.PRODUCTS:
        assert EXPECTED_REGISTRY[code][0] == name, (
            f"{code} 名称错配: config='{name}' 东财='{EXPECTED_REGISTRY[code][0]}'"
        )


def test_money_fund_classification():
    """货币基金判定：4 只货基，8 只非货基。"""
    money = [c for c, _n, cat in config.PRODUCTS if _is_money(cat)]
    assert money == ["000198", "000009", "000620", "952100"]


# ------------------------------------------------------------------ 解析逻辑
SAMPLE_PINGZHONG = """
var fS_name = "天弘余额宝货币";
var fS_code = "000198";
var syl_1n="0.92";
var syl_6y="0.42";
var syl_3y="0.21";
var syl_1y="0.07";
var Data_sevenDaysYearIncome = [[1767225600000,1.0],[1793980800000,0.797]];
var Data_millionCopiesIncome = [[1767225600000,0.30],[1793980800000,0.2168]];
"""


def test_parse_pingzhong_fields():
    p = _parse_pingzhong(SAMPLE_PINGZHONG)
    assert p["name"] == "天弘余额宝货币"
    assert p["syl_1n"] == 0.92
    assert p["syl_6y"] == 0.42
    assert p["syl_3y"] == 0.21
    assert p["syl_1y"] == 0.07
    assert p["seven_day_yield"] == 0.797
    assert p["million_copies"] == 0.2168


def test_parse_pingzhong_takes_last_point():
    """只取最后一个数据点（最新），不是第一个。"""
    p = _parse_pingzhong(SAMPLE_PINGZHONG)
    assert p["seven_day_yield"] != 1.0
    assert p["million_copies"] != 0.30


def test_parse_pingzhong_missing_vars():
    """非货币基金没有 7 日年化变量时不应报错。"""
    p = _parse_pingzhong('var fS_name = "某债基"; var syl_1n="3.63";')
    assert p["name"] == "某债基"
    assert p["syl_1n"] == 3.63
    assert "seven_day_yield" not in p


# ------------------------------------------------------------------ 联网实跑
@pytest.mark.network
def test_collect_products_live():
    """实跑：12 只全部拿到，8 只非货基双源 verified，0 冲突。"""
    d = collect_products()
    prods = d["products"]
    assert len(prods) == len(config.PRODUCTS)

    by_code = {p["code"]: p for p in prods}
    for code, (name, _ftype) in EXPECTED_REGISTRY.items():
        p = by_code[code]
        assert p["src_name"] == name, f"{code} 源名称不符: {p['src_name']}"

    money = [p for p in prods if p["kind"] == "money"]
    nav = [p for p in prods if p["kind"] != "money"]

    for p in money:
        assert p["nav"] is None, "货币基金不应有单位净值"
        assert p["seven_day_yield"] is not None, f"{p['code']} 缺 7 日年化"
        assert p["million_copies"] is not None, f"{p['code']} 缺万份收益"
        assert p["cross"]["status"] == "single_source"

    for p in nav:
        assert p["nav"] is not None, f"{p['code']} 缺净值"
        assert p["cross"]["status"] == "verified", (
            f"{p['code']} 净值双源未通过: {p['cross']}"
        )
        assert p["cross"]["n_sources"] == 2

    assert d["conflicts"] == [], f"存在口径冲突: {d['conflicts']}"
