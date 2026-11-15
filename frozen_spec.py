"""冻结规格：把 canonical OOS 的策略参数抽成一份**不可变**的规格。

为什么必须有这个文件：
    前瞻账本要跑半年到一年。半年里如果有人改了因子、改了分位、改了持有期，
    那么账本里前后两段就**不是同一个策略**，跟踪出来的曲线是拼接的，
    结论全部作废。

    所以规则必须「冻结」：从 canonical oos_latest.json 一次性抽出规格，
    算一个 spec_hash 写进账本每一条记录。规格变了 hash 就变，
    账本能立刻发现"规则被改过"，并把新段标成另一条时间线。

这里**不含任何数据指纹** —— 数据天天在增长是正常的，
把数据指纹混进 spec_hash 会导致每天 hash 都变、失去意义。
数据指纹单独记在账本每条记录里，用于溯源，不用于判定规则变更。

零 LLM，纯确定性。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OOS_LATEST = ROOT / "data" / "research" / "oos_latest.json"

# 规格里必须逐字复刻的常量（与 ooslab 保持一致，不允许各写一份）
DEFAULT_HOLD = 5
DEFAULT_LAG = 1
DEFAULT_COST = 0.003
DEFAULT_QUANTILE = 0.5
DEFAULT_WINSOR = 0.01
DEFAULT_MIN_VALID = 50


class SpecMissing(RuntimeError):
    """canonical OOS 证据缺失 —— 不允许用默认值凑数。"""


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def spec_hash(spec: dict[str, Any]) -> str:
    """规格指纹。规则变一个字符，hash 就变。"""
    payload = {
        "factor_names": spec["factor_names"],
        "signed_weights": spec["signed_weights"],
        "hold": spec["hold"],
        "lag": spec["lag"],
        "quantile": spec["quantile"],
        "construction": spec["construction"],
        "cost_roundtrip": spec["cost_roundtrip"],
        "winsor": spec["winsor"],
        "min_valid": spec["min_valid"],
        "benchmark": spec["benchmark"],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def load_spec(path: Path | str | None = None) -> dict[str, Any]:
    """从 canonical OOS 抽出冻结规格。

    只用训练段选出来的因子和权重；持有期/分位/构造方式取训练段选定的
    `strategy_config.chosen`；成本、lag 取 OOS config。
    """
    p = Path(path) if path else OOS_LATEST
    if not p.exists():
        raise SpecMissing(f"canonical OOS 证据不存在: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SpecMissing(f"canonical OOS 证据读不出来: {exc}") from exc

    discovered = raw.get("discovered") or []
    if not discovered:
        raise SpecMissing("canonical OOS 证据里没有 discovered（发现期因子）")

    factor_names = [d["name"] for d in discovered]
    # 权重符号 = 发现期 IC 的符号（与 ooslab.main 完全一致）
    signed_weights = [1.0 if float(d["mean"]) > 0 else -1.0 for d in discovered]

    cfg = raw.get("config") or {}
    chosen = raw.get("strategy_config") or {}
    if not chosen:
        raise SpecMissing("canonical OOS 证据里没有 strategy_config（训练段选定的构造方式）")

    spec: dict[str, Any] = {
        "factor_names": factor_names,
        "signed_weights": signed_weights,
        "factor_means": {d["name"]: float(d["mean"]) for d in discovered},
        "hold": int(chosen.get("horizon") or DEFAULT_HOLD),
        "lag": int(cfg.get("lag") or DEFAULT_LAG),
        "quantile": float(chosen.get("quantile") if chosen.get("quantile") is not None
                         else DEFAULT_QUANTILE),
        "construction": str(chosen.get("construction") or "decile"),
        "cost_roundtrip": float(cfg.get("cost_roundtrip") or DEFAULT_COST),
        "winsor": DEFAULT_WINSOR,
        "min_valid": DEFAULT_MIN_VALID,
        "benchmark": "equal_weight_cross_section_winsorized",
        "source": str(p),
        "source_generated_at": raw.get("generated_at"),
    }
    if spec["construction"] != "decile":
        raise SpecMissing(
            f"冻结规格只实现了 decile 构造，当前是 {spec['construction']}，"
            "不能拿别的构造冒充"
        )
    spec["spec_hash"] = spec_hash(spec)
    return spec


def spec_digest_lines(spec: dict[str, Any]) -> list[str]:
    """人可读的规格摘要（给日报/账本用）。"""
    # 权重符号逐个拼成 "ma_dev_60+" / "low_vol-" 这种形式。
    # ⚠️ 不要写成嵌套 f-string：f"{f'{n}{'+'}'}" 里的嵌套同引号 f-string
    #    只有 Python 3.12+ (PEP 701) 才合法，本机是 3.11 —— 那样写会
    #    让整个 frozen_spec 无法 import，连带 execution_realism /
    #    forward_ledger 全部瘫痪。所以这里用普通循环拼接。
    weight_signs = []
    for name, w in zip(spec["factor_names"], spec["signed_weights"]):
        weight_signs.append(f"{name}{'+' if w > 0 else '-'}")
    return [
        f"因子 {len(spec['factor_names'])} 个：{'、'.join(spec['factor_names'])}",
        f"权重符号：{'、'.join(weight_signs)}",
        f"持有期 {spec['hold']} 个交易日，买入滞后 {spec['lag']} 日",
        f"分位 {spec['quantile']:.0%}（买最高分 {spec['quantile']:.0%}）",
        f"成本假设 {spec['cost_roundtrip']:.2%} 往返",
        f"规则指纹 {spec['spec_hash'][:12]}",
    ]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="打印冻结规格")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    s = load_spec()
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        for line in spec_digest_lines(s):
            print(line)
