"""冻结规格（frozen_spec）的回归测试。

四条必须锁死的性质：

1. **canonical 证据缺失必须显式失败** —— 不允许用默认值凑数。
   canonical OOS 证据（oos_latest.json）是冻结规格唯一的来源，
   它没了就抛 SpecMissing，绝不返回"看起来正常"的默认规格。
2. **spec_hash 只认规则，不认数据** —— 规格里**不含任何数据指纹**。
   数据天天在增长是正常的，把数据指纹混进 spec_hash 会导致每天 hash 都变、
   失去意义。所以：规则变一个字符 → hash 变；数据指纹变 → hash 不变。
3. **构造方式只能是 decile** —— 不能拿别的构造冒充。
4. **权重符号来自发现期 IC 的符号** —— IC<0 的因子必须给负权重，
   否则会把反向因子当正向信号用。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import frozen_spec as FS  # noqa: E402


# ── 造一份最小的 canonical OOS 证据 ────────────────────────────────────────


def _canonical_raw(**over) -> dict:
    raw = {
        "generated_at": "2026-11-14 02:36:16",
        "config": {"lag": 1, "cost_roundtrip": 0.003},
        "strategy_config": {
            "horizon": 5,
            "quantile": 0.5,
            "construction": "decile",
        },
        "discovered": [
            {"name": "ma_dev_60", "mean": -0.048, "t": -4.02},
            {"name": "low_vol", "mean": 0.0406, "t": 3.23},
            {"name": "atr_ratio", "mean": -0.011, "t": -1.2},
        ],
    }
    raw.update(over)
    return raw


def _write_canonical(tmp_path: Path, name: str = "oos_latest.json", **over) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(_canonical_raw(**over), ensure_ascii=False),
                 encoding="utf-8")
    return p


# ── 1. 规格抽取 ────────────────────────────────────────────────────────────


def test_load_spec_extracts_rule_fields(tmp_path):
    """规格必须逐字复刻 canonical 里的规则参数。"""
    p = _write_canonical(tmp_path)
    spec = FS.load_spec(p)

    assert spec["factor_names"] == ["ma_dev_60", "low_vol", "atr_ratio"]
    assert spec["hold"] == 5
    assert spec["lag"] == 1
    assert spec["quantile"] == 0.5
    assert spec["construction"] == "decile"
    assert spec["cost_roundtrip"] == 0.003
    assert spec["winsor"] == FS.DEFAULT_WINSOR
    assert spec["min_valid"] == FS.DEFAULT_MIN_VALID
    assert spec["benchmark"] == "equal_weight_cross_section_winsorized"


def test_signed_weights_follow_ic_sign(tmp_path):
    """权重符号 = 发现期 IC 的符号：IC<0 → 负权重。"""
    p = _write_canonical(tmp_path)
    spec = FS.load_spec(p)
    assert spec["signed_weights"] == [-1.0, 1.0, -1.0]

    # 与 ooslab 的口径一致（正 IC → +1，负 IC → -1），不能自己另发明一套
    for name, w in zip(spec["factor_names"], spec["signed_weights"]):
        ic = spec["factor_means"][name]
        assert (w > 0) == (ic > 0), f"{name} 的权重符号与 IC 符号不一致"


def test_factor_means_are_recorded_for_traceability(tmp_path):
    """发现期 IC 也要留在规格里 —— 出问题时要能回溯是哪批因子。"""
    p = _write_canonical(tmp_path)
    spec = FS.load_spec(p)
    assert set(spec["factor_means"]) == set(spec["factor_names"])
    assert spec["factor_means"]["ma_dev_60"] == pytest.approx(-0.048)


# ── 2. canonical 证据缺失必须大声失败 ──────────────────────────────────────


def test_missing_file_raises_spec_missing(tmp_path):
    with pytest.raises(FS.SpecMissing):
        FS.load_spec(tmp_path / "根本不存在.json")


def test_corrupt_json_raises_spec_missing(tmp_path):
    p = tmp_path / "oos_latest.json"
    p.write_text("{ 这不是合法 JSON", encoding="utf-8")
    with pytest.raises(FS.SpecMissing):
        FS.load_spec(p)


def test_missing_discovered_raises_spec_missing(tmp_path):
    """没有 discovered（发现期因子）→ 不能猜一份因子出来。"""
    raw = _canonical_raw()
    raw.pop("discovered")
    p = tmp_path / "oos_latest.json"
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(FS.SpecMissing):
        FS.load_spec(p)


def test_empty_discovered_raises_spec_missing(tmp_path):
    raw = _canonical_raw(discovered=[])
    p = tmp_path / "oos_latest.json"
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(FS.SpecMissing):
        FS.load_spec(p)


def test_missing_strategy_config_raises_spec_missing(tmp_path):
    """没有 strategy_config（训练段选定的构造方式）→ 不能默认成 decile。"""
    raw = _canonical_raw()
    raw.pop("strategy_config")
    p = tmp_path / "oos_latest.json"
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(FS.SpecMissing):
        FS.load_spec(p)


def test_non_decile_construction_is_rejected(tmp_path):
    """冻结规格只实现了 decile 构造，不能拿别的构造冒充。"""
    raw = _canonical_raw(strategy_config={
        "horizon": 5, "quantile": 0.5, "construction": "long_only_top",
    })
    p = tmp_path / "oos_latest.json"
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(FS.SpecMissing) as ei:
        FS.load_spec(p)
    assert "decile" in str(ei.value)


def test_spec_missing_is_a_runtime_error():
    """调用方可以用 except RuntimeError 一把兜住。"""
    assert issubclass(FS.SpecMissing, RuntimeError)


# ── 3. spec_hash：只认规则 ─────────────────────────────────────────────────


def test_spec_hash_is_deterministic(tmp_path):
    p = _write_canonical(tmp_path)
    a = FS.load_spec(p)["spec_hash"]
    b = FS.load_spec(p)["spec_hash"]
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_spec_hash_changes_when_any_rule_field_changes(tmp_path):
    """规则变一个字符 → hash 就变（这是账本能发现"规则被改过"的前提）。"""
    base = FS.load_spec(_write_canonical(tmp_path))["spec_hash"]

    for field, new_value in [
        ("hold", 10),
        ("lag", 2),
        ("quantile", 0.2),
        ("cost_roundtrip", 0.005),
        ("winsor", 0.02),
        ("min_valid", 40),
        ("factor_names", ["ma_dev_60", "low_vol"]),
        ("signed_weights", [1.0, -1.0]),
    ]:
        spec = FS.load_spec(_write_canonical(tmp_path))
        spec[field] = new_value
        assert FS.spec_hash(spec) != base, f"{field} 变了 hash 却没变"


def test_spec_hash_contains_no_data_fingerprint(tmp_path):
    """**这里不含任何数据指纹** —— 数据天天在增长是正常的。

    把数据指纹混进 spec_hash 会导致每天 hash 都变、失去意义。
    所以：只改数据侧的字段（生成时间、面板元信息），hash 必须不变。
    """
    p1 = _write_canonical(tmp_path, generated_at="2026-11-14 02:36:16")
    p2 = _write_canonical(tmp_path, generated_at="2026-11-15 02:36:16",
                          panel={"n_codes": 1298, "n_dates": 2882})
    h1 = FS.load_spec(p1)["spec_hash"]
    h2 = FS.load_spec(p2)["spec_hash"]
    assert h1 == h2, "数据指纹被混进了 spec_hash —— 每天 hash 都会变"

    # spec 里可以带数据指纹用于溯源，但它不参与 hash 计算
    spec = FS.load_spec(p2)
    assert spec["source_generated_at"] == "2026-11-15 02:36:16"
    assert spec["source"] == str(p2)
    assert FS.spec_hash(spec) == h1


def test_spec_hash_is_stable_across_source_paths(tmp_path):
    """同一套规则放在不同路径下，hash 必须一样（路径不是规则）。"""
    d = tmp_path / "sub"
    d.mkdir()
    p1 = _write_canonical(tmp_path)
    p2 = _write_canonical(d)
    assert FS.load_spec(p1)["spec_hash"] == FS.load_spec(p2)["spec_hash"]


def test_spec_hash_of_two_different_rule_sets_differs(tmp_path):
    p1 = _write_canonical(tmp_path, name="a.json")
    p2 = _write_canonical(tmp_path, name="b.json", strategy_config={
        "horizon": 10, "quantile": 0.5, "construction": "decile",
    })
    assert FS.load_spec(p1)["spec_hash"] != FS.load_spec(p2)["spec_hash"]


# ── 4. 人可读摘要 ──────────────────────────────────────────────────────────


def test_digest_lines_cover_the_rule(tmp_path):
    """摘要要给人和账本看：因子、权重符号、持有期、分位、成本、指纹。"""
    p = _write_canonical(tmp_path)
    spec = FS.load_spec(p)
    lines = FS.spec_digest_lines(spec)
    text = "\n".join(lines)

    assert len(lines) == 6
    assert "因子 3 个" in text
    assert "ma_dev_60-" in text and "low_vol+" in text
    assert "持有期 5" in text
    assert "买入滞后 1" in text
    assert "分位 50%" in text
    assert "0.30%" in text
    assert spec["spec_hash"][:12] in text


def test_digest_lines_are_python_311_safe(tmp_path):
    """frozen_spec 被 execution_realism / forward_ledger 直接 import。

    一旦这个文件在本机 Python 3.11 上 import 失败（典型原因：嵌套同引号
    f-string 是 3.12+ 才合法），整条链路会瘫痪。这里实测一遍，
    确认摘要能正常生成（而不是在 import 期就炸）。
    """
    import sys as _sys

    assert _sys.version_info[:2] >= (3, 9)
    spec = FS.load_spec(_write_canonical(tmp_path))
    assert FS.spec_digest_lines(spec)  # 不抛异常即可


# ── 5. 真实 canonical 文件（只读，不改任何产物）────────────────────────────


def test_real_canonical_file_loads_and_is_decile():
    """仓库里的 canonical OOS 证据必须能被解析成规格（只读）。"""
    if not FS.OOS_LATEST.exists():
        pytest.skip("canonical OOS 证据不存在，跳过")
    spec = FS.load_spec()
    assert spec["construction"] == "decile"
    assert spec["factor_names"], "发现期因子不能为空"
    assert len(spec["factor_names"]) == len(spec["signed_weights"])
    assert spec["spec_hash"] == FS.spec_hash(spec)
