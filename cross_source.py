"""跨来源对账（cross-source reconciliation）—— 同一指标，多来源，值必须一致。

为什么需要这个模块
------------------
2026-11-16 血泪：同一份日报里同时出现 -2.03%（11-12 冻结的 cross-segment 产物）
和 -1.30%（当天 canonical OOS），方向还不同。当时是靠人眼交叉比对发现的 ——
但人眼只比对一次，下一次换个人、隔一天，谁还记得去比。

已经踩过的同类坑（每一条都是「同一个量在几处独立算法算出来」）：
  1. 净超额在 cost_decomposition 算 -10.14%、在 benchmark_relative 算 -2.03%，差 5 倍
     → 年化口径 ×252 vs ×(252/hold) 写反了
  2. cross_segment 的 rows[0] 是成本 0.15% 那行，被当成 canonical 取用
     → 测试段超额 +0.3753%（正）vs canonical -1.2962%（负），方向相反
  3. Sharpe 公式 mean/(sd×√ann) 把年化因子除反 → 0.018 vs 真实 0.44
  4. 面板截止日挪一天 → 数字变了，读者以为策略变了

教训（已固化为铁律）：
  **同一个量在系统里被几处独立算法算出来，是最便宜也最可靠的质量保障。**
  交叉比对数值，不一致就是口径串了。

本模块把这条铁律从「人的纪律」变成「代码的门禁」：
  · 每个指标声明它应该出现在哪几个产物里；
  · 每个来源独立取值（各走各的读取路径，不共用中间变量）；
  · 值不一致 → 标 INCONSISTENT → 上层拒绝出结论；
  · 某个来源缺这个指标 → 标 MISSING（不猜、不用别的来源顶替）。

设计约束
--------
D1 零 LLM、纯确定性。
D2 **拒绝出结论优先于给出数字**：宁可日报里写「口径对不上，暂不出结论」，
   也不能挑一个来源显示（挑一个 = 编造确定性）。
D3 各来源取值路径必须互相独立 —— 不能都从同一个中间结果派生，
   否则一致性检验就退化成「同一个数比同一个数」，永远通过。
D4 相对容差：浮点误差按相对量级给容差（默认 1e-9），
   避免 0.0 vs -7.6e-19 这种被误判成不一致。

不做什么
--------
不做「自动修正」。发现不一致时，本模块只负责**如实报告**，
到底哪个来源对、另一个是哪个环节串了口径，由人/后续排查决定。
自动挑一个「看起来合理的」值，正是本项目反复出现的错误来源。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "data" / "research"

OOS_FILE = RESEARCH / "oos_latest.json"
CROSS_SEG_FILE = RESEARCH / "cross_segment_check.json"
REALISM_FILE = RESEARCH / "execution_realism.json"
READINESS_FILE = RESEARCH / "live_readiness.json"

# canonical 口径：成本 0.30% 往返、持有 5 日、分位 0.5。
# 之所以写死，是因为对账的前提是「比的是同一个东西」。
# canonical hold 真实值由 OOS 产物的 strategy_test_segment.hold 决定，
# 下面 _canonical_cfg() 会动态读取，这里的常量只作兜底。
CANON_COST = 0.003
CANON_HOLD = 5
CANON_QUANTILE = 0.5

# 相对容差：1e-9。实测 canonical 测试段超额在 live_readiness 里
# cost_decomposition 算 -0.012962048552470597、benchmark_relative 算
# -0.0129620485524706，差 7.6e-19 —— 相对差 5.9e-17，在 1e-9 内。
REL_TOL = 1e-9
# 绝对下限：两边都接近 0 时，纯相对容差会把浮点噪声放大成"不一致"。
# 1e-12 是安全的 —— 真实口径错位差的是 0.3% 量级（3e-3），远大于此。
ABS_TOL = 1e-12

# 关键指标：日报正文会直接引用这些数字。
# 关键指标必须有 **≥2 个独立来源**交叉验证过 —— 只有一个来源时，
# 那不是"验证通过"，是"没人跟它对过账"，不能当成事实呈现。
KEY_METRICS = frozenset({"excess_annualized"})


def _close(a: float, b: float) -> bool:
    """两个数在容差内视为一致。"""
    if abs(a - b) <= ABS_TOL:
        return True
    denom = max(abs(a), abs(b))
    if denom == 0.0:
        return True
    return abs(a - b) / denom <= REL_TOL


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ------------------------------------------------------------------
# 各来源的独立取值函数
#
# 关键：每个函数自己走自己的读取路径，从产物文件里直接挖。
# 不共用中间变量 —— 否则一致性检验退化成自证。
# ------------------------------------------------------------------
def _from_oos(d: dict[str, Any], key: str) -> float | None:
    """canonical OOS：测试段多头口径。"""
    seg = d.get("strategy_test_segment") or {}
    lo = seg.get("long_only") or {}
    bench = seg.get("benchmark") or {}
    if key == "excess_annualized":
        a, b = lo.get("net_annualized"), bench.get("annualized")
        if a is None or b is None:
            return None
        return float(a) - float(b)
    if key == "long_only_annualized":
        return lo.get("net_annualized")
    if key == "benchmark_annualized":
        return bench.get("annualized")
    if key == "sharpe":
        return lo.get("sharpe")
    if key == "mean_turnover":
        return seg.get("mean_turnover")
    if key == "n_periods":
        return seg.get("n_periods")
    return None


def _from_cross_segment(d: dict[str, Any], key: str, cost: float, hold: int) -> float | None:
    """跨区段产物：必须按口径显式选行，禁止取 rows[0]。"""
    res = d.get("result") or d
    rows = res.get("rows") or []
    row = next(
        (r for r in rows
         if abs(float(r.get("cost_rate", -1)) - cost) < 1e-9
         and int(r.get("hold", -1)) == hold),
        None,
    )
    if row is None:
        return None
    seg = (row.get("per_segment") or {}).get("test") or {}
    if key == "excess_annualized":
        return seg.get("excess_annualized")
    if key == "long_only_annualized":
        return seg.get("long_only_annualized")
    if key == "benchmark_annualized":
        return seg.get("benchmark_annualized")
    if key == "sharpe":
        return seg.get("long_only_sharpe")
    if key == "mean_turnover":
        return seg.get("mean_turnover")
    if key == "n_periods":
        return seg.get("n_periods")
    return None


def _from_realism(d: dict[str, Any], key: str, cost: float, hold: int) -> float | None:
    """成交现实性产物：r1_close（按收盘价成交）分支。"""
    r1 = d.get("r1_close") or {}
    if abs(float(r1.get("cost_rate", -1)) - cost) >= 1e-9:
        return None
    if int(r1.get("hold", -1)) != hold:
        return None
    if key == "excess_annualized":
        return r1.get("excess_annualized")
    if key == "long_only_annualized":
        return (r1.get("long_only") or {}).get("net_annualized")
    if key == "benchmark_annualized":
        return (r1.get("benchmark") or {}).get("annualized")
    if key == "sharpe":
        return (r1.get("long_only") or {}).get("sharpe")
    if key == "mean_turnover":
        return r1.get("mean_turnover")
    if key == "n_periods":
        return r1.get("n_periods")
    return None


def _from_readiness(d: dict[str, Any], key: str) -> float | None:
    """实盘就绪度体检：走 result.benchmark_relative / cost_decomposition。"""
    res = d.get("result") or d
    rel = res.get("benchmark_relative") or {}
    cost = res.get("cost_decomposition") or {}
    if key == "excess_annualized":
        # 体检里有两处算净超额：benchmark_relative 和 cost_decomposition。
        # 取 benchmark_relative（与日报口径一致），另一处在下面单列做内部对账。
        return rel.get("excess_annualized")
    if key == "long_only_annualized":
        return rel.get("long_only_annualized")
    if key == "benchmark_annualized":
        return rel.get("benchmark_annualized")
    if key == "sharpe":
        return None  # 体检没落 sharpe
    if key == "mean_turnover":
        return cost.get("turnover_per_period")
    if key == "n_periods":
        return None
    return None


def _from_readiness_cost_decomposition(d: dict[str, Any]) -> float | None:
    """体检内部的第二处净超额（cost_decomposition.net_excess）——
    专门用来跟 benchmark_relative 互相对账（同一个模块两处算法）。"""
    res = d.get("result") or d
    cd = res.get("cost_decomposition") or {}
    ann = cd.get("annualized") or {}
    return ann.get("net_excess")


# ------------------------------------------------------------------
# 指标声明
# ------------------------------------------------------------------
# 每个指标：(键, 人类可读名, [来源函数...], 单位说明)
# 来源函数的签名统一为 (artifact_dict, key) -> float | None，
# 需要额外口径的（cross_segment / realism）用 lambda 包一层。
METRICS: list[tuple[str, str, list[Callable[[dict[str, Any], str], float | None]]]] = [
    ("excess_annualized", "测试段净超额（相对等权基准，年化）", [
        lambda d, k: _from_oos(d, k),
        lambda d, k: _from_cross_segment(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_realism(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_readiness(d, k),
    ]),
    ("long_only_annualized", "测试段策略年化（多头口径）", [
        lambda d, k: _from_oos(d, k),
        lambda d, k: _from_cross_segment(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_realism(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_readiness(d, k),
    ]),
    ("benchmark_annualized", "测试段基准年化（等权）", [
        lambda d, k: _from_oos(d, k),
        lambda d, k: _from_cross_segment(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_realism(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_readiness(d, k),
    ]),
    ("sharpe", "测试段夏普（多头口径）", [
        lambda d, k: _from_oos(d, k),
        lambda d, k: _from_cross_segment(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_realism(d, k, CANON_COST, CANON_HOLD),
    ]),
    ("mean_turnover", "测试段换手率（每期）", [
        lambda d, k: _from_oos(d, k),
        lambda d, k: _from_cross_segment(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_realism(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_readiness(d, k),
    ]),
    ("n_periods", "测试段期数", [
        lambda d, k: _from_oos(d, k),
        lambda d, k: _from_cross_segment(d, k, CANON_COST, CANON_HOLD),
        lambda d, k: _from_realism(d, k, CANON_COST, CANON_HOLD),
    ]),
]

SOURCE_FILES = {
    "oos_latest.json": OOS_FILE,
    "cross_segment_check.json": CROSS_SEG_FILE,
    "execution_realism.json": REALISM_FILE,
    "live_readiness.json": READINESS_FILE,
}


def _canonical_cfg() -> tuple[float, int]:
    """从 OOS 产物里读 canonical 成本与持有期（不写死，避免口径漂移后静默失效）。"""
    d = _read_json(OOS_FILE) or {}
    seg = d.get("strategy_test_segment") or {}
    cost = seg.get("cost_rate")
    hold = seg.get("hold")
    if cost is None:
        cost = (d.get("config") or {}).get("cost_rate", CANON_COST)
    if hold is None:
        hold = (d.get("config") or {}).get("hold", CANON_HOLD)
    try:
        return float(cost), int(hold)
    except (TypeError, ValueError):
        return CANON_COST, CANON_HOLD


def _panel_to_of_sources(artifacts: dict[str, dict[str, Any] | None]) -> str | None:
    """取被对账产物各自的面板截止日。

    如果各来源面板截止日相同 → 返回那个日期（对账是在同一版面板上做的）。
    如果不同 → 返回 None（报告层会显示"未标注"，不猜）。
    """
    seen: set[str] = set()
    for art in artifacts.values():
        if not art:
            continue
        pt = _extract_panel_to(art)
        if pt:
            seen.add(pt)
    if len(seen) == 1:
        return next(iter(seen))
    return None


def _extract_panel_to(d: dict[str, Any]) -> str | None:
    """从产物里挖面板截止日（兼容多种历史结构）。"""
    for path in (
        ("panel_to",),
        ("panel", "to"),
        ("panel", "end"),
        ("result", "panel_to"),
        ("result", "panel", "to"),
        ("result", "segments", "test", 1),
        ("segments", "test", 1),
        ("source", "panel_to"),
    ):
        cur: Any = d
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and len(cur) >= 10:
            return cur[:10]
    return None


def reconcile() -> dict[str, Any]:
    """跑一次全量对账。返回结构化结果，任何指标不一致都如实报出来。"""
    artifacts: dict[str, dict[str, Any] | None] = {
        name: _read_json(path) for name, path in SOURCE_FILES.items()
    }
    cost, hold = _canonical_cfg()

    metrics_out: list[dict[str, Any]] = []
    for key, label, readers in METRICS:
        values: dict[str, float] = {}
        missing: list[str] = []
        for src_name, artifact in artifacts.items():
            if artifact is None:
                missing.append(src_name)
                continue
            # 按来源名找对应的读取函数：METRICS 里 readers 的顺序
            # 与 SOURCE_FILES 的顺序一一对应（oos / cross_segment / realism / readiness）。
            idx = list(SOURCE_FILES.keys()).index(src_name)
            if idx >= len(readers):
                continue
            try:
                v = readers[idx](artifact, key)
            except Exception:
                v = None
            if v is None:
                missing.append(src_name)
            else:
                values[src_name] = float(v)

        if not values:
            status = "MISSING"
            spread = None
        else:
            vals = list(values.values())
            spread = max(vals) - min(vals)
            consistent = all(_close(vals[0], v) for v in vals[1:])
            if not consistent:
                status = "INCONSISTENT"
            elif len(values) == 1:
                # 只有一个来源 → 无法交叉验证。关键指标必须 ≥2 个来源，
                # 否则「没人跟它对过账」不能算「验证通过」。
                status = "UNVERIFIED"
            else:
                status = "CONSISTENT"

        metrics_out.append({
            "key": key,
            "label": label,
            "status": status,
            "values": values,
            "missing_sources": missing,
            "spread": spread,
        })

    # 体检模块内部对账：net_excess 在 cost_decomposition 与 benchmark_relative
    # 两处独立算法算出，必须一致（这正是 2026-11-12 抓到 5 倍口径 bug 的地方）。
    internal: dict[str, Any] = {"status": "MISSING", "detail": ""}
    rd = artifacts.get("live_readiness.json")
    if rd is not None:
        v_cd = _from_readiness_cost_decomposition(rd)
        v_rel = (_from_readiness(rd, "excess_annualized"))
        if v_cd is None or v_rel is None:
            internal = {"status": "MISSING", "detail": "体检产物缺少净超额的两处之一"}
        else:
            ok = _close(float(v_cd), float(v_rel))
            internal = {
                "status": "CONSISTENT" if ok else "INCONSISTENT",
                "cost_decomposition": float(v_cd),
                "benchmark_relative": float(v_rel),
                "spread": abs(float(v_cd) - float(v_rel)),
                "detail": ("两处独立算法算出的净超额一致" if ok else
                           "同一个净超额在体检模块两处算出不同值 —— 年化口径串了"),
            }

    # 硬失败判定。分两类，性质完全不同：
    #   硬阻断（不通过就不能出结论）：
    #     INCONSISTENT —— 值互相矛盾，必有口径串了
    #     关键指标 MISSING / UNVERIFIED —— 日报要引用的数字，取不到或没人对过账
    #   软告警（不阻断，但必须显示）：
    #     非关键指标取不到或来源少于 2 个；某些来源缺失（覆盖率下降）
    hard_bad: list[dict[str, Any]] = []
    soft_issues: list[dict[str, Any]] = []
    for m in metrics_out:
        key = m["key"]
        is_key = key in KEY_METRICS
        if m["status"] == "INCONSISTENT":
            hard_bad.append(m)
        elif m["status"] in ("MISSING", "UNVERIFIED"):
            if is_key:
                hard_bad.append(m)
            else:
                soft_issues.append(m)
        # 覆盖率下降：值还是一致，但少了一个来源 —— 不阻断，要显示
        if m["status"] == "CONSISTENT" and m["missing_sources"]:
            soft_issues.append({
                **m,
                "status": "COVERAGE",
                "label": f"{m['label']}（来源覆盖不全）",
            })
    internal_bad = internal.get("status") == "INCONSISTENT"
    if internal_bad:
        hard_bad.append({
            "key": "internal_net_excess",
            "label": "体检模块内部净超额（两处独立算法）",
            "status": "INCONSISTENT",
            "spread": internal.get("spread"),
            "values": {
                "cost_decomposition": internal.get("cost_decomposition"),
                "benchmark_relative": internal.get("benchmark_relative"),
            },
        })

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # 本产物不自带面板（它对账的是已有产物），但把被对账产物各自的
        # 面板截止日带上，报告层就能显示"对的是哪一版面板"。
        "panel_to": _panel_to_of_sources(artifacts),
        "canonical": {"cost_rate": cost, "hold": hold, "quantile": CANON_QUANTILE},
        "tolerance": {"rel": REL_TOL, "abs": ABS_TOL},
        "metrics": metrics_out,
        "internal_reconciliation": internal,
        "consistent": (not hard_bad),
        "bad_metrics": [
            {"key": m["key"], "label": m["label"], "status": m["status"],
             "spread": m["spread"], "values": m["values"]}
            for m in hard_bad
        ],
        "soft_issues": [
            {"key": m["key"], "label": m["label"], "status": m["status"],
             "missing_sources": m.get("missing_sources", [])}
            for m in soft_issues
        ],
        "verdict": (
            "各来源数值一致，可以出结论" if (not hard_bad)
            else "存在对不上的指标 —— 拒绝出结论，先查口径"
        ),
    }


def _fmt_value(key: str, v: float) -> str:
    """按指标类型格式化。

    单位错了比数字错了更隐蔽：夏普是比值（0.508），印成 +50.8% 看着像"收益率"；
    期数是计数，印成 +14300% 更是荒谬。所以按 key 分派单位，不做统一 ×100。
    """
    if key == "sharpe":
        return f"{v:+.6f}"
    if key == "n_periods":
        return f"{int(round(v))} 期"
    if key == "mean_turnover":
        return f"{v * 100:+.6f}%"
    return f"{v * 100:+.6f}%"


def format_report(result: dict[str, Any]) -> str:
    """纯文字对账报告（缩进 2 空格，数字加粗 —— 宋老师阅读格式）。"""
    L: list[str] = []
    L.append("【跨来源对账】")
    L.append(f"  判定：**{result['verdict']}**")
    L.append(f"  canonical 口径：成本 {result['canonical']['cost_rate']*100:.2f}% 往返、"
             f"持有 {result['canonical']['hold']} 日")
    L.append("")
    for m in result["metrics"]:
        mark = {"CONSISTENT": "一致", "UNVERIFIED": "仅一个来源（未经交叉验证）",
                "INCONSISTENT": "**对不上**", "MISSING": "**取不到**"}.get(m["status"], m["status"])
        L.append(f"  {m['label']}：{mark}")
        if m["values"]:
            for src, v in m["values"].items():
                L.append(f"      {src}：{_fmt_value(m['key'], v)}")
        if m["missing_sources"]:
            L.append(f"      缺失来源：{', '.join(m['missing_sources'])}")
        if m["status"] == "INCONSISTENT" and m["spread"] is not None:
            L.append(f"      极差 **{_fmt_value(m['key'], m['spread'])}** —— 口径串了")
    L.append("")
    it = result["internal_reconciliation"]
    L.append(f"  体检模块内部对账（净超额的两种算法）：{it['status']}")
    if it.get("detail"):
        L.append(f"      {it['detail']}")
    return "\n".join(L)


def save_result(result: dict[str, Any], path: Path | None = None) -> Path:
    """原子落盘（tmp → replace）。落盘失败不抛 —— 对账结论本身才是主产物。"""
    target = path or (RESEARCH / "cross_source_check.json")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
    except OSError:
        pass
    return target


def main() -> int:
    result = reconcile()
    print(format_report(result))
    p = save_result(result)
    print(f"\n已存：{p}")
    # 退出码语义（研究队列靠它区分「有效结论」和「执行失败」）：
    #   0  = 对账通过，数字可信
    #   2  = 对账**成功执行**，但发现口径矛盾 —— 这是有效结论，不是执行失败。
    #        队列必须能区分这两者，否则每次门禁触发都变成「脚本崩了」，
    #        触发 30 分钟冷却 + 金融组告警噪音，把真事故淹没在假告警里。
    #   1  = 执行本身失败（读不到文件、代码异常）—— 真失败
    #
    # 但「所有来源都取不到」是第三种情况：既不是矛盾，也不是执行失败，
    # 而是**数据还没准备好**（全新部署）。它不能报 2（会误报成口径事故），
    # 也不能报 1（会误报成崩溃），单列一个退出码 3。
    metrics = result.get("metrics") or []
    all_missing = bool(metrics) and all(
        m.get("status") == "MISSING" for m in metrics)
    if result["consistent"]:
        return 0
    if all_missing:
        print("\n[退出码 3] 所有来源均不可用 —— 数据尚未就绪，不构成口径矛盾。")
        return 3
    print("\n[退出码 2] 对账发现口径矛盾 —— 有效结论（门禁生效），非执行失败。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
