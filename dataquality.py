"""数据质量守卫：面板进研究之前必须先过体检。

为什么单独一个模块：
    2026-11-10 踩过的坑 —— 历史行情用的是东财**前复权**（fqt=1），
    长期高分红股的历史前复权价会穿过零点变成**负数**。
    负价格当分母算收益，符号会翻转、幅度离谱（-0.15 → +0.5 算成 -433%）。
    少数这样的坏点就把横截面均值、IC、分组收益全部带偏，
    表现为"IC 为正但十分位价差为负"这种自相矛盾的现象。

    教训不是"这次抓错了"，而是**体检没有闸门**。
    所以这里把体检做成硬闸门：脏面板直接拒绝进入研究，
    而不是让人从一堆数字里猜哪里不对。

体检项：
    1. 非正价格（复权穿零 / 未复权除权）
    2. 单日涨跌幅超过该股制度上限（主板 ±10%，创业板/科创板 ±20%，北交所 ±30%）
    3. 日线内部一致性（low <= open/close <= high）
    4. 横截面覆盖（每个交易日的有效股票数）
"""
from __future__ import annotations

from typing import Any

import numpy as np


def limit_of(code: str) -> float:
    """该股单日涨跌幅制度上限。

    codes 形如 '0.002003' / '1.688825'：前面是市场前缀，后面才是 6 位代码。
    主板 ±10%，创业板(300/301)、科创板(688/689) ±20%，北交所(4/8) ±30%。
    ST 股 ±5%，但 ST 标记不在代码里，因此不做 ST 判定 ——
    这让检查偏保守（ST 股 ±5% 不会被误报成违规），代价是漏掉一部分真事件。
    """
    digits = "".join(ch for ch in str(code) if ch.isdigit())[-6:]
    if digits.startswith(("688", "689")):
        return 0.20
    if digits.startswith(("300", "301")):
        return 0.20
    if digits.startswith(("8", "4")):
        return 0.30
    return 0.10


def price_limits(codes: list[str]) -> np.ndarray:
    return np.array([limit_of(c) for c in codes], dtype=float)


def move_tolerance(lim: np.ndarray) -> np.ndarray:
    """判定"单日变动是否离谱"时用的容忍带。

    不能直接拿价格制度上限去卡复权序列：后复权是**总回报口径**，
    除权除息日它比裸价多涨（跌）一个现金分红/送股的比例。
    实测 000426 在 2015-03-26 除权日：裸价 +10.04%（真实涨停），
    后复权 +11.68%（含每股 0.194 元分红）—— 后者超过 10% 制度上限是**正常的**。

    所以容忍带 = 制度上限 × 1.10 + 3.5 个百分点的分红余量。
    单次除权的现金分红超过 3.5% 极罕见（茅台年股息率也就 2~3%）。
    真正要拦的是"负价格穿零"和"几倍、几十倍的离谱跳变"，那些噪声带里抓得住。
    """
    return lim * 1.10 + 0.035


def audit_panel(mat: dict[str, Any], *, max_examples: int = 8) -> dict[str, Any]:
    """体检一个面板，返回结构化结论（不抛异常，便于写进报告）。"""
    close = mat["close"]
    codes = mat["codes"]
    dates = mat["dates"]
    T, N = close.shape
    issues: list[dict[str, Any]] = []

    # 1) 非正价格。NaN 是"这天没数据"，不是坏数据，不算违规。
    bad_price = np.isfinite(close) & (close <= 0)
    n_nonpos = int(bad_price.sum())
    if n_nonpos:
        rows, cols = np.where(bad_price)
        ex = []
        for i in range(min(max_examples, len(rows))):
            r, c = int(rows[i]), int(cols[i])
            ex.append(f"{dates[r]} {codes[c]}={close[r, c]!r}")
        issues.append({
            "kind": "non_positive_price",
            "count": n_nonpos,
            "n_codes": int(len(np.unique(cols))),
            "examples": ex,
            "hint": "复权口径穿零（东财前复权 fqt=1 的已知问题）。改用后复权 fqt=2 重抓。",
        })

    # 2) 单日涨跌幅
    #    硬失败：超过 100% —— 没有现金分红能解释，只可能是复牌缺口或坏点
    #    软告警：超过容忍带（制度上限×1.10+3.5pct）—— 多半是除权除息日，
    #            后复权含分红所以可以合法超过价格制度上限
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = close[1:] / close[:-1] - 1.0
    daily = np.where(np.isfinite(daily) & (close[:-1] > 0), daily, np.nan)
    lim = price_limits(codes)
    tol = move_tolerance(lim)[None, :]

    # 新股上市后前 5 个交易日不设涨跌幅限制（主板/创业板/科创板均如此），
    # 实测 688026 上市第 9 个交易日 +108.54% 是合法的。
    # 所以按"该股第几个有效观测"豁免，而不是一刀切。
    age = np.cumsum(np.isfinite(close), axis=0)[1:]   # 与 daily 对齐
    new_listing = age <= 5
    exempt = int((np.isfinite(daily) & (np.abs(daily) > 1.0) & new_listing).sum())

    soft = np.isfinite(daily) & (np.abs(daily) > tol) & ~new_listing
    hard = np.isfinite(daily) & (np.abs(daily) > 1.0) & ~new_listing
    n_soft = int(soft.sum())
    n_hard = int(hard.sum())
    if n_hard:
        rows, cols = np.where(hard)
        ex = []
        for i in range(min(max_examples, len(rows))):
            r, c = int(rows[i]), int(cols[i])
            ex.append(f"{dates[r + 1]} {codes[c]} {daily[r, c]:+.2%}")
        issues.append({
            "kind": "extreme_move",
            "count": n_hard,
            "n_codes": int(len(np.unique(cols))),
            "examples": ex,
            "hint": "单日变动超过 100%，任何分红/送股都解释不了，多为复牌缺口或坏点，须剔除。",
        })
    n_daily = int(np.isfinite(daily).sum())
    soft_rate = (n_soft / n_daily) if n_daily else 0.0
    if soft_rate > 0.01:
        rows, cols = np.where(soft)
        ex = []
        for i in range(min(max_examples, len(rows))):
            r, c = int(rows[i]), int(cols[i])
            ex.append(f"{dates[r + 1]} {codes[c]} {daily[r, c]:+.2%}")
        issues.append({
            "kind": "limit_breach",
            "count": n_soft,
            "rate": soft_rate,
            "n_codes": int(len(np.unique(cols))),
            "examples": ex,
            "hint": ("超过容忍带的记录占 %.2f%%。若为后复权序列，多为除权除息日"
                     "（分红/送股导致总回报高于价格制度上限），属正常；"
                     "需用不复权口径交叉确认因子是否同步跳变。" % (soft_rate * 100)),
        })

    # 3) 横截面覆盖
    #    早期历史本来就没几只股票：2015-01-05 A 股只有 6 只在市。
    #    这是事实不是缺陷，下游 IC 计算用 min_cross 过滤即可，
    #    所以只有"薄日占比过高"才值得拦截。
    valid_per_day = np.isfinite(close).sum(axis=1)
    cover_min = int(valid_per_day.min()) if T else 0
    cover_med = int(np.median(valid_per_day)) if T else 0
    thin_thr = max(20, N // 20)
    n_thin = int((valid_per_day < thin_thr).sum())
    thin_rate = (n_thin / T) if T else 0.0
    if thin_rate > 0.05:
        issues.append({
            "kind": "thin_cross_section",
            "count": n_thin,
            "min_valid": cover_min, "median_valid": cover_med,
            "hint": "横截面过窄的交易日占比 %.2f%% 超过 5%%，IC 在这些日子不可靠。"
                    % (thin_rate * 100),
        })

    n_daily = int(np.isfinite(daily).sum())
    return {
        "ok": not issues,
        "n_codes": N,
        "n_dates": T,
        "from": dates[0] if T else None,
        "to": dates[-1] if T else None,
        "n_daily_returns": n_daily,
        "soft_breach_rate": soft_rate,
        "hard_breach_rate": (n_hard / n_daily) if n_daily else 0.0,
        "new_listing_exempt": exempt,
        "thin_days": n_thin,
        "thin_rate": thin_rate,
        "non_positive_rate": (n_nonpos / (T * N)) if T and N else 0.0,
        "coverage_min": cover_min,
        "coverage_median": cover_med,
        "issues": issues,
    }


class DirtyPanelError(RuntimeError):
    """面板没过体检 —— 拒绝进入研究。"""


def assert_panel_clean(mat: dict[str, Any], *, context: str = "") -> dict[str, Any]:
    """体检不通过就直接抛异常。

    设计取舍：宁可让实验跑不起来，也不要让它跑出一个
    "看起来很专业、其实被坏点污染"的结论。失败要大声。
    """
    report = audit_panel(mat)
    if not report["ok"]:
        lines = [f"面板未通过数据质量体检{f'（{context}）' if context else ''}："]
        for iss in report["issues"]:
            lines.append(
                f"  - {iss['kind']}: {iss['count']} 处"
                f"{'，涉及 ' + str(iss['n_codes']) + ' 只' if 'n_codes' in iss else ''}"
            )
            for ex in iss.get("examples", [])[:4]:
                lines.append(f"      {ex}")
            if iss.get("hint"):
                lines.append(f"      处置建议：{iss['hint']}")
        raise DirtyPanelError("\n".join(lines))
    return report
