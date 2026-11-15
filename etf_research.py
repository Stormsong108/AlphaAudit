# -*- coding: utf-8 -*-
"""ETF 通道研究：训练选参 → 验证确认 → 冻结 → 测试只跑一次。

与个股通道（ooslab / live_readiness）同一套纪律，同一套成本口径，
只是通道成本地板不同：ETF 无印花税、无过户费。

产出：
  data/research/etf_frozen_spec.json   冻结规格（测试段前落盘）
  data/research/etf_channel.json        完整结果

纪律（2026-11-14 修正）
----------------------
旧实现把 `chosen` 取成**验证段超额最高**的那一个（`passed[0]`），即
「训练段选参 → 验证段再选一次」。这等于把验证段变成第二个训练段，
它的漂亮数字是选择偏差，不是策略能力。实测后果：
    ma_only k=3 R=5  训练段超额 -11.91%（12 名里垫底）
                     验证段超额 +14.72%（12 名里第一）
                     测试段超额 -12.38%
验证段的好成绩完全来自「在 12 个候选里挑验证段最好的那个」，
到测试段立刻被打回。

现在的实现：
  1. 选参**唯一来源**是训练段净超额（train_grid[0]），此后再不改。
  2. 验证段只回答一个问题：训练段选出的那一个，过不过关。
     不过关就**停止**，不换参、不跑测试段（换参 = 泄漏）。
  3. 冻结规格先落盘（含 spec_hash），测试段从磁盘读回并校验 hash，
     确认跑的就是冻结的那一份。
  4. 基准与策略对齐到**同一批期次**（dates_filter），
     否则「策略 183 期 vs 基准 192 期」的超额是两套样本相减。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etf_lab as E  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, "etf_channel.json")
FROZEN_PATH = os.path.join(E.OUT_DIR, "etf_frozen_spec.json")

FACTOR_SETS: dict[str, tuple[list[str], dict[str, float]]] = {
    "mom60":        (["mom_60"], {}),
    "mom_dual":     (["mom_20", "mom_60"], {}),
    "trend3":       (["mom_20", "mom_60", "mom_120"], {}),
    "trend_lowvol": (["mom_20", "mom_60", "mom_120", "vol_60"], {"vol_60": -1.0}),
    "trend_dd":     (["mom_20", "mom_60", "mom_120", "dd_120"], {}),
    "ma_only":      (["ma_dev_60"], {}),
    "ma_trend":     (["ma_dev_60", "mom_60"], {}),
    "mixed":        (["mom_60", "ma_dev_60", "vol_60"], {"vol_60": -1.0}),
    "trend_rsi":    (["mom_20", "mom_60", "mom_120", "rsi_14"], {}),
}
K_SET = [1, 2, 3, 5]
R_SET = [5, 10, 20, 40]
CAPITALS = [100_000.0, 300_000.0, 1_000_000.0, 3_000_000.0, 10_000_000.0]
PRICE_MODE = "open"          # 信号用 t-1 收盘，t 日开盘买，t+R 日开盘卖
MIN_BARS = 120               # 至少 120 根 K 才进入横截面
TOP_N_TRAIN = 12             # 训练段取前 N 进入验证（诊断用，不参与选参）
TRAIN_CAPITAL = 1_000_000.0

# 冻结件里除规格本身之外附加的元数据字段（不参与 hash）
_META_KEYS = ("spec_hash", "frozen_at", "selected_by", "validation_excess_ann")


def cost_floor_table(capital: float, k: int, slippage_bp_edge: float = E.SLIPPAGE_BP_EDGE) -> dict[str, Any]:
    """ETF 单笔往返成本（bp），含 ¥5 最低佣金，无印花税/过户费。"""
    per = capital / max(k, 1)
    r = E.etf_roundtrip_cost_rate(per, slippage_bp_edge)
    r.update({
        "capital_yuan": capital, "n_positions": k, "per_order_yuan": per,
        "min_commission_binding": bool(per * E.COMMISSION_RATE < E.COMMISSION_MIN),
    })
    return r


def build_spec(fset: str, k: int, R: int, capital: float,
               seg: dict[str, tuple[int, int]], n_etf: int) -> dict[str, Any]:
    """冻结规格：测试段到底跑什么，全在这个字典里。"""
    return {
        "factor_set": fset, "k": k, "R": R, "capital": capital,
        "price_mode": PRICE_MODE, "segments": seg, "n_etf": n_etf,
        "min_bars": MIN_BARS,
        "cost_model": {
            "commission_rate": E.COMMISSION_RATE,
            "commission_min_yuan": E.COMMISSION_MIN,
            "stamp_duty": E.STAMP_DUTY,
            "transfer_fee": E.TRANSFER_FEE,
            "slippage_bp_edge": E.SLIPPAGE_BP_EDGE,
        },
    }


def spec_hash(spec: dict[str, Any]) -> str:
    payload = {kk: vv for kk, vv in spec.items() if kk not in _META_KEYS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bench(panel: dict[str, Any], R: int, lo: int, hi: int,
           periods: list[dict[str, Any]]) -> dict[str, Any]:
    """基准，且与策略对齐到同一批期次。"""
    return E.equal_weight_benchmark(
        panel, R, lo, hi, PRICE_MODE,
        dates_filter=[p["date"] for p in periods])


def run_grid(panel: dict[str, Any], scores: dict[str, np.ndarray],
             seg: dict[str, tuple[int, int]], capital: float) -> list[dict[str, Any]]:
    """在给定区间上跑全部 (因子集, k, R) 组合。"""
    out = []
    lo, hi = seg["train"]
    for fset, (keys, signs) in FACTOR_SETS.items():
        sc = scores[fset]
        for k in K_SET:
            for R in R_SET:
                res = E.run_strategy(panel, sc, k, R, price_mode=PRICE_MODE,
                                     capital=capital, lo=lo, hi=hi)
                if not res["periods"]:
                    continue
                m = E.metrics(res["periods"], R)
                bench = _bench(panel, R, lo, hi, res["periods"])
                out.append({
                    "factor_set": fset, "k": k, "R": R, "capital": capital,
                    "train": m, "bench_train": bench,
                    "excess_train_ann": (m["net_annualized"]
                                         - bench.get("annualized", float("nan"))),
                })
    return out


def main() -> None:
    # 数据质量门禁：审计未通过的品种直接剔除，不进策略
    excluded: list[str] = []
    dq_path = os.path.join(E.OUT_DIR, "etf_dataquality.json")
    if os.path.exists(dq_path):
        dq = json.load(open(dq_path, encoding="utf-8"))
        # ── 失败关闭：致命问题 = 整块面板不可信 ──────────────────
        # 不能「剔除掉坏的就当没事」—— 被剔掉的往往正是主流宽基/行业 ETF
        # （实测纳指ETF / 医药ETF / 新能源车ETF…），剔除会让 ETF 池产生
        # 系统性选择偏差，而这种偏差不会在结果里留下任何痕迹。
        fatal = [r for r in dq["results"] if r.get("fatal_issues")]
        if fatal:
            codes = ", ".join(r["code"] for r in fatal[:10])
            raise RuntimeError(
                f"ETF 数据质量门禁有 {len(fatal)} 只致命失败（{codes}"
                f"{'...' if len(fatal) > 10 else ''}）—— 整块 ETF 面板不可信，"
                f"研究失败关闭。先修 etf_adj.py 的复权，不要用「剔除坏品种」掩盖。")
        for r in dq["results"]:
            if not r["ok"]:
                excluded.append(r["code"])
    else:
        raise RuntimeError("缺少 etf_dataquality.json —— 必须先过数据质量门禁")
    ex_set = set(excluded)
    panel = E.load_panel(min_bars=MIN_BARS, exclude=ex_set)
    T = len(panel["dates"])
    seg = E.split(T)
    print(f"ETF 面板 {panel['close'].shape}  {panel['dates'][0]} → {panel['dates'][-1]}")
    print(f"质量门禁剔除 {len(excluded)} 只: {excluded}")
    print(f"分段 train {seg['train']} val {seg['val']} test {seg['test']}")

    factors = E.compute_factors(panel["close"])
    scores = {name: E.composite_score(factors, keys, signs)
              for name, (keys, signs) in FACTOR_SETS.items()}

    # ── 第一步：训练段选参（唯一选参来源）──────────────────────
    train_grid = run_grid(panel, scores, seg, capital=TRAIN_CAPITAL)
    train_grid.sort(key=lambda r: -r["excess_train_ann"])
    top = train_grid[:TOP_N_TRAIN]
    print(f"\n训练段前 {len(top)} 名（净超额年化）：")
    for r in top:
        print(f"  {r['factor_set']:<12} k={r['k']} R={r['R']:>2}  "
              f"净年化 {r['train']['net_annualized']*100:+7.2f}%  "
              f"基准 {r['bench_train'].get('annualized', float('nan'))*100:+7.2f}%  "
              f"超额 {r['excess_train_ann']*100:+7.2f}%")

    # 选参结果在此**锁定**。后面验证段只做通过/不通过判定，不再改参。
    chosen = top[0]
    chosen_by = "train_excess_ann"
    fset, k, R = chosen["factor_set"], chosen["k"], chosen["R"]
    print(f"\n📌 训练段选定（锁定）：factor_set={fset} k={k} R={R}  "
          f"训练段超额 {chosen['excess_train_ann']*100:+.2f}%")

    # ── 第二步：验证段确认（只判定通过与否，不改参）────────────
    val_rows = []
    for r in top:
        f2, k2, R2 = r["factor_set"], r["k"], r["R"]
        res = E.run_strategy(panel, scores[f2], k2, R2, price_mode=PRICE_MODE,
                             capital=TRAIN_CAPITAL,
                             lo=seg["val"][0], hi=seg["val"][1])
        if not res["periods"]:
            continue
        mv = E.metrics(res["periods"], R2)
        bv = _bench(panel, R2, seg["val"][0], seg["val"][1], res["periods"])
        exc = mv["net_annualized"] - bv.get("annualized", float("nan"))
        val_rows.append({**r, "val": mv, "bench_val": bv, "excess_val_ann": exc})
    val_rows.sort(key=lambda r: -r["excess_val_ann"])
    print("\n验证段（诊断用；判定只看锁定的那一个）：")
    for r in val_rows:
        mark = "  ← 锁定参数" if (r["factor_set"], r["k"], r["R"]) == (fset, k, R) else ""
        print(f"  {r['factor_set']:<12} k={r['k']} R={r['R']:>2}  "
              f"超额 {r['excess_val_ann']*100:+7.2f}%{mark}")

    chosen_val = next(r for r in val_rows
                      if (r["factor_set"], r["k"], r["R"]) == (fset, k, R))
    if chosen_val["excess_val_ann"] <= 0:
        print("\n❌ 训练段选出的参数在验证段净超额不为正 —— 停止。"
              "不换参、不跑测试段（换参就是用验证段选参 = 泄漏）。")
        out = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "stage": "validation_failed",
            "n_etf": panel["close"].shape[1],
            "chosen": {"factor_set": fset, "k": k, "R": R},
            "selected_by": chosen_by,
            "train_grid_top": top,
            "val_rows": val_rows,
            "test": None,
        }
        os.makedirs(E.OUT_DIR, exist_ok=True)
        json.dump(out, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"已写 {OUT_PATH}")
        return

    print(f"\n✅ 验证通过：锁定参数在验证段净超额 "
          f"{chosen_val['excess_val_ann']*100:+.2f}%")

    # ── 第三步：冻结规格（先落盘，再跑测试段）──────────────────
    spec = build_spec(fset, k, R, TRAIN_CAPITAL, seg, panel["close"].shape[1])
    spec["spec_hash"] = spec_hash(spec)
    spec["frozen_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    spec["selected_by"] = chosen_by
    spec["validation_excess_ann"] = chosen_val["excess_val_ann"]
    os.makedirs(E.OUT_DIR, exist_ok=True)
    with open(FROZEN_PATH, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=1)
    print(f"🔒 规格已冻结 → {FROZEN_PATH}  hash={spec['spec_hash'][:12]}")

    # ── 第四步：测试段只跑这一次 ───────────────────────────────
    # 从磁盘读回冻结件，确认测试段跑的就是冻结的那一份（防中途改参）
    frozen = json.load(open(FROZEN_PATH, encoding="utf-8"))
    assert frozen["spec_hash"] == spec_hash(frozen), \
        "冻结规格在落盘后被改动 —— 本次测试段结果已失效"
    fset, k, R = frozen["factor_set"], frozen["k"], frozen["R"]
    print(f"  测试段使用冻结件：factor_set={fset} k={k} R={R}")

    res_t = E.run_strategy(panel, scores[fset], k, R, price_mode=PRICE_MODE,
                           capital=frozen["capital"],
                           lo=seg["test"][0], hi=seg["test"][1])
    mt = E.metrics(res_t["periods"], R)
    bt = _bench(panel, R, seg["test"][0], seg["test"][1], res_t["periods"])
    exc_t = mt["net_annualized"] - bt.get("annualized", float("nan"))
    print("\n【测试段 · 只跑这一次】")
    print(f"  净年化 {mt['net_annualized']*100:+.2f}%  基准 {bt.get('annualized', float('nan'))*100:+.2f}%  "
          f"超额 {exc_t*100:+.2f}%  Sharpe {mt['sharpe']:.3f}")
    print(f"  中位数口径年化 {mt['net_median_annualized']*100:+.2f}%  缩尾均值 {mt['net_winsor_mean_annualized']*100:+.2f}%  "
          f"胜率 {mt['win_rate']*100:.1f}%  期数 {mt['n_periods']}")
    print(f"  成本 {mt['mean_cost_per_period_bp']:.2f}bp/期  毛 {mt['gross_per_period']*1e4:.2f}bp/期  "
          f"最大回撤 {mt['max_drawdown_ann']*100:.2f}%")

    # ── 资金规模敏感化（ETF 的 ¥5 最低佣金是非线性的）──────────
    cap_sweep = []
    for cap in CAPITALS:
        rs = E.run_strategy(panel, scores[fset], k, R, price_mode=PRICE_MODE,
                            capital=cap, lo=seg["test"][0], hi=seg["test"][1])
        ms = E.metrics(rs["periods"], R)
        ms["cost_floor"] = cost_floor_table(cap, k)
        cap_sweep.append({"capital": cap, **ms})
        print(f"  资金 {cap/1e4:>7.0f}万 → 净年化 {ms['net_annualized']*100:+7.2f}%  "
              f"成本 {ms['mean_cost_per_period_bp']:.2f}bp/期  "
              f"地板 {ms['cost_floor']['total_bp_roundtrip']:.2f}bp/单")

    out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "stage": "completed",
        "n_etf": panel["close"].shape[1],
        "etf_names": panel["names"],
        "panel": {"T": T, "first": panel["dates"][0], "last": panel["dates"][-1]},
        "segments": seg,
        "price_mode": PRICE_MODE,
        "cost_model": {
            "commission_rate": E.COMMISSION_RATE, "commission_min_yuan": E.COMMISSION_MIN,
            "stamp_duty": E.STAMP_DUTY, "transfer_fee": E.TRANSFER_FEE,
            "slippage_bp_edge": E.SLIPPAGE_BP_EDGE,
            "source": "国新证券《证券交易佣金收取标准》/ 中金财富佣金公示：场内基金无印花税、无过户费，只收佣金",
        },
        "selection_rule": (
            "选参唯一来源=训练段净超额年化最大；验证段只做通过/不通过判定，"
            "不参与选参；冻结件先落盘并带 spec_hash，测试段从磁盘读回校验后只跑一次。"
        ),
        "selected_by": chosen_by,
        "chosen": {"factor_set": fset, "k": k, "R": R},
        "frozen_spec": spec,
        "train_grid_top": top,
        "val_rows": val_rows,
        "test": {"metrics": mt, "benchmark": bt, "excess_ann": exc_t},
        "capital_sweep": cap_sweep,
        "survivorship_bias_disclosure": (
            "ETF 池为当前存续活跃 ETF 名单，存在幸存者偏差；池内成员按上市日点时生效，"
            "不做向前回填。ETF 标的是宽基/行业指数，死亡率远低于个股，但偏差未消除。"
        ),
        "superseded_runs": [{
            "date": "2026-11-14",
            "issue": "旧实现按验证段超额选参（passed[0]），验证段被当第二个训练段",
            "chosen_then": {"factor_set": "ma_only", "k": 3, "R": 5},
            "train_excess_ann": -0.119128,
            "val_excess_ann": 0.14715,
            "test_excess_ann": -0.123797,
            "artifact": "data/research/etf_channel_INVALID_val_leak_20261114.json",
            "note": "该次运行不作数；其测试段数字不得被引用为策略表现",
        }],
    }
    json.dump(out, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n已写 {OUT_PATH}")


if __name__ == "__main__":
    main()
