#!/usr/bin/env python
"""研究队列 —— 空闲自动续跑驱动。

设计原则（2026-11-10 立规）：
1. 每个步骤独立、幂等、可单独重试；一步失败不阻塞后面的步骤。
2. 失败必须留痕：写进 state/research_queue.json，不静默丢弃。
3. 步骤之间不并发写同一输出文件：每个步骤用独立的时间戳日志。
4. 只跑"到期"的步骤，避免空转浪费 CPU。
5. 全部是模拟盘/研究记录，不下真实订单。

步骤来源有两类，走的是同一套"到期判断 + 落盘 + 留痕"机制：
  · ooslab.py    —— 数据质量体检与样本外验证（纯本地计算）
  · daily_loop.py —— 每日 live 闭环（刷新行情 → 出 live 预测 → 用真实行情判定）

daily_loop 三个阶段之间有先后依赖（刷新 → 预测 → 判定），所以它们在
队列里是**一整步**（key=loop），内部仍由 daily_loop 自己保证顺序与失败关闭：
上游行情不可用时 refresh 判失败并保留旧数据，队列记失败、30 分钟后重试，
不会把"刷新失败"当成"刷新成功"往下走。

用法：
    python research_queue.py --list
    python research_queue.py --run           # 跑到当前所有到期步骤为止
    python research_queue.py --step oos     # 强制跑某一步
    python research_queue.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state" / "research_queue.json"
LOGDIR = ROOT / "logs"
CST = timezone(timedelta(hours=8))

# 每个步骤：key, 标题, 最小重跑间隔(小时), 命令, 前置步骤(可选)
#
# min_hours 是"至少隔这么久才重跑"的下限，不是"到点就跑"：
#   · audit/oos 是纯本地计算，间隔短无所谓；
#   · refresh 要联网刷 1312 只标的 × 2 个源（东财 + 腾讯），一轮十几分钟，
#     而且请求密度高了会被上游掐（连不上 → 整轮判失败）。
#     行情每天收盘后才变一次，所以给 12 小时 —— 一天两轮足够，
#     失败时靠 retry_after_at 兜底重试，不硬刷上游。
#
# requires 是"前置步骤必须成功"：刷新失败时不能接着出预测/做判定，
# 否则就是拿旧行情冒充今天的闭环（失败关闭）。
STEPS: list[dict] = [
    {
        "key": "refresh",
        "title": "每日闭环 · 增量刷新后复权日线（双源交叉校验）",
        "min_hours": 12.0,
        "cmd": [sys.executable, str(ROOT / "daily_loop.py"), "--stage", "refresh"],
    },
    {
        "key": "predict",
        "title": "每日闭环 · 生成当天 live 预测（落盘即不可改写）",
        "min_hours": 6.0,
        "requires": "refresh",
        "cmd": [sys.executable, str(ROOT / "daily_loop.py"), "--stage", "predict"],
    },
    {
        "key": "judge",
        "title": "每日闭环 · 用真实行情判定到期的预测",
        "min_hours": 6.0,
        "requires": "predict",
        "cmd": [sys.executable, str(ROOT / "daily_loop.py"), "--stage", "judge"],
    },
    {
        "key": "audit",
        "title": "行情数据质量体检（后复权口径）",
        "min_hours": 6.0,
        "cmd": [sys.executable, str(ROOT / "ooslab.py"), "--audit-only"],
    },
    {
        "key": "oos",
        "title": "样本外验证实验室（成本后 / 时间外 / 概率校准）",
        "min_hours": 12.0,
        "cmd": [sys.executable, str(ROOT / "ooslab.py")],
    },
    {
        "key": "realism",
        "title": "成交真实性检验（收盘 / 次日开盘 / 涨停不可买 三口径）",
        "min_hours": 12.0,
        "requires": "oos",
        "cmd": [sys.executable, str(ROOT / "execution_realism.py")],
    },
    {
        "key": "anatomy",
        "title": "信号解剖（把超额按时间段拆开，定位机制与不可交易段）",
        "min_hours": 12.0,
        "requires": "realism",
        "cmd": [sys.executable, str(ROOT / "signal_anatomy.py")],
    },
    {
        # ETF 通道有自己的数据底座（复权、份额折算），质量门禁独立于个股通道，
        # 所以单列一步。etf_research.py 自己会**失败关闭**：质量门禁出现致命
        # 问题（假跳空等）时直接抛错，不允许「剔除坏品种后照跑」。
        "key": "etf_audit",
        "title": "ETF 面板数据质量 + 复权质量门禁（含单日跳变硬门禁）",
        "min_hours": 12.0,
        "cmd": [sys.executable, str(ROOT / "validate_etf.py")],
    },
    {
        "key": "etf",
        "title": "ETF 低成本通道研究（训练选参 → 验证确认 → 冻结 → 测试只跑一次）",
        "min_hours": 12.0,
        "requires": "etf_audit",
        "cmd": [sys.executable, str(ROOT / "etf_research.py")],
    },
]

# ── 节奏闸门（宋老师 2026-11-11 立规）────────────────────────────────────
# 「只要minimax没有被用…你就干自己的活，等别人也开始用了，你可以慢慢停下来」
#
# 重要：这些步骤都是**纯 Python 脚本，零 LLM 调用**（实测 ooslab.py、daily_loop.py
# 等全项目 0 处 LLM 调用），所以它们**不受配额影响，任何时候都该跑**。
# 闸门只用来决定「是否启动额外的、烧 token 的深度分析工作」。
#
# 判定：读 state/rhythm.json（由 rhythm.py 的 cron 每 15 分钟刷新）。
#   IDLE  → 允许深度分析
#   WARM  → 允许，但收紧单次预算
#   BUSY  → 只跑零 token 的活
#   YIELD → 只跑零 token 的活，等窗口过去
RHYTHM_STATE = ROOT / "state" / "rhythm.json"


def rhythm_state() -> str:
    """读当前节奏档位。读不到就保守当作 BUSY（宁可少烧，不可乱烧）。"""
    try:
        data = json.loads(RHYTHM_STATE.read_text(encoding="utf-8"))
        st = data.get("state")
        if st in ("IDLE", "WARM", "BUSY", "YIELD"):
            return st
    except Exception:  # noqa: BLE001
        pass
    return "BUSY"


def deep_work_allowed() -> tuple[bool, str]:
    """是否可以启动烧 token 的深度分析。返回 (允许, 理由)。"""
    st = rhythm_state()
    if st == "IDLE":
        return True, "节奏 IDLE：配额基本没人用，放开深度分析"
    if st == "WARM":
        return True, "节奏 WARM：轻度占用，允许但收紧单次预算"
    return False, f"节奏 {st}：别人在用配额，只跑零 token 的活，让路"


def now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"steps": {}, "created_at": now_iso()}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    st["updated_at"] = now_iso()
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)          # 原子写，避免读到半截 JSON


def hours_since(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(ts)
    except Exception:
        return None
    return (datetime.now(CST) - t).total_seconds() / 3600.0


def due(step: dict, st: dict) -> tuple[bool, str]:
    """返回 (是否到期, 原因)。"""
    rec = st.get("steps", {}).get(step["key"], {})
    if rec.get("last_status") == "running":
        h = hours_since(rec.get("started_at"))
        if h is not None and h < 6:
            return False, "上次仍在运行"

    # 前置步骤没成功 → 本步不跑。刷新失败时不能接着出预测：
    # 那是拿旧行情冒充今天的闭环，失败关闭。
    dep = step.get("requires")
    if dep:
        dep_rec = st.get("steps", {}).get(dep, {})
        dep_status = dep_rec.get("last_status")
        if dep_status != "ok":
            why = dep_rec.get("last_error") or "尚未成功运行"
            return False, f"前置步骤 {dep} 未成功（{why}）"

    # 失败后的冷却期：不要在上游故障时反复硬刷。
    retry_at = rec.get("retry_after_at")
    if retry_at:
        try:
            t = datetime.fromisoformat(retry_at)
        except (TypeError, ValueError):
            t = None
        if t is not None:
            if t.tzinfo is None:
                t = t.replace(tzinfo=CST)
            if datetime.now(CST) < t:
                return False, f"失败冷却中，{retry_at} 后再试"

    h = hours_since(rec.get("last_ok_at"))
    if h is None:
        return True, "从未成功跑过"
    if h >= step["min_hours"]:
        return True, "距上次成功 %.1f 小时（阈值 %.0f 小时）" % (h, step["min_hours"])
    return False, "距上次成功 %.1f 小时，未到 %.0f 小时阈值" % (h, step["min_hours"])


def run_step(step: dict, st: dict) -> dict:
    key = step["key"]
    rec = st.setdefault("steps", {}).setdefault(key, {})
    LOGDIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
    log = LOGDIR / ("queue_%s_%s.log" % (key, stamp))
    rec.update({
        "title": step["title"],
        "last_status": "running",
        "started_at": now_iso(),
        "last_log": str(log),
    })
    save_state(st)

    t0 = time.time()
    try:
        proc = subprocess.run(
            step["cmd"], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=3600, text=True,
        )
        out = proc.stdout or ""
        log.write_text(out, encoding="utf-8")
        rc = proc.returncode
        rec["last_exit_code"] = rc
        rec["duration_sec"] = round(time.time() - t0, 1)
        if rc == 0:
            rec["last_status"] = "ok"
            rec["last_ok_at"] = now_iso()
            rec["last_error"] = None
            # 成功了就清掉冷却期，否则上一次失败的 retry_after_at 会一直挡着。
            rec.pop("retry_after_at", None)
        else:
            rec["last_status"] = "failed"
            rec["last_error"] = "exit %d，日志 %s" % (rc, log)
            # 失败后 30 分钟内不重试，避免坏数据把机器跑满
            rec["retry_after_at"] = (
                datetime.now(CST) + timedelta(minutes=30)).isoformat(timespec="seconds")
    except subprocess.TimeoutExpired:
        log.write_text("TIMEOUT after 3600s\n", encoding="utf-8")
        rec["last_status"] = "failed"
        rec["last_error"] = "超时 3600s"
        rec["duration_sec"] = round(time.time() - t0, 1)
        rec["retry_after_at"] = (
            datetime.now(CST) + timedelta(minutes=30)).isoformat(timespec="seconds")
    except Exception as exc:                      # noqa: BLE001
        log.write_text("EXCEPTION: %r\n" % (exc,), encoding="utf-8")
        rec["last_status"] = "failed"
        rec["last_error"] = repr(exc)
        rec["retry_after_at"] = (
            datetime.now(CST) + timedelta(minutes=30)).isoformat(timespec="seconds")
    rec["last_run_at"] = now_iso()
    save_state(st)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="跑所有到期步骤")
    ap.add_argument("--step", help="强制跑指定步骤")
    ap.add_argument("--list", action="store_true", help="列出步骤与到期状态")
    ap.add_argument("--status", action="store_true", help="打印队列状态")
    args = ap.parse_args()

    st = load_state()

    if args.status:
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return 0

    if args.list:
        allowed, why = deep_work_allowed()
        print(f"节奏档位: {rhythm_state()}   深度分析(烧token): {'允许' if allowed else '暂停'}")
        print(f"  → {why}")
        print()
        for s in STEPS:
            ok, why_s = due(s, st)
            print("%-6s %-8s %s" % (s["key"], "到期" if ok else "未到期", why_s))
        return 0

    if args.step:
        for s in STEPS:
            if s["key"] == args.step:
                rec = run_step(s, st)
                print("%s: %s %s" % (s["key"], rec["last_status"], rec.get("last_error") or ""))
                return 0 if rec["last_status"] == "ok" else 1
        print("未知步骤: %s" % args.step, file=sys.stderr)
        return 2

    if args.run:
        ran = 0
        for s in STEPS:
            ok, why = due(s, st)
            if not ok:
                print("跳过 %s —— %s" % (s["key"], why))
                continue
            print("执行 %s —— %s" % (s["key"], s["title"]))
            rec = run_step(s, st)
            ran += 1
            print("  → %s %s" % (rec["last_status"], rec.get("last_error") or ""))
        if not ran:
            print("没有到期步骤")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
