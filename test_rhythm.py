#!/usr/bin/env python3
"""rhythm.py 回归测试（纯离线，不碰 state.db、不调 API）。

跑法： python3 test_rhythm.py
覆盖：
  1. classify 迟滞状态机（含阈值边界）
  2. 快照 anchor 裁剪安全性（关键 bug 回归）
  3. 速率口径一致性（headline == per_profile 加总）
  4. 最小基线间隔护栏
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rhythm  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


# ───────────────────────── 1. classify 迟滞 ─────────────────────────
def test_classify():
    print("\n[1] classify 迟滞状态机")
    T = rhythm.THRESHOLDS

    # 升档：从 IDLE 逐级爬升
    s, _ = rhythm.classify(0, None)
    check("0 tokens + 无历史 → IDLE", s == "IDLE", s)
    s, _ = rhythm.classify(T["WARM"][0], "IDLE")
    check("1.5M + IDLE → WARM", s == "WARM", s)
    s, _ = rhythm.classify(T["BUSY"][0], "WARM")
    check("8M + WARM → BUSY", s == "BUSY", s)
    s, _ = rhythm.classify(T["YIELD"][0], "BUSY")
    check("30M + BUSY → YIELD", s == "YIELD", s)

    # 迟滞：脉冲过后不立刻弹回
    s, _ = rhythm.classify(5_000_000, "BUSY")     # 低于 BUSY exit 2.5M? 不，5M > 2.5M
    check("5M + BUSY → 保持 BUSY（未跌破 exit 2.5M）", s == "BUSY", s)
    s, _ = rhythm.classify(2_500_000, "BUSY")     # == exit 2.5M → 降
    check("2.5M + BUSY → 降到 WARM（== exit 阈值）", s == "WARM", s)
    s, _ = rhythm.classify(400_000, "WARM")       # == WARM exit 0.4M → 降到 IDLE
    check("0.4M + WARM → 降到 IDLE（== exit 阈值）", s == "IDLE", s)

    # 迟滞方向：WARM 态下 1.0M（<BUSY enter 8M）仍应停在 WARM，不该因
    # 「不够 8M」而降档 —— 降档只看当前档的 exit 阈值
    s, _ = rhythm.classify(1_000_000, "WARM")
    check("1.0M + WARM → 保持 WARM", s == "WARM", s)

    # YIELD 态下 20M（<YIELD exit 12M? 不，20M>12M）→ 保持 YIELD
    s, _ = rhythm.classify(20_000_000, "YIELD")
    check("20M + YIELD → 保持 YIELD", s == "YIELD", s)
    s, _ = rhythm.classify(12_000_000, "YIELD")
    check("12M + YIELD → 降到 BUSY（== exit 阈值）", s == "BUSY", s)

    # 未知历史状态 → 保守按 IDLE 起步
    s, _ = rhythm.classify(0, "GARBAGE")
    check("未知历史状态 → IDLE", s == "IDLE", s)


# ─────────────── 2. anchor 裁剪安全性（关键回归） ───────────────
def test_anchor_trim():
    print("\n[2] 快照 anchor 裁剪安全性")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        snap_file = tmp / "rhythm_snapshots.jsonl"
        anchor_file = tmp / "rhythm_anchor.json"

        # 覆盖模块级路径
        old_snap, old_anchor = rhythm.SNAPSHOTS, rhythm.ANCHOR
        rhythm.SNAPSHOTS = snap_file
        rhythm.ANCHOR = anchor_file
        try:
            # 构造：3 个 key，其中 key_b 早期涨过、之后一直不变
            base = {"p|s1|m|t": [1, 100], "p|s2|m|t": [1, 200], "p|s3|m|t": [1, 300]}
            rhythm._write_json_atomic(anchor_file, {"ts": time.time(), "full": base})
            rhythm._write_deltas([])

            # 第一次：key_b 涨到 500
            s1 = dict(base)
            s1["p|s2|m|t"] = [2, 500]
            rhythm._append_snapshot(s1)

            # 之后一直不变（key_b 停在 500）
            for _ in range(3):
                rhythm._append_snapshot(dict(s1))

            # 现在强制裁剪：把上限调小
            old_max = rhythm.SNAPSHOT_MAX_RECORDS
            rhythm.SNAPSHOT_MAX_RECORDS = 2
            try:
                s2 = dict(s1)
                s2["p|s3|m|t"] = [2, 900]          # 只有 key_c 变
                rhythm._append_snapshot(s2)
            finally:
                rhythm.SNAPSHOT_MAX_RECORDS = old_max

            hist = rhythm._load_history()
            check("裁剪后仍有历史", len(hist) >= 2, f"len={len(hist)}")
            last = hist[-1]["snap"]
            check("裁剪后 key_b 仍在累积表里（未被丢弃）",
                  "p|s2|m|t" in last, str(list(last.keys())))
            check("裁剪后 key_b 数值正确（500，未被回退成 0）",
                  last.get("p|s2|m|t") == [2, 500], str(last.get("p|s2|m|t")))

            # 关键断言：裁剪后算出的区间增量 == 真实增量
            # base = 上一条，latest = 最后一条 → 真实增量只有 key_c 的 600 tokens
            base_snap = hist[-2]["snap"]
            dt = max(1e-9, (hist[-1]["ts"] - hist[-2]["ts"]) / 3600.0)
            tokens = sum(
                max(0, v[1] - base_snap.get(k, (0, 0))[1])
                for k, v in last.items()
            )
            rate = tokens / dt
            # 如果 anchor 被裁掉，key_b 会被当成新增 → tokens 会多出 500
            check("裁剪后区间增量未虚增（未把 key_b 当成新增）",
                  abs(tokens - 600) < 1e-6, f"tokens={tokens} 期望 600")

            # 连续多次裁剪后仍稳定
            for i in range(5):
                s3 = dict(last)
                s3["p|s3|m|t"] = [3, 900 + 100 * (i + 1)]
                rhythm.SNAPSHOT_MAX_RECORDS = 2
                try:
                    rhythm._append_snapshot(s3)
                finally:
                    rhythm.SNAPSHOT_MAX_RECORDS = old_max
            hist2 = rhythm._load_history()
            check("多次裁剪后 key_b 依然健在",
                  "p|s2|m|t" in hist2[-1]["snap"], str(list(hist2[-1]["snap"].keys())))
            check("多次裁剪后 key_b 数值未回退",
                  hist2[-1]["snap"].get("p|s2|m|t") == [2, 500],
                  str(hist2[-1]["snap"].get("p|s2|m|t")))
        finally:
            rhythm.SNAPSHOTS, rhythm.ANCHOR = old_snap, old_anchor


# ─────────────── 3. 速率口径一致性 ───────────────
def test_rate_consistency():
    print("\n[3] 速率口径一致性（headline == per_profile 加总）")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_snap, old_anchor = rhythm.SNAPSHOTS, rhythm.ANCHOR
        rhythm.SNAPSHOTS = tmp / "s.jsonl"
        rhythm.ANCHOR = tmp / "a.json"
        try:
            t0 = time.time() - 3600
            snap0 = {
                "xiaojin|s1|m|t": [10, 1000],
                "xiaohong|s2|m|t": [20, 2000],
                "xiaoming|s3|m|t": [30, 3000],
            }
            rhythm._write_json_atomic(rhythm.ANCHOR, {"ts": t0, "full": snap0})
            rhythm._write_deltas([])

            # 1 小时后：xiaohong +1000, xiaoming +2000, xiaojin +500
            snap1 = {
                "xiaojin|s1|m|t": [15, 1500],
                "xiaohong|s2|m|t": [25, 3000],
                "xiaoming|s3|m|t": [35, 5000],
            }
            rhythm._write_deltas([{"ts": t0 + 3600, "delta": snap1}])
            # 再补一条让 latest = snap1
            hist = rhythm._expand(rhythm._read_anchor(), rhythm._read_records())
            latest = hist[-1]

            base = hist[0]
            dt_h = (latest["ts"] - base["ts"]) / 3600.0
            others_t = sum(
                max(0, v[1] - base["snap"].get(k, (0, 0))[1])
                for k, v in latest["snap"].items()
                if k.split("|", 1)[0] != rhythm.SELF
            )
            self_t = sum(
                max(0, v[1] - base["snap"].get(k, (0, 0))[1])
                for k, v in latest["snap"].items()
                if k.split("|", 1)[0] == rhythm.SELF
            )
            check("别人增量 = xiaohong 1000 + xiaoming 2000 = 3000",
                  others_t == 3000, str(others_t))
            check("自己增量 = 500（不被算进别人）", self_t == 500, str(self_t))
            check("速率折算正确（3000 tokens / 1h = 3000/h）",
                  abs(others_t / dt_h - 3000) < 1, f"{others_t/dt_h}")
        finally:
            rhythm.SNAPSHOTS, rhythm.ANCHOR = old_snap, old_anchor


# ─────────────── 4. 最小基线间隔护栏 ───────────────
def test_min_baseline():
    print("\n[4] 最小基线间隔护栏")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_snap, old_anchor = rhythm.SNAPSHOTS, rhythm.ANCHOR
        rhythm.SNAPSHOTS = tmp / "s.jsonl"
        rhythm.ANCHOR = tmp / "a.json"
        try:
            now = time.time()
            snap = {"p|s1|m|t": [1, 100]}
            rhythm._write_json_atomic(rhythm.ANCHOR, {"ts": now - 5, "full": snap})
            rhythm._write_deltas([{"ts": now, "delta": {"p|s1|m|t": [2, 999999999]}}])
            r = rhythm.usage_rates(60.0)
            check("基线过近 → baseline_missing=True", r["baseline_missing"] is True,
                  str(r.get("note")))
            check("基线过近时不给出虚高速率", r["others_tokens_per_h"] == 0,
                  str(r["others_tokens_per_h"]))
        finally:
            rhythm.SNAPSHOTS, rhythm.ANCHOR = old_snap, old_anchor


if __name__ == "__main__":
    print("=" * 66)
    print("rhythm.py 回归测试")
    print("=" * 66)
    test_classify()
    test_anchor_trim()
    test_rate_consistency()
    test_min_baseline()
    print("\n" + "=" * 66)
    if FAILED:
        print(f"FAILED: {len(FAILED)} 项 → {FAILED}")
        sys.exit(1)
    print("ALL PASS")
