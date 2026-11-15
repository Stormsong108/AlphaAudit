#!/usr/bin/env python
"""节奏闸门 —— 宋老师立规（2026-11-11）

> 「只要minimax没有被用，或者一定时间内，用的很少，你就干自己的活，
>   等别人也开始用了，你可以慢慢停下来，然后周而复始，你按照自己的想法去做」

## 为什么要做成代码

这条规则如果只记在脑子里，每次会话都要重新判断、容易拍脑袋。
本模块把它变成一个**确定性状态机**：读真实用量 → 判定状态 → 落盘 → 供调用方查。

## 判定输入（全部来自真实数据，无估算）

1. **MiniMax 配额**：直接调 `/v1/token_plan/remains` 拿 5h 窗口与本周剩余百分比。
   ⚠️ `current_interval_remaining_percent` 是**剩余 %**，已用 % = 100 - remaining。
2. **别人用量**：遍历所有 profile 的 `state.db`，读 `session_model_usage`
   （累计计数，按 `last_seen` 归窗），排除自己（xiaojin）。
3. **配额↔token 换算**：由本窗口实测得出「1% 配额 ≈ 1.91M tokens」，
   用于把别人的 token 用量翻译成「占了我多少配额」。

## 关键设计：脉冲 vs 持续

实测别人（小红）的用量是**脉冲式**的 —— 24h 内 249M tokens 里 211M 集中在 02:00
一个 agent 模式 cron 脉冲。所以：

- 单看瞬时窗口会误判（脉冲一结束就以为没人用）
- 必须用**滑动窗口 + 迟滞阈值**（进入严苛档的阈值高于退出档），避免来回抖动

## 四档状态

| 状态 | 含义 | 我该干什么 |
|------|------|-----------|
| `IDLE` | 别人基本没在用 | **全力干自己的活**（含烧 token 的深度分析） |
| `WARM` | 别人轻度使用 | 正常干活，但控制单次任务的 token 预算 |
| `BUSY` | 别人中度使用 | 优先跑零 token 的活（数据管线/研究队列），深度分析降频 |
| `YIELD` | 别人重度使用 | **让路**：只跑零 token 的活，等窗口过去 |

## 铁律：零 token 的活永不停止

`market_daily` 全项目 0 处 LLM 调用（实测确认）。数据采集、渲染、回测、因子研究
都是纯确定性脚本 —— 这些**不受配额影响，任何时候都继续跑**。
只有烧 MiniMax 配额的活（我的推理、深度分析、子 agent）才需要让路。

## 用法

    python rhythm.py status      # 打印当前状态 + 依据
    python rhythm.py gate        # 供脚本调用：EXIT 0=可烧 token，EXIT 2=应让路
    python rhythm.py record <a>  # 记录一次 token 消耗决策（审计用）
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path("/Users/sst/.hermes")
PROJ = Path(__file__).resolve().parent
STATE_DIR = PROJ / "state"
STATE_FILE = STATE_DIR / "rhythm.json"
SNAPSHOTS = STATE_DIR / "rhythm_snapshots.jsonl"   # 增量日志（小），差分求速率用
ANCHOR = STATE_DIR / "rhythm_anchor.json"          # 全量锚点（大），增量链的起算基准
DECISIONS = STATE_DIR / "rhythm_decisions.jsonl"     # 状态变迁日志

SELF = "xiaojin"
CST = timezone(timedelta(hours=8))

# ── 阈值（全部按「别人每小时 token 用量」标定，来源见模块 docstring 的实测分布）──
# 别人小时用量实测：p50=1.70M  p75=18.21M  p90=202.08M  max=551.53M（近 7 天 46 个活跃小时）
#
# 用「占配额百分比」表达更直观。1% 配额 ≈ 1.91M tokens（5h 窗口实测换算），
# 所以 1M tokens ≈ 0.52% 配额。
#
# 迟滞设计：ENTER 阈值 > EXIT 阈值，脉冲过去后不会立刻反弹成 IDLE。
THRESHOLDS = {
    # 状态: (进入该档的阈值, 退出该档的阈值)  单位 = 别人最近 1 小时 token 数
    "YIELD": (30_000_000, 12_000_000),   # 别人 >=30M/h → 让路；降到 12M/h 以下才退出
    "BUSY":  (8_000_000,  2_500_000),   # 别人 >=8M/h  → 降频；降到 2.5M/h 以下才降档
    "WARM":  (1_500_000,  400_000),     # 别人 >=1.5M/h → 控预算；降到 0.4M/h 以下算空闲
}
ORDER = ["IDLE", "WARM", "BUSY", "YIELD"]

# ── 配额换算常数（本窗口实测：17.21M tokens ↔ 9% 已用）──
TOKENS_PER_QUOTA_PCT = 1_910_000

# 速率计量的最小基线间隔（秒）。低于这个跨度，折算成每小时会严重失真：
# 实测 5.5 秒的增量折算得出「572 次调用/小时」这种假数。
MIN_BASELINE_SEC = 300          # 5 分钟

# 快照文件的结构与体积控制。
# ⚠️ 铁律：增量链必须**永远从一条 anchor（全量快照）起算**。
# 如果把 anchor 当成普通行裁掉，_expand() 重建累计值时会漏掉
# 「自 anchor 以来从未变动的 key」，差分时 base 取到 (0,0)
# → 区间增量被算成累计总量 → 速率被放大成天文数字
# （实测 00:00 会报 78M，而那个小时真实只有 ~13M）。
# 所以：anchor 单独存一个文件（很少重写），增量日志只按条数裁剪，
# 永远保留 anchor → 增量 的完整起算链。
SNAPSHOT_MAX_RECORDS = 40      # 增量日志保留最近 40 条（15 分钟一次 ≈ 10 小时）


def _read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = ROOT / "profiles" / SELF / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def fetch_quota() -> dict:
    """调 MiniMax 官方接口拿真实配额。失败时返回 status=error，不编造数字。"""
    env = _read_env()
    key = env.get("MINIMAX_CN_API_KEY")
    if not key:
        return {"status": "error", "error": "缺少 MINIMAX_CN_API_KEY"}
    base = env.get("MINIMAX_CN_BASE_URL", "https://api.minimaxi.com/v1").rstrip("/")
    url = f"{base}/token_plan/remains"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    out = {}
    for m in body.get("model_remains", []):
        if m.get("model_name") != "general":
            continue
        rem = m.get("current_interval_remaining_percent")
        wrem = m.get("current_weekly_remaining_percent")
        out["interval_remaining_pct"] = rem
        out["interval_used_pct"] = (100 - rem) if isinstance(rem, (int, float)) else None
        out["weekly_remaining_pct"] = wrem
        out["weekly_used_pct"] = (100 - wrem) if isinstance(wrem, (int, float)) else None
        out["interval_reset_min"] = round(m.get("remains_time", 0) / 60000)
        out["interval_status"] = m.get("current_interval_status")
    out["status"] = "ok" if "interval_remaining_pct" in out else "error"
    if out["status"] == "error":
        out["error"] = "响应里没有 general 配额条目"
    return out


def _profile_dbs() -> list[tuple[str, Path]]:
    dbs = [("root", ROOT / "state.db")]
    pdir = ROOT / "profiles"
    if pdir.exists():
        for d in sorted(pdir.iterdir()):
            if d.is_dir() and (d / "state.db").exists():
                dbs.append((d.name, d / "state.db"))
    return dbs


def _collect_cumulative() -> dict:
    """抓取所有 profile 的累计用量快照。

    `session_model_usage` 是**累计计数**（api_call_count / tokens 都是累加值），
    不是事件流。直接按 last_seen 归窗会把长会话的全部历史算进当前窗口
    （实测：朴素法在 00:00 报出 78M，实际那个小时只有 ~12M）。
    所以必须靠**差分**：与上次快照相减，得到区间真实增量。

    ⚠️ 会话会被删除（session_reset / 清理策略），删掉时该行的累计值也一并消失。
    实测：9.4 小时窗口内一次批量删除带走 131 个会话、132.82M tokens。
    若把消失的 key 直接丢掉，累计总量会出现 **-132M 的负跳变**，
    该窗口的用量被算成 0（漏计）；若把消失的 key 当成新增，又会虚增。
    正确做法：**冻结**（保留最后一次已知的值），让账本单调不减 ——
    被删会话在窗口内的增量本来就无法观测，宁可少算也不能倒扣。
    """
    snap: dict[str, dict] = {}
    for name, db in _profile_dbs():
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = c.execute(
                """SELECT session_id, model, task, api_call_count,
                          input_tokens+output_tokens+cache_read_tokens
                     FROM session_model_usage
                    WHERE model LIKE '%MiniMax%'""",
            ).fetchall()
            c.close()
        except Exception:  # noqa: BLE001
            continue
        for sid, model, task, calls, tokens in rows:
            key = f"{name}|{sid}|{model}|{task}"
            snap[key] = [int(calls or 0), int(tokens or 0)]
    return snap


def _read_records() -> list[dict]:
    """读增量日志的全部记录行（跳过坏行）。"""
    if not SNAPSHOTS.exists():
        return []
    out: list[dict] = []
    for line in SNAPSHOTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def _read_anchor() -> dict | None:
    """读全量锚点。锚点是增量链的起算基准，缺失则无法可靠差分。"""
    if not ANCHOR.exists():
        return None
    try:
        a = json.loads(ANCHOR.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(a.get("full"), dict) or not a["full"]:
        return None
    return a


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _expand(anchor: dict | None, deltas: list[dict]) -> list[dict]:
    """把 [anchor, delta...] 还原成累积快照序列（用于差分）。

    anchor（{"ts":..,"full":{...}}）是**起算基准**；
    delta（{"ts":..,"delta":{key:[calls,tokens]}}）里的值是**绝对值**，
    未出现的 key 沿用上一条，所以增量链与历史顺序无关（可安全压缩）。

    ⚠️ 值可以是 null —— 表示「该 key 在本次快照中已消失」。
    消失的 key **保留最后一次已知值**（冻结），而不是从累积表里删掉。
    原因：会话被删除时其累计值一并消失，若直接删 key，
    累计总量会出现负跳变（实测 -132.82M），该窗口用量被漏计。
    冻结后账本单调不减，差分结果只会少算、不会倒扣。
    """
    out: list[dict] = []
    acc: dict[str, list] = dict(anchor["full"]) if anchor else {}
    if anchor:
        out.append({"ts": anchor["ts"], "snap": dict(acc)})
    for h in deltas:
        for key, val in (h.get("delta") or {}).items():
            if val is not None:
                acc[key] = val
            # val is None → 冻结：保持 acc[key] 原值不动
        out.append({"ts": h["ts"], "snap": dict(acc)})
    return out


def _load_history() -> list[dict]:
    """读历史，返回**累积形式**的 {ts, snap} 列表。

    ⚠️ 铁律：累积链必须从一条 **anchor（全量快照）** 起算。
    若 anchor 缺失，「自 anchor 起从未变动的 key」会从累积表里消失，
    差分时 base 取到 (0,0) → 区间增量被算成累计总量 → 速率放大成天文数字
    （实测会报 78M，而那个小时真实只有 ~13M）。

    锚点缺失时**返回空历史**（而不是拿残缺的增量链硬算）：
    调用方会看到 baseline_missing=True，走保守路径；
    同时 _append_snapshot() 会用当前全量重建锚点。
    """
    anchor = _read_anchor()
    if anchor is None:
        return []
    return _expand(anchor, _read_records())


def _write_deltas(recs: list[dict]) -> None:
    """原子写增量日志。"""
    tmp = SNAPSHOTS.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
        encoding="utf-8",
    )
    tmp.replace(SNAPSHOTS)          # 原子写


def _append_snapshot(snap: dict) -> list[dict]:
    """把本次累计快照落到历史，返回落盘后的累积历史。

    结构分工：
      - ANCHOR（大，~0.5MB）：全量锚点，增量链的起算基准。
      - SNAPSHOTS（小）：增量日志，每行只存「与上一条不同的 key」，值为绝对值。

    ⚠️ 裁剪安全性：增量日志从头部裁掉旧记录时，
    **必须把被裁掉部分的累积状态提升为新锚点**。
    因为 delta 存的是绝对值、只有「变动的 key」才出现，
    若不提升锚点，裁剪后重建的累积表会漏掉
    「自旧锚点起变动过、但在被裁区间内没再变动」的 key
    → 差分 base 取到 (0,0) → 速率放大成天文数字。
    早期实现用 `lines[-40:]` 硬裁（不提升锚点）正是这个 bug。
    """
    SNAPSHOTS.parent.mkdir(parents=True, exist_ok=True)
    anchor = _read_anchor()
    deltas = _read_records()

    if anchor is None:
        # 无锚点（首次运行 / 锚点丢失）→ 用当前全量重建锚点。
        # 历史不可信，清空增量日志，基线标记为缺失（保守处理）。
        new_anchor = {"ts": time.time(), "full": snap}
        _write_json_atomic(ANCHOR, new_anchor)
        _write_deltas([])
        return _expand(new_anchor, [])

    hist = _expand(anchor, deltas)
    prev = hist[-1]["snap"] if hist else {}

    # 增量 = 本次与上次不同的部分。
    # 消失的 key 用 None 标记，语义是「冻结最后已知值」（见 _expand），**不是归零** ——
    # 归零会让累计账本出现负跳变（实测 -132.82M），把被删会话的历史用量倒扣掉。
    delta: dict[str, list] = {}
    for key, val in snap.items():
        if prev.get(key) != val:
            delta[key] = val
    for key in prev:
        if key not in snap:
            delta[key] = None

    now = time.time()
    all_deltas = deltas + [{"ts": now, "delta": delta}]

    # 裁剪：超出条数上限时，把被裁掉部分的累积状态提升为新锚点。
    if len(all_deltas) > SNAPSHOT_MAX_RECORDS:
        keep_from = len(all_deltas) - SNAPSHOT_MAX_RECORDS
        full_hist = _expand(anchor, all_deltas)
        # full_hist[k] = 累计状态（已应用 deltas[0..k-1]）
        promoted = {"ts": all_deltas[keep_from - 1]["ts"],
                    "full": full_hist[keep_from]["snap"]}
        _write_json_atomic(ANCHOR, promoted)
        anchor = promoted
        all_deltas = all_deltas[keep_from:]

    _write_deltas(all_deltas)
    return _expand(anchor, all_deltas)


def usage_rates(minutes: float = 60.0) -> dict:
    """算「别人」与「我」的每小时平均速率。

    ⚠️ 口径铁律：**一个基线、一个窗口、一套差分**。
    历史实现里 headline（others_tokens_per_h）走 `_rate_from_snapshots()`、
    per_profile 走另一段独立差分，两边选基线的规则不同，
    结果出现「headline 14.43M/h 但 per_profile 加总只有 2.01M/h」的自相矛盾。
    现在统一在下面一次差分里出数，headline 与 breakdown 必然对得上。

    首次运行时没有历史基线，此时返回 baseline_missing=True，
    调用方应知道这个数不可靠（此时按保守策略处理）。
    """
    snap = _collect_cumulative()
    hist = _append_snapshot(snap)      # 含本次快照，且已原子落盘

    if len(hist) < 2:
        return {
            "baseline_missing": True,
            "others_calls_per_h": 0,
            "others_tokens_per_h": 0,
            "self_calls_per_h": 0,
            "self_tokens_per_h": 0,
            "per_profile": {},
            "note": "首次运行，无快照基线，速率不可靠；已写入基线，下次起准确",
        }

    latest = hist[-1]
    latest_ts = latest["ts"]
    target = latest_ts - minutes * 60

    # 选基线：**取 <= target 的最新一条**（即最贴近窗口起点的那条）。
    # 关键：必须与 headline 用同一条 base，否则两边口径会漂。
    base = None
    for h in hist:
        if h["ts"] <= target:
            base = h          # 不 break → 取最新的那一条
    if base is None:
        # 窗口起点之前没有快照 → 退而用最早的一条
        base = hist[0]

    span_sec = latest_ts - base["ts"]
    if span_sec < MIN_BASELINE_SEC:
        return {
            "baseline_missing": True,
            "others_calls_per_h": 0, "others_tokens_per_h": 0,
            "self_calls_per_h": 0, "self_tokens_per_h": 0,
            "per_profile": {},
            "note": f"快照跨度不足 {MIN_BASELINE_SEC}s，速率不可靠（需更长基线间隔）",
        }

    dt_h = span_sec / 3600.0
    base_snap = base["snap"]

    self_c = self_t = 0
    others_c = others_t = 0
    per: dict[str, dict] = {}
    for key, (c_now, t_now) in latest["snap"].items():
        prof = key.split("|", 1)[0]
        c_old, t_old = base_snap.get(key, (0, 0))
        dc, dt = max(0, c_now - c_old), max(0, t_now - t_old)
        if not dc and not dt:
            continue
        if prof == SELF:
            self_c += dc
            self_t += dt
        else:
            others_c += dc
            others_t += dt
            p = per.setdefault(prof, {"calls": 0, "tokens": 0})
            p["calls"] += dc
            p["tokens"] += dt

    for v in per.values():
        v["calls"] = int(v["calls"] / dt_h)
        v["tokens"] = int(v["tokens"] / dt_h)

    return {
        "baseline_missing": False,
        "others_calls_per_h": int(others_c / dt_h),
        "others_tokens_per_h": int(others_t / dt_h),
        "self_calls_per_h": int(self_c / dt_h),
        "self_tokens_per_h": int(self_t / dt_h),
        "per_profile": per,
        "window_minutes": minutes,
        "window_span_sec": round(span_sec, 1),
        "base_ts": base["ts"],
        "latest_ts": latest_ts,
    }


def classify(others_1h_tokens: int, prev_state: str | None) -> tuple[str, str]:
    """带迟滞的状态判定。返回 (状态, 依据说明)。

    迟滞：从严进、从宽出 —— 别人脉冲结束后不会立刻弹回 IDLE 去猛烧 token，
    要等用量确实回落一段才恢复。这避免了在脉冲边缘反复横跳。
    """
    cur = prev_state if prev_state in ORDER else "IDLE"
    idx = ORDER.index(cur)

    # 降档：用**当前档**的 EXIT 阈值（退出当前档的条件），而不是下一档的。
    # 语义：THRESHOLDS[state] = (进入该档的阈值, 退出该档的阈值)。
    # 从 YIELD 退到 BUSY 的条件是「跌破 YIELD 自己的 exit 阈值 12M」。
    while idx > 0:
        exit_thr = THRESHOLDS.get(ORDER[idx], (0, 0))[1]
        if others_1h_tokens <= exit_thr:
            idx -= 1
        else:
            break

    # 升档：用目标档的 ENTER 阈值。
    while idx < len(ORDER) - 1:
        higher = ORDER[idx + 1]
        enter_thr = THRESHOLDS.get(higher, (0, 0))[0]
        if others_1h_tokens >= enter_thr:
            idx += 1
        else:
            break

    # 兜底：降档级联已覆盖此情形 —— 降档循环在「用量 <= 当前档 exit 阈值」时
    # 会一路降到最低满足条件的那一档；升档循环则保证 tokens >= 目标档 enter 阈值。
    # 两段跑完后必然满足「idx==0 或 tokens > exit_thr(ORDER[idx])」，
    # 故此处无需再单独降档（旧的兜底分支是死代码，已移除）。

    state = ORDER[idx]
    # 说明文案必须引用**代码里真实存在的阈值**，不能写臆想的数字。
    if state == "IDLE":
        why = (f"别人最近1小时 {others_1h_tokens/1e6:.2f}M tokens"
               f"（< {THRESHOLDS['WARM'][0]/1e6:.1f}M 进入阈值）→ 配额基本没人用，全力干")
    elif state == "WARM":
        why = (f"别人最近1小时 {others_1h_tokens/1e6:.2f}M tokens"
               f"（>= {THRESHOLDS['WARM'][0]/1e6:.1f}M 进入阈值）→ 轻度占用，控单次预算")
    elif state == "BUSY":
        why = (f"别人最近1小时 {others_1h_tokens/1e6:.2f}M tokens"
               f"（>= {THRESHOLDS['BUSY'][0]/1e6:.0f}M 进入阈值）→ 中度占用，优先零token活")
    else:
        why = (f"别人最近1小时 {others_1h_tokens/1e6:.2f}M tokens"
               f"（>= {THRESHOLDS['YIELD'][0]/1e6:.0f}M 进入阈值）→ 重度占用，让路")
    return state, why


def evaluate() -> dict:
    """跑一次完整判定，返回可落盘的状态。"""
    quota = fetch_quota()
    prev = None
    if STATE_FILE.exists():
        try:
            prev = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("state")
        except Exception:  # noqa: BLE001
            prev = None

    others = usage_rates(60.0)
    state, why = classify(others["others_tokens_per_h"], prev)

    # 配额兜底：无论别人用没用，我自己配额见底就得收手（宋老师 98% 铁律）
    hard_stop = False
    used = quota.get("interval_used_pct")
    if isinstance(used, (int, float)) and used >= 98:
        hard_stop = True
        why += f"；且我方 5h 配额已用 {used}%（>=98% 硬停线）→ 全面收手"

    return {
        "ts": datetime.now(CST).isoformat(timespec="seconds"),
        "state": state,
        "prev_state": prev,
        "reason": why,
        "hard_stop": hard_stop,
        "quota": quota,
        "others": others,
        "actions": {
            "IDLE": "全力干自己的活：深度分析、子 agent、研究实验都放开",
            "WARM": "正常干活：控制单次任务 token 预算，避免长会话累积",
            "BUSY": "优先零 token 的活（数据管线/研究队列/回测），深度分析降频",
            "YIELD": "让路：只跑零 token 的活，等 5h 窗口过去再恢复",
        }[state],
    }


def save_state(st: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)          # 原子写
    with DECISIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": st["ts"], "state": st["state"], "prev_state": st["prev_state"],
            "others_tokens_per_h": st["others"]["others_tokens_per_h"],
            "others_quota_pct": st["others"]["others_tokens_per_h"] / TOKENS_PER_QUOTA_PCT,
            "self_tokens_per_h": st["others"]["self_tokens_per_h"],
            "interval_used_pct": st["quota"].get("interval_used_pct"),
        }, ensure_ascii=False) + "\n")


def cmd_status() -> int:
    st = evaluate()
    save_state(st)
    q = st["quota"]
    o = st["others"]
    print(f"节奏状态: {st['state']}   (上一档: {st['prev_state'] or '无'})")
    print(f"依据: {st['reason']}")
    print(f"该做什么: {st['actions']}")
    print()
    if q.get("status") == "ok":
        print(f"配额(general) 5h 剩余 {q['interval_remaining_pct']}%  "
              f"已用 {q['interval_used_pct']}%  距reset {q['interval_reset_min']}分钟")
        print(f"配额(general) 本周剩余 {q['weekly_remaining_pct']}%  已用 {q['weekly_used_pct']}%")
    else:
        print(f"配额: 取不到 —— {q.get('error')}")
    print()
    if o.get("baseline_missing"):
        print(f"别人速率: 基线缺失（首次运行），速率不可靠 —— {o.get('note')}")
    else:
        pct = o["others_tokens_per_h"] / TOKENS_PER_QUOTA_PCT
        print(f"别人速率(近{o.get('window_minutes',60):.0f}分钟折算每小时): "
              f"{o['others_calls_per_h']} 次调用 / {o['others_tokens_per_h']/1e6:.2f}M tokens "
              f"(≈ {pct:.2f}% 配额)")
        for k, v in sorted(o["per_profile"].items(), key=lambda x: -x[1]["tokens"]):
            print(f"    {k:12s} {v['calls']:5d} 次/h  {v['tokens']/1e6:8.2f}M/h")
        print(f"我速率:     {o['self_calls_per_h']} 次调用/h / {o['self_tokens_per_h']/1e6:.2f}M tokens/h")
    print()
    print(f"烧token闸门: {'❌ 让路（只跑零token活）' if (st['state']=='YIELD' or st['hard_stop']) else '✅ 可烧'}")
    return 0


def cmd_gate() -> int:
    """供其他脚本调用。EXIT 0 = 可以烧 token；EXIT 2 = 应该让路。"""
    st = evaluate()
    save_state(st)
    if st["state"] == "YIELD" or st["hard_stop"]:
        print(st["reason"])
        return 2
    print(st["state"], st["reason"])
    return 0


def cmd_record(arg: str) -> int:
    """记录一次决策，用于审计。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with DECISIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(CST).isoformat(timespec="seconds"),
                            "action": arg}, ensure_ascii=False) + "\n")
    print("recorded:", arg)
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        return cmd_status()
    if cmd == "gate":
        return cmd_gate()
    if cmd == "record":
        return cmd_record(sys.argv[2] if len(sys.argv) > 2 else "?")
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
