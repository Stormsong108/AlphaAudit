#!/usr/bin/env python3
"""财经日报主流程。

链路：采集 → 双源核验 → 模拟组合建议 → 历史复盘 → 落不可变快照 → 投递一份 HTML 文件。

用法:
  python3 main.py              # 采集 + 建议 + 复盘 + 落盘 + 投递 HTML 文件
  python3 main.py --dry-run    # 只跑到落盘，不投递
  python3 main.py --stdout     # 把文本日报打到 stdout（不自行投递）
  python3 main.py --force      # 当天已投递过也强制重发
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import deliver as deliver_mod  # noqa: E402
from advice import build_advice  # noqa: E402
from analysis import build_research_report  # noqa: E402
from collectors.bank_wp import collect_bank_wp  # noqa: E402
from collectors.events import collect_events  # noqa: E402
from collectors.futures_fx import collect_futures_fx  # noqa: E402
from collectors.products import collect_products  # noqa: E402
from collectors.quotes import collect_quotes  # noqa: E402
from render import render_daily, render_html  # noqa: E402
from review import build_review  # noqa: E402
from trading_calendar import resolve_market_session  # noqa: E402

RAW_DIR = HERE / "data" / "raw"
ARCHIVE_DIR = HERE / "data" / "archive"
HTML_DIR = HERE / "data" / "html"
SNAPSHOT_DIR = HERE / "data" / "snapshots"
STATE_DIR = HERE / "state"
LOG_DIR = HERE / "logs"

LARK_CLI = "/opt/homebrew/bin/lark-cli"


DEBOUNCE_MIN = 120  # 同一天内两次投递的最小间隔（分钟）


def _last_delivered_ts(day: str) -> float:
    """返回当天最近一次成功投递的时间戳；没有则 0。

    注意：不能只看「当天有没有投过」。白天手动跑过一次，晚上 21:30 的定时
    任务就会被静默跳过 —— 日报就少了一期。用时间窗口做防抖才对。
    """
    marker = STATE_DIR / f"delivered_{day}.marker"
    if not marker.exists():
        return 0.0
    raw = marker.read_text(encoding="utf-8").strip()
    # 兼容旧格式（纯日期）和新格式（完整时间戳）
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(raw, fmt))
        except ValueError:
            continue
    return 0.0


def _already_delivered(day: str) -> bool:
    """最近 DEBOUNCE_MIN 分钟内是否已投递过。"""
    ts = _last_delivered_ts(day)
    if not ts:
        return False
    return (time.time() - ts) < DEBOUNCE_MIN * 60


def _mark_delivered(day: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"delivered_{day}.marker").write_text(
        time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8"
    )


# ------------------------------------------------------------------ 采集
def collect_all() -> dict:
    """跑四个采集器。单源失败不阻断整体，降级为空结构。"""
    t0 = time.time()

    def _safe(fn, empty):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[collect] {fn.__name__} failed: {exc}", file=sys.stderr)
            return empty

    quotes = _safe(collect_quotes, {"indexes": [], "stocks": [], "conflicts": [], "sources": {}})
    events = _safe(
        collect_events,
        {"events": [], "macro": [], "conflicts": [], "sources": {}, "stats": {}},
    )
    futures = _safe(
        collect_futures_fx,
        {"cn_futures": [], "global_futures": [], "fx": [], "fx_mid": None, "conflicts": [], "sources": {}},
    )
    products = _safe(
        collect_products,
        {"products": [], "conflicts": [], "sources": {}},
    )
    # 银行理财单独一个采集器（中国理财网登记口径），在流水线层合并，
    # 不动 collect_products 自身的返回契约。
    bank = _safe(collect_bank_wp, {"products": [], "sources": {}})
    if bank.get("products"):
        products = {
            "products": list(products.get("products") or []) + list(bank["products"]),
            "conflicts": products.get("conflicts") or [],
            "sources": {**(products.get("sources") or {}), **(bank.get("sources") or {})},
        }

    print(
        "[collect] 指数 {} 股票 {} 事件 {} 宏观 {} 国内期货 {} 外盘 {} 外汇 {} 理财 {}".format(
            len(quotes.get("indexes") or []),
            len(quotes.get("stocks") or []),
            len(events.get("events") or []),
            len(events.get("macro") or []),
            len(futures.get("cn_futures") or []),
            len(futures.get("global_futures") or []),
            len(futures.get("fx") or []),
            len(products.get("products") or []),
        ),
        file=sys.stderr,
    )
    print(f"[collect] done in {time.time() - t0:.1f}s", file=sys.stderr)
    return {
        "quotes": quotes,
        "events": events,
        "futures": futures,
        "products": products,
    }


# ------------------------------------------------------------------ 每日闭环


def run_daily_loop() -> dict:
    """触发每日 live 闭环的判定阶段（predict → judge）。

    只跑判定相关的两段，不跑 refresh：行情刷新由研究队列负责
    （research_queue.py 的 refresh 步骤），日报只消费结果，不抢上游。

    判定失败时**不静默跳过**：返回 ok=False + 原因，由日报正文如实显示，
    不用上一次的旧判定冒充"今天已判定"。

    返回 daily_loop_status() 形状：
        {"ok": bool, "stages": {refresh/predict/judge: {...}}}
    """
    import daily_loop

    for name in ("predict", "judge"):
        payload = daily_loop.run_stage(name)
        if not payload.get("ok"):
            print(
                f"[daily_loop] {name} 失败：{payload.get('reason') or '未知原因'}",
                file=sys.stderr,
            )
    return daily_loop.daily_loop_status()


# ------------------------------------------------------------------ 落盘
def save_snapshot(bundle: dict, text: str, html: str, day: str) -> Path:
    """落运行产物（当天可重复覆盖；历史日期不再改动）。"""
    for d in (RAW_DIR, ARCHIVE_DIR, HTML_DIR, STATE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"{day}.json"
    raw_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html_path = HTML_DIR / f"{day}.html"
    html_path.write_text(html, encoding="utf-8")

    # 归档：日报正文，长度受限以免无限膨胀
    arch_path = ARCHIVE_DIR / f"{day}.md"
    arch_path.write_text(text, encoding="utf-8")

    # 状态：最近一次运行
    (STATE_DIR / "last_run.json").write_text(
        json.dumps(
            {
                "day": day,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "counts": {
                    "indexes": len(bundle["quotes"].get("indexes") or []),
                    "stocks": len(bundle["quotes"].get("stocks") or []),
                    "events": len(bundle["events"].get("events") or []),
                    "macro": len(bundle["events"].get("macro") or []),
                    "cn_futures": len(bundle["futures"].get("cn_futures") or []),
                    "global_futures": len(bundle["futures"].get("global_futures") or []),
                    "fx": len(bundle["futures"].get("fx") or []),
                    "products": len(bundle.get("products", {}).get("products") or []),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_path


def save_immutable_snapshot(
    bundle: dict,
    advice: dict,
    review: dict,
    day: str,
    session: dict | None = None,
    research: dict | None = None,
    evidence: dict | None = None,
) -> Path:
    """写一次就锁定的每日快照：固化当期的判断、依据、关注标的与模拟组合。

    规则：文件已存在就不再改写。历史结论只能被后续日报「追加复盘」，
    不允许事后回头修改。
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{day}.json"
    if path.exists():
        print(f"[snapshot] {path.name} 已固化，跳过改写", file=sys.stderr)
        return path

    quotes = bundle.get("quotes") or {}
    events = bundle.get("events") or {}
    futures = bundle.get("futures") or {}
    products = bundle.get("products") or {}
    # 完整研究证据（门禁 / 判定 / 闭环）+ 采集口径来源，合成一份落盘。
    # 注意：以前这里写了两个 "evidence" 键，后者把前者整个覆盖掉了，
    # 结果快照里只剩 sources/conflicts，判定与门禁证据全丢 —— 不可变快照必须完整。
    ev_payload = dict(evidence or {})
    ev_payload["sources"] = {
        "quotes": quotes.get("sources"),
        "events": events.get("sources"),
        "futures": futures.get("sources"),
        "products": products.get("sources"),
    }
    ev_payload["conflicts"] = {
        "quotes": quotes.get("conflicts") or [],
        "events": events.get("conflicts") or [],
        "futures": futures.get("conflicts") or [],
        "products": products.get("conflicts") or [],
    }
    payload = {
        "day": day,
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "immutable": True,
        "market_session": session or {},
        "research": research or {},
        "evidence": ev_payload,
        "prices": {
            s.get("code"): s.get("price")
            for s in (quotes.get("stocks") or [])
            if s.get("code") and isinstance(s.get("price"), (int, float))
        },
        "advice": advice,
        "review_at_run": {
            "available": (review or {}).get("available"),
            "note": (review or {}).get("note"),
            "stats": (review or {}).get("stats"),
        },
        "raw_ref": f"data/raw/{day}.json",
        "html_ref": f"data/html/{day}.html",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 投递
def _resolve_targets() -> list[tuple[str, str, str]]:
    """算出投递目标 [(kind, receive_id, id_type)]，并强制禁止名单。"""
    targets = config.FEISHU.get("deliver_to") or []
    group = config.FEISHU.get("group_chat_id")
    user = config.FEISHU.get("open_id")
    forbidden = set(config.FEISHU.get("forbidden_chat_ids") or [])

    out: list[tuple[str, str, str]] = []
    for kind in targets:
        if kind == "group" and group:
            if group in forbidden:
                raise RuntimeError(
                    f"投递目标 {group} 在禁止名单内，拒绝投递（金融内容不进留学群）"
                )
            out.append(("group", group, "chat_id"))
        elif kind == "dm" and user:
            out.append(("dm", user, "open_id"))
    return out


def _should_mark_delivered(results: list[dict], dry_run: bool) -> bool:
    """能不能写「已投递」标记。

    ⚠️ dry-run 绝不写。deliver_html(dry_run=True) 返回 ok=True 的假结果，
    若直接 `all(ok)` 判定，会误写标记 → 当晚真实投递被「最近 N 分钟内已投递」
    静默跳过 = 少发一期日报。
    """
    if dry_run:
        return False
    return bool(results) and all(r.get("ok") for r in results)


def deliver(text: str, *, dry_run: bool = False) -> list[dict]:
    """投递纯文本（兼容路径）。正式日报走 deliver_html。"""
    if dry_run:
        return [{"target": "dry-run", "ok": True, "message_id": None}]

    results: list[dict] = []
    for kind, rid, idt in _resolve_targets():
        results.append({"target": f"{kind}:{rid}", **_try_send(text, rid, idt)})
    return results


def deliver_html(html_path: Path | str, *, dry_run: bool = False) -> list[dict]:
    """把日报作为**一份文件**投到飞书。

    上传一次拿 file_key，多个目标共用同一个 file_key ——
    保证每个收件方收到的都是同一份 HTML 文件，而不是文本 + 附件混发。
    """
    if dry_run:
        return [{"target": "dry-run", "ok": True, "message_id": None}]

    targets = _resolve_targets()
    if not targets:
        return []
    file_key = deliver_mod.upload_file(html_path)
    results: list[dict] = []
    for kind, rid, idt in targets:
        results.append({"target": f"{kind}:{rid}", **_try_send_file(file_key, rid, idt)})
    return results


def _try_send(text: str, receive_id: str, id_type: str) -> dict:
    """投递单个目标。失败返回 ok=False，不抛异常（其他目标继续投）。"""
    try:
        return deliver_mod.send_text(text, receive_id=receive_id, id_type=id_type)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:500], "message_id": None}


def _try_send_file(file_key: str, receive_id: str, id_type: str) -> dict:
    """投递单个目标的文件消息。失败返回 ok=False，不抛异常。"""
    try:
        return deliver_mod.send_file(file_key, receive_id=receive_id, id_type=id_type)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:500], "message_id": None}



# ------------------------------------------------------------------ 看板运营
KANBAN_BOARD = "fin-daily"
KANBAN_VERIFIER = "xiaoyun"  # 运营质量总监：独立验收当天日报


def _hermes_bin() -> str | None:
    import shutil

    p = shutil.which("hermes")
    if p:
        return p
    fallback = Path.home() / ".hermes" / "hermes-agent" / ".hermes" / "bin" / "hermes"
    return str(fallback) if fallback.exists() else None


def create_verification_task(day: str, deliver_results: list[dict]) -> str | None:
    """日报产出后，在 fin-daily 看板派一条核验任务给另一个 bot。

    设计意图：生产数据链路保持确定性（不把 LLM 放进采集/渲染/投递），
    但「运营层」交给群组 —— 由另一个 profile 独立验收当天产物是否满足
    验收契约，并把结论写回任务评论，形成可追溯的运营闭环。

    幂等：key 里带日期，同一天重复跑不会重复建单。
    """
    bin_path = _hermes_bin()
    if not bin_path:
        print("[kanban] 找不到 hermes CLI，跳过核验任务", file=sys.stderr)
        return None

    html_path = HTML_DIR / f"{day}.html"
    snap_path = SNAPSHOT_DIR / f"{day}.json"
    delivered = all(r.get("ok") for r in deliver_results) if deliver_results else False
    body = f"""【任务】独立验收 {day} 的财经日报产物，**只读不改、不发飞书**。

要检查的文件：
  HTML：{html_path}
  快照：{snap_path}
  投递结果：{json.dumps(deliver_results, ensure_ascii=False)}

执行方式：运行项目自带的确定性验收脚本（不联网、不改文件）：
  cd {HERE}
  .venv/bin/python verify_report.py {day}

脚本会逐条打印 `[PASS]/[FAIL] <检查项> —— <证据>`，末尾给出
`CHECKS: N  FAILED: M` 和 `VERDICT: PASS|FAIL`。

你的职责（不要自己去通读 HTML，脚本已经覆盖清单）：
  1. 先运行上面这条命令，原样捕获全部输出。
  2. 如果 VERDICT 是 PASS：用 hermes kanban comment 在本任务下评论，
     正文第一行严格是 `VERIFY PASS`，第二行起附上脚本输出摘要
     （CHECKS 数、FAILED 数，以及组合行数/关注标的数/仓位合计等关键数字）。
  3. 如果 VERDICT 是 FAIL：第一行严格是 `VERIFY FAIL`，第二行起逐条列出
     每个 FAIL 项及其证据（脚本输出里的原文），并说明失败原因判断。
  4. 如果脚本本身无法运行（退出码非 0，或脚本报错），第一行写
     `VERIFY FAIL`，第二行写明执行失败的具体原因（命令、退出码、stderr 前 300 字）。
  5. 评论完成后用 hermes kanban complete 标记本任务完成。

硬约束：
  - 不要修改任何文件（不要写、不要删、不要改数据）
  - 不要发任何飞书消息，不要重新投递日报
  - 不要自己另写一套验收标准，只跑脚本并如实转述它的结论
"""
    body_file = STATE_DIR / f"verify_{day}.md"
    body_file.parent.mkdir(parents=True, exist_ok=True)
    body_file.write_text(body, encoding="utf-8")

    cmd = [
        bin_path, "kanban", "--board", KANBAN_BOARD, "create",
        f"日报核验 {day}",
        "--body-file", str(body_file),
        "--assignee", KANBAN_VERIFIER,
        "--workspace", f"dir:{HERE}",
        "--idempotency-key", f"daily-verify-{day}",
        "--max-retries", "2",
        "--max-runtime", "15m",
        "--json",
    ]
    try:
        import subprocess

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            print(f"[kanban] 建核验任务失败 rc={proc.returncode}: {proc.stderr[:300]}", file=sys.stderr)
            return None
        import re

        m = re.search(r'"id":\s*"([^"]+)"', proc.stdout)
        task_id = m.group(1) if m else None
        print(f"[kanban] 核验任务 {task_id} 已建（verifier={KANBAN_VERIFIER}, delivered={delivered}）", file=sys.stderr)
        return task_id
    except Exception as exc:  # noqa: BLE001
        print(f"[kanban] 建核验任务异常: {exc}", file=sys.stderr)
        return None

# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不投递")
    ap.add_argument(
        "--stdout",
        action="store_true",
        help="把日报正文打到 stdout 且不自行投递（给 cron no_agent 模式用，由调度器投递）",
    )
    ap.add_argument("--text", action="store_true", help="--stdout 的别名")
    ap.add_argument("--force", action="store_true", help="当天已投递过也强制重发")
    ap.add_argument("--title", default="财经日报")
    ap.add_argument("--no-kanban", action="store_true", help="不向看板派核验任务")
    args = ap.parse_args()

    day = time.strftime("%Y-%m-%d")
    bundle = collect_all()

    # 交易日判定：从行情接口自己报的 trade_date 反推，不硬编码节假日表
    session = resolve_market_session(bundle["quotes"])
    # 挂回 quotes：复盘据此判断当日报价是不是"新的收盘价"
    bundle["quotes"]["market_session"] = session
    if not session.get("is_trading_day"):
        print(
            f"[session] {day} A 股休市，行情对应 {session.get('data_date')}，"
            "日报将标注休市状态",
            file=sys.stderr,
        )

    # 复盘要先于今日快照落盘：今天不能拿自己复盘自己
    review = build_review(SNAPSHOT_DIR, bundle["quotes"])
    advice = build_advice(bundle["quotes"], bundle.get("products"))
    # 每日研报：汇总当日全部数据后的综合判断（确定性规则生成）
    research = build_research_report(
        bundle["quotes"],
        bundle["events"],
        bundle["futures"],
        advice,
        review,
        session,
        bundle.get("products"),
    )

    # 每日闭环：先出 live 预测、再用真实行情判定，然后才让研究证据读已落盘的判定结果。
    # 顺序不能反：先 build_evidence 就会读到上一轮的旧判定。
    # 失败不抛异常也不静默：状态写进 evidence["daily_loop"]，日报正文如实显示失败。
    daily_status = run_daily_loop()
    if not daily_status.get("ok"):
        failed = [
            name for name, st in (daily_status.get("stages") or {}).items()
            if st.get("status") == "failed"
        ]
        print(
            f"[daily_loop] 闭环未跑通：{('、'.join(failed) or '部分阶段')} 失败；"
            "日报会显示失败状态，不会拿旧结果冒充今天已完成",
            file=sys.stderr,
        )

    # 研究能力实测：样本外 / 成本后 / 概率校准 / 预测闭环
    # 读不到或出错都不能影响日报主流程 —— 研究数据缺失就如实标"无数据"，不编造。
    evidence: dict = {}
    try:
        import research_evidence

        evidence = research_evidence.build_evidence()
        # 用刚才这次运行的状态覆盖，保证日报显示的是"今天这一次"的结果，
        # 而不是 build_evidence 读到的上一轮状态。
        evidence["daily_loop"] = daily_status
        if evidence.get("available"):
            print(
                f"[evidence] 样本外实验已载入："
                f"{evidence.get('oos_pass', 0)}/{evidence.get('oos_total', 0)} 因子时间外通过，"
                f"门禁 {evidence.get('verdict', {}).get('passed', 0)}/"
                f"{evidence.get('verdict', {}).get('total', 0)}",
                file=sys.stderr,
            )
        else:
            print("[evidence] 尚无样本外实验结果，研究板块将显示空态", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[evidence] 载入失败（不影响日报）: {exc}", file=sys.stderr)
        evidence = {"daily_loop": daily_status}

    text = render_daily(
        bundle["quotes"],
        bundle["events"],
        bundle["futures"],
        products=bundle.get("products"),
        advice=advice,
        review=review,
        session=session,
        title=args.title,
    )
    # 每日研究汇报（大白话 · 七段式）—— 2026-11-14 起嵌进同一份 HTML，
    # 不再单独发一条飞书消息（「每天 21:30 只发一份 HTML」铁律）。
    # 取不到就如实显示原因，绝不静默留空（空白会被误读成系统挂了）。
    brief_text: str | None = None
    brief_error: str | None = None
    try:
        import daily_brief

        brief_text = daily_brief.build_brief()
    except Exception as exc:  # noqa: BLE001
        brief_error = f"每日研究汇报：这轮取不到（{type(exc).__name__}: {exc}）"
        print(f"[brief] 载入失败（不影响日报）: {exc}", file=sys.stderr)

    html = render_html(
        bundle["quotes"],
        bundle["events"],
        bundle["futures"],
        products=bundle.get("products"),
        advice=advice,
        review=review,
        session=session,
        research=research,
        evidence=evidence,
        brief_text=brief_text,
        brief_error=brief_error,
        title=args.title,
    )

    raw = save_snapshot(bundle, text, html, day)
    snap = save_immutable_snapshot(bundle, advice, review, day, session, research, evidence)
    print(f"[save] {raw} | [snapshot] {snap}", file=sys.stderr)

    if args.stdout or args.text:
        print(text)
        return 0

    if _already_delivered(day) and not args.force:
        print(
            f"[deliver] {day} 最近 {DEBOUNCE_MIN} 分钟内已投递，跳过"
            "（加 --force 可强制重发）",
            file=sys.stderr,
        )
        return 0

    html_path = HTML_DIR / f"{day}.html"
    results = deliver_html(html_path, dry_run=args.dry_run)
    for r in results:
        print(
            f"[deliver] {r['target']} ok={r['ok']} mid={r.get('message_id')}",
            file=sys.stderr,
        )
        if not r.get("ok"):
            print(f"[deliver][error] {r.get('error')}", file=sys.stderr)

    # ⚠️ dry-run 必须**不写**投递标记（见 _should_mark_delivered 的说明）。
    if _should_mark_delivered(results, args.dry_run):
        _mark_delivered(day)
    elif args.dry_run:
        print("[deliver] dry-run：未投递，未写投递标记", file=sys.stderr)

    # 结果写入日志，方便复盘时追溯投递是否真的成功
    log_path = LOG_DIR / f"{day}.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "day": day,
                    "mode": "html_file",
                    "html": str(html_path),
                    "deliver": results,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    ok_all = all(r.get("ok") for r in results) if results else True

    # 派单给另一个 bot 独立验收当天产物（看板运营层）
    if not args.no_kanban:
        create_verification_task(day, results)

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
