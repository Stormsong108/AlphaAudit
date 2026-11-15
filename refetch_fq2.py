"""历史行情重抓 + 双口径交叉校验。

数据源两个口径同时拉：
    * fqt=0 不复权 —— 真实成交价，**涨跌幅制度约束的是这一条**
    * fqt=2 后复权 —— 总回报序列（含现金分红与送股），**研究用这一条**

为什么必须两个都要：
    2026-11-10 的教训链：
      1. 旧数据用 fqt=1 前复权 → 高分红股历史价穿零变负 → 负价格当分母算收益
         → 符号翻转、幅度离谱（-0.15 → +0.5 算成 -433%）→ 少数坏点带偏全部横截面统计
      2. 换成 fqt=2 后复权后，又出现"单日涨幅 +15.65% 超过 ±10% 制度上限"的告警
      3. 查证后发现是**误报**：000750 在除权日 10派1.5元，后复权涨幅含分红，
         裸价 +9.62% 完全合法。涨跌幅制度管裸价，管不住总回报。

    所以正确的校验是两条独立检查：
      A. 裸价的单日变动必须在制度上限内（抓真实数据错误）
      B. 后复权出现超上限变动时，复权因子必须**同步跳变**（证明是除权除息，
         而不是复权算法出了 bug）；因子没跳而涨幅离谱 = 真异常，剔除

校验通过才落盘；不通过的标的写进问题清单，不进面板。

用法：
    .venv/bin/python refetch_fq2.py --workers 8 --swap
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import dataquality  # noqa: E402
import history  # noqa: E402

OLD_DIR = ROOT / "data" / "history" / "kline"
STAGE_DIR = ROOT / "data" / "history" / "kline_fq2"
PROGRESS = ROOT / "data" / "history" / "refetch_fq2.progress"
BACKUP_DIR = ROOT / "data" / "history" / "kline_fq1_old"
FLAGS_DIR = ROOT / "data" / "history" / "flags_fq2"


def _fetch_raw(secid: str, beg: str = "20150101") -> list[list[str]]:
    """拉不复权 K 线，返回原始 kline 字符串列表。"""
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={secid}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57"
        f"&klt=101&fqt=0&beg={beg}&end=20500101"
    )
    import ssl
    import urllib.request
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
    with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
        payload = json.loads(resp.read().decode("utf-8", "replace"))
    return (payload.get("data") or {}).get("klines") or []


def cross_validate(secid: str, adj_rows: list[dict], raw_klines: list[str]) -> dict:
    """双口径交叉校验。

    核心判据是**复权因子 F = 后复权价 / 裸价 在这天跳没跳**：

        F 跳了  → 公司行为（分红/送股/配股）
                   裸价会跳（送股直接腰斩），后复权会跳（现金分红计入总回报）
                   两者都**合法**，涨跌幅制度管的是除权参考价，不是这两个序列。
        F 没跳  → 价格序列本身出问题了（复牌缺口、坏点、源方数据错）
                   → 标记这一天为 bad_day，评估时排除

    实测（000426）：2015-10-08 裸价 -45.08%（10 送 8 除权），
    裸价越界是**正常的**，同一天后复权平滑 —— 所以裸价越界不能直接否掉整只票。

    返回 {"ok", "why", "ex_rights": [...], "bad_days": [...], "n_rows"}
    """
    lim = dataquality.limit_of(secid)
    adj_tol = lim * 1.10 + 0.035      # 后复权容忍带：制度上限 + 分红余量
    raw_tol = lim * 1.03 + 0.002      # 裸价容忍带：制度上限 + 四舍五入余量

    raw = {}
    for line in raw_klines:
        p = line.split(",")
        if len(p) >= 7:
            raw[p[0]] = float(p[2])

    factor = {}
    for r in adj_rows:
        rc = raw.get(r["date"])
        if rc is not None and rc > 0:
            factor[r["date"]] = r["close"] / rc

    ex_rights: list[str] = []
    bad_days: list[dict] = []

    def _jump_thresh(prev_raw: float | None, prev_adj: float | None) -> float:
        """判定复权因子"跳变"的阈值。

        F = 后复权价 / 裸价，两个分子分母都只有 2 位小数，
        所以 F 天然带舍入噪声：噪声量级 ≈ 0.01/raw + 0.01/adj。
        固定阈值 1e-4 远低于噪声底，会把几乎每一天都误判成除权日
        （实测 200 只就误报 189,273 个"除权日"）。

        真实公司行为的幅度远大于噪声：送股 ≥ 10%，
        现金分红 0.1 元/股在 3 元股上也有 3.4%。
        所以阈值取噪声底的 3 倍，并设 0.5% 地板。
        """
        noise = 0.0
        if prev_raw and prev_raw > 0:
            noise += 0.01 / prev_raw
        if prev_adj and prev_adj > 0:
            noise += 0.01 / prev_adj
        return max(0.005, 3.0 * noise)

    prev_adj = prev_raw = prev_date = prev_factor = None
    for r in adj_rows:
        d = r["date"]
        c = r["close"]
        rc = raw.get(d)
        f = factor.get(d)

        thr = _jump_thresh(prev_raw, prev_adj)
        jumped = (prev_factor is not None and f is not None
                  and abs(f / prev_factor - 1.0) > thr)
        if jumped:
            ex_rights.append(d)

        if not jumped:
            # 没跳因子却出现越界 —— 这才是真异常
            if prev_raw is not None and rc is not None and prev_raw > 0:
                mv = rc / prev_raw - 1.0
                if abs(mv) > raw_tol:
                    bad_days.append({"date": d, "kind": "raw_move", "move": mv,
                                     "prev_date": prev_date})
            if prev_adj is not None and prev_adj > 0:
                mv = c / prev_adj - 1.0
                if abs(mv) > adj_tol:
                    bad_days.append({"date": d, "kind": "adj_move", "move": mv,
                                     "prev_date": prev_date})

        prev_adj, prev_raw, prev_date = c, rc, d
        if f is not None:
            prev_factor = f

    ok = True
    why = "OK"
    if len(adj_rows) < 60:
        ok = False
        why = f"行数过少 {len(adj_rows)}"
    elif len(bad_days) > max(5, len(adj_rows) * 0.05):
        # 坏日子太多说明整条复权序列不可信，整只弃掉
        ok = False
        why = f"异常日过多 {len(bad_days)}/{len(adj_rows)}"
    return {"ok": ok, "why": why, "ex_rights": ex_rights, "bad_days": bad_days,
            "n_rows": len(adj_rows), "n_ex_rights": len(ex_rights),
            "n_bad_days": len(bad_days)}


def fetch_one(secid: str) -> tuple[str, str, int, int]:
    """抓一只（双口径）并落盘。返回 (secid, 状态, 行数, 除权日数)。"""
    path = STAGE_DIR / f"{history.secid_safe(secid)}.json"
    try:
        rows = history._fetch_kline(secid, beg="20150101")
        raw_klines = _fetch_raw(secid, beg="20150101")
    except Exception as exc:
        return secid, f"ERR {type(exc).__name__}: {exc}", 0, 0
    if not rows:
        return secid, "EMPTY", 0, 0
    bad = [r for r in rows if not (r["open"] > 0 and r["close"] > 0
                                   and r["high"] > 0 and r["low"] > 0)]
    if bad:
        return secid, f"DIRTY {len(bad)}/{len(rows)} 非正价格", 0, 0
    v = cross_validate(secid, rows, raw_klines)
    if not v["ok"]:
        return secid, f"REJECT {v['why']}", 0, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)
    (FLAGS_DIR / f"{history.secid_safe(secid)}.json").write_text(
        json.dumps({"ex_rights": v["ex_rights"], "bad_days": v["bad_days"]},
                   ensure_ascii=False), encoding="utf-8")
    return secid, "OK", len(rows), v["n_ex_rights"]


def target_secids() -> list[str]:
    out: set[str] = set()
    for p in OLD_DIR.glob("*.json"):
        out.add(p.stem.replace("_", "."))
    uf = Path("/tmp/universe.json")
    if uf.exists():
        try:
            out.update(json.loads(uf.read_text(encoding="utf-8")).keys())
        except Exception:
            pass
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--swap", action="store_true")
    args = ap.parse_args()

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if PROGRESS.exists():
        done = {ln.strip() for ln in PROGRESS.read_text(encoding="utf-8").splitlines() if ln.strip()}

    secids = target_secids()
    if args.limit:
        secids = secids[: args.limit]
    todo = [s for s in secids if s not in done]
    print(f"目标 {len(secids)} 只，已完成 {len(secids) - len(todo)}，待抓 {len(todo)}", flush=True)

    ok = 0
    total_ex = 0
    total_bad = 0
    failed: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, s): s for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            secid, status, n, nx = fut.result()
            if status == "OK":
                ok += 1
                total_ex += nx
                with PROGRESS.open("a", encoding="utf-8") as fh:
                    fh.write(secid + "\n")
            else:
                failed.append(f"{secid}:{status}")
            if i % 200 == 0:
                print(f"[{i}/{len(todo)}] ok={ok} fail={len(failed)} "
                      f"除权日累计 {total_ex} elapsed={time.time() - t0:.0f}s", flush=True)

    print(f"抓取完成 ok={ok} fail={len(failed)} 除权日累计 {total_ex} "
          f"elapsed={time.time() - t0:.0f}s", flush=True)
    # 坏日统计
    nb = 0
    kinds: dict[str, int] = {}
    if FLAGS_DIR.exists():
        for p in FLAGS_DIR.glob("*.json"):
            try:
                fl = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            for b in fl.get("bad_days", []):
                nb += 1
                kinds[b.get("kind", "?")] = kinds.get(b.get("kind", "?"), 0) + 1
    print(f"异常日合计 {nb}（{kinds}），已写入 flags_fq2/", flush=True)
    if failed:
        (ROOT / "data" / "history" / "refetch_fq2_failures.txt").write_text(
            "\n".join(failed), encoding="utf-8")
        from collections import Counter
        kinds = Counter(f.split(":")[1].split()[0] for f in failed)
        print("失败原因分布:", dict(kinds), flush=True)
        for f in failed[:10]:
            print("  ", f, flush=True)

    if args.swap and ok >= len(secids) * 0.95:
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        OLD_DIR.rename(BACKUP_DIR)
        STAGE_DIR.rename(OLD_DIR)
        print(f"已切换：{OLD_DIR} ← {STAGE_DIR.name}"
              f"（旧数据留存于 {BACKUP_DIR.name}）", flush=True)
    else:
        print("未切换目录（未达 95% 成功率或未指定 --swap）", flush=True)


if __name__ == "__main__":
    main()
