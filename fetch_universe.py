"""批量拉取全市场历史日线（用于大样本横截面因子检验）。

后台运行：.venv/bin/python fetch_universe.py
结果落在 data/history/kline/，与 history.py 共用缓存。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import history  # noqa: E402

UNIVERSE_FILE = Path("/tmp/universe.json")


def main() -> None:
    if not UNIVERSE_FILE.exists():
        print("universe missing")
        return
    universe: dict[str, str] = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    done = 0
    failed: list[str] = []
    t0 = time.time()
    for i, (secid, name) in enumerate(universe.items(), 1):
        if history.load_kline(secid):
            done += 1
            continue
        try:
            rows = history.get_history(secid)
            if len(rows) > 100:
                done += 1
            else:
                failed.append(f"{secid}:{name}:short")
        except Exception as exc:
            failed.append(f"{secid}:{name}:{type(exc).__name__}")
        if i % 100 == 0:
            print(f"[{i}/{len(universe)}] ok={done} fail={len(failed)} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"DONE ok={done} fail={len(failed)} elapsed={time.time() - t0:.0f}s")
    if failed:
        (ROOT / "data" / "history" / "fetch_failures.txt").write_text(
            "\n".join(failed), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
