"""开盘价面板：与主面板（close）对齐的 open 矩阵。

为什么需要它：
    历史回测一直假设「信号次日**收盘**成交」。这是整个研究里最乐观的
    一个假设 —— 收盘价是当天最后一刻才形成的，你不可能在知道收盘价的同时
    用收盘价成交。

    要验证方法能不能真的落地，必须把成交价换成更现实的**次日开盘价**，
    看超额还剩多少。这个模块提供对齐后的 open 矩阵，让成交真实性检验
    和前瞻账本都能用真实开盘价算。

口径铁律：
1. 与主面板**逐格对齐**（同样的 dates、同样的 codes、同样的顺序）。
   对不齐就是拿错了数据，宁可炸掉也不能错位。
2. 只保留正开盘价；缺失一律 NaN，不插值、不前向填充 ——
   缺失本身是信息（停牌/无数据），填平了就把摩擦藏起来了。
3. 缓存带 K 线指纹，指纹变了缓存失效，必须重建（和主面板同一纪律）。

零 LLM，纯确定性。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import history
import ooslab

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "panel" / "open_panel.npz"


def load_open_panel(
    *,
    panel: dict[str, Any] | None = None,
    cache: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """加载（或重建）与主面板对齐的 open 矩阵。"""
    if panel is None:
        panel = ooslab.load_panel()

    dates = [str(d) for d in panel["dates"]]
    codes = [str(c) for c in panel["codes"]]
    if not dates or not codes:
        raise RuntimeError("主面板为空，无法构建 open 面板")

    cache = cache or CACHE
    fp = ooslab._kline_fingerprint()

    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            cached_fp = str(z["fingerprint"]) if "fingerprint" in z else ""
            same_dates = [str(d) for d in z["dates"]] == dates
            same_codes = [str(c) for c in z["codes"]] == codes
            # 缓存无指纹字段也必须判失效 —— 写 `if cached_fp and ...` 会短路放行
            if cached_fp == fp and same_dates and same_codes and "open" in z:
                return {
                    "dates": dates,
                    "codes": codes,
                    "open": z["open"],
                    "fingerprint": fp,
                    "cached": True,
                }
        except Exception:  # noqa: BLE001
            pass  # 缓存坏了就重建

    date_idx = {d: i for i, d in enumerate(dates)}
    open_m = np.full((len(dates), len(codes)), np.nan, dtype=np.float32)

    n_rows = 0
    n_missing_code = 0
    for ci, code in enumerate(codes):
        rows = history.load_kline(code)
        if not rows:
            n_missing_code += 1
            continue
        for r in rows:
            di = date_idx.get(r["date"])
            if di is None:
                continue
            o = r.get("open")
            if o is None:
                continue
            try:
                ov = float(o)
            except (TypeError, ValueError):
                continue
            if not (ov > 0):      # 非正开盘价一律丢弃，不插值
                continue
            open_m[di, ci] = ov
            n_rows += 1

    if verbose:
        coverage = float(np.isfinite(open_m).mean()) if open_m.size else 0.0
        print(
            f"[open_panel] {len(dates)} 日 × {len(codes)} 只  "
            f"开盘价覆盖 {coverage:.2%}  无历史 {n_missing_code} 只  "
            f"写入 {n_rows} 格"
        )

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            dates=np.array(dates, dtype=object).astype("U16"),
            codes=np.array(codes, dtype=object).astype("U16"),
            open=open_m,
            fingerprint=np.array(fp),
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "dates": dates,
        "codes": codes,
        "open": open_m,
        "fingerprint": fp,
        "cached": False,
    }


if __name__ == "__main__":
    import time

    t0 = time.time()
    op = load_open_panel()
    print(f"fingerprint={op['fingerprint'][:12]}  cached={op['cached']}  "
          f"用时 {time.time() - t0:.1f}s")
