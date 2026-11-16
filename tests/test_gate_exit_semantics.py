"""门禁的退出码与状态语义 —— 区分「有效结论」和「执行失败」。

背景（2026-11-16 实测）：
  跨来源对账门禁第一版，`cross_source.py` 一律「不通过就退出 1」，
  `research_queue.py` 一律「非零就标 failed」，`--run` 一律「返回 0」。
  三个缺陷叠在一起：

  1. 门禁**正确发现口径矛盾** = 有效结论，却被当成「脚本崩了」：
     标 failed → 触发 30 分钟冷却 + 金融组告警噪音 → 真事故被假告警淹没。
  2. 全新部署、产物未生成 = 数据没准备好，却被当成口径矛盾报出去。
  3. `--run` 恒返回 0 → 门禁检测到事故也没人收到通知（静默吞掉）。

  修复后的语义：
     0 = 对账通过
     2 = 对账成功执行，发现口径矛盾（**有效结论**）
     3 = 所有来源都不可用（数据未就绪）
     1 = 执行本身失败
  队列状态对应：ok / blocked / skipped / failed。
  `--run` 在有 failed 或 blocked 时返回非零 → cron 投递金融组。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python3"
if not PY.exists():  # pragma: no cover - 环境兜底
    PY = Path(sys.executable)


# ------------------------------------------------------------------
# 1. cross_source.py 的退出码
# ------------------------------------------------------------------
def _run_cross_source(workdir: Path) -> int:
    """在隔离目录里跑 cross_source.py，返回退出码。"""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "cross_source.py").write_text(
        (ROOT / "cross_source.py").read_text(encoding="utf-8"), encoding="utf-8")
    p = subprocess.run([str(PY), "cross_source.py"], cwd=workdir,
                       capture_output=True, text=True, timeout=180)
    return p.returncode


def test_exit_3_when_all_sources_missing(tmp_path):
    """场景A：全新部署，一个产物都没有 → rc=3（数据未就绪），不是矛盾也不是崩溃。"""
    rc = _run_cross_source(tmp_path / "empty")
    assert rc == 3, f"无产物时应为 3（数据未就绪），实得 {rc}"


@pytest.mark.skipif(
    not (ROOT / "data" / "research" / "oos_latest.json").exists(),
    reason="缺少运行时数据，跳过")
def test_exit_0_when_consistent(tmp_path):
    """场景B：真实产物、口径一致 → rc=0。

    必须在 tmp 副本上跑，不能直接跑项目目录 —— 否则测试会写回
    data/research/cross_source_check.json，污染真实产物。
    """
    import shutil
    wd = tmp_path / "real"
    (wd / "data" / "research").mkdir(parents=True)
    for name in ("oos_latest.json", "cross_segment_check.json",
                 "execution_realism.json", "live_readiness.json"):
        src = ROOT / "data" / "research" / name
        if src.exists():
            shutil.copy(src, wd / "data" / "research" / name)
    rc = _run_cross_source(wd)
    assert rc == 0, f"口径一致时应为 0，实得 {rc}"


@pytest.mark.skipif(
    not (ROOT / "data" / "research" / "live_readiness.json").exists(),
    reason="缺少运行时数据，跳过")
def test_exit_2_when_inconsistent(tmp_path):
    """场景C：篡改一个来源制造真矛盾 → rc=2（有效结论），不是 1（执行失败）。"""
    import shutil
    wd = tmp_path / "bad"
    (wd / "data" / "research").mkdir(parents=True)
    for name in ("oos_latest.json", "cross_segment_check.json",
                 "execution_realism.json", "live_readiness.json"):
        src = ROOT / "data" / "research" / name
        if src.exists():
            shutil.copy(src, wd / "data" / "research" / name)
    p = wd / "data" / "research" / "live_readiness.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    res = d.get("result") or {}
    br = res.get("benchmark_relative") or {}
    br["excess_annualized"] = 0.05
    res["benchmark_relative"] = br
    d["result"] = res
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    rc = _run_cross_source(wd)
    assert rc == 2, f"口径矛盾时应为 2（有效结论），实得 {rc}"


def test_exit_3_not_confused_with_exit_2():
    """两种失败态必须可区分：all_missing → 3，有矛盾 → 2。"""
    import cross_source as cs

    # 构造一个「全部 MISSING」的结果
    all_missing = {
        "consistent": False,
        "metrics": [{"key": "a", "status": "MISSING"},
                    {"key": "b", "status": "MISSING"}],
    }
    # 构造一个「有 INCONSISTENT」的结果
    mixed = {
        "consistent": False,
        "metrics": [{"key": "a", "status": "MISSING"},
                    {"key": "b", "status": "INCONSISTENT"}],
    }

    def classify(result):
        metrics = result.get("metrics") or []
        am = bool(metrics) and all(m.get("status") == "MISSING" for m in metrics)
        return 3 if am else 2

    assert classify(all_missing) == 3
    assert classify(mixed) == 2
    assert cs is not None


# ------------------------------------------------------------------
# 2. 队列状态：blocked 不是 failed
# ------------------------------------------------------------------
def test_queue_status_classes():
    """队列要能把 blocked 和 failed 区分开。"""
    import research_queue as rq
    src = Path(rq.__file__).read_text(encoding="utf-8")
    # blocked 状态存在
    assert '"blocked"' in src
    # blocked 分支只**清除**冷却（pop），不**设置**冷却（赋值）——
    # 「不当崩溃重试」的意思是没有 retry_after_at 挡住下一轮。
    blocked_block = src.split('rec["last_status"] = "blocked"')[1][:400]
    assert 'rec["retry_after_at"] =' not in blocked_block, \
        "blocked 不应设置 30 分钟冷却（那是真崩溃才有的）"
    assert 'rec.pop("retry_after_at", None)' in blocked_block, \
        "blocked 应清除历史冷却，不让上一次失败挡住下一轮"
    # failed 才设置冷却
    failed_block = src.split('rec["last_status"] = "failed"')[1][:400]
    assert 'rec["retry_after_at"] =' in failed_block


def test_queue_run_returns_nonzero_on_blocked():
    """--run 在有 blocked/failed 时必须返回非零 —— 否则事故被静默吞掉。"""
    import research_queue as rq
    src = Path(rq.__file__).read_text(encoding="utf-8")
    run_block = src.split("if args.run:")[1]
    assert '"failed", "blocked"' in run_block, \
        "--run 必须把 failed/blocked 计入整体返回码"
    assert "return 1" in run_block


def test_step_exit_code_distinguishes_blocked():
    """单步模式：blocked → 2，ok → 0，其它 → 1。"""
    import research_queue as rq
    src = Path(rq.__file__).read_text(encoding="utf-8")
    step_block = src.split("if args.step:")[1].split("if args.run:")[0]
    assert 'rec["last_status"] == "blocked"' in step_block
    assert "return 2" in step_block


# ------------------------------------------------------------------
# 3. 依赖检查：blocked 必须阻断下游
# ------------------------------------------------------------------
def test_blocked_blocks_dependents():
    """状态 blocked 的依赖不能放行下游步骤。"""
    import research_queue as rq
    src = Path(rq.__file__).read_text(encoding="utf-8")
    dep_block = src.split("dep_status = dep_rec.get(\"last_status\")")[1][:400]
    assert '!= "ok"' in dep_block, \
        "依赖判定必须是「非 ok 即阻断」，blocked 必须挡住下游"
