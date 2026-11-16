# -*- coding: utf-8 -*-
"""「缺 scipy 时公开仓库不得在收集阶段炸掉」的回归锁。

背景（2026-11-16 实测踩坑）：
``overfit_audit.py`` 顶层曾硬写 ``from scipy.stats import norm``，而仓库
没有 ``requirements.txt``。后果是**任何没装 scipy 的克隆者**跑 pytest 都会
在**收集阶段** ImportError → ``exit 2`` → 599 条测试一条都跑不出来，
真正的问题反而被淹没。

而这个仓库的 22 个 CPCV 测试**根本不碰 scipy** —— 它们只是被模块顶层
import 连坐。

本文件锁住三件事：

1. **无 scipy 时 ``overfit_audit`` 仍能 import**，且不依赖 norm 的功能
   （CPCV 组合计数等）照常可用；
2. **无 scipy 时 pytest 收集不中断**（rc != 2），CPCV 用例照常全跑，
   只有真正依赖 norm 的用例被 skip；
3. **依赖 norm 的函数显式抛 ``ScipyRequiredError``**，绝不静默降级 ——
   因为这些数字与公开参考实现是逐位对齐的（|d| = 0），换个近似算法
   末位会漂移，审计报告的数字就与已发布版本不可比了。

实现方式：用子进程 + ``sitecustomize.py`` 在 ``sys.meta_path`` 上装一个
导入拦截器把 scipy 屏蔽掉。这样**完全离线**，不碰网络、不装包。

跑法： cd market_daily && .venv/bin/python -m pytest tests/test_scipy_optional.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")

# 屏蔽 scipy 的 sitecustomize：装一个 meta_path 拦截器
_BLOCK_SCIPY = textwrap.dedent(
    """
    import sys


    class _BlockScipy:
        def find_spec(self, name, path=None, target=None):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy blocked for test")
            return None


    sys.meta_path.insert(0, _BlockScipy())
    """
).strip()


def _run_without_scipy(code: str, tmp_path) -> subprocess.CompletedProcess:
    """在「假装没有 scipy」的子进程里执行 code，返回 CompletedProcess。

    净化 PYTHONPATH 是必须的：本机 Hermes 的 PYTHONPATH 指向 python3.14
    的 site-packages，里面的 numpy 是 cp314 编译产物，会抢在 3.11 venv
    之前被加载并直接炸掉 import —— 那是环境问题，与本用例无关。
    """
    site_dir = tmp_path / "blocked"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(_BLOCK_SCIPY, encoding="utf-8")

    script = tmp_path / "script.py"
    script.write_text(code, encoding="utf-8")

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(site_dir)

    return subprocess.run(
        [VENV_PY, str(script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=300,
    )


# ── 1. 缺 scipy 时模块仍能 import，非 norm 功能照常 ────────────────────
def test_module_imports_and_works_without_scipy(tmp_path):
    """无 scipy：import 成功，CPCV 计数照常，norm 函数显式报错。"""
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {ROOT!r})

        try:
            import scipy
            print("PROBE_FAILED_SCIPY_IMPORTED")
        except ImportError:
            pass

        import overfit_audit as M
        print("SCIPY_AVAILABLE =", M.SCIPY_AVAILABLE)

        # 不依赖 norm 的功能必须照常工作
        print("split_count =", M.cpcv_split_count(10, 5))
        print("path_count  =", M.cpcv_path_count(10, 5))

        # 依赖 norm 的必须显式抛错，不能静默降级
        try:
            M.probabilistic_sharpe_ratio(0.10, 0.0, 250, 0.0, 3.0)
            print("NO_RAISE")
        except M.ScipyRequiredError:
            print("RAISED_ScipyRequiredError")
        """
    )
    r = _run_without_scipy(code, tmp_path)
    assert r.returncode == 0, f"子进程失败:\n{r.stdout}\n{r.stderr}"
    out = r.stdout

    assert "PROBE_FAILED_SCIPY_IMPORTED" not in out, "屏蔽 scipy 失败，测试无效"
    assert "SCIPY_AVAILABLE = False" in out
    # C(N,5)=252, φ[10,5]=C(9,4)=126 —— 与 AFML Eq.39/40 一致
    assert "split_count = 252" in out
    assert "path_count  = 126" in out
    assert "RAISED_ScipyRequiredError" in out
    assert "NO_RAISE" not in out, "缺 scipy 时不得静默降级出近似数字"


# ── 2. 缺 scipy 时 pytest 收集不中断，CPCV 照常全跑 ───────────────────
def test_pytest_collection_does_not_abort_without_scipy(tmp_path):
    """无 scipy：pytest 不得 exit 2；CPCV 用例全跑，norm 用例被 skip。"""
    site_dir = tmp_path / "blocked"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(_BLOCK_SCIPY, encoding="utf-8")

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(site_dir)

    r = subprocess.run(
        [VENV_PY, "-m", "pytest",
         "tests/test_cpcv.py", "tests/test_overfit_audit.py",
         "-q", "--no-header", "-p", "no:cacheprovider", "-rs"],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=600,
    )
    combined = r.stdout + r.stderr

    # 关键断言：不是「收集阶段中断」
    assert "Interrupted" not in combined, f"pytest 在收集阶段被中断:\n{combined}"
    assert r.returncode != 2, f"pytest 以 exit 2 结束（收集阶段失败）:\n{combined}"
    assert r.returncode == 0, f"pytest 应成功退出:\n{combined}"

    # CPCV 的 22 个用例**一个都不能被跳过**（它们不依赖 scipy）
    summary = combined.split("short test summary")[-1]
    assert "test_cpcv" not in summary, \
        f"CPCV 用例不应被 skip —— 它们不依赖 scipy:\n{summary}"

    # 恰好 14 个用例被跳过（= 真依赖 norm 的用例数），且理由说清是缺 scipy
    assert "14 skipped" in combined, \
        f"跳过的用例数应恰好是 14（真依赖 norm 的那批）:\n{combined}"
    assert "scipy" in combined.lower(), \
        f"skip 理由必须说明是缺 scipy:\n{combined}"


# ── 3. 有 scipy 时一条都不许跳 ─────────────────────────────────────────
def test_with_scipy_nothing_is_skipped():
    """装 scipy 时，这批用例必须**全部实跑**，不许被 skip 掩盖。"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    r = subprocess.run(
        [VENV_PY, "-m", "pytest",
         "tests/test_cpcv.py", "tests/test_overfit_audit.py",
         "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=600,
    )
    combined = r.stdout + r.stderr
    assert r.returncode == 0, f"pytest 应成功退出:\n{combined}"
    assert "skipped" not in combined, \
        f"装了 scipy 却有跳过 —— 门禁被绕过:\n{combined}"


# ── 4. 依赖声明完整性：公开仓库必须写清依赖 ───────────────────────────
def test_requirements_txt_declares_all_third_party_imports():
    """requirements.txt 必须覆盖仓库里真正用到的第三方依赖。

    纪律：公开仓库缺依赖声明 = 克隆者撞 ModuleNotFoundError = 测试全红。
    这里把「仓库 import 的第三方顶层模块」与「requirements.txt 声明的包」
    做集合比对，漏声明即失败。
    """
    import ast

    req_path = os.path.join(ROOT, "requirements.txt")
    assert os.path.exists(req_path), "公开仓库必须有 requirements.txt"

    declared = set()
    with open(req_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            declared.add(line.split(">=")[0].split("<")[0].split("==")[0].split("[")[0].strip().lower())

    stdlib = set(sys.stdlib_module_names)
    local = {
        os.path.splitext(f)[0]
        for f in os.listdir(ROOT) if f.endswith(".py")
    }
    local |= {os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "tests"))
              if f.endswith(".py")}
    local |= {os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "render"))
              if f.endswith(".py")}

    used = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in {".venv", ".git", "__pycache__", "afml_ref", "zl_books"}]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        used.add(a.name.split(".")[0].lower())
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        used.add(node.module.split(".")[0].lower())

    third_party = {u for u in used
                   if u not in stdlib and u not in local and u not in declared}
    # pytest 是测试框架依赖；collectors/render 是本仓库自带的本地包
    third_party -= {"pytest", "collectors", "render"}
    assert not third_party, (
        f"以下第三方依赖没有写进 requirements.txt: {sorted(third_party)}"
    )
    assert "numpy" in declared, "numpy 必须显式声明"
    assert "scipy" in declared, "scipy 必须显式声明（overfit_audit 用到 norm）"
