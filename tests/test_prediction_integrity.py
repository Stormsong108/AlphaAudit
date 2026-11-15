"""预测记录不可变性测试。

回归背景（2026-11-12）：泄漏探针 /tmp/leak_probe.py 只改了 prediction.PRED_DIR，
而 predictor 内部还持有一份 import 时绑定的独立引用，generate_predictions(force=True)
里的 path.unlink() 因此打到了真实目录，把 2019-06-19 的预测记录删掉了
（563 条判定一起丢失，靠 resolved 里存的 prediction_fingerprint 才逐位复原）。

这两条测试锁死两个不变量：
1. 预测目录只有一个所有者，任何时候都实时读它；
2. force 覆盖只改名归档，**绝不删除**不可变记录。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prediction  # noqa: E402
import predictor  # noqa: E402


def test_pred_dir_has_single_owner(tmp_path, monkeypatch):
    """predictor 取目录必须实时读 prediction.PRED_DIR，不能自己绑一份。"""
    assert not hasattr(predictor, "PRED_DIR"), \
        "predictor 不应再持有 PRED_DIR 的独立引用（那是误删真实记录的根因）"
    monkeypatch.setattr(prediction, "PRED_DIR", tmp_path)
    assert predictor._pred_dir() == tmp_path


def test_force_archives_instead_of_deleting(tmp_path, monkeypatch):
    """force 覆盖不得删除旧记录：改名归档，内容与指纹都要保住。"""
    monkeypatch.setattr(prediction, "PRED_DIR", tmp_path)
    target = tmp_path / "2019-06-19.json"
    payload = {"day": "2019-06-19", "predictions": [
        {"day": "2019-06-19", "target": "x", "target_code": "0.000001", "kind": "stock",
         "horizon_days": 5, "prob_up": 0.5, "direction": "up", "reason": "r",
         "composite_rank": 1},
    ], "meta": {"fingerprint": "abc"}}
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    before = target.read_text(encoding="utf-8")

    archived = predictor._archive_existing(target)

    assert archived is not None, "归档没发生"
    assert archived.exists(), "归档文件不存在"
    assert archived.read_text(encoding="utf-8") == before, "归档内容与原文件不一致"
    assert archived.parent == tmp_path / "_superseded"
    # 归档不能被当成一次新预测重复统计
    assert "2019-06-19.json" not in [p.name for p in tmp_path.glob("*.json")]
    assert not any(p.name.startswith("2019-06-19.superseded") for p in tmp_path.glob("*.json"))


def test_archive_is_noop_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "PRED_DIR", tmp_path)
    assert predictor._archive_existing(tmp_path / "nope.json") is None
    assert not (tmp_path / "_superseded").exists()


def test_real_prediction_files_are_intact():
    """真实目录里的每一天预测都必须还在，且指纹自洽。"""
    pdir = ROOT / "data" / "predictions"
    files = sorted(pdir.glob("*.json"))
    assert len(files) >= 190, f"预测文件只有 {len(files)} 个，疑似被删"
    # 归档目录不能混进主目录统计
    assert not any("_superseded" in p.name for p in files)
    # 抽查一天：指纹能重算出来
    d = json.loads(files[0].read_text(encoding="utf-8"))
    fp = prediction.fingerprint(d["predictions"])
    assert fp == d["meta"]["fingerprint"], "预测文件指纹对不上（内容被改过？）"
