"""逻辑回归（无外部依赖）+ 概率校准。

为什么自己实现：
    环境里没有 sklearn/scipy。这个实现是完整的批量梯度下降 + L2 正则，
    带 Newton 步长自适应的稳定版本，足够用于千级样本、十几个特征的二分类。

为什么必须做概率校准：
    模型输出 0.63 不代表"真的 63% 会涨"。原始 logistic 分数通常是过自信的。
    校准（isotonic 或 Platt）把原始分数映射到真实频率，
    这样 Brier score 和校准曲线才有意义。

严格防泄漏：
    训练、验证、测试三段严格按时间切分，绝不随机打乱（金融数据有强时序性）。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# ------------------------------------------------------------------ 逻辑回归


class LogisticRegression:
    """L2 正则化的逻辑回归，带 Newton 步长自适应。

    特征必须已标准化（均值 0、标准差 1），由外部传入 scaler 参数。
    """

    def __init__(
        self,
        *,
        l2: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-8,
        lr: float = 0.5,
    ):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0
        self.converged_: bool = False

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # 数值稳定版 sigmoid，避免 exp 溢出
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """X: [n, d] 已标准化；y: [n] ∈ {0,1}"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        if n == 0:
            raise ValueError("空样本")
        # 截距不参与正则
        A = np.hstack([np.ones((n, 1)), X])
        w = np.zeros(d + 1)
        reg = np.eye(d + 1) * self.l2
        reg[0, 0] = 0.0
        prev_loss = None
        for it in range(self.max_iter):
            p = self._sigmoid(A @ w)
            # 稳定计算 log-loss
            eps = 1e-12
            p_c = np.clip(p, eps, 1 - eps)
            loss = -np.mean(y * np.log(p_c) + (1 - y) * np.log(1 - p_c))
            loss += 0.5 * self.l2 * float(w[1:] @ w[1:]) / n
            grad = A.T @ (p - y) / n
            grad[1:] += self.l2 * w[1:] / n
            # 自适应步长
            lr = self.lr
            for _ in range(30):
                w_new = w - lr * grad
                p_new = self._sigmoid(A @ w_new)
                p_new = np.clip(p_new, eps, 1 - eps)
                l_new = -np.mean(y * np.log(p_new) + (1 - y) * np.log(1 - p_new))
                l_new += 0.5 * self.l2 * float(w_new[1:] @ w_new[1:]) / n
                if l_new <= loss + 1e-12:
                    break
                lr *= 0.5
            if abs(loss - l_new) < self.tol and prev_loss is not None:
                w = w_new
                self.n_iter_ = it + 1
                self.converged_ = True
                break
            w = w_new
            prev_loss = loss
            self.n_iter_ = it + 1
        self.coef_ = w[1:]
        self.intercept_ = float(w[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型未训练")
        X = np.asarray(X, dtype=float)
        z = X @ self.coef_ + self.intercept_
        return self._sigmoid(z)


# ------------------------------------------------------------------ 校准


def fit_platt_scaler(
    raw_scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Platt scaling：用逻辑回归把原始分数映射到概率。

    raw_scores: [n] 模型原始输出（概率或分数）
    labels:     [n] ∈ {0,1}
    返回 (a, b)，使 calibrated = sigmoid(a * raw + b)
    """
    raw = np.asarray(raw_scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(raw) < 20:
        return 1.0, 0.0
    lr = LogisticRegression(l2=1e-6, max_iter=300)
    lr.fit(raw.reshape(-1, 1), y)
    return float(lr.coef_[0]), float(lr.intercept_)


def fit_isotonic(raw_scores: np.ndarray, labels: np.ndarray) -> list[tuple[float, float]]:
    """保序回归（PAV 算法）做校准。

    返回有序的 (x_threshold, y_prob) 断点，预测时做线性插值 + 端点外推。
    """
    raw = np.asarray(raw_scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(raw) < 20:
        return [(float(raw.min()) if len(raw) else 0.0, 0.5),
                (float(raw.max()) if len(raw) else 1.0, 0.5)]
    order = np.argsort(raw)
    xs = raw[order]
    ys = y[order]

    # PAV: 合并违反单调的块
    vals: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for i in range(len(xs)):
        vals.append(float(ys[i]))
        weights.append(1.0)
        starts.append(i)
        ends.append(i)
        while len(vals) >= 2 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), weights.pop()
            s2, e2 = starts.pop(), ends.pop()
            v1, w1 = vals.pop(), weights.pop()
            s1, e1 = starts.pop(), ends.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            weights.append(w1 + w2)
            starts.append(s1)
            ends.append(e2)
    # 断点：块起点 x，块均值 y
    out: list[tuple[float, float]] = []
    for s, v in zip(starts, vals):
        x = float(xs[s])
        if out and abs(out[-1][0] - x) < 1e-12:
            out[-1] = (x, v)
        else:
            out.append((x, v))
    return out


def apply_isotonic(cal: list[tuple[float, float]], scores: np.ndarray) -> np.ndarray:
    """应用保序校准。"""
    if not cal:
        return np.asarray(scores, dtype=float)
    xs = np.array([c[0] for c in cal], dtype=float)
    ys = np.array([c[1] for c in cal], dtype=float)
    s = np.asarray(scores, dtype=float)
    return np.interp(s, xs, ys, left=float(ys[0]), right=float(ys[-1]))


# ------------------------------------------------------------------ 时间序列 CV


def walk_forward_split(
    n: int,
    *,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
    min_train: int = 200,
) -> list[tuple[int, int, int]]:
    """按时间切分（前=训练，中=验证，后=测试）。返回 [(tr_end, va_end, te_end)]。

    这不是标准 CV（金融数据不能随机打乱），而是"滚动扩展窗口"的简化版：
    一段训练 → 一段验证 → 一段测试，然后整体向后滚动。
    """
    splits: list[tuple[int, int, int]] = []
    train_size = int(n * train_frac)
    val_size = int(n * val_frac)
    if train_size < min_train:
        return []
    # 至少 3 段滚动
    step = max(1, (n - train_size - val_size) // 3)
    end = train_size
    while end + val_size < n:
        splits.append((end, end + val_size, n))
        end += step
        if len(splits) >= 4:
            break
    return splits


def evaluate_probability_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
) -> dict[str, Any]:
    """完整的时间序列样本外评估。

    流程：
        1. 前 train_frac 训练
        2. 中 val_frac 拟合校准器
        3. 后段（从未见过）作为测试，输出 Brier / 命中率 / 校准曲线

    返回的指标全部来自测试段，是真正的样本外表现。
    """
    n = len(y)
    splits = walk_forward_split(n, train_frac=train_frac, val_frac=val_frac)
    if not splits:
        return {"ok": False, "reason": "样本不足，无法切分"}
    tr_end, va_end, te_end = splits[0]
    if te_end <= va_end:
        return {"ok": False, "reason": "无测试段"}

    Xtr, ytr = X[:tr_end], y[:tr_end]
    Xva, yva = X[tr_end:va_end], y[tr_end:va_end]
    Xte, yte = X[va_end:te_end], y[va_end:te_end]

    model = LogisticRegression(l2=1.0, max_iter=300)
    model.fit(Xtr, ytr)
    raw_va = model.predict_proba(Xva)
    raw_te = model.predict_proba(Xte)

    # 校准：用验证段
    a, b = fit_platt_scaler(raw_va, yva)
    iso = fit_isotonic(raw_va, yva)
    cal_te = apply_isotonic(iso, raw_te)
    platt_te = 1.0 / (1.0 + np.exp(-(a * raw_te + b)))

    def metrics(probs: np.ndarray, truth: np.ndarray) -> dict[str, float]:
        if len(truth) == 0:
            return {}
        bs = float(np.mean((probs - truth) ** 2))
        hit = float(np.mean((probs > 0.5) == (truth > 0.5)))
        base = float(max(np.mean(truth), 1 - np.mean(truth)))
        return {
            "n": len(truth),
            "brier": bs,
            "hit_rate": hit,
            "baseline": base,
            "edge": hit - base,
            "avg_prob": float(np.mean(probs)),
            "actual_rate": float(np.mean(truth)),
        }

    return {
        "ok": True,
        "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "raw": metrics(raw_te, yte),
        "platt": metrics(platt_te, yte),
        "isotonic": metrics(cal_te, yte),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_,
    }
