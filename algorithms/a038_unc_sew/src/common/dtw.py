"""DTW（论文 §III-F）：40-band log-mel、16ms 帧移、局部代价 = 余弦距离。

- 轴约定：cost[i, j] = cos_dist(mel_y[:, i], mel_rec[:, j])；y 为查询轴；
- 标准对称 DTW（无 gap 惩罚），路径单调覆盖每个 y 帧 ≥ 1 次；
- 逐帧代价：y 帧 i 上路径代价的均值（无匹配帧按 0 代价对齐）；
- numba 不可用时退化为纯 numpy（慢，仅测试用）。
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if (a and callable(a[0])) else deco


__all__ = ["cosine_dist_matrix", "dtw_align", "frame_costs", "smooth", "top_pct_score"]


def cosine_dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A (d,N), B (d,M) → (N,M) 余弦距离 = 1 - cos。"""
    A = A + 1e-10
    B = B + 1e-10
    an = A / np.linalg.norm(A, axis=0, keepdims=True)
    bn = B / np.linalg.norm(B, axis=0, keepdims=True)
    return np.clip(1.0 - an.T @ bn, 0.0, 2.0).astype(np.float32)


@njit(cache=True)
def _dtw_core(cost: np.ndarray):
    N, M = cost.shape
    D = np.full((N + 1, M + 1), np.inf, np.float64)
    bp = np.zeros((N + 1, M + 1), np.int8)
    D[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            c = cost[i - 1, j - 1]
            best, src = D[i - 1, j - 1] + c, 1  # 对角
            if D[i - 1, j] + c < best:
                best, src = D[i - 1, j] + c, 2  # 上（y 帧独占）
            if D[i, j - 1] + c < best:
                best, src = D[i, j - 1] + c, 3  # 左（rec 帧独占）
            D[i, j], bp[i, j] = best, src
    return D, bp


def dtw_align(cost: np.ndarray):
    """返回 (总代价, 回溯路径 [(i,j)...]，i 为 y 帧下标)。"""
    N, M = cost.shape
    D, bp = _dtw_core(cost.astype(np.float32))
    i, j, path = N, M, []
    while (i, j) != (0, 0):
        path.append((i - 1, j - 1))
        s = bp[i, j]
        if s == 1:
            i, j = i - 1, j - 1
        elif s == 2:
            i = i - 1
        else:
            j = j - 1
    path.reverse()
    return float(D[N, M]), path


def frame_costs(cost: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    """y 轴逐帧代价（该帧所有路径点的局部代价均值）。"""
    N = cost.shape[0]
    sums = np.zeros(N, np.float64)
    counts = np.zeros(N, np.int64)
    for i, j in path:
        sums[i] += cost[i, j]
        counts[i] += 1
    out = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return out.astype(np.float32)


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """滑动平均（w 为帧数，奇数居中）。"""
    if w <= 1:
        return x
    k = np.ones(int(w), np.float32) / int(w)
    return np.convolve(x, k, mode="same").astype(np.float32)


def top_pct_score(costs: np.ndarray, pct: float = 0.01, norm: str = "raw") -> float:
    """语句级检测得分 = 平滑后逐帧代价 top pct 均值（论文口径）。

    norm: "raw" 原始值 | "median"/"mean" 逐句自归一（÷全帧中位/均值）——
    2026-10-25 p4_matrix 200 条实测：authentic 分数长右尾（med 0.011/mean 0.016，
    codec 重建误差逐句基线差），归一后总 EER 0.25→0.19、tts 0.22→0.11。
    """
    n = max(1, int(round(len(costs) * pct)))
    s = float(np.sort(costs)[-n:].mean())
    if norm == "median":
        return s / (float(np.median(costs)) + 1e-8)
    if norm == "mean":
        return s / (float(np.mean(costs)) + 1e-8)
    if norm != "raw":
        raise ValueError(f"未知 score_norm: {norm}（raw | median | mean）")
    return s
