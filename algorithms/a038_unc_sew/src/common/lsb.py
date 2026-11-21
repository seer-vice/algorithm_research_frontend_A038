"""LSB 嵌入/提取（论文 §III-B：k ∈ {1,2,4} bits/sample，16kHz 下容量 16k/32k/64k bps）。

约定：
- 载波统一在 int16 量化域上操作（wav 本身即 int16），浮点仅作存储表示；
- 每样本承载 k bit，样本内 bit 顺序 = 低位在前（bit0 → LSB）；
- 嵌入从样本 0 开始顺序铺满，尾部不足部分以 0 填充。
"""
from __future__ import annotations

import numpy as np

__all__ = ["float_to_int16", "int16_to_float", "embed_lsb", "extract_lsb", "lsb_capacity"]


def float_to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, np.float64) * 32767.0), -32768, 32767).astype(np.int16)


def int16_to_float(xi: np.ndarray) -> np.ndarray:
    return (xi.astype(np.float32) / 32767.0).astype(np.float32)


def lsb_capacity(n_samples: int, k: int) -> int:
    """总容量（bit 数）= 每样本 k bit。"""
    return n_samples * k


def embed_lsb(x: np.ndarray, bits: np.ndarray, k: int) -> np.ndarray:
    """把 bit 序列嵌入波形，返回水印后的浮点波形。

    x: float 波形 [-1,1]；bits: uint8 (0/1)；k: 1/2/4。
    len(bits) 必须 <= len(x)*k，多余容量补 0。
    """
    assert k in (1, 2, 4), f"k 必须是 1/2/4，收到 {k}"
    x = np.asarray(x, np.float64)
    xi = float_to_int16(x).astype(np.int32)
    n = len(xi)
    cap = n * k
    bits = np.asarray(bits, np.uint8)
    assert len(bits) <= cap, f"payload {len(bits)} bit 超出容量 {cap}"
    if len(bits) < cap:
        bits = np.concatenate([bits, np.zeros(cap - len(bits), np.uint8)])

    g = bits.reshape(n, k).astype(np.int32)
    val = np.zeros(n, np.int32)
    for j in range(k):
        val |= g[:, j] << j
    mask = (1 << k) - 1
    xi_w = ((xi & ~mask) | val).astype(np.int16)
    return int16_to_float(xi_w)


def extract_lsb(y: np.ndarray, k: int, n_bits: int | None = None) -> np.ndarray:
    """从波形提取 LSB bit 序列（对输入先做同样的 int16 量化）。"""
    assert k in (1, 2, 4)
    yi = float_to_int16(np.asarray(y, np.float64)).astype(np.int32)
    mask = (1 << k) - 1
    vals = yi & mask
    bits = ((vals[:, None] >> np.arange(k)) & 1).astype(np.uint8).reshape(-1)
    return bits if n_bits is None else bits[:n_bits]
