"""音频工具：读写、log-mel 特征（DTW 用）、波形拼接（attacker 用）。

论文口径：40-band log-mel、16ms 帧移（窗长未披露，取 32ms/512 点，config 可调）；
操纵侧 overlap-add：12ms 窗、6ms 移步、2 帧平滑缓冲 → 边界 12ms 升余弦交叉淡化。
"""
from __future__ import annotations

import numpy as np

__all__ = ["load_wav", "save_wav", "logmel", "crossfade", "splice",
           "energy_segments", "resample_to16k"]

SR = 16000


def load_wav(path, target_sr: int = SR) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1)  # 立体声并单声道
    if sr != target_sr:
        x = resample_to16k(x, sr, target_sr)
    return x.astype(np.float32)


def save_wav(path, x: np.ndarray, sr: int = SR):
    import soundfile as sf
    # subtype 必须显式 FLOAT：WAV 默认 PCM_16 量化会摧毁 LSB 水印
    # （2026-10-30 实锤：默认写法 LSB 保持率 0.46≈随机；FLOAT=1.0。
    #  P7 及以前 res/<exp>/audio/*.wav 均为 PCM_16——水印已亡，仅作听感参考，
    #  复用须重新嵌入。）
    sf.write(str(path), np.asarray(x, np.float32), sr, subtype="FLOAT")


def resample_to16k(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    import librosa
    return librosa.resample(np.asarray(x, np.float32), orig_sr=sr_in, target_sr=sr_out)


def logmel(x: np.ndarray, sr: int = SR, n_mels: int = 40, hop_ms: float = 16.0,
           n_fft: int = 512) -> np.ndarray:
    """返回 (n_mels, T) log-mel。帧数 T ≈ len(x)/hop。"""
    import librosa
    hop = int(sr * hop_ms / 1000)
    m = librosa.feature.melspectrogram(
        y=np.asarray(x, np.float32), sr=sr, n_fft=n_fft,
        hop_length=hop, n_mels=n_mels, power=2.0)
    return np.log(m + 1e-10).astype(np.float32)


def crossfade(a: np.ndarray, b: np.ndarray, cf: int) -> np.ndarray:
    """升余弦交叉淡化拼接。cf = 淡化样本数（12ms→192 样本）。"""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if cf <= 0 or len(a) < cf or len(b) < cf:
        return np.concatenate([a, b])
    w = 0.5 * (1 - np.cos(np.pi * np.arange(1, cf + 1) / (cf + 1))).astype(np.float32)
    return np.concatenate([a[:-cf], a[-cf:] * (1 - w) + b[:cf] * w, b[cf:]])


def splice(x: np.ndarray, i0: int, i1: int, repl: np.ndarray,
           cf_samples: int = 192) -> np.ndarray:
    """把样本区间 [i0, i1) 替换为 repl（可变长，可为空 = 删除）。

    端点各自做 cf 样本升余弦淡化；repl 过短时退化为直拼。
    """
    x = np.asarray(x, np.float32)
    i0, i1 = max(0, int(i0)), min(len(x), int(i1))
    head, tail = x[:i0], x[i1:]
    repl = np.asarray(repl, np.float32)
    if len(repl) == 0:  # 删除：直接把 tail 交叉淡化到 head 尾部
        cf = min(cf_samples, len(head), len(tail))
        return crossfade(head, tail, cf)
    if len(repl) >= 2 * cf_samples:
        body = repl
    else:
        body = repl
        cf_samples = 0  # 过短不淡化
    out = crossfade(head, body, min(cf_samples, len(head), len(body)))
    return crossfade(out, tail, min(cf_samples, len(out), len(tail)))


def energy_segments(x: np.ndarray, sr: int = SR, win_ms: float = 30,
                    thresh_ratio: float = 0.10, merge_gap_ms: float = 80,
                    min_len_ms: float = 60, max_seg_ms: float = 800) -> list[tuple[int, int]]:
    """轻量能量分段（dev 数据无词级 GT 时的近似词边界）。返回样本区间列表。
    连续无停顿的信号会产生超长段 → 按 max_seg_ms 强制再切分（近似多词）。"""
    win = int(sr * win_ms / 1000)
    n = len(x) // win
    if n == 0:
        return []
    xx = np.asarray(x, np.float32)[: n * win]
    rms = np.sqrt((xx.reshape(n, win) ** 2).mean(1))
    th = rms.max() * thresh_ratio
    act = rms > th
    segs, s = [], None
    for i, a in enumerate(act):
        if a and s is None:
            s = i
        elif not a and s is not None:
            segs.append((s * win, i * win))
            s = None
    if s is not None:
        segs.append((s * win, n * win))
    merge_gap = int(sr * merge_gap_ms / 1000)
    min_len = int(sr * min_len_ms / 1000)
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    merged = [(a, b) for a, b in merged if b - a >= min_len]
    # 超长段强制再切分（无停顿信号的保护）
    max_seg = int(sr * max_seg_ms / 1000)
    out = []
    for a, b in merged:
        L = b - a
        if L <= max_seg:
            out.append((a, b))
            continue
        parts = int(np.ceil(L / max_seg))
        step = L // parts
        out += [(a + i * step, a + (i + 1) * step if i < parts - 1 else b)
                for i in range(parts)]
    return out
