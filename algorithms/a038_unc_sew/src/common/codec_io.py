"""codec 统一抽象 + 整数位打包工具。

论文口径（Table I 对齐）：
- SemantiCodec 0.65kbps：token ∈ [0, 8192)，每 token 13 bit（2^13 恰好覆盖词表），
  ρ = 50 token/s × 13 bit = 650 bps → k=1 时 R = ⌊16000/(650+64)⌋ = 22（论文一致）。
- SNAC / TAAE：P4 阶段接入（TAAE 需 WSL2，见交付文档）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["ints_to_bits", "bits_to_ints", "CodecBase", "SemantiCodecCodec",
           "MockCodec", "build_codec"]


# ---------------- 位打包（MSB 在前） ----------------

def ints_to_bits(vals: np.ndarray, nbits: int) -> np.ndarray:
    vals = np.asarray(vals, np.int64)
    assert vals.min() >= 0 and vals.max() < (1 << nbits), "整数超出 nbits 表示范围"
    shifts = np.arange(nbits - 1, -1, -1)
    return ((vals[:, None] >> shifts[None, :]) & 1).astype(np.uint8).reshape(-1)


def bits_to_ints(bits: np.ndarray, nbits: int) -> np.ndarray:
    bits = np.asarray(bits, np.uint8)
    n = len(bits) // nbits * nbits
    bits = bits[:n].reshape(-1, nbits)
    w = (1 << np.arange(nbits - 1, -1, -1)).astype(np.int64)
    return (bits.astype(np.int64) * w[None, :]).sum(axis=1)


# ---------------- codec 抽象 ----------------

class CodecBase:
    name = "base"
    bits_per_second: float = 0.0  # 实际嵌入码率 ρ（用于 R 计算）

    def encode(self, wav: np.ndarray) -> np.ndarray:
        """float 波形 → payload bit 序列（uint8 0/1）。"""
        raise NotImplementedError

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """payload bit 序列 → float 波形（16kHz）。"""
        raise NotImplementedError


class SemantiCodecCodec(CodecBase):
    """SemantiCodec 0.65kbps（token_rate=50, vocab=8192, 13bit/token）。

    语义码 + 两码本 token 打平为一维序列；encoder/decoder 走官方推理包。
    encode 内部经临时 wav 文件（官方 API 只收 filepath）。
    """
    name = "semanticodec"
    TOKEN_BITS = 13
    TOKENS_PER_STEP = 2  # 官方 token 张量 [1, T, 2]

    def __init__(self, token_rate: int = 50, vocab: int = 8192,
                 ddim_sample_step: int = 50, cfg_scale: float = 2.0,
                 cache_path: str | None = None):
        self.token_rate, self.vocab = token_rate, vocab
        self._kw = dict(token_rate=token_rate, semantic_vocab_size=vocab,
                        ddim_sample_step=ddim_sample_step, cfg_scale=cfg_scale)
        if cache_path:
            self._kw["cache_path"] = cache_path
        self._codec = None
        self._sr = 16000
        # ρ 估算：名义 50 token/s × 13 bit（实际按每条音频 encode 结果定 N）
        self.bits_per_second = token_rate * self.TOKEN_BITS

    def _lazy(self):
        if self._codec is None:
            from semanticodec import SemantiCodec  # 延迟导入，CLI 不拖模型
            self._codec = SemantiCodec(**self._kw)
        return self._codec

    def encode(self, wav: np.ndarray) -> np.ndarray:
        import soundfile as sf
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "in.wav"
            sf.write(p, np.asarray(wav, np.float32), self._sr)
            tokens = self._lazy().encode(str(p))          # [1, T, 2] long（在 codec.device 上）
        if hasattr(tokens, "detach"):                     # v2: torch 张量（含 cuda）→ CPU numpy
            tokens = tokens.detach().cpu().numpy()
        vals = np.asarray(tokens).reshape(-1).astype(np.int64)
        assert vals.max() < (1 << self.TOKEN_BITS)
        return ints_to_bits(vals, self.TOKEN_BITS)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        vals = bits_to_ints(bits, self.TOKEN_BITS)
        steps = len(vals) // self.TOKENS_PER_STEP
        tokens = vals[: steps * self.TOKENS_PER_STEP].reshape(1, steps, self.TOKENS_PER_STEP)
        import torch
        t = torch.from_numpy(tokens).long()
        codec = self._lazy()
        dev = getattr(codec, "device", None)              # v2: 送回 codec 设备（GPU）
        if dev is not None and str(dev) != "cpu":
            t = t.to(dev)
        wave = codec.decode(t)
        out = wave[0, 0]
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        return np.asarray(out, np.float32)


class MockCodec(CodecBase):
    """测试用假 codec：DS 倍降采样 + QBITS 位量化（默认 250Hz×8bit=2kbps）。
    码率压低保证短音频也有 R≥3 份重复可投票；测试可通过 DS=16 提升保真度。"""
    name = "mock"

    def __init__(self, DS: int = 64, QBITS: int = 8):
        self.DS = int(DS)
        self.QBITS = int(QBITS)
        self.bits_per_second = (16000 // self.DS) * self.QBITS

    def encode(self, wav: np.ndarray) -> np.ndarray:
        x = np.asarray(wav, np.float64)
        n = len(x) // self.DS * self.DS
        q = np.round((x[:n].reshape(-1, self.DS).mean(1) + 1) / 2 * 255).astype(np.int64)
        return ints_to_bits(q, self.QBITS)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        q = bits_to_ints(bits, self.QBITS)
        v = (q.astype(np.float64) / 255 * 2 - 1)
        return np.repeat(v, self.DS).astype(np.float32)


class SnacCodec(CodecBase):
    """SNAC 24kHz（hubertsiuzdak/snac_24khz，3 层 RVQ，词表 4096 = 12bit/token）。

    实测（2026-10-30 沙箱）：层结构恒 (n0, 2n0, 4n0) → total = 7·n0，解码端
    由 payload 长度自反推层切分（N/12/7）；ρ ≈ 991 bps（≈1kbps 档，高于
    SemantiCodec 0.65kbps）；CPU 3-5s/句（无扩散，可 CPU 冒烟）。
    工程管线 16kHz：encode 前 16k→24k 上采样，decode 后回 24k→16k。
    HF_ENDPOINT 未设时权重可能直连超时（同 SemantiCodec 坑）。
    """
    name = "snac"
    TOKEN_BITS = 12
    SR_NATIVE = 24000
    SR_PIPE = 16000
    bits_per_second = 991.0

    def __init__(self, repo_id: str = "hubertsiuzdak/snac_24khz"):
        import torch  # 惰性：mock 冒烟不需要 torch
        from snac import SNAC
        self._torch = torch
        self.model = SNAC.from_pretrained(repo_id).eval()

    def encode(self, wav: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly
        x = np.asarray(wav, np.float32)
        x24 = resample_poly(x, 3, 2)  # 16k → 24k
        t = self._torch.tensor(x24)[None, None, :]
        with self._torch.no_grad():
            codes = self.model.encode(t)
        vals = np.concatenate([np.asarray(c.reshape(-1), np.int64) for c in codes])
        assert vals.min() >= 0 and vals.max() < (1 << self.TOKEN_BITS), "token 超出 12bit"
        return ints_to_bits(vals, self.TOKEN_BITS)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly
        vals = bits_to_ints(np.asarray(bits, np.uint8), self.TOKEN_BITS)
        total = len(vals)
        assert total % 7 == 0, f"SNAC token 数 {total} 不是 7 的倍数（层结构 1:2:4）"
        n0 = total // 7
        t = self._torch
        codes = [t.tensor(vals[:n0].reshape(1, n0), dtype=t.long),
                 t.tensor(vals[n0:3 * n0].reshape(1, 2 * n0), dtype=t.long),
                 t.tensor(vals[3 * n0:].reshape(1, 4 * n0), dtype=t.long)]
        with t.no_grad():
            y = self.model.decode(codes)
        y = np.asarray(y.reshape(-1), np.float32)
        return resample_poly(y, 2, 3).astype(np.float32)  # 24k → 16k


def build_codec(cfg) -> CodecBase:
    import os
    name = getattr(cfg, "codec", "mock")
    if name == "mock":
        return MockCodec()
    if name == "semanticodec":
        kw = dict(token_rate=getattr(cfg, "sc_token_rate", 50),
                  vocab=getattr(cfg, "sc_vocab", 8192))
        # A038_PRETRAINED：SemantiCodec 权重目录（cache_path 是 CWD 相对路径的坑，
        # 换目录跑必触发 3GB 重下——用绝对路径指向已有权重即可免下）
        cp = os.environ.get("A038_PRETRAINED")
        if cp:
            kw["cache_path"] = cp
        return SemantiCodecCodec(**kw)
    if name == "snac":
        return SnacCodec()
    raise ValueError(f"未知 codec: {name}")
