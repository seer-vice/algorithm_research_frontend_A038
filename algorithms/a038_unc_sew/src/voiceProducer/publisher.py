"""发布方（论文 §III 流程）：codec 编码 → R 次重复（含 64bit 同步字）→ LSB 嵌入。

扩展接口（预留）：
- cfg.payload_mode = 'codec' | 'hash'（P6 对照：hash 基线上限，只检测不恢复）；
- Publisher.embed_hook(bits, x) → x̃：自定义嵌入器（未来鲁棒水印替换 LSB）；
- Publisher.post_embed(x̃)：水印后处理钩子。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.common import lsb, payload as pl
from src.common.codec_io import CodecBase, build_codec
from src.common.audio_utils import SR

__all__ = ["Publisher", "PublishResult"]


@dataclass
class PublishResult:
    uid: str
    xw: np.ndarray                 # 水印后波形 x̃
    n_payload_bits: int            # N
    repetitions: int               # R
    lsb_k: int
    payload_sha: str               # 恢复校验用
    layout: str = "sync"


class Publisher:
    def __init__(self, cfg, codec: CodecBase | None = None):
        self.cfg = cfg
        self.codec = codec or build_codec(cfg)
        self.embed_hook = None      # 扩展点：自定义嵌入器
        self.post_embed = None      # 扩展点：水印后处理
        assert getattr(cfg, "payload_mode", "codec") == "codec", \
            "hash 基线在 P6 对照实验接入（payload_mode='hash'）"

    def publish(self, uid: str, x: np.ndarray) -> PublishResult:
        x = np.asarray(x, np.float32)
        # 1) codec 编码 → payload bits
        bits = self.codec.encode(x)
        N = len(bits)
        # 2) 容量与重复数（论文式：R = ⌊C/(N+帧头)⌋，C = 16000·k·T；
        #    P7 crc 模式帧头 = sync64 + crc16）
        k = int(self.cfg.lsb_k)
        crc = bool(getattr(self.cfg, "payload_crc", False))
        C = lsb.lsb_capacity(len(x), k)
        R = C // (N + pl.SYNC_LEN + (pl.CRC_LEN if crc else 0))
        assert R >= 1, f"容量不足：C={C}, N={N}（音频太短或 k 太小）"
        # 3) 构造载体 bit 流并嵌入
        carrier = pl.build_carrier_bits(bits, R, C, crc=crc)
        xw = self.embed_hook(carrier, x) if self.embed_hook else lsb.embed_lsb(x, carrier, k)
        if self.post_embed:
            xw = self.post_embed(xw)
        import hashlib
        sha = hashlib.sha256(bits.tobytes()).hexdigest()[:16]
        return PublishResult(uid=uid, xw=xw, n_payload_bits=N, repetitions=R,
                             lsb_k=k, payload_sha=sha)
