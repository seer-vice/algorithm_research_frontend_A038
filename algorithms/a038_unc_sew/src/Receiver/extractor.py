"""接收方 · 载荷提取与自重建：LSB 提取 → sync 搜索/间距推 N → 多数投票 → codec 解码。

论文 §III：提取端是盲的（只依赖 config 的 k 与 codec），N/R 由 sync 结构自恢复。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common import lsb, payload as pl
from src.common.codec_io import CodecBase

__all__ = ["ExtractResult", "PayloadExtractor"]


@dataclass
class ExtractResult:
    payload: np.ndarray | None    # 恢复的载荷 bit
    n_payload_bits: int
    mode: str                     # sync / sequential / failed
    reps_found: int
    agreement: float              # 多数投票平均一致率（载荷健康度）
    reconstruction: np.ndarray | None  # R(y) 自重建波形
    integrity: str = "n/a"        # P7：verified/partial/corrupted/absent（legacy=n/a）
    n_corrupt: int = -1           # P7：sync 在场但 CRC 失败份数（-1=legacy 无 CRC）
    carrier_pattern: np.ndarray | None = None  # P7 标尺：[sync||crc||payload] 单周期
    per_rep: int = 0              # P7 标尺：单周期 bit 数（= 帧头 + N）


class PayloadExtractor:
    def __init__(self, cfg, codec: CodecBase):
        self.cfg = cfg
        self.codec = codec
        self.err_max = int(getattr(cfg, "sync_err_max", 8))
        self.crc = bool(getattr(cfg, "payload_crc", False))

    def extract(self, y: np.ndarray) -> ExtractResult:
        y = np.asarray(y, np.float32)
        k = int(self.cfg.lsb_k)
        bits = lsb.extract_lsb(y, k)
        C = len(bits)
        # N 上界：容量至少容纳 1 份（N + 帧头）；同步搜索自动发现实际 N
        res = pl.extract_payload(bits, capacity=C, err_max=self.err_max,
                                 crc=self.crc)
        payload = res["payload"]
        rec = None
        agree = float(res.get("agreement", 0.0))
        n = int(res.get("N", 0))
        if payload is not None and n % 8 == 0 and n // 8 > 0:
            try:
                rec = self.codec.decode(payload[: n // 8 * 8])
            except Exception:
                rec = None
        # P7 标尺：载荷恢复 → 单周期 pattern（verifier 用于 LSB 一致率定位）
        pattern, per_rep = None, 0
        if payload is not None:
            head = (np.concatenate([pl.SYNC_WORD, pl.crc16_bits(payload)])
                    if self.crc else pl.SYNC_WORD)
            pattern = np.concatenate([head, np.asarray(payload, np.uint8)])
            per_rep = len(pattern)
        return ExtractResult(payload=payload, n_payload_bits=n,
                             mode=res["mode"], reps_found=res["reps_found"],
                             agreement=agree, reconstruction=rec,
                             integrity=res.get("integrity", "n/a"),
                             n_corrupt=int(res.get("n_corrupt", -1)),
                             carrier_pattern=pattern, per_rep=per_rep)
