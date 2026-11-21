"""载荷层：R 次重复 + 每次前置 64 bit 同步字 + 多数投票（论文 §III-C/D，式(3)）。

结构（发布端铺入 LSB 通道）：
    legacy（P4/P5，cfg.payload_crc=False）：[sync64 || payload_N] × R + 零填充
    P7（cfg.payload_crc=True）：             [sync64 || crc16 || payload_N] × R + 零填充
接收端盲恢复流程：
    1) 在整条 bit 流上滑动相关搜索 sync（允许 ≤ err_max 位误码）；
    2) 相邻 sync 间距的众数 - 帧头长 → 推断载荷长度 N（删除/插入会使个别间距
       变大/变小，众数抗扰）；
    3) P7：逐份 CRC16 校验 → 只用通过校验的份投票（完整性通道：
       sync 命中但 CRC 失败 = "水印在场但被破坏" 的篡改证据）；
    4) 逐份切片，按位多数投票 → m̂；
    5) 同步搜索失败时回退为顺序布局（从头按 R×(帧头+N) 直接切）。

P7 integrity 三态：verified（载荷恢复且全部在场份 CRC 通过）/ partial
（恢复成功但有份损坏 = 报警证据）/ corrupted（sync 在场但无份通过 = 报警）/
absent（sync 全毁 = 信道死亡，guard 模式下报警）。
"""
from __future__ import annotations

import binascii

import numpy as np

__all__ = ["SYNC_WORD", "build_carrier_bits", "find_syncs", "infer_payload_len",
           "majority_vote", "extract_payload", "crc16_bits", "CRC_LEN"]

# 固定 64-bit 同步字（种子公开，保证两端一致；伪随机序列自相关性良好）
SYNC_WORD: np.ndarray = np.random.default_rng(0xA038).integers(0, 2, 64).astype(np.uint8)
SYNC_LEN = 64
CRC_LEN = 16


def crc16_bits(payload_bits: np.ndarray) -> np.ndarray:
    """CRC-CCITT(binascii.crc_hqx) over bit-packed payload → 16 bit 数组。"""
    b = np.packbits(np.asarray(payload_bits, np.uint8)).tobytes()
    v = binascii.crc_hqx(b, 0xFFFF)
    return ((v >> np.arange(15, -1, -1)) & 1).astype(np.uint8)


def build_carrier_bits(payload_bits: np.ndarray, R: int, capacity: int,
                       crc: bool = False) -> np.ndarray:
    """构造 [sync(||crc16)||payload]×R + 填充 的完整 bit 序列。"""
    payload = np.asarray(payload_bits, np.uint8)
    head = np.concatenate([SYNC_WORD, crc16_bits(payload)]) if crc else SYNC_WORD
    rep = np.concatenate([head, payload])
    carrier = np.tile(rep, R)
    assert len(carrier) <= capacity, f"载体 {len(carrier)} bit 超出 LSB 容量 {capacity}"
    if len(carrier) < capacity:
        carrier = np.concatenate([carrier, np.zeros(capacity - len(carrier), np.uint8)])
    return carrier


def _xcorr_peaks(bits: np.ndarray, err_max: int = 8):
    """滑窗匹配：与 SYNC_WORD 的逐位一致数 ≥ 64-err_max 的位置（FFT 相关加速）。"""
    sig = (np.asarray(bits, np.float32) * 2 - 1)
    syn = (SYNC_WORD.astype(np.float32) * 2 - 1)[::-1]
    n, m = len(sig), len(syn)
    L = 1 << int(np.ceil(np.log2(n + m - 1)))
    fs, fsn = np.fft.rfft(sig, L), np.fft.rfft(syn, L)
    corr = np.fft.irfft(fs * fsn, L)[: n + m - 1]  # corr[t + m - 1] = match at offset t
    matched = corr[m - 1: n]  # 与 bits[t:t+64] 的逐位一致数（-64..64）
    thresh = SYNC_LEN - 2 * err_max  # 一致数 = 64 - 2*错误数
    hits = np.where(matched >= thresh)[0]
    if len(hits) == 0:
        return []
    # 贪心非极大值抑制：按匹配数降序接受，最小间隔 64 bit
    order = hits[np.argsort(-matched[hits])]
    accepted: list[int] = []
    for p in order.tolist():
        if all(abs(p - q) >= SYNC_LEN for q in accepted):
            accepted.append(p)
    return sorted(accepted)


def find_syncs(bits: np.ndarray, err_max: int = 8) -> list[int]:
    return _xcorr_peaks(bits, err_max)


def infer_payload_len(sync_positions: list[int], total_bits: int,
                      fallback: int | None = None) -> int:
    """由相邻 sync 间距众数推断 N；不足两个 sync 时用 fallback（-64）。"""
    if len(sync_positions) >= 2:
        gaps = np.diff(np.asarray(sync_positions))
        N = int(np.bincount(gaps).argmax()) - SYNC_LEN
        if N > 0:
            return N
    if fallback is None:
        raise ValueError("无法推断载荷长度：sync 不足且未提供 fallback")
    return max(fallback - SYNC_LEN, 0)


def majority_vote(slices: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """按位多数投票。返回 (m̂, 诊断信息)。"""
    if not slices:
        raise ValueError("无有效重复份")
    M = np.stack([np.asarray(s, np.uint8) for s in slices])  # (R', N)
    votes = (M.sum(axis=0) * 2 > M.shape[0]).astype(np.uint8)  # 严格多数；平票判 0
    agree = (M == votes[None, :]).mean(axis=1)  # 每份与投票结果的一致率
    diag = {
        "n_reps": M.shape[0],
        "per_rep_agreement": agree.tolist(),
        "mean_agreement": float(agree.mean()),
    }
    return votes, diag


def extract_payload(bits: np.ndarray, capacity: int,
                    err_max: int = 8, R_hint: int | None = None,
                    crc: bool = False) -> dict:
    """接收端盲提取：sync 搜索 → N 推断 → （P7：逐份 CRC）→ 多数投票。

    bits: LSB 提取出的整条 bit 流；capacity: len(x)*k（发布端容量）。
    返回 {payload, n_reps, N, sync_positions, agreement, mode}
    P7 crc=True 额外返回 {integrity, n_corrupt}（integrity ∈
    verified/partial/corrupted/absent；n_corrupt = sync 在场但 CRC 失败份数）。
    """
    hdr = SYNC_LEN + (CRC_LEN if crc else 0)
    syncs = find_syncs(bits, err_max)
    if crc and len(syncs) >= 2:
        N = int(np.bincount(np.diff(np.asarray(syncs))).argmax()) - hdr
        if N <= 0:
            # 布局不可读（sync 间距异常）→ sync 在场但内容不可信 = 报警
            return {"payload": None, "N": 0, "mode": "failed",
                    "sync_positions": syncs, "reps_found": len(syncs),
                    "agreement": 0.0, "per_rep_agreement": np.zeros(len(syncs)),
                    "integrity": "corrupted", "n_corrupt": len(syncs)}
        slices, crc_ok = [], []
        n_trunc = 0
        for s in syncs:
            seg = bits[s + hdr: s + hdr + N]
            if len(seg) != N:
                # sync 在场但窗口被截断（如句尾删除：内容流无零填充尾即终）
                # → 完好水印不会出现（载体尾部必有零填充兜底）→ 篡改证据
                n_trunc += 1
                continue
            c = bits[s + SYNC_LEN: s + SYNC_LEN + CRC_LEN]
            ok = bool(np.array_equal(c, crc16_bits(seg)))
            crc_ok.append(ok)
            if ok:
                slices.append(seg)
        n_corrupt = int(sum(1 for ok in crc_ok if not ok)) + n_trunc
        if slices:
            payload, diag = majority_vote(slices)
            # 投票后整体 CRC 复核（多数投票可能恰好修复单份错误）
            vote_ok = bool(np.array_equal(crc16_bits(payload),
                                          crc16_bits(slices[0]))) or n_corrupt == 0
            integ = "verified" if n_corrupt == 0 and vote_ok else "partial"
            return {"payload": payload, "N": N, "mode": "sync",
                    "sync_positions": syncs, "reps_found": diag["n_reps"],
                    "agreement": diag["mean_agreement"],
                    "per_rep_agreement": diag["per_rep_agreement"],
                    "integrity": integ, "n_corrupt": n_corrupt}
        # sync 在场但无一份通过 CRC：水印在场但被破坏（篡改证据）
        return {"payload": None, "N": N, "mode": "failed",
                "sync_positions": syncs, "reps_found": len(syncs),
                "agreement": 0.0, "per_rep_agreement": np.zeros(len(syncs)),
                "integrity": "corrupted", "n_corrupt": len(syncs)}
    if len(syncs) >= 2 and not crc:
        N = infer_payload_len(syncs, len(bits))
        slices = []
        for s in syncs:
            seg = bits[s + SYNC_LEN: s + SYNC_LEN + N]
            if len(seg) == N:
                slices.append(seg)
        if len(slices) >= 1:
            payload, diag = majority_vote(slices)
            return {"payload": payload, "N": N, "mode": "sync",
                    "sync_positions": syncs, "reps_found": diag["n_reps"],
                    "agreement": diag["mean_agreement"],
                    "per_rep_agreement": diag["per_rep_agreement"]}
    if len(syncs) == 1:
        # P6 实测发现（2026-10-30）：R>=3 设计下仅 1 个 sync 命中 = 至少 R-1 份
        # 重复已被毁；且单份无投票、无 CRC，agreement 无从验证——旧版此处硬编码
        # agreement=1.0 且把乱码 bit 流照喂 decode，在 stretch105 下产生
        # "payload 完好"假阳性（实测位保持率 0.51≈随机）。降级为 failed：
        # 单 sync 不构成载荷恢复，上层走 rec=None（零重建）路径。
        # [2026-10-31 口径补丁] P7 实测（p7_channel）：stretch105 全 30 行落此
        # 分支且 return 无 integrity → extractor 填 n/a → 语义丢失（单 sync =
        # 至少 R-1 份被毁 = 篡改证据）且 verifier 的 partial/corrupted boost
        # 分支吃不到 → EER 雪上加霜。补标 corrupted（n_corrupt=1：在场那一份
        # 也无 CRC/投票佐证）。legacy（crc=False）同样走此分支——不补字段，
        # 维持 P6 复现口径不变；P7 布局才有四态。
        return {"payload": None, "N": 0, "mode": "failed",
                "sync_positions": syncs, "reps_found": 1,
                "agreement": 0.0, "per_rep_agreement": np.zeros(1),
                **({"integrity": "corrupted", "n_corrupt": 1} if crc else {})}
    if len(syncs) == 0:
        # 全部重复份被毁（如整条音频被替换）：载荷不可恢复。P7：水印死亡
        # （guard 模式下 = 信道操作证据 → 报警；论文口径下与未水印 authentic
        # 不可区分，走零重建路径由 DTW 分数兜底）。
        return {"payload": None, "N": 0, "mode": "failed",
                "sync_positions": [], "reps_found": 0,
                "agreement": 0.0, "per_rep_agreement": np.zeros(0),
                "integrity": "absent", "n_corrupt": 0}
    # 回退：顺序布局（需要发布端 R 信息）
    if R_hint is None:
        raise ValueError("sync 搜索退化且无 R_hint，无法回退顺序布局")
    mode = "sequential"
    # R_hint 与 N 未知 → 用容量自洽求解：假设填充为 0，尝试 N = capacity/R - 帧头
    N = capacity // R_hint - hdr
    slices = [bits[r * (hdr + N) + hdr: (r + 1) * (hdr + N)]
              for r in range(R_hint)]
    slices = [s for s in slices if len(s) == N]
    payload, diag = majority_vote(slices)
    return {"payload": payload, "N": N, "mode": mode, "sync_positions": [],
            "reps_found": diag["n_reps"], "agreement": diag["mean_agreement"],
            "per_rep_agreement": diag["per_rep_agreement"]}
