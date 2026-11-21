"""接收方 · 判决：DTW 逐帧代价 → 语句级得分（top-1% 均值）+ 帧级定位曲线 + 恢复波形。

论文 §III-F/§IV：
- 得分 s(y) = mean(top 1% of 平滑逐帧代价)；语句级 EER 由 metrics 聚合；
- 帧级：代价投影在 y 时间轴上，与词级 GT 做 AUC；
- 恢复：R(y) = codec 解码的自重建（P5 做"仅替换检出区域"精细恢复）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.Receiver.extractor import ExtractResult, PayloadExtractor
from src.common import audio_utils as au, dtw as dtw_mod

__all__ = ["VerifyResult", "Verifier"]


@dataclass
class VerifyResult:
    uid: str
    attack: str
    label: int
    score: float                  # 语句级检测得分
    frame_costs: np.ndarray       # 平滑后逐帧代价（16ms 网格）
    frame_times: np.ndarray       # 秒
    payload_mode: str
    payload_agreement: float
    reconstruction: np.ndarray | None
    recovered: np.ndarray | None = None     # 恢复输出（P5 起按区域替换）
    gt_segments: list = field(default_factory=list)
    # P7 LSB 标尺通道（cfg.lsb_fusion=True 时有效）
    score_raw: float = -1.0       # DTW-only 得分（paper 口径对照）
    a_min: float = -1.0           # 帧级 LSB 一致率最小值（1.0=完好）
    junction_s: float = -1.0      # 接缝定位（y 时间轴，秒；-1=未检出）
    integrity: str = "n/a"
    n_corrupt: int = -1


class Verifier:
    def __init__(self, cfg, codec):
        self.cfg = cfg
        self.extractor = PayloadExtractor(cfg, codec)
        self.n_mels = int(getattr(cfg, "n_mels", 40))
        self.hop_ms = float(getattr(cfg, "hop_ms", 16))
        self.smooth_w = int(getattr(cfg, "smooth_frames", 31))
        self.top_pct = float(getattr(cfg, "top_pct", 0.01))
        self.score_norm = str(getattr(cfg, "score_norm", "median"))   # raw|median|mean（v2 口径）
        self.lsb_fusion = bool(getattr(cfg, "lsb_fusion", False))
        self.lsb_boost_w = float(getattr(cfg, "lsb_boost_w", 1.0))

    def verify(self, uid: str, attacked, use_rec: bool = True) -> VerifyResult:
        y = np.asarray(attacked.y, np.float32)
        ex: ExtractResult = self.extractor.extract(y)
        rec = ex.reconstruction if use_rec else None
        if rec is None:  # 载荷恢复失败 → 以零重建（代价全高）
            rec = np.zeros_like(y)
        L = min(len(y), len(rec))
        my = au.logmel(y[:L], n_mels=self.n_mels, hop_ms=self.hop_ms)
        mr = au.logmel(rec[:L], n_mels=self.n_mels, hop_ms=self.hop_ms)
        cost = dtw_mod.cosine_dist_matrix(my, mr)
        _, path = dtw_mod.dtw_align(cost)
        fc = dtw_mod.frame_costs(cost, path)
        fc = dtw_mod.smooth(fc, self.smooth_w)
        n_frames = len(fc)
        score = dtw_mod.top_pct_score(fc, self.top_pct, norm=self.score_norm)
        # P7 LSB 标尺通道：载荷在手 → y 的 LSB 与重建 pattern 逐帧一致率。
        # 论文威胁模型（词级操纵，不重绘 LSB）下，任何局部篡改在接缝处留下
        # 一致率断崖：删/插 → 接缝后整体失配；换段 → 失配段。诚实边界：
        # 攻击者可提取并重绘 LSB（水印无密钥），此时本通道失效退回 DTW 基线。
        a_min, junction_s, boost = -1.0, -1.0, 0.0
        if self.lsb_fusion and ex.carrier_pattern is not None:
            a_min, junction_s = self._lsb_profile(y, ex)
            if a_min >= 0:
                boost = self.lsb_boost_w * (1.0 - a_min)
            # 证据兜底（句尾删除类：A 剖面全吻合但 CRC/截断已抓到篡改）：
            # 无 dip 且 integrity≠verified → boost 记 0.5、接缝取内容末端
            if ex.integrity in ("partial", "corrupted"):
                boost = max(boost, 0.5 * self.lsb_boost_w)
                if junction_s < 0:
                    from src.common import lsb as _lsb
                    by = np.asarray(_lsb.extract_lsb(y, int(self.cfg.lsb_k)), np.int8)
                    nz = np.flatnonzero(by != 0)
                    if len(nz):
                        junction_s = float((nz[-1] + 1) / au.SR)
        vr = VerifyResult(
            uid=uid, attack=attacked.attack, label=attacked.label,
            score=score + boost,
            frame_costs=fc,
            frame_times=np.arange(n_frames) * self.hop_ms / 1000.0,
            payload_mode=ex.mode,
            payload_agreement=ex.agreement,
            reconstruction=ex.reconstruction,
            recovered=self._recover(y, fc, ex, path),
            gt_segments=list(attacked.gt_segments),
            score_raw=score if self.lsb_fusion else -1.0,
            a_min=a_min, junction_s=junction_s,
            integrity=ex.integrity, n_corrupt=ex.n_corrupt)
        return vr

    def _lsb_profile(self, y, ex: ExtractResult):
        """帧级 LSB 一致率 + 接缝定位。

        原理：发布端 carrier = [sync||crc||payload] 周期平铺 + 尾部零填充；
        接收端载荷恢复后重建同 pattern。y 未被篡改的区段逐位吻合（一致率
        ≈1.0）；删除/插入接缝之后整体位移失配（≈0.5），替换段内容非发布
        端 bit（≈0.5）。首帧跌破阈值 = 接缝。尾部零填充先剔除（无信息量）。
        """
        from src.common import lsb as lsb_mod
        k = int(self.cfg.lsb_k)
        by = np.asarray(lsb_mod.extract_lsb(y, k), np.int8)
        # 剔除尾部零填充（painted zeros 无定位信息）
        nz = np.flatnonzero(by != 0)
        n_eff = (nz[-1] + 1) if len(nz) else 0
        if n_eff < 2 * ex.per_rep:
            return -1.0, -1.0
        pos = np.arange(n_eff)
        cyc = np.asarray(ex.carrier_pattern, np.int8)
        ref = cyc[pos % ex.per_rep]
        match = (by[:n_eff] == ref).astype(np.float32)
        # 帧网格：16ms × SR/1000 bit/帧（与 fc 网格同构）
        bits_per_frame = max(1, int(self.hop_ms / 1000 * au.SR))
        n_f = n_eff // bits_per_frame
        if n_f < 1:
            return -1.0, -1.0
        A = match[: n_f * bits_per_frame].reshape(n_f, bits_per_frame).mean(1)
        a_min = float(A.min()) if len(A) else -1.0
        # 接缝 = 首个跌破阈值的帧（阈值 0.75 = 1.0 与 0.5 的中点）
        low = np.flatnonzero(A < 0.75)
        junction_s = float(low[0] * self.hop_ms / 1000.0) if len(low) else -1.0
        return a_min, junction_s

    def _recover(self, y, fc, ex: ExtractResult, path=None):
        """编辑序列式恢复（v2，2026-11-21 重写）。

        v1（同索引替换 out[i]=rec[i]）只在 direct 等长攻击下对齐；insert/delete
        后 y 与 R(y) 时间轴整体错位——插入段不消失、接缝后内容错配。
        v2 按 DTW 对齐路径重建（帧域判定 → 样本域拼接）：
          - 好区（fc < p99）：保留 y 原样（未篡改区保持原始质量）；
          - 替换段（bad 且 y↔R 对齐跨度相当，如 direct）：填 R 对应段；
          - 插入段（bad 且 y 帧被 DTW 压缩映射到 R 窄区间）：整段丢弃 → 恢复
            原时长（"插入的地方删掉"）；
          - 删除段（R 帧在 path 上无 y 覆盖的空洞）：补回 R 对应内容。
        恢复上限 = codec 自重建（0.65kbps，PESQ≈1.63，论文 §III-F 口径），
        非原始录音质量。path=None 时退回 v1 行为（兼容旧调用方）。
        """
        if ex.reconstruction is None:
            return None
        rec = np.asarray(ex.reconstruction, np.float32)
        y = np.asarray(y, np.float32)
        if path is None:                      # v1 兼容路径
            return self._recover_v1(y, rec, fc)
        hop = int(self.hop_ms / 1000 * au.SR)
        n_f = len(fc)
        thr = float(np.quantile(fc, 0.99)) if n_f else 1.0
        # path → y帧/R帧 双向映射（path 为单调 (y_frame, rec_frame) 对）
        j_of_i = {}
        for i, j in path:
            j_of_i.setdefault(int(i), []).append(int(j))
        bad = fc >= thr
        cap = len(y) + len(rec) + 4 * hop
        out = np.zeros(cap, np.float32)
        w = 0
        segs = []                              # [(段, 源)]：源切换边界 12ms OLA
        def _put(seg, src):
            segs.append((np.asarray(seg, np.float32), src))
        i = 0
        last_j = -1                            # 已消费到的 R 帧右界（删除空洞检测）
        while i < n_f:
            js = j_of_i.get(i, [])
            if js:
                j_start = min(js)
                if last_j >= 0 and j_start > last_j + 1:
                    # R 帧 [last_j+1, j_start) 无 y 覆盖 → 被删内容：补回
                    a, b = (last_j + 1) * hop, min(len(rec), j_start * hop)
                    if b > a:
                        _put(rec[a:b], "rec")
                last_j = max(last_j, max(js))
            if bad[i]:
                i2 = i
                while i2 < n_f and bad[i2]:
                    i2 += 1
                js_bad = sorted(j for t in range(i, i2) for j in j_of_i.get(t, []))
                y_span = i2 - i
                r_span = (js_bad[-1] - js_bad[0] + 1) if js_bad else 0
                if js_bad and r_span <= max(1, y_span // 2):
                    pass                        # 插入段：丢弃（DTW 压缩映射证据）
                elif js_bad:
                    a, b = js_bad[0] * hop, min(len(rec), (js_bad[-1] + 1) * hop)
                    if b > a:
                        _put(rec[a:b], "rec")   # 替换段：R 对应内容
                i = i2
            else:
                a, b = i * hop, min(len(y), (i + 1) * hop)
                if b > a:
                    _put(y[a:b], "y")           # 好区：原样保留
                i += 1
        # 句尾删除：R 帧剩余部分补回
        if last_j >= 0 and (last_j + 1) * hop < len(rec):
            _put(rec[(last_j + 1) * hop:], "rec")
        # 顺序拼接：源切换（y↔rec）边界做 12ms cos 交叉淡化，消除拼接爆音
        cf = min(int(0.012 * au.SR), hop)      # ≤1 帧，避免好区相邻帧失真
        prev_src = None
        for seg, src in segs:
            if prev_src is not None and src != prev_src and w >= cf and len(seg) >= cf:
                g = 0.5 * (1.0 - np.cos(np.pi * np.arange(cf) / cf))
                out[w - cf:w] = out[w - cf:w] * (1.0 - g) + seg[:cf] * g
                seg = seg[cf:]
            out[w:w + len(seg)] = seg
            w += len(seg)
            prev_src = src
        return out[:w].copy()

    def _recover_v1(self, y, rec, fc):
        """v1 检出区域同索引替换（direct 场景正确；保留作回退口径）。"""
        thr = float(np.quantile(fc, 0.99))
        out = y.copy()
        L = min(len(out), len(rec))
        bad = np.zeros(L, bool)
        hop = int(self.hop_ms / 1000 * au.SR)
        for i, c in enumerate(fc):
            if c >= thr:
                a, b = i * hop, min((i + 1) * hop, L)
                bad[a:b] = True
        if not bad.any():
            return out
        idx = np.flatnonzero(bad)
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for seg in splits:
            out[seg[0]:min(seg[-1] + 1, L)] = rec[seg[0]:min(seg[-1] + 1, L)]
        return out
