"""实验配置（论文口径默认值 + YAML 覆盖）。

关键默认对齐论文：
- codec=semanticodec (token_rate=50, vocab=8192 → 13bit/token，ρ=650bps)；
- lsb_k=1（P3 冒烟用 1，正式矩阵 k∈{1,2,4}）；
- 40-band log-mel / 16ms 帧移 / top-1% 得分 / 平滑窗 31 帧（窗长未披露，可调）；
- sync 容错 8/64 bit。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # ---- data ----
    data_root: str = "dataset"
    data_mode: str = "dev"            # dev | val
    n_utts: int = 5                   # 冒烟规模
    seed: int = 2026

    # ---- codec / payload ----
    codec: str = "semanticodec"       # semanticodec | snac | mock
    payload_mode: str = "codec"       # codec | hash（P6 对照）
    lsb_k: int = 1                    # 1/2/4 bits per sample
    sync_err_max: int = 8             # 64bit 同步字容错位数

    # semanticodec
    sc_token_rate: int = 50
    sc_vocab: int = 8192
    sc_ddim_steps: int = 50
    sc_cfg_scale: float = 2.0

    # ---- DTW / 评分 ----
    n_mels: int = 40
    hop_ms: float = 16.0
    n_fft: int = 512
    smooth_frames: int = 31
    top_pct: float = 0.01
    score_norm: str = "median"        # raw | median | mean（v2：逐句基线归一，200 条实测 EER 0.25→0.19）

    # ---- attack ----
    attacks: list = field(default_factory=lambda: ["none", "direct", "delete", "insert"])

    # ---- P7 检测改进（默认关 = P4/P5 legacy 行为，字节级不变）----
    payload_crc: bool = False        # per-replica CRC16 完整性通道（integrity 三态）
    lsb_fusion: bool = False         # LSB 标尺通道（帧级一致率 + score 融合 + 接缝定位）
    lsb_boost_w: float = 1.0         # LSB 通道得分增益权重（×(1-a_min)）

    # donor / direct 口径（P5-A1 论文对齐：同说话人词级段+电平归一+multi-word）
    donor_mode: str = "fixed"          # fixed（旧行为：固定 10 条池） | same_speaker
    direct_max_words: int = 2          # direct 替换词数上限（论文 "one or more word segments"）
    donor_cap: int = 8                 # 同说话人 donor 每句采样条数上限
    delete_word_min_dur: float = 0.0   # P5-A1b：delete 选词时长下限（秒，0=随机均匀=旧行为；0.3≈论文可比口径）

    # ---- output ----
    out_dir: str = "res"
    exp_name: str = "p3_smoke"
    save_figs: bool = True
    save_audio: bool = True


def load_config(yaml_path: str | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    if yaml_path:
        import yaml
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
        _apply(cfg, data)
    if overrides:
        _apply(cfg, overrides)
    return cfg


def _apply(cfg: Config, data: dict):
    valid = {f.name for f in fields(cfg)}
    for k, v in data.items():
        if k in valid:
            setattr(cfg, k, v)
        else:
            raise KeyError(f"未知配置项: {k}（可用: {sorted(valid)}）")
