"""A038 算法适配层：平台请求 → A038 纯算法模块（与复现报告同源，零改动移植）。

任务：
- embed：x → x̃（float32 WAV，LSB 水印）+ 嵌入报告
- verify：y → 检测判决（P7 改进口径：CRC 完整性四态 + LSB 标尺接缝定位 + 内容恢复）

判决规则（decision）：
- AUTHENTIC           integrity=verified 且无接缝 → 完好水印信号
- TAMPERED            integrity∈{partial,corrupted} 或检出接缝 → 词级局部篡改
- UNMARKED_OR_CHANNEL integrity=absent → 无水印或信道退化（论文口径下不可区分）
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from src.config.base import Config            # noqa: E402
from src.common import audio_utils as au      # noqa: E402
from src.common.codec_io import build_codec   # noqa: E402
from src.voiceProducer.publisher import Publisher       # noqa: E402
from src.Receiver.verifier import Verifier              # noqa: E402

import os
CODEC_NAME = os.environ.get("A038_CODEC", "mock")
MIN_SEC, MAX_SEC = 3.0, float(os.environ.get("A038_MAX_SECONDS", "30"))


def _cfg(params: dict) -> Config:
    cfg = Config()
    cfg.codec = CODEC_NAME
    cfg.lsb_k = int(params.get("lsb_k", 1))
    cfg.payload_crc = bool(params.get("payload_crc", True))    # P7 完整性通道
    cfg.lsb_fusion = bool(params.get("lsb_fusion", True))      # P7 LSB 标尺
    cfg.save_figs = False
    return cfg


_CODEC = None


def _codec(cfg):
    global _CODEC
    if _CODEC is None or getattr(_CODEC, "name", "") != cfg.codec:
        _CODEC = build_codec(cfg)
    return _CODEC


def _load_audio(data: bytes, filename: str) -> np.ndarray:
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:
        raise ValueError(f"无法解析音频（支持 WAV；MP3 等请先转 WAV）：{e}")
    x = x.mean(axis=1)  # 立体声 → 单声道
    if sr != au.SR:     # 重采样至 16 kHz（论文口径）
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), au.SR)
        x = resample_poly(x, au.SR // g, sr // g).astype(np.float32)
    if len(x) < MIN_SEC * au.SR:
        raise ValueError(f"音频过短（< {MIN_SEC:.0f}s，水印容量不足）")
    if len(x) > MAX_SEC * au.SR:
        raise ValueError(f"音频过长（> {MAX_SEC:.0f}s，请分段上传）")
    if not np.isfinite(x).all() or float(np.abs(x).max()) > 1.0:
        x = np.clip(np.nan_to_num(x), -1.0, 1.0)
    return np.ascontiguousarray(x, np.float32)


def _save_wav(path: Path, x: np.ndarray):
    # float32 WAV：逐位保留 LSB 水印（PCM16/MP3 重量化会摧毁水印——论文理想信道口径）
    sf.write(str(path), x, au.SR, subtype="FLOAT")


def _pesq(ref: np.ndarray, deg: np.ndarray):
    try:
        from pesq import pesq as _p
        return round(float(_p(au.SR, ref, deg, "wb")), 3)
    except Exception:
        return None


def process(data: bytes, filename: str, params: dict, out: Path) -> dict:
    task = str(params.get("task", "embed"))
    if task not in ("embed", "verify"):
        raise ValueError("task 必须为 embed 或 verify")
    x = _load_audio(data, filename)
    cfg = _cfg(params)
    codec = _codec(cfg)
    t0 = time.time()

    if task == "embed":
        pub = Publisher(cfg, codec)
        pr = pub.publish("upload", x)
        _save_wav(out / "watermarked.wav", pr.xw)
        report = {
            "task": "embed", "duration_s": round(len(x) / au.SR, 2),
            "payload_bits": pr.n_payload_bits, "repetitions": pr.repetitions,
            "lsb_k": cfg.lsb_k, "codec": CODEC_NAME,
            "codec_bitrate_bps": {"semanticodec": 650, "mock": 2000}.get(CODEC_NAME, None),
            "pesq_x_vs_watermarked": _pesq(x, pr.xw),
            "note": "水印信号为 float32 WAV；任何 PCM16/MP3 等重量化将摧毁 LSB 水印（论文口径为理想信道）。",
            "elapsed_s": round(time.time() - t0, 1),
        }
        (out / "embed_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "message": f"水印嵌入完成（R={pr.repetitions} 份重复，PESQ={report['pesq_x_vs_watermarked']}）",
            "artifacts": [
                {"kind": "audio", "label": "水印信号 x̃（float32 WAV）", "filename": "watermarked.wav"},
                {"kind": "json", "label": "嵌入报告", "filename": "embed_report.json"},
            ],
        }

    # ---- verify ----
    ver = Verifier(cfg, codec)
    att = SimpleNamespace(y=x, attack="upload", label=1, gt_segments=[])
    vr = ver.verify("upload", att)
    # 判决规则：integrity n/a = legacy 布局（无 CRC）但载荷恢复（mode=sync）
    # —— 按"无报警证据 + 无接缝"判 AUTHENTIC（与 verified 同级处理）。
    verified_like = vr.integrity in ("verified", "n/a") and vr.payload_mode == "sync"
    decision = (
        "TAMPERED" if (vr.integrity in ("partial", "corrupted") or
                        (vr.a_min >= 0 and vr.a_min < 0.75))
        else "AUTHENTIC" if verified_like
        else "UNMARKED_OR_CHANNEL")
    report = {
        "task": "verify", "decision": decision, "duration_s": round(len(x) / au.SR, 2),
        "score": round(vr.score, 4),
        "score_raw_dtw": round(vr.score_raw, 4) if vr.score_raw >= 0 else None,
        "watermark_integrity": vr.integrity,
        "integrity_note": {
            "verified": "水印在场且全部重复份 CRC 校验通过",
            "partial": "水印在场但有重复份损坏（篡改证据）",
            "corrupted": "同步字在场但载荷校验失败（篡改证据）",
            "absent": "水印不辩：音频未经水印或信道操作已摧毁水印（论文口径下不可区分）",
        }.get(vr.integrity, "—"),
        "corrupted_replicas": vr.n_corrupt,
        "payload_agreement": round(vr.payload_agreement, 3),
        "lsb_ruler": {
            "a_min": round(vr.a_min, 4) if vr.a_min >= 0 else None,
            "tamper_junction_s": round(vr.junction_s, 4) if vr.junction_s >= 0 else None,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    artifacts = [{"kind": "json", "label": "检测报告", "filename": "verify_report.json"}]
    (out / "verify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 帧代价曲线 + 接缝标注
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ft, fc = vr.frame_times, vr.frame_costs
        ax.plot(ft, fc, lw=1.2, color="#33518c")
        if vr.junction_s >= 0:
            ax.axvline(vr.junction_s, color="#d13438", ls="--", lw=1.5,
                       label=f"tamper junction = {vr.junction_s:.3f}s")
        ax.set_xlabel("time (s)"); ax.set_ylabel("DTW frame cost")
        ax.set_title(f"decision: {decision} | integrity: {vr.integrity}")
        if vr.junction_s >= 0:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "frame_costs.png", dpi=140)
        plt.close(fig)
        artifacts.append({"kind": "image", "label": "帧级代价曲线与篡改接缝",
                          "filename": "frame_costs.png"})
    except Exception:
        pass

    # 内容恢复（篡改时用自重建替换高代价区域，论文 §III）
    if vr.recovered is not None and decision == "TAMPERED":
        _save_wav(out / "recovered.wav", vr.recovered)
        artifacts.append({"kind": "audio", "label": "恢复输出（自重建替换检出区域）",
                          "filename": "recovered.wav"})

    msg = {
        "TAMPERED": f"检出局部篡改（integrity={vr.integrity}"
                    + (f"，接缝 {vr.junction_s:.3f}s" if vr.junction_s >= 0 else "") + "）",
        "AUTHENTIC": "未检出篡改（水印完好）",
        "UNMARKED_OR_CHANNEL": "无水印或水印已被信道操作摧毁（不可区分，论文理想信道口径）",
    }[decision]
    return {"message": msg, "artifacts": artifacts}
