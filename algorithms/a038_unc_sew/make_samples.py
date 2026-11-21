"""生成前端展示样例包：1 句用例 ×（原声 embed 演示 + 11 种攻击）= 12 条，贴满平台契约上限。

平台约束（docs/算法接入规范.md + contract.js，前端源代码零改动）：
    examples ≤ 12 条；样例总量 ≤ 20MB；src 相对站点根（samples/方法ID/…）。
默认 12 条：real（embed 演示：input=原声 → output=水印信号 x̃，听感透明度对比）
+ 11 攻击 verify 样例（论文三类 insert/delete/direct + 信道 8 类 noise20/noise10/
lowpass/pitch2/stretch105/mp3/ogg/gauss——去掉判读雷同的 highpass/resample）。
选句由 --seed 控制：换 seed = 换句子（同 seed 确定性复现）。

产物（--out 目录，默认指向前端仓库 public/samples/a038_unc_sew/）：
    real/<uid>.wav|.png         原始干净语音（dataset/wav_val）+ 波形图
    real/<uid>.wav|.png          原声（embed 样例输入；波形图）
    watermarked/<uid>.wav       水印信号 x̃（res 落盘件拷贝；embed 样例输出）
    insert|delete|direct/<uid>.wav|.png  论文三类词级操纵（FLOAT32——判决依赖水印在场）
    noise20|noise10|lowpass|highpass|pitch2|stretch105|mp3|ogg|resample|gauss/
                                <uid>.wav|.png  信道扰动（PCM_16——判决本为死亡报警，
                                重量化不改变其展示语义）
    texts.json                  每句转写（word_gt.json 词序列）
    examples_gen.json           examples 片段（title=[类] 前缀；description 含原句转写
                                + 攻击参数 + verify 判决与观察点）——供合并进方法 JSON

攻击重放与 runner 完全同源：rng = default_rng(seed + crc32(uid|attack) % 100000)，
fixed donor 池（同旧行为）；水印信号取 res/<exp>/audio/<uid>_watermarked.wav 落盘件
（不重跑 embed，样例与实验数据同源）。

用法（A038_P3 根目录；真样例须 GPU 机器 A038_CODEC=semanticodec）：
    python make_examples.py --res-dir res/p7_improve \
        --out ../algorithm_research_frontend_A038/public/samples/a038_unc_sew
    python make_examples.py --res-dir res/p7_improve --uids idxxx,idyyy   # 显式指定
    python make_examples.py ... --update-json ../algorithm_research_frontend_A038/public/methods/a038_unc_sew.json
    python make_examples.py ... --channel-subset mp3,noise10              # 信道类减配
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

from types import SimpleNamespace

import numpy as np
import soundfile as sf

from src.config.base import load_config
from src.common import audio_utils as au
from src.common.codec_io import build_codec
from src.Receiver.verifier import Verifier
from src.voiceProducer.attacker import build_attacker
from src.voiceProducer.channel_attacks import CHANNEL_NAMES
from src.voiceProducer.dataloader import DataLoader, Utterance

# 平台契约（docs/算法接入规范.md + contract.js，前端源代码零改动）：
# examples ≤ 12 条、样例总量 ≤ 20MB。默认 12 条 = real（embed 演示）+ 11 攻击。
PAPER_ATTACKS = ["insert", "delete", "direct"]
DEFAULT_CHANNEL = ["noise20", "noise10", "lowpass", "pitch2", "stretch105",
                   "mp3", "ogg", "gauss"]        # 8/10 信道类（契约内代表集）
FLOAT_KINDS = PAPER_ATTACKS + ["watermarked"]   # FLOAT32 保水印（判决依赖 LSB 在场）

_CN = {
    "real": "原声", "watermarked": "水印信号",
    "insert": "词级插入", "delete": "词级删除", "direct": "词级替换",
    "noise20": "加噪 20dB", "noise10": "加噪 10dB", "lowpass": "低通 3.4kHz",
    "highpass": "高通 300Hz", "pitch2": "变调 +2半音", "stretch105": "时长 ×1.05",
    "mp3": "MP3 128kbps", "ogg": "OGG 64kbps", "resample": "重采样 16k→8k→16k",
    "gauss": "高斯平滑 2ms",
}
# 方法页样例标签（filter_tabs 按钮文字）短名——title 是横向 flex 无 wrap 容器，
# 长标题（如"词级插入 → UNMARKED_OR_CHANNEL（6.2s）"≈325px ×12 条 ≈3400px）会撑破
# 1320px 主容器造成整页横向溢出；短名 12 条总宽 ~1.0kpx 单行放下。判决/时长等
# 详情在 description（卡片区，宽度充裕）展示，信息不丢。前端源代码零改动约束下
# 这是数据侧唯一根治位（2026-11-21 UI 溢出问题定案）。
_SHORT_TAB = {
    "real": "原声 embed",
    "insert": "插入", "delete": "删除", "direct": "替换",
    "noise20": "加噪20dB", "noise10": "加噪10dB", "lowpass": "低通",
    "highpass": "高通", "pitch2": "变调", "stretch105": "变速",
    "mp3": "MP3", "ogg": "OGG", "resample": "重采样", "gauss": "平滑", "tts": "TTS",
}
_OBS = {
    "real": "原始录音（未嵌水印）：verify 判 UNMARKED_OR_CHANNEL（fail-safe 报警侧——与水印死亡不可区分）。也可作为 embed 任务的输入素材。",
    "watermarked": "未篡改水印信号：verify 判 AUTHENTIC、CRC 全 verified、帧代价曲线无接缝。",
    "insert": "观察判决 TAMPERED、LSB 一致率在插入段塌陷、红色虚线标出插入接缝（真实位置红带）。",
    "delete": "观察判决 TAMPERED、帧级代价在删除段持续抬升、接缝定位亚帧精度（红带=真实删除区间）。",
    "direct": "同说话人词级替换：判决 TAMPERED，替换段一致率塌陷（红带=真实替换区间）。",
    "noise20": "信道扰动：水印死亡 → integrity=absent，判 UNMARKED_OR_CHANNEL（水印不存在或失效）。噪声较弱时 DTW 分数本身几乎不变——靠 CRC 通道报警（本组改进）。",
    "noise10": "信道扰动：加噪 10dB 摧毁 LSB 载荷 → absent 报警。",
    "lowpass": "信道扰动：3.4kHz 低通摧毁载荷 → absent 报警（分数反降，DTW 无判别力——靠 CRC）。",
    "highpass": "信道扰动：300Hz 高通摧毁载荷 → absent 报警（论文静默失效场景，CRC 兜底）。",
    "pitch2": "信道扰动：+2 半音变调摧毁载荷 → absent 报警。",
    "stretch105": "信道扰动：时长 ×1.05——单同步退化：integrity=corrupted（至少 R-1 份被毁，2026-10-31 口径）→ TAMPERED 证据级报警。",
    "mp3": "信道扰动：MP3 128kbps 无声移除水印（论文未测）→ absent 报警。最温和的现实信道即可击穿 LSB 层。",
    "ogg": "信道扰动：OGG 64kbps → absent 报警。",
    "resample": "信道扰动：16k→8k→16k 重采样 → absent/corrupted 报警。",
    "gauss": "信道扰动：2ms 高斯滑窗平滑 → absent 报警（分数反向，DTW 失效——CRC 兜底）。",
}


def _decision(vr) -> str:
    """复刻 service.py 判决三态（与线上服务行为一致）。"""
    verified_like = vr.integrity in ("verified", "n/a") and vr.payload_mode == "sync"
    if vr.integrity in ("partial", "corrupted") or (vr.a_min >= 0 and vr.a_min < 0.75):
        return "TAMPERED"
    if verified_like:
        return "AUTHENTIC"
    return "UNMARKED_OR_CHANNEL"


def _write_wav(path: Path, y: np.ndarray, kind: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    # FLOAT 类保水印（判决依赖 LSB）；信道类判决本为死亡报警，PCM_16 省体积
    subtype = "FLOAT" if kind in FLOAT_KINDS else "PCM_16"
    sf.write(path, y.astype(np.float32), au.SR, subtype=subtype)


def _mpl_cjk():
    """CJK 字体链：沙箱 WenQuanYi → Linux Noto CJK → Windows 雅黑/黑体 → 放弃（方框但不崩）。"""
    import matplotlib
    import matplotlib.font_manager as fm
    for name in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Serif SC",
                 "Microsoft YaHei", "SimHei"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    import warnings
    warnings.warn("未找到 CJK 字体——图中中文将显示为方框（安装 fonts-wqy-zenhei 可解）")


def _combo_png(path: Path, y: np.ndarray, vr, gt_segments, title: str, decision: str):
    """组合图：上=波形（红带=真实篡改区间）+ 接缝线；下=帧级代价曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    _mpl_cjk()
    import matplotlib.pyplot as plt

    t = np.arange(len(y)) / au.SR
    fc = vr.frame_costs
    ft = vr.frame_times
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), dpi=100,
                                   sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    ax1.plot(t, y, lw=0.3, color="#2b6cb0")
    for a, b in (gt_segments or []):
        ax1.axvspan(a, b, color="red", alpha=0.25)
    if vr.junction_s is not None and vr.junction_s >= 0:
        ax1.axvline(vr.junction_s, color="red", ls="--", lw=1.2, label=f"接缝 {vr.junction_s:.2f}s")
        ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylabel("幅度"); ax1.set_ylim(-1, 1)
    ax1.set_title(title, fontsize=10)
    if len(fc):
        ax2.plot(ft, fc, lw=0.8, color="#2f855a", label="帧级 DTW 代价")
        if len(gt_segments or []):
            a, b = gt_segments[0]
            ax2.axvspan(a, b, color="red", alpha=0.18)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_ylabel("代价"); ax2.set_xlabel("秒")
    fig.suptitle(f"decision: {decision} | integrity: {vr.integrity}", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def _wave_png(path: Path, y: np.ndarray, title: str, note: str):
    """纯波形图（real 类用）。"""
    import matplotlib
    matplotlib.use("Agg")
    _mpl_cjk()
    import matplotlib.pyplot as plt
    t = np.arange(len(y)) / au.SR
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=100)
    ax.plot(t, y, lw=0.3, color="#2b6cb0")
    ax.set_ylabel("幅度"); ax.set_xlabel("秒"); ax.set_ylim(-1, 1)
    ax.set_title(title, fontsize=10)
    fig.text(0.02, 0.02, note, fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", required=True, help="res/<exp>（含 audio/<uid>_watermarked.wav 与 rows.jsonl）")
    ap.add_argument("--out", default="../algorithm_research_frontend_A038/public/samples/a038_unc_sew",
                    help="样例输出目录（前端 public/samples/a038_unc_sew）")
    ap.add_argument("--dataset", default="dataset", help="A038 dataset 根（含 wav_val/ word_gt.json）")
    ap.add_argument("--codec", default="semanticodec", help="semanticodec | mock（沙箱自测用）")
    ap.add_argument("--n", type=int, default=1, help="选句数（契约上限 12 条 = 1 句 × 12 类）")
    ap.add_argument("--uids", default=None, help="逗号分隔，显式指定（覆盖 --n）")
    ap.add_argument("--channel-subset", default=None, help="逗号分隔的信道类子集（默认 8 类代表集）")
    ap.add_argument("--seed", type=int, default=2026, help="选句随机种子：换 seed = 换句子（同 seed 确定性）")
    ap.add_argument("--update-json", default=None, help="方法 JSON 路径：生成后把 examples 合并进去（自动备份 .bak）")
    ap.add_argument("--dur-min", type=float, default=3.5, help="选句时长下限")
    ap.add_argument("--dur-max", type=float, default=9.0, help="选句时长上限")
    args = ap.parse_args()

    res = Path(args.res_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 与 p7_improve preset 同参（val + semanticodec + P7 布局）——不 import runner，显式写死
    cfg = load_config(None, dict(data_mode="val", codec=args.codec, lsb_k=1,
                                 payload_crc=True, lsb_fusion=True,
                                 data_root=args.dataset, seed=2026, exp_name="make_examples"))
    dl = DataLoader(cfg)
    word_gt = dl._word_gt
    wavs = {w.stem: w for w in dl.wav_dir.glob("*.wav")}

    # ---- 选句：rows.jsonl 覆盖 + watermarked 在场 + 时长 + 有转写；--seed 换句 ----
    rows = [json.loads(l) for l in (res / "rows.jsonl").open(encoding="utf-8")]
    covered = sorted({r["uid"] for r in rows})
    if args.uids:
        uids = [u.strip() for u in args.uids.split(",") if u.strip()]
    else:
        cands = []
        for uid in covered:
            wm = res / "audio" / f"{uid}_watermarked.wav"
            if not wm.exists() or uid not in wavs or uid not in word_gt:
                continue
            d = len(au.load_wav(str(wavs[uid]))) / au.SR
            if args.dur_min <= d <= args.dur_max:
                cands.append((d, uid))
        pool = [u for _, u in cands]
        # seed 控制换句：同一 seed 确定性选句，换 seed = 换句子
        rng_pick = np.random.default_rng(args.seed)
        if len(pool) > args.n:
            idx = rng_pick.choice(len(pool), args.n, replace=False)
            pool = [pool[int(i)] for i in sorted(idx)]
        uids = pool[:args.n]
    if len(uids) < (args.uids and len(args.uids) or args.n):
        print(f"[examples] 警告：实际选出 {len(uids)} 句（候选不足——放宽 --dur-min/max 或显式 --uids）")
    print(f"[examples] 选中 {len(uids)} 句：{uids}")

    channel = DEFAULT_CHANNEL if not args.channel_subset else \
        [c.strip() for c in args.channel_subset.split(",") if c.strip() in CHANNEL_NAMES]
    # 契约上限 12：1 句 × (real + 3 论文 + 8 信道) = 12 贴满
    attacks = PAPER_ATTACKS + channel
    all_kinds = ["real"] + attacks
    if len(uids) * len(all_kinds) > 12:
        print(f"[examples] 警告：{len(uids)} 句 × {len(all_kinds)} 类 = "
              f"{len(uids) * len(all_kinds)} 条，超平台契约上限 12 —— 前端将拒载！")

    # ---- 攻击重放依赖（与 runner 同源：fixed donor 池 + per-(uid,attack) rng）----
    codec = build_codec(cfg)
    ver = Verifier(cfg, codec)
    utts = [Utterance(uid=uid, path=str(wavs[uid]),
                      duration=len(au.load_wav(str(wavs[uid]))) / au.SR,
                      word_bounds=[(w["s"], w["e"]) for w in (word_gt.get(uid) or [])])
            for uid in uids]
    donor_pool = dl.donors(n=min(10, max(1, len(covered))), seed=cfg.seed + 1)

    texts, examples, total_bytes = {}, [], 0
    for u in utts:
        x = u.load()
        xw = au.load_wav(str(res / "audio" / f"{u.uid}_watermarked.wav"))
        words = word_gt.get(u.uid) or []
        text = " ".join(w["w"] for w in words)
        texts[u.uid] = {"text": text, "dur_s": round(len(x) / au.SR, 2),
                        "words": [(w["w"], w["s"], w["e"]) for w in words]}
        dur = len(x) / au.SR
        print(f"\n[examples] === {u.uid}（{dur:.1f}s）原句：{text[:60]}{'…' if len(text) > 60 else ''}")

        # real：embed 样例（input=原声 → output=水印信号 x̃，听感透明度对比）
        _write_wav(out / "real" / f"{u.uid}.wav", x, "real")
        _write_wav(out / "watermarked" / f"{u.uid}.wav", xw, "watermarked")
        _wave_png(out / "real" / f"{u.uid}.png", x,
                  f"real · {u.uid}（{dur:.1f}s）", f"原句：{text}")
        examples.append(dict(
            title=_SHORT_TAB["real"],
            description=f"embed 任务演示：原始干净语音（AV-Deepfake1M val）→ 水印信号 x̃（PESQ≈4.6，听感无差异）。原句：{text}。对比输入输出音频验证透明度；对 x̃ 再跑 verify 会得到 AUTHENTIC/verified。{_OBS['real']}",
            input={"src": f"samples/a038_unc_sew/real/{u.uid}.wav", "kind": "audio", "label": "原声（未水印）"},
            output={"src": f"samples/a038_unc_sew/watermarked/{u.uid}.wav", "kind": "audio", "label": "水印信号 x̃"},
            parameters={"task": "embed", "lsb_k": 1, "payload_crc": True, "lsb_fusion": True}))
        print(f"[examples]   real(embed) → {out/'real'/f'{u.uid}.wav'} + x̃")

        # 攻击重放（论文三类 + 信道）
        for atk in attacks:
            rng = np.random.default_rng(cfg.seed + zlib.crc32(f"{u.uid}|{atk}".encode()) % 100000)
            atker = build_attacker(atk, cfg)
            sig = atker.attack(xw, u, donor_pool, rng)
            vr = ver.verify(u.uid, sig)
            dec = _decision(vr)
            _write_wav(out / atk / f"{u.uid}.wav", sig.y, atk)
            _combo_png(out / atk / f"{u.uid}.png", sig.y, vr, sig.gt_segments, f"{atk} · {u.uid}", dec)
            j = vr.junction_s if (vr.junction_s is not None and vr.junction_s >= 0) else None
            gt = ", ".join(f"[{a:.2f},{b:.2f}]s" for a, b in (sig.gt_segments or [])) or "—"
            extra = f" 篡改区间（真值）：{gt}。" if sig.gt_segments else ""
            if j is not None:
                extra += f" 接缝定位：{j:.3f}s。"
            examples.append(dict(
                title=_SHORT_TAB[atk],
                description=f"{_CN[atk]}攻击。原句：{text}。判决：{dec}（integrity={vr.integrity}）。{extra}{_OBS[atk]}",
                input={"src": f"samples/a038_unc_sew/{atk}/{u.uid}.wav", "kind": "audio", "label": f"{_CN[atk]}后音频"},
                output={"src": f"samples/a038_unc_sew/{atk}/{u.uid}.png", "kind": "image", "label": "波形+帧级代价"},
                parameters={"task": "verify", "lsb_k": 1, "payload_crc": True, "lsb_fusion": True}))
            print(f"[examples]   {atk:<11} → {dec}/{vr.integrity}"
                  + (f" junction={j:.2f}s" if j is not None else ""))

    # ---- texts.json + examples_gen.json ----
    (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "examples_gen.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    # 体积盘点
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"\n[examples] done → {out}")
    print(f"[examples] 样例 {len(examples)} 条 / {len(all_kinds)} 类 × {len(uids)} 句；总体积 {total:.1f} MB")
    if total > 20:
        print("[examples] 警告：总体积 >20MB——建议 --channel-subset 减配信道类（PCM_16 已省 4×）")

    # ---- 可选：合并进方法 JSON ----
    if args.update_json:
        mj = Path(args.update_json)
        bak = mj.with_suffix(".json.bak")
        if not bak.exists():
            bak.write_text(mj.read_text(encoding="utf-8"), encoding="utf-8")
        doc = json.loads(mj.read_text(encoding="utf-8"))
        doc["examples"] = examples
        mj.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[examples] 方法 JSON 已更新（{len(examples)} 条 examples；原版备份 {bak.name}）")


if __name__ == "__main__":
    main()
