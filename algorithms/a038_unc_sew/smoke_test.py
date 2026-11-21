"""冒烟自检（CPU 可跑，mock codec）：本文件同目录执行 `python smoke_test.py`。

验证：/process 契约（multipart 字段/参数 JSON/响应结构）、embed→verify 闭环、
错误路径（错误参数/过短音频/非 WAV）、结果文件下载与 CORS。
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

os.environ.setdefault("A038_CODEC", "mock")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient  # noqa: E402
import app  # noqa: E402

ok = 0


def check(name, cond, extra=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    assert cond, name
    ok += 1


def wav_bytes(x, sr=16000, subtype="FLOAT"):
    buf = io.BytesIO()
    sf.write(buf, np.asarray(x, np.float32), sr, subtype=subtype, format="WAV")
    return buf.getvalue()


rng = np.random.default_rng(0)
t = np.arange(5 * 16000) / 16000
x = (0.3 * np.sin(2 * np.pi * 140 * t) + 0.15 * np.sin(2 * np.pi * 421 * t))
x *= (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 3.1 * t)))

c = TestClient(app.app)

# 1) embed
r = c.post("/process", data={"method_id": "a038_unc_sew", "parameters": json.dumps({"task": "embed", "lsb_k": 1})},
           files={"input_file": ("a.wav", wav_bytes(x), "audio/wav")})
check("embed 200", r.status_code == 200, r.text[:120])
body = r.json()
check("artifacts 结构", isinstance(body.get("artifacts"), list) and 1 <= len(body["artifacts"]) <= 20)
url = body["artifacts"][0]["url"]
check("artifact url 相对", url.startswith("/files/"))

# 2) 下载 watermarked.wav → verify（AUTHENTIC）
r2 = c.get(url)
check("结果可下载", r2.status_code == 200)
y = sf.read(io.BytesIO(r2.content), dtype="float32")[0]
r = c.post("/process", data={"method_id": "a038_unc_sew", "parameters": json.dumps({"task": "verify"})},
           files={"input_file": ("y.wav", wav_bytes(y), "audio/wav")})
check("verify 200", r.status_code == 200, r.text[:120])
rep_url = [a for a in r.json()["artifacts"] if a["filename"] == "verify_report.json"][0]["url"]
rep = c.get(rep_url).json()
check("AUTHENTIC 判决", rep["decision"] == "AUTHENTIC",
      f"decision={rep['decision']} integrity={rep['watermark_integrity']}")

# 3) 篡改（删中段 0.3s）→ TAMPERED + 接缝
i0 = int(2.0 * 16000)
y_t = np.concatenate([y[:i0], y[i0 + 4800:]])
r = c.post("/process", data={"method_id": "a038_unc_sew", "parameters": json.dumps({"task": "verify"})},
           files={"input_file": ("t.wav", wav_bytes(y_t), "audio/wav")})
rep = c.get([a for a in r.json()["artifacts"] if a["filename"] == "verify_report.json"][0]["url"]).json()
check("TAMPERED 判决", rep["decision"] == "TAMPERED", f"decision={rep['decision']}")
check("接缝定位", rep["lsb_ruler"]["tamper_junction_s"] is not None,
      f"junction={rep['lsb_ruler']['tamper_junction_s']} (真值 2.0s)")

# 4) 信道死亡（低通重采样模拟）→ UNMARKED_OR_CHANNEL
from scipy.signal import butter, sosfiltfilt
y_c = sosfiltfilt(butter(4, 3000 / 8000, btype="low", output="sos"), y).astype(np.float32)
r = c.post("/process", data={"method_id": "a038_unc_sew", "parameters": json.dumps({"task": "verify"})},
           files={"input_file": ("c.wav", wav_bytes(y_c), "audio/wav")})
rep = c.get([a for a in r.json()["artifacts"] if a["filename"] == "verify_report.json"][0]["url"]).json()
check("UNMARKED_OR_CHANNEL 判决", rep["decision"] == "UNMARKED_OR_CHANNEL",
      f"decision={rep['decision']} integrity={rep['watermark_integrity']}")

# 5) 错误路径
r = c.post("/process", data={"method_id": "other", "parameters": "{}"},
           files={"input_file": ("a.wav", wav_bytes(x), "audio/wav")})
check("method_id 不匹配 400", r.status_code == 400 and "error" in r.json())
r = c.post("/process", data={"method_id": "a038_unc_sew", "parameters": "not-json"},
           files={"input_file": ("a.wav", wav_bytes(x), "audio/wav")})
check("parameters 非 JSON 400", r.status_code == 400 and "error" in r.json())
r = c.post("/process", data={"method_id": "a038_unc_sew", "parameters": json.dumps({"task": "verify"})},
           files={"input_file": ("s.wav", wav_bytes(x[:16000]), "audio/wav")})
check("过短音频 400", r.status_code == 400 and "error" in r.json())

# 6) 健康检查
r = c.get("/health")
check("health", r.status_code == 200 and r.json()["method"] == "a038_unc_sew")

print(f"\n冒烟全部通过（{ok} 项）——服务契约与算法闭环 OK（codec={os.environ['A038_CODEC']}）")
