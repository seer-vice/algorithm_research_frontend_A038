# UNC-SEW 算法服务（A038 自包含实现）

A038 复现工程（论文 arXiv 2608.25289，复现报告见仓库根 README）的服务器化：
FastAPI 薄适配层 + `src/` 内 12 个纯算法模块（与复现报告**同源零改动**，
P7 改进默认开启：CRC 完整性通道 + LSB 标尺定位）。

## 目录

```
app.py            平台契约层（POST /process、/files 静态结果、CORS、TTL 清理、GPU 串行锁）
service.py        算法适配（embed/verify、判决规则、报告/曲线/恢复输出）
smoke_test.py     CPU 冒烟自检（13 项，mock codec，无需 GPU）
src/              A038 纯算法模块（publisher/verifier/extractor/payload/lsb/dtw/codec_io...）
requirements.txt
```

## 环境与启动（三步，幂等）

```bash
# 1) venv + 依赖（CPU 冒烟级）
python -m venv .venv && source .venv/bin/activate   # win: .venv\Scripts\activate
pip install -r requirements.txt

# 2) 冒烟自检（mock codec，无 GPU 也能跑）
python smoke_test.py           # 预期 13 项全 PASS

# 3) 真 codec（GPU 机器）+ 启动
pip install torch --index-url https://download.pytorch.org/whl/cu128
# SemantiCodec 推理包（私有部署，详见组内文档）：
pip install -e <SemantiCodec-inference 路径>
export A038_CODEC=semanticodec
# 权重目录用绝对路径指过去（cache_path 是 CWD 相对坑，不设会重下 3GB）：
export A038_PRETRAINED=<SemantiCodec 权重目录绝对路径>
uvicorn app:app --host 0.0.0.0 --port 8710
# 免 GPU 备选：SNAC（pip install snac，CPU 3-5s/句，ρ≈1kbps）：
pip install snac && export A038_CODEC=snac
```

环境变量：`A038_CODEC`（mock|snac|semanticodec）、`A038_MAX_SECONDS`（默认 30）、
`A038_TTL_H`（结果保留小时，默认 1）、`A038_CONCURRENCY`（GPU 并发，默认 1）、
`A038_PRETRAINED`（SemantiCodec 权重目录绝对路径，免重下）。Windows 写法：
`$env:A038_CODEC="semanticodec"; $env:A038_PRETRAINED="D:\...\pretrained"`。
SNAC 权重首次下载需 `export HF_ENDPOINT=https://hf-mirror.com`。

## 服务器最低配置要求

- GPU：**T4 级或更高（CUDA 12.8）**——3-30s 音频单次处理约 5-40s；显存
  ≥8 GB（SemantiCodec 权重约 3 GB）。无 GPU 时可用 SNAC 备选（CPU 3-5s/句）。
- 磁盘：40-60 GB（权重 + 环境 + 结果 TTL 缓存）。
- 系统：Ubuntu 22.04 + CUDA 12.8 驱动。
- 网络：放行 8710（API）+ 22（SSH）。
- 常驻（可选）：`systemd` 一条单元即够（`Restart=always` + `A038_CODEC=semanticodec`）。
- 公开演示的 HTTPS：挂 caddy/nginx 反代到 8710（前端若部署在 HTTPS 站点，
  接口须同为 HTTPS；本地 `npm run dev` 调试则 HTTP 即可）。

## 前端对接

- 方法配置：`public/methods/a038_unc_sew.json`（`adapter.endpoint` 部署后
  改为 `http://<公网IP>:8710/process` 或 HTTPS 域名）。
- 契约自检：仓库根 `npm test`（校验方法 JSON 与样例结构）+ `npm run build`。
- 展示样例为静态提交（`public/samples/a038_unc_sew/`，12 条贴满契约上限），
  无需生成步骤即可跑通方法页与冒烟自检。

## 已知边界（与复现报告口径一致）

- 水印承载在 float32 LSB：**任何有损转码/重量化会摧毁水印**（触发完整性
  报警，判决 UNMARKED_OR_CHANNEL）；请以 float32 WAV 传递水印信号。
- verify 对"从未水印"与"水印被信道摧毁"不可区分（论文理想信道口径）；
  两者均判 UNMARKED_OR_CHANNEL（fail-safe 报警侧）。同代码独立对照复核
  （2026-10-31）：legacy 失效检测率 0% vs P7 100%（300/300），完好水印
  零误报；详见复现报告 §6.3 表 2b。
- 理论可被提取并重绘 LSB 的自适应攻击者绕过（水印无密钥）。
- 同步等待式契约：单次 3-30s 音频在 T4 上约 5-40s；更长的音频会被拒绝。
