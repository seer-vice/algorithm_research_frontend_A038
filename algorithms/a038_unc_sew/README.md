# UNC-SEW 算法服务（A038 自包含实现）

A038 复现工程（论文 arXiv 2608.25289，复现报告见仓库根 README）的服务器化：
FastAPI 薄适配层 + `src/` 内 12 个纯算法模块（与复现报告**同源零改动**，
P7 改进默认开启：CRC 完整性通道 + LSB 标尺定位）。

## 目录

```
app.py            平台契约层（POST /process、/files 静态结果、CORS、TTL 清理、GPU 串行锁）
service.py        算法适配（embed/verify、判决规则、报告/曲线/恢复输出）
smoke_test.py     CPU 冒烟自检（13 项，mock codec，无需 GPU）
make_samples.py   生成前端展示样例（须真 codec 机器）
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

# 3) 真 codec（华为云 GPU 机器）+ 启动
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

## 华为云部署建议（500 元代金券）

- 机型：**GPU 型 T4（g6v/p2s 系）× 1，按需计费**——T4 按需约 ¥1.2~2.8/h，
  500 元代金券 ≈ 200~400 演示小时；演示/验收时段开机，平时关机（代金券
  自动抵扣按需费用）。磁盘 40~60 GB 通用型即可（SemantiCodec 权重 + 环境）。
- 系统：Ubuntu 22.04 + CUDA 12.8 驱动（GPU 镜像自带）。
- 安全组：放行 8710（API）+ 22（SSH）。
- 常驻（可选）：`systemd` 一条单元即够（`Restart=always` + `A038_CODEC=semanticodec`）。
- 公开演示的 HTTPS：挂 caddy/nginx 反代到 8710（前端若部署在 HTTPS 站点，
  接口须同为 HTTPS；本地 `npm run dev` 调试则 HTTP 即可）。

## 前端对接

- 方法配置：`public/methods/a038_unc_sew.json`（`adapter.endpoint` 部署后
  改为 `http://<公网IP>:8710/process` 或 HTTPS 域名）。
- 契约自检：仓库根 `npm test`（校验方法 JSON 与样例结构）+ `npm run build`。
- 样例生成（12 条契约版，A038_P3 工程根目录跑）：
  `python make_examples.py --res-dir res/p7_improve --out <本仓库>/public/samples/a038_unc_sew --update-json <本仓库>/public/methods/a038_unc_sew.json`
  （`--seed N` 换句；旧单句样例 `make_samples.py` 保留可作调试）。

## 已知边界（与复现报告口径一致）

- 水印承载在 float32 LSB：**任何有损转码/重量化会摧毁水印**（触发完整性
  报警，判决 UNMARKED_OR_CHANNEL）；请以 float32 WAV 传递水印信号。
- verify 对"从未水印"与"水印被信道摧毁"不可区分（论文理想信道口径）；
  两者均判 UNMARKED_OR_CHANNEL（fail-safe 报警侧）。同代码独立对照复核
  （2026-10-31）：legacy 失效检测率 0% vs P7 100%（300/300），完好水印
  零误报；详见复现报告 §6.3 表 2b。
- 理论可被提取并重绘 LSB 的自适应攻击者绕过（水印无密钥）。
- 同步等待式契约：单次 3-30s 音频在 T4 上约 5-40s；更长的音频会被拒绝。
