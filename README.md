# 第 A038 组：UNC-SEW 超低码率神经编解码器自嵌入水印

## 1. 小组与项目

| 信息 | 本组填写 |
| --- | --- |
| 组号 | A038 |
| 选题名称 | UNC-SEW（Ultra-low-bitrate Neural Codec Self-Embedding Watermarking） |
| 方法 ID | `a038_unc_sew`（与 `public/methods/a038_unc_sew.json` 一致） |
| 本组仓库与提交分支 | https://github.com/seer-vice/algorithm_research_frontend_A038（分支 `a038_submission`） |
| 算法代码位置 | 本仓库 `algorithms/a038_unc_sew/`（自包含，见下） |

本组方法为语音提供**主动防御**能力：发布端把语音经超低码率神经编解码器
（SemantiCodec，0.65 kbps）压成码流并以 R 份重复嵌入浮点 LSB，听感无差异
（PESQ≈4.6）；接收端盲提取载荷、自重建参考语音并与待测语音做 DTW 帧级
对比，实现**词级局部篡改（替换/删除/插入）的检测、接缝定位（约 4-12 ms）
与内容恢复**；水印被信道操作（有损压缩、加噪等）摧毁时触发完整性报警。

## 2. 论文与代码来源

- 论文：Self-embedding watermarking with ultra-low-bitrate neural codecs
  （arXiv 2608.25289，NII）。
- 基线 codec：SemantiCodec（arXiv 2405.00233，超低码率语义音频编解码器，
  0.65 kbps 配置）。
- 本组实现：完整复现论文三方（发布/攻击/接收）管线与 §IV 评估，并新增
  论文不具备的两个检测通道（CRC16 完整性校验 + LSB 一致性标尺定位），
  代码在本仓库 `algorithms/a038_unc_sew/src/`（纯算法模块 12 个，与
  复现报告同源零改动）。复现对表结果见 §6。

## 3. 本组实现内容

- **embed（嵌入）**：语音 → 神经 codec 码流 → `[sync64‖CRC16‖payload]×R`
  → float32 LSB → 水印信号 x̃（WAV）。输出嵌入报告（容量、R、PESQ）。
- **verify（检测）**：载荷盲提取（sync 搜索 + 间距推 N + 逐份 CRC + 多数
  投票）→ 自重建 R(y) → DTW 帧级代价 → 判决三态：
  `AUTHENTIC`（完好）/ `TAMPERED`（局部篡改，附接缝秒级定位与恢复音频）/
  `UNMARKED_OR_CHANNEL`（无水印或信道退化，触发完整性报警）。
- **本组改进（论文原版不具备）**：per-replica CRC16 完整性通道（水印死亡
  报警，零误报）；LSB 标尺通道（篡改接缝亚帧定位 4-12 ms）。
- 服务层：FastAPI 契约实现（同步处理、GPU 串行锁、结果 TTL 清理）。

## 4. 输入、输出与参数

- 输入：一个音频文件（WAV，16 kHz 单声道最佳，3-30 秒，≤20 MB；
  embed 上传原始语音，verify 上传待检测音频）。
- 输出：embed → 水印信号 WAV（float32）+ 嵌入报告 JSON；verify → 判决
  报告 JSON + 帧级代价曲线 PNG +（篡改时）恢复音频 WAV。
- 参数：`task`（embed/verify）；`lsb_k`（LSB 位数 1-4，默认 1=论文对表
  口径）；`payload_crc`（CRC 完整性校验，默认开）；`lsb_fusion`（LSB 标尺
  定位，默认开）。各参数含义详见方法配置与 `algorithms/a038_unc_sew/README.md`。

## 5. 环境、安装、启动与首次处理

```bash
# 1) 算法服务（华为云 T4 / 任何 CUDA 12.8 机器）
cd algorithms/a038_unc_sew
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py                      # CPU 冒烟：13 项全 PASS（mock codec）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e <SemantiCodec-inference>   # 真 codec（组内私有部署，见算法 README）
A038_CODEC=semanticodec uvicorn app:app --host 0.0.0.0 --port 8710

# 2) 样例生成（真 codec 机器，A038_P3 工程根目录，一次即可）
#    12 条契约样例：原声（embed 演示）+ 11 种攻击（论文三类 + 信道 8 类）
#    换句用 --seed（同 seed 确定性）；不满意判决换 seed 重跑即可
cd <A038_P3 根目录> && python make_examples.py --res-dir res/p7_improve \
    --out <本仓库>/public/samples/a038_unc_sew \
    --update-json <本仓库>/public/methods/a038_unc_sew.json

# 3) 前端
npm install && npm run dev                # http://127.0.0.1:5173
#    方法页：#/methods/a038_unc_sew（endpoint 已写入配置；部署后替换 IP）
```

首次处理：方法页选择"嵌入水印"任务 → 上传 3-30s 语音 WAV → 得到水印
信号下载；切换"篡改检测与恢复"→ 上传水印信号（或任一攻击样例音频）→
查看判决、接缝曲线与恢复音频。

## 6. 输入输出样例、复现参数与验证结果

样例位于 `public/samples/a038_unc_sew/`（由 A038_P3 工程的 `make_examples.py`
生成，12 条贴满平台契约上限）：原声 embed 演示（real → 水印信号 x̃，听感
透明度对比）+ 11 种攻击检测样例（论文三类词级操纵 → TAMPERED + 接缝定位；
信道 8 类扰动 → 水印死亡报警 UNMARKED_OR_CHANNEL）。每条样例的 description
含原句转写、篡改区间真值与观察重点。

复现验证（AV-Deepfake1M val，SemantiCodec 0.65 kbps，k=1）：

| 攻击 | 论文 EER | 本组复现 | 本组改进 | 接缝误差(中位) |
| --- | --- | --- | --- | --- |
| 直接替换 direct | 8.4 | 8.0 | 3.3 | 12 ms |
| TTS 替换 | 9.8 | 10.5 | 3.3 | 9 ms |
| 删除 delete | 17.5 | 18.0 | 6.7 | 4 ms |
| 插入 insert | 18.1 | 18.0 | 3.3 | 12 ms |

水印死亡报警（论文静默失效场景）：完好水印 30 例零误报、10 类信道扰动
300 例 100% 报警（absent 262 / corrupted 38）。同代码版独立对照复核
（2026-10-31，`p6_channel_re` vs `p7_channel`）：legacy 布局失效检测率
0%（静默）vs P7 **100%**；信道扰动下句级 EER 两侧均为无判别力区间
（0.30–0.93，DTW 主通道对整段扰动本无判别力），正确主指标为失效
检测率。判决口径：absent = 水印不存在或失效（fail-safe 报警侧）。
透明度 x̃ PESQ 4.644（论文≈4.6）；恢复 PESQ 比 1.58（论文 1.63）。

## 7. 已知限制

- 水印承载在浮点 LSB：任何有损转码（MP3 等）或重量化会摧毁水印（此时
  触发完整性报警而非误报）；水印信号须以 float32 WAV 传递。
- verify 无法区分"从未水印"与"水印被信道摧毁"（论文理想信道口径）。
- 理论上可被提取并重绘 LSB 的自适应攻击者绕过（水印无密钥，论文口径
  已知边界；密钥化为根治方向）。
- 单次处理为同步等待（平台契约）：3-30s 音频在 T4 上约 5-40s。

## 8. 附加文档入口

- 算法服务说明：[algorithms/a038_unc_sew/README.md](algorithms/a038_unc_sew/README.md)
- 复现报告（方法定义、公式、对表与改进实验全记录）：
  [algorithms/a038_unc_sew/docs/复现报告_A038.md](algorithms/a038_unc_sew/docs/复现报告_A038.md)
- 提交分支：`a038_submission`
