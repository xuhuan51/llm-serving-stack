# V4 — Workload Jitter Attribution

## Goal
分离 **workload-induced** 与 **communication-induced** 的 P95 抖动来源；
为 V5 chunked prefill / 调度参数对照实验提供 baseline。

## Hardware
- 8×NVIDIA A30 (24GB) PCIe-only, no NVLink
- 模型路径：`/home/liuguangli/models/Qwen2.5-7B-Instruct` (BF16)

## Model Config
- dtype = bfloat16
- max-model-len = 4096
- gpu-memory-utilization = 0.85

## 4-Quadrant Workload Matrix
| Quadrant | Input prompts (vLLM PROMPT_MODE) | Approx input tokens | Output tokens (max) | ignore_eos | Real-world analog |
|----------|----------------------------------|---------------------|---------------------|------------|-------------------|
| Q1 short-short | `short` (5 个一句话问题) | ~30 | 50 | no | API 调用、简单问答 |
| Q2 short-long  | `short`                          | ~30 | 1000 | **yes** | 生成式（写文章/代码） |
| Q3 long-short  | `long-context` (40 段背景 + 短问题) | ~1500-2500 | 50 | no | RAG / 文档摘要 |
| Q4 long-long   | `long-context`                   | ~1500-2500 | 1000 | **yes** | 长文翻译 / 深度对话 |

**ignore_eos** 仅对长输出象限开启，强制生成到 max_tokens，使 output 长度可控可比。

## Test Config (per run)
- concurrency = 16
- num_requests = 100
- warmup = 5
- 每个象限独立跑（避免象限互相串味）
- 象限之间 `sleep 5` 让 batch / KV cache 沉淀

## TP Configs
| Tag | TP | GPUs | Port | 备注 |
|-----|----|----|------|------|
| **T1** | 1 | 7 | 8000 | 单卡 baseline，无通信噪声 |
| **T2** | 2 PIX | 4,5 | 8001 | 同 PCIe Switch，对照通信放大 |

## Total Runs
4 象限 × 2 TP = **8 runs**

## Output Files (per run)
每个象限产出 4 个文件：
- `R{TAG}_{Q}_result.json` — benchmark 主输出（分位数 + 原始数据 + wall_start_ts）
- `R{TAG}_{Q}_raw.txt`     — 终端日志
- `R{TAG}_{Q}_metrics.jsonl` — Prometheus 时间序列（1Hz 采样）
- `R{TAG}_{Q}_dmon.log`    — `nvidia-smi dmon` GPU 利用率

外加（手动）：
- `R{TAG}_{Q}_grafana.png` — 同时段 Grafana 截图（TPOT P95 + num_requests_running 叠图）

## Hypotheses (V4 跑完回填)
- **H1**：Q4 (长-长) P95 ≫ Q1 (短-短)，因为 prefill 撞 decode + KV cache 累积压力
- **H2**：T2/Q4 vs T1/Q4 P95 增幅 > T2/Q1 vs T1/Q1，证明通信开销在重 workload 下放大
- **H3**：Q3 (长-短) TTFT 主导总耗时（prefill 重）；Q2 (短-长) TPOT 主导（decode 累）
- **H4**：metrics.jsonl 里 num_requests_running 尖刺与 TPOT P95 spike 时间戳对应

## Resume Pitch (V4 完成后回填)
> "在 V3 多卡 TP scaling 基础上，进一步设计 4 象限混合 workload 实验，
> 用 wall-clock 对齐的 Prometheus 时间序列把 TPOT P95 抖动归因到
> {prefill 撞 decode / KV cache 满 / batch 涌入} 三类机制，量化 PCIe 拓扑下
> 通信开销对长 workload 的放大系数为 X.X×。该方法学为 V5 chunked prefill
> 改进提供了归因基线。"

## Next: V5 Chunked Prefill 对照
用同样 8 runs 跑一遍 chunked prefill 开启版（`--enable-chunked-prefill`），对照 P95 改善幅度。
