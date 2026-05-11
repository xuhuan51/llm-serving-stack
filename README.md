# LLM Serving Stack — 端到端 LLM 推理服务建设与性能优化

> 在 8×NVIDIA A30 (24GB, PCIe-only, 无 NVLink) 实验环境上从零搭建的 LLM 推理服务，
> 围绕 **K8s 部署、可观测性、张量并行、压力测试与瓶颈定位**展开的迭代项目。

## 项目目标

构建一套生产风格的 LLM 推理服务，覆盖：
- 容器化部署与 GPU 调度
- 全链路可观测性（TTFT / TPOT / Throughput / Queue / KV Cache / GPU 利用率）
- 多卡张量并行下的吞吐与延迟分析
- 长 prefill 干扰短请求的 P95 抖动归因
- 推理调度参数与拓扑优化方案对照
- 弹性扩缩容

## 版本路线图

| 版本 | 主题 | 状态 |
|------|------|------|
| V0 | 单卡 vLLM 推理基线 | ✅ |
| V1 | Kubernetes 部署（kubeadm + Calico CNI + NVIDIA Device Plugin） | ✅ |
| V2 | 可观测性体系（Prometheus + Grafana + SLO 面板） | ✅ |
| V3 | 多卡张量并行 scaling 实测（TP=1/2/4，含拓扑对照） | ✅ |
| V4 | 4 象限工作负载压测与 P95 抖动归因（46 类 metric） | ✅ |
| V5 | 70B AWQ 双引擎极限并发 + prefix cache ablation（vLLM × SGLang） | ✅ |
| V6 | HPA 弹性扩缩容 / 灰度上线 | 📋 |

## 已完成版本要点

### V1 — Kubernetes 部署

- kubeadm 装单节点 K8s v1.31.14 + Calico CNI（Pod CIDR `10.244.0.0/16`）
- containerd 配 nvidia runtime + NVIDIA Device Plugin
- 用 `NVIDIA_VISIBLE_DEVICES` 显式限制 device plugin 的 GPU 视野，避免与裸 docker GPU 容器冲突
- Pod / Service / Deployment 三件套实操：自愈、扩缩容、滚动升级、回滚
- vLLM Pod (Qwen2.5-7B-AWQ) + NodePort Service 端到端推理验证

详见 [docs/v1-k8s.md](docs/v1-k8s.md)，YAML 在 [deploy/vllm/](deploy/vllm/)。

### V2 — 可观测性体系（Prometheus + Grafana + SLO 面板）

在 K8s 集群内部署 Prometheus + Grafana，基于 vLLM `/metrics` 端点构建 11 个核心 panel 的 SLO 仪表板，覆盖从基础 SLI 到 LLM 推理专属指标的完整链路。

**Dashboard 11 panel 设计：**

| 类别 | Panel | 关键 PromQL |
|------|-------|-------------|
| **基础 SLI** | QPS / TTFT P95 / TPOT P95 / E2E P95+P99 | `histogram_quantile` over vLLM histogram metrics |
| **调度状态** | Running / Waiting Requests | `vllm:num_requests_running`、`_waiting` |
| **资源** | KV Cache Usage | `vllm:kv_cache_usage_perc` |
| **LLM 专属** | Prefill vs Decode P95 | `request_prefill_time` / `request_decode_time` 分段 |
| | Queue Time P95 | `request_queue_time_seconds` — 容量瓶颈信号 |
| | Preemption Rate | `rate(vllm:num_preemptions_total)` — KV 抢占红线 |
| | Prefix Cache Hit Rate | `prefix_cache_hits / prefix_cache_queries` |
| | Token Throughput | `prompt_tokens` vs `generation_tokens` 拆解 |

**验证压测（Qwen2.5-7B-AWQ, 24 并发 × 300 请求 × max_tokens=256）：**

- Sustained generation throughput **200 tokens/s**（A30 单卡 + 24 并发）
- Decode-bound 工作负载：prefill ≈ 0s（prefix cache 加速）+ decode ≈ 30s 主导 E2E ~40s
- KV cache 仅 1.5%，preemption=0 — 当前瓶颈是 GPU 算力分摊而非显存
- TPOT P95 = 150 ms/token，呈现 decode 阶段 memory-bound 的典型并发摊薄行为

**截图：**

基础 6 panel（QPS / TTFT / TPOT / E2E / Running-Waiting / KV Cache）：

![v2-basic](docs/screenshots/v2.4-grafana-vllm-slo-part1.png)

进阶 5 panel（Prefill-Decode 拆解 / Queue Time / Preemption / Prefix Cache / Token Throughput）：

![v2-advanced](docs/screenshots/v2.4-grafana-vllm-slo-part2.png)

部署 manifest：[deploy/monitoring/](deploy/monitoring/)（Prometheus + Grafana + Dashboard JSON）

### V3 — Tensor Parallel Scaling 实测

在 PCIe-only 拓扑下对 Qwen2.5-7B (BF16) 跑 TP=1/2/4 对照实验：

| 配置 | GPUs | req/s | TPOT P50 | TPOT P95 |
|------|------|------:|---------:|---------:|
| TP=1 | 4 | 14.45 | 23.6 ms | 401 ms |
| TP=2 PIX | 4,5 | 18.15 | 17.1 ms | 295 ms |
| TP=2 PHB | 2,4 | 17.62 | 17.2 ms | 295 ms |
| TP=4 PIX | 4-7 | 14.89 | 14.6 ms | **421 ms** |

**核心发现：**

1. **Decode TPOT scaling**：TP=2 提升 1.38×、TP=4 提升 1.62× — 子线性，受限于每层 all-reduce
2. **拓扑对照（TP=2）**：同 PCIe switch (PIX) vs 跨 host bridge (PHB) 仅差 ~3% — 7B 单 token AR 数据量约 230 KB/step，未打满 host bridge 链路
3. **TP=4 在 PCIe + 短输出场景下净亏损**：wall-time req/s 反而比 TP=2 慢 18%，all-reduce 开销吞掉计算并行收益；TPOT P95 从 295ms 退化到 421ms

数据：[benchmark/results/v3-tp-scaling/](benchmark/results/v3-tp-scaling/) ｜ 详细分析：[docs/v3-tp-scaling.md](docs/v3-tp-scaling.md)（写作中）

### V4 — 4 象限工作负载 P95 抖动归因

对 Qwen2.5-7B BF16 在 TP=1 / TP=2 PIX 两种配置下做 4 象限工作负载压测，覆盖短/长 prompt × 短/长输出共 8 个 run（n=100/run），基于 vLLM V1 engine 46 类 metric 做服务端归因。

**4 象限设计：**

| Q | input | output | 模拟场景 |
|---|-------|--------|----------|
| Q1 | 30 | ~30 | 短问答 chatbot |
| Q2 | 30 | ~894 | 写作 / 长生成 |
| Q3 | 2086 | ~50 | RAG 检索摘要 |
| Q4 | 2086 | ~932 | RAG + 长对话 |

**核心发现（6 finding，详见 [results/p1_4_workload_jitter/V4_summary.md](benchmark/results/p1_4_workload_jitter/V4_summary.md)）：**

1. **TP=2 decode 加速 1.58-1.61×** 跨象限稳定（比 V3 测的 1.38× 高，V3 被 cold-start 污染低估）
2. **ITL P95 干净归因**：T1-Q3 inter-token latency P95 = 33.1ms（长 prompt 短输出，prefill 挤占 decode），T2 降到 24.2ms
3. **Prefix cache 命中率 99.84%**（4.31M 查询）— vLLM V1 默认开启 prefix cache，Q3/Q4 长 prompt prefill 工作量 99.8% 被免；**该数字依赖 workload 同模板结构，写简历需 caveat**
4. **Batch 1→16 token 吞吐 ×15**（不是 ×16）→ 0.94 效率，验证 HBM 带宽 memory-bound → batching → compute-bound 理论曲线的物理上限
5. **Queue P95 ≈ 0**：concurrency=16 下无排队拥塞，E2E P95 完全由 decode P95 主导
6. **0 preemption + KV 利用率仅 20%**：concurrency=16 离 KV 压力区 5× 远，并发上限在 V5 推到 256+

**11 张图（A 时序 × 8 + B 分布 + C P95 拆解 + D throughput 对比）：**

| 类别 | 图 | 关键洞察 |
|------|-----|----------|
| A. 时序 | [`figures/A_timeseries_T*_Q*.png`](benchmark/results/p1_4_workload_jitter/figures/) (8 张) | batch_size / KV / TTFT / TPOT 时间序列叠图 |
| B. 分布 | [`figures/B_iteration_tokens_dist.png`](benchmark/results/p1_4_workload_jitter/figures/B_iteration_tokens_dist.png) | iteration_tokens 双峰：纯 decode vs prefill chunk 混入 |
| C. 拆解 | [`figures/C_p95_decomposition.png`](benchmark/results/p1_4_workload_jitter/figures/C_p95_decomposition.png) | E2E P95 三段拆解：queue / prefill / decode 各自占比 |
| D. 吞吐 | [`figures/D_throughput_compare.png`](benchmark/results/p1_4_workload_jitter/figures/D_throughput_compare.png) | TP=1 vs TP=2 PIX 在 4 个象限的 token throughput 对比 |

工具产出：[`v4_quadrant_runner.py`](benchmark/v4_quadrant_runner.py)（4 象限 benchmark 客户端）· [`v4_metrics_poller.py`](benchmark/v4_metrics_poller.py)（V1 engine 46 字段 1 Hz polling）· [`run_v4.sh`](benchmark/run_v4.sh)（orchestrator）

### V5 — 70B AWQ 双引擎极限并发 + prefix cache Ablation

在 PCIe-only TP=8 上对 **Qwen2.5-72B-Instruct-AWQ** 部署 vLLM v0.19.1 V1 engine + SGLang v0.5.10.post1 双引擎，做 6 点 concurrency sweep（c=32/64/128/256/384/512, n=500/点, Q4 长上下文 2086 in × 932 out, ignore_eos）+ 1 组 ablation。

**关键设置：**
- Qwen2.5-72B-Instruct-AWQ（4-bit weight quantization + Marlin kernel, sm_80）
- vLLM / SGLang 均启用 `--quantization awq_marlin` 走 Marlin 路径
- TP=8 over 8×A30, `gpu_memory_utilization / mem_fraction_static = 0.78`
- KV budget 实测 12.64 GB / 卡 × 8 = 101 GB，vLLM 自报 max_concurrency = 80.87× @ 4096 tok/req

**主 sweep 数据（c=32-256，8 run）：**

| 引擎 | c=32 tok/s | c=64 | c=128 | c=256 | 趋势 |
|------|-----------:|-----:|------:|------:|------|
| vLLM | 506 | 627 | 805 | **911** | 低并发赢 |
| SGLang | 471 | 618 | **823** | **920** | c≥128 反超 |

**6 个核心 finding（详见 [V5_summary.md](benchmark/results/p1_5_chunked_prefill/V5_summary.md)）：**

1. **0 preemption 全程**（8 run + 极限 c=384/512 共 12 run）— 但需 caveat：原因不是引擎强，是 prefix cache 99.84% 命中（同模板换尾巴 workload），下方 Ablation 验证
2. **吞吐交叉点 c=128**：vLLM 启动+warmup 优势在低并发；SGLang 激进 chunked prefill 调度在高并发反超
3. **尾延迟反向交叉**：c≤128 SGLang TPOT max 干净 2-4×（无 vLLM CUDA graph cold-start），c=256 vLLM max 459ms < SGLang 542ms 反过来更稳
4. **`max_num_seqs=256` 是真硬天花板（非 KV）**：c=384/512 实测 running 永远 ≤ 256, 多余进 waiting；TPOT 不变（440ms 一致）+ TTFT P95 暴涨 120× (2s → 259s)
5. **TTFT 是饱和并发下的真瓶颈**：c=256 TTFT P95 飙到 1.7-2.1s 但 waiting=0，是 chunked prefill split 跟随大 batch decode 拖慢；**生产 SLO P95 TTFT < 1s 时 c ≈ 100 是单实例经济点，超出应水平扩副本**
6. **Prefix cache 是 load-bearing wall（Ablation 验证）**：见下

**Ablation: `--no-enable-prefix-caching` 重跑 c=128/256**

| 指标 | c=128 cache-on | c=128 cache-off | Δ |
|------|---------------:|----------------:|---|
| tok/s | 805 | **280** | **-65%** |
| TTFT P95 | 882 ms | **176,921 ms** | **+200×** |
| TPOT P95 | 247 ms | 617 ms | +150% |
| TPOT P99 | 255 ms | **747 ms** | P99/P95 拉开 1.21× — preempt 尾巴 |
| KV peak | 39.6% | **100%** | 撞墙 |
| `num_preemptions_total` | 0 | **86** | 触发 |

c=256 ablation 客户端 `httpx.ReadTimeout` crash（部分请求 >60s 超时）。服务端 PromQL 显示 KV 撞 100% + preempt Δ=42 + waiting 高峰 250 — **关掉 prefix cache 时 c=256 在该硬件 + 70B 上实际无法工作**。

**反直觉点**：c=128 cache-off 的 280 tok/s **低于** c=32 cache-on 的 506 tok/s — 关 cache 时增加并发反而损害吞吐（preempt recompute 浪费的算力 > 增量并发收益）。

**5 张对比图：** [F1 throughput](benchmark/results/p1_5_chunked_prefill/figures/F1_throughput.png) · [F2 P99 ITL](benchmark/results/p1_5_chunked_prefill/figures/F2_p99_itl.png) · [F3a KV usage](benchmark/results/p1_5_chunked_prefill/figures/F3a_kv_usage.png) · [F3b preemption](benchmark/results/p1_5_chunked_prefill/figures/F3b_preemption.png) · [F4 TTFT](benchmark/results/p1_5_chunked_prefill/figures/F4_ttft.png)

**面试 Follow-up 防守清单**：[V5_interview_qa.md](benchmark/results/p1_5_chunked_prefill/V5_interview_qa.md)（10 题标准答 + 3 个 known-unknown）

工具产出：[`run_v5_sweep.sh`](benchmark/run_v5_sweep.sh)（engine-agnostic concurrency sweep wrapper）· [`run_v5_sglang.sh`](benchmark/run_v5_sglang.sh)（SGLang 启动器）· [`run_v5_extreme.sh`](benchmark/run_v5_extreme.sh)（c=384/512 极限延伸）· [`v5_promql_pull.py`](benchmark/v5_promql_pull.py)（per-run PromQL window 查询 + SGLang docker logs retract grep）· [`v5_plot.py`](benchmark/v5_plot.py)（双引擎对比 5 张图）

## 技术栈

- **推理引擎**：vLLM (BF16 / AWQ)，对照 SGLang
- **编排**：Kubernetes 1.31 (kubeadm) · Calico CNI · containerd · NVIDIA Device Plugin
- **可观测性**：Prometheus · Grafana · vLLM `/metrics` · `nvidia-smi dmon`
- **压测**：自研基于 OpenAI streaming API 的并发压测客户端
- **硬件**：8×NVIDIA A30 (24GB HBM2, PCIe Gen4, 无 NVLink)

## 仓库结构

```
llm-serving-stack/
├── docs/                          # 各版本独立文档
│   ├── v1-k8s.md
│   └── screenshots/               # Grafana SLO panel 截图
├── deploy/                        # K8s YAML
│   ├── vllm/                      # vLLM Pod / Service manifest
│   └── monitoring/                # Prometheus + Grafana + Dashboard JSON
├── benchmark/                     # 压测脚本 + 结果数据
│   ├── *.py / *.sh                # 并发压测、监控、PromQL 拉取、KV 传输微基准
│   ├── run_v4.sh                  # V4 4 象限 orchestrator
│   ├── run_v5_*.sh / v5_*.py      # V5 双引擎 sweep + PromQL pull + 出图
│   └── results/
│       ├── v3-tp-scaling/         # V3 TP=1/2/4 PCIe-only 拓扑对照
│       ├── p1_4_workload_jitter/  # V4 4 象限 × 8 run + 11 张图 + V4_summary.md
│       └── p1_5_chunked_prefill/  # V5 双引擎 sweep + 5 张图 + ablation + V5_summary + V5_interview_qa
└── notes/                         # 学习笔记与源码精读
    ├── vllm/                      # vLLM 源码与 scheduler 笔记
    ├── sglang/
    ├── disaggregated-prefill/
    ├── gpu-basics/
    ├── inference-optimizations/
    └── reports/                   # 实验记录与案例研究
```

## License

[MIT](LICENSE)
