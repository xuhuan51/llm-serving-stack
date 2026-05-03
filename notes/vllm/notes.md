# vLLM 技术笔记

服务参数、调度行为、诊断模式、环境配置。

---

## 关键服务参数

### `max_num_seqs`
调度器同时允许的最大活跃请求数。

- 客户端 `--concurrency` 是客户端压力；`max_num_seqs` 是服务端容量控制
- 客户端并发 > `max_num_seqs` → 超出部分排队 → TTFT 升高，TPOT 正常
- 太小 → GPU 填不满，吞吐低；太大 → KV 压力大，P95/P99 变差

### `max_num_batched_tokens`
每个调度步骤的 token 预算，不同于 `max_model_len`：
- `max_model_len=8192`，`max_num_batched_tokens=2048` → 6000 token 的 prompt 允许，但分多个步骤完成 prefill
- `max_model_len=4096` → 6000 token 的 prompt 直接拒绝
- 预算越大 → 长 prompt TTFT 越低（prefill 步骤少），但可能阻塞短请求的 decode

### `max_model_len`
每个请求允许的最大上下文长度（prompt + 生成 token + chat template 开销）。

- 降低此值 → 减少 KV cache 压力，提升可并发数
- 生产环境通常设 4096–8192，即使模型支持 128k，因为 KV cache 容量有限

### `gpu_memory_utilization`
vLLM 可使用的 GPU 显存比例（含权重 + 运行时 + KV cache）。

- 0.85 保守，0.90 常用，0.95 激进；共享环境不要用 1.0

### 调参起点（Qwen2.5-7B-Instruct，A30 24GB，对话场景）
```
gpu_memory_utilization=0.90
max_model_len=4096
max_num_seqs=16
max_num_batched_tokens=8192
```
然后根据 TTFT/TPOT/P95 实测调整。

---

## Chunked Prefill、Prefix Caching、Speculative Decoding

**Chunked Prefill（分块 prefill）**：把长 prompt 拆成多个调度轮次完成，每块的 K/V 存入 KV cache，后续块 attend 到已缓存的 K/V。是分块不是截断，所有 token 都保留。
- 权衡：块越大 → 长请求 TTFT 越低，但可能阻塞短请求的 decode

**Prefix Caching（前缀缓存）**：重复/共享的 prompt 前缀复用已计算的 per-layer K/V，减少重复 prefill 工作（如系统 prompt、RAG 上下文）。
- 降低缓存命中请求的 TTFT；对 TPOT 无影响

**Speculative Decoding（推测解码）**：小草稿模型提出若干未来 token，大目标模型一次性验证。接受率高时可降低有效 TPOT。
- 不是"信任小模型"——目标模型控制验证；收益取决于草稿速度和接受率

---

## 诊断模式速查

| 症状 | 可能原因 | 处理方向 |
|------|---------|---------|
| TTFT 高，TPOT 正常，GPU 未满，`running` 被限制 | 调度排队（`max_num_seqs` 太小） | 增大 `max_num_seqs` |
| TTFT 高，TPOT 高，GPU ~98%，显存高 | 整体饱和 | 限流、加 GPU、缩短输出长度 |
| 长 prompt TTFT 高，短 prompt 正常，TPOT 正常 | 大 prefill 块阻塞 decode | 检查 `max_num_batched_tokens`，启用 chunked prefill |
| OOM / "KV cache 不足"，`max_model_len=32768`，`max_num_seqs=64` | KV cache 耗尽 | 降低 `max_model_len` 和/或 `max_num_seqs` |
| TTFT 正常，TPOT 高，GPU 接近满，输出很长 | Decode 算力瓶颈 | 缩短输出或加 GPU |
| 某个并发点 TTFT 毛刺，TPOT 稳定 | 调度/缓存状态偶发异常 | 复现后再下结论 |

---

## GPU 选卡

```bash
# 单卡，绑定到 GPU1
CUDA_VISIBLE_DEVICES=1 vllm serve <模型> --host 0.0.0.0 --port 8000

# 多卡张量并行（GPU1 和 GPU3，TP=2）
CUDA_VISIBLE_DEVICES=1,3 vllm serve <模型> --tensor-parallel-size 2
```

容器内：`CUDA_VISIBLE_DEVICES` 把宿主机 GPU N 映射为容器 `cuda:0`。

查看 GPU 占用：
```bash
nvidia-smi --query-compute-apps=gpu_name,pid,process_name,used_memory --format=csv,noheader
ps -fp <pid>
```

---

## 本机 Docker 容器一览

| 容器名 | GPU | 端口 | 模型 | 备注 |
|--------|-----|------|------|------|
| vllm-qwen-awq | 0+1 | 8000 | qwen-awq（AWQ 量化） | 长期运行，共享 |
| vllm-qwen-bf16-tmp | 3 | 8002 | qwen（BF16） | 对比实验用 |
| vllm-prefill | 4 | 8010 | qwen-pd（BF16） | PD 分离实验，prefill 端 |
| vllm-decode | 5 | 8011 | qwen-pd（BF16） | PD 分离实验，decode 端 |

PD 分离专用镜像：`vllm-with-msgpack:latest`（基础镜像缺 msgpack）。

---

## vLLM Scheduler 源码（待读）

当前理解：
- 调度器维护 waiting/running 队列，决定每步进入 batch 的请求
- `running` 上限为 `max_num_seqs`，超出排入 `waiting`
- `max_num_batched_tokens` 是每步 token 预算；prefill 消耗多，decode 每步约 1 token/活跃请求

还不清楚：
- vLLM v1（0.19.x）中 waiting/running/preempted/swapped 状态的具体切换逻辑
- 源码路径：`vllm/v1/core/sched/` — 下一步阅读
