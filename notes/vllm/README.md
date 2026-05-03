# Week 1: vLLM 单卡 7B 推理

目标: 用 Docker 跑 vLLM，单卡启 Qwen2.5-7B-Instruct，暴露 OpenAI 兼容 API

这个目录是顶层项目 `AI Infra Lab` 的第一阶段模块：**LLM Inference Benchmark and Observability**。

它要证明的能力不是“能调用大模型”，而是:

```text
能部署推理服务
能设计压测 workload
能测 TTFT / TPOT / P95 / tokens/s
能把 benchmark 结果和 vLLM scheduler / GPU / KV cache 指标关联起来
能给出瓶颈判断和调参方向
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `benchmark_v1.py` | 单组 OpenAI-compatible benchmark，支持 streaming、TTFT、TPOT 和不同 prompt mode。 |
| `sweep_concurrency.py` | 对多个 client concurrency 做横向对比，找吞吐和延迟的拐点。 |
| `sweep_matrix.py` | 对 concurrency 和 max_tokens 做矩阵实验。 |
| `monitor_vllm.py` | 抓 vLLM `/metrics` 和 GPU stats，输出 scheduler、KV cache、P95 latency 和规则诊断。 |

## 标准实验流程

推荐每次实验都按这个顺序跑:

```text
1. 确认 GPU 和 vLLM 服务状态
2. 启动 monitor_vllm.py
3. 跑 benchmark_v1.py 或 sweep_concurrency.py
4. 记录 benchmark 表格
5. 记录 monitor diagnosis
6. 把结果写入 reports/
```

小规模验证:

```bash
python3 week1-vllm-basics/benchmark_v1.py \
  --base-url http://localhost:8000/v1 \
  --model qwen \
  --prompt-mode short \
  --stream \
  --concurrency 2 \
  --num-requests 4 \
  --max-tokens 30 \
  --warmup-requests 0
```

长输出压测:

```bash
python3 week1-vllm-basics/benchmark_v1.py \
  --base-url http://localhost:8000/v1 \
  --model qwen \
  --prompt-mode long-output \
  --stream \
  --concurrency 8 \
  --num-requests 16 \
  --max-tokens 300 \
  --warmup-requests 1
```

并发 sweep:

```bash
python3 week1-vllm-basics/sweep_concurrency.py \
  --base-url http://localhost:8000/v1 \
  --model qwen \
  --prompt-mode long-output \
  --stream \
  --concurrency-list 8,16,24,32 \
  --num-requests 64 \
  --max-tokens 300 \
  --warmup-requests 1
```

长上下文 vs 长输出:

```bash
python3 week1-vllm-basics/benchmark_v1.py \
  --base-url http://localhost:8000/v1 \
  --model qwen \
  --prompt-mode long-context-output \
  --stream \
  --concurrency 8 \
  --num-requests 16 \
  --max-tokens 300 \
  --warmup-requests 1
```

## GPU placement 快速笔记

### 运行中判断用了哪张 GPU

```bash
nvidia-smi
nvidia-smi pmon -c 1
nvidia-smi --query-compute-apps=gpu_name,gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits
ps -fp <PID>
```

排查顺序:

1. 用 `nvidia-smi` 看每张卡的显存和 `Processes` 表。
2. 找到占显存的 PID。
3. 用 `ps -fp <PID>` 把 PID 映射回 `ollama`、`vllm` 或其他 Python 进程。
4. 用 `nvidia-smi pmon -c 1` 看 `sm` 利用率，确认哪张卡正在计算。

Benchmark 脚本通常只是 HTTP 客户端，不直接占 GPU；真正占 GPU 的是模型服务进程。

### 启动前指定 GPU

`CUDA_VISIBLE_DEVICES` 是进程级环境变量，用来限制一个进程能看见哪些物理 GPU。

```bash
CUDA_VISIBLE_DEVICES=1 python app.py
```

这表示 `app.py` 只能看见宿主机物理 GPU 1。注意：进程内部会把可见 GPU 重新从 `cuda:0` 开始编号，所以物理 GPU 1 在进程内部通常显示为 `cuda:0`。

多个 GPU:

```bash
CUDA_VISIBLE_DEVICES=1,3 python app.py
```

这表示进程内部只看见两张卡:

- 进程内 `cuda:0` -> 宿主机 GPU 1
- 进程内 `cuda:1` -> 宿主机 GPU 3

### vLLM 示例

单卡启动:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

两卡 tensor parallel:

```bash
CUDA_VISIBLE_DEVICES=1,3 vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
```

关键点: `--tensor-parallel-size` 要和希望 vLLM 使用的可见 GPU 数量匹配。

### Ollama 服务注意事项

如果 Ollama 是 systemd 服务启动的，当前 shell 里设置 `CUDA_VISIBLE_DEVICES` 不会影响已经运行的服务。需要在服务启动环境里设置，再重启服务。

示例思路:

```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES=1"
```

然后重启 Ollama 服务。共享 GPU 机器上不要随便重启公共模型服务，先确认没有其他人在用。

## vLLM 监控小脚本

`monitor_vllm.py` 是教学版监控脚本，直接抓 vLLM 的 Prometheus `/metrics`，并尝试读取本机 `nvidia-smi`。

在运行 vLLM 的 GPU 服务器上执行:

```bash
python3 week1-vllm-basics/monitor_vllm.py \
  --metrics-url http://localhost:8000/metrics \
  --interval 2 \
  --gpu-index 1 \
  --max-num-seqs 32
```

如果只想从本机抓远端 vLLM metrics，不读本机 GPU:

```bash
python3 week1-vllm-basics/monitor_vllm.py \
  --metrics-url http://<server-ip>:8000/metrics \
  --no-gpu
```

`--max-num-seqs` 是可选参数。传入 vLLM 启动时配置的 `max_num_seqs` 后，脚本会额外计算:

`--gpu-index` 也是可选参数。共享多卡机器上建议传入 vLLM 服务实际使用的宿主机 GPU，例如 `--gpu-index 1`，这样 GPU util 诊断不会被其他用户在别的卡上的负载污染。

```text
queue_ratio = waiting / (running + waiting)
running_saturation = running / max_num_seqs
```

脚本也会输出一行规则诊断:

```text
diagnosis name=scheduler_queueing severity=warning
 reason: waiting=24, queue_ratio=0.75; p95 TTFT=18.270s is high while p95 TPOT is normal; running saturation=1.00
 action: increase max_num_seqs if KV cache and GPU memory allow; reduce client concurrency or add another serving replica
```

教学阶段先用规则诊断，不直接调用大模型。原因是监控判断要稳定、可复现、低延迟；后面可以再把这些结构化诊断结果交给 LLM 生成排障报告。

重点看这些输出:

- `running`: vLLM 里正在服务的请求数。
- `waiting`: 等待进入调度的请求数；高了通常会拉高 TTFT。
- `swapped`: 出现时通常说明 KV cache 或显存压力已经比较明显。
- `queue_ratio`: 等待请求占 active 请求的比例，比单看 `waiting` 更容易判断排队严重程度。
- `running_saturation`: `running / max_num_seqs`，接近 `1.0` 说明 active slots 基本打满。
- `prompt tok/s`: prefill 输入 token 吞吐。
- `output tok/s`: decode 输出 token 吞吐。
- `p95 ttft`: 首 token 尾延迟，通常对应排队和 prefill 压力。
- `p95 tpot`: 输出 token 尾延迟，通常对应 decode 压力。
- `kv/cache usage`: KV cache 使用比例，接近上限时要关注 `max_model_len`、`max_num_seqs` 和显存。
- `GPU util/mem`: GPU 算力和显存使用情况。

判断套路:

```text
waiting 高 + TTFT 高 + TPOT 正常
=> 排队或 prefill 压力

output tok/s 高 + TPOT 高 + GPU util 高
=> decode 算力瓶颈

KV/cache usage 高 + 显存高 + 错误率上升
=> KV cache 或显存压力

P95/P99 高但 P50 正常
=> 尾部请求、长上下文或调度公平性问题
```

## 诊断闭环

这个模块的核心不是单独看某个指标，而是把四类信号连起来:

```text
benchmark:
P95 latency / TTFT / TPOT / tokens/s

vLLM scheduler:
running / waiting / swapped / max_num_seqs

memory:
KV cache usage / GPU memory

compute:
GPU util / output tok/s / prompt tok/s
```

常见结论:

| 现象 | 诊断 | 优先动作 |
|---|---|---|
| TTFT 高，TPOT 正常，waiting 高 | scheduler queueing | 降低 client concurrency，或在显存允许时提高 `max_num_seqs`。 |
| TTFT 正常，TPOT 高，GPU util 高 | decode bottleneck | 减少输出长度，增加实例，或尝试更快模型/量化。 |
| TTFT 和 TPOT 都高，GPU util 高 | overall saturation | 降低流量压力、拆分 workload、增加 serving capacity。 |
| swapped > 0 或 KV cache 接近上限 | KV cache / memory pressure | 降低 `max_model_len`、`max_num_seqs`，限制长上下文。 |

## 展示用实验报告

实验结果统一沉淀到顶层 `reports/`:

```text
reports/llm-inference-observability-template.md
reports/llm-inference-case-studies.md
```

每次报告都要回答:

```text
这个 workload 压到了什么？
benchmark 指标怎么变化？
monitor 看到 scheduler / GPU / KV cache 怎么变化？
诊断结论是什么？
下一步调参动作是什么？
```

这就是简历里要讲的 AI Infra 能力:

```text
不是只会跑模型，而是能定位模型服务为什么慢、为什么贵、为什么不稳定。
```
