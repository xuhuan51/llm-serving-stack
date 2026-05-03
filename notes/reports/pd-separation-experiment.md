# PD 分离实验

三步实验，验证 prefill-decode 分离在 8 卡 A30 上的效果。

**核心问题**：把 prefill 和 decode 分到不同 GPU 上，能否减少互相干扰、提升 TPOT 稳定性？

---

## 实验环境

- 服务器：8x NVIDIA A30（24GB 每卡，全机无 NVLink）
- GPU 拓扑：GPU0–3 PHB（跨 CPU PCIe），GPU4–7 PIX（单桥 PCIe，约 21 GB/s 对等带宽）
- 已有容器：GPU0+1 → vLLM AWQ port 8000，GPU2 → SGLang BF16 port 8001，GPU3 → vLLM BF16 port 8002
- PD 实验用卡：GPU4（prefill），GPU5（decode）

---

## Step 1：混合工作负载基线 ✅ 已完成

**目标**：证明长上下文 prefill 会让短请求的 decode TPOT 升高。

模型：Qwen2.5-7B-Instruct-AWQ，vLLM port 8000。`concurrency=32`，`num_requests=200`，`warmup=4`，`max_tokens=200`，`short_ratio=0.7`。

脚本：`serving-benchmark/mixed_workload.py`

### 基线（仅短请求，n=200）

| 指标 | min | p50 | p95 | p99 | max | 均值 |
|------|-----|-----|-----|-----|-----|------|
| TTFT | 42.7ms | 96.5ms | 132.4ms | 216.7ms | 217.9ms | 92.7ms |
| TPOT | 9.9ms | 12.1ms | 13.1ms | 13.4ms | 13.4ms | 12.0ms |
| E2E | 281ms | 414.8ms | 529.1ms | 600.4ms | 618.1ms | 413.9ms |

### 混合（70% 短请求 + 30% 长上下文，n=200）

| 指标 | min | p50 | p95 | p99 | max | 均值 |
|------|-----|-----|-----|-----|-----|------|
| 短请求 TTFT | 66.3ms | 101.8ms | 253.3ms | 261.7ms | 271.5ms | 122.1ms |
| 短请求 TPOT | 12.0ms | 21.6ms | 25.3ms | 26.0ms | 26.1ms | 20.7ms |
| 短请求 E2E | 401ms | 684ms | 935.9ms | 1003.6ms | 1038ms | 678.7ms |
| 长请求 TTFT | 80.8ms | 111.6ms | 260.3ms | 331.2ms | 331.2ms | 133.2ms |
| 长请求 TPOT | 16.4ms | 21.8ms | 23.5ms | 24.1ms | 24.1ms | 21.3ms |
| 长请求 E2E | 831ms | 1437ms | 1784ms | 2089ms | 2089ms | 1373ms |

### 干扰汇总

| 指标 | 基线 | 混合 | 变化 |
|------|------|------|------|
| 短请求 P50 TPOT | 12.1ms | 21.6ms | **+79%** |
| 短请求 P95 TPOT | 13.1ms | 25.3ms | **+93%** |
| 短请求 P95 TTFT | 132ms | 253ms | **+92%** |
| 短请求 E2E P95 | 529ms | 936ms | **+77%** |

**关键观察**：基线 TPOT 极稳定（9.9–13.4ms）。混合模式扩散到 12–26ms。最早一批短请求（id 0–31）因为与第一波长上下文 prefill 碰撞，TTFT 达 232–271ms。干扰是脉冲式的，不是随机噪声。

**结论**：基线证据已建立，PD 分离的动机成立。

---

## Step 2：KV Transfer 微基准 ✅ 已完成

**目标**：量化 GPU4→GPU5 KV tensor 拷贝的时间成本。

脚本：`serving-benchmark/kv_transfer_bench.py`
运行环境：`Copilot_Hybrid RAG` 虚拟环境（cu128，不要用 t4-simulate cu130）

### 实测结果（GPU4→GPU5，PCIe PIX，peer=True，reps=50）

| 数据量 | P50 延迟 | P95 延迟 | P50 带宽 | 均值带宽 |
|--------|---------|---------|---------|---------|
| 128MB | 6.86ms | 6.89ms | 18.7 GB/s | 18.6 GB/s |
| 256MB | 12.59ms | 12.65ms | 20.3 GB/s | 20.3 GB/s |
| 512MB | 25.11ms | 25.19ms | 20.4 GB/s | 20.4 GB/s |

### KV cache 大小参考（Qwen2.5-7B，fp16，28层，4个 KV head，head_dim=128）

| 上下文长度 | KV cache 大小 | 估算传输时间 |
|-----------|-------------|------------|
| 4096 token | 约 230MB | 约 11ms |
| 8192 token | 约 460MB | 约 22ms |

**结论**：PIX PCIe 约 21 GB/s。KV 传输开销真实存在，但对典型上下文长度来说可接受。

---

## Step 3：PD 分离完整实验 ⚠️ 真 connector 未跑通，主线暂停

**目标**：prefill 在 GPU4（port 8010），decode 在 GPU5（port 8011），proxy 在 port 8012。与统一 BF16（port 8002）对比 TTFT/TPOT。

相关脚本：
- `serving-benchmark/pd_proxy.py` — FastAPI proxy，tokenize 请求，先调 prefill 再调 decode
- Docker 镜像：`vllm-with-msgpack:latest`（基础镜像缺少 msgpack 包）

---

### vLLM 0.19.1 P2pNcclConnector 踩坑记录

**ZMQ 端口分配规则**：
```
端口 = kv_port + world_rank
```
单卡容器的 `world_rank` 永远是 0。两个容器必须用**不同的 kv_port**，否则都绑同一端口冲突：
- Prefill：`kv_port=14579` → 绑定 `192.168.1.101:14579`
- Decode：`kv_port=14580` → 绑定 `192.168.1.101:14580`

**`get_ip()` 的坑**：返回 LAN IP（`192.168.1.101`），不是 `127.0.0.1`。ZMQ ROUTER 绑在 LAN IP 上，所有对端地址必须用 LAN IP，不能用 localhost。

**`request_id` 必须嵌入双方 ZMQ 地址**，`P2pNcclConnector.parse_request_id()` 用正则从 request_id 里提取对端地址：
- Prefill 读：`___decode_addr_IP:PORT`
- Decode 读：`___prefill_addr_IP:PORT___`

正确格式：
```
{uuid}___prefill_addr_192.168.1.101:14579___decode_addr_192.168.1.101:14580
```

**最后一次尝试的启动方式（GET 模式，正确 IP + NCCL 环境变量）**：
```bash
# Prefill（GPU4，port 8010）
docker run -d --name vllm-prefill --init --ipc=host --network host --gpus device=4 \
  -v /home/liuguangli/models:/models \
  -e NCCL_SOCKET_IFNAME=ens18 -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=PIX \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  vllm-with-msgpack:latest \
  --model /models/Qwen2.5-7B-Instruct --port 8010 --max-model-len 4096 \
  --served-model-name qwen-pd \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_ip":"192.168.1.101","kv_port":14579,"kv_connector_extra_config":{"send_type":"GET","nccl_num_channels":"4"}}'

# Decode（GPU5，port 8011）
docker run -d --name vllm-decode --init --ipc=host --network host --gpus device=5 \
  -v /home/liuguangli/models:/models \
  -e NCCL_SOCKET_IFNAME=ens18 -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=PIX \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  vllm-with-msgpack:latest \
  --model /models/Qwen2.5-7B-Instruct --port 8011 --max-model-len 4096 \
  --served-model-name qwen-pd \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_ip":"192.168.1.101","kv_port":14580,"kv_connector_extra_config":{"send_type":"GET","nccl_num_channels":"4"}}'
```

---

### 已确认的事实

- 裸 NCCL 跨容器 / 跨 GPU4-GPU5 成功：`torch.distributed` NCCL all-reduce 返回 `3.0`
- NCCL 环境变量有效：`NCCL_SOCKET_IFNAME=ens18`、`NCCL_IB_DISABLE=1`、`NCCL_P2P_LEVEL=PIX`
- GPU4-GPU5 拓扑是 PIX，裸拷贝带宽约 21 GB/s
- vLLM GET 模式下，prefill `/inference/v1/generate` 能返回 200
- decode 收到请求后进入 `set_p2p_nccl_context`，随后请求无 token 返回，GPU 利用率不继续上升

### 当前结论

这不是机器不能跨卡通信，也不是 Docker 基础 NCCL 配置完全错误；更像是 vLLM 0.19.1 experimental `P2pNcclConnector` 在当前“两独立单卡容器 + 手写 proxy + GET/PUT 请求配对”链路里的兼容性/调用方式问题。

项目叙事保留到 Step1/Step2/proxy 架构即可：能讲清楚为什么需要 PD 分离、prefill 如何干扰 decode、KV transfer 成本大概多少、真实 connector 为什么复杂。真 connector 后续作为 optional deep dive，不再阻塞 scheduler / PagedAttention / MQA-GQA-MLA 主线。

---

### 如果以后继续 Step3，需要收集的指标

| 指标 | 统一 BF16（port 8002） | PD 分离（port 8012） |
|------|----------------------|---------------------|
| TTFT P50/P95 | | |
| TPOT P50/P95 | | |
| 混合负载下短请求 TPOT | | |
| Prefill GPU 利用率 | — | |
| Decode GPU 利用率 | — | |
| KV transfer 时间 | — | |
