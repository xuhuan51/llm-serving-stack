# vLLM Scheduler 学习笔记

按 codex 8 题清单整理，对应面试可直接讲。

---

## 1. 为什么需要 scheduler

GPU 每一步只能执行一次 forward，但同时有：
- 几十个**正在 decode** 的请求（每步 1 token）
- 几十个**等待 prefill** 的新请求（每步几十~几千 token）
- 显存（KV cache）有限
- 用户对 TTFT / TPOT 都有期望

Scheduler 解决的核心问题：**"这一步把哪些请求塞进 batch、各自算多少 token"**。这是一个多目标约束决策（吞吐 vs 延迟 vs 显存 vs 公平性）。

**面试一句话**：scheduler 是 vLLM 的大脑，每一步决定算什么、算多少，平衡 TTFT/TPOT/吞吐/显存。

---

## 2. waiting / running 两个队列

| 队列 | 含义 | 状态特征 |
|------|------|---------|
| `waiting` | 还没分到 KV cache 块的请求 | 一个 token 都没算 |
| `running` | 已分到 KV 块、正在 prefill 或 decode | 每步都参与 forward |

**调度循环**（每个 step）：
1. 看 `running` 里能继续 decode 的请求 → 留在 batch
2. 看 `running` 里要做 prefill 的请求 → 按 token 预算切块
3. 看 `waiting` 里能不能上车 → 受 `max_num_seqs` 和 KV 显存限制
4. 形成本步 batch → 一次 forward
5. 完成的请求出队，新到的请求进 `waiting`

**还有 preempted / swapped**：KV 显存压力大时，scheduler 可以把某些 running 请求踢回 waiting（preempt），下轮重新分配。

---

## 3. Continuous Batching 在 scheduler 里的实现

不是"一次性凑齐 batch 再算"，而是**每个 step 重新组装 batch**：

```
step N:    [req_a decode][req_b decode][req_c prefill 块1]
step N+1:  [req_a decode][req_b 完成-移除][req_c prefill 块2][req_d 新加入]
step N+2:  [req_a decode][req_c decode][req_d prefill][req_e 新加入]
```

关键点：
- **完成的请求立刻让位**，不必等同 batch 其他请求
- **新请求随时加入**，不必等下一个完整 batch
- KV cache 不搬动 — PagedAttention block table 改一下指针就行
- 调度开销 < 1%（CPU 几十微秒；GPU 一步毫秒级）

**面试一句话**：continuous batching = 每个 step 重新组 batch + KV 不搬动 + 完成立即让位。

---

## 4. Prefill 和 Decode 调度方式不同

| 阶段 | 计算特征 | 一步消耗 token | 调度优先级 |
|------|---------|--------------|----------|
| Prefill | **Compute-bound**（一次算几百~几千 token，矩阵乘大块） | 大（几百~`max_num_batched_tokens`） | 容易吃掉所有预算 |
| Decode | **Memory-bound**（每步只算 1 token，主要是读权重 + KV） | 小（每请求 1 token） | 多个请求并发摊销权重读取 |

**调度需要协调**：
- 纯 prefill 一步 → decode 请求被晾着 → TPOT 抖动
- 纯 decode 一步 → prefill 等很久 → TTFT 飙升
- **Chunked prefill** = 把长 prefill 切块、每步留点预算给 decode → 平滑两边

vLLM v1 默认开启 chunked prefill，把 prefill 分块和 decode 在同一个 step 混跑。

---

## 5. `max_num_seqs`

**含义**：一个 step 最多多少个**活跃请求**（running batch 大小上限）。

**调参权衡**：
- 太小 → GPU 填不满，吞吐低
- 太大 → KV 压力 + 调度开销 + P95 抖动

**典型值**：A30 24G + Qwen2.5-7B → 16~32

**和客户端 concurrency 的关系**：
- 客户端并发 > `max_num_seqs` → 超出部分进 waiting → **TTFT 升高、TPOT 正常**
- 这就是"强制排队"模式（你的实验：concurrency=32, max_num_seqs=8 → TTFT 暴涨 9 秒，TPOT 仍然 20ms）

---

## 6. `max_num_batched_tokens` / Chunked Prefill

**含义**：一个 step 最多算多少 token（含所有 running 请求的 prefill + decode token 总和）。

**和 `max_model_len` 的区别**：
- `max_model_len`：单个请求的最大上下文长度（拒绝点）
- `max_num_batched_tokens`：每步预算（切块点）
- 例：`max_model_len=8192`, `max_num_batched_tokens=2048` → 6000 token prompt **允许**（不拒绝），但分多个 step 完成 prefill

**Chunked prefill 工作机制**：
```
prompt = 6000 token, max_num_batched_tokens = 2048
step 1: prefill 块 1 (2048 token)  + 已 running 请求 decode
step 2: prefill 块 2 (2048 token)  + 已 running 请求 decode
step 3: prefill 块 3 (1904 token)  + 已 running 请求 decode + 新 decode token
step 4+: 这个请求进入 decode 阶段
```

每块 prefill 完成后，K/V 写入 cache，下一块 attend 到已缓存的 K/V — **是切块不是截断**，所有 token 都参与。

**调参权衡**：
- 预算大 → 长 prompt TTFT 低（少切几块），但可能阻塞短请求 decode
- 预算小 → 长 prompt TTFT 高，但 decode 更平滑

---

## 7. waiting 队列高 → TTFT 高

逻辑链：
```
请求到达 → 进 waiting
↓ (受 max_num_seqs / KV 显存限制)
不能立刻 admit → 卡在 waiting
↓ (waiting 时长 = 排队时间)
TTFT = 排队时间 + 第一个 token 计算时间
```

**怎么观察**：
- vLLM Prometheus metrics: `vllm:num_requests_waiting`、`vllm:num_requests_running`
- waiting > 0 持续上升 → 排队中
- waiting 经常 0 → 还有 admit 余量

---

## 8. TPOT 正常 + TTFT 高 = 典型排队问题

这是面试高频诊断题：

| 现象 | 含义 |
|------|------|
| TTFT 升 + TPOT **稳定** + GPU 没满 | **scheduler 排队**（admit 限制） |
| TTFT 升 + TPOT **也升** + GPU ~100% | **整体饱和**（算力跟不上） |
| TTFT 短请求正常 + 长 prompt TTFT 抖动 | **prefill 块阻塞 decode**（chunked prefill 没开或预算不当） |
| TPOT 升 + 输出长 + GPU 满 | **decode 阶段算力瓶颈** |

**自验实验**（已做过）：
- `concurrency=32`, `max_num_seqs=8`, long-output → TTFT 9.2s, TPOT 20ms, GPU 100%, running=8, waiting=16→0
- 这就是 case 1：scheduler 排队，模型本身没问题

**面试一句话**：TPOT 是模型算得快不快的指标，TTFT 是排队 + 模型起步的指标；TPOT 稳 + TTFT 升 = 调度限制不是算力限制，加 max_num_seqs 或加机器。

---

## 源码定位（待精读）

`vllm/v1/core/sched/`：
- `scheduler.py` — 主循环
- `request.py` — 请求状态机（waiting/running/preempted）
- 调度策略分散在 `_schedule_running` / `_schedule_waiting` / `_schedule_chunked_prefill`

下次精读优先点：
1. `schedule()` 函数的主循环顺序
2. KV 显存判断逻辑（什么时候 preempt）
3. chunked prefill 和 decode token 的预算分配规则
