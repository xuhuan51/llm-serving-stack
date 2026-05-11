# V5 面试 Follow-up 防守清单

> 每条 V5 简历 bullet 必扛的 10 个追问 + 答案大纲。
> 面试前刷 + 默写一遍。

---

## Q1 — 「你说 0 preemption，怎么验证不是字段没采到？」

**答**：
- vLLM：`vllm:num_preemptions_total` 是 V1 engine 标准 metric（V0 引擎才叫 `num_preemptions`），我用 PromQL `[end]@end - [start]@start` 差分跨整个 sweep 窗口 = 0
- SGLang：原生不导出 retract counter（这本身是发现），我用 `docker logs --since wall_start | grep -c retract` 计数，4 个 c 全 0
- 双工具 cross-validate，且 P99 ≈ P95（无 recompute 尖峰）行为也支持「真没 preempt」

## Q2 — 「prefix cache 99.84% 命中率为啥这么高？是不是数据集作弊？」

**答**：是 V4 已发现并 caveat 的特性。Q4 workload 用同一段 2086 token 的 RAG 长 prompt 模板填不同尾巴 → vLLM/SGLang 自动复用 prefix KV，命中率自然高。**实际 RAG 生产场景里 system prompt + 文档片段也常 8-9 成命中**，所以 caveat 但不算作弊。简历主推时会注明 workload 类型。

## Q3 — 「c=256 SGLang 920 vs vLLM 911 差 1%，这能算 SGLang 赢吗？」

**答**：单点不能。但**趋势可以**——4 个 c 点连起来看，c=128/256 SGLang 都领先，c=32/64 vLLM 领先，**有交叉点 c=128 这个一致信号**。1% 单点差是噪声，但趋势的方向稳定。最严谨的写法是「**两引擎在 c=256 收敛到 ~920 tok/s（差异 1%），表明这是该硬件 + 模型 + workload 的物理上限**」，不强行说 SGLang 赢。

## Q4 — 「TP=8 在 PCIe 上每层 all-reduce 多大数据？怎么算的」

**答**：
- Qwen2.5-72B：hidden_size = 8192, num_layers = 80, BF16 activation
- 单 token 单层 AR = `2 × hidden × bytes` = 2 × 8192 × 2 = 32KB
- 单 forward 80 层 = 80 × 32KB = 2.5MB AR / token
- PCIe Gen4 x16 单向 ~32GB/s → 单次 AR 78μs，80 次 = 6.24ms / forward 通信 only
- 结合 c=256 实测 TPOT 250-280ms 看 **AR 占比 ~2-3%**——通信不是主瓶颈，HBM 带宽 + compute 才是

## Q5 — 「为啥不用 nsys profile 验证 AR 时间？」

**答**：nsys 在 docker + 多 GPU + 长 run 上 setup 成本不小，本次 V5 优先 PromQL + 数学推算。如果要做精确归因，下一步会 nsys 抓 1-2s 短采样验证 AR span。这是 V5 的 known limitation，**不是没考虑，是时间窗内没必要**。

## Q6 — 「c=32 vLLM TTFT P95 9670ms 怎么解释？是不是测错了？」

**答**：CUDA graph cold-start outlier。第 1 个 prefill 请求触发 graph 编译（~9 秒），污染整组 P95（n=500 时 P95 是倒数第 25 个，前 5 个 outlier 完全够把 P95 拉爆）。这是 V3/V4 已知坑，可通过 warmup 加大或丢弃前 N 个 outlier 修。**SGLang 没这个问题因为它 model load 阶段就 capture CUDA graph 了**。这条 caveat 简历上会写「c≥64 起跳」或注脚说明。

## Q7 — 「SGLang 为啥默认不导出 retract metric？这是 SGLang 的 bug 吗？」

**答**：不是 bug，是设计选择。SGLang 团队认为 retract 是内部调度细节，对应用方暴露 `num_running / num_queue / token_usage` 三个 gauge 已足够 SLO 监控。但**对深度调优场景不够**，这正好是我做的工程补丁——`docker logs grep retract` 是 ad-hoc 但 work，长期方案是给 SGLang 提 PR 加 prom counter（已在我 todo）。**这条「我发现并补齐了 metric gap」本身就是一条强信号**——证明我不只是用工具，而是会扩展工具。

## Q8 — 「你这个结论在 NVLink 服务器上成立吗？」

**答**：**不可推广**。本实验所有结论限定 PCIe-only TP=8 拓扑。NVLink 下 AR 带宽从 32GB/s 跳到 600GB/s（A100/H100 NVLink），AR 开销可忽略，TP=8 比 TP=4 更划算（PCIe 下 V3 已观察到 7B TP=4 反而慢）。同样 prefix cache 命中率假设也依赖 workload，长 RAG 命中率高、纯 chat 命中率低。简历主推时会**明确标注硬件 + workload 假设**，不让读者误推广。

## Q9 — 「为啥不测 SGLang 0.5.11 而用 0.5.10？」

**答**：0.5.11 release 时间 < 1 个月（2026-05-05），生产 bake 不够。我选 0.5.10.post1 是 2026-04-09 release，已有 1 个月社区验证。这是工程稳定性 vs 新特性的常规取舍，**与 vLLM 选 v0.19.1 而非更新的 nightly 是一致的保守原则**。

## Q10 — 「你这个实验对生产部署的具体建议是什么？」

**答**：3 条具体建议：
1. **70B AWQ TP=8 PCIe 单实例最优经济点 c≈100**（P95 TTFT < 1s SLO 下），超出应水平扩副本而不是纵向加并发
2. **SLA 优先选 vLLM**（c=256 TPOT max 459ms 比 SGLang 542 更稳）；**总吞吐优先选 SGLang**（高并发区领先 1-3%）
3. **prefix cache 必须开**（vLLM 默认开），命中率 99.84% 是 c=256 不撞墙的关键缓冲；监控 `vllm:prefix_cache_hits_total / queries_total` 命中率，<80% 就要警惕 KV 撞墙

---

## 已知答不上的（提前准备）

### Q* — 「Marlin kernel 在 A30 sm_80 vs A100 sm_80 性能一样吗？」
答不上：没测过 A100。Marlin paper 里测 A100 居多，A30 数据少。需补 vLLM Marlin benchmark 文档查。

### Q* — 「为啥 prefix cache 命中率算法可能有 bug 不？」
答不上：vLLM `prefix_cache_hits_total` 计算口径要看源码确认是 token-level 还是 block-level，影响 99.84% 这个数的解释。需读 `vllm/v1/core/kv_cache_manager.py` 验证。

### Q* — 「为啥 SGLang 没用最新的 RadixAttention v2？」
答不上：v0.5.10 用什么 attention 实现没看源码。要查 [`sglang/srt/layers/attention`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention)。
