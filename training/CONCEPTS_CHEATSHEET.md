# 训练侧概念 Cheat Sheet（V0-V6 训练补丁配套）

> 本次会话覆盖到的所有训练相关概念整理。
> 面试前刷一遍，每个 "面试官问→答" 默写 80% 即过关。
> 不熟的点重新复读对应 § 章节。

---

## § 1. LLM 训练 4 阶段总览

```
Stage 1: Pretrain（预训练）
  从随机权重 → 喂海量无监督文本（5-15T tokens）
  目标：让模型学语言 + 世界知识
  产物：base 模型（如 Qwen2.5-32B-base）
  成本：千万-亿美元 GPU × 3-6 月（H100 集群）
  我们做吗？❌ 绝对不做

Stage 2: SFT (Supervised Fine-Tuning, 监督微调)
  在 base 上用 instruction-output 标注数据继续训练
  目标：让模型学"按指令格式回答"
  产物：Instruct 模型（如 Qwen2.5-32B-Instruct）
  我们做吗？✅ 这是我们的 Continued SFT 阶段

Stage 3: RLHF / DPO（人类偏好对齐）
  用偏好数据让模型 Helpful / Harmless / Honest
  产物：最终 Aligned 模型
  我们做吗？📋 Week 2/3 可选补 DPO（已淘汰传统 PPO）

Stage 4 (旁支): Continued Pretrain / Domain SFT
  在特定领域语料继续训练
  产物：领域适配模型
  我们做吗？📋 不在 plan
```

**面试官问 → 答**：
- 「LLM 训练有几个阶段」→ Pretrain → SFT → RLHF/DPO 三大主线 + Continued Pretrain 旁支
- 「你做的是哪个阶段」→ Continued SFT，在 Qwen 团队已 Instruct 过的模型上叠加 LoRA 适配器
- 「为啥不做 Pretrain」→ 千万美元 GPU + 3-6 月时间，业余项目不现实

---

## § 2. 三种并行策略（DP / TP / PP）

```
DP (Data Parallel) 数据并行
  每卡放完整模型 copy + 自己一份不同 batch
  反向传播后 AllReduce gradient 同步
  ★ 单卡能装下时用，最简单
  通信：每 step 1 次 AllReduce on gradient（数据量 = weight size）

TP (Tensor Parallel) 张量并行  
  一层 weight 在多卡间列切/行切
  forward 中间结果 AllReduce
  ★ 单卡装不下时用，每层都通信，对带宽极敏感
  通信：每层 1 次 AllReduce on activation（数据量 = hidden × batch × bytes）

PP (Pipeline Parallel) 流水线并行
  按 layer 切：卡 0 算 1-10 层，卡 1 算 11-20 层
  micro-batch 错位执行
  ★ 跨节点最常用，通信少但有 bubble
  通信：每 layer 边界 1 次 send/recv on activation
```

**3D Parallelism**：大厂训超大模型 `DP × TP × PP` 三个一起用。

**面试官问 → 答**：
- 「三种并行有啥区别」→ 按上面表格答
- 「我们 8 卡 A30 训 7B 用啥」→ 纯 DP，单卡装得下 7B，TP/PP 用不上
- 「为啥 TP 对带宽敏感」→ 每层都要 AllReduce，PCIe-only 拓扑下 70B 训练 AR 开销吞 30-50% wall time

---

## § 3. ZeRO-1 / -2 / -3 详解

DP 简单但每卡都要存完整模型状态，**7B Full FT 单卡需 56GB（weight 14 + grad 14 + opt 28），A30 24GB 装不下**。

**ZeRO 思路**：DP 仍跑各自 batch，但模型状态在卡间切开存不复制。

```
ZeRO-1：切 optimizer state（最大头，2× weights）
        7B FFT: 28GB/8 = 3.5GB optimizer + 14GB weight + 14GB grad = 31.5GB/卡
        → A30 仍装不下

ZeRO-2：切 optimizer + gradients
        7B FFT: 3.5GB + 1.75GB + 14GB = 19.25GB/卡
        → ✅ A30 8 卡跑 7B 全 FT 推荐配置

ZeRO-3：切 optimizer + gradients + weights  ← 32B 必用
        7B FFT: 3.5GB + 1.75GB + 1.75GB = 7GB/卡
        → 显存最省，但通信代价最大
```

**ZeRO-3 工作流（每 step）**：
```
forward 每层:
  AllGather → 8 卡拿到完整 weight → 算 forward → Release（释放临时 buffer）

backward 每层:
  AllGather → 拿完整 weight 算 backward
  ReduceScatter → 每卡只拿自己 1/8 的 gradient

optimizer step:
  纯本地更新自己 1/8 的 weight（用本地 grad + 本地 optimizer state）
```

**3 个调优旋钮**：
- `overlap_comm=true` ← AllGather 跟 compute 重叠
- `stage3_prefetch_bucket_size` ← 预取下一层 weight
- `stage3_max_live_parameters` ← 临时显存上限（太小通信频繁，太大 OOM）

**面试官问 → 答**：
- 「ZeRO-1/2/3 切啥」→ optimizer / + grad / + weight，每级多切一项，显存递减通信递增
- 「7B Full FT 用啥」→ ZeRO-2 甜点（装得下 + 通信不重）
- 「70B Full FT 用啥」→ ZeRO-3 + offload，或上 TP+PP
- 「LoRA 为啥也要 ZeRO-3」→ base weight 32B 单卡装不下，必须切；LoRA 自己的 grad/opt 极小不需要 ZeRO

---

## § 4. LoRA 原理 + 适用场景

**LoRA = Low-Rank Adaptation**（Microsoft 2021）：

```
原 Linear:    y = W·x          ← W 大矩阵 (8192×8192 = 67M 参数)

LoRA Linear: y = W·x + (B·A)·x  ← W 冻结
                  ↑      ↑
               (8192×r)  (r×8192)  r = rank（典型 16-32）
               共 ~260k 参数（缩 256×）

训练：W 冻结，只更新 A、B
推理：W + B·A 合并成新 weight
```

**LoRA vs Full FT**：
| 维度 | Full FT | LoRA |
|---|---|---|
| 训练显存 | 高（W 的 grad+opt） | 低（只 A,B 的 grad+opt） |
| 训练速度 | 慢 | 快 |
| 质量 | 上限高 | 中（适合"风格"，不改深层能力） |
| 灵活性 | 一次训一个模型 | 多个 LoRA 适配器随插随切 |

**适用判断**：
- 通用领域加层风格 → LoRA 够
- 深层能力升级（数学、代码） → 倾向 Full FT
- 显存紧张 / 多个适配器需求 → LoRA

**面试官问 → 答**：
- 「LoRA 是啥」→ 低秩分解适配器，base 冻结只训 A·B 两个小矩阵，参数缩 256×
- 「LoRA 缺点」→ 表面风格调可以，深层能力提升弱；rank 选小了容量不够，选大了不如 Full FT
- 「rank/alpha 怎么选」→ rank=16, alpha=32 是典型起点；rank 越大容量越大但接近 Full FT

---

## § 5. 量化基础（W4A16 / AWQ / Marlin）

**量化 = 用更少 bit 存每个参数**：
- FP32 4 字节 / FP16 BF16 2 字节 / INT8 1 字节 / **INT4 0.5 字节**

**W{X}A{Y} 命名**：
- W = Weight（权重）量化到 X bit
- A = Activation（中间张量）量化到 Y bit
- AWQ / GPTQ = W4A16（weight 4-bit, activation FP16）
- W8A8 = SmoothQuant 系列

**AWQ (Activation-aware Weight Quantization)**：
- 找出对应大 activation 的"重要列" → 给更高精度
- 不重要列 → 大胆 4-bit
- 推理时 GEMM = `INT4 weight × FP16 activation` → 输出 FP16

**Marlin Kernel**：
- 4-bit GEMM 专用 CUDA kernel
- 把 dequantize + 矩阵乘融合在寄存器，省中间显存读写
- vs 原始 AWQ kernel 快 **2-3×**
- 支持 sm_80+（A30 / A100 / H100 都能用）

**FP8 caveat**：A30 Ampere 不支持 FP8 Tensor Core（要 H100 Hopper），简历**不能写「在 A30 实现 FP8 加速」**。

**面试官问 → 答**：
- 「AWQ 是啥」→ 4-bit weight quantization，activation-aware 重要列保护
- 「Marlin 解决啥问题」→ 4-bit GEMM 专用 kernel，融合 dequant + matmul，避免中间显存读写
- 「为啥不用 FP8」→ A30 是 Ampere，FP8 硬件支持极弱（要 H100）

---

## § 6. 2026 训练方法热度排行

| 方法 | 工业地位 | 简历必备？ |
|---|---|---|
| **LoRA** | 基础必会 | ✅（baseline） |
| **QLoRA** | LoRA 省显存版 | ✅（生产常用） |
| **Full FT** | 金标准，质量上限高 | ✅（证明能跑大训练） |
| **DPO** | 2024-26 主流对齐，已取代多数 PPO | ⭐⭐⭐（强烈推荐补） |
| **DoRA** | 2024 SOTA，比 LoRA 收敛快 | 🟡 加分 |
| **GaLore** | 2024 Full FT 低显存 | 🟡 加分 |
| **PPO + GRPO** | DeepSeek 推热，模型公司专用 | ⭐⭐（T1 加分） |
| **MoE 训练** | DeepSeek-V3 / Qwen3-MoE 推热 | ⭐⭐（T1 加分） |
| Megatron 3D 并行 | T1 必懂（70B+） | ⭐⭐ |
| Adapter / Prefix / P-Tuning | 已淘汰 | ❌ |
| 纯 PPO RLHF | 概念懂即可 | ❌ |

**当前 plan**：Week 1 LoRA → Week 2 Full FT。DPO/MoE 是后续 stretch。

---

## § 7. MoE (Mixture of Experts) 概念

**核心**：每层 FFN 拆成多个 expert，每 token 只激活 top-K：

```
传统 dense FFN:
  hidden → 1 个 FFN → output
  每 token 算 100% FFN 参数

MoE FFN:
  hidden → Router (gate 网络) → 选 top-2 expert
                                  ↓
              Expert 1 / 2 / ... / 256
  每 token 算 2/256 = 0.8% FFN 参数
```

**经济性**：
- DeepSeek-V3：**671B 总参 / 37B activated** → 容量 ~671B、算力 ~37B
- Qwen3-235B-A22B：235B / 22B
- Mixtral 8x7B：47B / 13B

**训练挑战**：
- Expert Parallelism (EP) → expert 跨卡分布
- AllToAll 通信（不是 AllReduce）
- Load balance loss → 防止 expert collapse
- LoRA on MoE：哪些 expert 加？给 router 加？tooling 不成熟

**我们能跑的尺寸**：
- DeepSeek-MoE-16B / Qwen3-30B-A3B → ✅ ZeRO-3 8 卡能跑（教科书 sweet spot）
- Mixtral-8x7B → ⚠️ 紧
- 235B+ → ❌ 不够

**面试官问 → 答**：
- 「MoE 优势」→ 「大模型容量、小模型算力」，DeepSeek-V3 用 1/10 训练成本对标 GPT-4o
- 「MoE 训练特殊在哪」→ 引入 expert parallelism + AllToAll 通信，需 monitor router collapse
- 「MoE 推理 vs Dense 区别」→ 总显存仍需装全部 expert（serving 难），但 active params 少（latency 优）

---

## § 8. 训练数据集生态（2026）

| 数据集 | 规模 | 来源 | 质量 |
|---|---|---|---|
| **ShareGPT-zh / ShareGPT** | ~90k | 真实用户 + GPT-4 多轮 | ⭐⭐⭐⭐ |
| **OpenHermes-2.5** | 1M | 多源混合 + 清洗 | ⭐⭐⭐⭐ |
| **COIG-PC / COIG-CQIA** | 几十万 | 中科院清洗 | ⭐⭐⭐⭐ |
| **Belle-2M** | 2M | 中文社区 | ⭐⭐⭐ |
| **Firefly-train-1.1M** | 1.1M | 多任务混合 | ⭐⭐⭐ |
| **alpaca-gpt4-data-zh**（我们用的） | 48k | GPT-4 一次性生成 | ⭐⭐（**入门标配**） |
| alpaca-zh（原版） | 52k | text-davinci-003 | ⭐ 老旧 |

**简历 caveat（重要）**：
- ❌ 不能写「我训练的模型质量提升 X%」（5k 数据 + LoRA 不可能提升）
- ✅ 能写「跑通训练 pipeline + 实测吞吐 / GPU 利用率 / 通信开销，验证工程可行性」

---

## § 9. 训练 vs 推理 资源需求对比

```
训练单卡显存需求（同一 7B BF16 模型）:
  推理：  weight 14GB + KV cache 5-15GB ≈ 20-30GB → 单 A30 24GB 紧
  Full FT：weight 14GB + grad 14GB + opt 28GB + activation = 60GB+ → 单 A30 装不下
  LoRA SFT：weight 14GB + LoRA 50MB + grad+opt(LoRA) 100MB + activation = ~20GB → 单 A30 可跑

训练比推理对通信更敏感:
  推理 TP=8 PCIe AR 开销占 wall time ~5%
  训练 ZeRO-3 PCIe AllGather+ReduceScatter 占 ~30-50%
```

---

## § 10. 速记记忆点（背下来）

| 数 / 公式 | 含义 | 例子 |
|---|---|---|
| `4 KB / token / layer` | Qwen 70B GQA KV (FP16) | 8 kv_heads × 128 dim × 2 bytes × 2(K+V) |
| `320 KB / token` | Qwen 70B 全模型 KV | 4KB × 80 层 |
| `2 × 训练参数 = optimizer` | AdamW momentum + variance | 7B → 28GB opt |
| `weight ≈ params × bytes` | 模型基础显存 | 7B BF16 = 14GB |
| `Full FT = weight + grad + opt + activation` | ≈ 4× weight 显存 | 7B → 56GB+ |
| `LoRA = weight + tiny_adapter` | ≈ 1.5× weight 显存 | 7B → 20GB |
| `rank 16 / alpha 32` | LoRA 典型起点 | 适配器 ~50MB on 7B |
| `--max_seq_length 1024` | smoke / 训练序列长度 | 越长显存越大 |
| `grad_accum 4 + per_dev_batch 1` | 等效 batch=4 | accum 越大显存越小但慢 |
