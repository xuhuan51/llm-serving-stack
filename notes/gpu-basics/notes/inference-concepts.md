# 核心推理概念

GPU 架构、注意力数学，以及 LLM serving 的三大关键优化。

---

## GPU 执行层次（含内存作用域）

| 层次 | 可用内存 | 通信范围 |
|------|---------|---------|
| thread（线程） | 私有寄存器 | 只有自己 |
| warp（32线程） | 寄存器 + warp shuffle | 同一 warp 内线程可快速交换寄存器值 |
| block（块） | 寄存器 + shared memory | 同一 block 内线程通过 shared memory 通信 |
| grid（网格） | global memory / HBM | 不同 block 主要通过 global memory 通信（慢） |

关键点：**shared memory 由程序员显式管理**（不像 CPU L1 cache 是硬件管理的）。这是 GPU kernel 优化的核心手段。

内存速度层次：`寄存器 → shared memory → L2 → HBM（global memory）`

### 访存密集 vs 计算密集

- **向量加法**：读两个值，加法，写回 — 无复用，永远是访存密集
- **矩阵乘法（朴素）**：每个线程独立从 HBM 反复读取重叠的行/列 → 访存密集
- **矩阵乘法（分块 tiling）**：一个 block 协作把一块数据从 HBM 加载到 shared memory，所有线程复用 → 趋近计算密集

数字举例：`1024×1024` 矩阵乘，`16×16` 分块 → 每次 HBM 加载被约 16 个输出元素共享 → 比朴素方法减少约 16 倍 HBM 访问。

### 和 LLM 推理的关系

- **批处理**：一次读权重，给多个 token 同时用，摊销 HBM 权重读取成本
- **KV cache**：存历史 K/V，避免重复计算
- **AWQ 量化**：INT4 权重 → 每次 decode 从 HBM 读取的字节更少
- **FlashAttention**：避免把完整的 `[seq×seq]` 注意力矩阵写到 HBM
- **CUDA Graph**：把重复的 decode kernel 序列录制成图，消除反复的 CPU kernel 启动开销

---

## Transformer 与 KV Cache（仅解码器）

- 解码器模型（Qwen/GPT 风格）：`prefill` 一次性处理所有 prompt token；`decode` 每次生成一个 token
- 每个 Transformer 层用当前层的输入 hidden state 各自计算 Q、K、V
- **Q**：当前步骤用完就丢
- **K/V**：当前步骤用，同时缓存 — 未来每一步 decode 都要 attend 到历史 K/V
- KV cache 按层存储，因为每层的 hidden state 和 W_K/W_V 权重各不相同

### Prefill vs Decode 对比

| 阶段 | 做什么 | 计算特征 |
|------|-------|---------|
| prefill | 所有 prompt token 经过所有层，K/V 写入 cache | 计算密集，token 多 |
| decode | 一个 token 经过所有层，追加 K/V，attend 全部历史 K/V | 访存密集（读取所有历史 K/V） |

---

## 注意力数学

符号：`seq` = 序列长度，`d_model` = 隐层维度，`h` = head 数，`d_k = d_model/h`。

```
Q = X × W_Q    [seq × d_k]
K = X × W_K    [seq × d_k]
V = X × W_V    [seq × d_v]

Attention(Q,K,V) = softmax(Q × K^T / sqrt(d_k)) × V
```

- `Q × K^T` → `[seq × seq]` 的相关性得分矩阵
- 除以 `sqrt(d_k)` 防止 softmax 过于极端
- softmax 把每行归一化为注意力权重
- 乘以 V 按权重混合内容

**多头注意力**：同一个 X 输入 h 套独立的 W_Q/W_K/W_V，每个 head 独立做注意力，输出拼接后经 W_O 投影。不等价于单头 — 每个 head 有独立的学习参数。

---

## FlashAttention

**朴素注意力的问题**：
1. 把完整的 `[seq×seq]` 得分矩阵写到 HBM → 内存二次方增长，大量 HBM 读写
2. 对中间结果反复读写 HBM

**核心思路（在线 softmax）**：分块处理 K/V，维护一个运行中的输出累加器 O 和 softmax 归一化因子。新块到来时，把旧贡献乘以 `d_old/d_new` 后再加入新贡献。

类比：增量计算班级平均分 — 新同学加入时，把旧平均分乘以 `旧人数/新人数`，再加入新贡献。

**结果**：完整的 `[seq×seq]` 矩阵从不写入 HBM，中间结果留在 shared memory / 寄存器里。

学员总结：**"中间结果存着，不写回"**

---

## PagedAttention

**朴素 KV cache 的问题**：按 `max_seq_len` 预分配连续显存 → 实际利用率只有 20–40%（碎片化 + 生成长度不确定）。

**思路（借鉴操作系统虚拟内存）**：
- 固定大小的 KV block（如每块 16 个 token）
- block table 映射逻辑 token 位置 → 物理 GPU 显存块
- 物理块全局不连续，但每块内部连续（保留 GPU coalesced 访问特性）
- 特定注意力 kernel 按 block table 读取 K/V

**好处**：
- 显存利用率提升到 ~90%+ → 支持更多并发用户，更高吞吐
- 天然支持前缀共享：多个请求可指向同一物理前缀块（SGLang Radix Cache 的基础）

注：PagedAttention 来自 vLLM 论文，现已成为 SGLang、TensorRT-LLM、TGI 等的事实标准。

---

## Continuous Batching（连续批处理）

**静态批处理的问题**：整个 batch 等最长的请求，早完成的请求白白占位。

**思路**：每个 decode 步骤重新组 batch。完成的请求立刻离开，排队的请求尽快加入。

**实现细节**：
- 重组 batch 的开销：主要在 CPU 侧，约几十微秒；GPU decode 步骤约毫秒级 → 调度开销通常 < 1%
- KV cache 不随请求进出而移动：PagedAttention 只改 block table 元数据，不搬 K/V tensor
- Prefill vs decode 特征不同：prefill 计算密集、token 多；decode 访存密集、逐步生成
- Chunked prefill：把长 prompt 分块，与 decode 请求交叉调度，避免 prefill 独占 GPU

**实际代价**：加入 prefill 会延长一个调度步骤；活跃 batch 很大时 per-step 时间增加 → 解释了高并发时 P95 延迟升高的现象。

---

## 三大优化综合

```
FlashAttention      → 减少注意力计算中的 HBM 读写
PagedAttention      → 改善 KV cache 显存管理和共享
Continuous Batching → 改善多请求调度，保持 GPU 满载
```

三者共同解释了为什么 vLLM 风格的服务比朴素模型推理快 10–100 倍。

---

## CUDA Graph

- **GPU kernel**：运行在 GPU 上的并行计算函数（矩阵乘、注意力、softmax、AWQ 融合反量化+矩阵乘等）
- **Kernel 启动**：CPU 告诉 GPU 执行一个 kernel，有开销（~微秒级），即使 GPU 本身很快
- **CUDA Graph**：把重复的 kernel 启动序列录制成图，以图的形式回放 → 消除反复的 CPU 启动开销
- Decode 最适合 CUDA Graph：每生成一个 token 都重复同样的 transformer forward 路径
- 关闭 CUDA Graph 会明显降低长输出吞吐（我们的 SGLang 实验中约 3–6%）

---

## 量化（AWQ/Marlin）

AWQ **不是**先把 INT4 权重还原成 BF16 再计算。Marlin kernel 直接读 INT4 权重，做**融合反量化 + 矩阵乘**：
- Decode 阶段权重读取是主要瓶颈（访存密集） → 读更少字节的代价远低于额外的反量化计算
- AWQ group 量化：每组（如 128 个权重）存一组 scale 和 zero-point，在反量化时使用
- "量化一定更快"是错的 — 取决于 kernel 支持、batch size、预热/编译效果和质量要求
