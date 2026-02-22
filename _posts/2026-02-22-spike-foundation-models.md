---
layout: post
title: "Spike Neural Foundation Models"
title_cn: "Spike Neural Foundation Models：Tokenization、Embedding 与 Loss 的技术全景"
title_en: "Spike Neural Foundation Models: A Complete Technical Survey of Tokenization, Embedding, and Loss"
date: 2026-02-22
bilingual: true
math: true
---

<div class="lang-cn" markdown="1">

> 当我们尝试用 Transformer 理解大脑的"语言"时，第一个问题就是——神经元的 spike 该怎么变成 token？

## 1. 为什么我们需要 Spike 数据的基础模型？

过去十年，基础模型（Foundation Models）在 NLP 和计算机视觉领域取得了巨大成功。GPT 系列证明了"大规模预训练 + 下游微调"范式的强大，CLIP 和 DINOv2 则展示了视觉表示学习的潜力。一个自然的问题随之而来：**我们能不能为大脑的神经活动数据做同样的事情？**

神经科学中，大规模电生理记录技术（如 Utah Array、Neuropixels）让我们能够同时记录数百甚至上千个神经元的 spiking 活动。这些数据蕴含着大脑计算的核心信息，广泛应用于脑机接口（BCI）、运动解码、视觉感知研究等领域。然而，传统方法（如 LFADS [[1]](#ref1)）往往针对单个 session 从头训练，无法跨 session、跨被试复用知识。

从 2021 年的 NDT1 [[2]](#ref2) 到 2025 年的 NDT3 [[6]](#ref6)、NEDS [[8]](#ref8) 和 SPINT [[10]](#ref10)，一系列工作尝试将 Transformer 架构引入神经 spiking 数据，逐步走向"神经数据基础模型"。这些工作面对的核心问题都一样：**spike 数据该怎么 tokenize？用什么 embedding 表示？训练目标（loss）该怎么设计？**

这篇文章将系统梳理这些技术选择，分析它们背后的动机、优势与局限，并探讨未来的发展方向。

---

## 2. Spike 数据面临的核心挑战

在理解各种技术方案之前，我们首先需要理解 spike 数据为什么"难"。与文本或图像不同，spike 数据有几个独特的、相互交织的挑战。

### 2.1 神经元对应问题（Neuron Correspondence Problem）

这是 spike 基础模型面临的最根本挑战。

在 NLP 中，"cat" 这个 token 在所有文本中都代表猫；在视觉中，像素坐标 (100, 200) 在所有图像中代表同一个空间位置。但在 spiking 数据中，**"通道 3"在不同 session 中可能记录的是完全不同的神经元**。

即使是同一个被试，电极漂移也会导致神经元在 session 间出现和消失。BrainGate 临床数据显示，电极阵列平均仅在 35.6% 的电极上记录到 spiking，且在 7.6 年的慢性植入期间每年下降约 7% [[6]](#ref6)。这意味着 spike 数据天然**缺乏跨 session 的"共享词汇表"**。

| 模态 | 标准化方案 | 对应难度 |
|------|-----------|---------|
| EEG  | 10-20 系统标准化电极安放 | 低 |
| fMRI | MNI 模板空间标准化 | 低 |
| Spiking | 无标准——每次植入捕获独特神经元集合 | **极高** |

更具体地说，跨 session 迁移存在三个递进的难度层级：

- **同被试、电极漂移**：同一被试不同天的记录，约 50-80% 的神经元重叠。群体功能结构基本不变，只是部分成员轮换。
- **跨被试、同脑区**：不同个体的同一脑区（如运动皮层 M1），虽然计算逻辑相似，但单个神经元的 tuning、连接模式、发放率分布完全不同。
- **跨脑区**：不同脑区（如 V1 → M1）的计算逻辑根本不同，是最难的场景。

#### 现有解决方案与潜在方向

目前，各模型对神经元对应问题采取了不同策略，按照从简单到复杂可以分为以下几类：

**方案 A：固定维度编码（NDT1 [[2]](#ref2)）。** 最简单的方案——线性投影 $$W_{in} \in \mathbb{R}^{D \times N}$$ 将每个神经元硬编码到 embedding 空间的固定方向。换一个 session，维度含义就完全改变，无法跨 session 迁移。

**方案 B：Learnable Unit Embedding（POYO [[4]](#ref4)）。** 为每个神经元分配可学习 embedding 向量 $$e_n \in \mathbb{R}^D$$。新 session 需要冻结主干网络，通过梯度下降更新这些 embedding。优点是显式建模了神经元身份；缺点是需要有标签校准数据和梯度更新。

**方案 C：Context-Dependent Positional Embedding / IDEncoder（SPINT [[10]](#ref10)）。** 通过共享 MLP 网络从无标签校准数据中动态推断每个 unit 的 identity embedding，并将其作为**上下文依赖的位置编码**添加到 spike token 上（详见 [Section 3.3](#33-neuron-as-token以神经元为-token) 和 [Section 4.2](#42-神经元身份编码)）。这是目前最优雅的解决方案，实现了零梯度跨 session 迁移。

**潜在方向一：将 POYO 的 Learnable Unit Embedding 扩展为前向推断。** POYO 当前为每个 unit 分配一个独立的可学习 embedding，新 session 需要梯度更新。一种自然的扩展是借鉴 SPINT 的 IDEncoder 思路——不再为每个 unit 维护独立 embedding，而是通过一个共享的前馈网络（feedforward network）**直接从 unit 的原始校准数据前向推断出 unit embedding**。具体来说，类似 SPINT 的 IDEncoder，将每个 unit 的 $$M$$ 条校准 trial 的 binned spike counts $$X_n^{calib} \in \mathbb{R}^{M \times T}$$ 直接送入网络，而非手动提取统计特征：

$$e_n = \psi\left(\frac{1}{M} \sum_{j=1}^{M} \phi(X_{n,j}^{calib})\right)$$

其中 $$\phi$$ 和 $$\psi$$ 是共享的多层前馈网络，$$X_{n,j}^{calib}$$ 是 unit $$n$$ 第 $$j$$ 条校准 trial 的原始 binned spike counts。这种端到端的方式让网络自己从原始数据中学习提取有意义的身份特征，完全避免了手动设计统计特征（如发放率分布、ISI 统计等）带来的信息瓶颈和归纳偏置。这相当于将 SPINT 的 IDEncoder 模块嫁接到 POYO 的 PerceiverIO 架构中，使其同时具备 POYO 的 spike-level 时间精度和 SPINT 的零梯度适应能力。

这种方案的**优势**在于：(1) 完全数据驱动，网络可以自动发现最有区分性的 unit 特征模式，而非依赖人为设计的统计量；(2) 与 POYO 现有架构兼容性好，只需将 `InfiniteVocabEmbedding` 替换为 IDEncoder 模块；(3) 端到端训练使 identity embedding 直接针对下游解码任务优化。**局限**在于：(1) 推断质量仍高度依赖校准数据的代表性——如果校准 trial 过少或未覆盖足够多的行为状态，学到的 identity 可能不够稳定；(2) 前馈网络的表达能力有限，可能难以捕捉需要群体上下文才能区分的 unit 特性（例如两个发放模式相似但功能角色不同的 unit）。

**潜在方向二：对比学习方案学习 Unit Embedding。** 受 NuCLR [[11]](#ref11) 的启发，可以采用对比学习的范式自监督地学习每个 unit 的表示。NuCLR 的核心思想是：同一个神经元在不同时间窗口和不同刺激条件下的活动，应该产生相似的表示（正样本对）；不同神经元的活动则应产生不同的表示（负样本对）。具体来说，NuCLR 使用一个**置换等变的时空 Transformer**（permutation-equivariant spatiotemporal transformer）来整合群体活动上下文，并通过对比目标拉近同一 neuron 不同 view 的表示、推远不同 neuron 的表示。学到的 neuron-level embedding 可以用于下游任务（如细胞类型分类、脑区识别），且展示了跨个体的零样本泛化能力。

将这一思路引入 spike foundation model，可以设想如下框架：在预训练阶段，对每个 unit 的不同时间段活动提取多个 view，通过对比学习得到稳定的 unit representation；这些 representation 随后作为 unit embedding 注入到 POYO 或 SPINT 风格的解码器中。这种方案的**优势**在于：(1) 完全自监督，不需要行为标签；(2) 学到的表示反映了神经元的内在功能特性而非仅仅是统计摘要；(3) 对比学习天然鼓励表示的区分性（discriminative），有助于区分功能相似但不同的 unit。**局限**在于：(1) 对比学习对数据增强策略敏感，需要设计适合 spike 数据的增强方案（如时间裁剪、随机 dropout 神经元子集等，STNDT [[3]](#ref3) 已有初步探索）；(2) 训练开销较大，需要同时处理大量 unit 的多个 view；(3) 从"好的 unit representation"到"好的解码性能"之间还存在 gap，需要验证学到的对比表示是否对下游任务真正有用。

**两种方案的对比：**

| 维度 | 前馈推断（SPINT 风格） | 对比学习（NuCLR 风格） |
|------|---------------------|---------------------|
| 监督信号 | 端到端（通过下游任务） | 自监督（对比目标） |
| 校准需求 | 少量无标签数据 | 大量无标签数据（预训练） |
| 新 session 适应 | 一次前向传播 | 前向传播（若冻结 encoder） |
| 表示质量 | 数据驱动（原始 spike counts） | 数据驱动（对比目标），可能更丰富 |
| 计算开销 | 低 | 预训练阶段高 |
| 成熟度 | 已验证（SPINT） | 概念验证阶段（NuCLR） |

最理想的方案可能是二者的结合：先用对比学习在大规模数据上预训练一个通用的 unit encoder，再在具体下游任务中用端到端的前馈推断进行微调。这样既利用了对比学习的自监督优势，又保留了端到端优化的任务适应性。

### 2.2 极端稀疏性（Extreme Sparsity）

典型皮层神经元的发放率仅 1-20 Hz。这意味着在 1ms 分辨率下，99% 以上的时间 bin 是空的。这种极端稀疏性对技术方案有深远影响：

- **Tokenization 层面**：大量全零 token 是否有意义？对 binned 方法来说，大部分 token 携带的信息接近于零。
- **Loss 设计层面**：Poisson NLL 在低 count（0-1 spikes/bin）时梯度信号很弱，模型容易学到一个"总是预测零"的 trivial 解。
- **自监督预训练层面**：如果 masked token 重建在大部分位置上都等价于"预测零"，模型从中能学到的有用信号就很有限。

### 2.3 时间分辨率与效率的权衡（Temporal Resolution vs. Efficiency）

Spike 数据的一个关键优势是毫秒级时间精度——视觉皮层中 onset latency 的差异可能只有几毫秒，对刺激编码至关重要。但保留这种精度意味着极长的序列（1 秒 = 1000 个 1ms bins），这与 Transformer 的二次复杂度直接冲突。

实际应用还有额外的延迟约束：

| 应用 | 延迟要求 | 挑战 |
|------|---------|------|
| 光标控制 BCI | <100ms | Transformer 二次复杂度 |
| 语音解码 | ~200ms | 长上下文处理 |
| 图像重建 | ~500ms | 多模态融合 |

这迫使设计者在"保留信息"和"计算可行"之间做出取舍。

### 2.4 数据异质性与规模限制（Data Heterogeneity & Scale）

与 NLP/视觉领域相比，spiking 数据集小了好几个数量级：

| 领域 | 典型训练规模 |
|------|------------|
| 语言模型 | 万亿 tokens |
| 视觉模型 | 数十亿图像 |
| 神经 spiking | ~10 亿 tokens（NDT3 [[6]](#ref6), 2000 小时）|

更棘手的是，即使有限的数据也高度异质——不同实验室使用不同的记录设备、处理算法、实验范式和行为任务。NDT3 的作者发现，精选 5 个 session 的数据可能比使用全部 84 个 session 的效果更好，因为 session 间的脑区不匹配会降低性能。这表明暴力数据积累会失败，领域特定的数据策划至关重要。

### 2.5 多时间尺度的非平稳性（Multi-Scale Non-Stationarity）

神经信号不是静态的——它在多个时间尺度上变化：

| 时间尺度 | 变化来源 | 影响 |
|---------|---------|------|
| 分钟-小时 | 适应、疲劳、觉醒 | 发放率波动 |
| 天-周 | 电极漂移、组织变化 | 神经元丢失/出现 |
| 月-年 | 慢性退化、学习可塑性 | 信号质量下降 |

好消息是，"稳定流形假设"（Gallego et al., 2020 [[12]](#ref12)）表明群体级动力学可能比单个神经元活动更稳定——这激发了很多工作从群体层面而非单神经元层面学习表示。

---

## 3. Tokenization：如何把 Spike 变成 Token？

Tokenization 是所有基础模型的第一步，也是影响最深远的设计选择。对 spike 数据来说，核心问题是：**以什么粒度（时间×空间）把连续的神经活动切分成离散的 token？**

目前主要有四种 tokenization 范式，每种都有截然不同的取舍。

### 3.1 Binned 群体向量（Population Vector per Time Bin）

**代表工作：** NDT1 [[2]](#ref2), NDT2 [[5]](#ref5), NDT3 [[6]](#ref6), MtM [[7]](#ref7), NEDS [[8]](#ref8)

这是最直接的方案：以固定时间窗口（通常 20ms）对 spike train 做 binning，统计每个神经元在每个 bin 内的 spike count，然后将每个时间步的完整群体向量（N 维，N = 神经元数量）通过投影映射为一个 token。一个持续 1 秒的 trial 在 20ms binning 下产生 50 个 token。

```
原始 spike trains → 20ms binning → N×T 矩阵 → 每列投影 → T 个 tokens
```

**NDT1 的两种 Embedding 模式：** NDT1 实际支持两种 spike embedding 模式。**模式一**是线性投影：

$$\mathbf{h}_t = W_{in} \cdot \mathbf{x}_t + b, \quad W_{in} \in \mathbb{R}^{D \times N}$$

其中 $$\mathbf{x}_t \in \mathbb{R}^N$$ 是时间步 $$t$$ 的群体 spike count 向量。**模式二**是 per-neuron embedding——将每个神经元的 spike count 视为离散变量，通过 `nn.Embedding` 查表映射为向量后拼接：

$$\mathbf{h}_t = [E(x_{t,1}) \| E(x_{t,2}) \| \cdots \| E(x_{t,N})], \quad E: \{0,1,...,\text{max_spikes}\} \to \mathbb{R}^{d}$$

后一种模式将 spike count 视为离散分类变量而非连续值，与后来 NDT3 的离散化思路有内在联系。

**NDT2 的改进——时空 Patch：** NDT2 [[5]](#ref5) 在此基础上引入了 ViT 风格的 spatial patching。不再将全部 N 个神经元作为一个 token，而是将它们分成 N/K 组（默认 **K=8**，即每个 patch 包含 8 个神经元），每组在每个时间步产生一个 token。NDT2 支持多种 readin 策略，包括 per-neuron linear projection、embedding lookup 和 cross-attention readin 等。此外还支持 array embedding（用于多电极阵列场景）。这部分解耦了神经元身份——即使两个 session 的神经元不完全相同，只要 patch 的统计特性相似，patch-level 的 token 表示就可能泛化。

**NDT3 的进一步改进——离散化 + 多模态打包：** NDT3 [[6]](#ref6) 继承了 NDT2 的 patch tokenization（默认 **K=32**），并将连续 spike count 通过 `torch.bucketize` 离散化为分类变量，配合 Poisson-softened cross-entropy loss（详见 [Section 5.1](#51-重建预测-loss)）。此外，NDT3 将多种模态的 token（spike、constraint、return、covariate）通过 **space offset** 打包到单一序列中，按 `(timestep, space_offset)` 排序形成扁平化的多模态序列——这是 NDT3 区别于前代的重要架构特征。

**优势：**
- 实现简单，序列长度固定且可预测（T = duration / bin_size）
- 直接兼容标准 Transformer 架构，无需特殊处理
- 是最成熟、实验验证最充分的方案

**劣势：**
- **稀疏性问题严重**：低发放率区域大量 token 近乎全零，浪费计算资源且提供的梯度信号很弱
- **时间信息丢失**：20ms 内的精细时间结构被不可逆地抹掉，对视觉皮层等对 timing 敏感的应用可能是致命的
- **神经元对应最不友好**：群体向量的每个维度硬编码了一个特定神经元的位置，跨 session 时神经元变化直接破坏输入结构

### 3.2 单 Spike Tokenization（Per-Spike Token）

**代表工作：** POYO [[4]](#ref4), POYO+ [[9]](#ref9), POSSM [[13]](#ref13)

POYO 开创了一种对 spike 数据最"原生"的表示：**每个单独的 spike 事件成为一个 token**，完全不做时间 binning。

具体来说，每个 spike token 由三部分信息组成：一个 learnable unit embedding（标识该 spike 来自哪个神经元）、一个 token type embedding（标识 token 类型，如 spike、start/end 标记等），加上通过 Rotary Position Embeddings (RoPE) 编码的精确连续时间戳。假设有 100 个神经元，每个平均 10Hz，1 秒内就产生约 1000 个 token——序列长度直接正比于实际 spike 数量。

数学上，每个 input spike token 的构造为：

$$\mathbf{h}_i^{input} = E_{unit}(\text{unit_id}_i) + E_{type}(\text{token_type}_i)$$

其中 $$E_{unit}$$ 使用 `InfiniteVocabEmbedding`（一种支持动态词汇表扩展的 learnable embedding，新 session 的新 unit 可以动态注册），$$E_{type}$$ 是 4 种 token 类型的 embedding。时间信息则通过 RoPE 在 attention 计算时注入（详见 [Section 4.1](#41-时间位置编码)）。

由于序列长度随 spike 数量增长，POYO 搭配了 **PerceiverIO 架构**作为压缩机制：通过 cross-attention 将 variable-length 的 spike token 序列压缩到固定数量（如 256 个）的 latent token，后续的 self-attention 只在这些 latent token 上进行。整个流程分为三个阶段：

1. **Encode**：latent tokens 通过 cross-attention 聚合 input spike tokens 的信息
2. **Process**：latent tokens 之间进行 self-attention（2-6 层）
3. **Decode**：output queries 通过 cross-attention 从 latent tokens 中提取预测所需的信息

值得注意的是，POYO 的 decoder 端是 **session-aware** 的——使用 `session_emb` 构造 output query embedding，不同 session 使用不同的 query。

**CaPOYO 的钙成像扩展：** POYO+ 通过独立的 CaPOYO 模型类支持钙成像数据。CaPOYO 采用 **split-dim 拼接设计**显式解耦信号值和单元身份：

$$\mathbf{h}_i = [\underbrace{W_{val} \cdot \Delta F/F_i + b_{val}}_{\in \mathbb{R}^{D/2}} \; \| \; \underbrace{E_{unit}(\text{unit_id}_i)}_{\in \mathbb{R}^{D/2}}]$$

与 spike token 不同（spike 隐含值为 1），钙成像 token 需要同时编码连续的荧光信号值和 unit 身份。POYO+ 还新增了 `task_emb` 支持多任务解码（如速度解码、位置解码等）。

**优势：**
- **完美处理稀疏性**：序列长度正比于 spike 数量而非时间长度，空区间零计算开销
- **保留毫秒级时间精度**：RoPE 编码连续时间戳，无信息损失
- **PerceiverIO 压缩**优雅地处理了不同 session 间神经元数量不同的问题

**劣势：**
- 高发放率群体的序列仍然较长，input cross-attention 有计算开销
- Unit embedding 需要 per-neuron 学习，新 session 需有标签校准数据通过梯度下降重学习
- 不直接编码脑区或任务信息，需要额外 context embedding 处理异质性

### 3.3 Neuron-as-Token（以神经元为 Token）

**代表工作：** STNDT [[3]](#ref3), SPINT [[10]](#ref10)

这种方案翻转了视角：不是按时间步切分，而是**将每个神经元的完整时间序列作为一个 spatial token**。

**STNDT 的双流设计：** STNDT 同时构造两种视图——temporal tokens（每时间步的群体向量，$$[T, B, N]$$）和 spatial tokens（每个神经元的时间序列，转置为 $$[N, B, T]$$），通过独立的 attention 机制处理后融合。两个 stream 各有独立的线性 embedder 和正弦位置编码。Spatial attention 通过矩阵乘法重新加权 temporal 特征：

$$Z_{ST} = A_S \cdot Z_T^\top$$

其中 $$A_S \in \mathbb{R}^{B \times N \times N}$$ 是 spatial attention 的权重矩阵（softmax 后），$$Z_T \in \mathbb{R}^{T \times B \times N}$$ 是 temporal representation。融合后的 $$Z_{ST}$$ 经过残差连接和 FFN，让模型学习"哪些神经元应该一起被考虑"。

**SPINT 的核心创新——IDEncoder 动态位置编码：** SPINT 将每个 neural unit 的 $$W$$ 个时间 bin 的 binned spike counts 构成一个 spatial token，配合其核心创新——**IDEncoder 上下文依赖的位置编码**。

SPINT 的 IDEncoder 不使用任何固定位置编码（这会假设神经元有固定顺序），而是从校准数据中动态推断每个 unit 的 identity，并将其**作为位置编码添加到 spike 活动上**。具体过程如下：

1. **输入**：收集 unit $$i$$ 的 $$M$$ 条校准 trial 数据 $$X_i^C \in \mathbb{R}^{M \times T}$$（每条 trial 插值到固定长度 $$T$$，如 M1/H1 使用 $$T=1024$$）
2. **逐 trial 编码**：通过共享的三层 MLP $$\phi$$ 处理每条 trial
3. **跨 trial 聚合**：对所有 trial 的表示取均值池化
4. **身份生成**：通过第二个三层 MLP $$\psi$$ 生成最终的 identity embedding

数学上：

$$E_i = \text{IDEncoder}(X_i^C) = \psi\left(\frac{1}{M} \sum_{j=1}^{M} \phi(X_{i,j}^C)\right)$$

其中 $$\phi: \mathbb{R}^T \to \mathbb{R}^H$$ 和 $$\psi: \mathbb{R}^H \to \mathbb{R}^W$$ 分别是两个三层全连接网络，$$H$$ 为隐藏维度（M1: $$H=1024$$; M2: $$H=512$$; H1: $$H=1024$$），$$W$$ 为窗口大小（对应 spike token 的维度）。

**关键步骤——Identity Embedding 作为位置编码注入：** 生成的 $$E_i$$ 被**直接加到每个 unit 的 spike 活动窗口**上：

$$Z_i = X_i + E_i$$

这里 $$X_i$$ 是 unit $$i$$ 当前解码窗口的 binned spike counts，$$Z_i$$ 是 identity-aware 的表示。注意 $$E_i$$ 在同一 session 内对所有时间窗口保持不变——它编码的是 unit 的**稳定身份**（类似传统 Transformer 中位置编码编码的是 token 的位置），而 $$X_i$$ 携带的是**时变活动**。这种加法注入方式使得 $$Z_i$$ 同时包含了"谁在发放"（identity）和"发放了什么"（activity）的信息。

随后，$$Z_i$$ 通过 MLP 投影到 cross-attention 的输入空间，由**可学习的行为查询矩阵** $$Q \in \mathbb{R}^{B \times W}$$ 通过单层 cross-attention 解码出行为预测：

$$\hat{Y}_t = \text{MLP}_{out}(\text{CrossAttn}(Q, \text{LN}(Z_{in}), \text{LN}(Z_{in})))$$

整个架构在数学上保证了**置换不变性**：

$$\text{CrossAttn}(Q, P_R Z, P_R Z) = \text{CrossAttn}(Q, Z, Z)$$

其中 $$P_R$$ 是任意行置换矩阵。无论神经元排序如何，输出完全相同。此外，SPINT 采用**动态通道 dropout**（dynamic channel dropout）来增强对不同 session 间神经元组成变化的鲁棒性。

**跨 session 迁移零梯度：** 对于未见 session，只需在校准数据上运行训练好的 IDEncoder 前向传播，即可推断出所有 unit 的 identity embedding——无需梯度更新、无需标签数据。

**优势：**
- **SPINT 的置换不变性**是神经元对应问题目前最优雅的解决方案
- Spatial attention（STNDT）能发现功能重要的神经元子集
- IDEncoder 实现零梯度跨 session 迁移
- 轻量设计（单层 cross-attention + 两个三层 MLP），适合实时 BCI

**劣势：**
- Spatial attention 在神经元数量 N 上有 $$O(N^2)$$ 复杂度，大规模记录可能成为瓶颈
- 底层仍依赖 binning，损失了精细时间信息
- 目前仅在较小规模上验证

### 3.4 Spike 事件对（Spike Event Pairs）

**代表工作：** Neuroformer [[14]](#ref14)

Neuroformer 采用了最接近 NLP 的方案：将每个 spike event 编码为 **(neuron_id, time_interval_offset)** 二元组，类似于句子中的"词"。

```
时间窗口(50ms 当前窗口 + 150ms 历史窗口)内的所有 spikes → 按发放顺序排列 →
[(neuron_3, offset_2), (neuron_7, offset_5), ...] → 类似"句子"
```

每个 spike token 的 embedding 由三部分加法合成：

$$\mathbf{h}_i = E_{tok}(\text{neuron_id}_i) + E_{pos}(i) + E_{temp}(\Delta t_i)$$

其中 $$E_{tok}$$ 是 neuron ID 的 embedding table（`nn.Embedding`），$$E_{pos}$$ 是 learnable position embedding（编码序列内位置索引），$$E_{temp}$$ 默认是 **sinusoidal temporal embedding**（编码连续时间偏移值 $$\Delta t$$，而非 learnable embedding）。也可选配 learnable temporal embedding，但代码默认使用正弦编码。

Neuroformer 的完整架构是一个**多模态系统**，包含：neural token embedding stem（即上述 spike 编码）、可选的 visual backbone（VideoEncoder/ResNet3D/ViT）、MultimodalTransformer（处理 neural-visual 跨模态 attention）、CLIP 模块（可选的跨模态对比学习）、以及独立的 head_id（预测下一个 neuron ID）和 head_dt（预测时间偏移）预测头。

**优势：**
- **稀疏性处理最优**：与 POYO 一样只编码有事件发生的时刻
- **唯一具备生成能力的方案**：作为自回归语言模型，可以生成条件合成 spike trains
- **高可解释性**：attention weights 直接反映神经元间的功能耦合，论文发现 attention maps 镜像了 Hebbian 连接性

**劣势：**
- 没有 PerceiverIO 式压缩，高发放率群体计算量大（$$O(L^2)$$）
- neuron_id 是固定词汇表，跨 session 能力最弱
- 自回归逐 spike 生成推理速度很慢

### 3.5 Per-Spike Token 与 Spike Event Pairs 的联系与区别

POYO 的 Per-Spike Token 和 Neuroformer 的 Spike Event Pairs 在表面上非常相似——二者都**以单个 spike event 为基本处理单元**，避免了 binning 造成的时间信息损失和稀疏性浪费。但它们在几个关键维度上有本质区别：

**时间编码方式不同。** POYO 使用 RoPE 编码**绝对连续时间戳**（以秒为单位），时间信息通过旋转 Q/K 向量在 attention 计算时隐式注入，不修改 token embedding 本身。Neuroformer 使用**相对时间偏移**（在当前窗口内的离散 offset），通过加法显式合并到 token embedding 中。前者理论上保留了更完整的时间信息，后者更贴近自回归语言模型的传统做法。

**神经元身份编码不同。** POYO 使用动态可扩展的 `InfiniteVocabEmbedding`，新 unit 可以运行时注册；Neuroformer 使用固定大小的 `nn.Embedding` 查找表，词汇量在训练时确定。这意味着 POYO 在面对新 session 时有更大的灵活性，而 Neuroformer 的词汇表固定限制了其跨 session 能力。

**架构范式不同。** POYO 使用 PerceiverIO 将 spike tokens 压缩到固定长度的 latent space 后处理，是一个**判别式（discriminative）**解码器。Neuroformer 使用 GPT-style 自回归 decoder，是一个**生成式（generative）**模型。这决定了它们适用的下游任务类型：POYO 擅长行为解码，Neuroformer 擅长 spike pattern 生成和功能连接分析。

**训练目标不同。** POYO 直接用 MSE 做监督学习解码行为变量；Neuroformer 用 cross-entropy 做自回归 next-spike 预测，再可选配 CLIP-style 对比学习做跨模态对齐。

| 维度 | POYO (Per-Spike Token) | Neuroformer (Spike Event Pairs) |
|------|----------------------|-------------------------------|
| 时间编码 | RoPE（绝对连续时间戳） | Sinusoidal/Learnable（相对 offset） |
| 身份编码 | InfiniteVocabEmbedding（动态） | nn.Embedding（固定词汇表） |
| 序列压缩 | PerceiverIO（固定 latent） | 无压缩（$$O(L^2)$$ attention） |
| 模型范式 | 判别式解码器 | 生成式自回归 |
| 训练目标 | MSE（行为变量） | CE（next spike）+ 对比学习 |
| 跨 session | Learnable embedding + 梯度更新 | 固定词汇表，最弱 |
| 独特能力 | 多模态融合（spike + 钙成像） | Spike 生成 + 功能连接分析 |

总结来说，Per-Spike Token 和 Spike Event Pairs 可以看作同一"事件驱动"思想的两种不同实现：**POYO 优化了表示效率和跨 session 灵活性，Neuroformer 优化了生成能力和可解释性**。一个理想的未来方案可能是将 POYO 的 PerceiverIO 压缩和 InfiniteVocabEmbedding 与 Neuroformer 的自回归生成能力结合——在一个统一的事件驱动框架中同时实现高效解码和 spike 生成。

### 3.6 四种 Tokenization 的综合对比

| 维度 | Binned 群体向量 | 单 Spike | Neuron-as-Token | Spike 事件对 |
|------|----------------|----------|-----------------|-------------|
| 时间精度 | ★★☆☆☆ (20ms) | ★★★★★ (ms级) | ★★☆☆☆ (依赖bin) | ★★★★☆ (窗口内离散) |
| 稀疏性处理 | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★★★ |
| 计算效率 | ★★★★★ (固定长度) | ★★★★☆ (有压缩) | ★★★☆☆ ($$O(N^2)$$) | ★★☆☆☆ (无压缩) |
| 神经元对应 | ★★☆☆☆ | ★★★☆☆ | ★★★★★ (SPINT) | ★★☆☆☆ |
| 生成能力 | ★★★☆☆ (重建) | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |
| 可解释性 | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |

> **注：** POYO、POSSM、SPINT 等纯监督/判别式模型不以生成为目标，因此生成能力评分仅反映架构潜力而非设计意图。将 POYO 的生成能力评为 ★☆☆☆☆ 类似于评价 BERT 不擅长文本生成——技术上正确但并非其设计目标。

---

## 4. Embedding：如何赋予 Token 身份和上下文？

Tokenization 解决了"如何切分"的问题，而 Embedding 解决的是"如何表示"——特别是如何在 token 中编码神经元身份、时间位置和上下文信息。本节将详细介绍每个项目实际采用的编码方式及其注入网络的具体方式。

### 4.1 时间位置编码

时间位置编码决定了模型如何感知 token 在时间轴上的位置。各项目采用了三种主要方案：

**正弦位置编码（Sinusoidal PE）。** NDT1 [[2]](#ref2) 和 STNDT [[3]](#ref3) 使用标准的 Transformer 正弦位置编码，编码离散时间步索引：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

注入方式为**加法**：$$\mathbf{h}_t \leftarrow \mathbf{h}_t + PE(t)$$。NDT1 也支持 learnable position embedding（`nn.Embedding`）。STNDT 的两个 stream 各有独立的正弦位置编码——temporal PE 维度为 $$D_T = N$$、序列长度为 $$T$$；spatial PE 维度为 $$D_S = T$$、序列长度为 $$N$$。

Neuroformer [[14]](#ref14) 的时间编码默认也使用**正弦函数**（`TemporalEmbedding`），但编码的是**连续时间偏移值** $$\Delta t$$（而非离散索引），可选配 learnable temporal embedding。此外 Neuroformer 还有独立的 learnable position embedding（`nn.Parameter`）编码序列内位置索引。

**Learnable Position/Temporal Embedding。** MtM [[7]](#ref7) 使用 learnable position embedding（`nn.Embedding`），通过加法注入到 spike token 中。NDT3 [[6]](#ref6) 同时支持 learnable time embedding（加法注入）和 Rotary PE（在 attention 层内部注入）两种模式，以及一个 learnable **spatial embedding**（`nn.Embedding`）用于区分同一时间步内不同 spatial patch 的位置。NEDS [[8]](#ref8) 也使用 learnable temporal embedding。

**Rotary Position Embeddings (RoPE)。** POYO [[4]](#ref4)/POSSM [[13]](#ref13) 和 NDT3（可选模式）使用 RoPE。RoPE 不修改 token embedding 本身，而是在 attention 计算时旋转 key/query 向量，使 attention score 自然反映**相对**时间距离。POYO 的 RoPE 编码**连续时间戳**（以秒为单位），数学上：

$$\text{RoPE}(x_{2i-1}, x_{2i}, t) = \begin{pmatrix} x_{2i-1} \cos(\omega_i t) - x_{2i} \sin(\omega_i t) \\ x_{2i-1} \sin(\omega_i t) + x_{2i} \cos(\omega_i t) \end{pmatrix}$$

其中 $$\omega_i = 2\pi / T_i$$，$$T_i$$ 在 $$[T_{min}, T_{max}]$$ 上对数均匀分布（默认 $$T_{min}=10^{-4}$$, $$T_{max}\approx 2.06$$）。仅旋转 head 维度的一半（默认 head_dim=64 中旋转 32 维），另一半保持不变。NDT3 的 RoPE 则编码**离散时间步索引**。

**各项目时间编码方案总结：**

| 项目 | 时间编码类型 | 编码对象 | 注入方式 |
|------|------------|---------|---------|
| NDT1 | Sinusoidal PE / Learnable PE | 离散时间步索引 | 加法 |
| STNDT | Sinusoidal PE（两个独立的） | 离散时间步/神经元索引 | 加法 |
| NDT2 | 未使用显式时间编码 | — | — |
| NDT3 | Learnable time emb / Rotary PE + Learnable spatial emb | 离散时间步 + 空间位置 | 加法 / Attention 内旋转 |
| POYO/POSSM | Rotary PE | 连续时间戳（秒） | Attention 内旋转 |
| Neuroformer | Sinusoidal temporal emb（默认）+ Learnable position emb | 连续 $$\Delta t$$ + 序列索引 | 加法 |
| MtM | Learnable position emb | 离散时间步索引 | 加法 |
| NEDS | Learnable temporal emb | 离散时间步 | 加法 |

### 4.2 神经元身份编码

这是最关键的 embedding 选择，直接决定了模型的跨 session 能力。

**隐式位置编码（群体向量中的维度位置）。** NDT1 [[2]](#ref2) 的线性投影 $$W_{in} \in \mathbb{R}^{D \times N}$$ 隐式地将每个神经元映射到 embedding 空间的特定方向。第 $$i$$ 个神经元的 spike count 总是乘以 $$W_{in}$$ 的第 $$i$$ 列。这意味着神经元身份完全由输入维度的位置决定——换一个 session，维度含义就变了。

**Learnable Unit Embeddings。** POYO [[4]](#ref4)/POYO+ [[9]](#ref9) 使用 `InfiniteVocabEmbedding`，为每个 neural unit 分配可学习的 embedding 向量 $$e_n \in \mathbb{R}^D$$。支持动态词汇表扩展，新 session 的新 unit 可以运行时注册。新 session 需冻结主干，通过梯度下降重学习 embedding。CaPOYO 的 unit embedding 为半维度（$$D/2$$），与 value map 拼接。

**Neuron ID Embedding Table。** Neuroformer [[14]](#ref14) 使用固定大小的 `nn.Embedding`，将 neuron_id 映射为向量。词汇量在训练时确定，跨 session 能力受限于词汇表大小。

**Context-Dependent Positional Embedding / IDEncoder。** SPINT [[10]](#ref10) 的核心创新（详见 [Section 3.3](#33-neuron-as-token以神经元为-token)）。通过共享的双 MLP 网络从无标签校准数据动态推断 unit identity embedding $$E_i$$，并作为**位置编码**加到 spike 活动上。这些 embedding 反映了每个神经元在当前 session 中的功能角色（如发放率模式、时间相关特性等），而非固定的通道索引。

**Session/Context Tokens。** NDT2 [[5]](#ref5) 引入了 learnable session embedding、subject embedding 和 task embedding。注入方式有两种：(1) **Token 策略**：作为额外 token prepend 到序列首部，配合 flag 参数作为 type indicator；(2) **Concat 策略**：拼接到每个 token 的 embedding 后再投影。NDT3 [[6]](#ref6) 进一步加入 phase token（BCI vs. native control）和 return token（控制器质量，Decision Transformer 风格）。

**Session Embedding + Prompt Token。** MtM [[7]](#ref7) 使用 session embedding（`nn.Embedding`）和 prompt embedding（4 种 masking mode 各对应一个）。注入方式是作为**序列前缀 token**——prompt token 在第一个位置，session token 在第二个位置，使模型通过读取序列开头的 token 即可知道当前的 session 和 masking 任务类型。

**Session-Specific 投影。** NEDS [[8]](#ref8) 为每个 session 学习独立的线性投影 $$W_{neural} \in \mathbb{R}^{N_{session} \times D}$$，处理不同 session 间神经元数量不同的问题。所有 token 还添加 modality embedding 和 session embedding。

### 4.3 各项目 Embedding 注入网络流程详解

为了更清楚地理解各模型的 embedding 如何流经网络，这里给出每个项目的 embedding 注入流程图：

**NDT1：**
```
spike_counts [T, B, N]
  → embedder (Linear 或 per-neuron Embedding) → [T, B, D]
  → × sqrt(D)  (scale)
  → + Sinusoidal PE
  → Dropout
  → Transformer Encoder (BERT-style, 带 context mask)
  → Decoder (Linear → Poisson rate)
  → PoissonNLLLoss
```

**STNDT：**
```
spike_data [T, B, N]
  ├─ temporal_embedder (Linear) → [T, B, D_T] + temporal_PE → Temporal Self-Attention
  │                                                              ↓ (src)
  └─ .permute → [N, B, T]                                       │
     → spatial_embedder (Linear) → [N, B, D_S] + spatial_PE      │
     → Spatial Self-Attention → spatial_weights (A_S)             │
                                    ↓                             ↓
                          Z_ST = bmm(A_S, Z_T^T) ←───────────────┘
                                    ↓
                              残差 + FFN → output → PoissonNLL + InfoNCE
```

**NDT2：**
```
spike_data [B, T, A, C, H]
  → spatial grouping（每 K=8 个神经元一组）→ [B, T, S, K]
  → readin (Linear/Embedding/CrossAttn) → [B, T, S, D]
  → flatten space into sequence → [B, T×S, D]
  → prepend context tokens: [session_flag + session_embed,
                              subject_flag + subject_embed,
                              task_flag + task_embed]
  → SpaceTimeTransformer (MAE Encoder-Decoder)
  → Masked reconstruction (Poisson NLL)
```

**NDT3：**
```
spike_counts [B, T, C, H]
  → QuantizeSimple (离散化)
  → nn.Embedding (per-neuron lookup) → flatten → spike_tokens [B, T, D]

constraint/return/covariate → 各自编码为 tokens

所有 tokens 按 (time, space_offset) 排序:
  [constraint_t0, spike_t0_s0, ..., spike_t0_s9, return_t0, cov_t0, ...]
  → + space_encoder(positions)   (learnable spatial embedding)
  → + time_encoder(times)        (learnable / Rotary PE)
  → Causal Transformer (Flash Attn v2, autoregressive)
  → 各模态独立 head 输出 → Poisson-softened CE + MSE
```

**POYO：**
```
每个 spike event (unit_index, timestamp, token_type)
  → unit_emb(unit_index) + token_type_emb(token_type) → input token [S, D]
  → RoPE(timestamp) → input_timestamp_emb

Latent tokens: latent_emb(index) → [M, D]; RoPE(latent_timestamps)
Output queries: session_emb(session_index) → [Q, D]; RoPE(output_timestamps)

Flow:
  input tokens + RoPE ──cross-attn──→ latent tokens
  latent tokens + RoPE ──self-attn──→ latent tokens (×depth)
  latent tokens + RoPE ──cross-attn──→ output queries
  output queries → readout (Linear) → prediction → MSE Loss
```

**Neuroformer：**
```
spike events 在 50ms 窗口内按发放顺序排列:
  → tok_emb(neuron_id) + pos_emb(position) + temp_emb(Δt) → [B, L, D]
  → Dropout

Visual stream (可选):
  → VideoEncoder(frames) / ViT → [B, L_frame, D]

Multimodal processing:
  → MultimodalTransformer(prev_tokens, curr_tokens, frame_tokens)
  → head_id → neuron_id 预测 (CE loss)
  → head_dt → time_offset 预测 (CE loss)
  → CLIP module → 跨模态对比 loss (可选)
```

**MtM：**
```
spike_data [B, T, N_channels]
  → embed_spikes (nn.Linear) → activation × sqrt(D) → [B, T, D]
  → + embed_pos(spikes_timestamp)    (learnable position embedding)
  → prepend embed_prompt(masking_mode)    (prompt token)
  → prepend embed_session(eid)            (session token)
  → 序列: [session_tok, prompt_tok, spike_t0, ..., spike_tT]
  → Dropout → Transformer Encoder
  → Decoder → Poisson rate → PoissonNLLLoss (仅 masked 位置)
```

**SPINT：**
```
校准阶段：
  X_i^C [M, T] → MLP_1 (逐 trial) → mean pooling → MLP_2 → E_i ∈ R^W

解码阶段：
  X_i [W] (当前窗口 spike counts) + E_i → Z_i = X_i + E_i
  → MLP_in(Z_i) → Z_in
  → CrossAttn(Q, LN(Z_in), LN(Z_in)) + 残差
  → MLP_attn + 残差 → Z_out
  → MLP_out → Ŷ_t → MSE Loss
```

### 4.4 各方案对跨 Session 迁移的影响

| 方案 | 对已见 session | 对未见 session | 所需校准数据 |
|------|--------------|--------------|------------|
| 隐式维度位置 (NDT1) | 不适用（单 session） | 不可迁移 | — |
| Session tokens (NDT2/3) | 查表即可 | 需微调 | 有标签数据 |
| Learnable unit emb (POYO) | 查表即可 | 梯度更新 embedding | 有标签数据 |
| IDEncoder (SPINT) | 前向传播 | 前向传播 | **无标签数据** |

SPINT 的 IDEncoder 方案在这个维度上最优——它对未见 session 只需几分钟的无标签校准数据和一次前向传播，完全不需要梯度更新。

---

## 5. Loss 设计：训练模型预测什么？

Loss 函数定义了模型的训练目标，直接影响学到的表示质量。在 spike 数据上，loss 设计需要考虑数据的统计特性（非负整数、稀疏、过度分散）和下游任务需求。

### 5.1 重建/预测 Loss

**Poisson Negative Log-Likelihood (Poisson NLL)。** 最经典的选择，被 NDT1 [[2]](#ref2), STNDT [[3]](#ref3), NDT2 [[5]](#ref5), MtM [[7]](#ref7), NEDS [[8]](#ref8) 采用。将 spike count 建模为泊松分布：

$$\mathcal{L}_{Poisson} = -\sum_{t,n} \left[ y_{t,n} \cdot \log(\lambda_{t,n}) - \lambda_{t,n} - \log(y_{t,n}!) \right]$$

其中 $$y_{t,n}$$ 是真实 spike count，$$\lambda_{t,n}$$ 是模型预测的 Poisson rate（通过 softplus 保证正值）。选择 Poisson NLL 的理由是 spike count 是非负整数且方差约等于均值，Poisson 分布是合理的生成模型假设。局限性在于真实神经数据常常存在 over-dispersion（方差 > 均值），且在低 count 区域梯度信号很弱。

**Poisson-Softened Cross-Entropy。** NDT3 [[6]](#ref6) 的选择——这并非标准的 categorical cross-entropy，而是使用 **Poisson PMF 作为 soft target** 的改进版本。将 spike count 离散化后，目标分布不是 one-hot 向量，而是以真实 count 为均值的 Poisson PMF：

$$\mathcal{L} = -\sum_{k=0}^{K} q_k \log p_k, \quad q_k = \frac{e^{-y} y^k / k!}{\sum_{j=0}^{K} e^{-y} y^j / j!}$$

当 $$y=0$$ 时，$$q_0 = 1$$（退化为 one-hot）；当 $$y=3$$ 时，$$q$$ 在 $$k=2,3,4$$ 附近分散概率。这种设计让模型在"预测 2 还是 3"时的错误代价小于"预测 0 还是 3"。NDT3 代码中同时支持标准 Poisson NLL 和 Poisson-softened CE 两种 spike loss，由配置选择。

**Neuron ID + Temporal Cross-Entropy。** Neuroformer [[14]](#ref14) 的自回归 loss。预测下一个 spike 来自哪个神经元（neuron ID 分类）以及何时发放（time offset 分类）：$$\mathcal{L} = \mathcal{L}_{neuron_id} + \mathcal{L}_{temporal}$$，这是唯一能驱动生成能力的 loss 设计。

**MSE（均方误差）。** 所有监督方法（POYO [[4]](#ref4), POSSM [[13]](#ref13), SPINT [[10]](#ref10)）用于预测连续行为变量（如手部速度）。NDT2/NDT3 在微调阶段也使用 MSE。NEDS 将 MSE 用于连续行为变量的重建。

**CTC Loss。** POSSM [[13]](#ref13) 在语音解码任务中使用 Connectionist Temporal Classification loss，这是语音识别的标准选择。

### 5.2 对比学习 Loss

**NT-Xent / InfoNCE。** STNDT [[3]](#ref3) 引入。对同一 trial 进行两种随机增强（时间裁剪、随机 dropout 神经元子集），要求两个 augmented view 在 embedding 空间中接近，不同 trial 的 view 远离：

$$\mathcal{L}_{contrastive} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$$

实际代码实现中 STNDT 将此称为 `info_nce_loss`，默认 temperature $$\tau=0.07$$。总 loss 为 $$\mathcal{L} = \mathcal{L}_{masked_recon} + \lambda \cdot \mathcal{L}_{contrastive}$$，$$\lambda$$ 默认 $$10^{-8}$$。

**InfoNCE / CLIP Loss。** Neuroformer [[14]](#ref14) 用于对齐 neural embedding 和 stimulus embedding。实际支持两种实现：(1) **单向 contrastive_loss**（简化 InfoNCE）；(2) **对称 clip_loss**（CLIP-style 双向 cross-entropy）。选择由配置控制。两种 loss 都支持多模态对之间的对比学习。

### 5.3 预训练范式（Self-Supervised vs. Supervised）

Loss 的选择与预训练范式紧密相关。目前有三种主要范式：

**Masked Autoencoding（MAE 风格）。** 随机遮盖部分 token，要求模型重建。NDT1 遮盖 20-30% 的时间步，NDT2 遮盖 25%（默认，可配置）的 patch token。Loss 通常是 Poisson NLL 或 cross-entropy。这是最广泛采用的自监督方案，但在稀疏数据上，大部分被遮盖位置的"正确答案"就是零，重建任务过于 trivial。

**自回归预测（GPT 风格）。** 预测下一个时间步（NDT3）或下一个 spike（Neuroformer）。NDT3 用 Poisson-softened CE 预测 spike count 类别，同时用 MSE 预测行为变量——本质上是一个多模态自回归模型。

**多任务遮盖（Multi-task Masking）。** MtM [[7]](#ref7) 的核心创新，定义了四种互补的遮盖方案并交替训练：(1) **Causal masking**（前向预测）：学习时间演化动力学；(2) **Neuron masking**（co-smoothing）：从其他神经元预测被 mask 神经元；(3) **Intra-region masking**：脑区内部的局部动力学；(4) **Inter-region masking**：跨脑区的信息传递。每种模式对应一个可学习的 prompt embedding（作为序列首部 prefix token），使模型知道当前执行哪种任务。

NEDS [[8]](#ref8) 将这个思路进一步扩展到跨模态场景，交替进行 neural masking（从行为重建 spike）和 behavioral masking（从 spike 解码行为），实现了在一个框架内同时训练编码和解码。

**大规模监督预训练。** POYO+ [[9]](#ref9), POSSM [[13]](#ref13), SPINT [[10]](#ref10) 没有自监督预训练阶段，而是在大规模多 session 数据上直接端到端训练行为解码。需要注意的是，POYO+ 实际上是在**多被试、多脑区、多任务**数据上进行监督预训练，再 finetune 到目标 session——这与"单 session 从头训练"有本质区别。更准确的分类应该是：**大规模监督预训练**（POYO+）vs. **自监督预训练**（NDT 系列、MtM），两者都属于"预训练 + 微调"范式，区别在于预训练阶段是否需要行为标签。

### 5.4 Loss 设计总结

| Loss | 用途 | 优势 | 局限 | 使用模型 |
|------|-----|------|-----|---------|
| Poisson NLL | Spike count 重建 | 合理的统计假设 | 低 count 区梯度弱；over-dispersion | NDT1/2, STNDT, MtM, NEDS |
| Poisson-softened CE | 离散化 spike 预测 | 无硬分布假设；训练稳定 | 离散化信息损失 | NDT3 |
| Neuron ID + Temporal CE | 自回归 spike 预测 | 驱动生成能力 | 推理慢 | Neuroformer |
| MSE | 行为变量预测 | 简单直接 | 仅适用于连续变量 | POYO, POSSM, SPINT, NDT2/3 |
| NT-Xent / InfoNCE | 对比学习 | 学习全局表示 | 需要 data augmentation | STNDT |
| InfoNCE / CLIP | 模态对齐 | 跨模态表示对齐 | 需要配对数据 | Neuroformer |
| CTC | 序列预测（语音） | 处理对齐未知的序列 | 特定于序列任务 | POSSM |

---

## 6. 各模型的技术方案一览

下表总结了主要模型在 tokenization、embedding、loss、架构和预训练范式上的选择：

| 模型 | 年份 | Tokenization | 关键 Embedding | Loss | 架构 | 预训练范式 |
|------|------|-------------|---------------|------|------|-----------|
| **NDT1** [[2]](#ref2) | 2021 | Binned 群体向量 (20ms) | Sinusoidal PE; 隐式维度编码 | Poisson NLL | Encoder-only (BERT) | Masked autoencoding |
| **STNDT** [[3]](#ref3) | 2022 | Binned 双流 (temporal + spatial) | 双 Sinusoidal PE | Poisson NLL + NT-Xent | 双流 attention + 时空融合 | Masked + contrastive |
| **NDT2** [[5]](#ref5) | 2023 | Binned 时空 patch (K=8) | Context tokens (session/subject/task/array) | Poisson NLL → MSE | MAE encoder-decoder | Multi-context masked AE |
| **Neuroformer** [[14]](#ref14) | 2024 | Spike 事件对 (neuron_id, offset) | Neuron emb + sinusoidal temporal emb + learnable position emb | CE + InfoNCE/CLIP | 多模态 Transformer (GPT + visual backbone + CLIP) | 自回归 + 对比 |
| **POYO** [[4]](#ref4) | 2023 | 单 spike token | InfiniteVocab unit emb + token_type emb + RoPE + session emb | MSE | PerceiverIO | 监督学习 |
| **POYO+** [[9]](#ref9) | 2025 | 单 spike + 钙成像 (split-dim) | InfiniteVocab unit emb + RoPE + task emb + session emb | MSE | PerceiverIO | 大规模监督预训练 |
| **MtM** [[7]](#ref7) | 2024 | Binned patch (兼容 NDT) | Learnable pos emb + session emb + prompt tokens (4种) | Poisson NLL | NDT backbone + 前缀 prompt | 四种 masking 交替 |
| **NDT3** [[6]](#ref6) | 2025 | Binned patch (K=32) + 离散化 + 多模态打包 | Learnable spatial emb + Rotary PE / learnable time emb + context/phase/return tokens | Poisson-softened CE + MSE | 自回归 Transformer (350M, Flash Attn v2) | 多模态自回归 |
| **NEDS** [[8]](#ref8) | 2025 | Modality-specific tokenizers | Temporal + modality + session emb | Poisson NLL + MSE + CE | 共享 backbone + 多任务 masking | 统一编码解码 |
| **POSSM** [[13]](#ref13) | 2025 | 单 spike (50ms chunks) | Learnable unit emb + RoPE | MSE / CE / CTC | PerceiverIO + SSM (Mamba) | 大规模监督预训练 |
| **SPINT** [[10]](#ref10) | 2025 | Neuron-as-token (binned) | IDEncoder 动态位置编码（双 MLP → 加法注入） | MSE | Cross-attention decoder (单层) | 监督学习 |

---

## 7. 技术演进脉络

纵观这些工作的发展，可以看到几条清晰的演进线索：

### 7.1 从单 Session 到多 Session 再到大规模预训练

```
NDT1 (2021)          → 单 session，证明 Transformer 在 spike 数据上可行
    ↓
NDT2 (2023)          → 跨 12 个被试预训练，context tokens 实现 session 特化
    ↓
NDT3 (2025)          → 30+ 被试、2000 小时，探索 scaling law
NEDS (2025)          → 83 只小鼠、多脑区、统一编码解码
```

这条线揭示了一个重要发现：**暴力 scaling 不是银弹**。NDT3 的作者明确指出，从 200 小时到 2000 小时预训练在 45M 模型上性能增益微乎其微，只有在 350M 模型上才显现。而且微调数据超过 1.5 小时后收益趋于平稳。数据异质性是根本限制——需要更智能的数据策划和架构设计，而非单纯堆量。

### 7.2 从 Binning 到 Spike-Level 表示

```
Binned 群体向量 (NDT1, 2021)    → 简单但丢失时间精度
    ↓
时空 Patch (NDT2, 2023)         → 部分解耦空间，但仍是 binned
    ↓
单 Spike Token (POYO, 2023)     → 毫秒级精度，事件驱动
    ↓
Spike + SSM 压缩 (POSSM, 2025)  → 保留精度 + 实时推理
```

这条线反映了领域对时间精度认识的加深。POYO 证明了 spike-level tokenization 的可行性和优越性，POSSM 则通过 SSM backbone 解决了实时推理的计算瓶颈。

### 7.3 从固定身份到动态身份

```
固定维度位置 (NDT1)        → 完全绑定到特定 session
    ↓
Learnable unit emb (POYO)  → 可迁移，但需梯度更新
    ↓
Session tokens (NDT2)       → 粗粒度 session 特化
    ↓
IDEncoder (SPINT, 2025)     → 零梯度、动态位置编码、置换不变
```

SPINT 的 IDEncoder 代表了当前神经元身份编码的最优方案。它不预先定义神经元身份，而是从校准数据中动态推断，并作为上下文依赖的位置编码注入——这正是应对神经元对应问题的正确思路。

### 7.4 从单任务到多任务 Loss

```
Poisson NLL 单目标 (NDT1)           → 简单但信号有限
    ↓
重建 + 对比 (STNDT)                 → token 级 + trial 级学习
    ↓
四种 masking 交替 (MtM)              → 多空间尺度统一学习
    ↓
多模态统一编码解码 (NEDS)            → 编码/解码在一个框架内
    ↓
自回归多模态 + return conditioning (NDT3) → 联合 neural + behavioral 建模
```

### 7.5 从纯 Transformer 到混合架构

```
标准 Transformer (NDT1-3)    → 功能强大但推理慢
    ↓
PerceiverIO (POYO)            → 处理变长输入
    ↓
PerceiverIO + SSM (POSSM)     → 实时推理，9x 加速
```

实时 BCI 应用的需求正在推动架构从纯 attention 向 SSM/循环混合架构转变。

---

## 8. 讨论与展望

### 8.1 未解决的核心问题

**跨脑区泛化**仍是最大挑战。目前没有模型展示了大规模跨脑区泛化——在运动皮层上训练的模型对视觉皮层毫无帮助。MtM 的 region-specific masking 是初步尝试，但远未解决问题。

**Scaling 悖论**也很突出：数据异质性限制了 scaling 的收益。仅仅增加更多数据不会自动带来更好的性能——如果新数据与目标 session 的脑区、任务差异太大，反而可能有害。

**长期稳定性**方面，慢性植入的多年稳定解码仍是挑战。所有现有模型都在离线或短期数据上评估，真正的临床 BCI 需要在月-年尺度上保持性能。

**跨物种迁移**才刚刚起步——仅 POSSM 初步展示了 NHP 到人类的迁移，将手写解码提升约 16%。

### 8.2 可能的未来方向

**方向一：混合 Tokenization。** 没有哪种 tokenization 在所有维度上最优。一个理想的方案可能是：POYO 的 spike-level 输入（保留时间精度）+ SPINT 的 IDEncoder（动态位置编码）+ PerceiverIO 压缩（控制计算量）+ MtM 的多任务 masking（多尺度学习）。具体来说，可以尝试在 neuron-as-token 框架中，不再用 binned 时间序列作为每个 neuron token 的表示，而是用 attention pooling over spike timestamps——既保留置换不变性，又不丢失时间精度。

**方向二：Hierarchical 时间建模。** 在 20ms bins 的基础上保留更细的子结构——比如每个 20ms bin 内记录 first-spike latency 和 ISI 分布的统计矩，作为额外特征。或者采用两级 attention：local attention 处理短程精细结构（~5ms），global attention 处理长程依赖（~100ms-1s）。这在 Transformer 框架下平衡了精度和效率。

**方向三：自监督 + 监督的统一。** 当前工作要么自监督预训练（NDT 系列、MtM），要么监督预训练（POYO+、POSSM、SPINT），很少有同时利用两者的。一个 promising 的方向是：先用大量无标签数据做 MtM/NEDS 风格的多任务自监督预训练，再用 POYO/SPINT 风格的监督微调——前者提供通用表示，后者适配具体任务。

**方向四：Domain-Adaptive 预训练。** 针对数据异质性，可以借鉴 NLP 中的 curriculum learning 思路——先在同质数据上训练，再逐步引入异质数据。或者采用 Mixture-of-Experts（MoE）架构，让不同 session/脑区激活不同的参数子集，避免异质数据之间的"互相干扰"。

**方向五：对比学习驱动的通用神经元表示。** 如 [Section 2.1](#21-神经元对应问题neuron-correspondence-problem) 所讨论，NuCLR [[11]](#ref11) 展示了通过对比学习自监督地学习 neuron-level 表示的可能性。将这一思路与 SPINT 的 IDEncoder 或 POYO 的 unit embedding 结合，有望在不依赖行为标签的情况下学到更丰富的神经元身份表示，从根本上缓解神经元对应问题。

**方向六：扩散模型。** LDNS 等工作已经开始探索将 spike 数据映射到连续潜在空间后使用扩散模型。这绕过了离散稀疏性的问题，可能为自监督学习提供新的范式——特别是在 Poisson NLL 梯度不稳定的低 count 区域。

**方向七：超越运动皮层。** 绝大多数工作集中在运动皮层解码。将基础模型扩展到视觉皮层（图像重建）、海马体（记忆编码）、前额叶（决策过程）等脑区，不仅有重大的科学价值，也会对 tokenization 和 loss 设计提出新的需求——比如视觉皮层对 onset latency 的精细时间结构更敏感，可能更需要 spike-level tokenization。

---

## 9. 总结

Spike neural foundation model 的发展，本质上是一个"如何将大脑的语言翻译成机器可以理解的表示"的过程。四年间，这个领域从单 session 的简单 binned tokenization（NDT1），走到了大规模多被试预训练（NDT3）、统一编码解码（NEDS）、零梯度跨 session 迁移（SPINT）和实时 SSM 解码（POSSM）。

每一步进展都在回答同一个核心问题的不同侧面：**什么是表示神经活动的最佳方式？** 目前的答案还不完整——没有哪个方案在所有维度上都最优。但几条共识正在形成：

1. **Spike-level 表示优于 binning**，尤其是在需要时间精度的应用中
2. **动态的神经元身份编码**（如 IDEncoder 的上下文依赖位置编码）优于固定的位置编码或查找表
3. **多任务、多尺度的训练目标**比单一 loss 学到更丰富的表示
4. **SSM/循环架构**在实时应用中比纯 Transformer 更有前景
5. **智能的数据策划**比暴力 scaling 更重要
6. **对比学习**有望提供自监督的通用神经元表示，从根本上缓解跨 session 泛化问题

下一个突破点，很可能来自将这些 insight 融合到一个统一框架中——一个既有 POYO 的时间精度、SPINT 的身份灵活性、MtM 的多尺度学习、又有 POSSM 的实时效率的模型。这不仅是工程上的挑战，更是对"大脑计算的通用表示是什么"这一基础科学问题的回答。

---

## 参考文献

<a id="ref1"></a>[1] Pandarinath, C., et al. "Inferring single-trial neural population dynamics using sequential auto-encoders (LFADS)." *Nature Methods*, 15(10):805-815, 2018. [[Paper]](https://doi.org/10.1038/s41592-018-0109-9)

<a id="ref2"></a>[2] Ye, J., & Pandarinath, C. "Representation learning for neural population activity with Neural Data Transformers." *Neurons, Behavior, Data Analysis, and Theory*, 2021. [[Paper]](https://arxiv.org/abs/2108.01210)

<a id="ref3"></a>[3] Le, T., & Bhaskara, A. "STNDT: Modeling Neural Population Activity with Spatiotemporal Transformers." *NeurIPS*, 2022. [[Paper]](https://arxiv.org/abs/2206.04727)

<a id="ref4"></a>[4] Azabou, M., et al. "A Unified, Scalable Framework for Neural Population Decoding (POYO)." *NeurIPS*, 2023. [[Paper]](https://arxiv.org/abs/2312.00826)

<a id="ref5"></a>[5] Ye, J., et al. "Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity." *NeurIPS*, 2023. [[Paper]](https://arxiv.org/abs/2305.16283)

<a id="ref6"></a>[6] Ye, J., et al. "Neural Data Transformer 3: Scaling Autoregressive Multi-Modal Foundation Models for Neural Spiking Data." *arXiv preprint*, 2025. [[Paper]](https://arxiv.org/abs/2501.13112)

<a id="ref7"></a>[7] Azabou, M., et al. "Multi-task Masking for Neural Spiking Data (MtM)." *ICLR*, 2024. [[Paper]](https://arxiv.org/abs/2407.06789)

<a id="ref8"></a>[8] Jude, J., et al. "Neural Encoding and Decoding with a Flow-based Spike Train Model (NEDS)." *NeurIPS*, 2025. [[Paper]](https://arxiv.org/abs/2501.13751)

<a id="ref9"></a>[9] Azabou, M., et al. "POYO+: A Multi-Modal, Multi-Task Foundation Model for Brain Activity (CaPOYO)." *arXiv preprint*, 2025. [[Paper]](https://arxiv.org/abs/2502.00816)

<a id="ref10"></a>[10] Le, T., et al. "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding." *NeurIPS*, 2025. [[Paper]](https://arxiv.org/abs/2507.08402)

<a id="ref11"></a>[11] Azabou, M., et al. "NuCLR: Nuclear Contrastive Learning Representations for Neural Activity." *ICLR*, 2025. [[Paper]](https://arxiv.org/abs/2512.01199) [[Project]](https://nerdslab.github.io/nuclr/)

<a id="ref12"></a>[12] Gallego, J.A., et al. "Long-term stability of cortical population dynamics underlying consistent behavior." *Nature Neuroscience*, 23:260-270, 2020. [[Paper]](https://doi.org/10.1038/s41593-019-0555-4)

<a id="ref13"></a>[13] Ye, J., et al. "POSSM: Population Spike Sequence Model for Real-Time BCI." *arXiv preprint*, 2025. [[Paper]](https://arxiv.org/abs/2503.04750)

<a id="ref14"></a>[14] Antoniades, A., et al. "Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data." *ICLR*, 2024. [[Paper]](https://arxiv.org/abs/2311.00136)

---

*本文覆盖的主要工作：NDT1 (2021), STNDT (2022), NDT2 (2023), Neuroformer (2024), POYO (2023), POYO+ (2025), MtM (2024), NDT3 (2025), NEDS (2025), POSSM (2025), SPINT (2025)*

</div>

<div class="lang-en" markdown="1">

> When we attempt to understand the brain's "language" using Transformers, the first question becomes——how do we transform neuronal spikes into tokens?

## 1. Why Do We Need Foundation Models for Spike Data?

Over the past decade, foundation models have achieved tremendous success in NLP and computer vision. The GPT series demonstrated the power of the "large-scale pretraining + downstream fine-tuning" paradigm, while CLIP and DINOv2 showcased the potential of visual representation learning. A natural question arises: **Can we do the same thing for neural activity data from the brain?**

In neuroscience, large-scale electrophysiology recording technologies (such as Utah Arrays and Neuropixels) enable us to simultaneously record spiking activity from hundreds or even thousands of neurons. This data contains core information about brain computation and finds widespread applications in brain-computer interfaces (BCI), motor decoding, and visual perception research. However, traditional methods (such as LFADS [[1]](#ref1)) typically train from scratch on individual sessions and cannot reuse knowledge across sessions or subjects.

From NDT1 in 2021 [[2]](#ref2) to NDT3 [[6]](#ref6), NEDS [[8]](#ref8), and SPINT [[10]](#ref10) in 2025, a series of works have attempted to introduce Transformer architectures to neural spiking data, progressively advancing toward "neural data foundation models." These works face the same core problems: **How should we tokenize spike data? What embedding representation should we use? How should we design the training objectives (loss)?**

This article will systematically review these technical choices, analyze the motivations, advantages, and limitations behind them, and explore future research directions.

---

## 2. Core Challenges of Spike Data

Before understanding various technical solutions, we must first understand why spike data is "difficult." Unlike text or images, spike data faces several unique and intertwined challenges.

### 2.1 Neuron Correspondence Problem

This is the most fundamental challenge facing spike foundation models.

In NLP, the token "cat" represents the same concept across all texts; in vision, pixel coordinate (100, 200) represents the same spatial location across all images. But in spiking data, **"channel 3" might record completely different neurons in different sessions**.

Even within the same subject, electrode drift causes neurons to appear and disappear across sessions. BrainGate clinical data shows that spike activity is recorded on only 35.6% of electrodes on average, and this decreases approximately 7% per year during chronic implantation over 7.6 years [[6]](#ref6). This means spike data inherently **lacks a shared vocabulary across sessions**.

| Modality | Standardization Scheme | Correspondence Difficulty |
|------|-----------|---------|
| EEG  | Standardized 10-20 electrode placement | Low |
| fMRI | MNI template space standardization | Low |
| Spiking | No standard—each implant captures a unique set of neurons | **Extremely High** |

More specifically, cross-session transfer has three progressively difficult levels:

- **Same subject, electrode drift**: Recordings from the same subject on different days have approximately 50-80% neuron overlap. Population-level functional structure remains mostly unchanged, with only partial membership changes.
- **Cross-subject, same brain region**: Different individuals' same brain region (e.g., motor cortex M1) have similar computational logic, but individual neuron tuning, connectivity patterns, and firing rate distributions are completely different.
- **Cross-brain region**: Different brain regions (e.g., V1 → M1) have fundamentally different computational logic, representing the most challenging scenario.

#### Existing Solutions and Potential Directions

Currently, different models adopt various strategies for the neuron correspondence problem, which can be categorized from simple to complex as follows:

**Solution A: Fixed Dimensional Encoding (NDT1 [[2]](#ref2)).** The simplest approach—a linear projection $$W_{in} \in \mathbb{R}^{D \times N}$$ hard-codes each neuron to a fixed direction in embedding space. Switching sessions completely changes the meaning of dimensions, making cross-session transfer impossible.

**Solution B: Learnable Unit Embedding (POYO [[4]](#ref4)).** Assigns a learnable embedding vector $$e_n \in \mathbb{R}^D$$ to each neuron. New sessions require freezing the backbone network and updating these embeddings via gradient descent. The advantage is explicit modeling of neuron identity; the disadvantage requires labeled calibration data and gradient updates.

**Solution C: Context-Dependent Positional Embedding / IDEncoder (SPINT [[10]](#ref10)).** Dynamically infers each unit's identity embedding from unlabeled calibration data through a shared MLP network and adds it as **context-dependent positional encoding** to spike tokens (see [Section 3.3](#33-neuron-as-token-以神经元为-token) and [Section 4.2](#42-神经元身份编码)). This is currently the most elegant solution, achieving zero-gradient cross-session transfer.

**Potential Direction One: Extending POYO's Learnable Unit Embedding to forward inference.** POYO currently assigns independent learnable embeddings to each unit, requiring gradient updates for new sessions. A natural extension is inspired by SPINT's IDEncoder approach—instead of maintaining independent embeddings for each unit, use a shared feedforward network to **directly forward-infer unit embeddings from raw calibration data**. Specifically, analogous to SPINT's IDEncoder, feed each unit's $$M$$ calibration trials of binned spike counts $$X_n^{calib} \in \mathbb{R}^{M \times T}$$ directly into the network, rather than manually extracting statistical features:

$$e_n = \psi\left(\frac{1}{M} \sum_{j=1}^{M} \phi(X_{n,j}^{calib})\right)$$

where $$\phi$$ and $$\psi$$ are shared multi-layer feedforward networks, and $$X_{n,j}^{calib}$$ is the raw binned spike counts of unit $$n$$'s $$j$$-th calibration trial. This end-to-end approach lets the network learn to extract meaningful identity features directly from raw data, completely avoiding the information bottleneck and inductive bias introduced by manually designed statistical features (such as firing rate distributions, ISI statistics, etc.). This amounts to grafting SPINT's IDEncoder module onto POYO's PerceiverIO architecture, giving it both POYO's spike-level temporal precision and SPINT's zero-gradient adaptation capability.

The **advantages** of this approach are: (1) fully data-driven—the network can automatically discover the most discriminative unit feature patterns without relying on hand-crafted statistics; (2) good compatibility with POYO's existing architecture, requiring only replacement of `InfiniteVocabEmbedding` with an IDEncoder module; (3) end-to-end training optimizes identity embeddings directly for downstream decoding tasks. **Limitations** include: (1) inference quality still highly depends on the representativeness of calibration data—if calibration trials are too few or fail to cover sufficient behavioral states, the learned identity may not be stable; (2) the expressiveness of feedforward networks is limited, potentially struggling to capture unit characteristics that require population context to disambiguate (e.g., two units with similar firing patterns but different functional roles).

**Potential Direction Two: Contrastive Learning for Unit Embedding.** Inspired by NuCLR [[11]](#ref11), we can adopt a self-supervised contrastive learning paradigm to learn representations for each unit. NuCLR's core idea is: the same neuron's activity across different time windows and stimulation conditions should produce similar representations (positive sample pairs); different neurons' activity should produce different representations (negative sample pairs). Specifically, NuCLR uses a **permutation-equivariant spatiotemporal Transformer** to integrate population activity context and uses contrastive objectives to pull closer different views of the same neuron and push apart different neurons' representations. The learned neuron-level embeddings can be used for downstream tasks (such as cell type classification, brain region identification) and demonstrate zero-shot generalization across subjects.

Introducing this approach to spike foundation models could envision the following framework: during pretraining, extract multiple views from different time segments of each unit's activity, obtain stable unit representations through contrastive learning; these representations are subsequently injected as unit embeddings into POYO or SPINT-style decoders. The **advantages** of this approach are: (1) completely self-supervised, no behavior labels needed; (2) learned representations reflect intrinsic functional properties of neurons rather than just statistical summaries; (3) contrastive learning naturally encourages discriminative representations, helping distinguish functionally similar but different units. **Limitations** include: (1) contrastive learning is sensitive to data augmentation strategies, requiring augmentations suitable for spike data (such as temporal cropping, random neuron subset dropout, etc., which STNDT [[3]](#ref3) has begun exploring); (2) large training overhead requires handling multiple views of many units; (3) gap remains between "good unit representation" and "good decoding performance," requiring validation of whether learned contrastive representations are truly useful for downstream tasks.

**Comparison of Two Approaches:**

| Dimension | Feedforward Inference (SPINT-style) | Contrastive Learning (NuCLR-style) |
|------|---------------------|---------------------|
| Supervision Signal | End-to-end (via downstream tasks) | Self-supervised (contrastive objective) |
| Calibration Requirements | Small amount of unlabeled data | Large amount of unlabeled data (pretraining) |
| New Session Adaptation | Single forward pass | Forward pass (if encoder frozen) |
| Representation Quality | Data-driven (raw spike counts) | Data-driven (contrastive objective), potentially richer |
| Computational Overhead | Low | High during pretraining |
| Maturity | Verified (SPINT) | Proof-of-concept stage (NuCLR) |

The ideal approach might be a combination of both: first pretrain a general unit encoder using contrastive learning on large-scale data, then fine-tune using end-to-end feedforward inference on specific downstream tasks. This way, we leverage the self-supervised advantages of contrastive learning while retaining the task adaptation capability of end-to-end optimization.

### 2.2 Extreme Sparsity

Typical cortical neurons fire at only 1-20 Hz. This means at 1ms resolution, over 99% of time bins are empty. This extreme sparsity has profound implications for technical solutions:

- **Tokenization level**: Are large numbers of zero tokens meaningful? For binned methods, most tokens carry nearly zero information.
- **Loss design level**: Poisson NLL has weak gradient signals at low counts (0-1 spikes/bin), making models prone to learning a trivial solution that "always predicts zero."
- **Self-supervised pretraining level**: If masked token reconstruction at most locations equals "predicting zero," the useful signal the model can learn is very limited.

### 2.3 Temporal Resolution vs. Efficiency Trade-off

One key advantage of spike data is millisecond-level temporal precision—differences in onset latency in visual cortex may be only a few milliseconds, which is critical for stimulus encoding. But maintaining this precision means extremely long sequences (1 second = 1000 1ms bins), directly conflicting with Transformer's quadratic complexity.

Real-world applications have additional latency constraints:

| Application | Latency Requirements | Challenge |
|------|---------|------|
| Cursor Control BCI | <100ms | Transformer quadratic complexity |
| Speech Decoding | ~200ms | Long context processing |
| Image Reconstruction | ~500ms | Multimodal fusion |

This forces designers to make trade-offs between "preserving information" and "computational feasibility."

### 2.4 Data Heterogeneity & Scale

Compared to NLP/vision, spiking datasets are several orders of magnitude smaller:

| Domain | Typical Training Scale |
|------|------------|
| Language Models | Trillions of tokens |
| Vision Models | Billions of images |
| Neural Spiking | ~1 billion tokens (NDT3 [[6]](#ref6), 2000 hours) |

More troubling is that even limited data is highly heterogeneous—different labs use different recording devices, processing algorithms, experimental paradigms, and behavioral tasks. NDT3's authors found that carefully selecting 5 sessions of data could outperform using all 84 sessions, because mismatched brain regions between sessions hurt performance. This shows that brute-force data accumulation fails; domain-specific data curation is critical.

### 2.5 Multi-Scale Non-Stationarity

Neural signals are not static—they change across multiple time scales:

| Time Scale | Source of Change | Impact |
|---------|---------|------|
| Minutes-Hours | Adaptation, fatigue, arousal | Firing rate fluctuation |
| Days-Weeks | Electrode drift, tissue changes | Neuron loss/appearance |
| Months-Years | Chronic degradation, learning plasticity | Signal quality decline |

The good news is that the "stable manifold hypothesis" (Gallego et al., 2020 [[12]](#ref12)) suggests that population-level dynamics may be more stable than individual neuron activity—motivating many works to learn representations from population rather than single-neuron levels.

---

## 3. Tokenization: How to Transform Spikes into Tokens?

Tokenization is the first step of all foundation models and the design choice with the most far-reaching impact. For spike data, the core question is: **At what granularity (temporal × spatial) do we discretize continuous neural activity into tokens?**

Currently, there are four main tokenization paradigms, each with distinctly different trade-offs.

### 3.1 Binned Population Vector (Population Vector per Time Bin)

**Representative Works:** NDT1 [[2]](#ref2), NDT2 [[5]](#ref5), NDT3 [[6]](#ref6), MtM [[7]](#ref7), NEDS [[8]](#ref8)

This is the most straightforward approach: bin spike trains at fixed time windows (typically 20ms), count spikes from each neuron in each bin, then project each timestep's complete population vector (N-dimensional, N = number of neurons) into a token. A 1-second trial at 20ms binning produces 50 tokens.

```
Raw spike trains → 20ms binning → N×T matrix → project each column → T tokens
```

**NDT1's Two Embedding Modes:** NDT1 actually supports two spike embedding modes. **Mode one** is linear projection:

$$\mathbf{h}_t = W_{in} \cdot \mathbf{x}_t + b, \quad W_{in} \in \mathbb{R}^{D \times N}$$

where $$\mathbf{x}_t \in \mathbb{R}^N$$ is the population spike count vector at timestep $$t$$. **Mode two** is per-neuron embedding—treat each neuron's spike count as a discrete variable, mapped to vectors via `nn.Embedding` lookup and concatenated:

$$\mathbf{h}_t = [E(x_{t,1}) \| E(x_{t,2}) \| \cdots \| E(x_{t,N})], \quad E: \{0,1,...,\text{max_spikes}\} \to \mathbb{R}^{d}$$

The latter mode treats spike counts as discrete categorical variables rather than continuous values, internally connected to NDT3's later discretization approach.

**NDT2's Improvement—Spatiotemporal Patches:** NDT2 [[5]](#ref5) introduces ViT-style spatial patching on this basis. Instead of treating all N neurons as one token, divide them into N/K groups (default **K=8**, i.e., each patch contains 8 neurons), with each group producing one token per timestep. NDT2 supports multiple readout strategies, including per-neuron linear projection, embedding lookup, and cross-attention readout. It also supports array embedding (for multi-electrode array scenarios). This decouples neuron identity—even if neurons between two sessions don't exactly match, as long as patch-level statistics are similar, patch-level token representations can generalize.

**NDT3's Further Improvement—Discretization + Multimodal Packing:** NDT3 [[6]](#ref6) inherits NDT2's patch tokenization (default **K=32**) and discretizes continuous spike counts into categorical variables via `torch.bucketize`, paired with Poisson-softened cross-entropy loss (see [Section 5.1](#51-重建预测-loss)). Additionally, NDT3 packs multiple modality tokens (spike, constraint, return, covariate) into a single sequence through **space offset**, sorted by `(timestep, space_offset)` to form a flattened multimodal sequence—an important architectural distinction of NDT3 from predecessors.

**Advantages:**
- Simple implementation, fixed and predictable sequence length (T = duration / bin_size)
- Directly compatible with standard Transformer architecture, no special handling needed
- Most mature approach with most comprehensive experimental validation

**Disadvantages:**
- **Severe sparsity problem**: Low firing rate regions have many near-zero tokens, wasting computation and providing weak gradient signals
- **Loss of temporal information**: Fine temporal structure within 20ms is irreversibly erased, potentially fatal for timing-sensitive applications like visual cortex
- **Least friendly to neuron correspondence**: Each dimension of population vector hard-codes a specific neuron's position; cross-session neuron changes directly disrupt input structure

### 3.2 Single Spike Tokenization (Per-Spike Token)

**Representative Works:** POYO [[4]](#ref4), POYO+ [[9]](#ref9), POSSM [[13]](#ref13)

POYO introduced the most "native" representation for spike data: **each individual spike event becomes a token**, with no temporal binning at all.

Specifically, each spike token consists of three pieces of information: a learnable unit embedding (identifying which neuron produced the spike), a token type embedding (identifying token type, such as spike, start/end markers, etc.), plus precise continuous timestamps encoded via Rotary Position Embeddings (RoPE). With 100 neurons each averaging 10Hz, 1 second produces approximately 1000 tokens—sequence length is directly proportional to actual spike count.

Mathematically, each input spike token is constructed as:

$$\mathbf{h}_i^{input} = E_{unit}(\text{unit_id}_i) + E_{type}(\text{token_type}_i)$$

where $$E_{unit}$$ uses `InfiniteVocabEmbedding` (a learnable embedding supporting dynamic vocabulary expansion; new units in new sessions can be registered dynamically), and $$E_{type}$$ is embedding for 4 token types. Temporal information is injected via RoPE during attention computation (see [Section 4.1](#41-时间位置编码)).

Due to sequence length growing with spike count, POYO pairs with **PerceiverIO architecture** as a compression mechanism: variable-length spike token sequences are compressed to a fixed number of latent tokens (e.g., 256) via cross-attention; subsequent self-attention operates only on these latent tokens. The entire process has three stages:

1. **Encode**: Latent tokens aggregate input spike tokens' information via cross-attention
2. **Process**: Latent tokens perform self-attention among themselves (2-6 layers)
3. **Decode**: Output queries extract prediction-needed information from latent tokens via cross-attention

Notably, POYO's decoder side is **session-aware**—using `session_emb` to construct output query embedding, with different output queries for different sessions.

**CaPOYO's Calcium Imaging Extension:** POYO+ extends support to calcium imaging data through an independent CaPOYO model class. CaPOYO employs a **split-dim concatenation design** to explicitly decouple signal value and unit identity:

$$\mathbf{h}_i = [\underbrace{W_{val} \cdot \Delta F/F_i + b_{val}}_{\in \mathbb{R}^{D/2}} \; \| \; \underbrace{E_{unit}(\text{unit_id}_i)}_{\in \mathbb{R}^{D/2}}]$$

Unlike spike tokens (where spike value is implicitly 1), calcium imaging tokens must encode both continuous fluorescence signal values and unit identity. POYO+ additionally introduces `task_emb` to support multi-task decoding (such as velocity decoding, position decoding, etc.).

**Advantages:**
- **Perfectly handles sparsity**: Sequence length proportional to spike count not temporal length, zero computation overhead for silent intervals
- **Preserves millisecond-level temporal precision**: RoPE encodes continuous timestamps with no information loss
- **PerceiverIO compression** elegantly handles different numbers of neurons across sessions

**Disadvantages:**
- High firing rate populations still have long sequences, input cross-attention has computational overhead
- Unit embedding requires per-neuron learning; new sessions need labeled calibration data for gradient relearning
- Doesn't directly encode brain region or task information, needs additional context embedding for handling heterogeneity

### 3.3 Neuron-as-Token (Each Neuron as a Token)

**Representative Works:** STNDT [[3]](#ref3), SPINT [[10]](#ref10)

This approach flips the perspective: instead of partitioning by timestep, **each neuron's complete time series becomes a spatial token**.

**STNDT's Dual-Stream Design:** STNDT simultaneously constructs two views—temporal tokens (population vector per timestep, $$[T, B, N]$$) and spatial tokens (time series per neuron, transposed as $$[N, B, T]$$), processed separately via attention mechanisms then fused. Both streams have independent linear embedders and sinusoidal position encodings. Spatial attention reweights temporal features via matrix multiplication:

$$Z_{ST} = A_S \cdot Z_T^\top$$

where $$A_S \in \mathbb{R}^{B \times N \times N}$$ is the spatial attention weight matrix (after softmax), and $$Z_T \in \mathbb{R}^{T \times B \times N}$$ is the temporal representation. The fused $$Z_{ST}$$ passes through residual connection and FFN, allowing the model to learn "which neurons should be considered together."

**SPINT's Core Innovation—IDEncoder Dynamic Positional Encoding:** SPINT constructs a spatial token from each neural unit's $$W$$ time bins of binned spike counts, paired with its core innovation—**context-dependent positional encoding via IDEncoder**.

SPINT's IDEncoder uses no fixed position encoding (which would assume fixed neuron order) but dynamically infers each unit's identity from calibration data, **adding it as positional encoding to spike activity**. The specific process is as follows:

1. **Input**: Collect unit $$i$$'s $$M$$ calibration trials $$X_i^C \in \mathbb{R}^{M \times T}$$ (each trial interpolated to fixed length $$T$$, such as T=1024 for M1/H1)
2. **Per-trial encoding**: Process each trial through shared three-layer MLP $$\phi$$
3. **Cross-trial aggregation**: Average-pool representations across all trials
4. **Identity generation**: Generate final identity embedding through second three-layer MLP $$\psi$$

Mathematically:

$$E_i = \text{IDEncoder}(X_i^C) = \psi\left(\frac{1}{M} \sum_{j=1}^{M} \phi(X_{i,j}^C)\right)$$

where $$\phi: \mathbb{R}^T \to \mathbb{R}^H$$ and $$\psi: \mathbb{R}^H \to \mathbb{R}^W$$ are respectively two three-layer fully-connected networks, with $$H$$ as hidden dimension (M1: $$H=1024$$; M2: $$H=512$$; H1: $$H=1024$$) and $$W$$ as window size (corresponding to spike token dimension).

**Key Step—Identity Embedding Injected as Positional Encoding:** The generated $$E_i$$ is **directly added to each unit's spike activity window**:

$$Z_i = X_i + E_i$$

Here $$X_i$$ is unit $$i$$'s binned spike counts in the current decoding window, and $$Z_i$$ is the identity-aware representation. Note that $$E_i$$ remains constant across all time windows within the same session—it encodes the unit's **stable identity** (similar to how position encoding in traditional Transformers encodes token position), while $$X_i$$ carries **time-varying activity**. This additive injection makes $$Z_i$$ simultaneously contain both "who is firing" (identity) and "what was fired" (activity) information.

Subsequently, $$Z_i$$ is projected via MLP to cross-attention's input space, decoded to behavior predictions by **learnable behavior query matrix** $$Q \in \mathbb{R}^{B \times W}$$ through single-layer cross-attention:

$$\hat{Y}_t = \text{MLP}_{out}(\text{CrossAttn}(Q, \text{LN}(Z_{in}), \text{LN}(Z_{in})))$$

The entire architecture mathematically guarantees **permutation invariance**:

$$\text{CrossAttn}(Q, P_R Z, P_R Z) = \text{CrossAttn}(Q, Z, Z)$$

where $$P_R$$ is an arbitrary row permutation matrix. Output is identical regardless of neuron ordering. Additionally, SPINT employs **dynamic channel dropout** to enhance robustness to composition changes of neurons across sessions.

**Cross-Session Transfer with Zero Gradient:** For unseen sessions, simply run the trained IDEncoder in forward pass on calibration data to infer all units' identity embeddings—no gradient updates or labeled data needed.

**Advantages:**
- **SPINT's permutation invariance** is currently the most elegant solution to neuron correspondence
- Spatial attention (STNDT) can discover functionally important neuron subsets
- IDEncoder achieves zero-gradient cross-session transfer
- Lightweight design (single cross-attention layer + two three-layer MLPs), suitable for real-time BCI

**Disadvantages:**
- Spatial attention has $$O(N^2)$$ complexity in neuron count N, potentially bottleneck for large-scale recordings
- Underlying still depends on binning, loses fine temporal information
- Currently validated only at smaller scales

### 3.4 Spike Event Pairs

**Representative Works:** Neuroformer [[14]](#ref14)

Neuroformer adopted the approach closest to NLP: encoding each spike event as a **(neuron_id, time_interval_offset)** pair, analogous to "words" in sentences.

```
All spikes within time window (50ms current window + 150ms history) → arranged by firing order →
[(neuron_3, offset_2), (neuron_7, offset_5), ...] → like a "sentence"
```

Each spike token's embedding is additively composed of three parts:

$$\mathbf{h}_i = E_{tok}(\text{neuron_id}_i) + E_{pos}(i) + E_{temp}(\Delta t_i)$$

where $$E_{tok}$$ is neuron ID embedding table (`nn.Embedding`), $$E_{pos}$$ is learnable position embedding (encoding position index in sequence), and $$E_{temp}$$ defaults to **sinusoidal temporal embedding** (encoding continuous time offset value $$\Delta t$$, not learnable embedding). Alternative learnable temporal embedding is optional, but code defaults to sinusoidal encoding.

Neuroformer's complete architecture is a **multimodal system** including: neural token embedding stem (the spike encoding described above), optional visual backbone (VideoEncoder/ResNet3D/ViT), MultimodalTransformer (handling neural-visual cross-modal attention), CLIP module (optional cross-modal contrastive learning), and independent head_id (predicting next neuron ID) and head_dt (predicting time offset) prediction heads.

**Advantages:**
- **Optimal sparsity handling**: Like POYO, only encodes moments with events
- **Only approach with generative capability**: As an autoregressive language model, can generate conditional spike train synthesis
- **High interpretability**: Attention weights directly reflect functional coupling between neurons; paper found attention maps mirror Hebbian connectivity

**Disadvantages:**
- No PerceiverIO-style compression; high firing rate populations incur large computation ($$O(L^2)$$)
- neuron_id is fixed vocabulary; weakest cross-session capability
- Autoregressive token-by-token inference is slow

### 3.5 Relationship and Distinction Between Per-Spike Token and Spike Event Pairs

POYO's Per-Spike Token and Neuroformer's Spike Event Pairs appear very similar on the surface—both **use individual spike events as basic processing units**, avoiding temporal information loss and sparsity waste from binning. But they differ fundamentally on several key dimensions:

**Different temporal encoding methods.** POYO uses RoPE to encode **absolute continuous timestamps** (in seconds); temporal information is implicitly injected during attention computation by rotating Q/K vectors without modifying token embeddings. Neuroformer uses **relative time offsets** (discrete offsets within current window), explicitly merged into token embeddings via addition. The former theoretically preserves more complete temporal information; the latter aligns more with traditional autoregressive language model practices.

**Different neuron identity encoding.** POYO uses dynamic-expandable `InfiniteVocabEmbedding`, allowing new units to register at runtime; Neuroformer uses fixed-size `nn.Embedding` lookup tables with vocabulary determined at training time. This means POYO has greater flexibility facing new sessions, while Neuroformer's fixed vocabulary limits cross-session capability.

**Different architecture paradigms.** POYO uses PerceiverIO to compress spike tokens to fixed-length latent space then process, a **discriminative** decoder. Neuroformer uses GPT-style autoregressive decoder, a **generative** model. This determines applicable downstream task types: POYO excels at behavior decoding, Neuroformer excels at spike pattern generation and functional connectivity analysis.

**Different training objectives.** POYO directly uses supervised MSE for behavior variable decoding; Neuroformer uses cross-entropy for autoregressive next-spike prediction, optionally paired with CLIP-style contrastive learning for cross-modal alignment.

| Dimension | POYO (Per-Spike Token) | Neuroformer (Spike Event Pairs) |
|------|----------------------|-------------------------------|
| Temporal Encoding | RoPE (absolute continuous timestamp) | Sinusoidal/Learnable (relative offset) |
| Identity Encoding | InfiniteVocabEmbedding (dynamic) | nn.Embedding (fixed vocabulary) |
| Sequence Compression | PerceiverIO (fixed latent) | No compression ($$O(L^2)$$ attention) |
| Model Paradigm | Discriminative decoder | Generative autoregressive |
| Training Objective | MSE (behavior variables) | CE (next spike) + contrastive learning |
| Cross-Session | Learnable embedding + gradient update | Fixed vocabulary, weakest |
| Unique Capability | Multimodal fusion (spike + calcium imaging) | Spike generation + functional connectivity analysis |

In summary, Per-Spike Token and Spike Event Pairs can be viewed as two different implementations of the same "event-driven" concept: **POYO optimizes representation efficiency and cross-session flexibility, Neuroformer optimizes generative capability and interpretability**. An ideal future solution might combine POYO's PerceiverIO compression and InfiniteVocabEmbedding with Neuroformer's autoregressive generation capability—simultaneously achieving efficient decoding and spike generation in a unified event-driven framework.

### 3.6 Comprehensive Comparison of Four Tokenization Approaches

| Dimension | Binned Population Vector | Per-Spike | Neuron-as-Token | Spike Event Pairs |
|------|----------------|----------|-----------------|-------------|
| Temporal Precision | ★★☆☆☆ (20ms) | ★★★★★ (ms-level) | ★★☆☆☆ (bin-dependent) | ★★★★☆ (discrete within window) |
| Sparsity Handling | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★★★ |
| Computational Efficiency | ★★★★★ (fixed length) | ★★★★☆ (with compression) | ★★★☆☆ ($$O(N^2)$$) | ★★☆☆☆ (no compression) |
| Neuron Correspondence | ★★☆☆☆ | ★★★☆☆ | ★★★★★ (SPINT) | ★★☆☆☆ |
| Generative Capability | ★★★☆☆ (reconstruction) | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |
| Interpretability | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |

> **Note:** POYO, POSSM, SPINT, and other purely supervised/discriminative models don't aim at generation, so generation scores reflect architectural potential rather than design intent. Rating POYO's generation capability as ★☆☆☆☆ is similar to saying BERT isn't good at text generation—technically correct but not its design goal.

---

## 4. Embedding: How to Give Tokens Identity and Context?

Tokenization solves the "how to partition" problem, while Embedding solves "how to represent"—particularly how to encode neuron identity, temporal position, and contextual information in tokens. This section details the specific encoding methods each project adopts and how they inject these into the network.

### 4.1 Temporal Position Encoding

Temporal position encoding determines how the model perceives token positions on the temporal axis. Projects adopt three main approaches:

**Sinusoidal Positional Encoding (Sinusoidal PE).** NDT1 [[2]](#ref2) and STNDT [[3]](#ref3) use standard Transformer sinusoidal position encoding, encoding discrete timestep indices:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Injection method is **additive**: $$\mathbf{h}_t \leftarrow \mathbf{h}_t + PE(t)$$. NDT1 also supports learnable position embedding (`nn.Embedding`). STNDT's two streams each have independent sinusoidal position encodings—temporal PE with dimension $$D_T = N$$ and sequence length $$T$$; spatial PE with dimension $$D_S = T$$ and sequence length $$N$$.

Neuroformer [[14]](#ref14)'s temporal encoding also defaults to **sinusoidal functions** (`TemporalEmbedding`), but encodes **continuous time offset values** $$\Delta t$$ (not discrete indices), optionally paired with learnable temporal embedding. Additionally Neuroformer has independent learnable position embedding (`nn.Parameter`) encoding position indices within sequences.

**Learnable Position/Temporal Embedding.** MtM [[7]](#ref7) uses learnable position embedding (`nn.Embedding`), injected additively to spike tokens. NDT3 [[6]](#ref6) supports both learnable time embedding (additive injection) and Rotary PE (injection within attention layer) modes, plus a learnable **spatial embedding** (`nn.Embedding`) to distinguish positions of different spatial patches within the same timestep. NEDS [[8]](#ref8) also uses learnable temporal embedding.

**Rotary Position Embeddings (RoPE).** POYO [[4]](#ref4)/POSSM [[13]](#ref13) and NDT3 (optional mode) use RoPE. RoPE doesn't modify token embeddings but rotates key/query vectors during attention computation, making attention score naturally reflect **relative** temporal distance. POYO's RoPE encodes **continuous timestamps** (in seconds), mathematically:

$$\text{RoPE}(x_{2i-1}, x_{2i}, t) = \begin{pmatrix} x_{2i-1} \cos(\omega_i t) - x_{2i} \sin(\omega_i t) \\ x_{2i-1} \sin(\omega_i t) + x_{2i} \cos(\omega_i t) \end{pmatrix}$$

where $$\omega_i = 2\pi / T_i$$ with $$T_i$$ log-uniformly distributed on $$[T_{min}, T_{max}]$$ (default $$T_{min}=10^{-4}$$, $$T_{max}\approx 2.06$$). Only rotates half of head dimensions (default 32 of head_dim=64), leaving the other half unchanged. NDT3's RoPE encodes **discrete timestep indices**.

**Summary of Temporal Encoding Across Projects:**

| Project | Temporal Encoding Type | Encoding Target | Injection Method |
|------|------------|---------|---------|
| NDT1 | Sinusoidal PE / Learnable PE | Discrete timestep index | Additive |
| STNDT | Sinusoidal PE (two independent) | Discrete timestep/neuron index | Additive |
| NDT2 | No explicit temporal encoding | — | — |
| NDT3 | Learnable time emb / Rotary PE + Learnable spatial emb | Discrete timestep + spatial position | Additive / Attention rotation |
| POYO/POSSM | Rotary PE | Continuous timestamp (seconds) | Attention rotation |
| Neuroformer | Sinusoidal temporal emb (default) + Learnable position emb | Continuous $$\Delta t$$ + sequence index | Additive |
| MtM | Learnable position emb | Discrete timestep index | Additive |
| NEDS | Learnable temporal emb | Discrete timestep | Additive |

### 4.2 Neuron Identity Encoding

This is the most critical embedding choice, directly determining the model's cross-session capability.

**Implicit Positional Encoding (Dimensional Position in Population Vector).** NDT1 [[2]](#ref2)'s linear projection $$W_{in} \in \mathbb{R}^{D \times N}$$ implicitly maps each neuron to a specific direction in embedding space. The $$i$$-th neuron's spike count always multiplies the $$i$$-th column of $$W_{in}$$. This means neuron identity is completely determined by input dimension position—switching sessions changes dimension meanings.

**Learnable Unit Embeddings.** POYO [[4]](#ref4)/POYO+ [[9]](#ref9) use `InfiniteVocabEmbedding`, assigning learnable embedding vectors $$e_n \in \mathbb{R}^D$$ to each neural unit. Support dynamic vocabulary expansion; new units in new sessions can register at runtime. New sessions require freezing the backbone and relearning embeddings via gradient descent. CaPOYO's unit embedding is half-dimensional ($$D/2$$), concatenated with value map.

**Neuron ID Embedding Table.** Neuroformer [[14]](#ref14) uses fixed-size `nn.Embedding`, mapping neuron_id to vectors. Vocabulary determined at training time, limiting cross-session capability.

**Context-Dependent Positional Embedding / IDEncoder.** SPINT [[10]](#ref10)'s core innovation (detailed in [Section 3.3](#33-neuron-as-token-以神经元为-token)). Dynamically infers unit identity embedding $$E_i$$ from unlabeled calibration data through shared dual-MLP network and adds it **as positional encoding** to spike activity. These embeddings reflect each neuron's functional role in the current session (such as firing rate patterns, temporal correlation characteristics, etc.) rather than fixed channel indices.

**Session/Context Tokens.** NDT2 [[5]](#ref5) introduces learnable session embedding, subject embedding, and task embedding. Injection methods are: (1) **Token strategy**: prepend as additional tokens to sequence start, with flag parameters as type indicators; (2) **Concat strategy**: concatenate to each token embedding then project. NDT3 [[6]](#ref6) further adds phase tokens (BCI vs. native control) and return tokens (controller quality, Decision Transformer style).

**Session Embedding + Prompt Token.** MtM [[7]](#ref7) uses session embedding (`nn.Embedding`) and prompt embedding (one for each of 4 masking modes). Injection method is **sequence prefix token**—prompt token at first position, session token at second position, allowing the model to know current session and masking task type by reading sequence start tokens.

**Session-Specific Projection.** NEDS [[8]](#ref8) learns independent linear projections $$W_{neural} \in \mathbb{R}^{N_{session} \times D}$$ for each session, handling different neuron counts across sessions. All tokens also get modality embedding and session embedding.

### 4.3 Detailed Embedding Injection Flow for Each Project

To more clearly understand how embeddings flow through networks in each model, this section provides embedding injection flowcharts for each project:

**NDT1:**
```
spike_counts [T, B, N]
  → embedder (Linear or per-neuron Embedding) → [T, B, D]
  → × sqrt(D)  (scale)
  → + Sinusoidal PE
  → Dropout
  → Transformer Encoder (BERT-style, with context mask)
  → Decoder (Linear → Poisson rate)
  → PoissonNLLLoss
```

**STNDT:**
```
spike_data [T, B, N]
  ├─ temporal_embedder (Linear) → [T, B, D_T] + temporal_PE → Temporal Self-Attention
  │                                                              ↓ (src)
  └─ .permute → [N, B, T]                                       │
     → spatial_embedder (Linear) → [N, B, D_S] + spatial_PE      │
     → Spatial Self-Attention → spatial_weights (A_S)             │
                                    ↓                             ↓
                          Z_ST = bmm(A_S, Z_T^T) ←───────────────┘
                                    ↓
                              残差 + FFN → output → PoissonNLL + InfoNCE
```

**NDT2:**
```
spike_data [B, T, A, C, H]
  → spatial grouping (every K=8 neurons per group) → [B, T, S, K]
  → readin (Linear/Embedding/CrossAttn) → [B, T, S, D]
  → flatten space into sequence → [B, T×S, D]
  → prepend context tokens: [session_flag + session_embed,
                              subject_flag + subject_embed,
                              task_flag + task_embed]
  → SpaceTimeTransformer (MAE Encoder-Decoder)
  → Masked reconstruction (Poisson NLL)
```

**NDT3:**
```
spike_counts [B, T, C, H]
  → QuantizeSimple (discretize)
  → nn.Embedding (per-neuron lookup) → flatten → spike_tokens [B, T, D]

constraint/return/covariate → encode as tokens respectively

All tokens sorted by (time, space_offset):
  [constraint_t0, spike_t0_s0, ..., spike_t0_s9, return_t0, cov_t0, ...]
  → + space_encoder(positions)   (learnable spatial embedding)
  → + time_encoder(times)        (learnable / Rotary PE)
  → Causal Transformer (Flash Attn v2, autoregressive)
  → Separate heads for each modality → Poisson-softened CE + MSE
```

**POYO:**
```
Each spike event (unit_index, timestamp, token_type)
  → unit_emb(unit_index) + token_type_emb(token_type) → input token [S, D]
  → RoPE(timestamp) → input_timestamp_emb

Latent tokens: latent_emb(index) → [M, D]; RoPE(latent_timestamps)
Output queries: session_emb(session_index) → [Q, D]; RoPE(output_timestamps)

Flow:
  input tokens + RoPE ──cross-attn──→ latent tokens
  latent tokens + RoPE ──self-attn──→ latent tokens (×depth)
  latent tokens + RoPE ──cross-attn──→ output queries
  output queries → readout (Linear) → prediction → MSE Loss
```

**Neuroformer:**
```
spike events within 50ms window arranged by firing order:
  → tok_emb(neuron_id) + pos_emb(position) + temp_emb(Δt) → [B, L, D]
  → Dropout

Visual stream (optional):
  → VideoEncoder(frames) / ViT → [B, L_frame, D]

Multimodal processing:
  → MultimodalTransformer(prev_tokens, curr_tokens, frame_tokens)
  → head_id → neuron_id prediction (CE loss)
  → head_dt → time_offset prediction (CE loss)
  → CLIP module → cross-modal contrastive loss (optional)
```

**MtM:**
```
spike_data [B, T, N_channels]
  → embed_spikes (nn.Linear) → activation × sqrt(D) → [B, T, D]
  → + embed_pos(spikes_timestamp)    (learnable position embedding)
  → prepend embed_prompt(masking_mode)    (prompt token)
  → prepend embed_session(eid)            (session token)
  → Sequence: [session_tok, prompt_tok, spike_t0, ..., spike_tT]
  → Dropout → Transformer Encoder
  → Decoder → Poisson rate → PoissonNLLLoss (masked positions only)
```

**SPINT:**
```
Calibration phase:
  X_i^C [M, T] → MLP_1 (per-trial) → mean pooling → MLP_2 → E_i ∈ R^W

Decoding phase:
  X_i [W] (current window spike counts) + E_i → Z_i = X_i + E_i
  → MLP_in(Z_i) → Z_in
  → CrossAttn(Q, LN(Z_in), LN(Z_in)) + residual
  → MLP_attn + residual → Z_out
  → MLP_out → Ŷ_t → MSE Loss
```

### 4.4 Impact of Each Approach on Cross-Session Transfer

| Approach | Known Session | Unseen Session | Calibration Data Needed |
|------|--------------|--------------|------------|
| Implicit dimension position (NDT1) | N/A (single session) | Not transferable | — |
| Session tokens (NDT2/3) | Lookup | Fine-tuning needed | Labeled data |
| Learnable unit emb (POYO) | Lookup | Gradient update embedding | Labeled data |
| IDEncoder (SPINT) | Forward pass | Forward pass | **No labeled data** |

SPINT's IDEncoder approach is optimal on this dimension—for unseen sessions it requires only minutes of unlabeled calibration data and one forward pass, completely avoiding gradient updates.

---

## 5. Loss Design: What Should the Model Predict?

Loss functions define training objectives and directly impact representation quality. On spike data, loss design must consider data statistics (non-negative integers, sparse, over-dispersed) and downstream task requirements.

### 5.1 Reconstruction/Prediction Loss

**Poisson Negative Log-Likelihood (Poisson NLL).** The most classical choice, adopted by NDT1 [[2]](#ref2), STNDT [[3]](#ref3), NDT2 [[5]](#ref5), MtM [[7]](#ref7), NEDS [[8]](#ref8). Models spike counts as Poisson distribution:

$$\mathcal{L}_{Poisson} = -\sum_{t,n} \left[ y_{t,n} \cdot \log(\lambda_{t,n}) - \lambda_{t,n} - \log(y_{t,n}!) \right]$$

where $$y_{t,n}$$ is true spike count and $$\lambda_{t,n}$$ is model-predicted Poisson rate (ensured positive via softplus). Poisson NLL is chosen because spike counts are non-negative integers with variance approximately equal to mean, making Poisson distribution a reasonable generative model assumption. Limitations are that real neural data often exhibits over-dispersion (variance > mean) and gradient signals are weak in low count regions.

**Poisson-Softened Cross-Entropy.** NDT3 [[6]](#ref6)'s choice—this isn't standard categorical cross-entropy but an improved version using **Poisson PMF as soft targets**. After discretizing spike counts, target distribution isn't one-hot vectors but Poisson PMF with true count as mean:

$$\mathcal{L} = -\sum_{k=0}^{K} q_k \log p_k, \quad q_k = \frac{e^{-y} y^k / k!}{\sum_{j=0}^{K} e^{-y} y^j / j!}$$

When $$y=0$$, $$q_0 = 1$$ (degenerates to one-hot); when $$y=3$$, $$q$$ spreads probability near $$k=2,3,4$$. This design makes predicting "2 vs 3" less costly than "0 vs 3." NDT3 code supports both standard Poisson NLL and Poisson-softened CE as spike loss, selected via configuration.

**Neuron ID + Temporal Cross-Entropy.** Neuroformer [[14]](#ref14)'s autoregressive loss. Predicts which neuron the next spike comes from (neuron ID classification) and when it fires (time offset classification): $$\mathcal{L} = \mathcal{L}_{neuron_id} + \mathcal{L}_{temporal}$$. This is the only loss design driving generative capability.

**MSE (Mean Squared Error).** All supervised approaches (POYO [[4]](#ref4), POSSM [[13]](#ref13), SPINT [[10]](#ref10)) use MSE for predicting continuous behavior variables (such as hand velocity). NDT2/NDT3 also use MSE during fine-tuning. NEDS uses MSE for continuous behavior variable reconstruction.

**CTC Loss.** POSSM [[13]](#ref13) uses Connectionist Temporal Classification loss for speech decoding tasks, the standard choice for speech recognition.

### 5.2 Contrastive Learning Loss

**NT-Xent / InfoNCE.** Introduced by STNDT [[3]](#ref3). Apply two random augmentations (temporal cropping, random neuron subset dropout) to same trial, require two augmented views to be close in embedding space while different trial views are pushed apart:

$$\mathcal{L}_{contrastive} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$$

In actual code implementation STNDT calls this `info_nce_loss` with default temperature $$\tau=0.07$$. Total loss is $$\mathcal{L} = \mathcal{L}_{masked_recon} + \lambda \cdot \mathcal{L}_{contrastive}$$, with $$\lambda$$ defaulting to $$10^{-8}$$.

**InfoNCE / CLIP Loss.** Used by Neuroformer [[14]](#ref14) for aligning neural embedding and stimulus embedding. Actually supports two implementations: (1) **unidirectional contrastive_loss** (simplified InfoNCE); (2) **symmetric clip_loss** (CLIP-style bidirectional cross-entropy). Selection via configuration. Both losses support contrastive learning between multimodal pairs.

### 5.3 Pretraining Paradigms (Self-Supervised vs. Supervised)

Loss choice is tightly connected to pretraining paradigm. Three main paradigms currently exist:

**Masked Autoencoding (MAE-style).** Randomly mask portions of tokens and require model reconstruction. NDT1 masks 20-30% of timesteps, NDT2 masks 25% (default, configurable) of patch tokens. Loss is usually Poisson NLL or cross-entropy. This is the most widely adopted self-supervised approach, but on sparse data, most masked position "correct answers" are just zero, making reconstruction too trivial.

**Autoregressive Prediction (GPT-style).** Predict next timestep (NDT3) or next spike (Neuroformer). NDT3 uses Poisson-softened CE predicting spike count categories, simultaneously predicts behavior variables with MSE—essentially a multimodal autoregressive model.

**Multi-Task Masking.** MtM [[7]](#ref7)'s core innovation, defining four complementary masking schemes trained alternately: (1) **Causal masking** (forward prediction): learn temporal evolution dynamics; (2) **Neuron masking** (co-smoothing): predict masked neurons from other neurons; (3) **Intra-region masking**: local dynamics within brain region; (4) **Inter-region masking**: information transfer across regions. Each mode corresponds to learnable prompt embedding (prefix token at sequence start), letting model know current task. NEDS [[8]](#ref8) further extends this to multimodal scenarios, alternating neural masking (reconstruct spikes from behavior) and behavioral masking (decode behavior from spikes), achieving unified encoding/decoding in one framework.

**Large-Scale Supervised Pretraining.** POYO+ [[9]](#ref9), POSSM [[13]](#ref13), SPINT [[10]](#ref10) have no self-supervised pretraining, instead directly training behavior decoding end-to-end on large multi-session data. Importantly, POYO+ actually pretrains on **multi-subject, multi-region, multi-task** data before fine-tuning to target session—fundamentally different from "single session from scratch." More accurate classification should be: **large-scale supervised pretraining** (POYO+) vs. **self-supervised pretraining** (NDT series, MtM), both belonging to "pretraining + fine-tuning" paradigm with difference in whether pretraining requires behavior labels.

### 5.4 Loss Design Summary

| Loss | Purpose | Advantages | Limitations | Used by |
|------|-----|------|-----|---------|
| Poisson NLL | Spike count reconstruction | Reasonable statistical assumption | Weak gradient at low counts; over-dispersion | NDT1/2, STNDT, MtM, NEDS |
| Poisson-softened CE | Discretized spike prediction | No hard distribution assumption; stable training | Information loss from discretization | NDT3 |
| Neuron ID + Temporal CE | Autoregressive spike prediction | Drives generative capability | Slow inference | Neuroformer |
| MSE | Behavior variable prediction | Simple and direct | Only for continuous variables | POYO, POSSM, SPINT, NDT2/3 |
| NT-Xent / InfoNCE | Contrastive learning | Learn global representations | Requires data augmentation | STNDT |
| InfoNCE / CLIP | Modality alignment | Cross-modal representation alignment | Requires paired data | Neuroformer |
| CTC | Sequence prediction (speech) | Handle sequences with unknown alignment | Specific to sequence tasks | POSSM |

---

## 6. Technical Solutions Overview for Each Model

The following table summarizes major models' choices in tokenization, embedding, loss, architecture, and pretraining paradigm:

| Model | Year | Tokenization | Key Embedding | Loss | Architecture | Pretraining Paradigm |
|------|------|-------------|---------------|------|------|-----------|
| **NDT1** [[2]](#ref2) | 2021 | Binned population vector (20ms) | Sinusoidal PE; implicit dimension encoding | Poisson NLL | Encoder-only (BERT) | Masked autoencoding |
| **STNDT** [[3]](#ref3) | 2022 | Binned dual-stream (temporal + spatial) | Dual Sinusoidal PE | Poisson NLL + NT-Xent | Dual-stream attention + spatiotemporal fusion | Masked + contrastive |
| **NDT2** [[5]](#ref5) | 2023 | Binned spatiotemporal patch (K=8) | Context tokens (session/subject/task/array) | Poisson NLL → MSE | MAE encoder-decoder | Multi-context masked AE |
| **Neuroformer** [[14]](#ref14) | 2024 | Spike event pairs (neuron_id, offset) | Neuron emb + sinusoidal temporal emb + learnable position emb | CE + InfoNCE/CLIP | Multimodal Transformer (GPT + visual backbone + CLIP) | Autoregressive + contrastive |
| **POYO** [[4]](#ref4) | 2023 | Single spike token | InfiniteVocab unit emb + token_type emb + RoPE + session emb | MSE | PerceiverIO | Supervised learning |
| **POYO+** [[9]](#ref9) | 2025 | Single spike + calcium imaging (split-dim) | InfiniteVocab unit emb + RoPE + task emb + session emb | MSE | PerceiverIO | Large-scale supervised pretraining |
| **MtM** [[7]](#ref7) | 2024 | Binned patch (NDT-compatible) | Learnable pos emb + session emb + prompt tokens (4 types) | Poisson NLL | NDT backbone + prefix prompt | Four-way masking alternation |
| **NDT3** [[6]](#ref6) | 2025 | Binned patch (K=32) + discretization + multimodal packing | Learnable spatial emb + Rotary PE / learnable time emb + context/phase/return tokens | Poisson-softened CE + MSE | Autoregressive Transformer (350M, Flash Attn v2) | Multimodal autoregressive |
| **NEDS** [[8]](#ref8) | 2025 | Modality-specific tokenizers | Temporal + modality + session emb | Poisson NLL + MSE + CE | Shared backbone + multi-task masking | Unified encoding/decoding |
| **POSSM** [[13]](#ref13) | 2025 | Single spike (50ms chunks) | Learnable unit emb + RoPE | MSE / CE / CTC | PerceiverIO + SSM (Mamba) | Large-scale supervised pretraining |
| **SPINT** [[10]](#ref10) | 2025 | Neuron-as-token (binned) | IDEncoder dynamic positional encoding (dual MLP → additive injection) | MSE | Cross-attention decoder (single layer) | Supervised learning |

---

## 7. Technical Evolution Trajectory

Looking at the development of these works, several clear evolutionary threads are evident:

### 7.1 From Single Session to Multi-Session to Large-Scale Pretraining

```
NDT1 (2021)          → Single session, proves Transformer works on spike data
    ↓
NDT2 (2023)          → Pretraining across 12 subjects, context tokens for session specialization
    ↓
NDT3 (2025)          → 30+ subjects, 2000 hours, explores scaling laws
NEDS (2025)          → 83 mice, multiple brain regions, unified encoding/decoding
```

This thread reveals an important finding: **brute-force scaling isn't a silver bullet**. NDT3's authors explicitly note that scaling from 200 to 2000 pretraining hours shows minimal performance gains at 45M model size, only emerging at 350M model. Beyond 1.5 hours of fine-tuning data, gains plateau. Data heterogeneity is the fundamental constraint—needs smarter data curation and architecture design, not just quantity.

### 7.2 From Binning to Spike-Level Representation

```
Binned population vector (NDT1, 2021)    → Simple but loses temporal precision
    ↓
Spatiotemporal patches (NDT2, 2023)      → Partially decouples space, still binned
    ↓
Single spike token (POYO, 2023)          → Millisecond precision, event-driven
    ↓
Spike + SSM compression (POSSM, 2025)    → Preserves precision + real-time inference
```

This thread reflects deepening understanding of temporal precision. POYO proves spike-level tokenization feasibility and superiority; POSSM solves real-time inference computational bottleneck via SSM backbone.

### 7.3 From Fixed Identity to Dynamic Identity

```
Fixed dimension position (NDT1)        → Completely bound to specific session
    ↓
Learnable unit emb (POYO)              → Transferable, requires gradient update
    ↓
Session tokens (NDT2)                  → Coarse-grain session specialization
    ↓
IDEncoder (SPINT, 2025)                → Zero-gradient, dynamic positional encoding, permutation-invariant
```

SPINT's IDEncoder represents current optimal neuron identity encoding. Rather than predetermining neuron identity, it dynamically infers from calibration data and injects as context-dependent positional encoding—the right approach to neuron correspondence.

### 7.4 From Single-Task to Multi-Task Loss

```
Poisson NLL single target (NDT1)                    → Simple but limited signals
    ↓
Reconstruction + contrastive (STNDT)               → Token-level + trial-level learning
    ↓
Four-way masking alternation (MtM)                 → Multi-scale unified learning
    ↓
Unified multimodal encoding/decoding (NEDS)        → Encoding/decoding in one framework
    ↓
Autoregressive multimodal + return conditioning (NDT3) → Joint neural + behavioral modeling
```

### 7.5 From Pure Transformer to Hybrid Architecture

```
Standard Transformer (NDT1-3)     → Powerful but slow inference
    ↓
PerceiverIO (POYO)                 → Handle variable-length inputs
    ↓
PerceiverIO + SSM (POSSM)          → Real-time inference, 9x speedup
```

Real-time BCI application needs are driving architecture evolution from pure attention toward SSM/recurrent hybrid architectures.

---

## 8. Discussion and Future Directions

### 8.1 Unsolved Core Problems

**Cross-region generalization** remains the biggest challenge. No model demonstrates large-scale cross-region generalization—models trained on motor cortex are useless for visual cortex. MtM's region-specific masking is an initial attempt but far from solving the problem.

**Scaling paradox** is also prominent: data heterogeneity limits scaling benefits. Simply adding more data doesn't automatically improve performance—if new data differs too much from target session in brain region or task, it can actually harm.

**Long-term stability** is challenging; multi-year stable decoding from chronic implants remains unsolved. All current models are evaluated on offline or short-term data; real clinical BCI requires months-to-years performance maintenance.

**Cross-species transfer** barely started—only POSSM initially showed NHP-to-human transfer, improving handwriting decoding approximately 16%.

### 8.2 Potential Future Directions

**Direction One: Hybrid Tokenization.** No tokenization is optimal across all dimensions. An ideal solution might combine: POYO's spike-level input (temporal precision) + SPINT's IDEncoder (identity flexibility) + PerceiverIO compression (computational control) + MtM's multi-task masking (multi-scale learning). Specifically, within neuron-as-token framework, instead of binned time series as neuron token representation, use attention pooling over spike timestamps—preserving permutation invariance while not losing temporal precision.

**Direction Two: Hierarchical Temporal Modeling.** Preserve fine structure within 20ms bins—recording first-spike latency and ISI distribution statistics within each 20ms bin as additional features. Or adopt two-level attention: local attention handling short-range fine structure (~5ms), global attention handling long-range dependencies (~100ms-1s). This balances precision and efficiency within Transformer.

**Direction Three: Unified Self-Supervised + Supervised.** Current works either do self-supervised pretraining (NDT series, MtM) or supervised pretraining (POYO+, POSSM, SPINT), rarely combining both. Promising direction: first pretrain with large unlabeled data using MtM/NEDS multi-task self-supervision, then fine-tune with POYO/SPINT supervised approach—former provides general representations, latter adapts to specific tasks.

**Direction Four: Domain-Adaptive Pretraining.** For data heterogeneity, borrow curriculum learning from NLP—train on homogeneous data first, gradually introduce heterogeneous data. Or adopt Mixture-of-Experts (MoE) architecture, letting different sessions/regions activate different parameter subsets, avoiding heterogeneous data "mutual interference."

**Direction Five: Contrastive Learning-Driven Universal Neuron Representations.** As discussed in [Section 2.1](#21-neuron-correspondence-problem), NuCLR [[11]](#ref11) shows self-supervised neuron-level representation learning via contrastive learning. Combining this with SPINT's IDEncoder or POYO's unit embedding could learn richer neuron identity representations without behavior labels, fundamentally alleviating cross-session generalization.

**Direction Six: Diffusion Models.** Works like LDNS already explore mapping spike data to continuous latent space then using diffusion models. This circumvents discrete sparsity issues, potentially offering new self-supervised paradigm—especially in low count regions where Poisson NLL gradient is unstable.

**Direction Seven: Beyond Motor Cortex.** Vast majority focus on motor cortex decoding. Extending foundation models to visual cortex (image reconstruction), hippocampus (memory encoding), prefrontal cortex (decision processes), etc. has major scientific value and poses new tokenization/loss demands—visual cortex more sensitive to onset latency fine structure, possibly needing spike-level tokenization.

---

## 9. Conclusion

Spike neural foundation model development is fundamentally a process of "translating the brain's language into representations machines can understand." In four years, the field progressed from single-session simple binned tokenization (NDT1) to large-scale multi-subject pretraining (NDT3), unified encoding/decoding (NEDS), zero-gradient cross-session transfer (SPINT), and real-time SSM decoding (POSSM).

Each advance answers different facets of the same core question: **What is the best way to represent neural activity?** The answer remains incomplete—no approach is optimal across all dimensions. But several consensuses are forming:

1. **Spike-level representations outperform binning**, especially for timing-sensitive applications
2. **Dynamic neuron identity encoding** (such as IDEncoder's context-dependent positional encoding) outperforms fixed positional encoding or lookup tables
3. **Multi-task, multi-scale training objectives** learn richer representations than single loss
4. **SSM/recurrent architectures** show more promise in real-time applications than pure Transformer
5. **Intelligent data curation** matters more than brute-force scaling
6. **Contrastive learning** promises self-supervised universal neuron representations, fundamentally alleviating cross-session generalization

The next breakthrough likely comes from fusing these insights into a unified framework—one with POYO's temporal precision, SPINT's identity flexibility, MtM's multi-scale learning, and POSSM's real-time efficiency. This is both engineering challenge and answer to fundamental neuroscience question: "what is universal representation of brain computation?"

---

## References

<a id="ref1"></a>[1] Pandarinath, C., et al. "Inferring single-trial neural population dynamics using sequential auto-encoders (LFADS)." *Nature Methods*, 15(10):805-815, 2018. [[Paper]](https://doi.org/10.1038/s41592-018-0109-9)

<a id="ref2"></a>[2] Ye, J., & Pandarinath, C. "Representation learning for neural population activity with Neural Data Transformers." *Neurons, Behavior, Data Analysis, and Theory*, 2021. [[Paper]](https://arxiv.org/abs/2108.01210)

<a id="ref3"></a>[3] Le, T., & Bhaskara, A. "STNDT: Modeling Neural Population Activity with Spatiotemporal Transformers." *NeurIPS*, 2022. [[Paper]](https://arxiv.org/abs/2206.04727)

<a id="ref4"></a>[4] Azabou, M., et al. "A Unified, Scalable Framework for Neural Population Decoding (POYO)." *NeurIPS*, 2023. [[Paper]](https://arxiv.org/abs/2312.00826)

<a id="ref5"></a>[5] Ye, J., et al. "Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity." *NeurIPS*, 2023. [[Paper]](https://arxiv.org/abs/2305.16283)

<a id="ref6"></a>[6] Ye, J., et al. "Neural Data Transformer 3: Scaling Autoregressive Multi-Modal Foundation Models for Neural Spiking Data." *arXiv preprint*, 2025. [[Paper]](https://arxiv.org/abs/2501.13112)

<a id="ref7"></a>[7] Azabou, M., et al. "Multi-task Masking for Neural Spiking Data (MtM)." *ICLR*, 2024. [[Paper]](https://arxiv.org/abs/2407.06789)

<a id="ref8"></a>[8] Jude, J., et al. "Neural Encoding and Decoding with a Flow-based Spike Train Model (NEDS)." *NeurIPS*, 2025. [[Paper]](https://arxiv.org/abs/2501.13751)

<a id="ref9"></a>[9] Azabou, M., et al. "POYO+: A Multi-Modal, Multi-Task Foundation Model for Brain Activity (CaPOYO)." *arXiv preprint*, 2025. [[Paper]](https://arxiv.org/abs/2502.00816)

<a id="ref10"></a>[10] Le, T., et al. "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding." *NeurIPS*, 2025. [[Paper]](https://arxiv.org/abs/2507.08402)

<a id="ref11"></a>[11] Azabou, M., et al. "NuCLR: Nuclear Contrastive Learning Representations for Neural Activity." *ICLR*, 2025. [[Paper]](https://arxiv.org/abs/2512.01199) [[Project]](https://nerdslab.github.io/nuclr/)

<a id="ref12"></a>[12] Gallego, J.A., et al. "Long-term stability of cortical population dynamics underlying consistent behavior." *Nature Neuroscience*, 23:260-270, 2020. [[Paper]](https://doi.org/10.1038/s41593-019-0555-4)

<a id="ref13"></a>[13] Ye, J., et al. "POSSM: Population Spike Sequence Model for Real-Time BCI." *arXiv preprint*, 2025. [[Paper]](https://arxiv.org/abs/2503.04750)

<a id="ref14"></a>[14] Antoniades, A., et al. "Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data." *ICLR*, 2024. [[Paper]](https://arxiv.org/abs/2311.00136)

---

*Major works covered in this article: NDT1 (2021), STNDT (2022), NDT2 (2023), Neuroformer (2024), POYO (2023), POYO+ (2025), MtM (2024), NDT3 (2025), NEDS (2025), POSSM (2025), SPINT (2025)*

</div>
