# Spike Neural Foundation Models：Tokenization、Embedding 与 Loss 的技术全景

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

**方案 A：固定维度编码（NDT1 [[2]](#ref2)）。** 最简单的方案——线性投影 $W_{in} \in \mathbb{R}^{D \times N}$ 将每个神经元硬编码到 embedding 空间的固定方向。换一个 session，维度含义就完全改变，无法跨 session 迁移。

**方案 B：Learnable Unit Embedding（POYO [[4]](#ref4)）。** 为每个神经元分配可学习 embedding 向量 $e_n \in \mathbb{R}^D$。新 session 需要冻结主干网络，通过梯度下降更新这些 embedding。优点是显式建模了神经元身份；缺点是需要有标签校准数据和梯度更新。

**方案 C：Context-Dependent Positional Embedding / IDEncoder（SPINT [[10]](#ref10)）。** 通过共享 MLP 网络从无标签校准数据中动态推断每个 unit 的 identity embedding，并将其作为**上下文依赖的位置编码**添加到 spike token 上（详见 [Section 3.3](#33-neuron-as-token以神经元为-token) 和 [Section 4.2](#42-神经元身份编码)）。这是目前最优雅的解决方案，实现了零梯度跨 session 迁移。

**潜在方向一：将 POYO 的 Learnable Unit Embedding 扩展为前向推断。** POYO 当前为每个 unit 分配一个独立的可学习 embedding，新 session 需要梯度更新。一种自然的扩展是借鉴 SPINT 的 IDEncoder 思路——不再为每个 unit 维护独立 embedding，而是通过一个共享的前馈网络（feedforward network）**直接从 unit 的原始校准数据前向推断出 unit embedding**。具体来说，类似 SPINT 的 IDEncoder，将每个 unit 的 $M$ 条校准 trial 的 binned spike counts $X_n^{calib} \in \mathbb{R}^{M \times T}$ 直接送入网络，而非手动提取统计特征：

$$e_n = \psi\left(\frac{1}{M} \sum_{j=1}^{M} \phi(X_{n,j}^{calib})\right)$$

其中 $\phi$ 和 $\psi$ 是共享的多层前馈网络，$X_{n,j}^{calib}$ 是 unit $n$ 第 $j$ 条校准 trial 的原始 binned spike counts。这种端到端的方式让网络自己从原始数据中学习提取有意义的身份特征，完全避免了手动设计统计特征（如发放率分布、ISI 统计等）带来的信息瓶颈和归纳偏置。这相当于将 SPINT 的 IDEncoder 模块嫁接到 POYO 的 PerceiverIO 架构中，使其同时具备 POYO 的 spike-level 时间精度和 SPINT 的零梯度适应能力。

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

其中 $\mathbf{x}_t \in \mathbb{R}^N$ 是时间步 $t$ 的群体 spike count 向量。**模式二**是 per-neuron embedding——将每个神经元的 spike count 视为离散变量，通过 `nn.Embedding` 查表映射为向量后拼接：

$$\mathbf{h}_t = [E(x_{t,1}) \| E(x_{t,2}) \| \cdots \| E(x_{t,N})], \quad E: \{0,1,...,\text{max\_spikes}\} \to \mathbb{R}^{d}$$

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

$$\mathbf{h}_i^{input} = E_{unit}(\text{unit\_id}_i) + E_{type}(\text{token\_type}_i)$$

其中 $E_{unit}$ 使用 `InfiniteVocabEmbedding`（一种支持动态词汇表扩展的 learnable embedding，新 session 的新 unit 可以动态注册），$E_{type}$ 是 4 种 token 类型的 embedding。时间信息则通过 RoPE 在 attention 计算时注入（详见 [Section 4.1](#41-时间位置编码)）。

由于序列长度随 spike 数量增长，POYO 搭配了 **PerceiverIO 架构**作为压缩机制：通过 cross-attention 将 variable-length 的 spike token 序列压缩到固定数量（如 256 个）的 latent token，后续的 self-attention 只在这些 latent token 上进行。整个流程分为三个阶段：

1. **Encode**：latent tokens 通过 cross-attention 聚合 input spike tokens 的信息
2. **Process**：latent tokens 之间进行 self-attention（2-6 层）
3. **Decode**：output queries 通过 cross-attention 从 latent tokens 中提取预测所需的信息

值得注意的是，POYO 的 decoder 端是 **session-aware** 的——使用 `session_emb` 构造 output query embedding，不同 session 使用不同的 query。

**CaPOYO 的钙成像扩展：** POYO+ 通过独立的 CaPOYO 模型类支持钙成像数据。CaPOYO 采用 **split-dim 拼接设计**显式解耦信号值和单元身份：

$$\mathbf{h}_i = [\underbrace{W_{val} \cdot \Delta F/F_i + b_{val}}_{\in \mathbb{R}^{D/2}} \; \| \; \underbrace{E_{unit}(\text{unit\_id}_i)}_{\in \mathbb{R}^{D/2}}]$$

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

**STNDT 的双流设计：** STNDT 同时构造两种视图——temporal tokens（每时间步的群体向量，$[T, B, N]$）和 spatial tokens（每个神经元的时间序列，转置为 $[N, B, T]$），通过独立的 attention 机制处理后融合。两个 stream 各有独立的线性 embedder 和正弦位置编码。Spatial attention 通过矩阵乘法重新加权 temporal 特征：

$$Z_{ST} = A_S \cdot Z_T^\top$$

其中 $A_S \in \mathbb{R}^{B \times N \times N}$ 是 spatial attention 的权重矩阵（softmax 后），$Z_T \in \mathbb{R}^{T \times B \times N}$ 是 temporal representation。融合后的 $Z_{ST}$ 经过残差连接和 FFN，让模型学习"哪些神经元应该一起被考虑"。

**SPINT 的核心创新——IDEncoder 动态位置编码：** SPINT 将每个 neural unit 的 $W$ 个时间 bin 的 binned spike counts 构成一个 spatial token，配合其核心创新——**IDEncoder 上下文依赖的位置编码**。

SPINT 的 IDEncoder 不使用任何固定位置编码（这会假设神经元有固定顺序），而是从校准数据中动态推断每个 unit 的 identity，并将其**作为位置编码添加到 spike 活动上**。具体过程如下：

1. **输入**：收集 unit $i$ 的 $M$ 条校准 trial 数据 $X_i^C \in \mathbb{R}^{M \times T}$（每条 trial 插值到固定长度 $T$，如 M1/H1 使用 $T=1024$）
2. **逐 trial 编码**：通过共享的三层 MLP $\phi$ 处理每条 trial
3. **跨 trial 聚合**：对所有 trial 的表示取均值池化
4. **身份生成**：通过第二个三层 MLP $\psi$ 生成最终的 identity embedding

数学上：

$$E_i = \text{IDEncoder}(X_i^C) = \psi\left(\frac{1}{M} \sum_{j=1}^{M} \phi(X_{i,j}^C)\right)$$

其中 $\phi: \mathbb{R}^T \to \mathbb{R}^H$ 和 $\psi: \mathbb{R}^H \to \mathbb{R}^W$ 分别是两个三层全连接网络，$H$ 为隐藏维度（M1: $H=1024$; M2: $H=512$; H1: $H=1024$），$W$ 为窗口大小（对应 spike token 的维度）。

**关键步骤——Identity Embedding 作为位置编码注入：** 生成的 $E_i$ 被**直接加到每个 unit 的 spike 活动窗口**上：

$$Z_i = X_i + E_i$$

这里 $X_i$ 是 unit $i$ 当前解码窗口的 binned spike counts，$Z_i$ 是 identity-aware 的表示。注意 $E_i$ 在同一 session 内对所有时间窗口保持不变——它编码的是 unit 的**稳定身份**（类似传统 Transformer 中位置编码编码的是 token 的位置），而 $X_i$ 携带的是**时变活动**。这种加法注入方式使得 $Z_i$ 同时包含了"谁在发放"（identity）和"发放了什么"（activity）的信息。

随后，$Z_i$ 通过 MLP 投影到 cross-attention 的输入空间，由**可学习的行为查询矩阵** $Q \in \mathbb{R}^{B \times W}$ 通过单层 cross-attention 解码出行为预测：

$$\hat{Y}_t = \text{MLP}_{out}(\text{CrossAttn}(Q, \text{LN}(Z_{in}), \text{LN}(Z_{in})))$$

整个架构在数学上保证了**置换不变性**：

$$\text{CrossAttn}(Q, P_R Z, P_R Z) = \text{CrossAttn}(Q, Z, Z)$$

其中 $P_R$ 是任意行置换矩阵。无论神经元排序如何，输出完全相同。此外，SPINT 采用**动态通道 dropout**（dynamic channel dropout）来增强对不同 session 间神经元组成变化的鲁棒性。

**跨 session 迁移零梯度：** 对于未见 session，只需在校准数据上运行训练好的 IDEncoder 前向传播，即可推断出所有 unit 的 identity embedding——无需梯度更新、无需标签数据。

**优势：**
- **SPINT 的置换不变性**是神经元对应问题目前最优雅的解决方案
- Spatial attention（STNDT）能发现功能重要的神经元子集
- IDEncoder 实现零梯度跨 session 迁移
- 轻量设计（单层 cross-attention + 两个三层 MLP），适合实时 BCI

**劣势：**
- Spatial attention 在神经元数量 N 上有 $O(N^2)$ 复杂度，大规模记录可能成为瓶颈
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

$$\mathbf{h}_i = E_{tok}(\text{neuron\_id}_i) + E_{pos}(i) + E_{temp}(\Delta t_i)$$

其中 $E_{tok}$ 是 neuron ID 的 embedding table（`nn.Embedding`），$E_{pos}$ 是 learnable position embedding（编码序列内位置索引），$E_{temp}$ 默认是 **sinusoidal temporal embedding**（编码连续时间偏移值 $\Delta t$，而非 learnable embedding）。也可选配 learnable temporal embedding，但代码默认使用正弦编码。

Neuroformer 的完整架构是一个**多模态系统**，包含：neural token embedding stem（即上述 spike 编码）、可选的 visual backbone（VideoEncoder/ResNet3D/ViT）、MultimodalTransformer（处理 neural-visual 跨模态 attention）、CLIP 模块（可选的跨模态对比学习）、以及独立的 head_id（预测下一个 neuron ID）和 head_dt（预测时间偏移）预测头。

**优势：**
- **稀疏性处理最优**：与 POYO 一样只编码有事件发生的时刻
- **唯一具备生成能力的方案**：作为自回归语言模型，可以生成条件合成 spike trains
- **高可解释性**：attention weights 直接反映神经元间的功能耦合，论文发现 attention maps 镜像了 Hebbian 连接性

**劣势：**
- 没有 PerceiverIO 式压缩，高发放率群体计算量大（$O(L^2)$）
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
| 序列压缩 | PerceiverIO（固定 latent） | 无压缩（$O(L^2)$ attention） |
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
| 计算效率 | ★★★★★ (固定长度) | ★★★★☆ (有压缩) | ★★★☆☆ ($O(N^2)$) | ★★☆☆☆ (无压缩) |
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

注入方式为**加法**：$\mathbf{h}_t \leftarrow \mathbf{h}_t + PE(t)$。NDT1 也支持 learnable position embedding（`nn.Embedding`）。STNDT 的两个 stream 各有独立的正弦位置编码——temporal PE 维度为 $D_T = N$、序列长度为 $T$；spatial PE 维度为 $D_S = T$、序列长度为 $N$。

Neuroformer [[14]](#ref14) 的时间编码默认也使用**正弦函数**（`TemporalEmbedding`），但编码的是**连续时间偏移值** $\Delta t$（而非离散索引），可选配 learnable temporal embedding。此外 Neuroformer 还有独立的 learnable position embedding（`nn.Parameter`）编码序列内位置索引。

**Learnable Position/Temporal Embedding。** MtM [[7]](#ref7) 使用 learnable position embedding（`nn.Embedding`），通过加法注入到 spike token 中。NDT3 [[6]](#ref6) 同时支持 learnable time embedding（加法注入）和 Rotary PE（在 attention 层内部注入）两种模式，以及一个 learnable **spatial embedding**（`nn.Embedding`）用于区分同一时间步内不同 spatial patch 的位置。NEDS [[8]](#ref8) 也使用 learnable temporal embedding。

**Rotary Position Embeddings (RoPE)。** POYO [[4]](#ref4)/POSSM [[13]](#ref13) 和 NDT3（可选模式）使用 RoPE。RoPE 不修改 token embedding 本身，而是在 attention 计算时旋转 key/query 向量，使 attention score 自然反映**相对**时间距离。POYO 的 RoPE 编码**连续时间戳**（以秒为单位），数学上：

$$\text{RoPE}(x_{2i-1}, x_{2i}, t) = \begin{pmatrix} x_{2i-1} \cos(\omega_i t) - x_{2i} \sin(\omega_i t) \\ x_{2i-1} \sin(\omega_i t) + x_{2i} \cos(\omega_i t) \end{pmatrix}$$

其中 $\omega_i = 2\pi / T_i$，$T_i$ 在 $[T_{min}, T_{max}]$ 上对数均匀分布（默认 $T_{min}=10^{-4}$, $T_{max}\approx 2.06$）。仅旋转 head 维度的一半（默认 head_dim=64 中旋转 32 维），另一半保持不变。NDT3 的 RoPE 则编码**离散时间步索引**。

**各项目时间编码方案总结：**

| 项目 | 时间编码类型 | 编码对象 | 注入方式 |
|------|------------|---------|---------|
| NDT1 | Sinusoidal PE / Learnable PE | 离散时间步索引 | 加法 |
| STNDT | Sinusoidal PE（两个独立的） | 离散时间步/神经元索引 | 加法 |
| NDT2 | 未使用显式时间编码 | — | — |
| NDT3 | Learnable time emb / Rotary PE + Learnable spatial emb | 离散时间步 + 空间位置 | 加法 / Attention 内旋转 |
| POYO/POSSM | Rotary PE | 连续时间戳（秒） | Attention 内旋转 |
| Neuroformer | Sinusoidal temporal emb（默认）+ Learnable position emb | 连续 $\Delta t$ + 序列索引 | 加法 |
| MtM | Learnable position emb | 离散时间步索引 | 加法 |
| NEDS | Learnable temporal emb | 离散时间步 | 加法 |

### 4.2 神经元身份编码

这是最关键的 embedding 选择，直接决定了模型的跨 session 能力。

**隐式位置编码（群体向量中的维度位置）。** NDT1 [[2]](#ref2) 的线性投影 $W_{in} \in \mathbb{R}^{D \times N}$ 隐式地将每个神经元映射到 embedding 空间的特定方向。第 $i$ 个神经元的 spike count 总是乘以 $W_{in}$ 的第 $i$ 列。这意味着神经元身份完全由输入维度的位置决定——换一个 session，维度含义就变了。

**Learnable Unit Embeddings。** POYO [[4]](#ref4)/POYO+ [[9]](#ref9) 使用 `InfiniteVocabEmbedding`，为每个 neural unit 分配可学习的 embedding 向量 $e_n \in \mathbb{R}^D$。支持动态词汇表扩展，新 session 的新 unit 可以运行时注册。新 session 需冻结主干，通过梯度下降重学习 embedding。CaPOYO 的 unit embedding 为半维度（$D/2$），与 value map 拼接。

**Neuron ID Embedding Table。** Neuroformer [[14]](#ref14) 使用固定大小的 `nn.Embedding`，将 neuron_id 映射为向量。词汇量在训练时确定，跨 session 能力受限于词汇表大小。

**Context-Dependent Positional Embedding / IDEncoder。** SPINT [[10]](#ref10) 的核心创新（详见 [Section 3.3](#33-neuron-as-token以神经元为-token)）。通过共享的双 MLP 网络从无标签校准数据动态推断 unit identity embedding $E_i$，并作为**位置编码**加到 spike 活动上。这些 embedding 反映了每个神经元在当前 session 中的功能角色（如发放率模式、时间相关特性等），而非固定的通道索引。

**Session/Context Tokens。** NDT2 [[5]](#ref5) 引入了 learnable session embedding、subject embedding 和 task embedding。注入方式有两种：(1) **Token 策略**：作为额外 token prepend 到序列首部，配合 flag 参数作为 type indicator；(2) **Concat 策略**：拼接到每个 token 的 embedding 后再投影。NDT3 [[6]](#ref6) 进一步加入 phase token（BCI vs. native control）和 return token（控制器质量，Decision Transformer 风格）。

**Session Embedding + Prompt Token。** MtM [[7]](#ref7) 使用 session embedding（`nn.Embedding`）和 prompt embedding（4 种 masking mode 各对应一个）。注入方式是作为**序列前缀 token**——prompt token 在第一个位置，session token 在第二个位置，使模型通过读取序列开头的 token 即可知道当前的 session 和 masking 任务类型。

**Session-Specific 投影。** NEDS [[8]](#ref8) 为每个 session 学习独立的线性投影 $W_{neural} \in \mathbb{R}^{N_{session} \times D}$，处理不同 session 间神经元数量不同的问题。所有 token 还添加 modality embedding 和 session embedding。

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

其中 $y_{t,n}$ 是真实 spike count，$\lambda_{t,n}$ 是模型预测的 Poisson rate（通过 softplus 保证正值）。选择 Poisson NLL 的理由是 spike count 是非负整数且方差约等于均值，Poisson 分布是合理的生成模型假设。局限性在于真实神经数据常常存在 over-dispersion（方差 > 均值），且在低 count 区域梯度信号很弱。

**Poisson-Softened Cross-Entropy。** NDT3 [[6]](#ref6) 的选择——这并非标准的 categorical cross-entropy，而是使用 **Poisson PMF 作为 soft target** 的改进版本。将 spike count 离散化后，目标分布不是 one-hot 向量，而是以真实 count 为均值的 Poisson PMF：

$$\mathcal{L} = -\sum_{k=0}^{K} q_k \log p_k, \quad q_k = \frac{e^{-y} y^k / k!}{\sum_{j=0}^{K} e^{-y} y^j / j!}$$

当 $y=0$ 时，$q_0 = 1$（退化为 one-hot）；当 $y=3$ 时，$q$ 在 $k=2,3,4$ 附近分散概率。这种设计让模型在"预测 2 还是 3"时的错误代价小于"预测 0 还是 3"。NDT3 代码中同时支持标准 Poisson NLL 和 Poisson-softened CE 两种 spike loss，由配置选择。

**Neuron ID + Temporal Cross-Entropy。** Neuroformer [[14]](#ref14) 的自回归 loss。预测下一个 spike 来自哪个神经元（neuron ID 分类）以及何时发放（time offset 分类）：$\mathcal{L} = \mathcal{L}_{neuron\_id} + \mathcal{L}_{temporal}$，这是唯一能驱动生成能力的 loss 设计。

**MSE（均方误差）。** 所有监督方法（POYO [[4]](#ref4), POSSM [[13]](#ref13), SPINT [[10]](#ref10)）用于预测连续行为变量（如手部速度）。NDT2/NDT3 在微调阶段也使用 MSE。NEDS 将 MSE 用于连续行为变量的重建。

**CTC Loss。** POSSM [[13]](#ref13) 在语音解码任务中使用 Connectionist Temporal Classification loss，这是语音识别的标准选择。

### 5.2 对比学习 Loss

**NT-Xent / InfoNCE。** STNDT [[3]](#ref3) 引入。对同一 trial 进行两种随机增强（时间裁剪、随机 dropout 神经元子集），要求两个 augmented view 在 embedding 空间中接近，不同 trial 的 view 远离：

$$\mathcal{L}_{contrastive} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$$

实际代码实现中 STNDT 将此称为 `info_nce_loss`，默认 temperature $\tau=0.07$。总 loss 为 $\mathcal{L} = \mathcal{L}_{masked\_recon} + \lambda \cdot \mathcal{L}_{contrastive}$，$\lambda$ 默认 $10^{-8}$。

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
