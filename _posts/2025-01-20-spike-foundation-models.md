---
layout: post
title: "Spike Neural Foundation Models"
title_cn: "Spike神经基础模型：Tokenization、Embedding与Loss的技术全景"
title_en: "Spike Neural Foundation Models: A Technical Survey of Tokenization, Embedding, and Loss"
date: 2025-01-20
bilingual: true
---

<div class="lang-cn" markdown="1">

> 当我们尝试用 Transformer 理解大脑的"语言"时，第一个问题就是——神经元的 spike 该怎么变成 token？

## 1. 为什么我们需要 Spike 数据的基础模型？

过去十年，基础模型（Foundation Models）在 NLP 和计算机视觉领域取得了巨大成功。GPT 系列证明了"大规模预训练 + 下游微调"范式的强大，CLIP 和 DINOv2 则展示了视觉表示学习的潜力。一个自然的问题随之而来：**我们能不能为大脑的神经活动数据做同样的事情？**

神经科学中，大规模电生理记录技术（如 Utah Array、Neuropixels）让我们能够同时记录数百甚至上千个神经元的 spiking 活动。这些数据蕴含着大脑计算的核心信息，广泛应用于脑机接口（BCI）、运动解码、视觉感知研究等领域。然而，传统方法（如 LFADS）往往针对单个 session 从头训练，无法跨 session、跨被试复用知识。

从 2021 年的 NDT1 到 2025 年的 NDT3、NEDS 和 SPINT，一系列工作尝试将 Transformer 架构引入神经 spiking 数据，逐步走向"神经数据基础模型"。这些工作面对的核心问题都一样：**spike 数据该怎么 tokenize？用什么 embedding 表示？训练目标（loss）该怎么设计？**

这篇文章将系统梳理这些技术选择，分析它们背后的动机、优势与局限，并探讨未来的发展方向。

---

## 2. Spike 数据面临的核心挑战

### 2.1 神经元对应问题（Neuron Correspondence Problem）

这是 spike 基础模型面临的最根本挑战。在 NLP 中，"cat" 这个 token 在所有文本中都代表猫；但在 spiking 数据中，**"通道 3"在不同 session 中可能记录的是完全不同的神经元**。

即使是同一个被试，电极漂移也会导致神经元在 session 间出现和消失。这意味着 spike 数据天然**缺乏跨 session 的"共享词汇表"**。

| 模态 | 标准化方案 | 对应难度 |
|------|-----------|---------|
| EEG  | 10-20 系统标准化电极安放 | 低 |
| fMRI | MNI 模板空间标准化 | 低 |
| Spiking | 无标准——每次植入捕获独特神经元集合 | **极高** |

跨 session 迁移存在三个递进的难度层级：同被试电极漂移、跨被试同脑区、跨脑区。

### 2.2 极端稀疏性（Extreme Sparsity）

典型皮层神经元的发放率仅 1–20 Hz。在 1ms 分辨率下，99% 以上的时间 bin 是空的。这导致大量 token 近乎全零，梯度信号很弱，模型容易学到"总是预测零"的 trivial 解。

### 2.3 时间分辨率与效率的权衡

Spike 数据的关键优势是毫秒级时间精度，但保留这种精度意味着极长的序列（1 秒 = 1000 个 1ms bins），与 Transformer 的二次复杂度直接冲突。

### 2.4 数据异质性与规模限制

与 NLP/视觉领域相比，spiking 数据集小了好几个数量级。更棘手的是，即使有限的数据也高度异质——不同实验室使用不同的记录设备、处理算法和实验范式。有研究发现，精选 5 个 session 的数据可能比使用全部 84 个 session 的效果更好。

---

## 3. Tokenization：如何把 Spike 变成 Token？

### 3.1 Binned 群体向量（Population Vector per Time Bin）

**代表工作：** NDT1, NDT2, NDT3, MtM, NEDS

以固定时间窗口（通常 20ms）对 spike train 做 binning，统计每个神经元在每个 bin 内的 spike count，将每个时间步的完整群体向量通过线性投影映射为一个 token。

```
原始 spike trains → 20ms binning → N×T 矩阵 → 每列线性投影 → T 个 tokens
```

**优势：** 实现简单，序列长度固定，与标准 Transformer 直接兼容。

**劣势：** 稀疏性问题严重；时间信息丢失（20ms 内精细结构被抹掉）；神经元对应最不友好。

### 3.2 单 Spike Tokenization（Per-Spike Token）

**代表工作：** POYO, POYO+, POSSM

POYO 开创了最"原生"的表示：**每个单独的 spike 事件成为一个 token**，完全不做时间 binning。每个 spike token 由 learnable unit embedding（标识神经元）加上 RoPE 编码的精确时间戳组成。通过 PerceiverIO 架构将变长 spike 序列压缩到固定数量的 latent token。

**优势：** 完美处理稀疏性；保留毫秒级时间精度；PerceiverIO 优雅处理神经元数量变化。

### 3.3 Neuron-as-Token（以神经元为 Token）

**代表工作：** STNDT, SPINT

将每个神经元的完整时间序列作为一个 spatial token。SPINT 引入 **IDEncoder 动态位置编码**——不使用固定位置编码，而是通过共享 MLP 从校准数据动态生成每个 unit 的 identity embedding，保证**置换不变性**，零梯度跨 session 迁移。

### 3.4 Spike 事件对（Spike Event Pairs）

**代表工作：** Neuroformer

将每个 spike event 编码为 **(neuron_id, time_interval_offset)** 二元组，类似 NLP 中的词。唯一具备生成能力的方案，attention weights 直接反映神经元间的功能耦合。

### 3.5 四种 Tokenization 对比

| 维度 | Binned 群体向量 | 单 Spike | Neuron-as-Token | Spike 事件对 |
|------|----------------|----------|-----------------|-------------|
| 时间精度 | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | ★★★★☆ |
| 稀疏性处理 | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★★★ |
| 计算效率 | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ |
| 神经元对应 | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |
| 生成能力 | ★★★☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |

---

## 4. Embedding：如何赋予 Token 身份和上下文？

**时间位置编码：** 从 NDT1 的正弦编码，到 POYO 的 RoPE（在 attention 计算时旋转 key/query，保留连续时间戳精度），再到 Neuroformer 的 learnable temporal embedding。

**神经元身份编码的演进：**

| 方案 | 对未见 session | 所需校准数据 |
|------|--------------|------------|
| 隐式维度位置 (NDT1) | 不可迁移 | — |
| Session tokens (NDT2/3) | 需微调 | 有标签数据 |
| Learnable unit emb (POYO) | 梯度更新 embedding | 有标签数据 |
| IDEncoder (SPINT) | 前向传播 | **无标签数据** |

SPINT 的 IDEncoder 方案最优——对未见 session 只需无标签校准数据和一次前向传播。

---

## 5. Loss 设计

**Poisson NLL：** 最经典的选择（NDT1/2, STNDT, MtM, NEDS）。将 spike count 建模为泊松分布，在低 count 区梯度信号弱。

**Categorical Cross-Entropy：** NDT3 将 spike count 离散化为类别，避免分布假设，训练更稳定。

**对比学习（NT-Xent）：** STNDT 引入，对同一 trial 的两种增强视图做对比，学习 trial 级别的全局表示。

**多任务遮盖（MtM）：** 定义四种互补的遮盖方案（causal、neuron、intra-region、inter-region masking）交替训练，是目前最综合的自监督方案。

---

## 6. 各模型技术方案一览

| 模型 | 年份 | Tokenization | 关键 Embedding | Loss | 预训练范式 |
|------|------|-------------|---------------|------|-----------|
| NDT1 | 2021 | Binned 群体向量 | Sinusoidal PE | Poisson NLL | Masked AE |
| NDT2 | 2023 | Binned 时空 patch | Context tokens | Poisson NLL | Multi-context masked AE |
| Neuroformer | 2024 | Spike 事件对 | Neuron emb + temporal | CE + InfoNCE | 自回归 + 对比 |
| POYO | 2023 | 单 spike token | Unit emb + RoPE | MSE | 纯监督 |
| MtM | 2024 | Binned patch | Session + prompt tokens | Poisson NLL | 四种 masking 交替 |
| NDT3 | 2025 | Binned + 离散化 | Context + phase + return | Categorical CE + MSE | 多模态自回归 |
| NEDS | 2025 | 多模态 tokenizers | Temporal + modality + session | Poisson NLL + MSE | 统一编码解码 |
| SPINT | 2025 | Neuron-as-token | IDEncoder 动态 identity | MSE | 纯监督 |

---

## 7. 技术演进脉络

**从单 Session 到大规模预训练：** NDT1（单 session）→ NDT2（12 被试）→ NDT3（30+ 被试，2000 小时）。但发现暴力 scaling 不是银弹——数据异质性是根本限制，需智能数据策划。

**从 Binning 到 Spike-Level 表示：** Binned 群体向量 → 时空 Patch (NDT2) → 单 Spike Token (POYO) → Spike + SSM 压缩 (POSSM)。

**从固定身份到动态身份：** 固定维度位置 → Learnable unit emb (POYO) → Session tokens (NDT2) → IDEncoder (SPINT)。

---

## 8. 展望

几条清晰的共识正在形成：

1. **Spike-level 表示优于 binning**，尤其在需要时间精度的应用中
2. **动态神经元身份编码**（如 IDEncoder）优于固定位置编码
3. **多任务、多尺度训练目标**比单一 loss 学到更丰富的表示
4. **SSM/循环架构**在实时应用中比纯 Transformer 更有前景
5. **智能数据策划**比暴力 scaling 更重要

下一个突破点，很可能来自将这些 insight 融合到一个统一框架中——既有 POYO 的时间精度、SPINT 的身份灵活性、MtM 的多尺度学习，又有 POSSM 的实时效率。

---

*本文覆盖的主要工作：NDT1 (2021), STNDT (2022), NDT2 (2023), Neuroformer (2024), POYO (2023), POYO+ (2025), MtM (2024), NDT3 (2025), NEDS (2025), POSSM (2025), SPINT (2025)*

</div>

<div class="lang-en" markdown="1">

> When we try to make Transformers understand the "language" of the brain, the first question is: how do we turn neuronal spikes into tokens?

## 1. Why Do We Need Spike Foundation Models?

The past decade has seen foundation models achieve remarkable success in NLP and computer vision. GPT demonstrated the power of large-scale pre-training followed by fine-tuning; CLIP and DINOv2 showed the potential of visual representation learning. A natural question follows: **can we do the same for the brain's neural activity data?**

In neuroscience, large-scale electrophysiology (Utah Arrays, Neuropixels) allows simultaneous recording of hundreds to thousands of neurons. These data carry core computational information from the brain and are widely used in brain-computer interfaces (BCI), motor decoding, and visual perception research. Yet traditional approaches (e.g., LFADS) train from scratch on single sessions, unable to reuse knowledge across sessions or subjects.

From NDT1 (2021) to NDT3, NEDS, and SPINT (2025), a series of works have sought to apply Transformer architectures to neural spiking data, moving toward "neural data foundation models." They all face the same core questions: **how to tokenize spike data? What embeddings to use? How to design the training objective (loss)?**

This post systematically reviews these technical choices, analyzing their motivations, strengths, and limitations.

---

## 2. Core Challenges in Spike Data

### 2.1 The Neuron Correspondence Problem

This is the most fundamental challenge for spike foundation models. In NLP, the token "cat" represents the same concept across all texts. But in spiking data, **"channel 3" may record completely different neurons in different sessions**.

Even for the same subject, electrode drift causes neurons to appear and disappear across sessions, meaning spike data naturally **lacks a shared vocabulary across sessions**.

| Modality | Standardization | Correspondence Difficulty |
|----------|----------------|--------------------------|
| EEG  | 10-20 standard electrode placement | Low |
| fMRI | MNI template space | Low |
| Spiking | None — each implant captures a unique neuron set | **Extremely high** |

Cross-session transfer has three levels of difficulty: same-subject electrode drift, cross-subject same-region, and cross-region.

### 2.2 Extreme Sparsity

Typical cortical neurons fire at only 1–20 Hz. At 1ms resolution, over 99% of time bins are empty. This causes most tokens to be near-zero, with weak gradient signals — models easily learn a trivial "always predict zero" solution.

### 2.3 Temporal Resolution vs. Efficiency Trade-off

A key advantage of spike data is millisecond-level temporal precision, but preserving this means very long sequences (1 second = 1000 bins at 1ms), which conflicts with Transformer's quadratic complexity.

### 2.4 Data Heterogeneity and Scale

Compared to NLP/vision, spiking datasets are orders of magnitude smaller. Even the available data is highly heterogeneous — different labs use different recording equipment, processing pipelines, and experimental paradigms. Research has found that curating 5 sessions can outperform using all 84 sessions.

---

## 3. Tokenization: How to Turn Spikes into Tokens?

### 3.1 Binned Population Vector

**Representative works:** NDT1, NDT2, NDT3, MtM, NEDS

The most straightforward approach: bin spike trains into fixed time windows (typically 20ms), count spikes per neuron per bin, then project each population vector to a token via linear mapping.

```
Raw spike trains → 20ms binning → N×T matrix → linear projection per column → T tokens
```

**Strengths:** Simple to implement; fixed sequence length; directly compatible with standard Transformers.

**Weaknesses:** Severe sparsity waste; loss of fine temporal structure; worst approach for neuron correspondence across sessions.

### 3.2 Per-Spike Tokenization

**Representative works:** POYO, POYO+, POSSM

POYO pioneered the most "native" representation: **each individual spike event becomes a token**, with no binning at all. Each spike token consists of a learnable unit embedding (identifying the neuron) plus a RoPE-encoded precise timestamp. A PerceiverIO architecture compresses the variable-length spike sequence into a fixed number of latent tokens.

**Strengths:** Perfect sparsity handling; preserves millisecond temporal precision; PerceiverIO elegantly handles variable neuron counts across sessions.

### 3.3 Neuron-as-Token

**Representative works:** STNDT, SPINT

Inverts the perspective: each neuron's complete time series becomes a spatial token. SPINT introduces the **IDEncoder** — instead of fixed positional encodings, a shared MLP dynamically generates each unit's identity embedding from calibration data, guaranteeing **permutation invariance** and zero-gradient cross-session transfer.

### 3.4 Spike Event Pairs

**Representative works:** Neuroformer

Encodes each spike event as a **(neuron_id, time_offset)** pair, analogous to words in NLP. The only approach with generative capability; attention weights directly reflect functional coupling between neurons.

### 3.5 Tokenization Comparison

| Dimension | Binned Vector | Per-Spike | Neuron-as-Token | Event Pairs |
|-----------|--------------|-----------|-----------------|-------------|
| Temporal precision | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | ★★★★☆ |
| Sparsity handling | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★★★ |
| Compute efficiency | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ |
| Neuron correspondence | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |
| Generative capability | ★★★☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |

---

## 4. Embedding: Identity and Context

**Temporal position encoding:** From NDT1's sinusoidal encoding to POYO's RoPE (rotating key/query vectors during attention, preserving continuous timestamp precision) to Neuroformer's learnable temporal embeddings.

**Neuron identity encoding evolution:**

| Approach | Unseen sessions | Required calibration |
|----------|----------------|---------------------|
| Implicit dimension position (NDT1) | Not transferable | — |
| Session tokens (NDT2/3) | Requires fine-tuning | Labeled data |
| Learnable unit embeddings (POYO) | Gradient update | Labeled data |
| IDEncoder (SPINT) | Forward pass only | **Unlabeled data** |

SPINT's IDEncoder is optimal — it requires only unlabeled calibration data and a single forward pass for unseen sessions.

---

## 5. Loss Function Design

**Poisson NLL:** The classic choice (NDT1/2, STNDT, MtM, NEDS). Models spike counts as a Poisson distribution. Gradient signals are weak in low-count regions.

**Categorical Cross-Entropy:** NDT3 discretizes spike counts into categories, avoiding distributional assumptions and yielding more stable training.

**Contrastive Learning (NT-Xent):** STNDT applies this to learn trial-level global representations by pulling together two augmented views of the same trial.

**Multi-task Masking (MtM):** Defines four complementary masking schemes (causal, neuron, intra-region, inter-region) trained alternately — currently the most comprehensive self-supervised approach.

---

## 6. Model Comparison

| Model | Year | Tokenization | Key Embedding | Loss | Training |
|-------|------|-------------|---------------|------|---------|
| NDT1 | 2021 | Binned vector | Sinusoidal PE | Poisson NLL | Masked AE |
| NDT2 | 2023 | Spatiotemporal patch | Context tokens | Poisson NLL | Multi-context masked AE |
| Neuroformer | 2024 | Spike event pairs | Neuron emb + temporal | CE + InfoNCE | Autoregressive + contrastive |
| POYO | 2023 | Per-spike token | Unit emb + RoPE | MSE | Supervised |
| MtM | 2024 | Binned patch | Session + prompt tokens | Poisson NLL | 4-way masking |
| NDT3 | 2025 | Binned + discretized | Context + phase + return | Categorical CE + MSE | Multimodal autoregressive |
| NEDS | 2025 | Multi-modal tokenizers | Temporal + modality + session | Poisson NLL + MSE | Unified encode/decode |
| SPINT | 2025 | Neuron-as-token | IDEncoder dynamic identity | MSE | Supervised |

---

## 7. Technical Evolution

**From single-session to large-scale pretraining:** NDT1 (single session) → NDT2 (12 subjects) → NDT3 (30+ subjects, 2000 hours). But brute-force scaling is not a silver bullet — data heterogeneity is the fundamental bottleneck, requiring intelligent curation over raw volume.

**From binning to spike-level representation:** Binned population vectors → spatiotemporal patches (NDT2) → per-spike tokens (POYO) → spikes + SSM compression (POSSM) for real-time inference.

**From fixed to dynamic identity:** Fixed dimension position → learnable unit embeddings (POYO) → session tokens (NDT2) → IDEncoder (SPINT, zero-gradient transfer).

---

## 8. Future Directions

Several key insights are crystallizing:

1. **Spike-level representations outperform binning**, especially when temporal precision matters
2. **Dynamic neuron identity encoding** (IDEncoder) outperforms fixed positional encodings or lookup tables
3. **Multi-task, multi-scale training objectives** learn richer representations than single losses
4. **SSM/recurrent hybrid architectures** are more promising than pure Transformers for real-time BCI
5. **Intelligent data curation** matters more than brute-force scaling

The next breakthrough will likely come from unifying these insights: the temporal precision of POYO, the identity flexibility of SPINT, the multi-scale learning of MtM, and the real-time efficiency of POSSM — in a single coherent framework. This is not just an engineering challenge but a fundamental scientific question: **what is the universal representation of brain computation?**

---

*Models covered: NDT1 (2021), STNDT (2022), NDT2 (2023), Neuroformer (2024), POYO (2023), POYO+ (2025), MtM (2024), NDT3 (2025), NEDS (2025), POSSM (2025), SPINT (2025)*

</div>
