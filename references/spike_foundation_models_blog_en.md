# Spike Neural Foundation Models: A Complete Technical Survey of Tokenization, Embedding, and Loss

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

**Solution A: Fixed Dimensional Encoding (NDT1 [[2]](#ref2)).** The simplest approach—a linear projection $W_{in} \in \mathbb{R}^{D \times N}$ hard-codes each neuron to a fixed direction in embedding space. Switching sessions completely changes the meaning of dimensions, making cross-session transfer impossible.

**Solution B: Learnable Unit Embedding (POYO [[4]](#ref4)).** Assigns a learnable embedding vector $e_n \in \mathbb{R}^D$ to each neuron. New sessions require freezing the backbone network and updating these embeddings via gradient descent. The advantage is explicit modeling of neuron identity; the disadvantage requires labeled calibration data and gradient updates.

**Solution C: Context-Dependent Positional Embedding / IDEncoder (SPINT [[10]](#ref10)).** Dynamically infers each unit's identity embedding from unlabeled calibration data through a shared MLP network and adds it as **context-dependent positional encoding** to spike tokens (see [Section 3.3](#33-neuron-as-token-以神经元为-token) and [Section 4.2](#42-神经元身份编码)). This is currently the most elegant solution, achieving zero-gradient cross-session transfer.

**Potential Direction One: Extending POYO's Learnable Unit Embedding to forward inference.** POYO currently assigns independent learnable embeddings to each unit, requiring gradient updates for new sessions. A natural extension is inspired by SPINT's IDEncoder approach—instead of maintaining independent embeddings for each unit, use a shared feedforward network to **directly forward-infer unit embeddings from raw calibration data**. Specifically, analogous to SPINT's IDEncoder, feed each unit's $M$ calibration trials of binned spike counts $X_n^{calib} \in \mathbb{R}^{M \times T}$ directly into the network, rather than manually extracting statistical features:

$$e_n = \psi\left(\frac{1}{M} \sum_{j=1}^{M} \phi(X_{n,j}^{calib})\right)$$

where $\phi$ and $\psi$ are shared multi-layer feedforward networks, and $X_{n,j}^{calib}$ is the raw binned spike counts of unit $n$'s $j$-th calibration trial. This end-to-end approach lets the network learn to extract meaningful identity features directly from raw data, completely avoiding the information bottleneck and inductive bias introduced by manually designed statistical features (such as firing rate distributions, ISI statistics, etc.). This amounts to grafting SPINT's IDEncoder module onto POYO's PerceiverIO architecture, giving it both POYO's spike-level temporal precision and SPINT's zero-gradient adaptation capability.

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

where $\mathbf{x}_t \in \mathbb{R}^N$ is the population spike count vector at timestep $t$. **Mode two** is per-neuron embedding—treat each neuron's spike count as a discrete variable, mapped to vectors via `nn.Embedding` lookup and concatenated:

$$\mathbf{h}_t = [E(x_{t,1}) \| E(x_{t,2}) \| \cdots \| E(x_{t,N})], \quad E: \{0,1,...,\text{max\_spikes}\} \to \mathbb{R}^{d}$$

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

$$\mathbf{h}_i^{input} = E_{unit}(\text{unit\_id}_i) + E_{type}(\text{token\_type}_i)$$

where $E_{unit}$ uses `InfiniteVocabEmbedding` (a learnable embedding supporting dynamic vocabulary expansion; new units in new sessions can be registered dynamically), and $E_{type}$ is embedding for 4 token types. Temporal information is injected via RoPE during attention computation (see [Section 4.1](#41-时间位置编码)).

Due to sequence length growing with spike count, POYO pairs with **PerceiverIO architecture** as a compression mechanism: variable-length spike token sequences are compressed to a fixed number of latent tokens (e.g., 256) via cross-attention; subsequent self-attention operates only on these latent tokens. The entire process has three stages:

1. **Encode**: Latent tokens aggregate input spike tokens' information via cross-attention
2. **Process**: Latent tokens perform self-attention among themselves (2-6 layers)
3. **Decode**: Output queries extract prediction-needed information from latent tokens via cross-attention

Notably, POYO's decoder side is **session-aware**—using `session_emb` to construct output query embedding, with different output queries for different sessions.

**CaPOYO's Calcium Imaging Extension:** POYO+ extends support to calcium imaging data through an independent CaPOYO model class. CaPOYO employs a **split-dim concatenation design** to explicitly decouple signal value and unit identity:

$$\mathbf{h}_i = [\underbrace{W_{val} \cdot \Delta F/F_i + b_{val}}_{\in \mathbb{R}^{D/2}} \; \| \; \underbrace{E_{unit}(\text{unit\_id}_i)}_{\in \mathbb{R}^{D/2}}]$$

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

**STNDT's Dual-Stream Design:** STNDT simultaneously constructs two views—temporal tokens (population vector per timestep, $[T, B, N]$) and spatial tokens (time series per neuron, transposed as $[N, B, T]$), processed separately via attention mechanisms then fused. Both streams have independent linear embedders and sinusoidal position encodings. Spatial attention reweights temporal features via matrix multiplication:

$$Z_{ST} = A_S \cdot Z_T^\top$$

where $A_S \in \mathbb{R}^{B \times N \times N}$ is the spatial attention weight matrix (after softmax), and $Z_T \in \mathbb{R}^{T \times B \times N}$ is the temporal representation. The fused $Z_{ST}$ passes through residual connection and FFN, allowing the model to learn "which neurons should be considered together."

**SPINT's Core Innovation—IDEncoder Dynamic Positional Encoding:** SPINT constructs a spatial token from each neural unit's $W$ time bins of binned spike counts, paired with its core innovation—**context-dependent positional encoding via IDEncoder**.

SPINT's IDEncoder uses no fixed position encoding (which would assume fixed neuron order) but dynamically infers each unit's identity from calibration data, **adding it as positional encoding to spike activity**. The specific process is as follows:

1. **Input**: Collect unit $i$'s $M$ calibration trials $X_i^C \in \mathbb{R}^{M \times T}$ (each trial interpolated to fixed length $T$, such as T=1024 for M1/H1)
2. **Per-trial encoding**: Process each trial through shared three-layer MLP $\phi$
3. **Cross-trial aggregation**: Average-pool representations across all trials
4. **Identity generation**: Generate final identity embedding through second three-layer MLP $\psi$

Mathematically:

$$E_i = \text{IDEncoder}(X_i^C) = \psi\left(\frac{1}{M} \sum_{j=1}^{M} \phi(X_{i,j}^C)\right)$$

where $\phi: \mathbb{R}^T \to \mathbb{R}^H$ and $\psi: \mathbb{R}^H \to \mathbb{R}^W$ are respectively two three-layer fully-connected networks, with $H$ as hidden dimension (M1: $H=1024$; M2: $H=512$; H1: $H=1024$) and $W$ as window size (corresponding to spike token dimension).

**Key Step—Identity Embedding Injected as Positional Encoding:** The generated $E_i$ is **directly added to each unit's spike activity window**:

$$Z_i = X_i + E_i$$

Here $X_i$ is unit $i$'s binned spike counts in the current decoding window, and $Z_i$ is the identity-aware representation. Note that $E_i$ remains constant across all time windows within the same session—it encodes the unit's **stable identity** (similar to how position encoding in traditional Transformers encodes token position), while $X_i$ carries **time-varying activity**. This additive injection makes $Z_i$ simultaneously contain both "who is firing" (identity) and "what was fired" (activity) information.

Subsequently, $Z_i$ is projected via MLP to cross-attention's input space, decoded to behavior predictions by **learnable behavior query matrix** $Q \in \mathbb{R}^{B \times W}$ through single-layer cross-attention:

$$\hat{Y}_t = \text{MLP}_{out}(\text{CrossAttn}(Q, \text{LN}(Z_{in}), \text{LN}(Z_{in})))$$

The entire architecture mathematically guarantees **permutation invariance**:

$$\text{CrossAttn}(Q, P_R Z, P_R Z) = \text{CrossAttn}(Q, Z, Z)$$

where $P_R$ is an arbitrary row permutation matrix. Output is identical regardless of neuron ordering. Additionally, SPINT employs **dynamic channel dropout** to enhance robustness to composition changes of neurons across sessions.

**Cross-Session Transfer with Zero Gradient:** For unseen sessions, simply run the trained IDEncoder in forward pass on calibration data to infer all units' identity embeddings—no gradient updates or labeled data needed.

**Advantages:**
- **SPINT's permutation invariance** is currently the most elegant solution to neuron correspondence
- Spatial attention (STNDT) can discover functionally important neuron subsets
- IDEncoder achieves zero-gradient cross-session transfer
- Lightweight design (single cross-attention layer + two three-layer MLPs), suitable for real-time BCI

**Disadvantages:**
- Spatial attention has $O(N^2)$ complexity in neuron count N, potentially bottleneck for large-scale recordings
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

$$\mathbf{h}_i = E_{tok}(\text{neuron\_id}_i) + E_{pos}(i) + E_{temp}(\Delta t_i)$$

where $E_{tok}$ is neuron ID embedding table (`nn.Embedding`), $E_{pos}$ is learnable position embedding (encoding position index in sequence), and $E_{temp}$ defaults to **sinusoidal temporal embedding** (encoding continuous time offset value $\Delta t$, not learnable embedding). Alternative learnable temporal embedding is optional, but code defaults to sinusoidal encoding.

Neuroformer's complete architecture is a **multimodal system** including: neural token embedding stem (the spike encoding described above), optional visual backbone (VideoEncoder/ResNet3D/ViT), MultimodalTransformer (handling neural-visual cross-modal attention), CLIP module (optional cross-modal contrastive learning), and independent head_id (predicting next neuron ID) and head_dt (predicting time offset) prediction heads.

**Advantages:**
- **Optimal sparsity handling**: Like POYO, only encodes moments with events
- **Only approach with generative capability**: As an autoregressive language model, can generate conditional spike train synthesis
- **High interpretability**: Attention weights directly reflect functional coupling between neurons; paper found attention maps mirror Hebbian connectivity

**Disadvantages:**
- No PerceiverIO-style compression; high firing rate populations incur large computation ($O(L^2)$)
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
| Sequence Compression | PerceiverIO (fixed latent) | No compression ($O(L^2)$ attention) |
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
| Computational Efficiency | ★★★★★ (fixed length) | ★★★★☆ (with compression) | ★★★☆☆ ($O(N^2)$) | ★★☆☆☆ (no compression) |
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

Injection method is **additive**: $\mathbf{h}_t \leftarrow \mathbf{h}_t + PE(t)$. NDT1 also supports learnable position embedding (`nn.Embedding`). STNDT's two streams each have independent sinusoidal position encodings—temporal PE with dimension $D_T = N$ and sequence length $T$; spatial PE with dimension $D_S = T$ and sequence length $N$.

Neuroformer [[14]](#ref14)'s temporal encoding also defaults to **sinusoidal functions** (`TemporalEmbedding`), but encodes **continuous time offset values** $\Delta t$ (not discrete indices), optionally paired with learnable temporal embedding. Additionally Neuroformer has independent learnable position embedding (`nn.Parameter`) encoding position indices within sequences.

**Learnable Position/Temporal Embedding.** MtM [[7]](#ref7) uses learnable position embedding (`nn.Embedding`), injected additively to spike tokens. NDT3 [[6]](#ref6) supports both learnable time embedding (additive injection) and Rotary PE (injection within attention layer) modes, plus a learnable **spatial embedding** (`nn.Embedding`) to distinguish positions of different spatial patches within the same timestep. NEDS [[8]](#ref8) also uses learnable temporal embedding.

**Rotary Position Embeddings (RoPE).** POYO [[4]](#ref4)/POSSM [[13]](#ref13) and NDT3 (optional mode) use RoPE. RoPE doesn't modify token embeddings but rotates key/query vectors during attention computation, making attention score naturally reflect **relative** temporal distance. POYO's RoPE encodes **continuous timestamps** (in seconds), mathematically:

$$\text{RoPE}(x_{2i-1}, x_{2i}, t) = \begin{pmatrix} x_{2i-1} \cos(\omega_i t) - x_{2i} \sin(\omega_i t) \\ x_{2i-1} \sin(\omega_i t) + x_{2i} \cos(\omega_i t) \end{pmatrix}$$

where $\omega_i = 2\pi / T_i$ with $T_i$ log-uniformly distributed on $[T_{min}, T_{max}]$ (default $T_{min}=10^{-4}$, $T_{max}\approx 2.06$). Only rotates half of head dimensions (default 32 of head_dim=64), leaving the other half unchanged. NDT3's RoPE encodes **discrete timestep indices**.

**Summary of Temporal Encoding Across Projects:**

| Project | Temporal Encoding Type | Encoding Target | Injection Method |
|------|------------|---------|---------|
| NDT1 | Sinusoidal PE / Learnable PE | Discrete timestep index | Additive |
| STNDT | Sinusoidal PE (two independent) | Discrete timestep/neuron index | Additive |
| NDT2 | No explicit temporal encoding | — | — |
| NDT3 | Learnable time emb / Rotary PE + Learnable spatial emb | Discrete timestep + spatial position | Additive / Attention rotation |
| POYO/POSSM | Rotary PE | Continuous timestamp (seconds) | Attention rotation |
| Neuroformer | Sinusoidal temporal emb (default) + Learnable position emb | Continuous $\Delta t$ + sequence index | Additive |
| MtM | Learnable position emb | Discrete timestep index | Additive |
| NEDS | Learnable temporal emb | Discrete timestep | Additive |

### 4.2 Neuron Identity Encoding

This is the most critical embedding choice, directly determining the model's cross-session capability.

**Implicit Positional Encoding (Dimensional Position in Population Vector).** NDT1 [[2]](#ref2)'s linear projection $W_{in} \in \mathbb{R}^{D \times N}$ implicitly maps each neuron to a specific direction in embedding space. The $i$-th neuron's spike count always multiplies the $i$-th column of $W_{in}$. This means neuron identity is completely determined by input dimension position—switching sessions changes dimension meanings.

**Learnable Unit Embeddings.** POYO [[4]](#ref4)/POYO+ [[9]](#ref9) use `InfiniteVocabEmbedding`, assigning learnable embedding vectors $e_n \in \mathbb{R}^D$ to each neural unit. Support dynamic vocabulary expansion; new units in new sessions can register at runtime. New sessions require freezing the backbone and relearning embeddings via gradient descent. CaPOYO's unit embedding is half-dimensional ($D/2$), concatenated with value map.

**Neuron ID Embedding Table.** Neuroformer [[14]](#ref14) uses fixed-size `nn.Embedding`, mapping neuron_id to vectors. Vocabulary determined at training time, limiting cross-session capability.

**Context-Dependent Positional Embedding / IDEncoder.** SPINT [[10]](#ref10)'s core innovation (detailed in [Section 3.3](#33-neuron-as-token-以神经元为-token)). Dynamically infers unit identity embedding $E_i$ from unlabeled calibration data through shared dual-MLP network and adds it **as positional encoding** to spike activity. These embeddings reflect each neuron's functional role in the current session (such as firing rate patterns, temporal correlation characteristics, etc.) rather than fixed channel indices.

**Session/Context Tokens.** NDT2 [[5]](#ref5) introduces learnable session embedding, subject embedding, and task embedding. Injection methods are: (1) **Token strategy**: prepend as additional tokens to sequence start, with flag parameters as type indicators; (2) **Concat strategy**: concatenate to each token embedding then project. NDT3 [[6]](#ref6) further adds phase tokens (BCI vs. native control) and return tokens (controller quality, Decision Transformer style).

**Session Embedding + Prompt Token.** MtM [[7]](#ref7) uses session embedding (`nn.Embedding`) and prompt embedding (one for each of 4 masking modes). Injection method is **sequence prefix token**—prompt token at first position, session token at second position, allowing the model to know current session and masking task type by reading sequence start tokens.

**Session-Specific Projection.** NEDS [[8]](#ref8) learns independent linear projections $W_{neural} \in \mathbb{R}^{N_{session} \times D}$ for each session, handling different neuron counts across sessions. All tokens also get modality embedding and session embedding.

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

where $y_{t,n}$ is true spike count and $\lambda_{t,n}$ is model-predicted Poisson rate (ensured positive via softplus). Poisson NLL is chosen because spike counts are non-negative integers with variance approximately equal to mean, making Poisson distribution a reasonable generative model assumption. Limitations are that real neural data often exhibits over-dispersion (variance > mean) and gradient signals are weak in low count regions.

**Poisson-Softened Cross-Entropy.** NDT3 [[6]](#ref6)'s choice—this isn't standard categorical cross-entropy but an improved version using **Poisson PMF as soft targets**. After discretizing spike counts, target distribution isn't one-hot vectors but Poisson PMF with true count as mean:

$$\mathcal{L} = -\sum_{k=0}^{K} q_k \log p_k, \quad q_k = \frac{e^{-y} y^k / k!}{\sum_{j=0}^{K} e^{-y} y^j / j!}$$

When $y=0$, $q_0 = 1$ (degenerates to one-hot); when $y=3$, $q$ spreads probability near $k=2,3,4$. This design makes predicting "2 vs 3" less costly than "0 vs 3." NDT3 code supports both standard Poisson NLL and Poisson-softened CE as spike loss, selected via configuration.

**Neuron ID + Temporal Cross-Entropy.** Neuroformer [[14]](#ref14)'s autoregressive loss. Predicts which neuron the next spike comes from (neuron ID classification) and when it fires (time offset classification): $\mathcal{L} = \mathcal{L}_{neuron\_id} + \mathcal{L}_{temporal}$. This is the only loss design driving generative capability.

**MSE (Mean Squared Error).** All supervised approaches (POYO [[4]](#ref4), POSSM [[13]](#ref13), SPINT [[10]](#ref10)) use MSE for predicting continuous behavior variables (such as hand velocity). NDT2/NDT3 also use MSE during fine-tuning. NEDS uses MSE for continuous behavior variable reconstruction.

**CTC Loss.** POSSM [[13]](#ref13) uses Connectionist Temporal Classification loss for speech decoding tasks, the standard choice for speech recognition.

### 5.2 Contrastive Learning Loss

**NT-Xent / InfoNCE.** Introduced by STNDT [[3]](#ref3). Apply two random augmentations (temporal cropping, random neuron subset dropout) to same trial, require two augmented views to be close in embedding space while different trial views are pushed apart:

$$\mathcal{L}_{contrastive} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$$

In actual code implementation STNDT calls this `info_nce_loss` with default temperature $\tau=0.07$. Total loss is $\mathcal{L} = \mathcal{L}_{masked\_recon} + \lambda \cdot \mathcal{L}_{contrastive}$, with $\lambda$ defaulting to $10^{-8}$.

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
