# Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer

Z-Image Team, Alibaba Group  
arXiv:[2511.22699](https://arxiv.org/abs/2511.22699) (v5, 2026-07-06) · CC BY 4.0

| | |
| --- | --- |
| GitHub | https://github.com/Tongyi-MAI/Z-Image |
| Hugging Face | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo |
| ModelScope | https://modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo |
| Demos | [Hugging Face](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) · [ModelScope](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=469191&modelType=Checkpoint&sdVersion=Z_IMAGE_TURBO&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image-Turbo%3Frevision%3Dmaster) |
| Gallery | [Online](https://modelscope.cn/studios/Tongyi-MAI/Z-Image-Gallery/summary) · [PDF](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo/file/view/master/assets/Z-Image-Gallery.pdf) |

## Abstract

Open T2I models in the 20B–80B range (Qwen-Image, FLUX.2, Hunyuan-Image-3.0) are impractical to train or serve on consumer hardware; closed systems (Nano Banana Pro, Seedream 4.0) are not reproducible. Distilling proprietary teachers is cheap but homogenizes data and caps visual novelty.

**Z-Image** is a 6B single-stream diffusion transformer (S3-DiT) trained only on real-world data. The full pipeline costs **314K H800 GPU hours (~$628K)**. Few-step distillation plus reward post-training yields **Z-Image-Turbo**: 8 NFE, sub-second latency on H800 (FlashAttention-3 + `torch.compile`), **&lt;16 GB VRAM**. Omni-pre-training also yields **Z-Image-Edit**. Strengths: photorealism and bilingual (ZH/EN) text rendering at a fraction of competitor compute.

## Family and cost

| Variant | Role |
| --- | --- |
| Z-Image | 6.15B S3-DiT foundation (T2I) |
| Z-Image-Turbo | 8-NFE distilled + RL; no CFG at official inference |
| Z-Image-Edit | Instruction editing from the omni-pretrained backbone |

| Stage | H800 GPU hours | USD (@ $2/h) |
| --- | --- | --- |
| Low-res pre-training | 147.5K | $295K |
| Omni-pre-training | 142.5K | $285K |
| Post-training | 24K | $48K |
| **Total** | **314K** | **$628K** |

Four design pillars: curated data infrastructure, compact single-stream architecture, progressive curriculum, few-step distillation (Decoupled DMD + DMDR).

---

## 1. Data infrastructure

Goal under a fixed compute budget: conceptually broad, non-redundant, multilingual-aligned data, structured for curriculum learning. Four coupled modules.

### 1.1 Data Profiling Engine

Multi-dimensional features over a large internal copyrighted pool, with source-specific heuristics.

- **Metadata:** resolution, file size, aspect ratio; perceptual hash for near-duplicate removal.
- **Technical quality:** compression ratio vs uncompressed size; in-house scores for color cast, blur, watermarks, noise; low-entropy filter (border-pixel variance + JPEG bytes-per-pixel as complexity proxy).
- **Semantics / aesthetics:** professional-annotator aesthetic scorer; AIGC classifier (Imagen 3-style, to protect photorealism); VLM tags (objects, people counts, Chinese-culture concepts, NSFW).
- **Cross-modal:** CN-CLIP image–alt-text correlation; VLM multi-level captions (tags, short, long). OCR and watermarks are prompted into the caption rather than run as separate modules.

### 1.2 Cross-modal Vector Engine

SD3-style semantic dedup recast as GPU k-NN proximity graph + community detection (~8 h / 1B items on 8×H800, including index + 100-NN). Modularity of clusters also supports fine-grained balancing.

Multimodal retrieval (CN-CLIP + ANN) fills conceptual gaps and, given failure cases, locates and prunes the training clusters that caused them.

### 1.3 World Knowledge Topological Graph

1. Wikipedia entities + hyperlinks, pruned by PageRank (drop isolates) and a VLM “visual generatability” filter (drop abstract concepts).
2. Hierarchical clustering of caption-tag embeddings from the internal pool; VLM names parent nodes.
3. Manual up-weight of frequent user-prompt concepts; inject trending entities.

Sampling weight per example: BM25 of mapped tags plus parent–child structure. Used for staged, concept-balanced draws from the pool.

### 1.4 Active Curation Engine

Uncurated pool → embed, dedup, rule filters. Z-Image itself diagnoses long-tail failures (e.g. 松鼠鳜鱼 as a dish vs “squirrel + fish”) and drives targeted retrieval.

Human-in-the-loop: topology + reward model sample a balanced unlabeled subset → captioner/reward assign pseudo-labels → human+AI verify, humans correct rejects → retrain captioner and reward model.

### 1.5 Editing pairs

- **Experts + mixed edits:** taxonomy of tasks, task-specific experts, several operations packed into one pair.
- **Graph on one source:** one input + *N* edits → \(\binom{N+1}{2}\) (×2 with inverses) pairs by permuting versions: mixed edits, inverse (synthetic→real) pairs, zero extra generation cost.
- **Video frames:** naturally related groups; CN-CLIP cosine filter. High diversity, coupled pose/background/style changes, scalable.
- **Text edits:** controllable renderer (font, color, size, position) so the instruction is the known render op. Natural images are too sparse/imbalanced for this.

---

## 2. Z-Captioner

One VLM for T2I captions and edit instructions. Shared image-understanding objective; world knowledge is injected so named entities render correctly.

**OCR-first CoT.** Recognize all visible text in the original language (no translation), then caption. Direct “describe everything” captions miss dense/long text; OCR in the caption is tightly correlated with accurate generated rendering.

**Five caption types,** all conditioned on meta/world knowledge to cut hallucination of people, landmarks, events:

| Type | Role |
| --- | --- |
| Long | Dense, factual, OCR-complete; plain style; no subjective filler |
| Medium | Intermediate |
| Short | Relatively complete scene summary |
| Tags | Concept coverage |
| Simulated user prompts | Short, incomplete, focus on one interest — match real users |

**Difference captions (editing), three-step CoT:** (1) OCR-inclusive caption of source and target; (2) visual+textual discrepancy analysis; (3) concise instruction.

---

## 3. Architecture (S3-DiT)

Inspired by decoder-only LLMs: one stream, dense cross-modal interaction every layer, vs dual-stream MM-DiT that isolates text and image.

| Config | Value |
| --- | --- |
| Parameters | 6.15B |
| Layers | 30 |
| Hidden dim | 3840 |
| Attention heads | 32 |
| FFN intermediate | 10240 |
| 3D RoPE \((d_t, d_h, d_w)\) | (32, 48, 48) |

**Encoders.** Text: Qwen3-4B (bilingual). Image latents: Flux VAE. Editing only: SigLIP 2 semantic tokens from the reference image.

**Sequence.** Lightweight 2-block modality processors, then concatenate text / semantic / VAE tokens. 3D unified RoPE: image tokens expand in space, text along time. For editing, reference and target share spatial RoPE but are offset by one unit in time; they also get different time-conditioning (clean vs noisy).

**Stability / cond.** QK-Norm on attention; Sandwich-Norm on attn/FFN I/O; RMSNorm throughout. Condition vectors → scale and gate via a **shared down-projection + per-layer up-projection** (low-rank, fewer params).

---

## 4. Training

### 4.1 Systems

- Frozen VAE and text encoder: data parallel. DiT: **FSDP2** (shard optimizer + grads).
- Gradient checkpointing on all DiT layers; **`torch.compile`** on blocks.
- Mixed-resolution: precompute sequence length from stored H×W; pack similar lengths; **dynamic batch size** (small for long seq, large for short) to cut padding and OOM.

### 4.2 Flow matching

Linear path \(x_t = t\,x_1 + (1-t)\,x_0\), predict velocity \(v_t = x_1 - x_0\):

\[
\mathcal{L} = \mathbb{E}_{t,x_0,x_1,y}\big[\|u(x_t,y,t;\theta) - (x_1-x_0)\|^2\big].
\]

Logit-normal timestep sampler (SD3). Multi-resolution SNR: Flux-style **dynamic time shifting**.

### 4.3 Pre-training

**Low-res (≈ half of pre-train compute).** Fixed \(256^2\), T2I only. Cross-modal alignment and concept/style/composition coverage, including Chinese text rendering.

**Omni-pre-training** (several internal stages):

1. **Arbitrary resolution.** Map native size into a training range; mixed AR. Less downsample loss, better data efficiency. After the last stage: up to ~1k–1.5k, image+text conditioning.
2. **Joint T2I + I2I.** Weakly aligned natural pairs from §1.5, paid for by the pre-train budget. Strong edit init; **no observed T2I regression**.
3. **Bilingual multi-level captions** from Z-Captioner, plus original metadata with small probability (world knowledge). For I2I, random choice of target caption vs difference caption (reference-guided gen vs multi-task edit).

### 4.4 SFT

- **Distribution narrowing.** Curated images + super-detailed grounded captions; drop low-quality modes; quality over diversity.
- **Concept balancing.** Graph + on-the-fly BM25 rarity; up-weight long-tail, down-weight heads — avoid forgetting while converging.
- **Model merging.** Several SFT runs from the same backbone with slight capability biases; linear mix \(\theta_{\mathrm{final}} = \sum_i \alpha_i\theta_i\) for robustness without routing.

PE-aware SFT: all prompts (and edit images) go through the Prompt Enhancer so the 6B DiT aligns to PE outputs without training the VLM.

### 4.5 Few-step distillation → Turbo

SFT teacher: ~**100 NFE with CFG**. Target: **8 NFE**. Vanilla DMD (distribution matching distillation) lost high-frequency detail and shifted color.

**Decoupled DMD.** DMD is two mechanisms, not one:

| Term | Role |
| --- | --- |
| CFG-augmentation (CA) | Main driver of few-step skill (under-discussed in prior work) |
| Distribution matching (DM) | Regularizer: stability, artifact suppression |

Separate renoising schedules for CA vs DM. Restores sharpness and color; distilled student can beat the 100-step teacher on photorealism.

**DMDR.** RL on the student + DM as **intrinsic** regularizer against reward hacking (instead of extra external penalties). Aesthetic/semantic lift without collapse.

Result: Turbo 8-step often matches or exceeds the teacher in perceived quality.

### 4.6 RLHF

Multi-dimensional reward: instruction following, AIGC-perception, aesthetics. Instruction score: prompt decomposed into subject / attributes / actions / spatial / style; raters click unsatisfied elements; fraction satisfied is the target.

**Stage 1 — DPO (objective axes).** Text rendering, counting, etc. VLMs propose chosen/rejected pairs; humans clean. Curriculum: simple → hard; start with moderate pair gaps, then subtler/larger diffs (DPO is sensitive to gap size).

**Stage 2 — GRPO (online).** Composite advantage from the reward model (realism, aesthetics, instruction, …). Multi-signal better than a single reward for competing qualities.

### 4.7 Z-Image-Edit continued training

From the omni backbone + T2I SFT mix (quality floor).

1. All edit data at \(512^2\) for a few thousand steps, then \(1024^2\).
2. **T2I : I2I ≈ 4 : 1** — edit pairs are scarce; too little T2I degrades quality.
3. SFT on a task-balanced human subset. Rendered text-edit pairs are instruction-perfect but OOD vs users → **heavily downsampled** here.

### 4.8 Prompt Enhancer

6B DiT is a strong visual decoder, weak at world knowledge / intent / planning. PE = frozen pretrained VLM + system prompt + **structured reasoning chain** (subject analysis → knowledge/problem-solving → aesthetics → full description).

Without reasoning, GPS coords are painted as text; with reasoning, the scene (e.g. West Lake) is inferred. Same for multi-step procedures (tea brewing) and ambiguous edit prompts (“design a poster”). Alignment is PE-aware SFT, not VLM finetuning.

---

## 5. Evaluation

Qualitative (Turbo unless noted): photoreal close-ups and phone-like scenes; bilingual posters/couplets on par with Nano Banana Pro; Edit supports add/remove/replace, bbox/scribble localization with shadow/refraction consistency, yaw/pitch novel views (VGGT-checkable geometry), multi-image ID, composite bilingual instructions. PE enables puzzles, poetry, coords. Multilingual prompts and local cultural landmarks emerge from bilingual training.

### 5.1 Human preference

**Artificial Analysis Image Arena.** Turbo Elo **1161**, **8th overall**, **1st open-source**; among top-10, smallest (6B) and cheapest (**$5.0 / 1k images**).

**Alibaba AI Arena (T2I).** Turbo **4th** globally, 1st OSS: Elo **1025**, win rate **45%** (Imagen 4 Ultra 1048 / 48%, Seedream 4.0 1039 / 46%, Qwen-Image 20B 1008 / 41%).

**Z-Image vs Flux 2 dev (32B).** 222 user-style prompts, 3 annotators: Good **46.4%**, Same **41.0%**, Bad **12.6%** → G+S **87.4%**.

### 5.2 Automated T2I (headline numbers)

| Benchmark | Z-Image | Turbo | Note |
| --- | --- | --- | --- |
| CVTG-2K word acc. / CLIP | **0.8671** / 0.7969 | 0.8585 / **0.8048** | EN text; #1 / #2 vs GPT-Image-1 0.8569, Qwen-Image 0.8288 |
| LongText-Bench EN / ZH | 0.935 / 0.936 | 0.917 / 0.926 | ZH #2 after Qwen-Image 0.946 |
| OneIG-EN overall (Text) | **0.546** (**0.987**) | 0.528 (0.994) | EN SOTA overall; Turbo Text 0.994 |
| OneIG-ZH overall (Text) | 0.535 (**0.988**) | 0.507 (0.982) | #2 after Qwen-Image 0.548 |
| GenEval overall | 0.84 | 0.82 | Tie #2 with Seedream 3.0 / GPT-Image-1; Qwen-Image 0.87 |
| DPG-Bench overall (Attr.) | 88.14 (**93.16**) | 84.86 | #3; Attribute above Qwen-Image 92.02 |
| TIIF testmini overall | 80.20 (#4) | 77.73 (#5) | GPT-Image-1 89.15, Qwen-Image 86.14 |
| PRISM-Bench EN overall | — | **77.4** (#3) | Ahead of base and Qwen-Image |
| PRISM-Bench ZH overall | **75.3** (#2) | — | Text 83.4, Composition 88.6 |

### 5.3 Editing

| Benchmark | Z-Image-Edit | Rank |
| --- | --- | --- |
| ImgEdit overall | 4.30 | 3rd (UniWorld-V2 4.49, Qwen-Image-Edit 2509 4.35). Strong Add 4.40, Extract 4.30 |
| GEdit-EN G_O | 7.57 | 3rd (UniWorld-V2 7.83, Qwen 2509 7.54) |
| GEdit-CN G_O | 7.54 | Tied with Qwen 2509 |

---

## 6. Conclusion

A 6B S3-DiT, trained end-to-end on real data for &lt;$630K, matches much larger and closed systems on photorealism and bilingual text. Turbo is 8-NFE and consumer-VRAM; Edit reuses omni-pre-training. The reusable pieces are the data stack, single-stream design, PE-aware SFT, Decoupled DMD, and DMDR — not scale-at-all-costs.
