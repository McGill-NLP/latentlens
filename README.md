# LatentLens

[![arXiv](https://img.shields.io/badge/arXiv-2602.00462-b31b1b.svg)](https://arxiv.org/abs/2602.00462)
[![Demo](https://img.shields.io/badge/Demo-Interactive-orange.svg)](https://bennokrojer.com/vlm_interp_demo/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**LatentLens** interprets what hidden representations in LLM-based models (LLMs, VLMs, ...) encode by finding their nearest neighbors in a bank of contextual text embeddings. Unlike Logit Lens (which projects to vocabulary space), LatentLens compares against embeddings *in context* — yielding highly interpretable results.

Works with any HuggingFace model (LLMs, VLMs, etc.). No training required.

## Getting Started

```bash
pip install latentlens
```

**Option A: Build your own index of contextual embeddings** — point to any HuggingFace model + a text corpus:

```python
import latentlens

# Use the bundled concepts.txt (117k sentences, 23k WordNet concepts)
index = latentlens.build_index("meta-llama/Meta-Llama-3-8B", corpus="concepts.txt")
index.save("llama3_index/")

# Or use your own domain-specific corpus
index = latentlens.build_index("meta-llama/Meta-Llama-3-8B", corpus="my_texts.txt")
```

**Option B: Load a pre-built index** (we provide indices for popular models):

```python
index = latentlens.ContextualIndex.from_pretrained("McGill-NLP/contextual_embeddings-llama3.1-8b")
```

**Search** — pass any hidden states `[num_tokens, hidden_dim]` and get back interpretable nearest neighbors:

```python
results = index.search(hidden_states, top_k=5)
# results[i] = [Neighbor(token_str=' dog', similarity=0.42, contextual_layer=27), ...]

# Or search only specific contextual layers:
results = index.search(hidden_states, top_k=5, layers=[8, 27])
```

<p align="center">
  <img src="figures/method.png" width="80%" alt="LatentLens method overview">
</p>

## Full Example: Interpret Hidden States

```python
import latentlens

# Load any HuggingFace model
model, tokenizer = latentlens.load_model("Qwen/Qwen2-7B")

# Load or build a contextual index
index = latentlens.ContextualIndex.from_directory("qwen_index/")

# Get hidden states from your input
# hidden_states[0] = input embeddings, hidden_states[i] = output of transformer block i
inputs = tokenizer("a photo of a dog", return_tensors="pt").to("cuda")
hidden_states = latentlens.get_hidden_states(model, inputs["input_ids"])

# Interpret layer 27 — search auto-normalizes the query
results = index.search(hidden_states[27].squeeze(0), top_k=5)

for i, neighbors in enumerate(results):
    token = tokenizer.decode(inputs["input_ids"][0, i])
    nn = neighbors[0]
    print(f"{token:>10} → {nn.token_str!r} (sim={nn.similarity:.2f}, layer={nn.contextual_layer})")
```

### VLM Example: Interpret Visual Tokens

For VLMs, pass model-specific inputs (e.g., `pixel_values`) as keyword arguments to `get_hidden_states()`:

```python
import latentlens
from transformers import AutoProcessor

# Load a VLM and its processor
model, tokenizer = latentlens.load_model("Qwen/Qwen2-VL-7B-Instruct", dtype=torch.float16)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Process image + text
inputs = processor(images=[image], text="Describe this image.", return_tensors="pt").to("cuda")

# Extract hidden states — pass all processor outputs
hidden_states = latentlens.get_hidden_states(model, **inputs)

# Find vision token positions and interpret them
# hidden_states[layer][0, vision_start:vision_end, :] contains the visual tokens
results = index.search(hidden_states[27][0, vision_start:vision_end, :], top_k=5)
```

## What You Need

| Component | What it is | How to get it |
|-----------|-----------|---------------|
| **A model** | Any HuggingFace LLM or VLM | `latentlens.load_model("model_name")` |
| **A contextual index** | Bank of text embeddings from that model | `build_index(model, corpus)` or `from_pretrained()` |
| **Hidden states to interpret** | Your tokens of interest | `latentlens.get_hidden_states(model, input_ids)` |

The index is built once and reused. See [Bundled Corpus](#bundled-corpus-conceptstxt) below for details on the included corpus, or provide your own domain-specific text.

**Tip:** If you already have a loaded model, pass it directly to avoid loading twice: `build_index("model", corpus, model=model, tokenizer=tokenizer)`

## Bundled Corpus: `concepts.txt`

We include `concepts.txt` — a general-purpose corpus of **117k sentences covering 23k WordNet concepts** (5 sentences per concept at varying lengths). All [pre-built indices](#pre-built-indices) are built from this corpus using prefix deduplication (identical prefixes produce identical embeddings in causal LMs, so we store each unique prefix only once).

You can also provide your own domain-specific corpus as a `.txt` file (one sentence per line), a `.csv` file (first column), or a Python list of strings:

```python
# Custom corpus
index = latentlens.build_index("meta-llama/Meta-Llama-3-8B", corpus="my_domain_texts.txt")
index = latentlens.build_index("meta-llama/Meta-Llama-3-8B", corpus=["sentence 1", "sentence 2"])
```

> **Relation to the paper:** The original LatentLens paper ([Krojer et al., 2026](https://arxiv.org/abs/2602.00462)) used ~3M Visual Genome phrases (`reproduce/vg_phrases.txt`) — a corpus tailored to interpreting visual tokens in VLMs. The library instead uses `concepts.txt`, which provides broad coverage of concepts humans care about, making it suitable for interpreting *any* LLM-based model, not just VLMs. We validate this improved corpus and extraction pipeline in a forthcoming companion paper. To reproduce the original paper results exactly, see [Reproducing Paper Results](#reproducing-paper-results).

## Pre-built Indices

We provide pre-computed contextual embeddings for popular LLMs and VLMs, built from the bundled `concepts.txt` corpus across 8 layers each. Browse all indices in our [HuggingFace Collection](https://huggingface.co/collections/McGill-NLP/latentlens-contextual-embeddings-6997a0e5a50d7999075ffef6).

**LLMs:**

| Model | HuggingFace Repo | Layers | Size |
|-------|-----------------|--------|------|
| **Llama-3.1-8B** | [`McGill-NLP/contextual_embeddings-llama3.1-8b`](https://huggingface.co/McGill-NLP/contextual_embeddings-llama3.1-8b) | 1, 2, 4, 8, 16, 24, 30, 31 | 32 GB |
| **Gemma-2-9B** | [`McGill-NLP/contextual_embeddings-gemma2-9b`](https://huggingface.co/McGill-NLP/contextual_embeddings-gemma2-9b) | 1, 2, 4, 8, 16, 24, 40, 41 | 15 GB |
| **Qwen2.5-7B** | [`McGill-NLP/contextual_embeddings-qwen2.5-7b`](https://huggingface.co/McGill-NLP/contextual_embeddings-qwen2.5-7b) | 1, 2, 4, 8, 16, 24, 26, 27 | 28 GB |

**VLMs** (text-only forward passes through the VLM's finetuned LLM backbone):

| Model | HuggingFace Repo | Layers | Size |
|-------|-----------------|--------|------|
| **Qwen2.5-VL-7B** | [`McGill-NLP/contextual_embeddings-qwen2.5-vl-7b`](https://huggingface.co/McGill-NLP/contextual_embeddings-qwen2.5-vl-7b) | 1, 2, 4, 8, 16, 24, 26, 27 | 28 GB |
| **Qwen3-VL-8B** | [`McGill-NLP/contextual_embeddings-qwen3-vl-8b`](https://huggingface.co/McGill-NLP/contextual_embeddings-qwen3-vl-8b) | 1, 2, 4, 8, 16, 24, 34, 35 | 32 GB |
| **Molmo2-8B** | [`McGill-NLP/contextual_embeddings-molmo2-8b`](https://huggingface.co/McGill-NLP/contextual_embeddings-molmo2-8b) | 1, 2, 4, 8, 16, 24, 34, 35 | 32 GB |

Load specific layers to save memory and download time:

```python
# Load only early + late layers (~2 layers instead of 8)
index = latentlens.ContextualIndex.from_pretrained(
    "McGill-NLP/contextual_embeddings-llama3.1-8b", layers=[1, 31]
)
```

## Quickstart Script (Qwen2-VL visual tokens)

For a self-contained demo interpreting visual tokens in Qwen2-VL (no library install needed):

```bash
python quickstart.py                            # uses bundled example.png
python quickstart.py --image path/to/image.jpg  # your own image
```

Pre-computed contextual embeddings are downloaded automatically from [HuggingFace](https://huggingface.co/McGill-NLP). Requires a GPU with >=24GB VRAM.

---

## Reproducing Paper Results

> **Note:** This section reproduces the original paper, which uses a different corpus (~3M Visual Genome phrases) and extraction pipeline (reservoir sampling, float8 storage) than the library above. The library's `build_index()` and pre-built indices use `concepts.txt` instead.

This section walks through reproducing our main results on visual token interpretability in VLMs.

### Overview

We study how frozen LLMs process visual tokens from vision encoders. We train MLP connectors mapping visual tokens to LLM embedding space, then analyze interpretability using three methods:

| Method | What it does |
|--------|--------------|
| **EmbeddingLens** | Nearest neighbors in LLM input embedding matrix |
| **LogitLens** | Apply LM head to intermediate representations |
| **LatentLens** (ours) | Nearest neighbors in *contextual* text embeddings |

### Step 1: Install Package

```bash
git clone https://github.com/McGill-NLP/latentlens.git
cd latentlens

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Step 2: Download Models

Downloads our trained MLP connectors from HuggingFace, then downloads and converts the base LLMs and vision encoders to Molmo's weight format (~50GB total).

**9 model configurations (3 LLMs × 3 vision encoders):**

| LLM / Vision | ViT-L/14-336 (CLIP) | DINOv2-L-336 | SigLIP-L |
|--------------|---------------------|--------------|----------|
| **OLMo-7B** | `olmo-vit` | `olmo-dino` | `olmo-siglip` |
| **LLaMA3-8B** | `llama-vit` | `llama-dino` | `llama-siglip` |
| **Qwen2-7B** | `qwen-vit` | `qwen-dino` | `qwen-siglip` |

We also support **Qwen2-VL-7B-Instruct** (off-the-shelf VLM, no connector needed).

```bash
# Download all models (connectors + base models)
./reproduce/step1_download.sh

# Or download just connector weights (~3GB)
./reproduce/step1_download.sh --connectors-only
```

**What gets downloaded and converted:**

| Component | Source | Size |
|-----------|--------|------|
| MLP Connectors (9) | `McGill-NLP/latentlens-connectors` | ~350MB each |
| OLMo-7B | `allenai/OLMo-7B-1024-preview` | ~14GB |
| LLaMA3-8B | `meta-llama/Meta-Llama-3-8B` | ~16GB |
| Qwen2-7B | `Qwen/Qwen2-7B` | ~14GB |
| ViT-L/14-336 | `openai/clip-vit-large-patch14-336` | ~1GB |
| DINOv2-L-336 | `facebook/dinov2-large` | ~1GB |
| SigLIP-L | `google/siglip-so400m-patch14-384` | ~1GB |

**Directory structure after download:**
```
checkpoints/           # Connector weights + model configs
├── olmo-vit/
│   ├── model.pt       # Connector weights (~350MB)
│   └── config.yaml    # Model architecture config
├── olmo-dino/
│   └── ...
└── ...

pretrained/            # Converted base models (Molmo format)
├── olmo-1024-preview.pt
├── llama3-8b.pt
├── qwen2-7b.pt
├── vit-l-14-336.pt
├── dinov2-large-336.pt
└── siglip-so400m-14-384.pt
```

### Step 3: Extract Contextual Embeddings

For LatentLens analysis, you need contextual text embeddings from each LLM. This is the most time-consuming step (~13h per LLM on a single GPU, processing ~3M Visual Genome phrases).

```bash
# Extract for all LLMs sequentially
./reproduce/step2_extract_contextual.sh

# Or for a specific LLM:
python reproduce/scripts/extract_embeddings.py \
    --model allenai/OLMo-7B-1024-preview \
    --layers 1 2 4 8 16 24 30 31 \
    --output-dir contextual_embeddings/olmo-7b
```

**Speed up with multiple GPUs:** The fastest approach is to run each LLM on a separate GPU in parallel, reducing wall time from ~40h to ~13h:

```bash
CUDA_VISIBLE_DEVICES=0 ./reproduce/step2_extract_contextual.sh olmo  &
CUDA_VISIBLE_DEVICES=1 ./reproduce/step2_extract_contextual.sh llama &
CUDA_VISIBLE_DEVICES=2 ./reproduce/step2_extract_contextual.sh qwen  &
wait
```

The script supports checkpointing — if interrupted, it resumes from the last saved progress.

### Step 4: Run Analysis

**LatentLens (contextual nearest neighbors):**
```bash
CUDA_VISIBLE_DEVICES=0 python reproduce/scripts/run_latentlens.py \
    --ckpt-path checkpoints/olmo-vit \
    --contextual-dir contextual_embeddings/olmo-7b/allenai_OLMo-7B-1024-preview \
    --visual-layer 0,1,2,4,8,16,24,30,31 \
    --num-images 300 \
    --output-dir results/latentlens/olmo-vit
```

**LogitLens:**
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python reproduce/scripts/run_logitlens.py \
    --ckpt-path checkpoints/olmo-vit \
    --layers 0,1,2,4,8,16,24,30,31 \
    --num-images 300 \
    --output-dir results/logitlens/olmo-vit

# Multi-GPU (optional, faster)
torchrun --nproc_per_node=4 reproduce/scripts/run_logitlens.py \
    --ckpt-path checkpoints/olmo-vit \
    --layers 0,1,2,4,8,16,24,30,31 \
    --num-images 300 \
    --output-dir results/logitlens/olmo-vit
```

**EmbeddingLens:**
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python reproduce/scripts/run_embedding_lens.py \
    --ckpt-path checkpoints/olmo-vit \
    --llm_layer 0,1,2,4,8,16,24,30,31 \
    --num-images 300 \
    --output-base-dir results/embedding_lens/olmo-vit

# Multi-GPU (optional, faster)
torchrun --nproc_per_node=4 reproduce/scripts/run_embedding_lens.py \
    --ckpt-path checkpoints/olmo-vit \
    --llm_layer 0,1,2,4,8,16,24,30,31 \
    --num-images 300 \
    --output-base-dir results/embedding_lens/olmo-vit
```

### Step 5: Evaluate Interpretability (Optional)

The paper's main results use GPT-5 to evaluate whether nearest neighbors are semantically related to image patches. This requires an OpenAI API key and costs ~$80-100 for full reproduction.

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Evaluate a single model (~$1 for 100 patches)
python reproduce/scripts/evaluate/evaluate_interpretability.py \
    --results-dir results/latentlens/olmo-vit \
    --images-dir /path/to/pixmo-cap/validation \
    --output-dir evaluation/latentlens/olmo-vit \
    --num-patches 100

# For SigLIP or DINOv2 models, pass --model-name so the evaluator
# uses the correct vision encoder grid size:
python reproduce/scripts/evaluate/evaluate_interpretability.py \
    --results-dir results/latentlens/olmo-siglip \
    --images-dir /path/to/pixmo-cap/validation \
    --output-dir evaluation/latentlens/olmo-siglip \
    --model-name olmo-siglip \
    --num-patches 100

# Aggregate results across models
python reproduce/scripts/evaluate/aggregate_results.py \
    --eval-dir evaluation/ \
    --output results/my_results.json
```

> **Note on `--model-name`:** The evaluation script needs to know the vision encoder to determine the correct patch grid size. Pass `--model-name` matching the model you are evaluating:
> - **CLIP (ViT-L/14) models** (`olmo-vit`, `llama-vit`, `qwen-vit`): No `--model-name` needed — CLIP's 24x24 grid is the default.
> - **SigLIP models** (`olmo-siglip`, `llama-siglip`, `qwen-siglip`): Pass `--model-name` containing "siglip" (e.g., `--model-name olmo-siglip`).
> - **DINOv2 models** (`olmo-dino`, `llama-dino`, `qwen-dino`): Pass `--model-name` containing "dinov2" (e.g., `--model-name olmo-dino`).
> - **Qwen2-VL**: Pass `--model-name qwen2vl`.

### Reproduce Main Results (all 9 models)

```bash
# Run all experiments
./reproduce/run_all.sh

# Or step by step:
./reproduce/step2_extract_contextual.sh  # ~40h sequential, ~13h with 3 GPUs in parallel
./reproduce/step3_run_analysis.sh        # ~13.5h (9 models × 3 methods)
```

---

## Model Configurations

| Model | LLM Layers | Vision Patches | Layers Analyzed |
|-------|------------|----------------|-----------------|
| OLMo-7B / LLaMA3-8B | 32 | 576 (24×24) | 0, 1, 2, 4, 8, 16, 24, 30, 31 |
| Qwen2-7B / Qwen2-VL | 28 | 729 (27×27) | 0, 1, 2, 4, 8, 16, 24, 26, 27 |

**Note:** SigLIP uses 27×27 patches (729 total), while CLIP and DINOv2 use 24×24 (576 total).

---

## Project Structure

```
├── quickstart.py             # Try LatentLens in 5 minutes (standalone)
├── latentlens/               # Library: build & search contextual indices
│   ├── index.py              # ContextualIndex, Neighbor, search, save/load
│   ├── extract.py            # build_index(), corpus loading, prefix dedup
│   └── models.py             # load_model(), get_hidden_states(), SUPPORTED_MODELS
├── molmo/                    # Molmo VLM infrastructure (for reproduction)
│   ├── model.py              # Model architecture with layer hooks
│   ├── config.py             # Configuration classes
│   └── data/                 # Image preprocessing, datasets
└── reproduce/                # Paper reproduction
    ├── scripts/              # Analysis scripts
    │   ├── run_latentlens.py
    │   ├── run_logitlens.py
    │   ├── run_embedding_lens.py
    │   ├── extract_embeddings.py
    │   └── evaluate/         # LLM judge evaluation
    ├── configs/              # Model configurations (YAML)
    ├── vg_phrases.txt        # Visual Genome phrases corpus
    ├── step1_download.sh
    ├── step2_extract_contextual.sh
    └── step3_run_analysis.sh
```

---

## Citation

```bibtex
@article{krojer2026latentlens,
  title={LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs},
  author={Krojer, Benno and Nayak, Shravan and Ma{\~n}as, Oscar and Adlakha, Vaibhav and Elliott, Desmond and Reddy, Siva and Mosbach, Marius},
  journal={arXiv preprint arXiv:2602.00462},
  year={2026}
}
```

---

## Acknowledgments

This project builds on [Molmo codebase](https://github.com/allenai/molmo) by the Allen Institute for AI. We thank them for releasing their code under the Apache 2.0 license.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
