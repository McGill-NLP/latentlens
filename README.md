# LatentLens

[![arXiv](https://img.shields.io/badge/arXiv-2602.00462-b31b1b.svg)](https://arxiv.org/abs/2602.00462)
[![Demo](https://img.shields.io/badge/Demo-Interactive-orange.svg)](https://bennokrojer.com/vlm_interp_demo/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


**LatentLens** interprets (continuous) token representations in language models by finding their nearest neighbors in contextual text embedding space. In our paper we specifically study visual tokens and find they are highly interpretable under LatentLens (but e.g. not with Logit Lens). We imagine it can be used beyond vision, across many other domains.

<p align="center">
  <img src="figures/method.png" width="80%" alt="LatentLens method overview">
</p>

---

## This Repository

- [x] [Quickstart](#quick-start) for Qwen2-VL visual token interpretation
- [x] General-purpose [`latentlens` library](#library-api) for any LLM
- [x] [Paper reproduction](#reproducing-paper-results) scripts and configs

1. **[Using LatentLens](#using-latentlens)** — Analyze continuous token representations in any LLM: go here for your custom use case, or to get started quickly
2. **[Library API](#library-api)** — Build and search contextual embedding indices programmatically
3. **[Reproducing Paper Results](#reproducing-paper-results)** — Reproduce our main results on visual tokens: go here if you want to reproduce our paper

---

## Using LatentLens

LatentLens works by comparing continuous token representations against a bank of contextual text embeddings. The core idea:

1. **Build a contextual embedding bank** by running your LLM on text data
2. **Find nearest neighbors** to interpret what the continuous tokens of your interest encode

### Quick Start

Try LatentLens on **Qwen2-VL-7B-Instruct** — an off-the-shelf VLM that requires no connector training. The script uses only `transformers` and `torch` (no `latentlens` package imports needed):

```bash
python quickstart.py                            # uses bundled example.png
python quickstart.py --image path/to/image.jpg  # your own image
```

On first run, pre-computed contextual embeddings (~2GB per layer) are downloaded from [HuggingFace](https://huggingface.co/McGill-NLP/latentlens-qwen2vl-embeddings). The script:

1. Loads Qwen2-VL and processes your image
2. Extracts hidden states at layers 2, 8, and 27
3. Finds nearest text neighbors in contextual embedding space
4. Outputs a visualization .png file with the input image and sampled visual token interpretations 

Customize with `--layers 1,4,8,16,24,27` and `--top-k 10`. Requires a GPU with >=24GB VRAM.

---

## Library API

The `latentlens` package provides a Python API for building and searching contextual embedding indices with any HuggingFace causal LM.

### Build an index from a text corpus

```python
import latentlens

# Build index (loads model, processes corpus, deduplicates by prefix)
index = latentlens.build_index(
    "meta-llama/Meta-Llama-3-8B",
    corpus="my_corpus.txt",  # .txt, .csv, or list of strings
)
index.save("my_llama_index/")
```

### Load a pre-built index and search

```python
import latentlens

# Load from HuggingFace Hub (pre-computed Qwen2-VL embeddings)
index = latentlens.ContextualIndex.from_pretrained("McGill-NLP/latentlens-qwen2vl-embeddings")

# Or from a local directory
index = latentlens.ContextualIndex.from_directory("my_llama_index/")

# Search with your own hidden states [num_tokens, hidden_dim]
results = index.search(hidden_states, top_k=5)
# results[i] = [Neighbor(token_str=' dog', similarity=0.42, contextual_layer=27, ...), ...]
```

### Full pipeline: inference + search

```python
import torch
import latentlens

model, tokenizer = latentlens.load_model("Qwen/Qwen2-7B")
index = latentlens.ContextualIndex.from_directory("qwen_index/")

inputs = tokenizer("a photo of a dog", return_tensors="pt").to("cuda")
hidden_states = latentlens.get_hidden_states(model, inputs["input_ids"])

# Interpret layer 27
hs = torch.nn.functional.normalize(hidden_states[27].squeeze(0).float(), dim=-1)
results = index.search(hs, top_k=5)
```

### Custom Models

The library works with any HuggingFace `AutoModelForCausalLM`. For models not in `SUPPORTED_MODELS`, layer selection defaults to `auto_layers()` which picks early/mid/late layers automatically. To add explicit support for a new model:

1. Add an entry to `latentlens.SUPPORTED_MODELS` with `num_hidden_layers`, `hidden_size`, and `default_layers`
2. Open a PR to share with the community

---

## Reproducing Paper Results

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
