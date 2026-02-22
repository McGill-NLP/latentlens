# LatentLens Release Progress

Tracks development progress, decisions, and issues for the public release.

---

## 2026-02-19 — Blind usability test + fixes

**Usability test:** Fresh Claude instance used the library with zero prior context, given only the README and a use case (interpret Pythia-1.4B hidden states using concepts.txt). Results:
- **Zero API friction** — figured out full pipeline from README alone
- **117k sentences through Pythia-1.4B in ~7 minutes** on A6000 (fp16, batch_size=32)
- Script ran first try, produced correct results

**4 issues found and fixed (commit `477c712`):**
1. `search()` now auto-normalizes queries — users no longer need manual `F.normalize()`
2. `build_index()` accepts `model=` and `tokenizer=` to avoid loading the model twice
3. README now documents layer indexing convention (`hidden_states[0]` = embeddings, `[i]` = block i output)
4. README documents `search(layers=...)` parameter for per-layer analysis
5. Fixed `torch_dtype` deprecation warning → use `dtype=` in `from_pretrained()`

---

## 2026-02-18 — VLM support + concepts corpus + README restructure

**README restructured (commit `983779e`):**
- Leads with `pip install` → Option A (build index) / Option B (pre-built) → Search
- Concise getting-started flow so users understand usage in a few lines
- Quickstart script moved below library usage; reproduction section unchanged

**VLM support (commit `202035d`):**

**VLM support (commit `202035d`):**
- `load_model()` now tries `AutoModelForCausalLM`, falls back to `AutoModel` for VLMs
- `get_num_hidden_layers()` handles nested `config.text_config` (common in VLMs)
- E2E tested: Qwen2-VL-2B, SmolVLM-256M (Idefics3)

**concepts.txt added:**
- 117,125 sentences derived from 23,425 WordNet concepts (5 sentences each, varying complexity)
- Source: bensaine/ll2 corpus (GPT-generated sentences covering broad human concepts)
- Intended as the default corpus for `build_index()` when users don't provide their own

**53/53 tests pass** (6 model architectures: distilgpt2, tiny-gpt2, pythia-70m, opt-125m, Qwen2-VL-2B, SmolVLM-256M).

### Open TODOs
- [x] ~~User provides a curated `concepts.txt`~~ — done (117k sentences, WordNet-derived)
- [ ] Pre-compute indices for popular models (OLMo, LLaMA, Qwen2, Qwen2-VL) using concepts.txt, host on McGill-NLP HuggingFace
- [ ] Write a feature/spec doc explaining user-facing capabilities in prose
- [x] ~~Relax `torch<2.5.0` upper bound in pyproject.toml~~ — resolved (no upper bound in current deps)
- [x] ~~Register `slow` pytest mark in pyproject.toml~~ — resolved (added in pytest.ini_options)
- [ ] Review API ergonomics with co-authors
- [ ] Merge `feature/library-api` → `main` when ready

---

## 2026-02-22 — Pre-merge release prep

**6 fixes before merge + PyPI publish:**
1. **CRITICAL:** Bundle `concepts.txt` inside the wheel via hatch `force-include`, so `corpus="concepts.txt"` works after `pip install` (not just from repo root)
2. Remove broken CLI entry point (`[project.scripts]` pointed at `molmo.cli:main` — paper reproduction, not library API)
3. Add `packaging>=21.0` to explicit dependencies (was only transitive via transformers)
4. Remove premature `SUPPORTED_MODELS` backwards-compat alias (no existing users for v0.1.0)
5. Update project description: "Interpret what hidden representations encode via contextual nearest neighbors"
6. Mark resolved TODOs as done (`torch` upper bound, `slow` pytest mark)

---

## 2026-02-13 — Library API (feature/library-api branch)

**Implemented general-purpose `latentlens` library:**

| File | Purpose | Status |
|------|---------|--------|
| `latentlens/index.py` | ContextualIndex + Neighbor, cross-layer search, save/load, HF Hub | Done |
| `latentlens/extract.py` | build_index() with prefix dedup, auto_layers(), load_corpus() | Done |
| `latentlens/models.py` | load_model(), get_hidden_states(), SUPPORTED_MODELS | Done |
| `latentlens/__init__.py` | Public exports | Done |
| `README.md` | Library API section with 3 examples, checked TODO box | Done |
| `tests/test_user_e2e.py` | 11 e2e tests across 6 architectures (4 LLMs + 2 VLMs) | Done |
| `tests/test_latentlens_index.py` | 16 fast tests (synthetic data) + 1 slow (HF download) | Done |
| `tests/test_latentlens_extract.py` | 5 fast tests + 3 slow (GPU) | Done |
| `tests/test_latentlens_models.py` | 5 fast tests + 2 slow (GPU) | Done |

**Key design decisions:**
- Prefix deduplication instead of reservoir sampling (causal LMs produce identical embeddings for identical prefixes)
- Cross-layer merge as default search (core LatentLens insight)
- Cache format backwards-compatible with existing `embeddings_cache.pt` files
- Soft cap of 50 unique contexts per token (configurable)
