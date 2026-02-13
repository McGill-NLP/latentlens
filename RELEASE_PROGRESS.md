# LatentLens Release Progress

Tracks development progress, decisions, and issues for the public release.

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
| `tests/test_latentlens_index.py` | 16 fast tests (synthetic data) + 1 slow (HF download) | Done |
| `tests/test_latentlens_extract.py` | 5 fast tests + 3 slow (GPU) | Done |
| `tests/test_latentlens_models.py` | 5 fast tests + 2 slow (GPU) | Done |

**Key design decisions:**
- Prefix deduplication instead of reservoir sampling (causal LMs produce identical embeddings for identical prefixes)
- Cross-layer merge as default search (core LatentLens insight)
- Cache format backwards-compatible with existing `embeddings_cache.pt` files
- Soft cap of 50 unique contexts per token (configurable)

**Test results:** 42/42 fast tests pass (31 new library + 11 existing).

**Commit:** `d6a76a4` pushed to `origin/feature/library-api`.

### Still TODO before merging to main
- [ ] Run slow tests on GPU (model loading, build_index, from_pretrained)
- [ ] End-to-end test: build_index on small corpus → save → reload → search
- [ ] Review API ergonomics with co-authors
- [ ] Register `slow` pytest mark in pyproject.toml (currently shows warnings)
- [ ] Decide: should CLAUDE.md be committed? (useful for contributors, but non-standard)
