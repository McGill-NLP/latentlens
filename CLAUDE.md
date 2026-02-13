# CLAUDE.md — LatentLens Public Repository

## Repository Overview

This is the **public release repo** for LatentLens (github.com/McGill-NLP/latentlens).
Primary audience: researchers who read the paper and want to apply LatentLens to their own models.

## Directory Structure

| Directory | Purpose | Audience |
|-----------|---------|----------|
| `latentlens/` | Core library (index, extract, models) | Everyone |
| `reproduce/` | Paper reproduction (configs, scripts, golden data) | Reproducers |
| `tests/` | Test suite | Developers |
| `molmo/` | Molmo VLM infrastructure (for reproduction) | Reproducers |

Top-level: `quickstart.py` + `example.png` (getting started)

## Mandatory First Steps

**Every session**, read:
- **RELEASE_PROGRESS.md** — what's done, what's in progress, known issues
- **README.md** — public-facing docs, project structure

## The Three Rules

### 1. EDIT, NEVER REWRITE
- Always edit existing code with surgical changes. Never rewrite from scratch.
- If about to write 50+ lines doing something similar to existing code, STOP and reuse.

### 2. VALIDATE BEFORE SHIPPING
- Run tests: `PYTHONPATH=. python -m pytest tests/ -v --timeout=120 -m "not slow"`
- Verify imports: `python -c "from latentlens import ContextualIndex, build_index"`
- Check that existing tests still pass after changes.

### 3. COMMIT AND DOCUMENT
- Commit after every meaningful change (don't batch).
- Write clear commit messages.
- Update `RELEASE_PROGRESS.md` after every significant change.
- We're on a feature branch — push freely, PR when ready.

## Running Tests

```bash
# Fast tests (no GPU, no model downloads)
PYTHONPATH=. python -m pytest tests/ -v --timeout=120 -m "not slow"

# All tests including slow (GPU + model downloads)
PYTHONPATH=. python -m pytest tests/ -v --timeout=600
```

## Git Workflow

- **main** = public release (single squashed commit). Do not push here without review.
- **feature/library-api** = current working branch for the library implementation.
- Push to feature branch freely. Open PR when ready for review.

## Code Style

- `ruff` for linting (config in `pyproject.toml`)
- All imports use `latentlens`, never `olmo` or internal names
- Type hints encouraged but not required for every function
- Docstrings for public API functions (NumPy style)

## Key Design Decisions (Library)

1. **Prefix deduplication** in `build_index()` — causal LMs produce identical embeddings for identical prefixes. We hash and skip duplicates.
2. **Cross-layer merge** is the default search — queries all contextual layers, ranks globally.
3. **Cache format is backwards-compatible** — reads existing `layer_N/embeddings_cache.pt` files.
4. **No model needed for search** — `ContextualIndex` works standalone; model only needed for `build_index()`.

## Error Handling

- Let errors fail loudly. No silent try-except.
- Only catch specific, expected errors with explicit recovery (e.g., ImportError for optional deps).

## Communication

- Show evidence for claims (raw data, computation steps).
- Provide full absolute paths to created/modified files.
- When something seems wrong, investigate root cause before patching symptoms.
