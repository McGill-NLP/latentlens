"""
Tests for scripts/evaluate/evaluate_interpretability.py

Tests pure data-processing functions against golden JSON files.
No GPU, API key, or model loading required.
"""
import os
import sys
import json
import pytest
from pathlib import Path

# Add scripts/evaluate/ to sys.path so that the relative imports
# (from utils import ..., from prompts import ...) inside
# evaluate_interpretability.py resolve correctly.
EVALUATE_DIR = Path(__file__).resolve().parents[1] / "reproduce" / "scripts" / "evaluate"
sys.path.insert(0, str(EVALUATE_DIR))

from evaluate_interpretability import (
    detect_method,
    extract_words_for_patch,
    extract_full_word_from_token,
    get_images_from_data,
    load_analysis_results,
)

# ---------------------------------------------------------------------------
# Paths to golden data (set ORIGINAL_REPO_ROOT env var to enable these tests)
# ---------------------------------------------------------------------------
_repo_root = os.environ.get("ORIGINAL_REPO_ROOT", None)
ANALYSIS_RESULTS = Path(_repo_root) / "analysis_results" if _repo_root else None

MODEL_DIR = "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10"

LATENTLENS_DIR = ANALYSIS_RESULTS / "contextual_nearest_neighbors" / MODEL_DIR if ANALYSIS_RESULTS else None
LOGITLENS_DIR = ANALYSIS_RESULTS / "logit_lens" / MODEL_DIR if ANALYSIS_RESULTS else None
EMBEDDINGLENS_DIR = ANALYSIS_RESULTS / "nearest_neighbors" / MODEL_DIR if ANALYSIS_RESULTS else None

LATENTLENS_FILE = LATENTLENS_DIR / "contextual_neighbors_visual8_allLayers.json" if LATENTLENS_DIR else None
LOGITLENS_FILE = LOGITLENS_DIR / "logit_lens_layer8_topk5_multi-gpu.json" if LOGITLENS_DIR else None
EMBEDDINGLENS_FILE = EMBEDDINGLENS_DIR / "nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer8.json" if EMBEDDINGLENS_DIR else None


# ---------------------------------------------------------------------------
# Fixtures: load golden data once per session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def latentlens_data():
    if LATENTLENS_FILE is None or not LATENTLENS_FILE.exists():
        pytest.skip(f"Golden file not found (set ORIGINAL_REPO_ROOT): {LATENTLENS_FILE}")
    with open(LATENTLENS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def logitlens_data():
    if LOGITLENS_FILE is None or not LOGITLENS_FILE.exists():
        pytest.skip(f"Golden file not found (set ORIGINAL_REPO_ROOT): {LOGITLENS_FILE}")
    with open(LOGITLENS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def embeddinglens_data():
    if EMBEDDINGLENS_FILE is None or not EMBEDDINGLENS_FILE.exists():
        pytest.skip(f"Golden file not found (set ORIGINAL_REPO_ROOT): {EMBEDDINGLENS_FILE}")
    with open(EMBEDDINGLENS_FILE) as f:
        return json.load(f)


# ===================================================================
# 1. detect_method()
# ===================================================================
class TestDetectMethod:
    """Verify detect_method correctly identifies all 3 analysis methods."""

    def test_detects_latentlens(self, latentlens_data):
        assert detect_method(latentlens_data) == "latentlens"

    def test_detects_logitlens(self, logitlens_data):
        assert detect_method(logitlens_data) == "logitlens"

    def test_detects_embeddinglens(self, embeddinglens_data):
        assert detect_method(embeddinglens_data) == "embeddinglens"

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="Cannot detect method"):
            detect_method({})

    def test_raises_on_ambiguous(self):
        """A JSON with results but no recognisable patch keys should fail."""
        data = {"results": [{"chunks": [{"patches": [{"unknown_key": []}]}]}]}
        with pytest.raises(ValueError, match="Cannot detect method"):
            detect_method(data)


# ===================================================================
# 2. extract_words_for_patch()
# ===================================================================
class TestExtractWordsForPatch:
    """Verify word extraction from real patches for all 3 methods."""

    def _first_patch(self, data, method):
        """Get the first patch from the first image."""
        images = get_images_from_data(data, method)
        return images[0]["chunks"][0]["patches"][0]

    def test_latentlens_returns_words(self, latentlens_data):
        patch = self._first_patch(latentlens_data, "latentlens")
        words = extract_words_for_patch(patch, "latentlens")
        assert isinstance(words, list)
        assert len(words) == 5
        # All words should be non-empty strings
        for w in words:
            assert isinstance(w, str)
            assert len(w) > 0

    def test_latentlens_first_patch_values(self, latentlens_data):
        """The first patch of image 0 has token_str '416' from caption
        'epj416 licence plate on penguin express bus'.
        extract_full_word_from_token should expand '416' to 'epj416'."""
        patch = self._first_patch(latentlens_data, "latentlens")
        words = extract_words_for_patch(patch, "latentlens")
        assert words[0] == "epj416"

    def test_logitlens_returns_tokens(self, logitlens_data):
        patch = self._first_patch(logitlens_data, "logitlens")
        words = extract_words_for_patch(patch, "logitlens")
        assert isinstance(words, list)
        assert len(words) == 5
        # First token from golden data is "Produto"
        assert words[0] == "Produto"

    def test_logitlens_exact_top5(self, logitlens_data):
        """Verify exact top-5 tokens for first patch."""
        patch = self._first_patch(logitlens_data, "logitlens")
        words = extract_words_for_patch(patch, "logitlens")
        expected = ["Produto", "\u0447\u0435\u0442", " Sho", "\u0438\u0441\u043a", " sect"]
        assert words == expected

    def test_embeddinglens_returns_tokens(self, embeddinglens_data):
        patch = self._first_patch(embeddinglens_data, "embeddinglens")
        words = extract_words_for_patch(patch, "embeddinglens")
        assert isinstance(words, list)
        assert len(words) == 5

    def test_embeddinglens_exact_top5(self, embeddinglens_data):
        """Verify exact top-5 tokens for first patch."""
        patch = self._first_patch(embeddinglens_data, "embeddinglens")
        words = extract_words_for_patch(patch, "embeddinglens")
        # From golden data: first patch of image 0
        expected_first = "\ufffd"
        assert words[0] == expected_first
        expected = ["\ufffd", "722", "\u05d5\ufffd", "958", "izzie"]
        assert words == expected

    def test_empty_patch_returns_empty(self):
        """A patch with no recognized keys should return empty list."""
        assert extract_words_for_patch({}, "latentlens") == []
        assert extract_words_for_patch({}, "logitlens") == []
        assert extract_words_for_patch({}, "embeddinglens") == []


# ===================================================================
# 3. extract_full_word_from_token()
# ===================================================================
class TestExtractFullWordFromToken:
    """Test subword -> full word expansion with real LatentLens data."""

    def test_predomin_to_predominately(self):
        """Token ' predomin' in caption 'a predominately white surf board'
        should expand to 'predominately'."""
        result = extract_full_word_from_token(
            "a predominately white surf board", " predomin"
        )
        assert result == "predominately"

    def test_predomin_to_predominance(self):
        """Different caption, same token -> different expanded word."""
        result = extract_full_word_from_token(
            "a predominance of males, includes two females.", " predomin"
        )
        assert result == "predominance"

    def test_416_to_epj416(self):
        """Numeric subword '416' in 'epj416 licence plate...' -> 'epj416'."""
        result = extract_full_word_from_token(
            "epj416 licence plate on penguin express bus", "416"
        )
        assert result == "epj416"

    def test_881_to_4h881(self):
        """Numeric subword '881' in '4h881 is on boat' -> '4h881'."""
        result = extract_full_word_from_token("4h881 is on boat", "881")
        assert result == "4h881"

    def test_854_to_tgm854m(self):
        """Numeric subword '854' in 'tgm854m is written in number plate.' -> 'tgm854m'."""
        result = extract_full_word_from_token(
            "tgm854m is written in number plate.", "854"
        )
        assert result == "tgm854m"

    def test_whole_word_unchanged(self):
        """A token that is already a whole word should be returned as-is."""
        result = extract_full_word_from_token(
            "a predominately white surf board", "white"
        )
        assert result == "white"

    def test_token_not_in_caption(self):
        """Token not found in caption -> return stripped token."""
        result = extract_full_word_from_token("hello world", "xyz")
        assert result == "xyz"

    def test_empty_sentence(self):
        """Empty sentence -> return stripped token."""
        result = extract_full_word_from_token("", "foo")
        assert result == "foo"

    def test_empty_token(self):
        """Empty token -> return empty string."""
        result = extract_full_word_from_token("hello world", "")
        assert result == ""

    def test_real_latentlens_patch_expansion(self, latentlens_data):
        """Verify expansion on an actual patch from golden data.
        Patch 1 of image 0 has token_str=' predomin' with caption
        'a predominately white surf board' -> should expand to 'predominately'."""
        patch1 = latentlens_data["results"][0]["chunks"][0]["patches"][1]
        neighbor = patch1["nearest_contextual_neighbors"][0]
        token = neighbor["token_str"]
        caption = neighbor["caption"]
        assert token == " predomin"
        assert caption == "a predominately white surf board"
        expanded = extract_full_word_from_token(caption, token)
        assert expanded == "predominately"


# ===================================================================
# 4. get_images_from_data()
# ===================================================================
class TestGetImagesFromData:
    """Verify correct image lists are returned for all 3 methods."""

    def test_latentlens_image_count(self, latentlens_data):
        images = get_images_from_data(latentlens_data, "latentlens")
        assert len(images) == 10

    def test_logitlens_image_count(self, logitlens_data):
        images = get_images_from_data(logitlens_data, "logitlens")
        assert len(images) == 10

    def test_embeddinglens_image_count(self, embeddinglens_data):
        images = get_images_from_data(embeddinglens_data, "embeddinglens")
        assert len(images) == 10

    def test_latentlens_returns_results(self, latentlens_data):
        """LatentLens should return data['results']."""
        images = get_images_from_data(latentlens_data, "latentlens")
        assert images is latentlens_data["results"]

    def test_logitlens_returns_results(self, logitlens_data):
        """LogitLens should return data['results']."""
        images = get_images_from_data(logitlens_data, "logitlens")
        assert images is logitlens_data["results"]

    def test_embeddinglens_returns_splits_validation(self, embeddinglens_data):
        """EmbeddingLens should return data['splits']['validation']['images']."""
        images = get_images_from_data(embeddinglens_data, "embeddinglens")
        assert images is embeddinglens_data["splits"]["validation"]["images"]

    def test_images_have_chunks(self, latentlens_data, logitlens_data, embeddinglens_data):
        """Every image entry should have a 'chunks' key."""
        for method, data in [
            ("latentlens", latentlens_data),
            ("logitlens", logitlens_data),
            ("embeddinglens", embeddinglens_data),
        ]:
            images = get_images_from_data(data, method)
            for img in images:
                assert "chunks" in img, f"{method}: image missing 'chunks' key"

    def test_images_have_image_idx(self, latentlens_data, logitlens_data, embeddinglens_data):
        """Every image entry should have an image_idx key."""
        for method, data in [
            ("latentlens", latentlens_data),
            ("logitlens", logitlens_data),
            ("embeddinglens", embeddinglens_data),
        ]:
            images = get_images_from_data(data, method)
            for img in images:
                assert "image_idx" in img, f"{method}: image missing 'image_idx'"


# ===================================================================
# 5. Grid size detection
# ===================================================================
class TestGridSizeDetection:
    """Verify grid_size is computed correctly from golden data.
    OLMo-ViT uses a 24x24 grid (576 patches)."""

    def _compute_grid_size(self, data, method):
        """Replicate the grid size detection logic from evaluate_model()."""
        images = get_images_from_data(data, method)
        first_patches = images[0]["chunks"][0]["patches"]
        max_row = max(p.get("patch_row", 0) for p in first_patches)
        max_col = max(p.get("patch_col", 0) for p in first_patches)
        return max(max_row + 1, max_col + 1)

    def test_latentlens_grid_24(self, latentlens_data):
        grid_size = self._compute_grid_size(latentlens_data, "latentlens")
        assert grid_size == 24

    def test_logitlens_grid_24(self, logitlens_data):
        grid_size = self._compute_grid_size(logitlens_data, "logitlens")
        assert grid_size == 24

    def test_embeddinglens_grid_24(self, embeddinglens_data):
        grid_size = self._compute_grid_size(embeddinglens_data, "embeddinglens")
        assert grid_size == 24

    def test_576_patches(self, latentlens_data):
        """24x24 grid = 576 patches per image."""
        patches = latentlens_data["results"][0]["chunks"][0]["patches"]
        assert len(patches) == 576

    def test_patch_size_from_grid(self):
        """For a 512x512 image with grid_size=24, each patch is ~21.33px."""
        grid_size = 24
        patch_size = 512.0 / grid_size
        assert abs(patch_size - 21.333333) < 0.001


# ===================================================================
# 6. load_analysis_results()
# ===================================================================
class TestLoadAnalysisResults:
    """Verify load_analysis_results() finds the right file for each method."""

    def test_latentlens_visual_pattern(self):
        """LatentLens files use *visual{N}* naming convention."""
        if LATENTLENS_DIR is None or not LATENTLENS_DIR.exists():
            pytest.skip(f"Directory not found: {LATENTLENS_DIR}")
        data = load_analysis_results(str(LATENTLENS_DIR), 8)
        assert "results" in data
        # Verify it loaded the correct layer
        assert data["visual_layer"] == 8

    def test_logitlens_layer_pattern(self):
        """LogitLens files use *layer{N}* naming convention."""
        if LOGITLENS_DIR is None or not LOGITLENS_DIR.exists():
            pytest.skip(f"Directory not found: {LOGITLENS_DIR}")
        data = load_analysis_results(str(LOGITLENS_DIR), 8)
        assert "results" in data
        assert data["layer_idx"] == 8

    def test_embeddinglens_layer_pattern(self):
        """EmbeddingLens files use *layer{N}* naming convention."""
        if EMBEDDINGLENS_DIR is None or not EMBEDDINGLENS_DIR.exists():
            pytest.skip(f"Directory not found: {EMBEDDINGLENS_DIR}")
        data = load_analysis_results(str(EMBEDDINGLENS_DIR), 8)
        assert "splits" in data
        assert data["llm_layer"] == 8

    def test_latentlens_layer0(self):
        """Verify loading layer 0 works (different layer number)."""
        if LATENTLENS_DIR is None or not LATENTLENS_DIR.exists():
            pytest.skip(f"Directory not found: {LATENTLENS_DIR}")
        data = load_analysis_results(str(LATENTLENS_DIR), 0)
        assert data["visual_layer"] == 0

    def test_logitlens_layer31(self):
        """Verify loading layer 31 works."""
        if LOGITLENS_DIR is None or not LOGITLENS_DIR.exists():
            pytest.skip(f"Directory not found: {LOGITLENS_DIR}")
        data = load_analysis_results(str(LOGITLENS_DIR), 31)
        assert data["layer_idx"] == 31

    def test_missing_layer_raises(self):
        """Requesting a non-existent layer should raise FileNotFoundError."""
        if LATENTLENS_DIR is None or not LATENTLENS_DIR.exists():
            pytest.skip(f"Directory not found: {LATENTLENS_DIR}")
        with pytest.raises(FileNotFoundError, match="No results found for layer 999"):
            load_analysis_results(str(LATENTLENS_DIR), 999)

    def test_missing_dir_raises(self, tmp_path):
        """Requesting from a non-existent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_analysis_results(str(tmp_path), 8)

    def test_loaded_data_detectable(self):
        """Data loaded via load_analysis_results should be detectable by detect_method."""
        if LATENTLENS_DIR is None or not LATENTLENS_DIR.exists():
            pytest.skip(f"Directory not found: {LATENTLENS_DIR}")
        data = load_analysis_results(str(LATENTLENS_DIR), 8)
        assert detect_method(data) == "latentlens"

    def test_loaded_logitlens_detectable(self):
        if LOGITLENS_DIR is None or not LOGITLENS_DIR.exists():
            pytest.skip(f"Directory not found: {LOGITLENS_DIR}")
        data = load_analysis_results(str(LOGITLENS_DIR), 8)
        assert detect_method(data) == "logitlens"

    def test_loaded_embeddinglens_detectable(self):
        if EMBEDDINGLENS_DIR is None or not EMBEDDINGLENS_DIR.exists():
            pytest.skip(f"Directory not found: {EMBEDDINGLENS_DIR}")
        data = load_analysis_results(str(EMBEDDINGLENS_DIR), 8)
        assert detect_method(data) == "embeddinglens"
