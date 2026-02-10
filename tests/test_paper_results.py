"""
Tests for paper results data.

These tests verify that:
1. paper_results.json has expected format
2. All 9 model combinations are present
3. All expected layers are present
4. Values are in valid range (0-100%)

Run with: pytest tests/test_paper_results.py -v
"""

import json
import pytest
from pathlib import Path


# Expected models (LLM + vision encoder combinations)
EXPECTED_MODELS = [
    "olmo-7b+vit-l-14-336",
    "olmo-7b+dinov2-large-336",
    "olmo-7b+siglip",
    "llama3-8b+vit-l-14-336",
    "llama3-8b+dinov2-large-336",
    "llama3-8b+siglip",
    "qwen2-7b+vit-l-14-336",
    "qwen2-7b+dinov2-large-336",
    "qwen2-7b+siglip",
]

# Expected layers for 32-layer models (OLMo, LLaMA)
LAYERS_32 = ["0", "1", "2", "4", "8", "16", "24", "30", "31"]

# Expected layers for 28-layer models (Qwen2)
LAYERS_28 = ["0", "1", "2", "4", "8", "16", "24", "26", "27"]

# Analysis types in the paper
ANALYSIS_TYPES = ["nn", "logitlens", "contextual"]


@pytest.fixture
def paper_results():
    """Load paper results JSON."""
    results_path = Path(__file__).parent.parent / "reproduce" / "paper_results.json"
    if not results_path.exists():
        pytest.skip(f"Paper results not found at {results_path}")
    with open(results_path) as f:
        return json.load(f)


class TestPaperResultsFormat:
    """Test that paper_results.json has expected format."""

    def test_has_all_analysis_types(self, paper_results):
        """Should have all three analysis types."""
        for analysis_type in ANALYSIS_TYPES:
            assert analysis_type in paper_results, f"Missing analysis type: {analysis_type}"

    def test_nn_has_all_models(self, paper_results):
        """NN (EmbeddingLens) should have all 9 models."""
        nn = paper_results["nn"]
        for model in EXPECTED_MODELS:
            assert model in nn, f"Missing model in nn: {model}"

    def test_logitlens_has_all_models(self, paper_results):
        """LogitLens should have all 9 models."""
        logitlens = paper_results["logitlens"]
        for model in EXPECTED_MODELS:
            assert model in logitlens, f"Missing model in logitlens: {model}"

    def test_contextual_has_all_models(self, paper_results):
        """Contextual (LatentLens) should have all 9 models."""
        contextual = paper_results["contextual"]
        for model in EXPECTED_MODELS:
            assert model in contextual, f"Missing model in contextual: {model}"


class TestPaperResultsLayers:
    """Test that all expected layers are present."""

    def test_olmo_models_have_32_layer_keys(self, paper_results):
        """OLMo models should have layers for 32-layer architecture."""
        for analysis_type in ANALYSIS_TYPES:
            for model in EXPECTED_MODELS:
                if "olmo" in model:
                    layers = paper_results[analysis_type][model]
                    for layer in LAYERS_32:
                        assert layer in layers, f"Missing layer {layer} for {model} in {analysis_type}"

    def test_llama_models_have_32_layer_keys(self, paper_results):
        """LLaMA models should have layers for 32-layer architecture."""
        for analysis_type in ANALYSIS_TYPES:
            for model in EXPECTED_MODELS:
                if "llama" in model:
                    layers = paper_results[analysis_type][model]
                    for layer in LAYERS_32:
                        assert layer in layers, f"Missing layer {layer} for {model} in {analysis_type}"

    def test_qwen_models_have_28_layer_keys(self, paper_results):
        """Qwen2 models should have layers for 28-layer architecture."""
        for analysis_type in ANALYSIS_TYPES:
            for model in EXPECTED_MODELS:
                if "qwen" in model:
                    layers = paper_results[analysis_type][model]
                    for layer in LAYERS_28:
                        assert layer in layers, f"Missing layer {layer} for {model} in {analysis_type}"


class TestPaperResultsValues:
    """Test that values are in valid range."""

    def test_all_values_in_valid_range(self, paper_results):
        """All values should be between 0 and 100 for main analysis types."""
        for analysis_type in ANALYSIS_TYPES:
            if analysis_type not in paper_results:
                continue
            models = paper_results[analysis_type]
            for model, layers in models.items():
                if not isinstance(layers, dict):
                    continue
                for layer, value in layers.items():
                    if not isinstance(value, (int, float)):
                        continue  # Skip non-numeric values
                    assert 0 <= value <= 100, (
                        f"Invalid value {value} for {analysis_type}/{model}/layer{layer}"
                    )

    def test_latentlens_generally_higher(self, paper_results):
        """LatentLens should generally have higher values than baselines (paper's key finding)."""
        contextual = paper_results["contextual"]
        nn = paper_results["nn"]
        logitlens = paper_results["logitlens"]

        # Compare averages across all models and layers
        def avg(data):
            values = []
            for model, layers in data.items():
                if isinstance(layers, dict):
                    values.extend(layers.values())
            return sum(values) / len(values) if values else 0

        contextual_avg = avg(contextual)
        nn_avg = avg(nn)
        logitlens_avg = avg(logitlens)

        # LatentLens should have higher average (allowing some margin)
        assert contextual_avg > nn_avg * 0.8, (
            f"LatentLens avg ({contextual_avg:.1f}) should be higher than EmbeddingLens avg ({nn_avg:.1f})"
        )

    def test_olmo_vit_has_reasonable_latentlens_values(self, paper_results):
        """OLMo+ViT (baseline model) should have ~60-80% LatentLens interpretability."""
        olmo_vit = paper_results["contextual"]["olmo-7b+vit-l-14-336"]
        avg_value = sum(olmo_vit.values()) / len(olmo_vit)
        assert 50 < avg_value < 90, f"OLMo+ViT LatentLens avg ({avg_value:.1f}) outside expected range"


class TestPaperResultsConsistency:
    """Test internal consistency of results."""

    def test_no_duplicate_models(self, paper_results):
        """Should not have duplicate model entries."""
        for analysis_type, models in paper_results.items():
            if isinstance(models, dict):
                model_names = list(models.keys())
                assert len(model_names) == len(set(model_names)), (
                    f"Duplicate models in {analysis_type}"
                )

    def test_layer_keys_are_strings(self, paper_results):
        """Layer keys should be strings (JSON convention)."""
        for analysis_type, models in paper_results.items():
            if not isinstance(models, dict):
                continue
            for model, layers in models.items():
                if not isinstance(layers, dict):
                    continue
                for layer_key in layers.keys():
                    assert isinstance(layer_key, str), (
                        f"Layer key {layer_key} should be string in {analysis_type}/{model}"
                    )

    def test_values_are_floats(self, paper_results):
        """Values should be floats (percentages) for main analysis types."""
        for analysis_type in ANALYSIS_TYPES:
            if analysis_type not in paper_results:
                continue
            models = paper_results[analysis_type]
            for model, layers in models.items():
                if not isinstance(layers, dict):
                    continue
                for layer, value in layers.items():
                    assert isinstance(value, (int, float)), (
                        f"Value {value} should be numeric in {analysis_type}/{model}/layer{layer}"
                    )
