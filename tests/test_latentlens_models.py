"""
Tests for latentlens.models — load_model, get_hidden_states, MODEL_DEFAULTS.
"""

import pytest

from latentlens.models import MODEL_DEFAULTS


class TestModelDefaults:
    def test_olmo_present(self):
        assert "allenai/OLMo-7B-1024-preview" in MODEL_DEFAULTS

    def test_llama_present(self):
        assert "meta-llama/Meta-Llama-3-8B" in MODEL_DEFAULTS

    def test_qwen_present(self):
        assert "Qwen/Qwen2-7B" in MODEL_DEFAULTS

    def test_required_keys(self):
        for name, info in MODEL_DEFAULTS.items():
            assert "num_hidden_layers" in info, f"{name} missing num_hidden_layers"
            assert "hidden_size" in info, f"{name} missing hidden_size"
            assert "default_layers" in info, f"{name} missing default_layers"

    def test_layer_counts(self):
        assert MODEL_DEFAULTS["allenai/OLMo-7B-1024-preview"]["num_hidden_layers"] == 32
        assert MODEL_DEFAULTS["meta-llama/Meta-Llama-3-8B"]["num_hidden_layers"] == 32
        assert MODEL_DEFAULTS["Qwen/Qwen2-7B"]["num_hidden_layers"] == 28



class TestLoadModel:
    @pytest.mark.slow
    def test_load_model_sets_eval(self):
        from latentlens.models import load_model

        model, tokenizer = load_model("allenai/OLMo-7B-1024-preview", device="cpu")
        assert not model.training
        assert tokenizer.pad_token is not None

    @pytest.mark.slow
    def test_get_hidden_states_shape(self):
        import torch
        from latentlens.models import load_model, get_hidden_states

        model, tokenizer = load_model("allenai/OLMo-7B-1024-preview", device="cpu")
        inputs = tokenizer("hello world", return_tensors="pt")
        hidden_states = get_hidden_states(model, inputs["input_ids"])

        # Should have num_hidden_layers + 1 entries (embedding + each block)
        assert len(hidden_states) == 33  # 32 layers + 1 embedding
        seq_len = inputs["input_ids"].shape[1]
        for hs in hidden_states:
            assert hs.shape == (1, seq_len, 4096)
