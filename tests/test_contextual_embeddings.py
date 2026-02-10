"""
Tests for contextual embedding extraction.

These tests verify that:
1. Existing contextual embeddings have expected structure
2. Metadata matches expected format
3. Running extract_embeddings.py reproduces golden embeddings (actual value comparison)

Run with: pytest tests/test_contextual_embeddings.py -v
"""
import json
import subprocess
import sys
import tempfile
import pytest
import numpy as np
from pathlib import Path


class TestContextualEmbeddingsStructure:
    """Test that existing contextual embeddings have expected structure."""

    def test_metadata_exists(self, contextual_embeddings_olmo):
        """Metadata file should exist and be valid JSON."""
        metadata_file = contextual_embeddings_olmo / "metadata.json"
        assert metadata_file.exists(), f"metadata.json not found at {metadata_file}"

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert "model_name" in metadata
        assert "layers_extracted" in metadata
        assert "num_captions_processed" in metadata

    def test_metadata_values_olmo(self, contextual_embeddings_olmo):
        """Test that OLMo metadata has expected values."""
        metadata_file = contextual_embeddings_olmo / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Check model name
        assert metadata["model_name"] == "allenai/OLMo-7B-1024-preview"

        # Check layers (OLMo has 32 layers, we extract 1,2,4,8,16,24,30,31)
        expected_layers = [1, 2, 4, 8, 16, 24, 30, 31]
        assert metadata["layers_extracted"] == expected_layers

        # Check embedding dtype
        assert metadata["embedding_dtype"] == "float8"

        # Check dataset
        assert metadata["dataset"] == "vg"

    def test_layer_directories_exist(self, contextual_embeddings_olmo):
        """Each extracted layer should have a directory."""
        expected_layers = [1, 2, 4, 8, 16, 24, 30, 31]

        for layer in expected_layers:
            layer_dir = contextual_embeddings_olmo / f"layer_{layer}"
            assert layer_dir.exists(), f"layer_{layer} directory not found"

            # Check for required files
            cache_file = layer_dir / "embeddings_cache.pt"
            token_file = layer_dir / "token_embeddings.json"

            assert cache_file.exists(), f"embeddings_cache.pt not found in layer_{layer}"
            assert token_file.exists(), f"token_embeddings.json not found in layer_{layer}"

    def test_layer_statistics(self, contextual_embeddings_olmo):
        """Layer statistics should match expected values."""
        metadata_file = contextual_embeddings_olmo / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        layer_stats = metadata["layer_statistics"]

        # All layers should have same number of unique tokens (Visual Genome vocabulary)
        expected_unique_tokens = 26862
        for layer_name, stats in layer_stats.items():
            assert stats["num_unique_tokens"] == expected_unique_tokens, \
                f"Layer {layer_name} has {stats['num_unique_tokens']} unique tokens, expected {expected_unique_tokens}"


class TestContextualEmbeddingsQwen:
    """Test Qwen2-VL contextual embeddings (different architecture)."""

    @pytest.fixture
    def contextual_embeddings_qwen2vl(self, molmo_data_dir):
        """Path to Qwen2-VL contextual embeddings."""
        return molmo_data_dir / "contextual_llm_embeddings_vg" / "Qwen_Qwen2-VL-7B-Instruct"

    def test_qwen2vl_metadata(self, contextual_embeddings_qwen2vl):
        """Test Qwen2-VL has correct layer configuration (28 layers)."""
        if not contextual_embeddings_qwen2vl.exists():
            pytest.skip("Qwen2-VL embeddings not found")

        metadata_file = contextual_embeddings_qwen2vl / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Qwen2 has 28 layers, we extract 1,2,4,8,16,24,26,27
        expected_layers = [1, 2, 4, 8, 16, 24, 26, 27]
        assert metadata["layers_extracted"] == expected_layers


class TestAllContextualEmbeddings:
    """Test all 4 LLM contextual embeddings exist and have consistent structure."""

    @pytest.fixture
    def all_contextual_dirs(self, molmo_data_dir):
        """All 4 LLM contextual embedding directories."""
        base = molmo_data_dir / "contextual_llm_embeddings_vg"
        return {
            "olmo": base / "allenai_OLMo-7B-1024-preview",
            "llama": base / "meta-llama_Meta-Llama-3-8B",
            "qwen": base / "Qwen_Qwen2-7B",
            "qwen2vl": base / "Qwen_Qwen2-VL-7B-Instruct",
        }

    def test_all_llms_have_embeddings(self, all_contextual_dirs):
        """All 4 LLMs should have contextual embeddings."""
        for name, path in all_contextual_dirs.items():
            assert path.exists(), f"Contextual embeddings for {name} not found at {path}"

            metadata_file = path / "metadata.json"
            assert metadata_file.exists(), f"metadata.json not found for {name}"


@pytest.mark.slow
@pytest.mark.gpu
class TestExtractEmbeddingsReproduction:
    """Run extract_embeddings.py on a small subset and compare to golden data.

    This tests the ACTUAL extraction pipeline end-to-end:
    1. Runs the script on 50 VG phrases with OLMo-7B, layer 8
    2. Verifies output structure matches golden format
    3. Compares actual embedding values for overlapping (token, caption) pairs
    """

    MODEL = "allenai/OLMo-7B-1024-preview"
    LAYER = 8
    NUM_CAPTIONS = 50

    @pytest.fixture(scope="class")
    def extraction_output(self):
        """Run extract_embeddings.py once and share output across tests in this class."""
        from tests.conftest import ORIGINAL_REPO_ROOT
        vg_file = ORIGINAL_REPO_ROOT / "vg_phrases.txt"
        if not vg_file.exists():
            pytest.skip(f"VG phrases file not found at {vg_file}")

        repo_root = Path(__file__).parent.parent
        script = repo_root / "scripts" / "extract_embeddings.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--model", self.MODEL,
                    "--layers", str(self.LAYER),
                    "--num-captions", str(self.NUM_CAPTIONS),
                    "--dataset", "vg",
                    "--vg-file", str(vg_file),
                    "--output-dir", tmpdir,
                    "--embedding-dtype", "float8",
                    "--seed", "42",
                ],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(repo_root),
            )
            assert result.returncode == 0, (
                f"extract_embeddings.py failed:\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
            )

            # The script appends _vg suffix and model name to output dir
            output_dir = Path(tmpdir + "_vg") / self.MODEL.replace("/", "_")
            if not output_dir.exists():
                # Try without _vg suffix
                output_dir = Path(tmpdir) / self.MODEL.replace("/", "_")
            assert output_dir.exists(), (
                f"Output dir not found. Tried {tmpdir}_vg and {tmpdir}. "
                f"ls {tmpdir}: {list(Path(tmpdir).parent.glob(Path(tmpdir).name + '*'))}"
            )

            # Load all output data into memory before tmpdir is cleaned up
            metadata = json.load(open(output_dir / "metadata.json"))

            layer_dir = output_dir / f"layer_{self.LAYER}"
            token_data = json.load(open(layer_dir / "token_embeddings.json"))

            # Load all embeddings into memory
            embeddings = {}
            for token_str, entries in token_data.items():
                for entry in entries:
                    emb_path = layer_dir / entry["embedding_path"]
                    if emb_path.exists():
                        embeddings[entry["embedding_path"]] = np.load(emb_path)

            # Load the auto-built cache
            import torch
            cache_file = layer_dir / "embeddings_cache.pt"
            cache_data = None
            if cache_file.exists():
                cache_data = torch.load(cache_file, map_location="cpu", weights_only=False)

            yield {
                "metadata": metadata,
                "token_data": token_data,
                "embeddings": embeddings,
                "cache_data": cache_data,
                "stdout": result.stdout,
            }

    def test_script_runs_successfully(self, extraction_output):
        """Script should complete without errors."""
        # If we get here, the fixture already verified returncode == 0
        assert extraction_output["metadata"] is not None

    def test_metadata_format(self, extraction_output):
        """Output metadata should match golden format."""
        meta = extraction_output["metadata"]

        assert meta["model_name"] == self.MODEL
        assert self.LAYER in meta["layers_extracted"]
        assert meta["embedding_dtype"] == "float8"
        assert meta["dataset"] == "vg"
        # num_captions_processed may be slightly less than NUM_CAPTIONS
        # due to empty lines being skipped in the VG phrases file
        assert meta["num_captions_processed"] >= self.NUM_CAPTIONS - 5
        assert "layer_statistics" in meta
        assert f"layer_{self.LAYER}" in meta["layer_statistics"]

    def test_token_data_format(self, extraction_output):
        """Token embeddings JSON should have correct structure."""
        token_data = extraction_output["token_data"]

        assert len(token_data) > 0, "No tokens extracted"

        for token_str, entries in token_data.items():
            assert isinstance(token_str, str)
            assert isinstance(entries, list)
            assert len(entries) > 0
            assert len(entries) <= 20, f"Token {token_str!r} has {len(entries)} entries (max 20)"

            for entry in entries:
                assert "embedding_path" in entry
                assert "caption" in entry
                assert "position" in entry
                assert "token_id" in entry
                assert "dtype" in entry
                assert entry["dtype"] == "float8_e4m3fn"

    def test_embedding_dimensions(self, extraction_output):
        """Embeddings should have correct dimensions for OLMo-7B (4096)."""
        import ml_dtypes

        embeddings = extraction_output["embeddings"]
        assert len(embeddings) > 0, "No embeddings loaded"

        for path, emb in embeddings.items():
            assert emb.shape == (4096,), f"Embedding {path} has shape {emb.shape}, expected (4096,)"
            # Verify it's float8 stored as raw bytes
            assert str(emb.dtype).startswith("|V") or "float8" in str(emb.dtype), (
                f"Embedding {path} has dtype {emb.dtype}, expected float8 raw bytes"
            )

            # Convert and check reasonable range
            emb_f32 = emb.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
            assert not np.any(np.isnan(emb_f32)), f"NaN in embedding {path}"
            assert np.linalg.norm(emb_f32) > 0, f"Zero-norm embedding {path}"

    def test_cache_built_automatically(self, extraction_output):
        """extract_embeddings.py should auto-build embeddings_cache.pt for run_latentlens.py."""
        import torch

        cache_data = extraction_output["cache_data"]
        assert cache_data is not None, (
            "embeddings_cache.pt was not created. "
            "extract_embeddings.py should build the cache automatically after extraction."
        )

        # Check cache structure matches what run_latentlens.py expects
        assert "embeddings" in cache_data, "Cache missing 'embeddings' tensor"
        assert "metadata" in cache_data, "Cache missing 'metadata' list"
        assert "token_to_indices" in cache_data, "Cache missing 'token_to_indices' mapping"

        embeddings_tensor = cache_data["embeddings"]
        metadata_list = cache_data["metadata"]
        token_to_indices = cache_data["token_to_indices"]

        # Tensor should be float32, shape (N, 4096)
        assert embeddings_tensor.dim() == 2, f"Expected 2D tensor, got {embeddings_tensor.dim()}D"
        assert embeddings_tensor.shape[1] == 4096, f"Expected dim 4096, got {embeddings_tensor.shape[1]}"
        assert embeddings_tensor.dtype == torch.float32

        # Metadata count should match tensor rows
        assert len(metadata_list) == embeddings_tensor.shape[0], (
            f"Metadata count ({len(metadata_list)}) != tensor rows ({embeddings_tensor.shape[0]})"
        )

        # Each metadata entry should have the expected fields
        for meta in metadata_list[:5]:  # spot check first 5
            assert "token_str" in meta
            assert "token_id" in meta
            assert "caption" in meta
            assert "position" in meta

        # Token-to-indices should map token strings to lists of indices
        assert len(token_to_indices) > 0
        for token_str, indices in list(token_to_indices.items())[:3]:
            assert isinstance(token_str, str)
            assert isinstance(indices, list)
            for idx in indices:
                assert 0 <= idx < len(metadata_list)

        # Cache embeddings should match the individual .npy files
        token_data = extraction_output["token_data"]
        npy_embeddings = extraction_output["embeddings"]
        import ml_dtypes

        matched = 0
        for token_str, entries in list(token_data.items())[:5]:
            if token_str not in token_to_indices:
                continue
            cache_indices = token_to_indices[token_str]
            for entry in entries:
                npy_raw = npy_embeddings.get(entry["embedding_path"])
                if npy_raw is None:
                    continue
                npy_f32 = npy_raw.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
                # Find matching cache entry by caption
                for ci in cache_indices:
                    if metadata_list[ci]["caption"] == entry["caption"]:
                        cache_emb = embeddings_tensor[ci].numpy()
                        assert np.allclose(npy_f32, cache_emb, atol=1e-6), (
                            f"Cache embedding doesn't match .npy for token {token_str!r}"
                        )
                        matched += 1
                        break

        assert matched > 0, "No cache-vs-npy matches found"
        print(f"\nCache validation: {matched} embeddings matched between .npy and cache")

    def test_embeddings_match_golden(self, extraction_output):
        """Embeddings for same (token, caption) pairs should match golden data exactly.

        This is the key value-comparison test. For each token in our small extraction,
        we find the same token in the golden data. If the golden data has an entry with
        the same caption, the embeddings should be byte-identical (same model, same input,
        same quantization).
        """
        import ml_dtypes
        from tests.conftest import MOLMO_DATA

        golden_base = MOLMO_DATA / "contextual_llm_embeddings_vg" / "allenai_OLMo-7B-1024-preview"
        if not golden_base.exists():
            pytest.skip(f"Golden embeddings not found at {golden_base}")
        golden_dir = golden_base / f"layer_{self.LAYER}"
        golden_token_file = golden_dir / "token_embeddings.json"
        if not golden_token_file.exists():
            pytest.skip(f"Golden token data not found at {golden_token_file}")

        golden_token_data = json.load(open(golden_token_file))
        our_token_data = extraction_output["token_data"]
        our_embeddings = extraction_output["embeddings"]

        matched = 0
        mismatched = 0
        skipped = 0

        for token_str, our_entries in our_token_data.items():
            if token_str not in golden_token_data:
                skipped += 1
                continue

            golden_entries = golden_token_data[token_str]
            # Build lookup: caption -> golden entry
            golden_by_caption = {}
            for ge in golden_entries:
                key = (ge["caption"], ge["position"])
                golden_by_caption[key] = ge

            for our_entry in our_entries:
                key = (our_entry["caption"], our_entry["position"])
                if key not in golden_by_caption:
                    skipped += 1
                    continue

                golden_entry = golden_by_caption[key]

                # Same token_id
                assert our_entry["token_id"] == golden_entry["token_id"], (
                    f"Token ID mismatch for {token_str!r} in caption {our_entry['caption']!r}: "
                    f"got {our_entry['token_id']}, expected {golden_entry['token_id']}"
                )

                # Load our embedding
                our_emb_raw = our_embeddings.get(our_entry["embedding_path"])
                if our_emb_raw is None:
                    skipped += 1
                    continue

                # Load golden embedding
                golden_emb_path = golden_dir / golden_entry["embedding_path"]
                if not golden_emb_path.exists():
                    skipped += 1
                    continue
                golden_emb_raw = np.load(golden_emb_path)

                # Convert both to float32 for comparison
                our_emb = our_emb_raw.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
                golden_emb = golden_emb_raw.view(ml_dtypes.float8_e4m3fn).astype(np.float32)

                # Cosine similarity should be very high (same model, same input)
                cos_sim = np.dot(our_emb, golden_emb) / (
                    np.linalg.norm(our_emb) * np.linalg.norm(golden_emb) + 1e-8
                )

                if cos_sim > 0.99:
                    matched += 1
                else:
                    mismatched += 1
                    # Also check if they're close in absolute terms
                    max_diff = np.max(np.abs(our_emb - golden_emb))
                    assert False, (
                        f"Embedding mismatch for token {token_str!r} "
                        f"(caption: {our_entry['caption']!r}, pos: {our_entry['position']}): "
                        f"cosine_sim={cos_sim:.6f}, max_diff={max_diff:.6f}"
                    )

        assert matched > 0, (
            f"No matching (token, caption) pairs found between extraction output and golden data. "
            f"Skipped: {skipped}. This likely means the first {self.NUM_CAPTIONS} VG phrases "
            f"don't overlap with the golden data's reservoir-sampled entries."
        )
        # Report coverage
        print(f"\nGolden comparison: {matched} matched, {mismatched} mismatched, {skipped} skipped")
