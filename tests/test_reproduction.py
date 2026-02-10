"""
Reproduction tests — run release scripts and compare to original golden outputs.

These tests actually load a model, run the release scripts on a small number of
images, and verify that the output matches the original analysis_results/ golden
files. This catches regressions from code refactoring (e.g., loop reordering).

Requirements:
    - GPU available
    - Checkpoint at molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded
    - Golden files in analysis_results/ (symlinked from original repo)

Run with:
    PYTHONPATH=. python -m pytest tests/test_reproduction.py -v -s --timeout=600

    Or skip these in CI:
    pytest -m "not slow"
"""
import json
import subprocess
import sys
import tempfile
import pytest
from pathlib import Path

from tests.conftest import MOLMO_DATA, ANALYSIS_RESULTS


# Mark all tests as slow + gpu
pytestmark = [pytest.mark.slow, pytest.mark.gpu]

# Tolerances for fp16 numerical comparison
LOGIT_ATOL = 0.05    # absolute tolerance for logit values
SIM_ATOL = 0.005     # absolute tolerance for cosine similarity values


def _run_script(script_name, args, cwd, timeout=600):
    """Run a release script and return subprocess result."""
    cmd = [sys.executable, f"reproduce/scripts/{script_name}"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd),
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
    return result


def _compare_logitlens_to_golden(release_data, golden_data, layer, num_images=2):
    """Compare LogitLens release output to golden data for a specific layer.

    Checks:
    - Same image indices
    - Same patch count
    - Top-5 token strings match exactly (sample of patches)
    - Logit values within tolerance
    """
    matched_patches = 0

    for img_idx in range(num_images):
        release_img = release_data["results"][img_idx]
        golden_img = golden_data["results"][img_idx]

        assert release_img["image_idx"] == golden_img["image_idx"], \
            f"L{layer} image index mismatch: {release_img['image_idx']} vs {golden_img['image_idx']}"

        release_patches = release_img["chunks"][0]["patches"]
        golden_patches = golden_img["chunks"][0]["patches"]

        assert len(release_patches) == len(golden_patches), \
            f"L{layer} image {img_idx}: patch count {len(release_patches)} vs {len(golden_patches)}"

        sample_indices = [0, len(release_patches) // 2, len(release_patches) - 1]
        for pidx in sample_indices:
            r_patch = release_patches[pidx]
            g_patch = golden_patches[pidx]

            r_tokens = [p["token"] for p in r_patch["top_predictions"]]
            g_tokens = [p["token"] for p in g_patch["top_predictions"]]
            assert r_tokens == g_tokens, \
                f"L{layer} image {img_idx}, patch {pidx}: tokens differ.\n" \
                f"  Release: {r_tokens}\n  Golden:  {g_tokens}"

            for k in range(5):
                r_logit = r_patch["top_predictions"][k]["logit"]
                g_logit = g_patch["top_predictions"][k]["logit"]
                assert abs(r_logit - g_logit) < LOGIT_ATOL, \
                    f"L{layer} image {img_idx}, patch {pidx}, rank {k}: " \
                    f"logit {r_logit} vs golden {g_logit}"

            matched_patches += 1

    return matched_patches


def _compare_embedding_lens_to_golden(release_data, golden_data, layer, num_images=2):
    """Compare EmbeddingLens release output to golden data for a specific layer.

    Checks:
    - Same image indices
    - Same patch count
    - Top-5 token strings match exactly (sample of patches)
    - Similarity values within tolerance
    """
    matched_patches = 0

    release_images = release_data["splits"]["validation"]["images"]
    golden_images = golden_data["splits"]["validation"]["images"]

    for img_idx in range(num_images):
        release_img = release_images[img_idx]
        golden_img = golden_images[img_idx]

        assert release_img["image_idx"] == golden_img["image_idx"], \
            f"L{layer} image index mismatch: {release_img['image_idx']} vs {golden_img['image_idx']}"

        release_patches = release_img["chunks"][0]["patches"]
        golden_patches = golden_img["chunks"][0]["patches"]

        assert len(release_patches) == len(golden_patches), \
            f"L{layer} image {img_idx}: patch count {len(release_patches)} vs {len(golden_patches)}"

        sample_indices = [0, len(release_patches) // 2, len(release_patches) - 1]
        for pidx in sample_indices:
            r_patch = release_patches[pidx]
            g_patch = golden_patches[pidx]

            r_tokens = [n["token"] for n in r_patch["nearest_neighbors"]]
            g_tokens = [n["token"] for n in g_patch["nearest_neighbors"]]
            assert r_tokens == g_tokens, \
                f"L{layer} image {img_idx}, patch {pidx}: tokens differ.\n" \
                f"  Release: {r_tokens}\n  Golden:  {g_tokens}"

            for k in range(min(5, len(r_tokens))):
                r_sim = r_patch["nearest_neighbors"][k]["similarity"]
                g_sim = g_patch["nearest_neighbors"][k]["similarity"]
                assert abs(r_sim - g_sim) < SIM_ATOL, \
                    f"L{layer} image {img_idx}, patch {pidx}, rank {k}: " \
                    f"similarity {r_sim} vs golden {g_sim}"

            matched_patches += 1

    return matched_patches


class TestLogitLensReproduction:
    """Run release LogitLens on 2 images across multiple LLM layers, compare to golden."""

    # Test early (0), mid (8), and late (24) LLM layers in a single script run.
    LLM_LAYERS = [0, 8, 24]

    @pytest.fixture(scope="class")
    def golden_logitlens_multi(self):
        """Load golden LogitLens outputs for multiple layers."""
        golden_dir = (
            ANALYSIS_RESULTS /
            "logit_lens" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10"
        )
        if not golden_dir.exists():
            pytest.skip(f"Golden directory not found: {golden_dir}")

        golden = {}
        for layer in self.LLM_LAYERS:
            path = golden_dir / f"logit_lens_layer{layer}_topk5_multi-gpu.json"
            if not path.exists():
                pytest.skip(f"Golden file not found: {path}")
            with open(path) as f:
                golden[layer] = json.load(f)
        return golden

    @pytest.fixture(scope="class")
    def logitlens_run_output(self):
        """Run run_logitlens.py once with multiple layers, share across tests."""
        checkpoint = MOLMO_DATA / "checkpoints" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336" / "step12000-unsharded"
        if not checkpoint.exists():
            pytest.skip("Checkpoint not found")

        release_dir = Path(__file__).parent.parent
        layer_str = ",".join(str(l) for l in self.LLM_LAYERS)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_script(
                "run_logitlens.py",
                [
                    "--ckpt-path", str(checkpoint),
                    "--layers", layer_str,
                    "--num-images", "2",
                    "--top-k", "5",
                    "--output-dir", tmpdir,
                ],
                cwd=release_dir,
                timeout=600,
            )
            assert result.returncode == 0, (
                f"run_logitlens.py failed:\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
            )

            # Load all output files into memory before tmpdir cleanup
            outputs = {}
            for layer in self.LLM_LAYERS:
                files = list(Path(tmpdir).rglob(f"logit_lens_layer{layer}_topk5_multi-gpu.json"))
                assert len(files) == 1, f"Expected 1 output for layer {layer}, found {len(files)}"
                with open(files[0]) as f:
                    outputs[layer] = json.load(f)

            yield {"outputs": outputs, "stdout": result.stdout}

    def test_script_completes(self, logitlens_run_output):
        """Script should complete without errors for all layers."""
        assert len(logitlens_run_output["outputs"]) == len(self.LLM_LAYERS)

    def test_layer0_matches_golden(self, logitlens_run_output, golden_logitlens_multi):
        """LLM layer 0 (early) output should match golden data."""
        matched = _compare_logitlens_to_golden(
            logitlens_run_output["outputs"][0], golden_logitlens_multi[0], layer=0
        )
        print(f"\nLogitLens L0: {matched} patches matched")

    def test_layer8_matches_golden(self, logitlens_run_output, golden_logitlens_multi):
        """LLM layer 8 (mid) output should match golden data."""
        matched = _compare_logitlens_to_golden(
            logitlens_run_output["outputs"][8], golden_logitlens_multi[8], layer=8
        )
        print(f"\nLogitLens L8: {matched} patches matched")

    def test_layer24_matches_golden(self, logitlens_run_output, golden_logitlens_multi):
        """LLM layer 24 (late) output should match golden data."""
        matched = _compare_logitlens_to_golden(
            logitlens_run_output["outputs"][24], golden_logitlens_multi[24], layer=24
        )
        print(f"\nLogitLens L24: {matched} patches matched")


class TestEmbeddingLensReproduction:
    """Run release EmbeddingLens on 2 images across multiple LLM layers, compare to golden."""

    # Test early (0), mid (8), and late (24) LLM layers in a single script run.
    LLM_LAYERS = [0, 8, 24]

    @pytest.fixture(scope="class")
    def golden_embedding_lens_multi(self):
        """Load golden EmbeddingLens outputs for multiple layers."""
        golden_dir = (
            ANALYSIS_RESULTS /
            "nearest_neighbors" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10"
        )
        if not golden_dir.exists():
            pytest.skip(f"Golden directory not found: {golden_dir}")

        golden = {}
        for layer in self.LLM_LAYERS:
            path = golden_dir / f"nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{layer}.json"
            if not path.exists():
                pytest.skip(f"Golden file not found: {path}")
            with open(path) as f:
                golden[layer] = json.load(f)
        return golden

    @pytest.fixture(scope="class")
    def embedding_lens_run_output(self):
        """Run run_embedding_lens.py once with multiple layers, share across tests."""
        checkpoint = MOLMO_DATA / "checkpoints" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336" / "step12000-unsharded"
        if not checkpoint.exists():
            pytest.skip("Checkpoint not found")

        release_dir = Path(__file__).parent.parent
        layer_str = ",".join(str(l) for l in self.LLM_LAYERS)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_script(
                "run_embedding_lens.py",
                [
                    "--ckpt-path", str(checkpoint),
                    "--llm_layer", layer_str,
                    "--num-images", "2",
                    "--output-base-dir", tmpdir,
                ],
                cwd=release_dir,
                timeout=600,
            )
            assert result.returncode == 0, (
                f"run_embedding_lens.py failed:\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
            )

            # Load all output files into memory before tmpdir cleanup
            outputs = {}
            for layer in self.LLM_LAYERS:
                files = list(Path(tmpdir).rglob(f"nearest_neighbors_*_layer{layer}.json"))
                assert len(files) == 1, f"Expected 1 output for layer {layer}, found {len(files)}"
                with open(files[0]) as f:
                    outputs[layer] = json.load(f)

            yield {"outputs": outputs, "stdout": result.stdout}

    def test_script_completes(self, embedding_lens_run_output):
        """Script should complete without errors for all layers."""
        assert len(embedding_lens_run_output["outputs"]) == len(self.LLM_LAYERS)

    def test_layer0_matches_golden(self, embedding_lens_run_output, golden_embedding_lens_multi):
        """LLM layer 0 (early) output should match golden data."""
        matched = _compare_embedding_lens_to_golden(
            embedding_lens_run_output["outputs"][0], golden_embedding_lens_multi[0], layer=0
        )
        print(f"\nEmbeddingLens L0: {matched} patches matched")

    def test_layer8_matches_golden(self, embedding_lens_run_output, golden_embedding_lens_multi):
        """LLM layer 8 (mid) output should match golden data."""
        matched = _compare_embedding_lens_to_golden(
            embedding_lens_run_output["outputs"][8], golden_embedding_lens_multi[8], layer=8
        )
        print(f"\nEmbeddingLens L8: {matched} patches matched")

    def test_layer24_matches_golden(self, embedding_lens_run_output, golden_embedding_lens_multi):
        """LLM layer 24 (late) output should match golden data."""
        matched = _compare_embedding_lens_to_golden(
            embedding_lens_run_output["outputs"][24], golden_embedding_lens_multi[24], layer=24
        )
        print(f"\nEmbeddingLens L24: {matched} patches matched")


def _compare_latentlens_to_golden(release_data, golden_data, visual_layer, num_images=2):
    """Compare LatentLens release output to golden data for a specific visual layer.

    Checks:
    - Same image indices
    - Same patch count
    - Top-5 token strings match exactly (sample of patches)
    - Similarity values within tolerance
    """
    matched_patches = 0

    for img_idx in range(num_images):
        release_img = release_data["results"][img_idx]
        golden_img = golden_data["results"][img_idx]

        assert release_img["image_idx"] == golden_img["image_idx"], \
            f"vl{visual_layer} image index mismatch: {release_img['image_idx']} vs {golden_img['image_idx']}"

        release_patches = release_img["chunks"][0]["patches"]
        golden_patches = golden_img["chunks"][0]["patches"]

        assert len(release_patches) == len(golden_patches), \
            f"vl{visual_layer} image {img_idx}: patch count {len(release_patches)} vs {len(golden_patches)}"

        # Check sample patches (first, middle, last)
        sample_indices = [0, len(release_patches) // 2, len(release_patches) - 1]
        for pidx in sample_indices:
            r_neighbors = release_patches[pidx]["nearest_contextual_neighbors"]
            g_neighbors = golden_patches[pidx]["nearest_contextual_neighbors"]

            assert len(r_neighbors) == len(g_neighbors), \
                f"vl{visual_layer} image {img_idx}, patch {pidx}: neighbor count differs"

            r_tokens = [n["token_str"] for n in r_neighbors]
            g_tokens = [n["token_str"] for n in g_neighbors]
            assert r_tokens == g_tokens, \
                f"vl{visual_layer} image {img_idx}, patch {pidx}: tokens differ.\n" \
                f"  Release: {r_tokens}\n  Golden:  {g_tokens}"

            for k in range(len(r_neighbors)):
                r_sim = r_neighbors[k]["similarity"]
                g_sim = g_neighbors[k]["similarity"]
                assert abs(r_sim - g_sim) < SIM_ATOL, \
                    f"vl{visual_layer} image {img_idx}, patch {pidx}, rank {k}: " \
                    f"similarity {r_sim:.6f} vs golden {g_sim:.6f}"

            matched_patches += 1

    return matched_patches


class TestLatentLensReproduction:
    """Run release LatentLens on 2 images across multiple visual layers, compare to golden."""

    # Test early (1), mid (8), and late (24) visual layers in a single script run.
    VISUAL_LAYERS = [1, 8, 24]

    @pytest.fixture(scope="class")
    def golden_latentlens_multi(self):
        """Load golden LatentLens outputs for multiple visual layers."""
        golden_dir = (
            ANALYSIS_RESULTS /
            "contextual_nearest_neighbors" /
            "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10"
        )
        if not golden_dir.exists():
            pytest.skip(f"Golden directory not found: {golden_dir}")

        golden = {}
        for vl in self.VISUAL_LAYERS:
            path = golden_dir / f"contextual_neighbors_visual{vl}_allLayers.json"
            if not path.exists():
                pytest.skip(f"Golden file not found: {path}")
            with open(path) as f:
                golden[vl] = json.load(f)
        return golden

    @pytest.fixture(scope="class")
    def latentlens_run_output(self):
        """Run run_latentlens.py once with multiple visual layers, share across tests."""
        checkpoint = MOLMO_DATA / "checkpoints" / "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336" / "step12000-unsharded"
        if not checkpoint.exists():
            pytest.skip("Checkpoint not found")

        contextual_dir = MOLMO_DATA / "contextual_llm_embeddings_vg" / "allenai_OLMo-7B-1024-preview"
        if not contextual_dir.exists():
            pytest.skip(f"Contextual embeddings not found: {contextual_dir}")

        release_dir = Path(__file__).parent.parent
        visual_layer_str = ",".join(str(vl) for vl in self.VISUAL_LAYERS)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_script(
                "run_latentlens.py",
                [
                    "--ckpt-path", str(checkpoint),
                    "--contextual-dir", str(contextual_dir),
                    "--visual-layer", visual_layer_str,
                    "--num-images", "2",
                    "--output-dir", tmpdir,
                ],
                cwd=release_dir,
                timeout=600,
            )
            assert result.returncode == 0, (
                f"run_latentlens.py failed:\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
            )

            # Load all output files into memory before tmpdir cleanup
            outputs = {}
            for vl in self.VISUAL_LAYERS:
                files = list(Path(tmpdir).rglob(f"contextual_neighbors_visual{vl}_*.json"))
                assert len(files) == 1, f"Expected 1 output for vl{vl}, found {len(files)}"
                with open(files[0]) as f:
                    outputs[vl] = json.load(f)

            yield {"outputs": outputs, "stdout": result.stdout}

    def test_script_completes(self, latentlens_run_output):
        """Script should complete without errors for all visual layers."""
        assert len(latentlens_run_output["outputs"]) == len(self.VISUAL_LAYERS)

    def test_layer1_matches_golden(self, latentlens_run_output, golden_latentlens_multi):
        """Visual layer 1 (early) output should match golden data."""
        matched = _compare_latentlens_to_golden(
            latentlens_run_output["outputs"][1], golden_latentlens_multi[1], visual_layer=1
        )
        print(f"\nvl1: {matched} patches matched")

    def test_layer8_matches_golden(self, latentlens_run_output, golden_latentlens_multi):
        """Visual layer 8 (mid) output should match golden data."""
        matched = _compare_latentlens_to_golden(
            latentlens_run_output["outputs"][8], golden_latentlens_multi[8], visual_layer=8
        )
        print(f"\nvl8: {matched} patches matched")

    def test_layer24_matches_golden(self, latentlens_run_output, golden_latentlens_multi):
        """Visual layer 24 (late) output should match golden data."""
        matched = _compare_latentlens_to_golden(
            latentlens_run_output["outputs"][24], golden_latentlens_multi[24], visual_layer=24
        )
        print(f"\nvl24: {matched} patches matched")


class TestEmbeddingLensMultiLayerEfficiency:
    """Verify the multi-layer optimization produces correct results.

    This test runs EmbeddingLens with multiple layers in a single call and
    verifies that processing all layers together (one forward pass per image)
    produces the same results as processing each layer individually.
    """

    def test_multi_layer_matches_single_layer(self, checkpoint_olmo_vit):
        """Multi-layer mode should produce identical results to single-layer mode."""
        if not checkpoint_olmo_vit.exists():
            pytest.skip("Checkpoint not found")

        release_dir = Path(__file__).parent.parent

        with tempfile.TemporaryDirectory() as tmpdir:
            single_dir = Path(tmpdir) / "single"
            multi_dir = Path(tmpdir) / "multi"

            # Run layer 0 and layer 8 separately
            for layer in ["0", "8"]:
                result = _run_script(
                    "run_embedding_lens.py",
                    [
                        "--ckpt-path", str(checkpoint_olmo_vit),
                        "--llm_layer", layer,
                        "--num-images", "2",
                        "--output-base-dir", str(single_dir),
                    ],
                    cwd=release_dir,
                )
                assert result.returncode == 0, f"Single-layer run for layer {layer} failed"

            # Run layers 0 and 8 together
            result = _run_script(
                "run_embedding_lens.py",
                [
                    "--ckpt-path", str(checkpoint_olmo_vit),
                    "--llm_layer", "0,8",
                    "--num-images", "2",
                    "--output-base-dir", str(multi_dir),
                ],
                cwd=release_dir,
            )
            assert result.returncode == 0, "Multi-layer run failed"

            # Compare output for each layer
            for layer in [0, 8]:
                single_files = list(single_dir.rglob(f"*_layer{layer}.json"))
                multi_files = list(multi_dir.rglob(f"*_layer{layer}.json"))

                assert len(single_files) == 1, f"Expected 1 single-layer file for layer {layer}"
                assert len(multi_files) == 1, f"Expected 1 multi-layer file for layer {layer}"

                with open(single_files[0]) as f:
                    single_data = json.load(f)
                with open(multi_files[0]) as f:
                    multi_data = json.load(f)

                # Compare validation images
                single_images = single_data["splits"]["validation"]["images"]
                multi_images = multi_data["splits"]["validation"]["images"]

                assert len(single_images) == len(multi_images)

                for img_idx in range(len(single_images)):
                    s_patches = single_images[img_idx]["chunks"][0]["patches"]
                    m_patches = multi_images[img_idx]["chunks"][0]["patches"]

                    for pidx in range(len(s_patches)):
                        s_tokens = [n["token"] for n in s_patches[pidx]["nearest_neighbors"]]
                        m_tokens = [n["token"] for n in m_patches[pidx]["nearest_neighbors"]]
                        assert s_tokens == m_tokens, \
                            f"Layer {layer}, image {img_idx}, patch {pidx}: " \
                            f"tokens differ between single/multi mode"
