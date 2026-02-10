"""Tests that shell scripts use the correct argument names for Python scripts.

The shell scripts (reproduce/step3_run_analysis.sh) must use argument names that
match the argparse definitions in the Python scripts. These names have been a
recurring source of bugs (e.g., --llm-layer vs --layers).
"""
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "reproduce" / "scripts"
REPRODUCE_DIR = REPO_ROOT / "reproduce"


def extract_argparse_flags(script_path):
    """Extract all --flag names from argparse add_argument calls."""
    source = script_path.read_text()
    flags = set()
    for match in re.finditer(r'add_argument\(\s*["\'](-{1,2}[\w-]+)["\']', source):
        flags.add(match.group(1))
    return flags


def extract_shell_flags(shell_content, script_name):
    """Extract --flag names used in a shell script when calling a specific Python script."""
    flags = set()
    # Find lines that invoke the given script and subsequent continuation lines
    in_command = False
    for line in shell_content.split('\n'):
        stripped = line.strip()
        if script_name in stripped:
            in_command = True
        if in_command:
            for match in re.finditer(r'(--[\w_-]+)', stripped):
                flag = match.group(1)
                # Skip flags that are part of torchrun, not the python script
                if flag in ('--nproc_per_node', '--master_port'):
                    continue
                flags.add(flag)
            # If line doesn't end with \, command is done
            if not stripped.endswith('\\'):
                in_command = False
    return flags


class TestShellScriptArgNames:
    """Verify shell script arguments match Python argparse definitions."""

    def test_latentlens_args(self):
        """step3 should use correct args for run_latentlens.py."""
        py_flags = extract_argparse_flags(SCRIPTS_DIR / "run_latentlens.py")
        shell = (REPRODUCE_DIR / "step3_run_analysis.sh").read_text()
        shell_flags = extract_shell_flags(shell, "run_latentlens.py")

        for flag in shell_flags:
            # argparse accepts both --foo-bar and --foo_bar
            normalized = flag.lstrip('-').replace('-', '_')
            matches = any(
                f.lstrip('-').replace('-', '_') == normalized
                for f in py_flags
            )
            assert matches, (
                f"Shell uses {flag} for run_latentlens.py, but argparse only defines: {sorted(py_flags)}"
            )

    def test_logitlens_args(self):
        """step3 should use correct args for run_logitlens.py."""
        py_flags = extract_argparse_flags(SCRIPTS_DIR / "run_logitlens.py")
        shell = (REPRODUCE_DIR / "step3_run_analysis.sh").read_text()
        shell_flags = extract_shell_flags(shell, "run_logitlens.py")

        for flag in shell_flags:
            normalized = flag.lstrip('-').replace('-', '_')
            matches = any(
                f.lstrip('-').replace('-', '_') == normalized
                for f in py_flags
            )
            assert matches, (
                f"Shell uses {flag} for run_logitlens.py, but argparse only defines: {sorted(py_flags)}"
            )

    def test_embedding_lens_args(self):
        """step3 should use correct args for run_embedding_lens.py."""
        py_flags = extract_argparse_flags(SCRIPTS_DIR / "run_embedding_lens.py")
        shell = (REPRODUCE_DIR / "step3_run_analysis.sh").read_text()
        shell_flags = extract_shell_flags(shell, "run_embedding_lens.py")

        for flag in shell_flags:
            normalized = flag.lstrip('-').replace('-', '_')
            matches = any(
                f.lstrip('-').replace('-', '_') == normalized
                for f in py_flags
            )
            assert matches, (
                f"Shell uses {flag} for run_embedding_lens.py, but argparse only defines: {sorted(py_flags)}"
            )


class TestStep2ArgNames:
    """Verify step2_extract_contextual.sh uses correct args for extract_embeddings.py."""

    def test_extract_embeddings_args(self):
        """step2 should use correct args for extract_embeddings.py."""
        py_flags = extract_argparse_flags(SCRIPTS_DIR / "extract_embeddings.py")
        shell = (REPRODUCE_DIR / "step2_extract_contextual.sh").read_text()
        shell_flags = extract_shell_flags(shell, "extract_embeddings.py")

        for flag in shell_flags:
            normalized = flag.lstrip('-').replace('-', '_')
            matches = any(
                f.lstrip('-').replace('-', '_') == normalized
                for f in py_flags
            )
            assert matches, (
                f"Shell uses {flag} for extract_embeddings.py, but argparse only defines: {sorted(py_flags)}"
            )

    def test_step2_model_ids_are_hf_format(self):
        """step2 should use HuggingFace model IDs (org/model format)."""
        content = (REPRODUCE_DIR / "step2_extract_contextual.sh").read_text()
        # Should use HF IDs like "allenai/OLMo-7B-1024-preview", not paths
        assert "allenai/OLMo" in content
        assert "meta-llama/Meta-Llama" in content
        assert "Qwen/Qwen2" in content


class TestShellScriptStructure:
    """Verify shell script has required structure."""

    def test_step3_exists(self):
        assert (REPRODUCE_DIR / "step3_run_analysis.sh").exists()

    def test_step3_has_all_methods(self):
        """Shell script should have functions for all 3 analysis methods."""
        content = (REPRODUCE_DIR / "step3_run_analysis.sh").read_text()
        assert "run_latentlens" in content
        assert "run_logitlens" in content
        assert "run_embedding_lens" in content

    def test_step3_has_all_9_models(self):
        """Shell script should define all 9 model combinations."""
        content = (REPRODUCE_DIR / "step3_run_analysis.sh").read_text()
        # 6 models with 32 layers + 3 models with 28 layers = 9
        model_lines = re.findall(r'"[\w-]+:[\w-]+:[\d,]+:[\w._-]+"', content)
        assert len(model_lines) == 9, f"Expected 9 models, found {len(model_lines)}: {model_lines}"

    def test_step3_supports_single_gpu(self):
        """Shell script should support single-GPU mode (no mandatory torchrun)."""
        content = (REPRODUCE_DIR / "step3_run_analysis.sh").read_text()
        # LatentLens should always use plain python (no torchrun)
        assert 'python reproduce/scripts/run_latentlens.py' in content
        # LogitLens/EmbeddingLens should use $RUN_CMD which defaults to python
        assert '$RUN_CMD reproduce/scripts/run_logitlens.py' in content
        assert '$RUN_CMD reproduce/scripts/run_embedding_lens.py' in content


class TestShellScriptRepoRoot:
    """Verify all shell scripts cd to repo root for relative path safety."""

    @pytest.mark.parametrize("script", [
        "step1_download.sh",
        "step2_extract_contextual.sh",
        "step3_run_analysis.sh",
        "run_all.sh",
    ])
    def test_script_cds_to_repo_root(self, script):
        """Every reproduce script should cd to repo root."""
        content = (REPRODUCE_DIR / script).read_text()
        assert 'cd "$(dirname "$0")/.."' in content, (
            f"{script} should cd to repo root for correct relative paths"
        )


class TestShellScriptModelSpecs:
    """Verify the model specifications in step3 are correct and complete."""

    EXPECTED_MODELS = {
        # checkpoint_name -> (contextual_name, layers, hf_model_dir)
        "olmo-vit": ("olmo-7b", "0,1,2,4,8,16,24,30,31", "allenai_OLMo-7B-1024-preview"),
        "olmo-dino": ("olmo-7b", "0,1,2,4,8,16,24,30,31", "allenai_OLMo-7B-1024-preview"),
        "olmo-siglip": ("olmo-7b", "0,1,2,4,8,16,24,30,31", "allenai_OLMo-7B-1024-preview"),
        "llama-vit": ("llama3-8b", "0,1,2,4,8,16,24,30,31", "meta-llama_Meta-Llama-3-8B"),
        "llama-dino": ("llama3-8b", "0,1,2,4,8,16,24,30,31", "meta-llama_Meta-Llama-3-8B"),
        "llama-siglip": ("llama3-8b", "0,1,2,4,8,16,24,30,31", "meta-llama_Meta-Llama-3-8B"),
        "qwen-vit": ("qwen2-7b", "0,1,2,4,8,16,24,26,27", "Qwen_Qwen2-7B"),
        "qwen-dino": ("qwen2-7b", "0,1,2,4,8,16,24,26,27", "Qwen_Qwen2-7B"),
        "qwen-siglip": ("qwen2-7b", "0,1,2,4,8,16,24,26,27", "Qwen_Qwen2-7B"),
    }

    @pytest.fixture
    def model_specs(self):
        """Parse model specs from step3_run_analysis.sh."""
        content = (REPRODUCE_DIR / "step3_run_analysis.sh").read_text()
        specs = {}
        for match in re.finditer(r'"([\w-]+):([\w-]+):([\d,]+):([\w._-]+)"', content):
            ckpt, ctx, layers, hf_dir = match.groups()
            specs[ckpt] = (ctx, layers, hf_dir)
        return specs

    def test_all_9_models_present(self, model_specs):
        """Every expected model combination should be in the script."""
        for model in self.EXPECTED_MODELS:
            assert model in model_specs, f"Missing model: {model}"

    def test_no_extra_models(self, model_specs):
        """No unexpected models should be in the script."""
        for model in model_specs:
            assert model in self.EXPECTED_MODELS, f"Unexpected model: {model}"

    def test_olmo_models_use_olmo_contextual(self, model_specs):
        for name in ("olmo-vit", "olmo-dino", "olmo-siglip"):
            assert model_specs[name][0] == "olmo-7b", f"{name} should use olmo-7b contextual"

    def test_llama_models_use_llama_contextual(self, model_specs):
        for name in ("llama-vit", "llama-dino", "llama-siglip"):
            assert model_specs[name][0] == "llama3-8b", f"{name} should use llama3-8b contextual"

    def test_qwen_models_use_qwen_contextual(self, model_specs):
        for name in ("qwen-vit", "qwen-dino", "qwen-siglip"):
            assert model_specs[name][0] == "qwen2-7b", f"{name} should use qwen2-7b contextual"

    def test_32_layer_models_have_correct_layers(self, model_specs):
        """OLMo and LLaMA (32-layer) should analyze layers 0,1,2,4,8,16,24,30,31."""
        expected_layers = "0,1,2,4,8,16,24,30,31"
        for name in ("olmo-vit", "olmo-dino", "olmo-siglip",
                      "llama-vit", "llama-dino", "llama-siglip"):
            assert model_specs[name][1] == expected_layers, (
                f"{name}: expected layers {expected_layers}, got {model_specs[name][1]}"
            )

    def test_28_layer_models_have_correct_layers(self, model_specs):
        """Qwen (28-layer) should analyze layers 0,1,2,4,8,16,24,26,27."""
        expected_layers = "0,1,2,4,8,16,24,26,27"
        for name in ("qwen-vit", "qwen-dino", "qwen-siglip"):
            assert model_specs[name][1] == expected_layers, (
                f"{name}: expected layers {expected_layers}, got {model_specs[name][1]}"
            )

    def test_hf_model_dirs_match_expected(self, model_specs):
        """Each model spec should include the correct HF model directory."""
        for name, expected in self.EXPECTED_MODELS.items():
            assert model_specs[name][2] == expected[2], (
                f"{name}: expected hf_dir {expected[2]}, got {model_specs[name][2]}"
            )

    def test_latentlens_uses_hf_dir_in_contextual_path(self):
        """run_latentlens should pass contextual-dir with hf_dir subpath."""
        content = (REPRODUCE_DIR / "step3_run_analysis.sh").read_text()
        assert "$CONTEXTUAL_BASE/$ctx/$hf_dir" in content, (
            "LatentLens should use $CONTEXTUAL_BASE/$ctx/$hf_dir for --contextual-dir"
        )
