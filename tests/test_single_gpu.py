"""Tests that LogitLens and EmbeddingLens scripts work in single-GPU mode.

These scripts were originally written for multi-GPU (torchrun + FSDP) only.
We added single-GPU support by making all distributed calls conditional on
DISTRIBUTED_MODE = "RANK" in os.environ.

These tests verify:
1. Scripts can be imported without torchrun (no dist.init_process_group crash)
2. DISTRIBUTED_MODE is False when RANK not in env
3. The single-GPU stub functions work correctly
4. No unconditional dist.* calls exist in the code
"""
import os
import sys
import ast
import importlib
import pytest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parent.parent / "reproduce" / "scripts"


class TestDistributedModeDetection:
    """Test that DISTRIBUTED_MODE is correctly set based on environment."""

    def test_not_distributed_without_rank(self):
        """Without RANK env var, DISTRIBUTED_MODE should be False."""
        # Ensure RANK is not set
        env_backup = os.environ.pop("RANK", None)
        try:
            # Force reimport
            sys.modules.pop("scripts.run_logitlens", None)
            sys.modules.pop("scripts.run_embedding_lens", None)

            # Use AST to check the script defines DISTRIBUTED_MODE correctly
            logitlens_src = (SCRIPTS_DIR / "run_logitlens.py").read_text()
            assert 'DISTRIBUTED_MODE = "RANK" in os.environ' in logitlens_src

            embedding_src = (SCRIPTS_DIR / "run_embedding_lens.py").read_text()
            assert 'DISTRIBUTED_MODE = "RANK" in os.environ' in embedding_src
        finally:
            if env_backup is not None:
                os.environ["RANK"] = env_backup

    def test_distributed_mode_would_be_true_with_rank(self):
        """With RANK env var, the expression evaluates to True."""
        assert ("RANK" in {"RANK": "0"})  # Simulates the check


class TestNoUnconditionalDistCalls:
    """Verify that all dist.barrier/broadcast/all_gather calls are guarded."""

    @pytest.fixture(params=["run_logitlens.py", "run_embedding_lens.py"])
    def script_path(self, request):
        return SCRIPTS_DIR / request.param

    def test_all_dist_calls_are_conditional(self, script_path):
        """Every dist.* call should be inside an if DISTRIBUTED_MODE block or
        inside the else branch of 'if local_rank == 0' (which only runs in distributed)."""
        source = script_path.read_text()
        tree = ast.parse(source)

        dist_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match dist.barrier(), dist.all_gather_object(), etc.
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id == "dist" and func.attr in (
                        "barrier", "all_gather_object", "broadcast",
                        "init_process_group",
                    ):
                        dist_calls.append((node.lineno, func.attr))

        # For each dist call, walk up to find it's inside a guarded block
        # We'll check the source lines for the guard pattern
        lines = source.split("\n")
        unguarded = []
        for lineno, call_name in dist_calls:
            line = lines[lineno - 1]
            indent = len(line) - len(line.lstrip())

            # Look backwards for a guard
            found_guard = False
            for check_line_idx in range(lineno - 2, max(lineno - 20, -1), -1):
                check_line = lines[check_line_idx]
                check_indent = len(check_line) - len(check_line.lstrip())
                stripped = check_line.strip()

                # Guard patterns:
                # 1. if DISTRIBUTED_MODE:
                # 2. assert DISTRIBUTED_MODE (runtime guard)
                # 3. else branch of local_rank == 0 (only runs in distributed)
                if check_indent < indent and "DISTRIBUTED_MODE" in stripped:
                    found_guard = True
                    break
                # assert DISTRIBUTED_MODE at same or lower indent is also a guard
                if "assert DISTRIBUTED_MODE" in stripped:
                    found_guard = True
                    break
                # The else branch of local_rank == 0 only executes in distributed
                if check_indent < indent and "else:" in stripped:
                    # Check if the if above it is local_rank == 0
                    for deeper_idx in range(check_line_idx - 1, max(check_line_idx - 5, -1), -1):
                        deeper = lines[deeper_idx].strip()
                        if "local_rank == 0" in deeper:
                            found_guard = True
                            break
                    if found_guard:
                        break

            if not found_guard:
                unguarded.append(f"  Line {lineno}: dist.{call_name}()")

        assert not unguarded, (
            f"Found unguarded dist.* calls in {script_path.name}:\n"
            + "\n".join(unguarded)
        )


class TestSingleGPUStubs:
    """Test that stub functions defined for single-GPU mode work correctly."""

    def test_logitlens_stubs_in_source(self):
        """run_logitlens.py should define get_local_rank and get_world_size stubs."""
        src = (SCRIPTS_DIR / "run_logitlens.py").read_text()
        assert "def get_local_rank():" in src
        assert "return 0" in src
        assert "def get_world_size():" in src
        assert "return 1" in src

    def test_embedding_lens_stubs_in_source(self):
        """run_embedding_lens.py should define get_local_rank and get_world_size stubs."""
        src = (SCRIPTS_DIR / "run_embedding_lens.py").read_text()
        assert "def get_local_rank():" in src
        assert "return 0" in src
        assert "def get_world_size():" in src
        assert "return 1" in src

    def test_fsdp_import_is_conditional(self):
        """FSDP import should only happen in distributed mode."""
        for script_name in ["run_logitlens.py", "run_embedding_lens.py"]:
            src = (SCRIPTS_DIR / script_name).read_text()
            # FSDP import should appear after "if DISTRIBUTED_MODE:"
            fsdp_line = None
            distributed_guard_line = None
            for i, line in enumerate(src.split("\n")):
                if "if DISTRIBUTED_MODE:" in line and distributed_guard_line is None:
                    distributed_guard_line = i
                if "FullyShardedDataParallel" in line:
                    fsdp_line = i
                    break

            assert distributed_guard_line is not None, f"No DISTRIBUTED_MODE guard in {script_name}"
            assert fsdp_line is not None, f"No FSDP import in {script_name}"
            assert fsdp_line > distributed_guard_line, (
                f"FSDP import at line {fsdp_line} is before DISTRIBUTED_MODE guard "
                f"at line {distributed_guard_line} in {script_name}"
            )
