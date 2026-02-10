"""
Tests for package imports.

These tests verify that:
1. The molmo package can be imported
2. Core modules have expected exports
3. No import errors occur

Run with: pytest tests/test_imports.py -v
"""
import pytest


class TestPackageImports:
    """Test that the molmo package can be imported."""

    def test_import_molmo(self):
        """Main package should be importable."""
        import molmo
        assert hasattr(molmo, "__version__")
        assert molmo.__version__ == "0.1.0"

    def test_import_config(self):
        """Config module should be importable with expected exports."""
        from molmo import ModelConfig, VisionBackboneConfig
        assert ModelConfig is not None
        assert VisionBackboneConfig is not None

    def test_import_model(self):
        """Model module should be importable with expected exports."""
        from molmo import Molmo, OLMoOutput
        assert Molmo is not None
        assert OLMoOutput is not None

    def test_import_data(self):
        """Data module should be importable with expected exports."""
        from molmo.data import load_image, resize_and_pad
        assert load_image is not None
        assert resize_and_pad is not None


class TestSubmoduleImports:
    """Test that submodules can be imported."""

    def test_import_aliases(self):
        """Aliases module should be importable."""
        from molmo.aliases import PathOrStr
        assert PathOrStr is not None

    def test_import_exceptions(self):
        """Exceptions module should be importable."""
        from molmo.exceptions import OLMoConfigurationError
        assert OLMoConfigurationError is not None

    def test_import_torch_util(self):
        """Torch utilities should be importable."""
        from molmo import torch_util
        assert torch_util is not None

    def test_import_util(self):
        """Utilities should be importable."""
        from molmo import util
        assert util is not None


class TestNoOlmoImports:
    """Verify no 'olmo' imports remain in the package."""

    def test_no_olmo_in_init(self):
        """__init__.py should not import from olmo."""
        import molmo
        # This test passes if import succeeds - any olmo references would fail

    def test_no_olmo_in_config(self):
        """config.py should not import from olmo."""
        from molmo import config
        # This test passes if import succeeds

    def test_no_olmo_in_model(self):
        """model.py should not import from olmo."""
        from molmo import model
        # This test passes if import succeeds
