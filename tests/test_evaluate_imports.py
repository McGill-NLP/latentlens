"""
Test that evaluation scripts can be imported without errors.

Run with: pytest tests/test_evaluate_imports.py -v
"""

import pytest


class TestEvaluateImports:
    """Test that evaluation module imports work."""

    def test_import_utils(self):
        """Utils module should import without errors."""
        from reproduce.scripts.evaluate import utils
        assert hasattr(utils, 'draw_bbox_on_image')
        assert hasattr(utils, 'calculate_square_bbox_from_patch')
        assert hasattr(utils, 'process_image_with_mask')
        assert hasattr(utils, 'sample_valid_patch_positions')
        assert hasattr(utils, 'resize_and_pad')

    def test_import_prompts(self):
        """Prompts module should import without errors."""
        from reproduce.scripts.evaluate import prompts
        assert hasattr(prompts, 'IMAGE_PROMPT')
        assert hasattr(prompts, 'IMAGE_PROMPT_WITH_CROP')

    def test_prompts_have_placeholders(self):
        """Prompts should have candidate_words placeholder."""
        from reproduce.scripts.evaluate.prompts import IMAGE_PROMPT, IMAGE_PROMPT_WITH_CROP

        assert "{candidate_words}" in IMAGE_PROMPT
        assert "{candidate_words}" in IMAGE_PROMPT_WITH_CROP

    def test_utils_clip_constants(self):
        """Utils should have CLIP normalization constants."""
        from reproduce.scripts.evaluate.utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        assert len(OPENAI_CLIP_MEAN) == 3
        assert len(OPENAI_CLIP_STD) == 3
        assert all(0 < v < 1 for v in OPENAI_CLIP_MEAN)
        assert all(0 < v < 1 for v in OPENAI_CLIP_STD)


class TestEvaluateUtilsFunctions:
    """Test utility functions work correctly."""

    def test_calculate_bbox(self):
        """calculate_square_bbox_from_patch should return correct coordinates."""
        from reproduce.scripts.evaluate.utils import calculate_square_bbox_from_patch

        # 3x3 bbox starting at (0, 0) with patch_size=24
        bbox = calculate_square_bbox_from_patch(0, 0, patch_size=24, size=3)
        assert bbox == (0, 0, 72, 72)

        # 3x3 bbox starting at (1, 2)
        bbox = calculate_square_bbox_from_patch(1, 2, patch_size=24, size=3)
        assert bbox == (48, 24, 120, 96)  # (col*24, row*24, (col+3)*24, (row+3)*24)

    def test_clip_bbox_to_image(self):
        """clip_bbox_to_image should clip coordinates to image boundaries."""
        from reproduce.scripts.evaluate.utils import clip_bbox_to_image

        # Within bounds
        bbox = clip_bbox_to_image((10, 20, 100, 200), 512, 512)
        assert bbox == (10, 20, 100, 200)

        # Exceeds right/bottom
        bbox = clip_bbox_to_image((10, 20, 600, 600), 512, 512)
        assert bbox == (10, 20, 512, 512)

        # Negative values
        bbox = clip_bbox_to_image((-10, -20, 100, 200), 512, 512)
        assert bbox == (0, 0, 100, 200)

    def test_load_image(self):
        """load_image should return numpy array."""
        import numpy as np
        from reproduce.scripts.evaluate.utils import load_image
        from PIL import Image
        import tempfile
        import os

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name)
            temp_path = f.name

        try:
            loaded = load_image(temp_path)
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape == (100, 100, 3)
            assert loaded.dtype == np.uint8
        finally:
            os.unlink(temp_path)
