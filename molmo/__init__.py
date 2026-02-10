"""
Molmo VLM infrastructure for paper reproduction.

Derived from Molmo (https://github.com/allenai/molmo)
Copyright 2024 Allen Institute for AI
Licensed under the Apache License, Version 2.0
"""

__version__ = "0.1.0"

from .config import ModelConfig, VisionBackboneConfig, ActivationType, LayerNormType, TokenizerConfig
from .model import Molmo, OLMoOutput, OLMoGenerateOutput

__all__ = [
    "ModelConfig",
    "VisionBackboneConfig",
    "Molmo",
    "OLMoOutput",
    "OLMoGenerateOutput",
    "__version__",
]
