"""
Diffusion Language Models - Diffusion Language Models Are Natively Length-Aware

A research package for efficient text generation using diffusion language models
with dynamic length prediction optimization (SmartCrop).

Authors:
- Davide Beltrame
- Giacomo Cirò  
- Luca Gandolfi
- Vittorio Rossi

Bocconi University, Milan, Italy
"""

__version__ = "0.1.0"
__author__ = "Davide Beltrame, Giacomo Cirò, Luca Gandolfi, Vittorio Rossi"
__email__ = "vittorio.rossi2@studbocconi.it"

# Core imports for easy access
import dllm
from dllm.core.generation.generator import GeneratorOutput, GeneratorConfig, BaseGenerator
from diffusion_llms.generators.zero_shot_generator import ZeroShotGenerator, ZeroShotGeneratorConfig
from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler

__all__ = [
    "ZeroShotGenerator",
    "ZeroShotGeneratorConfig", 
    "GeneratorOutput",
    "GeneratorConfig",
    "BaseGenerator",
    "BaseAlphaScheduler",
    "LinearAlphaScheduler",
]