"""Efficiency metrics for diffusion Language Models (dLLMs)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math


@dataclass
class dLLMEfficiencyMetrics:
    """Container for dLLM efficiency metrics."""

    nfe: int = 0
    total_steps: int = 0
    effective_steps: int = 0
    blocks_processed: int = 0

    total_flops: Optional[float] = None
    per_step_flops: Optional[float] = None
    wall_time: Optional[float] = None

    tokens_generated: int = 0
    prompt_length: int = 0
    sequence_length: int = 0

    flops_per_token: Optional[float] = field(init=False, default=None)
    nfe_per_token: Optional[float] = field(init=False, default=None)
    tokens_per_second: Optional[float] = field(init=False, default=None)

    model_name: Optional[str] = None
    generation_mode: Optional[str] = None
    block_length: int = 0
    cfg_scale: float = 0.0
    temperature: float = 0.0
    remasking_strategy: str = ""

    def __post_init__(self):
        if self.tokens_generated > 0:
            if self.total_flops is not None:
                self.flops_per_token = self.total_flops / self.tokens_generated
            if self.nfe > 0:
                self.nfe_per_token = self.nfe / self.tokens_generated
        if self.wall_time is not None and self.wall_time > 0:
            self.tokens_per_second = self.tokens_generated / self.wall_time


def compute_nfe_efficiency_from_generator_output(
    generator_output,
    prompt_lengths: List[int],
    config: Dict[str, Any],
    model_name: Optional[str] = None,
    generation_mode: Optional[str] = None
) -> dLLMEfficiencyMetrics:
    """Compute NFE and efficiency metrics from a GeneratorOutput."""
    sequences = generator_output.sequences
    batch_size, seq_len = sequences.shape
    total_flops = getattr(generator_output, 'total_flops', None)
    per_forward_flops = getattr(generator_output, 'per_forward_flops', None)

    max_new_tokens = config.get("max_new_tokens", 128)
    block_length = config.get("block_length", 128)
    steps = config.get("steps", 128)
    cfg_scale = config.get("cfg_scale", 0.0)

    num_blocks = math.ceil(max_new_tokens / block_length)
    steps_per_block = math.ceil(steps / num_blocks)
    total_nfe = num_blocks * steps_per_block
    if cfg_scale > 0.0:
        total_nfe *= 2

    total_prompt_length = sum(prompt_lengths)
    tokens_generated = max(0, batch_size * seq_len - total_prompt_length)

    return dLLMEfficiencyMetrics(
        nfe=total_nfe,
        total_steps=total_nfe,
        effective_steps=total_nfe,
        blocks_processed=num_blocks,
        total_flops=total_flops,
        per_step_flops=per_forward_flops,
        tokens_generated=tokens_generated,
        prompt_length=total_prompt_length,
        sequence_length=batch_size * seq_len,
        model_name=model_name,
        generation_mode=generation_mode,
        block_length=block_length,
        cfg_scale=cfg_scale,
        temperature=config.get("temperature", 0.0),
        remasking_strategy=config.get("remasking", ""),
    )
