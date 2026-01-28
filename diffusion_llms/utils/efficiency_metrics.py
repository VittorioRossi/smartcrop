"""
Efficiency metrics for diffusion Language Models (dLLMs).

This module provides comprehensive metrics for evaluating dLLM efficiency,
with a focus on Number of Function Evaluations (NFE) and computational cost analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import torch
import math
import time


@dataclass
class dLLMEfficiencyMetrics:
    """Container for dLLM efficiency metrics."""
    
    # Core NFE metrics
    nfe: int = 0  # Number of Function Evaluations
    total_steps: int = 0  # Total denoising/unmasking steps
    effective_steps: int = 0  # Steps that actually transferred tokens
    blocks_processed: int = 0  # Number of generation blocks
    
    # Computational metrics
    total_flops: Optional[float] = None  # Total FLOPs consumed
    per_step_flops: Optional[float] = None  # Average FLOPs per step
    wall_time: Optional[float] = None  # Wall clock time in seconds
    
    # Generation metrics
    tokens_generated: int = 0  # Number of new tokens generated
    prompt_length: int = 0  # Length of input prompt
    sequence_length: int = 0  # Final sequence length
    
    # Efficiency ratios
    flops_per_token: Optional[float] = field(init=False, default=None)
    nfe_per_token: Optional[float] = field(init=False, default=None)
    tokens_per_second: Optional[float] = field(init=False, default=None)
    
    # Model configuration
    model_name: Optional[str] = None
    generation_mode: Optional[str] = None
    block_length: int = 0
    cfg_scale: float = 0.0
    temperature: float = 0.0
    remasking_strategy: str = ""
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.tokens_generated > 0:
            if self.total_flops is not None:
                self.flops_per_token = self.total_flops / self.tokens_generated
            if self.nfe > 0:
                self.nfe_per_token = self.nfe / self.tokens_generated
        
        if self.wall_time is not None and self.wall_time > 0:
            self.tokens_per_second = self.tokens_generated / self.wall_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nfe": self.nfe,
            "total_steps": self.total_steps,
            "effective_steps": self.effective_steps,
            "blocks_processed": self.blocks_processed,
            "total_flops": self.total_flops,
            "per_step_flops": self.per_step_flops,
            "wall_time": self.wall_time,
            "tokens_generated": self.tokens_generated,
            "prompt_length": self.prompt_length,
            "sequence_length": self.sequence_length,
            "flops_per_token": self.flops_per_token,
            "nfe_per_token": self.nfe_per_token,
            "tokens_per_second": self.tokens_per_second,
            "model_name": self.model_name,
            "generation_mode": self.generation_mode,
            "block_length": self.block_length,
            "cfg_scale": self.cfg_scale,
            "temperature": self.temperature,
            "remasking_strategy": self.remasking_strategy,
        }


class dLLMEfficiencyTracker:
    """Tracks and computes efficiency metrics during dLLM generation."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.nfe_count = 0
        self.total_steps = 0
        self.effective_steps = 0
        self.blocks_processed = 0
        self.total_flops = 0.0
        self.per_step_flops = None
        self.generation_config = {}
        self.prompt_lengths = []
        self.final_lengths = []
    
    def start_generation(self):
        """Mark the start of generation."""
        self.start_time = time.time()
    
    def end_generation(self):
        """Mark the end of generation."""
        self.end_time = time.time()
    
    def record_step(self, 
                   step_flops: Optional[float] = None,
                   tokens_transferred: int = 0,
                   is_effective: bool = True):
        """Record a single generation step."""
        self.nfe_count += 1
        self.total_steps += 1
        
        if is_effective and tokens_transferred > 0:
            self.effective_steps += 1
        
        if step_flops is not None:
            self.total_flops += step_flops
            if self.per_step_flops is None:
                self.per_step_flops = step_flops
            else:
                # Running average
                self.per_step_flops = (self.per_step_flops * (self.nfe_count - 1) + step_flops) / self.nfe_count
    
    def record_block_completion(self):
        """Record completion of a generation block."""
        self.blocks_processed += 1
    
    def set_generation_config(self, config: Dict[str, Any]):
        """Store generation configuration."""
        self.generation_config = config
    
    def record_sequence_info(self, prompt_length: int, final_length: int):
        """Record sequence length information."""
        self.prompt_lengths.append(prompt_length)
        self.final_lengths.append(final_length)
    
    def compute_metrics(self, 
                       generation_mode: Optional[str] = None) -> dLLMEfficiencyMetrics:
        """Compute comprehensive efficiency metrics."""
        
        # Calculate timing
        wall_time = None
        if self.start_time is not None and self.end_time is not None:
            wall_time = self.end_time - self.start_time
        
        # Calculate token metrics
        total_prompt_length = sum(self.prompt_lengths) if self.prompt_lengths else 0
        total_final_length = sum(self.final_lengths) if self.final_lengths else 0
        tokens_generated = total_final_length - total_prompt_length
        
        # Extract config values
        block_length = self.generation_config.get("block_length", 0)
        cfg_scale = self.generation_config.get("cfg_scale", 0.0)
        temperature = self.generation_config.get("temperature", 0.0)
        remasking = self.generation_config.get("remasking", "")
        
        return dLLMEfficiencyMetrics(
            nfe=self.nfe_count,
            total_steps=self.total_steps,
            effective_steps=self.effective_steps,
            blocks_processed=self.blocks_processed,
            total_flops=self.total_flops if self.total_flops > 0 else None,
            per_step_flops=self.per_step_flops,
            wall_time=wall_time,
            tokens_generated=max(0, tokens_generated),
            prompt_length=total_prompt_length,
            sequence_length=total_final_length,
            model_name=self.model_name,
            generation_mode=generation_mode,
            block_length=block_length,
            cfg_scale=cfg_scale,
            temperature=temperature,
            remasking_strategy=remasking,
        )


def compute_nfe_efficiency_from_generator_output(
    generator_output, 
    prompt_lengths: List[int],
    config: Dict[str, Any],
    model_name: Optional[str] = None,
    generation_mode: Optional[str] = None
) -> dLLMEfficiencyMetrics:
    """
    Compute NFE and efficiency metrics from a GeneratorOutput.
    
    Args:
        generator_output: The GeneratorOutput from generate()
        prompt_lengths: List of prompt lengths for each sequence
        config: Generation configuration dictionary
        model_name: Name of the model used
        generation_mode: Generation mode (e.g., "native", "zero_shot", "adapter")
    
    Returns:
        dLLMEfficiencyMetrics object with comprehensive metrics
    """
    
    # Extract basic info
    sequences = generator_output.sequences
    batch_size, seq_len = sequences.shape
    total_flops = getattr(generator_output, 'total_flops', None)
    per_forward_flops = getattr(generator_output, 'per_forward_flops', None)
    
    # Calculate NFE from configuration
    max_new_tokens = config.get("max_new_tokens", 128)
    block_length = config.get("block_length", 128)
    steps = config.get("steps", 128)
    cfg_scale = config.get("cfg_scale", 0.0)
    
    # NFE calculation based on the generation algorithm
    num_blocks = math.ceil(max_new_tokens / block_length)
    steps_per_block = math.ceil(steps / num_blocks)
    total_nfe = num_blocks * steps_per_block
    
    # Adjust for CFG (Classifier-Free Guidance) - doubles forward passes
    if cfg_scale > 0.0:
        total_nfe *= 2
    
    # Calculate token metrics
    total_prompt_length = sum(prompt_lengths)
    total_final_length = batch_size * seq_len
    tokens_generated = total_final_length - total_prompt_length
    
    return dLLMEfficiencyMetrics(
        nfe=total_nfe,
        total_steps=total_nfe,  # In this context, steps == NFE
        effective_steps=total_nfe,  # Assume all steps are effective
        blocks_processed=num_blocks,
        total_flops=total_flops,
        per_step_flops=per_forward_flops,
        wall_time=None,  # Not available from generator output
        tokens_generated=max(0, tokens_generated),
        prompt_length=total_prompt_length,
        sequence_length=total_final_length,
        model_name=model_name,
        generation_mode=generation_mode,
        block_length=block_length,
        cfg_scale=cfg_scale,
        temperature=config.get("temperature", 0.0),
        remasking_strategy=config.get("remasking", ""),
    )


def create_efficiency_callback(tracker: dLLMEfficiencyTracker) -> Callable:
    """
    Create a step callback function for the BaseLengthGenerator.
    
    Args:
        tracker: The efficiency tracker to update
    
    Returns:
        Callback function that can be passed to the generator
    """
    
    def step_callback(step_info: Dict[str, Any]) -> None:
        """Callback function to track efficiency metrics during generation."""
        
        # Extract step information
        block_idx = step_info.get("block", 0)
        step_idx = step_info.get("step", 0)
        num_transfer_tokens = step_info.get("num_transfer_tokens")
        
        # Record the step
        tokens_transferred = 0
        if num_transfer_tokens is not None:
            # num_transfer_tokens is typically a tensor [batch_size, steps]
            if isinstance(num_transfer_tokens, torch.Tensor):
                tokens_transferred = int(num_transfer_tokens[:, step_idx].sum().item())
        
        # Estimate FLOPs for this step (would need per_forward_flops from config)
        step_flops = None  # Could be passed via step_info if available
        
        tracker.record_step(
            step_flops=step_flops,
            tokens_transferred=tokens_transferred,
            is_effective=(tokens_transferred > 0)
        )
        
        # Record block completion (when it's the last step of a block)
        # This is a simplification - actual detection would need more context
        if step_idx == step_info.get("max_step_in_block", float('inf')):
            tracker.record_block_completion()
    
    return step_callback