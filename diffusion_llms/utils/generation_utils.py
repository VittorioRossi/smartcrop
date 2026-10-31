"""
Generation utilities for diffusion language models.

Originally from DLLM (https://github.com/ML-GSAI/dllm)
Adapted for diffusion_llms project.
"""

import dataclasses
from typing import Any, Sequence, Type, TypeVar, Union, cast
import torch

from dllm.core.schedulers import BaseAlphaScheduler

# Type aliases for better typing support
TensorOrList = Union[torch.Tensor, list]
PromptInput = Sequence[TensorOrList]

_T = TypeVar("_T")


def upgrade_config(source: Any, target_cls: Type[_T]) -> _T:
    """Instantiate target_cls and copy any matching fields from source."""
    upgraded = target_cls()
    for f in dataclasses.fields(upgraded):  # type: ignore[arg-type]
        if hasattr(source, f.name):
            setattr(upgraded, f.name, getattr(source, f.name))
    return upgraded


def convert_prompts_to_tensors(prompts: PromptInput, device: torch.device) -> list[torch.Tensor]:
    """
    Convert prompts to tensors and move them to the specified device.
    
    Args:
        prompts: Sequence of prompts (can be tensors or lists of ints)
        device: Target device for the tensors
    
    Returns:
        List of tensors on the specified device
    """
    tensor_prompts: list[torch.Tensor] = []
    for p in prompts:
        if isinstance(p, torch.Tensor):
            tensor_prompts.append(p.to(device))
        else:  # assume sequence of ints
            tensor_prompts.append(
                torch.as_tensor(p, dtype=torch.long, device=device)
            )
    return tensor_prompts


def estimate_forward_flops(
    model: torch.nn.Module,
    seq_len: int,
    batch_size: int,
    cfg_active: bool = False,
) -> float | None:
    """Approximate dense Transformer forward FLOPs.

    Uses common HuggingFace config attribute names; returns None if essential
    fields aren't found.

    Formula (per layer, per forward pass):
      Projections (Q,K,V,O): 4 * B * S * H * H
      Attention score matmul (QK^T)    : B * A * S * S * (H/A)
      Attention value matmul (scores V): B * A * S * S * (H/A)
      Softmax + scaling (approx)       : 2 * B * A * S * S
      MLP (two matmuls)                : 4 * B * S * H * I   (I≈4H)

    Total per layer = sum above; multiply by number of layers.
    If CFG active, effective batch doubles because we concatenate conditional
    and unconditional branches.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None

    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    num_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    intermediate_size = getattr(cfg, "intermediate_size", None)
    if intermediate_size is None and hidden_size is not None:
        intermediate_size = 4 * hidden_size  # standard transformer default

    if None in (hidden_size, num_layers, num_heads, intermediate_size):
        return None

    # Cast to int to satisfy type checker (validated non-None above)
    hidden_size = cast(int, hidden_size)
    num_layers = cast(int, num_layers)
    num_heads = cast(int, num_heads)
    intermediate_size = cast(int, intermediate_size)

    B_eff = batch_size * (2 if cfg_active else 1)
    H = hidden_size
    intermediate = intermediate_size
    L = num_layers
    A = num_heads
    S = seq_len

    # Projections (Q,K,V plus output)
    proj_flops = 4 * B_eff * S * H * H
    # Attention score & value matmuls
    attn_score = B_eff * A * S * S * (H // A)
    attn_value = B_eff * A * S * S * (H // A)
    # Softmax + scaling (rough cost)
    attn_softmax = 2 * B_eff * A * S * S
    # MLP (two matmuls, each B*S*H*intermediate & B*S*intermediate*H ≈ 2*B*S*H*intermediate, doubled => 4*B*S*H*intermediate)
    mlp_flops = 4 * B_eff * S * H * intermediate

    per_layer = proj_flops + attn_score + attn_value + attn_softmax + mlp_flops
    total_flops = per_layer * L
    return float(total_flops)
