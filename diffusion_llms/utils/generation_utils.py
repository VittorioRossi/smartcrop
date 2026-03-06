"""
Generation utilities for diffusion language models.

Originally from DLLM (https://github.com/ML-GSAI/dllm)
Adapted for diffusion_llms project.
"""

from typing import Sequence, Union, cast
import torch

from dllm.core.schedulers import BaseAlphaScheduler

# Type aliases for better typing support
TensorOrList = Union[torch.Tensor, list]
PromptInput = Sequence[TensorOrList]


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


def compute_zero_shot_length_prediction(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    mask_id: int,
    eos_id: int,
    max_new_tokens: int,
    eos_quantile: float,
    safe_margin: int,
    device: torch.device | None = None,
    logits: torch.Tensor | None = None,
) -> int | None:
    """Perform a single forward pass over prompt + mask tail and derive a predicted
    total length (prompt + generated) using EOS distribution over masked tokens.

    Returns None if eos_quantile <= 0 or if no cutoff found (fallback to caller's
    default length). Assumes batch size 1.
    """
    if eos_quantile <= 0.0:
        return None

    device = device or prompt.device
    prompt_len = prompt.shape[0]
    total_alloc = prompt_len + max_new_tokens

    # Build temporary canvas
    x = torch.full((1, total_alloc), eos_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = prompt
    x[0, prompt_len:] = mask_id
    attention_mask = (x != eos_id).bool()

    with torch.no_grad():
        if logits is None:
            logits = model(x, attention_mask=attention_mask).logits  # [1, T, V]
        # 1. Compute probabilities over vocabulary at each step
        # logits: [1, T, V] -> step_probs: [T, V]
        step_probs = torch.softmax(logits[0, prompt_len:], dim=-1)
        
        # 2. Extract EOS probability at each step
        # eos_probs: [max_new_tokens]
        eos_probs = step_probs[:, eos_id]
        
        # 3. Compute cumulative probability of stopping by step t
        # P(stop <= t) = 1 - P(stop > t)
        # P(stop > t) = Product_{i=0}^t (1 - p_i)
        not_stopped_prob = torch.cumprod(1.0 - eos_probs, dim=0) 
        stopped_prob = 1.0 - not_stopped_prob
        
        # 4. Find first step where cumulative stopping probability >= quantile
        cutoff_idx = torch.where(stopped_prob >= eos_quantile)[0]
        if cutoff_idx.numel() == 0:
            return None

        cutoff_tokens = int(cutoff_idx[0].item()) + safe_margin
        cutoff_tokens = min(cutoff_tokens, max_new_tokens)
        predicted_total_length = prompt_len + cutoff_tokens
        return predicted_total_length