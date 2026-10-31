import torch
from typing import Any


def ensure_token_id(raw, name: str, tokenizer, candidates: list[str] | None = None) -> int:
    if isinstance(raw, int):
        return raw

    # Try to resolve from string candidates if provided
    if raw is None and candidates:
        conv = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(conv):
            for cand in candidates:
                cid = conv(cand)
                if isinstance(cid, int):
                    return cid

    if isinstance(raw, str):
        conv = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(conv):
            cid = conv(raw)
            if isinstance(cid, int):
                return cid

    raise TypeError(f"{name} must resolve to an int token id; got {raw!r} ({type(raw)})")


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    We use float32 for MPS compatibility (Apple Silicon doesn't support float64).

    Reference: Adapted from DLLM pipelines/llada/generator.py
    """
    if temperature == 0:
        return logits

    # Only force float32 for MPS (Apple Silicon) which doesn't support float64/some ops in half precision well
    if logits.device.type == "mps":
        logits = logits.to(torch.float32)

    if logits.dtype not in [torch.float32, torch.float64] and logits.device.type != "mps":
        # If bfloat16/float16, we compute noise in higher precision then cast back to match logits
        noise = torch.rand_like(logits, dtype=torch.float32)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise.to(logits.dtype)

    # Fallback / MPS path
    logits_f32 = logits.to(torch.float32)
    noise = torch.rand_like(logits_f32, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return (logits_f32.exp() / gumbel_noise).to(logits.dtype)
