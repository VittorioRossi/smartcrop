"""CDF-remasking sampler for masked diffusion LMs.

Single t=1 forward pass on the fully-masked canvas. Computes raw p_eos at each
position and finds the first position where p_eos >= cdf_quantile (default 0.90).
Crops the canvas to that length, then runs standard MDLMSampler denoising on the
shorter sequence.

Optionally suppresses EOS from the vocabulary during denoising (eos_free=True).

With threshold set, the cropped canvas is denoised with confidence-threshold
parallel decoding (DAEDAL's schedule) instead of one token per step.

With no_crop=True, the t=1 probe is skipped and the full max_new_tokens canvas is
denoised: the fixed-length reference for the same decoding schedule.
"""

from copy import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise


@dataclass
class CDFRemaskingSamplerConfig(MDLMSamplerConfig):
    cdf_quantile: float = 0.90   # first position where p_eos >= this is the predicted length
    eos_free: bool = False        # suppress EOS from vocab during denoising
    threshold: float | None = None  # None: one token per step; else commit all conf > threshold per step
    no_crop: bool = False         # skip the t=1 probe and denoise the full max_new_tokens canvas


@torch.no_grad()
def _predict_lengths_from_t1(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    mask_id: int,
    eos_id: int,
    max_new_tokens: int,
    quantile: float = 0.90,
    t1_batch_size: int = 8,
    right_shift_logits: bool = False,
    return_probs: bool = False,
) -> list[int] | tuple[list[int], list[np.ndarray]]:
    """Chunked t=1 forward passes → raw p_eos → first crossing of quantile threshold.

    Processes inputs in mini-batches of t1_batch_size to avoid OOM when N is large.
    Set right_shift_logits=True for AR-shift models like DREAM.

    Args:
        return_probs: if True, also return full softmax distributions over vocab for
            each sample, shape [max_new_tokens, vocab_size] per sample (float16 numpy).

    Returns:
        pred_lens if return_probs=False, else (pred_lens, prob_arrays).
    """
    pred_lens = []
    prob_arrays = [] if return_probs else None

    for start in range(0, len(inputs), t1_batch_size):
        chunk = inputs[start:start + t1_batch_size]
        prompt_lens = [p.shape[0] for p in chunk]
        T = max(prompt_lens) + max_new_tokens
        Bc = len(chunk)

        x = torch.full((Bc, T), eos_id, dtype=torch.long, device=model.device)
        attn = torch.zeros((Bc, T), dtype=torch.long, device=model.device)
        for i, p in enumerate(chunk):
            pl = prompt_lens[i]
            x[i, :pl] = p
            x[i, pl:pl + max_new_tokens] = mask_id
            attn[i, :pl + max_new_tokens] = 1

        logits = model(x, attention_mask=attn).logits  # [Bc, T, V]
        if right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        p_all = F.softmax(logits, dim=-1)

        for b in range(Bc):
            pl = prompt_lens[b]
            gen_eos = p_all[b, pl:pl + max_new_tokens, eos_id].float().cpu().numpy()
            crossings = np.where(gen_eos >= quantile)[0]
            pred_len = int(crossings[0]) + 1 if len(crossings) > 0 else max_new_tokens
            pred_lens.append(max(1, pred_len))

            if return_probs:
                gen_probs = p_all[b, pl:pl + max_new_tokens, :]  # [L, V]
                p_eos_col = gen_probs[:, eos_id]                  # [L]
                gen_probs_no_eos = gen_probs.clone()
                gen_probs_no_eos[:, eos_id] = 0.0
                p_top_non_eos = gen_probs_no_eos.max(dim=-1).values  # [L]
                arr = torch.stack([p_eos_col, p_top_non_eos], dim=-1)  # [L, 2]
                prob_arrays.append(arr.to(torch.float16).cpu().numpy())

    if return_probs:
        return pred_lens, prob_arrays
    return pred_lens


@torch.no_grad()
def _threshold_denoise(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    mask_id: int,
    eos_id: int,
    max_new_tokens: int,
    block_size: int,
    threshold: float,
    temperature: float = 0.0,
    suppress_tokens: list[int] | None = None,
    right_shift_logits: bool = False,
) -> torch.Tensor:
    """Confidence-threshold parallel decoding over a fixed canvas.

    Mirrors the Stage-2 decoding of DAEDAL (rho-EOS models/LLaDA_DAEDAL.py), minus
    its mask insertion. Blocks of block_size are decoded left to right. Each step
    commits every masked position in the current block whose prediction is not the
    mask token and whose confidence is > threshold. If none qualifies, the single
    most confident non-mask prediction is committed. A sample moves to its next
    block once the current one is fully unmasked.
    """
    prompt_lens = [p.shape[0] for p in inputs]
    B = len(inputs)
    T = max(prompt_lens) + max_new_tokens

    x = torch.full((B, T), eos_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros((B, T), dtype=torch.long, device=model.device)
    for j, p in enumerate(inputs):
        pl = prompt_lens[j]
        x[j, :pl] = p
        x[j, pl:pl + max_new_tokens] = mask_id
        attention_mask[j, :pl + max_new_tokens] = 1

    gen_end = [pl + max_new_tokens for pl in prompt_lens]
    block_start = list(prompt_lens)

    while any(block_start[j] < gen_end[j] for j in range(B)):
        logits = model(x, attention_mask=attention_mask).logits
        if suppress_tokens:
            for token_id in suppress_tokens:
                logits[:, :, token_id] = -torch.inf
        if right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        for j in range(B):
            s = block_start[j]
            if s >= gen_end[j]:
                continue
            e = min(s + block_size, gen_end[j])
            blk_logits = logits[j, s:e]
            pred = torch.argmax(add_gumbel_noise(blk_logits, temperature=temperature), dim=-1)
            conf = torch.gather(F.softmax(blk_logits, dim=-1), -1, pred.unsqueeze(-1)).squeeze(-1)

            masked = x[j, s:e] == mask_id
            candidate = masked & (pred != mask_id)
            commit = candidate & (conf > threshold)
            if not commit.any():
                if candidate.any():
                    commit[torch.where(candidate, conf, -torch.inf).argmax()] = True
                else:
                    # Every masked position predicts the mask token: re-pick without it.
                    alt = blk_logits.clone()
                    alt[:, mask_id] = -torch.inf
                    alt_conf, alt_pred = F.softmax(alt, dim=-1).max(dim=-1)
                    pos = torch.where(masked, alt_conf, -torch.inf).argmax()
                    pred[pos] = alt_pred[pos]
                    commit[pos] = True
            x[j, s:e][commit] = pred[commit]

            while block_start[j] < gen_end[j] and not (
                x[j, block_start[j]:min(block_start[j] + block_size, gen_end[j])] == mask_id
            ).any():
                block_start[j] += block_size

    return x


@dataclass
class CDFRemaskingSampler(MDLMSampler):
    """MDLMSampler with single t=1 canvas crop based on raw p_eos threshold."""

    @torch.no_grad()
    def sample(
        self,
        inputs,
        config=None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = CDFRemaskingSamplerConfig()

        cdf_quantile      = kwargs.get("cdf_quantile",      config.cdf_quantile)
        eos_free          = kwargs.get("eos_free",          config.eos_free)
        max_new_tokens    = kwargs.get("max_new_tokens",    config.max_new_tokens)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        threshold         = kwargs.get("threshold",         config.threshold)
        no_crop           = kwargs.get("no_crop",           config.no_crop)

        mask_id = self.tokenizer.mask_token_id
        eos_id  = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        if no_crop:
            cropped = max_new_tokens
            print(f"[CDF crop] no_crop: full canvas {cropped}  eos_free={eos_free}")
        else:
            pred_lens = _predict_lengths_from_t1(
                self.model, inputs, mask_id, eos_id, max_new_tokens,
                quantile=cdf_quantile, right_shift_logits=right_shift_logits,
            )
            cropped = max(pred_lens)
            print(f"[CDF crop] predicted lengths: {pred_lens}  → canvas cropped to {cropped}  eos_free={eos_free}")

        cropped_config = copy(config)
        cropped_config.remasking = "low_confidence"
        cropped_config.max_new_tokens = cropped
        cropped_config.steps = cropped
        if eos_free:
            existing = list(cropped_config.suppress_tokens or [])
            if eos_id not in existing:
                existing.append(eos_id)
            cropped_config.suppress_tokens = existing

        if threshold is None:
            return super().sample(inputs, cropped_config, **kwargs)

        assert kwargs.get("cfg_scale", config.cfg_scale) == 0.0, "threshold decoding has no CFG"
        x = _threshold_denoise(
            self.model, inputs, mask_id, eos_id, cropped,
            block_size=kwargs.get("block_size", config.block_size),
            threshold=threshold,
            temperature=kwargs.get("temperature", config.temperature),
            suppress_tokens=kwargs.get("suppress_tokens", cropped_config.suppress_tokens),
            right_shift_logits=right_shift_logits,
        )
        if not kwargs.get("return_dict", config.return_dict):
            return x
        return BaseSamplerOutput(sequences=x, histories=None)
