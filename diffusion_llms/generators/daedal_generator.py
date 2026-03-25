"""DAEDAL generator: Dynamic Adaptive length Expansion for Diffusion ALgorithm.

Ports the DAEDAL inference strategy (Li et al., 2025, arXiv 2508.00819) into the
diffusion_llms generator interface for direct comparison with SmartCrop.

DAEDAL operates in two phases:
  1. Initial Length Adjustment: starts from a short canvas and expands it by
     checking EOS confidence until the model signals the budget is sufficient.
  2. Iterative Denoising with Mask Insertion: runs a block-by-block denoising
     loop; whenever a masked position has very low confidence it is replaced by
     `expansion_factor` new mask tokens, giving the model "breathing room".

Unlike SmartCrop (which crops once before denoising and runs a fixed-T schedule),
DAEDAL dynamically grows the canvas during generation with no fixed step budget.

Reference hyperparameters (from DAEDAL's official eval scripts):
    initial_gen_length=64, expansion_factor=8, block_length=32
    high_conf_threshold=0.90, low_conf_threshold=0.10
    eos_confidence_threshold=0.5, expand_eos_confidence_threshold=0.9
    eos_check_tokens=32
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from dllm.core.generation.generator import BaseGenerator, GeneratorConfig

from diffusion_llms.generators.base_length_generator import BaseGeneratorOutput
from diffusion_llms.utils.common import add_gumbel_noise, ensure_token_id
from diffusion_llms.utils.generation_utils import (
    PromptInput,
    convert_prompts_to_tensors,
    estimate_forward_flops,
)


@dataclass
class DaedalGeneratorConfig(GeneratorConfig):
    max_new_tokens: int = 512
    initial_gen_length: int = 64
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    high_conf_threshold: float = 0.90
    low_conf_threshold: float = 0.10
    eos_confidence_threshold: float = 0.5
    expand_eos_confidence_threshold: float = 0.9
    expansion_factor: int = 8
    eos_check_tokens: int = 32
    measure_flops: bool = False


@dataclass
class DaedalGeneratorOutput(BaseGeneratorOutput):
    # Number of actual model forward passes (both Stage 1 + Stage 2).
    # Used in eval.py instead of the scheduler-derived NFE used for SmartCrop/FC.
    nfe: int = 0


@dataclass
class DaedalGenerator(BaseGenerator):

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError("DAEDAL does not support infilling")

    @torch.no_grad()
    def generate(
        self,
        prompts: PromptInput,
        config: Optional[GeneratorConfig] = None,
        **kwargs,
    ) -> DaedalGeneratorOutput:
        if not isinstance(config, DaedalGeneratorConfig):
            config = DaedalGeneratorConfig()

        max_new_tokens           = kwargs.get("max_new_tokens",                config.max_new_tokens)
        initial_gen_length       = kwargs.get("initial_gen_length",            config.initial_gen_length)
        block_length             = kwargs.get("block_length",                  config.block_length)
        temperature              = kwargs.get("temperature",                   config.temperature)
        cfg_scale                = kwargs.get("cfg_scale",                     config.cfg_scale)
        high_conf_threshold      = kwargs.get("high_conf_threshold",           config.high_conf_threshold)
        low_conf_threshold       = kwargs.get("low_conf_threshold",            config.low_conf_threshold)
        eos_conf_threshold       = kwargs.get("eos_confidence_threshold",      config.eos_confidence_threshold)
        expand_eos_conf_threshold= kwargs.get("expand_eos_confidence_threshold", config.expand_eos_confidence_threshold)
        expansion_factor         = kwargs.get("expansion_factor",              config.expansion_factor)
        eos_check_tokens         = kwargs.get("eos_check_tokens",              config.eos_check_tokens)
        measure_flops            = kwargs.get("measure_flops",                 config.measure_flops)

        device = self.model.device
        inputs = convert_prompts_to_tensors(prompts, device)

        assert len(inputs) == 1, "DaedalGenerator supports batch_size=1 only"

        prompt = inputs[0].unsqueeze(0)  # (1, L_p)
        prompt_length = prompt.shape[1]

        mask_id = ensure_token_id(
            getattr(self.tokenizer, "mask_token_id", None),
            "mask_token_id",
            tokenizer=self.tokenizer,
            candidates=["<mask>", "[MASK]", "<|mask|>", "<|mdm_mask|>", "[gMASK]"],
        )
        eos_id = ensure_token_id(
            getattr(self.tokenizer, "eos_token_id", None),
            "eos_token_id",
            tokenizer=self.tokenizer,
        )

        total_flops = 0.0
        nfe = 0

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _forward(x_in: torch.Tensor) -> torch.Tensor:
            """One model forward pass with length-based attention mask."""
            nonlocal total_flops, nfe
            seq_len = x_in.shape[1]
            # Length-based mask: attend to all real positions (not eos padding)
            attn_mask = (x_in != eos_id).long()
            if cfg_scale > 0.0:
                un_x = x_in.clone()
                un_x[:, :prompt_length] = mask_id
                x_ = torch.cat([x_in, un_x], dim=0)
                attn_ = torch.cat([attn_mask, attn_mask], dim=0)
                logits_both = self.model(x_, attention_mask=attn_).logits
                logits_cond, logits_un = torch.chunk(logits_both, 2, dim=0)
                logits = logits_un + (cfg_scale + 1) * (logits_cond - logits_un)
            else:
                logits = self.model(x_in, attention_mask=attn_mask).logits
            if measure_flops:
                step_flops = estimate_forward_flops(
                    self.model, seq_len=seq_len, batch_size=1,
                    cfg_active=(cfg_scale > 0.0),
                )
                if step_flops is not None:
                    total_flops += step_flops
            nfe += 1
            return logits

        def _eos_confidence(logits: torch.Tensor, gen_length: int) -> float:
            """Average EOS prob over the last eos_check_tokens positions predicting EOS."""
            total_len = prompt_length + gen_length
            confidences = F.softmax(logits[0, :total_len], dim=-1)
            predicted_tokens = torch.argmax(logits[0, :total_len], dim=-1)
            eos_confs = []
            for pos in range(total_len - 1, prompt_length - 1, -1):
                if len(eos_confs) >= eos_check_tokens:
                    break
                if predicted_tokens[pos].item() == eos_id:
                    eos_confs.append(confidences[pos, eos_id].item())
            if not eos_confs:
                return 0.0
            return sum(eos_confs) / eos_check_tokens

        # ------------------------------------------------------------------
        # Stage 1: Initial Length Adjustment
        # Expand the canvas until the model signals sufficient length via EOS
        # confidence, or until max_new_tokens is reached.
        # ------------------------------------------------------------------
        gen_length = min(initial_gen_length, max_new_tokens)
        x = torch.full(
            (1, prompt_length + gen_length), mask_id, dtype=torch.long, device=device
        )
        x[0, :prompt_length] = prompt[0]

        while True:
            logits = _forward(x)
            eos_conf = _eos_confidence(logits, gen_length)
            if eos_conf >= eos_conf_threshold or gen_length >= max_new_tokens:
                break
            new_gen_length = min(gen_length + expansion_factor, max_new_tokens)
            new_x = torch.full(
                (1, prompt_length + new_gen_length), eos_id, dtype=torch.long, device=device
            )
            new_x[0, :prompt_length + gen_length] = x[0, :prompt_length + gen_length]
            new_x[0, prompt_length + gen_length : prompt_length + new_gen_length] = mask_id
            x = new_x
            gen_length = new_gen_length

        # Add eos_check_tokens/2 buffer tokens (same as DAEDAL reference impl)
        buffer = min(eos_check_tokens // 2, max_new_tokens - gen_length)
        if buffer > 0:
            new_gen_length = gen_length + buffer
            new_x = torch.full(
                (1, prompt_length + new_gen_length), eos_id, dtype=torch.long, device=device
            )
            new_x[0, :prompt_length] = prompt[0]
            new_x[0, prompt_length : prompt_length + gen_length] = mask_id
            x = new_x
            gen_length = new_gen_length

        # ------------------------------------------------------------------
        # Stage 2: Iterative Denoising with Mask Insertion
        # Block-by-block loop: fill high-confidence tokens, expand low-confidence
        # masked positions by inserting expansion_factor new mask tokens.
        # ------------------------------------------------------------------
        current_pos = prompt_length
        denoise_only = False

        while current_pos < prompt_length + gen_length:
            if gen_length >= max_new_tokens and not denoise_only:
                denoise_only = True

            total_len = prompt_length + gen_length
            logits = _forward(x)
            eos_conf = _eos_confidence(logits, gen_length)

            predicted_tokens = torch.argmax(
                add_gumbel_noise(logits[0, :total_len], temperature=temperature), dim=-1
            )  # (total_len,)
            confidences = F.softmax(logits[0, :total_len], dim=-1)
            predicted_confs = confidences[
                torch.arange(total_len, device=device), predicted_tokens
            ]  # (total_len,)

            # Block window for this step
            block_end = min(current_pos + block_length, total_len)
            currently_masked = (x[0, :total_len] == mask_id)  # (total_len,)
            in_block = torch.zeros(total_len, dtype=torch.bool, device=device)
            in_block[current_pos:block_end] = True

            # --- Identify high-confidence fills ---
            high_conf = (
                (predicted_confs > high_conf_threshold)
                & in_block
                & currently_masked
                & (predicted_tokens != mask_id)
            )

            # Fallback: if nothing exceeds threshold, pick the best candidate
            if not high_conf[current_pos:block_end].any():
                candidates = torch.where(in_block & currently_masked)[0]
                if len(candidates) > 0:
                    cand_confs = predicted_confs[candidates]
                    cand_tokens = predicted_tokens[candidates]
                    sorted_idx = torch.argsort(cand_confs, descending=True)
                    filled = False
                    for sidx in sorted_idx:
                        if cand_tokens[sidx].item() != mask_id:
                            high_conf[candidates[sidx]] = True
                            filled = True
                            break
                    if not filled:
                        # All candidates predict mask; force best non-mask token
                        stuck_logits = logits[0, candidates].clone()
                        stuck_logits[:, mask_id] = -torch.inf
                        new_confs = F.softmax(stuck_logits, dim=-1)
                        new_best_confs, new_best_tokens = torch.max(new_confs, dim=-1)
                        best_local = torch.argmax(new_best_confs)
                        pos_to_fill = candidates[best_local]
                        predicted_tokens[pos_to_fill] = new_best_tokens[best_local]
                        high_conf[pos_to_fill] = True

            # Apply fills
            fill_positions = torch.where(high_conf)[0]
            x[0, fill_positions] = predicted_tokens[fill_positions]

            # --- Expansion: replace lowest-conf masked position with expansion_factor masks ---
            expanded = False
            if (
                not denoise_only
                and eos_conf < expand_eos_conf_threshold
                and gen_length < max_new_tokens
            ):
                expand_candidates = (
                    (predicted_confs < low_conf_threshold)
                    & in_block
                    & currently_masked
                    & ~high_conf
                )
                exp_positions = torch.where(expand_candidates)[0]
                if len(exp_positions) > 0:
                    lowest_idx = torch.argmin(predicted_confs[exp_positions])
                    expand_pos = exp_positions[lowest_idx].item()

                    # Net new tokens: expansion_factor - 1 (one mask → expansion_factor masks)
                    new_gen_length = min(gen_length + expansion_factor - 1, max_new_tokens)
                    new_total = prompt_length + new_gen_length
                    new_x = torch.full(
                        (1, new_total), eos_id, dtype=torch.long, device=device
                    )
                    # Copy prompt
                    new_x[0, :prompt_length] = x[0, :prompt_length]
                    # Copy gen tokens up to expand_pos, then insert expansion_factor masks
                    write_ptr = prompt_length
                    for j in range(prompt_length, total_len):
                        if write_ptr >= new_total:
                            break
                        if j == expand_pos:
                            end_write = min(write_ptr + expansion_factor, new_total)
                            new_x[0, write_ptr:end_write] = mask_id
                            write_ptr = end_write
                        else:
                            new_x[0, write_ptr] = x[0, j]
                            write_ptr += 1
                    x = new_x
                    gen_length = new_gen_length
                    expanded = True

            # --- Advance current_pos past fully-filled blocks ---
            new_total_len = prompt_length + gen_length
            while current_pos < new_total_len:
                block_s = current_pos
                block_e = min(block_s + block_length, new_total_len)
                if block_s >= block_e:
                    break
                if not (x[0, block_s:block_e] == mask_id).any():
                    current_pos = block_s + block_length
                else:
                    break

            # Stagnation guard: if nothing changed, force-advance to avoid infinite loop
            if len(fill_positions) == 0 and not expanded:
                current_pos = min(current_pos + block_length, prompt_length + gen_length)

        # Return full sequence as (1, L) tensor for compatibility with eval harness
        final_len = prompt_length + gen_length
        sequences = x[0, :final_len].unsqueeze(0)  # (1, final_len)

        return DaedalGeneratorOutput(
            sequences=sequences,
            total_flops=total_flops if measure_flops and total_flops > 0 else None,
            per_forward_flops=None,  # Varies per step in DAEDAL; not meaningful
            predicted_length=gen_length,
            nfe=nfe,
        )
