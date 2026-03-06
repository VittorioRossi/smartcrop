"""Zero-shot length-aware generator built on top of `BaseLengthGenerator`.

This variant only adds an EOS-quantile based length prediction heuristic. It:
  1. Optionally predicts the final sequence length (prompt + generation) if
     `eos_quantile > 0` and no external `length_prediction` is supplied.
  2. Delegates the actual masked diffusion loop to `BaseLengthGenerator` by
     passing the predicted length via the inherited `length_prediction` field.

All other generation mechanics (FLOPs accounting, CFG, remasking strategy,
callback support) are handled by the base generator.
"""

from dataclasses import dataclass
from typing import Sequence
import torch

from diffusion_llms.generators.base_length_generator import (
    BaseLengthGenerator,
    BaseLengthGeneratorConfig,
)
from diffusion_llms.utils.generation_utils import (
    compute_zero_shot_length_prediction,
)
from diffusion_llms.utils.common import ensure_token_id
from dllm.core.generation.generator import (
    GeneratorOutput,
    GeneratorConfig
)

@dataclass
class ZeroShotGeneratorConfig(BaseLengthGeneratorConfig):
    eos_quantile: float = 0.0  # Quantile for EOS probability mass (0 disables prediction)
    safe_margin: int = 32      # Extra tokens after quantile cutoff

    measure_flops: bool = False  # Whether to measure FLOPs during generation


@dataclass
class ZeroShotGenerator(BaseLengthGenerator):
    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[torch.Tensor] | Sequence[Sequence[int]],
        config: GeneratorConfig | None = None,
        **kwargs,
    ) -> GeneratorOutput:
        # Normalize / upgrade config
        if config is None:
            config = ZeroShotGeneratorConfig()
        if not isinstance(config, ZeroShotGeneratorConfig):
            upgraded = ZeroShotGeneratorConfig()
            upgraded.return_dict_in_generate = getattr(
                config, "return_dict_in_generate", False
            )
            upgraded.measure_flops = getattr(config, "measure_flops", False)
            # Copy shared BaseLengthGeneratorConfig fields if they exist
            for attr in [
                "max_new_tokens",
                "max_length",
                "block_length",
                "steps",
                "temperature",
                "remasking",
                "stochastic_transfer",
                "cfg_scale",
                "cfg_keep_tokens",
                "step_callback",
                "length_prediction",
                "logits_eos_inf",
            ]:
                if hasattr(config, attr):
                    setattr(upgraded, attr, getattr(config, attr))
            config = upgraded

        # Convert prompts to tensors (mirrors BaseLengthGenerator logic)
        tensor_prompts: list[torch.Tensor] = []
        for p in prompts:
            if isinstance(p, torch.Tensor):
                tensor_prompts.append(p.to(self.model.device))
            else:
                tensor_prompts.append(
                    torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                )

        # If external length_prediction not provided and eos_quantile enabled, predict
        predicted_length: int | None = None
        if (
            config.length_prediction is None
            and config.eos_quantile > 0.0
            and len(tensor_prompts) == 1
        ):
            prompt_tensor = tensor_prompts[0]
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
            predicted_length = compute_zero_shot_length_prediction(
                model=self.model,
                prompt=prompt_tensor,
                mask_id=mask_id,
                eos_id=eos_id,
                max_new_tokens=config.max_new_tokens,
                eos_quantile=config.eos_quantile,
                safe_margin=config.safe_margin,
                device=self.model.device,
            )

        # Build a base config clone and set length_prediction if we inferred one
        base_cfg = BaseLengthGeneratorConfig()
        for attr in [
            "max_new_tokens",
            "max_length",
            "block_length",
            "steps",
            "temperature",
            "remasking",
            "stochastic_transfer",
            "cfg_scale",
            "cfg_keep_tokens",
            "return_dict_in_generate",
            "measure_flops",
            "step_callback",
            "logits_eos_inf",
        ]:
            setattr(base_cfg, attr, getattr(config, attr))
        base_cfg.length_prediction = config.length_prediction or predicted_length

        return super().generate(prompts=tensor_prompts, config=base_cfg, **kwargs)
