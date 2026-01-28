"""Base length-aware generator abstraction.

Core masked diffusion generation logic with optional pre-computed
`length_prediction` and per-step callback. Supports FLOPs accounting.
"""

from dataclasses import dataclass
from typing import Callable, Any, Optional
import math
import torch
import torch.nn.functional as F

from dllm.core.generation.generator import (
    GeneratorOutput,
    GeneratorConfig,
    BaseGenerator
)
from dllm.utils.generation_utils import get_num_transfer_tokens
from diffusion_llms.utils.generation_utils import (
    estimate_forward_flops,
    PromptInput,
    convert_prompts_to_tensors,
)
from diffusion_llms.utils.common import add_gumbel_noise, _ensure_token_id


@dataclass
class BaseGeneratorOutput(GeneratorOutput):
    predicted_length: int | None = None  # Predicted total length (prompt + generation)
    total_flops: float | None = None  # Total FLOPs during generation
    per_forward_flops: float | None = None  # FLOPs per generated token

@dataclass
class BaseLengthGeneratorConfig(GeneratorConfig):
    max_new_tokens: int = 128
    max_length: int | None = None
    block_length: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"  # or "random"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    length_prediction: int | None = None  # externally predicted total length (prompt+gen)
    step_callback: Callable[[dict[str, Any]], Any] | None = None
    logits_eos_inf: bool = False  # Whether to mask EOS token logits to -inf

@dataclass
class BaseLengthGenerator(BaseGenerator):
    @torch.no_grad()
    def generate(
        self,
        prompts: PromptInput,
        config: Optional[BaseLengthGeneratorConfig] = None,
        **kwargs,
    ) -> GeneratorOutput:
        
        # Config normalization
        if config is None:
            config = BaseLengthGeneratorConfig()
        if not isinstance(config, BaseLengthGeneratorConfig):
            upgraded = BaseLengthGeneratorConfig()
            upgraded.return_dict_in_generate = getattr(
                config, "return_dict_in_generate", False
            )
            upgraded.measure_flops = getattr(config, "measure_flops", False)
            config = upgraded

        # Params & overrides
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_length = kwargs.get("block_length", config.block_length)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict_in_generate = kwargs.get(
            "return_dict_in_generate", config.return_dict_in_generate
        )
        length_prediction = kwargs.get("length_prediction", config.length_prediction)
        logits_eos_inf = kwargs.get("logits_eos_inf", getattr(config, "logits_eos_inf", False))

        assert 1 <= block_length
        assert 1 <= steps

                
        device = self.model.device

        # Convert prompts to tensors
        inputs = convert_prompts_to_tensors(prompts, device)


        mask_id = _ensure_token_id(
            getattr(self.tokenizer, "mask_token_id", None), 
            "mask_token_id", 
            tokenizer=self.tokenizer,
            candidates=["<mask>", "[MASK]", "<|mask|>", "<|mdm_mask|>", "[gMASK]"]
        )
        eos_id = _ensure_token_id(getattr(self.tokenizer, "eos_token_id", None), "eos_token_id", tokenizer=self.tokenizer)

        prompt_lens = [int(p.shape[0]) for p in inputs]

        if max_new_tokens and max_new_tokens > 0:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        if length_prediction is not None:
            assert len(inputs) == 1
            max_length = min(length_prediction, max_length)
            max_new_tokens = max_length - prompt_lens[0]

        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id
        attention_mask = (x != eos_id).bool()

        unmasked_index = (x != mask_id) & (x != eos_id)
        if cfg_keep_tokens:
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(max_new_tokens / block_length)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict_in_generate else None

        measure_flops = getattr(config, "measure_flops", False)
        per_forward_flops = None
        total_flops = None
        
        if measure_flops:
            per_forward_flops = estimate_forward_flops(
                self.model, seq_len=T, batch_size=B, cfg_active=(cfg_scale > 0.0)
            )
            if per_forward_flops is not None:
                total_flops = 0.0


        for b in range(num_blocks):
            block_mask_index = torch.zeros(
                (B, block_length), dtype=torch.bool, device=device
            )
            for j in range(B):
                start = prompt_lens[j] + b * block_length
                end = min(start + block_length, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = x[j, start:end] == mask_id

            assert self.scheduler is not None
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_global = x == mask_id
                if cfg_scale > 0.0:
                    # Optimize masking: use masked_fill_ on cloned tensor
                    un_x = x.clone().masked_fill_(unmasked_index, mask_id)
                    x_ = torch.cat([x, un_x], dim=0)
                    att_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = self.model(x_, attention_mask=att_mask_).logits
                                            
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    
                    # Accumulate FLOPs for CFG forward pass (includes both conditional and unconditional)
                    if measure_flops and per_forward_flops is not None:
                        total_flops += per_forward_flops  # CFG already accounted for in per_forward_flops
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits
                                        
                    # Accumulate FLOPs for regular forward pass
                    if measure_flops and per_forward_flops is not None:
                        total_flops += per_forward_flops

                if logits_eos_inf:
                    logits[:, :, eos_id] = -torch.inf


                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    probs = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_length :] = -torch.inf

                x0 = torch.where(mask_index_global, x0, x)
                confidence = torch.where(mask_index_global, x0_p, -torch.inf)

                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, idx = torch.topk(confidence[j], k=k)
                        x[j, idx] = x0[j, idx]

                if histories is not None:
                    histories.append(x.clone())

        return BaseGeneratorOutput(
            sequences=x,
            histories=histories,
            total_flops=total_flops,
            per_forward_flops=per_forward_flops,
            predicted_length=length_prediction,
        )

    @torch.no_grad()
    def infill(
        self,
        inputs: PromptInput,
        config: GeneratorConfig | None = None,
        **kwargs,
    ) -> GeneratorOutput:
        raise NotImplementedError("Infill not implemented for BaseLengthGenerator yet.")