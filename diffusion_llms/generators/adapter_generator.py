"""Adapter-based length-aware generator built on top of `BaseLengthGenerator`.

This variant uses a pre-trained MLP adapter to predict sequence length based on
the model's hidden states. It:
  1. Loads a pre-trained MLP adapter from a given path
  2. Extracts hidden states from the model for the input prompt
  3. Uses the adapter and length_estimator_fn to predict final sequence length
  4. Delegates the actual masked diffusion loop to `BaseLengthGenerator` by
     passing the predicted length via the inherited `length_prediction` field.

All other generation mechanics (FLOPs accounting, CFG, remasking strategy,
callback support) are handled by the base generator.
"""

from dataclasses import dataclass
from typing import Sequence, Callable
import torch
import torch.nn as nn

from diffusion_llms.generators.base_length_generator import (
    BaseLengthGenerator,
    BaseLengthGeneratorConfig,
)
from dllm.core.generation.generator import (
    GeneratorOutput,
    GeneratorConfig,
    BaseGenerator
)


@dataclass
class AdapterGeneratorConfig(BaseLengthGeneratorConfig):
    adapter_path: str | None = None  # Path to the pre-trained MLP adapter
    length_estimator_fn: Callable[[nn.Module, torch.Tensor], int] | None = None  # Function to estimate length from hidden states


@dataclass
class AdapterGenerator(BaseLengthGenerator):
    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        self._adapter_cache: dict[str, nn.Module] = {}

    def _load_adapter(self, adapter_path: str) -> nn.Module:
        """Load and cache MLP adapter from the given path."""
        if adapter_path in self._adapter_cache:
            return self._adapter_cache[adapter_path]
        
        adapter = torch.load(adapter_path, map_location=self.model.device)
        if isinstance(adapter, dict):
            # If it's a state dict, we need to construct the MLP
            # For now, assume it's already a nn.Module
            raise ValueError("Adapter must be a saved nn.Module, not a state dict")
        
        adapter.eval()
        adapter.to(self.model.device)
        self._adapter_cache[adapter_path] = adapter
        return adapter

    def _validate_adapter(self, adapter: nn.Module, hidden_states: torch.Tensor) -> None:
        """Validate that the adapter input shape matches the hidden states shape."""
        # Get the first linear layer to check input dimension
        first_layer = None
        for module in adapter.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        
        if first_layer is None:
            raise ValueError("Adapter must contain at least one nn.Linear layer")
        
        expected_input_dim = first_layer.in_features
        actual_input_dim = hidden_states.size(-1)
        
        if expected_input_dim != actual_input_dim:
            raise ValueError(
                f"Adapter input dimension {expected_input_dim} does not match "
                f"hidden states dimension {actual_input_dim}"
            )

    def _extract_hidden_states(self, prompt: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from the model for the given prompt."""
        with torch.no_grad():
            # Create attention mask for the prompt
            attention_mask = torch.ones_like(prompt, dtype=torch.long)
            
            # Forward pass to get hidden states
            outputs = self.model(prompt.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), output_hidden_states=True)
            
            # Get the last hidden states and take the mean across sequence length
            # Shape: (batch_size, seq_len, hidden_dim) -> (hidden_dim,)
            hidden_states = outputs.hidden_states[-1].squeeze(0)  # Remove batch dim
            pooled_hidden_states = hidden_states.mean(dim=0)  # Pool across sequence length
            
            return pooled_hidden_states

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[torch.Tensor] | Sequence[Sequence[int]],
        config: GeneratorConfig | None = None,
        **kwargs,
    ) -> GeneratorOutput:
        # Normalize / upgrade config
        if config is None:
            config = AdapterGeneratorConfig()
        if not isinstance(config, AdapterGeneratorConfig):
            upgraded = AdapterGeneratorConfig()
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

        # If external length_prediction not provided and adapter is available, predict using MLP.
        predicted_length: int | None = None
        if (
            config.length_prediction is None
            and config.adapter_path is not None
            and config.length_estimator_fn is not None
            and len(tensor_prompts) == 1
        ):
            prompt_tensor = tensor_prompts[0]
            
            # Load and validate adapter
            adapter = self._load_adapter(config.adapter_path)
            
            # Extract hidden states
            hidden_states = self._extract_hidden_states(prompt_tensor)
            
            # Validate adapter compatibility
            self._validate_adapter(adapter, hidden_states)
            
            # Use the length estimator function to predict length
            predicted_length = config.length_estimator_fn(adapter, hidden_states)
            
            # Ensure predicted length is valid
            if not isinstance(predicted_length, int) or predicted_length <= 0:
                raise ValueError(
                    f"length_estimator_fn must return a positive integer, got {predicted_length}"
                )

        # Build a base config clone and set length_prediction if we inferred one.
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
        ]:
            setattr(base_cfg, attr, getattr(config, attr))
        base_cfg.length_prediction = config.length_prediction or predicted_length

        return super().generate(prompts=tensor_prompts, config=base_cfg, **kwargs)