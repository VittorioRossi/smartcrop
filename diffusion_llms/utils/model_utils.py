"""
Model utilities for diffusion-llms package.

Adapted from DLLM (Diffusion Language Models library):
https://github.com/regenerativeailab/dllm

Copyright notice: This code is adapted from the DLLM library under the Apache 2.0 license.
"""

import torch
import transformers
from typing import Optional, Union


def get_model(
    model_name_or_path: str,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device_map: Optional[str] = "auto",
    load_in_4bit: bool = False,
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = False,
) -> transformers.PreTrainedModel:
    """
    Load a model with flexible configuration.

    Args:
        model_name_or_path: Path or name of the model to load
        dtype: Data type for the model (string or torch.dtype)
        device_map: Device mapping strategy
        load_in_4bit: Whether to use 4-bit quantization
        load_in_4bit: Whether to use 4-bit quantization
        attn_implementation: Attention implementation to use
        trust_remote_code: Whether to trust remote code

    Returns:
        transformers.PreTrainedModel
    """
    # Handle dtype conversion
    if dtype is None:
        dtype = torch.float16
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    # Setup quantization config if requested
    quant_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    params = {
        'torch_dtype': dtype,
        'device_map': device_map,
        'quantization_config': quant_config,
        'quantization_config': quant_config,
        'attn_implementation': attn_implementation,
        'trust_remote_code': trust_remote_code,
    }

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    # Try loading as causal LM first (for generation), fallback to other types
    try:
        # Special handling for Dream models which are not CausalLM or MaskedLM in the standard sense
        if "Dream" in model_name_or_path or "dream" in model_name_or_path.lower():
             model = transformers.AutoModel.from_pretrained(model_name_or_path, **params)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, **params)
    except Exception:
        try:
            model = transformers.AutoModelForMaskedLM.from_pretrained(model_name_or_path, **params)
        except Exception as e2:
            print(f"Failed to load as MaskedLM: {e2}")
            model = transformers.AutoModel.from_pretrained(model_name_or_path, **params)

    return model


def get_tokenizer(
    model_name_or_path: str,
    padding_side: str = "right",
    trust_remote_code: bool = False,
) -> transformers.PreTrainedTokenizer:
    """
    Load a tokenizer with basic configuration.

    Args:
    Args:
        model_name_or_path: Path or name of the model/tokenizer to load
        padding_side: Which side to pad on ("left" or "right")
        trust_remote_code: Whether to trust remote code

    Returns:
        transformers.PreTrainedTokenizer
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side=padding_side,
        trust_remote_code=trust_remote_code,
    )

    # Ensure we have the necessary special tokens
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        tokenizer.eos_token = tokenizer.pad_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.pad_token

    try:
        model_config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if hasattr(model_config, 'model_type') and 'modernbert' in model_config.model_type.lower():
            tokenizer.eot_token = "[/Answer]"
            if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                tokenizer.chat_template = """\
{% if messages[0]['role'] == 'system' %}
[SYS]
{{ messages[0]['content'] | trim }}
[/SYS]

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif -%}
{%- for message in loop_messages %}
{% if message['role'] == 'user' %}
[Question]
{{ message['content'] | trim }}
[/Question]

{% elif message['role'] == 'assistant' %}
[Answer]
{{ message['content'] | trim }}
[/Answer]

{% endif %}
{% endfor -%}
{%- if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
[Answer]
{% endif %}
"""
    except Exception as e:
        print(f"Could not configure model-specific tokenizer settings: {e}")

    return tokenizer
