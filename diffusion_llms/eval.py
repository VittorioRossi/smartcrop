"""
accelerate launch \
    diffusion_llms/eval.py \
    --include-path "diffusion_llms/tasks" \
    --num_processes 2 \
    --tasks longformqa \
    --model llada \
    --device cuda \
    --num_fewshot 8 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"
"""

from typing import Optional

import json
import accelerate
import torch
from datasets import Dataset
from tqdm import tqdm
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype

from diffusion_llms.utils import get_device, get_model, get_tokenizer
from diffusion_llms.generators import BaseLengthGenerator, ZeroShotGenerator, AdapterGenerator
from diffusion_llms.generators import ZeroShotGeneratorConfig, AdapterGeneratorConfig
from diffusion_llms.utils.efficiency_metrics import (
    compute_nfe_efficiency_from_generator_output,
)

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def log_generation_metrics(
    log_path: str,
    example_id: int,
    generation_method: str,
    generation_length: int,
    messages: Optional[dict[str, str]] = None,
    task_name: Optional[str] = None,
    quantile: Optional[float] = None,
    raw_query: Optional[str] = None,
    **kwargs
):
    """Efficiently log generation metrics to JSONL file.

    Args:
        log_path: Path to the JSONL log file
        example_id: Global example ID (unique across all ranks)
        generation_method: Name of the generator class used
        generation_length: Number of new tokens generated
        messages: Dict with prompt, raw_generated, final_answer
        task_name: Name of the benchmark task
        quantile: EOS quantile used for length prediction
        raw_query: Original query text before chat templating
        **kwargs: Additional metrics (nfe, total_flops, etc.)
    """
    # Ensure directory exists
    dirname = os.path.dirname(log_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Format total_flops in scientific notation if present
    if kwargs.get('total_flops') is not None:
        total_flops = kwargs['total_flops']
        flops_formatted = f"{total_flops:.2e}"
        kwargs['total_flops_formatted'] = flops_formatted

    log_entry = {
        "example_id": example_id,
        "task_name": task_name,
        "quantile": quantile,
        "generation_method": generation_method,
        "generation_length": generation_length,
        "raw_query": raw_query,
        "messages": messages or [],
        **kwargs,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def tokenize_helper(e, tokenizer):
    """Helper function to tokenize data, defined at top level to avoid pickling 'self'."""
    tokenized = tokenizer(e["prompt_text"], add_special_tokens=False)

    return {
        "question": tokenized["input_ids"],
        "prompt_text": e["prompt_text"],
        "raw_query": e.get("raw_query", ""),
        "until": e["until"],
    }


@register_model("llada")
class VarLengthEvalHarness(LM):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        # Pull args from config, allow kwargs to override
        pretrained = kwargs.get("pretrained")
        if pretrained is None:
            raise ValueError("pretrained model path must be specified in model_args")
        
        generation_mode = kwargs.get("generation_mode", 'native')

        dtype = kwargs.get("dtype", "bfloat16")
        batch_size = kwargs.get("batch_size", 1)
        device = kwargs.get("device", get_device())
        cfg = kwargs.get("cfg", 0.0)
        temperature = kwargs.get("temperature", 0.0)
        steps = kwargs.get("steps", 128)
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        block_length = kwargs.get("block_length", 512)
        max_length = kwargs.get("max_length", 512)
        remasking = kwargs.get("remasking", "low_confidence")
        eos_quantile = kwargs.get("eos_quantile", 0)
        trust_remote_code = kwargs.get("trust_remote_code", "True")
        if isinstance(trust_remote_code, str):
            trust_remote_code = trust_remote_code.lower() == "true"
        
        attn_implementation = kwargs.get("attn_implementation", None)
        
        logits_eos_inf = kwargs.get("logits_eos_inf", True)
        if isinstance(logits_eos_inf, str):
            logits_eos_inf = logits_eos_inf.lower() == "true"
        
        assert remasking in ['low_confidence', 'random'], f"Invalid remasking strategy: {remasking}"
        self.debug_logpath = kwargs.get("debug_logpath", "generation_debug.jsonl")

        # Task metadata for logging
        self.task_name = kwargs.get("task_name", "unknown")
        self.quantile = float(eos_quantile)

        
        accelerator = accelerate.Accelerator()

        # Get GLOBAL rank from torch.distributed (not accelerator)
        # Use accelerator for rank/world_size as it's more robust with accelerate launch
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes
        
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"DEBUG: Initialized Accelerator. Rank: {self._rank}, World Size: {self._world_size}")
        logger.info(f"DEBUG: Accelerator State: {accelerator.state}")
        
        # Explicitly set the device to ensure NCCL binds correctly
        if accelerator.num_processes > 1:
            # Use process_index to ensure unique device assignment
            device_index = accelerator.process_index % torch.cuda.device_count()
            torch.cuda.set_device(device_index)
            logger.info(f"DEBUG: Set CUDA device to index {device_index} (accelerator.device={accelerator.device})")
            
            # Log physical device details
            try:
                props = torch.cuda.get_device_properties(device_index)
                logger.info(f"DEBUG: Device Name: {props.name}")
                logger.info(f"DEBUG: Device Total Memory: {props.total_memory}")
                # UUID might not be directly available in properties in older torch, but let's try
            except Exception as e:
                logger.info(f"DEBUG: Could not get device properties: {e}")

            
        import os
        logger.info(f"DEBUG: Env LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
        logger.info(f"DEBUG: Env WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
        logger.info(f"DEBUG: Env CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"DEBUG: Env MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
        print(f"DEBUG: Env MASTER_PORT: {os.environ.get('MASTER_PORT')}")
        print(f"DEBUG: Env RANK: {os.environ.get('RANK')}")

        # Load model with appropriate device_map based on distributed setup
        # When using accelerate multi-GPU, device_map must be None
        use_device_map = None if accelerator.num_processes > 1 else "auto"
        
        self.model = get_model(
            model_name_or_path=pretrained,
            dtype=get_dtype(dtype),
            device_map=use_device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.model.requires_grad_(False) # Ensure no gradients
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

        if accelerator.num_processes > 1:
            # Let accelerator handle device placement
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device  # ← Accelerator figures out local device correctly
            self.accelerator = accelerator
        else:
            # Single GPU - model already on correct device via device_map="auto"
            self.device = torch.device(device)
            self.accelerator = None

        self.tokenizer = get_tokenizer(
            model_name_or_path=pretrained,
            trust_remote_code=trust_remote_code,
        )
        
        self.generator: BaseLengthGenerator

        if generation_mode == "native":
            # we can just use the ZeroShotGenerator and passing quantile == 0.
            # It will not perform length prediction
            self.generator_config_class = ZeroShotGeneratorConfig 
            self.generator = ZeroShotGenerator(model=self.model, tokenizer=self.tokenizer)
        elif generation_mode == "zero_shot":
            self.generator_config_class = ZeroShotGeneratorConfig
            self.generator = ZeroShotGenerator(model=self.model, tokenizer=self.tokenizer)
        elif generation_mode == "adapter":
            self.generator_config_class = AdapterGeneratorConfig
            self.generator = AdapterGenerator(model=self.model, tokenizer=self.tokenizer)
        else:
            raise ValueError(f"Unknown generation mode: {generation_mode}")

        # generation params
        self.mask_id = self.tokenizer.mask_token_id
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.max_new_tokens = int(max_new_tokens)
        self.block_length = int(block_length)
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.remasking = remasking
        self.logits_eos_inf = logits_eos_inf

        self.generation_config = self.generator_config_class(
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            block_length=self.block_length,
            steps=self.steps,
            temperature=temperature,
            remasking=self.remasking,
            cfg_scale=self.cfg,
            eos_quantile=eos_quantile,
            safe_margin=16,
            return_dict_in_generate=True,
            measure_flops=True,
            logits_eos_inf=self.logits_eos_inf,
        )


    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return str(chat_templated)

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _get_rank_logpath(self) -> str:
        """Get rank-specific log file path for multi-GPU runs."""
        if self._world_size > 1:
            base, ext = os.path.splitext(self.debug_logpath)
            return f"{base}_rank{self._rank}{ext}"
        return self.debug_logpath

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        """Generate responses for evaluation requests with proper multi-GPU support.

        Note: When using `accelerate launch`, lm-eval already shards data across ranks.
        Each rank receives only its portion of requests, so we should NOT shard again.
        We process all requests we receive and use req.doc_id for global example IDs.

        For multi-GPU runs:
        - lm-eval handles data sharding (we receive only our portion)
        - We use req.doc_id for unique global example IDs
        - Each rank writes to its own log file (merged later by merge_rank_logs)
        """
        # 1. Preprocess all requests with original query text
        # Store the original request so we can access doc_id later
        processed_requests = []
        for req in requests:
            raw_query = req.arguments[0]  # Original query before templating
            messages = [{"role": "user", "content": raw_query}]

            templated_prompt = self.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            processed_requests.append({
                "prompt_text": templated_prompt,
                "raw_query": raw_query,
                "until": req.arguments[1]["until"],
                "doc_id": req.doc_id if req.doc_id is not None else req.idx,
            })

        ds = Dataset.from_list(processed_requests)
        ds = ds.map(
            tokenize_helper,
            fn_kwargs={"tokenizer": self.tokenizer}
        )
        ds = ds.with_format("torch")

        # 2. Process all requests (lm-eval already sharded for multi-GPU)
        # No manual sharding needed - lm-eval with accelerate handles it
        desc = f"Rank {self._rank}/{self._world_size}" if self._world_size > 1 else "Generating"

        # 3. Process all requests
        results = []
        generation_method = type(self.generator).__name__
        log_path = self._get_rank_logpath()

        for i, elem in enumerate(tqdm(ds, desc=desc)):
            example_id = elem["doc_id"].item() if hasattr(elem["doc_id"], 'item') else elem["doc_id"]
            prompt = [elem["question"].to(self.device)]
            stop_tokens = elem["until"]

            # Generate with the configured generator
            generator_output = self.generator.generate(
                prompts=prompt,
                config=self.generation_config,  # type: ignore
            )

            # Extract sequences and metrics
            generated_sequences = generator_output.sequences

            # Calculate new tokens generated
            generation_length = len(generated_sequences[0]) - len(prompt[0])
            total_flops = getattr(generator_output, 'total_flops', None)

            # Decode raw output
            raw_generated = self.tokenizer.decode(
                generated_sequences[0][len(prompt[0]):],
                skip_special_tokens=False
            )

            # Apply stop tokens
            generated_answer = raw_generated
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]
                    break

            # Clean final answer
            final_answer = self.tokenizer.decode(
                self.tokenizer.encode(generated_answer),
                skip_special_tokens=True
            ).strip()

            # Compute efficiency metrics
            config_dict = {
                "max_new_tokens": self.max_new_tokens,
                "block_length": self.block_length,
                "steps": self.steps,
                "cfg_scale": self.cfg,
                "temperature": self.generation_config.temperature,
                "remasking": self.remasking,
            }

            efficiency_metrics = compute_nfe_efficiency_from_generator_output(
                generator_output,
                prompt_lengths=[len(prompt[0])],
                config=config_dict,
                model_name=getattr(self.model, 'name_or_path', 'unknown'),
                generation_mode=generation_method
            )

            # Log with global example ID (doc_id) and task metadata
            log_generation_metrics(
                log_path=log_path,
                example_id=example_id,
                generation_method=generation_method,
                generation_length=generation_length,
                task_name=self.task_name,
                quantile=self.quantile,
                raw_query=elem["raw_query"],
                total_flops=total_flops,
                nfe=efficiency_metrics.nfe,
                nfe_per_token=efficiency_metrics.nfe_per_token,
                flops_per_token=efficiency_metrics.flops_per_token,
                messages={
                    "prompt": elem["prompt_text"],
                    "raw_generated": raw_generated,
                    "final_answer": final_answer,
                }
            )

            results.append(final_answer)

        # 4. Return results directly - lm-eval handles result gathering across ranks
        # No need to gather here because lm-eval already manages distributed evaluation
        return results

if __name__ == "__main__":
    cli_evaluate()
