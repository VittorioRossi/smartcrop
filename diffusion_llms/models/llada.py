# Consolidated LLaDA implementation with official generate method

import torch
import torch.nn as nn
import lightning as pl
from transformers.models.auto.modeling_auto import AutoModel
import os
from diffusion_llms.utils.config_helper import TrainingConfig
from dataclasses import dataclass
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


class RandomLLaDa(nn.Module):
    """A model that only checks the input logic, and returns random, for debugging purposes."""

    @dataclass
    class RandomLLaDaOutput:
        last_hidden_state: torch.Tensor
        logits: torch.Tensor

    def __init__(self):
        super().__init__()
        self.id = nn.Identity()
        self.d_model = 4096

    def forward(self, input_ids, attention_mask=None):
        assert isinstance(input_ids, torch.Tensor)
        assert input_ids.ndim == 2
        assert input_ids.dtype == torch.long
        last_hidden_state = torch.randn(
            size=(input_ids.shape[0], input_ids.shape[1], self.d_model)
        )
        # Create random logits for vocabulary size (approximate)
        logits = torch.randn(
            size=(input_ids.shape[0], input_ids.shape[1], 50000)
        )
        return RandomLLaDa.RandomLLaDaOutput(last_hidden_state, logits)
    
    def generate(self, input_ids, max_length: int = 50, **kwargs):
        batch_size = input_ids.shape[0]
        generated_ids = torch.randint(
            low=0, high=50000, size=(batch_size, max_length), dtype=torch.long
        )
        return generated_ids


class LLaDa(nn.Module):
    def __init__(
        self, cache_dir: str = "cache", device: str = "auto", debug: bool = False
    ):
        super().__init__()

        # For accessing faster
        self.d_model = 4096

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        if debug:
            self.model = RandomLLaDa()
        else:
            self.model = AutoModel.from_pretrained(
                "GSAI-ML/LLaDA-8B-Instruct",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        # Move to device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        # Freeze gradients and set in eval mode
        self.model = self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Mask token ID
        self.mask_id = 126336

    def forward(self, input_ids):
        """Forward pass through the model"""
        return self.model(input_ids)

    def get_last_hidden_state(self, input_ids):
        # TODO: implement caching logic here
        # by creating a lookup dict
        # input_ids -> idx in tiledb object
        model_output = self.model(
            input_ids, return_dict=True, output_hidden_states=True
        )
        hidden_state = model_output.hidden_states[-1]
        return hidden_state
    
    def generate(self, prompt: str, config=None, **kwargs):
        """Generate text given a prompt using the model's generate method.
        
        Args:
            prompt: Input text prompt to generate from
            config: GenerationConfig instance. If None, creates default config from kwargs.
            **kwargs: Fallback parameters if config is not provided
            
        Returns:
            Generated text as string
        """
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        from diffusion_llms.utils.generation_config import GenerationConfig
        from diffusion_llms.generate import generate

        tokenizer = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
        )

        # Tokenize input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set mask token ID if not present in tokenizer
        if tokenizer.mask_token_id is None:
            tokenizer.mask_token_id = self.mask_id
        
        # Set pad token ID if not present (often same as EOS token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Create or use provided config
        if config is None:
            # Create config from kwargs with defaults
            config = GenerationConfig(
                steps=kwargs.get('steps', 128),
                gen_length=kwargs.get('gen_length', kwargs.get('max_length', 128)),
                block_length=kwargs.get('block_length', 32),
                temperature=kwargs.get('temperature', 0.0),
                cfg_scale=kwargs.get('cfg_scale', 0.0),
                remasking=kwargs.get('remasking', 'low_confidence'),
                mask_id=self.mask_id,
                logits_eos_inf=kwargs.get('logits_eos_inf', False),
                confidence_eos_eot_inf=kwargs.get('confidence_eos_eot_inf', False)
            )

        generated_ids = generate(
            model=self.model,
            prompt=inputs['input_ids'],
            config=config,
            attention_mask=inputs.get('attention_mask', None),
            tokenizer=tokenizer
        )

        # Decode generated ids to text
        generated_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        print(f"Generated IDs: {generated_ids}")

        return generated_text


class FFNN(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.d_model = d_model
        self.sequential = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x):
        assert x.shape[-1] == self.d_model, f"Expected [..., d_model], got {x.shape}"
        return self.sequential(x)


# Base Model to Train
class BaseModel(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.llada = LLaDa(cache_dir="", debug=config.debug)
        self.d_model = 4096
        self.lr = getattr(config, 'lr', 1e-3)  # Default learning rate
        self.n_steps = getattr(config, 'n_steps', 1000)  # Default number of steps

    def forward(
        self,
        input_ids,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Implement the .forward() logic: from model input to model output."
        )

    def step(self, batch, batch_idx) -> dict:
        # Implement the logic, either bce / mse
        raise NotImplementedError(
            "Implement the .step() logic: from model output to loss and metrics."
        )

    def _common_step(self, batch, batch_idx, stage: str):
        # Can return also accuracy, dispersion etc
        metrics = self.step(batch, batch_idx)

        # We must have loss to train
        assert "loss" in metrics, "'loss' required in .step() output"

        # Log everything
        for k, v in metrics.items():
            assert isinstance(v, torch.Tensor)
            self.log(f"{stage}_{k}", v.cpu().detach().item(), prog_bar=True)

        # We always train on loss (mse/bce)
        return metrics["loss"]

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            betas=(0.9, 0.9),
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=self.n_steps,
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,  # Check after each step
            },
        }


class LLaDaClassifier(BaseModel):
    """LLaDA with classification head for EOS token prediction"""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.classifier = FFNN(d_model=self.d_model, out_dim=1)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        hidden_states = self.llada.get_last_hidden_state(input_ids)
        # Predicted logit (eos/non-eos)
        # [B, seq_len, 1]
        logits = self.classifier(hidden_states)
        # [B, seq_len]
        return logits.squeeze(-1)

    def step(self, batch, batch_idx) -> dict:
        # Unpack the batch
        # LongTensor, LongTensor, FloatTensor, BoolTensor
        # eos_labels= 1 if is eos, 0 otherwise
        input_ids, eos_labels, response_length, input_mask = batch.values()

        logits = self.forward(input_ids)

        assert eos_labels.shape == logits.shape, (
            f"eos_labels.shape={eos_labels.shape}, logits.shape={logits.shape}"
        )
        assert isinstance(logits, torch.FloatTensor)
        assert isinstance(eos_labels, torch.LongTensor)

        # BCE
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits[input_mask].float(), target=eos_labels[input_mask].float()
        )

        # Accuracy
        with torch.no_grad():
            eos_probs = torch.nn.functional.sigmoid(logits)
            accuracy = ((eos_probs > 0.5) == eos_labels.bool())[
                input_mask
            ].sum() / input_mask.sum()

        return {"loss": loss, "accuracy": accuracy}


class LLaDaRegressor(BaseModel):
    """LLaDA with regression head for length prediction"""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.regressor = FFNN(d_model=self.d_model, out_dim=1)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        # [B, seq_len, d_model]
        hidden_state = self.llada.get_last_hidden_state(input_ids)
        # Get sentence-level embedding
        # mean along the seq_len dimension
        # [B, d_model]
        pooled_hidden_state = torch.mean(hidden_state, dim=1)
        assert pooled_hidden_state.ndim == 2
        # [B, seq_len]
        predicted_length = self.regressor(pooled_hidden_state)

        return predicted_length

    def step(self, batch, batch_idx) -> dict:
        # Unpack the batch
        # LongTensor, LongTensor, FloatTensor, BoolTensor
        # eos_labels= 1 if is eos, 0 otherwise
        input_ids, eos_labels, response_length, input_mask = batch.values()

        predicted_length = self.forward(input_ids)

        assert response_length.shape == predicted_length.shape, (
            f"response_length.shape={response_length.shape}, predicted_length.shape={predicted_length.shape}"
        )
        assert isinstance(predicted_length, torch.FloatTensor)
        assert isinstance(response_length, torch.FloatTensor)

        # BCE
        loss = torch.nn.functional.mse_loss(
            input=predicted_length.float(), target=response_length.float()
        )

        # Dispersion
        with torch.no_grad():
            std = torch.std(predicted_length)

        return {"loss": loss, "dispersion": std}
