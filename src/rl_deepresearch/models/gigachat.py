"""
GigaChat model wrappers.

Supports:
- GigaChat API (for inference/rollouts)
- Local GigaChat-Lightning (for training)
"""

import os
import json
from typing import Any

import torch
import torch.nn.functional as F

from ..config import ModelConfig
from .base import BaseModelWrapper, GenerationOutput, BatchGenerationOutput


class GigaChatWrapper(BaseModelWrapper):
    """
    Wrapper for GigaChat API.

    Used for:
    - Rollout generation (fast inference)
    - Reference policy (KL computation)

    Note: API doesn't support gradient computation.
    For training, use GigaChatLocalWrapper.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "GigaChat-Lightning",
        base_url: str = "https://gigachat.devices.sberbank.ru/api/v1",
    ):
        self.api_key = api_key or os.environ.get("GIGACHAT_API_KEY")
        self.model_name = model_name
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("GigaChat API key required. Set GIGACHAT_API_KEY env var.")

        self._client = None

    @property
    def client(self):
        """Lazy client initialization."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
        **kwargs,
    ) -> GenerationOutput:
        """Generate using GigaChat API."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        text = data["choices"][0]["message"]["content"]

        return GenerationOutput(
            text=text,
            token_ids=[],  # API doesn't return token IDs
            logprobs=None,
            metadata={
                "model": self.model_name,
                "usage": data.get("usage", {}),
            },
        )

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> BatchGenerationOutput:
        """Generate for batch (sequential API calls)."""
        texts = []
        for prompt in prompts:
            output = self.generate(prompt, max_new_tokens, temperature, top_p, **kwargs)
            texts.append(output.text)

        return BatchGenerationOutput(
            texts=texts,
            token_ids=[[] for _ in texts],
            metadata={"batch_size": len(prompts)},
        )

    def compute_logprobs(self, prompt: str, completion: str) -> torch.Tensor:
        """Not supported for API-only wrapper."""
        raise NotImplementedError("GigaChat API doesn't support logprob computation. Use GigaChatLocalWrapper.")

    def compute_logprobs_batch(self, prompts: list[str], completions: list[str]) -> list[torch.Tensor]:
        raise NotImplementedError("Use GigaChatLocalWrapper for training.")

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        return []  # API wrapper has no trainable params

    def save_checkpoint(self, path: str) -> None:
        pass

    def load_checkpoint(self, path: str) -> None:
        pass


class GigaChatLocalWrapper(BaseModelWrapper):
    """
    Local GigaChat-Lightning wrapper for training.

    Optimized for B200 GPU:
    - BF16 precision
    - Flash Attention 2
    - LoRA adapters
    - Gradient checkpointing
    """

    def __init__(
        self,
        config: ModelConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        self.model = None
        self.tokenizer = None
        self.lora_config = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model with B200 optimizations."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        # Determine dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        # Load tokenizer
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )

        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        # Flash Attention 2 (B200 optimized)
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif self.config.use_sdpa:
            model_kwargs["attn_implementation"] = "sdpa"

        # Quantization (optional)
        if self.config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        # Apply LoRA
        if self.config.use_lora:
            self.lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()

        # Move to device if not using device_map
        if self.device != "cuda" or "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
        **kwargs,
    ) -> GenerationOutput:
        """Generate text locally."""
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length - max_new_tokens,
        ).to(self.device)

        # Filter out unsupported kwargs (e.g., token_type_ids for DeepseekV3)
        inputs = self._filter_model_inputs(inputs)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=return_logprobs,
            )

        # Extract generated tokens (excluding prompt)
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0, prompt_length:].tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute logprobs if requested
        logprobs = None
        if return_logprobs and outputs.scores:
            logprobs = self._compute_generation_logprobs(outputs.scores, generated_ids)

        return GenerationOutput(
            text=generated_text,
            token_ids=generated_ids,
            logprobs=logprobs,
            metadata={
                "prompt_tokens": prompt_length,
                "generated_tokens": len(generated_ids),
            },
        )

    def _compute_generation_logprobs(
        self,
        scores: tuple[torch.Tensor, ...],
        generated_ids: list[int],
    ) -> torch.Tensor:
        """Compute log probabilities from generation scores."""
        logprobs = []
        for score, token_id in zip(scores, generated_ids):
            log_probs = F.log_softmax(score[0], dim=-1)
            logprobs.append(log_probs[token_id].item())
        return torch.tensor(logprobs)

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> BatchGenerationOutput:
        """Generate for batch with padding."""
        self.model.eval()

        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_context_length - max_new_tokens,
        ).to(self.device)

        # Filter out unsupported kwargs (e.g., token_type_ids for DeepseekV3)
        inputs = self._filter_model_inputs(inputs)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode each sequence
        texts = []
        token_ids = []
        for i, output_ids in enumerate(outputs):
            prompt_length = (inputs["attention_mask"][i] == 1).sum().item()
            generated = output_ids[prompt_length:].tolist()
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            texts.append(text)
            token_ids.append(generated)

        return BatchGenerationOutput(
            texts=texts,
            token_ids=token_ids,
            metadata={"batch_size": len(prompts)},
        )

    def compute_logprobs(
        self,
        prompt: str,
        completion: str,
    ) -> torch.Tensor:
        """
        Compute log probabilities for completion given prompt.

        Key for policy gradient: we need per-token logprobs
        for the policy update.
        """
        self.model.eval()

        # Tokenize prompt + completion
        full_text = prompt + completion
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length,
        ).to(self.device)

        # Filter out unsupported kwargs (e.g., token_type_ids for DeepseekV3)
        inputs = self._filter_model_inputs(inputs)

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        prompt_length = prompt_inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Shift for autoregressive loss
        shift_logits = logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Return only completion logprobs (after prompt)
        completion_logprobs = token_log_probs[0, prompt_length - 1:]

        return completion_logprobs

    def compute_logprobs_batch(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> list[torch.Tensor]:
        """Compute log probs for batch."""
        return [
            self.compute_logprobs(p, c)
            for p, c in zip(prompts, completions)
        ]

    def compute_forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass for training.

        Returns logits and optionally loss.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
        }

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Get trainable parameters (LoRA params if using LoRA)."""
        if self.config.use_lora:
            return [p for p in self.model.parameters() if p.requires_grad]
        return list(self.model.parameters())

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        if self.config.use_lora:
            # Save only LoRA weights
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        if self.config.use_lora:
            from peft import PeftModel
            # Load base model first, then LoRA
            self.model = PeftModel.from_pretrained(self.model, path)
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)

    def prepare_for_training(self) -> None:
        """Enable training mode and gradients."""
        self.model.train()
        if self.config.use_lora:
            # Ensure only LoRA params require grad
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def prepare_for_inference(self) -> None:
        """Disable gradients for fast inference."""
        self.model.eval()
        torch.set_grad_enabled(False)

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def _filter_model_inputs(self, inputs: dict) -> dict:
        """
        Remove tokenizer outputs not supported by the model.

        Some models (e.g., DeepseekV3) don't accept token_type_ids
        even though the tokenizer returns them.
        """
        # Keys that some models don't support
        unsupported_keys = {"token_type_ids"}
        return {k: v for k, v in inputs.items() if k not in unsupported_keys}


def create_model_wrapper(
    config: ModelConfig,
    use_api: bool = False,
    api_key: str | None = None,
) -> BaseModelWrapper:
    """Factory function to create model wrapper."""
    if use_api:
        return GigaChatWrapper(api_key=api_key, model_name=config.model_name)
    else:
        return GigaChatLocalWrapper(config=config)
