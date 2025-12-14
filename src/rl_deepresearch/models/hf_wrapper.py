"""
Generic HuggingFace model wrapper.

Can be used with any causal LM from HuggingFace Hub.
Useful for experiments with open models like Qwen, Llama, etc.
"""

import torch
import torch.nn.functional as F
from typing import Any

from ..config import ModelConfig
from .base import BaseModelWrapper, GenerationOutput, BatchGenerationOutput


class HFModelWrapper(BaseModelWrapper):
    """
    Generic wrapper for HuggingFace causal language models.

    Supports same optimizations as GigaChat wrapper:
    - LoRA
    - Flash Attention
    - Gradient checkpointing
    - BF16/FP16
    """

    def __init__(
        self,
        model_name: str,
        config: ModelConfig | None = None,
        device: str = "cuda",
        **kwargs,
    ):
        self.model_name = model_name
        self.config = config or ModelConfig(model_name=model_name)
        self.device = device

        self.model = None
        self.tokenizer = None
        self.reference_model = None  # For KL computation

        self._load_model(**kwargs)

    def _load_model(self, **kwargs) -> None:
        """Load model with optimizations."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name or self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            **kwargs,
        }

        # Attention implementation
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif self.config.use_sdpa:
            model_kwargs["attn_implementation"] = "sdpa"

        # Quantization
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        elif self.config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Apply LoRA
        if self.config.use_lora:
            self._apply_lora()

        # Gradient checkpointing
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def _apply_lora(self) -> None:
        """Apply LoRA adapters."""
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def create_reference_model(self) -> None:
        """
        Create frozen reference model for KL penalty.

        Key insight: KL-penalty anchors against forgetting.
        Reference model is the SFT checkpoint.
        """
        if self.reference_model is not None:
            return

        from copy import deepcopy

        if self.config.use_lora:
            # For LoRA, reference is base model without adapters
            self.reference_model = self.model.get_base_model()
        else:
            # Full copy for full fine-tuning
            self.reference_model = deepcopy(self.model)

        # Freeze reference
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
        **kwargs,
    ) -> GenerationOutput:
        """Generate text."""
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length - max_new_tokens,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=return_logprobs,
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0, prompt_length:].tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        logprobs = None
        if return_logprobs and outputs.scores:
            logprobs = self._scores_to_logprobs(outputs.scores, generated_ids)

        return GenerationOutput(
            text=generated_text,
            token_ids=generated_ids,
            logprobs=logprobs,
        )

    def _scores_to_logprobs(
        self,
        scores: tuple[torch.Tensor, ...],
        token_ids: list[int],
    ) -> torch.Tensor:
        """Convert generation scores to log probabilities."""
        logprobs = []
        for score, token_id in zip(scores, token_ids):
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
        """Batch generation with padding."""
        self.model.eval()

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_context_length - max_new_tokens,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        texts = []
        token_ids = []
        for i, output_ids in enumerate(outputs):
            prompt_len = (inputs["attention_mask"][i] == 1).sum().item()
            gen_ids = output_ids[prompt_len:].tolist()
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            texts.append(text)
            token_ids.append(gen_ids)

        return BatchGenerationOutput(texts=texts, token_ids=token_ids)

    def compute_logprobs(
        self,
        prompt: str,
        completion: str,
    ) -> torch.Tensor:
        """Compute log probabilities for completion."""
        full_text = prompt + completion
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length,
        ).to(self.model.device)

        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Shift for autoregressive
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs[0, prompt_len - 1:]

    def compute_logprobs_batch(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> list[torch.Tensor]:
        """Batch log probability computation."""
        return [self.compute_logprobs(p, c) for p, c in zip(prompts, completions)]

    def compute_kl_divergence(
        self,
        prompt: str,
        completion: str,
    ) -> torch.Tensor:
        """
        Compute KL divergence from reference model.

        KL(policy || reference) for regularization.
        """
        if self.reference_model is None:
            self.create_reference_model()

        full_text = prompt + completion
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        prompt_len = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

        with torch.no_grad():
            policy_outputs = self.model(**inputs)
            ref_outputs = self.reference_model(**inputs)

        # Get distributions for completion tokens
        policy_logits = policy_outputs.logits[:, prompt_len - 1:-1, :]
        ref_logits = ref_outputs.logits[:, prompt_len - 1:-1, :]

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # KL divergence per token
        kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)

        return kl.mean()

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Get trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        from transformers import AutoModelForCausalLM
        if self.config.use_lora:
            from peft import PeftModel
            base = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
            self.model = PeftModel.from_pretrained(base, path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path).to(self.model.device)

    def prepare_for_training(self) -> None:
        """Prepare for training."""
        self.model.train()
        if self.config.use_lora:
            for name, param in self.model.named_parameters():
                param.requires_grad = "lora" in name.lower()

    def prepare_for_inference(self) -> None:
        """Prepare for inference."""
        self.model.eval()
