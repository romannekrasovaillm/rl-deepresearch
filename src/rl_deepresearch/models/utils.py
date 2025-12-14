"""
Model utilities for B200 optimization and training setup.
"""

from typing import Any

import torch
import torch.nn as nn


def setup_lora(
    model: nn.Module,
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Apply LoRA to model.

    Key insight: LoRA as implicit regularization.
    Limited rank = limited capacity to overfit.

    Args:
        model: Base model
        r: LoRA rank
        alpha: LoRA alpha (scaling)
        dropout: Dropout rate
        target_modules: Modules to apply LoRA to

    Returns:
        Model with LoRA adapters
    """
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    return get_peft_model(model, config)


def setup_flash_attention(model: nn.Module) -> nn.Module:
    """
    Enable Flash Attention 2 if available.

    B200 optimized: significant speedup and memory savings.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2

        # Check if model supports flash attention
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "flash_attention_2"
            print("Flash Attention 2 enabled")
    except ImportError:
        print("Flash Attention 2 not available, using default attention")

    return model


def get_trainable_params(model: nn.Module) -> dict[str, Any]:
    """
    Get trainable parameter statistics.

    Returns:
        dict with:
        - total_params: int
        - trainable_params: int
        - trainable_percent: float
        - trainable_names: list[str]
    """
    total = 0
    trainable = 0
    trainable_names = []

    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            trainable_names.append(name)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_percent": 100 * trainable / total if total > 0 else 0,
        "trainable_names": trainable_names,
    }


def freeze_model(model: nn.Module, exceptions: list[str] | None = None) -> None:
    """
    Freeze all model parameters except specified.

    Args:
        model: Model to freeze
        exceptions: Parameter name patterns to keep trainable
    """
    exceptions = exceptions or []

    for name, param in model.named_parameters():
        should_freeze = True
        for pattern in exceptions:
            if pattern in name:
                should_freeze = False
                break

        param.requires_grad = not should_freeze


def prepare_optimizer_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_patterns: list[str] | None = None,
) -> list[dict]:
    """
    Prepare optimizer parameter groups with weight decay handling.

    Args:
        model: Model
        weight_decay: Weight decay value
        no_decay_patterns: Patterns for params without weight decay

    Returns:
        List of parameter groups for optimizer
    """
    if no_decay_patterns is None:
        no_decay_patterns = ["bias", "LayerNorm", "layer_norm"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(pattern in name for pattern in no_decay_patterns):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """
    Estimate memory usage for training.

    Returns memory in GB.
    """
    # Model parameters
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Gradients (same size as params)
    grad_bytes = param_bytes

    # Optimizer states (Adam: 2x params for m and v)
    optimizer_bytes = 2 * param_bytes

    # Activations (rough estimate)
    hidden_size = getattr(model.config, "hidden_size", 4096)
    num_layers = getattr(model.config, "num_hidden_layers", 32)
    bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    activation_bytes = (
        batch_size * seq_length * hidden_size * num_layers * bytes_per_element * 4
    )

    # Convert to GB
    to_gb = lambda x: x / (1024 ** 3)

    return {
        "parameters_gb": to_gb(param_bytes),
        "gradients_gb": to_gb(grad_bytes),
        "optimizer_gb": to_gb(optimizer_bytes),
        "activations_gb": to_gb(activation_bytes),
        "total_gb": to_gb(param_bytes + grad_bytes + optimizer_bytes + activation_bytes),
    }


def setup_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Enable gradient checkpointing for memory efficiency.

    Trades compute for memory - useful for B200 with large models.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    else:
        print("Model doesn't support gradient checkpointing")

    return model


def get_device_info() -> dict[str, Any]:
    """Get GPU device information."""
    if not torch.cuda.is_available():
        return {"available": False}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    return {
        "available": True,
        "device_name": props.name,
        "total_memory_gb": props.total_memory / (1024 ** 3),
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "is_b200": "B200" in props.name or "Blackwell" in props.name,
    }


class GradientAccumulator:
    """
    Gradient accumulation helper.

    Key for B200: allows larger effective batch sizes
    without OOM.
    """

    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.current_step = 0

    def backward(self, loss: torch.Tensor) -> bool:
        """
        Backward pass with accumulation.

        Returns True if gradients should be applied (accumulated enough).
        """
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        self.current_step += 1

        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True

        return False

    def clip_gradients(self) -> float:
        """Clip gradients and return norm."""
        return nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.model.zero_grad()


class MixedPrecisionContext:
    """
    Mixed precision training context manager.

    B200 optimized: use BF16 for best performance.
    """

    def __init__(
        self,
        dtype: str = "bf16",
        enabled: bool = True,
    ):
        self.dtype = dtype
        self.enabled = enabled
        self._context = None
        self._scaler = None

    def __enter__(self):
        if not self.enabled:
            return self

        if self.dtype == "bf16":
            self._context = torch.autocast("cuda", dtype=torch.bfloat16)
        elif self.dtype == "fp16":
            self._context = torch.autocast("cuda", dtype=torch.float16)
            self._scaler = torch.amp.GradScaler()

        if self._context:
            self._context.__enter__()

        return self

    def __exit__(self, *args):
        if self._context:
            self._context.__exit__(*args)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for FP16 training."""
        if self._scaler:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optimizer step with scaling."""
        if self._scaler:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
