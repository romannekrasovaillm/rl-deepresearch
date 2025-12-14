"""
Model wrappers for RL training.

Supports:
- GigaChat-Lightning (local or API)
- HuggingFace models with LoRA
- B200 optimization (bf16, Flash Attention, DeepSpeed)
"""

from .base import BaseModelWrapper, GenerationOutput
from .gigachat import GigaChatWrapper, GigaChatLocalWrapper
from .hf_wrapper import HFModelWrapper
from .utils import (
    setup_lora,
    setup_flash_attention,
    get_trainable_params,
    freeze_model,
)

__all__ = [
    "BaseModelWrapper",
    "GenerationOutput",
    "GigaChatWrapper",
    "GigaChatLocalWrapper",
    "HFModelWrapper",
    "setup_lora",
    "setup_flash_attention",
    "get_trainable_params",
    "freeze_model",
]
