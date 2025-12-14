"""
Base model wrapper interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GenerationOutput:
    """Output from model generation."""

    text: str
    token_ids: list[int]
    logprobs: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class BatchGenerationOutput:
    """Output from batch generation."""

    texts: list[str]
    token_ids: list[list[int]]
    logprobs: list[torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModelWrapper(ABC):
    """
    Base class for model wrappers.

    Provides unified interface for:
    - Generation (inference)
    - Log probability computation (for RL)
    - Gradient computation (for training)
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
        **kwargs,
    ) -> GenerationOutput:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> BatchGenerationOutput:
        """Generate text for batch of prompts."""
        pass

    @abstractmethod
    def compute_logprobs(
        self,
        prompt: str,
        completion: str,
    ) -> torch.Tensor:
        """
        Compute log probabilities for completion given prompt.

        Used for policy gradient computation.
        """
        pass

    @abstractmethod
    def compute_logprobs_batch(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> list[torch.Tensor]:
        """Compute log probabilities for batch."""
        pass

    @abstractmethod
    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Get parameters that should be trained."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        pass

    def prepare_for_training(self) -> None:
        """Prepare model for training (enable gradients, etc.)."""
        pass

    def prepare_for_inference(self) -> None:
        """Prepare model for inference (disable gradients, etc.)."""
        pass


class TokenMasking:
    """
    Utility for masking tokens during loss computation.

    Key insight: mask tool output tokens from gradients.
    Model should learn what actions to take, not predict tool outputs.
    """

    @staticmethod
    def create_action_mask(
        token_ids: list[int],
        action_start_tokens: list[int],
        action_end_tokens: list[int],
        include_thinking: bool = True,
    ) -> torch.Tensor:
        """
        Create mask that is 1 for action tokens, 0 for observation tokens.

        Args:
            token_ids: Full sequence token IDs
            action_start_tokens: Tokens that start action (e.g., '{"name":')
            action_end_tokens: Tokens that end action (e.g., '}')
            include_thinking: Whether to include thinking tokens in mask

        Returns:
            Boolean mask tensor
        """
        mask = torch.zeros(len(token_ids), dtype=torch.bool)

        in_action = False
        brace_count = 0

        for i, token in enumerate(token_ids):
            # Detect action start
            if token in action_start_tokens:
                in_action = True
                brace_count = 1
                mask[i] = True
                continue

            if in_action:
                mask[i] = True

                # Track braces for JSON
                if token in action_start_tokens:
                    brace_count += 1
                elif token in action_end_tokens:
                    brace_count -= 1
                    if brace_count <= 0:
                        in_action = False

        return mask

    @staticmethod
    def create_turn_masks(
        token_ids: list[int],
        turn_boundaries: list[int],
    ) -> list[torch.Tensor]:
        """
        Create separate masks for each turn.

        Useful for turn-level advantage computation (MT-GRPO).

        Args:
            token_ids: Full sequence token IDs
            turn_boundaries: Token indices where turns start

        Returns:
            List of masks, one per turn
        """
        masks = []
        num_tokens = len(token_ids)

        for i in range(len(turn_boundaries)):
            start = turn_boundaries[i]
            end = turn_boundaries[i + 1] if i + 1 < len(turn_boundaries) else num_tokens

            mask = torch.zeros(num_tokens, dtype=torch.bool)
            mask[start:end] = True
            masks.append(mask)

        return masks
