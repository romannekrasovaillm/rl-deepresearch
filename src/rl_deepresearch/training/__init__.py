"""
Training module for RL Deep Research.

Implements:
- GRPO (Group Relative Policy Optimization)
- Async rollout generation
- B200-optimized training loop
- Curriculum learning
"""

from .grpo import GRPOTrainer, GRPOBatch
from .rollout import RolloutGenerator, AsyncRolloutGenerator
from .data import ResearchDataset, DataCollator
from .callbacks import (
    TrainingCallback,
    WandbCallback,
    CheckpointCallback,
    EvalCallback,
)

__all__ = [
    "GRPOTrainer",
    "GRPOBatch",
    "RolloutGenerator",
    "AsyncRolloutGenerator",
    "ResearchDataset",
    "DataCollator",
    "TrainingCallback",
    "WandbCallback",
    "CheckpointCallback",
    "EvalCallback",
]
