"""
Training module for RL Deep Research.

Implements:
- GRPO (Group Relative Policy Optimization)
- Async rollout generation
- B200-optimized training loop
- Curriculum learning
- Comprehensive RL logging and monitoring
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
from .logging import (
    RLMetricsTracker,
    RLLoggingCallback,
    TensorBoardRLLogger,
    AlertLevel,
    Alert,
    create_rl_logging_stack,
)

__all__ = [
    # Core training
    "GRPOTrainer",
    "GRPOBatch",
    "RolloutGenerator",
    "AsyncRolloutGenerator",
    "ResearchDataset",
    "DataCollator",
    # Callbacks
    "TrainingCallback",
    "WandbCallback",
    "CheckpointCallback",
    "EvalCallback",
    # Logging
    "RLMetricsTracker",
    "RLLoggingCallback",
    "TensorBoardRLLogger",
    "AlertLevel",
    "Alert",
    "create_rl_logging_stack",
]
