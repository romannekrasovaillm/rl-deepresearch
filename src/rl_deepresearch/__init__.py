"""
RL Deep Research Agent
======================

A reinforcement learning framework for training deep research agents
using GRPO (Group Relative Policy Optimization) on GigaChat-Lightning.

Key Features:
- GRPO algorithm for agentic RL training
- Partial reward system with verifiable metrics
- Tool-augmented environment (search, read, analyze)
- B200 GPU optimization with LoRA and DeepSpeed
- Epistemic calibration ("I don't know" rewards)
- Efficiency bonuses for minimal search usage
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .config import ExperimentConfig, GRPOConfig, RewardConfig
from .training import GRPOTrainer
from .environment import ResearchEnvironment
from .agents import ResearchAgent

__all__ = [
    "ExperimentConfig",
    "GRPOConfig",
    "RewardConfig",
    "GRPOTrainer",
    "ResearchEnvironment",
    "ResearchAgent",
]
