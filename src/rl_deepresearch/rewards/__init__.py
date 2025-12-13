"""
Reward computation module for RL Deep Research.

Implements multi-faceted reward system based on research insights:
- Verifiable rewards (EM, F1, document match) for stability
- Format rewards as differentiable constraints
- Efficiency bonuses as anti-over-search mechanism
- "I don't know" rewards for epistemic calibration
- Progress rewards as bootstrap mechanism
"""

from .compute import RewardComputer
from .metrics import (
    compute_exact_match,
    compute_word_f1,
    compute_rouge,
    compute_information_gain,
    check_format_validity,
    check_citation_correctness,
)
from .shaping import (
    shape_terminal_reward,
    shape_progress_reward,
    shape_efficiency_reward,
)

__all__ = [
    "RewardComputer",
    "compute_exact_match",
    "compute_word_f1",
    "compute_rouge",
    "compute_information_gain",
    "check_format_validity",
    "check_citation_correctness",
    "shape_terminal_reward",
    "shape_progress_reward",
    "shape_efficiency_reward",
]
