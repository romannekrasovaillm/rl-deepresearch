"""
Main reward computation class.

Orchestrates all reward signals for RL training.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import RewardConfig

from .metrics import (
    compute_exact_match,
    compute_word_f1,
    compute_information_gain,
    check_format_validity,
    check_tool_call_validity,
    check_citation_correctness,
)
from .shaping import (
    ShapedReward,
    shape_terminal_reward,
    shape_progress_reward,
    shape_efficiency_reward,
    shape_format_penalty,
    compute_turn_level_advantages,
    aggregate_trajectory_reward,
)


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""

    turn_index: int
    action_type: str
    action_content: str
    action_parsed: dict[str, Any] | None
    observation: str
    is_terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete trajectory from question to answer."""

    question: str
    reference_answer: str
    reference_sources: list[str]
    steps: list[TrajectoryStep]
    final_answer: str | None
    cited_sources: list[str]
    total_searches: int = 0
    total_reads: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardComputer:
    """
    Compute rewards for research agent trajectories.

    Implements multi-faceted reward system based on insights:
    1. Verifiable metrics (EM, F1) for stability
    2. Format penalties as differentiable constraints
    3. Progress rewards as bootstrap signals
    4. Efficiency bonuses for anti-over-search
    5. "I don't know" rewards for calibration
    """

    def __init__(
        self,
        config: RewardConfig,
        available_tools: list[str] | None = None,
        tool_schemas: dict[str, dict] | None = None,
        embedder: Any | None = None,
    ):
        self.config = config
        self.available_tools = available_tools or [
            "keyword_search", "semantic_search", "read_document",
            "read_section", "answer", "dont_know"
        ]
        self.tool_schemas = tool_schemas
        self.embedder = embedder

    def compute_trajectory_reward(
        self,
        trajectory: Trajectory,
    ) -> dict[str, Any]:
        """
        Compute complete reward for a trajectory.

        Returns:
            dict with:
            - total_reward: float
            - turn_rewards: list[float]
            - turn_advantages: list[float]
            - component_breakdown: dict
            - metadata: dict
        """
        # Track accumulated knowledge for info gain
        accumulated_knowledge: list[str] = []

        # Collect rewards per step
        progress_rewards: list[ShapedReward] = []
        format_penalties: list[ShapedReward] = []

        # Process each step
        for step in trajectory.steps:
            if step.is_terminal:
                continue

            # Check format validity
            format_check = self._check_step_format(step)
            if not format_check["valid"]:
                format_penalty = shape_format_penalty(
                    format_check=format_check,
                    tool_check=None,
                    config=self.config,
                )
                format_penalties.append(format_penalty)
                progress_rewards.append(ShapedReward(total=0, components={}, metadata={}))
                continue

            # Check tool validity if tool call
            tool_check = None
            if step.action_parsed and "name" in step.action_parsed:
                tool_check = check_tool_call_validity(
                    step.action_parsed,
                    self.available_tools,
                    self.tool_schemas,
                )
                if not tool_check["valid"]:
                    format_penalty = shape_format_penalty(
                        format_check={"valid": True},
                        tool_check=tool_check,
                        config=self.config,
                    )
                    format_penalties.append(format_penalty)
                    progress_rewards.append(ShapedReward(total=0, components={}, metadata={}))
                    continue

            format_penalties.append(ShapedReward(total=0, components={}, metadata={}))

            # Compute progress reward for valid actions
            action_result = self._evaluate_action_result(
                step, accumulated_knowledge, trajectory
            )

            progress_reward = shape_progress_reward(
                action_type=step.action_type,
                action_result=action_result,
                existing_knowledge=accumulated_knowledge,
                config=self.config,
            )
            progress_rewards.append(progress_reward)

            # Update accumulated knowledge
            if step.observation and len(step.observation) > 50:
                accumulated_knowledge.append(step.observation)

        # Compute terminal reward
        is_dont_know = self._is_dont_know_response(trajectory.final_answer)

        answer_metrics = {}
        if trajectory.final_answer and not is_dont_know:
            answer_metrics["exact_match"] = compute_exact_match(
                trajectory.final_answer, trajectory.reference_answer
            )
            if self.config.use_word_f1:
                answer_metrics["word_f1"] = compute_word_f1(
                    trajectory.final_answer, trajectory.reference_answer
                )

        # Citation metrics
        citation_metrics = {}
        if trajectory.cited_sources and trajectory.reference_sources:
            citation_metrics = check_citation_correctness(
                cited_sources=trajectory.cited_sources,
                actual_sources=trajectory.reference_sources,
                answer=trajectory.final_answer or "",
                evidence=[{"content": k} for k in accumulated_knowledge],
            )

        terminal_reward = shape_terminal_reward(
            prediction=trajectory.final_answer or "",
            reference=trajectory.reference_answer,
            is_dont_know=is_dont_know,
            answer_metrics=answer_metrics,
            citation_metrics=citation_metrics,
            config=self.config,
        )

        # Compute efficiency reward
        is_correct = answer_metrics.get("exact_match", 0) == 1.0 or answer_metrics.get("word_f1", 0) > 0.8
        efficiency_reward = shape_efficiency_reward(
            num_searches=trajectory.total_searches,
            num_turns=len(trajectory.steps),
            is_correct=is_correct,
            config=self.config,
        )

        # Aggregate all rewards
        aggregated = aggregate_trajectory_reward(
            progress_rewards=progress_rewards,
            terminal_reward=terminal_reward,
            efficiency_reward=efficiency_reward,
            format_penalties=format_penalties,
        )

        # Compute turn-level advantages for MT-GRPO
        turn_rewards = [ShapedReward(
            total=p.total + f.total,
            components={**p.components, **f.components},
            metadata={}
        ) for p, f in zip(progress_rewards, format_penalties)]

        turn_advantages = compute_turn_level_advantages(
            turn_rewards=turn_rewards,
            final_reward=ShapedReward(
                total=terminal_reward.total + efficiency_reward.total,
                components={**terminal_reward.components, **efficiency_reward.components},
                metadata={}
            ),
            gamma=0.99,
            immediate_weight=0.3,
        )

        return {
            "total_reward": aggregated["total_reward"],
            "turn_rewards": aggregated["per_turn_rewards"],
            "turn_advantages": turn_advantages,
            "component_breakdown": aggregated["component_breakdown"],
            "answer_metrics": answer_metrics,
            "citation_metrics": citation_metrics,
            "metadata": {
                **aggregated["metadata"],
                "is_dont_know": is_dont_know,
                "is_correct": is_correct,
                "num_turns": len(trajectory.steps),
                "num_searches": trajectory.total_searches,
            }
        }

    def compute_group_rewards(
        self,
        trajectories: list[Trajectory],
    ) -> dict[str, Any]:
        """
        Compute rewards for a group of trajectories (GRPO).

        Key insight: GRPO evaluates trajectories relative to group,
        not absolute. This creates evolutionary pressure without
        external critic.

        Returns:
            dict with:
            - individual_rewards: list[dict]
            - group_mean: float
            - group_std: float
            - normalized_advantages: list[float]
            - has_sufficient_variance: bool
        """
        # Compute individual rewards
        individual_rewards = [
            self.compute_trajectory_reward(traj)
            for traj in trajectories
        ]

        # Extract total rewards
        totals = [r["total_reward"] for r in individual_rewards]

        # Group statistics
        import numpy as np
        group_mean = np.mean(totals)
        group_std = np.std(totals)

        # Check variance (GRPO needs variance for learning)
        has_sufficient_variance = group_std > 0.01

        # Normalize advantages relative to group (GRPO style)
        if group_std > 0:
            normalized_advantages = [(r - group_mean) / (group_std + 1e-8) for r in totals]
        else:
            normalized_advantages = [0.0] * len(totals)

        return {
            "individual_rewards": individual_rewards,
            "group_mean": float(group_mean),
            "group_std": float(group_std),
            "normalized_advantages": normalized_advantages,
            "has_sufficient_variance": has_sufficient_variance,
            "raw_totals": totals,
        }

    def _check_step_format(self, step: TrajectoryStep) -> dict[str, Any]:
        """Check format validity of a step."""
        if step.action_type in ["answer", "dont_know"]:
            return check_format_validity(step.action_content, "answer")
        else:
            return check_format_validity(step.action_content, "tool_call")

    def _is_dont_know_response(self, answer: str | None) -> bool:
        """Check if answer is "I don't know" type response."""
        if not answer:
            return False

        dont_know_patterns = [
            r"i don't know",
            r"i do not know",
            r"cannot determine",
            r"unable to find",
            r"not enough information",
            r"insufficient evidence",
            r"unclear from",
            r"<dont_know>",
            r"\[dont_know\]",
        ]

        answer_lower = answer.lower()
        return any(re.search(p, answer_lower) for p in dont_know_patterns)

    def _evaluate_action_result(
        self,
        step: TrajectoryStep,
        accumulated_knowledge: list[str],
        trajectory: Trajectory,
    ) -> dict[str, Any]:
        """Evaluate the result of an action."""
        result = {
            "found_relevant": False,
            "is_useful": False,
            "is_duplicate": False,
            "information_gain": 0.0,
            "num_results": 0,
        }

        if step.action_type in ["keyword_search", "semantic_search"]:
            # Check if search returned results
            if step.observation and len(step.observation) > 10:
                result["num_results"] = step.metadata.get("num_results", 1)

                # Check relevance to question
                question_tokens = set(trajectory.question.lower().split())
                obs_tokens = set(step.observation.lower().split())
                overlap = len(question_tokens & obs_tokens)
                result["found_relevant"] = overlap >= 2

        elif step.action_type in ["read_document", "read_section"]:
            if step.observation:
                # Check if content is useful (relates to question/answer)
                answer_tokens = set(trajectory.reference_answer.lower().split())
                obs_tokens = set(step.observation.lower().split())
                overlap = len(answer_tokens & obs_tokens)
                result["is_useful"] = overlap >= 1

                # Compute information gain
                if accumulated_knowledge:
                    result["information_gain"] = compute_information_gain(
                        step.observation,
                        accumulated_knowledge,
                        self.embedder,
                    )
                else:
                    result["information_gain"] = 1.0

                # Check for duplicate
                for prev in accumulated_knowledge:
                    if compute_word_f1(step.observation[:500], prev[:500]) > 0.8:
                        result["is_duplicate"] = True
                        break

        return result


def create_reward_computer(
    config: RewardConfig | None = None,
    **kwargs,
) -> RewardComputer:
    """Factory function to create RewardComputer with defaults."""
    if config is None:
        config = RewardConfig()

    return RewardComputer(config=config, **kwargs)
