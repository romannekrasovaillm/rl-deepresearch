"""
Reward shaping functions.

Key insights from research:
- Dense positive signals for progress, sparse negative for format violations
- Efficiency bonuses create emergent early stopping
- "I don't know" rewards calibrate epistemic humility
"""

from dataclasses import dataclass
from typing import Any

from ..config import RewardConfig


@dataclass
class ShapedReward:
    """Container for shaped reward with breakdown."""

    total: float
    components: dict[str, float]
    metadata: dict[str, Any]


def shape_terminal_reward(
    prediction: str,
    reference: str,
    is_dont_know: bool,
    answer_metrics: dict[str, float],
    citation_metrics: dict[str, Any],
    config: RewardConfig,
) -> ShapedReward:
    """
    Shape terminal (final answer) reward.

    Reward structure (from insights):
    - Correct answer: +2.0
    - Correct + efficient: +2.5
    - Partial correct: +0.5 to +1.5 (based on F1)
    - Wrong answer: -1.0
    - "I don't know": +0.3 (epistemic calibration)

    Key insight: "I don't know" > wrong answer
    This creates internal barrier against hallucination.
    """
    components = {}
    metadata = {}

    # Check for "I don't know" response
    if is_dont_know:
        components["dont_know"] = config.dont_know_reward
        return ShapedReward(
            total=config.dont_know_reward,
            components=components,
            metadata={"response_type": "dont_know"}
        )

    # Get metrics
    exact_match = answer_metrics.get("exact_match", 0.0)
    word_f1 = answer_metrics.get("word_f1", 0.0)

    # Correct answer
    if exact_match == 1.0:
        components["correct_answer"] = config.correct_answer
        metadata["response_type"] = "correct"

    # Partial correct (use F1 for soft matching)
    elif word_f1 > 0.3:
        # Scale partial reward by F1
        partial_reward = config.partial_correct_base + (config.correct_answer - config.partial_correct_base) * word_f1
        components["partial_correct"] = partial_reward
        metadata["response_type"] = "partial_correct"
        metadata["f1_score"] = word_f1

    # Wrong answer
    else:
        components["wrong_answer"] = config.wrong_answer
        metadata["response_type"] = "wrong"

    # Citation bonus/penalty
    if citation_metrics:
        citation_precision = citation_metrics.get("precision", 0.0)
        if citation_precision > 0.8:
            components["citation_correct"] = config.citation_correct_bonus
        elif citation_precision < 0.3 and citation_metrics.get("total_citations", 0) > 0:
            components["citation_wrong"] = config.citation_wrong_penalty

    total = sum(components.values())

    return ShapedReward(
        total=total,
        components=components,
        metadata=metadata
    )


def shape_progress_reward(
    action_type: str,
    action_result: dict[str, Any],
    existing_knowledge: list[str],
    config: RewardConfig,
) -> ShapedReward:
    """
    Shape progress (intermediate) reward.

    Key insight: progress rewards as bootstrap mechanism.
    Without them, model gets no signal until final answer.
    But be careful - dense rewards can cause reward hacking.

    Reward types:
    - Found relevant document: +0.3
    - Read useful content: +0.2
    - Information gain: +0.1 per unit
    """
    components = {}
    metadata = {"action_type": action_type}

    if action_type in ["keyword_search", "semantic_search"]:
        # Check if search found relevant documents
        found_relevant = action_result.get("found_relevant", False)
        num_results = action_result.get("num_results", 0)

        if found_relevant:
            components["found_relevant_doc"] = config.found_relevant_doc
            metadata["num_relevant"] = action_result.get("num_relevant", 0)

        # Penalty for empty search (indicates bad query)
        if num_results == 0:
            components["empty_search"] = -0.1

    elif action_type in ["read_document", "read_section"]:
        # Check if read content is useful
        content = action_result.get("content", "")
        is_useful = action_result.get("is_useful", False)

        if is_useful:
            components["read_useful"] = config.read_useful_content

            # Information gain (anti-redundancy)
            info_gain = action_result.get("information_gain", 0.0)
            if info_gain > 0:
                components["information_gain"] = config.information_gain * info_gain
                metadata["info_gain_raw"] = info_gain

        # Penalty for reading same content again
        if action_result.get("is_duplicate", False):
            components["redundant_read"] = config.redundant_search_penalty

    total = sum(components.values())

    return ShapedReward(
        total=total,
        components=components,
        metadata=metadata
    )


def shape_efficiency_reward(
    num_searches: int,
    num_turns: int,
    is_correct: bool,
    config: RewardConfig,
) -> ShapedReward:
    """
    Shape efficiency reward/penalty.

    Key insight: efficiency bonuses create emergent stopping behavior.
    Without this, agents use all available turns "just in case".

    - Correct with few searches: +0.5 bonus
    - Too many turns: -0.05 per turn over threshold
    """
    components = {}
    metadata = {
        "num_searches": num_searches,
        "num_turns": num_turns,
        "is_correct": is_correct,
    }

    # Efficiency bonus for correct answers with minimal search
    if is_correct and num_searches <= config.efficient_search_threshold:
        components["efficient_search_bonus"] = config.correct_efficient_bonus
        metadata["earned_efficiency_bonus"] = True

    # Penalty for excessive turns
    if num_turns > config.max_turns_before_penalty:
        excess_turns = num_turns - config.max_turns_before_penalty
        penalty = excess_turns * config.excessive_turns_penalty
        components["excessive_turns"] = penalty
        metadata["excess_turns"] = excess_turns

    total = sum(components.values())

    return ShapedReward(
        total=total,
        components=components,
        metadata=metadata
    )


def shape_format_penalty(
    format_check: dict[str, Any],
    tool_check: dict[str, Any] | None,
    config: RewardConfig,
) -> ShapedReward:
    """
    Shape format-related penalties.

    Key insight: format rewards as differentiable constraints.
    Format errors are worse than wrong answers because they break the pipeline.

    Penalty structure:
    - Unparseable output: -2.0
    - Bad tool name: -1.5
    - Bad tool args: -1.0
    """
    components = {}
    metadata = {}

    # Check format validity
    if not format_check.get("valid", True):
        error_type = format_check.get("error_type", "unknown")
        metadata["format_error_type"] = error_type

        if error_type in ["json_parse_error", "no_tool_found", "no_answer_found"]:
            components["format_error"] = config.format_error

    # Check tool call validity
    if tool_check and not tool_check.get("valid", True):
        error_type = tool_check.get("error_type", "unknown")
        metadata["tool_error_type"] = error_type

        if error_type == "bad_tool_name":
            components["bad_tool_name"] = config.bad_tool_name
        elif error_type == "bad_tool_args":
            components["bad_tool_args"] = config.bad_tool_args
        elif error_type == "missing_tool_name":
            components["format_error"] = config.format_error

    total = sum(components.values())

    return ShapedReward(
        total=total,
        components=components,
        metadata=metadata
    )


def compute_turn_level_advantages(
    turn_rewards: list[ShapedReward],
    final_reward: ShapedReward,
    gamma: float = 0.99,
    immediate_weight: float = 0.3,
) -> list[float]:
    """
    Compute per-turn advantages (MT-GRPO style).

    Key insight: turn-level credit assignment for temporal structure.
    Early turns get credit from future success, late turns from immediate outcomes.

    Args:
        turn_rewards: List of per-turn shaped rewards
        final_reward: Terminal reward
        gamma: Discount factor
        immediate_weight: Weight for immediate vs final rewards

    Returns:
        List of per-turn advantage values
    """
    num_turns = len(turn_rewards)
    if num_turns == 0:
        return []

    advantages = []
    final_value = final_reward.total

    # Backward pass to compute advantages
    running_return = final_value

    for t in range(num_turns - 1, -1, -1):
        immediate_reward = turn_rewards[t].total

        # Mixed advantage: combine immediate and future
        advantage = immediate_weight * immediate_reward + (1 - immediate_weight) * running_return

        advantages.insert(0, advantage)

        # Update running return with discount
        running_return = gamma * running_return + immediate_reward

    return advantages


def aggregate_trajectory_reward(
    progress_rewards: list[ShapedReward],
    terminal_reward: ShapedReward,
    efficiency_reward: ShapedReward,
    format_penalties: list[ShapedReward],
) -> dict[str, Any]:
    """
    Aggregate all rewards for a complete trajectory.

    Returns:
        dict with:
        - total_reward: float
        - component_breakdown: dict
        - per_turn_rewards: list
        - metadata: dict
    """
    # Sum all progress rewards
    progress_total = sum(r.total for r in progress_rewards)
    progress_components = {}
    for r in progress_rewards:
        for k, v in r.components.items():
            progress_components[k] = progress_components.get(k, 0) + v

    # Sum format penalties
    format_total = sum(r.total for r in format_penalties)
    format_components = {}
    for r in format_penalties:
        for k, v in r.components.items():
            format_components[k] = format_components.get(k, 0) + v

    # Combine all components
    all_components = {
        **progress_components,
        **terminal_reward.components,
        **efficiency_reward.components,
        **format_components,
    }

    total_reward = progress_total + terminal_reward.total + efficiency_reward.total + format_total

    return {
        "total_reward": total_reward,
        "component_breakdown": all_components,
        "progress_total": progress_total,
        "terminal_total": terminal_reward.total,
        "efficiency_total": efficiency_reward.total,
        "format_penalty_total": format_total,
        "per_turn_rewards": [r.total for r in progress_rewards],
        "metadata": {
            "terminal": terminal_reward.metadata,
            "efficiency": efficiency_reward.metadata,
        }
    }
