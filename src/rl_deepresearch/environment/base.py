"""
Research environment for RL training.

Key insights implemented:
- Structured action space (JSON tool calls)
- State compression (MEM1 style)
- Max turns as exploration budget
- Hierarchical document navigation
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import EnvironmentConfig
from .corpus import DocumentCorpus
from .tools import ToolRegistry, ToolResult, create_default_registry


@dataclass
class EnvironmentState:
    """
    Compressed state representation.

    Key insight (MEM1): long history degrades learning signal.
    Compress to essential state: question, found docs, current knowledge.
    """

    question: str
    turn: int = 0
    max_turns: int = 10

    # Compressed history
    searched_queries: list[str] = field(default_factory=list)
    found_doc_ids: list[str] = field(default_factory=list)
    read_doc_ids: list[str] = field(default_factory=list)
    knowledge_notes: list[str] = field(default_factory=list)

    # Current context (last observation)
    last_action: str | None = None
    last_observation: str | None = None

    # Terminal state
    is_done: bool = False
    final_answer: str | None = None
    cited_sources: list[str] = field(default_factory=list)
    is_dont_know: bool = False

    def to_prompt_context(self, max_tokens: int = 4096) -> str:
        """
        Convert state to prompt context.

        Implements state compression - only essential info.
        """
        lines = [
            f"Question: {self.question}",
            f"\nTurn: {self.turn}/{self.max_turns}",
        ]

        if self.searched_queries:
            lines.append(f"\nSearched: {', '.join(self.searched_queries[-5:])}")

        if self.found_doc_ids:
            lines.append(f"Found documents: {', '.join(self.found_doc_ids[-10:])}")

        if self.read_doc_ids:
            lines.append(f"Read documents: {', '.join(self.read_doc_ids[-5:])}")

        if self.knowledge_notes:
            lines.append("\nKey findings:")
            for note in self.knowledge_notes[-5:]:
                lines.append(f"  - {note[:200]}")

        if self.last_observation:
            # Truncate observation if needed
            obs = self.last_observation
            if len(obs) > 2000:
                obs = obs[:2000] + "\n[TRUNCATED]"
            lines.append(f"\nLast observation:\n{obs}")

        return "\n".join(lines)

    def add_knowledge(self, content: str, max_notes: int = 10) -> None:
        """Add to compressed knowledge (with limit)."""
        # Extract key sentence
        sentences = content.split(".")
        if sentences:
            key_sentence = max(sentences, key=len)[:300]
            if key_sentence not in self.knowledge_notes:
                self.knowledge_notes.append(key_sentence)
                if len(self.knowledge_notes) > max_notes:
                    self.knowledge_notes = self.knowledge_notes[-max_notes:]


@dataclass
class StepResult:
    """Result of environment step."""

    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


class ResearchEnvironment:
    """
    Tool-augmented research environment.

    Provides:
    - Structured action execution
    - State management with compression
    - Turn limits (exploration budget)
    - Automatic trajectory tracking
    """

    def __init__(
        self,
        corpus: DocumentCorpus,
        config: EnvironmentConfig | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.corpus = corpus
        self.config = config or EnvironmentConfig()
        self.tools = tool_registry or create_default_registry(corpus)

        self.state: EnvironmentState | None = None
        self.trajectory: list[dict] = []

        # Reference data for reward computation
        self.reference_answer: str | None = None
        self.reference_sources: list[str] | None = None

    def reset(
        self,
        question: str,
        reference_answer: str | None = None,
        reference_sources: list[str] | None = None,
    ) -> EnvironmentState:
        """
        Reset environment for new episode.

        Args:
            question: The research question
            reference_answer: Ground truth answer (for reward)
            reference_sources: Ground truth source documents

        Returns:
            Initial state
        """
        self.state = EnvironmentState(
            question=question,
            max_turns=self.config.max_turns,
        )
        self.trajectory = []
        self.reference_answer = reference_answer
        self.reference_sources = reference_sources

        return self.state

    def step(self, action: str | dict) -> StepResult:
        """
        Execute action and return result.

        Args:
            action: Either JSON string or dict with tool call

        Returns:
            StepResult with observation, reward, done, info
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        if self.state.is_done:
            return StepResult(
                observation="Episode already finished.",
                reward=0.0,
                done=True,
                info={"error": "episode_finished"},
            )

        # Increment turn
        self.state.turn += 1

        # Parse action
        parsed_action = self._parse_action(action)
        if parsed_action is None:
            return StepResult(
                observation="Failed to parse action. Use format: {\"name\": \"tool_name\", \"arguments\": {...}}",
                reward=-2.0,  # Format error penalty
                done=False,
                info={"error": "parse_error", "raw_action": str(action)},
            )

        tool_name = parsed_action.get("name", "")
        tool_args = parsed_action.get("arguments", {})

        # Execute tool
        result = self.tools.execute(tool_name, **tool_args)

        # Update state
        self._update_state(tool_name, tool_args, result)

        # Check if terminal
        is_terminal = result.metadata.get("is_terminal", False)
        if is_terminal or self.state.turn >= self.state.max_turns:
            self.state.is_done = True
            if result.metadata.get("is_dont_know"):
                self.state.is_dont_know = True
            elif "answer" in result.metadata:
                self.state.final_answer = result.metadata["answer"]
                self.state.cited_sources = result.metadata.get("sources", [])

        # Record trajectory step
        self.trajectory.append({
            "turn": self.state.turn,
            "action": parsed_action,
            "observation": result.output,
            "is_terminal": is_terminal,
            "metadata": result.metadata,
        })

        return StepResult(
            observation=result.output if result.success else f"Error: {result.error}",
            reward=0.0,  # Rewards computed separately
            done=self.state.is_done,
            info={
                "success": result.success,
                "tool_name": tool_name,
                "metadata": result.metadata,
            },
        )

    def _parse_action(self, action: str | dict) -> dict | None:
        """Parse action into structured format."""
        if isinstance(action, dict):
            return action

        # Try to extract JSON from string
        try:
            # Direct JSON parse
            return json.loads(action)
        except json.JSONDecodeError:
            pass

        # Try to extract from code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*"name"[^{}]*\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, action, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if '```' in pattern else match.group(0))
                except json.JSONDecodeError:
                    continue

        return None

    def _update_state(
        self,
        tool_name: str,
        tool_args: dict,
        result: ToolResult
    ) -> None:
        """Update state based on action result."""
        self.state.last_action = tool_name
        self.state.last_observation = result.output

        if tool_name in ["keyword_search", "semantic_search"]:
            query = tool_args.get("query", "")
            if query:
                self.state.searched_queries.append(query)
            if result.success:
                doc_ids = result.metadata.get("doc_ids", [])
                for doc_id in doc_ids:
                    if doc_id not in self.state.found_doc_ids:
                        self.state.found_doc_ids.append(doc_id)

        elif tool_name in ["read_document", "read_section"]:
            doc_id = result.metadata.get("doc_id", "")
            if doc_id and doc_id not in self.state.read_doc_ids:
                self.state.read_doc_ids.append(doc_id)

            # Compress into knowledge
            if result.success and self.config.compress_history:
                self.state.add_knowledge(result.output)

    def get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        tool_descriptions = self.tools.get_tool_descriptions()

        return f"""You are a research assistant that answers questions by searching and reading documents.

{tool_descriptions}

Instructions:
1. Search for relevant documents using keyword_search or semantic_search
2. Read promising documents to find evidence
3. When you have enough evidence, use the answer tool with citations
4. If you cannot find enough evidence, use dont_know instead of guessing

Output your action as JSON:
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}

Think step by step, but be efficient - avoid redundant searches."""

    def get_user_prompt(self) -> str:
        """Get user prompt with current state."""
        if self.state is None:
            raise RuntimeError("Environment not reset.")

        return self.state.to_prompt_context()

    def get_full_trajectory(self) -> dict:
        """Get complete trajectory for reward computation."""
        return {
            "question": self.state.question if self.state else "",
            "reference_answer": self.reference_answer,
            "reference_sources": self.reference_sources,
            "steps": self.trajectory,
            "final_answer": self.state.final_answer if self.state else None,
            "cited_sources": self.state.cited_sources if self.state else [],
            "is_dont_know": self.state.is_dont_know if self.state else False,
            "total_turns": self.state.turn if self.state else 0,
            "total_searches": len(self.state.searched_queries) if self.state else 0,
            "total_reads": len(self.state.read_doc_ids) if self.state else 0,
        }

    @property
    def action_space(self) -> list[str]:
        """Get list of available actions."""
        return self.tools.tool_names

    @property
    def observation_space(self) -> dict:
        """Get observation space description."""
        return {
            "type": "text",
            "max_length": self.config.max_read_tokens,
            "format": "natural_language",
        }


class VectorizedEnvironment:
    """
    Vectorized environment for parallel rollouts.

    Key insight: tool latency is bottleneck.
    Parallel environments hide latency.
    """

    def __init__(
        self,
        corpus: DocumentCorpus,
        num_envs: int,
        config: EnvironmentConfig | None = None,
    ):
        self.envs = [
            ResearchEnvironment(corpus, config)
            for _ in range(num_envs)
        ]
        self.num_envs = num_envs

    def reset(
        self,
        questions: list[str],
        reference_answers: list[str] | None = None,
        reference_sources: list[list[str]] | None = None,
    ) -> list[EnvironmentState]:
        """Reset all environments."""
        states = []
        for i, env in enumerate(self.envs):
            q = questions[i] if i < len(questions) else questions[-1]
            ref_ans = reference_answers[i] if reference_answers and i < len(reference_answers) else None
            ref_src = reference_sources[i] if reference_sources and i < len(reference_sources) else None
            states.append(env.reset(q, ref_ans, ref_src))
        return states

    def step(self, actions: list[str | dict]) -> list[StepResult]:
        """Step all environments."""
        results = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            results.append(env.step(action))
        return results

    def get_trajectories(self) -> list[dict]:
        """Get all trajectories."""
        return [env.get_full_trajectory() for env in self.envs]

    @property
    def dones(self) -> list[bool]:
        """Check which environments are done."""
        return [env.state.is_done if env.state else True for env in self.envs]
