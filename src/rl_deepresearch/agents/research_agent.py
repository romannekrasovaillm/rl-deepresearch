"""
High-level research agent for inference.
"""

import json
from typing import Any

from ..models.base import BaseModelWrapper
from ..environment.base import ResearchEnvironment, EnvironmentState
from ..environment.tools import ToolResult
from .prompts import PromptBuilder


class ResearchAgent:
    """
    High-level research agent for inference.

    Wraps model + environment for easy question answering.
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        environment: ResearchEnvironment,
        max_turns: int = 10,
        verbose: bool = False,
    ):
        self.model = model
        self.environment = environment
        self.max_turns = max_turns
        self.verbose = verbose
        self.prompt_builder = PromptBuilder()

    def answer(
        self,
        question: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Answer a research question.

        Returns:
            dict with:
            - answer: str
            - sources: list[str]
            - reasoning: list[dict] (steps taken)
            - confidence: str ("high", "medium", "low", "unknown")
        """
        self.model.prepare_for_inference()

        state = self.environment.reset(question=question)
        reasoning_steps = []

        while not state.is_done and state.turn < self.max_turns:
            # Build prompt
            system = self.environment.get_system_prompt()
            user = self.environment.get_user_prompt()
            prompt = f"{system}\n\n{user}"

            # Generate action
            output = self.model.generate(
                prompt=prompt,
                temperature=temperature,
            )

            # Execute
            result = self.environment.step(output.text)

            # Record step
            step = {
                "turn": state.turn,
                "action": output.text,
                "tool": result.info.get("tool_name", "unknown"),
                "success": result.info.get("success", False),
            }

            if self.verbose:
                print(f"\n--- Turn {state.turn} ---")
                print(f"Action: {output.text[:200]}...")
                print(f"Result: {result.observation[:200]}...")

            reasoning_steps.append(step)
            state = self.environment.state

        # Extract final answer
        traj_data = self.environment.get_full_trajectory()

        # Determine confidence
        confidence = self._assess_confidence(reasoning_steps, traj_data)

        return {
            "answer": traj_data.get("final_answer", "Unable to determine answer"),
            "sources": traj_data.get("cited_sources", []),
            "reasoning": reasoning_steps,
            "confidence": confidence,
            "is_dont_know": traj_data.get("is_dont_know", False),
            "num_searches": traj_data.get("total_searches", 0),
            "num_reads": traj_data.get("total_reads", 0),
        }

    def _assess_confidence(
        self,
        steps: list[dict],
        traj_data: dict,
    ) -> str:
        """Assess confidence in answer based on trajectory."""
        if traj_data.get("is_dont_know"):
            return "unknown"

        num_sources = len(traj_data.get("cited_sources", []))
        num_reads = traj_data.get("total_reads", 0)

        if num_sources >= 2 and num_reads >= 2:
            return "high"
        elif num_sources >= 1 or num_reads >= 1:
            return "medium"
        else:
            return "low"

    def batch_answer(
        self,
        questions: list[str],
        temperature: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Answer multiple questions."""
        return [self.answer(q, temperature) for q in questions]

    def interactive(self) -> None:
        """Interactive Q&A mode."""
        print("Research Agent Interactive Mode")
        print("Type 'quit' to exit, 'verbose' to toggle verbose mode")
        print("-" * 50)

        while True:
            try:
                question = input("\nQuestion: ").strip()

                if question.lower() == "quit":
                    break
                elif question.lower() == "verbose":
                    self.verbose = not self.verbose
                    print(f"Verbose mode: {self.verbose}")
                    continue
                elif not question:
                    continue

                print("\nSearching...")
                result = self.answer(question)

                print(f"\nAnswer: {result['answer']}")
                print(f"Confidence: {result['confidence']}")
                if result['sources']:
                    print(f"Sources: {', '.join(result['sources'])}")
                print(f"Steps taken: {len(result['reasoning'])}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        corpus_path: str | None = None,
        **kwargs,
    ) -> "ResearchAgent":
        """Load agent from checkpoint."""
        from ..models.gigachat import GigaChatLocalWrapper
        from ..config import ModelConfig, EnvironmentConfig
        from ..environment.corpus import DocumentCorpus

        # Load model
        config = ModelConfig()
        model = GigaChatLocalWrapper(config)
        model.load_checkpoint(checkpoint_path)

        # Load corpus
        if corpus_path:
            corpus = DocumentCorpus.load(corpus_path)
        else:
            corpus = DocumentCorpus.from_hotpotqa(max_docs=1000)

        # Create environment
        env_config = EnvironmentConfig()
        environment = ResearchEnvironment(corpus, env_config)

        return cls(model=model, environment=environment, **kwargs)
