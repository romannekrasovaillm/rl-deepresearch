"""
Rollout generation for RL training.

Key insights:
- Async rollouts hide tool latency
- Group generation for GRPO variance
- Staleness-aware for distributed training
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from typing import Any, Callable

import torch

from ..models.base import BaseModelWrapper, GenerationOutput
from ..environment.base import ResearchEnvironment, EnvironmentState
from ..rewards.compute import Trajectory, TrajectoryStep


@dataclass
class Rollout:
    """Single rollout from environment."""

    question: str
    trajectory: Trajectory
    generation_outputs: list[GenerationOutput]
    policy_version: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_stale(self, current_version: int, max_staleness: int = 3) -> bool:
        """Check if rollout is too stale for on-policy training."""
        return current_version - self.policy_version > max_staleness


class RolloutGenerator:
    """
    Synchronous rollout generator.

    Used for single-GPU training or debugging.
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        environment: ResearchEnvironment,
        max_turns: int = 10,
    ):
        self.model = model
        self.environment = environment
        self.max_turns = max_turns
        self.policy_version = 0

    def generate_single(
        self,
        question: str,
        reference_answer: str | None = None,
        reference_sources: list[str] | None = None,
        temperature: float = 0.7,
    ) -> Rollout:
        """Generate single rollout."""
        self.model.prepare_for_inference()

        state = self.environment.reset(
            question=question,
            reference_answer=reference_answer,
            reference_sources=reference_sources,
        )

        steps = []
        gen_outputs = []
        total_searches = 0
        total_reads = 0

        while not state.is_done and state.turn < self.max_turns:
            # Build prompt
            system = self.environment.get_system_prompt()
            user = self.environment.get_user_prompt()
            prompt = f"{system}\n\n{user}"

            # Generate
            output = self.model.generate(
                prompt=prompt,
                temperature=temperature,
                return_logprobs=True,
            )
            gen_outputs.append(output)

            # Execute
            result = self.environment.step(output.text)

            # Track stats
            tool_name = result.info.get("tool_name", "")
            if "search" in tool_name:
                total_searches += 1
            elif "read" in tool_name:
                total_reads += 1

            # Record step
            step = TrajectoryStep(
                turn_index=state.turn,
                action_type=tool_name,
                action_content=output.text,
                action_parsed=result.info.get("metadata", {}),
                observation=result.observation,
                is_terminal=result.done,
                metadata={
                    "token_ids": output.token_ids,
                    "num_tokens": output.num_tokens,
                },
            )
            steps.append(step)

            state = self.environment.state

        # Build trajectory
        traj_data = self.environment.get_full_trajectory()
        trajectory = Trajectory(
            question=question,
            reference_answer=reference_answer or "",
            reference_sources=reference_sources or [],
            steps=steps,
            final_answer=traj_data.get("final_answer"),
            cited_sources=traj_data.get("cited_sources", []),
            total_searches=total_searches,
            total_reads=total_reads,
        )

        return Rollout(
            question=question,
            trajectory=trajectory,
            generation_outputs=gen_outputs,
            policy_version=self.policy_version,
        )

    def generate_group(
        self,
        question: str,
        group_size: int,
        reference_answer: str | None = None,
        reference_sources: list[str] | None = None,
        temperature: float = 0.7,
    ) -> list[Rollout]:
        """Generate group of rollouts for GRPO."""
        return [
            self.generate_single(
                question=question,
                reference_answer=reference_answer,
                reference_sources=reference_sources,
                temperature=temperature,
            )
            for _ in range(group_size)
        ]

    def generate_batch(
        self,
        questions: list[dict],
        group_size: int,
        temperature: float = 0.7,
    ) -> list[list[Rollout]]:
        """Generate rollouts for batch of questions."""
        return [
            self.generate_group(
                question=q["question"],
                group_size=group_size,
                reference_answer=q.get("answer"),
                reference_sources=q.get("sources"),
                temperature=temperature,
            )
            for q in questions
        ]

    def update_policy_version(self) -> None:
        """Increment policy version after update."""
        self.policy_version += 1


class AsyncRolloutGenerator:
    """
    Asynchronous rollout generator.

    Key insight: tool latency is the bottleneck.
    Async generation hides this latency by parallelizing.

    Architecture:
    - Multiple worker threads generate rollouts
    - Queue for completed rollouts
    - Main thread consumes for training
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        environment_factory: Callable[[], ResearchEnvironment],
        num_workers: int = 4,
        queue_size: int = 32,
        max_turns: int = 10,
    ):
        self.model = model
        self.environment_factory = environment_factory
        self.num_workers = num_workers
        self.max_turns = max_turns

        # Queues
        self.task_queue: Queue = Queue(maxsize=queue_size * 2)
        self.result_queue: Queue = Queue(maxsize=queue_size)

        # Workers
        self.workers: list[Thread] = []
        self.running = False

        # Policy version for staleness tracking
        self.policy_version = 0

    def start(self) -> None:
        """Start worker threads."""
        self.running = True

        for i in range(self.num_workers):
            worker = Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

    def stop(self) -> None:
        """Stop worker threads."""
        self.running = False
        # Clear queues
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except:
                pass

    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread main loop."""
        # Each worker gets its own environment
        env = self.environment_factory()
        generator = RolloutGenerator(self.model, env, self.max_turns)

        while self.running:
            try:
                # Get task
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    continue

                # Generate rollout
                rollout = generator.generate_single(
                    question=task["question"],
                    reference_answer=task.get("answer"),
                    reference_sources=task.get("sources"),
                    temperature=task.get("temperature", 0.7),
                )

                # Update policy version
                rollout.policy_version = self.policy_version

                # Put result
                self.result_queue.put({
                    "task_id": task.get("id"),
                    "group_id": task.get("group_id"),
                    "rollout": rollout,
                })

            except Exception as e:
                if self.running:
                    print(f"Worker {worker_id} error: {e}")
                continue

    def submit_task(
        self,
        question: str,
        reference_answer: str | None = None,
        reference_sources: list[str] | None = None,
        task_id: str | None = None,
        group_id: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        """Submit task to workers."""
        self.task_queue.put({
            "question": question,
            "answer": reference_answer,
            "sources": reference_sources,
            "id": task_id,
            "group_id": group_id,
            "temperature": temperature,
        })

    def submit_group(
        self,
        question: str,
        group_size: int,
        reference_answer: str | None = None,
        reference_sources: list[str] | None = None,
        group_id: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        """Submit group of tasks."""
        for i in range(group_size):
            self.submit_task(
                question=question,
                reference_answer=reference_answer,
                reference_sources=reference_sources,
                task_id=f"{group_id}_{i}" if group_id else None,
                group_id=group_id,
                temperature=temperature,
            )

    def get_results(
        self,
        num_results: int,
        timeout: float = 60.0,
    ) -> list[dict]:
        """Get completed rollouts."""
        results = []
        start_time = time.time()

        while len(results) < num_results:
            try:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    break

                result = self.result_queue.get(timeout=min(remaining, 1.0))
                results.append(result)

            except:
                continue

        return results

    def get_group_results(
        self,
        group_id: str,
        group_size: int,
        timeout: float = 60.0,
    ) -> list[Rollout]:
        """Get all results for a group."""
        results = []
        start_time = time.time()

        while len(results) < group_size:
            try:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    break

                result = self.result_queue.get(timeout=min(remaining, 1.0))
                if result.get("group_id") == group_id:
                    results.append(result["rollout"])

            except:
                continue

        return results

    def update_policy_version(self) -> None:
        """Increment policy version after update."""
        self.policy_version += 1

    def filter_stale_rollouts(
        self,
        rollouts: list[Rollout],
        max_staleness: int = 3,
    ) -> list[Rollout]:
        """
        Filter out stale rollouts.

        Key insight (AReaL): async training can have stale data.
        Filter rollouts that are too many policy versions behind.
        """
        return [
            r for r in rollouts
            if self.policy_version - r.policy_version <= max_staleness
        ]


class BatchedRolloutGenerator:
    """
    Batched rollout generator for vLLM-style inference.

    Key insight: vLLM enables batched generation
    which is more efficient than sequential.
    """

    def __init__(
        self,
        model_name: str,
        environment_factory: Callable[[], ResearchEnvironment],
        max_batch_size: int = 32,
        max_turns: int = 10,
    ):
        self.model_name = model_name
        self.environment_factory = environment_factory
        self.max_batch_size = max_batch_size
        self.max_turns = max_turns

        self._llm = None
        self._sampling_params = None

    def _init_vllm(self, temperature: float = 0.7):
        """Initialize vLLM engine."""
        try:
            from vllm import LLM, SamplingParams

            if self._llm is None:
                self._llm = LLM(
                    model=self.model_name,
                    tensor_parallel_size=1,
                    trust_remote_code=True,
                )

            self._sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=2048,
            )
        except ImportError:
            raise ImportError("vLLM required for batched generation. Install with: pip install vllm")

    def generate_batch_step(
        self,
        prompts: list[str],
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate for batch of prompts."""
        self._init_vllm(temperature)

        outputs = self._llm.generate(prompts, self._sampling_params)
        return [o.outputs[0].text for o in outputs]

    def generate_batch(
        self,
        questions: list[dict],
        group_size: int,
        temperature: float = 0.7,
    ) -> list[list[Rollout]]:
        """
        Generate rollouts using batched inference.

        More efficient for large batches.
        """
        # Create environments for each (question, group_member) pair
        num_total = len(questions) * group_size
        environments = [self.environment_factory() for _ in range(num_total)]

        # Initialize all environments
        states = []
        for i, q in enumerate(questions):
            for j in range(group_size):
                idx = i * group_size + j
                state = environments[idx].reset(
                    question=q["question"],
                    reference_answer=q.get("answer"),
                    reference_sources=q.get("sources"),
                )
                states.append(state)

        # Track which envs are done
        done = [False] * num_total
        all_steps = [[] for _ in range(num_total)]
        all_gen_outputs = [[] for _ in range(num_total)]

        for turn in range(self.max_turns):
            # Collect prompts for active envs
            active_indices = [i for i, d in enumerate(done) if not d]
            if not active_indices:
                break

            prompts = []
            for idx in active_indices:
                system = environments[idx].get_system_prompt()
                user = environments[idx].get_user_prompt()
                prompts.append(f"{system}\n\n{user}")

            # Batch generate
            outputs = self.generate_batch_step(prompts, temperature)

            # Execute actions
            for prompt_idx, env_idx in enumerate(active_indices):
                action = outputs[prompt_idx]
                result = environments[env_idx].step(action)

                # Record step
                state = environments[env_idx].state
                step = TrajectoryStep(
                    turn_index=state.turn if state else turn,
                    action_type=result.info.get("tool_name", ""),
                    action_content=action,
                    action_parsed=result.info.get("metadata", {}),
                    observation=result.observation,
                    is_terminal=result.done,
                )
                all_steps[env_idx].append(step)

                if result.done:
                    done[env_idx] = True

        # Build rollouts
        all_rollouts = []
        for i, q in enumerate(questions):
            group_rollouts = []
            for j in range(group_size):
                idx = i * group_size + j
                traj_data = environments[idx].get_full_trajectory()

                trajectory = Trajectory(
                    question=q["question"],
                    reference_answer=q.get("answer", ""),
                    reference_sources=q.get("sources", []),
                    steps=all_steps[idx],
                    final_answer=traj_data.get("final_answer"),
                    cited_sources=traj_data.get("cited_sources", []),
                    total_searches=traj_data.get("total_searches", 0),
                    total_reads=traj_data.get("total_reads", 0),
                )

                rollout = Rollout(
                    question=q["question"],
                    trajectory=trajectory,
                    generation_outputs=[],  # vLLM doesn't return detailed outputs
                )
                group_rollouts.append(rollout)

            all_rollouts.append(group_rollouts)

        return all_rollouts
