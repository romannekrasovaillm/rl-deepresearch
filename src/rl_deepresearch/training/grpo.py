"""
GRPO (Group Relative Policy Optimization) trainer.

Key insights implemented:
- Group-relative advantages (no external critic needed)
- Turn-level credit assignment (MT-GRPO style)
- Tool output masking (implicit credit assignment)
- KL penalty for stability
- Efficiency bonuses for emergent stopping
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, GRPOConfig
from ..models.base import BaseModelWrapper
from ..rewards.compute import RewardComputer, Trajectory, TrajectoryStep
from ..environment.base import ResearchEnvironment


@dataclass
class GRPOBatch:
    """Batch of trajectories for GRPO training."""

    question_ids: list[str]
    trajectories: list[list[Trajectory]]  # [num_questions, group_size]
    rewards: list[list[float]]  # [num_questions, group_size]
    advantages: list[list[float]]  # Normalized per-group
    turn_advantages: list[list[list[float]]]  # Per-turn advantages

    # Tokenized data for training
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    action_masks: torch.Tensor | None = None  # Mask for action tokens only
    turn_masks: list[torch.Tensor] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


class GRPOTrainer:
    """
    GRPO trainer for research agents.

    Implements:
    1. Group sampling: generate multiple trajectories per question
    2. Group-relative advantage: compare within group, not absolute
    3. Turn-level credit assignment: MT-GRPO style advantages
    4. Tool masking: gradients only on action tokens
    5. KL penalty: anchor to reference policy
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        config: ExperimentConfig,
        reward_computer: RewardComputer,
        environment: ResearchEnvironment,
        reference_model: BaseModelWrapper | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.model = model
        self.config = config
        self.grpo_config = config.grpo
        self.reward_computer = reward_computer
        self.environment = environment
        self.reference_model = reference_model

        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float("-inf")

        # Metrics
        self.metrics_history = []

        # Device
        self.device = next(iter(model.get_trainable_parameters())).device

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay handling."""
        from ..models.utils import prepare_optimizer_groups

        param_groups = prepare_optimizer_groups(
            self.model.model if hasattr(self.model, "model") else self.model,
            weight_decay=self.grpo_config.weight_decay,
        )

        return torch.optim.AdamW(
            param_groups,
            lr=self.grpo_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warmup."""
        from transformers import get_linear_schedule_with_warmup

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.grpo_config.warmup_steps,
            num_training_steps=self.config.training.max_steps,
        )

    def generate_rollouts(
        self,
        questions: list[dict],
        group_size: int | None = None,
    ) -> list[list[Trajectory]]:
        """
        Generate group of rollouts for each question.

        Key insight: GRPO needs variance within group.
        Generate multiple trajectories with temperature sampling.
        """
        group_size = group_size or self.grpo_config.group_size
        all_trajectories = []

        self.model.prepare_for_inference()

        for question_data in questions:
            question = question_data["question"]
            reference_answer = question_data.get("answer")
            reference_sources = question_data.get("sources", [])

            group_trajectories = []

            for _ in range(group_size):
                # Reset environment
                state = self.environment.reset(
                    question=question,
                    reference_answer=reference_answer,
                    reference_sources=reference_sources,
                )

                trajectory_steps = []
                total_searches = 0
                total_reads = 0

                # Run episode
                while not state.is_done:
                    # Generate action
                    system_prompt = self.environment.get_system_prompt()
                    user_prompt = self.environment.get_user_prompt()
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"

                    output = self.model.generate(
                        prompt=full_prompt,
                        max_new_tokens=self.config.model.max_new_tokens,
                        temperature=self.config.model.temperature,
                        top_p=self.config.model.top_p,
                        return_logprobs=True,
                    )

                    # Execute action
                    result = self.environment.step(output.text)

                    # Track action type
                    if "search" in result.info.get("tool_name", ""):
                        total_searches += 1
                    elif "read" in result.info.get("tool_name", ""):
                        total_reads += 1

                    # Record step
                    step = TrajectoryStep(
                        turn_index=state.turn,
                        action_type=result.info.get("tool_name", "unknown"),
                        action_content=output.text,
                        action_parsed=result.info.get("metadata", {}),
                        observation=result.observation,
                        is_terminal=result.done,
                        metadata={
                            "logprobs": output.logprobs,
                            "token_ids": output.token_ids,
                        },
                    )
                    trajectory_steps.append(step)

                    state = self.environment.state

                # Build trajectory
                traj_data = self.environment.get_full_trajectory()
                trajectory = Trajectory(
                    question=question,
                    reference_answer=reference_answer or "",
                    reference_sources=reference_sources,
                    steps=trajectory_steps,
                    final_answer=traj_data["final_answer"],
                    cited_sources=traj_data["cited_sources"],
                    total_searches=total_searches,
                    total_reads=total_reads,
                )
                group_trajectories.append(trajectory)

            all_trajectories.append(group_trajectories)

        return all_trajectories

    def compute_group_advantages(
        self,
        trajectories: list[list[Trajectory]],
    ) -> tuple[list[list[float]], list[list[list[float]]], dict]:
        """
        Compute group-relative advantages.

        Key insight: GRPO compares within group, not absolute.
        This creates evolutionary pressure without external critic.
        """
        all_advantages = []
        all_turn_advantages = []
        metrics = {
            "mean_reward": 0.0,
            "mean_group_std": 0.0,
            "num_groups_with_variance": 0,
            "correct_rate": 0.0,
            "dont_know_rate": 0.0,
        }

        total_rewards = []
        correct_count = 0
        dont_know_count = 0
        total_count = 0

        for group in trajectories:
            # Compute rewards for group
            group_result = self.reward_computer.compute_group_rewards(group)

            # Store normalized advantages
            all_advantages.append(group_result["normalized_advantages"])

            # Turn-level advantages
            turn_advs = [
                r["turn_advantages"]
                for r in group_result["individual_rewards"]
            ]
            all_turn_advantages.append(turn_advs)

            # Collect metrics
            total_rewards.extend(group_result["raw_totals"])

            if group_result["has_sufficient_variance"]:
                metrics["num_groups_with_variance"] += 1
                metrics["mean_group_std"] += group_result["group_std"]

            for r in group_result["individual_rewards"]:
                total_count += 1
                if r["metadata"].get("is_correct"):
                    correct_count += 1
                if r["metadata"].get("is_dont_know"):
                    dont_know_count += 1

        # Aggregate metrics
        num_groups = len(trajectories)
        if num_groups > 0:
            metrics["mean_group_std"] /= num_groups

        if total_count > 0:
            metrics["mean_reward"] = sum(total_rewards) / len(total_rewards)
            metrics["correct_rate"] = correct_count / total_count
            metrics["dont_know_rate"] = dont_know_count / total_count

        # Additional reward statistics for monitoring
        if total_rewards:
            metrics["reward_std"] = (
                (sum((r - metrics["mean_reward"]) ** 2 for r in total_rewards) / len(total_rewards)) ** 0.5
                if len(total_rewards) > 1 else 0.0
            )
            metrics["reward_min"] = min(total_rewards)
            metrics["reward_max"] = max(total_rewards)
            metrics["reward_range"] = metrics["reward_max"] - metrics["reward_min"]

        return all_advantages, all_turn_advantages, metrics

    def prepare_batch(
        self,
        trajectories: list[list[Trajectory]],
        advantages: list[list[float]],
        turn_advantages: list[list[list[float]]],
    ) -> GRPOBatch:
        """
        Prepare batch for training.

        Tokenize trajectories and create action masks.
        """
        # Flatten for processing
        flat_trajectories = []
        flat_advantages = []
        flat_turn_advantages = []

        for group_trajs, group_advs, group_turn_advs in zip(
            trajectories, advantages, turn_advantages
        ):
            flat_trajectories.extend(group_trajs)
            flat_advantages.extend(group_advs)
            flat_turn_advantages.extend(group_turn_advs)

        # Tokenize each trajectory
        all_input_ids = []
        all_attention_masks = []
        all_action_masks = []

        tokenizer = self.model.tokenizer

        for traj in flat_trajectories:
            # Build full sequence: system + question + actions/observations
            full_text = self._build_trajectory_text(traj)

            # Tokenize
            encoded = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.model.max_context_length,
                padding="max_length",
                return_token_type_ids=False,
            )

            # Remove token_type_ids if present (some models don't support it)
            if "token_type_ids" in encoded:
                del encoded["token_type_ids"]

            all_input_ids.append(encoded["input_ids"][0])
            all_attention_masks.append(encoded["attention_mask"][0])

            # Create action mask (1 for action tokens, 0 for observations)
            action_mask = self._create_action_mask(traj, encoded["input_ids"][0])
            all_action_masks.append(action_mask)

        # Stack tensors
        input_ids = torch.stack(all_input_ids)
        attention_mask = torch.stack(all_attention_masks)
        action_masks = torch.stack(all_action_masks)

        return GRPOBatch(
            question_ids=[t.question[:50] for t in flat_trajectories],
            trajectories=trajectories,
            rewards=[[self.reward_computer.compute_trajectory_reward(t)["total_reward"]
                      for t in group] for group in trajectories],
            advantages=advantages,
            turn_advantages=turn_advantages,
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            action_masks=action_masks.to(self.device),
        )

    def _build_trajectory_text(self, traj: Trajectory) -> str:
        """Build full text sequence for trajectory."""
        parts = [
            self.environment.get_system_prompt(),
            f"\nQuestion: {traj.question}\n",
        ]

        for step in traj.steps:
            parts.append(f"\nAction: {step.action_content}")
            parts.append(f"\nObservation: {step.observation}")

        if traj.final_answer:
            parts.append(f"\nFinal Answer: {traj.final_answer}")

        return "".join(parts)

    def _create_action_mask(
        self,
        traj: Trajectory,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create mask for action tokens.

        Key insight: mask tool outputs from gradient computation.
        Model learns WHAT to do, not to predict tool outputs.
        """
        tokenizer = self.model.tokenizer
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # Find action boundaries in tokenized sequence
        # This is approximate - in production, track exact positions during tokenization
        full_text = self._build_trajectory_text(traj)

        for step in traj.steps:
            if step.action_content:
                # Find action in text
                action_start = full_text.find(step.action_content)
                if action_start >= 0:
                    # Approximate token position
                    prefix = full_text[:action_start]
                    prefix_tokens = tokenizer(prefix, return_tensors="pt", return_token_type_ids=False)["input_ids"].shape[1]

                    action_tokens = tokenizer(step.action_content, return_tensors="pt", return_token_type_ids=False)["input_ids"].shape[1]

                    # Set mask for action tokens
                    end_pos = min(prefix_tokens + action_tokens, len(mask))
                    mask[prefix_tokens:end_pos] = True

        return mask

    def compute_policy_loss(
        self,
        batch: GRPOBatch,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute GRPO policy loss.

        Loss = -advantage * log_prob * action_mask + kl_penalty
        """
        self.model.prepare_for_training()

        # Forward pass
        outputs = self.model.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        logits = outputs.logits

        # Compute log probs for actions taken
        shift_logits = logits[:, :-1, :]
        shift_labels = batch.input_ids[:, 1:]
        shift_mask = batch.action_masks[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Apply action mask (only compute loss on action tokens)
        masked_log_probs = token_log_probs * shift_mask

        # Flatten advantages to match batch
        flat_advantages = []
        for group_advs in batch.advantages:
            flat_advantages.extend(group_advs)
        advantages = torch.tensor(flat_advantages, device=self.device)

        # Compute per-sequence log prob
        seq_log_probs = masked_log_probs.sum(dim=-1) / (shift_mask.sum(dim=-1) + 1e-8)

        # Policy gradient loss
        pg_loss = -(advantages * seq_log_probs).mean()

        # KL penalty
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.reference_model is not None and self.grpo_config.kl_coef > 0:
            with torch.no_grad():
                ref_outputs = self.reference_model.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                )
                ref_logits = ref_outputs.logits

            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_token_log_probs = torch.gather(
                ref_log_probs,
                dim=-1,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)

            # KL divergence (approximation)
            kl = (token_log_probs - ref_token_log_probs) * shift_mask
            kl_loss = self.grpo_config.kl_coef * kl.sum(dim=-1).mean()

        # Entropy bonus
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        masked_entropy = (entropy * shift_mask).sum(dim=-1) / (shift_mask.sum(dim=-1) + 1e-8)
        mean_entropy = masked_entropy.mean()
        entropy_loss = -self.grpo_config.entropy_coef * mean_entropy

        # Total loss
        total_loss = pg_loss + kl_loss + entropy_loss

        # Compute additional metrics for monitoring
        advantage_std = advantages.std().item() if len(advantages) > 1 else 0.0

        metrics = {
            # Core losses
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            # Policy metrics
            "mean_log_prob": seq_log_probs.mean().item(),
            "log_prob_std": seq_log_probs.std().item() if len(seq_log_probs) > 1 else 0.0,
            # Advantage metrics
            "mean_advantage": advantages.mean().item(),
            "advantage_std": advantage_std,
            "advantage_max": advantages.max().item(),
            "advantage_min": advantages.min().item(),
            # Entropy metrics (critical for exploration)
            "entropy": mean_entropy.item(),
            "entropy_min": masked_entropy.min().item(),
            "entropy_max": masked_entropy.max().item(),
            # Action mask stats
            "action_mask_ratio": shift_mask.float().mean().item(),
        }

        return total_loss, metrics

    def train_step(self, batch: GRPOBatch) -> dict:
        """Single training step."""
        self.optimizer.zero_grad()

        loss, metrics = self.compute_policy_loss(batch)

        loss.backward()

        # Compute gradient statistics before clipping
        trainable_params = self.model.get_trainable_parameters()
        grad_norms = []
        grad_maxs = []
        for p in trainable_params:
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
                grad_maxs.append(p.grad.abs().max().item())

        if grad_norms:
            metrics["grad_norm_pre_clip"] = sum(grad_norms) / len(grad_norms)
            metrics["grad_max"] = max(grad_maxs)

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            trainable_params,
            self.grpo_config.max_grad_norm,
        )
        metrics["grad_norm"] = grad_norm.item()

        # Check for NaN/Inf gradients
        has_nan = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in trainable_params
        )
        metrics["grad_has_nan_inf"] = float(has_nan)

        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1
        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
        metrics["global_step"] = self.global_step

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        callbacks: list | None = None,
    ) -> dict:
        """Train for one epoch."""
        self.epoch += 1
        epoch_metrics = []

        # Call epoch start callbacks
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, "on_epoch_start"):
                    callback.on_epoch_start(self)

        for batch_idx, batch_data in enumerate(dataloader):
            # Generate rollouts
            trajectories = self.generate_rollouts(
                batch_data,
                group_size=self.grpo_config.group_size,
            )

            # Compute advantages
            advantages, turn_advantages, reward_metrics = self.compute_group_advantages(
                trajectories
            )

            # Skip if no variance (GRPO needs variance)
            if reward_metrics["num_groups_with_variance"] == 0:
                continue

            # Prepare batch
            batch = self.prepare_batch(trajectories, advantages, turn_advantages)

            # Train step
            step_metrics = self.train_step(batch)
            step_metrics.update(reward_metrics)

            epoch_metrics.append(step_metrics)

            # Callbacks - support both function and object callbacks
            if callbacks:
                should_stop = False
                for callback in callbacks:
                    if hasattr(callback, "on_step"):
                        callback.on_step(self, step_metrics)
                        # Check for early stopping
                        if hasattr(callback, "should_stop") and callback.should_stop:
                            should_stop = True
                    elif callable(callback):
                        callback(self, step_metrics)

                if should_stop:
                    break

        # Aggregate epoch metrics
        if epoch_metrics:
            agg_metrics = {
                k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                for k in epoch_metrics[0].keys()
                if isinstance(epoch_metrics[0].get(k), (int, float))
            }
            agg_metrics["epoch"] = self.epoch
            agg_metrics["steps_in_epoch"] = len(epoch_metrics)

            # Call epoch end callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_epoch_end"):
                        callback.on_epoch_end(self, agg_metrics)

            return agg_metrics

        return {"epoch": self.epoch, "steps_in_epoch": 0}

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int | None = None,
        max_steps: int | None = None,
        eval_dataloader: DataLoader | None = None,
        callbacks: list | None = None,
    ) -> dict:
        """Full training loop."""
        num_epochs = num_epochs or self.config.training.num_epochs
        max_steps = max_steps or self.config.training.max_steps

        all_metrics = []
        start_time = time.time()

        # Call train start callbacks
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, "on_train_start"):
                    callback.on_train_start(self)

        should_stop = False
        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(train_dataloader, callbacks)
            all_metrics.append(epoch_metrics)

            # Check for early stopping from callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "should_stop") and callback.should_stop:
                        should_stop = True
                        break

            if should_stop:
                break

            # Evaluation
            if eval_dataloader and (epoch + 1) % 1 == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                epoch_metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

            # Save best
            if epoch_metrics.get("mean_reward", 0) > self.best_reward:
                self.best_reward = epoch_metrics["mean_reward"]
                self.save_checkpoint(
                    Path(self.config.training.checkpoint_dir) / "best"
                )

            # Check max steps
            if self.global_step >= max_steps:
                break

        total_time = time.time() - start_time

        final_result = {
            "epochs": len(all_metrics),
            "total_steps": self.global_step,
            "total_time": total_time,
            "best_reward": self.best_reward,
            "final_metrics": all_metrics[-1] if all_metrics else {},
        }

        # Call train end callbacks
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, "on_train_end"):
                    callback.on_train_end(self, final_result)

        return final_result

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on dataset."""
        self.model.prepare_for_inference()

        all_rewards = []
        correct = 0
        dont_know = 0
        total = 0

        with torch.no_grad():
            for batch_data in dataloader:
                trajectories = self.generate_rollouts(batch_data, group_size=1)

                for group in trajectories:
                    for traj in group:
                        result = self.reward_computer.compute_trajectory_reward(traj)
                        all_rewards.append(result["total_reward"])
                        total += 1
                        if result["metadata"].get("is_correct"):
                            correct += 1
                        if result["metadata"].get("is_dont_know"):
                            dont_know += 1

        return {
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            "accuracy": correct / total if total > 0 else 0,
            "dont_know_rate": dont_know / total if total > 0 else 0,
            "num_samples": total,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save training checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_checkpoint(str(path / "model"))

        # Save optimizer and scheduler
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_reward": self.best_reward,
        }, path / "trainer_state.pt")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint."""
        path = Path(path)

        # Load model
        self.model.load_checkpoint(str(path / "model"))

        # Load optimizer and scheduler
        state = torch.load(path / "trainer_state.pt")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_reward = state["best_reward"]
