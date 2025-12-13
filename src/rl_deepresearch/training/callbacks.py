"""
Training callbacks for monitoring and checkpointing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .grpo import GRPOTrainer


class TrainingCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_step(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Called after each training step."""
        pass

    def on_epoch_start(self, trainer: "GRPOTrainer") -> None:
        """Called at start of epoch."""
        pass

    def on_epoch_end(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Called at end of epoch."""
        pass

    def on_train_start(self, trainer: "GRPOTrainer") -> None:
        """Called at start of training."""
        pass

    def on_train_end(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Called at end of training."""
        pass


class WandbCallback(TrainingCallback):
    """Weights & Biases logging callback."""

    def __init__(
        self,
        project: str = "rl-deepresearch",
        entity: str | None = None,
        name: str | None = None,
        config: dict | None = None,
        log_every: int = 1,
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config
        self.log_every = log_every
        self._initialized = False

    def _init_wandb(self) -> None:
        """Initialize wandb run."""
        if self._initialized:
            return

        try:
            import wandb

            wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config,
            )
            self._initialized = True
        except ImportError:
            print("wandb not installed, skipping logging")

    def on_train_start(self, trainer: "GRPOTrainer") -> None:
        """Initialize wandb at training start."""
        self._init_wandb()

    def on_step(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Log metrics to wandb."""
        if not self._initialized:
            return

        if trainer.global_step % self.log_every != 0:
            return

        import wandb

        # Filter out non-numeric values
        log_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        log_metrics["step"] = trainer.global_step

        wandb.log(log_metrics)

    def on_train_end(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Finish wandb run."""
        if not self._initialized:
            return

        import wandb
        wandb.finish()


class CheckpointCallback(TrainingCallback):
    """Checkpoint saving callback."""

    def __init__(
        self,
        save_dir: str | Path,
        save_every: int = 1000,
        save_best: bool = True,
        metric_name: str = "mean_reward",
        maximize: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.save_best = save_best
        self.metric_name = metric_name
        self.maximize = maximize

        self.best_metric = float("-inf") if maximize else float("inf")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Save checkpoint periodically."""
        if trainer.global_step % self.save_every != 0:
            return

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint-{trainer.global_step}"
        trainer.save_checkpoint(checkpoint_path)

        # Check for best
        if self.save_best and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            is_best = (
                (self.maximize and metric_value > self.best_metric) or
                (not self.maximize and metric_value < self.best_metric)
            )

            if is_best:
                self.best_metric = metric_value
                best_path = self.save_dir / "best"
                trainer.save_checkpoint(best_path)
                print(f"New best {self.metric_name}: {metric_value:.4f}")


class EvalCallback(TrainingCallback):
    """Evaluation callback."""

    def __init__(
        self,
        eval_dataloader: Any,
        eval_every: int = 500,
    ):
        self.eval_dataloader = eval_dataloader
        self.eval_every = eval_every

    def on_step(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Run evaluation periodically."""
        if trainer.global_step % self.eval_every != 0:
            return

        eval_metrics = trainer.evaluate(self.eval_dataloader)

        # Add to metrics
        for k, v in eval_metrics.items():
            metrics[f"eval_{k}"] = v

        print(f"Step {trainer.global_step} eval: {eval_metrics}")


class CurriculumCallback(TrainingCallback):
    """
    Curriculum learning callback.

    Advances curriculum stage based on performance.
    """

    def __init__(
        self,
        dataset: Any,  # CurriculumDataset
        advance_threshold: float = 0.7,
        metric_name: str = "correct_rate",
        check_every: int = 500,
    ):
        self.dataset = dataset
        self.advance_threshold = advance_threshold
        self.metric_name = metric_name
        self.check_every = check_every
        self.metrics_window = []

    def on_step(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Check if should advance curriculum."""
        if self.metric_name in metrics:
            self.metrics_window.append(metrics[self.metric_name])

        if trainer.global_step % self.check_every != 0:
            return

        if not self.metrics_window:
            return

        # Check average performance
        avg_metric = sum(self.metrics_window) / len(self.metrics_window)
        self.metrics_window = []

        if avg_metric >= self.advance_threshold:
            if hasattr(self.dataset, "advance_stage"):
                advanced = self.dataset.advance_stage()
                if advanced:
                    print(f"Advanced curriculum to stage {self.dataset.current_stage}")


class GradientMonitorCallback(TrainingCallback):
    """Monitor gradient statistics."""

    def __init__(self, log_every: int = 100):
        self.log_every = log_every

    def on_step(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Log gradient statistics."""
        if trainer.global_step % self.log_every != 0:
            return

        params = trainer.model.get_trainable_parameters()

        grad_norms = []
        grad_maxs = []

        for p in params:
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
                grad_maxs.append(p.grad.abs().max().item())

        if grad_norms:
            metrics["grad_norm_mean"] = sum(grad_norms) / len(grad_norms)
            metrics["grad_max"] = max(grad_maxs)


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping based on validation metric."""

    def __init__(
        self,
        patience: int = 5,
        metric_name: str = "eval_mean_reward",
        maximize: bool = True,
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.metric_name = metric_name
        self.maximize = maximize
        self.min_delta = min_delta

        self.best_metric = float("-inf") if maximize else float("inf")
        self.counter = 0
        self.should_stop = False

    def on_step(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Check for early stopping."""
        if self.metric_name not in metrics:
            return

        metric_value = metrics[self.metric_name]

        if self.maximize:
            improved = metric_value > self.best_metric + self.min_delta
        else:
            improved = metric_value < self.best_metric - self.min_delta

        if improved:
            self.best_metric = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered after {self.patience} evaluations without improvement")


class LoggingCallback(TrainingCallback):
    """Simple console logging callback."""

    def __init__(self, log_every: int = 10):
        self.log_every = log_every

    def on_step(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Log to console."""
        if trainer.global_step % self.log_every != 0:
            return

        step = trainer.global_step
        loss = metrics.get("total_loss", 0)
        reward = metrics.get("mean_reward", 0)
        lr = metrics.get("learning_rate", 0)
        correct = metrics.get("correct_rate", 0)

        print(
            f"Step {step:5d} | "
            f"Loss: {loss:.4f} | "
            f"Reward: {reward:.4f} | "
            f"Correct: {correct:.2%} | "
            f"LR: {lr:.2e}"
        )
