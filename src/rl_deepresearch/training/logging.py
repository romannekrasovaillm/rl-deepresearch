"""
Comprehensive RL logging and monitoring.

Critical metrics to track:
1. Reward collapse - rewards dropping to constant value
2. KL divergence explosion - policy diverging from reference
3. Entropy collapse - loss of exploration
4. Gradient explosion - NaN/Inf in gradients
5. Staleness in async - training on outdated rollouts

Key insight: Without proper logging, RL failures are invisible
until it's too late to recover.
"""

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .grpo import GRPOTrainer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("rl_deepresearch")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Training alert."""
    level: AlertLevel
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    step: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "step": self.step,
            "timestamp": self.timestamp,
        }


@dataclass
class MetricWindow:
    """Sliding window for metric statistics."""
    values: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, value: float) -> None:
        if value is not None and not math.isnan(value) and not math.isinf(value):
            self.values.append(value)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        mean = self.mean
        variance = sum((x - mean) ** 2 for x in self.values) / len(self.values)
        return math.sqrt(variance)

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def trend(self) -> float:
        """Calculate trend (positive = increasing, negative = decreasing)."""
        if len(self.values) < 10:
            return 0.0
        recent = list(self.values)[-10:]
        older = list(self.values)[-20:-10] if len(self.values) >= 20 else list(self.values)[:10]
        if not older:
            return 0.0
        return (sum(recent) / len(recent)) - (sum(older) / len(older))


class RLMetricsTracker:
    """
    Comprehensive metrics tracker for RL training.

    Monitors all critical failure modes:
    - Reward collapse
    - KL explosion
    - Entropy collapse
    - Gradient issues
    - Training staleness
    """

    def __init__(
        self,
        # Thresholds for alerts
        reward_collapse_threshold: float = 0.01,  # Std below this = collapse
        kl_explosion_threshold: float = 10.0,  # KL above this = explosion
        entropy_collapse_threshold: float = 0.1,  # Entropy below this = collapse
        grad_norm_threshold: float = 100.0,  # Grad norm above this = explosion
        staleness_threshold: int = 5,  # Policy versions behind = stale
        # Window sizes
        window_size: int = 100,
        # Callbacks
        alert_callback: Callable[[Alert], None] | None = None,
    ):
        self.reward_collapse_threshold = reward_collapse_threshold
        self.kl_explosion_threshold = kl_explosion_threshold
        self.entropy_collapse_threshold = entropy_collapse_threshold
        self.grad_norm_threshold = grad_norm_threshold
        self.staleness_threshold = staleness_threshold
        self.alert_callback = alert_callback

        # Metric windows
        self.windows: dict[str, MetricWindow] = {}

        # Alert history
        self.alerts: list[Alert] = []

        # Training state
        self.start_time = time.time()
        self.step = 0

        # Tracking flags
        self.reward_collapse_detected = False
        self.kl_explosion_detected = False
        self.entropy_collapse_detected = False
        self.gradient_explosion_detected = False

        # Critical metrics list
        self.critical_metrics = [
            "mean_reward",
            "reward_std",
            "kl_loss",
            "kl_divergence",
            "entropy",
            "entropy_loss",
            "grad_norm",
            "grad_max",
            "pg_loss",
            "total_loss",
            "mean_advantage",
            "advantage_std",
            "correct_rate",
            "dont_know_rate",
            "policy_version",
            "mean_staleness",
        ]

    def get_window(self, name: str) -> MetricWindow:
        """Get or create metric window."""
        if name not in self.windows:
            self.windows[name] = MetricWindow()
        return self.windows[name]

    def log_metrics(self, metrics: dict[str, Any], step: int) -> dict[str, Any]:
        """
        Log metrics and check for issues.

        Returns enriched metrics with statistics.
        """
        self.step = step
        enriched = dict(metrics)

        # Update windows and compute stats
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                window = self.get_window(name)
                window.add(value)

                # Add statistics
                if len(window.values) >= 10:
                    enriched[f"{name}_mean"] = window.mean
                    enriched[f"{name}_std"] = window.std
                    enriched[f"{name}_trend"] = window.trend

        # Check for issues
        self._check_reward_collapse(metrics)
        self._check_kl_explosion(metrics)
        self._check_entropy_collapse(metrics)
        self._check_gradient_issues(metrics)
        self._check_staleness(metrics)
        self._check_nan_inf(metrics)

        # Add health status
        enriched["training_health"] = self._compute_health_score()
        enriched["alerts_count"] = len([a for a in self.alerts if a.step == step])

        return enriched

    def _check_reward_collapse(self, metrics: dict) -> None:
        """Check for reward collapse (no variance in rewards)."""
        reward_window = self.get_window("mean_reward")

        if len(reward_window.values) < 20:
            return

        # Check if variance is too low
        if reward_window.std < self.reward_collapse_threshold:
            if not self.reward_collapse_detected:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    "REWARD COLLAPSE DETECTED: Reward variance near zero. "
                    "Model may have converged to degenerate solution.",
                    "reward_std",
                    reward_window.std,
                    self.reward_collapse_threshold,
                )
                self.reward_collapse_detected = True
        else:
            self.reward_collapse_detected = False

        # Check for reward dropping trend
        if reward_window.trend < -0.1:
            self._create_alert(
                AlertLevel.WARNING,
                f"Reward declining: trend = {reward_window.trend:.4f}",
                "reward_trend",
                reward_window.trend,
                -0.1,
            )

    def _check_kl_explosion(self, metrics: dict) -> None:
        """Check for KL divergence explosion."""
        kl_value = metrics.get("kl_loss") or metrics.get("kl_divergence", 0)

        if kl_value > self.kl_explosion_threshold:
            if not self.kl_explosion_detected:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"KL EXPLOSION DETECTED: KL = {kl_value:.4f}. "
                    "Policy diverging too far from reference. "
                    "Consider increasing KL penalty or reducing learning rate.",
                    "kl_divergence",
                    kl_value,
                    self.kl_explosion_threshold,
                )
                self.kl_explosion_detected = True
        elif kl_value > self.kl_explosion_threshold * 0.5:
            self._create_alert(
                AlertLevel.WARNING,
                f"KL divergence high: {kl_value:.4f}",
                "kl_divergence",
                kl_value,
                self.kl_explosion_threshold * 0.5,
            )
            self.kl_explosion_detected = False
        else:
            self.kl_explosion_detected = False

    def _check_entropy_collapse(self, metrics: dict) -> None:
        """Check for entropy collapse (loss of exploration)."""
        # Try different entropy metric names
        entropy = metrics.get("entropy") or metrics.get("policy_entropy")

        if entropy is None:
            # Compute from entropy loss if available
            entropy_loss = metrics.get("entropy_loss", 0)
            if entropy_loss != 0:
                # entropy_loss = -coef * entropy, so entropy = -entropy_loss / coef
                # Assume coef ~ 0.01
                entropy = abs(entropy_loss) / 0.01

        if entropy is None:
            return

        if entropy < self.entropy_collapse_threshold:
            if not self.entropy_collapse_detected:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"ENTROPY COLLAPSE DETECTED: Entropy = {entropy:.4f}. "
                    "Model has lost exploration capability. "
                    "Consider increasing entropy bonus or temperature.",
                    "entropy",
                    entropy,
                    self.entropy_collapse_threshold,
                )
                self.entropy_collapse_detected = True
        else:
            self.entropy_collapse_detected = False

    def _check_gradient_issues(self, metrics: dict) -> None:
        """Check for gradient explosion or vanishing."""
        grad_norm = metrics.get("grad_norm", 0)
        grad_max = metrics.get("grad_max", 0)

        # Check for explosion
        if grad_norm > self.grad_norm_threshold:
            if not self.gradient_explosion_detected:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"GRADIENT EXPLOSION: grad_norm = {grad_norm:.4f}. "
                    "Risk of NaN in weights. Reduce learning rate or increase clipping.",
                    "grad_norm",
                    grad_norm,
                    self.grad_norm_threshold,
                )
                self.gradient_explosion_detected = True
        else:
            self.gradient_explosion_detected = False

        # Check for vanishing
        grad_window = self.get_window("grad_norm")
        if len(grad_window.values) >= 20 and grad_window.mean < 1e-7:
            self._create_alert(
                AlertLevel.WARNING,
                f"Vanishing gradients: mean grad_norm = {grad_window.mean:.2e}",
                "grad_norm",
                grad_window.mean,
                1e-7,
            )

    def _check_staleness(self, metrics: dict) -> None:
        """Check for stale rollouts in async training."""
        mean_staleness = metrics.get("mean_staleness", 0)
        max_staleness = metrics.get("max_staleness", 0)

        if max_staleness > self.staleness_threshold:
            self._create_alert(
                AlertLevel.WARNING,
                f"Stale rollouts detected: max staleness = {max_staleness} versions. "
                "Consider filtering or weighting old rollouts.",
                "max_staleness",
                max_staleness,
                self.staleness_threshold,
            )

    def _check_nan_inf(self, metrics: dict) -> None:
        """Check for NaN or Inf values in metrics."""
        for name, value in metrics.items():
            if isinstance(value, float):
                if math.isnan(value):
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        f"NaN DETECTED in {name}! Training may be corrupted.",
                        name,
                        float('nan'),
                        0,
                    )
                elif math.isinf(value):
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        f"Inf DETECTED in {name}! Training may be corrupted.",
                        name,
                        float('inf'),
                        0,
                    )

    def _compute_health_score(self) -> float:
        """
        Compute overall training health score (0-1).

        1.0 = Healthy
        0.0 = Critical issues
        """
        score = 1.0

        # Penalize for detected issues
        if self.reward_collapse_detected:
            score -= 0.3
        if self.kl_explosion_detected:
            score -= 0.3
        if self.entropy_collapse_detected:
            score -= 0.2
        if self.gradient_explosion_detected:
            score -= 0.3

        # Penalize for recent alerts
        recent_alerts = [a for a in self.alerts if self.step - a.step < 100]
        critical_count = sum(1 for a in recent_alerts if a.level == AlertLevel.CRITICAL)
        warning_count = sum(1 for a in recent_alerts if a.level == AlertLevel.WARNING)

        score -= critical_count * 0.1
        score -= warning_count * 0.02

        return max(0.0, min(1.0, score))

    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ) -> None:
        """Create and log alert."""
        alert = Alert(
            level=level,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            step=self.step,
        )
        self.alerts.append(alert)

        # Log to console
        if level == AlertLevel.CRITICAL:
            logger.critical(f"[Step {self.step}] {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"[Step {self.step}] {message}")
        else:
            logger.info(f"[Step {self.step}] {message}")

        # Callback
        if self.alert_callback:
            self.alert_callback(alert)

    def get_summary(self) -> dict[str, Any]:
        """Get training summary."""
        elapsed = time.time() - self.start_time

        summary = {
            "total_steps": self.step,
            "elapsed_time": elapsed,
            "steps_per_second": self.step / elapsed if elapsed > 0 else 0,
            "health_score": self._compute_health_score(),
            "total_alerts": len(self.alerts),
            "critical_alerts": sum(1 for a in self.alerts if a.level == AlertLevel.CRITICAL),
            "warning_alerts": sum(1 for a in self.alerts if a.level == AlertLevel.WARNING),
        }

        # Add final metric values
        for name in self.critical_metrics:
            window = self.windows.get(name)
            if window and window.values:
                summary[f"final_{name}"] = window.values[-1]
                summary[f"mean_{name}"] = window.mean

        return summary

    def save_logs(self, path: str | Path) -> None:
        """Save all logs and alerts to file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save alerts
        alerts_data = [a.to_dict() for a in self.alerts]
        with open(path / "alerts.json", "w") as f:
            json.dump(alerts_data, f, indent=2)

        # Save summary
        summary = self.get_summary()
        with open(path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save metric history
        metrics_history = {
            name: list(window.values)
            for name, window in self.windows.items()
        }
        with open(path / "metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)

        logger.info(f"Logs saved to {path}")


class RLLoggingCallback:
    """
    Callback that integrates RLMetricsTracker with GRPOTrainer.

    Usage:
        tracker = RLMetricsTracker()
        callback = RLLoggingCallback(tracker)
        trainer.train(callbacks=[callback])
    """

    def __init__(
        self,
        tracker: RLMetricsTracker,
        log_every: int = 10,
        save_dir: str | Path | None = None,
        stop_on_critical: bool = False,
    ):
        self.tracker = tracker
        self.log_every = log_every
        self.save_dir = Path(save_dir) if save_dir else None
        self.stop_on_critical = stop_on_critical
        self.should_stop = False

    def on_step(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Process step metrics."""
        enriched = self.tracker.log_metrics(metrics, trainer.global_step)

        # Update metrics dict with enriched values
        metrics.update(enriched)

        # Log periodically
        if trainer.global_step % self.log_every == 0:
            self._log_status(trainer, enriched)

        # Check for critical issues
        if self.stop_on_critical:
            health = enriched.get("training_health", 1.0)
            if health < 0.3:
                logger.critical("Training health critical! Stopping training.")
                self.should_stop = True

    def _log_status(self, trainer: "GRPOTrainer", metrics: dict) -> None:
        """Log training status."""
        step = trainer.global_step
        health = metrics.get("training_health", 1.0)
        health_emoji = "🟢" if health > 0.7 else "🟡" if health > 0.3 else "🔴"

        log_parts = [
            f"Step {step:6d}",
            f"Health: {health_emoji} {health:.2f}",
        ]

        # Core metrics
        if "total_loss" in metrics:
            log_parts.append(f"Loss: {metrics['total_loss']:.4f}")
        if "mean_reward" in metrics:
            log_parts.append(f"Reward: {metrics['mean_reward']:.4f}")
        if "correct_rate" in metrics:
            log_parts.append(f"Correct: {metrics['correct_rate']:.1%}")
        if "kl_loss" in metrics:
            log_parts.append(f"KL: {metrics['kl_loss']:.4f}")
        if "grad_norm" in metrics:
            log_parts.append(f"GradNorm: {metrics['grad_norm']:.2f}")

        logger.info(" | ".join(log_parts))

    def on_train_start(self, trainer: "GRPOTrainer") -> None:
        """Called at training start."""
        logger.info("=" * 60)
        logger.info("RL Training Started")
        logger.info(f"Model: {trainer.config.model.model_name}")
        logger.info(f"Group size: {trainer.grpo_config.group_size}")
        logger.info(f"Learning rate: {trainer.grpo_config.learning_rate}")
        logger.info(f"KL coef: {trainer.grpo_config.kl_coef}")
        logger.info("=" * 60)

    def on_train_end(self, trainer: "GRPOTrainer", metrics: dict[str, Any]) -> None:
        """Called at training end."""
        summary = self.tracker.get_summary()

        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info(f"Total steps: {summary['total_steps']}")
        logger.info(f"Total time: {summary['elapsed_time']:.1f}s")
        logger.info(f"Steps/sec: {summary['steps_per_second']:.2f}")
        logger.info(f"Final health: {summary['health_score']:.2f}")
        logger.info(f"Critical alerts: {summary['critical_alerts']}")
        logger.info(f"Warnings: {summary['warning_alerts']}")
        logger.info("=" * 60)

        # Save logs
        if self.save_dir:
            self.tracker.save_logs(self.save_dir)


class TensorBoardRLLogger:
    """TensorBoard logging for RL metrics."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None

    @property
    def writer(self):
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(str(self.log_dir))
        return self._writer

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isnan(value) and not math.isinf(value):
                    self.writer.add_scalar(name, value, step)

    def log_histogram(self, name: str, values: torch.Tensor, step: int) -> None:
        """Log histogram to TensorBoard."""
        self.writer.add_histogram(name, values, step)

    def log_text(self, name: str, text: str, step: int) -> None:
        """Log text to TensorBoard."""
        self.writer.add_text(name, text, step)

    def close(self) -> None:
        if self._writer:
            self._writer.close()


def create_rl_logging_stack(
    log_dir: str | Path,
    use_wandb: bool = False,
    wandb_project: str = "rl-deepresearch",
    stop_on_critical: bool = False,
) -> tuple[RLMetricsTracker, list]:
    """
    Create complete logging stack for RL training.

    Returns:
        tracker: RLMetricsTracker instance
        callbacks: List of callbacks to pass to trainer
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create tracker
    tracker = RLMetricsTracker()

    # Create callbacks
    callbacks = []

    # Main logging callback
    main_callback = RLLoggingCallback(
        tracker=tracker,
        log_every=10,
        save_dir=log_dir / "logs",
        stop_on_critical=stop_on_critical,
    )
    callbacks.append(main_callback)

    # TensorBoard
    try:
        tb_logger = TensorBoardRLLogger(log_dir / "tensorboard")

        class TBCallback:
            def on_step(self, trainer, metrics):
                tb_logger.log_metrics(metrics, trainer.global_step)
            def on_train_end(self, trainer, metrics):
                tb_logger.close()

        callbacks.append(TBCallback())
    except ImportError:
        logger.warning("TensorBoard not available")

    # WandB
    if use_wandb:
        try:
            from .callbacks import WandbCallback
            callbacks.append(WandbCallback(project=wandb_project))
        except ImportError:
            logger.warning("wandb not available")

    return tracker, callbacks
