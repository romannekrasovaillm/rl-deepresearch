"""
CLI for RL Deep Research experiments.

Commands:
- train: Run GRPO training
- eval: Evaluate model
- serve: Interactive agent
- export: Export model
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="rl-research",
    help="RL Deep Research Agent CLI",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to config YAML file"
    ),
    model_name: str = typer.Option(
        "ai-sage/GigaChat3-10B-A1.8B-base",
        "--model", "-m",
        help="Model name or path (default: GigaChat3-10B MoE)"
    ),
    dataset: str = typer.Option(
        "hotpotqa",
        "--dataset", "-d",
        help="Dataset name (hotpotqa, custom path)"
    ),
    output_dir: str = typer.Option(
        "./outputs",
        "--output", "-o",
        help="Output directory"
    ),
    num_epochs: int = typer.Option(
        3,
        "--epochs", "-e",
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size", "-b",
        help="Batch size (questions per batch)"
    ),
    group_size: int = typer.Option(
        8,
        "--group-size", "-g",
        help="GRPO group size (trajectories per question)"
    ),
    learning_rate: float = typer.Option(
        1e-5,
        "--lr",
        help="Learning rate"
    ),
    use_lora: bool = typer.Option(
        True,
        "--lora/--no-lora",
        help="Use LoRA adapters"
    ),
    use_wandb: bool = typer.Option(
        False,
        "--wandb/--no-wandb",
        help="Enable Weights & Biases logging"
    ),
    wandb_project: str = typer.Option(
        "rl-deepresearch",
        "--wandb-project",
        help="W&B project name"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Debug mode (small dataset, verbose)"
    ),
):
    """
    Train research agent with GRPO.

    Default model: ai-sage/GigaChat3-10B-A1.8B-base (MoE: 10B total, 1.8B active)

    Example:
        rl-research train --epochs 3
        rl-research train --config configs/gigachat_b200.yaml
    """
    import torch
    from .config import ExperimentConfig, ModelConfig, GRPOConfig, TrainingConfig
    from .models.gigachat import GigaChatLocalWrapper
    from .environment.corpus import DocumentCorpus
    from .environment.base import ResearchEnvironment
    from .rewards.compute import RewardComputer
    from .training.grpo import GRPOTrainer
    from .training.data import ResearchDataset, create_dataloader
    from .training.callbacks import CheckpointCallback
    from .training.logging import create_rl_logging_stack

    console.print("[bold blue]RL Deep Research Training[/bold blue]")

    # Set seed
    torch.manual_seed(seed)

    # Load or create config
    if config_path:
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig(
            model=ModelConfig(
                model_name=model_name,
                use_lora=use_lora,
            ),
            grpo=GRPOConfig(
                learning_rate=learning_rate,
                group_size=group_size,
            ),
            training=TrainingConfig(
                num_epochs=num_epochs,
                output_dir=output_dir,
            ),
            seed=seed,
            debug=debug,
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config.to_yaml(output_path / "config.yaml")

    console.print(f"Output directory: {output_dir}")
    console.print(f"Model: {model_name}")
    console.print(f"Dataset: {dataset}")

    # Load model
    console.print("\n[yellow]Loading model...[/yellow]")
    model = GigaChatLocalWrapper(config.model)

    # Load corpus
    console.print("[yellow]Loading corpus...[/yellow]")
    max_docs = 1000 if debug else 10000
    corpus = DocumentCorpus.from_hotpotqa(max_docs=max_docs)
    console.print(f"Corpus size: {len(corpus)} documents")

    # Create environment
    environment = ResearchEnvironment(corpus, config.environment)

    # Create reward computer
    reward_computer = RewardComputer(config.reward)

    # Load dataset
    console.print("[yellow]Loading dataset...[/yellow]")
    max_samples = 100 if debug else 10000
    train_dataset = ResearchDataset.from_hotpotqa(
        split="train",
        max_samples=max_samples,
        min_difficulty=config.data.min_difficulty,
        require_multi_hop=config.data.require_multi_hop,
    )
    console.print(f"Training samples: {len(train_dataset)}")

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        config=config,
        reward_computer=reward_computer,
        environment=environment,
    )

    # Setup callbacks with comprehensive RL logging
    log_dir = output_path / "logs"
    tracker, rl_callbacks = create_rl_logging_stack(
        log_dir=log_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project if use_wandb else None,
        wandb_config=config.model_dump() if use_wandb else None,
    )

    callbacks = rl_callbacks + [
        CheckpointCallback(
            save_dir=output_path / "checkpoints",
            save_every=500,
        ),
    ]

    console.print(f"[yellow]RL logging enabled → {log_dir}[/yellow]")

    # Train
    console.print("\n[bold green]Starting training...[/bold green]")
    metrics = trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=num_epochs,
        callbacks=callbacks,
    )

    # Print final metrics
    console.print("\n[bold]Training Complete![/bold]")
    table = Table(title="Final Metrics")
    table.add_column("Metric")
    table.add_column("Value")

    for k, v in metrics.get("final_metrics", {}).items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))

    console.print(table)

    # Print training health summary
    health = tracker.get_health_score()
    health_color = "green" if health > 0.8 else "yellow" if health > 0.5 else "red"
    console.print(f"\n[bold]Training Health Score: [{health_color}]{health:.2f}[/{health_color}][/bold]")

    # Print any alerts
    alerts = tracker.get_recent_alerts(limit=10)
    if alerts:
        console.print("\n[bold yellow]Training Alerts:[/bold yellow]")
        for alert in alerts:
            level_color = "red" if alert.level.value >= 2 else "yellow"
            console.print(f"  [{level_color}]{alert.level.name}[/{level_color}]: {alert.message}")


@app.command()
def evaluate(
    checkpoint_path: str = typer.Argument(
        ...,
        help="Path to model checkpoint"
    ),
    dataset: str = typer.Option(
        "hotpotqa",
        "--dataset", "-d",
        help="Dataset to evaluate on"
    ),
    split: str = typer.Option(
        "validation",
        "--split", "-s",
        help="Dataset split"
    ),
    max_samples: int = typer.Option(
        500,
        "--max-samples", "-n",
        help="Maximum samples to evaluate"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for predictions"
    ),
):
    """
    Evaluate model on dataset.

    Example:
        rl-research evaluate ./checkpoints/best --dataset hotpotqa
    """
    import json
    from .config import ExperimentConfig, ModelConfig, EnvironmentConfig
    from .models.gigachat import GigaChatLocalWrapper
    from .environment.corpus import DocumentCorpus
    from .environment.base import ResearchEnvironment
    from .agents.research_agent import ResearchAgent
    from .training.data import ResearchDataset

    console.print("[bold blue]Evaluating Model[/bold blue]")
    console.print(f"Checkpoint: {checkpoint_path}")

    # Load model
    config = ModelConfig()
    model = GigaChatLocalWrapper(config)
    model.load_checkpoint(checkpoint_path)

    # Load corpus
    corpus = DocumentCorpus.from_hotpotqa(max_docs=5000)

    # Create environment and agent
    environment = ResearchEnvironment(corpus, EnvironmentConfig())
    agent = ResearchAgent(model, environment)

    # Load dataset
    eval_dataset = ResearchDataset.from_hotpotqa(
        split=split,
        max_samples=max_samples,
    )

    console.print(f"Evaluating on {len(eval_dataset)} samples...")

    # Evaluate
    correct = 0
    dont_know = 0
    predictions = []

    for i, item in enumerate(eval_dataset):
        result = agent.answer(item["question"])

        # Check correctness (simple exact match)
        expected = item.get("answer", "").lower()
        predicted = result["answer"].lower()
        is_correct = expected in predicted or predicted in expected

        if is_correct:
            correct += 1
        if result["is_dont_know"]:
            dont_know += 1

        predictions.append({
            "question": item["question"],
            "expected": item.get("answer"),
            "predicted": result["answer"],
            "correct": is_correct,
            "confidence": result["confidence"],
            "sources": result["sources"],
        })

        if (i + 1) % 50 == 0:
            acc = correct / (i + 1)
            console.print(f"Progress: {i+1}/{len(eval_dataset)} | Accuracy: {acc:.2%}")

    # Print results
    accuracy = correct / len(eval_dataset)
    dont_know_rate = dont_know / len(eval_dataset)

    console.print("\n[bold]Evaluation Results[/bold]")
    table = Table()
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Accuracy", f"{accuracy:.2%}")
    table.add_row("Don't Know Rate", f"{dont_know_rate:.2%}")
    table.add_row("Total Samples", str(len(eval_dataset)))
    console.print(table)

    # Save predictions
    if output_file:
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2)
        console.print(f"Predictions saved to {output_file}")


@app.command()
def serve(
    checkpoint_path: Optional[str] = typer.Option(
        None,
        "--checkpoint", "-c",
        help="Path to model checkpoint"
    ),
    model_name: str = typer.Option(
        "ai-sage/GigaChat3-10B-A1.8B-base",
        "--model", "-m",
        help="Model name (default: GigaChat3-10B MoE)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose mode"
    ),
):
    """
    Start interactive research agent.

    Example:
        rl-research serve --checkpoint ./checkpoints/best
    """
    from .config import ModelConfig, EnvironmentConfig
    from .models.gigachat import GigaChatLocalWrapper
    from .environment.corpus import DocumentCorpus
    from .environment.base import ResearchEnvironment
    from .agents.research_agent import ResearchAgent

    console.print("[bold blue]Starting Research Agent[/bold blue]")

    # Load model
    config = ModelConfig(model_name=model_name)
    model = GigaChatLocalWrapper(config)

    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
        console.print(f"Loaded checkpoint: {checkpoint_path}")

    # Load corpus
    console.print("Loading corpus...")
    corpus = DocumentCorpus.from_hotpotqa(max_docs=5000)

    # Create agent
    environment = ResearchEnvironment(corpus, EnvironmentConfig())
    agent = ResearchAgent(model, environment, verbose=verbose)

    # Start interactive mode
    agent.interactive()


@app.command()
def export(
    checkpoint_path: str = typer.Argument(
        ...,
        help="Path to checkpoint"
    ),
    output_path: str = typer.Option(
        "./exported_model",
        "--output", "-o",
        help="Output path"
    ),
    format: str = typer.Option(
        "huggingface",
        "--format", "-f",
        help="Export format (huggingface, onnx, safetensors)"
    ),
    merge_lora: bool = typer.Option(
        True,
        "--merge-lora/--no-merge-lora",
        help="Merge LoRA weights into base model"
    ),
):
    """
    Export model for deployment.

    Example:
        rl-research export ./checkpoints/best --output ./exported
    """
    console.print("[bold blue]Exporting Model[/bold blue]")

    from .config import ModelConfig
    from .models.gigachat import GigaChatLocalWrapper

    # Load model
    config = ModelConfig()
    model = GigaChatLocalWrapper(config)
    model.load_checkpoint(checkpoint_path)

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    if format == "huggingface":
        if merge_lora and config.use_lora:
            console.print("Merging LoRA weights...")
            model.model = model.model.merge_and_unload()

        model.model.save_pretrained(output)
        model.tokenizer.save_pretrained(output)
        console.print(f"Exported to {output}")

    elif format == "safetensors":
        from safetensors.torch import save_file
        state_dict = model.model.state_dict()
        save_file(state_dict, output / "model.safetensors")
        console.print(f"Exported to {output / 'model.safetensors'}")

    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """Show system and GPU information."""
    import torch
    from .models.utils import get_device_info

    console.print("[bold blue]System Information[/bold blue]")

    table = Table()
    table.add_column("Property")
    table.add_column("Value")

    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        device_info = get_device_info()
        table.add_row("GPU", device_info.get("device_name", "Unknown"))
        table.add_row("GPU Memory", f"{device_info.get('total_memory_gb', 0):.1f} GB")
        table.add_row("Compute Capability", device_info.get("compute_capability", "Unknown"))
        table.add_row("Is B200", str(device_info.get("is_b200", False)))

    console.print(table)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
