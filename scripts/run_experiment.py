#!/usr/bin/env python3
"""
Quick experiment runner for RL Deep Research.

Usage:
    python scripts/run_experiment.py --model ai-forever/gigachat-lightning
    python scripts/run_experiment.py --config configs/gigachat_b200.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Run RL Deep Research experiment")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="ai-forever/gigachat-lightning",
        help="Model name or path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./outputs/experiment",
        help="Output directory"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--group-size", "-g",
        type=int,
        default=8,
        help="GRPO group size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum training samples"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (small dataset)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load (for eval or continue training)"
    )

    args = parser.parse_args()

    # Import here to avoid slow import on --help
    import torch
    from rl_deepresearch.config import (
        ExperimentConfig, ModelConfig, GRPOConfig,
        TrainingConfig, RewardConfig, EnvironmentConfig, DataConfig
    )
    from rl_deepresearch.models.gigachat import GigaChatLocalWrapper
    from rl_deepresearch.environment.corpus import DocumentCorpus
    from rl_deepresearch.environment.base import ResearchEnvironment
    from rl_deepresearch.rewards.compute import RewardComputer
    from rl_deepresearch.training.grpo import GRPOTrainer
    from rl_deepresearch.training.data import ResearchDataset, create_dataloader
    from rl_deepresearch.training.callbacks import (
        LoggingCallback, CheckpointCallback, WandbCallback
    )

    print("=" * 60)
    print("RL Deep Research - GRPO Training Experiment")
    print("=" * 60)

    # Set seed
    torch.manual_seed(42)

    # Load or create config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = ExperimentConfig(
            model=ModelConfig(
                model_name=args.model,
                use_lora=not args.no_lora,
                dtype="bf16",
                use_flash_attention=True,
            ),
            grpo=GRPOConfig(
                learning_rate=args.lr,
                group_size=args.group_size,
            ),
            reward=RewardConfig(),
            environment=EnvironmentConfig(),
            data=DataConfig(
                batch_size=args.batch_size,
            ),
            training=TrainingConfig(
                num_epochs=args.epochs,
                output_dir=args.output,
                checkpoint_dir=f"{args.output}/checkpoints",
                use_wandb=args.wandb,
            ),
            debug=args.debug,
        )

    # Setup
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    config.to_yaml(output_path / "config.yaml")

    print(f"\nConfiguration:")
    print(f"  Model: {config.model.model_name}")
    print(f"  LoRA: {config.model.use_lora}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Group size: {config.grpo.group_size}")
    print(f"  Learning rate: {config.grpo.learning_rate}")
    print(f"  Output: {args.output}")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\n  WARNING: No GPU available!")

    print("\n" + "=" * 60)

    # Load model
    print("\nLoading model...")
    model = GigaChatLocalWrapper(config.model)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)

    # Load corpus
    print("Loading corpus...")
    max_docs = 1000 if args.debug else 10000
    corpus = DocumentCorpus.from_hotpotqa(max_docs=max_docs)
    print(f"  Corpus size: {len(corpus)} documents")

    # Create environment
    environment = ResearchEnvironment(corpus, config.environment)

    # Create reward computer
    reward_computer = RewardComputer(config.reward)

    if args.eval_only:
        # Evaluation only
        from rl_deepresearch.agents.research_agent import ResearchAgent

        print("\nRunning evaluation...")
        agent = ResearchAgent(model, environment, verbose=True)

        eval_dataset = ResearchDataset.from_hotpotqa(
            split="validation",
            max_samples=100 if args.debug else 500,
        )

        correct = 0
        for i, item in enumerate(eval_dataset):
            result = agent.answer(item["question"])
            expected = item.get("answer", "").lower()
            predicted = result["answer"].lower()
            if expected in predicted or predicted in expected:
                correct += 1

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(eval_dataset)} | Accuracy: {correct/(i+1):.2%}")

        print(f"\nFinal Accuracy: {correct/len(eval_dataset):.2%}")
        return

    # Load training dataset
    print("Loading training dataset...")
    max_samples = 100 if args.debug else args.max_samples
    train_dataset = ResearchDataset.from_hotpotqa(
        split="train",
        max_samples=max_samples,
        min_difficulty=config.data.min_difficulty,
        require_multi_hop=config.data.require_multi_hop,
    )
    print(f"  Training samples: {len(train_dataset)}")

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        config=config,
        reward_computer=reward_computer,
        environment=environment,
    )

    # Setup callbacks
    callbacks = [
        LoggingCallback(log_every=10),
        CheckpointCallback(
            save_dir=output_path / "checkpoints",
            save_every=500,
        ),
    ]

    if args.wandb:
        callbacks.append(WandbCallback(
            project="rl-deepresearch",
            config=config.model_dump(),
        ))

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    metrics = trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=config.training.num_epochs,
        callbacks=callbacks,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTotal steps: {metrics['total_steps']}")
    print(f"Total time: {metrics['total_time']:.1f}s")
    print(f"Best reward: {metrics['best_reward']:.4f}")

    if metrics.get('final_metrics'):
        print("\nFinal metrics:")
        for k, v in metrics['final_metrics'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    print(f"\nCheckpoints saved to: {output_path / 'checkpoints'}")


if __name__ == "__main__":
    main()
