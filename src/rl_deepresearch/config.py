"""
Configuration classes for RL Deep Research experiments.

Based on insights from research papers:
- GRPO requires sufficient trajectory variance (group_size >= 6)
- Reward structure: format errors (-2) < wrong answers (-1) < unknown (0-1) < correct (1-2)
- Efficiency bonuses for minimal search usage
- KL-penalty for stability
"""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class RewardType(str, Enum):
    """Reward signal types."""

    FORMAT_ERROR = "format_error"
    TOOL_ERROR = "tool_error"
    WRONG_ANSWER = "wrong_answer"
    DONT_KNOW = "dont_know"
    PARTIAL_CORRECT = "partial_correct"
    CORRECT = "correct"
    CORRECT_EFFICIENT = "correct_efficient"


class RewardConfig(BaseModel):
    """
    Reward configuration implementing insights from research:

    - Epistemic calibration: "I don't know" gets positive reward (0.0-1.0)
    - Format rewards as implicit regularization
    - Efficiency bonuses as anti-over-search mechanism
    - Progress rewards as bootstrap mechanism
    """

    # Terminal rewards (final answer)
    correct_answer: float = Field(default=2.0, description="Reward for correct final answer")
    correct_efficient_bonus: float = Field(default=0.5, description="Bonus for correct with minimal searches")
    partial_correct_base: float = Field(default=0.5, description="Base reward for partially correct")
    wrong_answer: float = Field(default=-1.0, description="Penalty for wrong answer")
    dont_know_reward: float = Field(default=0.3, description="Reward for admitting uncertainty")

    # Format penalties (sparse negative signals)
    format_error: float = Field(default=-2.0, description="Penalty for unparseable output")
    bad_tool_name: float = Field(default=-1.5, description="Penalty for invalid tool name")
    bad_tool_args: float = Field(default=-1.0, description="Penalty for invalid tool arguments")

    # Progress rewards (dense positive signals)
    found_relevant_doc: float = Field(default=0.3, description="Reward for finding relevant document")
    read_useful_content: float = Field(default=0.2, description="Reward for reading useful content")
    information_gain: float = Field(default=0.1, description="Reward per unit of new information")

    # Efficiency penalties (anti-redundancy)
    redundant_search_penalty: float = Field(default=-0.1, description="Penalty for duplicate searches")
    excessive_turns_penalty: float = Field(default=-0.05, description="Penalty per turn beyond optimal")

    # Efficiency thresholds
    efficient_search_threshold: int = Field(default=3, description="Max searches for efficiency bonus")
    max_turns_before_penalty: int = Field(default=6, description="Turns before efficiency penalty kicks in")

    # Word-level F1 for soft matching (instead of exact match)
    use_word_f1: bool = Field(default=True, description="Use word-level F1 for partial credit")

    # Citation correctness as grounding proxy
    citation_correct_bonus: float = Field(default=0.2, description="Bonus for correct source citation")
    citation_wrong_penalty: float = Field(default=-0.3, description="Penalty for wrong citation")


class GRPOConfig(BaseModel):
    """
    GRPO (Group Relative Policy Optimization) configuration.

    Key insights:
    - Group size must be sufficient for variance (>=6)
    - KL-penalty anchors against catastrophic forgetting
    - Temperature controls exploration-exploitation
    """

    # Group sampling
    group_size: int = Field(default=8, description="Number of trajectories per question (GRPO insight: needs variance)")
    min_group_variance: float = Field(default=0.01, description="Minimum reward variance to use batch")

    # Policy optimization
    learning_rate: float = Field(default=1e-5, description="Learning rate for policy updates")
    kl_coef: float = Field(default=0.05, description="KL penalty coefficient (anchor to reference)")
    clip_ratio: float = Field(default=0.2, description="PPO-style clipping ratio")

    # Value estimation
    use_baseline: bool = Field(default=False, description="Use value baseline (GRPO uses group mean)")
    gamma: float = Field(default=0.99, description="Discount factor")
    gae_lambda: float = Field(default=0.95, description="GAE lambda for advantage estimation")

    # Turn-level credit assignment
    use_turn_level_advantage: bool = Field(default=True, description="MT-GRPO style per-turn advantages")
    immediate_reward_weight: float = Field(default=0.3, description="Weight for immediate vs final rewards")

    # Entropy for exploration
    entropy_coef: float = Field(default=0.01, description="Entropy bonus coefficient")
    min_entropy_threshold: float = Field(default=0.1, description="Minimum entropy before branching (ARPO)")

    # Gradient computation
    max_grad_norm: float = Field(default=1.0, description="Gradient clipping norm")
    mask_tool_outputs: bool = Field(default=True, description="Mask tool output tokens from loss (credit assignment)")

    # Training stability
    warmup_steps: int = Field(default=100, description="Warmup steps for learning rate")
    weight_decay: float = Field(default=0.01, description="Weight decay for regularization")


class ModelConfig(BaseModel):
    """
    Model configuration for GigaChat-Lightning with B200 optimization.

    Key insights:
    - LoRA as implicit regularization (limits capacity to overfit)
    - YaRN for extended context (enables long-horizon RL)
    - Flash attention for efficiency
    """

    # Model selection
    model_name: str = Field(default="ai-forever/gigachat-lightning", description="Base model name/path")
    tokenizer_name: str | None = Field(default=None, description="Tokenizer (defaults to model_name)")

    # Precision for B200
    dtype: Literal["bf16", "fp16", "fp32"] = Field(default="bf16", description="Compute dtype (bf16 optimal for B200)")

    # LoRA configuration (implicit regularization)
    use_lora: bool = Field(default=True, description="Use LoRA adapters")
    lora_r: int = Field(default=64, description="LoRA rank")
    lora_alpha: int = Field(default=128, description="LoRA alpha scaling")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout")
    lora_target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Modules to apply LoRA"
    )

    # Context length (YaRN extension)
    max_context_length: int = Field(default=32768, description="Maximum context length")
    use_yarn: bool = Field(default=True, description="Use YaRN for context extension")
    yarn_scale: float = Field(default=4.0, description="YaRN scaling factor")

    # Attention optimization
    use_flash_attention: bool = Field(default=True, description="Use Flash Attention 2")
    use_sdpa: bool = Field(default=True, description="Use PyTorch SDPA as fallback")

    # Generation
    max_new_tokens: int = Field(default=2048, description="Maximum tokens per generation")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling threshold")

    # Quantization (optional for memory)
    load_in_8bit: bool = Field(default=False, description="Load in 8-bit quantization")
    load_in_4bit: bool = Field(default=False, description="Load in 4-bit quantization")


class EnvironmentConfig(BaseModel):
    """
    Research environment configuration.

    Key insights:
    - Max turns affects exploration budget (min ~4-6 for learning)
    - Tool interface as structured action space
    - Hierarchical document structure for navigation
    """

    # Episode limits
    max_turns: int = Field(default=10, description="Maximum turns per episode (insight: too low kills learning)")
    min_turns_for_training: int = Field(default=4, description="Minimum turns to include trajectory")

    # Tool configuration
    available_tools: list[str] = Field(
        default=["keyword_search", "semantic_search", "read_document", "read_section", "answer", "dont_know"],
        description="Available tools for agent"
    )
    max_search_results: int = Field(default=10, description="Maximum search results returned")
    max_read_tokens: int = Field(default=4096, description="Maximum tokens per read action")

    # Document corpus
    corpus_path: str | None = Field(default=None, description="Path to document corpus")
    use_wikipedia: bool = Field(default=False, description="Use Wikipedia as corpus")
    use_web_search: bool = Field(default=False, description="Enable real web search")

    # Simulation (ZeroSearch-style)
    use_simulated_search: bool = Field(default=False, description="Use LLM-simulated search results")
    simulation_noise_level: float = Field(default=0.1, description="Noise curriculum for simulation")

    # State management
    compress_history: bool = Field(default=True, description="Compress search history (MEM1 insight)")
    max_history_tokens: int = Field(default=8192, description="Maximum tokens for history state")

    # Hierarchical IDs for navigation
    use_hierarchical_ids: bool = Field(default=True, description="Use A:B:C style document IDs")


class DataConfig(BaseModel):
    """
    Data configuration for training.

    Key insights:
    - Difficulty filtering improves sample efficiency
    - Multi-source composition prevents shortcut learning
    - Decontamination is critical for valid evaluation
    """

    # Dataset
    train_dataset: str = Field(default="hotpotqa", description="Training dataset name")
    eval_dataset: str | None = Field(default=None, description="Evaluation dataset")

    # Filtering
    min_difficulty: float = Field(default=0.3, description="Minimum difficulty score")
    max_difficulty: float = Field(default=0.95, description="Maximum difficulty score")
    require_multi_hop: bool = Field(default=True, description="Require multi-hop reasoning")

    # Decontamination
    decontaminate: bool = Field(default=True, description="Filter questions model can answer from memory")
    decontamination_threshold: float = Field(default=0.8, description="Threshold for parametric knowledge")

    # Batch composition
    batch_size: int = Field(default=4, description="Questions per batch")
    effective_batch_size: int = Field(default=32, description="Effective batch size with GRPO groups")

    # Data augmentation
    duplicate_informative_samples: bool = Field(default=True, description="DUPO-style duplication")
    max_duplicates: int = Field(default=3, description="Maximum duplicates per sample")


class TrainingConfig(BaseModel):
    """
    Training configuration with B200 optimization.

    Key insights:
    - Async actor-learner for tool latency
    - Staleness-aware updates for distributed training
    - Curriculum from data, not reward complexity
    """

    # Basic training
    num_epochs: int = Field(default=3, description="Number of training epochs")
    max_steps: int = Field(default=10000, description="Maximum training steps")
    eval_steps: int = Field(default=500, description="Steps between evaluations")
    save_steps: int = Field(default=1000, description="Steps between checkpoints")

    # B200 optimization
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation")
    gradient_checkpointing: bool = Field(default=True, description="Use gradient checkpointing")

    # DeepSpeed
    use_deepspeed: bool = Field(default=True, description="Use DeepSpeed")
    deepspeed_stage: Literal[1, 2, 3] = Field(default=2, description="DeepSpeed ZeRO stage")

    # Distributed
    num_gpus: int = Field(default=1, description="Number of GPUs")

    # Async training (for tool latency)
    async_rollouts: bool = Field(default=True, description="Generate rollouts asynchronously")
    num_rollout_workers: int = Field(default=4, description="Number of rollout workers")
    max_rollout_staleness: int = Field(default=3, description="Maximum policy versions for rollout")

    # Curriculum
    use_curriculum: bool = Field(default=True, description="Use difficulty curriculum")
    curriculum_stages: int = Field(default=3, description="Number of curriculum stages")

    # Checkpointing
    output_dir: str = Field(default="./outputs", description="Output directory")
    checkpoint_dir: str = Field(default="./checkpoints", description="Checkpoint directory")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    use_wandb: bool = Field(default=True, description="Use Weights & Biases")
    wandb_project: str = Field(default="rl-deepresearch", description="W&B project name")
    wandb_entity: str | None = Field(default=None, description="W&B entity/team")


class ExperimentConfig(BaseSettings):
    """
    Complete experiment configuration.

    Combines all sub-configs for a full RL deep research experiment.
    """

    # Sub-configurations
    reward: RewardConfig = Field(default_factory=RewardConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Experiment metadata
    experiment_name: str = Field(default="gigachat-grpo-research", description="Experiment name")
    seed: int = Field(default=42, description="Random seed")
    debug: bool = Field(default=False, description="Debug mode")

    class Config:
        env_prefix = "RL_RESEARCH_"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        from omegaconf import OmegaConf

        conf = OmegaConf.load(path)
        return cls(**OmegaConf.to_container(conf, resolve=True))

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        from omegaconf import OmegaConf

        conf = OmegaConf.create(self.model_dump())
        OmegaConf.save(conf, path)

    def get_deepspeed_config(self) -> dict:
        """Generate DeepSpeed configuration for B200."""
        return {
            "train_batch_size": self.data.effective_batch_size,
            "train_micro_batch_size_per_gpu": self.data.batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "gradient_clipping": self.grpo.max_grad_norm,
            "bf16": {
                "enabled": self.model.dtype == "bf16"
            },
            "fp16": {
                "enabled": self.model.dtype == "fp16",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": self.training.deepspeed_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.training.deepspeed_stage >= 2 else "none",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu" if self.training.deepspeed_stage >= 3 else "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
            },
            "activation_checkpointing": {
                "partition_activations": self.training.gradient_checkpointing,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": True,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
            },
            "wall_clock_breakdown": False,
        }
