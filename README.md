# RL Deep Research

Reinforcement Learning framework for training Deep Research Agents using GRPO (Group Relative Policy Optimization) on **GigaChat3-10B-A1.8B-base**, optimized for NVIDIA B200 GPUs.

## Model

**Default**: [ai-sage/GigaChat3-10B-A1.8B-base](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base)
- **Architecture**: Mixture of Experts (MoE)
- **Total Parameters**: 10B
- **Active Parameters**: 1.8B per forward pass
- **Efficiency**: High throughput with low memory footprint

## Key Features

Based on research insights from Kimi K2, Tongyi DeepResearch, and RL Foundations:

### Training Algorithm
- **GRPO (Group Relative Policy Optimization)**: No external critic needed, uses group-relative advantages
- **Turn-level credit assignment**: MT-GRPO style per-turn advantages
- **Tool output masking**: Gradients only on action tokens (implicit credit assignment)
- **KL penalty**: Anchor to reference policy for stability

### Reward System
- **Verifiable rewards**: EM, F1, document match for stable gradients
- **Epistemic calibration**: "I don't know" rewards (0.3) > wrong answer (-1.0)
- **Efficiency bonuses**: Reward for correct answers with minimal searches
- **Progress rewards**: Dense signals for finding relevant documents
- **Format penalties**: Structured action space enforcement

### Environment
- **Dual search**: Keyword (BM25) + Semantic search as learned ensemble
- **Hierarchical documents**: A:B:C style IDs for navigation learning
- **State compression**: MEM1-style compact state representation
- **Tool-augmented**: Search, read, answer, and don't_know actions

### B200 Optimization
- **BF16 precision**: Native B200 support
- **Flash Attention 2**: Optimized attention
- **LoRA**: Parameter-efficient fine-tuning
- **DeepSpeed ZeRO-2**: Memory optimization
- **Async rollouts**: Hide tool latency

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/rl-deepresearch.git
cd rl-deepresearch

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e ".[dev]"

# Optional: Flash Attention for B200
pip install flash-attn --no-build-isolation
```

## Quick Start

### Training

```bash
# Basic training (uses GigaChat3-10B-A1.8B-base by default)
rl-research train --epochs 3

# With config file (recommended for B200)
rl-research train --config configs/gigachat_b200.yaml

# Quick test
python scripts/run_experiment.py --debug --epochs 1

# Explicit model specification
rl-research train --model ai-sage/GigaChat3-10B-A1.8B-base --epochs 3
```

### Evaluation

```bash
# Evaluate checkpoint
rl-research evaluate ./checkpoints/best --dataset hotpotqa

# Interactive mode
rl-research serve --checkpoint ./checkpoints/best
```

## Project Structure

```
rl-deepresearch/
├── src/rl_deepresearch/
│   ├── config.py           # Configuration classes
│   ├── cli.py              # Command-line interface
│   ├── models/             # Model wrappers (GigaChat, HF)
│   ├── environment/        # Research environment + tools
│   ├── rewards/            # Reward computation
│   ├── training/           # GRPO trainer + rollouts
│   └── agents/             # High-level agent interface
├── configs/
│   └── gigachat_b200.yaml  # B200-optimized config
├── scripts/
│   ├── train_b200.sh       # Training script
│   └── run_experiment.py   # Python runner
└── tests/
```

## Configuration

Key configuration options (see `configs/gigachat_b200.yaml`):

```yaml
# Model - GigaChat3-10B MoE (10B total, 1.8B active)
model:
  model_name: "ai-sage/GigaChat3-10B-A1.8B-base"
  dtype: "bf16"
  use_lora: true
  lora_r: 64
  use_flash_attention: true

# GRPO
grpo:
  group_size: 8        # Trajectories per question
  learning_rate: 1e-5
  kl_coef: 0.05        # KL penalty
  mask_tool_outputs: true

# Rewards
reward:
  correct_answer: 2.0
  wrong_answer: -1.0
  dont_know_reward: 0.3   # Epistemic calibration
  format_error: -2.0      # Worse than wrong

# Environment
environment:
  max_turns: 10           # Exploration budget
  compress_history: true  # State compression
```

## Key Insights Implemented

### 1. Epistemic Calibration
"I don't know" gets positive reward, creating internal barrier against hallucination:
```python
dont_know_reward: 0.3  # Better than wrong (-1.0)
```

### 2. Tool Output Masking
Gradients only on action tokens - model learns WHAT to do, not to predict outputs:
```python
mask_tool_outputs: true
```

### 3. Group Relative Advantages
No external critic - compare within group for evolutionary pressure:
```python
normalized_advantages = (rewards - group_mean) / group_std
```

### 4. Efficiency Bonuses
Reward correct answers with minimal searches (emergent stopping):
```python
if correct and num_searches <= threshold:
    reward += efficiency_bonus
```

### 5. Progress Rewards as Bootstrap
Dense positive signals for intermediate progress:
```python
found_relevant_doc: 0.3
read_useful_content: 0.2
information_gain: 0.1
```

## Research References

This implementation is based on insights from:

1. **GRPO** - Group Relative Policy Optimization
2. **Kimi K2** - Scaling law for research agents
3. **Tongyi DeepResearch** - Data flywheel for self-improvement
4. **MEM1** - State compression for long-horizon RL
5. **MT-GRPO** - Turn-level credit assignment
6. **R1-Searcher++** - Efficiency bonuses
7. **ARPO** - Adaptive branching on uncertainty

## Hardware Requirements

- **Minimum**: 1x NVIDIA A100 40GB
- **Recommended**: 1x NVIDIA B200 80GB
- **Optimal**: 2-4x NVIDIA B200 with DeepSpeed

## License

MIT License

## Citation

```bibtex
@software{rl_deepresearch,
  title={RL Deep Research: GRPO Training for Research Agents},
  year={2025},
  url={https://github.com/your-org/rl-deepresearch}
}
```
