#!/bin/bash
# Training script for GigaChat-Lightning on NVIDIA B200
# Optimized for RL deep research agent training with GRPO

set -e

# Configuration
CONFIG_PATH="${CONFIG_PATH:-configs/gigachat_b200.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/gigachat-grpo-$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Environment variables for B200 optimization
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

# Flash Attention 2
export FLASH_ATTENTION_FORCE_TRITON=1

echo "========================================"
echo "RL Deep Research - GRPO Training"
echo "========================================"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "========================================"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Create output directory
mkdir -p "$OUTPUT_DIR"
cp "$CONFIG_PATH" "$OUTPUT_DIR/config.yaml"

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        -m rl_deepresearch.cli train \
        --config "$CONFIG_PATH" \
        --output "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/train.log"
else
    # Single GPU
    python -m rl_deepresearch.cli train \
        --config "$CONFIG_PATH" \
        --output "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/train.log"
fi

echo "========================================"
echo "Training complete!"
echo "Output: $OUTPUT_DIR"
echo "========================================"
