#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Get dataset type from command line argument, default to BTCUSDT
DATASET_TYPE=ADAUSDT
# Get split from command line argument, default to train
SPLIT=train
# Get model name from command line argument, default to gpt-4.1-mini-2025-04-14
MODEL=gpt-4.1-mini-2025-04-14
# Get language from command line argument, default to kr
LANGUAGE=kr

# Set paths
PATCH_FILE="patches/${SPLIT}/${DATASET_TYPE}_patches.txt"
# Sanitize model name for directory name
MODEL_DIR=$(echo "$MODEL" | sed 's/-/_/g' | sed 's/\./_/g')
OUTPUT_DIR="responses/${SPLIT}/${MODEL_DIR}"

echo "=========================================="
echo "Running forecast for:"
echo "  Dataset: $DATASET_TYPE"
echo "  Split: $SPLIT"
echo "  Model: $MODEL"
echo "  Language: $LANGUAGE"
echo "=========================================="

# Run with num_input=3, num_predict=1
python -u src/main.py \
  --patch_file "$PATCH_FILE" \
  --num_input 3 \
  --num_predict 1 \
  --start_index 0 \
  --language "$LANGUAGE" \
  --model "$MODEL" \
  --output_dir "${OUTPUT_DIR}_3_1" \
  --save_prompt

# Run with num_input=3, num_predict=2
python -u src/main.py \
  --patch_file "$PATCH_FILE" \
  --num_input 3 \
  --num_predict 2 \
  --start_index 0 \
  --language "$LANGUAGE" \
  --model "$MODEL" \
  --output_dir "${OUTPUT_DIR}_3_2" \
  --save_prompt

# Run with num_input=3, num_predict=3
python -u src/main.py \
  --patch_file "$PATCH_FILE" \
  --num_input 3 \
  --num_predict 3 \
  --start_index 0 \
  --language "$LANGUAGE" \
  --model "$MODEL" \
  --output_dir "${OUTPUT_DIR}_3_3" \
  --save_prompt

# Run with num_input=5, num_predict=1
python -u src/main.py \
  --patch_file "$PATCH_FILE" \
  --num_input 5 \
  --num_predict 1 \
  --start_index 0 \
  --language "$LANGUAGE" \
  --model "$MODEL" \
  --output_dir "${OUTPUT_DIR}_5_1" \
  --save_prompt

# Run with num_input=5, num_predict=2
python -u src/main.py \
  --patch_file "$PATCH_FILE" \
  --num_input 5 \
  --num_predict 2 \
  --start_index 0 \
  --language "$LANGUAGE" \
  --model "$MODEL" \
  --output_dir "${OUTPUT_DIR}_5_2" \
  --save_prompt

# Run with num_input=5, num_predict=3
python -u src/main.py \
  --patch_file "$PATCH_FILE" \
  --num_input 5 \
  --num_predict 3 \
  --start_index 0 \
  --language "$LANGUAGE" \
  --model "$MODEL" \
  --output_dir "${OUTPUT_DIR}_5_3" \
  --save_prompt

echo "=========================================="
echo "Completed forecast runs for $DATASET_TYPE ($SPLIT, $MODEL, $LANGUAGE)"
echo "=========================================="

