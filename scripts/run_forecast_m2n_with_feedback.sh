#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Get dataset type from command line argument, default to ADAUSDT
DATASET_TYPE="${1:-ADAUSDT}"
# Get split from command line argument, default to train
SPLIT="${2:-train}"
# Get model name from command line argument, default to gpt-4.1-mini-2025-04-14
MODEL="${3:-gpt-4.1-mini-2025-04-14}"

# Set paths
PATCH_FILE="patches/${SPLIT}/${DATASET_TYPE}_patches.txt"
# Sanitize model name for directory name
MODEL_DIR=$(echo "$MODEL" | sed 's/-/_/g' | sed 's/\./_/g')
OUTPUT_DIR="responses/${SPLIT}/${MODEL_DIR}"

echo "=========================================="
echo "Running M2N forecast with feedback for:"
echo "  Dataset: $DATASET_TYPE"
echo "  Split: $SPLIT"
echo "  Model: $MODEL"
echo "=========================================="

# Function to run forecast with feedback using forecast_m2n_with_feedback.py directly
run_forecast_with_feedback() {
    local num_input=$1
    local num_predict=$2
    local output_suffix="${num_input}_${num_predict}"
    
    python -u <<EOF
from pathlib import Path
from src.forecast_m2n_with_feedback import load_patches_from_source, run_m2n_forecast_with_feedback

# Load patches
patches = load_patches_from_source(patch_file_path="$PATCH_FILE")
print(f"Total patches loaded: {len(patches)}")

# Determine file stem
file_stem = Path("$PATCH_FILE").stem.replace("_patches", "")

# Run forecast with feedback
num_input = $num_input
num_predict = $num_predict
print(f"Running forecast with feedback: num_input={num_input}, num_predict={num_predict}, start_index=0")
prompts, responses = run_m2n_forecast_with_feedback(
    patches=patches,
    num_input_patches=num_input,
    num_predictions=num_predict,
    model_name="$MODEL",
    start_index=0,
    temperature=0.0,
    restrict_to_prompt=False,
    output_dir="${OUTPUT_DIR}_feedback_${output_suffix}",
    save_prompt=True,
    file_stem=file_stem,
)

print(f"\nCompleted {len(responses)} iterations")
print(f"Generated {len(prompts)} prompts")
print(f"Received {len(responses)} responses")
print(f"Saved to: ${OUTPUT_DIR}_feedback_${output_suffix}")
EOF
}

# Run with num_input=3, num_predict=1
run_forecast_with_feedback 3 1

# Run with num_input=3, num_predict=2
run_forecast_with_feedback 3 2

# Run with num_input=3, num_predict=3
run_forecast_with_feedback 3 3

# Run with num_input=5, num_predict=1
run_forecast_with_feedback 5 1

# Run with num_input=5, num_predict=2
run_forecast_with_feedback 5 2

# Run with num_input=5, num_predict=3
run_forecast_with_feedback 5 3

echo "=========================================="
echo "Completed M2N forecast with feedback runs for $DATASET_TYPE ($SPLIT, $MODEL)"
echo "=========================================="

