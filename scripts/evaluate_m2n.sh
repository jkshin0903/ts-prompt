#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Get CSV file name from command line argument (required)
CSV_NAME="${1:-}"
if [ -z "$CSV_NAME" ]; then
    echo "Error: CSV file name is required"
    echo ""
    echo "Usage: $0 <csv_file_name> [model_name]"
    echo ""
    echo "Examples:"
    echo "  $0 dataset/ETT-small/ETTh1.csv"
    echo "  $0 ETTh1.csv gpt-4.1-mini-2025-04-14"
    exit 1
fi

# Get model name from command line argument, default to gpt-4.1-mini-2025-04-14
MODEL="${2:-gpt-4.1-mini-2025-04-14}"

# Find CSV file (check if it's a path or just filename)
if [[ "$CSV_NAME" == *.csv ]]; then
    if [[ "$CSV_NAME" == /* ]] || [[ "$CSV_NAME" == dataset/* ]]; then
        # Absolute path or starts with dataset/
        CSV_FILE="$CSV_NAME"
    else
        # Just filename, search in dataset directory
        CSV_FILE=$(find dataset -name "$CSV_NAME" -type f | head -n 1)
    fi
else
    # No .csv extension, search for it
    CSV_FILE=$(find dataset -name "${CSV_NAME}.csv" -type f | head -n 1)
fi

if [ -z "$CSV_FILE" ] || [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_NAME"
    exit 1
fi

# Extract dataset name from CSV file
DATASET_NAME=$(basename "$CSV_FILE" .csv)

echo "=========================================="
echo "Evaluating M2N forecast for:"
echo "  CSV file: $CSV_FILE"
echo "  Dataset: $DATASET_NAME"
echo "  Model: $MODEL"
echo "=========================================="

# Function to run evaluation using evaluate_m2n.py
run_evaluation() {
    local num_input=$1
    local num_predict=$2
    
    echo ""
    echo "--- Starting evaluation: num_input=$num_input, num_predict=$num_predict ---"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running evaluation..."
    
    python -u src/evaluate_m2n.py \
        --csv "$CSV_FILE" \
        --num_input "$num_input" \
        --num_predict "$num_predict" \
        --model "$MODEL" \
        --start_index 0 \
        --auto_response
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- Completed evaluation: num_input=$num_input, num_predict=$num_predict ---"
}

# Run evaluations with different combinations

run_evaluation 24 24

run_evaluation 36 36

run_evaluation 48 48

run_evaluation 60 60

run_evaluation 72 72

run_evaluation 84 84

run_evaluation 96 96

echo "=========================================="
echo "Completed M2N evaluation runs for $DATASET_NAME ($MODEL)"
echo "=========================================="
