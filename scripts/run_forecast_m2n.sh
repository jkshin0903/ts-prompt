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

# Sanitize model name for directory name
MODEL_DIR=$(echo "$MODEL" | sed 's/-/_/g' | sed 's/\./_/g')
# Extract dataset name from CSV file
DATASET_NAME=$(basename "$CSV_FILE" .csv)
OUTPUT_DIR="responses/${DATASET_NAME}/${MODEL_DIR}"

echo "=========================================="
echo "Running M2N forecast for:"
echo "  CSV file: $CSV_FILE"
echo "  Dataset: $DATASET_NAME"
echo "  Model: $MODEL"
echo "=========================================="

# Function to run forecast using forecast_m2n.py directly
run_forecast() {
    local num_input=$1
    local num_predict=$2
    local output_suffix="${num_input}_${num_predict}"
    
    echo ""
    echo "--- Starting forecast: num_input=$num_input, num_predict=$num_predict ---"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loading rows from CSV..."
    
    python -u <<EOF
import sys
import time
from pathlib import Path
from src.forecast_m2n import load_rows_from_source, run_m2n_forecast

# Load rows
start_time = time.time()
print("[{}] Loading rows from: $CSV_FILE".format(time.strftime('%Y-%m-%d %H:%M:%S')))
rows = load_rows_from_source(csv_path="$CSV_FILE")
load_time = time.time() - start_time
print("[{}] ✓ Loaded {} rows (took {:.2f}s)".format(time.strftime('%Y-%m-%d %H:%M:%S'), len(rows), load_time))
sys.stdout.flush()

# Determine file stem
file_stem = Path("$CSV_FILE").stem

# Run forecast
num_input = $num_input
num_predict = $num_predict
print("[{}] Generating prompt: num_input={}, num_predict={}, start_index=0".format(time.strftime('%Y-%m-%d %H:%M:%S'), num_input, num_predict))
sys.stdout.flush()

prompt_start = time.time()
prompt, response = run_m2n_forecast(
    rows=rows,
    num_input_rows=num_input,
    num_predict_rows=num_predict,
    model_name="$MODEL",
    start_index=0,
    temperature=0.0,
    output_dir="${OUTPUT_DIR}_${output_suffix}",
    save_prompt=True,
    file_stem=file_stem,
)
total_time = time.time() - prompt_start

print("[{}] ✓ Prompt generated ({} characters)".format(time.strftime('%Y-%m-%d %H:%M:%S'), len(prompt)))
print("[{}] ✓ Response received ({} characters)".format(time.strftime('%Y-%m-%d %H:%M:%S'), len(response)))
print("[{}] ✓ Saved to: ${OUTPUT_DIR}_${output_suffix}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
print("[{}] Total time: {:.2f}s".format(time.strftime('%Y-%m-%d %H:%M:%S'), total_time))
sys.stdout.flush()
EOF
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- Completed forecast: num_input=$num_input, num_predict=$num_predict ---"
}

# Run with num_input=3, num_predict=1

run_forecast 24 24

run_forecast 36 36

run_forecast 48 48

run_forecast 60 60

run_forecast 72 72

run_forecast 84 84

run_forecast 96 96

echo "=========================================="
echo "Completed M2N forecast runs for $DATASET_NAME ($MODEL)"
echo "=========================================="

