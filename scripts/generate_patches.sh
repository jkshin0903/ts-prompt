#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Get split from command line argument, default to train
SPLIT="${1:-test}"
# Get patch_size from command line argument, default to 16
PATCH_SIZE="${2:-8}"
# Get stride from command line argument, default to same as patch_size
STRIDE="${3:-$PATCH_SIZE}"

# Set output directory
OUTPUT_DIR="patches/${SPLIT}"

echo "=========================================="
echo "Generating patches from dataset:"
echo "  Split: $SPLIT"
echo "  Patch size: $PATCH_SIZE"
echo "  Stride: $STRIDE"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

# Generate patches using patches.py utilities
python -u <<EOF
from pathlib import Path
from utils.patches import generate_all_patches, write_patches_to_txt

# Determine base directory (project root)
base_dir = Path("$PROJECT_ROOT")

# Generate patches for all symbols
print("Generating patches from CSV files...")
patches = generate_all_patches(
    base_dir=base_dir,
    patch_size=${PATCH_SIZE},
    stride=${STRIDE},
    split="$SPLIT",
)

print(f"Generated patches for {len(patches)} symbols:")
for symbol, symbol_patches in patches.items():
    print(f"  {symbol}: {len(symbol_patches)} patches")

# Write patches to text files
output_path = base_dir / "$OUTPUT_DIR"
write_patches_to_txt(patches, output_path)

print(f"\nWrote patches for split '$SPLIT' to: {output_path}")
print("==========================================")
EOF

echo "Patch generation completed!"

