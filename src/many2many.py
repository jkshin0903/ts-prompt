from __future__ import annotations

from pathlib import Path
from typing import List

from utils.patches import load_patches_from_txt


def format_patch(patch: List[str], patch_index: int) -> str:
    """Format a single patch with header and data rows."""
    header = f"===== Patch {patch_index} =====\n"
    data = "\n".join(patch)
    return header + data


def create_forecast_prompt(
    all_patches: List[List[str]],
    num_input_patches: int,
    num_predictions: int,
    start_index: int | None = None,
    patch_structure_file: Path | None = None,
    instruction_file: Path | None = None,
) -> str:
    """Create a forecasting prompt asking to predict N patches from M input patches.
    
    Args:
        all_patches: List of all available patches, each patch is a list of strings (date,open,high,low,close)
        num_input_patches: Number of past patches to use as input (M)
        num_predictions: Number of patches to predict (N)
        start_index: Optional starting index. If None, uses the last M patches from all_patches
        patch_structure_file: Optional path to patch structure description file
        instruction_file: Optional path to instruction template file. If None, uses templates/instruction.txt
    
    Returns:
        Formatted prompt string
    """
    if num_input_patches <= 0:
        raise ValueError("num_input_patches must be greater than 0")
    if num_predictions <= 0:
        raise ValueError("num_predictions must be greater than 0")
    
    # Determine input patches
    if start_index is None:
        # Use last M patches
        if len(all_patches) < num_input_patches:
            raise ValueError(f"Not enough patches. Need at least {num_input_patches}, got {len(all_patches)}")
        input_patches = all_patches[-num_input_patches:]
        actual_start_index = len(all_patches) - num_input_patches
    else:
        # Use patches starting from start_index
        if start_index + num_input_patches > len(all_patches):
            raise ValueError(
                f"Not enough patches from index {start_index}. "
                f"Need {num_input_patches} patches, but only {len(all_patches) - start_index} available."
            )
        input_patches = all_patches[start_index:start_index + num_input_patches]
        actual_start_index = start_index
    
    structure_path = patch_structure_file or Path(__file__).parent.parent / "templates" / "patch_structure.txt"
    instruction_path = instruction_file or Path(__file__).parent.parent / "templates" / "instruction.txt"
    
    # Load instruction template from file
    if not instruction_path.exists():
        raise FileNotFoundError(f"Instruction template file not found: {instruction_path}")
    instruction_template = instruction_path.read_text(encoding="utf-8")

    # Load patch structure description if file exists
    structure_desc = ""
    if structure_path and structure_path.exists():
        structure_desc = structure_path.read_text(encoding="utf-8")
        structure_desc = "\n\n## Patch Structure Description\n\n" + structure_desc

    # Format input patches with correct indices
    formatted_patches_list = []
    for i, patch in enumerate(input_patches):
        formatted_patches_list.append(format_patch(patch, actual_start_index + i))
    
    input_patches_str = "\n\n".join(formatted_patches_list)

    # Create the full prompt
    prompt = instruction_template.format(
        m=len(input_patches),
        n=num_predictions,
        input_patches=input_patches_str,
    )
    
    if structure_desc:
        prompt = structure_desc + "\n\n" + prompt

    return prompt


if __name__ == "__main__":
    # Example usage - load patches from pre-generated text file
    example_patch_file = Path(__file__).parent.parent / "patches" / "train" / "ADAUSDT_patches.txt"
    
    if not example_patch_file.exists():
        print(f"Patch file not found: {example_patch_file}")
        print("Please run extract_patches.sh first to generate patch files.")
        exit(1)
    
    # Load patches from text file
    example_patches = load_patches_from_txt(example_patch_file)
    print(f"Loaded {len(example_patches)} patches from {example_patch_file}")
    
    # Use last 3 patches as input, predict next 2 patches
    prompt = create_forecast_prompt(
        all_patches=example_patches,
        num_input_patches=3,
        num_predictions=2
    )
    
    print("\n=== Prompt ===")
    print(prompt)

