"""Utility functions for creating and saving forecast prompts."""

from __future__ import annotations

from pathlib import Path
from typing import List

from utils.rows import format_rows


def create_m2n_prompt(
    all_rows: List[str],
    num_input_rows: int,
    num_predict_rows: int,
    start_index: int | None = None,
    instruction_file: Path | None = None,
) -> str:
    """Create a forecasting prompt asking to predict N rows from M input rows.
    
    Args:
        all_rows: List of all available rows, each row is a string in format "date,val1,val2,..."
        num_input_rows: Number of past rows to use as input (M)
        num_predict_rows: Number of rows to predict (N)
        start_index: Optional starting index. If None, uses the last M rows from all_rows
        instruction_file: Optional path to instruction template file. If None, uses templates/Instruction_m2n.txt
    
    Returns:
        Formatted prompt string
    """
    if num_input_rows <= 0:
        raise ValueError("num_input_rows must be greater than 0")
    if num_predict_rows <= 0:
        raise ValueError("num_predict_rows must be greater than 0")
    
    # Determine input rows
    if start_index is None:
        # Use last M rows
        if len(all_rows) < num_input_rows:
            raise ValueError(f"Not enough rows. Need at least {num_input_rows}, got {len(all_rows)}")
        input_rows = all_rows[-num_input_rows:]
    else:
        # Use rows starting from start_index
        if start_index + num_input_rows > len(all_rows):
            raise ValueError(
                f"Not enough rows from index {start_index}. "
                f"Need {num_input_rows} rows, but only {len(all_rows) - start_index} available."
            )
        input_rows = all_rows[start_index:start_index + num_input_rows]
    
    instruction_path = instruction_file or Path(__file__).parent.parent / "templates" / "Instruction_m2n.txt"
    
    # Load instruction template from file
    if not instruction_path.exists():
        raise FileNotFoundError(f"Instruction template file not found: {instruction_path}")
    instruction_template = instruction_path.read_text(encoding="utf-8")

    # Format input rows
    input_rows_str = format_rows(input_rows)

    # Create the full prompt
    prompt = instruction_template.format(
        m=len(input_rows),
        n=num_predict_rows,
        input_rows=input_rows_str,
    )

    return prompt


def create_m2n_prompt_with_feedback(
    predicted_patches: str,
    actual_patches: List[List[str]],
    feedback_start_index: int,
    prediction_start_index: int | None = None,
    num_predictions: int | None = None,
    instruction_file: Path | None = None,
) -> str:
    """Create a feedback-based prompt asking to re-predict patches after receiving actual values.
    
    This function creates a prompt that:
    1. Shows the LLM's previous predictions
    2. Provides the actual patch values as feedback
    3. Encourages the LLM to try again based on what it has learned
    
    Args:
        predicted_patches: String containing the patches that the LLM previously predicted
        actual_patches: List of actual patch values, each patch is a list of strings (date,open,high,low,close)
        feedback_start_index: Starting index for the patches that received feedback (previous prediction range)
        prediction_start_index: Starting index for the patches to predict next. If None, uses feedback_end_index + 1
        num_predictions: Number of patches to predict. If None, uses the same number as feedback patches
        instruction_file: Optional path to instruction template file. If None, uses templates/instruction_m2n_with_feedback.txt
    
    Returns:
        Formatted prompt string with feedback and re-prediction request
    """
    if not actual_patches:
        raise ValueError("actual_patches must not be empty")
    
    num_feedback_patches = len(actual_patches)
    previous_start_index = feedback_start_index
    previous_end_index = feedback_start_index + num_feedback_patches - 1
    
    # Determine next prediction range
    if prediction_start_index is None:
        next_start_index = previous_end_index + 1
    else:
        next_start_index = prediction_start_index
    
    if num_predictions is None:
        num_predictions = num_feedback_patches
    
    next_end_index = next_start_index + num_predictions - 1
    
    instruction_path = instruction_file or Path(__file__).parent.parent / "templates" / "instruction_m2n_with_feedback.txt"
    
    # Load instruction template from file
    if not instruction_path.exists():
        raise FileNotFoundError(f"Instruction template file not found: {instruction_path}")
    instruction_template = instruction_path.read_text(encoding="utf-8")
    
    # Format actual patches (each patch is a list of row strings)
    formatted_actual_patches_list = []
    for patch in actual_patches:
        # Each patch is a list of row strings, join them with newlines
        patch_str = format_rows(patch)
        formatted_actual_patches_list.append(patch_str)
    
    actual_patches_str = "\n\n".join(formatted_actual_patches_list)
    
    # Create the full prompt
    prompt = instruction_template.format(
        x=num_feedback_patches,
        previous_start_index=previous_start_index,
        previous_end_index=previous_end_index,
        next_start_index=next_start_index,
        next_end_index=next_end_index,
        predicted_patches=predicted_patches.strip(),
        actual_patches=actual_patches_str,
    )
    
    return prompt


def save_forecast_prompt(
    prompt: str,
    output_path: Path | str,
    encoding: str = "utf-8",
) -> None:
    """Save the forecast prompt to a file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding=encoding)

