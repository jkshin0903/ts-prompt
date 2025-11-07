"""M2N forecasting with feedback: Iterative prediction with feedback for fine-tuning/learning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.forecast_m2n import get_model, call_model, load_patches_from_source
from utils.prompt import create_m2n_prompt, create_m2n_prompt_with_feedback
from utils.patches import extract_single_patch_from_response


def run_m2n_forecast_with_feedback(
    patches: List[List[str]],
    num_input_patches: int,
    num_predictions: int,
    model_name: str,
    start_index: int | None = None,
    temperature: float = 0.0,
    restrict_to_prompt: bool = False,
    output_dir: Path | str | None = None,
    save_prompt: bool = False,
    file_stem: str | None = None,
) -> tuple[List[str], List[str]]:
    """Run M2N forecasting with iterative feedback: predict one patch at a time with feedback.
    
    This function:
    1. Uses m input patches to predict the first patch (patch_i)
    2. Provides feedback on the predicted patch vs actual patch
    3. Iteratively predicts the next patch with feedback for n-1 more iterations
    
    Args:
        patches: List of patches, each patch is a list of strings (date,open,high,low,close)
        num_input_patches: Number of past patches to use as input (M)
        num_predictions: Number of patches to predict (N)
        model_name: Model name (e.g., 'gpt-4.1-mini-2025-04-14', 'gemini-2.0-flash')
        start_index: Optional starting patch index. If None, uses the last M patches
        temperature: Model temperature (default: 0.0)
        restrict_to_prompt: If True, add system instruction to forbid external knowledge
        output_dir: Directory to save responses and prompts. If None, only returns results
        save_prompt: Whether to save the generated prompts to files
        file_stem: File stem for saved files. If None, auto-generated
    
    Returns:
        Tuple of (list of prompts, list of responses) for each iteration
    """
    # Determine input patches and prediction start index
    if start_index is None:
        if len(patches) < num_input_patches:
            raise ValueError(f"Not enough patches. Need at least {num_input_patches}, got {len(patches)}")
        prediction_start_index = len(patches)  # Start predicting from the end
    else:
        if start_index + num_input_patches > len(patches):
            raise ValueError(
                f"Not enough patches from index {start_index}. "
                f"Need {num_input_patches} patches, but only {len(patches) - start_index} available."
            )
        prediction_start_index = start_index + num_input_patches
    
    # Check if we have enough actual patches for feedback
    if prediction_start_index + num_predictions > len(patches):
        raise ValueError(
            f"Not enough patches for feedback. Need {num_predictions} patches starting from index {prediction_start_index}, "
            f"but only {len(patches) - prediction_start_index} available."
        )
    
    # Initialize model
    model = get_model(model_name=model_name, temperature=temperature)
    
    all_prompts: List[str] = []
    all_responses: List[str] = []
    
    # First iteration: predict first patch using m2n prompt
    print(f"Iteration 1/{num_predictions}: Predicting patch {prediction_start_index}...")
    first_prompt = create_m2n_prompt(
        all_patches=patches,
        num_input_patches=num_input_patches,
        num_predictions=1,  # Predict only 1 patch
        start_index=start_index,
    )
    all_prompts.append(first_prompt)
    
    first_response = call_model(model, first_prompt, restrict_to_prompt=restrict_to_prompt)
    all_responses.append(first_response)
    
    print(f"  Response received ({len(first_response)} characters)")
    
    # Extract the predicted patch
    predicted_patch_str = extract_single_patch_from_response(first_response, prediction_start_index)
    
    # Get actual patch for feedback
    actual_patch = patches[prediction_start_index]
    
    # Iterate for remaining n-1 patches
    for i in range(1, num_predictions):
        current_patch_index = prediction_start_index + i - 1  # The patch we just got feedback for
        next_patch_index = prediction_start_index + i  # The patch to predict next
        
        print(f"Iteration {i+1}/{num_predictions}: Predicting patch {next_patch_index} with feedback...")
        
        # Create feedback prompt
        feedback_prompt = create_m2n_prompt_with_feedback(
            predicted_patches=predicted_patch_str,
            actual_patches=[actual_patch],
            feedback_start_index=current_patch_index,
            prediction_start_index=next_patch_index,
            num_predictions=1,  # Predict only 1 patch
        )
        all_prompts.append(feedback_prompt)
        
        # Get model response
        response = call_model(model, feedback_prompt, restrict_to_prompt=restrict_to_prompt)
        all_responses.append(response)
        
        print(f"  Response received ({len(response)} characters)")
        
        # Extract the predicted patch for next iteration
        predicted_patch_str = extract_single_patch_from_response(response, next_patch_index)
        
        # Get actual patch for next feedback
        actual_patch = patches[next_patch_index]
    
    # Save results if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if file_stem is None:
            file_stem = "forecast_feedback"
        
        model_safe = model_name.replace("-", "_").replace(".", "_")
        
        # Save all responses
        for idx, response in enumerate(all_responses):
            response_file = output_dir / f"{file_stem}_response_{model_safe}_iter{idx+1}.txt"
            response_file.write_text(response, encoding="utf-8")
        
        # Save prompts if requested
        if save_prompt:
            for idx, prompt in enumerate(all_prompts):
                prompt_file = output_dir / f"{file_stem}_prompt_{model_safe}_iter{idx+1}.txt"
                prompt_file.write_text(prompt, encoding="utf-8")
    
    return all_prompts, all_responses


def main():
    parser = argparse.ArgumentParser(
        description="M2N forecasting with feedback: Iterative prediction with feedback for learning"
    )
    
    # Prompt generation arguments
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (e.g., dataset/train/ADAUSDT.csv). Mutually exclusive with --patch_file",
    )
    parser.add_argument(
        "--patch_file",
        type=str,
        default=None,
        help="Path to pre-generated patch text file (e.g., patches/train/ADAUSDT_patches.txt). Mutually exclusive with --csv",
    )
    parser.add_argument(
        "--num_input",
        type=int,
        default=3,
        help="Number of past patches to use as input (default: 3)",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=2,
        help="Number of patches to predict iteratively (default: 2)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Starting patch index. If not specified, uses the last M patches",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size for extracting patches (default: 16, only used with --csv)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for extracting patches (default: 1, only used with --csv)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'gpt-4.1-mini-2025-04-14', 'gemini-2.0-flash')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature (default: 0.0)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save responses. If not specified, only prints to console",
    )
    parser.add_argument(
        "--save_prompt",
        action="store_true",
        help="Also save the generated prompts to files",
    )
    parser.add_argument(
        "--restrict_to_prompt",
        action="store_true",
        help="Add a system instruction to forbid use of external knowledge (GPT/Gemini)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.csv and args.patch_file:
        print("Error: --csv and --patch_file are mutually exclusive. Use only one.")
        return
    
    if not args.csv and not args.patch_file:
        print("Error: Either --csv or --patch_file must be specified.")
        return

    print(f"Using model: {args.model}")
    print("-" * 80)

    # Load patches
    try:
        patches = load_patches_from_source(
            csv_path=args.csv,
            patch_file_path=args.patch_file,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        print(f"Total patches loaded: {len(patches)}")
        if len(patches) == 0:
            print("Error: No patches found")
            return
    except Exception as e:
        print(f"Error loading patches: {e}")
        return

    print(f"Input patches: {args.num_input}, Predict: {args.num_predict}")
    if args.start_index is not None:
        print(f"Start index: {args.start_index}")
    print("-" * 80)

    # Determine file stem for output
    if args.patch_file:
        file_stem = Path(args.patch_file).stem.replace("_patches", "")
    elif args.csv:
        file_stem = Path(args.csv).stem
    else:
        file_stem = None

    # Run forecast with feedback
    print("Running iterative forecast with feedback...")
    try:
        prompts, responses = run_m2n_forecast_with_feedback(
            patches=patches,
            num_input_patches=args.num_input,
            num_predictions=args.num_predict,
            model_name=args.model,
            start_index=args.start_index,
            temperature=args.temperature,
            restrict_to_prompt=args.restrict_to_prompt,
            output_dir=args.output_dir,
            save_prompt=args.save_prompt,
            file_stem=file_stem,
        )
        print(f"\nCompleted {len(responses)} iterations")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Display all responses
    for idx, response in enumerate(responses):
        print("\n" + "=" * 80)
        print(f"ITERATION {idx+1} RESPONSE:")
        print("=" * 80)
        print(response)
        print("=" * 80)

    # Print save locations if output directory was specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        model_safe = args.model.replace("-", "_").replace(".", "_")
        print(f"\nSaved {len(responses)} responses to: {output_dir}")
        
        if args.save_prompt:
            print(f"Saved {len(prompts)} prompts to: {output_dir}")


if __name__ == "__main__":
    main()

