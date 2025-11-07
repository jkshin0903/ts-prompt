from __future__ import annotations

import argparse
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Use local imports so this file can run as a script
from extract_patches import extract_patches_from_csv
from instruct_forcasting import (
    create_forecast_prompt,
    load_patches_from_txt,
)

load_dotenv()

def get_model(model_name: str, **kwargs):
    """Initialize and return a LangChain chat model.
    
    Args:
        model_name: Full model name (e.g., "gpt-4.1-mini-2025-04-14", "gemini-2.0-flash")
        **kwargs: Additional model parameters
    
    Returns:
        LangChain chat model instance
    
    Note:
        API keys should be set via environment variables:
        - OPENAI_API_KEY for GPT models
        - GOOGLE_API_KEY for Gemini models
    """
    model_lower = model_name.lower()
    if model_lower.startswith("gpt-") or model_lower.startswith("o1-") or model_lower.startswith("o3-"):
        return ChatOpenAI(model_name=model_name, **kwargs)
    elif model_lower.startswith("gemini-"):
        return ChatGoogleGenerativeAI(model=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Should start with 'gpt-', 'o1-', 'o3-', or 'gemini-'")


def call_model(model, prompt: str, restrict_to_prompt: bool = False) -> str:
    """Call the language model with a prompt and return the response.
    
    Args:
        model: LangChain chat model instance
        prompt: Prompt text
        restrict_to_prompt: If True, prepend a system message instructing the model
            to use only the provided content without external/world knowledge.
    
    Returns:
        Model response as string
    """
    messages = []
    if restrict_to_prompt:
        messages.append(SystemMessage(content=(
            "You are not allowed to use any external or world knowledge. "
            "Rely ONLY on the provided input content to generate the answer. "
            "If information is insufficient, output only what is derivable from the input."
        )))
    messages.append(HumanMessage(content=prompt))
    response = model.invoke(messages)
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Generate forecasting prompts and get LLM responses")
    
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
        help="Number of patches to predict (default: 2)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Starting patch index. If not specified, uses the last N patches",
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
        help="Directory to save response. If not specified, only prints to console",
    )
    parser.add_argument(
        "--save_prompt",
        action="store_true",
        help="Also save the generated prompt to file",
    )
    parser.add_argument(
        "--restrict_to_prompt",
        action="store_true",
        help="Add a system instruction to forbid use of external knowledge (GPT/Gemini)",
    )

    args = parser.parse_args()

    # Validate prompt generation arguments
    if args.csv and args.patch_file:
        print("Error: --csv and --patch_file are mutually exclusive. Use only one.")
        return
    
    if not args.csv and not args.patch_file:
        print("Error: Either --csv or --patch_file must be specified.")
        return

    print(f"Using model: {args.model}")
    print("-" * 80)

    # Load patches
    if args.patch_file:
        patch_file_path = Path(args.patch_file)
        if not patch_file_path.is_absolute():
            # Resolve relative to project root (one level above src/)
            patch_file_path = Path(__file__).parent.parent / patch_file_path
        
        if not patch_file_path.exists():
            print(f"Error: Patch file not found: {patch_file_path}")
            return
        
        print(f"Loading patches from: {patch_file_path}")
        try:
            patches = load_patches_from_txt(patch_file_path)
            print(f"Total patches loaded: {len(patches)}")
            if len(patches) == 0:
                print("Error: No patches found in patch file")
                return
        except Exception as e:
            print(f"Error loading patches: {e}")
            return
    else:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            # Resolve relative to project root (one level above src/)
            csv_path = Path(__file__).parent.parent / csv_path

        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            return

        print(f"Loading patches from CSV: {csv_path}")
        print(f"Patch size: {args.patch_size}, Stride: {args.stride}")
        try:
            patches = extract_patches_from_csv(csv_path, args.patch_size, args.stride)
            print(f"Total patches extracted: {len(patches)}")
            if len(patches) == 0:
                print("Error: No patches extracted from CSV file")
                return
        except Exception as e:
            print(f"Error extracting patches: {e}")
            return

    print(f"Input patches: {args.num_input}, Predict: {args.num_predict}")
    if args.start_index is not None:
        print(f"Start index: {args.start_index}")
    print("-" * 80)

    # Generate prompt
    print("Generating prompt...")
    try:
        prompt = create_forecast_prompt(
            all_patches=patches,
            num_input_patches=args.num_input,
            num_predictions=args.num_predict,
            start_index=args.start_index,
        )
        print(f"Prompt generated ({len(prompt)} characters)")
    except Exception as e:
        print(f"Error generating prompt: {e}")
        return

    # Initialize model
    print(f"Initializing model: {args.model}...")
    try:
        model = get_model(
            model_name=args.model,
            temperature=args.temperature,
        )
        print("Model initialized")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Call model
    print("Calling model API...")
    try:
        response = call_model(model, prompt, restrict_to_prompt=args.restrict_to_prompt)
        print(f"Response received ({len(response)} characters)")
    except Exception as e:
        print(f"Error calling model: {e}")
        return

    # Display response
    print("\n" + "=" * 80)
    print("MODEL RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80)

    # Save response if output directory is specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine file stem
        if args.patch_file:
            file_stem = Path(args.patch_file).stem.replace("_patches", "")
        else:
            file_stem = csv_path.stem
        
        # Save response (sanitize model name for filename)
        model_safe = args.model.replace("-", "_").replace(".", "_")
        response_file = output_dir / f"{file_stem}_response_{model_safe}.txt"
        response_file.write_text(response, encoding="utf-8")
        print(f"\nSaved response to: {response_file}")

        # Save prompt if requested
        if args.save_prompt:
            prompt_file = output_dir / f"{file_stem}_prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
            print(f"Saved prompt to: {prompt_file}")


if __name__ == "__main__":
    main()
    