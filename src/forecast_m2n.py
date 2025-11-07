"""M2N forecasting: Generate prompts and get LLM responses for many-to-many patch prediction."""

from __future__ import annotations
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from utils.prompt import create_m2n_prompt
from utils.patches import extract_patches_from_csv, load_patches_from_txt

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


def load_patches_from_source(
    csv_path: str | Path | None = None,
    patch_file_path: str | Path | None = None,
    patch_size: int = 16,
    stride: int = 1,
    base_dir: Path | None = None,
) -> list[list[str]]:
    """Load patches from either CSV file or patch text file.
    
    Args:
        csv_path: Path to CSV file (e.g., dataset/train/ADAUSDT.csv)
        patch_file_path: Path to pre-generated patch text file (e.g., patches/train/ADAUSDT_patches.txt)
        patch_size: Patch size for extracting patches (only used with csv_path)
        stride: Stride for extracting patches (only used with csv_path)
        base_dir: Base directory for resolving relative paths. If None, uses project root.
    
    Returns:
        List of patches, each patch is a list of strings (date,open,high,low,close)
    
    Raises:
        ValueError: If both or neither csv_path and patch_file_path are provided
        FileNotFoundError: If the specified file doesn't exist
    """
    if csv_path and patch_file_path:
        raise ValueError("csv_path and patch_file_path are mutually exclusive. Use only one.")
    
    if not csv_path and not patch_file_path:
        raise ValueError("Either csv_path or patch_file_path must be specified.")
    
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    if patch_file_path:
        patch_file = Path(patch_file_path)
        if not patch_file.is_absolute():
            patch_file = base_dir / patch_file
        
        if not patch_file.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_file}")
        
        return load_patches_from_txt(patch_file)
    else:
        csv_file = Path(csv_path)
        if not csv_file.is_absolute():
            csv_file = base_dir / csv_file
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        return extract_patches_from_csv(csv_file, patch_size, stride)


def run_m2n_forecast(
    patches: list[list[str]],
    num_input_patches: int,
    num_predictions: int,
    model_name: str,
    start_index: int | None = None,
    temperature: float = 0.0,
    restrict_to_prompt: bool = False,
    output_dir: Path | str | None = None,
    save_prompt: bool = False,
    file_stem: str | None = None,
) -> tuple[str, str]:
    """Run M2N forecasting: generate prompt, call model, and optionally save results.
    
    Args:
        patches: List of patches, each patch is a list of strings (date,open,high,low,close)
        num_input_patches: Number of past patches to use as input (M)
        num_predictions: Number of patches to predict (N)
        model_name: Model name (e.g., 'gpt-4.1-mini-2025-04-14', 'gemini-2.0-flash')
        start_index: Optional starting patch index. If None, uses the last M patches
        temperature: Model temperature (default: 0.0)
        restrict_to_prompt: If True, add system instruction to forbid external knowledge
        output_dir: Directory to save response and prompt. If None, only returns results
        save_prompt: Whether to save the generated prompt to file
        file_stem: File stem for saved files. If None, auto-generated
    
    Returns:
        Tuple of (prompt, response) strings
    """
    # Generate prompt
    prompt = create_m2n_prompt(
        all_patches=patches,
        num_input_patches=num_input_patches,
        num_predictions=num_predictions,
        start_index=start_index,
    )
    
    # Initialize model
    model = get_model(model_name=model_name, temperature=temperature)
    
    # Call model
    response = call_model(model, prompt, restrict_to_prompt=restrict_to_prompt)
    
    # Save results if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if file_stem is None:
            file_stem = "forecast"
        
        # Save response (sanitize model name for filename)
        model_safe = model_name.replace("-", "_").replace(".", "_")
        response_file = output_dir / f"{file_stem}_response_{model_safe}.txt"
        response_file.write_text(response, encoding="utf-8")
        
        # Save prompt if requested
        if save_prompt:
            prompt_file = output_dir / f"{file_stem}_prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
    
    return prompt, response

