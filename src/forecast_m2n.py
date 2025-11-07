"""M2N forecasting: Generate prompts and get LLM responses for many-to-many row prediction."""

from __future__ import annotations
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from utils.prompt import create_m2n_prompt
from utils.rows import load_rows_from_csv

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
    if model_lower.startswith("gpt-"):
        return ChatOpenAI(model_name=model_name, **kwargs)
    elif model_lower.startswith("gemini-"):
        return ChatGoogleGenerativeAI(model=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Should start with 'gpt-', 'gemini-'")


def call_model(model, prompt: str) -> str:
    """Call the language model with a prompt and return the response.
    
    Args:
        model: LangChain chat model instance
        prompt: Prompt text
    
    Returns:
        Model response as string
    """
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    return response.content


def load_rows_from_source(
    csv_path: str | Path | None = None,
    base_dir: Path | None = None,
) -> list[str]:
    """Load rows from CSV file.
    
    Args:
        csv_path: Path to CSV file
        base_dir: Base directory for resolving relative paths. If None, uses project root.
    
    Returns:
        List of row strings, each in format "date,val1,val2,..."
    
    Raises:
        ValueError: If csv_path is not provided
        FileNotFoundError: If the specified file doesn't exist
    """
    if not csv_path:
        raise ValueError("csv_path must be specified.")
    
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    csv_file = Path(csv_path)
    if not csv_file.is_absolute():
        csv_file = base_dir / csv_file
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    return load_rows_from_csv(csv_file)


def run_m2n_forecast(
    rows: list[str],
    num_input_rows: int,
    num_predict_rows: int,
    model_name: str,
    start_index: int | None = None,
    temperature: float = 0.0,
    output_dir: Path | str | None = None,
    save_prompt: bool = False,
    file_stem: str | None = None,
) -> tuple[str, str]:
    """Run M2N forecasting: generate prompt, call model, and optionally save results.
    
    Args:
        rows: List of row strings, each in format "date,val1,val2,..."
        num_input_rows: Number of past rows to use as input (M)
        num_predict_rows: Number of rows to predict (N)
        model_name: Model name (e.g., 'gpt-4.1-mini-2025-04-14', 'gemini-2.0-flash')
        start_index: Optional starting row index. If None, uses the last M rows
        temperature: Model temperature (default: 0.0)
        output_dir: Directory to save response and prompt. If None, only returns results
        save_prompt: Whether to save the generated prompt to file
        file_stem: File stem for saved files. If None, auto-generated
    
    Returns:
        Tuple of (prompt, response) strings
    """
    # Generate prompt
    import sys
    print(f"[Generating prompt...]", file=sys.stderr, flush=True)
    prompt = create_m2n_prompt(
        all_rows=rows,
        num_input_rows=num_input_rows,
        num_predict_rows=num_predict_rows,
        start_index=start_index,
    )
    print(f"[✓ Prompt generated: {len(prompt)} characters]", file=sys.stderr, flush=True)
    
    # Initialize model
    print(f"[Initializing model: {model_name}...]", file=sys.stderr, flush=True)
    model = get_model(model_name=model_name, temperature=temperature)
    print(f"[✓ Model initialized]", file=sys.stderr, flush=True)
    
    # Call model
    print(f"[Calling model API... (this may take a while)]", file=sys.stderr, flush=True)
    response = call_model(model, prompt)
    print(f"[✓ Response received: {len(response)} characters]", file=sys.stderr, flush=True)
    
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

