"""Utility functions for saving forecast prompts."""

from __future__ import annotations

from pathlib import Path


def save_forecast_prompt(
    prompt: str,
    output_path: Path | str,
    encoding: str = "utf-8",
) -> None:
    """Save the forecast prompt to a file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding=encoding)

