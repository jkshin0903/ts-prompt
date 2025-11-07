"""Utility functions for loading and working with time series rows (not patches)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required to run this script. Install with `pip install pandas`."
    ) from exc


def get_csv_column_names(csv_path: str | Path) -> List[str]:
    """Extract column names from CSV file.
    
    Returns:
        List of column names in order: [date_col, feature1, feature2, ...]
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Detect date column
    date_col = None
    for col in df.columns:
        if col.lower() in ["date", "time", "timestamp"]:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError(f"No date column found in {csv_path.name}")
    
    # date + all other numeric columns
    feature_cols = [date_col] + [col for col in df.columns if col != date_col and pd.api.types.is_numeric_dtype(df[col])]
    if len(feature_cols) < 2:
        raise ValueError(f"Not enough feature columns found in {csv_path.name}. Need at least one numeric column besides date.")
    
    return feature_cols


def load_rows_from_csv(csv_path: str | Path) -> List[str]:
    """Load all rows from CSV file as strings.
    
    Each row is formatted as: "date,val1,val2,..."
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of row strings, each in format "date,val1,val2,..."
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Normalize column names
    df.columns = df.columns.str.strip()
    
    # Detect date column
    date_col = None
    for col in df.columns:
        if col.lower() in ["date", "time", "timestamp"]:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError(f"No date column found in {csv_path.name}. Expected 'date', 'Date', 'time', or 'timestamp'")

    # date + all other numeric columns
    feature_cols = [date_col] + [col for col in df.columns if col != date_col and pd.api.types.is_numeric_dtype(df[col])]
    if len(feature_cols) < 2:
        raise ValueError(f"Not enough feature columns found in {csv_path.name}. Need at least one numeric column besides date.")
    
    # Select and clean data
    df = df[feature_cols].dropna().reset_index(drop=True)
    
    # Sort by date if possible
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)
        df[date_col] = df[date_col].astype(str)  # Convert back to string
    except Exception:
        # If date parsing fails, keep original order
        pass
    
    # Convert each row to string format
    rows: List[str] = []
    for row in df.itertuples(index=False):
        row_values = [str(getattr(row, col)) for col in feature_cols]
        rows.append(",".join(row_values))
    
    return rows


def format_rows(rows: List[str]) -> str:
    """Format a list of row strings into a single string.
    
    Args:
        rows: List of row strings, each in format "date,val1,val2,..."
    
    Returns:
        Formatted string with one row per line
    """
    return "\n".join(rows)


def parse_rows_from_response(response: str) -> List[str]:
    """Parse rows from model response string.
    
    Extracts data rows from response, ignoring any headers or explanations.
    Each line that looks like a data row (contains comma-separated values) is included.
    
    Args:
        response: Model response string
    
    Returns:
        List of row strings, each in format "date,val1,val2,..."
    """
    rows: List[str] = []
    
    for line in response.splitlines():
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip patch headers (if any remain)
        if line.startswith("===== Patch") and line.endswith("====="):
            continue
        
        # Skip lines that look like explanations or comments
        if line.startswith("#") or line.startswith("//") or line.startswith("*"):
            continue
        
        # If line contains comma, treat as data row
        if "," in line:
            rows.append(line)
    
    return rows


def rows_string_to_dict_array(rows_string: str, column_names: List[str] | None = None) -> List[Dict[str, str]]:
    """Convert a string containing rows into an array of dictionaries.
    
    Args:
        rows_string: String containing rows, one per line, in format "date,val1,val2,..."
        column_names: Optional list of column names. If None, auto-detects:
            Uses ["Date"] + ["Feature1", "Feature2", ...] based on number of columns
    
    Returns:
        List of dictionaries, each with keys matching column_names.
        All values are strings.
    """
    from typing import Dict
    
    result: List[Dict[str, str]] = []
    
    # Auto-detect column names from first row if not provided
    if column_names is None:
        for line in rows_string.splitlines():
            line = line.strip()
            if line and "," in line:
                parts = line.split(",")
                num_cols = len(parts)
                column_names = ["Date"] + [f"Feature{i}" for i in range(1, num_cols)]
                break
    
    if column_names is None:
        raise ValueError("Could not determine column names from rows string")
    
    for line in rows_string.splitlines():
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Parse data row
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(column_names):
            # Skip malformed lines
            continue
        
        row_dict = {col: val for col, val in zip(column_names, parts)}
        result.append(row_dict)
    
    return result

