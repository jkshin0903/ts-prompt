from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required to run this script. Install with `pip install pandas`."
    ) from exc


def find_csv_files(dataset_dir: Path) -> Iterable[Tuple[Path, Path]]:
    """Find all CSV files in dataset directory and its subdirectories.
    
    Yields tuples of (csv_path, relative_path_from_dataset_dir).
    
    Args:
        dataset_dir: Base dataset directory (e.g., project_root / "dataset")
    
    Yields:
        Tuple of (absolute_csv_path, relative_path_from_dataset_dir)
    """
    if not dataset_dir.exists():
        return
    
    for csv_path in dataset_dir.rglob("*.csv"):
        relative = csv_path.relative_to(dataset_dir)
        yield csv_path, relative


def detect_date_column(df: "pd.DataFrame") -> str | None:
    """Detect date column in dataframe (case-insensitive).
    
    Returns:
        Column name if found, None otherwise
    """
    for col in df.columns:
        if col.lower() in ["date", "time", "timestamp"]:
            return col
    return None


def read_csv_with_auto_header(csv_path: Path) -> "pd.DataFrame":
    """Read CSV file, automatically detecting header row.
    
    Tries header=0 first, then header=1 if first row looks like data.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with properly parsed data
    """
    # Try header=0 first (standard format)
    try:
        df = pd.read_csv(csv_path, header=0)
        # Check if first row looks like data (contains numeric values)
        if len(df) > 0:
            first_row = df.iloc[0]
            # If first row contains mostly numeric values, it's likely data, not header
            numeric_count = sum(
                1 for val in first_row 
                if pd.api.types.is_numeric_dtype(type(val)) 
                or (isinstance(val, str) and val.replace(".", "").replace("-", "").replace("/", "").isdigit())
            )
            if numeric_count < len(first_row) * 0.5:
                # Less than half are numeric, likely header row
                return df
    except Exception:
        pass
    
    # Try header=1 (some datasets have metadata in first row)
    try:
        df = pd.read_csv(csv_path, header=1)
        return df
    except Exception:
        # Fall back to header=0
        return pd.read_csv(csv_path, header=0)


def main() -> None:
    """Process CSV files in dataset directory: sort by date and optionally clean data.
    
    This script processes all CSV files in the dataset directory, sorts them chronologically,
    and optionally performs data cleaning. No train/test split is performed.
    """
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    processed_count = 0
    for csv_path, relative_path in find_csv_files(dataset_dir):
        try:
            # Read CSV with auto header detection
            df = read_csv_with_auto_header(csv_path)
            
            if len(df) == 0:
                print(f"Skipping {relative_path}: empty file")
                continue
            
            # Detect and sort by date column
            date_col = detect_date_column(df)
            if date_col:
                df = df.sort_values(
                    date_col, key=lambda s: pd.to_datetime(s, errors="coerce"), ascending=True
                ).reset_index(drop=True)
                print(f"Processed: {relative_path} ({len(df)} rows, sorted by {date_col})")
            else:
                # If no date column, assume data is already in chronological order
                print(f"Processed: {relative_path} ({len(df)} rows, no date column found)")
            
            processed_count += 1
            
        except Exception as exc:
            print(f"Error processing {relative_path}: {exc}")
            continue
    
    if processed_count == 0:
        print("No CSV files found to process")
    else:
        print(f"\nProcessed {processed_count} dataset(s)")


if __name__ == "__main__":
    main()


