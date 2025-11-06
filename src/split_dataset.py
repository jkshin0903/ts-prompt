from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required to run this script. Install with `pip install pandas`."
    ) from exc


def find_dataset_files(dataset_dir: Path) -> Iterable[Path]:
    """Yield CSV files matching the expected Binance naming convention.

    Expected file name pattern: "Binance_{SYMBOL}_d.csv"
    """
    pattern = re.compile(r"^Binance_(?P<symbol>.+?)_d\.csv$")
    for path in sorted(dataset_dir.glob("Binance_*_d.csv")):
        if pattern.match(path.name):
            yield path


def extract_symbol_from_filename(filename: str) -> str:
    match = re.match(r"^Binance_(?P<symbol>.+?)_d\.csv$", filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return match.group("symbol")


def split_dataframe_chronologically(df: "pd.DataFrame", train_ratio: float) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    n_rows = len(df)
    if n_rows == 0:
        return df.copy(), df.copy()

    split_index = int(n_rows * train_ratio)
    # Ensure at least one row in test set when there are 2+ rows
    if split_index >= n_rows and n_rows > 1:
        split_index = n_rows - 1

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset" / "original"

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    train_ratio = 0.7

    # Ensure output directories exist
    train_dir = project_root / "dataset" / "train"
    test_dir = project_root / "dataset" / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in find_dataset_files(dataset_dir):
        symbol = extract_symbol_from_filename(csv_path.name)

        # Use the second row (index 1) as header; data begins from the third row
        df = pd.read_csv(csv_path, header=1)
        # Ensure chronological order (oldest -> newest) before splitting
        if "Date" in df.columns:
            df = df.sort_values(
                "Date", key=lambda s: pd.to_datetime(s, errors="coerce"), ascending=True
            ).reset_index(drop=True)

        train_df, test_df = split_dataframe_chronologically(df, train_ratio)

        train_out = train_dir / f"{symbol}.csv"
        test_out = test_dir / f"{symbol}.csv"

        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)

        print(f"Saved: {train_out.name} ({len(train_df)} rows), {test_out.name} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()


