from __future__ import annotations

from pathlib import Path
from typing import List, Dict

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required to run this script. Install with `pip install pandas`."
    ) from exc


def extract_patches_from_csv(csv_path: str | Path, patch_size: int, stride: int) -> List[List[str]]:
    """Read a train CSV and return sliding-window patches of OHLC strings.

    - Each patch is a list of length `patch_size`.
    - Each element in a patch is a string: "Open,High,Low,Close" from that row.
    - Windows advance by `stride` rows.
    - Only rows with all OHLC values present are considered.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["Date", "Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {', '.join(missing)}"
        )

    df = df[required_cols].dropna().reset_index(drop=True)

    num_rows = len(df)
    if num_rows < patch_size:
        return []

    patches: List[List[str]] = []
    start = 0
    last_start = num_rows - patch_size
    while start <= last_start:
        window = df.iloc[start : start + patch_size]
        patch: List[str] = [
            f"{row.Date},{row.Open},{row.High},{row.Low},{row.Close}"
            for row in window.itertuples(index=False)
        ]
        patches.append(patch)
        start += stride

    return patches


def extract_patches_by_symbol(
    symbol: str,
    patch_size: int,
    stride: int,
    base_dir: str | Path = ".",
    split: str = "train",
) -> List[List[str]]:
    """Convenience wrapper to read from dataset/{split}/{SYMBOL}.csv.

    Example: symbol "ADAUSDT", split "test" -> dataset/test/ADAUSDT.csv
    """
    base_dir = Path(base_dir)
    csv_path = base_dir / "dataset" / split / f"{symbol}.csv"
    return extract_patches_from_csv(csv_path, patch_size, stride)


def generate_all_patches(
    base_dir: str | Path = ".",
    patch_size: int = 16,
    stride: int = 4,
    split: str = "train",
) -> Dict[str, List[List[str]]]:
    base_dir = Path(base_dir)
    data_dir = base_dir / "dataset" / split
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    symbols = [p.stem for p in sorted(data_dir.glob("*.csv"))]

    all_patches: Dict[str, List[List[str]]] = {}
    for symbol in symbols:
        patches = extract_patches_by_symbol(symbol, patch_size, stride, base_dir, split=split)
        all_patches[symbol] = patches
    return all_patches


def write_patches_to_txt(patches_by_symbol: Dict[str, List[List[str]]], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for symbol, patches in patches_by_symbol.items():
        file_path = output_path / f"{symbol}_patches.txt"
        with file_path.open("w", encoding="utf-8") as f:
            if not patches:
                f.write("(no patches)\n")
                continue
            for i, patch in enumerate(patches):
                f.write(f"===== Patch {i} =====\n")
                for row_str in patch:
                    f.write(f"{row_str}\n")
                f.write("\n")  # blank line between patches


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate patches from dataset and save to text files")
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split: train or test (default: train)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size (default: 16)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride (default: 1)",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Project root directory (default: script's parent directory)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for text files (default: patches/)",
    )

    args = parser.parse_args()

    # Determine base directory
    if args.base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(args.base_dir).resolve()

    # Determine output directory
    if args.out_dir is None:
        out_dir = base_dir / "patches"
    else:
        out_dir = Path(args.out_dir).resolve()

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate patches for all symbols
    patches = generate_all_patches(
        base_dir=base_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        split=args.split,
    )

    # Write patches to text files
    output_path = out_dir / args.split
    write_patches_to_txt(patches, output_path)
    print(f"Wrote patches for split '{args.split}' to: {output_path}")