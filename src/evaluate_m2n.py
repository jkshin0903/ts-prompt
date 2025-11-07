"""Evaluate M2N forecasting by computing MSE/MAE metrics between predicted and actual rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast_m2n import load_rows_from_source, run_m2n_forecast
from utils.rows import (
    parse_rows_from_response,
    rows_string_to_dict_array,
    get_csv_column_names,
)


def get_feature_keys(rows: List[dict[str, str]]) -> List[str]:
    """Extract feature column names (excluding Date) from rows."""
    if not rows:
        return []
    
    # Get all keys except Date
    feature_keys = [key for key in rows[0].keys() if key.lower() not in ["date", "time", "timestamp"]]
    return feature_keys


def convert_rows_to_floats(rows: List[dict[str, str]]) -> List[dict[str, float]]:
    """Convert string values to floats for numeric columns."""
    numeric_rows: List[dict[str, float]] = []
    for row in rows:
        numeric_row: dict[str, float] = {}
        for key, value in row.items():
            if key.lower() in ["date", "time", "timestamp"]:
                # Preserve date as-is (string) but skip from metrics
                continue
            try:
                numeric_row[key] = float(value)
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue
        numeric_rows.append(numeric_row)
    return numeric_rows


def compute_row_metrics(
    predicted_rows: List[dict[str, float]],
    actual_rows: List[dict[str, float]],
    feature_keys: List[str],
) -> Tuple[float, float]:
    """Compute MSE and MAE for rows."""
    squared_errors: List[float] = []
    absolute_errors: List[float] = []

    for pred_row, actual_row in zip(predicted_rows, actual_rows):
        for key in feature_keys:
            if key not in pred_row or key not in actual_row:
                continue
            diff = pred_row[key] - actual_row[key]
            squared_errors.append(diff * diff)
            absolute_errors.append(abs(diff))

    if not squared_errors:
        return float("nan"), float("nan")

    mse = sum(squared_errors) / len(squared_errors)
    mae = sum(absolute_errors) / len(absolute_errors)
    return mse, mae


def determine_prediction_start_index(
    num_input_rows: int,
    start_index: int | None,
    total_rows: int,
) -> int:
    """Determine the starting row index for predictions."""
    if start_index is None:
        if total_rows < num_input_rows:
            raise ValueError(
                f"Not enough rows. Need at least {num_input_rows}, got {total_rows}"
            )
        return total_rows

    if start_index + num_input_rows > total_rows:
        raise ValueError(
            f"Not enough rows from index {start_index}. "
            f"Need {num_input_rows} rows, but only {total_rows - start_index} available."
        )
    return start_index + num_input_rows


def evaluate_m2n(
    rows: List[str],
    response: str,
    num_input_rows: int,
    num_predict_rows: int,
    start_index: int | None,
    csv_path: str | Path | None = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute overall metrics for model response.
    
    Args:
        rows: List of all row strings
        response: Model response string containing predicted rows
        num_input_rows: Number of input rows used
        num_predict_rows: Number of rows predicted
        start_index: Starting row index for input (None means last M rows)
        csv_path: Optional CSV file path to extract column names from
    
    Returns:
        Tuple of (overall_metrics, overall_metrics) - kept for compatibility
        - overall_metrics: Tuple of (overall_mse, overall_mae)
    """
    # Determine prediction start index
    prediction_start_index = determine_prediction_start_index(
        num_input_rows=num_input_rows,
        start_index=start_index,
        total_rows=len(rows),
    )
    
    # Check if we have enough actual rows
    if prediction_start_index + num_predict_rows > len(rows):
        raise ValueError(
            f"Not enough rows for evaluation. Need {num_predict_rows} rows starting from index {prediction_start_index}, "
            f"but only {len(rows) - prediction_start_index} available."
        )
    
    # Determine column names
    column_names: List[str] | None = None
    if csv_path:
        try:
            column_names = get_csv_column_names(csv_path)
        except Exception:
            # Fall back to auto-detection if CSV reading fails
            column_names = None
    
    # Extract predicted rows from response
    predicted_row_strings = parse_rows_from_response(response)
    
    # Limit to expected number of rows
    if len(predicted_row_strings) > num_predict_rows:
        predicted_row_strings = predicted_row_strings[:num_predict_rows]
    elif len(predicted_row_strings) < num_predict_rows:
        raise ValueError(
            f"Not enough predicted rows. Expected {num_predict_rows}, got {len(predicted_row_strings)}"
        )
    
    # Get actual rows
    actual_rows_str = rows[prediction_start_index:prediction_start_index + num_predict_rows]
    
    # Determine feature keys from first actual row
    if not actual_rows_str:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    
    first_actual_row_str = actual_rows_str[0]
    first_actual_rows_raw = rows_string_to_dict_array(first_actual_row_str, column_names=column_names)
    if first_actual_rows_raw:
        feature_keys = get_feature_keys(first_actual_rows_raw)
        if not column_names:
            column_names = list(first_actual_rows_raw[0].keys())
    else:
        raise ValueError("Could not parse first actual row")
    
    if not feature_keys:
        raise ValueError("No feature columns found in rows")
    
    all_squared_errors: List[float] = []
    all_absolute_errors: List[float] = []
    
    # Parse and compare each row
    for i in range(num_predict_rows):
        predicted_row_str = predicted_row_strings[i]
        actual_row_str = actual_rows_str[i]
        
        # Parse rows (each is a single row string)
        predicted_rows_raw = rows_string_to_dict_array(predicted_row_str, column_names=column_names)
        actual_rows_raw = rows_string_to_dict_array(actual_row_str, column_names=column_names)
        
        if not predicted_rows_raw or not actual_rows_raw:
            continue
        
        # Convert to numeric (each should have exactly one row)
        predicted_row = convert_rows_to_floats(predicted_rows_raw)[0] if predicted_rows_raw else {}
        actual_row = convert_rows_to_floats(actual_rows_raw)[0] if actual_rows_raw else {}
        
        # Accumulate errors for overall metrics
        for key in feature_keys:
            if key not in predicted_row or key not in actual_row:
                continue
            diff = predicted_row[key] - actual_row[key]
            all_squared_errors.append(diff * diff)
            all_absolute_errors.append(abs(diff))
    
    if all_squared_errors:
        overall_mse = sum(all_squared_errors) / len(all_squared_errors)
        overall_mae = sum(all_absolute_errors) / len(all_absolute_errors)
    else:
        overall_mse = float("nan")
        overall_mae = float("nan")
    
    # Return tuple for compatibility (per_row_metrics, overall_metrics)
    return (overall_mse, overall_mae), (overall_mse, overall_mae)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate M2N forecasting by computing MSE/MAE metrics",
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file",
    )
    parser.add_argument(
        "--num_input",
        type=int,
        default=30,
        help="Number of past rows to use as input (default: 30)",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=30,
        help="Number of rows to predict (default: 30)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Starting row index. If not specified, uses the last M rows",
    )
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
    parser.add_argument(
        "--response_file",
        type=str,
        default=None,
        help="Path to saved model response file. If provided, uses this instead of calling model",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If provided, automatically determines response_file and metrics_output paths",
    )
    parser.add_argument(
        "--auto_response",
        action="store_true",
        help="Automatically find or generate response file based on csv, model, and num_input/num_predict",
    )

    args = parser.parse_args()

    print(f"Using model: {args.model}")
    print("-" * 80)

    # Load rows
    try:
        rows = load_rows_from_source(csv_path=args.csv)
        print(f"Total rows loaded: {len(rows)}")
        if len(rows) == 0:
            print("Error: No rows found")
            return
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error loading rows: {exc}")
        return

    print(f"Input rows: {args.num_input}, Predict: {args.num_predict}")
    if args.start_index is not None:
        print(f"Start index: {args.start_index}")
    print("-" * 80)

    # Auto-determine response file and metrics output if output_dir is provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine dataset name from csv
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = Path(__file__).parent.parent / csv_path
        dataset_name = csv_path.stem
        
        # Sanitize model name for filename
        model_safe = args.model.replace("-", "_").replace(".", "_")
        
        # Auto-determine response file if not provided
        if not args.response_file:
            response_file = output_dir / f"{dataset_name}_response_{model_safe}.txt"
            args.response_file = str(response_file)
        
        # Auto-determine metrics output if not provided
        if not args.metrics_output:
            args.metrics_output = str(output_dir / "metrics.json")
    
    # Auto-determine response file from csv if auto_response is enabled
    if args.auto_response and not args.response_file:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = Path(__file__).parent.parent / csv_path
        
        dataset_name = csv_path.stem
        model_safe = args.model.replace("-", "_").replace(".", "_")
        
        # Construct output directory similar to run_forecast_m2n.sh
        output_dir = Path(__file__).parent.parent / "responses" / dataset_name / f"{model_safe}_{args.num_input}_{args.num_predict}"
        response_file = output_dir / f"{dataset_name}_response_{model_safe}.txt"
        args.response_file = str(response_file)
        
        # Also set metrics output
        if not args.metrics_output:
            args.metrics_output = str(output_dir / "metrics.json")

    # Get model response
    if args.response_file:
        # Load from file
        response_file = Path(args.response_file)
        if not response_file.exists():
            print(f"Warning: Response file not found: {response_file}")
            print("Running forecast to generate response...")
            # Run forecast and save response
            try:
                # Determine output directory from response file path
                response_output_dir = response_file.parent
                file_stem = response_file.stem.replace(f"_response_{args.model.replace('-', '_').replace('.', '_')}", "")
                
                _, response = run_m2n_forecast(
                    rows=rows,
                    num_input_rows=args.num_input,
                    num_predict_rows=args.num_predict,
                    model_name=args.model,
                    start_index=args.start_index,
                    temperature=args.temperature,
                    output_dir=response_output_dir,
                    save_prompt=True,
                    file_stem=file_stem,
                )
                print(f"Generated and saved response ({len(response)} characters)")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Error during forecasting: {exc}")
                return
        else:
            response = response_file.read_text(encoding="utf-8")
            print(f"Loaded response from: {response_file}")
    else:
        # Run forecast
        print("Running forecast...")
        try:
            _, response = run_m2n_forecast(
                rows=rows,
                num_input_rows=args.num_input,
                num_predict_rows=args.num_predict,
                model_name=args.model,
                start_index=args.start_index,
                temperature=args.temperature,
                output_dir=None,
                save_prompt=False,
                file_stem=None,
            )
            print(f"Received response ({len(response)} characters)")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error during forecasting: {exc}")
            return

    # Compute metrics
    csv_path_for_columns = args.csv
    if csv_path_for_columns and not Path(csv_path_for_columns).is_absolute():
        csv_path_for_columns = Path(__file__).parent.parent / csv_path_for_columns
    
    try:
        _, overall_metrics = evaluate_m2n(
            rows=rows,
            response=response,
            num_input_rows=args.num_input,
            num_predict_rows=args.num_predict,
            start_index=args.start_index,
            csv_path=csv_path_for_columns,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error during evaluation: {exc}")
        return

    overall_mse, overall_mae = overall_metrics

    print("\nOverall metrics:")
    print(f"  Overall MSE: {overall_mse:.6f}")
    print(f"  Overall MAE: {overall_mae:.6f}")

    # Save metrics to JSON (always save if metrics_output is set, or auto-save if output_dir is set)
    if args.metrics_output:
        metrics_output = Path(args.metrics_output)
        metrics_output.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "overall": {
                "mse": overall_mse,
                "mae": overall_mae,
            },
        }

        metrics_output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nSaved metrics to: {metrics_output}")
    elif args.output_dir or args.auto_response:
        # Auto-save metrics if output_dir or auto_response is set
        if args.output_dir:
            metrics_output = Path(args.output_dir) / "metrics.json"
        elif args.auto_response:
            csv_path = Path(args.csv)
            if not csv_path.is_absolute():
                csv_path = Path(__file__).parent.parent / csv_path
            dataset_name = csv_path.stem
            model_safe = args.model.replace("-", "_").replace(".", "_")
            output_dir = Path(__file__).parent.parent / "responses" / dataset_name / f"{model_safe}_{args.num_input}_{args.num_predict}"
            metrics_output = output_dir / "metrics.json"
        else:
            metrics_output = None
        
        if metrics_output:
            metrics_output.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "overall": {
                    "mse": overall_mse,
                    "mae": overall_mae,
                },
            }
            metrics_output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"\nSaved metrics to: {metrics_output}")


if __name__ == "__main__":
    main()

