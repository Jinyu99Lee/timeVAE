#!/usr/bin/env python3
"""Convert CSV time-series files to TimeVAE-compatible NPZ datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DROP_COLS = ("time", "date", "datetime", "timestamp")
DEFAULT_TIME_COLS = ("time", "date", "datetime", "timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one CSV file or every CSV file in a directory into "
            "TimeVAE .npz files with data shaped as (N, T, D)."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a CSV file or a directory containing CSV files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where converted .npz files will be saved.",
    )
    parser.add_argument(
        "--seq-len",
        "-T",
        type=int,
        required=True,
        help="Window length T for each sample.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Step between window starts. Default: 1 for fully overlapping crops.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, search for CSV files recursively.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "Optional inclusive start date/time used to filter rows before "
            "windowing, for example 1979-01-01 or '1979-01-01 06:00:00'."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help=(
            "Optional inclusive end date/time used to filter rows before "
            "windowing. A date-only value includes the whole end date."
        ),
    )
    parser.add_argument(
        "--time-col",
        default=None,
        help=(
            "Column used with --start-date/--end-date. By default, the first "
            "available column among time, date, datetime, timestamp is used."
        ),
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=None,
        help=(
            "Feature columns to use. By default, all numeric columns except "
            "common time columns are used."
        ),
    )
    parser.add_argument(
        "--drop-cols",
        nargs="+",
        default=list(DEFAULT_DROP_COLS),
        help=(
            "Columns to exclude when --feature-cols is not set. "
            "Default: time date datetime timestamp."
        ),
    )
    parser.add_argument(
        "--missing",
        choices=("error", "drop", "ffill", "bfill", "zero", "mean"),
        default="error",
        help="How to handle missing or non-numeric feature values. Default: error.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Floating dtype stored in the output data array. Default: float32.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended to each output filename before .npz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files.",
    )
    return parser.parse_args()


def collect_csv_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Input file is not a CSV: {input_path}")
        return [input_path]

    if input_path.is_dir():
        pattern = "**/*.csv" if recursive else "*.csv"
        csv_files = sorted(input_path.glob(pattern))
        if not csv_files:
            raise ValueError(f"No CSV files found in: {input_path}")
        return csv_files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def is_date_only(date_value: str) -> bool:
    normalized = date_value.strip()
    return " " not in normalized and "T" not in normalized


def find_time_column(
    df: pd.DataFrame,
    requested_time_col: str | None,
    csv_file: Path,
) -> str:
    if requested_time_col is not None:
        if requested_time_col not in df.columns:
            raise ValueError(f"{csv_file}: time column not found: {requested_time_col}")
        return requested_time_col

    columns_by_lower = {col.lower(): col for col in df.columns}
    for col in DEFAULT_TIME_COLS:
        if col in columns_by_lower:
            return columns_by_lower[col]

    raise ValueError(
        f"{csv_file}: --start-date/--end-date requires a time column. "
        "Use --time-col to specify one."
    )


def filter_by_date_range(
    df: pd.DataFrame,
    csv_file: Path,
    time_col: str | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, str | None]:
    if start_date is None and end_date is None:
        return df, None

    selected_time_col = find_time_column(df, time_col, csv_file)
    timestamps = pd.to_datetime(df[selected_time_col], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(
            f"{csv_file}: failed to parse some values in time column "
            f"{selected_time_col!r}."
        )

    start_timestamp = pd.to_datetime(start_date) if start_date is not None else None
    end_timestamp = pd.to_datetime(end_date) if end_date is not None else None
    if (
        start_timestamp is not None
        and end_timestamp is not None
        and start_timestamp > end_timestamp
    ):
        raise ValueError("--start-date must be earlier than or equal to --end-date.")

    mask = pd.Series(True, index=df.index)
    if start_timestamp is not None:
        mask &= timestamps >= start_timestamp
    if end_timestamp is not None:
        if is_date_only(end_date):
            mask &= timestamps < end_timestamp + pd.Timedelta(days=1)
        else:
            mask &= timestamps <= end_timestamp

    filtered_df = df.loc[mask].copy()
    if filtered_df.empty:
        raise ValueError(f"{csv_file}: no rows remain after date filtering.")

    return filtered_df, selected_time_col


def select_feature_frame(
    df: pd.DataFrame,
    csv_file: Path,
    feature_cols: list[str] | None,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:

    if feature_cols is not None:
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{csv_file}: requested feature columns not found: {missing_cols}"
            )
        selected_cols = feature_cols
    else:
        drop_col_set = {col.lower() for col in drop_cols}
        candidate_cols = [
            col for col in df.columns if col.lower() not in drop_col_set
        ]
        selected_cols = list(
            df[candidate_cols].select_dtypes(include=[np.number]).columns
        )

    if not selected_cols:
        raise ValueError(
            f"{csv_file}: no numeric feature columns found. "
            "Use --feature-cols to select columns explicitly."
        )

    feature_frame = df[selected_cols].apply(pd.to_numeric, errors="coerce")
    return feature_frame, selected_cols


def handle_missing_values(frame: pd.DataFrame, mode: str, csv_file: Path) -> pd.DataFrame:
    if not frame.isna().any().any():
        return frame

    if mode == "error":
        missing_counts = frame.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            f"{csv_file}: missing or non-numeric feature values found: "
            f"{missing_counts}. Choose --missing to handle them."
        )
    if mode == "drop":
        return frame.dropna(axis=0)
    if mode == "ffill":
        return frame.ffill().bfill()
    if mode == "bfill":
        return frame.bfill().ffill()
    if mode == "zero":
        return frame.fillna(0.0)
    if mode == "mean":
        return frame.fillna(frame.mean(numeric_only=True))

    raise ValueError(f"Unsupported missing value mode: {mode}")


def make_overlapping_windows(values: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("--seq-len must be a positive integer.")
    if stride <= 0:
        raise ValueError("--stride must be a positive integer.")
    if values.shape[0] < seq_len:
        raise ValueError(
            f"CSV length L={values.shape[0]} is shorter than seq_len T={seq_len}."
        )

    window_view = np.lib.stride_tricks.sliding_window_view(
        values,
        window_shape=seq_len,
        axis=0,
    )
    windows = np.moveaxis(window_view, -1, 1)[::stride]
    return np.ascontiguousarray(windows)


def convert_csv(
    csv_file: Path,
    output_dir: Path,
    seq_len: int,
    stride: int,
    feature_cols: list[str] | None,
    drop_cols: list[str],
    missing: str,
    dtype: str,
    suffix: str,
    overwrite: bool,
    start_date: str | None,
    end_date: str | None,
    time_col: str | None,
) -> tuple[Path, tuple[int, ...]]:
    df = pd.read_csv(csv_file)
    df, selected_time_col = filter_by_date_range(
        df=df,
        csv_file=csv_file,
        time_col=time_col,
        start_date=start_date,
        end_date=end_date,
    )
    feature_frame, selected_cols = select_feature_frame(
        df, csv_file, feature_cols, drop_cols
    )
    feature_frame = handle_missing_values(feature_frame, missing, csv_file)
    values = feature_frame.to_numpy(dtype=np.dtype(dtype), copy=True)
    windows = make_overlapping_windows(values, seq_len=seq_len, stride=stride)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{csv_file.stem}{suffix}.npz"
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. Use --overwrite to replace it."
        )

    np.savez_compressed(
        output_file,
        data=windows,
        feature_cols=np.array(selected_cols),
        source_csv=str(csv_file),
        seq_len=np.array(seq_len),
        stride=np.array(stride),
        start_date=np.array(start_date if start_date is not None else ""),
        end_date=np.array(end_date if end_date is not None else ""),
        time_col=np.array(selected_time_col if selected_time_col is not None else ""),
    )
    return output_file, windows.shape


def main() -> None:
    args = parse_args()
    csv_files = collect_csv_files(args.input, recursive=args.recursive)

    for csv_file in csv_files:
        output_file, data_shape = convert_csv(
            csv_file=csv_file,
            output_dir=args.output_dir,
            seq_len=args.seq_len,
            stride=args.stride,
            feature_cols=args.feature_cols,
            drop_cols=args.drop_cols,
            missing=args.missing,
            dtype=args.dtype,
            suffix=args.suffix,
            overwrite=args.overwrite,
            start_date=args.start_date,
            end_date=args.end_date,
            time_col=args.time_col,
        )
        print(f"{csv_file} -> {output_file} data_shape={data_shape}")


if __name__ == "__main__":
    main()
