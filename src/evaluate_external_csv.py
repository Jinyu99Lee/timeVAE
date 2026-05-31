#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import paths
from data_preprocess.csv_to_npz import (
    DEFAULT_DROP_COLS,
    filter_by_date_range,
    handle_missing_values,
    make_overlapping_windows,
    select_feature_frame,
)
from data_utils import load_scaler, save_data, split_data
from evaluation_utils import (
    load_npz_data_and_feature_names,
    read_json,
    resolve_run_dir,
    save_distribution_evaluation,
    save_tsne_plot,
    select_split,
    write_json,
)
from vae.vae_utils import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate external CSV synthetic time-series against the best "
            "HPO run setting."
        )
    )
    parser.add_argument("--synthetic-csv", type=Path, required=True)
    parser.add_argument(
        "--best-run",
        type=Path,
        required=True,
        help="Path to outputs/hpo/<timestamp>/best_run.json.",
    )
    parser.add_argument(
        "--compare-split",
        choices=("train", "valid", "all"),
        default="train",
        help="Original split used for t-SNE and distribution comparison. Default: train.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-tsne-samples", type=int, default=2000)
    parser.add_argument(
        "--num-bins",
        type=int,
        default=None,
        help="Histogram bins for distribution metrics. Default: automatic.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive CSV start date/time before windowing.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive CSV end date/time before windowing.",
    )
    parser.add_argument("--time-col", default=None, help="CSV time column for date filtering.")
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=None,
        help="CSV feature columns, in comparison order.",
    )
    parser.add_argument(
        "--missing",
        choices=("error", "drop", "ffill", "bfill", "zero", "mean"),
        default="error",
        help="How to handle missing or non-numeric feature values. Default: error.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Window stride for the synthetic CSV. Default: 1.",
    )
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE plot generation.")
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save windowed external synthetic samples as NPZ.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed stored in the HPO run config.",
    )
    return parser.parse_args()


def json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def load_external_csv_windows(
    csv_file: Path,
    seq_len: int,
    stride: int,
    feature_cols: list[str] | None,
    missing: str,
    start_date: str | None,
    end_date: str | None,
    time_col: str | None,
) -> tuple[np.ndarray, list[str], str | None]:
    if not csv_file.is_file():
        raise FileNotFoundError(f"Synthetic CSV not found: {csv_file}")
    if csv_file.suffix.lower() != ".csv":
        raise ValueError(f"--synthetic-csv must point to a CSV file: {csv_file}")
    if stride <= 0:
        raise ValueError("--stride must be positive.")

    df = pd.read_csv(csv_file)
    df, selected_time_col = filter_by_date_range(
        df=df,
        csv_file=csv_file,
        time_col=time_col,
        start_date=start_date,
        end_date=end_date,
    )
    feature_frame, selected_cols = select_feature_frame(
        df=df,
        csv_file=csv_file,
        feature_cols=feature_cols,
        drop_cols=list(DEFAULT_DROP_COLS),
    )
    feature_frame = handle_missing_values(feature_frame, missing, csv_file)
    values = feature_frame.to_numpy(dtype=np.float32, copy=True)
    windows = make_overlapping_windows(values, seq_len=seq_len, stride=stride)
    return windows, selected_cols, selected_time_col


def main() -> None:
    args = parse_args()
    best_run_path = args.best_run.resolve()
    best_run = read_json(best_run_path)
    run_dir = resolve_run_dir(best_run, best_run_path)
    config = read_json(run_dir / "config.json")
    model_dir = run_dir / "best_model"
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else run_dir / "external_outputs" / args.synthetic_csv.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = config["dataset"]
    valid_perc = float(config["valid_perc"])
    split_method = config.get("split_method", "tail_holdout")
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_seeds(seed)

    dataset_file = Path(paths.DATASETS_DIR) / f"{dataset}.npz"
    data, original_feature_names = load_npz_data_and_feature_names(dataset_file)
    train_data, valid_data = split_data(
        data,
        valid_perc=valid_perc,
        shuffle=True,
        seed=seed,
        split_method=split_method,
    )
    compare_original = select_split(train_data, valid_data, args.compare_split)
    seq_len = int(data.shape[1])
    feature_dim = int(data.shape[2])

    requested_feature_cols = args.feature_cols
    if requested_feature_cols is None and original_feature_names is not None:
        requested_feature_cols = original_feature_names

    synthetic_data, selected_feature_cols, selected_time_col = load_external_csv_windows(
        csv_file=args.synthetic_csv,
        seq_len=seq_len,
        stride=args.stride,
        feature_cols=requested_feature_cols,
        missing=args.missing,
        start_date=args.start_date,
        end_date=args.end_date,
        time_col=args.time_col,
    )
    if synthetic_data.shape[1:] != compare_original.shape[1:]:
        raise ValueError(
            "External synthetic windows must match the original data (T, D). "
            f"synthetic={synthetic_data.shape}, original={compare_original.shape}"
        )
    if synthetic_data.shape[2] != feature_dim:
        raise ValueError(
            "External synthetic feature dimension "
            f"{synthetic_data.shape[2]} does not match dataset D={feature_dim}."
        )

    feature_names = (
        original_feature_names
        if original_feature_names is not None
        else selected_feature_cols
    )
    evaluation_info = save_distribution_evaluation(
        real_data=compare_original,
        synthetic_data=synthetic_data,
        output_dir=output_dir,
        feature_names=feature_names,
        num_bins=args.num_bins,
        real_label=f"Original {args.compare_split}",
        synthetic_label=args.synthetic_csv.stem,
    )

    synthetic_npz_file = None
    if args.save_npz:
        synthetic_npz_file = output_dir / f"{args.synthetic_csv.stem}_windows.npz"
        save_data(synthetic_data, str(synthetic_npz_file))

    tsne_info = None
    if not args.no_tsne:
        scaler = load_scaler(str(model_dir))
        scaled_train_data = scaler.transform(train_data)
        scaled_valid_data = scaler.transform(valid_data)
        compare_scaled = select_split(
            scaled_train_data, scaled_valid_data, args.compare_split
        )
        synthetic_scaled = scaler.transform(synthetic_data)
        tsne_file = output_dir / f"tsne_{args.synthetic_csv.stem}_vs_{args.compare_split}.png"
        tsne_info = save_tsne_plot(
            original_samples=compare_scaled,
            generated_samples=synthetic_scaled,
            original_name=f"Original {args.compare_split}",
            generated_name=args.synthetic_csv.stem,
            output_file=tsne_file,
            max_samples=args.max_tsne_samples,
            seed=seed,
        )

    summary = {
        "best_run_path": str(best_run_path),
        "source_run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "dataset": dataset,
        "valid_perc": valid_perc,
        "split_method": split_method,
        "seed": seed,
        "synthetic_csv": str(args.synthetic_csv),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "time_col": selected_time_col,
        "selected_feature_cols": selected_feature_cols,
        "stride": args.stride,
        "compare_split": args.compare_split,
        "compare_split_size": int(compare_original.shape[0]),
        "synthetic_shape": list(synthetic_data.shape),
        "synthetic_npz_file": (
            str(synthetic_npz_file) if synthetic_npz_file is not None else None
        ),
        "evaluation": evaluation_info,
        "tsne": tsne_info,
    }
    write_json(output_dir / "external_evaluation_config.json", summary)
    json_print(summary)


if __name__ == "__main__":
    main()
