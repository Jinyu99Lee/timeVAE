"""Evaluate an externally generated synthetic NPZ (e.g. Diffusion-TS output)
with the timeVAE evaluation utilities.

Unlike ``rerun_best_hpo.py`` this script does NOT assume a timeVAE HPO run
layout (no ``config.json`` / ``best_model`` / fitted scaler required). It only
needs:

  * the original (real) dataset NPZ with key ``data`` of shape (N, T, D);
  * the generated synthetic NPZ with key ``data`` of shape (M, T, D).

It reproduces the same sample-axis split as the generator (default
``full_train_recent_blocks``), fits a MinMaxScaler on the TRAIN split only
(matching Diffusion-TS / timeVAE), runs the distribution evaluation on the
original physical scale, and renders the t-SNE plot on the scaled data.

Example
-------
python timeVAE/src/eval_external_synth.py \
    --data-npz Diffusion-TS/Data/datasets/weather/T84/weather_london_2003_2014.npz \
    --synth-npz Diffusion-TS/outputs/hpo/weather/weather_T84_grid_london/best_synth.npz \
    --output-dir Diffusion-TS/outputs/hpo/weather/weather_T84_grid_london/timevae_eval \
    --split-method full_train_recent_blocks --valid-perc 0.1 \
    --compare-split train --max-tsne-samples 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Allow running the file directly: make sibling timeVAE/src modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import get_npz_data, split_data  # noqa: E402
from evaluation_utils import (  # noqa: E402
    load_npz_data_and_feature_names,
    save_distribution_evaluation,
    save_tsne_plot,
    select_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an external synthetic NPZ with timeVAE utilities."
    )
    parser.add_argument(
        "--data-npz",
        type=Path,
        required=True,
        help="Original/real dataset NPZ (key 'data', shape N x T x D).",
    )
    parser.add_argument(
        "--synth-npz",
        type=Path,
        required=True,
        help="Generated synthetic NPZ (key 'data', shape M x T x D).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where metrics/plots are written.",
    )
    parser.add_argument(
        "--split-method",
        default="full_train_recent_blocks",
        choices=("full_train_recent_blocks", "tail_holdout"),
        help="Sample-axis split used by the generator. Default: full_train_recent_blocks.",
    )
    parser.add_argument(
        "--valid-perc",
        type=float,
        default=0.1,
        help="Validation fraction (only used by tail_holdout). Default: 0.1.",
    )
    parser.add_argument(
        "--compare-split",
        choices=("train", "valid", "all"),
        default="train",
        help="Which real split to compare against. Default: train.",
    )
    parser.add_argument("--num-bins", type=int, default=None)
    parser.add_argument("--max-tsne-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-tsne", action="store_true", help="Skip t-SNE plot generation."
    )
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Skip the per-feature histogram PDF (recommended for high-dimensional "
        "data such as ILI; the feature_statistics.csv and averaged metrics are "
        "still written).",
    )
    return parser.parse_args()


def json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data, feature_names = load_npz_data_and_feature_names(args.data_npz.resolve())
    generated = get_npz_data(str(args.synth_npz.resolve()))

    if generated.ndim != 3 or generated.shape[1:] != data.shape[1:]:
        raise ValueError(
            "Synthetic NPZ must have shape (M, T, D) matching the dataset's "
            f"(T, D). synthetic={generated.shape}, dataset={data.shape}"
        )

    # Reproduce the generator's split. shuffle=False keeps it deterministic;
    # ordering is irrelevant for distribution metrics and only affects which
    # subset t-SNE samples when M/N exceed --max-tsne-samples.
    train_data, valid_data = split_data(
        data,
        valid_perc=args.valid_perc,
        shuffle=False,
        seed=args.seed,
        split_method=args.split_method,
    )
    compare_original = select_split(train_data, valid_data, args.compare_split)

    # Distribution metrics on the original physical scale (matches rerun_best_hpo).
    evaluation_info = save_distribution_evaluation(
        real_data=compare_original,
        synthetic_data=generated,
        output_dir=output_dir,
        feature_names=feature_names,
        num_bins=args.num_bins,
        real_label=f"Original {args.compare_split}",
        synthetic_label="Generated synthetic",
        make_histograms=not args.no_histogram,
    )

    tsne_info = None
    if not args.no_tsne:
        # Fit MinMaxScaler on the TRAIN split only (no validation leakage),
        # exactly as the generator does, then scale both sides for t-SNE.
        var_num = data.shape[2]
        scaler = MinMaxScaler()
        scaler.fit(train_data.reshape(-1, var_num))

        def scale(arr: np.ndarray) -> np.ndarray:
            return scaler.transform(arr.reshape(-1, var_num)).reshape(arr.shape)

        compare_scaled = scale(compare_original)
        generated_scaled = scale(generated)
        tsne_file = output_dir / f"tsne_generated_vs_{args.compare_split}.png"
        tsne_info = save_tsne_plot(
            original_samples=compare_scaled,
            generated_samples=generated_scaled,
            original_name=f"Original {args.compare_split}",
            generated_name="Generated synthetic",
            output_file=tsne_file,
            max_samples=args.max_tsne_samples,
            seed=args.seed,
        )

    summary = {
        "data_npz": str(args.data_npz),
        "synth_npz": str(args.synth_npz),
        "output_dir": str(output_dir),
        "split_method": args.split_method,
        "compare_split": args.compare_split,
        "num_real_samples": int(compare_original.shape[0]),
        "num_generated_samples": int(generated.shape[0]),
        "evaluation": evaluation_info,
        "tsne": tsne_info,
    }
    summary_file = output_dir / "external_eval_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    json_print(summary)


if __name__ == "__main__":
    main()
