#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import paths
from data_utils import inverse_transform_data, get_npz_data, load_scaler, save_data, split_data
from evaluation_utils import (
    load_npz_data_and_feature_names,
    read_json,
    resolve_run_dir,
    save_distribution_evaluation,
    save_tsne_plot,
    select_split,
    write_json,
)
from vae.vae_utils import get_prior_samples, load_vae_model, set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate outputs from the best HPO model.")
    parser.add_argument("--best-run", type=Path, required=True, help="Path to outputs/hpo/<timestamp>/best_run.json.")
    parser.add_argument(
        "--num-samples",
        default="train",
        help="Number of generated prior samples: train, valid, all, or an integer. Default: train.",
    )
    parser.add_argument(
        "--compare-split",
        choices=("train", "valid", "all"),
        default="train",
        help="Original split used for t-SNE and distribution comparison. Default: train.",
    )
    parser.add_argument("--max-tsne-samples", type=int, default=2000)
    parser.add_argument(
        "--num-bins",
        type=int,
        default=None,
        help="Histogram bins for distribution metrics. Default: automatic.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--existing-generated-npz",
        type=Path,
        default=None,
        help=(
            "Use an existing inverse-scaled generated NPZ and only run "
            "evaluation outputs instead of loading the VAE and resampling."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the seed stored in the HPO run config.")
    parser.add_argument("--save-scaled", action="store_true", help="Also save generated samples before inverse scaling.")
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE plot generation.")
    return parser.parse_args()


def resolve_num_samples(spec: str, train_data: np.ndarray, valid_data: np.ndarray) -> int:
    if spec == "train":
        return int(train_data.shape[0])
    if spec == "valid":
        return int(valid_data.shape[0])
    if spec == "all":
        return int(train_data.shape[0] + valid_data.shape[0])
    try:
        value = int(spec)
    except ValueError as exc:
        raise ValueError("--num-samples must be train, valid, all, or an integer.") from exc
    if value <= 0:
        raise ValueError("--num-samples integer must be positive.")
    return value


def json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    best_run_path = args.best_run.resolve()
    best_run = read_json(best_run_path)
    run_dir = resolve_run_dir(best_run, best_run_path)
    config = read_json(run_dir / "config.json")
    model_dir = run_dir / "best_model"
    output_dir = args.output_dir if args.output_dir is not None else run_dir / "best_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = config["dataset"]
    vae_type = config["vae_type"]
    valid_perc = float(config["valid_perc"])
    split_method = config.get("split_method", "tail_holdout")
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_seeds(seed)

    dataset_file = Path(paths.DATASETS_DIR) / f"{dataset}.npz"
    data, feature_names = load_npz_data_and_feature_names(dataset_file)
    train_data, valid_data = split_data(
        data,
        valid_perc=valid_perc,
        shuffle=True,
        seed=seed,
        split_method=split_method,
    )
    compare_original = select_split(train_data, valid_data, args.compare_split)

    scaler = None
    generated_file = output_dir / f"{vae_type}_{Path(dataset).name}_best_prior_samples.npz"
    scaled_file = None
    generation_mode = "sampled_prior"
    if args.existing_generated_npz is not None:
        generation_mode = "existing_generated_npz"
        generated_file = args.existing_generated_npz.resolve()
        generated = get_npz_data(str(generated_file))
        if generated.ndim != 3 or generated.shape[1:] != data.shape[1:]:
            raise ValueError(
                "Existing generated NPZ must contain data with shape (N, T, D) "
                f"matching the original dataset. generated={generated.shape}, "
                f"original={data.shape}"
            )
        generated_scaled = None
        if not args.no_tsne:
            scaler = load_scaler(str(model_dir))
            generated_scaled = scaler.transform(generated)
        num_samples = int(generated.shape[0])
        if args.save_scaled:
            raise ValueError(
                "--save-scaled cannot be used with --existing-generated-npz "
                "because the script is not generating new samples."
            )
    else:
        scaler = load_scaler(str(model_dir))
        vae_model = load_vae_model(vae_type, str(model_dir))
        num_samples = resolve_num_samples(args.num_samples, train_data, valid_data)
        generated_scaled = get_prior_samples(vae_model, num_samples=num_samples)
        generated = inverse_transform_data(generated_scaled, scaler)
        save_data(generated, str(generated_file))
        if args.save_scaled:
            scaled_file = output_dir / f"{vae_type}_{Path(dataset).name}_best_prior_samples_scaled.npz"
            save_data(generated_scaled, str(scaled_file))

    evaluation_info = save_distribution_evaluation(
        real_data=compare_original,
        synthetic_data=generated,
        output_dir=output_dir,
        feature_names=feature_names,
        num_bins=args.num_bins,
        real_label=f"Original {args.compare_split}",
        synthetic_label="Generated prior",
    )

    tsne_info = None
    if not args.no_tsne:
        if scaler is None:
            scaler = load_scaler(str(model_dir))
        scaled_train_data = scaler.transform(train_data)
        scaled_valid_data = scaler.transform(valid_data)
        compare_scaled = select_split(
            scaled_train_data, scaled_valid_data, args.compare_split
        )
        tsne_file = output_dir / f"tsne_generated_vs_{args.compare_split}.png"
        tsne_info = save_tsne_plot(
            original_samples=compare_scaled,
            generated_samples=generated_scaled,
            original_name=f"Original {args.compare_split}",
            generated_name="Generated prior",
            output_file=tsne_file,
            max_samples=args.max_tsne_samples,
            seed=seed,
        )

    summary = {
        "best_run_path": str(best_run_path),
        "source_run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "dataset": dataset,
        "vae_type": vae_type,
        "valid_perc": valid_perc,
        "split_method": split_method,
        "seed": seed,
        "monitor_metric": best_run.get("monitor_metric", config.get("monitor_metric")),
        "free_bits": best_run.get(
            "free_bits", config.get("hyperparameters", {}).get("free_bits")
        ),
        "kl_anneal_epochs": best_run.get(
            "kl_anneal_epochs",
            config.get("hyperparameters", {}).get("kl_anneal_epochs"),
        ),
        "histogram_distance_backend": best_run.get(
            "histogram_distance_backend",
            config.get("hyperparameters", {}).get(
                "histogram_distance_backend", config.get("histogram_distance_backend")
            ),
        ),
        "compute_train_histogram_distance": best_run.get(
            "compute_train_histogram_distance",
            config.get("hyperparameters", {}).get(
                "compute_train_histogram_distance",
                config.get("compute_train_histogram_distance"),
            ),
        ),
        "best_monitor_value": best_run.get("best_monitor_value"),
        "best_epoch": best_run.get("best_epoch"),
        "generation_mode": generation_mode,
        "num_samples_spec": args.num_samples,
        "num_samples": num_samples,
        "compare_split": args.compare_split,
        "compare_split_size": int(compare_original.shape[0]),
        "generated_shape": list(generated.shape),
        "generated_file": str(generated_file),
        "scaled_generated_file": str(scaled_file) if scaled_file is not None else None,
        "evaluation": evaluation_info,
        "tsne": tsne_info,
    }
    write_json(output_dir / "rerun_config.json", summary)
    json_print(summary)


if __name__ == "__main__":
    main()
