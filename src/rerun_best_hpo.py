#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import paths
from data_utils import inverse_transform_data, load_data, load_scaler, save_data, split_data
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
        help="Original split used for t-SNE comparison. Default: train.",
    )
    parser.add_argument("--max-tsne-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the seed stored in the HPO run config.")
    parser.add_argument("--save-scaled", action="store_true", help="Also save generated samples before inverse scaling.")
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE plot generation.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_run_dir(best_run: dict[str, Any], best_run_path: Path) -> Path:
    run_dir = Path(best_run["run_dir"])
    if run_dir.is_absolute():
        return run_dir
    return (best_run_path.parent / run_dir).resolve()


def select_split(train_data: np.ndarray, valid_data: np.ndarray, split: str) -> np.ndarray:
    if split == "train":
        return train_data
    if split == "valid":
        return valid_data
    if split == "all":
        return np.concatenate([train_data, valid_data], axis=0)
    raise ValueError(f"Unsupported split: {split}")


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


def save_tsne_plot(
    original_samples: np.ndarray,
    generated_samples: np.ndarray,
    original_name: str,
    generated_name: str,
    output_file: Path,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    if original_samples.ndim != 3 or generated_samples.ndim != 3:
        raise ValueError("t-SNE expects both arrays to have shape (N, T, D).")
    if original_samples.shape[1:] != generated_samples.shape[1:]:
        raise ValueError(
            "Original and generated samples must share (T, D) for t-SNE. "
            f"original={original_samples.shape}, generated={generated_samples.shape}"
        )

    original_used = min(original_samples.shape[0], max_samples)
    generated_used = min(generated_samples.shape[0], max_samples)
    if original_used < 2 or generated_used < 2:
        raise ValueError("Need at least two original and generated samples for t-SNE.")

    original_2d = np.mean(original_samples[:original_used], axis=2)
    generated_2d = np.mean(generated_samples[:generated_used], axis=2)
    combined = np.vstack([original_2d, generated_2d])
    total_used = combined.shape[0]
    perplexity = min(40, max(2, (total_used - 1) // 3))

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, random_state=seed)
    embedded = tsne.fit_transform(combined)
    tsne_df = pd.DataFrame(
        {
            "tsne_1": embedded[:, 0],
            "tsne_2": embedded[:, 1],
            "sample_type": [original_name] * original_used + [generated_name] * generated_used,
        }
    )

    plt.figure(figsize=(8, 8))
    for sample_type, color in ((original_name, "red"), (generated_name, "blue")):
        points = tsne_df[tsne_df["sample_type"] == sample_type]
        plt.scatter(points["tsne_1"], points["tsne_2"], label=sample_type, color=color, alpha=0.5, s=70)
    plt.title(f"t-SNE: {original_name} vs {generated_name}")
    plt.legend()
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close()

    return {
        "original_used": original_used,
        "generated_used": generated_used,
        "perplexity": perplexity,
        "output_file": str(output_file),
    }


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

    data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset)
    train_data, valid_data = split_data(
        data,
        valid_perc=valid_perc,
        shuffle=True,
        seed=seed,
        split_method=split_method,
    )
    scaler = load_scaler(str(model_dir))
    scaled_train_data = scaler.transform(train_data)
    scaled_valid_data = scaler.transform(valid_data)
    compare_scaled = select_split(scaled_train_data, scaled_valid_data, args.compare_split)
    compare_original = select_split(train_data, valid_data, args.compare_split)

    vae_model = load_vae_model(vae_type, str(model_dir))
    num_samples = resolve_num_samples(args.num_samples, train_data, valid_data)

    generated_scaled = get_prior_samples(vae_model, num_samples=num_samples)
    generated = inverse_transform_data(generated_scaled, scaler)

    generated_file = output_dir / f"{vae_type}_{Path(dataset).name}_best_prior_samples.npz"
    save_data(generated, str(generated_file))
    scaled_file = None
    if args.save_scaled:
        scaled_file = output_dir / f"{vae_type}_{Path(dataset).name}_best_prior_samples_scaled.npz"
        save_data(generated_scaled, str(scaled_file))

    tsne_info = None
    if not args.no_tsne:
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
        "num_samples_spec": args.num_samples,
        "num_samples": num_samples,
        "compare_split": args.compare_split,
        "compare_split_size": int(compare_original.shape[0]),
        "generated_shape": list(generated.shape),
        "generated_file": str(generated_file),
        "scaled_generated_file": str(scaled_file) if scaled_file is not None else None,
        "tsne": tsne_info,
    }
    write_json(output_dir / "rerun_config.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
