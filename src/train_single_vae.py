#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paths
from data_utils import (
    inverse_transform_data,
    load_data,
    load_yaml_file,
    save_data,
    save_scaler,
    scale_data,
    split_data,
)
from vae.vae_utils import (
    get_prior_samples,
    instantiate_vae_model,
    save_vae_model,
    train_vae,
)


class WandbEpochLogger:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.wandb = None

    def init(self, args: argparse.Namespace, config: dict[str, Any], run_dir: Path) -> None:
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "--log-backend wandb was requested, but wandB is not installed. "
                "Install it or use --log-backend none."
            ) from exc
        self.wandb = wandb
        self.wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_id,
            config=config,
            dir=str(run_dir),
        )

    def log_history(self, history: dict[str, list[Any]]) -> None:
        if not self.enabled or self.wandb is None:
            return
        epochs = max((len(values) for values in history.values()), default=0)
        for epoch in range(epochs):
            payload = {key: values[epoch] for key, values in history.items() if epoch < len(values)}
            self.wandb.log(payload, step=epoch)

    def finish(self, result: dict[str, Any]) -> None:
        if not self.enabled or self.wandb is None:
            return
        self.wandb.summary.update(result)
        self.wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one VAE experiment.")
    parser.add_argument("--dataset", required=True, help="Dataset path relative to data/, without .npz.")
    parser.add_argument("--vae-type", default="timeVAE", choices=("timeVAE", "vae_dense", "vae_conv"))
    parser.add_argument("--valid-perc", type=float, default=0.1)
    parser.add_argument(
        "--split-method",
        choices=("tail_holdout", "full_train_recent_blocks"),
        default="tail_holdout",
        help=(
            "Data split strategy. tail_holdout reserves the final "
            "valid-percentage samples for validation, then shuffles only "
            "training data; full_train_recent_blocks uses all samples for "
            "training and copies validation from three recent 122-sample "
            "blocks."
        ),
    )
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--reconstruction-wt", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument(
        "--loss-mode",
        choices=("current", "legacy"),
        default="current",
        help="VAE loss formula: current uses KL annealing/free bits/new validation loss; legacy uses the original pre-annealing loss.",
    )
    parser.add_argument(
        "--early-stopping-start-epoch",
        type=int,
        default=0,
        help="Do not count early-stopping patience before this zero-based epoch.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum monitored-loss improvement required by early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generate-after-train", action="store_true")
    parser.add_argument("--require-gpu", action="store_true", help="Fail fast if TensorFlow cannot see a GPU.")
    parser.add_argument("--log-backend", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-project", default="timevae-hpo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def write_history_csv(path: Path, history: dict[str, list[Any]]) -> None:
    keys = list(history.keys())
    rows = max((len(values) for values in history.values()), default=0)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", *keys])
        writer.writeheader()
        for epoch in range(rows):
            row = {"epoch": epoch}
            row.update({key: history[key][epoch] for key in keys if epoch < len(history[key])})
            writer.writerow(row)


def plot_loss_curve(path: Path, history: dict[str, list[Any]]) -> None:
    plt.figure(figsize=(8, 5))
    if "total_loss" in history:
        plt.plot(history["total_loss"], label="train_total_loss")
    elif "loss" in history:
        plt.plot(history["loss"], label="train_loss")
    if "val_total_loss" in history:
        plt.plot(history["val_total_loss"], label="val_total_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def best_metric_from_history(history: dict[str, list[Any]], key: str) -> tuple[float | None, int | None]:
    values = history.get(key, [])
    if not values:
        return None, None
    best_epoch = int(np.argmin(values))
    return float(values[best_epoch]), best_epoch


def tensorflow_gpu_status() -> dict[str, Any]:
    import tensorflow as tf

    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_version": tf.__version__,
        "built_with_cuda": bool(tf.test.is_built_with_cuda()),
        "physical_gpus": [device.name for device in physical_gpus],
        "logical_gpus": [device.name for device in logical_gpus],
    }


def main() -> None:
    args = parse_args()
    if args.run_id is None:
        args.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    if args.split_method == "tail_holdout" and not 0 < args.valid_perc < 1:
        raise ValueError("--valid-perc must be between 0 and 1 for tail_holdout splits.")
    if args.early_stopping_start_epoch < 0:
        raise ValueError("--early-stopping-start-epoch must be non-negative.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be non-negative.")

    run_dir = args.run_dir
    model_dir = run_dir / "best_model"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    start_wall = time.time()
    start_time = utc_now_iso()

    hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[args.vae_type].copy()
    hyperparameters.update(
        {
            "latent_dim": args.latent_dim,
            "reconstruction_wt": args.reconstruction_wt,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "loss_mode": args.loss_mode,
        }
    )

    gpu_status = tensorflow_gpu_status()
    config = {
        "run_id": args.run_id,
        "dataset": args.dataset,
        "vae_type": args.vae_type,
        "valid_perc": args.valid_perc,
        "split_method": args.split_method,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "early_stopping_start_epoch": args.early_stopping_start_epoch,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "loss_mode": args.loss_mode,
        "require_gpu": args.require_gpu,
        "gpu_status": gpu_status,
        "hyperparameters": hyperparameters,
        "generate_after_train": args.generate_after_train,
    }
    write_json(run_dir / "config.json", config)

    logger = WandbEpochLogger(args.log_backend == "wandb")
    logger.init(args, config, run_dir)

    status = "completed"
    error = None
    try:
        if args.require_gpu and not gpu_status["physical_gpus"]:
            raise RuntimeError(
                "--require-gpu was set, but TensorFlow cannot see any GPU. "
                f"gpu_status={gpu_status}"
            )
        data = load_data(data_dir=paths.DATASETS_DIR, dataset=args.dataset)
        train_data, valid_data = split_data(
            data,
            valid_perc=args.valid_perc,
            shuffle=True,
            seed=args.seed,
            split_method=args.split_method,
        )
        scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

        _, sequence_length, feature_dim = scaled_train_data.shape
        vae_model = instantiate_vae_model(
            vae_type=args.vae_type,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            seed=args.seed,
            **hyperparameters,
        )
        history_obj = train_vae(
            vae=vae_model,
            train_data=scaled_train_data,
            valid_data=scaled_valid_data,
            max_epochs=args.max_epochs,
            verbose=args.verbose,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_start_epoch=args.early_stopping_start_epoch,
        )
        history = history_obj.history
        logger.log_history(history)

        save_scaler(scaler=scaler, dir_path=str(model_dir))
        save_vae_model(vae=vae_model, dir_path=str(model_dir))
        write_history_csv(run_dir / "history.csv", history)
        plot_loss_curve(run_dir / "loss_curve.png", history)

        if args.generate_after_train:
            prior_samples = get_prior_samples(vae_model, num_samples=train_data.shape[0])
            inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
            save_data(
                data=inverse_scaled_prior_samples,
                output_file=str(run_dir / f"{args.vae_type}_{args.run_id}_prior_samples.npz"),
            )

        best_val, best_epoch = best_metric_from_history(history, "val_total_loss")
        if getattr(vae_model, "best_monitor_value", None) not in (None, np.inf):
            best_val = float(vae_model.best_monitor_value)
            best_epoch = None if vae_model.best_epoch is None else int(vae_model.best_epoch)
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        best_val = None
        best_epoch = None
        raise
    finally:
        end_wall = time.time()
        end_time = utc_now_iso()
        timing = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": end_wall - start_wall,
        }
        write_json(run_dir / "timing.json", timing)
        result = {
            "run_id": args.run_id,
            "status": status,
            "error": error,
            "dataset": args.dataset,
            "vae_type": args.vae_type,
            "latent_dim": args.latent_dim,
            "reconstruction_wt": args.reconstruction_wt,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "loss_mode": args.loss_mode,
            "valid_perc": args.valid_perc,
            "split_method": args.split_method,
            "early_stopping_start_epoch": args.early_stopping_start_epoch,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "require_gpu": args.require_gpu,
            "gpu_status": gpu_status,
            "best_val_total_loss": best_val,
            "best_epoch": best_epoch,
            "run_dir": str(run_dir),
            **timing,
        }
        write_json(run_dir / "result.json", result)
        logger.finish(result)


if __name__ == "__main__":
    main()
