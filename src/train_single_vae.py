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
from matplotlib.ticker import LogFormatterMathtext, LogLocator
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


MONITOR_METRIC_CHOICES = (
    "total_loss",
    "val_total_loss",
    "histogram_distance",
    "val_histogram_distance",
)

HISTOGRAM_DISTANCE_BACKEND_CHOICES = ("numpy", "tensorflow")


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
            "training data; full_train_recent_blocks holds out validation "
            "from three recent 122-day/488-sample blocks and uses the "
            "remaining samples for training."
        ),
    )
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--reconstruction-wt", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--free-bits",
        type=float,
        default=None,
        help="Override the VAE free-bits value from hyperparameters.yaml.",
    )
    parser.add_argument(
        "--kl-anneal-epochs",
        type=int,
        default=None,
        help="Override the VAE KL annealing epochs from hyperparameters.yaml.",
    )
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument(
        "--histogram-distance-backend",
        choices=HISTOGRAM_DISTANCE_BACKEND_CHOICES,
        default="numpy",
        help=(
            "Backend for batch histogram_distance monitoring. numpy matches "
            "rerun semantics most closely; tensorflow avoids per-batch NumPy callbacks."
        ),
    )
    parser.add_argument(
        "--disable-train-histogram-distance",
        action="store_true",
        help=(
            "Skip histogram_distance during training batches. Validation histogram "
            "distance is still computed for val_histogram_distance."
        ),
    )
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
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Number of unimproved monitored epochs to tolerate before early stopping.",
    )
    parser.add_argument(
        "--monitor-metric",
        choices=MONITOR_METRIC_CHOICES,
        default=None,
        help=(
            "Metric used for best-weight restore and early stopping, and for "
            "LR reduction when --enable-reduce-lr-on-plateau is set. "
            "Defaults to val_total_loss when validation data is provided."
        ),
    )
    parser.add_argument(
        "--enable-reduce-lr-on-plateau",
        action="store_true",
        help=(
            "Enable Keras ReduceLROnPlateau. Disabled by default. "
            "When enabled, it monitors --monitor-metric with factor=0.5 and patience=30."
        ),
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
    def plot_values(ax, values, label: str, color: str, linestyle: str) -> bool:
        if values is None:
            return False
        series = np.asarray(values, dtype=float)
        epochs = np.arange(series.shape[0])
        positive = np.isfinite(series) & (series > 0.0)
        if not np.any(positive):
            return False
        ax.plot(
            epochs[positive],
            series[positive],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
        )
        return True

    def plot_series(ax, key: str, label: str, color: str, linestyle: str) -> bool:
        values = history.get(key, [])
        if not values:
            return False
        return plot_values(ax, values, label, color, linestyle)

    def plot_sum_series(
        ax, key_a: str, key_b: str, label: str, color: str, linestyle: str
    ) -> bool:
        values_a = history.get(key_a, [])
        values_b = history.get(key_b, [])
        if not values_a or not values_b:
            return False
        length = min(len(values_a), len(values_b))
        values = np.asarray(values_a[:length], dtype=float) + np.asarray(
            values_b[:length], dtype=float
        )
        return plot_values(ax, values, label, color, linestyle)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    plotted |= plot_sum_series(
        ax, "reconstruction_loss", "kl_loss", "Train ELBO", "#8ecae6", "-"
    )
    plotted |= plot_sum_series(
        ax,
        "val_reconstruction_loss",
        "val_kl_loss",
        "Val ELBO",
        "#023e8a",
        "--",
    )
    plotted |= plot_series(
        ax, "reconstruction_loss", "Train recon", "#f6bd60", "-"
    )
    plotted |= plot_series(
        ax, "val_reconstruction_loss", "Val recon", "#d97706", "--"
    )
    plotted |= plot_series(ax, "kl_loss", "Train KL", "#95d5b2", "-")
    plotted |= plot_series(ax, "val_kl_loss", "Val KL", "#2d6a4f", "--")
    plotted |= plot_series(ax, "raw_kl_loss", "Train raw KL", "#90be6d", ":")
    plotted |= plot_series(ax, "val_raw_kl_loss", "Val raw KL", "#1b4332", ":")
    plotted |= plot_series(
        ax, "histogram_distance", "Train histogram", "#f4a3a3", "-"
    )
    plotted |= plot_series(
        ax, "val_histogram_distance", "Val histogram", "#9b1c1c", "--"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    if plotted:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def best_metric_from_history(history: dict[str, list[Any]], key: str) -> tuple[float | None, int | None]:
    values = history.get(key, [])
    if not values:
        return None, None
    series = np.asarray(values, dtype=float)
    finite = np.isfinite(series)
    if not np.any(finite):
        return None, None
    finite_indexes = np.flatnonzero(finite)
    best_epoch = int(finite_indexes[np.argmin(series[finite])])
    return float(series[best_epoch]), best_epoch


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
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative.")
    if args.free_bits is not None and args.free_bits < 0:
        raise ValueError("--free-bits must be non-negative.")
    if args.kl_anneal_epochs is not None and args.kl_anneal_epochs < 0:
        raise ValueError("--kl-anneal-epochs must be non-negative.")

    monitor_metric = args.monitor_metric or "val_total_loss"
    if args.disable_train_histogram_distance and monitor_metric == "histogram_distance":
        raise ValueError(
            "--disable-train-histogram-distance cannot be used with "
            "--monitor-metric histogram_distance. Use val_histogram_distance instead."
        )

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
            "free_bits": hyperparameters.get("free_bits"),
            "kl_anneal_epochs": hyperparameters.get("kl_anneal_epochs"),
            "loss_mode": args.loss_mode,
            "histogram_distance_backend": args.histogram_distance_backend,
            "compute_train_histogram_distance": (
                not args.disable_train_histogram_distance
            ),
        }
    )
    if args.free_bits is not None:
        hyperparameters["free_bits"] = args.free_bits
    if args.kl_anneal_epochs is not None:
        hyperparameters["kl_anneal_epochs"] = args.kl_anneal_epochs

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
        "early_stopping_patience": args.early_stopping_patience,
        "monitor_metric": monitor_metric,
        "histogram_distance_backend": args.histogram_distance_backend,
        "compute_train_histogram_distance": (
            not args.disable_train_histogram_distance
        ),
        "reduce_lr_on_plateau": args.enable_reduce_lr_on_plateau,
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
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_start_epoch=args.early_stopping_start_epoch,
            monitor_metric=monitor_metric,
            reduce_lr_on_plateau=args.enable_reduce_lr_on_plateau,
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

        best_val_total_loss, _ = best_metric_from_history(history, "val_total_loss")
        best_monitor_value, best_epoch = best_metric_from_history(history, monitor_metric)
        if getattr(vae_model, "best_monitor_value", None) not in (None, np.inf):
            best_monitor_value = float(vae_model.best_monitor_value)
            best_epoch = None if vae_model.best_epoch is None else int(vae_model.best_epoch)
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        best_val_total_loss = None
        best_monitor_value = None
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
            "free_bits": hyperparameters.get("free_bits"),
            "kl_anneal_epochs": hyperparameters.get("kl_anneal_epochs"),
            "loss_mode": args.loss_mode,
            "valid_perc": args.valid_perc,
            "split_method": args.split_method,
            "early_stopping_start_epoch": args.early_stopping_start_epoch,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "early_stopping_patience": args.early_stopping_patience,
            "monitor_metric": monitor_metric,
            "histogram_distance_backend": args.histogram_distance_backend,
            "compute_train_histogram_distance": (
                not args.disable_train_histogram_distance
            ),
            "reduce_lr_on_plateau": args.enable_reduce_lr_on_plateau,
            "require_gpu": args.require_gpu,
            "gpu_status": gpu_status,
            "best_val_total_loss": best_val_total_loss,
            "best_monitor_value": best_monitor_value,
            "best_epoch": best_epoch,
            "run_dir": str(run_dir),
            **timing,
        }
        write_json(run_dir / "result.json", result)
        logger.finish(result)


if __name__ == "__main__":
    main()
