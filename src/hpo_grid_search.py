#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import site
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import paths


MONITOR_METRIC_CHOICES = (
    "total_loss",
    "val_total_loss",
    "histogram_distance",
    "val_histogram_distance",
)

HISTOGRAM_DISTANCE_BACKEND_CHOICES = ("numpy", "tensorflow")


RESULT_FIELDS = [
    "run_id",
    "status",
    "error",
    "dataset",
    "vae_type",
    "latent_dim",
    "reconstruction_wt",
    "learning_rate",
    "batch_size",
    "free_bits",
    "kl_anneal_epochs",
    "loss_mode",
    "valid_perc",
    "split_method",
    "early_stopping_start_epoch",
    "early_stopping_min_delta",
    "early_stopping_patience",
    "monitor_metric",
    "histogram_distance_backend",
    "compute_train_histogram_distance",
    "reduce_lr_on_plateau",
    "best_val_total_loss",
    "best_monitor_value",
    "best_epoch",
    "duration_seconds",
    "gpu_id",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local grid search for TimeVAE.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset path relative to data/, without .npz. May be repeated.")
    parser.add_argument("--dataset-glob", default=None, help="Glob for npz files, e.g. data/new_npz_data/weather_data/T84/*.npz.")
    parser.add_argument("--vae-type", default="timeVAE", choices=("timeVAE", "vae_dense", "vae_conv"))
    parser.add_argument("--valid-perc", type=float, default=0.1)
    parser.add_argument(
        "--split-method",
        choices=("tail_holdout", "full_train_recent_blocks"),
        default="tail_holdout",
        help=(
            "Data split strategy passed to train_single_vae.py. "
            "tail_holdout reserves the final valid-percentage samples for "
            "validation, then shuffles only training data; "
            "full_train_recent_blocks uses all samples for training and "
            "copies validation from three recent 122-day/488-sample blocks."
        ),
    )
    parser.add_argument("--latent-dim", type=int, nargs="+", required=True)
    parser.add_argument("--reconstruction-wt", type=float, nargs="+", required=True)
    parser.add_argument("--learning-rate", type=float, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, nargs="+", required=True)
    parser.add_argument(
        "--free-bits",
        type=float,
        nargs="+",
        default=None,
        help="Optional grid values for VAE free bits. Defaults to hyperparameters.yaml when omitted.",
    )
    parser.add_argument(
        "--kl-anneal-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Optional grid values for VAE KL annealing epochs. Defaults to hyperparameters.yaml when omitted.",
    )
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument(
        "--histogram-distance-backend",
        choices=HISTOGRAM_DISTANCE_BACKEND_CHOICES,
        default="numpy",
        help=(
            "Backend for batch histogram_distance monitoring in child runs. "
            "numpy matches rerun semantics most closely; tensorflow avoids per-batch NumPy callbacks."
        ),
    )
    parser.add_argument(
        "--disable-train-histogram-distance",
        action="store_true",
        help=(
            "Skip histogram_distance during training batches in child runs. "
            "Validation histogram distance is still computed."
        ),
    )
    parser.add_argument(
        "--loss-mode",
        choices=("current", "legacy"),
        default="current",
        help="VAE loss formula passed to each child training run.",
    )
    parser.add_argument(
        "--early-stopping-start-epoch",
        type=int,
        default=0,
        help="Pass-through to train_single_vae.py; early stopping starts counting at this zero-based epoch.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Pass-through to train_single_vae.py; minimum monitored-loss improvement required by early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Pass-through to train_single_vae.py; unimproved monitored epochs to tolerate before early stopping.",
    )
    parser.add_argument(
        "--monitor-metric",
        choices=MONITOR_METRIC_CHOICES,
        default=None,
        help=(
            "Metric used by child runs for best-weight restore, early stopping, "
            "optional LR reduction, and HPO best-run selection. Defaults to val_total_loss."
        ),
    )
    parser.add_argument(
        "--enable-reduce-lr-on-plateau",
        action="store_true",
        help=(
            "Enable Keras ReduceLROnPlateau in child runs. Disabled by default. "
            "When enabled, it monitors --monitor-metric with factor=0.5 and patience=30."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-slots", required=True, help="GPU concurrency map, e.g. 0:2,1:2,2:4. Use none:1 for CPU.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Exact output directory. Overrides output root, group, and experiment name.")
    parser.add_argument("--output-root", type=Path, default=None, help="Root directory for HPO outputs. Defaults to outputs/hpo.")
    parser.add_argument("--experiment-group", default=None, help="Optional folder under the HPO output root, e.g. weather20032015.")
    parser.add_argument("--experiment-name", default=None, help="Human-readable experiment name appended after the timestamp.")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow a GPU-assigned job to continue on CPU if TensorFlow cannot see a GPU.",
    )
    parser.add_argument("--generate-after-train", action="store_true")
    parser.add_argument("--log-backend", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-project", default="timevae-hpo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def dataset_from_npz_path(path: str) -> str:
    npz_path = Path(path)
    data_root = Path(paths.DATASETS_DIR).resolve()
    try:
        rel = npz_path.resolve().relative_to(data_root)
    except ValueError:
        rel = npz_path
    if rel.suffix == ".npz":
        rel = rel.with_suffix("")
    return rel.as_posix()


def collect_datasets(args: argparse.Namespace) -> list[str]:
    datasets = list(args.dataset)
    if args.dataset_glob:
        matched = sorted(glob.glob(args.dataset_glob))
        datasets.extend(dataset_from_npz_path(path) for path in matched)
    seen = set()
    unique = []
    for dataset in datasets:
        dataset = dataset.removesuffix(".npz")
        if dataset.startswith("data/"):
            dataset = dataset[len("data/") :]
        if dataset not in seen:
            seen.add(dataset)
            unique.append(dataset)
    if not unique:
        raise ValueError("Provide at least one --dataset or a --dataset-glob that matches .npz files.")
    return unique


def parse_gpu_slots(spec: str) -> list[str]:
    slots = []
    for item in spec.split(","):
        if not item.strip():
            continue
        gpu_id, count = item.split(":", 1)
        slots.extend([gpu_id.strip()] * int(count))
    if not slots:
        raise ValueError("--gpu-slots must define at least one slot.")
    return slots


def slug(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def default_experiment_name(datasets: list[str]) -> str:
    if len(datasets) == 1:
        return Path(datasets[0]).name
    return "multi_dataset"


def resolve_output_dir(args: argparse.Namespace, datasets: list[str]) -> Path:
    if args.output_dir is not None:
        return args.output_dir

    output_root = args.output_root or Path(paths.OUTPUTS_DIR) / "hpo"
    if args.experiment_group:
        output_root = output_root / slug(args.experiment_group)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = slug(args.experiment_name or default_experiment_name(datasets))
    if experiment_name:
        return output_root / f"{timestamp}_{experiment_name}"
    return output_root / timestamp


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in RESULT_FIELDS})


def pip_nvidia_library_dirs() -> list[str]:
    dirs = []
    for site_dir in site.getsitepackages():
        nvidia_dir = Path(site_dir) / "nvidia"
        if not nvidia_dir.exists():
            continue
        dirs.extend(str(path) for path in sorted(nvidia_dir.glob("*/lib")))
    return dirs


def extend_ld_library_path(env: dict[str, str], extra_dirs: list[str]) -> None:
    if not extra_dirs:
        return
    existing = env.get("LD_LIBRARY_PATH")
    paths = list(extra_dirs)
    if existing:
        paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(paths)


def build_jobs(args: argparse.Namespace, datasets: list[str], output_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    free_bits_values = args.free_bits if args.free_bits is not None else [None]
    kl_anneal_epochs_values = (
        args.kl_anneal_epochs if args.kl_anneal_epochs is not None else [None]
    )
    combos = itertools.product(
        datasets,
        args.latent_dim,
        args.reconstruction_wt,
        args.learning_rate,
        args.batch_size,
        free_bits_values,
        kl_anneal_epochs_values,
    )
    for idx, (
        dataset,
        latent_dim,
        reconstruction_wt,
        learning_rate,
        batch_size,
        free_bits,
        kl_anneal_epochs,
    ) in enumerate(combos):
        run_id = (
            f"run_{idx:05d}_"
            f"{slug(dataset)}_"
            f"ld{latent_dim}_rw{reconstruction_wt}_lr{learning_rate}_bs{batch_size}"
        )
        if free_bits is not None:
            run_id += f"_fb{free_bits}"
        if kl_anneal_epochs is not None:
            run_id += f"_ka{kl_anneal_epochs}"
        run_dir = output_dir / "runs" / run_id
        jobs.append(
            {
                "run_id": run_id,
                "run_dir": run_dir,
                "dataset": dataset,
                "latent_dim": latent_dim,
                "reconstruction_wt": reconstruction_wt,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "free_bits": free_bits,
                "kl_anneal_epochs": kl_anneal_epochs,
                "loss_mode": args.loss_mode,
                "valid_perc": args.valid_perc,
                "split_method": args.split_method,
                "early_stopping_start_epoch": args.early_stopping_start_epoch,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "early_stopping_patience": args.early_stopping_patience,
                "monitor_metric": args.monitor_metric,
                "histogram_distance_backend": args.histogram_distance_backend,
                "compute_train_histogram_distance": (
                    not args.disable_train_histogram_distance
                ),
                "reduce_lr_on_plateau": args.enable_reduce_lr_on_plateau,
            }
        )
    return jobs


def launch_job(
    args: argparse.Namespace, job: dict[str, Any], gpu_id: str, nvidia_library_dirs: list[str]
) -> subprocess.Popen:
    env = os.environ.copy()
    extend_ld_library_path(env, nvidia_library_dirs)
    if gpu_id.lower() in ("none", "cpu", ""):
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("train_single_vae.py")),
        "--dataset",
        job["dataset"],
        "--vae-type",
        args.vae_type,
        "--valid-perc",
        str(args.valid_perc),
        "--split-method",
        args.split_method,
        "--latent-dim",
        str(job["latent_dim"]),
        "--reconstruction-wt",
        str(job["reconstruction_wt"]),
        "--learning-rate",
        str(job["learning_rate"]),
        "--batch-size",
        str(job["batch_size"]),
        "--max-epochs",
        str(args.max_epochs),
        "--loss-mode",
        args.loss_mode,
        "--histogram-distance-backend",
        args.histogram_distance_backend,
        "--early-stopping-start-epoch",
        str(args.early_stopping_start_epoch),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--monitor-metric",
        args.monitor_metric,
        "--seed",
        str(args.seed),
        "--run-dir",
        str(job["run_dir"]),
        "--run-id",
        job["run_id"],
        "--log-backend",
        args.log_backend,
        "--wandb-project",
        args.wandb_project,
        "--verbose",
        str(args.verbose),
    ]
    if job["free_bits"] is not None:
        cmd.extend(["--free-bits", str(job["free_bits"])])
    if job["kl_anneal_epochs"] is not None:
        cmd.extend(["--kl-anneal-epochs", str(job["kl_anneal_epochs"])])
    if args.disable_train_histogram_distance:
        cmd.append("--disable-train-histogram-distance")
    if args.enable_reduce_lr_on_plateau:
        cmd.append("--enable-reduce-lr-on-plateau")
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    if args.generate_after_train:
        cmd.append("--generate-after-train")
    if not args.allow_cpu_fallback and gpu_id.lower() not in ("none", "cpu", ""):
        cmd.append("--require-gpu")

    job["run_dir"].mkdir(parents=True, exist_ok=True)
    stdout = (job["run_dir"] / "stdout.log").open("w")
    stderr = (job["run_dir"] / "stderr.log").open("w")
    process = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
    process._timevae_stdout = stdout
    process._timevae_stderr = stderr
    return process


def read_result(job: dict[str, Any], gpu_id: str, return_code: int) -> dict[str, Any]:
    result_path = job["run_dir"] / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {
            "run_id": job["run_id"],
            "status": "failed",
            "error": f"subprocess exited with code {return_code} before writing result.json",
            "dataset": job["dataset"],
            "latent_dim": job["latent_dim"],
            "reconstruction_wt": job["reconstruction_wt"],
            "learning_rate": job["learning_rate"],
            "batch_size": job["batch_size"],
            "free_bits": job["free_bits"],
            "kl_anneal_epochs": job["kl_anneal_epochs"],
            "loss_mode": job["loss_mode"],
            "valid_perc": job["valid_perc"],
            "split_method": job["split_method"],
            "early_stopping_start_epoch": job["early_stopping_start_epoch"],
            "early_stopping_min_delta": job["early_stopping_min_delta"],
            "early_stopping_patience": job["early_stopping_patience"],
            "monitor_metric": job["monitor_metric"],
            "histogram_distance_backend": job["histogram_distance_backend"],
            "compute_train_histogram_distance": job[
                "compute_train_histogram_distance"
            ],
            "reduce_lr_on_plateau": job["reduce_lr_on_plateau"],
            "best_monitor_value": None,
            "run_dir": str(job["run_dir"]),
        }
    result.setdefault("free_bits", job["free_bits"])
    result.setdefault("kl_anneal_epochs", job["kl_anneal_epochs"])
    result.setdefault("monitor_metric", job["monitor_metric"])
    result.setdefault(
        "histogram_distance_backend", job["histogram_distance_backend"]
    )
    result.setdefault(
        "compute_train_histogram_distance",
        job["compute_train_histogram_distance"],
    )
    result.setdefault("reduce_lr_on_plateau", job["reduce_lr_on_plateau"])
    if (
        result.get("best_monitor_value") is None
        and result.get("monitor_metric") == "val_total_loss"
    ):
        result["best_monitor_value"] = result.get("best_val_total_loss")
    result["gpu_id"] = gpu_id
    return result


def main() -> None:
    args = parse_args()
    if args.early_stopping_start_epoch < 0:
        raise ValueError("--early-stopping-start-epoch must be non-negative.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be non-negative.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative.")
    if args.free_bits is not None and any(value < 0 for value in args.free_bits):
        raise ValueError("--free-bits values must be non-negative.")
    if args.kl_anneal_epochs is not None and any(
        value < 0 for value in args.kl_anneal_epochs
    ):
        raise ValueError("--kl-anneal-epochs values must be non-negative.")
    args.monitor_metric = args.monitor_metric or "val_total_loss"
    if args.disable_train_histogram_distance and args.monitor_metric == "histogram_distance":
        raise ValueError(
            "--disable-train-histogram-distance cannot be used with "
            "--monitor-metric histogram_distance. Use val_histogram_distance instead."
        )
    datasets = collect_datasets(args)
    gpu_slots = parse_gpu_slots(args.gpu_slots)
    output_dir = resolve_output_dir(args, datasets)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args, datasets, output_dir)
    write_json(
        output_dir / "search_config.json",
        {
            "datasets": datasets,
            "vae_type": args.vae_type,
            "valid_perc": args.valid_perc,
            "split_method": args.split_method,
            "latent_dim": args.latent_dim,
            "reconstruction_wt": args.reconstruction_wt,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "free_bits": args.free_bits,
            "kl_anneal_epochs": args.kl_anneal_epochs,
            "loss_mode": args.loss_mode,
            "max_epochs": args.max_epochs,
            "experiment_group": args.experiment_group,
            "experiment_name": args.experiment_name or default_experiment_name(datasets),
            "output_dir": str(output_dir),
            "allow_cpu_fallback": args.allow_cpu_fallback,
            "nvidia_library_dirs": pip_nvidia_library_dirs(),
            "early_stopping_start_epoch": args.early_stopping_start_epoch,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "early_stopping_patience": args.early_stopping_patience,
            "monitor_metric": args.monitor_metric,
            "histogram_distance_backend": args.histogram_distance_backend,
            "compute_train_histogram_distance": (
                not args.disable_train_histogram_distance
            ),
            "reduce_lr_on_plateau": args.enable_reduce_lr_on_plateau,
            "seed": args.seed,
            "gpu_slots": args.gpu_slots,
            "num_jobs": len(jobs),
        },
    )

    nvidia_library_dirs = pip_nvidia_library_dirs()
    pending = list(jobs)
    running: list[dict[str, Any]] = []
    available_slots = list(enumerate(gpu_slots))
    results: list[dict[str, Any]] = []
    results_jsonl = output_dir / "results.jsonl"

    print(f"Prepared {len(jobs)} jobs in {output_dir}")
    while pending or running:
        while pending and available_slots:
            slot_idx, gpu_id = available_slots.pop(0)
            job = pending.pop(0)
            process = launch_job(args, job, gpu_id, nvidia_library_dirs)
            running.append({"job": job, "process": process, "gpu_id": gpu_id, "slot_idx": slot_idx})
            print("Started {} on GPU slot {} device={}".format(job["run_id"], slot_idx, gpu_id))

        time.sleep(2)
        still_running = []
        for item in running:
            process = item["process"]
            return_code = process.poll()
            if return_code is None:
                still_running.append(item)
                continue
            process._timevae_stdout.close()
            process._timevae_stderr.close()
            available_slots.append((item["slot_idx"], item["gpu_id"]))
            available_slots.sort(key=lambda pair: pair[0])
            result = read_result(item["job"], item["gpu_id"], return_code)
            results.append(result)
            append_jsonl(results_jsonl, result)
            write_results_csv(output_dir / "results.csv", results)
            print(
                "Finished {} status={} {}={}".format(
                    result["run_id"],
                    result.get("status"),
                    args.monitor_metric,
                    result.get("best_monitor_value"),
                )
            )
        running = still_running

    completed = [
        r
        for r in results
        if r.get("status") == "completed"
        and r.get("best_monitor_value") is not None
    ]
    if completed:
        best = min(completed, key=lambda r: float(r["best_monitor_value"]))
        write_json(output_dir / "best_run.json", best)
        print(
            "Best run: {} {}={}".format(
                best["run_id"], args.monitor_metric, best["best_monitor_value"]
            )
        )
    else:
        print(f"No completed runs with {args.monitor_metric} were found.")


if __name__ == "__main__":
    main()
