#!/usr/bin/env python3
"""Manifest-driven TimeVAE experiments for downstream forecasting.

The forecasting generators consume the same pre-windowed train/validation NPZ
pairs as Diffusion-TS.  This module keeps the experiment matrix, HPO command,
rerun command, and Sonnet hand-off in one auditable place.  It deliberately
does not import TensorFlow, so contracts and dry-runs can be tested cheaply.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "forecasting_timevae.json"
LEARNING_RATES = (0.0001, 0.0005, 0.001, 0.01)
SONNET_LEARNING_RATES = (0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002)
SONNET_ATOMS = (8, 16, 32)
SONNET_ALPHAS = (0, 0.25, 0.5, 0.75)
TIMEVAE_PROTOCOL = {
    "vae_type": "timeVAE",
    "latent_dim": 8,
    "reconstruction_wt": 3.0,
    "learning_rates": LEARNING_RATES,
    "batch_size": 16,
    "max_epochs": 800,
    "loss_mode": "legacy",
    "monitor_metric": "val_total_loss",
    "monitor_kl_latent_ref": 8,
    "early_stopping_start_epoch": 0,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.0001,
    "histogram_distance_backend": "numpy",
    "compute_train_histogram_distance": False,
    "compute_val_histogram_distance": False,
    "reduce_lr_on_plateau": False,
    "valid_perc": 0.1,
    "split_method": "tail_holdout",
    "free_bits": 0.1,
    "kl_anneal_epochs": 50,
    "seed": 42,
}


@dataclass(frozen=True)
class ForecastExperiment:
    experiment_id: str
    dataset: str
    source_dataset: str
    fold: str
    lookback: int
    pred_len: int
    seq_len: int
    target_column: str
    train_npz: Path
    val_npz: Path
    source_meta: Path
    expected_num_windows: int
    hpo_output_dir: Path
    rerun_output_dir: Path
    sonnet_dataset: str
    sonnet_exp: str
    sonnet_seq_length: int
    sonnet_batch_sizes: tuple[int, ...]
    sonnet_hpo_profile: str

    @property
    def generated_npz(self) -> Path:
        return self.rerun_output_dir / f"timevae_{self.experiment_id}_synthetic.npz"

    @property
    def best_run(self) -> Path:
        return self.hpo_output_dir / "best_run.json"


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer.")
    resolved = int(value)
    if resolved <= 0 or resolved != value:
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    return resolved


def _positive_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON list.")
    resolved = tuple(
        _positive_int(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} must not contain duplicates.")
    return resolved


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string.")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[ForecastExperiment]:
    """Load and structurally validate the forecasting experiment manifest."""
    path = Path(path).resolve()
    payload = _require_mapping(json.loads(path.read_text()), "manifest")
    if payload.get("schema_version") != 1:
        raise ValueError("Forecast manifest schema_version must be 1.")
    path_base = payload.get("path_base", ".")
    base = _resolve(path.parent, path_base, "path_base")
    rows = payload.get("experiments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Forecast manifest experiments must be a non-empty list.")

    experiments: list[ForecastExperiment] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"experiments[{index}]")
        experiment_id = str(row.get("id", "")).strip()
        if not experiment_id:
            raise ValueError(f"experiments[{index}].id must be non-empty.")
        if experiment_id in seen:
            raise ValueError(f"Duplicate forecasting experiment id: {experiment_id}")
        seen.add(experiment_id)
        sonnet = _require_mapping(row.get("sonnet"), f"{experiment_id}.sonnet")
        lookback = _positive_int(row.get("lookback"), f"{experiment_id}.lookback")
        pred_len = _positive_int(row.get("pred_len"), f"{experiment_id}.pred_len")
        seq_len = _positive_int(row.get("seq_len"), f"{experiment_id}.seq_len")
        if seq_len != lookback + pred_len:
            raise ValueError(
                f"{experiment_id}: seq_len={seq_len} must equal "
                f"lookback+pred_len={lookback + pred_len}."
            )
        profile = str(sonnet.get("hpo_profile", ""))
        if profile not in {"etth1_raw", "energy_raw"}:
            raise ValueError(
                f"{experiment_id}: unsupported Sonnet HPO profile {profile!r}."
            )
        experiments.append(
            ForecastExperiment(
                experiment_id=experiment_id,
                dataset=str(row["dataset"]),
                source_dataset=str(row.get("source_dataset", row["dataset"])),
                fold=str(row.get("fold", "default")),
                lookback=lookback,
                pred_len=pred_len,
                seq_len=seq_len,
                target_column=str(row["target_column"]),
                train_npz=_resolve(base, row["train_npz"], f"{experiment_id}.train_npz"),
                val_npz=_resolve(base, row["val_npz"], f"{experiment_id}.val_npz"),
                source_meta=_resolve(base, row["source_meta"], f"{experiment_id}.source_meta"),
                expected_num_windows=_positive_int(
                    row["expected_num_windows"], f"{experiment_id}.expected_num_windows"
                ),
                hpo_output_dir=_resolve(
                    base, row["hpo_output_dir"], f"{experiment_id}.hpo_output_dir"
                ),
                rerun_output_dir=_resolve(
                    base, row["rerun_output_dir"], f"{experiment_id}.rerun_output_dir"
                ),
                sonnet_dataset=str(sonnet["dataset"]),
                sonnet_exp=str(sonnet["exp"]),
                sonnet_seq_length=_positive_int(
                    sonnet["seq_length"], f"{experiment_id}.sonnet.seq_length"
                ),
                sonnet_batch_sizes=_positive_int_tuple(
                    sonnet["batch_sizes"], f"{experiment_id}.sonnet.batch_sizes"
                ),
                sonnet_hpo_profile=profile,
            )
        )
    return experiments


def select_experiments(
    experiments: Sequence[ForecastExperiment], selected: Sequence[str] | None
) -> list[ForecastExperiment]:
    if not selected:
        return list(experiments)
    requested = set(selected)
    by_id = {experiment.experiment_id: experiment for experiment in experiments}
    missing = sorted(requested - set(by_id))
    if missing:
        raise ValueError(f"Unknown forecasting experiment id(s): {missing}")
    return [experiment for experiment in experiments if experiment.experiment_id in requested]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_float32_c(data: np.ndarray) -> str:
    values = np.ascontiguousarray(data, dtype=np.float32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _scalar_json(value: np.ndarray, label: str) -> dict[str, Any]:
    if value.shape != ():
        raise ValueError(f"{label} must be a scalar JSON string, got shape {value.shape}.")
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return _require_mapping(json.loads(str(raw)), label)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc


def _json_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}.")
    return value


def _scalar_npz_int(value: np.ndarray, label: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a scalar integer, got {array.dtype} {array.shape}.")
    return int(array.item())


def _scalar_npz_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "SU":
        raise ValueError(f"{label} must be a scalar string, got {array.dtype} {array.shape}.")
    raw = array.item()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _sha256_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} is not hexadecimal.") from exc
    if value != value.lower():
        raise ValueError(f"{label} must use lowercase hexadecimal.")
    return value


def _json_ranges(value: Any, label: str, upper: int) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list of [start, end) ranges.")
    ranges: list[list[int]] = []
    previous_end = 0
    for index, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"{label}[{index}] must be [start, end).")
        start = _json_int(raw[0], f"{label}[{index}][0]", minimum=0)
        end = _json_int(raw[1], f"{label}[{index}][1]", minimum=0)
        if not start < end <= upper or start < previous_end:
            raise ValueError(
                f"{label}[{index}]={raw!r} must be ordered, non-overlapping, "
                f"and within [0, {upper}]."
            )
        ranges.append([start, end])
        previous_end = end
    return ranges


def _load_source_npz(path: Path, expected_split: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Forecast source NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "data",
            "sample_indices",
            "window_start_indices",
            "feature_cols",
            "seq_len",
            "lookback",
            "pred_len",
            "target_delay",
            "layout",
            "stride",
            "split",
            "meta",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}.")
        data = np.asarray(payload["data"])
        if data.ndim != 3:
            raise ValueError(f"{path}: data must be (N,T,D), got {data.shape}.")
        if data.dtype != np.float32:
            raise ValueError(f"{path}: data must be float32, got {data.dtype}.")
        if not np.isfinite(data).all():
            raise ValueError(f"{path}: data contains NaN or infinite values.")
        feature_array = np.asarray(payload["feature_cols"])
        if feature_array.ndim != 1 or feature_array.dtype.kind not in "SU":
            raise ValueError(f"{path}: feature_cols must be a one-dimensional string array.")
        feature_cols = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in feature_array.tolist()
        ]
        if (
            len(feature_cols) != data.shape[2]
            or len(set(feature_cols)) != len(feature_cols)
            or any(not value for value in feature_cols)
        ):
            raise ValueError(
                f"{path}: feature_cols must contain {data.shape[2]} unique names."
            )
        meta = _scalar_json(payload["meta"], f"{path}:meta")
        sample_indices = np.asarray(payload["sample_indices"])
        window_starts = np.asarray(payload["window_start_indices"])
        if sample_indices.dtype != np.int64 or sample_indices.shape != (data.shape[0],):
            raise ValueError(
                f"{path}: sample_indices must be int64 with shape ({data.shape[0]},)."
            )
        if window_starts.dtype != np.int64 or window_starts.shape != (data.shape[0],):
            raise ValueError(
                f"{path}: window_start_indices must be int64 with shape ({data.shape[0]},)."
            )
        geometry = {
            "seq_len": _scalar_npz_int(payload["seq_len"], f"{path}:seq_len"),
            "lookback": _scalar_npz_int(payload["lookback"], f"{path}:lookback"),
            "pred_len": _scalar_npz_int(payload["pred_len"], f"{path}:pred_len"),
            "target_delay": _scalar_npz_int(
                payload["target_delay"], f"{path}:target_delay"
            ),
            "layout": _scalar_npz_text(payload["layout"], f"{path}:layout"),
            "stride": _scalar_npz_int(payload["stride"], f"{path}:stride"),
            "split": _scalar_npz_text(payload["split"], f"{path}:split"),
        }
    if str(meta.get("split")) != expected_split:
        raise ValueError(
            f"{path}: meta.split={meta.get('split')!r}, expected {expected_split!r}."
        )
    if geometry["split"] != expected_split:
        raise ValueError(
            f"{path}: split={geometry['split']!r}, expected {expected_split!r}."
        )
    if _json_int(meta.get("split_samples"), f"{path}:meta.split_samples", minimum=0) != data.shape[0]:
        raise ValueError(f"{path}: meta.split_samples does not match data.shape[0].")
    if feature_cols != meta.get("feature_cols"):
        raise ValueError(f"{path}: feature_cols differs from meta.feature_cols.")
    if data.shape[1] != geometry["seq_len"]:
        raise ValueError(f"{path}: data T differs from scalar seq_len.")
    for key, value in geometry.items():
        if meta.get(key) != value:
            raise ValueError(
                f"{path}: scalar {key}={value!r} differs from meta.{key}={meta.get(key)!r}."
            )
    if geometry["stride"] <= 0:
        raise ValueError(f"{path}: stride must be positive.")
    if sample_indices.size:
        if sample_indices[0] < 0 or np.any(np.diff(sample_indices) <= 0):
            raise ValueError(f"{path}: sample_indices must be nonnegative and increasing.")
    expected_starts = sample_indices * geometry["stride"]
    if not np.array_equal(window_starts, expected_starts):
        raise ValueError(
            f"{path}: window_start_indices must equal sample_indices * stride."
        )
    return {
        "data": data,
        "feature_cols": feature_cols,
        "meta": meta,
        "sample_indices": sample_indices,
        "window_start_indices": window_starts,
        "geometry": geometry,
    }


def validate_source_pair(experiment: ForecastExperiment) -> dict[str, Any]:
    """Validate one Diffusion-TS source pair and return its reconciled contract."""
    train = _load_source_npz(experiment.train_npz, "train")
    valid = _load_source_npz(experiment.val_npz, "val")
    train_data = train["data"]
    valid_data = valid["data"]
    if train_data.shape[1:] != valid_data.shape[1:]:
        raise ValueError(
            f"{experiment.experiment_id}: train/val (T,D) differ: "
            f"{train_data.shape[1:]} vs {valid_data.shape[1:]}."
        )
    if train["feature_cols"] != valid["feature_cols"]:
        raise ValueError(f"{experiment.experiment_id}: train/val feature_cols differ.")

    source_meta = _require_mapping(
        json.loads(experiment.source_meta.read_text()),
        f"{experiment.experiment_id}:source_meta",
    )
    required_source_keys = {
        "dataset",
        "source_csv",
        "source_row_range",
        "source_first_timestamp",
        "source_last_timestamp",
        "source_frequency",
        "source_train_sha256",
        "feature_cols",
        "target_column",
        "target_index",
        "dtype",
        "seq_len",
        "lookback",
        "pred_len",
        "target_delay",
        "layout",
        "stride",
        "N",
        "total_samples",
        "raw_physical_scale",
        "val_sample_start_ratio",
        "val_sample_end_ratio",
        "val_sample_start",
        "val_sample_end_exclusive",
        "train_sample_ranges",
        "train_samples",
        "val_samples",
        "sample_index_semantics",
        "window_start_row_formula",
    }
    missing = sorted(required_source_keys - set(source_meta))
    if missing:
        raise ValueError(
            f"{experiment.experiment_id}: source metadata is missing {missing}."
        )
    for key in (
        "target_index",
        "seq_len",
        "lookback",
        "pred_len",
        "target_delay",
        "stride",
        "N",
        "total_samples",
        "val_sample_start",
        "val_sample_end_exclusive",
        "train_samples",
        "val_samples",
    ):
        _json_int(source_meta[key], f"{experiment.experiment_id}:source.{key}", minimum=0)

    invariant_keys = (
        "dataset",
        "source_csv",
        "feature_cols",
        "target_column",
        "target_index",
        "dtype",
        "seq_len",
        "lookback",
        "pred_len",
        "target_delay",
        "layout",
        "stride",
        "N",
        "total_samples",
        "raw_physical_scale",
        "source_row_range",
        "source_first_timestamp",
        "source_last_timestamp",
        "source_frequency",
        "source_train_sha256",
        "val_sample_start_ratio",
        "val_sample_end_ratio",
        "val_sample_start",
        "val_sample_end_exclusive",
        "train_sample_ranges",
        "train_samples",
        "val_samples",
        "sample_index_semantics",
        "window_start_row_formula",
    )
    invariant_keys += tuple(
        key for key in ("schema_version", "method", "fold", "year") if key in source_meta
    )
    for key in invariant_keys:
        expected = source_meta.get(key)
        for split_name, split in (("train", train), ("val", valid)):
            actual = split["meta"].get(key)
            if actual != expected:
                raise ValueError(
                    f"{experiment.experiment_id}: {split_name} meta.{key}={actual!r} "
                    f"does not match source metadata {expected!r}."
                )

    expected_values = {
        "dataset": experiment.source_dataset,
        "target_column": experiment.target_column,
        "lookback": experiment.lookback,
        "pred_len": experiment.pred_len,
        "seq_len": experiment.seq_len,
        "target_delay": 0,
        "layout": "aligned",
        "stride": 1,
        "raw_physical_scale": True,
        "dtype": "float32",
        "source_frequency": "1h",
        "N": experiment.expected_num_windows,
        "total_samples": experiment.expected_num_windows,
        "val_sample_start_ratio": 0.70,
        "val_sample_end_ratio": 0.85,
        "sample_index_semantics": "ordinal_in_full_sliding_window_set",
        "window_start_row_formula": "sample_index * stride",
    }
    for key, expected in expected_values.items():
        if source_meta.get(key) != expected:
            raise ValueError(
                f"{experiment.experiment_id}: source {key}={source_meta.get(key)!r}, "
                f"expected {expected!r}."
            )
    feature_cols = list(source_meta["feature_cols"])
    if feature_cols != train["feature_cols"]:
        raise ValueError(f"{experiment.experiment_id}: source feature_cols differ from NPZ.")
    if not feature_cols or feature_cols[-1] != experiment.target_column:
        raise ValueError(
            f"{experiment.experiment_id}: target {experiment.target_column!r} must be last."
        )
    target_index = _json_int(
        source_meta["target_index"],
        f"{experiment.experiment_id}:source.target_index",
        minimum=0,
    )
    if target_index != len(feature_cols) - 1:
        raise ValueError(
            f"{experiment.experiment_id}: target_index={target_index} is not the last channel."
        )
    if train_data.shape[1:] != (experiment.seq_len, len(feature_cols)):
        raise ValueError(
            f"{experiment.experiment_id}: source shape {train_data.shape[1:]} does not "
            f"match ({experiment.seq_len}, {len(feature_cols)})."
        )
    total = int(train_data.shape[0] + valid_data.shape[0])
    if total != experiment.expected_num_windows:
        raise ValueError(
            f"{experiment.experiment_id}: train+val={total}, expected "
            f"{experiment.expected_num_windows}."
        )
    if _json_int(source_meta["train_samples"], "source.train_samples", minimum=0) != train_data.shape[0]:
        raise ValueError(f"{experiment.experiment_id}: source train_samples mismatch.")
    if _json_int(source_meta["val_samples"], "source.val_samples", minimum=0) != valid_data.shape[0]:
        raise ValueError(f"{experiment.experiment_id}: source val_samples mismatch.")

    train_indices = train["sample_indices"]
    val_indices = valid["sample_indices"]
    combined = np.concatenate((train_indices, val_indices))
    if not np.array_equal(np.sort(combined), np.arange(total, dtype=np.int64)):
        raise ValueError(
            f"{experiment.experiment_id}: train/val sample_indices must partition 0..N-1."
        )
    val_start = _json_int(
        source_meta["val_sample_start"], "source.val_sample_start", minimum=0
    )
    val_end = _json_int(
        source_meta["val_sample_end_exclusive"],
        "source.val_sample_end_exclusive",
        minimum=0,
    )
    if (val_start, val_end) != (int(total * 0.70), int(total * 0.85)):
        raise ValueError(
            f"{experiment.experiment_id}: validation boundaries {(val_start, val_end)} "
            "must be floor(0.70*N), floor(0.85*N)."
        )
    if not np.array_equal(val_indices, np.arange(val_start, val_end, dtype=np.int64)):
        raise ValueError(f"{experiment.experiment_id}: validation indices mismatch metadata.")
    train_ranges = _json_ranges(
        source_meta["train_sample_ranges"], "source.train_sample_ranges", total
    )
    expected_train = np.concatenate(
        [np.arange(start, end, dtype=np.int64) for start, end in train_ranges]
    )
    if not np.array_equal(train_indices, expected_train):
        raise ValueError(
            f"{experiment.experiment_id}: training indices mismatch train_sample_ranges."
        )
    expected_ranges = [[0, val_start], [val_end, total]]
    if train_ranges != expected_ranges:
        raise ValueError(
            f"{experiment.experiment_id}: train ranges {train_ranges} differ from "
            f"the fixed holdout protocol {expected_ranges}."
        )

    row_range = source_meta["source_row_range"]
    if not isinstance(row_range, list) or len(row_range) != 2:
        raise ValueError(f"{experiment.experiment_id}: source_row_range must be [start, end).")
    row_start = _json_int(row_range[0], "source.source_row_range[0]", minimum=0)
    row_end = _json_int(row_range[1], "source.source_row_range[1]", minimum=0)
    if row_end <= row_start:
        raise ValueError(f"{experiment.experiment_id}: source_row_range is empty.")
    expected_source_rows = (total - 1) * int(source_meta["stride"]) + experiment.seq_len
    if row_end - row_start != expected_source_rows:
        raise ValueError(
            f"{experiment.experiment_id}: source row count {row_end - row_start} "
            f"does not match N/stride/seq_len ({expected_source_rows})."
        )
    try:
        first_timestamp = datetime.fromisoformat(str(source_meta["source_first_timestamp"]))
        last_timestamp = datetime.fromisoformat(str(source_meta["source_last_timestamp"]))
    except ValueError as exc:
        raise ValueError(
            f"{experiment.experiment_id}: source timestamps must be ISO-8601."
        ) from exc
    if last_timestamp - first_timestamp != timedelta(hours=expected_source_rows - 1):
        raise ValueError(
            f"{experiment.experiment_id}: timestamps do not span the declared hourly rows."
        )
    source_csv = Path(str(source_meta["source_csv"]))
    if not source_csv.is_absolute():
        raise ValueError(f"{experiment.experiment_id}: source_csv must be absolute provenance.")

    source_train_sha256 = _sha256_hex(
        source_meta["source_train_sha256"], "source.source_train_sha256"
    )
    source_block = np.empty((expected_source_rows, len(feature_cols)), dtype=np.float32)
    zero_location = np.flatnonzero(train_indices == 0)
    zero_data = train_data
    if not zero_location.size:
        zero_location = np.flatnonzero(val_indices == 0)
        zero_data = valid_data
    if zero_location.size != 1:
        raise ValueError(f"{experiment.experiment_id}: exactly one window must have index 0.")
    source_block[: experiment.seq_len] = zero_data[int(zero_location[0])]
    for split in (train, valid):
        indices = split["sample_indices"]
        positive = indices > 0
        source_block[experiment.seq_len - 1 + indices[positive]] = split["data"][positive, -1]
    reconstructed_sha256 = sha256_float32_c(source_block)
    if reconstructed_sha256 != source_train_sha256:
        raise ValueError(
            f"{experiment.experiment_id}: reconstructed source SHA-256 "
            f"{reconstructed_sha256} != {source_train_sha256}."
        )
    all_windows = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(
            source_block, window_shape=experiment.seq_len, axis=0
        ),
        -1,
        1,
    )[:: int(source_meta["stride"])]
    for split_name, split in (("train", train), ("val", valid)):
        indices = split["sample_indices"]
        for start in range(0, len(indices), 256):
            stop = min(start + 256, len(indices))
            if not np.array_equal(all_windows[indices[start:stop]], split["data"][start:stop]):
                raise ValueError(
                    f"{experiment.experiment_id}: {split_name} windows do not exactly "
                    "match their declared source-window indices."
                )

    if experiment.fold != "default":
        source_fold = source_meta.get("fold", source_meta.get("year"))
        if str(source_fold) != experiment.fold:
            raise ValueError(
                f"{experiment.experiment_id}: source fold/year={source_fold!r}, "
                f"expected {experiment.fold!r}."
            )
    return {
        "train_data": train_data,
        "val_data": valid_data,
        "feature_cols": feature_cols,
        "source_meta": source_meta,
        "num_real_windows": total,
        "source_train_sha256": source_train_sha256,
    }


def build_hpo_command(
    experiment: ForecastExperiment,
    python_executable: str = sys.executable,
    gpu_slots: str = "0:1",
    skip_completed: bool = False,
    allow_cpu_fallback: bool = False,
) -> list[str]:
    command = [
        python_executable,
        str(REPO_ROOT / "src" / "hpo_grid_search.py"),
        "--train-npz",
        str(experiment.train_npz),
        "--val-npz",
        str(experiment.val_npz),
        "--vae-type",
        "timeVAE",
        "--latent-dim",
        "8",
        "--reconstruction-wt",
        "3",
        "--learning-rate",
        *(str(value) for value in LEARNING_RATES),
        "--batch-size",
        "16",
        "--valid-perc",
        "0.1",
        "--split-method",
        "tail_holdout",
        "--max-epochs",
        "800",
        "--loss-mode",
        "legacy",
        "--free-bits",
        "0.1",
        "--kl-anneal-epochs",
        "50",
        "--histogram-distance-backend",
        "numpy",
        "--monitor-metric",
        "val_total_loss",
        "--monitor-kl-latent-ref",
        "8",
        "--early-stopping-start-epoch",
        "0",
        "--early-stopping-patience",
        "20",
        "--early-stopping-min-delta",
        "0.0001",
        "--disable-train-histogram-distance",
        "--disable-val-histogram-distance",
        "--seed",
        "42",
        "--gpu-slots",
        gpu_slots,
        "--output-dir",
        str(experiment.hpo_output_dir),
        "--experiment-name",
        experiment.experiment_id,
    ]
    if skip_completed:
        command.append("--skip-completed")
    if allow_cpu_fallback:
        command.append("--allow-cpu-fallback")
    return command


def build_rerun_command(
    experiment: ForecastExperiment,
    manifest_path: Path,
    python_executable: str = sys.executable,
    no_tsne: bool = True,
) -> list[str]:
    command = [
        python_executable,
        str(REPO_ROOT / "src" / "rerun_best_hpo.py"),
        "--best-run",
        str(experiment.best_run),
        "--num-samples",
        "all",
        "--compare-split",
        "all",
        "--forecast-manifest",
        str(Path(manifest_path).resolve()),
        "--forecast-experiment",
        experiment.experiment_id,
        "--output-dir",
        str(experiment.rerun_output_dir),
    ]
    if no_tsne:
        command.append("--no-tsne")
    return command


def _resolve_recorded_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string.")
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def validate_forecasting_hpo_config(
    config: dict[str, Any],
    best_run: dict[str, Any],
    experiment: ForecastExperiment,
    run_dir: Path,
) -> dict[str, Any]:
    """Validate and return the complete fixed generator protocol for a rerun."""
    hyperparameters = _require_mapping(config.get("hyperparameters"), "config.hyperparameters")
    top_expected = {
        "vae_type": TIMEVAE_PROTOCOL["vae_type"],
        "valid_perc": TIMEVAE_PROTOCOL["valid_perc"],
        "split_method": TIMEVAE_PROTOCOL["split_method"],
        "seed": TIMEVAE_PROTOCOL["seed"],
        "max_epochs": TIMEVAE_PROTOCOL["max_epochs"],
        "early_stopping_start_epoch": TIMEVAE_PROTOCOL[
            "early_stopping_start_epoch"
        ],
        "early_stopping_min_delta": TIMEVAE_PROTOCOL["early_stopping_min_delta"],
        "early_stopping_patience": TIMEVAE_PROTOCOL["early_stopping_patience"],
        "monitor_metric": TIMEVAE_PROTOCOL["monitor_metric"],
        "histogram_distance_backend": TIMEVAE_PROTOCOL[
            "histogram_distance_backend"
        ],
        "compute_train_histogram_distance": False,
        "compute_val_histogram_distance": False,
        "reduce_lr_on_plateau": False,
        "loss_mode": TIMEVAE_PROTOCOL["loss_mode"],
        "generate_after_train": False,
    }
    for key, value in top_expected.items():
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise ValueError(
                f"Forecasting rerun requires config.{key}={value!r}, "
                f"got {config.get(key)!r}."
            )
    hp_expected = {
        "latent_dim": TIMEVAE_PROTOCOL["latent_dim"],
        "reconstruction_wt": TIMEVAE_PROTOCOL["reconstruction_wt"],
        "batch_size": TIMEVAE_PROTOCOL["batch_size"],
        "free_bits": TIMEVAE_PROTOCOL["free_bits"],
        "kl_anneal_epochs": TIMEVAE_PROTOCOL["kl_anneal_epochs"],
        "loss_mode": TIMEVAE_PROTOCOL["loss_mode"],
        "histogram_distance_backend": TIMEVAE_PROTOCOL[
            "histogram_distance_backend"
        ],
        "compute_train_histogram_distance": False,
        "compute_val_histogram_distance": False,
        "monitor_kl_latent_ref": TIMEVAE_PROTOCOL["monitor_kl_latent_ref"],
        "hidden_layer_sizes": [50, 100, 200],
        "use_residual_conn": True,
        "trend_poly": 0,
        "custom_seas": None,
    }
    for key, value in hp_expected.items():
        if hyperparameters.get(key) != value or type(hyperparameters.get(key)) is not type(value):
            raise ValueError(
                f"Forecasting rerun requires hyperparameters.{key}={value!r}, "
                f"got {hyperparameters.get(key)!r}."
            )
    learning_rate = hyperparameters.get("learning_rate")
    if type(learning_rate) is not float or learning_rate not in LEARNING_RATES:
        raise ValueError(
            "Forecasting rerun learning_rate must come from the declared HPO grid "
            f"{LEARNING_RATES}, got {learning_rate!r}."
        )
    if _resolve_recorded_path(config.get("train_npz"), "config.train_npz") != experiment.train_npz:
        raise ValueError("Best-run train_npz does not match the forecast manifest.")
    if _resolve_recorded_path(config.get("val_npz"), "config.val_npz") != experiment.val_npz:
        raise ValueError("Best-run val_npz does not match the forecast manifest.")
    expected_dataset = experiment.train_npz.name
    if expected_dataset.endswith("_train.npz"):
        expected_dataset = expected_dataset[: -len("_train.npz")]
    if config.get("dataset") != expected_dataset:
        raise ValueError(
            f"config.dataset={config.get('dataset')!r}, expected {expected_dataset!r}."
        )
    if not isinstance(config.get("run_id"), str) or not config["run_id"]:
        raise ValueError("config.run_id must be non-empty.")

    best_expected = {
        "status": "completed",
        "error": None,
        "run_id": config["run_id"],
        "dataset": expected_dataset,
        "monitor_kl_latent_ref": TIMEVAE_PROTOCOL["monitor_kl_latent_ref"],
        "vae_type": TIMEVAE_PROTOCOL["vae_type"],
        "latent_dim": TIMEVAE_PROTOCOL["latent_dim"],
        "reconstruction_wt": TIMEVAE_PROTOCOL["reconstruction_wt"],
        "learning_rate": learning_rate,
        "batch_size": TIMEVAE_PROTOCOL["batch_size"],
        "free_bits": TIMEVAE_PROTOCOL["free_bits"],
        "kl_anneal_epochs": TIMEVAE_PROTOCOL["kl_anneal_epochs"],
        "loss_mode": TIMEVAE_PROTOCOL["loss_mode"],
        "valid_perc": TIMEVAE_PROTOCOL["valid_perc"],
        "split_method": TIMEVAE_PROTOCOL["split_method"],
        "early_stopping_start_epoch": TIMEVAE_PROTOCOL[
            "early_stopping_start_epoch"
        ],
        "early_stopping_min_delta": TIMEVAE_PROTOCOL["early_stopping_min_delta"],
        "early_stopping_patience": TIMEVAE_PROTOCOL["early_stopping_patience"],
        "monitor_metric": TIMEVAE_PROTOCOL["monitor_metric"],
        "histogram_distance_backend": TIMEVAE_PROTOCOL[
            "histogram_distance_backend"
        ],
        "compute_train_histogram_distance": False,
        "compute_val_histogram_distance": False,
        "reduce_lr_on_plateau": False,
    }
    for key, value in best_expected.items():
        if best_run.get(key) != value or type(best_run.get(key)) is not type(value):
            raise ValueError(
                f"best_run.{key}={best_run.get(key)!r}, expected {value!r}."
            )
    for key, expected_path in (
        ("train_npz", experiment.train_npz),
        ("val_npz", experiment.val_npz),
    ):
        if _resolve_recorded_path(best_run.get(key), f"best_run.{key}") != expected_path:
            raise ValueError(f"best_run.{key} does not match the forecast manifest.")
    if _resolve_recorded_path(best_run.get("run_dir"), "best_run.run_dir") != Path(run_dir).resolve():
        raise ValueError("best_run.run_dir does not match the selected run directory.")
    monitor_value = best_run.get("best_monitor_value")
    if isinstance(monitor_value, bool) or not isinstance(monitor_value, (int, float)):
        raise ValueError("best_run.best_monitor_value must be numeric.")
    if not np.isfinite(float(monitor_value)):
        raise ValueError("best_run.best_monitor_value must be finite.")
    best_epoch = best_run.get("best_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 0:
        raise ValueError("best_run.best_epoch must be a nonnegative integer.")

    protocol = dict(TIMEVAE_PROTOCOL)
    protocol["learning_rates"] = list(LEARNING_RATES)
    protocol["selected_learning_rate"] = learning_rate
    protocol["hidden_layer_sizes"] = [50, 100, 200]
    protocol["use_residual_conn"] = True
    protocol["trend_poly"] = 0
    protocol["custom_seas"] = None
    return protocol


def build_generated_metadata(
    experiment: ForecastExperiment,
    source_contract: dict[str, Any],
    best_run: dict[str, Any],
    best_run_path: Path,
    run_dir: Path,
    model_dir: Path,
    seed: int,
    generated: np.ndarray,
    generator_protocol: dict[str, Any],
) -> dict[str, Any]:
    source_meta = source_contract["source_meta"]
    num_real = int(source_contract["num_real_windows"])
    num_generated = int(generated.shape[0])
    return {
        "schema_version": 1,
        "method": "timevae",
        "dataset": experiment.dataset,
        "fold": experiment.fold,
        "feature_cols": list(source_contract["feature_cols"]),
        "target_column": experiment.target_column,
        "target_index": len(source_contract["feature_cols"]) - 1,
        "lookback": experiment.lookback,
        "pred_len": experiment.pred_len,
        "seq_len": experiment.seq_len,
        "target_delay": 0,
        "layout": "aligned",
        "stride": 1,
        "raw_physical_scale": True,
        "dtype": "float32",
        "source_row_range": source_meta["source_row_range"],
        "source_first_timestamp": source_meta["source_first_timestamp"],
        "source_last_timestamp": source_meta["source_last_timestamp"],
        "source_frequency": source_meta.get("source_frequency"),
        "source_csv": source_meta["source_csv"],
        "source_dataset": source_meta["dataset"],
        "source_fold": source_meta.get("fold", "default"),
        "source_train_sha256": source_meta["source_train_sha256"],
        "source_train_npz": str(experiment.train_npz),
        "source_val_npz": str(experiment.val_npz),
        "source_meta_path": str(experiment.source_meta),
        "source_npz_sha256": {
            "train": sha256_file(experiment.train_npz),
            "val": sha256_file(experiment.val_npz),
        },
        "source_metadata_sha256": sha256_file(experiment.source_meta),
        "source_N": source_meta["N"],
        "source_train_sample_ranges": source_meta["train_sample_ranges"],
        "source_val_sample_start": source_meta["val_sample_start"],
        "source_val_sample_end_exclusive": source_meta["val_sample_end_exclusive"],
        "source_train_samples": source_meta["train_samples"],
        "source_val_samples": source_meta["val_samples"],
        "sample_index_semantics": "ordinal_one_to_one_count_with_full_source_window_set",
        "window_start_row_formula": "sample_index * stride",
        "num_real_windows": num_real,
        "num_generated": num_generated,
        "N": num_generated,
        "one_to_one_with_source_windows": num_generated == num_real,
        "best_run_path": str(Path(best_run_path).resolve()),
        "best_run_sha256": sha256_file(best_run_path),
        "best_run_id": best_run.get("run_id", best_run.get("best_run_id")),
        "best_checkpoint": str(model_dir.resolve()),
        "source_run_dir": str(run_dir.resolve()),
        "source_run_config": str((Path(run_dir) / "config.json").resolve()),
        "source_run_config_sha256": sha256_file(Path(run_dir) / "config.json"),
        "best_monitor_metric": best_run.get("monitor_metric", "val_total_loss"),
        "best_monitor_value": best_run.get("best_monitor_value"),
        "best_epoch": best_run.get("best_epoch"),
        "seed": int(seed),
        "generator_protocol": generator_protocol,
        "generated_data_sha256": sha256_float32_c(generated),
    }


def save_generated_npz(
    path: Path, generated: np.ndarray, metadata: dict[str, Any]
) -> None:
    """Write the strict scalar-JSON NPZ contract consumed by Sonnet."""
    values = np.asarray(generated, dtype=np.float32)
    expected_shape = (
        int(metadata["num_generated"]),
        int(metadata["seq_len"]),
        len(metadata["feature_cols"]),
    )
    if values.shape != expected_shape:
        raise ValueError(
            f"Generated data shape {values.shape} does not match {expected_shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Generated data contains NaN or infinite values.")
    if metadata.get("method") != "timevae":
        raise ValueError("Generated metadata method must be 'timevae'.")
    if not metadata.get("one_to_one_with_source_windows"):
        raise ValueError("Forecasting output must be 1:1 with all source windows.")
    if metadata.get("dtype") != "float32":
        raise ValueError("Generated metadata dtype must be 'float32'.")
    if metadata.get("generated_data_sha256") != sha256_float32_c(values):
        raise ValueError("Generated metadata data hash does not match generated values.")
    num_generated = int(metadata["num_generated"])
    stride = int(metadata["stride"])
    if (
        int(metadata["N"]) != num_generated
        or int(metadata["num_real_windows"]) != num_generated
    ):
        raise ValueError("Generated metadata N/count fields must agree exactly.")
    if stride != 1 or metadata.get("source_frequency") != "1h":
        raise ValueError("Forecasting output requires stride=1 and source_frequency='1h'.")
    if int(metadata["target_index"]) != len(metadata["feature_cols"]) - 1:
        raise ValueError("Generated target_index must identify the final channel.")
    sample_indices = np.arange(num_generated, dtype=np.int64)
    window_starts = sample_indices * stride
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=values,
        sample_indices=sample_indices,
        window_start_indices=window_starts,
        feature_cols=np.asarray(metadata["feature_cols"]),
        target_index=np.asarray(metadata["target_index"], dtype=np.int64),
        seq_len=np.asarray(metadata["seq_len"], dtype=np.int64),
        lookback=np.asarray(metadata["lookback"], dtype=np.int64),
        pred_len=np.asarray(metadata["pred_len"], dtype=np.int64),
        target_delay=np.asarray(metadata["target_delay"], dtype=np.int64),
        layout=np.asarray(metadata["layout"]),
        stride=np.asarray(stride, dtype=np.int64),
        N=np.asarray(metadata["N"], dtype=np.int64),
        dtype=np.asarray("float32"),
        source_frequency=np.asarray(metadata["source_frequency"]),
        source_train_sha256=np.asarray(metadata["source_train_sha256"]),
        method=np.asarray("timevae"),
        meta=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def validate_generated_npz(
    experiment: ForecastExperiment, path: Path | None = None
) -> dict[str, Any]:
    path = Path(path or experiment.generated_npz)
    if path.resolve() != experiment.generated_npz:
        raise ValueError(
            f"TimeVAE generated NPZ must match the manifest path {experiment.generated_npz}."
        )
    if not path.is_file():
        raise FileNotFoundError(f"TimeVAE generated NPZ not found: {path}")
    source_contract = validate_source_pair(experiment)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "data",
            "sample_indices",
            "window_start_indices",
            "feature_cols",
            "target_index",
            "seq_len",
            "lookback",
            "pred_len",
            "target_delay",
            "layout",
            "stride",
            "N",
            "dtype",
            "source_frequency",
            "source_train_sha256",
            "method",
            "meta",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path}: missing required arrays {missing}.")
        data = np.asarray(payload["data"])
        meta = _scalar_json(payload["meta"], f"{path}:meta")
        feature_array = np.asarray(payload["feature_cols"])
        if feature_array.ndim != 1 or feature_array.dtype.kind not in "SU":
            raise ValueError(f"{path}: feature_cols must be a one-dimensional string array.")
        feature_cols = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in feature_array.tolist()
        ]
        sample_indices = np.asarray(payload["sample_indices"])
        window_starts = np.asarray(payload["window_start_indices"])
        scalar_values = {
            "target_index": _scalar_npz_int(payload["target_index"], f"{path}:target_index"),
            "seq_len": _scalar_npz_int(payload["seq_len"], f"{path}:seq_len"),
            "lookback": _scalar_npz_int(payload["lookback"], f"{path}:lookback"),
            "pred_len": _scalar_npz_int(payload["pred_len"], f"{path}:pred_len"),
            "target_delay": _scalar_npz_int(
                payload["target_delay"], f"{path}:target_delay"
            ),
            "layout": _scalar_npz_text(payload["layout"], f"{path}:layout"),
            "stride": _scalar_npz_int(payload["stride"], f"{path}:stride"),
            "N": _scalar_npz_int(payload["N"], f"{path}:N"),
            "dtype": _scalar_npz_text(payload["dtype"], f"{path}:dtype"),
            "source_frequency": _scalar_npz_text(
                payload["source_frequency"], f"{path}:source_frequency"
            ),
            "source_train_sha256": _scalar_npz_text(
                payload["source_train_sha256"], f"{path}:source_train_sha256"
            ),
            "method": _scalar_npz_text(payload["method"], f"{path}:method"),
        }
    if data.dtype != np.float32 or data.shape != (
        experiment.expected_num_windows,
        experiment.seq_len,
        len(feature_cols),
    ):
        raise ValueError(f"{path}: invalid generated dtype/shape {data.dtype} {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: generated values are not finite.")
    if sample_indices.dtype != np.int64 or sample_indices.shape != (len(data),):
        raise ValueError(f"{path}: sample_indices must be int64 with shape ({len(data)},).")
    if window_starts.dtype != np.int64 or window_starts.shape != (len(data),):
        raise ValueError(
            f"{path}: window_start_indices must be int64 with shape ({len(data)},)."
        )
    expected_indices = np.arange(len(data), dtype=np.int64)
    if not np.array_equal(sample_indices, expected_indices):
        raise ValueError(f"{path}: generated sample_indices must be exactly 0..N-1.")
    if not np.array_equal(window_starts, expected_indices * scalar_values["stride"]):
        raise ValueError(f"{path}: generated window starts do not match stride=1.")
    for key, value in scalar_values.items():
        if meta.get(key) != value:
            raise ValueError(f"{path}: scalar {key} differs from meta.{key}.")
    source_meta = source_contract["source_meta"]
    expected = {
        "schema_version": 1,
        "method": "timevae",
        "dataset": experiment.dataset,
        "fold": experiment.fold,
        "feature_cols": source_contract["feature_cols"],
        "target_column": experiment.target_column,
        "target_index": len(feature_cols) - 1,
        "lookback": experiment.lookback,
        "pred_len": experiment.pred_len,
        "seq_len": experiment.seq_len,
        "target_delay": 0,
        "layout": "aligned",
        "stride": 1,
        "raw_physical_scale": True,
        "dtype": "float32",
        "source_row_range": source_meta["source_row_range"],
        "source_first_timestamp": source_meta["source_first_timestamp"],
        "source_last_timestamp": source_meta["source_last_timestamp"],
        "source_frequency": "1h",
        "source_csv": source_meta["source_csv"],
        "source_dataset": source_meta["dataset"],
        "source_fold": source_meta.get("fold", "default"),
        "source_train_sha256": source_meta["source_train_sha256"],
        "source_train_npz": str(experiment.train_npz),
        "source_val_npz": str(experiment.val_npz),
        "source_meta_path": str(experiment.source_meta),
        "source_npz_sha256": {
            "train": sha256_file(experiment.train_npz),
            "val": sha256_file(experiment.val_npz),
        },
        "source_metadata_sha256": sha256_file(experiment.source_meta),
        "source_N": source_meta["N"],
        "source_train_sample_ranges": source_meta["train_sample_ranges"],
        "source_val_sample_start": source_meta["val_sample_start"],
        "source_val_sample_end_exclusive": source_meta["val_sample_end_exclusive"],
        "source_train_samples": source_meta["train_samples"],
        "source_val_samples": source_meta["val_samples"],
        "sample_index_semantics": "ordinal_one_to_one_count_with_full_source_window_set",
        "window_start_row_formula": "sample_index * stride",
        "num_real_windows": experiment.expected_num_windows,
        "num_generated": experiment.expected_num_windows,
        "N": experiment.expected_num_windows,
        "one_to_one_with_source_windows": True,
        "seed": TIMEVAE_PROTOCOL["seed"],
        "best_monitor_metric": TIMEVAE_PROTOCOL["monitor_metric"],
        "generated_data_sha256": sha256_float32_c(data),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{path}: meta.{key}={meta.get(key)!r}, expected {value!r}.")
    if feature_cols[-1] != experiment.target_column:
        raise ValueError(f"{path}: target column must be last.")
    best_run_path = Path(str(meta.get("best_run_path", "")))
    if not best_run_path.is_absolute() or best_run_path.resolve() != experiment.best_run:
        raise ValueError(f"{path}: best_run_path must match the manifest HPO output.")
    if not best_run_path.is_file():
        raise FileNotFoundError(f"{path}: missing best-run provenance {best_run_path}.")
    if meta.get("best_run_sha256") != sha256_file(best_run_path):
        raise ValueError(f"{path}: best_run_sha256 mismatch.")
    best_run = _require_mapping(json.loads(best_run_path.read_text()), "best_run")
    run_dir = Path(str(meta.get("source_run_dir", "")))
    if not run_dir.is_absolute() or not run_dir.is_dir():
        raise ValueError(f"{path}: source_run_dir must be an existing absolute directory.")
    config_path = Path(str(meta.get("source_run_config", "")))
    if config_path.resolve() != (run_dir / "config.json").resolve() or not config_path.is_file():
        raise ValueError(f"{path}: source_run_config does not identify run_dir/config.json.")
    if meta.get("source_run_config_sha256") != sha256_file(config_path):
        raise ValueError(f"{path}: source_run_config_sha256 mismatch.")
    config = _require_mapping(json.loads(config_path.read_text()), "config")
    protocol = validate_forecasting_hpo_config(config, best_run, experiment, run_dir)
    if meta.get("generator_protocol") != protocol:
        raise ValueError(f"{path}: generator_protocol does not match the fixed protocol.")
    if meta.get("best_run_id") != best_run["run_id"]:
        raise ValueError(f"{path}: best_run_id mismatch.")
    if meta.get("best_monitor_value") != best_run["best_monitor_value"]:
        raise ValueError(f"{path}: best_monitor_value mismatch.")
    if meta.get("best_epoch") != best_run["best_epoch"]:
        raise ValueError(f"{path}: best_epoch mismatch.")
    checkpoint = Path(str(meta.get("best_checkpoint", "")))
    if checkpoint.resolve() != (run_dir / "best_model").resolve() or not checkpoint.is_dir():
        raise ValueError(f"{path}: best_checkpoint provenance is invalid.")
    return meta


def build_sonnet_command(
    experiment: ForecastExperiment,
    sonnet_root: Path,
    sonnet_python: str = "python3",
    gpu_slots: str = "0:1",
) -> list[str]:
    if experiment.sonnet_hpo_profile == "etth1_raw":
        expected = ("exp_data_config/etth1", "etth", 336, (16, 32, 64))
    else:
        expected = (
            experiment.sonnet_dataset,
            experiment.sonnet_exp,
            experiment.sonnet_seq_length,
            (64,),
        )
    actual = (
        experiment.sonnet_dataset,
        experiment.sonnet_exp,
        experiment.sonnet_seq_length,
        experiment.sonnet_batch_sizes,
    )
    if actual != expected:
        raise ValueError(
            f"{experiment.experiment_id}: Sonnet profile {experiment.sonnet_hpo_profile} "
            f"expects {expected}, got {actual}."
        )
    output_subdir = (
        f"timevae_hpo/syn/{experiment.dataset}/fold_{experiment.fold}/"
        f"h{experiment.pred_len}/sonnet"
    )
    return [
        sonnet_python,
        str(Path(sonnet_root).resolve() / "scripts" / "run_synthetic_hpo.py"),
        "--synthetic-dir",
        str(experiment.generated_npz.parent),
        "--pattern",
        experiment.generated_npz.name,
        "--gpu-slots",
        gpu_slots,
        "--output-subdir",
        output_subdir,
        "--batch-name",
        f"timevae_{experiment.experiment_id}_sonnet",
        "-m",
        "model=sonnet",
        f"dataset={experiment.sonnet_dataset}",
        f"exp={experiment.sonnet_exp}",
        "seed=2021",
        f"exp.seq_length={experiment.sonnet_seq_length}",
        f"exp.pred_length={experiment.pred_len}",
        "exp.lr_scheduler=linear_decay_lr",
        "exp.monitor_metric=val_target_seq_loss",
        "exp.batch_size=" + ",".join(str(value) for value in experiment.sonnet_batch_sizes),
        "model.model_params.d_model=64",
        "model.model_params.revin=1",
        "exp.learning_rate=" + ",".join(str(value) for value in SONNET_LEARNING_RATES),
        "model.model_params.n_atoms=" + ",".join(str(value) for value in SONNET_ATOMS),
        "model.model_params.alpha=" + ",".join(str(value) for value in SONNET_ALPHAS),
    ]


def _write_command_manifest(
    path: Path, experiments: Sequence[ForecastExperiment], commands: Sequence[list[str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for experiment, command in zip(experiments, commands):
            handle.write(
                json.dumps(
                    {
                        "experiment_id": experiment.experiment_id,
                        "command": command,
                        "command_shell": shlex.join(command),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _execute_commands(
    experiments: Sequence[ForecastExperiment], commands: Sequence[list[str]], dry_run: bool
) -> None:
    for experiment, command in zip(experiments, commands):
        print(f"[{experiment.experiment_id}] {shlex.join(command)}")
        if dry_run:
            continue
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--only", action="append", default=[], help="Experiment id to select; repeatable."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate all source NPZ pairs.")
    _common_parser(validate)

    hpo = subparsers.add_parser("hpo", help="Run or dry-run the 4-point TimeVAE HPO.")
    _common_parser(hpo)
    hpo.add_argument("--python", default=sys.executable)
    hpo.add_argument("--gpu-slots", default="0:1")
    hpo.add_argument("--skip-completed", action="store_true")
    hpo.add_argument("--allow-cpu-fallback", action="store_true")
    hpo.add_argument("--dry-run", action="store_true")
    hpo.add_argument(
        "--commands-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "forecasting_hpo_commands.jsonl",
    )

    rerun = subparsers.add_parser("rerun", help="Generate full-N NPZs from best HPO runs.")
    _common_parser(rerun)
    rerun.add_argument("--python", default=sys.executable)
    rerun.add_argument("--with-tsne", action="store_true")
    rerun.add_argument("--dry-run", action="store_true")
    rerun.add_argument(
        "--commands-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "forecasting_rerun_commands.jsonl",
    )

    export = subparsers.add_parser(
        "export-sonnet", help="Validate reruns and export Sonnet HPO commands."
    )
    _common_parser(export)
    export.add_argument("--sonnet-root", type=Path, default=REPO_ROOT.parent / "Sonnet")
    export.add_argument("--sonnet-python", default="python3")
    export.add_argument("--gpu-slots", default="0:1")
    export.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "manifests" / "forecasting_timevae.csv",
    )
    export.add_argument(
        "--commands-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "manifests" / "forecasting_timevae_sonnet.sh",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = select_experiments(load_manifest(args.manifest), args.only)
    if args.command == "validate":
        for experiment in experiments:
            contract = validate_source_pair(experiment)
            print(
                f"[ok] {experiment.experiment_id}: "
                f"N={contract['num_real_windows']} "
                f"shape={contract['train_data'].shape[1:]}"
            )
        return

    if args.command == "hpo":
        if not args.dry_run:
            for experiment in experiments:
                validate_source_pair(experiment)
        commands = [
            build_hpo_command(
                experiment,
                python_executable=args.python,
                gpu_slots=args.gpu_slots,
                skip_completed=args.skip_completed,
                allow_cpu_fallback=args.allow_cpu_fallback,
            )
            for experiment in experiments
        ]
        _write_command_manifest(args.commands_output, experiments, commands)
        _execute_commands(experiments, commands, args.dry_run)
        return

    if args.command == "rerun":
        if not args.dry_run:
            for experiment in experiments:
                validate_source_pair(experiment)
        commands = [
            build_rerun_command(
                experiment,
                args.manifest,
                python_executable=args.python,
                no_tsne=not args.with_tsne,
            )
            for experiment in experiments
        ]
        _write_command_manifest(args.commands_output, experiments, commands)
        _execute_commands(experiments, commands, args.dry_run)
        return

    if args.command == "export-sonnet":
        rows = []
        commands = []
        for experiment in experiments:
            meta = validate_generated_npz(experiment)
            rows.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "method": "timevae",
                    "dataset": experiment.dataset,
                    "fold": experiment.fold,
                    "lookback": experiment.lookback,
                    "pred_len": experiment.pred_len,
                    "seq_len": experiment.seq_len,
                    "target_column": experiment.target_column,
                    "num_generated": meta["num_generated"],
                    "synthetic_npz": str(experiment.generated_npz),
                    "synthetic_npz_sha256": sha256_file(experiment.generated_npz),
                    "source_train_sha256": meta["source_train_sha256"],
                }
            )
            commands.append(
                build_sonnet_command(
                    experiment,
                    args.sonnet_root,
                    sonnet_python=args.sonnet_python,
                    gpu_slots=args.gpu_slots,
                )
            )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        args.commands_output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for experiment, command in zip(experiments, commands):
            lines.extend((f"# {experiment.experiment_id}", shlex.join(command), ""))
        args.commands_output.write_text("\n".join(lines), encoding="utf-8")
        os.chmod(args.commands_output, 0o755)
        print(f"Wrote {args.output_csv} and {args.commands_output}")
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
