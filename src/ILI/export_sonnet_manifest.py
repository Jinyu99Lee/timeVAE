#!/usr/bin/env python3
"""Export completed ILI TimeVAE reruns through Sonnet's generic CSV contract.

The source JSONL manifest remains the authority for subset identity and feature
ordering.  For every row, this exporter follows the latest completed
``best_run.json`` to that exact run's ``best_outputs/rerun_config.json`` and
refuses any train/validation/source mismatch.  It never copies or rewrites the
generated NPZ files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import ili_config as cfg


SCHEMA_VERSION = 1
SUBSET_RE = re.compile(
    r"^ili_(?P<region>[A-Za-z0-9]+)_(?P<year>\d{4}_\d{4})_"
    r"T(?P<T>\d+)_p(?P<pred>\d+)$"
)
OUTPUT_FIELDS = [
    "schema_version",
    "method",
    "dataset",
    "region",
    "year",
    "h",
    "tau",
    "T",
    "pred_length",
    "synthetic_path",
    "feature_cols_npz",
    "subset_id",
    "rerun_config",
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    manifest = path.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"ILI source manifest not found: {manifest}")
    rows: list[dict[str, Any]] = []
    seen_subset: set[str] = set()
    seen_train: set[Path] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {manifest}:{line_number}: {exc}") from exc
        required = {
            "subset_id",
            "name",
            "tau",
            "train_path",
            "val_path",
            "N_train",
            "T",
            "D",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"{manifest}:{line_number} is missing: {', '.join(missing)}"
            )
        subset_id = str(row["subset_id"])
        train_path = Path(row["train_path"]).expanduser().resolve()
        if subset_id in seen_subset:
            raise ValueError(f"Duplicate subset_id={subset_id!r} in {manifest}.")
        if train_path in seen_train:
            raise ValueError(f"Duplicate train_path={train_path} in {manifest}.")
        seen_subset.add(subset_id)
        seen_train.add(train_path)
        rows.append(row)
    if not rows:
        raise ValueError(f"ILI source manifest contains no rows: {manifest}")
    return rows


def tau_folder_to_decimal(folder: str) -> str:
    """Convert tau03/tau005 to 0.3/0.05 without float round-off."""

    match = re.fullmatch(r"tau(\d+)", str(folder).strip())
    if match is None:
        raise ValueError(f"Invalid tau folder {folder!r}; expected e.g. tau03.")
    digits = match.group(1)
    value = Decimal(int(digits)).scaleb(-(len(digits) - 1))
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def find_latest_best_run(hpo_root: Path, subset_id: str) -> Path | None:
    candidates = sorted(hpo_root.glob(f"*_{subset_id}/best_run.json"))
    return candidates[-1].resolve() if candidates else None


def _same_path(left: object, right: object) -> bool:
    return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()


def build_export_row(
    source: dict[str, Any],
    *,
    best_run_path: Path,
    method: str,
    dataset: str,
    validate_arrays: bool,
) -> dict[str, object]:
    subset_id = str(source["subset_id"])
    if not best_run_path.parent.name.endswith(f"_{subset_id}"):
        raise ValueError(
            f"best_run experiment directory does not end with subset_id={subset_id}: "
            f"{best_run_path.parent}"
        )

    best = _load_json(best_run_path)
    if best.get("status") != "completed" or best.get("error") not in {None, ""}:
        raise ValueError(f"Best run is not completed cleanly: {best_run_path}")
    for field, source_field in (
        ("dataset", "name"),
        ("train_npz", "train_path"),
        ("val_npz", "val_path"),
    ):
        actual, expected = best.get(field), source[source_field]
        equal = str(actual) == str(expected) if field == "dataset" else _same_path(actual, expected)
        if not equal:
            raise ValueError(
                f"{field} mismatch for {subset_id}: {actual!r} != {expected!r}"
            )

    run_dir = Path(str(best.get("run_dir"))).expanduser().resolve()
    rerun_path = run_dir / "best_outputs" / "rerun_config.json"
    if not rerun_path.is_file():
        raise FileNotFoundError(f"Missing rerun_config for current best run: {rerun_path}")
    rerun = _load_json(rerun_path)
    checks = (
        ("best_run_path", best_run_path),
        ("source_run_dir", run_dir),
        ("train_npz", source["train_path"]),
        ("val_npz", source["val_path"]),
    )
    for field, expected in checks:
        if not _same_path(rerun.get(field), expected):
            raise ValueError(
                f"rerun_config {field} mismatch for {subset_id}: "
                f"{rerun.get(field)!r} != {str(expected)!r}"
            )
    if rerun.get("dataset") != source["name"]:
        raise ValueError(
            f"rerun_config dataset mismatch for {subset_id}: "
            f"{rerun.get('dataset')!r} != {source['name']!r}"
        )

    generated = Path(str(rerun.get("generated_file"))).expanduser().resolve()
    if not generated.is_file():
        raise FileNotFoundError(f"Generated NPZ not found for {subset_id}: {generated}")
    if generated.parent != rerun_path.parent:
        raise ValueError(
            f"generated_file is not in the selected run's best_outputs directory: {generated}"
        )

    expected_shape = (
        int(source["N_train"]),
        int(source["T"]),
        int(source["D"]),
    )
    declared_shape = tuple(int(v) for v in rerun.get("generated_shape", []))
    if declared_shape != expected_shape:
        raise ValueError(
            f"rerun generated_shape mismatch for {subset_id}: "
            f"{declared_shape} != {expected_shape}"
        )

    train_path = Path(source["train_path"]).expanduser().resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"Source train NPZ not found for {subset_id}: {train_path}")
    with np.load(train_path, allow_pickle=True) as payload:
        if "feature_cols" not in payload.files:
            raise ValueError(f"Source train NPZ has no feature_cols: {train_path}")
        if len(payload["feature_cols"]) != expected_shape[2]:
            raise ValueError(
                f"feature_cols length mismatch for {subset_id}: "
                f"{len(payload['feature_cols'])} != {expected_shape[2]}"
            )

    if validate_arrays:
        with np.load(generated, allow_pickle=False) as payload:
            if "data" not in payload.files:
                raise ValueError(f"Generated NPZ has no data array: {generated}")
            data = payload["data"]
            if tuple(int(v) for v in data.shape) != expected_shape:
                raise ValueError(
                    f"Generated data shape mismatch for {subset_id}: "
                    f"{data.shape} != {expected_shape}"
                )
            if not np.issubdtype(data.dtype, np.number):
                raise ValueError(f"Generated data is not numeric for {subset_id}: {data.dtype}")
            if not np.isfinite(data).all():
                raise ValueError(f"Generated data contains NaN/Inf for {subset_id}: {generated}")

    match = SUBSET_RE.fullmatch(str(source["name"]))
    if match is None:
        raise ValueError(f"Unexpected ILI subset name: {source['name']!r}")
    timesteps = int(match.group("T"))
    pred_length = int(match.group("pred"))
    if timesteps != int(source["T"]):
        raise ValueError(
            f"Name/manifest T mismatch for {subset_id}: {timesteps} != {source['T']}"
        )
    if subset_id != f"{source['name']}_{source['tau']}":
        raise ValueError(
            f"subset_id does not equal name_tau for {subset_id}: "
            f"{source['name']}_{source['tau']}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "dataset": dataset,
        "region": match.group("region"),
        "year": match.group("year"),
        "h": f"h{pred_length}",
        "tau": tau_folder_to_decimal(str(source["tau"])),
        "T": timesteps,
        "pred_length": pred_length,
        "synthetic_path": str(generated),
        "feature_cols_npz": str(train_path),
        "subset_id": subset_id,
        "rerun_config": str(rerun_path),
    }


def write_csv_atomic(rows: Iterable[dict[str, object]], output: Path) -> None:
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=cfg.MANIFEST_PATH)
    parser.add_argument(
        "--hpo-root",
        type=Path,
        default=cfg.REPO_ROOT / "outputs" / "hpo" / cfg.slug(cfg.EXPERIMENT_GROUP),
        help="Exact TimeVAE ILI HPO group directory to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.REPO_ROOT / "outputs" / "manifests" / f"{cfg.ILI_DATASET}_timevae.csv",
    )
    parser.add_argument("--method", default="timevae")
    parser.add_argument("--dataset", default=cfg.ILI_DATASET)
    completeness = parser.add_mutually_exclusive_group()
    completeness.add_argument("--strict", dest="strict", action="store_true", default=True)
    completeness.add_argument("--allow-missing", dest="strict", action="store_false")
    parser.add_argument(
        "--skip-array-validation",
        action="store_true",
        help="Trust rerun_config generated_shape instead of reading every generated array.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    method = str(args.method).strip()
    dataset = str(args.dataset).strip()
    if not method or not dataset:
        raise ValueError("--method and --dataset must be non-empty.")

    hpo_root = args.hpo_root.expanduser().resolve()
    if not hpo_root.is_dir():
        raise FileNotFoundError(f"HPO root not found: {hpo_root}")
    source_rows = load_source_rows(args.manifest)
    exported: list[dict[str, object]] = []
    problems: list[str] = []
    for source in source_rows:
        subset_id = str(source["subset_id"])
        best_run_path = find_latest_best_run(hpo_root, subset_id)
        if best_run_path is None:
            problems.append(f"{subset_id}: no best_run.json under {hpo_root}")
            continue
        try:
            exported.append(
                build_export_row(
                    source,
                    best_run_path=best_run_path,
                    method=method,
                    dataset=dataset,
                    validate_arrays=not args.skip_array_validation,
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            problems.append(f"{subset_id}: {exc}")

    if problems and args.strict:
        details = "\n  - ".join(problems)
        raise RuntimeError(
            f"Strict export refused {len(problems)} incomplete/invalid subset(s):\n  - {details}"
        )
    for problem in problems:
        print(f"WARNING: {problem}")
    if not exported:
        raise RuntimeError("No valid TimeVAE rerun outputs were found.")

    exported.sort(
        key=lambda row: (
            str(row["region"]),
            str(row["year"]),
            int(row["pred_length"]),
            Decimal(str(row["tau"])),
        )
    )
    write_csv_atomic(exported, args.output)
    print(
        f"Exported {len(exported)}/{len(source_rows)} TimeVAE subsets to "
        f"{args.output.expanduser().resolve()}"
    )
    print(f"Method={method} dataset={dataset} strict={args.strict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
