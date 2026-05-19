#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TIME_COLS = {"time", "date", "datetime", "timestamp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot random synthetic/original NPZ samples into per-sample PDFs."
    )
    parser.add_argument("--synthetic-npz", type=Path, required=True)
    parser.add_argument("--original-npz", type=Path, required=True)
    parser.add_argument("--num-pdfs", type=int, required=True, help="Number of PDF files to create.")
    parser.add_argument("--original-csv", type=Path, default=None, help="CSV used only to infer feature names.")
    parser.add_argument("--feature-names", nargs="+", default=None, help="Manual feature names, one per D dimension.")
    parser.add_argument("--output-subdir", default="plots", help="Subfolder created next to synthetic npz. Default: plots.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-label", default="Synthetic")
    parser.add_argument("--original-label", default="Original")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_npz_data(path: Path) -> tuple[np.ndarray, list[str] | None]:
    loaded = np.load(path, allow_pickle=False)
    data = loaded["data"]
    feature_cols = None
    if "feature_cols" in loaded.files:
        feature_cols = [str(col) for col in loaded["feature_cols"].tolist()]
    return data, feature_cols


def feature_names_from_csv(path: Path) -> list[str]:
    with path.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
    return [col for col in header if col.lower() not in DEFAULT_TIME_COLS]


def resolve_feature_names(
    dim: int,
    manual_names: Sequence[str] | None,
    original_feature_cols: Sequence[str] | None,
    original_csv: Path | None,
) -> list[str]:
    if manual_names is not None:
        names = list(manual_names)
    elif original_feature_cols is not None:
        names = list(original_feature_cols)
    elif original_csv is not None:
        names = feature_names_from_csv(original_csv)
    else:
        names = [f"feature_{idx}" for idx in range(dim)]

    if len(names) != dim:
        raise ValueError(f"Expected {dim} feature names, got {len(names)}: {names}")
    return names


def validate_arrays(synthetic: np.ndarray, original: np.ndarray) -> None:
    if synthetic.ndim != 3 or original.ndim != 3:
        raise ValueError(
            "Both npz files must contain data with shape (N, T, D). "
            f"synthetic={synthetic.shape}, original={original.shape}"
        )
    if synthetic.shape[1:] != original.shape[1:]:
        raise ValueError(
            "Synthetic and original samples must share the same (T, D). "
            f"synthetic={synthetic.shape}, original={original.shape}"
        )


def plot_one_pdf(
    output_file: Path,
    synthetic_sample: np.ndarray,
    original_sample: np.ndarray,
    synthetic_index: int,
    original_index: int,
    feature_names: Sequence[str],
    synthetic_label: str,
    original_label: str,
    dpi: int,
) -> None:
    _, dim = synthetic_sample.shape
    fig, axes = plt.subplots(dim, 2, figsize=(12, max(2.2 * dim, 5)), sharex=True)
    if dim == 1:
        axes = np.array([axes])

    x_axis = np.arange(synthetic_sample.shape[0])
    for feature_idx, feature_name in enumerate(feature_names):
        left = axes[feature_idx, 0]
        right = axes[feature_idx, 1]
        left.plot(x_axis, synthetic_sample[:, feature_idx], linewidth=1.4)
        right.plot(x_axis, original_sample[:, feature_idx], linewidth=1.4, color="tab:orange")
        left.set_ylabel(feature_name)
        right.set_ylabel(feature_name)
        left.grid(alpha=0.25)
        right.grid(alpha=0.25)
        if feature_idx == 0:
            left.set_title(f"{synthetic_label} idx={synthetic_index}")
            right.set_title(f"{original_label} idx={original_index}")
        if feature_idx == dim - 1:
            left.set_xlabel("timestep")
            right.set_xlabel("timestep")

    fig.suptitle("Random synthetic and original time-series samples", y=0.995)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_pdfs <= 0:
        raise ValueError("--num-pdfs must be positive.")

    synthetic, _ = load_npz_data(args.synthetic_npz)
    original, original_feature_cols = load_npz_data(args.original_npz)
    validate_arrays(synthetic, original)

    feature_names = resolve_feature_names(
        dim=synthetic.shape[2],
        manual_names=args.feature_names,
        original_feature_cols=original_feature_cols,
        original_csv=args.original_csv,
    )

    output_dir = args.synthetic_npz.parent / args.output_subdir
    rng = np.random.default_rng(args.seed)
    manifest = []
    for pdf_idx in range(args.num_pdfs):
        synthetic_index = int(rng.integers(0, synthetic.shape[0]))
        original_index = int(rng.integers(0, original.shape[0]))
        output_file = output_dir / f"sample_pair_{pdf_idx + 1:03d}.pdf"
        plot_one_pdf(
            output_file=output_file,
            synthetic_sample=synthetic[synthetic_index],
            original_sample=original[original_index],
            synthetic_index=synthetic_index,
            original_index=original_index,
            feature_names=feature_names,
            synthetic_label=args.synthetic_label,
            original_label=args.original_label,
            dpi=args.dpi,
        )
        manifest.append(
            {
                "pdf": str(output_file),
                "synthetic_index": synthetic_index,
                "original_index": original_index,
            }
        )

    manifest_file = output_dir / "manifest.csv"
    with manifest_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("pdf", "synthetic_index", "original_index"))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Wrote {args.num_pdfs} PDF files to {output_dir}")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
