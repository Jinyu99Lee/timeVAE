from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_run_dir(best_run: dict[str, Any], best_run_path: Path) -> Path:
    run_dir = Path(best_run["run_dir"])
    if run_dir.is_absolute():
        return run_dir
    return (best_run_path.parent / run_dir).resolve()


def load_npz_data_and_feature_names(path: Path) -> tuple[np.ndarray, list[str] | None]:
    loaded = np.load(path, allow_pickle=False)
    data = loaded["data"]
    feature_names = None
    if "feature_cols" in loaded.files:
        feature_names = [str(col) for col in loaded["feature_cols"].tolist()]
    return data, feature_names


def resolve_feature_names(dim: int, feature_names: Sequence[str] | None) -> list[str]:
    if feature_names is None:
        return [f"feature_{idx}" for idx in range(dim)]
    resolved = [str(name) for name in feature_names]
    if len(resolved) != dim:
        raise ValueError(
            f"Expected {dim} feature names, got {len(resolved)}: {resolved}"
        )
    return resolved


def select_split(train_data: np.ndarray, valid_data: np.ndarray, split: str) -> np.ndarray:
    if split == "train":
        return train_data
    if split == "valid":
        return valid_data
    if split == "all":
        return np.concatenate([train_data, valid_data], axis=0)
    raise ValueError(f"Unsupported split: {split}")


def validate_series_pair(real_data: np.ndarray, synthetic_data: np.ndarray) -> None:
    if real_data.ndim != 3 or synthetic_data.ndim != 3:
        raise ValueError(
            "Both arrays must have shape (N, T, D). "
            f"real={real_data.shape}, synthetic={synthetic_data.shape}"
        )
    if real_data.shape[1:] != synthetic_data.shape[1:]:
        raise ValueError(
            "Real and synthetic data must share (T, D). "
            f"real={real_data.shape}, synthetic={synthetic_data.shape}"
        )


def freedman_diaconis_num_bins(num_values: int) -> int:
    if num_values <= 0:
        raise ValueError("num_values must be positive.")
    return max(1, int(round(2.0 * math.pow(num_values, 1.0 / 3.0), 0)))


def resolve_num_bins(real_data: np.ndarray, requested_num_bins: int | None) -> int:
    if requested_num_bins is not None:
        if requested_num_bins <= 0:
            raise ValueError("--num-bins must be positive when provided.")
        return int(requested_num_bins)
    return freedman_diaconis_num_bins(int(real_data.shape[0] * real_data.shape[1]))


def finite_feature_values(data: np.ndarray, feature_idx: int) -> np.ndarray:
    values = np.asarray(data[:, :, feature_idx], dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def histogram_distance_1d(
    real_values: np.ndarray, synthetic_values: np.ndarray, num_bins: int
) -> float:
    real_values = real_values[np.isfinite(real_values)]
    synthetic_values = synthetic_values[np.isfinite(synthetic_values)]
    if real_values.size == 0 or synthetic_values.size == 0:
        return float("nan")

    min_val = float(np.min(real_values))
    max_val = float(np.max(real_values))
    if abs(max_val - min_val) < 1e-10:
        min_val -= 1e-5
        max_val += 1e-5

    bins = np.linspace(min_val, max_val, num_bins + 1)
    real_counts, _ = np.histogram(real_values, bins=bins)
    synthetic_counts, _ = np.histogram(synthetic_values, bins=bins)

    real_density = real_counts / real_values.size * num_bins
    synthetic_density = synthetic_counts / synthetic_values.size * num_bins
    abs_metric = float(np.mean(np.abs(real_density - synthetic_density)))
    out_of_bounds = float(
        np.mean((synthetic_values < bins[0]) | (synthetic_values > bins[-1]))
    )
    return (abs_metric + out_of_bounds) / 2.0


def wasserstein_1d(real_values: np.ndarray, synthetic_values: np.ndarray) -> float:
    real_values = np.sort(real_values[np.isfinite(real_values)])
    synthetic_values = np.sort(synthetic_values[np.isfinite(synthetic_values)])
    if real_values.size == 0 or synthetic_values.size == 0:
        return float("nan")
    all_values = np.sort(np.concatenate([real_values, synthetic_values]))
    if all_values.size <= 1:
        return 0.0
    deltas = np.diff(all_values)
    if not np.any(deltas):
        return 0.0
    sample_points = all_values[:-1]
    real_cdf = (
        np.searchsorted(real_values, sample_points, side="right") / real_values.size
    )
    synthetic_cdf = (
        np.searchsorted(synthetic_values, sample_points, side="right")
        / synthetic_values.size
    )
    return float(np.sum(np.abs(real_cdf - synthetic_cdf) * deltas))


def compute_feature_statistics(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    feature_names: Sequence[str] | None,
    num_bins: int,
) -> list[dict[str, Any]]:
    validate_series_pair(real_data, synthetic_data)
    names = resolve_feature_names(real_data.shape[2], feature_names)
    rows: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(names):
        real_values = finite_feature_values(real_data, feature_idx)
        synthetic_values = finite_feature_values(synthetic_data, feature_idx)
        real_mean = float(np.mean(real_values)) if real_values.size else float("nan")
        synthetic_mean = (
            float(np.mean(synthetic_values)) if synthetic_values.size else float("nan")
        )
        real_variance = float(np.var(real_values)) if real_values.size else float("nan")
        synthetic_variance = (
            float(np.var(synthetic_values)) if synthetic_values.size else float("nan")
        )
        mean_diff = synthetic_mean - real_mean
        variance_diff = synthetic_variance - real_variance
        rows.append(
            {
                "feature": feature_name,
                "real_mean": real_mean,
                "generated_mean": synthetic_mean,
                "mean_diff": mean_diff,
                "abs_mean_diff": abs(mean_diff),
                "real_variance": real_variance,
                "generated_variance": synthetic_variance,
                "variance_diff": variance_diff,
                "abs_variance_diff": abs(variance_diff),
                "histogram_distance": histogram_distance_1d(
                    real_values, synthetic_values, num_bins=num_bins
                ),
                "wasserstein_1": wasserstein_1d(real_values, synthetic_values),
                "real_count": int(real_values.size),
                "generated_count": int(synthetic_values.size),
            }
        )

    metric_keys = (
        "abs_mean_diff",
        "abs_variance_diff",
        "histogram_distance",
        "wasserstein_1",
    )
    average_row: dict[str, Any] = {"feature": "AVERAGE"}
    for key in metric_keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        average_row[key] = float(np.nanmean(values)) if values.size else float("nan")
    for key in rows[0].keys() if rows else ():
        average_row.setdefault(key, "")
    rows.append(average_row)
    return rows


def write_feature_statistics_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No feature statistics rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_feature_histograms_pdf(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    feature_names: Sequence[str] | None,
    output_file: Path,
    num_bins: int,
    real_label: str,
    synthetic_label: str,
) -> None:
    validate_series_pair(real_data, synthetic_data)
    names = resolve_feature_names(real_data.shape[2], feature_names)
    dim = real_data.shape[2]
    fig_height = max(2.0 * dim, 5.0)
    fig, axes = plt.subplots(dim, 1, figsize=(9.5, fig_height))
    if dim == 1:
        axes = np.array([axes])

    for feature_idx, feature_name in enumerate(names):
        ax = axes[feature_idx]
        real_values = finite_feature_values(real_data, feature_idx)
        synthetic_values = finite_feature_values(synthetic_data, feature_idx)
        combined = np.concatenate([real_values, synthetic_values])
        combined = combined[np.isfinite(combined)]
        if combined.size == 0:
            ax.set_ylabel(feature_name)
            continue
        min_val = float(np.min(combined))
        max_val = float(np.max(combined))
        if abs(max_val - min_val) < 1e-10:
            min_val -= 1e-5
            max_val += 1e-5
        bins = np.linspace(min_val, max_val, num_bins + 1)
        if real_values.size:
            ax.hist(
                real_values,
                bins=bins,
                weights=np.ones(real_values.size) / real_values.size,
                alpha=0.55,
                color="#1f77b4",
                label=real_label,
            )
        if synthetic_values.size:
            ax.hist(
                synthetic_values,
                bins=bins,
                weights=np.ones(synthetic_values.size) / synthetic_values.size,
                alpha=0.45,
                color="#ff7f0e",
                label=synthetic_label,
            )
        ax.set_ylabel(feature_name)
        ax.grid(alpha=0.2)
        if feature_idx == 0:
            ax.legend(frameon=False)
        if feature_idx == dim - 1:
            ax.set_xlabel("Value")

    fig.suptitle(f"Feature distributions: {real_label} vs {synthetic_label}", y=0.995)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)


def save_tsne_plot(
    original_samples: np.ndarray,
    generated_samples: np.ndarray,
    original_name: str,
    generated_name: str,
    output_file: Path,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    validate_series_pair(original_samples, generated_samples)
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

    fig, ax = plt.subplots(figsize=(8, 8))
    original_points = embedded[:original_used]
    generated_points = embedded[original_used:]
    ax.scatter(
        original_points[:, 0],
        original_points[:, 1],
        label=original_name,
        color="red",
        alpha=0.5,
        s=70,
    )
    ax.scatter(
        generated_points[:, 0],
        generated_points[:, 1],
        label=generated_name,
        color="blue",
        alpha=0.5,
        s=70,
    )
    ax.set_title(f"t-SNE: {original_name} vs {generated_name}")
    ax.legend()
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)

    return {
        "original_used": original_used,
        "generated_used": generated_used,
        "perplexity": perplexity,
        "output_file": str(output_file),
    }


def save_distribution_evaluation(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    output_dir: Path,
    feature_names: Sequence[str] | None,
    num_bins: int | None,
    real_label: str,
    synthetic_label: str,
    make_histograms: bool = True,
) -> dict[str, Any]:
    validate_series_pair(real_data, synthetic_data)
    resolved_num_bins = resolve_num_bins(real_data, num_bins)
    rows = compute_feature_statistics(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        num_bins=resolved_num_bins,
    )
    statistics_file = output_dir / "feature_statistics.csv"
    histograms_file = output_dir / "feature_distribution_histograms.pdf"
    write_feature_statistics_csv(statistics_file, rows)
    # The per-feature histogram PDF stacks one subplot row per feature, so for
    # high-dimensional data (e.g. ILI with 818 features) it is both unreadable
    # and exceeds matplotlib's 2^16-pixel figure limit. Allow callers to skip it
    # while still writing the per-feature statistics CSV and averaged metrics.
    if make_histograms:
        plot_feature_histograms_pdf(
            real_data=real_data,
            synthetic_data=synthetic_data,
            feature_names=feature_names,
            output_file=histograms_file,
            num_bins=resolved_num_bins,
            real_label=real_label,
            synthetic_label=synthetic_label,
        )
    average_row = rows[-1]
    return {
        "num_bins": resolved_num_bins,
        "feature_statistics_file": str(statistics_file),
        "feature_histograms_file": str(histograms_file) if make_histograms else None,
        "average_abs_mean_diff": average_row["abs_mean_diff"],
        "average_abs_variance_diff": average_row["abs_variance_diff"],
        "average_histogram_distance": average_row["histogram_distance"],
        "average_wasserstein_1": average_row["wasserstein_1"],
    }
