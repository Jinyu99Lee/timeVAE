#!/usr/bin/env python3
"""Post-hoc memorization check for generated time-series windows.

Compares nearest-neighbour (NN) distances of three window sets:

  * gen->train    each generated window's distance to its closest train window
                  (the subject under test);
  * val->train    each validation window's distance to its closest train window
                  (the reference: how far a *genuinely novel* same-distribution
                  window sits from the train set);
  * train->train  each train window's distance to its closest *other* train
                  window (a lower bound only -- with stride-1 sliding windows
                  neighbouring windows share almost all their content, so this
                  distribution sits near zero by construction).

If gen->train distances are systematically below val->train, the generator is
reproducing its training data rather than sampling novel series.

Standalone: needs only numpy (+ matplotlib for the PDFs). Works on the npz
files produced by Diffusion-TS / DiMTS (``best_synth.npz``) and timeVAE
(``*_best_prior_samples.npz``); all store an ``(N, T, D)`` array under ``data``.

Usage:
  python check_memorization.py \
      --train-npz Data/datasets/.../xxx_train.npz \
      --synth-npz outputs/hpo/.../best_synth.npz \
      --val-npz   Data/datasets/.../xxx_val.npz \
      [--metric euclidean|dtw] [--top-k 8] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

PERCENTILES = (1, 5, 25, 50, 75, 95)
COPY_RATIO_ALARM = 0.5      # median(gen->train) / median(val->train) below this
FRAC_BELOW_P05_ALARM = 0.1  # >10% of gen windows closer than val's 5th pctile


def load_windows(path, key="data"):
    """Load an (N, T, D) float array from an npz file.

    Falls back to the first 3-D array in the file when ``key`` is absent.
    """
    with np.load(path, allow_pickle=True) as npz:
        if key in npz.files:
            arr = np.asarray(npz[key], dtype=np.float64)
        else:
            arr = None
            for name in npz.files:
                cand = np.asarray(npz[name])
                if cand.ndim == 3 and np.issubdtype(cand.dtype, np.number):
                    arr = cand.astype(np.float64)
                    print(f"[load] key '{key}' not in {path}; using '{name}'")
                    break
            if arr is None:
                raise KeyError(
                    f"{path}: no key '{key}' and no 3-D numeric array found "
                    f"(keys: {npz.files})")
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected (N, T, D) array, got {arr.shape}")
    return arr


def nn_rmse(queries, references, exclude_self=False, chunk=512):
    """Per-query RMSE distance to its nearest reference window.

    Windows are flattened to T*D vectors; distance is the Euclidean distance
    normalised by sqrt(T*D) so values read as per-timepoint RMSE.
    """
    n, t, d = queries.shape
    q = queries.reshape(n, -1)
    r = references.reshape(references.shape[0], -1)
    r_sq = (r ** 2).sum(axis=1)
    out = np.empty(n)
    for start in range(0, n, chunk):
        qc = q[start:start + chunk]
        d2 = (qc ** 2).sum(axis=1)[:, None] + r_sq[None, :] - 2.0 * qc @ r.T
        np.maximum(d2, 0.0, out=d2)
        if exclude_self:
            rows = np.arange(start, min(start + chunk, n))
            d2[np.arange(len(rows)), rows] = np.inf
        out[start:start + chunk] = np.sqrt(d2.min(axis=1) / (t * d))
    return out


def nn_rmse_argmin(queries, references, chunk=512):
    """Like nn_rmse but also returns the index of the nearest reference."""
    n, t, d = queries.shape
    q = queries.reshape(n, -1)
    r = references.reshape(references.shape[0], -1)
    r_sq = (r ** 2).sum(axis=1)
    dist, idx = np.empty(n), np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk):
        qc = q[start:start + chunk]
        d2 = (qc ** 2).sum(axis=1)[:, None] + r_sq[None, :] - 2.0 * qc @ r.T
        np.maximum(d2, 0.0, out=d2)
        rows = np.arange(d2.shape[0])
        idx[start:start + chunk] = d2.argmin(axis=1)
        dist[start:start + chunk] = np.sqrt(
            d2[rows, idx[start:start + chunk]] / (t * d))
    return dist, idx


def nn_dtw(queries, references, exclude_self=False):
    """Per-query DTW distance (dependent multivariate) to nearest reference.

    Requires the optional ``dtaidistance`` package. O(N_q * N_r * T^2) -- fine
    for a few hundred windows, slow beyond that.
    """
    try:
        from dtaidistance import dtw_ndim
    except ImportError as e:
        raise SystemExit(
            "--metric dtw requires the 'dtaidistance' package "
            "(pip install dtaidistance), or use the default euclidean metric."
        ) from e
    t = queries.shape[1]
    out = np.empty(queries.shape[0])
    for i, qw in enumerate(queries):
        best = np.inf
        for j, rw in enumerate(references):
            if exclude_self and i == j:
                continue
            best = min(best, dtw_ndim.distance(qw, rw))
        out[i] = best / np.sqrt(t)
    return out


def distance_stats(dist):
    dist = np.asarray(dist, dtype=float)
    stats = {
        "n": int(dist.size),
        "mean": float(dist.mean()),
        "min": float(dist.min()),
        "max": float(dist.max()),
    }
    for p in PERCENTILES:
        stats[f"p{p:02d}"] = float(np.percentile(dist, p))
    return stats


def save_ecdf_plot(dists, out_path):
    """ECDF overlay of the available NN-distance distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "gen->train": dict(color="tab:blue", linestyle="-", linewidth=1.8),
        "val->train": dict(color="tab:orange", linestyle="-", linewidth=1.8),
        "train->train": dict(color="0.55", linestyle="--", linewidth=1.2),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, dist in dists.items():
        x = np.sort(dist)
        y = np.arange(1, x.size + 1) / x.size
        ax.step(x, y, where="post", label=f"{name} (n={x.size})", **styles[name])
    ax.set_xlabel("NN distance (per-timepoint RMSE, train-standardized)")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("Nearest-neighbour distance to train set")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_topk_plot(synth, train, dist, nn_idx, out_path, top_k, max_features=4):
    """Overlay the top-k most train-like generated windows with their train NN."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = min(top_k, synth.shape[0])
    order = np.argsort(dist)[:k]
    n_feat = min(synth.shape[2], max_features)
    fig, axes = plt.subplots(
        k, n_feat, figsize=(3.2 * n_feat, 1.9 * k), squeeze=False, sharex=True)
    for row, gi in enumerate(order):
        ti = nn_idx[gi]
        for col in range(n_feat):
            ax = axes[row][col]
            ax.plot(synth[gi, :, col], color="tab:blue", linewidth=1.4,
                    label="generated" if row == col == 0 else None)
            ax.plot(train[ti, :, col], color="tab:orange", linewidth=1.4,
                    linestyle="--",
                    label="train NN" if row == col == 0 else None)
            if col == 0:
                ax.set_ylabel(f"gen#{gi}\nd={dist[gi]:.4f}", fontsize=8)
            if row == 0:
                ax.set_title(f"feature {col}", fontsize=9)
            ax.tick_params(labelsize=7)
    if synth.shape[2] > n_feat:
        fig.suptitle(f"Top-{k} closest generated windows vs their train NN "
                     f"(first {n_feat}/{synth.shape[2]} features)")
    else:
        fig.suptitle(f"Top-{k} closest generated windows vs their train NN")
    fig.legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="NN-distance memorization check for generated windows.")
    p.add_argument("--train-npz", required=True, help="Generative train set npz.")
    p.add_argument("--synth-npz", required=True, help="Generated samples npz.")
    p.add_argument("--val-npz", default=None,
                   help="Validation npz (novelty reference; recommended).")
    p.add_argument("--key", default="data", help="Array key inside the npz.")
    p.add_argument("--metric", default="euclidean",
                   choices=["euclidean", "dtw"])
    p.add_argument("--top-k", type=int, default=8,
                   help="Windows shown in the closest-pairs plot.")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: alongside --synth-npz).")
    args = p.parse_args()

    train = load_windows(args.train_npz, args.key)
    synth = load_windows(args.synth_npz, args.key)
    val = load_windows(args.val_npz, args.key) if args.val_npz else None
    if synth.shape[1:] != train.shape[1:]:
        raise ValueError(
            f"synth windows {synth.shape[1:]} do not match train {train.shape[1:]}")
    if val is not None and val.shape[1:] != train.shape[1:]:
        raise ValueError(
            f"val windows {val.shape[1:]} do not match train {train.shape[1:]}")

    # Standardize every set with per-feature train statistics so distances are
    # comparable across features of different scales.
    mu = train.mean(axis=(0, 1), keepdims=True)
    sd = train.std(axis=(0, 1), keepdims=True)
    sd[sd < 1e-8] = 1.0
    train_z = (train - mu) / sd
    synth_z = (synth - mu) / sd
    val_z = (val - mu) / sd if val is not None else None

    if args.metric == "dtw":
        gen_dist = nn_dtw(synth_z, train_z)
        # argmin pairs for the top-k plot always use the cheap euclidean NN
        _, nn_idx = nn_rmse_argmin(synth_z, train_z)
        val_dist = nn_dtw(val_z, train_z) if val_z is not None else None
        tt_dist = nn_dtw(train_z, train_z, exclude_self=True) \
            if train.shape[0] > 1 else None
    else:
        gen_dist, nn_idx = nn_rmse_argmin(synth_z, train_z)
        val_dist = nn_rmse(val_z, train_z) if val_z is not None else None
        tt_dist = nn_rmse(train_z, train_z, exclude_self=True) \
            if train.shape[0] > 1 else None

    dists = {"gen->train": gen_dist}
    if val_dist is not None:
        dists["val->train"] = val_dist
    if tt_dist is not None:
        dists["train->train"] = tt_dist

    report = {
        "files": {"train": args.train_npz, "synth": args.synth_npz,
                  "val": args.val_npz},
        "metric": args.metric,
        "distance": "per-timepoint RMSE on per-feature train-standardized windows",
        "shapes": {"train": list(train.shape), "synth": list(synth.shape),
                   "val": list(val.shape) if val is not None else None},
        "distributions": {name.replace("->", "_to_"): distance_stats(d)
                          for name, d in dists.items()},
        "caveats": (
            "train_to_train is a lower bound only: with stride-1 sliding "
            "windows neighbouring train windows overlap almost entirely, so "
            "that distribution sits near zero by construction. Judge novelty "
            "against val_to_train."),
    }
    if val_dist is not None:
        val_med = float(np.median(val_dist))
        val_p05 = float(np.percentile(val_dist, 5))
        copy_ratio = float(np.median(gen_dist)) / max(val_med, 1e-12)
        frac_below = float((gen_dist < val_p05).mean())
        report["copy_ratio"] = copy_ratio
        report["frac_gen_below_val_p05"] = frac_below
        report["alarm"] = {
            f"copy_ratio_below_{COPY_RATIO_ALARM}": copy_ratio < COPY_RATIO_ALARM,
            f"frac_below_val_p05_above_{FRAC_BELOW_P05_ALARM}":
                frac_below > FRAC_BELOW_P05_ALARM,
        }
        report["interpretation"] = (
            "copy_ratio ~ 1 means generated windows sit as far from the train "
            "set as genuinely novel val windows do; << 1 means they are much "
            "closer, i.e. the generator reproduces training data. Soft "
            "thresholds -- inspect the ECDF and top-k plots before concluding.")
    else:
        report["copy_ratio"] = None
        report["interpretation"] = (
            "No --val-npz given: no novelty reference. gen_to_train can only "
            "be compared against the (near-zero) train_to_train lower bound.")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.synth_npz))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "memcheck.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    save_ecdf_plot(dists, os.path.join(out_dir, "memcheck_ecdf.pdf"))
    save_topk_plot(synth, train, gen_dist, nn_idx,
                   os.path.join(out_dir, "memcheck_topk.pdf"), args.top_k)

    print(f"[memcheck] train={train.shape} synth={synth.shape} "
          f"val={val.shape if val is not None else None}")
    for name, d in dists.items():
        print(f"[memcheck] {name:13s} median={np.median(d):.5f} "
              f"p05={np.percentile(d, 5):.5f}")
    if val_dist is not None:
        alarm = any(report["alarm"].values())
        print(f"[memcheck] copy_ratio={report['copy_ratio']:.3f} "
              f"frac_gen_below_val_p05={report['frac_gen_below_val_p05']:.3f} "
              f"=> {'MEMORIZATION ALARM' if alarm else 'ok'}")
    print(f"[memcheck] wrote {json_path}, memcheck_ecdf.pdf, memcheck_topk.pdf")


if __name__ == "__main__":
    main()
