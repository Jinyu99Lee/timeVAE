#!/usr/bin/env python3
"""Step 1 — discover the pre-split ILI source datasets and write a manifest.

ILI ships one ``*_train.npz`` + ``*_val.npz`` per subset (Diffusion-TS reads them
directly via NPZDataset(train_path, val_path)). We mirror that: NO concatenation,
NO copying — we just read each pair's shape, compute ``wt_ili = 15/D``, and record
the absolute train/val paths in ``ili_manifest.jsonl`` for the generators.

Usage examples::

    python prepare_ili_datasets.py                 # all eng subsets
    python prepare_ili_datasets.py --tau tau005 tau05 --verify
    python prepare_ili_datasets.py --d-min 1000    # only the largest-D subsets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import ili_config as cfg

# Sonnet feature-selection meta (for the optional D == count+1 cross-check).
FEATURE_SELECTION_ROOT = (
    cfg.PROJECTS_ROOT / "Sonnet" / "datasets" / "ILI" / "feature_selection"
)


def find_pairs(region: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    if not cfg.ILI_SOURCE_ROOT.exists():
        raise FileNotFoundError(f"ILI source root not found: {cfg.ILI_SOURCE_ROOT}")
    for train_path in sorted(cfg.ILI_SOURCE_ROOT.glob("*/*/*_train.npz")):
        name = train_path.name[: -len("_train.npz")]
        if region and f"_{region}_" not in f"_{name}_":
            continue
        val_path = train_path.with_name(f"{name}_val.npz")
        if not val_path.exists():
            print(f"  ! skip {name}: missing val npz")
            continue
        pairs.append(
            {
                "name": name,
                "tau": train_path.parent.name,
                "period_dir": train_path.parent.parent.name,
                "train_path": train_path,
                "val_path": val_path,
            }
        )
    return pairs


def passes_filters(pair: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.tau and pair["tau"] not in args.tau:
        return False
    if args.period and pair["period_dir"] not in args.period:
        return False
    if args.subset and pair["name"] not in args.subset:
        return False
    if args.horizon and not (set(args.horizon) & set(pair["name"].split("_"))):
        return False
    return True


def _shape(npz_path: Path) -> tuple[int, int, int]:
    arr = np.load(npz_path, allow_pickle=True)["data"]
    return int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])


def _feature_selection_count(period_dir: str, tau: str, region: str) -> int | None:
    """Look up the selected-feature count from Sonnet meta.json (D should be +1)."""
    # period_dir like "2016_17test_tau"; meta uses "2016_2017".
    digits = period_dir.split("test")[0]  # "2016_17"
    parts = digits.split("_")
    if len(parts) != 2:
        return None
    season = f"{parts[0]}_20{parts[1]}" if len(parts[1]) == 2 else f"{parts[0]}_{parts[1]}"
    meta = FEATURE_SELECTION_ROOT / region / season / "meta.json"
    if not meta.exists():
        return None
    counts = json.loads(meta.read_text()).get("tau_feature_counts", {})
    key = tau.replace("tau", "")
    key = f"0.{key[1:]}" if key.startswith("0") and len(key) > 1 else f"0.{key}"
    val = counts.get(key)
    return int(val) if val is not None else None


def build_row(pair: dict[str, Any], verify: bool, region: str) -> dict[str, Any]:
    n_train, t, d = _shape(pair["train_path"])
    n_val, t_v, d_v = _shape(pair["val_path"])
    if (t, d) != (t_v, d_v):
        raise ValueError(f"{pair['name']}: train/val T,D mismatch ({t},{d})!=({t_v},{d_v})")

    if verify:
        count = _feature_selection_count(pair["period_dir"], pair["tau"], region)
        if count is not None and count + 1 != d:
            raise AssertionError(
                f"{pair['name']}: D={d} but feature_selection count+1={count + 1}"
            )

    return {
        "subset_id": cfg.slug(f"{pair['name']}_{pair['tau']}"),
        "name": pair["name"],
        "tau": pair["tau"],
        "period_dir": pair["period_dir"],
        "train_path": str(pair["train_path"].resolve()),
        "val_path": str(pair["val_path"].resolve()),
        "D": d,
        "T": t,
        "N_train": n_train,
        "N_val": n_val,
        "wt_ili": cfg.wt_for_dim(d),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=cfg.REGION)
    parser.add_argument("--tau", nargs="+", default=None)
    parser.add_argument("--period", nargs="+", default=None)
    parser.add_argument("--horizon", nargs="+", default=None, help="tokens like T42 p14")
    parser.add_argument("--subset", nargs="+", default=None)
    parser.add_argument("--d-min", type=int, default=None, dest="d_min")
    parser.add_argument("--d-max", type=int, default=None, dest="d_max")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verify", action="store_true", help="cross-check D == count+1")
    parser.add_argument("--manifest", type=Path, default=cfg.MANIFEST_PATH)
    args = parser.parse_args()

    pairs = [p for p in find_pairs(args.region) if passes_filters(p, args)]
    print(f"Found {len(pairs)} candidate subsets under {cfg.ILI_SOURCE_ROOT}")

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        row = build_row(pair, args.verify, args.region)
        if args.d_min is not None and row["D"] < args.d_min:
            continue
        if args.d_max is not None and row["D"] > args.d_max:
            continue
        rows.append(row)
        if args.limit is not None and len(rows) >= args.limit:
            break

    rows.sort(key=lambda r: (r["D"], r["subset_id"]))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as mf:
        for row in rows:
            mf.write(json.dumps(row) + "\n")
            print(
                f"  {row['subset_id']:<42} D={row['D']:<5} T={row['T']:<3} "
                f"N={row['N_train']}+{row['N_val']:<4} wt={row['wt_ili']}"
            )

    print(f"\nManifest: {args.manifest}  ({len(rows)} rows; verify={args.verify})")


if __name__ == "__main__":
    main()
