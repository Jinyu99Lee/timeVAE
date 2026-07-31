#!/usr/bin/env python3
"""Shared configuration for the ILI legacy-TimeVAE probe framework.

Probe ① : train legacy TimeVAE on each ILI subset with a per-dataset
``reconstruction_wt = wt_weather * d_weather / D = 15 / D`` so that the summed
reconstruction loss (``~ T*D``) is pulled back to roughly the same magnitude as
the KL loss (``~ latent``), reproducing the weather-proven ``recon:kl ~ 1:1``
balance. Nothing in the loss / training code is changed; the scripts only wrap
the existing ``hpo_grid_search.py`` and ``rerun_best_hpo.py`` entrypoints.

All generators import this module so run and rerun stay consistent.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# --- repository layout -------------------------------------------------------
# .../timeVAE/src/ILI/ili_config.py -> parents[2] == .../timeVAE
ILI_DIR = Path(__file__).resolve().parent
SRC_DIR = ILI_DIR.parent
REPO_ROOT = SRC_DIR.parent
PROJECTS_ROOT = REPO_ROOT.parent

# ILI source datasets live in the sibling Diffusion-TS project. ILI_DATASET
# selects the tree under Diffusion-TS/Data/datasets/ ("ili" by default; e.g.
# export ILI_DATASET=ili_delta). The manifest, script/log dirs and the
# experiment group all derive from it so variants never collide; the defaults
# below are byte-identical to the historical "ili" layout.
ILI_DATASET = os.environ.get("ILI_DATASET", "ili")
ILI_SOURCE_ROOT = PROJECTS_ROOT / "Diffusion-TS" / "Data" / "datasets" / ILI_DATASET
# "" for "ili", "_delta" for "ili_delta", "_foo" for any other variant name.
_SUFFIX = "" if ILI_DATASET == "ili" else "_" + ILI_DATASET.removeprefix("ili_")

# Framework working dirs.
MANIFEST_PATH = ILI_DIR / f"ili{_SUFFIX}_manifest.jsonl"
RUN_SCRIPTS_DIR = ILI_DIR / f"run_scripts{_SUFFIX}"
RERUN_SCRIPTS_DIR = ILI_DIR / f"rerun_scripts{_SUFFIX}"
RUN_LOG_DIR = ILI_DIR / "logs" / f"run{_SUFFIX}"
RERUN_LOG_DIR = ILI_DIR / "logs" / f"rerun{_SUFFIX}"
RUN_INDEX = RUN_SCRIPTS_DIR / "index.txt"
RERUN_INDEX = RERUN_SCRIPTS_DIR / "index.txt"

# --- rebalance anchor --------------------------------------------------------
WT_WEATHER = 3.0   # reconstruction_wt that works on weather
D_WEATHER = 5      # weather feature dimension


def wt_for_dim(feat_dim: int) -> float:
    """Per-dataset reconstruction_wt = wt_weather * d_weather / D."""
    if feat_dim <= 0:
        raise ValueError(f"feat_dim must be positive, got {feat_dim}.")
    return round(WT_WEATHER * D_WEATHER / float(feat_dim), 6)


# --- HPO grid (mirrors the weather grid; wt is NOT gridded, see plan) ---------
LATENT_DIMS = [8, 16]
# NOTE: the existing HPO dirs under outputs/hpo/ili_eng_oldloss_1overD/ were built
# with a 4-value LR grid (0.0001 0.0005 0.001 0.01 -> 24 runs). To reuse those run
# dirs with `generate_run_scripts.py --skip-completed`, pass the matching grid
# explicitly, e.g. `--learning-rate 0.0001 0.0005 0.001 0.01`; otherwise the grid
# guard in hpo_grid_search.py will refuse to resume (run_id indices would drift).
LEARNING_RATES = [1e-4, 1e-3, 1e-2]
BATCH_SIZES = [16, 32, 64]

# --- fixed training settings -------------------------------------------------
VAE_TYPE = "timeVAE"
LOSS_MODE = "legacy"
MONITOR_METRIC = "val_total_loss"
# Normalize the KL part of the monitored val_total_loss by ref/latent_dim so
# latent=8 and latent=16 compare fairly during HPO selection (selection-only).
MONITOR_KL_LATENT_REF = 8
HISTOGRAM_BACKEND = "tensorflow"
MAX_EPOCHS = 1000
EARLY_STOPPING_START_EPOCH = 0
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 1e-4
SEED = 42
EXPERIMENT_GROUP = f"ili{_SUFFIX}_eng_oldloss_1overD"
REGION = "eng"

# GPU concurrency map passed to hpo_grid_search.py (override per machine).
GPU_SLOTS = os.environ.get("ILI_GPU_SLOTS", "0:1")

# Python interpreter the generated scripts call. Defaults to the project env;
# override by exporting ILI_PYTHON before generating, or PYTHON at run time.
PYTHON_BIN = os.environ.get(
    "ILI_PYTHON", "/data/jinyuli/anaconda3/envs/timevae/bin/python"
)

# Entry-point scripts being wrapped.
HPO_SCRIPT = SRC_DIR / "hpo_grid_search.py"
RERUN_SCRIPT = SRC_DIR / "rerun_best_hpo.py"

# --- rerun settings ----------------------------------------------------------
RERUN_NUM_SAMPLES = "train"
RERUN_COMPARE_SPLIT = "valid"


def slug(value: str) -> str:
    """Filesystem/experiment-name safe slug (matches hpo_grid_search.slug)."""
    return re.sub(r"[^0-9A-Za-z_-]", "_", value).strip("_")


def find_latest_hpo_dir(group: str, subset_id: str) -> Path | None:
    """Latest existing HPO output dir for a subset, i.e.
    outputs/hpo/<group>/<timestamp>_<subset_id>/. The ``<timestamp>_`` prefix
    sorts chronologically, so the lexicographically last match is the newest.
    Returns None when no dir exists yet (subset never run)."""
    group_dir = REPO_ROOT / "outputs" / "hpo" / slug(group)
    if not group_dir.exists():
        return None
    candidates = sorted(p for p in group_dir.glob(f"*_{subset_id}") if p.is_dir())
    return candidates[-1] if candidates else None
