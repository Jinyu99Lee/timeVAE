#!/usr/bin/env python3
"""Step 2 — turn the manifest into one ``hpo_grid_search.py`` run script per
ILI subset. Each script mirrors the weather run but uses the subset's
``valid_perc`` and a single per-dataset ``reconstruction_wt = 15/D``; the
lr/batch/latent grid (from ili_config) is what HPO selects over.

    python generate_run_scripts.py                      # all manifest rows
    python generate_run_scripts.py --max-epochs 2 \
        --subset ili_eng_2016_2017_T42_p14_tau005       # smoke variant
"""
from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

import ili_config as cfg


def fmt_list(values) -> str:
    return " ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values)


def render_script(row: dict, args: argparse.Namespace, output_dir: Path | None) -> str:
    subset_id = row["subset_id"]
    log_path = cfg.RUN_LOG_DIR / f"{subset_id}.log"
    cli = [
        ('"$PYTHON"', str(cfg.HPO_SCRIPT)),
        ("--train-npz", row["train_path"]),
        ("--val-npz", row["val_path"]),
        ("--vae-type", cfg.VAE_TYPE),
        ("--latent-dim", fmt_list(args.latent_dim)),
        ("--reconstruction-wt", f"{row['wt_ili']:g}"),
        ("--learning-rate", fmt_list(args.learning_rate)),
        ("--batch-size", fmt_list(args.batch_size)),
        ("--max-epochs", str(args.max_epochs)),
        ("--loss-mode", cfg.LOSS_MODE),
        ("--monitor-metric", cfg.MONITOR_METRIC),
        ("--monitor-kl-latent-ref", str(args.monitor_kl_latent_ref)),
        ("--histogram-distance-backend", cfg.HISTOGRAM_BACKEND),
        ("--disable-train-histogram-distance", ""),
        ("--disable-val-histogram-distance", ""),
        ("--seed", str(cfg.SEED)),
        ("--early-stopping-start-epoch", str(args.early_stopping_start_epoch)),
        ("--early-stopping-patience", str(args.early_stopping_patience)),
        ("--early-stopping-min-delta", f"{args.early_stopping_min_delta:g}"),
        ("--gpu-slots", args.gpu_slots),
        ("--experiment-group", args.experiment_group),
        ("--experiment-name", subset_id),
    ]
    if args.skip_completed and output_dir is not None:
        # Resume into the existing HPO dir: hpo_grid_search.py skips the runs
        # already completed there and reruns only the failed/missing ones, then
        # rebuilds that dir's results.* and best_run.json in place. (--output-dir
        # overrides the timestamped path; group/name above are still recorded in
        # search_config.json.)
        cli += [
            ("--output-dir", str(output_dir)),
            ("--skip-completed", ""),
        ]
    lines = ["  " + (f"{flag} {val}" if val else flag).rstrip() for flag, val in cli]
    body = " \\\n".join(lines)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'PYTHON="${{PYTHON:-{cfg.PYTHON_BIN}}}"\n'
        f'mkdir -p "{cfg.RUN_LOG_DIR}"\n'
        f"# {subset_id}: D={row['D']} T={row['T']} "
        f"N={row['N_train']}+{row['N_val']} wt={row['wt_ili']}\n"
        f"{body} \\\n"
        f'  > "{log_path}" 2>&1\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=cfg.MANIFEST_PATH)
    parser.add_argument("--subset", nargs="+", default=None, help="subset_ids to keep")
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="subset_ids to drop (applied after --subset), e.g. the highD subsets.",
    )
    parser.add_argument(
        "--skip-completed", action="store_true",
        help=(
            "Resume mode: point each script at the subset's latest existing HPO "
            "dir and pass --skip-completed so hpo_grid_search.py reruns only the "
            "failed/missing runs and rebuilds that dir's summaries in place. "
            "Pass a --learning-rate grid matching the existing dirs (the grid "
            "guard refuses to resume on a mismatch)."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=cfg.MAX_EPOCHS)
    parser.add_argument("--gpu-slots", default=cfg.GPU_SLOTS)
    parser.add_argument("--experiment-group", default=cfg.EXPERIMENT_GROUP)
    # HPO grid overrides (default to ili_config). Applied to every subset; wt
    # stays per-dataset (15/D) and is never gridded.
    parser.add_argument(
        "--latent-dim", type=int, nargs="+", default=cfg.LATENT_DIMS,
        help=f"latent_dim candidates (default {cfg.LATENT_DIMS}).",
    )
    parser.add_argument(
        "--learning-rate", type=float, nargs="+", default=cfg.LEARNING_RATES,
        help=f"learning_rate candidates (default {cfg.LEARNING_RATES}).",
    )
    parser.add_argument(
        "--batch-size", type=int, nargs="+", default=cfg.BATCH_SIZES,
        help=f"batch_size candidates (default {cfg.BATCH_SIZES}).",
    )
    parser.add_argument(
        "--monitor-kl-latent-ref", type=int, default=cfg.MONITOR_KL_LATENT_REF,
        help=f"KL latent-norm reference (default {cfg.MONITOR_KL_LATENT_REF}).",
    )
    parser.add_argument(
        "--early-stopping-start-epoch", type=int,
        default=cfg.EARLY_STOPPING_START_EPOCH,
        help=f"default {cfg.EARLY_STOPPING_START_EPOCH}.",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int,
        default=cfg.EARLY_STOPPING_PATIENCE,
        help=f"default {cfg.EARLY_STOPPING_PATIENCE}.",
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float,
        default=cfg.EARLY_STOPPING_MIN_DELTA,
        help=f"default {cfg.EARLY_STOPPING_MIN_DELTA:g}.",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"manifest not found: {args.manifest}. Run prepare_ili_datasets.py first."
        )
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.subset:
        rows = [r for r in rows if r["subset_id"] in set(args.subset)]
    if args.exclude:
        rows = [r for r in rows if r["subset_id"] not in set(args.exclude)]
    if not rows:
        raise SystemExit("No manifest rows matched the filters.")

    cfg.RUN_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    index_lines, missing = [], []
    for row in rows:
        subset_id = row["subset_id"]
        output_dir = None
        if args.skip_completed:
            output_dir = cfg.find_latest_hpo_dir(args.experiment_group, subset_id)
            if output_dir is None:
                # No existing dir to resume into; fall back to a fresh run.
                missing.append(subset_id)
        script_path = cfg.RUN_SCRIPTS_DIR / f"run_{subset_id}.sh"
        script_path.write_text(render_script(row, args, output_dir))
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
        index_lines.append(str(script_path))
        resume = f"  resume -> {output_dir.name}" if output_dir is not None else ""
        print(f"  wrote {script_path.name}  (D={row['D']}, wt={row['wt_ili']}){resume}")

    cfg.RUN_INDEX.write_text("\n".join(index_lines) + "\n")
    print(f"\n{len(rows)} run scripts in {cfg.RUN_SCRIPTS_DIR}\nindex: {cfg.RUN_INDEX}")
    if args.skip_completed and missing:
        print(
            f"No existing HPO dir for {len(missing)} subset(s); generated a fresh "
            f"run instead: {', '.join(missing)}"
        )


if __name__ == "__main__":
    main()
