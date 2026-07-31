#!/usr/bin/env python3
"""Step 4 — for each subset that finished HPO, locate its best_run.json and
render a ``rerun_best_hpo.py`` script that generates prior samples and the
distribution / t-SNE evaluation against the held-out ILI validation windows.

    python generate_rerun_scripts.py
    python generate_rerun_scripts.py --subset ili_eng_2016_2017_T42_p14_tau005 --no-tsne
"""
from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path

import ili_config as cfg


def find_best_run(group: str, subset_id: str) -> Path | None:
    group_dir = cfg.REPO_ROOT / "outputs" / "hpo" / cfg.slug(group)
    if not group_dir.exists():
        return None
    # hpo output dirs are "<timestamp>_<experiment-name>"; the timestamp prefix
    # sorts chronologically, so we pick the LATEST dir that actually has a
    # best_run.json. This is the "latest" rerun semantics: in --skip-completed
    # mode each subset resumes into one dir (rebuilt best_run.json, no duplicate
    # runs), and subsets whose latest attempt produced no completed run (no
    # best_run.json) are skipped here automatically.
    candidates = sorted(group_dir.glob(f"*_{subset_id}/best_run.json"))
    return candidates[-1] if candidates else None


def render_script(subset_id: str, best_run: Path, args: argparse.Namespace) -> str:
    log_path = cfg.RERUN_LOG_DIR / f"{subset_id}.log"
    cli = [
        ('"$PYTHON"', str(cfg.RERUN_SCRIPT)),
        ("--best-run", str(best_run)),
        ("--num-samples", cfg.RERUN_NUM_SAMPLES),
        ("--compare-split", cfg.RERUN_COMPARE_SPLIT),
    ]
    if args.no_tsne:
        cli.append(("--no-tsne", ""))
    lines = ["  " + (f"{flag} {val}" if val else flag).rstrip() for flag, val in cli]
    body = " \\\n".join(lines)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'PYTHON="${{PYTHON:-{cfg.PYTHON_BIN}}}"\n'
        f'mkdir -p "{cfg.RERUN_LOG_DIR}"\n'
        f"# rerun {subset_id}\n"
        f"{body} \\\n"
        f'  > "{log_path}" 2>&1\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=cfg.MANIFEST_PATH)
    parser.add_argument("--subset", nargs="+", default=None, help="subset_ids to keep")
    parser.add_argument("--experiment-group", default=cfg.EXPERIMENT_GROUP)
    parser.add_argument("--no-tsne", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    if args.subset:
        rows = [r for r in rows if r["subset_id"] in set(args.subset)]

    cfg.RERUN_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    index_lines, missing = [], []
    for row in rows:
        subset_id = row["subset_id"]
        best_run = find_best_run(args.experiment_group, subset_id)
        if best_run is None:
            missing.append(subset_id)
            continue
        script_path = cfg.RERUN_SCRIPTS_DIR / f"rerun_{subset_id}.sh"
        script_path.write_text(render_script(subset_id, best_run, args))
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
        index_lines.append(str(script_path))
        print(f"  wrote {script_path.name}  -> {best_run}")

    if index_lines:
        cfg.RERUN_INDEX.write_text("\n".join(index_lines) + "\n")
    print(f"\n{len(index_lines)} rerun scripts in {cfg.RERUN_SCRIPTS_DIR}")
    if missing:
        print(f"No best_run.json yet for {len(missing)} subset(s): {', '.join(missing)}")


if __name__ == "__main__":
    main()
