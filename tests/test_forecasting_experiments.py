from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import forecasting_experiments as fx  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _save_source_split(
    path: Path,
    windows: np.ndarray,
    indices: np.ndarray,
    metadata: dict,
    split: str,
) -> None:
    split_meta = dict(metadata)
    split_meta["split"] = split
    split_meta["split_samples"] = len(indices)
    np.savez_compressed(
        path,
        data=np.asarray(windows[indices], dtype=np.float32),
        sample_indices=np.asarray(indices, dtype=np.int64),
        window_start_indices=np.asarray(indices, dtype=np.int64),
        feature_cols=np.asarray(metadata["feature_cols"]),
        seq_len=np.asarray(metadata["seq_len"], dtype=np.int64),
        lookback=np.asarray(metadata["lookback"], dtype=np.int64),
        pred_len=np.asarray(metadata["pred_len"], dtype=np.int64),
        target_delay=np.asarray(metadata["target_delay"], dtype=np.int64),
        layout=np.asarray(metadata["layout"]),
        stride=np.asarray(metadata["stride"], dtype=np.int64),
        split=np.asarray(split),
        meta=np.asarray(json.dumps(split_meta, sort_keys=True)),
    )


def _fixture(root: Path) -> tuple[fx.ForecastExperiment, dict, dict, Path]:
    source = np.arange(30, dtype=np.float32).reshape(10, 3)
    seq_len = 4
    windows = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(source, seq_len, axis=0),
        -1,
        1,
    )
    num_windows = len(windows)
    val_start = int(num_windows * 0.70)
    val_end = int(num_windows * 0.85)
    train_indices = np.concatenate(
        (
            np.arange(0, val_start, dtype=np.int64),
            np.arange(val_end, num_windows, dtype=np.int64),
        )
    )
    val_indices = np.arange(val_start, val_end, dtype=np.int64)
    source_csv = (root / "source.csv").resolve()
    source_csv.write_text("fixture\n")
    metadata = {
        "dataset": "Energy",
        "source_csv": str(source_csv),
        "source_row_range": [0, len(source)],
        "source_first_timestamp": "2020-01-01T00:00:00",
        "source_last_timestamp": "2020-01-01T09:00:00",
        "source_frequency": "1h",
        "source_train_sha256": fx.sha256_float32_c(source),
        "feature_cols": ["feature_a", "feature_b", "target"],
        "target_column": "target",
        "target_index": 2,
        "dtype": "float32",
        "raw_physical_scale": True,
        "seq_len": seq_len,
        "lookback": 2,
        "pred_len": 2,
        "target_delay": 0,
        "layout": "aligned",
        "stride": 1,
        "N": num_windows,
        "total_samples": num_windows,
        "val_sample_start_ratio": 0.70,
        "val_sample_end_ratio": 0.85,
        "val_sample_start": val_start,
        "val_sample_end_exclusive": val_end,
        "train_sample_ranges": [[0, val_start], [val_end, num_windows]],
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "sample_index_semantics": "ordinal_in_full_sliding_window_set",
        "window_start_row_formula": "sample_index * stride",
    }
    train_npz = root / "toy_train.npz"
    val_npz = root / "toy_val.npz"
    source_meta = root / "toy_meta.json"
    _save_source_split(train_npz, windows, train_indices, metadata, "train")
    _save_source_split(val_npz, windows, val_indices, metadata, "val")
    _write_json(source_meta, metadata)

    hpo_dir = root / "hpo"
    run_dir = hpo_dir / "runs" / "run_00000"
    model_dir = run_dir / "best_model"
    model_dir.mkdir(parents=True)
    config = {
        "run_id": "run_00000",
        "dataset": "toy",
        "train_npz": str(train_npz.resolve()),
        "val_npz": str(val_npz.resolve()),
        "vae_type": "timeVAE",
        "valid_perc": 0.1,
        "split_method": "tail_holdout",
        "seed": 42,
        "max_epochs": 800,
        "early_stopping_start_epoch": 0,
        "early_stopping_min_delta": 0.0001,
        "early_stopping_patience": 20,
        "monitor_metric": "val_total_loss",
        "histogram_distance_backend": "numpy",
        "compute_train_histogram_distance": False,
        "compute_val_histogram_distance": False,
        "reduce_lr_on_plateau": False,
        "loss_mode": "legacy",
        "generate_after_train": False,
        "hyperparameters": {
            "latent_dim": 8,
            "hidden_layer_sizes": [50, 100, 200],
            "reconstruction_wt": 3.0,
            "learning_rate": 0.001,
            "batch_size": 16,
            "kl_anneal_epochs": 50,
            "free_bits": 0.1,
            "use_residual_conn": True,
            "trend_poly": 0,
            "custom_seas": None,
            "loss_mode": "legacy",
            "histogram_distance_backend": "numpy",
            "compute_train_histogram_distance": False,
            "compute_val_histogram_distance": False,
            "monitor_kl_latent_ref": 8,
        },
    }
    _write_json(run_dir / "config.json", config)
    best_run = {
        "status": "completed",
        "error": None,
        "run_id": "run_00000",
        "run_dir": str(run_dir.resolve()),
        "dataset": "toy",
        "train_npz": str(train_npz.resolve()),
        "val_npz": str(val_npz.resolve()),
        "monitor_kl_latent_ref": 8,
        "vae_type": "timeVAE",
        "latent_dim": 8,
        "reconstruction_wt": 3.0,
        "learning_rate": 0.001,
        "batch_size": 16,
        "free_bits": 0.1,
        "kl_anneal_epochs": 50,
        "loss_mode": "legacy",
        "valid_perc": 0.1,
        "split_method": "tail_holdout",
        "early_stopping_start_epoch": 0,
        "early_stopping_min_delta": 0.0001,
        "early_stopping_patience": 20,
        "monitor_metric": "val_total_loss",
        "histogram_distance_backend": "numpy",
        "compute_train_histogram_distance": False,
        "compute_val_histogram_distance": False,
        "reduce_lr_on_plateau": False,
        "best_monitor_value": 1.25,
        "best_epoch": 17,
    }
    _write_json(hpo_dir / "best_run.json", best_run)
    experiment = fx.ForecastExperiment(
        experiment_id="toy",
        dataset="energy",
        source_dataset="Energy",
        fold="default",
        lookback=2,
        pred_len=2,
        seq_len=4,
        target_column="target",
        train_npz=train_npz.resolve(),
        val_npz=val_npz.resolve(),
        source_meta=source_meta.resolve(),
        expected_num_windows=num_windows,
        hpo_output_dir=hpo_dir.resolve(),
        rerun_output_dir=(root / "rerun").resolve(),
        sonnet_dataset="exp_data_config/energy",
        sonnet_exp="energy",
        sonnet_seq_length=2,
        sonnet_batch_sizes=(64,),
        sonnet_hpo_profile="energy_raw",
    )
    return experiment, config, best_run, run_dir


class ForecastManifestTests(unittest.TestCase):
    def test_declared_matrix_and_hpo_counts(self) -> None:
        experiments = fx.load_manifest()
        self.assertEqual(len(experiments), 14)
        etth1 = [item for item in experiments if item.dataset == "etth1"]
        energy = [item for item in experiments if item.dataset == "energy"]
        elec = [item for item in experiments if item.dataset.startswith("electricity_")]
        self.assertEqual([item.pred_len for item in etth1], [96, 192, 336, 720])
        self.assertEqual([item.pred_len for item in energy], [24, 48, 72, 168])
        self.assertEqual(len(elec), 6)
        self.assertEqual({item.fold for item in elec}, {"2020", "2021"})
        self.assertTrue(all(item.sonnet_batch_sizes == (16, 32, 64) for item in etth1))
        self.assertTrue(all(item.sonnet_batch_sizes == (64,) for item in energy + elec))
        self.assertEqual(len(experiments) * len(fx.LEARNING_RATES), 56)
        self.assertEqual(
            len(fx.SONNET_LEARNING_RATES)
            * 3
            * len(fx.SONNET_ATOMS)
            * len(fx.SONNET_ALPHAS),
            216,
        )
        self.assertEqual(
            len(fx.SONNET_LEARNING_RATES)
            * len(fx.SONNET_ATOMS)
            * len(fx.SONNET_ALPHAS),
            72,
        )

    def test_commands_freeze_generator_and_downstream_grids(self) -> None:
        experiments = fx.load_manifest()
        hpo = fx.build_hpo_command(experiments[0], python_executable="python")
        expected_tokens = {
            "--vae-type": "timeVAE",
            "--latent-dim": "8",
            "--reconstruction-wt": "3",
            "--batch-size": "16",
            "--max-epochs": "800",
            "--loss-mode": "legacy",
            "--monitor-metric": "val_total_loss",
            "--monitor-kl-latent-ref": "8",
            "--early-stopping-patience": "20",
            "--free-bits": "0.1",
            "--kl-anneal-epochs": "50",
            "--seed": "42",
        }
        for flag, value in expected_tokens.items():
            self.assertEqual(hpo[hpo.index(flag) + 1], value)
        lr_start = hpo.index("--learning-rate") + 1
        self.assertEqual(
            hpo[lr_start : lr_start + 4],
            [str(value) for value in fx.LEARNING_RATES],
        )
        self.assertIn("--disable-train-histogram-distance", hpo)
        self.assertIn("--disable-val-histogram-distance", hpo)
        rerun = fx.build_rerun_command(experiments[0], fx.DEFAULT_MANIFEST)
        self.assertEqual(rerun[rerun.index("--num-samples") + 1], "all")
        self.assertEqual(rerun[rerun.index("--compare-split") + 1], "all")
        etth_sonnet = fx.build_sonnet_command(experiments[0], REPO_ROOT.parent / "Sonnet")
        self.assertIn("exp.batch_size=16,32,64", etth_sonnet)
        energy_sonnet = fx.build_sonnet_command(experiments[4], REPO_ROOT.parent / "Sonnet")
        self.assertIn("exp.batch_size=64", energy_sonnet)


class ForecastContractTests(unittest.TestCase):
    def test_source_pair_reconstructs_and_hashes_canonical_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, _, _, _ = _fixture(Path(directory))
            contract = fx.validate_source_pair(experiment)
            self.assertEqual(contract["num_real_windows"], 7)
            self.assertEqual(contract["feature_cols"][-1], "target")
            self.assertEqual(
                contract["source_train_sha256"],
                contract["source_meta"]["source_train_sha256"],
            )

    def test_source_pair_rejects_bad_window_starts_and_window_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, _, _, _ = _fixture(Path(directory))
            with np.load(experiment.train_npz, allow_pickle=False) as payload:
                arrays = {key: np.asarray(payload[key]) for key in payload.files}
            arrays["window_start_indices"] = arrays["window_start_indices"].copy()
            arrays["window_start_indices"][1] += 1
            np.savez_compressed(experiment.train_npz, **arrays)
            with self.assertRaisesRegex(ValueError, "window_start_indices"):
                fx.validate_source_pair(experiment)

        with tempfile.TemporaryDirectory() as directory:
            experiment, _, _, _ = _fixture(Path(directory))
            with np.load(experiment.train_npz, allow_pickle=False) as payload:
                arrays = {key: np.asarray(payload[key]) for key in payload.files}
            arrays["data"] = arrays["data"].copy()
            arrays["data"][1, 1, 0] += np.float32(0.25)
            np.savez_compressed(experiment.train_npz, **arrays)
            with self.assertRaisesRegex(ValueError, "windows do not exactly match"):
                fx.validate_source_pair(experiment)

    def test_complete_generator_protocol_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, config, best_run, run_dir = _fixture(Path(directory))
            protocol = fx.validate_forecasting_hpo_config(
                config, best_run, experiment, run_dir
            )
            self.assertEqual(protocol["learning_rates"], list(fx.LEARNING_RATES))
            self.assertEqual(protocol["selected_learning_rate"], 0.001)
            self.assertEqual(protocol["reconstruction_wt"], 3.0)
            self.assertEqual(protocol["loss_mode"], "legacy")
            self.assertEqual(protocol["monitor_metric"], "val_total_loss")

            changed = copy.deepcopy(config)
            changed["compute_val_histogram_distance"] = True
            with self.assertRaisesRegex(ValueError, "compute_val_histogram_distance"):
                fx.validate_forecasting_hpo_config(
                    changed, best_run, experiment, run_dir
                )

    def test_full_n_generated_npz_round_trip_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, config, best_run, run_dir = _fixture(Path(directory))
            contract = fx.validate_source_pair(experiment)
            protocol = fx.validate_forecasting_hpo_config(
                config, best_run, experiment, run_dir
            )
            generated = np.linspace(
                -3.5,
                8.5,
                experiment.expected_num_windows * experiment.seq_len * 3,
                dtype=np.float32,
            ).reshape(experiment.expected_num_windows, experiment.seq_len, 3)
            metadata = fx.build_generated_metadata(
                experiment=experiment,
                source_contract=contract,
                best_run=best_run,
                best_run_path=experiment.best_run,
                run_dir=run_dir,
                model_dir=run_dir / "best_model",
                seed=42,
                generated=generated,
                generator_protocol=protocol,
            )
            fx.save_generated_npz(experiment.generated_npz, generated, metadata)
            validated = fx.validate_generated_npz(experiment)
            self.assertTrue(validated["one_to_one_with_source_windows"])
            self.assertTrue(validated["raw_physical_scale"])
            self.assertEqual(validated["N"], experiment.expected_num_windows)
            with np.load(experiment.generated_npz, allow_pickle=False) as payload:
                self.assertEqual(payload["data"].dtype, np.float32)
                np.testing.assert_array_equal(
                    payload["sample_indices"],
                    np.arange(experiment.expected_num_windows, dtype=np.int64),
                )
                np.testing.assert_array_equal(
                    payload["window_start_indices"],
                    np.arange(experiment.expected_num_windows, dtype=np.int64),
                )

    def test_generated_npz_rejects_tampered_data_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, config, best_run, run_dir = _fixture(Path(directory))
            contract = fx.validate_source_pair(experiment)
            protocol = fx.validate_forecasting_hpo_config(
                config, best_run, experiment, run_dir
            )
            generated = np.zeros(
                (experiment.expected_num_windows, experiment.seq_len, 3),
                dtype=np.float32,
            )
            metadata = fx.build_generated_metadata(
                experiment,
                contract,
                best_run,
                experiment.best_run,
                run_dir,
                run_dir / "best_model",
                42,
                generated,
                protocol,
            )
            fx.save_generated_npz(experiment.generated_npz, generated, metadata)
            with np.load(experiment.generated_npz, allow_pickle=False) as payload:
                arrays = {key: np.asarray(payload[key]) for key in payload.files}
            arrays["data"] = arrays["data"].copy()
            arrays["data"][0, 0, 0] = np.float32(99)
            np.savez_compressed(experiment.generated_npz, **arrays)
            with self.assertRaisesRegex(ValueError, "generated_data_sha256"):
                fx.validate_generated_npz(experiment)


if __name__ == "__main__":
    unittest.main()
