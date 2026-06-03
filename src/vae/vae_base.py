import os, warnings, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


def _freedman_diaconis_num_bins(num_values: int) -> int:
    if num_values <= 0:
        return 1
    return max(1, int(round(2.0 * np.power(num_values, 1.0 / 3.0), 0)))


def _histogram_distance_1d_np(
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


def _batch_histogram_distance_np(
    real_data: np.ndarray, synthetic_data: np.ndarray
) -> np.float32:
    real_data = np.asarray(real_data, dtype=float)
    synthetic_data = np.asarray(synthetic_data, dtype=float)
    if real_data.ndim != 3 or synthetic_data.ndim != 3:
        return np.float32(np.nan)
    if real_data.shape[1:] != synthetic_data.shape[1:]:
        return np.float32(np.nan)

    num_bins = _freedman_diaconis_num_bins(int(real_data.shape[0] * real_data.shape[1]))
    distances = []
    for feature_idx in range(real_data.shape[2]):
        real_values = real_data[:, :, feature_idx].reshape(-1)
        synthetic_values = synthetic_data[:, :, feature_idx].reshape(-1)
        distances.append(
            _histogram_distance_1d_np(real_values, synthetic_values, num_bins)
        )
    if not distances:
        return np.float32(np.nan)
    return np.float32(np.nanmean(np.asarray(distances, dtype=float)))


def _freedman_diaconis_num_bins_tf(num_values):
    num_values_float = tf.cast(tf.maximum(num_values, 1), tf.float32)
    num_bins = tf.cast(
        tf.round(2.0 * tf.pow(num_values_float, 1.0 / 3.0)), tf.int32
    )
    return tf.maximum(num_bins, 1)


def _histogram_counts_tf(values, min_val, max_val, num_bins, count_out_of_bounds):
    num_values = tf.shape(values)[0]
    num_features = tf.shape(values)[1]
    num_bins_float = tf.cast(num_bins, values.dtype)
    width = (max_val - min_val) / num_bins_float
    raw_bins = tf.floor((values - min_val) / width)
    bin_ids = tf.clip_by_value(tf.cast(raw_bins, tf.int32), 0, num_bins - 1)

    in_bounds = tf.logical_and(values >= min_val, values <= max_val)
    if count_out_of_bounds:
        weights = tf.cast(in_bounds, tf.float32)
    else:
        weights = tf.ones_like(values, dtype=tf.float32)

    feature_ids = tf.tile(tf.range(num_features)[tf.newaxis, :], [num_values, 1])
    segment_ids = tf.reshape(feature_ids * num_bins + bin_ids, [-1])
    counts = tf.math.unsorted_segment_sum(
        tf.reshape(weights, [-1]),
        segment_ids,
        num_segments=num_features * num_bins,
    )
    return tf.reshape(counts, [num_features, num_bins])


def _batch_histogram_distance_tf(real_data, synthetic_data):
    real_data = tf.cast(real_data, tf.float32)
    synthetic_data = tf.cast(synthetic_data, tf.float32)
    real_values = tf.reshape(real_data, [-1, tf.shape(real_data)[2]])
    synthetic_values = tf.reshape(synthetic_data, [-1, tf.shape(synthetic_data)[2]])

    num_values = tf.shape(real_values)[0]
    num_bins = _freedman_diaconis_num_bins_tf(num_values)
    min_val = tf.reduce_min(real_values, axis=0)
    max_val = tf.reduce_max(real_values, axis=0)
    constant_features = tf.abs(max_val - min_val) < 1e-10
    min_val = tf.where(constant_features, min_val - 1e-5, min_val)
    max_val = tf.where(constant_features, max_val + 1e-5, max_val)

    real_counts = _histogram_counts_tf(
        real_values, min_val, max_val, num_bins, count_out_of_bounds=False
    )
    synthetic_counts = _histogram_counts_tf(
        synthetic_values, min_val, max_val, num_bins, count_out_of_bounds=True
    )

    density_scale = tf.cast(num_bins, tf.float32) / tf.cast(num_values, tf.float32)
    real_density = real_counts * density_scale
    synthetic_density = synthetic_counts * density_scale
    abs_metric = tf.reduce_mean(tf.abs(real_density - synthetic_density), axis=1)

    out_of_bounds = tf.reduce_mean(
        tf.cast(
            tf.logical_or(synthetic_values < min_val, synthetic_values > max_val),
            tf.float32,
        ),
        axis=0,
    )
    return tf.reduce_mean((abs_metric + out_of_bounds) / 2.0)


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class KLAnnealingCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        kl_anneal_epochs = self.model.kl_anneal_epochs
        if kl_anneal_epochs > 0:
            kl_weight = min(1.0, epoch / kl_anneal_epochs)
        else:
            kl_weight = 1.0
        self.model.kl_weight.assign(kl_weight)


class RestoreBestWeights(Callback):
    def __init__(
        self, monitor="val_total_loss", min_delta=0.0, mode="min", start_epoch=0
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.start_epoch = start_epoch
        self.monitor_op = np.less if mode == "min" else np.greater
        self.best = np.inf if mode == "min" else -np.inf
        self.best_epoch = None
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        current = float(current)
        improved = (
            current + self.min_delta < self.best
            if self.monitor_op == np.less
            else current - self.min_delta > self.best
        )
        if improved:
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        self.model.best_monitor = self.monitor
        self.model.best_monitor_value = self.best
        self.model.best_epoch = self.best_epoch


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        super().on_epoch_end(epoch, logs)


class BaseVariationalAutoencoder(Model, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        reconstruction_wt=3.0,
        batch_size=16,
        learning_rate=0.001,
        kl_anneal_epochs=50,
        free_bits=0.1,
        loss_mode="current",
        histogram_distance_backend="numpy",
        compute_train_histogram_distance=True,
        **kwargs,
    ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        if loss_mode not in ("current", "legacy"):
            raise ValueError("loss_mode must be one of: current, legacy.")
        if histogram_distance_backend not in ("numpy", "tensorflow"):
            raise ValueError(
                "histogram_distance_backend must be one of: numpy, tensorflow."
            )
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_anneal_epochs = kl_anneal_epochs
        self.free_bits = free_bits
        self.loss_mode = loss_mode
        self.histogram_distance_backend = histogram_distance_backend
        self.compute_train_histogram_distance = compute_train_histogram_distance
        self.normalize_legacy_total_loss_metrics = False
        self.kl_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.raw_kl_loss_tracker = Mean(name="raw_kl_loss")
        self.reconstruction_component_loss_tracker = Mean(
            name="reconstruction_component_loss"
        )
        self.kl_component_loss_tracker = Mean(name="kl_component_loss")
        self.histogram_distance_tracker = Mean(name="histogram_distance")
        self.encoder = None
        self.decoder = None

    def fit_on_data(
        self,
        train_data,
        valid_data=None,
        max_epochs=1000,
        verbose=0,
        early_stopping_patience=50,
        early_stopping_min_delta=1e-4,
        early_stopping_start_epoch=0,
        monitor_metric=None,
        reduce_lr_on_plateau=False,
    ):
        if monitor_metric is None:
            monitor_metric = "val_total_loss" if valid_data is not None else "total_loss"
        if valid_data is None and monitor_metric.startswith("val_"):
            raise ValueError(
                f"monitor_metric={monitor_metric!r} requires validation data."
            )
        self.normalize_legacy_total_loss_metrics = (
            self.loss_mode == "legacy"
            and monitor_metric in ("total_loss", "val_total_loss")
        )
        best_weights = RestoreBestWeights(
            monitor=monitor_metric,
            min_delta=early_stopping_min_delta,
            mode="min",
            start_epoch=early_stopping_start_epoch,
        )
        early_stopping = DelayedEarlyStopping(
            start_epoch=early_stopping_start_epoch,
            monitor=monitor_metric,
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            mode="min",
        )
        callbacks = [KLAnnealingCallback(), best_weights, early_stopping]
        if reduce_lr_on_plateau:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=monitor_metric, factor=0.5, patience=30, mode="min"
                )
            )
        return self.fit(
            train_data,
            validation_data=valid_data,
            epochs=max_epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.raw_kl_loss_tracker,
            self.reconstruction_component_loss_tracker,
            self.kl_component_loss_tracker,
            self.histogram_distance_tracker,
        ]

    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def get_num_trainable_variables(self):
        trainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.trainable_weights])
        )
        nonTrainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights])
        )
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams

    def get_prior_samples(self, num_samples):
        Z = np.random.randn(num_samples, self.latent_dim)
        samples = self.decoder.predict(Z, verbose=0)
        return samples

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        return samples

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def _get_current_reconstruction_loss(self, X, X_recons):
        err = tf.math.squared_difference(X, X_recons)
        return tf.reduce_mean(err)

    def _get_legacy_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(axis):
            x_r = tf.reduce_mean(X, axis=axis)
            x_c_r = tf.reduce_mean(X_recons, axis=axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            return tf.reduce_sum(err)

        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)
        reconst_loss += get_reconst_loss_by_axis(axis=[2])
        return reconst_loss

    def _get_reconstruction_loss(self, X, X_recons):
        if self.loss_mode == "legacy":
            return self._get_legacy_reconstruction_loss(X, X_recons)
        return self._get_current_reconstruction_loss(X, X_recons)

    def _get_current_kl_loss(self, z_mean, z_log_var, apply_free_bits=True):
        kl_per_dim = 0.5 * (
            tf.square(z_mean) + tf.exp(z_log_var) - z_log_var - 1
        )
        if apply_free_bits and self.free_bits > 0.0:
            kl_per_dim = tf.maximum(kl_per_dim, self.free_bits)
        kl_per_sample = tf.reduce_sum(kl_per_dim, axis=1)
        return tf.reduce_mean(kl_per_sample)

    def _get_legacy_kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))

    def _get_kl_loss(self, z_mean, z_log_var, apply_free_bits=True):
        if self.loss_mode == "legacy":
            return self._get_legacy_kl_loss(z_mean, z_log_var)
        return self._get_current_kl_loss(
            z_mean, z_log_var, apply_free_bits=apply_free_bits
        )

    def _as_tensor(self, value):
        if isinstance(value, (list, tuple)):
            value = value[0]
        return value

    def _get_histogram_distance(self, X, X_recons):
        X = tf.stop_gradient(X)
        X_recons = tf.stop_gradient(X_recons)
        if self.histogram_distance_backend == "tensorflow":
            return _batch_histogram_distance_tf(X, X_recons)
        histogram_distance = tf.numpy_function(
            _batch_histogram_distance_np,
            [X, X_recons],
            tf.float32,
        )
        histogram_distance.set_shape(())
        return histogram_distance

    def _logged_total_loss(self):
        total_loss = self.total_loss_tracker.result()
        if self.normalize_legacy_total_loss_metrics:
            total_loss = total_loss / tf.cast(self.batch_size, total_loss.dtype)
        return total_loss

    def train_step(self, X):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)

            reconstruction = self._as_tensor(self.decoder(z))

            reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

            kl_loss = self._get_kl_loss(z_mean, z_log_var)
            raw_kl_loss = self._get_kl_loss(
                z_mean, z_log_var, apply_free_bits=False
            )

            if self.loss_mode == "legacy":
                total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
                reconstruction_component_loss = (
                    self.reconstruction_wt * reconstruction_loss
                )
                kl_component_loss = total_loss - reconstruction_component_loss
            else:
                reconstruction_component_loss = (
                    self.reconstruction_wt * reconstruction_loss
                )
                kl_component_loss = self.kl_weight * kl_loss
                total_loss = reconstruction_component_loss + kl_component_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        batch_size = tf.shape(X)[0]
        sample_weight = tf.cast(batch_size, tf.float32)
        self.total_loss_tracker.update_state(total_loss, sample_weight=sample_weight)
        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss, sample_weight=sample_weight
        )
        self.kl_loss_tracker.update_state(kl_loss, sample_weight=sample_weight)
        self.raw_kl_loss_tracker.update_state(
            raw_kl_loss, sample_weight=sample_weight
        )
        self.reconstruction_component_loss_tracker.update_state(
            reconstruction_component_loss, sample_weight=sample_weight
        )
        self.kl_component_loss_tracker.update_state(
            kl_component_loss, sample_weight=sample_weight
        )

        logged_total_loss = self._logged_total_loss()
        logs = {
            "loss": logged_total_loss,
            "total_loss": logged_total_loss,
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "raw_kl_loss": self.raw_kl_loss_tracker.result(),
            "reconstruction_component_loss": (
                self.reconstruction_component_loss_tracker.result()
            ),
            "kl_component_loss": self.kl_component_loss_tracker.result(),
            "kl_weight": self.kl_weight,
        }
        if self.compute_train_histogram_distance:
            histogram_distance = self._get_histogram_distance(X, reconstruction)
            self.histogram_distance_tracker.update_state(
                histogram_distance, sample_weight=sample_weight
            )
            logs["histogram_distance"] = self.histogram_distance_tracker.result()
        return logs

    def test_step(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self._as_tensor(self.decoder(z))
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

        kl_loss = self._get_kl_loss(z_mean, z_log_var)
        raw_kl_loss = self._get_kl_loss(z_mean, z_log_var, apply_free_bits=False)

        if self.loss_mode == "legacy":
            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
            reconstruction_component_loss = self.reconstruction_wt * reconstruction_loss
            kl_component_loss = total_loss - reconstruction_component_loss
        else:
            reconstruction_component_loss = reconstruction_loss
            kl_component_loss = kl_loss
            total_loss = reconstruction_loss + kl_loss

        batch_size = tf.shape(X)[0]
        sample_weight = tf.cast(batch_size, tf.float32)
        prior_z = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float32)
        prior_samples = self._as_tensor(self.decoder(prior_z))
        histogram_distance = self._get_histogram_distance(X, prior_samples)
        self.total_loss_tracker.update_state(total_loss, sample_weight=sample_weight)
        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss, sample_weight=sample_weight
        )
        self.kl_loss_tracker.update_state(kl_loss, sample_weight=sample_weight)
        self.raw_kl_loss_tracker.update_state(
            raw_kl_loss, sample_weight=sample_weight
        )
        self.reconstruction_component_loss_tracker.update_state(
            reconstruction_component_loss, sample_weight=sample_weight
        )
        self.kl_component_loss_tracker.update_state(
            kl_component_loss, sample_weight=sample_weight
        )
        self.histogram_distance_tracker.update_state(
            histogram_distance, sample_weight=sample_weight
        )

        logged_total_loss = self._logged_total_loss()
        return {
            "loss": logged_total_loss,
            "total_loss": logged_total_loss,
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "raw_kl_loss": self.raw_kl_loss_tracker.result(),
            "reconstruction_component_loss": (
                self.reconstruction_component_loss_tracker.result()
            ),
            "kl_component_loss": self.kl_component_loss_tracker.result(),
            "histogram_distance": self.histogram_distance_tracker.result(),
        }

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(
            encoder_wts, os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        joblib.dump(
            decoder_wts, os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

    def load_weights(self, model_dir):
        encoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        decoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "learning_rate": self.learning_rate,
            "kl_anneal_epochs": self.kl_anneal_epochs,
            "free_bits": self.free_bits,
            "loss_mode": self.loss_mode,
            "histogram_distance_backend": self.histogram_distance_backend,
            "compute_train_histogram_distance": self.compute_train_histogram_distance,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)


#####################################################################################################
#####################################################################################################


if __name__ == "__main__":
    pass
