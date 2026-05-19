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
        **kwargs,
    ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_anneal_epochs = kl_anneal_epochs
        self.free_bits = free_bits
        self.kl_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
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
    ):
        loss_to_monitor = "val_total_loss" if valid_data is not None else "total_loss"
        best_weights = RestoreBestWeights(
            monitor=loss_to_monitor,
            min_delta=early_stopping_min_delta,
            mode="min",
            start_epoch=early_stopping_start_epoch,
        )
        early_stopping = DelayedEarlyStopping(
            start_epoch=early_stopping_start_epoch,
            monitor=loss_to_monitor,
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            mode="min",
        )
        reduce_lr = ReduceLROnPlateau(
            monitor=loss_to_monitor, factor=0.5, patience=30, mode="min"
        )
        return self.fit(
            train_data,
            validation_data=valid_data,
            epochs=max_epochs,
            batch_size=self.batch_size,
            callbacks=[KLAnnealingCallback(), best_weights, early_stopping, reduce_lr],
            verbose=verbose,
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
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

    def _get_reconstruction_loss(self, X, X_recons):
        err = tf.math.squared_difference(X, X_recons)
        return tf.reduce_mean(err)

    def _get_kl_loss(self, z_mean, z_log_var, apply_free_bits=True):
        kl_per_dim = 0.5 * (
            tf.square(z_mean) + tf.exp(z_log_var) - z_log_var - 1
        )
        if apply_free_bits and self.free_bits > 0.0:
            kl_per_dim = tf.maximum(kl_per_dim, self.free_bits)
        kl_per_sample = tf.reduce_sum(kl_per_dim, axis=1)
        return tf.reduce_mean(kl_per_sample)

    def train_step(self, X):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)

            reconstruction = self.decoder(z)

            reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

            kl_loss = self._get_kl_loss(z_mean, z_log_var)

            total_loss = (
                self.reconstruction_wt * reconstruction_loss
                + self.kl_weight * kl_loss
            )

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": self.kl_weight,
        }

    def test_step(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

        kl_loss = self._get_kl_loss(z_mean, z_log_var, apply_free_bits=False)

        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
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
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)


#####################################################################################################
#####################################################################################################


if __name__ == "__main__":
    pass
