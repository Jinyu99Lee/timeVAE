import itertools
import logging
import sys
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

from src.utils import tpp_utils
from src.utils.fix_seq_ends import set_seq_to_zero_from_index
from src.nn.architectures.tpp_architecture import TPPArchitecture
from src.nn.architectures.mark_prediction_utils import (
    extract_marks_without_anchor_from_batch,
    compute_mark_ce_from_logits,
)
from src.nn.rnn.recurrent_nn import Recurrent_nn, RNNType
from src.nn.nn.vae_modules.vae_encoder import VAEEncoder
from src.nn.nn.vae_modules.vae_decoder import VAEDecoder
from src.data_types.sigw_loss_data_props import SigWLossDataProps


class Architecture_VAE(TPPArchitecture):
    """Temporal Conditional Variational AutoEncoder for TPP inter-arrival times.

    Training:  ELBO  =  KL(q_ξ(z|τ_i, h_{i-1}) ‖ N(0,I))  +  MSE(τ̂_i, τ_i)
    Sampling:  z ~ N(0,I)  →  decoder(z, h_{i-1})  →  τ̂_i

    Improvements over a vanilla VAE
    --------------------------------
    KL annealing (kl_anneal_epochs):
        The KL weight β is linearly ramped from 0 to 1 over the first
        kl_anneal_epochs epochs.  Training begins as pure reconstruction,
        which lets the encoder learn a meaningful latent code before the
        regulariser starts competing.  Validation always uses β = 1.

    Free bits (free_bits):
        Each latent dimension d is floored at λ = free_bits nats:
            KL̃_d = max(λ, KL_d)
        Gradient through dimension d is zeroed when KL_d < λ, preventing
        the encoder from collapsing that dimension to the prior.  The total
        KL has a hard lower bound of D: λ nats per position.

    Reconstruction weight (recon_weight):
        Scales the MSE term in the ELBO.  Useful when the reconstruction
        and KL terms are on very different scales.
    """

    @staticmethod
    def filter_patho_seqs(
        tensor1: torch.Tensor,
        lens_for_masking: torch.Tensor,
        tensor2: torch.Tensor = None,
    ):
        """Remove sequences of length ≤ 1 (trigger NaN in signature metric)."""
        mask_valid = lens_for_masking > 1
        tensor1, tensor2 = tpp_utils.apply_mask(tensor1, mask_valid, tensor2)
        lens_for_masking = lens_for_masking[mask_valid]
        return tensor1, lens_for_masking, tensor2

    def __init__(
        self,
        data_train: torch.Tensor,
        data_train_lens: torch.Tensor,
        data_val: torch.Tensor,
        data_val_lens: torch.Tensor,
        train_marks: torch.Tensor,
        val_marks: torch.Tensor,
        lr: float,
        hidden_size_rnn: int,
        concentration_factor: float,
        latent_dim: int,
        t_max: int,
        num_marks: int,
        period_plot_val: int,
        output_dir: str = None,
        enable_plot: bool = False,
        plot_every_n_val_steps: int = 10,
        kl_anneal_epochs: int = 200,
        free_bits: float = 0.0,
        recon_weight: float = 1.0,
        mark_loss_weight: float = 1.0,
        **kwargs,
    ):
        self.sigw_loss_properties = SigWLossDataProps(5, False, True)

        super().__init__(
            t_max,
            num_marks,
            data_train,
            data_train_lens,
            data_val,
            data_val_lens,
            train_marks,
            val_marks,
            concentration_factor,
            output_dir=output_dir,
            enable_plot=enable_plot,
            period_plot_val=period_plot_val,
            plot_every_n_val_steps=plot_every_n_val_steps,
        )

        if latent_dim >= hidden_size_rnn:
            logger.warning(
                f"latent_dim ({latent_dim}) >= hidden_size_rnn ({hidden_size_rnn}); "
                "the latent space is not a bottleneck of the history representation."
            )

        self.latent_dim: int = latent_dim
        self.lr: float = lr
        self.hid_size_rep: int = hidden_size_rnn
        self.kl_anneal_epochs: int = kl_anneal_epochs
        self.free_bits: float = free_bits
        self.recon_weight: float = recon_weight
        self.seq_len = data_train.shape[1] - 1  # number of inter-arrival times

        # Learnable mark head (CE-head pattern, same as sigwgan/wgan/score).
        self.mark_loss_weight: float = mark_loss_weight
        rnn_input_size = self._init_mark_components(history_size=hidden_size_rnn)

        # History encoder (same pattern as Architecture_Score)
        self.enc_rnn = Recurrent_nn(
            rnn_input_size,
            self.hid_size_rep,
            1,
            False,
            0.0,
            sys.maxsize,
            RNNType.LSTM,
            True,
        )

        # VAE components
        self.vae_encoder = VAEEncoder(self.hid_size_rep, latent_dim)
        self.vae_decoder = VAEDecoder(latent_dim, self.hid_size_rep)

        self.register_gradient_clipping()
        # reduction='none' so we can apply the sequence-length mask manually
        self.mse_loss = nn.MSELoss(reduction='none')
        return

    def configure_optimizers(self):
        param_groups = [
            self.enc_rnn.parameters(),
            self.vae_encoder.parameters(),
            self.vae_decoder.parameters(),
        ]
        if self.use_marks:
            param_groups.extend([self.event_emb.parameters(), self.mark_predictor.parameters()])
        else:
            param_groups.append(self.time_emb.parameters())
        return torch.optim.Adam(
            itertools.chain(*param_groups),
            lr=self.lr,
            weight_decay=0.0,
        )

    # ------------------------------------------------------------------
    # ELBO loss
    # ------------------------------------------------------------------

    def _embed_history(
        self,
        log_inter_arr_times: torch.Tensor,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed a full teacher-forced history using time-only or time+mark embeddings."""
        if self.use_marks and marks is not None:
            return self.event_emb(log_inter_arr_times, marks)
        return self.time_emb(log_inter_arr_times)

    def _encode_history(
        self,
        log_inter_arr_times: torch.Tensor,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a full teacher-forced history without sampling."""
        emb = self._embed_history(log_inter_arr_times, marks)
        hn, cn = self.enc_rnn.get_first_hidden_state(log_inter_arr_times.shape[0])
        h_all, _ = self.enc_rnn(emb, (hn, cn))
        return h_all

    def _compute_mark_latent_history_for_eval(
        self,
        *,
        marks_with_anchor: torch.Tensor,
        marks_full: torch.Tensor,
        dts: torch.Tensor,
        dt_lens: torch.Tensor,
        current_targets: torch.Tensor,
    ) -> typing.Optional[torch.Tensor]:
        """Build next-mark-aligned latent history for shared mark evaluation."""
        with torch.no_grad():
            val_dts_scaled = self.scaler_exp(dts)
            h_all = self._encode_history(val_dts_scaled, marks_full)
            return h_all[:, :-1, :]

    def _compute_elbo_loss(
        self,
        log_inter_arr_times: torch.Tensor,
        lengths: torch.Tensor,
        marks: typing.Optional[torch.Tensor] = None,
        kl_weight: float = 1.0,
    ) -> typing.Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        typing.Optional[torch.Tensor],
        torch.Tensor,
        typing.Optional[torch.Tensor],
    ]:
        """Vectorised ELBO over the full sequence.

        Args:
            log_inter_arr_times: (N, L, 1)  scaled log inter-arrival times
            lengths:             (N,)       number of valid positions in the L dimension
            marks:               (N, L) or None  mark indices (anchor already stripped)

        Returns:
            Tuple of ``(total_elbo, kl_mean, recon_mean, ce_loss, time_elbo, mark_logits)``.
        """
        # 1. History encoding
        h_all = self._encode_history(log_inter_arr_times, marks)  # (N, L, H)

        # Align: predict τ_{1..L-1} given h_{0..L-2}
        h_prev = h_all[:, :-1, :]  # (N, L-1, H)
        tau_obs = log_inter_arr_times[:, 1:, :]  # (N, L-1, 1)

        # 2. Variational encoding
        mu, log_var = self.vae_encoder(tau_obs, h_prev)  # (N, L-1, D)
        z = VAEEncoder.reparameterize(mu, log_var)  # (N, L-1, D)

        # 3. Decoding
        tau_hat = self.vae_decoder(z, h_prev)  # (N, L-1, 1)

        # 4. Mask for valid positions.  set_seq_to_zero_from_index zeros positions
        # >= (lengths-2), so valid positions are 0 .. lengths-3 in the L-1 tensor.
        # This is intentionally one position shorter than the full sequence, matching
        # the Architecture_Score convention (the last inter-arrival time is excluded).
        mask = set_seq_to_zero_from_index(torch.ones_like(tau_obs), lengths - 2)  # (N, L-1, 1), 0/1

        # 5. KL divergence: closed-form KL(N(μ, σ²) ‖ N(0,I))
        #    = 0.5 * Σ_d (μ² + exp(log_var) - log_var - 1)
        kl_per_dim = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1.0)  # (N, L-1, D)
        if self.free_bits > 0.0:
            kl_per_dim = kl_per_dim.clamp(min=self.free_bits)
        kl_per_pos = kl_per_dim.sum(dim=-1, keepdim=True)  # (N, L-1, 1)
        kl_masked = kl_per_pos * mask

        # 6. MSE reconstruction loss
        recon_masked = self.mse_loss(tau_hat, tau_obs) * mask  # (N, L-1, 1)

        # 7. Mean over valid positions
        n_valid = mask.sum().clamp(min=1.0)
        kl_mean = kl_masked.sum() / n_valid
        recon_mean = recon_masked.sum() / n_valid
        time_elbo = kl_weight * kl_mean + self.recon_weight * recon_mean

        # 8. Mark CE loss (added to ELBO when marks are present)
        ce_loss = None
        mark_logits = None
        total_elbo = time_elbo
        if self.use_marks:
            mark_logits = self.mark_predictor(h_prev)
            ce_loss = compute_mark_ce_from_logits(mark_logits, marks, lengths)
            total_elbo = time_elbo + self.mark_loss_weight * ce_loss

        return total_elbo, kl_mean, recon_mean, ce_loss, time_elbo, mark_logits

    def training_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor], batch_nb: int):
        data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
        kl_weight = min(1.0, self.current_epoch / self.kl_anneal_epochs) if self.kl_anneal_epochs > 0 else 1.0
        marks = extract_marks_without_anchor_from_batch(batch) if self.use_marks else None
        total_loss, kl, recon, ce_loss, time_elbo, mark_logits = self._compute_elbo_loss(
            data_dts_scaled, data_dts_lens, marks=marks, kl_weight=kl_weight
        )
        metrics = {'elbo': time_elbo, 'kl': kl, 'recon': recon}
        if self.use_marks and ce_loss is not None and mark_logits is not None:
            self._compute_and_log_mark_metrics_from_logits(
                mark_logits=mark_logits,
                marks=marks,
                lengths=data_dts_lens,
                prefix='train_',
                include_ce=True,
                include_accuracy=False,
                precomputed_ce=ce_loss,
            )
        self._log_all_metrics(metrics, 'train_')
        return total_loss

    def validation_step(self, batch, batch_nb):
        data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
        marks = extract_marks_without_anchor_from_batch(batch) if self.use_marks else None
        total_loss, kl, recon, ce_loss, time_elbo, mark_logits = self._compute_elbo_loss(
            data_dts_scaled, data_dts_lens, marks=marks, kl_weight=1.0
        )
        if self.use_marks:
            self._compute_and_log_mark_metrics_from_logits(
                mark_logits=mark_logits,
                marks=marks,
                lengths=data_dts_lens,
                prefix="val_",
                include_ce=True,
                precomputed_ce=ce_loss,
            )

        uncond_result = self.sample_and_fix_seqs(num_seq=batch[0].shape[0])
        hist_it = self.metrics_val.histogram_loss_it(uncond_result.its_scaled_nan, [])
        hist_int = self.metrics_val.histogram_loss_cum(uncond_result.cum_rel_nan, [])

        metrics = {
            'elbo': time_elbo,
            'kl': kl,
            'recon': recon,
            'epdf': (hist_it + hist_int) / 2.0,
            'hist_it': hist_it,
            'hist_int': hist_int,
        }

        self._log_all_metrics(metrics, "val_")
        return total_loss

    def sample(
        self,
        *,
        num_seq: typing.Optional[int] = None,
        starting_times: typing.Optional[torch.Tensor] = None,
        log_inter_arr_times: typing.Optional[torch.Tensor] = None,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        """Generate scaled inter-arrival times.

        Either num_seq (unconditional) or log_inter_arr_times (conditional) must be
        provided : not both.  Returns (samples, latent_rep_history, gen_marks) of shapes
        (N, L, 1), (N, L, H), and Optional (N, L) respectively.
        """
        assert (num_seq is not None) ^ (
            log_inter_arr_times is not None
        ), "Provide exactly one of num_seq or log_inter_arr_times."
        assert (starting_times is None) == (
            log_inter_arr_times is None
        ), "starting_times must be provided iff log_inter_arr_times is provided."

        if log_inter_arr_times is not None:
            samples, h_all = self._sample_conditional(starting_times, log_inter_arr_times, marks=marks)
            return samples, h_all, None
        samples, latent_rep_history, gen_marks = self._sample_unconditional(num_seq)
        return samples, latent_rep_history, gen_marks

    def _sample_conditional(
        self,
        starting_times: torch.Tensor,
        log_inter_arr_times: torch.Tensor,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Predict the full sequence conditioned on the true history.

        For each position i in 1..L-1, samples z ~ N(0,I) and decodes given h_{i-1}.
        Position 0 is taken directly from log_inter_arr_times (seed / first IT).

        Args:
            starting_times:      (N, 1, 1)  anchor times (unused here, kept for API compat.)
            log_inter_arr_times: (N, L, 1)  scaled log inter-arrival times (true history)
            marks:               (N, L) or None  mark indices (anchor already stripped)
        Returns:
            samples:             (N, L, 1)  scaled samples  (pos-0 from data, rest from model)
            h_all:               (N, L, H)  full history encoding
        """
        h_all = self._encode_history(log_inter_arr_times, marks)

        h_prev = h_all[:, :-1, :]  # (N, L-1, H)

        # Sample from prior
        tau_hat = self.vae_decoder.sample(h_prev)  # (N, L-1, 1)
        tau_hat = tau_hat.clamp(min=self.MIN_SCALED_DATA, max=self.MAX_SCALED_DATA)

        # Prepend the seed (position 0 taken from the true data)
        first_it = log_inter_arr_times[:, :1, :]  # (N, 1, 1)
        samples = torch.cat([first_it, tau_hat], dim=1)  # (N, L, 1)

        return samples, h_all

    def _sample_unconditional(
        self, num_seq: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        """Generate full sequences autoregressively without conditioning on true data.

        Sequence layout (same as Architecture_Score):
          - Position 0: first inter-arrival time sampled from training distribution.
          - Position i>0: decoder(z ~ N(0,I), h_{i-1}).

        Args:
            num_seq: number of sequences to generate
        Returns:
            samples:             (N, L_train+1, 1)  scaled samples
            latent_rep_history:  (N, L_train, H)
            gen_marks:           (N, L_train+1) or None
        """
        L_train = self.data_train_dts.shape[1]
        L_full = L_train + 1

        samples = torch.zeros((num_seq, L_full, self.num_dim_seqs), device=self.device)
        latent_rep_history = torch.zeros((num_seq, L_train, self.enc_rnn.hidden_size), device=self.device)

        # Co-sample first IT and first mark from the same training sequence.
        first_it_seqs, first_mark = self._sample_first_event(num_seq)
        samples[:, 0] = first_it_seqs

        hn, cn = self.enc_rnn.get_first_hidden_state(num_seq)

        # Allocate mark history for unconditional generation.
        gen_marks = None
        if self.num_marks > 1:
            gen_marks = torch.full((num_seq, L_full), -1, dtype=torch.long, device=self.device)
            gen_marks[:, 0] = first_mark
            running_marks = first_mark
        else:
            running_marks = None

        for i in range(1, L_full):
            if running_marks is not None:
                emb_i = self.event_emb(samples[:, i - 1 : i, :], running_marks.unsqueeze(1))
            else:
                emb_i = self.time_emb(samples[:, i - 1 : i, :])  # (N, 1, E)
            latent_rep_history[:, i - 1 : i], (hn, cn) = self.enc_rnn(emb_i, (hn, cn))

            tau_hat = self.vae_decoder.sample(latent_rep_history[:, i - 1 : i])  # (N, 1, 1)
            samples[:, i : i + 1] = tau_hat.clamp(min=self.MIN_SCALED_DATA, max=self.MAX_SCALED_DATA)

            # Sample next mark autoregressively from the mark predictor.
            if running_marks is not None:
                mark_logits = self.mark_predictor(latent_rep_history[:, i - 1 : i])
                mark_probs = F.softmax(mark_logits.squeeze(1), dim=-1)
                running_marks = torch.multinomial(mark_probs, 1).squeeze(-1)
                gen_marks[:, i] = running_marks

        return samples, latent_rep_history, gen_marks
