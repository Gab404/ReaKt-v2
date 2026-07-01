"""
src/autoencoder/model.py
========================
Raman spectra autoencoders used as frozen feature extractors for the
downstream PI-LSTM / Neural ODE soft-sensor benchmark.

Two classes are exposed:

* ``CDAE_Raman`` — Convolutional Denoising Autoencoder
* ``CVAE_Raman`` — Convolutional Variational Autoencoder

Both share the same 1D convolutional encoder / decoder backbone:

    Encoder  (input length L, e.g. 2001)
        Conv1d(  1 ->  32, k=7) + BatchNorm + ReLU + MaxPool(2)
        Conv1d( 32 ->  64, k=5) + BatchNorm + ReLU + MaxPool(2)
        Conv1d( 64 -> 128, k=3) + BatchNorm + ReLU + MaxPool(2)
        Flatten ( = 128 * L//8 )
        Linear -> latent_dim                         (CDAE)
        Linear -> mu, log_var (heads)                (CVAE)

    Decoder
        Linear -> 128 * L//8 + Reshape
        ConvTranspose1d(128 ->  64, k=3, stride=2) + BatchNorm + ReLU
        ConvTranspose1d( 64 ->  32, k=5, stride=2) + BatchNorm + ReLU
        ConvTranspose1d( 32 ->   1, k=7, stride=2)
        Crop / pad to input length L

The classes are designed to be **loadable from the published V2
checkpoints** (``checkpoints/cdae_best.pt`` / ``checkpoints/cvae_best.pt``)
whose state-dict keys are::

    enc_block1.0 / .1                              (Conv1d + BatchNorm1d)
    enc_block2.0 / .1
    enc_block3.0 / .1
    encoder_fc                                     (CDAE)
    fc_mu / fc_log_var                             (CVAE)
    decoder_fc
    dec_block1.0 / .1                              (ConvTranspose1d + BatchNorm1d)
    dec_block2.0 / .1
    dec_block3.0                                   (ConvTranspose1d, no BN)

Only the encode() method is used by the benchmark; decode() / forward()
exist for completeness and self-tests.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CDAE
# ─────────────────────────────────────────────────────────────────────────────
class CDAE_Raman(nn.Module):
    """
    Convolutional Denoising Autoencoder for 1-D Raman spectra.

    Parameters
    ----------
    input_length : int
        Number of wavenumber channels in the input spectrum (e.g. 2001).
    latent_dim   : int
        Bottleneck dimensionality (default 64).
    noise_std    : float
        Gaussian noise std added to the input during ``forward`` (training).
        Set to 0.0 for deterministic inference (default in encoders).
    """

    def __init__(
        self,
        input_length: int = 2001,
        latent_dim:   int = 64,
        noise_std:    float = 0.1,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dim   = int(latent_dim)
        self.noise_std    = float(noise_std)

        # After 3 MaxPool(2) layers, the spatial dimension is input_length // 8
        self._pooled_len  = self.input_length // 8
        self._flatten_dim = 128 * self._pooled_len

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc_block1 = nn.Sequential(
            nn.Conv1d(  1,  32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_block2 = nn.Sequential(
            nn.Conv1d( 32,  64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_block3 = nn.Sequential(
            nn.Conv1d( 64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.encoder_fc = nn.Linear(self._flatten_dim, self.latent_dim)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoder_fc = nn.Linear(self.latent_dim, self._flatten_dim)
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose1d(128,  64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose1d( 64,  32, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose1d( 32,   1, kernel_size=7,
                               stride=2, padding=3, output_padding=1),
        )

    # ── Public API ──────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, L)`` or ``(B, 1, L)`` spectra to a ``(B, latent_dim)`` latent."""
        if x.dim() == 2:
            x = x.unsqueeze(1)               # (B, 1, L)
        z = self.enc_block1(x)
        z = self.enc_block2(z)
        z = self.enc_block3(z)
        z = z.view(z.size(0), -1)            # (B, 128 * L//8)
        return self.encoder_fc(z)            # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a ``(B, latent_dim)`` latent back to a ``(B, L)`` spectrum."""
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 128, self._pooled_len)
        h = self.dec_block1(h)
        h = self.dec_block2(h)
        h = self.dec_block3(h)               # (B, 1, ~L)
        h = h.squeeze(1)
        # Crop / pad to the exact input length (ConvTranspose may overshoot/undershoot)
        if h.size(1) > self.input_length:
            h = h[:, : self.input_length]
        elif h.size(1) < self.input_length:
            pad = self.input_length - h.size(1)
            h = F.pad(h, (0, pad))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add denoising noise (training mode) and reconstruct the spectrum."""
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.decode(self.encode(x))


# ─────────────────────────────────────────────────────────────────────────────
# CVAE
# ─────────────────────────────────────────────────────────────────────────────
class CVAE_Raman(nn.Module):
    """
    Convolutional Variational Autoencoder for 1-D Raman spectra.

    Shares the same conv backbone as :class:`CDAE_Raman` but exposes two
    bottleneck heads (``fc_mu``, ``fc_log_var``) and a reparameterised
    sampling step.  At inference the posterior mean ``mu`` is used as the
    deterministic latent representation.

    Parameters
    ----------
    input_length : int    -- number of wavenumber channels (default 2001)
    latent_dim   : int    -- latent dimensionality (default 64)
    """

    def __init__(
        self,
        input_length: int = 2001,
        latent_dim:   int = 64,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dim   = int(latent_dim)

        self._pooled_len  = self.input_length // 8
        self._flatten_dim = 128 * self._pooled_len

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc_block1 = nn.Sequential(
            nn.Conv1d(  1,  32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_block2 = nn.Sequential(
            nn.Conv1d( 32,  64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_block3 = nn.Sequential(
            nn.Conv1d( 64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.fc_mu      = nn.Linear(self._flatten_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self._flatten_dim, self.latent_dim)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoder_fc = nn.Linear(self.latent_dim, self._flatten_dim)
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose1d(128,  64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose1d( 64,  32, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose1d( 32,   1, kernel_size=7,
                               stride=2, padding=3, output_padding=1),
        )

    # ── Public API ──────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the posterior parameters ``(mu, log_var)``.

        Accepts inputs of shape ``(B, L)`` or ``(B, 1, L)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.enc_block1(x)
        z = self.enc_block2(z)
        z = self.enc_block3(z)
        z = z.view(z.size(0), -1)
        return self.fc_mu(z), self.fc_log_var(z)

    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 128, self._pooled_len)
        h = self.dec_block1(h)
        h = self.dec_block2(h)
        h = self.dec_block3(h)
        h = h.squeeze(1)
        if h.size(1) > self.input_length:
            h = h[:, : self.input_length]
        elif h.size(1) < self.input_length:
            pad = self.input_length - h.size(1)
            h = F.pad(h, (0, pad))
        return h

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, mu, log_var)``."""
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var) if self.training else mu
        return self.decode(z), mu, log_var
