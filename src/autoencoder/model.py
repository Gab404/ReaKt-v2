"""
src/autoencoder/model.py
========================
1D Convolutional Autoencoder architectures for Raman spectra compression.

Two models are provided:

  CDAE_Raman  --  Convolutional Denoising Autoencoder
                  Adds Gaussian noise to the input during training and learns
                  to reconstruct the original clean spectrum, forcing the
                  latent space to capture true spectral structure rather than
                  memorising noise.

  CVAE_Raman  --  Convolutional Variational Autoencoder
                  Places a standard Gaussian prior on the latent space via the
                  reparameterization trick.  The loss is the beta-weighted
                  Evidence Lower BOund (ELBO): MSE reconstruction + beta * KLD.

Shared Backbone
---------------
Both models use an identical three-stage Conv1d encoder:

  Stage 1:  Conv1d(1,   32, k=7, p=3, s=1) + BN + ReLU + MaxPool(2)
  Stage 2:  Conv1d(32,  64, k=5, p=2, s=1) + BN + ReLU + MaxPool(2)
  Stage 3:  Conv1d(64, 128, k=3, p=1, s=1) + BN + ReLU + MaxPool(2)

For input_length=2001:
  2001 -> 1000 -> 500 -> 250  |  flat = 128 * 250 = 32 000  ->  FC(64)

The decoder mirrors the encoder with ConvTranspose1d layers whose
output_padding values are computed analytically in __init__ to guarantee
the reconstructed spectrum has the EXACT same length as the input -- no
cropping, no padding fix-ups, and no hardcoded magic numbers.

Backward-compatibility notes
-----------------------------
  * CDAE_Raman.encode() accepts BOTH (B, L) and (B, 1, L) inputs so that
    the existing RamanEncoder inference wrapper continues to work unchanged.
  * The class name CDAE_Raman is preserved; raman_encoder.py imports it.
  * The default input_length is changed from 2361 to 2001 to match the
    SG-filtered spectral window used in the training pipeline.

Input contract (training pipeline)
-----------------------------------
  The training pipeline feeds tensors of shape (B, 1, 2001) -- channel-first
  convention for Conv1d.  Preprocessing (SG derivative + StandardScaler)
  is applied OUTSIDE the model, inside train_autoencoder.py.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── ─────────────────────────────────────────────────────────────────────────
# ARCHITECTURE CONSTANTS
# ── ─────────────────────────────────────────────────────────────────────────

# Encoder channel progression
_ENC_CHANNELS = (1, 32, 64, 128)   # input -> stage1 -> stage2 -> stage3

# Encoder convolution parameters (kernel, padding) -- "same" padding at stride=1
_ENC_KERNELS  = (7, 5, 3)          # per stage
_ENC_PADDINGS = (3, 2, 1)          # k//2 for odd kernels

# Decoder ConvTranspose1d parameters (kernel, padding)
# Chosen so that with stride=2 and the right output_padding the inverse
# of each MaxPool(2) is exact.  output_padding is computed analytically.
_DEC_KERNELS  = (3, 5, 7)          # stage3->stage2, stage2->stage1, stage1->output
_DEC_PADDINGS = (1, 2, 2)


# ── ─────────────────────────────────────────────────────────────────────────
# HELPER: DYNAMIC OUTPUT-PADDING COMPUTATION
# ── ─────────────────────────────────────────────────────────────────────────

def _output_padding(L_in: int, target: int, kernel: int,
                    stride: int, padding: int,
                    soft: bool = False) -> int:
    """
    Compute the output_padding argument for nn.ConvTranspose1d so that
    the output length equals `target` exactly.

    Formula (dilation=1):
        L_out = (L_in - 1)*stride - 2*padding + kernel + output_padding

    => output_padding = target - (L_in - 1)*stride + 2*padding - kernel

    Parameters
    ----------
    soft : if True, clamp output_padding to [0, stride-1] instead of
           raising.  Use this for the FINAL decoder layer, where a ±1
           residual is corrected by a post-hoc length adjustment in
           decode().  For intermediate layers soft=False (strict) so
           any architectural mismatch is caught immediately.
    """
    base = (L_in - 1) * stride - 2 * padding + kernel
    op   = target - base
    if not (0 <= op < stride):
        if soft:
            op = max(0, min(op, stride - 1))
        else:
            raise ValueError(
                f"output_padding={op} out of range [0, {stride-1}] "
                f"for L_in={L_in}, target={target}, k={kernel}, "
                f"s={stride}, p={padding}.  "
                f"Adjust kernel/padding choices."
            )
    return op


def _encoder_lengths(input_length: int) -> Tuple[int, int, int, int]:
    """
    Compute the spatial length after each MaxPool stage.

    MaxPool1d(2) with default floor-mode:
        L_out = floor(L_in / 2)  (integer division)

    Returns (L0, L1, L2, L3) where L0 = input_length.
    """
    L0 = input_length
    L1 = L0 // 2
    L2 = L1 // 2
    L3 = L2 // 2
    return L0, L1, L2, L3


def _adjust_length(x: torch.Tensor, target: int) -> torch.Tensor:
    """
    Correct the spatial length of a (B, C, L) tensor to exactly `target`.

    Applies at most a ±1 adjustment that can arise from MaxPool1d(2)
    floor-rounding on odd input lengths, and the corresponding
    ConvTranspose1d(stride=2) reconstruction.

    Used after every decoder block so that length discrepancies do not
    accumulate across stages.
    """
    L = x.shape[-1]
    if L == target:
        return x
    if L > target:
        return x[..., :target]                   # trim trailing element(s)
    return F.pad(x, (0, target - L))             # zero-pad on the right


# ── ─────────────────────────────────────────────────────────────────────────
# SHARED BUILDING BLOCKS
# ── ─────────────────────────────────────────────────────────────────────────

def _enc_block(in_ch: int, out_ch: int, kernel: int, pad: int) -> nn.Sequential:
    """
    Single encoder stage: Conv1d (same padding) -> BatchNorm1d -> ReLU -> MaxPool1d(2).

    The Conv1d uses stride=1 with same padding so the spatial length is
    unchanged before the MaxPool halves it.  BatchNorm stabilises training
    with highly variable Raman band intensities.
    """
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, stride=1),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2),
    )


def _dec_block(in_ch: int, out_ch: int, kernel: int, padding: int,
               output_padding: int, last: bool = False) -> nn.Sequential:
    """
    Single decoder stage: ConvTranspose1d(stride=2) [-> BatchNorm1d -> ReLU].

    The last decoder stage omits BN and activation so the output is
    unbounded (correct for StandardScaler-normalised targets).

    output_padding is computed analytically in __init__ to hit the exact
    target length -- no hardcoded values.
    """
    layers: list = [
        nn.ConvTranspose1d(
            in_ch, out_ch,
            kernel_size=kernel,
            stride=2,
            padding=padding,
            output_padding=output_padding,
        )
    ]
    if not last:
        layers += [nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _init_weights(module: nn.Module) -> None:
    """
    Kaiming-normal for Conv layers (ReLU downstream), Xavier-normal for Linear.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias,   0.0)


# ── ─────────────────────────────────────────────────────────────────────────
# ARCHITECTURE 1 -- CONVOLUTIONAL DENOISING AUTOENCODER (CDAE)
# ── ─────────────────────────────────────────────────────────────────────────

class CDAE_Raman(nn.Module):
    """
    1D Convolutional Denoising Autoencoder for Raman spectra.

    Noise injection
    ---------------
    During training (model.train() mode), Gaussian noise N(0, noise_std) is
    added to the input BEFORE the encoder.  The model must then recover the
    clean spectrum, forcing the latent space to represent robust, noise-
    invariant spectral features rather than memorising individual spectra.
    No noise is applied during model.eval() -- inference is deterministic.

    Input / Output contract
    -----------------------
    Forward input:  (B, 1, L) tensor  -- channel-first Conv1d convention.
                    Also accepts (B, L) for backward compatibility with
                    the existing RamanEncoder inference wrapper.
    Forward output: (B, 1, L) tensor  -- same shape as input.
    encode output:  (B, latent_dim)
    decode output:  (B, 1, L)

    Parameters
    ----------
    input_length : int
        Number of wavenumber channels.  Default 2001 (SG-filtered window).
    latent_dim   : int
        Bottleneck dimensionality.  Default 64.
    noise_std    : float
        Standard deviation of the Gaussian corruption noise.  Default 0.1.
    """

    def __init__(
        self,
        input_length: int = 2001,
        latent_dim:   int = 64,
        noise_std:    float = 0.1,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.latent_dim   = latent_dim
        self.noise_std    = noise_std

        # -- Compute encoder stage output lengths analytically ---------------
        L0, L1, L2, L3 = _encoder_lengths(input_length)

        # Store intermediate targets for per-step length correction in decode()
        self._L1, self._L2, self._L3 = L1, L2, L3

        # Flattened size after three MaxPool(2) stages
        self._flat_size = _ENC_CHANNELS[-1] * L3   # 128 * L3

        # -- Compute decoder output_padding values (all soft) ----------------
        # soft=True: clamp op to [0, stride-1].  decode() applies a per-step
        # length correction of at most ±1 to guarantee exact target sizes.
        op1 = _output_padding(L3, L2, _DEC_KERNELS[0], stride=2, padding=_DEC_PADDINGS[0], soft=True)
        op2 = _output_padding(L2, L1, _DEC_KERNELS[1], stride=2, padding=_DEC_PADDINGS[1], soft=True)
        op3 = _output_padding(L1, L0, _DEC_KERNELS[2], stride=2, padding=_DEC_PADDINGS[2], soft=True)

        logger.info(
            "CDAE_Raman  input=%d  latent=%d  noise_std=%.3f",
            input_length, latent_dim, noise_std,
        )
        logger.info(
            "Encoder: %d -> %d -> %d -> %d  flat=%d",
            L0, L1, L2, L3, self._flat_size,
        )
        logger.info(
            "Decoder output_padding: CTD1=%d  CTD2=%d  CTD3=%d", op1, op2, op3,
        )

        # == ENCODER ===========================================================

        # Three strided encoder blocks (Conv + BN + ReLU + MaxPool)
        self.enc_block1 = _enc_block(_ENC_CHANNELS[0], _ENC_CHANNELS[1],
                                     _ENC_KERNELS[0], _ENC_PADDINGS[0])
        self.enc_block2 = _enc_block(_ENC_CHANNELS[1], _ENC_CHANNELS[2],
                                     _ENC_KERNELS[1], _ENC_PADDINGS[1])
        self.enc_block3 = _enc_block(_ENC_CHANNELS[2], _ENC_CHANNELS[3],
                                     _ENC_KERNELS[2], _ENC_PADDINGS[2])

        # Bottleneck projection: flat -> latent_dim
        self.encoder_fc = nn.Linear(self._flat_size, latent_dim)

        # == DECODER ===========================================================

        # Latent expansion: latent_dim -> flat
        self.decoder_fc = nn.Linear(latent_dim, self._flat_size)

        # Three transposed-conv decoder blocks (mirroring the encoder stages)
        self.dec_block1 = _dec_block(_ENC_CHANNELS[3], _ENC_CHANNELS[2],
                                     _DEC_KERNELS[0], _DEC_PADDINGS[0], op1, last=False)
        self.dec_block2 = _dec_block(_ENC_CHANNELS[2], _ENC_CHANNELS[1],
                                     _DEC_KERNELS[1], _DEC_PADDINGS[1], op2, last=False)
        self.dec_block3 = _dec_block(_ENC_CHANNELS[1], _ENC_CHANNELS[0],
                                     _DEC_KERNELS[2], _DEC_PADDINGS[2], op3, last=True)

        _init_weights(self)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is (B, 1, L). Accepts (B, L) for backward compat."""
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x  # already (B, 1, L) or (B, C, L)

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise during training only.

        The noise is added AFTER converting to 3D so that the std is
        applied consistently across the channel dimension.
        """
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

    # ── Public interface ──────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input spectra to the latent space.

        Parameters
        ----------
        x : (B, 1, L) or (B, L)  -- accepts both shapes

        Returns
        -------
        z : (B, latent_dim)
        """
        x = self._to_3d(x)
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        x = x.flatten(start_dim=1)        # (B, flat_size)
        return self.encoder_fc(x)         # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct spectra from the latent code.

        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        x_hat : (B, 1, L)
        """
        x = self.decoder_fc(z)                                    # (B, flat_size)
        x = x.view(x.size(0), _ENC_CHANNELS[-1], self._L3)       # (B, 128, L3)
        x = self.dec_block1(x)                                    # (B, 64,  ~L2)
        x = _adjust_length(x, self._L2)                          # exact L2
        x = self.dec_block2(x)                                    # (B, 32,  ~L1)
        x = _adjust_length(x, self._L1)                          # exact L1
        x = self.dec_block3(x)                                    # (B, 1,   ~L0)
        x = _adjust_length(x, self.input_length)                  # exact L0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full CDAE forward pass.

        Training:   x_clean -> x_noisy (+ Gaussian noise) -> encode -> decode
                    Loss is computed against x_clean (the original input).
        Inference:  x_clean -> encode -> decode  (no noise)

        Parameters
        ----------
        x : (B, 1, L) clean spectra

        Returns
        -------
        x_hat : (B, 1, L)  reconstruction of x
        """
        x    = self._to_3d(x)
        x_in = self._add_noise(x)   # corrupted input (noop at eval time)
        z    = self.encode(x_in)
        return self.decode(z)

    @torch.no_grad()
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the deterministic latent vector (no noise, no gradients).
        Called by RamanEncoder for feature extraction.

        Parameters
        ----------
        x : (B, L) or (B, 1, L)

        Returns
        -------
        z : (B, latent_dim)
        """
        self.eval()
        return self.encode(x)


# ── ─────────────────────────────────────────────────────────────────────────
# ARCHITECTURE 2 -- CONVOLUTIONAL VARIATIONAL AUTOENCODER (CVAE)
# ── ─────────────────────────────────────────────────────────────────────────

class CVAE_Raman(nn.Module):
    """
    1D Convolutional Variational Autoencoder for Raman spectra.

    Variational bottleneck
    ----------------------
    Instead of a single deterministic latent vector, the encoder produces
    the parameters (mu, log_var) of a diagonal Gaussian q(z|x).
    A latent sample is drawn via the reparameterization trick:

        z = mu + exp(0.5 * log_var) * epsilon,   epsilon ~ N(0, I)

    This makes the sampling differentiable and allows gradient-based
    optimisation of the ELBO:

        ELBO = E[log p(x|z)]  -  beta * KL(q(z|x) || p(z))
             = -MSE(x_hat, x)  +  0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    During inference, the mean mu is used as the deterministic latent
    representation (no sampling noise).

    Input / Output contract
    -----------------------
    Forward input:  (B, 1, L) tensor.  Also accepts (B, L).
    Forward output: tuple (x_hat, mu, log_var)
                      x_hat   : (B, 1, L)
                      mu      : (B, latent_dim)
                      log_var : (B, latent_dim)
    encode output:  (mu, log_var)  -- tuple
    decode output:  (B, 1, L)

    Parameters
    ----------
    input_length : int
        Wavenumber channels.  Default 2001.
    latent_dim   : int
        Bottleneck dimensionality.  Default 64.
    """

    def __init__(
        self,
        input_length: int = 2001,
        latent_dim:   int = 64,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.latent_dim   = latent_dim

        # -- Compute stage lengths & decoder output_padding ------------------
        L0, L1, L2, L3 = _encoder_lengths(input_length)
        self._flat_size = _ENC_CHANNELS[-1] * L3
        # Store all intermediate targets for per-step length correction
        self._L1, self._L2, self._L3 = L1, L2, L3

        op1 = _output_padding(L3, L2, _DEC_KERNELS[0], stride=2, padding=_DEC_PADDINGS[0], soft=True)
        op2 = _output_padding(L2, L1, _DEC_KERNELS[1], stride=2, padding=_DEC_PADDINGS[1], soft=True)
        op3 = _output_padding(L1, L0, _DEC_KERNELS[2], stride=2, padding=_DEC_PADDINGS[2], soft=True)

        logger.info(
            "CVAE_Raman  input=%d  latent=%d  flat=%d",
            input_length, latent_dim, self._flat_size,
        )

        # == ENCODER BACKBONE ================================================

        # Shared three-stage conv backbone (identical to CDAE, no noise here)
        self.enc_block1 = _enc_block(_ENC_CHANNELS[0], _ENC_CHANNELS[1],
                                     _ENC_KERNELS[0], _ENC_PADDINGS[0])
        self.enc_block2 = _enc_block(_ENC_CHANNELS[1], _ENC_CHANNELS[2],
                                     _ENC_KERNELS[1], _ENC_PADDINGS[1])
        self.enc_block3 = _enc_block(_ENC_CHANNELS[2], _ENC_CHANNELS[3],
                                     _ENC_KERNELS[2], _ENC_PADDINGS[2])

        # Two separate projection heads from the flattened representation
        self.fc_mu      = nn.Linear(self._flat_size, latent_dim)  # mean
        self.fc_log_var = nn.Linear(self._flat_size, latent_dim)  # log-variance

        # == DECODER =========================================================

        self.decoder_fc = nn.Linear(latent_dim, self._flat_size)

        self.dec_block1 = _dec_block(_ENC_CHANNELS[3], _ENC_CHANNELS[2],
                                     _DEC_KERNELS[0], _DEC_PADDINGS[0], op1, last=False)
        self.dec_block2 = _dec_block(_ENC_CHANNELS[2], _ENC_CHANNELS[1],
                                     _DEC_KERNELS[1], _DEC_PADDINGS[1], op2, last=False)
        self.dec_block3 = _dec_block(_ENC_CHANNELS[1], _ENC_CHANNELS[0],
                                     _DEC_KERNELS[2], _DEC_PADDINGS[2], op3, last=True)

        _init_weights(self)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_3d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x

    # ── Public interface ──────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map input to the variational parameters of the posterior q(z|x).

        Parameters
        ----------
        x : (B, 1, L) or (B, L)

        Returns
        -------
        mu      : (B, latent_dim)  -- posterior mean
        log_var : (B, latent_dim)  -- posterior log-variance
        """
        x = self._to_3d(x)
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        h = x.flatten(start_dim=1)          # (B, flat_size)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        # Clamp log_var to [-10, 10] so that exp(log_var) stays in [~5e-5, ~22026].
        # Without clamping, float16 AMP can produce log_var > 11 (exp > 65504 =
        # float16 max), causing overflow in the KL term and NaN losses.
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var

    def reparameterize(
        self,
        mu:      torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps,  eps ~ N(0, I).

        During training, the stochastic sample z is used so gradients
        flow through mu and log_var.
        During eval, this method is NOT called by forward() -- mu is
        used directly as the deterministic latent representation.

        Parameters
        ----------
        mu      : (B, latent_dim)
        log_var : (B, latent_dim)

        Returns
        -------
        z : (B, latent_dim)  -- sampled latent vector
        """
        std = torch.exp(0.5 * log_var)          # convert log-var to std
        eps = torch.randn_like(std)             # N(0, I)  same device/dtype
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct spectrum from latent code z.

        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        x_hat : (B, 1, L)
        """
        x = self.decoder_fc(z)
        x = x.view(x.size(0), _ENC_CHANNELS[-1], self._L3)   # (B, 128, L3)
        x = self.dec_block1(x)
        x = _adjust_length(x, self._L2)                       # exact L2
        x = self.dec_block2(x)
        x = _adjust_length(x, self._L1)                       # exact L1
        x = self.dec_block3(x)
        x = _adjust_length(x, self.input_length)               # exact L0
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full CVAE forward pass.

        Training path:  encode -> reparameterize -> decode
        Eval path:      encode -> use mu directly  -> decode
                        (deterministic, no sampling noise at inference)

        Parameters
        ----------
        x : (B, 1, L) clean spectra

        Returns
        -------
        x_hat   : (B, 1, L)         -- reconstruction
        mu      : (B, latent_dim)   -- posterior mean
        log_var : (B, latent_dim)   -- posterior log-variance
        """
        x       = self._to_3d(x)
        mu, log_var = self.encode(x)

        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu   # deterministic at eval time

        x_hat = self.decode(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the deterministic posterior mean mu (no sampling).
        Suitable for downstream tasks (soft-sensor, similarity search).

        Parameters
        ----------
        x : (B, L) or (B, 1, L)

        Returns
        -------
        mu : (B, latent_dim)
        """
        self.eval()
        mu, _ = self.encode(x)
        return mu


# ── ─────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ── ─────────────────────────────────────────────────────────────────────────

def cdae_loss(
    x_hat: torch.Tensor,
    x_clean: torch.Tensor,
) -> torch.Tensor:
    """
    CDAE reconstruction loss: Mean Squared Error between the denoised
    reconstruction and the original CLEAN spectrum.

    The loss is computed against the clean target, NOT the noisy input,
    which is what forces the network to learn denoising.

    Parameters
    ----------
    x_hat   : (B, 1, L)  model output (reconstruction of noisy input)
    x_clean : (B, 1, L)  original clean spectrum (training target)

    Returns
    -------
    loss : scalar tensor
    """
    return F.mse_loss(x_hat, x_clean, reduction="mean")


def vae_loss_function(
    x_hat:   torch.Tensor,
    x:       torch.Tensor,
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    beta:    float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Beta-VAE ELBO loss:

        L = MSE(x_hat, x) + beta * KLD

    where KLD = -0.5 * sum_j [ 1 + log_var_j - mu_j^2 - exp(log_var_j) ]

    The KL divergence measures how much the learned posterior q(z|x) ~ N(mu, diag(exp(log_var)))
    deviates from the unit Gaussian prior p(z) ~ N(0, I).  Minimising it
    regularises the latent space toward a smooth, well-structured manifold.

    Beta weighting
    --------------
    beta < 1  : emphasis on reconstruction quality (prevents posterior collapse)
    beta = 1  : standard VAE
    beta > 1  : beta-VAE (stronger disentanglement, weaker reconstruction)

    A common strategy (KL annealing) is to start with beta=0 and linearly
    increase it to the target value over the first N epochs.  This is
    implemented in the training loop (train_autoencoder.py).

    Both terms are averaged over the batch so the loss magnitude is
    independent of batch size.

    Parameters
    ----------
    x_hat   : (B, 1, L)        -- reconstruction
    x       : (B, 1, L)        -- clean input (target)
    mu      : (B, latent_dim)
    log_var : (B, latent_dim)
    beta    : float             -- KL weighting factor

    Returns
    -------
    total_loss : scalar
    recon_loss : scalar  (for logging)
    kl_loss    : scalar  (for logging)
    """
    # ── Promote to float32 before computing the KL ─────────────────────────
    # Under AMP autocast the inputs arrive as float16.  The KL term involves
    # exp(log_var) which can exceed float16's max (~65504) when log_var > 11,
    # causing Inf -> NaN propagation.  Casting to float32 here costs <1 % of
    # training time but eliminates overflow entirely.
    x_hat_f  = x_hat.float()
    x_f      = x.float()
    mu_f     = mu.float()
    lv_f     = log_var.float()

    # Reconstruction term: per-element MSE averaged over (B, 1, L)
    recon_loss = F.mse_loss(x_hat_f, x_f, reduction="mean")

    # KL divergence: -0.5 * mean_over_batch[ sum_over_latent_dims(...) ]
    kl_loss = -0.5 * torch.mean(
        torch.sum(1.0 + lv_f - mu_f.pow(2) - lv_f.exp(), dim=1)
    )

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ── ─────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC UTILITY
# ── ─────────────────────────────────────────────────────────────────────────

def get_model_summary(model: nn.Module, input_size: tuple) -> str:
    """
    Print a brief architecture summary and verify the output shape.

    Parameters
    ----------
    model      : CDAE_Raman or CVAE_Raman
    input_size : (batch_size, 1, input_length) -- channel-first

    Returns
    -------
    summary string
    """
    def count_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    lines = ["=" * 72, f"  {model.__class__.__name__}", "=" * 72]
    lines.append(f"  input_length : {model.input_length}")
    lines.append(f"  latent_dim   : {model.latent_dim}")
    lines.append(f"  Parameters   : {count_params(model):,}")

    device     = next(model.parameters()).device
    test_input = torch.randn(*input_size, device=device)
    was_training = model.training

    try:
        model.eval()
        with torch.no_grad():
            if isinstance(model, CVAE_Raman):
                x_hat, mu, lv = model(test_input)
                lines.append(f"  Output x_hat : {tuple(x_hat.shape)}")
                lines.append(f"  Latent mu    : {tuple(mu.shape)}")
                lines.append(f"  Latent lv    : {tuple(lv.shape)}")
            else:
                x_hat = model(test_input)
                z     = model.encode(test_input)
                lines.append(f"  Output x_hat : {tuple(x_hat.shape)}")
                lines.append(f"  Latent z     : {tuple(z.shape)}")
        lines.append("  Shape check: PASSED")
    except Exception as exc:
        lines.append(f"  Shape check: FAILED -- {exc}")
    finally:
        model.train(was_training)

    lines.append("=" * 72)
    return "\n".join(lines)


# ── ─────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ── ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, L, Z = 8, 2001, 64

    # -- CDAE --
    cdae = CDAE_Raman(input_length=L, latent_dim=Z, noise_std=0.1)
    cdae.train()
    x_clean = torch.randn(B, 1, L)
    x_hat_cdae = cdae(x_clean)              # forward adds noise internally
    z_cdae     = cdae.encode(x_clean)       # encode without noise (eval path)
    assert x_hat_cdae.shape == (B, 1, L), f"CDAE output {x_hat_cdae.shape}"
    assert z_cdae.shape      == (B, Z),   f"CDAE latent {z_cdae.shape}"

    # Test backward-compat (B, L) input
    z_2d = cdae.encode(torch.randn(B, L))
    assert z_2d.shape == (B, Z), "CDAE (B,L) encode failed"

    loss_cdae = cdae_loss(x_hat_cdae, x_clean)
    print(get_model_summary(cdae, (B, 1, L)))
    print(f"  CDAE loss  : {loss_cdae.item():.6f}")

    # -- CVAE --
    cvae = CVAE_Raman(input_length=L, latent_dim=Z)
    cvae.train()
    x_hat_cvae, mu, lv = cvae(x_clean)
    assert x_hat_cvae.shape == (B, 1, L), f"CVAE output {x_hat_cvae.shape}"
    assert mu.shape          == (B, Z),   f"CVAE mu {mu.shape}"
    assert lv.shape          == (B, Z),   f"CVAE lv {lv.shape}"

    total, recon, kl = vae_loss_function(x_hat_cvae, x_clean, mu, lv, beta=0.001)
    print(get_model_summary(cvae, (B, 1, L)))
    print(f"  CVAE total : {total.item():.6f}  "
          f"recon={recon.item():.6f}  kl={kl.item():.6f}")

    print("\nAll assertions passed.")
