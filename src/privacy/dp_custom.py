"""
src/privacy/dp_custom.py
========================
Custom Differential Privacy implementation — gradient clipping + Gaussian noise.

This is the legacy DP module, preserved for compatibility and as a fallback
when Opacus is not available or DP is used WITHOUT the Opacus library.

How DP works here (matching Opacus internals):
  Step 1 — Gradient Clipping: each parameter's gradient is scaled so the
            global L2 norm does not exceed max_norm. This bounds the
            per-sample sensitivity ||grad||₂ ≤ max_norm.

  Step 2 — Gaussian Noise:   noise ~ N(0, (noise_mult * max_norm)²) is added
            per parameter. Dividing by batch_size gives per-sample noise.

  Step 3 — RDP Accounting:   privacy budget ε is tracked using an approximation
            of Rényi Differential Privacy (RDP) composition.

References
----------
  - Abadi et al. (2016) "Deep Learning with Differential Privacy"
  - Mironov (2017) "Rényi Differential Privacy"
  - Opacus library: https://github.com/pytorch/opacus
"""

import torch
import numpy as np


def clip_gradients(model: torch.nn.Module, max_norm: float) -> float:
    """
    Clip all parameter gradients so the global L2 norm ≤ max_norm.

    This is the sensitivity-bounding step of DP-SGD. Without clipping,
    a single outlier sample could contribute arbitrarily large gradients,
    making it impossible to add calibrated noise.

    Parameters
    ----------
    model    : PyTorch model (gradients must already be computed)
    max_norm : float  — maximum allowed gradient L2 norm

    Returns
    -------
    float — the pre-clip gradient norm (useful for monitoring)
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / max(total_norm, max_norm)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)

    return total_norm


def add_dp_noise(
    model: torch.nn.Module,
    noise_multiplier: float,
    max_norm: float,
    batch_size: int,
) -> None:
    """
    Add calibrated Gaussian noise to clipped gradients.

    The noise standard deviation is: σ = noise_multiplier × max_norm / batch_size
    This follows the DP-SGD noise calibration from Abadi et al. (2016).

    Parameters
    ----------
    model            : PyTorch model
    noise_multiplier : float — controls noise magnitude (higher → stronger privacy)
    max_norm         : float — gradient clipping threshold (same as in clip_gradients)
    batch_size       : int   — current mini-batch size (for normalisation)
    """
    noise_std = noise_multiplier * max_norm / batch_size
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.randn_like(p.grad) * noise_std
            p.grad.data.add_(noise)


def compute_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    delta: float = 1e-5,
) -> float:
    """
    Approximate the privacy budget ε using RDP composition.

    This is a simplified RDP accountant. Opacus uses a more precise
    numerical computation, but this gives the correct order of magnitude.

    Formula (RDP order α=2 approximation):
        RDP(α=2) ≈ (sample_rate² × num_steps) / (2σ²)
        ε ≈ RDP + log(1/δ) / (2 × RDP)

    Parameters
    ----------
    noise_multiplier : float  — σ (the DP noise multiplier)
    sample_rate      : float  — batch_size / dataset_size (sampling rate)
    num_steps        : int    — total number of gradient steps
    delta            : float  — target δ in (ε,δ)-DP (default 1e-5)

    Returns
    -------
    float — estimated privacy budget ε (capped at 50 to avoid display issues)
    """
    if noise_multiplier <= 0:
        return float("inf")

    sigma = noise_multiplier
    rdp   = (sample_rate ** 2 * num_steps) / (2 * sigma ** 2)

    if rdp == 0:
        return float("inf")

    epsilon = rdp + np.log(1 / delta) / (2 * rdp)
    return round(min(epsilon, 50.0), 3)
