"""
src/privacy/dp_manager.py
=========================
Unified Differential Privacy Manager — toggle between backends without
changing training code.

Backends:
  "opacus"  — Uses Meta's Opacus PrivacyEngine (recommended, accurate)
  "custom"  — Uses manual gradient clipping + Gaussian noise (legacy)
  "none"    — No DP applied (for centralized/FedAvg baselines)

Usage (in training loop):
    dm = DPManager(backend="opacus", noise_multiplier=1.1, max_grad_norm=1.0)

    # Before training (Opacus mode):
    model, optimizer, loader = dm.setup(model, optimizer, loader, n_samples)

    # During training (custom mode — call after loss.backward()):
    dm.apply_custom_dp(model, batch_size)

    # After training:
    epsilon = dm.get_epsilon(num_steps, sample_rate)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.privacy.dp_custom import clip_gradients, add_dp_noise, compute_epsilon

logger = logging.getLogger(__name__)


class DPManager:
    """
    Unified interface for Differential Privacy across backends.

    This allows the FL training code to be written once and switch
    between DP implementations by changing a single parameter.

    Parameters
    ----------
    backend         : str   — "opacus" | "custom" | "none"
    noise_multiplier: float — Gaussian noise scale σ
    max_grad_norm   : float — gradient clipping threshold C
    delta           : float — target δ in (ε,δ)-DP
    """

    def __init__(
        self,
        backend: str = "custom",
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
    ):
        assert backend in ("opacus", "custom", "none"), \
            f"Unknown backend '{backend}'. Choose: opacus | custom | none"

        self.backend          = backend
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm    = max_grad_norm
        self.delta            = delta
        self._opacus_wrapper  = None

        if backend == "opacus":
            from src.privacy.dp_opacus import OpacusPrivacyWrapper, OPACUS_AVAILABLE
            if not OPACUS_AVAILABLE:
                logger.warning(
                    "Opacus not available — falling back to custom DP. "
                    "Install with: pip install opacus>=1.4.0"
                )
                self.backend = "custom"
            else:
                self._opacus_wrapper = OpacusPrivacyWrapper(
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm,
                    delta=delta,
                )

    @property
    def is_active(self) -> bool:
        """Returns True if any DP is applied."""
        return self.backend != "none"

    def setup(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        n_samples: Optional[int] = None,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        """
        Prepare model, optimizer, and dataloader for DP training.

        For Opacus backend: attaches PrivacyEngine (must be called once before training).
        For custom/none backend: returns inputs unchanged.

        Parameters
        ----------
        model       : PyTorch model
        optimizer   : Optimizer
        data_loader : DataLoader
        n_samples   : int — dataset size (used for sample rate computation)

        Returns
        -------
        (model, optimizer, data_loader) — possibly wrapped by Opacus
        """
        if self.backend == "opacus" and self._opacus_wrapper is not None:
            logger.info("Setting up Opacus PrivacyEngine for this client.")
            model, optimizer, data_loader = self._opacus_wrapper.make_private(
                model, optimizer, data_loader
            )
        return model, optimizer, data_loader

    def apply_custom_dp(self, model: nn.Module, batch_size: int) -> float:
        """
        Apply custom (manual) DP after loss.backward().

        Call this AFTER loss.backward() and BEFORE optimizer.step()
        when backend == "custom".

        Parameters
        ----------
        model      : PyTorch model (gradients populated)
        batch_size : int — size of current mini-batch

        Returns
        -------
        float — pre-clip gradient norm (for monitoring)
        """
        if self.backend != "custom":
            return 0.0
        norm = clip_gradients(model, self.max_grad_norm)
        add_dp_noise(model, self.noise_multiplier, self.max_grad_norm, batch_size)
        return norm

    def get_epsilon(
        self,
        num_steps: int = 0,
        sample_rate: float = 0.0,
        delta: Optional[float] = None,
    ) -> float:
        """
        Get current privacy budget ε.

        For Opacus: reads from PrivacyEngine accountant.
        For custom:  computes using RDP approximation.
        For none:    returns 0.

        Parameters
        ----------
        num_steps   : int   — total training steps (custom backend)
        sample_rate : float — batch_size/N (custom backend)
        delta       : float — δ (default: self.delta)

        Returns
        -------
        float — current ε estimate
        """
        d = delta or self.delta
        if self.backend == "none":
            return 0.0
        elif self.backend == "opacus" and self._opacus_wrapper is not None:
            return self._opacus_wrapper.get_epsilon(delta=d)
        else:
            # custom backend — use RDP approximation
            if num_steps == 0 or sample_rate == 0:
                return 0.0
            return compute_epsilon(self.noise_multiplier, sample_rate, num_steps, d)

    def backend_label(self) -> str:
        """Human-readable backend description for UI display."""
        labels = {
            "opacus": "Opacus PrivacyEngine",
            "custom": "Custom DP (Gradient Clip + Gaussian Noise)",
            "none":   "No DP",
        }
        return labels.get(self.backend, self.backend)
