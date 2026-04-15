"""
src/privacy/dp_opacus.py
========================
Opacus-based Differential Privacy integration.

Opacus (by Meta/Facebook) is the reference implementation of DP-SGD for PyTorch.
It replaces manual gradient clipping + noise injection with a proper per-sample
gradient computation engine, which is mathematically more accurate.

Key differences from our custom DP:
  - Opacus computes per-SAMPLE gradients (not per-batch), then clips and noises them.
    This is the correct DP-SGD algorithm.
  - Our custom DP clips the aggregated batch gradient — a valid but less precise approach.
  - Opacus uses a RDP accountant that is numerically precise (our custom one approximates).
  - Opacus wraps the model, optimizer, and dataloader into private versions.

Usage:
    dp = OpacusPrivacyWrapper(noise_multiplier=1.1, max_grad_norm=1.0)
    private_model, private_optimizer, private_loader = dp.make_private(
        model, optimizer, data_loader, sample_rate=64/1000
    )
    # Train normally — privacy is applied automatically
    epsilon = dp.get_epsilon(delta=1e-5)

NOTE: The model must NOT use BatchNorm (use GroupNorm instead).
      See src/models/model.py for the Opacus-compatible CreditNet.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Try importing Opacus — fail gracefully so the app still runs without it
# ─────────────────────────────────────────────────────────────────────────────
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = None
    ModuleValidator = None


class OpacusPrivacyWrapper:
    """
    Wraps Opacus PrivacyEngine for per-client DP training in FL.

    Each bank client gets its own OpacusPrivacyWrapper instance so that
    epsilon is tracked independently per client per round.

    Parameters
    ----------
    noise_multiplier : float  — Gaussian noise scale (σ). Higher → more private.
    max_grad_norm    : float  — per-sample gradient clipping norm (C).
    delta            : float  — target δ in (ε,δ)-DP.
    """

    def __init__(
        self,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
    ):
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is not installed. Run: pip install opacus>=1.4.0\n"
                "Or use dp_backend='custom' in DPManager."
            )
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm    = max_grad_norm
        self.delta            = delta
        self._privacy_engine: Optional[PrivacyEngine] = None

    def validate_model(self, model: nn.Module) -> nn.Module:
        """
        Check and fix the model for Opacus compatibility.

        Opacus requires:
        - No BatchNorm layers (use GroupNorm)
        - No custom backward hooks that interfere with grad computation

        Returns the model (possibly modified to be compatible).
        """
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            model = ModuleValidator.fix(model)
        return model

    def make_private(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        sample_rate: Optional[float] = None,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        """
        Attach Opacus PrivacyEngine to model, optimizer, and dataloader.

        After calling this, training proceeds normally — Opacus hooks into
        the backward pass to apply per-sample clipping and Gaussian noise.

        Parameters
        ----------
        model       : PyTorch model (GroupNorm required, no BatchNorm)
        optimizer   : Adam/SGD optimizer
        data_loader : DataLoader for training data
        sample_rate : float  — batch_size / N (computed automatically if None)

        Returns
        -------
        (private_model, private_optimizer, private_loader)
        — Drop-in replacements for the originals.
        """
        model = self.validate_model(model)

        self._privacy_engine = PrivacyEngine()
        private_model, private_optimizer, private_loader = (
            self._privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
        )
        return private_model, private_optimizer, private_loader

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Get the current privacy budget ε consumed so far.

        Parameters
        ----------
        delta : float  — target δ (defaults to self.delta)

        Returns
        -------
        float — cumulative ε spent
        """
        if self._privacy_engine is None:
            return 0.0
        d = delta or self.delta
        try:
            return self._privacy_engine.get_epsilon(delta=d)
        except Exception:
            return 0.0
