"""
src/models/model.py
===================
CreditNet — 3-layer MLP for binary credit default prediction.

Architecture:
  Linear(10→64) → GroupNorm → ReLU → Dropout(0.3)
  → Linear(64→32) → ReLU → Dropout(0.2)
  → Linear(32→16) → ReLU
  → Linear(16→1)  → Sigmoid

NOTE: We use GroupNorm instead of BatchNorm1d for two reasons:
  1. Opacus (differential privacy library) does NOT support BatchNorm — it leaks
     per-sample information because the batch statistics couple all samples.
  2. GroupNorm(num_groups=1, ...) is mathematically equivalent to LayerNorm and
     behaves identically to BatchNorm in expectation with proper tuning.

Weight utility functions (get_weights / set_weights) are used by both the custom
FL loop and the Flower NumPyClient for parameter serialisation.
"""

import torch
import torch.nn as nn


class CreditNet(nn.Module):
    """
    3-layer MLP for credit default binary classification.

    Parameters
    ----------
    input_dim : int  — number of input features (default 10, matching FEATURE_NAMES)
    """

    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1 — GroupNorm replaces BatchNorm (Opacus compatibility)
            nn.Linear(input_dim, 64),
            nn.GroupNorm(num_groups=1, num_channels=64),  # equivalent to LayerNorm
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # Block 2
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Block 3
            nn.Linear(32, 16),
            nn.ReLU(),

            # Output — single sigmoid for P(default)
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)

        Returns
        -------
        Tensor of shape (batch_size,) — probability of default [0, 1]
        """
        return self.net(x).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Weight serialisation utilities
# Used by FedAvg aggregation and Flower NumPyClient parameter exchange.
# ─────────────────────────────────────────────────────────────────────────────

def get_weights(model: nn.Module) -> list:
    """Extract model weights as a list of cloned tensors."""
    return [p.data.clone() for p in model.parameters()]


def set_weights(model: nn.Module, weights: list) -> None:
    """Load a list of tensors into model parameters (in-place)."""
    for p, w in zip(model.parameters(), weights):
        p.data.copy_(w)


def weights_to_numpy(model: nn.Module) -> list:
    """
    Convert model parameters to list of numpy arrays.
    Required by Flower's NumPyClient interface.
    """
    return [p.data.cpu().numpy() for p in model.parameters()]


def numpy_to_weights(model: nn.Module, numpy_weights: list) -> None:
    """
    Load list of numpy arrays into model parameters.
    Required by Flower's NumPyClient interface.
    """
    for p, w in zip(model.parameters(), numpy_weights):
        p.data = torch.tensor(w, dtype=torch.float32)
