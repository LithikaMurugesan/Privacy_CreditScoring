import torch
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# GRADIENT CLIPPING
# ═════════════════════════════════════════════════════════════════════════════

def clip_gradients(model, max_norm):
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


# ═════════════════════════════════════════════════════════════════════════════
# ADD DIFFERENTIAL PRIVACY NOISE
# ═════════════════════════════════════════════════════════════════════════════

def add_dp_noise(model, noise_multiplier, max_norm, batch_size):
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.randn_like(p.grad) * (noise_multiplier * max_norm / batch_size)
            p.grad.data.add_(noise)


# ═════════════════════════════════════════════════════════════════════════════
# EPSILON CALCULATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_epsilon(noise_multiplier, sample_rate, num_steps, delta=1e-5):
    if noise_multiplier <= 0:
        return float('inf')

    sigma = noise_multiplier

    rdp = (sample_rate ** 2 * num_steps) / (2 * sigma ** 2)

    if rdp == 0:
        return float('inf')

    epsilon = rdp + np.log(1 / delta) / (2 * rdp)

    return round(min(epsilon, 50.0), 3)
