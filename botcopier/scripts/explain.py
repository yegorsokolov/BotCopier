from __future__ import annotations

import torch


def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: torch.Tensor | None = None,
    *,
    steps: int = 50,
) -> torch.Tensor:
    """Compute Integrated Gradients for ``inputs`` on ``model``.

    Parameters
    ----------
    model:
        The model for which attributions are computed.
    inputs:
        Input tensor for which to compute feature attributions. The tensor is
        assumed to require gradients.
    baseline:
        Baseline tensor from which the integration path starts. Defaults to a
        tensor of zeros with the same shape as ``inputs``.
    steps:
        Number of interpolation steps used to approximate the integral.
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    baseline = baseline.to(inputs.device)
    scaled_diff = inputs - baseline
    grads = torch.zeros_like(inputs)
    for i in range(1, steps + 1):
        scaled = baseline + (float(i) / steps) * scaled_diff
        scaled.requires_grad_(True)
        model.zero_grad()
        out = model(scaled)
        out.sum().backward()
        grads += scaled.grad.detach()
    avg_grads = grads / steps
    return scaled_diff * avg_grads
