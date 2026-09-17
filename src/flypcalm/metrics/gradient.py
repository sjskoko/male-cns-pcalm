from __future__ import annotations

from typing import Any

import torch

from ..model import ConnectomePCNetwork
from ..pcalm import InferenceConfig, infer, local_parameter_backward, target_loss


def _projection_gradients(model: ConnectomePCNetwork) -> list[torch.Tensor]:
    vectors: list[torch.Tensor] = []
    for projection in model.projections:
        chunks = [
            parameter.grad.detach().flatten().clone()
            for parameter in projection.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        vectors.append(torch.cat(chunks) if chunks else torch.empty(0))
    return vectors


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if left.numel() == 0 or right.numel() == 0 or float(denominator) <= 1e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def gradient_alignment(
    model: ConnectomePCNetwork,
    x: torch.Tensor,
    target: torch.Tensor,
    config: InferenceConfig,
) -> dict[str, Any]:
    """Compare projection updates from a method with end-to-end BP.

    The readout is excluded because its PC/PC-ALM gradient is directly driven
    by the supervised loss and would inflate alignment.
    """
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    bp_loss = target_loss(model(x), target)
    bp_loss.backward()
    bp_vectors = _projection_gradients(model)

    if config.method == "bp":
        method_vectors = [vector.clone() for vector in bp_vectors]
    else:
        result = infer(model, x, target, config)
        local_parameter_backward(model, x, target, result.states, result.duals, config.rho)
        method_vectors = _projection_gradients(model)

    per_layer = [
        _cosine(bp_vector, method_vector)
        for bp_vector, method_vector in zip(bp_vectors, method_vectors)
    ]
    bp_all = torch.cat(bp_vectors)
    method_all = torch.cat(method_vectors)
    result = {
        "global_cosine": _cosine(bp_all, method_all),
        "per_layer_cosine": per_layer,
        "bp_norm": float(bp_all.norm()),
        "method_norm": float(method_all.norm()),
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return result
