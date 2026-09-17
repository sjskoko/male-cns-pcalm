from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .connectome import LayeredConnectome


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-6)
    return value + torch.log(-torch.expm1(-value))


class SparseSignedProjection(nn.Module):
    """Sparse projection whose mask is fixed and whose known signs cannot flip."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        initial_values: torch.Tensor,
        sign: torch.Tensor,
        shape: tuple[int, int],
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = shape
        self.register_buffer("edge_index", edge_index.to(dtype=torch.long))
        self.register_buffer("sign", sign.to(dtype=torch.float32))
        initial_values = initial_values.to(dtype=torch.float32)
        known = self.sign != 0
        raw = initial_values.clone()
        raw[known] = _inverse_softplus(initial_values[known].abs())
        if trainable:
            self.raw_values = nn.Parameter(raw)
        else:
            self.register_buffer("raw_values", raw)
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if trainable else None

    def effective_values(self) -> torch.Tensor:
        known = self.sign != 0
        constrained = self.sign * F.softplus(self.raw_values)
        return torch.where(known, constrained, self.raw_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matrix = torch.sparse_coo_tensor(
            self.edge_index,
            self.effective_values(),
            size=(self.out_features, self.in_features),
            device=x.device,
            dtype=x.dtype,
            check_invariants=False,
        ).coalesce()
        output = torch.sparse.mm(matrix, x.transpose(0, 1)).transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output


class ConnectomePCNetwork(nn.Module):
    """Layered sparse connectome with a task-specific dense readout."""

    def __init__(
        self,
        connectome: LayeredConnectome,
        output_dim: int,
        *,
        activation: str = "tanh",
        trainable_edges: bool = True,
    ) -> None:
        super().__init__()
        connectome.validate()
        self.layer_names = tuple(
            [connectome.layers[0].source_name] + [layer.target_name for layer in connectome.layers]
        )
        self.projections = nn.ModuleList(
            [
                SparseSignedProjection(
                    layer.edge_index,
                    layer.initial_values,
                    layer.sign,
                    layer.shape,
                    trainable=trainable_edges,
                )
                for layer in connectome.layers
            ]
        )
        final_width = connectome.layer_sizes[-1]
        self.readout = nn.Linear(final_width, output_dim)
        nn.init.normal_(self.readout.weight, std=1.0 / math.sqrt(max(final_width, 1)))
        nn.init.zeros_(self.readout.bias)
        if activation not in {"linear", "relu", "tanh"}:
            raise ValueError("activation must be linear, relu, or tanh")
        self.activation = activation

    def phi(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation == "linear":
            return value
        if self.activation == "relu":
            return F.relu(value)
        return torch.tanh(value)

    def predict_hidden(self, x: torch.Tensor, states: list[torch.Tensor]) -> list[torch.Tensor]:
        predictions: list[torch.Tensor] = []
        previous = x
        for index, projection in enumerate(self.projections):
            prediction = self.phi(projection(previous))
            predictions.append(prediction)
            if index < len(self.projections) - 1:
                previous = states[index]
        return predictions

    def initial_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        states: list[torch.Tensor] = []
        previous = x
        for projection in self.projections:
            previous = self.phi(projection(previous))
            states.append(previous)
        return states

    def logits_from_states(self, states: list[torch.Tensor]) -> torch.Tensor:
        return self.readout(states[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_from_states(self.initial_states(x))
