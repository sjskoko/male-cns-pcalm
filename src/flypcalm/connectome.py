from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class LayerSpec:
    """One directed sparse projection from layer ``index`` to ``index + 1``.

    ``edge_index[0]`` indexes target neurons and ``edge_index[1]`` indexes
    source neurons. ``sign`` is -1 (inhibitory), +1 (excitatory), or 0
    (unconstrained/unknown) for every edge.
    """

    source_ids: torch.Tensor
    target_ids: torch.Tensor
    edge_index: torch.Tensor
    initial_values: torch.Tensor
    sign: torch.Tensor
    source_name: str
    target_name: str

    def validate(self) -> None:
        if self.source_ids.ndim != 1 or self.target_ids.ndim != 1:
            raise ValueError("source_ids and target_ids must be rank-1 tensors")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, number_of_edges]")
        edge_count = self.edge_index.shape[1]
        if self.initial_values.shape != (edge_count,) or self.sign.shape != (edge_count,):
            raise ValueError("initial_values and sign must have one entry per edge")
        if edge_count:
            if int(self.edge_index[0].min()) < 0 or int(self.edge_index[0].max()) >= len(
                self.target_ids
            ):
                raise ValueError("target edge index is out of range")
            if int(self.edge_index[1].min()) < 0 or int(self.edge_index[1].max()) >= len(
                self.source_ids
            ):
                raise ValueError("source edge index is out of range")
        allowed = torch.tensor([-1, 0, 1], dtype=self.sign.dtype, device=self.sign.device)
        if not torch.isin(self.sign, allowed).all():
            raise ValueError("sign entries must be -1, 0, or 1")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.target_ids), len(self.source_ids)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


@dataclass(frozen=True)
class LayeredConnectome:
    """A feed-forward projection of a connectome for layer-local inference."""

    layers: tuple[LayerSpec, ...]
    metadata: dict[str, Any]

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("at least one connectome projection is required")
        for layer in self.layers:
            layer.validate()
        for left, right in zip(self.layers, self.layers[1:]):
            if not torch.equal(left.target_ids.cpu(), right.source_ids.cpu()):
                raise ValueError("adjacent layer node IDs do not align")

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return (len(self.layers[0].source_ids),) + tuple(
            len(layer.target_ids) for layer in self.layers
        )

    @property
    def num_edges(self) -> int:
        return sum(layer.num_edges for layer in self.layers)

    def to(self, device: torch.device | str) -> LayeredConnectome:
        moved = tuple(
            replace(
                layer,
                source_ids=layer.source_ids.to(device),
                target_ids=layer.target_ids.to(device),
                edge_index=layer.edge_index.to(device),
                initial_values=layer.initial_values.to(device),
                sign=layer.sign.to(device),
            )
            for layer in self.layers
        )
        return LayeredConnectome(moved, dict(self.metadata))

    def save(self, path: str | Path) -> None:
        self.validate()
        payload = {
            "format_version": 1,
            "metadata": self.metadata,
            "layers": [
                {
                    "source_ids": layer.source_ids.cpu(),
                    "target_ids": layer.target_ids.cpu(),
                    "edge_index": layer.edge_index.cpu(),
                    "initial_values": layer.initial_values.cpu(),
                    "sign": layer.sign.cpu(),
                    "source_name": layer.source_name,
                    "target_name": layer.target_name,
                }
                for layer in self.layers
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> LayeredConnectome:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported connectome format")
        graph = cls(
            layers=tuple(LayerSpec(**raw) for raw in payload["layers"]),
            metadata=payload.get("metadata", {}),
        )
        graph.validate()
        return graph
