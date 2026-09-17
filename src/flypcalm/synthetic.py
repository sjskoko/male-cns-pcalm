from __future__ import annotations

import random
from itertools import pairwise

import torch

from .connectome import LayeredConnectome, LayerSpec


def _source_signs(layer: LayerSpec) -> torch.Tensor:
    """Infer one sign per source where all known outgoing signs agree."""
    source_count = layer.shape[1]
    source_sign = torch.zeros(source_count, dtype=torch.int8)
    for source in range(source_count):
        observed = torch.unique(layer.sign[layer.edge_index[1] == source])
        observed = observed[observed != 0]
        if len(observed) == 1:
            source_sign[source] = observed[0]
    return source_sign


def random_layered_connectome(
    layer_sizes: list[int] | tuple[int, ...],
    *,
    density: float = 0.25,
    inhibitory_fraction: float = 0.2,
    seed: int = 0,
) -> LayeredConnectome:
    if len(layer_sizes) < 2 or any(width < 1 for width in layer_sizes):
        raise ValueError("layer_sizes must contain at least two positive widths")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    generator = torch.Generator().manual_seed(seed)
    layers: list[LayerSpec] = []
    node_offset = 0
    source_ids = torch.arange(layer_sizes[0], dtype=torch.long)
    node_offset += layer_sizes[0]

    for layer_index, (source_width, target_width) in enumerate(pairwise(layer_sizes)):
        mask = torch.rand((target_width, source_width), generator=generator) < density
        # Avoid silent neurons on either side of a projection.
        for row in range(target_width):
            if not mask[row].any():
                mask[row, torch.randint(source_width, (1,), generator=generator)] = True
        for column in range(source_width):
            if not mask[:, column].any():
                mask[torch.randint(target_width, (1,), generator=generator), column] = True
        edge_index = mask.nonzero(as_tuple=False).transpose(0, 1).contiguous()
        edge_count = edge_index.shape[1]
        magnitudes = 0.25 + 0.25 * torch.rand(edge_count, generator=generator)
        source_sign = torch.where(
            torch.rand(source_width, generator=generator) < inhibitory_fraction,
            -torch.ones(source_width),
            torch.ones(source_width),
        )
        sign = source_sign[edge_index[1]].to(torch.int8)
        values = sign.to(torch.float32) * magnitudes / max(source_width * density, 1) ** 0.5
        target_ids = torch.arange(node_offset, node_offset + target_width, dtype=torch.long)
        layers.append(
            LayerSpec(
                source_ids=source_ids,
                target_ids=target_ids,
                edge_index=edge_index,
                initial_values=values,
                sign=sign,
                source_name=f"layer_{layer_index}",
                target_name=f"layer_{layer_index + 1}",
            )
        )
        source_ids = target_ids
        node_offset += target_width

    graph = LayeredConnectome(
        tuple(layers),
        {
            "kind": "synthetic",
            "seed": seed,
            "density": density,
            "inhibitory_fraction": inhibitory_fraction,
        },
    )
    graph.validate()
    return graph


def visual_proxy_connectome(
    layer_sizes: list[int] | tuple[int, ...],
    *,
    density: float = 0.25,
    inhibitory_fraction: float = 0.2,
    receptive_width: float = 0.32,
    seed: int = 0,
) -> LayeredConnectome:
    """Create a retinotopic sparse graph for virtual visual-motor experiments.

    It is intentionally a synthetic proxy, not a reconstruction of MaleCNS.
    Nearby visual-field positions connect more often than distant positions so
    rewiring controls destroy a known structural prior.
    """
    if len(layer_sizes) < 2 or any(width < 1 for width in layer_sizes):
        raise ValueError("layer_sizes must contain at least two positive widths")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    if receptive_width <= 0:
        raise ValueError("receptive_width must be positive")

    generator = torch.Generator().manual_seed(seed)
    positions = [torch.linspace(-1.0, 1.0, width) for width in layer_sizes]
    layers: list[LayerSpec] = []
    node_offset = 0
    source_ids = torch.arange(layer_sizes[0], dtype=torch.long)
    node_offset += layer_sizes[0]

    for layer_index, (source_width, target_width) in enumerate(pairwise(layer_sizes)):
        source_positions = positions[layer_index]
        target_positions = positions[layer_index + 1]
        fan_in = min(source_width, max(1, round(density * source_width)))
        pairs: list[tuple[int, int]] = []
        for target, target_position in enumerate(target_positions):
            distance = source_positions - target_position
            probability = torch.exp(-0.5 * (distance / receptive_width) ** 2) + 1e-3
            selected = torch.multinomial(
                probability, fan_in, replacement=False, generator=generator
            )
            pairs.extend((target, int(source)) for source in selected)

        covered_sources = {source for _, source in pairs}
        for source in range(source_width):
            if source not in covered_sources:
                nearest_target = int(
                    torch.argmin((target_positions - source_positions[source]).abs())
                )
                pairs.append((nearest_target, source))

        edge_index = torch.tensor(pairs, dtype=torch.long).transpose(0, 1).contiguous()
        edge_count = edge_index.shape[1]
        source_sign = torch.where(
            torch.rand(source_width, generator=generator) < inhibitory_fraction,
            -torch.ones(source_width),
            torch.ones(source_width),
        )
        sign = source_sign[edge_index[1]].to(torch.int8)
        magnitudes = 0.25 + 0.25 * torch.rand(edge_count, generator=generator)
        values = sign.to(torch.float32) * magnitudes / fan_in**0.5
        target_ids = torch.arange(node_offset, node_offset + target_width, dtype=torch.long)
        layers.append(
            LayerSpec(
                source_ids=source_ids,
                target_ids=target_ids,
                edge_index=edge_index,
                initial_values=values,
                sign=sign,
                source_name=f"visual_proxy_{layer_index}",
                target_name=f"visual_proxy_{layer_index + 1}",
            )
        )
        source_ids = target_ids
        node_offset += target_width

    graph = LayeredConnectome(
        tuple(layers),
        {
            "kind": "retinotopic_visual_proxy",
            "seed": seed,
            "density": density,
            "receptive_width": receptive_width,
            "input_positions": positions[0].tolist(),
            "biological_claim": False,
        },
    )
    graph.validate()
    return graph


def degree_preserving_rewire(
    connectome: LayeredConnectome,
    *,
    swaps_per_edge: float = 2.0,
    seed: int = 0,
) -> LayeredConnectome:
    """Directed bipartite double-edge swaps, independently per projection."""
    rng = random.Random(seed)
    rewired_layers: list[LayerSpec] = []
    for layer in connectome.layers:
        pairs = [tuple(pair) for pair in layer.edge_index.transpose(0, 1).tolist()]
        pair_set = set(pairs)
        attempts = int(max(0, swaps_per_edge) * len(pairs))
        if len(pairs) >= 2:
            for _ in range(attempts):
                first, second = rng.sample(range(len(pairs)), 2)
                target_a, source_a = pairs[first]
                target_b, source_b = pairs[second]
                candidate_a = (target_a, source_b)
                candidate_b = (target_b, source_a)
                if (
                    target_a == target_b
                    or source_a == source_b
                    or candidate_a in pair_set
                    or candidate_b in pair_set
                ):
                    continue
                pair_set.remove(pairs[first])
                pair_set.remove(pairs[second])
                pairs[first], pairs[second] = candidate_a, candidate_b
                pair_set.add(candidate_a)
                pair_set.add(candidate_b)
        edge_index = torch.tensor(pairs, dtype=torch.long).transpose(0, 1).contiguous()
        sign = _source_signs(layer)[edge_index[1]]
        magnitudes = layer.initial_values.abs()
        initial_values = torch.where(sign == 0, magnitudes, sign * magnitudes)
        rewired_layers.append(
            LayerSpec(
                source_ids=layer.source_ids.clone(),
                target_ids=layer.target_ids.clone(),
                edge_index=edge_index,
                initial_values=initial_values,
                sign=sign,
                source_name=layer.source_name,
                target_name=layer.target_name,
            )
        )
    graph = LayeredConnectome(
        tuple(rewired_layers),
        {**connectome.metadata, "rewired": "degree_preserving", "rewire_seed": seed},
    )
    graph.validate()
    return graph


def randomize_edges(
    connectome: LayeredConnectome,
    *,
    seed: int = 0,
) -> LayeredConnectome:
    """Randomize masks while preserving each layer size and edge count."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    randomized_layers: list[LayerSpec] = []
    for layer in connectome.layers:
        target_count, source_count = layer.shape
        capacity = target_count * source_count
        if layer.num_edges > capacity:
            raise ValueError("edge count exceeds bipartite projection capacity")
        flat = rng.sample(range(capacity), layer.num_edges)
        edge_index = torch.tensor(
            [[index // source_count for index in flat], [index % source_count for index in flat]],
            dtype=torch.long,
        )

        # Recover a source-neuron sign where the original edges agree. A source
        # with conflicting, missing, or unknown observations stays unconstrained.
        source_sign = _source_signs(layer)
        sign = source_sign[edge_index[1]]
        magnitudes = layer.initial_values.abs()
        if layer.num_edges:
            magnitudes = magnitudes[torch.randperm(layer.num_edges, generator=generator)]
        initial_values = torch.where(sign == 0, magnitudes, sign * magnitudes)
        randomized_layers.append(
            LayerSpec(
                source_ids=layer.source_ids.clone(),
                target_ids=layer.target_ids.clone(),
                edge_index=edge_index,
                initial_values=initial_values,
                sign=sign,
                source_name=layer.source_name,
                target_name=layer.target_name,
            )
        )
    graph = LayeredConnectome(
        tuple(randomized_layers),
        {**connectome.metadata, "rewired": "random_edge_count", "rewire_seed": seed},
    )
    graph.validate()
    return graph
