from __future__ import annotations

from collections import Counter

import pandas as pd

from flypcalm.connectome import LayeredConnectome
from flypcalm.data import build_layered_connectome
from flypcalm.synthetic import degree_preserving_rewire, random_layered_connectome, randomize_edges


def test_prepare_projects_only_adjacent_forward_edges(tmp_path):
    edges = pd.DataFrame(
        {
            "body_pre": [1, 1, 2, 3, 4],
            "body_post": [3, 5, 4, 5, 2],
            "weight": [10, 10, 10, 10, 10],
        }
    )
    assignments = pd.DataFrame(
        {
            "body_id": [1, 2, 3, 4, 5],
            "layer": [0, 0, 1, 1, 2],
            "module": ["input", "input", "middle", "middle", "output"],
            "sign": [1, 1, -1, 1, 0],
        }
    )
    edges_path = tmp_path / "edges.feather"
    assignments_path = tmp_path / "assignments.csv"
    edges.to_feather(edges_path)
    assignments.to_csv(assignments_path, index=False)
    graph, report = build_layered_connectome(edges_path, assignments_path)
    assert graph.num_edges == 3
    assert report["dropped_skip_forward_edges"] == 1
    assert report["dropped_feedback_edges"] == 1

    output = tmp_path / "graph.pt"
    graph.save(output)
    restored = LayeredConnectome.load(output)
    assert restored.layer_sizes == (2, 2, 1)


def test_degree_preserving_rewire_preserves_bipartite_degrees():
    graph = random_layered_connectome([10, 12, 8], density=0.4, seed=0)
    rewired = degree_preserving_rewire(graph, swaps_per_edge=10, seed=1)
    for original, changed in zip(graph.layers, rewired.layers):
        original_rows = Counter(original.edge_index[0].tolist())
        original_cols = Counter(original.edge_index[1].tolist())
        assert original_rows == Counter(changed.edge_index[0].tolist())
        assert original_cols == Counter(changed.edge_index[1].tolist())
        assert (
            len(set(map(tuple, changed.edge_index.transpose(0, 1).tolist()))) == changed.num_edges
        )


def test_random_control_preserves_layer_sizes_and_edge_counts():
    graph = random_layered_connectome([10, 12, 8], density=0.4, seed=0)
    randomized = randomize_edges(graph, seed=5)
    assert randomized.layer_sizes == graph.layer_sizes
    assert [layer.num_edges for layer in randomized.layers] == [
        layer.num_edges for layer in graph.layers
    ]
