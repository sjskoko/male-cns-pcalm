from __future__ import annotations

import pandas as pd

from flypcalm.malecns import select_visual_descending_circuit


def test_select_visual_descending_circuit_is_balanced_and_disjoint(tmp_path):
    visual = [1, 2, 3, 4]
    central_1 = [10, 11, 12]
    central_2 = [20, 21, 22]
    descending = [30, 31]
    body_ids = visual + central_1 + central_2 + descending
    annotations = pd.DataFrame(
        {
            "bodyId": body_ids,
            "superclass": (
                ["visual_projection"] * 4
                + ["cb_intrinsic"] * 6
                + ["descending_neuron"] * 2
            ),
            "somaSide": ["L", "L", "R", "R"] + ["L"] * 8,
            "type": [f"type-{body_id}" for body_id in body_ids],
            "instance": [f"instance-{body_id}" for body_id in body_ids],
            "status": ["Traced"] * len(body_ids),
        }
    )
    neurotransmitters = pd.DataFrame(
        {
            "body": body_ids,
            "consensus_nt": ["acetylcholine"] * len(body_ids),
            "predicted_nt_confidence": [0.9] * len(body_ids),
        }
    )
    edges = pd.DataFrame(
        {
            "body_pre": [1, 2, 3, 4, 10, 11, 12, 20, 21, 22],
            "body_post": [10, 11, 12, 10, 20, 21, 22, 30, 31, 31],
            "weight": [10, 11, 12, 9, 20, 21, 22, 30, 31, 32],
        }
    )
    annotations_path = tmp_path / "annotations.feather"
    neurotransmitters_path = tmp_path / "neurotransmitters.feather"
    weights_path = tmp_path / "weights.feather"
    annotations.to_feather(annotations_path)
    neurotransmitters.to_feather(neurotransmitters_path)
    edges.to_feather(weights_path)

    assignments, report = select_visual_descending_circuit(
        annotations_path,
        neurotransmitters_path,
        weights_path,
        layer_sizes=(4, 3, 3, 2),
        min_weight=5,
    )

    assert not assignments["body_id"].duplicated().any()
    assert report["selected_layer_sizes"] == [4, 3, 3, 2]
    assert report["input_side_counts"] == {"L": 2, "R": 2}
    assert set(assignments["sign"]) == {1}
