from __future__ import annotations

import math

from flypcalm.metrics import gradient_alignment
from flypcalm.model import ConnectomePCNetwork
from flypcalm.pcalm import InferenceConfig
from flypcalm.synthetic import visual_proxy_connectome
from flypcalm.tasks import make_visual_motor_dataset


def test_gradient_alignment_is_exact_for_bp_and_finite_for_pcalm():
    graph = visual_proxy_connectome([8, 10, 6], density=0.5, seed=0)
    dataset = make_visual_motor_dataset(8, 16, seed=1)
    model = ConnectomePCNetwork(graph, output_dim=4)
    bp = gradient_alignment(model, dataset.inputs, dataset.targets, InferenceConfig(method="bp"))
    assert math.isclose(bp["global_cosine"], 1.0, abs_tol=1e-6)

    pcalm = gradient_alignment(
        model,
        dataset.inputs,
        dataset.targets,
        InferenceConfig(method="pcalm", budget=2, state_lr=0.02, alpha=0.25),
    )
    assert math.isfinite(pcalm["global_cosine"])
    assert len(pcalm["per_layer_cosine"]) == 2
