from __future__ import annotations

import torch

from flypcalm.model import ConnectomePCNetwork
from flypcalm.pcalm import InferenceConfig, infer, local_parameter_backward, train_step
from flypcalm.synthetic import random_layered_connectome


def case():
    torch.manual_seed(0)
    graph = random_layered_connectome([4, 6, 5], density=0.6, seed=2)
    model = ConnectomePCNetwork(graph, output_dim=3)
    x = torch.randn(7, 4)
    target = torch.nn.functional.one_hot(torch.arange(7) % 3, 3).float()
    return model, x, target


def test_pcalm_duals_update():
    model, x, target = case()
    result = infer(
        model,
        x,
        target,
        InferenceConfig(method="pcalm", budget=3, state_lr=0.03, alpha=0.5),
    )
    assert len(result.duals) == 2
    assert any(float(dual.norm()) > 0 for dual in result.duals)


def test_alpha_zero_matches_pc_states_and_local_gradients():
    model_pc, x, target = case()
    model_alm, _, _ = case()
    pc = infer(model_pc, x, target, InferenceConfig(method="pc", budget=3, state_lr=0.03))
    alm = infer(
        model_alm,
        x,
        target,
        InferenceConfig(method="pcalm", budget=3, inner_steps=1, state_lr=0.03, alpha=0.0),
    )
    for left, right in zip(pc.states, alm.states):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)
    local_parameter_backward(model_pc, x, target, pc.states, pc.duals, rho=1.0)
    local_parameter_backward(model_alm, x, target, alm.states, alm.duals, rho=1.0)
    for left, right in zip(model_pc.parameters(), model_alm.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-6)


def test_one_pcalm_step_changes_parameters_and_is_finite():
    model, x, target = case()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = train_step(
        model,
        optimizer,
        x,
        target,
        InferenceConfig(method="pcalm", budget=2, state_lr=0.02, alpha=0.25),
    )
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_signed_edges_cannot_flip():
    model, _, _ = case()
    projection = model.projections[0]
    with torch.no_grad():
        projection.raw_values.copy_(torch.linspace(-100, 100, projection.raw_values.numel()))
    values = projection.effective_values()
    known = projection.sign != 0
    assert torch.equal(torch.sign(values[known]), projection.sign[known])
