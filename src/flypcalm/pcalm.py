from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .model import ConnectomePCNetwork

Method = Literal["bp", "pc", "pcalm"]


@dataclass(frozen=True)
class InferenceConfig:
    method: Method = "pcalm"
    budget: int = 8
    inner_steps: int = 1
    state_lr: float = 0.1
    rho: float = 1.0
    alpha: float = 1.0
    weight_credit_timing: Literal["pre_dual", "post_dual"] = "pre_dual"

    def validate(self) -> None:
        if self.method not in {"bp", "pc", "pcalm"}:
            raise ValueError(f"unknown method: {self.method}")
        if self.budget < 1 or self.inner_steps < 1:
            raise ValueError("budget and inner_steps must be positive")
        if self.state_lr <= 0 or self.rho <= 0:
            raise ValueError("state_lr and rho must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.weight_credit_timing not in {"pre_dual", "post_dual"}:
            raise ValueError("weight_credit_timing must be pre_dual or post_dual")


@dataclass
class InferenceResult:
    states: list[torch.Tensor]
    duals: list[torch.Tensor]
    residual_norms: list[float]


def target_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample half squared error, matching the PC-ALM reference objective."""
    if logits.shape != target.shape:
        raise ValueError(f"target shape {target.shape} must match logits shape {logits.shape}")
    return 0.5 * ((logits - target) ** 2).sum(dim=-1).mean()


def constraint_residuals(
    model: ConnectomePCNetwork,
    x: torch.Tensor,
    states: list[torch.Tensor],
) -> list[torch.Tensor]:
    predictions = model.predict_hidden(x, states)
    return [state - prediction for state, prediction in zip(states, predictions)]


def augmented_energy(
    model: ConnectomePCNetwork,
    x: torch.Tensor,
    target: torch.Tensor,
    states: list[torch.Tensor],
    duals: list[torch.Tensor],
    rho: float,
) -> torch.Tensor:
    total = target_loss(model.logits_from_states(states), target)
    for residual, dual in zip(constraint_residuals(model, x, states), duals):
        shifted = residual + dual / rho
        total = total + 0.5 * rho * (shifted**2).sum(dim=-1).mean()
    return total


def _leaf_states(states: list[torch.Tensor]) -> list[torch.Tensor]:
    return [state.detach().requires_grad_(True) for state in states]


def _settle(
    model: ConnectomePCNetwork,
    x: torch.Tensor,
    target: torch.Tensor,
    states: list[torch.Tensor],
    duals: list[torch.Tensor],
    *,
    rho: float,
    state_lr: float,
    steps: int,
) -> list[torch.Tensor]:
    current = _leaf_states(states)
    for _ in range(steps):
        energy = augmented_energy(model, x, target, current, duals, rho)
        gradients = torch.autograd.grad(energy, current, create_graph=False)
        # Energy is a batch mean. Multiplication by B recovers the per-sample
        # activity step used by the PC-ALM reference implementation.
        effective_lr = state_lr * x.shape[0]
        current = _leaf_states(
            [state - effective_lr * gradient for state, gradient in zip(current, gradients)]
        )
    return current


def infer(
    model: ConnectomePCNetwork,
    x: torch.Tensor,
    target: torch.Tensor,
    config: InferenceConfig,
) -> InferenceResult:
    config.validate()
    if config.method == "bp":
        raise ValueError("bp does not run lifted-state inference")

    with torch.enable_grad():
        states = _leaf_states(model.initial_states(x))
        duals = [torch.zeros_like(state) for state in states]

        if config.method == "pc":
            states = _settle(
                model,
                x,
                target,
                states,
                duals,
                rho=config.rho,
                state_lr=config.state_lr,
                steps=config.budget,
            )
        else:
            duals_before = duals
            for _ in range(config.budget - 1):
                states = _settle(
                    model,
                    x,
                    target,
                    states,
                    duals_before,
                    rho=config.rho,
                    state_lr=config.state_lr,
                    steps=config.inner_steps,
                )
                residuals = constraint_residuals(model, x, states)
                duals_before = [
                    (dual + config.alpha * residual).detach()
                    for dual, residual in zip(duals_before, residuals)
                ]

            states = _settle(
                model,
                x,
                target,
                states,
                duals_before,
                rho=config.rho,
                state_lr=config.state_lr,
                steps=config.inner_steps,
            )
            residuals = constraint_residuals(model, x, states)
            duals_after = [
                (dual + config.alpha * residual).detach()
                for dual, residual in zip(duals_before, residuals)
            ]
            duals = duals_before if config.weight_credit_timing == "pre_dual" else duals_after

        residuals = constraint_residuals(model, x, states)
        return InferenceResult(
            states=[state.detach() for state in states],
            duals=[dual.detach() for dual in duals],
            residual_norms=[float(residual.detach().norm()) for residual in residuals],
        )


def _assign_gradients(
    parameters: list[torch.nn.Parameter], gradients: tuple[torch.Tensor, ...]
) -> None:
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.detach()


def local_parameter_backward(
    model: ConnectomePCNetwork,
    x: torch.Tensor,
    target: torch.Tensor,
    states: list[torch.Tensor],
    duals: list[torch.Tensor],
    rho: float,
) -> torch.Tensor:
    """Populate gradients using only each projection's adjacent constraint.

    Autograd differentiates each local scalar separately. It never traverses a
    chain of connectome layers because the settled states and duals are detached.
    """
    model.zero_grad(set_to_none=True)
    states = [state.detach() for state in states]
    duals = [dual.detach() for dual in duals]

    predictions = model.predict_hidden(x.detach(), states)
    for index, (projection, prediction, state, dual) in enumerate(
        zip(model.projections, predictions, states, duals)
    ):
        residual = state - prediction
        local_energy = 0.5 * rho * ((residual + dual / rho) ** 2).sum(dim=-1).mean()
        parameters = [parameter for parameter in projection.parameters() if parameter.requires_grad]
        if parameters:
            gradients = torch.autograd.grad(local_energy, parameters, retain_graph=False)
            _assign_gradients(parameters, gradients)

    logits = model.logits_from_states(states)
    loss = target_loss(logits, target)
    readout_parameters = [
        parameter for parameter in model.readout.parameters() if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(loss, readout_parameters)
    _assign_gradients(readout_parameters, gradients)
    return loss.detach()


def train_step(
    model: ConnectomePCNetwork,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    config: InferenceConfig,
) -> dict[str, float | list[float]]:
    config.validate()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if config.method == "bp":
        logits = model(x)
        loss = target_loss(logits, target)
        loss.backward()
        residual_norms: list[float] = []
        dual_norm = 0.0
    else:
        result = infer(model, x, target, config)
        loss = local_parameter_backward(model, x, target, result.states, result.duals, config.rho)
        residual_norms = result.residual_norms
        dual_norm = float(sum(dual.norm().item() for dual in result.duals))

    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "residual_norms": residual_norms,
        "dual_norm": dual_norm,
    }
