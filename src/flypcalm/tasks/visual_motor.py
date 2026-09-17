from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

ACTION_NAMES = ("turn_left", "forward", "turn_right", "stop")


@dataclass(frozen=True)
class VisualMotorDataset:
    inputs: torch.Tensor
    targets: torch.Tensor
    labels: torch.Tensor
    input_positions: torch.Tensor

    def to(self, device: torch.device | str) -> VisualMotorDataset:
        return VisualMotorDataset(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            labels=self.labels.to(device),
            input_positions=self.input_positions.to(device),
        )


def make_visual_motor_dataset(
    input_dim: int,
    samples: int,
    *,
    seed: int,
    input_positions: list[float] | torch.Tensor | None = None,
    noise_std: float = 0.04,
) -> VisualMotorDataset:
    """Create a balanced, graph-independent visual-to-action proxy task.

    A strong signal on the right calls for a left turn, a strong signal on the
    left calls for a right turn, low symmetric flow calls for forward motion,
    and high bilateral flow represents looming and calls for stopping.
    """
    if input_dim < 4:
        raise ValueError("visual-motor task requires at least four input neurons")
    if samples < len(ACTION_NAMES):
        raise ValueError("samples must be at least the number of actions")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    if input_positions is None:
        positions = torch.linspace(-1.0, 1.0, input_dim)
    else:
        positions = torch.as_tensor(input_positions, dtype=torch.float32)
        if positions.shape != (input_dim,):
            raise ValueError("input_positions must have one value per input neuron")
        if not torch.isfinite(positions).all():
            raise ValueError("input_positions must be finite")
        if float(positions.min()) < -1.0 or float(positions.max()) > 1.0:
            raise ValueError("input_positions must be normalized to [-1, 1]")

    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(samples, dtype=torch.long) % len(ACTION_NAMES)
    labels = labels[torch.randperm(samples, generator=generator)]
    inputs = torch.empty(samples, input_dim, dtype=torch.float32)
    left_bump = torch.exp(-0.5 * ((positions + 0.58) / 0.28) ** 2)
    right_bump = torch.exp(-0.5 * ((positions - 0.58) / 0.28) ** 2)
    bilateral = 0.75 + 0.25 * torch.exp(-0.5 * (positions / 0.65) ** 2)

    for index, label in enumerate(labels.tolist()):
        amplitude = 0.75 + 0.25 * torch.rand((), generator=generator)
        background = 0.03 + 0.05 * torch.rand((), generator=generator)
        if label == 0:  # obstacle/flow on the right -> turn left
            pattern = background + amplitude * right_bump
        elif label == 1:  # low, balanced flow -> continue forward
            pattern = background + 0.16 * bilateral
        elif label == 2:  # obstacle/flow on the left -> turn right
            pattern = background + amplitude * left_bump
        else:  # bilateral looming -> stop
            pattern = background + amplitude * bilateral
        nuisance_center = -0.8 + 1.6 * torch.rand((), generator=generator)
        nuisance = 0.08 * torch.exp(-0.5 * ((positions - nuisance_center) / 0.18) ** 2)
        noise = noise_std * torch.randn(input_dim, generator=generator)
        inputs[index] = (pattern + nuisance + noise).clamp(0.0, 1.25)

    targets = F.one_hot(labels, num_classes=len(ACTION_NAMES)).to(torch.float32)
    return VisualMotorDataset(inputs, targets, labels, positions)
