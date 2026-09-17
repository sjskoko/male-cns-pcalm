from __future__ import annotations

import torch

from flypcalm.tasks import make_visual_motor_dataset


def test_visual_motor_dataset_is_balanced_reproducible_and_lateralized():
    first = make_visual_motor_dataset(12, 80, seed=4)
    second = make_visual_motor_dataset(12, 80, seed=4)
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(torch.bincount(first.labels), torch.tensor([20, 20, 20, 20]))

    left_half = first.input_positions < 0
    right_half = first.input_positions > 0
    turn_left = first.inputs[first.labels == 0]
    turn_right = first.inputs[first.labels == 2]
    forward = first.inputs[first.labels == 1]
    stop = first.inputs[first.labels == 3]
    assert turn_left[:, right_half].mean() > turn_left[:, left_half].mean()
    assert turn_right[:, left_half].mean() > turn_right[:, right_half].mean()
    assert stop.mean() > forward.mean()
