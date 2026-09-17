"""Task generators used by controlled connectome experiments."""

from .visual_motor import ACTION_NAMES, VisualMotorDataset, make_visual_motor_dataset

__all__ = ["ACTION_NAMES", "VisualMotorDataset", "make_visual_motor_dataset"]
