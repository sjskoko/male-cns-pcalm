"""MaleCNS × PC-ALM research prototype."""

from .connectome import LayeredConnectome, LayerSpec
from .model import ConnectomePCNetwork
from .pcalm import InferenceConfig, infer, train_step

__all__ = [
    "ConnectomePCNetwork",
    "InferenceConfig",
    "LayerSpec",
    "LayeredConnectome",
    "infer",
    "train_step",
]

__version__ = "0.2.0"
