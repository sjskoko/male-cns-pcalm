from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .connectome import LayeredConnectome
from .model import ConnectomePCNetwork
from .pcalm import InferenceConfig, target_loss, train_step
from .synthetic import degree_preserving_rewire, random_layered_connectome, randomize_edges


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_teacher_dataset(
    connectome: LayeredConnectome,
    *,
    output_dim: int,
    samples: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(samples, connectome.layer_sizes[0], generator=generator).to(device)
    seed_everything(seed + 1)
    teacher = ConnectomePCNetwork(connectome.to(device), output_dim).to(device)
    with torch.no_grad():
        labels = teacher(x).argmax(dim=-1)
        target = F.one_hot(labels, output_dim).to(torch.float32)
    return x, target


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    requested_device = config.get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    if config.get("connectome"):
        connectome = LayeredConnectome.load(config["connectome"])
    else:
        connectome = random_layered_connectome(
            list(config.get("layer_sizes", [12, 20, 16, 10])),
            density=float(config.get("density", 0.25)),
            seed=seed,
        )
    if config.get("rewire") == "degree_preserving":
        connectome = degree_preserving_rewire(connectome, seed=seed)
    elif config.get("rewire") == "random":
        connectome = randomize_edges(connectome, seed=seed)
    connectome = connectome.to(device)

    output_dim = int(config.get("output_dim", 3))
    x, target = make_teacher_dataset(
        connectome,
        output_dim=output_dim,
        samples=int(config.get("samples", 256)),
        seed=seed + 10,
        device=device,
    )
    model = ConnectomePCNetwork(
        connectome,
        output_dim,
        activation=str(config.get("activation", "tanh")),
        trainable_edges=bool(config.get("trainable_edges", True)),
    ).to(device)
    inference = InferenceConfig(
        method=str(config.get("method", "pcalm")),  # type: ignore[arg-type]
        budget=int(config.get("budget", 8)),
        inner_steps=int(config.get("inner_steps", 1)),
        state_lr=float(config.get("state_lr", 0.08)),
        rho=float(config.get("rho", 1.0)),
        alpha=float(config.get("alpha", 0.5)),
        weight_credit_timing=str(config.get("weight_credit_timing", "pre_dual")),  # type: ignore[arg-type]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 3e-3)))
    batch_size = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 5))
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        order = torch.randperm(len(x), device=device)
        losses = []
        for start in range(0, len(x), batch_size):
            indices = order[start : start + batch_size]
            metrics = train_step(model, optimizer, x[indices], target[indices], inference)
            losses.append(float(metrics["loss"]))
        with torch.no_grad():
            logits = model(x)
            evaluation_loss = float(target_loss(logits, target))
            accuracy = float((logits.argmax(dim=-1) == target.argmax(dim=-1)).float().mean())
        history.append(
            {
                "epoch": epoch + 1,
                "train_local_loss": float(np.mean(losses)),
                "forward_loss": evaluation_loss,
                "accuracy": accuracy,
            }
        )

    return {
        "method": inference.method,
        "device": str(device),
        "layer_sizes": connectome.layer_sizes,
        "num_edges": connectome.num_edges,
        "history": history,
        "final": history[-1],
    }


def write_results(result: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["history"][0].keys()))
        writer.writeheader()
        writer.writerows(result["history"])
