from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .connectome import LayeredConnectome
from .experiment import seed_everything
from .metrics import gradient_alignment
from .model import ConnectomePCNetwork
from .pcalm import InferenceConfig, target_loss, train_step
from .synthetic import (
    degree_preserving_rewire,
    random_layered_connectome,
    randomize_edges,
    visual_proxy_connectome,
)
from .tasks import ACTION_NAMES, VisualMotorDataset, make_visual_motor_dataset

TOPOLOGIES = ("native", "degree_preserving", "random")
METHODS = ("bp", "pc", "pcalm")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _base_connectome(config: dict[str, Any]) -> LayeredConnectome:
    if config.get("connectome"):
        return LayeredConnectome.load(config["connectome"])
    layer_sizes = list(config.get("layer_sizes", [16, 24, 20, 12]))
    graph_kind = str(config.get("synthetic_graph", "visual_proxy"))
    common = {
        "density": float(config.get("density", 0.25)),
        "seed": int(config.get("graph_seed", 17)),
    }
    if graph_kind == "visual_proxy":
        return visual_proxy_connectome(
            layer_sizes,
            receptive_width=float(config.get("receptive_width", 0.32)),
            **common,
        )
    if graph_kind == "random":
        return random_layered_connectome(layer_sizes, **common)
    raise ValueError("synthetic_graph must be visual_proxy or random")


def _topology_variant(base: LayeredConnectome, topology: str, *, seed: int) -> LayeredConnectome:
    if topology == "native":
        return base
    if topology == "degree_preserving":
        return degree_preserving_rewire(base, swaps_per_edge=8.0, seed=seed)
    if topology == "random":
        return randomize_edges(base, seed=seed)
    raise ValueError(f"unknown topology: {topology}")


def _inference_config(config: dict[str, Any], method: str) -> InferenceConfig:
    return InferenceConfig(
        method=method,  # type: ignore[arg-type]
        budget=int(config.get("budget", 6)),
        inner_steps=int(config.get("inner_steps", 1)),
        state_lr=float(config.get("state_lr", 0.04)),
        rho=float(config.get("rho", 1.0)),
        alpha=float(config.get("alpha", 0.5)),
        weight_credit_timing=str(config.get("weight_credit_timing", "pre_dual")),  # type: ignore[arg-type]
    )


def _evaluate(model: ConnectomePCNetwork, dataset: VisualMotorDataset) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(dataset.inputs)
        loss = float(target_loss(logits, dataset.targets))
        accuracy = float((logits.argmax(dim=-1) == dataset.labels).float().mean())
    return loss, accuracy


def _run_trial(
    graph: LayeredConnectome,
    train_data: VisualMotorDataset,
    test_data: VisualMotorDataset,
    *,
    topology: str,
    method: str,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed + 1_000)
    model = ConnectomePCNetwork(
        graph.to(device),
        len(ACTION_NAMES),
        activation=str(config.get("activation", "tanh")),
        trainable_edges=bool(config.get("trainable_edges", True)),
    ).to(device)
    inference = _inference_config(config, method)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 3e-3)))
    batch_size = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 6))
    diagnostic_size = min(batch_size, len(train_data.inputs))
    diagnostic_x = train_data.inputs[:diagnostic_size]
    diagnostic_target = train_data.targets[:diagnostic_size]
    alignment_before = gradient_alignment(model, diagnostic_x, diagnostic_target, inference)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_data.inputs), device=device)
        local_losses: list[float] = []
        residuals: list[float] = []
        dual_norms: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            metrics = train_step(
                model,
                optimizer,
                train_data.inputs[indices],
                train_data.targets[indices],
                inference,
            )
            local_losses.append(float(metrics["loss"]))
            residuals.extend(float(value) for value in metrics["residual_norms"])
            dual_norms.append(float(metrics["dual_norm"]))
        train_loss, train_accuracy = _evaluate(model, train_data)
        test_loss, test_accuracy = _evaluate(model, test_data)
        history.append(
            {
                "topology": topology,
                "method": method,
                "seed": seed,
                "epoch": epoch,
                "train_local_loss": float(np.mean(local_losses)),
                "train_forward_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "mean_residual_norm": float(np.mean(residuals)) if residuals else 0.0,
                "mean_dual_norm": float(np.mean(dual_norms)) if dual_norms else 0.0,
            }
        )

    runtime_seconds = time.perf_counter() - started
    alignment_after = gradient_alignment(model, diagnostic_x, diagnostic_target, inference)
    threshold_epoch = next(
        (int(row["epoch"]) for row in history if float(row["test_accuracy"]) >= 0.8), None
    )
    trial = {
        "topology": topology,
        "method": method,
        "seed": seed,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "best_test_accuracy": max(row["test_accuracy"] for row in history),
        "test_accuracy_auc": float(np.mean([row["test_accuracy"] for row in history])),
        "epochs_to_80pct": threshold_epoch,
        "final_test_loss": history[-1]["test_loss"],
        "gradient_cosine_before": alignment_before["global_cosine"],
        "gradient_cosine_after": alignment_after["global_cosine"],
        "runtime_seconds": runtime_seconds,
        "num_edges": graph.num_edges,
    }
    details = {
        "topology": topology,
        "method": method,
        "seed": seed,
        "alignment_before": alignment_before,
        "alignment_after": alignment_after,
    }
    return trial, history, details


def _aggregate(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topology in TOPOLOGIES:
        for method in METHODS:
            selected = [
                row for row in trials if row["topology"] == topology and row["method"] == method
            ]
            if not selected:
                continue

            def values(key: str, group: list[dict[str, Any]] = selected) -> list[float]:
                return [float(row[key]) for row in group]

            accuracy = values("final_test_accuracy")
            threshold_epochs = [
                float(row["epochs_to_80pct"])
                for row in selected
                if row["epochs_to_80pct"] is not None
            ]
            rows.append(
                {
                    "topology": topology,
                    "method": method,
                    "runs": len(selected),
                    "accuracy_mean": statistics.fmean(accuracy),
                    "accuracy_std": statistics.stdev(accuracy) if len(accuracy) > 1 else 0.0,
                    "best_accuracy_mean": statistics.fmean(values("best_test_accuracy")),
                    "accuracy_auc_mean": statistics.fmean(values("test_accuracy_auc")),
                    "threshold_80pct_success_rate": len(threshold_epochs) / len(selected),
                    "epochs_to_80pct_mean": (
                        statistics.fmean(threshold_epochs) if threshold_epochs else ""
                    ),
                    "loss_mean": statistics.fmean(values("final_test_loss")),
                    "gradient_cosine_before_mean": statistics.fmean(
                        values("gradient_cosine_before")
                    ),
                    "gradient_cosine_after_mean": statistics.fmean(values("gradient_cosine_after")),
                    "runtime_seconds_mean": statistics.fmean(values("runtime_seconds")),
                }
            )
    return rows


def _paired_effects(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute paired native-minus-control effects using matching random seeds."""

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in trials if row["method"] == method]
        by_condition = {
            (str(row["topology"]), int(row["seed"])): row for row in method_rows
        }
        for control in ("degree_preserving", "random"):
            seeds = sorted(
                seed
                for topology, seed in by_condition
                if topology == "native" and (control, seed) in by_condition
            )
            if not seeds:
                continue
            accuracy_deltas = [
                float(by_condition[("native", seed)]["final_test_accuracy"])
                - float(by_condition[(control, seed)]["final_test_accuracy"])
                for seed in seeds
            ]
            auc_deltas = [
                float(by_condition[("native", seed)]["test_accuracy_auc"])
                - float(by_condition[(control, seed)]["test_accuracy_auc"])
                for seed in seeds
            ]
            cosine_deltas = [
                float(by_condition[("native", seed)]["gradient_cosine_after"])
                - float(by_condition[(control, seed)]["gradient_cosine_after"])
                for seed in seeds
            ]
            rows.append(
                {
                    "method": method,
                    "control": control,
                    "paired_runs": len(seeds),
                    "accuracy_delta_mean": statistics.fmean(accuracy_deltas),
                    "accuracy_delta_std": (
                        statistics.stdev(accuracy_deltas) if len(seeds) > 1 else 0.0
                    ),
                    "accuracy_native_win_rate": sum(delta > 0 for delta in accuracy_deltas)
                    / len(seeds),
                    "accuracy_auc_delta_mean": statistics.fmean(auc_deltas),
                    "gradient_cosine_delta_mean": statistics.fmean(cosine_deltas),
                }
            )
    return rows


def run_visual_motor_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    device = _device(str(config.get("device", "auto")))
    base = _base_connectome(config)
    topologies = list(config.get("topologies", TOPOLOGIES))
    methods = list(config.get("methods", METHODS))
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    unknown_topologies = set(topologies) - set(TOPOLOGIES)
    unknown_methods = set(methods) - set(METHODS)
    if unknown_topologies:
        raise ValueError(f"unknown topologies: {sorted(unknown_topologies)}")
    if unknown_methods:
        raise ValueError(f"unknown methods: {sorted(unknown_methods)}")
    if not seeds:
        raise ValueError("at least one seed is required")

    positions = base.metadata.get("input_positions")
    layout = "connectome_metadata" if positions is not None else "index_proxy"
    trials: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for seed in seeds:
        train_data = make_visual_motor_dataset(
            base.layer_sizes[0],
            int(config.get("train_samples", 256)),
            seed=seed + 10_000,
            input_positions=positions,  # type: ignore[arg-type]
            noise_std=float(config.get("noise_std", 0.04)),
        ).to(device)
        test_data = make_visual_motor_dataset(
            base.layer_sizes[0],
            int(config.get("test_samples", 128)),
            seed=seed + 20_000,
            input_positions=positions,  # type: ignore[arg-type]
            noise_std=float(config.get("noise_std", 0.04)),
        ).to(device)
        for topology in topologies:
            graph = _topology_variant(base, topology, seed=seed + 30_000)
            for method in methods:
                trial, history, diagnostic = _run_trial(
                    graph,
                    train_data,
                    test_data,
                    topology=topology,
                    method=method,
                    seed=seed,
                    config=config,
                    device=device,
                )
                trials.append(trial)
                metrics.extend(history)
                details.append(diagnostic)

    uses_malecns_topology = bool(base.metadata.get("kind") == "MaleCNS projection")
    return {
        "experiment": "visual_motor_topology_x_learning_rule",
        "device": str(device),
        "base_graph_kind": base.metadata.get("kind", "unknown"),
        "uses_malecns_topology": uses_malecns_topology,
        "task_is_synthetic": True,
        "biological_claim": False,
        "input_layout": layout,
        "layer_sizes": base.layer_sizes,
        "base_edges": base.num_edges,
        "actions": list(ACTION_NAMES),
        "config": config,
        "trials": trials,
        "metrics": metrics,
        "diagnostics": details,
        "aggregate": _aggregate(trials),
        "paired_effects": _paired_effects(trials),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Visual–motor topology × learning-rule benchmark",
        "",
        f"- Base graph: `{result['base_graph_kind']}`",
        f"- Device: `{result['device']}`",
        f"- Layer sizes: `{list(result['layer_sizes'])}`",
        f"- Base edges: `{result['base_edges']}`",
        f"- Input layout: `{result['input_layout']}`",
        "",
    ]
    if result["uses_malecns_topology"]:
        lines.extend(
            [
                "> This run uses a MaleCNS topology with a synthetic proxy task. It is not evidence of fly behavior.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> This is a virtual pipeline check with a synthetic graph. It is not a MaleCNS result.",
                "",
            ]
        )
    lines.extend(
        [
            "| Topology | Method | Runs | Test accuracy | Accuracy AUC | BP-gradient cosine | Runtime (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["aggregate"]:
        lines.append(
            "| {topology} | {method} | {runs} | {accuracy_mean:.3f} ± {accuracy_std:.3f} "
            "| {accuracy_auc_mean:.3f} | {gradient_cosine_after_mean:.3f} "
            "| {runtime_seconds_mean:.2f} |".format(**row)
        )
    if result.get("paired_effects"):
        lines.extend(
            [
                "",
                "## Paired native-minus-control effects",
                "",
                "Positive values favor the native topology. Runs are paired by seed.",
                "",
                "| Method | Control | Runs | Accuracy delta | Native win rate | AUC delta | Gradient-cosine delta |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["paired_effects"]:
            lines.append(
                "| {method} | {control} | {paired_runs} | {accuracy_delta_mean:+.3f} ± "
                "{accuracy_delta_std:.3f} | {accuracy_native_win_rate:.2f} | "
                "{accuracy_auc_delta_mean:+.3f} | {gradient_cosine_delta_mean:+.3f} |".format(
                    **row
                )
            )
    lines.extend(
        [
            "",
            "Interpret accuracy differences only after running multiple seeds. The primary MaleCNS test is",
            "native versus degree-preserving rewiring under PC-ALM; the virtual graph only validates the workflow.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_results(result: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    _write_csv(output / "trials.csv", result["trials"])
    _write_csv(output / "metrics.csv", result["metrics"])
    _write_csv(output / "aggregate.csv", result["aggregate"])
    _write_csv(output / "paired_effects.csv", result.get("paired_effects", []))
