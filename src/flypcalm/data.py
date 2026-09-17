from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .connectome import LayeredConnectome, LayerSpec

SOURCE_ALIASES = ("body_pre", "source", "source_id", "pre", "pre_id")
TARGET_ALIASES = ("body_post", "target", "target_id", "post", "post_id")
WEIGHT_ALIASES = ("weight", "syn_count", "synapse_count", "count")
NODE_ALIASES = ("body_id", "bodyid", "body", "node_id", "root_id")


def _read_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".feather", ".arrow"}:
        return pd.read_feather(path, columns=columns)
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", usecols=columns)
    raise ValueError(f"unsupported table format: {path.suffix}")


def _columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".feather", ".arrow"}:
        import pyarrow as pa
        from pyarrow import ipc

        with pa.memory_map(str(path), "r") as source:
            return list(ipc.open_file(source).schema.names)
    if suffix == ".parquet":
        from pyarrow import parquet

        return list(parquet.read_schema(path).names)
    if suffix in {".csv", ".tsv"}:
        return list(pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=0).columns)
    raise ValueError(f"unsupported table format: {path.suffix}")


def _resolve(available: Iterable[str], aliases: Iterable[str], label: str) -> str:
    by_normalized = {name.lower().replace("-", "_"): name for name in available}
    for alias in aliases:
        if alias in by_normalized:
            return by_normalized[alias]
    raise ValueError(f"could not find {label}; available columns: {sorted(available)}")


def _normalize_weights(raw: torch.Tensor, targets: torch.Tensor, target_count: int) -> torch.Tensor:
    transformed = torch.log1p(raw.clamp_min(0).to(torch.float32))
    squared_sum = torch.zeros(target_count, dtype=torch.float32)
    squared_sum.scatter_add_(0, targets, transformed.square())
    denominator = squared_sum.sqrt().clamp_min(1.0)
    return transformed / denominator[targets]


def build_layered_connectome(
    edges_path: str | Path,
    assignments_path: str | Path,
    *,
    min_weight: float = 1.0,
    report_path: str | Path | None = None,
) -> tuple[LayeredConnectome, dict[str, int | float | list[str]]]:
    """Project a directed connectome onto explicitly assigned adjacent layers.

    Assignment schema: ``body_id, layer, module`` and optional ``sign``. Layer
    must be a contiguous integer starting at zero. Sign is -1, 0, or +1 and is
    interpreted as a presynaptic/neuron sign.
    """
    edges_path = Path(edges_path)
    assignments_path = Path(assignments_path)
    assignment_columns = _columns(assignments_path)
    node_col = _resolve(assignment_columns, NODE_ALIASES, "assignment node ID")
    required = [node_col, "layer", "module"]
    if "sign" in assignment_columns:
        required.append("sign")
    if "input_position" in assignment_columns:
        required.append("input_position")
    assignments = _read_frame(assignments_path, required).rename(columns={node_col: "body_id"})
    if assignments["body_id"].duplicated().any():
        raise ValueError("assignments contain duplicate body IDs")
    assignments["layer"] = assignments["layer"].astype(int)
    layer_numbers = sorted(assignments["layer"].unique().tolist())
    if layer_numbers != list(range(len(layer_numbers))) or len(layer_numbers) < 2:
        raise ValueError("assignment layers must be contiguous integers 0..N with N >= 1")
    if "sign" not in assignments:
        assignments["sign"] = 0
    assignments["sign"] = assignments["sign"].fillna(0).astype(int)
    if not assignments["sign"].isin([-1, 0, 1]).all():
        raise ValueError("assignment sign must be -1, 0, or 1")

    edge_columns = _columns(edges_path)
    source_col = _resolve(edge_columns, SOURCE_ALIASES, "presynaptic/source ID")
    target_col = _resolve(edge_columns, TARGET_ALIASES, "postsynaptic/target ID")
    weight_col = _resolve(edge_columns, WEIGHT_ALIASES, "connection weight")
    edges = _read_frame(edges_path, [source_col, target_col, weight_col]).rename(
        columns={source_col: "source", target_col: "target", weight_col: "weight"}
    )
    edges = edges[edges["weight"] >= min_weight]

    source_info = assignments[["body_id", "layer", "sign"]].rename(
        columns={"body_id": "source", "layer": "source_layer", "sign": "source_sign"}
    )
    target_info = assignments[["body_id", "layer"]].rename(
        columns={"body_id": "target", "layer": "target_layer"}
    )
    merged = edges.merge(source_info, on="source", how="inner").merge(
        target_info, on="target", how="inner"
    )
    delta = merged["target_layer"] - merged["source_layer"]
    report: dict[str, int | float | list[str]] = {
        "input_edges_after_threshold": len(edges),
        "edges_with_both_nodes_assigned": len(merged),
        "retained_adjacent_forward_edges": int((delta == 1).sum()),
        "dropped_intra_layer_edges": int((delta == 0).sum()),
        "dropped_feedback_edges": int((delta < 0).sum()),
        "dropped_skip_forward_edges": int((delta > 1).sum()),
        "layer_names": [],
        "min_weight": float(min_weight),
    }

    layers: list[LayerSpec] = []
    for layer_number in range(len(layer_numbers) - 1):
        source_nodes = assignments[assignments["layer"] == layer_number].sort_values("body_id")
        target_nodes = assignments[assignments["layer"] == layer_number + 1].sort_values("body_id")
        source_ids = source_nodes["body_id"].to_numpy(dtype=np.int64, copy=True)
        target_ids = target_nodes["body_id"].to_numpy(dtype=np.int64, copy=True)
        source_lookup = {node_id: index for index, node_id in enumerate(source_ids.tolist())}
        target_lookup = {node_id: index for index, node_id in enumerate(target_ids.tolist())}
        selected = merged[
            (merged["source_layer"] == layer_number) & (merged["target_layer"] == layer_number + 1)
        ]
        target_index = torch.tensor(
            [target_lookup[node_id] for node_id in selected["target"]], dtype=torch.long
        )
        source_index = torch.tensor(
            [source_lookup[node_id] for node_id in selected["source"]], dtype=torch.long
        )
        edge_index = torch.stack([target_index, source_index])
        weights = _normalize_weights(
            torch.tensor(selected["weight"].to_numpy(copy=True), dtype=torch.float32),
            target_index,
            len(target_ids),
        )
        signs = torch.tensor(selected["source_sign"].to_numpy(copy=True), dtype=torch.int8)
        values = torch.where(signs == 0, weights, signs.to(torch.float32) * weights)
        source_name = str(source_nodes["module"].mode().iloc[0])
        target_name = str(target_nodes["module"].mode().iloc[0])
        layers.append(
            LayerSpec(
                source_ids=torch.from_numpy(source_ids),
                target_ids=torch.from_numpy(target_ids),
                edge_index=edge_index,
                initial_values=values,
                sign=signs,
                source_name=source_name,
                target_name=target_name,
            )
        )
        report["layer_names"].append(source_name)  # type: ignore[union-attr]
    report["layer_names"].append(layers[-1].target_name)  # type: ignore[union-attr]

    metadata: dict[str, object] = {
        "kind": "MaleCNS projection",
        "source_edges": edges_path.name,
        "source_assignments": assignments_path.name,
        "projection_report": report,
    }
    if "input_position" in assignments:
        input_nodes = assignments[assignments["layer"] == 0].sort_values("body_id")
        if input_nodes["input_position"].isna().any():
            raise ValueError("layer-0 neurons must all have input_position when the column exists")
        input_positions = input_nodes["input_position"].astype(float)
        if not input_positions.between(-1.0, 1.0).all():
            raise ValueError("input_position must be normalized to the [-1, 1] range")
        metadata["input_positions"] = input_positions.tolist()

    connectome = LayeredConnectome(
        tuple(layers),
        metadata,
    )
    connectome.validate()
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return connectome, report
