from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import read_connectome_edges

LAYER_MODULES = (
    "visual_projection",
    "central_brain_1",
    "central_brain_2",
    "descending_neuron",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_annotations(path: Path) -> pd.DataFrame:
    columns = ["bodyId", "superclass", "somaSide", "type", "instance", "status"]
    if path.suffix.lower() in {".feather", ".arrow"}:
        return pd.read_feather(path, columns=columns)
    return pd.read_parquet(path, columns=columns)


def _read_neurotransmitters(path: Path) -> pd.DataFrame:
    columns = ["body", "consensus_nt", "predicted_nt_confidence"]
    if path.suffix.lower() in {".feather", ".arrow"}:
        return pd.read_feather(path, columns=columns)
    return pd.read_parquet(path, columns=columns)


def _top_ids(scores: pd.Series, count: int) -> set[int]:
    ordered = scores.sort_values(ascending=False, kind="stable")
    return set(ordered.head(count).index.astype(int).tolist())


def _balanced_visual_inputs(
    edges: pd.DataFrame,
    annotations: pd.DataFrame,
    count: int,
) -> set[int]:
    scores = edges.groupby("source", sort=False)["weight"].sum().rename("score")
    metadata = annotations.set_index("bodyId")[["somaSide"]]
    ranked = scores.to_frame().join(metadata, how="left")
    left_count = count // 2
    right_count = count - left_count
    selected = set(
        ranked[ranked["somaSide"].eq("L")]
        .sort_values("score", ascending=False, kind="stable")
        .head(left_count)
        .index.astype(int)
    )
    selected.update(
        ranked[ranked["somaSide"].eq("R")]
        .sort_values("score", ascending=False, kind="stable")
        .head(right_count)
        .index.astype(int)
    )
    if len(selected) < count:
        remaining = ranked.drop(index=list(selected), errors="ignore")
        selected.update(
            remaining.sort_values("score", ascending=False, kind="stable")
            .head(count - len(selected))
            .index.astype(int)
        )
    return selected


def _prune_disconnected(
    layers: list[set[int]], edge_frames: list[pd.DataFrame]
) -> tuple[list[set[int]], list[pd.DataFrame]]:
    for _ in range(20):
        selected_edges = [
            frame[frame["source"].isin(layers[index]) & frame["target"].isin(layers[index + 1])]
            for index, frame in enumerate(edge_frames)
        ]
        revised = [set(layer) for layer in layers]
        revised[0] &= set(selected_edges[0]["source"])
        revised[-1] &= set(selected_edges[-1]["target"])
        for index in range(1, len(layers) - 1):
            revised[index] &= set(selected_edges[index - 1]["target"])
            revised[index] &= set(selected_edges[index]["source"])
        if [len(layer) for layer in revised] == [len(layer) for layer in layers]:
            return revised, selected_edges
        layers = revised
    raise RuntimeError("circuit pruning did not converge")


def _sign_from_nt(row: pd.Series, confidence_threshold: float) -> int:
    if pd.isna(row.get("predicted_nt_confidence")) or float(
        row["predicted_nt_confidence"]
    ) < confidence_threshold:
        return 0
    neurotransmitter = str(row.get("consensus_nt", "")).lower()
    if neurotransmitter == "acetylcholine":
        return 1
    if neurotransmitter in {"gaba", "histamine"}:
        return -1
    return 0


def select_visual_descending_circuit(
    annotations_path: str | Path,
    neurotransmitters_path: str | Path,
    weights_path: str | Path,
    *,
    layer_sizes: tuple[int, int, int, int] = (32, 48, 40, 24),
    min_weight: float = 5.0,
    nt_confidence: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a deterministic visual→central-brain→descending MaleCNS circuit.

    Middle-layer neurons are ranked by support on real three-edge paths. The
    score for an internal edge is

    ``log1p(input support) * log1p(edge weight) * log1p(output support)``.

    Only exact MaleCNS v1.0 superclass labels are used; no neuron type is
    hand-picked. Disconnected selected nodes are pruned before export.
    """

    if len(layer_sizes) != 4 or any(size < 1 for size in layer_sizes):
        raise ValueError("layer_sizes must contain four positive integers")
    annotations_path = Path(annotations_path)
    neurotransmitters_path = Path(neurotransmitters_path)
    weights_path = Path(weights_path)
    annotations = _read_annotations(annotations_path)
    candidates = {
        superclass: annotations.loc[
            annotations["superclass"].eq(superclass), "bodyId"
        ].astype(np.int64)
        for superclass in ("visual_projection", "cb_intrinsic", "descending_neuron")
    }
    if any(values.empty for values in candidates.values()):
        raise ValueError("required MaleCNS superclasses are missing from annotations")

    edges_01, threshold_count = read_connectome_edges(
        weights_path,
        source_ids=candidates["visual_projection"],
        target_ids=candidates["cb_intrinsic"],
        min_weight=min_weight,
    )
    edges_23, _ = read_connectome_edges(
        weights_path,
        source_ids=candidates["cb_intrinsic"],
        target_ids=candidates["descending_neuron"],
        min_weight=min_weight,
    )
    if edges_01.empty or edges_23.empty:
        raise ValueError("no visual→central-brain→descending candidate edges were found")
    edges_12, _ = read_connectome_edges(
        weights_path,
        source_ids=edges_01["target"].unique(),
        target_ids=edges_23["source"].unique(),
        min_weight=min_weight,
    )
    if edges_12.empty:
        raise ValueError("no central-brain middle edges complete a three-edge path")

    input_support = edges_01.groupby("target")["weight"].sum()
    output_support = edges_23.groupby("source")["weight"].sum()
    edges_12 = edges_12.copy()
    edges_12["path_score"] = (
        np.log1p(edges_12["source"].map(input_support).astype(float))
        * np.log1p(edges_12["weight"].astype(float))
        * np.log1p(edges_12["target"].map(output_support).astype(float))
    )

    layer_1 = _top_ids(edges_12.groupby("source")["path_score"].sum(), layer_sizes[1])
    layer_2: set[int] = set()
    for _ in range(3):
        from_layer_1 = edges_12[
            edges_12["source"].isin(layer_1) & ~edges_12["target"].isin(layer_1)
        ]
        layer_2 = _top_ids(
            from_layer_1.groupby("target")["path_score"].sum(), layer_sizes[2]
        )
        into_layer_2 = edges_12[
            edges_12["target"].isin(layer_2) & ~edges_12["source"].isin(layer_2)
        ]
        layer_1 = _top_ids(
            into_layer_2.groupby("source")["path_score"].sum(), layer_sizes[1]
        )

    layer_0 = _balanced_visual_inputs(
        edges_01[edges_01["target"].isin(layer_1)], annotations, layer_sizes[0]
    )
    layer_3 = _top_ids(
        edges_23[edges_23["source"].isin(layer_2)].groupby("target")["weight"].sum(),
        layer_sizes[3],
    )
    layers, selected_edges = _prune_disconnected(
        [layer_0, layer_1, layer_2, layer_3], [edges_01, edges_12, edges_23]
    )
    if any(not layer for layer in layers):
        raise ValueError("selection produced an empty layer after connectivity pruning")

    selected_ids = set().union(*layers)
    selected_annotations = annotations[annotations["bodyId"].isin(selected_ids)].set_index(
        "bodyId"
    )
    neurotransmitters = _read_neurotransmitters(neurotransmitters_path).set_index("body")
    rows: list[dict[str, Any]] = []
    for layer_index, body_ids in enumerate(layers):
        for body_id in sorted(body_ids):
            annotation = selected_annotations.loc[body_id]
            nt = (
                neurotransmitters.loc[body_id]
                if body_id in neurotransmitters.index
                else pd.Series(dtype=object)
            )
            side = annotation["somaSide"]
            input_position = np.nan
            if layer_index == 0:
                input_position = {"L": -0.58, "R": 0.58}.get(str(side), 0.0)
            rows.append(
                {
                    "body_id": body_id,
                    "layer": layer_index,
                    "module": LAYER_MODULES[layer_index],
                    "sign": _sign_from_nt(nt, nt_confidence),
                    "input_position": input_position,
                    "superclass": annotation["superclass"],
                    "type": annotation["type"],
                    "instance": annotation["instance"],
                    "soma_side": side,
                    "consensus_nt": nt.get("consensus_nt", np.nan),
                    "nt_confidence": nt.get("predicted_nt_confidence", np.nan),
                }
            )
    assignments = pd.DataFrame(rows).sort_values(["layer", "body_id"]).reset_index(drop=True)

    edge_summary = []
    for index, frame in enumerate(selected_edges):
        capacity = len(layers[index]) * len(layers[index + 1])
        edge_summary.append(
            {
                "projection": f"{LAYER_MODULES[index]}->{LAYER_MODULES[index + 1]}",
                "edges": len(frame),
                "density": len(frame) / capacity,
                "weight_sum": int(frame["weight"].sum()),
            }
        )
    report: dict[str, Any] = {
        "selection_version": 2,
        "source": "MaleCNS v1.0 minconf 0.5",
        "source_files": {
            annotations_path.name: _sha256(annotations_path),
            neurotransmitters_path.name: _sha256(neurotransmitters_path),
            weights_path.name: _sha256(weights_path),
        },
        "rules": {
            "superclass_path": [
                "visual_projection",
                "cb_intrinsic",
                "cb_intrinsic",
                "descending_neuron",
            ],
            "module_names": list(LAYER_MODULES),
            "minimum_connection_weight": min_weight,
            "middle_path_score": "log1p(input_support)*log1p(edge_weight)*log1p(output_support)",
            "input_balance": "top visual-projection neurons by support, 50% somaSide L/R",
            "input_position_proxy": {"L": -0.58, "R": 0.58},
            "known_signs": {"acetylcholine": 1, "gaba": -1, "histamine": -1},
            "unknown_or_ambiguous_sign": 0,
            "nt_confidence_threshold": nt_confidence,
        },
        "candidate_neurons": {key: len(values) for key, values in candidates.items()},
        "candidate_edges": {
            "all_edges_at_or_above_threshold": threshold_count,
            "visual_to_central": len(edges_01),
            "central_to_central": len(edges_12),
            "central_to_descending": len(edges_23),
        },
        "requested_layer_sizes": list(layer_sizes),
        "selected_layer_sizes": [len(layer) for layer in layers],
        "selected_edges": edge_summary,
        "input_side_counts": assignments.loc[assignments["layer"].eq(0), "soma_side"]
        .value_counts()
        .to_dict(),
        "sign_counts": assignments["sign"].value_counts().sort_index().to_dict(),
        "scope": "Real MaleCNS topology; synthetic static visual-action labels; no behavior claim.",
    }
    return assignments, report


def write_circuit_selection(
    assignments: pd.DataFrame,
    report: dict[str, Any],
    assignments_path: str | Path,
    report_path: str | Path,
) -> None:
    assignments_path = Path(assignments_path)
    report_path = Path(report_path)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_path, index=False)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
