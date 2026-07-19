from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage

from roadgraph.benchmark import create_synthetic_gaps, evaluate_healing
from roadgraph.extract import graph_from_mask, neighbor_count, skeleton_from_mask
from roadgraph.healing import HealingConfig, heal_mask
from run_part2_experiments import select_rows

from .io import write_csv, write_graphml


def _candidate_config(gap: int, angle: int, confidence: float) -> HealingConfig:
    return HealingConfig(
        experiment=f"B007_g{gap}_a{angle}_c{confidence:.2f}",
        maximum_gap_pixels=float(gap),
        maximum_angle_degrees=float(angle),
        use_angle=True,
        use_astar=True,
        use_mst=True,
        bridge_radius=1,
        minimum_path_confidence=float(confidence),
    )


def benchmark_configuration(
    config: HealingConfig,
    rows: list[dict[str, str]],
    gaps_per_tile: int,
    seed: int,
) -> dict[str, float]:
    records = []
    for index, row in enumerate(rows):
        original = np.asarray(Image.open(row["mask_path"])) > 127
        broken, gap_target, probability = create_synthetic_gaps(
            original, seed + index, gap_count=gaps_per_tile
        )
        healed, _ = heal_mask(broken, probability, config)
        records.append(
            evaluate_healing(
                original, broken, healed, gap_target, seed + 10_000 + index
            )
        )
    names = records[0].keys()
    return {
        name: float(np.mean([record[name] for record in records]))
        for name in names
    }


def graph_node_coverage(mask: np.ndarray) -> tuple[int, int, float]:
    skeleton = skeleton_from_mask(mask)
    skeleton_components, component_count = ndimage.label(
        skeleton, structure=np.ones((3, 3), dtype=np.uint8)
    )
    anchors = skeleton & (neighbor_count(skeleton) != 2)
    anchor_labels, anchor_count = ndimage.label(
        anchors, structure=np.ones((3, 3), dtype=np.uint8)
    )
    nodes_by_component = np.zeros(component_count + 1, dtype=np.int64)
    for anchor in range(1, anchor_count + 1):
        pixels = np.argwhere(anchor_labels == anchor)
        if pixels.size == 0:
            continue
        components = skeleton_components[pixels[:, 0], pixels[:, 1]]
        components = components[components > 0]
        if components.size:
            nodes_by_component[int(np.bincount(components).argmax())] += 1
    for component in range(1, component_count + 1):
        if nodes_by_component[component] == 0 and np.any(
            skeleton_components == component
        ):
            nodes_by_component[component] = 1
    total_nodes = int(nodes_by_component.sum())
    largest_nodes = int(nodes_by_component.max()) if total_nodes else 0
    coverage = largest_nodes / max(total_nodes, 1)
    return total_nodes, largest_nodes, coverage


def run_consolidation(config: dict) -> dict:
    paths = config["paths"]
    settings = config["consolidation"]
    output = Path(paths["consolidation_output"])
    output.mkdir(parents=True, exist_ok=True)
    rows = select_rows(
        Path(paths["benchmark_manifest"]), int(settings["benchmark_tiles"])
    )
    with rasterio.open(paths["raw_mask"]) as source:
        raw_mask = source.read(1).astype(bool)
        raster_profile = source.profile.copy()
        transform = source.transform
        crs = source.crs.to_string()
    with rasterio.open(paths["probability"]) as source:
        probability = source.read(1).astype(np.float32)

    grid_rows = []
    safe_results = []
    combinations = list(
        itertools.product(
            settings["maximum_gap_pixels"],
            settings["maximum_angle_degrees"],
            settings["minimum_path_confidence"],
        )
    )
    for index, (gap, angle, confidence) in enumerate(combinations, start=1):
        candidate = _candidate_config(gap, angle, confidence)
        start = time.perf_counter()
        benchmark = benchmark_configuration(
            candidate,
            rows,
            int(settings["gaps_per_tile"]),
            int(config["seed"]),
        )
        record = {
            "config_id": candidate.experiment,
            "maximum_gap_pixels": gap,
            "maximum_angle_degrees": angle,
            "minimum_path_confidence": confidence,
            **benchmark,
            "benchmark_runtime_seconds": time.perf_counter() - start,
        }
        record["safe"] = (
            record["false_bridge_rate"] <= settings["false_bridge_limit"]
        )
        grid_rows.append(record)
        if record["safe"]:
            healed, bridges = heal_mask(raw_mask, probability, candidate)
            total_nodes, largest_nodes, coverage = graph_node_coverage(healed)
            record.update(
                {
                    "graph_nodes_estimated": total_nodes,
                    "largest_component_nodes_estimated": largest_nodes,
                    "node_coverage": coverage,
                    "bridges_added": len(bridges),
                }
            )
            safe_results.append((record, candidate, healed, bridges))
        print(
            f"[{index}/{len(combinations)}] {candidate.experiment} "
            f"false={record['false_bridge_rate']:.4f} "
            f"coverage={record.get('node_coverage', 0):.4f}",
            flush=True,
        )

    if not safe_results:
        raise RuntimeError("No consolidation configuration passed the false-bridge gate")
    selected_record, selected_config, selected_mask, selected_bridges = max(
        safe_results, key=lambda item: item[0]["node_coverage"]
    )
    gate_met = (
        selected_record["node_coverage"] >= settings["target_node_coverage"]
    )
    selected_record["coverage_gate_met"] = gate_met
    selected_record["false_bridge_gate_met"] = True

    raster_profile.update(dtype="uint8", count=1, compress="deflate")
    with rasterio.open(output / "consolidated_mask.tif", "w", **raster_profile) as destination:
        destination.write(selected_mask.astype(np.uint8), 1)

    graph = graph_from_mask(
        selected_mask,
        resolution_m=float(abs(transform.a)),
        origin_xy=(float(transform.c), float(transform.f)),
    )
    graph.graph.update(
        {
            "crs": crs,
            "consolidation_config": selected_config.experiment,
            "coverage_gate_met": bool(gate_met),
        }
    )
    write_graphml(graph, output / "consolidated_graph.graphml")
    write_csv(output / "consolidation_grid.csv", grid_rows)
    write_csv(output / "accepted_bridges.csv", selected_bridges)
    summary = {
        "selected": selected_record,
        "target_node_coverage": settings["target_node_coverage"],
        "false_bridge_limit": settings["false_bridge_limit"],
        "coverage_gate_met": gate_met,
        "routing_policy": (
            "largest_safe_component"
            if not gate_met
            else "largest_consolidated_component"
        ),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "graph_components": nx.number_connected_components(nx.Graph(graph)),
        "outputs": {
            "graph": str(output / "consolidated_graph.graphml"),
            "mask": str(output / "consolidated_mask.tif"),
        },
    }
    (output / "consolidation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary
