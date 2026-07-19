from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import rasterio
import torch
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from torch.utils.data import DataLoader

from roadgraph.benchmark import component_statistics
from roadgraph.extract import (
    clean_mask,
    graph_from_mask,
    graph_to_geojson,
)
from roadgraph.healing import EXPERIMENTS, heal_mask
from roadseg.data import RoadDataset
from roadseg.model import build_model


def read_json(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not decode JSON metadata: {path}")


def crop_transform() -> tuple[Affine, str, tuple[int, int]]:
    metadata = read_json(
        "data/processed/bengaluru_edge/t256_s128/crop_metadata.json"
    )
    bbox = read_json("data/metadata/bengaluru_edge.bbox.json")
    red_path = next(
        Path("data/raw/sentinel2/bengaluru_edge").rglob("B04_red.tif")
    )
    with rasterio.open(red_path) as source:
        projector = Transformer.from_crs(
            "EPSG:4326", source.crs, always_xy=True
        )
        west, south = projector.transform(bbox["west"], bbox["south"])
        east, north = projector.transform(bbox["east"], bbox["north"])
        window = from_bounds(
            min(west, east),
            min(south, north),
            max(west, east),
            max(south, north),
            transform=source.transform,
        ).round_offsets().round_lengths()
        transform = source.window_transform(window)
        crs = source.crs.to_string()
    return transform, crs, tuple(metadata["crop_shape"])


@torch.inference_mode()
def predict_mosaic(
    checkpoint_path: Path,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    data_config = config["data"]
    dataset = RoadDataset(
        config["manifest"],
        "test",
        data_config["mean"],
        data_config["std"],
        training=False,
        occlusion_probability=0.0,
        seed=config["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    row_lookup = {row["tile_id"]: row for row in dataset.rows}
    _, _, shape = crop_transform()
    probability_sum = np.zeros(shape, dtype=np.float32)
    observation_count = np.zeros(shape, dtype=np.float32)
    for batch in loader:
        probabilities = model(batch["image"].to(device)).sigmoid().cpu().numpy()
        for index, tile_id in enumerate(batch["tile_id"]):
            row = row_lookup[tile_id]
            top, left = int(row["row"]), int(row["col"])
            height, width = probabilities[index, 0].shape
            bottom = min(top + height, shape[0])
            right = min(left + width, shape[1])
            tile = probabilities[index, 0, : bottom - top, : right - left]
            probability_sum[top:bottom, left:right] += tile
            observation_count[top:bottom, left:right] += 1
    probability = np.divide(
        probability_sum,
        observation_count,
        out=np.zeros_like(probability_sum),
        where=observation_count > 0,
    )
    mask = clean_mask(
        probability >= threshold,
        minimum_object_size=12,
        maximum_hole_size=12,
    )
    return probability, mask, config


def save_raster(
    path: Path,
    array: np.ndarray,
    transform: Affine,
    crs: str,
    dtype: str,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as destination:
        destination.write(array.astype(dtype), 1)


def write_graphml(graph: nx.MultiGraph, path: Path) -> None:
    export = nx.MultiGraph()
    export.graph.update(
        {
            key: value
            for key, value in graph.graph.items()
            if isinstance(value, (str, int, float, bool))
        }
    )
    for node, data in graph.nodes(data=True):
        export.add_node(
            node,
            **{
                key: value
                for key, value in data.items()
                if isinstance(value, (str, int, float, bool))
            },
        )
    for source, target, key, data in graph.edges(keys=True, data=True):
        attributes = {
            name: value
            for name, value in data.items()
            if isinstance(value, (str, int, float, bool))
        }
        attributes["pixels_json"] = json.dumps(data.get("pixels", []))
        export.add_edge(source, target, key=key, **attributes)
    nx.write_graphml(export, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/e013_pretrained_segformer_rgbn/best.pt",
    )
    parser.add_argument("--threshold", type=float, default=0.425)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="B007")
    parser.add_argument("--output", default="runs/part2_healing/real_graph")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    transform, crs, _ = crop_transform()
    probability, raw_mask, model_config = predict_mosaic(
        Path(args.checkpoint), args.threshold
    )
    healed_mask, bridges = heal_mask(
        raw_mask, probability, EXPERIMENTS[args.experiment]
    )
    before_components, before_largest = component_statistics(raw_mask)
    after_components, after_largest = component_statistics(healed_mask)

    save_raster(
        output / "road_probability.tif",
        probability,
        transform,
        crs,
        "float32",
    )
    save_raster(
        output / "road_mask_raw.tif",
        raw_mask.astype(np.uint8),
        transform,
        crs,
        "uint8",
    )
    save_raster(
        output / "road_mask_healed.tif",
        healed_mask.astype(np.uint8),
        transform,
        crs,
        "uint8",
    )
    Image.fromarray((healed_mask * 255).astype(np.uint8)).save(
        output / "road_mask_healed.png"
    )

    resolution = float(abs(transform.a))
    graph = graph_from_mask(
        healed_mask,
        resolution_m=resolution,
        origin_xy=(transform.c, transform.f),
    )
    graph.graph["crs"] = crs
    graph.graph["healing_experiment"] = args.experiment
    graph_to_geojson(graph, output / "road_graph.geojson")
    write_graphml(graph, output / "road_graph.graphml")

    if bridges:
        with (output / "accepted_bridges.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(bridges[0]))
            writer.writeheader()
            writer.writerows(bridges)
    summary = {
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "experiment": args.experiment,
        "crs": crs,
        "shape": list(raw_mask.shape),
        "components_before": before_components,
        "components_after": after_components,
        "largest_component_before": before_largest,
        "largest_component_after": after_largest,
        "bridges_added": len(bridges),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "model_manifest": model_config["manifest"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
