from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.morphology import (
    closing,
    disk,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
)

NEIGHBORS = tuple(
    (row, column)
    for row in (-1, 0, 1)
    for column in (-1, 0, 1)
    if (row, column) != (0, 0)
)


def clean_mask(
    mask: np.ndarray,
    minimum_object_size: int = 8,
    maximum_hole_size: int = 16,
    closing_radius: int = 0,
) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool)
    cleaned = remove_small_objects(
        cleaned, max_size=max(minimum_object_size - 1, 0)
    )
    cleaned = remove_small_holes(
        cleaned, max_size=max(maximum_hole_size - 1, 0)
    )
    if closing_radius > 0:
        cleaned = closing(cleaned, footprint=disk(closing_radius))
    return np.asarray(cleaned, dtype=bool)


def skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    return skeletonize(np.asarray(mask, dtype=bool))


def neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant")


def _pixel_neighbors(
    pixel: tuple[int, int], pixels: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    row, column = pixel
    return [
        (row + delta_row, column + delta_column)
        for delta_row, delta_column in NEIGHBORS
        if (row + delta_row, column + delta_column) in pixels
    ]


def graph_from_skeleton(
    skeleton: np.ndarray,
    resolution_m: float = 10.0,
    origin_xy: tuple[float, float] | None = None,
) -> nx.MultiGraph:
    skeleton = np.asarray(skeleton, dtype=bool)
    pixels = {tuple(pixel) for pixel in np.argwhere(skeleton)}
    graph = nx.MultiGraph()
    graph.graph["resolution_m"] = float(resolution_m)
    if not pixels:
        return graph

    counts = neighbor_count(skeleton)
    anchor_mask = skeleton & (counts != 2)
    labels, anchor_count = ndimage.label(
        anchor_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )

    # A pure loop contains no natural endpoint or intersection.
    skeleton_labels, component_count = ndimage.label(
        skeleton, structure=np.ones((3, 3), dtype=np.uint8)
    )
    for component in range(1, component_count + 1):
        component_pixels = np.argwhere(skeleton_labels == component)
        if not np.any(anchor_mask[skeleton_labels == component]):
            row, column = component_pixels[0]
            anchor_count += 1
            labels[row, column] = anchor_count
            anchor_mask[row, column] = True

    pixel_to_node: dict[tuple[int, int], int] = {}
    for label in range(1, anchor_count + 1):
        coordinates = np.argwhere(labels == label)
        if coordinates.size == 0:
            continue
        node_id = len(graph)
        row, column = coordinates.mean(axis=0)
        x = float(column * resolution_m)
        y = float(-row * resolution_m)
        if origin_xy is not None:
            x += origin_xy[0]
            y += origin_xy[1]
        graph.add_node(
            node_id,
            row=float(row),
            col=float(column),
            x=x,
            y=y,
            pixel_count=int(len(coordinates)),
            node_type="endpoint" if len(coordinates) == 1 and counts[tuple(coordinates[0])] <= 1 else "junction",
        )
        for coordinate in coordinates:
            pixel_to_node[tuple(coordinate)] = node_id

    visited: set[frozenset[tuple[int, int]]] = set()
    for start_pixel, start_node in list(pixel_to_node.items()):
        for first in _pixel_neighbors(start_pixel, pixels):
            if pixel_to_node.get(first) == start_node:
                continue
            first_segment = frozenset((start_pixel, first))
            if first_segment in visited:
                continue
            visited.add(first_segment)
            path = [start_pixel, first]
            previous, current = start_pixel, first
            while current not in pixel_to_node:
                options = [
                    neighbor
                    for neighbor in _pixel_neighbors(current, pixels)
                    if neighbor != previous
                ]
                if not options:
                    break
                next_pixel = options[0]
                segment = frozenset((current, next_pixel))
                if segment in visited:
                    break
                visited.add(segment)
                path.append(next_pixel)
                previous, current = current, next_pixel
            if current not in pixel_to_node:
                continue
            end_node = pixel_to_node[current]
            length_pixels = sum(
                math.hypot(
                    second[0] - first_pixel[0],
                    second[1] - first_pixel[1],
                )
                for first_pixel, second in zip(path, path[1:])
            )
            graph.add_edge(
                start_node,
                end_node,
                length_m=float(length_pixels * resolution_m),
                pixels=[(int(row), int(column)) for row, column in path],
                source_type="predicted",
                healing_confidence=1.0,
            )
    return graph


def graph_from_mask(
    mask: np.ndarray,
    resolution_m: float = 10.0,
    origin_xy: tuple[float, float] | None = None,
) -> nx.MultiGraph:
    return graph_from_skeleton(
        skeleton_from_mask(mask), resolution_m=resolution_m, origin_xy=origin_xy
    )


def graph_to_geojson(graph: nx.MultiGraph, path: str | Path) -> None:
    features = []
    for node, data in graph.nodes(data=True):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [data["x"], data["y"]],
                },
                "properties": {
                    "feature": "node",
                    "node_id": int(node),
                    "node_type": data.get("node_type", "unknown"),
                },
            }
        )
    resolution = graph.graph.get("resolution_m", 10.0)
    for source, target, key, data in graph.edges(keys=True, data=True):
        coordinates = []
        for row, column in data.get("pixels", []):
            source_node = graph.nodes[source]
            origin_x = source_node["x"] - source_node["col"] * resolution
            origin_y = source_node["y"] + source_node["row"] * resolution
            coordinates.append(
                [origin_x + column * resolution, origin_y - row * resolution]
            )
        if len(coordinates) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "feature": "edge",
                    "source": int(source),
                    "target": int(target),
                    "key": int(key),
                    "length_m": data.get("length_m", 0.0),
                    "source_type": data.get("source_type", "predicted"),
                    "healing_confidence": data.get("healing_confidence", 1.0),
                },
            }
        )
    Path(path).write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )
