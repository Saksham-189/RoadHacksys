from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import dilation, disk

from .extract import graph_from_mask, neighbor_count, skeleton_from_mask


def create_synthetic_gaps(
    original: np.ndarray,
    seed: int,
    gap_count: int = 3,
    minimum_radius: int = 3,
    maximum_radius: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = np.asarray(original, dtype=bool)
    skeleton = skeleton_from_mask(original)
    counts = neighbor_count(skeleton)
    candidates = np.argwhere(skeleton & (counts == 2))
    candidates = [
        tuple(pixel)
        for pixel in candidates
        if 12 <= pixel[0] < original.shape[0] - 12
        and 12 <= pixel[1] < original.shape[1] - 12
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    removed_region = np.zeros_like(original)
    centers: list[tuple[int, int]] = []
    for center in candidates:
        if any(math.dist(center, existing) < 30 for existing in centers):
            continue
        radius = rng.randint(minimum_radius, maximum_radius)
        stamp = np.zeros_like(original)
        stamp[center] = True
        stamp = dilation(stamp, footprint=disk(radius))
        gap_pixels = original & stamp
        if gap_pixels.sum() < 3:
            continue
        removed_region |= stamp
        centers.append(center)
        if len(centers) >= gap_count:
            break
    broken = original & ~removed_region
    gap_target = original & removed_region

    probability = ndimage.gaussian_filter(broken.astype(np.float32), sigma=1.1)
    faint_strength = rng.uniform(0.16, 0.32)
    probability = np.maximum(
        probability, ndimage.gaussian_filter(gap_target.astype(np.float32), 1.2) * faint_strength
    )
    noise = np.random.default_rng(seed).normal(0, 0.025, original.shape)
    probability = np.clip(probability + noise, 0, 1).astype(np.float32)
    return broken, gap_target, probability


def component_statistics(mask: np.ndarray) -> tuple[int, float]:
    labels, count = ndimage.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    total = float(mask.sum())
    if count == 0 or total == 0:
        return int(count), 0.0
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return int(count), float(np.max(sizes) / total)


def _nearest_nodes(
    source_graph: nx.MultiGraph, target_graph: nx.MultiGraph
) -> dict[int, int]:
    if not target_graph.nodes:
        return {}
    target_nodes = list(target_graph.nodes)
    coordinates = np.asarray(
        [
            (target_graph.nodes[node]["row"], target_graph.nodes[node]["col"])
            for node in target_nodes
        ]
    )
    tree = cKDTree(coordinates)
    mapping = {}
    for node, data in source_graph.nodes(data=True):
        _, index = tree.query((data["row"], data["col"]))
        mapping[node] = target_nodes[int(index)]
    return mapping


def route_metrics(
    original: np.ndarray,
    healed: np.ndarray,
    seed: int,
    pair_count: int = 12,
) -> tuple[float, float]:
    original_graph = graph_from_mask(original, resolution_m=10)
    healed_graph = graph_from_mask(healed, resolution_m=10)
    if len(original_graph) < 2 or len(healed_graph) < 2:
        return 0.0, 1.0
    components = list(nx.connected_components(nx.Graph(original_graph)))
    largest = max(components, key=len)
    nodes = sorted(largest)
    if len(nodes) < 2:
        return 0.0, 1.0
    mapping = _nearest_nodes(original_graph, healed_graph)
    rng = random.Random(seed)
    possible = [(first, second) for i, first in enumerate(nodes) for second in nodes[i + 1 :]]
    rng.shuffle(possible)
    pairs = possible[:pair_count]
    successful = 0
    errors = []
    for first, second in pairs:
        try:
            original_length = nx.shortest_path_length(
                original_graph, first, second, weight="length_m"
            )
            healed_first, healed_second = mapping[first], mapping[second]
            healed_length = nx.shortest_path_length(
                healed_graph, healed_first, healed_second, weight="length_m"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        successful += 1
        errors.append(
            abs(healed_length - original_length) / max(original_length, 1)
        )
    if not pairs:
        return 0.0, 1.0
    return successful / len(pairs), float(np.mean(errors)) if errors else 1.0


def evaluate_healing(
    original: np.ndarray,
    broken: np.ndarray,
    healed: np.ndarray,
    gap_target: np.ndarray,
    seed: int,
) -> dict[str, float]:
    components_before, largest_before = component_statistics(broken)
    components_after, largest_after = component_statistics(healed)
    added = healed & ~broken
    recovered = added & gap_target
    gap_recovery = float(recovered.sum() / max(gap_target.sum(), 1))
    tolerated_original = dilation(original, footprint=disk(2))
    false_pixels = added & ~tolerated_original
    false_bridge_rate = float(false_pixels.sum() / max(added.sum(), 1))
    route_success, path_error = route_metrics(original, healed, seed)
    original_length = max(skeleton_from_mask(original).sum(), 1)
    healed_length = skeleton_from_mask(healed).sum()
    network_length_error = float(
        abs(float(healed_length) - float(original_length)) / original_length
    )
    connectivity_gain = max(
        0.0,
        (largest_after - largest_before) / max(1.0 - largest_before, 1e-6),
    )
    score = (
        0.25 * gap_recovery
        + 0.25 * route_success
        + 0.20 * min(connectivity_gain, 1.0)
        + 0.15 * (1.0 - min(path_error, 1.0))
        + 0.15 * (1.0 - false_bridge_rate)
    )
    return {
        "components_before": float(components_before),
        "components_after": float(components_after),
        "largest_component_before": largest_before,
        "largest_component_after": largest_after,
        "connectivity_gain": connectivity_gain,
        "gap_recovery_rate": gap_recovery,
        "false_bridge_rate": false_bridge_rate,
        "route_success_rate": route_success,
        "path_length_error": path_error,
        "network_length_error": network_length_error,
        "healing_score": score,
    }
