from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.graph import route_through_array
from skimage.morphology import closing, dilation, disk

from .extract import NEIGHBORS, neighbor_count, skeleton_from_mask


@dataclass(frozen=True)
class HealingConfig:
    experiment: str
    closing_radius: int = 0
    maximum_gap_pixels: float = 12.0
    maximum_angle_degrees: float = 35.0
    use_angle: bool = False
    use_astar: bool = False
    use_mst: bool = False
    bridge_radius: int = 1
    minimum_path_confidence: float = 0.08


@dataclass
class Candidate:
    first: tuple[int, int]
    second: tuple[int, int]
    first_component: int
    second_component: int
    distance: float
    angle: float
    confidence: float
    cost: float


class DisjointSet:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: int) -> int:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: int, second: int) -> bool:
        first_root, second_root = self.find(first), self.find(second)
        if first_root == second_root:
            return False
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1
        return True


def _neighbors(
    pixel: tuple[int, int], skeleton: np.ndarray
) -> list[tuple[int, int]]:
    row, column = pixel
    height, width = skeleton.shape
    return [
        (row + delta_row, column + delta_column)
        for delta_row, delta_column in NEIGHBORS
        if 0 <= row + delta_row < height
        and 0 <= column + delta_column < width
        and skeleton[row + delta_row, column + delta_column]
    ]


def _endpoint_direction(
    endpoint: tuple[int, int], skeleton: np.ndarray, lookback: int = 7
) -> np.ndarray:
    path = [endpoint]
    previous = None
    current = endpoint
    for _ in range(lookback):
        options = [
            neighbor
            for neighbor in _neighbors(current, skeleton)
            if neighbor != previous
        ]
        if not options:
            break
        next_pixel = options[0]
        path.append(next_pixel)
        previous, current = current, next_pixel
    if len(path) < 2:
        return np.zeros(2, dtype=np.float32)
    return np.asarray(path[0], dtype=np.float32) - np.asarray(
        path[-1], dtype=np.float32
    )


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 180.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1, 1))
    return math.degrees(math.acos(cosine))


def _straight_confidence(
    probability: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
) -> float:
    rows, columns = line(first[0], first[1], second[0], second[1])
    return float(probability[rows, columns].mean())


def generate_candidates(
    skeleton: np.ndarray,
    probability: np.ndarray,
    maximum_gap_pixels: float,
    maximum_angle_degrees: float,
    use_angle: bool,
) -> tuple[list[Candidate], np.ndarray]:
    components, _ = ndimage.label(
        skeleton, structure=np.ones((3, 3), dtype=np.uint8)
    )
    endpoints = [
        tuple(pixel)
        for pixel in np.argwhere(skeleton & (neighbor_count(skeleton) == 1))
    ]
    directions = {
        endpoint: _endpoint_direction(endpoint, skeleton) for endpoint in endpoints
    }
    candidates = []
    if len(endpoints) < 2:
        return [], components
    endpoint_array = np.asarray(endpoints, dtype=np.float32)
    nearby_pairs = cKDTree(endpoint_array).query_pairs(maximum_gap_pixels)
    for first_index, second_index in sorted(nearby_pairs):
        first = endpoints[first_index]
        second = endpoints[second_index]
        first_component = int(components[first])
        second_component = int(components[second])
        if first_component == second_component:
            continue
        delta = np.asarray(second, dtype=np.float32) - np.asarray(
            first, dtype=np.float32
        )
        distance = float(np.linalg.norm(delta))
        if distance > maximum_gap_pixels or distance < 1.5:
            continue
        first_angle = _angle_between(directions[first], delta)
        second_angle = _angle_between(directions[second], -delta)
        angle = max(first_angle, second_angle)
        if use_angle and angle > maximum_angle_degrees:
            continue
        confidence = _straight_confidence(probability, first, second)
        cost = (
            0.50 * distance / maximum_gap_pixels
            + 0.35 * angle / 180.0
            + 0.15 * (1.0 - confidence)
        )
        candidates.append(
            Candidate(
                first,
                second,
                first_component,
                second_component,
                distance,
                angle,
                confidence,
                cost,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.cost), components


def _candidate_path(
    candidate: Candidate,
    probability: np.ndarray,
    skeleton: np.ndarray,
    use_astar: bool,
) -> list[tuple[int, int]]:
    if not use_astar:
        rows, columns = line(
            candidate.first[0],
            candidate.first[1],
            candidate.second[0],
            candidate.second[1],
        )
        return list(zip(rows.tolist(), columns.tolist()))

    margin = max(6, int(candidate.distance * 0.6))
    top = max(0, min(candidate.first[0], candidate.second[0]) - margin)
    bottom = min(
        probability.shape[0], max(candidate.first[0], candidate.second[0]) + margin + 1
    )
    left = max(0, min(candidate.first[1], candidate.second[1]) - margin)
    right = min(
        probability.shape[1], max(candidate.first[1], candidate.second[1]) + margin + 1
    )
    local_probability = probability[top:bottom, left:right]
    cost = 0.15 + (1.0 - local_probability) ** 2
    local_skeleton = skeleton[top:bottom, left:right]
    cost[local_skeleton] = 0.05
    start = (candidate.first[0] - top, candidate.first[1] - left)
    end = (candidate.second[0] - top, candidate.second[1] - left)
    path, _ = route_through_array(
        cost, start, end, fully_connected=True, geometric=True
    )
    return [(row + top, column + left) for row, column in path]


def heal_mask(
    mask: np.ndarray,
    probability: np.ndarray,
    config: HealingConfig,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    base = np.asarray(mask, dtype=bool)
    if config.closing_radius:
        base = closing(base, footprint=disk(config.closing_radius))
    if config.experiment in {"B001", "B002"}:
        return base, []

    skeleton = skeleton_from_mask(base)
    candidates, components = generate_candidates(
        skeleton,
        probability,
        config.maximum_gap_pixels,
        config.maximum_angle_degrees,
        config.use_angle,
    )
    component_ids = sorted(int(value) for value in np.unique(components) if value)
    disjoint_set = DisjointSet(component_ids)
    used_endpoints: set[tuple[int, int]] = set()
    additions = np.zeros_like(base)
    accepted = []
    for candidate in candidates:
        if candidate.confidence < config.minimum_path_confidence:
            continue
        if config.use_mst:
            if not disjoint_set.union(
                candidate.first_component, candidate.second_component
            ):
                continue
        elif (
            candidate.first in used_endpoints
            or candidate.second in used_endpoints
        ):
            continue
        path = _candidate_path(
            candidate, probability, skeleton | additions, config.use_astar
        )
        path_mask = np.zeros_like(base)
        rows, columns = zip(*path)
        path_mask[rows, columns] = True
        if config.bridge_radius:
            path_mask = dilation(
                path_mask, footprint=disk(config.bridge_radius)
            )
        additions |= path_mask
        used_endpoints.update((candidate.first, candidate.second))
        accepted.append(
            {
                "first": candidate.first,
                "second": candidate.second,
                "distance_pixels": candidate.distance,
                "angle_degrees": candidate.angle,
                "mean_probability": candidate.confidence,
                "cost": candidate.cost,
                "source_type": "astar" if config.use_astar else "straight",
            }
        )
    return base | additions, accepted


EXPERIMENTS = {
    "B001": HealingConfig("B001"),
    "B002": HealingConfig("B002", closing_radius=2),
    "B003": HealingConfig("B003", maximum_gap_pixels=12),
    "B004": HealingConfig(
        "B004", maximum_gap_pixels=12, use_angle=True
    ),
    "B005": HealingConfig(
        "B005", maximum_gap_pixels=14, use_angle=True, use_astar=True
    ),
    "B006": HealingConfig(
        "B006", maximum_gap_pixels=14, use_angle=True, use_mst=True
    ),
    "B007": HealingConfig(
        "B007",
        closing_radius=0,
        maximum_gap_pixels=16,
        maximum_angle_degrees=40,
        use_angle=True,
        use_astar=True,
        use_mst=True,
    ),
}
