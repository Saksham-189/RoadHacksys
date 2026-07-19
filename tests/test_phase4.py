import csv
import json

import networkx as nx
import pytest
import yaml
from pyproj import Transformer

from mobility.scenario import ScenarioSpec, resolve_scenario
from mobility.simulation import SimulationEngine


def toy_graph():
    graph = nx.Graph(crs="EPSG:32643")
    graph.add_node("a", x=0.0, y=0.0)
    graph.add_node("b", x=100.0, y=0.0)
    graph.add_node("c", x=200.0, y=0.0)
    for first, second in (("a", "b"), ("b", "c")):
        graph.add_edge(
            first,
            second,
            length_m=100.0,
            relative_capacity=10.0,
            free_flow_time_min=1.0,
        )
    return graph


def scenario(actions):
    return ScenarioSpec.from_dict(
        {
            "scenario_id": "test",
            "name": "Test scenario",
            "hazard_type": "test",
            "actions": actions,
        }
    )


def test_node_closure_does_not_mutate_baseline():
    graph = toy_graph()
    result = resolve_scenario(
        graph, scenario([{"action": "close_nodes", "node_ids": ["b"]}])
    )
    assert result.graph.number_of_nodes() == 2
    assert result.graph.number_of_edges() == 0
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_capacity_derating_and_validation():
    graph = toy_graph()
    result = resolve_scenario(
        graph,
        scenario(
            [
                {
                    "action": "capacity_derating",
                    "edges": [
                        {
                            "source": "a",
                            "target": "b",
                            "capacity_factor": 0.5,
                        }
                    ],
                }
            ]
        ),
    )
    assert result.graph["a"]["b"]["relative_capacity"] == 5.0
    with pytest.raises(ValueError):
        resolve_scenario(
            graph,
            scenario(
                [
                    {
                        "action": "capacity_derating",
                        "edges": [
                            {
                                "source": "a",
                                "target": "b",
                                "capacity_factor": 0,
                            }
                        ],
                    }
                ]
            ),
        )


def test_unknown_ids_and_bad_circle_rejected():
    graph = toy_graph()
    with pytest.raises(ValueError):
        resolve_scenario(
            graph,
            scenario([{"action": "close_nodes", "node_ids": ["unknown"]}]),
        )
    with pytest.raises(ValueError):
        resolve_scenario(
            graph,
            scenario(
                [{"action": "close_circle", "x": 0, "y": 0, "radius_m": -1}]
            ),
        )


def test_wgs84_polygon_reprojects():
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32643", always_xy=True
    )
    x, y = transformer.transform(77.59, 12.97)
    graph = nx.Graph(crs="EPSG:32643")
    graph.add_node("inside", x=x, y=y)
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [77.589, 12.969],
                [77.591, 12.969],
                [77.591, 12.971],
                [77.589, 12.971],
                [77.589, 12.969],
            ]
        ],
    }
    result = resolve_scenario(
        graph,
        scenario(
            [
                {
                    "action": "close_polygon",
                    "geometry": polygon,
                    "crs": "EPSG:4326",
                }
            ]
        ),
    )
    assert "inside" not in result.graph


def test_empty_and_bridge_scenarios(tmp_path):
    graph = toy_graph()
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)
    demand_path = tmp_path / "demand.csv"
    with demand_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["origin", "destination", "demand"]
        )
        writer.writeheader()
        writer.writerow({"origin": "a", "destination": "c", "demand": 1})
    config = {
        "paths": {
            "graph": str(graph_path),
            "demand": str(demand_path),
            "output": str(tmp_path / "output"),
        },
        "assignment": {
            "bpr_alpha": 0.15,
            "bpr_beta": 4,
            "max_iterations": 20,
            "tolerance": 0.001,
        },
        "impact": {
            "affected_threshold": 0.2,
            "grid_size_m": 500,
            "route_examples": 10,
        },
    }
    engine = SimulationEngine(config)
    empty = engine.simulate(scenario([]))
    assert empty.summary["service_adjusted_resilience"] == pytest.approx(1)
    assert 0 <= empty.summary["path_resilience"] <= 1
    bridge = engine.simulate(
        scenario(
            [
                {
                    "action": "close_edges",
                    "edges": [{"source": "a", "target": "b"}],
                }
            ]
        )
    )
    assert bridge.summary["served_demand_ratio"] == 0
    assert bridge.summary["service_adjusted_resilience"] == 0
    assert bridge.od_impacts[0]["disrupted_distance_m"] is None
