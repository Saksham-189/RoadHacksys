import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mvp.app import create_app
from mvp.artifacts import ArtifactRegistry
from mvp.config import load_mvp_config


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(warm_model=False)) as value:
        yield value


def test_artifact_registry_is_ready():
    registry = ArtifactRegistry(load_mvp_config())
    assert registry.status.status == "ready"
    assert registry.status.fingerprint
    assert registry.status.details["graph_nodes"] == 3345
    assert registry.status.details["demand_pairs"] == 2000


def test_health_and_bootstrap_contract(client):
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] in {"ready", "degraded"}
    bootstrap = client.get("/api/v1/bootstrap")
    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    assert payload["baseline"]["served_demand_ratio"] == 1
    assert len(payload["scenarios"]) == 9
    assert payload["capabilities"]["offline"] is True


def test_web_layers_use_valid_wgs84(client):
    response = client.get("/api/v1/layers/node_criticality")
    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    for feature in payload["features"][::250]:
        longitude, latitude = feature["geometry"]["coordinates"]
        assert -180 <= longitude <= 180
        assert -90 <= latitude <= 90


def test_d002_regression(client):
    response = client.get("/api/v1/scenarios/D002")
    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["disconnected_demand_ratio"] == pytest.approx(
        0.19288153356339066, abs=1e-6
    )
    assert summary["service_adjusted_resilience"] == pytest.approx(
        0.8071184664366093, abs=1e-6
    )


def test_invalid_requests_are_rejected(client):
    response = client.post(
        "/api/v1/simulations/preview",
        json={
            "name": "Invalid",
            "hazard_type": "test",
            "actions": [
                {
                    "action": "close_circle",
                    "longitude": 77.5,
                    "latitude": 13,
                    "radius_m": -1,
                }
            ],
        },
    )
    assert response.status_code == 422
    assert client.get("/api/v1/scenarios/D999").status_code == 404
    assert client.get("/api/v1/layers/unknown").status_code == 404


def test_preview_recreates_d001_and_preserves_baseline(client):
    layer = client.get("/api/v1/layers/node_criticality").json()
    node = next(
        feature
        for feature in layer["features"]
        if feature["properties"]["node_id"] == "3900"
    )
    longitude, latitude = node["geometry"]["coordinates"]
    before = client.app.state.scenarios.graph_signature()
    response = client.post(
        "/api/v1/simulations/preview",
        json={
            "name": "Controlled D001",
            "hazard_type": "flood",
            "actions": [
                {
                    "action": "close_circle",
                    "longitude": longitude,
                    "latitude": latitude,
                    "radius_m": 250,
                }
            ],
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["summary"]["closed_nodes"] == 3
    assert result["summary"]["service_adjusted_resilience"] == pytest.approx(
        0.8182018021694735
    )
    assert result["summary"]["runtime_seconds"] < 5
    assert client.app.state.scenarios.graph_signature() == before
    artifact = client.get(
        result["artifacts"]["affected_zones.geojson"]
    )
    assert artifact.status_code == 200
    assert artifact.json()["type"] == "FeatureCollection"


def test_exact_job_transitions_to_complete(client):
    response = client.post(
        "/api/v1/simulations/exact",
        json={
            "name": "Capacity test",
            "hazard_type": "construction",
            "actions": [
                {
                    "action": "capacity_derating",
                    "edges": [
                        {
                            "source": "3905",
                            "target": "3906",
                            "capacity_factor": 0.5,
                        }
                    ],
                }
            ],
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    terminal = None
    for _ in range(80):
        terminal = client.get(f"/api/v1/jobs/{job_id}").json()
        if terminal["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert terminal is not None
    assert terminal["status"] == "completed", terminal
    assert terminal["result"]["summary"]["converged"] is True


def test_inference_unknown_tile_and_fallback(client):
    unknown = client.post(
        "/api/v1/inference",
        json={"tile_id": "unknown", "occlusion": "none", "seed": 42},
    )
    assert unknown.status_code == 422
    tiles = client.get("/api/v1/inference/tiles")
    assert tiles.status_code == 200
    assert len(tiles.json()) == 304
    fallback = client.post(
        "/api/v1/inference",
        json={
            "tile_id": "bengaluru_edge_00303",
            "occlusion": "cloud",
            "seed": 12345,
        },
    )
    assert fallback.status_code == 200
    assert fallback.json()["mode"] == "cached_fallback"


def test_targeted_failure_is_worse_than_random(client):
    targeted = client.get("/api/v1/scenarios/D007").json()["summary"]
    random = client.get("/api/v1/scenarios/D009").json()["summary"]
    assert (
        targeted["service_adjusted_resilience"]
        <= random["service_adjusted_resilience"]
    )
