from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mvp.artifacts import ArtifactRegistry
from mvp.config import load_mvp_config
from mvp.inference import InferenceService
from mvp.jobs import JobService
from mvp.layers import CLIENT_LAYERS, LayerService
from mvp.scenario_service import ScenarioService
from mvp.schemas import InferenceRequest, JobResponse, SimulationRequest


def create_app(
    config_path: str | Path = "configs/mvp.yaml",
    *,
    warm_model: bool | None = None,
) -> FastAPI:
    config = load_mvp_config(config_path)
    static_dir = Path(config["paths"]["static"])
    inference_dir = Path(config["paths"]["inference"]).parent

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.perf_counter()
        registry = ArtifactRegistry(config)
        if registry.status.missing:
            raise RuntimeError(
                "MVP artifacts are missing:\n"
                + "\n".join(registry.status.missing)
            )
        layers = LayerService(
            config, registry.status.fingerprint or "unavailable"
        )
        scenarios = ScenarioService(config, layers)
        should_warm = (
            config["inference"]["warm_on_startup"]
            if warm_model is None
            else warm_model
        )
        inference = InferenceService(config, warm=False)
        jobs = JobService(config["server"]["job_timeout_seconds"])
        app.state.config = config
        app.state.registry = registry
        app.state.layers = layers
        app.state.scenarios = scenarios
        app.state.inference = inference
        app.state.jobs = jobs
        app.state.started_seconds = time.perf_counter() - started
        if should_warm:
            threading.Thread(
                target=inference.warm,
                name="mvp-model-warmup",
                daemon=True,
            ).start()
        yield
        jobs.close()

    app = FastAPI(
        title="Route Resilience MVP",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        registry = app.state.registry
        inference = app.state.inference.status()
        status_value = registry.status.status
        warnings = list(registry.status.warnings)
        if not inference["available"]:
            status_value = "degraded"
            warnings.append("Live inference unavailable; cached fallback enabled")
        return {
            "status": status_value,
            "startup_seconds": app.state.started_seconds,
            "artifacts": registry.status.to_dict(),
            "inference": inference,
            "warnings": warnings,
        }

    @app.get("/api/v1/bootstrap")
    def bootstrap() -> dict[str, Any]:
        baseline = json.loads(
            (
                Path(config["paths"]["baseline"])
                / "baseline_summary.json"
            ).read_text(encoding="utf-8")
        )
        return {
            "bounds": app.state.layers.bounds,
            "baseline": baseline,
            "layers": list(CLIENT_LAYERS),
            "scenarios": app.state.scenarios.presets(),
            "resilience_bands": [
                {"minimum": 0.8, "label": "resilient"},
                {"minimum": 0.6, "label": "moderate degradation"},
                {"minimum": 0.4, "label": "vulnerable"},
                {"minimum": 0.2, "label": "severe disruption"},
                {"minimum": 0.0, "label": "systemic failure"},
            ],
            "capabilities": {
                "live_inference": app.state.inference.model is not None,
                "preview": True,
                "exact": True,
                "offline": True,
            },
            "caveats": [
                "Traffic and capacity are relative estimates, not measured counts.",
                "Routing covers 63.87% of estimated graph nodes.",
            ],
        }

    @app.get("/api/v1/layers/{layer_name}")
    def layer(layer_name: str):
        path = app.state.layers.layer_path(layer_name)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="Unknown map layer")
        return FileResponse(path, media_type="application/geo+json")

    @app.get("/api/v1/scenarios")
    def scenarios() -> list[dict[str, Any]]:
        return app.state.scenarios.presets()

    @app.get("/api/v1/scenarios/{scenario_id}")
    def scenario(scenario_id: str) -> dict[str, Any]:
        value = app.state.scenarios.preset(scenario_id)
        if value is None:
            raise HTTPException(status_code=404, detail="Unknown scenario")
        return value

    @app.post("/api/v1/simulations/preview")
    def preview(request: SimulationRequest) -> dict[str, Any]:
        try:
            return app.state.scenarios.preview(request.model_dump())
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post(
        "/api/v1/simulations/exact",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def exact(request: SimulationRequest) -> JobResponse:
        payload = request.model_dump()
        job = app.state.jobs.submit(
            lambda: app.state.scenarios.exact(payload)
        )
        return JobResponse(job_id=job.job_id, status="queued")

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        value = app.state.jobs.get(job_id)
        if value is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return value.to_dict(app.state.jobs.timeout_seconds)

    @app.get("/api/v1/results/{result_id}/{artifact}")
    def result_artifact(result_id: str, artifact: str):
        path = app.state.scenarios.artifact_path(result_id, artifact)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="Unknown result artifact")
        media_type = (
            "application/geo+json"
            if path.suffix == ".geojson"
            else "application/json"
            if path.suffix == ".json"
            else "text/csv"
        )
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
        )

    @app.get("/api/v1/inference/tiles")
    def inference_tiles() -> list[dict[str, Any]]:
        return app.state.inference.tiles()

    @app.post("/api/v1/inference")
    def inference(request: InferenceRequest) -> dict[str, Any]:
        try:
            return app.state.inference.infer(
                request.tile_id, request.occlusion, request.seed
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/fallback/{kind}")
    def fallback(kind: str):
        if kind not in {"clean", "occluded"}:
            raise HTTPException(status_code=404, detail="Unknown fallback")
        path = Path(
            config["inference"][
                "fallback_clean"
                if kind == "clean"
                else "fallback_occluded"
            ]
        )
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        return FileResponse(path, media_type="image/png")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount(
        "/generated",
        StaticFiles(directory=inference_dir),
        name="generated",
    )
    return app


app = create_app()
