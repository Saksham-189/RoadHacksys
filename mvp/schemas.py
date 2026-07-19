from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class EdgeRef(BaseModel):
    source: str
    target: str


class CapacityEdge(EdgeRef):
    capacity_factor: float = Field(gt=0, le=1)


class CloseNodesAction(BaseModel):
    action: Literal["close_nodes"]
    node_ids: list[str] = Field(min_length=1)


class CloseEdgesAction(BaseModel):
    action: Literal["close_edges"]
    edges: list[EdgeRef] = Field(min_length=1)


class CapacityAction(BaseModel):
    action: Literal["capacity_derating"]
    edges: list[CapacityEdge] = Field(min_length=1)


class CircleAction(BaseModel):
    action: Literal["close_circle"]
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    radius_m: float = Field(ge=100, le=1500)


ScenarioAction = Annotated[
    Union[
        CloseNodesAction,
        CloseEdgesAction,
        CapacityAction,
        CircleAction,
    ],
    Field(discriminator="action"),
]


class SimulationRequest(BaseModel):
    name: str = Field(default="Interactive disruption", max_length=120)
    hazard_type: str = Field(default="interactive", max_length=60)
    actions: list[ScenarioAction] = Field(min_length=1, max_length=20)


class InferenceRequest(BaseModel):
    tile_id: str
    occlusion: Literal[
        "none", "trees", "cloud", "shadow", "cutout", "vehicles", "haze"
    ] = "none"
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]

