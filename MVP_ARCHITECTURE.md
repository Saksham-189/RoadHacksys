# Route Resilience MVP Architecture

## Runtime

```text
Browser
  |-- local Leaflet map and controls
  |-- tile inference diagnostics
  |-- baseline/disrupted metrics
  |
FastAPI /api/v1
  |-- ArtifactRegistry
  |-- LayerService (UTM -> cached WGS84)
  |-- InferenceService (E013, CUDA with CPU fallback)
  |-- ScenarioService (preview and exact MSA)
  |-- JobService (single exact worker)
  |
Verified artifacts
  |-- E013 checkpoint and test manifest
  |-- Part 3 transport graph and gravity demand
  |-- Part 4 baseline cache and D001-D009
```

FastAPI serves both the API and static interface. The application requires no
database, authentication, Node build or internet connection.

## Data boundaries

- Simulation remains in `EPSG:32643` so radii and lengths use metres.
- Browser layers are cached in WGS84 under `runs/mvp/cache/layers`.
- City-scale graph and demand are precomputed.
- Live E013 inference is limited to held-out manifest tiles.
- Interactive outputs are isolated under `runs/mvp/sessions/{uuid}`.

## Reliability

- Baseline MSA is reconstructed from its verified cache at startup.
- Source fingerprints invalidate stale baseline and map caches.
- Exact simulations run one at a time.
- The baseline graph is checked before and after each simulation.
- Preset scenarios remain immediately available if live inference is degraded.
