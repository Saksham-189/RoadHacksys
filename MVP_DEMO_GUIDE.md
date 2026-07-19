# Route Resilience MVP Demo Guide

## Start

```powershell
.\run_mvp.ps1
```

Open the printed local URL, normally `http://127.0.0.1:8765`.

## Four-to-five-minute demonstration

1. In **Road Extraction**, select `bengaluru_edge_00303` and run clean E013
   inference. Repeat with a cloud or tree occlusion.
2. Open **Criticality**. Toggle relative flow and critical nodes, then inspect
   node `3900`.
3. Open **Disruption** and load D002. Point out `19.29%` disconnected demand
   and service-adjusted resilience `0.807`.
4. Load D003. Explain that connectivity survives while congested travel time
   rises by `625.34%`.
5. Build an interactive flood: choose **Flood**, set `250 m`, click the map,
   run Preview, then Exact.
6. Close by comparing targeted critical failures with random failure and state
   the limitations: relative traffic, 63.87% routing coverage and pending
   Resourcesat validation.

## Troubleshooting

- **Yellow health indicator:** live model inference is degraded; cached
  diagnostics and all graph scenarios remain available.
- **Missing map:** run `.\setup_mvp.ps1` and verify local Leaflet assets.
- **CUDA error:** inference retries on CPU automatically.
- **Exact job failure:** use Preview or load a cached D001-D009 preset.
- **Port busy:** the launcher automatically checks ports 8765 through 8775.
- **No internet:** expected; the map uses local assets and a neutral background.

## Pre-demo check

```powershell
.\.venv-win\Scripts\python.exe -m mvp.preflight
.\.venv-win\Scripts\python.exe -m pytest -q
```
