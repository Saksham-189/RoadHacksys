# Large Artifact Bundles

The GitHub repository stores large runtime artifacts through Git LFS under the
`artifacts/` directory. These files are not ordinary source files; they are
zip bundles used to restore the demo state after cloning the repository.

## Bundles

| File | Purpose |
|---|---|
| `artifacts/e013_checkpoint_and_examples.zip` | E013 SegFormer checkpoint, metrics, threshold calibration and example predictions. |
| `artifacts/processed_dataset_final_tiles.zip` | Processed final Sentinel-2 RGB/RGBN tiles, masks, manifests and AOI metadata needed for training/evaluation without rebuilding tiles. |
| `artifacts/part3_graph_and_flow_outputs.zip` | Consolidated graph, transport graph, demand, centrality, flow and stress-test outputs. |
| `artifacts/part4_disruption_outputs.zip` | Baseline route cache, D001-D009 disruption outputs, scenario scoreboard and GeoJSON result artifacts. |
| `artifacts/mvp_runtime_cache.zip` | MVP cache/session/inference outputs for fast offline demo startup. |

## Restore

After cloning the repository with Git LFS enabled, extract each zip at the
repository root. The folders inside the zips preserve their original paths:

```powershell
Expand-Archive artifacts\e013_checkpoint_and_examples.zip -DestinationPath . -Force
Expand-Archive artifacts\processed_dataset_final_tiles.zip -DestinationPath . -Force
Expand-Archive artifacts\part3_graph_and_flow_outputs.zip -DestinationPath . -Force
Expand-Archive artifacts\part4_disruption_outputs.zip -DestinationPath . -Force
Expand-Archive artifacts\mvp_runtime_cache.zip -DestinationPath . -Force
```

Then run:

```powershell
.\setup_mvp.ps1
.\run_mvp.ps1
```

The raw Sentinel-2 scenes under `data/raw/` are not bundled because the final
processed tiles and manifests are enough to resume model evaluation, graph
analysis and the MVP demo.
