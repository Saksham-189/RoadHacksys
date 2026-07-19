# E012: Occlusion-Robust RGBN Road Segmentation

This project trains an occlusion-robust road segmenter using Sentinel-2 red,
green, blue, and near-infrared bands. The final E013 model fine-tunes NVIDIA's
ADE20K-pretrained SegFormer-B0. Its RGB input projection is expanded to RGBN by
initializing the NIR weights from the mean pretrained RGB weights. Synthetic
tree, cloud, shadow, cutout, vehicle, and haze occlusions are applied only to
training imagery while road masks remain unchanged.

## Dataset protocol

- Train: 1,804 tiles from 13 AOIs
- Validation: 460 tiles from Hyderabad, Chandigarh, Indore, and Mysuru
- Test: 304 Bengaluru edge tiles
- Tile size: 256 x 256 at 10 m Sentinel-2 resolution
- Labels: rasterized OpenStreetMap roads

The geographic split is intentionally retained to measure cross-region
generalization. Test data is never used for checkpoint or threshold selection.

## Commands

```powershell
.\.venv-win\Scripts\python.exe scripts\expand_sentinel_dataset.py
.\.venv-win\Scripts\python.exe scripts\build_final_manifest.py
.\.venv-win\Scripts\python.exe validate_data.py
.\.venv-win\Scripts\python.exe -m unittest discover tests -v
.\.venv-win\Scripts\python.exe train.py --config configs/e013_pretrained_segformer_rgbn.yaml
.\.venv-win\Scripts\python.exe calibrate_threshold.py
.\.venv-win\Scripts\python.exe evaluate.py --checkpoint runs\e013_pretrained_segformer_rgbn\best.pt
```

Training writes the best checkpoint, history, and summary under
`runs/e013_pretrained_segformer_rgbn/`. Evaluation writes clean and occluded
metrics plus visual prediction examples to the same directory.

## Final score

The held-out score is:

```text
0.20 IoU + 0.20 Dice + 0.15 Recall + 0.20 Occlusion Recall
+ 0.15 Connectivity Ratio + 0.10 Relaxed IoU
```

## Latest measured run: E013

The RTX 3050 run fine-tuned for 20 epochs and selected epoch 17. Threshold
calibration used the complete final-score formula on validation data and froze
the conservative lower bound of 0.10 before testing.

| Held-out metric | Value |
|---|---:|
| Clean IoU | 0.3003 |
| Clean Dice | 0.4619 |
| Clean Precision | 0.3275 |
| Clean Recall | 0.7832 |
| Clean Relaxed IoU | 0.3261 |
| Occluded IoU | 0.2962 |
| Occluded Dice | 0.4570 |
| Occlusion Recall | 0.7795 |
| Final Score under the newer calibrated protocol | 0.4934 |

These values come from all 304 Bengaluru-edge test tiles. The broad predictions
show that resolving thin OSM centerlines at 10 m remains a limitation. The
0.10 score-calibrated threshold favors recall and connectivity; the
Dice-optimal threshold of 0.425 gives higher precision but a lower aggregate
score under the specified formula.

## Protocol-safe comparison

The original E001-E012 experiments used a fixed threshold of 0.5, asymmetric
two-pixel relaxed IoU, largest-predicted-component connectivity, and the
original deterministic occlusion generator. That evaluator was recovered
exactly as `legacy_v1` and must be used for comparisons with the historical
0.5734 result.

| Model | IoU | Dice | Precision | Recall | Occlusion Recall | Connectivity | Legacy Final Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| E012 historical | 0.2058 | 0.3413 | 0.2157 | 0.8178 | 0.9179 | 0.8711 | **0.5734** |
| E013 pretrained | 0.2730 | 0.4290 | 0.5314 | 0.3596 | 0.2807 | 0.4504 | 0.3592 |

E012 remains the official winner under the original experiment contract.
E013 improves boundary and pixel precision but does not preserve the
high-recall, occlusion-robust behavior rewarded by the final score.

Run the exact historical evaluator with:

```powershell
.\.venv-win\Scripts\python.exe evaluate_legacy.py
```

## Part 2: Graph Skeletonization and Healing

The Part 2 implementation converts binary masks into compressed NetworkX
graphs and compares seven healing configurations using controlled deletions
from OSM road masks.

```powershell
.\.venv-win\Scripts\python.exe run_part2_experiments.py --tiles 40 --gaps 3
.\.venv-win\Scripts\python.exe generate_part2_graph.py --experiment B007
```

Measured benchmark results:

| Rank | Experiment | Healing Score | False Bridge Rate | Route Success | Path Error |
|---:|---|---:|---:|---:|---:|
| 1 | B003 distance matching | 0.6676 | 0.1827 | 0.9563 | 0.0383 |
| 2 | B007 constrained hybrid | 0.6300 | 0.0771 | 0.9375 | 0.0229 |
| 3 | B005 direction-aware A* | 0.6178 | 0.0725 | 0.9375 | 0.0215 |

B003 wins the declared composite score. B007 is retained as the safer
planner-facing option because it cuts false bridges by more than half while
keeping network-length error below 1%.

The real Bengaluru-edge B007 export contains 5,614 nodes and 5,300 edges. It
reduces predicted-mask components from 914 to 752 and raises the largest
component ratio from 0.6720 to 0.7319. The graph pipeline is model-agnostic;
its real export currently uses the available E013 probability mosaic because
the historical E012 checkpoint was deleted during the requested clean rebuild.

## Part 3: Structural Intelligence and Stress Testing

Part 3 implements degree centrality as the explainable baseline, approximate
weighted betweenness as the required gatekeeper reference, and a flow-aware
advanced model using gravity demand, BPR congestion and method-of-successive-
averages traffic assignment.

```powershell
.\.venv-win\Scripts\python.exe consolidate_part2_graph.py --config configs\part3_flow.yaml
.\.venv-win\Scripts\python.exe prepare_transport_graph.py --config configs\part3_flow.yaml
.\.venv-win\Scripts\python.exe run_part3.py --config configs\part3_flow.yaml
```

The safe consolidation search tested 36 configurations. No configuration met
the requested 70% node-coverage target while staying below 10% false bridges.
The selected graph therefore uses the largest safe component:

```text
node coverage: 63.87%
false-bridge rate: 7.40%
routing graph: 3,345 nodes / 3,749 edges
```

The gravity model contains 2,000 OD pairs. MSA converged in 173 iterations to
a relative gap of 0.000914. Capacities and flows are relative satellite-derived
estimates and must not be interpreted as measured vehicle counts.

Resilience after removing 10% of nodes:

| Removal strategy | Resilience Index |
|---|---:|
| Random | 0.5523 |
| Highest degree | 0.1435 |
| Highest betweenness | 0.1630 |
| Highest flow criticality | **0.1285** |

The advanced flow-critical ranking finds the most damaging targeted failures,
while degree centrality remains the simpler baseline. Detailed results and
limitations are documented in
`runs/part3/PART3_IMPLEMENTATION_REPORT.md`.
# Phase 4: disruption simulation

Run one JSON-defined scenario:

```powershell
.\.venv-win\Scripts\python.exe simulate_disruption.py `
  --scenario configs\scenarios\D001.json `
  --mode exact
```

Generate and run the complete D001-D009 preview and exact suite:

```powershell
.\.venv-win\Scripts\python.exe run_phase4_scenarios.py `
  --config configs\phase4.yaml
```

Results are written to `runs/part4`, with one artifact directory per scenario,
a master scoreboard, comparison figures, and the implementation report.
