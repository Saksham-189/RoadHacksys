# Occlusion-Robust Road Extraction Experiment Report

## Problem Statement

The objective of Part 1 was to extract roads from Sentinel-2 satellite imagery even when roads are partially hidden by shadows, vegetation, vehicles, clouds, or urban clutter. The predicted road mask must be useful for later graph construction, so we evaluated not only pixel accuracy but also recall, occlusion robustness, and connectivity.

## Dataset Preparation

We prepared three Sentinel-2 AOIs using OSM road vectors as reference labels:

- `bengaluru_core`: train split, 124 tiles
- `hyderabad_mixed`: validation split, 16 tiles
- `bengaluru_edge`: held-out test split, 304 tiles

The final merged dataset has 444 image-mask tiles, including 364 road-positive tiles. Each tile is 256 x 256 pixels with stride 128. RGB imagery was used for baseline experiments, while RGBN imagery was used for the final occlusion-aware SegFormer runs.

## Experimental Pipeline

1. Download Sentinel-2 L2A bands B02, B03, B04, and B08.
2. Download OSM road vectors for the same AOIs.
3. Reproject OSM roads to the Sentinel-2 CRS.
4. Buffer OSM road lines by class-dependent road widths.
5. Rasterize roads into binary masks.
6. Tile images and masks into 256 x 256 samples.
7. Train all model families using the same train/validation/test protocol.
8. Evaluate every model on clean and synthetic-occluded test inputs.
9. Rank models using a final score that rewards road continuity and occlusion recovery.

## Models Tested

The experiment covered CNN baselines and transformer/attention-heavy models:

- U-Net
- ResNet U-Net
- UNet++
- DeepLabV3+
- SegFormer
- Swin-Unet
- TransUNet
- Mask2Former
- DINO/ViT Head
- DeepLabV3+ with RGBN and synthetic occlusion training
- SegFormer with RGBN and synthetic occlusion training
- Refined SegFormer with RGBN and synthetic occlusion training

## Metrics

The final score was:

`0.20 * IoU + 0.20 * Dice + 0.15 * Recall + 0.20 * Occlusion Recall + 0.15 * Connectivity Ratio + 0.10 * Relaxed IoU`

This score is appropriate for the hackathon problem because road masks must feed a graph-theoretic pipeline. A model with high visual overlap but broken road continuity is less useful than a model that preserves connected road structures.

## Final Ranking

| rank | exp_id | model | input_bands | iou | dice | recall | occlusion_recall | connectivity_ratio | final_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | E012 | SegFormer | RGBN | 0.205798 | 0.341348 | 0.817789 | 0.917907 | 0.871124 | 0.573358 |
| 2 | E011 | SegFormer | RGBN | 0.237167 | 0.383404 | 0.434683 | 0.596838 | 0.743627 | 0.453476 |
| 3 | E002 | ResNet U-Net | RGB | 0.047497 | 0.090687 | 0.051536 | 0.350083 | 0.420109 | 0.174407 |
| 4 | E005 | SegFormer | RGB | 0.041784 | 0.080217 | 0.045822 | 0.027762 | 0.716104 | 0.150198 |
| 5 | E001 | U-Net | RGB | 0.070739 | 0.132131 | 0.075172 | 0.001935 | 0.455754 | 0.129517 |
| 6 | E004 | DeepLabV3+ | RGB | 0.030576 | 0.059337 | 0.033030 | 0.213953 | 0.315687 | 0.120815 |
| 7 | E003 | UNet++ | RGB | 0.001836 | 0.003666 | 0.001856 | 0.157153 | 0.512862 | 0.110084 |
| 8 | E006 | Swin-Unet | RGB | 0.062520 | 0.117683 | 0.074128 | 0.023139 | 0.253359 | 0.100152 |
| 9 | E009 | DINO/ViT Head | RGB | 0.027467 | 0.053465 | 0.028675 | 0.062654 | 0.274497 | 0.078030 |
| 10 | E007 | TransUNet | RGB | 0.001768 | 0.003530 | 0.001784 | 0.106293 | 0.311982 | 0.070216 |
| 11 | E010 | DeepLabV3+ | RGBN | 0.007690 | 0.015263 | 0.007756 | 0.000302 | 0.280598 | 0.050158 |
| 12 | E008 | Mask2Former | RGB | 0.000268 | 0.000536 | 0.000268 | 0.000117 | 0.204060 | 0.030919 |

## Best Model

The best model is **SegFormer**, experiment **E012**, using **RGBN** input.

Key held-out clean test metrics:

- IoU: 0.205798
- Dice: 0.341348
- Recall: 0.817789
- Relaxed IoU: 0.270101
- Connectivity Ratio: 0.871124

Key synthetic-occluded test metrics:

- Occlusion Recall: 0.917907
- Occluded Recall: 0.854123
- Occluded Connectivity Ratio: 0.924833

Final combined score: **0.573358**

## Why This Model Won

The refined SegFormer model won because it combined long-range context with explicit occlusion-aware training. Roads are elongated structures, and transformer-style patch attention helps connect spatial evidence across a tile. Synthetic occlusion training forced the model to predict roads even when visible image evidence was degraded. Compared with the CNN baselines, the final model achieved much higher recall, occlusion recall, and connectivity ratio.

## Fit to the Original Problem

The problem asks for occlusion-robust road extraction that can support graph-theoretic criticality analysis. The final model fits this requirement because it produces road masks that are more complete under occlusion and more connected spatially. This makes it more suitable for the next phase: skeletonization, graph healing, bottleneck identification, and resilience simulation.

## Limitations and Failure Cases

The experiment used OSM-derived labels, so label noise and road-width assumptions affect the scores. Sentinel-2 has 10 m resolution, which limits fine-grained lane-level road extraction. Some failure cases still show overprediction in dense urban areas and missed narrow roads. These limitations are expected and should be improved with higher-resolution Resourcesat/Cartosat imagery and more AOIs.

## Conclusion

Based on the completed experiments, **E012: SegFormer + RGBN + synthetic occlusion training** is the best Part 1 model. It is the recommended segmentation model for the next graph reconstruction phase because it gives the strongest balance of road recovery, occlusion robustness, and connectivity.
