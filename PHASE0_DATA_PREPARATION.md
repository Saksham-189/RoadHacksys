# Phase 0: Dataset Preparation for Occlusion-Robust Road Extraction

## Problem Scope

Part 1 solves satellite-image road segmentation:

```text
Satellite image tile -> binary road mask
```

The target is not only high pixel accuracy. The mask must preserve road continuity so that Part 2 can skeletonize it into a connected graph.

## Datasets

### Sentinel-2

- Resolution: 10 m for visible RGB and NIR bands.
- Product preference: Level-2A surface reflectance.
- Best use: highways, arterial roads, broad urban corridors, mixed urban-rural study regions.
- Limitation: many smaller city roads are below or near pixel scale.

Recommended bands:

```text
B02 Blue
B03 Green
B04 Red
B08 NIR
```

Initial input options:

```text
RGB: B04, B03, B02
RGBN: B04, B03, B02, B08
```

### Resourcesat LISS-IV

- Resolution: 5.8 m.
- Best use: finer road structures than Sentinel-2.
- Limitation: access and preprocessing may be less convenient than Sentinel-2.

Recommended first use:

```text
Use Resourcesat after Sentinel-2 pipeline is stable.
```

## Ground Truth Strategy

Use OpenStreetMap road vectors as the first ground-truth source.

Pipeline:

```text
OSM road vectors
        ↓
filter road classes
        ↓
reproject to satellite CRS
        ↓
rasterize roads into binary masks
        ↓
train segmentation models
```

Initial OSM road classes:

```text
motorway
trunk
primary
secondary
tertiary
residential
service
unclassified
```

For Sentinel-2, we may later exclude very narrow classes if masks become too noisy at 10 m.

## Study Area Selection

Start with three geographically different regions:

| Area Type | Suggested City/Region | Why |
|---|---|---|
| Dense urban | Bengaluru | complex roads, shadows, trees, urban clutter |
| Planned/mixed urban | Hyderabad or Pune | broader roads, mixed land cover |
| Urban edge/suburban | Bengaluru outskirts or Mysuru edge | vegetation occlusion, rural-urban transition |

Use geographic splits, not random tile splits:

```text
Train: 70%
Validation: 15%
Test: 15%
```

This prevents nearby tiles from leaking into validation and making results look better than they are.

## Phase 0 Steps

### Step 0.1: Pick Area of Interest

Define AOIs as bounding boxes or polygons.

Minimum first AOI:

```text
Bengaluru urban core
```

Preferred final Phase 0 AOIs:

```text
Bengaluru dense urban
Hyderabad or Pune mixed urban
suburban/vegetated edge region
```

### Step 0.2: Download Satellite Imagery

For Sentinel-2:

```text
Use Level-2A scenes with low cloud cover.
Prefer recent dry-season imagery for clearer roads.
```

For Resourcesat:

```text
Use LISS-IV scenes over the same or comparable AOIs.
Match acquisition season as closely as possible.
```

### Step 0.3: Download OSM Road Vectors

Acquire OSM roads for the selected AOIs.

Filter road geometry to relevant classes and remove non-road paths unless intentionally included.

### Step 0.4: Align CRS and Resolution

All imagery and vector labels must use the same coordinate system.

Recommended working target:

```text
UTM zone appropriate for the AOI
```

Rasterize labels at sensor resolution:

```text
Sentinel-2 masks: 10 m
Resourcesat masks: 5.8 m
```

### Step 0.5: Rasterize Road Masks

Road vectors need width before rasterization.

Initial width assumptions:

| OSM Class | Approx Width |
|---|---:|
| motorway/trunk | 20-30 m |
| primary/secondary | 12-20 m |
| tertiary/residential | 6-12 m |
| service/unclassified | 4-8 m |

At Sentinel-2 resolution, very thin roads may collapse into weak labels. Use buffered vectors.

### Step 0.6: Tile Images and Masks

Start with:

```text
256 x 256 tiles
```

Also prepare:

```text
512 x 512 tiles
```

Why:

- 256 tiles train faster.
- 512 tiles preserve more road context.
- Transformer models may benefit from larger context.

### Step 0.7: Quality Control

Manually inspect a sample of image-mask pairs.

Check:

- image and mask alignment
- road width reasonableness
- cloud contamination
- missing OSM roads
- false OSM roads
- tile has enough positive road pixels

Remove or flag bad tiles.

### Step 0.8: Build Dataset Manifest

Every tile should have metadata:

```text
tile_id
sensor
aoi_name
image_path
mask_path
split
cloud_score
road_pixel_ratio
date
resolution_m
```

This manifest becomes the source of truth for experiments.

## Acceptance Criteria for Phase 0

Phase 0 is complete when we have:

- selected AOIs
- downloaded or identified satellite scenes
- obtained OSM roads
- generated raster masks
- created aligned image-mask tiles
- created train/val/test splits
- manually inspected sample tiles
- created a dataset manifest

## Current Status

Prepared AOIs:

```text
bengaluru_core
hyderabad_mixed
bengaluru_edge
```

Downloaded raw data:

- OSM road extracts from Overpass API.
- Sentinel-2 L2A scenes for all prepared AOIs.
- Bands: B02, B03, B04, B08, and visual composite for each AOI.

Prepared tile datasets:

```text
data/processed/bengaluru_core/t256_s128
data/processed/hyderabad_mixed/t256_s128
data/processed/bengaluru_edge/t256_s128
data/processed/merged_t256_s128_manifest.csv
```

Tile settings:

```text
tile_size = 256
stride = 128
sensor = Sentinel-2
bands = RGB + NIR
mask source = rasterized OSM roads
```

Current merged tile count:

```text
444 total tiles
124 train tiles      bengaluru_core
16 validation tiles  hyderabad_mixed
304 test tiles       bengaluru_edge
364 road-positive tiles
```

Generated artifacts:

```text
tile_manifest.csv
images/*.png       RGB visual tiles
images/*.npy       RGBN arrays
masks/*.png        binary road masks
previews/*.png     QA overlays
crop_metadata.json
```

Resourcesat LISS-IV is not downloaded yet because access usually requires NRSC/Bhuvan credentials or hackathon-provided data. The pipeline is prepared so Resourcesat can be added later using the same image-mask tiling structure.

## Key Phase 0 Risks

### Sentinel-2 Resolution Risk

At 10 m, many urban roads are too narrow. This may reduce IoU and make labels noisy.

Mitigation:

```text
focus evaluation on major and medium roads first
use relaxed IoU / buffer IoU
use Resourcesat for finer road experiments
```

### OSM Label Noise

OSM may be incomplete, misaligned, or inconsistent.

Mitigation:

```text
geographic validation
manual QC sample
buffered labels
relaxed metrics
```

### Cloud and Shadow Contamination

Clouds and shadows can confuse both training and evaluation.

Mitigation:

```text
start with low-cloud scenes
later intentionally add synthetic occlusions
track cloud score in manifest
```

## Sources

- Sentinel-2 documentation: https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html
- Copernicus Data Space: https://dataspace.copernicus.eu/
- ISRO Bhuvan: https://bhuvan.nrsc.gov.in/home/index.php
- Geofabrik India OSM extracts: https://download.geofabrik.de/asia/india.html
