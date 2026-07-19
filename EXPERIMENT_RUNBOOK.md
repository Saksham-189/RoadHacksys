# Occlusion-Robust Road Segmentation Runbook

## 1. Environment

Use the Windows virtual environment:

```powershell
.\.venv-win\Scripts\python.exe
```

Install dependencies if needed:

```powershell
.\.venv-win\Scripts\python.exe -m pip install -r requirements-experiments.txt --only-binary=:all:
```

## 2. Add the Remaining Sentinel-2 AOIs

The downloader has presets for:

```text
bengaluru_core
hyderabad_mixed
bengaluru_edge
```

Download and prepare the two remaining AOIs:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_phase0_data.ps1 -AoiName hyderabad_mixed
.\.venv-win\Scripts\python.exe scripts\prepare_phase0_tiles.py --aoi hyderabad_mixed --tile-size 256 --stride 128 --max-empty-tiles 80 --variant t256_s128

powershell -ExecutionPolicy Bypass -File .\scripts\download_phase0_data.ps1 -AoiName bengaluru_edge
.\.venv-win\Scripts\python.exe scripts\prepare_phase0_tiles.py --aoi bengaluru_edge --tile-size 256 --stride 128 --max-empty-tiles 80 --variant t256_s128
```

Build the merged manifest:

```powershell
.\.venv-win\Scripts\python.exe scripts\build_merged_manifest.py --variant t256_s128 --out data\processed\merged_t256_s128_manifest.csv --train-aois bengaluru_core --val-aois hyderabad_mixed --test-aois bengaluru_edge
```

After merging, update `configs/experiments/base.yaml`:

```yaml
data:
  manifest_path: data/processed/merged_t256_s128_manifest.csv
  train_aois: bengaluru_core
  test_aois: bengaluru_edge
```

## 3. Validate Before Training

```powershell
.\.venv-win\Scripts\python.exe scripts\validate_dataset.py --manifest data\processed\merged_t256_s128_manifest.csv
.\.venv-win\Scripts\python.exe scripts\run_smoke_tests.py
```

Run 2-epoch CLI smoke experiments:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_experiment_matrix.ps1 -SmokeOnly
```

## 4. Train the Full Experiment Matrix

Confirm GPU availability before launching the full matrix:

```powershell
.\.venv-win\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The local desktop environment used during setup reported CPU-only PyTorch, so full training should be run on a CUDA-enabled machine, Colab, or Kaggle for practical runtimes.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_experiment_matrix.ps1
```

This trains:

```text
E001 U-Net
E002 ResNet U-Net
E003 UNet++
E004 DeepLabV3+
E005 SegFormer
E006 Swin-Unet
E007 TransUNet
E008 Mask2Former
E009 DINO/ViT Head
E010 DeepLabV3+ + Synthetic Occlusion
E011 SegFormer + Synthetic Occlusion
```

Each checkpoint is evaluated on clean and synthetic-occluded test inputs.

## 5. Outputs

Primary scoreboard:

```text
runs/model_scoreboard.csv
```

Each experiment run contains:

```text
best.pt
history.csv
best_val_metrics.json
metrics_test_clean.json
metrics_test_occluded.json
overlays_test_clean/
overlays_test_occluded/
resolved_config.json
```

The final winner is the row with rank `1` in `runs/model_scoreboard.csv`.
