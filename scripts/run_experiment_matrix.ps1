param(
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"

if ($SmokeOnly) {
    & $Python train.py --config configs\smoke\E001_unet_smoke.yaml
    & $Python evaluate.py --checkpoint runs\SMOKE_UNET_unet\best.pt --split test
    & $Python train.py --config configs\smoke\E005_segformer_smoke.yaml
    & $Python evaluate.py --checkpoint runs\SMOKE_SEGFORMER_segformer\best.pt --split test
    exit 0
}

$configs = @(
    "configs\experiments\E001_unet.yaml",
    "configs\experiments\E002_resnet_unet.yaml",
    "configs\experiments\E003_unetpp.yaml",
    "configs\experiments\E004_deeplabv3plus.yaml",
    "configs\experiments\E005_segformer.yaml",
    "configs\experiments\E006_swin_unet.yaml",
    "configs\experiments\E007_transunet.yaml",
    "configs\experiments\E008_mask2former.yaml",
    "configs\experiments\E009_dinov2_vit_head.yaml",
    "configs\experiments\E010_deeplabv3plus_occlusion.yaml",
    "configs\experiments\E011_segformer_occlusion.yaml"
)

foreach ($config in $configs) {
    & $Python train.py --config $config
}

Get-ChildItem -Path runs -Filter best.pt -Recurse | ForEach-Object {
    & $Python evaluate.py --checkpoint $_.FullName --split test
    & $Python evaluate.py --checkpoint $_.FullName --split test --occluded
}
