param(
    [string]$AoiName = "bengaluru_core",
    [string]$StartDate = "2025-01-01",
    [string]$EndDate = "2025-03-31",
    [double]$MaxCloud = 10,
    [string]$Bbox = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$DataRoot = Join-Path $Root "data"
$OsmDir = Join-Path $DataRoot "raw\osm\$AoiName"
$S2Dir = Join-Path $DataRoot "raw\sentinel2\$AoiName"
$MetaDir = Join-Path $DataRoot "metadata"

New-Item -ItemType Directory -Force -Path $OsmDir, $S2Dir, $MetaDir | Out-Null

$AoiPresets = @{
    bengaluru_core = @{
        west = 77.48
        south = 12.84
        east = 77.78
        north = 13.08
    }
    hyderabad_mixed = @{
        west = 78.43
        south = 17.37
        east = 78.50
        north = 17.43
    }
    bengaluru_edge = @{
        west = 77.32
        south = 13.02
        east = 77.56
        north = 13.22
    }
}

if ($Bbox) {
    $Parts = $Bbox.Split(",") | ForEach-Object { [double]$_.Trim() }
    if ($Parts.Count -ne 4) {
        throw "Custom -Bbox must be 'west,south,east,north'."
    }
    $BboxValues = @{
        west = $Parts[0]
        south = $Parts[1]
        east = $Parts[2]
        north = $Parts[3]
    }
}
elseif ($AoiPresets.ContainsKey($AoiName)) {
    $BboxValues = $AoiPresets[$AoiName]
}
else {
    throw "Unknown AOI '$AoiName'. Use one of: $($AoiPresets.Keys -join ', '), or pass -Bbox 'west,south,east,north'."
}

$BboxJson = $BboxValues | ConvertTo-Json
Set-Content -LiteralPath (Join-Path $MetaDir "$AoiName.bbox.json") -Value $BboxJson -Encoding UTF8

Write-Host "Downloading OSM roads for $AoiName..."
$OverpassQuery = @"
[out:xml][timeout:180];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|service|unclassified)$"]($($BboxValues.south),$($BboxValues.west),$($BboxValues.north),$($BboxValues.east));
);
(._;>;);
out body;
"@

$OverpassBody = "data=$([System.Uri]::EscapeDataString($OverpassQuery))"

$OsmPath = Join-Path $OsmDir "$AoiName-roads.osm"
$OverpassEndpoints = @(
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
)

if ((Test-Path $OsmPath) -and ((Get-Item $OsmPath).Length -gt 0)) {
    Write-Host "OSM extract already exists, skipping: $OsmPath"
}
else {
    $OsmDownloaded = $false
    foreach ($Endpoint in $OverpassEndpoints) {
        try {
            Write-Host "Trying Overpass endpoint: $Endpoint"
            Invoke-WebRequest `
                -Uri $Endpoint `
                -Method Post `
                -Body $OverpassBody `
                -ContentType "application/x-www-form-urlencoded" `
                -UserAgent "IsroHackathonPhase0/1.0" `
                -OutFile $OsmPath `
                -TimeoutSec 240
            $OsmDownloaded = $true
            break
        }
        catch {
            Write-Host "Overpass endpoint failed: $Endpoint"
            Write-Host $_.Exception.Message
        }
    }

    if (-not $OsmDownloaded) {
        throw "All Overpass endpoints failed for $AoiName."
    }
}

Write-Host "Saved OSM extract: $OsmPath"

Write-Host "Searching Sentinel-2 L2A scenes from Earth Search..."
$StacSearch = @{
    collections = @("sentinel-2-l2a")
    bbox = @($BboxValues.west, $BboxValues.south, $BboxValues.east, $BboxValues.north)
    datetime = "$($StartDate)T00:00:00Z/$($EndDate)T23:59:59Z"
    limit = 10
    query = @{
        "eo:cloud_cover" = @{
            lt = $MaxCloud
        }
    }
    sortby = @(
        @{
            field = "properties.eo:cloud_cover"
            direction = "asc"
        }
    )
} | ConvertTo-Json -Depth 10

$StacResponsePath = Join-Path $S2Dir "stac_search_response.json"
$StacResponse = Invoke-RestMethod `
    -Uri "https://earth-search.aws.element84.com/v1/search" `
    -Method Post `
    -Body $StacSearch `
    -ContentType "application/json" `
    -TimeoutSec 120

$StacResponse | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $StacResponsePath -Encoding UTF8

if (-not $StacResponse.features -or $StacResponse.features.Count -eq 0) {
    throw "No Sentinel-2 scenes found for $AoiName with cloud cover below $MaxCloud between $StartDate and $EndDate."
}

$Scene = $StacResponse.features[0]
$SceneId = $Scene.id
$SceneCloud = $Scene.properties.'eo:cloud_cover'
$SceneDate = $Scene.properties.datetime

Write-Host "Selected Sentinel-2 scene: $SceneId"
Write-Host "Scene date: $SceneDate"
Write-Host "Cloud cover: $SceneCloud"

$SceneDir = Join-Path $S2Dir $SceneId
New-Item -ItemType Directory -Force -Path $SceneDir | Out-Null

$Scene | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $SceneDir "stac_item.json") -Encoding UTF8

$Assets = @(
    @{ name = "blue"; file = "B02_blue.tif" },
    @{ name = "green"; file = "B03_green.tif" },
    @{ name = "red"; file = "B04_red.tif" },
    @{ name = "nir"; file = "B08_nir.tif" },
    @{ name = "visual"; file = "visual.tif" }
)

foreach ($Asset in $Assets) {
    $AssetName = $Asset.name
    $Href = $Scene.assets.$AssetName.href
    if (-not $Href) {
        Write-Host "Skipping missing asset: $AssetName"
        continue
    }

    $OutPath = Join-Path $SceneDir $Asset.file
    $RemoteLength = $null
    try {
        $Head = Invoke-WebRequest -Uri $Href -Method Head -TimeoutSec 120
        if ($Head.Headers["Content-Length"]) {
            $RemoteLength = [int64]($Head.Headers["Content-Length"] | Select-Object -First 1)
        }
    }
    catch {
        Write-Host "Could not read remote size for $AssetName; continuing with download."
    }

    if ($RemoteLength -and (Test-Path $OutPath)) {
        $LocalLength = (Get-Item $OutPath).Length
        if ($LocalLength -eq $RemoteLength) {
            Write-Host "Sentinel-2 asset already complete, skipping $AssetName`: $OutPath"
            continue
        }
        elseif ($LocalLength -gt $RemoteLength) {
            Write-Host "Local file is larger than remote asset; restarting $AssetName."
            Remove-Item -LiteralPath $OutPath -Force
        }
    }
    elseif ($AssetName -eq "visual" -and (Test-Path $OutPath) -and ((Get-Item $OutPath).Length -gt 100MB)) {
        Write-Host "Sentinel-2 visual asset exists and is large enough, skipping $AssetName`: $OutPath"
        continue
    }
    elseif ((Test-Path $OutPath) -and ((Get-Item $OutPath).Length -gt 200MB)) {
        Write-Host "Sentinel-2 asset exists and is large enough for the full 10 m COG, skipping $AssetName`: $OutPath"
        continue
    }
    elseif ((Test-Path $OutPath) -and ((Get-Item $OutPath).Length -gt 0)) {
        Write-Host "Removing partial Sentinel-2 asset before clean restart: $OutPath"
        Remove-Item -LiteralPath $OutPath -Force
    }

    Write-Host "Downloading Sentinel-2 asset $AssetName..."
    & curl.exe --fail --location --continue-at - --retry 5 --retry-delay 5 --output $OutPath $Href
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed while downloading $AssetName from $Href"
    }

    if ($RemoteLength) {
        $FinalLength = (Get-Item $OutPath).Length
        if ($FinalLength -ne $RemoteLength) {
            throw "Downloaded size mismatch for $AssetName. Expected $RemoteLength bytes, got $FinalLength bytes."
        }
    }
    Write-Host "Saved: $OutPath"
}

$ManifestPath = Join-Path $MetaDir "phase0_download_manifest.csv"
if (-not (Test-Path $ManifestPath)) {
    "dataset,aoi,path,source,date_or_version,notes" | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
}

"OSM,$AoiName,$OsmPath,Overpass API,$((Get-Date).ToString('yyyy-MM-dd')),AOI road extract" | Add-Content -LiteralPath $ManifestPath -Encoding UTF8
"Sentinel-2,$AoiName,$SceneDir,Earth Search STAC,$SceneDate,Scene $SceneId cloud_cover=$SceneCloud" | Add-Content -LiteralPath $ManifestPath -Encoding UTF8

Write-Host "Phase 0 download complete."
Write-Host "Manifest: $ManifestPath"
