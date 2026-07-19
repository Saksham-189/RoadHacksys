from __future__ import annotations

import argparse
import csv
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

AOIS = {
    "pune": (18.5204, 73.8567, "train"),
    "chennai": (13.0827, 80.2707, "train"),
    "ahmedabad": (23.0225, 72.5714, "train"),
    "jaipur": (26.9124, 75.7873, "train"),
    "lucknow": (26.8467, 80.9462, "train"),
    "bhopal": (23.2599, 77.4126, "train"),
    "surat": (21.1702, 72.8311, "train"),
    "nagpur": (21.1458, 79.0882, "train"),
    "kochi": (9.9312, 76.2673, "train"),
    "bhubaneswar": (20.2961, 85.8245, "train"),
    "coimbatore": (11.0168, 76.9558, "train"),
    "visakhapatnam": (17.6868, 83.2185, "train"),
    "chandigarh": (30.7333, 76.7794, "val"),
    "indore": (22.7196, 75.8577, "val"),
    "mysuru": (12.2958, 76.6394, "val"),
}

ROAD_WIDTHS = {
    "motorway": 4,
    "motorway_link": 3,
    "trunk": 4,
    "trunk_link": 3,
    "primary": 3,
    "primary_link": 3,
    "secondary": 3,
    "secondary_link": 2,
    "tertiary": 2,
    "tertiary_link": 2,
    "residential": 2,
    "living_street": 2,
    "unclassified": 2,
    "service": 1,
    "road": 2,
}


def bbox_around(latitude: float, longitude: float, half_height: float) -> list[float]:
    half_width = half_height / max(math.cos(math.radians(latitude)), 0.3)
    return [
        longitude - half_width,
        latitude - half_height,
        longitude + half_width,
        latitude + half_height,
    ]


def request_with_retries(
    method: str, url: str, *, attempts: int = 4, **kwargs
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, timeout=240, **kwargs)
            response.raise_for_status()
            return response
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def find_scene(bbox: list[float]) -> dict:
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": "2024-01-01T00:00:00Z/2025-12-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 8}},
        "limit": 40,
    }
    response = request_with_retries("POST", STAC_SEARCH, json=payload).json()
    features = response.get("features", [])
    if not features:
        raise RuntimeError(f"No low-cloud Sentinel-2 scene found for {bbox}")
    return min(features, key=lambda item: item["properties"].get("eo:cloud_cover", 100))


def read_rgbn_crop(
    item: dict, bbox: list[float]
) -> tuple[np.ndarray, rasterio.Affine, str]:
    asset_names = ("red", "green", "blue", "nir")
    urls = [item["assets"][name]["href"] for name in asset_names]
    environment = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    }
    with rasterio.Env(**environment), rasterio.open(urls[0]) as reference:
        transformer = Transformer.from_crs("EPSG:4326", reference.crs, always_xy=True)
        west, south = transformer.transform(bbox[0], bbox[1])
        east, north = transformer.transform(bbox[2], bbox[3])
        bounds = (min(west, east), min(south, north), max(west, east), max(south, north))
        window = from_bounds(*bounds, transform=reference.transform).round_offsets().round_lengths()
        height, width = int(window.height), int(window.width)
        target_transform = reference.window_transform(window)
        target_crs = reference.crs.to_string()

    channels = []
    with rasterio.Env(**environment):
        for url in urls:
            with rasterio.open(url) as source:
                source_window = from_bounds(
                    *bounds, transform=source.transform
                ).round_offsets().round_lengths()
                channel = source.read(
                    1,
                    window=source_window,
                    out_shape=(height, width),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
                channels.append(channel)
    image = np.stack(channels)
    valid = image > 0
    for channel in range(4):
        values = image[channel][valid[channel]]
        low, high = np.percentile(values, (2, 98)) if values.size else (0, 1)
        image[channel] = np.clip((image[channel] - low) / max(high - low, 1), 0, 1)
    return (image * 255).round().astype(np.uint8), target_transform, target_crs


def download_osm(bbox: list[float], destination: Path) -> None:
    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
    query = (
        f'[out:xml][timeout:180];way["highway"]'
        f"({south},{west},{north},{east});(._;>;);out body;"
    )
    headers = {"User-Agent": "ISRO-Hackathon-RoadSeg/2.0"}
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = request_with_retries(
                "POST", endpoint, data={"data": query}, headers=headers, attempts=2
            )
            destination.write_bytes(response.content)
            return
        except Exception as error:
            last_error = error
    raise RuntimeError("All Overpass endpoints failed") from last_error


def rasterize_roads(
    osm_path: Path,
    shape: tuple[int, int],
    transform: rasterio.Affine,
    crs: str,
) -> np.ndarray:
    root = ET.parse(osm_path).getroot()
    nodes = {
        node.attrib["id"]: (float(node.attrib["lon"]), float(node.attrib["lat"]))
        for node in root.findall("node")
    }
    projection = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inverse_transform = ~transform
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for way in root.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        width = ROAD_WIDTHS.get(tags.get("highway", ""))
        if width is None:
            continue
        coordinates = []
        for reference in way.findall("nd"):
            lon_lat = nodes.get(reference.attrib["ref"])
            if lon_lat is None:
                continue
            x, y = projection.transform(*lon_lat)
            col, row = inverse_transform * (x, y)
            coordinates.append((round(col), round(row)))
        if len(coordinates) >= 2:
            draw.line(coordinates, fill=255, width=width, joint="curve")
    return np.asarray(canvas, dtype=np.uint8)


def write_tiles(
    name: str,
    split: str,
    scene_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    output_root: Path,
    tile_size: int,
    stride: int,
) -> list[dict[str, object]]:
    aoi_root = output_root / name
    image_dir = aoi_root / "images"
    mask_dir = aoi_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for top in range(0, image.shape[1] - tile_size + 1, stride):
        for left in range(0, image.shape[2] - tile_size + 1, stride):
            tile = image[:, top : top + tile_size, left : left + tile_size]
            label = mask[top : top + tile_size, left : left + tile_size]
            if np.count_nonzero(tile) < tile.size * 0.75:
                continue
            tile_id = f"{name}_{index:05d}"
            image_path = image_dir / f"{tile_id}_rgbn.npy"
            rgb_path = image_dir / f"{tile_id}_rgb.png"
            mask_path = mask_dir / f"{tile_id}_mask.png"
            np.save(image_path, tile)
            Image.fromarray(tile[:3].transpose(1, 2, 0)).save(rgb_path)
            Image.fromarray(label).save(mask_path)
            rows.append(
                {
                    "tile_id": tile_id,
                    "aoi": name,
                    "sensor": "Sentinel-2",
                    "scene": scene_id,
                    "split": split,
                    "image_rgb_path": str(rgb_path),
                    "image_rgbn_path": str(image_path),
                    "mask_path": str(mask_path),
                    "row": top,
                    "col": left,
                    "tile_size": tile_size,
                    "resolution_m": 10,
                    "road_pixel_ratio": float((label > 0).mean()),
                    "split_strategy": "geographic_aoi_v2",
                }
            )
            index += 1
    return rows


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--half-height", type=float, default=0.08)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--only", nargs="*", choices=AOIS)
    args = parser.parse_args()
    selected = args.only or list(AOIS)
    root = Path("data/expanded_v2")
    raw_root = root / "raw"
    tile_root = root / "tiles"
    all_rows: list[dict[str, object]] = []

    for position, name in enumerate(selected, start=1):
        latitude, longitude, split = AOIS[name]
        bbox = bbox_around(latitude, longitude, args.half_height)
        raw_aoi = raw_root / name
        raw_aoi.mkdir(parents=True, exist_ok=True)
        print(f"[{position}/{len(selected)}] {name}: selecting scene", flush=True)
        item = find_scene(bbox)
        (raw_aoi / "stac_item.json").write_text(json.dumps(item, indent=2))
        (raw_aoi / "bbox.json").write_text(json.dumps(bbox))
        print(
            f"  {item['id']} cloud={item['properties'].get('eo:cloud_cover')}",
            flush=True,
        )
        image, transform, crs = read_rgbn_crop(item, bbox)
        osm_path = raw_aoi / "roads.osm"
        download_osm(bbox, osm_path)
        mask = rasterize_roads(osm_path, image.shape[1:], transform, crs)
        rows = write_tiles(
            name,
            split,
            item["id"],
            image,
            mask,
            tile_root,
            args.tile_size,
            args.stride,
        )
        write_manifest(rows, tile_root / name / "tile_manifest.csv")
        all_rows.extend(rows)
        print(
            f"  crop={image.shape[2]}x{image.shape[1]} tiles={len(rows)} "
            f"road_mean={np.mean([row['road_pixel_ratio'] for row in rows]):.4f}",
            flush=True,
        )
        time.sleep(2)

    write_manifest(all_rows, root / "expanded_manifest.csv")
    split_counts = {
        split: sum(row["split"] == split for row in all_rows)
        for split in ("train", "val")
    }
    print(f"Complete: {len(all_rows)} tiles {split_counts}", flush=True)


if __name__ == "__main__":
    main()
