from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import box
from tqdm import tqdm


ROAD_WIDTH_M = {
    "motorway": 30,
    "trunk": 24,
    "primary": 20,
    "secondary": 16,
    "tertiary": 12,
    "residential": 8,
    "service": 6,
    "unclassified": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 0 Sentinel-2/OSM road tiles.")
    parser.add_argument("--aoi", default="bengaluru_core")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-road-ratio", type=float, default=0.001)
    parser.add_argument("--max-empty-tiles", type=int, default=80)
    parser.add_argument("--variant", default=None)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_bbox(root: Path, aoi: str) -> dict:
    bbox_path = root / "data" / "metadata" / f"{aoi}.bbox.json"
    with bbox_path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def scene_dir(root: Path, aoi: str) -> Path:
    sentinel_root = root / "data" / "raw" / "sentinel2" / aoi
    scenes = [p for p in sentinel_root.iterdir() if p.is_dir()]
    if not scenes:
        raise FileNotFoundError(f"No Sentinel-2 scene directory found in {sentinel_root}")
    return scenes[0]


def read_band_paths(scene: Path) -> dict[str, Path]:
    paths = {
        "blue": scene / "B02_blue.tif",
        "green": scene / "B03_green.tif",
        "red": scene / "B04_red.tif",
        "nir": scene / "B08_nir.tif",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Sentinel-2 bands: {missing}")
    return paths


def compute_aoi_window(red_path: Path, bbox_wgs84: dict) -> tuple[Window, rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(red_path) as src:
        bbox_geo = gpd.GeoSeries(
            [box(bbox_wgs84["west"], bbox_wgs84["south"], bbox_wgs84["east"], bbox_wgs84["north"])],
            crs="EPSG:4326",
        ).to_crs(src.crs)
        minx, miny, maxx, maxy = bbox_geo.total_bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        window = window.round_offsets().round_lengths()
        window = Window(
            max(0, int(window.col_off)),
            max(0, int(window.row_off)),
            min(int(window.width), src.width - int(window.col_off)),
            min(int(window.height), src.height - int(window.row_off)),
        )
        return window, window_transform(window, src.transform), src.crs


def read_stack(band_paths: dict[str, Path], window: Window) -> np.ndarray:
    arrays = []
    for name in ("red", "green", "blue", "nir"):
        with rasterio.open(band_paths[name]) as src:
            arrays.append(src.read(1, window=window))
    return np.stack(arrays, axis=0)


def normalize_to_uint8(tile: np.ndarray) -> np.ndarray:
    out = np.zeros_like(tile, dtype=np.uint8)
    for idx in range(tile.shape[0]):
        band = tile[idx].astype(np.float32)
        valid = band[band > 0]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, (2, 98))
        if hi <= lo:
            hi = max(lo + 1.0, band.max())
        scaled = np.clip((band - lo) / (hi - lo), 0, 1)
        out[idx] = (scaled * 255).astype(np.uint8)
    return out


def extract_highway_class(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # GDAL OSM sometimes stores semicolon-separated values.
    return text.split(";")[0]


def rasterize_osm_roads(root: Path, aoi: str, out_shape: tuple[int, int], transform, crs) -> np.ndarray:
    osm_path = root / "data" / "raw" / "osm" / aoi / f"{aoi}-roads.osm"
    roads = gpd.read_file(osm_path, layer="lines")
    if roads.empty:
        raise ValueError(f"No line features found in {osm_path}")

    roads = roads[roads.geometry.notna()].copy()
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    roads = roads.to_crs(crs)

    if "highway" not in roads.columns:
        raise ValueError("OSM lines layer does not contain a highway column.")

    roads["road_class"] = roads["highway"].map(extract_highway_class)
    roads = roads[roads["road_class"].isin(ROAD_WIDTH_M)].copy()
    if roads.empty:
        raise ValueError("No matching OSM highway classes after filtering.")

    bounds = array_bounds(out_shape[0], out_shape[1], transform)
    aoi_poly = box(bounds[0], bounds[1], bounds[2], bounds[3])
    roads = roads[roads.intersects(aoi_poly)].copy()

    buffered_geoms = []
    for _, row in roads.iterrows():
        width = ROAD_WIDTH_M[row["road_class"]]
        buffered_geoms.append(row.geometry.buffer(width / 2.0))

    mask = rasterize(
        ((geom, 1) for geom in buffered_geoms if geom and not geom.is_empty),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return mask


def split_for_tile(tile_id_num: int) -> str:
    bucket = tile_id_num % 20
    if bucket < 14:
        return "train"
    if bucket < 17:
        return "val"
    return "test"


def save_rgb_png(path: Path, rgb_tile: np.ndarray) -> None:
    image = np.moveaxis(rgb_tile[:3], 0, -1)
    Image.fromarray(image, mode="RGB").save(path)


def save_mask_png(path: Path, mask_tile: np.ndarray) -> None:
    Image.fromarray((mask_tile * 255).astype(np.uint8), mode="L").save(path)


def main() -> None:
    args = parse_args()
    root = project_root()
    scene = scene_dir(root, args.aoi)
    band_paths = read_band_paths(scene)
    bbox_wgs84 = load_bbox(root, args.aoi)

    variant = args.variant or f"t{args.tile_size}_s{args.stride}"
    processed_root = root / "data" / "processed" / args.aoi / variant
    image_dir = processed_root / "images"
    mask_dir = processed_root / "masks"
    preview_dir = processed_root / "previews"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    window, crop_transform, crs = compute_aoi_window(band_paths["red"], bbox_wgs84)
    stack = read_stack(band_paths, window)
    mask = rasterize_osm_roads(root, args.aoi, (stack.shape[1], stack.shape[2]), crop_transform, crs)
    stack_u8 = normalize_to_uint8(stack)

    crop_meta_path = processed_root / "crop_metadata.json"
    crop_meta = {
        "aoi": args.aoi,
        "scene": scene.name,
        "crs": str(crs),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "crop_shape": [int(stack.shape[1]), int(stack.shape[2])],
        "bands": ["red", "green", "blue", "nir"],
        "road_width_m": ROAD_WIDTH_M,
    }
    crop_meta_path.write_text(json.dumps(crop_meta, indent=2), encoding="utf-8")

    manifest_path = processed_root / "tile_manifest.csv"
    rows = []
    tile_num = 0
    empty_kept = 0
    h, w = mask.shape
    max_row = h - args.tile_size
    max_col = w - args.tile_size

    for row in tqdm(range(0, max_row + 1, args.stride), desc="Tiling rows"):
        for col in range(0, max_col + 1, args.stride):
            img_tile = stack_u8[:, row : row + args.tile_size, col : col + args.tile_size]
            mask_tile = mask[row : row + args.tile_size, col : col + args.tile_size]
            road_ratio = float(mask_tile.mean())

            if road_ratio < args.min_road_ratio:
                if empty_kept >= args.max_empty_tiles:
                    continue
                empty_kept += 1

            tile_id = f"{args.aoi}_{tile_num:05d}"
            split = split_for_tile(tile_num)
            image_path = image_dir / f"{tile_id}_rgb.png"
            rgbn_path = image_dir / f"{tile_id}_rgbn.npy"
            mask_path = mask_dir / f"{tile_id}_mask.png"

            save_rgb_png(image_path, img_tile)
            np.save(rgbn_path, img_tile)
            save_mask_png(mask_path, mask_tile)

            rows.append(
                {
                    "tile_id": tile_id,
                    "aoi": args.aoi,
                    "sensor": "Sentinel-2",
                    "scene": scene.name,
                    "split": split,
                    "image_rgb_path": str(image_path.relative_to(root)),
                    "image_rgbn_path": str(rgbn_path.relative_to(root)),
                    "mask_path": str(mask_path.relative_to(root)),
                    "row": row,
                    "col": col,
                    "tile_size": args.tile_size,
                    "resolution_m": 10,
                    "road_pixel_ratio": f"{road_ratio:.6f}",
                }
            )
            tile_num += 1

    if not rows:
        raise ValueError("No tiles were generated. Check AOI, tile size, and road-ratio settings.")

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    preview_rows = rows[: min(12, len(rows))]
    top_road_rows = sorted(rows, key=lambda item: float(item["road_pixel_ratio"]), reverse=True)[: min(12, len(rows))]
    preview_seen = set()
    for row in preview_rows + top_road_rows:
        if row["tile_id"] in preview_seen:
            continue
        preview_seen.add(row["tile_id"])
        rgb = Image.open(root / row["image_rgb_path"]).convert("RGB")
        mask_img = Image.open(root / row["mask_path"]).convert("L")
        overlay = np.array(rgb).copy()
        mask_arr = np.array(mask_img) > 0
        overlay[mask_arr] = (255, 70, 40)
        preview = Image.blend(rgb, Image.fromarray(overlay, mode="RGB"), alpha=0.45)
        preview.save(preview_dir / f"{row['tile_id']}_overlay_road_{row['road_pixel_ratio']}.png")

    print(f"Prepared {len(rows)} tiles")
    print(f"Manifest: {manifest_path}")
    print(f"Previews: {preview_dir}")


if __name__ == "__main__":
    main()
