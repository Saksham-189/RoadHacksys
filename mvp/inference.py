from __future__ import annotations

import csv
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mvp.artifacts import file_signature


class InferenceService:
    def __init__(
        self,
        config: dict[str, Any],
        warm: bool = True,
    ) -> None:
        self.config = config
        self.paths = {key: Path(value) for key, value in config["paths"].items()}
        self.rows = self._read_tiles(self.paths["manifest"])
        self.model: Any | None = None
        self.device: Any | None = None
        self.torch: Any | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.error: str | None = None
        self.lock = threading.Lock()
        self.checkpoint_fingerprint = file_signature(
            self.paths["checkpoint"]
        )["sha256"]
        if warm:
            self.warm()

    @staticmethod
    def _read_tiles(path: Path) -> dict[str, dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return {
                row["tile_id"]: row
                for row in csv.DictReader(handle)
                if row["split"] == "test"
            }

    def warm(self) -> None:
        try:
            import torch

            from roadseg.model import build_model

            self.torch = torch
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            checkpoint = torch.load(
                self.paths["checkpoint"],
                map_location=self.device,
                weights_only=False,
            )
            model = build_model(checkpoint["config"]["model"]).to(self.device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self.checkpoint = checkpoint
            self.model = model
            self.error = None
        except Exception as error:
            self.error = str(error)
            self.model = None

    def status(self) -> dict[str, Any]:
        torch = self.torch
        gpu_name = (
            torch.cuda.get_device_name(0)
            if torch is not None and torch.cuda.is_available()
            else None
        )
        gpu_memory = (
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024**3),
                2,
            )
            if torch is not None and torch.cuda.is_available()
            else None
        )
        return {
            "available": self.model is not None,
            "device": str(self.device) if self.device is not None else "warming",
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory,
            "error": self.error,
            "tiles": len(self.rows),
        }

    def tiles(self) -> list[dict[str, Any]]:
        default = self.config["inference"]["default_tile"]
        rows = sorted(
            (
                {
                    "tile_id": row["tile_id"],
                    "aoi": row["aoi"],
                    "road_pixel_ratio": float(row["road_pixel_ratio"]),
                    "default": row["tile_id"] == default,
                }
                for row in self.rows.values()
            ),
            key=lambda row: (not row["default"], row["tile_id"]),
        )
        return rows

    def infer(
        self, tile_id: str, occlusion: str, seed: int
    ) -> dict[str, Any]:
        if tile_id not in self.rows:
            raise ValueError(f"Unknown test tile: {tile_id}")
        cache_key = hashlib.sha256(
            f"{self.checkpoint_fingerprint}:{tile_id}:{occlusion}:{seed}".encode()
        ).hexdigest()[:20]
        output = self.paths["inference"] / cache_key
        response_path = output / "response.json"
        if response_path.exists():
            return json.loads(response_path.read_text(encoding="utf-8"))
        if self.model is None:
            return self._fallback(tile_id, occlusion)
        try:
            with self.lock:
                return self._run(tile_id, occlusion, seed, output)
        except RuntimeError as error:
            if self.device is not None and self.device.type == "cuda":
                try:
                    assert self.torch is not None
                    self.device = self.torch.device("cpu")
                    self.model = self.model.to(self.device)
                    return self._run(tile_id, occlusion, seed, output)
                except Exception:
                    self.error = str(error)
            return self._fallback(tile_id, occlusion)

    def _run(
        self,
        tile_id: str,
        occlusion: str,
        seed: int,
        output: Path,
    ) -> dict[str, Any]:
        assert self.model is not None and self.checkpoint is not None
        assert self.torch is not None and self.device is not None
        torch = self.torch
        from roadseg.metrics import binary_counts, metrics_from_counts

        row = self.rows[tile_id]
        image = np.load(row["image_rgbn_path"]).astype(np.float32) / 255.0
        target = (
            np.asarray(Image.open(row["mask_path"]), dtype=np.float32) / 255.0
        )
        displayed = image.copy()
        if occlusion != "none":
            displayed = self._apply_requested_occlusion(
                displayed, occlusion, seed
            )
        data = self.checkpoint["config"]["data"]
        mean = np.asarray(data["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(data["std"], dtype=np.float32)[:, None, None]
        normalized = (displayed - mean) / std
        tensor = torch.from_numpy(
            np.ascontiguousarray(normalized[None])
        ).float().to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=use_amp
        ):
            probability = self.model(tensor).sigmoid()[0, 0].float().cpu().numpy()
        threshold = float(self.checkpoint.get("threshold", 0.1))
        prediction = probability >= threshold
        target_bool = target >= 0.5
        metrics = metrics_from_counts(
            *binary_counts(
                torch.from_numpy(prediction),
                torch.from_numpy(target_bool),
            )
        )
        output.mkdir(parents=True, exist_ok=True)
        rgb = np.clip(displayed[:3].transpose(1, 2, 0), 0, 1)
        overlay = rgb.copy()
        overlay[prediction] = (
            0.55 * overlay[prediction]
            + 0.45 * np.asarray([1.0, 0.15, 0.05])
        )
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            output / "input.png"
        )
        Image.fromarray((target_bool * 255).astype(np.uint8)).save(
            output / "target.png"
        )
        Image.fromarray((probability * 255).astype(np.uint8)).save(
            output / "probability.png"
        )
        Image.fromarray((overlay * 255).astype(np.uint8)).save(
            output / "overlay.png"
        )
        response = {
            "tile_id": tile_id,
            "occlusion": occlusion,
            "seed": seed,
            "mode": "live",
            "device": str(self.device),
            "threshold": threshold,
            "metrics": metrics,
            "panels": {
                name: f"/generated/inference/{output.name}/{name}.png"
                for name in ("input", "target", "probability", "overlay")
            },
        }
        (output / "response.json").write_text(
            json.dumps(response, indent=2), encoding="utf-8"
        )
        return response

    @staticmethod
    def _apply_requested_occlusion(
        image: np.ndarray, kind: str, seed: int
    ) -> np.ndarray:
        from roadseg.data import apply_synthetic_occlusion

        for offset in range(100):
            rng = np.random.default_rng(seed + offset)
            candidate, _, generated = apply_synthetic_occlusion(image, rng)
            if generated == kind:
                return candidate
        raise RuntimeError(f"Could not generate deterministic {kind} occlusion")

    def _fallback(self, tile_id: str, occlusion: str) -> dict[str, Any]:
        source = Path(
            self.config["inference"][
                "fallback_clean"
                if occlusion == "none"
                else "fallback_occluded"
            ]
        )
        if not source.is_absolute():
            source = Path(__file__).resolve().parents[1] / source
        return {
            "tile_id": tile_id,
            "occlusion": occlusion,
            "mode": "cached_fallback",
            "device": None,
            "threshold": 0.1,
            "metrics": None,
            "panels": {
                "diagnostic": f"/fallback/{'clean' if occlusion == 'none' else 'occluded'}"
            },
            "error": self.error,
        }
