from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_mvp_config(path: str | Path = "configs/mvp.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    for key, value in config["paths"].items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        config["paths"][key] = str(candidate)
    return config

