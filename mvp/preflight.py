from __future__ import annotations

import argparse
import json

from mvp.artifacts import ArtifactRegistry
from mvp.config import load_mvp_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Route Resilience MVP")
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    registry = ArtifactRegistry(load_mvp_config(args.config))
    print(json.dumps(registry.status.to_dict(), indent=2))
    if registry.status.missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
