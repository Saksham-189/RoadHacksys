from __future__ import annotations

import argparse
import json

from mobility.config import load_config
from mobility.consolidation import run_consolidation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part3_flow.yaml")
    args = parser.parse_args()
    print(json.dumps(run_consolidation(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
