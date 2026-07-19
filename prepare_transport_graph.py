from __future__ import annotations

import argparse
import json

from mobility.config import load_config
from mobility.transport import prepare_transport_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part3_flow.yaml")
    args = parser.parse_args()
    _, report = prepare_transport_graph(load_config(args.config))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
